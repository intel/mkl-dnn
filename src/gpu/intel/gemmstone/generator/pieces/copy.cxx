/*******************************************************************************
* Copyright 2019 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/


#include "gemmstone/generator.hpp"
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "quantization.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


// Register-to-register copy of a single block, ignoring register offsets in the block.
template <HW hw>
void Generator<hw>::copyRegisterBlock(Type Ts, Type Td, const RegisterBlock &blockSrc, const RegisterBlock &blockDst,
                                      const GRFMultirange &src, const GRFMultirange &dst, int dOffR, int dOffC,
                                      const CommonStrategy &strategy, CommonState &state, bool preserveSrc)
{
    RegisterLayout modSrc{Ts, {blockSrc}}, modDst{Td, {blockDst}};
    modSrc[0].offsetBytes %= GRF::bytes(hw);
    modDst[0].offsetBytes %= GRF::bytes(hw);
    copyRegisters(Ts, Td, modSrc, modDst, src, dst, dOffR, dOffC, false, strategy, state, preserveSrc);
}

// Register-to-register copy, with no type-punning or scaling.
template <HW hw>
void Generator<hw>::copyRegisters(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                  const GRFMultirange &src, const GRFMultirange &dst,
                                  const CommonStrategy &strategy, CommonState &state, bool preserveSrc, bool s4Shift)
{
    copyRegisters(layoutSrc.type(), layoutDst.type(), layoutSrc, layoutDst, src, dst, 0, 0, false, strategy, state, preserveSrc, s4Shift);
}

// Register-to-register copy, with no type-punning or scaling.
template <HW hw>
void Generator<hw>::copyRegisters(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                  const GRFMultirange &src, const GRFMultirange &dst,
                                  int dOffR, int dOffC, bool conjugate,
                                  const CommonStrategy &strategy, CommonState &state, bool preserveSrc, bool s4Shift)
{
    copyRegisters(layoutSrc.type(), layoutDst.type(), layoutSrc, layoutDst, src, dst, dOffR, dOffC, Scalar{1},
                  SubregisterPair(), SubregisterPair(), conjugate, strategy, state, preserveSrc, s4Shift);
}

// Register-to-register copy, with no scaling.
template <HW hw>
void Generator<hw>::copyRegisters(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                  const GRFMultirange &src, const GRFMultirange &dst,
                                  const CommonStrategy &strategy, CommonState &state, bool preserveSrc, bool s4Shift)
{
    copyRegisters(Ts, Td, layoutSrc, layoutDst, src, dst, 0, 0, false, strategy, state, preserveSrc, s4Shift);
}

// Register-to-register copy, with no scaling.
template <HW hw>
void Generator<hw>::copyRegisters(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                  const GRFMultirange &src, const GRFMultirange &dst,
                                  int dOffR, int dOffC, bool conjugate,
                                  const CommonStrategy &strategy, CommonState &state, bool preserveSrc, bool s4Shift)
{
    copyRegisters(Ts, Td, layoutSrc, layoutDst, src, dst, dOffR, dOffC, Scalar{1},
                  SubregisterPair(), SubregisterPair(), conjugate, strategy, state, preserveSrc, s4Shift);
}

// Register-to-register copy, with scaling.
template <HW hw>
void Generator<hw>::copyRegisters(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                  const GRFMultirange &src, const GRFMultirange &dst,
                                  int dOffR, int dOffC, const Scalar &alpha, const SubregisterPair &alpha_real, const SubregisterPair &alpha_imag,
                                  bool conjugate, const CommonStrategy &strategy, CommonState &state, bool preserveSrc, bool s4Shift)
{
    auto ned = elementsPerGRF(hw, Td.real());

    // Special s4 upconversion path for pre-shifted data.
    bool preshiftedS4 = (Ts == Type::s4 && !s4Shift);
    if (alpha == 1 && !conjugate && !preserveSrc && preshiftedS4) {
        RegisterLayout emptyLayout;
        GRFMultirange emptyRegs;
        if (Ts != layoutSrc.type() || Td != layoutDst.type()) stub("No type punning allowed on this path");
        if (canDequantizeSubByteInt(layoutSrc, layoutDst, emptyLayout, emptyLayout)) {
            dequantizeSubByteInt(true, layoutSrc, layoutDst, emptyLayout, emptyLayout,
                                 src, dst, emptyRegs, emptyRegs, dOffR, dOffC, 0, 1, 1, nullptr, strategy, state, s4Shift);
            return;
        }
    }
    if (preshiftedS4) stub("Pre-shifted s4 data not supported on this path");

    // Check layouts.
    bool sCM = layoutSrc.colMajor();
    bool dCM = layoutDst.colMajor();
    if (sCM != dCM) {
        /* Don't allow transposition in registers, except for vectors. */
        if (layoutSrc.rows() > 1 && layoutSrc.cols() > 1)
            stub("Strategy requires transposing a matrix in registers");
    }

    auto RegisterBlock::*nx = sCM ? &RegisterBlock::nr : &RegisterBlock::nc;
    auto RegisterBlock::*ny = sCM ? &RegisterBlock::nc : &RegisterBlock::nr;

    // Accumulate copy pseudo-instructions.
    CopyPlan plan(getProductFamily(), strategy.systolicAvailable);

    for (auto &sblock : layoutSrc) {
    if (Ts.is3() && (!sblock.colMajor || sblock.byteGlue
                || sblock.crosspack <= 1 || sblock.ebytes <= 0))
        stub("Flat u3 copy is unsupported.");
    bool u3ColMajorGroups = Ts.is3();
    if (u3ColMajorGroups
            && ((dCM ? layoutDst[0].nc : layoutDst[0].nr) % 8))
        stub();
    for (int eoffY = 0; eoffY < sblock.*ny; eoffY += (u3ColMajorGroups ? 8 : 1)) {
    for (int eoffX = 0; eoffX < sblock.*nx;) {
        auto eoffR = sblock.colMajor ? eoffX : eoffY;
        auto eoffC = sblock.colMajor ? eoffY : eoffX;

        int ns, nd;
        const RegisterBlock *dblock;

        int cxComp = 0;

        // Locate source and destination registers.
        CopyOperand sOp = sblock.findRegion(Ts,                 eoffR,                          eoffC,         src, &ns,          cxComp, 0,     true);
        CopyOperand dOp = layoutDst.findRegion(sblock.offsetR + eoffR + dOffR, sblock.offsetC + eoffC + dOffC, dst, &nd, &dblock, cxComp, 0,     true);
        dOp.type = Td.real().ngen();
        int n = std::min(ns, nd);

        if (!preserveSrc) {
            sOp.overwrite = true;
            sOp.overwriteStride = (sblock.*ny == 1 && sblock.crosspack > 1);
        }

        dOp.overwriteStride = (dblock->*ny == 1 && dblock->crosspack > 1);

        // Prepare instruction modifiers.
        if (!alpha.fixed())
            n = std::min(n, 2 * ned);   /* for bank conflict avoidance against alpha */
        if (sCM != dCM)
            n = 1;                      /* use scalar moves if transposing */

        InstructionModifier mod;
        if (Td.isInteger() && !Ts.isSubsetOf(Td))
            mod |= sat;

        // Prepare operands.
        if (alpha == -1)
            sOp = -sOp;
        else if (alpha.fixed() && alpha != 1)
            stub("Unsupported scaling factor");

        // Issue pseudo-instructions.
        if (alpha.fixed()) {
            auto &copy = plan.append(Opcode::mov, n, mod, dOp, sOp);
            if (Ts.is3()) {
                copy.u3Regioning.columnPitchBytes = sblock.crosspack;
                copy.u3Regioning.rowPitchBytes = sblock.ld * sblock.crosspack;
            }
        } else
            plan.append(Opcode::mul, n, mod, dOp, sOp, alpha_real.getRegAvoiding(hw, sOp.ngen()));

        eoffX += n;
    } /* eoffX loop */
    } /* eoffY loop */
    } /* sblock loop */

    copyExecute(std::move(plan), state);
}

template <HW hw>
void Generator<hw>::copyExecute(CopyPlan &&plan, CommonState &state)
{
    int nflag = FlagRegister::subcount(hw);
    constexpr int nflagMax = 8;
    std::array<bool, nflagMax> usedFlags = {};

    if (nflag > nflagMax) stub();

    // Lower plan to valid instructions.
    plan.transform();

    int flagRegs = div_up(plan.tempFlagBytes(), 4);

    // Prepare temporary VirtualFlagAllocators for flag allocations.
    // Use lock bits to track overwritable flags.
    auto raVFlag0 = state.raVFlag;
    if (!state.vflagsEnabled())
        for (int i = 0; i < nflag; i++)
            if (!raVFlag0.isFree(VirtualFlag{i}))
                raVFlag0.lock(VirtualFlag{i}, true);
    auto raVFlag = raVFlag0;

    // If we have enough free flags, use those.
    // Otherwise, use unlocked flags if there are enough.
    // Otherwise, free up the locked flags as well, and prepare to save them.
    auto countFree = [&]() {
        int free = 0;
        for (int i = 0; i < nflag/2; i++)
            if (raVFlag.isFree(FlagRegister{i}))
                free++;
        return free;
    };

    Subregister savedFlags[nflagMax/2];
    if (flagRegs > countFree()) {
        raVFlag.freeUnlocked();
        if (flagRegs > countFree()) {
            raVFlag = VirtualFlagAllocator{hw};
            for (int i = 0; i < nflag/2; i++)
                if (raVFlag0.isLocked(VirtualFlag{2*i}) || raVFlag0.isLocked(VirtualFlag{2*i + 1}))
                    savedFlags[i] = state.ra.allocSub<uint32_t>();
        }
    }

    // Allocate temporaries.
    CopyPlan::GRFAllocator grfAllocator = [&](int count, GRFRange &range) {
        if (count > 0)
            range = state.ra.tryAllocRange(count);
        else
            state.ra.release(range);
    };

    CopyPlan::FlagAllocator flagAllocator = [&](int bytes, FlagRegister &flag) {
        if (bytes > 0) {
            int n = bytes >> 1;
            flag = raVFlag.tryAlloc(n);
            for (int i = 0; i < n; i++)
                usedFlags[flag.index(hw) + i] = true;
        } else
            raVFlag.release(flag);
    };

    plan.materializeTemps(grfAllocator, flagAllocator);

    // Save off flag registers that will be clobbered.
    auto clobbered = usedFlags;
    for (int i = 0; i < nflag; i++)
        clobbered[i] &= raVFlag0.isLocked(VirtualFlag{i});
    for (int i = 0; i < nflag/2; i++)
        if (clobbered[2*i] || clobbered[2*i + 1])
            mov(1, savedFlags[i], FlagRegister(i));

    // Generate code.
    plan.execute(*this);

    // Restore flag registers that were clobbered, and invalidate
    //  virtual flags as needed.
    for (int i = 0; i < nflag/2; i++) {
        if (clobbered[2*i] || clobbered[2*i + 1])
            mov(1, FlagRegister(i), savedFlags[i]);
        state.ra.safeRelease(savedFlags[i]);
    }
    for (int i = 0; i < nflag; i++)
        if (usedFlags[i] && !raVFlag0.isFree(VirtualFlag{i}))
            state.activeVFlags[i].clear();
}

// Copy one GRFMultirange to another, allowing overlap between the two.
template <HW hw>
void Generator<hw>::overlappedCopy(const GRFMultirange &src, const GRFMultirange &dst, CommonState &state)
{
    constexpr int regs = GRF::maxRegs();
    std::array<int16_t, regs> map;

    std::vector<int16_t> temps;
    temps.reserve(src.getLen());

    std::vector<GRF> alloced;

    for (auto &m: map) m = -1;

    for (int i = 0; i < src.getLen(); i++)
        if (src[i].getBase() != dst[i].getBase())
            map[src[i].getBase()] = dst[i].getBase();

    int n = 1, ne = elementsPerGRF<uint32_t>(hw);
    bool done = false;
    bool useFloat = false;

    while (!done) {
        bool progress = false;
        done = true;

        // Copy registers where dst doesn't overlap src, then clear associated entries.
        for (int i = 0; i < regs; i += n) {
            n = 1;
            if (map[i] >= 0)
                done = false;

            if (map[i] >= 0 && map[map[i]] < 0) {
                temps.push_back(i);
                if (i + 1 < regs && map[i + 1] == map[i] + 1) {
                    /* copy 2 consecutive registers at once */
                    temps.push_back(map[i + 1]);
                    map[i + 1] = -1;
                    n++;
                }

                auto dt = useFloat ? DataType::f : DataType::ud;
                useFloat = !useFloat;

                mov(n * ne, GRF(map[i]).retype(dt), GRF(i).retype(dt));
                map[i] = -1;
                progress = true;
            }
        }

        if (!progress && !done) {
            // Get a few temporaries to break cycles, copy and continue.
            // Grab temporaries from already-moved src registers if available.
            int unstuck = 0;
            constexpr int maxUnstuck = 8;
            std::array<int16_t, maxUnstuck> from, to;

            for (int i = 0; i < regs; i++) if (map[i] >= 0) {
                GRF temp;
                if (temps.empty()) {
                    temp = state.ra.tryAlloc();
                    if (temp.isInvalid()) {
                        if (unstuck == 0) throw out_of_registers_exception();
                        break;
                    }
                    alloced.push_back(temp);
                } else {
                    temp = GRF(temps.back());
                    temps.pop_back();
                }

                mov<int32_t>(ne, temp, GRF(i));
                from[unstuck] = temp.getBase();
                to[unstuck] = map[i];
                map[i] = -1;
                if (++unstuck >= maxUnstuck) break;  /* that's enough for now */
            }

            for (int j = 0; j < unstuck; j++)
                map[from[j]] = to[j];
        }
    }

    for (auto &r: alloced)
        state.ra.release(r);
}

GEMMSTONE_NAMESPACE_END
