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


#include "alloc_utils.hpp"
#include "gemmstone/generator.hpp"
#include "map.hpp"
#include "layout_utils.hpp"
#include "state_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;


// Generate the kernel prologue.
template <HW hw>
void Generator<hw>::prologue(const CommonStrategy &strategy, int internalSIMD)
{
    uint32_t cr0Enable;

    interface.generatePrologue(*this);

    bool denorm = strategy.ieeeDenormals;
    cr0Enable = 0x1000;                                     // IEEE float->int rounding.
    if (denorm)                   cr0Enable |= 0x4C0;       // Enable hf|f|df denormals.
    if (hw >= HW::Xe2 && denorm)  cr0Enable |= 0x40000000;  // Enable bf|tf|f8|f4 denormals.
    if (strategy.spf)             cr0Enable |= 0x4;         // Enable single program flow.

    or_(1, cr0, cr0, cr0Enable);

    InstructionModifier imod = 1;

    if (internalSIMD == 16 && interface.getSIMD() < 16)
        mov(imod, sr0[2], uint16_t(0xFFFF));
    if (internalSIMD == 32 && interface.getSIMD() < 32)
        mov(imod, sr0[2], uint32_t(0xFFFFFFFF));
}

template <HW hw>
void Generator<hw>::prologue(const GEMMStrategy &strategy, GEMMState &state)
{
    prologue(strategy, state.internalSIMD());
}

// Generate the kernel epilogue.
template <HW hw>
void Generator<hw>::epilogue(const CommonStrategy &strategy, CommonState &state)
{
    auto r0_info = state.r0_info;

#ifdef STANDALONE
    if (strategy.tracking.startEnd)
        addTimepoint(TimepointThreadEnd, state);
#endif

    if (!getEfficient64Bit() && r0_info.getBase() < 112) {
        mov<uint32_t>(r0DWords(hw), r127, r0_info);
        r0_info = r127;
    }

    if (strategy.finalFence) {
        memfence(r124, r0_info);
        fencewait();
    }

    threadend(r0_info);
}

// Pad the end of the kernel to accommodate instruction prefetching.
template <HW hw>
void Generator<hw>::padding()
{
    for (int q = 0; q < 8; q++)
        nop();
}

// Create a copy of a SubregisterPair in the other bank.
template <HW hw>
void Generator<hw>::duplicateScalar(SubregisterPair &val, CommonState &state)
{
    auto reg0 = val.getReg(0);

    if (val.isDuplicated()) return;
    if (reg0.isInvalid()) return;

    auto bundle = Bundle::locate(hw, reg0);
    auto reg1 = state.ra.alloc_sub(reg0.getType(), Bundle(bundle.bank_id ^ 1, Bundle::any));

    mov(1, reg1, reg0);
    val = SubregisterPair(reg0, reg1);
}

template <HW hw>
void Generator<hw>::deduplicateScalar(SubregisterPair &val, CommonState &state)
{
    auto reg0 = val.getReg(0), reg1 = val.getReg(1);
    if (reg0 != reg1) {
        state.ra.release(reg1);
        val = SubregisterPair(reg0);
    }
}

// Create multiple versions of the input subregister reg, shifted by amounts specified by the shifts bitmask.
// The input subregister is used for one of the versions.
template <HW hw>
MultishiftSubregister Generator<hw>::multishift(const Subregister &reg, unsigned int shifts,
                                                const CommonStrategy &strategy, CommonState &state, Bundle hint)
{
    MultishiftSubregister ms;

    while (shifts != 0) {
        int shift = bsr(shifts);
        shifts &= ~(1 << shift);

        if (shifts != 0) {
            Subregister s = state.ra.alloc_sub(reg.getType(), hint);
            ms.set(shift, s);
            eshr(1, s, reg, shift, strategy, state);
        } else {
            ms.set(shift, reg);
            if (shift > 0)
                eshr(1, reg, reg, shift, strategy, state);
        }
    }

    return ms;
}

template <HW hw>
FlagRegister Generator<hw>::getPhysicalFlag(VirtualFlag vflag, CommonState &state)
{
    VirtualFlag pflag;

    if (state.vflagsEnabled()) {
        // Check if virtual flag is currently active.
        int pidx = -1;
        for (int i = 0; i < FlagRegister::subcount(hw); i += vflag.n)
            if (state.activeVFlags[i] == vflag)
                pidx = i;
        for (int i = 1; i < int(vflag.n); i++)
            if (state.activeVFlags[pidx + i] != vflag)
                pidx = -1;

        // If flag is not currently active, load it into a physical flag.
        if (pidx == -1) {
            auto freg = state.raVFlag.assignPhysical(vflag);
            pidx = freg.index();
            mov(1, freg, getMaskFlag(hw, vflag, state));
            for (int i = 0; i < int(vflag.n); i++)
                state.activeVFlags[pidx + i] = vflag;
        }

        pflag = VirtualFlag{pidx, vflag.n};
    } else {
        if (state.raVFlag.isVirtual(vflag))
            stub("Need virtual flag registers");

        pflag = vflag;
    }

    return pflag.toPhysical(hw);
}

template <HW hw>
void Generator<hw>::allocVFlagStorage(const CommonStrategy &strategy, CommonState &state, bool saveCurrent)
{
    if (!state.vflagStorage.empty())
        return;

    state.vflagStorage.append(state.ra.alloc(getHint(HintType::LongTerm, strategy)));

    if (saveCurrent)
        for (int i = 0; i < FlagRegister::count(hw); i++)
            mov(1, state.vflagStorage[0].ud(i), FlagRegister(i).ud());
}

template <HW hw>
void Generator<hw>::deallocVFlagStorage(CommonState &state, bool saveCurrent)
{
    if (state.vflagStorage.empty())
        return;

    if (saveCurrent) for (int i = 0; i < FlagRegister::subcount(hw); i++) {
        auto flag = FlagRegister::createFromIndex(i);
        if ((i & 1) == 0 && !state.raVFlag.isLocked(VirtualFlag(i, 2))) {
            mov(1, flag.ud(), state.vflagStorage[0].ud(i >> 1));
            i++;
        } else if (!state.raVFlag.isLocked(VirtualFlag(i)))
            mov(1, flag, state.vflagStorage[0].uw(i));
    }

    safeReleaseRanges(state.vflagStorage, state);
}

// Get ID of fused thread (0/1), multiplied by a scaling factor. Assumes r1 has not been overwritten,
//  or state.lid0 is set to a subregister containing local ID 0 (divided by the subgroup size).
template <HW hw>
void Generator<hw>::getFusedID(int scale, const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state)
{
    if (strategy.fused) {
        state.fusedID = state.ra.alloc_sub<uint16_t>(getHint(HintType::LongTerm, strategy));
        if (state.lid0.isValid()) {
            if (is_zero_or_pow2(scale) && scale > 1 && (state.fusedID.getByteOffset() & 3) == 0)
                bfi2(1, state.fusedID, scale, state.lid0, 0);
            else {
                and_(1, state.fusedID, state.lid0, 1);
                mulConstant(1, state.fusedID, state.fusedID, scale);
            }
        } else if (is_zero_or_pow2(scale)) {
            int shift = ilog2(scale) - ilog2(strategy.subgroupSize);
            Subregister lid0 = r1.uw(0);

            if (shift > 0)
                shl(1, state.fusedID, lid0, uint16_t(shift));
            else if (shift < 0)
                shr(1, state.fusedID, lid0, uint16_t(-shift));

            and_(1, state.fusedID, (shift == 0) ? lid0 : state.fusedID, uint16_t(scale));
        } else {
            shr(1, state.fusedID, r1.uw(0), uint16_t(ilog2(strategy.subgroupSize)));
            and_(1, state.fusedID, state.fusedID, uint16_t(1));
            mulConstant(1, state.fusedID, state.fusedID, uint16_t(scale));
        }
    }
}

// Move r0 information to another register if configured.
template <HW hw>
void Generator<hw>::moveR0(const CommonStrategy &strategy, CommonState &state)
{
    if (state.movedR0)
        return;
    if (state.r0_info.isInvalid()) {
        switch (strategy.moveR0) {
            case MoveR0::None: state.r0_info = r0.ud();   state.movedR0 = true; return;
            case MoveR0::Acc:  state.r0_info = acc0.ud(); break;
            case MoveR0::Addr: state.r0_info = a0.ud();   break;
            case MoveR0::GRF:
                state.r0_info = state.ra.alloc(getHint(HintType::R0Info, strategy));
                break;
        }
    }

    mov<uint32_t>(r0DWords(hw), state.r0_info, r0);

    state.ra.release(r0);

    state.movedR0 = true;
}

template <HW hw>
void Generator<hw>::moveR0(const GEMMStrategy &strategy, GEMMState &state)
{
    if (state.movedR0)
        return;
    if (strategy.moveR0 == MoveR0::GRF) {
        if (strategy.registerScheme == GEMMStrategy::ACB || strategy.registerScheme == GEMMStrategy::BCA) {
            state.r0_info = r127;
            state.ra.claim(r127);
        }
    }
    moveR0(static_cast<CommonStrategy>(strategy), state);
}

// Divide out subgroup size from x local size and local ID.
template <HW hw>
void Generator<hw>::removeSG(const CommonProblem &problem, const CommonStrategy &strategy, const CommonState &state)
{
    uint16_t sss = ilog2(strategy.subgroupSize);

    auto localSize0 = interface.getLocalSize(0);
    auto localID0 = interface.getLocalID(0);

    shr(1, localSize0, localSize0, sss);
    shr(1, localID0.uw(0), localID0.uw(0), sss);
}

// Swap bit 0 of local ID x and y if needed so that threads are ordered according to specified EU fusion.
template <HW hw>
void Generator<hw>::reorderFusedEUs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (!strategy.fused) return;

    if (strategy.loopOrder[0] != strategy.fusedLoop) {
        auto temp = state.ra.alloc_sub<uint32_t>();
        and_(1, temp, state.inputs.localIDN.ud(), uint16_t(1));
        bfi2(1, state.inputs.localIDN.ud(), uint16_t(1), state.inputs.localIDM.ud(), state.inputs.localIDN.ud());
        bfi2(1, state.inputs.localIDM.ud(), uint16_t(1), temp, state.inputs.localIDM.ud());
        state.ra.safeRelease(temp);
    }
}

template <HW hw>
Subregister Generator<hw>::copySubregister(const Subregister &reg, CommonState &state, Bundle hint)
{
    if (reg.isInvalid()) return reg;
    auto copy = state.ra.alloc_sub(reg.getType(), hint);
    mov(1, copy, reg);
    return copy;
}

template <HW hw>
GRF Generator<hw>::loadScalars(Type T, const std::vector<Subregister> &src, const CommonStrategy &strategy, CommonState &state)
{
    auto n = int(src.size());
    int simd = roundup_pow2(n);

    if (n == 0) return GRF{};

    GRF header = GRF(src[0].getBase());
    GRF data = state.ra.alloc();
    bool tempHeader = false;
    bool lsc = (hw >= HW::XeHPG);

    int esize = std::min(T.paddedSize(), 8);
    int expand = T.paddedSize() / esize;

    // Try to read addresses in place if possible.
    if (expand > 1)
        tempHeader = true;
    else for (int i = 0; i < n; i++) {
        if (src[i].getOffset() != i) {
            tempHeader = true;
            break;
        }
    }

    // If not, create a header.
    if (tempHeader) {
        header = state.ra.alloc();
        int off = 0;
        for (int i = 0; i < n; i++) {
            emov(1, header.uq(off++), src[i], strategy, state);
            for (int x = 1; x < expand; x++)
                eadd(1, header.uq(off++), src[i], x * esize, strategy, state);
        }
    }

    InstructionModifier mod = simd * expand;
    FlagRegister flag;
    if (n < simd) {
        flag = state.raVFlag.alloc();
        mov(1, flag, (1u << (n * expand)) - 1);
        mod = mod | flag;
    }

    switch (esize) {
        case 1:
            lsc ? load(mod, data, D8U32 | CacheSettingsLSC::L1C_L3C,  A64, header)
                : load(mod, data, scattered_byte(1),                  A64, header);
            break;
        case 2:
            lsc ? load(mod, data, D16U32 | CacheSettingsLSC::L1C_L3C, A64, header)
                : load(mod, data, scattered_byte(2),                  A64, header);
            break;
        case 4:
            lsc ? load(mod, data, D32 | CacheSettingsLSC::L1C_L3C,    A64, header)
                : load(mod, data, scattered_dword(),                  A64, header);
            break;
        case 8:
            lsc ? load(mod, data, D64 | CacheSettingsLSC::L1C_L3C,    A64, header)
                : load(mod, data, scattered_qword(),                  A64, header);
            break;
        default: stub();
    }

    if (tempHeader)
        state.ra.safeRelease(header);
    state.raVFlag.safeRelease(flag);

    return data.retype(T.ngen());
}

// Load a contiguous vector from memory, with optional remainder handling.
template <HW hw>
RegisterRange Generator<hw>::loadVector(Type Tsrc, Type Tdst, Subregister ptr, int n, Subregister rem, const CommonStrategy &strategy, CommonState &state)
{
    MatrixAddressing atype;
    MatrixAddressingStrategy astrategy;
    vector<RegisterRange> addrs;
    vector<MaskAssignment> masks;
    Subregister rems[3] = {rem};
    Subregister remTemp;

    if (Tsrc.isSubByteInt() && Tdst.isSubByteInt()) {
        // Temporary int4/2 path until copyRegisters supports sub-byte int copies.
        if (Tsrc.bits() != Tdst.bits()) stub();
        if (rem.isValid()) {
            remTemp = state.ra.alloc_sub<int32_t>();
            avg(1, remTemp, rem, 0);
            rems[0] = remTemp;
        }
        int bits = Tsrc.bits();
        Tsrc = Tdst = Type::u8;
        n = (n * bits + 7) >> 3;
    }

    atype.layout = MatrixLayout::N;
    atype.packSize = 0;
    atype.crosspack = 1;
    atype.alignment = Tsrc.paddedSize();

    astrategy.base = A64;
    astrategy.accessType = AccessType::Block;
    astrategy.newDP = (hw >= HW::XeHPG);

    RegisterLayout layout(hw, Tsrc, n, 1, atype, astrategy, rem.isValid(), false, false);

    auto regs = state.ra.alloc_range(layout.regs());

    allocAddrRegs(hw, addrs, layout, state);
    setupAddr(addrs, ptr, layout, Subregister(), strategy, state);
    if (!assignMasks(layout, LoopM, LoopN, masks, strategy, state, true)) stub();
    loadMasks(masks, rems, strategy, state);
    loadMatrix(regs, layout, addrs, strategy, state);
    safeReleaseMaskAssignments(masks, state);
    safeReleaseRanges(addrs, state);
    state.ra.safeRelease(remTemp);

    if (!layout.hasFullCrosspack(1) || Tsrc.bits() != Tdst.bits()) {
        // Data didn't come in with unit stride. Repack it.
        RegisterLayout nlayout(hw, Tdst, n, 1, true);
        auto nregs = state.ra.alloc_range(nlayout.regs());
        copyRegisters(layout, nlayout, regs, nregs, strategy, state, false);

        state.ra.safeRelease(regs);
        regs = std::move(nregs);
    } else if (Tsrc != Tdst)
        convert(regs, Tsrc, Tdst, strategy, state);

    return regs;
}

// Set a matrix to zero.
template <HW hw>
void Generator<hw>::zeroMatrix(const GRFMultirange &r, const CommonStrategy &strategy)
{
    int i = 0;
    map<uint32_t>(hw, r, r, strategy, [&](int esize, GRF reg, GRF _) {
        (i++ & 1) ? mov(esize, reg.f(), 0.f)
                  : mov(esize, reg, 0);
    });
}

template <HW hw>
void Generator<hw>::extendIndexVec(int n, CommonState &state)
{
    auto &indexVec = state.indexVec;
    auto &ivEntries = state.ivEntries;

    if (n > ivEntries) {
        int simd = GRF::bytes(hw) >> 1;
        int nregs = div_up(n, simd);
        int cregs = indexVec.getLen();
        if (nregs > cregs)
            indexVec.ranges.push_back(state.ra.alloc_range(nregs - cregs));
        if (ivEntries == 0) {
            mov<uint16_t>(8, indexVec[0][0](1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
            ivEntries = 8;
        }
        if (n > 8 && ivEntries < 16) {
            mov<uint16_t>(8, indexVec[0][8](1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));
            ivEntries = 16;
        }
        if (GRF::bytes(hw) > 32 && n > 16 && ivEntries < 32) {
            add<uint16_t>(16, indexVec[0][16](1), indexVec[0].uw(0)(1), 16);
            ivEntries = 32;
        }
        if (n > ivEntries) {
            for (int e = std::max(cregs, 1); e < nregs; e++)
                add<uint16_t>(simd, indexVec[e], indexVec[0], simd * e);
            ivEntries = nregs * simd;
        }
    }
}

template <HW hw>
Subregister Generator<hw>::accessIndexVec(int n, CommonState &state)
{
    if (n >= state.ivEntries)
        extendIndexVec(n, state);

    int simd = GRF::bytes(hw) >> 1;
    return state.indexVec[n / simd].uw(n % simd);
}

template <HW hw>
LDMultiples Generator<hw>::createLDMultiples(bool a64, int nmultiples, const Subregister &ld,
                                             const CommonStrategy &strategy, CommonState &state)
{
    int simd = GRF::bytes(hw) >> (a64 ? 3 : 2);
    int nregs = div_up(nmultiples, simd);
    auto r = state.ra.try_alloc_range(nregs);

    GRF tempHi = state.emulate.temp[0], tempLo = state.emulate.temp[1];
    bool freeTempHi = false, freeTempLo = false;
    if (a64) {
        if (tempHi.isInvalid()) { tempHi = state.ra.alloc(); freeTempHi = true; }
        if (tempLo.isInvalid()) { tempLo = state.ra.alloc(); freeTempLo = true; }
    }

    if (r.isValid()) {
        extendIndexVec(nmultiples, state);
        for (int i = 0; i < nregs; i += 2) {
            auto thisSIMD = simd * std::min(nregs - i, 2);
            auto iv = accessIndexVec(simd * i, state)(1);
            if (a64) {
                if (!strategy.emulate.emulate64_mul) {
                    mov(thisSIMD, r[i].ud(0)(2), iv);
                    mul(thisSIMD, r[i].uq(), r[i].ud(0)(2), ld);
                } else {
                    if (strategy.moveR0 == MoveR0::Acc) stub();
                    mul<uint32_t>(thisSIMD, acc0, ld, iv);
                    mach<uint32_t>(thisSIMD, tempHi, ld, Immediate::ud(0));
                    mov<uint32_t>(thisSIMD, tempLo, acc0);
                    mov<uint32_t>(thisSIMD, r[i][1](2), tempHi);
                    mov<uint32_t>(thisSIMD, r[i][0](2), tempLo);
                }
            } else
                mul<uint32_t>(thisSIMD, r[i], ld, iv);
        }
    }

    if (freeTempHi) state.ra.safeRelease(tempHi);
    if (freeTempLo) state.ra.safeRelease(tempLo);

    if (!r.isValid())
        return LDMultiples{};

    LDMultiples result;
    result.range = r;
    result.a64 = a64;
    result.count = nregs * simd;

    return result;
}

template <HW hw>
Subregister Generator<hw>::findLDMultiple(const LDMultiples &multiples, bool a64, int n, const CommonStrategy &strategy, CommonState &state)
{
    int simd = GRF::bytes(hw) >> (multiples.a64 ? 3 : 2);
    int off = (n / simd), sub = (n % simd);

    if (multiples.range.isInvalid())
        return Subregister();
    if (off < 0 || off >= multiples.range.getLen())
        return Subregister();
    if (a64 && !multiples.a64)
        return Subregister();

    return !multiples.a64 ? multiples.range[off].ud(sub) :
                      a64 ? multiples.range[off].uq(sub) :
                            multiples.range[off].ud(2 * sub);
}

// Calculate and cache a specific ld multiple.
template <HW hw>
void Generator<hw>::calcIncrement(LDIncrements &increments, SubregisterPair &base, int scale,
                                  const CommonStrategy &strategy, CommonState &state,
                                  bool avoidConflicts)
{
    avoidConflicts &= strategy.avoidIncConflicts;

    // Check for existing increment.
    for (auto &inc: increments)
        if (inc.first == scale)
            return;

    // Copy base for scale = 1.
    if (scale == 1) {
        if (strategy.avoidIncConflicts)
            duplicateScalar(base, state);
        increments.push_back(std::make_pair(1, base));
        return;
    }

    // General scaling.
    SubregisterPair scaled;
    if (avoidConflicts)
        scaled = SubregisterPair(state.ra.alloc_sub(increments.type, getHint(HintType::LongTerm0, strategy)),
                                 state.ra.alloc_sub(increments.type, getHint(HintType::LongTerm1, strategy)));
    else
        scaled = SubregisterPair(state.ra.alloc_sub(increments.type, getHint(HintType::LongTerm, strategy)));

    emulConstant(1, scaled.getReg(0), base, scale, strategy, state);
    if (scaled.isDuplicated()) emov(1, scaled.getReg(1), scaled.getReg(0), strategy, state);

    increments.push_back(std::make_pair(scale, scaled));
}

// Get a multiple of lda/ldb for address increments.
template <HW hw>
SubregisterPair Generator<hw>::lookupIncrement(const LDIncrements &increments, const SubregisterPair &base, int scale, const CommonStrategy &strategy, CommonState &state, bool *release)
{
    if (release) *release = false;

    for (auto &inc: increments)
        if (inc.first == scale)
            return inc.second;

    if (!release)
        return SubregisterPair();

    auto result = state.ra.alloc_sub(increments.type);
    emulConstant(1, result, base, scale, strategy, state);
    *release = true;

    return SubregisterPair(result);
}

template <HW hw>
void Generator<hw>::broadcastToWG(FlagRegister leaderFlag, GRF value, const CommonStrategy &strategy, CommonState &state, int slmOffset)
{
    if (getBytes(value.getType()) != 4) stub();

    auto header = state.ra.alloc().ud();
    mov<uint32_t>(1, header, slmOffset);

    (hw >= HW::XeHPG) ? store(1 | leaderFlag, D32,                           SLM, header, value)
                      : store(1 | leaderFlag, surface_dword(ChannelMask::r), SLM, header, value);
    useTempAndR0(state, [&](GRF temp, GRF r0_info) {
        slmBarrier(temp, r0_info, strategy);
    });
    (hw >= HW::XeHPG) ? load(1 | ~leaderFlag, value, D32,                           SLM, header)
                      : load(1 | ~leaderFlag, value, surface_dword(ChannelMask::r), SLM, header);

    state.ra.safeRelease(header);
}

// Common interface initialization code.
template <HW hw>
void Generator<hw>::initInterface(const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state)
{
    interface.requireArbitrationMode(strategy.arbitrationMode);
}

// Common state initialization code.
template <HW hw>
void Generator<hw>::initState(const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state)
{
    interface.requireLocalID(3);
    interface.requireLocalSize();
    if (problem.nonuniformWGs)
        interface.requireNonuniformWGs();

    if (strategy.wgInSS)
        interface.requireBarrier();

    interface.requireSIMD(strategy.subgroupSize);

    interface.requireNoPreemption();

    if (strategy.raHW != hw)
        state.ra = RegisterAllocator(strategy.raHW);

    requireGRF(strategy.GRFs);
    interface.requireGRF(strategy.GRFs);
    state.ra.setRegisterCount(strategy.GRFs);
    state.tokenAllocator = TokenAllocator(hw, strategy.GRFs);

    setEfficient64Bit(interface.getEfficient64Bit());

    // sendgx encodes r511 as null.
    if (getEfficient64Bit() && strategy.GRFs > 256)
        state.ra.claim(GRF(511));

    if (problem.gtpinSupport)
        interface.requireScratch(128);

    for (int i = 0; i < FlagRegister::subcount(hw); i++)
        state.activeVFlags[i].clear();
}

GEMMSTONE_NAMESPACE_END
