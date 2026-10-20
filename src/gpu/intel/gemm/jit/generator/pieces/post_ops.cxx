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
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "state_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


// Perform alpha scaling and update problem to reflect it.
template <HW hw>
void Generator<hw>::gemmAlphaScale(GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool cxCombine)
{
    auto Tacc = state.Tacc;
    auto  &alpha = problem.alpha;
    auto valphar = state.inputs.alpha_real;

    if (alpha == -1) {
        map(hw, Tacc.real(), state.C_regs[0], state.C_regs[0], strategy,
            [&](int esize, GRF acc, GRF _) {
                mov(esize, acc, -acc);
            }
        );
    } else if (alpha != 1) {
        map(hw, Tacc.real(), state.C_regs[0], state.C_regs[0], strategy,
            [&](int esize, GRF acc, GRF _) {
                alpha.fixed() ? mul(esize, acc, acc, cast(Tacc.real(), alpha))
                              : mul(esize, acc, acc, valphar.getRegAvoiding(hw, acc));
            }
        );
    }

    alpha = 1;
}

// Perform beta scaling, including type conversions.
template <HW hw>
void Generator<hw>::gemmBetaScale(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    Label labelBetaDone;

    auto Ts = problem.Ts;
    auto beta = problem.beta;
    auto vbetar = state.inputs.beta_real;

    if (state.beta1.isValid()) {
        if (strategy.fused) {
            cmp(16 | lt | state.flagAP, null.d(), state.beta1, int16_t(0));
            goto12(16 | state.flagAP, labelBetaDone);
        } else {
            cmp(1 | lt | state.flagAP, null.d(), state.beta1, int16_t(0));
            jmpi(1 | state.flagAP, labelBetaDone);
        }
    }

    gemmConvertC(problem.Ts, problem, strategy, state);

    if (beta != 1) {
        map(hw, Ts.real(), state.C_regs[0], state.C_regs[0], strategy, [&](int esize, GRF acc, GRF _) {
            beta.fixed() ? mul(esize, acc, acc, cast(Ts.real(), beta))
                         : mul(esize, acc, acc, vbetar.getRegAvoiding(hw, acc));
        });

        if (problem.sumA || problem.sumB) {
            auto Tc = problem.Tc;
            auto &Xs_regs   = problem.sumA ? state.As_regs   : state.Bs_regs;
            auto &Xs_layout = problem.sumA ? state.As_layout : state.Bs_layout;

            auto Xs_usedRegs = Xs_regs.subrange(0, Xs_layout.regs());

            map(hw, Tc.real(), Xs_usedRegs, Xs_usedRegs, strategy, [&](int esize, GRF acc, GRF _) {
                beta.fixed() ? mul(esize, acc, acc, cast(Tc.real(), beta))
                             : mul(esize, acc, acc, vbetar.getRegAvoiding(hw, acc));
            });
        }
    }

    gemmConvertC(problem.Tc, problem, strategy, state);

    mark(labelBetaDone);

    if (state.beta1.isValid() && strategy.fused)
        join(16);
}

template <HW hw>
void Generator<hw>::binaryOp(BinaryOp op, int simd, const RegData &dst, const RegData &src0, const RegData &src1, CommonState &state)
{
    switch (op) {
        case BinaryOp::Add: add(simd, dst, src0, src1); break;
        case BinaryOp::Sub: add(simd, dst, src0, -src1); break;
        case BinaryOp::Mul: mul(simd, dst, src0, src1); break;
        case BinaryOp::Div: stub();
        case BinaryOp::Min: min_(simd, dst, src0, src1); break;
        case BinaryOp::Max: max_(simd, dst, src0, src1); break;
        case BinaryOp::Prelu: {
            auto T = src1.getType();
            auto nRegs = GRF::bytesToGRFs(hw, simd * getBytes(T));
            auto tempRng = state.ra.alloc_range(nRegs);
            auto temp = tempRng[0].retype(T);
            mul(simd, temp, src0, src1);
            csel(simd | le, dst, temp, src0, src0);
            state.ra.release(tempRng);
            break;
        }
        default: stub();
    }
}

// Apply binary operation to C with a scalar operand.
template <HW hw>
void Generator<hw>::gemmScalarBinaryOpC(BinaryOp op, Type Tco, const GRFMultirange &offsets,
                                        const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tacc = state.Tacc;
    auto offsetTc = offsets[0].sub(0, Tacc.ngen());

    if (Tco != Tacc) {
        RegisterLayout scalarLayout(hw, Tacc, 1, 1, true);
        copyRegisters(Tco, Tacc, scalarLayout, scalarLayout, offsets, offsets, strategy, state);
    }
    if (op == BinaryOp::Div && one_of(state.Tacc, {Type::f32, Type::f16})) {
        inv(1, offsetTc, offsetTc);
        op = BinaryOp::Mul;
    }

    map(hw, state.Tacc, state.C_regs[0], state.C_layout, strategy, [&](int simd, const RegData &r) {
        binaryOp(op, simd, r, r, offsetTc, state);
    });
}

// Apply binary operation to C with a scalar operand from a Subregister (e.g., host side scalar)
template <HW hw>
void Generator<hw>::gemmScalarBinaryOpC(BinaryOp op, Type Tco, const Subregister &scalar,
                                        const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tacc = state.Tacc;
    auto offsetTc = scalar;

    if (Tco != Tacc) {
        offsetTc = state.ra.alloc_sub(Tacc.ngen());
        CopyPlan plan(getProductFamily(), strategy.systolicAvailable);
        plan.append(Opcode::mov, 1, offsetTc, scalar);
        copyExecute(std::move(plan), state);
    }

    if (op == BinaryOp::Div && one_of(state.Tacc, {Type::f32, Type::f16})) {
        inv(1, offsetTc, offsetTc);
        op = BinaryOp::Mul;
    }

    map(hw, state.Tacc, state.C_regs[0], state.C_layout, strategy, [&](int simd, const RegData &r) {
        binaryOp(op, simd, r, r, offsetTc, state);
    });

    if (Tco != Tacc)
        state.ra.safeRelease(offsetTc);
}

// Apply binary operation to C with a vector operand, optionally multiplied by a scalar.
template <HW hw>
void Generator<hw>::gemmVectorBinaryOpC(BinaryOp op, bool column, const GRFMultirange &offsets, const Subregister &scaleIn,
                                        const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                        Type Tco, RegisterLayout CO_layout, int y0, int y1)
{
    auto Tacc = state.Tacc;
    auto ne = elementsPerGRF(hw, Tacc);
    auto globalCM = state.C_layout.colMajor();
    auto unrollX = strategy.unroll[globalCM ? LoopM : LoopN];
    auto unrollY = strategy.unroll[globalCM ? LoopN : LoopM];
    auto crosspack = CO_layout.empty() ? 1 : CO_layout[0].crosspack;
    auto stride = [&]() { return (column == globalCM) ? 0 : crosspack; };
    auto scale = scaleIn;
    const GRFMultirange *offsetsPtr = &offsets;

    if (Tco == Type::invalid) Tco = Tacc;

    if (Tacc == Type::f32 && scale.isValid() && scale.getType() != DataType::f) {
        scale = state.ra.alloc_sub<float>();
        mov(1, scale, scaleIn);
    }

    bool needRepack = (Tacc != Tco);
    needRepack |= (stride() > 1 && hw >= HW::XeHP && Tacc.isFP());

    GRFMultirange repackOffsets;
    if (needRepack) {
        // Repack data to unit stride as float pipe can't swizzle.
        int r =  column ? 1 : strategy.unroll[LoopM];
        int c = !column ? 1 : strategy.unroll[LoopN];
        RegisterLayout repackLayout(hw, Tacc, r, c, !column);
        repackOffsets = state.ra.allocRange(repackLayout.regs());
        copyRegisters(CO_layout, repackLayout, offsets, repackOffsets, strategy, state);
        crosspack = 1;
        offsetsPtr = &repackOffsets;

        // Late inversion, for binary divide.
        if (op == BinaryOp::Div && one_of(Tacc, {Type::f32, Type::f16})) {
            map(hw, Tacc, repackOffsets, repackOffsets, strategy, [&](int simd, GRF r, GRF) {
                inv(simd, r, r);
            });
            op = BinaryOp::Mul;
        }
    }

    if (y0 < 0) y0 = 0;
    if (y1 < 0) y1 = unrollY;

    for (int y = y0; y < y1; y++) {
        for (int x = 0; x < unrollX;) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;
            int nc;
            Subregister C = state.C_layout.find(i, j, state.C_regs[0], &nc);

            nc = std::min({nc, strategy.fmaSIMD / crosspack, 2 * ne});
            auto nco = (column ? j : i) * crosspack;
            auto offBase = (*offsetsPtr)[nco / ne].sub(nco % ne, Tacc.ngen());
            if (scale.isValid()) {
                if (op == BinaryOp::Add)
                    mad(nc, C(1), C(1), offBase(stride()), scale);
                else if (op == BinaryOp::Sub)
                    mad(nc, C(1), C(1), offBase(stride()), -scale);
                else
                    stub();
            } else if (strategy.dotVL > 0) {
                // Dot-based kernels pack the C-tile into one register when possible, which results in
                // register region restriction complications. Split binary operations into execSize=1 pieces.
                for (int off = 0; off < nc; off++) {
                    auto val = C.offset(off)(1);
                    auto src1 = offBase.offset(off)(stride());
                    binaryOp(op, 1, val, val, src1, state);
                }
            } else
                binaryOp(op, nc, C(1), C(1), offBase(stride()), state);

            x += nc;
        }
    }

    if (scale != scaleIn)
        state.ra.safeRelease(scale);
    safeReleaseRanges(repackOffsets, state);
}

// Apply binary operation to C.
template <HW hw>
bool Generator<hw>::gemmBinaryOpC(BinaryOp op, bool row, bool column,
                                  Type Tco, MatrixAddressing CO, MatrixAddressingStrategy CO_strategy,
                                  Subregister base, Subregister ld,
                                  const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    std::vector<RegisterRange> CO_addrs;
    std::vector<MaskAssignment> masks;
    auto globalCM = state.C_layout.colMajor();

    bool recip = false;
    if (op == BinaryOp::Div && Tco == Type::f32) {
        // Implement div as inv+mul for speed, especially when broadcasting.
        recip = true;
        op = BinaryOp::Mul;
    }

    bool matrix = row && column;
    if (matrix) {
        // Matrix case implemented as loop over rows/columns, depending on C's layout.
        row &= globalCM;
        column &= !globalCM;
        CO_strategy.accessType = (isColMajor(CO.layout) == row) ? AccessType::Block :
                                 CO_strategy.base.isStateless() ? AccessType::Scattered
                                                                : AccessType::ChannelScattered;
    } else {
        CO.layout = column ? MatrixLayout::T : MatrixLayout::N;
        CO_strategy.accessType = AccessType::Block;
    }

    bool coColMajor = isColMajor(CO.layout);

#ifdef STANDALONE
    if (problem.readCheck && problem.readCheckC)
        return true;
#endif

    auto cor  = row    ? strategy.unroll[LoopM] : 1;
    auto coc  = column ? strategy.unroll[LoopN] : 1;
    auto remR = row    && !CO_strategy.padded && strategy.remHandling[LoopM] != RemainderHandling::Ignore;
    auto remC = column && !CO_strategy.padded && strategy.remHandling[LoopN] != RemainderHandling::Ignore;

    RegisterLayout CO_layout(hw, Tco, cor, coc, CO, CO_strategy, remR, remC, false);

    allocAddrRegs(hw, CO_addrs, CO_layout, state);
    setupAddr(CO_addrs, base, CO_layout, ld, strategy, state);

    if (!assignMasks(CO_layout, LoopM, LoopN, masks, strategy, state, true)) return false;

    loadMasks(masks, state.remainders, strategy, state);

    if (matrix) {
        auto LoopY = globalCM ? LoopN : LoopM;
        auto unrollY = strategy.unroll[LoopY];
        auto remY = state.remainders[LoopY];
        Label lDone;
        bool checkRemY = !CO_strategy.padded && strategy.remHandling[LoopY] != RemainderHandling::Ignore;
        bool simtCF = strategy.fused && (strategy.fusedLoop == LoopY);
        int simt = simtCF ? 16 : 1;

        if (checkRemY)
            cmp(simt | gt | state.flagAP, remY, 0);

        // Reserve some registers for use in gemmVectorBinaryOpC if needed
        int nreserve = op == BinaryOp::Prelu ? 2 : 0; // temp regs used in the operation
        bool needRepack = (state.Tacc != Tco);
        if (needRepack) {
            auto repackLayout = RegisterLayout(hw, state.Tacc, cor, coc, !column);
            nreserve += repackLayout.regs();
        }
        ngen::RegisterRange reserve;
        reserve = state.ra.alloc_range(nreserve);

        constexpr int max_grouped_ops = 16;
        std::array<ngen::RegisterRange, max_grouped_ops> allCORegs;

        int grouped_ops = 0;
        for (auto &r : allCORegs) {
            r = state.ra.tryAllocRange(CO_layout.regs());
            if (!r.isValid()) break;
            grouped_ops++;
        }

        state.ra.safeRelease(reserve);

        for (int y0 = 0; y0 < unrollY; y0 += grouped_ops) {
            Label lDoneLoading;
            // Perform loads
            for (int i = 0; i < grouped_ops; i++) {
                const int y = y0 + i;
                if (y >= unrollY) break;
                auto &CO_regs = allCORegs[i];
                if (checkRemY) {
                    simtCF ? goto12(16 | ~state.flagAP, lDoneLoading)
                           :   jmpi(1  | ~state.flagAP, lDoneLoading);
                }
                loadMatrix(CO_regs, CO_layout, CO_addrs, strategy, state);
                if (checkRemY && (y + 1 < unrollY))
                    cmp(simt | gt | state.flagAP, remY, y + 1);
                if (coColMajor == globalCM)
                    incAddr(CO_addrs, ld, int(row), int(column), CO_layout, strategy, state);
                else
                    incAddr(CO_addrs, Tco.size(), int(row), int(column), CO_layout, strategy, state);
            }

            mark(lDoneLoading);
            if (simtCF && checkRemY) join(16);
            if (checkRemY)
                cmp(simt | gt | state.flagAP, remY, y0);

            // Perform binary op
            for (int i = 0; i < grouped_ops; i++) {
                auto &CO_regs = allCORegs[i];
                const int y = y0 + i;
                if (y >= unrollY) break;
                if (checkRemY) {
                    simtCF ? goto12(16 | ~state.flagAP, lDone)
                        :   jmpi(1  | ~state.flagAP, lDone);
                }
                if (recip) map(hw, Tco, CO_regs, CO_regs, strategy, [&](int simd, GRF r, GRF) {
                    inv(simd, r, r);
                });
                if (checkRemY && (y + 1 < unrollY))
                    cmp(simt | gt | state.flagAP, remY, y + 1);

                gemmVectorBinaryOpC(op, column, CO_regs, Subregister(), problem, strategy, state, Tco, CO_layout, y, y+1);
            }
        }

        for (auto &r: allCORegs) state.ra.safeRelease(r);

        mark(lDone);
        if (simtCF) join(16);
    } else {
        auto CO_regs = state.ra.allocRange(CO_layout.regs());
        loadMatrix(CO_regs, CO_layout, CO_addrs, strategy, state);
        if (recip) map(hw, Tco, CO_regs, CO_regs, strategy, [&](int simd, GRF r, GRF) {
            inv(simd, r, r);
        });

        if (!row && !column)
            gemmScalarBinaryOpC(op, Tco, CO_regs, problem, strategy, state);
        else
            gemmVectorBinaryOpC(op, column, CO_regs, Subregister(), problem, strategy, state, Tco, CO_layout);
        state.ra.safeRelease(CO_regs);
    }

    safeReleaseMaskAssignments(masks, state);
    safeReleaseRanges(CO_addrs, state);

    return true;
}

// Check kernel input for desired C offset and apply it.
template <HW hw>
bool Generator<hw>::gemmApplyCOffsetDispatch(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    Label labelCOColumn, labelCORow, labelCOMatrix, labelCODone;
    bool doMatrix = problem.allowMatrixOffset();
    auto Tco = problem.Tco;
    auto &CO = problem.CO;
    auto &CO_strategy = strategy.CO;
    auto &effCO = state.effCO;
    auto &ldco = state.inputs.ldco;

    bool ok = true;

    if (state.flagSwizzle.isValid()) state.raVFlag.release(state.flagSwizzle);

    auto flagNonfinal = state.raVFlag.alloc();
    auto flagCOC = state.raVFlag.alloc();
    auto flagCOR = state.raVFlag.alloc();

    and_(1 | nz | flagNonfinal, null.ud(), state.inputs.flags, FlagNonfinalKBlock);
    and_(1 | nz | flagCOC, null.ud(), state.inputs.flags, FlagCOColumn);
    and_(1 | nz | flagCOR, null.ud(), state.inputs.flags, FlagCORow);
    jmpi(1 | flagNonfinal, labelCODone);
    jmpi(1 | flagCOC, labelCOColumn);
    jmpi(1 | flagCOR, labelCORow);

    state.raVFlag.safeRelease(flagNonfinal);
    state.raVFlag.safeRelease(flagCOC);
    state.raVFlag.safeRelease(flagCOR);

    if (state.flagSwizzle.isValid()) state.raVFlag.claim(state.flagSwizzle);

    if (problem.cOffsetHostScalar() && state.inputs.co.isValid()) {
        status << "Applying host scalar C offset" << status_stream::endl;
        gemmScalarBinaryOpC(BinaryOp::Add, Tco, state.inputs.co, problem, strategy, state);
    } else {
        status << "Applying fixed C offset" << status_stream::endl;
        ok = ok && gemmBinaryOpC(BinaryOp::Add, false, false, Tco, CO, CO_strategy, effCO, ldco, problem, strategy, state);
    }
    jmpi(1, labelCODone);

    mark(labelCOColumn);
    if (!problem.cOffsetHostScalar()) {
        if (doMatrix) jmpi(1 | flagCOR, labelCOMatrix);
        status << "Applying column-wise C offset" << status_stream::endl;
        ok = ok && gemmBinaryOpC(BinaryOp::Add, false, true, Tco, CO, CO_strategy, effCO, ldco, problem, strategy, state);
    }
    jmpi(1, labelCODone);

    mark(labelCORow);
    if (!problem.cOffsetHostScalar()) {
        status << "Applying row-wise C offset" << status_stream::endl;
        ok = ok && gemmBinaryOpC(BinaryOp::Add, true, false, Tco, CO, CO_strategy, effCO, ldco, problem, strategy, state);
    }

    if (doMatrix && !problem.cOffsetHostScalar()) {
        jmpi(1, labelCODone);

        mark(labelCOMatrix);
        status << "Applying matrix C offset" << status_stream::endl;
        ok = ok && gemmBinaryOpC(BinaryOp::Add, true, true, Tco, CO, CO_strategy, effCO, ldco, problem, strategy, state);
    }

    mark(labelCODone);

    if (!strategy.persistentLoop()) {
        state.ra.safeRelease(ldco);
        state.ra.safeRelease(effCO);
    }

    return ok;
}

#ifndef STANDALONE
template <HW hw>
void Generator<hw>::gemmLoadBinaryOpArgs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (hw < HW::XeHP) stub();

    std::vector<ngen::Subregister *> argList;
    argList.reserve(state.inputs.binaryOffsets.size() * 5);

    for (auto &r: state.inputs.binarySrcs)
        if (r.isValid()) argList.push_back(&r);
    for (auto &r: state.inputs.binaryOffsets)
        if (r.isValid()) argList.push_back(&r);
    for (auto &r: state.inputs.binaryLDs)
        if (r.isValid()) argList.push_back(&r);

    if (problem.batch == BatchMode::Strided) {
        for (auto &rs: state.inputs.binaryStrides)
            for (auto &r: rs)
                if (r.isValid()) argList.push_back(&r);
    }

    int loadBase = interface.getArgLoadBase().getBase();
    int nGRFs = 0;
    for (auto arg: argList) {
        int base = arg->getBase();
        if (base < loadBase) stub();
        nGRFs = std::max(nGRFs, base - loadBase + 1);
    }

    auto temp = state.ra.alloc();
    auto args = state.ra.alloc_range(nGRFs);

    if (state.r0_info.isARF() || state.r0_info.getBase() != 0) stub();
    loadargs(args, nGRFs, temp, false);

    int grfOffset = args.getBase() - loadBase;

    state.ra.release(args);

    for (auto arg: argList) {
        arg->setBase(arg->getBase() + grfOffset);
        state.ra.claim(*arg);
    }

    // CTI: Shift to the starting column + increase LDs due to interleaving
    if (strategy.cInterleave) {
        for (size_t i = 0; i < problem.postOps.len(); i++) {
            auto offset = state.inputs.binaryOffsets[i];
            if (!offset.isValid()) continue;

            bool row = problem.postOps.binaryRow[i];
            bool col = problem.postOps.binaryCol[i];
            if (!(row && col)) continue;

            auto ld = state.inputs.binaryLDs[i];
            emad(1, offset, offset, state.ctiShiftJ0, ld, strategy, state);
            mulConstant(1, ld, ld, strategy.cInterleaveChunk(problem.Tc_ext));
        }
        if (!strategy.persistentLoop()) state.ra.safeRelease(state.ctiShiftJ0);
    }

    state.ra.safeRelease(temp);
}

template <HW hw>
void Generator<hw>::gemmApplyPostOps(size_t poMin, size_t poMax, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (poMin >= poMax && !problem.binaryPostProcess()) return;

    Label lSkip;
    and_(1 | nz | state.flagAP, null.ud(), state.inputs.flags, FlagNonfinalKBlock);

    (void) gemmConvertC(problem.Ts, problem, strategy, state);

    jmpi(1 | state.flagAP, lSkip);

    // Binary preparations: load binary-related args + calculate starting addresses
    if (problem.hasBinaryPostOp() && state.effBinary.empty()) {
        auto &postOps = problem.postOps;
        size_t poCount = postOps.len();

        gemmLoadBinaryOpArgs(problem, strategy, state);

#define FOR_EACH_BINARY \
    for (size_t i = 0; i < poCount; i++) \
        if (postOps[i].is_binary())

        FOR_EACH_BINARY {
            const auto &ld = state.inputs.binaryLDs[i];
            auto T = problem.Tbinary[i];
            if (ld.isValid())
                emulConstant(1, ld, ld, T, strategy, state);
            emulConstant(1, state.inputs.binaryOffsets[i], state.inputs.binaryOffsets[i], T, strategy, state);
            if (problem.batch == BatchMode::Strided) for (int b = 0; b < problem.batchDims; b++) {
                const auto &stride = state.inputs.binaryStrides[i][b];
                if (stride.isValid())
                    emulConstant(1, stride, stride, T, strategy, state);
            }
        }

        for (int b = 0; b < problem.batchDims; b++) {
            FOR_EACH_BINARY {
                const auto &stride = state.inputs.binaryStrides[i][b];
                if (stride.isValid())
                    emul(1, stride, stride, state.batchID[b], strategy, state);
            }
        }

        for (int b = 0; b < problem.batchDims; b++) {
            FOR_EACH_BINARY {
                auto &offsetStride = state.inputs.binaryStrides[i][b];
                if (offsetStride.isValid())
                    eadd(1, state.inputs.binaryOffsets[i], state.inputs.binaryOffsets[i], offsetStride, strategy, state);
                state.ra.safeRelease(offsetStride);
            }
        }

        gemmOffsetABC(true, state.i0, state.j0, state.h0, Subregister(), Subregister(), problem, strategy, state, false, false, false, true);

        state.effBinary.resize(poCount);

        FOR_EACH_BINARY {
            if (strategy.binary[i].base.isStateless()) {
                state.effBinary[i] = state.inputs.binarySrcs[i];
                eadd(1, state.effBinary[i], state.inputs.binarySrcs[i], state.inputs.binaryOffsets[i], strategy, state);
                state.ra.safeRelease(state.inputs.binaryOffsets[i]);
            } else
                state.effBinary[i] = state.inputs.binaryOffsets[i];
        }
#undef FOR_EACH_BINARY
    }

    // Apply post-ops to all of C.
    int C_grfs[GRF::maxRegs()];
    int C_ngrf = state.C_regs[0].getLen();
    for (int r = 0; r < C_ngrf; r++)
        C_grfs[r] = state.C_regs[0][r].getBase();

    for (size_t i = poMin; i < poMax; i++) {
        auto &entry = problem.postOps[i];
        if(entry.is_binary()) {
            auto &ld = state.inputs.binaryLDs[i];
            auto &eff = state.effBinary[i];
            auto op = PostOpsProblem::toBinaryOp(entry);

            bool ok = gemmBinaryOpC(op, problem.postOps.binaryRow[i], problem.postOps.binaryCol[i],
                                    problem.Tbinary[i], problem.binary[i], strategy.binary[i],
                                    eff, ld, problem, strategy, state);
            if (!ok) stub();

            state.ra.safeRelease(ld);
            state.ra.safeRelease(eff);
        } else {

            if (state.Tacc != Type::f32) stub();

            problem.postOps.injectNonBinaryPostOps(entry, this, state.ra, C_grfs, C_ngrf);
        }
    }
    if(problem.postOps.cStochasticRound) {
        problem.postOps.injectStochasticRound(this, state.ra, C_grfs, C_ngrf, state.inputs.sroundSeed, problem.Tc_ext.ngen());
    }

    mark(lSkip);
}

template <HW hw>
void Generator<hw>::gemmApplyMXScale(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (!problem.hasCMXScale()) return;
    Label lSkip;
    and_(1 | nz | state.flagAP, null.ud(), state.inputs.flags, FlagNonfinalKBlock);

    jmpi(1 | state.flagAP, lSkip);

    int C_grfs[GRF::maxRegs()];
    int C_ngrf = state.C_regs[0].getLen();
    for (int r = 0; r < C_ngrf; r++)
        C_grfs[r] = state.C_regs[0][r].getBase();

    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    int n_elems = (unrollN / problem.cqGroupN) * (unrollM / problem.cqGroupM);
    int m_stride = state.C_scaleLayout[0].ld / 4;
    int n_regs = std::max(1, (n_elems * m_stride) / GRF::bytes(hw));
    auto tmpCScales = state.ra.alloc_range(n_regs);
    vector<MaskAssignment> masks;
    // Without a row mask, an M-remainder tile may store its out-of-range group
    // rows past the end of the scale row, corrupting the next column's
    // first scale entries. Mask rows using the M remainder converted to group units.
    bool maskM = state.remainders[LoopM].isValid()
            && (unrollM > problem.cqGroupM);
    if (maskM && !ngen::utils::is_zero_or_pow2(problem.cqGroupM)) stub();
    if (problem.cqGroupN > 1) stub("MX C scale groups along N are not supported");
    assignMasks(state.C_scaleLayout, maskM ? LoopM : LoopNone, LoopN, masks, strategy, state);
    if (maskM) {
        Subregister remInd[3];
        for (int i = 0; i < 3; i++) remInd[i] = state.remainders[i];
        auto qremM = state.ra.alloc_sub<uint32_t>();
        add(1 | sat, qremM, state.remainders[LoopM], problem.cqGroupM - 1);
        shr(1, qremM, qremM, ilog2(problem.cqGroupM));
        remInd[LoopM] = qremM;
        loadMasks(masks, remInd, strategy, state);
        state.ra.safeRelease(qremM);
    } else
        loadMasks(masks, state.remainders, strategy, state);

    problem.postOps.injectMXScale(this, state.ra, C_grfs, C_ngrf, tmpCScales.sub(hw, 0, ngen::DataType::ub), problem.Tc_ext.ngen(), unrollN);
    storeMatrix(tmpCScales, state.C_scaleLayout, state.C_scaleAddrs, strategy, state);
    state.ra.safeRelease(tmpCScales);
    safeReleaseMaskAssignments(masks, state);

    mark(lSkip);
}
#endif


// Calculate addresses of A/B sums in packed input data. Sums are stored at the end of each panel.
template <HW hw>
void Generator<hw>::gemmCalcABOffsetAddrs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool doA = (problem.bOffset == ABOffset::Load) && isPacked(problem.A.layout);
    bool doB = (problem.aOffset == ABOffset::Load) && isPacked(problem.B.layout);

    auto &effAs = state.effAs;
    auto &effBs = state.effBs;

    auto Tc = problem.Tc;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    if (doA && effAs.isInvalid()) effAs = state.ra.alloc_sub(state.effA.getType());
    if (doB && effBs.isInvalid()) effBs = state.ra.alloc_sub(state.effB.getType());

    if (doA) mulConstant(1, effAs.ud(), state.inputs.lda, unrollM);
    if (doB) mulConstant(1, effBs.ud(), state.inputs.ldb, unrollN);
    if (doA) add(1, effAs.ud(), effAs.ud(), -unrollM * Tc);
    if (doB) add(1, effBs.ud(), effBs.ud(), -unrollN * Tc);
    if (doA) eadd(1, effAs, effAs.ud(), state.effA, strategy, state);
    if (doB) eadd(1, effBs, effBs.ud(), state.effB, strategy, state);
}

// Load A/B sums from packed input data.
template <HW hw>
bool Generator<hw>::gemmLoadABOffset(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool doA = (problem.bOffset == ABOffset::Load);
    bool doB = (problem.aOffset == ABOffset::Load);
    if (!doA && !doB)
        return true;

    auto Tc = problem.Tc;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    MatrixAddressing As = problem.A, Bs = problem.B;
    As.crosspack = 1;
    Bs.crosspack = 1;
    As.tileR = As.tileC = 0;
    Bs.tileR = Bs.tileC = 0;

    if (!isPacked(As.layout)) {
        As.layout = MatrixLayout::N;
        As.setAlignment(Tc.size());
    }
    if (!isPacked(Bs.layout)) {
        Bs.layout = MatrixLayout::T;
        Bs.setAlignment(Tc.size());
    }

    MatrixAddressingStrategy As_strategy = strategy.A, Bs_strategy = strategy.B;
    As_strategy.accessType = AccessType::Block;
    Bs_strategy.accessType = AccessType::Block;
    As_strategy.tileR = As_strategy.tileC = 0;
    Bs_strategy.tileR = Bs_strategy.tileC = 0;
    As_strategy.dpasw = Bs_strategy.dpasw = false;

    if (doA) state.As_layout.clear();
    if (doB) state.Bs_layout.clear();

    bool ok = true;
    if (doA) ok = ok && (state.As_layout = RegisterLayout::tryCreate(hw, Tc, unrollM, 1, As, As_strategy));
    if (doB) ok = ok && (state.Bs_layout = RegisterLayout::tryCreate(hw, Tc, 1, unrollN, Bs, Bs_strategy));
    if (!ok) return false;

    if (doA) state.As_regs = state.ra.allocRange(state.As_layout.regs());
    if (doB) state.Bs_regs = state.ra.allocRange(state.Bs_layout.regs());

    vector<RegisterRange> As_addrs, Bs_addrs;
    if (doA) allocAddrRegs(hw, As_addrs, state.As_layout, state);
    if (doB) allocAddrRegs(hw, Bs_addrs, state.Bs_layout, state);

    if (state.effAs.isInvalid() && state.effBs.isInvalid())
        gemmCalcABOffsetAddrs(problem, strategy, state);

    if (doA) setupAddr(As_addrs, state.effAs, state.As_layout, Subregister(), strategy, state);
    if (doB) setupAddr(Bs_addrs, state.effBs, state.Bs_layout, Subregister(), strategy, state);

    if (doA) loadMatrix(state.As_regs, state.As_layout, As_addrs, strategy, state);
    if (doB) loadMatrix(state.Bs_regs, state.Bs_layout, Bs_addrs, strategy, state);

    state.ra.safeRelease(state.effAs);
    state.ra.safeRelease(state.effBs);
    safeReleaseRanges(As_addrs, state);
    safeReleaseRanges(Bs_addrs, state);

    return true;
}

// Rank-1 update of matrix C.
template <HW hw>
void Generator<hw>::gemmRank1UpdateC(const GRFMultirange &r, const GRFMultirange &c, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tacc = state.Tacc;
    auto ne = elementsPerGRF(hw, Tacc);
    auto globalCM = state.C_layout.colMajor();
    auto unrollX = strategy.unroll[globalCM ? LoopM : LoopN];
    auto unrollY = strategy.unroll[globalCM ? LoopN : LoopM];

    if (Tacc != problem.Tc) stub();

    for (int y = 0; y < unrollY; y++) {
        for (int x = 0; x < unrollX;) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;
            int nc;
            Subregister C = state.C_layout.find(i, j, state.C_regs[0], &nc);

            nc = std::min({nc, strategy.fmaSIMD, 2 * ne});

            auto offR = r[i / ne].sub(i % ne, Tacc.ngen());
            auto offC = c[j / ne].sub(j % ne, Tacc.ngen());

            globalCM ? emad(nc, C(1), C(1), offR(1), offC(0), strategy, state)
                     : emad(nc, C(1), C(1), offC(1), offR(0), strategy, state);

            x += nc;
        }
    }
}

// Apply contributions from A/B offsets to C matrix, using previously loaded/computed
// A row sums and B column sums.
template <HW hw>
void Generator<hw>::gemmApplyABOffset(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool aOffset = (problem.aOffset != ABOffset::None) && !problem.earlyDequantizeA() && !problem.quantized2DA();
    bool bOffset = (problem.bOffset != ABOffset::None) && !problem.earlyDequantizeB() && !problem.quantized2DB();
    if (!aOffset && !bOffset)
        return;

#ifdef STANDALONE
    if (problem.readCheck && problem.readCheckC)
        return;
#endif

    auto Tao = problem.Tao, Tbo = problem.Tbo, Tc = problem.Tc;

    bool aoVector = aOffset && (problem.aoPtrDims == 1);
    bool boVector = bOffset && (problem.boPtrDims == 1);
    // Host-scalar ao/bo are raw (non-negated), unlike vector/device-memory
    // scalars which are pre-negated on load.
    bool a_host_scalar = problem.aOffsetHostScalar();
    bool b_host_scalar = problem.bOffsetHostScalar();
    RegisterRange aoData, boData;

    auto temp = [&]() {
        if (!(aOffset && bOffset)) return Subregister{};

        auto ret = state.ra.alloc_sub(problem.Tc.ngen());
        if (!boVector)
            mul(1, ret, state.k,
                    (aoVector && b_host_scalar) ? -state.inputs.bo
                                                 : state.inputs.bo);
        else
            mov(1, ret, state.k);

        return ret;
    }();

    // Two steps: (O = all-1s matrix)
    //   1) C += A * O * bo
    //   2) C += (O * B + bo * k) * ao
    // Separate paths, depending on whether A/B offsets are vector or scalar.

    if (aoVector || boVector) {
        // Vector offset path.

        Subregister aoBase, boBase;
        if (aoVector) {
            aoBase = state.ra.alloc_sub<uint64_t>();
            eaddScaled(1, aoBase, state.inputs.aoPtr, state.i0, Tao, strategy, state);
        }
        if (boVector) {
            boBase = state.ra.alloc_sub<uint64_t>();
            eaddScaled(1, boBase, state.inputs.boPtr, state.j0, Tbo, strategy, state);
        }

        if (aoVector) aoData = loadVector(Tao, Tc, aoBase, strategy.unroll[LoopM], state.remainders[LoopM], strategy, state);
        if (boVector) boData = loadVector(Tbo, Tc, boBase, strategy.unroll[LoopN], state.remainders[LoopN], strategy, state);

        state.ra.safeRelease(aoBase);
        state.ra.safeRelease(boBase);

        if (aoVector) map(hw, Tc, aoData, aoData, strategy, [&](int ne, RegData r, RegData) {
            mov(ne, r, -r);
        });
        if (boVector) map(hw, Tc, boData, boData, strategy, [&](int ne, RegData r, RegData) {
            mov(ne, r, -r);
        });

        if (aoVector && !state.Bs_layout.hasFullCrosspack(1)) stub();
        if (boVector && !state.As_layout.hasFullCrosspack(1)) stub();

        if (bOffset) {
            boVector ? gemmRank1UpdateC(state.As_regs, boData, problem, strategy, state)
                     : gemmVectorBinaryOpC(b_host_scalar ? BinaryOp::Sub : BinaryOp::Add, false,
                                           state.As_regs, state.inputs.bo,
                                           problem, strategy, state, problem.Tc, state.As_layout);
        }

        if (aOffset && bOffset) for (int r = 0; r < state.Bs_regs.getLen(); r++) {
            auto ne = elementsPerGRF(hw, Tc);
            auto Bs = state.Bs_regs[r].retype(Tc.ngen());
            boVector ? emad(ne, Bs, Bs, boData[r].retype(Tc.ngen()), temp, strategy, state)
                     : add(ne, Bs, Bs, temp);
        };

        state.ra.safeRelease(temp);

        if (aOffset) {
            aoVector ? gemmRank1UpdateC(aoData, state.Bs_regs, problem, strategy, state)
                     : gemmVectorBinaryOpC(a_host_scalar ? BinaryOp::Sub : BinaryOp::Add, true,
                                           state.Bs_regs, state.inputs.ao,
                                           problem, strategy, state, problem.Tc, state.Bs_layout);
        }
    } else {
        // Scalar offset path.
        // TODO: combine C adds into add3 on XeHP+.
        if (aOffset && bOffset) {
            mul(1, temp, temp, b_host_scalar ? -state.inputs.ao : state.inputs.ao);
            map(hw, Tc, state.Bs_regs, state.Bs_layout, strategy, [&](int ne, RegData r) {
                mad(ne, r, temp, r, state.inputs.ao);
            });
        }
        state.ra.safeRelease(temp);

        auto As_scale = state.inputs.bo;
        auto Bs_scale = (bOffset) ? Subregister() : state.inputs.ao;

        if (bOffset) gemmVectorBinaryOpC(b_host_scalar ? BinaryOp::Sub : BinaryOp::Add, false, state.As_regs, As_scale, problem, strategy, state, problem.Tc, state.As_layout);
        if (aOffset) gemmVectorBinaryOpC(a_host_scalar ? BinaryOp::Sub : BinaryOp::Add, true,  state.Bs_regs, Bs_scale, problem, strategy, state, problem.Tc, state.Bs_layout);
    }

    state.ra.safeRelease(aoData);
    state.ra.safeRelease(boData);
    safeReleaseRanges(state.As_regs, state);
    safeReleaseRanges(state.Bs_regs, state);
    if (!strategy.persistentLoop()) {
        state.ra.safeRelease(state.inputs.ao);
        state.ra.safeRelease(state.inputs.bo);
    }
    state.As_layout.clear();
    state.Bs_layout.clear();
}

GEMMSTONE_NAMESPACE_END
