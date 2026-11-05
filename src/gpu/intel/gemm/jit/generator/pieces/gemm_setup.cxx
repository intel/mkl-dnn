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
#include "compute_utils.hpp"
#include "cooperative_split.hpp"
#include "gemmstone/generator.hpp"
#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "ngen_object_helpers.hpp"
#include "state_utils.hpp"
#include "token_alloc_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


// Generate code for checking whether 32-bit address arithmetic can be used inside k loop.
// Assumes leading dimensions have not been shifted yet.
template <HW hw>
void Generator<hw>::gemmCheck32(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    if (!strategy.checkAdd32)
        return;

    bool checkA = (strategy.A.base.getModel() == ModelA64);
    bool checkB = (strategy.B.base.getModel() == ModelA64);

    // Quantization buffers are also advanced by the k loop and must be bounded too.
    bool checkAO = problem.aOffset2D()      && state.inputs.aoPtr.isValid()     && state.inputs.ldao.isValid()     && (strategy.AO.base.getModel()      == ModelA64);
    bool checkBO = problem.bOffset2D()      && state.inputs.boPtr.isValid()     && state.inputs.ldbo.isValid()     && (strategy.BO.base.getModel()      == ModelA64);
    bool checkAS = problem.aScale2D()       && state.inputs.aScalePtr.isValid() && state.inputs.ldaScale.isValid() && (strategy.A_scale.base.getModel() == ModelA64);
    bool checkBS = problem.bScale2D()       && state.inputs.bScalePtr.isValid() && state.inputs.ldbScale.isValid() && (strategy.B_scale.base.getModel() == ModelA64);
    bool checkAG = problem.needsAGroupSums() && state.inputs.agPtr.isValid()    && state.inputs.ldag.isValid()     && (strategy.Ag.base.getModel()      == ModelA64);
    bool checkBG = problem.needsBGroupSums() && state.inputs.bgPtr.isValid()    && state.inputs.ldbg.isValid()     && (strategy.Bg.base.getModel()      == ModelA64);

    if (!checkA && !checkB && !checkAO && !checkBO && !checkAS && !checkBS && !checkAG && !checkBG)
        return;

    bool emulate = strategy.emulate.emulate64_mul;
    auto &m = state.inputs.m;
    auto &n = state.inputs.n;
    auto &k = state.fullK.isValid() ? state.fullK : state.inputs.k;
    auto &lda = state.inputs.lda;
    auto &ldb = state.inputs.ldb;
    auto temp1GRF = state.ra.alloc();
    auto temp2GRF = state.ra.alloc();
    auto temp1 = temp1GRF.uq(0);
    auto temp1Hi = temp1GRF.ud(emulate ? 0 : 1);
    auto temp2 = temp2GRF.ud(0);
    auto temp3 = temp2GRF.ud(4);
    auto flag = state.raVFlag.alloc();

    auto mulHigh = [&](Subregister dst, Subregister src0, Subregister src1) {
        if (emulate)
            emul32High(1, dst.ud(), src0, src1);
        else
            mul(1, dst, src0, src1);
    };

    if (checkA) {
        state.offsetA.isValid() ? add(1, temp2, state.effA.ud(), state.offsetA.ud())
                                : mov(1, temp2, state.effA.ud());
        switch (problem.A.layout) {                 // Conservatively estimate upper bound for size of A.
            case MatrixLayout::N:  mulHigh(temp1, lda, k); break;
            case MatrixLayout::T:  mulHigh(temp1, lda, m); break;
            case MatrixLayout::Pc: {
                if (strategy.fixedWG(problem))
                    add(1, temp3, m, uint16_t(strategy.wg[LoopM] * strategy.unroll[LoopM] - 1));
                else
                    emad(1, temp3, m, state.inputs.localSizeM, strategy.unroll[LoopM], strategy, state);
                mulHigh(temp1, lda, temp3); break;
            }
            default: stub();
        }
        add(1 | ov | flag, temp2, acc0.ud(0), temp2);
        cmp(1 | ~flag | ne | flag, temp1Hi, uint16_t(0));
    }

    if (checkB) {
        state.offsetB.isValid() ? add(1, temp2, state.effB.ud(), state.offsetB.ud())
                                : mov(1, temp2, state.effB.ud());
        switch (problem.B.layout) {
            case MatrixLayout::T:  mulHigh(temp1, ldb, k); break;
            case MatrixLayout::N:  mulHigh(temp1, ldb, n); break;
            case MatrixLayout::Pr: {
                if (strategy.fixedWG(problem))
                    add(1, temp3, n, uint16_t(strategy.wg[LoopN] * strategy.unroll[LoopN] - 1));
                else
                    emad(1, temp3, n, state.inputs.localSizeN, strategy.unroll[LoopN], strategy, state);
                mulHigh(temp1, ldb, temp3); break;
            }
            default: stub();
        }
        InstructionModifier mod = 1;
        if (checkA)
            mod |= ~flag;
        add(mod | ov | flag, temp2, acc0.ud(0), temp2);
        cmp(1 | ~flag | ne | flag, temp1Hi, uint16_t(0));
    }

    // Conservatively estimate upper bound for each buffer's size as ld*k.
    bool havePrev = checkA || checkB;
    auto checkQuantBuffer = [&](const ngen::Subregister &ptr, const ngen::Subregister &ld, const ngen::Subregister &offset) {
        offset.isValid() ? add(1, temp2, ptr.ud(), offset.ud())
                         : mov(1, temp2, ptr.ud());
        mulHigh(temp1, ld, k);
        InstructionModifier mod = 1;
        if (havePrev)
            mod |= ~flag;
        add(mod | ov | flag, temp2, acc0.ud(0), temp2);
        cmp(1 | ~flag | ne | flag, temp1Hi, uint16_t(0));
        havePrev = true;
    };

    if (checkAO) checkQuantBuffer(state.inputs.aoPtr, state.inputs.ldao, state.inputs.offsetAO);
    if (checkBO) checkQuantBuffer(state.inputs.boPtr, state.inputs.ldbo, state.inputs.offsetBO);
    if (checkAS) checkQuantBuffer(state.inputs.aScalePtr, state.inputs.ldaScale, state.inputs.offsetAScale);
    if (checkBS) checkQuantBuffer(state.inputs.bScalePtr, state.inputs.ldbScale, state.inputs.offsetBScale);
    if (checkAG) checkQuantBuffer(state.inputs.agPtr, state.inputs.ldag, state.inputs.offsetAg);
    if (checkBG) checkQuantBuffer(state.inputs.bgPtr, state.inputs.ldbg, state.inputs.offsetBg);

    state.add64 = state.ra.alloc_sub<uint16_t>();
    and_(1, state.add64, flag, 1u);
    state.raVFlag.safeRelease(flag);

    state.ra.safeRelease(temp1GRF); temp1 = invalid;
    state.ra.safeRelease(temp2GRF); temp2 = invalid; temp3 = invalid;
}

// Calculate A offset for SLM copies or cooperative prefetches for this local ID.
template <HW hw>
void Generator<hw>::gemmCalcWorkshareAOffset(Subregister &off, Subregister &offR, Subregister &offC,
                                             const MatrixAddressing &A, const MatrixAddressingStrategy &A_strategy, int ma, int ka,
                                             const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool splitM = (state.effCoopA == CoopSplit::MN);
    bool splitLinear = (state.effCoopA == CoopSplit::Linear);
    bool splitKFull = (state.effCoopA == CoopSplit::FullK);

    auto lid = splitKFull ? gemmMNLinearID(strategy, state)
                          : state.lidN;

    if (A_strategy.address2D) {
        if (splitLinear) stub();
        if (splitM) {
            offR = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            mulConstant(1, offR, lid, ma);
        } else {
            offC = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            mulConstant(1, offC, lid, ka);
        }
    } else {
        auto Ta_ext = problem.Ta_ext;
        auto Toff = A_strategy.base.isA64() ? DataType::uq : DataType::ud;
        off = state.ra.alloc_sub(Toff, getHint(HintType::TempComp0, strategy));

        switch (A.layout) {
            case MatrixLayout::Pc:
                emulConstant(1, off, lid, ma * ka * Ta_ext, strategy, state);
                break;
            case MatrixLayout::T:
                if (splitLinear) stub();
                if (splitM) {
                    emul(1, off, state.inputs.lda, lid, strategy, state);
                    emulConstant(1, off, off, ma, strategy, state);
                } else
                    emulConstant(1, off, lid, ka * Ta_ext, strategy, state);
                break;
            case MatrixLayout::N:
                if (splitLinear) stub();
                if (splitM)
                    emulConstant(1, off, lid, ma * Ta_ext, strategy, state);
                else {
                    emul(1, off, state.inputs.lda, lid, strategy, state);
                    emulConstant(1, off, off, ka, strategy, state);
                }
                break;
            default: stub();
        }
    }

    if (splitKFull) state.ra.safeRelease(lid);
}

// Calculate B offset for SLM copies or cooperative prefetches for this local ID.
template <HW hw>
void Generator<hw>::gemmCalcWorkshareBOffset(Subregister &off, Subregister &offR, Subregister &offC,
                                             const MatrixAddressing &B, const MatrixAddressingStrategy &B_strategy, int kb, int nb,
                                             const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool splitN = (state.effCoopB == CoopSplit::MN);
    bool splitLinear = (state.effCoopB == CoopSplit::Linear);
    bool splitKFull = (state.effCoopB == CoopSplit::FullK);

    auto lid = splitKFull ? gemmMNLinearID(strategy, state)
                          : state.lidM;

    if (B_strategy.address2D) {
        if (splitLinear) stub();
        if (splitN) {
            offC = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            mulConstant(1, offC, lid, nb);
        } else {
            offR = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            mulConstant(1, offR, lid, kb);
        }
    } else {
        auto Tb_ext = problem.Tb_ext;
        auto Toff = B_strategy.base.isA64() ? DataType::uq : DataType::ud;
        off = state.ra.alloc_sub(Toff, getHint(HintType::TempComp0, strategy));

        switch (B.layout) {
            case MatrixLayout::Pr:
                emulConstant(1, off, lid, nb * kb * Tb_ext, strategy, state);
                break;
            case MatrixLayout::N:
                if (splitLinear) stub();
                if (splitN) {
                    emul(1, off, state.inputs.ldb, lid, strategy, state);
                    emulConstant(1, off, off, nb, strategy, state);
                } else
                    emulConstant(1, off, lid, kb * Tb_ext, strategy, state);
                break;
            case MatrixLayout::T:
                if (splitLinear) stub();
                if (splitN)
                    emulConstant(1, off, lid, nb * Tb_ext, strategy, state);
                else {
                    emul(1, off, state.inputs.ldb, lid, strategy, state);
                    emulConstant(1, off, off, kb, strategy, state);
                }
                break;
            default: stub();
        }
    }

    if (splitKFull) state.ra.safeRelease(lid);
}

// Calculate m,n joint linear local ID.
template <HW hw>
Subregister Generator<hw>::gemmMNLinearID(const GEMMStrategy &strategy, GEMMState &state)
{
    auto lid = state.ra.alloc_sub<uint16_t>();
    if (strategy.loopOrder[0] == LoopM)
        emad(1, lid, state.lidM, state.lidN, strategy.wg[LoopM], strategy, state);
    else
        emad(1, lid, state.lidN, state.lidM, strategy.wg[LoopN], strategy, state);
    return lid;
}

// TODO: move me
template <HW hw>
CoopSplit Generator<hw>::effCoopSplitA(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (isPacked(problem.A.layout))
        return CoopSplit::Linear;
    else if (!isRegisterColMajor(problem.Ta_ext, problem.A, strategy.A)
                && (strategy.unroll[LoopM] % strategy.wg[LoopN] == 0)
                && !isBlock2D(strategy.A.accessType)
                && strategy.coopA != CoopSplit::FullK)
        return CoopSplit::MN;
    else
        return strategy.coopA;
}

template <HW hw>
CoopSplit Generator<hw>::effCoopSplitB(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    if (isPacked(problem.B.layout))
        return CoopSplit::Linear;
    else if (isRegisterColMajor(problem.Tb_ext, problem.B, strategy.B)
                && (strategy.unroll[LoopN] % strategy.wg[LoopM] == 0)
                && !isBlock2D(strategy.B.accessType)
                && strategy.coopB != CoopSplit::FullK)
        return CoopSplit::MN;
    else
        return strategy.coopB;
}

// Offset A pointer in m dimension by a variable value.
template <HW hw>
void Generator<hw>::gemmOffsetAm(const Subregister &i, const Subregister &effA, const MatrixAddressing &globalA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta_ext = problem.Ta_ext;
    switch (globalA.layout) {
        case MatrixLayout::N:  eaddScaled(1, effA, effA, i, Ta_ext, strategy, state); break;
        case MatrixLayout::Pc:
        case MatrixLayout::T:  emad(1, effA, effA, state.inputs.lda, i, strategy, state); break;
        default: stub();
    }
}

// Offset A pointer in k dimension by a constant value.
template <HW hw>
void Generator<hw>::gemmOffsetAk(int h, const Subregister &effA, const MatrixAddressing &globalA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta_ext = problem.Ta_ext;
    if (h) switch (globalA.layout) {
        case MatrixLayout::N:  emad(1, effA, effA, state.inputs.lda, Immediate::w(h), strategy, state); break;
        case MatrixLayout::T:  eadd(1, effA, effA, h * Ta_ext, strategy, state); break;
        case MatrixLayout::Pc: eadd(1, effA, effA, h * globalA.packSize * Ta_ext, strategy, state); break;
        default: stub();
    }
}

// Offset A pointer in k dimension by a variable value.
template <HW hw>
void Generator<hw>::gemmOffsetAk(const Subregister &h, const Subregister &effA, const MatrixAddressing &globalA, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta_ext = problem.Ta_ext;
    switch (globalA.layout) {
        case MatrixLayout::N:  emad(1, effA, effA, state.inputs.lda, h, strategy, state); break;
        case MatrixLayout::T:  eaddScaled(1, effA, effA, h, Ta_ext, strategy, state); break;
        case MatrixLayout::Pc: emad(1, effA, effA, h, globalA.packSize * Ta_ext, strategy, state); break;
        default: stub();
    }
}

// Offset B pointer in k dimension by a constant value.
template <HW hw>
void Generator<hw>::gemmOffsetBk(int h, const Subregister &effB, const MatrixAddressing &globalB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tb_ext = problem.Tb_ext;
    if (h) switch (globalB.layout) {
        case MatrixLayout::N:  eadd(1, effB, effB, h * Tb_ext, strategy, state); break;
        case MatrixLayout::Pr: eadd(1, effB, effB, h * globalB.packSize * Tb_ext, strategy, state); break;
        case MatrixLayout::T:  emad(1, effB, effB, state.inputs.ldb, Immediate::w(h), strategy, state); break;
        default: stub();
    }
}

// Offset B pointer in k dimension by a variable value.
template <HW hw>
void Generator<hw>::gemmOffsetBk(const Subregister &h, const Subregister &effB, const MatrixAddressing &globalB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tb_ext = problem.Tb_ext;
    switch (globalB.layout) {
        case MatrixLayout::T:  emad(1, effB, effB, state.inputs.ldb, h, strategy, state); break;
        case MatrixLayout::N:  eaddScaled(1, effB, effB, h, Tb_ext, strategy, state); break;
        case MatrixLayout::Pr: emad(1, effB, effB, h, globalB.packSize * Tb_ext, strategy, state); break;
        default: stub();
    }
}

// Offset B pointer in n dimension by a variable value.
template <HW hw>
void Generator<hw>::gemmOffsetBn(const Subregister &j, const Subregister &effB, const MatrixAddressing &globalB, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tb_ext = problem.Tb_ext;
    switch (globalB.layout) {
        case MatrixLayout::Pr:
        case MatrixLayout::N:  emad(1, effB, effB, state.inputs.ldb, j, strategy, state); break;
        case MatrixLayout::T:  eaddScaled(1, effB, effB, j, Tb_ext, strategy, state); break;
        default: stub();
    }
}

// Adjust A, B, C to start at (i0, j0).
//  initial is true to adjust offset_{A,B,C}, false to adjust A,B,C pointers.
template <HW hw>
void Generator<hw>::gemmOffsetABC(bool initial, Subregister i0, Subregister j0, Subregister h0, Subregister i0p, Subregister j0p,
                                  const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                  bool doA, bool doB, bool doC, bool doBinary)
{
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext, Tc_ext = problem.Tc_ext, Tco = problem.Tco;
    auto &offsetA  = initial ? state.offsetA    : state.effA;
    auto &offsetB  = initial ? state.offsetB    : state.effB;
    auto &offsetC0 = initial ? state.offsetC[0] : state.effC[0];
    auto &offsetAp = initial ? state.offsetAp   : state.effAp;
    auto &offsetBp = initial ? state.offsetBp   : state.effBp;
    auto &offsetCp = initial ? state.offsetCp   : state.effCp;
    auto  offsetCO = initial ? state.offsetCO   : state.effCO;
    bool doCO = doC && (problem.cOffset != COffset::None) && !problem.cOffsetHostScalar();
    bool doAp = doA, doBp = doB;

    Subregister tempQ0 = state.ra.alloc_sub<int64_t>(getHint(HintType::TempComp0, strategy));
    Subregister tempQ1 = state.ra.alloc_sub<int64_t>(getHint(HintType::TempComp1, strategy));

    bool a2D = strategy.A.address2D;
    bool b2D = strategy.B.address2D;
    bool c2D = strategy.C.address2D;
    bool ap2D = strategy.prefetchA ? strategy.A_prefetch.address2D : a2D;
    bool bp2D = strategy.prefetchB ? strategy.B_prefetch.address2D : b2D;
    bool cp2D = strategy.prefetchC ? strategy.C_prefetch.address2D : c2D;

    if (a2D)         doA  = false;
    if (ap2D)        doAp = false;
    if (b2D)         doB  = false;
    if (bp2D)        doBp = false;
    if (c2D && cp2D) doC  = false;

    if (doA && doAp && i0 == i0p) doAp = false;
    if (doB && doBp && j0 == j0p) doBp = false;

    bool splitAp = (a2D != ap2D) || (doA && doAp);
    bool splitBp = (b2D != bp2D) || (doB && doBp);

    if ((doA || doAp) && splitAp && (offsetAp.isInvalid() || offsetA == offsetAp)) {
        offsetAp = state.ra.alloc_sub(offsetA.getType(), getHint(HintType::LongTerm, strategy));
        emov(1, offsetAp, offsetA, strategy, state);
    }
    if ((doB || doBp) && splitBp && (offsetBp.isInvalid() || offsetB == offsetBp)) {
        offsetBp = state.ra.alloc_sub(offsetB.getType(), getHint(HintType::LongTerm, strategy));
        emov(1, offsetBp, offsetB, strategy, state);
    }
    if (doC && (c2D != cp2D)) {
        if (offsetCp.isInvalid() || offsetC0 == offsetCp) {
            offsetCp = state.ra.alloc_sub(offsetC0.getType(), getHint(HintType::LongTerm, strategy));
            emov(1, offsetCp, offsetC0, strategy, state);
        } else if (c2D && !cp2D)
            std::swap(offsetC0, offsetCp);
    }

    // To do: interleave code.
    // A += i0 (N) i0 * lda (T, Pc)
    // B += j0 * ldb (N, Pr) j0 (T)
    // C += i0 + j0 * ldc (N, Pr) j0 + i0 * ldc (T, Pc)
    // CO += i0 (row offsets) j0 (col offsets)
    auto doAOffset = [&](Subregister offsetAx, Subregister i0x) {
        if (problem.A.layout == MatrixLayout::Nontranspose)
            eaddScaled(1, offsetAx, offsetAx, i0x, Ta_ext, strategy, state);
        else {
            emul(1, tempQ1, i0x, state.inputs.lda, strategy, state);
            eadd(1, offsetAx, offsetAx, tempQ1.reinterpret(0, offsetAx.getType()), strategy, state);
        }
    };

    auto doBOffset = [&](Subregister offsetBx, Subregister j0x) {
        if (problem.B.layout == MatrixLayout::Transpose)
            eaddScaled(1, offsetBx, offsetBx, j0x, Tb_ext, strategy, state);
        else {
            emul(1, tempQ0, j0x, state.inputs.ldb, strategy, state);
            eadd(1, offsetBx, offsetBx, tempQ0.reinterpret(0, offsetBx.getType()), strategy, state);
        }
    };

    if (doA  && i0.isValid())  doAOffset(offsetA,  i0);
    if (doAp && i0p.isValid()) doAOffset(offsetAp, i0p);
    if (doB  && j0.isValid())  doBOffset(offsetB,  j0);
    if (doBp && j0p.isValid()) doBOffset(offsetBp, j0p);

    FlagRegister flagCOR, flagCOC;
    if (doCO) {
        flagCOR = state.raVFlag.alloc();
        flagCOC = state.raVFlag.alloc();
        and_(1 | nz | flagCOC, null.ud(), state.inputs.flags, FlagCOColumn);
        and_(1 | nz | flagCOR, null.ud(), state.inputs.flags, FlagCORow);
    }
    if (doC) {
        for (int q = 0; q < state.C_count; q++) {
            auto offsetC = initial ? state.offsetC[q] : state.effC[q];

            Subregister x, y;
            int xstride = Tc_ext.bits() >> 3;
            switch (problem.C.layout) {
                case MatrixLayout::Pr:  xstride = strategy.unroll[LoopN] * Tc_ext;   /* fall through */
                case MatrixLayout::N:   x = i0; y = j0;             break;
                case MatrixLayout::Pc:  xstride = strategy.unroll[LoopM] * Tc_ext;   /* fall through */
                case MatrixLayout::T:   x = j0; y = i0;             break;
            }
            if (Tc_ext.isSubByte()) {
                eshr(1, tempQ0, x, Tc_ext.logPerByte(), strategy, state);
                eadd(1, offsetC, offsetC, tempQ0, strategy, state);
            } else
                emad(1, offsetC, offsetC, x, xstride, strategy, state);
            emul(1, tempQ0, y, state.inputs.ldc[q], strategy, state);
            eadd(1, offsetC, offsetC, tempQ0.reinterpret(0, offsetC.getType()), strategy, state);       // Gen12: Use add3.
        }
    }
    if (doCO) {
        Label lNoMatrixOffset, lDone;
        if (problem.allowMatrixOffset()) {
            auto x0 = isColMajor(problem.CO.layout) ? i0 : j0;
            auto y0 = isColMajor(problem.CO.layout) ? j0 : i0;
            jmpi(1 | ~flagCOC, lNoMatrixOffset);
            jmpi(1 | ~flagCOR, lNoMatrixOffset);
            eaddScaled(1, offsetCO, offsetCO, x0, Tco, strategy, state);
            emul(1, tempQ0, y0, state.inputs.ldco, strategy, state);
            eadd(1, offsetCO, offsetCO, tempQ0.reinterpret(0, offsetCO.getType()), strategy, state);
            jmpi(1, lDone);
            mark(lNoMatrixOffset);
        }
        eaddScaled(1 | flagCOC, offsetCO, offsetCO, j0, Tco, strategy, state);
        eaddScaled(1 | flagCOR, offsetCO, offsetCO, i0, Tco, strategy, state);
        state.raVFlag.safeRelease(flagCOR);
        state.raVFlag.safeRelease(flagCOC);
        if (problem.allowMatrixOffset())
            mark(lDone);
    }
    if (doBinary) for (size_t i = 0; i < problem.postOps.len(); i++) {
        if (!problem.postOps[i].is_binary()) continue;
        bool row = problem.postOps.binaryRow[i],
             col = problem.postOps.binaryCol[i];
        auto T = problem.Tbinary[i];
        auto &ld = state.inputs.binaryLDs[i];
        auto offset = initial ? state.inputs.binaryOffsets[i] : state.effBinary[i];
        if (row && col) {
            auto x0 = isColMajor(problem.binary[i].layout) ? i0 : j0;
            auto y0 = isColMajor(problem.binary[i].layout) ? j0 : i0;
            eaddScaled(1, offset, offset, x0, T, strategy, state);
            emul(1, tempQ0, y0, ld, strategy, state);
            eadd(1, offset, offset, tempQ0.reinterpret(0, offset.getType()), strategy, state);
        } else if (row)
            eaddScaled(1, offset, offset, i0, T, strategy, state);
        else if (col)
            eaddScaled(1, offset, offset, j0, T, strategy, state);
    }
    if (doC && problem.sumA) eaddScaled(1, offsetCO, offsetCO, i0, Tco, strategy, state);
    if (doC && problem.sumB) eaddScaled(1, offsetCO, offsetCO, j0, Tco, strategy, state);

    // When k blocking (or certain triangular source kernels)
    //   A += h0 * lda (N) h0 (T) h0 * mb (Pc)
    //   B += h0 (N) h0 * ldb (T) h0 * nb (Pr)
    if (!h0.isInvalid()) {
        if (doA || doAp) switch (problem.A.layout) {
            case MatrixLayout::Nontranspose:
                emul(1, tempQ1, h0, state.inputs.lda, strategy, state);
                if (doA)  eadd(1, offsetA,  offsetA,  tempQ1.reinterpret(0, offsetA.getType()), strategy, state);
                if (doAp) eadd(1, offsetAp, offsetAp, tempQ1.reinterpret(0, offsetAp.getType()), strategy, state);
                break;
            case MatrixLayout::Transpose:
                if (doA)  eaddScaled(1, offsetA,  offsetA,  h0, Ta_ext, strategy, state);
                if (doAp) eaddScaled(1, offsetAp, offsetAp, h0, Ta_ext, strategy, state);
                break;
            case MatrixLayout::PackedColumns:
                if (doA)  emad(1, offsetA,  offsetA,  h0, strategy.unroll[LoopM] * Ta_ext, strategy, state);
                if (doAp) emad(1, offsetAp, offsetAp, h0, strategy.unroll[LoopM] * Ta_ext, strategy, state);
                break;
            default: stub();
        }
        if (doB || doBp) switch (problem.B.layout) {
            case MatrixLayout::Nontranspose:
                if (doB)  eaddScaled(1, offsetB,  offsetB,  h0, Tb_ext, strategy, state);
                if (doBp) eaddScaled(1, offsetBp, offsetBp, h0, Tb_ext, strategy, state);
                break;
            case MatrixLayout::Transpose:
                emul(1, tempQ0, h0, state.inputs.ldb, strategy, state);
                if (doB)  eadd(1, offsetB,  offsetB,  tempQ0.reinterpret(0, offsetB.getType()), strategy, state);
                if (doBp) eadd(1, offsetBp, offsetBp, tempQ0.reinterpret(0, offsetBp.getType()), strategy, state);
                break;
            case MatrixLayout::PackedRows:
                if (doB)  emad(1, offsetB,  offsetB,  h0, strategy.unroll[LoopN] * Tb_ext, strategy, state);
                if (doBp) emad(1, offsetBp, offsetBp, h0, strategy.unroll[LoopN] * Tb_ext, strategy, state);
                break;
            default: stub();
        }
    }

    state.ra.safeRelease(tempQ0);
    state.ra.safeRelease(tempQ1);

    if (doC && c2D && !cp2D) std::swap(offsetC0, offsetCp);
}

template <ngen::HW hw>
void Generator<hw>::gemmOffsetBatchABC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{

    // Strided batch support.
    if (problem.batch == BatchMode::Strided) {
        Subregister bOffsetA[4], bOffsetB[4], bOffsetC[4];
        Subregister bOffsetAs[4], bOffsetBs[4], bOffsetCs[4];
        Subregister bOffsetAo[4], bOffsetBo[4];
        Subregister bOffsetAg[4], bOffsetBg[4];

        for (int b = 0; b < problem.batchDims; b++) {
            bOffsetA[b] = state.inputs.strideA[b];
            bOffsetB[b] = state.inputs.strideB[b];
            bOffsetC[b] = state.inputs.strideC[b];
            if(problem.hasAScalePtr()){
                bOffsetAs[b] = state.inputs.strideScaleA[b];
            }
            if(problem.hasBScalePtr()){
                bOffsetBs[b] = state.inputs.strideScaleB[b];
            }
            if(problem.hasCMXScale()){
                bOffsetCs[b] = state.inputs.strideScaleC[b];
            }
            if(problem.hasAOffsetPtr()){
                bOffsetAo[b] = state.inputs.strideOffsetA[b];
            }
            if(problem.hasBOffsetPtr()){
                bOffsetBo[b] = state.inputs.strideOffsetB[b];
            }
            if(problem.needsAGroupSums()){
                bOffsetAg[b] = state.inputs.strideGroupSumsA[b];
            }
            if(problem.needsBGroupSums()){
                bOffsetBg[b] = state.inputs.strideGroupSumsB[b];
            }
            if (strategy.A.base.isStateless()) bOffsetA[b] = state.ra.alloc_sub<uint64_t>();
            if (strategy.B.base.isStateless()) bOffsetB[b] = state.ra.alloc_sub<uint64_t>();
            if (strategy.C.base.isStateless()) bOffsetC[b] = state.ra.alloc_sub<uint64_t>();
        }

        for (int b = 0; b < problem.batchDims; b++) {
            emul(1, bOffsetA[b], state.inputs.strideA[b], state.batchID[b], strategy, state);
            emul(1, bOffsetB[b], state.inputs.strideB[b], state.batchID[b], strategy, state);
            emul(1, bOffsetC[b], state.inputs.strideC[b], state.batchID[b], strategy, state);
            if(problem.hasAScalePtr()){
                    emul(1, bOffsetAs[b], state.inputs.strideScaleA[b], state.batchID[b], strategy, state);
            }
            if(problem.hasBScalePtr()){
                    emul(1, bOffsetBs[b], state.inputs.strideScaleB[b], state.batchID[b], strategy, state);
            }
            if(problem.hasCMXScale()){
                    emul(1, bOffsetCs[b], state.inputs.strideScaleC[b], state.batchID[b], strategy, state);
            }
            if(problem.hasAOffsetPtr()){
                    emul(1, bOffsetAo[b], state.inputs.strideOffsetA[b], state.batchID[b], strategy, state);
            }
            if(problem.hasBOffsetPtr()){
                    emul(1, bOffsetBo[b], state.inputs.strideOffsetB[b], state.batchID[b], strategy, state);
            }
            if(problem.needsAGroupSums()){
                    emul(1, bOffsetAg[b], state.inputs.strideGroupSumsA[b], state.batchID[b], strategy, state);
            }
            if(problem.needsBGroupSums()){
                    emul(1, bOffsetBg[b], state.inputs.strideGroupSumsB[b], state.batchID[b], strategy, state);
            }
        }

        if(problem.hasAScalePtr() && state.offsetAs.isInvalid()){
            state.offsetAs = state.ra.alloc_sub(state.offsetA.getType());
            emov(1, state.offsetAs, 0, strategy, state);
        }
        if(problem.hasBScalePtr() && state.offsetBs.isInvalid()){
            state.offsetBs = state.ra.alloc_sub(state.offsetB.getType());
            emov(1, state.offsetBs, 0, strategy, state);
        }
        if(problem.hasCMXScale() && state.offsetCs.isInvalid()){
            state.offsetCs = state.ra.alloc_sub(ngen::DataType::q);
            emov(1, state.offsetCs, 0, strategy, state);
        }
        if(problem.hasAOffsetPtr() && state.offsetAo.isInvalid()){
            state.offsetAo = state.ra.alloc_sub(state.offsetA.getType());
            emov(1, state.offsetAo, 0, strategy, state);
        }
        if(problem.hasBOffsetPtr() && state.offsetBo.isInvalid()){
            state.offsetBo = state.ra.alloc_sub(state.offsetB.getType());
            emov(1, state.offsetBo, 0, strategy, state);
        }
        if(problem.needsAGroupSums() && state.inputs.offsetAg.isInvalid()){
            state.inputs.offsetAg = state.ra.alloc_sub(state.offsetA.getType());
            emov(1, state.inputs.offsetAg, 0, strategy, state);
        }
        if(problem.needsBGroupSums() && state.inputs.offsetBg.isInvalid()){
            state.inputs.offsetBg = state.ra.alloc_sub(state.offsetB.getType());
            emov(1, state.inputs.offsetBg, 0, strategy, state);
        }

        for (int b = 0; b < problem.batchDims; b++) {
            eadd(1, state.offsetA, state.offsetA, bOffsetA[b], strategy, state);
            eadd(1, state.offsetB, state.offsetB, bOffsetB[b], strategy, state);
            for (int q = 0; q < state.C_count; q++) {
                auto offsetC = state.offsetC[q];
                eadd(1, offsetC, offsetC, bOffsetC[b], strategy, state);
            }
            if(problem.hasAScalePtr()){
                    eadd(1, state.offsetAs, state.offsetAs, bOffsetAs[b], strategy, state);
            }
            if(problem.hasBScalePtr()){
                    eadd(1, state.offsetBs, state.offsetBs, bOffsetBs[b], strategy, state);
            }
            if(problem.hasCMXScale()){
                    eadd(1, state.offsetCs, state.offsetCs, bOffsetCs[b], strategy, state);
            }
            if(problem.hasAOffsetPtr()){
                    eadd(1, state.offsetAo, state.offsetAo, bOffsetAo[b], strategy, state);
            }
            if(problem.hasBOffsetPtr()){
                    eadd(1, state.offsetBo, state.offsetBo, bOffsetBo[b], strategy, state);
            }
            if(problem.needsAGroupSums()){
                    eadd(1, state.inputs.offsetAg, state.inputs.offsetAg, bOffsetAg[b], strategy, state);
            }
            if(problem.needsBGroupSums()){
                    eadd(1, state.inputs.offsetBg, state.inputs.offsetBg, bOffsetBg[b], strategy, state);
            }
            if (!strategy.persistentLoop()) {
                state.ra.safeRelease(state.inputs.strideA[b]);
                state.ra.safeRelease(state.inputs.strideB[b]);
                state.ra.safeRelease(state.inputs.strideC[b]);
            }
            if (strategy.A.base.isStateless()) state.ra.safeRelease(bOffsetA[b]);
            if (strategy.B.base.isStateless()) state.ra.safeRelease(bOffsetB[b]);
            if (strategy.C.base.isStateless()) state.ra.safeRelease(bOffsetC[b]);
        }
    }

}

// Prepare for persistent GEMM by folding offsets into A/B/C pointers (if stateless),
//  or saving offsets (if stateful)
template <HW hw>
void Generator<hw>::gemmFoldOffsets(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto foldOrSave = [&](const MatrixAddressingStrategy &sX, Subregister &inputX, Subregister &offsetX, const Subregister &inputOffsetX, Subregister &saveOffsetX, bool newInput = false) {
        if (sX.base.isStateless()) {
            auto oldInputX = inputX;
            if (newInput)
                inputX = state.ra.alloc_sub(DataType::uq, getHint(HintType::LongTerm, strategy));
            eadd(1, inputX, oldInputX, offsetX, strategy, state);
            if (getBytes(offsetX.getType()) < 8) {
                state.ra.safeRelease(offsetX);
                offsetX = state.ra.alloc_sub(DataType::uq, getHint(HintType::LongTerm, strategy));
            }
            emov(1, offsetX, 0, strategy, state);
        } else {
            offsetX = state.ra.alloc_sub(offsetX.getType(), getHint(HintType::LongTerm, strategy));
            mov(1, offsetX, inputOffsetX);
        }
        saveOffsetX = offsetX;
    };

    bool deduplicateAB = (state.inputs.A == state.inputs.B);

    foldOrSave(strategy.A, state.inputs.A, state.offsetA, state.inputs.offsetA, state.saveOffsetA, deduplicateAB);
    foldOrSave(strategy.B, state.inputs.B, state.offsetB, state.inputs.offsetB, state.saveOffsetB);
    for (int q = 0; q < state.C_count; q++)
        foldOrSave(strategy.C, state.inputs.C[q], state.offsetC[q], state.inputs.offsetC[q], state.saveOffsetC[q]); // todo init for hpl
    if (problem.usesCOPtr())
        foldOrSave(strategy.CO, state.inputs.coPtr, state.offsetCO, state.inputs.offsetCO, state.saveOffsetCO);

    if (deduplicateAB)
        state.effA = state.inputs.A;
}

// Restore input offsets from saved copies, for persistent GEMM.
template <HW hw>
void Generator<hw>::gemmRestoreOffsets(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto zeroOrRestore = [&](const MatrixAddressingStrategy &sX, const Subregister &offsetX, const Subregister &inputOffsetX) {
        if (sX.base.isStateless())
            emov(1, offsetX, 0, strategy, state);
        else
            mov(1, offsetX, inputOffsetX);
    };

    zeroOrRestore(strategy.A, state.saveOffsetA, state.inputs.offsetA);
    zeroOrRestore(strategy.B, state.saveOffsetB, state.inputs.offsetB);
    for (int q = 0; q < state.C_count; q++)
        zeroOrRestore(strategy.C, state.saveOffsetC[q], state.inputs.offsetC[q]);
    if (problem.usesCOPtr())
        zeroOrRestore(strategy.CO, state.saveOffsetCO, state.inputs.offsetCO);
}

// Prepare final A/B/C pointers for a GEMM-like inner loop.
template <HW hw>
void Generator<hw>::gemmSetupABC(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (strategy.persistentLoop()) {
        state.effA = state.offsetA;
        state.effB = state.offsetB;
        for (int q = 0; q < state.C_count; q++)
            state.effC[q] = state.offsetC[q];
        state.effCO = state.offsetCO;
    }

    if (strategy.C.base.isStateless()) {
        for (int q = 0; q < state.C_count; q++) {
            auto Csrc = state.inputs.C[q];
            if ((q > 0) && strategy.C.base.isStateless() && state.inputs.base.isValid())
                state.effC[q] = state.inputs.C[q] = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));

            eadd(1, state.effC[q], Csrc, state.offsetC[q], strategy, state);
            if (strategy.persistentLoop())
                state.offsetC[q] = invalid;
            else
                state.ra.safeRelease(state.offsetC[q]);
        }
    }

    if (problem.usesCOPtr() && strategy.CO.base.isStateless()) {
        eadd(1, state.effCO, state.inputs.coPtr, state.offsetCO, strategy, state);
        if (strategy.persistentLoop())
            state.offsetCO = invalid;
        else
            state.ra.safeRelease(state.offsetCO);
    }

    if (state.offsetAp.isValid()) {
        if (strategy.A.base.isStateless()) {
            state.effAp = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));
            eadd(1, state.effAp, state.inputs.A, state.offsetAp, strategy, state);
            state.ra.safeRelease(state.offsetAp);
        } else
            state.effAp = state.offsetAp;
    }

    if (state.offsetBp.isValid()) {
        if (strategy.B.base.isStateless()) {
            state.effBp = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));
            eadd(1, state.effBp, state.inputs.B, state.offsetBp, strategy, state);
            state.ra.safeRelease(state.offsetBp);
        } else
            state.effBp = state.offsetBp;
    }

    if (state.offsetCp.isValid()) {
        if (strategy.C.base.isStateless()) {
            state.effCp = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));
            eadd(1, state.effCp, state.inputs.C[0], state.offsetCp, strategy, state);
            state.ra.safeRelease(state.offsetCp);
        } else
            state.effCp = state.offsetCp;
    }

    if (strategy.A.base.isStateless()) {
        auto Asrc = state.inputs.A;
        if (strategy.B.base.isStateless() && (state.effA == state.effB))
            state.effA = state.inputs.A = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));

        eadd(1, state.effA, Asrc, state.offsetA, strategy, state);
        if (strategy.persistentLoop())
            state.offsetA = invalid;
        else
            state.ra.safeRelease(state.offsetA);
    }

    if (strategy.B.base.isStateless()) {
        eadd(1, state.effB, state.inputs.B, state.offsetB, strategy, state);
        if (strategy.persistentLoop())
            state.offsetB = invalid;
        else
            state.ra.safeRelease(state.offsetB);
    }

    if (strategy.prefetchA && state.effAp.isInvalid()) state.effAp = state.effA;
    if (strategy.prefetchB && state.effBp.isInvalid()) state.effBp = state.effB;
    if (strategy.prefetchC && state.effCp.isInvalid()) state.effCp = state.effC[0];
}

// Get (possibly multidimensional) batch IDs.
template <HW hw>
void Generator<hw>::gemmGetBatchIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (problem.batchDims == 0) return;

    if (problem.batchDims == 1) {
        state.batchID[0] = state.inputs.groupIDK;
        return;
    }

    for (int i = 0; i < problem.batchDims; i++)
        state.batchID[i] = state.ra.alloc_sub<uint32_t>();

    auto div = state.ra.alloc_sub<uint32_t>();
    mov(1, div, state.inputs.groupIDK);
    for (int i = problem.batchDims - 1; i >= 0; i--) {
        auto idx = problem.batchDims - 1 - i;
        mov(1, state.batchID[idx], div);
        if (i > 0) {
            divDown(div, div, state.inputs.batchSize[i - 1], state.inputs.recipBatchSize[i - 1], state.flagAP, strategy, state);
            emad(1, state.batchID[idx], state.batchID[idx], -div, state.inputs.batchSize[i - 1], strategy, state);
            if (!strategy.persistentLoop()) {
                state.ra.safeRelease(state.inputs.batchSize[i - 1]);
                state.ra.safeRelease(state.inputs.recipBatchSize[i - 1]);
            }
        }
    }
    state.ra.safeRelease(div);
}

template <HW hw>
void Generator<hw>::gemmReleaseBatchIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (problem.batch != BatchMode::Strided) return;
    if (problem.batchDims == 1 && state.r0_info == r0) return;
    if (problem.hasBinaryPostOp()) return;
    for (int b = 0; b < problem.batchDims; b++)
        state.ra.safeRelease(state.batchID[b]);
}


// Convert leading dimension and offset inputs to bytes.
template <ngen::HW hw>
void Generator<hw>::gemmScaleInputs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext, Tc_ext = problem.Tc_ext, Tco = problem.Tco;
    auto &inputs = state.inputs;

    auto scale = [&](Type T, Subregister &s, Subregister defaultSrc = Subregister()) {
        if (s.isValid())
            emulConstant(1, s, s, T, strategy, state);
        else if (defaultSrc.isValid()) {
            s = state.ra.alloc_sub(defaultSrc.getType(), getHint(HintType::LongTerm, strategy));
            emulConstant(1, s, defaultSrc, T, strategy, state);
        }
    };

    scale(Ta_ext, inputs.lda);
    if (state.inputs.ldb != state.inputs.lda)
        scale(Tb_ext, inputs.ldb);
    for (int q = 0; q < state.C_count; q++)
        scale(Tc_ext, inputs.ldc[q]);
    scale(Tco, inputs.ldco);

    {
        if (strategy.A.base.getModel() != ModelSLM)
            scale(Ta_ext, inputs.offsetA);
        if (strategy.B.base.getModel() != ModelSLM)
            scale(Tb_ext, inputs.offsetB);
        for (int q = 0; q < state.C_count; q++)
            scale(Tc_ext, inputs.offsetC[q]);
        if (problem.usesCOPtr())
            scale(Tco, inputs.offsetCO);
    }

    if (problem.batch == BatchMode::Strided){
        for (int b = 0; b < problem.batchDims; b++) {
            scale(Ta_ext, inputs.strideA[b]);
            scale(Tb_ext, inputs.strideB[b]);
            scale(Tc_ext, inputs.strideC[b]);
            if (problem.hasAScalePtr())
                scale(problem.Ta_scale, state.inputs.strideScaleA[b]);
            if (problem.hasBScalePtr())
                scale(problem.Tb_scale, state.inputs.strideScaleB[b]);
            if(problem.hasCMXScale())
                scale(problem.Tc_scale, state.inputs.strideScaleC[b]);
            if (problem.hasAOffsetPtr())
                scale(problem.Tao, state.inputs.strideOffsetA[b]);
            if (problem.hasBOffsetPtr())
                scale(problem.Tbo, state.inputs.strideOffsetB[b]);
            if(problem.needsAGroupSums())
                scale(problem.Tag, state.inputs.strideGroupSumsA[b]);
            if(problem.needsBGroupSums())
                scale(problem.Tbg, state.inputs.strideGroupSumsB[b]);
        }
    }

    auto ldaq = inputs.ldaq, ldbq = inputs.ldbq, ldcq = inputs.ldcq;
    if (ldaq.isInvalid()) ldaq = inputs.m;
    if (ldbq.isInvalid()) ldbq = inputs.n;

    if (problem.aOffset2D()) 
        scale(problem.Tao, inputs.ldao, ldaq);
    if (problem.aoPtrDims >= 0)
        scale(problem.Tao, inputs.offsetAO, inputs.offsetAq);
    if (problem.bOffset2D()) 
        scale(problem.Tbo, inputs.ldbo, ldbq);
    if (problem.boPtrDims >= 0)
        scale(problem.Tbo, inputs.offsetBO, inputs.offsetBq);
    if (problem.aScale2D()) {
        scale(problem.Ta_scale, inputs.ldaScale, ldaq);
        scale(problem.Ta_scale, inputs.offsetAScale, inputs.offsetAq);
    }
    if (problem.bScale2D()) {
        scale(problem.Tb_scale, inputs.ldbScale, ldbq);
        scale(problem.Tb_scale, inputs.offsetBScale, inputs.offsetBq);
    }
    if (problem.hasCMXScale()) {
        scale(problem.Tc_scale, inputs.ldcScale, ldcq);
    }
    if (problem.needsAGroupSums()) {
        scale(problem.Tag, inputs.ldag, ldaq);
        scale(problem.Tag, inputs.offsetAg, inputs.offsetAq);
    }
    if (problem.needsBGroupSums()) {
        scale(problem.Tbg, inputs.ldbg, ldbq);
        scale(problem.Tbg, inputs.offsetBg, inputs.offsetBq);
    }

    state.ldao = inputs.ldao;
    state.ldbo = inputs.ldbo;
    state.ldaScale = inputs.ldaScale;
    state.ldbScale = inputs.ldbScale;
    state.ldcScale = inputs.ldcScale;
    state.ldag = inputs.ldag;
    state.ldbg = inputs.ldbg;

    state.ra.safeRelease(inputs.ldaq);
    state.ra.safeRelease(inputs.ldbq);
    state.ra.safeRelease(inputs.offsetAq);
    state.ra.safeRelease(inputs.offsetBq);
}

// Calculate workgroup m/n remainders.
template <HW hw>
void Generator<hw>::gemmCalcWGRemainders(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (wgRemCheck(problem, strategy)) {
        state.remaindersWG[LoopM] = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
        state.remaindersWG[LoopN] = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
        add(1 | sat, state.remaindersWG[LoopM], -state.wgI0, state.inputs.m);
        add(1 | sat, state.remaindersWG[LoopN], -state.wgJ0, state.inputs.n);
        if (strategy.cInterleave)
            divUp(state.remaindersWG[LoopN], state.remaindersWG[LoopN], strategy.cInterleaveChunk(problem.Tc_ext), strategy, state);
    }
    if (strategy.coopA != CoopSplit::FullK) state.ra.safeRelease(state.wgI0);
    if (strategy.coopB != CoopSplit::FullK) state.ra.safeRelease(state.wgJ0);
}

// Cache multiples of lda/ldb for later address calculations.
template <HW hw>
void Generator<hw>::gemmCacheLDABMultiples(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA, bool doB)
{
    int na = 0, nb = 0;

    if (doA && !strategy.A.address2D) switch (problem.A.layout) {
        case MatrixLayout::N: na = std::max(strategy.ka_load, strategy.ka_prefetch); break;
        case MatrixLayout::T:
            na = strategy.unroll[LoopM];
            if (isTransposing(strategy.A.accessType)) na = std::min(na, maxScatteredSIMD(hw, strategy.A));
            break;
        default: break;
    }

    if (doB && !strategy.B.address2D) switch (problem.B.layout) {
        case MatrixLayout::T: nb = std::max(strategy.kb_load, strategy.kb_prefetch); break;
        case MatrixLayout::N:
            nb = strategy.unroll[LoopN];
            if (isTransposing(strategy.B.accessType)) nb = std::min(nb, maxScatteredSIMD(hw, strategy.B));
            break;
        default: break;
    }

    if (na <= 2) na = 0;
    if (nb <= 2) nb = 0;

    if (na || nb)
        extendIndexVec(std::max(na, nb), state);

    if (na) {
        bool a64 = (strategy.A.base.getModel() == ModelA64);
        state.ldaMultiples = createLDMultiples(a64, na, state.lda, strategy, state);
    }

    if (nb) {
        bool a64 = (strategy.B.base.getModel() == ModelA64);
        state.ldbMultiples = createLDMultiples(a64, nb, state.ldb, strategy, state);
    }
}

// Cache multiples of ldc for later address calculations.
template <HW hw>
void Generator<hw>::gemmCacheLDCMultiples(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool prefetch)
{
    if ((prefetch ? strategy.C_prefetch : strategy.C).address2D) return;

    int nc = 0;
    switch (problem.C.layout) {
        case MatrixLayout::N: nc = strategy.unroll[LoopN]; break;
        case MatrixLayout::T: nc = strategy.unroll[LoopM]; break;
        default: break;
    }

    if (nc <= 2) return;

    bool a64 = (strategy.C.base.getModel() == ModelA64);
    int C_count = prefetch ? 1 : state.C_count;
    for (int q = 0; q < C_count; q++)
        state.ldcMultiples[q] = createLDMultiples(a64, nc, state.inputs.ldc[q], strategy, state);
}

template <HW hw>
void Generator<hw>::gemmDowngradeAccess(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    bool pfA = strategy.prefetchA, pfB = strategy.prefetchB;
    bool a2D = strategy.A.address2D, ap2D = strategy.A_prefetch.address2D;
    bool b2D = strategy.B.address2D, bp2D = strategy.B_prefetch.address2D;

    bool oldA2D = a2D, oldAp2D = ap2D, oldB2D = b2D, oldBp2D = bp2D;

    strategy.A.accessType = strategy.unalignedAccA;
    strategy.B.accessType = strategy.unalignedAccB;

    a2D &= isBlock2D(strategy.A.accessType);
    b2D &= isBlock2D(strategy.B.accessType);

    int minAlignA = block2DMinAlignment(hw, problem.A, strategy.A_prefetch);
    int minAlignB = block2DMinAlignment(hw, problem.B, strategy.B_prefetch);

    if (pfA && isBlock2D(strategy.A_prefetch.accessType) && problem.A.alignment < minAlignA) {
        downgradeAPFAccess(problem, strategy);
        ap2D = false;
    }

    if (pfB && isBlock2D(strategy.B_prefetch.accessType) && problem.B.alignment < minAlignB) {
        downgradeBPFAccess(problem, strategy);
        bp2D = false;
    }

    if (pfA && !a2D && !ap2D) {
        if (oldAp2D && !oldA2D) state.effAp = state.effA;
        if (oldA2D && !oldAp2D) state.effA = state.effAp;
    }
    if (pfB && !b2D && !bp2D) {
        if (oldBp2D && !oldB2D) state.effBp = state.effB;
        if (oldB2D && !oldBp2D) state.effB = state.effBp;
    }

    bool applyOffsetA = !pfA ? (oldA2D && !a2D) : (oldA2D && oldAp2D && (!a2D || !ap2D));
    bool applyOffsetB = !pfB ? (oldB2D && !b2D) : (oldB2D && oldBp2D && (!b2D || !bp2D));

    strategy.A.address2D = a2D;
    strategy.B.address2D = b2D;
    strategy.A_prefetch.address2D = ap2D;
    strategy.B_prefetch.address2D = bp2D;

    if (applyOffsetA || applyOffsetB)
        gemmOffsetABC(false, state.i0, state.j0, state.h0, Subregister(), Subregister(), problem, strategy, state, applyOffsetA, applyOffsetB, false);
}


static inline bool needsKLoopReset(const GEMMProblem &problem)
{
    return false;
}

// Setup for C accumulation.
// NOTE: modifies problem/strategy/state.
template <HW hw>
bool Generator<hw>::gemmAccumulateCSetup(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto &Ta = problem.Ta, &Tb = problem.Tb, Tc = problem.Tc, Tc_compute = problem.Tc_compute();
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext, Tc_ext = problem.Tc_ext;
    auto Tao = problem.Tao, Tbo = problem.Tbo;
    auto Ta_scale = problem.Ta_scale, Tb_scale = problem.Tb_scale;
    auto Tag = problem.Tag, Tbg = problem.Tbg;
    auto &Ta_load = state.Ta_load, &Tb_load = state.Tb_load;
    bool slmA = strategy.slmA, slmB = strategy.slmB;

    bool cLoadAhead = strategy.cLoadAhead;
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto unrollK = strategy.unroll[LoopK];
    auto unrollKSLM = strategy.unrollKSLM;

    // Grab a whole flag register now to do emulated SIMD16 64-bit adds, if needed.
    FlagRegister simd16EmulationFlag;
    if (state.emulate.temp[0].isValid() && GRF::bytes(hw) == 64
            && strategy.emulate.emulate64 && !strategy.emulate.emulate64_add32)
        simd16EmulationFlag = state.raVFlag.alloc();

    // Decide what remainder handling needs to be done.
    bool remainderM = strategy.remHandling[LoopM] != RemainderHandling::Ignore;
    bool remainderN = strategy.remHandling[LoopN] != RemainderHandling::Ignore;
    bool remainderK = strategy.remHandling[LoopK] != RemainderHandling::Ignore;
    bool remM_A = remainderM && !strategy.A.padded;
    bool remK_A = false;
    bool remK_B = false;
    bool remN_B = remainderN && !strategy.B.padded;
    bool remM_C, remN_C;
    getCRemainders(hw, problem, strategy, remM_C, remN_C);
    bool remM_Ce = remM_C;
    bool remN_Ce = remN_C;

    if (state.copyC)
        remM_C = remN_C = false;

    // 2D addressing parameters.
    auto &A_params = state.A_params, &B_params = state.B_params;
    auto &Ai_params = state.Ai_params, &Bi_params = state.Bi_params;
    auto &Ap_params = state.Ap_params, &Bp_params = state.Bp_params;
    A_params.rows = state.inputs.m;
    A_params.cols = state.fullK;
    A_params.offR = state.i0;
    A_params.offC = state.h0;
    A_params.remR = state.remainders[LoopM];
    B_params.rows = state.fullK;
    B_params.cols = state.inputs.n;
    B_params.offR = state.h0;
    B_params.offC = state.j0;
    B_params.remC = state.remainders[LoopN];
    Ai_params = A_params, Bi_params = B_params;
    Ap_params = A_params, Bp_params = B_params;

    // Decide which dimensions to split for WG-cooperative operations (SLM copy, cooperative PF).
    state.effCoopA = effCoopSplitA(problem, strategy);
    state.effCoopB = effCoopSplitB(problem, strategy);

    if (state.effCoopA == CoopSplit::FullK) Ap_params.offR = Ai_params.offR = state.wgI0;
    if (state.effCoopB == CoopSplit::FullK) Bp_params.offC = Bi_params.offC = state.wgJ0;

    // Prepare remainders for cooperative operations.
    for (LoopType loop: {LoopM, LoopN, LoopK})
        state.remaindersCoop[loop] = state.remainders[loop];

    auto calcMNRemCoop = [&](CoopSplit split, bool isM) {
        auto loopX = isM ? LoopM : LoopN;
        auto loopY = isM ? LoopN : LoopM;
        switch (split) {
            default:
                return state.remainders[loopX];
            case CoopSplit::FullK:
                return state.remaindersWG[loopX];
            case CoopSplit::MN: {
                auto rem = state.ra.alloc_sub<uint16_t>();
                int32_t chunk = strategy.unroll[loopX] / strategy.wg[loopY];
                auto lid = isM ? state.lidN : state.lidM;
                emad(1 | sat, rem, state.remainders[loopX], -lid.w(), chunk, strategy, state);
                return rem;
            }
        }
    };

    if ((slmA || (strategy.prefetchA && strategy.cooperativePF)) && remM_A) {
        state.remaindersCoop[LoopM] = calcMNRemCoop(state.effCoopA, true);
    }

    if ((slmB || (strategy.prefetchB && strategy.cooperativePF)) && remN_B) {
          state.remaindersCoop[LoopN] = calcMNRemCoop(state.effCoopB, false);
    }

    // Layout creation error handling.
#define CREATE_LAYOUT(l, ...) do {                      \
        l = RegisterLayout::tryCreate(hw, __VA_ARGS__); \
        if (!l) return false;                           \
    } while (false)

    // Prepare layouts for prefetch.
    bool remM_Cp = remM_C && strategy.C.base.isStateless();
    bool remN_Cp = remN_C && strategy.C.base.isStateless();

    state.ma_prefetch = state.ka_prefetch = state.kb_prefetch = state.nb_prefetch = 0;
    if (strategy.prefetchA) coopSplit(true,  state.ma_prefetch, state.ka_prefetch, unrollM, strategy.ka_prefetch, strategy.wgTile(LoopM), state.effCoopA, problem.A, strategy);
    if (strategy.prefetchB) coopSplit(false, state.kb_prefetch, state.nb_prefetch, strategy.kb_prefetch, unrollN, strategy.wgTile(LoopN), state.effCoopB, problem.B, strategy);

    if (strategy.prefetchA) CREATE_LAYOUT(state.Ap_layout, Ta_ext, state.ma_prefetch, state.ka_prefetch, problem.A, strategy.A_prefetch, remM_A,  remK_A,  false);
    if (strategy.prefetchB) CREATE_LAYOUT(state.Bp_layout, Tb_ext, state.kb_prefetch, state.nb_prefetch, problem.B, strategy.B_prefetch, remK_B,  remN_B,  false);
    if (strategy.prefetchC) CREATE_LAYOUT(state.Cp_layout, Tc_ext, unrollM,           unrollN,           problem.C, strategy.C_prefetch, remM_Cp, remN_Cp, false);

    if (state.Cp_layout.hasMasking() || state.Cp_layout.hasFragmenting())
        stub();

    gemmABPrefetchAddrSetup(problem, strategy, state);

    // Prepare layouts and starting addresses for SLM copies and adjust problem.
    if (strategy.slmBuffers > 0) {
        int A_slmCP, B_slmCP;
        int A_tileR, A_tileC, B_tileR, B_tileC;
        std::tie(A_slmCP, B_slmCP) = targetSLMCrosspack(problem, strategy);
        std::tie(A_tileR, A_tileC, B_tileR, B_tileC) = targetKernelTiling(hw, problem, strategy);
        auto opCount = outerProductCount(problem, strategy);

        if (slmA) {
            coopSplit(true, state.ma_slm, state.ka_slm, unrollM, unrollKSLM, strategy.wgTile(LoopM), state.effCoopA, problem.A, strategy);

            if (state.effCoopA == CoopSplit::MN)
                remK_A = remainderK && strategy.slmEarlyKMask;

            if (strategy.slmATrans) {
                A_slmCP = state.ka_slm;
                if (strategy.ka_load % A_slmCP)
                    stub("ka_load must be a multiple of ka_slm");
            }
            if ((state.ka_slm < A_slmCP) && (unrollKSLM != A_slmCP) && (A_tileC != A_slmCP))
                stub("ka_slm must be a multiple of crosspack, or unrollKSLM = crosspack.");
            if (isPacked(problem.A.layout) && problem.A.packSize != static_cast<uint32_t>(unrollM))
                stub("A panel height must match unroll");

            // Layout in from memory...
            state.Ai = problem.A;
            state.Ai_strategy = strategy.A;
            if (state.Ai_strategy.dpasw) {
                state.Ai_strategy.dpasw = false;
                state.Ai_strategy.tileR = 0;
            }

            // ... layout out to SLM.
            state.Ao.layout = MatrixLayout::Pc;
            state.Ao.packSize = unrollM;
            state.Ao.panelLength = unrollKSLM;
            state.Ao.crosspack = A_slmCP;
            state.Ao.setAlignment(state.Ao.packSize * Ta);
            state.Ao.tileR = A_tileR;
            state.Ao.tileC = (A_tileC || !A_tileR) ? A_tileC : std::max(opCount, strategy.ka_load);

            bool colMajorIn = isRegisterColMajor(Ta_ext, state.Ai, state.Ai_strategy);
            bool colMajorSLM = !isLargeCrosspack(Ta, A_slmCP);
            state.Ao_strategy.base = SLM;
            state.Ao_strategy.accessType = (colMajorIn == colMajorSLM) ? AccessType::Block
                                                                       : AccessType::Scattered;
            state.Ao_strategy.smode = ScatterSIMD::Default;

            if (state.Ai.layout == MatrixLayout::N && one_of(state.Ai_strategy.accessType, {AccessType::Block2D, AccessType::Block2DVNNI}) && isLargeCrosspack(Ta, A_slmCP)) {
                state.Ao_strategy.accessType = AccessType::ChannelScattered;
                state.Ao_strategy.smode = ScatterSIMD::Narrow;
            }
            state.Ao_strategy.padded = true;
            state.Ao_strategy.atomic = false;
            state.Ao_strategy.address2D = false;
            state.Ao_strategy.newDP = (hw >= HW::XeHPG);
            state.Ao_strategy.cachingW = CacheSettingsLSC::Default;

            // Layout in from memory...
            CREATE_LAYOUT(state.Ai_layout, Ta_ext, state.ma_slm, state.ka_slm, state.Ai, state.Ai_strategy, remM_A, remK_A, false);

            if (state.Ai_layout.hasFragmenting(false, true)) {
                status << "Can't fragment in m dimension." << status_stream::endl;
                return false;
            }

            // ... layout out to SLM...
            remM_A = remK_A = false;
            CREATE_LAYOUT(state.Ao_layout, Ta, state.ma_slm, state.ka_slm, state.Ao, state.Ao_strategy, remM_A, remK_A, true);

            // ... and layout back from SLM.
            problem.A = state.Ao;
            strategy.A.base = SLM;
            strategy.A.accessType = AccessType::Block;
            strategy.A.address2D = false;
            strategy.A.newDP = (hw >= HW::XeHPG);
            strategy.A.cachingR = CacheSettingsLSC::Default;
            Ta_load = Ta;
            state.aioShare = matchBidirectional(state.Ai_layout, state.Ao_layout);

            // If we will add k-masking later, check if extra registers are needed.
            state.Ai_regCount = state.Ai_layout.regs();
            if (!remK_A && remainderK && !state.Ai_strategy.address2D && !isRegisterColMajor(Ta_ext, state.Ai, state.Ai_strategy)) {
                auto Ai_layoutKMasked = RegisterLayout::tryCreate(hw, Ta_ext, state.ma_slm, state.ka_slm, state.Ai, state.Ai_strategy, remM_A, true);
                state.Ai_regCount = std::max(state.Ai_regCount, Ai_layoutKMasked.regs());
            }

            // Offset A addresses in and out.
            state.effAi = state.effA;
            state.effA = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            state.effAo = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));

            auto temp = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            Subregister temp2;

            uint32_t noff, noffTile, tileSplit = 1;

            switch (state.effCoopA) {
                case CoopSplit::Linear:
                    // FIXME: assumes compatible tiling between global and SLM layouts.
                    noff = state.ma_slm * state.ka_slm;
                    break;
                case CoopSplit::MN:
                    noff = untile(Ta, state.Ao, 0, state.ma_slm, 0, state.Ao.packSize, unrollKSLM);
                    if (state.ma_slm < state.Ao.tileR && state.Ao.tileR < state.Ao.packSize) {
                        // m division splits tiles -- starting offsets no longer a linear sequence.
                        if (state.Ao.tileR % state.ma_slm) stub();
                        tileSplit = state.Ao.tileR / state.ma_slm;
                        noffTile = untile(Ta, state.Ao, 0, state.Ao.tileR, 0, state.Ao.packSize, unrollKSLM);
                    }
                    break;
                case CoopSplit::K:
                    noff = untile(Ta, state.Ao, 0, 0, state.ka_slm, state.Ao.packSize, unrollKSLM);
                    if (state.ka_slm < state.Ao.tileC && state.Ao.tileC < unrollKSLM) {
                        // k division splits tiles -- starting offsets no longer a linear sequence.
                        if (state.Ao.tileC % state.ka_slm) stub();
                        tileSplit = state.Ao.tileC / state.ka_slm;
                        noffTile = untile(Ta, state.Ao, 0, 0, state.Ao.tileC, state.Ao.packSize, unrollKSLM);
                    }
                    break;
                default: stub();
            }

            int32_t A_slmStride = strategy.slmABufBlockSize(problem) * strategy.slmBuffers;

            if (tileSplit > 1) {
                if (!is_zero_or_pow2(tileSplit)) stub();
                shr(1, temp, state.lidN, ilog2(tileSplit));
            }
            gemmCalcWorkshareAOffset(temp2, Ai_params.offR, Ai_params.offC, state.Ai, state.Ai_strategy, state.ma_slm, state.ka_slm, problem, strategy, state);
            if (tileSplit > 1) {
                mulConstant(1, temp, temp, (noffTile - noff * tileSplit) * Ta);
                emad(1, temp, temp, state.lidN, noff * Ta, strategy, state);
            } else
                mulConstant(1, temp, state.lidN, noff * Ta);
            mulConstant(1, state.effA, state.lidM, A_slmStride);
            if (strategy.wg[LoopK] > 1)
                emad(1, state.effA, state.effA, state.lidK, A_slmStride * strategy.wg[LoopM], strategy, state);
            if (state.Ai_strategy.address2D) {
                if (Ai_params.offR != A_params.offR && A_params.offR.isValid())
                    add(1, Ai_params.offR, Ai_params.offR, A_params.offR);
                if (Ai_params.offC != A_params.offC && A_params.offC.isValid())
                    add(1, Ai_params.offC, Ai_params.offC, A_params.offC);
            } else
                eadd(1, state.effAi, state.effAi, temp2, strategy, state);
            makeSLMBaseRelative(state.effA, state);
            add(1, state.effAo, state.effA, temp);
            if (problem.backward())
                add(1, state.effA, state.effA, (unrollKSLM - strategy.ka_load) * unrollM * Ta);

            state.ra.safeRelease(temp2);
            state.ra.safeRelease(temp);
        }
        if (slmB) {
            coopSplit(false, state.kb_slm, state.nb_slm, unrollKSLM, unrollN, strategy.wgTile(LoopN), state.effCoopB, problem.B, strategy);

            if (state.effCoopB == CoopSplit::MN)
                remK_B = remainderK && strategy.slmEarlyKMask;

            if (strategy.slmBTrans) {
                B_slmCP = state.kb_slm;
                if (strategy.kb_load % B_slmCP)
                    stub("kb_load must be a multiple of kb_slm");
            }
            if ((state.kb_slm < B_slmCP) && (unrollKSLM != B_slmCP) && (B_tileR != B_slmCP))
                stub("kb_slm must be a multiple of crosspack, or unrollKSLM = crosspack.");
            if (isPacked(problem.B.layout) && problem.B.packSize != static_cast<uint32_t>(unrollN))
                stub("B panel height must match unroll");

            // Layout in from memory...
            state.Bi = problem.B;
            state.Bi_strategy = strategy.B;
            if (state.Bi_strategy.dpasw) {
                state.Bi_strategy.dpasw = false;
                state.Bi_strategy.tileC = 0;
            }

            // ... layout out to SLM.
            state.Bo.layout = MatrixLayout::Pr;
            state.Bo.packSize = unrollN;
            state.Bo.panelLength = unrollKSLM;
            state.Bo.crosspack = B_slmCP;
            state.Bo.setAlignment(state.Bo.packSize * Tb);
            state.Bo.tileR = (B_tileR || !B_tileC) ? B_tileR : std::max(opCount, strategy.kb_load);
            state.Bo.tileC = B_tileC;

            bool colMajorIn = isRegisterColMajor(Tb_ext, state.Bi, state.Bi_strategy);
            bool colMajorSLM = isLargeCrosspack(Tb, B_slmCP);
            state.Bo_strategy.base = SLM;
            state.Bo_strategy.accessType = (colMajorIn == colMajorSLM) ? AccessType::Block
                                                                       : AccessType::Scattered;
            state.Bo_strategy.smode = ScatterSIMD::Default;

            if (state.Bi.layout == MatrixLayout::T && one_of(state.Bi_strategy.accessType, {AccessType::Block2D, AccessType::Block2DVNNI}) && isLargeCrosspack(Tb, B_slmCP)) {
                state.Bo_strategy.accessType = AccessType::ChannelScattered;
                state.Bo_strategy.smode = ScatterSIMD::Narrow;
            }
            state.Bo_strategy.padded = true;
            state.Bo_strategy.atomic = false;
            state.Bo_strategy.address2D = false;
            state.Bo_strategy.newDP = (hw >= HW::XeHPG);
            state.Bo_strategy.cachingW = CacheSettingsLSC::Default;

            // Layout in from memory...
            CREATE_LAYOUT(state.Bi_layout, Tb_ext, state.kb_slm, state.nb_slm, state.Bi, state.Bi_strategy, remK_B, remN_B, false);

            if (state.Bi_layout.hasFragmenting(true, false)) {
                status << "Can't fragment in n dimension." << status_stream::endl;
                return false;
            }

            // ... layout out to SLM...
            remK_B = remN_B = false;
            CREATE_LAYOUT(state.Bo_layout, Tb, state.kb_slm, state.nb_slm, state.Bo, state.Bo_strategy, remK_B, remN_B, true);

            // ... and layout back from SLM.
            problem.B = state.Bo;
            strategy.B.base = SLM;
            strategy.B.accessType = AccessType::Block;
            strategy.B.address2D = false;
            strategy.B.newDP = (hw >= HW::XeHPG);
            strategy.B.cachingR = CacheSettingsLSC::Default;
            Tb_load = Tb;
            state.bioShare = matchBidirectional(state.Bi_layout, state.Bo_layout);

            // If we will add k-masking later, check if extra registers are needed.
            state.Bi_regCount = state.Bi_layout.regs();
            if (!remK_B && remainderK && !state.Bi_strategy.address2D && isRegisterColMajor(Tb_ext, state.Bi, state.Bi_strategy)) {
                auto Bi_layoutKMasked = RegisterLayout::tryCreate(hw, Tb_ext, state.kb_slm, state.nb_slm, state.Bi, state.Bi_strategy, true, remN_B);
                state.Bi_regCount = std::max(state.Bi_regCount, Bi_layoutKMasked.regs());
            }

            // Offset B addresses in and out.
            state.effBi = state.effB;
            state.effB = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            state.effBo = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));

            auto temp = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
            Subregister temp2;

            uint32_t moff, moffTile, tileSplit = 1;

            switch (state.effCoopB) {
                case CoopSplit::Linear:
                    moff = state.kb_slm * state.nb_slm;
                    break;
                case CoopSplit::MN:
                    moff = untile(Tb, state.Bo, 0, 0, state.nb_slm, unrollKSLM, state.Bo.packSize);
                    if (state.nb_slm < state.Bo.tileC && state.Bo.tileC < state.Bo.packSize) {
                        if (state.Bo.tileC % state.nb_slm) stub();
                        tileSplit = state.Bo.tileC / state.nb_slm;
                        moffTile = untile(Tb, state.Bo, 0, 0, state.Bo.tileC, unrollKSLM, state.Bo.packSize);
                    }
                    break;
                case CoopSplit::K:
                    moff = untile(Tb, state.Bo, 0, state.kb_slm, 0, unrollKSLM, state.Bo.packSize);
                    if (state.kb_slm < state.Bo.tileR) {
                        if (state.Bo.tileR % state.kb_slm) stub();
                        tileSplit = state.Bo.tileR / state.kb_slm;
                        moffTile = untile(Tb, state.Bo, 0, state.Bo.tileR, 0, unrollKSLM, state.Bo.packSize);
                    }
                    break;
                default: stub();
            }

            int32_t B_slmStride = strategy.slmBBufBlockSize(problem) * strategy.slmBuffers;

            if (tileSplit > 1) {
                if (!is_zero_or_pow2(tileSplit)) stub();
                shr(1, temp, state.lidM, ilog2(tileSplit));
            }
            gemmCalcWorkshareBOffset(temp2, Bi_params.offR, Bi_params.offC, state.Bi, state.Bi_strategy, state.kb_slm, state.nb_slm, problem, strategy, state);
            if (tileSplit > 1) {
                mulConstant(1, temp, temp, (moffTile - moff * tileSplit) * Tb);
                emad(1, temp, temp, state.lidM, moff * Tb, strategy, state);
            } else
                mulConstant(1, temp, state.lidM, moff * Tb);
            mulConstant(1, state.effB, state.lidN, B_slmStride);
            if (strategy.wg[LoopK] > 1)
                emad(1, state.effB, state.effB, state.lidK, B_slmStride * strategy.wg[LoopN], strategy, state);
            if (state.Bi_strategy.address2D) {
                if (Bi_params.offR != B_params.offR && B_params.offR.isValid())
                    add(1, Bi_params.offR, Bi_params.offR, B_params.offR);
                if (Bi_params.offC != B_params.offC && B_params.offC.isValid())
                    add(1, Bi_params.offC, Bi_params.offC, B_params.offC);
            } else
                eadd(1, state.effBi, state.effBi, temp2, strategy, state);
            if (strategy.slmABufSize(problem) > 0) {
                if (strategy.kParallelLocal) {
                    int32_t A_perKSLM = strategy.slmABufBlockSize(problem) * strategy.wg[LoopM] * strategy.slmBuffers;
                    emad(1, state.effB, state.effB, state.lszK, A_perKSLM, strategy, state);
                } else
                    add(1, state.effB, state.effB, strategy.slmABufSize(problem));
            }
            makeSLMBaseRelative(state.effB, state);
            add(1, state.effBo, state.effB, temp);
            if (problem.backward())
                add(1, state.effB, state.effB, (unrollKSLM - strategy.kb_load) * unrollN * Tb);

            state.ra.safeRelease(temp2);
            state.ra.safeRelease(temp);
        }
    }

    // Starting address and remainder adjustments for DPASW.
    bool cColMajor = isRegisterColMajor(Tc_ext, problem.C, strategy.C);
    Subregister saveRemM = state.remainders[LoopM];
    Subregister saveRemN = state.remainders[LoopN];
    if (strategy.dpasw) {
        if (cColMajor) {
            int t = strategy.B.tileC;
            and_(1 | nz | state.flagAP, null.uw(), state.lidM, 1);
            switch (problem.B.layout) {
                case MatrixLayout::N:  emad(1 | state.flagAP, state.effB, state.effB, state.inputs.ldb, Immediate::w(t), strategy, state);                               break;
                case MatrixLayout::T:  eadd(1 | state.flagAP, state.effB, state.effB, t * Tb_load, strategy, state);                                                     break;
                case MatrixLayout::Pr: eadd(1 | state.flagAP, state.effB, state.effB, untile(Tb_load, problem.B, 0, 0, t, unrollK, unrollN) * Tb_load, strategy, state); break;
                default: stub();
            }
            if (!slmB && remN_B) {
                state.remainders[LoopN] = state.ra.alloc_sub<int16_t>();
                mov(1, state.remainders[LoopN], saveRemN);
                add(1 | state.flagAP, state.remainders[LoopN], saveRemN, -t);
            }
        } else {
            int t = strategy.A.tileR;
            and_(1 | nz | state.flagAP, null.uw(), state.lidN, 1);
            switch (problem.A.layout) {
                case MatrixLayout::T:  emad(1 | state.flagAP, state.effA, state.effA, state.inputs.lda, Immediate::w(t), strategy, state);                               break;
                case MatrixLayout::N:  eadd(1 | state.flagAP, state.effA, state.effA, t * Ta_load, strategy, state);                                                     break;
                case MatrixLayout::Pc: eadd(1 | state.flagAP, state.effA, state.effA, untile(Ta_load, problem.A, 0, t, 0, unrollM, unrollK) * Ta_load, strategy, state); break;
                default: stub();
            }
            if (!slmA && remM_A) {
                state.remainders[LoopM] = state.ra.alloc_sub<int16_t>();
                mov(1, state.remainders[LoopM], saveRemM);
                add(1 | state.flagAP, state.remainders[LoopM], saveRemM, -t);
            }
        }
    }

    // Get register layouts for A/B/C.
    CREATE_LAYOUT(state.A_layout, Ta_load, unrollM,          strategy.ka_load, problem.A, strategy.A, remM_A, remK_A, false);
    CREATE_LAYOUT(state.B_layout, Tb_load, strategy.kb_load, unrollN,          problem.B, strategy.B, remK_B, remN_B, false);

    if (state.copyC) {
        if (state.useTempC)
            CREATE_LAYOUT(state.C_layout, Tc, unrollM, unrollN, state.tempC, state.tempCStrategy, false, false, true);
        else {
            state.C_layout = RegisterLayout(hw, Tc, unrollM, unrollN, cColMajor, 1, strategy.C.tileR, strategy.C.tileC, false);
        }
        CREATE_LAYOUT(state.C_layoutExt, Tc_ext, unrollM, unrollN, problem.C, state.Cext_strategy, remM_Ce, remN_Ce, true, AllowFragDescNFM);
    } else
        CREATE_LAYOUT(state.C_layout, Tc, unrollM, unrollN, problem.C, strategy.C, remM_C, remN_C, true, AllowFragDescNFM);

#undef CREATE_LAYOUT

    auto &layoutExt = state.copyC ? state.C_layoutExt : state.C_layout;
    if (!strategy.altCRemainder && (remM_Ce || remN_Ce)) {
        // Try preparing C layout without masking (may reduce memory accesses).
        // Only use it if compatible with the masked layout, and saves on send instructions (or avoids pseudoblock/slow scattered byte accesses).
        state.C_layoutExtUnmasked = RegisterLayout::tryCreate(hw, Tc_ext, unrollM, unrollN, problem.C, state.Cext_strategy, false, false, true, AllowFragDescNFM);

        bool useUnmasked = true;
        if (!state.copyC)
            useUnmasked &= layoutExt.match(state.C_layoutExtUnmasked);
        if (state.C_layoutExtUnmasked.blocks() == layoutExt.blocks())
            useUnmasked &= (Tc_ext.paddedSize() < 4) || (needsPseudoblock(hw, Tc_ext, unrollM, unrollN, problem.C, state.Cext_strategy, true, false)
                                                      != needsPseudoblock(hw, Tc_ext, unrollM, unrollN, problem.C, state.Cext_strategy, true, true));
        if (!useUnmasked)
            state.C_layoutExtUnmasked.clear();
    }

    if (state.Cext_strategy.atomic && (!problem.beta1() || strategy.fuseBeta)) {
        auto nonatomicCExt = state.Cext_strategy;
        nonatomicCExt.atomic = false;
        state.C_layoutExtNonatomicUnmasked = RegisterLayout::tryCreate(hw, Tc_ext, unrollM, unrollN, problem.C, nonatomicCExt, false, false, true, AllowFragDescNFM);
        bool ok = true;
        if (!state.C_layoutExtUnmasked.empty()) {
            ok = state.C_layoutExtUnmasked.match(state.C_layoutExtNonatomicUnmasked);
            if (ok && !layoutExt.match(state.C_layoutExtNonatomicUnmasked)) stub();
        } else
            ok = layoutExt.match(state.C_layoutExtNonatomicUnmasked);

        if (!ok) state.C_layoutExtNonatomicUnmasked.clear();
    }

    if (!state.copyC)
        state.C_layoutExt = state.C_layout;

    int mx = std::max(1, strategy.dotVL *  cColMajor);
    int nx = std::max(1, strategy.dotVL * !cColMajor);
    if (strategy.dotVL) {
        state.C_layoutReduced = RegisterLayout(hw, Tc, unrollM * mx, unrollN * nx, cColMajor, 1, 0, 0, true);
        std::swap(state.C_layout, state.C_layoutReduced);
    }

    if (state.A_layout.hasFragmenting(false, true) || state.B_layout.hasFragmenting(true, false)) {
        status << "Can't fragment in m/n dimensions." << status_stream::endl;
        return false;
    }

    bool globalCM = state.C_layout.colMajor();

    // Prepare to repack A/B if needed.
    int crosspackA, crosspackB, tileM_A, tileK_A, tileK_B, tileN_B;
    std::tie(crosspackA, crosspackB) = targetKernelCrosspack(hw, problem, strategy);
    std::tie(tileM_A, tileK_A, tileK_B, tileN_B) = targetKernelTiling(hw, problem, strategy);

    state.repackA |= (crosspackA && !state.A_layout.hasFullCrosspack(crosspackA));
    state.repackB |= (crosspackB && !state.B_layout.hasFullCrosspack(crosspackB));

    state.repackA |= !state.A_layout.hasTiling(tileM_A, tileK_A);
    state.repackB |= !state.B_layout.hasTiling(tileK_B, tileN_B);

    state.repackA |= (Ta.bits() != Ta_ext.bits() || Ta.components() != Ta_ext.components()) && !slmA;
    state.repackB |= (Tb.bits() != Tb_ext.bits() || Tb.components() != Tb_ext.components()) && !slmB;

    state.repackA |= problem.earlyDequantizeA() && !slmA;
    state.repackB |= problem.earlyDequantizeB() && !slmB;

    if (crosspackA == 0) crosspackA = 1;
    if (crosspackB == 0) crosspackB = 1;

    bool splitA = false, splitB = false;

    state.ka_repack = strategy.ka_load;
    if (state.repackA) {
        // Repacked data can use significantly more registers than the loaded
        // data. Lazy repacking can reduce register utilization and improve load
        // pipelining at (in some cases) the expense of more work.
        // u3 is excluded: its packed 3-byte groups can't be sliced at arbitrary k.
        bool lazyRepack = state.Ta_load.isSubByte() && !state.Ta_load.is3()
                && one_of(Ta, {Type::f16, Type::bf16, Type::f32});    // Other cases are unimplemented
        if (lazyRepack)
            state.ka_repack = std::min(state.ka_repack, strategy.kb_load);
        int repackN = state.ka_repack;
        if (problem.aqGroupK % state.ka_repack != 0 && state.ka_repack % problem.aqGroupK != 0){
                state.ka_repack = gcd(problem.aqGroupK, state.ka_repack);
                repackN = std::max(state.ka_repack, outerProductCount(problem, strategy));
        }
	    state.Ar_layout = RegisterLayout(hw, Ta, unrollM, repackN, state.A_layout.colMajor(), crosspackA, tileM_A, tileK_A, true, splitA);
    }

    if (state.repackB) {
	    state.Br_layout = RegisterLayout(hw, Tb, strategy.kb_load, unrollN, state.B_layout.colMajor(), crosspackB, tileK_B, tileN_B, true, splitB);
    }

    int opCountAB = outerProductCount(problem, strategy);
    if ((state.repackA ? state.Ar_layout.cols() : state.A_layout.cols()) < opCountAB)
        stub("A k-granularity too small for outer product count");
    if ((state.repackB ? state.Br_layout.rows() : state.B_layout.rows()) < opCountAB)
        stub("B k-granularity too small for outer product count");

    // Prepare to repack C if needed, and choose repack tile size.
    if (Tc != Tc_compute || problem.forceLateQuant(minOuterProductCount(problem, strategy))) {
        auto &period = state.cRepackPeriod;
        int panel = strategy.cRepackPanel;
        if (panel == 0)
            panel = 2 * elementsPerGRF(hw, Tc_compute);

        int Cr_unrollM = unrollM, Cr_unrollN = unrollN;
        auto &Cr_unrollX = globalCM ? Cr_unrollM : Cr_unrollN;

        if (Cr_unrollX <= panel) {
            // Repack full tiles.
            if (problem.aScale2D() && problem.bScale2D())
                period = gcd(problem.aqGroupK, problem.bqGroupK);
            else if (problem.aScale2D())
                period = problem.aqGroupK;
            else if (problem.bScale2D())
                period = problem.bqGroupK;
            else
                period = strategy.repackC ? strategy.repackC : strategy.unroll[LoopK];
        } else {
            // Repack partial tiles, interleaved with computation.
            Cr_unrollX = panel;
            period = outerProductCount(problem, strategy);
        }

        if (strategy.kInterleave)
            period = gcd(period, strategy.kInterleaveChunk);

        // C is repacked (and late scales applied) every `period` k-elements, but
        // advances in units of opCountAB. If opCountAB does not divide period,
        // scale groups can be silently skipped.
        if (period % opCountAB != 0)
            stub("C repack period incompatible with outer product count");

        state.Cr_layout = RegisterLayout(hw, Tc_compute, Cr_unrollM * mx, Cr_unrollN * nx, globalCM, 1, strategy.C.tileR, strategy.C.tileC, true);
    }

    // Prepare layouts for row/column sum calculation.
    if (problem.needsASums()) {
        state.systolicSumA = strategy.systolic && globalCM;
        state.slmASums = slmA && !state.systolicSumA && !usesLateScale(problem, strategy, true);

        if (!state.slmASums && !globalCM && strategy.dpasw) stub();  /* don't have full A data */

        auto As_srcLayout = state.slmASums ? state.Ao_layout :
                             state.repackA ? state.Ar_layout :
                                             state.A_layout;
        makeSumLayout(false, As_srcLayout, Tc_compute, state.As_layout, strategy, state, state.systolicSumA);
        if (Tc != Tc_compute) {
            std::swap(state.Asr_layout, state.As_layout);   /* TODO: trim down */
            state.As_layout = RegisterLayout(hw, Tc, unrollM, 1, true, 1);
        }
        if (state.systolicSumA)
            setupTeardownAccumulateSumSystolic(true, Tb, problem, strategy, state);
    }
    if (problem.needsBSums()) {
        state.systolicSumB = strategy.systolic && !globalCM;
        state.slmBSums = slmB && !state.systolicSumB && !usesLateScale(problem, strategy, false);

        if (!state.slmBSums && globalCM && strategy.dpasw) stub();

        auto Bs_srcLayout = state.slmBSums ? state.Bo_layout :
                             state.repackB ? state.Br_layout :
                                             state.B_layout;
        makeSumLayout(true, Bs_srcLayout, Tc_compute, state.Bs_layout, strategy, state, state.systolicSumB);
        if (Tc != Tc_compute) {
            std::swap(state.Bsr_layout, state.Bs_layout);
            state.Bs_layout = RegisterLayout(hw, Tc, 1, unrollN, false, 1);
        }
        if (state.systolicSumB)
            setupTeardownAccumulateSumSystolic(true, Ta, problem, strategy, state);
    }

    // Prepare strategies and layouts for 2D A/B grouped quantization parameters (offsets and scales).
    bool as2D = problem.aScale2D();
    bool bs2D = problem.bScale2D();
    bool ao2D = problem.aOffset2D();
    bool bo2D = problem.bOffset2D();
    bool ag2D = problem.needsAGroupSums();
    bool bg2D = problem.needsBGroupSums();
    bool aoTo2D = problem.aOffset == ABOffset::Calc && !ao2D && (problem.earlyDequantizeA() || bg2D);
    bool boTo2D = problem.bOffset == ABOffset::Calc && !bo2D && (problem.earlyDequantizeB() || ag2D);

    for (bool isA : {true, false})
        gemmMake2DQuantizationLayouts(isA, problem, strategy, state);

    gemmCalcQuantizationIncrements(problem, strategy, state);

    if (problem.hasCMXScale()){
        MatrixAddressing Cs = problem.C;
        Cs.crosspack = 1;
        Cs.tileR = Cs.tileC = 0;
        MatrixAddressingStrategy Cs_strategy = strategy.C;
        Cs_strategy.accessType = AccessType::Block;
        Cs_strategy.tileR = Cs_strategy.tileC = 0;
        Cs_strategy.dpasw = false;
        Cs_strategy.accessType = gemmstone::AccessType::Scattered;
        state.C_scaleLayout = RegisterLayout::tryCreate(hw, Type::f8_e8m0,(unrollM / problem.cqGroupM), (unrollN / problem.cqGroupN),  Cs, Cs_strategy, true, true, true);
    }

    // Grab flag registers now for named barriers. TODO: unlock these.
    if (strategy.needsNamedBarriersM(problem))
        state.barrierM = state.raVFlag.allocSubreg0();
    if (strategy.needsNamedBarriersN(problem))
        state.barrierN = state.raVFlag.allocSubreg0();

    // Round up needed A/B flag registers; hold off on C.
    // Try first without virtual flags and retry if needed.
    // m/n cooperative SLM copies may use k masking; skip those masks for now.
    auto &masks     = state.AB_masks;
    auto &masksCoop = state.AB_masksCoop;
    auto &A_cmasks = (state.effCoopA == CoopSplit::K) ? masks : masksCoop;
    auto &B_cmasks = (state.effCoopB == CoopSplit::K) ? masks : masksCoop;

    auto assignAllMasks = [&]() {
        return assignMasks(state.A_layout,       LoopM,    LoopK,       masks, strategy, state)
            && assignMasks(state.A_offsetLayout, LoopM,    LoopK,       masks, strategy, state)
            && assignMasks(state.A_scaleLayout,  LoopM,    LoopK,       masks, strategy, state)
            && assignMasks(state.Ag_layout,      LoopM,    LoopK,       masks, strategy, state)
            && assignMasks(state.Ap_layout,      LoopM,    LoopK,    A_cmasks, strategy, state)
            && assignMasks(state.Ai_layout,      LoopM,    LoopNone, A_cmasks, strategy, state)
            && assignMasks(state.B_layout,       LoopK,    LoopN,       masks, strategy, state)
            && assignMasks(state.B_offsetLayout, LoopK,    LoopN,       masks, strategy, state)
            && assignMasks(state.B_scaleLayout,  LoopK,    LoopN,       masks, strategy, state)
            && assignMasks(state.Bg_layout,      LoopK,    LoopN,       masks, strategy, state)
            && assignMasks(state.Bp_layout,      LoopK,    LoopN,    B_cmasks, strategy, state)
            && assignMasks(state.Bi_layout,      LoopNone, LoopN,    B_cmasks, strategy, state)
        ;
    };

    state.lateKLoopCheck = false;
    bool success = assignAllMasks();
    if (!success && !state.vflagsEnabled()) {
        status << "Retrying with virtual flags." << status_stream::endl;
        allocVFlagStorage(strategy, state, false);
        success = assignAllMasks();
        state.lateKLoopCheck = true;
    }

    if (!success) return false;

    loadMasks(masks,        state.remainders,        strategy, state);
    loadMasks(masksCoop,    state.remaindersCoop,    strategy, state);

    state.remainders[LoopM] = saveRemM;
    state.remainders[LoopN] = saveRemN;

    if (!state.simd32KMasks)
        releaseCoopRemainders(state);     /* may need SLM m/n remainders for k masking later */

    // Apply panel masks, if defined, to all A/B blocks.
    if (state.panelMaskA.isValid()) {
        (slmA ? state.Ai_layout : state.A_layout).assignUniformMask(state.panelMaskA);
        state.Ap_layout.assignUniformMask(state.panelMaskA);
    }
    if (state.panelMaskB.isValid()) {
        (slmB ? state.Bi_layout : state.B_layout).assignUniformMask(state.panelMaskB);
        state.Bp_layout.assignUniformMask(state.panelMaskB);
    }

    // Temporary: move add64 out of the way (later: general cramming).
    if (state.add64.isValid()) {
        auto oldAdd64 = state.add64;
        state.ra.safeRelease(state.add64);
        state.add64 = state.ra.alloc_sub<uint32_t>();
        if (oldAdd64 != state.add64)
            mov(1, state.add64, oldAdd64);
    }

    // Allocate data registers.
    gemmAllocRegs(problem, strategy, state);
    gemmAllocAoBoRegs(strategy, state);

    // Allocate address registers for A/B loads. We don't need C addresses yet.
    allocAddrRegs(state.A_addrs, state.A_layout, state);
    allocAddrRegs(state.B_addrs, state.B_layout, state);
    allocAddrRegs(state.Ap_addrs, state.Ap_layout, state);
    allocAddrRegs(state.Bp_addrs, state.Bp_layout, state);
    allocAddrRegs(state.Ai_addrs, state.Ai_layout, state);
    allocAddrRegs(state.Bi_addrs, state.Bi_layout, state);
    allocAddrRegs(state.Ao_addrs, state.Ao_layout, state);
    allocAddrRegs(state.Bo_addrs, state.Bo_layout, state);
    allocAddrRegs(state.A_offsetAddrs, state.A_offsetLayout, state);
    allocAddrRegs(state.B_offsetAddrs, state.B_offsetLayout, state);
    allocAddrRegs(state.A_scaleAddrs, state.A_scaleLayout, state);
    allocAddrRegs(state.B_scaleAddrs, state.B_scaleLayout, state);
    allocAddrRegs(state.C_scaleAddrs, state.C_scaleLayout, state);
    allocAddrRegs(state.Ag_addrs, state.Ag_layout, state);
    allocAddrRegs(state.Bg_addrs, state.Bg_layout, state);
    if (strategy.pfaux) {
        if (strategy.prefetchA) {
            allocAddrRegs(state.Ap_offsetAddrs, state.A_offsetLayout, state);
            allocAddrRegs(state.Ap_scaleAddrs,  state.A_scaleLayout,  state);
            allocAddrRegs(state.Agp_addrs,      state.Ag_layout,       state);
        }
        if (strategy.prefetchB) {
            allocAddrRegs(state.Bp_offsetAddrs, state.B_offsetLayout, state);
            allocAddrRegs(state.Bp_scaleAddrs,  state.B_scaleLayout,  state);
            allocAddrRegs(state.Bgp_addrs,      state.Bg_layout,      state);
        }
    }

    // Free up some C registers temporarily for use in address calculations.
    releaseRanges(state.C_regs, state);

    // Set up address registers.
    gemmCacheLDABMultiples(problem, strategy, state);
    setupAddr(state.Ap_addrs, state.effAp, state.Ap_layout, state.inputs.lda, strategy, state, Ap_params, state.ldaMultiples);
    setupAddr(state.Bp_addrs, state.effBp, state.Bp_layout, state.inputs.ldb, strategy, state, Bp_params, state.ldbMultiples);
    setupAddr(state.Ai_addrs, state.effAi, state.Ai_layout, state.inputs.lda, strategy, state, Ai_params, state.ldaMultiples);
    setupAddr(state.Bi_addrs, state.effBi, state.Bi_layout, state.inputs.ldb, strategy, state, Bi_params, state.ldbMultiples);
    setupAddr(state.Ao_addrs, state.effAo, state.Ao_layout, Subregister(),    strategy, state);
    setupAddr(state.Bo_addrs, state.effBo, state.Bo_layout, Subregister(),    strategy, state);
    setupAddr(state.A_addrs,  state.effA,  state.A_layout,  state.inputs.lda, strategy, state, A_params,  state.ldaMultiples);
    setupAddr(state.B_addrs,  state.effB,  state.B_layout,  state.inputs.ldb, strategy, state, B_params,  state.ldbMultiples);

    // 2D quantization address setup.
    auto i0q = state.i0, j0q = state.j0;
    Subregister A_h0q, B_h0q;

    if (state.h0.isValid()) {
        if (ao2D || as2D || ag2D) {
            A_h0q = state.ra.alloc_sub<uint32_t>();
            divDown(A_h0q, state.h0, problem.aqGroupK, strategy, state);
        }
        if (bo2D || bs2D || bg2D) {
            B_h0q = state.ra.alloc_sub<uint32_t>();
            divDown(B_h0q, state.h0, problem.bqGroupK, strategy, state);
        }
    }

    bool lateOffsetA = bg2D;
    bool lateOffsetB = ag2D;

    auto i0qLate = i0q, j0qLate = j0q;
    auto A_h0qLate = A_h0q, B_h0qLate = B_h0q;
    auto cMX_i0q = i0q, cMX_j0q = j0q;

    if (slmA && (((ao2D || aoTo2D) && !lateOffsetA) || (as2D && !state.lateScale2DA))) {
        if (state.ma_slm < unrollM) {
            if (state.ma_slm * strategy.wg[LoopN] != unrollM) stub();
            i0q = state.ra.alloc_sub<uint32_t>();
            emad(1, i0q, state.i0, state.lidN, state.ma_slm, strategy, state);
        }
        if (((ao2D && !lateOffsetA) || (as2D && !state.lateScale2DA)) && state.ka_slm < strategy.unrollKSLM && problem.aqGroupK < strategy.unrollKSLM) {
            if (state.lateScale2DA || lateOffsetA || ag2D)
                A_h0qLate = copySubregister(A_h0q, state);
            if (A_h0q.isInvalid()) {
                A_h0q = state.ra.alloc_sub<uint32_t>();
                mov(1, A_h0q, 0);
            }
            addScaled(1, A_h0q, A_h0q, state.lidN, state.ka_slm, problem.aqGroupK, state, true);
        }
    }
    if (slmB && (((bo2D || boTo2D) && !lateOffsetB) || (bs2D && !state.lateScale2DB))) {
        if (state.nb_slm < unrollN) {
            if (state.nb_slm * strategy.wg[LoopM] != unrollN) stub();
            j0q = state.ra.alloc_sub<uint32_t>();
            emad(1, j0q, state.j0, state.lidM, state.nb_slm, strategy, state);
        }
        if (((bo2D && !lateOffsetB) || (bs2D && !state.lateScale2DB)) && state.kb_slm < strategy.unrollKSLM && problem.bqGroupK < strategy.unrollKSLM) {
            if (state.lateScale2DB || lateOffsetB || bg2D)
                B_h0qLate = copySubregister(B_h0q, state);
            if (B_h0q.isInvalid()) {
                B_h0q = state.ra.alloc_sub<uint32_t>();
                mov(1, B_h0q, 0);
            }
            addScaled(1, B_h0q, B_h0q, state.lidM, state.kb_slm, problem.bqGroupK, state, true);
        }
    }

    if (problem.aqGroupM > 1 && (ao2D || as2D)) {
        auto inI0Q = i0q, inI0S = i0qLate;
        if (i0q == state.i0) i0q = state.ra.alloc_sub<uint32_t>();
        divDown(i0q, inI0Q, problem.aqGroupM, strategy, state);
        if (inI0S == inI0Q)
            i0qLate = i0q;
        else
            divDown(i0qLate, i0qLate, problem.aqGroupM, strategy, state);
    }

    if (problem.bqGroupN > 1 && (bo2D || bs2D)) {
        auto inJ0Q = j0q, inJ0S = j0qLate;
        if (j0q == state.j0) j0q = state.ra.alloc_sub<uint32_t>();
        divDown(j0q, inJ0Q, problem.bqGroupN, strategy, state);
        if (inJ0S == inJ0Q)
            j0qLate = j0q;
        else
            divDown(j0qLate, j0qLate, problem.bqGroupN, strategy, state);
    }

    auto setupQAddr = [&](Type T, vector<GRFRange> &addrs, const RegisterLayout &layout,
                          Subregister ptr, Subregister r0, Subregister c0, Subregister ld,
                          Subregister base = Subregister())
    {
        auto tmpBase = state.ra.alloc_sub(ptr.getType());
        if (!isColMajor(layout.addressing().layout)) std::swap(r0, c0);
        if (r0.isValid()) eaddScaled(1, tmpBase, ptr, r0, T, strategy, state);
        if (c0.isValid()) emad(1, tmpBase, r0.isValid() ? tmpBase : ptr, c0, ld, strategy, state);
        if (r0.isInvalid() && c0.isInvalid()) emov(1, tmpBase, ptr, strategy, state);
        if (base.isValid()) eadd(1, tmpBase, tmpBase, base, strategy, state);
        setupAddr(addrs, tmpBase, layout, ld, strategy, state);
        state.ra.safeRelease(tmpBase);
    };

    if (ao2D) {
        auto &i0o   = lateOffsetA ? i0qLate   : i0q;
        auto &A_h0o = lateOffsetA ? A_h0qLate : A_h0q;
        setupQAddr(Tao, state.A_offsetAddrs, state.A_offsetLayout, state.inputs.aoPtr,
                   i0o, A_h0o, state.inputs.ldao, state.offsetAo);
        if (strategy.pfaux && strategy.prefetchA)
            setupQAddr(Tao, state.Ap_offsetAddrs, state.A_offsetLayout, state.inputs.aoPtr,
                       i0o, A_h0o, state.inputs.ldao, state.offsetAo);
    }
    if (as2D) {
        auto &i0s   = state.lateScale2DA ? i0qLate : i0q;
        auto &A_h0s = state.lateScale2DA ? A_h0qLate : A_h0q;
        setupQAddr(Ta_scale, state.A_scaleAddrs, state.A_scaleLayout, state.inputs.aScalePtr,
                   i0s, A_h0s, state.inputs.ldaScale, state.offsetAs);
        if (strategy.pfaux && strategy.prefetchA)
            setupQAddr(Ta_scale, state.Ap_scaleAddrs, state.A_scaleLayout, state.inputs.aScalePtr,
                       i0s, A_h0s, state.inputs.ldaScale, state.offsetAs);
    }
    if (ag2D) {
        setupQAddr(Tag, state.Ag_addrs, state.Ag_layout, state.inputs.agPtr,
                   i0qLate, A_h0qLate, state.inputs.ldag, state.inputs.offsetAg);
        if (strategy.pfaux && strategy.prefetchA)
            setupQAddr(Tag, state.Agp_addrs, state.Ag_layout, state.inputs.agPtr,
                       i0qLate, A_h0qLate, state.inputs.ldag, state.inputs.offsetAg);
    }
    if (bo2D) {
        auto &j0o   = lateOffsetB ? j0qLate   : j0q;
        auto &B_h0o = lateOffsetB ? B_h0qLate : B_h0q;
        setupQAddr(Tbo, state.B_offsetAddrs, state.B_offsetLayout, state.inputs.boPtr,
                   B_h0o, j0o, state.inputs.ldbo, state.offsetBo);
        if (strategy.pfaux && strategy.prefetchB)
            setupQAddr(Tbo, state.Bp_offsetAddrs, state.B_offsetLayout, state.inputs.boPtr,
                       B_h0o, j0o, state.inputs.ldbo, state.offsetBo);
    }
    if (bs2D) {
        auto &j0s   = state.lateScale2DB ? j0qLate   : j0q;
        auto &B_h0s = state.lateScale2DB ? B_h0qLate : B_h0q;
        setupQAddr(Tb_scale, state.B_scaleAddrs, state.B_scaleLayout, state.inputs.bScalePtr,
                   B_h0s, j0s, state.inputs.ldbScale, state.offsetBs);
        if (strategy.pfaux && strategy.prefetchB)
            setupQAddr(Tb_scale, state.Bp_scaleAddrs, state.B_scaleLayout, state.inputs.bScalePtr,
                       B_h0s, j0s, state.inputs.ldbScale, state.offsetBs);
    }
    if (bg2D) {
        setupQAddr(Tbg, state.Bg_addrs, state.Bg_layout, state.inputs.bgPtr,
                   B_h0qLate, j0qLate, state.inputs.ldbg, state.inputs.offsetBg);
        if (strategy.pfaux && strategy.prefetchB)
            setupQAddr(Tbg, state.Bgp_addrs, state.Bg_layout, state.inputs.bgPtr,
                       B_h0qLate, j0qLate, state.inputs.ldbg, state.inputs.offsetBg);
    }

    if (problem.hasCMXScale()) {
        auto i0qs = state.ra.alloc_sub(cMX_j0q.getType(), getHint(HintType::LongTerm, strategy));
        divDown(i0qs, cMX_i0q, problem.cqGroupM, strategy, state);
        setupQAddr(Type::u8, state.C_scaleAddrs, state.C_scaleLayout, state.inputs.cScalePtr,
               i0qs, cMX_j0q, state.inputs.ldcScale, state.offsetCs);
    }

    if (i0qLate != state.i0) state.ra.safeRelease(i0qLate);
    if (j0qLate != state.j0) state.ra.safeRelease(j0qLate);
    state.ra.safeRelease(A_h0q);
    state.ra.safeRelease(B_h0q);
    state.ra.safeRelease(A_h0qLate);
    state.ra.safeRelease(B_h0qLate);

    // Load and convert 0D/1D offsets for 2D dequantization.
    if (aoTo2D) {
        RegisterLayout A_offsetLayout;
        GRFRange aoLoad;
        bool releaseAOLoad = true;
        if (problem.aoPtrDims == 1) {
            bool slmAO = slmA && !lateOffsetA;
            reclaimRanges(state.C_regs, state);
            auto aoBase = state.ra.alloc_sub<uint64_t>();
            if (problem.batch == BatchMode::Strided) {
                eadd(1, aoBase, state.inputs.aoPtr, state.offsetAo, strategy, state);
                eaddScaled(1, aoBase, aoBase, slmAO ? i0q : state.i0, problem.Tao, strategy, state);
            } else {
                eaddScaled(1, aoBase, state.inputs.aoPtr, slmAO ? i0q : state.i0, problem.Tao, strategy, state);
            }
            auto rem = slmAO ? state.remaindersCoop[LoopM] : state.remainders[LoopM];
            auto r = slmAO ? state.ma_slm : strategy.unroll[LoopM];
            aoLoad = loadVector(problem.Tao, problem.Tao, aoBase, r, rem, strategy, state);
            A_offsetLayout = RegisterLayout(hw, problem.Tao, r, 1, true);
            state.ra.safeRelease(aoBase);
        } else if (problem.aoPtrDims == 0) {
            auto grf = loadScalars(problem.Tao, {state.inputs.aoPtr}, strategy, state);
            aoLoad = grf-grf;
            A_offsetLayout = state.Ar_offsetLayout;
            A_offsetLayout.cast(problem.Tao);
        } else {
            GRF grf{state.inputs.ao.getBase()};
            aoLoad = grf-grf;
            releaseAOLoad = false;
            A_offsetLayout = state.Ar_offsetLayout;
            A_offsetLayout.cast(problem.Tao);
            A_offsetLayout[0].offsetBytes = state.inputs.ao.getByteOffset();
        }
        gemmRepack2DOffsetData(problem.Ta_ext, A_offsetLayout, state.Ar_offsetLayout, aoLoad, state.Ar_offsetRegs, problem, strategy, state, problem.aOffsetAfterScale);
        if (releaseAOLoad) state.ra.safeRelease(aoLoad);
        if (!strategy.persistentLoop())
            state.ra.safeRelease(state.inputs.aoPtr);
    }
    if (boTo2D) {
        RegisterLayout B_offsetLayout;
        GRFRange boLoad;
        bool releaseBOLoad = true;
        if (problem.boPtrDims == 1) {
            bool slmBO = slmB && !lateOffsetB;
            reclaimRanges(state.C_regs, state);
            auto boBase = state.ra.alloc_sub<uint64_t>();
            auto rem = slmBO ? state.remaindersCoop[LoopN] : state.remainders[LoopN];
            auto c = slmBO ? state.nb_slm : strategy.unroll[LoopN];
            if (problem.batch == BatchMode::Strided) {
                eadd(1, boBase, state.inputs.boPtr, state.offsetBo, strategy, state);
                eaddScaled(1, boBase, boBase, slmBO ? j0q : state.j0, Tbo, strategy, state);
            } else {
                eaddScaled(1, boBase, state.inputs.boPtr, slmBO ? j0q : state.j0, Tbo, strategy, state);
            }
            boLoad = loadVector(problem.Tbo, problem.Tbo, boBase, c, rem, strategy, state);
            B_offsetLayout = RegisterLayout(hw, problem.Tbo, 1, c, false);
            state.ra.safeRelease(boBase);
        } else if (problem.boPtrDims == 0) {
            auto grf = loadScalars(problem.Tbo, {state.inputs.boPtr}, strategy, state);
            boLoad = grf-grf;
            B_offsetLayout = state.Br_offsetLayout;
            B_offsetLayout.cast(problem.Tbo);
        } else {
            GRF grf{state.inputs.bo.getBase()};
            boLoad = grf-grf;
            releaseBOLoad = false;
            B_offsetLayout = state.Br_offsetLayout;
            B_offsetLayout.cast(problem.Tbo);
            B_offsetLayout[0].offsetBytes = state.inputs.bo.getByteOffset();
        }
        gemmRepack2DOffsetData(problem.Tb_ext, B_offsetLayout, state.Br_offsetLayout, boLoad, state.Br_offsetRegs, problem, strategy, state, problem.bOffsetAfterScale);
        if (releaseBOLoad) state.ra.safeRelease(boLoad);
        if (!strategy.persistentLoop())
            state.ra.safeRelease(state.inputs.boPtr);
    }
    if (i0q != state.i0) state.ra.safeRelease(i0q);
    if (j0q != state.j0) state.ra.safeRelease(j0q);

    // Free unneeded registers after address setup.
    if (!state.isNested) {
        if (strategy.A.address2D && (!strategy.prefetchA || strategy.A_prefetch.address2D) && !strategy.l3PrefetchA)
            state.ra.safeRelease(state.inputs.lda);
        if (strategy.B.address2D && (!strategy.prefetchB || strategy.B_prefetch.address2D) && !strategy.l3PrefetchB)
            state.ra.safeRelease(state.inputs.ldb);
        if (!strategy.C.address2D && (!strategy.prefetchC || !strategy.C_prefetch.address2D) && !keepIJ0(problem, strategy)) {
            state.ra.safeRelease(state.i0);
            state.ra.safeRelease(state.j0);
        }
        if (!keepH0(problem, strategy))
            state.ra.safeRelease(state.h0);
    }

    if (!needsKLoopReset(problem)) {
        if (state.Ai_strategy.address2D) {
            if (Ai_params.offR != A_params.offR) state.ra.safeRelease(Ai_params.offR);
            if (Ai_params.offC != A_params.offC) state.ra.safeRelease(Ai_params.offC);
        }
        if (state.Bi_strategy.address2D) {
            if (Bi_params.offR != B_params.offR) state.ra.safeRelease(Bi_params.offR);
            if (Bi_params.offC != B_params.offC) state.ra.safeRelease(Bi_params.offC);
        }
        if (strategy.A_prefetch.address2D) {
            if (Ap_params.offR != A_params.offR) state.ra.safeRelease(Ap_params.offR);
            if (Ap_params.offC != A_params.offC) state.ra.safeRelease(Ap_params.offC);
        }
        if (strategy.B_prefetch.address2D) {
            if (Bp_params.offR != B_params.offR) state.ra.safeRelease(Bp_params.offR);
            if (Bp_params.offC != B_params.offC) state.ra.safeRelease(Bp_params.offC);
        }

        if (!one_of(state.effAp, {state.effA, state.effAi})) state.ra.safeRelease(state.effAp);
        if (!one_of(state.effBp, {state.effB, state.effBi})) state.ra.safeRelease(state.effBp);
    }

    releaseLDMultiples(state.ldaMultiples, state);
    releaseLDMultiples(state.ldbMultiples, state);
    releaseIndexVec(state);

    reclaimRanges(state.C_regs, state);

    // Allocate tokens.
    if (strategy.tokenAlloc)
        gemmAllocateTokens(problem, strategy, state);

    // Preloading C and fused beta scaling need some extra registers for C headers.
    // Temporarily free up A/B data registers for that purpose.
    releaseRanges(state.A_regs, state);
    releaseRanges(state.B_regs, state);
    releaseRanges(state.Ar_regs, state);
    releaseRanges(state.Br_regs, state);

    // Load C now if configured (and perform beta scaling).
    if (cLoadAhead) {
        if (problem.checkBeta0 && !problem.beta.fixed())   stub();
        if (state.C_accCount > 0)                          stub();
        if (strategy.kParallelLocal)                       stub();

        status << "Loading C" << status_stream::endl;
        gemmAccessC(COperation::Load, problem, strategy, state);

        gemmBetaScale(problem, strategy, state);
    }

    // Beta prescaling for atomic kernels.
    if (strategy.fuseBeta)
        gemmFusedBetaScale(problem, strategy, state);

    // Reclaim A/B data registers.
    reclaimRanges(state.Br_regs, state);
    reclaimRanges(state.Ar_regs, state);
    reclaimRanges(state.B_regs, state);
    reclaimRanges(state.A_regs, state);

    for (int q = 0; q < state.C_count; q++)
        releaseLDMultiples(state.ldcMultiples[q], state);
    releaseIndexVec(state);

    // Workaround for Xe3 HW bug: dpas with uninitialized src1/src2 can cause
    //  data corruption in other threads.
    if (hw == HW::Xe3 && strategy.systolic) {
        if (state.A_layout.hasFlags() && state.Ar_regs.empty())
            for (auto &A_regs: state.A_regs)
                zeroMatrix(A_regs, strategy);
        if (state.B_layout.hasFlags() && state.Br_regs.empty())
            for (auto &B_regs: state.B_regs)
                zeroMatrix(B_regs, strategy);
    }

    // Release 64-bit emulation registers as they aren't needed in the inner loop.
    // Could also move r0 to acc here.
    if (state.emulate.temp[0].isValid()) {
        for (int q = 0; q < 2; q++) {
            state.emulate64TempSave[q] = state.emulate.temp[q];
            state.ra.safeRelease(state.emulate.temp[q]);
        }
        if (strategy.emulate.emulate64 && !strategy.emulate.emulate64_add32) {
            if (GRF::bytes(hw) == 64) {
                state.emulate.flag = simd16EmulationFlag;
                state.emulate.flagOffset = 0;
            } else {
                state.emulate.flag = state.flagAP;
                state.emulate.flagOffset = 8;
                state.lateKLoopCheck = false;
            }
        }
    }

    return true;
}

template <HW hw>
void Generator<hw>::gemmAccumulateCTeardown(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    // We're done with A and B. Free their address, data, and flag registers.
    // Also done with loop counter.
    safeReleaseMaskAssignments(state.AB_masks, state);
    safeReleaseMaskAssignments(state.AB_masksCoop, state);
    safeReleaseRanges(state.A_addrs, state);
    safeReleaseRanges(state.B_addrs, state);
    safeReleaseRanges(state.A_addrsAlt, state);
    safeReleaseRanges(state.B_addrsAlt, state);
    safeReleaseRanges(state.Ai_addrs, state);
    safeReleaseRanges(state.Bi_addrs, state);
    safeReleaseRanges(state.Ao_addrs, state);
    safeReleaseRanges(state.Bo_addrs, state);
    safeReleaseRanges(state.Ap_addrs, state);
    safeReleaseRanges(state.Bp_addrs, state);
    safeReleaseRanges(state.Ap_addrsAlt, state);
    safeReleaseRanges(state.Bp_addrsAlt, state);
    safeReleaseRanges(state.A_offsetAddrs, state);
    safeReleaseRanges(state.B_offsetAddrs, state);
    safeReleaseRanges(state.A_scaleAddrs, state);
    safeReleaseRanges(state.B_scaleAddrs, state);
    safeReleaseRanges(state.Ag_addrs, state);
    safeReleaseRanges(state.Bg_addrs, state);
    safeReleaseRanges(state.Ap_offsetAddrs, state);
    safeReleaseRanges(state.Bp_offsetAddrs, state);
    safeReleaseRanges(state.Ap_scaleAddrs, state);
    safeReleaseRanges(state.Bp_scaleAddrs, state);
    safeReleaseRanges(state.Agp_addrs, state);
    safeReleaseRanges(state.Bgp_addrs, state);

    safeReleaseRanges(state.A_regs, state);
    safeReleaseRanges(state.Ar_regs, state);
    safeReleaseRanges(state.Ai_regs, state);
    safeReleaseRanges(state.Ao_regs, state);
    safeReleaseRanges(state.Ap_regs, state);
    safeReleaseRanges(state.A_offsetRegs, state);
    safeReleaseRanges(state.A_scaleRegs, state);
    safeReleaseRanges(state.Ag_regs, state);
    safeReleaseRanges(state.Ar_offsetRegs, state);
    safeReleaseRanges(state.Ar_scaleRegs, state);
    safeReleaseRanges(state.Agr_regs, state);
    safeReleaseRanges(state.B_regs, state);
    safeReleaseRanges(state.Br_regs, state);
    safeReleaseRanges(state.Bi_regs, state);
    safeReleaseRanges(state.Bo_regs, state);
    safeReleaseRanges(state.Bp_regs, state);
    safeReleaseRanges(state.B_offsetRegs, state);
    safeReleaseRanges(state.B_scaleRegs, state);
    safeReleaseRanges(state.Bg_regs, state);
    safeReleaseRanges(state.Br_offsetRegs, state);
    safeReleaseRanges(state.Br_scaleRegs, state);
    safeReleaseRanges(state.Bgr_regs, state);
    safeReleaseRanges(state.tempMul_regs, state);
    clearTokenAllocations(hw, state);
    releaseCoopRemainders(state);

    deduplicateScalar(state.lda, state);
    deduplicateScalar(state.ldb, state);

    state.raVFlag.safeRelease(state.barrierM);
    state.raVFlag.safeRelease(state.barrierN);

    if (state.effCp != state.effC[0])
        state.ra.safeRelease(state.effCp);

    state.A_layout.clear();
    state.B_layout.clear();
    state.A_layoutAlt.clear();
    state.B_layoutAlt.clear();
    state.Ai_layout.clear();
    state.Bi_layout.clear();
    state.Ao_layout.clear();
    state.Bo_layout.clear();
    state.Ar_layout.clear();
    state.Br_layout.clear();
    state.Ap_layout.clear();
    state.Bp_layout.clear();
    state.Cp_layout.clear();
    state.Ap_layoutAlt.clear();
    state.Bp_layoutAlt.clear();
    state.A_offsetLayout.clear();
    state.B_offsetLayout.clear();
    state.A_scaleLayout.clear();
    state.B_scaleLayout.clear();
    state.Ag_layout.clear();
    state.Bg_layout.clear();

    if (state.systolicSumA || state.systolicSumB)
        setupTeardownAccumulateSumSystolic(false, Type::invalid, problem, strategy, state);

    // Restore effA/B if needed.
    bool restoreEffAB = false;

    restoreEffAB |= (problem.aOffset == ABOffset::Load);
    restoreEffAB |= (problem.bOffset == ABOffset::Load);

    if (restoreEffAB) {
        Subregister aiOff, biOff;
        Subregister aiOffR, biOffR;
        Subregister aiOffC, biOffC;
        if (strategy.slmA)
            gemmCalcWorkshareAOffset(aiOff, aiOffR, aiOffC, state.Ai, state.Ai_strategy, state.ma_slm, state.ka_slm, problem, strategy, state);
        if (strategy.slmB)
            gemmCalcWorkshareBOffset(biOff, biOffR, biOffC, state.Bi, state.Bi_strategy, state.kb_slm, state.nb_slm, problem, strategy, state);
        if (strategy.slmA) eadd(1, state.effAi, state.effAi, -aiOff, strategy, state);
        if (strategy.slmB) eadd(1, state.effBi, state.effBi, -biOff, strategy, state);

        state.ra.safeRelease(aiOff);
        state.ra.safeRelease(biOff);
        state.ra.safeRelease(aiOffR);
        state.ra.safeRelease(biOffR);
        state.ra.safeRelease(aiOffC);
        state.ra.safeRelease(biOffC);
    }

    // Restore A/B addresses and strategies that were modified by SLM copies.

    if (strategy.slmA) {
        state.ra.safeRelease(state.effA);
        state.ra.safeRelease(state.effAo);
        state.effA = state.effAi;
        state.effAi = invalid;
        state.Ta_load = problem.Ta_ext;
        problem.A = state.Ai;
        strategy.A = state.Ai_strategy;
    }
    if (strategy.slmB) {
        state.ra.safeRelease(state.effB);
        state.ra.safeRelease(state.effBo);
        state.effB = state.effBi;
        state.effBi = invalid;
        state.Tb_load = problem.Tb_ext;
        problem.B = state.Bi;
        strategy.B = state.Bi_strategy;
    }

    // Put accumulators with the rest of C.
    if (state.C_accCount > 0) {
        // Reclaim the bottom registers of C.
        reclaimRanges(state.C_regs[0], state);

        auto e = elementsPerGRF<uint32_t>(hw);
        for (int i = 0; i < state.C_accCount; i += 2)
            mov<uint32_t>(2 * e, state.C_regs[0][i], AccumulatorRegister(i));
    }

    // Restore emulation registers.
    if (state.emulate64TempSave[0].isValid()) {
        for (int q = 0; q < 2; q++) {
            state.emulate.temp[q] = state.emulate64TempSave[q];
            if (state.emulate64TempSave[q].isValid())
                state.ra.claim(state.emulate64TempSave[q]);
        }
        if (GRF::bytes(hw) == 64)
            state.raVFlag.release(state.emulate.flag);
        state.emulate.flag = invalid;
        state.emulate.flagOffset = 0;
    }
}

// Calculate and cache multiples of lda/ldb used for address increments in the inner loop.
template <HW hw>
void Generator<hw>::gemmCalcIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                       int ka_load, int kb_load, bool doA, bool doB)
{
    gemmFreeIncrements(problem, strategy, state, doA, doB);

    doA &= (problem.A.layout == MatrixLayout::N);
    doB &= (problem.B.layout == MatrixLayout::T);

    if (ka_load == 0) ka_load = strategy.ka_inc();
    if (kb_load == 0) kb_load = strategy.kb_inc();

    auto calcInterleavedIncrement = [&](bool isA, int inc) {
        auto &increments = isA ? state.ldaIncrements : state.ldbIncrements;
        auto &base       = isA ? state.lda           : state.ldb;
        if (strategy.kInterleave) {
            int chunk = strategy.kInterleaveChunk;
            if (inc < chunk)
                calcIncrement(increments, base, inc, strategy, state);
            calcIncrement(increments, base, inc + chunk * (strategy.wg[LoopK] - 1), strategy, state);
        } else
            calcIncrement(increments, base, inc, strategy, state);
    };

    if (doA) {
        if (!strategy.A.address2D)
            calcInterleavedIncrement(true, ka_load);
        if (strategy.prefetchA && !strategy.A_prefetch.address2D)
            calcInterleavedIncrement(true, strategy.ka_pfStride);
    }

    if (doB) {
        if (!strategy.B.address2D)
            calcInterleavedIncrement(false, kb_load);
        if (strategy.prefetchB && !strategy.B_prefetch.address2D)
            calcInterleavedIncrement(false, strategy.kb_pfStride);
    }
}

template <HW hw>
void Generator<hw>::gemmCalcQuantizationIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool ao2D = problem.aOffset2D();
    bool bo2D = problem.bOffset2D();
    bool as2D = problem.aScale2D();
    bool bs2D = problem.bScale2D();
    bool ag2D = problem.needsAGroupSums();
    bool bg2D = problem.needsBGroupSums();

    auto calcInterleavedQIncrement = [&](bool isA, SubregisterPair &base, LDIncrements &increments) {
        auto inc   = isA ? state.kaqStride  : state.kbqStride;
        auto group = isA ? problem.aqGroupK : problem.bqGroupK;
        if (strategy.kInterleave) {
            int chunk = strategy.kInterleaveChunk;
            if (group < chunk) {
                calcIncrement(increments, base, inc, strategy, state, false);
                calcIncrement(increments, base, (inc * group + chunk * (strategy.wg[LoopK] - 1)) / group, strategy, state, false);
            } else
                calcIncrement(increments, base, chunk * strategy.wg[LoopK] / group, strategy, state, false);
        } else
            calcIncrement(increments, base, inc, strategy, state, false);
    };

    if (ao2D && problem.AO.layout == MatrixLayout::N)
        calcInterleavedQIncrement(true,  state.ldao,     state.ldaoIncrements);
    if (as2D && problem.A_scale.layout == MatrixLayout::N)
        calcInterleavedQIncrement(true,  state.ldaScale, state.ldasIncrements);
    if (ag2D && problem.Ag.layout == MatrixLayout::N)
        calcInterleavedQIncrement(true,  state.ldag,     state.ldagIncrements);
    if (bo2D && problem.BO.layout == MatrixLayout::T)
        calcInterleavedQIncrement(false, state.ldbo,     state.ldboIncrements);
    if (bs2D && problem.B_scale.layout == MatrixLayout::T)
        calcInterleavedQIncrement(false, state.ldbScale, state.ldbsIncrements);
    if (bg2D && problem.Bg.layout == MatrixLayout::T)
        calcInterleavedQIncrement(false, state.ldbg,     state.ldbgIncrements);
}

// Free cached multiples of lda/ldb.
template <HW hw>
void Generator<hw>::gemmFreeIncrements(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA, bool doB)
{
    auto freeIncrements = [&](SubregisterPair &base, LDIncrements &increments) {
        for (auto &inc: increments)
            safeRelease(inc.second, state);
        deduplicateScalar(base, state);
        state.ra.claim(base.getReg(0));
        increments.clear();
    };

    if (doA) {
        freeIncrements(state.lda, state.ldaIncrements);
        freeIncrements(state.ldao, state.ldaoIncrements);
        freeIncrements(state.ldaScale, state.ldasIncrements);
        freeIncrements(state.ldag, state.ldagIncrements);
    }

    if (doB) {
        freeIncrements(state.ldb, state.ldbIncrements);
        freeIncrements(state.ldbo, state.ldboIncrements);
        freeIncrements(state.ldbScale, state.ldbsIncrements);
        freeIncrements(state.ldbg, state.ldbgIncrements);
    }
}

// Adjust addresses for worksharing.
template <HW hw>
void Generator<hw>::gemmApplyWorkshareOffset(bool isA, Subregister &base, Subregister alias, Address2DParams &params2D,
                                             const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                             int r, int c, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    Subregister off;
    auto &offR = params2D.offR, &offC = params2D.offC;
    auto offR0 = offR, offC0 = offC;
    isA ? gemmCalcWorkshareAOffset(off, offR, offC, atype, astrategy, r, c, problem, strategy, state)
        : gemmCalcWorkshareBOffset(off, offR, offC, atype, astrategy, r, c, problem, strategy, state);
    if (astrategy.address2D) {
        if (offR0.isValid() && offR != offR0) add(1, offR, offR, offR0);
        if (offC0.isValid() && offC != offC0) add(1, offC, offC, offC0);
    } else {
        auto base0 = base;
        if (base == alias)
            base = state.ra.alloc_sub(base.getType());
        eadd(1, base, base0, off, strategy, state);
    }
    state.ra.safeRelease(off);
}

// Prepare A/B prefetch addresses.
template <HW hw>
void Generator<hw>::gemmABPrefetchAddrSetup(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool doA, bool doB)
{
    if (doA && strategy.cooperativePF && strategy.prefetchA) {
        auto &A_offR = state.A_params.offR, &Ap_offR = state.Ap_params.offR;
        auto &A_offC = state.A_params.offC, &Ap_offC = state.Ap_params.offC;
        Ap_offR = A_offR, Ap_offC = A_offC;
        gemmApplyWorkshareOffset(true, state.effAp, state.effA, state.Ap_params, problem.A, strategy.A_prefetch,
                                 state.ma_prefetch, state.ka_prefetch, problem, strategy, state);
    }

    if (doB && strategy.cooperativePF && strategy.prefetchB) {
        auto &B_offR = state.B_params.offR, &Bp_offR = state.Bp_params.offR;
        auto &B_offC = state.B_params.offC, &Bp_offC = state.Bp_params.offC;
        Bp_offR = B_offR, Bp_offC = B_offC;
        gemmApplyWorkshareOffset(false, state.effBp, state.effB, state.Bp_params, problem.B, strategy.B_prefetch,
                                 state.kb_prefetch, state.nb_prefetch, problem, strategy, state);
    }

    if (problem.backward()) {
        if (doA && strategy.prefetchA)
            gemmOffsetAk(strategy.ka_load - strategy.ka_prefetch, state.effAp, problem.A, problem, strategy, state);
        if (doB && strategy.prefetchB)
            gemmOffsetBk(strategy.kb_load - strategy.kb_prefetch, state.effBp, problem.B, problem, strategy, state);
    }
}

// Allocate tokens for k loop loads/stores.
template <HW hw>
void Generator<hw>::gemmAllocateTokens(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool success = true;
    for (int q = 0; q < strategy.A_copies; q++)
        success &= allocateTokens(state.A_layout, state.A_regs[q], state);
    for (int q = 0; q < strategy.B_copies; q++)
        success &= allocateTokens(state.B_layout, state.B_regs[q], state);
    for (int q = 0; q < strategy.slmCopies; q++) {
        if (strategy.slmA) success &= allocateTokens(state.Ai_layout, state.Ai_regs[q], state);
        if (strategy.slmB) success &= allocateTokens(state.Bi_layout, state.Bi_regs[q], state);
    }
    if (strategy.slmA && !state.aioShare)
        success &= allocateTokens(state.Ao_layout, state.Ao_regs, state);
    if (strategy.slmB && !state.bioShare)
        success &= allocateTokens(state.Bo_layout, state.Bo_regs, state);
    success &= allocateTokens(state.Ap_layout, state.Ap_regs, state, state.Ap_addrs);
    success &= allocateTokens(state.Bp_layout, state.Bp_regs, state, state.Bp_addrs);

    success &= allocateTokens(state.A_offsetLayout, state.A_offsetRegs, state);
    success &= allocateTokens(state.B_offsetLayout, state.B_offsetRegs, state);
    success &= allocateTokens(state.A_scaleLayout, state.A_scaleRegs, state);
    success &= allocateTokens(state.B_scaleLayout, state.B_scaleRegs, state);
    success &= allocateTokens(state.Ag_layout, state.Ag_regs, state);
    success &= allocateTokens(state.Bg_layout, state.Bg_regs, state);

    if (!success) {
        status << "Not enough tokens for k loop." << status_stream::endl;
        clearMappedTokenAllocations(hw, state);
    }
}

// Offset an SLM address by the base SLM address if present.
template <HW hw>
void Generator<hw>::makeSLMBaseRelative(Subregister addr, const GEMMState &state)
{
    if (state.inputs.slmBase.isValid())
        add(1, addr, addr, state.inputs.slmBase);
}

// Initialize the interface.
template <HW hw>
void Generator<hw>::gemmInitInterface(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state, bool inSK)
{
    Subregister localSize[3];
    GRF localID[3];
    Subregister tgids[3] = {r0.ud(1), r0.ud(6), r0.ud(7)};   // X, Y, Z threadgroup IDs

    initInterface(problem, strategy, state);
    if (strategy.systolic)
        interface.requireDPAS();
    if (strategy.C.atomic)
        interface.requireGlobalAtomics();
    if (strategy.barrierFreq > 0 || strategy.fuseBeta || strategy.fusePostOps)
        interface.requireBarrier();

    auto slmSize = gemmSLMSize(hw, problem, strategy);
    auto slmPerK = gemmPerKSLMSize(hw, problem, strategy);
    if (slmSize > 0 || slmPerK > 0) {
        status << "SLM usage: " << slmSize / 1024. << 'k';
        if (slmPerK)
            status << " (" << slmPerK / 1024. << "k per-k)";
        status << status_stream::endl;
        if (!slmPerK)
            interface.requireSLM(slmSize);
        interface.requireBarrier();
    }

    size_t slm = maxSLMPerWG(problem.product, strategy.GRFs);
    if (slmSize > slm || slmPerK * strategy.wg[LoopK] > slm)
        stub("Strategy requests more SLM than available");

    if (strategy.fixedWG(problem)) {
        auto wgX = strategy.wg[strategy.loopOrder[0]];
        auto wgY = strategy.wg[strategy.loopOrder[1]];
        auto wgZ = strategy.wg[strategy.loopOrder[2]];
        if (strategy.splitCopy)
            wgY *= 2;
        wgY *= strategy.wgPadFactor;
        if (wgZ <= 1)
            interface.requireWorkgroup(strategy.subgroupSize * wgX, wgY, wgZ);
    }
    interface.requireWalkOrder(0, 1, 2);

    bool needStatelessWrites = strategy.C.base.isStateless();
    if (problem.sumA || problem.sumB)
        needStatelessWrites |= strategy.CO.base.isStateless();

    interface.requireStatelessWrites(needStatelessWrites);

    int nb = int(strategy.needsNamedBarriersM(problem)) * strategy.namedBarriers[LoopM] * strategy.wg[LoopK]
           + int(strategy.needsNamedBarriersN(problem)) * strategy.namedBarriers[LoopN] * strategy.wg[LoopK];
    if (nb) {
        if (strategy.needsUnnamedBarrier(problem)) nb++;
        if (nb > 32) hw_unsupported();                    /* also causes IGC/NEO crashes */
        interface.requireBarriers(nb);
    }

    interface.finalize();

    for (int dim = 0; dim < 3; dim++) {
        localID[dim] = interface.getLocalID(dim);
        localSize[dim] = interface.getLocalSize(dim);
    }

    // Get input arguments.
    state.inputs.base = interface.getArgumentIfExists("base");
    auto baseSurface = interface.getArgumentSurfaceIfExists("base");
    if (state.inputs.base.isInvalid() && baseSurface == InterfaceHandler::noSurface) {
        state.inputs.A = interface.getArgumentIfExists("A");
        state.inputs.B = interface.getArgumentIfExists("B");
        state.inputs.C[1] = interface.getArgumentIfExists("P");
        state.inputs.surfaceA = interface.getArgumentSurfaceIfExists("A");
        state.inputs.surfaceB = interface.getArgumentSurfaceIfExists("B");
        state.inputs.surfaceC[1] = interface.getArgumentSurfaceIfExists("P");
    } else {
        state.inputs.A = state.inputs.B = state.inputs.base;
        state.inputs.surfaceA = state.inputs.surfaceB = baseSurface;
        if (interface.getArgumentIfExists("offset_P").isValid()) {
            state.inputs.C[1] = state.inputs.base;
            state.inputs.surfaceC[1] = state.inputs.surfaceA;
        }
    }

    state.inputs.C[0] = interface.getArgumentIfExists("C");
    state.inputs.surfaceC[0] = interface.getArgumentSurfaceIfExists("C");
    state.C_count = state.inputs.C[1].isValid() ? 2 : 1;
    if (problem.usesCOPtr()) {
        state.inputs.coPtr = interface.getArgumentIfExists("co_ptr");
        state.inputs.surfaceCO = interface.getArgumentSurfaceIfExists("co_ptr");
    } else if (problem.cOffsetHostScalar()) {
        state.inputs.co = interface.getArgumentIfExists("co");
    }
    if (state.useTempC) {
        state.inputs.tempC = interface.getArgumentIfExists("temp_C");
        state.inputs.surfaceTempC = interface.getArgumentSurfaceIfExists("temp_C");
    }

    bool aOffset = (problem.aOffset != ABOffset::None);
    bool bOffset = (problem.bOffset != ABOffset::None);
    if (aOffset || bOffset) {
        state.inputs.abo = interface.getArgumentIfExists("abo");
        if (state.inputs.abo.isValid()) {
            // A/B offset are two words packed into a single dword argument.
            state.inputs.ao = state.inputs.abo.w(0);
            state.inputs.bo = state.inputs.abo.w(1);
        } else {
            state.inputs.ao = interface.getArgumentIfExists("ao");
            state.inputs.bo = interface.getArgumentIfExists("bo");
        }
        state.inputs.aoPtr = interface.getArgumentIfExists("ao_ptr");
        state.inputs.boPtr = interface.getArgumentIfExists("bo_ptr");
        state.inputs.surfaceAO = interface.getArgumentSurfaceIfExists("ao_ptr");
        state.inputs.surfaceBO = interface.getArgumentSurfaceIfExists("bo_ptr");
    }
    if (problem.aScale2D()) {
        state.inputs.aScalePtr = interface.getArgumentIfExists("a_scale_ptr");
        state.inputs.surfaceAScale = interface.getArgumentSurfaceIfExists("a_scale_ptr");
    }
    if (problem.bScale2D()) {
        state.inputs.bScalePtr = interface.getArgumentIfExists("b_scale_ptr");
        state.inputs.surfaceBScale = interface.getArgumentSurfaceIfExists("b_scale_ptr");
    }
    if (problem.hasCMXScale()) {
        state.inputs.cScalePtr = interface.getArgumentIfExists("c_scale_ptr");
        state.inputs.surfaceCScale = interface.getArgumentSurfaceIfExists("c_scale_ptr");
    }
    if (problem.needsAGroupSums()) {
        state.inputs.agPtr = interface.getArgumentIfExists("ag_ptr");
        state.inputs.surfaceAg = interface.getArgumentSurfaceIfExists("ag_ptr");
    }
    if (problem.needsBGroupSums()) {
        state.inputs.bgPtr = interface.getArgumentIfExists("bg_ptr");
        state.inputs.surfaceBg = interface.getArgumentSurfaceIfExists("bg_ptr");
    }
    if (problem.postOps.cStochasticRound)
        state.inputs.sroundSeedPtr = interface.getArgument("sround_seed");

    state.inputs.offsetA = interface.getArgumentIfExists("offset_A");
    state.inputs.offsetB = interface.getArgumentIfExists("offset_B");
    state.inputs.offsetC[0] = interface.getArgumentIfExists("offset_C");
    state.inputs.offsetC[1] = interface.getArgumentIfExists("offset_P");
    state.inputs.offsetAO = interface.getArgumentIfExists("offset_AO");
    state.inputs.offsetBO = interface.getArgumentIfExists("offset_BO");
    state.inputs.offsetCO = interface.getArgumentIfExists("offset_CO");
    state.inputs.offsetAScale = interface.getArgumentIfExists("offset_A_scale");
    state.inputs.offsetBScale = interface.getArgumentIfExists("offset_B_scale");
    state.inputs.offsetAg = interface.getArgumentIfExists("offset_Ag");
    state.inputs.offsetBg = interface.getArgumentIfExists("offset_Bg");
    state.inputs.offsetAq = interface.getArgumentIfExists("offset_Aq");
    state.inputs.offsetBq = interface.getArgumentIfExists("offset_Bq");
    if (problem.batch == BatchMode::Strided) {
        for (int i = 0; i < problem.batchDims; i++) {
            auto istr = std::to_string(i);
            state.inputs.strideA.push_back(interface.getArgument("stride_A" + istr));
            state.inputs.strideB.push_back(interface.getArgument("stride_B" + istr));
            state.inputs.strideC.push_back(interface.getArgument("stride_C" + istr));
            if(problem.hasAScalePtr()){
                    state.inputs.strideScaleA.push_back(interface.getArgument("scale_stride_A" + istr));
            }
            if(problem.hasBScalePtr()){
                    state.inputs.strideScaleB.push_back(interface.getArgument("scale_stride_B" + istr));
            }
            if(problem.hasCMXScale()){
                    state.inputs.strideScaleC.push_back(interface.getArgument("scale_stride_C" + istr));
            }
            if(problem.hasAOffsetPtr()){
                    state.inputs.strideOffsetA.push_back(interface.getArgument("offset_stride_A" + istr));
            }
            if(problem.hasBOffsetPtr()){
                    state.inputs.strideOffsetB.push_back(interface.getArgument("offset_stride_B" + istr));
            }
            if(problem.needsAGroupSums()){
                    state.inputs.strideGroupSumsA.push_back(interface.getArgument("group_sums_stride_A" + istr));
            }
            if(problem.needsBGroupSums()){
                    state.inputs.strideGroupSumsB.push_back(interface.getArgument("group_sums_stride_B" + istr));
            }
            if (i < problem.batchDims - 1) {
                state.inputs.batchSize.push_back(interface.getArgument("batch_size" + istr));
                state.inputs.recipBatchSize.push_back(interface.getArgument("recip_batch_size" + istr));
            }
        }
    } else if (problem.batch == BatchMode::Nonstrided)
        state.inputs.offsetBatch = interface.getArgumentIfExists("offset_batch");
    else if (problem.batch == BatchMode::Variable) {
        state.inputs.incr_a_array = interface.getArgumentIfExists("incr_a_array");
        state.inputs.incr_b_array = interface.getArgumentIfExists("incr_b_array");
    }
    state.inputs.lda = interface.getArgumentIfExists("lda");
    state.inputs.ldb = interface.getArgumentIfExists("ldb");
    state.inputs.ldc[0] = interface.getArgumentIfExists("ldc");
    state.inputs.ldc[1] = interface.getArgumentIfExists("ldp");
    state.inputs.ldao = interface.getArgumentIfExists("ldao");
    state.inputs.ldbo = interface.getArgumentIfExists("ldbo");
    state.inputs.ldco = interface.getArgumentIfExists("ldco");
    state.inputs.ldaScale = interface.getArgumentIfExists("lda_scale");
    state.inputs.ldbScale = interface.getArgumentIfExists("ldb_scale");
    state.inputs.ldcScale = interface.getArgumentIfExists("ldc_scale");
    state.inputs.ldag = interface.getArgumentIfExists("ldag");
    state.inputs.ldbg = interface.getArgumentIfExists("ldbg");
    state.inputs.ldaq = interface.getArgumentIfExists("ldaq");
    state.inputs.ldbq = interface.getArgumentIfExists("ldbq");
    state.inputs.ldcq = interface.getArgumentIfExists("ldcq");
    state.inputs.m = interface.getArgumentIfExists("m");
    state.inputs.n = interface.getArgumentIfExists("n");
    state.inputs.k = interface.getArgumentIfExists("k");
    state.inputs.k0 = interface.getArgumentIfExists("k0");
    state.inputs.alpha_real = interface.getArgumentIfExists("alpha_real");
    state.inputs.beta_real = interface.getArgumentIfExists("beta_real");
    if (problem.batch == BatchMode::Variable) {
        state.inputs.alpha_array = interface.getArgumentIfExists("alpha_array");
        state.inputs.beta_array = interface.getArgumentIfExists("beta_array");
        state.inputs.incr_alpha = interface.getArgumentIfExists("incr_alpha");
        state.inputs.incr_beta = interface.getArgumentIfExists("incr_beta");
    }
    state.inputs.diagA = interface.getArgumentIfExists("diag_A");
    state.inputs.diagB = interface.getArgumentIfExists("diag_B");
    state.inputs.diagC = interface.getArgumentIfExists("diag_C");
    state.inputs.flags = interface.getArgumentIfExists("flags");
    state.inputs.slmBase = interface.getArgumentIfExists("slm_base");

    if (strategy.linearOrder()) {
        state.inputs.groupCountM = interface.getArgument("group_count_m");
        state.inputs.groupCountN = interface.getArgument("group_count_n");
        if (strategy.scramble[LoopM])
            state.inputs.gcMRecip = interface.getArgumentIfExists("group_count_m_recip");
        if (strategy.scramble[LoopN])
            state.inputs.gcNRecip = interface.getArgumentIfExists("group_count_n_recip");
    }
    if (one_of(strategy.cWalkOrder, {WalkOrder::SimpleLinear, WalkOrder::NestedLinear}))
        state.inputs.gcMNRecip = interface.getArgument("group_count_recip");
    else if (strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        state.inputs.hilbertVD = interface.getArgumentIfExists("hilbert_vd");
        state.inputs.hilbertUVDRecip = interface.getArgumentIfExists("hilbert_uvd_recip");
        state.inputs.hilbertBail = interface.getArgumentIfExists("hilbert_bail");
    } else if (strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        state.inputs.bslice = interface.getArgument("bslice");
        state.inputs.bthresh = interface.getArgument("bthresh");
    }
    if (strategy.scramble[LoopM])
        state.inputs.wgStride[LoopM] = interface.getArgument("wg_stride_m").uw();
    if (strategy.scramble[LoopN])
        state.inputs.wgStride[LoopN] = interface.getArgument("wg_stride_n").uw();
    if (strategy.persistent)
        state.inputs.groupCountMN = interface.getArgumentIfExists("group_count");
    if (strategy.persistent || strategy.kParallelVariable)
        state.inputs.groupStride = interface.getArgument("group_stride");
    if (strategy.kParallelVariable) {
        state.inputs.kvConfig = interface.getArgument("kv_config");
        state.inputs.kRecip = interface.getArgument("k_recip");
    }
    if (strategy.kParallel && strategy.fuseBeta)
        state.inputs.groupCountK = interface.getArgument("group_count_k");
    if (strategy.fuseBeta || strategy.fusePostOps)
        state.inputs.statusBuffer = interface.getArgumentIfExists("status");

    size_t poCount = problem.postOps.len();
    state.inputs.binarySrcs.resize(poCount);
    state.inputs.binaryOffsets.resize(poCount);
    state.inputs.binaryLDs.resize(poCount);
    state.inputs.binaryStrides.resize(poCount);
    state.inputs.binarySurfaces.resize(poCount);
    for (size_t i = 0; i < poCount; i++) {
        std::string srcName = "binary" + std::to_string(i);
        state.inputs.binarySrcs[i] = interface.getArgumentIfExists(srcName);
        state.inputs.binarySurfaces[i] = interface.getArgumentSurfaceIfExists(srcName);
        state.inputs.binaryOffsets[i] = interface.getArgumentIfExists("offset_" + srcName);
        state.inputs.binaryLDs[i] = interface.getArgumentIfExists("ld" + srcName);
        if (problem.batch == BatchMode::Strided)
            for (int b = 0; b < problem.batchDims; b++)
                state.inputs.binaryStrides[i].push_back(interface.getArgumentIfExists("stride" + std::to_string(b) + srcName));
    }

    Subregister tgids_reordered[3];
    GRF lids_reordered[3];
    Subregister lszs_reordered[3];

    for (int l = 0; l < 3; l++) {
        int i = static_cast<int>(strategy.loopOrder[l]);
        tgids_reordered[i] = tgids[l];
        lids_reordered[i] = localID[l];
        lszs_reordered[i] = localSize[l];
    }
    state.inputs.groupIDM = tgids_reordered[0];
    state.inputs.groupIDN = tgids_reordered[1];
    state.inputs.groupIDK = tgids_reordered[2];
    state.inputs.localIDM = lids_reordered[0];
    state.inputs.localIDN = lids_reordered[1];
    state.inputs.localIDK = lids_reordered[2];
    state.inputs.localSizeM = lszs_reordered[0];
    state.inputs.localSizeN = lszs_reordered[1];
    state.inputs.localSizeK = lszs_reordered[2];

    if (strategy.linearOrder()) {
        state.groupIDMN = state.inputs.groupIDMN = tgids[0];
        state.inputs.groupIDM = invalid;
        state.inputs.groupIDN = invalid;
        state.groupCountMN = state.inputs.groupCountMN;
    }

    // Move SLM pointers to offset arguments.
    if (strategy.A.base.getModel() == ModelSLM) std::swap(state.inputs.A, state.inputs.offsetA);
    if (strategy.B.base.getModel() == ModelSLM) std::swap(state.inputs.B, state.inputs.offsetB);
    if (strategy.C.base.getModel() == ModelSLM) std::swap(state.inputs.C[0], state.inputs.offsetC[0]);

    // Downgrade offsets to 32 bits for non-A64 accesses.
    if (strategy.A.base.getModel() != ModelA64)
        state.inputs.offsetA = state.inputs.offsetA.d();
    if (strategy.B.base.getModel() != ModelA64)
        state.inputs.offsetB = state.inputs.offsetB.d();
    if (strategy.C.base.getModel() != ModelA64)
        for (int q = 0; q < state.C_count; q++)
            state.inputs.offsetC[q] = state.inputs.offsetC[q].d();
    if (problem.usesCOPtr() && strategy.CO.base.getModel() != ModelA64)
        state.inputs.offsetCO = state.inputs.offsetCO.d();
    for (auto &off: state.inputs.binaryOffsets)
        off = off.d();

    // For now, reinterpret m/n/k/ld/diag variables to 32-bit if they are 64-bit.
    state.inputs.m = state.inputs.m.d();
    state.inputs.n = state.inputs.n.d();
    state.inputs.k = state.inputs.k.d();
    state.inputs.lda = state.inputs.lda.ud();
    state.inputs.ldb = state.inputs.ldb.ud();
    for (int q = 0; q < state.C_count; q++)
        state.inputs.ldc[q] = state.inputs.ldc[q].ud();
    state.inputs.ldao = state.inputs.ldao.ud();
    state.inputs.ldbo = state.inputs.ldbo.ud();
    state.inputs.ldco = state.inputs.ldco.ud();
    state.inputs.diagA = state.inputs.diagA.d();
    state.inputs.diagB = state.inputs.diagB.d();
    state.inputs.diagC = state.inputs.diagC.d();

    // Claim registers.
    auto claimIfValid = [&](Subregister s) {
        if (s.isValid()) state.ra.claim(s);
    };

    for (int i = 0; i < r0DWords(hw); i++)
        state.ra.claim(r0.ud(i));

    if (strategy.A.base.isStateless()) state.ra.claim(state.inputs.A);
    if (strategy.B.base.isStateless()) state.ra.claim(state.inputs.B);
    if (strategy.C.base.isStateless())
        for (int q = 0; q < state.C_count; q++)
            state.ra.claim(state.inputs.C[q]);

    if (aOffset) {
        claimIfValid(state.inputs.ao);
        claimIfValid(state.inputs.aoPtr);
        claimIfValid(state.inputs.offsetAO);
    }

    if (bOffset) {
        claimIfValid(state.inputs.bo);
        claimIfValid(state.inputs.boPtr);
        claimIfValid(state.inputs.offsetBO);
    }

    if (problem.aScale2D()) {
        state.ra.claim(state.inputs.aScalePtr);
        claimIfValid(state.inputs.offsetAScale);
    }

    if (problem.bScale2D()) {
        state.ra.claim(state.inputs.bScalePtr);
        claimIfValid(state.inputs.offsetBScale);
    }

    if (problem.hasCMXScale()) {
        state.ra.claim(state.inputs.cScalePtr);
    }

    if (problem.needsAGroupSums()) {
        state.ra.claim(state.inputs.agPtr);
        claimIfValid(state.inputs.offsetAg);
    }

    if (problem.needsBGroupSums()) {
        state.ra.claim(state.inputs.bgPtr);
        claimIfValid(state.inputs.offsetBg);
    }

    claimIfValid(state.inputs.ldaq);
    claimIfValid(state.inputs.ldbq);
    claimIfValid(state.inputs.ldcq);
    claimIfValid(state.inputs.offsetAq);
    claimIfValid(state.inputs.offsetBq);

    if (problem.usesCOPtr()) {
        if (strategy.CO.base.isStateless())
            state.ra.claim(state.inputs.coPtr);
        state.ra.claim(state.inputs.offsetCO);
    } else if (problem.cOffsetHostScalar()) {
        state.ra.claim(state.inputs.co);
    }

    if (state.useTempC)
        if (strategy.C.base.isStateless())
            state.ra.claim(state.inputs.tempC);

    state.ra.claim(state.inputs.offsetA);
    state.ra.claim(state.inputs.offsetB);
    for (int q = 0; q < state.C_count; q++)
        state.ra.claim(state.inputs.offsetC[q]);
    state.ra.claim(state.inputs.lda);
    state.ra.claim(state.inputs.ldb);
    for (int q = 0; q < state.C_count; q++)
        state.ra.claim(state.inputs.ldc[q]);
    claimIfValid(state.inputs.ldao);
    claimIfValid(state.inputs.ldbo);
    if (problem.allowMatrixOffset())
        state.ra.claim(state.inputs.ldco);
    claimIfValid(state.inputs.ldaScale);
    claimIfValid(state.inputs.ldbScale);
    claimIfValid(state.inputs.ldcScale);
    claimIfValid(state.inputs.ldag);
    claimIfValid(state.inputs.ldbg);
    state.ra.claim(state.inputs.m);
    state.ra.claim(state.inputs.n);
    state.ra.claim(state.inputs.k);
    if (strategy.kParallel || strategy.kParallelLocal || strategy.kParallelVariable)
        state.ra.claim(state.inputs.k0);

    if (problem.alpha == Scalar::Variable)
        state.ra.claim(state.inputs.alpha_real);
    else if (problem.alpha.pointer())
        state.ra.claim(state.inputs.alphaPtr);

    if (problem.beta == Scalar::Variable)
        state.ra.claim(state.inputs.beta_real);
    else if (problem.beta.pointer())
        state.ra.claim(state.inputs.betaPtr);

    if (!inSK) {
        state.ra.claim(state.inputs.localIDM);
        state.ra.claim(state.inputs.localIDN);
        if (!strategy.fixedWG(problem)) {
            state.ra.claim(state.inputs.localSizeM);
            state.ra.claim(state.inputs.localSizeN);
        } else
            state.inputs.localSizeM = state.inputs.localSizeN = invalid;
        if (strategy.kParallel || strategy.kParallelLocal) {
            state.ra.claim(state.inputs.localIDK);
            state.ra.claim(state.inputs.localSizeK);
        }
    }

    if (state.inputs.flags.isValid())
        state.ra.claim(state.inputs.flags);
    if (problem.postOps.cStochasticRound)
        state.ra.claim(state.inputs.sroundSeedPtr);
    if (state.inputs.slmBase.isValid())
        state.ra.claim(state.inputs.slmBase);

    if (problem.batch == BatchMode::Strided) {
        for (int i = 0; i < problem.batchDims; i++) {
            state.ra.claim(state.inputs.strideA[i]);
            state.ra.claim(state.inputs.strideB[i]);
            state.ra.claim(state.inputs.strideC[i]);
            if(problem.hasAScalePtr()){
                    state.ra.claim(state.inputs.strideScaleA[i]);
            }
            if(problem.hasBScalePtr()){
                    state.ra.claim(state.inputs.strideScaleB[i]);
            }
            if(problem.hasCMXScale()){
                    state.ra.claim(state.inputs.strideScaleC[i]);
            }
            if(problem.hasAOffsetPtr()){
                    state.ra.claim(state.inputs.strideOffsetA[i]);
            }
            if(problem.hasBOffsetPtr()){
                    state.ra.claim(state.inputs.strideOffsetB[i]);
            }
            if(problem.needsAGroupSums()){
                    state.ra.claim(state.inputs.strideGroupSumsA[i]);
            }
            if(problem.needsBGroupSums()){
                    state.ra.claim(state.inputs.strideGroupSumsB[i]);
            }
        }
        for (int i = 0; i < problem.batchDims - 1; i++) {
            state.ra.claim(state.inputs.batchSize[i]);
            state.ra.claim(state.inputs.recipBatchSize[i]);
        }
        state.ra.claim(state.inputs.groupIDK);
    } else if (problem.batch == BatchMode::Nonstrided) {
        state.ra.claim(state.inputs.offsetBatch);
        state.ra.claim(state.inputs.groupIDK);
    }
    else if (problem.batch == BatchMode::Variable) {
        state.ra.claim(state.inputs.incr_a_array);
        state.ra.claim(state.inputs.incr_b_array);
        state.ra.claim(state.inputs.alpha_array);
        state.ra.claim(state.inputs.beta_array);
        state.ra.claim(state.inputs.incr_alpha);
        state.ra.claim(state.inputs.incr_beta);
        state.ra.claim(state.inputs.groupIDK);
    }

    if (strategy.linearOrder()) {
        state.ra.claim(state.inputs.groupCountM);
        state.ra.claim(state.inputs.groupCountN);
        if (strategy.scramble[LoopM])
            state.ra.claim(state.inputs.gcMRecip);
        if (strategy.scramble[LoopN])
            state.ra.claim(state.inputs.gcNRecip);
    }

    if (one_of(strategy.cWalkOrder, {WalkOrder::SimpleLinear, WalkOrder::NestedLinear}))
        state.ra.claim(state.inputs.gcMNRecip);
    else if (strategy.cWalkOrder == WalkOrder::Hilbertlike) {
        {
            state.ra.claim(state.inputs.hilbertVD);
            state.ra.claim(state.inputs.hilbertUVDRecip);
        }
        state.ra.claim(state.inputs.hilbertBail);
    } else if (strategy.cWalkOrder == WalkOrder::Boustrophedon) {
        state.ra.claim(state.inputs.bslice);
        state.ra.claim(state.inputs.bthresh);
    }

    if (strategy.persistent || strategy.kParallelVariable)
        state.ra.claim(state.inputs.groupStride);

    claimIfValid(state.inputs.groupCountMN);

    if (strategy.kParallelVariable) {
        state.ra.claim(state.inputs.kvConfig);
        state.ra.claim(state.inputs.kRecip);
    }

    if (strategy.kParallel && strategy.fuseBeta)
        state.ra.claim(state.inputs.groupCountK);

    if (strategy.fuseBeta || strategy.fusePostOps)
        state.ra.claim(state.inputs.statusBuffer);

    // Binary-related arguments are not claimed here, but instead
    //  are reloaded later in the kernel when needed.

}

// Initialize the state structure.
template <HW hw>
void Generator<hw>::gemmInitState(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state, bool inSK)
{
    auto Tc = problem.Tc;
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext;

    state.useTempC = strategy.needsTempC(problem);

    {
        initState(problem, strategy, state);
        gemmInitInterface(problem, strategy, state, inSK);
        state.isNested |= strategy.fused;
        state.isNested |= strategy.persistentLoop();
    }

    state.effA = strategy.A.base.isStateless() ? state.inputs.A
                                               : state.inputs.offsetA.d();
    state.effB = strategy.B.base.isStateless() ? state.inputs.B
                                               : state.inputs.offsetB.d();
    for (int q = 0; q < state.C_count; q++) {
        state.effC[q] = strategy.C.base.isStateless() ? state.inputs.C[q]
                                                      : state.inputs.offsetC[q].d();
    }
    if (problem.usesCOPtr()) {
        state.effCO = strategy.CO.base.isStateless() ? state.inputs.coPtr
                                                     : state.inputs.offsetCO.d();
    }
    if (state.useTempC) {
        state.effTempC = strategy.C.base.isStateless() ? state.inputs.tempC
                                                       : state.ra.alloc_sub<uint32_t>();
    }

    state.offsetA = state.inputs.offsetA;
    state.offsetB = state.inputs.offsetB;
    for (int q = 0; q < state.C_count; q++)
        state.offsetC[q] = state.inputs.offsetC[q];
    state.offsetCO = state.inputs.offsetCO;

    state.flagAP = state.raVFlag.alloc();

    state.allocEmulate64Temp(strategy.emulate);

    state.Ta_load = problem.Ta_ext;
    state.Tb_load = problem.Tb_ext;

    state.Tacc = problem.Tc;
    state.copyC = (problem.Tc != problem.Tc_ext)
               || (!strategy.altCRemainder && (Tc.size() < 4))
               || strategy.forceCopyC
               || (strategy.C.base.getModel() == ModelInvalid);

    state.Cext_strategy = strategy.C;
    state.Cext_strategy.tileR = state.Cext_strategy.tileC = 0;

    state.lidM = state.inputs.localIDM[0];
    state.lidN = state.inputs.localIDN[0];
    if (strategy.kParallel || strategy.kParallelLocal)
        state.lidK = state.inputs.localIDK[0];

    state.k = state.wgK = state.inputs.k;

    state.lda = state.inputs.lda;
    state.ldb = state.inputs.ldb;

    if (GRF::bytes(hw) == 64) {
        if (!isRegisterColMajor(Ta_ext, problem.A, strategy.A) && strategy.allowDoubleMasking(LoopM)) {
            int ka = strategy.slmA ? strategy.unrollKSLM : strategy.ka_load;
            int effAlign = isBlocklike(strategy.A.accessType) ? problem.A.alignment : 1;
            state.simd32KMasks |= (ka >= 32 * (std::min<int>(4, effAlign) / std::min(4, Ta_ext.paddedSize())));
        }
        if (isRegisterColMajor(Tb_ext, problem.B, strategy.B) && strategy.allowDoubleMasking(LoopN)) {
            int kb = strategy.slmB ? strategy.unrollKSLM : strategy.kb_load;
            int effAlign = isBlocklike(strategy.B.accessType) ? problem.B.alignment : 1;
            state.simd32KMasks |= (kb >= 32 * (std::min<int>(4, effAlign) / std::min(4, Tb_ext.paddedSize())));
        }
    }

    if (state.useTempC) {
        // TODO: consider remainder handling to reduce BW.
        bool cColMajor = isRegisterColMajor(problem.Tc_ext, problem.C, strategy.C);
        state.tempC.layout = cColMajor ? MatrixLayout::Pc : MatrixLayout::Pr;
        state.tempC.crosspack = 1;
        state.tempC.packSize = strategy.unroll[cColMajor ? LoopM : LoopN];
        state.tempC.panelLength = 0;
        state.tempC.setAlignment(64);

        state.tempCStrategy = strategy.C;
        state.tempCStrategy.accessType = AccessType::Block;
        state.tempCStrategy.address2D = false;
        state.tempCStrategy.padded = true;
    }

    state.useBDPAS = problem.preferBDPAS() && strategy.systolic;
}

GEMMSTONE_NAMESPACE_END
