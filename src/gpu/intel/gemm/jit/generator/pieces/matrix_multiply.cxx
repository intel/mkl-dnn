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


#include "compute_utils.hpp"
#include "gemmstone/generator.hpp"
#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;

// Do one or more outer products (k = 1 slices) of A*B, updating C.
//  ha and hb are the k indices within the A and B chunks, respectively.
//  A_copy, B_copy are the indices of the A, B copies to use.
template <HW hw>
void Generator<hw>::outerProduct(int h, int ha_period, int hb_period, int opCount, bool rem,
                                 const RegisterLayout &A_layout, const RegisterLayout &B_layout,
                                 const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                 const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
#ifdef STANDALONE
    if (problem.readCheck)
        noop();
    else if (problem.mock.multiply) {
        if (hw >= HW::Gen12LP)
            wrdepRanges({A_regs, B_regs});
    } else
#endif
    if (strategy.dotVL)
        innerProductFMA(h, ha_period, hb_period, opCount, rem, A_layout, B_layout, A_regs, B_regs, problem, strategy, state);
    else if (strategy.systolic)
        outerProductSystolic(h, ha_period, hb_period, opCount, rem, A_layout, B_layout, A_regs, B_regs, problem, strategy, state);
    else
        outerProductFMA(h, ha_period, hb_period, opCount, rem, A_layout, B_layout, A_regs, B_regs, problem, strategy, state);
}

// FMA or dp4a-based outer product implementation.
template <HW hw>
void Generator<hw>::outerProductFMA(int h, int ha_period, int hb_period, int opCount, bool rem, const RegisterLayout &A_layout, const RegisterLayout &B_layout,
                                    const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                    const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc_compute();

    bool mixedMode = ((Tc.real() == Type::f32) && (Ta.real() != Type::f32 || Tb.real() != Type::f32));
    bool useDP4A = problem.isIGEMM();
    bool atomicFMA = strategy.atomicFMA;
    bool noAccSBSet = strategy.extendedAtomicFMA && (hw >= HW::XeHPC);

    int minOPCount = minOuterProductCount(problem, strategy);
    int kChain = opCount / minOPCount;
    int aCP, bCP;
    std::tie(aCP, bCP) = targetKernelCrosspack(hw, problem, strategy);

    int accNum = 0;
    Subregister Clast;
    int nec = elementsPerGRF(hw, Tc);
    bool globalCM = state.C_layout.colMajor();
    int fmaSIMD = strategy.fmaSIMD;

    bool repackC = !state.Cr_layout.empty();
    bool startRepackC = false, endRepackC = false;
    int Cr_unrollM = state.Cr_layout.rows(), Cr_unrollN = state.Cr_layout.cols();
    int Cr_unrollX = globalCM ? Cr_unrollM : Cr_unrollN;
    if (repackC) {
        auto rphase = local_k_index(h, opCount, state.cRepackPeriod, problem);
        startRepackC = rem || (rphase == 0);
        endRepackC = rem || (rphase + opCount >= state.cRepackPeriod);
    }

    bool csplit = false, mixedRC = false;
    int icompCount = 1, ocompCount = 1, ivcompCount = 1, ovcompCount = 1;

    // Emit an FMA instruction.
    auto outputFMA = [&](InstructionModifier mod, const Subregister &A, const Subregister &B, const Subregister &C, const RegData &bcastSrc, bool colMajor, int hh, bool ivfirst, bool ivlast) {
        auto Cacc = AccumulatorRegister(accNum).sub(C.getOffset(), Tc.real().ngen());
        auto Csrc = (hh == 0                    && ivfirst) ? C : Cacc;
        auto Cdst = (hh == opCount - minOPCount && ivlast)  ? C : Cacc;
        if (startRepackC && hh == 0)
            Csrc = null.retype(Csrc.getType())[0];

        bool thisNoAccSBSet = (noAccSBSet && Cacc == Csrc);
        if (thisNoAccSBSet) {
            mod |= NoAccSBSet;
            setDefaultAutoSWSB(false);
        }
        if (A.isInvalid() || B.isInvalid()) {
            if (Csrc != Cdst)
                mov(mod, Cdst(1), Csrc(1));
        } else if (useDP4A) {
            auto Ar = A.reinterpret(0, isSigned(A.getType()) ? DataType::d : DataType::ud);
            auto Br = B.reinterpret(0, isSigned(B.getType()) ? DataType::d : DataType::ud);

            if (Csrc.isNull()) {
                colMajor       ? dp4a(mod, Cdst(1), 0, Ar(1), Br(0))
                               : dp4a(mod, Cdst(1), 0, Br(1), Ar(0));
            } else {
                colMajor       ? dp4a(mod, Cdst(1), Csrc(1), Ar(1), Br(0))
                               : dp4a(mod, Cdst(1), Csrc(1), Br(1), Ar(0));
            }
        } else if (C.isARF() && hw < HW::XeHP) {
            colMajor           ? mac(mod, C(1), A(1), bcastSrc)
                               : mac(mod, C(1), bcastSrc, B(1));
        } else if (Csrc.isNull()) {
            colMajor           ? mul(mod, C(1), A(1), bcastSrc)
                               : mul(mod, C(1), B(1), bcastSrc);
        } else {
            // Always put broadcast in src2 for better bank conflict avoidance.
            colMajor           ? mad(mod, Cdst(1), Csrc(1), A(1), bcastSrc) :
                                 mad(mod, Cdst(1), Csrc(1), B(1), bcastSrc);
        }
        if (thisNoAccSBSet)
            setDefaultAutoSWSB(true);
    };

    int ha = local_k_index(h, opCount, ha_period, problem);
    int hb = local_k_index(h, opCount, hb_period, problem);

    if (atomicFMA && hw >= HW::XeHPC) {
        // PVC onward can't use {Atomic} on float pipes.
        // Use wrdep to wait on all outstanding A/B dependencies instead.
        atomicFMA = false;
        for (auto &block: A_layout)
            if (block.offsetC >= ha && block.offsetC < ha + opCount)
                wrdep(A_regs[block.offsetReg()]);
        for (auto &block: B_layout)
            if (block.offsetR >= hb && block.offsetR < hb + opCount)
                wrdep(B_regs[block.offsetReg()]);
    }

    // Decide whether to loop in column or row major order.
    //   x = vectorized dimension
    //   y = non-vectorized dimension
    int nx = globalCM ? strategy.unroll[LoopM] : strategy.unroll[LoopN];
    int ny = globalCM ? strategy.unroll[LoopN] : strategy.unroll[LoopM];
    int nx1 = mixedMode ? nx : std::min(nx, fmaSIMD);

    // Prepare for chaining FMAs through accumulator registers.
    int necAcc = nec * (csplit ? 2 : 1);
    int accCount = AccumulatorRegister::count(hw, strategy.GRFs, Tc.ngen());
    int accPerFMA = div_up(std::min(nx, fmaSIMD), necAcc);
    int minAccPerFMA = Tc.isFP() ? 1 : 2;
    accPerFMA = std::max(accPerFMA, minAccPerFMA);
    int independentAccs = div_up(accCount, accPerFMA);

    // Mirror canColMajor/canRowMajor logic below. If neither, downgrade simd to 1.
    bool fmaVectorized = (isRegisterColMajor(problem.Ta, problem.A, strategy.A) && globalCM)
                      || (!isRegisterColMajor(problem.Tb, problem.B, strategy.B) && !globalCM);
    int fmaExpected = fmaVectorized ? fmaSIMD : 1;

    int nx1i = 1, ny1 = 1;
    if (kChain > 1) {
        if (independentAccs < icompCount) hw_unsupported();
        int indepAccComp = div_up(independentAccs, icompCount);

        nx1i = std::min(nx1, indepAccComp * fmaExpected);
        ny1 = div_up(indepAccComp, div_up(nx1i, fmaExpected));

        noAccSBSet &= Tc.isFP()
                   && (div_up(nx1i, necAcc) * ny1 * icompCount >= 8)
                   && (nx % nx1i == 0);     /* {NoAccSBSet} requires all 8 accumulator registers to be used */
    } else
        noAccSBSet = false;

    if (repackC && Cr_unrollX < nx && (Cr_unrollX % nx1))
        stub("Invalid C repack panel size");

    // Last A/B blocks found.
    const RegisterBlock *A_blockLast = nullptr, *B_blockLast = nullptr;

    for (int x0 = 0; x0 < nx; x0 += nx1) {
        for (int ovcomp = 0; ovcomp < ovcompCount; ovcomp++) {
        for (int ocomp = 0; ocomp < ocompCount; ocomp++) {
        for (int y0 = 0; y0 < ny; y0 += ny1) {
        for (int x1 = 0; x1 < nx1 && (x0 + x1) < nx;) {
            int x1New = x1;
            for (int ivcomp = 0; ivcomp < ivcompCount; ivcomp++) {
            for (int hh = 0; hh < opCount; hh += minOPCount) {
                accNum = 0;
                for (int y1 = 0; y1 < ny1 && y0 + y1 < ny; y1++) {
                for (int x1i = x1; (x1i < x1 + nx1i) && (x0 + x1i < nx);) {
                    auto x = x0 + x1i;
                    auto y = y0 + y1;
                    auto i = globalCM ? x : y;
                    auto j = globalCM ? y : x;
                    auto hha = ha + hh;
                    auto hhb = hb + hh;
                    auto ia = i, jb = j;

                    int fmaCount = 1;

                    for (int icomp = 0; icomp < icompCount; icomp++) {
                        // Find the appropriate A and B registers.
                        int na, nb;
                        int vcomp = ivcomp + ovcomp;
                        int ncomp = (vcomp ^ ocomp) + icomp;
                        int compA = globalCM ? vcomp : ncomp;
                        int compB = globalCM ? ncomp : vcomp;

                        const RegisterBlock *A_block, *B_block;
                        Subregister A = A_layout.find(ia, hha, A_regs, &na, &A_block, compA);
                        Subregister B = B_layout.find(hhb, jb, B_regs, &nb, &B_block, compB);

                        // Check for expected crosspack.
                        if (globalCM ? (aCP && A_block->crosspack != aCP)
                                     : (bCP && B_block->crosspack != bCP)) stub();

                        // Check if we should specify {Atomic}.
                        bool atomic = (atomicFMA && (A_block == A_blockLast) && (B_block == B_blockLast));
                        A_blockLast = A_block;
                        B_blockLast = B_block;

                        // Find the appropriate C register.
                        int C_buffer = csplit ? 0 : (icomp + ocomp);
                        int compC    = csplit ? ocomp : 0;
                        int nc;
                        const RegisterBlock *C_block;
                        Subregister C = repackC ? state.Cr_layout.find(i % Cr_unrollM, j % Cr_unrollN, state.Cr_regs, &nc, &C_block, compC)
                                                : state.C_layout.find(i, j, state.C_regs[C_buffer], &nc, &C_block, compC);
                        if (C_block->crosspack > 1) stub();

                        // Swap out C register for an accumulator, if necessary.
                        if (strategy.cAccumulators) {
                            auto C_roff = C.getBase() - state.C_regs[0].ranges[0].getBase();
                            if (C_roff < state.C_accCount)
                                C = AccumulatorRegister(C_roff).sub(C.getOffset(), Tc.ngen());
                        }

                        InstructionModifier mod;

                        // Use requested execution size if possible, but limited to available elements.
                        // Decide on kernel type based on register block layouts.
                        bool canColMajor = (A_block->colMajor && globalCM);
                        bool canRowMajor = (!B_block->colMajor && !globalCM);
                        bool colMajor = globalCM;

                        if (!canColMajor && !canRowMajor)
                            fmaCount = 1;
                        else if (canColMajor)
                            fmaCount = rounddown_pow2(std::min({fmaSIMD, na, nc}));
                        else
                            fmaCount = rounddown_pow2(std::min({fmaSIMD, nb, nc}));

                        int simdSize = fmaCount;

                        // Crosspacked kernels: ensure broadcast matrix is contiguous in k.
                        if (minOPCount > 1) {
                            bool nativeDir = (globalCM ? B_block->colMajor : !A_block->colMajor);
                            auto bcastCrosspack = (globalCM ? B_block : A_block)->crosspack;
                            if (nativeDir) {
                                if ((globalCM ? nb : na) < minOPCount) stub();
                                if (bcastCrosspack > 1) stub();
                            } else {
                                if (bcastCrosspack % minOPCount) stub();
                            }
                        }

                        // Add Atomic if appropriate.
                        if (atomic) mod |= Atomic;

                        // Handle broadcast duties.
                        Subregister bcastSrcSub = colMajor ? B : A;
                        RegData bcastSrc = bcastSrcSub;

                        bool ivfirst = mixedRC || (ivcomp == 0);
                        bool ivlast  = mixedRC || (ivcomp == ivcompCount - 1);

                        // Finally, perform the long-awaited FMA.
                        outputFMA(simdSize | mod, A, B, C, bcastSrc, colMajor, hh, ivfirst, ivlast);
                        Clast = C;

                        if (kChain > 1 && accNum >= accCount) stub();
                        accNum += std::max(minAccPerFMA, div_up(fmaCount, necAcc));
                    } /* icomp */

                    x1i += fmaCount;
                    x1New = x1i;
                } /* x1i */
                } /* y1 */
            } /* hh */
            } /* ivcomp */
            x1 = x1New;
        } /* x1 */
        } /* y0 */
        } /* ocomp */
        } /* ovcomp */

        // A/B sums: dp4a against all 1s into Asr/Bsr, which are repacked alongside C.
        if (globalCM ? state.fmaSumA : state.fmaSumB) {
            auto &Xsr_layout = globalCM ? state.Asr_layout : state.Bsr_layout;
            auto &Xsr_regs   = globalCM ? state.Asr_regs   : state.Bsr_regs;
            auto all1s = state.sysSumAll1s.ud(0);
            for (int hh = 0; hh < opCount; hh += minOPCount) {
                for (int x1 = 0; x1 < nx1 && (x0 + x1) < nx;) {
                    int x = x0 + x1, xr = x % Cr_unrollX, nsrc, nsum;
                    const RegisterBlock *block;
                    auto src = globalCM ? A_layout.find(x, ha + hh, A_regs, &nsrc, &block)
                                        : B_layout.find(hb + hh, x, B_regs, &nsrc, &block);
                    if (block->colMajor != globalCM || block->crosspack != minOPCount) stub();
                    auto sum = globalCM ? Xsr_layout.find(xr, 0, Xsr_regs, &nsum)
                                        : Xsr_layout.find(0, xr, Xsr_regs, &nsum);
                    int simd = rounddown_pow2(std::min({fmaSIMD, nsrc, nsum, nx1 - x1, Cr_unrollX - xr}));
                    auto srcD = src.reinterpret(0, isSigned(src.getType()) ? DataType::d : DataType::ud);
                    if (startRepackC && hh == 0)
                        dp4a(simd, sum(1), 0, srcD(1), all1s(0));
                    else
                        dp4a(simd, sum(1), sum(1), srcD(1), all1s(0));
                    x1 += simd;
                }
            }
        }

        if (endRepackC) {
            int xr = x0 - Cr_unrollX + nx1;
            if (xr >= 0)
                outerProductRepackC(xr, xr % Cr_unrollX, nx1, h, rem, problem, strategy, state);
        }
    } /* x0 */

    if (endRepackC) for (int xr = nx - Cr_unrollX + nx1; xr < nx; xr += nx1)
        outerProductRepackC(xr, xr % Cr_unrollX, nx1, h, rem, problem, strategy, state);
}

// FMA-based inner product (dot) implementation.
template <HW hw>
void Generator<hw>::innerProductFMA(int h, int ha_period, int hb_period, int opCount, bool rem, const RegisterLayout &A_layout, const RegisterLayout &B_layout,
                                    const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                    const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    int vl = strategy.dotVL;
    bool useDP4A = (hw >= HW::Gen12LP && problem.isIGEMM());

    bool repackC = !state.Cr_layout.empty();
    bool startRepackC = false, endRepackC = false;
    if (repackC) {
        if (state.Cr_layout.rows() != state.C_layout.rows()) stub();
        if (state.Cr_layout.cols() != state.C_layout.cols()) stub();
        auto rphase = local_k_index(h, opCount, state.cRepackPeriod, problem);
        startRepackC = rem || (rphase == 0);
        endRepackC = rem || (rphase + opCount >= state.cRepackPeriod);
    }
    if (strategy.kChain > 1) stub();

    const auto &C_layout = repackC ? state.Cr_layout : state.C_layout;
    const auto &C_regs   = repackC ? state.Cr_regs   : state.C_regs[0];

    bool globalCM = C_layout.colMajor();
    int ha = local_k_index(h, opCount, ha_period, problem);
    int hb = local_k_index(h, opCount, hb_period, problem);

    int nx = globalCM ? strategy.unroll[LoopM] : strategy.unroll[LoopN];
    int ny = globalCM ? strategy.unroll[LoopN] : strategy.unroll[LoopM];
    int oc = useDP4A ? 4 : 1;

    for (int y0 = 0; y0 < ny; y0++) {
    for (int x0 = 0; x0 < nx; x0++) {
    for (int h0 = 0; h0 < vl; ) {
        auto x0c = x0 * vl + h0;
        auto i0 = globalCM ? x0 : y0, i0c = globalCM ? x0c : y0;
        auto j0 = globalCM ? y0 : x0, j0c = globalCM ? y0 : x0c;

        const RegisterBlock *A_block = nullptr, *B_block = nullptr;
        int na, nb, nc;
        auto A = A_layout.find(i0, h0 + ha, A_regs, &na, &A_block);
        auto B = B_layout.find(h0 + hb, j0, B_regs, &nb, &B_block);
        auto C = C_layout.find(i0c, j0c, C_regs, &nc);

        if (A_block->colMajor || !B_block->colMajor) stub();

        int ne = std::min({na / oc, nb / oc, nc, vl, strategy.fmaSIMD});

        if (useDP4A) {
            auto Ar = A.reinterpret(0, isSigned(A.getType()) ? DataType::d : DataType::ud);
            auto Br = B.reinterpret(0, isSigned(B.getType()) ? DataType::d : DataType::ud);
            startRepackC ? dp4a(ne, C(1),    0, Ar(1), Br(1))
                         : dp4a(ne, C(1), C(1), Ar(1), Br(1));
        } else {
            startRepackC ? mul(ne, C(1),       A(1), B(1))
                         : mad(ne, C(1), C(1), A(1), B(1));
        }
        h0 += ne * oc;
    } /* h0 */
    } /* x0 */
    } /* y0 */

    if (endRepackC)
        outerProductRepackC(0, 0, nx * vl, h, rem, problem, strategy, state);
}

// Accumulate multiple outer products using the systolic array.
template <HW hw>
void Generator<hw>::outerProductSystolic(int h, int ha_period, int hb_period, int opCount, bool rem,
                                         const RegisterLayout &A_layout, const RegisterLayout &B_layout,
                                         const GRFMultirange &A_regs, const GRFMultirange &B_regs,
                                         const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb;
    auto Tc = problem.Tc_compute();
    bool globalCM = state.C_layout.colMajor();
    auto params = systolicParams(problem);
    auto ksys = params.ksys;
    auto osys = params.osys;
    auto sdepth = params.sdepth;
    auto rcountMax = params.rcountMax;
    int dpaswTile = strategy.dpasw ? (globalCM ? strategy.B.tileC : strategy.A.tileR) : 0;
    bool rsFix = (strategy.readSuppressionWA && (globalCM ? A_layout : B_layout).hasMasking());
    bool canAtomicNon8x8 = (hw >= HW::XeHPC);
    bool sum = globalCM ? state.systolicSumA
                        : state.systolicSumB;
    bool snake = strategy.fmaBoustrophedon && !sum;
    bool repackC = !state.Cr_layout.empty();
    bool startRepackC = false, endRepackC = false;
    int Cr_unrollM = state.Cr_layout.rows(), Cr_unrollN = state.Cr_layout.cols();
    int Cr_unrollX = globalCM ? Cr_unrollM : Cr_unrollN;
    if (repackC) {
        auto rphase = local_k_index(h, opCount, state.cRepackPeriod, problem);
        startRepackC = rem || (rphase == 0);
        endRepackC = rem || (rphase + opCount >= state.cRepackPeriod);
    }

    RegisterBlock sumBlock;
    sumBlock.colMajor = globalCM;
    sumBlock.crosspack = 1;

    int ha = local_k_index(h, opCount, ha_period, problem);
    int hb = local_k_index(h, opCount, hb_period, problem);

    // Decide whether to loop in column or row major order, to facilitate macro sequences.
    //  x is the non-accumulating dimension of dpas src1 (V matrix)
    //  y is the non-accumulating dimension of dpas src2 (N matrix)
    int nx = strategy.unroll[globalCM ? LoopM : LoopN];
    int ny = strategy.unroll[globalCM ? LoopN : LoopM];

    int xinc = osys;

    // Split y loop for k-chaining or boustrophedon ordering.
    bool reverse = false;
    int ny1 = 1;
    if (opCount > ksys || snake)
        ny1 = rcountMax * (strategy.dpasw ? 2 : 1);

    const int compA = 0, compB = 0;
    const int icxCompNCount = 1, ocxCompNCount = 1;

    for (int x = 0; x < nx; x += xinc) {
        Subregister A0, B0, C0;
        RegData AS, BS, AS0, BS0;
        int rcount = 0, ybase = 0, hhbase = 0;

        auto issueDPAS = [&](bool last) {
            InstructionModifier mod = osys;

            bool useDPASW = strategy.dpasw && ybase < ny;
            auto rc = rcount * (useDPASW ? 2 : 1);
            auto V0 = globalCM ? A0 : B0;
            auto N0 = globalCM ? B0 : A0;
            RegData srcC0 = C0;

            if (rsFix) {
                GRF v0GRF{V0.getBase()};
                mov<uint32_t>(8, v0GRF, v0GRF);
                rsFix = false;
            }

            if (strategy.atomicFMA) {
                if (!last && (rc == 8 || canAtomicNon8x8))
                    mod |= Atomic;
                if (rc != 8 && strategy.extendedAtomicFMA) hw_unsupported();
            }

            const bool canFwd = (hw == HW::Xe3p)
                && (problem.product.family >= ProductFamily::CRI)
                && (rc == 8)
                && mod.isAtomic()
                && (problem.Tc != Type::f16);
            if (hhbase + ksys < opCount && canFwd) mod |= Fwd;

            bool nullC = startRepackC && hhbase == 0;
            if (nullC)
                srcC0 = null.retype(C0.getType());

            if (state.useBDPAS) {
                bdpas(mod, sdepth, rc, C0, srcC0, V0, N0, AS0, BS0);
            } else {
                useDPASW ? dpasw(mod, sdepth, rc, C0, srcC0, V0, N0) :
                            dpas(mod, sdepth, rc, C0, srcC0, V0, N0);
            }
        };

        for (int ocxCompN = 0; ocxCompN < ocxCompNCount; ocxCompN++, reverse = !reverse) {
        for (int y0_ = 0; y0_ < ny + sum; y0_ += ny1) {
            int y0 = (snake && reverse) ? (ny - ny1 - y0_) : y0_;
            for (int hh = 0; hh < opCount; hh += ksys) {
            for (int y1 = 0; y1 < ny1 && y0 + y1 < ny + sum; y1++) {
            for (int icxCompN = 0; icxCompN < icxCompNCount; icxCompN++) {
                int y = y0 + y1;
                int hha = ha + hh, hhb = hb + hh;

                const int cxCompA = -1, cxCompB = -1, cxCompC = -1, cBuffer = 0;
                auto bdpasScaleArg = [&](const RegisterLayout Xr_scaleLayout, Type Tx_scaleOp,
                                         GRFMultirange Xr_scaleRegs, bool isA, int x) {
                    RegData XS;
                    if (state.useBDPAS && !Xr_scaleLayout.empty()) {
                        int xqGroupK = isA ? problem.aqGroupK : problem.bqGroupK;
                        int kaq = state.lateScale2DA ? state.kaqLate : state.kaq;
                        int kbq = state.lateScale2DB ? state.kbqLate : state.kbq;
                        int kxq = isA ? kaq : kbq;
                        int kxq_load = xqGroupK * kxq;
                        int hq = local_k_index(h, opCount, kxq_load, problem);
                        hq = (hq + hh) % kxq_load;
                        int k = hq / xqGroupK;

                        int io0 = isA ? x : k;
                        int jo0 = isA ? k : x;

                        XS = Xr_scaleLayout.find(io0, jo0, Xr_scaleRegs);
                    } else
                        XS = NullRegister().setType(Type::ngen_e8m0());

                    return XS;
                };

                // Find the appropriate A and B registers.
                int na, nb, nc;
                const RegisterBlock *A_block, *B_block, *C_block;
                Subregister A, B, C;
                if (y < ny) {
                    if (strategy.dpasw && (y % (2 * dpaswTile) >= dpaswTile))
                        continue;

                    int i = globalCM ? x : y;
                    int j = globalCM ? y : x;

                    A = A_layout.find(i, hha, A_regs, &na, &A_block, cxCompA, compA);
                    B = B_layout.find(hhb, j, B_regs, &nb, &B_block, cxCompB, compB);
                    if (repackC)
                        C = state.Cr_layout.find(i % Cr_unrollM, j % Cr_unrollN, state.Cr_regs, &nc, &C_block, cxCompC);
                    else
                        C = state.C_layout.find(i, j, state.C_regs[cBuffer], &nc, &C_block, cxCompC);
                    AS = bdpasScaleArg(state.Ar_scaleLayout, state.Ta_scaleInt, state.Ar_scaleRegs, true, i);
                    BS = bdpasScaleArg(state.Br_scaleLayout, state.Tb_scaleInt, state.Br_scaleRegs, false, j);
                } else if (state.systolicSumA) {
                    A = A_layout.find(x, hha, A_regs, &na, &A_block);
                    B = state.sysSumAll1s[0];
                    nb = elementsPerGRF(hw, Tb);
                    B_block = &sumBlock;
                    if (repackC)
                        C = state.Asr_layout.find(x % Cr_unrollM, 0, state.Asr_regs, &nc, &C_block);
                    else
                        C = state.As_layout.find(x, 0, state.As_regs, &nc, &C_block);
                    AS = bdpasScaleArg(state.Ar_scaleLayout, state.Ta_scaleInt, state.Ar_scaleRegs, true, x);
                    BS = NullRegister().setType(Type::ngen_e8m0());
                } else {
                    A = state.sysSumAll1s[0];
                    na = elementsPerGRF(hw, Ta);
                    A_block = &sumBlock;
                    B = B_layout.find(hhb, x, B_regs, &nb, &B_block);
                    if (repackC)
                        C = state.Bsr_layout.find(0, x % Cr_unrollN, state.Bsr_regs, &nc, &C_block);
                    else
                        C = state.Bs_layout.find(0, x, state.Bs_regs, &nc, &C_block);
                    AS = NullRegister().setType(Type::ngen_e8m0());
                    BS = bdpasScaleArg(state.Br_scaleLayout, state.Tb_scaleInt, state.Br_scaleRegs, false, x);
                }

                int nv = globalCM ? na : nb;
                int nn = globalCM ? nb : na;

                // Verify DPAS requirements.
                if (globalCM) {
                    if (A_block->crosspack * Ta.real() != std::max(4, Ta.real().paddedSize())) stub();
                    if (B_block->crosspack > 1) stub();
                } else {
                    if (B_block->crosspack * Tb.real() != std::max(4, Tb.real().paddedSize())) stub();
                    if (A_block->crosspack > 1) stub();
                }
                if (A_block->colMajor != globalCM || B_block->colMajor != globalCM) stub();
                if (C_block->crosspack > 1) stub();

                if (nv != osys) stub();
                if (nn < ksys) stub();

                // Check if current DPAS can be fused with the previous one.
                bool chain = false;
                if (A0.isValid()) {
                    chain = globalCM ? (elementDiff(hw, B, B0) == (y - ybase) * ksys)
                                     : (elementDiff(hw, A, A0) == (y - ybase) * ksys);
                    chain = chain && (elementDiff(hw, C, C0) == (y - ybase) * osys);
                    chain = chain && (rcount < rcountMax);
                    chain = chain && (hh == hhbase);
                    if (strategy.dpasw)
                        chain = chain && y < ny && (y % (2 * dpaswTile) > 0);
                }

                if (chain)
                    rcount++;
                else {
                    if (strategy.dpasw && y < ny && rcount > 0 && rcount != dpaswTile) stub();
                    if (A0.isValid()) issueDPAS(false);
                    A0 = A; B0 = B; C0 = C; rcount = 1;
                    AS0 = AS; BS0 = BS;
                    A0.setType(Ta.ngen());
                    B0.setType(Tb.ngen());
                    C0.setType(Tc.ngen());
                    ybase = y;
                    hhbase = hh;
                }
            } /* icxCompN loop */
            } /* y1 loop */
            } /* hh loop */
        } /* y loop */
        } /* ocxCompN loop */

        bool finishChain = !strategy.extendedAtomicFMA || (x + osys >= nx) || (repackC && x >= (Cr_unrollX - 2*xinc));
        issueDPAS(finishChain);

        if (endRepackC) {
            int xr = x - Cr_unrollX + xinc;
            if (xr >= 0)
                outerProductRepackC(xr, xr % Cr_unrollX, xinc, h, rem, problem, strategy, state);
        }
    } /* x loop */

    if (endRepackC) for (int xr = nx - Cr_unrollX + xinc; xr < nx; xr += xinc)
        outerProductRepackC(xr, xr % Cr_unrollX, xinc, h, rem, problem, strategy, state);
}

// Repack (part of) a C tile, converting types and scaling as needed.
//  x0 (resp. xr0) are the offsets into the C tile (resp. repacked C tile) in the vectorized dimension.
template <HW hw>
void Generator<hw>::outerProductRepackC(int x0, int xr0, int nx, int h, bool rem,
                                        const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc = problem.Tc, Tc_compute = problem.Tc_compute();
    const auto &C_layout = state.C_layout, &Cr_layout = state.Cr_layout;
    const auto &C_regs = state.C_regs[0], &Cr_regs = state.Cr_regs;
    int nec = elementsPerGRF(hw, Tc_compute);
    bool globalCM = C_layout.colMajor();
    bool scaleA = state.lateScale2DA, scaleB = state.lateScale2DB;

    // Folds raw sums (from systolic or non-systolic accumulateSum) in Asr/Bsr into
    // As/Bs, using the same k-group scale/indexing as the A*B product above.
    // Only systolic/FMA sums are accumulated in Asr/Bsr; others go directly to As/Bs.
    bool sumA = state.systolicSumA || state.fmaSumA;
    bool sumB = state.systolicSumB || state.fmaSumB;
    if (globalCM ? sumB : sumA) stub();

    if (Tc.size() != Tc_compute.size()) stub();
    if (state.C_buffers > 1) stub();

    int ny = strategy.unroll[globalCM ? LoopN : LoopM];
    int nacc = AccumulatorRegister::count(hw, strategy.GRFs, Tc.real().ngen());

    /* Apply late A/B offsets to pre-repacked data */
    bool doAO2DLate = problem.needsBGroupSums();
    bool doBO2DLate = problem.needsAGroupSums();
    if (doAO2DLate || doBO2DLate) {
        auto period = gcd(problem.aqGroupK, problem.bqGroupK);
        bool offsetTime = rem ? ((h % period) < minOuterProductCount(problem, strategy))
                              : (period - (h % period) <= state.cRepackPeriod);
        if (offsetTime) {
            if (doAO2DLate) applyLateABOffset(true,  h, problem, strategy, state, x0, xr0, nx);
            if (doBO2DLate) applyLateABOffset(false, h, problem, strategy, state, x0, xr0, nx);
            if (doAO2DLate && doBO2DLate) stub("Simultaneous late A/B offsets not supported");
        }
    }

    /* Main repacking logic */
    struct WorkItem {
        Subregister C, Cr;
        int simd, iacc;
        std::array<Subregister, 2> scale;
        std::array<int, 2> scaleStride;

        RegData acc(Type T) const {
            return AccumulatorRegister(iacc).retype(T.ngen());
        }
    };

    std::vector<WorkItem> items;
    items.reserve(nacc);

    auto processItems = [&] {
        for (const auto &i: items)
            mov(i.simd, i.acc(Tc), i.Cr(1));
        for (const auto &i : items)
            if (i.scale[1].isValid())
                mul(i.simd, i.acc(Tc), i.acc(Tc), i.scale[1](i.scaleStride[1]));
        for (const auto &i: items) {
            if (i.scale[0].isValid())
                mad(i.simd, i.C(1), i.C(1), i.acc(Tc), i.scale[0](i.scaleStride[0]));
            else
                add(i.simd, i.C(1), i.C(1), i.acc(Tc));
        }
        items.clear();
    };

    int iacc = 0;

    for (int x1 = 0; x1 < nx; x1 += 2 * nec) {
        int x = x0 + x1, xr = xr0 + x1;
        int xchunk = std::min(nx - x1, 2 * nec);
        for (int y = 0; y < ny + sumA + sumB; y++) {
            auto i = globalCM ? x : y;
            auto j = globalCM ? y : x;
            auto ir = globalCM ? xr : y;
            auto jr = globalCM ? y : xr;

            int ne = 0, ner = 0, nes[2] = { 0 };
            const RegisterBlock *C_block, *Cr_block;
            Subregister C, Cr;

            bool doASum = sumA && y == ny;
            bool doBSum = sumB && y == ny;

            if (y < ny) {
                C  =  C_layout.find(i,  j,  C_regs,  &ne,  &C_block);
                Cr = Cr_layout.find(ir, jr, Cr_regs, &ner, &Cr_block);
            } else if (doASum) {
                C  =  state.As_layout.find(x,  0, state.As_regs,  &ne,  &C_block);
                Cr = state.Asr_layout.find(state.systolicSumA ? xr : x, 0, state.Asr_regs, &ner, &Cr_block);
            } else if (doBSum) {
                C  =  state.Bs_layout.find(0, x,  state.Bs_regs,  &ne,  &C_block);
                Cr = state.Bsr_layout.find(0, state.systolicSumB ? xr : x, state.Bsr_regs, &ner, &Cr_block);
            } else stub();

            ne = std::min({ne, ner, xchunk});

            std::array<Subregister, 2> scale;
            std::array<int, 2> scaleStride = {0, 0};
            int nscale = 0;
            if (scaleA && !doBSum) {
                // TODO: handle non-repacked case correctly.
                // TODO: handle M/N grouping < M/N unroll correctly.
                int hs = (h / problem.aqGroupK) % state.kaqLate;
                int is = i / problem.aqGroupM;
                scale[nscale] = state.Ar_scaleLayout.find(is, hs, state.Ar_scaleRegs, &nes[nscale]);
                scaleStride[nscale] = globalCM ? 1 : 0;
                if (nes[nscale] < ne) scaleStride[nscale] = 0;
                nscale++;
            }
            if (scaleB && !doASum) {
                int hs = (h / problem.bqGroupK) % state.kbqLate;
                int js = j / problem.bqGroupN;
                scale[nscale] = state.Br_scaleLayout.find(hs, js, state.Br_scaleRegs, &nes[nscale]);
                scaleStride[nscale] = globalCM ? 0 : 1;
                if (nes[nscale] < ne) scaleStride[nscale] = 0;
                nscale++;
            }

            if (scaleStride[0] == 1) ne = std::min(ne, nes[0]);
            if (scaleStride[1] == 1) ne = std::min(ne, nes[1]);

            if (ne < xchunk) stub();
            if (C_block->crosspack != 1 || Cr_block->crosspack != 1) stub();

            WorkItem item = {C, Cr, ne, iacc, scale, scaleStride};
            bool coalesce = false;

            if (!items.empty()) {
                auto &last = items.back();
                coalesce = Tc.isFP();
                coalesce &= (ne == nec && last.simd == nec);
                coalesce &= (C.getBase() == last.C.getBase() + 1);
                coalesce &= (Cr.getBase() == last.Cr.getBase() + 1);
                for (int i = 0; i < 2; i++) if (scale[i].isValid()) {
                    if (scaleStride[i] == 0)
                        coalesce &= (scale[i] == last.scale[i]);
                    else {
                        coalesce &= (scale[i].getBase() == last.scale[i].getBase() + 1)
                                 && (getBytes(scale[i].getType()) == Tc.size());
                    }
                }
            }

            if (coalesce)
                items.back().simd += ne;
            else
                items.push_back(item);

            iacc += (ne > nec) ? 2 : 1;
            if (Tc.isInteger() && (iacc & 1))
                iacc++; /* only even accumulators allowed for integer types */
            if (iacc >= nacc) {
                processItems();
                iacc = 0;
            }
        }
    }

    processItems();
}

template <HW hw>
void Generator<hw>::applyLateABOffset(bool isA, int h, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int x0, int xr0, int nx)
{
    bool cRepack = !state.Cr_layout.empty();
    const auto &C_layout = cRepack ? state.Cr_layout : state.C_layout;
    const auto &C_regs   = cRepack ? state.Cr_regs   : state.C_regs[0];
    auto cm = C_layout.colMajor();
    auto Tc = problem.Tc;
    auto nec = elementsPerGRF(hw, Tc);
    Subregister A, B;

    if (isA ? !problem.needsBGroupSums() : !problem.needsAGroupSums()) return;

    if (nx < 0) nx = strategy.unroll[cm ? LoopM : LoopN];
    int ny = strategy.unroll[cm ? LoopN : LoopM];

    int haq = (h / problem.aqGroupK) % state.kaqLate;
    int hbq = (h / problem.bqGroupK) % state.kbqLate;

    if (isA ? (problem.aqGroupK < problem.bqGroupK)
            : (problem.bqGroupK < problem.aqGroupK))
        stub("Incompatible group sizes: group sums cannot be used for offset calculation");

    bool scalarA =  isA && (problem.aoPtrDims <= 0);
    bool scalarB = !isA && (problem.boPtrDims <= 0);
    bool scalarV = cm ? scalarA : scalarB;

    if (scalarA) A = state.Ar_offsetLayout.find(0, 0, state.Ar_offsetRegs);
    if (scalarB) B = state.Br_offsetLayout.find(0, 0, state.Br_offsetRegs);

    auto &C0 = C_layout[0];
    int xchunk0 = std::min<int>(2 * nec, C0.colMajor ? C0.nr : C0.nc);
    for (int xx = 0; xx < nx; xx += xchunk0) {
        int xchunk = std::min(nx - xx, xchunk0);
        for (int y = 0; y < ny; y++) {
            auto i = x0 + xx, j = y;
            auto ir = xr0 + xx, jr = y;
            if (!cm) std::swap(i, j), std::swap(ir, jr);

            int na = 2 * nec, nb = 2 * nec, nc;
            auto C = C_layout.find(ir, jr, C_regs, &nc);

            if (!scalarA) {
                A = isA ? state.Ar_offsetLayout.find(i / problem.aqGroupM, haq, state.Ar_offsetRegs, &na)
                        : state.Agr_layout.find(i / problem.aqGroupM, haq, state.Agr_regs, &na);
            }

            if (!scalarB) {
                B = isA ? state.Bgr_layout.find(hbq, j / problem.bqGroupN, state.Bgr_regs, &nb)
                        : state.Br_offsetLayout.find(hbq, j / problem.bqGroupN, state.Br_offsetRegs, &nb);
            }

            int nv = cm ? na : nb;
            auto V = cm ? A : B;
            auto N = cm ? B : A;

            auto ne = std::min({nv, nc, xchunk});
            if (ne != xchunk) stub();

            int vstride = 1;
            auto downcastW = [](Subregister &s) {
                if (s.getType() == DataType::ud)     s = s.uw();
                else if (s.getType() == DataType::d) s = s.w();
                else return false;
                return true;
            };

            if (downcastW(V)) vstride = 2;
            if (scalarV) vstride = 0;

            mad(ne, C(1), C(1), N(0), -V(vstride));
        }
    }
}

GEMMSTONE_NAMESPACE_END
