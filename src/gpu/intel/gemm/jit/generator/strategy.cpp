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

#include "gemmstone/driver_info.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "internal/utils.hpp"
#include "pieces/compute_utils.hpp"
#include "pieces/hw_utils.hpp"
#include "pieces/layout_utils.hpp"
#include "pieces/ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;


/* CommonStrategy member functions */
CommonStrategy::CommonStrategy(HW hw, int stepping) : raHW(hw), emulate(hw, stepping)
{
    fused = one_of(hw, {HW::Gen12LP, HW::XeHP, HW::XeHPG});
    systolicAvailable = (hw >= HW::XeHP);
}

void CommonStrategy::preflight(HW hw, const CommonProblem &problem)
{
    subgroupSize = std::max(subgroupSize, GRF::bytes(hw) >> 2);
    readSuppressionWA &= fused;

    bool emulateNeedsAcc = emulate.emulate64 || emulate.emulateDWxDW || emulate.emulate64_mul;
    if (moveR0 == MoveR0::Acc && emulateNeedsAcc)
        moveR0 = MoveR0::None;
    if (hw >= HW::Xe3p) moveR0 = MoveR0::None;

    spf &= !fused;
}

/* GEMMStrategy member functions */

// Check if a non-named barrier is needed in addition to named barriers.
bool GEMMStrategy::needsUnnamedBarrier(const GEMMProblem &problem) const
{
    if (needsKLoopBarrier() && (!namedBarriers[LoopM] || !namedBarriers[LoopN]))
        return true;
    if (slmBuffers > 0) {
        if (persistentLoop()) return true;
        if (problem.needsASums() || problem.needsBSums()) return true;
    }
    if (kParallelLocal) return true;
    if (fuseBeta || fusePostOps) return true;
    return false;
}

bool GEMMStrategy::needsNamedBarriersM(const GEMMProblem &problem) const
{
    if (!namedBarriers[LoopM])      return false;
    if (slmA || barrierFreq)        return true;
    return false;
}

bool GEMMStrategy::needsNamedBarriersN(const GEMMProblem &problem) const
{
    if (!namedBarriers[LoopN])      return false;
    if (slmB || barrierFreq)        return true;
    return false;
}

// Check if atomic C updates should be automatically used.
bool useAutoAtomic(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool ignoreBeta)
{
    if (!strategy.autoatomic) return false;

    return (hw >= HW::XeHPG)
            && (ignoreBeta || problem.beta1())
            && hasNativeAtomicAdd(hw, problem.Tc_ext.real(), problem.C, strategy.C)
            && (problem.Tacc_min == Type::invalid || problem.Tacc_min.isSubsetOf(problem.Tc_ext))
            && !strategy.cLoadAhead
            && (problem.postOps.len() == 0 || problem.hasSum1PostOpAtEnd())
            && (problem.cOffset != COffset::Post)
            && !isBlock2D(strategy.C.accessType);
}

// Validate a GEMM strategy, correcting settings as necessary.
void GEMMStrategy::preflight(HW hw, const GEMMProblem &problem)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc, Tc_ext = problem.Tc_ext;
    auto Ta_real = Ta.real();

    // Safety checks for alignment.
    if (!legalAAlignment(problem, problem.A.alignment))
        stub("A alignment will be lost during m-parallelization");
    if (!legalBAlignment(problem, problem.B.alignment))
        stub("B alignment will be lost during n-parallelization");

    // Addressing preflight.

    if (hw >= HW::Xe2) for (auto *s: {&A, &B, &C, &AO, &BO, &CO, &A_scale, &B_scale,
                                      &A_prefetch, &B_prefetch, &C_prefetch, &AB_prefetchL3})
        s->newDP = true;

    if (isBlock2D(A_prefetch.accessType) && !isPacked(problem.A.layout) && !A_prefetch.address2D)
        downgradeAPFAccess(problem, *this);
    if (isBlock2D(B_prefetch.accessType) && !isPacked(problem.B.layout) && !B_prefetch.address2D)
        downgradeBPFAccess(problem, *this);

    // Remove variable k-parallelization when batching.
    if (kParallelVariable && problem.batch != BatchMode::None)
        C.atomic = CO.atomic = kParallelVariable = kParallelLocal = false;

    // Stream-K combines per-workgroup partial C via atomic add, which is
    // safe for plain grouped scaling (no cross-workgroup shared state) but
    // not late 2D scaling.
    bool aLateScale = usesLateScale(problem, *this, /*isA=*/true);
    bool bLateScale = usesLateScale(problem, *this, /*isA=*/false);
    bool lateScaleHazard = (aLateScale || bLateScale) && problem.Tc_ext.isInteger();
    if (kParallelVariable && lateScaleHazard)
        C.atomic = CO.atomic = kParallelVariable = kParallelLocal = false;

    // Offsets applied after the scales are fused with early 2D scales into one mad.
    for (bool isA: {true, false}) {
        bool offsetAfterScale = isA ? problem.aOffsetAfterScale : problem.bOffsetAfterScale;
        bool xo2D = isA ? problem.aOffset2D() : problem.bOffset2D();
        bool xs2D = isA ? problem.aScale2D() : problem.bScale2D();
        bool lateOffset = isA ? problem.needsBGroupSums() : problem.needsAGroupSums();
        bool lateScale = isA ? aLateScale : bLateScale;
        if (offsetAfterScale && (!xo2D || !xs2D || lateOffset || lateScale))
            stub("Offsets after scales require early 2D offsets and scales");
    }

    C.atomic |= useAutoAtomic(hw, problem, *this);

    if (C.atomic && !C.base.isStateless() && !C.newDP)
        C.forceA64();

    // Fused EU handling.
    if (fusedLoop == LoopM && (wg[LoopM] & 1))
        fusedLoop = LoopN;
    if (fusedLoop == LoopN && (wg[LoopN] & 1))
        fusedLoop = LoopM;

    // Fused beta/post-op configuration.
    fuseBeta &= (kParallel || kParallelVariable);
    fusePostOps &= (kParallel || kParallelVariable);
    relaxedAccumulation &= hasNativeAtomicAdd(hw, Tc_ext, problem.C, C);

    bool needsFusedPostOps = false;

    needsFusedPostOps |= (problem.cOffset == COffset::Post);
    if (!relaxedAccumulation)
        needsFusedPostOps |= (Tc.bits() != Tc_ext.bits());
    for (size_t i = 0; i < problem.postOps.len(); i++)
        needsFusedPostOps |= (!problem.postOps[i].is_sum());
    if (problem.Ts != problem.Tc) {
        needsFusedPostOps |= !(problem.alpha1() || problem.alphaM1());
        needsFusedPostOps |= !(problem.beta0()  || problem.beta1());
    }

    fusePostOps &= C.atomic;
    fusePostOps &= needsFusedPostOps;

    fuseBeta &= C.atomic;
    fuseBeta &= !problem.beta1();

    fuseBeta |= (fusePostOps && needsTempC(problem));

    zeroTempC &= needsTempC(problem);
    fuseBeta &= !zeroTempC;

    altFusedBeta &= fuseBeta;

    if (!(kParallelVariable || (kParallel && altFusedBeta)))
        kPadding = 0;

    slmA &= (slmBuffers > 0);
    slmB &= (slmBuffers > 0);

    A.preflight(hw); A_prefetch.preflight(hw); AO.preflight(hw);
    B.preflight(hw); B_prefetch.preflight(hw); BO.preflight(hw);
    C.preflight(hw); C_prefetch.preflight(hw); CO.preflight(hw);
    A_scale.preflight(hw); Ag.preflight(hw);
    B_scale.preflight(hw); Bg.preflight(hw);
    AB_prefetchL3.preflight(hw);

    bool globalCM = isRegisterColMajor(problem.Tc, problem.C, C);

    altCRemainder &= (Tc_ext.bits() >= 8);

    block2DCRemainder &= (hw >= HW::XeHPC);
    block2DCRemainder &= !isPacked(problem.C.layout);
    block2DCRemainder &= !isBlock2D(C.accessType);
    auto X = unroll[isTransposing(C.accessType) ? LoopN : LoopM];
    block2DCRemainder &= (X * problem.Tc_ext) % 4 == 0;
    block2DCFull |= (Tc_ext.paddedSize() < 4);
    block2DCFull &= block2DCRemainder;

    extendedAtomicFMA &= !problem.needsASums() && !problem.needsBSums();
    if (systolic)
        extendedAtomicFMA &= ((unroll[globalCM ? LoopN : LoopM]) % 8 == 0);

    if ((scramble[LoopM] || scramble[LoopN] || tlbWarmup) && !linearOrder())
        cWalkOrder = WalkOrder::SimpleLinear;

    // Default SIMD setting.
    if (fmaSIMD == 0) {
        fmaSIMD = std::min(32, 2 * GRF::bytes(hw) / std::max<int>({Ta.paddedSize(), Tb.paddedSize(), Tc.paddedSize()}));
    }

    slmFenceWARWA |= (hw >= HW::XeHPG);

    if (problem.batch != BatchMode::None) {
        persistent = false;
        kParallel = false;
    }

    if (coopA == CoopSplit::K && slmATrans) coopA = CoopSplit::MN;
    if (coopB == CoopSplit::K && slmBTrans) coopB = CoopSplit::MN;

    checkBeta1 |= C.atomic && !problem.beta1();

    GRFs = std::min(GRFs, GRF::maxRegs(hw));
    // CRI has unique GRF mode behaviour - automatically update to optimal options
    if (problem.product.family == ngen::ProductFamily::CRI) {
        if (GRFs == 128) GRFs = 192;
        if (GRFs == 256) GRFs = 512;
    }

    // Fixed systolic kernel handling.
    if (fixedSystolic) {
        if (wg[LoopM] == 0) wg[LoopM] = 4;
        if (wg[LoopN] == 0) wg[LoopN] = 4;
        bool doubleM = (wg[LoopM] == 8);

        slmCopies = (slmCopies == 3) ? 3 : 1;
        slmBuffers = (splitCopy || doubleM) ? 4 : 3;
        slmA = slmB = true;
        GRFs = 256;
        altCRemainder = false;
        loopOrder[0] = LoopM;
        loopOrder[1] = LoopN;
        loopOrder[2] = LoopK;
        A.accessType = B.accessType = AccessType::Block;
        ka_load = kb_load = 32 / Ta_real;
        dpasw = true;
    }

    dpasw &= systolic && fused;

    if (kChain == 0) {
        kChain = 1;
        if (!cAccumulators) {
            if (!systolic && !dotVL && Ta.isFP() && Tb.isFP() && hw >= HW::XeHP)
                kChain = gcd(ka_load, kb_load);
            if (systolic && problem.product.family == ProductFamily::CRI) {
                int minOPCount = minOuterProductCount(problem, *this);
                kChain = std::max(1, gcd(ka_load, kb_load) / minOPCount);
            }
        }
    }

    if (!systolic && AccumulatorRegister::count(hw, GRFs, problem.Tc.real().ngen()) == 0)
        kChain = 1;
    // Late 2D scales are only read once per C repack, which occurs every
    // gcd(aqGroupK, bqGroupK) k-elements. A k-chain spanning more than that would
    // skip scale groups, so trim it to fit.
    int minOPC = minOuterProductCount(problem, *this);
    if (kChain > 1 && problem.forceLateQuant(minOPC)) {
        int period = problem.aScale2D() ? problem.aqGroupK : 0;
        if (problem.bScale2D())
            period = period ? gcd(period, problem.bqGroupK) : problem.bqGroupK;
        if (kInterleave) period = gcd(period, kInterleaveChunk);
        if (period % minOPC == 0)
            kChain = gcd(kChain, period / minOPC);
    }
    cAccumulators &= (kChain == 1);

    // Accumulator usage: 64-bit emulation, or k chaining, or extra C registers, or storage for r0 header.
    // Priority: k chaining > extra C registers > r0 header storage.
    //                         64-bit emulation > r0 header storage.
    bool emulateNeedsAcc = emulate.emulate64 || emulate.emulateDWxDW;
    if (moveR0 == MoveR0::Acc)
        if (cAccumulators || emulateNeedsAcc || xParallel || (kChain > 1) || barrierFreq || fuseBeta)
            moveR0 = MoveR0::None;

    // SIMT control flow is used by jump tables, (emulated) atomics, and double masking.
    spf &= !noJumpTables;
    spf &= !C.atomic;
    spf &= !doubleMasking;

    checkAdd32 &= !emulate.emulate64_add32;
    checkAdd32 &= (A.base.isStateless() || B.base.isStateless() || problem.quantized2DA() || problem.quantized2DB());
    checkAdd32 &= !(A.address2D && B.address2D && (!prefetchA || A_prefetch.address2D) && (!prefetchB || B_prefetch.address2D));

    int opCount = outerProductCount(problem, *this);
    int minOPCount = minOuterProductCount(problem, *this);
    int ukAlign = opCount;

    if (kParallelLocal)
        moveR0 = MoveR0::None;

    // SLM copy logic.
    int slmVersions = std::max(1, lcm(slmCopies, slmBuffers));
    if (slmBuffers > 0) {
        moveR0 = MoveR0::None;
        barrierFreq = 0;
        if (wg[LoopM] <= 0 || wg[LoopN] <= 0)
            stub("Workgroup sizes required.");
        if (slmA) ukAlign = lcm(ukAlign, wg[LoopN] * slmVersions);
        if (slmB) ukAlign = lcm(ukAlign, wg[LoopM] * slmVersions);
        slmUseIncrCopy &= (slmCopies == 1);
    }

    // ka/kb_load wranging.
    if (ka_load_masked == 0) ka_load_masked = ka_load;
    if (kb_load_masked == 0) kb_load_masked = kb_load;

    int maskedOPCount = minOPCount;
    if(problem.product.family == ProductFamily::CRI && kChain >= 2)
        // On CRI, dpas fwd modifier is required for optimal compute throughput
        maskedOPCount *= 2;

    ka_load = align_up(ka_load, opCount);
    kb_load = align_up(kb_load, opCount);
    ka_load_masked = align_up(ka_load_masked, maskedOPCount);
    kb_load_masked = align_up(kb_load_masked, maskedOPCount);

    if (ka_load == 0 || kb_load == 0)
        stub("Zero k-load granularity is invalid");

    // u3 Block2DTranspose loads address k in whole dwords (32 elements = 3 dwords),
    //  so each k load must cover a whole number of dwords.
    auto u3DwordAligned = [](int k) { return (k * 3) % 32 == 0; };
    if (problem.Ta_ext.is3() && A.accessType == AccessType::Block2DTranspose
            && !(u3DwordAligned(ka_load) && u3DwordAligned(ka_load_masked)))
        stub("u3 Block2DTranspose A loads must be dword-aligned in k");
    if (problem.Tb_ext.is3() && B.accessType == AccessType::Block2DTranspose
            && !(u3DwordAligned(kb_load) && u3DwordAligned(kb_load_masked)))
        stub("u3 Block2DTranspose B loads must be dword-aligned in k");

    // Systolic handling.
    if (systolic) {
        auto params = systolicParams(problem);

        ukAlign = lcm(ukAlign, params.ksys);
        auto tileX = params.osys;
        (globalCM ? C.tileR : C.tileC) = tileX;
        if (unroll[globalCM ? LoopM : LoopN] > tileX && isBlocklike(C.accessType))
            forceCopyC = true;
        dotVL = 0;
    }

    // Dot product handling.
    if (dotVL) {
        dotVL = align_up(dotVL, elementsPerGRF(hw, Tc));
        forceCopyC = doubleMasking = true;
    }

    // Prefetch handling.
    cooperativePF &= (prefetchA || prefetchB);

    if (problem.beta0())
        prefetchC = 0;
    else if (prefetchC && C.atomic)
        C_prefetch.cachingR = makeL1Uncacheable(C_prefetch.cachingR);

    if (prefetchABL3 && cWalkOrder == WalkOrder::HW2D)
        cWalkOrder = WalkOrder::SimpleLinear;

    // Propagate tiling requests to strategy.
    int tileM_A, tileK_A, tileK_B, tileN_B;
    std::tie(tileM_A, tileK_A, tileK_B, tileN_B) = targetKernelTiling(hw, problem, *this);
    if (A.accessType != AccessType::Block) {
        if (tileM_A && !A.tileR) A.tileR = tileM_A;
        if (tileK_A && !A.tileC) A.tileC = tileK_A;
    }
    if (B.accessType != AccessType::Block) {
        if (tileK_B && !B.tileR) B.tileR = tileK_B;
        if (tileN_B && !B.tileC) B.tileC = tileN_B;
    }

    if (dpasw) {
        auto params = systolicParams(problem);
        if (globalCM) {
            if (!fusedM()) stub();
            B.dpasw = true;
            B.tileC = std::max(1, std::min(unroll[LoopN], params.rcountMax) / 2);
            if (unroll[LoopN] % (2 * B.tileC))
                stub("Cannot use dpasw for this n tile size");
        } else {
            if (!fusedN()) stub();
            A.dpasw = true;
            A.tileR = std::max(1, std::min(unroll[LoopM], params.rcountMax) / 2);
            if (unroll[LoopM] % (2 * A.tileR))
                stub("Cannot use dpasw for this m tile size");
        }
    }

    // Always use 1D addressing for packed inputs.
    A.address2D &= !isPacked(problem.A.layout);
    B.address2D &= !isPacked(problem.B.layout);

    // SLM staging must cover the k-loads, on the main and the masked path.
    int minUnrollKSLM = 1, minUnrollKSLMMasked = 1;
    if (slmA) {
        minUnrollKSLM = lcm(minUnrollKSLM, ka_load);
        minUnrollKSLMMasked = lcm(minUnrollKSLMMasked, lcm(ka_load, ka_load_masked));
    }
    if (slmB) {
        minUnrollKSLM = lcm(minUnrollKSLM, kb_load);
        minUnrollKSLMMasked = lcm(minUnrollKSLMMasked, lcm(kb_load, kb_load_masked));
    }

    if (unrollKSLM > 0) minUnrollKSLM = unrollKSLM = lcm(unrollKSLM, minUnrollKSLM);
    if (unrollKSLMMasked > 0) unrollKSLMMasked = lcm(unrollKSLMMasked, minUnrollKSLMMasked);

    // k interleaving chunk size.
    if (kInterleave) {
        int kchunk0 = lcm(ka_inc(), kb_inc());
        if (prefetchA) kchunk0 = lcm(kchunk0, ka_pfStride);
        if (prefetchB) kchunk0 = lcm(kchunk0, kb_pfStride);
        if (problem.quantized2DA()) {
            if (problem.aqGroupK % wg[LoopK]) stub();
            kchunk0 = lcm(kchunk0, std::max(1, problem.aqGroupK / wg[LoopK]));
        }
        if (problem.quantized2DB()) {
            if (problem.bqGroupK % wg[LoopK]) stub();
            kchunk0 = lcm(kchunk0, std::max(1, problem.bqGroupK / wg[LoopK]));
        }
        kInterleaveChunk = align_up(kInterleaveChunk, kchunk0);
        kInterleaveChunk = std::max(kInterleaveChunk, kchunk0);
    }

    // k unroll wrangling.
    ukAlign = lcm(ukAlign, A_copies * ka_load);
    ukAlign = lcm(ukAlign, B_copies * kb_load);
    if (slmCopies > 1) {
        ukAlign = lcm(ukAlign, slmCopies * ka_load);
        ukAlign = lcm(ukAlign, slmCopies * kb_load);
    }
    if (ka_pfStride) ukAlign = lcm(ukAlign, ka_pfStride);
    if (kb_pfStride) ukAlign = lcm(ukAlign, kb_pfStride);

    ukAlign = lcm(ukAlign, minUnrollKSLM * slmVersions);

    if (kInterleave) ukAlign = lcm(ukAlign, kInterleaveChunk);
    if (repackC) ukAlign = lcm(ukAlign, repackC);

    if (problem.quantized2DA()) ukAlign = lcm(ukAlign, problem.aqGroupK);
    if (problem.quantized2DB()) ukAlign = lcm(ukAlign, problem.bqGroupK);
    if (l3PrefetchA) ukAlign = lcm(ukAlign, ka_prefetchL3);
    if (l3PrefetchB) ukAlign = lcm(ukAlign, kb_prefetchL3);

    unroll[LoopK] = align_up(unroll[LoopK], ukAlign);

    if (unrollKSLM == 0)
        unrollKSLM = unroll[LoopK] / slmVersions;

    if (fixedSystolic)
        unroll[LoopK] = unrollKSLM = 32 / Ta_real;

    barrierFreq = align_up(barrierFreq, unroll[LoopK]);
    prefetchABL3 = align_up(prefetchABL3, unroll[LoopK]);

    int kChunkA = (problem.A.tileC ? problem.A.tileC : problem.A.crosspack);
    int kChunkB = (problem.B.tileR ? problem.B.tileR : problem.B.crosspack);
    if (unroll[LoopK] <= std::min(kChunkA, kChunkB))
        remHandling[LoopK] = RemainderHandling::Ignore;

    // Default blocking.
    bool isZ = problem.Tc.size() >= 16;
    auto defaultMBlock = isZ ? 2048 : 4096;
    if (hw >= HW::XeHP) defaultMBlock *= 2;
    auto defaultNBlock = defaultMBlock;
    auto defaultMBlockNonHilbert = defaultMBlock;
    auto defaultNBlockNonHilbert = defaultNBlock;

    /* No more than (2^16 - 1) workgroups in m/n dimensions for linear orders, plus a huge safety margin. */
    if (linearOrder()) {
        defaultMBlock = 16384 * unroll[LoopM];
        defaultNBlock = 16384 * unroll[LoopN];
    }

    if (blocking[LoopM] <= 0) blocking[LoopM] = defaultMBlock;
    if (blocking[LoopN] <= 0) blocking[LoopN] = defaultNBlock;
    if (blocking[LoopK] <= 0) {
        if (hw >= HW::XeHPG)
            blocking[LoopK] = 16777216;
        else {
            int points = 1;
            if (slmA || (problem.A.layout != MatrixLayout::T)) points++;
            if (slmB || (problem.B.layout != MatrixLayout::N)) points++;
            blocking[LoopK] = std::min(2048, (2048 * points) / problem.Ta);
        }
    }

    auto defaultBlockAltK = blocking[LoopK];
    if (hw == HW::XeHP) defaultBlockAltK = std::min(defaultBlockAltK, 1024);

    if (hw > HW::XeHP) {
        defaultMBlockNonHilbert = defaultMBlock;
        defaultNBlockNonHilbert = defaultNBlock;
    }

    if (blockingAlt[LoopM] <= 0) blockingAlt[LoopM] = defaultMBlockNonHilbert;
    if (blockingAlt[LoopN] <= 0) blockingAlt[LoopN] = defaultNBlockNonHilbert;
    if (blockingAlt[LoopK] <= 0) blockingAlt[LoopK] = defaultBlockAltK;

    /* Block 2D is limited to matrices up to 2^24 elements in each dimension */
    bool a2D = isBlock2D(A.accessType) || (prefetchA && isBlock2D(A_prefetch.accessType));
    bool b2D = isBlock2D(B.accessType) || (prefetchB && isBlock2D(B_prefetch.accessType));
    bool c2D = isBlock2D(C.accessType) || (prefetchC && isBlock2D(C_prefetch.accessType));

    if (a2D || c2D) blocking[LoopM] = std::min(blocking[LoopM], 1 << 24);
    if (b2D || c2D) blocking[LoopN] = std::min(blocking[LoopN], 1 << 24);
    if (a2D || b2D) blocking[LoopK] = std::min(blocking[LoopK], 1 << 24);

    // Default workgroups.
    auto defaultWGX = 2, defaultWGY = 8;

    if (wg[loopOrder[0]] <= 0) wg[loopOrder[0]] = defaultWGX;
    if (wg[loopOrder[1]] <= 0) wg[loopOrder[1]] = defaultWGY;
    if (wg[LoopK] <= 0) {
        if (kParallelLocal)
            wg[LoopK] = (threadsPerEU(problem.product, *this) * eusPerSubslice(hw)) / (wg[LoopM] * wg[LoopN]);
        else
            wg[LoopK] = 1;
    }

    kParallelLocal &= (wg[LoopK] > 1);
    if (!kParallelLocal)
        wg[LoopK] = 1;

    skewLocalIDs &= (wg[LoopM] * wg[LoopN] > eusPerSubslice(hw));

    if (skewLocalIDs) forceWGUpdate = WGFixed;

    avoidIncConflicts &= (hw >= HW::XeHP);

    kPadding = align_up(kPadding, kAlign(problem));

    if (fixedWG(problem) && (!kParallelLocal || fixedWGK()))
        activeThreads = wg[LoopM] * wg[LoopN] * wg[LoopK] * (splitCopy ? 2 : 1);

    CommonStrategy::preflight(hw, problem);
}

// Reduce register pressure. Returns true if successful.
bool GEMMStrategy::minimize(HW hw, const GEMMProblem &problem)
{
    bool better = false;
    auto minOPCount = minOuterProductCount(problem, *this);
    auto ka_load_best_min = std::max<int>({1, 4 / problem.Ta, minOPCount});
    auto kb_load_best_min = std::max<int>({1, 4 / problem.Tb, minOPCount});

    // Reduce ka/b_load down to suggested minimums (not requiring crosspack)
    if (ka_load > ka_load_best_min) {
        ka_load = ka_load_best_min;
        better = true;
    }
    if (kb_load > kb_load_best_min) {
        kb_load = kb_load_best_min;
        better = true;
    }

    // Reduce A/B copies.
    A_copies = B_copies = 1;

    // Remove k chaining.
    kChain = 1;

    // Reduce k unroll for SLM copies.
    if (slmA || slmB) {
        auto oldUK = unroll[LoopK];
        unroll[LoopK] = 1;
        unrollKSLM = 0;
        preflight(hw, problem);
        better |= (unroll[LoopK] < oldUK);
    }

    if (better)
        return better;

    // Reduce ka/b_load to absolute minimum if that failed.
    if (ka_load > minOPCount) {
        ka_load = minOPCount;
        better = true;
    }
    if (kb_load > minOPCount) {
        kb_load = minOPCount;
        better = true;
    }

    return better;
}

// Return preferred k alignment for this kernel (avoiding remainder loads).
int GEMMStrategy::kAlign(const GEMMProblem &problem) const
{
    int align = lcm(ka_load, kb_load);
    align = lcm(align, extraKAlign);
    if (slmBuffers > 0) align = lcm(align, unrollKSLM);

    if (kParallelLocal && kInterleave) {
        /* Ensure k0 is aligned to prefetch strides */
        align = lcm(align, kInterleaveChunk);
    }

    if (problem.quantized2DA()) align = lcm(align, problem.aqGroupK);
    if (problem.quantized2DB()) align = lcm(align, problem.bqGroupK);

    return align;
}

// Check if this strategy needs a temporary C buffer.
bool GEMMStrategy::needsTempC(const GEMMProblem &problem) const
{
    if (!fusePostOps) return false;
    if (problem.Ts != problem.Tc) {
        if (!problem.alpha1() && !problem.alphaM1()) return true;
        if (!problem.beta0() && !problem.beta1()) return true;
    }
    if (problem.Tc.bits() != problem.Tc_ext.bits()) return true;
    if (!problem.beta0() && !problem.beta1() && altFusedBeta) return true;
    for (size_t i = 1; i < problem.postOps.len(); i++)
        if (problem.postOps[i].is_sum())
            return true;
    return false;
}

// Check if this strategy is nondeterministic.
bool GEMMStrategy::nondeterministic(const GEMMProblem &problem) const {
    if (!problem.Tc.isInteger()) {
        if (kParallel) return true;
        if (kParallelVariable && !altFusedBeta) return true;    /* Note: may still be nondeterministic with alt fused beta;
                                                                         handled by kernel selector. */
    }
    if (problem.sumA && slmA && coopA == CoopSplit::K && wg[LoopN] > 2) return true;
    if (problem.sumB && slmB && coopB == CoopSplit::K && wg[LoopM] > 2) return true;
    return false;
}

void MatrixAddressingStrategy::preflight(HW hw)
{
    newDP |= isBlock2D(accessType) || (hw >= HW::Xe2);
    padded |= (base.getModel() == ModelSLM);

    // Xe3p uses sendg[x] instructions: downgrade surface accesses to a64
    if (hw == HW::Xe3p && base.getModel() == ModelBTS)
        forceA64();

    if (prefetch && newDP && cachingR == CacheSettingsLSC::Default)
        cachingR = CacheSettingsLSC::L1C_L3C;

    if (accessType == AccessType::ChannelScattered && base.isStateless() && !newDP)
        base = AddressBase::createBTS(0);
}

void MatrixAddressingStrategy::forceA64()
{
    base = AddressBase::createA64(true);
    padded = false;
    if (accessType == AccessType::ChannelScattered && !newDP)
        accessType = AccessType::Scattered;
}

static inline void downgradePFAccess(AccessType &atype, int &k_prefetch, bool transposing, int unrollBytes)
{
    if (unrollBytes <= 64)
        atype = AccessType::Scattered;
    else if (transposing) {
        atype = AccessType::Scattered;
        k_prefetch = 1;
    } else
        atype = AccessType::Block;
}

void downgradeAPFAccess(const GEMMProblem &problem, GEMMStrategy &strategy)
{
    downgradePFAccess(strategy.A_prefetch.accessType, strategy.ka_prefetch,
                      problem.A.layout == MatrixLayout::T, strategy.unroll[LoopM] * problem.Ta_ext);
}

void downgradeBPFAccess(const GEMMProblem &problem, GEMMStrategy &strategy)
{
    downgradePFAccess(strategy.B_prefetch.accessType, strategy.kb_prefetch,
                      problem.B.layout == MatrixLayout::N, strategy.unroll[LoopN] * problem.Tb_ext);
}

void GEMMStrategy::trimKChain(HW hw, int k, const GEMMProblem &problem)
{
    int minOPCount = minOuterProductCount(problem, *this);
    kChain = gcd(kChain, k / minOPCount);
}


GEMMSTONE_NAMESPACE_END
