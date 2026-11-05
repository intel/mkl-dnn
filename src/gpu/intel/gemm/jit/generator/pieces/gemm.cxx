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


#include "atomic_fusions.hpp"
#include "cooperative_split.hpp"
#include "gemmstone/generator.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "state_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


// Create a GEMM kernel.
template <HW hw>
void Generator<hw>::gemm(GEMMProblem problem, GEMMStrategy strategy, const InterfaceHandler &interface_)
{
    GEMMState state(hw, strategy);
    interface = interface_;
    gemm(problem, strategy, state);
}

template <HW hw>
void Generator<hw>::gemm(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    const bool inFusedGEMM = false;
    bool anyKParallelFixed = strategy.kParallelLocal || strategy.kParallel;

    Label lKernelDone, lReentry, lLeavePersistentLoop, lPadThread, lKVPhaseDone;

    // By default, don't use dispatch mask.
    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Set up.
    problem.autoTypeConversions(strategy.systolic);
    gemmInitState(problem, strategy, state);

    // Transfer surface indices to strategy AddressBases.
    strategy.A.assignSurface(state.inputs.surfaceA);
    strategy.B.assignSurface(state.inputs.surfaceB);
    strategy.C.assignSurface(state.inputs.surfaceC[0]);
    state.Cext_strategy.assignSurface(state.inputs.surfaceC[0]);

    if (!strategy.C.base.isStateless() && state.C_count > 1) stub();
    if (state.useTempC)
        state.tempCStrategy.assignSurface(state.inputs.surfaceTempC);
    if (problem.usesCOPtr())
        strategy.CO.assignSurface(state.inputs.surfaceCO);

    for (size_t i = 0; i < strategy.binary.size(); i++)
        strategy.binary[i].assignSurface(state.inputs.binarySurfaces[i]);

    strategy.AO.assignSurface(state.inputs.surfaceAO);
    strategy.BO.assignSurface(state.inputs.surfaceBO);
    strategy.A_scale.assignSurface(state.inputs.surfaceAScale);
    strategy.B_scale.assignSurface(state.inputs.surfaceBScale);
    strategy.Ag.assignSurface(state.inputs.surfaceAg);
    strategy.Bg.assignSurface(state.inputs.surfaceBg);

    // Prologue.
    if (!inFusedGEMM)
        prologue(strategy, state);

    // Grab fused ID if needed, and multiply by unroll.
    getFusedID(strategy.unroll[strategy.fusedLoop], problem, strategy, state);

    if (!inFusedGEMM) {
        // Divide out subgroup size from local size 0 and local ID 0, and reorder threads for fusing if needed.
        removeSG(problem, strategy, state);
        reorderFusedEUs(problem, strategy, state);

    } /* !inFusedGEMM */

    // Initialize flags if needed and not provided as a kernel argument.
    bool needsFlags = strategy.kParallelVariable || strategy.fuseBeta || strategy.fusePostOps
                    || problem.sumA || problem.sumB;

    if (needsFlags && !state.inputs.flags.isValid()) {
        state.inputs.flags = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        mov(1, state.inputs.flags, 0);
    }

    // Check for copy or compute kernel.
    auto wgY = strategy.wg[strategy.loopOrder[1]];
    auto &localIDY = (strategy.loopOrder[1] == LoopN) ? state.lidN : state.lidM;
    if (strategy.splitCopy) {
        state.isCompute = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        cmp(1 | ge | f1[1], state.isCompute, localIDY, wgY);
        if (is_zero_or_pow2(wgY))
            and_(1, localIDY, localIDY, wgY - 1);
        else
            add(1 | f1[1], localIDY, localIDY, -wgY);
    }

    // Check if this is a padding thread, and exit if so.
    if (strategy.wgPadFactor > 1) {
        cmp(1 | ge | f1[0], localIDY, wgY * (strategy.splitCopy ? 2 : 1));
        jmpi(1 | f1[0], lPadThread);
    }

    // Surface handling for quantization parameters.
    auto replace0 = [&](Subregister &s) {
        if (s.isValid())
            state.ra.release(s);
        s = state.ra.alloc_sub<uint32_t>();
        mov(1, s, 0);
    };

    if (!strategy.AO.base.isStateless() && problem.aOffset2D())          replace0(state.inputs.aoPtr);
    if (!strategy.BO.base.isStateless() && problem.bOffset2D())          replace0(state.inputs.boPtr);
    if (!strategy.A_scale.base.isStateless() && problem.aScale2D())      replace0(state.inputs.aScalePtr);
    if (!strategy.B_scale.base.isStateless() && problem.bScale2D())      replace0(state.inputs.bScalePtr);
    if (!strategy.Ag.base.isStateless() && problem.needsSeparateASums()) replace0(state.inputs.agPtr);
    if (!strategy.Bg.base.isStateless() && problem.needsSeparateBSums()) replace0(state.inputs.bgPtr);

    // Scale LDs/offsets.
    gemmScaleInputs(problem, strategy, state);

    // Check if this is a TLB warmup thread, and perform warmup if so.
    if (strategy.tlbWarmup) {
        Label lNotTLBWarmup;
        state.groupIDMN = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        add(1 | ge | f1[0], state.groupIDMN.d(), state.inputs.groupIDMN, -1);
        jmpi(1 | f1[0], lNotTLBWarmup);
        status << "TLB warmup" << status_stream::endl;
        auto mstate = state;
        moveR0(strategy, mstate);
        gemmGetBatchIDs(problem, strategy, mstate);
        gemmOffsetBatchABC(problem, strategy, mstate);
        gemmSetupABC(problem, strategy, mstate);
        gemmTLBWarmup(problem, strategy, mstate);
        epilogue(strategy, mstate);
        mark(lNotTLBWarmup);
    }

    // Local ID handling and saving.
    gemmReorderLocalIDs(problem, strategy, state);

    if (strategy.needsMNLocalIDs())
        saveMNLocalIDs(strategy, state);

    if (strategy.needsKLocalIDs())
        saveKLocalIDSize(strategy, state);

    // Save full k size if needed.
    bool anyAB2D = strategy.A.address2D || strategy.B.address2D
                || (strategy.prefetchA && strategy.A_prefetch.address2D)
                || (strategy.prefetchB && strategy.B_prefetch.address2D);
    if (anyKParallelFixed || strategy.kParallelVariable) {
        if (strategy.persistentLoop() || strategy.fuseBeta || strategy.fusePostOps || anyAB2D) {
            state.fullK = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            mov(1, state.fullK, state.inputs.k);
        }
    } else
        state.fullK = state.inputs.k;

    // Variable k-slicing initialization.
    if (strategy.kParallelVariable) {
        state.k0Rem = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        state.inputs.kSlicedTiles = state.inputs.kvConfig.uw(0);
        state.inputs.kSyncSlabs = state.inputs.kvConfig.uw(1);
        and_(1 | nz | f1[0], null.uw(), state.inputs.kSyncSlabs, 0x8000);
        and_(1, state.inputs.kSyncSlabs, state.inputs.kSyncSlabs, 0x7FFF);
        or_(1 | f1[0], state.inputs.flags, state.inputs.flags, FlagKSlice2);
    }

    // L3 prefetch warmup.
    if (strategy.prefetchABL3) {
        gemmInitL3Prefetch(false, problem, strategy, state);
        gemmWarmupL3Prefetch(problem, strategy, state);
    }

    // Compute group count if needed and not passed as an argument.
    if (state.groupCountMN.isInvalid() && (strategy.persistent || strategy.kParallelVariable)) {
        state.groupCountMN = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        if (strategy.kParallelVariable)
            emul(1, state.groupCountMN, state.inputs.groupCountM, state.inputs.groupCountN, strategy, state);
    }

    // A/B offset pointer handling.
    bool aOffset = (problem.aOffset != ABOffset::None);
    bool bOffset = (problem.bOffset != ABOffset::None);
    if (aOffset && state.inputs.offsetAO.isValid())
        eadd(1, state.inputs.aoPtr, state.inputs.aoPtr, state.inputs.offsetAO, strategy, state);
    if (bOffset && state.inputs.offsetBO.isValid())
        eadd(1, state.inputs.boPtr, state.inputs.boPtr, state.inputs.offsetBO, strategy, state);

    state.ra.safeRelease(state.inputs.offsetAO);
    state.ra.safeRelease(state.inputs.offsetBO);

    // Load scalar ao/bo from memory as needed.
    bool aoScalarLoad = aOffset && problem.aoPtrDims == 0 && !problem.earlyDequantizeA() && !problem.needsBGroupSums();
    bool boScalarLoad = bOffset && problem.boPtrDims == 0 && !problem.earlyDequantizeB() && !problem.needsAGroupSums();
    auto Tc = problem.Tc;

    if (Tc.isInteger() && (aoScalarLoad || boScalarLoad)) {
        state.inputs.abo = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        if (aoScalarLoad) state.inputs.ao = state.inputs.abo.w(0);
        if (boScalarLoad) state.inputs.bo = state.inputs.abo.w(1);
    } else {
        if (aoScalarLoad) state.inputs.ao = state.ra.alloc_sub(Tc.ngen(), getHint(HintType::LongTerm, strategy));
        if (boScalarLoad) state.inputs.bo = state.ra.alloc_sub(Tc.ngen(), getHint(HintType::LongTerm, strategy));
    }

    auto loadABO = [&](Type T, const ngen::Subregister &xo, ngen::Subregister &xoPtr) {
        if (xoPtr.isInvalid())
            mov(1, xo, cast(Tc, 0));
        else {
            vector<Subregister> srcs;
            srcs.push_back(xoPtr);
            auto xoLoad = loadScalars(T, srcs, strategy, state);
            if (T.isInteger() && T.paddedSize() > 2)
                xoLoad = xoLoad.w();
            if (T.isInteger() && T.paddedSize() == 1 && Tc.isFP()
                    && one_of(getProductFamily(), {ngen::ProductFamily::MTL, ngen::ProductFamily::ARL})) {
                auto xoLoadD = state.ra.alloc_sub<int32_t>();
                mov(1, xoLoadD, xoLoad);
                mov(1, xo, -xoLoadD);
                state.ra.safeRelease(xoLoadD);
            } else
                mov(1, xo, -xoLoad);
            state.ra.safeRelease(xoPtr);
            state.ra.safeRelease(xoLoad);
        }
    };

    if (aoScalarLoad) loadABO(problem.Tao, state.inputs.ao, state.inputs.aoPtr);
    if (boScalarLoad) loadABO(problem.Tbo, state.inputs.bo, state.inputs.boPtr);

    if (problem.postOps.cStochasticRound) {
        state.inputs.sroundSeed = state.ra.alloc_sub(DataType::ud, getHint(HintType::LongTerm, strategy));
        vector<Subregister> srcs;
        srcs.push_back(state.inputs.sroundSeedPtr);
        auto seedLoad = loadScalars(Type::u32, srcs, strategy, state);
        mov(1, state.inputs.sroundSeed, seedLoad);
    }

    // 2D scale address handling.
    if (problem.aScale2D() && state.inputs.offsetAScale.isValid())
        eadd(1, state.inputs.aScalePtr, state.inputs.aScalePtr, state.inputs.offsetAScale, strategy, state);
    if (problem.bScale2D() && state.inputs.offsetBScale.isValid())
        eadd(1, state.inputs.bScalePtr, state.inputs.bScalePtr, state.inputs.offsetBScale, strategy, state);

    state.ra.safeRelease(state.inputs.offsetAScale);
    state.ra.safeRelease(state.inputs.offsetBScale);

    // Sum address handling.
    if (problem.needsSeparateASums() && state.inputs.offsetAg.isValid())
        eadd(1, state.inputs.agPtr, state.inputs.agPtr, state.inputs.offsetAg, strategy, state);
    if (problem.needsSeparateBSums() && state.inputs.offsetBg.isValid())
        eadd(1, state.inputs.bgPtr, state.inputs.bgPtr, state.inputs.offsetBg, strategy, state);

    state.ra.safeRelease(state.inputs.offsetAg);
    state.ra.safeRelease(state.inputs.offsetBg);

    if (problem.aqGroupK == 0) problem.aqGroupK = strategy.slmA ? strategy.unrollKSLM : strategy.ka_load;
    if (problem.bqGroupK == 0) problem.bqGroupK = strategy.slmB ? strategy.unrollKSLM : strategy.kb_load;
    if (problem.aqGroupM == 0) problem.aqGroupM = 1;
    if (problem.bqGroupN == 0) problem.bqGroupN = 1;
    if (problem.cqGroupM == 0) problem.cqGroupM = 1;
    if (problem.cqGroupN == 0) problem.cqGroupN = 1;

    if (problem.hasCMXScale()) {
        // Temporary storage for generated MX scales: the injector writes one
        // dword-strided scale per M group per column before they are stored via C_scaleLayout.
        if (problem.cqGroupM != 32 || problem.cqGroupN != 1) stub("Unexpected MX C scale groups");
        if (strategy.unroll[LoopM] % problem.cqGroupM) stub();
        int n_scales = (strategy.unroll[LoopM] / problem.cqGroupM) * strategy.unroll[LoopN];
        state.tmpCScales = state.ra.alloc_range(GRF::bytesToGRFs(hw, n_scales * 4));
    }

    ngen::Subregister ldbStrided, ldcStrided;
    ngen::Subregister fullN;
    if (strategy.cInterleave) {
        ldbStrided = state.ra.alloc_sub<uint32_t>();
        ldcStrided = state.ra.alloc_sub<uint32_t>();
        mulConstant(1, ldbStrided, state.inputs.ldb, strategy.cInterleaveChunk(problem.Tc_ext));
        mulConstant(1, ldcStrided, state.inputs.ldc[0], strategy.cInterleaveChunk(problem.Tc_ext));

        fullN = state.inputs.n;
        if (strategy.persistentLoop()) {
            fullN = state.ra.alloc_sub<uint32_t>();
            mov(1, fullN, state.inputs.n);
        }
    }

    // Persistent thread preparation and re-entry.
    if (strategy.persistentLoop()) {
        if (!strategy.linearOrder()) stub();
        if (problem.batch != BatchMode::None) stub();       // need to wrangle groupIDK also

        if (state.groupIDMN == state.inputs.groupIDMN) {
            state.groupIDMN = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            mov(1, state.groupIDMN, state.inputs.groupIDMN);
        }

        if (state.effTempC == state.inputs.tempC)
            state.effTempC = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));

        gemmFoldOffsets(problem, strategy, state);

        mark(lReentry);

    }

    // Variable k-slicing logic.
    gemmStreamKSetup(lKVPhaseDone, lKernelDone, problem, strategy, state);

    if (state.groupCountMN.isValid() && state.inputs.groupCountMN.isInvalid())
        state.ra.release(state.groupCountMN);

    if (strategy.kParallel && (strategy.fuseBeta || strategy.fusePostOps)) {
        if (!strategy.linearOrder()) stub();
        gemmFusedBetaPOInit(state.groupIDMN, problem, strategy, state);
    }

    // Group ID remapping.
    gemmReorderGlobalIDs(problem, strategy, state);

    // Batch handling.
    gemmGetBatchIDs(problem, strategy, state);

    // Compute offset for A, B, C for non-strided and strided batch.
    gemmOffsetBatchABC(problem, strategy, state);

    // 32-bit add check. TODO: move out of persistent loop for non-batch.
    gemmCheck32(problem, strategy, state);

    // Calculate i0, j0, h0 -- the initial i/j/h indices for this thread.
    bool needH0 = anyKParallelFixed;

    state.i0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    state.j0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    if (needH0 && state.h0.isInvalid())
        state.h0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));

    bool wgCheck = wgRemCheck(problem, strategy);
    bool gemmtBarriers = problem.gemmt() && strategy.needsBarrier();

    Subregister idM, idN, idK;

    idM = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));
    idN = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
    if (strategy.kParallel)
        idK = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));

    if (strategy.cInterleave) {
        divDown(idN, state.inputs.groupIDN, strategy.cInterleaveChunk(problem.Tc_ext), strategy, state);
    }

    auto &srcIDN = strategy.cInterleave ? idN : state.inputs.groupIDN;
    if (strategy.fixedWG(problem)) {
        mulConstant(1, idM, state.inputs.groupIDM, strategy.wg[LoopM]);
        mulConstant(1, idN, srcIDN, strategy.wg[LoopN]);
    } else {
        mul(1, idM, state.inputs.groupIDM, state.inputs.localSizeM.uw());
        mul(1, idN, srcIDN, state.inputs.localSizeN.uw());
    }

    if (strategy.kParallel) {
        mul(1, idK, state.inputs.groupIDK, state.inputs.localSizeK.uw());
        if (strategy.kPadding) {
            add(1 | lt | state.flagAP, idK.d(), idK, -state.inputs.localSizeK);
            mov(1 | state.flagAP, state.inputs.k0, 0);
        }
        if ((strategy.fuseBeta || strategy.fusePostOps) && strategy.kParallelLocal) {
            state.wgK = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            auto temp = state.ra.alloc_sub<uint32_t>();
            emul(1, state.wgK, idK, state.inputs.k0, strategy, state);
            mul(1, temp, state.inputs.k0, state.inputs.localSizeK.uw());
            add(1 | sat, state.wgK, state.inputs.k, -state.wgK);
            min_(1, state.wgK, state.wgK, temp);
            state.ra.safeRelease(temp);
        }
    }

    Subregister wgChunkRem;
    if (strategy.cInterleave) {
        wgChunkRem = state.ra.alloc_sub<uint32_t>();
        mod(wgChunkRem, state.inputs.groupIDN, strategy.cInterleaveChunk(problem.Tc_ext), strategy, state);
    }

    if (wgCheck || gemmtBarriers) {
        state.wgI0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp0, strategy));
        state.wgJ0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::TempComp1, strategy));

        mulConstant(1, state.wgI0, idM, strategy.unroll[LoopM]);
        mulConstant(1, state.wgJ0, idN, strategy.unroll[LoopN] * strategy.cInterleaveChunk(problem.Tc_ext));

        if (strategy.cInterleave) {
            add(1, state.wgJ0, state.wgJ0, wgChunkRem);
        }
    }

    add(1, idM, idM, state.lidM);
    add(1, idN, idN, state.lidN);
    if (strategy.kParallel && !strategy.kInterleave)
        add(1 | gt | state.flagAP, idK, idK, state.lidK);

    mulConstant(1, state.i0, idM, strategy.unroll[LoopM]);
    mulConstant(1, state.j0, idN, strategy.unroll[LoopN] * strategy.cInterleaveChunk(problem.Tc_ext));

    if (strategy.cInterleave) {
        add(1, state.j0, state.j0, wgChunkRem);
        state.ra.safeRelease(wgChunkRem);
    }

    if (strategy.kParallelVariable)
        noop(); /* h0 already calculated */
    else if (strategy.kParallelLocal && strategy.kInterleave) {
        mulConstant(1, state.h0, state.lidK, strategy.kInterleaveChunk);
        if (strategy.kParallel)
            emad(1, state.h0, state.h0, idK, state.inputs.k0, strategy, state);
    } else if (strategy.kParallel)
        emul(1, state.h0, idK, state.inputs.k0, strategy, state);
    else if (strategy.kParallelLocal)
        mul(1, state.h0, state.inputs.k0, state.lidK);

    if (strategy.kParallel) {
        if (strategy.kInterleave)
            add(1 | gt | state.flagAP, idK, idK, state.lidK);
        if (state.inputs.flags.isValid() && problem.cOffset == COffset::Pre)
            or_(1 | state.flagAP, state.inputs.flags, state.inputs.flags, FlagNonfinalKBlock);
    }

    gemmReverseLoops(problem, strategy, state);

    state.ra.safeRelease(idM);
    state.ra.safeRelease(idN);
    state.ra.safeRelease(idK);
    if (!strategy.persistentLoop()) {
        state.ra.safeRelease(state.inputs.localSizeM);
        state.ra.safeRelease(state.inputs.localSizeN);
    }
    if (anyKParallelFixed) {
        state.ra.safeRelease(state.inputs.localIDK);
        if (!strategy.persistentLoop())
            state.ra.safeRelease(state.inputs.localSizeK);
    }
    if (strategy.linearOrder() || strategy.persistentLoop()) {
        state.ra.safeRelease(state.inputs.groupIDM);
        state.ra.safeRelease(state.inputs.groupIDN);
        state.ra.claim(state.nextGroupIDM);
        state.ra.claim(state.nextGroupIDN);
    }

    moveR0(strategy, state);

    // Adjust k range for local/global k-reduction.
    if (anyKParallelFixed && !strategy.kParallelVariable) {
        add(1 | sat, state.inputs.k.ud(), strategy.persistentLoop() ? state.fullK : state.inputs.k, -state.h0);

        if (strategy.kInterleave) {
            // k <- floor(k / (chunk * k local size)) * chunk + min(k % (chunk * k local size), chunk)
            auto chunk = strategy.kInterleaveChunk;
            auto wgChunk = chunk * strategy.wg[LoopK];
            auto temp1 = state.ra.alloc_sub<uint32_t>();
            auto temp2 = state.ra.alloc_sub<uint32_t>();
            divDown(temp1, state.inputs.k.ud(), wgChunk, strategy, state);
            emad(1, temp2, state.inputs.k.ud(), -temp1, wgChunk, strategy, state);
            emul(1, temp1, temp1, chunk, strategy, state);
            min_(1, temp2, temp2, chunk);
            add(1, state.inputs.k.ud(), temp1, temp2);

            state.ra.safeRelease(temp1);
            state.ra.safeRelease(temp2);
        }

        if (strategy.kParallel && !strategy.kParallelLocal && (strategy.fuseBeta || strategy.fusePostOps)) {
            state.wgK = state.ra.allocSub<uint32_t>(getHint(HintType::LongTerm));
            min_(1, state.wgK, state.inputs.k, state.inputs.k0);
        }
        min_(1, state.inputs.k, state.inputs.k, state.inputs.k0);

        bool keepK0 = false;
        keepK0 |= strategy.kParallelLocal && (strategy.barrierFreq > 0 ||
                std::max(strategy.slmBuffersA, strategy.slmBuffersB) > 0);
        keepK0 |= strategy.persistentLoop();

        if (keepK0)
            state.threadK0 = state.inputs.k0;
        else
            state.ra.safeRelease(state.inputs.k0);
    }

    state.ra.safeRelease(state.inputs.localIDM);
    state.ra.safeRelease(state.inputs.localIDN);
    if (!strategy.needsMNLocalIDs())
        state.lidM = state.lidN = invalid;

    if (strategy.cInterleave) {
        // Idea: The B and C matrices are reinterpreted as being made of X strided submatrices (X = cInterleaveChunk), with the columns interleaved.
        //    Due to interleaving, the leading dimension of each of these matrices is X * ldc. When each submatrix
        //    has an ld that's cache-line-aligned, we can shift these matrices along the rows to a cache line boundary
        //    and write to C without cache line contention. Implementation is as follows:
        // Important notes:
        // 1. Work groups are divided among the submatrices, so #work groups in a row must be padded to a multiple of X.
        // 2. Due to i0 shifting, in unaligned cases we need additional rows of work groups to handle extra remainders
        // 3. Only works for TNN layout, due to assumptions about leading dimensions
        if (problem.A.layout != MatrixLayout::T) stub();
        if (problem.B.layout != MatrixLayout::N) stub();
        if (problem.C.layout != MatrixLayout::N) stub();

        auto shiftJ0 = state.wgJ0.isValid() ? state.wgJ0 : state.j0;
        add(1, state.inputs.n, -shiftJ0, fullN);
        divUp(state.inputs.n, state.inputs.n, strategy.cInterleaveChunk(problem.Tc_ext), strategy, state);

        emad(1, state.offsetB, state.offsetB, shiftJ0, state.inputs.ldb, strategy, state);
        emad(1, state.offsetC[0], state.offsetC[0], shiftJ0, state.inputs.ldc[0], strategy, state);
        if (state.wgJ0.isValid()) {
            add(1, state.j0, state.j0, -shiftJ0);
            divUp(state.j0, state.j0, strategy.cInterleaveChunk(problem.Tc_ext), strategy, state);
        }
        if (problem.hasBinaryPostOp() && strategy.cInterleave) {
            state.ctiShiftJ0 = state.ra.alloc_sub<int32_t>();
            mov(1, state.ctiShiftJ0, shiftJ0);
        }
        mov(1, shiftJ0, 0);

        if (!strategy.persistentLoop()) {
            state.ra.safeRelease(state.inputs.ldb);
            state.ra.safeRelease(state.inputs.ldc[0]);
        }
        state.inputs.ldb = ldbStrided;
        state.inputs.ldc[0] = ldcStrided;

        // Shift back i0 to align with a cache line
        Subregister cBasePtr = state.ra.alloc_sub(state.inputs.C[0].getType());
        add(1, cBasePtr, state.inputs.C[0], state.offsetC[0]);
        auto shiftI0 = state.ra.alloc_sub<uint32_t>();
        mod(shiftI0, cBasePtr.ud(0), 64, strategy, state);
        divDown(shiftI0, shiftI0, problem.Tc_ext.size(), strategy, state);

        state.unclampedI0 = state.ra.alloc_sub<int32_t>();
        add(1, state.unclampedI0, state.i0, -shiftI0);
        mov(1 | sat, state.i0, state.unclampedI0);
        if (wgCheck || gemmtBarriers) {
            add(1 | sat, state.wgI0, state.wgI0, -shiftI0);
        }
        state.ra.safeRelease(shiftI0);
    }

    // Calculate workgroup remainders if needed.
    gemmCalcWGRemainders(problem, strategy, state);

    // Compute base addresses for A, B, C.
    auto &i0p = (strategy.coopA == CoopSplit::FullK) ? state.wgI0 : state.i0;
    auto &j0p = (strategy.coopB == CoopSplit::FullK) ? state.wgJ0 : state.j0;
    gemmOffsetABC(true, state.i0, state.j0, state.h0, i0p, j0p, problem, strategy, state);
    if (!(strategy.prefetchA && strategy.A_prefetch.address2D)) state.ra.safeRelease(state.wgI0);
    if (!(strategy.prefetchB && strategy.B_prefetch.address2D)) state.ra.safeRelease(state.wgJ0);

    // Fixed systolic kernels don't support checking for out-of-bounds panels.
    // Instead, move A/B pointers for out-of-bounds panels back in bounds.
    if (strategy.panelCheck && strategy.fixedSystolic) {
        bool checkA = strategy.A.base.isStateless();
        bool checkB = strategy.B.base.isStateless();
        if (checkA) cmp(2 | ge | f0[1], state.i0, state.inputs.m);
        if (checkB) cmp(2 | ge | f1[1], state.j0, state.inputs.n);
        if (checkA) emov(1 | f0[1], state.offsetA, 0, strategy, state);
        if (checkB) emov(1 | f1[1], state.offsetB, 0, strategy, state);
    }

    // Compute addresses for separate A/B row/column sums.
    if (problem.bOffset == ABOffset::Load && !isPacked(problem.A.layout)) {
        state.effAs = state.ra.allocSub(state.inputs.agPtr.getType());
        emad(1, state.effAs, state.inputs.agPtr, state.i0, Tc.size(), strategy, state);
    }
    if (problem.aOffset == ABOffset::Load && !isPacked(problem.B.layout)) {
        state.effBs = state.ra.allocSub(state.inputs.bgPtr.getType());
        emad(1, state.effBs, state.inputs.bgPtr, state.j0, Tc.size(), strategy, state);
    }

    gemmSetupABC(problem, strategy, state);
    gemmSubkernel(problem, strategy, state);

    mark(lKernelDone);

    // Persistent thread loop. Advance group ID and re-enter kernel if there's more work to do.
    if (strategy.persistentLoop()) {
        status << "Persistent loop" << status_stream::endl;

        GRF temp;
        auto doBarrier = [&](const GRF &r0_info) {      /* GCC nested lambda bug */
            MOCK_BARRIERS activeThreadBarrier(temp, r0_info, strategy);
        };

        auto persistentRestore = [&] {
            gemmRestoreOffsets(problem, strategy, state);

            if (std::max(strategy.slmBuffersA, strategy.slmBuffersB) > 0) {
                temp = state.ra.alloc();
                useR0(state, doBarrier);
                state.ra.safeRelease(temp);
            }
        };

        // Recompute group count if needed.
        if (state.inputs.groupCountMN.isInvalid()) {
            state.ra.claim(state.groupCountMN);
            emul(1, state.groupCountMN, state.inputs.groupCountM, state.inputs.groupCountN, strategy, state);
        }

        // Clear flags as needed.
        uint32_t flagsToClear = 0;

        if (strategy.fuseBeta)
            flagsToClear |= FlagDidBeta | FlagSkipBetaCheck | FlagKPartitioned;
        if (strategy.fusePostOps)
            flagsToClear |= FlagKPartitioned;
        if (strategy.kParallelVariable && problem.cOffset == COffset::Pre)
            flagsToClear |= FlagNonfinalKBlock;
        if (problem.sumA || problem.sumB)
            flagsToClear |= FlagStoreSums;

        if (flagsToClear)
            and_(1, state.inputs.flags, state.inputs.flags, ~uint32_t(flagsToClear));

        if (state.movedR0 && state.r0_info != r0.ud())
            mov<uint32_t>(r0DWords(hw), r0, state.r0_info);

        if (strategy.kParallelVariable) {
            Label lNotKSliced;
            and_(1 | ze | f0[0], null.ud(), state.inputs.flags, FlagKSlicing);
            if (strategy.persistent)
                and_(1 | nz | f0[1], null.ud(), state.inputs.flags, FlagKSlice2);
            cmp(1 | gt | f1[0], state.k0Rem, 0);

            jmpi(1 | f0[0], lNotKSliced);

            // Continue work on this slice, if any remaining.
            persistentRestore();
            jmpi(1 | f1[0], lReentry);

            // Otherwise, advance to next slice, if any.
            if (strategy.persistent) {
                Label lCheckSlice2;

                mark(lCheckSlice2);
                jmpi(1 | ~f0[1], lLeavePersistentLoop);
                gemmStreamKPrepareSlice2(problem, strategy, state);
                jmpi(1, lReentry);

                mark(lKVPhaseDone);
                and_(1 | nz | f0[1], null.ud(), state.inputs.flags, FlagKSlice2);
                jmpi(1, lCheckSlice2);
            } else
                mark(lKVPhaseDone);
            mark(lNotKSliced);
        }

        if (strategy.persistent) {
            add(1, state.groupIDMN, state.groupIDMN, state.inputs.groupStride);
            if (strategy.kParallelVariable)
                cmp(1 | gt | f1[0], state.inputs.kSlicedTiles, 0);
            cmp(1 | lt | f0[0], state.groupIDMN, state.groupCountMN);

            persistentRestore();
            if (strategy.kParallelVariable)
                jmpi(1 | f1[0], lReentry);
            jmpi(1 | f0[0], lReentry);
        }

        if (strategy.kParallelVariable)
            mark(lLeavePersistentLoop);

        state.ra.safeRelease(state.inputs.groupCountMN);
        state.ra.safeRelease(state.groupCountMN);
    }

    if (strategy.cInterleave) {
        state.ra.safeRelease(ldcStrided);
    }

    mark(lPadThread);

    if (!inFusedGEMM) {
        epilogue(strategy, state);
        padding();
    }
}

template <HW hw>
void Generator<hw>::gemmSubkernel(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState state)
{
    Label labelSubkernelDone, labelSubkernelEarlyExit;

    status << "Begin subkernel: unroll " << strategy.unroll[LoopM] << 'x' << strategy.unroll[LoopN] << status_stream::endl;

    // Calculate remainders for m/n loops: clamp(m - i0, 0, unrollM), clamp(n - j0, 0, unrollN).
    // Careful with this clamping, because unroll may change in remainder handling.
    bool remM = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remN = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
    bool fusedremM = remM && strategy.fused && (strategy.fusedLoop == LoopM);
    bool fusedremN = remN && strategy.fused && (strategy.fusedLoop == LoopN);

    state.doLateExit = strategy.lateExit();
    bool earlyExit = !state.doLateExit;

    if (fusedremM || fusedremN) {
        state.remFusedStorage = state.ra.alloc_sub<uint32_t>();
        add(1, state.remFusedStorage, -state.fusedID, uint16_t(strategy.unroll[strategy.fusedLoop]));
    }
    if (remM || !earlyExit) {
        state.remaindersFused[LoopM] = state.remainders[LoopM] = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        InstructionModifier mod = 1 | sat;
        if (!fusedremM && earlyExit && !strategy.cInterleave)
            mod = mod | le | f0[1];
        add(mod, state.remainders[LoopM], -state.i0, state.inputs.m);
    }
    if (remN || !earlyExit) {
        state.remaindersFused[LoopN] = state.remainders[LoopN] = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        InstructionModifier mod = 1 | sat;
        if (!fusedremN && earlyExit)
            mod = mod | le | f1[1];
        add(mod, state.remainders[LoopN], -state.j0, state.inputs.n);
    }
    if (fusedremM || fusedremN) {
        state.remaindersFused[strategy.fusedLoop] = state.remFusedStorage;
        add(1 | sat, state.remFusedStorage, -state.remFusedStorage, state.remainders[strategy.fusedLoop]);
        if (earlyExit) {
            cmp(1 | le | (fusedremM ? f0[1] : f1[1]), null.d(), state.remainders[strategy.fusedLoop].d(), -state.fusedID);
            state.allowEmptyC = true;
        }
    }
    if (remM) {
        if (strategy.cInterleave) {
            // if state.unclampedI0 < 0: Use unrollM + state.unclampedI0 for upper bound
            FlagRegister flag = state.ra.alloc_flag();
            Subregister upper = state.ra.alloc_sub<uint32_t>();
            cmp(1 | lt | flag, state.unclampedI0, 0);
            mov(1, upper, strategy.unroll[LoopM]);
            add(1 | sat | flag, upper, state.unclampedI0, strategy.unroll[LoopM]);
            emin_(1, state.remainders[LoopM], state.remainders[LoopM], upper, state);
            cmp(1 | le | f0[1], state.remainders[LoopM], 0);
            state.ra.safeRelease(flag);
            state.ra.safeRelease(state.unclampedI0);
        } else {
            min_(1, state.remainders[LoopM], state.remainders[LoopM], uint16_t(strategy.unroll[LoopM]));
        }
    }
    if (remN) {
        min_(1, state.remainders[LoopN], state.remainders[LoopN], uint16_t(strategy.unroll[LoopN]));
    }

    gemmCalcIncrements(problem, strategy, state);

    // Early exit if nothing to do. Keep fused threads together.
    if (earlyExit && (remM || remN)) {
        InstructionModifier cond;
        if (remM && remN)
            cond = 1 | f0[1] | anyv;
        else if (remM)
            cond = 1 | f0[1];
        else
            cond = 1 | f1[1];

        ejmpi(cond, labelSubkernelDone);
    }

    // Create the kernel body. If enabled, create two versions, one with A/B more aligned.
    bool success;
    int optAlignA = strategy.optAlignAB;
    int optAlignB = strategy.optAlignAB;

    // Handle block 2D alignment checks.
    if (strategy.optAlignAB2D) {
        optAlignA = std::max({
            optAlignA,
            block2DMinAlignment(hw, problem.A, strategy.A),
            block2DMinAlignment(hw, problem.A, strategy.A_prefetch)
        });
        optAlignB = std::max({
            optAlignB,
            block2DMinAlignment(hw, problem.B, strategy.B),
            block2DMinAlignment(hw, problem.B, strategy.B_prefetch)
        });
    }

    // If it's not possible to force the higher alignment, force the unaligned path.
    bool forceDowngrade = false;
    if (optAlignA && problem.A.layout == MatrixLayout::N)
        forceDowngrade |= ((strategy.unroll[LoopM] * problem.Ta) % optAlignA) != 0;
    if (optAlignB && problem.B.layout == MatrixLayout::T)
        forceDowngrade |= ((strategy.unroll[LoopN] * problem.Tb) % optAlignB) != 0;

    if (forceDowngrade) {
        optAlignA = optAlignB = 0;
        gemmDowngradeAccess(problem, strategy, state);
    }

    uint16_t maskA = (optAlignA - 1);
    uint16_t maskB = (optAlignB - 1);
    bool doA = (optAlignA && (problem.A.alignment & maskA));
    bool doB = (optAlignB && (problem.B.alignment & maskB));

    if (!doA && !doB)
        success = gemmMEdge(problem, strategy, state);
    else {
        // Check alignment of effA, effB, lda, and ldb.
        Label labelUnaligned;
        bool checkLDA = !isPacked(problem.A.layout);
        bool checkLDB = !isPacked(problem.B.layout);
        if (doA) {
            and_(1 | nz | f0[0], null.uw(), state.effA.uw(), maskA);
            if (checkLDA) and_(1 | nz | f1[0], null.uw(), state.inputs.lda.uw(), maskA);
        }
        if (doB) {
            and_(1 | nz | f0[1], null.uw(), state.effB.uw(), maskB);
            if (checkLDB) and_(1 | nz | f1[1], null.uw(), state.inputs.ldb.uw(), maskB);
        }
        if (doA) {
            InstructionModifier amod = checkLDA ? 1 | f0[0] | anyv : 1 | f0[0];
            ejmpi(amod, labelUnaligned);
        }
        if (doB) {
            InstructionModifier bmod = checkLDB ? 1 | f0[1] | anyv : 1 | f0[1];
            ejmpi(bmod, labelUnaligned);
        }
        if (strategy.optAlignAB2D) {
            if (doA) and_(1 | nz | f0[0], null.ud(), state.inputs.lda, 0xFF000000);
            if (doB) and_(1 | nz | f1[0], null.ud(), state.inputs.ldb, 0xFF000000);
            if (doA) jmpi(1 | f0[0], labelUnaligned);
            if (doB) jmpi(1 | f1[0], labelUnaligned);
        }

        auto alignedProblem = problem;
        if (doA) alignedProblem.A.setAlignment(std::max<int>(problem.A.alignment, optAlignA));
        if (doB) alignedProblem.B.setAlignment(std::max<int>(problem.B.alignment, optAlignB));

        status << "Aligned A/B" << status_stream::endl;
        success = gemmMEdge(alignedProblem, strategy, state);

        if (!success && lastException) std::rethrow_exception(lastException);

        state.isNested ? jmpi(1, labelSubkernelDone) : epilogue(strategy, state);

        mark(labelUnaligned);

        auto modStrategy = strategy;

        gemmDowngradeAccess(problem, modStrategy, state);

        status << "Unaligned A/B" << status_stream::endl;
        if (!gemmMEdge(problem, modStrategy, state)) {
            modStrategy.checkAdd32 = false;                     // Don't optimize additions on this (slow) path to reduce code size.
            status << "Reducing register usage" << status_stream::endl;
            success = success && modStrategy.minimize(hw, problem);

            gemmCalcIncrements(problem, modStrategy, state);    // Recalculate ld increments as they may have changed.

            success = success && gemmMEdge(problem, modStrategy, state);
        }
    }

    if (!success) lastException ? std::rethrow_exception(lastException)
                                : stub("Could not generate kernel.");

    mark(labelSubkernelDone);

    gemmFreeIncrements(problem, strategy, state);
}

// Handle outer-level m edge cases.
template <HW hw>
bool Generator<hw>::gemmMEdge(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    if (strategy.jointSplit && strategy.remHandling[LoopM] == RemainderHandling::Split && strategy.remHandling[LoopN] == RemainderHandling::Split)
        return mnJointSplitRemainderHandling(problem, strategy, state, &Generator<hw>::gemmBody);
    else
        return mnRemainderHandling(LoopM, problem, strategy, state, &Generator<hw>::gemmNEdge);
}

// Handle outer-level n edge cases.
template <HW hw>
bool Generator<hw>::gemmNEdge(GEMMProblem problem, GEMMStrategy strategy, GEMMState state)
{
    return mnRemainderHandling(LoopN, problem, strategy, state, &Generator<hw>::gemmBody);
}

// Entrypoint for generating the GEMM kernel body, returning true if successful.
template <HW hw>
bool Generator<hw>::gemmBody(GEMMProblem problem, GEMMStrategy strategy, GEMMState state)
{
    // Record whether we are in the first row/column for fused sum kernels.
    if (problem.sumA || problem.sumB) {
        if (problem.sumA && problem.sumB) stub();
        auto &y0 = problem.sumA ? state.j0 : state.i0;
        cmp(1 | eq | state.flagAP, y0, 0);
        or_(1 | state.flagAP, state.inputs.flags, state.inputs.flags, FlagStoreSums);
    }

    // Out-of-bounds panel checks.
    bool panelCheck = strategy.panelCheck && strategy.lateExit() && !strategy.fixedSystolic;
    if (panelCheck && isPacked(problem.A.layout) && strategy.remHandling[LoopM] != RemainderHandling::Ignore) {
        state.panelMaskA = state.raVFlag.alloc();
        cmp(16 | gt | state.panelMaskA, state.remainders[LoopM], 0);
    }

    if (panelCheck && isPacked(problem.B.layout) && strategy.remHandling[LoopN] != RemainderHandling::Ignore) {
        state.panelMaskB = state.raVFlag.alloc();
        cmp(16 | gt | state.panelMaskB, state.remainders[LoopN], 0);
    }

    // Release variables that are no longer needed.
    bool saveIJ0 = keepIJ0(problem, strategy), saveH0 = keepH0(problem, strategy);
    bool a2D = strategy.A.address2D || (strategy.prefetchA && strategy.A_prefetch.address2D);
    bool b2D = strategy.B.address2D || (strategy.prefetchB && strategy.B_prefetch.address2D);
    bool c2D = strategy.C.address2D || (strategy.prefetchC && strategy.C_prefetch.address2D);

    if (!a2D && !c2D && !saveIJ0) state.ra.safeRelease(state.i0);
    if (!b2D && !c2D && !saveIJ0) state.ra.safeRelease(state.j0);
    if (!a2D && !b2D && !saveH0)  state.ra.safeRelease(state.h0);
    if (!strategy.altCRemainder && !strategy.block2DCRemainder)
        releaseFusedRemainders(state);
    if (strategy.coopA != CoopSplit::FullK)
        state.ra.safeRelease(state.remaindersWG[LoopM]);
    if (strategy.coopB != CoopSplit::FullK)
        state.ra.safeRelease(state.remaindersWG[LoopN]);

    // If A/B are masked, check if we need to change ka_load/kb_load. If so, recalculate ld increments.
    if (gemmPrepMaskedAB(problem, strategy, state))
        gemmCalcIncrements(problem, strategy, state);

    // Disable C prefetch in remainder handling if it needs masks/fragmenting.
    if (strategy.remHandling[LoopM] != RemainderHandling::Ignore || strategy.remHandling[LoopN] != RemainderHandling::Ignore) {
        if (strategy.C.base.isStateless() && !strategy.C.padded && strategy.prefetchC && !isBlock2D(strategy.C_prefetch.accessType)) {
            status << "Auto-disabling C prefetch in masked region" << status_stream::endl;
            strategy.prefetchC = 0;
            if (state.effCp != state.effC[0]) state.ra.safeRelease(state.effCp);
        }
    }

    // Try generating kernel body with current strategy.
    bool success = false;
    pushStream();
    try {
        success = gemmBodyInternal(problem, strategy, state);
    } catch (...) {
        lastException = std::current_exception();
    }
    success ? appendCurrentStream() : discardStream();

    return success;
}

// Generate the GEMM kernel body, returning true if successful.
template <HW hw>
bool Generator<hw>::gemmBodyInternal(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc = problem.Tc;

    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto &remM = state.remainders[LoopM];
    auto &remN = state.remainders[LoopN];

    // Accumulate C with panel*panel multiply.
    if (!gemmAccumulateC(problem, strategy, state)) return false;

    // Final C horizontal reduction for dot product kernels.
    gemmDotReduce(problem, strategy, state);

    // Add A/B offsets.
    gemmLoadABOffset(problem, strategy, state);
    if (!gemmFinalizeSums(problem, strategy, state)) return false;
    gemmApplyABOffset(problem, strategy, state);

    // If C is packed, update remainders and prepare to mask out border regions.
    bool remaskC_M = isPacked(problem.C.layout) && (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remaskC_N = isPacked(problem.C.layout) && (strategy.remHandling[LoopN] != RemainderHandling::Ignore);

    if (remaskC_M || remaskC_N) {
        if (remaskC_M) setupTeardownRemask(Tc, 0, true, unrollM, remM, strategy, state);
        if (remaskC_N) setupTeardownRemask(Tc, 1, true, unrollN, remN, strategy, state);

        int C_mgran, C_ngran;
        getGranularities(problem.C, C_mgran, C_ngran);
        if (!remaskC_M || C_mgran == unrollM) C_mgran = 1;
        if (!remaskC_N || C_ngran == unrollN) C_ngran = 1;
        if (!is_zero_or_pow2(C_mgran)) stub();
        if (!is_zero_or_pow2(C_ngran)) stub();

        if (C_mgran > 1) add(1, remM, remM, C_mgran - 1);
        if (C_ngran > 1) add(1, remN, remN, C_ngran - 1);
        if (C_mgran > 1) and_(1, remM, remM, uint32_t(~(C_mgran - 1)));
        if (C_ngran > 1) and_(1, remN, remN, uint32_t(~(C_ngran - 1)));
    }

    // Local k reduction.
    if (strategy.kParallelLocal)
        gemmKReduce(problem, strategy, state);

    // Wait for fused beta completion (involves a barrier) before late exit.
    if (strategy.fuseBeta)
        gemmFusedBetaWaitCompletion(problem, strategy, state);

    // Late exit.
    Label labelLateExit;
    if (state.doLateExit && !strategy.fusePostOps && !(strategy.registerOutput() && outputCRange.empty()))
        gemmOOBExit(labelLateExit, strategy, state);

    // Handle fused post-ops for atomic update kernels.
    if (strategy.fusePostOps) {
        if (!gemmFusedPostOpsFinalize(labelLateExit, problem, strategy, state)) return false;
    }

    if (strategy.registerOutput()) {
        // Marshal C into output registers. The main path defines the output registers.
        if (outputCRange.empty()) {
            outputCRange = state.C_regs[0];
            outputCLayout = state.C_layout;
        } else {
            // FIXME: check that layouts are compatible, and rearrange if not.
            overlappedCopy(state.C_regs[0], outputCRange, state);
        }
    } else {
        // Regular C update into memory.
        if (!gemmUpdateCDispatch(problem, strategy, state)) return false;
    }

    // Cleanup.
    if (remaskC_M) setupTeardownRemask(Tc, 0, false, unrollM, remM, strategy, state);
    if (remaskC_N) setupTeardownRemask(Tc, 1, false, unrollN, remN, strategy, state);

    if (state.doLateExit || strategy.fusePostOps) {
        mark(labelLateExit);
        if (strategy.fused)
            join(16);
    }

    if (strategy.altFusedBeta && !strategy.fusePostOps)
        gemmFusedBetaNotifyCompletion(problem, strategy, state);

    return true;
}

// Perform the body of the GEMM computation, updating a block of C.
template <HW hw>
bool Generator<hw>::gemmAccumulateC(GEMMProblem &problem_, GEMMStrategy &strategy_, GEMMState &state)
{
    if (strategy_.fixedSystolic) {
        if (problem_.sumA || problem_.sumB) stub();
        if (problem_.aOffset == ABOffset::Calc || problem_.bOffset == ABOffset::Calc) stub();

        return strategy_.splitCopy ? sysgemm2AccumulateC(problem_, strategy_, state)
                                   : sysgemmAccumulateC(problem_, strategy_, state);
    }

    auto problem = problem_;
    auto strategy = strategy_;

    if (!gemmAccumulateCSetup(problem, strategy, state))
        return false;

    // Synthesize k loop. If configured, choose between 32-bit adds and 64-bit adds.
    if (strategy.checkAdd32 && state.add64.isValid()) {
        Label loop64, done;
        bool success = true;

        cmp(1 | ne | state.flagAP, state.add64, uint16_t(0));
        jmpi(1 | state.flagAP, loop64);
        state.ra.safeRelease(state.add64);

        status << "k loop: 32-bit address update" << status_stream::endl;
        strategy.emulate.emulate64_add32 = true;
        auto substate32 = state;
        success &= gemmKLoop(problem, strategy, substate32);
        jmpi(1, done);

        mark(loop64);
        status << "k loop: 64-bit address update" << status_stream::endl;
        strategy.emulate.emulate64_add32 = false;
        success &= gemmKLoop(problem, strategy, state);

        mark(done);
        if (!success) return false;
    } else {
        state.ra.safeRelease(state.add64);
        if (!gemmKLoop(problem, strategy, state))
            return false;
    }

    gemmAccumulateCTeardown(problem, strategy, state);

    return true;
}

template <HW hw>
bool Generator<hw>::gemmKLoop(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    return kLoopSingle(KLoop::GEMM, problem, strategy, state);
}


/**********************/
/* Remainder Handling */
/**********************/

// Synthesize a jump conditional on whether this C tile is completely outside the C matrix.
template <HW hw>
void Generator<hw>::gemmOOBExit(Label &target, const GEMMStrategy &strategy, GEMMState &state)
{
    int simt = strategy.fused ? 16 : 1;

    cmp(simt | le | f0[0], state.remainders[LoopM], uint16_t(0));
    cmp(simt | le | f1[0], state.remainders[LoopN], uint16_t(0));

    InstructionModifier cond = simt | f0[0] | anyv;

    strategy.fused ? goto12(cond, target)
                   :  ejmpi(cond, target);
}

// Check whether all threads in a thread group should stay together in m/n remainder handling.
template <HW hw>
bool Generator<hw>::wgRemCheck(const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    bool mnSplitNeedsWGRem
            = (strategy.slmA && (effCoopSplitA(problem, strategy) == CoopSplit::MN) && (strategy.remHandling[LoopM] != RemainderHandling::Ignore) && !strategy.A.padded)
           || (strategy.slmB && (effCoopSplitB(problem, strategy) == CoopSplit::MN) && (strategy.remHandling[LoopN] != RemainderHandling::Ignore) && !strategy.B.padded);

    bool splitBranchNeedsUniformRem
            = (strategy.slmA && (strategy.remHandling[LoopM] == RemainderHandling::Split) && (strategy.wg[LoopM] > 1) && !strategy.A.padded)
           || (strategy.slmB && (strategy.remHandling[LoopN] == RemainderHandling::Split) && (strategy.wg[LoopN] > 1) && !strategy.B.padded);

    return mnSplitNeedsWGRem
        || splitBranchNeedsUniformRem
        || strategy.kParallelLocal
        || ((strategy.barrierFreq > 0 || strategy.cooperativePF) && (strategy.prefetchA || strategy.prefetchB || strategy.prefetchC))
        || (strategy.coopA == CoopSplit::FullK)
        || (strategy.coopB == CoopSplit::FullK);
}

// Do outer-level m/n remainder handling.
template <HW hw>
template <typename Problem>
bool Generator<hw>::mnRemainderHandling(LoopType loop, Problem &problem, GEMMStrategy &strategy, GEMMState &state,
                                        bool (Generator<hw>::*func)(Problem, GEMMStrategy, GEMMState))
{
    auto method = strategy.remHandling[loop];
    auto &unroll = strategy.unroll[loop];
    auto mn = (loop == LoopM) ? state.inputs.m : state.inputs.n;
    auto splitThresh = (loop == LoopM) ? strategy.mSplitThresh : strategy.nSplitThresh;

    Label label_done;

    auto originalCheckAdd32 = strategy.checkAdd32;

    if (method == RemainderHandling::Split) {
        Label label_remainder;

        // Jump to remainder loop if needed.
        // If threads fused in this direction, factor fused ID into calculation.
        if (wgRemCheck(problem, strategy)) {
            cmp(1 | lt | f0[0], null.d(), state.remaindersWG[loop], uint16_t(unroll * strategy.wg[loop]));
        } else {
            cmp(1 | lt | f0[0], null.d(), state.remaindersFused[loop], uint16_t(unroll));
        }

        if (splitThresh) {
            cmp(1 | lt | f1[0], null.d(), mn, int32_t(splitThresh));
            ejmpi(1 | f0[0] | anyv, label_remainder);
        } else
            jmpi(1 | f0[0], label_remainder);

        // First generate code that ignores remainder handling.
        GEMMStrategy substrategy = strategy;
        substrategy.remHandling[loop] = RemainderHandling::Ignore;

        status << "Generating " << "MNK"[static_cast<int>(loop)] << " non-remainder kernel for unroll " << unroll << '.' << status_stream::endl;
        if (!(this->*func)(problem, substrategy, state)) {
            status << "Non-remainder kernel failed, aborting." << status_stream::endl;
            return false;
        }

        // Return, unless this is part of a larger computation, in which case jump to end.
        if (state.isNested)
            jmpi(1, label_done);
        else
            epilogue(strategy, state);

        mark(label_remainder);

        strategy.checkAdd32 = strategy.checkAdd32Rem();
    }

    // OK, great! Now try to create remainder-handling code.
    status << "Attempting to generate " << "MNK"[static_cast<int>(loop)] << " general kernel for unroll " << unroll << '.' << status_stream::endl;
    bool success = (this->*func)(problem, strategy, state);

    strategy.checkAdd32 = originalCheckAdd32;
    if (success) {
        mark(label_done);
        return true;
    }

#ifndef ALLOW_REMAINDERS
    // Disable remainder code for now.
    return false;
#else
    auto &bound  = (loop == LoopN) ? state.inputs.n : state.inputs.m;
    auto &index  = (loop == LoopN) ? state.j0       : state.i0;
    auto &remainders = state.remainders[loop];

    if (method == RemainderHandling::Ignore)
        stub("Could not generate kernel.");

    // It failed, so break up the loop into the next smaller power of 2 along this dimension,
    //  plus the remainder (recursively).
    Label label_next_rem;

    if (unroll == 1) {
        // No more splitting to do.
        // We don't know if this was originally split, so just output a warning.
        status << "NOTE: Split remainder handling is required for loop " << "MNK"[static_cast<int>(loop)] << '.' << status_stream::endl;
        return true;
    }
    int chunkSize = rounddown_pow2(unroll - 1);

    // Jump to next remainder loop if needed.
    pushStream(); {
        cmp(1 | lt | state.flagAP, null.d(), remainders, chunkSize);
        jmpi(1 | state.flagAP, label_next_rem);

        {
            GEMMStrategy substrategy = strategy;
            GEMMState substate = state;
            substrategy.remHandling[loop] = RemainderHandling::Ignore;
            substrategy.unroll[loop] = chunkSize;
            substate.isNested = true;
            status << "Generating " << "MNK"[static_cast<int>(loop)] << " remainder kernel with unroll " << chunkSize << '.' << status_stream::endl;
            if (!(this->*func)(problem, substrategy, substate)) {
                discardStream();
                return false;
            }
        }

        // Adjust remainder.
        add(1, remainders, remainders, -chunkSize);

        // Adjust pointers as needed.
        // A += i0 (N) i0 * lda (T, Pc)
        // B += j0 * ldb (N, Pr) j0 (T)
        // C += i0 + j0 * ldc (N, Pr) j0 + i0 * ldc (T, Pc)
        switch (loop) {
            case LoopM:
                if (problem.A.layout == MatrixLayout::N)
                    eadd(1, state.effA, state.effA, chunkSize * Ta, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.lda, chunkSize);
                    eadd(1, state.effA, state.effA, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                if (problem.C.layout == MatrixLayout::N || problem.C.layout == MatrixLayout::Pr)
                    eadd(1, state.effC, state.effC, chunkSize * transaction_safe, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.lda, chunkSize);
                    eadd(1, state.effA, state.effA, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                break;
            case LoopN:
                if (problem.B.layout == MatrixLayout::T)
                    eadd(1, state.effB, state.effB, chunkSize * Tb, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.ldb, chunkSize);
                    eadd(1, state.effB, state.effB, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                if (problem.C.layout == MatrixLayout::T || problem.C.layout == MatrixLayout::Pc)
                    eadd(1, state.effC, state.effC, chunkSize * Tc, strategy, state);
                else {
                    Subregister temp = state.ra.alloc_sub<uint32_t>();
                    mulConstant(1, temp, state.inputs.ldb, chunkSize);
                    eadd(1, state.effB, state.effB, temp, strategy, state);
                    state.ra.safeRelease(temp);
                }
                break;
        }

        mark(label_next_rem);

        // Handle the remainder recursively.
        {
            GEMMStrategy substrategy = strategy;
            substrategy.remHandling[loop] = RemainderHandling::General;
            substrategy.unroll[loop] -= chunkSize;
            if (!mnRemainderHandling(loop, problem, substrategy, state, func)) {
                discardStream();
                return false;
            }
        }
    } /* end stream */

    appendCurrentStream();

    return true; /* success */
#endif
}

// Synthesize remainder handling in m/n simultaneously.
template <HW hw>
template <typename Problem>
bool Generator<hw>::mnJointSplitRemainderHandling(Problem &problem, GEMMStrategy &strategy, GEMMState &state,
                                                  bool (Generator<hw>::*func)(Problem, GEMMStrategy, GEMMState))
{
    Label label_done, label_remainder;
    bool success = false;

    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];

    pushStream(); do {
        // Jump to remainder loop if needed:
        //  - if m/n below split thresholds (when enabled)
        //  - if in a remainder kernel.
        bool wgCheck = wgRemCheck(problem, strategy);

        if (strategy.mSplitThresh && strategy.nSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.m, int32_t(strategy.mSplitThresh));
            cmp(1 | lt | f1[0], null.d(), state.inputs.n, int32_t(strategy.nSplitThresh));
            ejmpi(1 | f0[0] | anyv, label_remainder);
        } else if (strategy.mSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.m, int32_t(strategy.mSplitThresh));
            jmpi(1 | f0[0], label_remainder);
        } else if (strategy.nSplitThresh) {
            cmp(1 | lt | f0[0], null.d(), state.inputs.n, int32_t(strategy.nSplitThresh));
            jmpi(1 | f0[0], label_remainder);
        }
        if (wgCheck) {
            cmp(1 | lt | f0[0], null.d(), state.remaindersWG[LoopM], uint16_t(unrollM * strategy.wg[LoopM]));
            cmp(1 | lt | f1[0], null.d(), state.remaindersWG[LoopN], uint16_t(unrollN * strategy.wg[LoopN]));
        } else {
            cmp(1 | lt | f0[0], null.d(), state.remaindersFused[LoopM], uint16_t(unrollM));
            cmp(1 | lt | f1[0], null.d(), state.remaindersFused[LoopN], uint16_t(unrollN));
        }
        ejmpi(1 | f0[0] | anyv, label_remainder);
        // First generate code that ignores remainder handling.
        GEMMStrategy substrategy = strategy;
        substrategy.remHandling[LoopM] = RemainderHandling::Ignore;
        substrategy.remHandling[LoopN] = RemainderHandling::Ignore;

        status << "Generating MN non-remainder kernel." << status_stream::endl;
        if (!(this->*func)(problem, substrategy, state)) {
            status << "Non-remainder kernel failed, aborting." << status_stream::endl;
            break;
        }

        // Return, unless this is part of a larger computation, in which case jump to end.
        if (state.isNested)
            jmpi(1, label_done);
        else
            epilogue(strategy, state);

        mark(label_remainder);

        // Finally, generate remainder handling kernel.
        substrategy = strategy;
        substrategy.remHandling[LoopM] = substrategy.remHandling[LoopN] =
            (wgCheck ? RemainderHandling::General : RemainderHandling::KnownRemainder);
        substrategy.checkAdd32 = substrategy.checkAdd32Rem();
        status << "Generating MN general kernel." << status_stream::endl;
        success = (this->*func)(problem, substrategy, state);

        mark(label_done);
    } while (false);

    success ? appendCurrentStream() : discardStream();

    return success;
}

GEMMSTONE_NAMESPACE_END
