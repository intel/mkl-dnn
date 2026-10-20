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


#include <numeric>

#include "alloc_utils.hpp"
#include "atomic_fusions.hpp"
#include "gemmstone/generator.hpp"
#include "generator/pieces/copy_plan.hpp"
#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"
#include "ngen_object_helpers.hpp"
#include "state_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


// Decide whether to use the legacy post-op injector inside C update.
// Needed if we can't convert C to f32 in-place, but doesn't support binary post-ops.
static inline bool useEltwiseInjector(const GEMMProblem &problem)
{
    return problem.hasNonSum1PostOp() && (problem.Tc.paddedSize() < 4);
}

// Main entrypoint for the C matrix update step.
template <HW hw>
bool Generator<hw>::gemmUpdateC(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
#ifdef STANDALONE
    if (problem.mock.cUpdate)
        return true;
#endif

    auto Tc = problem.Tc;
    auto Tco = problem.Tco;
    auto Ts = problem.Ts;

    status << "C update" << status_stream::endl;

    auto &beta = problem.beta;

    if (strategy.fuseBeta && !strategy.altFusedBeta && !strategy.fusePostOps && !strategy.kParallelVariable)
        beta = 1;
    else if (strategy.cLoadAhead) {
        beta = 0;
        if (!problem.alpha1())
            stub();
    }

    // C early offset.
    if (problem.cOffset == COffset::Pre) {
        if (Tc.isInteger() && Tco.isFP() && Ts.isFP()) {
            if (!gemmConvertC(Ts, problem, strategy, state))
                return false;
        } else if (Tc.isInteger() ^ Tco.isInteger()) {
            // It's unclear what data type to accumulate in for this scenario.
            stub();
        }
        if (!gemmApplyCOffsetDispatch(problem, strategy, state)) return false;
    }

#ifndef STANDALONE
    // Prepare legacy eltwise postop injector if configured.
    GRFRange postOpScratch;
    if (useEltwiseInjector(problem)) {
        if (problem.hasBinaryPostOp()) stub();

        postOpInjector.reset(new Injector(this, problem.Ts.ngen(), problem.postOps.ops, GRFRange(), problem.postOps.fwd));
        if (!postOpInjector) stub();

        postOpScratch = state.ra.try_alloc_range(postOpInjector->preferred_scratch_regs());
        if (postOpScratch.isInvalid())
            postOpScratch = state.ra.alloc_range(postOpInjector->min_scratch_regs());
        postOpInjector->set_scratch(postOpScratch);
    }
#endif

    // Convert C to the type of alpha/beta if needed and if possible (no data size change).
    // If not possible, must be done at a lower level during C update.
    bool successfulConvert = true;

    if (problem.needsTsConvert())
        successfulConvert = gemmConvertC(Ts, problem, strategy, state);

    // Scale by alpha now if alpha and beta are both nontrivial. Todo: move above beta = 0 check,
    //  handle double precision correctly (load alpha to register first).
    // Also scale if atomically updating C or for split-complex.
    bool nontrivialAlpha = !problem.alpha1() && !problem.alphaM1();
    bool forceScale = !problem.alpha1() && strategy.C.atomic;

    if (!problem.alpha1() && problem.hasBinaryPostOp()) {
        forceScale = true;
        if (!successfulConvert) stub();
    }

    if (successfulConvert && ((nontrivialAlpha && !problem.beta1()) || forceScale)) {
        gemmAlphaScale(problem, strategy, state);
    }

    // Do the actual updating.
    if (!gemmAccessC(COperation::UpdateStore, problem, strategy, state))
        return false;

#ifndef STANDALONE
    // Postop cleanup.
    if (useEltwiseInjector(problem)) {
        postOpInjector.reset();
        state.ra.safeRelease(postOpScratch);
    }
#endif

    // Free C data and layout.
    safeReleaseRanges(state.C_regs, state);
    state.C_layout.clear();
    state.C_layoutExt.clear();

    state.raVFlag.safeRelease(state.flagSwizzle);

    // Free A/B sum data and layouts.
    safeReleaseRanges(state.As_regs, state);
    safeReleaseRanges(state.Bs_regs, state);
    state.As_layout.clear();
    state.Bs_layout.clear();

    // Success!
    return true;
}

// Load from, update, and/or store to C, with complete remainder handling.
// If op == COperation::Load, only load C.
// If op == COperation::Update, load and update C.
// If op == COperation::UpdateStore, perform full C update with alpha/beta scaling. Unless state.isNested == true, assumed
//   to be the conclusion of the kernel.
// If op == COperation::Store, only store C.
template <HW hw>
bool Generator<hw>::gemmAccessC(COperation op, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    Label labelStdCRemainder, labelAltCRemainder, labelBlock2DCRemainder, labelCRemDone, labelSkip;

    if (op == COperation::Store) {
        auto modProblem = problem;
        modProblem.alpha = 1;
        modProblem.beta = 0;
#ifndef STANDALONE
        modProblem.postOps = PostOps{};
#endif
        modProblem.cOffset = COffset::None;
        return gemmAccessC(COperation::UpdateStore, modProblem, strategy, state);
    }

    gemmAccessSums(op, problem, strategy, state);

    int C_count = (op == COperation::UpdateStore) ? state.C_count : 1;
    bool remainderM = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
    bool remainderN = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
    bool remM_C, remN_C;
    getCRemainders(hw, problem, strategy, remM_C, remN_C);
    bool block2DOverride = (problem.Ta_ext.bits() == 2) || (problem.Tb_ext.bits() == 2);
    bool block2DCRemainder = strategy.block2DCRemainder && !strategy.C.padded && (remainderM || remainderN)
        && !block2DOverride;
    bool block2DCFull = strategy.block2DCFull && !isPacked(problem.C.layout) && !block2DOverride;
    block2DCRemainder |= block2DCFull;
    block2DCRemainder &= !strategy.C.atomic;
    block2DCFull &= !strategy.C.atomic;
    bool altCRemainder = strategy.altCRemainder && !strategy.C.padded && (remainderM || remainderN || problem.gemmt());
    bool stdCRemainder = !(altCRemainder && (strategy.remHandling[LoopM] == RemainderHandling::KnownRemainder)
                                         && (strategy.remHandling[LoopN] == RemainderHandling::KnownRemainder));

    if (state.allowEmptyC && (remainderM || remainderN)) {
        if (!state.isNested) stub();
        int simt = strategy.fused ? 16 : 1;
        InstructionModifier mod;
        if (remainderM) cmp(simt | le | f0[0], null.ud(), state.remainders[LoopM], 0);
        if (remainderN) cmp(simt | le | f1[0], null.ud(), state.remainders[LoopN], 0);
        if (remainderM && remainderN)
            mod = mod | f0[0] | anyv;
        else if (remainderM)
            mod = mod | f0[0];
        else if (remainderN)
            mod = mod | f1[0];
        strategy.fused ? goto12(16 | mod, labelSkip)
                       :  ejmpi(1  | mod, labelSkip);
    }

    // MX dst scaling must be applied after all other post-ops before store.
    bool splitUpdateStore = (problem.cOffset == COffset::Post) || problem.hasCMXScale();

#ifndef STANDALONE
    // New post-op path: do all post-ops up to sum, if any.
    size_t poSum = 0;
    bool newPostOps = !useEltwiseInjector(problem);
    if (op == COperation::UpdateStore && newPostOps) {
        for (poSum = 0; poSum < problem.postOps.len(); poSum++)
            if (problem.postOps[poSum].is_sum())
                break;
        gemmApplyPostOps(0, poSum, problem, strategy, state);
        splitUpdateStore |= (poSum + 1 < problem.postOps.len());
    }
#endif

    if (op == COperation::UpdateStore && splitUpdateStore) {
        // C postoffset is implemented by splitting the update and store steps.
        bool ok = true;
        bool oldAllowEmptyC = state.allowEmptyC;
        state.allowEmptyC = false;

        if (!(problem.alpha1() && problem.beta0()))
            ok = ok && gemmAccessC(COperation::Update, problem, strategy, state);

        auto storeProblem = problem;
        storeProblem.cOffset = COffset::None;
        storeProblem.alpha = 1;
        storeProblem.beta = 0;

#ifndef STANDALONE
        // Do any post-sum post-ops.
        if (newPostOps)
            gemmApplyPostOps(poSum + 1, problem.postOps.len(), problem, strategy, state);
        storeProblem.postOps = PostOps{};
        gemmApplyMXScale(problem, strategy, state);
#endif
        storeProblem.cMXScale = false;

        if (problem.cOffset == COffset::Post)
            ok = ok && gemmApplyCOffsetDispatch(problem, strategy, state);

        ok = ok && gemmAccessC(COperation::UpdateStore, storeProblem, strategy, state);

        state.allowEmptyC = oldAllowEmptyC;
        if (ok && state.allowEmptyC && (remainderM || remainderN)) {
            mark(labelSkip);
            if (strategy.fused) join(16);
        }
        return ok;
    }

    auto leave = [&] {
        if (state.isNested || (op != COperation::UpdateStore))
            jmpi(1, labelCRemDone);
        else
            epilogue(strategy, state);
    };

    if (stdCRemainder) {
        // Check to see if we should jump to alternate C remainder handling path, when enabled:
        //  - if this a remainder kernel
        //  - for triangular updates, if the diagonal crosses this block.
        //       When fusing, check diagonal for thread 0 for (fused in n) upper/m lower, thread 1 for n lower/m upper.
        if (block2DCFull)
            jmpi(1, labelBlock2DCRemainder);
        else if (altCRemainder || block2DCRemainder) {
            if (remainderM) cmp(1 | lt | f0[0], null.ud(), state.remaindersFused[LoopM], strategy.unroll[LoopM]);
            if (remainderN) cmp(1 | lt | f1[0], null.ud(), state.remaindersFused[LoopN], strategy.unroll[LoopN]);

            auto &remLabel = block2DCRemainder ? labelBlock2DCRemainder : labelAltCRemainder;
            if (remainderM && remainderN)
                ejmpi(1 | f0[0] | anyv, remLabel);
            else if (remainderM)
                jmpi(1 | f0[0], remLabel);
            else if (remainderN)
                jmpi(1 | f1[0], remLabel);
        }

        if (block2DCRemainder && !altCRemainder)
            mark(labelStdCRemainder);

        // Release the all-purpose flag temporarily to free up flag registers if it won't be needed.
        auto saveFlagAP = state.flagAP;
        if (!problem.hasPostOp())
        if (!strategy.fused && !strategy.noJumpTables && state.emulate.flag != state.flagAP)
            state.raVFlag.safeRelease(state.flagAP);

        // Decide on the C remainder handling strategy.
        bool fragments[2] = {false, false};
        bool fragPositives[2] = {true, true};
        int fragSizes[2] = {1 << 16, 1 << 16};

        // Check for fragmenting.
        bool useNonatomic = !(state.C_layoutExtNonatomicUnmasked.empty() || strategy.C.atomic);
        auto &C_layoutExt = state.C_layoutExt;
        auto &C_layoutExtUnmasked = useNonatomic ? state.C_layoutExtNonatomicUnmasked
                                                 : state.C_layoutExtUnmasked;
        bool remDescs[2] = {false, false};
        bool remMasks[2] = {false, false};

        // Loop over rows (rc = 0) and columns (rc = 1).
        for (int rc = 0; rc < 2; rc++) {
            if (!(rc ? remN_C : remM_C)) continue;      // Skip if not doing remainder handling in this dimension.

            for (auto &l : C_layoutExt) {
                auto qFragment = rc ? l.colFragment : l.rowFragment;
                bool qZeroOK = rc ? l.noColsOK : l.noRowsOK;
                bool qMasked = rc ? (bool) l.colMask : (bool) l.rowMask;
                bool qDescRem = rc ? l.descRemC : l.descRemR;

                if (qFragment > 0) {
                    fragments[rc] = true;
                    fragSizes[rc] = std::min<int>(fragSizes[rc], qFragment);
                    if (qZeroOK) fragPositives[rc] = false;

                    if (qFragment > 1) {
                        remDescs[rc] |= qDescRem;
                        remMasks[rc] |= !qDescRem;
                    }
                } else
                    remMasks[rc] |= qMasked;
            }
        }

        // Disable fragmentation if fragment size is bigger than unroll.
        fragments[0] &= fragSizes[0] < strategy.unroll[LoopM];
        fragments[1] &= fragSizes[1] < strategy.unroll[LoopN];

        // Sanity check the requirements.
        if ((remDescs[0] && remMasks[0]) || (remDescs[1] && remMasks[1])) {
            status << "Different remainder types mixed in C layout." << status_stream::endl;
            return false;
        }
        if (remMasks[0] && remMasks[1]) {
            status << "Both dimensions are masked (not supported)." << status_stream::endl;
            return false;
        }
        if (remDescs[0] && remDescs[1]) {
            status << "Both dimensions use descriptors (not supported)." << status_stream::endl;
            return false;
        }

        // Set remainder handling types.
        StdCRemType remTypes[2] = {StdCRemType::Ignore, StdCRemType::Ignore};
        for (int rc = 0; rc < 2; rc++) {
            if (remDescs[rc])      remTypes[rc] = StdCRemType::Descriptor;
            else if (remMasks[rc]) remTypes[rc] = StdCRemType::Mask;
        }

        // Decide whether to do m or n first. Criteria, in order of priority:
        //   - Do an ignored dimension first.
        //   - Do a fragmented dimension first.
        //   - Do descriptors first.
        //   - Do whichever dimension of C is strided first.
        bool nFirst;
        if (remTypes[0] == StdCRemType::Ignore || remTypes[1] == StdCRemType::Ignore)
            nFirst = (remTypes[1] == StdCRemType::Ignore);
        else if (fragments[0] != fragments[1])
            nFirst = fragments[1];
        else if (remDescs[0] || remDescs[1])
            nFirst = remDescs[1];
        else
            nFirst = (problem.C.layout == MatrixLayout::N);

        // Cache ldc multiples.
        gemmCacheLDCMultiples(problem, strategy, state);

        // Prepare for load/store descriptor generation.
        if (remDescs[0] || remDescs[1])
            setupTeardownLoadStoreDesc(true, C_layoutExt, strategy, state);

        // Set up address for the beginning of C.
        RegisterRange C_addr0[2], C_addr0Unmasked[2];
        setupCAddr0(C_addr0, C_addr0Unmasked, C_layoutExt, C_layoutExtUnmasked, C_count, problem, strategy, state);

        // Try to load C masks. If that fails, fragment the masked dimension down to the size of current blocks.
        vector<MaskAssignment> masks;
        if (!assignMasks(C_layoutExt, LoopM, LoopN, masks, strategy, state)) {
            for (int rc = 0; rc < 2; rc++) {
                if (remMasks[rc]) {
                    fragments[rc] = true;
                    fragSizes[rc] = rc ? C_layoutExt[0].nc : C_layoutExt[0].nr;
                }
            }
        } else
            loadMasks(masks, state.remainders, strategy, state);

        // Call the remainder handling routine. If it fails, try again, switching M and N.
        // If that still fails, then try again with complete fragmentation if partial
        //  fragmentation attempted the first time.
        bool columns[2] = {nFirst, !nFirst};
        bool switchedColumns[2] = {!nFirst, nFirst};
        do {
            if (doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false, columns, remTypes, fragments, fragPositives, fragSizes, C_addr0, C_addr0Unmasked, op, masks, problem, strategy, state)) break;
            if (doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false, switchedColumns, remTypes, fragments, fragPositives, fragSizes, C_addr0, C_addr0Unmasked, op, masks, problem, strategy, state)) break;

            if ((fragments[0] && (fragSizes[0] > 1)) || (fragments[1] && (fragSizes[1] > 1))) {
                fragSizes[0] = fragSizes[1] = 1;

                if (doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false, columns, remTypes, fragments, fragPositives, fragSizes, C_addr0, C_addr0Unmasked, op, masks, problem, strategy, state)) break;
                if (doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false, switchedColumns, remTypes, fragments, fragPositives, fragSizes, C_addr0, C_addr0Unmasked, op, masks, problem, strategy, state)) break;
            }
            return false;
        } while (false);

        // Free cached ldc multiples.
        for (int q = 0; q < state.C_count; q++)
            releaseLDMultiples(state.ldcMultiples[q], state);
        releaseIndexVec(state);

        // Free address header for block 0.
        for (int q = 0; q < C_count; q++)
            state.ra.safeRelease(C_addr0[q]);

        // Free C mask registers.
        safeReleaseMaskAssignments(masks, state);
        for (auto &block: C_layoutExt)
            block.clearFlag();

        // Clean up after load/store descriptor generation.
        if (remDescs[0] || remDescs[1])
            setupTeardownLoadStoreDesc(false, C_layoutExt, strategy, state);

        // Restore all-purpose flag.
        state.flagAP = saveFlagAP;
        state.raVFlag.claim(saveFlagAP);

        // Leave.
        if (block2DCRemainder || altCRemainder)
            leave();
    }

    // Do block 2D C remainder handling if enabled.
    if (block2DCRemainder) {
        mark(labelBlock2DCRemainder);

        // Check for transposition.
        bool doTranspose = isTransposing(strategy.C.accessType);

        // Check if alignment requirements are met, and stride fits in 24 bits.
        auto Tc = problem.Tc, Tc_ext = problem.Tc_ext;
        uint16_t align = block2DMinAlignment(hw, problem.C, strategy.C, true);

        for (int q = 0; q < state.C_count; q++) {
            bool checkAlign = (problem.C.alignment % align) != 0;
            bool checkWidth = (q == 0 && Tc_ext.paddedSize() < 4 && op != COperation::Load);
            auto &labelNonBlock2DRem = altCRemainder ? labelAltCRemainder : labelStdCRemainder;

            if (checkAlign) {
                and_(1 | nz | f0[0], null.uw(), state.effC[q].uw(),       align - 1);
                and_(1 | nz | f1[0], null.uw(), state.inputs.ldc[q].uw(), align - 1);
            }
            if (checkWidth)
                and_(1 | nz | f0[1], null.uw(), state.remainders[isColMajor(problem.C.layout) ? LoopM : LoopN], (4 / Tc_ext) - 1);
            and_(1 | nz | f1[1], null.ud(), state.inputs.ldc[q], 0xFF000000);
            if (checkAlign)
                ejmpi(1 | f0[0] | anyv, labelNonBlock2DRem);
            if (checkWidth)
                jmpi(1 | f0[1], labelNonBlock2DRem);
            jmpi(1 | f1[1], labelNonBlock2DRem);
        }

        status << "2D block C remainder handling (m/n)" << status_stream::endl;

        // Rustle up a new layout.
        // Match existing C layout if possible and deemed worthwhile.
        RegisterLayout C_layout2D;
        auto modProblem = problem;
        auto modStrategy = strategy;
        auto modState = state;

        modProblem.C.setAlignment(align);
        modStrategy.C = state.Cext_strategy;
        modStrategy.C.newDP = true;
        modStrategy.C.address2D = true;
        modStrategy.C.accessType = doTranspose ? AccessType::Block2DTranspose : AccessType::Block2D;

        auto &C_layout            = modState.C_layout;
        auto &C_layoutExt         = modState.C_layoutExt;
        auto &C_layoutExtUnmasked = modState.C_layoutExtUnmasked;

        bool inplace = (Tc == Tc_ext) && (C_layout2D = C_layout.tryUpgradeToBlock2D(modProblem.C, modStrategy.C));

        for (auto &block: C_layout2D)
            inplace = inplace && state.C_regs[0].contiguous(block.offsetReg(), block.nregs());

        modState.copyC = !inplace;

        if (inplace)
            C_layoutExt = std::move(C_layout2D);
        else
            C_layoutExt = RegisterLayout(hw, Tc_ext, strategy.unroll[LoopM], strategy.unroll[LoopN], modProblem.C, modStrategy.C, remM_C, remN_C, true, AllowFragDesc);

        C_layoutExtUnmasked.clear();
        C_layout.unlinkFromMemory();

        // Do the update.
        Address2DParams params;
        params.rows = state.remainders[LoopM];
        params.cols = state.remainders[LoopN];

        RegisterRange C_addr0[2], C_addr0Unmasked[2];
        setupCAddr0(C_addr0, C_addr0Unmasked, C_layoutExt, C_layoutExtUnmasked, C_count, modProblem, modStrategy, modState, &params);

        bool columns[2] = {false, true};
        StdCRemType remTypes[2] = {StdCRemType::Ignore, StdCRemType::Ignore};
        bool fragments[2] = {false, false};
        bool fragPositives[2] = {false, false};
        int fragSizes[2] = {1 << 16, 1 << 16};
        vector<MaskAssignment> masks;

        if (!doStdCRemainder(C_layoutExt, C_layoutExtUnmasked, false, columns, remTypes, fragments, fragPositives, fragSizes, C_addr0, C_addr0Unmasked, op, masks, modProblem, modStrategy, modState)) stub();

        if (altCRemainder)
            leave();
    }

    // Do alternate C remainder handling if enabled.
    if (altCRemainder) {
        mark(labelAltCRemainder);
        doAlternateCRemainder(op, problem, strategy, state);
    }

    if (altCRemainder || block2DCRemainder)
        mark(labelCRemDone);

    if (state.allowEmptyC && (remainderM || remainderN)) {
        mark(labelSkip);
        if (strategy.fused) join(16);
    }

    return true;    /* Successful! */
}

// Update C. If configured, choose between regular beta and beta = 0 or beta = 1 updates now.
template <HW hw>
bool Generator<hw>::gemmUpdateCDispatch(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto &beta = problem.beta;
    auto vbetar = state.inputs.beta_real;

    // Decide if we need to create special paths for beta = 0 or beta = 1 cases.
    bool checkBeta0 =  problem.checkBeta0 && !beta.fixed();
    bool checkBeta1 = strategy.checkBeta1 && !beta.fixed();
    bool checkTRMMBeta1 = state.beta1.isValid();

    bool fuseCheckBeta0 = false;
    bool fuseCheckBeta1 = false;

    bool oldNested = state.isNested;
    if (strategy.fuseBeta && strategy.altFusedBeta)
        state.isNested = true;

    if (strategy.fusePostOps) {
        if (!state.useTempC) {
            fuseCheckBeta0 |= strategy.kParallelVariable;
            checkBeta0 |= fuseCheckBeta0;
        }
    } else if (strategy.fuseBeta) {
        if (strategy.kParallelVariable || strategy.altFusedBeta) {
            fuseCheckBeta1 = !problem.beta1();
            checkBeta1 |= fuseCheckBeta1;
        } else
            checkBeta0 = checkBeta1 = false;
    }

    if (checkTRMMBeta1 && (checkBeta0 || checkBeta1)) stub();

    // Decide if we need special paths for L1-uncached/cached cases.
    bool checkUC = strategy.altFusedBeta && strategy.kParallelVariable && !strategy.fusePostOps && strategy.C.newDP;

    auto C_l1UCW = makeL1Uncacheable(strategy.C.cachingW);
    auto Ce_l1UCW = makeL1Uncacheable(state.Cext_strategy.cachingW);

    checkUC = checkUC && (C_l1UCW != strategy.C.cachingW || Ce_l1UCW != state.Cext_strategy.cachingW);

    if (strategy.altFusedBeta && !checkUC && !strategy.fusePostOps)
        setCCachingW(strategy, state, C_l1UCW, Ce_l1UCW, strategy.C.atomic);

    // Generate the various paths needed.
    if (!checkBeta0 && !checkBeta1 && !checkTRMMBeta1 && !checkUC) {
        if (!gemmUpdateC(problem, strategy, state)) return false;
    } else {
        Label labelBeta0, labelBeta1, labelCached, labelBeta0Cached, labelBetaDone;
        InstructionModifier mod0 = 1 | f0[0];
        InstructionModifier mod1 = 1 | f0[1];
        InstructionModifier modC = 1 | f1[0];
        bool simtCF1 = false;

        if (checkUC)
            and_(1 | ze | f1[0], null.ud(), state.inputs.flags, FlagKPartitioned);

        if (checkBeta1 && !beta.fixed()) {
            auto immediate = cast(problem.Ts, 1.0);
            if (problem.Ts.size() > 4) {
                auto temp = state.ra.alloc_sub(problem.Ts.ngen());
                mov(1, temp, immediate);
                cmp(1 | eq | f0[1], vbetar.getReg(0), temp);
                state.ra.safeRelease(temp);
            } else
                cmp(1 | eq | f0[1], vbetar.getReg(0), immediate);
        }

        if (checkBeta0 && !beta.fixed()) {
            cmp0(1 | eq | f0[0], vbetar.getReg(0));
        }

        if (fuseCheckBeta1) {
            if (strategy.altFusedBeta) {
                auto mod = beta.fixed() ? (1 |          ze | f0[1])
                                        : (1 | ~f0[1] | ze | f0[1]);
                if (strategy.kParallelVariable && !checkUC) {
                    auto temp = state.ra.alloc_sub<uint32_t>();
                    xor_(1, temp, state.inputs.flags, FlagKPartitioned);
                    and_(mod, null.ud(), temp, FlagDidBeta | FlagKPartitioned);
                    state.ra.safeRelease(temp);
                } else
                    and_(mod, null.ud(), state.inputs.flags, FlagDidBeta);
            } else if (strategy.kParallelVariable) {
                auto mod = beta.fixed() ? (1 |          nz | f0[1])
                                        : (1 | ~f0[1] | nz | f0[1]);
                and_(mod, null.ud(), state.inputs.flags, FlagKPartitioned);
            }
        }

        if (fuseCheckBeta0) {
            auto mod = beta.fixed() ? (1 |          nz | f0[0])
                                    : (1 | ~f0[0] | nz | f0[0]);
            and_(mod, null.ud(), state.inputs.flags, FlagKPartitioned);
        }

        if (checkUC)
            jmpi(modC, labelCached);

        if (checkBeta0 && !fuseCheckBeta1)
            jmpi(mod0, labelBeta0);

        if (checkBeta1 || checkTRMMBeta1) {
            simtCF1 ?  if_(mod1, labelBeta1, labelBetaDone)
                    : jmpi(mod1, labelBeta1);
        }

        if (checkBeta0 && fuseCheckBeta1)
            jmpi(mod0, labelBeta0);

        // Regular update.
        {
            auto subproblem = problem;
            auto substrategy = strategy;
            auto substate = state;

            if (strategy.C.atomic && !strategy.C.base.isStateless() && !strategy.C.newDP)
                stub(); /* need to shift addresses */
            substrategy.C.atomic = substrategy.CO.atomic = false;
            substate.Cext_strategy.atomic = false;
            if (checkUC)
                setCCachingW(substrategy, substate, C_l1UCW, Ce_l1UCW, false);

            if (!gemmUpdateC(subproblem, substrategy, substate)) return false;
        }

        simtCF1        ? else_(16, labelBetaDone) :
        state.isNested ? jmpi(1, labelBetaDone)
                       : epilogue(strategy, state);

        // beta = 1 update.
        if (checkBeta1 || checkTRMMBeta1) {
            status << "Special path: beta = 1" << status_stream::endl;
            mark(labelBeta1);

            auto subproblem = problem;
            auto substate = state;

            subproblem.beta = 1;

#ifndef STANDALONE
            if (subproblem.postOps.len() > 0) {
                auto &lastPO = subproblem.postOps[subproblem.postOps.len() - 1];
                if (lastPO.is_sum()) lastPO.set_scale(1.0f);
            }
#endif

            if (!gemmUpdateC(subproblem, strategy, substate)) return false;

            if (checkBeta0) {
                (simtCF1 || state.isNested) ? jmpi(1, labelBetaDone)
                                            : epilogue(strategy, state);
            }
        }

        // beta = 0 update.
        if (checkBeta0) {
            status << "Special path: beta = 0" << status_stream::endl;
            mark(labelBeta0);

            auto subproblem = problem;
            auto substrategy = strategy;
            auto substate = state;

            subproblem.beta = 0;
            subproblem.removeFinalSumPostOp();
            if (checkUC)
                setCCachingW(substrategy, substate, C_l1UCW, Ce_l1UCW, false);

            substrategy.C.atomic = substrategy.CO.atomic = false;
            substate.Cext_strategy.atomic = false;

            if (!gemmUpdateC(subproblem, substrategy, substate)) return false;
        }

        // Updates with L1 caching enabled for non-k-sliced tiles.
        if (checkUC) {
            state.isNested ? jmpi(1, labelBetaDone)
                           : epilogue(strategy, state);

            status << "Special path: L1 caching enabled" << status_stream::endl;
            mark(labelCached);
            if (checkBeta0) jmpi(mod0, labelBeta0Cached);

            {
                auto subproblem = problem;
                auto substrategy = strategy;
                auto substate = state;

                substrategy.C.atomic = substrategy.CO.atomic = false;
                substate.Cext_strategy.atomic = false;
                if (!gemmUpdateC(subproblem, substrategy, substate)) return false;
            }

            if (checkBeta0) {
                state.isNested ? jmpi(1, labelBetaDone)
                               : epilogue(strategy, state);

                status << "Special path: beta = 0, L1 caching enabled" << status_stream::endl;
                mark(labelBeta0Cached);

                auto subproblem = problem;
                auto substrategy = strategy;
                auto substate = state;

                subproblem.beta = 0;
                subproblem.removeFinalSumPostOp();
                substrategy.C.atomic = substrategy.CO.atomic = false;
                substate.Cext_strategy.atomic = false;

                if (!gemmUpdateC(subproblem, substrategy, substate)) return false;
            }
        }

        mark(labelBetaDone);
        if (simtCF1) endif(16);
    }

    // Cleanup.
    state.isNested = oldNested;

    return true;
}

// Perform C update operation on C_acc, given original C data in C_load.
// All inputs and outputs are assumed to be of type problem.Ts.
template <HW hw>
void Generator<hw>::updateC(const GRFMultirange &C_acc, const GRFMultirange &C_accSwap, const GRFMultirange &C_load,
                            const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Ts = problem.Ts;
    auto &alpha = problem.alpha;
    auto &beta = problem.beta;
    auto valphar = state.inputs.alpha_real;
    auto vbetar  = state.inputs.beta_real;

    bool alpha1 = (alpha == 1);
    bool alphaM1 = (alpha == -1);
    bool beta1 = (beta == 1);
    bool beta0 = (beta == 0);
    bool betaM1 = (beta == -1);

#define FOR_EACH_C(f) do { \
        map(hw, state.Tacc.real(), C_load, C_acc, strategy, [&](int esize, GRF loaded, GRF acc) { \
            f; \
        }); \
    } while (false)

#define FOR_EACH_C_CX(f) do { \
        map(hw, state.Tacc.real(), C_load, C_acc, C_accSwap, strategy, [&](int esize, GRF loaded, GRF acc, GRF accswap) { \
            f; \
        }); \
    } while (false)

    if (!beta0) {
        if (alpha1 || alphaM1) {
            if (beta1)
                FOR_EACH_C(add(esize, acc, loaded, alpha1 ? acc : -acc));
            else if (betaM1)
                FOR_EACH_C(add(esize, acc, -loaded, alpha1 ? acc : -acc));
            else if (beta.fixed())
                stub();                                                                     // beta should be put in a register first.
            else {
                FOR_EACH_C(mad(esize, acc, alpha1 ? acc : -acc, loaded, vbetar.getRegAvoiding(hw, loaded)));
            }
        } else {
            bool neg = false;
            if (!beta1) {
                if (betaM1)
                    neg = true;
                else if (!beta.fixed() && !Ts.isComplex())
                    FOR_EACH_C(mul(esize, loaded, loaded, vbetar.getRegAvoiding(hw, acc)));
                else
                    stub();
            }
            if (alpha.fixed())
                stub();                                                                     // alpha should be put in a register first.
            else {
                FOR_EACH_C(mad(esize, acc, neg ? -loaded : loaded, acc, valphar.getRegAvoiding(hw, acc)));
            }
        }
    } else if (alphaM1)
        FOR_EACH_C(mov(esize, acc, -acc));
    else if (alpha1)
        /* no op */;
    else if (alpha.fixed())
        stub();                                                                             // alpha should be put in a register first.
    else {
        FOR_EACH_C(mul(esize, acc, acc, valphar.getRegAvoiding(hw, acc)));
    }

#ifndef STANDALONE
    if (useEltwiseInjector(problem)) {
        Label labelPostOpDone;
        bool allocFlag = state.flagAP.isInvalid();
        auto flagNonfinal = allocFlag ? state.raVFlag.alloc() : state.flagAP;
        and_(1 | nz | flagNonfinal, null.ud(), state.inputs.flags,
                FlagNonfinalKBlock);
        jmpi(1 | flagNonfinal, labelPostOpDone);
        if (allocFlag)
            state.raVFlag.safeRelease(flagNonfinal);
        if (state.Tacc != Type::f32 || !postOpInjector) stub();
        for (const auto &range: C_acc.ranges)
            postOpInjector->compute(range);
        mark(labelPostOpDone);
    }
#endif

#undef FOR_EACH_C
#undef FOR_EACH_C_CX
}


static inline void getDefaultCParams(Address2DParams &params, GEMMState &state)
{
    params.rows = state.inputs.m;
    params.cols = state.inputs.n;
    params.offR = state.i0;
    params.offC = state.j0;
    params.remR = state.remainders[LoopM];
    params.remC = state.remainders[LoopN];
}

// Update an entire C layout.
template <HW hw>
void Generator<hw>::updateCLayout(const RegisterLayout &layoutExt, const RegisterRange (&C_addr0)[2],
                                  const RegisterBlock &C_block0, COperation op,
                                  const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
#define FOR_EACH_C for (int q = 0; q < C_count; q++)
    auto Tc = problem.Tc, Tc_ext = problem.Tc_ext, Ts = problem.Ts;
    bool loadOnly = (op == COperation::Load);
    bool atomicUpdate = (op == COperation::UpdateStore) && strategy.C.atomic;
    bool beta0 = problem.beta0();
    bool needLoad = (!beta0 && !loadOnly);
    bool copyC = state.copyC;
    int C_count = (op == COperation::UpdateStore) ? state.C_count : 1;

    int nblocks = layoutExt.blocks();
    bool haveDescs = layoutExt[0].descAssigned;

    vector<RegisterRange> (&C_addrs)[2] = state.C_addrs;
    GRFMultirange C_extRange, C_copyRange;
    GRFMultirange &C_accRange = state.C_regs[0];
    auto &C_extRegs = C_extRange.ranges;
    auto &C_copyRegs = C_copyRange.ranges;
    vector<GRFRange> C_convertRegs;

    FOR_EACH_C C_addrs[q].clear();

    // Map layout to blocks in internal C layout as needed.
    RegisterLayout layout;
    vector<int> blockMap;
    if (copyC)
        layout = state.C_layout.reblock(blockMap, layoutExt);
    else {
        layout = layoutExt;
        blockMap.resize(nblocks + 1);
        for (int i = 0; i <= nblocks; i++)
            blockMap[i] = i;
    }

    // Prepare for late C conversion.
    bool lateCConvert = (!loadOnly && !atomicUpdate && problem.needsTsConvert() && state.Tacc != Ts);
    bool copyCLoad = needLoad && (copyC || lateCConvert);
    if (lateCConvert && Tc.isComplex()) stub();

    // Load as much of C as is possible at a time, given register space.
    for (int lstart = 0; lstart < nblocks;) {
        int lend;

        // Allocate address and data registers for C updating. If allocator chokes,
        //  proceed with the registers we were able to allocate.
        //
        // At the same time, build up three layouts for this chunk of C:
        //   sublayoutExt:   C data to be loaded/stored
        //   sublayoutCopy:  copied C data
        //   sublayoutAcc:   C data in accumulators
        bool allocOK = true;
        auto tryAlloc = [&](int regs, RegFile rf, Bundle hint = Bundle()) {
            if (rf != RegFileGRF)
                stub("Must be GRF");
            RegisterRange range = state.ra.try_alloc_range(regs, hint);
            allocOK &= range.isValid();
            return range;
        };

        // Save some space for temporary allocations inside C update.
        int saveRegs = 1;
        if (Tc != Tc_ext)
            saveRegs += 16;
        auto save = chunkAlloc(saveRegs, 2, state);

        vector<RegisterBlock> sublistExt, sublistCopy, sublistAcc;
        size_t sublistCopySize = 0;
        int bytes = 0, bytesConvert = 0;
        auto initOA = layoutExt[lstart].offsetAddr;
        auto lanchor = lstart;
        int tokens = 0, maxTokens = 256;
        if (needLoad)
            maxTokens = tokenCount(hw, strategy.GRFs);

        for (lend = lstart; (lend < nblocks) && (tokens < maxTokens); lend++, tokens++) {
            auto li0 = blockMap[lend], li1 = blockMap[lend + 1];
            int expand = lateCConvert ? div_up(Ts.size(), state.Tacc.size()) : 1;

            if (copyCLoad) for (int li = li0; li < li1; li++) {
                auto block = layout[li];
                block.compact(state.Tacc);
                block.offsetBytes = bytesConvert;
                bytesConvert += block.nregs() * expand * GRF::bytes(hw);
                sublistCopy.push_back(block);
            }

            auto blockExt = layoutExt[lend];
            bool origin = (blockExt.offsetR == 0 && blockExt.offsetC == 0 && blockExt.cxComponent <= 0);
            if (blockExt.offsetAddr == 0)
                initOA = 0, lanchor = lend;
            else {
                bool is2D = strategy.C.address2D;
                blockExt.subAddrOffset(initOA, is2D);  // Handle case where we start with an offset block.
                if (!copyC)
                    layout[lend].subAddrOffset(initOA, is2D);
            }
            auto naddr = blockExt.addrGRFs(problem.C, strategy.C);
            RegFile allocaterf = RegFileGRF;
            FOR_EACH_C C_addrs[q].push_back(origin ? C_addr0[q] :
                                            (blockExt.offsetAddr == 0) ? tryAlloc(naddr, allocaterf)
                                                                        : C_addrs[q][lanchor - lstart]);
            if (needLoad || copyC) {
                auto new_reg = tryAlloc(blockExt.nregs(), RegFileGRF, getHint(HintType::CLoad, strategy));
                C_extRegs.push_back(new_reg);
            }
            if (copyCLoad) for (int li = li0; li < li1; li++) {
                auto new_reg = tryAlloc(sublistCopy[li - li0 + sublistCopySize].nregs() * expand, RegFileGRF, getHint(HintType::CLoad, strategy));
                C_copyRegs.push_back(new_reg);
            }
            if (lateCConvert) for (int li = li0; li < li1; li++) {
                auto new_reg = tryAlloc(layout[li].nregs() * expand, RegFileGRF);
                C_convertRegs.push_back(new_reg);
            }
            if (!allocOK)
                break;

            blockExt.offsetBytes = bytes;
            bytes += blockExt.nregs() * GRF::bytes(hw);
            sublistExt.push_back(blockExt);

            sublistCopySize = sublistCopy.size();
        }

        if (lstart == lend) throw out_of_registers_exception();

        sublistCopy.resize(sublistCopySize);

        int listart = blockMap[lstart];
        int liend   = blockMap[lend];

        sublistAcc.reserve(liend - listart);
        for (int l = listart; l < liend; l++)
            sublistAcc.push_back(layout[l]);

        safeReleaseRanges(save, state);

        auto sublistWith0 = sublistExt;
        sublistWith0.insert(sublistWith0.begin(), C_block0);

        RegisterLayout sublayoutExt(Tc_ext, problem.C, strategy.C, std::move(sublistExt));
        RegisterLayout sublayoutWith0(Tc_ext, problem.C, strategy.C, std::move(sublistWith0));
        RegisterLayout sublayoutAcc(state.Tacc, layout.addressing(), layout.addressingStrategy(), std::move(sublistAcc));
        RegisterLayout sublayoutCopy(state.Tacc, std::move(sublistCopy));

        // Set up C addresses relative to prior blocks.
        // TODO: use inline address offsets instead of setupAddrRel for constant offsets.
        FOR_EACH_C {
            Address2DParams C_params;
            auto C_addrsWith0 = C_addrs[q];
            C_addrsWith0.insert(C_addrsWith0.begin(), C_addr0[q]);
            getDefaultCParams(C_params, state);
            setupAddr(C_addrsWith0, state.effC[q], sublayoutWith0, state.inputs.ldc[q], strategy, state, C_params, state.ldcMultiples[q], 1);
        }

        if (atomicUpdate) {
            // Atomic update.
            // Alpha scaling is done earlier; beta scaling isn't supported.
            if (!problem.alpha1() || !problem.beta1()) stub();
            if (copyC)
                copyRegisters(sublayoutAcc, sublayoutExt, C_accRange, C_extRange, strategy, state);
            else if (state.Tacc != Tc_ext) {
                if (state.Tacc.size() != Tc_ext.size()) stub();
                for (auto &block: sublayoutAcc) {
                    auto C_acc = subrange(C_accRange, hw, state.Tacc, block);
                    convert(C_acc, state.Tacc, Tc_ext, strategy, state);
                }
            }
            auto &sublayoutSrc = copyC ? sublayoutExt : sublayoutAcc;
            auto &C_srcRange = copyC ? C_extRange : C_accRange;
            FOR_EACH_C atomicAddMatrix(C_srcRange, sublayoutSrc, C_addrs[q], problem, strategy, state);
        } else {
            // Data types before and after scaling phase.
            auto Tacc_final = Tc;
            if (op == COperation::Update || (op == COperation::UpdateStore && copyC))
                Tacc_final = state.Tacc;

            // Regular update.
            auto Tload = Tc_ext;
            if (!beta0 || loadOnly) {
                // Set up a0.0 descriptor for loads if needed.
                if (lstart > 0 && haveDescs)
                    mov(1, a0.ud(0), a0.ud(3));

                // Load C data.
                auto &sublayoutLoad = (loadOnly && !copyC) ? sublayoutAcc : sublayoutExt;
                auto &C_loadRange   = (loadOnly && !copyC) ? C_accRange   : C_extRange;
#ifdef STANDALONE
                if (!problem.mock.cLoad)
#endif
                    loadMatrix(C_loadRange, sublayoutLoad, C_addrs[0], strategy, state);

                // Set up a0.0 descriptor for stores (and save load descriptors) if needed.
                if (haveDescs && !loadOnly) {
                    if (lend < nblocks)
                        mov(1, a0.ud(3), a0.ud(0));
                    mov(1, a0.ud(0), a0.ud(2));
                }

                // Copy loaded data as needed.
                if (copyCLoad || (loadOnly && copyC)) {
                    auto &sublayoutDst = loadOnly ? sublayoutAcc : sublayoutCopy;
                    auto &C_dstRange   = loadOnly ? C_accRange   : C_copyRange;
                    Tload = lateCConvert ? Ts : state.Tacc;
                    copyRegisters(Tc_ext, Tload, sublayoutExt, sublayoutDst, C_extRange, C_dstRange, strategy, state);
                }
            }

            // Late C conversion.
            auto originalTacc = state.Tacc;
            if (lateCConvert) {
                for (int li = listart; li < liend; li++) {
                    auto C_acc = subrange(state.C_regs[0], hw, state.Tacc, layout[li]);
                    copyRegisterBlock(state.Tacc, Ts, layout[li], layout[li], C_acc, C_convertRegs[li - listart], 0, 0, strategy, state);
                }
                state.setTacc(Ts);
            }

            // Alpha/beta scaling, optional fp32<->int32 conversion, and masking.
            bool remaskC_M = isPacked(problem.C.layout) && (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
            bool remaskC_N = isPacked(problem.C.layout) && (strategy.remHandling[LoopN] != RemainderHandling::Ignore);

            bool skipUpdate = problem.alpha1() && beta0 && state.Tacc == Tacc_final && !remaskC_M && !remaskC_N;
            skipUpdate &= !useEltwiseInjector(problem);

            if (!loadOnly && !skipUpdate) for (int phase = 0; phase < 3; phase++) {
                vector<GRFMultirange> C_accs, C_accSwaps, C_loads;
                C_accs.reserve(liend - listart);
                C_accSwaps.reserve(liend - listart);
                C_loads.reserve(liend - listart);

                for (int li = listart; li < liend; li++) {
                    GRFMultirange C_acc0 = subrange(state.C_regs[0], hw, state.Tacc, layout[li]);
                    GRFMultirange C_acc = lateCConvert ? C_convertRegs[li - listart]
                                                       : C_acc0;
                    GRFMultirange C_accSwap;
                    GRFMultirange C_load = beta0 ? C_acc :
                                       copyCLoad ? C_copyRegs[li - listart] :
                                                   C_extRegs[li - listart];
                    switch (phase) {
                        case 0:
                            if (!beta0) convert(C_load, Tload, state.Tacc, strategy, state);
                            break;
                        case 1:
                            {
                                C_accs.push_back(std::move(C_acc));
                                C_accSwaps.push_back(std::move(C_accSwap));
                                C_loads.push_back(std::move(C_load));
                            }
                            break;
                        case 2:
                            if (lateCConvert)
                                copyRegisterBlock(state.Tacc, Tacc_final, layout[li], layout[li], C_acc, C_acc0, 0, 0, strategy, state);
                            else
                                convert(C_acc, state.Tacc, Tacc_final, strategy, state);
                            break;
                    }
                }

                if (phase == 1) {
                    std::vector<int> order(liend - listart);
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(), [&](int a, int b) {
                        auto *rangeA = &C_accs[a], *rangeB = &C_accs[b];
                        if (rangeA->empty() || rangeB->empty()) return false;
                        return (*rangeA)[0].getBase() < (*rangeB)[0].getBase();
                    });
                    GRFMultirange C_accsSorted, C_accSwapsSorted, C_loadsSorted;
                    vector<RegisterBlock> C_accSortedList;

                    for (int i = 0; i < (liend - listart); i++) {
                        if (remaskC_M || remaskC_N) {
                            auto block = layout[listart + order[i]];
                            block.offsetBytes = C_accsSorted.getLen() << GRF::log2Bytes(hw);
                            C_accSortedList.push_back(block);
                        }

                        C_accsSorted.append(C_accs[order[i]]);
                        C_accSwapsSorted.append(C_accSwaps[order[i]]);
                        C_loadsSorted.append(C_loads[order[i]]);
                    }

                        updateC(C_accsSorted, C_accSwapsSorted, C_loadsSorted, problem, strategy, state);

                    if (remaskC_M || remaskC_N) {
                        RegisterLayout C_accSortedLayout(state.Tacc, state.C_layout.addressing(), state.C_layout.addressingStrategy(), std::move(C_accSortedList));
                        if (remaskC_M) remaskLayout(0, false, C_accSortedLayout, C_accsSorted, strategy, state);
                        if (remaskC_N) remaskLayout(1, true,  C_accSortedLayout, C_accsSorted, strategy, state);
                    }
                }
            }

            state.setTacc(Tacc_final);

            // Store updated data.
            if (op == COperation::UpdateStore) {
                bool needsFullK = false;

                if (copyC)
                    copyRegisters(sublayoutAcc, sublayoutExt, C_accRange, C_extRange, strategy, state);

                auto &sublayoutSrc = copyC || needsFullK ? sublayoutExt : sublayoutAcc;
                auto &C_srcRange = copyC || needsFullK ? C_extRange : C_accRange;

#ifdef STANDALONE
                if (!problem.mock.cStore) {
#endif
                FOR_EACH_C storeMatrix(C_srcRange, sublayoutSrc, C_addrs[q], strategy, state);
#ifdef STANDALONE
                }
#endif
            }

            state.setTacc(originalTacc);
        }

        // Free address and data registers, including C accumulators that are no longer used...
        //  ... except C_addr0. I need that!
        FOR_EACH_C safeReleaseRanges(C_addrs[q], state);
        safeReleaseRanges(C_extRange, state);
        safeReleaseRanges(C_copyRange, state);
        safeReleaseRanges(C_convertRegs, state);
        bool releaseC = (op == COperation::UpdateStore);
        for (int li = listart; li < liend; li++)
            releaseC &= layout[li].grfAligned();
        if (layout.back().offsetBytes == 0)
            releaseC = false;
        if (releaseC) for (int li = listart; li < liend; li++)
            for (int b = 0; b < state.C_buffers; b++)
                releaseRanges(subrange(state.C_regs[b], hw, state.Tacc, layout[li]), state);
        FOR_EACH_C state.ra.claim(C_addr0[q]);

        // Check for forward progress.
        if (lend == lstart)
            throw out_of_registers_exception();
        lstart = lend;
    }

    // Re-claim all the C registers we freed, so as not to disturb the caller's RegisterAllocator.
    for (int b = 0; b < state.C_buffers; b++)
        reclaimRanges(state.C_regs[b], state);
#undef FOR_EACH_C
}

// Output code for standard C remainder handling.
template <HW hw>
bool Generator<hw>::doStdCRemainder(RegisterLayout &layoutExt, RegisterLayout &layoutExtUnmasked,
                                    bool inside, bool columns[2], StdCRemType remTypes[2], bool fragments[2], bool fragPositives[2], int fragSizes[2],
                                    const RegisterRange (&C_addr0)[2], const RegisterRange (&C_addr0Unmasked)[2], COperation op, vector<MaskAssignment> &masks,
                                    const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState state,
                                    RegisterBlock *C_block0, RegisterBlock *C_blockUnmasked0)
{
    auto Tc_ext = problem.Tc_ext;
    auto column = columns[inside];
    LoopType loop = column ? LoopN : LoopM;
    auto remType = remTypes[loop];
    auto fragment = fragments[loop];
    auto fragPositive = fragPositives[loop];
    auto fragSize = fragSizes[loop];
    auto unroll = strategy.unroll[loop];
    auto remainder = state.remainders[loop];

    if (!C_block0) C_block0 = &layoutExt[0];
    if (!C_blockUnmasked0 && !layoutExtUnmasked.empty()) C_blockUnmasked0 = &layoutExtUnmasked[0];

    bool canEOT = !state.isNested && (op == COperation::UpdateStore);
    bool atomicUpdate = strategy.C.atomic && (op == COperation::UpdateStore);

    Label lEnd;

    // The "q" dimension is the one whose remainder we are currently handling.
    auto RegisterBlock::*nq      = column ? &RegisterBlock::nc      : &RegisterBlock::nr;
    auto RegisterBlock::*offsetQ = column ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    // Status message.
    status << "C remainder handling (" << char('m' + column) << ") " << remType;
    if (fragment)     status << ", fragment";
    if (fragPositive) status << ", no empty accesses";
    status << status_stream::endl;

    // Allocate temporaries for emulated atomic addition if needed.
    if (!inside && atomicUpdate) allocEAtomicAddRegs(hw, Tc_ext, layoutExt, problem.C, strategy.C, state);

    // Handle a subproblem. Return true if successful.
    auto descend = [&](RegisterLayout &sublayoutExt, RegisterLayout &sublayoutExtUnmasked, bool full = false) -> bool {
        bool success = true;
        auto nMasksOriginal = int(masks.size());

        if (remType == StdCRemType::Mask) {
            if (!full) {
                // Assign and load any extra masks needed.
                if (!assignMasks(sublayoutExt, LoopM, LoopN, masks, strategy, state))
                    return false;
                loadMasks(masks, state.remainders, strategy, state, nMasksOriginal);
                sublayoutExtUnmasked.clear();
            } else {
                // Clear out mask assignments in this dimension.
                for (auto &block: layoutExt)
                    block.clearFlag();
            }
        }

        // Recursively handle subproblem.
        if (!inside)
            success = doStdCRemainder(sublayoutExt, sublayoutExtUnmasked, true, columns, remTypes, fragments, fragPositives, fragSizes, C_addr0, C_addr0Unmasked, op, masks, problem, strategy, state, C_block0, C_blockUnmasked0);
        else if (sublayoutExtUnmasked.empty())
            updateCLayout(sublayoutExt, C_addr0, *C_block0, op, problem, strategy, state);
        else
            updateCLayout(sublayoutExtUnmasked, C_addr0Unmasked, *C_blockUnmasked0, op, problem, strategy, state);

        // Free any new masks.
        if (remType == StdCRemType::Mask)
            safeReleaseMaskAssignments(masks, state, nMasksOriginal);
        return success;
    };

    // Exit remainder handling.
    auto done = [&]() {
        if (!canEOT)
            jmpi(1, lEnd);
        else
            epilogue(strategy, state);
    };

    // Main code.
    bool success = false;
    pushStream();

    if (!fragment) {
        // If descriptor-based remainders requested, all blocks should be smaller than fragSize.
        // Load descriptors based on total remainder in this (rare) case.
        if (remType == StdCRemType::Descriptor) {
            loadLoadStoreDescriptors(!problem.beta0(), true, layoutExt[0], remainder, problem.C, strategy.C, strategy, state);
            if (!layoutExt.assignAllDescs())
                goto failed;
        }
        if (remType != StdCRemType::Ignore && inside && !layoutExtUnmasked.empty() && layoutExt.blocks() == state.C_layoutExt.blocks()) {
            // If unmasked layout is available, implement full remainder case specially.
            const bool useSIMTFlow = strategy.fused && (strategy.fusedLoop == loop || strategy.fusedLoop == LoopAny);
            Label labelRem, labelDone;

            if (useSIMTFlow) {
                cmp(16 | ge | state.flagAP, remainder, unroll);
                if_(16 | state.flagAP, labelRem, labelDone);
            } else if (strategy.fused) {
                cmp(1 | ge | state.flagAP, remainder, unroll);
                jmpi(1 | ~state.flagAP, labelRem);
            } else {
                // No flag registers guaranteed -- use a jump table.
                auto tempQ = state.ra.alloc_sub<uint64_t>();
                auto temp = tempQ.ud(0);

                add(1 | sat, temp, remainder, -unroll + 1);
                isGen12 ? mad(1, temp, 16, temp, 16)
                        : shl(1, temp, temp, 4);
                jmpi(1, temp.d());
                jmpi(1, labelRem);

                state.ra.safeRelease(tempQ);
            }

            status << "Code for full " << char('m' + column) << " remainder" << status_stream::endl;
            if (!descend(layoutExt, layoutExtUnmasked, true))
                goto failed;

            useSIMTFlow ? else_(16, labelDone)
                        : jmpi(1, labelDone);
            mark(labelRem);

            status << "Code for generic " << char('m' + column) << " remainder" << status_stream::endl;
            if (!descend(layoutExt, layoutExtUnmasked))
                goto failed;

            mark(labelDone);
            if (useSIMTFlow)
                endif(16);
        } else {
            // Otherwise, nothing else to do: go down a level.
            if (!descend(layoutExt, layoutExtUnmasked))
                goto failed;
        }
    } else {
        // Use SIMT control flow if remainders could be different between fused threads or if jump tables disabled.
        const bool useSIMTFlow = strategy.noJumpTables || (strategy.fused && (strategy.fusedLoop == loop || strategy.fusedLoop == LoopAny));

        // Fix up fragment size (fragSize).
        //  - Check that every block starts at a multiple of fragSize; if not fall back on fragSize 1.
        //  - Max fragment size is 16.
        //  - Should check unmasked layout, but it will have the same kind of fragmenting as the masked layout.
        fragSize = std::min<int>(fragSize, 16);
        for (auto &block : layoutExt) {
            if (block.*offsetQ % fragSize) {
                fragSize = 1;
                break;
            }
        }

        auto doSliceExt = [&](bool column, int q0, int q1,
                              RegisterLayout &C_subblocksExt, RegisterLayout &C_subblocksExtUnmasked) {
            C_subblocksExt = layoutExt.trySlice(column, q0, q1, false);
            if (!C_subblocksExt.valid()) return false;

            if (!layoutExtUnmasked.empty()) {
                C_subblocksExtUnmasked = layoutExtUnmasked.trySlice(column, q0, q1, false);
                if (!C_subblocksExtUnmasked.valid()) return false;
            }

            return true;
        };

        // There are two strategies for fragmenting for remainder handling:
        //    fragSize = 1:  Try to get the largest blocks as possible. These are always fragPositive.
        //    fragSize > 1:  Always use blocks of size fragSize in the q dimension.
        if (fragSize == 1) {
            if (!useSIMTFlow) {
                // SIMD control flow, using a jump table.
                Subregister temp = state.ra.alloc_sub<uint32_t>();
                vector<Label> rlabels(unroll);

                // Generate jump table.
                shl(1, temp, remainder, uint16_t(4));     // Multiply by instruction length.
                if (isGen12)                              // Gen12+ jmpi is relative to current IP.
                    add(1, temp, temp, uint16_t(16));
                jmpi(1, temp.d());                        // Indexed jump into jump table.
                for (int r = 0; r < unroll; r++)
                    jmpi(1, rlabels[r]);

                // Full remainder case: continue downward.
                status << "Code for full " << char('m' + column) << " remainder" << status_stream::endl;
                if (!descend(layoutExt, layoutExtUnmasked, true))
                    goto failed;
                inside ? jmpi(1, rlabels[0])
                       : done();

                // Remainder handling.
                vector<bool> qdone(unroll, false);
                qdone[0] = true;
                int qnext = 0;
                for (int nqtodo = unroll - 2; nqtodo >= 0; nqtodo--) {
                    // Decide which q to do.
                    int q;
                    if (qnext > 0)
                        q = qnext;
                    else {
                        for (q = unroll - 1; q >= 0; q--)
                            if (!qdone[q])
                                break;
                    }

                    status << "Code for " << char('m' + column) << " remainder " << q << status_stream::endl;

                    mark(rlabels[q]);

                    // Figure out how many rows/columns to take.
                    int chunkSize = q & ~(q - 1);               // = 1 << lowest set bit

                    // Look through all blocks in this row/column, and reduce chunk size if appropriate.
                    for (auto &block : layoutExt) {
                        if (!block.isLoadBlock()) stub();       // Dummy blocks should be replaced by real ones...
                        int qq = q - block.*offsetQ;            // Note q = 1 + last row/column.
                        if (qq > 0 && qq <= block.*nq)
                            chunkSize = std::min<int>(chunkSize, qq);
                    }

                    // With chunk size chosen, get rows/columns [q - chunkSize, q) of intersecting blocks.
                    RegisterLayout C_subblocksExt, C_subblocksExtUnmasked;
                    if (!doSliceExt(column, q - chunkSize, q, C_subblocksExt, C_subblocksExtUnmasked))
                        goto failed;

                    // Perform the requested update.
                    if (!descend(C_subblocksExt, C_subblocksExtUnmasked))
                        goto failed;

                    // Go to next remainder handler, or return.
                    qdone[q] = true;
                    qnext = q - chunkSize;
                    if (nqtodo > 0) {
                        if (qnext == 0 && canEOT)
                            epilogue(strategy, state);
                        else if (qdone[qnext]) {
                            jmpi(1, rlabels[qnext]);
                            qnext = 0;
                        }
                    }
                }
                mark(rlabels[0]);

                state.ra.safeRelease(temp);
            } else {
                // SIMT control flow: massively nested if-else.

                // Handle remainder in the range [q0, q1).
                std::function<bool(int, int)> handleRemainder = [&](int q0, int q1) -> bool {
                    Label labelElse, labelEndif;

                    int qChunk = rounddown_pow2(q1 - q0 - 1);

                    if (qChunk == 0)
                        qChunk = 1;

                    status << "Code for " << char('m' + column) << " remainders " << q0 << " - " << (q1 - 1) << status_stream::endl;

                    if (q1 - q0 > 1) {
                        cmp(16 | ge | state.flagAP, remainder, uint16_t(q0 + qChunk));
                        if_(16 | state.flagAP, (qChunk > 1) ? labelElse : labelEndif, labelEndif);
                    }

                    RegisterLayout C_subblocksExt, C_subblocksExtUnmasked;
                    if (!doSliceExt(column, q0, q0 + qChunk, C_subblocksExt, C_subblocksExtUnmasked)) return false;
                    if (!descend(C_subblocksExt, C_subblocksExtUnmasked))                             return false;

                    if (q1 - q0 > 1) {
                        if (qChunk > 1) {
                            if (!handleRemainder(q0 + qChunk, q1))
                                return false;

                            else_(16, labelEndif);
                            mark(labelElse);

                            if (!handleRemainder(q0, q0 + qChunk))
                                return false;
                        }

                        mark(labelEndif);
                        endif(16);
                    }

                    return true;
                };

                Label labelRem, labelRemDone, labelDone;

                cmp(16 | ge | state.flagAP, remainder, uint16_t(unroll));
                if_(16 | state.flagAP, labelRem, labelDone);

                status << "Code for " << char('m' + column) << " full remainder" << status_stream::endl;
                if (!descend(layoutExt, layoutExtUnmasked, true))
                    goto failed;

                else_(16, labelDone);
                mark(labelRem);

                if (!handleRemainder(0, unroll))
                    goto failed;

                mark(labelDone);
                endif(16);
                setDefaultNoMask(true);
            }
        } else {
            auto handleRemainderFP = [&](int q0, int q1) -> bool {
                // Get rows/columns [q0, q1) of intersecting blocks.
                RegisterLayout C_subblocksExt, C_subblocksExtUnmasked;
                if (!doSliceExt(column, q0, q1, C_subblocksExt, C_subblocksExtUnmasked))
                    return false;

                if (remType == StdCRemType::Descriptor) {
                    // Load address registers for subsequent loads and stores.
                    Subregister rcount = state.ra.alloc_sub<uint32_t>();
                    Subregister mremainder = remainder;

                    if (q0 != 0) {
                        add(1 | sat, rcount, mremainder, int16_t(-q0));
                        mremainder = rcount;
                    }
                    if (q1 < unroll) {
                        min_(1, rcount, mremainder, uint16_t(fragSize));
                        mremainder = rcount;
                    }

                    loadLoadStoreDescriptors(!problem.beta0(), true, C_subblocksExt[0], mremainder, problem.C, strategy.C, strategy, state);
                    if (!C_subblocksExt.assignAllDescs() || !C_subblocksExtUnmasked.assignAllDescs())
                        return false;

                    state.ra.safeRelease(rcount);
                }

                // Perform the requested update.
                return descend(C_subblocksExt, C_subblocksExtUnmasked);
            };

            if (!useSIMTFlow) {
                // SIMD control flow, possibly using a jump table.
                int N = div_up(unroll, fragSize);
                vector<Label> rlabels(N);      // Targets for jump table.
                Label rdone;

                // Create a jump table, if needed.
                if (fragPositive) {
                    Subregister t1 = state.ra.alloc_sub<uint32_t>();
                    Subregister t2 = state.ra.alloc_sub<uint32_t>();

                    add(1 | sat, t2, remainder, int16_t(-unroll + 1));
                    add(1, t1, remainder, int16_t(-1 + (isGen12 ? fragSize : 0)));
                    add(1, t1, t1, t2);                         // Increment index if remainder == unroll.
                    if (fragSize < 16)                          // Precondition: fragSize <= 16.
                        mulConstant(1, t1, t1, 16 / fragSize);  // Multiply by instruction length (16b/uncompacted instruction)
                    and_(1, t1, t1, uint16_t(0xFFF0));          // Mask off unwanted bits.
                    jmpi(1, t1.d());                            // Indexed jump into jump table.
                    for (int r = 0; r < N; r++)
                        jmpi(1, rlabels[r]);

                    state.ra.safeRelease(t2);
                    state.ra.safeRelease(t1);
                }

                // Full loop.
                status << "Code for " << char('m' + column) << " full remainder" << status_stream::endl;
                if (!descend(layoutExt, layoutExtUnmasked, true))
                    goto failed;
                inside ? jmpi(1, rdone)
                       : done();

                // Remainder handling.
                for (int r = N - 1; r >= 0; r--) {
                    int q0 = r * fragSize;
                    int q1 = std::min<int>(q0 + fragSize, unroll);

                    status << "Code for " << char('m' + column) << " remainders " << q0 + 1 << " - " << q1 << status_stream::endl;

                    mark(rlabels[r]);

                    if (!handleRemainderFP(q0, q1))
                        goto failed;
                }

                if (inside) mark(rdone);
            } else {
                // SIMT control flow version.
                Label labelRem, labelRemDone, labelDone;

                cmp(16 | ge | state.flagAP, remainder, uint16_t(unroll));
                if_(16 | state.flagAP, labelRem, labelDone);

                status << "Code for " << char('m' + column) << " full remainder" << status_stream::endl;
                if (!descend(layoutExt, layoutExtUnmasked, true))
                    goto failed;

                else_(16, labelDone);
                mark(labelRem);

                for (int q0 = 0; q0 < unroll; q0 += fragSize) {
                    int q1 = std::min<int>(q0 + fragSize, unroll);

                    cmp(16 | le | state.flagAP, remainder, uint16_t(q0));
                    goto12(16 | state.flagAP, labelRemDone);
                    status << "Code for " << char('m' + column) << " remainders " << q0 + 1 << " - " << q1 << status_stream::endl;

                    if (!handleRemainderFP(q0, q1))
                        goto failed;
                }

                mark(labelRemDone);
                join(16);

                mark(labelDone);
                endif(16);
            }
        }
    }

    // Success!
    success = true;
failed:

    mark(lEnd);
    success ? appendCurrentStream() : discardStream();

    if (!inside && atomicUpdate) freeEAtomicAddRegs(state);

    return success;
}

// Alternate code path for C remainder handling, based on a simple double loop
//  and indirect addressing.
template <HW hw>
void Generator<hw>::doAlternateCRemainder(COperation op, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc = problem.Tc, Tc_ext = problem.Tc_ext;
    int C_count = (op == COperation::UpdateStore) ? state.C_count : 1;
#define FOR_EACH_C     for (int q = 0; q < C_count; q++)
#define FOR_EACH_C_REV for (int q = C_count - 1; q >= 0; q--)

    bool lateYLoopCheck = false;
    bool atomicUpdate = strategy.C.atomic && (op == COperation::UpdateStore);
    bool atomicLoad   = strategy.C.atomic && !atomicUpdate;

    bool surface = !strategy.C.base.isStateless();
    bool loadOnly = (op == COperation::Load);

    bool fbfEmulate = (!strategy.systolicAvailable && op == COperation::UpdateStore
            && state.Tacc.real() == Type::f32 && Tc_ext.real() == Type::bf16);

    // Vector length in inner loop.
    const auto nbytes = 64;
    auto nec = std::min(nbytes / Tc, nbytes / Tc_ext);

    // 1- and 2-byte types must be padded to 4 bytes.
    bool byte_access = (Tc_ext.size() < 4);
    if (byte_access)
        nec = nbytes >> 2;

    // 8-byte+ types can use scattered qword. Only atomic for now.
    bool nativeAtomic = atomicUpdate && hasNativeAtomicAdd(hw, Tc_ext.real(), problem.C, strategy.C);
    bool qword = false;
    int rshift = qword ? 3 : 2;     // log2(data stride in regs)
    int rsimd;
    rsimd = 64 >> rshift;

#ifdef STANDALONE
    status << "Generic C remainder handling (m/n)" << status_stream::endl;
#endif

    auto &block0 = state.C_layout[0];
    bool cColMajorMem = isColMajor(problem.C.layout);
    bool cColMajorReg = block0.colMajor;
    bool transpose = (cColMajorReg != cColMajorMem);
    if (isPacked(problem.C.layout)) stub();

    // x is the contiguous dimension (in registers), y is the other dimension.
    auto LoopX = cColMajorReg ? LoopM : LoopN;
    auto LoopY = cColMajorReg ? LoopN : LoopM;
    int unrollX = strategy.unroll[LoopX];
    int unrollY = strategy.unroll[LoopY];

    // Check the layout:
    //  - C is a contiguous block of registers.
    //  - nx must be divisible by 2 (unpacked) GRFs, unless x unroll is < 2 GRFs,
    //      or there's an extra GRF at the end of C.
    //  - register offsets must be in a uniform 2D grid
    //  - all blocks must share same ordering (row/column major).
    // Otherwise use non-uniform path, and indirectly load GRFs.

    auto Tcx = Tc, Tcy = Tc;
    bool uniform = true;
    int16_t xByteInc = 0, yByteInc = 0;
    bool cAtEnd = (state.C_regs[0][state.C_regs[0].getLen() - 1].getBase() + 1) >= strategy.GRFs;

    if (state.C_regs[0].ranges.size() != 1)
        uniform = false;

    for (int i = 0; i < state.C_layout.blocks(); i++) {
        auto &block = state.C_layout[i];
        if (block.colMajor != block0.colMajor) stub();

        int nx = cColMajorReg ? block.nr : block.nc;
        int ny = cColMajorReg ? block.nc : block.nr;
        int ox = cColMajorReg ? block.offsetR : block.offsetC;
        int oy = cColMajorReg ? block.offsetC : block.offsetR;

        ox /= nec;

        if ((nx & (nec - 1)) && cAtEnd)
            uniform = false;

        if (xByteInc == 0 && nx > nec) xByteInc = nec * Tcx;
        if (yByteInc == 0 && ny > 1)   yByteInc = block.ld * Tcy;

        if (block.offsetBytes != ox * xByteInc + oy * yByteInc) {
            if (xByteInc == 0 && ox > 0)
                xByteInc = (block.offsetBytes - oy * yByteInc) / ox;
            else if (yByteInc == 0 && oy > 0)
                yByteInc = (block.offsetBytes - ox * xByteInc) / oy;
            else
                uniform = false;
        }

    }

    RegisterRange bases;
    bool nonuniformSubs = false;

    if (!uniform) {
        int maxGRFs = GRF::maxRegs(hw);
        std::vector<uint8_t> baseIndices(maxGRFs, 0);
        std::vector<uint16_t> offIndices(maxGRFs, 0);

        // Workaround for spurious maybe-uninitialized warning in GCC11
        for (int i = 0; i < maxGRFs; i++) offIndices[i] = 0;

        if (state.Tacc.size() == 1) stub();

        xByteInc = div_up(nec * Tcx, GRF::bytes(hw));
        int nec1 = nec / xByteInc;
        yByteInc = div_up(unrollX, nec1);

        int ncx = 1;

        int idx = 0;
        for (int y = 0; y < unrollY; y++) {
            for (int xx = 0; xx < yByteInc; xx++) {
                for (int cxComponent = 0; cxComponent < ncx; cxComponent++, idx++) {
                    auto x = xx * nec1;
                    auto i = cColMajorReg ? x : y;
                    auto j = cColMajorReg ? y : x;
                    int ne;
                    auto sub = state.C_layout.find(i, j, state.C_regs[0], &ne, nullptr, cxComponent);
                    nonuniformSubs |= (sub.getOffset() != 0);
                    if (ne < std::min(nec1, unrollX - x)) stub();
                    baseIndices[idx] = sub.getBase();
                    offIndices[idx] = sub.getByteOffset() + sub.getBase() * GRF::bytes(hw);
                }
            }
        }

        if (nonuniformSubs) xByteInc *= 2, yByteInc *= 2;

        bases = state.ra.alloc_range(div_up(unrollY * yByteInc, GRF::bytes(hw)));
        bool haveDF = !strategy.emulate.emulate64;
        haveDF |= (hw >= HW::XeHPC);
        if (haveDF) {
            for (int i = 0; i < unrollY * yByteInc; i += 8) {
                auto sub = bases[i / GRF::bytes(hw)].df((i % GRF::bytes(hw)) / 8);
                auto data = nonuniformSubs ? reinterpret_cast<double *>(&offIndices[i / 2])
                                           : reinterpret_cast<double *>(&baseIndices[i]);
                mov(1, sub, *data);
            }
        } else {
            for (int i = 0; i < unrollY * yByteInc; i += 4) {
                auto sub = bases[i / GRF::bytes(hw)].ud((i % GRF::bytes(hw)) / 4);
                auto data = nonuniformSubs ? reinterpret_cast<uint32_t *>(&offIndices[i / 2])
                                           : reinterpret_cast<uint32_t *>(&baseIndices[i]);
                mov(1, sub, *data);
            }
        }
    }

    // Claim flags.
    auto saveFlagAP = state.flagAP;
    state.raVFlag.safeRelease(state.flagAP);
    state.raVFlag.claim(f0[0]);
    state.raVFlag.claim(f0[1]);
    state.raVFlag.claim(f1[0]);
    if (fbfEmulate) {
        state.raVFlag.claim(f1[1]);
        state.emulate.flag = f1[1];
        state.emulate.flagOffset = 0;
    }

    // Clear f0[1] for any16h trick.
    if (strategy.fused && !lateYLoopCheck) {
        mov(1, f0[1], uint16_t(0));
    }

    // Update C with scattered accesses.
    // Get mask and set up header.
    RegisterRange header[2];
    auto hregs = (surface ? 1 : 2) * (qword ? 1 : 2);
    FOR_EACH_C header[q] = state.ra.alloc_range(hregs);
    Subregister temp = state.ra.alloc_sub<uint32_t>();
    Subregister mask = state.ra.alloc_sub<uint32_t>();
    Subregister xIndex = state.remainders[LoopX];

    GRF indexVec, ivContig, ivScatter;

    indexVec = state.ra.alloc();
    indexVec.setType(DataType::w);
    mov(8, indexVec[0](1), Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
    if (rsimd > 8) mov(8, indexVec[8](1), Immediate::uv(8, 9, 10, 11, 12, 13, 14, 15));

    auto oshift = std::min<int>(rshift, Tc_ext.log2Size());

    // Prepare x mask in f1.0 and prepare header for loads/stores.
    if (Tc_ext.size() > 4) {
        mulConstant(1, temp, xIndex, uint16_t(Tc_ext.size() >> rshift));
        xIndex = temp;
    }

    ivScatter = indexVec;
    bool splitScatter = transpose && (Tc_ext.log2Size() > rshift);
    if (splitScatter) {
        ivContig = state.ra.alloc();
        ivContig.setType(DataType::w);
        auto shift = Tc_ext.log2Size() - rshift;
        auto m = (1 << shift) - 1;

        asr(16, ivScatter, indexVec, uint16_t(shift));
        mov(16, ivContig, Immediate::uv((0&m)<<rshift, (1&m)<<rshift, (2&m)<<rshift, (3&m)<<rshift, (4&m)<<rshift, (5&m)<<rshift, (6&m)<<rshift, (7&m)<<rshift));

    }

    add(1, temp, xIndex, int16_t(-1));          FOR_EACH_C  transpose ? mul(rsimd, header[q][0].d(), state.inputs.ldc[q], ivScatter)
                                                                      : shl(rsimd, header[q][0].d(), indexVec, uint16_t(oshift));
                                                FOR_EACH_C  if (splitScatter) add(rsimd, header[q][0].d(), header[q][0].d(), ivContig);

    int hs = 1;
    bool header4 = !qword && !surface;
    int neq = elementsPerGRF(hw, DataType::uq);

    header4 &= (GRF::bytes(hw) < 64);
    if (hw >= HW::XeHP && !surface) {
        if (header4)
            FOR_EACH_C mov<uint32_t>(2 * neq, header[q][2][0](2), header[q][1]);
        FOR_EACH_C mov<uint32_t>(neq, header[q][1][0](2), header[q][0][neq](1));
        FOR_EACH_C mov<uint32_t>(neq, header[q][0][0](2), header[q][0][0](1));
        hs = 2;
    }

    and_(1, temp, ~temp, uint16_t(rsimd - 1));  FOR_EACH_C  surface ? add(rsimd, header[q][0].d(), header[q][0].d(), state.effC[q]) :
                                                            header4 ? eadd(8, header[q][2].uq(), header[q][hs].d(0)(hs), state.effC[q], strategy, state)
                                                                    : noop();
    mov(1, mask, uint16_t((1<<rsimd) - 1));     FOR_EACH_C  if (!surface) eadd(2 * neq, header[q][0].uq(), header[q][0].d(0)(hs), state.effC[q], strategy, state);
    shr(1, f1[0], mask, temp);

    state.ra.safeRelease(mask);
    state.ra.safeRelease(temp);
    state.ra.safeRelease(ivContig);

    // Synthesize double loop updating 2 GRFs (indirectly addressed) at a time.
    GRF ix = state.ra.alloc();
    Subregister ix_init = state.ra.alloc_sub<uint16_t>();
    Subregister iy = state.ra.alloc_sub<int16_t>();
    Subregister cXInc[2], cYInc[2];
    FOR_EACH_C cYInc[q] = state.ra.alloc_sub<int32_t>();
    Label yLoop, xLoop;
    RegisterRange Cacc = state.ra.alloc_range(2);
    RegisterRange CaccSwap{};
    RegisterRange Cload = state.ra.alloc_range(2, getHint(HintType::CLoad, strategy));

    if (transpose) FOR_EACH_C {
        cXInc[q] = state.ra.alloc_sub<int32_t>();
        mulConstant(1, cXInc[q], state.inputs.ldc[q], nec);
    }

    bool remX = (strategy.remHandling[LoopX] != RemainderHandling::Ignore);
    bool remY = (strategy.remHandling[LoopY] != RemainderHandling::Ignore);

    remX ? add(1, ix_init, state.remainders[LoopX], int16_t(-1))
         : mov(1, ix_init, strategy.unroll[LoopX] - 1);
    remY ? mov(1, iy, state.remainders[LoopY])
         : mov(1, iy, strategy.unroll[LoopY]);
    shr(1, ix_init, ix_init, uint16_t(ilog2(nec)));

    if (uniform)
        mov(1, a0[0], state.C_regs[0][0].getBase() * GRF::bytes(hw));
    else
        mov(1, a0[0], bases.getBase() * GRF::bytes(hw));

    add(1, cYInc[0], ix_init, uint16_t(1));
    mulConstant(1, cYInc[0], cYInc[0], uint16_t(nec * (!transpose ?  Tc_ext.size() : 1)));
    if (!transpose) {
        FOR_EACH_C_REV add(1, cYInc[q], -cYInc[0], state.inputs.ldc[q]);
    } else {
        FOR_EACH_C_REV mul(1, cYInc[q], state.inputs.ldc[q], cYInc[0].w());
        FOR_EACH_C_REV add(1, cYInc[q], -cYInc[q], uint16_t(Tc_ext.size()));
    }

    wrdepRanges(state.C_regs[0]);

    InstructionModifier mod;
    mark(yLoop);
    mov<uint16_t>(16, ix, ix_init);
    if (!lateYLoopCheck)
        add(1 | gt | f0[1], iy, iy, int16_t(-1));
    mov(1, a0[1], a0[0]);

    mark(xLoop);
    add<int16_t>(16 | ge | f0[0], ix, ix, int16_t(-1));

    // Update. The anyv is a trick to use the generated m mask (f1.0) on the last
    //  iteration of the loop, and no mask (0xFFFF) on the other iterations.
    mod = mod | f0[0] | anyv;

    // Alas, no anyv on PVC.
    if (hw >= HW::XeHPC) {
        mov(1 | ~f0[0], f0[0], f1[0]);
        mod = InstructionModifier() | f0[0];
    }

    if (!uniform) {
        subdep(Operand::src0, GRFRange(bases.getBase(), bases.getLen()));
        nonuniformSubs ? mov(xByteInc >> 1, a0[2](1), indirect[a0[1]].uw())
                       : shl(xByteInc,      a0[2](1), indirect[a0[1]].ub(), GRF::log2Bytes(hw));
    }

#define IGNORE_SWSB ignoredep(Operand::src0);

    if (!loadOnly) {
        if (uniform) switch (state.Tacc.size()) {
            case 1:  IGNORE_SWSB mov<uint32_t>(16, Cacc, indirect[a0[1]].ub()); break;
            case 2:  IGNORE_SWSB mov<uint32_t>(16, Cacc, indirect[a0[1]].uw()); break;
            default:
                IGNORE_SWSB mov<uint32_t>(16, Cacc, indirect[a0[1]]); break;
        } else if (xByteInc == 1) switch (state.Tacc.size()) {
            case 2:  IGNORE_SWSB mov<uint32_t>(16, Cacc, indirect[a0[2]].uw()); break;
            default: IGNORE_SWSB mov<uint32_t>(16, Cacc, indirect[a0[2]]);      break;
        } else switch (state.Tacc.size()) {
            case 2:  IGNORE_SWSB mov<uint32_t>(16, Cacc, indirect[a0[2]].uw(0)(16 / xByteInc, 1)); break;
            default: IGNORE_SWSB mov<uint32_t>(16, Cacc, indirect[a0[2]].ud(0)(16 / xByteInc, 1)); break;
        }
    }

#undef IGNORE_SWSB

    if (atomicUpdate) {
        // Atomic update. Requires beta = 0/1, alpha prescaled.
        if (!problem.alpha1() || !problem.beta1()) stub();
        if (C_count > 1) stub();
        if (op != COperation::UpdateStore) stub();

        convert(GRFRange(Cacc.getBase(), Cacc.getLen()), state.Tacc, Tc_ext, strategy, state);

        RegisterLayout layout(hw, Tc_ext, 1, 1, problem.C, strategy.C);

        auto &block = layout[0];
        block.ebytes = qword ? 8 : Tc_ext.real().size();
        block.simdSize = rsimd;
        block.bytes = 64;
        block.extra = 1;
        block.count = 1;

        GRFMultirange saveVFlags;
        std::swap(saveVFlags, state.vflagStorage);  /* temporarily disable vflags */

        Label labelEndAtomic;
        if (!nativeAtomic)
            allocEAtomicAddRegs(hw, Tc_ext, layout, problem.C, strategy.C, state, f1[1]);

        bool branchAtomic = !nativeAtomic || (hw < HW::XeHPC);
        if (branchAtomic) {
            if_(16 | mod, labelEndAtomic);
            setDefaultNoMask(false);
        } else
            block.flag[0] = VirtualFlag(mod.getFlagReg(), hw);

#ifdef STANDALONE
        if (!(problem.mock.cLoad && problem.mock.cStore))
#endif
        atomicAddMatrixBlock(Tc_ext, GRFRange(Cacc.getBase(), Cacc.getLen()), block, problem.C, strategy.C, header[0], problem, strategy, state);

        if (branchAtomic) {
            setDefaultNoMask(true);
            mark(labelEndAtomic);
            endif(16);
        }

        if (!nativeAtomic)
            freeEAtomicAddRegs(state, f1[1]);

        std::swap(saveVFlags, state.vflagStorage);
    } else {
        // Late C conversion, if needed.
        auto originalTacc = state.Tacc;
        if (problem.needsTsConvert() && state.Tacc != problem.Ts) {
            convert(GRFRange(Cacc.getBase(), Cacc.getLen()), state.Tacc, problem.Ts, strategy, state);
            state.setTacc(problem.Ts);
        }

        // Regular update.
#ifdef STANDALONE
        if (!problem.mock.cLoad)
#endif
        if (loadOnly || !problem.beta0()) {
            doReadSuppressionWA(strategy, state);
            if (atomicLoad && hw >= HW::Xe2) {
                if (!strategy.C.newDP) stub();
                !byte_access         ? atomic(AtomicOp::load, 16 | mod, Cload, D32    | strategy.C.cachingR, strategy.C.base, header[0]) :
                (Tc_ext.size() == 2) ? atomic(AtomicOp::load, 16 | mod, Cload, D16U32 | strategy.C.cachingR, strategy.C.base, header[0])
                                     : stub();
            } else if (strategy.C.newDP) {
                !byte_access         ? load(16 | mod, Cload, D32    | strategy.C.cachingR, strategy.C.base, header[0]) :
                (Tc_ext.size() == 2) ? load(16 | mod, Cload, D16U32 | strategy.C.cachingR, strategy.C.base, header[0])
                                     : load(16 | mod, Cload, D8U32  | strategy.C.cachingR, strategy.C.base, header[0]);
            } else {
                byte_access ? load(16 | mod, Cload, scattered_byte(Tc_ext.size()), strategy.C.base, header[0]) :
                   !surface ? load(16 | mod, Cload, scattered_dword(),             strategy.C.base, header[0])
                            : load(16 | mod, Cload, surface_dword(ChannelMask::r), strategy.C.base, header[0]);
            }
        }

        if (!loadOnly) {
            auto Tc_out = (op == COperation::UpdateStore) ? problem.Tc_ext : state.Tacc;
            if (!problem.beta0())
                convert(Cload, problem.Tc_ext, state.Tacc, strategy, state);
            updateC(Cacc, CaccSwap, Cload, problem, strategy, state);
            convert(Cacc, state.Tacc, Tc_out, strategy, state);
        }

#define IGNORE_SWSB ignoredep(Operand::dst);

        if (op != COperation::UpdateStore) {
            auto src = (op == COperation::Load) ? GRFRange(Cload.getBase(), Cload.getLen()) : GRFRange(Cacc.getBase(), Cacc.getLen());
            if (uniform) switch (Tc.size()) {
                case 1:  IGNORE_SWSB mov<uint32_t>(16 | mod, indirect[a0[1]].ub(), src); break;
                case 2:  IGNORE_SWSB mov<uint32_t>(16 | mod, indirect[a0[1]].uw(), src); break;
                default: IGNORE_SWSB mov<uint32_t>(16 | mod, indirect[a0[1]],      src); break;
            } else if (xByteInc == 1) switch (state.Tacc.size()) {
                case 2:  IGNORE_SWSB mov<uint32_t>(16 | mod, indirect[a0[2]].uw(), src); break;
                default: IGNORE_SWSB mov<uint32_t>(16 | mod, indirect[a0[2]],      src); break;
            } else if (xByteInc == 2) switch (state.Tacc.size()) {
                case 2:  IGNORE_SWSB mov<uint32_t>(8  | mod,      indirect[a0[2]].uw(), src);
                         IGNORE_SWSB mov<uint32_t>(8  | mod | M8, indirect[a0[3]].uw(), src.sub(hw, 8, DataType::ud)(1)); break;
                default: IGNORE_SWSB mov<uint32_t>(8  | mod,      indirect[a0[2]].ud(), src);
                         IGNORE_SWSB mov<uint32_t>(8  | mod | M8, indirect[a0[3]].ud(), src.sub(hw, 8, DataType::ud)(1)); break;
            } else stub();
        } else
#ifdef STANDALONE
        if (!problem.mock.cStore)
#endif
        FOR_EACH_C {
            if (strategy.C.newDP) {
                !byte_access         ? store(16 | mod, D32    | strategy.C.cachingW, strategy.C.base, header[q], Cacc) :
                (Tc_ext.size() == 2) ? store(16 | mod, D16U32 | strategy.C.cachingW, strategy.C.base, header[q], Cacc)
                                     : store(16 | mod, D8U32  | strategy.C.cachingW, strategy.C.base, header[q], Cacc);
            } else {
                byte_access ? store(16 | mod, scattered_byte(Tc_ext.size()), strategy.C.base, header[q], Cacc) :
                   !surface ? store(16 | mod, scattered_dword(),             strategy.C.base, header[q], Cacc)
                            : store(16 | mod, surface_dword(ChannelMask::r), strategy.C.base, header[q], Cacc);
            }
        }

        state.setTacc(originalTacc);
    }

#undef IGNORE_SWSB

    if (hw >= HW::XeHPC)
        cmp<int16_t>(1 | ge | f0[0], ix, 0);

    add(1, a0[1], a0[1], xByteInc);
    if (!transpose) {
        uint16_t inc = nec * Tc_ext;
        if (!surface) {
            FOR_EACH_C eadd<uint64_t>(std::min(2 * neq, rsimd), header[q][0], header[q][0], inc, strategy, state);
            if (header4)
                FOR_EACH_C eadd<uint64_t>(8, header[q][2], header[q][2], inc, strategy, state);
        } else
            FOR_EACH_C add<uint32_t>(rsimd, header[q][0], header[q][0], inc);
    } else {
        if (!surface) {
            FOR_EACH_C eadd<uint64_t>(std::min(2 * neq, rsimd), header[q][0], header[q][0], cXInc[q], strategy, state);
            if (header4)
                FOR_EACH_C eadd<uint64_t>(8, header[q][2], header[q][2], cXInc[q], strategy, state);
        } else
            FOR_EACH_C add<uint32_t>(rsimd, header[q][0], header[q][0], cXInc[q]);
    }

    // Bottom of x loop.
    //  Fused threads must use SIMT control flow instructions.
    strategy.fused ? simtDoWhileLoop(16 | f0[0], xLoop)
                   :            jmpi(1  | f0[0], xLoop);

    if (lateYLoopCheck)
        add(1 | gt | f0[1], iy, iy, int16_t(-1));
    add(1, a0[0], a0[0], yByteInc);
    if (!surface) {
        FOR_EACH_C eadd<uint64_t>(std::min(2 * neq, rsimd), header[q][0], header[q][0], cYInc[q], strategy, state);
        if (header4)
            FOR_EACH_C eadd<uint64_t>(8, header[q][2], header[q][2], cYInc[q], strategy, state);
    } else
        FOR_EACH_C add<uint32_t>(rsimd, header[q][0], header[q][0], cYInc[q]);

    // Bottom of y loop.
    //  The any16h is a trick: only the lowest bit of f0[1] is updated when decrementing iy,
    //  but we want to apply it to all channels.
    strategy.fused ? simtDoWhileLoop(16 | f0[1] | any16h, yLoop)
                   : jmpi(1 | f0[1], yLoop);

    // Wait for indirect moves to C registers.
    if (op != COperation::UpdateStore)
        sync.nop(SWSB<uint32_t>(1));

    // Cleanup.
    state.raVFlag.release(f0[0]);
    state.raVFlag.release(f0[1]);
    state.raVFlag.release(f1[0]);
    state.ra.safeRelease(bases);

    if (fbfEmulate)
        state.raVFlag.safeRelease(state.emulate.flag);

    state.ra.safeRelease(indexVec);
    state.ra.safeRelease(Cload);
    state.ra.safeRelease(CaccSwap);
    state.ra.safeRelease(Cacc);
    FOR_EACH_C state.ra.safeRelease(cXInc[q]);
    FOR_EACH_C state.ra.safeRelease(cYInc[q]);
    state.ra.safeRelease(iy);
    state.ra.safeRelease(ix);
    state.ra.safeRelease(ix_init);
    FOR_EACH_C state.ra.safeRelease(header[q]);

    state.flagAP = saveFlagAP;
    if (state.flagAP.isValid())
        state.raVFlag.claim(state.flagAP);

#undef FOR_EACH_C
}

// Convert register range in place to a new type.
// If types are different sizes, we assume that the smaller type's stride is the width of the larger type.
template <HW hw>
void Generator<hw>::convert(const GRFMultirange &range, Type Told, Type Tnew, const CommonStrategy &strategy, CommonState &state)
{
    if (Told == Tnew)
        return;
    if (Told.isSubByteInt() || Tnew.isSubByteInt()) stub();
    if (Told == Type::hf8) stub();

    // Special path: x32->FP.
    if (one_of(Told, {Type::s32, Type::u32}) && Tnew.isFP()) {
        map(hw, Told, range, range, strategy, [&](int esize, GRF r, GRF _) {
            mov(esize, r.f(), r);
        });
        Told = Type::f32;
        if (Told == Tnew) return;
    }

    // Special path: f32->fp8.
    if (Told == Type::f32 && one_of(Tnew, {Type::bf8, Type::hf8})) {
        int ne = elementsPerGRF<uint32_t>(hw);
        CopyPlan plan(getProductFamily(), strategy.systolicAvailable);
        for (int i = 0; i < range.getLen(); i++) {
            CopyOperand sOp = range[i].f();
            CopyOperand dOp = range[i].retype(Tnew.ngen());
            dOp.stride = 4;
            plan.append(Opcode::mov, ne, dOp, sOp);
        }
        copyExecute(std::move(plan), state);
        return;
    }

    // Special path: s16->f16.
    if (Told == Type::s16 && Tnew == Type::f16) {
        int ne = elementsPerGRF<uint32_t>(hw);
        for (int i = 0; i < range.getLen(); i++)
            mov(ne, range[i].hf(0)(2), range[i].w(0)(2));
        for (int i = 0; i < range.getLen(); i++)
            rol(ne, range[i].ud(), range[i].ud(), 16);
        for (int i = 0; i < range.getLen(); i++)
            mov(ne, range[i].hf(0)(2), range[i].w(0)(2));
        for (int i = 0; i < range.getLen(); i++)
            rol(ne, range[i].ud(), range[i].ud(), 16);
        return;
    }

    // Special path: s16->bf16.
    if (Told == Type::s16 && Tnew == Type::bf16) {
        int ne = elementsPerGRF<uint16_t>(hw);
        CopyPlan plan(getProductFamily(), strategy.systolicAvailable);
        for(int i = 0; i < range.getLen(); i++) {
            CopyOperand sOp(range[i]), dOp(sOp);
            sOp.type = Told.ngen();
            dOp.type = Tnew.ngen();
            plan.append(Opcode::mov, ne, dOp, sOp);
        }
        copyExecute(std::move(plan), state);
        return;
    }

    int maxLS = std::max(Told.log2Size(), Tnew.log2Size());
    int hsOld = 1 << (maxLS - Told.log2Size());
    int hsNew = 1 << (maxLS - Tnew.log2Size());
    auto Tmax = (Told.size() < Tnew.size()) ? Tnew : Told;

    InstructionModifier mod;
    if (Told != Tnew && Tnew.isInteger() && Tnew.size() <= Told.size())
        mod = mod | sat;

    map(hw, Tmax, range, range, strategy, [&](int esize, GRF r, GRF _) {
        emov(esize | mod, r.sub(0, Tnew.ngen())(hsNew), r.sub(0, Told.ngen())(hsOld), strategy, state);
    });
}

// Convert C accumulator registers to a new type. Returns true if successful, or false if old and new type are different sizes.
template <HW hw>
bool Generator<hw>::gemmConvertC(Type Tnew, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Told = state.Tacc;
    int ncomp = (problem.Tc.isComplex() && state.haveCSwap && state.cSwapActive) ? 2 : 1;

    if (Tnew.bits() != Told.bits())
        return false;

    for (int comp = 0; comp < ncomp; comp++)
        convert(state.C_regs[comp], Told, Tnew, strategy, state);

    state.setTacc(Tnew);

    return true;
}

// Store A/B sum data into CO.
template <HW hw>
void Generator<hw>::gemmAccessSums(COperation op, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    bool sumA = problem.sumA, sumB = problem.sumB;

    if (!sumA && !sumB) return;
    if (sumA && sumB) stub();   // can only store one of the two in CO.

    auto Tc = problem.Tc;
    auto Tco = problem.Tco;
    auto cor = sumA ? strategy.unroll[LoopM] : 1;
    auto coc = sumB ? strategy.unroll[LoopN] : 1;
    bool atomicUpdate = strategy.CO.atomic && (op == COperation::UpdateStore);
    bool loadOnly = (op == COperation::Load);
    bool load = (op != COperation::Store && !problem.beta0() && !(problem.beta1() && atomicUpdate));

    auto CO = problem.CO;
    auto CO_strategy = strategy.CO;
    std::vector<RegisterRange> CO_addrs;
    std::vector<MaskAssignment> masks;
    GRFMultirange CO_regs;
    CO_strategy.accessType = AccessType::Block;

    auto &Xs_regs   = sumA ? state.As_regs   : state.Bs_regs;
    auto &Xs_layout = sumA ? state.As_layout : state.Bs_layout;

    int Xs_nregs = Xs_layout.regs();
    auto Xs_usedRegs = Xs_regs.subrange(0, Xs_nregs);

    CO.layout = sumA ? MatrixLayout::N : MatrixLayout::T;

    auto remR = sumA && !strategy.CO.padded && strategy.remHandling[LoopM] != RemainderHandling::Ignore;
    auto remC = sumB && !strategy.CO.padded && strategy.remHandling[LoopN] != RemainderHandling::Ignore;

    RegisterLayout CO_layout(hw, Tco, cor, coc, CO, CO_strategy, remR, remC, true);

    bool share = (Tc == Tco) && CO_layout.match(Xs_layout);

    Label noAccess;
    and_(16 | ne | state.flagAP, null.ud(), state.inputs.flags, FlagStoreSums);
    if_(16 | state.flagAP, noAccess);

    allocAddrRegs(hw, CO_addrs, CO_layout, state);
    setupAddr(CO_addrs, state.effCO, CO_layout, Subregister(), strategy, state);

    if (!assignMasks(CO_layout, LoopM, LoopN, masks, strategy, state, true)) stub();

    loadMasks(masks, state.remainders, strategy, state);

    if (load) {
        GRFMultirange CO_regsLoad, CO_regsLoadConv;

        if (loadOnly) {
            CO_regsLoadConv = CO_regsLoad = Xs_usedRegs;
            if (!share)
                CO_regsLoad = state.ra.alloc_range(CO_layout.regs());
        } else {
            CO_regsLoadConv = CO_regsLoad = state.ra.alloc_range(CO_layout.regs());
            if (!share)
                CO_regsLoadConv = state.ra.alloc_range(Xs_nregs);
        }

        loadMatrix(CO_regsLoad, CO_layout, CO_addrs, strategy, state);
        if (!share)
            copyRegisters(CO_layout, Xs_layout, CO_regsLoad, CO_regsLoadConv, strategy, state);

        auto &beta = problem.beta;

        if (!loadOnly) map(hw, Tc, Xs_usedRegs, CO_regsLoadConv, strategy, [&](int esize, GRF acc, GRF loaded) {
            if (beta == 1)
                add(esize, acc, acc, loaded);
            else if (beta.fixed())
                mad(esize, acc, acc, loaded, cast(Tc.real(), beta));
            else
                mad(esize, acc, acc, loaded, state.inputs.beta_real.getRegAvoiding(hw, acc));
        });

        if (!loadOnly)
            safeReleaseRanges(CO_regsLoadConv, state);
        if (!share)
            safeReleaseRanges(CO_regsLoad, state);
    }

    if (!loadOnly) {
        if (!share) {
            CO_regs = state.ra.alloc_range(CO_layout.regs());
            copyRegisters(Xs_layout, CO_layout, Xs_regs, CO_regs, strategy, state);
            releaseRanges(Xs_regs, state);
        }

        auto &effCO_regs = share ? Xs_regs : CO_regs;
        if (atomicUpdate) {
            allocEAtomicAddRegs(hw, Tco, CO_layout, CO, CO_strategy, state, state.flagAP);
            atomicAddMatrix(effCO_regs, CO_layout, CO_addrs, problem, strategy, state);
            freeEAtomicAddRegs(state, state.flagAP);
        } else
            storeMatrix(effCO_regs, CO_layout, CO_addrs, strategy, state);
    }

    mark(noAccess);
    endif(16);

    safeReleaseMaskAssignments(masks, state);

    if (!share) {
        safeReleaseRanges(CO_regs, state);
        reclaimRanges(Xs_regs, state);
    }
}

// Generate code for summing C across k dimension through SLM.
template <HW hw>
void Generator<hw>::gemmKReduce(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc = problem.Tc;
    Label lDone;

#ifdef STANDALONE
    if (problem.mock.cUpdate && !problem.mock.keepKReduce) return;
#endif

    // Early exit if nothing to do. All branching scalar since no fusing in k dimension.
    if (!strategy.fixedWGK()) {
        cmp(1 | le | state.flagAP, state.lszK, 1);
        jmpi(1 | state.flagAP, lDone);
    } else if (state.lszK.isInvalid()) {
        state.lszK = state.ra.alloc_sub<uint32_t>();
        mov(1, state.lszK, strategy.wg[LoopK]);
    }

    status << "k reduction through SLM" << status_stream::endl;
    cmp(1 | eq | state.flagAP, state.lidK, 0);

    auto C_regs = state.C_regs[0];

    // Reduce A/B sums at the same time.
    if (problem.sumA) C_regs.append(state.As_regs);
    if (problem.sumB) C_regs.append(state.Bs_regs);

    // In general SLM isn't large enough to do the reduction in one step.
    // Slice C into pieces that will fit.
    int maxMNThreads = strategy.wg[LoopM] * strategy.wg[LoopN];
    if (maxMNThreads <= 0) stub("Max workgroup size not specified");

    int regs = C_regs.getLen();
    int sliceRegs = int(gemmPerKSLMSize(hw, problem, strategy) / (maxMNThreads * GRF::bytes(hw)));
    if (sliceRegs <= 0) stub("Not enough SLM for k reduction");
    sliceRegs = std::min<int>(sliceRegs, C_regs.getLen());

    // Temporaries.
    auto kt = state.ra.alloc_sub<int32_t>();
    auto flagKTLoop = state.raVFlag.alloc();
    auto barrierTemp = state.ra.alloc();

    GRF r0_info = state.r0InfoGRF();

    bool initialBarrier = (std::max(strategy.slmBuffersA, strategy.slmBuffersB) || strategy.persistentLoop());
    MOCK_BARRIERS if (initialBarrier)
        activeThreadBarrierSignal(barrierTemp, r0_info, strategy);

    // Set up addressing.
    auto addr0 = state.ra.alloc_sub<uint32_t>();
    emad(1, addr0, state.lidM, state.lidN, strategy.wg[LoopM], strategy, state);
    emad(1, addr0, addr0, state.lidK, strategy.wg[LoopM] * strategy.wg[LoopN], strategy, state);
    mulConstant(1, addr0, addr0, sliceRegs * GRF::bytes(hw));
    makeSLMBaseRelative(addr0, state);

    int unrollKSLMStride = strategy.wg[LoopM] * strategy.wg[LoopN] * sliceRegs * GRF::bytes(hw);
    Subregister unrollKSLMReturn = state.ra.alloc_sub<int32_t>();

    mulConstant(1, unrollKSLMReturn, -state.lszK, unrollKSLMStride);

    MatrixAddressing C_slm;
    MatrixAddressingStrategy C_slmStrategy;

    C_slm.layout = MatrixLayout::Pc;
    C_slm.packSize = elementsPerGRF(hw, Tc);
    C_slm.crosspack = 1;
    C_slm.setAlignment(GRF::bytes(hw));

    C_slmStrategy.base = SLM;
    C_slmStrategy.accessType = AccessType::Block;
    C_slmStrategy.padded = true;
    if (hw >= HW::XeHPG)
        C_slmStrategy.newDP = true;

    vector<RegisterRange> C_load;
    RegisterLayout C_slmLayout;
    vector<RegisterRange> C_slmAddrs;

    // Find maximum # registers of C we can transfer to/from SLM at once.
    int maxContig = rounddown_pow2(regs);
    for (; maxContig > 1; maxContig >>= 1) {
        bool ok = true;
        for (int offsetReg = 0; offsetReg < regs; offsetReg += maxContig) {
            int nr = std::min(regs - offsetReg, maxContig);
            if (!C_regs.contiguous(offsetReg, nr)) {
                ok = false;
                break;
            }
        }
        if (ok) break;
    }

    if (sliceRegs > maxContig)
        sliceRegs = align_down(sliceRegs, maxContig);
    else if (sliceRegs < maxContig)
        sliceRegs = rounddown_pow2(sliceRegs);

    // Allocate address and data registers, automatically shrinking sliceRegs if
    //  there are not enough registers.
    C_load.resize(1);
    for (; sliceRegs > 0; sliceRegs = rounddown_pow2(sliceRegs - 1)) {
        bool ok = true;

        C_load[0] = state.ra.try_alloc_range(sliceRegs);
        ok = ok && C_load[0].isValid();

        C_slmLayout = RegisterLayout(hw, Tc, elementsPerGRF(hw, Tc), sliceRegs, C_slm, C_slmStrategy,
                                     false, false, true, AvoidFragment, 0, maxContig);
        ok = ok && tryAllocAddrRegs(hw, C_slmAddrs, C_slmLayout, state);

        if (ok) break;

        state.ra.safeRelease(C_load[0]);
    }

    if (sliceRegs <= 0)
        throw out_of_registers_exception();

    // Allocate additional data registers for unrolling the loop.
    while (int(C_load.size()) < strategy.wg[LoopK] - 1) {
        auto range = state.ra.try_alloc_range(sliceRegs);
        if (range.isInvalid()) break;
        C_load.push_back(range);
    }

    setupAddr(C_slmAddrs, addr0, C_slmLayout, Subregister(), strategy, state);

    MOCK_BARRIERS if (initialBarrier)
        barrierwait();

    // Loop over slices.
    for (int rr = 0; rr < regs; rr += sliceRegs) {
        Label lSkipWrite, lSkipReduce, lTop, lTopMulti, lBottom, lBottomMulti;

        int nreg = std::min(sliceRegs, regs - rr);
        auto C_range = C_regs.subrange(rr, nreg);

        MOCK_BARRIERS if (rr > 0) slmBarrier(barrierTemp, r0_info, strategy);

        // Trim down SLM layout for final loop.
        if (nreg < sliceRegs) {
            vector<RegisterRange> subaddrs;
            auto sublayout = C_slmLayout.slice(subaddrs, C_slmAddrs, true, 0, nreg, true);
            std::swap(sublayout, C_slmLayout);
            std::swap(subaddrs, C_slmAddrs);
        }

        // Non-leaders write to SLM.
        jmpi(1 | state.flagAP, lSkipWrite);
        storeMatrix(C_range, C_slmLayout, C_slmAddrs, strategy, state);
        mark(lSkipWrite);

        MOCK_BARRIERS slmBarrier(barrierTemp, r0_info, strategy);

        // Leader reads SLM data and accumulates C.
        jmpi(1 | ~state.flagAP, lSkipReduce);

        auto doLoad = [&](int b) {
            loadMatrix(GRFRange(C_load[b].getBase(), C_load[b].getLen()), C_slmLayout, C_slmAddrs, strategy, state);
        };

        auto doInc = [&] {
            incAddr(C_slmAddrs, unrollKSLMStride, C_slmLayout, strategy, state);
        };

        auto doReduce = [&](int b) {
            map(hw, Tc.real(), C_range, C_load[b], strategy, [&](int simd, GRF r1, GRF r2) {
                add(simd, r1, r1, r2);
            });
        };

        auto loadBufs = int(C_load.size());
        cmp(1 | gt | flagKTLoop, state.lszK, loadBufs);
        add(1, kt, state.lszK, -1);
        doInc();

        jmpi(1 | ~flagKTLoop, lTop);

        add(1 | gt | flagKTLoop, kt, kt, -2*loadBufs + 1);
        for (int b = 0; b < loadBufs; b++) {
            doLoad(b);
            doInc();
        }

        jmpi(1 | ~flagKTLoop, lBottomMulti);
        mark(lTopMulti);
        add(1 | gt | flagKTLoop, kt, kt, -loadBufs);
        for (int b = 0; b < loadBufs; b++) {
            doReduce(b);
            doLoad(b);
            doInc();
        }
        jmpi(1 | flagKTLoop, lTopMulti);
        mark(lBottomMulti);

        add(1 | gt | flagKTLoop, kt, kt, loadBufs - 1);

        for (int b = 0; b < loadBufs; b++)
            doReduce(b);

        jmpi(1 | ~flagKTLoop, lBottom);

        mark(lTop);
        add(1 | gt | flagKTLoop, kt, kt, -1);
        doLoad(0);
        doInc();
        doReduce(0);
        jmpi(1 | flagKTLoop, lTop);
        mark(lBottom);

        if (rr + nreg < regs)
            incAddr(C_slmAddrs, unrollKSLMReturn, C_slmLayout, strategy, state);

        mark(lSkipReduce);
    }

    // Followers will not update C.
    mov(1 | ~state.flagAP, state.remainders[LoopM], 0);
    mov(1 | ~state.flagAP, state.remainders[LoopN], 0);

    state.raVFlag.safeRelease(flagKTLoop);
    safeReleaseRanges(C_load, state);
    state.ra.safeRelease(kt);
    state.ra.safeRelease(unrollKSLMReturn);
    state.ra.safeRelease(addr0);
    state.ra.safeRelease(barrierTemp);
    safeReleaseRanges(C_slmAddrs, state);

    mark(lDone);
}

// Final horizontal reduction of C accumulators after dot-product inner loop.
//    TODO: move to new copy planner once ready.
template <HW hw>
void Generator<hw>::gemmDotReduce(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto Tc = problem.Tc;
    int vl = strategy.dotVL;
    int ne = elementsPerGRF(hw, Tc);
    auto &C_layout = state.C_layout, &C_layoutReduced = state.C_layoutReduced;
    auto &C_regs = state.C_regs[0];

    bool globalCM = state.C_layout.colMajor();

    if (!vl) return;
    if (vl % ne) stub();
    if (state.C_buffers > 1) stub();

    int nx = strategy.unroll[globalCM ? LoopM : LoopN];
    int ny = strategy.unroll[globalCM ? LoopN : LoopM];

    auto C_regsReduced = state.ra.alloc_range(C_layoutReduced.regs());

    bool needSwizzle = (hw > HW::Gen12LP && Tc.isFP());
    GRFMultirange temps;

    if (needSwizzle)
        temps = tryChunkAlloc(nx * ny, 1, BundleGroup::AllBundles(), BundleGroup::AllBundles(), state);

    for (int cvl = vl; cvl > 1; ) {
        int simd = (cvl > ne) ? ne : (cvl >> 1);
        bool canSwizzle = (cvl == 2 || cvl > ne || !needSwizzle);

        for (bool shift : {true, false}) {
            if (shift && (canSwizzle || temps.empty())) continue;
            for (int y = 0; y < ny; y++) {
            for (int x = 0; x < nx; x++) {
                auto x0 = x * vl, x1 = x0 + cvl - simd;
                auto i0 = globalCM ? x0 : y, i1 = globalCM ? x1 : y;
                auto j0 = globalCM ? y : x0, j1 = globalCM ? y : x1;
                auto i  = globalCM ? x : y,   j = globalCM ? y : x;

                int nc0, nc1;
                auto C0 = C_layout.find(i0, j0, C_regs, &nc0);
                auto C1 = C_layout.find(i1, j1, C_regs, &nc1);
                if (nc0 < simd || nc1 < simd) stub();

                auto dst = C0;
                if (cvl == 2)
                    dst = C_layoutReduced.find(i, j, C_regsReduced, &nc0);

                if (canSwizzle)
                    add(simd, dst(1), C0(1), C1(1));
                else if (!temps.empty()) {
                    auto temp = temps[x + y*nx].retype(Tc.real().ngen());
                    auto tempI = temp[0], C1I = C1;
                    moveToIntPipe(tempI);
                    moveToIntPipe(C1I);
                    shift ? mov(simd, tempI(1), C1I(1))
                          : add(simd, C0(1), C0(1), temp);
                } else {
                    auto temp = state.ra.alloc().retype(Tc.real().ngen());
                    auto tempI = temp[0], C1I = C1;
                    moveToIntPipe(tempI);
                    moveToIntPipe(C1I);
                    mov(simd, tempI(1), C1I(1));
                    add(simd, C0(1), C0(1), temp);
                    state.ra.safeRelease(temp);
                }
            }
            }
        }

        cvl -= simd;
    }

    safeReleaseRanges(C_regs, state);
    safeReleaseRanges(temps, state);

    C_layout = C_layoutReduced;
    C_regs = std::move(C_regsReduced);
}

template <HW hw>
void Generator<hw>::gemmPrefetchC(const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    bool checkBeta0 = problem.checkBeta0 && !problem.beta.fixed();
    bool checkIDK = strategy.kParallelLocal;

    releaseRanges(state.Ap_regs, state);
    releaseRanges(state.Bp_regs, state);

    status << "Prefetch C" << status_stream::endl;

    if (checkBeta0) {
        cmp0(1 | eq | state.flagAP, state.inputs.beta_real.getReg(0));
    }

    Address2DParams Cp_params;
    if (strategy.C.address2D) {
        Cp_params.rows = state.inputs.m;
        Cp_params.cols = state.inputs.n;
        Cp_params.offR = state.i0;
        Cp_params.offC = state.j0;
    } else {
        Cp_params.rows = state.remainders[LoopM];
        Cp_params.cols = state.remainders[LoopN];
    }
    Cp_params.remR = state.remainders[LoopM];
    Cp_params.remC = state.remainders[LoopN];

    bool oldAdd32 = strategy.emulate.emulate64_add32;
    strategy.emulate.emulate64_add32 = false;

    gemmCacheLDCMultiples(problem, strategy, state, 1);

    if (checkIDK) {
        if (checkBeta0)
            cmp(1 | ~state.flagAP | gt | state.flagAP, state.lidK, 0);
        else
            cmp(1 | gt | state.flagAP, state.lidK, 0);
    }

    allocAddrRegs(hw, state.Cp_addrs, state.Cp_layout, state);
    setupAddr(state.Cp_addrs, state.effCp, state.Cp_layout, state.inputs.ldc[0], strategy, state, Cp_params, state.ldcMultiples[0]);

    Label lSkipPrefetchC;
    if (checkBeta0 || checkIDK)
        jmpi(1 | state.flagAP, lSkipPrefetchC);

    state.Cp_regs = state.ra.alloc_range(state.Cp_layout.regs());

    loadMatrix(state.Cp_regs, state.Cp_layout, state.Cp_addrs, strategy, state);
#ifdef STANDALONE
    if (problem.readCheck && problem.readCheckC)
        readCheck(state.Cp_regs, strategy, state);
#endif

    safeReleaseRanges(state.Cp_regs, state);
    safeReleaseRanges(state.Cp_addrs, state);

    releaseLDMultiples(state.ldcMultiples[0], state);
    releaseIndexVec(state);

    if (checkBeta0 || checkIDK)
        mark(lSkipPrefetchC);

    strategy.emulate.emulate64_add32 = oldAdd32;

    reclaimRanges(state.Ap_regs, state);
    reclaimRanges(state.Bp_regs, state);
}

template <HW hw>
void Generator<hw>::setupCAddr0(RegisterRange (&C_addr0)[2], RegisterRange (&C_addr0Unmasked)[2],
                                const RegisterLayout &C_layout, const RegisterLayout &C_layoutUnmasked, int C_count,
                                const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, const Address2DParams *params)
{
    Address2DParams defaultParams;
    if (!params) {
        getDefaultCParams(defaultParams, state);
        params = &defaultParams;
    }
    for (int q = 0; q < C_count; q++) {
        C_addr0[q] = state.ra.alloc_range(C_layout[0].addrGRFs(problem.C, strategy.C));
        setupAddr(problem.Tc_ext, C_addr0[q], state.effC[q], C_layout[0], state.inputs.ldc[q], problem.C, strategy.C, strategy, state, *params, state.ldcMultiples[q]);
    }
    if (!C_layoutUnmasked.empty()) for (int q = 0; q < C_count; q++) {
        C_addr0Unmasked[q] = state.ra.alloc_range(C_layoutUnmasked[0].addrGRFs(problem.C, strategy.C));
        setupAddr(problem.Tc_ext, C_addr0Unmasked[q], state.effC[q], C_layoutUnmasked[0], state.inputs.ldc[q], problem.C, strategy.C, strategy, state, *params, state.ldcMultiples[q]);
    }
}

GEMMSTONE_NAMESPACE_END
