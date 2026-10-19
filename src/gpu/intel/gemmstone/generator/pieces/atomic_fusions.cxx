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
#include "atomic_fusions.hpp"
#include "gemmstone/generator.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


// Preparatory work for fused beta/post-ops.
// For fused beta, check whether this thread is responsible for performing beta scaling.
template <HW hw>
void Generator<hw>::gemmFusedBetaPOInit(const Subregister &groupID, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto &addr = state.statusFlagAddr;

    if (strategy.fuseBeta) {
        auto header = state.ra.alloc_range(2);
        auto data = state.ra.alloc_range(2);
        addr = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));

        /* Check for group leader */
        add(1 | ze | f0[0], null.uw(), state.lidM, state.lidN);

        /* Alternate mode uses a static check to determine who should do
           beta responsibility: the workgroup with h0 == 0. */
        if (strategy.altFusedBeta && strategy.kParallel)
            cmp(1 | eq | f0[1], null.ud(), state.inputs.groupIDK, 0);

        /* Group leader check, cont'd. */
        if (strategy.kParallelLocal)
            cmp(1 | f0[0] | ze | f0[0], null.uw(), state.lidK, 0);

        /* Regular mode: group leader does cmpxchg to change counter from 0 -> -1.
           If successful, beta check is our responsibility. */
        shl(1, header[0].ud(2), groupID, ilog2(strategy.statusFlagStride()));
        mov<int32_t>(1, data[0], 0);
        mov<int32_t>(1, data[1], -256);
        eadd(1, header[0].uq(0), state.inputs.statusBuffer, header[0].ud(2), strategy, state);
        or_(1 | f0[0], state.inputs.flags, state.inputs.flags, FlagLeader);
        if (!strategy.altFusedBeta) {
            if (hw >= HW::XeHPG)
                atomic(AtomicOp::cmpwr, 1 | f0[0], data, D32, A64, header, data);
            else
                atomic(AtomicOp::cmpwr, 1 | f0[0], data, scattered_dword(), A64, header, data);
        }
        emov(1, addr, header[0].uq(0), strategy, state);
        if (strategy.altFusedBeta && strategy.kParallel)
            or_(1 | f0[1], state.inputs.flags, state.inputs.flags, FlagDidBeta);

        state.ra.safeRelease(header);

        state.betaCheckReturn = data[0].d();
        state.ra.safeRelease(data);
        state.ra.claim(state.betaCheckReturn);
    } else if (strategy.fusePostOps) {
        addr = state.ra.alloc_sub<uint64_t>(getHint(HintType::LongTerm, strategy));

        add(1 | ze | f0[0], null.uw(), state.lidM, state.lidN);
        shl(1, addr.ud(), groupID, ilog2(strategy.statusFlagStride()));
        if (strategy.kParallelLocal)
            cmp(1 | f0[0] | ze | f0[0], null.uw(), state.lidK, 0);
        or_(1 | f0[0], state.inputs.flags, state.inputs.flags, FlagLeader);
        eadd(1, addr, state.inputs.statusBuffer, addr.ud(), strategy, state);
    }

    if (state.useTempC) {
        if (problem.batch != BatchMode::None) stub();

        /* Calculate base address for temporary C buffer */
        auto temp1 = state.ra.alloc_sub<uint32_t>();
        auto temp2 = state.ra.alloc_sub<uint32_t>();

        int stride = tempCThreadStride(problem, strategy);
        mulConstant(1, temp1, groupID, strategy.wg[LoopM] * strategy.wg[LoopN]);
        emad(1, temp2, state.lidM, state.lidN, strategy.wg[LoopM], strategy, state);
        add(1, temp1, temp1, temp2);
        if (strategy.C.base.isStateless()) {
            mulConstant(1, temp1, temp1, stride);
            eadd(1, state.effTempC, state.inputs.tempC, temp1, strategy, state);
        } else
            mulConstant(1, state.effTempC, temp1, stride);

        state.ra.safeRelease(temp1);
        state.ra.safeRelease(temp2);
    }
}

// Zero C matrix in memory.
template <HW hw>
void Generator<hw>::gemmStoreZeroC(GEMMProblem problem, GEMMStrategy strategy, GEMMState state, bool initialZeroing)
{
    int nreg = 0;

    if (!initialZeroing) problem.sumA = problem.sumB = false;

    if (state.useTempC) {
        gemmRedirectToTempC(problem, strategy, state);
        setCCachingW(strategy, state, CacheSettingsLSC::L1UC_L3WB, CacheSettingsLSC::L1UC_L3WB, false);
    }

    auto collapse = [&](RegisterLayout &layout) {
        for (auto &block: layout) {
            block.offsetBytes = 0;
            nreg = std::max<int>(nreg, block.msgRegs);
        }
    };

    collapse(state.C_layoutExt);
    collapse(state.C_layoutExtUnmasked);
    collapse(state.C_layoutExtNonatomicUnmasked);

    if (strategy.altCRemainder)
        nreg = state.C_regs[0].getLen();
    else {
        if (state.copyC) {
            state.copyC = false;
            state.C_layout = state.C_layoutExt;
        }
        if (initialZeroing)
            for (auto &rr: state.C_regs)
                safeReleaseRanges(rr, state);
        state.C_regs[0] = state.ra.alloc_range(nreg);
    }

    zeroMatrix(state.C_regs[0], strategy);
    if (problem.sumA) zeroMatrix(state.As_regs, strategy);
    if (problem.sumB) zeroMatrix(state.Bs_regs, strategy);
    gemmAccessC(COperation::Store, problem, strategy, state);
}

// Perform beta scaling if necessary, for atomic kernels with fused beta scaling.
template <HW hw>
void Generator<hw>::gemmFusedBetaScale(GEMMProblem problem, GEMMStrategy strategy, GEMMState &state)
{
    Label lNoScale, lScaleDone, lBeta0;
    bool checkIfEnabled = strategy.kParallelVariable;

    if (strategy.altFusedBeta) return;  /* Beta scaling is performed later on the alt path. */

    status << "Beta scaling prior to atomic C update" << status_stream::endl;

    Subregister flagSave[4];
    for (int i = 0; i < FlagRegister::count(hw); i++) {
        flagSave[i] = state.ra.alloc_sub<uint32_t>();
        mov(1, flagSave[i], FlagRegister(i).ud());
    }

    if (checkIfEnabled)
        and_(1 | ze | f0[0], null, state.inputs.flags, FlagKPartitioned);

    // Leader previously issued check if we need to do beta scaling (gemmFusedBetaPOInit).
    // Broadcast result to the rest of the workgroup.
    and_(1 | nz | f1[0], null, state.inputs.flags, FlagLeader);
    if (checkIfEnabled)
        jmpi(1 | f0[0], lNoScale);
    broadcastToWG(f1[0], state.betaCheckReturn, strategy, state, 0);

    // Check if our tile is completely outside C.
    int simt = strategy.fused ? 16 : 1;
    cmp(simt | le | f0[1], state.remainders[LoopM], uint16_t(0));
    cmp(simt | le | f1[1], state.remainders[LoopN], uint16_t(0));

    // Skip beta scaling if we weren't elected to the job.
    cmp(1 | ne | f1[0], state.betaCheckReturn, 0); /* Is beta scaling started?  */
    cmp(1 | gt | f0[0], state.betaCheckReturn, 0); /* Is beta scaling complete? */
    jmpi(1 | f1[0], lNoScale);

    state.ra.safeRelease(state.betaCheckReturn);

    or_(1, state.inputs.flags, state.inputs.flags, FlagDidBeta);

    // Skip if our tile is completely outside C.
    InstructionModifier cond = simt | f0[1] | anyv;

#ifdef STANDALONE
    if (problem.mock.cUpdate || problem.mock.cStore)
        jmpi(1, lScaleDone);
    else
#endif
    strategy.fused ? goto12(cond, lScaleDone)
                   :  ejmpi(cond, lScaleDone);

    // Adjust problem and strategy.
    bool beta0 = problem.beta0() || state.useTempC;
    auto &beta = problem.beta;
    bool checkBeta0 = !beta.fixed();

    setCCachingW(strategy, state, CacheSettingsLSC::L1UC_L3WB, CacheSettingsLSC::L1UC_L3WB, false);

    bool nested = true;
    std::swap(nested, state.isNested);

    if (!beta0) {
        if (checkBeta0) {
            auto vbetar = state.inputs.beta_real;
                cmp0(1 | eq | f0[1], vbetar.getReg(0));
            jmpi(1 | f0[1], lBeta0);
        }

        gemmAccessC(COperation::Load, problem, strategy, state);

        gemmBetaScale(problem, strategy, state);

        gemmAccessC(COperation::Store, problem, strategy, state);
        jmpi(1, lScaleDone);
        mark(lBeta0);
    }

    if (beta0 || checkBeta0)
        gemmStoreZeroC(problem, strategy, state);

    mark(lScaleDone);
    if (strategy.fused)
        join(simt);

    auto &lastCRange = state.C_regs[state.C_buffers - 1];
    auto lastC = lastCRange[lastCRange.getLen() - 1];

    useR0(state, [&](GRF r0_info) {
        // Zeroing C will synchronize on this but not in efficient 64-bit addressing mode.
        globalMemFence(lastC, r0_info, strategy);
        if (getEfficient64Bit())
            fencewait();
    });

    mark(lNoScale);
    if (strategy.fused)
        join(simt);

    /* Note: beta scaling may corrupt f0[0] but it doesn't matter.
       Flag also set if fused beta is disabled for this tile. */
    or_(1 | f0[0], state.inputs.flags, state.inputs.flags, FlagSkipBetaCheck);

    for (int i = 0; i < FlagRegister::count(hw); i++) {
        mov(1, FlagRegister(i).ud(), flagSave[i]);
        state.ra.safeRelease(flagSave[i]);
    }

    std::swap(nested, state.isNested);
}

// Notify other threads that beta scaling is complete for this tile.
template <HW hw>
void Generator<hw>::gemmFusedBetaNotifyCompletion(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    Label lSkipNotify;

    auto header = state.ra.alloc_range(2);
    auto data = state.ra.alloc().ud();
    auto &addr = state.statusFlagAddr;

    status << "Notify other workgroups that beta scaling is complete" << status_stream::endl;

    and_(1 | nz | state.flagAP, null.ud(), state.inputs.flags, FlagDidBeta);
    if (strategy.kParallelVariable)
        and_(1 | nz | f1[0], null.ud(), state.inputs.flags, FlagKPartitioned);
    jmpi(1 | ~state.flagAP, lSkipNotify);
    if (strategy.kParallelVariable)
        jmpi(1 | ~f1[0], lSkipNotify);

    useTempAndR0(state, [&](GRF temp, GRF r0_info) {
        if (strategy.altFusedBeta)
            globalMemFence(temp, r0_info, strategy);
        else
            slmfence(temp, r0_info);

        add(1 | sat, data.ud(0), state.fullK, -state.wgK);

        fencewait();
        activeThreadBarrierSignal(temp, r0_info, strategy);
    });

    and_(1 | nz | state.flagAP, null.ud(), state.inputs.flags, FlagLeader);
    emov(1, header[0].uq(0), addr, strategy, state);

    barrierwait();

    if (hw >= HW::XeHPG)
        store(1 | state.flagAP, D32 | CacheSettingsLSC::L1UC_L3WB, A64, header, data);
    else if (hw == HW::XeHP)
        atomic(AtomicOp::mov, 1 | state.flagAP, scattered_dword(), A64, header, data);
    else
        store(1 | state.flagAP, scattered_dword(), A64, header, data);     /* no L1 */

    state.ra.safeRelease(header);
    state.ra.safeRelease(data);

    mark(lSkipNotify);
}

// Wait for beta scaling to be complete.
template <HW hw>
void Generator<hw>::gemmFusedBetaWaitCompletion(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    Label lSkipCheck, lCheckAgain, lReady, lFinalDec, lCheckDone;
    bool checkIfEnabled = strategy.kParallelVariable;

    auto header = state.ra.alloc().uq();
    auto data = state.ra.alloc().d();
    auto &addr = state.statusFlagAddr;
    bool simtCF = strategy.fused;
    int simt = simtCF ? 16 : 1;

    status << "Wait for beta scaling" << status_stream::endl;

    and_(1 | nz | f1[0], null.ud(), state.inputs.flags, FlagDidBeta);
    if (checkIfEnabled)
        and_(1 | ze | f1[1], null.ud(), state.inputs.flags, FlagKPartitioned);
    and_(simt | nz | state.flagAP, null.ud(), state.inputs.flags, FlagLeader);
    if (strategy.fuseBeta && !strategy.altFusedBeta)
        and_(1 | nz | f0[1], null.ud(), state.inputs.flags, FlagSkipBetaCheck);

    emov(1, header, addr, strategy, state);
    mov(1, data, -state.wgK);

    jmpi(1 | f1[0], lSkipCheck);            /* If we did beta scaling ourselves, no need to check */
    if (checkIfEnabled)
        jmpi(1 | f1[1], lSkipCheck);        /* C tile isn't shared; no need to check */
    simtCF ? goto12(16 | ~state.flagAP, lCheckDone)    /* Followers wait for leader to check */
           :   jmpi(1  | ~state.flagAP, lCheckDone);
    if (strategy.fuseBeta && !strategy.altFusedBeta)
        jmpi(1 | f0[1], lFinalDec);         /* If beta scaling known to be complete, just decrement counter */

    if (hw >= HW::XeHPG)
        atomic(AtomicOp::add, 1, data, D32 | CacheSettingsLSC::L1UC_L3C, A64, header, data);
    else
        atomic(AtomicOp::add, 1, data, scattered_dword(), A64, header, data);

    cmp(simt | gt | state.flagAP, data.d(0), 0);
    simtCF ? goto12(16 | state.flagAP, lCheckDone)
           :   jmpi(1  | state.flagAP, lCheckDone);

    mark(lCheckAgain);

    if (hw >= HW::XeHPG)
        load(1, data, D32 | CacheSettingsLSC::L1UC_L3C, A64, header);
    else if (hw >= HW::XeHP) {
        mov(1, data, 0);
        atomic(AtomicOp::or_, 1, data, scattered_dword(), A64, header, data);
    } else
        load(1, data, scattered_dword(), A64, header);     /* no L1 */

    cmp(simt | gt | state.flagAP, data.d(0), 0);
    simtCF ? goto12(16 | state.flagAP, lReady)
           :   jmpi(1  | state.flagAP, lReady);

    pause(strategy);
    jmpi(1, lCheckAgain);

    mark(lReady);
    mov(1, data, -state.wgK);

    mark(lFinalDec);
    if (simtCF) join(16);

    if (hw >= HW::XeHPG)
        atomic(AtomicOp::add, 1, D32 | CacheSettingsLSC::L1UC_L3C, A64, header, data);
    else
        atomic(AtomicOp::add, 1, scattered_dword(), A64, header, data);

    mark(lCheckDone);
    if (simtCF) join(16);

    useTempAndR0(state, [&](GRF temp, GRF r0_info) {
        activeThreadBarrier(temp, r0_info, strategy);
    });

    mark(lSkipCheck);

    state.ra.safeRelease(header);
    state.ra.safeRelease(data);
}

// Swap out C for the temporary buffer.
template <HW hw>
void Generator<hw>::gemmRedirectToTempC(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    problem.Tc_ext = problem.Tc;
    problem.C = state.tempC;

    if (problem.cMXScale) stub("MX Scale unsupported with Temp C.");

    strategy.C = state.tempCStrategy;
    strategy.remHandling[LoopM] = RemainderHandling::Ignore;
    strategy.remHandling[LoopN] = RemainderHandling::Ignore;

    state.effC[0] = state.effTempC;
    state.C_layoutExt = state.C_layout;
    state.C_layoutExt.retarget(problem.C, strategy.C);
    state.C_layoutExtUnmasked.clear();
    state.C_layoutExtNonatomicUnmasked.clear();
    state.inputs.ldc[0] = invalid;
    state.copyC = false;

    if (problem.sumA || problem.sumB) {
        problem.Tco = problem.Tc;
        problem.CO.setAlignment(64);
        strategy.CO.base = strategy.C.base;
        strategy.CO.padded = true;
        state.effCO = state.ra.alloc_sub(state.effTempC.getType());
        eadd(1, state.effCO, state.effTempC, strategy.unroll[LoopM] * strategy.unroll[LoopN] * problem.Tc, strategy, state);
    }
}

// Handle accumulation of C data from multiple WGs prior to post-op calculation.
template <HW hw>
bool Generator<hw>::gemmFusedPostOpsFinalize(Label &labelLateExit, GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto doLateExit = [&](Label &target) {
        if (state.doLateExit) gemmOOBExit(target, strategy, state);
    };

    /* Check if we need to accumulate data from multiple WGs */
    Label labelNoSlicing, labelSkipCUpdate, labelFPODone;
    if (strategy.kParallelVariable) {
        and_(1 | nz | f0[0], null.ud(), state.inputs.flags, FlagKPartitioned);
        jmpi(1 | ~f0[0], labelNoSlicing);
    }

    auto modProblem = problem;
    auto modStrategy = strategy;
    auto modState = state;

    modProblem.beta = 1;
#ifndef STANDALONE
    modProblem.postOps = PostOps{};
#endif
    if (modProblem.cOffset == COffset::Post)
        modProblem.cOffset = COffset::None;

    modState.isNested = true;

    if (state.useTempC)
        gemmRedirectToTempC(modProblem, modStrategy, modState);

    doLateExit(labelSkipCUpdate);

    /* Accumulate C buffer + fence */
    status << "Accumulate local contribution to C tile" << status_stream::endl;
    bool alphaScale = !state.useTempC;
    if (alphaScale)
        gemmAlphaScale(modProblem, modStrategy, modState);
    else
        modProblem.alpha = 1;

    if (strategy.altFusedBeta) {
        Label lTileAccumulate;
        auto strategy0 = modStrategy;
        auto state0 = modState;

        if (!problem.beta0() && !state.useTempC) stub();    /* TODO: implement beta scaling */

        and_(1 | nz | state.flagAP, null.ud(), state.inputs.flags, FlagDidBeta);
        jmpi(1 | ~state.flagAP, lTileAccumulate);
        setCCachingW(strategy0, state0, CacheSettingsLSC::L1UC_L3WB, CacheSettingsLSC::L1UC_L3WB, false);
        if (!gemmAccessC(COperation::Store, modProblem, strategy0, state0)) return false;
        jmpi(1, labelSkipCUpdate);
        mark(lTileAccumulate);
    }

    if (!gemmAccessC(COperation::UpdateStore, modProblem, modStrategy, modState)) return false;
    mark(labelSkipCUpdate);
    if (strategy.fused)
        join(16);

    useTempAndR0(modState, [&](GRF temp, GRF r0_info) {
        globalMemBarrier(temp, r0_info, strategy);
    });

    if (strategy.altFusedBeta) {
        cmp(1 | eq | f0[1], state.wgK, 0);
        gemmFusedBetaNotifyCompletion(problem, strategy, state);
        jmpi(1 | f0[1], labelLateExit);     /* Zero-length chunks do not participate in post-ops. */
    }

    /* Leader adds our k slice to the total "k done" counter and broadcasts to WG.
        Consider splitting counter and doing this check earlier to absorb latency. */
    auto header = modState.ra.alloc_range(2);
    auto kDone = modState.ra.alloc().ud();
    auto zero = modState.ra.alloc().ud();
    auto temp = header[1].ud(0);

    status << "Check if post-ops need to be applied" << status_stream::endl;

    and_(1 | nz | f0[0], null.ud(), state.inputs.flags, FlagLeader);
    mov(1 | sat, kDone, state.wgK);
    eadd<uint64_t>(1, header, state.statusFlagAddr, strategy.fuseBeta ? 64 : 0, strategy, state);
    if (hw >= HW::XeHPG)
        atomic(AtomicOp::add, 1 | f0[0], kDone, D32, A64, header, kDone);
    else
        atomic(AtomicOp::add, 1 | f0[0], kDone, scattered_dword(), A64, header, kDone);

    broadcastToWG(f0[0], kDone, modStrategy, modState, 4);

    if (std::max(strategy.slmBuffersA, strategy.slmBuffersB) > 0 || strategy.kParallelLocal) {
        useTempAndR0(modState, [&](GRF temp, GRF r0_info) {
            slmBarrier(temp, r0_info, strategy);
        });
    }

    doLateExit(labelLateExit);

    and_(1 | nz | f0[0], null.ud(), state.inputs.flags, FlagLeader);

    /* If full k range has been accumulated, it's our responsibility to do post-ops. */
    add(1, temp, state.fullK, -state.wgK);          mov(1, zero, 0);
    cmp(1 | lt | f1[0], kDone, temp);               and_(1, state.inputs.flags, state.inputs.flags, ~uint32_t(FlagNonfinalKBlock));
    jmpi(1 | f1[0], labelLateExit);

    /* Reset counter to zero to clean up after ourselves. */
    if (hw >= HW::XeHPG)
        store(1 | f0[0], D32, A64, header, zero);
    else
        store(1 | f0[0], scattered_dword(), A64, header, zero);

    modState.ra.safeRelease(header);
    modState.ra.safeRelease(kDone);
    modState.ra.safeRelease(zero);

    status << "Load completed C tile" << status_stream::endl;
    if (hw < HW::Xe2) {
        /* Xe2 + later need atomic loads */
        modStrategy.C.atomic = modStrategy.CO.atomic = false;
        modState.Cext_strategy.atomic = false;
    }
    gemmAccessC(COperation::Load, modProblem, modStrategy, modState);

    if (strategy.zeroTempC) {
        status << "Reset temporary memory to zero" << status_stream::endl;
        modStrategy.altCRemainder = false;
        gemmStoreZeroC(modProblem, modStrategy, modState, false);
    }

    jmpi(1, labelFPODone);

    if (strategy.kParallelVariable) {
        /* If not slicing, perform alpha scaling and late exit to match sliced path. */
        mark(labelNoSlicing);
        doLateExit(labelLateExit);
        if (alphaScale)
            gemmAlphaScale(problem, strategy, state);
    }
    mark(labelFPODone);

    /* Update problem to reflect remaining post-op work. */
    if (alphaScale)
        problem.alpha = 1;
    if (!strategy.kParallelVariable && !state.useTempC)
        problem.beta = 0;

    bool disableAtomics = (problem.beta.fixed() && !problem.beta1())
                        || (!useAutoAtomic(hw, problem, strategy));
    if (disableAtomics) {
        strategy.C.atomic = strategy.CO.atomic = false;
        state.Cext_strategy.atomic = false;
    }

    return true;
}

GEMMSTONE_NAMESPACE_END
