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
#include "kernel_queries.hpp"
#include "gemmstone/generator.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


static inline void sysgemmSetupCLayouts(HW hw, const GRFMultirange &C_regs, int accStride, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state);

/**********************************************************************/
/*                 Fixed Systolic GEMM (XeHP/XeHPG)                   */
/**********************************************************************/
namespace sysgemm {
    static GRFRange A_copy0 = GRF(40)-GRF(47);
    static GRFRange B_copy0 = GRF(2)-GRF(13);
    static GRFRange A_regs  = GRF(48)-GRF(63);
    static GRFRange B_regs  = GRF(14)-GRF(37);
    static GRFRange C_regs  = GRF(64)-GRF(255);
    static GRFRange A_copy1 = GRF(96)-GRF(103);
    static GRFRange B_copy1 = GRF(104)-GRF(111);
    static GRFRange A_copy2 = GRF(144)-GRF(151);
    static GRFRange B_copy2 = GRF(152)-GRF(159);
    static GRFRange A_copy[3] = {A_copy0, A_copy1, A_copy2};
    static GRFRange B_copy[3] = {B_copy0, B_copy1, B_copy2};
    static GRF addr0 = GRF(1);
    static GRF addr1 = GRF(38);
    static GRF addr2 = GRF(39);
    static GRF addr3 = GRF(0);
    static Subregister A_ptr64 = addr1.uq(3);
    static Subregister B_ptr64 = addr2.uq(3);
    static Subregister C_ptr64 = addr2.uq(2);
    static Subregister slmAOffsetLoad = addr1.uw(8); // offsets in OWords
    static Subregister slmBOffsetLoad = addr1.uw(9);
    static Subregister slmAOffsetStore = addr1.uw(10);
    static Subregister slmBOffsetStore = addr1.uw(11);
    static Subregister slmAOffsetLoadInit = addr1.uw(6);
    static Subregister slmBOffsetLoadInit = addr1.uw(7);
    static Subregister slmAOffsetStoreInit = addr2.uw(6);
    static Subregister slmBOffsetStoreInit = addr2.uw(7);
    static Subregister kCounter = AccumulatorRegister(2).d(0);
    static Subregister barrierVal = AddressRegister(0).ud(0);
    static constexpr int accStride = 48;
}

template <HW hw>
bool Generator<hw>::sysgemmAccumulateC(GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    using namespace sysgemm;
    auto params = systolicParams(problem);
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto wgM = strategy.wg[LoopM];
    auto wgN = strategy.wg[LoopN];
    auto localIDM = state.lidM;
    auto localIDN = state.lidN;
    bool doubleM = (wgM == 8);
    bool surfaceAB = !strategy.A.base.isStateless();
    bool surfaceC = !strategy.C.base.isStateless();

    if (unrollM != 32) stub();
    if (unrollN != 32 && unrollN != 48) stub();
    if (wgM != 4 && wgM != 8) stub();
    if (wgN != 4) stub();
    if (strategy.A.base.getModel() != strategy.B.base.getModel()) stub();
    if (problem.A.layout != MatrixLayout::Pc) stub();
    if (problem.A.crosspack != params.opsPerChan) stub();
    if (problem.A.tileR != params.osys) stub();
    if (problem.A.tileC != params.ksys) stub();
    if (problem.B.layout != MatrixLayout::Pr) stub();
    if (problem.B.crosspack != params.ksys) stub();
    if (problem.B.tileR != 0 || problem.B.tileC != 0) stub();

    state.ra.claim(C_regs);

    // Adjust A/B addresses and SLM offsets.
    auto tempStorage = C_regs[0];
    auto suboffsetA = tempStorage.ud(0);
    auto suboffsetB = tempStorage.ud(1);
    auto tempA = tempStorage.ud(2);
    auto wlidM = tempStorage.uw(6);
    auto tempB = tempStorage.ud(4);
    auto suboffsetBl = tempStorage.ud(5);

    if (doubleM) {
        and_(1, wlidM, localIDM, 3);
        and_(1 | ne | f1[1], null.uw(), localIDM, 4);
    }
    and_(1 | ne | state.flagAP, null.uw(), localIDM, 1);
    mulConstant(1, suboffsetA, localIDN, unrollM * (32 / 4));
    if (doubleM) {
        mulConstant(1, suboffsetB, wlidM, unrollN * (32 / 4));
        mulConstant(1, suboffsetBl, localIDM, unrollN * (32 / 4));
    } else {
        mulConstant(1, suboffsetB, localIDM, unrollN * (32 / 4));
        suboffsetBl = suboffsetB;
    }

    auto A_ptr = A_ptr64, B_ptr = B_ptr64, C_ptr = C_ptr64;
    if (surfaceAB) {
        if (!strategy.A.newDP || !strategy.B.newDP) stub();
        A_ptr = A_ptr.ud();
        B_ptr = B_ptr.ud();
    }
    if (surfaceC) C_ptr = C_ptr.ud();

    eadd(1, A_ptr, state.effA, suboffsetA, strategy, state);
    eadd(1, B_ptr, state.effB, suboffsetBl, strategy, state);
    emov(1, C_ptr, state.effC[0], strategy, state);

    shr(2, suboffsetA(1), suboffsetA(1), 4);

    mul(1, tempA, localIDM, (unrollM * 36) / 16);
    mad(1, tempB, (wgM * unrollM * 36) / 16, localIDN, (unrollN * 32) / 16);

    mov(1, slmAOffsetLoadInit.uw(), tempA.uw());
    add(1 | state.flagAP, slmBOffsetLoadInit.uw(), tempB.uw(), (unrollN / 2) * (32 / 16));
    mov(1 | ~state.flagAP, slmBOffsetLoadInit.uw(), tempB.uw());
    add(1, slmAOffsetStoreInit.uw(), tempA.uw(), suboffsetA.uw());
    add(1, slmBOffsetStoreInit.uw(), tempB.uw(), suboffsetB.uw());
    mov(2, slmAOffsetLoad(1), slmAOffsetLoadInit(1));

    // Marshal data needed later into acc2 for safekeeping.
    auto saveData = state.ra.alloc_range(2);
    auto kLoops = saveData[0].d(0);
    auto ldc    = saveData[0].ud(1);
    auto flags  = saveData[0].ud(2);
    auto k      = saveData[0].ud(3);
    auto remM   = saveData[0].uw(8);
    auto remN   = saveData[0].uw(9);
    auto abo    = saveData[0].ud(5);
    auto ao     = saveData[0].w(10);
    auto bo     = saveData[0].w(11);
    auto alpha  = saveData[0].ud(6).reinterpret(0, problem.Ts.ngen());
    auto beta   = saveData[0].ud(7).reinterpret(0, problem.Ts.ngen());
    auto remFusedStorage = saveData[1].ud(0);
    auto diagC  = saveData[1].ud(1);
    auto effCO  = saveData[1].uq(1);
    auto ldco   = saveData[1].ud(5);
    auto effAs  = saveData[1].uq(2).reinterpret(0, state.effA.getType());
    auto effBs  = saveData[1].uq(3).reinterpret(0, state.effB.getType());
    auto saveI0 = saveData[1].ud(1);
    auto saveJ0 = saveData[1].ud(3);

    if (state.r0_info != acc0.ud())
        mov<uint32_t>(8, acc0, state.r0_info);

    add(1, kLoops, state.k, params.ksys - 1);
    mov(1, ldc, state.inputs.ldc[0]);
    if (state.inputs.flags.isValid())
        mov(1, flags, state.inputs.flags);
    mov(1, k, state.k);
    if (state.remainders[LoopM].isValid())
        mov(1, remM, state.remainders[LoopM]);
    if (state.remainders[LoopN].isValid())
        mov(1, remN, state.remainders[LoopN]);
    if (state.inputs.abo.isValid())
        mov(1, abo, state.inputs.abo);
    else {
        if (state.inputs.ao.isValid())
            mov(1, ao, state.inputs.ao);
        if (state.inputs.bo.isValid())
            mov(1, bo, state.inputs.bo);
    }
    if (state.inputs.alpha_real.isValid())
        mov(1, alpha, state.inputs.alpha_real);
    if (state.inputs.beta_real.isValid())
        mov(1, beta, state.inputs.beta_real);
    shr(1, kLoops, kLoops, ilog2(params.ksys));
    if (state.remFusedStorage.isValid())
        mov(1, remFusedStorage, state.remFusedStorage);
    if (state.diagC.isValid())
        mov(1, diagC, state.diagC);
    if (state.effCO.isValid()) {
        effCO = effCO.reinterpret(0, state.effCO.getType());
        emov(1, effCO, state.effCO, strategy, state);
        if (state.inputs.ldco.isValid())
            mov(1, ldco, state.inputs.ldco);
    }
    if (problem.hasBinaryPostOp()) {
        if (state.diagC.isValid()) stub();
        if (state.effCO.isValid() && effCO.getBytes() > 4) stub();
        mov(1, saveI0, state.i0);
        mov(1, saveJ0, state.j0);
    }
    bool abOffset = (problem.aOffset != ABOffset::None) || (problem.bOffset != ABOffset::None);
    if (abOffset) {
        state.effAs = effAs;
        state.effBs = effBs;
        gemmCalcABOffsetAddrs(problem, strategy, state);
    }

    releaseSavedMNLocalIDs(state);
    state.ra.safeRelease(state.effA);
    state.ra.safeRelease(state.effB);
    state.ra.safeRelease(state.effC[0]);
    state.ra.safeRelease(state.inputs.lda);
    state.ra.safeRelease(state.inputs.ldb);

    state.ra.release(state.inputs.ldc[0]);
    state.ra.release(state.k);
    state.ra.release(state.remainders[LoopM]);
    state.ra.release(state.remainders[LoopN]);
    state.ra.release(state.inputs.abo);
    state.ra.release(state.inputs.ao);
    state.ra.release(state.inputs.bo);
    state.ra.release(state.inputs.alpha_real);
    state.ra.release(state.inputs.beta_real);
    state.ra.release(state.remFusedStorage);
    state.ra.release(state.diagC);
    state.ra.release(state.effCO);
    state.ra.release(state.inputs.ldco);

    if (state.r0_info.isARF()) stub();
    GRF r0_info{state.r0_info.getBase()};
    if (hw >= HW::XeHPG) {
        mov(1, barrierVal.uw(0), Immediate::uw(0));
        mov(2, barrierVal.ub(2)(1), r0_info.ub(11)(0));
    } else
        and_(1, barrierVal, r0_info.ud(2), 0x7F000000);

    mov<float>(16, acc2, saveData[0]);

    sync.nop(SWSB<AllPipes>(1));

    if (!doubleM)
        sysgemmKLoop(problem, strategy, state);
    else {
        Label oddB, done;
        jmpi(1 | f1[1], oddB);
        sysgemmKLoop4(problem, strategy, state, false);
        jmpi(1, done);
        mark(oddB);
        sysgemmKLoop4(problem, strategy, state, true);
        mark(done);
    }

    mov<float>(16, saveData[0], acc2);

    state.effC[0] = C_ptr;
    state.inputs.ldc[0] = ldc;
    if (state.inputs.flags.isValid())
        state.inputs.flags = flags;
    state.k = k;
    if (state.remainders[LoopM].isValid())
        state.remainders[LoopM] = remM;
    if (state.remainders[LoopN].isValid())
        state.remainders[LoopN] = remN;
    if (state.inputs.abo.isValid())
        state.inputs.abo = abo;
    if (state.inputs.ao.isValid())
        state.inputs.ao = ao;
    if (state.inputs.bo.isValid())
        state.inputs.bo = bo;
    if (state.inputs.alpha_real.isValid())
        state.inputs.alpha_real = alpha;
    if (state.inputs.beta_real.isValid())
        state.inputs.beta_real = beta;
    if (state.remFusedStorage.isValid()) {
        state.remFusedStorage = remFusedStorage;
        state.remaindersFused[LoopM] = state.remainders[LoopM];
        state.remaindersFused[LoopN] = state.remainders[LoopN];
        state.remaindersFused[strategy.fusedLoop] = remFusedStorage;
    }
    if (state.diagC.isValid())
        state.diagC = diagC;
    if (state.effCO.isValid())
        state.effCO = effCO;
    if (state.inputs.ldco.isValid())
        state.inputs.ldco = ldco;
    if (problem.hasBinaryPostOp()) {
        state.i0 = saveI0;
        state.j0 = saveJ0;
    }

    state.ra.claim(C_ptr);

    sysgemmSetupCLayouts(hw, C_regs, accStride, problem, strategy, state);

    if (state.r0_info != acc0.ud())
        mov<uint32_t>(8, state.r0_info, acc0);

    return true;    // Success!
}

inline void sysgemmSetupCLayouts(HW hw, const GRFMultirange &C_regs, int accStride, const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto unrollM = strategy.unroll[LoopM], unrollN = strategy.unroll[LoopN];

    // Set up C internal layout and registers.
    state.C_regs.resize(1);
    state.C_regs[0] = C_regs;

    vector<RegisterBlock> C_list;
    C_list.reserve((unrollM / 8) * (unrollN / 4));
    for (int j0 = 0; j0 < unrollN; j0 += 4) {
        for (int i0 = 0; i0 < unrollM; i0 += 8) {
            RegisterBlock block;
            block.hw = hw;
            block.colMajor = true;
            block.splitComplex = false;
            block.cxComponent = RegisterBlock::Interleaved;
            block.nr = block.ld = 8;
            block.nc = 4;
            block.component = 0;
            block.offsetR = i0;
            block.offsetC = j0;
            block.crosspack = 1;
            block.bytes = 8 * 4 * problem.Tc;
            block.simdSize = 0;

            int j0Interleaved = j0 << 1;
            if (j0Interleaved >= unrollN)
                j0Interleaved += 4 - unrollN;

            block.offsetBytes = (accStride * i0 / 8 + j0Interleaved) * GRF::bytes(hw);
            C_list.push_back(block);
        }
    }

    state.C_layout = RegisterLayout(problem.Tc, std::move(C_list));

    // Set up C external layout.
    state.copyC = true;
    bool remM_Ce, remN_Ce;
    getCRemainders(hw, problem, strategy, remM_Ce, remN_Ce);

    state.C_layoutExt = RegisterLayout(hw, problem.Tc_ext, unrollM, unrollN, problem.C, state.Cext_strategy, remM_Ce, remN_Ce, true, AllowFragDesc);
    if (remM_Ce || remN_Ce)
        state.C_layoutExtUnmasked = RegisterLayout(hw, problem.Tc_ext, unrollM, unrollN, problem.C, state.Cext_strategy, false, false, true, AllowFragDesc);
}

template <HW hw>
void Generator<hw>::sysgemmKLoop(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    using namespace sysgemm;
    Label top, bottom, skipMain, remTop, remBottom;

    auto nbBarrierWait = [&]() {
        if (!strategy.slmAltBarriers)
            barrierwait();
    };
    auto nbStoreSignal = [&](bool forceFence = false) {
        if (!strategy.slmAltBarriers)
            sysgemmStoreSignal(problem, strategy, state, forceFence);
    };
    auto storeSignal = [&](bool forceFence = false) {
        sysgemmStoreSignal(problem, strategy, state, forceFence);
    };
    auto copyLoad = [&](int storeBuffer, bool useC = false) {
        sysgemmCopyLoad(problem, strategy, state, storeBuffer, useC);
    };
    auto copyStore = [&](int storeBuffer, bool first = false) {
        sysgemmCopyStore(problem, strategy, state, storeBuffer, first);
    };
    auto multiply = [&](int buffer, bool lastMultiply = false) {
        sysgemmMultiply(problem, strategy, state, buffer, lastMultiply);
    };

    bool oldDefaultAutoSWSB = getDefaultAutoSWSB();
    setDefaultAutoSWSB(false);

    if (strategy.slmCopies == 1) {
        cmp(1 | lt | f1[1], kCounter, 3);
        add(1 | le | f0[1], kCounter, kCounter, -5);

        jmpi(1 | f1[1], skipMain);

        copyLoad(0, true);                       // L0 -> C
        copyLoad(1);                             // L1
        copyStore(0, true);                      // S0 <- C
        storeSignal(true);                       // Signal 0 ready
        zeroMatrix(C_regs, strategy);
        sync.nop(SWSB<AllPipes>(1));
        copyStore(1);                            // S1

        nbBarrierWait();                         // Wait 0 ready
        nbStoreSignal();                         // Signal 1 ready

        jmpi(1 | f0[1], bottom);                 // Zero-trip loop check

        mark(top);
        add(1 | gt | f0[1], kCounter, kCounter, -3);

        copyLoad(2);                             // L2
        multiply(0);                             // M0
        nbBarrierWait();                         // Wait 0 ready
        copyStore(2);                            // S2
        nbStoreSignal();                         // Signal 2 ready

        copyLoad(0);                             // L0
        multiply(1);                             // M1
        nbBarrierWait();                         // Wait 2 ready
        copyStore(0);                            // S0
        nbStoreSignal();                         // Signal 0 ready

        copyLoad(1);                             // L1
        multiply(2);                             // M2
        nbBarrierWait();                         // Wait 0 ready
        copyStore(1);                            // S1
        nbStoreSignal();                         // Signal 1 ready

        jmpi(1 | f0[1], top);
        mark(bottom);

        copyLoad(2);                             // L2
        multiply(0);                             // M0
        nbBarrierWait();                         // Wait 1 ready
        copyStore(2);                            // S2
        nbStoreSignal();                         // Signal 2 ready

        multiply(1);                             // M1

        nbBarrierWait();                         // Wait 2 ready

        multiply(2, true);                       // M2

        add(1 | le | f0[1], kCounter, kCounter, 2);
        jmpi(1 | f0[1], remBottom);
        jmpi(1, remTop);

        mark(skipMain);

        zeroMatrix(C_regs, strategy);
        add(1, kCounter, kCounter, 5);

        mov(2, slmAOffsetStore(1), slmAOffsetStoreInit(1));
        sync.nop(SWSB<AllPipes>(1));

        mark(remTop);

        cmp(1 | lt | f0[1], kCounter, 2);
        copyLoad(0);
        copyStore(0);
        storeSignal(true);
        nbBarrierWait();
        multiply(0, true);

        jmpi(1 | f0[1], remBottom);
        copyLoad(1);
        copyStore(1);
        storeSignal(true);
        nbBarrierWait();
        multiply(1, true);

        mark(remBottom);
    } else if (strategy.slmCopies == 3) {
        // Triple-buffered global memory load + SLM pipeline.
        cmp(1 | lt | f1[1], kCounter, 4);
        add(1 | le | f0[1], kCounter, kCounter, -6);

        jmpi(1 | f1[1], skipMain);

        copyLoad(0);                             // L0
        copyLoad(1);                             // L1
        copyLoad(2);                             // L2
        copyStore(0, true);                      // S0
        storeSignal(true);                       // Signal 0 ready
        zeroMatrix(C_regs, strategy);
        copyLoad(0);                             // L0
        sync.nop(SWSB<uint32_t>(1));
        copyStore(1);                            // S1

        nbBarrierWait();                         // Wait 0 ready
        nbStoreSignal();                         // Signal 1 ready

        jmpi(1 | f0[1], bottom);                 // Zero-trip loop check

        mark(top);
        add(1 | gt | f0[1], kCounter, kCounter, -3);

        copyLoad(1);                             // L1
        multiply(0);                             // M0
        nbBarrierWait();                         // Wait 0 ready
        copyStore(2);                            // S2
        nbStoreSignal();                         // Signal 2 ready

        copyLoad(2);                             // L2
        multiply(1);                             // M1
        nbBarrierWait();                         // Wait 2 ready
        copyStore(0);                            // S0
        nbStoreSignal();                         // Signal 0 ready

        copyLoad(0);                             // L0
        multiply(2);                             // M2
        nbBarrierWait();                         // Wait 0 ready
        copyStore(1);                            // S1
        nbStoreSignal();                         // Signal 1 ready

        jmpi(1 | f0[1], top);
        mark(bottom);

        multiply(0);                             // M0
        nbBarrierWait();                         // Wait 1 ready
        copyStore(2);                            // S2
        nbStoreSignal();                         // Signal 2 ready

        multiply(1);                             // M1
        nbBarrierWait();                         // Wait 2 ready
        copyStore(0);                            // S0
        nbStoreSignal();                         // Signal 0 ready

        multiply(2);                             // M2

        nbBarrierWait();                         // Wait 0 ready

        multiply(0, true);                       // M0

        add(1 | le | f0[1], kCounter, kCounter, 2);
        jmpi(1 | f0[1], remBottom);
        jmpi(1, remTop);

        mark(skipMain);

        zeroMatrix(C_regs, strategy);
        add(1 | le | f0[1], kCounter, kCounter, 5);

        mov(2, slmAOffsetStore(1), slmAOffsetStoreInit(1));
        sync.nop(SWSB<uint32_t>(1));

        copyLoad(0);
        copyStore(0);
        storeSignal(true);
        nbBarrierWait();
        multiply(0, true);

        jmpi(1 | f0[1], remBottom);

        mark(remTop);

        cmp(1 | lt | f0[1], kCounter, 2);

        copyLoad(1);
        copyStore(1);
        storeSignal(true);
        nbBarrierWait();
        multiply(1, true);

        jmpi(1 | f0[1], remBottom);

        copyLoad(2);
        copyStore(2);
        storeSignal(true);
        nbBarrierWait();
        multiply(2, true);

        mark(remBottom);
    } else stub();

    sync.allwr();
    setDefaultAutoSWSB(oldDefaultAutoSWSB);
}

template <HW hw>
void Generator<hw>::sysgemmKLoop4(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool oddB)
{
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    Label top, bottom, skipMain, done;
    Label skipLoad0, skipLoad1, skipLoad2;
    Label skipStore0, skipStore1, skipStore2;
    Label sskipLoad12, sskipStore1, sskipStore2Load3, sskipStore3;

    auto clearDepAddr = [&]() {
        for (int i = 0; i < 4; i++) depAddr[i] = InstructionModifier();
    };
    auto storeSignal = [&]() {
        sysgemmStoreSignal(problem, strategy, state, true);
    };
    auto copyLoad = [&](int storeBuffer, int useC = 0, bool forceLoadB = false, RegData flagLoadB = RegData()) {
        sysgemmCopyLoad4(problem, strategy, state, storeBuffer, ((storeBuffer & 1) != oddB) || forceLoadB, useC, flagLoadB);
    };
    auto copyLoadRem = [&](int storeBuffer, RegData flagLoadB) {
        sysgemmCopyLoad4(problem, strategy, state, storeBuffer, (storeBuffer & 1) != oddB, 0, flagLoadB);
    };
    auto copyStore = [&](int storeBuffer, int useC = 0, int useC_B = 0) {
        sysgemmCopyStore4(problem, strategy, state, storeBuffer, (storeBuffer & 1) == oddB, useC, useC_B);
    };
    auto multiply = [&](int buffer, bool firstMultiply = false) {
        sysgemmMultiply4(problem, strategy, state, buffer, firstMultiply);
    };
    auto multiplyRem = [&](int buffer, RegData flagWaitLoad, RegData flagSignal, bool firstMultiply = false) {
        sysgemmMultiply4(problem, strategy, state, buffer, firstMultiply, flagWaitLoad, flagSignal, &done);
    };
    auto slmRead = [&]() {
        mov(1 | depAddr[0], addr0.ud(2), slmAOffsetLoad);
        mov(1 | depAddr[1], addr1.ud(2), slmBOffsetLoad);
        add(1 | depAddr[2], addr2.ud(2), slmBOffsetLoad, 8 * 32 / 16);
        add(1 | depAddr[3], addr3.ud(2), slmBOffsetLoad, 16 * 32 / 16);

        load(16 | SWSB<AllPipes>(sb3, 4), A_regs[0], block_oword(16), SLM, addr0);
        load(16 | SWSB<AllPipes>(sb0, 3), B_regs[0], block_oword(16), SLM, addr1);
        load(16 | SWSB<AllPipes>(sb1, 2), B_regs[8], block_oword(16), SLM, addr2);
        load(16 | SWSB<AllPipes>(sb2, 1), B_regs[16], block_oword(16), SLM, addr3);
        depAddr[0] = sb3.src;
        depAddr[1] = sb0.src;
        depAddr[2] = sb1.src;
        depAddr[3] = sb2.src;

        add(1 | depAddr[0], addr0.ud(2), slmAOffsetLoad, 8 * 32 / 16);
        load(16 | SWSB<AllPipes>(sb4, 1), A_regs[8], block_oword(16), SLM, addr0);
        depAddr[0] = sb4.src;
    };

    bool oldDefaultAutoSWSB = getDefaultAutoSWSB();
    setDefaultAutoSWSB(false);

    clearDepAddr();
    mov(1, f1.ud(), 0);
    mov(1, f0.ud(), 0);
    cmp(1 | lt | f1[1], kCounter, 4);
    add(1 | le | f0[1], kCounter, kCounter, -7);

    jmpi(1 | f1[1], skipMain);

    status << "Main path, " << (oddB ? "odd B" : "even B") << status_stream::endl;

    copyLoad(0, 1, true);                    // L0 -> C1
    copyLoad(1, 2);                          // L1 -> C2
    copyLoad(2);                             // L2
    copyStore(0, 1, 1);                      // S0 <- C1
    storeSignal();
    copyStore(1, 2, 1);                      // S1 <- C2
    barrierwait();
    slmRead();
    storeSignal();
    copyStore(2, 0, 2);                      // S2
    if (!oddB) sync.allrd(0x3000);
    zeroMatrix(C_regs, strategy);
    sync.allrd(SWSB<AllPipes>(1));

    jmpi(1 | f0[1], bottom);                 // Zero-trip loop check

    depAddr[0] = sb8.src;
    depAddr[1] = !oddB ? sb9.src : sb0.src;
    depAddr[2] = !oddB ? sb10.src : sb4.src;
    depAddr[3] = sb3.src;

    mark(top);
    add(1 | gt | f0[1], kCounter, kCounter, -4);

    copyLoad(3);
    multiply(0);
    copyStore(3);

    copyLoad(0);
    multiply(1);
    copyStore(0);

    copyLoad(1);
    multiply(2);
    copyStore(1);

    copyLoad(2);
    multiply(3);
    copyStore(2);

    jmpi(1 | f0[1], top);
    mark(bottom);

    cmp(1 | gt | f0[0], kCounter, -4 + 1);
    cmp(1 | gt | f0[1], kCounter, -4 + 2);
    cmp(1 | gt | f1[0], kCounter, -4 + 3);

    copyLoadRem(3, f0[0]);
    multiply(0);
    copyStore(3);

    sync.allrd();
    jmpi(1 | ~f0[0], skipLoad0);
    copyLoadRem(0, f0[1]);
    mark(skipLoad0);
    multiply(1);
    sync.allrd();
    jmpi(1 | ~f0[0], skipStore0);
    copyStore(0);
    mark(skipStore0);

    sync.allrd();
    jmpi(1 | ~f0[1], skipLoad1);
    copyLoadRem(1, f1[0]);
    mark(skipLoad1);
    multiplyRem(2, FlagRegister(), f0[0]);
    sync.allrd();
    jmpi(1 | ~f0[1], skipStore1);
    copyStore(1);
    mark(skipStore1);

    sync.allrd();
    jmpi(1 | ~f1[0], skipLoad2);
    copyLoadRem(2, null);
    mark(skipLoad2);
    multiplyRem(3, f0[0], f0[1]);
    sync.allrd();
    jmpi(1 | ~f1[0], skipStore2);
    copyStore(2);
    mark(skipStore2);

    multiplyRem(0, f0[1], f1[0]);
    multiplyRem(1, f1[0], null);
    multiplyRem(2, null, null);

    jmpi(1, done);

    status << "Small-k path, " << (oddB ? "odd B" : "even B") << status_stream::endl;

    clearDepAddr();
    mark(skipMain);

    // Short loops: special case for 1-4 unrolls
    cmp(1 | gt | f0[0], kCounter, -7 + 1);
    cmp(1 | gt | f0[1], kCounter, -7 + 2);
    cmp(1 | gt | f1[0], kCounter, -7 + 3);

    auto flagLoadB0 = oddB ? f0[0] : FlagRegister();
    copyLoad(0, 1, true, flagLoadB0);
    sync.allrd();
    jmpi(1 | ~f0[0], sskipLoad12);
    copyLoad(1, 2, false, f0[1]);
    sync.allrd();
    jmpi(1 | ~f0[1], sskipLoad12);
    copyLoadRem(2, f1[0]);
    mark(sskipLoad12);
    copyStore(0, 1, 1);
    storeSignal();
    sync.allrd();
    jmpi(1 | ~f0[0], sskipStore1);
    copyStore(1, 2, 1);
    mark(sskipStore1);
    barrierwait();
    slmRead();
    sync.allrd();
    jmpi(1 | ~f0[0], sskipStore2Load3);
    storeSignal();
    sync.allrd();
    jmpi(1 | ~f0[1], sskipStore2Load3);
    copyStore(2, 0, 2);
    sync.allrd();
    jmpi(1 | ~f1[0], sskipStore2Load3);
    copyLoadRem(3, null);
    mark(sskipStore2Load3);
    multiplyRem(0, f0[0], f0[1], true);
    jmpi(1 | ~f0[0], done);
    sync.allrd();
    jmpi(1 | ~f1[0], sskipStore3);
    copyStore(3);
    mark(sskipStore3);
    multiplyRem(1, f0[1], f1[0]);
    multiplyRem(2, f1[0], null);
    multiplyRem(3, null, null);

    mark(done);

    sync.allwr();
    setDefaultAutoSWSB(oldDefaultAutoSWSB);
}

template <HW hw>
void Generator<hw>::sysgemmStoreSignal(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, bool forceFence)
{
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    if (!strategy.slmAltBarriers || forceFence) {
        // Signal SLM data ready once memory fence returns, asynchronously
        sync.nop(depAddr[0]);
        sysgemmBarrierPrep(depAddr[3], addr3);

        slmfence(SWSB<AllPipes>(sb15, 1), addr0);
        barriermsg(sb15, addr3);
        depAddr[0] = InstructionModifier();
        depAddr[3] = sb15.src;
    } else {
        sysgemmBarrierPrep(depAddr[3], addr3);
        barriermsg(SWSB<AllPipes>(sb15, 1), addr3);
        depAddr[3] = sb15.src;
    }
}

template <HW hw>
void Generator<hw>::sysgemmCopyLoad(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int storeBuffer, bool useC)
{
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    bool surface = !strategy.A.base.isStateless();
    bool emulate64 = strategy.emulate.emulate64;
    int unrollM = strategy.unroll[LoopM];
    int unrollN = strategy.unroll[LoopN];

    auto A_ptr = A_ptr64, B_ptr = B_ptr64;
    if (surface) {
        A_ptr = A_ptr.ud();
        B_ptr = B_ptr.ud();
        emulate64 = false;
    }

    // Load new A and B and increment load pointers.
    if (surface) {
        sync(SyncFunction::nop, SWSB<uint32_t>(1));
        mov(1 | depAddr[0], addr0.ud(0), A_ptr);
        mov(1 | depAddr[1], addr1.ud(0), B_ptr);
        add(1 | depAddr[2], addr2.ud(0), B_ptr, 8 * 32);
    } else if (!emulate64) {
        sync(SyncFunction::nop, SWSB<uint64_t>(1));
        mov(1 | depAddr[0], addr0.uq(0), A_ptr);
        mov(1 | depAddr[1], addr1.uq(0), B_ptr);
        add(1 | depAddr[2], addr2.uq(0), B_ptr, 8 * 32);
    } else {
        sync(SyncFunction::nop, SWSB<uint32_t>(1));
        mov(1 | depAddr[2], addr2.ud(1), B_ptr.ud(1));
        add(1 | ov | f1[1], addr2.ud(0), B_ptr.ud(0), 8 * 32);
        mov(2 | depAddr[0], addr0.ud(0)(1), A_ptr.ud()(1));
        mov(2 | depAddr[1], addr1.ud(0)(1), B_ptr.ud()(1));
        add(1 | f1[1] | SWSB(4), addr2.ud(1), addr2.ud(1), 1);
    }

    if (useC) {
        if (surface) {
            load(1 | SWSB<AllPipes>(sb11, 3), C_regs[0], D64T(32) | strategy.A.cachingR, strategy.A.base, addr0);
            load(1 | SWSB<AllPipes>(sb12, 2), C_regs[8], D64T(32) | strategy.B.cachingR, strategy.B.base, addr1);
            if (strategy.unroll[LoopN] > 32)
                load(1 | SWSB<AllPipes>(sb13, 1), C_regs[16], D64T(16) | strategy.B.cachingR, strategy.B.base, addr2);
        } else {
            load(16 | SWSB<AllPipes>(sb11, 3), C_regs[0], block_hword(8), A64, addr0);
            load(16 | SWSB<AllPipes>(sb12, 2), C_regs[8], block_hword(8), A64, addr1);
            if (strategy.unroll[LoopN] > 32)
                load(16 | SWSB<AllPipes>(sb13, 1), C_regs[16], block_hword(4), A64, addr2);
        }
        depAddr[0] = sb11.src;
        depAddr[1] = sb12.src;
        if (strategy.unroll[LoopN] > 32)
            depAddr[2] = sb13.src;
        if (strategy.simulation)
            sync.allrd(0x3000);
    } else {
        // Stronger than necessary dependencies... can load as soon as prev. store inputs are read.
        int loadBuffer = (strategy.slmCopies == 3) ? storeBuffer : 0;
        int t0 = 8 + loadBuffer * 2;
        SBID token0{t0}, token1{t0 + 1}, token2{t0 + 2};

        if (surface) {
            load(1 | SWSB<AllPipes>(token0, 3), A_copy[loadBuffer][0], D64T(32) | strategy.A.cachingR, strategy.A.base, addr0);
            load(1 | SWSB<AllPipes>(token1, 2), B_copy[loadBuffer][0], D64T(32) | strategy.B.cachingR, strategy.B.base, addr1);
            if (strategy.unroll[LoopN] > 32)
                load(1 | SWSB<AllPipes>(token2, 1), B_copy[loadBuffer][8], D64T(16) | strategy.B.cachingR, strategy.B.base, addr2);
        } else {
            load(16 | SWSB<AllPipes>(token0, 3), A_copy[loadBuffer][0], block_hword(8), A64, addr0);
            load(16 | SWSB<AllPipes>(token1, 2), B_copy[loadBuffer][0], block_hword(8), A64, addr1);
            if (strategy.unroll[LoopN] > 32)
                load(16 | SWSB<AllPipes>(token2, 1), B_copy[loadBuffer][8], block_hword(4), A64, addr2);
        }
        depAddr[0] = token0.src;
        depAddr[1] = token1.src;
        if (strategy.unroll[LoopN] > 32)
            depAddr[2] = token2.src;
        if (strategy.simulation)
            sync.allrd(0x6 << t0);
    }

    if (!emulate64) {
        add(1 | SWSB(3), A_ptr, A_ptr, unrollM * 32);
        add(1 | SWSB(3), B_ptr, B_ptr, unrollN * 32);
    } else {
        add(1 | ov | f1[0] | SWSB(3), A_ptr.ud(0), A_ptr.ud(0), unrollM * 32);
        add(1 | ov | f1[1] | SWSB(3), B_ptr.ud(0), B_ptr.ud(0), unrollN * 32);
        add(1 | f1[0], A_ptr.ud(1), A_ptr.ud(1), 1);
        add(1 | f1[1], B_ptr.ud(1), B_ptr.ud(1), 1);
    }
}

template <HW hw>
void Generator<hw>::sysgemmCopyLoad4(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int storeBuffer, bool loadB, int useC, RegData flagLoadB)
{
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    bool emulate64 = strategy.emulate.emulate64;
    bool surface = !strategy.A.base.isStateless();
    int unrollM = strategy.unroll[LoopM];
    int unrollN = strategy.unroll[LoopN];
    int x48 = (unrollN > 32);
    InstructionModifier loadBMod;
    loadB &= !(flagLoadB.isValid() && flagLoadB.isNull());
    if (flagLoadB.isValid())
        loadBMod = loadBMod | static_cast<FlagRegister&>(flagLoadB) | any16h;

    // Load new A and B and increment load pointers.
    auto A_ptr = A_ptr64, B_ptr = B_ptr64;
    if (surface) {
        A_ptr = A_ptr.ud();
        B_ptr = B_ptr.ud();
        emulate64 = false;
    }

    if (surface) {
        sync.nop(SWSB(Pipe::I, 1));
        mov(1 | depAddr[0], addr0.ud(0), A_ptr);
        if (loadB) {
            mov(1 | depAddr[1], addr1.ud(0), B_ptr);
            add(1 | depAddr[2], addr2.ud(0), B_ptr, 8 * 32);
        }
    } else if (!emulate64) {
        sync.nop(SWSB(Pipe::L, 1));
        mov(1 | depAddr[0], addr0.uq(0), A_ptr);
        if (loadB) {
            mov(1 | depAddr[1], addr1.uq(0), B_ptr);
            add(1 | depAddr[2], addr2.uq(0), B_ptr, 8 * 32);
        }
    } else {
        sync.nop(SWSB(Pipe::I, 1));
        if (loadB) {
            mov(1 | depAddr[2], addr2.ud(1), B_ptr.ud(1));
            add(1 | ov | f1[1], addr2.ud(0), B_ptr.ud(0), 8 * 32);
        }
        mov(2 | depAddr[0], addr0.ud(0)(1), A_ptr.ud()(1));
        if (loadB) {
            mov(2 | depAddr[1], addr1.ud(0)(1), B_ptr.ud()(1));
            add(1 | f1[1], addr2.ud(1), addr2.ud(1), 1);
        }
    }

    SBID tokenA(0), tokenB0(0), tokenB1(0);
    GRF dstA, dstB0, dstB1;

    if (useC) {
        tokenA = SBID((useC == 1) ? 5 : 11);
        tokenB0 = SBID((useC == 1) ? 6 : 12);
        tokenB1 = SBID((useC == 1) ? 7 : 13);
        int C_off = (useC == 1) ? 0 : 20;
        dstA = C_regs[C_off + 0];
        dstB0 = C_regs[C_off + 8];
        dstB1 = C_regs[C_off + 16];
    } else {
        // Stronger than necessary dependencies... can load as soon as prev. store inputs are read.
        int loadBuffer = (strategy.slmCopies == 3) ? storeBuffer : 0;
        int t0 = 8 + loadBuffer * 2;
        tokenA = SBID(t0 + 0);
        tokenB0 = SBID(t0 + 1);
        tokenB1 = SBID(t0 + 2);
        dstA = A_copy[loadBuffer][0];
        dstB0 = B_copy[loadBuffer][0];
        dstB1 = B_copy[loadBuffer][8];
    }

    if (surface) {
        load(1 | tokenA | SWSB<AllPipes>(1 + loadB * (1 + x48)), dstA, D64T(32) | strategy.A.cachingR, strategy.A.base, addr0);
        if (loadB) {
            load(1 | tokenB0 | loadBMod | SWSB<AllPipes>(1 + x48), dstB0, D64T(32) | strategy.B.cachingR, strategy.B.base, addr1);
            if (x48)
                load(1 | tokenB1 | loadBMod | SWSB<AllPipes>(1), dstB1, D64T(16) | strategy.B.cachingR, strategy.B.base, addr2);
        }
    } else {
        load(16 | tokenA | SWSB<AllPipes>(1 + loadB * (1 + x48)), dstA, block_hword(8), A64, addr0);
        if (loadB) {
            load(16 | tokenB0 | loadBMod | SWSB<AllPipes>(1 + x48), dstB0, block_hword(8), A64, addr1);
            if (x48)
                load(16 | tokenB1 | loadBMod | SWSB<AllPipes>(1), dstB1, block_hword(4), A64, addr2);
        }
    }
    depAddr[0] = tokenA.src;
    if (loadB) {
        depAddr[1] = tokenB0.src;
        if (x48)
            depAddr[2] = tokenB1.src;
    }
    if (strategy.simulation) {
        uint16_t tmask = (1 << tokenA.getID());
        if (loadB) tmask |= (1 << tokenB0.getID()) | (1 << tokenB1.getID());
        sync.allrd(tmask);
    }

    if (!emulate64) {
        add(1, A_ptr, A_ptr, unrollM * 32);
        if (loadB)
            add(1, B_ptr, B_ptr, 2 * unrollN * 32);
    } else {
        add(1 | ov | f1[1], A_ptr.ud(0), A_ptr.ud(0), unrollM * 32);
        if (loadB)
            add(1 | ov | f1[1] | M8, B_ptr.ud(0), B_ptr.ud(0), 2 * unrollN * 32);
        add(1 | f1[1], A_ptr.ud(1), A_ptr.ud(1), 1);
        if (loadB)
            add(1 | f1[1] | M8, B_ptr.ud(1), B_ptr.ud(1), 1);
    }
}

template <HW hw>
void Generator<hw>::sysgemmCopyStore(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int storeBuffer, bool first)
{
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    auto aoffset = first ? slmAOffsetStoreInit : slmAOffsetStore;
    auto boffset = first ? slmBOffsetStoreInit : slmBOffsetStore;

    // Store A and B and advance store pointers to next buffer.
    mov(1 | depAddr[0], addr0.ud(2), aoffset);
    mov(1 | depAddr[1], addr1.ud(2), boffset);
    add(1 | depAddr[2], addr2.ud(2), boffset, 8 * 32 / 16);

    if (first && strategy.slmCopies == 1) {
        store(16 | SWSB<AllPipes>(sb11, 3), block_oword(16), SLM, addr0, C_regs[0]);
        store(16 | SWSB<AllPipes>(sb12, 2), block_oword(16), SLM, addr1, C_regs[8]);
        if (strategy.unroll[LoopN] > 32)
            store(16 | SWSB<AllPipes>(sb13, 1), block_oword(8), SLM, addr2, C_regs[16]);
        depAddr[0] = sb11.src;
        depAddr[1] = sb12.src;
        if (strategy.unroll[LoopN] > 32)
            depAddr[2] = sb13.src;
        if (strategy.simulation)
            sync.allrd(0x3000);
    } else {
        int loadBuffer = (strategy.slmCopies == 3) ? storeBuffer : 0;
        int t0 = 8 + loadBuffer * 2;
        SBID token0{t0}, token1{t0 + 1}, token2{t0 + 2};

        store(16 | SWSB<AllPipes>(token0, 3), block_oword(16), SLM, addr0, A_copy[loadBuffer][0]);
        store(16 | SWSB<AllPipes>(token1, 2), block_oword(16), SLM, addr1, B_copy[loadBuffer][0]);
        if (strategy.unroll[LoopN] > 32)
            store(16 | SWSB<AllPipes>(token2, 1), block_oword(8), SLM, addr2, B_copy[loadBuffer][8]);
        depAddr[0] = token0.src;
        depAddr[1] = token1.src;
        if (strategy.unroll[LoopN] > 32)
            depAddr[2] = token2.src;
        if (strategy.simulation)
            sync.allrd(0x6 << t0);
    }

    if (storeBuffer == 2)
        mov(2, slmAOffsetStore(1), slmAOffsetStoreInit(1));
    else
        add(2, slmAOffsetStore(1), aoffset(1), strategy.slmSysgemmBlockSize() / 16);
}

template <HW hw>
void Generator<hw>::sysgemmCopyStore4(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int storeBuffer, bool storeB, int useC, int useC_B)
{
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;
    bool first = (useC == 1);
    bool x48 = (strategy.unroll[LoopN] > 32);

    auto aoffset = first ? slmAOffsetStoreInit : slmAOffsetStore;
    auto boffset = first ? slmBOffsetStoreInit : slmBOffsetStore;

    // Store A and B and advance store pointers to next buffer.
    mov(1 | depAddr[0], addr0.ud(2), aoffset);
    if (storeB) {
        mov(1 | depAddr[1], addr1.ud(2), boffset);
        if (x48)
            add(1 | depAddr[2], addr2.ud(2), boffset, 8 * 32 / 16);
    }

    int loadBuffer = (strategy.slmCopies == 3) ? storeBuffer : 0;
    int t0 = 8 + loadBuffer * 2;
    auto tokenA = SBID(t0 + 0);
    auto tokenB0 = SBID(t0 + 1);
    auto tokenB1 = SBID(t0 + 2);
    auto srcA = A_copy[loadBuffer][0];
    auto srcB0 = B_copy[loadBuffer][0];
    auto srcB1 = B_copy[loadBuffer][8];

    if (useC) {
        tokenA = SBID((useC == 1) ? 5 : 11);
        int C_off = (useC == 1) ? 0 : 20;
        srcA = C_regs[C_off + 0];
    }

    if (useC_B) {
        tokenB0 = SBID((useC_B == 1) ? 6 : 12);
        tokenB1 = SBID((useC_B == 1) ? 7 : 13);
        int C_off = (useC_B == 1) ? 0 : 20;
        srcB0 = C_regs[C_off + 8];
        srcB1 = C_regs[C_off + 16];
    }

    store(16 | tokenA | SWSB<AllPipes>(1 + storeB * (1 + x48)), block_oword(16), SLM, addr0, srcA);
    if (storeB) {
        store(16 | tokenB0 | SWSB<AllPipes>(1 + x48), block_oword(16), SLM, addr1, srcB0);
        if (x48)
            store(16 | tokenB1 | SWSB<AllPipes>(1), block_oword(8), SLM, addr2, srcB1);
    }

    depAddr[0] = tokenA.src;
    if (storeB) {
        depAddr[1] = tokenB0.src;
        if (x48)
            depAddr[2] = tokenB1.src;
    }
    if (strategy.simulation) {
        uint16_t tmask = (1 << tokenA.getID());
        if (storeB) tmask |= (1 << tokenB0.getID()) | (1 << tokenB1.getID());
        sync.allrd(tmask);
    }

    if (storeBuffer == 3)
        mov(2, slmAOffsetStore(1), slmAOffsetStoreInit(1));
    else
        add(2, slmAOffsetStore(1), aoffset(1), strategy.slmSysgemmBlockSize() / 16);
}

template <HW hw>
void Generator<hw>::sysgemmMultiply(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int buffer, bool lastMultiply)
{
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;

    // Load half of A (16x32) -- hopefully broadcast from SLM to this row -- and half of B, interleaved.
    InstructionModifier swsb = lastMultiply ? SWSB(1) : depAddr[0];

    mov(1 | swsb, addr0.ud(2), slmAOffsetLoad);
    mov(1 | depAddr[1], addr1.ud(2), slmBOffsetLoad);
    add(1 | depAddr[2], addr2.ud(2), slmBOffsetLoad, 8 * 32 / 16);
    add(1 | depAddr[3], addr3.ud(2), slmBOffsetLoad, 16 * 32 / 16);

    if (strategy.slmAltBarriers) barrierwait();

    if (strategy.simulation) sync(SyncFunction::nop, SWSB<int64_t>(1));
    sync.nop(sb5.src);
    load(16 | SWSB<AllPipes>(sb3, 4), A_regs[0], block_oword(16), SLM, addr0);
    load(16 | SWSB<AllPipes>(sb0, 3), B_regs[0], block_oword(16), SLM, addr1);
    load(16 | SWSB<AllPipes>(sb1, 2), B_regs[8], block_oword(16), SLM, addr2);
    if (strategy.unroll[LoopN] > 32)
        load(16 | SWSB<AllPipes>(sb2, 1), B_regs[16], block_oword(16), SLM, addr3);

    add(1 | sb3.src, addr0.ud(2), slmAOffsetLoad, 8 * 32 / 16);
    add(1 | sb0.src, addr1.ud(2), slmAOffsetLoad, 16 * 32 / 16);
    add(1 | sb1.src, addr2.ud(2), slmAOffsetLoad, 24 * 32 / 16);
    load(16 | SWSB<AllPipes>(sb4, 3), A_regs[8], block_oword(16), SLM, addr0);

    // Wait for A data to load.
    sync.allwr(0x18);

    if (strategy.slmAltBarriers && !lastMultiply) {
        sysgemmBarrierPrep(sb2.src, addr3);
        barriermsg(SWSB<AllPipes>(sb15, 1), addr3);
    }

    // Rows 0-7
    sysgemmMultiplyChunk(problem, strategy, false, 0, 0, true, false, sb0.dst, sb3);

    // Rows 8-15
    sysgemmMultiplyChunk(problem, strategy, false, 8, 8, false, false, InstructionModifier(), sb4);

    // Load third quarter of A (8x32)
    load(16 | SWSB<AllPipes>(sb3, 2), A_regs[0], block_oword(16), SLM, addr1);

    // Rows 16-23
    sysgemmMultiplyChunk(problem, strategy, false, 0, 16, false, false, sb3.dst);

    // Load last quarter of A (8x32)
    load(16 | SWSB<AllPipes>(sb4, 1), A_regs[8], block_oword(16), SLM, addr2);

    // Increment A and B to next buffer.
    swsb = strategy.simulation ? InstructionModifier(sb3.src) : InstructionModifier();
    if (buffer == 2)
        mov(2 | swsb, slmAOffsetLoad(1), slmAOffsetLoadInit(1));
    else
        add(2 | swsb, slmAOffsetLoad(1), slmAOffsetLoad(1), strategy.slmSysgemmBlockSize() / 16);

    // Rows 24-31
    sysgemmMultiplyChunk(problem, strategy, false, 8, 24, false, false, sb4.dst, sb5);

    // Remember dependencies for address registers.
    depAddr[0] = InstructionModifier{};
    depAddr[1] = sb3.src;
    depAddr[2] = sb4.src;
    depAddr[3] = strategy.slmAltBarriers ? sb15.src : sb2.src;
}

template <HW hw>
void Generator<hw>::sysgemmMultiply4(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                     int buffer, bool firstMultiply, RegData flagWaitLoad, RegData flagSignal, Label *labelDone)
{
    using namespace sysgemm;
    auto &depAddr = state.sysgemm.depAddr;
    bool x48 = (strategy.unroll[LoopN] > 32);
    uint16_t slmStride = strategy.slmSysgemmBlockSize() / 16;
    int16_t slmAdvance = ((buffer == 3) ? -3 : 1) * slmStride;

    InstructionModifier loadMod{}, signalMod{};
    bool cooldownWaitLoad = flagWaitLoad.isValid();
    bool cooldownSignal = flagSignal.isValid();
    bool doWaitLoad = !cooldownWaitLoad || !flagWaitLoad.isNull();
    bool doSignal = !cooldownSignal || !flagSignal.isNull();
    auto fWaitLoad = static_cast<FlagRegister&>(flagWaitLoad);
    auto fSignal = static_cast<FlagRegister&>(flagSignal);
    if (doWaitLoad && cooldownWaitLoad) loadMod = loadMod | fWaitLoad | any16h;
    if (doSignal && cooldownSignal) signalMod = signalMod | fSignal | any16h;

    // Fence.
    if (doSignal) {
        sync.nop(depAddr[0]);
        slmfence(sb15 | signalMod, addr0);
        depAddr[0] = sb15.dst;
    }

    // Rows 0-7. Upper half of A (16x32) is already loaded.
    sync.nop(sb3.dst);
    depAddr[3] = InstructionModifier();
    sysgemmMultiplyChunk(problem, strategy, firstMultiply, 0, 0, true, false, sb0.dst, sb3);

    // Prepare addresses for loading lower half of A, and part of B
    add(1 | depAddr[1], addr1.ud(2), slmAOffsetLoad, 16 * 32 / 16);
    add(1 | depAddr[2], addr2.ud(2), slmAOffsetLoad, 24 * 32 / 16);
    sysgemmBarrierPrep(depAddr[3], addr3);

    // Rows 8-15.
    sysgemmMultiplyChunk(problem, strategy, firstMultiply, 8, 8, false, false, sb4.dst, sb4);

    // Load lower half of A (16x32) -- hopefully broadcast from SLM to this row.
    load(16 | SWSB<AllPipes>(sb3, 3), A_regs[0], block_oword(16), SLM, addr1);
    load(16 | SWSB<AllPipes>(sb4, 2), A_regs[8], block_oword(16), SLM, addr2);
    depAddr[1] = sb3.src;
    depAddr[2] = sb4.src;

    // Rows 16-23.
    sysgemmMultiplyChunk(problem, strategy, firstMultiply, 0, 16, false, false, sb3.dst, sb3);
    depAddr[1] = InstructionModifier();

    // Address prep, part 2.
    add(1 | depAddr[1], addr1.ud(2), slmBOffsetLoad, slmAdvance + 0 * 32 / 16);
    add(1 | depAddr[2], addr2.ud(2), slmBOffsetLoad, slmAdvance + 8 * 32 / 16);
    if (x48)
        add(1 | depAddr[0], addr0.ud(2), slmBOffsetLoad, slmAdvance + 16 * 32 / 16);    // consider moving after next dpasw block

    // Rows 24-31.
    sysgemmMultiplyChunk(problem, strategy, firstMultiply, 8, 24, false, true, sb4.dst, sb2);

    if (doWaitLoad) {
        if (cooldownWaitLoad)
            jmpi(1 | ~fWaitLoad, *labelDone);

        // Split barrier.
        barrierwait();
        if (doSignal) {
            barriermsg(SWSB<AllPipes>(sb15, x48 ? 4 : 3) | signalMod, addr3);
            depAddr[3] = sb15.src;
        }

        // Load next B data and upper 16x32 of A.
        load(16 | SWSB<AllPipes>(sb0, x48 ? 3 : 2), B_regs[0], block_oword(16), SLM, addr1);
        load(16 | SWSB<AllPipes>(sb1, x48 ? 2 : 1), B_regs[8], block_oword(16), SLM, addr2);
        if (x48)
            load(16 | SWSB<AllPipes>(sb2, 1), B_regs[16], block_oword(16), SLM, addr0);
        depAddr[1] = sb0.src;
        depAddr[2] = sb1.src;
        if (x48)
            depAddr[0] = sb2.src;

        add(1 | depAddr[3], addr3.ud(2), slmAOffsetLoad, slmAdvance + 0 * 32 / 16);
        add(1 | depAddr[2], addr2.ud(2), slmAOffsetLoad, slmAdvance + 8 * 32 / 16);
        InstructionModifier swsb;
        if (strategy.simulation) swsb = sb2.src;
        if (buffer == 3)
            mov(2 | swsb, slmAOffsetLoad(1), slmAOffsetLoadInit(1));
        else
            add(2 | swsb, slmAOffsetLoad(1), slmAOffsetLoad(1), slmAdvance);

        load(16 | SWSB<AllPipes>(sb3, 3), A_regs[0], block_oword(16), SLM, addr3);
        load(16 | SWSB<AllPipes>(sb4, 2), A_regs[8], block_oword(16), SLM, addr2);
        depAddr[3] = sb3.src;
        depAddr[2] = sb4.src;
    }
}

template <HW hw>
void Generator<hw>::sysgemmMultiplyChunk(const GEMMProblem &problem, const GEMMStrategy &strategy, bool first, int ao, int i0, bool waitB, bool prepB, const InstructionModifier &swsb0, const InstructionModifier &swsbEnd)
{
    using namespace sysgemm;
    int co = i0 * 6;

    auto dpaswTyped = [&](InstructionModifier mod, uint8_t sdepth, uint8_t rcount,
                          const GRF &cReg, const GRF &aReg, const GRF &bReg) {
        auto A = aReg.retype(problem.Ta.ngen());
        auto B = bReg.retype(problem.Tb.ngen());
        auto C = cReg.retype(problem.Tc.ngen());
#ifdef STANDALONE
        if (problem.mock.multiply) {
            mod.setExecSize(0);
            sync.nop(mod);
        } else
#endif
        first ? dpasw(mod, sdepth, rcount, C, null.retype(problem.Tc.ngen()), A, B)
              : dpasw(mod, sdepth, rcount, C, C, A, B);
    };

    if (strategy.unroll[LoopN] > 32) {
        if (waitB) {
            dpaswTyped(8 | swsb0 | Atomic,   8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
            dpaswTyped(8,                    8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
            dpaswTyped(8 | sb1.dst | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(8,                    8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
            dpaswTyped(8 | sb2.dst | Atomic, 8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
            dpaswTyped(8 | swsbEnd,          8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
        } else if (prepB) {
            dpaswTyped(8 | swsb0 | Atomic,   8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
            dpaswTyped(8 | sb0,              8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(8 | sb1,              8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
            dpaswTyped(8 | swsbEnd,          8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
        } else {
            dpaswTyped(8 | swsb0 | Atomic,   8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
            dpaswTyped(8 | swsbEnd,          8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
        }
    } else {
        if (waitB) {
            dpaswTyped(8 | swsb0 | Atomic,   8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
            dpaswTyped(8,                    8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
            dpaswTyped(8 | sb1.dst | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(8 | swsbEnd,          8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        } else if (prepB) {
            dpaswTyped(8 | swsb0 | Atomic,   8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
            dpaswTyped(8 | sb0,              8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(8 | swsbEnd,          8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        } else {
            dpaswTyped(8 | swsb0 | Atomic,   8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
            dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
            dpaswTyped(8 | swsbEnd,          8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        }
    }
}

template <HW hw>
void Generator<hw>::sysgemmBarrierPrep(const InstructionModifier &swsb, const GRF &header)
{
    using namespace sysgemm;
    mov<uint32_t>(1 | swsb, header[2], barrierVal);
}

template <HW hw>
void Generator<hw>::sysgemmReorderLocalIDs(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (strategy.splitCopy) return;
    if (strategy.wg[LoopM] != 8) return;

    auto storage = state.ra.alloc_sub<uint64_t>();
    auto temp = storage.uw(0);
    auto temp2 = storage.uw(2);
    auto lidM = state.inputs.localIDM;
    auto lidN = state.inputs.localIDN;

    // Remap local IDs so that upper 4x4 threads come first, then lower 4x4 threads.
    bfi2(1, temp, 0x08, lidN, lidM);
    shr(1, temp2, lidN, 1);
    shr(1, lidN, temp, 2);
    bfi2(1, lidM, 0x04, temp2, lidM);

    state.ra.safeRelease(storage);
}

namespace sysgemm2 {
    namespace x48 {
        static GRFRange A_regs = GRF(32)-GRF(63);
        static GRFRange B_regs = GRF(2)-GRF(25);
        static GRFRange C_regs = GRF(64)-GRF(255);
        static Subregister B_addr[3] = {GRF(26).ud(2), GRF(27).ud(2), GRF(1).ud(2)};
        static Subregister A_addr[4] = {GRF(28).ud(2), GRF(29).ud(2), GRF(30).ud(2), GRF(31).ud(2)};
        static GRF headerTemp = GRF(0);
    }

    namespace x32 {
        static GRFRange A_regs[2] = {GRF(32)-GRF(63), GRF(96)-GRF(127)};
        static GRFRange B_regs[2] = {GRF(2)-GRF(17), GRF(66)-GRF(81)};
        static GRFRange C_regs = GRF(128)-GRF(255);
        static Subregister B_addr[2][2] = {{GRF(26).ud(2), GRF(27).ud(2)},
                                           {GRF(90).ud(2), GRF(91).ud(2)}};
        static Subregister A_addr[2][4] = {{GRF(28).ud(2), GRF(29).ud(2), GRF(30).ud(2), GRF(31).ud(2)},
                                           {GRF(92).ud(2), GRF(93).ud(2), GRF(94).ud(2), GRF(95).ud(2)}};
        static GRF barrierHeader = GRF(0);
        static GRF fenceHeader = GRF(64);
    }

    static GRFRange copyInputs = GRF(254)-GRF(255);
    static Subregister A_copyLoadAddr0 = GRF(254).uq(0);
    static Subregister A_copyLoadAddrSurf0 = GRF(254).ud(2);
    static Subregister slmAOff = GRF(254).d(4);
    static Subregister lda = GRF(254).ud(6);
    static Subregister B_copyLoadAddr0 = GRF(255).uq(0);
    static Subregister B_copyLoadAddrSurf0 = GRF(255).ud(2);
    static Subregister slmBOff[2] = {GRF(255).d(4), GRF(255).d(5)};
    static Subregister ldb = GRF(255).ud(6);

    static Subregister kCounter = AccumulatorRegister(2).d(0);
    static Subregister barrierVal = AddressRegister(0).ud(0);
}

template <HW hw>
bool Generator<hw>::sysgemm2AccumulateC(GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    using namespace sysgemm2;
    auto params = systolicParams(problem);
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    auto localIDM = state.lidM;
    auto localIDN = state.lidN;
    auto C_regs = (unrollN == 48) ? x48::C_regs : x32::C_regs;

    if (unrollM != 16 && unrollM != 32) stub();
    if (unrollN != 32 && unrollN != 48) stub();
    if (isPacked(problem.A.layout)) {
        if (problem.A.crosspack != params.opsPerChan) stub();
        if (problem.A.tileR != params.osys) stub();
        if (problem.A.tileC != params.ksys) stub();
    }
    if (isPacked(problem.B.layout)) {
        if (problem.B.crosspack != params.ksys) stub();
        if (problem.B.tileR != 0 || problem.B.tileC != 0) stub();
    }

    state.ra.claim(C_regs);

    // Check whether this thread will do copy or compute. Copy threads get priority (lower thread #).
    auto flagCompute = f1[1];
    mov(1, flagCompute, state.isCompute.uw());
    state.ra.safeRelease(state.isCompute);

    // Calculate A/B addresses and SLM offsets.
    auto tempStorage = C_regs[0];
    auto suboffsetA = tempStorage.ud(0);
    auto suboffsetB = tempStorage.ud(1);
    auto tempB = tempStorage.ud(2);
    auto ldaUnrollM4 = tempStorage.ud(3);
    auto aInc = tempStorage.ud(4);
    auto ldbUnrollN4 = tempStorage.ud(5);
    auto bInc = tempStorage.ud(6);

    if (problem.A.layout == MatrixLayout::T)
        mulConstant(1, ldaUnrollM4, state.inputs.lda, unrollM / 4);
    if (problem.B.layout == MatrixLayout::N)
        mulConstant(1, ldbUnrollN4, state.inputs.ldb, unrollN / 4);
    if (!isPacked(problem.A.layout))
        mov(1, lda, state.inputs.lda);
    if (!isPacked(problem.B.layout))
        mov(1, ldb, state.inputs.ldb);

    and_(1 | ne | state.flagAP, null.uw(), localIDM, 1);

    switch (problem.A.layout) {
        case MatrixLayout::Pc:
            mulConstant(1, aInc, localIDN, unrollM * (32 / 4));
            break;
        case MatrixLayout::N:
            mulConstant(1, aInc, localIDN, unrollM * problem.Ta / 4);
            break;
        case MatrixLayout::T:
            mul(1, aInc, ldaUnrollM4, localIDN.uw());
            break;
        default: stub();
    }
    switch (problem.B.layout) {
        case MatrixLayout::Pr:
            mulConstant(1, bInc, localIDM, unrollN * (32 / 4));
            break;
        case MatrixLayout::N:
            mul(1, bInc, ldbUnrollN4, localIDM.uw());
            break;
        case MatrixLayout::T:
            mulConstant(1, bInc, localIDM, unrollN * problem.Tb / 4);
            break;
        default: stub();
    }

    mulConstant(1, suboffsetA, localIDN, unrollM * (32 / 4) / 16);
    mulConstant(1, suboffsetB, localIDM, unrollN * (32 / 4) / 16);

    if (strategy.A.base.isStateless())
        eadd(1, A_copyLoadAddr0, state.effA, aInc, strategy, state);
    else
        add(1, A_copyLoadAddrSurf0, state.effA, aInc);

    if (strategy.B.base.isStateless())
        eadd(1, B_copyLoadAddr0, state.effB, bInc, strategy, state);
    else
        add(1, B_copyLoadAddrSurf0, state.effB, bInc);

    mad(1, tempB, (4 * unrollM * 36) / 16, localIDN, (unrollN * 32) / 16);

    mul(1, x48::A_addr[0], localIDM, (unrollM * 36) / 16);
    add(1 | state.flagAP, x48::B_addr[0], tempB, (unrollN / 2) * (32 / 16));
    mov(1 | ~state.flagAP, x48::B_addr[0], tempB);

    add(1, slmAOff, x48::A_addr[0], suboffsetA);
    add(1, slmBOff[0], tempB, suboffsetB);
    add3(1, slmBOff[1], tempB, suboffsetB, 8 * 32 / 16);

    // Marshal data needed later into acc2 for safekeeping.
    auto saveData = state.ra.alloc_range(2);
    auto kLoops = saveData[0].d(0);
    auto ldc    = saveData[0].ud(1);
    auto flags  = saveData[0].ud(2);
    auto k      = saveData[0].ud(3);
    auto remM   = saveData[0].uw(8);
    auto remN   = saveData[0].uw(9);
    auto abo    = saveData[0].ud(5);
    auto ao     = saveData[0].w(10);
    auto bo     = saveData[0].w(11);
    auto alpha  = saveData[0].ud(6).reinterpret(0, problem.Ts.ngen());
    auto beta   = saveData[0].ud(7).reinterpret(0, problem.Ts.ngen());
    auto remFusedStorage = saveData[1].ud(0);
    auto diagC  = saveData[1].ud(1);
    auto effCO  = saveData[1].uq(1);
    auto C_ptr  = saveData[1].uq(2);
    auto ldco   = saveData[1].ud(7);
    auto effAs  = a0.ud(4);     // dwords 4-5
    auto effBs  = a0.ud(6);     // dwords 6-7
    auto saveI0 = saveData[1].ud(1);
    auto saveJ0 = saveData[1].ud(3);

    if (state.r0_info != acc0.ud())
        mov<uint32_t>(8, acc0, state.r0_info);

    add(1, kLoops, state.k, params.ksys - 1);
    mov(1, ldc, state.inputs.ldc[0]);
    emov(1, C_ptr, state.effC[0], strategy, state);
    if (state.inputs.flags.isValid())
        mov(1, flags, state.inputs.flags);
    mov(1, k, state.k);
    if (state.remainders[LoopM].isValid())
        mov(1, remM, state.remainders[LoopM]);
    if (state.remainders[LoopN].isValid())
        mov(1, remN, state.remainders[LoopN]);
    if (state.inputs.abo.isValid())
        mov(1, abo, state.inputs.abo);
    else {
        if (state.inputs.ao.isValid())
            mov(1, ao, state.inputs.ao);
        if (state.inputs.bo.isValid())
            mov(1, bo, state.inputs.bo);
    }
    if (state.inputs.alpha_real.isValid())
        mov(1, alpha, state.inputs.alpha_real);
    if (state.inputs.beta_real.isValid())
        mov(1, beta, state.inputs.beta_real);
    shr(1, kLoops, kLoops, ilog2(params.ksys));
    if (state.remFusedStorage.isValid())
        mov(1, remFusedStorage, state.remFusedStorage);
    if (state.diagC.isValid())
        mov(1, diagC, state.diagC);
    if (state.effCO.isValid()) {
        effCO = effCO.reinterpret(0, state.effCO.getType());
        emov(1, effCO, state.effCO, strategy, state);
        if (state.inputs.ldco.isValid())
            mov(1, ldco, state.inputs.ldco);
    }
    if (problem.hasBinaryPostOp()) {
        if (state.diagC.isValid()) stub();
        if (state.effCO.isValid() && effCO.getBytes() > 4) stub();
        mov(1, saveI0, state.i0);
        mov(1, saveJ0, state.j0);
    }
    bool abOffset = (problem.aOffset != ABOffset::None) || (problem.bOffset != ABOffset::None);
    if (abOffset) {
        GRF temp = state.ra.alloc();
        state.effAs = temp.uq(0).reinterpret(0, state.effA.getType());
        state.effBs = temp.uq(1).reinterpret(0, state.effB.getType());
        gemmCalcABOffsetAddrs(problem, strategy, state);
        mov<uint32_t>(4, effAs(1), temp);
        state.ra.safeRelease(temp);
    }

    if (state.isNested) {
        // To do: replace with sel
        mov(2 | ~flagCompute, remM(1), 0);
        mov(1 | ~flagCompute, remFusedStorage, 0);
    }

    releaseSavedMNLocalIDs(state);
    state.ra.safeRelease(state.effA);
    state.ra.safeRelease(state.effB);
    state.ra.safeRelease(state.effC[0]);
    state.ra.safeRelease(state.inputs.lda);
    state.ra.safeRelease(state.inputs.ldb);

    state.ra.release(state.inputs.ldc[0]);
    state.ra.release(state.k);
    state.ra.release(state.remainders[LoopM]);
    state.ra.release(state.remainders[LoopN]);
    state.ra.release(state.inputs.abo);
    state.ra.release(state.inputs.ao);
    state.ra.release(state.inputs.bo);
    state.ra.release(state.inputs.alpha_real);
    state.ra.release(state.inputs.beta_real);
    state.ra.release(state.remFusedStorage);
    state.ra.release(state.diagC);
    state.ra.release(state.effCO);
    state.ra.release(state.inputs.ldco);

    if (state.r0_info.isARF()) stub();
    GRF r0_info{state.r0_info.getBase()};
    if (hw >= HW::XeHPG) {
        mov(1, barrierVal.uw(0), Immediate::uw(0));
        mov(2, barrierVal.ub(2)(1), r0_info.ub(11)(0));
    } else
        and_(1, barrierVal, r0_info.ud(2), 0x7F000000);

    mov<float>(16, acc2, saveData[0]);

    Label labelCompute, labelDone;

    jmpi(1 | f1[1], labelCompute);
        sysgemm2KLoopCopy(problem, strategy, state);
        if (state.isNested) {
            jmpi(1, labelDone);
        } else
            epilogue(strategy, state);
    mark(labelCompute);
        sysgemm2KLoopCompute(problem, strategy, state);
    mark(labelDone);

    mov<float>(16, saveData[0], acc2);

    state.effC[0] = C_ptr;
    state.inputs.ldc[0] = ldc;
    if (state.inputs.flags.isValid())
        state.inputs.flags = flags;
    state.k = k;
    if (state.remainders[LoopM].isValid())
        state.remainders[LoopM] = remM;
    if (state.remainders[LoopN].isValid())
        state.remainders[LoopN] = remN;
    if (state.inputs.abo.isValid())
        state.inputs.abo = abo;
    if (state.inputs.ao.isValid())
        state.inputs.ao = ao;
    if (state.inputs.bo.isValid())
        state.inputs.bo = bo;
    if (state.inputs.alpha_real.isValid())
        state.inputs.alpha_real = alpha;
    if (state.inputs.beta_real.isValid())
        state.inputs.beta_real = beta;
    if (state.remFusedStorage.isValid()) {
        state.remFusedStorage = remFusedStorage;
        state.remaindersFused[LoopM] = state.remainders[LoopM];
        state.remaindersFused[LoopN] = state.remainders[LoopN];
        state.remaindersFused[strategy.fusedLoop] = remFusedStorage;
    }
    if (state.diagC.isValid())
        state.diagC = diagC;
    if (state.effCO.isValid())
        state.effCO = effCO;
    if (state.inputs.ldco.isValid())
        state.inputs.ldco = ldco;
    if (abOffset) {
        auto tas = state.effAs.getType();
        auto tbs = state.effBs.getType();
        state.effAs = state.ra.alloc_sub(tas);
        state.effBs = state.ra.alloc_sub(tbs);
        mov<uint32_t>(getDwords(tas), state.effAs.ud()(1), effAs(1));
        mov<uint32_t>(getDwords(tbs), state.effBs.ud()(1), effBs(1));
    }
    if (problem.hasBinaryPostOp()) {
        state.i0 = saveI0;
        state.j0 = saveJ0;
    }

    sysgemmSetupCLayouts(hw, C_regs, unrollN, problem, strategy, state);

    if (state.r0_info != acc0.ud())
        mov<uint32_t>(8, state.r0_info, acc0);

    return true;    // Success!
}

template <HW hw>
void Generator<hw>::sysgemm2KLoopCopy(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    using namespace sysgemm2;

    Label top, bottom, smallK, skipSmallK, reenterSmallK, done;

    bool surfaceA = !strategy.A.base.isStateless();
    bool surfaceB = !strategy.B.base.isStateless();
    auto unrollM = strategy.unroll[LoopM];
    auto unrollN = strategy.unroll[LoopN];
    bool _32x = (unrollM == 32);
    bool x48 = (unrollN == 48);
    bool int8 = problem.Ta.size() == 1;

    int globalBuffers = strategy.A_copies;
    auto slmStride = strategy.slmSysgemmBlockSize() / 16;

    if (globalBuffers < 2) stub();
    if (globalBuffers > 5) stub();

    auto saveRA = state.ra;
    state.ra.release(r0-r127);
    state.ra.release(r128-r255);
    state.ra.claim(copyInputs);

    // Register and token allocation.
    int aTokens = 0, aLoadAddrs = 0, aStoreAddrs = 1;
    int bTokens = 0, bLoadAddrs = 0, bStoreAddrs = 1 + x48;
    int aiRegCount = unrollM / 4, biRegCount = unrollN / 4;
    bool aRepack = false, bRepack = false;

    if (problem.A.alignment & 3 || problem.B.alignment & 3)
        stub();

    switch (problem.A.layout) {
        case MatrixLayout::Pc:
            aTokens = aLoadAddrs = (surfaceA && _32x) ? 2 : 1;
            break;
        case MatrixLayout::N:
            if (!surfaceA) stub();
            aTokens = int8 ? 2 : 1;
            aLoadAddrs = aTokens * 2;
            aRepack = true;
            break;
        case MatrixLayout::T:
            if (!surfaceA) stub();
            aTokens = aLoadAddrs = 2;
            break;
        default: stub();
    }

    switch (problem.B.layout) {
        case MatrixLayout::Pr:
            bTokens = bLoadAddrs = (surfaceB ? 2 : 1) + x48;
            break;
        case MatrixLayout::N:
            if (!surfaceB) stub();
            bTokens = 2;
            bLoadAddrs = x48 ? 4 : 2;
            if (x48) biRegCount = 16;
            bRepack = true;
            break;
        case MatrixLayout::T:
            if (!surfaceB) stub();
            bTokens = (int8 || x48) ? 2 : 1;
            bLoadAddrs = bTokens * 2;
            bRepack = true;
            break;
        default: stub();
    }

    int tokenStride = aTokens + bTokens;
    if (tokenStride * globalBuffers > 15)
        stub("Not enough tokens available.");

    auto &Ai_regs = state.Ai_regs;
    auto &Bi_regs = state.Bi_regs;
    auto &Ao_regs = state.Ao_regs;
    auto &Bo_regs = state.Bo_regs;
    auto &Ai_addrs = state.Ai_addrs;
    auto &Bi_addrs = state.Bi_addrs;
    auto &Ao_addrs = state.Ao_addrs;
    auto &Bo_addrs = state.Bo_addrs;
    RegisterRange ldaMultiples, ldbMultiples;
    FlagRegister flag12;
    GRF A_swizzle, B_swizzle;
    Subregister lda16, ldb16, ldaK, ldbK;
    Subregister Ai_advance, Bi_advance;
    GRF copyBarrierHeader = state.ra.alloc();
    GRF copyFenceHeader = state.ra.alloc();
    GRF slmBase = state.ra.alloc().d();
    GRF temp = state.ra.alloc().d();

    Ai_regs.reserve(globalBuffers);
    Bi_regs.reserve(globalBuffers);
    Ai_addrs.reserve(globalBuffers);
    Bi_addrs.reserve(globalBuffers);
    for (int i = 0; i < globalBuffers; i++) {
        Ai_regs.push_back(state.ra.alloc_range(aiRegCount));
        Bi_regs.push_back(state.ra.alloc_range(biRegCount));
        Ai_addrs.push_back(state.ra.alloc_range(aLoadAddrs));
        Bi_addrs.push_back(state.ra.alloc_range(bLoadAddrs));
    }

    if (aRepack) Ao_regs = state.ra.alloc_range(8);
    if (bRepack) Bo_regs = state.ra.alloc_range(unrollN / 4);
    Ao_addrs.push_back(state.ra.alloc_range(aStoreAddrs));
    Bo_addrs.push_back(state.ra.alloc_range(bStoreAddrs));

    if (state.emulate.temp[0].isValid()) {
        for (int q = 0; q < 2; q++)
            state.ra.safeRelease(state.emulate.temp[q]);
        state.emulate.flag = f1[1];
        state.emulate.flagOffset = 8;
    }

    // Address initialization.
    if (surfaceA && isPacked(problem.A.layout)) shr(1, A_copyLoadAddrSurf0, A_copyLoadAddrSurf0, 4);
    if (surfaceB && isPacked(problem.B.layout)) shr(1, B_copyLoadAddrSurf0, B_copyLoadAddrSurf0, 4);

    mov(1, slmBase[0], 0);
    mov(1, slmBase[1], -4 * slmStride);

    auto makeLDMultiples = [&](RegisterRange &multiples, const Subregister &ld, int n) {
        multiples = state.ra.alloc_range(n / 8);
        mov<uint16_t>(8, multiples[0], Immediate::uv(0,1,2,3,4,5,6,7));
        if (n > 8)
            mov<uint16_t>(8, multiples[1], Immediate::uv(8,9,10,11,12,13,14,15));
        mul<uint32_t>(8, multiples[0], ld, multiples[0].uw());
        if (n > 8)
            mul<uint32_t>(8, multiples[1], ld, multiples[1].uw());
    };

    switch (problem.A.layout) {
        case MatrixLayout::N:
            lda16 = state.ra.alloc_sub<uint32_t>();
            Ai_advance = state.ra.alloc_sub<uint32_t>();
            if (!int8) {
                A_swizzle = state.ra.alloc().uw();
                mov(4, A_swizzle[0](1), Immediate::uv(0,4,2,6,0,0,0,0));
                add(4, A_swizzle[4](1), A_swizzle[0](1), 64);
                add(8, A_swizzle[8](1), A_swizzle[0](1), 128);
            }
            makeLDMultiples(ldaMultiples, lda, 16);
            mulConstant(1, lda16, lda, 16);
            if (int8) {
                ldaK = state.ra.alloc_sub<uint32_t>();
                mulConstant(1, ldaK, lda, 32);
            } else
                ldaK = lda16;
            mulConstant(1, Ai_advance, lda, (int8 ? 32 : 16) * globalBuffers);
            break;
        case MatrixLayout::T:
            makeLDMultiples(ldaMultiples, lda, 8);
            break;
        default: break;
    }

    switch (problem.B.layout) {
        case MatrixLayout::N:
            B_swizzle = state.ra.alloc().uw();
            mov(8, B_swizzle[0](1), Immediate::uv(0,1,2,3,4,5,6,7));
            if (x48) {
                flag12 = state.raVFlag.alloc();
                mov(1, flag12, 0x0FFF);
            }
            mad(8, B_swizzle[8](1), 4, B_swizzle[0](1), x48 ? 64 : 32);
            mulConstant(8, B_swizzle[0](1), B_swizzle[0](1), x48 ? 64 : 32);
            makeLDMultiples(ldbMultiples, ldb, x48 ? 16 : 8);
            break;
        case MatrixLayout::T:
            makeLDMultiples(ldbMultiples, ldb, 16);
            if (int8) {
                ldb16 = state.ra.alloc_sub<uint32_t>();
                mulConstant(1, ldb16, ldb, 16);
            }
            Bi_advance = state.ra.alloc_sub<uint32_t>();
            ldbK = state.ra.alloc_sub<uint32_t>();
            mulConstant(1, Bi_advance, ldb, (int8 ? 32 : 16) * globalBuffers);
            mulConstant(1, ldbK, ldb, int8 ? 32 : 16);
        default: break;
    }

    for (int i = 0; i < globalBuffers; i++) switch (problem.A.layout) {
        case MatrixLayout::Pc:
            if (surfaceA) {
                add(1, Ai_addrs[i][0].ud(2), A_copyLoadAddrSurf0, i * unrollM * 32 / 16);
                if (_32x)
                    add(1, Ai_addrs[i][1].ud(2), A_copyLoadAddrSurf0, (i * unrollM * 32 + 4 * 32) / 16);
            } else
                eadd(1, Ai_addrs[i][0].uq(0), A_copyLoadAddr0, i * unrollM * 32, strategy, state);
            break;
        case MatrixLayout::N:
            if (i == 0) {
                add<uint32_t>(16, Ai_addrs[i][0], A_copyLoadAddrSurf0, ldaMultiples);
                if (int8)
                    add3<uint32_t>(16, Ai_addrs[i][2], A_copyLoadAddrSurf0, ldaMultiples, lda16);
            } else {
                add<uint32_t>(16, Ai_addrs[i][0], Ai_addrs[i-1][0], ldaK);
                if (int8)
                    add<uint32_t>(16, Ai_addrs[i][2], Ai_addrs[i-1][2], ldaK);
            }
            break;
        case MatrixLayout::T:
            add3<uint32_t>(8, Ai_addrs[i][0], A_copyLoadAddrSurf0, ldaMultiples, i * 32 + 0);
            add3<uint32_t>(8, Ai_addrs[i][1], A_copyLoadAddrSurf0, ldaMultiples, i * 32 + 16);
            break;
        default: stub();
    }

    for (int i = 0; i < globalBuffers; i++) switch (problem.B.layout) {
        case MatrixLayout::Pr:
            if (surfaceB) {
                add(1, Bi_addrs[i][0].ud(2), B_copyLoadAddrSurf0, i * unrollN * 32 / 16);
                add(1, Bi_addrs[i][1].ud(2), B_copyLoadAddrSurf0, (i * unrollN * 32 + 4 * 32) / 16);
                if (x48)
                    add(1, Bi_addrs[i][2].ud(2), B_copyLoadAddrSurf0, (i * unrollN * 32 + 8 * 32) / 16);
            } else {
                eadd(1, Bi_addrs[i][0].uq(0), B_copyLoadAddr0, i * unrollN * 32, strategy, state);
                if (x48)
                    eadd(1, Bi_addrs[i][1].uq(0), B_copyLoadAddr0, i * unrollN * 32 + 8 * 32, strategy, state);
            }
            break;
        case MatrixLayout::N:
            add3<uint32_t>(x48 ? 16 : 8, Bi_addrs[i][0], B_copyLoadAddrSurf0, ldbMultiples, i * 32 + 0);
            add3<uint32_t>(x48 ? 16 : 8, Bi_addrs[i][x48 ? 2 : 1], B_copyLoadAddrSurf0, ldbMultiples, i * 32 + 16);
            break;
        case MatrixLayout::T:
            if (i == 0) {
                add<uint32_t>(16, Bi_addrs[i][0], B_copyLoadAddrSurf0, ldbMultiples);
                if (int8)
                    add3<uint32_t>(16, Bi_addrs[i][2], B_copyLoadAddrSurf0, ldbMultiples, ldb16);
                else if (x48)
                    add3<uint32_t>(16, Bi_addrs[i][2], B_copyLoadAddrSurf0, ldbMultiples, 16);
            } else {
                add<uint32_t>(16, Bi_addrs[i][0], Bi_addrs[i-1][0], ldbK);
                if (int8 || x48)
                    add<uint32_t>(16, Bi_addrs[i][2], Bi_addrs[i-1][2], ldbK);
            }
            break;
        default: stub();
    }

    sysgemmBarrierPrep(InstructionModifier(), copyBarrierHeader);

    mov(2, slmBase[4](1), slmBase[0](1));

    // Main logic.
    auto copyLoad = [&](int buffer) {
        int atbase = tokenStride * buffer;
        int btbase = atbase + aTokens;
        switch (problem.A.layout) {
            case MatrixLayout::Pc:
                if (surfaceA) {
                    load(16 | SBID(atbase + 0), Ai_regs[buffer][0], block_oword(8), strategy.A.base, Ai_addrs[buffer][0]);
                    if (_32x)
                        load(16 | SBID(atbase + 1), Ai_regs[buffer][4], block_oword(8), strategy.A.base, Ai_addrs[buffer][1]);
                } else
                    load(16 | SBID(atbase + 0), Ai_regs[buffer][0], block_hword(_32x ? 8 : 4), strategy.A.base, Ai_addrs[buffer]);
                break;
            case MatrixLayout::N:
                if (int8) {
                    load(16 | SBID(atbase + 0), Ai_regs[buffer][0], surface_dword(ChannelMask::rg), strategy.A.base, Ai_addrs[buffer][0]);
                    load(16 | SBID(atbase + 1), Ai_regs[buffer][4], surface_dword(ChannelMask::rg), strategy.A.base, Ai_addrs[buffer][2]);
                } else
                    load(16 | SBID(atbase + 0), Ai_regs[buffer][0], surface_dword(ChannelMask::rgba), strategy.A.base, Ai_addrs[buffer][0]);
                break;
            case MatrixLayout::T:
                load(8 | SBID(atbase + 0), Ai_regs[buffer][0], surface_dword(ChannelMask::rgba), strategy.A.base, Ai_addrs[buffer][0]);
                load(8 | SBID(atbase + 1), Ai_regs[buffer][4], surface_dword(ChannelMask::rgba), strategy.A.base, Ai_addrs[buffer][1]);
                break;
            default: stub();
        }

        switch (problem.B.layout) {
            case MatrixLayout::Pr:
                if (surfaceB) {
                    load(16 | SBID(btbase + 0), Bi_regs[buffer][0], block_oword(8), strategy.B.base, Bi_addrs[buffer][0]);
                    load(16 | SBID(btbase + 1), Bi_regs[buffer][4], block_oword(8), strategy.B.base, Bi_addrs[buffer][1]);
                    if (x48) load(16 | SBID(btbase + 2), Bi_regs[buffer][8], block_oword(8), strategy.B.base, Bi_addrs[buffer][2]);
                } else {
                    load(16 | SBID(btbase + 0), Bi_regs[buffer][0], block_hword(8), strategy.B.base, Bi_addrs[buffer][0]);
                    if (x48) load(16 | SBID(btbase + 1), Bi_regs[buffer][8], block_hword(4), strategy.B.base, Bi_addrs[buffer][1]);
                }
                break;
            case MatrixLayout::N:
                if (x48) {
                    load(16 | SBID(btbase + 0) | flag12 | any4h, Bi_regs[buffer][0], surface_dword(ChannelMask::rgba), strategy.B.base, Bi_addrs[buffer][0]);
                    load(16 | SBID(btbase + 1) | flag12 | any4h, Bi_regs[buffer][8], surface_dword(ChannelMask::rgba), strategy.B.base, Bi_addrs[buffer][2]);
                } else {
                    load(8 | SBID(btbase + 0), Bi_regs[buffer][0], surface_dword(ChannelMask::rgba), strategy.B.base, Bi_addrs[buffer][0]);
                    load(8 | SBID(btbase + 1), Bi_regs[buffer][4], surface_dword(ChannelMask::rgba), strategy.B.base, Bi_addrs[buffer][1]);
                }
                break;
            case MatrixLayout::T:
                if (int8) {
                    auto cmask = x48 ? ChannelMask::rgb : ChannelMask::rg;
                    load(16 | SBID(btbase + 0), Bi_regs[buffer][0], surface_dword(cmask), strategy.B.base, Bi_addrs[buffer][0]);
                    load(16 | SBID(btbase + 1), Bi_regs[buffer][x48 ? 6 : 4], surface_dword(cmask), strategy.B.base, Bi_addrs[buffer][2]);
                } else {
                    load(16 | SBID(btbase + 0), Bi_regs[buffer][0], surface_dword(ChannelMask::rgba), strategy.B.base, Bi_addrs[buffer][0]);
                    if (x48)
                        load(16 | SBID(btbase + 1), Bi_regs[buffer][8], surface_dword(ChannelMask::rg), strategy.B.base, Bi_addrs[buffer][2]);
                }
                break;
            default: stub();
        }
    };

    auto copyRepack = [&](int buffer) {
        int atbase = tokenStride * buffer;
        int btbase = atbase + aTokens;

        switch (problem.A.layout) {
            case MatrixLayout::N:
                if (int8) {
                    for (int j = 0; j < 4; j++) {
                        int reg = (j >> 1);
                        int sub = (j & 1) << 4;
                        mov<uint8_t>(16, Ao_regs[j + 0][0](1),  Ai_regs[buffer][reg + 0][sub](1,4,4));
                        mov<uint8_t>(16, Ao_regs[j + 4][0](1),  Ai_regs[buffer][reg + 4][sub](1,4,4));
                        mov<uint8_t>(16, Ao_regs[j + 0][16](1), Ai_regs[buffer][reg + 2][sub](1,4,4));
                        mov<uint8_t>(16, Ao_regs[j + 4][16](1), Ai_regs[buffer][reg + 6][sub](1,4,4));
                    }
                } else {
                    // a0: 0 4 2 6 64 68 66 70...
                    add(16, a0, A_swizzle, Ai_regs[buffer][0].getBase() * 32);
                    setDefaultAutoSWSB(false);
                    sync.allwr(0b1 << atbase);
                    for (int j = 0; j < 8; j++)
                        mov<uint16_t>(16, Ao_regs[j], indirect[a0].uw(j*8)(1,0));
                    setDefaultAutoSWSB(true);
                }
                break;
            default: break;
        }

        switch (problem.B.layout) {
            case MatrixLayout::N:
                // a0 (x32): 0 32 64 96 128 160 192 228 4 36 68...
                // a0 (x48): 0 64 128 192 256 320 384 448 4 68 132 196...
                add(16, a0, B_swizzle, Bi_regs[buffer][0].getBase() * 32);
                setDefaultAutoSWSB(false);
                sync.allwr(0b11 << btbase);
                for (int j = 0; j < unrollN/4; j += 2)      // 2 cols at a time
                    mov<uint32_t>(16, Bo_regs[j], indirect[a0].ud(j*4)(1,0));
                setDefaultAutoSWSB(true);
                break;
            case MatrixLayout::T:
                if (int8) {
                    for (int j = 0; j < unrollN/4; j++)
                        mov<uint8_t>(16, Bo_regs[j][0](1), Bi_regs[buffer][(j & ~3) >> 1][j & 3](4));
                    for (int j = 0; j < unrollN/4; j++)
                        mov<uint8_t>(16, Bo_regs[j][16](1), Bi_regs[buffer][(x48 ? 6 : 4) + ((j & ~3) >> 1)][j & 3](4));
                } else {
                    for (int j = 0; j < unrollN/4; j++)
                        mov<uint16_t>(16, Bo_regs[j], Bi_regs[buffer][j & ~1][j & 1](2));
                }
                break;
            default: break;
        }
    };

    auto copyStore = [&](int buffer) {
        int atbase = tokenStride * buffer;
        int btbase = atbase + aTokens;

        auto A_regs = aRepack ? Ao_regs : Ai_regs[buffer];
        auto B_regs = bRepack ? Bo_regs : Bi_regs[buffer];

        copyRepack(buffer);

        auto b1 = (surfaceB && isPacked(problem.B.layout)) ? 2 : 1;

        store(16 | SBID(atbase + 0), block_oword(_32x ? 16 : 8), SLM, Ao_addrs[0][0], A_regs[0]);
        store(16 | SBID(btbase + 0), block_oword(16), SLM, Bo_addrs[0][0], B_regs[0]);
        if (x48) store(16 | SBID(btbase + b1), block_oword(8), SLM, Bo_addrs[0][1], B_regs[8]);
    };

    auto advanceLoad = [&](int buffer) {
        switch (problem.A.layout) {
            case MatrixLayout::Pc:
                if (surfaceA) {
                    add(1, Ai_addrs[buffer][0].ud(2), Ai_addrs[buffer][0].ud(2), globalBuffers * 32 * unrollM / 16);
                    if (_32x)
                        add(1, Ai_addrs[buffer][1].ud(2), Ai_addrs[buffer][1].ud(2), globalBuffers * 32 * unrollM / 16);
                } else
                    eadd(1, Ai_addrs[buffer][0].uq(0), Ai_addrs[buffer][0].uq(0), globalBuffers * 32 * unrollM, strategy, state);
                break;
            case MatrixLayout::N:
                add<uint32_t>(16, Ai_addrs[buffer][0], Ai_addrs[buffer][0], Ai_advance);
                if (int8)
                    add<uint32_t>(16, Ai_addrs[buffer][2], Ai_addrs[buffer][2], Ai_advance);
                break;
            case MatrixLayout::T:
                add<uint32_t>(8, Ai_addrs[buffer][0], Ai_addrs[buffer][0], 32 * globalBuffers);
                add<uint32_t>(8, Ai_addrs[buffer][1], Ai_addrs[buffer][1], 32 * globalBuffers);
                break;
            default: stub();
        }

        switch (problem.B.layout) {
            case MatrixLayout::Pr:
                if (surfaceB) {
                    add(1, Bi_addrs[buffer][0].ud(2), Bi_addrs[buffer][0].ud(2), globalBuffers * 32 * unrollN / 16);
                    add(1, Bi_addrs[buffer][1].ud(2), Bi_addrs[buffer][1].ud(2), globalBuffers * 32 * unrollN / 16);
                    if (x48) add(1, Bi_addrs[buffer][2].ud(2), Bi_addrs[buffer][2].ud(2), globalBuffers * 32 * unrollN / 16);
                } else {
                    eadd(1, Bi_addrs[buffer][0].uq(0), Bi_addrs[buffer][0].uq(0), globalBuffers * 32 * unrollN, strategy, state);
                    if (x48) eadd(1, Bi_addrs[buffer][1].uq(0), Bi_addrs[buffer][1].uq(0), globalBuffers * 32 * unrollN, strategy, state);
                }
                break;
            case MatrixLayout::N:
                add<uint32_t>(16, Bi_addrs[buffer][0], Bi_addrs[buffer][0], 32 * globalBuffers);
                if (x48)
                    add<uint32_t>(16, Bi_addrs[buffer][2], Bi_addrs[buffer][2], 32 * globalBuffers);
                break;
            case MatrixLayout::T:
                add<uint32_t>(16, Bi_addrs[buffer][0], Bi_addrs[buffer][0], Bi_advance);
                if (int8 || x48)
                    add<uint32_t>(16, Bi_addrs[buffer][2], Bi_addrs[buffer][2], Bi_advance);
                break;
            default: stub();
        }
    };

    auto advanceStore = [&](int buffer = -1) {
        add(2, temp, slmBase, slmStride);
        add(1, Ao_addrs[0][0].ud(2), slmBase, slmAOff);
        add(1, Bo_addrs[0][0].ud(2), slmBase, slmBOff[0]);
        if (x48) add(1, Bo_addrs[0][1].ud(2), slmBase, slmBOff[1]);

        csel(2 | ge | f0[0], slmBase, slmBase[4](1,1,0), temp[0](1,1,0), temp[1]);
    };

    auto fence = [&]() {
        slmfence(sb15, copyFenceHeader, copyFenceHeader);
    };

    auto splitBarrier = [&]() {
        barrierwait();
        barriermsg(sb15, copyBarrierHeader);
    };

    // Warmup.
    if (globalBuffers > 1) cmp(1 | gt | f0[0], kCounter, 1);
    if (globalBuffers > 2) cmp(1 | gt | f0[1], kCounter, 2);
    if (globalBuffers > 3) cmp(1 | gt | f1[0], kCounter, 3);
    if (globalBuffers > 4) cmp(1 | gt | f1[1], kCounter, 4);
    if (globalBuffers > 1) {
        copyLoad(0);
        jmpi(1 | ~f0[0], smallK);
    }
    if (globalBuffers > 2) {
        copyLoad(1);
        jmpi(1 | ~f0[1], smallK);
    }
    if (globalBuffers > 3) {
        copyLoad(2);
        jmpi(1 | ~f1[0], smallK);
    }
    if (globalBuffers > 4) {
        copyLoad(3);
        jmpi(1 | ~f1[1], smallK);
    }

    auto flagLast = FlagRegister::createFromIndex(globalBuffers - 2);
    cmp(1 | le | flagLast, kCounter, globalBuffers);
    copyLoad(globalBuffers - 1);
    jmpi(1 | flagLast, smallK);

    add(1 | gt | f0[0], kCounter, kCounter, -2 * globalBuffers);

    advanceStore();
    advanceLoad(0);
    if (globalBuffers > 1) advanceLoad(1);
    if (globalBuffers > 2) advanceLoad(2);
    if (globalBuffers > 3) advanceLoad(3);

    copyStore(0);
    if (globalBuffers > 4) advanceLoad(4);

    fence();
    advanceStore(0);
    copyLoad(0);
    barriermsg(sb15, copyBarrierHeader);
    copyStore(1);

    advanceLoad(0);

    jmpi(1 | ~f0[0], bottom);
    sync.nop(SWSB<AllPipes>(1));
    mark(top);
    {
        add(1 | gt | f0[0], kCounter, kCounter, -globalBuffers);
        for (int i = 0; i < globalBuffers; i++) {
            fence();
            advanceStore((i + 1) % globalBuffers);
            copyLoad((i + 1) % globalBuffers);  // move after barrier?
            splitBarrier();
            copyStore((i + 2) % globalBuffers);
            advanceLoad((i + 1) % globalBuffers);
        }
    }
    jmpi(1 | f0[0], top);
    mark(bottom);

    if (globalBuffers > 1) cmp(1 | gt | f0[0], kCounter, -globalBuffers + 1);
    if (globalBuffers > 2) cmp(1 | gt | f0[1], kCounter, -globalBuffers + 2);
    if (globalBuffers > 3) cmp(1 | gt | f1[0], kCounter, -globalBuffers + 3);
    if (globalBuffers > 4) cmp(1 | gt | f1[1], kCounter, -globalBuffers + 4);

    // Cooldown loop. All buffers but #1 loaded.
    for (int i = 1; i < globalBuffers; i++) {
        Label skipLoad;
        fence();
        advanceStore(i);
        jmpi(1 | ~FlagRegister::createFromIndex(i - 1), skipLoad);
        copyLoad(i);
        mark(skipLoad);
        splitBarrier();
        copyStore((i + 1) % globalBuffers);
        advanceLoad(i);
    }

    jmpi(1, skipSmallK);
    mark(smallK);

    advanceStore();
    copyStore(0);

    fence();
    advanceStore(0);
    barriermsg(sb15, copyBarrierHeader);
    jmpi(1, reenterSmallK);

    mark(skipSmallK);

    for (int i = 0; i < globalBuffers - 1; i++) {
        fence();
        advanceStore(i);
        splitBarrier();
        if (i == 0)
            mark(reenterSmallK);
        jmpi(1 | ~FlagRegister::createFromIndex(i), done);
        copyStore(i + 1);
    }

    fence();
    splitBarrier();

    mark(done);
    barrierwait();

    state.ra = saveRA;
}

template <HW hw>
void Generator<hw>::sysgemm2KLoopCompute(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    using namespace sysgemm2;

    Label top, remainder, done;
    bool _32x = (strategy.unroll[LoopM] == 32);
    bool x48 = (strategy.unroll[LoopN] == 48);
    bool keepBarHdr = strategy.skipFence || !x48;
    auto slmStride = strategy.slmSysgemmBlockSize() / 16;
    auto barrierHeader = x32::barrierHeader;

    mov(1, f0.ud(0), 0);
    mov(1, f1.ud(0), 0);
    if (x48) {
        using namespace x48;
        add(1, A_addr[1], A_addr[0], 8  * 32 / 16);
        if (_32x) {
            add(1, A_addr[2], A_addr[0], 16 * 32 / 16);
            add(1, A_addr[3], A_addr[0], 24 * 32 / 16);
        }
        add(1, B_addr[1], B_addr[0], 8  * 32 / 16);
        add(1, B_addr[2], B_addr[0], 16 * 32 / 16);
    } else {
        using namespace x32;
        add(1, A_addr[0][1], A_addr[0][0], 8  * 32 / 16);
        if (_32x) {
            add(1, A_addr[0][2], A_addr[0][0], 16 * 32 / 16);
            add(1, A_addr[0][3], A_addr[0][0], 24 * 32 / 16);
        }
        add(1, A_addr[1][0], A_addr[0][0], 0  * 32 / 16 + slmStride);
        add(1, A_addr[1][1], A_addr[0][0], 8  * 32 / 16 + slmStride);
        if (_32x) {
            add(1, A_addr[1][2], A_addr[0][0], 16 * 32 / 16 + slmStride);
            add(1, A_addr[1][3], A_addr[0][0], 24 * 32 / 16 + slmStride);
        }
        add(1, B_addr[0][1], B_addr[0][0], 8  * 32 / 16);
        add(1, B_addr[1][0], B_addr[0][0], 0  * 32 / 16 + slmStride);
        add(1, B_addr[1][1], B_addr[0][0], 8  * 32 / 16 + slmStride);
    }

    if (keepBarHdr)
        sysgemmBarrierPrep(InstructionModifier(), barrierHeader);

    // Warmup: signal, split barrier, load
    cmp(1 | gt | f1[1], kCounter, 1);
    add(1 | gt | f0[0], kCounter, kCounter, -5);

    if (!keepBarHdr)
        sysgemmBarrierPrep(InstructionModifier(), barrierHeader);
    barriermsg(sb15, barrierHeader);
    barrierwait();
    barriermsg(sb15 | f1[1], barrierHeader);

    bool oldDefaultAutoSWSB = getDefaultAutoSWSB();
    setDefaultAutoSWSB(false);
    sync.nop(SWSB<AllPipes>(1));

    load(16 | sb0, x48::A_regs[0],  block_oword(16), SLM, x48::A_addr[0]);
    load(16 | sb1, x48::A_regs[8],  block_oword(16), SLM, x48::A_addr[1]);
    if (_32x) {
        load(16 | sb2, x48::A_regs[16], block_oword(16), SLM, x48::A_addr[2]);
        load(16 | sb3, x48::A_regs[24], block_oword(16), SLM, x48::A_addr[3]);
    }
    load(16 | sb4, x48::B_regs[0],  block_oword(16), SLM, x48::B_addr[0]);
    load(16 | sb5, x48::B_regs[8],  block_oword(16), SLM, x48::B_addr[1]);
    if (x48)
        load(16 | sb6, x48::B_regs[16], block_oword(16), SLM, x48::B_addr[2]);

    zeroMatrix(x48 ? x48::C_regs : x32::C_regs, strategy);

    jmpi(1 | ~f0[0], remainder);

    mark(top);
    {
        add(1 | gt | f0[0], kCounter, kCounter, -4);
        sysgemm2Multiply(problem, strategy, state, 0);
        sysgemm2Multiply(problem, strategy, state, 1);
        sysgemm2Multiply(problem, strategy, state, 2);
        sysgemm2Multiply(problem, strategy, state, 3);
    }
    jmpi(1 | f0[0], top);

    mark(remainder);

    cmp(1 | gt | f0[0], kCounter, 1 - 5);
    cmp(1 | gt | f0[1], kCounter, 2 - 5);
    cmp(1 | gt | f1[0], kCounter, 3 - 5);
    cmp(1 | gt | f1[1], kCounter, 4 - 5);
    sysgemm2Multiply(problem, strategy, state, 0, true, f0[0], f0[1]);
    jmpi(1 | ~f0[0], done);

    sysgemm2Multiply(problem, strategy, state, 1, true, f0[1], f1[0]);
    jmpi(1 | ~f0[1], done);

    sysgemm2Multiply(problem, strategy, state, 2, true, f1[0], f1[1]);
    jmpi(1 | ~f1[0], done);

    sysgemm2Multiply(problem, strategy, state, 3, true, f1[1]);
    jmpi(1 | ~f1[1], done);

    sysgemm2Multiply(problem, strategy, state, 0, true);

    mark(done);

    setDefaultAutoSWSB(oldDefaultAutoSWSB);
}

template <HW hw>
void Generator<hw>::sysgemm2Multiply(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int slmBuffer, bool cooldown, FlagRegister flagWaitLoad, FlagRegister flagSignal)
{
    if (strategy.unroll[LoopN] == 48)
        sysgemm2MultiplyX48(problem, strategy, state, slmBuffer, cooldown, flagWaitLoad, flagSignal);
    else
        sysgemm2MultiplyX32(problem, strategy, state, slmBuffer, cooldown, flagWaitLoad, flagSignal);
}

template <HW hw>
void Generator<hw>::sysgemm2MultiplyX48(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int slmBuffer, bool cooldown, FlagRegister flagWaitLoad, FlagRegister flagSignal)
{
    using namespace sysgemm2;
    using namespace sysgemm2::x48;

    auto slmStride = strategy.slmSysgemmBlockSize() / 16;
    int16_t advance = ((slmBuffer == 3) ? -3 : 1) * slmStride;
    InstructionModifier loadMod{}, signalMod{};
    bool doWaitLoad = !cooldown || flagWaitLoad.isValid();
    bool doSignal = !cooldown || flagSignal.isValid();
    if (cooldown) {
        if (doWaitLoad) loadMod = loadMod | flagWaitLoad | any16h;
        if (doSignal) signalMod = signalMod | flagSignal | any16h;
    }

    if (strategy.unroll[LoopM] != 32) stub();

    if (doWaitLoad) {
        Label skipWait;
        if (cooldown)
            jmpi(1 | ~flagWaitLoad, skipWait);

        // SLM fence
        if (!strategy.skipFence && doSignal)
            slmfence(sb15 | signalMod, headerTemp, headerTemp);

        // Barrier wait
        barrierwait();

        // Barrier signal
        if (doSignal) {
            if (!strategy.skipFence) {
                sysgemmBarrierPrep(sb15.dst | signalMod, headerTemp);
                barriermsg(sb15 | SWSB<AllPipes>(1) | signalMod, headerTemp);
            } else
                barriermsg(sb15 | signalMod, headerTemp);
        }

        if (cooldown)
            mark(skipWait);
    }

    // Advance A0 address (note dst)
    if (doWaitLoad)
        add(1 | sb0.dst, A_addr[0], A_addr[0], advance);

    // Rows 0-7
    sysgemm2MultiplyChunkX48(problem, strategy, 0);

    // Advance A1 address
    if (doWaitLoad)
        add(1 | sb1.src, A_addr[1], A_addr[1], advance);

    // Rows 8-15
    sysgemm2MultiplyChunkX48(problem, strategy, 1);

    if (doWaitLoad) {
        // Load new A0
        load(16 | sb0 | loadMod, A_regs[0], block_oword(16), SLM, A_addr[0]);

        // Advance B, A2, A3 addresses
        add(1 | sb4.src, B_addr[0], B_addr[0], advance);
        add(1 | sb5.src, B_addr[1], B_addr[1], advance);
        add(1 | sb6.src, B_addr[2], B_addr[2], advance);

        add(1 | sb2.src, A_addr[2], A_addr[2], advance);
        add(1 | sb3.src, A_addr[3], A_addr[3], advance);
    }

    // Rows 16-23
    sysgemm2MultiplyChunkX48(problem, strategy, 2);

    // Load new A1
    if (doWaitLoad)
        load(16 | sb1 | loadMod, A_regs[8], block_oword(16), SLM, A_addr[1]);

    // Rows 24-31
    sysgemm2MultiplyChunkX48(problem, strategy, 3);

    if (doWaitLoad) {
        // Load new B data
        load(16 | sb4 | loadMod, B_regs[0], block_oword(16), SLM, B_addr[0]);
        load(16 | sb5 | loadMod, B_regs[8], block_oword(16), SLM, B_addr[1]);
        load(16 | sb6 | loadMod, B_regs[16], block_oword(16), SLM, B_addr[2]);

        // Load new A2,A3
        load(16 | sb2 | loadMod, A_regs[16], block_oword(16), SLM, A_addr[2]);
        load(16 | sb3 | loadMod, A_regs[24], block_oword(16), SLM, A_addr[3]);
    }
}

template <HW hw>
void Generator<hw>::sysgemm2MultiplyX32(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state, int slmBuffer, bool cooldown, FlagRegister flagWaitLoad, FlagRegister flagSignal)
{
    using namespace sysgemm2;
    using namespace sysgemm2::x32;

    auto slmStride = strategy.slmSysgemmBlockSize() / 16;
    int16_t advance = ((slmBuffer >= 2) ? -2 : 2) * slmStride;
    InstructionModifier loadMod{}, signalMod{};
    bool doWaitLoad = !cooldown || flagWaitLoad.isValid();
    bool doSignal = !cooldown || flagSignal.isValid();
    if (cooldown) {
        if (doWaitLoad) loadMod = loadMod | flagWaitLoad | any16h;
        if (doSignal) signalMod = signalMod | flagSignal | any16h;
    }
    bool _32x = (strategy.unroll[LoopM] == 32);
    bool odd = (slmBuffer & 1);
    int tokenBase = odd ? 8 : 0;
    int otokenBase = odd ? 0 : 8;

    if (doWaitLoad) {
        Label skipWait;
        if (cooldown)
            jmpi(1 | ~flagWaitLoad, skipWait);

        // SLM fence
        if (!strategy.skipFence && doSignal)
            slmfence(sb15 | signalMod, fenceHeader, fenceHeader);

        add(1 | SBID(tokenBase + 0).src, A_addr[odd][0], A_addr[odd][0], advance);  // TODO: reuse src0
        add(1 | SBID(tokenBase + 4).src, B_addr[odd][0], B_addr[odd][0], advance);
        add(1 | SBID(tokenBase + 5).src, B_addr[odd][1], B_addr[odd][1], advance);
        add(1 | SBID(tokenBase + 1).src, A_addr[odd][1], A_addr[odd][1], advance);
        if (_32x) {
            add(1 | SBID(tokenBase + 2).src, A_addr[odd][2], A_addr[odd][2], advance);
            add(1 | SBID(tokenBase + 3).src, A_addr[odd][3], A_addr[odd][3], advance);
        }

        // Barrier wait
        barrierwait();

        if (hw >= HW::XeHPG) {
            // Wait for SLM loads to return before signaling.
            sync.allwr(0x3F << tokenBase);
        }

        // Barrier signal
        if (doSignal)
            barriermsg(sb15 | signalMod, barrierHeader);

        if (cooldown)
            mark(skipWait);
    }

    if (doWaitLoad) {
        load(16 | SBID(otokenBase + 0) | loadMod, A_regs[!odd][0],  block_oword(16), SLM, A_addr[!odd][0]);
        load(16 | SBID(otokenBase + 4) | loadMod, B_regs[!odd][0],  block_oword(16), SLM, B_addr[!odd][0]);
        load(16 | SBID(otokenBase + 5) | loadMod, B_regs[!odd][8],  block_oword(16), SLM, B_addr[!odd][1]);
        load(16 | SBID(otokenBase + 1) | loadMod, A_regs[!odd][8],  block_oword(16), SLM, A_addr[!odd][1]);
        if (_32x) {
            load(16 | SBID(otokenBase + 2) | loadMod, A_regs[!odd][16], block_oword(16), SLM, A_addr[!odd][2]);
            load(16 | SBID(otokenBase + 3) | loadMod, A_regs[!odd][24], block_oword(16), SLM, A_addr[!odd][3]);
        }
    }

    sysgemm2MultiplyChunkX32(problem, strategy, 0, odd);
    sysgemm2MultiplyChunkX32(problem, strategy, 1, odd);
    if (_32x) {
        sysgemm2MultiplyChunkX32(problem, strategy, 2, odd);
        sysgemm2MultiplyChunkX32(problem, strategy, 3, odd);
    }
}

template <HW hw>
void Generator<hw>::sysgemm2MultiplyChunkX48(const GEMMProblem &problem, const GEMMStrategy &strategy, int chunkA)
{
    using namespace sysgemm2;
    using namespace sysgemm2::x48;
    int ao = chunkA * 8;
    int co = ao * 6;
    bool waitB = (chunkA == 0);
    bool prepB = (chunkA == 3);
    SBID sbA(chunkA);

    auto dpaswTyped = [&](InstructionModifier mod, uint8_t sdepth, uint8_t rcount,
                          const GRF &cReg, const GRF &aReg, const GRF &bReg) {
#ifdef STANDALONE
        if (problem.mock.multiply) {
            mod.setExecSize(0);
            sync.nop(mod);
        } else
#endif
        dpasw(mod, sdepth, rcount, cReg.retype(problem.Tc.ngen()), cReg.retype(problem.Tc.ngen()),
                                   aReg.retype(problem.Ta.ngen()), bReg.retype(problem.Tb.ngen()));
    };

    if (waitB) {
        /* sync.nop(sbA.dst); */                // arranged by caller
        dpaswTyped(8 | sb4.dst | Atomic, 8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
        dpaswTyped(8,                    8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
        dpaswTyped(8 | sb5.dst | Atomic, 8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
        dpaswTyped(8,                    8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        dpaswTyped(8 | sb6.dst | Atomic, 8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
        dpaswTyped(8 | sbA,              8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
    } else if (prepB) {
        dpaswTyped(8 | sbA.dst | Atomic, 8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
        dpaswTyped(8 | sb4,              8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
        dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
        dpaswTyped(8 | sb5,              8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
        dpaswTyped(8 | sb6,              8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
    } else {
        dpaswTyped(8 | sbA.dst | Atomic, 8, 8, C_regs[co],      A_regs[ao], B_regs[0]);
        dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 8],  A_regs[ao], B_regs[4]);
        dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 16], A_regs[ao], B_regs[8]);
        dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 24], A_regs[ao], B_regs[12]);
        dpaswTyped(8 | Atomic,           8, 8, C_regs[co + 32], A_regs[ao], B_regs[16]);
        dpaswTyped(8 | sbA,              8, 8, C_regs[co + 40], A_regs[ao], B_regs[20]);
    }
}

template <HW hw>
void Generator<hw>::sysgemm2MultiplyChunkX32(const GEMMProblem &problem, const GEMMStrategy &strategy, int chunkA, bool odd)
{
    using namespace sysgemm2;
    using namespace sysgemm2::x32;
    int ao = chunkA * 8;
    int co = ao * 4;
    int nchunks = strategy.unroll[LoopM] / 8;
    bool waitB = (chunkA == 0);
    bool prepB = (chunkA == nchunks - 1);
    int tokenBase = odd ? 8 : 0;
    SBID sbA(tokenBase + chunkA);
    SBID sbB0(tokenBase + 4);
    SBID sbB1(tokenBase + 5);

    auto dpaswTyped = [&](InstructionModifier mod, uint8_t sdepth, uint8_t rcount,
                          const GRF &cReg, const GRF &aReg, const GRF &bReg) {
#ifdef STANDALONE
        if (problem.mock.multiply) {
            mod.setExecSize(0);
            sync.nop(mod);
        } else
#endif
        dpasw(mod, sdepth, rcount, cReg.retype(problem.Tc.ngen()), cReg.retype(problem.Tc.ngen()),
                                   aReg.retype(problem.Ta.ngen()), bReg.retype(problem.Tb.ngen()));
    };

    if (waitB) {
        sync.nop(sbA.dst);
        dpaswTyped(8 | sbB0.dst | Atomic, 8, 8, C_regs[co],      A_regs[odd][ao], B_regs[odd][0]);
        dpaswTyped(8,                     8, 8, C_regs[co + 8],  A_regs[odd][ao], B_regs[odd][4]);
        dpaswTyped(8 | sbB1.dst | Atomic, 8, 8, C_regs[co + 16], A_regs[odd][ao], B_regs[odd][8]);
        dpaswTyped(8 | sbA,               8, 8, C_regs[co + 24], A_regs[odd][ao], B_regs[odd][12]);
    } else if (prepB) {
        dpaswTyped(8 | sbA.dst | Atomic,  8, 8, C_regs[co],      A_regs[odd][ao], B_regs[odd][0]);
        dpaswTyped(8 | sbB0,              8, 8, C_regs[co + 8],  A_regs[odd][ao], B_regs[odd][4]);
        dpaswTyped(8 | Atomic,            8, 8, C_regs[co + 16], A_regs[odd][ao], B_regs[odd][8]);
        dpaswTyped(8 | sbB1,              8, 8, C_regs[co + 24], A_regs[odd][ao], B_regs[odd][12]);
    } else {
        dpaswTyped(8 | sbA.dst | Atomic,  8, 8, C_regs[co],      A_regs[odd][ao], B_regs[odd][0]);
        dpaswTyped(8 | Atomic,            8, 8, C_regs[co + 8],  A_regs[odd][ao], B_regs[odd][4]);
        dpaswTyped(8 | Atomic,            8, 8, C_regs[co + 16], A_regs[odd][ao], B_regs[odd][8]);
        dpaswTyped(8 | sbA,               8, 8, C_regs[co + 24], A_regs[odd][ao], B_regs[odd][12]);
    }
}

GEMMSTONE_NAMESPACE_END
