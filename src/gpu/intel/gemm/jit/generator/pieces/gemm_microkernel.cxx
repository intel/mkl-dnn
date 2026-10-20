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

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;


template <HW hw>
void Generator<hw>::gemmMicrokernel(GEMMProblem problem, GEMMStrategy strategy, const ngen::InterfaceHandler &interface_)
{
    GEMMState state(hw, strategy);

    interface = interface_;

    problem.autoTypeConversions(strategy.systolic);
    gemmInitState(problem, strategy, state);
    for (int q = 0; q < 2; q++)
        state.ra.safeRelease(state.emulate.temp[q]);

#ifdef STANDALONE
    gemmKernelNotes(problem, strategy);
#endif

    outputCRange = GRFMultirange();
    outputCLayout.clear();
    knownClobbers.clear();

    strategy.forceWGUpdate = WGFixed;
    strategy.forceFixedWGK = true;
    strategy.activeThreads = strategy.wg[LoopM] * strategy.wg[LoopN] * strategy.wg[LoopK];

    strategy.AO.base = A64;
    strategy.BO.base = A64;
    strategy.A_scale.base = A64;
    strategy.B_scale.base = A64;

    state.isNested = true;

    /* Reserve the host kernel's thread payload, below the microkernel's arguments. */
    auto argumentBase = interface.getArgumentBase();
    if (argumentBase.isInvalid() || argumentBase.getBase() < 1)
        stub("Microkernel argument base not set");
    int claimGRFs = argumentBase.getBase();
    // XXX: Keep the legacy claim as a workaround for NVL-P to avoid
    // performance regressions. IGC is sensitive to register range changes.
    if (getProductFamily() == ngen::ProductFamily::NVLP)
        claimGRFs = std::max<int>(claimGRFs, 10);
    state.ra.claim(GRFRange(0, claimGRFs));

    moveR0(strategy, state);

    state.fullK = state.inputs.k;

    bool registerC = strategy.registerOutput();

    // Locate and claim additional inputs.
    auto getAndClaim = [&](const char *name) {
        auto sub = interface.getArgument(name);
        state.ra.claim(sub);
        return sub;
    };

    state.i0 = getAndClaim("i0");
    state.j0 = getAndClaim("j0");
    state.h0 = getAndClaim("h0");

    state.lidM = getAndClaim("local_id_m").uw();
    state.lidN = getAndClaim("local_id_n").uw();
    if (strategy.kParallelLocal)
        state.lidK = getAndClaim("local_id_k").uw();

    state.allocEmulate64Temp(strategy.emulate);

    setDefaultNoMask();
    setDefaultAutoSWSB();

    // Save and modify dispatch mask as needed.
    Subregister dmaskSave;
    int minSIMD = GRF::bytes(hw) >> 2;
    if (minSIMD < state.internalSIMD()) {
        dmaskSave = state.ra.alloc_sub<uint32_t>();
        mov(1, dmaskSave, sr0[2]);
        mov(1 | SWSB<AllPipes>(1), sr0[2], uint32_t(uint64_t(1) << state.internalSIMD()) - 1);
    }

    // Synchronize and save flag registers from host kernel.
    syncall();
    Subregister flagSave[4];
    for (int i = 0; i < FlagRegister::count(hw); i++) {
        flagSave[i] = state.ra.alloc_sub<uint32_t>();
        mov(1, flagSave[i], FlagRegister(i));
    }

    // Save accumulator registers used by the host kernel, if we'll use them here
    // TODO: Only save the accumulators used by the kernel. Potentially move into the package,
    // which already decodes/scans the microkernel binary for register usage.
    const int accSaveCount = AccumulatorRegister::count(hw, strategy.GRFs, problem.Tc.real().ngen());
    const int accElts = GRF::bytes(hw) >> 2;  // dwords per accumulator register
    GRFRange accSave = state.ra.alloc_range(accSaveCount);
    for (int i = 0; i < accSave.getLen(); i++)
        mov<uint32_t>(accElts, accSave[i], AccumulatorRegister(i));

    knownClobbers.addFlag(0, FlagRegister::count(hw));
    knownClobbers.addAcc(0, accSaveCount);

    // Beginning of microkernel:
    //   - check32
    //   - fused ID calculation
    //   - ld scaling
    //   - i0/j0/h0 calculations (inside WG)
    //   - A/B/C offsets
    bool wgCheck = wgRemCheck(problem, strategy);
    bool gemmtBarriers = problem.gemmt() && strategy.needsBarrier();

    auto &k = state.inputs.k;
    auto &k0 = state.inputs.k0;

    state.lid0 = (strategy.fusedLoop == LoopN) ? state.lidN : state.lidM;
    getFusedID(strategy.unroll[strategy.fusedLoop], problem, strategy, state);

    gemmScaleInputs(problem, strategy, state);

    if (wgCheck || gemmtBarriers) {
        state.wgI0 = copySubregister(state.i0, state);
        state.wgJ0 = copySubregister(state.j0, state);
    }

    if (strategy.kParallelLocal) {
        /* Select k0 automatically */
        int wgK = strategy.wg[LoopK];
        if (!is_zero_or_pow2(wgK)) stub();
        k0 = state.ra.alloc_sub<uint32_t>();
        add(1, k0, k, wgK - 1);
        shr(1, k0, k0, ilog2(wgK));
        alignUp(k0, k0, strategy.kAlign(problem), strategy, state);
    }

    emad(1, state.i0, state.i0, state.lidM, strategy.unroll[LoopM], strategy, state);
    emad(1, state.j0, state.j0, state.lidN, strategy.unroll[LoopN], strategy, state);
    if (strategy.kParallelLocal) {
        bool anyAB2D = strategy.A.address2D || strategy.B.address2D
                || (strategy.prefetchA && strategy.A_prefetch.address2D)
                || (strategy.prefetchB && strategy.B_prefetch.address2D);
        if (anyAB2D) {
            state.fullK = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            mov(1, state.fullK, state.inputs.k);
        }

        if (strategy.kInterleave) {
            emad(1, state.h0, state.h0, state.lidK, strategy.kInterleaveChunk, strategy, state);
        } else {
            emad(1, state.h0, state.h0, k0, state.lidK, strategy, state);
        }

        add(1 | sat, k.ud(), k, -state.h0);

        if (strategy.kInterleave) {
            // k <- floor(k / (chunk * k local size)) * chunk + min(k % (chunk * k local size), chunk)
            auto chunk = strategy.kInterleaveChunk;
            auto wgChunk = chunk * strategy.wg[LoopK];
            auto temp1 = state.ra.alloc_sub<uint32_t>();
            auto temp2 = state.ra.alloc_sub<uint32_t>();
            divDown(temp1, k.ud(), wgChunk, strategy, state);
            emad(1, temp2, k.ud(), -temp1, wgChunk, strategy, state);
            emul(1, temp1, temp1, chunk, strategy, state);
            min_(1, temp2, temp2, chunk);
            add(1, state.inputs.k.ud(), temp1, temp2);

            state.ra.safeRelease(temp1);
            state.ra.safeRelease(temp2);
        }

        min_(1, k, k, k0);
        state.threadK0 = k0;
    }

    gemmCalcWGRemainders(problem, strategy, state);
    gemmCheck32(problem, strategy, state);

    auto &i0p = (strategy.coopA == CoopSplit::FullK) ? state.wgI0 : state.i0;
    auto &j0p = (strategy.coopB == CoopSplit::FullK) ? state.wgJ0 : state.j0;

    gemmOffsetABC(false, state.i0, state.j0, state.h0, i0p, j0p, problem, strategy, state, true, true, !registerC);

    if (!(strategy.prefetchA && strategy.A_prefetch.address2D)) state.ra.safeRelease(state.wgI0);
    if (!(strategy.prefetchB && strategy.B_prefetch.address2D)) state.ra.safeRelease(state.wgJ0);

    if (strategy.prefetchA && state.effAp.isInvalid()) state.effAp = state.effA;
    if (strategy.prefetchB && state.effBp.isInvalid()) state.effBp = state.effB;

    gemmSubkernel(problem, strategy, state);

    // Restore accumulator registers if saved above.
    for (int i = 0; i < accSave.getLen(); i++)
        mov<uint32_t>(accElts, AccumulatorRegister(i), accSave[i]);
    state.ra.safeRelease(accSave);

    // Restore flag registers and dispatch mask and return to host kernel.
    for (int i = 0; i < FlagRegister::count(hw); i++)
        mov(1, FlagRegister(i), flagSave[i]);
    if (dmaskSave.isValid())
        mov(1, sr0[2], dmaskSave);
    syncall();
}

#if defined(BINARY_OUTPUT) || !defined(STANDALONE)
static inline microkernel::StructuredType::Type microType(Type T);

template <HW hw>
microkernel::Package Generator<hw>::gemmMicrokernelPackage(const GEMMProblem &problem_, const GEMMStrategy &strategy, const ngen::InterfaceHandler &interface_,
                                                     const microkernel::Protocol &protocol, uint32_t gmdid, bool transposeC)
{
    using namespace microkernel;
    Package package;

    auto problem = problem_;
    problem.autoTypeConversions(strategy.systolic);

    gemmMicrokernel(problem, strategy, interface_);

    package.protocol = protocol;
    package.gmdidCompat = gmdid;
    package.binary = this->getCode();

    for (auto parg: package.protocol.arguments()) {
        Argument arg;
        arg.name = parg.name;

        /* Map microkernel argument name to gemmstone internal name (iname) */
        auto iname = arg.name;

        if (transposeC) {
            if (iname == "a")               iname = "b";
            else if (iname == "b")          iname = "a";
            else if (iname == "lda")        iname = "ldb";
            else if (iname == "ldb")        iname = "lda";
            else if (iname == "m")          iname = "n";
            else if (iname == "n")          iname = "m";
            else if (iname == "i0")         iname = "j0";
            else if (iname == "j0")         iname = "i0";
            else if (iname == "local_id_m") iname = "local_id_n";
            else if (iname == "local_id_n") iname = "local_id_m";
            else if (iname == "a_scale")    iname = "b_scale";
            else if (iname == "b_scale")    iname = "a_scale";
            else if (iname == "a_offset")   iname = "b_offset";
            else if (iname == "b_offset")   iname = "a_offset";
            else if (iname == "ldaq")       iname = "ldbq";
            else if (iname == "ldbq")       iname = "ldaq";
        }

        if (iname == "a")             iname = "A";
        else if (iname == "b")        iname = "B";
        else if (iname == "slm")      iname = "slm_base";
        else if (iname == "a_offset") iname = problem.aOffsetHostScalar() ? "ao" : "ao_ptr";
        else if (iname == "b_offset") iname = problem.bOffsetHostScalar() ? "bo" : "bo_ptr";
        else if (iname == "a_scale")  iname = "a_scale_ptr";
        else if (iname == "b_scale")  iname = "b_scale_ptr";

        /* Locate argument registers */
        if (iname == "c") {
            int tileM = strategy.unroll[LoopM];
            int tileN = strategy.unroll[LoopN];
            int blockM = outputCLayout[0].nr;
            int blockN = outputCLayout[0].nc;

            for (auto &block: outputCLayout) {
                if (blockM != block.nr) stub();
                if (blockN != block.nc) stub();
            }
            if (!outputCLayout.colMajor()) stub(); /* Swap dims and block ordering */

            int blockGRF = 8;
            for (auto &r: outputCRange.ranges)
                blockGRF = gcd(blockGRF, r.getLen());
            int maxBlock = blockGRF * elementsPerGRF(hw, problem.Tc);

            if (blockM > maxBlock) {
                if (blockM % maxBlock) stub();
                blockM = maxBlock;
                blockN = 1;
            } else if (blockM * blockN > maxBlock) {
                int split = (blockM * blockN / maxBlock);
                if (blockM * blockN % maxBlock || blockN % split) stub();
                blockN /= split;
            }

            arg.location.resize((tileM * tileN) / (blockM * blockN));
            int idx = 0;
            for (int bj = 0; bj < tileN; bj += blockN) {
                for (int bi = 0; bi < tileM; bi += blockM) {
                    auto topLeft = outputCLayout.find(bi, bj, outputCRange);
                    arg.location[idx].boffset = topLeft.getBase() * GRF::bytes(hw) + topLeft.getByteOffset();
                    arg.location[idx].blen    = blockM * blockN * problem.Tc;
                    idx++;
                }
            }

            arg.sizes.dims[0] = tileM;
            arg.sizes.dims[1] = tileN;
            arg.sizes.block[0] = blockM;
            arg.sizes.block[1] = blockN;
        } else {
            auto reg = interface.getArgument(iname);
            arg.location.resize(1);
            arg.location[0].boffset = reg.getBase() * GRF::bytes(hw) + reg.getByteOffset();
            arg.location[0].blen = reg.getBytes();
        }

        /* Provide actual argument types */
        if (iname == "A")           arg.actualType = microType(problem.Ta_ext);
        if (iname == "B")           arg.actualType = microType(problem.Tb_ext);
        if (iname == "c")           arg.actualType = microType(problem.Tc);
        if (iname == "ao_ptr")      arg.actualType = microType(problem.Tao);
        if (iname == "bo_ptr")      arg.actualType = microType(problem.Tbo);
        if (iname == "a_scale_ptr") arg.actualType = microType(problem.Ta_scale);
        if (iname == "b_scale_ptr") arg.actualType = microType(problem.Tb_scale);

        package.arguments.push_back(std::move(arg));
    }

    auto slmSize = std::max(gemmSLMSize(hw, problem, strategy, true),
                            gemmPerKSLMSize(hw, problem, strategy) * strategy.wg[LoopK]);

    auto effLoopM = !transposeC ? LoopM : LoopN;
    auto effLoopN = !transposeC ? LoopN : LoopM;
    package.settings.push_back({"sg_tile_m", strategy.unroll[effLoopM]});
    package.settings.push_back({"sg_tile_n", strategy.unroll[effLoopN]});
    package.settings.push_back({"wg_tile_m", strategy.wgTile(effLoopM)});
    package.settings.push_back({"wg_tile_n", strategy.wgTile(effLoopN)});
    package.settings.push_back({"sg_per_wg_m", strategy.wg[effLoopM]});
    package.settings.push_back({"sg_per_wg_n", strategy.wg[effLoopN]});
    package.settings.push_back({"sg_per_wg_k", strategy.wg[LoopK]});
    package.settings.push_back({"slm_size", int(slmSize)});

    package.barrierCount = interface.getBarrierCount();
    package.argumentBase = interface.getArgumentBase().getBase() * GRF::bytes(hw);

    package.finalize(knownClobbers);

    return package;
}

static inline microkernel::StructuredType::Type microType(Type T)
{
    using ST = microkernel::StructuredType::Type;
#define CASE(x) case Type::x: return ST::x;
    switch (T) {
        CASE(f64)
        CASE(f32)
        CASE(f16)
        CASE(bf16)
        CASE(s32)
        CASE(s16)
        CASE(s8)
        CASE(s4)
        CASE(u3)
        CASE(u32)
        CASE(u16)
        CASE(u8)
        CASE(u4)
        CASE(bf8)
        CASE(hf8)
        CASE(f8_e8m0)
        CASE(f4_e2m1)
        default: stub("Unsupported type");
    }
#undef CASE
}

#endif

GEMMSTONE_NAMESPACE_END
