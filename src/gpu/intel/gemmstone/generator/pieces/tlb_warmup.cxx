/*******************************************************************************
* Copyright 2025 Intel Corporation
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
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "state_utils.hpp"
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;



template <HW hw>
void Generator<hw>::gemmTLBWarmup(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto lid = state.ra.allocSub<uint32_t>();
    int whose = 0;

    emad(1, lid, state.inputs.localIDM, state.inputs.localIDN, strategy.wg[LoopM], strategy, state);
    if (strategy.kParallelLocal)
        emad(1, lid, lid, state.inputs.localIDK, strategy.wg[LoopM] * strategy.wg[LoopN], strategy, state);

    if (problem.quantized2DA()) {
        auto mq = state.ra.allocSub<uint32_t>();
        auto kq = state.ra.allocSub<uint32_t>();
        divDown(mq, state.inputs.m, problem.aqGroupM, strategy, state);
        divDown(kq, state.inputs.k, problem.aqGroupK, strategy, state);
        if (problem.aScale2D()) {
            tlbWarmup(problem.A_scale, strategy.A_scale, state.inputs.aScalePtr,
                      mq, kq, state.ldaScale, lid, whose++, problem, strategy, state);
        }
        if (problem.aOffset2D()) {
            tlbWarmup(problem.AO, strategy.AO, state.inputs.aoPtr,
                      mq, kq, state.ldao, lid, whose++, problem, strategy, state);
        }
        state.ra.safeRelease(mq);
        state.ra.safeRelease(kq);
    }

    if (problem.quantized2DB()) {
        auto kq = state.ra.allocSub<uint32_t>();
        auto nq = state.ra.allocSub<uint32_t>();
        divDown(kq, state.inputs.k, problem.bqGroupK, strategy, state);
        divDown(nq, state.inputs.n, problem.bqGroupN, strategy, state);
        if (problem.bScale2D()) {
            tlbWarmup(problem.B_scale, strategy.B_scale, state.inputs.bScalePtr,
                      kq, nq, state.ldbScale, lid, whose++, problem, strategy, state);
        }
        if (problem.bOffset2D()) {
            tlbWarmup(problem.BO, strategy.BO, state.inputs.boPtr,
                      kq, nq, state.ldbo, lid, whose++, problem, strategy, state);
        }
        state.ra.safeRelease(kq);
        state.ra.safeRelease(nq);
    }

    tlbWarmup(problem.A, strategy.A, state.effA,
              state.inputs.m, state.inputs.k, state.inputs.lda, lid, whose++,
              problem, strategy, state);
    tlbWarmup(problem.B, strategy.B, state.effB,
              state.inputs.k, state.inputs.n, state.inputs.ldb, lid, whose++,
              problem, strategy, state);

    state.ra.safeRelease(lid);
}

template <HW hw>
void Generator<hw>::tlbWarmup(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                              const Subregister &ptr, const Subregister &r, const Subregister &c,
                              const Subregister &ld, const Subregister &lid, int whose,
                              const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state)
{
    auto flag = state.raVFlag.alloc();
    const uint32_t byteLimit = 256 * 1024 * 1024;

    auto bytes = state.ra.allocSub<uint64_t>();
    emul(1, bytes, ld, isColMajor(atype.layout) ? c : r, strategy, state);
    cmp(1 | nz | flag, bytes.ud(1), 0);
    min_(1, bytes.ud(), bytes.ud(), byteLimit);
    mov(1 | flag, bytes.ud(), byteLimit);

    state.raVFlag.safeRelease(flag);

    tlbWarmup(astrategy.base, ptr, bytes.ud(), lid, whose, problem, strategy, state);

    state.ra.safeRelease(bytes);
}

template <HW hw>
void Generator<hw>::tlbWarmup(AddressBase base, const Subregister &ptr, const Subregister &bytes,
                              const Subregister &lid, int whose,
                              const CommonProblem &problem, const CommonStrategy &strategy, CommonState &state)
{
    bool a64 = base.isA64();
    auto Taddr = a64 ? DataType::uq : DataType::ud;
    const int simd = elementsPerGRF<uint32_t>(hw);
    const int log2Stride = 16;      // 64kb stride.
    const int log2TwiddleStride = 6;

    int udStride = a64 ? 2 : 1;
    auto addr = state.ra.allocRange(udStride);
    auto addr0 = addr[0].retype(Taddr);
    auto addrLo = addr0.ud(0)(udStride);
    auto off = state.ra.allocRange(udStride);
    auto off0 = off[0].ud(0)(udStride);
    auto twiddle = state.ra.alloc().ud();
    auto data = state.ra.alloc().ud();
    auto count = state.ra.alloc().d();
    auto flag = state.raVFlag.alloc();

    extendIndexVec(simd, state);

    auto iv = accessIndexVec(0, state)(1);

    cmp(1 | nz | flag, lid, whose);         /* Check if we are responsible thread */

    shl(simd, off0, iv, log2Stride);
    shl(simd, twiddle, iv, log2TwiddleStride);
    eadd(simd, addr0, ptr, off0, strategy, state);
    xor_(simd, addrLo, addrLo, twiddle);    /* Perturb low bits to avoid cache hotspotting */

    add(1, count, bytes, ((simd + 1) << log2Stride) - 1);
    shr(1, count, count, log2Stride);
    add(simd, count, count[0], -iv);

    Label lTop, lSkip;
    jmpi(1 | flag, lSkip);

    mark(lTop);
    add(simd | gt | flag, count, count, -simd);
    if (hw >= HW::Xe3p)
        load(simd | flag, null, D8U32 | L1C_L2C_L3C, base, addr);
    else if (hw >= HW::XeHPC)
        load(simd | flag, null, D8U32 | L1C_L3C, base, addr);
    else if (hw >= HW::XeHPG)
        load(simd | flag, data, D8U32 | L1C_L3C, base, addr);
    else
        load(simd | flag, data, scattered_byte(), base, addr);
    xor_(simd, addrLo, addrLo, twiddle);
    add(simd, twiddle, twiddle, simd << log2TwiddleStride);
    and_(simd, twiddle, twiddle, 0xFFF);    /* Don't cross 4K page boundaries */
    eadd(simd, addr0, addr0, simd << log2Stride, strategy, state);
    xor_(simd, addrLo, addrLo, twiddle);
    jmpi(1 | flag, lTop);
    mark(lSkip);

    releaseIndexVec(state);
    state.raVFlag.safeRelease(flag);
    state.ra.safeRelease(off);
    state.ra.safeRelease(twiddle);
    state.ra.safeRelease(addr);
    state.ra.safeRelease(data);
    state.ra.safeRelease(count);
}

GEMMSTONE_NAMESPACE_END
