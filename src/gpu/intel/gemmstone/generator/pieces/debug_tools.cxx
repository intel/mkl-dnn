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
#include "hw_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


template <typename T> static inline T randomFloat() { return 2 * (T(std::rand()) / T(RAND_MAX)) - 1; }
template <typename T> static inline T randomInt()   { return T(std::rand() | (std::rand() << 16)); }

// Initialize a matrix with some dummy data for mockup kernels.
template <HW hw>
void Generator<hw>::mockInitMatrix(Type T, const GRFMultirange &r, const CommonStrategy &strategy, bool random)
{
    if (random && !r.empty()) {
        T = T.real();
        int ne = elementsPerGRF(hw, T);
#define MOCK_INIT(TT, randomFunc) \
                        for (int i = 0; i < ne; i++) \
                            mov<TT>(1, r[0][i], randomFunc<TT>()); \
                        for (int i = 1; i < r.getLen(); i++) \
                            add<TT>(ne, r[i], r[0], randomFunc<TT>());
#define MOCK_INIT_REUSE(TT, randomFunc) \
                        for (int i = 0; i < ne; i++) \
                            mov<TT>(1, r[0][i], randomFunc<TT>()); \
                        for (int i = 1; i < r.getLen(); i++) \
                            add<TT>(ne, r[i], r[0], r[0][i%ne]);
        switch (T) {
            case Type::f16: MOCK_INIT(half, randomFloat); break;
            case Type::f32: MOCK_INIT(float, randomFloat); break;
            case Type::f64: MOCK_INIT_REUSE(double, randomFloat); break;
            default:
                if (!T.isInteger()) stub();
                MOCK_INIT(uint32_t, randomInt);
                break;
#undef MOCK_INIT
#undef MOCK_INIT_REUSE
        }
    } else
        zeroMatrix(r, strategy);
}

// Initialize SLM with NaNs to help catch usage of uninitialized SLM data.
template <HW hw>
void Generator<hw>::mockInitSLM(Type T, size_t bytes, CommonState &state)
{
    if (bytes == 0) return;

    int simdAddr = elementsPerGRF<uint32_t>(hw);
    int simdData = elementsPerGRF(hw, T);

    auto data = state.ra.alloc();
    auto addr = state.ra.alloc();
    auto counter = state.ra.alloc_sub<int32_t>();

    mov(8, addr.uw(), Immediate::uv(0,1,2,3,4,5,6,7));
    mov(simdData, data.retype(T.ngen()), cast(T, std::numeric_limits<double>::quiet_NaN()));
    shl(8, addr.ud(), addr.uw(), 2);
    mov(1, counter, div_up(int(bytes), simdData * 4));
    if (simdAddr == 16)
        add(8, addr.ud(8)(1), addr.ud(0)(1), 8 * 4);

    Label lTop;
    mark(lTop);
    add(1 | gt | f0[0], counter, counter, -1);
    store(simdAddr, surface_dword(ChannelMask::r), SLM, addr, data);
    add<uint32_t>(simdAddr, addr, addr, simdAddr * 4);
    jmpi(1 | f0[0], lTop);

    slmfence(data);
    fencewait();
    barrier(data);

    state.ra.safeRelease(data);
    state.ra.safeRelease(addr);
    state.ra.safeRelease(counter);
}

// Initialize matrix registers prior to read check loads.
template <HW hw>
void Generator<hw>::readCheckInitMatrix(const GRFMultirange &r, const CommonStrategy &strategy)
{
    auto init = (strategy.strongReadCheck ? ~uint32_t(0) : uint32_t(0));
    map<uint32_t>(hw, r, r, strategy, [&](int esize, GRF reg, GRF _) {
        mov(esize, reg, init);
    });
}

template <HW hw>
void Generator<hw>::readCheckInitMatrix(const GRFMultirange &r, const RegisterLayout &layout, const CommonStrategy &strategy)
{
    auto init = (strategy.strongReadCheck ? ~uint32_t(0) : uint32_t(0));
    map<uint32_t>(hw, r, layout, strategy, [&](int esize, RegisterRegion rr) {
        mov(esize, rr, init);
    });
}

// Check if the input GRF range is entirely zero (all 1s for strong read check).
// Read checks are used for automatically detecting out-of-bounds accesses.
template <HW hw>
void Generator<hw>::readCheck(const GRFMultirange &r, const CommonStrategy &strategy, CommonState &state,
                              const RegisterLayout *layout)
{
    using T = uint32_t;

    if (r.getLen() == 0) return;

    constexpr auto ne = elementsPerGRF<T>(hw);
    GRF first;
    int firstSIMD = ne;

    auto reduce = [&](int simd, RegData r1, RegData r2) {
        if (strategy.strongReadCheck)
            and_<T>(simd, r1, r1, r2);
        else
            or_<T>(simd, r1, r1, r2);
    };

    if (layout) {
        map<T>(hw, r, *layout, strategy, [&](int simd, RegisterRegion r) {
            if (first.isValid()) {
                reduce(simd, first, r);
                if (simd > firstSIMD) stub();
            } else {
                first = GRF(r.getBase());
                firstSIMD = simd;
            }
        });
    } else {
        first = r[0];
        for (int i = 1; i < r.getLen(); i++)
            reduce(ne, first, r[i]);
    }
    if (first.isInvalid()) {    // Layout covers less than 1 DW
        first = r[0];
        firstSIMD = 1;

        switch ((*layout)[0].bytes) {
            case 1: first = first.ub(); break;
            case 2: first = first.uw(); break;
            default: break;
        }
    }
    if (firstSIMD > ne) {
        reduce(ne, first, first.advance(1));
        firstSIMD = ne;
    }
    {
        for (int simd = firstSIMD >> 1; simd > 0; simd >>= 1)
            reduce(simd, first[0](1), first[simd](1));
        if (state.readFailures != r[0].sub<T>(0))
            reduce(1, state.readFailures, first[0]);
    }
}


// Prepare read check data before storage.
template <HW hw>
void Generator<hw>::gemmFinalizeReadCheck(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (problem.readCheck) {
        {
            auto rf = state.readFailures.ud();
            if (strategy.strongReadCheck)
                xor_(1, rf, rf, 0xFFFFFFFF);
            min_(1, rf, rf, 1);
            if (strategy.strongReadCheck)
                add(1, rf, rf, -1);
            else
                mov(1, rf, -rf);
            if (!problem.Tc.isFP())
                and_(1, rf, rf, 0xAAAAAAAA);
        }
    }
}

// Store read check data in the upper-left corner of C.
template <HW hw>
void Generator<hw>::gemmStoreReadCheck(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (problem.readCheck) {
        if (state.readFailures.getOffset() != 0) stub();
        auto rf = GRF(state.readFailures.getBase()).ud();

        int cBytes = problem.Tc.paddedSize();
        for (int i = 1; i < cBytes / 4; i++)
            mov(1, rf[i], rf[0]);

        bool lsc = strategy.C.newDP;
        bool stateless = strategy.C.base.isStateless();
        auto htemp = state.ra.alloc();

        emov(1, stateless ? htemp[0].uq() : htemp[0].ud(), state.effC[0], strategy, state);
        if (!lsc && !stateless && cBytes > 4)
            shr<uint32_t>(1, htemp, htemp, ilog2(cBytes));

        if (lsc) switch (cBytes) {
            case 1:  store(1, D8U32,             strategy.C.base, htemp, rf); break;
            case 2:  store(1, D16U32,            strategy.C.base, htemp, rf); break;
            default: store(1, D32T(cBytes >> 2), strategy.C.base, htemp, rf); break;
        } else switch (cBytes) {
            case 1:
            case 2:
            case 4:  store(1, scattered_byte(cBytes),   strategy.C.base, htemp, rf); break;
            case 8:  store(1, scattered_qword(cBytes),  strategy.C.base, htemp, rf); break;
            default: store(1, block_oword(cBytes >> 4), strategy.C.base, htemp, rf); break;
        }
        state.ra.safeRelease(htemp);
    }
}

// Initialize performance tracking.
template <HW hw>
void Generator<hw>::gemmInitTracking(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (!strategy.tracking.enabled) return;

    /* Need to implement logic to generate linear IDs... */
    if (!strategy.linearOrder()) stub();
    if (strategy.kParallel || problem.batch != BatchMode::None) stub();

    /* We are called very early, before dividing out subgroup size from local IDs */
    uint16_t sss = ngen::utils::log2(strategy.subgroupSize);
    auto lid0 = interface.getLocalID(0);

    shr(1, lid0, lid0, sss);

    auto temp1 = state.ra.alloc_sub<uint32_t>();
    auto temp2 = state.ra.alloc_sub<uint32_t>();
    auto &wg = strategy.wg;

    mulConstant(1, temp1, state.inputs.groupIDMN, wg[LoopM] * wg[LoopN] * wg[LoopK]);
    emad(1, temp2, state.inputs.localIDM, state.inputs.localIDN, wg[LoopM], strategy, state);
    if (strategy.kParallelLocal)
        emad(1, temp2, temp2, state.inputs.localIDK, wg[LoopM] * wg[LoopN], strategy, state);
    add(1, temp1, temp1, temp2);
    shl(1, lid0, lid0, sss);    /* restore lid0 for later */
    mul(1, dbg0, temp1, state.tracking.stride.uw());

    state.ra.safeRelease(temp1);
    state.ra.safeRelease(temp2);
    state.ra.safeRelease(state.tracking.stride);
}

// Add a timepoint to tracking data.
template <HW hw>
template <typename TT>
void Generator<hw>::addTimepoint(TT type, CommonState &state)
{
    int ne = elementsPerGRF<float>(hw);

    auto header = state.ra.try_alloc();
    auto data = state.ra.try_alloc();
    bool alloced = header.isValid() && data.isValid();

    if (!alloced) {
        state.ra.release(header);
        state.ra.release(data);

        if (state.r0_info == acc0.ud()) stub();

        mov<float>(2 * ne, acc0, r0);
        header = r0;
        data = r1;
    }

    and_<uint32_t>(1, data[0], tm0[2], 1);
    mov<uint32_t>(1, data[1], tm0[0]);
    mov<uint32_t>(1, header, dbg0);
    or_<uint32_t>(1, data[0], data[0], type);

    if (hw >= HW::XeHPG)
        store(1, D32 | V2T, Surface(state.tracking.surface), header, data);
    else {
        add<uint32_t>(1, header[1], dbg0, 4);
        store(2, surface_dword(ChannelMask::r), Surface(state.tracking.surface), header, data);
    }

    add<uint32_t>(1, dbg0, header, 8);

    if (alloced) {
        state.ra.release(header);
        state.ra.release(data);
    } else
        mov<float>(2 * ne, r0, acc0);
}

GEMMSTONE_NAMESPACE_END
