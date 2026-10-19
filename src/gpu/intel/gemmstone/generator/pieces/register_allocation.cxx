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
#include "gemmstone/generator.hpp"
#include "layout_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


// Common register allocator hints.
template <HW hw>
Bundle Generator<hw>::getHint(HintType type)
{
    switch (type) {
        case HintType::Bank0: return Bundle(0, Bundle::any);
        case HintType::Bank1: return Bundle(1, Bundle::any);
        case HintType::LongTerm0: return Bundle(0, Bundle::any);
        case HintType::LongTerm1: return Bundle(1, Bundle::any);
        default:              break;
    }

    return Bundle();
}

template <HW hw>
Bundle Generator<hw>::getHint(HintType type, const CommonStrategy &strategy)
{
    return getHint(type);
}

// GEMM register allocation hints.
template <HW hw>
Bundle Generator<hw>::getHint(HintType type, const GEMMStrategy &strategy)
{
    switch (strategy.registerScheme) {
        case GEMMStrategy::CSeparate:
            switch (type) {
                case HintType::A0Broadcast:
                case HintType::A0:          return Bundle(1, Bundle::any);
                case HintType::A1Broadcast:
                case HintType::A1:          return Bundle(0, Bundle::any);
                case HintType::B0Broadcast:
                case HintType::B0:          return Bundle(0, Bundle::any);
                case HintType::B1Broadcast:
                case HintType::B1:          return Bundle(1, Bundle::any);
                case HintType::C:           return Bundle(0, 0);
                case HintType::CLoad:       return Bundle(1, Bundle::any);
                default:                    break;
            }
            break;
        case GEMMStrategy::ACB:
        case GEMMStrategy::BCA:
            if (strategy.systolic) switch (type) {
                case HintType::A0:
                case HintType::B0:          return Bundle(0, Bundle::any);
                case HintType::A1:
                case HintType::B1:          return Bundle(1, Bundle::any);
                case HintType::A0Broadcast:
                case HintType::B0Broadcast: return Bundle(1, Bundle::any);
                case HintType::A1Broadcast:
                case HintType::B1Broadcast: return Bundle(0, Bundle::any);
                case HintType::C:           return Bundle(0, Bundle::any);
                default:                    break;
            }
            /* else fall through */
        case GEMMStrategy::VNC:
            switch (type) {
                case HintType::A0:
                case HintType::B0:          return Bundle(1, Bundle::any);
                case HintType::A1:
                case HintType::B1:          return Bundle(0, Bundle::any);
                case HintType::A0Broadcast:
                case HintType::B0Broadcast: return Bundle(0, Bundle::any);
                case HintType::A1Broadcast:
                case HintType::B1Broadcast: return Bundle(1, Bundle::any);
                case HintType::C:           return Bundle(0, Bundle::any);
                default:                    break;
            }
            break;
        case GEMMStrategy::ABInterleave:
            switch (type) {
                case HintType::A0:
                case HintType::A1:
                case HintType::A0Broadcast:
                case HintType::A1Broadcast: return Bundle(1, 0);
                case HintType::B0:
                case HintType::B1:
                case HintType::B0Broadcast:
                case HintType::B1Broadcast: return Bundle(1, 4);
                case HintType::C:           return Bundle(0, Bundle::any);
                default:                    break;
            }
            break;
        case GEMMStrategy::NSeparate:
            switch (type) {
                case HintType::A0:
                case HintType::B0:          return Bundle(1, Bundle::any);
                case HintType::A1:
                case HintType::B1:          return Bundle(0, Bundle::any);
                case HintType::A0Broadcast:
                case HintType::B0Broadcast:
                case HintType::A1Broadcast:
                case HintType::B1Broadcast: return Bundle();
                case HintType::C:           return Bundle(0, Bundle::any);
                case HintType::C1:          return Bundle(1, Bundle::any);
                default:                    break;
            }
            break;
        case GEMMStrategy::VAvoid:
            switch (type) {
                case HintType::A0:
                case HintType::B0:          return Bundle(0, Bundle::any);
                case HintType::A1:
                case HintType::B1:          return Bundle(1, Bundle::any);
                case HintType::A0Broadcast:
                case HintType::B0Broadcast:
                case HintType::A1Broadcast:
                case HintType::B1Broadcast: return Bundle(1, Bundle::any);
                case HintType::C:           return Bundle(0, Bundle::any);
                case HintType::C1:          return Bundle(1, Bundle::any);
                default:                    break;
            }
            break;
    }

    return getHint(type);
}

// Allocate register ranges for A/B/C.
template <HW hw>
void Generator<hw>::gemmAllocRegs(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    // Summary: order of allocations is important.
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    auto raHW = strategy.raHW;

    auto A_copies = strategy.A_copies;
    auto B_copies = strategy.B_copies;
    int A_regCount  = state.A_layout.regs();
    int Ar_regCount = state.Ar_layout.regs();
    int B_regCount  = state.B_layout.regs();
    int Br_regCount = state.Br_layout.regs();

    bool repackC = !state.Cr_layout.empty();
    const auto &C_layoutExt = !state.C_layoutExtNonatomicUnmasked.empty() ? state.C_layoutExtNonatomicUnmasked :
                                       !state.C_layoutExtUnmasked.empty() ? state.C_layoutExtUnmasked
                                                                          : state.C_layoutExt;
    const auto &C_layout = repackC ? state.Cr_layout :
                       state.copyC ? state.C_layout
                                   : C_layoutExt;

    int C_regCountPerBuffer = C_layout.regs();
    int C_regCount = state.C_buffers * C_regCountPerBuffer;
    GRFMultirange C_regs;

    bool globalCM = C_layout.colMajor();

    auto hintA0 =  globalCM ? HintType::A0 : HintType::A0Broadcast;
    auto hintB0 = !globalCM ? HintType::B0 : HintType::B0Broadcast;

    auto Tv =  globalCM ? Ta : Tb;
    auto Tn = !globalCM ? Ta : Tb;
    auto Tv_load = globalCM ? state.Ta_load : state.Tb_load;
    auto Tn_load = globalCM ? state.Tb_load : state.Ta_load;

    auto &A_layout = state.A_layout;
    auto &B_layout = state.B_layout;

    auto &V_layout =    globalCM ? A_layout        : B_layout;
    auto &Vr_layout =   globalCM ? state.Ar_layout : state.Br_layout;
    auto &V_regs =      globalCM ? state.A_regs    : state.B_regs;
    auto &Vr_regs =     globalCM ? state.Ar_regs   : state.Br_regs;
    auto V_copies =     globalCM ? A_copies        : B_copies;
    auto V_regCount =   globalCM ? A_regCount      : B_regCount;
    auto Vr_regCount =  globalCM ? Ar_regCount     : Br_regCount;
    auto &N_layout =   !globalCM ? A_layout        : B_layout;
    auto &N_regs =     !globalCM ? state.A_regs    : state.B_regs;
    auto &Nr_regs =    !globalCM ? state.Ar_regs   : state.Br_regs;
    auto N_copies =    !globalCM ? A_copies        : B_copies;
    auto N_regCount =  !globalCM ? A_regCount      : B_regCount;
    auto Nr_regCount = !globalCM ? Ar_regCount     : Br_regCount;

    int C_chunk = state.copyC ? 1 : C_layoutExt.maxLoadBlock();
    int Vr_chunk = 2, Nr_chunk = 2;

    C_chunk = alignup_pow2(C_chunk, Bundle(0, 0).group_size(raHW) * 2);
    if (strategy.systolic) {
        auto params = systolicParams(problem);
        C_chunk = std::max(C_chunk, (params.osys * params.rcountMax * Tc.real()) / GRF::bytes(hw));
        Vr_chunk = std::max(Vr_chunk, (params.osys * params.ksys * Tv.real()) / GRF::bytes(hw));
        Nr_chunk = std::max(Nr_chunk, (params.rcountMax * params.ksys * Tn.real()) / GRF::bytes(hw));
    }

    state.C_accCount = strategy.cAccumulators ? AccumulatorRegister::count(hw, strategy.GRFs, Tc.ngen()) : 0;

    state.A_regs.resize(A_copies);
    state.B_regs.resize(B_copies);

    switch (strategy.registerScheme) {
        case GEMMStrategy::CSeparate:
        {
            // Standard allocation. A and B allocated together in lower half of registers.
            // Interleave allocation of A and B to minimize wasted registers. Test the waters to find out
            //  whether to try bank 0 or 1 first.
            int bases[2];
            for (int bank = 0; bank < 2; bank++) {
                auto r = state.ra.alloc_range(4, Bundle(bank, Bundle::any));
                bases[bank] = r.getBase();
                state.ra.safeRelease(r);
            }

            // Order of the banks.
            int banks[2];
            banks[0] = (bases[1] < bases[0]) ? 1 : 0;
            banks[1] = 1 - banks[0];

            // Allocate all the registers needed from bank 0, then all the registers needed from bank 1.
            for (int bank : banks) {
                if (getHint(hintA0, strategy).bank_id == bank) {
                    for (int copy = 0; copy < A_copies; copy++)
                        state.A_regs[copy] = state.ra.alloc_range(A_regCount, getHint(hintA0, strategy));
                    if (Ar_regCount > 0)
                        state.Ar_regs = state.ra.alloc_range(Ar_regCount, getHint(hintA0, strategy));
                }

                if (getHint(hintB0, strategy).bank_id == bank) {
                    for (int copy = 0; copy < B_copies; copy++)
                        state.B_regs[copy] = state.ra.alloc_range(B_regCount, getHint(hintB0, strategy));
                    if (Br_regCount > 0)
                        state.Br_regs = state.ra.alloc_range(Br_regCount, getHint(hintB0, strategy));
                }
            }

            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount, getHint(HintType::C, strategy));
            break;
        }
        case GEMMStrategy::ACB:
            for (int copy = 0; copy < A_copies; copy++)
                state.A_regs[copy] = state.ra.alloc_range(A_regCount, getHint(hintA0, strategy));
            if (Ar_regCount > 0)
                state.Ar_regs = state.ra.alloc_range(Ar_regCount, getHint(hintA0, strategy));

            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount, getHint(HintType::C, strategy));

            for (int copy = 0; copy < B_copies; copy++)
                state.B_regs[copy] = state.ra.alloc_range(B_regCount, getHint(hintB0, strategy));
            if (Br_regCount > 0)
                state.Br_regs = state.ra.alloc_range(Br_regCount, getHint(hintB0, strategy));
            break;
        case GEMMStrategy::BCA:
            for (int copy = 0; copy < B_copies; copy++)
                state.B_regs[copy] = state.ra.alloc_range(B_regCount, getHint(hintB0, strategy));
            if (Br_regCount > 0)
                state.Br_regs = state.ra.alloc_range(Br_regCount, getHint(hintB0, strategy));

            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount, getHint(HintType::C, strategy));

            for (int copy = 0; copy < A_copies; copy++)
                state.A_regs[copy] = state.ra.alloc_range(A_regCount, getHint(hintA0, strategy));
            if (Ar_regCount > 0)
                state.Ar_regs = state.ra.alloc_range(Ar_regCount, getHint(hintA0, strategy));
            break;
        case GEMMStrategy::VNC: {
            // Assign non-broadcast input matrix (V), then broadcast input matrix (N), then C.
            auto unrollVBytes = strategy.unroll[globalCM ? LoopM : LoopN] * Tv;
            auto unrollNBytes = strategy.unroll[globalCM ? LoopN : LoopM] * Tn;
            auto regUnrollV = div_up(unrollVBytes, GRF::bytes(hw));
            auto regUnrollN = div_up(unrollNBytes, GRF::bytes(hw));
            auto hintV = getHint(HintType::A0, strategy);
            auto hintN = getHint((regUnrollN == 1) ? HintType::A0 : HintType::A0Broadcast, strategy);   // Put V and N in same bundle if we can avoid N<->C conflicts.
            auto hintC = getHint(HintType::C, strategy);
            RegisterRange tempPadding;

            for (int copy = 0; copy < V_copies; copy++)
                V_regs[copy] = state.ra.alloc_range(V_regCount, hintV);
            if (Vr_regCount > 0)
                Vr_regs = state.ra.alloc_range(Vr_regCount, hintV);

            N_regs[0] = state.ra.alloc_range(N_regCount, hintN);

            // Check if A * B outer product 0 has a bank conflict. If so, move N to avoid this.
            auto stride = Bundle(0, 0).stride(raHW);
            auto offN = (N_regs[0][0].getBase() - V_regs[0][0].getBase()) & (stride - 1);
            auto offNMin = offN - ((regUnrollV - 1) & ~1);
            auto offNMax = offN + regUnrollN - 1;
            if (offNMax >= stride) offNMax -= stride, offNMin -= stride;
            if (offNMin <= 0) {
                unsigned obAlign = Bundle(0, 0).group_size(raHW);
                if (hintN.bank_id != Bundle::any) obAlign *= 2;
                offNMax = alignup_pow2(offNMax, obAlign);
                safeReleaseRanges(N_regs[0], state);
                tempPadding = state.ra.alloc_range(offNMax, hintN);
                N_regs[0] = state.ra.alloc_range(N_regCount, hintN);
            }

            for (int copy = 1; copy < N_copies; copy++)
                N_regs[copy] = state.ra.alloc_range(N_regCount, hintN);
            if (Nr_regCount > 0)
                Nr_regs = state.ra.alloc_range(Nr_regCount, hintN);

            state.ra.safeRelease(tempPadding);

            C_regs = state.ra.alloc_range(C_regCount - state.C_accCount, hintC);
            break;
        }
        case GEMMStrategy::ABInterleave: {
            // Interleave A and B, place C afterward.
            auto chunk = Bundle(0, 0).stride(raHW) >> 1;

            // Test allocation. Put A earlier if it has more registers.
            int A_regTotal = A_regCount * A_copies + Ar_regCount;
            int B_regTotal = B_regCount * B_copies + Br_regCount;
            auto hintA = BundleGroup(raHW) | getHint(HintType::A0, strategy);
            auto hintB = BundleGroup(raHW) | getHint(HintType::B0, strategy);
            auto hintC = BundleGroup(raHW) | getHint(HintType::C, strategy);
            auto testA = state.ra.allocRange(8, hintA);
            auto testB = state.ra.allocRange(8, hintB);
            if ((testA.getBase() < testB.getBase()) == (A_regTotal < B_regTotal))
                std::swap(hintA, hintB);
            state.ra.safeRelease(testA);
            state.ra.safeRelease(testB);

            for (int copy = 0; copy < A_copies; copy++)
                state.A_regs[copy] = chunkAlloc(A_regCount, chunk, hintA, state);
            if (Ar_regCount > 0)
                state.Ar_regs = chunkAlloc(Ar_regCount, chunk, hintA, state);
            for (int copy = 0; copy < B_copies; copy++)
                state.B_regs[copy] = chunkAlloc(B_regCount, chunk, hintB, state);
            if (Br_regCount > 0)
                state.Br_regs = chunkAlloc(Br_regCount, chunk, hintB, state);
            C_regs = state.ra.allocRange(C_regCount - state.C_accCount, hintC);
            break;
        }
        case GEMMStrategy::NSeparate: {
            // Broadcast matrix (N) has dedicated bundle(s) (both banks)
            // V and C start in opposite banks in other bundles.
            if (state.C_accCount > 0) stub();

            bool repackV = (Vr_regCount > 0);
            bool repackN = (Nr_regCount > 0);
            int bundles = Bundle::bundle_count(raHW) * Bundle::bank_count(raHW);
            int bregsConsecutive = Bundle(0, 0).group_size(raHW);
            int bregs = strategy.GRFs / bundles;
            int V_chunk = (repackV ? Vr_chunk : V_layout.maxLoadBlock());
            int N_chunk = (repackN ? Nr_chunk : N_layout.maxLoadBlock());
            V_chunk = std::max(V_chunk, strategy.nSeparateChunk);
            N_chunk = std::max(N_chunk, strategy.nSeparateChunk);
            bool forceVNChunk = (strategy.nSeparateChunk > 0);

            int N_nregs = repackN ? Nr_regCount : N_regCount * N_copies;
            int N_nbundles = std::max(div_up(N_chunk, bregsConsecutive),
                                      div_up(N_nregs, bregs));
            BundleGroup N_bundles(raHW), VC_bundles(raHW);

            auto hintV0 = BundleGroup(raHW) | getHint(HintType::A0, strategy);
            auto hintV1 = BundleGroup(raHW) | getHint(HintType::A1, strategy);
            auto hintN = BundleGroup(raHW) | getHint(HintType::A0Broadcast, strategy);
            auto hintC0 = BundleGroup(raHW) | getHint(HintType::C, strategy);
            auto hintC1 = BundleGroup(raHW) | getHint(HintType::C1, strategy);

            // Give bundles starting at the end to broadcast matrix.
            for (int bundle = Bundle::bundle_count(raHW) - 1; bundle >= 0; bundle--) {
                for (int bank = Bundle::bank_count(raHW) - 1; bank >= 0; bank--) {
                    if (N_nbundles-- > 0)
                        N_bundles |= Bundle(bank, bundle);
                    else
                        VC_bundles |= Bundle(bank, bundle);
                }
            }

#ifdef STANDALONE
            auto N_mask = N_bundles.regMask(0) & ((uint64_t(1) << Bundle(0,0).stride(hw)) - 1);
            status << "N bundle mask: " << std::hex << N_mask << std::dec << status_stream::endl;
#endif

            if (repackV)
                Vr_regs = chunkAlloc(Vr_regCount, V_chunk, hintV0, VC_bundles, state);
            else for (int copy = 0; copy < V_copies; copy++)
                V_regs[copy] = splitOrChunkAlloc(raHW, Tv_load, V_layout, V_chunk, {hintV0, hintV1}, VC_bundles, state, forceVNChunk);
            if (!strategy.systolic)
                C_regs = trySplitAlloc(raHW, Tc, C_layout, {hintC0, hintC1}, VC_bundles, state, state.C_buffers);
            if (C_regs.empty())
                C_regs = chunkAlloc(C_regCount, C_chunk, hintC0, VC_bundles, state);
            if (repackN)
                Nr_regs = chunkAlloc(Nr_regCount, N_chunk, hintN, N_bundles, state);
            else for (int copy = 0; copy < N_copies; copy++)
                N_regs[copy] = splitOrChunkAlloc(raHW, Tn_load, N_layout, N_chunk, {hintN, hintN}, N_bundles, state, forceVNChunk);

            if (repackV)
              for (int copy = 0; copy < V_copies; copy++)
                V_regs[copy] = splitOrChunkAlloc(
                    raHW, Tv_load, V_layout, V_chunk,
                    {BundleGroup::AllBundles(), BundleGroup::AllBundles()},
                    BundleGroup::AllBundles(), state);
            if (repackN)
              for (int copy = 0; copy < N_copies; copy++)
                N_regs[copy] = splitOrChunkAlloc(
                    raHW, Tn_load, N_layout, N_chunk,
                    {BundleGroup::AllBundles(), BundleGroup::AllBundles()},
                    BundleGroup::AllBundles(), state);
            break;
        }
        case GEMMStrategy::VAvoid: {
            // Broadcast matrix (N) has dedicated starting bank.
            // V and C share starting banks, but C allocations chosen to miss matching V allocations.
            auto hintV = getHint(HintType::A0, strategy);
            auto hintN = getHint(HintType::A0Broadcast, strategy);
            auto hintC = getHint(HintType::C, strategy);

            for (int copy = 0; copy < N_copies; copy++)
                N_regs[copy] = state.ra.alloc_range(N_regCount, hintN);
            if (Nr_regCount > 0)
                Nr_regs = state.ra.alloc_range(Nr_regCount, hintN);

            for (int copy = 0; copy < V_copies; copy++)
                V_regs[copy] = state.ra.alloc_range(V_regCount, hintV);
            if (Vr_regCount > 0)
                Vr_regs = state.ra.alloc_range(Vr_regCount, hintV);

            auto V_innerLayout = (Vr_regCount > 0) ? Vr_layout : V_layout;
            int kv  = globalCM ? V_innerLayout.cols() : V_innerLayout.rows();
            int mnv = globalCM ? V_innerLayout.rows() : V_innerLayout.cols();
            int mnStride = mnv;
            if (repackC)
                mnStride = globalCM ? state.Cr_layout.rows() : state.Cr_layout.cols();

            int minOPCount = minOuterProductCount(problem, strategy);
            int lastMN0 = -1;
            int sliceRegs = 0;
            BundleGroup V_bundles(raHW);

            vector<GRFMultirange> C_extra(state.C_buffers - 1);
            auto allocSlice = [&]() {
                if (sliceRegs <= 0) return;
                auto C_bundles = ~V_bundles & hintC;

                C_regs.append(chunkAlloc(sliceRegs, C_chunk, C_bundles, BundleGroup::AllBundles(), state));
                for (int copy = 1; copy < state.C_buffers; copy++)
                   C_extra[copy - 1].append(chunkAlloc(sliceRegs, C_chunk, C_bundles, BundleGroup::AllBundles(), state));

                sliceRegs = 0;
            };

            for (const auto &block: C_layout) {
                int mn0 = globalCM ? block.offsetR : block.offsetC;
                if (mn0 == lastMN0) {
                    sliceRegs += block.nregs();
                    continue;
                }

                allocSlice();

                V_bundles = BundleGroup(raHW);
                for (int mn = mn0; mn < mnv; mn += mnStride) {
                for (int h0 = 0; h0 < kv; h0 += minOPCount) {
                    int r = globalCM ? mn : h0;
                    int c = globalCM ? h0 : mn;
                    int comp = 0;
                    if (Vr_regCount == 0) for (int copy = 0; copy < V_copies; copy++) {
                        auto V0 = V_layout.find(r, c, V_regs[copy], nullptr, nullptr, 0, comp);
                        V_bundles |= Bundle::locate(raHW, V0);
                    } else {
                        auto V0 = Vr_layout.find(r, c, Vr_regs, nullptr, nullptr, 0, comp);
                        V_bundles |= Bundle::locate(raHW, V0);
                    }
                }
                }

                lastMN0 = mn0;
                sliceRegs = block.nregs();
            }

            allocSlice();

            for (int copy = 1; copy < state.C_buffers; copy++)
                C_regs.append(C_extra[copy - 1]);
        }
    }

    // Track clobbers which cannot be determined from decoded assembly.
    auto addClobbers = [&](const std::vector<GRFMultirange> &regs) {
        for (const auto &reg : regs) {
            for (auto &r : reg.ranges) {
                knownClobbers.add(r.getBase(), r.getLen());
            }
        }
    };

    if (strategy.A.base.getModel() != ngen::ModelInvalid) { addClobbers(state.A_regs); }
    if (strategy.B.base.getModel() != ngen::ModelInvalid) { addClobbers(state.B_regs); }

    if (repackC) {
        state.Cr_regs = C_regs;
        C_regCountPerBuffer = state.C_layout.regs();
        C_regCount = state.C_buffers * C_regCountPerBuffer;
        C_regs = chunkAlloc(C_regCount, C_chunk, state);
    }

    // Assign C_regs, adding in GRFs (in place of accumulators) to use later.
    state.C_regs.resize(state.C_buffers);

    auto it = C_regs.ranges.begin();
    int off = -state.C_accCount;
    for (int buf = 0; buf < state.C_buffers; buf++) {
        for (int todo = C_regCountPerBuffer; todo > 0;) {
            if (it == C_regs.ranges.end())
                stub("Not enough C registers allocated.");
            int left = it->getLen() - off;
            int take = std::min(left, todo);
            state.C_regs[buf].ranges.push_back(GRFRange(it->getBase() + off, take));
            todo -= take;
            off += take;
            if (off >= it->getLen())
                off = 0, it++;
        }
    }

    // Allocate registers for SLM copies.
    state.Ai_regs.resize(strategy.slmCopies);
    state.Bi_regs.resize(strategy.slmCopies);
    if (strategy.slmA) {
        for (int q = 0; q < strategy.slmCopies; q++)
            state.Ai_regs[q] = state.ra.alloc_range(state.Ai_regCount);
        addClobbers(state.Ai_regs);
    }
    if (strategy.slmB) {
        for (int q = 0; q < strategy.slmCopies; q++)
            state.Bi_regs[q] = state.ra.alloc_range(state.Bi_regCount);
        addClobbers(state.Bi_regs);
    }

    // Allocate registers for A/B sums.
    state.Asr_regs = state.ra.alloc_range(state.Asr_layout.regs());
    state.Bsr_regs = state.ra.alloc_range(state.Bsr_layout.regs());
    state.As_regs = state.ra.alloc_range(state.As_layout.regs());
    state.Bs_regs = state.ra.alloc_range(state.Bs_layout.regs());

    // Allocate registers for A/B prefetch.
    auto needsPFRegs = [](const MatrixAddressingStrategy &as) {
        return as.prefetch && !as.newDP;
    };
    if (state.Ap_layout.valid() && needsPFRegs(state.Ap_layout.addressingStrategy()))
        state.Ap_regs = state.ra.alloc_range(state.Ap_layout.regs());
    if (state.Bp_layout.valid() && needsPFRegs(state.Bp_layout.addressingStrategy()))
        state.Bp_regs = state.ra.alloc_range(state.Bp_layout.regs());

    // bdpas has additional GRF conflict requirements: A scales must be in the same bank as A.
    auto Ascale_hint = state.useBDPAS ? getHint(HintType::A0, strategy) : Bundle();

    // Allocate registers for A/B quantization parameters.
    state.A_offsetRegs = state.ra.alloc_range(state.A_offsetLayout.regs());
    state.B_offsetRegs = state.ra.alloc_range(state.B_offsetLayout.regs());
    state.Ar_offsetRegs = state.ra.alloc_range(state.Ar_offsetLayout.regs());
    state.Br_offsetRegs = state.ra.alloc_range(state.Br_offsetLayout.regs());
    state.A_scaleRegs = state.ra.alloc_range(state.A_scaleLayout.regs(), state.Ar_scaleLayout.empty() ? Ascale_hint : Bundle());
    state.B_scaleRegs = state.ra.alloc_range(state.B_scaleLayout.regs());
    state.Ar_scaleRegs = state.ra.alloc_range(state.Ar_scaleLayout.regs(), Ascale_hint);
    state.Br_scaleRegs = state.ra.alloc_range(state.Br_scaleLayout.regs());
    state.Ag_regs = state.ra.alloc_range(state.Ag_layout.regs());
    state.Bg_regs = state.ra.alloc_range(state.Bg_layout.regs());
    state.Agr_regs = state.ra.alloc_range(state.Agr_layout.regs());
    state.Bgr_regs = state.ra.alloc_range(state.Bgr_layout.regs());

#ifdef STANDALONE
    if (problem.readCheck)
        state.readFailures = state.C_regs[0][0].ud(0);
#endif
}

template <HW hw>
void Generator<hw>::gemmAllocAoBoRegs(const GEMMStrategy &strategy, GEMMState &state)
{
    bool allocAo = false, allocBo = false;
    int Ao_nregs = state.Ao_layout.regs(), Bo_nregs = state.Bo_layout.regs();

    if (strategy.slmA && state.Ao_regs.empty() && !state.aioShare) {
        allocAo = true;
        if (strategy.slmRepackAhead == 0 && strategy.A_copies == 1) {
            auto &defaultRegs = state.A_regs[0];
            allocAo = (defaultRegs.getLen() < Ao_nregs);

            if (!allocAo) {
                state.Ao_regs = defaultRegs.subrange(0, Ao_nregs);
                state.aoReuseA = true;
            }
        }
    }

    if (strategy.slmB && state.Bo_regs.empty() && !state.bioShare) {
        allocBo = true;
        if (strategy.slmRepackAhead == 0 && strategy.B_copies == 1) {
            auto &defaultRegs = state.B_regs[0];
            allocBo = (defaultRegs.getLen() < Bo_nregs);

            if (!allocBo) {
                state.Bo_regs = defaultRegs.subrange(0, Bo_nregs);
                state.boReuseB = true;
            }
        }
    }

    if (allocAo && !state.allocedAo) {
        state.allocedAo = true;
        state.Ao_regs = state.ra.alloc_range(Ao_nregs);
    }

    if (allocBo && !state.allocedBo) {
        state.allocedBo = true;
        state.Bo_regs = state.ra.alloc_range(Bo_nregs);
    }
}

GEMMSTONE_NAMESPACE_END
