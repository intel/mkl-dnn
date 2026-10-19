/*******************************************************************************
* Copyright 2023 Intel Corporation
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
#include "gemmstone/generator.hpp"


GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


template <HW hw>
void Generator<hw>::gemmStreamKPrepareSlice2(const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    auto done = state.ra.alloc_sub<uint32_t>();
    auto statusInc = state.ra.alloc_sub<uint32_t>();
    auto tempCInc = state.ra.alloc_sub<uint32_t>();

    avg(1, done, state.inputs.kSlicedTiles, 0);
    if (strategy.persistent)
        and_(1, state.inputs.flags, state.inputs.flags, ~uint32_t(FlagKSlicing | FlagKSlice2));
    if (state.inputs.statusBuffer.isValid())
        mulConstant(1, statusInc, done, strategy.statusFlagStride());
    if (state.inputs.tempC.isValid())
        mulConstant(1, tempCInc, done, tempCWGStride(problem, strategy));
    add(1, state.inputs.kSlicedTiles, state.inputs.kSlicedTiles, -done);
    if (state.inputs.statusBuffer.isValid())
        eadd(1, state.inputs.statusBuffer, state.inputs.statusBuffer, statusInc, strategy, state);
    if (state.inputs.tempC.isValid())
        eadd(1, state.inputs.tempC, state.inputs.tempC, tempCInc, strategy, state);
    if (strategy.persistent)
        eadd3(1, state.groupIDMN, state.groupCountMN, -state.inputs.kSlicedTiles, state.inputs.groupIDMN);

    state.ra.safeRelease(statusInc);
    state.ra.safeRelease(tempCInc);
    state.ra.safeRelease(done);
}

// Variable k-slicing/stream-k main logic.
template <HW hw>
void Generator<hw>::gemmStreamKSetup(Label &lKVPhaseDone, Label &lKernelDone,
                                     const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (!strategy.kParallelVariable) return;

    state.h0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
    Label lNoKSlice, lAlreadySliced, lGotSlice, lSync, lFirst;

    auto slicedTileIdx   = state.ra.alloc_sub<uint32_t>();
    auto nonKSyncedWGs   = state.ra.alloc_sub<uint16_t>();
    auto kSlicedTiles    = state.ra.alloc_sub<uint16_t>();
    auto temp            = state.ra.alloc_sub<uint32_t>();
    auto virtualKPadding = state.ra.alloc_sub<uint32_t>();
    auto kPadStorage     = state.ra.alloc_sub<uint32_t>();
    auto kUnsynced       = state.ra.alloc_sub<uint32_t>();
    auto &h0             = state.h0;
    auto h1              = state.ra.alloc_sub<uint32_t>();
    auto &k0Rem          = state.k0Rem;

    Subregister kPad = state.fullK;

    status << "Variable k-slicing" << status_stream::endl;

    // Find the size of our k-sliced region, if there are two regions ("phases").
    and_(1 | nz | f1[1], null.ud(), state.inputs.flags, FlagKSlice2);
    mov(1, kSlicedTiles, state.inputs.kSlicedTiles);
    avg(1 | f1[1], kSlicedTiles, state.inputs.kSlicedTiles, 0);

    // Check if we have reached the k-sliced region yet, and if so,
    //   if we need to do k-slicing computations.
    and_(1 | nz | f0[1], null.ud(), state.inputs.flags, FlagKSlicing);
    eadd3(1 | ge | f0[0], slicedTileIdx.d(), state.groupIDMN, -state.groupCountMN, state.inputs.kSlicedTiles);
    mov(1, state.h0, 0);
    if (strategy.kParallelLocal)
        mov(1, state.inputs.k, state.fullK);
    jmpi(1 | ~f0[0], lNoKSlice);

    if (!strategy.persistent) {
        // Check if we are in the second phase, and adjust as needed.
        add(1 | ge | f0[0], temp.d(), slicedTileIdx.d(), -state.inputs.groupStride);
        jmpi(1 | f0[1], lFirst);
        jmpi(1 | ~f0[0], lFirst);
        mov(1, slicedTileIdx, temp);
        gemmStreamKPrepareSlice2(problem, strategy, state);
        mov(1, kSlicedTiles, state.inputs.kSlicedTiles);
        mark(lFirst);
    }

    // Divide up k-space into two parts:
    //   k-synchronized: WGs have aligned h0, one tile per WG
    //   unsynchronized: scattered h0, possibly multiple tiles per WG.
    // At the same time, compute any virtual k-padding.
    emad(1, nonKSyncedWGs, state.inputs.groupStride, -state.inputs.kSyncSlabs, kSlicedTiles, strategy, state);
    if (strategy.kPadding)
        shl(1 | sat, virtualKPadding, state.inputs.k0, 1);
    emad(1, kUnsynced, state.fullK, state.inputs.k0, -state.inputs.kSyncSlabs, strategy, state);
    add(1 | ge | f0[0], temp.d(), slicedTileIdx, -nonKSyncedWGs);
    if (strategy.kPadding)
        min_(1, virtualKPadding, virtualKPadding, strategy.kPadding);
    alignUp(kUnsynced, kUnsynced, strategy.kAlign(problem), strategy, state);
    if (strategy.kPadding) {
        add(1, kUnsynced, kUnsynced, virtualKPadding);
        add(1, kPadStorage, state.fullK, virtualKPadding);
        kPad = kPadStorage;
    }

    jmpi(1 | f0[1], lAlreadySliced);
    jmpi(1 | f0[0], lSync);
    {
        // Unsynchronized section ("stream-k" part).
        // Each workgroup gets a roughly k0-sized range in k space, which may span
        //  multiple C tiles.

        // Traverse tiles backward:
        eadd3(1, temp, nonKSyncedWGs, -slicedTileIdx, -1);

        // Locate ending tile and k value:
        //    h1 <- (k0 * slicedTileIdx) % kUnsynced
        //  tile <- (k0 * slicedTileIdx) / kUnsynced
        mul(1, temp, state.inputs.k0, temp.uw());
        divDown(slicedTileIdx, temp, kUnsynced, state.inputs.kRecip, f1[0], strategy, state);
        emad(1, h1, temp, -kUnsynced, slicedTileIdx.uw(), strategy, state);

        // Restore tile order:
        eadd3(1 | lt | f0[0], slicedTileIdx.d(), kSlicedTiles, -slicedTileIdx, -1);
        add(1, h1, kUnsynced, -h1);

        // Keep track of total k work allotted to this WG.
        mov(1, k0Rem, state.inputs.k0);

        // Find beginning of k range:
        add(1 | sat, h0.ud(), h1, -state.inputs.k0);
    }
    jmpi(1, lGotSlice);
    mark(lSync);
    {
        // k-synchronized section. On entry, temp holds WG's index within this section.
        //    h0 <- (temp / kSlicedTiles) * k0 + kUnsynced
        //  tile <- (temp % kSlicedTiles)
        divMod(h0, slicedTileIdx, temp, kSlicedTiles, strategy, state);
        emad(1, h0, kUnsynced, state.inputs.k0, h0.uw(), strategy, state);
        mov(1, h1, kPad);
        mov(1, k0Rem, 0);
        cmp(1 | ge | f0[0], h0, kPad);
    }
    jmpi(1, lGotSlice);
    mark(lAlreadySliced);
    {
        // Starting a new tile in the unsynchronized section.
        add(1 | lt | f0[0], slicedTileIdx.d(), slicedTileIdx, -1);
        add(1 | sat, h0.ud(), kUnsynced, -k0Rem);
        mov(1, h1, kUnsynced);
    }
    mark(lGotSlice);

    or_(1, state.inputs.flags, state.inputs.flags, FlagKSlicing);

    // Early exit if no work for this WG.
    jmpi(1 | f0[0], lKVPhaseDone);

    // Check if our WG is responsible for beta scaling.
    // Choose the WG covering:
    //     h = 0   (no virtual padding);
    //     h = -1  (virtual padding).
    if (strategy.altFusedBeta) {
        if (strategy.kPadding)
            cmp(1 | lt | f1[0], h0, virtualKPadding);
        else
            cmp(1 | eq | f1[0], h0, 0);
    }

    // Clamp k range.
    add(1, temp, h0, state.inputs.k0);
    min_(1, h1, h1, temp);

    // Update remaining k work to do. TODO: can skip for k-sync'ed.
    add(1 | sat, temp.ud(), h1, -h0);
    add(1 | sat, k0Rem.ud(), k0Rem, -temp);

    // If needed, check how much k work is left after us on this tile.
    if (strategy.altFusedBeta && strategy.kPadding)
        add(1 | sat, temp.ud(), kPad, -h1);

    // Part 2 of beta scaling check.
    if (strategy.altFusedBeta && strategy.kPadding)
        cmp(1 | f1[0] | ge | f1[0], h1, virtualKPadding);

    // Remove (virtual) padding from bottom of the k range.
    if (strategy.kPadding) {
        add(1 | sat, h1.ud(), h1, -virtualKPadding);
        add(1 | sat, h0.ud(), h0, -virtualKPadding);
    }

    // Remove (alignment) padding from top of the k range.
    min_(1, h1, h1, state.fullK);

    // Determine physical k range for this slice, and update group ID.
    add(1 | sat | ze | f1[1], state.inputs.k, h1, -h0);
    eadd3(1, state.groupIDMN, slicedTileIdx, state.groupCountMN, -state.inputs.kSlicedTiles);
    min_(1, state.inputs.k, state.inputs.k, state.inputs.k0);

    // Assign responsibility for beta scaling.
    if (strategy.altFusedBeta)
        or_(1 | f1[0], state.inputs.flags, state.inputs.flags, FlagDidBeta);

    // If needed, check if we are handling the entire tile.
    if (strategy.fuseBeta || strategy.fusePostOps)
        cmp(1 | eq | f0[1], state.inputs.k, state.fullK);

    // With k padding, we might have a zero-size slice. Handle appropriately.
    if (strategy.kPadding) {
        if (strategy.altFusedBeta) {
            Label lContinue;
            jmpi(1 | ~f1[1], lContinue);
            // This WG is wholly inside the padded region.
            // We still need to do beta scaling if it's our responsibility, unless
            //   there's only one other WG on the tile.
            cmp(1 | ge | f0[1], state.inputs.k0, temp);
            jmpi(1 | ~f1[0], lKernelDone);                  // Skip if zero-size and not doing beta scaling.
            jmpi(1 | f0[1], lKernelDone);                   // Skip if another WG is taking care of this entire tile.
            mark(lContinue);
        } else
            jmpi(1 | f1[1], lKernelDone);                   // Skip if zero-size.
    }

    // Beta/post-op fusing: check if we need to do it.
    if (strategy.fuseBeta || strategy.fusePostOps) {
        Label lNoCheck;
        jmpi(1 | f0[1], lNoCheck);
        or_(1, state.inputs.flags, state.inputs.flags, FlagKPartitioned);
        gemmFusedBetaPOInit(slicedTileIdx, problem, strategy, state);
        mark(lNoCheck);
    }

    mark(lNoKSlice);

    // Further slice k range within workgroup if requested.
    // k local size must be a power of 2.
    if (strategy.kParallelLocal) {
        if (!is_zero_or_pow2(strategy.wg[LoopK])) stub();
        if (strategy.kInterleave) stub();
        state.threadK0 = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
        if (strategy.fuseBeta || strategy.fusePostOps) {
            state.wgK = state.ra.alloc_sub<uint32_t>(getHint(HintType::LongTerm, strategy));
            mov(1, state.wgK, state.inputs.k);
        }
        fbl(1, temp, state.lszK);
        eadd3(1, state.threadK0, state.inputs.k, state.lszK, -1);
        shr(1, state.threadK0, state.threadK0, temp);
        alignUp(state.threadK0, state.threadK0, strategy.kAlign(problem), strategy, state);
        mul(1, temp, state.threadK0, state.lidK.uw());
        add(1 | sat, state.inputs.k.ud(), state.inputs.k, -temp);
        add(1, state.h0, state.h0, temp);
        min_(1, state.inputs.k, state.inputs.k, state.threadK0);
    }

    state.ra.safeRelease(slicedTileIdx);
    state.ra.safeRelease(nonKSyncedWGs);
    state.ra.safeRelease(kSlicedTiles);
    state.ra.safeRelease(temp);
    state.ra.safeRelease(virtualKPadding);
    state.ra.safeRelease(kPadStorage);
    state.ra.safeRelease(kUnsynced);
    state.ra.safeRelease(h1);
}

GEMMSTONE_NAMESPACE_END
