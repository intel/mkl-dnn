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
#include "gemmstone/generator.hpp"
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "map.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


// Adjust address registers as needed for a newly-created subblock.
template <HW hw>
void Generator<hw>::adjustSubblockAddrs(const RegisterLayout &sublayout, const vector<RegisterRange> &subaddrs,
                                        const RegisterLayout &layout, const vector<RegisterRange> &addrs,
                                        const CommonStrategy &strategy, const CommonState &state)
{
    auto &atype = layout.addressing();
    auto &astrategy = layout.addressingStrategy();

    bool a64 = (astrategy.base.getModel() == ModelA64);

    int nsubs = sublayout.blocks();
    int nblocks = layout.blocks();

    for (int isub = 0; isub < nsubs; isub++) {
        // Find parent block by comparing address registers.
        auto &subaddr = subaddrs[isub];
        const RegisterBlock *pptr = nullptr;
        for (int i = 0; i < nblocks; i++) {
            if (addrs[i].getBase() == subaddr.getBase()) {
                pptr = &layout[i];
                break;
            }
        }
        if (!pptr) stub();

        auto &block = *pptr;
        auto &subblock = sublayout[isub];

        auto off = getAddr0Offset(block, atype, astrategy);
        auto suboff = getAddr0Offset(subblock, atype, astrategy);

        // Perform any necessary shifts/moves. Moves are only for non-A64 block->pseudoblock settings.
        if (suboff != off) {
            if (subblock.simdSize != 1) stub(); // Need to prepare more pseudoblock addresses.
            mov<uint32_t>(1, subaddr[0][suboff], subaddr[0][off]);
        }
        if (subblock.addrShift != block.addrShift) {
            map(hw, a64 ? Type::u64 : Type::u32, subaddr, subaddr, strategy, [&](int simd, GRF r, GRF _) {
                auto shift = block.addrShift - subblock.addrShift;
                (shift > 0) ? eshl(simd, r, r, +shift, strategy, state)
                            : eshr(simd, r, r, -shift, strategy, state);
            });
        }

        if (isBlock2D(astrategy.accessType)) {
            // Adjust 2D block header as needed.
            int bw, bh, bcount;
            bool memCM = isColMajor(atype.layout);
            auto RegisterBlock::* nw = memCM ? &RegisterBlock::nr : &RegisterBlock::nc;
            auto RegisterBlock::* nh = memCM ? &RegisterBlock::nc : &RegisterBlock::nr;
            bool remW = memCM ? subblock.remainderR : subblock.remainderC;
            bool remH = memCM ? subblock.remainderC : subblock.remainderR;
            subblock.getBlock2DWH(bw, bh, bcount, atype, astrategy.prefetch);

            if (!astrategy.address2D) {
                if (subblock.*nw != block.*nw || subblock.count != block.count) {
                    int newW = bw * bcount * subblock.ebytes - 1;
                    remW ? min_(1, subaddr[0].ud(2), subaddr[0].ud(2), newW)
                         : mov(1, subaddr[0].ud(2), newW);
                }
                if (subblock.*nh != block.*nh) {
                    int newH = bh * subblock.ebytes - 1;
                    remH ? min_(1, subaddr[0].ud(3), subaddr[0].ud(3), newH)
                         : mov(1, subaddr[0].ud(3), newH);
                }
            }
            if (subaddr.isValid())
                updateBlock2DSizes(GRF(subaddr[0].getBase()), subblock, block, atype, astrategy.prefetch);
        }
    }
}

// Update block 2D width/height/count parameters as needed after cloning an address register.
template <HW hw>
void Generator<hw>::updateBlock2DSizes(GRF addr, const RegisterBlock &dst, const RegisterBlock &src, const MatrixAddressing &atype, bool prefetch)
{
    int bw, bh, bcount;
    dst.getBlock2DWH(bw, bh, bcount, atype, prefetch);

    if (dst.nr != src.nr || dst.nc != src.nc || dst.count != src.count)
        mov(1, addr.ud(7), (bw - 1) | ((bh - 1) << 8) | ((bcount - 1) << 16));
}

// Attempt to add remainder handling to an existing block. Returns true if successful.
template <HW hw>
bool Generator<hw>::tryAddRemainder(Type T, RegisterBlock &block, bool remainderR, bool remainderC, RemainderOptions remOpts,
                                    const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    auto blockNew = block;
    blockNew.remainderR |= remainderR;
    blockNew.remainderC |= remainderC;

    auto curAccessType = block.implAccessType(atype, astrategy);

    if (curAccessType == AccessType::Block) {
        if (astrategy.newDP) return false;
        if (hw >= HW::XeHPC) return false;
    }

    bool remChanged = (remainderR && !block.remainderR)
                   || (remainderC && !block.remainderC);

    if (remChanged && !isBlock2D(curAccessType)) {
        if (!blockNew.tryRecreate(hw, T, atype, astrategy, remOpts))
            return false;
        blockNew.offsetAddr = block.offsetAddr;
        if (blockNew.nr != block.nr || blockNew.nc != block.nc)
            return false;
        if (blockNew.implAccessType(atype, astrategy) != curAccessType)
            return false;
        if (curAccessType != AccessType::Block) {
            if (blockNew.ebytes != block.ebytes)
                return false;
            if (blockNew.ebytes == 1 && blockNew.count != block.count)
                return false;
        }
    }

    block = blockNew;
    return true;
}

// Attempt to add remainder handling to a layout without changing its blocks. Returns true if successful.
template <HW hw>
bool Generator<hw>::tryAddRemainder(RegisterLayout &layout, bool remainderR, bool remainderC, RemainderOptions remOpts)
{
    auto layoutNew = layout;
    for (auto &block: layoutNew) {
        if (!tryAddRemainder(layout.type(), block, remainderR, remainderC, remOpts,
                             layout.addressing(), layout.addressingStrategy()))
            return false;
    }
    std::swap(layout, layoutNew);
    return true;
}

// Add remainder handling to a layout without changing its blocks. Throws if unsuccessful.
template <HW hw>
void Generator<hw>::addRemainder(RegisterLayout &layout, bool remainderR, bool remainderC, RemainderOptions remOpts)
{
    for (auto &block: layout) {
        if (!tryAddRemainder(layout.type(), block, remainderR, remainderC, remOpts,
                             layout.addressing(), layout.addressingStrategy()))
            stub();
    }
}

// Add remainder handling to a layout, setting it up again from scratch if required.
template <HW hw>
void Generator<hw>::addRemainder(RegisterLayout &layout, vector<RegisterRange> &addrs, const Subregister &ld,
                                 bool remainderR, bool remainderC, RemainderOptions remOpts,
                                 const CommonStrategy &strategy, CommonState &state, int dataRegs)
{
    // Check if masking can be trivially enabled without changing the layout.
    if (tryAddRemainder(layout, remainderR, remainderC, remOpts))
        return;

    // If not, tear down the old layout and create a new one in its place, recalculating address registers.
    bool remR = remainderR || layout.hasRemainders(true, false);
    bool remC = remainderC || layout.hasRemainders(false, true);
    RegisterLayout layoutNew(hw, layout.type(), layout.rows(), layout.cols(),
                             layout.addressing(), layout.addressingStrategy(),
                             remR, remC, false, remOpts);
    if (dataRegs < 0) dataRegs = layout.regs();
    if (layoutNew.regs() > dataRegs) stub();
    if (layoutNew.colMajor() != layout.colMajor()) stub();

    int shift = 0;
    auto addr0 = getOriginAddr(layout, addrs, &shift);
    std::swap(layout, layoutNew);
    if (shift > 0)
        shl(1, addr0, addr0, shift);
    safeReleaseRanges(addrs, state);
    state.ra.claim(addr0);

    Address2DParams params2D{};
    if (layout.addressingStrategy().address2D) stub();
    allocAddrRegs(hw, addrs, layout, state);
    setupAddr(addrs, addr0, layout, ld, strategy, state, params2D);

    state.ra.safeRelease(addr0);
}


GEMMSTONE_NAMESPACE_END
