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

#include "gemmstone/strategy.hpp"

#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "register_layout.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;

static int int3RowSpreadOffset(int byte, int ld, int crosspack)
{
    return (byte / crosspack) * (ld * crosspack) + byte % crosspack;
}

namespace SendSpec {
int MaxBlock2DWidthBytes(ngen::HW hw, bool prefetch) { return (hw == HW::Xe3p && prefetch) ? 256 : 64; }
} // namespace SendSpec

RegisterBlock::RegisterBlock(HW hw_, Type T, int r, int c, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                             bool remainderR_, bool remainderC_, bool writable_, RemainderOptions remOpts,
                             int maxRBlock, int maxCBlock)
        : RegisterBlock(hw_, T, r, c, atype, astrategy, remainderR_, remainderC_, writable_, remOpts, maxRBlock, maxCBlock, false)
{
    if (!valid()) stub("Could not create register block.");
}

RegisterBlock RegisterBlock::tryCreate(HW hw_, Type T, int r, int c, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                       bool remainderR_, bool remainderC_, bool writable_, RemainderOptions remOpts,
                                       int maxRBlock, int maxCBlock)
{
    return RegisterBlock(hw_, T, r, c, atype, astrategy, remainderR_, remainderC_, writable_, remOpts, maxRBlock, maxCBlock, true);
}

void RegisterBlock::recreate(HW hw_, Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                             RemainderOptions remOpts)
{
    if (!tryRecreate(hw_, T, atype, astrategy, remOpts)) stub("Could not recreate register block.");
}

bool RegisterBlock::tryRecreate(HW hw_, Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                             RemainderOptions remOpts)
{
    auto offR = offsetR, offC = offsetC;
    auto offB = offsetBytes, b = bytes;
    auto comp = component;

    auto blockNew = RegisterBlock::tryCreate(hw_, T, nr, nc, atype, astrategy, bool(remainderR), bool(remainderC), bool(writable), remOpts);
    if (!blockNew)
        return false;
    *this = blockNew;

    offsetR = offR;
    offsetC = offC;
    offsetBytes = offB;
    bytes = b;              // NB: may need updating by caller depending on new parameters
    component = comp;
    return true;
}

// Set up a RegisterBlock structure.
RegisterBlock::RegisterBlock(HW hw_, Type T, int r, int c, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                             bool remainderR_, bool remainderC_, bool writable_, RemainderOptions remOpts,
                             int maxRBlock, int maxCBlock, bool)
        : hw(hw_), remainderR(remainderR_), remainderC(remainderC_), writable(writable_)
{
    bool avoidFragment = (remOpts & AllowFragment) == 0;
    bool allowDesc = (remOpts & AllowDescriptors) && (hw < HW::Xe2);
    bool allowFixedMasks = (remOpts & NoFixedMasks) == 0;
    bool prefetch = astrategy.prefetch;
    bool atomic = astrategy.atomic;
#ifdef STANDALONE
    prefetch |= astrategy.pfLoad;
#endif

    int R = rounddown_pow2(r);
    int C = rounddown_pow2(c);

    if (maxRBlock == 0) maxRBlock = r;
    if (maxCBlock == 0) maxCBlock = c;

    int rblock = 0, cblock = 0;

    if (isPacked(atype.layout)) {
        // Don't cross nonconsecutive tiles in a packed layout.
        bool cm = isColMajor(atype.layout) ^ isTransposing(astrategy.accessType);
        if (cm) {
            if (static_cast<uint32_t>(maxRBlock) < atype.packSize && atype.tileC > 0)
                maxCBlock = std::min<int>(maxCBlock, atype.tileC);
        } else {
            if (static_cast<uint32_t>(maxCBlock) < atype.packSize && atype.tileR > 0)
                maxRBlock = std::min<int>(maxRBlock, atype.tileR);
        }
    }

    // Set default parameters.
    colMajor = isColMajor(atype.layout);
    splitComplex = false;
    byteGlue = false;
    cxComponent = RegisterBlock::Interleaved;
    crosspack = 1;
    rowMask = MaskInfo::None();
    colMask = MaskInfo::None();
    rowFragment = 0;
    colFragment = 0;
    noRowsOK = false;
    noColsOK = false;
    descRemR = false;
    descRemC = false;
    descAssigned = false;
    addrShift = 0;
    clearFlag();
    msgRegs = 0;
    bytes = 0;
    hasNoLoad = false;
    offsetAddr = 0;

    auto &vrmask = rowMask.variable;
    auto &vcmask = colMask.variable;

    vrmask.rsize = 0;
    vcmask.rsize = 0;

    auto accessType = astrategy.accessType;

    switch (accessType) {
        case AccessType::ChannelScattered:
        case AccessType::Scattered:
        {
            if (T.is3() && !prefetch) {
                // u3 (3-bit) scattered access; prefetches fall through
                // to generic scattered. Only D32xV3/D64xV3 give a
                // multiple of 3 bytes, else stub; reuses Block2DTranspose.
                bool channelScattered = (accessType == AccessType::ChannelScattered);

                if (channelScattered || atomic || !astrategy.newDP || isColMajor(atype.layout)
                        || remainderR || remainderC)
                         stub("u3 scattered access requires newDP, non-atomic, non-channel-scattered, "
                         "row-major layout, and no remainder masking.");

                auto maxSIMD = maxScatteredSIMD(hw, astrategy);
                auto minSIMD = minScatteredSIMD(hw, astrategy);

                int simd = rounddown_pow2(std::min({r, maxRBlock, maxSIMD}));
                if (simd < minSIMD) simd = std::min(maxSIMD, minSIMD);
                if (simd <= 0 || r % simd)
                    stub("u3 scattered access: row count does not fit an exact SIMD width.");

                // Prefer the larger D64xV3 (64-element) group when it fits exactly;
                // fall back to the smaller D32xV3 (32-element) group otherwise.
                int ebytesChoice = 0, elemsPerLane = 0;
                for (auto cand : {std::make_pair(8, 64), std::make_pair(4, 32)}) {
                    int eb = cand.first, epl = cand.second;
                    if (epl > maxCBlock || epl > c) continue;
                    if (c % epl == 0) { ebytesChoice = eb; elemsPerLane = epl; break; }
                }
                if (ebytesChoice == 0)
                    stub("u3 scattered access: no groups-of-8 message (D32xV3/D64xV3) exactly fits the column block.");

                rblock = simd;
                cblock = elemsPerLane;
                colMajor = true;
                ebytes = ebytesChoice;
                crosspack = ebytesChoice;      // power-of-2 unit width (D32 = 4 / D64 = 8 bytes)
                count = 3;                     // LSC vector count V3 -- the multiple-of-3-bytes vcount
                simdSize = rblock;
                ld = roundup_pow2(rblock);
                extra = 1;                     // "consecutive" elements/address; u3 groups are addressed as 1 unit
                addrShift = 0;

                break;
            }

            bool channelScattered = (accessType == AccessType::ChannelScattered);

            // Detect large crosspack case.
            bool largeCP = isLargeCrosspack(T, atype.crosspack);
            int effCP = largeCP ? 1 : atype.crosspack;

            // Scattered read/write messages effectively transpose DW/QW matrices.
            colMajor = !colMajor ^ largeCP;

            // Let X be the contiguous dimension, Y the scattered dimension (in memory).
            int *xblock, *yblock;
            int maxXBlock, maxYBlock;
            int X, Y;
            bool remainderX, remainderY;
            int tileX, tileY;
            int scxExpand = 1;
            auto &vxmask = colMajor ? vcmask : vrmask;
            auto &vymask = colMajor ? vrmask : vcmask;
            auto &fragment = colMajor ? colFragment : rowFragment;
            auto smode = astrategy.smode;

            if (colMajor) {
                Y = allowFixedMasks ? r : R; X = C;
                yblock = &rblock;
                xblock = &cblock;
                maxYBlock = maxRBlock;
                maxXBlock = maxCBlock;
                remainderY = remainderR;
                remainderX = remainderC;
                tileY = atype.tileR;
                tileX = atype.tileC;
            } else {
                X = R; Y = allowFixedMasks ? c : C;
                xblock = &rblock;
                yblock = &cblock;
                maxXBlock = maxRBlock;
                maxYBlock = maxCBlock;
                remainderX = remainderR;
                remainderY = remainderC;
                tileX = atype.tileR;
                tileY = atype.tileC;
            }

            if (tileX > 0) maxXBlock = std::min(maxXBlock, tileX);
            if (tileY > 0) maxYBlock = std::min(maxYBlock, tileY);

            // Allowed accesses:
            //   A64             Essentially max 256 bytes.
            //                    8 slots x (1,2,4,8) dwords [Gen12/surface: 1,2,4]
            //                    8 slots x (1,2,4) qwords
            //                   16 slots x (1,2,4) dwords
            //                   16 slots x (1,2) qwords
            //   Others           8 slots x 1 dword
            //                   16 slots x 1 dword
            // Slot counts doubled for 64-byte GRFs.

            // Native (col major in memory) matrix block sizes, as a result:
            //   SIMD8:          1x8  2x4 4x2 8x1      (count 1)  2x8  4x8  8x8  [others]
            //   SIMD16:         1x16 2x8 4x4 8x2 16x1 (count 1)  2x16 4x16
            // Other layouts are possible too but require dummy (non-load) blocks.
            // Only kx8 and kx16 are supported for now for {4,8}-byte types.
            // For 16-byte types, only 1x4 and 1x8 are supported.

            auto maxSIMD = maxScatteredSIMD(hw, astrategy);
            auto minSIMD = minScatteredSIMD(hw, astrategy);

            auto Xc = ((avoidFragment && !allowDesc && remainderX) || atomic) ? 1 : X;
            bool byte = (atype.alignment < 4) || (Xc * T * effCP < 4);
            bool a64 = (astrategy.base.getModel() == ModelA64);

            channelScattered |= byte;

            bool qword = (T.paddedSize() >= 8 && !channelScattered && !prefetch && (a64 || astrategy.newDP));
            if (atomic && hasNativeAtomicAdd(hw, T.real(), atype, astrategy))
                qword &= (T.real().paddedSize() >= 8);
            int width = qword ? 8 : 4;
            ebytes = byte ? 1 : width;
            crosspack = std::max<int>(1, width / T);
            int consecutive = std::max<int>(1, T.paddedSize() / width);

            if (prefetch) consecutive = 1;

            if (ebytes == 4 && astrategy.base.getModel() == ModelSLM && !astrategy.newDP)
                channelScattered = true;

            bool simd1 = !a64 && !channelScattered && !astrategy.newDP;

            // Handle source crosspack.
            int uncrosspack = 1;
            if (effCP > 1) {
                if (effCP == crosspack) {
                    crosspack = 1;
                    uncrosspack = effCP;
                } else
                    stub();
            }

            // Try to fit a native matrix block size to X and Y.
            auto logicalSlots = std::min(Y, maxYBlock) * consecutive / uncrosspack;
            auto slots = roundup_pow2(logicalSlots);
            if (prefetch) {
                // Prefetch only: maximize Y usage.
                simdSize = maxSIMD;
            } else if (smode == ScatterSIMD::Narrow || (smode == ScatterSIMD::Default && ebytes * minSIMD > GRF::bytes(hw))) {
                // Maximize X usage because we always have at least 2 consecutive GRFs.
                simdSize = (slots >= maxSIMD && X <= 2) ? maxSIMD : minSIMD;
            } else {
                // Otherwise, try to maximize Y usage (larger SIMD, worse memory access).
                simdSize = maxSIMD;
            }
            simdSize = slots = std::min<int>(simdSize, slots);
            logicalSlots = std::min<int>(simdSize, logicalSlots);

            bool no8x8DW = (hw >= HW::Gen12LP) && !astrategy.newDP;
            bool no16x4QW = (GRF::bytes(hw) == 64) && !astrategy.newDP;

            int hwMaxXBlock;

            if (prefetch)
                hwMaxXBlock = 64 / T;
            else if (consecutive > 1)
                hwMaxXBlock = 1;
            else if (byte)
                hwMaxXBlock = (remainderX || atomic) ? 1 : crosspack;
            else if (simd1)
                hwMaxXBlock = crosspack;
            else if (remainderX && avoidFragment && !allowDesc)
                hwMaxXBlock = crosspack * scxExpand;
            else if (atomic)
                hwMaxXBlock = crosspack * scxExpand;
            else if (channelScattered || (ebytes == 4 && no8x8DW) || (ebytes == 8 && no16x4QW) || (simdSize == maxSIMD))
                hwMaxXBlock = 16 / T / uncrosspack;
            else
                hwMaxXBlock = 32 / T / uncrosspack;

            maxXBlock = std::min(maxXBlock, hwMaxXBlock);

            *xblock = std::min<int>(X, maxXBlock);
            count = *xblock;

            *yblock = logicalSlots * uncrosspack / consecutive;

            if (prefetch)
                count = 1;
            else if (byte)
                count *= T;
            else
                count = std::max<int>(1, count / crosspack);

            // LD is determined by actual # of SIMD slots in HW. But for X = 1 we may
            //  shrink the LD to avoid allocating unnecessary registers.
            auto ldSIMD = simdSize;
            if (*xblock > 1 || (minSIMD * ebytes <= GRF::bytes(hw)))
                ldSIMD = std::max<int>(ldSIMD, minSIMD);
            ld = ldSIMD * uncrosspack / consecutive;

            // Handle remainder. Masking handles Y remainders.
            if (remainderY) {
                vymask.isFixed = false;
                vymask.bitRep = consecutive;
                vymask.maskRep = 1;
                vymask.rsize = *yblock;
                vymask.rshift = 0;
            } else if (logicalSlots < slots) {
                auto &fymask = colMajor ? rowMask.fixed : colMask.fixed;
                fymask.isFixed = true;
                fymask.rsize = *yblock;
                fymask.value = (uint32_t(1) << logicalSlots) - 1;
            }

            // X remainders require fragmenting. Channel scattered float doesn't need complete fragmenting.
            //   (ditto for regular scattered float with new dataport messages.)
            //  Otherwise, fragment 2 is possible for DWord+ types but not implemented.
            if (remainderX) {
                if (avoidFragment && (*xblock == 1 || count == 1)) {
                    vxmask.isFixed = false;
                    vxmask.bitRep = (simdSize > 16) ? 32 : 16;
                    vxmask.maskRep = 1;
                    vxmask.rsize = 1;
                    vxmask.rshift = 0;
                } else if (allowDesc && (channelScattered || astrategy.newDP) && *xblock > 1 && !byte) {
                    fragment = std::min(*xblock, 4 * width / T);
                    if (colMajor)             // Clang can't handle the ternary operator equivalent of this.
                        descRemC = true;
                    else
                        descRemR = true;
                } else
                    fragment = 1;
            }

            extra = consecutive;

            // BTS scattered accesses are addressed by elements.
            if (!astrategy.newDP && !channelScattered && !astrategy.base.isStateless())
                addrShift = ilog2(ebytes);

            break;
        }
        case AccessType::Block:
        case AccessType::PseudoBlock:
        {
            if (T.is3()) stub("u3 block and pseudo-block access is unsupported.");

            // Three types of block messages:
            //    block_oword: 16 byte align, BLK masking (= dw)
            //  aligned_oword:  4 byte align, no masking, read only
            //    block_hword: [12LP] A64; 4 byte align R, BLKCM masking (= dw)
            //                        A64; 16 byte align W
            //                 [XeHP]   A64/BTS; 32 byte align R/W
            // New dataport messages support {DW, QW}x{1...64} with DW/QW alignment, no masking.
            //
            // Prefer block_hword in all cases. When block_hword can't be used:
            //   Use oword if alignment can be assured (i.e. packed row/column layout, or oword-sized scalar)
            //   Otherwise, use aligned oword. load/storeMatrixBlock will emit an error if masking/stores attempted.
            //
            // Pseudoblock messages have similar layouts, but are limited to
            //  {8,16}x{dw,qw} sizes, so lengths 8,16 allowed for float, 4,8,16 for double.

            bool effCM = colMajor ^ isLargeCrosspack(T, atype.crosspack);
            auto consecutive = consecutiveElements(r, c, atype);
            bool masking = (effCM ? remainderR : remainderC);
            bool bytePartialCP = (T.paddedSize() & 3) && ((colMajor ? C : R) % atype.crosspack);
            bool byte = (atype.alignment & 3) || (consecutive * T & 3) || bytePartialCP || ((T.paddedSize() & 3) && writable && masking);
            bool byte1PerSlot = byte && (bytePartialCP || masking || atomic);
            bool pseudo = (accessType == AccessType::PseudoBlock)
                        | needsPseudoblock(hw, T, R, C, atype, astrategy, writable, masking);
            int maxElements = 0;
            int maskGranularity = 1;
            int maxSIMD = maxScatteredSIMD(hw, astrategy);
            bool oword = false, aoword = false;
            int npack = 0;
            bool canQW = false, mustQW = false;

            bool a32 = (astrategy.base.getModel() == ModelA32);
            bool a64 = (astrategy.base.getModel() == ModelA64);
            bool sc = (astrategy.base.getModel() == ModelSC);
            bool slm = (astrategy.base.getModel() == ModelSLM);

            if (!pseudo && byte) return;

            if (astrategy.newDP && !pseudo) {
                bool qword = ((atype.alignment | (consecutive * T)) % 8 == 0);
                ebytes = qword ? 8 : 4;
                maxElements = (64 * ebytes) / T;
                maskGranularity = T.paddedSize();         // Convenience value; LSC cannot mask individual elements
            } else if (!pseudo) {
                int maxCount = 8;
                oword = !a64;
                aoword = ((atype.alignment & 0xF) != 0) || sc;
                if (hw > HW::Gen12LP) {
                    oword |= (atype.alignment & 0x1F) != 0;
                    if (slm) maxCount = 16;
                }
                ebytes = oword ? 16 : 32;
                maxElements = maxCount * ebytes / T;
                maskGranularity = 4;                // Block accesses mask by dwords
            } else {
                bool nativeAtomic = atomic && hasNativeAtomicAdd(hw, T.real(), atype, astrategy);
                canQW = ((atype.alignment | (consecutive * T)) % 8 == 0);
                if (astrategy.newDP)
                    canQW |= byte;
                else
                    canQW &= !byte && a64;
                if (slm && atomic)        // QW SLM atomics are implemented in XeHPC, but seeing functionality issues.
                    canQW = false;
                if (remainderR || remainderC)
                    canQW &= (T.paddedSize() % 8 == 0);
                if (nativeAtomic)
                    canQW = mustQW = (T.real().paddedSize() >= 8);
                auto stride = canQW ? 8 : 4;
                auto maxNPack = byte1PerSlot ? 1 : std::max<int>(1, stride / T.paddedSize());
                int simdCap = maxSIMD;
                if (atomic && !nativeAtomic)
                    simdCap = 16;
                if (byte1PerSlot && a64 && !writable)
                    simdCap = std::min(simdCap, 16);
                maxElements = simdCap * maxNPack;
                if (T.paddedSize() > stride)
                    maxElements = maxElements * stride / T;
                if (allowFixedMasks)
                    R = r, C = c;
            }

            auto maxABlock = maxElements / (byte1PerSlot ? 1 : atype.crosspack);

            auto choosePackedRCBlock = [=](int &xblock, int &yblock, int tileX, int tileY, int X, int Y) {
                xblock = std::min<int>(maxABlock, X);

                if (tileX) {
                    int ntileY = tileY ? (maxElements / (xblock * tileY)) : 0;
                    if (static_cast<uint32_t>(xblock) < atype.packSize || Y < tileY || ntileY == 0)
                        xblock = std::min<int>(xblock, tileX);
                }
                if ((tileX ? tileX : atype.packSize) <= static_cast<uint32_t>(xblock)) {
                    yblock = std::min<int>(maxElements / xblock, Y);
                    if (yblock < atype.crosspack && isLargeCrosspack(T, atype.crosspack)) {
                        yblock = atype.crosspack;
                        xblock = std::min<int>(xblock, maxElements / yblock);
                    }
                    if (tileY > 0 && yblock > tileY)
                        yblock = align_down(yblock, tileY);
                } else
                    yblock = atype.crosspack;     // Remainder loop: no longer packed in memory
            };

            switch (atype.layout) {
                case MatrixLayout::Pc:
                    choosePackedRCBlock(rblock, cblock, atype.tileR, atype.tileC, R, C);
                    crosspack = atype.crosspack;
                    break;
                case MatrixLayout::N:
                    if (atype.crosspack > 1) stub();
                    if (atype.tileR == R && R <= maxElements) {
                        cblock = std::min<int>(maxElements / R, C);
                        rblock = R;
                    } else {
                        cblock = 1;
                        rblock = std::min<int>(maxElements, R);
                    }
                    break;
                case MatrixLayout::Pr:
                    choosePackedRCBlock(cblock, rblock, atype.tileC, atype.tileR, C, R);
                    crosspack = atype.crosspack;
                    break;
                case MatrixLayout::T:
                    if (atype.crosspack > 1) stub();
                    if (atype.tileC == C && C <= maxElements) {
                        rblock = std::min<int>(maxElements / C, R);
                        cblock = C;
                    } else {
                        rblock = 1;
                        cblock = std::min<int>(maxElements, C);
                    }
                    break;
            }

            rblock = std::min(rblock, maxRBlock);
            cblock = std::min(cblock, maxCBlock);

            if (pseudo) {
                bool qword = mustQW || (canQW && (rblock * cblock * T >= 4 * maxSIMD));
                npack = std::max<int>(1, (qword ? 8 : 4) / T);
                if (byte1PerSlot) {
                    if (isLargeCrosspack(T, crosspack)) {
                        if (crosspack == (colMajor ? cblock : rblock))
                            colMajor = effCM;
                        else
                            stub();
                    }
                    crosspack = npack / T.perByte();
                    byteGlue = (T.bits() < 8);
                    npack = T.perByte();
                    (effCM ? cblock : rblock) = 1;
                }
                maskGranularity = qword ? 8 :
                           byte1PerSlot ? T.paddedSize() :
                                          4;
            }

            if (remainderR) {
                if (effCM) {
                    // rblock cannot be more than 16 dwords = 64 bytes for masking
                    //  except for pseudo-block
                    int rblockLimit = pseudo ? rblock : 64 / T;

                    if (avoidFragment) rblock = std::min<int>(rblock, rblockLimit);
                    if (rblock > rblockLimit)
                        rowFragment = rblockLimit;
                    else {
                        // For sizeof(T) < maskGranularity, this is a bit of a cheat.
                        //
                        // As long as we do not need to write to this matrix, we can read
                        // in maskGranularity-sized chunks knowing we will never cross a page boundary.

                        if (writable && (T.paddedSize() & (maskGranularity - 1)))
                            return;
                        if (!pseudo && oword && aoword)
                            hw_unsupported();

                        if (!pseudo && !(isPacked(atype.layout) && (atype.packSize == static_cast<uint32_t>(rblock)))) cblock = 1;

                        vrmask.isFixed = false;
                        vrmask.rsize = rblock;
                        vrmask.bitRep = std::max<int>(T.paddedSize() / maskGranularity, 1);
                        vrmask.maskRep = cblock;
                        vrmask.rshift = ilog2(std::max<int>(maskGranularity / T, 1));
                    }
                } else {
                    if (avoidFragment) {
                        // No native masking in this dimension. One mask/row.
                        rblock = 1;
                        vrmask.isFixed = false;
                        vrmask.bitRep = 0;  /* will be filled in later */
                        vrmask.maskRep = 1;
                        vrmask.rsize = 1;
                        vrmask.rshift = 0;
                    } else {
                        // Fragment it. Could actually handle rowFragment = 2 by changing descriptor.
                        rowFragment = 1;
                    }
                }
            }

            if (remainderC) {
                if (!effCM) {
                    // cblock cannot be more than 16 dwords = 64 bytes except for pseudo-block
                    int cblockLimit = pseudo ? cblock : 64 / T;

                    if (avoidFragment) cblock = std::min<int>(cblock, cblockLimit);
                    if (cblock > cblockLimit)
                        colFragment = cblockLimit;
                    else {
                        if (writable && (T.paddedSize() & (maskGranularity - 1)))
                            return;
                        if (!pseudo && oword && aoword)
                            hw_unsupported();

                        if (!pseudo && !(isPacked(atype.layout) && (atype.packSize == static_cast<uint32_t>(cblock)))) rblock = 1;

                        vcmask.isFixed = false;
                        vcmask.rsize = cblock;
                        vcmask.bitRep = std::max<int>(T.paddedSize() / maskGranularity, 1);
                        vcmask.maskRep = rblock;
                        vcmask.rshift = ilog2(std::max<int>(maskGranularity / T, 1));
                    }
                } else {
                    if (avoidFragment) {
                        // No native masking in this dimension. One mask/column.
                        cblock = 1;
                        vcmask.isFixed = false;
                        vcmask.bitRep = 0;
                        vcmask.maskRep = 1;
                        vcmask.rsize = 1;
                        vcmask.rshift = 0;
                    } else {
                        // Fragment it. Could actually handle colFragment = 2 by changing descriptor.
                        colFragment = 1;
                    }
                }
            }

            bool needFRMask = pseudo && (!remainderR && !is_zero_or_pow2(rblock));
            bool needFCMask = pseudo && (!remainderC && !is_zero_or_pow2(cblock));

            if (needFRMask || needFCMask) {
                // Create fixed mask for this
                auto &fmask = needFRMask ? rowMask.fixed : colMask.fixed;
                int logicalSlots = (rblock * cblock * T) / maskGranularity;
                fmask.isFixed = true;
                fmask.rsize = rblock * cblock;
                fmask.value = (uint32_t(1) << logicalSlots) - 1;
            }

            int nbytes = roundup_pow2(rblock * cblock) * T;
            simdSize = clamp(roundup_pow2(nbytes) / maskGranularity, 1, maxSIMD);
            ld = colMajor ? rblock : cblock;
            if (!pseudo) {
                if (astrategy.newDP) simdSize = 1;
                count = div_up(nbytes, ebytes);
                extra = aoword;
                if (ebytes == 16 && !(a32 || a64) && !aoword)         // BTS/SLM oword loads are oword-addressed.
                    addrShift = 4;
            } else {
                count = byte ? std::min(nbytes, npack * T) : 1;
                ebytes = byte ? 1 : maskGranularity;
                extra = 1;
                if (!(a32 || a64 || pseudoblockUseChannelScattered(atype, astrategy) || atomic))
                    addrShift = ilog2(ebytes);
            }
            if (astrategy.newDP) addrShift = 0;

            int maskAllBitRep = (pseudo && simdSize > 16) ? 32 : 16;
            if (!vrmask.isFixed && vrmask.bitRep == 0) vrmask.bitRep = maskAllBitRep;
            if (!vcmask.isFixed && vcmask.bitRep == 0) vcmask.bitRep = maskAllBitRep;
            break;
        }
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI: {
            // bytes * array length <= 8
            // width * array length <= 64 bytes (or == 256 bytes on Xe3p)
            //  => width <= 1 GRF
            // height <= 32 (load) 8 (store)
            // array length = 1 for store, transpose
            //
            // normal: width >= 4 bytes
            // transpose: d32/d64 only
            // vnni: d8/d16 only, height >= 4 bytes
            bool transpose = (accessType == AccessType::Block2DTranspose);
            bool vnni      = (accessType == AccessType::Block2DVNNI);

            bool memCM = colMajor;
            colMajor ^= transpose;
            auto X = memCM ? r : c;
            auto Y = memCM ? c : r;
            auto &xblock = memCM ? rblock : cblock;
            auto &yblock = memCM ? cblock : rblock;
            auto maxXBlock = memCM ? maxRBlock : maxCBlock;
            auto maxYBlock = memCM ? maxCBlock : maxRBlock;

            if (hw < HW::XeHPC || !astrategy.newDP) hw_unsupported();

            // Choose underlying type.
            auto Tblock = T;
            if (transpose) {
                int maxW;
                if (Tblock.paddedSize() > 8) hw_unsupported();
                if (Tblock.paddedSize() > 4) {
                    Tblock = Type::u64;
                    maxW = 4;
                    maxYBlock = 8;
                } else {
                    Tblock = Type::u32;
                    maxW = (hw == HW::Xe3p) ? 16 : 8;
                    // 1-8, 16 supported on Xe3p
                    if (maxXBlock < (maxW * Tblock) / T) maxW = 8;
                }
                maxXBlock = std::min(maxXBlock, (maxW * Tblock) / T);
            } else if (vnni) {
                if (Tblock.paddedSize() >= 4) hw_unsupported();
                if ((Y * Tblock) % 4) hw_unsupported();
                maxXBlock = std::min(maxXBlock, 16);
            } else {
                // find()/blockRegion()'s u3 group addressing needs the
                // row-spread packing from a transposing 2D block message;
                // plain Block2D doesn't produce that layout.
                if (T.is3() && !prefetch)
                    stub("u3 is only supported with Block2DTranspose access.");
                if (Tblock.paddedSize() > 8) Tblock = Type::u64;
                crosspack = atype.crosspack;
            }
            if (T.is3()) {
                // u3 is not addressable at native GRF granularity (see
                // RegisterBlock::find()), so the byte-alignment check below
                // (which assumes X*T lands on a whole byte) doesn't apply.
            } else if ((X * T) % 4) hw_unsupported();

            int minAlign = block2DMinAlignment(hw, atype, astrategy);
            if (atype.alignment % minAlign) hw_unsupported();

            // Reinterpret X/maxXBlock to underlying type.
            auto X_logical = X;
            if (T.is3() && !prefetch) {
                // u3 uses a 4-byte group pitch (crosspack = 4 below).
                // Round up instead of flooring, so a small tail rounds
                // to one group (over-reads into padding) not zero.
                const int groupBytes = 4;
                auto toGroupsCeil = [&](int elems) {
                    int bytes = static_cast<int>(elems * T);
                    return div_up(bytes, groupBytes);
                };
                maxXBlock = std::max(1, toGroupsCeil(maxXBlock));
                X = toGroupsCeil(X);
            } else {
                maxXBlock = (maxXBlock * T) / Tblock;
                X = (X * T) / Tblock;
            }

            // Carve out a maximal allowed block size - up to 64 bytes, or exactly 256 byte prefetches
            int maxBlockBytes = SendSpec::MaxBlock2DWidthBytes(hw, prefetch);
            if (X * Tblock < maxBlockBytes) {
                // Max-size prefetches only, or fall back to regular block size limitations
                maxBlockBytes = SendSpec::MaxBlock2DWidthBytes(hw, false);
            }
            xblock = std::min(X, maxBlockBytes / Tblock);
            xblock = std::max(xblock, 4 / Tblock);
            // Block2D transpose messages only support power-of-2 widths
            // (u3's 3-dword tail can't be covered by widening). Round
            // down; appendBlocks() recurses on the leftover remainder.
            if (transpose && (!T.is3() || !prefetch)) xblock = rounddown_pow2(xblock);
            int yblockLimit = writable ? 8 : 32;

            if (isPacked(atype.layout) && 2 * xblock <= X && static_cast<uint32_t>(X_logical) == atype.packSize) {
                // Split logical x dimension into multiple spans to accomodate width restriction.
                if (astrategy.address2D) stub();
                int multiX = X / xblock;
                xblock *= multiX;
                yblockLimit /= multiX;
            }

            yblock = std::min({maxYBlock, Y, yblockLimit});

            if (transpose && Tblock.paddedSize() == 8 && yblock != 8) hw_unsupported();

            // Choose # of blocks. In postprocess, this RegisterBlock will be
            //  split into one RegisterBlock for each block in the array.
            int icount = 1;
            if (!(writable || transpose)) {
                icount = rounddown_pow2(xblock / maxXBlock);
                icount = std::min({icount, 8 / Tblock, 4});
                icount = std::max(icount, 1);
            }
            xblock = std::min(xblock, maxXBlock * icount);

            count = icount;

            // Crosspack calculation.
            int icrosspack = (transpose || vnni) ? std::max(1, 4 / T) : 1;
            if (atype.crosspack == 1)
                crosspack = icrosspack;
            else if (atype.crosspack == icrosspack)
                crosspack = 1;
            else return;
            // u3's packed-group addressing is incompatible with crosspack
            // rearrangement; repurpose crosspack to record the 4-byte
            // column pitch instead, consumed by find()/blockRegion().
            if (T.is3() && transpose) crosspack = 4;

            // Convert size from underlying type to our actual type.
            if (T.is3() && !prefetch) {
                int bytes = xblock * Tblock;
                xblock = div_up(bytes * 8, T.bits());

                // Transpose sends split on dword boundaries; the element
                // straddling a boundary is unpacked from the merged layout.
                if (transpose)
                    xblock = (bytes * 8) / T.bits();
                xblock = std::min(xblock, X_logical);
            } else {
                xblock = (xblock * Tblock) / T;
            }

            simdSize = 1;
            ld = roundup_pow2(transpose ? yblock : xblock);
            // u3 has no native per-element hardware register type, so
            // ebytes reflects the column pitch of its 8-element/3-byte
            // packing group (crosspack) rather than Tblock's own size.
            ebytes = T.is3() ? crosspack : Tblock.paddedSize();
            extra = T.bits();
            auto bytes = align_up((colMajor ? cblock : rblock) / count, crosspack) * ld * count * T;
            // u3's row pitch isn't necessarily a power of 2; round up so
            // the block message's register footprint is sized correctly.
            if (T.is3())
                bytes = ((colMajor ? cblock : rblock) / count) * roundup_pow2(ld * T) * count;
            msgRegs = GRF::bytesToGRFs(hw, bytes);
            if (T.is3() && vnni)
                stub("u3 does not support Block2DVNNI access; use Block2D or Block2DTranspose instead.");
            if (vnni && (T.bits() < 8)) {
                byteGlue = true;
                crosspack /= T.perByte();
            }

            // Xe2: manually mask in the height dimension to work around slow LSC
            //      out-of-bounds checks.
            bool remainderH = memCM ? remainderC : remainderR;
            if (hw >= HW::Xe2 && remainderH) {
                auto &vymask = memCM ? colMask.variable : rowMask.variable;
                vymask.isFixed = false;
                vymask.bitRep = vymask.maskRep = vymask.rsize = 1;
                vymask.rshift = 0;
            }
            break;
        }
        case AccessType::CacheLine: {
            if (T.is3())
                stub("u3 is only supported with Block2DTranspose access.");

            // Let X be the contiguous dimension in memory, Y the scattered dimension.
            int x = colMajor ? r : c;
            int y = colMajor ? c : r;
            auto &xblock = colMajor ? rblock : cblock;
            auto &yblock = colMajor ? cblock : rblock;
            auto maxXBlock = colMajor ? maxRBlock : maxCBlock;
            auto maxYBlock = colMajor ? maxCBlock : maxRBlock;
            bool remainderX = colMajor ? remainderR : remainderC;
            bool remainderY = colMajor ? remainderC : remainderR;

            auto maxSIMD = maxScatteredSIMD(hw, astrategy);
            int trailing = (atype.alignment % 64) ? 1 : 0;      // Do we need a trailing pointer?
            int elemsPerCL = 64 / T;

            xblock = std::min({maxXBlock, x, (maxSIMD - trailing) * elemsPerCL});
            int xCacheLines = roundup_pow2(div_up(x * T, 64) + trailing);

            yblock = rounddown_pow2(std::min({maxYBlock, y, maxSIMD / xCacheLines}));

            simdSize = xCacheLines * yblock;
            ld = xCacheLines;
            if (atype.alignment >= 4 && x * T >= 4) {
                ebytes = 4;
                count = 1;
            } else {
                ebytes = 1;
                count = uint8_t(std::min<int>({atype.alignment, 4, x * T}));
            }
            extra = 1;

            if (remainderX) {
                // All on/off mask for x remainders. Finer grained remainders
                //  are handled by adjusting addresses to be in bounds.
                auto &vxmask = colMajor ? rowMask.variable : colMask.variable;
                vxmask.isFixed = false;
                vxmask.bitRep = simdSize;
                vxmask.maskRep = vxmask.rsize = 1;
                vxmask.rshift = 0;
            }

            if (remainderY) {
                auto &vymask = colMajor ? colMask.variable : rowMask.variable;
                vymask.isFixed = false;
                vymask.bitRep = xCacheLines;
                vymask.maskRep = 1;
                vymask.rsize = yblock;
                vymask.rshift = 0;
            }
            break;
        }
    }

    // The mask moduli are almost always rblock/c
    // Also, clamp mask reps to ensure mask length does not exceed SIMD size.
    if (rowMask && !rowMask.fixed.isFixed) {
        if (vrmask.rsize == 0)
            vrmask.rsize = rblock;
        vrmask.maskRep = std::min<int>(vrmask.maskRep, std::max<int>(1, (simdSize << vrmask.rshift) / (vrmask.bitRep * vrmask.rsize)));
        noRowsOK = true;          // All-zero masks are always OK.
    }
    if (colMask && !colMask.fixed.isFixed) {
        if (vcmask.rsize == 0)
            vcmask.rsize = cblock;
        vcmask.maskRep = std::min<int>(vcmask.maskRep, std::max<int>(1, (simdSize << vcmask.rshift) / (vcmask.bitRep * vcmask.rsize)));
        noColsOK = true;
    }

    nr = rblock;
    nc = cblock;
}

RegisterBlock RegisterBlock::slice(Type T, bool column, int x1, int x2, int x1Unclamped, int x2Unclamped, bool overrunOK,
                                   const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const
{
    auto result = trySlice(T, column, x1, x2, x1Unclamped, x2Unclamped, overrunOK, atype, astrategy);
    if (!result) stub("Could not slice register block.");
    return result;
}

RegisterBlock RegisterBlock::trySlice(Type T, bool column, int x1, int x2, int x1Unclamped, int x2Unclamped, bool overrunOK,
                                      const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const
{
    auto Telem = T;
    auto effAccessType = effectiveAccessType(atype, astrategy);
    auto blockDst = *this;

    auto &ns = (column ? blockDst.nc : blockDst.nr);
    auto &nt = (column ? blockDst.nr : blockDst.nc);
    int oldNS = ns;

    (column ? blockDst.offsetC : blockDst.offsetR) += x1;
    ns = x2 - x1;

    if ((ns == oldNS) && (overrunOK || !hasNoLoad))
        return blockDst;

    if (colMajor == column) {
        if (T.is3() && !isLoadBlock() && colMajor && !byteGlue) {
            if (x1 % 8) return RegisterBlock();
            int phase = offsetBytes % crosspack;
            blockDst.offsetBytes += int3RowSpreadOffset(phase + x1 / 8 * 3, ld, crosspack) - phase;
        } else {
            if (x1 % crosspack) return RegisterBlock();
            blockDst.offsetBytes += (x1 * bytes) / oldNS;
        }

        if (isLoadBlock()) switch (effAccessType) {
            case AccessType::Scattered:
            case AccessType::ChannelScattered:
                blockDst.count = x2 - x1;
                if (blockDst.ebytes == 1)
                    blockDst.count *= T;
                else if (blockDst.splitComplex)
                    blockDst.count *= 2;
                else if (T.paddedSize() < blockDst.ebytes) {
                    // Extra alignment path with small types.
                    // Check to see if we can still use this element size,
                    //  if not downgrade to scattered byte.
                    // Note for surface accesses this requires shifting the addresses back.
                    auto bcount = blockDst.count * T;
                    if (bcount % 4) {
                        blockDst.ebytes = 1;
                        blockDst.addrShift = 0;
                        blockDst.count = bcount;
                        if (blockDst.count > 4) stub();
                    } else
                        blockDst.count = bcount >> 2;
                }
                break;
            case AccessType::Block:
            case AccessType::PseudoBlock: {
                auto offBytes = x1 * nt * T;
                if (offBytes % blockDst.ebytes)
                    return RegisterBlock();
                auto reqBytes = (x2 - x1) * nt * T;
                auto align = std::min<int>(blockDst.ebytes, blockDst.simdSize * 4);
                if (!overrunOK && (reqBytes & (align - 1)))
                    return RegisterBlock();
                auto ncount = div_up(reqBytes, blockDst.ebytes);
                auto count = roundup_pow2(ncount);
                if (!overrunOK && (count != ncount))
                    return RegisterBlock();
                if (effAccessType == AccessType::Block)
                    blockDst.count = count;
                else
                    blockDst.simdSize = std::max(1, count / blockDst.count);
                break;
            }
            case AccessType::Block2D:
                break;
            case AccessType::Block2DTranspose:
            case AccessType::Block2DVNNI: {
                int crosspack = std::max(1, 4 / blockDst.ebytes);
                if (x1 % crosspack || x2 % crosspack)
                    return RegisterBlock();
                break;
            }
            case AccessType::CacheLine:
                blockDst.simdSize = blockDst.simdSize * (x2 - x1) / ns;
                break;
        }
    } else {
        blockDst.offsetBytes += (T.is3() && !isLoadBlock() && colMajor && !byteGlue)
                ? x1 * crosspack : x1 * Telem * crosspack;

        if (isLoadBlock()) switch (effAccessType) {
            case AccessType::Block:
            case AccessType::PseudoBlock: {
                // Update count and mask information.
                // Beware, cheat: with DW-aligned sub-DW types, true block may be downgraded to byte PseudoBlock,
                //                which requires 2 address registers, though only 1 is used, and only 1 may be allocated.
                auto opts = (blockDst.rowFragment || blockDst.colFragment) ? AllowFragment
                                                                           : AvoidFragment;
                blockDst.recreate(hw, T, atype, astrategy, opts);
                blockDst.flag = flag;
                if (blockDst.flag[column] && x1 > 0) stub();
                blockDst.simplify(T);
                break;
            }
            case AccessType::Scattered:
            case AccessType::ChannelScattered: {
                if (x1 != 0)              return RegisterBlock();
                if (!is_zero_or_pow2(x2)) return RegisterBlock();

                // Keep the same number of lanes per row/column. Non-power-of-2 blocks have padding lanes.
                blockDst.simdSize = div_up(ns * simdSize, roundup_pow2(oldNS));

                auto minSIMD = minScatteredSIMD(hw, astrategy);
                if (blockDst.simdSize <= minSIMD && simdSize > minSIMD) {
                    if (blockDst.count > 1 && blockDst.ebytes > 1)
                        return RegisterBlock();
                    blockDst.ld >>= 1;
                }
                break;
            }
            case AccessType::Block2D:
            case AccessType::Block2DTranspose:
            case AccessType::Block2DVNNI:
                if (ns != oldNS) stub();        // Can do this, but not implemented.
                if (blockDst.simdSize != 0)     // Recompute block array length.
                    blockDst.count = div_up(x2Unclamped, isColMajor(atype.layout) ? blockDst.nr : blockDst.nc);
                // TODO: need to recompute ld
                break;
            case AccessType::CacheLine:
                if (ns != oldNS) stub();
                break;
        }
    }

    if (!blockDst.isLoadBlock()) {
        // Shrink LD
        auto nx = blockDst.colMajor ? blockDst.nr : blockDst.nc;
        auto ny = blockDst.colMajor ? blockDst.nc : blockDst.nr;
        if (ny == 1)
            blockDst.ld = std::min(blockDst.ld, nx);
    }

    blockDst.calcBytes(T, astrategy);

    return blockDst;
}

Subregister RegisterBlock::find(Type T, int ii, int jj, const GRFMultirange &regs,
                                int *nelems, int cxComponent_, int component_) const
{
    auto Te = T;

    if (ii < 0 || ii >= nr || jj < 0 || jj >= nc || component != component_ || !one_of(cxComponent, {-1, cxComponent_}))
        stub("Requested out-of-bounds element.");

    int xx = colMajor ? ii : jj;
    int yy = colMajor ? jj : ii;
    int nx = colMajor ? nr : nc;

    // u3 has no native register type; return the start of the 3-byte
    // group of 8 elements containing (ii, jj), as Type::ngen_u3() --
    // unpacked only by CopyPlan::planInt3Upconvert.
    if (T.is3()) {
        if (!colMajor || byteGlue)
            stub("Flat u3 register access is unsupported.");

        int elIndex = xx + yy * ld;
        if (elIndex & 7)
            stub("u3 element index must be aligned to an 8-element (3-byte) group boundary.");

        int group = yy / 8;
        // A sliced layout can start partway through a packed dword.
        int phase = offsetBytes % crosspack;
        int byteOff = int3RowSpreadOffset(phase + group * 3, ld, crosspack)
                - phase + xx * crosspack;
        byteOff += offsetBytes;

        int consecutive;
        auto result = regs.sub(hw, byteOff, Te.ngen(), &consecutive);
        // Groups run along the outer axis and are vectorized across the
        // inner row-spread lane axis in a single mov.
        if (nelems) *nelems = nx - xx;
        return result;
    }

    int ne = nx - xx;

    int yyx = yy % crosspack;
    yy -= yyx;

    if (byteGlue) {
        int xxx = xx & (T.perByte() - 1);
        yyx = yyx * T.perByte() + xxx;
        xx -= xxx;
        ne = 1;
    }

    int elFixed = yyx + (xx * crosspack);
    int elLD = yy;

    int el = elFixed + elLD * ld;
    el += offsetBytes / Te;

    int consecutive;
    auto result = regs.sub(hw, el, Te.ngen(), &consecutive);

    ne = std::min(ne, div_up(consecutive, crosspack));

    if (nelems) *nelems = ne;
    return result;
}

static RegisterRegion blockRegion(Type T, const Subregister &reg, const RegisterBlock &block,
                                  int rr, int cc, int *nelems, int cxComponent, bool allow2D)
{
    auto cp = block.crosspack;

    if (block.byteGlue && allow2D && T.bits() < 8) {
        if (nelems)
            *nelems = block.colMajor ? (block.nr - rr) : (block.nc - cc);
        return reg(cp / T, 1 / T, 1);
    } else
        return reg(cp);
}

RegisterRegion RegisterBlock::findRegion(Type T, int ii, int jj, const GRFMultirange &regs, int *nelems,
                                         int cxComponent, int component, bool allow2D) const
{
    auto reg = find(T, ii, jj, regs, nelems, cxComponent, component);
    return blockRegion(T, reg, *this, ii, jj, nelems, cxComponent, allow2D);
}

void RegisterBlock::calcBytes(Type T, const MatrixAddressingStrategy &astrategy)
{
    if (astrategy.newDP && astrategy.prefetch)
        bytes = 0;
    else {
        calcBytes(T);
        if (T.is3() && astrategy.accessType == AccessType::Block2DTranspose && isLoadBlock())
            bytes = msgRegs * GRF::bytes(hw);
    }
}

void RegisterBlock::calcBytes(Type T)
{
    if (cxComponent != Interleaved)
        T = T.real();
    if (T.is3() && !isLoadBlock() && colMajor && !byteGlue) {
        int phase = offsetBytes % crosspack;
        int lastByte = phase + div_up(nc * T.bits(), 8) - 1;
        bytes = int3RowSpreadOffset(lastByte, ld, crosspack) - phase
                + (nr - 1) * crosspack + 1;
    } else
        bytes = align_up(colMajor ? nc : nr, crosspack) * ld * T;
    if (isLoadBlock() && msgRegs == 0)
        msgRegs = GRF::bytesToGRFs(hw, bytes);
}

void RegisterBlock::simplify(Type T)
{
    if (crosspack == (colMajor ? nc : nr) && isLargeCrosspack(T, crosspack)) {
        auto od = colMajor ? nr : nc;
        if (ld == od) {
            colMajor = !colMajor;
            ld = crosspack;
            crosspack = 1;
        }
    }
}

void RegisterBlock::compact(Type T)
{
    auto newLD = std::max<int>(roundup_pow2(colMajor ? nr : nc), GRF::bytes(hw) / T);
    if (newLD < ld) {
        ld = newLD;
        calcBytes(T);
    }
}

int RegisterBlock::nregs() const
{
    if (!grfAligned()) stub();
    return GRF::bytesToGRFs(hw, bytes);
}

int RegisterBlock::offsetReg() const
{
    if (!grfAligned()) stub();
    return offsetBytes >> GRF::log2Bytes(hw);
}

AccessType RegisterBlock::effectiveAccessType(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const
{
    auto type = astrategy.accessType;
    if (!isLoadBlock())
        return type;
    if (type == AccessType::Block && ebytes < 16 && extra)
        type = AccessType::PseudoBlock;
    else if (type == AccessType::Scattered && astrategy.base.getModel() == ModelSLM && ebytes == 4 && !astrategy.newDP)
        type = AccessType::ChannelScattered;
    else if (type == AccessType::ChannelScattered && (ebytes != 4 || astrategy.atomic))
        type = AccessType::Scattered;
    return type;
}

AccessType RegisterBlock::implAccessType(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const
{
    auto type = effectiveAccessType(atype, astrategy);
    if (type == AccessType::PseudoBlock)
        type = pseudoblockUseChannelScattered(atype, astrategy) ? AccessType::ChannelScattered : AccessType::Scattered;
    else if (type == AccessType::CacheLine)
        type = AccessType::Scattered;
    return type;
}

bool RegisterBlock::pseudoblockUseChannelScattered(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const
{
    return (astrategy.base.getModel() == ModelSLM) && (ebytes == 4) && !astrategy.atomic;
}

void RegisterBlock::getBlock2DWH(int &w, int &h, int &count, const MatrixAddressing &atype, bool prefetch, int *outMultiX) const
{
    int multiX = 1;
    bool transpose = (isColMajor(atype.layout) != colMajor);
    w = isColMajor(atype.layout) ? nr : nc;
    h = isColMajor(atype.layout) ? nc : nr;
    if (transpose && extra == 3 && !prefetch) {
        int offset = isColMajor(atype.layout) ? offsetR : offsetC;
        w = div_up((offset + w) * extra, ebytes * 8)
                - div_up(offset * extra, ebytes * 8);
    } else
        w = (w * extra) / (ebytes * 8);
    if (isPacked(atype.layout)) {
        // prefetch: allow max-size or regular max width fallback
        int maxWBytes = SendSpec::MaxBlock2DWidthBytes(hw, prefetch);
        if (w * ebytes < maxWBytes) {
            maxWBytes = SendSpec::MaxBlock2DWidthBytes(hw, false);
        }
        int maxW = maxWBytes / ebytes;
        multiX = div_up(w, maxW);
        w /= multiX;
        h *= multiX;
    }
    count = this->count;
    if (transpose) {
        h *= count;
        count = 1;
    }
    if (outMultiX) *outMultiX = multiX;
}

int RegisterBlock::addrGRFs(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const
{
    // Non-load blocks don't get address registers.
    if (!isLoadBlock())
        return 0;

    // Offset blocks don't either -- they will share existing address registers.
    if (offsetAddr != 0)
        return 0;

    switch (effectiveAccessType(atype, astrategy)) {
        case AccessType::Scattered:
        case AccessType::ChannelScattered:
        case AccessType::PseudoBlock:
        case AccessType::CacheLine: {
            auto bytesPerAddr = (astrategy.base.getModel() == ModelA64) ? 8 : 4;
            auto baseSIMD = std::max<int>(simdSize, 8);
            return GRF::bytesToGRFs(hw, bytesPerAddr * baseSIMD);
        }
        case AccessType::Block:
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI:
            return 1;
    }
    stub("Invalid addressing.");
}

/**************************/
/* RegisterLayout methods */
/**************************/

// Create a register layout for a matrix.
RegisterLayout::RegisterLayout(HW hw_, Type T_, int r, int c, const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_,
                               bool remainderR_, bool remainderC_, bool writable_, RemainderOptions remOpts,
                               int maxRBlock, int maxCBlock, bool reverseOrder)
        : RegisterLayout(hw_, T_, r, c, atype_, astrategy_, remainderR_, remainderC_, writable_, remOpts, maxRBlock, maxCBlock, reverseOrder, false)
{
    if (!valid()) stub("Could not create register layout for matrix tile.");
}

RegisterLayout RegisterLayout::tryCreate(ngen::HW hw_, Type T_, int r, int c, const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_,
                                          bool remainderR_, bool remainderC_, bool writable_, RemainderOptions remOpts,
                                          int maxRBlock, int maxCBlock, bool reverseOrder)
{
    return RegisterLayout(hw_, T_, r, c, atype_, astrategy_, remainderR_, remainderC_, writable_, remOpts, maxRBlock, maxCBlock, reverseOrder, true);
}

RegisterLayout::RegisterLayout(HW hw_, Type T_, int r, int c, const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_,
                               bool remainderR_, bool remainderC_, bool writable_, RemainderOptions remOpts,
                               int maxRBlock, int maxCBlock, bool reverseOrder, bool)
        : hw(hw_), T(T_), nr(r), nc(c), atype(atype_), astrategy(astrategy_),
          remainderR(remainderR_), remainderC(remainderC_), writable(writable_)
{
    // If no associated address space, create an empty layout.
    if (astrategy.base.getModel() == ModelInvalid) {
        initialized = true;
        return;
    }

    // Tiling handling.
    auto forceTiling = [](int &maxBlock, int tile) {
        maxBlock = (maxBlock == 0) ? tile : gcd(tile, maxBlock);
    };

    if (astrategy.tileR > 0) forceTiling(maxRBlock, astrategy.tileR);
    if (astrategy.tileC > 0) forceTiling(maxCBlock, astrategy.tileC);

    if (atype.layout == MatrixLayout::Pc) forceTiling(maxRBlock, atype.packSize);
    if (atype.layout == MatrixLayout::Pr) forceTiling(maxCBlock, atype.packSize);

    // Two separate strategies for creating register layout:
    //    - standard 2D partitioning
    //    - special 1D partitioning for block access to packed inputs.
    if (((atype.layout == MatrixLayout::Pc && atype.packSize == static_cast<uint32_t>(r))
      || (atype.layout == MatrixLayout::Pr && atype.packSize == static_cast<uint32_t>(c)))
            && (astrategy.accessType == AccessType::Block)
            && !remainderR && !remainderC
            && !atype.tileR && !atype.tileC
            && (maxRBlock >= r || maxRBlock == 0)
            && (maxCBlock >= c || maxCBlock == 0)) {
        initialized = append1DBlocks(r, c);
    }
    if (!initialized) {
        initialized = appendBlocks(r, c, 0, 0, remOpts, maxRBlock, maxCBlock);
        sort(reverseOrder);
        postprocess();
    }
    if (!initialized)
        return;

    finalize();
    coalesceAddrs();
}

// Create a register layout for a uniform matrix not backed by memory.
RegisterLayout::RegisterLayout(HW hw_, Type T_, int r, int c, bool colMajor,
                               int crosspack, int tileR, int tileC,
                               bool allowPartialRegs, bool fullySplitCx)
        : hw(hw_), T(T_), nr(r), nc(c), remainderR(false), remainderC(false), writable(true), initialized(true)
{
    RegisterBlock block;

    auto y = (colMajor ? c : r);
    if (y > crosspack && y % crosspack) stub();

    if (tileR <= 0) tileR = r;
    if (tileC <= 0) tileC = c;

    int offsetBytes = 0;
    int qCXMin = -1, qCXMax = -1;

    if (tileR > 0 && tileC > 0)
        list.reserve(div_up(r, tileR) * div_up(c, tileC));

    for (int qCX = qCXMin; qCX <= qCXMax; qCX++) {
        for (int q = 0; q < T.components(); q++) {
            for (int i = 0; i < r; i += tileR) {
                for (int j = 0; j < c; j += tileC) {
                    block.hw = hw;
                    block.nr = std::min(r - i, tileR);
                    block.nc = std::min(c - j, tileC);
                    block.ld = colMajor ? tileR : tileC;
                    if (!allowPartialRegs)
                        block.ld = align_up(block.ld, elementsPerGRF(hw, T) / crosspack);
                    block.offsetR = i;
                    block.offsetC = j;
                    block.colMajor = colMajor;
                    block.crosspack = crosspack;
                    block.offsetBytes = offsetBytes;
                    block.splitComplex = false;
                    block.byteGlue = false;
                    block.cxComponent = qCX;
                    block.component = q;
                    block.remainderR = false;
                    block.remainderC = false;
                    block.simdSize = 0;         // Not backed by memory.

                    block.calcBytes(T);
                    offsetBytes += block.bytes;

                    list.push_back(block);
                }
            }
        }
    }

}

RegisterLayout::RegisterLayout(Type T_, const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_,
                               vector<RegisterBlock> &&list_)
        : T(T_), atype(atype_), astrategy(astrategy_), initialized(true), list(std::move(list_))
{
    for (const auto &block: *this) {
        remainderR |= block.remainderR;
        remainderC |= block.remainderC;
        writable &= block.writable;
        hw = block.hw;
    }

    updateDims();
}

RegisterLayout RegisterLayout::int3UnpackLayout() const
{
    if (!T.is3() || astrategy.accessType != AccessType::Block2DTranspose)
        return *this;

    vector<RegisterBlock> merged;
    merged.reserve(list.size());
    bool memCM = isColMajor(atype.layout);
    for (size_t i = 0; i < list.size(); i++) {
        auto block = list[i];
        auto width = memCM ? &RegisterBlock::nr : &RegisterBlock::nc;
        auto offset = memCM ? &RegisterBlock::offsetR : &RegisterBlock::offsetC;
        auto otherSize = memCM ? &RegisterBlock::nc : &RegisterBlock::nr;
        auto otherOffset = memCM ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

        while (!is_zero_or_pow2(int(block.*width))) {
            if (i + 1 >= list.size()) stub("Incomplete u3 transpose load.");
            const auto &next = list[i + 1];
            if (next.*offset != block.*offset + block.*width
                    || next.*otherOffset != block.*otherOffset
                    || next.*otherSize != block.*otherSize
                    || next.offsetBytes != block.offsetBytes + block.bytes
                    || next.component != block.component
                    || next.cxComponent != block.cxComponent
                    || next.colMajor != block.colMajor
                    || next.ld != block.ld || next.crosspack != block.crosspack)
                stub("Noncontiguous u3 transpose load.");
            block.*width += next.*width;
            block.bytes += next.bytes;
            block.simdSize = 0;
            block.msgRegs = 0;
            i++;
        }
        merged.push_back(block);
    }
    return RegisterLayout(T, atype, astrategy, std::move(merged));
}


bool RegisterLayout::appendBlocks(int r, int c, int roff, int coff, RemainderOptions remOpts, int maxRBlock, int maxCBlock)
{
    auto blockTemplate = RegisterBlock::tryCreate(hw, T, r, c, atype, astrategy,
                                                  remainderR, remainderC, writable, remOpts, maxRBlock, maxCBlock);
    int rblock = blockTemplate.nr, cblock = blockTemplate.nc;

    if (!blockTemplate.valid() || rblock == 0 || cblock == 0)
        return false;       /* Cannot handle requested block and remainder. */

    if (T.is3() && astrategy.accessType == AccessType::Block2DTranspose
            && !astrategy.prefetch) {
        bool memCM = isColMajor(atype.layout);
        int offset = memCM ? roff : coff;
        int &width = memCM ? rblock : cblock;
        // Transpose messages address u3 in whole dwords (32 elements = 3 dwords).
        //  Sub-blocks may split groups, but the full load must end on a dword.
        if (((offset + (memCM ? r : c)) * T.bits()) % 32)
            return false;
        int messageWidth = div_up((offset + width) * T.bits(), 32)
                - div_up(offset * T.bits(), 32);
        int lastElement = ((div_up(offset * T.bits(), 32) + messageWidth) * 32)
                / T.bits();
        width = std::min(memCM ? r : c, lastElement - offset);
        (memCM ? blockTemplate.nr : blockTemplate.nc) = width;
    }

    list.reserve(list.size() + T.components() * (r / rblock) * (c / cblock));

    auto addBlock = [&](int i, int j) {
        auto block = blockTemplate;
        block.offsetR = i + roff;
        block.offsetC = j + coff;
        list.push_back(block);
    };

    for (int q = 0; q < T.components(); q++) {
        blockTemplate.component = q;
        if (isColMajor(atype.layout)) {
            // Order blocks in column-major fashion.
            for (int j = 0; j + cblock <= c; j += cblock)
                for (int i = 0; i + rblock <= r; i += rblock)
                    addBlock(i, j);
        } else {
            // Order blocks in row-major fashion.
            for (int i = 0; i + rblock <= r; i += rblock)
                for (int j = 0; j + cblock <= c; j += cblock)
                    addBlock(i, j);
        }
    }

    // Handle remainder recursively, checking for infinite recursion.
    int rrem = r % rblock;
    int crem = c % cblock;

    bool success = true;
    if (rrem || crem) {
        if ((r == rrem || rrem == 0) && (c == crem || crem == 0))
            success = false;
        else {
            if (rrem) success &= appendBlocks(rrem, c - crem, r - rrem, 0, remOpts, maxRBlock, maxCBlock);
            if (crem) success &= appendBlocks(r, crem, 0, c - crem, remOpts, maxRBlock, maxCBlock);
        }
    }
    return success;
}

bool RegisterLayout::append1DBlocks(int r, int c)
{
    // Skip pseudoblock cases (possible to support though).
    if (needsPseudoblock(hw, T, r, c, atype, astrategy, writable, false))
        return false;

    // Get total number of bytes to load. No masking supported, so stub if
    //  number of bytes not divisible by 16 (1 oword).
    int nbytes = r * c * T * T.components();
    int align = 16;
    if (astrategy.newDP) align = 4;

    if (nbytes & (align - 1))
        return false;

    // Get block info.
    int maxBBytes = 0;
    int ebytes = 0;
    int extra = 0;
    int addrShift = 0;
    int maxSIMD = 1;

    if (astrategy.newDP) {
        bool qword = (nbytes | atype.alignment) % 8 == 0;
        ebytes = qword ? 8 : 4;
        maxBBytes = ebytes * 64;
    } else {
        bool a64 = (astrategy.base.getModel() == ModelA64);
        bool oword = !a64;
        bool aoword = (astrategy.base.getModel() == ModelSC); // SC only does aligned oword
        if (hw >= HW::XeHP) oword |= ((atype.alignment & 0x1F) != 0);

        extra = aoword;
        ebytes = oword ? 16 : 32;
        maxBBytes = oword ? 128 : 256;
        if (astrategy.base.getModel() == ModelSLM && hw >= HW::XeHP) maxBBytes = 256;
        addrShift = (!a64 && oword && !aoword) ? 4 : 0;
        maxSIMD = 16;
    }

    // Get normalized dimensions.
    bool colMajor = isColMajor(atype.layout);
    int x = colMajor ? r : c;
    auto crosspack = atype.crosspack;

    // Counters for current x and y positions.
    int cx = 0, cy = 0;

    while (nbytes > 0) {
        // Carve out the largest chunk possible.
        int bbytes = std::min<int>(maxBBytes, rounddown_pow2(nbytes));
        int belems = bbytes / T;

        // Create a true load block for first (possibly partial) row/column.
        // Then, create additional no-load blocks for any further (possible partial)
        //   rows/columns until block is exhausted.
        bool first = true;
        while (belems > 0) {
            int nxRem = belems / crosspack;
            int nx = std::min<int>(nxRem, x - cx);
            if (nx <= 0) stub();
            if (cy % crosspack) return false;

            RegisterBlock block;

            block.hw = hw;
            block.ld = nx;
            (colMajor ? block.nr : block.nc) = nx;
            (colMajor ? block.nc : block.nr) = crosspack;
            (colMajor ? block.offsetR : block.offsetC) = cx;
            (colMajor ? block.offsetC : block.offsetR) = cy;
            block.component = 0;
            block.colMajor = colMajor;
            block.splitComplex = false;
            block.byteGlue = false;
            block.cxComponent = RegisterBlock::Interleaved;

            if (first) {
                block.ebytes = ebytes;
                block.count = div_up(bbytes, ebytes);
                block.simdSize = std::min(maxSIMD, roundup_pow2(bbytes) >> 2);
            } else
                block.ebytes = block.count = block.simdSize = 0;

            block.extra = extra;
            block.clearFlag();
            block.colMask = MaskInfo::None();
            block.rowMask = MaskInfo::None();
            block.colFragment = 0;
            block.rowFragment = 0;

            block.crosspack = crosspack;
            block.remainderR = false;
            block.remainderC = false;
            block.noRowsOK = false;
            block.noColsOK = false;
            block.descRemR = false;
            block.descRemC = false;
            block.descAssigned = false;
            block.addrShift = addrShift;
            block.offsetAddr = 0;
            block.hasNoLoad = false;
            block.msgRegs = std::max(1, bbytes >> GRF::log2Bytes(hw));

            if (first && cx == 0 && (nxRem % x) == 0) {
                // Shortcut: one register block can represent this block access.
                int ny = belems / x;
                (colMajor ? block.nc : block.nr) = ny;
                cy += ny;
                belems = 0;
            } else {
                cx += nx;
                belems -= nx * crosspack;
                if (cx == x) {
                    cy += crosspack; cx = 0;
                }
                block.hasNoLoad = first && (belems > 0);
                first = false;
            }

            list.push_back(block);
        }

        nbytes -= bbytes;
    }

    return true;
}

// Repoint this layout at a different in-memory representation.
void RegisterLayout::retarget(const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_)
{
    atype = atype_;
    astrategy = astrategy_;

    // Any previously-computed address-coalescing offsets were derived for the old
    //   addressing scheme and may not be valid for the new one (e.g. a packed layout
    //   with a different row/column pitch than the original memory layout). Clear them
    //   before recomputing, so that blocks the new coalesceAddrs() pass doesn't touch
    //   don't retain stale, incorrect offsets.
    for (auto &block: list)
        block.offsetAddr = 0;

    coalesceAddrs();
}

// Return maximum immediate address offset for a send message.
static inline int maxOffsetAddr(Type T, const MatrixAddressingStrategy &astrategy)
{
    switch (astrategy.base.getModel()) {
        case ModelA64:
        case ModelSLM: return 1 << 19;
        case ModelA32:
        case ModelBTS: return 1 << 11;
        default: return 0;
    }
}

// Identify and combine block address registers that differ only by constant offsets.
void RegisterLayout::coalesceAddrs()
{
    if (hw < HW::Xe2 || empty() || !astrategy.newDP || astrategy.noCoalesce) return;

    RegisterBlock *anchor = &list[0];
    int max = maxOffsetAddr(T, astrategy);

    for (auto &block: *this) {
        int dr = block.offsetR - anchor->offsetR;
        int dc = block.offsetC - anchor->offsetC;

        auto accessType = block.implAccessType(atype, astrategy);

        if (isBlock2D(accessType)) {
            if (block.nr == anchor->nr && block.nc == anchor->nc && block.count == anchor->count) {
                int ox, oy;
                switch (atype.layout) {
                    case MatrixLayout::N: ox = dr; oy = dc; break;
                    case MatrixLayout::T: ox = dc; oy = dr; break;
                    default: return;
                }
                block.set2DOffset(ox * T / block.ebytes, oy);
            } else {
                // No match. Make this block the new anchor.
                anchor = &block;
            }
        } else {
            switch (atype.layout) {
                case MatrixLayout::N: if (dc == 0) block.offsetAddr = dr; break;
                case MatrixLayout::T: if (dr == 0) block.offsetAddr = dc; break;
                case MatrixLayout::Pr:
                case MatrixLayout::Pc:
                    auto offsetX = (atype.layout == MatrixLayout::Pc) ? &RegisterBlock::offsetR
                                                                      : &RegisterBlock::offsetC;
                    if (block.*offsetX / atype.packSize == anchor->*offsetX / atype.packSize)
                        block.offsetAddr = untile(T, atype, block);
                    break;
            }

            block.offsetAddr *= T;
            if (block.offsetAddr >= max || block.offsetAddr < -max)
                block.offsetAddr = 0;
            if (one_of(accessType, {AccessType::Scattered, AccessType::ChannelScattered}))
                if (block.simdSize > anchor->simdSize)
                    block.offsetAddr = 0;
            if (block.offsetAddr & 0x3)
                block.offsetAddr = 0;

            if (block.offsetAddr == 0)
                anchor = &block;
        }
    }
}

void RegisterLayout::postprocess()
{
    postprocess2D();
    postprocessMultitile();
    postprocessLargeCP();
    postprocessDPASW();
}

// Split 2D block array loads into multiple blocks.
void RegisterLayout::postprocess2D()
{
    if (!isBlock2D(astrategy.accessType)) return;

    int maxCount = 1;
    for (auto &block: *this)
        maxCount = std::max(maxCount, int(block.count));
    if (maxCount == 1) return;

    vector<RegisterBlock> xlist;
    xlist.reserve(blocks() * maxCount);

    for (auto &block: *this) {
        bool cm = block.colMajor;
        auto RegisterBlock::* nx      = cm ? &RegisterBlock::nr      : &RegisterBlock::nc;
        auto RegisterBlock::* offsetX = cm ? &RegisterBlock::offsetR : &RegisterBlock::offsetC;

        auto nblock = block;
        nblock.*nx /= block.count;
        nblock.ld /= block.count;

        for (int i = 0; i < block.count; i++) {
            xlist.push_back(nblock);
            nblock.*offsetX += nblock.*nx;
            nblock.simdSize = 0;           // Blocks > 0 do not need loads.
        }
    }

    list = std::move(xlist);
}

// Split blocks that span multiple tiles. Requires each tile to be contained within a single block.
void RegisterLayout::postprocessMultitile()
{
    if (!atype.tileR || !atype.tileC) return;
    if (isLargeCrosspack(T, atype.crosspack)) return;

    bool needToSplit = false;
    for (const auto &block: *this)
        needToSplit |= (block.colMajor ? (block.nr > atype.tileR) : (block.nc > atype.tileC));

    if (!needToSplit) return;

    vector<RegisterBlock> xlist;
    xlist.reserve(blocks());

    for (const auto &block: *this) {
        auto nx      = block.colMajor ? &RegisterBlock::nr      : &RegisterBlock::nc;
        auto ny      = block.colMajor ? &RegisterBlock::nc      : &RegisterBlock::nr;
        auto offsetX = block.colMajor ? &RegisterBlock::offsetR : &RegisterBlock::offsetC;
        auto offsetY = block.colMajor ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;
        auto tileX   = block.colMajor ? atype.tileR             : atype.tileC;
        auto tileY   = block.colMajor ? atype.tileC             : atype.tileR;

        if (block.*nx == tileX) {
            xlist.push_back(block);
            continue;
        }

        if (block.*nx % tileX || block.*offsetX % tileX || block.*ny % tileY || block.*offsetY % tileY) stub();
        if (isTransposing(astrategy.accessType)) stub();

        auto nblock = block;
        nblock.*nx = tileX;
        nblock.*ny = tileY;
        nblock.ld = tileX;

        for (int j = 0; j < block.*ny / tileY; j++) {
            for (int i = 0; i < block.*nx / tileX; i++) {
                nblock.*offsetX = block.*offsetX + i * tileX;
                nblock.*offsetY = block.*offsetY + j * tileY;
                xlist.push_back(nblock);
                nblock.simdSize = 0;
            }
        }
    }

    list = std::move(xlist);
}

// Split large crosspack blocks into smaller pieces so that they can be transposed.
void RegisterLayout::postprocessLargeCP()
{
    if (!isLargeCrosspack(T, atype.crosspack))
        return;

    bool haveLargeCP = false;
    for (const auto &block: *this) {
        haveLargeCP |= isLargeCrosspack(T, block.crosspack);
        if (haveLargeCP) break;
    }

    if (!haveLargeCP) return;

    vector<RegisterBlock> xlist;
    xlist.reserve(blocks());

    for (const auto &block: *this) {
        if (!isLargeCrosspack(T, block.crosspack))
            xlist.push_back(block);
        else {
            auto ny      = block.colMajor ? &RegisterBlock::nc      : &RegisterBlock::nr;
            auto offsetY = block.colMajor ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

            if (block.*ny % block.crosspack)
                return;
            int blocks = (block.*ny / block.crosspack);
            auto nblock = block;
            nblock.*ny = block.crosspack;
            nblock.simplify(T);
            for (int i = 0; i < blocks; i++) {
                xlist.push_back(nblock);
                nblock.simdSize = 0;
                nblock.*offsetY += nblock.*ny;
            }
        }
    }

    list = std::move(xlist);
}

// Remove implied blocks from a dpasw src2 layout.
void RegisterLayout::postprocessDPASW()
{
    if (!astrategy.dpasw)
        return;

    vector<RegisterBlock> nlist;
    nlist.reserve(blocks() / 2);

    auto tile    = colMajor() ? astrategy.tileC         : astrategy.tileR;
    auto offsetX = colMajor() ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;

    for (const auto &block: *this)
        if ((block.*offsetX % (2 * tile)) < tile)
            nlist.push_back(block);

    list = std::move(nlist);
}

void RegisterLayout::finalize()
{
    int offsetBytes = 0;
    for (auto &block: *this) {
        if (T.is3() && astrategy.accessType == AccessType::Block2DTranspose
                && !astrategy.prefetch && block.isLoadBlock()) {
            int width, height, count;
            block.getBlock2DWH(width, height, count, atype, astrategy.prefetch);
            block.msgRegs = GRF::bytesToGRFs(hw, width * height * block.ebytes * count);
        }
        if (block.isLoadBlock() || isBlock2D(astrategy.accessType))
            offsetBytes = ngen::utils::alignup_pow2(offsetBytes, GRF::bytes(hw));
        block.calcBytes(T, astrategy);
        block.offsetBytes = offsetBytes;
        offsetBytes += block.bytes;
        block.simplify(T);
    }
}

Subregister RegisterLayout::find(int i, int j, const GRFMultirange &regs, int *outNElems, const RegisterBlock **outBlock,
                                 int cxComponent, int component) const
{
    int ecomponent = component;
    for (auto &block: *this) {
        int ii = i - block.offsetR;
        int jj = j - block.offsetC;
        if (ii >= 0 && ii < block.nr && jj >= 0 && jj < block.nc
                    && ecomponent == block.component
                    && one_of(block.cxComponent, {int8_t(cxComponent), RegisterBlock::Interleaved})) {
            if (outBlock) *outBlock = &block;
            return block.find(T, ii, jj, regs, outNElems, cxComponent, component);
        }
    }

    stub("Could not find requested matrix element in layout.");
}

RegisterRegion RegisterLayout::findRegion(int i, int j, const GRFMultirange &regs, int *outNElems, const RegisterBlock **outBlock,
                                          int cxComponent, int component, bool allow2D) const
{
    const RegisterBlock *block;
    auto reg = find(i, j, regs, outNElems, &block, cxComponent, component);
    if (outBlock)
        *outBlock = block;
    return blockRegion(T, reg, *block, i - block->offsetR, j - block->offsetC, outNElems, cxComponent, allow2D);
}

bool RegisterLayout::colMajor() const
{
    if (empty()) stub("Empty layout.");
    return list[0].colMajor;              // All layouts we create are homogeneous currently.
}

int RegisterLayout::crosspack() const
{
    if (empty()) stub("Empty layout.");

    int crosspack = list[0].crosspack;
    for (auto &block: list)
        if (block.crosspack != crosspack)
            stub("Crosspack is not uniform");
    return crosspack;
}

bool RegisterLayout::hasFullCrosspack(int crosspack) const
{
    for (auto &block: *this) {
        if (block.crosspack != crosspack)
            return false;
        if ((block.colMajor ? block.nc : block.nr) % crosspack)
            return false;
    }
    return true;
}

bool RegisterLayout::hasTiling(int tileR, int tileC) const
{
    for (auto &block: *this) {
        if (tileR > 0)
            if (block.offsetR / tileR != (block.offsetR + block.nr - 1) / tileR)
                return false;
        if (tileC > 0)
            if (block.offsetC / tileC != (block.offsetC + block.nc - 1) / tileC)
                return false;
    }
    return true;
}

bool RegisterLayout::hasRemainders(bool remainderR, bool remainderC) const
{
    for (auto &block: *this)
        if ((remainderR && block.remainderR) || (remainderC && block.remainderC))
            return true;
    return false;
}

bool RegisterLayout::hasFragmenting(bool ignoreWholeFragR, bool ignoreWholeFragC) const
{
    if (empty()) return false;

    for (auto &block: *this) {
        if (block.rowFragment && !(ignoreWholeFragR && block.rowFragment >= nr))
            return true;
        if (block.colFragment && !(ignoreWholeFragC && block.colFragment >= nc))
            return true;
    }
    return false;
}

bool RegisterLayout::hasMasking() const
{
    for (auto &block: *this)
        if (block.rowMask || block.colMask || block.hasFlag())
            return true;
    return false;
}

bool RegisterLayout::hasFlags() const
{
    for (auto &block: *this)
        if (block.hasFlag())
            return true;
    return false;
}

int RegisterLayout::maxLoadBlock() const
{
    int result = 0;
    for (auto &block: *this)
        result = std::max<int>(result, block.msgRegs);
    return result;
}

int RegisterLayout::regs() const
{
    if (empty()) return 0;

    int lastByte = 0;
    for (auto &block: *this) {
        lastByte = std::max(lastByte, block.offsetBytes + block.bytes);
    }

    return GRF::bytesToGRFs(hw, lastByte);
}

void RegisterLayout::unlinkFromMemory()
{
    for (auto &block: *this)
        block.unlinkFromMemory();
    astrategy.base = AddressBase();
}

void RegisterLayout::sort(bool reverse)
{
    auto order = [this, reverse](const RegisterBlock &block) {
        return untile(T, atype, block, nr, nc, astrategy.tileR, astrategy.tileC, reverse);
    };

    std::sort(list.begin(), list.end(), [&](const RegisterBlock &b1, const RegisterBlock &b2) {
        return (order(b1) < order(b2));
    });
}

void RegisterLayout::assignUniformMask(FlagRegister flag, HW hw, int idx)
{
    for (auto &block: *this) {
        if (block.flag[idx]) stub();     /* Already has a flag? */
        block.flag[idx] = VirtualFlag(flag, hw);
    }
}

bool RegisterLayout::assignAllDescs()
{
    for (auto &block: *this) {
        if (!block.descRemR && !block.descRemC)
            continue;
        if (block.simdSize != list[0].simdSize)
            return false;
        block.descAssigned = true;
        block.sfid = list[0].sfid;
    }

    return true;
}

bool RegisterLayout::match(const RegisterLayout &ref)
{
    auto nlist = list;

    if (ref.regs() >= GRF::maxRegs()) return false;
    if (T.bits() != ref.T.bits()) return false;
    if (T.components() != ref.T.components()) return false;

    int lastByteAdjust = 0;
    int grfBytes = GRF::bytes(hw);

    for (auto &nblock: nlist) {
        const RegisterBlock *blockRef;
        auto sr = ref.find(nblock.offsetR, nblock.offsetC, GRFRange(0, GRF::maxRegs() - 2), nullptr, &blockRef);

        // Check:
        //  1. Does this register block's offset match the reference block's offset?
        if (sr.getByteOffset() != (nblock.offsetBytes & (grfBytes - 1))) return false;

        //  2. Is there any free space in the register block?
        if (nblock.nr * nblock.nc * T != nblock.bytes) return false;

        //  3. Does this register block's data layout match the reference block's layout?
        if (blockRef->colMajor != nblock.colMajor) return false;
        if (blockRef->crosspack != nblock.crosspack) return false;

        //  4. Does this register block fit inside the reference block?
        auto RegisterBlock::* nx = nblock.colMajor ? &RegisterBlock::nr : &RegisterBlock::nc;
        auto RegisterBlock::* ny = nblock.colMajor ? &RegisterBlock::nc : &RegisterBlock::nr;

        if (nblock.*nx > blockRef->*nx) return false;
        if (nblock.*ny > blockRef->*ny) return false;

        //  5. Are the leading dimensions and padding compatible?
        if (nblock.*nx < blockRef->*nx) {
            if (nblock.*ny > nblock.crosspack) return false;
            if (nblock.*ny < nblock.crosspack && nblock.*ny < blockRef->*ny) return false;
        }

        if (nblock.*ny > nblock.crosspack && (nblock.ld != blockRef->ld))
            return false;

        // Point this register block where it belongs.
        auto newOffsetBytes = sr.getBase() * grfBytes + sr.getByteOffset();
        auto byteAdjust = newOffsetBytes - nblock.offsetBytes;

        // No-load blocks need to stay with their parent blocks.
        if (nblock.simdSize == 0 && byteAdjust != lastByteAdjust)
            return false;

        nblock.offsetBytes = newOffsetBytes;
        lastByteAdjust = byteAdjust;
    }

    // Success! Commit changes.
    std::swap(nlist, list);
    return true;
}

void RegisterLayout::updateDims()
{
    if (blocks() == 0) {
        nr = nc = 0;
        return;
    }

    int r0 = std::numeric_limits<int>::max();
    int r1 = 0;
    int c0 = std::numeric_limits<int>::max();
    int c1 = 0;

    for (const auto &block: *this) {
        r0 = std::min<int>(r0, block.offsetR);
        c0 = std::min<int>(c0, block.offsetC);
        r1 = std::max<int>(r1, block.offsetR + block.nr);
        c1 = std::max<int>(c1, block.offsetC + block.nc);
    }

    nr = r1 - r0;
    nc = c1 - c0;
}

RegisterLayout RegisterLayout::reblock(vector<int32_t> &blockMap, const RegisterLayout &ref) const
{
    auto nblockRef = ref.blocks();

    auto dst = *this;
    dst.list.clear();
    dst.list.reserve(nblockRef);

    blockMap.clear();
    blockMap.reserve(nblockRef + 1);
    blockMap.push_back(0);

    for (auto &blockRef: ref) {
        for (auto &blockSrc: *this) {
            int ii0 = blockRef.offsetR - blockSrc.offsetR, ii1 = ii0 + blockRef.nr;
            int jj0 = blockRef.offsetC - blockSrc.offsetC, jj1 = jj0 + blockRef.nc;
            if (ii0 >= blockSrc.nr || ii1 <= 0) continue;
            if (jj0 >= blockSrc.nc || jj1 <= 0) continue;
            ii0 = std::max(ii0, 0);
            jj0 = std::max(jj0, 0);
            ii1 = std::min(ii1, int(blockSrc.nr));
            jj1 = std::min(jj1, int(blockSrc.nc));
            auto blockDst = blockSrc.trySlice(T, false, ii0, ii1, ii0, ii1, true, atype, astrategy)
                                    .trySlice(T, true,  jj0, jj1, jj0, jj1, true, atype, astrategy);
            if (!blockDst.valid()) stub("Could not reblock layout");
            dst.list.push_back(blockDst);
        }
        blockMap.push_back(int32_t(dst.list.size()));
    }

    return dst;
}

RegisterLayout RegisterLayout::slice(bool column, int x1, int x2, bool overrunOK, bool decoalesce) const
{
    auto result = trySlice(column, x1, x2, overrunOK, decoalesce);
    if (!result) stub("Could not slice register layout");
    return result;
}

RegisterLayout RegisterLayout::slice(vector<RegisterRange> &subaddrs, const vector<RegisterRange> &addrs, bool column, int x1, int x2, bool overrunOK) const
{
    auto result = trySlice(subaddrs, addrs, column, x1, x2, overrunOK);
    if (!result) stub("Could not slice register layout");
    return result;
}

RegisterLayout RegisterLayout::slice(vector<int> &indices, bool column, int x1, int x2, bool overrunOK) const
{
    auto result = trySlice(indices, column, x1, x2, overrunOK);
    if (!result) stub("Could not slice register layout");
    return result;
}

RegisterLayout RegisterLayout::trySlice(bool column, int x1, int x2, bool overrunOK, bool decoalesce) const
{
    return trySlice(nullptr, nullptr, nullptr, column, x1, x2, overrunOK, decoalesce);
}

RegisterLayout RegisterLayout::trySlice(vector<RegisterRange> &subaddrs, const vector<RegisterRange> &addrs, bool column, int x1, int x2, bool overrunOK) const
{
    return trySlice(&subaddrs, nullptr, &addrs, column, x1, x2, overrunOK);
}

RegisterLayout RegisterLayout::trySlice(vector<int> &indices, bool column, int x1, int x2, bool overrunOK) const
{
    return trySlice(nullptr, &indices, nullptr, column, x1, x2, overrunOK);
}

RegisterLayout RegisterLayout::trySlice(vector<RegisterRange> *subaddrs, vector<int> *indices, const vector<RegisterRange> *addrs,
                                        bool column, int x1, int x2, bool overrunOK, bool decoalesce) const
{
    auto RegisterBlock::*nq      = column ? &RegisterBlock::nc      : &RegisterBlock::nr;
    auto RegisterBlock::*offsetQ = column ? &RegisterBlock::offsetC : &RegisterBlock::offsetR;
    auto RegisterLayout::*nql    = column ? &RegisterLayout::nc     : &RegisterLayout::nr;

    auto sublayout = *this;
    sublayout.list.clear();
    sublayout.*nql = clamp(x2, 0, sublayout.*nql)
                   - clamp(x1, 0, sublayout.*nql);

    if (subaddrs) subaddrs->clear();
    if (indices) indices->clear();

    bool sharedOK = true;

    for (int b = 0; b < blocks(); b++) {
        auto &block = list[b];
        if (block.offsetAddr == 0) sharedOK = true;

        int qq1Unclamped = x1 - block.*offsetQ;
        int qq2Unclamped = x2 - block.*offsetQ;
        int qq1 = clamp<int>(qq1Unclamped, 0, block.*nq);
        int qq2 = clamp<int>(qq2Unclamped, 0, block.*nq);
        if (qq2 > qq1) {
            auto subblock = block.trySlice(T, column, qq1, qq2, qq1Unclamped, qq2Unclamped, overrunOK, atype, astrategy);
            bool ok = subblock.valid();
            if (subaddrs || indices) {
                ok = ok && (subblock.offsetR == block.offsetR)
                        && (subblock.offsetC == block.offsetC)
                        && (subblock.offsetAddr == 0 || sharedOK);
            } else if (decoalesce)
                subblock.offsetAddr = 0;

            if (!ok)
                return RegisterLayout();

            sublayout.list.push_back(subblock);
            if (subaddrs) subaddrs->push_back((*addrs)[b]);
            if (indices) indices->push_back(b);
        } else if (block.offsetAddr == 0)
            sharedOK = false;
    }

    return sublayout;
}

RegisterLayout RegisterLayout::tryUpgradeToBlock2D(const MatrixAddressing &atype2D, const MatrixAddressingStrategy &astrategy2D) const
{
    auto layout2D = *this;
    layout2D.list.clear();

    if (empty())                  return layout2D;
    if (isPacked(atype2D.layout)) return RegisterLayout();

    bool transpose = isTransposing(astrategy.accessType);
    bool regCM = colMajor();

    if (transpose && !one_of(sizeof(T), {4u, 8u}))
        return RegisterLayout();
    if (astrategy2D.accessType != (transpose ? AccessType::Block2DTranspose : AccessType::Block2D))
        return RegisterLayout();

    layout2D.atype = atype2D;
    layout2D.astrategy = astrategy2D;

    int r0 = -1, c0 = -1, b0 = -1;
    int nr = 0, nc = 0;
    bool ok = true, lastDense = false;

    auto make2DBlock = [&] {
        if (r0 < 0 || c0 < 0) return;
        ok = ok && layout2D.appendBlocks(nr, nc, r0, c0, AllowFragment);
    };

    for (auto &block: *this) {
        unsigned omask = GRF::bytes(hw) - 1;

        if ((block.offsetBytes & omask) || (block.bytes & omask))          return RegisterLayout();
        if (!transpose && (block.colMajor ? block.nr : block.nc) * T > 64) return RegisterLayout();    /* avoid lots of small blocks */

        // 2D block messages pad each row in GRF to the next power of two.
        int inner = block.colMajor ? block.nr : block.nc;
        int outer = block.colMajor ? block.nc : block.nr;
        int ld2D = roundup_pow2(inner);
        if (block.crosspack != 1 || (outer > 1 && block.ld != ld2D)) return RegisterLayout();

        // Only merge with the previous block if it had no trailing padding.
        bool consecutive = lastDense && (block.offsetBytes == (b0 + GRF::bytes(hw)));
        if (regCM && block.offsetC == c0 + nc && consecutive && nr == block.nr)
            nc++;
        else if (!regCM && block.offsetR == r0 + nr && consecutive && nc == block.nc)
            nr++;
        else {
            make2DBlock();
            r0 = block.offsetR; c0 = block.offsetC;
            nr = block.nr; nc = block.nc;
        }
        b0 = block.offsetBytes;
        lastDense = (block.bytes == ld2D * outer * T);
    }

    make2DBlock();
    if (!ok) return RegisterLayout();

    layout2D.sort();
    layout2D.postprocess();
    layout2D.finalize();

    // Update offsets to match source layout.
    for (auto &block: layout2D) {
        auto sr = find(block.offsetR, block.offsetC, GRFRange(0, 254));

        block.offsetBytes = sr.getBase() * GRF::bytes(hw) + sr.getByteOffset();
    }

    return layout2D;
}

GEMMSTONE_NAMESPACE_END
