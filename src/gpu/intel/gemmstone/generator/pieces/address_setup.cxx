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
#include "layout_utils.hpp"
#include "state_utils.hpp"
#include "ngen_object_helpers.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;

// Ugly helpers handling address shifts. constexpr if would clean this all up.
template <HW hw>
template <typename BO>
typename std::enable_if<!std::is_base_of<RegData, BO>::value, BO>::type
Generator<hw>::startShift(const BO &ptr, int shift, CommonState &state)
{
    return ptr >> shift;
}

template <HW hw>
Subregister Generator<hw>::startShift(const MultishiftSubregister &ptr, int shift, CommonState &state)
{
    return ptr >> shift;
}

template <HW hw>
SubregisterPair Generator<hw>::startShift(const SubregisterPair &ptr, int shift, CommonState &state)
{
    if (shift == 0)
        return ptr;
    else
        return SubregisterPair(startShift(ptr.getReg(0), shift, state));
}

template <HW hw>
template <typename BO>
typename std::enable_if<std::is_base_of<RegData, BO>::value, BO>::type
Generator<hw>::startShift(const BO &ptr, int shift, CommonState &state)
{
    BO ptrShifted = ptr;

    // Shift pointer as necessary.
    if (shift > 0) {
        ptrShifted = state.ra.alloc_sub(ptr.getType());
        shr(1, ptrShifted, ptr, shift);
    }

    return ptrShifted;
}

template <HW hw>
template <typename BO, typename BI>
typename std::enable_if<!std::is_base_of<RegData, BO>::value>::type
Generator<hw>::doneShift(const BO &ptr, const BI &ptrShifted, int shift, CommonState &state) {}

template <HW hw>
template <typename BO, typename BI>
typename std::enable_if<std::is_base_of<RegData, BO>::value>::type
Generator<hw>::doneShift(const BO &ptr, const BI &ptrShifted, int shift, CommonState &state)
{
    if (shift > 0)
        state.ra.release(ptrShifted);
}

template <HW hw>
void Generator<hw>::doneShift(const SubregisterPair &ptr, const SubregisterPair &ptrShifted, int shift, CommonState &state)
{
    if (shift > 0)
        doneShift(ptr.getReg(0), ptrShifted.getReg(0), shift, state);
}

// Bank conflict avoidance helpers.
inline namespace {
    template <typename T> struct ACHelper {
        static T avoidConflict(HW hw, const T &x, const RegData &other) { return x; }
    };
    template <> struct ACHelper<SubregisterPair> {
        static Subregister avoidConflict(HW hw, const SubregisterPair &x, const RegData &other) {
            return x.getRegAvoiding(hw, other);
        }
    };
}
template <typename T>
decltype(ACHelper<T>::avoidConflict(HW::Unknown, std::declval<T>(), RegData()))
avoidConflict(HW hw, const T &x, const RegData &other) {
    return ACHelper<T>::avoidConflict(hw, x, other);
}

// Address setup utility routines.
static inline bool canRelAddr(const RegisterBlock &blockSrc, const RegisterBlock &blockDst,
                              const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    if (!blockSrc.isLoadBlock() || !blockDst.isLoadBlock())
        return false;

    auto accessSrc = blockSrc.implAccessType(atype, astrategy);
    auto accessDst = blockDst.implAccessType(atype, astrategy);
    if (accessSrc == AccessType::Block && accessDst == AccessType::Block)
        return true;
    if (accessSrc == AccessType::Scattered && accessDst == AccessType::Scattered) {
        if (blockSrc.ebytes != blockDst.ebytes)
            return false;
        if (blockSrc.ebytes == 1 && blockSrc.count != blockDst.count)
            return false;
    }
    return (blockSrc.simdSize >= blockDst.simdSize);
}

static inline int getPartialCrosspack(Type T, const MatrixAddressing &atype, const RegisterBlock &block)
{
    if (block.ebytes == 1 && !isLargeCrosspack(T, atype.crosspack))
        return div_up(atype.crosspack, block.colMajor ? block.nc : block.nr);
    else
        return 1;
}

// Output code for setting up address/header GRFs for a single block, given
//  the base pointer (a Subregister, MultishiftSubregister or integer) and leading dimension.
template <HW hw>
template <typename BO>
void Generator<hw>::setupAddr(Type T, const RegisterRange &addr, const BO &ptr, const RegisterBlock &block, const Subregister &bld,
                              const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                              const CommonStrategy &strategy, CommonState &state,
                              const Address2DParams &params, LDMultiples ldMultiples)
{
    bool a64 = astrategy.base.getModel() == ModelA64;

    auto ensureLDMultiples = [&](int n) {
        if (ldMultiples.count < n) {
            ldMultiples = createLDMultiples(a64, n, bld, strategy, state);
            if (ldMultiples.range.isInvalid()) throw out_of_registers_exception();
            return true;
        } else
            return false;
    };

    // Nothing to do for non-load blocks.
    if (!block.isLoadBlock())
        return;

    auto effAccessType = block.effectiveAccessType(atype, astrategy);
    switch (effAccessType) {
        case AccessType::Scattered:
        case AccessType::ChannelScattered:
        case AccessType::PseudoBlock:
        {
            int simdSize = block.simdSize;
            auto consecutive = block.extra;
            bool pseudo = (effAccessType == AccessType::PseudoBlock);
            auto Tptr = a64 ? DataType::uq : DataType::ud;
            int ne = elementsPerGRF(hw, Tptr);
            int preshift = 0;

            auto oldIndexVec = state.indexVec;
            auto oldIVEntries = state.ivEntries;

            if (!pseudo && !isPacked(atype.layout)) {
                // Get pointers to successive rows/columns, strided by ld.
                bool allocLDMultiples = ensureLDMultiples(simdSize);

                for (int r = 0; r < addr.getLen(); r += 2) {
                    int nr = std::min(2, addr.getLen() - r);
                    int simd = nr * ne;
                    simd = std::min(32, simd);
                    auto ld0 = findLDMultiple(ldMultiples, a64, r * ne / consecutive, strategy, state);
                    auto ldStride = (ldMultiples.a64 && !a64) ? 2 : 1;
                    auto ldR = ld0(ldStride, consecutive, 0);
                    auto addrR = addr[r].retype(Tptr);
                    if (a64 && consecutive > 1 && hw >= HW::XeHP && !strategy.emulate.emulate64) { /* no swizzle in L pipe */
                        mov(simd, addr[r].ud(0)(2), ld0.ud(0)(ldStride * 2, consecutive, 0));
                        mov(simd, addr[r].ud(1)(2), ld0.ud(1)(ldStride * 2, consecutive, 0));
                        if (ptr != 0) add(simd, addrR, addrR, ptr);
                    } else if (ptr != 0)
                        eadd(simd, addrR, ldR, ptr, strategy, state);
                    else
                        emov(simd, addrR, ldR, strategy, state);
                }
                if (allocLDMultiples)
                    releaseLDMultiples(ldMultiples, state);
            } else {
                // Get pointers to successive elements, with constant stride.
                int tblock = isPacked(atype.layout) ? 1 : (block.colMajor ? block.nc : block.nr);
                extendIndexVec(simdSize, state);
                auto ivBase = accessIndexVec(0, state);
                auto iv = ivBase(tblock, tblock * consecutive, 0);
                uint16_t stride;
                preshift = block.addrShift;
                auto ptrShifted = startShift(ptr, block.addrShift, state);

                if (pseudo) {
                    stride = (block.ebytes * block.count * getPartialCrosspack(T, atype, block) * consecutive / tblock) >> preshift;
                } else {
                    int tile = isColMajor(atype.layout) ? atype.tileR : atype.tileC;
                    if (tile == 0)
                        tile = atype.packSize;
                    int psElems = (isLargeCrosspack(T, atype.crosspack) ? 1 : tile) * atype.crosspack;
                    stride = uint16_t(psElems * T) >> preshift;
                }

                if (a64) {
                    const int udStride = (hw >= HW::XeHP) ? 2 : 1;
                    int simd1 = std::min(2 * ne, simdSize);
                    int simd2 = simdSize - simd1;
                    if (udStride == 2 && simd2) {
                        auto iv2 = accessIndexVec(simd1 / consecutive, state)(tblock, tblock * consecutive, 0);
                        mulConstant(simd2, addr[2].ud(0)(udStride), iv2, stride);
                        mulConstant(simd1, addr[0].ud(0)(udStride), iv, stride);
                    } else {
                        mulConstant(simdSize, addr[0].ud(0)(udStride), iv, stride);
                    }
                    if (simd2)
                        eadd(simd2, addr[2].uq(), addr[udStride].ud(0)(udStride), ptrShifted, strategy, state);
                    eadd(simd1, addr[0].uq(), addr[0].ud(0)(udStride), ptrShifted, strategy, state);
                } else if (ptrShifted != 0) {
                    if (consecutive > 1 || tblock > 1 || hw >= HW::Xe3p)
                    {
                        mulConstant<uint32_t>(simdSize, addr, iv, stride);
                        add<uint32_t>(simdSize, addr, addr, ptrShifted);
                    } else
                        emad(simdSize, addr[0].ud(), ptrShifted, ivBase(1), int32_t(stride), strategy, state);
                } else
                    mulConstant<uint32_t>(simdSize, addr, iv, stride);

                doneShift(ptr, ptrShifted, block.addrShift, state);
            }

            // Add offsets for consecutive elements in scattered accesses.
            if (consecutive > 1) {
                if ((consecutive - 1) * block.ebytes >= 0x10) stub();
                if (consecutive > 4) stub();
                uint8_t incs[4];
                for (int idx = 0; idx < 4; idx++)
                    incs[idx] = (block.ebytes * (idx % consecutive)) >> preshift;

                if (!a64) {
                    auto incImm = Immediate::uv(incs[0], 0, incs[1], 0, incs[2], 0, incs[3], 0);
                    add<uint32_t>(simdSize, addr, addr, incImm);
                } else {
                    if (consecutive > 2) stub();
                    auto incImm = Immediate::uv(incs[0], 0, 0, 0, incs[1], 0, 0, 0);
                    auto temp = state.ra.alloc_range(2);
                    mov<uint32_t>(2 * elementsPerGRF<uint32_t>(hw), temp, incImm);
                    map(hw, Tptr, addr, addr, strategy, [&](int simd, GRF r1, GRF _) {
                        eadd<uint64_t>(simd, r1, temp[0].ud(0)(2), r1, strategy, state);
                    });
                    state.ra.safeRelease(temp);
                }
            }

            // Scale if needed.
            if (block.addrShift > preshift)
                shr<uint32_t>(simdSize, addr, addr, block.addrShift - preshift);

            // Restore original cached index vector in case we extended it.
            releaseRanges(state.indexVec, state);
            state.indexVec  = std::move(oldIndexVec);
            state.ivEntries = oldIVEntries;
            reclaimRanges(state.indexVec, state);
            break;
        }
        case AccessType::Block:
            if (astrategy.base.getModel() == ModelA64) {
                emov(1, addr[0].uq(0), ptr, strategy, state);
            } else if (astrategy.newDP) {
                mov(1, addr[0].ud(0), ptr);
            } else if (block.addrShift > 0)
                shr(1, addr[0].ud(2), ptr, block.addrShift);
            else
                mov(1, addr[0].ud(2), ptr);
            break;
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI: {
            if (astrategy.base.getModel() != ModelA64) hw_unsupported();

            // Assemble some information.
            bool memCM = isColMajor(atype.layout);
            int bw, bh, bcount, multiX;
            block.getBlock2DWH(bw, bh, bcount, atype, astrategy.prefetch, &multiX);

            auto iremR = params.remR, iremC = params.remC;
            if (!block.remainderR) iremR.invalidate();
            if (!block.remainderC) iremC.invalidate();

            auto remW   = memCM ? iremR : iremC;
            auto remH   = memCM ? iremC : iremR;
            auto &nx    = memCM ? params.rows : params.cols;
            auto &ny    = memCM ? params.cols : params.rows;
            auto fixedX = memCM ? params.fixedRows : params.fixedCols;
            auto fixedY = memCM ? params.fixedCols : params.fixedRows;
            auto &offX  = memCM ? params.offR : params.offC;
            auto &offY  = memCM ? params.offC : params.offR;
            auto boffX  = memCM ? block.offsetR : block.offsetC;
            auto boffY  = memCM ? block.offsetC : block.offsetR;

            if (T.is3() && astrategy.accessType == AccessType::Block2DTranspose)
                boffX = div_up(boffX * T.bits(), block.ebytes * 8);
            else {
                boffX *= T;
                if (boffX % block.ebytes) stub();
                boffX /= block.ebytes;
            }

            // If the base address may not be aligned to HW requirements,
            //  we need to emit code to align it down and offset x/width appropriately.
            int baseAlign = block2DBaseAlignment(hw, getStepping());
            bool doBaseAdjust = (atype.alignment & (baseAlign - 1)) != 0;
            if (doBaseAdjust && !astrategy.address2D) stub();
            Subregister baStorage, baseAdjust, baseAdjustElems;

            int widthAlign = block2DWidthAlignment(T, block, atype, astrategy);

            if (!astrategy.address2D)
                mov(4, addr[0].ud(4)(1), 0u);

            if (doBaseAdjust) {
                baStorage = state.ra.alloc_sub<uint32_t>();
                baseAdjust = baStorage.uw(0);
                baseAdjustElems = baStorage.uw(1);
                if (!offX.isValid())
                    baseAdjustElems = addr[0].ud(5);

                and_(1, baseAdjust, ptr.ud(0), baseAlign - 1);
                and_(1, addr[0].ud(0), ptr.ud(0), ~uint32_t(baseAlign - 1));
                mov(1, addr[0].ud(1), ptr.ud(1));
                if (block.ebytes > 1)
                    shr(1, baseAdjustElems, baseAdjust, ilog2(block.ebytes));
                else
                    baseAdjustElems = baseAdjust;
            } else
                emov(1, addr[0].uq(0), ptr, strategy, state);

            if (astrategy.address2D) {
                if (params.rows.isInvalid() && params.fixedRows == 0) stub("Unknown matrix size.");

                nx.isValid() ? addScaled(1, addr[0].ud(2), -1, nx, T, state, true)
                             : mov(1, addr[0].ud(2), fixedX * T - 1);
                ny.isValid() ? add(1, addr[0].ud(3), ny, -1)
                             : mov(1, addr[0].ud(3), fixedY - 1);
                // u3 advances block.ebytes bytes per group of 8 elements,
                // so the scaling denominator is 8 * block.ebytes.
                offX.isValid() ? addScaled(1, addr[0].ud(5), boffX, offX, int(T.paddedSize()),
                                            T.is3() ? (8 * block.ebytes) : (block.ebytes * T.perByte()),
                                            state, astrategy.prefetch ? false : true) :
                  doBaseAdjust ? add(1, addr[0].ud(5), baseAdjustElems, boffX)
                               : mov(1, addr[0].ud(5), boffX);
                offY.isValid() ? add(1, addr[0].ud(6), offY, boffY)
                               : mov(1, addr[0].ud(6), boffY);
                if (doBaseAdjust) {
                    add(1, addr[0].ud(2), addr[0].ud(2), baseAdjust);
                    if (offX.isValid())
                        add(1, addr[0].ud(5), addr[0].ud(5), baseAdjustElems);
                }
                if (T.paddedSize() < widthAlign)
                    or_(1, addr[0].ud(2), addr[0].ud(2), widthAlign - 1);
            } else if (remW.isInvalid() && remH.isInvalid())
                emov(1, addr[0].uq(1), ((uint64_t(bw) * bcount * block.ebytes - 1) | (uint64_t(bh) * block.ebytes - 1) << 32), strategy, state);
            else {
                if (remW.isValid() && multiX > 1) stub();
                remW.isValid() ? addScaled(1, addr[0].ud(2), -1, remW.uw(), T, state, true)
                               : mov(1, addr[0].ud(2), bw * bcount * block.ebytes - 1);
                remH.isValid() ? mad(1, addr[0].ud(3), -1, remH.uw(), multiX)
                               : mov(1, addr[0].ud(3), bh - 1);
                if (remW.isValid() && T.paddedSize() < widthAlign)
                    or_(1, addr[0].ud(2), addr[0].ud(2), widthAlign - 1);
            }

            if (isPacked(atype.layout)) {
                auto pitch = bw * bcount * block.ebytes;
                if (pitch < 64 || pitch & 0xF) hw_unsupported();
                mov(1, addr[0].ud(4), pitch - 1);
            } else
                add(1, addr[0].ud(4), bld, -1);

            mov(1, addr[0].ud(7), (bw - 1) | ((bh - 1) << 8) | ((bcount - 1) << 16));

            state.ra.safeRelease(baStorage);
            break;
        }
        case AccessType::CacheLine: {
            int nx           = block.colMajor ? block.nr         : block.nc;
            int ny           = block.colMajor ? block.nc         : block.nr;
            bool hasRemX     = block.colMajor ? block.remainderR : block.remainderC;
            const auto &remX = block.colMajor ? params.remR      : params.remC;

            int trailing = (atype.alignment % 64) ? 1 : 0;
            int setback = block.ebytes * block.count;
            int xCacheLines = roundup_pow2(div_up(nx * T, 64) + trailing);
            int n = block.simdSize;

            bool allocLDMultiples = (ny > 2) && ensureLDMultiples(ny);
            extendIndexVec(xCacheLines, state);
            auto ivBase = accessIndexVec(0, state);

            shl<uint32_t>(xCacheLines, addr, ivBase(1), 6 - block.addrShift);
            if (hasRemX && remX.isValid()) {
                auto limit = state.ra.alloc_sub<uint32_t>();
                eaddScaled(1 | sat, limit, -setback, remX.uw(), T, strategy, state);
                min_<uint32_t>(xCacheLines, addr, addr, limit);
                state.ra.safeRelease(limit);
            } else
                min_<uint32_t>(xCacheLines, addr, addr, nx * T - setback);

            if (a64) {
                int n0 = std::min(n, elementsPerGRF<uint64_t>(hw));
                int n1 = std::min(n, elementsPerGRF<uint32_t>(hw));
                if (n > n1) {
                    mov<uint32_t>(n - n1, addr[2][0](2), (xCacheLines == n) ? addr[1][0](1)
                                                                            : addr[0][0](0, xCacheLines, 1));
                }
                if (n > n0)
                    mov<uint32_t>(n1 - n0, addr[1][0](2), addr[0][n0 % xCacheLines](0, xCacheLines, 1));
                mov<uint32_t>(n0, addr[0][0](2), addr[0][0](0, xCacheLines, 1));
                for (int y = 1; y < ny; y++) {
                    auto r = GRFRange(addr.getBase(), addr.getLen()).sub(hw, xCacheLines * y, DataType::uq);
                    auto ld = (y == 1) ? bld : findLDMultiple(ldMultiples, a64, y, strategy, state);
                    eadd(xCacheLines, r(1), r.ud(0)(2), ld, strategy, state);
                }
                map(hw, Type::u64, addr, addr, strategy, [&](int simd, GRF r, GRF _) {
                    eadd<uint64_t>(simd, r, r.ud(0)(2), ptr, strategy, state);
                });
            } else {
                auto ld0 = findLDMultiple(ldMultiples, a64, 0, strategy, state);
                add<uint32_t>(n, addr[0], addr[0][0](0, xCacheLines, 1), ld0(1, xCacheLines, 0));
                if (ptr.isValid())
                    add<uint32_t>(n, addr, addr, ptr);
            }

            if (allocLDMultiples)
                releaseLDMultiples(ldMultiples, state);

            break;
        }
    }
}

// Shift an address block by a combination of a fixed and LD offset.
template <HW hw>
void Generator<hw>::offsetAddr(const RegisterRange &addrDst, const RegisterRange &addrSrc,
                               const RegisterBlock &blockDst, const RegisterBlock &blockSrc,
                               int offsetFixed, int offsetLD, const Subregister &ld,
                               const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                               const CommonStrategy &strategy, CommonState &state, const LDMultiples &ldMultiples)
{
    bool a64 = (astrategy.base.getModel() == ModelA64);

    if (astrategy.address2D) stub();

    if (offsetLD == 0) {
        if (offsetFixed != 0)
            incAddr(addrDst, addrSrc, offsetFixed, blockDst, blockSrc, atype, astrategy, strategy, state);
    } else {
        // Reuse ld * offsetLD calculation if available.
        Subregister ldInc;
        ldInc = findLDMultiple(ldMultiples, a64, offsetLD, strategy, state);

        if (ldInc.isValid() && offsetFixed == 0)
            incAddr(addrDst, addrSrc, (offsetLD == 1) ? ld : ldInc, blockDst, blockSrc, atype, astrategy, strategy, state);
        else {
            Subregister incAlloc = state.ra.alloc_sub(a64 ? DataType::uq : DataType::ud);
            auto inc = incAlloc;

            if (ldInc.isInvalid()) {
                if (offsetLD == 1)
                    ldInc = ld;
                else {
                    emulConstant(1, inc, ld, offsetLD, strategy, state);
                    ldInc = inc;
                }
            }
            if (offsetFixed != 0)
                eadd(1, inc, ldInc, offsetFixed, strategy, state);
            else
                inc = ldInc;
            incAddr(addrDst, addrSrc, inc, blockDst, blockSrc, atype, astrategy, strategy, state);

            state.ra.safeRelease(incAlloc);
        }
    }
}

// Output code for initializing address/header GRFs for one block based on another block's headers.
template <HW hw>
void Generator<hw>::setupAddrRel(const RegisterRange &addrDst, const RegisterRange &addrSrc,
                                 const RegisterBlock &blockDst, const RegisterBlock &blockSrc, const RegisterLayout &layout,
                                 const Subregister &ld, const CommonStrategy &strategy, CommonState &state, const LDMultiples &ldMultiples)
{
    auto T = layout.type();
    auto &atype = layout.addressing();
    auto &astrategy = layout.addressingStrategy();

    int deltaR = blockDst.offsetR - blockSrc.offsetR;
    int deltaC = blockDst.offsetC - blockSrc.offsetC;

    if (blockDst.offsetAddr) return;       // nothing to do

    if (astrategy.address2D)
        incAddr(addrDst, addrSrc, Subregister(), deltaR, deltaC, blockDst, blockSrc, atype, astrategy, strategy, state);
    else {
        int offsetFixed = 0, offsetLD = 0;

        switch (atype.layout) {
            case MatrixLayout::N:  offsetFixed = deltaR; offsetLD = deltaC; break;
            case MatrixLayout::T:  offsetFixed = deltaC; offsetLD = deltaR; break;
            case MatrixLayout::Pc:
            case MatrixLayout::Pr:
                offsetFixed = untile(T, atype, blockDst) - untile(T, atype, blockSrc);
                offsetLD = align_down((atype.layout == MatrixLayout::Pc) ? deltaR : deltaC, atype.packSize);
                break;
        }

        offsetFixed *= T;

        offsetAddr(addrDst, addrSrc, blockDst, blockSrc, offsetFixed, offsetLD, ld, atype, astrategy, strategy, state, ldMultiples);
    }

    if (isBlock2D(astrategy.accessType)) {
        GRF addr = GRF(addrDst.getBase());
        updateBlock2DSizes(addr, blockDst, blockSrc, atype, astrategy.prefetch);
    }
}

static inline int findBaseBlock(const RegisterBlock &block, const RegisterLayout &layout, int start, int end,
                                const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    int bbase = -1;
    for (int bb = start; bb < end; bb++) {
        auto &other = layout[bb];
        if (canRelAddr(other, block, atype, astrategy)) {
            if (bbase < 0)
                bbase = bb;
            bool bestFit = (other.offsetR == block.offsetR || other.offsetC == block.offsetC);
            if (bestFit)
                return bb;
        }
    }
    return bbase;
}

// Output code for initializing address/header GRFs for an entire register layout.
//  ptr is an integer, Subregister, or MultishiftSubregister holding the base pointer/offset.
template <HW hw>
template <typename BO>
void Generator<hw>::setupAddr(const vector<RegisterRange> &addr, const BO &ptr, const RegisterLayout &layout, const Subregister &ld,
                              const CommonStrategy &strategy, CommonState &state,
                              const Address2DParams &params, const LDMultiples &ldMultiples, int start)
{
    auto T = layout.type();
    auto &atype = layout.addressing();
    auto &astrategy = layout.addressingStrategy();

    for (int b = start; b < layout.blocks(); b++) {
        auto &block = layout[b];

        // Skip non-load blocks.
        if (!block.isLoadBlock())
            continue;

        // Skip offset blocks.
        if (block.offsetAddr != 0)
            continue;

        auto bparams = params;
        Subregister tempRem;
        if (isBlock2D(astrategy.accessType) && !astrategy.address2D) {
            tempRem = state.ra.alloc_sub<uint32_t>();
            if (bparams.remR.isValid())                  bparams.remR = tempRem.uw(0);
            if (bparams.remC.isValid())                  bparams.remC = tempRem.uw(1);
            if (bparams.remR.isValid() && block.offsetR) add(1 | sat, bparams.remR, params.remR, -block.offsetR);
            if (bparams.remC.isValid() && block.offsetC) add(1 | sat, bparams.remC, params.remC, -block.offsetC);
            if (bparams.remR.isValid())                  min_(1, bparams.remR, block.offsetR ? bparams.remR : params.remR, block.nr);
            if (bparams.remC.isValid())                  min_(1, bparams.remC, block.offsetC ? bparams.remC : params.remC, block.nc);
        }
        // Look for a block to base this one off of.
        int bbase = findBaseBlock(block, layout, 0, b, atype, astrategy);

        if (bbase < 0) {
            // No base address, set up a new base address.
            setupAddr(T, addr[b], ptr, block, ld, atype, astrategy, strategy, state, bparams, ldMultiples);
            state.ra.safeRelease(tempRem);
        }

        // Increment as appropriate.
        if (bbase >= 0) {
            setupAddrRel(addr[b], addr[bbase], block, layout[bbase], layout, ld, strategy, state, ldMultiples);
        } else if (!astrategy.address2D) {
            int offsetFixed = 0, offsetLD = 0;
            switch (atype.layout) {
                case MatrixLayout::N:  offsetFixed = block.offsetR; offsetLD = block.offsetC; break;
                case MatrixLayout::T:  offsetFixed = block.offsetC; offsetLD = block.offsetR; break;
                case MatrixLayout::Pc: offsetFixed = untile(T, atype, block); offsetLD = align_down(block.offsetR, atype.packSize); break;
                case MatrixLayout::Pr: offsetFixed = untile(T, atype, block); offsetLD = align_down(block.offsetC, atype.packSize); break;
            }

            offsetFixed *= T;

            offsetAddr(addr[b], addr[b], block, block, offsetFixed, offsetLD, ld, atype, astrategy, strategy, state, ldMultiples);
        }
    }
}


// Output code for incrementing the pointers for a given block by a specified # of bytes.
// The amount may be an immediate, Subregister, or MultishiftSubregister.
template <HW hw>
template <typename I, typename Ir, typename Ic>
void Generator<hw>::incAddr(const RegisterRange &addrDst, const RegisterRange &addrSrc, I inc, Ir incR, Ic incC,
                            const RegisterBlock &blockDst, const RegisterBlock &blockSrc,
                            const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                            const CommonStrategy &strategy, CommonState &state)
{
    auto incShifted = startShift(inc, blockDst.addrShift, state);

    incAddrShifted(addrDst, addrSrc, incShifted, incR, incC, blockDst, blockSrc, atype, astrategy, strategy, state);

    doneShift(inc, incShifted, blockDst.addrShift, state);
}

template <HW hw>
template <typename I>
void Generator<hw>::incAddr(const RegisterRange &addrDst, const RegisterRange &addrSrc, I inc,
                            const RegisterBlock &blockDst, const RegisterBlock &blockSrc,
                            const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                            const CommonStrategy &strategy, CommonState &state)
{
    if (astrategy.address2D) stub();
    incAddr(addrDst, addrSrc, inc, Subregister(), Subregister(), blockDst, blockSrc, atype, astrategy, strategy, state);
}

template <HW hw>
template <typename I, typename Ir, typename Ic>
void Generator<hw>::incAddrShifted(const RegisterRange &addrDst, const RegisterRange &addrSrc, I inc, Ir incR, Ic incC,
                                   const RegisterBlock &blockDst, const RegisterBlock &blockSrc,
                                   const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                   const CommonStrategy &strategy, CommonState &state)
{
    // Handle non-load blocks.
    if (!blockDst.isLoadBlock())
        return;
    if (!blockSrc.isLoadBlock())
        stub();

    // Skip offset blocks.
    if (blockDst.offsetAddr != 0)
        return;

    if (blockDst.addrShift != blockSrc.addrShift)
        stub();

    auto cinc  = avoidConflict(hw, inc, addrSrc[0]);
    auto cincR = avoidConflict(hw, incR, addrSrc[0]);
    auto cincC = avoidConflict(hw, incC, addrSrc[0]);

    switch (blockSrc.effectiveAccessType(atype, astrategy)) {
        case AccessType::PseudoBlock:
            if (blockSrc.ebytes != blockDst.ebytes) stub();
            // fall through
        case AccessType::ChannelScattered:
        case AccessType::Scattered:
        case AccessType::CacheLine: {
            int naddrDst = blockDst.simdSize;
            int naddrSrc = blockSrc.simdSize;
            if (naddrDst > naddrSrc) stub();
            if (astrategy.base.getModel() == ModelA64) {
                auto simd = 2 * elementsPerGRF(hw, Type::u64);
                for (int ar = 0; naddrDst > 0; ar += 2, naddrDst -= simd)
                    eadd<uint64_t>(std::min(naddrDst, simd), addrDst[ar], addrSrc[ar], avoidConflict(hw, inc, addrSrc[ar]), strategy, state);
            } else
                add<uint32_t>(naddrDst, addrDst[0], addrSrc[0], cinc);
            break;
        }
        case AccessType::Block:
            if (astrategy.base.getModel() == ModelA64) {
                eadd(1, addrDst[0].uq(0), addrSrc[0].uq(0), cinc, strategy, state);
            } else if (astrategy.newDP) {
                add(1, addrDst[0].ud(0), addrSrc[0].ud(0), cinc);
            } else
                add(1, addrDst[0].ud(2), addrSrc[0].ud(2), cinc);
            break;
        case AccessType::Block2D:
        case AccessType::Block2DTranspose:
        case AccessType::Block2DVNNI:
            if (addrDst != addrSrc)
                mov<uint32_t>(8, addrDst[0], addrSrc[0]);
            if (astrategy.address2D) {
                // Prefetches and u3 (non-power-of-2 bits, rarely divides
                // evenly) don't need exactness; allow conservative round-up.
                bool exact = !astrategy.prefetch && is_zero_or_pow2(int(blockDst.extra));
                if (isColMajor(atype.layout)) {
                    if (cincR != 0) {
                        if (blockDst.extra == 3 && blockDst.offsetR != blockSrc.offsetR)
                            add(1, addrDst[0].d(5), addrDst[0].d(5),
                                    div_up(blockDst.offsetR * 3, blockDst.ebytes * 8)
                                    - div_up(blockSrc.offsetR * 3, blockDst.ebytes * 8));
                        else
                            addScaled(1, addrDst[0].d(5), addrDst[0].d(5), cincR, blockDst.extra, blockDst.ebytes * 8, state, exact);
                    }
                    if (cincC != 0) add(1, addrDst[0].d(6), addrDst[0].d(6), cincC);
                } else {
                    if (cincC != 0) {
                        if (blockDst.extra == 3 && blockDst.offsetC != blockSrc.offsetC)
                            add(1, addrDst[0].d(5), addrDst[0].d(5),
                                    div_up(blockDst.offsetC * 3, blockDst.ebytes * 8)
                                    - div_up(blockSrc.offsetC * 3, blockDst.ebytes * 8));
                        else
                            addScaled(1, addrDst[0].d(5), addrDst[0].d(5), cincC, blockDst.extra, blockDst.ebytes * 8, state, exact);
                    }
                    if (cincR != 0) add(1, addrDst[0].d(6), addrDst[0].d(6), cincR);
                }
            } else
                eadd(1, addrDst[0].uq(0), addrSrc[0].uq(0), cinc, strategy, state);
            break;
    }
}

// Output code for incrementing all pointers for a register layout by a specified # of bytes.
// The amount may be an immediate or a subregister.
template <HW hw>
template <typename I, typename Ir, typename Ic>
void Generator<hw>::incAddr(const vector<RegisterRange> &addr, I inc, Ir incR, Ic incC, const RegisterLayout &layout,
                            const CommonStrategy &strategy, CommonState &state)
{
    for (int b = 0; b < layout.blocks(); b++) {
        incAddr(addr[b], addr[b], inc, incR, incC, layout[b], layout[b],
                layout.addressing(), layout.addressingStrategy(), strategy, state);
    }
}

template <HW hw>
template <typename I>
void Generator<hw>::incAddr(const vector<RegisterRange> &addr, I inc, const RegisterLayout &layout,
                            const CommonStrategy &strategy, CommonState &state)
{
    if (layout.addressingStrategy().address2D) stub();
    incAddr(addr, inc, Subregister(), Subregister(), layout, strategy, state);
}

template <HW hw>
template <typename I, typename Ir, typename Ic>
void Generator<hw>::incAddrShifted(const vector<RegisterRange> &addr, I inc, Ir incR, Ic incC, const RegisterLayout &layout,
                                   const CommonStrategy &strategy, CommonState &state)
{
    for (int b = 0; b < layout.blocks(); b++) {
        incAddrShifted(addr[b], addr[b], inc, incR, incC, layout[b], layout[b],
                       layout.addressing(), layout.addressingStrategy(), strategy, state);
    }
}

template <HW hw>
template <typename I>
void Generator<hw>::incAddrShifted(const vector<RegisterRange> &addr, I inc, const RegisterLayout &layout,
                                   const CommonStrategy &strategy, CommonState &state)
{
    if (layout.addressingStrategy().address2D) stub();
    incAddrShifted(addr, inc, Subregister(), Subregister(), layout, strategy, state);
}

template <typename T> struct NegativeType    { using type = T;       };
template <> struct NegativeType<uint8_t>     { using type = int8_t;  };
template <> struct NegativeType<uint16_t>    { using type = int16_t; };
template <> struct NegativeType<uint32_t>    { using type = int32_t; };
template <> struct NegativeType<int>         { using type = int32_t; };
template <> struct NegativeType<int64_t>     { using type = int32_t; };

// Output code for incrementing or decrementing all pointers for a register layout by a specified # of bytes.
// The amount may be an immediate or a MultishiftSubregister.
template <HW hw>
template <typename A, typename I, typename Ir, typename Ic>
void Generator<hw>::incDecAddr(const A &addr, I inc, Ir incR, Ic incC, const RegisterLayout &layout,
                               const CommonStrategy &strategy, CommonState &state, bool decrement)
{
    typename NegativeType<I>::type  signedInc  = decrement ? -inc  : inc;
    typename NegativeType<Ir>::type signedIncR = decrement ? -incR : incR;
    typename NegativeType<Ic>::type signedIncC = decrement ? -incC : incC;

    incAddr(addr, signedInc, signedIncR, signedIncC, layout, strategy, state);
}

template <HW hw>
template <typename A, typename I>
void Generator<hw>::incDecAddr(const A &addr, I inc, const RegisterLayout &layout,
                               const CommonStrategy &strategy, CommonState &state, bool decrement)
{
    if (layout.addressingStrategy().address2D) stub();
    incDecAddr(addr, inc, Subregister(), Subregister(), layout, strategy, state, decrement);
}

template <HW hw>
void Generator<hw>::incAddrK(const vector<RegisterRange> &addr, bool column, int k,
                             const SubregisterPair &ld, const LDIncrements &incs, const RegisterLayout &layout,
                             const CommonStrategy &strategy, CommonState &state)
{
    if (isColMajor(layout.addressing().layout) == column) {
        bool release = false;
        auto inc = lookupIncrement(incs, ld, k, strategy, state, &release);
        incAddr(addr, inc, layout, strategy, state);
        if (release) state.ra.safeRelease(inc);
    } else
        incAddr(addr, k * layout.type(), layout, strategy, state);
}

template <HW hw>
void Generator<hw>::setAddrRemainder(Type T, const RegisterRange &addr, const RegisterBlock &block, const Subregister &remR, const Subregister &remC,
                                     const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                     const CommonStrategy &strategy, CommonState &state)
{
    if (!isBlock2D(astrategy.accessType) || astrategy.address2D) return;

    auto tempRem = state.ra.alloc_sub<uint32_t>();
    Subregister thisRemR = remR, thisRemC = remC;

    auto memCM = isColMajor(atype.layout);
    auto &remW = memCM ? thisRemR : thisRemC;
    auto &remH = memCM ? thisRemC : thisRemR;
    int bw, bh, bcount, multiX;
    block.getBlock2DWH(bw, bh, bcount, atype, astrategy.prefetch, &multiX);

    if (!block.remainderR)                   thisRemR.invalidate();
    if (!block.remainderC)                   thisRemC.invalidate();
    if (thisRemR.isValid())                  thisRemR = tempRem.uw(0);
    if (thisRemC.isValid())                  thisRemC = tempRem.uw(1);
    if (thisRemR.isValid() && block.offsetR) add(1 | sat, thisRemR, remR, -block.offsetR);
    if (thisRemC.isValid() && block.offsetC) add(1 | sat, thisRemC, remC, -block.offsetC);
    if (thisRemR.isValid())                  min_(1, thisRemR, block.offsetR ? thisRemR : remR, block.nr);
    if (thisRemC.isValid())                  min_(1, thisRemC, block.offsetC ? thisRemC : remC, block.nc);

    if (remW.isValid()) {
        if (block.count > 1 || multiX > 1) stub();
        addScaled(1, addr[0].ud(2), -1, remW.uw(), T, state, true);
    }
    if (remH.isValid())
        addScaled(1, addr[0].ud(3), -1, remH.uw(), T, state,
                (T.bits() * multiX) % 8 == 0, multiX);
    if (remW.isValid() && T.paddedSize() < 4)
        or_(1, addr[0].ud(2), addr[0].ud(2), 3);

    state.ra.safeRelease(tempRem);
}

template <HW hw>
void Generator<hw>::setAddrRemainder(const vector<RegisterRange> &addr, const RegisterLayout &layout,
                                     const Subregister &remR, const Subregister &remC,
                                     const CommonStrategy &strategy, CommonState &state)
{
    for (int b = 0; b < layout.blocks(); b++)
        setAddrRemainder(layout.type(), addr[b], layout[b], remR, remC, layout.addressing(), layout.addressingStrategy(), strategy, state);
}

GEMMSTONE_NAMESPACE_END
