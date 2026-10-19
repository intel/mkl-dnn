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
#include "hw_utils.hpp"
#include "layout_utils.hpp"
#include "map.hpp"
#include "ngen_object_helpers.hpp"
#include "quantization.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


// Prepare 2D dequantization layouts.
template <HW hw>
bool Generator<hw>::gemmMake2DQuantizationLayouts(bool isA, const GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto lateOffset = isA ? problem.needsBGroupSums() : problem.needsAGroupSums();
    int xoPtrDims = (isA ? problem.aoPtrDims : problem.boPtrDims);
    bool xo2D = isA ? problem.aOffset2D()       : problem.bOffset2D();
    bool xs2D = isA ? problem.aScale2D()        : problem.bScale2D();
    bool xg2D = isA ? problem.needsAGroupSums() : problem.needsBGroupSums();
    bool xoTo2D = !xo2D && (isA ? problem.aOffset == ABOffset::Calc && (problem.earlyDequantizeA() || lateOffset)
                                : problem.bOffset == ABOffset::Calc && (problem.earlyDequantizeB() || lateOffset));
    bool cColMajor = isRegisterColMajor(problem.Tc_ext, problem.C, strategy.C);

    if (!xo2D && !xoTo2D && !xs2D && !xg2D) return true;

    if (state.useBDPAS && (xo2D || xoTo2D || xg2D))
        stub("BDPAS with 2D offsets or group sums not supported");

    auto &X_strategy       = isA ? strategy.A             : strategy.B;
    auto &X_offsetStrategy = isA ? strategy.AO            : strategy.BO;
    auto &X_scaleStrategy  = isA ? strategy.A_scale       : strategy.B_scale;
    auto &Xg_strategy      = isA ? strategy.Ag            : strategy.Bg;
    auto &X_offsetLayout   = isA ? state.A_offsetLayout   : state.B_offsetLayout;
    auto &X_scaleLayout    = isA ? state.A_scaleLayout    : state.B_scaleLayout;
    auto &Xg_layout        = isA ? state.Ag_layout        : state.Bg_layout;
    auto &Xr_offsetLayout  = isA ? state.Ar_offsetLayout  : state.Br_offsetLayout;
    auto &Xr_scaleLayout   = isA ? state.Ar_scaleLayout   : state.Br_scaleLayout;
    auto &Xgr_layout       = isA ? state.Agr_layout       : state.Bgr_layout;

    auto &XO         = isA ? problem.AO         : problem.BO;
    auto &XS         = isA ? problem.A_scale    : problem.B_scale;
    auto &Xg         = isA ? problem.Ag         : problem.Bg;
    auto Tx_ext      = isA ? problem.Ta_ext     : problem.Tb_ext;
    auto Tx          = isA ? problem.Ta         : problem.Tb;
    auto Txo         = isA ? problem.Tao        : problem.Tbo;
    auto Txs         = isA ? problem.Ta_scale   : problem.Tb_scale;
    auto Txg         = isA ? problem.Tag        : problem.Tbg;
    auto xqGroupK    = isA ? problem.aqGroupK   : problem.bqGroupK;
    auto xqGroupMN   = isA ? problem.aqGroupM   : problem.bqGroupN;
    auto &Txo_int    = isA ? state.Tao_int      : state.Tbo_int;
    auto &Txs_int    = isA ? state.Ta_scaleInt  : state.Tb_scaleInt;
    auto &Txg_int    = isA ? state.Tag_int      : state.Tbg_int;
    auto &lateScale  = isA ? state.lateScale2DA : state.lateScale2DB;

    auto wgTileMN = strategy.wgTile(isA ? LoopM : LoopN);

    // Offsets applied after the scales (x = q * scale - offset) are fused with the scales into one mad.
    // bf16 data uses f32 quantization data (mixed mode), except on Xe3p+ which has pure bf16 mad.
    bool offsetAfterScale = isA ? problem.aOffsetAfterScale : problem.bOffsetAfterScale;
    bool pureBf16 = (hw >= HW::Xe3p) && offsetAfterScale && Tx == Type::bf16 && Txs == Tx && Txo == Tx;

    Txo_int = Txo.isInteger() ? Tx.asSignedInt() : Tx;
    Txs_int = Tx;
    if (Tx == Type::bf16 && !pureBf16)
        Txs_int = Type::f32;
    Txg_int = Txg;

    int cpoDiv = 1;
    if (Txo_int.isInt8()) Txo_int = Type::s16, cpoDiv = 2;
    // Use lateScale for cases of applying scale to inputs that will be natively dpas'd
    // but do not support add/mul.
    if (xs2D && usesLateScale(problem, strategy, isA)) {
        lateScale = true;
        Txs_int = problem.Tc;
    }

    bool subByteIntSpecialPath = Tx_ext.isSubByteInt() && one_of(Tx, {Type::f16, Type::f32});
    if (subByteIntSpecialPath) {
        Txo_int = Type::f16;
        Txs_int = Tx;
        if (Tx == Type::bf16) Txs_int = Type::f16;
    }

    if (offsetAfterScale)
        Txo_int = Txs_int;

    if (lateOffset && (Txo.isSubByteInt() || Txo.isInt8()))
        Txo_int = Type::s32;

    if (Txs == Type::f8_e8m0 && state.useBDPAS)
	Txs_int = Type::f8_e8m0;
    // Get tile sizes, depending on whether A/B are copied to SLM.
    // For late scaling (after compute), scales are always applied to the whole tile.
    int r, c, k, rNoSLM, cNoSLM;
    int tileR = 0, tileC = 0;
    bool remR = false, remC = false;

    // K-elements processed per BDPAS pass.
    int bBlockK = state.useBDPAS ? bdpasBlockK(problem) : 0;
    if (isA) {
        bool slmA = strategy.slmA;
        rNoSLM = strategy.unroll[LoopM];
        cNoSLM = strategy.ka_load;
        r = slmA ? state.ma_slm : rNoSLM;
        c = slmA ? state.ka_slm : cNoSLM;
        k = slmA ? strategy.unrollKSLM : cNoSLM;
        r = std::max(1, r / xqGroupMN);
        c = state.kaq = std::max(1, c % xqGroupK == 0 ? c / xqGroupK : 1);
        state.kaqStride = std::max(1, k % xqGroupK == 0 ? k /  xqGroupK : 1);
        rNoSLM = std::max(1, rNoSLM / xqGroupMN);
        cNoSLM = state.kaqLate = std::max(1, cNoSLM % xqGroupK == 0 ? cNoSLM /  xqGroupK: 1);
        remR = (strategy.remHandling[LoopM] != RemainderHandling::Ignore);
        if (state.useBDPAS) {
            tileR = bdpasTileMN;
            tileC = (xqGroupK > bBlockK) ? 1 : state.kaq;
        } else if (xqGroupMN <= 1 && xqGroupK > 1) {
            tileC = 1;
        }
        if (xqGroupMN > 1 && (xqGroupMN % strategy.unroll[LoopM] && strategy.unroll[LoopM] % xqGroupMN))
            stub("Tile size not compatible with group size in m dimension");
    } else {
        bool slmB = strategy.slmB;
        cNoSLM = strategy.unroll[LoopN];
        rNoSLM = strategy.kb_load;
        c = slmB ? state.nb_slm : cNoSLM;
        r = slmB ? state.kb_slm : rNoSLM;
        k = slmB ? strategy.unrollKSLM : rNoSLM;
        c = std::max(1, c / xqGroupMN);
        r = state.kbq = std::max(1, r % xqGroupK == 0 ? r / xqGroupK : 1);
        state.kbqStride = std::max(1, k % xqGroupK == 0 ?  k / xqGroupK : 1);
        cNoSLM = std::max(1, cNoSLM / xqGroupMN);
        rNoSLM = state.kbqLate = std::max(1, rNoSLM % xqGroupK == 0 ?  rNoSLM / xqGroupK : 1);
        remC = (strategy.remHandling[LoopN] != RemainderHandling::Ignore);
        if (state.useBDPAS) {
            tileR = (xqGroupK > bBlockK) ? 1 : state.kbq;
            tileC = bdpasTileMN;
        } else if (xqGroupMN <= 1 && xqGroupK > 1) {
            tileR = 1;
        }
        if (xqGroupMN > 1 && (xqGroupMN % strategy.unroll[LoopN] && strategy.unroll[LoopN] % xqGroupMN))
            stub("Tile size not compatible with group size in n dimension");
    }

    int ro = lateOffset ? rNoSLM : r;
    int co = lateOffset ? cNoSLM : c;
    int rs = lateScale  ? rNoSLM : r;
    int cs = lateScale  ? cNoSLM : c;

    if (X_strategy.padded) {
        X_offsetStrategy.padded = X_scaleStrategy.padded = Xg_strategy.padded = true;
        remR = remC = false;
    }

    bool wantCM = state.useBDPAS ? isA : isA ^ (xqGroupMN > 1);
    auto chooseAccess = [=](const MatrixAddressing &Xq) {
        return (wantCM == isColMajor(Xq.layout)) ? AccessType::Block : AccessType::Scattered;
    };

    X_offsetStrategy.accessType = chooseAccess(XO);
    X_scaleStrategy.accessType  = chooseAccess(XS);
    Xg_strategy.accessType      = chooseAccess(Xg);

    if (xo2D && !(X_offsetLayout = RegisterLayout::tryCreate(hw, Txo, ro,     co,     XO, X_offsetStrategy, remR, remC))) return false;
    if (xs2D &&  !(X_scaleLayout = RegisterLayout::tryCreate(hw, Txs, rs,     cs,     XS, X_scaleStrategy,  remR, remC))) return false;
    if (xg2D &&      !(Xg_layout = RegisterLayout::tryCreate(hw, Txg, rNoSLM, cNoSLM, Xg, Xg_strategy,      remR, remC))) return false;

    if (strategy.pfaux && (isA ? strategy.prefetchA : strategy.prefetchB)) {
        auto &Xp_offsetLayout = isA ? state.Ap_offsetLayout : state.Bp_offsetLayout;
        auto &Xp_scaleLayout  = isA ? state.Ap_scaleLayout  : state.Bp_scaleLayout;
        auto &Xgp_layout      = isA ? state.Agp_layout      : state.Bgp_layout;
        int pfChunk = isA ? strategy.ka_prefetch : strategy.kb_prefetch;

        int xqPfGroups = 1;
        if (isA ? problem.quantized2DA() : problem.quantized2DB()) {
            if (xqGroupK <= 0 || pfChunk % xqGroupK != 0)
                stub("aux prefetch requires ka_prefetch/kb_prefetch to be a multiple of aqGroupK/bqGroupK (see GEMMStrategy::preflight)");
            xqPfGroups = pfChunk / xqGroupK;
        }

        int rPFo = isA ? rNoSLM : xqPfGroups;
        int cPFo = isA ? xqPfGroups : cNoSLM;
        int rPFs = isA ? rNoSLM : xqPfGroups;
        int cPFs = isA ? xqPfGroups : cNoSLM;
        int rPFg = isA ? rNoSLM : xqPfGroups;
        int cPFg = isA ? xqPfGroups : cNoSLM;

        if (xo2D && !(Xp_offsetLayout = RegisterLayout::tryCreate(hw, Txo, rPFo, cPFo, XO, X_offsetStrategy, remR, remC))) return false;
        if (xs2D && !(Xp_scaleLayout  = RegisterLayout::tryCreate(hw, Txs, rPFs, cPFs, XS, X_scaleStrategy,  remR, remC))) return false;
        if (xg2D && !(Xgp_layout      = RegisterLayout::tryCreate(hw, Txg, rPFg, cPFg, Xg, Xg_strategy,      remR, remC))) return false;
    }

    // Adjust masks for m/n grouping.
    auto adjustMask = [=](MaskInfo &mask) {
        if (!mask || xqGroupMN <= 1) return;
        if (!is_zero_or_pow2(xqGroupMN)) {
            if (xqGroupMN < wgTileMN) stub();
            return;
        }
        if (mask.fixed.isFixed) stub();
        // Load remainder index is in ungrouped units, so divide it (rshift) and
        // rescale the span (rsize) to keep the mask-bit count rsize>>rshift.
        int shift = ilog2(xqGroupMN);
        mask.variable.rshift += shift;
        // rsize is only 8 bits wide. When xqGroupMN covers (or exceeds) the whole
        // dimension leave rsize untouched (it's already the degenerate 
        // single-group value).
        if (state.useBDPAS && (int(mask.variable.rsize) << shift) <= 0xFF)
            mask.variable.rsize <<= shift;
    };

    for (auto *Xq_layout: {&X_offsetLayout, &X_scaleLayout, &Xg_layout}) {
        for (auto &block: *Xq_layout) {
            adjustMask(block.rowMask);
            adjustMask(block.colMask);
        }
    }

    // Quantization parameters will be upconverted to the size of A/B and duplicated to match crosspack.
    auto &lsrc = isA ? (strategy.slmA ? state.Ao_layout : !state.Ar_layout.empty() ? state.Ar_layout : state.A_layout)
                     : (strategy.slmB ? state.Bo_layout : !state.Br_layout.empty() ? state.Br_layout : state.B_layout);
    if (lsrc.empty()) stub();
    int crosspack = lsrc[0].crosspack;
    if (xqGroupMN > 1)
        crosspack = 1;

    int cpo = lateOffset ? 1 : div_up(crosspack, cpoDiv);
    int cps = lateScale  ? 1 : crosspack;

    // Xe3p+ FP ops need non-scalar sources at the same sub-GRF offset as the destination. If k-rows
    // of A/B share a GRF, keep one copy of the per-k scale/offset vector per row position within the GRF.
    auto &qCopies = isA ? state.aqCopies : state.bqCopies;
    qCopies = 1;
    int kRow = 0;
    if (hw >= HW::Xe3p && Tx.isFP() && xqGroupMN > 1 && xqGroupK == 1 && !state.useBDPAS && lsrc[0].crosspack == 1
            && lsrc[0].colMajor != isA /* k contiguous */) {
        kRow = isA ? lsrc[0].nc : lsrc[0].nr;
        int kq = isA ? c : r;
        int epg = elementsPerGRF(hw, Tx);
        if (kq % kRow == 0 && kRow < epg && epg % kRow == 0)
            qCopies = epg / kRow;
    }

    auto makeQRepack = [&, tileR, tileC](Type Txq, Type Txq_int, RegisterLayout &repack, const RegisterLayout &src,
                                         int m, int n, int cp, bool forceRepack, bool allowBcast) {
        // Broadcast along M/N: BDPAS reads one scale per output lane; Xe3p+ keeps qCopies per k-row block,
        // packed as [group][copy] so each GRF holds every copy of one block.
        int tR = tileR, tC = tileC;
        int bcast = 1;
        bool allowPartialRegs = false;
        auto &mn = isA ? m : n;
        if (allowBcast && state.useBDPAS && xqGroupMN > 1)
            bcast = strategy.unroll[isA ? LoopM : LoopN] / mn;
        else if (allowBcast && qCopies > 1 && Txq_int == Tx) {
            bcast = qCopies;
            (isA ? tR : tC) = qCopies;
            (isA ? tC : tR) = kRow;
            allowPartialRegs = true;
        }
        mn *= bcast;
        if (cp > 1 || (cColMajor && (cp != src[0].crosspack)) || Txq != Txq_int || forceRepack || bcast > 1) {
            // Each BDPAS pass reads scales from one half-GRF. When xqGroupK == bdpasBlockK,
            // each group maps to one pass and scales pack naturally as [group0, group1].
            // When xqGroupK > bdpasBlockK, both passes share the same group and scales
            // must be duplicated to the upper half-GRF: [group0, group0].
            allowPartialRegs |= state.useBDPAS && xqGroupK == bBlockK;
            if (state.useBDPAS && xqGroupK > bBlockK) {
                // Do not support group sizes that aren't multiples of ksys - this case produces mixed
                // [group0, group0] and [group0, group1] scales args, which is not currently supported.
                int ksys = systolicParams(problem).ksys;
                if (xqGroupK % ksys != 0)
                    stub("BDPAS scale group straddling not supported");
            }
            repack = RegisterLayout(hw, Txq_int, m, n, wantCM, cp, tR, tC, allowPartialRegs);
        }
    };

    if (xo2D) makeQRepack(Txo, Txo_int, Xr_offsetLayout, X_offsetLayout, ro,     co,     cpo, false,     !lateOffset && !state.useBDPAS);
    if (xs2D) makeQRepack(Txs, Txs_int, Xr_scaleLayout,  X_scaleLayout,  rs,     cs,     cps, lateScale, !lateScale || state.useBDPAS);
    if (xg2D) makeQRepack(Txg, Txg_int, Xgr_layout,      Xg_layout,      rNoSLM, cNoSLM, 1,   true,      false);

    if (xoTo2D) {
        if (xoPtrDims <= 0)
            Xr_offsetLayout = RegisterLayout(hw, Txo_int, 1, 1, isA);
        else if (xoPtrDims == 1)
            Xr_offsetLayout = RegisterLayout(hw, Txo_int, ro, co, isA, cpo, tileR, tileC, false);
        else stub();
    }

    return true;
}

// Convert and repack 2D grouped quantization data in preparation for dequantization.
template <HW hw>
void Generator<hw>::gemmRepack2DQuantizationData(Type Ts, Type Td, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                                 const GRFMultirange &src, const GRFMultirange &dst,
                                                 const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    if (layoutDst.empty()) return;

    // Copy, broadcasting 1D to 2D data as needed. If the repacked layout is wider than the loaded one
    // along a single axis, each loaded value is broadcast to bcast consecutive lanes ([group][copy]).
    int bcastR = layoutDst.rows() / layoutSrc.rows(), bcastC = layoutDst.cols() / layoutSrc.cols();
    bool alongR = (bcastR > 1 && bcastC == 1 && layoutDst.rows() % layoutSrc.rows() == 0);
    bool alongC = (bcastC > 1 && bcastR == 1 && layoutDst.cols() % layoutSrc.cols() == 0);
    if (alongR || alongC) {
        int bcast = alongR ? bcastR : bcastC;
        int srcMN = alongR ? layoutSrc.rows() : layoutSrc.cols();
        // Compact copy: loaded value g lands in lane g. Then expand each lane over its bcast lanes,
        // high-to-low so lane g is read before a lower group's expansion overwrites it.
        copyRegisters(Ts, Td, layoutSrc, layoutDst, src, dst, 0, 0, false, strategy, state);
        for (int g = srcMN - 1; g >= 0; g--) {
            auto lane = layoutDst.slice(alongC, g, g + 1, false);
            for (int t = (g == 0 ? 1 : 0); t < bcast; t++) {
                int doff = g * (bcast - 1) + t;
                copyRegisters(Td, Td, lane, layoutDst, dst, dst, alongR ? doff : 0, alongC ? doff : 0, false, strategy, state);
            }
        }
    } else {
        for (int doffR = 0; doffR < layoutDst.rows(); doffR += layoutSrc.rows())
            for (int doffC = 0; doffC < layoutDst.cols(); doffC += layoutSrc.cols())
                copyRegisters(Ts, Td, layoutSrc, layoutDst, src, dst, doffR, doffC, false, strategy, state);
    }

    // BDPAS: duplicate scale data to the upper half of each GRF for the second pass.
    // When xqGroupK > bBlockK, both passes within a bdpas share the same scale group,
    // so the first half's data must be replicated to the second half.
    if (state.useBDPAS) {
        if (layoutDst[0].ld == elementsPerGRF(hw, Td)) {
            auto upperLayout = layoutDst;
            for (auto &block : upperLayout)
                block.offsetBytes += bdpasTileMN;
            copyRegisters(Td, Td, layoutDst, upperLayout, dst, dst, 0, 0, false, strategy, state);
        }
    }

    // Duplicate data in padded region. TODO: do this as part of the copy.
    int cp = layoutDst[0].crosspack;
    int p0 = layoutDst[0].colMajor ? layoutDst[0].nc : layoutDst[0].nr;

    if (cp > 1) map(hw, Td, dst, layoutDst, strategy, [&](int simd, RegData r) {
        Subregister r0 = GRF(r.getBase()).sub(r.getOffset(), r.getType());
        moveToIntPipe(r0);
        auto r1 = r0;
        for (int i = p0; i < cp; i++) {
            r1.setOffset(r1.getOffset() + 1);
            mov(simd / cp, r1(cp), r0(cp));
        }
    });
}

template <HW hw>
void Generator<hw>::gemmRepack2DQuantizationData(const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                                 const GRFMultirange &src, const GRFMultirange &dst,
                                                 const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state)
{
    gemmRepack2DQuantizationData(layoutSrc.type(), layoutDst.type(), layoutSrc, layoutDst,
                                 src, dst, problem, strategy, state);
}

template <HW hw>
void Generator<hw>::gemmRepack2DOffsetData(Type Text, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst,
                                           const GRFMultirange &src, const GRFMultirange &dst,
                                           const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                           bool offsetAfterScale)
{
    auto Ts = layoutSrc.type(), Td = layoutDst.type();

    bool signedInput = Text.isSigned();
    bool i4 = (Text.bits() == 4);
    bool s8 = (Ts == Type::s8);
    bool u8 = (Ts == Type::u8);

    // Offsets applied after the scales are subtracted from dequantized values as they are,
    // so they don't take part in the sub-byte integer rescaling below.
    bool subByteIntSpecialPath = Text.isSubByteInt() && Td == Type::f16 && !offsetAfterScale;
    auto tmpType = Td;

    if (subByteIntSpecialPath) {
        if (u8) tmpType = Type::u16;
        if (s8) tmpType = Type::s16;
    }

    gemmRepack2DQuantizationData(Ts, tmpType, layoutSrc, layoutDst, src, dst, problem, strategy, state);

    if (subByteIntSpecialPath) {
        if (s8 || u8) {
            int off = signedInput ? 1 << (Text.bits() - 1) : 0;

            // Shift s8 -> u8 data.
            if (s8) {
                map(hw, Type::s16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                    add(esize, r, r, 0x80);
                });
                off -= 0x80;
            }

            // Reinterpret as f16 and undo offsets.
            if (off != 0) map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                uint16_t offF16 = std::abs(off);
                if (off < 0) offF16 |= 0x8000;
                add(esize, r, r, Immediate::hf(offF16));
            });

            // Rescale into normal range. End result is 2^(-12) * intended offset.
            map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                mul(esize, r, r, Immediate::hf(0x6C00));
            });
        } else {
            map(hw, Type::f16, dst, dst, strategy, [&](int esize, RegData r, RegData _) {
                signedInput ? mad(esize, r, Immediate::hf(i4 ? 0x1800 : 0x1000), r, Immediate::hf(0x0C00))     // 0x1800 = 8 * 2^(-12)
                            : mul(esize, r,                                      r, Immediate::hf(0x0C00));    // 0x0C00 = 2^(-12)
            });
        }
    }
}

// Apply a single 2D group dequantize operation (scale/multiply).
template <HW hw>
void Generator<hw>::gemmDequantizeOperation(bool doA, Type T, Type Tq, BinaryOp op,
                                            const RegisterLayout &layout, const RegisterLayout &qlayout,
                                            const GRFMultirange &regs, const GRFMultirange &qregs,
                                            int h, int kab_load, int kq_load, const GEMMProblem &problem, const CommonStrategy &strategy, CommonState &state,
                                            const RegisterLayout *olayout, const GRFMultirange *oregs)
{
    // MulSub (x = x * scale - offset in one mad) takes the scales as the quantization data and the offsets,
    // which have the same groups and type, as a second operand.
    if ((op == BinaryOp::MulSub) != (olayout && oregs)) stub();

    int xqGroupK  = doA ? problem.aqGroupK : problem.bqGroupK;
    int xqGroupMN = doA ? problem.aqGroupM : problem.bqGroupN;
    int qCopies = doA ? state.aqCopies : state.bqCopies;

    bool common = (qlayout.rows() * qlayout.cols()) == 1;
    bool colMajor = layout.colMajor();
    int xqGroupX = colMajor == doA ? xqGroupMN : xqGroupK;
    int xqGroupY = colMajor == doA ? xqGroupK : xqGroupMN;

    bool bfSpecialPath = (T == Type::bf16) && (Tq == Type::f32) && (xqGroupMN > 1) && layout.hasFullCrosspack(2) && (xqGroupY == 1);
    GRFMultirange qPairs;
    int npairs = 0;

    if (bfSpecialPath) {
        for (npairs = 8; npairs > 0; npairs >>= 1) {
            qPairs = tryChunkAlloc(npairs, 1, BundleGroup::AllBundles(), BundleGroup::AllBundles(), state);
            if (!qPairs.empty()) break;
        }
        if (qPairs.empty()) throw out_of_registers_exception();
    }

    // MulSub on bf16 data: mixed mode with f32 quantization data (pre-Xe3p), otherwise pure bf16 mad (Xe3p+).
    bool bfMixed = (T == Type::bf16) && (Tq == Type::f32);
    if (op == BinaryOp::MulSub && (bfMixed ? hw >= HW::Xe3p : (T != Tq || !one_of(T, {Type::f16, Type::f32, Type::bf16}))))
        stub("Unsupported MulSub types");

    for (auto &block: layout) {
        auto crosspack = block.crosspack;
        int nx = colMajor ? block.nr : block.nc;
        int ny = colMajor ? block.nc : block.nr;

        // If crosspack spans multiple groups, use a stride to restrict to one group
        bool qbroadcastY = (xqGroupY % crosspack == 0) || common || bfSpecialPath;
        int strided = 1;
        if (!qbroadcastY) {
            strided = crosspack;
        }

        bool qbroadcastX = xqGroupX > 1 || common;
        int strideq = qbroadcastX ? 0 : 1;
        if (bfSpecialPath)
            strideq = 1;

        // Unpack iteration # and block offsets into MNK indices
        auto h_block = align_down(h, kab_load) + (doA ? block.offsetC : block.offsetR);
        auto mn_block = doA ? block.offsetR : block.offsetC;

        for(int y0 = 0; y0 < ny; y0 += qbroadcastY ? crosspack : 1) {
        for(int x0 = 0; x0 < nx; ) {
            auto ii0 = colMajor ? x0 : y0;
            auto jj0 = colMajor ? y0 : x0;

            auto h_final = h_block + (doA ? jj0 : ii0);
            auto mn_final = mn_block + (doA ? ii0 : jj0);
            auto mnq = mn_final / xqGroupMN;
            auto hq = (h_final % kq_load) / xqGroupK;

            auto io0 = doA ? mnq : hq;
            auto jo0 = doA ? hq : mnq;

            // Common scales always load the first element
            if (common) io0 = jo0 = 0;

            // Copied scales/offsets are indexed [group][copy] (see gemmMake2DQuantizationLayouts).
            bool bcastRep = qCopies > 1 && T.isFP() && T == Tq;
            if (bcastRep) (doA ? io0 : jo0) *= qCopies;

            int ne, neq;
            const RegisterBlock *qblock;
            auto data = block.find(T, ii0, jj0, regs, &ne);
            auto qdata = qlayout.find(io0, jo0, qregs, &neq, &qblock);

            // Select the copy matching the data's sub-GRF offset.
            if (bcastRep) {
                int epg = elementsPerGRF(hw, T);
                int kRow = epg / qCopies;
                int delta = (data.getOffset() - qdata.getOffset() + epg) % epg;
                if (delta % kRow == 0 && delta > 0) {
                    (doA ? io0 : jo0) += delta / kRow;
                    qdata = qlayout.find(io0, jo0, qregs, &neq, &qblock);
                }
            }

            if (!qbroadcastX) ne = std::min(ne, neq);

            Subregister odata;
            if (op == BinaryOp::MulSub) {
                int neo;
                odata = olayout->find(io0, jo0, *oregs, &neo);
                if (!qbroadcastX) ne = std::min(ne, neo);
            }

            // If a group lies along the X-direction, limit ne to the end of the current group
            if (xqGroupX > 1) ne = std::min(ne, xqGroupX - x0 % xqGroupX);

            // Broadcast pairs of scales, in case of stride-2 bfloat16 data.
            if (bfSpecialPath) {
                int npair = (qdata.getOffset() / 2) % npairs;
                if (x0 == 0 && npair == 0)
                    for (int p = 0; p < npairs; p++)
                        mov(elementsPerGRF(hw, Tq), qPairs[p].ud(), qdata.ud(2*p)(0,2,1));
                qdata = qPairs[npair].sub(0, Tq.ngen());
            }

            int maxSIMD = (op == BinaryOp::Sub && T.isInt8()) ? 64 : 32;
            if (Tq == Type::f32) maxSIMD = elementsPerGRF(hw, Tq);
            int simd = std::min({ne * crosspack / strided, 2 * elementsPerGRF(hw, T) / strided, maxSIMD});

            switch (op) {
                case BinaryOp::Sub:
                    if (T.isInt8() && strided == 1) {
                        add(simd / 2, data(2), data(2), -qdata(strideq * 2 / Tq));
                        data.setOffset(data.getOffset() + 1);
                        qdata.setOffset(qdata.getOffset() + strideq / Tq);
                        add(simd / 2, data(2), data(2), -qdata(strideq * 2 / Tq));
                    } else
                        add(simd, data(strided), data(strided), -qdata(strideq));
                    break;
                case BinaryOp::Mul:
                    emul(simd, data(strided), data(strided), qdata(strideq), strategy, state);
                    break;
                case BinaryOp::ScaleSub:
                    if (T != Type::f16) stub();
                    mad(simd, data(strided), -qdata(strideq), data(strided), Immediate::hf(0x6C00));  /* 0x6C00 = 2^12 */
                    break;
                case BinaryOp::MulSub:
                    // Mixed mode: packed bf16 must start at sub-GRF offset 0 or 16.
                    if (bfMixed && strided == 1 && data.getOffset() % 16)
                        stub("Unsupported bf16 mixed-mode mad");
                    // Pure bf16 mad: vector sources must share the destination's sub-GRF offset.
                    if (T == Type::bf16 && !bfMixed && strideq && (qdata.getOffset() != data.getOffset() || odata.getOffset() != data.getOffset()))
                        stub("Misaligned bf16 quantization data");
                    mad(simd, data(strided), -odata(strideq), data(strided), qdata(strideq));
                    break;
                default: stub();
            }
            x0 += simd * strided / crosspack;
        }
        }
    }

    safeReleaseRanges(qPairs, state);
}

// Shift s4/2 data by 8/2 to transfrom it into u4/2 data.
template <HW hw>
void Generator<hw>::dequantizeSubByteIntShift(Type Tsrc, GRFMultirange src, const CommonStrategy &strategy)
{
    if (!Tsrc.isSigned()) return;
    int shiftVal = (Tsrc.bits() == 4) ? 0x8888 : 0xAAAA;
    map(hw, Type::u16, src, src, strategy, [&](int esize, RegData r, RegData _) {
        xor_(esize, r, r, shiftVal);
    });
}

// Optimized int2/int3/int4 -> f16/bf16/f32 dequantization sequence.
template <HW hw>
void Generator<hw>::dequantizeSubByteInt(bool doA, const RegisterLayout& layoutSrc, const RegisterLayout& layoutDst,
    const RegisterLayout& layoutOffset, const RegisterLayout& layoutScale,
    const GRFMultirange& src, const GRFMultirange& dst, const GRFMultirange& offset, const GRFMultirange& scale,
    int offR, int offC, int h, int kab_load, int kq_load,
    const GEMMProblem* problem, const CommonStrategy& strategy, CommonState& state, bool signedShift)
{
    auto Tsrc = layoutSrc.type(), Tdst = layoutDst.type();
    if (!canDequantizeSubByteInt(layoutSrc, layoutDst, layoutOffset, layoutScale))
        stub("Cannot perform dequantizeSubByteInt");

    // Offsets applied after the scales (x = q * scale - offset) go into the scale mad.
    bool offsetAfterScale = problem && (doA ? problem->aOffsetAfterScale : problem->bOffsetAfterScale);
    bool earlyOffset = !layoutOffset.empty() && !offsetAfterScale;
    auto scaleOp = offsetAfterScale ? BinaryOp::MulSub : BinaryOp::Mul;
    auto lateOffsetLayout = offsetAfterScale ? &layoutOffset : nullptr;
    auto lateOffsetRegs = offsetAfterScale ? &offset : nullptr;

    bool signedInt = Tsrc.isSigned();
    bool i4 = (Tsrc.bits() == 4);
    bool f32 = (Tdst == Type::f32);
    bool bf16 = (Tdst == Type::bf16);

    RegisterLayout layoutDstF16;
    const RegisterLayout *effLayoutDst = &layoutDst;
    GRFMultirange dstF16;
    const GRFMultirange *effDst = &dst;
    if (f32 || bf16) {
        layoutDstF16 = RegisterLayout(hw, Type::f16, layoutSrc.rows(), layoutSrc.cols(), layoutDst.colMajor(), layoutDst.crosspack(), 0, 0, (hw < HW::XeHP));
        for (auto &block: layoutDstF16) {
            block.offsetR += layoutDst[0].offsetR;
            block.offsetC += layoutDst[0].offsetC;
        }
        dstF16 = chunkAlloc(layoutDstF16.regs(), 2, state);
        effLayoutDst = &layoutDstF16;
        effDst = &dstF16;
    }

    // 1) Shift s4/2 data to u4/2 data by adding 8/2.
    if (signedInt && signedShift)
        dequantizeSubByteIntShift(Tsrc, src, strategy);

    // 2) Copy u4/2 -> u16 data.
    copyRegisters(Tsrc.asUnsigned(), Type::u16, layoutSrc, *effLayoutDst, src, *effDst, offR, offC, false, strategy, state);

    // 3) Reinterpret u16 data as denormal f16, scale into normal range and subtract (rescaled) offsets if available.
    //     The required rescaling factor (2^24) is necessarily outside f16 range,
    //     so two multiplications are needed.
    if (earlyOffset) {
        if (!problem) stub();
        gemmDequantizeOperation(doA, Type::f16, Type::f16, BinaryOp::ScaleSub, *effLayoutDst, layoutOffset, *effDst, offset, h, kab_load, kq_load, *problem, strategy, state);
    } else {
        map(hw, Type::f16, *effDst, *effLayoutDst, strategy, [&](int esize, RegData r) {
            signedInt ? mad(esize, r, Immediate::hf(i4 ? 0x9800 : 0x9000), r, Immediate::hf(0x6C00)) /* 0x9800 = -8*2^(-12), 0x6C00 = 2^12 */
                      : mul(esize, r, r, Immediate::hf(0x6C00));
        });
    }

    // 4) Finish rescaling -- remaining factor is 2^12.
    map(hw, Type::f16, *effDst, *effLayoutDst, strategy, [&](int esize, RegData r) {
        mul(esize, r, r, Immediate::hf(0x6C00));
    });

    // 5) Apply scales if present. If the scales are not too large (absolute value < 128),
    //      this could be merged into the previous multiplication.
    if (!f32 && !layoutScale.empty()) {
        if (!problem) stub();
        gemmDequantizeOperation(doA, Type::f16, Type::f16, scaleOp, *effLayoutDst, layoutScale, *effDst, scale, h, kab_load, kq_load, *problem, strategy, state, lateOffsetLayout, lateOffsetRegs);
    }

    // 6) Convert to dst type if needed.
    if (f32 || bf16) {
        copyRegisters(Type::f16, Tdst, layoutDstF16, layoutDst, dstF16, dst, offR, offC, false, strategy, state);
        safeReleaseRanges(dstF16, state);
    }

    // 7) Apply scales for f32 after f16->f32 upconversion.
    if (f32 && !layoutScale.empty()) {
        if (!problem) stub();
        gemmDequantizeOperation(doA, Type::f32, Type::f32, scaleOp, layoutDst, layoutScale, dst, scale, h, kab_load, kq_load, *problem, strategy, state, lateOffsetLayout, lateOffsetRegs);
    }
}

// Dequantize A/B, given 2D grouped quantization data.
template <HW hw>
void Generator<hw>::gemmDequantizeAB(bool doA, const RegisterLayout &layoutSrc, const RegisterLayout &layoutDst0,
                                     const GRFMultirange &src, const GRFMultirange &dst0, int h, int kab_load, int kab_repack, int kq_load,
                                     const GEMMProblem &problem, const GEMMStrategy &strategy, GEMMState &state,
                                     bool s4Shift)
{
    auto Tsrc = layoutSrc.type(), Tdst = layoutDst0.type();
    auto Txo_int     = doA ? state.Tao_int             : state.Tbo_int;
    auto Txs_int     = doA ? state.Ta_scaleInt         : state.Tb_scaleInt;
    auto &oiLayout   = doA ? state.A_offsetLayout      : state.B_offsetLayout;
    auto &orLayout   = doA ? state.Ar_offsetLayout     : state.Br_offsetLayout;
    auto &oiRegs     = doA ? state.A_offsetRegs        : state.B_offsetRegs;
    auto &orRegs     = doA ? state.Ar_offsetRegs       : state.Br_offsetRegs;
    auto &siLayout   = doA ? state.A_scaleLayout       : state.B_scaleLayout;
    auto &srLayout   = doA ? state.Ar_scaleLayout      : state.Br_scaleLayout;
    auto &siRegs     = doA ? state.A_scaleRegs         : state.B_scaleRegs;
    auto &srRegs     = doA ? state.Ar_scaleRegs        : state.Br_scaleRegs;
    bool lateOffset  = doA ? problem.needsBGroupSums() : problem.needsAGroupSums();
    bool lateScale   = doA ? state.lateScale2DA        : state.lateScale2DB;

    auto &oLayout = orLayout.empty() ? oiLayout : orLayout;
    auto &oRegs   = orLayout.empty() ? oiRegs   : orRegs;
    auto &sLayout = srLayout.empty() ? siLayout : srLayout;
    auto &sRegs   = srLayout.empty() ? siRegs   : srRegs;

    bool xo2D = !oLayout.empty() && !lateOffset;
    bool xs2D = !sLayout.empty() && !lateScale;
    // Offsets applied after the scales (x = q * scale - offset) go into the scale mad.
    bool offsetAfterScale = doA ? problem.aOffsetAfterScale : problem.bOffsetAfterScale;
    bool xo2DEarly = xo2D && !offsetAfterScale;

    bool copy = !layoutDst0.empty();
    auto layoutDst = copy ? layoutDst0 : layoutSrc;
    auto dst       = copy ? dst0 : src;

    auto Tx_int = xo2DEarly ? Txo_int : Tdst;

    if (xo2DEarly && !xs2D && (Txo_int.bits() > Tdst.bits()))
        Tx_int = Tdst;

    int offR = doA ? 0 : h % kab_repack;
    int offC = doA ? h % kab_repack : 0;
    int offR0 = offR, offC0 = offC;

    int ms = layoutSrc.rows(), ns = layoutSrc.cols();
    int md = layoutDst.rows(), nd = layoutDst.cols();

    if (ms < md || ns < nd) {
        if (!copy) stub();
        layoutDst = RegisterLayout(hw, Tdst, ms, ns, layoutDst0.colMajor(), layoutDst[0].crosspack);
        dst = chunkAlloc(layoutDst.regs(), 2, state);
        offR = offC = 0;
    }

    if (canDequantizeSubByteInt(layoutSrc, layoutDst, oLayout, sLayout)) {
        dequantizeSubByteInt(doA, layoutSrc, layoutDst, oLayout, sLayout,
                       src, dst, oRegs, sRegs, offR, offC, h, kab_load, kq_load, &problem,
                       strategy, state, s4Shift);
    } else {
        if (copy)
            copyRegisters(Tsrc, Tx_int, layoutSrc, layoutDst, src, dst, offR, offC, false, strategy, state);
        else if (Tsrc.asSigned() != Tx_int.asSigned())
            convert(src, Tsrc, Tx_int, strategy, state);

        if (xo2DEarly) {
            if (!state.useBDPAS)
            {
                gemmDequantizeOperation(doA, Tx_int, Txo_int, BinaryOp::Sub, layoutDst, oLayout, dst, oRegs, h, kab_load, kq_load, problem, strategy, state);
                convert(dst, Tx_int, Tdst, strategy, state);
            }
        }

        if (xs2D)
            if (!state.useBDPAS)
            {
                gemmDequantizeOperation(doA, Tdst, Txs_int, offsetAfterScale ? BinaryOp::MulSub : BinaryOp::Mul, layoutDst, sLayout, dst, sRegs, h, kab_load, kq_load, problem, strategy, state,
                                        offsetAfterScale ? &oLayout : nullptr, offsetAfterScale ? &oRegs : nullptr);
	        }
    }

    if (ms < md || ns < nd) {
        copyRegisters(Tdst, Tdst, layoutDst, layoutDst0, dst, dst0, offR0, offC0, false, strategy, state);
        safeReleaseRanges(dst, state);
    }
}

GEMMSTONE_NAMESPACE_END
