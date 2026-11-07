/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_transpose_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::utils;
using namespace Xbyak;

#define GET_OFF(x) offsetof(ctx_t, x)

struct jit_trans_iw_ic_t : public jit_trans_src_t, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_iw_ic_t)
    jit_trans_iw_ic_t(const jit_conv_conf_t *conf)
        : jit_trans_src_t(conf)
        , jit_generator_t(jit_name())
        , typesize(conf->src_dt == data_type::undef
                          ? 2
                          : types::data_type_size(conf->src_dt))
        , is_layout_nxc(utils::one_of(conf_->src_tag, format_tag::ndhwc,
                  format_tag::nhwc, format_tag::nwc)) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    dim_t typesize = 0;
    bool is_layout_nxc = false;
    static constexpr int transpose_size = 16;
    dim_t src_stride = 0, tr_src_stride = 0;

    Opmask kLoadMask1 = k1;
    Opmask kLoadMask2 = k2;
    Opmask kPerm1 = k3;
    Opmask kPerm2 = k4;
    Opmask kTail = k5;
    Opmask kLPad = k6;
    Opmask kRPad = k7;

    Reg64 reg_src = r8;
    Reg64 reg_tr_src = r9;
    Reg64 reg_src_prf = r10;
    Reg64 reg_tr_src_prf = r11;
    Reg64 reg_loop = r12;
    Reg64 reg_tr_src_tmp = r13;
    Reg32 regw_tmp = r14d;
    Reg64 imm_addr64 = rbx;

    Zmm vidx1 = zmm31;
    Zmm vidx2 = zmm30;
    Zmm vidx3 = zmm29;
    Zmm vidx4 = zmm28;
    Zmm vidx5 = zmm27;
    Zmm zmm_tmp = zmm26;
    Zmm zmm_zero = zmm25;

    void kmovw(Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    }
    void kmovd(Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovd(k, regw_tmp);
    }
    Zmm src_zmm(int i) { return Zmm(i); }
    Ymm src_ymm(int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    }
    Xmm src_xmm(int i) {
        assert(i >= 0 && i < 16);
        return Xmm(i);
    }
    void vmovdqa64(Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa64(z, ptr[imm_addr64]);
    }

    void vmovdqa32(Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa32(z, ptr[imm_addr64]);
    }

    void transpose(
            dim_t nrows, dim_t l_pad, dim_t r_pad, bool nontemporal_stores);
    void transpose_2b(
            dim_t nrows, dim_t l_pad, dim_t r_pad, bool nontemporal_stores);
    void transpose_1b(
            dim_t nrows, dim_t l_pad, dim_t r_pad, bool nontemporal_stores);
    void generate() override;
};

void jit_trans_iw_ic_t::transpose(
        dim_t nrows, dim_t l_pad, dim_t r_pad, bool nontemporal_stores) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;

    if (typesize == 2)
        transpose_2b(nrows, l_pad, r_pad, nontemporal_stores);
    else if (typesize == 1)
        transpose_1b(nrows, l_pad, r_pad, nontemporal_stores);
    else
        assert(!"unsupported data type");
}

void jit_trans_iw_ic_t::transpose_2b(
        dim_t nrows, dim_t l_pad, dim_t r_pad, bool nontemporal_stores) {
    auto load_ymm = [this](int i) {
        vmovups(src_ymm(i), EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto kmovd = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovd(k, regw_tmp);
    };
    dim_t l_pad_tail {0}, l_pad_rows {0};
    dim_t r_pad_tail {0}, r_pad_rows {0};

    if (l_pad > 0) {
        dim_t store_pad = 2 * transpose_size;
        l_pad_rows = l_pad / store_pad;
        l_pad_tail = div_up(l_pad % store_pad, 2);
        kmovw(kLPad, (1 << l_pad_tail) - 1);
    }
    if (r_pad > 0) {
        dim_t store_pad = div_up(r_pad, 2);
        r_pad_rows = store_pad / transpose_size;
        r_pad_tail = store_pad % transpose_size;
        kmovw(kRPad, (1 << r_pad_tail) - 1);
    }

    auto padding = [this](Reg64 base, dim_t addr_shift, dim_t pad_rows,
                           dim_t pad_tail, const Opmask &mask, dim_t i) {
        // note: pad can be bigger than 16 because of dilation
        const dim_t row_offset = 2 * transpose_size * typesize;
        const auto pshift = addr_shift * typesize + i * tr_src_stride;
        for (dim_t i_row = 0; i_row < pad_rows; i_row++) {
            auto addr = EVEX_compress_addr(base, pshift + i_row * row_offset);
            vmovups(addr, zmm_zero);
        }
        if (pad_tail > 0) {
            base.setOpmaskIdx(mask.getIdx(), true);
            auto addr
                    = EVEX_compress_addr(base, pshift + pad_rows * row_offset);
            vmovups(addr, zmm_zero);
        }
    };

    auto store = [&](Zmm r, dim_t i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (l_pad > 0) {
            padding(reg_tr_src_tmp, 0, l_pad_rows, l_pad_tail, kLPad, i);
            add(reg_tr_src_tmp, l_pad * typesize);
        }
        if (r_pad > 0) {
            dim_t addr_shift = nrows - r_pad % 2;
            padding(reg_tr_src_tmp, addr_shift, r_pad_rows, r_pad_tail, kRPad,
                    i);
        }

        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(kTail.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovups(addr, r);
    };

    if (l_pad > 0 || r_pad > 0) vpxord(zmm_zero, zmm_zero, zmm_zero);
    dim_t store_tail = rnd_up(nrows, 2);
    kmovw(kTail, (1 << store_tail / 2) - 1);

    const dim_t ic_block = conf_->ic_block;
    const bool is_short_block = ic_block != 16;
    const dim_t ic_tail = conf_->ic_tail;
    // Assertion below as we need vmovdqu16 for ic_tails.
    // If needed, can be extended by using load_bytes() helper.
    assert(IMPLICATION(ic_tail, mayiuse(avx512_core)));
    // load by two rows of src into each even register and permute it
    if (mayiuse(avx512_core)) {
        if (conf_->stride_w > 1 || nrows % 2 || is_layout_nxc)
            kmovd(kLoadMask1, (1 << ic_block) - 1);
        if (conf_->stride_w > 1 || is_layout_nxc) kmovd(kLoadMask2, 0xffff0000);

        if (is_layout_nxc && ic_tail) {
            Label done;
            cmp(dword[param1 + GET_OFF(ch_work)], ic_block);
            je(done, T_NEAR);
            kmovd(kLoadMask1, (1 << ic_tail) - 1);
            kshiftld(kLoadMask2, kLoadMask1, 16);
            L(done);
        }

        for (int i = 0; i < rnd_dn(nrows, 2); i += 2) {
            auto zmm_src0 = src_zmm(i);
            if (conf_->stride_w == 1 && !is_layout_nxc) {
                // load two rows at a time
                vmovdqu16(
                        zmm_src0, EVEX_compress_addr(reg_src, i * src_stride));
            } else {
                // load even row
                vmovdqu16(zmm_src0 | kLoadMask1 | T_z,
                        EVEX_compress_addr(reg_src, i * src_stride));
                // load odd row to the second half of register
                if (is_short_block || ic_tail) {
                    auto zmm_src_tmp = src_zmm(i + 1);
                    vmovdqu16(zmm_src_tmp | kLoadMask1 | T_z,
                            EVEX_compress_addr(reg_src, (i + 1) * src_stride));
                    vinsertf64x4(zmm_src0, zmm_src0, src_ymm(i + 1), 1);
                } else {
                    vmovdqu16(zmm_src0 | kLoadMask2,
                            EVEX_compress_addr(
                                    reg_src, (i + 1) * src_stride - 32));
                }
            }
            vpermw(zmm_src0, vidx5, zmm_src0);
        }

        // for odd numbers we need to mix row with zeroes
        if (nrows % 2) {
            int i = static_cast<int>(nrows - 1);
            auto zmm_src0 = src_zmm(i);
            vmovdqu16(zmm_src0 | kLoadMask1 | T_z,
                    EVEX_compress_addr(reg_src, i * src_stride));
            vpermw(zmm_src0, vidx5, zmm_src0);
        }

        for (int i = static_cast<int>(rnd_up(nrows, 2)); i < 16; i += 2) {
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
        }
    } else {
        // all loads
        for (int i = 0; i < 16; i++) {
            vpxord(src_zmm(i), src_zmm(i), src_zmm(i));
        }

        for (int i = 0; i < rnd_dn(nrows, 2); i += 2) {
            auto src0 = src_ymm(i);
            auto src1 = src_ymm(i + 1);
            auto zmm_src0 = src_zmm(i);
            load_ymm(i);

            vpunpcklwd(src1, src0,
                    EVEX_compress_addr(reg_src, (i + 1) * src_stride));
            vpunpckhwd(src0, src0,
                    EVEX_compress_addr(reg_src, (i + 1) * src_stride));
            vinserti64x4(zmm_src0, zmm_src0, src1, 1);
            vpermps(zmm_src0 | kLoadMask1, vidx4, zmm_src0);
        }

        // for odd numbers we need to mix row with zeroes
        if (nrows % 2) {
            int i = static_cast<int>(nrows - 1);
            auto src0 = src_ymm(i);
            auto src1 = src_ymm(i + 1); // zero

            auto zmm_src0 = src_zmm(i);
            vpxor(src1, src1, src1);

            load_ymm(i);
            vpunpckhwd(src0, src0, src1);
            vinserti64x4(zmm_tmp, zmm_tmp, src0, 0);
            vpxor(src0, src0, src0);
            load_ymm(i);
            vpunpcklwd(src1, src0, src1);
            vinserti64x4(zmm_tmp, zmm_tmp, src1, 1);
            vpxord(zmm_src0, zmm_src0, zmm_src0);
            vmovups(zmm_src0, zmm_tmp);
            vpermps(zmm_src0 | kLoadMask1, vidx4, zmm_src0);
        }
    }
    kmovw(kPerm1, 0x5555);
    kmovw(kPerm2, 0xaaaa);

    // swap 1
    for (int i = 0; i < 16; i += 4) {
        auto zmm0 = src_zmm(i);
        auto zmm1 = src_zmm(i + 2);
        auto tmp0 = src_zmm(i + 1);
        auto tmp1 = src_zmm(i + 3);

        vmovups(tmp0, zmm0);
        vmovups(tmp1, zmm1);

        vpermps(tmp0 | kPerm2, vidx3, zmm1);
        vpermps(tmp1 | kPerm1, vidx3, zmm0);
    }
    // swap 2
    int base_idx;
    base_idx = 0;

    kmovw(kPerm1, 0xaa);
    kmovw(kPerm2, 0x55);

    for (int i = 0; i < 4; i += 2) {
        auto zmm0 = src_zmm(base_idx + i + 1);
        auto zmm1 = src_zmm(base_idx + i + 5);

        auto tmp0 = src_zmm(base_idx + i);
        auto tmp1 = src_zmm(base_idx + i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kPerm1, vidx2, zmm1);
        vpermpd(tmp1 | kPerm2, vidx2, zmm0);
    }
    base_idx = 8;
    for (int i = 0; i < 4; i += 2) {
        auto zmm0 = src_zmm(base_idx + i + 1);
        auto zmm1 = src_zmm(base_idx + i + 5);

        auto tmp0 = src_zmm(base_idx + i);
        auto tmp1 = src_zmm(base_idx + i + 4);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kPerm1, vidx2, zmm1);
        vpermpd(tmp1 | kPerm2, vidx2, zmm0);
    }

    kmovw(kPerm1, 0xcc);
    kmovw(kPerm2, 0x33);

    // swap 3
    for (int i = 0; i < 8; i += 2) {
        auto zmm0 = src_zmm(i);
        auto zmm1 = src_zmm(i + 8);

        auto tmp0 = src_zmm(i + 1);
        auto tmp1 = src_zmm(i + 9);

        vmovupd(tmp0, zmm0);
        vmovupd(tmp1, zmm1);

        vpermpd(tmp0 | kPerm1, vidx1, zmm1);
        vpermpd(tmp1 | kPerm2, vidx1, zmm0);
    }

    // all stores
    for (int i = 0; i < 16; i += 2)
        vextracti64x4(src_ymm(i), src_zmm(i + 1), 1);

    auto get_vec_idx = [](int ic_idx) {
        assert(ic_idx < 16 && ic_idx >= 0);
        switch (ic_idx) {
            case 0: return 1;
            case 1: return 0;
            case 2: return 3;
            case 3: return 2;
            case 4: return 9;
            case 5: return 8;
            case 6: return 11;
            case 7: return 10;
            case 8: return 5;
            case 9: return 4;
            case 10: return 7;
            case 11: return 6;
            case 12: return 13;
            case 13: return 12;
            case 14: return 15;
            default: return 14;
        }
    };

    for (int ic = 0; ic < ic_block; ic++)
        store(src_zmm(get_vec_idx(ic)), ic);
}

void jit_trans_iw_ic_t::transpose_1b(
        dim_t nrows, dim_t l_pad, dim_t r_pad, bool nontemporal_stores) {

    auto load = [this, nrows](int i) {
        auto zmm_src = src_zmm(i);
        if (i < nrows) {
            auto addr = EVEX_compress_addr(reg_src, i * src_stride);
            vmovdqu8(zmm_src | kLoadMask1 | T_z, addr);
        } else
            vpxord(zmm_src, zmm_src, zmm_src);
    };

    dim_t l_pad_tail {0}, r_pad_tail {0}, l_pad_rows {0}, r_pad_rows {0};

    if (l_pad > 0) {
        l_pad_rows = l_pad / transpose_size;
        l_pad_tail = l_pad % transpose_size;
        kmovw(kLPad, (1 << l_pad_tail) - 1);
    }
    if (r_pad > 0) {
        r_pad_rows = r_pad / transpose_size;
        r_pad_tail = r_pad % transpose_size;
        kmovw(kRPad, (1 << r_pad_tail) - 1);
    }

    auto padding = [this](Reg64 base, dim_t addr_shift, dim_t pad_rows,
                           dim_t pad_tail, const Opmask &mask, dim_t i) {
        // note: pad can be bigger than 16 because of dilation
        const dim_t row_off = transpose_size;
        auto xmm_zero = Xmm(zmm_zero.getIdx());
        const auto pshift = addr_shift * typesize + i * tr_src_stride;
        for (dim_t i_row = 0; i_row < pad_rows; i_row++) {
            auto addr = EVEX_compress_addr(base, pshift + i_row * row_off);
            vmovups(addr, xmm_zero);
        }
        if (pad_tail > 0) {
            base.setOpmaskIdx(mask.getIdx(), true);
            auto addr = EVEX_compress_addr(base, pshift + pad_rows * row_off);
            vmovdqu8(addr, xmm_zero);
        }
    };

    auto store = [&](Zmm r, dim_t i) {
        mov(reg_tr_src_tmp, reg_tr_src);
        if (l_pad > 0) {
            padding(reg_tr_src_tmp, 0, l_pad_rows, l_pad_tail, kLPad, i);
            add(reg_tr_src_tmp, l_pad);
        }
        if (r_pad > 0) {
            padding(reg_tr_src_tmp, nrows, r_pad_rows, r_pad_tail, kRPad, i);
        }

        auto base = reg_tr_src_tmp;
        base.setOpmaskIdx(kTail.getIdx(), true);

        auto addr = EVEX_compress_addr(base, i * tr_src_stride);
        vmovdqu8(addr, r);
    };

    if (l_pad > 0 || r_pad > 0) vpxord(zmm_zero, zmm_zero, zmm_zero);
    dim_t store_tail = rnd_up(nrows, 4);
    kmovw(kTail, (1 << store_tail) - 1);

    // load rows and swap bytes
    for (int i = 0; i < nrows; i += 4) {
        load(i);
        load(i + 1);
        load(i + 2);
        load(i + 3);

        // concatenate 4 rows
        auto zmm_src0 = src_zmm(i);
        auto ymm_src0 = src_ymm(i);
        auto ymm_src2 = src_ymm(i + 2);
        auto xmm_src1 = src_xmm(i + 1);
        auto xmm_src3 = src_xmm(i + 3);
        vinserti64x2(ymm_src0, ymm_src0, xmm_src1, 1);
        vinserti64x2(ymm_src2, ymm_src2, xmm_src3, 1);
        vinserti64x4(zmm_src0, zmm_src0, ymm_src2, 1);

        // swap bytes
        vpermb(zmm_src0, vidx1, zmm_src0);
    }
    // zero rest zmm_src
    for (int i = static_cast<int>(rnd_up(nrows, 4)); i < transpose_size;
            i += 4) {
        auto zmm_src0 = src_zmm(i);
        vpxord(zmm_src0, zmm_src0, zmm_src0);
    }
    // At this point every fourth zmm contains four transposed lines from src

    // swap doubles
    for (int i = 0; i < 2; i++) {
        auto idx0 = 8 * i;
        auto idx1 = idx0 + 4;

        auto zmm_src0 = src_zmm(idx0);
        auto zmm_src1 = src_zmm(idx1);

        auto zmm_tmp0 = src_zmm(idx0 + 1);
        auto zmm_tmp1 = src_zmm(idx1 + 1);

        vmovups(zmm_tmp0, vidx2);
        vmovups(zmm_tmp1, vidx3);

        vpermi2d(zmm_tmp0, zmm_src0, zmm_src1);
        vpermi2d(zmm_tmp1, zmm_src0, zmm_src1);
    }

    // swap quads
    for (int i = 0; i < 2; i++) {
        auto idx0 = 4 * i;
        auto idx1 = idx0 + 8;

        auto zmm_src0 = src_zmm(idx0 + 1);
        auto zmm_src1 = src_zmm(idx1 + 1);

        auto zmm_tmp0 = src_zmm(idx0);
        auto zmm_tmp1 = src_zmm(idx1);

        vmovups(zmm_tmp0, vidx4);
        vmovups(zmm_tmp1, vidx5);

        vpermi2q(zmm_tmp0, zmm_src0, zmm_src1);
        vpermi2q(zmm_tmp1, zmm_src0, zmm_src1);
    }

    // extract columns
    for (int i = 0; i < 16; i += 4) {
        vextracti64x4(src_ymm(i + 2) | T_z, src_zmm(i), 1);
        vextracti32x4(src_xmm(i + 1) | T_z, src_zmm(i), 1);
        vextracti32x4(src_xmm(i + 3) | T_z, src_ymm(i + 2), 1);
    }

    auto get_vec_idx = [](int col_idx) {
        assert(col_idx < transpose_size && col_idx >= 0);

        const auto div = col_idx / 4;
        const auto mod = col_idx % 4;

        return mod * 4 + div;
    };

    const dim_t ic_block = conf_->ic_block;
    for (int col_idx = 0; col_idx < ic_block; col_idx++) {
        store(src_zmm(get_vec_idx(col_idx)), col_idx);
    }
}

void jit_trans_iw_ic_t::generate() {
    preamble();

    if (mayiuse(avx512_core)) {
        const dim_t ic_block = conf_->ic_block;
        const int ic_tail = conf_->ic_tail;
        if (conf_->stride_w > 1 || is_layout_nxc) {
            kmovd(kLoadMask1, (1 << ic_block) - 1);
            kmovd(kLoadMask2, 0xffff0000);
        }

        if (is_layout_nxc && ic_tail) {
            Label done;
            cmp(dword[param1 + GET_OFF(ch_work)], ic_block);
            je(done, T_NEAR);
            kmovd(kLoadMask1, (1 << ic_tail) - 1);
            kshiftld(kLoadMask2, kLoadMask1, 16);
            L(done);
        }
    } else {
        kmovw(kLoadMask1, 0xffff);
    }

    if (typesize == 2) {
        alignas(64) static constexpr const int64_t idx1[8]
                = {2, 3, 0, 1, 6, 7, 4, 5};
        alignas(64) static constexpr const int64_t idx2[8]
                = {1, 0, 3, 2, 5, 4, 7, 6};
        alignas(64) static constexpr const int32_t idx3[16]
                = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
        alignas(64) static constexpr const int32_t idx4[16]
                = {8, 10, 12, 14, 0, 2, 4, 6, 9, 11, 13, 15, 1, 3, 5, 7};
        alignas(64) static constexpr const uint16_t idx5[32]
                = {0, 16, 2, 18, 8, 24, 10, 26, 4, 20, 6, 22, 12, 28, 14, 30, 1,
                        17, 3, 19, 9, 25, 11, 27, 5, 21, 7, 23, 13, 29, 15, 31};

        vmovdqa64(vidx1, idx1);
        vmovdqa64(vidx2, idx2);
        vmovdqa32(vidx3, idx3);
        vmovdqa32(vidx4, idx4);
        vmovdqa32(vidx5, (const int32_t *)idx5);
    } else if (typesize == 1) {
        alignas(64) static constexpr const uint8_t idx1[64] = {0, 16, 32, 48, 1,
                17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36, 52, 5, 21,
                37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25, 41,
                57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45,
                61, 14, 30, 46, 62, 15, 31, 47, 63};
        alignas(64) static constexpr const uint32_t idx2[16]
                = {0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30};
        alignas(64) static constexpr const uint32_t idx3[16]
                = {1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31};
        alignas(64) static constexpr const uint64_t idx4[8]
                = {0, 8, 2, 10, 4, 12, 6, 14};
        alignas(64) static constexpr const uint64_t idx5[8]
                = {1, 9, 3, 11, 5, 13, 7, 15};

        vmovdqa64(vidx1, (const int64_t *)idx1);
        vmovdqa64(vidx2, (const int64_t *)idx2);
        vmovdqa64(vidx3, (const int64_t *)idx3);
        vmovdqa64(vidx4, (const int64_t *)idx4);
        vmovdqa64(vidx5, (const int64_t *)idx5);
    } else
        assert(!"unsupported data type");

    const dim_t ic_block = conf_->ic_block;
    const dim_t src_mult
            = is_layout_nxc ? conf_->ngroups * conf_->ic : ic_block;
    const dim_t iw = conf_->iw;
    const dim_t tr_iw = conf_->tr_iw;
    const dim_t str_w = conf_->stride_w;
    assert(tr_iw % str_w == 0);
    const dim_t tr_iw_s = tr_iw / str_w;
    assert(transpose_size >= ic_block);

    // Data for every strided case is placed consecutively
    // For 1x1 convolutions with strides we transpose only needed elements
    const auto str_w_end = (conf_->kw == 1) ? 1 : str_w;
    for (dim_t s = 0; s < str_w_end; s++) {
        const dim_t left_pad
                = div_up(nstl::max<dim_t>(0, conf_->l_pad - s), str_w);
        const dim_t iw1 = iw + conf_->l_pad;
        const dim_t iw_s
                = (s < (iw1 % str_w) ? div_up(nstl::max<dim_t>(0, iw1), str_w)
                                     : iw1 / str_w)
                - left_pad;
        const dim_t right_pad = tr_iw_s - iw_s - left_pad;

        const dim_t transposes
                = utils::div_up(nstl::max<dim_t>(0, iw_s), transpose_size);
        dim_t loop_iters = nstl::max<dim_t>(0, transposes - 1);
        dim_t tail = iw_s - loop_iters * transpose_size;

        src_stride = src_mult * typesize * str_w;
        tr_src_stride = tr_iw * typesize;

        bool nontemporal_stores = false;

        const dim_t src_step = src_mult * transpose_size * str_w * typesize;
        const dim_t tr_src_step = transpose_size * typesize;

        mov(reg_src, ptr[param1 + GET_OFF(src)]);
        mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
        mov(reg_src_prf, ptr[param1 + GET_OFF(src_prf)]);
        mov(reg_tr_src_prf, ptr[param1 + GET_OFF(tr_src_prf)]);

        if (str_w > 1) {
            dim_t tr_src_shift = s;
            dim_t src_shift = (str_w - (conf_->l_pad % str_w) + s) % str_w;
            add(reg_src, src_shift * src_mult * typesize);
            add(reg_tr_src, tr_src_shift * tr_iw_s * typesize);
            add(reg_src_prf, src_shift * src_mult * typesize);
            add(reg_tr_src_prf, tr_src_shift * tr_iw_s * typesize);
        }

        if (left_pad > 0 && loop_iters > 0) {
            loop_iters--;
            transpose(transpose_size, left_pad, 0, nontemporal_stores);
            add(reg_src, src_step);
            add(reg_tr_src, tr_src_step + left_pad * typesize);
            add(reg_src_prf, src_step);
            add(reg_tr_src_prf, tr_src_step + left_pad * typesize);
        }

        if (loop_iters) {
            mov(reg_loop, loop_iters);
            Label loop;
            L(loop);
            {
                transpose(transpose_size, 0, 0, nontemporal_stores);
                add(reg_src, src_step);
                add(reg_tr_src, tr_src_step);
                add(reg_src_prf, src_step);
                add(reg_tr_src_prf, tr_src_step);
                sub(reg_loop, 1);
                jnz(loop);
            }
        }
        if (transposes > 1)
            transpose(tail, 0, right_pad, nontemporal_stores);
        else
            transpose(tail, left_pad, right_pad, nontemporal_stores);
    }
    postamble();
}

struct jit_trans_ow_oc_t : public jit_trans_dst_t, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_trans_ow_oc_t)
    jit_trans_ow_oc_t(const jit_conv_conf_t *conf)
        : jit_trans_dst_t(conf)
        , jit_generator_t(jit_name())
        , typesize(conf->dst_dt == data_type::undef
                          ? 2
                          : types::data_type_size(conf->dst_dt))
        , is_layout_nxc(utils::one_of(conf_->dst_tag, format_tag::ndhwc,
                  format_tag::nhwc, format_tag::nwc))
        , vnni_block(conf->dst_dt == data_type::undef
                          ? 2
                          : data_type_vnni_granularity(conf->dst_dt)) {}

    void operator()(const ctx_t *ctx) override {
        jit_generator_t::operator()(ctx);
    }

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    dim_t typesize = 0;
    bool is_layout_nxc = false;
    dim_t vnni_block = 0;
    static constexpr int transpose_size = 16;
    dim_t src_stride = 0, tr_src_stride = 0;
    dim_t tail = 0;

    Opmask kFF = k1;
    Opmask mask_lo = k2;
    Opmask k_oc_tail = k3;

    Zmm vidx1 = zmm31;
    Zmm vidx2 = zmm30;
    Zmm vidx3 = zmm29;
    Zmm vidx4 = zmm28;

    Reg64 reg_src = r8;
    Reg64 reg_tr_src = r9;
    Reg64 reg_src_prf = r10;
    Reg64 reg_loop = r12;
    Reg64 reg_tr_src_tmp = r13;
    Reg32 regw_tmp = r14d;
    Reg64 imm_addr64 = rbx;

    void vmovdqa64(Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa64(z, ptr[imm_addr64]);
    }
    void kmovw(Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    }
    void kmovd(Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovd(k, regw_tmp);
    }
    Zmm src_zmm(int i) { return Zmm(i); }
    Ymm src_ymm(int i) {
        assert(i >= 0 && i < 16);
        return Ymm(i);
    }
    Xmm src_xmm(int i) {
        assert(i >= 0 && i < 16);
        return Xmm(i);
    }

    void transpose(
            dim_t nrows, bool nontemporal_stores, bool do_convert = true);
    void transpose_2b(
            dim_t nrows, bool nontemporal_stores, bool do_convert = true);
    void transpose_1b(
            dim_t nrows, bool nontemporal_stores, bool do_convert = true);
    void generate() override;
};

// do_convert (default is 'true') is a flag that determines when to do the
// transformation of the input data and when to simply zero out the output data
void jit_trans_ow_oc_t::transpose(
        dim_t nrows, bool nontemporal_stores, bool do_convert) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 16, "Unsupported transpose size");
    if (!nrows) return;
    if (typesize == 2)
        transpose_2b(nrows, nontemporal_stores, do_convert);
    else if (typesize == 1)
        transpose_1b(nrows, nontemporal_stores, do_convert);
    else
        assert(!"unsupported data type");
}

void jit_trans_ow_oc_t::transpose_2b(
        dim_t nrows, bool nontemporal_stores, bool do_convert) {

    auto load_ymm = [this](int i) {
        auto ymm_reg = src_ymm(i);
        auto addr = EVEX_compress_addr(reg_src, i * src_stride);
        if (conf_->oc_tail) {
            ymm_reg = ymm_reg | k_oc_tail | T_z;
            // Assertion below as we need vmovdqu16 for tails.
            // If needed, can be removed by using load_bytes() helper.
            assert(mayiuse(avx512_core));
            vmovdqu16(ymm_reg, addr);
        } else {
            vmovups(ymm_reg, addr);
        }
    };

    auto store = [this, nontemporal_stores](Zmm r, int i) {
        auto addr = EVEX_compress_addr(reg_tr_src, i * tr_src_stride);
        if (nontemporal_stores)
            vmovntps(addr, r);
        else
            vmovups(addr, r);
    };

    const auto row_pad = nrows % 2;

    if (mayiuse(avx512_core) && !is_layout_nxc) {
        // TODO: adopt for nhwc?
        for (int i = 0; i < rnd_dn(nrows, 2); i += 2) {
            auto zmm_src0 = src_zmm(i);
            if (do_convert) {
                vmovdqu16(
                        zmm_src0, EVEX_compress_addr(reg_src, i * src_stride));
                vpermw(zmm_src0, vidx2, zmm_src0);
            } else {
                vpxord(zmm_src0, zmm_src0, zmm_src0);
            }
            store(zmm_src0, i);
        }
        if (row_pad > 0) {
            auto zmm_src0 = src_zmm(29);
            if (do_convert) {
                vmovdqu16(zmm_src0 | mask_lo | T_z,
                        EVEX_compress_addr(reg_src, (nrows - 1) * src_stride));
                vpermw(zmm_src0, vidx2, zmm_src0);
            } else {
                vpxord(zmm_src0, zmm_src0, zmm_src0);
            }
            store(zmm_src0, static_cast<int>(nrows - 1));
        }
    } else {
        for (int i = 0; i < rnd_dn(nrows, 2); i += 2) {
            auto src0 = src_ymm(i);
            auto src1 = src_ymm(i + 1);
            auto zmm_src0 = src_zmm(i);
            if (do_convert) {
                load_ymm(i);
                if (is_layout_nxc && conf_->oc_tail) {
                    load_ymm(i + 1);
                    auto ymm_tmp = Ymm(30);
                    vpunpcklwd(ymm_tmp, src0, src1);
                    vpunpckhwd(src0, src0, src1);
                    vinserti64x4(zmm_src0, zmm_src0, ymm_tmp, 1);
                } else {
                    vpunpcklwd(src1, src0,
                            EVEX_compress_addr(reg_src, (i + 1) * src_stride));
                    vpunpckhwd(src0, src0,
                            EVEX_compress_addr(reg_src, (i + 1) * src_stride));
                    vinserti64x4(zmm_src0, zmm_src0, src1, 1);
                }
                vpermpd(zmm_src0 | kFF, vidx1, zmm_src0);
            } else {
                vpxord(zmm_src0, zmm_src0, zmm_src0);
            }
            store(zmm_src0, i);
        }
        if (row_pad > 0) {
            auto src0 = src_ymm(static_cast<int>(nrows - 1));
            auto src1 = src_ymm(static_cast<int>(nrows));
            auto zmm_src0 = src_zmm(30);
            if (do_convert) {
                load_ymm(static_cast<int>(nrows - 1));

                vpxor(src1, src1, src1);
                vpunpckhwd(src1, src0, src1);
                vinserti64x4(zmm_src0, zmm_src0, src1, 0);
                vpxor(src1, src1, src1);
                vpunpcklwd(src0, src0, src1);
                vinserti64x4(zmm_src0, zmm_src0, src0, 1);
                vpermpd(zmm_src0 | kFF, vidx1, zmm_src0);
            } else {
                vpxord(zmm_src0, zmm_src0, zmm_src0);
            }
            store(zmm_src0, static_cast<int>(nrows - 1));
        }
    }
}

void jit_trans_ow_oc_t::transpose_1b(
        dim_t nrows, bool nontemporal_stores, bool do_convert) {
    auto load_xmm = [this](int i) {
        auto xmm_reg = src_xmm(i);
        auto addr = EVEX_compress_addr(reg_src, i * src_stride);
        if (conf_->oc_tail) {
            xmm_reg = xmm_reg | k_oc_tail | T_z;
            // Assertion below as we need vmovdqu16 for tails.
            // If needed, can be removed by using load_bytes() helper.
            assert(mayiuse(avx512_core));
            vmovdqu8(xmm_reg, addr);
        } else {
            vmovups(xmm_reg, addr);
        }
    };

    auto store = [this, nontemporal_stores](Zmm r, int i) {
        auto addr = EVEX_compress_addr(reg_tr_src, i * tr_src_stride);
        if (nontemporal_stores)
            vmovntps(addr, r);
        else
            vmovups(addr, r);
    };
    assert(is_layout_nxc);
    assert(vnni_block == 4);

    for (int i = 0; i < rnd_up(nrows, vnni_block);
            i += static_cast<int>(vnni_block)) {
        const auto idx0 = i;
        const auto idx1 = i + 1;
        const auto idx2 = i + 2;
        const auto idx3 = i + 3;
        auto src0 = src_xmm(idx0);
        auto src1 = src_xmm(idx1);
        auto src2 = src_xmm(idx2);
        auto src3 = src_xmm(idx3);
        // two registers from next iteration used as temporal
        auto src4 = src_xmm((i + 4) % 16);
        auto src5 = src_xmm((i + 5) % 16);
        auto zmm_src0 = src_zmm(i);
        if (do_convert) {
            load_xmm(idx0);
            if (idx1 < nrows)
                load_xmm(idx1);
            else
                vpxord(src1, src1, src1);
            if (idx2 < nrows)
                load_xmm(idx2);
            else
                vpxord(src2, src2, src2);
            if (idx3 < nrows)
                load_xmm(idx3);
            else
                vpxord(src3, src3, src3);

            vpunpcklbw(src4, src0, src1);
            vpunpckhbw(src5, src0, src1);
            vpunpcklbw(src0, src2, src3);
            vpunpckhbw(src1, src2, src3);

            vpunpcklwd(src2, src4, src0);
            vpunpckhwd(src3, src4, src0);
            vpunpcklwd(src4, src5, src1);
            vpunpckhwd(src5, src5, src1);

            vinserti64x2(zmm_src0, zmm_src0, src2, 0);
            vinserti64x2(zmm_src0, zmm_src0, src3, 1);
            vinserti64x2(zmm_src0, zmm_src0, src4, 2);
            vinserti64x2(zmm_src0, zmm_src0, src5, 3);
        } else {
            vpxord(zmm_src0, zmm_src0, zmm_src0);
        }
        store(zmm_src0, i);
    }
}

void jit_trans_ow_oc_t::generate() {
    preamble();

    if (typesize == 2) {
        alignas(64) static constexpr const int64_t idx1[8]
                = {4, 5, 0, 1, 6, 7, 2, 3};
        alignas(64) static constexpr const int16_t idx2[32] = {0, 16, 1, 17, 2,
                18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11,
                27, 12, 28, 13, 29, 14, 30, 15, 31};
        vmovdqa64(vidx1, idx1);
        vmovdqa64(vidx2, (const int64_t *)idx2);
    } else if (typesize == 1) {

        alignas(64) static constexpr const uint8_t idx_lo_16[64] = {0, 1, 64,
                65, 4, 5, 68, 69, 2, 3, 66, 67, 6, 7, 70, 71, 8, 9, 72, 73, 12,
                13, 76, 77, 10, 11, 74, 75, 14, 15, 78, 79, 16, 17, 80, 81, 20,
                21, 84, 85, 18, 19, 82, 83, 22, 23, 86, 87, 24, 25, 88, 89, 28,
                29, 92, 93, 26, 27, 90, 91, 30, 31, 94, 95};

        alignas(64) static constexpr const uint8_t idx_hi_16[64] = {32, 33, 96,
                97, 36, 37, 100, 101, 34, 35, 98, 99, 38, 39, 102, 103, 40, 41,
                104, 105, 44, 45, 108, 109, 42, 43, 106, 107, 46, 47, 110, 111,
                48, 49, 112, 113, 52, 53, 116, 117, 50, 51, 114, 115, 54, 55,
                118, 119, 56, 57, 120, 121, 60, 61, 124, 125, 58, 59, 122, 123,
                62, 63, 126, 127};

        alignas(64) static constexpr const uint8_t idx_lo_8[64] = {0, 64, 2, 66,
                1, 65, 3, 67, 8, 72, 10, 74, 9, 73, 11, 75, 4, 68, 6, 70, 5, 69,
                7, 71, 12, 76, 14, 78, 13, 77, 15, 79, 16, 80, 18, 82, 17, 81,
                19, 83, 24, 88, 26, 90, 25, 89, 27, 91, 20, 84, 22, 86, 21, 85,
                23, 87, 28, 92, 30, 94, 29, 93, 31, 95};

        alignas(64) static constexpr const uint8_t idx_hi_8[64] = {32, 96, 34,
                98, 33, 97, 35, 99, 40, 104, 42, 106, 41, 105, 43, 107, 36, 100,
                38, 102, 37, 101, 39, 103, 44, 108, 46, 110, 45, 109, 47, 111,
                48, 112, 50, 114, 49, 113, 51, 115, 56, 120, 58, 122, 57, 121,
                59, 123, 52, 116, 54, 118, 53, 117, 55, 119, 60, 124, 62, 126,
                61, 125, 63, 127};

        vmovdqa64(vidx1 /*vreg_idx_lo_256*/, (const int64_t *)idx_lo_16);
        vmovdqa64(vidx2 /*vreg_idx_hi_256*/, (const int64_t *)idx_hi_16);
        vmovdqa64(vidx3 /*vreg_idx_lo_128*/, (const int64_t *)idx_lo_8);
        vmovdqa64(vidx4 /*vreg_idx_hi_128*/, (const int64_t *)idx_hi_8);
    }

    const dim_t oc_block = conf_->oc_block;
    const dim_t src_mult
            = is_layout_nxc ? conf_->ngroups * conf_->oc : oc_block;
    const dim_t ow = conf_->ow;
    const dim_t transposes = utils::div_up(ow, transpose_size);
    dim_t loop_iters = nstl::max<dim_t>(0, transposes - 1);
    tail = ow - loop_iters * transpose_size;

    src_stride = src_mult * typesize;
    tr_src_stride = oc_block * typesize;

    bool nontemporal_stores = conf_->use_nt_stores_ddst;

    const dim_t src_step = src_mult * transpose_size * typesize;
    const dim_t tr_src_step = oc_block * transpose_size * typesize;

    const dim_t zero_tr_ow
            = nstl::max<dim_t>(0, conf_->tr_ow - rnd_up(ow, vnni_block));

    mov(reg_src, ptr[param1 + GET_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_OFF(tr_src)]);
    mov(reg_src_prf, ptr[param1 + GET_OFF(src_prf)]);

    kmovw(kFF, 0xFF);
    kmovd(mask_lo, 0x0000ffff);

    if (is_layout_nxc && conf_->oc_tail) {
        Label done;
        kxnorw(k_oc_tail, k_oc_tail, k_oc_tail);
        cmp(dword[param1 + GET_OFF(ch_work)], conf_->oc_block);
        je(done, T_NEAR);
        kmovw(k_oc_tail, (1 << conf_->oc_tail) - 1);
        L(done);
    }

    if (loop_iters) {
        mov(reg_loop, loop_iters);
        Label loop;
        L(loop);
        {
            transpose(transpose_size, nontemporal_stores);
            add(reg_src, src_step);
            add(reg_tr_src, tr_src_step);
            add(reg_src_prf, src_step);
            sub(reg_loop, 1);
            jnz(loop);
        }
    }
    transpose(tail, nontemporal_stores);
    if (zero_tr_ow) {
        const auto zero_transposes = utils::div_up(zero_tr_ow, transpose_size);
        const auto zero_loop_iters = nstl::max<dim_t>(0, zero_transposes - 1);
        const auto zero_tail = zero_tr_ow - zero_loop_iters * transpose_size;

        // shift over tail
        add(reg_tr_src, oc_block * rnd_up(tail, vnni_block) * typesize);

        // zero the tr_ow - ow
        if (zero_loop_iters) {
            mov(reg_loop, zero_loop_iters);
            Label zero_loop;
            L(zero_loop);
            {
                transpose(transpose_size, nontemporal_stores, false);
                add(reg_tr_src, tr_src_step);
                sub(reg_loop, 1);
                jnz(zero_loop);
            }
        }
        transpose(zero_tail, nontemporal_stores, false);
    }

    postamble();
}

/*
// -------------------------------------------------
// jit_transpose4x16_src_t
// -------------------------------------------------
*/

void jit_transpose4x16_src_t::transpose(dim_t nrows) {
    assert(nrows >= 0 && nrows <= transpose_size);
    static_assert(transpose_size == 4, "Unsupported transpose size");
    if (!nrows) return;

    auto pf_src_t0 = [this](int i) {
        if (tparams->src_pf0_distance)
            prefetcht0(EVEX_compress_addr(
                    reg_src, (tparams->src_pf0_distance + i) * src_stride));
    };

    auto pf_tr_src_t0 = [this](int i) {
        if (tparams->tr_src_pf0_distance)
            prefetcht0(EVEX_compress_addr(reg_tr_src,
                    (tparams->tr_src_pf0_distance + i) * src_stride));
    };

    auto pf_src_t1 = [this](int i) {
        if (tparams->src_pf1)
            prefetcht1(EVEX_compress_addr(reg_src_prf, i * src_stride));
    };

    auto pf_tr_src_t1 = [this](int i) {
        if (tparams->tr_src_pf1)
            prefetcht1(EVEX_compress_addr(reg_tr_src_prf, i * tr_src_stride));
    };

    auto src_zmm = [](int i) {
        assert(i >= 0 && i < 4);
        return Zmm(i);
    };

    auto tmp_zmm = [](int i) {
        assert(i >= 0 && i < 4);
        return Zmm(4 + i);
    };

    auto load = [this, src_zmm](int i) {
        vmovups(src_zmm(i), EVEX_compress_addr(reg_src, i * src_stride));
    };

    auto store = [this](Zmm r, int i) {
        vmovups(EVEX_compress_addr(reg_tr_src, i * tr_src_stride), r);
    };

    auto tmp0 = tmp_zmm(0);
    auto tmp1 = tmp_zmm(1);
    auto tmp2 = tmp_zmm(2);
    auto tmp3 = tmp_zmm(3);

    auto src0 = src_zmm(0);
    auto src1 = src_zmm(1);
    auto src2 = src_zmm(2);
    auto src3 = src_zmm(3);
    for (int i = 0; i < nrows; i++) {
        load(i);
    }

    for (dim_t i = nrows; i < 4; i++) {
        const int reg_idx = static_cast<int>(i);
        vpxord(src_zmm(reg_idx), src_zmm(reg_idx), src_zmm(reg_idx));
    }

    vmovupd(tmp0, src0);
    vmovupd(tmp1, src1);
    pf_src_t0(0);
    vpermpd(tmp0 | kF0, vidx01, src2);
    vpermpd(tmp1 | kF0, vidx01, src3);

    valignd(src0, src0, src0, 8);
    valignd(src1, src1, src1, 8);
    pf_src_t0(1);
    vmovupd(tmp2, src0);
    vmovupd(tmp3, src1);
    pf_src_t0(2);
    vpermpd(tmp2 | kF0, vidx10, src2);
    vpermpd(tmp3 | kF0, vidx10, src3);
    pf_src_t0(3);

    vmovupd(src0, tmp0);
    pf_src_t1(0);
    vmovupd(src1, tmp2);
    pf_src_t1(1);
    vmovupd(src2, tmp1);
    pf_src_t1(2);
    vmovupd(src3, tmp3);
    pf_src_t1(3);
    vpermpd(src0 | kCC, vidx1, tmp1);
    vpermpd(src1 | kCC, vidx1, tmp3);
    pf_tr_src_t0(0);
    vpermpd(src2 | k33, vidx1, tmp0);
    vpermpd(src3 | k33, vidx1, tmp2);
    pf_tr_src_t0(1);

    vmovupd(tmp0, src0);
    vmovupd(tmp1, src2);
    pf_tr_src_t0(2);
    vmovupd(tmp2, src1);
    vmovupd(tmp3, src3);
    pf_tr_src_t0(3);
    vpermps(tmp0 | kFFFF, vidxP, src0);
    pf_tr_src_t1(0);
    vpermps(tmp1 | kFFFF, vidxP, src2);
    pf_tr_src_t1(1);
    vpermps(tmp2 | kFFFF, vidxP, src1);
    pf_tr_src_t1(3);
    vpermps(tmp3 | kFFFF, vidxP, src3);
    pf_tr_src_t1(4);

    store(tmp0, 0);
    store(tmp1, 1);
    store(tmp2, 2);
    store(tmp3, 3);
}

alignas(64) static constexpr const int64_t idx01[8] = {0, 0, 0, 0, 0, 1, 2, 3};
alignas(64) static constexpr const int64_t idx10[8] = {0, 0, 0, 0, 4, 5, 6, 7};
alignas(64) static constexpr const int64_t idx1[8] = {2, 3, 0, 1, 6, 7, 4, 5};
alignas(64) static constexpr const int32_t idxP[16]
        = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};

void jit_transpose4x16_src_t::generate() {
    preamble();

    const dim_t ic_block = params->ic_block;
    const dim_t is = params->is;
    dim_t tail = is % transpose_size;

    src_stride = ic_block * typesize;
    assert(src_stride == 64);
    tr_src_stride = ic_block * typesize;

    const dim_t src_step = ic_block * transpose_size * typesize;
    const dim_t tr_src_step = ic_block * transpose_size * typesize;

#define GET_TR_OFF(x) offsetof(jit_transpose_src_args_t, x)
    mov(reg_loop, ptr[param1 + GET_TR_OFF(size)]);
    mov(reg_src, ptr[param1 + GET_TR_OFF(src)]);
    mov(reg_tr_src, ptr[param1 + GET_TR_OFF(tr_src)]);
    mov(reg_src_prf, ptr[param1 + GET_TR_OFF(src_prf)]);
    mov(reg_tr_src_prf, ptr[param1 + GET_TR_OFF(tr_src_prf)]);
#undef GET_TR_OFF

    auto kmovw = [this](Opmask k, unsigned w) {
        mov(regw_tmp, w);
        jit_generator_t::kmovw(k, regw_tmp);
    };

    auto vmovdqa64 = [this](Zmm z, const int64_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa64(z, ptr[imm_addr64]);
    };

    auto vmovdqa32 = [this](Zmm z, const int32_t *addr) {
        mov(imm_addr64, reinterpret_cast<size_t>(addr));
        jit_generator_t::vmovdqa32(z, ptr[imm_addr64]);
    };

    kmovw(kF0, 0xf0); // 11110000
    kmovw(kCC, 0xcc); // 11001100
    kmovw(k33, 0x33); // 00110011
    kmovw(kFFFF, 0xffff); // 1111111111111111

    vmovdqa64(vidx01, idx01);
    vmovdqa64(vidx10, idx10);
    vmovdqa64(vidx1, idx1);
    vmovdqa32(vidxP, idxP);

    Label loop_label;
    Label tail_label;

    cmp(reg_loop, transpose_size);
    jl(tail_label, T_NEAR);

    L(loop_label);
    {
        transpose(transpose_size);
        add(reg_src, src_step);
        add(reg_tr_src, tr_src_step);
        add(reg_src_prf, src_step);
        add(reg_tr_src_prf, tr_src_step);
        sub(reg_loop, transpose_size);
        cmp(reg_loop, transpose_size);
        jge(loop_label, T_NEAR);
    }
    L(tail_label);
    transpose(tail);

    postamble();
}

#undef GET_OFF

#define GET_OFF(field) offsetof(jit_conv_args_t, field)

void jit_diff_wei_trans_to_vnni_t::generate() {
    /* Reorder part of F32 weights tensor
       from [VNNI_GRANULARITY][I][kd][kh][kw][16i][16o] to VNNI format [kd][kh][kw][16i][16o][VNNI_GRANULARITY][i]
       and down-convert it to required float. */
    const dim_t ts_out = types::data_type_size(out_dt_);
    const dim_t ts_inp = 4;
    const dim_t simd_w = 16;

    const Reg64 &reg_output = r15;
    const Reg64 &reg_output_kd = r14;
    const Reg64 &reg_input_kw = r13;
    const Reg64 &reg_input_kh = r12;
    const Reg64 &reg_input_kd = r11;
    const Reg64 &reg_prm_table = r9;
    const Reg64 &reg_last_ic_block = rax;
    const Reg64 &reg_kd = rsi;
    const Reg64 &reg_kh = abi_not_param1;
    const Reg64 &reg_tmp = rdx;

    Zmm emu_reserv_1 = Zmm(30);
    Zmm emu_reserv_2 = Zmm(29);
    Zmm emu_reserv_3 = Zmm(28);
    Zmm emu_reserv_4 = Zmm(27);
    Zmm emu_reserv_5 = Zmm(26);
    Reg64 emu_scratch = reg_tmp;
    Xbyak::Opmask emu_mask = Xbyak::Opmask(4);

    std::unique_ptr<fp8_conversion_base_t> f8_cvt;
    if (out_dt_ == data_type::f8_e5m2)
        f8_cvt = utils::make_unique<fp8_conversion_e5m2_t>(this, emu_reserv_1,
                emu_reserv_2, emu_reserv_3, emu_mask, emu_scratch);
    else if (out_dt_ == data_type::f8_e4m3)
        f8_cvt = utils::make_unique<fp8_conversion_e4m3_t>(this, emu_reserv_1,
                emu_reserv_2, emu_reserv_3, emu_reserv_4, emu_reserv_5,
                emu_scratch);
    if (utils::one_of(out_dt_, data_type::f8_e5m2, data_type::f8_e4m3)
            && f8_cvt == nullptr) {
        assert(!"Failed to create f8 emulation kernel.");
        return;
    }

    const Zmm &zmm_idx = Zmm(31);
    auto get_zmm_src = [&](int idx, int ic) { return Zmm(4 * idx + ic); };
    auto get_zmm_bf16 = [&](int ic) { return Zmm(16 + ic); };

    const dim_t vnni_granularity = data_type_vnni_granularity(out_dt_);

    Xbyak::Label prm_table, zero_buffer;
    Xbyak::Label kd_loop_label, kh_loop_label;

    preamble();

    mov(reg_last_ic_block, ptr[abi_param1 + GET_OFF(last_ic_block)]);
    mov(reg_input_kd, ptr[abi_param1 + GET_OFF(src)]);
    mov(reg_output_kd, ptr[abi_param1 + GET_OFF(dst)]);

    mov(reg_prm_table, prm_table);
    vmovups(zmm_idx, ptr[reg_prm_table]);

    dim_t inp_kw_offset = (dim_t)ts_inp * ic_block_ * oc_block_;
    dim_t inp_bc_offset = inp_kw_offset * kd_ * kh_ * kw_;
    dim_t out_kw_offset
            = (dim_t)ts_out * ic_block_ * oc_block_ * vnni_granularity;

    xor_(reg_kd, reg_kd);
    L(kd_loop_label);
    {
        mov(reg_output, reg_output_kd);
        mov(reg_input_kh, reg_input_kd);
        xor_(reg_kh, reg_kh);
        L(kh_loop_label);
        {
            for (int kw = 0; kw < kw_; kw++) {
                for (int bc = 0; bc < vnni_granularity; bc++) {
                    Xbyak::Label last_ic_label, done_ic_label;

                    cmp(reg_last_ic_block, 0);
                    jne(last_ic_label, T_NEAR);
                    {
                        mov(reg_input_kw, reg_input_kh);
                        safe_add(reg_input_kw,
                                bc * inp_bc_offset + kw * inp_kw_offset,
                                reg_tmp);
                        jmp(done_ic_label, T_NEAR);
                    }
                    L(last_ic_label);
                    {
                        if (bc < (nb_ic_ % vnni_granularity)) {
                            mov(reg_input_kw, reg_input_kh);
                            safe_add(reg_input_kw,
                                    bc * inp_bc_offset + kw * inp_kw_offset,
                                    reg_tmp);
                        } else
                            mov(reg_input_kw, zero_buffer);
                    }
                    L(done_ic_label);

                    for_(int ocb = 0; ocb < oc_block_; ocb += simd_w)
                    for (int icc = 0; icc < ic_block_ / vnni_granularity;
                            icc++) {
                        int ic_count = static_cast<int>(
                                bc * (ic_block_ / vnni_granularity) + icc);

                        auto zmm_out = get_zmm_bf16(icc);

                        for (int idx = 0; idx < vnni_granularity; idx++) {
                            auto zmm_src = get_zmm_src(idx, icc);
                            const auto src_offset = ts_inp
                                    * ((vnni_granularity * icc + idx)
                                                    * oc_block_
                                            + ocb);
                            vmovups(zmm_src, ptr[reg_input_kw + src_offset]);
                        }
                        const auto src_offset = ts_inp
                                * ((vnni_granularity * icc) * oc_block_ + ocb);

                        if (one_of(out_dt_, data_type::bf16, data_type::f16)) {
                            const auto zmm_src_0 = get_zmm_src(0, icc);
                            const auto zmm_src_1 = get_zmm_src(1, icc);
                            const auto src_off0 = src_offset;
                            const auto src_off1 = src_off0 + ts_inp * oc_block_;
                            vmovups(zmm_src_0, ptr[reg_input_kw + src_off0]);
                            vmovups(zmm_src_1, ptr[reg_input_kw + src_off1]);
                            if (out_dt_ == data_type::bf16) {
                                vcvtne2ps2bf16(zmm_out, zmm_src_1, zmm_src_0);
                            } else if (out_dt_ == data_type::f16) {
                                Ymm ymm_src_0(zmm_src_0.getIdx());
                                Ymm ymm_src_1(zmm_src_1.getIdx());
                                vcvtps2phx(ymm_src_0, zmm_src_0);
                                vcvtps2phx(ymm_src_1, zmm_src_1);
                                vinsertf32x8(zmm_out, zmm_src_0, ymm_src_1, 1);
                            }
                            vpermw(zmm_out, zmm_idx, zmm_out);
                        } else if (one_of(out_dt_, data_type::f8_e5m2,
                                           data_type::f8_e4m3)) {
                            const auto zmm_src_0 = get_zmm_src(0, icc);
                            const auto zmm_src_1 = get_zmm_src(1, icc);
                            const auto zmm_src_2 = get_zmm_src(2, icc);
                            const auto zmm_src_3 = get_zmm_src(3, icc);
                            Xmm xmm_src_0(zmm_src_0.getIdx());
                            Xmm xmm_src_1(zmm_src_1.getIdx());
                            Xmm xmm_src_2(zmm_src_2.getIdx());
                            Xmm xmm_src_3(zmm_src_3.getIdx());

                            const auto src_off0 = src_offset;
                            const auto src_off1 = src_off0 + ts_inp * oc_block_;
                            const auto src_off2 = src_off1 + ts_inp * oc_block_;
                            const auto src_off3 = src_off2 + ts_inp * oc_block_;

                            // NOTE: the operands must be explicitly sized -
                            // vcvt_f32_to_f8() derives the element count from
                            // the width of the source.
                            f8_cvt->vcvt_f32_to_f8(
                                    xmm_src_0, zword[reg_input_kw + src_off0]);
                            f8_cvt->vcvt_f32_to_f8(
                                    xmm_src_1, zword[reg_input_kw + src_off1]);
                            f8_cvt->vcvt_f32_to_f8(
                                    xmm_src_2, zword[reg_input_kw + src_off2]);
                            f8_cvt->vcvt_f32_to_f8(
                                    xmm_src_3, zword[reg_input_kw + src_off3]);
                            vinserti64x2(zmm_out, zmm_out, xmm_src_0, 0);
                            vinserti64x2(zmm_out, zmm_out, xmm_src_1, 1);
                            vinserti64x2(zmm_out, zmm_out, xmm_src_2, 2);
                            vinserti64x2(zmm_out, zmm_out, xmm_src_3, 3);
                            vpermb(zmm_out, zmm_idx, zmm_out);
                        } else {
                            assert(!"unsupported data type");
                        }

                        vmovups(ptr[reg_output + kw * out_kw_offset
                                        + ts_out
                                                * (ic_count * oc_block_
                                                                * vnni_granularity
                                                        + ocb * vnni_granularity)],
                                zmm_out);
                    }
                }
            }
            safe_add(reg_output,
                    (dim_t)ts_out * kw_ * vnni_granularity * ic_block_
                            * oc_block_,
                    reg_tmp);
            safe_add(reg_input_kh, (dim_t)ts_inp * kw_ * ic_block_ * oc_block_,
                    reg_tmp);

            add(reg_kh, 1);
            cmp(reg_kh, kh_);
            jl(kh_loop_label, T_NEAR);
        }
        safe_add(reg_output_kd,
                (dim_t)ts_out * kh_ * kw_ * vnni_granularity * ic_block_
                        * oc_block_,
                reg_tmp);
        safe_add(reg_input_kd,
                (dim_t)ts_inp * kh_ * kw_ * ic_block_ * oc_block_, reg_tmp);

        add(reg_kd, 1);
        cmp(reg_kd, kd_);
        jl(kd_loop_label, T_NEAR);
    }

    postamble();

    align(64);
    if (one_of(out_dt_, data_type::f8_e5m2, data_type::f8_e4m3)) {

        L(prm_table);
        uint8_t prm_array[64];
        for (size_t i = 0; i < 16; i++) {
            prm_array[4 * i] = static_cast<uint8_t>(i);
            prm_array[4 * i + 1] = static_cast<uint8_t>(i + 16);
            prm_array[4 * i + 2] = static_cast<uint8_t>(i + 32);
            prm_array[4 * i + 3] = static_cast<uint8_t>(i + 48);
        }

        for (size_t i = 0; i < 64; ++i)
            db(prm_array[i]);
    } else {
        L(prm_table);
        const uint16_t prm_array[32] = {0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5,
                21, 6, 22, 7, 23, 8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29,
                14, 30, 15, 31};
        for (size_t i = 0; i < 32; ++i)
            dw(prm_array[i]);
    }

    align(64);
    L(zero_buffer);
    const uint16_t zero = 0;
    for (int i = 0; i < ts_inp * oc_block_ * ic_block_; ++i)
        db(zero);

    if (f8_cvt) f8_cvt->prepare_table();
}

#undef GET_OFF

jit_trans_src_t *create_trans_src(const jit_conv_conf_t *conf) {
    if (conf->has_vnni && IMPLICATION(conf->is_1stconv, conf->transpose_src))
        return new jit_trans_iw_ic_t(conf);
    assert(!"unsupported configuration");
    return nullptr;
}

jit_trans_dst_t *create_trans_dst(const jit_conv_conf_t *conf) {

    if (conf->has_vnni) return new jit_trans_ow_oc_t(conf);
    assert(!"unsupported configuration");
    return nullptr;
}
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
