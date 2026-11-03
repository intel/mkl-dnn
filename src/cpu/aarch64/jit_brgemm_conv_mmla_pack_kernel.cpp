/*******************************************************************************
* Copyright 2026 FUJITSU LIMITED
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

#include "cpu/aarch64/jit_brgemm_conv_mmla_pack_kernel.hpp"

#include "common/utils.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace jit_brgemm_conv_mmla_pack_kernel {

using namespace Xbyak_aarch64;

#define GET_OFF(field) \
    static_cast<uint32_t>(offsetof(jit_brgemm_conv_mmla_pack_args_t, field))
#define GET_OFF_BATCH(field) \
    static_cast<uint32_t>(offsetof(brgemm_batch_element_t, field))

jit_brgemm_conv_mmla_pack_kernel_t::jit_brgemm_conv_mmla_pack_kernel_t(
        const jit_brgemm_conv_conf_t &ajcp, int K, dim_t lda_bytes)
    : jit_generator_t(nullptr, MAX_CODE_SIZE, true, ajcp.isa)
    , K_(K)
    , lda_(lda_bytes)
    , vlen_(static_cast<int>(isa_max_vlen(ajcp.isa))) {
    assert(K_ > 0 && lda_ > 0);
}

void jit_brgemm_conv_mmla_pack_kernel_t::select_row_src(
        const XReg &dst, const XReg &row_idx, const XReg &row_addr) {
    Label done;
    mov(dst, reg_pad);
    cmp(row_idx, reg_top);
    b(LT, done);
    cmp(row_idx, reg_vend);
    b(GE, done);
    mov(dst, row_addr);
    L(done);
}

// Emits the packing of one row pair: for every vlen-byte group of K, rows
// 2p and 2p+1 are loaded, interleaved at 8-byte granularity with zip1/zip2
// (giving [r0 c | r1 c] for consecutive chunks c) and stored contiguously.
void jit_brgemm_conv_mmla_pack_kernel_t::pack_pair() {
    // vector offsets used with MUL_VL addressing are limited to [-8, 7]; dst
    // uses two vectors per group, so re-base every 4 groups.
    constexpr int groups_per_base = 4;
    const int n_groups = K_ / vlen_;
    const int k_rem = K_ % vlen_;

    mov(reg_dst_cur, reg_dst);
    for (int g = 0; g < n_groups; g++) {
        const int o = g % groups_per_base;
        if (o == 0 && g > 0) {
            add_imm(reg_src0, reg_src0, groups_per_base * vlen_, X_TMP_0);
            add_imm(reg_src1, reg_src1, groups_per_base * vlen_, X_TMP_0);
            add_imm(reg_dst_cur, reg_dst_cur, 2 * groups_per_base * vlen_,
                    X_TMP_0);
        }
        ld1b(z_r0.b, P_ALL_ONE / T_z, ptr(reg_src0, o, MUL_VL));
        ld1b(z_r1.b, P_ALL_ONE / T_z, ptr(reg_src1, o, MUL_VL));
        zip1(z_lo.d, z_r0.d, z_r1.d);
        zip2(z_hi.d, z_r0.d, z_r1.d);
        st1b(z_lo.b, P_ALL_ONE, ptr(reg_dst_cur, 2 * o, MUL_VL));
        st1b(z_hi.b, P_ALL_ONE, ptr(reg_dst_cur, 2 * o + 1, MUL_VL));
    }

    if (k_rem > 0) {
        int o = n_groups % groups_per_base;
        if (o == 0 && n_groups > 0) {
            add_imm(reg_src0, reg_src0, groups_per_base * vlen_, X_TMP_0);
            add_imm(reg_src1, reg_src1, groups_per_base * vlen_, X_TMP_0);
            add_imm(reg_dst_cur, reg_dst_cur, 2 * groups_per_base * vlen_,
                    X_TMP_0);
        }
        // the predicate zeroes the bytes past K, so the partial last chunk
        // is zero-padded and nothing beyond a row's K bytes is read
        ld1b(z_r0.b, p_tail_ld / T_z, ptr(reg_src0, o, MUL_VL));
        ld1b(z_r1.b, p_tail_ld / T_z, ptr(reg_src1, o, MUL_VL));
        zip1(z_lo.d, z_r0.d, z_r1.d);
        st1b(z_lo.b, p_tail_st0, ptr(reg_dst_cur, 2 * o, MUL_VL));
        const int out_bytes = utils::div_up(k_rem, 8) * 16;
        if (out_bytes > vlen_) {
            zip2(z_hi.d, z_r0.d, z_r1.d);
            st1b(z_hi.b, p_tail_st1, ptr(reg_dst_cur, 2 * o + 1, MUL_VL));
        }
    }
}

void jit_brgemm_conv_mmla_pack_kernel_t::generate() {
    preamble();

    const int k_rem = K_ % vlen_;
    if (k_rem > 0) {
        const int out_bytes = utils::div_up(k_rem, 8) * 16;
        set_preg(p_tail_ld.b, k_rem, X_TMP_0);
        set_preg(p_tail_st0.b, nstl::min(out_bytes, vlen_), X_TMP_0);
        if (out_bytes > vlen_)
            set_preg(p_tail_st1.b, out_bytes - vlen_, X_TMP_0);
    }
    const dim_t pair_bytes = static_cast<dim_t>(utils::div_up(K_, 8)) * 16;

    ldr(reg_batch, ptr(reg_param, GET_OFF(batch)));
    ldr(reg_dst, ptr(reg_param, GET_OFF(dst)));
    ldr(reg_pad, ptr(reg_param, GET_OFF(pad_row)));
    ldr(reg_bs, ptr(reg_param, GET_OFF(bs)));
    ldr(reg_rows, ptr(reg_param, GET_OFF(rows)));

    Label tap_loop, pair_loop, done;
    cbz(reg_bs, done);
    cbz(reg_rows, done);

    L(tap_loop);
    {
        ldr(reg_row0, ptr(reg_batch, GET_OFF_BATCH(ptr.A)));
        ldr(reg_top, ptr(reg_batch, GET_OFF_BATCH(vvpad.top)));
        ldr(reg_tmp, ptr(reg_batch, GET_OFF_BATCH(vvpad.bottom)));
        sub(reg_vend, reg_rows, reg_tmp);
        mov(reg_r, 0);

        L(pair_loop);
        {
            select_row_src(reg_src0, reg_r, reg_row0);
            add(reg_r1, reg_r, 1);
            add_imm(reg_tmp, reg_row0, lda_, X_TMP_0);
            select_row_src(reg_src1, reg_r1, reg_tmp);

            pack_pair();

            add_imm(reg_dst, reg_dst, pair_bytes, X_TMP_0);
            add_imm(reg_row0, reg_row0, 2 * lda_, X_TMP_0);
            add(reg_r, reg_r, 2);
            cmp(reg_r, reg_rows);
            b(LT, pair_loop);
        }

        add_imm(reg_batch, reg_batch, sizeof(brgemm_batch_element_t), X_TMP_0);
        subs(reg_bs, reg_bs, 1);
        b(GT, tap_loop);
    }
    L(done);

    postamble();
}

#undef GET_OFF
#undef GET_OFF_BATCH

} // namespace jit_brgemm_conv_mmla_pack_kernel

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
