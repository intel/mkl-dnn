/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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
#include "cpu/aarch64/jit_generator.hpp"
#include "xbyak_aarch64/xbyak_aarch64_reg.h"

#include "cpu/aarch64/matmul/brgemm_matmul_copy_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::utils;
using namespace Xbyak_aarch64;

#define GET_OFF(x) (uint32_t) offsetof(ctx_t, x)

struct jit_brgemm_matmul_copy_b_f32_t : public jit_brgemm_matmul_copy_b_t,
                                        public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_f32_t)

    jit_brgemm_matmul_copy_b_f32_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , dt_in_(data_type::f32)
        , typesize_in_(types::data_type_size(dt_in_))
        , src_stride_(conf_->wei_tag == acbd ? conf_->copy_B_wei_stride
                                             : conf_->N * typesize_in_)
        , tr_src_stride_(conf_->LDB * typesize_out_) {
        MAYBE_UNUSED(src_stride_);
        MAYBE_UNUSED(tr_src_stride_);
    }

    void operator()(ctx_t *ctx) override { jit_generator_t::operator()(ctx); }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak_aarch64::XReg;
    using reg32_t = const Xbyak_aarch64::WReg;
    using opmask_t = const Xbyak_aarch64::PReg;
    using zmm = const Xbyak_aarch64::ZReg;

    enum { n_blk_step = 8, max_regs_available = 30 };
    const data_type_t dt_in_;
    const size_t typesize_in_;
    const size_t typesize_out_ = sizeof(float);
    dim_t src_stride_, tr_src_stride_;

    opmask_t kTail = p7;
    opmask_t kFFFF = p6;

    reg64_t reg_src = x1;
    reg64_t reg_tr_src = x2;

    reg64_t reg_K_iters = x8;
    reg64_t reg_N_blk = x9;
    reg64_t reg_K_start = x10;
    reg32_t regw_tmp = w14;
    reg64_t imm_addr64 = x15;

    zmm zmm_permw = z30;
    zmm zmm_zero = z31;

    void copy_16_8_x_n_block(int nrows, int ncolumns);
    void compute_k_loop(int ncolumns);
    void generate() override;
};

void jit_brgemm_matmul_copy_b_f32_t::copy_16_8_x_n_block(
        int nrows, int ncolumns) {

    int n_blk_step = get_sve_length() / typesize_in_;

    auto get_zmm = [](int reg_idx) {
        assert(reg_idx >= 0 && reg_idx < max_regs_available);
        return ZRegS(reg_idx);
    };

    auto load = [this, get_zmm](int blk, int k, int n, opmask_t current_mask) {
        auto src_zmm = get_zmm(blk);
        add_imm(X_DEFAULT_ADDR, reg_src, k * src_stride_ + n * typesize_in_,
                X_TMP_0);
        ld1w(src_zmm, current_mask / T_z, ptr(X_DEFAULT_ADDR));
    };

    const int columns_tail = ncolumns % n_blk_step;

    if (columns_tail < n_blk_step) set_preg(kTail.s, columns_tail, X_TMP_0);

    int iter = 0;
    for_(int k = 0; k < nrows; k++) //nrows = unroll
    for (int n = 0; n < conf_->wei_n_blk; n += n_blk_step) {

        const dim_t tr_src_off = k * tr_src_stride_ + n * typesize_out_;
        const int zero_padding = ncolumns - n;
        if (zero_padding <= 0) {
            add_imm(X_DEFAULT_ADDR, reg_tr_src, tr_src_off, X_TMP_0);
            str(zmm_zero, ptr(X_DEFAULT_ADDR));
            continue;
        }

        const opmask_t curr_msk = zero_padding < n_blk_step ? kTail : P_ALL_ONE;
        const int blk_idx = iter % max_regs_available;
        load(blk_idx, k, n, curr_msk);
        add_imm(X_DEFAULT_ADDR, reg_tr_src, tr_src_off, X_TMP_0);
        const auto src_zmm0 = ZReg(blk_idx);
        str(src_zmm0, ptr(X_DEFAULT_ADDR));
        iter++;
    }
}

void jit_brgemm_matmul_copy_b_f32_t::compute_k_loop(int ncolumns) {

    auto compute_uni_k_loop = [&](int unroll) {
        Label K_start_label, K_end_label;

        L(K_start_label);
        cmp_imm(reg_K_iters, unroll, X_TMP_0);
        b(LT, K_end_label);

        copy_16_8_x_n_block(unroll, ncolumns);
        add_imm(reg_src, reg_src, unroll * src_stride_, X_TMP_0);
        add_imm(reg_tr_src, reg_tr_src, unroll * tr_src_stride_, X_TMP_0);

        sub_imm(reg_K_iters, reg_K_iters, unroll, X_TMP_0);
        bl(K_start_label);

        L(K_end_label);
    };

    int k_unroll = get_sve_length() / typesize_in_;
    compute_uni_k_loop(k_unroll);
    compute_uni_k_loop(1);
}

void jit_brgemm_matmul_copy_b_f32_t::generate() {

    preamble();
    eor(zmm_zero.d, zmm_zero.d, zmm_zero.d);
    ldr_imm(reg_src, param1, GET_OFF(src));
    ldr_imm(reg_tr_src, param1, GET_OFF(tr_src));
    ldr_imm(reg_K_iters, param1, GET_OFF(current_K_iters));
    ldr_imm(reg_N_blk, param1, GET_OFF(current_N_blk));
    ptrue(kFFFF.s);

    Label done;
    if (conf_->N_tail > 0) {
        Label not_N_tail;
        cmp_imm(reg_N_blk, conf_->N_tail, X_TMP_0);
        b(NE, not_N_tail);
        compute_k_loop(conf_->N_tail);
        bl(done);

        L(not_N_tail);
    }
    compute_k_loop(conf_->N_blk);
    L(done);

    postamble();
}

template <cpu_isa_t isa>
struct jit_brgemm_matmul_copy_b_transposed_t
    : public jit_brgemm_matmul_copy_b_t,
      public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_matmul_copy_b_transposed_t)

    jit_brgemm_matmul_copy_b_transposed_t(const brgemm_matmul_conf_t *conf)
        : jit_brgemm_matmul_copy_b_t(conf)
        , src_typesize_(conf_->b_dt_sz)
        , tr_src_typesize_(conf_->tr_b_dt_sz)
        , block_step_(simd_elems(conf_->wei_dt, isa))
        , src_stride_(conf_->wei_tag == format_tag::adbc
                          ? conf_->copy_B_wei_stride
                          : conf_->K * src_typesize_)
        , tr_src_stride_(conf_->LDB * tr_src_typesize_)
        , N_tail_(conf_->N % block_step_) {}

    void operator()(ctx_t *ctx) override { jit_generator_t::operator()(ctx); }
    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

private:
    using reg64_t = const Xbyak_aarch64::XReg;

    const int src_typesize_;
    const int tr_src_typesize_;
    const int block_step_;

    const dim_t src_stride_, tr_src_stride_, N_tail_;

    reg64_t reg_src_base = x1;
    reg64_t reg_tr_src_base = x2;

    reg64_t reg_K_iters = x8;
    reg64_t reg_N_iters = x9;
    reg64_t reg_src = x10;
    reg64_t reg_tr_src = x11;

    PReg p_0F = p4;
    PReg p_tail = p5;
    const ZReg z_tmp = ZReg(31);

    void load_src_row(int i, int nrows, int columns_tail) {
        if (i >= nrows) {
            eor(ZReg(i).d, ZReg(i).d, ZReg(i).d);
            return;
        }

        add_imm(X_DEFAULT_ADDR, reg_src, i * src_stride_, X_TMP_0);
        if (columns_tail > 0)
            ld1w(ZReg(i).s, p_tail / T_z, ptr(X_DEFAULT_ADDR));
        else
            ldr(ZReg(i), ptr(X_DEFAULT_ADDR));
    }

    void transpose_s_pair(int i, int j) {
        trn1(z_tmp.s, ZReg(i).s, ZReg(j).s);
        trn2(ZReg(j).s, ZReg(i).s, ZReg(j).s);
        mov(ZReg(i).d, z_tmp.d);
    }

    void transpose_d_pair(int i, int j) {
        trn1(z_tmp.d, ZReg(i).d, ZReg(j).d);
        trn2(ZReg(j).d, ZReg(i).d, ZReg(j).d);
        mov(ZReg(i).d, z_tmp.d);
    }

    // xbyak_aarch64 does not expose the 128-bit trn form, so
    // exchange quadwords using the original splice sequence
    void transpose_q_pair(int i, int j) {
        mov(z_tmp.d, ZReg(i).d);
        ext(z_tmp.b, ZReg(i).b, 16);
        splice(ZReg(i).s, p_0F, ZReg(j).s);
        mov(ZReg(j).s, p_0F / T_m, z_tmp.s);
    }

    void copy_row_x_col(int nrows, int ncolumns);
    void compute_K_loop(bool is_N_tail, int curr_K_tail);
    void compute_N_loop(int curr_K_tail);
    void generate() override;
};

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::copy_row_x_col(
        int nrows, int ncolumns) {
    const int columns_tail = ncolumns % block_step_;
    if (columns_tail > 0) set_preg(p_tail.s, columns_tail, X_TMP_0);

    for (int i = 0; i < block_step_; i += 2) {
        load_src_row(i, nrows, columns_tail);
        load_src_row(i + 1, nrows, columns_tail);
        transpose_s_pair(i, i + 1);
    }
    for (int i = 0; i < block_step_; i += 4) {
        transpose_d_pair(i, i + 2);
        transpose_d_pair(i + 1, i + 3);
    }

    switch (simd_bytes(isa)) {
        case util::SVE_128: break;
        case util::SVE_256:
            for (int i = 0; i < block_step_ / 2; ++i)
                transpose_q_pair(i, i + block_step_ / 2);
            break;
        default: assert(!"unsupported SVE vector length");
    }

    for (int i = 0; i < block_step_; ++i) {
        add_imm(X_DEFAULT_ADDR, reg_tr_src, i * tr_src_stride_, X_TMP_0);
        str(ZReg(i), ptr(X_DEFAULT_ADDR));
    }
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::compute_K_loop(
        bool is_N_tail, int curr_K_tail) {
    Label K_loop, K_loop_tail_or_done;
    ldr_imm(reg_K_iters, param1, GET_OFF(current_K_iters));

    mov(reg_src, reg_src_base);
    mov(reg_tr_src, reg_tr_src_base);
    if (curr_K_tail > 0) {
        cmp_imm(reg_K_iters, block_step_, X_TMP_0);
        b(LT, K_loop_tail_or_done);
    }

    const int nrows = is_N_tail ? N_tail_ : block_step_;
    L(K_loop);
    copy_row_x_col(nrows, block_step_);
    add_imm(reg_src, reg_src, block_step_ * src_typesize_, X_TMP_0);
    add_imm(reg_tr_src, reg_tr_src, block_step_ * tr_src_stride_, X_TMP_0);

    sub_imm(reg_K_iters, reg_K_iters, block_step_, X_TMP_0);
    cmp_imm(reg_K_iters, block_step_, X_TMP_0);
    b(GE, K_loop);

    L(K_loop_tail_or_done);
    if (curr_K_tail > 0) copy_row_x_col(nrows, curr_K_tail);
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::compute_N_loop(
        int curr_K_tail) {
    Label N_loop, N_loop_tail_or_done;
    if (N_tail_ > 0) {
        cmp_imm(reg_N_iters, block_step_, X_TMP_0);
        b(LT, N_loop_tail_or_done);
    }

    L(N_loop);
    compute_K_loop(false, curr_K_tail);
    add_imm(reg_src_base, reg_src_base, block_step_ * src_stride_, X_TMP_0);
    add_imm(reg_tr_src_base, reg_tr_src_base, block_step_ * tr_src_typesize_,
            X_TMP_0);

    sub_imm(reg_N_iters, reg_N_iters, block_step_, X_TMP_0);
    cmp_imm(reg_N_iters, block_step_, X_TMP_0);
    b(GE, N_loop);

    L(N_loop_tail_or_done);
    if (N_tail_ > 0) {
        Label N_loop_done;
        cmp_imm(reg_N_iters, 0, X_TMP_0);
        b(LE, N_loop_done);

        compute_K_loop(true, curr_K_tail);
        L(N_loop_done);
    }
}

template <cpu_isa_t isa>
void jit_brgemm_matmul_copy_b_transposed_t<isa>::generate() {
    preamble();

    switch (simd_bytes(isa)) {
        case util::SVE_128: break;
        case util::SVE_256: set_preg(p_0F.s, 4, X_TMP_0); break;
        default: assert(!"unsupported SVE vector length");
    }

    ldr_imm(reg_src_base, param1, GET_OFF(src));
    ldr_imm(reg_tr_src_base, param1, GET_OFF(tr_src));
    ldr_imm(reg_K_iters, param1, GET_OFF(current_K_iters));
    ldr_imm(reg_N_iters, param1, GET_OFF(current_N_blk));

    const auto K_blk_tail = nstl::min(conf_->K, conf_->K_blk) % block_step_;
    const auto K_tail_tail = (conf_->K % conf_->K_blk) % block_step_;

    Label compute_body_done;
    if (conf_->K_tail > 0 && K_blk_tail != K_tail_tail) {
        Label not_K_tail;
        cmp_imm(reg_K_iters, conf_->K_blk, X_TMP_0);
        b(EQ, not_K_tail);
        compute_N_loop(K_tail_tail);
        b(compute_body_done);

        L(not_K_tail);
    }

    compute_N_loop(K_blk_tail);
    L(compute_body_done);
    postamble();
}

status_t create_brgemm_matmul_copy_b(
        std::unique_ptr<jit_brgemm_matmul_copy_b_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {
    const bool is_B_transposed
            = one_of(conf->wei_tag, ba, acb, abdc, adbc, abced, abcdfe, abcdegf,
                    abcdefhg, abcdefgih, abcdefghji, abcdefghikj, abcdefghijlk);
    const bool is_bf16
            = everyone_is(data_type::bf16, conf->src_dt, conf->wei_dt);
    const bool is_f32 = everyone_is(data_type::f32, conf->src_dt, conf->wei_dt);

    const bool is_f16 = everyone_is(data_type::f16, conf->src_dt, conf->wei_dt);

    if (is_B_transposed) {
        if (!one_of(conf->isa, sve_128, sve_256) || !is_f32)
            return status::unimplemented;
        CHECK(safe_ptr_assign(copy_ker,
                new jit_brgemm_matmul_copy_b_transposed_t<sve>(conf)));
    } else {
        if (is_bf16 || is_f16 || conf->is_bf32) {
            return status::unimplemented;
        } else if (is_f32) {
            CHECK(safe_ptr_assign(
                    copy_ker, new jit_brgemm_matmul_copy_b_f32_t(conf)));
        } else {
            return status::unimplemented;
        }
    }

    return copy_ker->create_kernel();
}

status_t create_brgemm_matmul_copy_a(
        std::unique_ptr<jit_brgemm_matmul_copy_a_t> &copy_ker,
        const brgemm_matmul_conf_t *conf) {

    assert(!"unreachable");
    return status::unimplemented;
}

template struct jit_brgemm_matmul_copy_b_transposed_t<sve>;

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
