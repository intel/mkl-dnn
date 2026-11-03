/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
* Copyright 2024-2026 Arm Ltd. and affiliates
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

#include <memory>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose_msg.hpp"

#include "cpu/matmul/matmul_utils.hpp"

#include "cpu/aarch64/brgemm/brgemm_utils.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/aarch64/matmul/brgemm_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace dnnl::impl::cpu::matmul;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;
namespace {

int get_brg_kernel_index(const brgemm_matmul_conf_t &bgmmc, bool is_bs_tail,
        bool do_initialization, int m_ker_idx, bool is_N_tail, bool is_K_tail,
        int bs) {
    const int max_m_ker_idx
            = bgmmc.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;
    if (m_ker_idx >= max_m_ker_idx) return -1;

    auto vM = m_ker_idx > 0
            ? (bgmmc.is_runtime_M ? dynamic_m_tails[m_ker_idx - 1]
                                  : bgmmc.M_tail)
            : bgmmc.M_blk;
    auto vN = (is_N_tail) ? bgmmc.N_tail : bgmmc.N_blk;
    auto vK = (is_K_tail) ? bgmmc.K_tail : bgmmc.K_blk;
    if (vM == 0 || vN == 0 || vK == 0 || bs == 0 || bgmmc.LDA < vK
            || bgmmc.LDB < vN || bgmmc.LDC < vN)
        return -1;

    int idx = 16 * m_ker_idx + 8 * (int)is_bs_tail + 4 * (int)do_initialization
            + 2 * (int)is_N_tail + (int)is_K_tail;
    assert(idx < max_num_brg_kernels_matmul);
    return idx;
}

int get_brg_batchsize(
        const brgemm_matmul_conf_t &bgmmc, bool is_bs_tail, bool is_K_tail) {
    auto bs = is_K_tail  ? 1
            : is_bs_tail ? bgmmc.brgemm_batch_tail_size
                         : bgmmc.brgemm_batch_size;
    return bs;
}

} // anonymous namespace

template <cpu_isa_t isa>
int brgemm_matmul_t<isa>::pd_t::get_brg_kernel_idx(bool is_bs_tail,
        bool do_initialization, int m_ker_idx, bool is_N_tail,
        bool is_K_tail) const {
    int bs = get_brg_batchsize(bgmmc_, is_bs_tail, is_K_tail);
    return get_brg_kernel_index(bgmmc_, is_bs_tail, do_initialization,
            m_ker_idx, is_N_tail, is_K_tail, bs);
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::pd_t::init(const engine_t *engine) {
    const auto src_dt = src_md_.data_type;
    const auto wei_dt = weights_md_.data_type;
    const auto dst_dt = dst_md_.data_type;

    const bool is_f32 = everyone_is(f32, src_dt, wei_dt, dst_dt);
    const bool is_int8 = one_of(src_dt, u8, s8) && wei_dt == s8
            && one_of(dst_dt, u8, s8, s32, f32, bf16);
    const bool is_bf16
            = everyone_is(bf16, src_dt, wei_dt) && one_of(dst_dt, bf16, f32);
    const bool is_f16
            = everyone_is(f16, src_dt, wei_dt) && one_of(dst_dt, f16, f32);

    auto check_bias = [&]() -> bool {
        const auto bia_dt = weights_md(1)->data_type;

        const bool is_bia_dt_correct
                = IMPLICATION(is_int8 == true,
                          one_of(bia_dt, f32, s32, s8, u8, bf16))
                && IMPLICATION(!is_int8, one_of(bia_dt, f32, src_dt));
        return IMPLICATION(with_bias(), is_bia_dt_correct && is_bias_1xN());
    };

    auto check_attr_scales = [&]() -> status_t {
        const std::vector<int> supported_args
                = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        CHECK(attr_scales_ok(engine, supported_args));
        if (!attr()->scales_.has_default_values(DNNL_ARG_SRC)
                && !attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)
                && attr()->scales_.get_mask(DNNL_ARG_WEIGHTS) > 0) {
            // This case requires scratchpad
            VDISPATCH_MATMUL(
                    !is_runtime_value(N()), VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        VDISPATCH_MATMUL(attr()->post_ops_.sum_with_default_dt(),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        return status::success;
    };

    auto check_attr_zero_points = [&]() -> bool {
        const auto &zp = attr()->zero_points_;
        static const std::vector<int> supported_args {
                DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        for (int arg : supported_args) {
            if (!zp.has_default_values(arg)) {
                const int mask = zp.get_mask(arg);
                if (mask > 0) return false;
            }
        }
        return true;
    };

    // The current version supports runtime value for M dimension in the case
    // of 2d problems only and do not support any runtime strides for B and C
    // tensors. A tensor strides correctness check is performed in
    // init_brgemm_matmul_conf()
    const bool no_dynamic_strides_for_B_and_C
            = !memory_desc_wrapper(weights_md_).has_runtime_strides()
            && !memory_desc_wrapper(dst_md_).has_runtime_strides();
    const bool problem_dt_correct = is_int8 || is_bf16 || is_f32 || is_f16;
    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_NONTRIVIAL_STRIDE);
    VDISPATCH_MATMUL(mayiuse(isa), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_MATMUL(problem_dt_correct, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_MATMUL(
            IMPLICATION(is_bf16, mayiuse_bf16()), VERBOSE_UNSUPPORTED_ISA);
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(
            no_dynamic_strides_for_B_and_C, VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_MATMUL(
            attr()->has_default_values(primitive_attr_t::skip_mask_t::scales
                            | primitive_attr_t::skip_mask_t::zero_points
                            | primitive_attr_t::skip_mask_t::post_ops
                            | primitive_attr_t::skip_mask_t::sum_dt,
                    dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_MATMUL(attr()->post_ops_.check_sum_consistency(dst_dt, is_int8),
            VERBOSE_UNSUPPORTED_DT);
    CHECK(check_attr_scales());
    VDISPATCH_MATMUL(check_attr_zero_points(), VERBOSE_UNSUPPORTED_ZP_CFG);
    VDISPATCH_MATMUL(check_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    CHECK(init_brgemm_matmul_conf(isa, bgmmc_, *desc(), src_md_, weights_md_,
            dst_md_, bias_md_, attr_));

    const float alpha = 1.0;
    const float beta = 1.0;
    const float beta_init = 0.0;

    const int max_m_ker_idx
            = bgmmc_.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;

    const auto backup_isa = isa;
    for_(int i_bs = 0; i_bs < 2; i_bs++)
    for_(int i_init = 0; i_init < 2; i_init++)
    for_(int i_M = 0; i_M < max_m_ker_idx; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for (int i_K = 0; i_K < 2; i_K++) {
        auto vbeta = (i_init) ? beta_init : beta;
        auto vM = (i_M) == 0 ? bgmmc_.M_blk
                             : (bgmmc_.is_runtime_M ? dynamic_m_tails[i_M - 1]
                                                    : bgmmc_.M_tail);
        auto vN = (i_N) ? bgmmc_.N_tail : bgmmc_.N_blk;
        auto vK = (i_K) ? bgmmc_.K_tail : bgmmc_.K_blk;

        int idx = get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K);
        if (idx < 0) continue;
        brgemm_desc_t &brg = brg_descs_[idx];
        auto LDA = i_K && bgmmc_.use_buffer_a_tail_only
                ? (dim_t)bgmmc_.wei_k_blk
                : bgmmc_.LDA;
        const auto kernel_isa = i_M == max_m_ker_idx - 1 ? backup_isa : isa;
        // In the case of BA 1xK gemmv, we set the layout to column major, so there is no
        // need to transpose the weights.
        const auto layout = (one_of(bgmmc_.wei_tag, ba) && bgmmc_.M == 1)
                ? brgemm_col_major
                : brgemm_row_major;
        CHECK(brgemm_desc_init(&brg, kernel_isa, bgmmc_.brg_type, bgmmc_.src_dt,
                bgmmc_.wei_dt, false, false, layout, alpha, vbeta, LDA,
                bgmmc_.LDB, bgmmc_.LDC, vM, vN, vK));

        auto LDD = bgmmc_.LDD;
        CHECK(brgemm_desc_set_postops(
                &brg, attr(), &dst_md_, LDD, bgmmc_.bia_dt));

        brgemm_attr_t brgattr;
        brgattr.generate_skip_accumulation
                = bgmmc_.post_ops_applicable && bgmmc_.nthr_k > 1;
        if (bgmmc_.use_mmla) {
            brgemm_utils::set_mmla_attr(brgattr, vM);
            brgattr.max_bs = get_brg_batchsize(bgmmc_, i_bs, i_K);
        }

        CHECK(brgemm_desc_set_attr(&brg, brgattr));

        CHECK(brgemm_desc_finalize(&brg));

        bgmmc_.wsp_tile_per_thr_bytes = nstl::max(
                brg.get_wsp_buffer_size(), bgmmc_.wsp_tile_per_thr_bytes);
    }

    auto scratchpad = scratchpad_registry().registrar();
    init_scratchpad(scratchpad, bgmmc_);

    constexpr size_t page_alignment_bytes = 4096;
    if (bgmmc_.with_scales) {
        const size_t scales_size = bgmmc_.is_oscale_per_n
                ? nstl::max(static_cast<size_t>(N()),
                          scale_utils::scales_simd_w(isa))
                : scale_utils::scales_simd_w(isa);

        scratchpad.template book<float>(
                key_precomputed_scales, scales_size, page_alignment_bytes);
    }

    if (bgmmc_.with_dst_scales) {
        scratchpad.template book<float>(key_matmul_dst_scales,
                scale_utils::scales_simd_w(isa), page_alignment_bytes);
    }

    const bool is_B_transposed = one_of(bgmmc_.wei_tag, abdc, ba, acb, adbc,
            abced, abcdfe, abcdegf, abcdefhg, abcdefgih, abcdefghji,
            abcdefghikj, abcdefghijlk);

    const bool is_A_transposed = one_of(bgmmc_.src_tag, abdc, ba, acb, adbc,
            abced, abcdfe, abcdegf, abcdefhg, abcdefgih, abcdefghji,
            abcdefghikj, abcdefghijlk);

    VDISPATCH_MATMUL(!is_A_transposed, VERBOSE_UNSUPPORTED_TAG_S, "src");

    if ((mayiuse(sve_512) && is_B_transposed)) return status::unimplemented;

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::init(engine_t *engine) {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const int max_m_ker_idx
            = bgmmc.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;
    for_(int i_bs = 0; i_bs < 2; i_bs++)
    for_(int i_M = 0; i_M < max_m_ker_idx; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for_(int i_K = 0; i_K < 2; i_K++)
    for (int i_init = 0; i_init < 2; i_init++) {
        int idx = pd()->get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K);
        if (idx < 0) continue;

        brgemm_kernel_t *ker = nullptr;
        CHECK(brgemm_kernel_create(&ker, pd()->get_brg_desc(idx)));
        CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
    }

    if (bgmmc.use_buffer_b) {
        CHECK(create_brgemm_matmul_copy_b(copy_B_kernel_, &bgmmc));
    }

    if (bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only)
        CHECK(create_brgemm_matmul_copy_a(copy_A_kernel_, &bgmmc));

    if (bgmmc.nthr_k > 1 && bgmmc.acc_dt == f32) {
        CHECK(safe_ptr_assign(
                acc_ker_f32_, new cpu_accumulator_1d_t<data_type::f32>()));
        CHECK(acc_ker_f32_->create_kernel());
    } else if (bgmmc.nthr_k > 1 && bgmmc.acc_dt == s32) {
        CHECK(safe_ptr_assign(
                acc_ker_s32_, new cpu_accumulator_1d_t<data_type::s32>()));
        CHECK(acc_ker_s32_->create_kernel());
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::execute_body(const exec_ctx_t &ctx) const {
    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    matmul_helper_t helper(src_d, weights_d, dst_d);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    float *oscales = bgmmc.with_scales
            ? scratchpad.template get<float>(key_precomputed_scales)
            : nullptr;

    float *inv_dst_scales = bgmmc.with_dst_scales
            ? scratchpad.template get<float>(key_matmul_dst_scales)
            : nullptr;

    if (bgmmc.with_scales || bgmmc.with_dst_scales) {
        parallel(1, [= COMPAT_THIS_CAPTURE](int, int) {
            scale_utils::precompute_oscales(isa, oscales, pd()->attr(),
                    src_scales, wei_scales, pd()->N(), 1.f);

            scale_utils::precompute_inv_dst_scales(
                    isa, inv_dst_scales, pd()->attr(), dst_scales);
        });
    }

    const auto &brgmm_ctx_ptr = std::make_shared<brg_matmul_exec_ctx_t>(
            ctx, pd(), oscales, inv_dst_scales, helper);

    const int num_threads
            = brgmm_ctx_ptr->get_num_threads_for_parallelization();

    parallel(num_threads,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &brgmm_ctx = *brgmm_ctx_ptr;

        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        const int M_chunks = brgmm_ctx.get_M_chunks();
        const int M_chunk_size = brgmm_ctx.get_M_chunk_size();
        const int M_chunk_tail = brgmm_ctx.get_M_chunk_tail();

        const auto &bgmmc = pd()->get_brgemm_matmul_conf();
        const bool use_buffer_a
                = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;

        if (ithr_bmn < 0 || ithr_k < 0) return;
        int start {0}, end {0};
        balance211(brgmm_ctx.get_parallel_work_amount(),
                brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn, start, end);
        int kc_start {0}, kc_end {bgmmc.K_chunks};
        if (brgmm_ctx.parallel_reduction_is_used())
            balance211((int)bgmmc.K_chunks, brgmm_ctx.get_num_threads_for_k(),
                    ithr_k, kc_start, kc_end);

        int prev_ker_idx = -1;

        int b {0}, mc {0}, nc {0};
        nd_iterator_init(
                start, b, bgmmc.batch, mc, M_chunks, nc, bgmmc.N_chunks);
        int mc_prev = -1;
        int nc_prev = -1;
        int b_prev = -1;
        while (start < end) {
            auto m_start = mc * M_chunk_size;
            const bool m_chunk_tail = mc == M_chunks - 1 && M_chunk_tail > 0;
            auto m_end = m_start + (m_chunk_tail ? M_chunk_tail : M_chunk_size);
            auto n_start = nc * bgmmc.N_chunk_size;
            auto n_end = nstl::min(
                    (nc + 1) * bgmmc.N_chunk_size, bgmmc.num_N_blocks);
            int kc_prev = -1;
            for_(int kc = kc_start; kc < kc_end; kc++)
            for (int nb = n_start; nb < n_end; nb++) {
                const bool skip_copy_b = nc_prev == nc && kc_prev == kc
                        && (b_prev == b
                                || bgmmc.bcast_B_desc
                                           .bcast_across_all_batch_dims);
                if (bgmmc.use_buffer_b && !skip_copy_b)
                    copy_b_chunk_in_buffer(brgmm_ctx, ithr, b, nb, kc);
                for (int mb = m_start; mb < m_end; mb++) {
                    const bool skip_copy_a = mc_prev == mc && kc_prev == kc
                            && (b_prev == b
                                    || bgmmc.bcast_A_desc
                                               .bcast_across_all_batch_dims);
                    if (use_buffer_a && nb == n_start && !skip_copy_a)
                        copy_a_chunk_in_buffer(brgmm_ctx, ithr, b, mb, kc);
                    compute_kernel(brgmm_ctx, ithr, b, mb, nb, kc,
                            kc == kc_start, prev_ker_idx);
                }
                kc_prev = kc;
            }
            mc_prev = mc;
            nc_prev = nc;
            b_prev = b;
            ++start;
            nd_iterator_step(b, bgmmc.batch, mc, M_chunks, nc, bgmmc.N_chunks);
        }
    });

    maybe_reduce_partial_results_and_apply_postops(brgmm_ctx_ptr);

    return status::success;
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::compute_kernel(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, int b_idx,
        int m_blk_idx, int n_blk_idx, int k_chunk_idx, bool do_init,
        int &prev_ker_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const auto addr_batch = brgmm_ctx.get_batch_elem_ptr(ithr);

    const dim_t m = brgmm_ctx.get_M_idx(m_blk_idx);
    const bool need_copy_d = brgmm_ctx.copy_d_required(m_blk_idx);
    const int n = n_blk_idx * bgmmc.N_blk;
    const int k_blk_idx = k_chunk_idx * bgmmc.brgemm_batch_size;

    const dim_t M = brgmm_ctx.get_M();
    const int m_ker_idx = brgmm_ctx.get_M_kernel_idx(m_blk_idx);
    const bool is_N_tail = (bgmmc.N - n < bgmmc.N_blk);
    const bool is_last_K_chunk = brgmm_ctx.is_last_K_chunk(k_chunk_idx);

    const int remaining_k_blks
            = (bgmmc.use_buffer_a ? utils::rnd_up(bgmmc.K, bgmmc.K_blk)
                                  : bgmmc.K)
            - k_chunk_idx * bgmmc.K_chunk_elems;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);
    const bool is_K_tail
            = is_last_K_chunk && (gemm_batch * bgmmc.K_blk) != remaining_k_blks;
    auto is_bs_tail = (gemm_batch != bgmmc.brgemm_batch_size);
    const int brg_ker_idx = pd()->get_brg_kernel_idx(
            is_bs_tail, do_init, m_ker_idx, is_N_tail, false);
    const auto ptr_bias = brgmm_ctx.get_bias_ptr(n);
    auto ptr_D = need_copy_d ? brgmm_ctx.get_buf_D_ptr(m_blk_idx, n_blk_idx)
                             : brgmm_ctx.get_data_C_ptr(b_idx, m, n);
    auto ptr_C = (bgmmc.use_buffer_c)
            ? brgmm_ctx.get_buf_C_ptr(ithr, m_blk_idx, n_blk_idx)
            : ptr_D;

    const auto zp_comp_a
            = brgmm_ctx.get_zp_a_compensation_ptr(ithr, b_idx, n_blk_idx);
    const auto zp_comp_b
            = brgmm_ctx.get_zp_b_compensation_result_ptr(ithr, m_blk_idx);
    const auto zp_c_val_ptr = brgmm_ctx.get_zp_c_val_ptr();
    const auto &post_ops_binary_rhs_arg_vec
            = brgmm_ctx.get_post_ops_binary_rhs_arg_vec();
    const bool post_ops_applicable = bgmmc.post_ops_applicable
            && (brgmm_ctx.get_num_threads_for_k() <= 1 || bgmmc.K_chunks == 1);

    if (need_copy_d && bgmmc.with_sum)
        brgmm_ctx.copy_dst_values_to_buffer(b_idx, m_blk_idx, n_blk_idx);

    if (gemm_batch > 0 && brg_ker_idx >= 0) {
        const auto brg_kernel = brg_kernels_[brg_ker_idx].get();
        assert(brg_kernel != nullptr);

        brgmm_ctx.init_brgemm_batch_elements_values(
                ithr, 0, gemm_batch, b_idx, m_blk_idx, k_blk_idx, n_blk_idx);

        if (post_ops_applicable && is_last_K_chunk && !is_K_tail) {
            void *scratch = static_cast<void *>(
                    brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx));

            const size_t dst_row_logical_off
                    = brgmm_ctx.get_M_idx(m_blk_idx, true);
            const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                    ? b_idx / bgmmc.batch_without_first_dim
                    : 0;
            const size_t first_mb_matrix_addr_off
                    = batch_first_dim_idx * (M * bgmmc.N)
                    + (dst_row_logical_off * bgmmc.N + n);
            const char *dst_anchor_point
                    = brgmm_ctx.get_dst_anchor_point_for_binary_post_ops(
                            b_idx, m_blk_idx, n_blk_idx);
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(ptr_bias),
                    brgmm_ctx.get_oscales_ptr(n),
                    post_ops_binary_rhs_arg_vec.data(), static_cast<size_t>(n),
                    dst_row_logical_off, dst_anchor_point,
                    first_mb_matrix_addr_off,
                    static_cast<const void *>(zp_comp_a),
                    static_cast<const void *>(zp_comp_b),
                    static_cast<const void *>(zp_c_val_ptr), false, 1, false,
                    false, brgmm_ctx.get_dst_scales_ptr()};

            brgemm_kernel_execute_postops(brg_kernel, gemm_batch, addr_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch);
        } else {
            brgemm_kernel_execute(
                    brg_kernel, gemm_batch, addr_batch, (void *)ptr_C, nullptr);
        }
    }
    if (is_K_tail) {
        brgmm_ctx.init_brgemm_batch_elements_values(
                ithr, gemm_batch, 1, b_idx, m_blk_idx, k_blk_idx, n_blk_idx);

        const bool use_init_ker = (do_init && gemm_batch == 0);
        const int brg_ker_idx = pd()->get_brg_kernel_idx(
                false, use_init_ker, m_ker_idx, is_N_tail, true);
        assert(brg_ker_idx >= 0);
        const auto brg_kernel_k_tail = brg_kernels_[brg_ker_idx].get();

        if (post_ops_applicable) {
            void *scratch = static_cast<void *>(
                    brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx));

            const size_t dst_row_logical_off
                    = brgmm_ctx.get_M_idx(m_blk_idx, true);
            const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                    ? b_idx / bgmmc.batch_without_first_dim
                    : 0;
            const size_t first_mb_matrix_addr_off
                    = batch_first_dim_idx * (M * bgmmc.N)
                    + (dst_row_logical_off * bgmmc.N + n);
            const char *dst_anchor_point
                    = brgmm_ctx.get_dst_anchor_point_for_binary_post_ops(
                            b_idx, m_blk_idx, n_blk_idx);
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(ptr_bias),
                    brgmm_ctx.get_oscales_ptr(n),
                    post_ops_binary_rhs_arg_vec.data(), static_cast<size_t>(n),
                    dst_row_logical_off, dst_anchor_point,
                    first_mb_matrix_addr_off,
                    static_cast<const void *>(zp_comp_a),
                    static_cast<const void *>(zp_comp_b),
                    static_cast<const void *>(zp_c_val_ptr), false, 1, false,
                    false, brgmm_ctx.get_dst_scales_ptr()};

            brgemm_kernel_execute_postops(brg_kernel_k_tail, 1, addr_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch);
        } else {
            brgemm_kernel_execute(
                    brg_kernel_k_tail, 1, addr_batch, (void *)ptr_C, nullptr);
        }
    }

    if (need_copy_d)
        brgmm_ctx.copy_dst_values_from_buffer(b_idx, m_blk_idx, n_blk_idx);
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::maybe_reduce_partial_results_and_apply_postops(
        const std::shared_ptr<brg_matmul_exec_ctx_t> &brgmm_ctx_ptr) const {
    if (!brgmm_ctx_ptr->parallel_reduction_is_used()) return;

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const int num_threads
            = brgmm_ctx_ptr->get_num_threads_for_parallelization();

    parallel(num_threads,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &brgmm_ctx = *brgmm_ctx_ptr;

        const int nthr_k = brgmm_ctx.get_num_threads_for_k();
        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;

        const int num_reduction_buffers = nstl::min(nthr_k, bgmmc.K_chunks);

        int bmn_start {0}, bmn_end {0};
        int start {0}, end {0};
        balance211(brgmm_ctx.get_parallel_work_amount(),
                brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn, bmn_start,
                bmn_end);
        balance211(bmn_end - bmn_start, nthr_k, ithr_k, start, end);

        int b {0}, mc {0}, nc {0};

        const dim_t M = brgmm_ctx.get_M();
        const int M_chunks = brgmm_ctx.get_M_chunks();
        const int M_chunk_size = brgmm_ctx.get_M_chunk_size();
        const int M_chunk_tail = brgmm_ctx.get_M_chunk_tail();
        assert(bgmmc.batch == 1);
        nd_iterator_init(bmn_start + start, b, bgmmc.batch, mc, M_chunks, nc,
                bgmmc.N_chunks);
        while (start < end) {
            auto mb_start = mc * M_chunk_size;
            const bool m_chunk_tail = mc == M_chunks - 1 && M_chunk_tail > 0;
            auto mb_end
                    = mb_start + (m_chunk_tail ? M_chunk_tail : M_chunk_size);
            auto nb_start = nc * bgmmc.N_chunk_size;
            auto nb_end = nstl::min(
                    (nc + 1) * bgmmc.N_chunk_size, bgmmc.num_N_blocks);
            for (int mb = mb_start; mb < mb_end; mb++) {
                const int curr_M_blk = brgmm_ctx.get_M_kernel_size(mb);
                const int m_ker_idx = brgmm_ctx.get_M_kernel_idx(mb);
                const int curr_N_chunk_size
                        = nstl::min(bgmmc.N, nb_end * bgmmc.N_blk)
                        - nb_start * bgmmc.N_blk;
                char *buf_reduced_base = brgmm_ctx.get_buf_C_par_reduction_ptr(
                        0, mb, nb_start);
                const size_t m_offset = bgmmc.LDC * bgmmc.acc_dt_sz;
                for (int r = 1; r < num_reduction_buffers; r++) {
                    const char *buf_to_reduce_base
                            = brgmm_ctx.get_buf_C_par_reduction_ptr(
                                    r, mb, nb_start);
                    for (int m = 0; m < curr_M_blk; m++) {
                        accumulate(buf_reduced_base + m * m_offset,
                                buf_to_reduce_base + m * m_offset,
                                curr_N_chunk_size);
                    }
                }
                if (bgmmc.post_ops_applicable) {
                    for (int nb = nb_start; nb < nb_end; nb++) {
                        const bool is_N_tail
                                = (bgmmc.N - nb * bgmmc.N_blk < bgmmc.N_blk);
                        const int brg_ker_idx = pd()->get_brg_kernel_idx(
                                false, false, m_ker_idx, is_N_tail, false);
                        const auto brg_kernel = brg_kernels_[brg_ker_idx].get();
                        const int m = brgmm_ctx.get_M_idx(mb);
                        const int n = nb * bgmmc.N_blk;
                        const auto ptr_bias = brgmm_ctx.get_bias_ptr(n);
                        auto ptr_D = brgmm_ctx.get_data_C_ptr(b, m, n);
                        auto ptr_C = brgmm_ctx.get_buf_C_par_reduction_ptr(
                                0, mb, nb);

                        // TODO: support reduction for zp/s8s8 compensations
                        // computed in copy routines
                        const auto zp_comp_a
                                = brgmm_ctx.get_zp_a_compensation_ptr(
                                        ithr, b, nb);
                        const auto zp_comp_b
                                = brgmm_ctx.get_zp_b_compensation_result_ptr(
                                        ithr, mb);
                        const auto zp_c_val_ptr = brgmm_ctx.get_zp_c_val_ptr();
                        const auto &post_ops_binary_rhs_arg_vec
                                = brgmm_ctx.get_post_ops_binary_rhs_arg_vec();

                        const size_t dst_row_logical_off
                                = brgmm_ctx.get_M_idx(mb, true);
                        const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                                ? b / bgmmc.batch_without_first_dim
                                : 0;
                        const size_t first_mb_matrix_addr_off
                                = batch_first_dim_idx * (M * bgmmc.N)
                                + (m * bgmmc.N + n);
                        // apply post-ops and convert to dst data type only
                        constexpr bool skip_accumulation = true;
                        const char *dst_anchor_point
                                = brgmm_ctx
                                          .get_dst_anchor_point_for_binary_post_ops(
                                                  b, mb, nb);
                        const brgemm_post_ops_data_t post_ops_data {
                                static_cast<const void *>(ptr_bias),
                                brgmm_ctx.get_oscales_ptr(n),
                                post_ops_binary_rhs_arg_vec.data(),
                                static_cast<size_t>(n), dst_row_logical_off,
                                dst_anchor_point, first_mb_matrix_addr_off,
                                static_cast<const void *>(zp_comp_a),
                                static_cast<const void *>(zp_comp_b),
                                static_cast<const void *>(zp_c_val_ptr),
                                skip_accumulation, 1, false, false,
                                brgmm_ctx.get_dst_scales_ptr()};

                        brgemm_kernel_execute_postops(brg_kernel, 0, nullptr,
                                (void *)ptr_C, (void *)ptr_D, post_ops_data,
                                nullptr);
                    }
                }
            }
            ++start;
            nd_iterator_step(b, bgmmc.batch, mc, M_chunks, nc, bgmmc.N_chunks);
        }
    });
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::copy_a_chunk_in_buffer(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, int b_idx,
        int m_blk_idx, int k_chunk_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();

    auto ctx = jit_brgemm_matmul_copy_a_t::ctx_t();
    const int k_start = k_chunk_idx * bgmmc.K_chunk_elems;
    const bool is_K_tail
            = brgmm_ctx.is_last_K_chunk(k_chunk_idx) && bgmmc.K_tail > 0;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);
    const int gemm_batch_iters = bgmmc.use_buffer_a_tail_only ? 0 : gemm_batch;

    const dim_t m = brgmm_ctx.get_M_idx(m_blk_idx, true);

    ctx.current_M_blk = brgmm_ctx.get_M_kernel_size(m_blk_idx);
    ctx.zp_b_compensation_buffer_ptr
            = (void *)brgmm_ctx.get_zp_b_compensation_buffer_ptr(
                    ithr, m_blk_idx);
    ctx.zp_a_compensation_result_ptr
            = (void *)brgmm_ctx.get_zp_b_compensation_result_ptr(
                    ithr, m_blk_idx);
    ctx.zp_b_neg_value_ptr = (void *)brgmm_ctx.get_zp_b_neg_val_ptr();
    ctx.zp_ab_comp_ptr = (void *)brgmm_ctx.get_zp_ab_mixed_comp_ptr();
    ctx.dynamic_src_ld = brgmm_ctx.get_src_stride();

    for (int gb = 0; gb < gemm_batch_iters; gb++) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_A_ptr(b_idx, m, k);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_A_ptr(ithr, m_blk_idx, gb);
        ctx.current_K_blk = nstl::min(bgmmc.K_blk, bgmmc.K);
        ctx.current_K_start = k;

        (*copy_A_kernel_)(&ctx);
    }
    if (is_K_tail) {
        const auto K_tail = bgmmc.K % bgmmc.K_blk;
        const int k = k_start + gemm_batch * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_A_ptr(b_idx, m, k);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_A_ptr(
                ithr, m_blk_idx, gemm_batch_iters);
        ctx.current_K_blk = K_tail;
        ctx.current_K_start = k;

        (*copy_A_kernel_)(&ctx);
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::copy_b_chunk_in_buffer(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, int b_idx,
        int n_blk_idx, int k_chunk_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();

    const int k_start = k_chunk_idx * bgmmc.K_chunk_elems;
    const bool is_K_tail
            = brgmm_ctx.is_last_K_chunk(k_chunk_idx) && bgmmc.K_tail > 0;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_chunk_idx);
    auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();

    const int n = n_blk_idx * bgmmc.N_blk;
    const bool is_N_tail = (bgmmc.N - n < bgmmc.N_blk);
    ctx.current_N_blk = is_N_tail ? bgmmc.N_tail : bgmmc.N_blk;
    ctx.zp_a_compensation_ptr = (void *)brgmm_ctx.get_zp_a_compensation_ptr(
            ithr, b_idx, n_blk_idx);
    ctx.zp_a_neg_value_ptr = (void *)brgmm_ctx.get_zp_a_neg_val_ptr();

    int gb = 0;
    for (; gb < gemm_batch; gb++) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_B_ptr(b_idx, k, n);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_B_ptr(ithr, gb, n_blk_idx);
        ctx.compensation_ptr
                = (void *)brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx);
        ctx.current_K_start = k;
        ctx.current_K_iters = nstl::min(bgmmc.K_blk, bgmmc.K);
        (*copy_B_kernel_)(&ctx);
    }

    if (is_K_tail) {
        const int k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_B_ptr(b_idx, k, n);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_B_ptr(ithr, gb, n_blk_idx);
        ctx.compensation_ptr
                = (void *)brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx);
        ctx.current_K_start = k;
        ctx.current_K_iters = bgmmc.K % bgmmc.K_blk;
        (*copy_B_kernel_)(&ctx);
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::accumulate(
        char *result_ptr, const char *reduce_ptr, size_t size) const {
    if (pd()->get_brgemm_matmul_conf().acc_dt == f32)
        acc_ker_f32_->accumulate(
                (float *)result_ptr, (const float *)reduce_ptr, size);
    else if (pd()->get_brgemm_matmul_conf().acc_dt == s32)
        acc_ker_s32_->accumulate(
                (int32_t *)result_ptr, (const int32_t *)reduce_ptr, size);
    else
        assert(!"unsupported accumulation data type");
}

template <cpu_isa_t isa>
struct brgemm_matmul_t<isa>::brg_matmul_exec_ctx_t {
    brg_matmul_exec_ctx_t(const exec_ctx_t &ctx, const pd_t *pd,
            const float *oscales, const float *dst_scales,
            matmul_helper_t &helper)
        : bgmmc_(pd->get_brgemm_matmul_conf())
        , data_A_ptr_(CTX_IN_MEM(const char *, DNNL_ARG_SRC))
        , data_B_ptr_(CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS))
        , data_C_ptr_(CTX_OUT_MEM(char *, DNNL_ARG_DST))
        , wsp_tile_ptr_(nullptr)
        , bias_ptr_(CTX_IN_MEM(const char *, DNNL_ARG_BIAS))
        , oscales_ptr_(oscales)
        , dst_scales_ptr_(dst_scales) {

        // setup scales / zp pointers
        const void *src_zero_points = CTX_IN_MEM(
                const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
        const void *wei_zero_points = CTX_IN_MEM(
                const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
        const void *dst_zero_points = CTX_IN_MEM(
                const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

        zero_point_a_negative_val_ = src_zero_points
                ? -cpu::io::load_int_value(
                          pd->attr()->zero_points_.get_data_type(DNNL_ARG_SRC),
                          src_zero_points, 0)
                : 0;
        const int zero_point_b_val = wei_zero_points
                ? cpu::io::load_int_value(
                          pd->attr()->zero_points_.get_data_type(
                                  DNNL_ARG_WEIGHTS),
                          wei_zero_points, 0)
                : 0;
        zero_point_b_negative_val_ = -zero_point_b_val;
        zero_point_c_val_ = dst_zero_points
                ? cpu::io::load_int_value(
                          pd->attr()->zero_points_.get_data_type(DNNL_ARG_DST),
                          dst_zero_points, 0)
                : 0;

        const auto &scratchpad = ctx.get_scratchpad_grantor();
        const auto &bgmmc = pd->get_brgemm_matmul_conf();

        batch_element_ptr_ = scratchpad.template get<brgemm_batch_element_t>(
                key_brgemm_primitive_batch);

        const bool use_buffer_a
                = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;
        buf_A_ptr_ = (use_buffer_a)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
                : nullptr;

        buf_B_ptr_ = (bgmmc.use_buffer_b)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
                : nullptr;

        buf_C_ptr_ = (bgmmc.use_buffer_c)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
                : nullptr;

        buf_D_ptr_ = (bgmmc.is_runtime_M)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_d)
                : nullptr;

        const memory_desc_wrapper weights_d(pd->weights_md(0));
        const dim_t comp_offset = bgmmc_.b_dt_sz
                * (weights_d.size() - weights_d.additional_buffer_size());
        s8s8_compensation_ptr_ = (bgmmc.s8s8_compensation_required)
                ? ((bgmmc.use_buffer_b)
                                  ? scratchpad.template get<int32_t>(
                                            key_brgemm_primitive_buffer_comp)
                                  : const_cast<int32_t *>(
                                            reinterpret_cast<const int32_t *>(
                                                    &data_B_ptr_[comp_offset])))
                : nullptr;
        assert(IMPLICATION(bgmmc.s8s8_compensation_required,
                bgmmc_.b_dt_sz == bgmmc_.tr_b_dt_sz));

        zero_point_a_compensations_ptr_ = bgmmc.has_zero_point_a
                ? scratchpad.template get<int32_t>(
                          key_brgemm_primitive_zp_comp_a)
                : nullptr;
        zero_point_b_compensations_ptr_ = bgmmc.has_zero_point_b
                ? scratchpad.template get<int32_t>(
                          key_brgemm_primitive_zp_comp_b)
                : nullptr;

        zero_point_mixed_ab_compensation_component_
                = bgmmc.K * zero_point_a_negative_val_;

        post_ops_binary_rhs_arg_vec_ = binary_injector::prepare_binary_args(
                pd->attr()->post_ops_, ctx);
        base_brg_ker_idx_
                = pd->get_brg_kernel_idx(false, true, 0, false, false);
        vnni_factor = data_type_vnni_granularity(bgmmc.wei_dt);

        reorder_zp_a_comp_ptr_ = nullptr;
        if (bgmmc_.has_zero_point_a && bgmmc_.blocked_B) {
            // Store the pointer to computed in reorder compensation values to
            // scale them locally by zp_a value just before usage in post-ops.
            // Using the single global scaling before parallel section might
            // produce significant overhead for small problems running in
            // multitreaded execution mode
            const size_t reorder_zp_a_comp_offset
                    = weights_d.size() - weights_d.additional_buffer_size();
            const size_t b_batch
                    = get_bb_idx(bgmmc.batch - 1, bgmmc_.bcast_B_desc) + 1;
            assert(IMPLICATION(bgmmc.s8s8_compensation_required,
                    !is_runtime_value(bgmmc.s8s8_comp_b_str)));
            const size_t s8s8_buffer_sz = bgmmc.s8s8_compensation_required
                    ? sizeof(int32_t) * b_batch * bgmmc.s8s8_comp_b_str
                    : 0;
            reorder_zp_a_comp_ptr_
                    = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(
                            &data_B_ptr_[reorder_zp_a_comp_offset
                                    + s8s8_buffer_sz]));
        }

        // Set last_chunk_brgemm_batch_size_ to brgemm_batch_size
        // when K_tail = 0 and brgemm_batch_tail_size = 0
        last_chunk_brgemm_batch_size_ = bgmmc.brgemm_batch_tail_size;
        if (bgmmc.K_tail == 0 && last_chunk_brgemm_batch_size_ == 0)
            last_chunk_brgemm_batch_size_ = bgmmc.brgemm_batch_size;

        copy_A_src_stride_ = bgmmc.copy_A_src_stride;
        if (bgmmc.is_runtime_M) {
            M_ = helper.M();
            M_chunks_ = M_ / bgmmc.M_chunk_elems;
            M_chunk_size_ = bgmmc.M_chunk_size;
            num_M_blocks_ = M_chunks_ * M_chunk_size_;
            M_chunk_tail_elements_ = M_ % bgmmc.M_chunk_elems;
            int tail = M_chunk_tail_elements_;
            dim_t m_idx = M_ - tail;
            int tail_idx = 0;
            dim_t m_c_buf_idx = 0;
            while (tail > 0) {
                int tail_ker_size = dynamic_m_tails[tail_idx];
                int ker_idx = tail_idx + 1;
                int prev_tail_ker_size = tail_idx > 0
                        ? dynamic_m_tails[tail_idx - 1]
                        : (int)bgmmc.M_blk;
                bool last_tail_kernel = tail_idx == max_num_dynamic_m_tails - 1;
                if (tail > tail_ker_size && M_ >= prev_tail_ker_size) {
                    tail_ker_size = prev_tail_ker_size;
                    ker_idx--;
                } else if (tail < tail_ker_size && !last_tail_kernel) {
                    // skip this tail kernel, try the next one
                    tail_idx++;
                    continue;
                }
                int kernel_m_shift = nstl::max(tail_ker_size - tail, 0);

                m_tail_processing_.push_back({m_idx, ker_idx, tail_ker_size,
                        kernel_m_shift, m_c_buf_idx});
                tail -= tail_ker_size;
                m_idx += tail_ker_size - kernel_m_shift;
                m_c_buf_idx += tail_ker_size;
                if (!last_tail_kernel && tail_ker_size != bgmmc.M_blk)
                    tail_idx++;
            }
            M_chunk_tail_ = m_tail_processing_.size();
            if (M_chunk_tail_ > 0) {
                M_chunks_++;
                num_M_blocks_ += M_chunk_tail_;
            }
            M_tail_block_start_ = num_M_blocks_ - M_chunk_tail_;
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                A_strides_[dim_idx] = bgmmc.a_dt_sz
                        * helper.get_a_stride(bgmmc.ndims - 1 - dim_idx);
            A_ptr_shift_b_ = bgmmc.A_ptr_shift_b;
            if (bgmmc.transposed_A)
                copy_A_src_stride_
                        = helper.get_a_stride(bgmmc.ndims - 1) * bgmmc.a_dt_sz;
        } else {
            M_ = bgmmc.M;
            M_chunks_ = bgmmc.M_chunks;
            num_M_blocks_ = bgmmc.num_M_blocks;
            M_chunk_size_ = bgmmc.M_chunk_size;
            M_chunk_tail_ = num_M_blocks_ % M_chunk_size_;
            M_chunk_tail_elements_ = M_ % bgmmc.M_chunk_elems;
            M_tail_block_start_ = num_M_blocks_ - (bgmmc.M_tail > 0);
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                A_strides_[dim_idx] = bgmmc.A_strides[dim_idx];
            A_ptr_shift_b_ = bgmmc.A_ptr_shift_b;
        }
        // parallelization
        parallel_work_amount_ = bgmmc.batch * M_chunks_ * bgmmc.N_chunks;

        // The number of threads available during primitive execution may
        // increase (ex. Eigen threadpool implementation) or decrease
        // (ex. nested parallelism) compared to the
        // number of threads available during primitive creation.
        // So we limit the total number of threads to the
        // minimum of these two values to prevent potential OOM issues.
        nthr_ = nstl::min(dnnl_get_current_num_threads(), bgmmc.nthr);

        nthr_k_ = bgmmc.nthr_k > 0 && bgmmc.nthr_k <= nthr_ ? bgmmc.nthr_k : 1;
        nthr_bmn_ = nthr_ / nthr_k_;
        num_threads_used_ = nthr_k_ * nthr_bmn_;

        // If parallel_work_amount_ == 1 and parallel reduction is not used, we
        // limit num threads to 1 as parallel(1, ...) does not create parallel
        // section at all. We do not limit number of threads for case
        // 1 < parallel_work_amount_ < dnnl_get_max_threads() to avoid potential
        // overhead on spawning different number of OMP threads from layer to
        // layer.
        if (parallel_work_amount_ == 1 && !parallel_reduction_is_used())
            nthr_ = nthr_bmn_ = nthr_k_ = 1;

        const bool need_to_calculate_compensation_for_a
                = bgmmc.has_zero_point_b;
        const bool need_to_calculate_compensation_for_b = !IMPLICATION(
                (bgmmc.has_zero_point_a || bgmmc.s8s8_compensation_required),
                bgmmc.blocked_B);
        const bool calculate_compensations_in_copy_routines
                = need_to_calculate_compensation_for_a
                || need_to_calculate_compensation_for_b;
        // currently parallel reduction is supported only for case of
        // non-batched problems without computation of any compensations in
        // copy routines
        assert(IMPLICATION(parallel_reduction_is_used(),
                bgmmc.batch == 1 && !calculate_compensations_in_copy_routines));
        MAYBE_UNUSED(need_to_calculate_compensation_for_a);
        MAYBE_UNUSED(need_to_calculate_compensation_for_b);
        MAYBE_UNUSED(calculate_compensations_in_copy_routines);
    }

    // NOTE: gb --> generalized batch, bb --> broadcast batch
    int get_bb_idx(int gb_idx, const brgemm_matmul_bcast_desc_t &bd) const {
        if (!bd.bcast_mask) // no broadcast
            return gb_idx;

        if (bd.bcast_across_all_batch_dims) return 0;

        int gb_off_before_bcast = utils::rnd_dn(
                gb_idx, bd.first_bcast_dim_to_last_batch_dim_prod);
        int bb_idx = gb_off_before_bcast / (bd.bcast_dims_prod);

        dim_t cur_bcast_dims_prod = bd.bcast_dims_prod;
        int mask = 1 << (bgmmc_.batch_ndims - bd.first_bcast_dim - 1);
        for (int d = bd.first_bcast_dim; d < bd.last_bcast_dim; ++d) {
            if (bd.bcast_mask & mask) // broadcast
                cur_bcast_dims_prod /= bd.batch_dims[d];
            else {
                int cur_b = (gb_idx / bd.gb_off[d]) % bd.batch_dims[d];
                bb_idx += cur_b * (bd.gb_off[d] / cur_bcast_dims_prod);
            }
            mask >>= 1;
        }
        bb_idx += gb_idx % bd.gb_off[bd.last_bcast_dim];
        return bb_idx;
    }

    const char *get_data_A_ptr(int b, int m, int k) const {
        int cur_b = get_bb_idx(b, bgmmc_.bcast_A_desc);
        return data_A_ptr_ + get_data_A_off(cur_b, m, k);
    }

    const char *get_data_B_ptr(int b, int k, int n) const {
        int cur_b = get_bb_idx(b, bgmmc_.bcast_B_desc);
        return data_B_ptr_ + get_data_B_off(cur_b, k, n);
    }

    char *get_data_C_ptr(int b, int m, int n) const {
        return data_C_ptr_ + get_data_C_off(b, m, n);
    }

    // Binary post-ops injector requires dst anchor point for offset calculation
    // for different types of broadcast strategy based on the current value of
    // ptr_D.
    char *get_dst_anchor_point_for_binary_post_ops(
            int b, int m_blk_idx, int n_blk_idx) const {
        // General case - no copy buffer for destination tensor is used, so we
        // need to provide C_ptr base pointer as anchor
        if (!copy_d_required(m_blk_idx)) return get_data_C_ptr(0, 0, 0);

        // We calculate matmul with all the post-ops and data conversion to
        // temporal buffer. In this case we need to provide anchor pointer based
        // on buffer pointer used to allow binary injector to calculate required
        // offsets correctly. Need to double check this approach for batched
        // case before enabling runtime dimensions support.
        assert(bgmmc_.batch == 1);
        const int n = n_blk_idx * bgmmc_.N_blk;
        const dim_t m = get_M_idx(m_blk_idx, true);
        ptrdiff_t offset_for_real_dst
                = get_data_C_off(b, m, n) - get_data_C_off(0, 0, 0);
        return get_buf_D_ptr(m_blk_idx, n_blk_idx) - offset_for_real_dst;
    }

    brgemm_batch_element_t *get_batch_elem_ptr(int ithr) const {
        return batch_element_ptr_
                + ithr * bgmmc_.brgemm_batch_element_per_thr_sz;
    }

    void init_brgemm_batch_elements_values(int ithr, int brg_batch_start,
            int brg_batch_iters, int b_idx, int m_blk_idx, int k_blk_idx,
            int n_blk_idx) const {
        auto addr_batch = get_batch_elem_ptr(ithr);

        const dim_t m = get_M_idx(m_blk_idx, true);
        const int n = n_blk_idx * bgmmc_.N_blk;

        for (int b_iter = 0; b_iter < brg_batch_iters; b_iter++) {
            const int brg_batch_idx = brg_batch_start + b_iter;
            const int k = (k_blk_idx + brg_batch_idx) * bgmmc_.K_blk;
            addr_batch[b_iter].ptr.A = bgmmc_.use_buffer_a
                    ? get_buf_A_ptr(ithr, m_blk_idx, brg_batch_idx)
                    : get_data_A_ptr(b_idx, m, k);
            addr_batch[b_iter].ptr.B = (bgmmc_.use_buffer_b)
                    ? get_buf_B_ptr(ithr, brg_batch_idx, n_blk_idx)
                    : get_data_B_ptr(b_idx, k, n);
        }
    }

    char *get_buf_A_ptr(int ithr, int m_blk_idx, int k_blk_idx) const {
        if (!bgmmc_.use_buffer_a && !bgmmc_.use_buffer_a_tail_only)
            return nullptr;

        const int k_blk_local = bgmmc_.use_buffer_a_tail_only ? 0 : k_blk_idx;
        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const int tail_idx = get_M_tail_block_idx(m_blk_idx);
            const int curr_m_block_size
                    = m_tail_processing_[tail_idx].kernel_size;
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[tail_idx].buf_dim_idx;
            const dim_t ld = bgmmc_.tr_a_dt_sz
                    * (bgmmc_.use_buffer_a_tail_only ? bgmmc_.wei_k_blk
                                                     : bgmmc_.LDA);
            const int batch = bgmmc_.use_buffer_a_tail_only
                    ? 1
                    : bgmmc_.brgemm_batch_size;
            const dim_t offset = curr_m_buf_shift * ld * batch
                    + k_blk_local * ld * curr_m_block_size;
            return buf_A_ptr_ + ithr * bgmmc_.buffer_a_per_thread_sz + offset;
        }

        const int m_blk_local = m_blk_idx % M_chunk_size_;
        return buf_A_ptr_ + ithr * bgmmc_.buffer_a_per_thread_sz
                + m_blk_local * bgmmc_.buffer_a_chunk_shift_along_m
                + k_blk_local * bgmmc_.buffer_a_chunk_sz;
    }

    char *get_buf_B_ptr(int ithr, int k_blk_idx, int n_blk_idx) const {
        UNUSED(n_blk_idx);
        if (!bgmmc_.use_buffer_b) return nullptr;

        return buf_B_ptr_ + ithr * bgmmc_.buffer_b_per_thread_sz
                + k_blk_idx * bgmmc_.buffer_b_chunk_sz;
    }

    char *get_buf_C_ptr(int ithr, int m_blk_idx, int n_blk_idx) const {
        if (!bgmmc_.use_buffer_c) return nullptr;

        if (bgmmc_.nthr_k > 1) {
            const int nthr_k = bgmmc_.nthr_k <= nthr_ ? bgmmc_.nthr_k : 1;
            const int nthr_bmn = nthr_ / nthr_k;
            const int ithr_k = ithr / nthr_bmn;
            return get_buf_C_par_reduction_ptr(ithr_k, m_blk_idx, n_blk_idx);
        }

        const int n_blk_local = n_blk_idx % bgmmc_.N_chunk_size;
        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const int tail_idx = get_M_tail_block_idx(m_blk_idx);
            const int curr_m_block_size
                    = m_tail_processing_[tail_idx].kernel_size;
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[tail_idx].buf_dim_idx;
            const dim_t offset = bgmmc_.acc_dt_sz * bgmmc_.LDC
                    * (curr_m_buf_shift * bgmmc_.N_chunk_size
                            + n_blk_local * curr_m_block_size);
            return buf_C_ptr_ + ithr * bgmmc_.buffer_c_per_thread_sz + offset;
        }
        const int m_blk_local = m_blk_idx % M_chunk_size_;
        const int buf_idx = bgmmc_.N_chunk_size * m_blk_local + n_blk_local;

        return buf_C_ptr_ + ithr * bgmmc_.buffer_c_per_thread_sz
                + buf_idx * bgmmc_.buffer_c_chunk_sz;
    }

    char *get_buf_D_ptr(int m_blk_idx, int n_blk_idx) const {
        if (!is_runtime_M_tail_chunk(m_blk_idx)) return nullptr;

        const dim_t curr_m_buf_shift
                = m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                          .buf_dim_idx;

        return buf_D_ptr_ + bgmmc_.c_dt_sz * bgmmc_.LDD * curr_m_buf_shift
                + bgmmc_.c_dt_sz * bgmmc_.N_blk * n_blk_idx;
    }

    char *get_buf_C_par_reduction_ptr(
            int ithr_k, int m_blk_idx, int n_blk_idx) const {
        if (bgmmc_.nthr_k <= 1) return nullptr;

        const int m = m_blk_idx * bgmmc_.M_blk;
        const int n = n_blk_idx * bgmmc_.N_blk;

        if (!bgmmc_.post_ops_applicable && ithr_k == 0)
            return get_data_C_ptr(0, m, n);

        int k_buf_idx = ithr_k - (!bgmmc_.post_ops_applicable ? 1 : 0);
        return buf_C_ptr_ + k_buf_idx * bgmmc_.buffer_c_per_thread_sz
                + get_data_C_off(0, m, n) * bgmmc_.acc_dt_sz / bgmmc_.c_dt_sz;
    }

    // Auxiliary functions for getting offsets with pre-calculated memory
    // strides for each tensor to get general solution for all possible
    // dimension without significant overhead
    dim_t get_data_A_off(int b, int m, int k) const {
        using namespace format_tag;
        if (one_of(bgmmc_.src_tag, acbd, adbc)
                /* this is a special case when src can be represented
                   by plain and transposed tags due to a batch dim equal to 1 */
                || (one_of(bgmmc_.src_tag, abcd, abdc)
                        && bgmmc_.A_ptr_shift_b != 0)) {
            dim_t b_off = 0;
            if (!bgmmc_.bcast_A_desc.bcast_mask) { // no broadcast
                const dim_t batch_dim1 = bgmmc_.bcast_A_desc.batch_dims[1];
                b_off = A_strides_[2] * (b % batch_dim1)
                        + (b / batch_dim1) * A_ptr_shift_b_;
            } else {
                b_off = b * A_ptr_shift_b_;
            }
            return b_off + A_strides_[1] * m + A_strides_[0] * k;
        } else {
            return A_strides_[2] * b + A_strides_[1] * m + A_strides_[0] * k;
        }
    }

    dim_t get_data_B_off(int b, int k, int n) const {
        using namespace format_tag;
        if (one_of(bgmmc_.wei_tag, acbd, adbc)
                /* this is a special case when weights can be represented
                   by plain and transposed tags due to a batch dim equal to 1 */
                || (one_of(bgmmc_.wei_tag, abcd, abdc)
                        && bgmmc_.B_ptr_shift_b != 0)) {
            dim_t b_off = 0;
            if (!bgmmc_.bcast_B_desc.bcast_mask) { // no broadcast
                const dim_t batch_dim1 = bgmmc_.bcast_B_desc.batch_dims[1];
                b_off = bgmmc_.B_strides[2] * (b % batch_dim1)
                        + (b / batch_dim1) * bgmmc_.B_ptr_shift_b;
            } else {
                b_off = b * bgmmc_.B_ptr_shift_b;
            }
            return b_off + bgmmc_.B_strides[1] * k + bgmmc_.B_strides[0] * n;
        } else {
            int dt_b_k_blk = bgmmc_.is_bf32
                    ? data_type_vnni_simd_elems<sve_512>(f32)
                    : bgmmc_.wei_k_blk;
            int k_idx = bgmmc_.blocked_B ? k / dt_b_k_blk : k;
            int n_idx = bgmmc_.blocked_B ? n / bgmmc_.wei_n_blk : n;
            return bgmmc_.B_strides[2] * b + bgmmc_.B_strides[1] * k_idx
                    + bgmmc_.B_strides[0] * n_idx
                    + get_data_B_off_within_block(k, n);
        }
    }

    dim_t get_data_B_off_within_block(int k, int n) const {
        using namespace format_tag;

        if (!bgmmc_.blocked_B) return 0;

        int x0 = k % bgmmc_.wei_k_blk;
        int x1 = n % bgmmc_.wei_n_blk;
        dim_t offset = (x0 / vnni_factor) * vnni_factor * bgmmc_.wei_n_blk
                + x1 * vnni_factor + x0 % vnni_factor;
        return bgmmc_.b_dt_sz * offset;
    }

    dim_t get_data_C_off(int b, int m, int n) const {
        using namespace format_tag;
        assert(bgmmc_.dst_tag != adbc);
        if (bgmmc_.dst_tag == acbd
                || (one_of(bgmmc_.dst_tag, abcd, abdc)
                        && bgmmc_.C_ptr_shift_b != 0)) {
            const dim_t batch_dim1 = bgmmc_.bcast_A_desc.batch_dims[1];
            dim_t b_off = bgmmc_.C_strides[2] * (b % batch_dim1)
                    + (b / batch_dim1) * bgmmc_.C_ptr_shift_b;
            return b_off + bgmmc_.C_strides[1] * m + bgmmc_.C_strides[0] * n;
        } else {
            return bgmmc_.C_strides[2] * b + bgmmc_.C_strides[1] * m
                    + bgmmc_.C_strides[0] * n;
        }
    }

    const char *get_bias_ptr(int n) const {
        if (!bgmmc_.with_bias) return nullptr;

        return bias_ptr_ + n * bgmmc_.bias_dt_sz;
    }

    int32_t *get_s8s8_comp_ptr(int ithr, int b, int n_blk_idx) const {
        if (!bgmmc_.s8s8_compensation_required) return nullptr;

        const int n_blk_local = bgmmc_.use_buffer_b
                ? n_blk_idx % bgmmc_.N_chunk_size
                : n_blk_idx;
        assert(!is_runtime_value(bgmmc_.s8s8_comp_b_str));
        return s8s8_compensation_ptr_ + ithr * bgmmc_.s8s8_comp_ithr_str
                + get_bb_idx(b, bgmmc_.bcast_B_desc) * bgmmc_.s8s8_comp_b_str
                + n_blk_local * bgmmc_.s8s8_comp_n_str;
    }

    const float *get_oscales_ptr(int n) const {
        return bgmmc_.with_scales ? oscales_ptr_ + bgmmc_.is_oscale_per_n * n
                                  : nullptr;
    }

    const float *get_dst_scales_ptr() const { return dst_scales_ptr_; }

    const int32_t *get_zp_a_neg_val_ptr() const {
        return &zero_point_a_negative_val_;
    }

    const int32_t *get_zp_b_neg_val_ptr() const {
        return &zero_point_b_negative_val_;
    }

    const int32_t *get_zp_ab_mixed_comp_ptr() const {
        return &zero_point_mixed_ab_compensation_component_;
    }

    const int32_t *get_zp_c_val_ptr() const { return &zero_point_c_val_; }

    int32_t *get_zp_a_compensation_ptr(
            int ithr, int b_idx, int n_blk_idx) const {
        if (!bgmmc_.has_zero_point_a) return nullptr;

        const int n_blk_local = n_blk_idx % bgmmc_.N_chunk_size;
        int32_t *zp_comp = zero_point_a_compensations_ptr_
                + ithr * bgmmc_.zp_a_comp_elems_per_thr
                + n_blk_local * bgmmc_.zp_a_comp_shift_n;

        if (bgmmc_.blocked_B) {
            // Scale computed in reorder compensation values by zp_a value
            // locally just before usage. Using the single global scaling before
            // parallel section might produce significant overhead for small
            // problems running in multitreaded execution mode
            const int base_offset = get_bb_idx(b_idx, bgmmc_.bcast_B_desc)
                            * rnd_up(bgmmc_.N, bgmmc_.wei_n_blk)
                    + n_blk_idx * bgmmc_.wei_n_blk;
            PRAGMA_OMP_SIMD()
            for (int b = 0; b < bgmmc_.wei_n_blk; b++)
                zp_comp[b] = -zero_point_a_negative_val_
                        * reorder_zp_a_comp_ptr_[base_offset + b];
        }
        return zp_comp;
    }

    int32_t *get_zp_b_compensation_result_ptr(int ithr, int m_blk_idx) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;

        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                              .buf_dim_idx;
            return zero_point_b_compensations_ptr_
                    + ithr * bgmmc_.zp_b_comp_elems_per_thr + curr_m_buf_shift;
        }

        const int m_blk_local = m_blk_idx % M_chunk_size_;
        return zero_point_b_compensations_ptr_
                + ithr * bgmmc_.zp_b_comp_elems_per_thr
                + m_blk_local * bgmmc_.zp_b_comp_result_shift_m;
    }

    int32_t *get_zp_b_compensation_buffer_ptr(int ithr, int m_blk_idx) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;

        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                              .buf_dim_idx;
            return get_zp_b_compensation_result_ptr(ithr, 0)
                    + bgmmc_.zp_b_comp_buffer_start + curr_m_buf_shift;
        }

        const int m_blk_local = m_blk_idx % M_chunk_size_;
        return get_zp_b_compensation_result_ptr(ithr, 0)
                + bgmmc_.zp_b_comp_buffer_start
                + m_blk_local * bgmmc_.zp_b_comp_buffer_shift_m;
    }

    char *get_tile_workspace(int ithr) const { return nullptr; }

    const std::vector<const void *> &get_post_ops_binary_rhs_arg_vec() const {
        return post_ops_binary_rhs_arg_vec_;
    }

    int get_base_brgemm_kernel_idx() const { return base_brg_ker_idx_; }

    bool is_last_K_chunk(int k_chunk_idx) const {
        return k_chunk_idx == bgmmc_.K_chunks - 1;
    }

    int get_brgemm_batch_size(int k_chunk_idx) const {
        return is_last_K_chunk(k_chunk_idx) ? last_chunk_brgemm_batch_size_
                                            : bgmmc_.brgemm_batch_size;
    }

    int get_parallel_work_amount() const { return parallel_work_amount_; }
    int get_num_threads_for_k() const { return nthr_k_; }
    bool parallel_reduction_is_used() const {
        return nthr_k_ > 1 && bgmmc_.K_chunks > 1;
    }
    int get_num_threads_for_bmn() const { return nthr_bmn_; }
    // ithr = ithr_k * nthr_bmn + ithr_bmn
    int get_thread_idx_for_k(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_k = ithr / nthr_bmn_;
        return ithr_k < bgmmc_.K_chunks ? ithr_k : -1;
    }
    int get_thread_idx_for_bmn(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_bmn = ithr % nthr_bmn_;
        return ithr_bmn < parallel_work_amount_ ? ithr_bmn : -1;
    }
    int get_num_threads_for_parallelization() const { return nthr_; }
    dim_t get_M() const { return M_; }
    int get_M_chunks() const { return M_chunks_; }
    int get_num_M_blocks() const { return num_M_blocks_; }
    int get_M_chunk_size() const { return M_chunk_size_; }
    int get_M_chunk_tail() const { return M_chunk_tail_; }
    int get_M_tail_block_idx(int m_block_idx) const {
        return m_block_idx - M_tail_block_start_;
    }
    bool is_M_tail_processing(int m_block_idx) const {
        return get_M_tail_block_idx(m_block_idx) >= 0;
    }
    bool is_runtime_M_tail_chunk(int m_block_idx) const {
        return bgmmc_.is_runtime_M && is_M_tail_processing(m_block_idx);
    }

    int get_M_kernel_idx(int m_block_idx) const {
        if (!is_M_tail_processing(m_block_idx))
            return 0;
        else if (!bgmmc_.is_runtime_M)
            return 1;

        assert(is_runtime_M_tail_chunk(m_block_idx)
                && !m_tail_processing_.empty());
        return m_tail_processing_[get_M_tail_block_idx(m_block_idx)].kernel_idx;
    }

    int get_M_kernel_size(int m_block_idx) const {
        if (!is_M_tail_processing(m_block_idx))
            return bgmmc_.M_blk;
        else if (!bgmmc_.is_runtime_M)
            return bgmmc_.M_tail;

        assert(is_runtime_M_tail_chunk(m_block_idx)
                && !m_tail_processing_.empty());
        return m_tail_processing_[get_M_tail_block_idx(m_block_idx)]
                .kernel_size;
    }

    dim_t get_M_idx(
            int m_block_idx, bool adjust_for_kernel_overlap = false) const {
        if (is_runtime_M_tail_chunk(m_block_idx)) {
            const int tail_idx = get_M_tail_block_idx(m_block_idx);
            const int shift = adjust_for_kernel_overlap
                    ? m_tail_processing_[tail_idx].shift
                    : 0;
            return m_tail_processing_[tail_idx].idx - shift;
        }
        return m_block_idx * bgmmc_.M_blk;
    }

    bool copy_d_required(int m_block_idx) const {
        if (!is_runtime_M_tail_chunk(m_block_idx)) return false;

        return m_tail_processing_[get_M_tail_block_idx(m_block_idx)].shift > 0;
    }
    dim_t get_src_stride() const { return copy_A_src_stride_; }

    void copy_dst_values_to_buffer(
            int b_idx, int m_blk_idx, int n_blk_idx) const {
        if (IMPLICATION(bgmmc_.with_sum, !copy_d_required(m_blk_idx))) return;
        dim_t m = get_M_idx(m_blk_idx);
        const int n = n_blk_idx * bgmmc_.N_blk;
        const bool is_N_tail = (bgmmc_.N - n < bgmmc_.N_blk);

        const int runtime_M_tail_kernel_idx = get_M_tail_block_idx(m_blk_idx);
        const int rows_to_skip
                = m_tail_processing_[runtime_M_tail_kernel_idx].shift;
        const int rows_to_copy
                = m_tail_processing_[runtime_M_tail_kernel_idx].kernel_size;
        const dim_t ld = bgmmc_.LDD * bgmmc_.c_dt_sz;
        const dim_t bytes_to_copy
                = bgmmc_.c_dt_sz * (is_N_tail ? bgmmc_.N_tail : bgmmc_.N_blk);
        auto copy_from = get_data_C_ptr(b_idx, m - rows_to_skip, n);
        auto copy_to = get_buf_D_ptr(m_blk_idx, n_blk_idx);
        for (int r = 0; r < rows_to_copy; r++) {
            utils::array_copy(copy_to, copy_from, bytes_to_copy);
            copy_from += ld;
            copy_to += ld;
        }
    }

    void copy_dst_values_from_buffer(
            int b_idx, int m_blk_idx, int n_blk_idx) const {
        if (!copy_d_required(m_blk_idx)) return;
        dim_t m = get_M_idx(m_blk_idx);
        const int n = n_blk_idx * bgmmc_.N_blk;
        const bool is_N_tail = (bgmmc_.N - n < bgmmc_.N_blk);

        const int runtime_M_tail_kernel_idx = get_M_tail_block_idx(m_blk_idx);
        const int rows_to_skip
                = m_tail_processing_[runtime_M_tail_kernel_idx].shift;
        const int rows_to_copy
                = m_tail_processing_[runtime_M_tail_kernel_idx].kernel_size
                - rows_to_skip;
        const dim_t ld = bgmmc_.LDD * bgmmc_.c_dt_sz;
        const dim_t bytes_to_copy
                = bgmmc_.c_dt_sz * (is_N_tail ? bgmmc_.N_tail : bgmmc_.N_blk);
        auto copy_from
                = get_buf_D_ptr(m_blk_idx, n_blk_idx) + rows_to_skip * ld;
        auto copy_to = get_data_C_ptr(b_idx, m, n);
        for (int r = 0; r < rows_to_copy; r++) {
            utils::array_copy(copy_to, copy_from, bytes_to_copy);
            copy_from += ld;
            copy_to += ld;
        }
    }

private:
    struct tail_processing_t {
        // dimension index kernel is applied to
        dim_t idx;
        // index of tail processing kernel, 0 is reserved for main block
        int kernel_idx;
        // block size of tail kernel
        int kernel_size;
        // shift wrt dimension index when kernel is applied w/ computational
        // overlapping with other kernel, dim_idx_to_apply_kernel = idx - shift
        int shift;
        // if shift > 0 (computational overlapping case) we have to use buffer
        // for kernel dst to avoid result values spoiling, this value
        // represents dimensional idx for dst buffer
        dim_t buf_dim_idx;
    };

    const brgemm_matmul_conf_t &bgmmc_;
    const char *data_A_ptr_;
    const char *data_B_ptr_;
    char *data_C_ptr_;
    brgemm_batch_element_t *batch_element_ptr_;

    char *buf_A_ptr_;
    char *buf_B_ptr_;
    char *buf_C_ptr_;
    char *buf_D_ptr_;

    char *wsp_tile_ptr_;
    const char *bias_ptr_;
    const float *oscales_ptr_;
    const float *dst_scales_ptr_;
    int32_t *s8s8_compensation_ptr_;

    int32_t *zero_point_a_compensations_ptr_;
    int32_t *zero_point_b_compensations_ptr_;
    int32_t *reorder_zp_a_comp_ptr_;

    int32_t zero_point_a_negative_val_;
    int32_t zero_point_b_negative_val_;
    int32_t zero_point_mixed_ab_compensation_component_;
    int32_t zero_point_c_val_;
    std::vector<const void *> post_ops_binary_rhs_arg_vec_;

    int base_brg_ker_idx_;
    int vnni_factor;

    // parallelization parameters
    int parallel_work_amount_;
    int nthr_, nthr_k_, nthr_bmn_, num_threads_used_;
    int last_chunk_brgemm_batch_size_;
    dim_t M_;
    int M_chunks_;
    int num_M_blocks_;
    int M_chunk_size_;
    int M_chunk_tail_;
    int M_chunk_tail_elements_;
    int M_tail_block_start_;
    dim_t A_strides_[3];
    dim_t A_ptr_shift_b_;
    dim_t copy_A_src_stride_;
    std::vector<tail_processing_t> m_tail_processing_;
};

template struct brgemm_matmul_t<sme>;
template struct brgemm_matmul_t<sve_512>;
template struct brgemm_matmul_t<sve_256>;
template struct brgemm_matmul_t<sve_128>;

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
