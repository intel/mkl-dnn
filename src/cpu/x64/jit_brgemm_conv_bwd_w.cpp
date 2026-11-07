/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_primitive.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"

#include "cpu/x64/jit_brgemm_conv_bwd_w.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

status_t brgemm_convolution_bwd_weights_t::pd_t::init(const engine_t *engine) {
    const auto src_type = src_md(0)->data_type;
    const auto diff_wei_type = diff_weights_md(0)->data_type;
    const auto diff_bia_type = diff_weights_md(1)->data_type;
    const auto diff_dst_type = diff_dst_md(0)->data_type;
    VDISPATCH_CONV(is_bwd_w(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(utils::one_of(src_type, bf16, f16, f8_e5m2, f8_e4m3),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(diff_dst_type == src_type, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(
            utils::one_of(diff_wei_type, f32, f8_e5m2, f8_e4m3, src_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(utils::one_of(diff_bia_type, data_type::undef, f32, f8_e5m2,
                           f8_e4m3, src_type),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(impl::is_dense_format_kind({src_md(0), diff_weights_md(0),
                           diff_weights_md(1), diff_dst_md(0)}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    auto scratchpad = scratchpad_registry().registrar();

    // TODO: make `init_conf` assign initialized object to `jcp_`
    CHECK(brgemm_convolution_utils::init_conf_bwd_w(jcp_, *desc(), src_md_,
            diff_weights_md_, diff_bias_md_, diff_dst_md_, attr_,
            dnnl_get_max_threads()));

    CHECK(brgemm_convolution_utils::init_scratchpad_bwd_w(
            scratchpad, jcp_, src_md_, diff_weights_md_, diff_dst_md_));

    copy2jit_jcp();

    bs_c = jcp_.var_bs ? 1 : (jcp_.max_batch + 1);
    batchsizes.resize(bs_c + 1);
    for (int i = 0; i <= bs_c; i++)
        batchsizes[i] = -1;

    batchsizes[1] = 0;

    const auto adj_M = nstl::max(jcp_.M, jcp_.M_tail);
    brgs_sz_ = bs_c * (adj_M + 1) * 2 * 2 * 2;
    brgs_ = std::make_shared<brgemm_containers::brgemm_desc_container_t>();
    brgs_->resize(brgs_sz_);

    const float alpha = 1.0f;
    const float beta = 1.0f;

    int M_begin = 0;
    int M_end = (jcp_.M_tail == jcp_.M || jcp_.M_tail == 0) ? 1 : 2;
    int N_begin = 0;
    int N_end = (jcp_.N_tail == jcp_.N || jcp_.N_tail == 0) ? 1 : 2;
    int K_begin = 0;
    int K_end = (jcp_.K_tail == jcp_.K || jcp_.K_tail == 0) ? 1 : 2;
    int init_begin = 0;
    int init_end = 2;

    const auto wei_type = src_type;

    for (int i = M_begin; i < M_end; i++) {
        auto M = (i) ? jcp_.M_tail : jcp_.M;
        if (M <= 0) continue;
        // init only needed brgemm descriptors
        const auto bs_end = jcp_.var_bs ? 1 : jcp_.max_batch;
        for (int bs = 0; bs <= bs_end; bs++) {
            if (batchsizes[bs] == -1) continue;
            for_(int i_init = init_begin; i_init < init_end; i_init++)
            for_(int i_N = N_begin; i_N < N_end; i_N++)
            for (int i_K = K_begin; i_K < K_end; i_K++) {
                auto vbeta = (i_init) ? 0 : beta;
                auto vN = (i_N) ? jcp_.N_tail : jcp_.N;
                auto vK = (i_K) ? jcp_.K_tail : jcp_.K;
                if (vN == 0 || vK == 0) continue;
                auto brg_idx = get_brg_idx(bs, M, i_init, i_N, i_K);
                // if brgemm_desc_t already created then skip this iteration
                if ((*brgs_)[brg_idx] != nullptr) continue;
                brgemm_desc_t brg;
                CHECK(brgemm_desc_init(&brg, jcp_.isa, jcp_.brg_type, src_type,
                        wei_type, false, false, brgemm_row_major, alpha, vbeta,
                        jcp_.LDA, jcp_.LDB, jcp_.LDC, M, vN, vK, nullptr));

                brgemm_attr_t brgattr;
                brgattr.use_uker = jcp_.use_uker;
                brgattr.use_interleave_stores = jcp_.use_interleave_stores;
                brgattr.hint_prefetching = jcp_.hint_prefetching;
                brgattr.var_bs = jcp_.var_bs;
                brgattr.max_bs = jcp_.max_batch;
                brgattr.hint_innermost_loop = jcp_.brgemm_bd_loop_innermost
                        ? brgemm_bd_loop_innermost
                        : brgemm_innermost_undef;
                brgattr.use_ace = jcp_.is_ace;
                brgattr.enable_size_driven_roll = true;

                brgattr.hint_expected_A_size = 0;
                brgattr.hint_expected_B_size = 0;
                brgattr.hint_expected_C_size = 0;

                brgattr.wary_A_k_tail_read = false;
                brgattr.bd_mask_level = jcp_.use_M_mask;

                brgattr.max_top_vpad = 0;
                brgattr.max_bottom_vpad = 0;

                const auto lda2_size = static_cast<size_t>(jcp_.tr_iw)
                        * jcp_.ih_block * jcp_.id;
                VDISPATCH_CONV_IC(lda2_size <= INT_MAX,
                        VERBOSE_UNSUPPORTED_FEATURE,
                        "lda2_size > INT_MAX is not supported");
                brgattr.LDA2 = lda2_size;

                const auto ldb2_size = static_cast<size_t>(jcp_.tr_ow)
                        * jcp_.oc_block * jcp_.oh_block * jcp_.od;
                VDISPATCH_CONV_IC(ldb2_size <= INT_MAX,
                        VERBOSE_UNSUPPORTED_FEATURE,
                        "ldb2_size > INT_MAX is not supported");
                brgattr.LDB2 = ldb2_size;

                brgattr.LDC2_M = jcp_.oc_block * jcp_.kd * jcp_.kh * jcp_.kw;
                brgattr.LDC2_N = jcp_.nb_ic * jcp_.ic_block * jcp_.oc_block
                        * jcp_.kd * jcp_.kh * jcp_.kw;

                CHECK(brgemm_desc_set_attr(&brg, brgattr));
                CHECK(brgemm_desc_finalize(&brg));

                brgs_->insert(brg_idx, brg);
            }
        }
    }
    return status::success;
}

// jit_jcp used to initialize transpose kernels shared with jit implementation
void brgemm_convolution_bwd_weights_t::pd_t::copy2jit_jcp() {
    jit_jcp_ = zero<decltype(jit_jcp_)>();
    jit_jcp_.prop_kind = jcp_.prop_kind;
    jit_jcp_.src_dt = jcp_.src_dt;
    jit_jcp_.dst_dt = jcp_.dst_dt;
    jit_jcp_.has_vnni = true; // Needed for transpose routines
    jit_jcp_.harness = jcp_.harness;
    jit_jcp_.simd_w = jcp_.simd_w;
    jit_jcp_.ndims = jcp_.ndims;
    jit_jcp_.mb = jcp_.mb;
    jit_jcp_.ngroups = jcp_.ngroups;
    jit_jcp_.ic = jcp_.ic;
    jit_jcp_.oc = jcp_.oc;
    jit_jcp_.oc_without_padding = jcp_.oc;
    jit_jcp_.ic_without_padding = jcp_.ic_without_padding;
    jit_jcp_.id = jcp_.id;
    jit_jcp_.ih = jcp_.ih;
    jit_jcp_.iw = jcp_.iw;
    jit_jcp_.od = jcp_.od;
    jit_jcp_.oh = jcp_.oh;
    jit_jcp_.ow = jcp_.ow;
    jit_jcp_.f_pad = jcp_.f_pad;
    jit_jcp_.l_pad = jcp_.l_pad;
    jit_jcp_.t_pad = jcp_.t_pad;
    jit_jcp_.back_pad = jcp_.back_pad;
    jit_jcp_.r_pad = jcp_.r_pad;
    jit_jcp_.b_pad = jcp_.b_pad;
    jit_jcp_.kd = jcp_.kd;
    jit_jcp_.kh = jcp_.kh;
    jit_jcp_.kw = jcp_.kw;
    jit_jcp_.stride_d = jcp_.stride_d;
    jit_jcp_.stride_h = jcp_.stride_h;
    jit_jcp_.stride_w = jcp_.stride_w;
    jit_jcp_.dilate_d = jcp_.dilate_d;
    jit_jcp_.dilate_h = jcp_.dilate_h;
    jit_jcp_.dilate_w = jcp_.dilate_w;
    jit_jcp_.src_tag = jcp_.src_tag;
    jit_jcp_.wei_tag = jcp_.wei_tag;
    jit_jcp_.dst_tag = jcp_.dst_tag;
    jit_jcp_.with_bias = jcp_.with_bias;
    jit_jcp_.with_sum = jcp_.with_sum;
    jit_jcp_.with_eltwise = jcp_.with_eltwise;
    jit_jcp_.with_binary = jcp_.with_binary;
    jit_jcp_.is_fused_conv = jcp_.is_fused_conv;
    jit_jcp_.nb_ic = jcp_.nb_ic;
    jit_jcp_.ic_block = jcp_.tr_ic_block;
    jit_jcp_.nb_oc = jcp_.nb_oc;
    jit_jcp_.oc_block = jcp_.oc_block;
    jit_jcp_.nb_oc_blocking = jcp_.nb_oc_blocking;

    jit_jcp_.ic_tail = static_cast<int>(jcp_.tr_ic_tail);
    jit_jcp_.oc_tail = jcp_.oc_tail;

    jit_jcp_.tr_iw = jcp_.tr_iw;
    jit_jcp_.tr_ow = jcp_.tr_ow;
    jit_jcp_.tr_diff_dst_buf_size = jcp_.tr_diff_dst_block_size;
    jit_jcp_.typesize_in = jcp_.typesize_in;
    jit_jcp_.typesize_out = jcp_.typesize_out;
    jit_jcp_.ddst_dt = jcp_.dst_dt;
}

status_t brgemm_convolution_bwd_weights_t::add_brg_kernel(
        int bs, dim_t M, int i_N, int i_K, int i_init) {
    if (M <= 0) return status::success;
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &brgs = *(_pd->brgs_);

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    auto K = (i_K) ? jcp.K_tail : jcp.K;
    if (N <= 0 || K <= 0) return status::success;
    auto brg_idx = _pd->get_brg_idx(bs, M, i_init, i_N, i_K);
    auto brg = brgs[brg_idx];
    if (!brg_kernels_[brg_idx] && brg && brg->bcast_dim > 0 && brg->load_dim > 0
            && brg->reduce_dim > 0) {
        CHECK(brg_kernels_.insert(brg_idx, brg));
        brgemm_palettes_.insert(brg_idx, brg);
    }
    return status::success;
}

status_t brgemm_convolution_bwd_weights_t::init(engine_t *engine) {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &jit_jcp = pd()->jit_jcp_;

    CHECK(safe_ptr_assign(trans_kernel_, create_trans_src(&jit_jcp)));
    CHECK(trans_kernel_->create_kernel());
    CHECK(safe_ptr_assign(trans_dst_kernel_, create_trans_dst(&jit_jcp)));
    CHECK(trans_dst_kernel_->create_kernel());

    if (jcp.with_bias) {
        CHECK(safe_ptr_assign(diff_bias_kernel_,
                new jit_avx512_core_amx_bwd_bias_kernel_t(jit_jcp)));
        CHECK(diff_bias_kernel_->create_kernel());
    }

    if (jcp.nthr_mb > 1) {
        CHECK(safe_ptr_assign(
                acc_ker_, new cpu_accumulator_1d_t<data_type::f32>()));
        CHECK(acc_ker_->create_kernel());
    }
    if (jcp.transform_to_vnni) {
        CHECK(safe_ptr_assign(diff_wei_trans_kernel_,
                new jit_diff_wei_trans_to_vnni_t(jcp.wei_dt,
                        static_cast<int>(jcp.kd), static_cast<int>(jcp.kh),
                        static_cast<int>(jcp.kw),
                        static_cast<int>(jcp.ic_block),
                        static_cast<int>(jcp.oc_block),
                        static_cast<int>(jcp.nb_ic))));
        CHECK(diff_wei_trans_kernel_->create_kernel());
    }

    brg_kernels_.resize(_pd->brgs_sz_);
    brgemm_palettes_.resize(_pd->brgs_sz_);

    int M_begin = 0;
    int M_end = (jcp.M_tail == jcp.M || jcp.M_tail == 0) ? 1 : 2;
    int N_begin = 0;
    int N_end = (jcp.N_tail == jcp.N || jcp.N_tail == 0) ? 1 : 2;
    int K_begin = 0;
    int K_end = (jcp.K_tail == jcp.K || jcp.K_tail == 0) ? 1 : 2;
    int init_begin = 0;
    int init_end = 2;

    const auto bs_end = jcp.var_bs ? 1 : jcp.max_batch;
    for (int bs = 0; bs <= bs_end; bs++) {
        if (_pd->batchsizes[bs] == -1) continue;

        for_(int i_N = N_begin; i_N < N_end; i_N++)
        for_(int i_M = M_begin; i_M < M_end; i_M++)
        for_(int i_init = init_begin; i_init < init_end; i_init++)
        for (int i_K = K_begin; i_K < K_end; i_K++) {
            auto M = (i_M) ? jcp.M_tail : jcp.M;
            if (M <= 0) continue;
            add_brg_kernel(bs, M, i_N, i_K, i_init);
        }
    }

    return status::success;
}

struct brgemm_convolution_bwd_weights_t::thread_info_t {
    const char *const __restrict src = nullptr;
    const char *const __restrict diff_dst = nullptr;
    const char *diff_weights = nullptr;
    const char *diff_bias = nullptr;

    const brgemm_convolution_bwd_weights_t *self;
    const memory_tracking::grantor_t &scratchpad;

    char *tr_src = nullptr;
    char *tr_diff_dst = nullptr;
    simple_barrier::ctx_t *tr_src_bctx = nullptr;
    simple_barrier::ctx_t *tr_diff_dst_bctx = nullptr;

    float *wei_bia_reduction = nullptr;
    float *bia_reduction = nullptr;
    simple_barrier::ctx_t *wei_bia_reduction_bctx = nullptr;

    // All nthreads are mapped to a multidimensional "cube" with sizes:
    // (nthr_mb, nthr_g, nthr_oc, nthr_ic).
    // Variables ithr_* define the coordinates and "layers" of the current
    // thread in this "cube"
    int ithr = 0;
    int ithr_ic_b = 0, ithr_oc_b = 0, ithr_g = 0, ithr_mb = 0;
    int ithr_but_oc = 0;
    int ithr_but_ic = 0;

    dim_t img_start = 0, img_end = 0, img_work = 0;
    dim_t g_start = 0, g_end = 0, g_work = 0;
    dim_t oc_b_start = 0, oc_b_end = 0, oc_b_work = 0;
    dim_t ic_b_start = 0, ic_b_end = 0, ic_b_work = 0;

    int cur_brg_idx = -1;
    brgemm_batch_element_t *__restrict brg_batch;
    char *wsp_tile;
    const exec_ctx_t &exec_ctx;
    const jit_brgemm_conv_conf_t &jcp;
    const memory_desc_wrapper src_d;
    const memory_desc_wrapper diff_dst_d;
    const memory_desc_wrapper diff_weights_d;

    thread_info_t(const brgemm_convolution_bwd_weights_t *pcnv,
            const exec_ctx_t &ctx, int ithr)
        : src(CTX_IN_MEM(const char *, DNNL_ARG_SRC))
        , diff_dst(CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST))
        , diff_weights(CTX_OUT_MEM(char *, DNNL_ARG_DIFF_WEIGHTS))
        , self(pcnv)
        , scratchpad(ctx.get_scratchpad_grantor())
        , ithr(ithr)
        , exec_ctx(ctx)
        , jcp(self->pd()->jcp_)
        , src_d(self->pd()->src_md())
        , diff_dst_d(self->pd()->diff_dst_md())
        , diff_weights_d(self->pd()->diff_weights_md(0)) {

        diff_bias = self->pd()->with_bias() && (jcp.oc % jcp.oc_block != 0)
                        && self->pd()->jcp_.bia_dt == data_type::f32
                ? (char *)scratchpad.template get<float>(key_conv_padded_bias)
                : CTX_OUT_MEM(char *, DNNL_ARG_DIFF_BIAS);

        tr_src = scratchpad.template get<char>(key_conv_tr_src);
        if (jcp.global_transpose)
            tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_src_bctx);

        tr_diff_dst = scratchpad.template get<char>(key_conv_tr_diff_dst);
        if (jcp.global_transpose)
            tr_diff_dst_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_diff_dst_bctx);
        wei_bia_reduction
                = scratchpad.template get<float>(key_conv_wei_bia_reduction);
        bia_reduction = nullptr;
        if (jcp.with_bias) {
            const size_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block
                    * jcp.nb_ic * jcp.ic_block * jcp.kh * jcp.kw * jcp.kd;
            const int num_wei_buffers = jcp.wei_dt != data_type::f32
                    ? jcp.nthr_mb
                    : jcp.nthr_mb - 1;
            bia_reduction = wei_bia_reduction + wei_size * num_wei_buffers;
        }

        wei_bia_reduction_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx);

        ithr_ic_b = ithr % jcp.nthr_ic_b;
        ithr_oc_b = ithr / jcp.nthr_ic_b % jcp.nthr_oc_b;
        ithr_g = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b % jcp.nthr_g;
        ithr_mb = ithr / jcp.nthr_ic_b / jcp.nthr_oc_b / jcp.nthr_g;

        ithr_but_oc
                = (ithr_mb * jcp.nthr_g + ithr_g) * jcp.nthr_ic_b + ithr_ic_b;

        ithr_but_ic
                = (ithr_mb * jcp.nthr_g + ithr_g) * jcp.nthr_oc_b + ithr_oc_b;

        dim_t work_amount = jcp.nthr_mb_work;
        /* reduction dimension */
        balance211(work_amount, jcp.nthr_mb, ithr_mb, img_start, img_end);
        img_work = img_end - img_start;

        /* independent dimensions */
        balance211(jcp.ngroups, jcp.nthr_g, ithr_g, g_start, g_end);
        g_work = g_end - g_start;

        balance211(jcp.nb_oc, jcp.nthr_oc_b, ithr_oc_b, oc_b_start, oc_b_end);
        oc_b_work = oc_b_end - oc_b_start;

        if (jcp.transform_to_vnni) {
            const int vnni_granularity
                    = static_cast<int>(data_type_vnni_granularity(jcp.wei_dt));
            if (vnni_granularity == 0) {
                assert(!"Invalid vnni granularity.");
                return;
            }

            const auto icb_work = div_up(jcp.nb_ic, vnni_granularity);
            balance211(
                    icb_work, jcp.nthr_ic_b, ithr_ic_b, ic_b_start, ic_b_end);
            ic_b_start = nstl::min<dim_t>(
                    jcp.nb_ic, ic_b_start * vnni_granularity);
            ic_b_end = nstl::min<dim_t>(jcp.nb_ic, ic_b_end * vnni_granularity);
        } else {
            balance211(
                    jcp.nb_ic, jcp.nthr_ic_b, ithr_ic_b, ic_b_start, ic_b_end);
        }

        ic_b_work = ic_b_end - ic_b_start;

        brgemm_batch_element_t *const __restrict brg_batch_global
                = (jcp.brg_type == brgemm_strd)
                ? nullptr
                : scratchpad.template get<brgemm_batch_element_t>(
                          key_brgemm_primitive_batch);
        brg_batch = brg_batch_global
                + static_cast<size_t>(ithr) * jcp.adjusted_batch_size;

        auto wsp_tile_global
                = scratchpad.template get<char>(key_conv_amx_tile_buffer);
        wsp_tile = wsp_tile_global + ithr * 2 * brgemm_convolution_utils::P4K;
    }

    const pd_t *pd() const { return self->pd(); }

    inline dim_t get_inp_start(dim_t out_s, dim_t pad, dim_t str) const {
        return nstl::max<dim_t>(0, -pad + out_s * str);
    }

    inline dim_t get_inp_end(
            dim_t out_e, dim_t is, dim_t pad, dim_t str, dim_t ek) const {
        return nstl::min<dim_t>(is, -pad + (out_e - 1) * str + ek);
    }

    inline dim_t get_id_start(dim_t od_s) const {
        return get_inp_start(od_s, jcp.f_pad, jcp.stride_d);
    }
    inline dim_t get_ih_start(dim_t oh_s) const {
        return get_inp_start(oh_s, jcp.t_pad, jcp.stride_h);
    }

    inline dim_t get_id_end(dim_t od_e) const {
        return get_inp_end(od_e, jcp.id, jcp.f_pad, jcp.stride_d, jcp.ext_kd);
    }
    inline dim_t get_ih_end(dim_t oh_e) const {
        return get_inp_end(oh_e, jcp.ih, jcp.t_pad, jcp.stride_h, jcp.ext_kh);
    }

    size_t tr_src_buf_number(dim_t g, dim_t icb) const {
        return jcp.global_transpose
                ? ithr_mb * jcp.nb_ic * jcp.ngroups + g * jcp.nb_ic + icb
                : ithr;
    }

    size_t tr_diff_dst_buf_number(dim_t g, dim_t ocb) const {
        // for current loop order (xoi) if jcp.tr_ocb_chunk then we can reuse
        // same area in tr_diff_dst buffer
        if (jcp.tr_ocb_chunk)
            return jcp.global_transpose
                    ? ((ithr_mb * jcp.ngroups + g) * jcp.nthr_oc_b + ithr_oc_b)
                                    * jcp.nb_oc_blocking
                            + (ocb - oc_b_start) % jcp.nb_oc_blocking
                    : ithr;
        else
            return jcp.global_transpose
                    ? ithr_mb * jcp.nb_oc * jcp.ngroups + g * jcp.nb_oc + ocb
                    : ithr;
    }

    size_t tr_src_off(dim_t g, dim_t icb, dim_t id, dim_t ih) const {
        const size_t tr_row_size = jcp.tr_iw * jcp.ic_block;
        const size_t tr_3d_size = tr_row_size * jcp.ih_block;
        // Aligned to buffer end to use guard elements
        return (tr_src_buf_number(g, icb) * jcp.tr_src_buf_size
                       + id * tr_3d_size + ih * tr_row_size)
                * jcp.src_dsz;
    }

    inline size_t tr_ic_block_src_off(
            dim_t g, dim_t tr_icb, dim_t id, dim_t ih) const {
        const dim_t nb_tr_icb = jcp.ic_block / jcp.tr_ic_block;
        return tr_src_off(g, tr_icb / nb_tr_icb, id, ih)
                + (tr_icb % nb_tr_icb) * jcp.tr_ic_block * jcp.tr_iw
                * jcp.src_dsz;
    }

    inline size_t tr_diff_dst_off(
            dim_t g, dim_t ocb, dim_t od, dim_t oh) const {
        const size_t tr_row_size = jcp.tr_ow * jcp.oc_block;
        const size_t tr_3d_size = tr_row_size * jcp.oh_block;
        return (tr_diff_dst_buf_number(g, ocb) * jcp.tr_diff_dst_buf_size
                       + od * tr_3d_size + oh * tr_row_size)
                * jcp.dst_dsz;
    }

    void trans_src_nxc(char *tr_src, const char *src_base, dim_t tr_icb,
            dim_t row_count, dim_t ih_s) const {
        const dim_t src_stride = jcp.iw * jcp.ngroups * jcp.ic * jcp.src_dsz;
        const dim_t tr_src_stride = jcp.tr_iw * jcp.ic_block * jcp.src_dsz;

        dim_t sp_work = row_count;
        const char *src = src_base;
        const dim_t tr_ic_tail_work
                = jcp.tr_ic_tail ? jcp.tr_ic_tail : jcp.tr_ic_block;
        for (dim_t iwork = 0; iwork < sp_work; iwork++) {
            // For 1x1 convolutions with strides we transpose only
            // needed lines
            if (IMPLICATION(jcp.kh == 1, (ih_s + iwork) % jcp.stride_h == 0)) {
                auto ctx = jit_trans_src_t::ctx_t();
                ctx.src = src;
                ctx.tr_src = tr_src;
                ctx.ch_work = static_cast<int>((tr_icb + 1) == jcp.nb_tr_ic
                                ? tr_ic_tail_work
                                : jcp.tr_ic_block);
                ctx.src_prf = nullptr;
                ctx.tr_src_prf = nullptr;
                (*self->trans_kernel_)(&ctx);
            }
            src += src_stride;
            tr_src += tr_src_stride;
        }
    }

    void trans_dst_nxc(char *tr_diff_dst, const char *diff_dst_base,
            dim_t spatial_start, dim_t spatial_start_offset, dim_t ocb_start,
            dim_t chb_stride, dim_t row_count) const {
        const dim_t diff_dst_stride
                = jcp.ow * jcp.ngroups * jcp.oc * jcp.dst_dsz;
        const dim_t tr_diff_dst_stride = jcp.tr_ow * jcp.oc_block * jcp.dst_dsz;
        dim_t work_rest = row_count;
        const dim_t max_spatial_work = jcp.od * jcp.oh;
        dim_t sp_work = nstl::min(work_rest, max_spatial_work - spatial_start);
        const char *diff_dst
                = diff_dst_base + spatial_start_offset * jcp.dst_dsz;
        dim_t ocb = 0;
        const dim_t oc_tail_work = jcp.oc_tail ? jcp.oc_tail : jcp.oc_block;
        while (work_rest > 0) {
            for (dim_t iwork = 0; iwork < sp_work; iwork++) {
                auto ctx = jit_trans_dst_t::ctx_t();
                ctx.src = diff_dst;
                ctx.tr_src = tr_diff_dst;
                assert(ocb_start + ocb < jcp.nb_oc);
                ctx.ch_work = static_cast<int>(
                        (ocb_start + ocb + 1) == jcp.nb_oc ? oc_tail_work
                                                           : jcp.oc_block);
                ctx.src_prf = nullptr;
                ctx.tr_src_prf = nullptr;
                (*self->trans_dst_kernel_)(&ctx);
                diff_dst += diff_dst_stride;
                tr_diff_dst += tr_diff_dst_stride;
            }
            work_rest -= sp_work;
            sp_work = nstl::min(work_rest, max_spatial_work);
            ocb++;
            diff_dst = diff_dst_base + ocb * chb_stride;
        }
    }

    void maybe_global_transpose(dim_t img, dim_t ocb_s, dim_t ocb_e,
            dim_t icb_s, dim_t icb_e, dim_t od_s, dim_t odb_s, dim_t odb_e,
            dim_t oh_s, dim_t ohb_s, dim_t ohb_e) const {
        if (!jcp.global_transpose) return;

        using simple_barrier::barrier;
        // use tr_ic_block to transform src
        assert(jcp.ic_block % jcp.tr_ic_block == 0);
        const dim_t nb_tr_icb = jcp.ic_block / jcp.tr_ic_block;
        const dim_t tr_icb_s = icb_s * nb_tr_icb;
        const dim_t tr_icb_e
                = nstl::min<dim_t>(icb_e * nb_tr_icb, jcp.nb_tr_ic);
        const dim_t tr_icb_work = tr_icb_e - tr_icb_s;
        const dim_t ocb_work = ocb_e - ocb_s;

        // The barrier should stay outside of work condition to avoid
        // possible hang
        if (jcp.nthr_oc_b > 1)
            barrier(&tr_src_bctx[ithr_but_oc], jcp.nthr_oc_b);

        if (tr_icb_work > 0) {
            const auto id_s = get_id_start(od_s);
            const auto ih_s = get_ih_start(oh_s);

            const auto idb_s = get_id_start(odb_s);
            const auto idb_e = get_id_end(odb_e);

            const auto ihb_s = get_ih_start(ohb_s);
            const auto ihb_e = get_ih_end(ohb_e);

            dim_t work_amount
                    = g_work * tr_icb_work * (idb_e - idb_s) * (ihb_e - ihb_s);
            dim_t tr_start {0}, tr_end {0};
            balance211(work_amount, jcp.nthr_oc_b, ithr_oc_b, tr_start, tr_end);

            dim_t g {0}, tr_ic_b {0}, jd {0}, jh {0};
            nd_iterator_init(tr_start, g, g_work, tr_ic_b, tr_icb_work, jd,
                    idb_e - idb_s, jh, ihb_e - ihb_s);

            while (tr_start < tr_end) {
                dim_t g_ = g + g_start;
                dim_t tr_ic_b_ = tr_ic_b + tr_icb_s;

                dim_t jd_s = jd + idb_s;

                dim_t jh_s = jh + ihb_s;
                dim_t jh_e = jh_s + nstl::min(tr_end - tr_start, ihb_e - jh_s);

                const dim_t ic_off_idx
                        = g_ * jcp.ic + tr_ic_b_ * jcp.tr_ic_block;

                const char *p_src {nullptr};
                if (jcp.harness == harness_2d_reduction) {
                    p_src = &src[src_d.blk_off(img, ic_off_idx, jh_s)
                            * jcp.src_dsz];
                } else if (jcp.harness == harness_3d_reduction) {
                    p_src = &src[src_d.blk_off(img, ic_off_idx, jd_s, jh_s)
                            * jcp.src_dsz];
                } else
                    assert(!"Invalid harness type");

                char *p_tr_src = &tr_src[tr_ic_block_src_off(
                        g_, tr_ic_b_, jd_s - id_s, jh_s - ih_s)];
                trans_src_nxc(p_tr_src, p_src, tr_ic_b_, jh_e - jh_s, jh_s);

                nd_iterator_jump(tr_start, tr_end, g, g_work, tr_ic_b,
                        tr_icb_work, jd, idb_e - idb_s, jh, ihb_e - ihb_s);
            }
        }
        if (jcp.nthr_oc_b > 1)
            barrier(&tr_src_bctx[ithr_but_oc], jcp.nthr_oc_b);

        // The barrier should stay outside of work condition to avoid
        // possible hang
        if (jcp.nthr_ic_b > 1)
            barrier(&tr_diff_dst_bctx[ithr_but_ic], jcp.nthr_ic_b);

        if (ocb_work > 0) {
            dim_t jd = 0;
            dim_t jh = 0;
            dim_t work_amount
                    = g_work * ocb_work * (odb_e - odb_s) * (ohb_e - ohb_s);
            dim_t tr_start = 0;
            dim_t tr_end = 0;
            balance211(work_amount, jcp.nthr_ic_b, ithr_ic_b, tr_start, tr_end);

            dim_t g = 0;
            dim_t oc_b = 0;
            nd_iterator_init(tr_start, g, g_work, oc_b, ocb_work, jd,
                    odb_e - odb_s, jh, ohb_e - ohb_s);

            while (tr_start < tr_end) {
                dim_t g_ = g + g_start;
                dim_t oc_b_ = oc_b + ocb_s;
                dim_t jd_s = jd + odb_s;
                dim_t jh_s = jh + ohb_s;
                dim_t jh_e = jh_s + nstl::min(tr_end - tr_start, ohb_e - jh_s);
                const dim_t oc_off_idx = g_ * jcp.oc + oc_b_ * jcp.oc_block;

                const char *p_diff_dst {nullptr};
                if (jcp.harness == harness_2d_reduction) {
                    p_diff_dst = &diff_dst[diff_dst_d.blk_off(
                                                   img, oc_off_idx, jh_s)
                            * jcp.dst_dsz];
                } else if (jcp.harness == harness_3d_reduction) {
                    p_diff_dst = &diff_dst[diff_dst_d.blk_off(
                                                   img, oc_off_idx, jd_s, jh_s)
                            * jcp.dst_dsz];
                } else
                    assert(!"Invalid harness type");

                char *p_tr_diff_dst = &tr_diff_dst[tr_diff_dst_off(
                        g_, oc_b_, jd_s - od_s, jh_s - oh_s)];
                trans_dst_nxc(
                        p_tr_diff_dst, p_diff_dst, 0, 0, oc_b_, 0, jh_e - jh_s);

                nd_iterator_jump(tr_start, tr_end, g, g_work, oc_b, ocb_work,
                        jd, odb_e - odb_s, jh, ohb_e - ohb_s);
            }
        }
        if (jcp.nthr_ic_b > 1)
            barrier(&tr_diff_dst_bctx[ithr_but_ic], jcp.nthr_ic_b);
    }

    void maybe_local_traspose(char *&p_src, char *&p_dst, dim_t img, dim_t g,
            dim_t ic_b, dim_t oc_b, dim_t od_s, dim_t odb_s, dim_t odb_e,
            dim_t oh_s, dim_t ohb_s, dim_t ohb_e) const {

        const dim_t idb_s = get_id_start(odb_s);
        const dim_t ihb_s = get_ih_start(ohb_s);

        const dim_t idb_e = get_id_end(odb_e);
        const dim_t ihb_e = get_ih_end(ohb_e);

        const dim_t id_s = get_id_start(od_s);
        const dim_t ih_s = get_ih_start(oh_s);

        if (jcp.global_transpose) {
            p_src = &tr_src[tr_src_off(g, ic_b, 0, 0)];
            p_dst = &tr_diff_dst[tr_diff_dst_off(g, oc_b, 0, 0)];
            return;
        }

        const dim_t nb_ic_blocks = (ic_b + jcp.nb_ic_blocking > ic_b_end)
                ? 1
                : jcp.nb_ic_blocking;

        const dim_t nb_oc_blocks = (oc_b + jcp.nb_oc_blocking > oc_b_end)
                ? 1
                : jcp.nb_oc_blocking;

        for_(dim_t idb = idb_s; idb < idb_e; idb++)
        for (dim_t icb = 0; icb < nb_ic_blocks; icb++) {
            const dim_t ic_off_idx = g * jcp.ic + (ic_b + icb) * jcp.ic_block;
            char *p_tr_src
                    = &tr_src[tr_src_off(0, 0, idb - id_s, ihb_s - ih_s)];
            char *tr_src_local
                    = p_tr_src + icb * jcp.tr_src_block_size * jcp.src_dsz;
            const char *p_raw_src {nullptr};
            if (jcp.harness == harness_2d_reduction) {
                p_raw_src = (char *)&src[src_d.blk_off(img, ic_off_idx, ihb_s)
                        * jcp.src_dsz];
            } else if (jcp.harness == harness_3d_reduction) {
                p_raw_src = (char *)&src[src_d.blk_off(
                                                 img, ic_off_idx, idb, ihb_s)
                        * jcp.src_dsz];
            } else
                assert(!"Invalid harness type");
            trans_src_nxc(tr_src_local, p_raw_src,
                    (ic_b + icb) * (jcp.ic_block / jcp.tr_ic_block),
                    (ihb_e - ihb_s), ihb_s);
        }

        p_src = &tr_src[tr_src_off(0, 0, 0, 0)]; // p_tr_src;

        for_(dim_t odb = odb_s; odb < odb_e; odb++)
        for (dim_t ocb = 0; ocb < nb_oc_blocks; ocb++) {
            const dim_t oc_off_idx = g * jcp.oc + (oc_b + ocb) * jcp.oc_block;
            const char *p_raw_diff_dst {nullptr};
            if (jcp.harness == harness_2d_reduction) {
                p_raw_diff_dst
                        = &diff_dst[diff_dst_d.blk_off(img, oc_off_idx, ohb_s)
                                * jcp.dst_dsz];
            } else if (jcp.harness == harness_3d_reduction) {
                p_raw_diff_dst = &diff_dst[diff_dst_d.blk_off(
                                                   img, oc_off_idx, odb, ohb_s)
                        * jcp.dst_dsz];
            } else
                assert(!"Invalid harness type");
            char *p_tr_diff_dst = &tr_diff_dst[tr_diff_dst_off(
                    0, 0, odb - od_s, ohb_s - oh_s)];
            char *tr_diff_dst_local = p_tr_diff_dst
                    + ocb * jcp.tr_diff_dst_block_size * jcp.dst_dsz;
            trans_dst_nxc(tr_diff_dst_local, p_raw_diff_dst, 0, 0, (oc_b + ocb),
                    0, (ohb_e - ohb_s));
        }
        p_dst = &tr_diff_dst[tr_diff_dst_off(0, 0, 0, 0)]; //   p_tr_diff_dst;
    }

    bool just_init_output(
            dim_t start, dim_t end, float *diff_wei, float *diff_bias) {
        if (g_start >= g_end || oc_b_start >= oc_b_end
                || ic_b_start >= ic_b_end)
            return false;
        if (start >= end) {
            // for rare case if thread has no work by spatial dimension then we
            // need to initialize the output at least
            if (jcp.with_bias) {
                for_(dim_t g = g_start; g < g_end; ++g)
                {
                    void *p_bias = diff_bias + g * rnd_up(jcp.oc, jcp.oc_block)
                            + oc_b_start * jcp.oc_block;
                    auto bias_amount = (oc_b_end - oc_b_start) * jcp.oc_block;
                    std::memset(p_bias, 0, bias_amount * jcp.acc_dsz);
                }
            }

            for_(dim_t g = g_start; g < g_end; ++g)
            for (dim_t oc_b = oc_b_start; oc_b < oc_b_end; oc_b++) {
                auto wei_offs_ext = pd()->ndims() == 3
                        ? wht_blk_off(diff_weights_d, g, oc_b, ic_b_start, 0)
                        : (pd()->ndims() == 4
                                          ? wht_blk_off(diff_weights_d, g, oc_b,
                                                    ic_b_start, 0, 0)
                                          : wht_blk_off(diff_weights_d, g, oc_b,
                                                    ic_b_start, 0, 0, 0));
                void *ptr_C = (jcp.transform_to_vnni) ? diff_wei
                                + self->wei_offset_int(
                                        g, oc_b, ic_b_start, 0, 0, 0)
                                                      : diff_wei + wei_offs_ext;

                auto C_amount = jcp.kd * jcp.kh * jcp.kw
                        * (ic_b_end - ic_b_start) * jcp.ic_block * jcp.oc_block;

                std::memset(ptr_C, 0, C_amount * jcp.acc_dsz);
            }
            return true;
        }
        if (jcp.M < jcp.ic_block * jcp.nb_ic_blocking) {
            // For small ic we may calculate only needed part of diff_weights.
            // So we have to initialize diff_weights
            // TODO: initialize only not calculated part of diff_weights
            for_(dim_t g = g_start; g < g_end; ++g)
            for (dim_t oc_b = oc_b_start; oc_b < oc_b_end; oc_b++) {
                auto wei_offs_ext = pd()->ndims() == 3
                        ? wht_blk_off(diff_weights_d, g, oc_b, ic_b_start, 0)
                        : (pd()->ndims() == 4
                                          ? wht_blk_off(diff_weights_d, g, oc_b,
                                                    ic_b_start, 0, 0)
                                          : wht_blk_off(diff_weights_d, g, oc_b,
                                                    ic_b_start, 0, 0, 0));
                void *ptr_C = (jcp.transform_to_vnni) ? diff_wei
                                + self->wei_offset_int(
                                        g, oc_b, ic_b_start, 0, 0, 0)
                                                      : diff_wei + wei_offs_ext;

                auto C_amount = jcp.kd * jcp.kh * jcp.kw
                        * (ic_b_end - ic_b_start) * jcp.ic_block * jcp.oc_block;

                std::memset(ptr_C, 0, C_amount * jcp.acc_dsz);
            }
            return false;
        }
        return false;
    }
};

void brgemm_convolution_bwd_weights_t::call_brgemm_kernel(
        thread_info_t &btc, int brg_idx, dim_t batch_size, void *ptr_C) const {

    const auto brg_ker = brg_kernels_[brg_idx];
    assert(brg_ker != nullptr);

    const auto &jcp = pd()->jcp_;
    const bool is_amx = brgemm_convolution_utils::is_amx(jcp.isa);
    brgemm_palettes_.maybe_tile_configure(is_amx, btc.cur_brg_idx, brg_idx);

    brgemm_kernel_execute(brg_ker, batch_size, btc.brg_batch, ptr_C,
            static_cast<void *>(btc.wsp_tile));
}

void brgemm_convolution_bwd_weights_t::compute_diff_weights_2d(
        thread_info_t *ti) const {

    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const dim_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kd * jcp.kh * jcp.kw;
    const dim_t bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    const dim_t optimal_spblock = jcp.spatial_blk_size;

    float *diff_wei;
    if (diff_weights_d.data_type() != data_type::f32)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    float *diff_bias = nullptr;
    if (jcp.with_bias) {
        if (jcp.bia_dt != data_type::f32)
            diff_bias = ti->bia_reduction + (ti->ithr_mb) * bias_buf_size;
        else
            diff_bias = ti->ithr_mb == 0
                    ? (float *)ti->diff_bias
                    : ti->bia_reduction + (ti->ithr_mb - 1) * bias_buf_size;
    }

    dim_t img {0}, oh_s {0};
    dim_t start = ti->img_start;
    dim_t end = ti->img_end;

    dim_t img_s {0};
    nd_iterator_init(start, img_s, jcp.mb, oh_s, jcp.oh);
    img = img_s;

    auto do_brgemm_call
            = [&](dim_t g, dim_t bs, dim_t ic_b, dim_t oc_b, dim_t ohb_s,
                      dim_t bs_ih_s, const char *p_src, const char *p_dst,
                      dim_t kh, dim_t kw, bool do_init) {
        const dim_t ihb_s = ti->get_ih_start(ohb_s);

        const dim_t bs_oh_s = utils::saturate<dim_t>(0, jcp.oh,
                (bs_ih_s + jcp.t_pad - kh * (jcp.dilate_h + 1)) / jcp.stride_h);

        auto ocb_end = get_end(oc_b, jcp.nb_oc_blocking, ti->oc_b_end);
        auto icb_end = get_end(ic_b, jcp.nb_ic_blocking, ti->ic_b_end);
        const dim_t src_stride_w_shift = jcp.tr_iw / jcp.stride_w;
        const auto a_off = _pd->filter_w_to_src(kw) / jcp.stride_w
                + (kw % jcp.stride_w) * src_stride_w_shift
                + (bs_ih_s - ihb_s) * jcp.tr_iw * jcp.ic_block;
        const char *ptr_A = p_src + a_off * jcp.src_dsz;
        const auto b_off = (bs_oh_s - ohb_s) * jcp.tr_ow * jcp.oc_block;
        const char *ptr_B = p_dst + b_off * jcp.dst_dsz;

        void *ptr_C = (jcp.transform_to_vnni)
                ? diff_wei + wei_offset_int(g, oc_b, ic_b, 0, kh, kw)
                : diff_wei
                        + (pd()->ndims() == 3 ? wht_blk_off(diff_weights_d, g,
                                                        oc_b, ic_b, kw)
                                              : wht_blk_off(diff_weights_d, g,
                                                        oc_b, ic_b, kh, kw));
        bool M_tail = (icb_end < ic_b + jcp.nb_ic_blocking);
        bool N_tail = (ocb_end < oc_b + jcp.nb_oc_blocking);

        auto brg_idx = _pd->get_brg_idx(
                bs, M_tail ? jcp.M_tail : jcp.M, do_init, N_tail, false);

        for (dim_t ohb = 0; ohb < bs; ohb++) {
            ti->brg_batch[ohb].ptr.A = (char *)ptr_A
                    + ohb * jcp.tr_iw * jcp.ic_block * jcp.stride_h
                            * jcp.src_dsz;
            ti->brg_batch[ohb].ptr.B = (char *)ptr_B
                    + ohb * jcp.tr_ow * jcp.oc_block * jcp.dst_dsz;
        }

        call_brgemm_kernel(*ti, brg_idx, bs, ptr_C);
    };

    if (ti->just_init_output(start, end, diff_wei, diff_bias)) return;

    while (start < end) {
        const dim_t oh_e = _pd->get_finish_oh(
                oh_s, start, get_end(start, jcp.oh_block, end));
        dim_t height_block
                = jcp.global_transpose ? oh_e - oh_s : optimal_spblock;

        // loop by ohb_s have only one iteration for global_transpose case
        // because height_block = oh_e - oh_s
        for (dim_t ohb_s = oh_s; ohb_s < oh_e; ohb_s += height_block) {
            const dim_t ohb_e = get_end(ohb_s, height_block, oh_e);
            assert(ohb_e <= jcp.oh);

            ti->maybe_global_transpose(img,
                    jcp.tr_ocb_chunk ? 0 : ti->oc_b_start,
                    jcp.tr_ocb_chunk ? 0 : ti->oc_b_end,
                    jcp.tr_icb_chunk ? 0 : ti->ic_b_start,
                    jcp.tr_icb_chunk ? 0 : ti->ic_b_end, 0, 0, 1, oh_s, ohb_s,
                    ohb_e);

            for_(dim_t g = ti->g_start; g < ti->g_end; ++g)
            for (dim_t oc_b = ti->oc_b_start; oc_b < ti->oc_b_end;
                    oc_b += jcp.nb_oc_blocking) {
                const dim_t oc_b_e
                        = get_end(oc_b, jcp.nb_oc_blocking, ti->oc_b_end);

                if (jcp.tr_ocb_chunk)
                    ti->maybe_global_transpose(img, oc_b, oc_b_e, 0, 0, 0, 0, 1,
                            oh_s, ohb_s, ohb_e);

                for (dim_t ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                        ic_b += jcp.nb_ic_blocking) {

                    const dim_t ic_b_e
                            = get_end(ic_b, jcp.nb_ic_blocking, ti->ic_b_end);

                    if (oc_b == ti->oc_b_start && jcp.tr_icb_chunk)
                        ti->maybe_global_transpose(img, 0, 0, ic_b, ic_b_e, 0,
                                0, 1, oh_s, ohb_s, ohb_e);

                    char *p_src {nullptr};
                    char *p_dst {nullptr};
                    ti->maybe_local_traspose(p_src, p_dst, img, g, ic_b, oc_b,
                            0, 0, 1, oh_s, ohb_s, ohb_e);

                    if (jcp.with_bias && ic_b == 0) {
                        auto bp = jit_conv_args_t();

                        bp.bias = diff_bias + g * rnd_up(jcp.oc, jcp.oc_block)
                                + oc_b * jcp.oc_block;
                        bp.channel
                                = (start == ti->img_start) && (ohb_s == oh_s);

                        bp.os_index_begin = ohb_s;
                        bp.os_index_end = ohb_e;

                        bp.last_oc_block
                                = ((oc_b_e - oc_b) == jcp.nb_oc_blocking) ? 0
                                                                          : 1;

                        bp.dst = p_dst;

                        (*diff_bias_kernel_)(&bp);
                    }

                    if (ti->g_start == ti->g_end
                            || ti->oc_b_start == ti->oc_b_end
                            || ti->ic_b_start == ti->ic_b_end)
                        continue;

                    const auto do_init = (start == ti->img_start);

                    for (dim_t kh = 0; kh < jcp.kh; kh++) {
                        const dim_t bs_ih_s = _pd->get_start_ih(kh, ohb_s);
                        const dim_t bs_ih_e = _pd->get_finish_ih(kh, ohb_e);
                        const auto bs
                                = div_up(nstl::max<dim_t>(0, bs_ih_e - bs_ih_s),
                                        jcp.stride_h);
                        if (bs == 0 && !do_init) continue;

                        for_(dim_t s = 0; s < jcp.stride_w; s++)
                        for (dim_t kw = s; kw < jcp.kw; kw += jcp.stride_w)
                            do_brgemm_call(g, bs, ic_b, oc_b, ohb_s, bs_ih_s,
                                    p_src, p_dst, kh, kw, do_init);
                    }
                }
            }
        }

        dim_t jump_start = start;
        const dim_t jump_end = get_end(start, jcp.oh_block, end);
        nd_iterator_jump(jump_start, jump_end, img, jcp.mb, oh_s, jcp.oh);
        start = jump_start;
    }
}

void brgemm_convolution_bwd_weights_t::compute_diff_weights_3d(
        thread_info_t *ti) const {

    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const dim_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kd * jcp.kh * jcp.kw;
    const dim_t bias_buf_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    const dim_t optimal_spblock = jcp.spatial_blk_size;

    float *diff_wei;
    if (diff_weights_d.data_type() != data_type::f32)
        diff_wei = ti->wei_bia_reduction + (ti->ithr_mb) * wei_size;
    else
        diff_wei = ti->ithr_mb == 0
                ? (float *)ti->diff_weights
                : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    float *diff_bias = nullptr;
    if (jcp.with_bias) {
        if (jcp.bia_dt != data_type::f32)
            diff_bias = ti->bia_reduction + (ti->ithr_mb) * bias_buf_size;
        else
            diff_bias = ti->ithr_mb == 0
                    ? (float *)ti->diff_bias
                    : ti->bia_reduction + (ti->ithr_mb - 1) * bias_buf_size;
    }

    dim_t img {0}, od_s {0};
    dim_t start = ti->img_start;
    dim_t end = ti->img_end;

    dim_t img_s {0};
    nd_iterator_init(start, img_s, jcp.mb, od_s, jcp.od);
    img = img_s;

    auto do_brgemm_call
            = [&](dim_t g, dim_t bs_d, dim_t bs_h, dim_t ic_b, dim_t oc_b,
                      dim_t od_s, dim_t oh_s, dim_t bs_id_s, dim_t bs_ih_s,
                      const char *p_src, const char *p_dst, dim_t kd, dim_t kh,
                      dim_t kw, bool do_init) {
        const dim_t id_s = ti->get_id_start(od_s);
        const dim_t ih_s = ti->get_ih_start(oh_s);

        const dim_t bs_od_s = utils::saturate<dim_t>(0, jcp.od,
                (bs_id_s + jcp.f_pad - kd * (jcp.dilate_d + 1)) / jcp.stride_d);

        const dim_t bs_oh_s = utils::saturate<dim_t>(0, jcp.oh,
                (bs_ih_s + jcp.t_pad - kh * (jcp.dilate_h + 1)) / jcp.stride_h);

        auto ocb_end = get_end(oc_b, jcp.nb_oc_blocking, ti->oc_b_end);
        auto icb_end = get_end(ic_b, jcp.nb_ic_blocking, ti->ic_b_end);
        const dim_t src_stride_w_shift = jcp.tr_iw / jcp.stride_w;
        const auto a_off = _pd->filter_w_to_src(kw) / jcp.stride_w
                + (kw % jcp.stride_w) * src_stride_w_shift
                + (bs_ih_s - ih_s) * jcp.tr_iw * jcp.ic_block
                + (bs_id_s - id_s) * jcp.ih_block * jcp.tr_iw * jcp.ic_block;
        const char *ptr_A = p_src + a_off * jcp.src_dsz;
        const auto b_off = (bs_oh_s - oh_s) * jcp.tr_ow * jcp.oc_block
                + (bs_od_s - od_s) * jcp.oh_block * jcp.tr_ow * jcp.oc_block;
        const char *ptr_B = p_dst + b_off * jcp.dst_dsz;
        void *ptr_C = (jcp.transform_to_vnni)
                ? diff_wei + wei_offset_int(g, oc_b, ic_b, kd, kh, kw)
                : diff_wei
                        + wht_blk_off(
                                diff_weights_d, g, oc_b, ic_b, kd, kh, kw);
        bool M_tail = (icb_end < ic_b + jcp.nb_ic_blocking);
        bool N_tail = (ocb_end < oc_b + jcp.nb_oc_blocking);

        const auto bs = bs_d * bs_h;
        auto brg_idx = _pd->get_brg_idx(
                bs, M_tail ? jcp.M_tail : jcp.M, do_init, N_tail, false);

        for (dim_t odb = 0; odb < bs_d; odb++) {
            for (dim_t ohb = 0; ohb < bs_h; ohb++) {
                const auto a_off_batch = (odb * jcp.ih_block * jcp.stride_d
                                                 + ohb * jcp.stride_h)
                        * jcp.tr_iw * jcp.ic_block * jcp.src_dsz;
                ti->brg_batch[odb * bs_h + ohb].ptr.A
                        = (char *)ptr_A + a_off_batch;
                const auto b_off_batch = (odb * jcp.oh_block + ohb) * jcp.tr_ow
                        * jcp.oc_block * jcp.dst_dsz;
                ti->brg_batch[odb * bs_h + ohb].ptr.B
                        = (char *)ptr_B + b_off_batch;
            }
        }

        call_brgemm_kernel(*ti, brg_idx, bs, ptr_C);
    };

    if (ti->just_init_output(start, end, diff_wei, diff_bias)) return;

    const auto oh_s = 0;
    const auto oh_e = jcp.oh;

    while (start < end) {
        const dim_t od_e = _pd->get_finish_od(
                od_s, start, get_end(start, jcp.od_block, end));
        dim_t sp_block = jcp.global_transpose ? od_e - od_s : optimal_spblock;

        // loop by odb_s have only one iteration for global_transpose case
        // because sp_block = od_e - od_s
        for (dim_t odb_s = od_s; odb_s < od_e; odb_s += sp_block) {
            const dim_t odb_e = get_end(odb_s, sp_block, od_e);
            assert(odb_e <= jcp.od);

            for (dim_t ohb_s = oh_s; ohb_s < oh_e; ohb_s += jcp.oh_block) {
                const auto ohb_e = get_end(ohb_s, jcp.oh_block, jcp.oh);

                ti->maybe_global_transpose(img,
                        jcp.tr_ocb_chunk ? 0 : ti->oc_b_start,
                        jcp.tr_ocb_chunk ? 0 : ti->oc_b_end,
                        jcp.tr_icb_chunk ? 0 : ti->ic_b_start,
                        jcp.tr_icb_chunk ? 0 : ti->ic_b_end, od_s, odb_s, odb_e,
                        oh_s, ohb_s, ohb_e);

                for_(dim_t g = ti->g_start; g < ti->g_end; ++g)
                for (dim_t oc_b = ti->oc_b_start; oc_b < ti->oc_b_end;
                        oc_b += jcp.nb_oc_blocking) {
                    const dim_t oc_b_e
                            = get_end(oc_b, jcp.nb_oc_blocking, ti->oc_b_end);
                    if (jcp.tr_ocb_chunk)
                        ti->maybe_global_transpose(img, oc_b, oc_b_e, 0, 0,
                                od_s, odb_s, odb_e, oh_s, ohb_s, ohb_e);

                    for (dim_t ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                            ic_b += jcp.nb_ic_blocking) {

                        const dim_t ic_b_e = get_end(
                                ic_b, jcp.nb_ic_blocking, ti->ic_b_end);

                        if (oc_b == ti->oc_b_start && jcp.tr_icb_chunk)
                            ti->maybe_global_transpose(img, 0, 0, ic_b, ic_b_e,
                                    od_s, odb_s, odb_e, oh_s, ohb_s, ohb_e);

                        char *p_src {nullptr};
                        char *p_dst {nullptr};
                        ti->maybe_local_traspose(p_src, p_dst, img, g, ic_b,
                                oc_b, od_s, odb_s, odb_e, oh_s, ohb_s, ohb_e);

                        if (jcp.with_bias && ic_b == 0) {
                            for (dim_t iodb = odb_s; iodb < odb_e; iodb++) {
                                auto bp = jit_conv_args_t();

                                bp.bias = diff_bias
                                        + g * rnd_up(jcp.oc, jcp.oc_block)
                                        + oc_b * jcp.oc_block;
                                bp.os_index_begin = ohb_s;
                                bp.os_index_end = ohb_e;

                                bp.last_oc_block
                                        = ((oc_b_e - oc_b)
                                                  == jcp.nb_oc_blocking)
                                        ? 0
                                        : 1;

                                bp.channel = (start == ti->img_start)
                                        && (odb_s == od_s) && (iodb == odb_s)
                                        && (ohb_s == oh_s);
                                const auto dst_idx
                                        = ((iodb - od_s) * jcp.oh_block
                                                  + (ohb_s - oh_s))
                                        * jcp.tr_ow * jcp.oc_block;
                                bp.dst = p_dst + dst_idx * jcp.dst_dsz;
                                (*diff_bias_kernel_)(&bp);
                            }
                        }

                        if (ti->g_start == ti->g_end
                                || ti->oc_b_start == ti->oc_b_end
                                || ti->ic_b_start == ti->ic_b_end)
                            continue;

                        const auto do_init
                                = (start == ti->img_start && ohb_s == oh_s);

                        for (dim_t kd = 0; kd < jcp.kd; kd++) {
                            const dim_t bs_id_s = _pd->get_start_id(kd, odb_s);
                            const dim_t bs_id_e = _pd->get_finish_id(kd, odb_e);
                            const auto bs_d = div_up(
                                    nstl::max<dim_t>(0, bs_id_e - bs_id_s),
                                    jcp.stride_d);
                            // bs_d may be 0 but we may still need to call brgemm to
                            // initialize output
                            if (bs_d == 0 && !do_init) continue;

                            for (dim_t kh = 0; kh < jcp.kh; kh++) {
                                const dim_t bs_ih_s
                                        = _pd->get_start_ih(kh, ohb_s);
                                const dim_t bs_ih_e
                                        = _pd->get_finish_ih(kh, ohb_e);
                                const auto bs_h = div_up(
                                        nstl::max<dim_t>(0, bs_ih_e - bs_ih_s),
                                        jcp.stride_h);
                                if (bs_h == 0 && !do_init) continue;

                                for_(dim_t s = 0; s < jcp.stride_w; s++)
                                for (dim_t kw = s; kw < jcp.kw;
                                        kw += jcp.stride_w)
                                    do_brgemm_call(g, bs_d, bs_h, ic_b, oc_b,
                                            od_s, oh_s, bs_id_s, bs_ih_s, p_src,
                                            p_dst, kd, kh, kw, do_init);
                            }
                        }
                    }
                }
            }
        }

        dim_t jump_start = start;
        const dim_t jump_end = get_end(start, jcp.od_block, end);
        nd_iterator_jump(jump_start, jump_end, img, jcp.mb, od_s, jcp.od);
        start = jump_start;
    }
}

void brgemm_convolution_bwd_weights_t::store_in_vnni_format(
        thread_info_t *ti) const {
    const auto &jcp = pd()->jcp_;
    if (one_of(0, ti->g_work, ti->oc_b_work, ti->ic_b_work)) return;

    const int vnni_granularity
            = static_cast<int>(data_type_vnni_granularity(jcp.wei_dt));
    if (vnni_granularity == 0) {
        assert(!"Invalid vnni granularity.");
        return;
    }

    const auto icb2_work = div_up(ti->ic_b_work, vnni_granularity);
    const auto work = ti->g_work * ti->oc_b_work * icb2_work;

    dim_t start {0}, end {0};
    balance211(work, jcp.nthr_mb, ti->ithr_mb, start, end);
    dim_t sub_g_start {0}, sub_oc_b_start {0}, sub_icb2_start {0};
    nd_iterator_init(start, sub_g_start, ti->g_work, sub_oc_b_start,
            ti->oc_b_work, sub_icb2_start, icb2_work);
    for (dim_t w = start; w < end; w++) {
        const dim_t g = ti->g_start + sub_g_start;
        const dim_t oc_b = ti->oc_b_start + sub_oc_b_start;
        const dim_t ic_b = ti->ic_b_start + vnni_granularity * sub_icb2_start;
        jit_conv_args_t p = jit_conv_args_t();

        float *input = ti->wei_bia_reduction + wei_offset_int(g, oc_b, ic_b, 0);
        char *output = (char *)ti->diff_weights
                + wei_offset_ext(g, oc_b, ic_b) * jcp.wei_dsz;

        p.src = (void *)input;
        p.dst = (void *)output;
        p.last_ic_block = ((ic_b + vnni_granularity) > jcp.nb_ic) ? 1 : 0;
        (*diff_wei_trans_kernel_)(&p);
        nd_iterator_step(sub_g_start, ti->g_work, sub_oc_b_start, ti->oc_b_work,
                sub_icb2_start, icb2_work);
    }
}

void brgemm_convolution_bwd_weights_t::reduce_and_convert_diff_weights_and_bias(
        thread_info_t *ti) const {

    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = pd()->jcp_;
    const dim_t wei_size = jcp.ngroups * jcp.nb_oc * jcp.oc_block * jcp.nb_ic
            * jcp.ic_block * jcp.kh * jcp.kw * ((jcp.ndims == 5) ? jcp.kd : 1);

    const auto wei_dt = diff_weights_d.data_type();
    const auto bia_dt = jcp.bia_dt;
    const bool is_f32_out = wei_dt == data_type::f32;
    const bool is_f32_bias = bia_dt == data_type::f32;

    if (jcp.nthr_mb == 1) {
        if (!is_f32_out) {
            // reduction is not required, only conversion
            if (jcp.transform_to_vnni) {
                store_in_vnni_format(ti);
            } else {
                for_(dim_t g = ti->g_start; g < ti->g_end; g++)
                for (dim_t oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; oc_b++) {
                    const size_t acc_size = (size_t)ti->ic_b_work * jcp.kh
                            * jcp.kw * ((jcp.ndims == 5) ? jcp.kd : 1)
                            * jcp.ic_block * jcp.oc_block;
                    const size_t off = wht_blk_off(
                            diff_weights_d, g, oc_b, ti->ic_b_start);
                    types::cvt_from_float(wei_dt,
                            (void *)(ti->diff_weights + off * jcp.wei_dsz),
                            (ti->wei_bia_reduction + off), acc_size);
                }
            }
        }
        if (pd()->with_bias() && !is_f32_bias && ti->ithr_ic_b == 0
                && ti->ic_b_work > 0) {
            for (dim_t g = ti->g_start; g < ti->g_end; g++) {
                dim_t result_start_idx
                        = g * jcp.oc + ti->oc_b_start * jcp.oc_block;
                dim_t buffer_start_idx = g * rnd_up(jcp.oc, jcp.oc_block)
                        + ti->oc_b_start * jcp.oc_block;
                const size_t acc_size
                        = nstl::min(jcp.oc, ti->oc_b_end * jcp.oc_block)
                        - ti->oc_b_start * jcp.oc_block;
                void *diff_bias = (char *)ti->diff_bias
                        + result_start_idx * jcp.bia_dsz;
                float *buffer = ti->bia_reduction + buffer_start_idx;
                types::cvt_from_float(
                        bia_dt, diff_bias, (const float *)buffer, acc_size);
            }
        }
        return;
    }

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    if (jcp.global_transpose)
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, jcp.nthr);

    const dim_t ic_b_kh_work
            = ti->ic_b_work * ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
    if (ic_b_kh_work <= 0 || ti->oc_b_work == 0 || ti->g_work == 0) {
        // TODO: double check if a barrier is needed here
        // and at the end of function
        if (jcp.transform_to_vnni && jcp.global_transpose)
            simple_barrier::barrier(ti->wei_bia_reduction_bctx, jcp.nthr);
        return;
    }

    const dim_t work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    dim_t start {0}, end {0};
    balance211(work, jcp.nthr_mb, ti->ithr_mb, start, end);
    if (!jcp.transform_to_vnni && start == end) return;

    const int _start_nthr_mb = 1;
    for (int thr_mb = _start_nthr_mb; thr_mb < jcp.nthr_mb; ++thr_mb) {
        dim_t w = start;
        dim_t sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_kh_start {0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const dim_t g = ti->g_start + sub_g_start;
            const dim_t oc_b = ti->oc_b_start + sub_oc_b_start;
            const dim_t ic_b = ti->ic_b_start
                    + sub_ic_b_kh_start / ((jcp.ndims == 5) ? jcp.kd : jcp.kh);
            const dim_t kX
                    = sub_ic_b_kh_start % ((jcp.ndims == 5) ? jcp.kd : jcp.kh);

            const size_t acc_size = (size_t)jcp.kw * jcp.ic_block * jcp.oc_block
                    * ((jcp.ndims == 5) ? jcp.kh : 1)
                    * nstl::min(end - w, ic_b_kh_work - sub_ic_b_kh_start);

            const size_t off_ext
                    = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kX);
            const size_t off_int = (jcp.transform_to_vnni)
                    ? wei_offset_int(g, oc_b, ic_b, kX)
                    : off_ext;

            float *wei_reduced = is_f32_out
                    ? (float *)(ti->diff_weights) + off_ext
                    : ti->wei_bia_reduction + off_int;

            int thr_mb_buffer_idx = is_f32_out ? thr_mb - 1 : thr_mb;
            float *wei_to_reduce = ti->wei_bia_reduction
                    + thr_mb_buffer_idx * wei_size + off_int;

            if (!jcp.transform_to_vnni && !is_f32_out
                    && thr_mb == jcp.nthr_mb - 1) {
                // the last iteration for bfloat16 requires conversion and
                // store to diff_weights array
                if (wei_dt == bf16)
                    add_floats_and_cvt_to_bfloat16(
                            (bfloat16_t *)(ti->diff_weights) + off_ext,
                            wei_reduced, wei_to_reduce, acc_size);
                else if (wei_dt == f16)
                    add_floats_and_cvt_to_float16(
                            (float16_t *)(ti->diff_weights) + off_ext,
                            wei_reduced, wei_to_reduce, acc_size);
                else if (wei_dt == f8_e5m2)
                    add_floats_and_cvt_to_f8_e5m2(
                            (float8_e5m2_t *)(ti->diff_weights) + off_ext,
                            wei_reduced, wei_to_reduce, acc_size);
                else if (wei_dt == f8_e4m3)
                    add_floats_and_cvt_to_f8_e4m3(
                            (float8_e4m3_t *)(ti->diff_weights) + off_ext,
                            wei_reduced, wei_to_reduce, acc_size);
            } else
                acc_ker_->accumulate(wei_reduced, wei_to_reduce, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
        if (jcp.with_bias && ti->ithr_ic_b == 0 && ti->ic_b_work > 0
                && ti->ithr_mb == 0 && ti->img_work > 0) {
            for (dim_t g = ti->g_start; g < ti->g_end; g++) {
                float *bias_reduced = is_f32_bias ? (float *)(ti->diff_bias)
                                                  : ti->bia_reduction;
                int thr_mb_buffer_idx = is_f32_bias ? thr_mb - 1 : thr_mb;
                const dim_t bias_buf_size
                        = jcp.ngroups * jcp.nb_oc * jcp.oc_block;
                float *bias_to_reduce
                        = ti->bia_reduction + thr_mb_buffer_idx * bias_buf_size;
                const size_t acc_size
                        = nstl::min(jcp.oc, ti->oc_b_end * jcp.oc_block)
                        - ti->oc_b_start * jcp.oc_block;
                dim_t idx = g * rnd_up(jcp.oc, jcp.oc_block)
                        + ti->oc_b_start * jcp.oc_block;
                if (!is_f32_bias && thr_mb == jcp.nthr_mb - 1) {
                    // the last iteration for bfloat16 requires conversion and
                    // store to diff_weights array
                    dim_t diff_bias_idx
                            = g * jcp.oc + ti->oc_b_start * jcp.oc_block;
                    if (bia_dt == bf16)
                        add_floats_and_cvt_to_bfloat16(
                                (bfloat16_t *)(ti->diff_bias) + diff_bias_idx,
                                &bias_reduced[idx], &bias_to_reduce[idx],
                                acc_size);
                    else if (bia_dt == f16)
                        add_floats_and_cvt_to_float16(
                                (float16_t *)(ti->diff_bias) + diff_bias_idx,
                                &bias_reduced[idx], &bias_to_reduce[idx],
                                acc_size);
                    else if (bia_dt == f8_e5m2)
                        add_floats_and_cvt_to_f8_e5m2(
                                (float8_e5m2_t *)(ti->diff_bias)
                                        + diff_bias_idx,
                                &bias_reduced[idx], &bias_to_reduce[idx],
                                acc_size);
                    else if (bia_dt == f8_e4m3)
                        add_floats_and_cvt_to_f8_e4m3(
                                (float8_e4m3_t *)(ti->diff_bias)
                                        + diff_bias_idx,
                                &bias_reduced[idx], &bias_to_reduce[idx],
                                acc_size);
                } else {
                    acc_ker_->accumulate(
                            &bias_reduced[idx], &bias_to_reduce[idx], acc_size);
                }
            }
        }
    }

    if (jcp.transform_to_vnni && jcp.global_transpose) {
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, jcp.nthr);
        store_in_vnni_format(ti);
    }
}

void brgemm_convolution_bwd_weights_t::prepare_scratchpad_data(
        const exec_ctx_t &ctx) const {
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = pd()->jcp_;

    auto tr_src = scratchpad.template get<char>(key_conv_tr_src);
    const auto bytes_to_zero = jcp.src_dsz * jcp.tr_src_num_guard_elems;
    if (jcp.oh_block < jcp.oh || jcp.id > 1) {
        // if (oh_block < oh) or (id > 1) then we zero end of each row because
        // last elements position may vary depending on position of od_s,
        // oh_block, padding and kh
        parallel_nd(jcp.tr_src_buf_count, jcp.ih_block * jcp.id,
                [=](size_t isb, size_t is) {
            const auto tr_src_idx = isb * jcp.tr_src_buf_size
                    + (is + 1) * jcp.tr_iw * jcp.ic_block;
            char *ts = &tr_src[tr_src_idx * jcp.src_dsz];
            std::memset(ts, 0, bytes_to_zero);
        });
        // Zero out last guard elements
        char *ts = &tr_src[jcp.tr_src_buf_count * jcp.tr_src_buf_size
                * jcp.src_dsz];
        std::memset(ts, 0, bytes_to_zero);
    } else {
        // Zero out guard elements that cross a buffer boundary to prevent a
        // race condition due to buffer overflows from memory optimization where
        // buffers sharing padding
        parallel_nd(jcp.tr_src_buf_count, [=](size_t isb) {
            char *ts = &tr_src[(isb + 1) * jcp.tr_src_buf_size * jcp.src_dsz];
            std::memset(ts, 0, bytes_to_zero);
        });
    }

    parallel(1, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        if (jcp.global_transpose && jcp.nthr_oc_b > 1) {
            const int tr_src_bctx_size = jcp.nthr / jcp.nthr_oc_b;
            auto tr_src_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                    key_conv_tr_src_bctx);
            for (int i = 0; i < tr_src_bctx_size; ++i)
                simple_barrier::ctx_init(&tr_src_bctx[i]);
        }
        if (jcp.global_transpose) {
            if (jcp.nthr_ic_b > 1) {
                const int tr_diff_dst_bctx_size = jcp.nthr / jcp.nthr_ic_b;
                auto tr_diff_dst_bctx
                        = scratchpad.template get<simple_barrier::ctx_t>(
                                key_conv_tr_diff_dst_bctx);
                for (int i = 0; i < tr_diff_dst_bctx_size; ++i)
                    simple_barrier::ctx_init(&tr_diff_dst_bctx[i]);
            }
        }

        if (jcp.nthr_mb > 1
                || pd()->diff_weights_md(0)->data_type != data_type::f32) {
            // TODO: don't use barrier for case
            // diff_weights_type != data_type::f32 && nthr_mb_ == 1
            simple_barrier::ctx_init(
                    scratchpad.template get<simple_barrier::ctx_t>(
                            key_conv_wei_bia_reduction_bctx));
        }
    });
}

void brgemm_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    prepare_scratchpad_data(ctx);

    const auto &jcp = pd()->jcp_;

    parallel(jcp.nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        assert(jcp.nthr == nthr);
        assert(utils::one_of(pd()->ndims(), 3, 4, 5));

        thread_info_t thread_info(this, ctx, ithr);
        switch (jcp.harness) {
            case harness_2d_reduction:
                compute_diff_weights_2d(&thread_info);
                if (jcp.global_transpose)
                    reduce_and_convert_diff_weights_and_bias(&thread_info);
                break;
            case harness_3d_reduction:
                compute_diff_weights_3d(&thread_info);
                if (jcp.global_transpose)
                    reduce_and_convert_diff_weights_and_bias(&thread_info);
                break;
            default: assert(!"Invalid harness type");
        }

        amx_tile_release();
    });

    if (!jcp.global_transpose) {
        parallel(jcp.nthr,
                [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
            assert(jcp.nthr == nthr);
            thread_info_t thread_info(this, ctx, ithr);
            reduce_and_convert_diff_weights_and_bias(&thread_info);
        });
    }

    if (jcp.transform_to_vnni && !jcp.global_transpose) {
        parallel(jcp.nthr,
                [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
            assert(jcp.nthr == nthr);
            thread_info_t thread_info(this, ctx, ithr);
            store_in_vnni_format(&thread_info);
        });
    }

    parallel(1, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        if (pd()->with_bias() && (jcp.oc % jcp.oc_block != 0)
                && jcp.bia_dt == data_type::f32) {
            auto diff_bias
                    = ctx.get_scratchpad_grantor().template get<const float>(
                            key_conv_padded_bias);
            auto diff_bias_in = CTX_OUT_MEM(float *, DNNL_ARG_DIFF_BIAS);
            const dim_t padded_stride = rnd_up(jcp.oc, jcp.oc_block);
            const dim_t stride = jcp.oc;
            for (dim_t g = 0; g < jcp.ngroups; ++g) {
                utils::array_copy(diff_bias_in + g * stride,
                        diff_bias + g * padded_stride, stride);
            }
        }
    });
}

} // namespace x64

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
