/*******************************************************************************
* Copyright 2021 Intel Corporation
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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/jit_brgemm_1x1_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using namespace utils;

    const auto src_type = src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;
    const bool is_int8 = one_of(src_type, u8, s8);
    const bool is_fp8 = one_of(src_type, f8_e4m3, f8_e5m2);

    VDISPATCH_CONV(
            impl::is_dense_format_kind({src_md(), weights_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    using skip_mask_t = primitive_attr_t::skip_mask_t;
    auto skip_mask = skip_mask_t::post_ops | skip_mask_t::sum_dt
            | skip_mask_t::zero_points | skip_mask_t::fpmath_mode;
    if (is_int8 || is_fp8) skip_mask |= skip_mask_t::scales;

    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(expect_data_types(src_type, wei_type, data_type::undef,
                           dst_type, data_type::undef),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(IMPLICATION(is_int8,
                           one_of(bias_md_.data_type, data_type::undef, f32,
                                   s32, s8, u8)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(IMPLICATION(!is_int8,
                           one_of(bias_md_.data_type, data_type::undef, f32,
                                   src_type)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(attr()->has_default_values(skip_mask, dst_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(attr()->post_ops_.check_sum_consistency(dst_type, is_int8),
            VERBOSE_UNSUPPORTED_POSTOP);
    CHECK(attr_scales_ok());
    CHECK(attr_zero_points_ok());

    CHECK(brgemm_convolution_utils::init_1x1_conf(jcp_, isa, *desc(), src_md_,
            weights_md_, dst_md_, bias_md_, attr_, dnnl_get_max_threads()));

    brgs_ = std::make_shared<brgemm_containers::brgemm_desc_container_t>(32);

    const bool need_compensation
            = (jcp_.src_zero_point || jcp_.s8s8_compensation_required)
            && !jcp_.req_brg_comp_pad;

    ic_chunks_ = div_up(jcp_.nb_ic, jcp_.nb_ic_blocking);
    need_postwork_ = jcp_.with_bias || jcp_.with_eltwise || jcp_.with_binary
            || jcp_.with_src_scales || jcp_.with_wei_scales
            || jcp_.with_dst_scales || need_compensation
            || (jcp_.dst_dt != jcp_.acc_dt) || jcp_.with_sum;

    const bool need_extra_m_kernel = get_extra_m_kernel_req(jcp_);
    const bool rtus_compute_partial_k = get_compute_partial_k_in_rtus(jcp_);
    const bool req_extra_accum_brgemm
            = ic_chunks_ > 1 || rtus_compute_partial_k;
    const int i_init_begin = req_extra_accum_brgemm ? 0 : 1;
    const int i_init_end = 2;
    for_(dim_t vM : {jcp_.M, jcp_.M_tail})
    for_(dim_t vN : {jcp_.N, jcp_.N_tail})
    for_(dim_t vK : {jcp_.K, jcp_.K_tail})
    for (int i_init = i_init_begin; i_init < i_init_end; i_init++) {
        if (vM == 0 || vN == 0 || vK == 0) continue;

        if (!jcp_.is_reduced_rtus) {
            brgemm_init_params_.emplace_front(
                    i_init, vM, vN, vK, jcp_.LDA, jcp_.extendable_k);
        } else {
            const bool is_accum_kernel = i_init == 0;

            // skip rtus block if it can be computed in M_tail brgemm
            const bool skip_rtus_M_blk = rtus_compute_partial_k
                    && jcp_.M_tail > 0 && vM == jcp_.M && is_accum_kernel;
            if (skip_rtus_M_blk) continue;

            const dim_t rtus_k = is_accum_kernel
                    ? jcp_.rtus_ic_size
                    : jcp_.ic_without_padding - jcp_.rtus_ic_size;
            const bool is_last_m_kernel = vM == jcp_.M_tail || jcp_.nb_os == 1;
            const bool use_rtus_K = rtus_compute_partial_k && is_last_m_kernel;
            const auto brgemm_K = use_rtus_K ? rtus_k : vK;
            const bool use_rtus_LDA = use_rtus_K && is_accum_kernel;
            const auto LDA = use_rtus_LDA ? jcp_.rtus_padded_ic_size : jcp_.LDA;
            brgemm_init_params_.emplace_front(
                    i_init, vM, vN, brgemm_K, LDA, false);
        }
    }

    if (need_extra_m_kernel) { // only used for 'reduced_rtus'
        assert(jcp_.K_tail == 0);
        const int rtus_K_kernels = 2;
        for_(dim_t vN : {jcp_.N, jcp_.N_tail})
        for (int idx = 0; idx < rtus_K_kernels; idx++) {
            if (vN == 0) continue;
            auto vM = jcp_.M;
            const bool is_accum_kernel = idx == 0;
            auto vK = is_accum_kernel
                    ? jcp_.rtus_ic_size
                    : jcp_.ic_without_padding - jcp_.rtus_ic_size;
            if (vM <= 0 || vK <= 0) continue;
            const bool use_rtus_LDA = is_accum_kernel;
            const auto LDA = use_rtus_LDA ? jcp_.rtus_padded_ic_size : jcp_.LDA;
            constexpr int extra_m_kernel_start_idx = 2;
            brgemm_init_params_.emplace_front(
                    extra_m_kernel_start_idx + idx, vM, vN, vK, LDA, false);
        }
    }

    CHECK(init_brgemm_desc());

    brgemm_convolution_utils::set_amx_wsp_per_thread(jcp_);
    auto scratchpad = scratchpad_registry().registrar();
    CHECK(brgemm_convolution_utils::init_scratchpad(
            scratchpad, jcp_, *src_md(), *weights_md(), *dst_md()));

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::pd_t::init_brgemm_desc() {

    const auto src_type = jcp_.src_dt;
    const auto wei_type = jcp_.wei_dt;
    const float alpha = 1.0;
    const float beta = 1.0;

    for (auto &params : brgemm_init_params_) {
        const auto vM = params.M_;
        const auto vN = params.N_;
        const auto vK = params.K_;
        const dim_t LDA = params.LDA_;

        const int k_accum_idx = params.k_accum_idx_;
        const bool req_k_accum = one_of(k_accum_idx, 0, 2);
        const auto vbeta = req_k_accum ? beta : 0;

        const auto brg_idx = get_brg_idx(jcp_, params);

        brgemm_desc_t brg;
        brg.fp8_with_f16_vnni_block
                = jcp_.is_fp8 && !jcp_.preconvert_fp8 && jcp_.vnni_block == 2;
        brgemm_strides_t brg_strides;
        brg_strides.stride_a = jcp_.brg_stride_a;
        brg_strides.stride_b = jcp_.brg_stride_b;
        const auto strides_ptr
                = (jcp_.brg_type == brgemm_strd) ? &brg_strides : nullptr;
        CHECK(brgemm_desc_init(&brg, isa, jcp_.brg_type, src_type, wei_type,
                false, false, brgemm_row_major, alpha, vbeta, LDA, jcp_.LDB,
                jcp_.LDC, vM, vN, vK, strides_ptr));

        brgemm_attr_t brgattr;
        brgattr.max_bs = jcp_.gemm_batch_size;
        brgattr.hint_innermost_loop = jcp_.brgemm_bd_loop_innermost
                ? brgemm_bd_loop_innermost
                : brgemm_innermost_undef;
        brgattr.max_top_vpad = jcp_.max_vpad;
        brgattr.max_bottom_vpad = jcp_.max_vpad;
        brgattr.hint_ununroll_bd_loop = jcp_.ununroll_bd_loop;

        // assuming 2x2 decomposition in amx brgemm kernel
        const auto bd_blocking = 2 * jcp_.amx_h;
        brgattr.hint_expected_A_size = bd_blocking * vK;
        brgattr.hint_expected_B_size = vN * vK;
        brgattr.hint_expected_C_size = bd_blocking * vN;

        brgattr.wary_A_k_tail_read = params.wary_tail_read_;
        brgattr.extendable_k = jcp_.extendable_k;
        brgattr.use_uker = jcp_.use_uker;
        brgattr.use_interleave_stores = jcp_.use_interleave_stores;
        brgattr.hint_prefetching = jcp_.hint_prefetching;
        brgattr.fpmath_mode = attr()->fpmath_.mode_;
        // if post-ops are required and there are no intermediate calculations
        // (like ic_chunks > 1) then we don't need code without post-ops in
        // brgemm kernel
        if (need_postwork_ && ic_chunks_ == 1 && (!jcp_.is_reduced_rtus))
            brgattr.postops_only = true;
        brgattr.use_ace = jcp_.is_ace;

        CHECK(brgemm_desc_set_attr(&brg, brgattr));
        auto LDD = jcp_.oc_without_padding;
        const auto &p = attr()->post_ops_;
        brg.with_sum = p.find(primitive_kind::sum) != -1;
        brg.with_weights_scale_adjust = jcp_.scale_adjust_factor != 1.0f;
        CHECK(brgemm_desc_set_postops(
                &brg, attr(), &dst_md_, LDD, jcp_.bia_dt));
        CHECK(brgemm_desc_finalize(&brg));

        jcp_.amx_buf_size_per_thread = nstl::max<dim_t>(
                brg.get_wsp_buffer_size(), jcp_.amx_buf_size_per_thread);
        brgs_->insert(brg_idx, brg);
    }
    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::init(engine_t *engine) {
    auto ndims = pd()->ndims();
    if (ndims < 3 || ndims > 5) assert(!"Invalid ndims!");

    const auto &jcp = pd()->jcp_;

    if (jcp.preconvert_fp8) {
        jit_brgemm_conv_fp8_cvt_kernel_t::cfg_t cfg;
        cfg.isa = isa;
        cfg.src_dt = pd()->src_md()->data_type;
        cfg.dst_dt = data_type::f16;
        CHECK(safe_ptr_assign(
                fp8_src_copy_, new jit_brgemm_conv_src_copy_kernel_t(cfg)));
        CHECK(fp8_src_copy_->create_kernel());
        cfg.src_dt = pd()->weights_md()->data_type;
        CHECK(safe_ptr_assign(
                fp8_weights_cvt_, new jit_brgemm_conv_fp8_cvt_kernel_t(cfg)));
        CHECK(fp8_weights_cvt_->create_kernel());
    }

    ID = ndims_pick(jcp.id, 1, 1);
    IH = ndims_pick(jcp.ih, jcp.ih, 1);
    IW = jcp.iw;

    OD = ndims_pick(jcp.od, 1, 1);
    OH = ndims_pick(jcp.oh, jcp.oh, 1);
    OW = jcp.ow;

    SD = ndims_pick(jcp.stride_d, 1, 1);
    SH = ndims_pick(jcp.stride_h, jcp.stride_h, 1);
    SW = jcp.stride_w;

    bia_dsz = jcp.bia_dsz;
    acc_dsz = jcp.acc_dsz;
    src_dsz = jcp.src_dsz;
    wei_dsz = jcp.wei_dsz;

    // const variables used for address calculations
    src_w_sz = (dim_t)IW * jcp.ngroups * jcp.ic_without_padding;
    src_h_sz = IH * src_w_sz;
    src_d_sz = ID * src_h_sz;
    dst_w_sz = (dim_t)OW * jcp.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;
    dst_d_sz = OD * dst_h_sz;

    const auto src_type = jcp.src_dt;
    const bool req_emulation = utils::one_of(isa, avx512_core_fp16, avx10_2);
    const data_type_t last_ic_block_dt
            = get_mac_emu_data_type(src_type, isa, req_emulation);
    const auto last_ic_block = data_type_vnni_granularity(last_ic_block_dt);

    wei_ic_stride = jcp.wei_plain ? jcp.oc_without_padding : jcp.oc_block;
    wei_ocb_stride = jcp.wei_plain
            ? jcp.oc_block
            : (dim_t)rnd_up(jcp.icp, last_ic_block) * jcp.oc_block;
    wei_g_stride = jcp.wei_plain ? jcp.oc : jcp.nb_oc * wei_ocb_stride;

    if (jcp.is_rtus) {
        CHECK(safe_ptr_assign(rtus_kernel_,
                new jit_avx512_core_brgemm_conv_trans_kernel::
                        jit_avx512_core_brgemm_conv_rtus_kernel_t(jcp)));
        CHECK(rtus_kernel_->create_kernel());
    }

    for (auto &params : pd()->brgemm_init_params_) {
        const auto brg_idx = get_brg_idx(jcp, params);
        const auto &brgs = *(pd()->brgs_);
        auto brg = brgs[brg_idx];
        if (brg != nullptr && brg->bcast_dim > 0 && brg->load_dim > 0
                && brg->reduce_dim > 0 && !brg_kernels_[brg_idx]) {
            CHECK(brg_kernels_.insert(brg_idx, brg));
            const bool is_amx = brgemm_convolution_utils::is_amx(isa);
            if (is_amx && brg->is_tmm) brgemm_palettes_.insert(brg_idx, brg);
        }
    }
    return status::success;
}

template <cpu_isa_t isa>
void brgemm_1x1_convolution_fwd_t<isa>::maybe_rtus(int ithr,
        const char *__restrict src, char *__restrict inp_buffer,
        uint8_t *__restrict inp_buffer_mask, int g, int n, dim_t icc, dim_t od,
        dim_t oh, dim_t ow) const {
    const auto &jcp = pd()->jcp_;
    if (!jcp.is_rtus) return;
    assert(jcp.is_os_blocking);
    const size_t src_dt_size = jcp.src_dsz;

    const auto os = (od * OH + oh) * OW + ow;
    const auto osb = os / jcp.os_block;
    const auto last_osb = jcp.nb_os - 1;
    const size_t rtus_ic_stride = jcp.rtus_padded_ic_size;

    if (jcp.is_reduced_rtus) {
        const bool exec_rtus = osb == last_osb;
        if (!exec_rtus) { return; }
    }
    const size_t bmask_offset = jcp.is_reduced_rtus ? 0 : icc * jcp.nb_os + osb;
    uint8_t *bmask = &inp_buffer_mask[bmask_offset];
    if (bmask && *bmask) return; // skip if already masked
    if (bmask) *bmask = 1; // set mask to skip next time

    const size_t icc_tail_start = jcp.is_reduced_rtus
            ? jcp.ic_without_padding - jcp.rtus_ic_size
            : icc * jcp.nb_ic_blocking * jcp.ic_block;
    const auto g_ic = g * jcp.ic_without_padding + icc_tail_start;

    const memory_desc_wrapper src_d(pd()->src_md());

    auto call_kernel = [&](dim_t nh, dim_t nw, dim_t od, dim_t oh, dim_t ow) {
        assert(nh == 0 || (nw == 0 && ow == 0));
        if (utils::everyone_is(0, nh, nw)) return;
        const dim_t id = od * jcp.stride_d;
        const dim_t ih = oh * jcp.stride_h;
        const dim_t iw = ow * jcp.stride_w;

        // Using blk_off to offset batch is motivated input\output striding aligment
        const auto inp_offset = src_d.off_l(0)
                + n * src_d.blk_off<false, true>(1) + id * src_h_sz
                + ih * src_w_sz + iw * jcp.ngroups * jcp.ic_without_padding
                + g_ic;
        auto p = jit_avx512_core_brgemm_conv_trans_kernel::
                jit_brgemm_conv_trans_kernel_args_t();
        p.h_count = nh;
        p.owb = nw;
        p.src = src + src_dt_size * inp_offset;
        p.dst = inp_buffer;
        (*rtus_kernel_)(&p);
        const size_t LDA = jcp.is_reduced_rtus ? rtus_ic_stride : jcp.LDA;
        inp_buffer += src_dt_size * (nh * jcp.ow + nw) * LDA;
    };

    const bool is_os_tail = jcp.os - os < jcp.os_block;
    dim_t count = is_os_tail ? jcp.M_tail : jcp.M;

    if (count < OW || ow > 0) {
        // copy to end of row
        const dim_t nw = nstl::min<dim_t>(count, OW - ow);
        call_kernel(0, nw, od, oh, ow);
        count -= nw;
        if (count == 0) return;
        ow = 0;
        oh = (oh + 1) % OH;
        if (oh == 0) od++;
    }

    while (od < OD) {
        // copy to end of column
        const auto nh = nstl::min(count / OW, OH - oh);
        if (nh > 0) {
            call_kernel(nh, 0, od, oh, ow);
            count -= nh * OW;
            if (count == 0) return;
            oh = (oh + nh) % OH;
            if (oh == 0) od++;
        }
        if (count < OW) {
            // copy partial row
            const auto nw = count;
            call_kernel(0, nw, od, oh, ow);
            return;
        }
    }
}

template <cpu_isa_t isa>
void brgemm_1x1_convolution_fwd_t<isa>::exec_ker(
        const brgemm_exec_ctx_t &brgemm_ctx, int ithr,
        brgemm_batch_element_t *const __restrict brg_batch,
        char *const c_buffer, const char *inp_buffer, int g, int n, int ocb,
        dim_t od, dim_t oh, dim_t ow, dim_t icc, int *last_brg_idx,
        const int32_t *src_zero_points, int32_t *src_zp_comp,
        const int32_t *dst_zero_points, int32_t *s8s8_compensation,
        const void *src_scales, const void *wei_scales, const void *dst_scales,
        char *fp8_convert_wsp, const bool is_last_os) const {

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const size_t src_dt_size = src_dsz;
    const size_t wei_dt_size = wei_dsz;
    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());

    const char *const __restrict src = brgemm_ctx.src;
    const char *const __restrict weights = brgemm_ctx.weights;
    const char *const __restrict bias = brgemm_ctx.bias;
    char *const __restrict dst = brgemm_ctx.dst;
    const std::vector<const void *> &post_ops_binary_rhs_arg_vec
            = brgemm_ctx.post_ops_binary_rhs_arg_vec;

    const auto &jcp = pd()->jcp_;
    auto ndims = pd()->ndims();

    const bool is_amx = brgemm_convolution_utils::is_amx(isa);
    char *const wsp_tile = is_amx
            ? brgemm_ctx.wsp_tile + ithr * jcp.amx_buf_size_per_thread
            : nullptr;

    const dim_t id = ndims_pick(od * SD, 0, 0);
    const dim_t ih = ndims_pick(oh * SH, oh * SH, 0);
    const dim_t iw = ow * SW;

    const dim_t oc = ocb * jcp.oc_block;
    const dim_t g_oc = g * jcp.oc + oc;

    const dim_t icb = icc * jcp.nb_ic_blocking;
    const dim_t ic = icb * jcp.ic_block;

    const bool use_special_m_idx = get_extra_m_kernel_req(jcp) && is_last_os;
    const int kernel_init = static_cast<int>(icc == 0) + 2 * use_special_m_idx;

    const auto os = (od * OH + oh) * OW + ow;

    const bool is_os_tail = jcp.is_os_blocking ? (jcp.os - os < jcp.os_block)
                                               : (OW - ow < jcp.ow_block);
    const bool is_oc_tail = (jcp.oc - oc < jcp.oc_block);
    const bool is_ic_tail = jcp.is_reduced_rtus
            ? is_last_os
            : (icc == pd()->ic_chunks_ - 1
                      && ((jcp.ic - ic) % jcp.ic_block != 0));

    // Using blk_off to offset batch is motivated input\output striding aligment
    // See `blk_off` definition.
    const auto src_mb_c_offset = src_dt_size
            * (src_d.off_l(0) + n * src_d.blk_off<false, true>(1)
                    + g * src_d.blk_off<false, true>(0, 1) * jcp.ic + ic);
    const auto src_hw_offset = src_dt_size
            * (id * src_h_sz + ih * src_w_sz
                    + iw * jcp.ngroups * jcp.ic_without_padding);

    const auto rtus_src = jcp.is_reduced_rtus
            ? src + src_mb_c_offset + src_hw_offset
            : inp_buffer;
    const auto src_base
            = jcp.is_rtus ? rtus_src : src + src_mb_c_offset + src_hw_offset;

    const auto wei_offset = g * wei_g_stride + ocb * wei_ocb_stride;
    const auto wei_base = weights + wei_dt_size * wei_offset;

    // Using blk_off to offset batch is motivated input\output striding aligment
    // See `blk_off` definition.
    const auto dst_base = dst_dt_size
            * (dst_d.off_l(0) + n * dst_d.blk_off<false, true>(1)
                    + g * dst_d.blk_off<false, true>(0, 1) * jcp.oc + oc);
    const auto dst_offset = dst_dt_size
            * (od * dst_h_sz + oh * dst_w_sz + ow * jcp.oc_without_padding);

    const auto ptr_D = dst + dst_base + dst_offset;
    char *const ptr_C = (jcp.use_buffer) ? c_buffer : (char *)ptr_D;

    const auto bias_w
            = bias ? bias + (bias_d.blk_off(g_oc) * bia_dsz) : nullptr;
    const dim_t nb_ic_b = nstl::min<dim_t>(jcp.nb_ic_blocking, jcp.nb_ic - icb)
            - (is_ic_tail ? 1 : 0);

    const auto comp_offset = (g * jcp.nb_oc + ocb) * jcp.oc_block;
    int32_t *src_zp_comp_ptr
            = (jcp.src_zero_point && icc == pd()->ic_chunks_ - 1)
            ? &src_zp_comp[comp_offset]
            : nullptr;
    int32_t *s8s8_comp_ptr
            = (jcp.s8s8_compensation_required && icc == pd()->ic_chunks_ - 1)
            ? &s8s8_compensation[comp_offset]
            : nullptr;

    const bool wary_tail_read = jcp.extendable_k;

    const auto call_brgemm
            = [&](int brg_idx, dim_t ic_block_s, dim_t n_ic_blocks,
                      bool do_postops, bool brgemm_is_ic_tail) {
        // NOTE: avoid some costly tile reconfigurations here by keeping track
        //       of the previous brg kernel tile configuration palette
        // TODO: adjust harness to avoid even more tile reconfigurations
        brgemm_palettes_.maybe_tile_configure(is_amx, *last_brg_idx, brg_idx);

        for (dim_t k = 0; k < n_ic_blocks; k++) {
            const dim_t ic_off = jcp.is_reduced_rtus
                    ? (brgemm_is_ic_tail ? jcp.ic_without_padding
                                              - jcp.rtus_ic_size
                                         : 0)
                    : (ic_block_s + k) * jcp.ic_block;
            const size_t src_ic = ic_off;
            const auto wei_ic = ic + ic_off;
            const auto ptr_A
                    = brgemm_is_ic_tail && is_last_os && jcp.is_reduced_rtus
                    ? inp_buffer
                    : src_base + src_dt_size * src_ic;
            const auto ptr_B = wei_base + wei_dt_size * wei_ic * wei_ic_stride;
            brg_batch[k].ptr.A = ptr_A;
            brg_batch[k].ptr.B = ptr_B;
            brg_batch[k].vvpad.top = 0;
            brg_batch[k].vvpad.bottom = 0;
        }

        const auto brg_ker = brg_kernels_[brg_idx];
        if (do_postops) {
            const int32_t src_zp_val = src_zero_points ? src_zero_points[0] : 0;
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(bias_w),
                    post_ops_binary_rhs_arg_vec.data(),
                    static_cast<size_t>(g_oc), 0, dst, 0,
                    static_cast<void *>(src_zp_comp_ptr), nullptr,
                    dst_zero_points, false, src_zp_val, false, false,
                    src_scales,
                    wei_scales ? static_cast<const char *>(wei_scales)
                                    + jcp.is_oc_scale * g_oc * sizeof(float)
                               : nullptr,
                    dst_scales};

            void *scratch = is_amx ? static_cast<void *>(wsp_tile)
                    : jcp.req_fp8_convert_wsp
                    ? static_cast<void *>(fp8_convert_wsp)
                    : static_cast<void *>(s8s8_comp_ptr);
            brgemm_kernel_execute_postops(brg_ker, n_ic_blocks, brg_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch);
        } else {
            void *scratch = is_amx ? static_cast<void *>(wsp_tile)
                    : jcp.req_fp8_convert_wsp
                    ? static_cast<void *>(fp8_convert_wsp)
                    : static_cast<void *>(s8s8_comp_ptr);
            brgemm_kernel_execute(
                    brg_ker, n_ic_blocks, brg_batch, (void *)ptr_C, scratch);
        }
    };

    const auto do_post_work = (pd()->need_postwork_ || jcp.use_buffer)
            && icc == pd()->ic_chunks_ - 1;
    if (jcp.is_reduced_rtus) {
        const auto brg_idx = get_brg_idx(
                kernel_init, is_os_tail, is_oc_tail, false, wary_tail_read);
        call_brgemm(brg_idx, 0, 1, do_post_work && !is_ic_tail, false);
    } else if (nb_ic_b > 0) {
        const auto brg_idx = get_brg_idx(
                kernel_init, is_os_tail, is_oc_tail, false, wary_tail_read);
        call_brgemm(brg_idx, 0, nb_ic_b, do_post_work && !is_ic_tail, false);
    }
    if (is_ic_tail) {
        const auto use_init_ker = jcp.is_reduced_rtus
                ? kernel_init - 1
                : kernel_init && (nb_ic_b == 0);
        const auto brg_idx = get_brg_idx(use_init_ker, is_os_tail, is_oc_tail,
                !jcp.is_reduced_rtus, wary_tail_read);
        call_brgemm(brg_idx, jcp.is_reduced_rtus ? 0 : nb_ic_b, 1, do_post_work,
                jcp.is_reduced_rtus);
    }
}

template <cpu_isa_t isa>
void brgemm_1x1_convolution_fwd_t<isa>::execute_os_blocking(
        const brgemm_exec_ctx_t &brgemm_ctx,
        brgemm_batch_element_t *const brg_batch_global, const void *src_scales,
        const void *wei_scales, const void *dst_scales, void *dst_scales_inv,
        const int32_t *src_zero_points, int32_t *src_zp_comp,
        const int32_t *dst_zero_points, int32_t *s8s8_compensation,
        char *const c_buffer_global, char *inp_buffer_base,
        uint8_t *inp_buffer_mask_base, char *fp8_wsp_base) const {

    const auto &jcp = pd()->jcp_;
    const bool is_amx = brgemm_convolution_utils::is_amx(isa);

    const dim_t os_chunks = div_up(jcp.nb_os, jcp.nb_os_blocking);
    const int work_amount
            = static_cast<int>(jcp.mb * jcp.ngroups * jcp.nb_oc * os_chunks);

    parallel(pd()->jcp_.nthr,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;
        brgemm_batch_element_t *const brg_batch
                = brg_batch_global + (size_t)ithr * jcp.adjusted_batch_size;
        char *const c_buffer = (jcp.use_buffer)
                ? c_buffer_global + ithr * acc_dsz * jcp.LDC * jcp.M
                : nullptr;
        char *inp_buffer = (jcp.is_rtus)
                ? inp_buffer_base + ithr * src_dsz * jcp.inp_buffer_size
                : nullptr;
        uint8_t *__restrict inp_buffer_mask = (jcp.is_rtus)
                ? inp_buffer_mask_base + ithr * jcp.inp_buffer_mask_size
                : nullptr;
        char *fp8_convert_wsp = jcp.req_fp8_convert_wsp
                ? fp8_wsp_base + ithr * jcp.fp8_convert_wsp_size
                : nullptr;

        float *dst_scales_inv_ptr = nullptr;
        if (jcp.with_dst_scales) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr = static_cast<float *>(dst_scales_inv) + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        int last_n = -1;
        int last_g = -1;
        int last_brg_idx = -1;
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, g {0}, ocb {0}, oss {0};

        if (jcp.loop_order == loop_ndhwgc)
            nd_iterator_init(start, n, jcp.mb, oss, os_chunks, g, jcp.ngroups,
                    ocb, jcp.nb_oc);
        else if (jcp.loop_order == loop_ngcdhw)
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc,
                    oss, os_chunks);
        else
            assert(!"Unknown loop order");

        for (auto work = start; work < end; work++) {
            if (jcp.is_rtus && (last_n != n || last_g != g))
                std::memset(inp_buffer_mask, 0, jcp.inp_buffer_mask_size);
            const auto osb_start = oss * jcp.nb_os_blocking;
            const dim_t osb_range = nstl::min<dim_t>(
                    jcp.nb_os - osb_start, jcp.nb_os_blocking);
            for (int osb = 0; osb < osb_range; osb++) {
                const dim_t os = (osb_start + osb) * jcp.os_block;
                const dim_t od = os / (OH * OW);
                const dim_t oh = (os % (OH * OW)) / OW;
                const dim_t ow = os % OW;
                const size_t rtus_offset
                        = jcp.is_reduced_rtus ? 0 : src_dsz * os * jcp.LDA;
                char *inp_buffer_sp
                        = jcp.is_rtus ? inp_buffer + rtus_offset : nullptr;
                for (int icc = 0; icc < pd()->ic_chunks_; icc++) {
                    if (jcp.is_rtus)
                        maybe_rtus(ithr, brgemm_ctx.src, inp_buffer_sp,
                                inp_buffer_mask, g, n, icc, od, oh, ow);
                    const bool is_last_os = (osb_start + osb) == jcp.nb_os - 1;
                    exec_ker(brgemm_ctx, ithr, brg_batch, c_buffer,
                            inp_buffer_sp, g, n, ocb, od, oh, ow, icc,
                            &last_brg_idx, src_zero_points, src_zp_comp,
                            dst_zero_points, s8s8_compensation, src_scales,
                            wei_scales, dst_scales_inv_ptr, fp8_convert_wsp,
                            is_last_os);
                }
            }
            last_n = n;
            last_g = g;
            if (jcp.loop_order == loop_ndhwgc)
                nd_iterator_step(n, jcp.mb, oss, os_chunks, g, jcp.ngroups, ocb,
                        jcp.nb_oc);
            else if (jcp.loop_order == loop_ngcdhw)
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, oss,
                        os_chunks);
            else
                assert(!"Unknown loop order");
        }
        if (is_amx) amx_tile_release();
    });
}

template <cpu_isa_t isa>
void brgemm_1x1_convolution_fwd_t<isa>::execute_full_spatial(
        const brgemm_exec_ctx_t &brgemm_ctx,
        brgemm_batch_element_t *const brg_batch_global, const void *src_scales,
        const void *wei_scales, const void *dst_scales, void *dst_scales_inv,
        const int32_t *src_zero_points, int32_t *src_zp_comp,
        const int32_t *dst_zero_points, int32_t *s8s8_compensation,
        char *const c_buffer_global, char *fp8_wsp_base) const {

    const auto &jcp = pd()->jcp_;
    const bool is_amx = brgemm_convolution_utils::is_amx(isa);
    const int work_amount = static_cast<int>(
            jcp.mb * jcp.ngroups * jcp.nb_oc * OD * OH * jcp.nb_ow);
    parallel(pd()->jcp_.nthr,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;
        brgemm_batch_element_t *const brg_batch
                = brg_batch_global + (size_t)ithr * jcp.adjusted_batch_size;
        char *const c_buffer = (jcp.use_buffer)
                ? c_buffer_global + ithr * acc_dsz * jcp.LDC * jcp.M
                : nullptr;
        char *fp8_convert_wsp = jcp.req_fp8_convert_wsp
                ? fp8_wsp_base + ithr * jcp.fp8_convert_wsp_size
                : nullptr;

        float *dst_scales_inv_ptr = nullptr;
        if (jcp.with_dst_scales) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr = static_cast<float *>(dst_scales_inv) + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        int last_brg_idx = -1;
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, g {0}, ocb {0}, od {0}, oh {0}, owb {0};

        if (jcp.loop_order == loop_ndhwgc)
            nd_iterator_init(start, n, jcp.mb, od, OD, oh, OH, owb, jcp.nb_ow,
                    g, jcp.ngroups, ocb, jcp.nb_oc);
        else if (jcp.loop_order == loop_ngcdhw)
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc,
                    od, OD, oh, OH, owb, jcp.nb_ow);
        else
            assert(!"Unknown loop order");

        for (auto work = start; work < end; work++) {
            for (dim_t icc = 0; icc < pd()->ic_chunks_; icc++) {
                const dim_t ow = owb * jcp.ow_block;
                exec_ker(brgemm_ctx, ithr, brg_batch, c_buffer, nullptr, g, n,
                        ocb, od, oh, ow, icc, &last_brg_idx, src_zero_points,
                        src_zp_comp, dst_zero_points, s8s8_compensation,
                        src_scales, wei_scales, dst_scales_inv_ptr,
                        fp8_convert_wsp);
            }
            if (jcp.loop_order == loop_ndhwgc)
                nd_iterator_step(n, jcp.mb, od, OD, oh, OH, owb, jcp.nb_ow, g,
                        jcp.ngroups, ocb, jcp.nb_oc);
            else if (jcp.loop_order == loop_ngcdhw)
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, od,
                        OD, oh, OH, owb, jcp.nb_ow);
            else
                assert(!"Unknown loop order");
        }
        if (is_amx) amx_tile_release();
    });
}

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::execute_forward_all(
        const exec_ctx_t &ctx) const {

    brgemm_exec_ctx_t brgemm_ctx(ctx, pd());

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = pd()->jcp_;
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    if (jcp.preconvert_fp8) {
        auto *src_f16 = scratchpad.template get<char>(key_conv_fp8_src);
        auto *wei_f16 = scratchpad.template get<char>(key_conv_fp8_weights);
        const auto src_nelems
                = memory_desc_wrapper(pd()->src_md()).nelems(true);
        const auto wei_nelems = weights_d.nelems(true);
        parallel(jcp.nthr, [&](int ithr, int nthr) {
            dim_t start = 0, end = 0;
            balance211(src_nelems, nthr, ithr, start, end);
            jit_brgemm_conv_src_copy_kernel_t::ctx_t p;
            p.src = brgemm_ctx.src + start;
            p.dst = src_f16 + start * jcp.src_dsz;
            p.nelems = end - start;
            (*fp8_src_copy_)(&p);
            balance211(wei_nelems, nthr, ithr, start, end);
            p.src = brgemm_ctx.weights + start;
            p.dst = wei_f16 + start * jcp.wei_dsz;
            p.nelems = end - start;
            (*fp8_weights_cvt_)(&p);
        });
        brgemm_ctx.src = src_f16;
        brgemm_ctx.weights = wei_f16;
    }

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const auto extra_data_offset
            = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<char *>(brgemm_ctx.weights);
    int32_t *s8s8_compensation = (jcp.s8s8_compensation_required)
            ? reinterpret_cast<int32_t *>(w + extra_data_offset)
            : nullptr;
    int32_t *zp_compensation = (jcp.src_zero_point)
            ? reinterpret_cast<int32_t *>(&w[extra_data_offset])
                    + (jcp.s8s8_compensation_required
                                    ? jcp.s8s8_comp_buffer_size
                                    : 0)
            : nullptr;

    brgemm_batch_element_t *const brg_batch_global
            = (jcp.brg_type != brgemm_strd)
            ? scratchpad.template get<brgemm_batch_element_t>(
                      key_brgemm_primitive_batch)
            : nullptr;
    char *const c_buffer_global = (jcp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;
    char *inp_buffer_base = (jcp.is_rtus)
            ? scratchpad.template get<char>(key_conv_brgemm_inp_buffer)
            : nullptr;
    uint8_t *inp_buffer_mask_base = (jcp.is_rtus)
            ? scratchpad.template get<uint8_t>(key_conv_brgemm_inp_buffer_mask)
            : nullptr;
    void *dst_scales_inv = jcp.with_dst_scales
            ? scratchpad.template get<void>(key_conv_dst_scales)
            : nullptr;
    char *fp8_convert_wsp_base = jcp.req_fp8_convert_wsp
            ? scratchpad.template get<char>(
                      key_brgemm_primitive_fp8_convert_wsp)
            : nullptr;

    if (jcp.is_os_blocking) {
        execute_os_blocking(brgemm_ctx, brg_batch_global, src_scales,
                wei_scales, dst_scales, dst_scales_inv, src_zero_points,
                zp_compensation, dst_zero_points, s8s8_compensation,
                c_buffer_global, inp_buffer_base, inp_buffer_mask_base,
                fp8_convert_wsp_base);
    } else {
        execute_full_spatial(brgemm_ctx, brg_batch_global, src_scales,
                wei_scales, dst_scales, dst_scales_inv, src_zero_points,
                zp_compensation, dst_zero_points, s8s8_compensation,
                c_buffer_global, fp8_convert_wsp_base);
    }

    return status::success;
}

template struct brgemm_1x1_convolution_fwd_t<avx2>;
template struct brgemm_1x1_convolution_fwd_t<avx2_vnni>;
template struct brgemm_1x1_convolution_fwd_t<avx2_vnni_2>;
template struct brgemm_1x1_convolution_fwd_t<avx512_core>;
template struct brgemm_1x1_convolution_fwd_t<avx512_core_vnni>;
template struct brgemm_1x1_convolution_fwd_t<avx512_core_bf16>;
template struct brgemm_1x1_convolution_fwd_t<avx512_core_fp16>;
template struct brgemm_1x1_convolution_fwd_t<avx512_core_amx>;
template struct brgemm_1x1_convolution_fwd_t<avx512_core_amx_fp16>;
template struct brgemm_1x1_convolution_fwd_t<avx10_2>;
template struct brgemm_1x1_convolution_fwd_t<avx10_2_amx_2>;
template struct brgemm_1x1_convolution_fwd_t<avx10_2_ace>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
