/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2024-2026 FUJITSU LIMITED
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

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_brdgmm_dw_conv.hpp"
#include <cpu/aarch64/cpu_isa_traits.hpp>

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

namespace {
struct blk_info_t {
    int n_lpad_blks;
    int rpad_blk_start_idx;
};
blk_info_t get_blocks_info(int sp_i, int sp_o, int k, int stride, int lpad,
        int rpad, int blk_size) {

    const int max_blks = div_up(sp_o, blk_size);
    const int blk_shift = stride * blk_size;
    const int n_lpad_blks = nstl::min(
            max_blks, div_up(lpad, blk_shift) + 1 /*include zero lpad*/);
    const int rpad_blk_start_idx = saturate(
            n_lpad_blks, max_blks, (sp_i + lpad - k + 1) / blk_shift);
    return {n_lpad_blks, rpad_blk_start_idx};
}
} // namespace

inline status_t init_tag(memory_desc_t &md, const memory_desc_wrapper &mdw,
        const format_tag_t tag_value, bool any_eligible) {

    format_tag_t tag;
    if (mdw.format_kind() == format_kind::any) {
        if (any_eligible) {
            CHECK(memory_desc_init_by_tag(md, tag_value));
            tag = tag_value;
        } else {
            tag = format_tag::undef;
        }
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }
    if (tag != tag_value) { return status::unimplemented; }

    return status::success;
}

bool post_ops_ok(jit_brdgmm_conv_conf_t &jcp, const primitive_attr_t &attr,
        const memory_desc_wrapper &dst_d) {
    using namespace injector;

    const auto &post_ops = attr.post_ops_;

    return injector::post_ops_ok(post_ops_ok_args_t(get_max_cpu_isa(),
            {sum, eltwise, binary}, post_ops, &dst_d,
            false /*sum_at_pos_0_only*/, false /*sum_requires_scale_one*/,
            false /*sum_requires_zp_zero*/, true /*sum_requires_same_params*/,
            {broadcasting_strategy_t::per_oc, broadcasting_strategy_t::scalar,
                    broadcasting_strategy_t::no_broadcast}));
}

template <cpu_isa_t isa>
status_t brdgmm_dw_convolution_fwd_t<isa>::pd_t::init(const engine_t *engine) {
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    const auto &cd = *desc();
    const auto src_type = cd.src_desc.data_type;
    const auto wei_type = cd.weights_desc.data_type;
    const auto bia_type = cd.bias_desc.data_type;
    const auto dst_type = cd.dst_desc.data_type;

    const bool is_f32 = everyone_is(f32, src_type, wei_type, dst_type);
    const bool is_bf16 = everyone_is(bf16, src_type, wei_type)
            && one_of(dst_type, bf16, f32);
    const bool is_f32_bf16
            = everyone_is(f32, src_type, dst_type) && wei_type == bf16;
    const bool is_int8 = one_of(src_type, s8, u8) && wei_type == s8
            && one_of(dst_type, s32, f32, u8, s8);

    auto skip_mask = skip_mask_t::post_ops;
    if (is_int8) skip_mask |= skip_mask_t::scales;

    VDISPATCH_CONV(DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL,
            VERBOSE_UNSUPPORTED_THREADPOOL_RUNTIME);
    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(one_of(true, is_f32, is_int8, is_bf16, is_f32_bf16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(
            IMPLICATION(is_int8,
                    one_of(bia_type, data_type::undef, f32, s32, s8, u8)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(
            IMPLICATION(is_bf16, one_of(bia_type, data_type::undef, bf16, f32)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(
            IMPLICATION(!is_int8 && !is_bf16,
                    one_of(bia_type, data_type::undef, src_type, dst_type)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(
            attr()->has_default_values(skip_mask), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(
            (isa != isa_undef) && mayiuse(isa), "undefined or unsupported isa");
    VDISPATCH_CONV(
            impl::is_dense_format_kind({src_md(), weights_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    auto &jcp = jcp_;

    const memory_desc_wrapper src_d(&src_md_);
    const memory_desc_wrapper weights_d(&weights_md_);
    const memory_desc_wrapper dst_d(&dst_md_);
    const memory_desc_wrapper bias_d(&bias_md_);

    const int ndims = src_d.ndims();
    const bool is_3d = ndims == 5;
    // Currently this kernel only supports 2D and 3D convolutions.
    VDISPATCH_CONV(utils::one_of(ndims, 4, 5), VERBOSE_UNSUPPORTED_FEATURE,
            "only 2D/3D convolutions are supported");
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    VDISPATCH_CONV(with_groups, VERBOSE_UNSUPPORTED_FEATURE,
            "Grouped convolution expected for depthwise convolution "
            "implementation");
    // dilations are not supported
    VDISPATCH_CONV(!(cd.dilates[0] != 0 || cd.dilates[1] != 0
                           || (is_3d && cd.dilates[2] != 0)),
            VERBOSE_UNSUPPORTED_FEATURE, "dilations are not supported");

    jcp = utils::zero<decltype(jcp)>();
    jcp.ngroups = weights_d.dims()[0];
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;

    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = is_3d ? weights_d.dims()[3] : 1;
    jcp.kh = weights_d.dims()[ndims - 1];
    jcp.kw = weights_d.dims()[ndims];
    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = cd.padding[0][is_3d];
    jcp.l_pad = cd.padding[0][is_3d + 1];
    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = cd.strides[is_3d];
    jcp.stride_w = cd.strides[is_3d + 1];
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, jcp.kd);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, jcp.kh);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, jcp.kw);
    jcp.src_dt = cd.src_desc.data_type;
    jcp.dst_dt = cd.dst_desc.data_type;
    jcp.wei_dt = cd.weights_desc.data_type;
    jcp.with_bias = with_bias();
    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;

    VDISPATCH_CONV((everyone_is(1, jcp.ic, jcp.oc)),
            "Depthwise BRGEMM implementation supports only 1 input and 1 "
            "output channel per group");

    const auto def_data_tag = is_3d ? format_tag::ndhwc : format_tag::nhwc;
    CHECK(init_tag(src_md_, src_d, def_data_tag, true));
    CHECK(init_tag(dst_md_, dst_d, def_data_tag, true));

    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md_, format_tag::x));
    }

    CHECK(attr_.set_default_formats(dst_md()));
    VDISPATCH_CONV(
            post_ops_ok(jcp, *attr(), dst_d), VERBOSE_UNSUPPORTED_POSTOP);
    jcp.with_post_ops = attr()->post_ops_.len() > 0;

    jcp.isa = isa;
    jcp.nthr = dnnl_get_max_threads();
    jcp.src_dsz = types::data_type_size(jcp.src_dt);
    jcp.wei_dsz = types::data_type_size(jcp.wei_dt);
    jcp.bia_dsz
            = jcp.with_bias ? types::data_type_size(cd.bias_desc.data_type) : 0;
    jcp.dst_dsz = types::data_type_size(jcp.dst_dt);

    const auto &src_scales = attr_.scales_.get(DNNL_ARG_SRC);
    const auto &wei_scales = attr_.scales_.get(DNNL_ARG_WEIGHTS);
    jcp.with_scale = !src_scales.has_default_values()
            || !wei_scales.has_default_values();
    jcp.is_oc_scale = wei_scales.get_mask() > 0;

    CHECK(attr_scales_ok());

    // strd is only feasible for 1D (i.e., height dim is one)
    // and if there are no tails (for calculating matrix_B strides).
    // Since, we cannot always predict the blocking is 8 or 16.
    if (jcp.kd == 1 && jcp.kh == 1 && jcp.ngroups % 8 == 0) {
        jcp.batch_kind = brgemm_strd;
    } else {
        jcp.batch_kind = brgemm_offs;
    }

    // to avoid cache concurrent access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.kd * jcp.kh * jcp.kw * sc_size, 4096), sc_size);
    CHECK(init_brdgmm_conf());
    // CHECK(init_scratchpad());
    if (jcp.with_scale) {
        auto scratchpad = scratchpad_registry().registrar();
        book_precomputed_scales(scratchpad, attr_.scales_, OC());
    }
    init_batch_elements();
    return status::success;
}

template <cpu_isa_t isa>
void brdgmm_dw_convolution_fwd_t<isa>::pd_t::init_batch_elements() {

    auto &jcp = jcp_;

    auto gen_batch_elements
            = [&jcp](int fpad, int backpad, int tpad, int bpad, int lpad,
                      int rpad, int &bs, brgemm_batch_element_t *batches) {
        const size_t src_w_stride = jcp.ngroups * jcp.src_dsz;
        const size_t src_h_stride = jcp.ngroups * jcp.iw * jcp.src_dsz;
        const size_t src_d_stride = jcp.ngroups * jcp.ih * jcp.iw * jcp.src_dsz;

        const size_t wei_w_stride
                = rnd_up(jcp.ngroups, jcp.ch_block) * jcp.wei_dsz;
        const size_t wei_h_stride = wei_w_stride * jcp.kw;
        const size_t wei_d_stride = wei_h_stride * jcp.kh;

        const int adj_backpad = nstl::max(0, backpad);
        const int adj_bpad = nstl::max(0, bpad);

        for_(int kd = 0; kd < jcp.kd; ++kd)
        for_(int kh = 0; kh < jcp.kh; ++kh)
        for (int kw = 0; kw < jcp.kw; ++kw) {
            const bool padded_bs = kd < fpad || kd >= jcp.kd - adj_backpad
                    || kh < tpad || kh >= jcp.kh - adj_bpad;
            if (padded_bs) continue;
            auto &batch = batches[bs];
            batch.vvpad.top = div_up(nstl::max(0, lpad - kw), jcp.stride_w);
            batch.vvpad.bottom = div_up(
                    nstl::max(0, rpad - jcp.kw + kw + 1), jcp.stride_w);
            //     batch.has_s8s8_comp_batch_pad = padded_bs;
            const dim_t offs_A
                    = kd * src_d_stride + kh * src_h_stride + kw * src_w_stride;
            const dim_t offs_B
                    = kd * wei_d_stride + kh * wei_h_stride + kw * wei_w_stride;
            if (jcp.batch_kind == brgemm_offs) {
                batch.offset.A = offs_A;
                batch.offset.B = offs_B;
            }
            ++bs;
        }
    };

    const int w_shift = jcp.ow_block * jcp.stride_w;
    const int h_shift = jcp.stride_h;
    const int d_shift = jcp.stride_d;

    const auto w_blk_info = get_blocks_info(jcp.iw, jcp.ow, jcp.kw,
            jcp.stride_w, jcp.l_pad, jcp.r_pad, jcp.ow_block);
    const int rpad_0
            = (jcp.ow_block - 1) * jcp.stride_w + jcp.kw - (jcp.iw + jcp.l_pad);
    const int rpad_1 = rpad_0 + (nstl::max(0, -rpad_0) / w_shift + 1) * w_shift;
    const int n_uniq_rpads
            = 1 + div_up(nstl::max(0, jcp.r_pad - (rpad_1 - w_shift)), w_shift);

    const auto h_blk_info = get_blocks_info(
            jcp.ih, jcp.oh, jcp.kh, jcp.stride_h, jcp.t_pad, jcp.b_pad, 1);

    const auto d_blk_info = get_blocks_info(
            jcp.id, jcp.od, jcp.kd, jcp.stride_d, jcp.f_pad, jcp.back_pad, 1);

    const int max_bs = jcp.kd * jcp.kh * jcp.kw;
    const int n_d_uniq_blks
            = d_blk_info.n_lpad_blks + (jcp.od - d_blk_info.rpad_blk_start_idx);
    const int n_h_uniq_blks
            = h_blk_info.n_lpad_blks + (jcp.oh - h_blk_info.rpad_blk_start_idx);
    const int n_w_uniq_lpads = w_blk_info.n_lpad_blks;
    const int uniq_blks
            = n_d_uniq_blks * n_h_uniq_blks * n_w_uniq_lpads * n_uniq_rpads;

    bs_.resize(uniq_blks, 0);
    batches_.resize(bs_.size() * max_bs);
    int bi = 0;

    for_(int odb = 0; odb < n_d_uniq_blks; ++odb)
    for_(int ohb = 0; ohb < n_h_uniq_blks; ++ohb)
    for_(int owb = 0; owb < n_w_uniq_lpads; ++owb)
    for (int rpad_i = 0; rpad_i < n_uniq_rpads; ++rpad_i) {
        const int lpad = jcp.l_pad - owb * w_shift;
        const int rpad = rpad_i == 0
                ? rpad_0
                : nstl::min(jcp.r_pad, rpad_1 + (rpad_i - 1) * w_shift);

        const int tpad = jcp.t_pad - ohb * h_shift;
        const int oh = ohb < h_blk_info.n_lpad_blks
                ? ohb
                : (h_blk_info.rpad_blk_start_idx
                          + (ohb - h_blk_info.n_lpad_blks));
        const int bpad = oh * h_shift + jcp.kh - (jcp.ih + jcp.t_pad);

        const int fpad = jcp.f_pad - odb * d_shift;
        const int od = odb < d_blk_info.n_lpad_blks
                ? odb
                : (d_blk_info.rpad_blk_start_idx
                          + (odb - d_blk_info.n_lpad_blks));
        const int backpad = od * d_shift + jcp.kd - (jcp.id + jcp.f_pad);

        gen_batch_elements(fpad, backpad, tpad, bpad, lpad, rpad, bs_[bi],
                &batches_[bi * max_bs]);
        ++bi;
        assert(static_cast<int>(bs_.size()) >= bi);
    }
    assert(static_cast<int>(bs_.size()) == bi);
}

template <cpu_isa_t isa>
status_t brdgmm_dw_convolution_fwd_t<isa>::pd_t::init_brdgmm_conf() {

    auto &jcp = jcp_;
    const bool is_3d = ndims() == 5;

    auto init_bcp = [&](int &idx, const int M, const int N) {
        const float alpha = 1.f;
        const float beta = 0.f;
        const int LDA = jcp.ngroups * jcp.stride_w;
        const int LDC = jcp.ngroups;
        const int LDD = jcp.ngroups;

        brgemm_attr_t brg_attr;
        brg_attr.max_bs = jcp.kw * jcp.kh * jcp.kd;
        brg_attr.max_top_vpad = nstl::max(0, jcp.l_pad);
        brg_attr.max_bottom_vpad = nstl::max(0, jcp.r_pad);

        // only needed for strd batch_kind
        const brgemm_strides_t strides
                = {static_cast<dim_t>(jcp.src_dsz) * jcp.ngroups,
                        static_cast<dim_t>(jcp.wei_dsz) * jcp.ngroups};

        auto &bcp = bcps_[idx];
        CHECK(brdgmm_desc_init(&bcp, jcp.isa, jcp.batch_kind, jcp.src_dt,
                jcp.wei_dt, false /*transA*/, brgemm_row_major, alpha, beta,
                LDA, LDC, M, N, &strides));
        CHECK(brgemm_desc_set_attr(&bcp, brg_attr));
        CHECK(brgemm_desc_set_postops(&bcp, attr(), dst_md(), LDD, jcp.bia_dt));
        CHECK(brgemm_desc_finalize(&bcp));
        ++idx;
        return status::success;
    };

    bcps_.resize(1);
    jcp.ow_block = jcp.ow;
    jcp.nb_ow = 1;
    jcp.nb_ch_blocking = jcp.ngroups;
    jcp.chb_tail = 0;
    int ker_idx = 0;
    CHECK(init_bcp(ker_idx, jcp.ow, jcp.ngroups)); // default full row kernel.

    const auto &bcp_0 = bcps_[0];
    jcp.ch_block = bcp_0.ld_block;
    jcp.nb_ch = div_up(jcp.ngroups, jcp.ch_block);
    const auto wei_tag = is_3d
            ? (jcp.ch_block == 16 ? format_tag::dhwioG16g
                                  : (jcp.ch_block == 8 ? format_tag::dhwioG8g
                                                       : format_tag::dhwioG4g))
            : (jcp.ch_block == 16 ? format_tag::hwioG16g
                                  : (jcp.ch_block == 8 ? format_tag::hwioG8g
                                                       : format_tag::hwioG4g));

    const memory_desc_wrapper weights_d(&weights_md_);
    CHECK(init_tag(weights_md_, weights_d, wei_tag, true));

    if ((jcp.mb * jcp.od * jcp.oh) % jcp.nthr != 0) {
        // determine ow_block
        {
            const size_t work_amount = jcp.mb * jcp.oh * jcp.od * jcp.ow;

            if (work_amount % jcp.nthr == 0) {
                const size_t work_per_thr = div_up(work_amount, jcp.nthr);
                const size_t ow_tail_block
                        = (work_per_thr / jcp.nb_ch) % jcp.ow;

                if (ow_tail_block && (jcp.ow % ow_tail_block == 0))
                    jcp.ow_block = ow_tail_block;
                else { jcp.ow_block = jcp.ow; }
            } else {
                const int max_ow_block = is_superset(jcp.isa, sve_512)
                        ? 6
                        : bcp_0.bd_block2 /*TODO: Tune for sve256*/;
                jcp.ow_block = nstl::min(max_ow_block, jcp.ow);
            }
            jcp.ow_tail = jcp.ow % jcp.ow_block;
        }
        jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

        // determine nb_ch_block
        {
            const size_t work_amount
                    = jcp.mb * jcp.nb_ch * jcp.oh * jcp.od * jcp.nb_ow;
            if (work_amount % jcp.nthr == 0) {
                const size_t work_per_thr = div_up(work_amount, jcp.nthr);
                const size_t ch_tail_block = work_per_thr % jcp.nb_ch;
                if (ch_tail_block && (jcp.nb_ch % ch_tail_block == 0)) {
                    jcp.nb_ch_blocking = ch_tail_block * jcp.ch_block;
                } else {
                    jcp.nb_ch_blocking = jcp.ngroups;
                }
            } else {
                const int max_ch_block2
                        = is_superset(jcp.isa, sve_512) ? 4 : bcp_0.ld_block;
                jcp.nb_ch_blocking
                        = nstl::min(max_ch_block2 * jcp.ch_block, jcp.ngroups);
            }
            jcp.chb_tail = jcp.ngroups % jcp.nb_ch_blocking;
        }

        const int n_owb_kernels = std::ceil(log2(jcp.nb_ow));
        const int num_kernels = 1 /*full ow*/ + n_owb_kernels
                + (jcp.chb_tail != 0) + (jcp.nb_ch_blocking != jcp.ngroups)
                + (jcp.ow_tail != 0);
        bcps_.resize(num_kernels);

        for (int i = 0; i < n_owb_kernels; ++i) {
            CHECK(init_bcp(ker_idx, jcp.ow_block * (1 << i), jcp.ngroups));
        }

        if (jcp.chb_tail) {
            jcp.chb_tail_idx = ker_idx;
            CHECK(init_bcp(ker_idx, jcp.ow_block, jcp.chb_tail));
        }

        if (jcp.ow_tail) {
            jcp.ow_tail_idx = ker_idx;
            CHECK(init_bcp(ker_idx, jcp.ow_tail, jcp.ngroups));
        }

        if (jcp.nb_ch_blocking != jcp.ngroups) {
            jcp.nb_ch_blocking_idx = ker_idx;
            CHECK(init_bcp(ker_idx, jcp.ow_block, jcp.nb_ch_blocking));
        }
        assert(num_kernels == ker_idx);
    }
    return status::success;
}

template <cpu_isa_t isa>
status_t brdgmm_dw_convolution_fwd_t<isa>::init(engine_t *engine) {
    const auto &bcps = pd()->bcps_;
    brdgmm_kernels_.resize(bcps.size());
    for (size_t idx = 0; idx < bcps.size(); ++idx) {
        const auto &bcp = bcps[idx];
        if (bcp.bcast_dim * bcp.load_dim /* M*N */ == 0) continue;
        brgemm_kernel_t *brg_kernel = nullptr;
        CHECK(brgemm_kernel_create(&brg_kernel, pd()->bcps_[idx]));
        CHECK(safe_ptr_assign(brdgmm_kernels_[idx], brg_kernel));
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t brdgmm_dw_convolution_fwd_t<isa>::execute(
        const exec_ctx_t &ctx) const {
    const char *const __restrict src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const char *const __restrict weights
            = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    const char *const __restrict bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    char *const __restrict dst = CTX_OUT_MEM(const char *, DNNL_ARG_DST);

    const std::vector<const void *> post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

    const auto &jcp = pd()->jcp_;

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const float *oscales = precompute_scales(ctx.get_scratchpad_grantor(),
            src_scales, wei_scales, pd()->OC(), pd()->attr());

    const int chb_step = jcp.nb_ch_blocking;
    const int chb_work = div_up(jcp.ngroups, chb_step);
    const int ow_step = jcp.ow_block;
    const int work_amount = jcp.mb * jcp.oh * jcp.od * jcp.nb_ow * chb_work;
    const int max_bs = jcp.kd * jcp.kh * jcp.kw;

    const size_t src_ch_stride = jcp.src_dsz;
    const size_t src_w_stride = jcp.ngroups * jcp.src_dsz;
    const size_t src_h_stride = jcp.ngroups * jcp.iw * jcp.src_dsz;
    const size_t src_d_stride = jcp.ngroups * jcp.ih * jcp.iw * jcp.src_dsz;
    const size_t src_mb_stride
            = jcp.ngroups * jcp.id * jcp.ih * jcp.iw * jcp.src_dsz;
    const size_t wei_ch_stride = jcp.wei_dsz;
    const size_t dst_ch_stride = jcp.dst_dsz;
    const size_t dst_w_stride = jcp.ngroups * jcp.dst_dsz;
    const size_t dst_h_stride = jcp.ngroups * jcp.ow * jcp.dst_dsz;
    const size_t dst_d_stride = jcp.ngroups * jcp.oh * jcp.ow * jcp.dst_dsz;
    const size_t dst_mb_stride
            = jcp.ngroups * jcp.od * jcp.oh * jcp.ow * jcp.dst_dsz;

    const auto w_blk_info = get_blocks_info(jcp.iw, jcp.ow, jcp.kw,
            jcp.stride_w, jcp.l_pad, jcp.r_pad, jcp.ow_block);
    const auto h_blk_info = get_blocks_info(
            jcp.ih, jcp.oh, jcp.kh, jcp.stride_h, jcp.t_pad, jcp.b_pad, 1);
    const auto d_blk_info = get_blocks_info(
            jcp.id, jcp.od, jcp.kd, jcp.stride_d, jcp.f_pad, jcp.back_pad, 1);

    const int n_w_blks = w_blk_info.n_lpad_blks;
    const int n_h_blks = h_blk_info.n_lpad_blks
            + max(0, jcp.oh - h_blk_info.rpad_blk_start_idx);
    const int w_shift = jcp.ow_block * jcp.stride_w;
    const int rpad_0
            = (jcp.ow_block - 1) * jcp.stride_w + jcp.kw - (jcp.iw + jcp.l_pad);
    const int rpad_1 = rpad_0 + (nstl::max(0, -rpad_0) / w_shift + 1) * w_shift;
    const int n_rpad_blks
            = 1 + div_up(nstl::max(0, jcp.r_pad - (rpad_1 - w_shift)), w_shift);

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, chb {0}, od {0}, oh {0}, owb {0};

        auto iwork = start;
        const brgemm_kernel_t *kernel = nullptr;
        const brgemm_kernel_t *kernel_chb_tail
                = brdgmm_kernels_[jcp.chb_tail_idx].get();
        brgemm_post_ops_data_t post_ops_data;
        post_ops_data.binary_post_ops_rhs = post_ops_binary_rhs_arg_vec.data();
        post_ops_data.data_C_ptr_ = dst;

        while (iwork < end) {
            nd_iterator_init(iwork, n, jcp.mb, od, jcp.od, oh, jcp.oh, owb,
                    jcp.nb_ow, chb, chb_work);
            const bool is_m_tail = jcp.ow_tail != 0 && (owb + 1 == jcp.nb_ow);
            const bool is_n_tail = jcp.chb_tail != 0 && (chb + 1 == chb_work);
            if (is_m_tail && chb != 0) {
                // the tail ow_block is not split btw threads to reduce the
                // number of kernels.
                utils::nd_iterator_jump(iwork, end, n, jcp.mb, od, jcp.od, oh,
                        jcp.oh, owb, jcp.nb_ow, chb, chb_work);
                continue;
            }

            // Begin: get number of owb to process and its correspoinding ker_idx
            const auto rem_work = end - iwork;
            const int rem_row_owb
                    = saturate(1, jcp.nb_ow - owb, rem_work / chb_work);
            int cur_n_owb = 1;
            int ker_idx = 0;
            if (is_n_tail) {
                ker_idx = jcp.chb_tail_idx;
            } else if (is_m_tail) {
                ker_idx = jcp.ow_tail_idx;
            } else if (chb != 0 || rem_work < chb_work) {
                ker_idx = jcp.nb_ch_blocking_idx;
            } else if (rem_row_owb == jcp.nb_ow) {
                ker_idx = 0;
                cur_n_owb = jcp.nb_ow;
            } else {
                // The ow_tail kernel is processed alone, subtract if it exists.
                const int log_rem_owb = log2(rem_row_owb
                        - (owb + rem_row_owb >= jcp.nb_ow)
                                * (jcp.ow_tail != 0));

                cur_n_owb = (1 << log_rem_owb);
                ker_idx = log_rem_owb + 1; // add 1 as 0th is full row.
            }

            kernel = brdgmm_kernels_[ker_idx].get();
            // end ker_idx

            // Begin: get batch_element idx
            const int ow = owb * ow_step;
            const int id_s = od * jcp.stride_d - jcp.f_pad;
            const int ih_s = oh * jcp.stride_h - jcp.t_pad;
            const int iw_s = ow * jcp.stride_w - jcp.l_pad;
            const int d_bi = nstl::min(od, d_blk_info.n_lpad_blks - 1)
                    + nstl::max(0, od - d_blk_info.rpad_blk_start_idx + 1);
            const int h_bi = nstl::min(oh, h_blk_info.n_lpad_blks - 1)
                    + nstl::max(0, oh - h_blk_info.rpad_blk_start_idx + 1);
            const int w_bi = nstl::min(owb, w_blk_info.n_lpad_blks - 1);

            const int ow_e
                    = nstl::min(ow + cur_n_owb * jcp.ow_block, jcp.ow) - 1;
            const int rpad = ow_e * jcp.stride_w - jcp.l_pad + jcp.kw - jcp.iw;
            const int rpad_i
                    = div_up(nstl::max(0, rpad - rpad_1 + w_shift), w_shift);
            const int bi //[d_bi][h_bi][w_bi][rpad_i] _
                    = ((d_bi * n_h_blks + h_bi) * n_w_blks + w_bi) * n_rpad_blks
                    + rpad_i;

            assert(static_cast<int>(pd()->batches_.size())
                    >= (bi + 1) * max_bs);

            const brgemm_batch_element_t *brg_batch
                    = &(pd()->batches_[bi * max_bs]);
            const int bs = pd()->bs_[bi];
            // end: get batch_element idx

            int ch = chb * chb_step;

            auto *ptr_A = src + n * src_mb_stride + id_s * src_d_stride
                    + ih_s * src_h_stride + iw_s * src_w_stride
                    + ch * src_ch_stride;
            auto *ptr_B = weights + ch * wei_ch_stride;
            auto *ptr_C = dst + n * dst_mb_stride + od * dst_d_stride
                    + oh * dst_h_stride + ow * dst_w_stride
                    + ch * dst_ch_stride;

            const int rem_chb_work = chb_work - chb;

            int chb_loop_work = is_m_tail || (chb == 0 && rem_work >= chb_work)
                    ? 1 // Compute entire chb_work in single jit call
                    : nstl::min(rem_work, rem_chb_work);
            iwork += cur_n_owb * nstl::min(rem_work, rem_chb_work);

            while (chb_loop_work) {
                post_ops_data.bias = bias + ch * jcp.bia_dsz;
                post_ops_data.scales = &oscales[jcp.is_oc_scale * ch];
                post_ops_data.oc_logical_off = ch;
                post_ops_data.dst_scales = dst_scales;
                brgemm_kernel_execute_postops(kernel, bs, ptr_A, ptr_B,
                        brg_batch, ptr_C, ptr_C, post_ops_data,
                        nullptr /*scratch*/);
                ++chb;
                if (jcp.chb_tail != 0 && chb + 1 == chb_work)
                    kernel = kernel_chb_tail;
                ch += chb_step;
                ptr_A += chb_step * src_ch_stride;
                ptr_B += chb_step * wei_ch_stride;
                ptr_C += chb_step * dst_ch_stride;
                --chb_loop_work;
            }
        }
    });
    return status::success;
}
template struct brdgmm_dw_convolution_fwd_t<sve_512>;
template struct brdgmm_dw_convolution_fwd_t<sve_256>;
template struct brdgmm_dw_convolution_fwd_t<sve_128>;
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
