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

#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_brdgmm_dw_conv.hpp"
#include <cpu/x64/cpu_isa_traits.hpp>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::status;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

namespace {
struct blk_info_t {
    dim_t n_lpad_blks;
    dim_t rpad_blk_start_idx;
};
blk_info_t get_blocks_info(dim_t sp_i, dim_t sp_o, dim_t k, dim_t stride,
        dim_t lpad, dim_t rpad, dim_t blk_size) {

    const dim_t max_blks = div_up(sp_o, blk_size);
    const dim_t blk_shift = stride * blk_size;
    const dim_t n_lpad_blks = nstl::min(
            max_blks, div_up(lpad, blk_shift) + 1 /*include zero lpad*/);
    const dim_t rpad_blk_start_idx = saturate(
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

    VDISPATCH_CONV_IC(tag == tag_value, VERBOSE_UNSUPPORTED_TAG);

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

cpu_isa_t get_supported_isa(bool is_f32, bool is_int8, bool is_bf16,
        bool is_f16, bool is_f32_bf16, bool is_f32_f16) {
    std::vector<cpu_isa_t> isa_list;
    if (one_of(true, is_f32, is_f32_bf16, is_f32_f16)) {
        isa_list = {avx512_core, avx2};
    } else if (is_int8) {
        isa_list = {avx10_2, avx512_core_vnni, avx2_vnni_2, avx2_vnni};
    } else if (is_bf16) {
        isa_list = {avx512_core_bf16, avx2_vnni_2};
    } else if (is_f16) {
        isa_list = {avx10_2, avx512_core_fp16, avx2_vnni_2};
    }

    for (auto isa : isa_list) {
        if (mayiuse(isa)) return isa;
    }
    return isa_undef;
}

status_t brdgmm_dw_convolution_fwd_t::pd_t::init(engine_t *engine) {

    using skip_mask_t = primitive_attr_t::skip_mask_t;

    const auto &cd = *desc();
    const auto src_type = cd.src_desc.data_type;
    const auto wei_type = cd.weights_desc.data_type;
    const auto bia_type = cd.bias_desc.data_type;
    const auto dst_type = cd.dst_desc.data_type;

    const bool is_f32 = everyone_is(f32, src_type, wei_type, dst_type);
    const bool is_int8 = one_of(src_type, u8, s8) && wei_type == s8
            && one_of(dst_type, s32, f32, u8, s8, bf16);
    const bool is_bf16 = everyone_is(bf16, src_type, wei_type)
            && one_of(dst_type, bf16, f32);
    const bool is_f16 = everyone_is(f16, src_type, wei_type)
            && one_of(dst_type, f16, f32);
    const bool is_f32_bf16
            = everyone_is(f32, src_type, dst_type) && wei_type == bf16;
    const bool is_f32_f16
            = everyone_is(f32, src_type, dst_type) && wei_type == f16;
    const cpu_isa_t isa = get_supported_isa(
            is_f32, is_int8, is_bf16, is_f16, is_f32_bf16, is_f32_f16);

    auto skip_mask = skip_mask_t::post_ops | skip_mask_t::sum_dt;
    if (is_int8) skip_mask |= (skip_mask_t::scales | skip_mask_t::zero_points);

    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(one_of(true, is_f32, is_int8, is_bf16, is_f16, is_f32_f16,
                           is_f32_bf16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(
            IMPLICATION(is_int8,
                    one_of(bia_type, data_type::undef, f32, s32, s8, u8)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(IMPLICATION(!is_int8,
                           one_of(bia_type, data_type::undef, data_type::f32,
                                   src_type, dst_type)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(
            (isa != isa_undef) && mayiuse(isa), "undefined or unsupported isa");
    VDISPATCH_CONV(attr()->has_default_values(skip_mask, dst_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(attr()->post_ops_.check_sum_consistency(dst_type, is_int8),
            VERBOSE_UNSUPPORTED_POSTOP);

    auto &jcp = jcp_;

    const memory_desc_wrapper src_d(&src_md_);
    const memory_desc_wrapper weights_d(&weights_md_);
    const memory_desc_wrapper dst_d(&dst_md_);
    const memory_desc_wrapper bias_d(&bias_md_);

    VDISPATCH_CONV(
            impl::is_dense_format_kind({src_md(), weights_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Big int (> INT_MAX) values are unsupported and jcp fields may overflow
    // TODO: change data type of jcp fields to size_t
    VDISPATCH_CONV_IC(!has_large_size(cd, src_d, weights_d, dst_d),
            VERBOSE_BAD_PARAM, "large size is not supported");

    const int ndims = src_d.ndims();
    const bool is_3d = ndims == 5;
    // Currently this kernel only supports 2D and 3D convolutions.
    VDISPATCH_CONV(utils::one_of(ndims, 4, 5), VERBOSE_UNSUPPORTED_FEATURE,
            "does not support 2d/3d convolutions");
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    VDISPATCH_CONV(with_groups, VERBOSE_UNSUPPORTED_FEATURE,
            "non-grouped convolution in depthwise convolution "
            "implementation");
    // dilations are not supported
    VDISPATCH_CONV(!(cd.dilates[0] != 0 || cd.dilates[1] != 0
                           || (is_3d && cd.dilates[2] != 0)),
            VERBOSE_UNSUPPORTED_FEATURE, "dilations are not supported");

    jcp = zero<decltype(jcp)>();
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
            "multiple input/output channels detected");

    const auto def_data_tag = is_3d ? format_tag::ndhwc : format_tag::nhwc;
    const bool any_eligible = (cd.prop_kind == prop_kind::forward_inference)
            || is_3d || is_int8 || is_f16 || (isa == avx2_vnni_2 && is_bf16);
    CHECK(init_tag(src_md_, src_d, def_data_tag, any_eligible));
    CHECK(init_tag(dst_md_, dst_d, def_data_tag, any_eligible));

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

    jcp.s8s8_compensation_required = jcp.src_dt == s8 && !isa_has_s8s8(jcp.isa);
    VDISPATCH_CONV(!(jcp.s8s8_compensation_required
                           && cd.weights_desc.format_kind != format_kind::any),
            "compensation required for current datatype combination");

    const auto &wei_scales = attr()->scales_.get(DNNL_ARG_WEIGHTS);
    jcp.is_oc_scale = wei_scales.get_mask() > 0;
    jcp.with_dst_scales
            = !attr()->scales_.get(DNNL_ARG_DST).has_default_values();

    CHECK(attr_scales_ok());
    CHECK(attr_zero_points_ok({{DNNL_ARG_SRC, {0, 2}}, {DNNL_ARG_DST, {0}}}));

    const auto &zp = attr()->zero_points_;
    jcp.src_zero_point = !zp.has_default_values(DNNL_ARG_SRC);
    jcp.dst_zero_point = !zp.has_default_values(DNNL_ARG_DST);

    // Source zero_point requires compensation, thus, must initialize weights
    // descriptor and can't take predefined one.
    const bool src_zp_format_ok = IMPLICATION(jcp.src_zero_point,
            cd.weights_desc.format_kind == format_kind::any);
    VDISPATCH_CONV(src_zp_format_ok, VERBOSE_UNSUPPORTED_ZP_CFG);

    // strd is only feasible for 1D (i.e., height dim is one)
    // and if there are no tails (for calculating matrix_B strides).
    // Since, we cannot always predict the blocking is 8 or 16.
    if (jcp.kd == 1 && jcp.kh == 1 && jcp.ngroups % 16 == 0) {
        jcp.batch_kind = brgemm_strd;
    } else {
        jcp.batch_kind = brgemm_offs;
    }

    // to avoid cache concurrent access from different threads
    size_t sc_size = sizeof(brgemm_batch_element_t);
    jcp.adjusted_batch_size
            = div_up(rnd_up(jcp.kd * jcp.kh * jcp.kw * sc_size, 4096), sc_size);
    CHECK(init_brdgmm_conf());

    init_batch_elements();
    init_scratchpad();
    return status::success;
}

void brdgmm_dw_convolution_fwd_t::pd_t::init_batch_elements() {

    auto &jcp = jcp_;

    auto gen_batch_elements
            = [&jcp](dim_t fpad, dim_t backpad, dim_t tpad, dim_t bpad,
                      dim_t lpad, dim_t rpad, int &bs,
                      brgemm_batch_element_t *batches) {
        const bool requires_batch_pad
                = jcp.s8s8_compensation_required || jcp.src_zero_point;
        const size_t src_w_stride = jcp.ngroups * jcp.src_dsz;
        const size_t src_h_stride = jcp.ngroups * jcp.iw * jcp.src_dsz;
        const size_t src_d_stride = jcp.ngroups * jcp.ih * jcp.iw * jcp.src_dsz;

        const size_t wei_w_stride
                = rnd_up(jcp.ngroups, jcp.ch_block) * jcp.wei_dsz;
        const size_t wei_h_stride = wei_w_stride * jcp.kw;
        const size_t wei_d_stride = wei_h_stride * jcp.kh;

        const dim_t adj_backpad = nstl::max(dim_t(0), backpad);
        const dim_t adj_bpad = nstl::max(dim_t(0), bpad);

        for_(dim_t kd = 0; kd < jcp.kd; ++kd)
        for_(dim_t kh = 0; kh < jcp.kh; ++kh)
        for (dim_t kw = 0; kw < jcp.kw; ++kw) {
            const bool padded_bs = kd < fpad || kd >= jcp.kd - adj_backpad
                    || kh < tpad || kh >= jcp.kh - adj_bpad;
            if (!requires_batch_pad && padded_bs) continue;
            auto &batch = batches[bs];
            batch.vvpad.top
                    = div_up(nstl::max(dim_t(0), lpad - kw), jcp.stride_w);
            batch.vvpad.bottom = div_up(
                    nstl::max(dim_t(0), rpad - jcp.kw + kw + 1), jcp.stride_w);
            batch.has_s8s8_comp_batch_pad = padded_bs;
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

    const dim_t w_shift = jcp.ow_block * jcp.stride_w;
    const dim_t h_shift = jcp.stride_h;
    const dim_t d_shift = jcp.stride_d;

    const auto w_blk_info = get_blocks_info(jcp.iw, jcp.ow, jcp.kw,
            jcp.stride_w, jcp.l_pad, jcp.r_pad, jcp.ow_block);
    const dim_t rpad_0
            = (jcp.ow_block - 1) * jcp.stride_w + jcp.kw - (jcp.iw + jcp.l_pad);
    const dim_t rpad_1
            = rpad_0 + (nstl::max(dim_t(0), -rpad_0) / w_shift + 1) * w_shift;
    const dim_t n_uniq_rpads = 1
            + div_up(nstl::max(dim_t(0), jcp.r_pad - (rpad_1 - w_shift)),
                    w_shift);

    const auto h_blk_info = get_blocks_info(
            jcp.ih, jcp.oh, jcp.kh, jcp.stride_h, jcp.t_pad, jcp.b_pad, 1);

    const auto d_blk_info = get_blocks_info(
            jcp.id, jcp.od, jcp.kd, jcp.stride_d, jcp.f_pad, jcp.back_pad, 1);

    const dim_t max_bs = jcp.kd * jcp.kh * jcp.kw;
    const dim_t n_d_uniq_blks
            = d_blk_info.n_lpad_blks + (jcp.od - d_blk_info.rpad_blk_start_idx);
    const dim_t n_h_uniq_blks
            = h_blk_info.n_lpad_blks + (jcp.oh - h_blk_info.rpad_blk_start_idx);
    const dim_t n_w_uniq_lpads = w_blk_info.n_lpad_blks;
    const dim_t uniq_blks
            = n_d_uniq_blks * n_h_uniq_blks * n_w_uniq_lpads * n_uniq_rpads;

    bs_.resize(uniq_blks, 0);
    batches_.resize(bs_.size() * max_bs);
    int bi = 0;

    for_(dim_t odb = 0; odb < n_d_uniq_blks; ++odb)
    for_(dim_t ohb = 0; ohb < n_h_uniq_blks; ++ohb)
    for_(dim_t owb = 0; owb < n_w_uniq_lpads; ++owb)
    for (dim_t rpad_i = 0; rpad_i < n_uniq_rpads; ++rpad_i) {
        const dim_t lpad = jcp.l_pad - owb * w_shift;
        const dim_t rpad = rpad_i == 0
                ? rpad_0
                : nstl::min(jcp.r_pad, rpad_1 + (rpad_i - 1) * w_shift);

        const dim_t tpad = jcp.t_pad - ohb * h_shift;
        const dim_t oh = ohb < h_blk_info.n_lpad_blks
                ? ohb
                : (h_blk_info.rpad_blk_start_idx
                          + (ohb - h_blk_info.n_lpad_blks));
        const dim_t bpad = oh * h_shift + jcp.kh - (jcp.ih + jcp.t_pad);

        const dim_t fpad = jcp.f_pad - odb * d_shift;
        const dim_t od = odb < d_blk_info.n_lpad_blks
                ? odb
                : (d_blk_info.rpad_blk_start_idx
                          + (odb - d_blk_info.n_lpad_blks));
        const dim_t backpad = od * d_shift + jcp.kd - (jcp.id + jcp.f_pad);

        gen_batch_elements(fpad, backpad, tpad, bpad, lpad, rpad, bs_[bi],
                &batches_[bi * max_bs]);
        ++bi;
        assert(static_cast<int>(bs_.size()) >= bi);
    }
    assert(static_cast<int>(bs_.size()) == bi);
}

status_t brdgmm_dw_convolution_fwd_t::pd_t::init_brdgmm_conf() {

    auto &jcp = jcp_;
    const bool is_3d = ndims() == 5;

    auto init_bcp = [&](int &idx, const dim_t M, const dim_t N) {
        const float alpha = 1.f;
        const float beta = 0.f;
        const dim_t LDA = jcp.ngroups * jcp.stride_w;
        const dim_t LDC = jcp.ngroups;
        const dim_t LDD = jcp.ngroups;

        brgemm_attr_t brg_attr;
        brg_attr.max_bs = jcp.kw * jcp.kh * jcp.kd;
        brg_attr.max_top_vpad
                = static_cast<int>(nstl::max(dim_t(0), jcp.l_pad));
        brg_attr.max_bottom_vpad
                = static_cast<int>(nstl::max(dim_t(0), jcp.r_pad));
        brg_attr.max_top_bpad = static_cast<int>(
                nstl::max(dim_t(0), nstl::max(jcp.t_pad, jcp.f_pad)));
        brg_attr.max_bottom_bpad = static_cast<int>(
                nstl::max(dim_t(0), nstl::max(jcp.b_pad, jcp.back_pad)));
        brg_attr.hint_bs_group
                = is_superset(jcp.isa, avx512_core) && jcp.stride_w == 1
                ? static_cast<int>(jcp.kw)
                : 1;

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

    const auto wei_tag = is_3d   ? (jcp.ch_block == 16 ? format_tag::dhwioG16g
                                                       : format_tag::dhwioG8g)
            : jcp.ch_block == 16 ? format_tag::hwioG16g
                                 : format_tag::hwioG8g;

    const memory_desc_wrapper weights_d(&weights_md_);
    CHECK(init_tag(weights_md_, weights_d, wei_tag, true));

    if (jcp.s8s8_compensation_required) {
        weights_md_.extra.flags
                = 0 | memory_extra_flags::compensation_conv_s8s8;
        weights_md_.extra.compensation_mask = 0x1;
    }

    if (jcp.src_zero_point) {
        weights_md_.extra.flags
                |= memory_extra_flags::compensation_conv_asymmetric_src;
        weights_md_.extra.asymm_compensation_mask = 0x1;
    }

    if ((jcp.mb * jcp.od * jcp.oh) % jcp.nthr != 0) {
        // determine ow_block
        {
            const size_t work_amount = jcp.mb * jcp.od * jcp.oh * jcp.ow;
            if (work_amount % jcp.nthr == 0) {
                const size_t work_per_thr = div_up(work_amount, jcp.nthr);
                const size_t ow_tail_block
                        = (work_per_thr / jcp.nb_ch) % jcp.ow;
                if (ow_tail_block && (jcp.ow % ow_tail_block == 0))
                    jcp.ow_block = ow_tail_block;
                else { jcp.ow_block = jcp.ow; }
            } else {
                const dim_t max_ow_block = is_superset(jcp.isa, avx512_core)
                        ? 6
                        : bcp_0.bd_block2 /*TODO: Tune for avx2*/;
                jcp.ow_block = nstl::min(max_ow_block, jcp.ow);
            }
            jcp.ow_tail = jcp.ow % jcp.ow_block;
        }
        jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

        // determine nb_ch_block
        {
            const size_t work_amount
                    = jcp.mb * jcp.nb_ch * jcp.od * jcp.oh * jcp.nb_ow;
            if (work_amount % jcp.nthr == 0) {
                const size_t work_per_thr = div_up(work_amount, jcp.nthr);
                const size_t ch_tail_block = work_per_thr % jcp.nb_ch;
                if (ch_tail_block && (jcp.nb_ch % ch_tail_block == 0))
                    jcp.nb_ch_blocking = ch_tail_block * jcp.ch_block;
                else
                    jcp.nb_ch_blocking = jcp.ngroups;
            } else {
                const dim_t max_ch_block2 = is_superset(jcp.isa, avx512_core)
                        ? 4
                        : bcp_0.ld_block2 /*TODO: Tune for avx2*/;
                jcp.nb_ch_blocking
                        = nstl::min(max_ch_block2 * jcp.ch_block, jcp.ngroups);
            }
            jcp.chb_tail = jcp.ngroups % jcp.nb_ch_blocking;
        }

        const int n_owb_kernels = static_cast<int>(
                std::ceil(log2(static_cast<double>(jcp.nb_ow))));
        const int num_kernels = 1 /*full ow*/ + n_owb_kernels
                + (jcp.chb_tail != 0) + (jcp.nb_ch_blocking != jcp.ngroups)
                + (jcp.ow_tail != 0);
        bcps_.resize(num_kernels);

        for (int i = 0; i < n_owb_kernels; ++i) {
            CHECK(init_bcp(
                    ker_idx, jcp.ow_block * (dim_t {1} << i), jcp.ngroups));
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

void brdgmm_dw_convolution_fwd_t::pd_t::init_scratchpad() {
    auto &jcp = jcp_;
    auto scratchpad = scratchpad_registry().registrar();

    if (jcp.with_dst_scales) {
        // See brgemm_types.hpp comment for `with_dst_scales`.
        scratchpad.book(key_conv_dst_scales,
                static_cast<size_t>(jcp.nthr) * sizeof(float), sizeof(float));
    }
}

status_t brdgmm_dw_convolution_fwd_t::init(engine_t *engine) {
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

status_t brdgmm_dw_convolution_fwd_t::execute(const exec_ctx_t &ctx) const {

    const char *const __restrict src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    const char *const __restrict weights
            = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    const char *const __restrict bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    char *const __restrict dst = CTX_OUT_MEM(const char *, DNNL_ARG_DST);
    const std::vector<const void *> post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

    const auto *jcp_ptr = &pd()->jcp_;

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

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const size_t wei_size = weights_d.size();
    const auto extra_data_offset
            = wei_size - weights_d.additional_buffer_size();
    const size_t s8_offset = jcp_ptr->s8s8_compensation_required
            ? rnd_up(jcp_ptr->ngroups, jcp_ptr->ch_block) * sizeof(int32_t)
            : 0;
    int32_t *s8s8_comp_ptr = jcp_ptr->s8s8_compensation_required
            ? reinterpret_cast<int32_t *>(
                      const_cast<char *>(weights) + extra_data_offset)
            : nullptr;
    int32_t *zp_compensation = jcp_ptr->src_zero_point
            ? reinterpret_cast<int32_t *>(const_cast<char *>(weights)
                      + extra_data_offset + s8_offset)
            : nullptr;

    parallel(jcp_ptr->nthr,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &jcp = *jcp_ptr;

        const int chb_step = static_cast<int>(jcp.nb_ch_blocking);
        const int chb_work = div_up(static_cast<int>(jcp.ngroups), chb_step);
        const int ow_step = static_cast<int>(jcp.ow_block);
        const int work_amount = static_cast<int>(
                jcp.mb * jcp.od * jcp.oh * jcp.nb_ow * chb_work);

        const int max_bs = static_cast<int>(jcp.kd * jcp.kh * jcp.kw);

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
        const auto d_blk_info = get_blocks_info(jcp.id, jcp.od, jcp.kd,
                jcp.stride_d, jcp.f_pad, jcp.back_pad, 1);
        const int n_w_blks = static_cast<int>(w_blk_info.n_lpad_blks);
        const int n_h_blks = static_cast<int>(h_blk_info.n_lpad_blks)
                + nstl::max(0,
                        static_cast<int>(
                                jcp.oh - h_blk_info.rpad_blk_start_idx));

        const int w_shift = static_cast<int>(jcp.ow_block * jcp.stride_w);
        const int rpad_0 = static_cast<int>((jcp.ow_block - 1) * jcp.stride_w
                + jcp.kw - (jcp.iw + jcp.l_pad));
        const int rpad_1
                = rpad_0 + (nstl::max(0, -rpad_0) / w_shift + 1) * w_shift;
        const int n_rpad_blks = 1
                + div_up(nstl::max(0,
                                 static_cast<int>(
                                         jcp.r_pad - (rpad_1 - w_shift))),
                        w_shift);

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

        float *dst_scales_inv_ptr = nullptr;
        if (jcp.with_dst_scales) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr
                    = scratchpad.template get<float>(key_conv_dst_scales)
                    + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

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

            // Begin: get number of owb to process and its corresponding ker_idx
            const auto rem_work = end - iwork;
            const int rem_row_owb = saturate(
                    1, static_cast<int>(jcp.nb_ow - owb), rem_work / chb_work);
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
                cur_n_owb = static_cast<int>(jcp.nb_ow);
            } else {
                // The ow_tail kernel is processed alone, subtract if it exists.
                const int log_rem_owb = static_cast<int>(log2(rem_row_owb
                        - (owb + rem_row_owb >= jcp.nb_ow)
                                * (jcp.ow_tail != 0)));
                cur_n_owb = (1 << log_rem_owb);
                ker_idx = log_rem_owb + 1; // add 1 as 0th is full row.
            }

            kernel = brdgmm_kernels_[ker_idx].get();
            // end ker_idx

            // Begin: get batch_element idx
            const int ow = owb * ow_step;

            const int id_s = static_cast<int>(od * jcp.stride_d - jcp.f_pad);
            const int ih_s = static_cast<int>(oh * jcp.stride_h - jcp.t_pad);
            const int iw_s = static_cast<int>(ow * jcp.stride_w - jcp.l_pad);

            const int d_bi
                    = nstl::min(
                              od, static_cast<int>(d_blk_info.n_lpad_blks - 1))
                    + nstl::max(0,
                            static_cast<int>(
                                    od - d_blk_info.rpad_blk_start_idx + 1));
            const int h_bi
                    = nstl::min(
                              oh, static_cast<int>(h_blk_info.n_lpad_blks - 1))
                    + nstl::max(0,
                            static_cast<int>(
                                    oh - h_blk_info.rpad_blk_start_idx + 1));
            const int w_bi = nstl::min(
                    owb, static_cast<int>(w_blk_info.n_lpad_blks - 1));

            const int ow_e
                    = nstl::min(ow + cur_n_owb * static_cast<int>(jcp.ow_block),
                              static_cast<int>(jcp.ow))
                    - 1;
            const int rpad = static_cast<int>(
                    ow_e * jcp.stride_w - jcp.l_pad + jcp.kw - jcp.iw);
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

            auto *ptr_A = src
                    + static_cast<ptrdiff_t>(n * src_mb_stride
                            + id_s * src_d_stride + ih_s * src_h_stride
                            + iw_s * src_w_stride + ch * src_ch_stride);
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
                post_ops_data.src_scales = src_scales;
                post_ops_data.wei_scales = wei_scales
                        ? static_cast<const char *>(wei_scales)
                                + jcp.is_oc_scale * ch * sizeof(float)
                        : nullptr;
                post_ops_data.dst_scales = dst_scales_inv_ptr;
                post_ops_data.oc_logical_off = ch;
                const bool is_bcast_zp
                        = pd()->attr()->zero_points_.get_mask(DNNL_ARG_SRC)
                        == 0;
                post_ops_data.a_zp_values = jcp.src_zero_point
                        ? src_zero_points + ch * !is_bcast_zp
                        : nullptr;
                post_ops_data.c_zp_values = dst_zero_points;
                post_ops_data.a_zp_compensations
                        = jcp.src_zero_point ? zp_compensation + ch : nullptr;

                void *scratch = jcp.s8s8_compensation_required
                        ? static_cast<void *>(s8s8_comp_ptr + ch)
                        : nullptr;
                brgemm_kernel_execute_postops(kernel, bs, ptr_A, ptr_B,
                        brg_batch, ptr_C, ptr_C, post_ops_data, scratch);
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
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
