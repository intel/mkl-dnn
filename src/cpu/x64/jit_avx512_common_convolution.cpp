/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#include "cpu/x64/jit_avx512_common_convolution.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using jit_conv_ker_t = void (*)(jit_conv_args_t *);

inline void jit_conv_ker_pipeline(const jit_conv_ker_t ker, jit_conv_args_t &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int reduce_work, int load_work) {
    p.src = src;
    p.dst = dst;
    p.filt = filt;
    p.bias = bias;
    p.channel = channel;
    // non-positive value of kh_padding is allowed, in this case kernel must
    // skip computation part and initialize output by zeroes
    p.kh_padding = kh_padding;
    p.reduce_work = reduce_work;
    p.load_work = load_work;

    ker(&p);
}
// The special case for the driver with iw-parallelization (BWD)
inline void jit_conv_ker_pipeline_iw_thr(const jit_conv_ker_t ker,
        jit_conv_args_t &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int iwb, int reduce_work,
        int load_work) {
    p.iwb = iwb;

    jit_conv_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            reduce_work, load_work);
}

inline void jit_conv_3d_ker_pipeline(const jit_conv_ker_t ker,
        jit_conv_args_t &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int kd_padding,
        int reduce_work, int load_work) {
    p.src = src;
    p.dst = dst;
    p.filt = filt;
    p.bias = bias;
    p.channel = channel;
    // non-positive value of both kd_padding and kh_padding is allowed, in this
    // case kernel must skip computation part and initialize output by zeroes
    p.kh_padding = kh_padding;
    p.kd_padding = kd_padding;
    p.reduce_work = reduce_work;
    p.load_work = load_work;

    ker(&p);
}
// The special case for the driver with ow-parallelization (FWD)
inline void jit_conv_ker_pipeline_ow_thr(jit_conv_ker_t ker, jit_conv_args_t &p,
        const void *src, const void *dst, const void *filt, const void *bias,
        int channel, int kh_padding, int owb, int reduce_work, int load_work,
        const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
        int flags) {
    p.owb = owb;
    p.flags = flags;

    p.dst_orig = dst_orig;
    p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;

    jit_conv_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            reduce_work, load_work);
}

// The special case for the driver with ow-parallelization (FWD)
// TODO: implement it for BWD_D and BWD_W too
inline void jit_conv_3d_ker_pipeline_ow_thr(const jit_conv_ker_t ker,
        jit_conv_args_t &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, int kd_padding, int owb,
        int reduce_work, int load_work, const void *post_ops_binary_rhs_arg_vec,
        const void *dst_orig, int flags) {

    p.dst_orig = dst_orig;
    p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;

    p.owb = owb;
    p.flags = flags;

    jit_conv_3d_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            kd_padding, reduce_work, load_work);
}

inline void jit_conv_ker_pipeline_bwd_w(const jit_conv_ker_t ker,
        jit_conv_args_t &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int kh_padding, size_t reduce_work,
        size_t load_work) {
    jit_conv_ker_pipeline(ker, p, src, dst, filt, bias, channel, kh_padding,
            static_cast<int>(reduce_work), static_cast<int>(load_work));
}

void jit_conv_2d_ker_bwd_w_pipeline(const jit_conv_ker_t ker,
        jit_conv_args_t &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int os_index_begin, int os_index_end,
        int kh_padding /* kh_work_size */, size_t kh_offset, size_t reduce_work,
        size_t load_work) {
    p.src = src;
    p.dst = dst;
    p.filt = filt;
    p.bias = bias;
    p.channel = channel;
    p.os_index_begin = os_index_begin;
    p.os_index_end = os_index_end;
    // non-positive value of kh_padding is allowed, in this case kernel must
    // skip kw loop computation and initialize output by zeroes
    p.kh_padding = kh_padding;
    p.kh_offset = kh_offset;
    p.reduce_work = reduce_work;
    p.load_work = load_work;

    ker(&p);
}

void jit_conv_3d_ker_bwd_w_pipeline(const jit_conv_ker_t ker,
        jit_conv_args_t &p, const void *src, const void *dst, const void *filt,
        const void *bias, int channel, int os_index_begin, int os_index_end,
        int kd_padding /* kd_work_size */, size_t kd_offset, size_t reduce_work,
        size_t load_work) {
    p.src = src;
    p.dst = dst;
    p.filt = filt;
    p.bias = bias;
    p.channel = channel;
    p.os_index_begin = os_index_begin;
    p.os_index_end = os_index_end;
    // non-positive value of kd_padding is allowed, in this case kernel must
    // skip kh loop computation and initialize output by zeroes
    p.kd_padding = kd_padding;
    p.kd_offset = kd_offset;
    p.reduce_work = reduce_work;
    p.load_work = load_work;

    ker(&p);
}
#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type,
        dst_type>::prepare_padded_bias(const dst_data_t *&bias,
        const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    auto padded_bias
            = scratchpad.template get<dst_data_t>(key_conv_padded_bias);
    utils::array_copy(padded_bias, bias, pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + pd()->jcp_.oc_without_padding, (dst_data_t)0,
            pd()->jcp_.oc - pd()->jcp_.oc_without_padding);
    bias = padded_bias;
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_1d(const exec_ctx_t &ctx) const {
    const auto *jcp_ptr = &pd()->jcp_;

    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp_ptr->post_ops, ctx);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    parallel(jcp_ptr->aligned_threads,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &jcp = *jcp_ptr;

        const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
        assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);
        dim_t oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        constexpr int g_blocking = 1;
        dim_t nb_groups = jcp.ngroups / g_blocking;
        dim_t work_amount = jcp.mb * nb_groups * oc_chunks * jcp.nb_ow;
        dim_t start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_args_t();

        // The second arg in template means sub_offset0 = true
        // See `blk_off` method definition.
        size_t src_c_stride = src_d.blk_off<false, true>(0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (dim_t icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            dim_t n {0}, gg {0}, occ {0}, owb {0};

            if (jcp.loop_order == loop_cwgn) {
                int dummy {0};
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gncw) {
                int dummy {0};
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, dummy, 1);
            } else if (jcp.loop_order == loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, owb, jcp.nb_ow, occ,
                        oc_chunks, gg, nb_groups);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                dim_t ocb = occ * jcp.nb_oc_blocking;
                dim_t g = gg * g_blocking;
                dim_t g_ocb = g * jcp.nb_oc + ocb;
                dim_t g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                dim_t ow_s = owb * jcp.ow_block;
                dim_t iw_s = ow_s * jcp.stride_w;
                const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::nwc;
                const dim_t oc_off_idx = is_dst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g_ocb;
                auto dst_w = dst + dst_d.blk_off(n, oc_off_idx, ow_s);
                const bool is_src_layout_nxc = jcp.src_tag == format_tag::nwc;
                const dim_t ic_off_idx = is_src_layout_nxc
                        ? g * jcp.ic + icb_l2 * jcp.ic_block
                        : g_icb + icb_l2;
                auto src_w = src + src_d.blk_off(n, ic_off_idx, iw_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, icb_l2);
                auto bias_w = bias ? bias
                                + oc_off_idx
                                        * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                                   : nullptr;

                dim_t icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                dim_t icb_end
                        = nstl::min<dim_t>(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                const dim_t oc_work = utils::this_block_size(ocb * jcp.oc_block,
                        jcp.oc_without_padding,
                        jcp.nb_oc_blocking * jcp.oc_block);

                dim_t ic_work = icb_step * jcp.ic_block;

                for (dim_t icb = icb_l2; icb < icb_end; icb += icb_step) {
                    int curr_nb_ic = static_cast<int>(
                            nstl::min<dim_t>(icb_step, icb_end - icb));
                    int flags = 0;
                    if (icb == 0) flags |= FLAG_IC_FIRST;
                    if (icb + curr_nb_ic >= jcp.nb_ic) {
                        flags |= FLAG_IC_LAST;
                        ic_work = utils::this_block_size(icb * jcp.ic_block,
                                jcp.ic, icb_step * jcp.ic_block);
                    }
                    jit_conv_ker_pipeline_ow_thr(jit_ker, par_conv, src_w,
                            dst_w, wht_w, bias_w, static_cast<int>(icb), 1,
                            static_cast<int>(owb), static_cast<int>(ic_work),
                            static_cast<int>(oc_work),
                            post_ops_binary_rhs_arg_vec.data(), dst, flags);

                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }
                if (jcp.loop_order == loop_cwgn) {
                    int dummy {0};
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, dummy, 1);
                } else if (jcp.loop_order == loop_gncw) {
                    int dummy {0};
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, dummy, 1);
                } else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, owb, jcp.nb_ow, occ, oc_chunks,
                            gg, nb_groups);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }
    });
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_2d(const exec_ctx_t &ctx) const {

    const auto *jcp_ptr = &pd()->jcp_;
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp_ptr->post_ops, ctx);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    parallel(jcp_ptr->aligned_threads,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &jcp = *jcp_ptr;

        const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
        assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);
        dim_t oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        constexpr int g_blocking = 1;
        dim_t nb_groups = jcp.ngroups / g_blocking;
        dim_t work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh * jcp.nb_ow;
        dim_t start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_args_t();

        // The second arg in template means sub_offset0 = true
        // See `blk_off` method definition.
        size_t src_h_stride = src_d.blk_off<false, true>(0, 0, 1);
        size_t src_c_stride = src_d.blk_off<false, true>(0, 1);
        size_t dst_h_stride = dst_d.blk_off<false, true>(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (dim_t icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            dim_t n {0}, gg {0}, occ {0}, oh_s {0}, owb {0};

            if (jcp.loop_order == loop_cwgn)
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow,
                        occ, oc_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                dim_t ocb = occ * jcp.nb_oc_blocking;
                dim_t g = gg * g_blocking;
                dim_t g_ocb = g * jcp.nb_oc + ocb;
                dim_t g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                dim_t work_rem = end - start;

                dim_t ow_s = owb * jcp.ow_block;
                dim_t iw_s = ow_s * jcp.stride_w;
                dim_t oh_e
                        = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    oh_e = oh_s + 1; //step instead

                for (dim_t oh_b = oh_s; oh_b < oh_e; oh_b += jcp.h_blocking) {
                    dim_t ih_b = -jcp.t_pad + oh_b * jcp.stride_h;
                    const bool is_dst_layout_nxc
                            = jcp.dst_tag == format_tag::nhwc;
                    const dim_t oc_off_idx = is_dst_layout_nxc
                            ? g * jcp.oc + ocb * jcp.oc_block
                            : g_ocb;
                    auto dst_w = dst + dst_d.blk_off(n, oc_off_idx, oh_b, ow_s);
                    const bool is_src_layout_nxc
                            = jcp.src_tag == format_tag::nhwc;
                    const dim_t ic_off_idx = is_src_layout_nxc
                            ? g * jcp.ic + icb_l2 * jcp.ic_block
                            : g_icb + icb_l2;
                    auto src_w = src + src_d.blk_off(n, ic_off_idx, ih_b, iw_s);
                    auto wht_w
                            = weights + wht_blk_off(weights_d, g, ocb, icb_l2);

                    dim_t icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                    dim_t icb_end = nstl::min<dim_t>(
                            jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                    auto bias_w = bias ? bias
                                    + oc_off_idx
                                            * (is_dst_layout_nxc ? 1
                                                                 : jcp.oc_block)
                                       : nullptr;
                    const dim_t oc_work = utils::this_block_size(
                            ocb * jcp.oc_block, jcp.oc_without_padding,
                            jcp.nb_oc_blocking * jcp.oc_block);
                    dim_t ic_work = icb_step * jcp.ic_block;
                    for (dim_t icb = icb_l2; icb < icb_end; icb += icb_step) {
                        int curr_nb_ic = static_cast<int>(
                                nstl::min<dim_t>(icb_step, icb_end - icb));
                        int flags = 0;
                        if (icb == 0) flags |= FLAG_IC_FIRST;
                        if (icb + curr_nb_ic >= jcp.nb_ic) {
                            flags |= FLAG_IC_LAST;
                            ic_work = utils::this_block_size(icb * jcp.ic_block,
                                    jcp.ic, icb_step * jcp.ic_block);
                        }
                        auto src_c = src_w;
                        auto dst_c = dst_w;
                        for (dim_t oj = oh_b, ij = ih_b; oj
                                < nstl::min<dim_t>(oh_e, oh_b + jcp.h_blocking);
                                ++oj, ij += jcp.stride_h) {
                            dim_t dilate_h = jcp.dilate_h + 1;
                            dim_t i_t_overflow
                                    = div_up(max<dim_t>(0, -ij), dilate_h);
                            dim_t i_b_overflow = div_up(
                                    max<dim_t>(0,
                                            ij - jcp.ih
                                                    + (jcp.kh - 1) * dilate_h
                                                    + 1),
                                    dilate_h);
                            dim_t kh_padding = nstl::max<dim_t>(
                                    0, jcp.kh - i_t_overflow - i_b_overflow);

                            auto aux_src = src_c
                                    + i_t_overflow * dilate_h * src_h_stride;
                            auto aux_wht = wht_w + i_t_overflow * wht_h_stride;

                            jit_conv_ker_pipeline_ow_thr(jit_ker, par_conv,
                                    aux_src, dst_c, aux_wht, bias_w,
                                    static_cast<int>(icb),
                                    static_cast<int>(kh_padding),
                                    static_cast<int>(owb),
                                    static_cast<int>(ic_work),
                                    static_cast<int>(oc_work),
                                    post_ops_binary_rhs_arg_vec.data(), dst,
                                    flags);

                            src_c += src_h_stride * jcp.stride_h;
                            dst_c += dst_h_stride;
                        }
                        src_w += src_c_stride;
                        wht_w += wht_ic_stride;
                    }
                }

                if (jcp.loop_order == loop_cwgn)
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, oh_s, jcp.oh);
                else if (jcp.loop_order == loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
                else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow,
                            occ, oc_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }
    });
}

template <data_type_t src_type, data_type_t wei_type, data_type_t dst_type>
void jit_avx512_common_convolution_fwd_t<src_type, wei_type,
        dst_type>::execute_forward_3d(const exec_ctx_t &ctx) const {
    const auto *jcp_ptr = &pd()->jcp_;
    auto src = CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const dst_data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(jcp_ptr->post_ops, ctx);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    parallel(jcp_ptr->nthr,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &jcp = *jcp_ptr;

        const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
        assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);
        dim_t oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
        constexpr int g_blocking = 1;
        dim_t nb_groups = jcp.ngroups / g_blocking;
        dim_t work_amount
                = jcp.mb * nb_groups * oc_chunks * jcp.od * jcp.oh * jcp.nb_ow;
        dim_t start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_args_t();

        // The second arg in template means sub_offset0 = true
        // See `blk_off` method definition.
        size_t src_d_stride = src_d.blk_off<false, true>(0, 0, 1);
        size_t src_h_stride = src_d.blk_off<false, true>(0, 0, 0, 1);
        size_t src_c_stride = src_d.blk_off<false, true>(0, 1);
        size_t dst_h_stride = dst_d.blk_off<false, true>(0, 0, 0, 1);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);
        size_t wht_ic_stride = wht_blk_off(weights_d, 0, 0, 1);

        for (dim_t icb_l2 = 0; icb_l2 < jcp.nb_ic; icb_l2 += jcp.nb_ic_L2) {
            start = start_copy;
            dim_t n {0}, gg {0}, occ {0}, oh_s {0}, od_s {0}, owb {0};

            if (jcp.loop_order == loop_cwgn)
                nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                        nb_groups, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ,
                        oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s, jcp.oh);
            else if (jcp.loop_order == loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh,
                        owb, jcp.nb_ow, occ, oc_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                dim_t ocb = occ * jcp.nb_oc_blocking;
                dim_t g = gg * g_blocking;
                dim_t g_ocb = g * jcp.nb_oc + ocb;
                dim_t g_icb = g * jcp.nb_ic * jcp.nonblk_group_off;

                dim_t work_rem = end - start;
                dim_t ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
                dim_t ow_s = owb * jcp.ow_block;
                dim_t iw_s = ow_s * jcp.stride_w;
                dim_t oh_e
                        = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    oh_e = oh_s + 1; //step instead

                dim_t id_s = -jcp.f_pad + od_s * jcp.stride_d;

                dim_t dilate_d = jcp.dilate_d + 1;
                dim_t d_t_overflow = div_up(max<dim_t>(0, -id_s), dilate_d);
                dim_t d_b_overflow = div_up(
                        max<dim_t>(
                                0, id_s - jcp.id + (jcp.kd - 1) * dilate_d + 1),
                        dilate_d);
                dim_t kd_padding = nstl::max<dim_t>(
                        0, jcp.kd - d_t_overflow - d_b_overflow);
                const bool is_dst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
                const dim_t oc_off_idx = is_dst_layout_nxc
                        ? g * jcp.oc + ocb * jcp.oc_block
                        : g_ocb;
                auto dst_w
                        = dst + dst_d.blk_off(n, oc_off_idx, od_s, oh_s, ow_s);
                const bool is_src_layout_nxc = jcp.src_tag == format_tag::ndhwc;
                const dim_t ic_off_idx = is_src_layout_nxc
                        ? g * jcp.ic + icb_l2 * jcp.ic_block
                        : g_icb + icb_l2;
                auto src_w = src
                        + src_d.blk_off(n, ic_off_idx, id_s, ih_s, iw_s)
                        + d_t_overflow * dilate_d * src_d_stride;
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, icb_l2)
                        + d_t_overflow * wht_d_stride;
                auto bias_w = bias ? bias
                                + oc_off_idx
                                        * (is_dst_layout_nxc ? 1 : jcp.oc_block)
                                   : nullptr;

                const dim_t icb_step = is_src_layout_nxc ? jcp.nb_ic_L2 : 1;
                dim_t icb_end
                        = nstl::min<dim_t>(jcp.nb_ic, icb_l2 + jcp.nb_ic_L2);
                const dim_t oc_work = utils::this_block_size(ocb * jcp.oc_block,
                        jcp.oc_without_padding,
                        jcp.nb_oc_blocking * jcp.oc_block);

                dim_t ic_work = icb_step * jcp.ic_block;
                for (dim_t icb = icb_l2; icb < icb_end; icb += icb_step) {
                    int curr_nb_ic = static_cast<int>(
                            nstl::min<dim_t>(icb_step, icb_end - icb));
                    int flags = 0;
                    if (icb == 0) flags |= FLAG_IC_FIRST;
                    if (icb + curr_nb_ic >= jcp.nb_ic) {
                        flags |= FLAG_IC_LAST;
                        ic_work = utils::this_block_size(icb * jcp.ic_block,
                                jcp.ic, icb_step * jcp.ic_block);
                    }
                    auto src_c = src_w;
                    auto dst_c = dst_w;
                    for (dim_t oj = oh_s, ij = ih_s; oj < oh_e;
                            ++oj, ij += jcp.stride_h) {
                        dim_t dilate_h = jcp.dilate_h + 1;
                        dim_t i_t_overflow
                                = div_up(max<dim_t>(0, -ij), dilate_h);
                        dim_t i_b_overflow = div_up(
                                max<dim_t>(0,
                                        ij - jcp.ih + (jcp.kh - 1) * dilate_h
                                                + 1),
                                dilate_h);
                        dim_t kh_padding = nstl::max<dim_t>(
                                0, jcp.kh - i_t_overflow - i_b_overflow);
                        jit_conv_3d_ker_pipeline_ow_thr(jit_ker, par_conv,
                                src_c + i_t_overflow * dilate_h * src_h_stride,
                                dst_c, wht_w + i_t_overflow * wht_h_stride,
                                bias_w, static_cast<int>(icb),
                                static_cast<int>(kh_padding),
                                static_cast<int>(kd_padding),
                                static_cast<int>(owb),
                                static_cast<int>(ic_work),
                                static_cast<int>(oc_work),
                                post_ops_binary_rhs_arg_vec.data(), dst, flags);

                        src_c += src_h_stride * jcp.stride_h;
                        dst_c += dst_h_stride;
                    }
                    src_w += src_c_stride;
                    wht_w += wht_ic_stride;
                }

                if (jcp.loop_order == loop_cwgn)
                    nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow,
                            gg, nb_groups, n, jcp.mb, od_s, jcp.od, oh_s,
                            jcp.oh);
                else if (jcp.loop_order == loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, occ,
                            oc_chunks, owb, jcp.nb_ow, od_s, jcp.od, oh_s,
                            jcp.oh);
                else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, od_s, jcp.od, oh_s, jcp.oh, owb,
                            jcp.nb_ow, occ, oc_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }
    });
}

template struct jit_avx512_common_convolution_fwd_t<data_type::f32>;

template <data_type_t diff_dst_type, data_type_t wei_type,
        data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
        diff_src_type>::execute_backward_data_1d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    dim_t ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    dim_t nb_groups = jcp.ngroups / g_blocking;
    dim_t work_amount = nb_groups * jcp.mb * ic_chunks * jcp.nb_iw;
    int nthr = jcp.nthr;

    parallel(nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        dim_t start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_args_t();

        // The second arg in template means sub_offset0 = true
        // See `blk_off` method definition.
        size_t diff_dst_c_stride = diff_dst_d.blk_off<false, true>(0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        for (dim_t ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            dim_t n {0}, gg {0}, icc {0}, iwb {0};
            if (jcp.loop_order == loop_cwgn) {
                int dummy {0};
                nd_iterator_init(start, icc, ic_chunks, iwb, jcp.nb_iw, gg,
                        nb_groups, n, jcp.mb, dummy, 1);
            } else if (jcp.loop_order == loop_gncw) {
                int dummy {0};
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, iwb, jcp.nb_iw, dummy, 1);
            } else if (jcp.loop_order == loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, iwb, jcp.nb_iw, icc,
                        ic_chunks, gg, nb_groups);
            } else {
                assert(!"unsupported loop order");
            }

            while (start < end) {
                dim_t icb = icc * jcp.nb_ic_blocking;
                dim_t g = gg * g_blocking;
                dim_t g_icb = g * jcp.nb_ic + icb;
                dim_t g_ocb = g * jcp.nb_oc;
                dim_t iw_s = iwb * jcp.iw_block;
                dim_t ow_s = iw_s / jcp.stride_w;

                const bool is_dsrc_layout_nxc = jcp.src_tag == format_tag::nwc;
                const dim_t ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w
                        = diff_src + diff_src_d.blk_off(n, ic_off_idx, iw_s);
                const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nwc;
                const dim_t oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w
                        = diff_dst + diff_dst_d.blk_off(n, oc_off_idx, ow_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb);

                dim_t ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                dim_t ocb_end
                        = nstl::min<dim_t>(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const dim_t load_work
                        = utils::this_block_size(icb * jcp.ic_block, jcp.ic,
                                jcp.nb_ic_blocking * jcp.ic_block);
                dim_t reduce_work = ocb_step * jcp.oc_block;
                for (dim_t ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    dim_t curr_nb_oc
                            = nstl::min<dim_t>(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }

                    jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv, diff_src_w,
                            diff_dst_w, wht_w, nullptr, static_cast<int>(ocb),
                            1, static_cast<int>(iwb),
                            static_cast<int>(reduce_work),
                            static_cast<int>(load_work));
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cwgn) {
                    int dummy {0};
                    nd_iterator_jump(start, end, icc, ic_chunks, iwb, jcp.nb_iw,
                            gg, nb_groups, n, jcp.mb, dummy, 1);
                } else if (jcp.loop_order == loop_gncw) {
                    int dummy {0};
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, iwb, jcp.nb_iw, dummy, 1);
                } else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, iwb, jcp.nb_iw, icc, ic_chunks,
                            gg, nb_groups);
                } else {
                    assert(!"unsupported loop order");
                }
            }
        }
    });
}

template <data_type_t diff_dst_type, data_type_t wei_type,
        data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
        diff_src_type>::execute_backward_data_2d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    dim_t ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    dim_t nb_groups = jcp.ngroups / g_blocking;
    dim_t work_amount = nb_groups * jcp.mb * ic_chunks * jcp.ih * jcp.nb_iw;
    int nthr = jcp.nthr;

    parallel(nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        dim_t start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_args_t();

        // The second arg in template means sub_offset0 = true
        // See `blk_off` method definition.
        size_t diff_src_h_stride = diff_src_d.blk_off<false, true>(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off<false, true>(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off<false, true>(0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        bool is_fast_path = jcp.dilate_h == 0 && jcp.stride_h == 1;

        for (dim_t ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            dim_t n {0}, gg {0}, icc {0}, ih_s {0}, iwb {0};

            if (jcp.loop_order == loop_cwgn) {
                nd_iterator_init(start, icc, ic_chunks, iwb, jcp.nb_iw, gg,
                        nb_groups, n, jcp.mb, ih_s, jcp.ih);
            } else if (jcp.loop_order == loop_gncw) {
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, iwb, jcp.nb_iw, ih_s, jcp.ih);
            } else if (jcp.loop_order == loop_nhwcg) {
                nd_iterator_init(start, n, jcp.mb, ih_s, jcp.ih, iwb, jcp.nb_iw,
                        icc, ic_chunks, gg, nb_groups);
            } else
                assert(!"unsupported loop order");

            while (start < end) {
                dim_t icb = icc * jcp.nb_ic_blocking;
                dim_t g = gg * g_blocking;
                dim_t g_icb = g * jcp.nb_ic + icb;
                dim_t g_ocb = g * jcp.nb_oc;

                dim_t work_rem = end - start;
                dim_t ih_e
                        = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    ih_e = ih_s + 1; //step instead
                dim_t iw_s = iwb * jcp.iw_block;
                dim_t ow_s = iw_s / jcp.stride_w;
                const bool is_dsrc_layout_nxc = jcp.src_tag == format_tag::nhwc;
                const dim_t ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w
                        = diff_src + diff_src_d.blk_off(n, ic_off_idx, 0, iw_s);
                const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nhwc;
                const dim_t oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w
                        = diff_dst + diff_dst_d.blk_off(n, oc_off_idx, 0, ow_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb);

                dim_t ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                dim_t ocb_end
                        = nstl::min<dim_t>(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const dim_t load_work
                        = utils::this_block_size(icb * jcp.ic_block, jcp.ic,
                                jcp.nb_ic_blocking * jcp.ic_block);
                dim_t reduce_work = ocb_step * jcp.oc_block;
                for (dim_t ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    dim_t curr_nb_oc
                            = nstl::min<dim_t>(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }
                    for (dim_t ij = ih_s; ij < ih_e; ++ij) {
                        dim_t oj, k_len, k_lo;
                        if (is_fast_path) { // dilate == 0 && stride == 1
                            dim_t i_t_overflow = max<dim_t>(
                                    0, jcp.kh - 1 - ij - jcp.t_pad);
                            dim_t i_b_overflow = max<dim_t>(
                                    0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else if (jcp.dilate_h != 0) { // stride == 1
                            dim_t dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            dim_t i_t_overflow
                                    = div_up(max<dim_t>(0,
                                                     (jcp.kh - 1) * dilate_h
                                                             - ij - jcp.t_pad),
                                            dilate_h);
                            dim_t i_b_overflow = div_up(
                                    max<dim_t>(0,
                                            (jcp.kh - 1) * dilate_h + 1 - jcp.ih
                                                    + ij - jcp.b_pad),
                                    dilate_h);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                        } else { // dilate == 0
                            dim_t i_t_overflow = max<dim_t>(0,
                                    (jcp.kh - 1 - ij - jcp.t_pad)
                                            / jcp.stride_h);
                            dim_t i_b_overflow = max<dim_t>(0,
                                    (jcp.kh - jcp.ih + ij - jcp.b_pad)
                                            / jcp.stride_h);
                            dim_t overflow_kh_hi = jcp.kh - 1
                                    - modulo(jcp.ih - 1 + jcp.b_pad - ij,
                                            jcp.stride_h);
                            dim_t overflow_kh_lo
                                    = (ij + jcp.t_pad) % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                            / jcp.stride_h
                                    + 1 - i_t_overflow - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }

                        jit_conv_ker_pipeline_iw_thr(jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride, nullptr,
                                static_cast<int>(ocb), static_cast<int>(k_len),
                                static_cast<int>(iwb),
                                static_cast<int>(reduce_work),
                                static_cast<int>(load_work));
                    }
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cwgn) {
                    nd_iterator_jump(start, end, icc, ic_chunks, iwb, jcp.nb_iw,
                            gg, nb_groups, n, jcp.mb, ih_s, jcp.ih);
                } else if (jcp.loop_order == loop_gncw) {
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, iwb, jcp.nb_iw, ih_s, jcp.ih);
                } else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, ih_s, jcp.ih, iwb, jcp.nb_iw,
                            icc, ic_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }
    });
}

template <data_type_t diff_dst_type, data_type_t wei_type,
        data_type_t diff_src_type>
void jit_avx512_common_convolution_bwd_data_t<diff_dst_type, wei_type,
        diff_src_type>::execute_backward_data_3d(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = pd()->jcp_;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();

    dim_t ic_chunks = jcp.nb_ic / jcp.nb_ic_blocking;
    int g_blocking = 1;
    dim_t nb_groups = jcp.ngroups / g_blocking;
    dim_t work_amount = nb_groups * jcp.mb * ic_chunks * jcp.id * jcp.ih;
    int nthr = jcp.nthr;

    parallel(nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        dim_t start {0}, end {0}, start_copy;
        balance211(work_amount, nthr, ithr, start, end);
        start_copy = start;

        auto par_conv = jit_conv_args_t();

        // The second arg in template means sub_offset0 = true
        // See `blk_off` method definition.
        size_t diff_src_h_stride = diff_src_d.blk_off<false, true>(0, 0, 0, 1);
        size_t diff_src_d_stride = diff_src_d.blk_off<false, true>(0, 0, 1);
        size_t diff_dst_h_stride = diff_dst_d.blk_off<false, true>(0, 0, 0, 1);
        size_t diff_dst_d_stride = diff_dst_d.blk_off<false, true>(0, 0, 1);
        size_t diff_dst_c_stride = diff_dst_d.blk_off<false, true>(0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 0, 1);
        size_t wht_d_stride = wht_blk_off(weights_d, 0, 0, 0, 1);
        size_t wht_oc_stride = wht_blk_off(weights_d, 0, 1);

        bool is_fast_path_d = jcp.dilate_d == 0 && jcp.stride_d == 1;
        bool is_fast_path_h = jcp.dilate_h == 0 && jcp.stride_h == 1;

        for (dim_t ocb_l2 = 0; ocb_l2 < jcp.nb_oc; ocb_l2 += jcp.nb_oc_L2) {
            start = start_copy;
            dim_t n {0}, gg {0}, icc {0}, ih_s {0}, id_s {0};
            // Input width threading is not currently implemented for 3d, so it
            // is not included in the iterator.
            if (jcp.loop_order == loop_cwgn)
                nd_iterator_init(start, icc, ic_chunks, gg, nb_groups, n,
                        jcp.mb, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_gncw)
                nd_iterator_init(start, gg, nb_groups, n, jcp.mb, icc,
                        ic_chunks, id_s, jcp.id, ih_s, jcp.ih);
            else if (jcp.loop_order == loop_nhwcg)
                nd_iterator_init(start, n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih,
                        icc, ic_chunks, gg, nb_groups);
            else
                assert(!"unsupported loop order");

            while (start < end) {
                dim_t icb = icc * jcp.nb_ic_blocking;
                dim_t g = gg * g_blocking;
                dim_t g_icb = g * jcp.nb_ic + icb;
                dim_t g_ocb = g * jcp.nb_oc;

                dim_t work_rem = end - start;
                dim_t ih_e
                        = ih_s + work_rem > jcp.ih ? jcp.ih : ih_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    ih_e = ih_s + 1; //step instead
                dim_t d_len = 0;
                dim_t d_lo = 0, d_oj = 0;
                if (is_fast_path_d) { // dilate == 0 && stride == 1
                    dim_t d_t_overflow
                            = max<dim_t>(0, jcp.kd - 1 - id_s - jcp.f_pad);
                    dim_t d_b_overflow = max<dim_t>(
                            0, jcp.kd - jcp.id + id_s - jcp.back_pad);
                    d_len = jcp.kd - d_t_overflow - d_b_overflow;
                    d_lo = d_b_overflow;
                    d_oj = id_s + jcp.f_pad - d_b_overflow;
                } else if (jcp.dilate_d != 0) { // stride == 1
                    dim_t dilate_d = jcp.dilate_d + 1;
                    // Note: use div_up to account for "holes" in filter
                    dim_t d_t_overflow = div_up(
                            max<dim_t>(0,
                                    (jcp.kd - 1) * dilate_d - id_s - jcp.f_pad),
                            dilate_d);
                    dim_t d_b_overflow = div_up(
                            max<dim_t>(0,
                                    (jcp.kd - 1) * dilate_d + 1 - jcp.id + id_s
                                            - jcp.back_pad),
                            dilate_d);
                    d_len = jcp.kd - d_t_overflow - d_b_overflow;
                    d_lo = d_b_overflow;
                    d_oj = id_s + jcp.f_pad - d_b_overflow * dilate_d;
                } else { // dilate == 0
                    dim_t d_t_overflow = max<dim_t>(
                            0, (jcp.kd - 1 - id_s - jcp.f_pad) / jcp.stride_d);
                    dim_t d_b_overflow = max<dim_t>(0,
                            (jcp.kd - jcp.id + id_s - jcp.back_pad)
                                    / jcp.stride_d);
                    dim_t overflow_kd_hi = jcp.kd - 1
                            - modulo(jcp.id - 1 + jcp.back_pad - id_s,
                                    jcp.stride_d);
                    dim_t overflow_kd_lo = (id_s + jcp.f_pad) % jcp.stride_d;

                    d_len = (overflow_kd_hi - overflow_kd_lo) / jcp.stride_d + 1
                            - d_t_overflow - d_b_overflow;
                    d_lo = overflow_kd_lo + d_b_overflow * jcp.stride_d;
                    d_oj = (id_s + jcp.f_pad - d_lo) / jcp.stride_d;
                }

                const bool is_dsrc_layout_nxc
                        = jcp.src_tag == format_tag::ndhwc;
                const dim_t ic_off_idx = is_dsrc_layout_nxc
                        ? g * jcp.ic + icb * jcp.ic_block
                        : g_icb;
                auto diff_src_w = diff_src + diff_src_d.blk_off(n, ic_off_idx)
                        + id_s * diff_src_d_stride;
                const bool is_ddst_layout_nxc
                        = jcp.dst_tag == format_tag::ndhwc;
                const dim_t oc_off_idx = is_ddst_layout_nxc
                        ? g * jcp.oc + ocb_l2 * jcp.oc_block
                        : g_ocb + ocb_l2;
                auto diff_dst_w = diff_dst + diff_dst_d.blk_off(n, oc_off_idx)
                        + d_oj * diff_dst_d_stride;
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb_l2, icb)
                        + d_lo * wht_d_stride;

                dim_t ocb_step = is_ddst_layout_nxc ? jcp.nb_oc_L2 : 1;
                dim_t ocb_end
                        = nstl::min<dim_t>(jcp.nb_oc, ocb_l2 + jcp.nb_oc_L2);
                const dim_t load_work
                        = utils::this_block_size(icb * jcp.ic_block, jcp.ic,
                                jcp.nb_ic_blocking * jcp.ic_block);
                dim_t reduce_work = ocb_step * jcp.oc_block;
                for (dim_t ocb = ocb_l2; ocb < ocb_end; ocb += ocb_step) {
                    dim_t curr_nb_oc
                            = nstl::min<dim_t>(ocb_step, ocb_end - ocb);
                    if (ocb + curr_nb_oc >= jcp.nb_oc) {
                        reduce_work = utils::this_block_size(ocb * jcp.oc_block,
                                jcp.oc, ocb_step * jcp.oc_block);
                    }
                    for (dim_t ij = ih_s; ij < ih_e; ++ij) {
                        dim_t oj, k_lo;
                        dim_t k_len;
                        if (is_fast_path_h) { // dilate == 0 && stride == 1
                            dim_t i_t_overflow = max<dim_t>(
                                    0, jcp.kh - 1 - ij - jcp.t_pad);
                            dim_t i_b_overflow = max<dim_t>(
                                    0, jcp.kh - jcp.ih + ij - jcp.b_pad);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow;
                        } else if (jcp.dilate_h != 0) { // stride == 1
                            dim_t dilate_h = jcp.dilate_h + 1;
                            // Note: use div_up to account for "holes" in filter
                            dim_t i_t_overflow
                                    = div_up(max<dim_t>(0,
                                                     (jcp.kh - 1) * dilate_h
                                                             - ij - jcp.t_pad),
                                            dilate_h);
                            dim_t i_b_overflow = div_up(
                                    max<dim_t>(0,
                                            (jcp.kh - 1) * dilate_h + 1 - jcp.ih
                                                    + ij - jcp.b_pad),
                                    dilate_h);
                            k_len = jcp.kh - i_t_overflow - i_b_overflow;
                            k_lo = i_b_overflow;
                            oj = ij + jcp.t_pad - i_b_overflow * dilate_h;
                        } else { // dilate == 0
                            dim_t i_t_overflow = max<dim_t>(0,
                                    (jcp.kh - 1 - ij - jcp.t_pad)
                                            / jcp.stride_h);
                            dim_t i_b_overflow = max<dim_t>(0,
                                    (jcp.kh - jcp.ih + ij - jcp.b_pad)
                                            / jcp.stride_h);
                            dim_t overflow_kh_hi = jcp.kh - 1
                                    - modulo(jcp.ih - 1 + jcp.b_pad - ij,
                                            jcp.stride_h);
                            dim_t overflow_kh_lo
                                    = (ij + jcp.t_pad) % jcp.stride_h;

                            k_len = (overflow_kh_hi - overflow_kh_lo)
                                            / jcp.stride_h
                                    + 1 - i_t_overflow - i_b_overflow;
                            k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                            oj = (ij + jcp.t_pad - k_lo) / jcp.stride_h;
                        }
                        assert(k_len >= 0);

                        jit_conv_3d_ker_pipeline(jit_ker, par_conv,
                                diff_src_w + ij * diff_src_h_stride,
                                diff_dst_w + oj * diff_dst_h_stride,
                                wht_w + k_lo * wht_h_stride, nullptr,
                                static_cast<int>(ocb), static_cast<int>(k_len),
                                static_cast<int>(d_len),
                                static_cast<int>(reduce_work),
                                static_cast<int>(load_work));
                    }
                    diff_dst_w += diff_dst_c_stride;
                    wht_w += wht_oc_stride;
                }

                if (jcp.loop_order == loop_cwgn)
                    nd_iterator_jump(start, end, icc, ic_chunks, gg, nb_groups,
                            n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih);
                else if (jcp.loop_order == loop_gncw)
                    nd_iterator_jump(start, end, gg, nb_groups, n, jcp.mb, icc,
                            ic_chunks, id_s, jcp.id, ih_s, jcp.ih);
                else if (jcp.loop_order == loop_nhwcg) {
                    ++start;
                    nd_iterator_step(n, jcp.mb, id_s, jcp.id, ih_s, jcp.ih, icc,
                            ic_chunks, gg, nb_groups);
                } else
                    assert(!"unsupported loop order");
            }
        }
    });
}

template struct jit_avx512_common_convolution_bwd_data_t<data_type::f32>;

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
status_t jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::init(engine_t *engine) {
    const auto &j = pd()->jcp_;

    nthr_ = j.nthr;
    nthr_mb_ = j.nthr_mb;
    nthr_g_ = j.nthr_g;
    nthr_oc_b_ = j.nthr_oc_b;
    nthr_ic_b_ = j.nthr_ic_b;

    CHECK(safe_ptr_assign(
            kernel_, new jit_avx512_common_conv_bwd_weights_kernel_f32_t(j)));
    CHECK(kernel_->create_kernel());

    if (nthr_mb_ > 1) {
        CHECK(safe_ptr_assign(
                acc_ker_, new cpu_accumulator_1d_t<diff_weights_type>()));
        CHECK(acc_ker_->create_kernel());
    }

    CHECK(safe_ptr_assign(reducer_bias_,
            new cpu_reducer_t<diff_weights_type>(pd()->reducer_bia_conf_)));
    CHECK(reducer_bias_->create_kernel());
    return status::success;
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
struct jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::thread_info_t {
    const src_data_t *src;
    const diff_dst_data_t *diff_dst;
    const diff_weights_data_t *diff_weights;
    diff_weights_data_t *diff_bias;

    const memory_tracking::grantor_t &scratchpad;

    src_data_t *tr_src;
    diff_dst_data_t *tr_diff_dst;

    diff_weights_data_t *wei_bia_reduction;
    simple_barrier::ctx_t *wei_bia_reduction_bctx;

    int ithr;
    int ithr_ic_b, ithr_oc_b, ithr_g, ithr_mb;
    int ithr_but_oc;
    int ithr_but_ic;

    dim_t img_work, img_start = 0, img_end = 0;
    dim_t g_work, g_start = 0, g_end = 0;
    dim_t oc_b_work, oc_b_start = 0, oc_b_end = 0;
    dim_t ic_b_work, ic_b_start = 0, ic_b_end = 0;

    thread_info_t(const jit_avx512_common_convolution_bwd_weights_t *self,
            const exec_ctx_t &ctx, int ithr)
        : src(CTX_IN_MEM(const src_data_t *, DNNL_ARG_SRC))
        , diff_dst(CTX_IN_MEM(const diff_dst_data_t *, DNNL_ARG_DIFF_DST))
        , diff_weights(
                  CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_WEIGHTS))
        , scratchpad(ctx.get_scratchpad_grantor())
        , tr_src(scratchpad.template get<src_data_t>(key_conv_tr_src))
        , tr_diff_dst(scratchpad.template get<diff_dst_data_t>(
                  key_conv_tr_diff_dst))
        , wei_bia_reduction(scratchpad.template get<diff_weights_data_t>(
                  key_conv_wei_bia_reduction))
        , ithr(ithr)
        , ithr_ic_b(ithr % self->nthr_ic_b_)
        , ithr_oc_b(ithr / self->nthr_ic_b_ % self->nthr_oc_b_)
        , ithr_g(ithr / self->nthr_ic_b_ / self->nthr_oc_b_ % self->nthr_g_)
        , ithr_mb(ithr / self->nthr_ic_b_ / self->nthr_oc_b_ / self->nthr_g_)
        , ithr_but_oc((ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_ic_b_
                  + ithr_ic_b)
        , ithr_but_ic((ithr_mb * self->nthr_g_ + ithr_g) * self->nthr_oc_b_
                  + ithr_oc_b) {

        const auto &jcp = self->kernel_->jcp;
        const bool is_bias_padded = self->pd()->with_bias()
                && jcp.oc_without_padding % jcp.oc_block != 0;
        diff_bias = is_bias_padded
                ? scratchpad.template get<diff_weights_data_t>(
                          key_conv_padded_bias)
                : CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_BIAS);

        wei_bia_reduction_bctx = scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx);

        /* reduction dimension */
        int oh_reduce = static_cast<int>(
                jcp.harness == harness_2d_reduction ? jcp.oh : 1);
        balance211(jcp.mb * jcp.od * oh_reduce, self->nthr_mb_, ithr_mb,
                img_start, img_end);
        img_work = img_end - img_start;

        /* independent dimensions */
        balance211(jcp.ngroups, self->nthr_g_, ithr_g, g_start, g_end);
        g_work = g_end - g_start;

        balance211(
                jcp.nb_oc, self->nthr_oc_b_, ithr_oc_b, oc_b_start, oc_b_end);
        oc_b_work = oc_b_end - oc_b_start;

        balance211(
                jcp.nb_ic, self->nthr_ic_b_, ithr_ic_b, ic_b_start, ic_b_end);
        ic_b_work = ic_b_end - ic_b_start;
    }
};

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_weights_nxc(const thread_info_t *ti)
        const {
    const auto &jcp = kernel_->jcp;

    const dim_t wei_size
            = jcp.ngroups * jcp.oc * jcp.ic * jcp.kh * jcp.kw * jcp.kd;
    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    auto diff_weights_offset = [&](dim_t g, dim_t i_kd, dim_t i_kh, dim_t i_kw,
                                       dim_t i_ic, dim_t i_oc) {
        const dim_t oc_block_size = 1;
        const dim_t ic_block_size = jcp.oc_block * oc_block_size;
        const dim_t kw_block_size = jcp.ic_block * ic_block_size;
        const dim_t kh_block_size = jcp.kw * kw_block_size;
        const dim_t kd_block_size = jcp.kh * kh_block_size;
        const dim_t icb_block_size = jcp.kd * kd_block_size;
        const dim_t ocb_block_size = jcp.nb_ic * icb_block_size;
        const dim_t g_block_size = jcp.nb_oc * ocb_block_size;

        dim_t icb = i_ic / jcp.ic_block;
        dim_t ocb = i_oc / jcp.oc_block;
        i_ic = i_ic % jcp.ic_block;
        i_oc = i_oc % jcp.oc_block;

        return g * g_block_size + ocb * ocb_block_size + icb * icb_block_size
                + i_kd * kd_block_size + i_kh * kh_block_size
                + i_kw * kw_block_size + i_ic * ic_block_size
                + i_oc * oc_block_size;
    };
    auto src_offset = [&](dim_t g, dim_t i_mb, dim_t i_id, dim_t i_ih,
                              dim_t i_ic, dim_t i_iw) {
        const dim_t ic_block_size = 1;
        const dim_t g_block_size = jcp.ic * ic_block_size;
        const dim_t iw_block_size = jcp.ngroups * g_block_size;
        const dim_t ih_block_size = jcp.iw * iw_block_size;
        const dim_t id_block_size = jcp.ih * ih_block_size;
        const dim_t mb_block_size = jcp.id * id_block_size;

        return g * g_block_size + i_mb * mb_block_size + i_id * id_block_size
                + i_ih * ih_block_size + i_iw * iw_block_size
                + i_ic * ic_block_size;
    };
    auto diff_dst_offset = [&](dim_t g, dim_t i_mb, dim_t i_od, dim_t i_oh,
                                   dim_t i_ow, dim_t i_oc) {
        const dim_t oc_block_size = 1;
        const dim_t g_block_size = jcp.oc * oc_block_size;
        const dim_t ow_block_size = jcp.ngroups * g_block_size;
        const dim_t oh_block_size = jcp.ow * ow_block_size;
        const dim_t od_block_size = jcp.oh * oh_block_size;
        const dim_t mb_block_size = jcp.od * od_block_size;

        return g * g_block_size + i_mb * mb_block_size + i_od * od_block_size
                + i_oh * oh_block_size + i_ow * ow_block_size
                + i_oc * oc_block_size;
    };
    auto zero_diff_weights = [&]() {
        PRAGMA_OMP_SIMD()
        for (dim_t i = 0; i < wei_size; i++)
            diff_wei[i] = 0;
    };

    dim_t kd_step = jcp.dilate_d + 1;
    dim_t kh_step = jcp.dilate_h + 1;
    dim_t stride_d = jcp.stride_d;
    dim_t stride_h = jcp.stride_h;
    dim_t f_pad = jcp.f_pad;
    dim_t t_pad = jcp.t_pad;

    dim_t work_amount
            = static_cast<dim_t>(jcp.mb) * jcp.od * jcp.oh * jcp.nb_ow;
    dim_t i_work {0}, i_work_end {0};
    balance211(work_amount, jcp.nthr_mb, ti->ithr_mb, i_work, i_work_end);

    dim_t i_mb {0}, i_od {0}, i_oh {0}, i_owb {0};
    nd_iterator_init(
            i_work, i_mb, jcp.mb, i_od, jcp.od, i_oh, jcp.oh, i_owb, jcp.nb_ow);

    zero_diff_weights();
    while (i_work < i_work_end) {
        dim_t kd_start = div_up(
                nstl::max<dim_t>(0, jcp.f_pad - jcp.stride_d * i_od), kd_step);
        dim_t kd_end = nstl::min<dim_t>(
                jcp.kd - 1, (jcp.id - 1 + f_pad - stride_d * i_od) / kd_step);
        dim_t i_id_base = stride_d * i_od - f_pad;
        dim_t kh_start = div_up(
                nstl::max<dim_t>(0, jcp.t_pad - jcp.stride_h * i_oh), kh_step);
        dim_t kh_end = nstl::min<dim_t>(
                jcp.kh - 1, (jcp.ih - 1 + t_pad - stride_h * i_oh) / kh_step);
        dim_t i_ih_base = jcp.stride_h * i_oh + -jcp.t_pad;
        dim_t i_ow_base = i_owb * jcp.ow_block;
        dim_t i_ow_end = nstl::min<dim_t>(jcp.ow, i_ow_base + jcp.ow_block);

        // The kernel is small so these loops produce measurable overhead. Since
        // these are simple loops, the compiler will likely make the loops just
        // as well as we can with the jitted assembly, so there is not
        // necessarily a reason to move these loops into assembly. Avoid placing
        // computationally heavy operations within the loops.
        for_(dim_t i_ow = i_ow_base; i_ow < i_ow_end; i_ow += jcp.ur_ow)
        for_(dim_t i_oc = 0; i_oc < jcp.oc; i_oc += jcp.oc_block)
        for_(dim_t g = 0; g < jcp.ngroups; g++)
        for_(dim_t i_kd = kd_start; i_kd <= kd_end; i_kd++)
        for (dim_t i_kh = kh_start; i_kh <= kh_end; i_kh++) {
            // Some Optimization Observations: It may be
            // worthwhile to move the kd and kh loops below the
            // icb loop in the kernel to further amortize the
            // ddst register loads. Alternatively, these
            // dimensions are independent on the weights kernel,
            // so can be used as a threading dimension that does
            // not require reduction.

            // The compiler seems to do a good job at optimizing these
            // computations. The offset functions likely need to be located
            // so that they will be inlined.
            dim_t i_iw = i_ow * jcp.stride_w - jcp.l_pad;
            dim_t i_id = i_id_base + i_kd * kd_step;
            dim_t i_ih = i_ih_base + i_kh * kh_step;
            dim_t ddst_offset
                    = diff_dst_offset(g, i_mb, i_od, i_oh, i_ow, i_oc);
            dim_t s_off_base = src_offset(g, i_mb, i_id, i_ih, 0, i_iw);
            dim_t dwei_off_base
                    = diff_weights_offset(g, i_kd, i_kh, 0, 0, i_oc);
            // ensure all parameters are 64bit, to comply with windows kernel
            // param access where the params from 5th are passed using stack.
            (*kernel_)(&diff_wei[dwei_off_base], &ti->src[s_off_base],
                    &ti->diff_dst[ddst_offset], i_iw, i_ow);
        }
        nd_iterator_step(
                i_mb, jcp.mb, i_od, jcp.od, i_oh, jcp.oh, i_owb, jcp.nb_ow);
        i_work++;
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_weights(const thread_info_t *ti)
        const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    const dim_t padded_oc = rnd_up(jcp.oc, jcp.oc_block);
    const dim_t wei_size = jcp.ngroups * padded_oc
            * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw * jcp.kd;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;

    const bool is_src_layout_nxc = utils::one_of(
            jcp.src_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);

    dim_t ic_b_step = jcp.nb_ic_blocking_max;
    dim_t icb_work = ti->ic_b_end - ti->ic_b_start;
    if (ic_b_step > 1 && icb_work > ic_b_step && icb_work < 2 * ic_b_step)
        ic_b_step = utils::div_up(icb_work, 2);

    for (dim_t img = ti->img_start; img < ti->img_end; ++img) {
        auto p = jit_conv_args_t();

        const dim_t max_oc
                = nstl::min<dim_t>(ti->oc_b_end * jcp.oc_block, jcp.oc);
        const dim_t max_ic
                = nstl::min<dim_t>(ti->ic_b_end * jcp.ic_block, jcp.ic);
        const bool is_ddst_layout_nxc = utils::one_of(jcp.dst_tag,
                format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
        for_(dim_t g = ti->g_start; g < ti->g_end; ++g)
        for_(dim_t oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (dim_t ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += ic_b_step) {
            const dim_t _oc = g * jcp.nb_oc + oc_b;
            const dim_t _ic = g * jcp.nb_ic + ic_b;
            const dim_t ic_to_compute = this_block_size(
                    ic_b * jcp.ic_block, max_ic, ic_b_step * jcp.ic_block);
            const dim_t oc_to_compute = this_block_size(
                    oc_b * jcp.oc_block, max_oc, jcp.oc_block);

            const dim_t ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : _ic;
            const dim_t oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : _oc;

            jit_conv_ker_pipeline_bwd_w(jit_ker, p,
                    &ti->src[src_d.blk_off(img, ic_off_idx)],
                    &ti->diff_dst[diff_dst_d.blk_off(img, oc_off_idx)],
                    diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                    nullptr, (img == ti->img_start), 0, ic_to_compute,
                    oc_to_compute);
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_weights_2d(const thread_info_t *ti)
        const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    const dim_t padded_oc = rnd_up(jcp.oc, jcp.oc_block);
    const dim_t wei_size = jcp.ngroups * padded_oc
            * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;
    diff_weights_data_t *diff_bia = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_bias
            : ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size
                    + (ti->ithr_mb - 1) * jcp.ngroups * padded_oc;

    dim_t img {0}, oh_s {0};
    dim_t img_start = ti->img_start, img_end = ti->img_end;
    nd_iterator_init(img_start, img, jcp.mb, oh_s, jcp.oh);
    const dim_t img_first = img;

    dim_t ic_b_step = jcp.nb_ic_blocking_max;
    dim_t icb_work = ti->ic_b_end - ti->ic_b_start;
    if (ic_b_step > 1 && icb_work > ic_b_step && icb_work < 2 * ic_b_step)
        ic_b_step = utils::div_up(icb_work, 2);
    while (img_start < img_end) {
        auto p = jit_conv_args_t();

        dim_t work_rem = img_end - img_start;
        const dim_t oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
        const dim_t ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
        const dim_t kh_top_overflow = nstl::max<dim_t>(0, -ih_s);
        const dim_t kh_bottom_overflow
                = nstl::max<dim_t>(0, ih_s - jcp.ih + jcp.kh);
        dim_t kh_padding = jcp.kh - kh_top_overflow - kh_bottom_overflow;
        dim_t kh_padding_offset = nstl::min<dim_t>(jcp.kh - 1, kh_top_overflow)
                * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.typesize_out;
        auto src_h = ti->src + src_d.blk_off(img, 0, ih_s + kh_top_overflow);
        auto diff_dst_h = ti->diff_dst + diff_dst_d.blk_off(img, 0, oh_s);

        const bool is_src_layout_nxc = jcp.src_tag == format_tag::nhwc;
        const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::nhwc;
        const dim_t max_oc
                = nstl::min<dim_t>(ti->oc_b_end * jcp.oc_block, jcp.oc);
        const dim_t max_ic
                = nstl::min<dim_t>(ti->ic_b_end * jcp.ic_block, jcp.ic);
        for_(dim_t g = ti->g_start; g < ti->g_end; ++g)
        for_(dim_t oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (dim_t ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += ic_b_step) {
            const dim_t _oc = g * jcp.nb_oc + oc_b;
            const dim_t _ic = g * jcp.nb_ic + ic_b;
            const dim_t ic_to_compute = this_block_size(
                    ic_b * jcp.ic_block, max_ic, ic_b_step * jcp.ic_block);
            const dim_t oc_to_compute = this_block_size(
                    oc_b * jcp.oc_block, max_oc, jcp.oc_block);
            const dim_t ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : _ic;
            const dim_t oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : _oc;
            auto src = src_h + src_d.blk_off(0, ic_off_idx);
            auto diff_dst = diff_dst_h + diff_dst_d.blk_off(0, oc_off_idx);
            p.flags = ic_b == 0 ? 0 : 1;
            jit_conv_2d_ker_bwd_w_pipeline(jit_ker, p, src, diff_dst,
                    diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                    diff_bia + _oc * jcp.oc_block, (img == img_first),
                    static_cast<int>(oh_s), static_cast<int>(oh_e),
                    static_cast<int>(kh_padding), kh_padding_offset,
                    ic_to_compute, oc_to_compute);
        }
        nd_iterator_jump(img_start, img_end, img, jcp.mb, oh_s, jcp.oh);
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_weights_3d(const thread_info_t *ti)
        const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const jit_conv_ker_t jit_ker = (decltype(jit_ker))kernel_->jit_ker();
    const dim_t padded_oc = rnd_up(jcp.oc, jcp.oc_block);
    const dim_t wei_size = jcp.ngroups * padded_oc
            * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw * jcp.kd;

    diff_weights_data_t *diff_wei = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_weights
            : ti->wei_bia_reduction + (ti->ithr_mb - 1) * wei_size;
    diff_weights_data_t *diff_bia = ti->ithr_mb == 0
            ? (diff_weights_data_t *)ti->diff_bias
            : ti->wei_bia_reduction + (nthr_mb_ - 1) * wei_size
                    + (ti->ithr_mb - 1) * jcp.ngroups * padded_oc;

    const bool is_src_layout_nxc = jcp.src_tag == format_tag::ndhwc;
    const dim_t inp_mult = is_src_layout_nxc
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : jcp.ic_block);
    const dim_t input_step = jcp.ih * jcp.iw * inp_mult;
    const bool is_ddst_layout_nxc = jcp.dst_tag == format_tag::ndhwc;
    const dim_t output_step = jcp.ow * jcp.oh
            * (is_ddst_layout_nxc ? jcp.ngroups * jcp.oc : jcp.oc_block);
    dim_t img {0}, od_s {0};
    dim_t img_start = ti->img_start, img_end = ti->img_end;
    nd_iterator_init(img_start, img, jcp.mb, od_s, jcp.od);
    const dim_t img_first = img;

    dim_t ic_b_step = jcp.nb_ic_blocking_max;
    dim_t icb_work = ti->ic_b_end - ti->ic_b_start;
    if (ic_b_step > 1 && icb_work > ic_b_step && icb_work < 2 * ic_b_step)
        ic_b_step = utils::div_up(icb_work, 2);

    while (img_start < img_end) {
        auto p = jit_conv_args_t();

        dim_t work_rem = img_end - img_start;
        const dim_t od_e = od_s + work_rem > jcp.od ? jcp.od : od_s + work_rem;
        const dim_t id_s = od_s * jcp.stride_d;
        const dim_t ik_overlap = nstl::max<dim_t>(0, id_s - jcp.f_pad);
        const dim_t kd_front_pad = nstl::max<dim_t>(0, jcp.f_pad - id_s);
        const dim_t kd_back_pad
                = nstl::max<dim_t>(0, id_s - jcp.f_pad - jcp.id + jcp.kd);
        dim_t kd_pad_off = nstl::min<dim_t>(jcp.kd - 1, kd_front_pad) * jcp.kh
                * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.typesize_out;

        const dim_t max_oc
                = nstl::min<dim_t>(ti->oc_b_end * jcp.oc_block, jcp.oc);
        const dim_t max_ic
                = nstl::min<dim_t>(ti->ic_b_end * jcp.ic_block, jcp.ic);

        for_(dim_t g = ti->g_start; g < ti->g_end; ++g)
        for_(dim_t oc_b = ti->oc_b_start; oc_b < ti->oc_b_end; ++oc_b)
        for (dim_t ic_b = ti->ic_b_start; ic_b < ti->ic_b_end;
                ic_b += ic_b_step) {
            const dim_t _oc = g * jcp.nb_oc + oc_b;
            const dim_t _ic = g * jcp.nb_ic + ic_b;

            const dim_t ic_to_compute = this_block_size(
                    ic_b * jcp.ic_block, max_ic, ic_b_step * jcp.ic_block);
            const dim_t oc_to_compute = this_block_size(
                    oc_b * jcp.oc_block, max_oc, jcp.oc_block);

            const dim_t ic_off_idx = is_src_layout_nxc
                    ? g * jcp.ic + ic_b * jcp.ic_block
                    : _ic;
            const dim_t oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + oc_b * jcp.oc_block
                    : _oc;
            auto src = &ti->src[src_d.blk_off(img, ic_off_idx)
                    + ik_overlap * input_step];
            auto dst = &ti->diff_dst[diff_dst_d.blk_off(img, oc_off_idx)
                    + od_s * output_step];
            auto diff_bia_ptr = diff_bia ? diff_bia + _oc * 16 : nullptr;
            p.flags = ic_b == 0 ? 0 : 1;
            jit_conv_3d_ker_bwd_w_pipeline(jit_ker, p, src, dst,
                    diff_wei + wht_blk_off(diff_weights_d, g, oc_b, ic_b),
                    diff_bia_ptr, (img == img_first), static_cast<int>(od_s),
                    static_cast<int>(od_e),
                    static_cast<int>(jcp.kd - kd_front_pad - kd_back_pad),
                    kd_pad_off, ic_to_compute, oc_to_compute);
        }
        nd_iterator_jump(img_start, img_end, img, jcp.mb, od_s, jcp.od);
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::reduce_diff_weights(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const dim_t padded_oc = rnd_up(jcp.oc, jcp.oc_block);
    const dim_t wei_size = jcp.ngroups * padded_oc
            * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw;

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    if (dnnl_thr_syncable())
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const dim_t ic_b_kh_work = ti->ic_b_work * jcp.kh;
    const dim_t work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int bal_start {0}, bal_end {0};
    balance211(
            static_cast<int>(work), nthr_mb_, ti->ithr_mb, bal_start, bal_end);
    if (bal_start == bal_end) return;

    for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
        dim_t w = bal_start, end = bal_end;
        dim_t sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_kh_start {0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const dim_t g = ti->g_start + sub_g_start;
            const dim_t oc_b = ti->oc_b_start + sub_oc_b_start;
            const dim_t ic_b = ti->ic_b_start + sub_ic_b_kh_start / jcp.kh;
            const dim_t kh = sub_ic_b_kh_start % jcp.kh;

            const dim_t acc_size = nstl::min<dim_t>(end - w,
                                           ic_b_kh_work - sub_ic_b_kh_start)
                    * jcp.kw * jcp.ic_block * jcp.oc_block;

            const size_t off = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kh);

            diff_weights_data_t *d
                    = (diff_weights_data_t *)ti->diff_weights + off;
            diff_weights_data_t *s
                    = ti->wei_bia_reduction + (thr_mb - 1) * wei_size + off;

            acc_ker_->accumulate(d, s, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::reduce_diff_weights_3d(const thread_info_t *ti)
        const {
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    const auto &jcp = kernel_->jcp;
    const dim_t wei_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block)
            * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw * jcp.kd;

    /* diff_weights[:] += sum(wei_reduction_[thr_mb][:]) */
    if (dnnl_thr_syncable())
        simple_barrier::barrier(ti->wei_bia_reduction_bctx, nthr_);

    const dim_t ic_b_kh_work = ti->ic_b_work * jcp.kd;
    const dim_t work = ti->g_work * ti->oc_b_work * ic_b_kh_work;

    int bal_start {0}, bal_end {0};
    balance211(
            static_cast<int>(work), nthr_mb_, ti->ithr_mb, bal_start, bal_end);
    if (bal_start == bal_end) return;

    for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
        dim_t w = bal_start, end = bal_end;
        dim_t sub_g_start {0}, sub_oc_b_start {0}, sub_ic_b_kh_start {0};
        nd_iterator_init(w, sub_g_start, ti->g_work, sub_oc_b_start,
                ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        while (w < end) {
            const dim_t g = ti->g_start + sub_g_start;
            const dim_t oc_b = ti->oc_b_start + sub_oc_b_start;
            const dim_t ic_b = ti->ic_b_start + sub_ic_b_kh_start / jcp.kd;
            const dim_t kd = sub_ic_b_kh_start % jcp.kd;

            const dim_t acc_size = nstl::min<dim_t>(end - w,
                                           ic_b_kh_work - sub_ic_b_kh_start)
                    * jcp.kw * jcp.ic_block * jcp.oc_block * jcp.kh;

            const size_t off = wht_blk_off(diff_weights_d, g, oc_b, ic_b, kd);
            diff_weights_data_t *d
                    = (diff_weights_data_t *)ti->diff_weights + off;
            diff_weights_data_t *s
                    = ti->wei_bia_reduction + (thr_mb - 1) * wei_size + off;
            acc_ker_->accumulate(d, s, acc_size);

            nd_iterator_jump(w, end, sub_g_start, ti->g_work, sub_oc_b_start,
                    ti->oc_b_work, sub_ic_b_kh_start, ic_b_kh_work);
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::compute_diff_bias(const thread_info_t *ti) const {
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());

    auto rb = this->reducer_bias_.get();
    assert(nthr_ == rb->balancer().nthr_);

    memory_tracking::grantor_t reducer_bia_scratchpad(
            ti->scratchpad, prefix_reducer_bia);

    const auto &jcp = kernel_->jcp;

    const int b_job_start = rb->balancer().ithr_job_off(ti->ithr);
    const int b_njobs = rb->balancer().ithr_njobs(ti->ithr);

    if (b_njobs == 0) return;

    /* reduction dimension */
    dim_t img_start {0}, img_end {0};
    balance211(jcp.mb, rb->balancer().nthr_per_group_,
            rb->balancer().id_in_group(ti->ithr), img_start, img_end);

    /* jobs */
    int g_start {0}, ocb_start {0};
    nd_iterator_init(b_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_oc);
    for (dim_t img = img_start; img < img_end; ++img) {
        int g = g_start, ocb = ocb_start;
        for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
            const dim_t _oc = g * jcp.nb_oc + ocb;
            const dim_t max_oc
                    = this_block_size(ocb * jcp.oc_block, jcp.oc, jcp.oc_block);

            const bool is_ddst_layout_nxc = utils::one_of(jcp.dst_tag,
                    format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
            const dim_t oc_off_idx = is_ddst_layout_nxc
                    ? g * jcp.oc + ocb * jcp.oc_block
                    : _oc;
            const diff_dst_data_t *d_dst
                    = &ti->diff_dst[diff_dst_d.blk_off(img, oc_off_idx)];
            diff_weights_data_t *d_bias
                    = rb->get_local_ptr(
                              ti->ithr, ti->diff_bias, reducer_bia_scratchpad)
                    + b_job_loc * rb->balancer().job_size_;

            if (img == img_start)
                for (int o = 0; o < jcp.oc_block; ++o)
                    d_bias[o] = 0;
            for (int hw = 0; hw < jcp.oh * jcp.ow * jcp.od; ++hw) {
                PRAGMA_OMP_SIMD()
                for (dim_t o = 0; o < max_oc; ++o)
                    d_bias[o] += d_dst[o];
                d_dst += is_ddst_layout_nxc ? jcp.ngroups * jcp.oc
                                            : jcp.oc_block;
            }

            nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc);
        }
    }

    if (dnnl_thr_syncable())
        rb->reduce(ti->ithr, ti->diff_bias, reducer_bia_scratchpad);
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::reduce_diff_bias(const thread_info_t *ti) const {
    const auto &jcp = kernel_->jcp;

    const size_t wei_size = (size_t)jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block)
            * rnd_up(jcp.ic, jcp.ic_block) * jcp.kh * jcp.kw * jcp.kd;
    const dim_t bia_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
    const diff_weights_data_t *diff_bias_ws
            = ti->wei_bia_reduction + (size_t)(nthr_mb_ - 1) * wei_size;

    if (dnnl_thr_syncable() && nthr_mb_ > 1) dnnl_thr_barrier();

    if (ti->ithr == 0) {
        for (int thr_mb = 1; thr_mb < nthr_mb_; ++thr_mb) {
            acc_ker_->accumulate(ti->diff_bias, diff_bias_ws, bia_size);
            diff_bias_ws += bia_size;
        }
    }
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::prepare_scratchpad_data(const exec_ctx_t &ctx)
        const {
    const auto &scratchpad = ctx.get_scratchpad_grantor();

    if (dnnl_thr_syncable() && nthr_mb_ > 1) {
        simple_barrier::ctx_init(scratchpad.template get<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx));
    }

    memory_tracking::grantor_t reducer_bia_scratchpad(
            scratchpad, prefix_reducer_bia);
    auto rb = this->reducer_bias_.get();
    rb->init(reducer_bia_scratchpad);
}

template <data_type_t src_type, data_type_t diff_dst_type,
        data_type_t diff_weights_type>
void jit_avx512_common_convolution_bwd_weights_t<src_type, diff_dst_type,
        diff_weights_type>::execute_backward_weights(const exec_ctx_t &ctx)
        const {
    prepare_scratchpad_data(ctx);

#if DNNL_THR_SYNC == 1
    parallel(nthr_, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        assert(nthr_ == nthr);

        thread_info_t thread_info(this, ctx, ithr);

        switch (pd()->jcp_.harness) {
            case harness_2d_reduction:
                compute_diff_weights_2d(&thread_info);
                if (nthr_mb_ > 1) reduce_diff_weights(&thread_info);
                if (pd()->with_bias()) reduce_diff_bias(&thread_info);
                break;
            case harness_3d_reduction:
                compute_diff_weights_3d(&thread_info);
                if (nthr_mb_ > 1) reduce_diff_weights_3d(&thread_info);
                if (pd()->with_bias()) reduce_diff_bias(&thread_info);
                break;
            case harness_mb_reduction:
                compute_diff_weights(&thread_info);
                if (nthr_mb_ > 1) reduce_diff_weights(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info);
                break;
            case harness_nxc:
                compute_diff_weights_nxc(&thread_info);
                if (nthr_mb_ > 1) reduce_diff_weights_3d(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info);
                break;
            default: assert(!"Invalid harness type");
        }
    });
#else
    parallel(nthr_, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        thread_info_t thread_info(this, ctx, ithr);
        switch (pd()->jcp_.harness) {
            case harness_nxc:
                compute_diff_weights_nxc(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info);
                break;
            case harness_2d_reduction:
                compute_diff_weights_2d(&thread_info);
                break;
            case harness_3d_reduction:
                compute_diff_weights_3d(&thread_info);
                break;
            case harness_mb_reduction:
                compute_diff_weights(&thread_info);
                if (pd()->with_bias()) compute_diff_bias(&thread_info);
                break;
            default: assert(!"Invalid harness type");
        }
    });

    parallel(nthr_, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        thread_info_t thread_info(this, ctx, ithr);
        if (nthr_mb_ > 1) {
            switch (pd()->jcp_.harness) {
                case harness_mb_reduction:
                case harness_2d_reduction:
                    reduce_diff_weights(&thread_info);
                    break;
                case harness_nxc:
                case harness_3d_reduction:
                    reduce_diff_weights_3d(&thread_info);
                    break;
                default: assert(!"Invalid harness type");
            }
        }
        if (pd()->with_bias()) {
            switch (pd()->jcp_.harness) {
                case harness_2d_reduction:
                case harness_3d_reduction:
                    reduce_diff_bias(&thread_info);
                    break;
                case harness_nxc:
                case harness_mb_reduction: {
                    auto rb = this->reducer_bias_.get();
                    assert(nthr == rb->balancer().nthr_);
                    if (rb->balancer().ithr_njobs(ithr) == 0) return;
                    memory_tracking::grantor_t reducer_bia_scratchpad(
                            thread_info.scratchpad, prefix_reducer_bia);
                    rb->reduce_nolock(thread_info.ithr, thread_info.diff_bias,
                            reducer_bia_scratchpad);
                } break;
                default: assert(!"Invalid harness type");
            }
        }
    });
#endif

    parallel(1, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        /* TODO: put that into compute_diff_bias() */
        auto &jcp = pd()->jcp_;
        if (pd()->with_bias() && jcp.oc_without_padding % jcp.oc_block != 0) {
            auto diff_bias = ctx.get_scratchpad_grantor()
                                     .template get<const diff_weights_data_t>(
                                             key_conv_padded_bias);
            auto diff_bias_in
                    = CTX_OUT_MEM(diff_weights_data_t *, DNNL_ARG_DIFF_BIAS);
            const dim_t padded_stride = rnd_up(jcp.oc, jcp.oc_block);
            const dim_t stride = jcp.oc_without_padding;
            for (int g = 0; g < jcp.ngroups; ++g) {
                utils::array_copy(diff_bias_in + g * stride,
                        diff_bias + g * padded_stride, stride);
            }
        }
    });
}

template struct jit_avx512_common_convolution_bwd_weights_t<data_type::f32>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
