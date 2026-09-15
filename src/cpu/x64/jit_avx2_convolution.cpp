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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/x64/jit_avx2_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace nstl;

#define src_blk_off(f, n, c, d, h, w) \
    (pd()->ndims() == 3)           ? (f).blk_off(n, c, w) \
            : (pd()->ndims() == 4) ? (f).blk_off(n, c, h, w) \
                                   : (f).blk_off(n, c, d, h, w)

#define wht_blk_off_(f, g, ...) \
    pd()->with_groups() ? (f).blk_off(g, __VA_ARGS__) : (f).blk_off(__VA_ARGS__)
#define wht_blk_off(f, g, oc, ic, kd, kh, kw) \
    (pd()->ndims() == 3)           ? wht_blk_off_(f, g, oc, ic, kw) \
            : (pd()->ndims() == 4) ? wht_blk_off_(f, g, oc, ic, kh, kw) \
                                   : wht_blk_off_(f, g, oc, ic, kd, kh, kw)

void jit_avx2_convolution_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    const auto *jcp_ptr = &pd()->jcp_;
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(pd()->jcp_.post_ops, ctx);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    // Lambda captures by copy, the bias logic must stay afront the lambda,
    // otherwise, `bias` will be captured unmodified and lead to a crash inside
    // the kernel.
    if (pd()->wants_padded_bias()) {
        auto padded_bias = ctx.get_scratchpad_grantor().get<data_t>(
                key_conv_padded_bias);
        utils::array_copy(padded_bias, bias, jcp_ptr->oc_without_padding);
        utils::array_set(padded_bias + jcp_ptr->oc_without_padding, 0.f,
                jcp_ptr->oc - jcp_ptr->oc_without_padding);
        bias = padded_bias;
    }

    parallel(jcp_ptr->nthr,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &jcp = *jcp_ptr;

        const dim_t ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
        const dim_t work_amount
                = jcp.mb * jcp.ngroups * ocb_work * jcp.od * jcp.oh;

        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        bool is_ic_physically_blocked = one_of(jcp.src_tag, format_tag::nCw8c,
                format_tag::nChw8c, format_tag::nCdhw8c);
        dim_t g_ic_offset = is_ic_physically_blocked ? jcp.nb_ic : jcp.ic;
        dim_t icb_ic_scale = is_ic_physically_blocked ? 1 : jcp.ic_block;

        bool is_oc_physically_blocked = one_of(jcp.dst_tag, format_tag::nCw8c,
                format_tag::nChw8c, format_tag::nCdhw8c);
        dim_t g_oc_offset = is_oc_physically_blocked ? jcp.nb_oc : jcp.oc;
        dim_t ocb_oc_scale = is_oc_physically_blocked ? 1 : jcp.oc_block;
        dim_t oc_bias_scale = is_oc_physically_blocked ? jcp.oc_block : 1;

        dim_t icbb = 0;
        while (icbb < jcp.nb_ic) {
            dim_t icb_step = jcp.nb_ic_blocking;
            dim_t icb_step_rem = jcp.nb_ic - icbb;
            if (icb_step_rem < jcp.nb_ic_blocking_max) icb_step = icb_step_rem;

            dim_t n {0}, g {0}, ocbb {0}, oh {0}, od {0};
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work,
                    od, jcp.od, oh, jcp.oh);
            for (dim_t iwork = start; iwork < end; ++iwork) {
                dim_t ocb = ocbb * jcp.nb_oc_blocking;
                dim_t ocb_num = jcp.nb_oc_blocking;

                for (dim_t icb = icbb; icb < icbb + icb_step; ++icb) {
                    auto par_conv = jit_conv_args_t();

                    const dim_t ij = oh * jcp.stride_h;
                    const dim_t i_t_overflow
                            = nstl::max<dim_t>(0, jcp.t_pad - ij);
                    const dim_t i_b_overflow
                            = nstl::max<dim_t>(jcp.ih,
                                      ij + (jcp.kh - 1) * (jcp.dilate_h + 1)
                                              - jcp.t_pad + 1)
                            - jcp.ih;

                    const dim_t dj = od * jcp.stride_d;
                    const dim_t d_t_overflow
                            = nstl::max<dim_t>(0, jcp.f_pad - dj);
                    const dim_t d_b_overflow
                            = nstl::max<dim_t>(jcp.id,
                                      dj + (jcp.kd - 1) * (jcp.dilate_d + 1)
                                              - jcp.f_pad + 1)
                            - jcp.id;

                    const dim_t _oc = g * g_oc_offset + ocb * ocb_oc_scale;
                    const dim_t _ic = g * g_ic_offset + icb * icb_ic_scale;

                    const dim_t ih = nstl::max<dim_t>(ij - jcp.t_pad
                                    + div_up(i_t_overflow, (jcp.dilate_h + 1))
                                            * (jcp.dilate_h + 1),
                            0);

                    const dim_t id = nstl::max<dim_t>(dj - jcp.f_pad
                                    + div_up(d_t_overflow, (jcp.dilate_d + 1))
                                            * (jcp.dilate_d + 1),
                            0);

                    par_conv.src = &src[src_blk_off(src_d, n, _ic, id, ih, 0)];

                    par_conv.dst = &dst[src_blk_off(dst_d, n, _oc, od, oh, 0)];

                    const dim_t wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
                    const dim_t wd = div_up(d_t_overflow, (jcp.dilate_d + 1));
                    par_conv.filt = &weights[wht_blk_off(
                            weights_d, g, ocb, icb, wd, wh, 0)];

                    if (icb == 0) {
                        if (bias)
                            par_conv.bias = &bias[bias_d.blk_off(
                                    _oc * oc_bias_scale)];

                        par_conv.flags |= FLAG_IC_FIRST;
                    }

                    if ((jcp.with_eltwise || jcp.with_binary)
                            && icb + 1 == jcp.nb_ic)
                        par_conv.flags |= FLAG_IC_LAST;

                    par_conv.reduce_work = this_block_size(
                            icb * jcp.ic_block, jcp.ic, jcp.ic_block);

                    par_conv.oc_blocks
                            = nstl::min<dim_t>(ocb + ocb_num, jcp.nb_oc) - ocb;

                    if (ocbb == ocb_work - 1) par_conv.oc_flag |= FLAG_OC_LAST;

                    par_conv.kw_padding = 0;
                    const dim_t kh_padding = jcp.kh
                            - div_up(i_t_overflow, (jcp.dilate_h + 1))
                            - div_up(i_b_overflow, (jcp.dilate_h + 1));
                    par_conv.kh_padding = nstl::max<dim_t>(0, kh_padding);

                    const dim_t kd_padding = jcp.kd
                            - div_up(d_t_overflow, (jcp.dilate_d + 1))
                            - div_up(d_b_overflow, (jcp.dilate_d + 1));
                    par_conv.kd_padding = nstl::max<dim_t>(0, kd_padding);

                    par_conv.post_ops_binary_rhs_arg_vec
                            = post_ops_binary_rhs_arg_vec.data();
                    par_conv.dst_orig = dst;

                    (*kernel_)(&par_conv);
                }
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work, od,
                        jcp.od, oh, jcp.oh);
            }
            icbb += icb_step;
        }
    });

    if (pd()->wants_zero_pad_dst()) ctx.zero_pad_output(DNNL_ARG_DST);
}

void jit_avx2_convolution_bwd_data_t::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const data_t *, DNNL_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto &jcp = kernel_->jcp;

    dim_t icb_work = jcp.nb_ic / jcp.nb_ic_blocking;
    dim_t ih_block_size = jcp.ih;
    dim_t num_ih_blocks = utils::div_up(jcp.ih, ih_block_size);
    dim_t work_amount = jcp.mb * jcp.ngroups * icb_work * num_ih_blocks;

    const auto data_size = sizeof(data_t);
    const auto L2 = platform::get_per_core_cache_size(2) / data_size;
    // input + output + weights per iteration by nb_oc_blocking
    auto ic_chunk = jcp.nb_ic_blocking * jcp.ic_block;
    auto oc_chunk = jcp.nb_oc_blocking * jcp.oc_block;
    auto iter_data_amount = (size_t)jcp.id * jcp.ih * jcp.iw * ic_chunk
            + (size_t)jcp.od * jcp.oh * jcp.ow * oc_chunk
            + (size_t)jcp.kd * jcp.kh * jcp.kw * ic_chunk * oc_chunk;

    if (work_amount < 2 * static_cast<dim_t>(jcp.nthr)
            || iter_data_amount > L2) {
        ih_block_size = 1;
        num_ih_blocks = utils::div_up(jcp.ih, ih_block_size);
        work_amount *= num_ih_blocks;
    }

    const dim_t ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    const dim_t ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);

    bool is_ic_physically_blocked = one_of(jcp.src_tag, format_tag::nCw8c,
            format_tag::nChw8c, format_tag::nCdhw8c);
    dim_t g_ic_offset = is_ic_physically_blocked ? jcp.nb_ic : jcp.ic;
    dim_t icb_ic_scale = is_ic_physically_blocked ? 1 : jcp.ic_block;

    bool is_oc_physically_blocked = one_of(jcp.dst_tag, format_tag::nCw8c,
            format_tag::nChw8c, format_tag::nCdhw8c);
    dim_t g_oc_offset = is_oc_physically_blocked ? jcp.nb_oc : jcp.oc;
    dim_t ocb_oc_scale = is_oc_physically_blocked ? 1 : jcp.oc_block;

    const bool is_ddst_layout_nxc = one_of(
            jcp.dst_tag, format_tag::nwc, format_tag::nhwc, format_tag::ndhwc);
    const dim_t oc_step = is_ddst_layout_nxc ? jcp.nb_oc_blocking : 1;

    auto ker = [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        dim_t n {0}, g {0}, icbb {0}, ihb {0};
        nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, icbb, icb_work, ihb,
                num_ih_blocks);
        for (dim_t iwork = start; iwork < end; ++iwork) {
            for_(dim_t oc = 0; oc < jcp.nb_oc; oc += jcp.nb_oc_blocking)
            for (dim_t id = 0; id < jcp.id; ++id) {
                const dim_t cur_nb_oc
                        = nstl::min<dim_t>(jcp.nb_oc - oc, jcp.nb_oc_blocking);

                auto par_conv = jit_conv_args_t();

                dim_t d_t_overflow, d_b_overflow, od;
                if (jcp.dilate_d != 0) { // stride == 1
                    const dim_t dilate_d = jcp.dilate_d + 1;
                    d_t_overflow = div_up(
                            nstl::max<dim_t>(0, ext_kd - 1 - id - jcp.f_pad),
                            dilate_d);
                    d_b_overflow = div_up(
                            nstl::max<dim_t>(
                                    0, ext_kd - jcp.id + id - jcp.back_pad),
                            dilate_d);
                    od = id + jcp.f_pad - d_b_overflow * dilate_d;
                } else {
                    d_t_overflow
                            = nstl::max<dim_t>(0, jcp.kd - 1 - id - jcp.f_pad);
                    d_b_overflow = nstl::max<dim_t>(
                            0, jcp.kd - 1 - (jcp.id - 1 - id) - jcp.back_pad);
                    od = id + jcp.f_pad - d_b_overflow;
                }
                par_conv.kd_padding = jcp.kd - d_t_overflow - d_b_overflow;

                dim_t ih_start = ihb * ih_block_size;
                dim_t ih_end
                        = nstl::min<dim_t>(jcp.ih, ih_start + ih_block_size);
                for (dim_t ih = ih_start; ih < ih_end; ++ih) {

                    dim_t k_lo, oh;
                    if (jcp.dilate_h != 0) { // stride == 1
                        const dim_t dilate_h = jcp.dilate_h + 1;
                        dim_t i_t_overflow
                                = div_up(nstl::max<dim_t>(0,
                                                 ext_kh - 1 - ih - jcp.t_pad),
                                        dilate_h);
                        dim_t i_b_overflow = div_up(
                                nstl::max<dim_t>(
                                        0, ext_kh - jcp.ih + ih - jcp.b_pad),
                                dilate_h);
                        par_conv.kh_padding
                                = jcp.kh - i_t_overflow - i_b_overflow;
                        k_lo = i_b_overflow;
                        oh = ih + jcp.t_pad - k_lo * dilate_h;
                    } else {
                        dim_t i_t_overflow = nstl::max<dim_t>(0,
                                (jcp.kh - 1 - ih - jcp.t_pad) / jcp.stride_h);
                        dim_t i_b_overflow = nstl::max<dim_t>(0,
                                (jcp.kh - jcp.ih + ih - jcp.b_pad)
                                        / jcp.stride_h);
                        dim_t overflow_kh_hi = jcp.kh - 1
                                - modulo(jcp.ih - 1 + jcp.b_pad - ih,
                                        jcp.stride_h);
                        dim_t overflow_kh_lo = (ih + jcp.t_pad) % jcp.stride_h;

                        par_conv.kh_padding = (overflow_kh_hi - overflow_kh_lo)
                                        / jcp.stride_h
                                + 1 - i_t_overflow - i_b_overflow;

                        k_lo = overflow_kh_lo + i_b_overflow * jcp.stride_h;
                        oh = (ih + jcp.t_pad - k_lo) / jcp.stride_h;
                    }
                    par_conv.kw_padding = 0;

                    par_conv.src = &diff_src[src_blk_off(diff_src_d, n,
                            g * g_ic_offset
                                    + jcp.nb_ic_blocking * icbb * icb_ic_scale,
                            id, ih, 0)];
                    par_conv.dst = &diff_dst[src_blk_off(diff_dst_d, n,
                            g * g_oc_offset + ocb_oc_scale * oc, od, oh, 0)];
                    par_conv.filt = &weights[wht_blk_off(weights_d, g, oc,
                            jcp.nb_ic_blocking * icbb, d_b_overflow, k_lo, 0)];

                    par_conv.src_prf = nullptr;
                    par_conv.dst_prf = nullptr;
                    par_conv.filt_prf = nullptr;
                    par_conv.channel = oc;
                    par_conv.ch_blocks = cur_nb_oc;

                    if (is_ddst_layout_nxc) {
                        par_conv.load_work = this_block_size(
                                icbb * jcp.nb_ic_blocking * jcp.ic_block,
                                jcp.ic, jcp.nb_ic_blocking * jcp.ic_block);
                        par_conv.reduce_work
                                = this_block_size(oc * jcp.oc_block, jcp.oc,
                                        oc_step * jcp.oc_block);

                        if (par_conv.load_work % jcp.ic_block > 0)
                            par_conv.flags |= FLAG_IC_LAST;
                    }

                    (*kernel_)(&par_conv);
                }
            }
            nd_iterator_step(n, jcp.mb, g, jcp.ngroups, icbb, icb_work, ihb,
                    num_ih_blocks);
        }
    };

    parallel(jcp.nthr, ker);
}

void jit_avx2_convolution_bwd_weights_t::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_WEIGHTS);
    auto diff_bias_in = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_BIAS);

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = kernel_->jcp;

    const bool is_bias_padded
            = pd()->with_bias() && (jcp.oc_without_padding % jcp.oc_block != 0);

    data_t *diff_bias = is_bias_padded
            ? scratchpad.get<data_t>(key_conv_padded_bias)
            : diff_bias_in;

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));

    memory_tracking::grantor_t reducer_bia_scratchpad(
            scratchpad, prefix_reducer_bia);
    auto rb = this->reducer_bias_.get();
    rb->init(reducer_bia_scratchpad);

    memory_tracking::grantor_t reducer_wei_scratchpad(
            scratchpad, prefix_reducer_wei);
    auto rw = this->reducer_weights_.get();
    rw->init(reducer_wei_scratchpad);

    bool is_ic_physically_blocked = one_of(jcp.src_tag, format_tag::nCw8c,
            format_tag::nChw8c, format_tag::nCdhw8c);
    dim_t g_ic_offset = is_ic_physically_blocked ? jcp.nb_ic : jcp.ic;
    dim_t icb_ic_scale = is_ic_physically_blocked ? 1 : jcp.ic_block;

    bool is_oc_physically_blocked = one_of(jcp.dst_tag, format_tag::nCw8c,
            format_tag::nChw8c, format_tag::nCdhw8c);
    bool is_ddst_layout_nxc = !is_oc_physically_blocked;
    dim_t g_oc_offset = is_oc_physically_blocked ? jcp.nb_oc : jcp.oc;
    dim_t ocb_oc_scale = is_oc_physically_blocked ? 1 : jcp.oc_block;

    auto ker = [= COMPAT_THIS_CAPTURE](int ithr, int nthr) {
        assert(nthr == rw->balancer().nthr_);

        const dim_t w_job_start = rw->balancer().ithr_job_off(ithr);
        const int w_njobs = rw->balancer().ithr_njobs(ithr);

        if (w_njobs == 0) return;

        /* reduction dimension */
        dim_t img_od_start {0}, img_od_end {0};
        dim_t img {0}, od_s {0};
        balance211(jcp.mb * jcp.od, rw->balancer().nthr_per_group_,
                rw->balancer().id_in_group(ithr), img_od_start, img_od_end);

        dim_t img_start = img_od_start;
        dim_t img_end = img_od_end;
        nd_iterator_init(img_start, img, jcp.mb, od_s, jcp.od);
        const dim_t img_first = img;

        /* jobs */
        dim_t g_start {0}, ocb_start {0}, icb_start {0};
        nd_iterator_init(w_job_start, g_start, jcp.ngroups, ocb_start,
                jcp.nb_oc, icb_start, jcp.nb_ic);

        while (img_start < img_end) {
            dim_t g = g_start, ocb = ocb_start, icb = icb_start;

            const dim_t work_rem = img_end - img_start;
            const dim_t od_e
                    = od_s + work_rem > jcp.od ? jcp.od : od_s + work_rem;
            const dim_t id_s = od_s * jcp.stride_d;
            const dim_t idp = jcp.id + jcp.f_pad + jcp.back_pad;

            if (id_s < idp - jcp.back_pad - jcp.kd + 1)
                for (int w_job_loc = 0; w_job_loc < w_njobs; ++w_job_loc) {
                    const dim_t _oc = g * g_oc_offset + ocb * ocb_oc_scale;
                    const dim_t _ic = g * g_ic_offset + icb * icb_ic_scale;

                    /* TODO: put dw <-- 0 in kernel */
                    if (img == img_first)
                        array_set(rw->get_local_ptr(ithr, diff_weights,
                                          reducer_wei_scratchpad)
                                        + w_job_loc * rw->balancer().job_size_,
                                0, rw->balancer().job_size_);

                    for (dim_t od = od_s; od < od_e; ++od) {
                        const dim_t id = od * jcp.stride_d;
                        if (id >= jcp.id - jcp.back_pad - jcp.kd + 1) break;

                        auto par_conv = jit_conv_args_t();
                        par_conv.src
                                = &src[src_blk_off(src_d, img, _ic, id, 0, 0)];
                        par_conv.dst = &diff_dst[src_blk_off(
                                diff_dst_d, img, _oc, od, 0, 0)];
                        par_conv.filt = rw->get_local_ptr(ithr, diff_weights,
                                                reducer_wei_scratchpad)
                                + w_job_loc * rw->balancer().job_size_;

                        if (ocb == jcp.nb_oc - 1)
                            par_conv.flags |= FLAG_OC_LAST;

                        par_conv.channel = this_block_size(
                                icb * jcp.ic_block, jcp.ic, jcp.ic_block);

                        (*kernel_)(&par_conv);
                    }
                    nd_iterator_step(
                            g, jcp.ngroups, ocb, jcp.nb_oc, icb, jcp.nb_ic);
                }
            nd_iterator_jump(img_start, img_end, img, jcp.mb, od_s, jcp.od);
        }

        if (dnnl_thr_syncable())
            rw->reduce(ithr, diff_weights, reducer_wei_scratchpad);
    };

    auto ker_bias = [=](int ithr, int nthr) {
        assert(nthr == rb->balancer().nthr_);

        const dim_t b_job_start = rb->balancer().ithr_job_off(ithr);
        const int b_njobs = rb->balancer().ithr_njobs(ithr);

        if (b_njobs == 0) return;

        /* reduction dimension */
        dim_t img_start {0}, img_end {0};
        balance211(jcp.mb, rb->balancer().nthr_per_group_,
                rb->balancer().id_in_group(ithr), img_start, img_end);

        /* jobs */
        dim_t g_start {0}, ocb_start {0};
        nd_iterator_init(
                b_job_start, g_start, jcp.ngroups, ocb_start, jcp.nb_oc);

        for (dim_t img = img_start; img < img_end; ++img) {
            dim_t g = g_start, ocb = ocb_start;
            for (int b_job_loc = 0; b_job_loc < b_njobs; ++b_job_loc) {
                const dim_t _oc = g * g_oc_offset + ocb * ocb_oc_scale;

                const data_t *d_dst = &diff_dst[diff_dst_d.blk_off(img, _oc)];
                data_t *d_bias = rb->get_local_ptr(ithr, diff_bias,
                                         reducer_bia_scratchpad)
                        + b_job_loc * rb->balancer().job_size_;

                if (img == img_start)
                    for (dim_t o = 0; o < jcp.oc_block; ++o)
                        d_bias[o] = 0.f;

                const dim_t max_oc = this_block_size(
                        ocb * jcp.oc_block, jcp.oc, jcp.oc_block);

                for (dim_t dhw = 0; dhw < jcp.od * jcp.oh * jcp.ow; ++dhw) {
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
            rb->reduce(ithr, diff_bias, reducer_bia_scratchpad);
    };

    if (dnnl_thr_syncable()) {
        assert(IMPLICATION(pd()->with_bias(),
                rw->balancer().nthr_ == rb->balancer().nthr_));
        parallel(rw->balancer().nthr_,
                [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
            ker(ithr, nthr);
            if (pd()->with_bias()) ker_bias(ithr, nthr);
        });
    } else {
        parallel(rw->balancer().nthr_,
                [=](int ithr, int nthr) { ker(ithr, nthr); });
        parallel(rw->balancer().nthr_, [=](int ithr, int nthr) {
            assert(nthr == rw->balancer().nthr_);
            MAYBE_UNUSED(nthr);
            if (rw->balancer().ithr_njobs(ithr) == 0) return;
            rw->reduce_nolock(ithr, diff_weights, reducer_wei_scratchpad);
        });
        if (pd()->with_bias()) {
            parallel(rb->balancer().nthr_,
                    [=](int ithr, int nthr) { ker_bias(ithr, nthr); });
            parallel(rb->balancer().nthr_, [=](int ithr, int nthr) {
                assert(nthr == rb->balancer().nthr_);
                MAYBE_UNUSED(nthr);
                if (rb->balancer().ithr_njobs(ithr) == 0) return;
                rb->reduce_nolock(ithr, diff_bias, reducer_bia_scratchpad);
            });
        }
    }

    parallel(1, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        /* TODO: put this in ker_bias */
        if (pd()->with_bias() && (jcp.oc_without_padding % jcp.oc_block != 0)) {
            const dim_t padded_stride = rnd_up(jcp.oc, jcp.oc_block);
            const dim_t stride = jcp.oc_without_padding;
            for (dim_t g = 0; g < jcp.ngroups; ++g)
                utils::array_copy(diff_bias_in + g * stride,
                        diff_bias + g * padded_stride, stride);
        }
    });
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
