/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/jit_avx512_core_amx_1x1_convolution.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

#define wht_blk_off(d, g, ...) \
    (pd()->with_groups() ? (d).blk_off((g), __VA_ARGS__) \
                         : (d).blk_off(__VA_ARGS__))

#define md_blk_off(md, n, c, d, h, w) \
    (pd()->ndims() == 3 \
                    ? (md).blk_off((n), (c), (w)) \
                    : (pd()->ndims() == 4 ? (md).blk_off((n), (c), (h), (w)) \
                                          : (md).blk_off( \
                                                    (n), (c), (d), (h), (w))))

void jit_avx512_core_amx_1x1_convolution_fwd_t::prepare_padded_bias(
        const char *&bias, const memory_tracking::grantor_t &scratchpad) const {
    if (!pd()->wants_padded_bias()) return;

    const size_t bia_dt_size = pd()->jcp_.typesize_bia;
    auto padded_bias = scratchpad.template get<char>(
            memory_tracking::names::key_conv_padded_bias);
    utils::array_copy(
            padded_bias, bias, bia_dt_size * pd()->jcp_.oc_without_padding);
    utils::array_set(padded_bias + bia_dt_size * pd()->jcp_.oc_without_padding,
            0.f, bia_dt_size * (pd()->jcp_.oc - pd()->jcp_.oc_without_padding));
    bias = padded_bias;
}

status_t jit_avx512_core_amx_1x1_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {

    auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(pd()->jcp_.post_ops, ctx);

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

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type)
            : 0;
    const size_t dst_dt_size
            = types::data_type_size(pd()->desc()->dst_desc.data_type);
    const size_t src_dt_size
            = types::data_type_size(pd()->desc()->src_desc.data_type);
    const size_t wei_dt_size
            = types::data_type_size(pd()->desc()->weights_desc.data_type);

    prepare_padded_bias(bias, ctx.get_scratchpad_grantor());

    const auto *jcp_ptr = &pd()->jcp_;
    assert(jcp_ptr->nb_oc % jcp_ptr->nb_oc_blocking == 0);

    const size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    const int32_t *zp_compensation = jcp_ptr->src_zero_point
            ? reinterpret_cast<const int32_t *>(&weights[offset])
            : nullptr;

    const bool is_ic_tail
            = jcp_ptr->ic_without_padding % jcp_ptr->ic_block_int_np;
    auto wsp = ctx.get_scratchpad_grantor().template get<int32_t>(
            key_conv_amx_wsp_buffer);
    int32_t *wsp_tile = (is_ic_tail)
            ? ctx.get_scratchpad_grantor().template get<int32_t>(
                      key_conv_amx_tile_buffer)
            : nullptr;
    auto global_tcfg = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_tilecfg);

    parallel(jcp_ptr->nthr,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &jcp = *jcp_ptr;

        const size_t wei_oc_shift = static_cast<size_t>(
                utils::rnd_up(jcp.ic_without_padding, jcp.ic_block_int)
                * jcp.oc_block * jcp.nb_oc_blocking);

        const dim_t nb_os = (jcp.tile_tail) ? jcp.nb_os + 1 : jcp.nb_os;
        const dim_t os_step = jcp.nb_os2_blocking * jcp.nb_os_blocking;
        const dim_t os_chunks = div_up(nb_os, os_step);

        const dim_t oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;

        const dim_t work_amount = jcp.mb * jcp.ngroups * os_chunks * oc_chunks;

        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_args_t();
        char *const __restrict tcfg = global_tcfg + 2 * ithr * AMX_PALETTE_SIZE;
        kernel_->tile_configure(tcfg);
        p.tile_cfg = tcfg;
        p.tile_cfg_tail = tcfg + 64;

        amx_tile_configure(tcfg);

        float *dst_scales_inv_ptr = nullptr;
        if (jcp.with_dst_scales) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr
                    = ctx.get_scratchpad_grantor().template get<float>(
                              key_conv_dst_scales)
                    + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        dim_t mb {0}, g {0}, _osb {0}, _ocb {0};
        nd_iterator_init(start, mb, jcp.mb, g, jcp.ngroups, _osb, os_chunks,
                _ocb, oc_chunks);

        while (start < end) {
            const dim_t osb = _osb * os_step;
            const dim_t ocb = _ocb * jcp.nb_oc_blocking;
            auto bias_w = bias
                    ? bias + (bias_d.blk_off(ocb * jcp.oc_block) * bia_dt_size)
                    : nullptr;

            const dim_t oc = g * jcp.oc_without_padding + ocb * jcp.oc_block;
            const dim_t ic = g * jcp.ic_without_padding;

            p.acc_s32 = wsp + ithr * jcp.wsp_buffer_size;
            p.src_prf = wsp_tile + ithr * (jcp.wsp_buffer_size / 2);
            p.filt = weights + wei_dt_size * _ocb * wei_oc_shift;
            p.bias = bias_w;
            p.src_scales = src_scales;
            p.wei_scales = jcp.with_wei_scales
                    ? static_cast<const float *>(wei_scales)
                            + jcp.is_oc_scale * oc
                    : nullptr;
            p.dst_scales = dst_scales_inv_ptr;
            p.oc_blocks = ocb;

            p.zp_compensation
                    = jcp.src_zero_point ? zp_compensation + oc : nullptr;
            p.src_zero_point = src_zero_points;
            p.dst_zero_point = dst_zero_points;

            p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
            p.dst_orig = dst;

            const bool check_last_sp = is_ic_tail && !(nb_os % 2);
            const bool is_overflow = (osb + os_step >= nb_os);
            if (is_overflow
                    && (os_chunks > 1 || (os_chunks == 1 && is_ic_tail))) {
                const dim_t step = (check_last_sp) ? 1 : jcp.nb_os_blocking;
                for (dim_t osi = 0; osi < nb_os - osb; osi += step) {
                    const dim_t osb_i = osi + osb;
                    dim_t od {0}, oh {0}, ow {0};
                    nd_iterator_init(osb_i * jcp.tile_width, od, jcp.od, oh,
                            jcp.oh, ow, jcp.ow);
                    size_t dst_offset = md_blk_off(dst_d, mb, oc, od, oh, ow);
                    p.dst = dst + dst_dt_size * dst_offset;

                    const dim_t id = od * jcp.stride_d;
                    const dim_t ih = oh * jcp.stride_h;
                    const dim_t iw = ow * jcp.stride_w;
                    size_t inp_offset = md_blk_off(src_d, mb, ic, id, ih, iw);
                    p.src = src + src_dt_size * inp_offset;

                    bool l_overflow = osb_i + jcp.nb_os_blocking >= nb_os;
                    p.last_h = (check_last_sp || (nb_os % 2 && l_overflow)) ? 1
                                                                            : 0;
                    p.is_osb = 0;
                    (*kernel_)(&p);
                }
            } else {
                dim_t od {0}, oh {0}, ow {0};
                nd_iterator_init(osb * jcp.tile_width, od, jcp.od, oh, jcp.oh,
                        ow, jcp.ow);
                size_t dst_offset = md_blk_off(dst_d, mb, oc, od, oh, ow);
                p.dst = dst + dst_dt_size * dst_offset;

                const dim_t id = od * jcp.stride_d;
                const dim_t ih = oh * jcp.stride_h;
                const dim_t iw = ow * jcp.stride_w;
                size_t inp_offset = md_blk_off(src_d, mb, ic, id, ih, iw);
                p.src = src + src_dt_size * inp_offset;

                p.last_h = 0;
                p.is_osb = 1;

                (*kernel_)(&p);
            }
            ++start;
            nd_iterator_step(mb, jcp.mb, g, jcp.ngroups, _osb, os_chunks, _ocb,
                    oc_chunks);
        }

        amx_tile_release();
    });
    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
