/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "cpu/reorder/ref_reorder.hpp"

#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

#include "common/dnnl_thread.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
// Note: offset is always coincide with logical index because quantization
// entries don't have a notion of physical formats.
dim_t get_quant_off(const dims_t &input_idx, const int ndims,
        const int quant_mask, const dim_t g0, const dim_t g1,
        const memory_desc_t &quant_md) {
    dims_t quant_idx {};
    utils::array_copy(quant_idx, input_idx, ndims);
    utils::apply_mask_on_dims(quant_idx, ndims, quant_mask);
    // Note: an `idx` must divide by a group value as grouped quantization
    // applies to consecutive points.
    if (ndims >= 2) {
        quant_idx[ndims - 1] /= g1;
        quant_idx[ndims - 2] /= g0;
    }

    const memory_desc_wrapper q_mdw(quant_md);
    return q_mdw.off_v(quant_idx);
}
} // namespace

status_t ref_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    using namespace data_type;

    VDISPATCH_REORDER(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const memory_desc_wrapper input_d(src_md());
    const memory_desc_wrapper output_d(dst_md());

    // This reorder serves to benchdnn data filling process, which processes
    // each tensor separately. It means that data types used in all primitives
    // are in conformance with a single check. However, this is not true, and
    // the most vivid example is f16 data type. Compute primitives support
    // f32:f16:f32 combination on AVX2+ while f16 data type support is claimed
    // on avx512_core_f16 and avx2_vnni_2 ISAs. Disabling this primitive
    // according to official rules makes those cases failing. Another use case
    // is scales data types which are not checked against supported platforms
    // at all. There are probably some more examples but these two are most
    // popular.
    //
    // In this sense, the following checks are disabled until there's a better
    // way to track support for data types and their legit combinations.
    //
    // The checks were proposed as a way to limit f64 and s64 data types which
    // don't have any usage among CPU implementations. While they are disabled
    // through the `platform::has_data_type_support`, there's an additional
    // mechanism of reorder lists with data types involved which doesn't let
    // those illegal data types to go through - they receive an empty impl list
    // and return unimplemented.
    //
    // TODO: enable checks once ready.
    // VDISPATCH_REORDER(platform::has_data_type_support(input_d.data_type()),
    //         VERBOSE_UNSUPPORTED_DT);
    // VDISPATCH_REORDER(platform::has_data_type_support(output_d.data_type()),
    //         VERBOSE_UNSUPPORTED_DT);

    // These types are not supported in this implementation as they would
    // require special handling on a common infrastructure side (loading and
    // storing 64-bit values for int64_t and double properly), which doesn't
    // seem to bring any value to the library at this point, thus, just making
    // sure they are not expected here.
    VDISPATCH_REORDER(!utils::one_of(input_d.data_type(), f64, s64),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(!utils::one_of(output_d.data_type(), f64, s64),
            VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_REORDER(
            input_d.is_blocking_desc() && !input_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    VDISPATCH_REORDER(
            output_d.is_blocking_desc() && !output_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(
            attr()->has_default_values(skip_mask_t::scales_data_type
                    | skip_mask_t::scales_groups
                    | skip_mask_t::zero_points_data_type
                    | skip_mask_t::zero_points_groups | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    // An e8m0 source or destination goes through a dedicated raw-conversion
    // path (see execute) that must not apply any floating-point operation
    // around the value, thus, no attributes can be supported.
    VDISPATCH_REORDER(IMPLICATION(utils::one_of(e8m0, input_d.data_type(),
                                          output_d.data_type()),
                              attr()->has_default_values()),
            VERBOSE_UNSUPPORTED_ATTR);

    const auto &post_ops = attr()->post_ops_;
    VDISPATCH_REORDER(
            IMPLICATION(post_ops.len() != 0,
                    post_ops.len() == 1 && post_ops.entry_[0].is_sum(false)),
            VERBOSE_UNSUPPORTED_POSTOP);

    auto gpu_zp = memory_extra_flags::compensation_gpu_conv_asymmetric_src;
    VDISPATCH_REORDER(!(dst_md()->extra.flags & gpu_zp),
            VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    return status::success;
}

status_t ref_reorder_t::execute(const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const auto dst_dt = output_d.data_type();

    // This kernel can be used for tensors with multiple inner blocks for
    // which generic zero padding must be used.
    ctx.zero_pad_output(DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_d
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const auto dst_scales_d
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const int src_scales_mask = scales.get_mask(DNNL_ARG_FROM);
    const int dst_scales_mask = scales.get_mask(DNNL_ARG_TO);
    const auto src_scales_group0 = scales.get_group(DNNL_ARG_FROM, -2);
    const auto src_scales_group1 = scales.get_group(DNNL_ARG_FROM, -1);
    memory_desc_t src_scales_md {};
    if (with_src_scales) {
        CHECK(scales.get(DNNL_ARG_FROM).get_md(src_scales_md, *input_d.md_));
    }
    memory_desc_t dst_scales_md {};
    if (with_dst_scales) {
        CHECK(scales.get(DNNL_ARG_TO).get_md(dst_scales_md, *input_d.md_));
    }

    const void *src_zero_points = CTX_IN_MEM(
            const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM);
    const void *dst_zero_points
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_TO);
    const auto src_zps_d
            = ctx.memory_mdw(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_FROM);
    const auto dst_zps_d
            = ctx.memory_mdw(DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_TO);
    const auto &zps = pd()->attr()->zero_points_;
    const bool with_src_zps = !zps.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_zps = !zps.has_default_values(DNNL_ARG_TO);
    const int src_zps_mask = zps.get_mask(DNNL_ARG_FROM);
    const int dst_zps_mask = zps.get_mask(DNNL_ARG_TO);
    const auto src_zps_group0 = zps.get_group(DNNL_ARG_FROM, -2);
    const auto src_zps_group1 = zps.get_group(DNNL_ARG_FROM, -1);
    memory_desc_t src_zps_md {};
    if (with_src_zps) {
        CHECK(zps.get(DNNL_ARG_FROM).get_md(src_zps_md, *input_d.md_));
    }
    memory_desc_t dst_zps_md {};
    if (with_dst_zps) {
        CHECK(zps.get(DNNL_ARG_TO).get_md(dst_zps_md, *output_d.md_));
    }

    const int ndims = input_d.ndims();
    const float beta = pd()->beta();

    // Parallelize at storage-unit granularity along the destination's innermost
    // physical dimension so that every packed storage unit (which a sub-byte
    // output shares across several elements of a single byte) is assembled by a
    // single thread. Splitting a unit across threads would race on the shared
    // byte's read-modify-write and corrupt neighboring elements.
    const int dt_bits = types::data_type_bits(output_d.data_type());
    constexpr int bits_in_byte = 8;
    const int elems_in_bytes = math::lcm(dt_bits, bits_in_byte) / dt_bits;
    const int innermost_dim = output_d.innermost_dim();
    const dim_t innermost_dim_size = output_d.dims()[innermost_dim];
    dims_t group_dims {};
    utils::array_copy(group_dims, output_d.dims(), ndims);
    group_dims[innermost_dim]
            = utils::div_up(innermost_dim_size, elems_in_bytes);
    dim_t n_groups = 1;
    for (int d = 0; d < ndims; d++)
        n_groups *= group_dims[d];

    parallel(0, [=](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(n_groups, nthr, ithr, start, end);

        for (dim_t g = start; g < end; g++) {
            dims_t idx {};
            utils::l_dims_by_l_offset(idx, g, group_dims, ndims);
            const dim_t pack_base = idx[innermost_dim] * elems_in_bytes;

            for (int k = 0; k < elems_in_bytes; k++) {
                const dim_t pack_pos = pack_base + k;
                // A storage unit at the tensor tail may reach into the padded
                // area; those nibbles are already handled by zero padding.
                if (pack_pos >= innermost_dim_size) break;
                idx[innermost_dim] = pack_pos;

                float src_scale = 1.f;
                if (with_src_scales) {
                    const dim_t src_scales_off = get_quant_off(idx, ndims,
                            src_scales_mask, src_scales_group0,
                            src_scales_group1, src_scales_md);
                    src_scale = io::load_float_value(src_scales_d.data_type(),
                            src_scales, src_scales_off);
                }

                float dst_scale = 1.f;
                if (with_dst_scales) {
                    const dim_t dst_scales_off = get_quant_off(
                            idx, ndims, dst_scales_mask, 1, 1, dst_scales_md);
                    dst_scale = io::load_float_value(dst_scales_d.data_type(),
                            dst_scales, dst_scales_off);
                }

                int src_zp_val = 0;
                if (with_src_zps) {
                    const dim_t src_zps_off
                            = get_quant_off(idx, ndims, src_zps_mask,
                                    src_zps_group0, src_zps_group1, src_zps_md);
                    src_zp_val = io::load_int_value(src_zps_d.data_type(),
                            src_zero_points, src_zps_off);
                }

                int dst_zp_val = 0;
                if (with_dst_zps) {
                    const dim_t dst_zps_off = get_quant_off(
                            idx, ndims, dst_zps_mask, 1, 1, dst_zps_md);
                    dst_zp_val = io::load_int_value(dst_zps_d.data_type(),
                            dst_zero_points, dst_zps_off);
                }

                const auto i_off = input_d.off_v(idx);
                const auto o_off = output_d.off_v(idx);
                if (src_dt == data_type::e8m0) {
                    // Reorder from e8m0 to f32 is used for benchdnn correctness
                    // validation purpose only. A dedicated path is required to
                    // preserve a minimal e8m0 value which gets converted by any
                    // compiler to 0 if any floating-point operation or
                    // comparison is applied around the value.
                    auto s = io::load_float_value(src_dt, input, i_off);
                    io::store_float_value(dst_dt, s, output, o_off);
                } else {
                    float s = io::load_float_value(src_dt, input, i_off);
                    float d = src_scale * (s - src_zp_val);
                    if (beta)
                        d += beta * io::load_float_value(dst_dt, output, o_off);
                    d = d / dst_scale + dst_zp_val;
                    io::store_float_value(dst_dt, d, output, o_off);
                }
            }
        }
    });
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
