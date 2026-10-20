/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::alg_kind;
using namespace dnnl::impl::types;

#define VCHECK_DECONV(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, deconvolution, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__)

#define VCHECK_DECONV_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, deconvolution, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__)
namespace {
status_t deconv_desc_init(deconvolution_desc_t *deconv_desc,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r) {
    VCHECK_DECONV(!any_null(deconv_desc, src_desc, weights_desc, dst_desc,
                          strides, padding_l),
            VERBOSE_NULL_ARG);
    VCHECK_DECONV(
            one_of(alg_kind, deconvolution_direct, deconvolution_winograd),
            VERBOSE_BAD_ALGORITHM);

    VCHECK_DECONV(!any_memory_desc_host_scalar(
                          src_desc, weights_desc, bias_desc, dst_desc),
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    if (padding_r == nullptr) padding_r = padding_l;

    auto dd = deconvolution_desc_t();
    dd.primitive_kind = primitive_kind::deconvolution;
    dd.prop_kind = prop_kind;
    dd.alg_kind = alg_kind;
    dd.use_inversion = false; // Must be always `false` for deconv.

    dd.diff_src_desc = dd.src_desc = zero_md();
    dd.diff_dst_desc = dd.dst_desc = zero_md();
    dd.diff_weights_desc = dd.weights_desc = zero_md();
    dd.diff_bias_desc = dd.bias_desc = zero_md();

    const bool is_fwd = one_of(prop_kind, forward_training, forward_inference);
    const bool with_bias
            = bias_desc && bias_desc->format_kind != format_kind::undef;
    const bool with_groups = weights_desc->ndims == src_desc->ndims + 1;

    bool runtime_dims_or_strides
            = memory_desc_wrapper(src_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(weights_desc).has_runtime_dims_or_strides()
            || memory_desc_wrapper(dst_desc).has_runtime_dims_or_strides();
    if (with_bias)
        runtime_dims_or_strides = runtime_dims_or_strides
                || memory_desc_wrapper(bias_desc).has_runtime_dims_or_strides();
    VCHECK_DECONV_UNIMPL(
            !runtime_dims_or_strides, VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    (prop_kind == backward_data ? dd.diff_src_desc : dd.src_desc) = *src_desc;
    (is_fwd ? dd.dst_desc : dd.diff_dst_desc) = *dst_desc;
    (prop_kind == backward_weights ? dd.diff_weights_desc : dd.weights_desc)
            = *weights_desc;
    if (with_bias)
        (prop_kind == backward_weights ? dd.diff_bias_desc : dd.bias_desc)
                = *bias_desc;

    int sp_dims = src_desc->ndims - 2;
    utils::array_copy(dd.strides, strides, sp_dims);
    utils::array_copy(dd.padding[0], padding_l, sp_dims);
    utils::array_copy(dd.padding[1], padding_r, sp_dims);
    if (dilates)
        utils::array_copy(dd.dilates, dilates, sp_dims);
    else
        utils::array_set(dd.dilates, 0, sp_dims);

    dd.accum_data_type = types::default_accum_data_type(src_desc->data_type,
            weights_desc->data_type, dst_desc->data_type, prop_kind);
    VCHECK_DECONV(dd.accum_data_type != data_type::undef,
            VERBOSE_INVALID_DATATYPE, "accumulation");

    const dim_t g = with_groups ? weights_desc->dims[0] : 1;
    VCHECK_DECONV(src_desc->ndims == dst_desc->ndims,
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src", "dst", src_desc->ndims,
            dst_desc->ndims);
    VCHECK_DECONV(utils::one_of(src_desc->ndims, 3, 4, 5), VERBOSE_BAD_NDIMS,
            "src", src_desc->ndims);
    VCHECK_DECONV(utils::one_of(weights_desc->ndims, src_desc->ndims,
                          src_desc->ndims + 1),
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src", "weights",
            weights_desc->ndims, src_desc->ndims);
    VCHECK_DECONV(IMPLICATION(with_bias, bias_desc->ndims == 1),
            VERBOSE_BAD_NDIMS, "bias", bias_desc->ndims);
    VCHECK_DECONV(
            IMPLICATION(with_bias, bias_desc->dims[0] == dst_desc->dims[1]),
            VERBOSE_INCONSISTENT_DIM, "bias", 0, "dst", 1);
    VCHECK_DECONV(src_desc->dims[0] == dst_desc->dims[0],
            VERBOSE_INCONSISTENT_DIM, "src", 0, "dst", 0);
    VCHECK_DECONV(src_desc->dims[1] == g * weights_desc->dims[with_groups + 1],
            VERBOSE_INCONSISTENT_DIM, "src", 1, "weights", with_groups + 1);
    VCHECK_DECONV(dst_desc->dims[1] == g * weights_desc->dims[with_groups + 0],
            VERBOSE_INCONSISTENT_DIM, "dst", 1, "weights", with_groups + 0);
    for (int i = 2; i < src_desc->ndims; ++i) {
        dim_t src = src_desc->dims[i];
        dim_t ker = weights_desc->dims[with_groups + i];
        dim_t dil = dd.dilates[i - 2];
        dim_t pad_l = padding_l[i - 2];
        dim_t pad_r = padding_r[i - 2];
        dim_t str = strides[i - 2];
        dim_t dst = dst_desc->dims[i];
        dim_t ker_range = 1 + (ker - 1) * (dil + 1);

        VCHECK_DECONV(str > 0, VERBOSE_BAD_DIM, "strides", i - 2);
        VCHECK_DECONV(dil >= 0, "%s: dilation (%d) must be non-negative",
                VERBOSE_INCONSISTENT_PRB, static_cast<int>(dil));
        VCHECK_DECONV(pad_l >= 0,
                "%s: left padding value (%d) must be non-negative",
                VERBOSE_INCONSISTENT_PRB, static_cast<int>(pad_l));
        VCHECK_DECONV(pad_r + str > 0,
                "%s: right padding (%d) and stride (%d) must sum up to a "
                "positive value",
                VERBOSE_INCONSISTENT_PRB, static_cast<int>(pad_r),
                static_cast<int>(str));
        VCHECK_DECONV((dst - ker_range + pad_l + pad_r) / str + 1 == src,
                "%s: mismatch between actual and computed src dims, src (%d) "
                "!= (dst(%d) - ker(%d) + pad_l(%d) + pad_r(%d))/ str(%d) + 1",
                VERBOSE_INCONSISTENT_PRB, static_cast<int>(src),
                static_cast<int>(dst), static_cast<int>(ker_range),
                static_cast<int>(pad_l), static_cast<int>(pad_r),
                static_cast<int>(str));
    }

    *deconv_desc = dd;
    return success;
}

status_t deconv_attr_check(const deconvolution_desc_t &desc,
        const engine_t *engine, const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    if (utils::one_of(desc.prop_kind, prop_kind::forward_inference,
                prop_kind::forward_training)) {
        const data_type_t src_dt = desc.src_desc.data_type;
        const data_type_t dst_dt = desc.dst_desc.data_type;

        auto fwd_attr_mask
                = smask_t::post_ops | smask_t::sum_dt | smask_t::fpmath_mode;

        bool is_int8 = utils::one_of(src_dt, data_type::s8, data_type::u8);
        if (engine->kind() == engine_kind::gpu)
            is_int8 = is_int8
                    || utils::one_of(dst_dt, data_type::s8, data_type::u8,
                            data_type::s32);
        if (is_int8) fwd_attr_mask |= smask_t::scales | smask_t::zero_points;

        VCHECK_DECONV_UNIMPL(attr->has_default_values(fwd_attr_mask, dst_dt),
                VERBOSE_UNSUPPORTED_ATTR);

        // Check scales
        if (!attr->scales_.has_default_values()) {
            VCHECK_DECONV_UNIMPL(attr->scales_.has_default_qmode(),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            const auto &sc = attr->scales_;
            const bool with_groups
                    = desc.src_desc.ndims != desc.weights_desc.ndims;
            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!sc.has_default_values(DNNL_ARG_SRC),
                            sc.get_mask(DNNL_ARG_SRC) == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!sc.has_default_values(DNNL_ARG_WEIGHTS),
                            utils::one_of(sc.get_mask(DNNL_ARG_WEIGHTS), 0,
                                    with_groups ? 3 : 1)),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!sc.has_default_values(DNNL_ARG_DST),
                            sc.get_mask(DNNL_ARG_DST) == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        // Check zero points
        if (!attr->zero_points_.has_default_values()) {
            VCHECK_DECONV_UNIMPL(attr->zero_points_.has_default_qmode(),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            const auto &zp = attr->zero_points_;

            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!zp.has_default_values(DNNL_ARG_SRC),
                            utils::one_of(
                                    zp.get_mask(DNNL_ARG_SRC), 0, 1 << 1)),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VCHECK_DECONV_UNIMPL(zp.has_default_values(DNNL_ARG_WEIGHTS),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VCHECK_DECONV_UNIMPL(
                    IMPLICATION(!zp.has_default_values(DNNL_ARG_DST),
                            utils::one_of(
                                    zp.get_mask(DNNL_ARG_DST), 0, 1 << 1)),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }

        // Check post-ops
        if (!attr->post_ops_.has_default_values()) {
            const auto &po = attr->post_ops_;
            using namespace primitive_kind;
            VCHECK_DECONV_UNIMPL(
                    po.has_default_values({binary, eltwise, prelu, sum}),
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Check sum
            VCHECK_DECONV_UNIMPL(
                    po.check_sum_consistency(dst_dt, is_int8, true),
                    VERBOSE_UNSUPPORTED_POSTOP);

            // Note: verbose support is inside the call.
            CHECK(po.validate_binary(engine->kind(), &desc.dst_desc));
        }
    } else {
        auto bwd_attr_mask = smask_t::fpmath_mode;
        VCHECK_DECONV_UNIMPL(attr->has_default_values(bwd_attr_mask),
                VERBOSE_UNSUPPORTED_ATTR);
    }

    return status::success;
}

} // namespace

status_t dnnl_deconvolution_forward_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        prop_kind_t prop_kind, alg_kind_t alg_kind,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r, const primitive_attr_t *attr) {
    if (!one_of(prop_kind, forward_training, forward_inference))
        return invalid_arguments;

    auto deconv_desc = deconvolution_desc_t();
    CHECK(deconv_desc_init(&deconv_desc, prop_kind, alg_kind, src_desc,
            weights_desc, bias_desc, dst_desc, strides, dilates, padding_l,
            padding_r));
    CHECK(deconv_attr_check(deconv_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&deconv_desc, nullptr, attr);
}

status_t dnnl_deconvolution_backward_data_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *diff_src_desc,
        const memory_desc_t *weights_desc, const memory_desc_t *diff_dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto deconv_desc = deconvolution_desc_t();
    CHECK(deconv_desc_init(&deconv_desc, backward_data, alg_kind, diff_src_desc,
            weights_desc, nullptr, diff_dst_desc, strides, dilates, padding_l,
            padding_r));
    CHECK(deconv_attr_check(deconv_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&deconv_desc, hint_fwd_pd, attr);
}

status_t dnnl_deconvolution_backward_weights_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        alg_kind_t alg_kind, const memory_desc_t *src_desc,
        const memory_desc_t *diff_weights_desc,
        const memory_desc_t *diff_bias_desc, const memory_desc_t *diff_dst_desc,
        const dims_t strides, const dims_t dilates, const dims_t padding_l,
        const dims_t padding_r, const primitive_desc_iface_t *hint_fwd_pd,
        const primitive_attr_t *attr) {

    auto deconv_desc = deconvolution_desc_t();
    CHECK(deconv_desc_init(&deconv_desc, backward_weights, alg_kind, src_desc,
            diff_weights_desc, diff_bias_desc, diff_dst_desc, strides, dilates,
            padding_l, padding_r));
    CHECK(deconv_attr_check(deconv_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&deconv_desc, hint_fwd_pd, attr);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
