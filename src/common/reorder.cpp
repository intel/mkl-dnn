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

#include <assert.h>
#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "engine.hpp"
#include "impl_list_item.hpp"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_desc_iterator.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::status;

namespace dnnl {
namespace impl {

#define VCHECK_REORDER(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, reorder, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_REORDER_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, reorder, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace {
const engine_t *get_reorder_engine(
        const engine_t *src_engine, const engine_t *dst_engine) {
    auto s_ek = src_engine->kind();
    auto d_ek = dst_engine->kind();
    auto s_rk = src_engine->runtime_kind();
    auto d_rk = dst_engine->runtime_kind();

    if (is_native_runtime(d_rk)) return src_engine;

    if (is_native_runtime(s_rk)) return dst_engine;

    if (d_ek == engine_kind::cpu) return src_engine;

    if (s_ek == engine_kind::cpu) return dst_engine;

    assert(s_ek == engine_kind::gpu);
    assert(d_ek == engine_kind::gpu);
    return src_engine;
}

// `attr` must be resolved (non-null) by the caller to keep the two public
// entries the single owner of the default-attributes fallback.
status_t reorder_desc_init(reorder_desc_t *reorder_desc,
        const memory_desc_t *src_md, const engine_t *src_engine,
        const memory_desc_t *dst_md, const engine_t *dst_engine,
        const primitive_attr_t *attr) {
    reorder_desc_t desc;

    auto s_ek = src_engine->kind();
    auto d_ek = dst_engine->kind();
    VCHECK_REORDER(IMPLICATION(s_ek != d_ek,
                           utils::one_of(engine_kind::cpu, s_ek, d_ek)),
            VERBOSE_BAD_ENGINE_KIND);
    // There are no sparse reorders for GPU engine.
    if (utils::one_of(engine_kind::gpu, s_ek, d_ek)
            && !impl::is_dense_format_kind({src_md, dst_md}))
        return status::unimplemented;

    desc.src_engine_kind = s_ek;
    desc.dst_engine_kind = d_ek;
    desc.is_cross_engine = src_engine != dst_engine
            && utils::one_of(engine_kind::gpu, s_ek, d_ek);

    VCHECK_REORDER(!memory_desc_wrapper(src_md).format_any(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    desc.src_desc = *src_md;

    VCHECK_REORDER(!memory_desc_wrapper(dst_md).format_any(),
            VERBOSE_UNSUPPORTED_TAG_S, "dst");
    desc.dst_desc = *dst_md;

    auto s_mdw = memory_desc_wrapper(*src_md);
    auto d_mdw = memory_desc_wrapper(*dst_md);
    VCHECK_REORDER(s_mdw.consistent_with(d_mdw), VERBOSE_INCONSISTENT_MDS,
            "src", "dst");

    // Zero points are only allowed for integral data types
    const auto &zero_points = attr->zero_points_;
    VCHECK_REORDER(IMPLICATION(!types::is_integral_dt(src_md->data_type),
                           zero_points.has_default_values(DNNL_ARG_SRC)),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    VCHECK_REORDER(IMPLICATION(!types::is_integral_dt(dst_md->data_type),
                           zero_points.has_default_values(DNNL_ARG_DST)),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    // Conflicting destination accumulation semantics, and the combination
    // lacks a practical use case.
    VCHECK_REORDER(IMPLICATION(attr->post_ops_.find(primitive_kind::sum) != -1,
                           zero_points.has_default_values(DNNL_ARG_DST)),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    VCHECK_REORDER_UNIMPL(
            zero_points.has_default_qmode(), VERBOSE_UNSUPPORTED_ZP_CFG);

    // Check scales
    if (!attr->scales_.has_default_values()) {
        VCHECK_REORDER_UNIMPL(attr->scales_.has_default_qmode(),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        static const std::vector<int> supported_args {
                DNNL_ARG_SRC, DNNL_ARG_DST};
        VCHECK_REORDER_UNIMPL(attr->scales_.has_default_values(supported_args),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        const auto &sc = attr->scales_;
        const auto &sc_src = sc.get(DNNL_ARG_SRC);
        const int mask_src = sc.get_mask(DNNL_ARG_SRC);

        VCHECK_REORDER(IMPLICATION(utils::one_of(src_md->data_type,
                                           data_type::s4, data_type::u4),
                               mask_src > 0),
                VERBOSE_INVALID_DATATYPE, "mask for int4 source");

        if (!sc_src.has_default_groups()) {
            const int src_ndims = s_mdw.ndims();
            const bool group_dims_are_consistent
                    = IMPLICATION(sc_src.get_group(0) > 1,
                              src_md->dims[src_ndims - 2] % sc_src.get_group(0)
                                      == 0)
                    && IMPLICATION(sc_src.get_group(1) > 1,
                            src_md->dims[src_ndims - 1] % sc_src.get_group(1)
                                    == 0);
            VCHECK_REORDER(group_dims_are_consistent,
                    "groups dimensions are not consistent with reorder "
                    "dimensions");

            // Groups are always applied to last two dimensions. Check that
            // input scale mask is consistent with this limitation.
            const bool mask_applies_to_last_two_dims
                    = (mask_src & (1 << (src_ndims - 1)))
                    && (mask_src & (1 << (src_ndims - 2)));
            VCHECK_REORDER(mask_applies_to_last_two_dims,
                    "mask is not consistent with groups");
        }

        VCHECK_REORDER(sc.get(DNNL_ARG_DST).has_default_groups(),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    *reorder_desc = desc;
    return status::success;
}
} // namespace

status_t reorder_primitive_desc_create(std::shared_ptr<primitive_desc_t> &pd,
        const engine_t *engine, const memory_desc_t *src_md,
        const engine_t *src_engine, const memory_desc_t *dst_md,
        const engine_t *dst_engine, const primitive_attr_t *attr) {
    pd.reset();

    if (attr == nullptr) attr = &default_attr();

    reorder_desc_t reorder_desc;
    CHECK(reorder_desc_init(
            &reorder_desc, src_md, src_engine, dst_md, dst_engine, attr));

    primitive_desc_iterator_t it(engine,
            reinterpret_cast<const op_desc_t *>(&reorder_desc), attr, nullptr,
            /* skip_idx = */ -1, src_engine, dst_engine);
    if (!it.is_initialized()) return out_of_memory;

    while (++it != it.end()) {
        if (*it) {
            pd = *it;
            return success;
        }
    }
    return unimplemented;
}

status_t reorder_primitive_desc_create(std::shared_ptr<primitive_desc_t> &pd,
        const engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *dst_md, const primitive_attr_t *attr) {
    return reorder_primitive_desc_create(
            pd, engine, src_md, engine, dst_md, engine, attr);
}

} // namespace impl
} // namespace dnnl

status_t dnnl_reorder_primitive_desc_create(
        primitive_desc_iface_t **reorder_pd_iface, const memory_desc_t *src_md,
        engine_t *src_engine, const memory_desc_t *dst_md, engine_t *dst_engine,
        const primitive_attr_t *attr) {
    if (any_null(reorder_pd_iface, src_engine, src_md, dst_engine, dst_md))
        return invalid_arguments;

    if (attr == nullptr) attr = &default_attr();

    reorder_desc_t reorder_desc;
    CHECK(reorder_desc_init(
            &reorder_desc, src_md, src_engine, dst_md, dst_engine, attr));

    auto *e = const_cast<engine_t *>(
            get_reorder_engine(src_engine, dst_engine));
    return primitive_desc_create(reorder_pd_iface, e,
            reinterpret_cast<const op_desc_t *>(&reorder_desc), nullptr, attr,
            src_engine, dst_engine);
}

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
