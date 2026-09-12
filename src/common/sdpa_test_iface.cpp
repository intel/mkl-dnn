/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "common/sdpa_test_iface.hpp"

#include "common/c_types_map.hpp"
#include "common/opdesc.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/sdpa_pd.hpp"
#include "common/sdpa_types.hpp"
#include "common/sdpa_utils.hpp"

using namespace dnnl::impl;

status_t sdpa_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *query_desc, const memory_desc_t *key_desc,
        const memory_desc_t *value_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *mask_desc, const memory_desc_t *scale_desc,
        bool invert_scale, dim_t kv_head_number, int attn_mask_type,
        alg_kind_t softmax_alg, prop_kind_t prop, const primitive_attr_t *attr,
        const primitive_attr_t *kq_attr, const primitive_attr_t *vs_attr,
        const memory_desc_t *stats_desc) {
    CHECK(sdpa_desc_check(query_desc, key_desc, value_desc, dst_desc, mask_desc,
            engine, attr, kq_attr, vs_attr));
    CHECK(sdpa_attr_check(query_desc, key_desc, value_desc, dst_desc, engine,
            attr, kq_attr, vs_attr));
    sdpa_desc_t sdpa_desc = create_sdpa_desc(query_desc, key_desc, value_desc,
            dst_desc, mask_desc, scale_desc, stats_desc, invert_scale,
            kv_head_number, static_cast<attn_mask_type_t>(attn_mask_type),
            softmax_alg, prop, kq_attr, vs_attr);
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&sdpa_desc, nullptr, attr);
}

status_t sdpa_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *query_desc, const memory_desc_t *key_desc,
        const memory_desc_t *value_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *mask_desc, const memory_desc_t *scale_desc,
        const memory_desc_t *diff_query_desc,
        const memory_desc_t *diff_key_desc,
        const memory_desc_t *diff_value_desc,
        const memory_desc_t *diff_dst_desc, const memory_desc_t *dS_desc,
        bool invert_scale, dim_t kv_head_number, int attn_mask_type,
        alg_kind_t softmax_alg, const primitive_attr_t *attr,
        const primitive_desc_iface_t *hint_fwd_pd) {
    CHECK(sdpa_desc_check(query_desc, key_desc, value_desc, dst_desc, mask_desc,
            diff_query_desc, diff_key_desc, diff_value_desc, diff_dst_desc,
            engine, attr));
    CHECK(sdpa_attr_check(dst_desc, key_desc, engine, attr));

    sdpa_desc_t sdpa_desc = create_sdpa_desc(query_desc, key_desc, value_desc,
            dst_desc, mask_desc, scale_desc, diff_query_desc, diff_key_desc,
            diff_value_desc, diff_dst_desc, dS_desc, invert_scale,
            kv_head_number, static_cast<attn_mask_type_t>(attn_mask_type),
            softmax_alg);
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&sdpa_desc, hint_fwd_pd, attr);
}
