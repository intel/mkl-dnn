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

#include "graph/backend/dnnl/executables/group_norm.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

#define VCHECK_OP_EXECUTABLE(cond, msg, ...) \
    if (!(cond)) { VERROR(graph, op_executable, msg, ##__VA_ARGS__); }

groupnorm_executable_t::desc_t groupnorm_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {

    // first look up the cache
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<
                dnnl::group_normalization_forward::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    dnnl::primitive_attr prm_attr;
    if (op->has_attr(op_attr::fusion_info)) {
        const fusion_info_t &fusion_info
                = op->get_attr<fusion_info_t>(op_attr::fusion_info);
        prm_attr = make_dnnl_primitive_attr(op, fusion_info);
    }

    prm_attr.set_scratchpad_mode(dnnl::scratchpad_mode::user);
    float epsilon = 1e-5f;
    if (op->has_attr(op_attr::epsilon))
        epsilon = op->get_attr<float>(op_attr::epsilon);
    bool keep_stats = true;
    if (op->has_attr(op_attr::keep_stats))
        keep_stats = op->get_attr<bool>(op_attr::keep_stats);
    bool use_affine = true;
    if (op->has_attr(op_attr::use_affine))
        use_affine = op->get_attr<bool>(op_attr::use_affine);
    int64_t group_num = 1;
    if (op->has_attr(op_attr::groups)) {
        group_num = op->get_attr<int64_t>(op_attr::groups);
    } else {
        VCHECK_OP_EXECUTABLE(
                false, "groups attribute is required for groupnorm");
    }
    auto flags = dnnl::normalization_flags::none;
    if (use_affine)
        flags |= (dnnl::normalization_flags::use_scale
                | dnnl::normalization_flags::use_shift);

    prop_kind pkind = keep_stats ? prop_kind::forward_training
                                 : prop_kind::forward_inference;

    auto src = make_dnnl_memory_desc(op->get_input_logical_tensor(0));
    auto dst = make_dnnl_memory_desc(op->get_output_logical_tensor(0));

    dst = to_format_any(dst);

    dnnl::group_normalization_forward::primitive_desc pd(
            p_engine, pkind, src, dst, group_num, epsilon, flags, prm_attr);

    pd_cache.insert({op.get(), pd});
    return {pd, false};
}

arg_indices_t groupnorm_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_norm(op);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
