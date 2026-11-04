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

#include "graph/backend/dnnl/executables/sum.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

sum_executable_t::desc_t sum_executable_t::create_desc(
        std::shared_ptr<op_t> &op, const dnnl::engine &p_engine,
        pd_cache_t &pd_cache, const graph_attr_t &graph_attr,
        bool use_block_layout) {
    if (pd_cache.find(op.get()) != pd_cache.end()) {
        auto pd = graph::utils::any_cast<dnnl::sum::primitive_desc>(
                pd_cache.at(op.get()));
        return {pd, true};
    }

    std::vector<dnnl::memory::desc> src_descs;
    src_descs.reserve(op->num_inputs());
    for (const auto &in_val : op->get_input_values()) {
        src_descs.emplace_back(
                make_dnnl_memory_desc(in_val->get_logical_tensor()));
    }

    auto dst_desc = make_dnnl_memory_desc(
            op->get_output_value(0)->get_logical_tensor());

    // create default scales
    std::vector<float> scales(op->num_inputs(), 1.f);

    dnnl::sum::primitive_desc pd(p_engine, dst_desc, scales, src_descs);

    pd_cache.insert({op.get(), pd});

    return {pd, false};
}

arg_indices_t sum_executable_t::get_arg_indices(const op_t *op) {
    return get_arg_indices_for_miso_op(op);
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
