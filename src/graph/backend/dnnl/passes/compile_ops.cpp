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

#include <memory>
#include <vector>
#include <unordered_map>

#include "graph/interface/c_types_map.hpp"
#include "graph/interface/value.hpp"

#include "graph/backend/dnnl/op_executable.hpp"
#include "graph/backend/dnnl/passes/compile_ops.hpp"

#include "oneapi/dnnl/dnnl.hpp"

#define VCHECK_COMPILE_OPS(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, compile_ops, (cond), status, msg, \
            ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
/// After the lower down, infer shape, infer type and layout propagation passes,
/// each op in the subgraph will has complete attributes and each edge will have
/// complete shape/dtype/layout information. We can create executable for these
/// ops.
status_t compile_ops(std::shared_ptr<subgraph_t> &sg) {
    const auto &p_engine = *(sg->p_engine_);
    auto &pd_cache = sg->pd_cache_;
    auto &graph_attr = sg->get_attr();
    bool use_block_layout = sg->can_use_blocked_layout_;

    return topo_order_visit(sg->get_output_ops(), [&](op_t *op) {
        auto creator = op_func_t::get_executable_creator(op->get_kind());
        VCHECK_COMPILE_OPS(creator != nullptr, status::invalid_graph_op,
                "no executable creator in schema of op %s",
                op->get_name().c_str());
        auto cur_op = op->shared_from_this();
        std::shared_ptr<op_executable_t> exec = creator(
                cur_op, p_engine, pd_cache, graph_attr, use_block_layout);
        VCHECK_COMPILE_OPS(exec != nullptr, status::invalid_graph_op,
                "unimplemented op, can't compile op %s",
                op->get_name().c_str());
        VCHECK_COMPILE_OPS(exec->is_initialized(), status::invalid_graph_op,
                "failed to create executable for op %s",
                op->get_name().c_str());

        sg->execs_.emplace_back(exec);

        sg->is_constant_.push_back(op->has_attr(op_attr::is_constant)
                && op->get_attr<bool>(op_attr::is_constant));
        return status::success;
    });
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
