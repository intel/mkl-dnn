/*******************************************************************************
 * Copyright 2024 Intel Corporation
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

#include "graph/backend/dnnl/dnnl_partition_impl.hpp"

#include "graph/backend/dnnl/kernels/kernels.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

namespace {
status_t get_ordered_inputs_outputs(
        const std::vector<logical_tensor_t> &expected,
        const std::vector<logical_tensor_t> &given,
        std::vector<logical_tensor_t> &ordered) {
    ordered.reserve(expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        for (size_t j = 0; j < given.size(); j++) {
            if (expected[i].id == given[j].id) {
                ordered.emplace_back(given[j]);
                break;
            }
        }
    }

    if (ordered.size() != expected.size()) return status::invalid_arguments;
    return status::success;
}
} // namespace

void dnnl_partition_impl_t::init(FCreateKernel kernel_creator) {
    init_inputs_outputs();

    // init kernel
    kernel_creator_ = std::move(kernel_creator);
}

void dnnl_partition_impl_t::add_op(const std::shared_ptr<op_t> &op) {
    ops_.emplace_back(op);
}

void dnnl_partition_impl_t::init_inputs_outputs() {
    inputs_.clear();
    outputs_.clear();
    std::unordered_set<op_t *> visit;
    for (auto &cur_op : ops_) {
        visit.insert(cur_op.get());
    }

    for (auto &cur_op : ops_) {
        // Handle End op: directly mark its input as output
        if (cur_op->get_kind() == graph::op_kind::End) {
            outputs_.push_back(cur_op->get_input_logical_tensor(0));
            continue;
        }

        for (size_t j = 0; j < cur_op->num_inputs(); ++j) {
            auto in_value = cur_op->get_input_value(j);
            if (!in_value->has_producer()
                    || !visit.count(&in_value->get_producer())) {
                inputs_.push_back(in_value->get_logical_tensor());
            }
        }
        for (size_t j = 0; j < cur_op->num_outputs(); ++j) {
            auto out_value = cur_op->get_output_value(j);
            // if out_value has no consumer
            // OR any of its consumers are not inside the pattern
            // it is output tensor
            bool is_output = out_value->get_consumers().empty();
            for (auto &consumer : out_value->get_consumers()) {
                if (!visit.count(&consumer.get_op())) {
                    is_output = true;
                    break;
                }
            }
            if (is_output) {
                outputs_.push_back(out_value->get_logical_tensor());
            }
        }
    }
}

FCreateKernel dnnl_partition_impl_t::get_kernel_creator() const {
    return kernel_creator_;
}

std::shared_ptr<partition_impl_t> dnnl_partition_impl_t::clone() const {
    auto ret = std::make_shared<dnnl_partition_impl_t>(
            get_engine_kind(), get_attr(), get_kind());
    ret->ops_ = graph_t::deep_copy(ops_);
    ret->inputs_ = inputs_;
    ret->outputs_ = outputs_;
    ret->kernel_creator_ = kernel_creator_;
    ret->id_ = id_;
    ret->can_use_blocked_layout_ = can_use_blocked_layout_;
    return ret;
}

const backend_t *dnnl_partition_impl_t::get_assigned_backend() const {
    return &dnnl_backend_t::get_singleton();
}

status_t dnnl_partition_impl_t::compile(
        compiled_partition_t *compiled_partition,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs, engine_t *eng) const {
    // compile will transform the subgraph in partition, so we make
    // a copy
    auto part = std::dynamic_pointer_cast<dnnl_partition_impl_t>(this->clone());

    // get kernel creator
    auto kernel_creator = part->get_kernel_creator();

    // This internal env var is used for test purpose. When setting
    // _DNNL_USE_LARGE_PARTITION_KERNEL to 1, all partitions will be
    // dispatched to the large partition kernel.
    if (graph::utils::getenv_int_internal("USE_LARGE_PARTITION_KERNEL", 0)) {
        kernel_creator = large_partition_kernel_creator;
    }

    // Dispatch to fake kernel if one of the output dimensions is zero.
    const std::vector<std::shared_ptr<op_t>> &fused_op = part->get_ops();
    auto agraph = graph_t(fused_op, get_engine_kind(), get_attr());
    agraph.set_user_inputs_outputs(inputs, outputs);
    agraph.infer_shape();
    for (const auto &val : agraph.get_output_values()) {
        if (logical_tensor_wrapper_t(val->get_logical_tensor())
                        .has_zero_dim()) {
            kernel_creator = dummy_kernel_creator;
            break;
        }
    }

    kernel_ptr kernel = kernel_creator();
    if (!kernel) return status::unimplemented;

    status_t ret;

    // compile kernel.
    // FIXME(qun) will modify the outputs inside the compile, which
    // break the constant semantics
    ret = kernel->compile(part.get(), eng, inputs, outputs);
    if (ret != status::success) return ret;

    std::vector<logical_tensor_t> ordered_inputs;
    std::vector<logical_tensor_t> ordered_outputs;
    ret = get_ordered_inputs_outputs(inputs_, inputs, ordered_inputs);
    if (status::success != ret) return ret;

    ret = get_ordered_inputs_outputs(outputs_, outputs, ordered_outputs);
    if (status::success != ret) return ret;

    // wrapper kernel to dnnl_compiled_partition_impl_t
    auto pimpl = std::make_shared<dnnl_compiled_partition_impl_t>(
            eng, ordered_inputs, ordered_outputs, kernel);
    compiled_partition->init(pimpl);

    return status::success;
}

status_t dnnl_partition_impl_t::infer_shape(
        std::vector<const logical_tensor_t *> &inputs,
        std::vector<logical_tensor_t *> &outputs) const {
    UNUSED(inputs);
    UNUSED(outputs);
    return status::success;
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
