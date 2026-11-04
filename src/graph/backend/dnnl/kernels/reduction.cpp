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

#include "graph/backend/dnnl/kernels/reduction.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "graph/backend/dnnl/op_executable.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

template <bool quantized>
status_t reduction_t<quantized>::compile_impl(const dnnl_partition_impl_t *part,
        engine_t *eng, const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*eng);

    subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
            part->get_attr(), part->get_use_blocked_layout(), true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline(vis);

    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_mul_sigmoid_to_swish);
    BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);

    BACKEND_DNNL_ADD_PASS(pipeline, insert_unsqueeze_and_squeeze_for_reduction);

    BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);

    pipeline.reset_visualize_arg(true, false);

    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

    auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
        return memory_planner_.run(sg);
    };
    pipeline.reset_visualize_arg(true, true);
    BACKEND_DNNL_ADD_PASS(pipeline, memory_plan);
    BACKEND_DNNL_ADD_PASS(pipeline, compile_ops);

    BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

    for (size_t i = 0; i < inputs.size(); i++) {
        auto &in = const_cast<logical_tensor_t &>(inputs[i]);
        in = subgraph_->ins_[i];
    }

    for (size_t i = 0; i < outputs.size(); i++) {
        auto &out = const_cast<logical_tensor_t &>(outputs[i]);
        out = subgraph_->outs_[i];
    }

    resource_ctor_ = [this]() {
        return this->memory_planner_.get_exec_args_set().clone();
    };

    return status::success;
}

template <bool quantized>
void reduction_t<quantized>::prepare_args_set(const execution_args_set_t *res,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const scratchpad_t &scratchpad) {
    // update the data of partition in/outputs args
    for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
        mem_idx.first.set_data_handle(inputs[mem_idx.second].get_data_handle());
    }
    for (const auto &mem_idx : res->get_mems_use_external_outputs()) {
        mem_idx.first.set_data_handle(
                outputs[mem_idx.second].get_data_handle());
    }

    grantor_t var_grantor = memory_planner_.internal_temporary_grantor(
            scratchpad.get_buffer());

    for (auto &mem_offkey : res->get_mems_use_internal_temporary()) {
        mem_offkey.first.set_data_handle(var_grantor.get(mem_offkey.second));
    }
}

template <bool quantized>
status_t reduction_t<quantized>::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    auto scratchpad = std::make_shared<scratchpad_t>(scratchpad_buf,
            memory_planner_.total_internal_temporary_size(), p_engine_);
    prepare_args_set(res, inputs, outputs, *scratchpad);

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        subgraph_->execs_[i]->execute(p_stream, res->get_exec_args()[i]);
    }

    prolong_scratchpad_lifetime(strm, scratchpad);

    return status::success;
}

#ifdef DNNL_WITH_SYCL
template <bool quantized>
status_t reduction_t<quantized>::sycl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {

    auto deps = sycl_deps;
    std::optional<::sycl::event> returned_event;
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    auto scratchpad = std::make_shared<scratchpad_t>(scratchpad_buf,
            memory_planner_.total_internal_temporary_size(), p_engine_);
    prepare_args_set(res, inputs, outputs, *scratchpad);

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        returned_event = subgraph_->execs_[i]->execute_sycl(
                p_stream, res->get_exec_args()[i], deps);
        if (returned_event) deps = {*returned_event};
    }

    scratchpad->set_deps(returned_event ? *returned_event : ::sycl::event {});
    if (sycl_event)
        *sycl_event = returned_event ? *returned_event : ::sycl::event {};

    return status::success;
}
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
template <bool quantized>
status_t reduction_t<quantized>::ocl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<ocl_event_t> &cl_deps, ocl_event_t &ret_event) {

    auto deps = cl_deps;
    ocl_event_t returned_event;
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    auto scratchpad = std::make_shared<scratchpad_t>(scratchpad_buf,
            memory_planner_.total_internal_temporary_size(), p_engine_);
    prepare_args_set(res, inputs, outputs, *scratchpad);

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        returned_event = subgraph_->execs_[i]->execute_ocl(
                p_stream, res->get_exec_args()[i], deps);
        deps = {returned_event};
    }

    scratchpad->set_deps(returned_event.get());
    ret_event = std::move(returned_event);

    return status::success;
}
#endif

template struct reduction_t<false>;
template struct reduction_t<true>;
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
