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
#include "graph/backend/dnnl/kernels/gen_index.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "graph/backend/dnnl/op_executable.hpp"

#define VCHECK_GENINDEX(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, genindex_t, (cond), status, msg, \
            ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t genindex_t::compile_impl(const dnnl_partition_impl_t *part,
        engine_t *eng, const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*eng);

    subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
            part->get_attr(), part->get_use_blocked_layout(), true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    if (p_engine_.get_kind() == engine::kind::gpu) {
#if (DNNL_GPU_RUNTIME != DNNL_RUNTIME_NONE) \
        && (DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL)
        int ndims = inputs[0].ndims;
        VCHECK_GENINDEX(ndims <= MAX_NDIMS, status::invalid_arguments,
                "only tensors of 6 or fewer dimensions are supported for "
                "genindex GPU, but got %dD",
                ndims);
#else
        return status::unimplemented;
#endif
    }

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline(vis);

    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);

    pipeline.reset_visualize_arg(true, false);

    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

    // constant propagation
    if (enabled_constant_cache()) {
        BACKEND_DNNL_ADD_PASS(pipeline, constant_propagation);
    }

    // bind the memory for each op
    auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
        return memory_planner_.run(sg);
    };
    pipeline.reset_visualize_arg(true, true);
    BACKEND_DNNL_ADD_PASS(pipeline, memory_plan);
    BACKEND_DNNL_ADD_PASS(pipeline, compile_ops);

    // Run the added passes
    BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

    // fill information for outputs logical tensors
    for (size_t i = 0; i < outputs.size(); i++) {
        auto &out = const_cast<logical_tensor_t &>(outputs[i]);
        out = subgraph_->outs_[i];
    }

    // generate a hash key for exec_args_mgr
    resource_ctor_ = [this]() {
        return this->memory_planner_.get_exec_args_set().clone();
    };

    return status::success;
}

void genindex_t::prepare_args_set(const execution_args_set_t *res,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs) {
    // update the data of partition in/outputs args
    for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
        mem_idx.first.set_data_handle(inputs[mem_idx.second].get_data_handle());
    }
    for (const auto &mem_idx : res->get_mems_use_external_outputs()) {
        mem_idx.first.set_data_handle(
                outputs[mem_idx.second].get_data_handle());
    }
}

status_t genindex_t::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);
    prepare_args_set(res, inputs, outputs);

    constant_tensor_cache_t::cached_t c_buffer;

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        if (subgraph_->is_constant_[i]) continue;
        subgraph_->execs_[i]->execute(p_stream, res->get_exec_args()[i]);
    }

    return status::success;
}
#ifdef DNNL_WITH_SYCL
status_t genindex_t::sycl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {
    if (p_engine_.get_kind() == engine::kind::gpu) {
        auto deps = sycl_deps;
        std::optional<::sycl::event> returned_event;
        dnnl::stream p_stream = make_dnnl_stream(*strm);

        thread_local_cache_t<execution_args_set_t> res_cache;
        execution_args_set_t *res = res_cache.get_or_add(
                reinterpret_cast<size_t>(this), resource_ctor_);
        prepare_args_set(res, inputs, outputs);
        for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
            if (subgraph_->is_constant_[i]) continue;
            returned_event = subgraph_->execs_[i]->execute_sycl(
                    p_stream, res->get_exec_args()[i], deps);
            if (returned_event) deps = {*returned_event};
        }

        if (sycl_event)
            *sycl_event = returned_event ? *returned_event : ::sycl::event {};

        return status::success;
    }
    return execute_impl(strm, inputs, outputs, scratchpad_buf);
}
#endif
#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
status_t genindex_t::ocl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<ocl_event_t> &ocl_deps, ocl_event_t &ocl_event) {
    auto deps = ocl_deps;
    ocl_event_t returned_event;
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    prepare_args_set(res, inputs, outputs);

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        if (subgraph_->is_constant_[i]) continue;
        returned_event = subgraph_->execs_[i]->execute_ocl(
                p_stream, res->get_exec_args()[i], deps);
        deps = {returned_event};
    }

    ocl_event = std::move(returned_event);

    return status::success;
}
#endif
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
