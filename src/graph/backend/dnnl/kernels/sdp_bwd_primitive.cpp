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

#include "graph/backend/dnnl/kernels/sdp_bwd_primitive.hpp"

#include "common/sdpa_pd.hpp"

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "gpu/intel/ocl/stream.hpp"
#elif DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL
#include "gpu/intel/sycl/stream.hpp"
#endif

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "graph/backend/dnnl/op_executable.hpp"

#include "common/verbose.hpp"

#define VCHECK_SDP_BWD_PRIMITIVE(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_bwd_primitive_kernel_t, (cond), \
            status, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t sdp_bwd_primitive_kernel_t::initial_check(
        const std::shared_ptr<subgraph_t> &sg,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    const bool is_f32 = inputs[0].data_type == data_type::f32;
    VCHECK_SDP_BWD_PRIMITIVE(!is_f32, status::unimplemented,
            "SDPA bwd primitive doesn't support f32 because of performance");
    return status::success;
}

status_t sdp_bwd_primitive_kernel_t::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
// sdp_bwd_primitive_kernel_t only supports Intel GPU.
#if defined(DNNL_WITH_SYCL) && DNNL_GPU_VENDOR != DNNL_VENDOR_INTEL
    return status::unimplemented;
#endif

    p_engine_ = make_dnnl_engine(*eng);

    // First, dry run on a deep copy
    subgraph_
            = std::make_shared<subgraph_t>(graph_t::deep_copy(part->get_ops()),
                    p_engine_, part->get_attr(), false, true);
    CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));
    CHECK(initial_check(subgraph_, inputs, outputs));

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline = pass_pipeline_t(vis);

    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_implicit_causal_mask);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);

    pipeline.reset_visualize_arg(true, false);
    BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_sdpa_bwd);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_sdpa_bwd);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_reshape_for_sdpa_bwd);
    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

    // bind the memory for each op
    auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
        return memory_planner_.run(sg);
    };
    pipeline.reset_visualize_arg(true, true);
    BACKEND_DNNL_ADD_PASS(pipeline, memory_plan);
    BACKEND_DNNL_ADD_PASS(pipeline, compile_ops);

    // Run the added passes
    BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

    // fill information for inputs logical tensors
    for (size_t i = 0; i < inputs.size(); i++) {
        auto &in = const_cast<logical_tensor_t &>(inputs[i]);
        in = subgraph_->ins_[i];
    }

    // fill information for outputs logical tensors
    for (size_t i = 0; i < outputs.size(); i++) {
        auto &out = const_cast<logical_tensor_t &>(outputs[i]);
        out = subgraph_->outs_[i];
    }

    resource_ctor_ = [this]() {
        return this->memory_planner_.get_exec_args_set().clone();
    };

    return status::success;
}

void sdp_bwd_primitive_kernel_t::prepare_args_set(
        const execution_args_set_t *res, const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const scratchpad_t &scratchpad) {
    // update the data of partition in/outputs args
    for (const auto &mem_idx : res->get_mems_use_external_inputs()) {
        const dnnl::memory &mem = mem_idx.first;
        const tensor_t &ts = inputs[mem_idx.second];
        const logical_tensor_t lt = ts.get_logical_tensor();
        const logical_tensor_wrapper_t ltw(lt);
        if (ltw.is_host_scalar()) {
            DNNL_HOST_SCALAR_TYPE_SWITCH(ltw.data_type(), DType, {
                mem.set_host_scalar_value(
                        *static_cast<DType *>(ts.get_data_handle()));
            });
        } else {
            mem.set_data_handle(ts.get_data_handle());
        }
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

status_t sdp_bwd_primitive_kernel_t::execute_impl(stream_t *strm,
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

    return status::success;
}

#ifdef DNNL_WITH_SYCL
status_t sdp_bwd_primitive_kernel_t::sycl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {
// sdp_bwd_primitive_kernel_t only supports Intel GPU.
#if DNNL_GPU_VENDOR != DNNL_VENDOR_INTEL
    return status::unimplemented;
#endif
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
        if (subgraph_->is_constant_[i]) continue;
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
status_t sdp_bwd_primitive_kernel_t::ocl_execute_impl(stream_t *strm,
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
        if (subgraph_->is_constant_[i]) continue;
        returned_event = subgraph_->execs_[i]->execute_ocl(
                p_stream, res->get_exec_args()[i], deps);
        deps = {returned_event};
    }

    scratchpad->set_deps(returned_event.get());
    ret_event = std::move(returned_event);

    return status::success;
}
#endif

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
