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

#include "graph/backend/dnnl/kernels/pool.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
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
status_t pooling_fwd_t<quantized>::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    // TODO(wuxun): since oneDNN pooling primitive only support u8u8 or
    // s8s8 on CPU device for now, we need to check whether the data types
    // between input and output are compatible. If we enable this check in
    // op schema or primitive supports u8s8/s8u8, then this check can be
    // safely removed.
    if (inputs[0].data_type != outputs[0].data_type)
        return status::unimplemented;

    p_engine_ = make_dnnl_engine(*eng);

    subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
            part->get_attr(), part->get_use_blocked_layout(), true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline(vis);

    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);

    BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);

    if (quantized) {
        BACKEND_DNNL_ADD_PASS(pipeline, lift_up_quantize);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_to_int8_pool);
        BACKEND_DNNL_ADD_PASS(pipeline, defer_src_zps_for_pool);
        BACKEND_DNNL_ADD_PASS(pipeline, combine_binary_post_op_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, fold_mul_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
        BACKEND_DNNL_ADD_PASS(pipeline, fold_sub_zps_add_zps);
        BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
        BACKEND_DNNL_ADD_PASS(pipeline, replace_quant_data_with_binary_post_op);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_runtime_mul_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_runtime_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dynamic_mul_scales_add_zps);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dynamic_sub_zps_mul_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_dynamic_quantize_ops);
    }

    BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
    BACKEND_DNNL_ADD_PASS(pipeline, pool_fwd_canonicalization);

    pipeline.reset_visualize_arg(true, false);
    // do constant propagation here so that we can
    // prepare constant info for other optimizations.
    if (enabled_constant_cache()) {
        BACKEND_DNNL_ADD_PASS(pipeline, constant_propagation);
    }
    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);
    // do constant propagation again since layout propagation may
    // insert/delete operators
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

    // generate a hash key for exec_args_mgr
    resource_ctor_ = [this]() {
        return this->memory_planner_.get_exec_args_set().clone();
    };

    const_md_hash_ = generate_constant_md_hash(part->id(),
            memory_planner_.get_exec_args_set().get_persistent_mem_desc_list());

    return status::success;
}

template <bool quantized>
void pooling_fwd_t<quantized>::prepare_args_set(const execution_args_set_t *res,
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
status_t pooling_fwd_t<quantized>::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    auto scratchpad = std::make_shared<scratchpad_t>(scratchpad_buf,
            memory_planner_.total_internal_temporary_size(), p_engine_);
    prepare_args_set(res, inputs, outputs, *scratchpad);

    constant_tensor_cache_t::cached_t c_buffer;
    if (enabled_constant_cache()) {
        const size_t encoded_key
                = encode_constant_cache_key(inputs, const_md_hash_);
        std::promise<constant_tensor_cache_t::cached_t> c_promise;
        constant_tensor_cache_t::value_t cached_value
                = dnnl_constant_cache_get_or_add(p_engine_, encoded_key,
                        memory_planner_.total_internal_persistent_size(),
                        c_promise.get_future());
        bool is_from_cache = cached_value.valid();
        if (is_from_cache) {
            c_buffer = cached_value.get();
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }
        } else {
            c_buffer = std::make_shared<dnnl_constant_buffer_t>(
                    memory_planner_.total_internal_persistent_size(),
                    *p_engine_.get());
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }

            for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
                if (!subgraph_->is_constant_[i]) continue;
                subgraph_->execs_[i]->execute(
                        p_stream, res->get_exec_args()[i]);
            }

            c_promise.set_value(c_buffer);
        }
    }

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        if (subgraph_->is_constant_[i]) continue;
        subgraph_->execs_[i]->execute(p_stream, res->get_exec_args()[i]);
    }

    prolong_scratchpad_lifetime(strm, scratchpad);

    return status::success;
}

#ifdef DNNL_WITH_SYCL
template <bool quantized>
status_t pooling_fwd_t<quantized>::sycl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {

    auto deps = sycl_deps;
    std::optional<::sycl::event> returned_event;
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    auto scratchpad = std::make_shared<scratchpad_t>(scratchpad_buf,
            memory_planner_.total_internal_temporary_size(), p_engine_);
    prepare_args_set(res, inputs, outputs, *scratchpad);

    constant_tensor_cache_t::cached_t c_buffer;
    if (enabled_constant_cache()) {
        const size_t encoded_key
                = encode_constant_cache_key(inputs, const_md_hash_);
        std::promise<constant_tensor_cache_t::cached_t> c_promise;
        constant_tensor_cache_t::value_t cached_value
                = dnnl_constant_cache_get_or_add(p_engine_, encoded_key,
                        memory_planner_.total_internal_persistent_size(),
                        c_promise.get_future());
        bool is_from_cache = cached_value.valid();
        if (is_from_cache) {
            c_buffer = cached_value.get();
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }
        } else {
            c_buffer = std::make_shared<dnnl_constant_buffer_t>(
                    memory_planner_.total_internal_persistent_size(),
                    *p_engine_.get());
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }

            for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
                if (!subgraph_->is_constant_[i]) continue;
                returned_event = subgraph_->execs_[i]->execute_sycl(
                        p_stream, res->get_exec_args()[i], deps);
                if (returned_event) deps = {*returned_event};
            }

            c_promise.set_value(c_buffer);
        }
    }

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
template <bool quantized>
status_t pooling_fwd_t<quantized>::ocl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<ocl_event_t> &cl_deps, ocl_event_t &ret_event) {

    auto deps = cl_deps;
    ocl_event_t returned_event;
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    auto scratchpad = std::make_shared<scratchpad_t>(scratchpad_buf,
            memory_planner_.total_internal_temporary_size(), p_engine_);
    prepare_args_set(res, inputs, outputs, *scratchpad);

    constant_tensor_cache_t::cached_t c_buffer;
    if (enabled_constant_cache()) {
        const size_t encoded_key
                = encode_constant_cache_key(inputs, const_md_hash_);
        std::promise<constant_tensor_cache_t::cached_t> c_promise;
        constant_tensor_cache_t::value_t cached_value
                = dnnl_constant_cache_get_or_add(p_engine_, encoded_key,
                        memory_planner_.total_internal_persistent_size(),
                        c_promise.get_future());
        bool is_from_cache = cached_value.valid();
        if (is_from_cache) {
            c_buffer = cached_value.get();
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }
        } else {
            c_buffer = std::make_shared<dnnl_constant_buffer_t>(
                    memory_planner_.total_internal_persistent_size(),
                    *p_engine_.get());
            grantor_t c_grantor = memory_planner_.internal_persistent_grantor(
                    c_buffer->data<char>());
            for (auto &mem_offkey : res->get_mems_use_internal_persistent()) {
                mem_offkey.first.set_data_handle(
                        c_grantor.get(mem_offkey.second));
            }

            for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
                if (!subgraph_->is_constant_[i]) continue;
                returned_event = subgraph_->execs_[i]->execute_ocl(
                        p_stream, res->get_exec_args()[i], deps);
                deps = {returned_event};
            }

            c_promise.set_value(c_buffer);
        }
    }

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

#if BUILD_TRAINING
status_t pooling_bwd_t::compile_impl(const dnnl_partition_impl_t *part,
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

    BACKEND_DNNL_ADD_PASS(pipeline, pool_fwd_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, pool_bwd_canonicalization);

    pipeline.reset_visualize_arg(true, false);
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

    // generate a hash key for exec_args_mgr
    resource_ctor_ = [this]() {
        return this->memory_planner_.get_exec_args_set().clone();
    };

    return status::success;
}

void pooling_bwd_t::prepare_args_set(const execution_args_set_t *res,
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

status_t pooling_bwd_t::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    // each thread's own local resource
    thread_local_cache_t<execution_args_set_t> res_cache;
    execution_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    auto scratchpad = std::make_shared<scratchpad_t>(scratchpad_buf,
            memory_planner_.total_internal_temporary_size(), p_engine_);
    prepare_args_set(res, inputs, outputs, *scratchpad);

    for (size_t i = 0; i < subgraph_->execs_.size(); i++) {
        if (subgraph_->is_constant_[i]) continue;
        subgraph_->execs_[i]->execute(p_stream, res->get_exec_args()[i]);
    }

    prolong_scratchpad_lifetime(strm, scratchpad);

    return status::success;
}

#ifdef DNNL_WITH_SYCL
status_t pooling_bwd_t::sycl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<::sycl::event> &sycl_deps,
        ::sycl::event *sycl_event) {

    auto deps = sycl_deps;
    std::optional<::sycl::event> returned_event;
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    // each thread's own local resource
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
status_t pooling_bwd_t::ocl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<ocl_event_t> &cl_deps, ocl_event_t &ret_event) {

    auto deps = cl_deps;
    ocl_event_t returned_event;
    dnnl::stream p_stream = make_dnnl_stream(*strm);

    // each thread's own local resource
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
#endif // BUILD_TRAINING

template struct pooling_fwd_t<false>;
template struct pooling_fwd_t<true>;
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
