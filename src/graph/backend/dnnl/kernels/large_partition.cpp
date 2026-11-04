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

#include "graph/backend/dnnl/kernels/large_partition.hpp"

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

void larger_partition_kernel_t::setup_pipeline_stage1(
        pass_pipeline_t &pipeline) {
    // Directly lower down (1 to 1 mapping)
    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);

    // handle the case that the input is a scalar tensor
    BACKEND_DNNL_ADD_PASS(pipeline, insert_host_scalar);

    // Indirectly lower down (N to 1 mapping)
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_reciprocal_mul_to_div);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_mul_sigmoid_to_swish);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_to_dnnl_sum);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_to_shuffle);
    BACKEND_DNNL_ADD_PASS(pipeline, decompose_softmax_with_stats);

    // TODO(xx) The implementation of these two passes relay on a non-fully
    // lowered subgraph. We need to improve them.
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_to_int8_concat);

    BACKEND_DNNL_ADD_PASS(pipeline, lift_up_weight_reshape_for_depthwiseconv);
    // Fusion and canonicalization passes begin
    BACKEND_DNNL_ADD_PASS(pipeline, lift_up_typecast);
    BACKEND_DNNL_ADD_PASS(pipeline, lift_up_quantize);

    BACKEND_DNNL_ADD_PASS(pipeline, fuse_bias_add);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_bn_folding);
    BACKEND_DNNL_ADD_PASS(pipeline, check_with_bias);

    BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, binary_broadcast_swap);

    BACKEND_DNNL_ADD_PASS(pipeline, fuse_typecast_to_matmul_or_conv);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_typecast_to_add);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_typecast_to_predecessor);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_typecast_to_mul_scales);
    BACKEND_DNNL_ADD_PASS(
            pipeline, insert_permute_for_dynamic_mul_scale_sub_zp);

    BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);

    BACKEND_DNNL_ADD_PASS(pipeline, convert_bias_to_f32);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_to_int8_pool);

    BACKEND_DNNL_ADD_PASS(pipeline, combine_binary_post_op_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_src_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_src_zero_points);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_zero_points);
    // tricky here.
    BACKEND_DNNL_ADD_PASS(pipeline, insert_runtime_u8_to_s8_for_matmul);

    BACKEND_DNNL_ADD_PASS(pipeline, insert_unsqueeze_and_squeeze_for_reduction);
    // bnorm here.
    BACKEND_DNNL_ADD_PASS(pipeline, swap_relu_mul_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, fold_pre_mul_scale_into_bn);
    BACKEND_DNNL_ADD_PASS(pipeline, fold_post_mul_scale_into_bn);

    // MQA pattern fusion
    BACKEND_DNNL_ADD_PASS(pipeline, lift_up_post_add_for_matmul);

    BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dropout);
    BACKEND_DNNL_ADD_PASS(pipeline, fold_mul_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_zero_points);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_zero_points);

    BACKEND_DNNL_ADD_PASS(pipeline, defer_src_zps_for_pool);
    BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
    BACKEND_DNNL_ADD_PASS(pipeline, fold_sub_zps_add_zps);
    BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
    BACKEND_DNNL_ADD_PASS(pipeline, replace_quant_data_with_binary_post_op);

    // fuse those new post-binaries converted from add_zps and mul_scales
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);

    BACKEND_DNNL_ADD_PASS(pipeline, convert_runtime_mul_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, convert_runtime_zero_points);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dynamic_mul_scales_add_zps);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dynamic_sub_zps_mul_scales);
    BACKEND_DNNL_ADD_PASS(pipeline, convert_dynamic_quantize_ops);

    BACKEND_DNNL_ADD_PASS(pipeline, insert_u8_to_s8_for_matmul);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_reshape_for_ndx2d_matmul);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_unsqueeze_and_squeeze_for_matmul);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_unsqueeze_for_prelu);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_unsqueeze_and_squeeze_for_prelu_bwd);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_unsqueeze_and_squeeze_for_reduction);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_conv_or_deconv);
    BACKEND_DNNL_ADD_PASS(
            pipeline, insert_permute_for_op_only_require_data_format);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_to_group_for_conv_or_deconv);
    BACKEND_DNNL_ADD_PASS(pipeline, conv_bwd_data_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, conv_bwd_weights_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, batchnorm_bwd_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, pool_fwd_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, pool_bwd_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_shuffle);
    BACKEND_DNNL_ADD_PASS(pipeline, reorder_canonicalization);
}

void larger_partition_kernel_t::setup_pipeline_stage2(pass_pipeline_t &pipeline,
        memory_planner_t &mem_planner, bool enable_constant_cache) {
    pipeline.reset_visualize_arg(true, false);
    // do constant propagation here so that we can
    // prepare constant info for other optimizations.
    if (enable_constant_cache) {
        BACKEND_DNNL_ADD_PASS(pipeline, constant_propagation);
    }
    BACKEND_DNNL_ADD_PASS(pipeline, infer_shape);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_transpose_to_matmul);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_transpose_to_predecessor);
    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);
    BACKEND_DNNL_ADD_PASS(pipeline, common_reorder_elimination);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_adjacent_reorders);

    // constant propagation
    if (enable_constant_cache) {
        BACKEND_DNNL_ADD_PASS(pipeline, constant_propagation);
    }

    auto memory_plan = [&](std::shared_ptr<subgraph_t> &sg) {
        return mem_planner.run(sg);
    };
    pipeline.reset_visualize_arg(true, true);
    BACKEND_DNNL_ADD_PASS(pipeline, memory_plan);
    BACKEND_DNNL_ADD_PASS(pipeline, compile_ops);
}

void larger_partition_kernel_t::setup_pipeline(pass_pipeline_t &pipeline,
        memory_planner_t &mem_planner, bool enable_constant_cache) {
    setup_pipeline_stage1(pipeline);
    setup_pipeline_stage2(pipeline, mem_planner, enable_constant_cache);
}

void larger_partition_kernel_t::prepare_args_set(
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

status_t larger_partition_kernel_t::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*eng);

    // get subgraph from the deep copied partition
    subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
            part->get_attr(), part->get_use_blocked_layout(), true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    // Populate the transform passes into the pipeline
    // Note: `std::call_once` should be kept in a single translation unit since
    // GCC 11.
    std::call_once(once_flag_, [&, this]() {
        vis_ = subgraph_visualizer_t(part->id(), [this](const value_t *val) {
            return this->memory_planner_.get_memory_info(val);
        });
        pipeline_ = pass_pipeline_t(vis_);
        setup_pipeline(pipeline_, memory_planner_, enabled_constant_cache());
    });

    // Run the added passes
    BACKEND_DNNL_CHECK(pipeline_.run(subgraph_));

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

    const_md_hash_ = generate_constant_md_hash(part->id(),
            memory_planner_.get_exec_args_set().get_persistent_mem_desc_list());

    return status::success;
}

status_t larger_partition_kernel_t::execute_impl(stream_t *strm,
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
status_t larger_partition_kernel_t::sycl_execute_impl(stream_t *strm,
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
status_t larger_partition_kernel_t::ocl_execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf,
        const std::vector<ocl_event_t> &ocl_deps, ocl_event_t &event) {
    auto deps = ocl_deps;
    ocl_event_t returned_event;
    dnnl::stream p_stream = make_dnnl_stream(*strm);

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
    event = std::move(returned_event);

    return status::success;
}
#endif

kernel_ptr large_partition_kernel_creator() {
    return std::make_shared<larger_partition_kernel_t>();
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
