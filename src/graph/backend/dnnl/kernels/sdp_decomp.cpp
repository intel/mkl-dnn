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

#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"

#include "graph/backend/dnnl/kernels/sdp_decomp.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "graph/backend/dnnl/op_executable.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "cpu/cpu_stream.hpp"
#endif

#define VCHECK_SDP_PRIMITIVE(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_decomp_kernel_t, (cond), status, msg, \
            ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
template <bool quantized, memory::data_type dt>
status_t sdp_decomp_kernel_t<quantized, dt>::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    VCHECK_SDP_PRIMITIVE(eng->kind() == engine_kind::cpu, status::unimplemented,
            "supports cpu only");

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_OMP \
        && DNNL_CPU_RUNTIME != DNNL_RUNTIME_THREADPOOL
    UNUSED(part);
    UNUSED(inputs);
    UNUSED(outputs);
    VCHECK_SDP_PRIMITIVE(false, status::unimplemented,
            "supports OMP or Threadpool runtime only");
#else

    p_engine_ = make_dnnl_engine(*eng);

    // get subgraph from the deep copied partition
    subgraph_ = std::make_shared<subgraph_t>(
            part->get_ops(), p_engine_, part->get_attr(), false, true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    // Check if it's supported by decomposition kernel
    if (!sdp_cfg_.initial_check(subgraph_, inputs, outputs))
        return status::unimplemented;

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline = pass_pipeline_t(vis);
    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_host_scalar);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_reshape_for_gqa);
    // Fusion and canonicalization passes begin
    if (quantized) {
        BACKEND_DNNL_ADD_PASS(pipeline, lift_up_typecast);
        BACKEND_DNNL_ADD_PASS(pipeline, lift_up_quantize);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_typecast_to_matmul_or_conv);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_typecast_to_predecessor);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_src_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_src_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_src_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, insert_runtime_u8_to_s8_for_matmul);
    }
    BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, sdp_fuse_post_ops);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);
    if (quantized) {
        BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_zero_points);
        // fuse those new post-binaries converted from add_zps and mul_scales
        BACKEND_DNNL_ADD_PASS(pipeline, replace_quant_data_with_binary_post_op);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
    }
    pipeline.reset_visualize_arg(true, false);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_transpose_to_predecessor);
    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

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

    resource_ctor_
            = [this]() { return std::make_shared<sdp_args_set_t>(this); };

    // Initialize and construct kernel params
    return sdp_cfg_.construct_params<quantized, dt>(
            subgraph_, sdp_registry_, p_engine_, inputs);
#endif
}

template <bool quantized, memory::data_type dt>
void sdp_decomp_kernel_t<quantized, dt>::prepare_sub_args(
        const grantor_t &var_grantor, const int id, const size_t block_size,
        std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map) {
    auto size_offset = id * block_size;
    if (!sdp_cfg_.sub_reorder0.is_alias())
        mem_map[sdp_cfg_.sub_mm1_src.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_src1_src2.get()])
                + size_offset);
    if (!sdp_cfg_.sub_reorder1.is_alias())
        mem_map[sdp_cfg_.sub_mm1_wei.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_mm1_wei.get()])
                + size_offset);
    // mm1
    mem_map[sdp_cfg_.sub_mm1_dst.get()][id].set_data_handle(
            var_grantor.get(
                    sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_dst1_wei2.get()])
            + size_offset);
    // select, prepare intermediate execution args for select if it cannot be
    // fused into mm1.
    if (sdp_cfg_.has_select && !sdp_cfg_.select_fusiable) {
        mem_map[sdp_cfg_.sub_select_dst.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_select_dst.get()])
                + size_offset);
    }
    // softmax
    mem_map[sdp_cfg_.sub_softmax_dst.get()][id].set_data_handle(
            var_grantor.get(
                    sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_src1_src2.get()])
            + size_offset);
    // mm2
    if (!sdp_cfg_.sub_reorder2.is_alias())
        mem_map[sdp_cfg_.sub_mm2_wei.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_max_dst1_wei2.get()])
                + size_offset);
    if (!sdp_cfg_.sub_reorder3.is_alias())
        mem_map[sdp_cfg_.sub_mm2_dst.get()][id].set_data_handle(
                var_grantor.get(
                        sdp_cfg_.mem_key_map[sdp_cfg_.sub_mm2_dst.get()])
                + size_offset);
    // scratchpad, each thread will have a largest scratchpad.
    mem_map[sdp_cfg_.sub_scratchpad.get()][id].set_data_handle(
            var_grantor.get(sdp_cfg_.mem_key_map[sdp_cfg_.sub_scratchpad.get()])
            + size_offset);
}

template <bool quantized, memory::data_type dt>
status_t sdp_decomp_kernel_t<quantized, dt>::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    dnnl::stream p_stream = make_dnnl_stream(*strm);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    auto *tp_stream
            = dnnl::impl::utils::downcast<dnnl::impl::cpu::cpu_stream_t *>(
                    strm);
#endif

    thread_local_cache_t<sdp_args_set_t> res_cache;
    sdp_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    dim_t MBO = sdp_cfg_.batch_size, MBI = sdp_cfg_.num_head_q;

    char *src1_user_pointer = static_cast<char *>(
            inputs[sdp_cfg_.graph_inport[sdp_decomp_config_t::mm1_src]]
                    .get_data_handle());
    char *wei1_user_pointer = static_cast<char *>(
            inputs[sdp_cfg_.graph_inport[sdp_decomp_config_t::mm1_wei]]
                    .get_data_handle());
    char *wei2_user_pointer = static_cast<char *>(
            inputs[sdp_cfg_.graph_inport[sdp_decomp_config_t::mm2_wei]]
                    .get_data_handle());
    char *dst2_user_pointer = static_cast<char *>(outputs[0].get_data_handle());

    size_t block_size = sdp_registry_.size();
    auto scratchpad = std::make_shared<scratchpad_t>(
            scratchpad_buf, block_size * sdp_cfg_.nthr, p_engine_);
    grantor_t var_grantor = sdp_registry_.grantor(scratchpad->get_buffer());

    const auto get_mem_dt_size = [](const memory &m) -> size_t {
        return memory::data_type_size(m.get_desc().get_data_type());
    };

    const auto loop
            = [= COMPAT_THIS_CAPTURE](int tid, int nthr, dim_t bo, dim_t bi) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        // Deactivating since nested usage model demands it.
        threadpool_utils::deactivate_threadpool();
#endif

        // prepare execution args and allocate real memory
        prepare_sub_args(var_grantor, tid, block_size, res->mem_map);

        const size_t group_head = sdp_cfg_.num_head_q / sdp_cfg_.num_head_kv;
        const size_t wei_head_offset = bi / group_head;
        const size_t group_id = bi % group_head;

        // reorder0
        auto &sub_src1_tid = res->mem_map[sdp_cfg_.sub_src1.get()][tid];
        // reorder1:
        auto &sub_wei1_user_tid
                = res->mem_map[sdp_cfg_.sub_wei1_user.get()][tid];

        // matmul1
        //post op index offset.
        size_t start_index = 0;
        if (sdp_cfg_.has_scale) {
            auto &sub_mm1_post_scale_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_mem[start_index++]
                                           .get()][tid];
            sub_mm1_post_scale_tid.set_data_handle(
                    inputs[sdp_cfg_.graph_inport
                                    [sdp_decomp_config_t::mm1_scale]]
                            .get_data_handle());
        }
        if (sdp_cfg_.has_soft_capping) {
            auto &sub_mm1_post_soft_cap_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_mem[start_index++]
                                           .get()][tid];
            sub_mm1_post_soft_cap_tid.set_data_handle(static_cast<char *>(
                    inputs[sdp_cfg_.graph_inport
                                    [sdp_decomp_config_t::mm1_soft_capping]]
                            .get_data_handle()));
        }
        if (sdp_cfg_.has_attention_mask) {
            auto &sub_mm1_post_add_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_mem[start_index++]
                                           .get()][tid];
            const auto &mask_input = inputs
                    [sdp_cfg_.graph_inport[sdp_decomp_config_t::mm1_add]];
            const auto mask_strides
                    = ltw(mask_input.get_logical_tensor()).vstrides();
            const auto mask_dims = ltw(mask_input.get_logical_tensor()).vdims();
            size_t mask_offset = 0;
            if (mask_dims.size() == 4) {
                if (mask_dims[0] != 1) mask_offset += bo * mask_strides[0];
                if (mask_dims[1] != 1) mask_offset += bi * mask_strides[1];
            } else if (mask_dims.size() == 5) {
                if (mask_dims[0] != 1) mask_offset += bo * mask_strides[0];
                if (mask_dims[1] != 1)
                    mask_offset += wei_head_offset * mask_strides[1];
                if (mask_dims[2] != 1)
                    mask_offset += group_id * mask_strides[2];
            }
            sub_mm1_post_add_tid.set_data_handle(
                    static_cast<char *>(mask_input.get_data_handle())
                    + mask_offset * get_mem_dt_size(sub_mm1_post_add_tid));
        }
        if (sdp_cfg_.has_select) {
            auto &sub_select_src_tid = sdp_cfg_.select_fusiable
                    ? res->mem_map[sdp_cfg_.sub_mm1_post_mem[start_index++]
                                           .get()][tid]
                    : res->mem_map[sdp_cfg_.sub_select_src.get()][tid];
            const auto &select_src_input
                    = inputs[sdp_cfg_.graph_inport
                                     [sdp_decomp_config_t::select_other_input]];
            const auto select_src_strides
                    = ltw(select_src_input.get_logical_tensor()).vstrides();
            const auto select_src_dims
                    = ltw(select_src_input.get_logical_tensor()).vdims();

            size_t select_src_offset = 0;
            if (select_src_dims.size() == 4) {
                if (select_src_dims[0] != 1)
                    select_src_offset += bo * select_src_strides[0];
                if (select_src_dims[1] != 1)
                    select_src_offset += bi * select_src_strides[1];
            } else if (select_src_dims.size() == 5) {
                if (select_src_dims[0] != 1)
                    select_src_offset += bo * select_src_strides[0];
                if (select_src_dims[1] != 1)
                    select_src_offset
                            += wei_head_offset * select_src_strides[1];
                if (select_src_dims[2] != 1)
                    select_src_offset += group_id * select_src_strides[2];
            }
            sub_select_src_tid.set_data_handle(
                    static_cast<char *>(select_src_input.get_data_handle())
                    + select_src_offset * get_mem_dt_size(sub_select_src_tid));

            auto &sub_select_cond_tid = sdp_cfg_.select_fusiable
                    ? res->mem_map[sdp_cfg_.sub_mm1_post_mem[start_index++]
                                           .get()][tid]
                    : res->mem_map[sdp_cfg_.sub_select_cond.get()][tid];
            const auto &select_cond_input
                    = inputs[sdp_cfg_.graph_inport
                                     [sdp_decomp_config_t::select_condition]];
            const auto select_cond_strides
                    = ltw(select_cond_input.get_logical_tensor()).vstrides();
            const auto select_cond_dims
                    = ltw(select_cond_input.get_logical_tensor()).vdims();

            size_t select_cond_offset = 0;
            if (select_cond_dims.size() == 4) {
                if (select_cond_dims[0] != 1)
                    select_cond_offset += bo * select_cond_strides[0];
                if (select_cond_dims[1] != 1)
                    select_cond_offset += bi * select_cond_strides[1];
            } else if (select_cond_dims.size() == 5) {
                if (select_cond_dims[0] != 1)
                    select_cond_offset += bo * select_cond_strides[0];
                if (select_cond_dims[1] != 1)
                    select_cond_offset
                            += wei_head_offset * select_cond_strides[1];
                if (select_cond_dims[2] != 1)
                    select_cond_offset += group_id * select_cond_strides[2];
            }
            sub_select_cond_tid.set_data_handle(
                    static_cast<char *>(select_cond_input.get_data_handle())
                    + select_cond_offset
                            * get_mem_dt_size(sub_select_cond_tid));
        }
        // reorder2:
        auto &sub_wei2_user_tid
                = res->mem_map[sdp_cfg_.sub_wei2_user.get()][tid];

        // reorder3
        auto &sub_dst_user_tid = res->mem_map[sdp_cfg_.sub_dst_user.get()][tid];

        // matmul2
        auto &sub_mm2_dst_tid = res->mem_map[sdp_cfg_.sub_mm2_dst.get()][tid];

        const size_t sub_src1_head_offset = sdp_cfg_.ndims == 4
                ? bi * sdp_cfg_.src1_strides[1]
                : wei_head_offset * sdp_cfg_.src1_strides[1]
                        + group_id * sdp_cfg_.src1_strides[2];
        const size_t sub_src1_offset
                = (bo * sdp_cfg_.src1_strides[0] + sub_src1_head_offset)
                * get_mem_dt_size(sub_src1_tid);

        const size_t sub_wei1_offset
                = (bo * sdp_cfg_.wei1_strides[0]
                          + wei_head_offset * sdp_cfg_.wei1_strides[1])
                * get_mem_dt_size(sub_wei1_user_tid);
        const size_t sub_wei2_offset
                = (bo * sdp_cfg_.wei2_strides[0]
                          + wei_head_offset * sdp_cfg_.wei2_strides[1])
                * get_mem_dt_size(sub_wei2_user_tid);

        const size_t sub_dst_user_head_offset = sdp_cfg_.ndims == 4
                ? bi * sdp_cfg_.dst_strides[1]
                : wei_head_offset * sdp_cfg_.dst_strides[1]
                        + group_id * sdp_cfg_.dst_strides[2];
        const size_t sub_dst_user_offset
                = (bo * sdp_cfg_.dst_strides[0] + sub_dst_user_head_offset)
                * get_mem_dt_size(sub_dst_user_tid);

        sub_wei1_user_tid.set_data_handle(wei1_user_pointer + sub_wei1_offset);
        sub_src1_tid.set_data_handle(src1_user_pointer + sub_src1_offset);
        sub_wei2_user_tid.set_data_handle(wei2_user_pointer + sub_wei2_offset);
        sub_dst_user_tid.set_data_handle(
                dst2_user_pointer + sub_dst_user_offset);

        // If the last reorder is inplace, it means we don't have to do
        // extra reorder, thus we should set matmul's output to the user's
        // output directly.
        if (sdp_cfg_.sub_reorder3.is_alias()) {
            sub_mm2_dst_tid.set_data_handle(
                    dst2_user_pointer + sub_dst_user_offset);
        }
        // in parallel region - these primitives should use single thread.
        sdp_cfg_.sub_reorder0.execute(p_stream, res->sub_reorder0_args[tid]);
        sdp_cfg_.sub_reorder1.execute(p_stream, res->sub_reorder1_args[tid]);
        sdp_cfg_.sub_mm1_prim.execute(p_stream, res->sub_mm1_args[tid]);
        if (sdp_cfg_.has_select && !sdp_cfg_.select_fusiable)
            dnnl_primitive_execute_without_tp_hook(sdp_cfg_.sub_select_prim,
                    p_stream, res->sub_select_args[tid]);
        dnnl_primitive_execute_without_tp_hook(sdp_cfg_.sub_softmax_prim,
                p_stream, res->sub_softmax_args[tid]);

        sdp_cfg_.sub_reorder2.execute(p_stream, res->sub_reorder2_args[tid]);

        sdp_cfg_.sub_mm2_prim.execute(p_stream, res->sub_mm2_args[tid]);
        sdp_cfg_.sub_reorder3.execute(p_stream, res->sub_reorder3_args[tid]);
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        auto tp = threadpool_utils::get_active_threadpool();
        threadpool_utils::activate_threadpool(tp);
#endif
    };
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    tp_stream->before_exec_hook();
#endif

    parallel_nd_ext(sdp_cfg_.nthr, MBO, MBI, loop);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    tp_stream->after_exec_hook();
#endif

    prolong_scratchpad_lifetime(strm, scratchpad);

    return status::success;
}

template struct sdp_decomp_kernel_t<false, dnnl::memory::data_type::f32>;
template struct sdp_decomp_kernel_t<true, dnnl::memory::data_type::bf16>;
template struct sdp_decomp_kernel_t<true, dnnl::memory::data_type::f32>;
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
