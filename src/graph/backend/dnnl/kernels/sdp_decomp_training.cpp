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

#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"

#include "graph/backend/dnnl/kernels/sdp_decomp_training.hpp"

#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "cpu/cpu_stream.hpp"
#endif

#define VCHECK_SDP_DECOMP_TRAIN(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_decomp_training_kernel_t, (cond), \
            status, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

status_t sdp_decomp_training_kernel_t::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    VCHECK_SDP_DECOMP_TRAIN(eng->kind() == engine_kind::cpu,
            status::unimplemented, "supports cpu only");

#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_OMP \
        && DNNL_CPU_RUNTIME != DNNL_RUNTIME_THREADPOOL
    UNUSED(part);
    UNUSED(inputs);
    UNUSED(outputs);
    VCHECK_SDP_DECOMP_TRAIN(false, status::unimplemented,
            "supports OMP or Threadpool runtime only");
#else

    p_engine_ = make_dnnl_engine(*eng);

    subgraph_ = std::make_shared<subgraph_t>(
            part->get_ops(), p_engine_, part->get_attr(), false, true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    // Check if supported by training decomposition kernel
    if (!sdp_cfg_.initial_check(subgraph_, inputs, outputs))
        return status::unimplemented;

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline = pass_pipeline_t(vis);
    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_host_scalar);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_reshape_for_gqa);
    BACKEND_DNNL_ADD_PASS(pipeline, binary_canonicalization);
    BACKEND_DNNL_ADD_PASS(pipeline, sdp_fuse_post_ops);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);
    pipeline.reset_visualize_arg(true, false);
    BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_transpose_to_predecessor);
    BACKEND_DNNL_ADD_PASS(pipeline, layout_propagation);

    // Run the added passes
    BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

    // Fill information for inputs logical tensors
    for (size_t i = 0; i < inputs.size(); i++) {
        auto &in = const_cast<logical_tensor_t &>(inputs[i]);
        in = subgraph_->ins_[i];
    }

    // Fill information for outputs logical tensors
    for (size_t i = 0; i < outputs.size(); i++) {
        auto &out = const_cast<logical_tensor_t &>(outputs[i]);
        out = subgraph_->outs_[i];
    }

    resource_ctor_
            = [this]() { return std::make_shared<sdp_args_set_t>(this); };

    return sdp_cfg_.construct_params(
            subgraph_, sdp_registry_, p_engine_, inputs, outputs);
#endif
}

void sdp_decomp_training_kernel_t::prepare_sub_args(
        const grantor_t &var_grantor, const int id, const size_t block_size,
        std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map) {
    auto size_offset = id * block_size;

    // Set data handle for memories that are in execution args
    auto set_handle = [&](const memory &m) {
        auto it = mem_map.find(m.get());
        if (it != mem_map.end()) {
            it->second[id].set_data_handle(
                    var_grantor.get(sdp_cfg_.mem_key_map.at(m.get()))
                    + size_offset);
        }
    };

    // Memories used in primitive args
    if (!sdp_cfg_.sub_reorder0.is_alias()) set_handle(sdp_cfg_.sub_mm1_src);
    if (!sdp_cfg_.sub_reorder1.is_alias()) set_handle(sdp_cfg_.sub_mm1_wei);
    set_handle(sdp_cfg_.sub_mm1_dst);
    set_handle(sdp_cfg_.sub_softmax_out);
    if (sdp_cfg_.needs_softmax_reorder) { set_handle(sdp_cfg_.sub_mm2_src); }
    if (!sdp_cfg_.sub_reorder2.is_alias()) set_handle(sdp_cfg_.sub_mm2_wei);
    if (!sdp_cfg_.sub_reorder3.is_alias()) set_handle(sdp_cfg_.sub_mm2_dst);
    set_handle(sdp_cfg_.sub_scratchpad);

    set_handle(sdp_cfg_.sub_log_max_P);
    set_handle(sdp_cfg_.sub_stats);
}

status_t sdp_decomp_training_kernel_t::execute_impl(stream_t *strm,
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
            inputs[sdp_cfg_.graph_inport[sdp_decomp_training_config_t::mm1_src]]
                    .get_data_handle());
    char *wei1_user_pointer = static_cast<char *>(
            inputs[sdp_cfg_.graph_inport[sdp_decomp_training_config_t::mm1_wei]]
                    .get_data_handle());
    char *wei2_user_pointer = static_cast<char *>(
            inputs[sdp_cfg_.graph_inport[sdp_decomp_training_config_t::mm2_wei]]
                    .get_data_handle());
    char *dst_user_pointer = static_cast<char *>(
            outputs[sdp_cfg_.attn_out_index].get_data_handle());
    char *stats_user_pointer = static_cast<char *>(
            outputs[sdp_cfg_.stats_out_index].get_data_handle());

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
        threadpool_utils::deactivate_threadpool();
#endif

        // Prepare execution args and allocate real memory
        prepare_sub_args(var_grantor, tid, block_size, res->mem_map);

        // reorder0: set Q pointer
        auto &sub_src1_tid = res->mem_map[sdp_cfg_.sub_src1.get()][tid];
        const size_t sub_src1_offset = (bo * sdp_cfg_.src1_strides[0]
                                               + bi * sdp_cfg_.src1_strides[1])
                * get_mem_dt_size(sub_src1_tid);
        sub_src1_tid.set_data_handle(src1_user_pointer + sub_src1_offset);

        // reorder1: set K pointer
        auto &sub_wei1_user_tid
                = res->mem_map[sdp_cfg_.sub_wei1_user.get()][tid];
        const size_t sub_wei1_offset = (bo * sdp_cfg_.wei1_strides[0]
                                               + bi * sdp_cfg_.wei1_strides[1])
                * get_mem_dt_size(sub_wei1_user_tid);
        sub_wei1_user_tid.set_data_handle(wei1_user_pointer + sub_wei1_offset);

        // mm1 post-ops: scale
        if (sdp_cfg_.has_scale) {
            auto &sub_mm1_post_scale_tid
                    = res->mem_map[sdp_cfg_.sub_mm1_post_mem[0].get()][tid];
            sub_mm1_post_scale_tid.set_data_handle(
                    inputs[sdp_cfg_.graph_inport
                                    [sdp_decomp_training_config_t::mm1_scale]]
                            .get_data_handle());
        }

        // reorder2: set V pointer
        auto &sub_wei2_user_tid
                = res->mem_map[sdp_cfg_.sub_wei2_user.get()][tid];
        const size_t sub_wei2_offset = (bo * sdp_cfg_.wei2_strides[0]
                                               + bi * sdp_cfg_.wei2_strides[1])
                * get_mem_dt_size(sub_wei2_user_tid);
        sub_wei2_user_tid.set_data_handle(wei2_user_pointer + sub_wei2_offset);

        // reorder3: set output pointer
        auto &sub_dst_user_tid = res->mem_map[sdp_cfg_.sub_dst_user.get()][tid];
        auto &sub_mm2_dst_tid = res->mem_map[sdp_cfg_.sub_mm2_dst.get()][tid];
        const size_t sub_dst_user_offset
                = (bo * sdp_cfg_.dst_strides[0] + bi * sdp_cfg_.dst_strides[1])
                * get_mem_dt_size(sub_dst_user_tid);
        sub_dst_user_tid.set_data_handle(
                dst_user_pointer + sub_dst_user_offset);

        if (sdp_cfg_.sub_reorder3.is_alias()) {
            sub_mm2_dst_tid.set_data_handle(
                    dst_user_pointer + sub_dst_user_offset);
        }

        // Execute pipeline: reorder0 -> reorder1 -> mm1
        sdp_cfg_.sub_reorder0.execute(p_stream, res->sub_reorder0_args[tid]);
        sdp_cfg_.sub_reorder1.execute(p_stream, res->sub_reorder1_args[tid]);
        sdp_cfg_.sub_mm1_prim.execute(p_stream, res->sub_mm1_args[tid]);

        // Softmax: scores -> P
        dnnl_primitive_execute_without_tp_hook(sdp_cfg_.sub_softmax_prim,
                p_stream, res->sub_softmax_args[tid]);

        // Compute stats: logsumexp = reduce_max(scores) - log(reduce_max(P))
        {
            auto &sub_stats_user_tid
                    = res->mem_map[sdp_cfg_.sub_stats_user.get()][tid];
            sub_stats_user_tid.set_data_handle(stats_user_pointer
                    + (bo * sdp_cfg_.stats_dst_strides[0]
                              + bi * sdp_cfg_.stats_dst_strides[1])
                            * sizeof(float));

            if (sdp_cfg_.sub_reorder_stats.is_alias()) {
                auto &sub_stats_tid
                        = res->mem_map[sdp_cfg_.sub_stats.get()][tid];
                sub_stats_tid.set_data_handle(
                        sub_stats_user_tid.get_data_handle());
            }

            dnnl_primitive_execute_without_tp_hook(
                    sdp_cfg_.sub_reduce_max_P_prim, p_stream,
                    res->sub_reduce_max_P_args[tid]);
            dnnl_primitive_execute_without_tp_hook(
                    sdp_cfg_.sub_reduce_max_src_prim, p_stream,
                    res->sub_reduce_max_src_args[tid]);
            sdp_cfg_.sub_reorder_stats.execute(
                    p_stream, res->sub_reorder_stats_args[tid]);
        }

        // reorder_softmax: f32 P -> xf16 for mm2
        if (sdp_cfg_.needs_softmax_reorder) {
            sdp_cfg_.sub_reorder_softmax.execute(
                    p_stream, res->sub_reorder_softmax_args[tid]);
        }
        // reorder2 -> mm2 -> reorder3
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

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
