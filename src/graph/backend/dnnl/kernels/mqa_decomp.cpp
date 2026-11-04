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

#include "graph/backend/dnnl/kernels/mqa_decomp.hpp"

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

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

template <bool quantized, memory::data_type dt>
status_t mqa_decomp_kernel_t<quantized, dt>::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    p_engine_ = make_dnnl_engine(*eng);

    // get subgraph from the deep copied partition
    subgraph_ = std::make_shared<subgraph_t>(part->get_ops(), p_engine_,
            part->get_attr(), part->get_use_blocked_layout(), true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    // Check if it's supported by decomposition kernel
    if (!mqa_cfg_.initial_check(subgraph_, inputs))
        return status::unimplemented;

    subgraph_visualizer_t vis(part->id(), [this](const value_t *val) {
        return this->memory_planner_.get_memory_info(val);
    });
    pass_pipeline_t pipeline = pass_pipeline_t(vis);

    BACKEND_DNNL_ADD_PASS(pipeline, lower_down);
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
    // MQA pattern fusion
    BACKEND_DNNL_ADD_PASS(pipeline, lift_up_post_add_for_matmul);

    BACKEND_DNNL_ADD_PASS(pipeline, fuse_post_ops);
    BACKEND_DNNL_ADD_PASS(pipeline, insert_permute_for_matmul);
    if (quantized) {
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_scales);
        BACKEND_DNNL_ADD_PASS(pipeline, convert_to_runtime_dst_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, fuse_dst_zero_points);
        BACKEND_DNNL_ADD_PASS(pipeline, remove_quant_data_with_no_effect);
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
            = [this]() { return std::make_shared<mqa_args_set_t>(this); };

    // Initialize and construct kernel params
    mqa_cfg_.construct_params<quantized, dt>(
            subgraph_, mqa_registry_, p_engine_, inputs);

    return status::success;
}

template <bool quantized, memory::data_type dt>
void mqa_decomp_kernel_t<quantized, dt>::prepare_sub_args(
        const grantor_t &var_grantor, const int id, const size_t block_size,
        std::unordered_map<dnnl_memory_t, std::vector<memory>> &mem_map) {
    auto size_offset = id * block_size;
    mem_map[mqa_cfg_.sub_mm1_wei.get()][id].set_data_handle(
            var_grantor.get(mqa_cfg_.mem_key_map[mqa_cfg_.sub_mm1_wei.get()])
            + size_offset);
    // mm1
    mem_map[mqa_cfg_.sub_mm1_src.get()][id].set_data_handle(
            var_grantor.get(
                    mqa_cfg_.mem_key_map[mqa_cfg_.sub_max_src1_src2.get()])
            + size_offset);
    mem_map[mqa_cfg_.sub_mm1_dst.get()][id].set_data_handle(
            var_grantor.get(
                    mqa_cfg_.mem_key_map[mqa_cfg_.sub_max_dst1_dst2.get()])
            + size_offset);
    // softmax
    mem_map[mqa_cfg_.sub_softmax_dst.get()][id].set_data_handle(
            var_grantor.get(
                    mqa_cfg_.mem_key_map[mqa_cfg_.sub_softmax_dst.get()])
            + size_offset);
    // mm2
    mem_map[mqa_cfg_.sub_mm2_src.get()][id].set_data_handle(
            var_grantor.get(
                    mqa_cfg_.mem_key_map[mqa_cfg_.sub_max_src1_src2.get()])
            + size_offset);
    mem_map[mqa_cfg_.sub_mm2_dst.get()][id].set_data_handle(
            var_grantor.get(
                    mqa_cfg_.mem_key_map[mqa_cfg_.sub_max_dst1_dst2.get()])
            + size_offset);
    // scratchpad, each thread will have a largest scratchpad.
    mem_map[mqa_cfg_.sub_scratchpad.get()][id].set_data_handle(
            var_grantor.get(mqa_cfg_.mem_key_map[mqa_cfg_.sub_scratchpad.get()])
            + size_offset);
}

template <bool quantized, memory::data_type dt>
status_t mqa_decomp_kernel_t<quantized, dt>::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    dnnl::stream p_stream = make_dnnl_stream(*strm);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    auto *tp_stream
            = dnnl::impl::utils::downcast<dnnl::impl::cpu::cpu_stream_t *>(
                    strm);
#endif

    // each thread's own local resource
    thread_local_cache_t<mqa_args_set_t> res_cache;
    mqa_args_set_t *res = res_cache.get_or_add(
            reinterpret_cast<size_t>(this), resource_ctor_);

    dim_t MBO = mqa_cfg_.batch_size, MBI = mqa_cfg_.num_head,
          M1 = mqa_cfg_.seq_len, K1 = mqa_cfg_.size_per_head,
          N1 = mqa_cfg_.seq_len, M2 = mqa_cfg_.size_per_head,
          K2 = mqa_cfg_.seq_len, N2 = mqa_cfg_.seq_len;

    char *src1_user_pointer = static_cast<char *>(
            inputs[mqa_cfg_.graph_inport[0]].get_data_handle());
    char *wei1_user_pointer = static_cast<char *>(
            inputs[mqa_cfg_.graph_inport[1]].get_data_handle());
    char *post_add_user_pointer = static_cast<char *>(
            inputs[mqa_cfg_.graph_inport[2]].get_data_handle());
    char *src2_user_pointer = static_cast<char *>(
            inputs[mqa_cfg_.graph_inport[3]].get_data_handle());
    char *dst2_user_pointer = static_cast<char *>(outputs[0].get_data_handle());

    // allocate the internal memory
    size_t block_size = mqa_registry_.size();
    auto scratchpad = std::make_shared<scratchpad_t>(
            scratchpad_buf, block_size * mqa_cfg_.nthr, p_engine_);
    grantor_t var_grantor = mqa_registry_.grantor(scratchpad->get_buffer());

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

        // reorder0
        auto &sub_src1_tid = res->mem_map[mqa_cfg_.sub_src1.get()][tid];
        // reorder1:
        auto &sub_wei1_user_tid
                = res->mem_map[mqa_cfg_.sub_wei1_user.get()][tid];

        auto &sub_mm1_post_add_tid
                = res->mem_map[mqa_cfg_.sub_mm1_post_add.get()][tid];

        // reorder2:
        auto &sub_src2_user_tid
                = res->mem_map[mqa_cfg_.sub_src2_user.get()][tid];

        //reorder3
        auto &sub_dst_user_tid = res->mem_map[mqa_cfg_.sub_dst_user.get()][tid];

        // matmul2
        auto &sub_mm2_dst_tid = res->mem_map[mqa_cfg_.sub_mm2_dst.get()][tid];

        const size_t sub_src1_offset
                = bo * M1 * K1 * get_mem_dt_size(sub_src1_tid);
        const size_t sub_wei1_offset = (bo * MBI * K1 * N1 + bi * N1)
                * get_mem_dt_size(sub_wei1_user_tid);
        const size_t sub_src2_offset
                = bo * M2 * K2 * get_mem_dt_size(sub_src2_user_tid);
        const size_t sub_post_add_offset = (bo * MBI * M1 * N1 + bi * M1 * N1)
                * get_mem_dt_size(sub_mm1_post_add_tid);
        const size_t sub_dst_user_offset = (bo * MBI * M2 * N2 + bi * N2)
                * get_mem_dt_size(sub_dst_user_tid);

        sub_wei1_user_tid.set_data_handle(wei1_user_pointer + sub_wei1_offset);
        sub_src1_tid.set_data_handle(src1_user_pointer + sub_src1_offset);
        sub_src2_user_tid.set_data_handle(src2_user_pointer + sub_src2_offset);
        sub_mm1_post_add_tid.set_data_handle(
                post_add_user_pointer + sub_post_add_offset);
        sub_dst_user_tid.set_data_handle(
                dst2_user_pointer + sub_dst_user_offset);

        // If the last reorder is inplace, it means we don't have to do
        // extra reorder, thus we should set matmul's output to the user's
        // output directly.
        if (mqa_cfg_.sub_reorder3.get_inplace()) {
            sub_mm2_dst_tid.set_data_handle(
                    dst2_user_pointer + sub_dst_user_offset);
        }

        // in parallel region - these primitives should use single thread.
        mqa_cfg_.sub_reorder0.execute(p_stream, res->sub_reorder0_args[tid]);
        mqa_cfg_.sub_reorder1.execute(p_stream, res->sub_reorder1_args[tid]);
        dnnl_primitive_execute_without_tp_hook(
                mqa_cfg_.sub_mm1_prim, p_stream, res->sub_mm1_args[tid]);

        dnnl_primitive_execute_without_tp_hook(mqa_cfg_.sub_softmax_prim,
                p_stream, res->sub_softmax_args[tid]);

        mqa_cfg_.sub_reorder2.execute(p_stream, res->sub_reorder2_args[tid]);

        dnnl_primitive_execute_without_tp_hook(
                mqa_cfg_.sub_mm2_prim, p_stream, res->sub_mm2_args[tid]);
        mqa_cfg_.sub_reorder3.execute(p_stream, res->sub_reorder3_args[tid]);
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        auto tp = threadpool_utils::get_active_threadpool();
        threadpool_utils::activate_threadpool(tp);
#endif
    };

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    tp_stream->before_exec_hook();
#endif

    parallel_nd_ext(mqa_cfg_.nthr, MBO, MBI, loop);

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    tp_stream->after_exec_hook();
#endif

    prolong_scratchpad_lifetime(strm, scratchpad);

    return status::success;
}

template struct mqa_decomp_kernel_t<false, dnnl::memory::data_type::f32>;

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
