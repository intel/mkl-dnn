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

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"

#include "cpu/platform.hpp"

#include "graph/backend/dnnl/kernels/sdp_fused_brgemm.hpp"

#include "graph/backend/dnnl/passes/compile_ops.hpp"
#include "graph/backend/dnnl/passes/constant_propagation.hpp"
#include "graph/backend/dnnl/passes/insert_ops.hpp"
#include "graph/backend/dnnl/passes/layout_propagation.hpp"
#include "graph/backend/dnnl/passes/lower.hpp"
#include "graph/backend/dnnl/passes/memory_planning.hpp"
#include "graph/backend/dnnl/passes/transform.hpp"
#include "graph/backend/dnnl/passes/utils.hpp"

#include "graph/backend/dnnl/op_executable.hpp"

#define VCHECK_SDP_FUSED_BRGEMM(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_fused_brgemm_base_t, (cond), status, \
            msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// Defined out-of-line so std::make_shared<sdp_fused_brgemm_online_kernel_t>()
// does not need to instantiate the destructor inline; fused_driver_ (whose own
// destructor needs the complete softmax_ir_kernel_t type) is complete here.
sdp_fused_brgemm_online_kernel_t::~sdp_fused_brgemm_online_kernel_t() = default;

status_t sdp_fused_brgemm_base_t::parse(const dnnl_partition_impl_t *part,
        engine_t *eng, const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    VCHECK_SDP_FUSED_BRGEMM(eng->kind() == engine_kind::cpu,
            status::unimplemented, "supports cpu only");

#if !DNNL_X64
    UNUSED(part);
    UNUSED(inputs);
    UNUSED(outputs);
    VCHECK_SDP_FUSED_BRGEMM(
            false, status::unimplemented, "fused kernel supports x64 only");
#elif DNNL_CPU_RUNTIME != DNNL_RUNTIME_OMP \
        && DNNL_CPU_RUNTIME != DNNL_RUNTIME_THREADPOOL
    UNUSED(part);
    UNUSED(inputs);
    UNUSED(outputs);
    VCHECK_SDP_FUSED_BRGEMM(false, status::unimplemented,
            "supports OMP or Threadpool runtime only");
#else
    using namespace dnnl::impl::cpu::x64;

    p_engine_ = make_dnnl_engine(*eng);

    // Get subgraph from the deep copied partition.
    subgraph_ = std::make_shared<subgraph_t>(
            part->get_ops(), p_engine_, part->get_fpmath_mode(), false, true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    // Detect whether the scale op is a division before lowering rewrites the
    // graph op kinds into dnnl_binary. Also capture mm1's transpose_b: the
    // blocked driver reads K straight from the user tensor, so it must honour
    // the QK^T transpose itself (the permute pass only rewrites the internal
    // matmul's operands, not the raw input the driver consumes).
    op_t *softmax_op = nullptr;
    std::vector<op_t *> matmul_ops;
    for (const auto &op : subgraph_->get_ops()) {
        if (op->get_kind() == graph::op_kind::Divide)
            prb_.scale_is_divide = true;
        if (op->get_kind() == graph::op_kind::MatMul)
            matmul_ops.push_back(op.get());
        if (op->get_kind() == graph::op_kind::SoftMax) softmax_op = op.get();
    }
    if (softmax_op && softmax_op->has_attr(op_attr::mode))
        prb_.softmax_inf_as_zero
                = softmax_op->get_attr<std::string>(op_attr::mode)
                == "inf_as_zero";
    // mm1 is the QK^T matmul: the one that does not consume the softmax output
    // (that is mm2 = P*V).
    for (op_t *mm : matmul_ops) {
        bool consumes_softmax = false;
        for (size_t i = 0; i < mm->num_inputs(); ++i) {
            const auto &in = mm->get_input_value(i);
            if (in->has_producer() && &in->get_producer() == softmax_op)
                consumes_softmax = true;
        }
        if (!consumes_softmax && mm->has_attr(op_attr::transpose_b))
            prb_.mm1_transpose_b = mm->get_attr<bool>(op_attr::transpose_b);
    }

    // Validate the SDP pattern and extract dims/flags. This fused kernel is
    // non-quantized, so the quantized lowering passes are skipped.
    //
    // The blocked driver parallelizes over (batch, num_head_q, query blocks),
    // so it does not need the decomp RATIO/thread gate (which only saturates
    // threads across batch*num_head); opt out of it.
    if (!sdp_cfg_.initial_check(subgraph_, inputs, outputs,
                /*enforce_thread_ratio=*/false))
        return status::unimplemented;

    // First iteration supports the non-quantized attention pattern:
    // QK^T -> scale -> select-mask -> softmax -> PV.
    prb_.has_scale = sdp_cfg_.has_scale;
    prb_.has_select = sdp_cfg_.has_select;
    prb_.select_fusiable = sdp_cfg_.select_fusiable;
    prb_.has_mask = sdp_cfg_.has_attention_mask;
    VCHECK_SDP_FUSED_BRGEMM(!sdp_cfg_.has_soft_capping, status::unimplemented,
            "fused kernel does not support soft-capping yet");

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
    BACKEND_DNNL_CHECK(pipeline.run(subgraph_));

    // Fill information for inputs/outputs logical tensors.
    for (size_t i = 0; i < inputs.size(); i++) {
        auto &in = const_cast<logical_tensor_t &>(inputs[i]);
        in = subgraph_->ins_[i];
    }
    for (size_t i = 0; i < outputs.size(); i++) {
        auto &out = const_cast<logical_tensor_t &>(outputs[i]);
        out = subgraph_->outs_[i];
    }

    // Locate mm2 (the P*V matmul) in the lowered subgraph and capture ITS
    // output value strides. A trailing StaticTranspose on the SDPA output
    // (e.g. bert's [B,H,S,D] -> [B,S,H,D]) is folded into mm2's output by
    // fuse_dst_transpose_to_predecessor: the partition output tensor then
    // carries the post-transpose axis order [B,S,H,D], but mm2's own output
    // value keeps the driver's [B,H,S,D] axis order with strides that encode
    // the transpose. Reading the partition output tensor's strides directly
    // would mis-map the head/seq axes; mm2's output value is the correct
    // per-(batch,head,seq,head_size_v) stride source. mm2 is the matmul that
    // consumes the softmax output; walk forward from the lowered softmax,
    // skipping any permute/reorder the lowering inserted, to that matmul.
    op_t *mm2_op = nullptr;
    for (const auto &op : subgraph_->get_ops()) {
        if (op->get_kind() != graph::op_kind::_softmax) continue;
        std::vector<op_t *> stack {op.get()};
        while (!stack.empty() && !mm2_op) {
            op_t *cur = stack.back();
            stack.pop_back();
            for (const auto &c : cur->get_output_value(0)->get_consumers()) {
                op_t *co = &c.get_op();
                if (co->get_kind() == graph::op_kind::_matmul) {
                    mm2_op = co;
                    break;
                }
                stack.push_back(co);
            }
        }
        break;
    }

    // Geometry and user strides. Both concrete kernels consume prb_ at compile
    // time; the online kernel also reads it at execute, and the blocked kernel
    // copies the relevant fields into its driver params.
    prb_.ndims = static_cast<int>(sdp_cfg_.ndims);
    prb_.batch = sdp_cfg_.batch_size;
    prb_.num_head_q = sdp_cfg_.num_head_q;
    prb_.num_head_kv = sdp_cfg_.num_head_kv;
    prb_.group_head = prb_.num_head_q / prb_.num_head_kv;
    prb_.seq_q = sdp_cfg_.seq_len_q;
    prb_.head_size_qk = sdp_cfg_.head_size_qk;
    prb_.head_size_v = sdp_cfg_.head_size_v;

    const auto &gi = sdp_cfg_.graph_inport;
    prb_.idx_q = gi[sdp_decomp_config_t::mm1_src];
    prb_.idx_k = gi[sdp_decomp_config_t::mm1_wei];
    prb_.idx_v = gi[sdp_decomp_config_t::mm2_wei];
    prb_.idx_scale = gi[sdp_decomp_config_t::mm1_scale];
    prb_.idx_cond = gi[sdp_decomp_config_t::select_condition];
    prb_.idx_fill = gi[sdp_decomp_config_t::select_other_input];
    prb_.idx_mask = gi[sdp_decomp_config_t::mm1_add];

    prb_.q_strides = ltw(inputs[prb_.idx_q]).vstrides();
    prb_.k_strides = ltw(inputs[prb_.idx_k]).vstrides();
    prb_.v_strides = ltw(inputs[prb_.idx_v]).vstrides();
    // Output strides come from mm2's own output value (driver [B,H,S,D] axis
    // order, transpose-fold aware), not the partition output tensor which may
    // be in a permuted axis order after a folded StaticTranspose. Fall back to
    // the partition output tensor if mm2 could not be located.
    prb_.o_strides = mm2_op
            ? ltw(mm2_op->get_output_value(0)->get_logical_tensor()).vstrides()
            : ltw(outputs[0]).vstrides();
    // K holds seq_kv on its last axis when consumed as K^T (transpose_b == 0),
    // otherwise on its second-to-last axis (natural [.., seq_kv, head_size]).
    prb_.seq_kv = prb_.mm1_transpose_b
            ? ltw(inputs[prb_.idx_k]).vdims()[prb_.ndims - 2]
            : ltw(inputs[prb_.idx_k]).vdims()[prb_.ndims - 1];
    if (prb_.has_select) {
        prb_.cond_strides = ltw(inputs[prb_.idx_cond]).vstrides();
        prb_.cond_dims = ltw(inputs[prb_.idx_cond]).vdims();
    }

    return status::success;
#endif
}

status_t sdp_fused_brgemm_online_kernel_t::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    CHECK(parse(part, eng, inputs, outputs));
#if DNNL_X64
    // Capability gate, mirroring the CPU sdpa primitive's fused driver: this
    // online (flash) kernel is f32-only, has no attention-mask post-op, and
    // consumes K already transposed to [.., head_size_qk, seq_kv] (unlike the
    // blocked driver, sdp_fused_driver_t never transposes K itself). Decline
    // other cases cleanly so the dispatch cascade / forced-impl testing falls
    // back to the blocked/decomp/large kernels instead of miscomputing.
    const auto q_dt = static_cast<dnnl::impl::data_type_t>(
            ltw(inputs[prb_.idx_q]).data_type());
    VCHECK_SDP_FUSED_BRGEMM(q_dt == dnnl::impl::data_type::f32,
            status::unimplemented, "online fused kernel supports f32 only");
    VCHECK_SDP_FUSED_BRGEMM(!prb_.has_mask, status::unimplemented,
            "online fused kernel does not support an attention mask");
    VCHECK_SDP_FUSED_BRGEMM(!prb_.mm1_transpose_b, status::unimplemented,
            "online fused kernel requires K pre-transposed "
            "(transpose_b=false)");

    sdp_fused_params_t fp;
    fp.ndims = prb_.ndims;
    fp.batch = prb_.batch;
    fp.num_head_q = prb_.num_head_q;
    fp.group_head = prb_.group_head;
    fp.seq_q = prb_.seq_q;
    fp.seq_kv = prb_.seq_kv;
    fp.head_size_qk = prb_.head_size_qk;
    fp.head_size_v = prb_.head_size_v;
    fp.q_strides = prb_.q_strides;
    fp.k_strides = prb_.k_strides;
    fp.v_strides = prb_.v_strides;
    fp.o_strides = prb_.o_strides;
    fp.cond_strides = prb_.cond_strides;
    fp.cond_dims = prb_.cond_dims;
    fp.has_select = prb_.has_select;
    fp.select_fusiable = prb_.select_fusiable;
    CHECK(fused_driver_.init(fp, eng));
    nthr_ = fused_driver_.nthr();
    fused_scratch_total_ = fused_driver_.scratch_total(nthr_);
    return status::success;
#else
    return status::unimplemented;
#endif
}

status_t sdp_fused_brgemm_blocked_kernel_t::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
        const std::vector<logical_tensor_t> &outputs) {
    CHECK(parse(part, eng, inputs, outputs));
#if DNNL_X64
    sdp_blocked_params_t bp;
    bp.ndims = prb_.ndims;
    bp.batch = prb_.batch;
    bp.num_head_q = prb_.num_head_q;
    bp.group_head = prb_.group_head;
    bp.seq_q = prb_.seq_q;
    bp.seq_kv = prb_.seq_kv;
    bp.head_size_qk = prb_.head_size_qk;
    bp.head_size_v = prb_.head_size_v;
    bp.q_strides = prb_.q_strides;
    bp.k_strides = prb_.k_strides;
    bp.v_strides = prb_.v_strides;
    bp.o_strides = prb_.o_strides;
    bp.cond_strides = prb_.cond_strides;
    bp.cond_dims = prb_.cond_dims;
    bp.has_select = prb_.has_select;
    bp.select_fusiable = prb_.select_fusiable;
    bp.mm1_transpose_b = prb_.mm1_transpose_b;
    bp.softmax_inf_as_zero = prb_.softmax_inf_as_zero;
    // Compute type (Q/K/V) and output type. The scores/pv tiles stay f32
    // (BRGEMM accumulates in f32); the driver down-converts P and the output to
    // these types.
    bp.mm_dt = static_cast<dnnl::impl::data_type_t>(
            ltw(inputs[prb_.idx_q]).data_type());
    bp.out_dt
            = static_cast<dnnl::impl::data_type_t>(ltw(outputs[0]).data_type());
    // mm1 post-op chain in graph order: scale (binary-mul, scalar rhs, already
    // reciprocated at execute if Divide) then the additive attention mask
    // (binary-add, tensor rhs offset per batch/head/tile). Soft-cap entries will
    // be appended here as they are enabled.
    if (prb_.has_scale) {
        sdp_mm1_post_op_t sc;
        sc.alg = dnnl::impl::alg_kind::binary_mul;
        sc.is_binary = true;
        sc.rhs_is_scalar = true;
        sc.rhs_dt = dnnl::impl::data_type::f32;
        bp.mm1_post_ops.push_back(sc);
    }
    if (prb_.has_mask) {
        sdp_mm1_post_op_t mk;
        mk.alg = dnnl::impl::alg_kind::binary_add;
        mk.is_binary = true;
        mk.rhs_is_scalar = false;
        mk.rhs_dt = static_cast<dnnl::impl::data_type_t>(
                ltw(inputs[prb_.idx_mask]).data_type());
        mk.rhs_dims = ltw(inputs[prb_.idx_mask]).vdims();
        mk.rhs_strides = ltw(inputs[prb_.idx_mask]).vstrides();
        bp.mm1_post_ops.push_back(mk);
    }
    CHECK(blocked_driver_.init(bp, eng));
    nthr_ = blocked_driver_.nthr();
    blocked_scratch_total_ = blocked_driver_.scratch_total(nthr_);
    return status::success;
#else
    return status::unimplemented;
#endif
}

status_t sdp_fused_brgemm_blocked_kernel_t::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    UNUSED(strm);
#if !DNNL_X64
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(scratchpad_buf);
    return status::unimplemented;
#else
    auto *q_base
            = static_cast<const char *>(inputs[prb_.idx_q].get_data_handle());
    auto *k_base
            = static_cast<const char *>(inputs[prb_.idx_k].get_data_handle());
    auto *v_base
            = static_cast<const char *>(inputs[prb_.idx_v].get_data_handle());
    auto *o_base = static_cast<char *>(outputs[0].get_data_handle());

    float scale_val = 1.0f;
    if (prb_.has_scale) {
        scale_val = *static_cast<const float *>(
                inputs[prb_.idx_scale].get_data_handle());
        if (prb_.scale_is_divide) scale_val = 1.0f / scale_val;
    }
    float fill_val = 0.0f;
    const char *cond_base = nullptr;
    if (prb_.has_select) {
        fill_val = *static_cast<const float *>(
                inputs[prb_.idx_fill].get_data_handle());
        cond_base = static_cast<const char *>(
                inputs[prb_.idx_cond].get_data_handle());
    }

    sdp_blocked_run_args_t args;
    args.q = q_base;
    args.k = k_base;
    args.v = v_base;
    args.cond = cond_base;
    args.out = o_base;
    args.fill = fill_val;
    // rhs base pointers for the mm1 binary post-ops, in chain order. Only the QK
    // scale is present today; scale_val is a stable local that outlives the
    // execute call below.
    if (prb_.has_scale) args.mm1_post_op_rhs.push_back(&scale_val);
    if (prb_.has_mask)
        args.mm1_post_op_rhs.push_back(inputs[prb_.idx_mask].get_data_handle());
    auto scratchpad = std::make_shared<scratchpad_t>(
            scratchpad_buf, blocked_scratch_total_, p_engine_);
    return blocked_driver_.execute(args, scratchpad->get_buffer(), nthr_);
#endif
}

status_t sdp_fused_brgemm_online_kernel_t::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    UNUSED(strm);
#if !DNNL_X64
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(scratchpad_buf);
    return status::unimplemented;
#else
    auto *q_base
            = static_cast<const char *>(inputs[prb_.idx_q].get_data_handle());
    auto *k_base
            = static_cast<const char *>(inputs[prb_.idx_k].get_data_handle());
    auto *v_base
            = static_cast<const char *>(inputs[prb_.idx_v].get_data_handle());
    auto *o_base = static_cast<char *>(outputs[0].get_data_handle());

    float scale_val = 1.0f;
    if (prb_.has_scale) {
        scale_val = *static_cast<const float *>(
                inputs[prb_.idx_scale].get_data_handle());
        if (prb_.scale_is_divide) scale_val = 1.0f / scale_val;
    }
    float fill_val = 0.0f;
    const char *cond_base = nullptr;
    if (prb_.has_select) {
        fill_val = *static_cast<const float *>(
                inputs[prb_.idx_fill].get_data_handle());
        cond_base = static_cast<const char *>(
                inputs[prb_.idx_cond].get_data_handle());
    }

    sdp_fused_run_args_t args;
    args.q = q_base;
    args.k = k_base;
    args.v = v_base;
    args.cond = cond_base;
    args.out = o_base;
    args.scale = scale_val;
    args.fill = fill_val;
    auto scratchpad = std::make_shared<scratchpad_t>(
            scratchpad_buf, fused_scratch_total_, p_engine_);
    return fused_driver_.execute(args, scratchpad->get_buffer(), nthr_);
#endif
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
