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
#include <unordered_set>

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

#if DNNL_X64
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"

#include "graph/backend/dnnl/kernels/sdp_fused_softmax_ir.hpp"
#endif

#define VCHECK_SDP_FUSED_BRGEMM(cond, status, msg, ...) \
    VCONDCHECK(graph, create, check, sdp_fused_brgemm_kernel_t, (cond), \
            status, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

// Scratchpad keys for the per-thread online-softmax working buffers.
enum mem_key : size_t {
    mem_scores = 0,
    mem_acc,
    mem_pv,
    mem_row_max,
    mem_row_denom,
    mem_old_coef,
};

// Constructor is defined here where the x64 IR kernel type is complete
// (unique_ptr members to a forward-declared type).
sdp_fused_brgemm_kernel_t::sdp_fused_brgemm_kernel_t() = default;

sdp_fused_brgemm_kernel_t::~sdp_fused_brgemm_kernel_t() {
#if DNNL_X64
    for (auto *k :
            {mm1_kernel_, mm2_kernel_, mm1_tail_kernel_, mm2_tail_kernel_}) {
        if (k) brgemm_kernel_destroy(k);
    }
#endif
}

status_t sdp_fused_brgemm_kernel_t::compile_impl(
        const dnnl_partition_impl_t *part, engine_t *eng,
        const std::vector<logical_tensor_t> &inputs,
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

    // KV tiling width for the streaming softmax: K/V are processed in chunks
    // of up to this many columns, bounding the per-thread scores tile
    // ([seq_q, kv_blk]).
    // TODO: this is a fixed heuristic; it should be derived from the cache
    // size, seq_q and head size so the scores/pv tiles stay cache-resident.
    constexpr dim_t kv_block_width = 512;

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
        if (op->get_kind() == graph::op_kind::Divide) scale_is_divide_ = true;
        if (op->get_kind() == graph::op_kind::MatMul)
            matmul_ops.push_back(op.get());
        if (op->get_kind() == graph::op_kind::SoftMax) softmax_op = op.get();
    }
    if (softmax_op && softmax_op->has_attr(op_attr::mode))
        softmax_inf_as_zero_ = softmax_op->get_attr<std::string>(op_attr::mode)
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
            mm1_transpose_b_ = mm->get_attr<bool>(op_attr::transpose_b);
    }

    // Validate the SDP pattern and extract dims/flags. This fused kernel is
    // fp32-only, so the quantized lowering passes are skipped.
    //
    // The blocked driver parallelizes over (batch, num_head_q, query blocks),
    // so it does not need the decomp RATIO/thread gate (which only saturates
    // threads across batch*num_head); opt out of it.
    if (!sdp_cfg_.initial_check(subgraph_, inputs, outputs,
                /*enforce_thread_ratio=*/false))
        return status::unimplemented;

    // First iteration only supports the plain fp32 GQA pattern:
    // QK^T -> scale -> select-mask -> softmax -> PV.
    has_scale_ = sdp_cfg_.has_scale;
    has_select_ = sdp_cfg_.has_select;
    select_fusiable_ = sdp_cfg_.select_fusiable;
    has_mask_ = sdp_cfg_.has_attention_mask;
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
    // per-(batch,head,seq,head_size_v) stride source. mm2 is the matmul whose
    // inputs trace back (through the softmax / reorder / permute ops) to the
    // other (QK^T) matmul's output.
    auto traces_to_other_matmul = [](op_t *m) -> bool {
        std::vector<const value_t *> stack;
        for (size_t i = 0; i < m->num_inputs(); ++i)
            stack.push_back(m->get_input_value(i).get());
        std::unordered_set<const value_t *> seen;
        while (!stack.empty()) {
            const value_t *v = stack.back();
            stack.pop_back();
            if (!v || !seen.insert(v).second) continue;
            if (!v->has_producer()) continue;
            op_t &prod = v->get_producer();
            if (&prod != m && prod.get_kind() == graph::op_kind::_matmul)
                return true;
            for (size_t i = 0; i < prod.num_inputs(); ++i)
                stack.push_back(prod.get_input_value(i).get());
        }
        return false;
    };
    op_t *mm2_op = nullptr;
    for (const auto &op : subgraph_->get_ops()) {
        if (op->get_kind() != graph::op_kind::_matmul) continue;
        if (traces_to_other_matmul(op.get())) {
            mm2_op = op.get();
            break;
        }
    }

    // Capture the geometry and user strides for the execute path.
    ndims_ = static_cast<int>(sdp_cfg_.ndims);
    batch_ = sdp_cfg_.batch_size;
    num_head_q_ = sdp_cfg_.num_head_q;
    const dim_t num_head_kv = sdp_cfg_.num_head_kv;
    group_head_ = num_head_q_ / num_head_kv;
    seq_q_ = sdp_cfg_.seq_len_q;
    const dim_t hs_qk = sdp_cfg_.head_size_qk;
    hs_v_ = sdp_cfg_.head_size_v;

    const auto &gi = sdp_cfg_.graph_inport;
    idx_q_ = gi[sdp_decomp_config_t::mm1_src];
    idx_k_ = gi[sdp_decomp_config_t::mm1_wei];
    idx_v_ = gi[sdp_decomp_config_t::mm2_wei];
    idx_scale_ = gi[sdp_decomp_config_t::mm1_scale];
    idx_cond_ = gi[sdp_decomp_config_t::select_condition];
    idx_fill_ = gi[sdp_decomp_config_t::select_other_input];
    idx_mask_ = gi[sdp_decomp_config_t::mm1_add];

    q_strides_ = ltw(inputs[idx_q_]).vstrides();
    k_strides_ = ltw(inputs[idx_k_]).vstrides();
    v_strides_ = ltw(inputs[idx_v_]).vstrides();
    // Output strides come from mm2's own output value (driver [B,H,S,D] axis
    // order, transpose-fold aware), not the partition output tensor which may
    // be in a permuted axis order after a folded StaticTranspose. Fall back to
    // the partition output tensor if mm2 could not be located.
    o_strides_ = mm2_op
            ? ltw(mm2_op->get_output_value(0)->get_logical_tensor()).vstrides()
            : ltw(outputs[0]).vstrides();
    // K holds seq_kv on its last axis when consumed as K^T (transpose_b == 0),
    // otherwise on its second-to-last axis (natural [.., seq_kv, head_size]).
    seq_kv_ = mm1_transpose_b_ ? ltw(inputs[idx_k_]).vdims()[ndims_ - 2]
                               : ltw(inputs[idx_k_]).vdims()[ndims_ - 1];
    if (has_select_) cond_strides_ = ltw(inputs[idx_cond_]).vstrides();

    // Alternative path: the decoupled query-axis blocked / two-pass-softmax
    // driver. It owns its own BRGEMM kernels and scratch sizing; the online
    // epilogue below is skipped entirely.
    if (blocked_) {
        sdp_blocked_params_t bp;
        bp.ndims = ndims_;
        bp.batch = batch_;
        bp.num_head_q = num_head_q_;
        bp.group_head = group_head_;
        bp.seq_q = seq_q_;
        bp.seq_kv = seq_kv_;
        bp.head_size_qk = hs_qk;
        bp.head_size_v = hs_v_;
        bp.q_strides = q_strides_;
        bp.k_strides = k_strides_;
        bp.v_strides = v_strides_;
        bp.o_strides = o_strides_;
        bp.cond_strides = cond_strides_;
        bp.has_select = has_select_;
        bp.select_fusiable = select_fusiable_;
        bp.mm1_transpose_b = mm1_transpose_b_;
        bp.softmax_inf_as_zero = softmax_inf_as_zero_;
        // Compute type (Q/K/V) and output type. The scores/pv tiles stay f32
        // (BRGEMM accumulates in f32); the driver down-converts P and the
        // output to these types.
        bp.mm_dt = static_cast<dnnl::impl::data_type_t>(
                ltw(inputs[idx_q_]).data_type());
        bp.out_dt = static_cast<dnnl::impl::data_type_t>(
                ltw(outputs[0]).data_type());
        // mm1 post-op chain in graph order: scale (binary-mul, scalar rhs,
        // already reciprocated at execute if Divide) then the additive
        // attention mask (binary-add, tensor rhs offset per batch/head/tile).
        // Soft-cap entries will be appended here as they are enabled.
        if (has_scale_) {
            sdp_mm1_post_op_t sc;
            sc.alg = dnnl::impl::alg_kind::binary_mul;
            sc.is_binary = true;
            sc.rhs_is_scalar = true;
            sc.rhs_dt = dnnl::impl::data_type::f32;
            bp.mm1_post_ops.push_back(sc);
        }
        if (has_mask_) {
            sdp_mm1_post_op_t mk;
            mk.alg = dnnl::impl::alg_kind::binary_add;
            mk.is_binary = true;
            mk.rhs_is_scalar = false;
            mk.rhs_dt = static_cast<dnnl::impl::data_type_t>(
                    ltw(inputs[idx_mask_]).data_type());
            mk.rhs_dims = ltw(inputs[idx_mask_]).vdims();
            mk.rhs_strides = ltw(inputs[idx_mask_]).vstrides();
            bp.mm1_post_ops.push_back(mk);
        }
        CHECK(blocked_driver_.init(bp, eng));
        nthr_ = blocked_driver_.nthr();
        blocked_scratch_total_ = blocked_driver_.scratch_total(nthr_);
        return status::success;
    }

    // Create the BRGEMM kernels. Shapes/leading dims are identical for every
    // slice, so one kernel per (full/tail) tile width suffices.
    //   mm1 (beta=0): scores_tile[seq_q, w] = Q[seq_q, hs_qk] * K[hs_qk, w]
    //   mm2 (beta=0): pv_tile[seq_q, hs_v]  = P_tile[seq_q, w] * V[w, hs_v]
    // where w is the KV tile width: kv_blk_ for full tiles, and the
    // seq_kv % kv_blk_ remainder for the last tile. mm2 writes a per-tile
    // buffer; the online-softmax epilogue accumulates it into the running
    // output.
    kv_blk_ = nstl::min<dim_t>(seq_kv_, kv_block_width);
    const dim_t kv_tail = seq_kv_ % kv_blk_;

    auto create_brgemm
            = [&](brgemm_kernel_t **out, float beta, dim_t M, dim_t N, dim_t K,
                      dim_t lda, dim_t ldb, dim_t ldc) -> status_t {
        brgemm_desc_t brg;
        CHECK(brgemm_desc_init(&brg, isa_undef, brgemm_addr,
                dnnl::impl::data_type::f32, dnnl::impl::data_type::f32,
                /*transA=*/false, /*transB=*/false, brgemm_row_major,
                /*alpha=*/1.0f, beta, lda, ldb, ldc, M, N, K,
                /*strides=*/nullptr));
        CHECK(brgemm_desc_finalize(&brg));
        brgemm_kernel_t *k = nullptr;
        CHECK(brgemm_kernel_create(&k, brg));
        *out = k;
        return status::success;
    };

    const dim_t row_dim = ndims_ - 2;
    // mm1 writes a dense [seq_q, w] tile (ldc = w); mm2 multiplies that dense
    // tile by V into a dense [seq_q, hs_v] per-tile buffer (beta=0). The
    // running normalized output is combined in the epilogue, so magnitudes
    // stay O(|V|) (matches the decomp kernel's normalize-before accuracy).
    auto create_tile_kernels
            = [&](brgemm_kernel_t **mm1, brgemm_kernel_t **mm2, dim_t w) {
        CHECK(create_brgemm(mm1, /*beta=*/0.0f, seq_q_, w, hs_qk,
                /*lda=*/q_strides_[row_dim],
                /*ldb=*/k_strides_[row_dim],
                /*ldc=*/w));
        CHECK(create_brgemm(mm2, /*beta=*/0.0f, seq_q_, hs_v_, w,
                /*lda=*/w, /*ldb=*/v_strides_[row_dim], /*ldc=*/hs_v_));
        return status::success;
    };

    CHECK(create_tile_kernels(&mm1_kernel_, &mm2_kernel_, kv_blk_));
    if (kv_tail != 0)
        CHECK(create_tile_kernels(
                &mm1_tail_kernel_, &mm2_tail_kernel_, kv_tail));

    // Build the JIT online-softmax epilogue (AVX2 IR). One softmax kernel per
    // tile width (full/tail), plus one acc-renormalization kernel. If AVX2 is
    // unavailable the execute path falls back to the scalar epilogue.
    if (mayiuse(avx2)) {
        using namespace sdp_softmax_ir;
        // Condition tensor row stride in elements; columns are contiguous.
        const int cond_stride
                = has_select_ ? static_cast<int>(cond_strides_[row_dim]) : 0;
        const int sq = static_cast<int>(seq_q_);
        auto build_ir_kernel = [](std::unique_ptr<softmax_ir_kernel_t> &slot,
                                       ir_t ir) -> status_t {
            std::unique_ptr<softmax_ir_kernel_t> k(
                    new softmax_ir_kernel_t(std::move(ir)));
            CHECK(k->create_kernel());
            slot = std::move(k);
            return status::success;
        };
        status_t st = build_ir_kernel(softmax_ir_kernel_,
                build_softmax_tile_ir(sq, static_cast<int>(kv_blk_),
                        has_select_, select_fusiable_, cond_stride));
        if (st == status::success && kv_tail != 0)
            st = build_ir_kernel(softmax_tail_ir_kernel_,
                    build_softmax_tile_ir(sq, static_cast<int>(kv_tail),
                            has_select_, select_fusiable_, cond_stride));
        if (st == status::success)
            st = build_ir_kernel(acc_renorm_ir_kernel_,
                    build_acc_renorm_ir(sq, static_cast<int>(hs_v_)));
        use_ir_epilogue_ = st == status::success;
    }

    // Book one online-softmax working set per thread; execute_impl slices this
    // by thread id instead of allocating std::vectors in the parallel loop.
    nthr_ = dnnl_get_max_threads();
    const size_t fsz = sizeof(float);
    registrar_t reg = sdp_registry_.registrar();
    reg.book(mem_scores, static_cast<size_t>(seq_q_) * kv_blk_ * fsz);
    reg.book(mem_acc, static_cast<size_t>(seq_q_) * hs_v_ * fsz);
    reg.book(mem_pv, static_cast<size_t>(seq_q_) * hs_v_ * fsz);
    reg.book(mem_row_max, static_cast<size_t>(seq_q_) * fsz);
    reg.book(mem_row_denom, static_cast<size_t>(seq_q_) * fsz);
    reg.book(mem_old_coef, static_cast<size_t>(seq_q_) * fsz);

    return status::success;
#endif
}

status_t sdp_fused_brgemm_kernel_t::execute_impl(stream_t *strm,
        const std::vector<tensor_t> &inputs,
        const std::vector<tensor_t> &outputs, const tensor_t *scratchpad_buf) {
    UNUSED(strm);
#if !DNNL_X64
    UNUSED(inputs);
    UNUSED(outputs);
    UNUSED(scratchpad_buf);
    return status::unimplemented;
#else
    using namespace dnnl::impl::cpu::x64;

    auto *q_base = static_cast<const char *>(inputs[idx_q_].get_data_handle());
    auto *k_base = static_cast<const char *>(inputs[idx_k_].get_data_handle());
    auto *v_base = static_cast<const char *>(inputs[idx_v_].get_data_handle());
    auto *o_base = static_cast<char *>(outputs[0].get_data_handle());

    float scale_val = 1.0f;
    if (has_scale_) {
        scale_val = *static_cast<const float *>(
                inputs[idx_scale_].get_data_handle());
        if (scale_is_divide_) scale_val = 1.0f / scale_val;
    }
    float fill_val = 0.0f;
    const char *cond_base = nullptr;
    if (has_select_) {
        fill_val = *static_cast<const float *>(
                inputs[idx_fill_].get_data_handle());
        cond_base = static_cast<const char *>(
                inputs[idx_cond_].get_data_handle());
    }

    // Alternative path: run the decoupled blocked driver over its own scratch.
    if (blocked_) {
        sdp_blocked_run_args_t args;
        args.q = q_base;
        args.k = k_base;
        args.v = v_base;
        args.cond = cond_base;
        args.out = o_base;
        args.fill = fill_val;
        // rhs base pointers for the mm1 binary post-ops, in chain order. Only
        // the QK scale is present today; scale_val is a stable local that
        // outlives the execute call below.
        if (has_scale_) args.mm1_post_op_rhs.push_back(&scale_val);
        if (has_mask_)
            args.mm1_post_op_rhs.push_back(inputs[idx_mask_].get_data_handle());
        auto scratchpad = std::make_shared<scratchpad_t>(
                scratchpad_buf, blocked_scratch_total_, p_engine_);
        return blocked_driver_.execute(args, scratchpad->get_buffer(), nthr_);
    }

    const dim_t seq_q = seq_q_, seq_kv = seq_kv_, hs_v = hs_v_;
    const dim_t kv_blk = kv_blk_;
    const dim_t group = group_head_;
    const int ndims = ndims_;
    const int row_dim = ndims - 2;
    // Element strides for addressing a KV tile within K / V.
    const dim_t k_col = k_strides_[ndims - 1]; // K[.., hs, seq_kv]: seq_kv step
    const dim_t v_row = v_strides_[row_dim]; // V[.., seq_kv, hs_v]: kv step
    const dim_t o_row = o_strides_[row_dim];
    const dim_t o_col = o_strides_[ndims - 1];
    const dim_t cond_row = has_select_ ? cond_strides_[row_dim] : 0;
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    // Query-side offset (Q / out / select-cond carry the group axis).
    const auto q_side_off = [&](const std::vector<dim_t> &s, dim_t bo, dim_t bi,
                                    dim_t kvh, dim_t gid) -> dim_t {
        return ndims == 4 ? bo * s[0] + bi * s[1]
                          : bo * s[0] + kvh * s[1] + gid * s[2];
    };
    // KV-side offset (K / V; the group axis has extent 1).
    const auto kv_side_off
            = [&](const std::vector<dim_t> &s, dim_t bo, dim_t kvh) -> dim_t {
        return bo * s[0] + kvh * s[1];
    };

    // One online-softmax working set per thread, carved from the scratchpad.
    const size_t block_size = sdp_registry_.size();
    auto scratchpad = std::make_shared<scratchpad_t>(
            scratchpad_buf, block_size * nthr_, p_engine_);
    grantor_t var_grantor = sdp_registry_.grantor(scratchpad->get_buffer());

    parallel_nd_ext(
            nthr_, batch_, num_head_q_, [&](int tid, int, dim_t bo, dim_t bi) {
        const dim_t kvh = bi / group;
        const dim_t gid = bi % group;

        const float *q_ptr = reinterpret_cast<const float *>(q_base
                + q_side_off(q_strides_, bo, bi, kvh, gid) * sizeof(float));
        const float *k_ptr = reinterpret_cast<const float *>(
                k_base + kv_side_off(k_strides_, bo, kvh) * sizeof(float));
        const float *v_ptr = reinterpret_cast<const float *>(
                v_base + kv_side_off(v_strides_, bo, kvh) * sizeof(float));
        float *o_ptr = reinterpret_cast<float *>(o_base
                + q_side_off(o_strides_, bo, bi, kvh, gid) * sizeof(float));
        const uint8_t *c_ptr = has_select_
                ? reinterpret_cast<const uint8_t *>(cond_base
                          + q_side_off(cond_strides_, bo, bi, kvh, gid)
                                  * sizeof(uint8_t))
                : nullptr;

        // Online-softmax running state kept in a numerically stable form: the
        // accumulator (acc) holds the *normalized* output so far, so its
        // magnitude stays O(|V|). Per tile, mm2 produces the raw P_tile*V_tile
        // into pv, then acc is renormalized. row_max (m) and row_denom (l) are
        // the running max and denominator. This replaces the full [seq_q,
        // seq_kv] scores materialization with [seq_q, kv_blk] + [seq_q, hs_v].
        // Buffers are per-thread slices of the scratchpad (see compile_impl).
        float *scores = reinterpret_cast<float *>(
                var_grantor.get(mem_scores) + tid * block_size);
        float *acc = reinterpret_cast<float *>(
                var_grantor.get(mem_acc) + tid * block_size);
        float *pv = reinterpret_cast<float *>(
                var_grantor.get(mem_pv) + tid * block_size);
        float *row_max = reinterpret_cast<float *>(
                var_grantor.get(mem_row_max) + tid * block_size);
        float *row_denom = reinterpret_cast<float *>(
                var_grantor.get(mem_row_denom) + tid * block_size);
        // Per-row renormalization coefficient for the current tile.
        float *old_coef = reinterpret_cast<float *>(
                var_grantor.get(mem_old_coef) + tid * block_size);
        // acc/row_max/row_denom carry running state across tiles, so they must
        // be initialized (scratchpad memory is uninitialized).
        std::fill(row_max, row_max + seq_q, neg_inf);
        std::fill(row_denom, row_denom + seq_q, 0.0f);
        std::fill(acc, acc + static_cast<size_t>(seq_q) * hs_v, 0.0f);

        for (dim_t kv0 = 0; kv0 < seq_kv; kv0 += kv_blk) {
            const dim_t w = nstl::min(kv_blk, seq_kv - kv0);
            const bool is_tail = w != kv_blk;
            const auto *mm1 = is_tail ? mm1_tail_kernel_ : mm1_kernel_;
            const auto *mm2 = is_tail ? mm2_tail_kernel_ : mm2_kernel_;

            // mm1: scores_tile[seq_q, w] = Q * K[:, kv0 : kv0 + w].
            brgemm_batch_element_t batch1;
            batch1.ptr.A = q_ptr;
            batch1.ptr.B = k_ptr + kv0 * k_col;
            brgemm_kernel_execute(mm1, 1, &batch1, scores, nullptr);

            // Online-softmax epilogue over this KV tile: apply scale + mask,
            // update the running max/denom, and form P_tile = exp(s - m_new).
            if (use_ir_epilogue_) {
                const auto &sm = is_tail ? softmax_tail_ir_kernel_
                                         : softmax_ir_kernel_;
                sdp_softmax_ir::softmax_row_args_t sargs;
                sargs.scores = scores;
                sargs.scale = &scale_val;
                sargs.m = row_max;
                sargs.l = row_denom;
                sargs.old_coef = old_coef;
                // cond points at this tile's first column (row 0); the kernel
                // advances by the compiled cond row stride per row.
                sargs.cond = c_ptr ? c_ptr + kv0 : nullptr;
                sargs.fill = &fill_val;
                (*sm)(&sargs);
            } else {
                for (dim_t i = 0; i < seq_q; ++i) {
                    float *srow = scores + i * w;
                    const uint8_t *crow
                            = c_ptr ? c_ptr + i * cond_row : nullptr;
                    float tile_max = neg_inf;
                    for (dim_t j = 0; j < w; ++j) {
                        float v = srow[j] * scale_val;
                        if (crow) {
                            const bool cond = crow[kv0 + j] != 0;
                            // not-fusiable (p1): cond ? fill : scores
                            // fusiable    (p2): cond ? scores : fill
                            const bool keep = select_fusiable_ ? cond : !cond;
                            if (!keep) v = fill_val;
                        }
                        srow[j] = v;
                        if (v > tile_max) tile_max = v;
                    }
                    const float m_old = row_max[i];
                    const float l_old = row_denom[i];
                    const float m_new = nstl::max(m_old, tile_max);
                    // corr rescales the old contributions to the new max; it is
                    // 0 for the first (m_old == -inf) tile.
                    const float corr
                            = m_old == neg_inf ? 0.0f : expf(m_old - m_new);
                    float tile_sum = 0.0f;
                    for (dim_t j = 0; j < w; ++j) {
                        const float e = expf(srow[j] - m_new);
                        srow[j] = e;
                        tile_sum += e;
                    }
                    const float l_new = l_old * corr + tile_sum;
                    const float inv = l_new > 0.0f ? 1.0f / l_new : 0.0f;
                    row_denom[i] = l_new;
                    row_max[i] = m_new;
                    // Pre-normalize P by the running denominator so mm2
                    // accumulates O(1) magnitudes (matches the decomp kernel's
                    // accuracy). acc then holds U/l; refresh it with old_coef =
                    // corr*l_old/l_new.
                    for (dim_t j = 0; j < w; ++j)
                        srow[j] *= inv;
                    old_coef[i] = corr * l_old * inv;
                }
            }

            // mm2: pv[seq_q, hs_v] = P_norm_tile * V[kv0 : kv0 + w, :].
            brgemm_batch_element_t batch2;
            batch2.ptr.A = scores;
            batch2.ptr.B = v_ptr + kv0 * v_row;
            brgemm_kernel_execute(mm2, 1, &batch2, pv, nullptr);

            // Renormalize the running output: acc = old_coef*acc + pv.
            if (use_ir_epilogue_) {
                sdp_softmax_ir::acc_renorm_args_t aargs;
                aargs.acc = acc;
                aargs.pv = pv;
                aargs.old_coef = old_coef;
                (*acc_renorm_ir_kernel_)(&aargs);
            } else {
                for (dim_t i = 0; i < seq_q; ++i) {
                    float *arow = acc + i * hs_v;
                    const float *prow = pv + i * hs_v;
                    const float a = old_coef[i];
                    for (dim_t d = 0; d < hs_v; ++d)
                        arow[d] = a * arow[d] + prow[d];
                }
            }
        }

        // acc already holds the normalized output; scatter to user output.
        for (dim_t i = 0; i < seq_q; ++i) {
            const float *arow = acc + i * hs_v;
            float *out_row = o_ptr + i * o_row;
            for (dim_t d = 0; d < hs_v; ++d)
                out_row[d * o_col] = arow[d];
        }
    });

    return status::success;
#endif
}

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
