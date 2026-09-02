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

#if DNNL_X64
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
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

    p_engine_ = make_dnnl_engine(*eng);

    // Get subgraph from the deep copied partition.
    subgraph_ = std::make_shared<subgraph_t>(
            part->get_ops(), p_engine_, part->get_fpmath_mode(), false, true);
    BACKEND_DNNL_CHECK(set_given_inputs_outputs(subgraph_, inputs, outputs));

    // Detect whether the scale op is a division before lowering rewrites the
    // graph op kinds into dnnl_binary.
    for (const auto &op : subgraph_->get_ops()) {
        if (op->get_kind() == graph::op_kind::Divide) scale_is_divide_ = true;
    }

    // Validate the SDP pattern and extract dims/flags. This fused kernel is
    // fp32-only, so the quantized lowering passes are skipped.
    //
    // TODO(MFDNN-14723): initial_check() also enforces the sdp_decomp
    // RATIO/thread gate. The fused kernel will parallelise over the query
    // sequence, so this must eventually be replaced by a fused-specific check.
    if (!sdp_cfg_.initial_check(subgraph_, inputs, outputs))
        return status::unimplemented;

    // First iteration only supports the plain fp32 GQA pattern:
    // QK^T -> scale -> select-mask -> softmax -> PV.
    has_scale_ = sdp_cfg_.has_scale;
    has_select_ = sdp_cfg_.has_select;
    select_fusiable_ = sdp_cfg_.select_fusiable;
    VCHECK_SDP_FUSED_BRGEMM(!sdp_cfg_.has_soft_capping, status::unimplemented,
            "fused kernel does not support soft-capping yet");
    VCHECK_SDP_FUSED_BRGEMM(!sdp_cfg_.has_attention_mask, status::unimplemented,
            "fused kernel does not support additive mask yet");

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

    // Capture the geometry and user strides for the execute path.
    ndims_ = static_cast<int>(sdp_cfg_.ndims);
    batch_ = sdp_cfg_.batch_size;
    num_head_kv_ = sdp_cfg_.num_head_kv;
    num_head_q_ = sdp_cfg_.num_head_q;
    group_head_ = num_head_q_ / num_head_kv_;
    seq_q_ = sdp_cfg_.seq_len_q;
    hs_qk_ = sdp_cfg_.head_size_qk;
    hs_v_ = sdp_cfg_.head_size_v;

    const auto &gi = sdp_cfg_.graph_inport;
    idx_q_ = gi[sdp_decomp_config_t::mm1_src];
    idx_k_ = gi[sdp_decomp_config_t::mm1_wei];
    idx_v_ = gi[sdp_decomp_config_t::mm2_wei];
    idx_scale_ = gi[sdp_decomp_config_t::mm1_scale];
    idx_cond_ = gi[sdp_decomp_config_t::select_condition];
    idx_fill_ = gi[sdp_decomp_config_t::select_other_input];

    q_strides_ = ltw(inputs[idx_q_]).vstrides();
    k_strides_ = ltw(inputs[idx_k_]).vstrides();
    v_strides_ = ltw(inputs[idx_v_]).vstrides();
    o_strides_ = ltw(outputs[0]).vstrides();
    // K is stored as [.., head_size, seq_kv]; its last dim is seq_kv.
    seq_kv_ = ltw(inputs[idx_k_]).vdims()[ndims_ - 1];
    if (has_select_) cond_strides_ = ltw(inputs[idx_cond_]).vstrides();

    // Create the BRGEMM kernels. Shapes/leading dims are identical for every
    // slice, so one kernel per (full/tail) tile width suffices.
    //   mm1 (beta=0): scores_tile[seq_q, w] = Q[seq_q, hs_qk] * K[hs_qk, w]
    //   mm2 (beta=1): out[seq_q, hs_v]     += P_tile[seq_q, w] * V[w, hs_v]
    // where w is the KV tile width (kv_blk_ for full tiles, kv_tail_ for the
    // last one). mm2 accumulates across tiles for the online softmax.
    const dim_t second_last = ndims_ - 2;
    kv_blk_ = nstl::min<dim_t>(seq_kv_, 512);
    kv_tail_ = seq_kv_ % kv_blk_;

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

    // mm1 writes a dense [seq_q, w] tile (ldc = w); mm2 multiplies that dense
    // tile by V into a dense [seq_q, hs_v] per-tile buffer (beta=0). The
    // running normalized output is combined in the epilogue, so magnitudes
    // stay O(|V|) (matches the decomp kernel's normalize-before accuracy).
    auto create_tile_kernels
            = [&](brgemm_kernel_t **mm1, brgemm_kernel_t **mm2, dim_t w) {
        CHECK(create_brgemm(mm1, /*beta=*/0.0f, seq_q_, w, hs_qk_,
                /*lda=*/q_strides_[second_last],
                /*ldb=*/k_strides_[second_last],
                /*ldc=*/w));
        CHECK(create_brgemm(mm2, /*beta=*/0.0f, seq_q_, hs_v_, w,
                /*lda=*/w, /*ldb=*/v_strides_[second_last], /*ldc=*/hs_v_));
        return status::success;
    };

    CHECK(create_tile_kernels(&mm1_kernel_, &mm2_kernel_, kv_blk_));
    if (kv_tail_ != 0)
        CHECK(create_tile_kernels(
                &mm1_tail_kernel_, &mm2_tail_kernel_, kv_tail_));

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

    const int second_last = ndims_ - 2;
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

    const dim_t cond_row = has_select_ ? cond_strides_[second_last] : 0;

    auto *mm1_full = mm1_kernel_;
    auto *mm2_full = mm2_kernel_;
    auto *mm1_tail = mm1_tail_kernel_;
    auto *mm2_tail = mm2_tail_kernel_;

    const dim_t seq_q = seq_q_, seq_kv = seq_kv_, hs_v = hs_v_;
    const dim_t kv_blk = kv_blk_;
    const dim_t group = group_head_;
    const int ndims = ndims_;
    // Element strides for addressing a KV tile within K / V.
    const dim_t k_col = k_strides_[ndims - 1]; // K[.., hs, seq_kv]: seq_kv step
    const dim_t v_row = v_strides_[second_last]; // V[.., seq_kv, hs_v]: kv step
    const dim_t o_row = o_strides_[second_last];
    const dim_t o_col = o_strides_[ndims - 1];
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
            const auto *mm1 = is_tail ? mm1_tail : mm1_full;
            const auto *mm2 = is_tail ? mm2_tail : mm2_full;

            // mm1: scores_tile[seq_q, w] = Q * K[:, kv0 : kv0 + w].
            brgemm_batch_element_t batch1;
            batch1.ptr.A = q_ptr;
            batch1.ptr.B = k_ptr + kv0 * k_col;
            brgemm_kernel_execute(mm1, 1, &batch1, scores, nullptr);

            // Online-softmax epilogue over this KV tile: apply scale + mask,
            // update the running max/denom, and form P_tile = exp(s - m_new).
            for (dim_t i = 0; i < seq_q; ++i) {
                float *srow = scores + i * w;
                const uint8_t *crow = c_ptr ? c_ptr + i * cond_row : nullptr;
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
                // corr rescales the old contributions to the new max; it is 0
                // for the first (m_old == -inf) tile.
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
                // Pre-normalize P by the running denominator so mm2 accumulates
                // O(1) magnitudes (matches the decomp kernel's accuracy). acc
                // then holds U/l; refresh it with old_coef = corr*l_old/l_new.
                for (dim_t j = 0; j < w; ++j)
                    srow[j] *= inv;
                old_coef[i] = corr * l_old * inv;
            }

            // mm2: pv[seq_q, hs_v] = P_norm_tile * V[kv0 : kv0 + w, :].
            brgemm_batch_element_t batch2;
            batch2.ptr.A = scores;
            batch2.ptr.B = v_ptr + kv0 * v_row;
            brgemm_kernel_execute(mm2, 1, &batch2, pv, nullptr);

            // Renormalize the running output: acc = old_coef*acc + pv.
            for (dim_t i = 0; i < seq_q; ++i) {
                float *arow = acc + i * hs_v;
                const float *prow = pv + i * hs_v;
                const float a = old_coef[i];
                for (dim_t d = 0; d < hs_v; ++d)
                    arow[d] = a * arow[d] + prow[d];
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
