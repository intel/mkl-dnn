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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_BLOCKED_DRIVER_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_BLOCKED_DRIVER_HPP

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {

// Forward declaration to avoid pulling the BRGEMM / softmax headers into
// consumers.
struct primitive_desc_t;
namespace cpu {
namespace x64 {
struct brgemm_kernel_t;
namespace softmax_impl {
struct jit_softmax_kernel_base_t;
} // namespace softmax_impl
} // namespace x64
} // namespace cpu

namespace graph {
namespace dnnl_impl {

// -----------------------------------------------------------------------------
// Decoupled blocked-softmax SDPA driver (x64, fp32).
//
// This is a self-contained compute routine for the fused SDPA:
//   scores = Q * K^T ; softmax(scale * scores [+ select-mask]) ; out = P * V
// It deliberately takes ONLY plain arguments (dims, strides in elements, raw
// pointers, scalar scale/fill). It has no dependency on the graph IR
// (subgraph_t / logical_tensor_t / memory_planner) or on the primitive
// framework (exec_ctx_t). That keeps the optimization -- work-splitting and
// cache blocking around the BRGEMM microkernel plus a plain two-pass softmax --
// portable, so the same driver can later be wrapped either behind the graph
// backend kernel (current use) or behind a CPU dnnl_sdpa primitive with only
// glue code changing.
//
// Algorithm (contrast with the online/flash epilogue in sdp_fused_brgemm):
//   * Block over the QUERY axis (seq_q) into q_block-row tiles chosen so the
//     [q_block x seq_kv] score tile stays L2-resident.
//   * For each query tile: mm1 forms the FULL [q_block x seq_kv] scores, then a
//     plain, numerically-stable TWO-PASS softmax over the seq_kv axis (each
//     query row already sees every key, so no online recurrence is needed --
//     the result is exact), then mm2 forms [q_block x head_size_v].
//   * Parallelize over batch x num_head_q x query-tiles for finer granularity
//     than one work-item per (batch, head).
//
// Rationale: on CPU the softmax exp() is compute-bound and memory is cheap
// relative to GPU, so re-reading the score tile in a second pass costs little
// while avoiding the per-tile rescale of the online formulation.
// -----------------------------------------------------------------------------
// One mm1 (QK^T) post-op carried from the graph, in chain order. Binary
// entries multiply/add a right-hand side (the QK scale scalar, an additive
// attention mask, a soft-cap multiplier); eltwise entries apply an activation
// (soft-cap tanh). This is the library-level, graph-type-free description the
// driver folds into the BRGEMM store, mirroring decomp's sub_matmul1_attr.
struct sdp_mm1_post_op_t {
    dnnl::impl::alg_kind_t alg = dnnl::impl::alg_kind::undef;
    bool is_binary = false; // false => eltwise
    // Eltwise parameters (is_binary == false).
    float alpha = 0.0f;
    float beta = 0.0f;
    // Binary right-hand side (is_binary == true). A scalar rhs (dims all 1) is
    // a [1 x 1] broadcast applied uniformly and never offset; otherwise the rhs
    // is a [seq_q x seq_kv] tile addressed per query block.
    bool rhs_is_scalar = true;
    dnnl::impl::data_type_t rhs_dt = dnnl::impl::data_type::f32;
    // Full user dims / strides of the rhs tensor (length == ndims), used to
    // offset the base pointer per (batch, head, query-tile). Unused for scalars.
    std::vector<dim_t> rhs_dims, rhs_strides;
};

struct sdp_blocked_params_t {
    int ndims = 0;
    dim_t batch = 0;
    dim_t num_head_q = 0;
    // GQA group size: num_head_q / num_head_kv (>= 1).
    dim_t group_head = 1;
    dim_t seq_q = 0;
    dim_t seq_kv = 0;
    dim_t head_size_qk = 0;
    dim_t head_size_v = 0;

    // User strides in elements. Q / output / select-condition carry the group
    // axis; K / V have group extent 1 (broadcast over the group).
    std::vector<dim_t> q_strides, k_strides, v_strides, o_strides, cond_strides;

    bool has_select = false;
    // Select semantics: fusiable (p2) keeps scores where cond != 0 and writes
    // fill elsewhere; non-fusiable (p1) is the inverse.
    bool select_fusiable = false;

    // The mm1 (QK^T) post-op chain, carried verbatim from the graph in graph
    // order (scale / soft-cap / attention-mask; the select is handled
    // separately). Mirrors decomp's sub_matmul1_attr post-ops but sliced to the
    // per-query-tile shape and folded into the BRGEMM store. Empty when mm1 has
    // no post-ops.
    std::vector<sdp_mm1_post_op_t> mm1_post_ops;
};

// Runtime pointers / scalars, resolved per execute call.
struct sdp_blocked_run_args_t {
    const void *q = nullptr;
    const void *k = nullptr;
    const void *v = nullptr;
    const void *cond = nullptr; // uint8 select condition, or null
    void *out = nullptr;
    float fill = 0.0f;
    // Base pointers for the mm1 binary post-op right-hand sides, one per binary
    // entry in sdp_blocked_params_t::mm1_post_ops (in the same order; eltwise
    // entries consume none). Each is offset per (batch, head, query-tile) using
    // the entry's dims/strides. For a scalar rhs (e.g. the QK scale) the pointer
    // addresses a single element and is not offset.
    std::vector<const void *> mm1_post_op_rhs;
};

// Owns the BRGEMM kernels for the (full / query-tail) tiles and drives the
// blocked execute loop. x64-only; on other builds init() returns unimplemented.
class sdp_blocked_driver_t {
public:
    sdp_blocked_driver_t() = default;
    ~sdp_blocked_driver_t();
    sdp_blocked_driver_t(const sdp_blocked_driver_t &) = delete;
    sdp_blocked_driver_t &operator=(const sdp_blocked_driver_t &) = delete;

    // Create the BRGEMM kernels and compute the query blocking + per-thread
    // scratch size. Must be called once before execute(). The engine is used
    // to instantiate the reused jit softmax kernel; the execute path stays
    // engine-free.
    status_t init(const sdp_blocked_params_t &params, engine_t *engine);

    // Per-thread scratch requirement in bytes (scores + pv tiles). The caller
    // books nthr() * scratch_per_thread() bytes and passes the base pointer.
    size_t scratch_per_thread() const { return scratch_per_thread_; }
    int nthr() const { return nthr_; }

    // Run the blocked SDPA. scratch_base points at a buffer of at least
    // nthr() * scratch_per_thread() bytes; each thread slices its own block.
    status_t execute(const sdp_blocked_run_args_t &args, void *scratch_base,
            int nthr) const;

private:
    sdp_blocked_params_t p_;
    dim_t q_block_ = 0;
    dim_t q_tail_ = 0; // seq_q % q_block_ (0 if evenly divided)
    size_t scratch_per_thread_ = 0;
    int nthr_ = 0;

    // mm1: scores[m, seq_kv] = Q[m, hs_qk] * K[hs_qk, seq_kv]
    // mm2: pv[m, hs_v]       = P[m, seq_kv] * V[seq_kv, hs_v]
    // *_tail handle the ragged last query tile (m = q_tail_).
    cpu::x64::brgemm_kernel_t *mm1_kernel_ = nullptr;
    cpu::x64::brgemm_kernel_t *mm2_kernel_ = nullptr;
    cpu::x64::brgemm_kernel_t *mm1_tail_kernel_ = nullptr;
    cpu::x64::brgemm_kernel_t *mm2_tail_kernel_ = nullptr;

    // Reused vectorized softmax: the jit softmax kernel (max/exp/normalize over
    // the seq_kv axis, per query row) plus the primitive descriptor it reads
    // its config from. When use_jit_softmax_ is false (no jit impl available)
    // the execute path falls back to a scalar two-pass softmax. The kernel is
    // per-row, so one instance serves both the full and query-tail blocks.
    std::shared_ptr<primitive_desc_t> softmax_pd_;
    std::shared_ptr<cpu::x64::softmax_impl::jit_softmax_kernel_base_t>
            softmax_kernel_;
    bool use_jit_softmax_ = false;

    // When true, mm1 applies the (fusiable, dense-condition) select-mask via a
    // binary_select post-op at its store, so the pre-pass is skipped entirely.
    bool mm1_select_postop_ = false;
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
