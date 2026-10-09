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

#ifndef CPU_X64_SDPA_SDPA_FULL_SOFTMAX_HPP
#define CPU_X64_SDPA_SDPA_FULL_SOFTMAX_HPP

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"

#include "cpu/x64/brgemm/brgemm_types.hpp"

namespace dnnl {
namespace impl {

struct primitive_desc_t;

namespace cpu {
namespace x64 {

struct brgemm_kernel_t;
namespace softmax_impl {
struct jit_softmax_kernel_base_t;
} // namespace softmax_impl
namespace sdpa_full_softmax_select_ir {
class select_ir_kernel_t;
} // namespace sdpa_full_softmax_select_ir

// -----------------------------------------------------------------------------
// SDPA with softmax over the complete key axis (x64):
//   scores = Q*K^T ; P = softmax(scale*scores [+ select-mask], over seq_kv)
//   out = P*V
// Block over seq_q into q_block tiles. Each tile computes scores for all seq_kv
// keys, then applies softmax over that axis using the JIT softmax kernel.
// Parallelize over batch x num_head_q x query-tiles.
// -----------------------------------------------------------------------------
// One mm1 (QK^T) post-op in chain order, folded into the BRGEMM store: binary
// entries apply their algorithm to an RHS; eltwise entries apply an activation.
struct sdpa_mm1_post_op_t {
    dnnl::impl::alg_kind_t alg = dnnl::impl::alg_kind::undef;
    bool is_binary = false; // false => eltwise
    // Eltwise parameters (is_binary == false).
    float alpha = 0.0f;
    float beta = 0.0f;
    // Binary rhs: a scalar uses a [1 x 1] descriptor and its pointer is not
    // query-tile-offset. A tensor rhs uses a query-row by key-position tile;
    // axes of extent 1 broadcast.
    bool rhs_is_scalar = true;
    dnnl::impl::data_type_t rhs_dt = dnnl::impl::data_type::f32;
    // Full user dims / strides of the rhs (length == ndims), to offset the base
    // pointer per (batch, head, query-tile). Unused for scalars.
    std::vector<dim_t> rhs_dims, rhs_strides;
};

struct sdpa_full_softmax_params_t {
    int ndims = 0;
    dim_t batch = 0;
    dim_t num_head_q = 0;
    // GQA group size: num_head_q / num_head_kv (>= 1).
    dim_t group_head = 1;
    dim_t seq_q = 0;
    dim_t seq_kv = 0;
    dim_t head_size_qk = 0;
    dim_t head_size_v = 0;

    // Q/K/V and BRGEMM inputs must be f32; scores, P, and output are f32 too.
    dnnl::impl::data_type_t mm_dt = dnnl::impl::data_type::f32;
    // SDPA output must be f32.
    dnnl::impl::data_type_t out_dt = dnnl::impl::data_type::f32;

    // User strides in elements. Q / output / select-condition carry the group
    // axis; K / V have group extent 1 (broadcast over the group).
    std::vector<dim_t> q_strides, k_strides, v_strides, o_strides;
    // Condition strides in elements. Effective strides are zeroed for axes
    // whose extent is 1 in cond_dims.
    std::vector<dim_t> cond_strides;
    std::vector<dim_t> cond_dims;

    bool has_select = false;
    // When true, keep scores where cond == 0; otherwise keep where cond != 0.
    bool invert_select = false;

    // SoftMax "inf_as_zero" mode: a fully-masked row (all -inf) produces an
    // all-zero probability row instead of NaN. Selects the softmax primitive
    // alg_kind (softmax_accurate_inf_as_zero vs softmax_accurate).
    bool softmax_inf_as_zero = false;

    // Supported mm1 (QK^T) post-ops, applied in order: scale then attention
    // mask. Select is handled separately. These ops are sliced to the query-tile
    // shape and folded into the BRGEMM store. Soft-cap is not wired into this path.
    // Empty when mm1 has no post-ops.
    std::vector<sdpa_mm1_post_op_t> mm1_post_ops;
};

// Runtime pointers / scalars, resolved per execute call.
struct sdpa_full_softmax_run_args_t {
    const void *q = nullptr;
    const void *k = nullptr;
    const void *v = nullptr;
    const void *cond = nullptr; // uint8 select condition, or null
    void *out = nullptr;
    float fill = 0.0f;
    // Base pointers for the mm1 binary post-op right-hand sides, one per binary
    // entry in sdpa_full_softmax_params_t::mm1_post_ops (in the same order; eltwise
    // entries consume none). Each is offset per (batch, head, query-tile) using
    // the entry's dims/strides. For a scalar rhs (e.g. the QK scale) the pointer
    // addresses a single element and is not offset.
    std::vector<const void *> mm1_post_op_rhs;
};

// Derived compute configuration: JIT-free query/KV blocking and per-thread
// scratch layout. Pure scalar config (mirroring brgemm_matmul_conf_t), owned by
// the primitive_desc so the pd sizes its scratchpad from it; the finalized
// BRGEMM descriptors live separately in sdpa_full_softmax_descs_t (like brg_descs_).
struct sdpa_full_softmax_conf_t {
    sdpa_full_softmax_params_t params;
    dim_t q_block = 0, q_tail = 0;
    dim_t kv_block = 0, kv_tail = 0;
    // mm2 accumulates across kv-blocks: beta = 0 for a single block, else the
    // first block uses a beta = 0 kernel and the rest beta = 1.
    float mm2_beta = 0.0f;
    // A non-inverted select mask with a dense condition is folded into mm1 as
    // a binary_select post-op; the descriptors are built accordingly.
    bool mm1_select_postop = false;
    size_t scratch_per_thread = 0;
    int nthr = 0;

    size_t scratch_total(int n) const {
        return scratch_per_thread * static_cast<size_t>(n);
    }
};

// Finalized (not compiled) BRGEMM descriptors for the full-softmax SDPA, built once
// by configure() and owned by the primitive_desc (mirroring brgemm_matmul's
// brg_descs_, kept separate from the conf). Indexed [is_q_tail][is_kv_tail];
// mm_valid marks which slots were built (tail slots stay unbuilt when the
// remainder is 0). mm2_desc_beta0 is the beta = 0 full-kv-block mm2 descriptor
// used for the first block when seq_kv is tiled (per q-tile).
struct sdpa_full_softmax_descs_t {
    brgemm_desc_t mm1_desc[2][2], mm2_desc[2][2], mm2_desc_beta0[2];
    bool mm_valid[2][2] = {};
    bool beta0_valid[2] = {};
};

// Compiled kernels for the full-softmax SDPA, owned by the primitive. The BRGEMM
// kernels are raw handles freed in the destructor; the jit softmax and select
// pre-pass kernels are held by shared_ptr.
struct sdpa_full_softmax_kernels_t {
    // Declared out-of-line (defined in sdpa_full_softmax.cpp): the shared_ptr members
    // hold forward-declared kernel types, so construction/destruction must be
    // instantiated where those types are complete.
    sdpa_full_softmax_kernels_t();
    ~sdpa_full_softmax_kernels_t();
    sdpa_full_softmax_kernels_t(const sdpa_full_softmax_kernels_t &) = delete;
    sdpa_full_softmax_kernels_t &operator=(const sdpa_full_softmax_kernels_t &)
            = delete;

    // mm1/mm2 BRGEMM kernels indexed [is_q_tail][is_kv_tail]; mm2_kernels_beta0
    // is the beta = 0 first-block mm2 (per q-tile), built only when tiled.
    brgemm_kernel_t *mm1_kernels[2][2] = {};
    brgemm_kernel_t *mm2_kernels[2][2] = {};
    brgemm_kernel_t *mm2_kernels_beta0[2] = {};

    // Vectorized JIT softmax kernel and its pd. execute() invokes the kernel
    // for each score row.
    std::shared_ptr<primitive_desc_t> softmax_pd;
    std::shared_ptr<softmax_impl::jit_softmax_kernel_base_t> softmax_kernel;

    // Final select-in-mm1 decision after kernel compilation (configure's
    // decision, possibly downgraded if the compiled ukernel rejects the
    // post-op); execute reads this, not the conf.
    bool mm1_select_postop = false;

    // Standalone JIT select-mask pre-pass kernel (null unless a select mask is
    // present and not folded into mm1); the condition is dense along seq_kv
    // (cond_col == 1, guaranteed by the pd).
    std::shared_ptr<sdpa_full_softmax_select_ir::select_ir_kernel_t>
            select_kernel;
};

// Query-blocked SDPA with softmax over the complete key axis, folded into the
// CPU primitive: the primitive_desc owns the conf + descs (configure), the
// primitive owns the kernels (create_kernels) and drives the loop (execute).
namespace sdpa_full_softmax {

// Compute the derived configuration (query/KV blocking and per-thread scratch
// layout) and build the finalized BRGEMM descriptors into `descs`.
status_t configure(const sdpa_full_softmax_params_t &params,
        sdpa_full_softmax_conf_t &conf, sdpa_full_softmax_descs_t &descs);

// JIT-compile the BRGEMM + softmax kernels from a conf + descs produced by
// configure(), into `kernels`. The engine instantiates the reused jit softmax
// kernel; the execute path stays engine-free.
status_t create_kernels(const sdpa_full_softmax_conf_t &conf,
        const sdpa_full_softmax_descs_t &descs, engine_t *engine,
        sdpa_full_softmax_kernels_t &kernels);

// Run the full-softmax SDPA. scratch_base points at a buffer of at least
// conf.scratch_total(nthr) bytes; each thread slices its own per-thread block
// and the shared transposed-K buffer (if any) follows them.
status_t execute(const sdpa_full_softmax_conf_t &conf,
        const sdpa_full_softmax_kernels_t &kernels,
        const sdpa_full_softmax_run_args_t &args, void *scratch_base, int nthr);

} // namespace sdpa_full_softmax

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
