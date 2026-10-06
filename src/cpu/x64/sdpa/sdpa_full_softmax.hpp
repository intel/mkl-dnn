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
// Full-softmax SDPA compute (x64):
//   scores = Q*K^T ; softmax(scale*scores [+ select-mask]) ; out = P*V
// Block over seq_q into q_block tiles (each [q_block x seq_kv] score tile stays
// L2-resident); per tile form the full scores, run an exact two-pass softmax
// over seq_kv, then mm2. Parallelize over batch x num_head_q x query-tiles.
// Two-pass rather than the online-softmax strategy because on CPU exp() is
// compute-bound and re-reading the score tile is cheap.
// -----------------------------------------------------------------------------
// One mm1 (QK^T) post-op in chain order, folded into the BRGEMM store: a binary
// multiplies/adds a right-hand side (scale scalar, additive mask, soft-cap), an
// eltwise applies an activation (soft-cap tanh).
struct sdpa_mm1_post_op_t {
    dnnl::impl::alg_kind_t alg = dnnl::impl::alg_kind::undef;
    bool is_binary = false; // false => eltwise
    // Eltwise parameters (is_binary == false).
    float alpha = 0.0f;
    float beta = 0.0f;
    // Binary rhs (is_binary == true): a scalar rhs (dims all 1) is a [1 x 1]
    // broadcast never offset; otherwise a [seq_q x seq_kv] tile per query block.
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

    // Compute data type for Q/K/V and the mm1/mm2 BRGEMM inputs (f32/f16/bf16).
    // The scores and pv tiles are always accumulated in f32 (BRGEMM C is f32);
    // the transposed-K buffer and P (mm2's A operand) are materialised in this
    // type, and the softmax runs on the f32 scores.
    dnnl::impl::data_type_t mm_dt = dnnl::impl::data_type::f32;
    // SDPA output (result) data type; the f32 pv tile is down-converted to it.
    dnnl::impl::data_type_t out_dt = dnnl::impl::data_type::f32;

    // User strides in elements. Q / output / select-condition carry the group
    // axis; K / V have group extent 1 (broadcast over the group).
    std::vector<dim_t> q_strides, k_strides, v_strides, o_strides, cond_strides;
    // Logical dims of the select-condition tensor; a dim of 1 is a broadcast
    // broadcast axis whose (meaningless) stride must contribute 0.
    std::vector<dim_t> cond_dims;

    bool has_select = false;
    // Select semantics: fusiable (p2) keeps scores where cond != 0 and writes
    // fill elsewhere; non-fusiable (p1) is the inverse.
    bool select_fusiable = false;

    // mm1 (QK^T) transpose_b: when set, K is stored as [.., seq_kv, head_size]
    // and the mm1 BRGEMM transposes it; otherwise K is [.., head_size, seq_kv].
    bool mm1_transpose_b = false;

    // SoftMax "inf_as_zero" mode: a fully-masked row (all -inf) produces an
    // all-zero probability row instead of NaN. Selects the softmax primitive
    // alg_kind (softmax_accurate_inf_as_zero vs softmax_accurate).
    bool softmax_inf_as_zero = false;

    // The mm1 (QK^T) post-op chain, carried verbatim from the graph in graph
    // order (scale / soft-cap / attention-mask; the select is handled
    // separately). Mirrors decomp's sub_matmul1_attr post-ops but sliced to the
    // per-query-tile shape and folded into the BRGEMM store. Empty when mm1 has
    // no post-ops.
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

// AMX tile configuration for a BRGEMM kernel. Non-AMX ISAs leave need_config
// false and wsp_size 0; AMX (bf16/f16) must load `palette` via
// amx_tile_configure() before running and needs a wsp_size-byte tile-store
// scratch per thread.
struct sdpa_amx_cfg_t {
    bool need_config = false;
    size_t wsp_size = 0;
    char palette[64] = {};
};

// Derived compute configuration: the JIT-free arithmetic (query/KV blocking,
// per-thread + global scratch layout, AMX palette/wsp). Pure scalar config
// (mirroring brgemm_matmul_conf_t), computed once by configure() and owned by
// the primitive_desc so the pd sizes its scratchpad from it; the finalized
// BRGEMM descriptors live separately in sdpa_full_softmax_descs_t (like brg_descs_).
struct sdpa_full_softmax_conf_t {
    sdpa_full_softmax_params_t params;
    dim_t q_block = 0, q_tail = 0;
    dim_t kv_block = 0, kv_tail = 0;
    // mm2 accumulates across kv-blocks: beta = 0 for a single block, else the
    // first block uses a beta = 0 kernel and the rest beta = 1.
    float mm2_beta = 0.0f;
    // mm2 writes its f32 result directly into the user output (no pv+scatter).
    bool mm2_direct = false;
    // BRGEMM B VNNI pack factor (1 plain, 2 VNNI2).
    dim_t b_k_pack = 1;
    bool mm1_transpose_k = false;
    dim_t k_seq_stride = 0, k_hs_stride = 0;
    // The (fusiable, dense-condition) select mask is folded into mm1 as a
    // binary_select post-op; the descriptors are built accordingly.
    bool mm1_select_postop = false;
    dim_t num_head_kv = 0;
    size_t amx_wsp_bytes = 0;
    size_t scratch_per_thread = 0;
    size_t kt_global_bytes = 0, vt_global_bytes = 0;
    int nthr = 0;

    // AMX cfg per kernel, indexed [is_q_tail][is_kv_tail].
    sdpa_amx_cfg_t mm1_amx[2][2], mm2_amx[2][2];

    size_t scratch_total(int n) const {
        return scratch_per_thread * static_cast<size_t>(n) + kt_global_bytes
                + vt_global_bytes;
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

    // Reused vectorized jit softmax (max/exp/normalize over seq_kv, per row)
    // plus the pd it reads its config from; when use_jit_softmax is false the
    // execute path runs a scalar two-pass softmax instead.
    std::shared_ptr<primitive_desc_t> softmax_pd;
    std::shared_ptr<softmax_impl::jit_softmax_kernel_base_t> softmax_kernel;
    bool use_jit_softmax = false;

    // Final select-in-mm1 decision after kernel compilation (configure's
    // decision, possibly downgraded if the compiled ukernel rejects the
    // post-op); execute reads this, not the conf.
    bool mm1_select_postop = false;

    // Standalone JIT select-mask pre-pass kernel (null unless a select mask is
    // present, not folded into mm1, jit softmax is used, and the condition is
    // dense along seq_kv); execute falls back to a scalar pre-pass when null.
    std::shared_ptr<sdpa_full_softmax_select_ir::select_ir_kernel_t>
            select_kernel;
};

// Query-axis blocked / two-pass-softmax SDPA compute, folded into the CPU
// primitive: the primitive_desc owns the conf + descs (configure), the
// primitive owns the kernels (create_kernels) and drives the loop (execute).
namespace sdpa_full_softmax {

// Compute the derived configuration (query/KV blocking, per-thread + global
// scratch layout, AMX palette/wsp) and build the finalized BRGEMM descriptors
// into `descs`. JIT-free, so a primitive_desc can size its scratchpad from the
// returned conf without compiling kernels.
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
