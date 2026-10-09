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

#ifndef CPU_X64_SDPA_SDPA_ONLINE_SOFTMAX_HPP
#define CPU_X64_SDPA_SDPA_ONLINE_SOFTMAX_HPP

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct brgemm_kernel_t;
namespace sdpa_softmax_ir {
class softmax_ir_kernel_t;
} // namespace sdpa_softmax_ir

// -----------------------------------------------------------------------------
// Online/flash-softmax SDPA compute (x64), folded into the CPU primitive.
//
// Self-contained compute routine for the online-softmax SDPA, using a streaming
// (online/flash-attention-style) softmax over KV tiles so the full
// [seq_q x seq_kv] score matrix is never materialized:
//   for each KV tile: scores = Q * K_tile^T ; rescale the running softmax
//   state (max/denominator) ; out = old_coef * out + P_tile * V_tile
// Takes ONLY plain arguments (dims, strides in elements, raw pointers, scalar
// scale/fill) over the pd's conf + the primitive's kernels.
//
// Unlike the full-key-axis path, which computes each query tile's scores for
// all seq_kv keys before softmax, this never materializes or re-reads a full
// score row. It instead rescales the running accumulator per KV tile. f32
// compute only today (no bf16/f16/AMX path, no attention-mask post-op); the
// full-key-axis path supports those data types and the additive attention mask.
// -----------------------------------------------------------------------------
struct sdpa_online_softmax_params_t {
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
    // axis; K / V have group extent 1 (broadcast over the group). K is always
    // read as [.., head_size_qk, seq_kv] (the non-transposed orientation);
    // callers that may see the other physical layout must transpose it
    // themselves (the online-softmax strategy does not, unlike the full-softmax one).
    std::vector<dim_t> q_strides, k_strides, v_strides, o_strides, cond_strides;
    // Logical dims of the select-condition tensor; a dim of 1 is a broadcast
    // axis whose (meaningless) stride must contribute 0.
    std::vector<dim_t> cond_dims;

    bool has_select = false;
    // When true, keep scores where cond == 0; otherwise keep where cond != 0.
    bool invert_select = false;
};

// Runtime pointers / scalars, resolved per execute call. No generic post-op
// chain (unlike sdpa_full_softmax_run_args_t): the scale is applied unconditionally
// (1.0f is a no-op) and there is no attention-mask support yet.
struct sdpa_online_softmax_run_args_t {
    const void *q = nullptr;
    const void *k = nullptr;
    const void *v = nullptr;
    const void *cond = nullptr; // uint8 select condition, or null
    void *out = nullptr;
    float scale = 1.0f;
    float fill = 0.0f;
};

// Derived compute configuration (KV tiling + per-thread scratch layout). Pure
// scalar config (mirroring brgemm_matmul_conf_t), computed once by configure()
// and owned by the primitive_desc so the pd sizes its scratchpad from it.
struct sdpa_online_softmax_conf_t {
    sdpa_online_softmax_params_t params;
    // KV tiling width for the streaming softmax (seq_kv in tiles of kv_blk).
    dim_t kv_blk = 0;
    // Query-axis blocking: the seq_q rows are processed in blocks of q_blk so
    // the per-block Q slice, scores tile and running accumulator stay
    // L2-resident across the KV sweep (avoids re-streaming all of Q per KV
    // tile). q_tail is the ragged last block (seq_q % q_blk, 0 if it divides).
    dim_t q_blk = 0;
    dim_t q_tail = 0;
    // Per-thread scratch: scores + acc + pv + row_max + row_denom + old_coef,
    // each 64-byte aligned; offsets into the per-thread block. Sized for one
    // query block (q_blk rows), not the full seq_q.
    size_t off_scores = 0, off_acc = 0, off_pv = 0, off_row_max = 0,
           off_row_denom = 0, off_old_coef = 0;
    size_t scratch_per_thread = 0;
    int nthr = 0;
    size_t scratch_total(int n) const {
        return scratch_per_thread * static_cast<size_t>(n);
    }
};

// Compiled kernels for the online-softmax SDPA, owned by the primitive. The BRGEMM
// kernels are raw handles freed in the destructor; the IR-softmax epilogue
// kernels are held by unique_ptr. When use_ir_epilogue is false (no AVX2), the
// execute path runs a scalar epilogue instead.
struct sdpa_online_softmax_kernels_t {
    // Declared out-of-line (defined in sdpa_online_softmax.cpp): the unique_ptr members
    // hold a forward-declared IR kernel type, so construction/destruction must
    // be instantiated where that type is complete.
    sdpa_online_softmax_kernels_t();
    ~sdpa_online_softmax_kernels_t();
    sdpa_online_softmax_kernels_t(const sdpa_online_softmax_kernels_t &)
            = delete;
    sdpa_online_softmax_kernels_t &operator=(
            const sdpa_online_softmax_kernels_t &)
            = delete;

    // mm1 computes a scores tile Q*K[:, tile]; mm2 computes the P_tile*V[tile,:]
    // partial (beta=0) that the epilogue rescales into the running output.
    // Indexed [q_tail?][kv_tail?]: a brgemm baked for the full q_blk rows vs the
    // ragged last query block, and the full kv_blk vs the ragged last KV tile.
    brgemm_kernel_t *mm1_kernel[2][2] = {};
    brgemm_kernel_t *mm2_kernel[2][2] = {};
    // JIT online-softmax epilogue (AVX2 IR): the softmax kernels apply scale +
    // select-mask + streaming-softmax to one KV tile, indexed [q_tail?][kv_tail?]
    // (row count baked from q_blk/q_tail, width from kv_blk/kv_tail); acc_renorm
    // rescales the running output by old_coef and adds the tile's P*V and
    // depends only on the query-block row count ([q_tail?]).
    std::unique_ptr<sdpa_softmax_ir::softmax_ir_kernel_t> softmax_ir_kernel[2]
                                                                           [2];
    std::unique_ptr<sdpa_softmax_ir::softmax_ir_kernel_t>
            acc_renorm_ir_kernel[2];
    bool use_ir_epilogue = false;
};

// Online-softmax (flash) SDPA compute, folded into the CPU primitive:
// the primitive_desc owns the conf (configure), the primitive owns the kernels
// (create_kernels) and drives the loop (execute).
namespace sdpa_online_softmax {

// Compute the derived configuration (KV tiling + per-thread scratch layout)
// from the plain params. JIT-free, so a primitive_desc can size its scratchpad
// from the returned conf without compiling kernels.
status_t configure(const sdpa_online_softmax_params_t &params,
        sdpa_online_softmax_conf_t &conf);

// JIT-compile the BRGEMM + IR-softmax kernels from a conf produced by
// configure(), into `kernels`. The engine is unused today (kept for interface
// parity with the full-softmax path / future engine-dependent kernel selection).
status_t create_kernels(const sdpa_online_softmax_conf_t &conf,
        engine_t *engine, sdpa_online_softmax_kernels_t &kernels);

// Run the online-softmax SDPA. scratch_base points at a buffer of at least
// conf.scratch_total(nthr) bytes; each thread slices its own block.
status_t execute(const sdpa_online_softmax_conf_t &conf,
        const sdpa_online_softmax_kernels_t &kernels,
        const sdpa_online_softmax_run_args_t &args, void *scratch_base,
        int nthr);

} // namespace sdpa_online_softmax

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
