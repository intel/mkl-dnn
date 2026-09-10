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

#ifndef CPU_X64_SDPA_SDP_FUSED_DRIVER_HPP
#define CPU_X64_SDPA_SDP_FUSED_DRIVER_HPP

#include <memory>
#include <vector>

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

struct brgemm_kernel_t;
namespace sdp_softmax_ir {
class softmax_ir_kernel_t;
} // namespace sdp_softmax_ir

// -----------------------------------------------------------------------------
// Decoupled online/flash-softmax SDPA driver (x64).
//
// Self-contained compute routine for the fused SDPA, using a streaming
// (online/flash-attention-style) softmax over KV tiles so the full
// [seq_q x seq_kv] score matrix is never materialized:
//   for each KV tile: scores = Q * K_tile^T ; rescale the running softmax
//   state (max/denominator) ; out = old_coef * out + P_tile * V_tile
// Like sdp_blocked_driver_t, this takes ONLY plain arguments (dims, strides in
// elements, raw pointers, scalar scale/fill) -- no dependency on the graph IR
// or the primitive framework -- so it can be driven either by the graph
// backend kernel or by a CPU dnnl_sdpa primitive.
//
// Contrast with sdp_blocked_driver_t (query-axis blocked, plain two-pass
// softmax): this driver never re-reads a full score row, trading that for a
// per-tile rescale of the running accumulator. f32 compute only today (no
// bf16/f16/AMX path, no attention-mask post-op); sdp_blocked_driver_t covers
// those.
// -----------------------------------------------------------------------------
struct sdp_fused_params_t {
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
    // themselves (this driver does not, unlike sdp_blocked_driver_t).
    std::vector<dim_t> q_strides, k_strides, v_strides, o_strides, cond_strides;
    // Logical dims of the select-condition tensor; a dim of 1 is a broadcast
    // axis whose (meaningless) stride must contribute 0.
    std::vector<dim_t> cond_dims;

    bool has_select = false;
    // Select semantics: fusiable (p2) keeps scores where cond != 0 and writes
    // fill elsewhere; non-fusiable (p1) is the inverse.
    bool select_fusiable = false;
};

// Runtime pointers / scalars, resolved per execute call. No generic post-op
// chain (unlike sdp_blocked_run_args_t): the scale is applied unconditionally
// (1.0f is a no-op) and there is no attention-mask support yet.
struct sdp_fused_run_args_t {
    const void *q = nullptr;
    const void *k = nullptr;
    const void *v = nullptr;
    const void *cond = nullptr; // uint8 select condition, or null
    void *out = nullptr;
    float scale = 1.0f;
    float fill = 0.0f;
};

// Owns the BRGEMM + IR-softmax kernels for the (full / KV-tail) tiles and
// drives the online-softmax execute loop. x64-only; on other builds init()
// returns unimplemented.
class sdp_fused_driver_t {
public:
    // Declared out-of-line (defined in sdp_fused_driver.cpp): the unique_ptr
    // members below hold a forward-declared IR kernel type, so their
    // construction/destruction must be instantiated where that type is
    // complete, not inline here.
    sdp_fused_driver_t();
    ~sdp_fused_driver_t();
    sdp_fused_driver_t(const sdp_fused_driver_t &) = delete;
    sdp_fused_driver_t &operator=(const sdp_fused_driver_t &) = delete;

    // Create the BRGEMM + IR-softmax kernels and compute the per-thread
    // scratch size. Must be called once before execute(). The engine
    // parameter is unused today (kept for interface parity with
    // sdp_blocked_driver_t / future engine-dependent kernel selection).
    // Equivalent to configure() followed by create_kernels().
    status_t init(const sdp_fused_params_t &params, engine_t *engine);

    // Two-phase split of init() for the CPU sdpa primitive: configure() does
    // only the JIT-free arithmetic (KV tiling, per-thread scratch size) so a
    // primitive_desc can size its scratchpad without compiling kernels;
    // create_kernels() then JIT-compiles the BRGEMM + IR-softmax kernels. Call
    // configure() first, create_kernels() before execute().
    status_t configure(const sdp_fused_params_t &params);
    status_t create_kernels(engine_t *engine);

    int nthr() const { return nthr_; }
    size_t scratch_total(int nthr) const {
        return scratch_per_thread_ * static_cast<size_t>(nthr);
    }

    // Run the online-softmax SDPA. scratch_base points at a buffer of at
    // least scratch_total(nthr) bytes; each thread slices its own block.
    status_t execute(const sdp_fused_run_args_t &args, void *scratch_base,
            int nthr) const;

private:
    sdp_fused_params_t p_;
    // KV tiling width for the streaming softmax: seq_kv is processed in tiles
    // of kv_blk_.
    dim_t kv_blk_ = 0;
    // Internal x64 BRGEMM kernels: mm1 computes a scores tile Q*K[:, tile];
    // mm2 computes the P_tile*V[tile, :] partial (beta=0) that the epilogue
    // rescales into the running output. The *_tail_ variants handle the ragged
    // last KV tile.
    brgemm_kernel_t *mm1_kernel_ = nullptr, *mm2_kernel_ = nullptr,
                    *mm1_tail_kernel_ = nullptr, *mm2_tail_kernel_ = nullptr;
    // JIT online-softmax epilogue built from the x64 CPU IR (AVX2): the softmax
    // kernels apply scale + select-mask + streaming-softmax to one KV tile
    // (full/tail width); acc_renorm rescales the running output by old_coef and
    // adds the tile's P*V. When use_ir_epilogue_ is false (no AVX2), execute
    // runs a scalar epilogue instead.
    std::unique_ptr<sdp_softmax_ir::softmax_ir_kernel_t> softmax_ir_kernel_,
            softmax_tail_ir_kernel_, acc_renorm_ir_kernel_;
    bool use_ir_epilogue_ = false;
    // Per-thread scratch: scores + acc + pv + row_max + row_denom + old_coef,
    // each 64-byte aligned (see sdp_fused_driver.cpp).
    size_t scratch_per_thread_ = 0;
    size_t off_scores_ = 0, off_acc_ = 0, off_pv_ = 0, off_row_max_ = 0,
           off_row_denom_ = 0, off_old_coef_ = 0;
    int nthr_ = 0;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
