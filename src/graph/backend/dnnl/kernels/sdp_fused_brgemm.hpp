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

#ifndef GRAPH_BACKEND_DNNL_KERNELS_SDP_FUSED_BRGEMM_HPP
#define GRAPH_BACKEND_DNNL_KERNELS_SDP_FUSED_BRGEMM_HPP

#include <memory>
#include <string>
#include <vector>

#include "graph/backend/dnnl/platform.hpp"

#include "graph/backend/dnnl/kernels/kernel_base.hpp"
#include "graph/backend/dnnl/kernels/sdp_blocked_driver.hpp"
#include "graph/backend/dnnl/kernels/sdp_decomp_config.hpp"

#include "graph/backend/dnnl/dnnl_partition_impl.hpp"
#include "graph/backend/dnnl/subgraph.hpp"

#include "graph/backend/dnnl/passes/memory_planning.hpp"

namespace dnnl {
namespace impl {

// Forward declarations to avoid including the BRGEMM / IR headers.
namespace cpu {
namespace x64 {
struct brgemm_kernel_t;
namespace sdp_softmax_ir {
class softmax_ir_kernel_t;
} // namespace sdp_softmax_ir
} // namespace x64
} // namespace cpu

namespace graph {
namespace dnnl_impl {

using brgemm_kernel_t = dnnl::impl::cpu::x64::brgemm_kernel_t;

// Parsed SDP problem (geometry, strides, flags, input indices), filled once by
// sdp_fused_brgemm_base_t::parse() and read by both concrete kernels.
struct sdp_problem_t {
    // MHA/MQA/GQA are all expressed via num_head_q vs num_head_kv.
    int ndims = 0;
    dim_t batch = 0, num_head_q = 0, num_head_kv = 0, group_head = 1;
    dim_t seq_q = 0, seq_kv = 0, head_size_qk = 0, head_size_v = 0;
    // User strides (elements) of Q / K / V / output / select-condition.
    std::vector<dim_t> q_strides, k_strides, v_strides, o_strides, cond_strides;
    // Select-condition logical dims; a dim of 1 is a broadcast axis whose
    // (meaningless) stride must contribute 0.
    std::vector<dim_t> cond_dims;
    // Indices into the external inputs vector (from sdp_cfg_.graph_inport).
    int idx_q = -1, idx_k = -1, idx_v = -1, idx_scale = -1, idx_cond = -1,
        idx_fill = -1, idx_mask = -1;
    bool has_scale = false, scale_is_divide = false, has_select = false,
         select_fusiable = false, has_mask = false;
    // mm1 (QK^T) transpose_b (K stored [.., seq_kv, hs] and transposed in the
    // BRGEMM when set), and the SoftMax "inf_as_zero" mode (fully-masked row ->
    // all-zero probabilities instead of NaN).
    bool mm1_transpose_b = false, softmax_inf_as_zero = false;
};

// Shared front-end for the two fused BRGEMM SDPA kernels, the CPU counterpart
// to the GPU-only fused sdp_primitive_kernel_t. Owns the subgraph / pattern
// plumbing and parse(), which validates + lowers the partition and fills prb_.
//
// Scope of the first iteration:
//   * non-quantized (fp32; bf16/f16 in the blocked kernel);
//   * the attention pattern: QK^T -> scale -> select-mask -> softmax -> PV.
//
// It is abstract: the two concrete kernels below add only their own compute
// state (online-softmax BRGEMM + IR kernels, or the blocked driver) and
// implement compile/execute. Selectable for A/B testing via
// ONEDNN_GRAPH_SDPA_IMPL={fused_brgemm|fused_brgemm_blocked} (see sdp_base_t).
struct sdp_fused_brgemm_base_t : public kernel_base_t {
protected:
    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;
    // Reused only to validate the pattern and extract dims/strides/flags; the
    // fused kernels do NOT build the decomposed sub-primitives.
    sdp_decomp_config_t sdp_cfg_;
    sdp_problem_t prb_;
    int nthr_ = 0;

    // Build the subgraph, validate + lower the SDP pattern, and fill prb_.
    status_t parse(const dnnl_partition_impl_t *part, engine_t *eng,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs);

public:
    ~sdp_fused_brgemm_base_t() override = default;

#ifdef DNNL_WITH_SYCL
    status_t sycl_execute_impl(stream_t *strm,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf,
            const std::vector<::sycl::event> &sycl_deps,
            ::sycl::event *sycl_event) override {
        UNUSED(strm);
        UNUSED(inputs);
        UNUSED(outputs);
        UNUSED(scratchpad_buf);
        UNUSED(sycl_deps);
        UNUSED(sycl_event);
        return status::unimplemented;
    }
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
    status_t ocl_execute_impl(stream_t *strm,
            const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf,
            const std::vector<ocl_event_t> &cl_deps,
            ocl_event_t &ret_event) override {
        UNUSED(strm);
        UNUSED(inputs);
        UNUSED(outputs);
        UNUSED(scratchpad_buf);
        UNUSED(cl_deps);
        UNUSED(ret_event);
        return status::unimplemented;
    }
#endif
};

// Online-softmax (flash-attention-style) fused SDPA kernel: streams the KV
// sequence in tiles with an online-softmax epilogue, so the full S x S score
// matrix is never materialized. Selected by ONEDNN_GRAPH_SDPA_IMPL=fused_brgemm.
struct sdp_fused_brgemm_online_kernel_t : public sdp_fused_brgemm_base_t {
private:
#if DNNL_X64
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
    std::unique_ptr<cpu::x64::sdp_softmax_ir::softmax_ir_kernel_t>
            softmax_ir_kernel_, softmax_tail_ir_kernel_, acc_renorm_ir_kernel_;
    bool use_ir_epilogue_ = false;
    // Per-thread scratchpad for the online-softmax working buffers (one block
    // per thread, sized in compile_impl).
    registry_t sdp_registry_;
#endif

public:
    // Out-of-line ctor/dtor: unique_ptr members to a forward-declared IR type.
    sdp_fused_brgemm_online_kernel_t();
    ~sdp_fused_brgemm_online_kernel_t() override;

    status_t compile_impl(const dnnl_partition_impl_t *part, engine_t *eng,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    status_t execute_impl(stream_t *strm, const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf) override;

    DEF_KERNEL_METHOD_STR(sdp_fused_brgemm_online_kernel_t)
    size_t get_scratchpad_size() const override {
#if DNNL_X64
        return sdp_registry_.size() * nthr_;
#else
        return 0;
#endif
    }
    DNNL_DISALLOW_COPY_AND_ASSIGN(sdp_fused_brgemm_online_kernel_t)
};

// Query-axis blocked / two-pass-softmax fused SDPA kernel: delegates to the
// decoupled blocked driver (which owns its own BRGEMM kernels and scratch).
// Selected by ONEDNN_GRAPH_SDPA_IMPL=fused_brgemm_blocked.
struct sdp_fused_brgemm_blocked_kernel_t : public sdp_fused_brgemm_base_t {
private:
#if DNNL_X64
    sdp_blocked_driver_t blocked_driver_;
    size_t blocked_scratch_total_ = 0;
#endif

public:
    sdp_fused_brgemm_blocked_kernel_t() = default;

    status_t compile_impl(const dnnl_partition_impl_t *part, engine_t *eng,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    status_t execute_impl(stream_t *strm, const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf) override;

    DEF_KERNEL_METHOD_STR(sdp_fused_brgemm_blocked_kernel_t)
    size_t get_scratchpad_size() const override {
#if DNNL_X64
        return blocked_scratch_total_;
#else
        return 0;
#endif
    }
    DNNL_DISALLOW_COPY_AND_ASSIGN(sdp_fused_brgemm_blocked_kernel_t)
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
