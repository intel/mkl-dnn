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

#include "graph/backend/dnnl/kernels/kernel_base.hpp"
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

// Fused CPU SDPA kernel, built on the internal x64 BRGEMM microkernel plus an
// online-softmax (flash-attention-style) epilogue, so the full S x S score
// matrix is never materialized. This is the CPU counterpart to the GPU-only
// fused sdp_primitive_kernel_t (which fuses via the sdpa primitive).
//
// Scope of the first iteration:
//   * fp32 only (non-quantized);
//   * the GQA attention pattern: QK^T -> scale -> select-mask -> softmax -> PV.
//
// It is selectable for A/B testing via ONEDNN_GRAPH_SDPA_IMPL=fused_brgemm (see
// sdp_base_t in sdp.hpp). The compile path reuses sdp_decomp_config only for
// pattern validation and dim/stride/flag extraction; the execute path streams
// the KV sequence in tiles with an online-softmax epilogue.
struct sdp_fused_brgemm_kernel_t : public kernel_base_t {
private:
    std::shared_ptr<subgraph_t> subgraph_;
    memory_planner_t memory_planner_;

    // Reused only to validate the pattern and to extract SDP dims, strides and
    // feature flags (scale/mask/select). The fused kernel does NOT build the
    // decomposed sub-primitives.
    sdp_decomp_config_t sdp_cfg_;

    // Parsed problem geometry (fp32 GQA), captured at compile time.
    int ndims_ = 0;
    dim_t batch_ = 0, num_head_kv_ = 0, num_head_q_ = 0, group_head_ = 1,
          seq_q_ = 0, seq_kv_ = 0, hs_qk_ = 0, hs_v_ = 0;
    // User strides of Q / K / V / output / select-condition, in elements.
    std::vector<dim_t> q_strides_, k_strides_, v_strides_, o_strides_,
            cond_strides_;
    // Indices into the external inputs vector (from sdp_cfg_.graph_inport).
    int idx_q_ = -1, idx_k_ = -1, idx_v_ = -1, idx_scale_ = -1, idx_cond_ = -1,
        idx_fill_ = -1;
    bool has_scale_ = false, scale_is_divide_ = false, has_select_ = false,
         select_fusiable_ = false;
    // KV tiling for the online (flash-style) softmax: seq_kv is processed in
    // tiles of kv_blk_ (last tile is kv_tail_ wide when seq_kv is not a
    // multiple).
    dim_t kv_blk_ = 0, kv_tail_ = 0;
    // Internal x64 BRGEMM kernels created in compile_impl. mm1 computes a
    // scores tile Q*K[:, tile]; mm2 computes the P_tile*V[tile, :] partial
    // (beta=0) that the epilogue rescales into the running output. The *_tail_
    // variants handle the ragged last KV tile. Null on non-x64 builds.
    brgemm_kernel_t *mm1_kernel_
            = nullptr; // scores[seq_q, kv_blk] = Q * K_tile
    brgemm_kernel_t *mm2_kernel_
            = nullptr; // pv[seq_q, head_size_v] = P * V_tile
    brgemm_kernel_t *mm1_tail_kernel_ = nullptr;
    brgemm_kernel_t *mm2_tail_kernel_ = nullptr;

    // JIT online-softmax epilogue kernels built from the x64 CPU IR (AVX2). The
    // softmax kernels apply scale + select-mask + streaming-softmax to one KV
    // tile of scores (full/tail width); acc_renorm rescales the running output
    // by old_coef and adds the tile's P*V. When use_ir_epilogue_ is false (no
    // AVX2), execute_impl runs the scalar epilogue instead. x64-only: the
    // kernel type is incomplete elsewhere, so the members are compiled out.
#if DNNL_X64
    std::unique_ptr<cpu::x64::sdp_softmax_ir::softmax_ir_kernel_t>
            softmax_ir_kernel_;
    std::unique_ptr<cpu::x64::sdp_softmax_ir::softmax_ir_kernel_t>
            softmax_tail_ir_kernel_;
    std::unique_ptr<cpu::x64::sdp_softmax_ir::softmax_ir_kernel_t>
            acc_renorm_ir_kernel_;
#endif
    bool use_ir_epilogue_ = false;

    // Per-thread scratchpad for the execute path's online-softmax working
    // buffers: one block per thread, sized in compile_impl. Replaces the
    // per-iteration std::vectors that would otherwise malloc in the hot loop.
    registry_t sdp_registry_;
    int nthr_ = 0;

public:
    sdp_fused_brgemm_kernel_t();
    ~sdp_fused_brgemm_kernel_t() override;

    status_t compile_impl(const dnnl_partition_impl_t *part, engine_t *eng,
            const std::vector<logical_tensor_t> &inputs,
            const std::vector<logical_tensor_t> &outputs) override;

    status_t execute_impl(stream_t *strm, const std::vector<tensor_t> &inputs,
            const std::vector<tensor_t> &outputs,
            const tensor_t *scratchpad_buf) override;

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

    DEF_KERNEL_METHOD_STR(sdp_fused_brgemm_kernel_t)
    size_t get_scratchpad_size() const override {
        return sdp_registry_.size() * nthr_;
    }
    DNNL_DISALLOW_COPY_AND_ASSIGN(sdp_fused_brgemm_kernel_t)
};

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif
