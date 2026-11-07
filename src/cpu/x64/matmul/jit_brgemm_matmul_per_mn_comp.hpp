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

#ifndef CPU_X64_MATMUL_JIT_BRGEMM_MATMUL_PER_MN_COMP_HPP
#define CPU_X64_MATMUL_JIT_BRGEMM_MATMUL_PER_MN_COMP_HPP

#include <memory>

#include "common/c_types_map.hpp"
#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

// JIT fill of the per-(M, N) f32 compensation tile consumed by the brgemm
// post-ops epilogue. For one (m_blk, n_blk) tile over one brgemm K-block
// of length k_len:
//
//   delta[m, n] = z_src * S[n] + z_wei * T[m] - k_len * z_src * z_wei
//
// with T[m] = sum_{k=0}^{k_len-1} src[m,k] and
//      S[n] = sum_{k=0}^{k_len-1} wei[k,n].
struct per_mn_comp_kernel_t {
    struct ctx_t {
        const void *src_batch_ptr = nullptr;
        const void *wei_batch_ptr = nullptr;

        // Batch-adjusted zero-point pointers (or null).
        const void *src_zp_ptr = nullptr;
        // Base index in elements into src_zp_ptr for this K-group / M-slice.
        int src_zp_nibble_idx = 0;
        const void *wei_zp_ptr = nullptr;

        // f32 delta tile output (size M_blk * rnd_up(N_blk, LDC)).
        float *delta_ptr = nullptr;

        // Tile coordinates.
        dim_t m_base = 0;
        dim_t n_base = 0;
        dim_t M_blk = 0;
        dim_t N_blk = 0;

        // Blocked weights only: `wei_batch_ptr` points at (K-block start,
        // n_base), which is column `wei_n_blk_off` of its weights N block.
        dim_t wei_n_blk_off = 0;

        // Length of the K-block
        dim_t k_len = 0;

        // Internal sub-call state consumed directly by the JIT.
        dim_t stripe_w = 0;
        float src_zp_f32 = 0.0f;
        float wei_zp_f32 = 0.0f;
        float g_sz_f32 = 0.0f;
    };

    // Dispatch-time validation: returns `status::unimplemented` when
    // the JIT cannot be built for `bgmmc` (unsupported ISA / dtype / stride
    // / LDC combo). Call from `pd_t::init()` via `VDISPATCH_MATMUL`.
    static status_t is_applicable(const brgemm_matmul_conf_t *bgmmc);

    // Factory. Returns `status::unimplemented` when the JIT cannot be built
    // for `bgmmc` (unsupported ISA / dtype / stride / LDC combo). Internally
    // re-runs `is_applicable()` so failures stay graceful.
    static status_t create(std::unique_ptr<per_mn_comp_kernel_t> &kernel,
            const brgemm_matmul_conf_t *bgmmc);

    virtual void operator()(const ctx_t *ctx) const = 0;
    virtual ~per_mn_comp_kernel_t() = default;

protected:
    per_mn_comp_kernel_t(const brgemm_matmul_conf_t *conf, bool wei_zp_per_n)
        : conf_(conf), wei_zp_per_n_(wei_zp_per_n) {}

    const brgemm_matmul_conf_t *conf_;
    const bool wei_zp_per_n_;
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
