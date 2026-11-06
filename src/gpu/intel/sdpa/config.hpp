/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GPU_INTEL_SDPA_CONFIG_HPP
#define GPU_INTEL_SDPA_CONFIG_HPP

#include "common/sdpa_pd.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

using fwd_pd_t = sdpa_fwd_pd_t;
using bwd_pd_t = sdpa_bwd_pd_t;

// Microkernel tile configuration of the fused forward kernel.
struct fwd_config_t {
    int unroll_m_kq, unroll_n_kq; // Subgroup tile sizes for K*Q GEMM
    int unroll_m_vs, unroll_n_vs; // Subgroup tile sizes for V*S GEMM
    int wg_m_kq, wg_n_kq; // Workgroup configuration for K*Q GEMM
    int wg_m_vs, wg_n_vs; // Workgroup configuration for V*S GEMM
};

// Microkernel tile configuration of the fused backward kernel.
struct bwd_config_t {
    int unroll_m_BcBr, unroll_n_BcBr; // Subgroup tile sizes for Br*Bc GEMMs
    int unroll_m_DBc, unroll_n_DBc; // Subgroup tile sizes for Bc*D GEMMs
    int unroll_m_DBr, unroll_n_DBr; // Subgroup tile sizes for Br*D GEMMs
    int wg_m_BcBr, wg_n_BcBr; // Workgroup configuration for Br*Bc GEMMs
    int wg_m_DBc, wg_n_DBc; // Workgroup configuration for Bc*D GEMMs
    int wg_m_DBr, wg_n_DBr; // Workgroup configuration for Br*D GEMMs
};

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
