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

#ifndef GPU_INTEL_GEMM_CONFIG_HPP
#define GPU_INTEL_GEMM_CONFIG_HPP

#include "common/primitive_exec_types.hpp"
#include "gpu/gpu_gemm_pd.hpp"
#include "gpu/intel/compute/kernel_arg_list.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

using pd_t = gpu_gemm_pd_t;
using desc_t = gemm_desc_t;

int append_post_ops_to_arg_list(const impl::exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops, memory_desc_wrapper dst_mdw);

// Candidate rank if the kernel override is a bare integer, else -1.
int kernel_rank(const primitive_attr_t *attr);

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
