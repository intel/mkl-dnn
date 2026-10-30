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

#include "gpu/intel/gemm/config.hpp"
#include "gpu/intel/primitive_attr.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

int append_post_ops_to_arg_list(const exec_args_t &args,
        compute::kernel_arg_list_t &arg_list, int post_op_idx,
        const post_ops_t &post_ops, memory_desc_wrapper dst_mdw) {
    return intel::append_post_ops_to_arg_list_base(
            args, arg_list, post_op_idx, post_ops, dst_mdw);
}

std::string kernel_override(const primitive_attr_t *attr) {
    auto *gpu_attr = utils::downcast<const gpu_primitive_attr_t *>(
            attr->gpu_attr_.get());
    return gpu_utils::dev_getenv("GEMM_KERNEL",
            gpu_attr ? gpu_attr->kernel_override() : std::string());
}

int kernel_rank(const std::string &s) {
    // <=9 digits always fit int, so std::stoi cannot overflow.
    bool is_rank = !s.empty() && s.size() <= 9
            && s.find_first_not_of("0123456789") == std::string::npos;
    return is_rank ? std::stoi(s) : -1;
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
