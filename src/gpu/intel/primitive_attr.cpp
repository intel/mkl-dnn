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

#include "gpu/intel/primitive_attr.hpp"

#include "common/c_types_map.hpp"
#include "common/primitive_desc_iface.hpp"
#include "common/utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::gpu::intel;

#ifdef DNNL_DEV_MODE
extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_set_kernel_override(
        primitive_attr_t *attr, const char *kernel) {
    if (utils::any_null(attr)) return status::invalid_arguments;

    if (!attr->gpu_attr_) CHECK(attr->set_gpu_attr(gpu_primitive_attr_t()));
    utils::downcast<gpu_primitive_attr_t *>(attr->gpu_attr_.get())
            ->set_kernel_override(kernel ? kernel : "");
    return status::success;
}

// Number of jit:gemm selection candidates, i.e. valid rank overrides; -1 if
// the selected implementation is not jit:gemm.
extern "C" dnnl_status_t DNNL_API dnnl_impl_gpu_intel_get_kernel_count(
        const_dnnl_primitive_desc_t pd, int *count) {
    if (utils::any_null(pd, count)) return status::invalid_arguments;

    if (pd->query(query::gpu_kernel_count, 0, count) != status::success)
        *count = -1;
    return status::success;
}
#endif // DNNL_DEV_MODE
