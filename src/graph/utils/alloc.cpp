/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "graph/utils/alloc.hpp"

#ifdef DNNL_WITH_SYCL
#include "oneapi/dnnl/dnnl_sycl.hpp"
#endif

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
#include "graph/utils/ocl_usm_utils.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace utils {

#if DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL
void *ocl_allocator_t::malloc(
        size_t size, size_t alignment, cl_device_id dev, cl_context ctx) {
    return ocl::malloc_device(dev, ctx, size, alignment);
}

void ocl_allocator_t::free(
        void *ptr, cl_device_id dev, cl_context ctx, cl_event event) {
    if (nullptr == ptr) return;
    if (event) { OCL_CHECK_V(xpu::ocl::clWaitForEvents(1, &event)); }
    ocl::free(ptr, dev, ctx);
}
#endif

#ifdef DNNL_WITH_SYCL
void *sycl_allocator_t::malloc(
        size_t size, size_t alignment, const void *dev, const void *ctx) {
    const size_t align = alignment == 0 ? DEFAULT_ALIGNMENT : alignment;
    return ::sycl::aligned_alloc_device(align, size,
            *static_cast<const ::sycl::device *>(dev),
            *static_cast<const ::sycl::context *>(ctx));
}

void sycl_allocator_t::free(
        void *ptr, const void *dev, const void *ctx, void *event) {
    UNUSED(dev);
    if (event) {
        auto sycl_deps_ptr = static_cast<::sycl::event *>(event);
        sycl_deps_ptr->wait();
    }
    ::sycl::free(ptr, *static_cast<const ::sycl::context *>(ctx));
}
#endif

void *cpu_allocator_t::malloc(size_t size, size_t alignment) {
    void *ptr = nullptr;
    const size_t align = alignment == 0 ? DEFAULT_ALIGNMENT : alignment;
#ifdef _WIN32
    ptr = _aligned_malloc(size, align);
    int rc = ((ptr) ? 0 : errno);
#else
    int rc = ::posix_memalign(&ptr, align, size);
#endif /* _WIN32 */
    return (rc == 0) ? ptr : nullptr;
}

void cpu_allocator_t::free(void *p) {
#ifdef _WIN32
    _aligned_free((void *)p);
#else
    ::free((void *)p);
#endif /* _WIN32 */
}

} // namespace utils
} // namespace graph
} // namespace impl
} // namespace dnnl
