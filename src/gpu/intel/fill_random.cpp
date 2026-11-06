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

#include <mutex>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "gpu/intel/compute/kernel.hpp"
#include "gpu/intel/compute/kernel_ctx.hpp"
#include "gpu/intel/engine.hpp"
#include "gpu/intel/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {

static status_t get_cached_kernel(
        intel::engine_t *engine, compute::kernel_t &kernel) {
    static std::unordered_map<engine_id_t, compute::kernel_t> cache;
    static std::mutex mutex;

    std::lock_guard<std::mutex> lock(mutex);
    auto it = cache.find(engine->engine_id());
    if (it != cache.end()) {
        kernel = it->second;
        return status::success;
    }

    compute::kernel_ctx_t ctx;
    std::vector<compute::kernel_t> kernels;
    CHECK(engine->create_kernels(&kernels, {"fill_random"}, ctx));
    kernel = cache.emplace(engine->engine_id(), kernels[0]).first->second;
    return status::success;
}

static uint32_t nan_safe_mask(data_type_t dt) {
    switch (dt) {
        case data_type::f32: return 0xFF7FFFFFu;
        case data_type::f16: return 0xFBFFFBFFu;
        case data_type::bf16: return 0xFF7FFF7Fu;
        case data_type::f8_e5m2: return 0xFBFBFBFBu;
        case data_type::f8_e4m3: return 0xF7F7F7F7u;
        case data_type::e8m0: return 0xFEFEFEFEu;
        case data_type::f64: return 0xFFEFFFFFu;
        default: return 0xFFFFFFFFu;
    }
}

status_t fill_random(impl::stream_t *stream, size_t size,
        impl::memory_t *memory, int buffer_index, uint32_t seed, uint32_t mask,
        uint64_t value) {
    if (size == 0) return status::success;

    auto *intel_stream = utils::downcast<intel::stream_t *>(stream);
    auto *intel_engine = utils::downcast<intel::engine_t *>(stream->engine());
    compute::kernel_t kernel;
    CHECK(get_cached_kernel(intel_engine, kernel));

    // Each work-item writes 16 bytes (4 × uint32) via vstore4.
    static constexpr size_t bytes_per_item = 16;
    compute::nd_range_t nd_range(
            compute::range_t(utils::div_up(size, bytes_per_item), 1, 1));
    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, *memory->memory_storage(buffer_index));
    arg_list.set(1, seed);
    arg_list.set(2, static_cast<uint64_t>(size));
    arg_list.set(3, nan_safe_mask(memory->md()->data_type) & mask);
    arg_list.set(4, value);

    CHECK(kernel.parallel_for(*stream, nd_range, arg_list,
            intel_stream->ctx().get_deps(), intel_stream->ctx().get_deps()));
    return status::success;
}

} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

extern "C" dnnl::impl::status_t DNNL_API dnnl_impl_gpu_fill_random(
        dnnl::impl::stream_t *stream, size_t size, dnnl::impl::memory_t *memory,
        int buffer_index, uint32_t seed, uint32_t mask, uint64_t value) {
    return dnnl::impl::gpu::intel::fill_random(
            stream, size, memory, buffer_index, seed, mask, value);
}
