/*******************************************************************************
* Copyright 2019 Intel Corporation
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

// The file is named sycl_stream.hpp rather than stream.hpp to avoid shadowing
// <sycl/stream.hpp> when MSVC is used as the SYCL host compiler. The file can
// be renamed back once https://github.com/intel/llvm/issues/23670 is fixed.

#ifndef GPU_INTEL_SYCL_SYCL_STREAM_HPP
#define GPU_INTEL_SYCL_SYCL_STREAM_HPP

#include <cstring>
#include <memory>
#include <CL/cl.h>

#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"
#include "common/primitive_iface.hpp"
#include "common/stream.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/context.hpp"
#include "xpu/sycl/memory_storage.hpp"
#include "xpu/sycl/stream_impl.hpp"
#include "xpu/sycl/stream_profiler.hpp"

#include "gpu/intel/engine.hpp"
#include "gpu/intel/stream.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sycl {

struct stream_t : public gpu::intel::stream_t {
    static status_t create_stream(impl::stream_t **stream,
            impl::engine_t *engine, impl::stream_impl_t *stream_impl) {
        std::unique_ptr<intel::sycl::stream_t> s(
                new intel::sycl::stream_t(engine, stream_impl));
        if (!s) return status::out_of_memory;

        status_t status = s->init();
        if (status != status::success) {
            // Stream owns stream_impl only if it's created successfully (including initialization).
            s->impl_.release();
            return status;
        }

        *stream = s.release();
        return status::success;
    }

    status_t wait() override {
        auto status = impl()->wait();
        // A verbose profiler polling call cannot be added here since sporadic
        // wait() calls outside the normal primitive execution flow may
        // interrupt in-flight primitives before all their events are complete.
        // This causes the profiler to incorrectly log incomplete execution
        // times and invalidate event storage for pending operations. Unlogged
        // primitives are instead captured at the next after_exec_hook()
        // polling cycle or during profiler destruction.
        return status;
    }

    void before_exec_hook() override;
    void after_exec_hook() override;

    status_t reset_profiling() override {
        if (!is_profiling_enabled()) return status::invalid_arguments;
        profiler_->reset();
        return status::success;
    }

    status_t get_profiling_data(profiling_data_kind_t data_kind,
            int *num_entries, uint64_t *data) const override {
        if (!is_profiling_enabled()) return status::invalid_arguments;
        return profiler_->get_info(data_kind, num_entries, data);
    }

    status_t run_verbose_profiler(const std::string &pd_info, double start_ms,
            uint64_t component) override;

    ::sycl::queue &queue() const { return *impl()->queue(); }

    status_t enqueue_primitive(const primitive_iface_t *prim_iface,
            exec_ctx_t &exec_ctx) override {
        return prim_iface->execute(exec_ctx);
    }

    status_t copy(const memory_storage_t &src, const memory_storage_t &dst,
            size_t size, const xpu::event_t &deps,
            xpu::event_t &out_dep) override {
        return impl()->copy(this, src, dst, size, deps, out_dep,
                profiler_.get(), verbose_profiler());
    }

    status_t fill(const memory_storage_t &dst, uint8_t pattern, size_t size,
            const xpu::event_t &deps, xpu::event_t &out_dep) override {
        return impl()->fill(dst, pattern, size, deps, out_dep, profiler_.get(),
                verbose_profiler());
    }

    status_t barrier() override { return impl()->barrier(); }

    xpu::sycl::context_t &sycl_ctx() { return impl()->sycl_ctx(); }
    xpu::context_t &ctx() override { return impl()->sycl_ctx(); }

    ::sycl::event get_output_event() const {
        return impl()->get_output_event();
    }

    bool recording() const;

protected:
    xpu::sycl::stream_impl_t *impl() const {
        return (xpu::sycl::stream_impl_t *)impl::stream_t::impl_.get();
    }

    stream_t(impl::engine_t *engine, impl::stream_impl_t *stream_impl)
        : gpu::intel::stream_t(engine, stream_impl) {}

private:
    status_t init();
};

} // namespace sycl
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
