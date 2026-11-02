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

#ifndef CPU_CPU_STREAM_IMPL_HPP
#define CPU_CPU_STREAM_IMPL_HPP

#include "common/stream_impl.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// CPU-specific stream implementation extending the common stream_impl_t.
// Adds threadpool handle storage and verbose profiler initialization
// for the threadpool runtime.
class stream_impl_t : public impl::stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags) : impl::stream_impl_t(flags) {}

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    stream_impl_t(threadpool_interop::threadpool_iface *threadpool)
        : impl::stream_impl_t(stream_flags::in_order)
        , threadpool_(threadpool) {}
#endif

    ~stream_impl_t() override = default;

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    bool is_syncable() const override {
        if (!threadpool_) return true;
        return !(threadpool_->get_flags()
                & threadpool_interop::threadpool_iface::ASYNCHRONOUS);
    }

    // Checks and initializes profiler for supported runtime configs
    status_t init_verbose_profiler(engine_kind_t engine_kind) override {
        use_verbose_profiler_ = false;

        // The checks are only relevant for a CPU engine
        if (engine_kind != engine_kind::cpu) return status::success;

        if (!dnnl::impl::get_verbose(dnnl::impl::verbose_t::exec_profile))
            return status::success;

        // CPU stream must be backed by an async threadpool to support
        // profiling
        if (!is_syncable()) return status::success;

        use_verbose_profiler_ = true;
        return status::success;
    }

    status_t get_threadpool(
            threadpool_interop::threadpool_iface **threadpool) const {
        *threadpool = threadpool_;
        return status::success;
    }
#endif

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_impl_t)
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    threadpool_interop::threadpool_iface *threadpool_ = nullptr;
#endif
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
