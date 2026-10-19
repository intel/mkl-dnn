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

#ifndef COMMON_STREAM_IMPL_HPP
#define COMMON_STREAM_IMPL_HPP

#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {

// TODO: introduce a stream_impl_t subclass for CPU streams to separate general
// and CPU-specific attributes.
class stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags)
        : use_verbose_profiler_(false), flags_(flags) {}
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    stream_impl_t(threadpool_interop::threadpool_iface *threadpool)
        : use_verbose_profiler_(false)
        , flags_(stream_flags::in_order)
        , threadpool_(threadpool) {}
#endif

    virtual ~stream_impl_t() = default;

    unsigned flags() const { return flags_; }

    bool is_profiling_enabled() const {
        return (flags() & dnnl::impl::stream_flags::profiling);
    }

    bool is_verbose_profiler_enabled() const {
        return dnnl::impl::get_verbose(dnnl::impl::verbose_t::exec_profile)
                && use_verbose_profiler_;
    }

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    bool is_async_threadpool() const {
        if (!threadpool_) return false;
        return threadpool_->get_flags()
                & threadpool_interop::threadpool_iface::ASYNCHRONOUS;
    }

    // Checks and initializes profiler for supported runtime configs
    virtual status_t init_verbose_profiler(engine_kind_t engine_kind) {
        use_verbose_profiler_ = false;

        // The checks are only relevant for a CPU engine
        if (engine_kind != engine_kind::cpu) return status::success;

        if (!dnnl::impl::get_verbose(dnnl::impl::verbose_t::exec_profile))
            return status::success;

        // CPU stream must be backed by an async threadpool to support
        // profiling
        if (!is_async_threadpool()) return status::success;

        use_verbose_profiler_ = true;
        return status::success;
    }

    status_t get_threadpool(
            threadpool_interop::threadpool_iface **threadpool) const {
        *threadpool = threadpool_;
        return status::success;
    }
#else
    bool is_async_threadpool() const { return false; }

    virtual status_t init_verbose_profiler(engine_kind_t) {
        use_verbose_profiler_ = false;
        return status::success;
    }
#endif

protected:
    bool use_verbose_profiler_;

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_impl_t)

    unsigned flags_;
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    threadpool_interop::threadpool_iface *threadpool_ = nullptr;
#endif
};

} // namespace impl
} // namespace dnnl

#endif
