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

#ifndef CPU_THREADPOOL_STREAM_IMPL_HPP
#define CPU_THREADPOOL_STREAM_IMPL_HPP

#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

#include "common/c_types_map.hpp"
#include "common/stream_impl.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace threadpool {

class stream_impl_t : public impl::stream_impl_t {
public:
    stream_impl_t() = delete;
    stream_impl_t(unsigned flags) : impl::stream_impl_t(flags) {}
    stream_impl_t(threadpool_interop::threadpool_iface *threadpool)
        : impl::stream_impl_t(stream_flags::in_order)
        , threadpool_(threadpool) {}

    ~stream_impl_t() override = default;

    threadpool_interop::threadpool_iface *get_threadpool() const {
        return threadpool_;
    }

    bool is_syncable() const override {
        if (!threadpool_) return true;
        return !(threadpool_->get_flags()
                & threadpool_interop::threadpool_iface::ASYNCHRONOUS);
    }

    // Marks the stream as capable of asynchronous verbose profiling. Whether
    // profiling is active is checked at runtime against the verbose level, so
    // dnnl_set_verbose() can enable or disable it for existing streams.
    status_t init_verbose_profiler(engine_kind_t engine_kind) override {
        use_verbose_profiler_ = false;

        // The checks are only relevant for a CPU engine
        if (engine_kind != engine_kind::cpu) return status::success;

        // Profiling relies on events from an asynchronous threadpool.
        if (is_syncable()) return status::success;

        use_verbose_profiler_ = true;
        return status::success;
    }

private:
    DNNL_DISALLOW_COPY_AND_ASSIGN(stream_impl_t)

    threadpool_interop::threadpool_iface *threadpool_ = nullptr;
};

} // namespace threadpool
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
