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

#include "common/dnnl_thread.hpp"
#include "common/verbose.hpp"

#include "cpu/threadpool/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace threadpool {

stream_t::stream_t(
        impl::engine_t *engine, threadpool::stream_impl_t *stream_impl)
    : cpu::cpu_stream_t(engine, stream_impl) {
    auto status = impl()->init_verbose_profiler(engine->kind());
    if (status != status::success)
        VERROR(common, runtime, "failed to initialize verbose profiling");
}

status_t stream_t::wait() {
    if (!impl()->is_syncable()) impl()->get_threadpool()->wait();
    return status::success;
}

void stream_t::before_exec_hook() {
    threadpool_utils::activate_threadpool(impl()->get_threadpool());

    // Instantiate the per-thread verbose profiler on first use and add a new
    // entry for the incoming primitive.
    if (is_verbose_profiler_enabled()) {
        auto &vp = verbose_profiler_.get_or_set(
                utils::make_unique<verbose_profiler_t>(this));
        vp->update_event_list();
    }
}

void stream_t::after_exec_hook() {
    if (auto *vp = verbose_profiler()) vp->check_for_completed_primitives();
    threadpool_utils::deactivate_threadpool();
}

status_t stream_t::run_verbose_profiler(
        const std::string &pd_info, double start_ms, uint64_t component) {
    if (!is_verbose_profiler_enabled()) {
        VERROR(primitive, exec,
                "running verbose profiler while it is not enabled");
        return status::success;
    }

    auto *vp = verbose_profiler();
    if (!vp) return status::success;

    auto *tp = impl()->get_threadpool();
    if (!tp) {
        VERROR(primitive, exec,
                "unable to fetch threadpool for verbose profiling");
        return status::success;
    }

    // Retrieve the completion event for the most recently dispatched
    // parallel_for(). This must be called after enqueue_primitive() returns
    // so the event corresponds to the correct dispatch. register_event() is a
    // no-op for a null event, which is the fallback for threadpools that do
    // not support profiling.
    vp->register_event(tp->get_event());
    vp->add_to_pending_primitive_list(start_ms, pd_info, component);

    return status::success;
}

} // namespace threadpool
} // namespace cpu
} // namespace impl
} // namespace dnnl
