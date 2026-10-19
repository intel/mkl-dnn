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

#ifndef CPU_CPU_STREAM_HPP
#define CPU_CPU_STREAM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "common/thread_local_storage.hpp"
#include "cpu/verbose_profiler.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"
#endif

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/stream.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct cpu_stream_t : public stream_t {
    cpu_stream_t(engine_t *engine, impl::stream_impl_t *stream_impl)
        : stream_t(engine, stream_impl) {
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        // Initialize the verbose profiler based on threadpool capabilities.
        // init_verbose_profiler() enables verbose profiling on the
        // stream_impl_t if all required conditions are met.
        // (Streams created via dnnl_threadpool_interop_stream_create() take
        // this path rather than the dedicated threadpool constructor, so
        // init_verbose_profiler() must be called here too.)
        impl()->init_verbose_profiler(engine->kind());
#endif
    }
    ~cpu_stream_t() override = default;

    dnnl::impl::status_t wait() override {
        // CPU execution is synchronous so return immediately
#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
        dnnl::threadpool_interop::threadpool_iface *tp;
        auto rc = this->get_threadpool(&tp);
        if (rc == status::success && tp) {
            if (tp->get_flags()
                    & threadpool_interop::threadpool_iface::ASYNCHRONOUS)
                tp->wait();
        }
#endif
        return dnnl::impl::status::success;
    }

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
    cpu_stream_t(engine_t *engine,
            dnnl::threadpool_interop::threadpool_iface *threadpool)
        : stream_t(engine, new impl::stream_impl_t(threadpool)) {
        // Initialize the verbose profiler based on threadpool capabilities.
        // init_verbose_profiler() enables verbose profiling on the
        // stream_impl_t if all required conditions are met.
        impl()->init_verbose_profiler(engine->kind());
    }

    // Returns a pointer to the active per-thread verbose profiler,
    // or nullptr if the profiler has not been initialized for this thread.
    cpu::verbose_profiler_t *verbose_profiler() const {
        return (is_verbose_profiler_enabled() && verbose_profiler_.is_set())
                ? verbose_profiler_.get().get()
                : nullptr;
    }

    void before_exec_hook() override {
        dnnl::threadpool_interop::threadpool_iface *tp;
        auto rc = this->get_threadpool(&tp);
        if (rc == status::success) threadpool_utils::activate_threadpool(tp);

        // Instantiate the per-thread verbose profiler on first use and
        // add a new entry for the incoming primitive.
        if (is_verbose_profiler_enabled()) {
            auto &verbose_profiler = verbose_profiler_.get_or_set(
                    utils::make_unique<cpu::verbose_profiler_t>(this));
            verbose_profiler->update_event_list();
        }
    }

    void after_exec_hook() override {
        if (auto *vp = verbose_profiler()) {
            vp->check_for_completed_primitives();
        }
        threadpool_utils::deactivate_threadpool();
    }

    status_t run_verbose_profiler(const std::string &pd_info, double start_ms,
            uint64_t component) override {
        if (!is_verbose_profiler_enabled()) return status::success;

        auto *vp = verbose_profiler();
        if (!vp) return status::success;

        // Retrieve the completion event for the most recently dispatched
        // parallel_for() from the threadpool. This must be called after
        // enqueue_primitive() returns to ensure the event corresponds to
        // the correct dispatch.
        dnnl::threadpool_interop::threadpool_iface *tp;
        CHECK(get_threadpool(&tp));
        if (!tp) {
            VERROR(primitive, exec,
                    "unable to fetch threadpool for verbose profiling");
            return status::success;
        }

        // register_event() is a no-op if event is null, which is the
        // fallback path for threadpools that do not support profiling
        const auto &event = tp->get_event();
        vp->register_event(event);
        vp->add_to_pending_primitive_list(start_ms, pd_info, component);

        return status::success;
    }

private:
    // Per-thread verbose profiler instance.
    // Thread-local storage ensures each submission thread maintains its
    // own independent profiling_data_ list and event tracking state.
    // Worker threads interact only with threadpool_event_iface_t directly via
    // the completion mechanism and never access the profiler state.
    utils::thread_local_storage_t<std::unique_ptr<cpu::verbose_profiler_t>>
            verbose_profiler_;

#endif
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
