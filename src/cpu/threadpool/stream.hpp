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

#ifndef CPU_THREADPOOL_STREAM_HPP
#define CPU_THREADPOOL_STREAM_HPP

#include <memory>
#include <string>

#include "common/c_types_map.hpp"
#include "common/thread_local_storage.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_stream.hpp"
#include "cpu/threadpool/stream_impl.hpp"
#include "cpu/threadpool/verbose_profiler.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace threadpool {

struct stream_t : public cpu::cpu_stream_t {
    stream_t(impl::engine_t *engine, threadpool::stream_impl_t *stream_impl);
    ~stream_t() override = default;

    threadpool::stream_impl_t *impl() const {
        return utils::downcast<threadpool::stream_impl_t *>(
                impl::stream_t::impl());
    }

    status_t wait() override;

    void before_exec_hook() override;
    void after_exec_hook() override;

    status_t run_verbose_profiler(const std::string &pd_info, double start_ms,
            uint64_t component) override;

    // Returns the active per-thread verbose profiler, or nullptr if the
    // profiler has not been initialized for this thread.
    verbose_profiler_t *verbose_profiler() const {
        return (is_verbose_profiler_enabled() && verbose_profiler_.is_set())
                ? verbose_profiler_.get().get()
                : nullptr;
    }

private:
    // Each submission thread maintains its own independent profiling state.
    // Worker threads interact only with threadpool_event_iface_t directly via
    // the completion mechanism and never access the profiler state.
    utils::thread_local_storage_t<std::unique_ptr<verbose_profiler_t>>
            verbose_profiler_;
};

} // namespace threadpool
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
