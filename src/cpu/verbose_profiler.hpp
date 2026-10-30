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

#ifndef CPU_VERBOSE_PROFILER_HPP
#define CPU_VERBOSE_PROFILER_HPP

#include "common/c_types_map.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL

#include <memory>
#include <string>
#include <vector>

#include "common/verbose_profiler.hpp"
#include "oneapi/dnnl/dnnl_threadpool_iface.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

// CPU async threadpool specialization of verbose_profiler_t.
// Tracks primitive completion using threadpool_event_iface_t handles registered
// with the threadpool after each parallel_for() dispatch.
// Instantiated on the submission thread via thread_local_storage_t on the
// CPU stream. Each submission thread maintains its own profiling_data_,
// operating independently from other primitive threads.
// Using a thread_local_storage_t definition is still valid as the
// profiling operations via enqueue_primtive() and pre-exec/post-exec hooks
// are done through a single calling thread.
struct verbose_profiler_t : public impl::verbose_profiler_t {
    using impl::verbose_profiler_t::verbose_profiler_t;

    ~verbose_profiler_t() override { cleanup(); }

    // Holds profiling metadata and threadpool event for a single pending
    // primitive.
    struct prim_profile_data_t {
        uint64_t component_kind_ = 0;
        double start_ms_ = 0.0;
        std::string pd_info_;
        std::vector<std::shared_ptr<
                dnnl::threadpool_interop::threadpool_event_iface_t>>
                prim_events_;
    };

    void update_event_list() override { profiling_data_.emplace_back(); }

    // Sets the threadpool event handle for the last entry in profiling_data_.
    // Called from enqueue_primitive() after registering the primitive event
    // with the threadpool.
    // Unlike the XPU profiler which may register multiple events per
    // primitive (one per kernel), the threadpool profiler registers a
    // single event per primitive dispatch.
    void register_event(const std::shared_ptr<
            dnnl::threadpool_interop::threadpool_event_iface_t> &event) {
        if (!event || profiling_data_.empty()) return;
        profiling_data_.back().prim_events_.push_back(event);
    }

    void add_to_pending_primitive_list(double start_ms,
            const std::string &pd_info, uint64_t component) override;

    void check_for_completed_primitives() override;

protected:
    std::vector<prim_profile_data_t> profiling_data_;

    // destructor logic to check for unlogged primitives before
    // stream destruction
    void cleanup();

private:
    // allows execution time-tracking for different component kinds
    void log_profile(const prim_profile_data_t &prof_data, double duration_ms);

    void wait_for_pending_primitives() override;

    status_t get_aggregate_exec_time(size_t index, double &duration_ms) const;
    bool is_event_complete(const std::shared_ptr<
            dnnl::threadpool_interop::threadpool_event_iface_t> &event) const;
    void wait_for_event_completion(const std::shared_ptr<
            dnnl::threadpool_interop::threadpool_event_iface_t> &event) const;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#endif // CPU_VERBOSE_PROFILER_HPP
