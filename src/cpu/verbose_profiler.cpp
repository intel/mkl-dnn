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

#include "cpu/verbose_profiler.hpp"

#if DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

void verbose_profiler_t::cleanup() {
    try {
        wait_for_pending_primitives();
    } catch (const std::bad_alloc &) {
        VERROR(primitive, exec,
                "profiler cleanup failed: out of memory during event "
                "processing");
    } catch (const std::runtime_error &e) {
        VERROR(primitive, exec, "profiler cleanup failed: runtime error - %s",
                e.what());
    } catch (const std::exception &e) {
        VERROR(primitive, exec, "profiler cleanup failed: %s", e.what());
    } catch (...) {
        VERROR(primitive, exec, "profiler cleanup failed: unknown error");
    }
}

void verbose_profiler_t::add_to_pending_primitive_list(
        double start_ms, const std::string &pd_info, uint64_t component) {
    assert(!profiling_data_.empty());

    auto &last_entry = profiling_data_.back();
    last_entry.start_ms_ = start_ms;
    last_entry.pd_info_ = pd_info;
    last_entry.component_kind_ = component;
}

void verbose_profiler_t::log_profile(
        const prim_profile_data_t &prof_data, double duration_ms) {
    switch (static_cast<component_t::flag_kind>(prof_data.component_kind_)) {
        case component_t::graph:
            VPROF(prof_data.start_ms_, graph, exec, VERBOSE_profile,
                    prof_data.pd_info_.c_str(), duration_ms);
            break;
        case component_t::ukernel:
            VPROF(prof_data.start_ms_, ukernel, exec, VERBOSE_profile,
                    prof_data.pd_info_.c_str(), duration_ms);
            break;
        case component_t::primitive:
        default:
            VPROF(prof_data.start_ms_, primitive, exec, VERBOSE_profile,
                    prof_data.pd_info_.c_str(), duration_ms);
            break;
    }
}

void verbose_profiler_t::check_for_completed_primitives() {
    if (!active_) return;

    // Polling stops at the first pending primitive to ensure primitives
    // are logged in the same order they were enqueued. Pending primitives
    // remain in profiling_data_ until the next polling cycle.
    size_t first_pending = profiling_data_.size();

    for (size_t index = 0; index < profiling_data_.size(); ++index) {
        const auto &prof_data = profiling_data_[index];
        const auto &evts = prof_data.prim_events_;
        double duration_ms = 0.0;

        // Handles primitives with no threadpool
        if (evts.empty()) {
            // A profiling data entry with a valid pd_info value but no
            // events is treated as a primitive that is executing
            // without a threadpool - the profiler logs zero execution
            // time with a debug warning.
            VDEBUGINFO(7, primitive, exec,
                    "profiling error: no threadpool event found for "
                    "profiling_data_ entry %zu",
                    index);

            if (!prof_data.pd_info_.empty())
                log_profile(prof_data, duration_ms);
            continue;
        }

        // No point in logging entries without info strings
        if (prof_data.pd_info_.empty()) continue;

        // Stop polling at the first incomplete primitive to preserve
        // in-order logging.
        if (!is_event_complete(evts.back())) {
            first_pending = index;
            break;
        }

        status_t status = get_aggregate_exec_time(index, duration_ms);
        if (status == status::success) {
            log_profile(prof_data, duration_ms);
        } else {
            VDEBUGINFO(7, common, runtime,
                    "%s, profiling error: failures in exec time computation",
                    prof_data.pd_info_.c_str());
        }
    }

    // A second pass through profiling_data_ erases the entry for all
    // completed to avoid blowing up the size of profiling_data_
    if (first_pending > 0) {
        profiling_data_.erase(profiling_data_.begin(),
                profiling_data_.begin() + first_pending);
    }
}

void verbose_profiler_t::wait_for_pending_primitives() {
    if (!active_) return;

    // For in-order threadpool dispatch, waiting on the last event is
    // sufficient since all prior dispatches complete before the last one.
    // Iterate in reverse to find the last primitive with a registered event.
    if (!profiling_data_.empty()) {
        for (auto it = profiling_data_.rbegin(); it != profiling_data_.rend();
                ++it) {
            const auto &evts = it->prim_events_;
            if (!evts.empty() && evts.back()) {
                wait_for_event_completion(evts.back());
                break;
            }
        }
    }

    check_for_completed_primitives();

    // Any remaining entries after the final check indicate a profiling
    // failure — log an error to avoid silent data loss
    if (!profiling_data_.empty())
        VERROR(primitive, runtime,
                "profiling error: failed to log all pending primitives");
}

status_t verbose_profiler_t::get_aggregate_exec_time(
        size_t index, double &duration_ms) const {
    if (!active_) return status::success;

    if (index >= profiling_data_.size()) {
        VDEBUGINFO(7, primitive, exec,
                "profiling error: invalid index %zu, profiling_data size "
                "is %zu",
                index, profiling_data_.size());
        return status::success;
    }

    const auto &prof_data = profiling_data_[index];
    const auto &evts = prof_data.prim_events_;

    // No event: primitive had no dispatch or profiler was paused
    if (evts.empty()) {
        duration_ms = 0.0;
        return status::success;
    }

    // Delegate to the threadpool implementor for the measured execution time.
    // exec_time_ms() is only valid after is_complete() returns true, which
    // is guaranteed by the check in check_for_completed_primitives().
    duration_ms = 0.0;
    for (const auto &ev : evts) {
        if (!ev) continue;
        // The XPU-variant of the profiler which computes the execution time
        // from the start time of the first event and the end time of the last
        // event to capture wall-clock overlap between concurrent kernels.
        // for threadpool, however, we sum exec_time_ms() directly since the
        // threadpool interface exposes only a single measured duration per event
        //  rather than separate start/end timestamps.
        duration_ms += ev->exec_time_ms();
    }
    return status::success;
}

bool verbose_profiler_t::is_event_complete(const std::shared_ptr<
        dnnl::threadpool_interop::threadpool_event_iface_t> &event) const {
    if (!active_) return true;

    // Empty event is treated as immediately complete and the check exits
    // with a debug warning
    if (!event) {
        VDEBUGINFO(7, primitive, exec,
                "profiling error: no threadpool event found");
        return true;
    }
    return event->is_complete();
}

void verbose_profiler_t::wait_for_event_completion(const std::shared_ptr<
        dnnl::threadpool_interop::threadpool_event_iface_t> &event) const {
    if (!active_) return;

    // Empty event is treated as immediately complete and the check exits
    // with a debug warning
    if (!event) {
        VDEBUGINFO(7, primitive, exec,
                "profiling error: no threadpool event found");
        return;
    }
    event->wait();
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL
