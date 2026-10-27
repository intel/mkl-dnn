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

#include "xpu/stream_profiler.hpp"
#include "common/verbose.hpp"

namespace dnnl {
namespace impl {
namespace xpu {

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

    // Adds metadata to the last entry
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
    // the polling check is paused at the first pending primitive
    // and resumed again at the next check. This ensures the
    // primitives are logged in the same order they were enqueued
    size_t first_pending = profiling_data_.size();

    for (size_t index = 0; index < profiling_data_.size(); ++index) {
        const auto &prof_data = profiling_data_[index];
        const auto &evts = prof_data.prim_events_;
        double duration_ms = 0.0;

        // Handles primitives with no kernel events
        if (evts.empty()) {
            // A profiling data entry with a valid pd_info value but no
            // events is treated as a zero-kernel primitive and will be
            // erased in the second pass
            if (!prof_data.pd_info_.empty())
                log_profile(prof_data, duration_ms);
            continue;
        }

        // No point in logging entries without info strings
        if (prof_data.pd_info_.empty()) continue;

        // Check if the current primitive has finished executing and break
        // the loop if it is pending completion
        if (!is_event_complete(evts.back())) {
            first_pending = index;
            break;
        }
        status_t status = get_aggregate_exec_time(index, duration_ms);
        if (status == status::success) {
            log_profile(prof_data, duration_ms);
        } else {
            // using debuginfo=7 level for debugging profiler errors, warnings
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
    // For in-order queues, waiting on the last event is sufficient since
    // all previous events will have completed when the last one completes
    if (!profiling_data_.empty()) {
        // Find the last primitive with non-empty events
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

    if (!profiling_data_.empty())
        VERROR(primitive, runtime,
                "profiling error: failed to log all pending primitives");
}

} // namespace xpu
} // namespace impl
} // namespace dnnl
