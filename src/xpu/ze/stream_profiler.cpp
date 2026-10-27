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

#include <limits>
#include <map>
#include <unordered_set>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "xpu/ze/context.hpp"
#include "xpu/ze/stream_profiler.hpp"

#include "gpu/gpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ze {

status_t verbose_profiler_t::get_aggregate_exec_time(
        size_t index, double &duration_ms) const {
    if (!active_) return status::success;

    if (index >= profiling_data_.size()) {
        VDEBUGINFO(7, primitive, exec,
                "profiling error: invalid index %zu, profiling_data size is "
                "%zu",
                index, profiling_data_.size());
        return status::success;
    }
    const auto &prof_data = profiling_data_[index];

    const auto &evts = prof_data.prim_events_;
    if (evts.empty()) {
        duration_ms = 0.0;
        return status::success;
    }

    uint64_t agg_start = std::numeric_limits<uint64_t>::max();
    uint64_t agg_end = 0;

    // For verbose logging, aggregate execution time for a primitive is
    // determined from the start time of the first queued primitive event
    // and the end time of the last primitive event
    for (const auto &ev : evts) {
        const ze::event_t &ze_event = ze::event_t::from(*ev);
        assert(ze_event.size() > 0);
        size_t last_idx = ze_event.size() - 1;

        ze_kernel_timestamp_result_t timestamp_result;
        ZE_CHECK(xpu::ze::zeEventQueryKernelTimestamp(
                ze_event[0], &timestamp_result));
        uint64_t start_ns = timestamp_result.global.kernelStart;
        ZE_CHECK(xpu::ze::zeEventQueryKernelTimestamp(
                ze_event[last_idx], &timestamp_result));
        uint64_t end_ns = timestamp_result.global.kernelEnd;

        agg_start = std::min(agg_start, start_ns);
        agg_end = std::max(agg_end, end_ns);
    }

    // Ideally, this calls for consolidation between timing calculation
    // routines between different profilers to avoid code duplication
    // but the different computation code flow for the two profilers
    // discourages this
    uint64_t duration_cycles = get_duration_cycles(agg_start, agg_end);
    uint64_t duration_ns
            = static_cast<uint64_t>(timestamp_freq_ * duration_cycles);
    duration_ms = static_cast<double>(duration_ns) * 1e-6;
    return status::success;
}

bool verbose_profiler_t::is_event_complete(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!active_) return true;
    if (!event) return true;

    const ze::event_t &ze_event = ze::event_t::from(*event);
    assert(ze_event.size() > 0);
    size_t last_idx = ze_event.size() - 1;

    ze_result_t result = xpu::ze::zeEventQueryStatus(ze_event[last_idx]);
    if (result != ZE_RESULT_SUCCESS)
        VDEBUGINFO(
                7, primitive, exec, "failed to get event status: %d", result);
    return result == ZE_RESULT_SUCCESS;
}

void verbose_profiler_t::wait_for_event_completion(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!active_) return;
    if (!event) return;

    const ze::event_t &ze_event = ze::event_t::from(*event);
    assert(ze_event.size() > 0);
    size_t last_idx = ze_event.size() - 1;

    ze_result_t result = xpu::ze::zeEventHostSynchronize(
            ze_event[last_idx], std::numeric_limits<uint64_t>::max());
    if (result != ZE_RESULT_SUCCESS) {
        // Note: Cannot throw from destructor context, so just logging error
        VDEBUGINFO(7, primitive, exec,
                "failed to wait for event completion: %d", result);
    }
}

} // namespace ze
} // namespace xpu
} // namespace impl
} // namespace dnnl
