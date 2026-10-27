/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include <CL/cl.h>

#include <limits>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "xpu/ocl/context.hpp"
#include "xpu/ocl/stream_profiler.hpp"

#include "gpu/gpu_stream.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace ocl {

namespace {

// Queries start and/or end timestamp in nanoseconds for a single event.
// Either start or end may be nullptr to skip that half of the query.
status_t query_event_timestamps(const xpu::ocl::event_t &event, size_t index,
        uint64_t *start, uint64_t *end) {

    if (start) {
        cl_ulong beg = 0;
        OCL_CHECK(xpu::ocl::clGetEventProfilingInfo(event[index].get(),
                CL_PROFILING_COMMAND_START, sizeof(beg), &beg, nullptr));
        *start = static_cast<uint64_t>(beg);
    }
    if (end) {
        cl_ulong e = 0;
        OCL_CHECK(xpu::ocl::clGetEventProfilingInfo(event[index].get(),
                CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr));
        *end = static_cast<uint64_t>(e);
    }
    return status::success;
}

} // namespace

status_t stream_profiler_t::get_info(profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) const {
    return get_info_generic(data_kind, num_entries, data);
}

status_t stream_profiler_t::query_event_time(
        const xpu::event_t &event, uint64_t &start, uint64_t &end) const {
    const auto &ocl_event = xpu::ocl::event_t::from(event);
    assert(ocl_event.size() == 1);
    return query_event_timestamps(ocl_event, 0, &start, &end);
}

status_t stream_profiler_t::query_event_freq(
        const xpu::event_t &event, double &freq) const {
    const auto *gpu_stream = utils::downcast<const gpu::stream_t *>(stream_);
    freq = gpu_stream->get_freq(event);
    return status::success;
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
    if (evts.empty()) {
        duration_ms = 0.0;
        return status::success;
    }

    cl_ulong agg_start = std::numeric_limits<cl_ulong>::max();
    cl_ulong agg_end = 0;

    // For verbose logging, aggregate execution time for a primitive is
    // determined from the start time of the first queued primitive event
    // and the end time of the last primitive event
    for (const auto &ev : evts) {
        const auto &ocl_event = xpu::ocl::event_t::from(*ev);
        assert(ocl_event.size() > 0);
        size_t last_idx = ocl_event.size() - 1;

        cl_ulong ev_start = 0, ev_end = 0;
        CHECK(query_event_timestamps(ocl_event, 0, &ev_start, nullptr));
        CHECK(query_event_timestamps(ocl_event, last_idx, nullptr, &ev_end));
        agg_start = std::min(agg_start, ev_start);
        agg_end = std::max(agg_end, ev_end);
    }

    if (agg_end < agg_start) { return status::runtime_error; }
    duration_ms = static_cast<double>(agg_end - agg_start) * 1e-6;
    return status::success;
}

bool verbose_profiler_t::is_event_complete(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!active_) return true;
    if (!event) return true;
    const auto &ocl_event = xpu::ocl::event_t::from(*event);
    assert(ocl_event.size() > 0);
    size_t last_idx = ocl_event.size() - 1;
    cl_int status;
    cl_int err = xpu::ocl::clGetEventInfo(ocl_event[last_idx],
            CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &status,
            nullptr);

    if (err != CL_SUCCESS) {
        VDEBUGINFO(7, primitive, exec, "failed to get event status: %d", err);
        return false;
    }

    return status == CL_COMPLETE;
}

void verbose_profiler_t::wait_for_event_completion(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!active_) return;
    if (!event) return;
    const auto &ocl_event = xpu::ocl::event_t::from(*event);
    assert(ocl_event.size() > 0);
    size_t last_idx = ocl_event.size() - 1;
    cl_event cl_ev = ocl_event[last_idx];
    cl_int err = xpu::ocl::clWaitForEvents(1, &cl_ev);

    if (err != CL_SUCCESS) {
        // Note: Cannot throw from destructor context, so just
        // logging error
        VDEBUGINFO(7, primitive, exec,
                "failed to wait for event completion: %d", err);
    }
}

} // namespace ocl
} // namespace xpu
} // namespace impl
} // namespace dnnl
