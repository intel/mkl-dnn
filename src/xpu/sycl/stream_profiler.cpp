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

#include <limits>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"

#include "xpu/sycl/context.hpp"
#include "xpu/sycl/stream_profiler.hpp"
#include "xpu/sycl/utils.hpp"

namespace dnnl {
namespace impl {
namespace xpu {
namespace sycl {

namespace {

// Queries start and/or end timestamp in nanoseconds for a single event.
// Either start or end may be nullptr to skip that half of the query.
status_t query_event_timestamps(const xpu::sycl::event_t &event, size_t index,
        uint64_t *start, uint64_t *end, bool use_tag_mode) {
    using namespace ::sycl::info;
    assert(index < event.size());

#ifdef DNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG
    // When using the SYCL profiling tags, elapsed time is measured
    // between two bracketing tag events that wrap all kernels:
    // - a "start" tag submitted just before the first kernel
    // - an "end" tag submitted just after the last kernel.
    //
    // event queue: ... |start_tag| |kernel| |end_tag| ...
    //
    // Per the sycl_ext_oneapi_profiling_tag spec, the time for the
    // kernel event completion is calculated from:
    // (start_tag.command_end - end_tag.command_start)
    // i.e. from the moment the start tag finished executing to the
    // moment the end tag began executing, which tightly bounds the
    // kernel execution time.
    if (use_tag_mode) {
        assert(event.event_tags_.size() == event.events.size());
        if (start)
            *start = event.event_tags_[index]
                             .first.get_profiling_info<
                                     event_profiling::command_end>();
        if (end)
            *end = event.event_tags_[index]
                           .second.get_profiling_info<
                                   event_profiling::command_start>();
        return status::success;
    }
#endif

    if (start)
        *start = event.events[index]
                         .get_profiling_info<event_profiling::command_start>();
    if (end)
        *end = event.events[index]
                       .get_profiling_info<event_profiling::command_end>();
    return status::success;
}

} // namespace

status_t stream_profiler_t::get_info(profiling_data_kind_t data_kind,
        int *num_entries, uint64_t *data) const {
    return get_info_generic(data_kind, num_entries, data);
}

status_t stream_profiler_t::query_event_time(
        const xpu::event_t &event, uint64_t &start, uint64_t &end) const {
    using namespace ::sycl::info;
    const auto &sycl_event = xpu::sycl::event_t::from(event);
    assert(sycl_event.size() == 1);
    // Pass use_tag_mode=false: stream_profiler_t never populates
    // event_tags_, so taking the tag branch would read garbage.
    return query_event_timestamps(
            sycl_event, 0, &start, &end, /* use_tag_mode = */ false);
}

status_t verbose_profiler_t::get_aggregate_exec_time(
        size_t index, double &duration_ms) const {
    if (!active_) return status::success;

    if (index >= profiling_data_.size()) {
        VERROR(primitive, exec,
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

    const bool tag_mode = use_ext_oneapi_tag();
    uint64_t agg_start = std::numeric_limits<uint64_t>::max();
    uint64_t agg_end = 0;

    // For verbose logging, aggregate execution time for a primitive is
    // determined from the start time of the first queued primitive event
    // and the end time of the last primitive event
    for (const auto &ev : evts) {
        if (!ev) continue;
        const auto &sycl_ev = xpu::sycl::event_t::from(*ev);
        assert(sycl_ev.size() > 0);
        const size_t last_idx = sycl_ev.size() - 1;

        uint64_t ev_start = 0, ev_end = 0;
        CHECK(query_event_timestamps(sycl_ev, 0, &ev_start, nullptr, tag_mode));
        CHECK(query_event_timestamps(
                sycl_ev, last_idx, nullptr, &ev_end, tag_mode));

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

    const auto &sycl_event = xpu::sycl::event_t::from(*event);
    assert(sycl_event.size() > 0);
    size_t last_idx = sycl_event.size() - 1;

    auto status
            = sycl_event[last_idx]
                      .get_info<
                              ::sycl::info::event::command_execution_status>();
    return (status == ::sycl::info::event_command_status::complete);
}

void verbose_profiler_t::wait_for_event_completion(
        const std::shared_ptr<xpu::event_t> &event) const {
    if (!active_) return;
    if (!event) return;

    const auto &sycl_event = xpu::sycl::event_t::from(*event);
    assert(sycl_event.size() > 0);
    size_t last_idx = sycl_event.size() - 1;

    try {
        ::sycl::event::wait({sycl_event[last_idx]});
    } catch (const ::sycl::exception &e) {
        // Note: Cannot throw from destructor context, so just
        // logging error
        VERROR(primitive, exec, "failed to wait for event completion: %s",
                e.what());
    }
}

} // namespace sycl
} // namespace xpu
} // namespace impl
} // namespace dnnl
