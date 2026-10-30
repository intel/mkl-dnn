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

#ifndef XPU_STREAM_PROFILER_HPP
#define XPU_STREAM_PROFILER_HPP

#include <algorithm>
#include <cassert>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <unordered_set>

#include "common/c_types_map.hpp"
#include "common/utils.hpp"
#include "common/verbose_profiler.hpp"

#include "xpu/context.hpp"

namespace dnnl {
namespace impl {
namespace xpu {

struct stream_profiler_t {
    stream_profiler_t(const stream_t *stream, int stamp = 0)
        : stamp_(stamp), stream_(stream) {}
    virtual ~stream_profiler_t() = default;

    struct entry_t {
        uint64_t min_nsec = std::numeric_limits<uint64_t>::max();
        uint64_t max_nsec = 0;
        double freq = 0;
        int kernel_count = 0;

        uint64_t get_nsec() const { return max_nsec - min_nsec; }
    };

    struct registered_event_t {
        registered_event_t(
                std::unique_ptr<xpu::event_t> &&event, uint64_t stamp)
            : event(std::move(event)), stamp(stamp) {}

        std::unique_ptr<xpu::event_t> event;
        uint64_t stamp;
    };

    virtual status_t get_info(profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const
            = 0;

    uint64_t stamp() const { return stamp_; }

    status_t count_entries(
            profiling_data_kind_t data_kind, int *num_entries) const {
        if (data_kind == profiling_data_kind::time_per_kernel) {
            *num_entries = (int)events_.size();
            return status::success;
        }
        std::unordered_set<uint64_t> seen;
        for (auto &ev : events_)
            seen.insert(ev.stamp);
        *num_entries = (int)seen.size();
        return status::success;
    }

    void register_event(std::unique_ptr<xpu::event_t> &&event) {
        events_.emplace_back(std::move(event), stamp_);
    }

    void reset() {
        events_.clear();
        m_.lock();
        stamp_ = 0;
        m_.unlock();
    }

    // The contract is profiler interfaces are called only in between
    // `start_profiling` and `stop_profiling`, which provide a secure
    // multi-threaded access because of the lock. It allows to strip the lock
    // from all other calls, e.g., `stamp, or `register_event` (except `reset`)
    // to reduce the overhead for profiling.
    void start_profiling() {
        m_.lock();
        stamp_++;
    }
    void stop_profiling() { m_.unlock(); }

    void set_callback(void (*callback)(uint64_t, uint64_t)) {
        callback_ = callback;
    }

    status_t notify_profiling_complete() const {
        if (callback_) callback_(0, std::numeric_limits<uint64_t>::max());
        return status::success;
    }

protected:
    // Per-event start/end in nanoseconds.
    virtual status_t query_event_time(
            const xpu::event_t &, uint64_t &, uint64_t &) const {
        return status::unimplemented;
    }

    virtual status_t query_event_freq(
            const xpu::event_t &, double &freq) const {
        freq = 0.0;
        return status::success;
    }

    status_t get_info_generic(profiling_data_kind_t data_kind, int *num_entries,
            uint64_t *data) const {
        if (!num_entries) return status::invalid_arguments;
        bool is_per_kernel
                = (data_kind == profiling_data_kind::time_per_kernel);
        if (!data) return count_entries(data_kind, num_entries);

        std::map<uint64_t, entry_t> stamp2entry;
        int idx = 0;
        for (auto &ev : events_) {
            uint64_t beg = 0, end = 0;
            CHECK(query_event_time(*ev.event, beg, end));
            if (is_per_kernel) {
                data[idx++] = end - beg;
                continue;
            }
            double freq = 0.0;
            CHECK(query_event_freq(*ev.event, freq));
            auto &entry = stamp2entry[ev.stamp];
            entry.min_nsec = std::min(entry.min_nsec, beg);
            entry.max_nsec = std::max(entry.max_nsec, end);
            entry.freq += freq;
            entry.kernel_count++;
        }
        if (is_per_kernel) return status::success;
        return get_info_impl(stamp2entry, data_kind, data);
    }

    status_t get_info_impl(const std::map<uint64_t, entry_t> &stamp2entry,
            profiling_data_kind_t data_kind, uint64_t *data) const {
        int idx = 0;
        for (auto &kv : stamp2entry) {
            auto &e = kv.second;
            switch ((int)data_kind) {
                case profiling_data_kind::time: data[idx] = e.get_nsec(); break;
                case profiling_data_kind::cycles: {
                    double freq = e.freq / e.kernel_count;
                    data[idx] = static_cast<uint64_t>(
                            freq * static_cast<double>(e.get_nsec()) / 1e9);
                    if (callback_) callback_(kv.first, e.get_nsec());
                    break;
                }
                default: assert(!"unexpected data kind");
            }
            idx++;
        }
        return status::success;
    }

    std::recursive_mutex m_;
    std::vector<registered_event_t> events_;
    uint64_t stamp_;
    const stream_t *stream_;
    void (*callback_)(uint64_t, uint64_t) = nullptr;
};

// XPU (OpenCL/SYCL/L0) specialization of verbose_profiler_t.
// Tracks primitive completion using device-side xpu::event_t handles.
// Device-measured execution times are retrieved via get_aggregate_exec_time()
// using runtime-specific event timestamp queries.
// Instantiated per-thread via thread_local_storage_t on the GPU stream.
struct verbose_profiler_t : public impl::verbose_profiler_t {
    using impl::verbose_profiler_t::verbose_profiler_t;

    struct prim_profile_data_t {
        uint64_t component_kind_ = 0;
        double start_ms_ = 0.0;
        std::string pd_info_;
        std::vector<std::shared_ptr<xpu::event_t>> prim_events_;
    };

    void update_event_list() override { profiling_data_.emplace_back(); }

    // appends primitive event to the last primitive entry in profiling_data_
    void register_event(const std::shared_ptr<xpu::event_t> &event) {
        if (!event || profiling_data_.empty()) return;
        profiling_data_.back().prim_events_.push_back(event);
    }

    // populates profiling metadata for the last primitive entry in
    // profiling_data_
    void add_to_pending_primitive_list(double start_ms,
            const std::string &pd_info, uint64_t component) override;

    // Completed primitive executions are periodically checked and logged
    // during after_exec_hook() calls and during stream destruction.
    // The profiler does not wait for pending events to complete
    // and instead prints them at the next concurrent after_exec_hook()
    // call.
    void check_for_completed_primitives() override;

protected:
    std::vector<prim_profile_data_t> profiling_data_;

    // destructor logic to check for unlogged primitives before
    // stream destruction
    void cleanup();

private:
    // allows execution time-tracking for different component kinds
    void log_profile(const prim_profile_data_t &prof_data, double duration_ms);

    // This is invoked during profiler destruction to account for any
    // pending primitives that have not yet been logged.
    void wait_for_pending_primitives() override;

    void reset() { profiling_data_.clear(); }

    virtual status_t get_aggregate_exec_time(
            size_t index, double &duration_ms) const
            = 0;
    virtual bool is_event_complete(
            const std::shared_ptr<xpu::event_t> &event) const
            = 0;
    virtual void wait_for_event_completion(
            const std::shared_ptr<xpu::event_t> &event) const
            = 0;
};

} // namespace xpu
} // namespace impl
} // namespace dnnl

#endif
