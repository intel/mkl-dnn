/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GEMMSTONE_DSL_UTILS_PROFILER_HPP
#define GEMMSTONE_DSL_UTILS_PROFILER_HPP

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "dsl/utils/utils.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

static double get_msec() {
    return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

// Record custom profiling information within a single thread.
//
// Basic Usage:
//        profiler_t profile("Profile Name"); // Start profiling
//        ...                                 // Some code
//        profile.stamp("Stamp Name 1");      // Record data
//        ...                                 // Some code
//        profile.stamp("Stamp Name 2");      // Record data
//        ...                                 // Some code
//        profile.stop("Stamp Name 3");       // Record data
//        std::cout << profile << "\n";       // Print recorded results
//
// The profiler_t structure can be used over multiple calls to a function, and
// all time stamps will be collated into a total time.
// Alternative Usage:
//        profiler_t profile("Profile Name");
//        function() {
//           profile.start()                     // Start Profiling
//           ...                                 // Some code
//           profile.stamp("Stamp Name 1");      // Record data
//           ...                                 // Some code
//           profile.stamp("Stamp Name 2");      // Record data
//           ...                                 // Some code
//           profile.stop("Stamp Name 3");       // Record data
//        }
//
//        high_level_function() {
//           profile.reset()                     // Clear unwanted profiling data
//           ...                                 // Some code
//           std::cout << profile << "\n";       // Print recorded results
//        }

// Note: To reduce overhead, the stamp names are initially recorded as pointers.
// All pointers need to be valid when stop() is called, at which point stamp
// names are copied into long term storage.

struct profiler_t : public stringify_t<profiler_t> {
    profiler_t(const std::string &profile_name) : _profile_name(profile_name) {
        // Reserve data on construction to reduce chance of recording
        // reallocation
        _run_data.reserve(128);
        start();
    }

    // Start recording timing data
    void start() {
        _run_data.clear();
        _state = RUNNING;
        _start_time = get_msec();
        optimization_barrier();
    }

    // Recording data
    void stamp(const char *name) {
        optimization_barrier();
        _run_data.emplace_back(name, get_msec());
        dsl_assert(_state == RUNNING);
        optimization_barrier();
    }

    void stop(const char *name) {
        optimization_barrier();
        _run_data.emplace_back(name, get_msec());
        stop();
    }

    void stop() {
        dsl_assert(_state == RUNNING);
        _state = STOPPED;
        collate();
    }

    void reset() {
        _data.clear();
        start();
    }

    std::string str() const {
        ostringstream_t oss;
        std::vector<record_t<std::string>> print_data(
                _data.begin(), _data.end());

        std::sort(print_data.begin(), print_data.end());

        prof_time_t total_time = 0;
        for (auto &record : print_data) {
            total_time += record.time;
        }

        const int max_name_width = 20;
        oss << _profile_name << ":\n";
        oss << std::setw(max_name_width) << "Total Time" << std::setw(0)
            << ":          " << std::setw(8) << std::fixed
            << std::setprecision(3) << total_time << std::setw(0) << " ms\n";
        for (const auto &record : print_data) {
            oss << std::setw(max_name_width)
                << record.name.substr(0, max_name_width) << std::setw(0)
                << ": interval " << std::setw(8) << std::fixed
                << std::setprecision(3) << record.time << std::setw(0)
                << " ms, total " << std::setw(8)
                << 100.0 * record.time / total_time << " % recorded time\n";
        }

        return oss.str();
    }

private:
    using prof_time_t = double;

    static void inline optimization_barrier() {
        atomic_signal_fence(std::memory_order_seq_cst);
    }

    template <typename T>
    struct record_t {
        T name;
        prof_time_t time;
        record_t(T name, prof_time_t time) : name(name), time(time) {}
        record_t(const std::pair<T, prof_time_t> &record)
            : name(record.first), time(record.second) {}
        // Reversed time ordering
        bool operator<(const record_t &b) const { return this->time > b.time; }
    };

    enum state_t {
        RUNNING,
        STOPPED,
    };

    // Record data in a vector to minimize overhead associated with recording
    // data. Data is flushed into a separate long term storage once time
    // recording has stopped.
    std::string _profile_name;
    std::vector<record_t<const char *>> _run_data;
    std::unordered_map<std::string, prof_time_t> _data;

    state_t _state = STOPPED;
    prof_time_t _start_time = 0;

    void collate() {
        dsl_assert(_state == STOPPED);
        prof_time_t last_stamp = _start_time;
        for (const auto &record : _run_data) {
            _data[std::string(record.name)] += record.time - last_stamp;
            last_stamp = record.time;
        }
        _run_data.clear();
    }
};

} // namespace dsl
GEMMSTONE_NAMESPACE_END
#endif
