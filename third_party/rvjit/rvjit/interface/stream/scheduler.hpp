/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef RVJIT_INTERFACE_STREAM_SCHEDULER_HPP
#define RVJIT_INTERFACE_STREAM_SCHEDULER_HPP

#include <algorithm>

namespace rvjit {
namespace stream {

/// Strategy to determine a `delay` value
///
/// @details The `max` parameter acts as a ceiling on the `delay`, e.g. to
///     keep it within the `buffer::buffer_t` sizes backing an unrolled
///     loop nest
///
/// @note The default `max = -1` expresses no upper limit
struct policy_t {
    /// Where the delay comes from
    enum class kind_t {
        memory, // The scheduler's memory instruction latency
        arithmetic, // The scheduler's arithmetic instruction latency
        literal, // The caller's `value`
    };

    kind_t kind;
    int value;
    int max;

    policy_t() : policy_t(zero()) {}

    /// Creates a policy
    ///
    /// @param k Where the delay comes from
    /// @param v The delay of a `literal` policy
    /// @param m The ceiling on the delay, -1 for none
    policy_t(kind_t k, int v, int m) : kind(k), value(v), max(m) {}

    /// Select the memory latency up to `max`
    ///
    /// @param max The ceiling on the delay, -1 for none
    static policy_t memory(int max = -1) {
        return policy_t(kind_t::memory, 0, max);
    }

    /// Select the arithmetic latency up to `max`
    ///
    /// @param max The ceiling on the delay, -1 for none
    static policy_t arithmetic(int max = -1) {
        return policy_t(kind_t::arithmetic, 0, max);
    }

    /// Select the delay `value` up to `max`
    ///
    /// @param value The delay
    /// @param max The ceiling on the delay, -1 for none
    static policy_t literal(int value, int max = -1) {
        return policy_t(kind_t::literal, value, max);
    }

    /// Shorthand for `literal(0, max)`
    ///
    /// @param max The ceiling on the delay, -1 for none
    static policy_t zero(int max = -1) {
        return policy_t(kind_t::literal, 0, max);
    }
};

/// Source of the `delay` staggering consumer calls behind producer iterations
///
/// @details The ideal distance follows the producer instruction latency
///
/// @pre The caller must validate `delay` candidates to ensure code correctness
struct scheduler_t {
    int mem_latency = 0; // Latency of producer memory instructions
    int aop_latency = 0; // Latency of producer arithmetic instructions

    /// Obtains the `delay` value according to a `policy`
    ///
    /// @param policy The policy
    int operator()(const policy_t &policy) const {
        int out = 0;
        switch (policy.kind) {
            case policy_t::kind_t::memory: out = mem_latency; break;
            case policy_t::kind_t::arithmetic: out = aop_latency; break;
            case policy_t::kind_t::literal: out = policy.value; break;
        }
        return (policy.max < 0) ? out : std::min(policy.max, out);
    }

    /// Builds a scheduler from producer memory/arithmetic instruction latencies
    ///
    /// @param mem_latency The latency of producer memory instructions
    /// @param aop_latency The latency of producer arithmetic instructions
    static scheduler_t from_params(int mem_latency, int aop_latency) {
        scheduler_t s;
        s.mem_latency = mem_latency;
        s.aop_latency = aop_latency;
        return s;
    }
};

} // namespace stream
} // namespace rvjit

#endif
