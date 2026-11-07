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

#ifndef RVJIT_INTERFACE_STREAM_COMBINATOR_HPP
#define RVJIT_INTERFACE_STREAM_COMBINATOR_HPP

#include <initializer_list>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/stream/scheduler.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"

namespace rvjit {
namespace stream {

/// Combinators for instruction streams
///
/// @details Provides `interleaved`/`nested`, staggering callback calls
///     relative to each stream's iterations (short-range prefetching)
template <typename InterleaveOp, typename NestOp>
struct combinator_t {
    combinator_t() = default;

    /// Creates a combinator resolving delays against `sched`
    ///
    /// @param sched The scheduler
    explicit combinator_t(const scheduler_t &sched)
        : interleave_(sched), nest_(sched) {}

    /// Interleaves the iterations of streams against `cb` and emits
    ///
    /// @param pol The policy resolving the delay of `cb`
    /// @param cb The callback fired once per combined position
    /// @param entries The streams, walked over the per-axis maximum shape; one
    ///     is enough, staggered against `cb`
    ///
    /// @pre Every stream shares the same number of dimensions
    void interleaved(const policy_t &pol, const on_interleave_t &cb,
            std::initializer_list<ref_t> entries) const {
        interleave_(pol, cb, entries);
    }

    /// Interleaves two or more streams without a callback and emits
    ///
    /// @param entries The streams
    void interleaved(std::initializer_list<ref_t> entries) const {
        interleave_(entries);
    }

    /// Nests two streams as a cartesian product with `ns` shared outer axes
    /// and emits
    ///
    /// @param ns The number of shared outer axes
    /// @param pol The policy resolving the delay of `cb`
    /// @param s1 The stream whose own axes vary fastest
    /// @param s2 The stream whose own axes are reused per shared position
    /// @param cb The callback fired once per position
    ///
    /// @pre s1.size(d) == s2.size(d) for every shared dim `d < ns`
    /// @pre The storage of `s1`/`s2` outlives the deferred `cb` calls (e.g.
    ///     a deep enough ring buffer)
    void nested(int ns, const policy_t &pol, ref_t s1, ref_t s2,
            const on_nest_t &cb) const {
        nest_(ns, pol, s1, s2, cb);
    }

    /// Nests two streams sharing exactly one outer axis and emits
    ///
    /// @param pol The policy resolving the delay of `cb`
    /// @param s1 The stream whose own axes vary fastest
    /// @param s2 The stream whose own axes are reused per shared position
    /// @param cb The callback fired once per position
    void nested_1d(const policy_t &pol, ref_t s1, ref_t s2,
            const on_nest_t &cb) const {
        nested(1, pol, s1, s2, cb);
    }

    /// Largest delay `nested` tolerates on `cb` before a deferred call finds
    /// a register of `s1`/`s2` overwritten
    ///
    /// @param ns The number of shared outer axes
    /// @param s1 The stream whose own axes vary fastest
    /// @param s2 The stream whose own axes are reused per shared position
    int max_nested_delay(int ns, ref_t s1, ref_t s2) const {
        return nest_.max_delay(ns, s1, s2);
    }

protected:
    InterleaveOp interleave_;
    NestOp nest_;
};

} // namespace stream
} // namespace rvjit

#endif
