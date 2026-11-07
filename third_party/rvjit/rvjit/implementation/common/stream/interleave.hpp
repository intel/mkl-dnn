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

#ifndef RVJIT_IMPLEMENTATION_COMMON_STREAM_INTERLEAVE_HPP
#define RVJIT_IMPLEMENTATION_COMMON_STREAM_INTERLEAVE_HPP

#include <algorithm>
#include <initializer_list>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/stream/scheduler.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace common {
namespace stream {

/// Interleaves two or more streams' iterations against one shared step
/// counter
struct interleave_t {
    static constexpr const char *NAME = "rvjit::common::stream";

    interleave_t() = default;

    /// Creates an interleaving resolving delays against `sched`
    ///
    /// @param sched The scheduler
    explicit interleave_t(const rvjit::stream::scheduler_t &sched)
        : sched_(sched) {}

    /// Emits the interleaving of `entries`
    ///
    /// @param pol The policy resolving the delay of `cb`
    /// @param cb The callback fired once per combined position
    /// @param entries The streams, walked over the per-axis maximum shape; one
    ///     is enough, staggered against `cb`
    ///
    /// @note Entries fire unshifted each step, while `cb` is deferred by the
    ///     delay and flushed on the last step
    ///
    /// @pre Every entry shares the same number of dimensions
    void operator()(const rvjit::stream::policy_t &pol,
            const rvjit::stream::on_interleave_t &cb,
            std::initializer_list<rvjit::stream::ref_t> entries) const {
        emit_(sched_(pol), cb, entries);
    }

    /// Emits the interleaving of `entries` with no combined callback
    ///
    /// @param entries The streams, walked over the per-axis maximum shape
    ///
    /// @pre Every entry shares the same number of dimensions
    void operator()(std::initializer_list<rvjit::stream::ref_t> entries) const {
        emit_(0, nullptr, entries);
    }

private:
    void emit_(int delay, const rvjit::stream::on_interleave_t &cb,
            std::initializer_list<rvjit::stream::ref_t> entries) const {
        // Check that the entries can be interleaved: a single one only against
        // the callback
        const size_t least = cb ? 1 : 2;
        CHECK_OR_RET(entries.size() >= least, , "too few streams");
        const int nd = entries.begin()->ndims();
        for (const rvjit::stream::ref_t &e : entries)
            CHECK_OR_RET(
                    e.ndims() == nd, , "every entry must share the same ndims");

        // The shape is the per-axis max across every entry
        rvjit::index_t shape {};
        for (const rvjit::stream::ref_t &e : entries)
            for (int d = 0; d < nd; ++d)
                shape[d] = std::max(shape[d], e.size(d));
        int steps = 1;
        for (int d = 0; d < nd; ++d)
            steps *= shape[d];

        // Walk the steps, firing each entry and then the deferred callback
        for (int t = 0; t < steps; ++t) {
            // Every entry fires at its own natural position, if it has one
            for (const rvjit::stream::ref_t &e : entries) {
                rvjit::index_t own_shape {};
                int flat = 1;
                for (int d = 0; d < nd; ++d) {
                    own_shape[d] = e.size(d);
                    flat *= own_shape[d];
                }
                if (t < flat)
                    e.iteration(rvjit::stream::unravel(t, nd, own_shape));
            }

            // Fire the deferred call, or flush the rest on the last step
            if (!cb) continue;
            if (t < steps - 1) {
                const int p = t - delay;
                if (p >= 0) cb(rvjit::stream::unravel(p, nd, shape));
            } else {
                for (int p = std::max(0, t - delay); p < steps; ++p)
                    cb(rvjit::stream::unravel(p, nd, shape));
            }
        }
    }

    rvjit::stream::scheduler_t sched_;
};

} // namespace stream
} // namespace common
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
