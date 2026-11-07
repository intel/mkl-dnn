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

#ifndef RVJIT_IMPLEMENTATION_COMMON_STREAM_NEST_HPP
#define RVJIT_IMPLEMENTATION_COMMON_STREAM_NEST_HPP

#include <algorithm>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/stream/scheduler.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace common {
namespace stream {

/// Nests two streams using a cartesian product with NS shared outer axes
struct nest_t {
    static constexpr const char *NAME = "rvjit::common::stream";

    nest_t() = default;

    /// Creates a nest resolving delays against `sched`
    ///
    /// @param sched The scheduler
    explicit nest_t(const rvjit::stream::scheduler_t &sched) : sched_(sched) {}

    /// Emits the nest of `s1` and `s2`, walking every shared position in
    /// ascending order
    ///
    /// @param ns The number of shared outer axes
    /// @param pol The policy resolving the delay of `cb`
    /// @param s1 The stream whose own axes vary fastest
    /// @param s2 The stream whose own axes are reused per shared position
    /// @param cb The callback fired once per position
    ///
    /// @note `cb` is deferred by `delay` across the whole nest and every call
    ///     not yet fired is flushed on the last position, so a large `delay`
    ///     can defer `cb` into an earlier shared position
    ///
    /// @pre s1.size(d) == s2.size(d) for every shared dim `d < ns`
    void operator()(int ns, const rvjit::stream::policy_t &pol,
            rvjit::stream::ref_t s1, rvjit::stream::ref_t s2,
            const rvjit::stream::on_nest_t &cb) const {
        CHECK_OR_RET(ns >= 0 && ns <= s1.ndims() && ns <= s2.ndims(), ,
                "shared axes exceeding either stream's own axes");
        for (int d = 0; d < ns; ++d)
            CHECK_OR_RET(s1.size(d) == s2.size(d), ,
                    "shared axis %d of different sizes", d);

        const geometry_t g(ns, s1, s2);
        const int delay = sched_(pol);
        if (g.shared_size == 0) return;

        const int total = g.own1 * g.own2; // positions per shared row
        const int grand_total = g.shared_size * total;

        // Fires cb for global position p, possibly in an earlier shared row
        const auto fire_at = [&](int p) {
            const rvjit::index_t k = g.shared_at(p / total);
            const int mn = p % total;
            cb(g.join(k, g.own1_nd, g.own1_shape, mn / g.own2),
                    g.join(k, g.own2_nd, g.own2_shape, mn % g.own2));
        };

        for (int kf = 0; kf < g.shared_size; ++kf) {
            const rvjit::index_t k = g.shared_at(kf);
            for (int mn = 0; mn < total; ++mn) {
                const int m = mn / g.own2;
                const int n = mn % g.own2;

                // s2 is produced on the first sweep of the row, reused after
                if (m == 0) s2.iteration(g.join(k, g.own2_nd, g.own2_shape, n));

                // s1 advances once per full sweep over s2
                if (n == 0) s1.iteration(g.join(k, g.own1_nd, g.own1_shape, m));

                // Fire the deferred call, or flush the rest on the last one
                const int pos = kf * total + mn;
                if (pos < grand_total - 1) {
                    const int p = pos - delay;
                    if (p >= 0) fire_at(p);
                } else {
                    for (int p = std::max(0, pos - delay); p < grand_total; ++p)
                        fire_at(p);
                }
            }
        }
    }

    /// Largest `cb` delay the registers of `s1`/`s2` tolerate
    ///
    /// @param ns The number of shared outer axes
    /// @param s1 The stream whose own axes vary fastest
    /// @param s2 The stream whose own axes are reused per shared position
    ///
    /// @note Replays the schedule of `operator()`, modeling register
    ///     overwrites only and not instruction latency
    ///
    /// @pre `get(idx)` names the register holding the position `idx` at the
    ///     streams' current shapes, and `iteration(idx)` writes only that
    ///     register
    /// @pre The registers `cb` reads are written only by the productions of
    ///     `s1` and `s2`, and `cb` reads them through `get()`
    /// @pre The shapes are the ones `operator()` will run at
    int max_delay(
            int ns, rvjit::stream::ref_t s1, rvjit::stream::ref_t s2) const {
        CHECK_OR_RET(ns >= 0 && ns <= s1.ndims() && ns <= s2.ndims(), 0,
                "shared axes exceeding either stream's own axes");

        const geometry_t g(ns, s1, s2);
        const int total = g.own1 * g.own2;
        const int grand_total = g.shared_size * total;
        if (grand_total == 0) return 0;

        // Whether every deferred call finds its operands in place
        const auto safe = [&](int delay) {
            // Track which position each register holds as the nest produces
            static constexpr int REGS = 64;
            int held1[REGS], held2[REGS];
            std::fill(held1, held1 + REGS, -1);
            std::fill(held2, held2 + REGS, -1);

            const auto intact = [&](int p) {
                const int kf = p / total;
                const rvjit::index_t k = g.shared_at(kf);
                const int mn = p % total;
                const int m = mn / g.own2;
                const int n = mn % g.own2;
                const int r1 = s1.get(g.join(k, g.own1_nd, g.own1_shape, m));
                const int r2 = s2.get(g.join(k, g.own2_nd, g.own2_shape, n));
                return held1[r1] == kf * g.own1 + m
                        && held2[r2] == kf * g.own2 + n;
            };

            // Replay the production schedule, checking each deferred call
            for (int kf = 0; kf < g.shared_size; ++kf) {
                const rvjit::index_t k = g.shared_at(kf);
                for (int mn = 0; mn < total; ++mn) {
                    const int m = mn / g.own2;
                    const int n = mn % g.own2;
                    const int r1
                            = s1.get(g.join(k, g.own1_nd, g.own1_shape, m));
                    const int r2
                            = s2.get(g.join(k, g.own2_nd, g.own2_shape, n));
                    if (m == 0) held2[r2] = kf * g.own2 + n;
                    if (n == 0) held1[r1] = kf * g.own1 + m;

                    const int pos = kf * total + mn;
                    if (pos < grand_total - 1) {
                        const int p = pos - delay;
                        if (p >= 0 && !intact(p)) return false;
                    } else {
                        for (int p = std::max(0, pos - delay); p < grand_total;
                                ++p)
                            if (!intact(p)) return false;
                    }
                }
            }
            return true;
        };

        // Overwrites only accumulate with the delay, so bisect
        int lo = 0, hi = grand_total;
        while (lo < hi) {
            const int mid = (lo + hi + 1) / 2;
            if (safe(mid))
                lo = mid;
            else
                hi = mid - 1;
        }
        return lo;
    }

private:
    // Axes and sizes of the shared and own parts of a nest
    struct geometry_t {
        int ns, own1_nd, own2_nd;
        rvjit::index_t shared_shape {}, own1_shape {}, own2_shape {};
        int shared_size = 1, own1 = 1, own2 = 1;

        geometry_t(int ns, rvjit::stream::ref_t s1, rvjit::stream::ref_t s2)
            : ns(ns), own1_nd(s1.ndims() - ns), own2_nd(s2.ndims() - ns) {
            for (int d = 0; d < ns; ++d) {
                shared_shape[d] = s1.size(d);
                shared_size *= shared_shape[d];
            }
            for (int d = 0; d < own1_nd; ++d) {
                own1_shape[d] = s1.size(ns + d);
                own1 *= own1_shape[d];
            }
            for (int d = 0; d < own2_nd; ++d) {
                own2_shape[d] = s2.size(ns + d);
                own2 *= own2_shape[d];
            }
        }

        // The shared position of flat index `flat`
        rvjit::index_t shared_at(int flat) const {
            return rvjit::stream::unravel(flat, ns, shared_shape);
        }

        // Joins a shared position with a flat own position
        rvjit::index_t join(const rvjit::index_t &shared, int own_nd,
                const rvjit::index_t &own_shape, int flat) const {
            rvjit::index_t r {};
            for (int d = 0; d < ns; ++d)
                r[d] = shared[d];
            const rvjit::index_t own
                    = rvjit::stream::unravel(flat, own_nd, own_shape);
            for (int d = 0; d < own_nd; ++d)
                r[ns + d] = own[d];
            return r;
        }
    };

    rvjit::stream::scheduler_t sched_;
};

} // namespace stream
} // namespace common
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
