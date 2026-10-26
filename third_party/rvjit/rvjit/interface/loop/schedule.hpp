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

#ifndef RVJIT_INTERFACE_LOOP_SCHEDULE_HPP
#define RVJIT_INTERFACE_LOOP_SCHEDULE_HPP

#include <algorithm>
#include <cstdint>

#include "rvjit/interface/isa/parameter.hpp"
#include "rvjit/interface/loop/plan.hpp"
#include "rvjit/interface/loop/unroll.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace loop {

/// How a loop dispatches its iterations, as a pure function of its `plan_t`
///
/// @details What the plan's size is known to be selects the shape: an exact
///     size leaves at most one tail, an upper bound licenses a single block
///     or a `switch_`, and an unbounded size keeps a main stage repeating
///     past the unroll factor. Needs no emitter or registers
struct schedule_t {
    /// How the dispatches are laid out
    enum class mode_t {
        literal, // Fully unrolled by constant factors
        switch_, // Fully unrolled by a dynamic factor <= N
        unroll, // Unrolled by descending factors
        unroll_switch, // `unroll`, tail dispatched by `switch_`
        unroll_literal, // `unroll`, tail a single literal call
    };

    mode_t mode = mode_t::unroll;

    /// The unrolling factors, in units of `grain`
    unroll_t factors;

    /// Work-units one factor unit covers
    int grain = 1;

    /// The most work-units the loop is entered with, 0 when unknown
    int bound = 0;

    /// Whether the factors are valid
    bool is_resolved() const { return factors.is_valid() && grain > 0; }

    /// The most dispatches a counted stage makes
    ///
    /// @param ur The stage's unroll factor
    /// @param grain The work-units one factor unit covers
    /// @param rem The most work-units the stage is entered with
    ///
    /// @note A one-unit stage dispatches once per granule left, whereas a wider
    ///     one only while a whole dispatch fits
    static int64_t stage_trips(int ur, int grain, int64_t rem) {
        if (ur == 1) return (rem + grain - 1) / grain;
        return rem / (static_cast<int64_t>(ur) * grain);
    }

    /// The most work-units a counted stage leaves behind
    ///
    /// @param ur The stage's unroll factor
    /// @param grain The work-units one factor unit covers
    /// @param rem The most work-units the stage is entered with
    static int64_t stage_exit(int ur, int grain, int64_t rem) {
        if (ur == 1) return 0;
        return std::min(rem, static_cast<int64_t>(ur) * grain - 1);
    }

    /// Wraps `value`, then the power-of-two breakdown below it, into an
    /// `unroll_t`
    ///
    /// @param value The largest factor
    ///
    /// @note A remainder below `value` is covered by the breakdown tails
    static unroll_t pow2_unroll(int value) {
        int buf[8];
        int n = 0;
        if (value > 0 && !is_pow2(value)) buf[n++] = value;
        n += pow2_breakdown(value, buf + n, 8 - n);
        return unroll_t(buf, n);
    }

    /// Resolves the cheapest valid schedule for `p`
    ///
    /// @param p The plan
    ///
    /// @note The size's source is not read: only whether it is a literal, a
    ///     bound, or neither
    static schedule_t of(const plan_t &p) {
        using lkind_t = plan_t::kind_t;

        schedule_t s;
        const size_parameter_t &sp = p.size;
        const int max_ur = std::max(p.max_ur, 1);
        const int granularity = p.granularity;
        s.grain = granularity;

        // Selects `m`, with `u` as its unrolling when valid
        const auto make = [&](mode_t m, const unroll_t &u) {
            s.mode = m;
            if (u.is_valid()) s.factors = u;
        };

        // A literal size; when `granularity` does not divide it, `pow2_adapt`
        // picks `pow2_unroll(max_ur)` over `{max_ur, 1}`
        const auto literal = [&](int size, bool pow2_adapt) {
            const bool aligned = granularity <= 1 || size % granularity == 0;
            const int units = granularity > 1 ? size / granularity : size;
            if (!aligned) {
                const unroll_t pair
                        = max_ur > 1 ? unroll_t {max_ur, 1} : unroll_t(1);
                make(mode_t::unroll, pow2_adapt ? pow2_unroll(max_ur) : pair);
            } else if (units <= max_ur) {
                make(mode_t::literal, unroll_t(units));
            } else if (units < 2 * max_ur) {
                // The main stage would run once: dispatch it, then the tail
                make(mode_t::literal, unroll_t {max_ur, units - max_ur});
            } else {
                const int tail = units % max_ur;
                make(mode_t::unroll_literal,
                        tail == 0 ? unroll_t(max_ur) : unroll_t {max_ur, tail});
            }
        };

        // Whatever bounds the size bounds the trips of its counted stages
        s.bound = sp.bound();

        // The granules a bounded size spans
        const int units = granularity > 1
                ? (sp.bound() + granularity - 1) / granularity
                : sp.bound();

        if (p.kind == lkind_t::reduce) {
            if (sp.is_literal())
                literal(sp.value(), false);
            else if (sp.is_less_eq())
                make(mode_t::unroll,
                        pow2_unroll(std::max(std::min(max_ur, units), 1)));
            else
                make(mode_t::unroll, pow2_unroll(max_ur));
            return s;
        }

        if (sp.is_literal()) {
            literal(sp.value(), true);
        } else if (sp.is_less_eq()) {
            // A coarser granularity ends in a one-granule stage covering any
            // runtime tail
            if (granularity > 1)
                make(mode_t::unroll,
                        pow2_unroll(std::max(std::min(max_ur, units), 1)));
            else if (sp.bound() <= max_ur)
                make(mode_t::switch_, unroll_t(std::max(sp.bound(), 1)));
            else
                make(mode_t::unroll_switch, unroll_t(max_ur));
        } else if (granularity > 1) {
            make(mode_t::unroll, pow2_unroll(max_ur));
        } else if (max_ur > 1) {
            make(mode_t::unroll_switch, unroll_t(max_ur));
        } else {
            make(mode_t::unroll, unroll_t(1));
        }
        return s;
    }
};

} // namespace loop
} // namespace rvjit

#endif
