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

#ifndef RVJIT_INTERFACE_STREAM_ITERATOR_STRATEGY_HPP
#define RVJIT_INTERFACE_STREAM_ITERATOR_STRATEGY_HPP

#include <array>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/stream/iterator/plan.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace iterator {

/// The simm12 offset window shared by consecutive axes of one plan
///
/// @note `group_off` is the offset accumulated since the last base register;
///     `reserve` is headroom a later axis needs on both ends of its window
struct window_t {
    int group_off = 0;
    int reserve = 0;

    window_t() = default;

    /// Creates a window
    ///
    /// @param group_off The offset accumulated since the last base register
    /// @param reserve The headroom a later axis needs on both ends
    window_t(int group_off, int reserve)
        : group_off(group_off), reserve(reserve) {}
};

/// Resources a strategy may claim while an iterator plan is resolved
struct budget_t {
    /// Integer registers still available
    int remaining = 0;

    /// Registers-hungry axes (needing more than one register) already claimed
    int multi = 0;

    /// Whether a claim must fit `remaining`; a last-resort fallback is
    /// unchecked, leaving the shortfall to register allocation
    bool strict = true;

    /// Whether a claim of `need` registers is within the budget
    ///
    /// @param need The registers claimed
    bool fits(int need) const { return !strict || need <= remaining; }
};

/// What a strategy resolved for the axes starting at the one it was asked about
struct claim_t {
    enum class status_t {
        /// Axes `[d, last_axis]` were resolved
        fit,

        /// Supported, but not within the budget or the strategy's own rules
        misfit,

        /// The strategy cannot service the axis at all
        unsupported,

        /// Deliberately declined for a structural reason; a fallback strategy
        /// must then fit its budget
        withheld,
    };

    status_t status = status_t::unsupported;

    /// Last axis covered (the claim covers `[d, last_axis]`)
    int last_axis = -1;

    /// Registers consumed from the budget
    int regs = 0;

    /// Whether the claim consumed a registers-hungry slot
    bool multi = false;

    /// The shared window after this claim
    window_t window;

    /// Whether axes `[d, last_axis]` were resolved
    bool is_fit() const { return status == status_t::fit; }
};

/// How one or more axes of a pointer iteration space are addressed
///
/// @details An implementation states what it can service (`supports`), what
///     resolving an axis costs (`claim`), and how it emits (`walk`, `step`,
///     `init`, `finish`). Which strategy serves which axis, and the register
///     budget they share, are decided by the caller of `claim`
///
/// @note Implementations are stateless, so a plan resolved by one instance is
///     emitted correctly by another
struct strategy_t {
    static constexpr const char *NAME = "rvjit::stream::iterator";

    virtual ~strategy_t() = default;

    /// Short name used in diagnostics
    virtual const char *name() const { return "strategy"; }

    /// Whether the strategy can service `axis` under `addressing` at all
    ///
    /// @param axis The axis
    /// @param addressing How the memory operation addresses elements
    virtual bool supports(const axis_t &axis, addressing_t addressing) const {
        (void)axis;
        (void)addressing;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Resolves axis `d` (and any later axes it absorbs) of `request`
    ///
    /// @param request What the caller asks of the iterator
    /// @param d The axis resolved first
    /// @param budget The resources still available
    /// @param window The simm12 offset window shared with the previous axes
    /// @param out The plan receiving the resolved fields of every covered axis
    ///
    /// @note `out` is written only when the returned claim is a fit
    /// @note A fit claims the same registers under any budget it fits
    virtual claim_t claim(const request_t &request, int d,
            const budget_t &budget, const window_t &window, plan_t &out) const {
        (void)request;
        (void)d;
        (void)budget;
        (void)window;
        (void)out;
        LOG_FAIL(__func__, "unimplemented");
        return claim_t();
    }

    /// Contributes axis `d`'s own base/offset to `out` when reading `idx`
    ///
    /// @param out The pointer accumulating the contribution of each axis
    /// @param plan The resolved plan
    /// @param d The axis
    /// @param idx The position read
    virtual void walk(pointer_t &out, const plan_t &plan, int d,
            const index_t &idx) const {
        (void)out;
        (void)plan;
        (void)d;
        (void)idx;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// Emits axis `d`'s step when `idx` advances along it
    ///
    /// @param arith The constant arithmetic component
    /// @param plan The resolved plan
    /// @param d The axis
    /// @param prev The pointer resolved through axes `[0, d]` at `idx`
    /// @param idx The position
    ///
    /// @note Emits nothing by default
    virtual void step(const operation::const_arithmetic_t &arith,
            const plan_t &plan, int d, const pointer_t &prev,
            const index_t &idx) const {
        (void)arith;
        (void)plan;
        (void)d;
        (void)prev;
        (void)idx;
    }

    /// Emits axis `d`'s setup at `idx`, the first time its inner axes are at 0
    ///
    /// @param e The emitter instructions are emitted through
    /// @param arith The constant arithmetic component
    /// @param xfer The value transfer component
    /// @param plan The resolved plan
    /// @param d The axis
    /// @param idx The position
    /// @param walked The pointer resolved through axes `[0, d)`
    ///
    /// @note Emits nothing by default
    virtual void init(emitter_t e, const operation::const_arithmetic_t &arith,
            const operation::transfer_t &xfer, const plan_t &plan, int d,
            const index_t &idx, const pointer_t &walked) const {
        (void)e;
        (void)arith;
        (void)xfer;
        (void)plan;
        (void)d;
        (void)idx;
        (void)walked;
    }

    /// Emits the plan's end-of-space step when the end axis is `end_axis`
    ///
    /// @param arith The constant arithmetic component
    /// @param plan The resolved plan
    /// @param end_axis The axis whose whole span the pointer moves past
    virtual void finish(const operation::const_arithmetic_t &arith,
            const plan_t &plan, int end_axis) const {
        (void)arith;
        (void)plan;
        (void)end_axis;
        LOG_FAIL(__func__, "unimplemented");
    }
};

/// An ordered set of strategies; the order is the resolver's preference
///
/// @details The order is also the cost ladder of an ISA: the first strategy
///     yields the best streams and may cost the most registers, and the last
///     is the cost floor, which cannot be made cheaper
///
/// @note Holds non-owning references; the strategies must outlive the set
struct strategy_set_t {
    static constexpr int CAPACITY = 4;

    /// Appends a strategy and returns its id; invalid when the set is full
    ///
    /// @param s The strategy, which must outlive the set
    strategy_id_t add(const strategy_t &s) {
        if (n_ >= CAPACITY) return strategy_id_t::any();
        e_[n_] = &s;
        return strategy_id_t(n_++);
    }

    /// Number of strategies
    int size() const { return n_; }

    /// The strategy named by `id`
    ///
    /// @param id The id, which the set `has()`
    const strategy_t &operator[](strategy_id_t id) const {
        return *e_[id.id()];
    }

    /// Whether `id` names a strategy of the set
    ///
    /// @param id The id
    bool has(strategy_id_t id) const { return !id.is_any() && id.id() < n_; }

private:
    std::array<const strategy_t *, CAPACITY> e_ {};
    int n_ = 0;
};

} // namespace iterator
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
