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

#ifndef RVJIT_INTERFACE_STREAM_ITERATOR_ITERATOR_HPP
#define RVJIT_INTERFACE_STREAM_ITERATOR_ITERATOR_HPP

#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/resource/resource_optimizable.hpp"
#include "rvjit/interface/stream/iterator/plan.hpp"
#include "rvjit/interface/stream/iterator/strategy.hpp"
#include "rvjit/interface/stream/iterator/types.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"
#include "rvjit/interface/stream/stream_table.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace iterator {

/// A stream of pointers, emitted through the strategies each axis resolved to
///
/// @note Changes in place; `resize` narrows or widens up to the shape of the
///     plan it was initialized with
struct iterator_impl_t : public stream_impl_t {
    static constexpr const char *NAME = "rvjit::stream::iterator";

    /// Creates an unconfigured iterator
    ///
    /// @param e The emitter instructions are emitted through
    /// @param arith The constant arithmetic component
    /// @param xfer The value transfer component
    /// @param strategies The strategies the plan axes resolved to, which must
    ///     outlive the iterator
    iterator_impl_t(emitter_t e, const operation::const_arithmetic_t &arith,
            const operation::transfer_t &xfer, const strategy_set_t &strategies)
        : e_(e), arith_(&arith), xfer_(&xfer), strategies_(&strategies) {}

    /// An iterator that is never valid
    iterator_impl_t() = default;

    /// The stream handed out in place of one that failed to be created
    static iterator_impl_t *invalid() {
        static iterator_impl_t none;
        return &none;
    }

    /// Configures the iterator from a resolved plan
    ///
    /// @param plan The resolved plan
    ///
    /// @note Fails if the plan is invalid or unresolved
    bool init(const plan_t &plan) {
        CHECK(plan.is_valid() && !plan.needs_regs(),
                "unresolved or invalid iterator plan");
        full_ = plan;
        current_ = plan;

        // Bookkeeping only; addressing still goes through the plan
        regs_ = buffer::buffer_t();
        for (int d = 0; d < plan.ndims(); ++d)
            regs_ = regs_ | plan.axis(d).regs;
        valid_ = true;
        return true;
    }

    /// The pointer at position `idx`
    ///
    /// @param idx The position
    pointer_t pointer(const index_t &idx) const {
        CHECK_OR_RET(valid_, pointer_t(), "uninitialized iterator");
        return walk_(current_, idx, current_.ndims());
    }

    int ndims() const override {
        CHECK_OR_RET(valid_, 0, "uninitialized iterator");
        return current_.ndims();
    }

    int size(int d) const override {
        CHECK_OR_RET(valid_, 0, "uninitialized iterator");
        return current_.space.axis(d).size;
    }

    bool is_valid() const override { return valid_; }

    /// @pre Every axis of `shape` is within the shape the plan was
    ///     initialized with
    bool resize(const index_t &shape) override {
        CHECK(valid_, "uninitialized iterator");
        for (int d = 0; d < full_.ndims(); ++d)
            CHECK(shape[d] > 0 && shape[d] <= full_.space.axis(d).size,
                    "an axis size outside the registered shape");
        current_ = full_.resized(shape);
        return true;
    }

    const buffer::buffer_t &buffer() const override {
        static const buffer::buffer_t none;
        CHECK_OR_RET(valid_, none, "uninitialized iterator");
        return regs_;
    }

protected:
    /// Emits this position's pointer advance and the plan's end-state step
    ///
    /// @pre Caller issues idx in ascending nested order
    void produce_(const index_t &idx) override {
        const plan_t &plan = current_;
        const int end_axis = plan.update.axis();

        // The innermost axis with room to advance increments at idx+1, like
        // an odometer digit; every axis after it rolls back to 0
        int advancing_axis = -1;
        for (int d = plan.ndims() - 1; d >= 0 && advancing_axis < 0; --d)
            if (idx[d] + 1 < plan.space.axis(d).size) advancing_axis = d;

        // The globally last cell lands the plan's advance-state step
        if (advancing_axis < 0) {
            if (plan.update.kind() == update_t::kind_t::preserve) return;
            strategy_(plan, end_axis).finish(*arith_, plan, end_axis);
            return;
        }

        // Walk through the advancing axis itself so `prev` holds its current
        // register, not its parent's (they differ past index 0)
        const pointer_t prev = walk_(plan, idx, advancing_axis + 1);
        strategy_(plan, advancing_axis)
                .step(*arith_, plan, advancing_axis, prev, idx);
    }

    /// Emits this position's per-axis setup
    void init_(const index_t &idx) override {
        const plan_t &plan = current_;
        for (int d = 0; d < plan.ndims(); ++d) {
            // No registers to build (e.g. a whole axis fitting as an immediate)
            if (plan.axis(d).nregs() == 0) continue;

            // Axis d's setup runs when every inner axis is still at its
            // start; the strategy picks the outer and own positions it needs
            bool first_visit = true;
            for (int dd = d + 1; dd < plan.ndims(); ++dd)
                first_visit = first_visit && idx[dd] == 0;
            if (!first_visit) continue;

            const pointer_t walked = walk_(plan, idx, d);
            strategy_(plan, d).init(e_, *arith_, *xfer_, plan, d, idx, walked);
        }
    }

private:
    const strategy_t &strategy_(const plan_t &plan, int d) const {
        return (*strategies_)[plan.axis(d).strategy];
    }

    /// Computes the register/offset representing axes `[0, upto_d)` at `idx`
    pointer_t walk_(const plan_t &plan, const index_t &idx, int upto_d) const {
        pointer_t out = plan.origin;
        for (int d = 0; d < upto_d; ++d)
            strategy_(plan, d).walk(out, plan, d, idx);
        return out;
    }

    emitter_t e_;
    const operation::const_arithmetic_t *arith_ = nullptr;
    const operation::transfer_t *xfer_ = nullptr;
    const strategy_set_t *strategies_ = nullptr;
    plan_t full_;
    plan_t current_;
    buffer::buffer_t regs_;
    bool valid_ = false;
};

/// Plans pointer iterators over multi-dimensional spaces and creates the
/// iterator streams emitting them
///
/// @details `resolve` decides, axis by axis, which strategy of an ordered
///     strategy set implements a request and how the register budget is split;
///     `iterator` creates an stream of pointers from a resolved plan
///
/// @note Transactions reclaim the iterators created since `begin()`
struct iterator_t : public stream_table_t<iterator_impl_t, 16> {
    static constexpr const char *NAME = "rvjit::stream::iterator";

    /// Creates an iterator factory
    ///
    /// @param e The emitter instructions are emitted through
    /// @param arith The constant arithmetic component
    /// @param xfer The value transfer component
    /// @param pool The pool the iterator registers are drawn from
    /// @param strategies The strategies requests resolve over, which must
    ///     outlive the factory
    iterator_t(emitter_t e, const operation::const_arithmetic_t &arith,
            const operation::transfer_t &xfer, buffer::buffer_pool_t &pool,
            const strategy_set_t &strategies)
        : stream_table_t<iterator_impl_t, 16>(e, arith, xfer, strategies)
        , proto_(e, arith, xfer, strategies)
        , strategies_(strategies)
        , pool_(pool) {}

    /// The strategies this factory resolves requests over
    const strategy_set_t &strategies() const { return strategies_; }

    /// An unconfigured iterator of this factory, owned by the caller
    ///
    /// @note It takes no slot, so transactions neither take nor reclaim it
    iterator_impl_t detached() const { return proto_; }

    /// Resolves a strategy and register cost for every axis of the request
    ///
    /// @param set The strategies tried, in preference order
    /// @param request What the caller asks of the iterator
    /// @param ntmp The integer registers available
    ///
    /// @details An explicit hint is the only strategy tried; `any` tries each
    ///     strategy of `set` in order, and the last one is not budget-checked
    ///     unless an earlier one withheld the axis
    ///
    /// @note Static: needs no emitter or pool state
    static plan_t resolve(
            const strategy_set_t &set, const request_t &request, int ntmp) {
        // Reject a request that cannot be resolved
        CHECK_OR_RET(
                request.origin.is_valid(), plan_t(), "invalid origin pointer");
        CHECK_OR_RET(
                request.space.is_valid(), plan_t(), "invalid tensor space");
        CHECK_OR_RET(
                ntmp >= 0, plan_t(), "invalid number of available temporaries");
        CHECK_OR_RET(request.update.axis() >= 0
                        && request.update.axis() < request.space.ndims(),
                plan_t(), "out-of-range end-state axis");

        // Resolve the axes in order, sharing one budget and offset window
        plan_t plan(request);
        budget_t budget;
        budget.remaining = ntmp;
        window_t window;
        window.group_off = request.origin.offset;

        for (int d = 0; d < request.space.ndims();) {
            const auto &ax = request.space.axis(d);
            const strategy_id_t hint = request.strategy;
            claim_t claim;
            strategy_id_t id;

            // The axis claimed carries its candidate's id, so a claim tells
            // which earlier axes it resolved itself
            if (!hint.is_any()) {
                plan.axis(d).strategy = hint;
                // An explicit hint is the only strategy tried
                CHECK_OR_RET(set.has(hint), plan_t(),
                        "dim %d's unknown strategy %d", d, hint.id());
                const strategy_t &s = set[hint];
                CHECK_OR_RET(s.supports(ax, request.addressing), plan_t(),
                        "dim %d's explicit %s request over a stride it cannot "
                        "service",
                        d, s.name());
                claim = s.claim(request, d, budget, window, plan);
                CHECK_OR_RET(claim.is_fit(), plan_t(),
                        "dim %d's explicit %s request needing more resources "
                        "than available or unsupported by its own rules",
                        d, s.name());
                id = hint;
            } else {
                // `any` takes the first strategy that fits; every one but the
                // last is budget-checked, and so is the last after a withhold
                bool withheld = false;
                bool fit = false;
                for (int i = 0; i < set.size() && !fit; ++i) {
                    const strategy_t &s = set[strategy_id_t(i)];
                    if (!s.supports(ax, request.addressing)) continue;
                    budget_t b = budget;
                    b.strict = withheld || i + 1 < set.size();
                    plan.axis(d).strategy = strategy_id_t(i);
                    claim = s.claim(request, d, b, window, plan);
                    fit = claim.is_fit();
                    if (fit) id = strategy_id_t(i);
                    withheld = withheld
                            || claim.status == claim_t::status_t::withheld;
                }
                CHECK_OR_RET(
                        fit, plan_t(), "dim %d resolved by no strategy", d);
            }

            // Commit the claim and move past the axes it covered
            for (int i = d; i <= claim.last_axis; ++i)
                plan.axis(i).strategy = id;
            budget.remaining -= claim.regs;
            if (claim.multi) ++budget.multi;
            window = claim.window;
            d = claim.last_axis + 1;
        }
        return plan;
    }

    /// Registers a resolution of `request` within `ntmp` draws, or
    /// `demand_t::INF` when it cannot be resolved within any budget
    ///
    /// @param set The strategies tried, in preference order
    /// @param request What the caller asks of the iterator
    /// @param ntmp The cap on the registers, negative for none
    static int need(
            const strategy_set_t &set, const request_t &request, int ntmp) {
        const plan_t p = resolve(set, request, ntmp < 0 ? demand_t::INF : ntmp);
        return p.axis(0).strategy.is_any() ? demand_t::INF : p.ngpr();
    }

    /// Creates an iterator stream from a resolved plan
    ///
    /// @param plan The resolved plan
    ///
    /// @note Returns `invalid()` if the plan is invalid or unresolved, or the
    ///     table is full
    iterator_impl_t *iterator(const plan_t &plan) {
        iterator_impl_t *it = next_();
        CHECK_OR_RET(it, invalid(), "iterator table full");
        CHECK_OR_RET(
                it->init(plan), invalid(), "unable to create the iterator");
        take_();
        return it;
    }

    /// Allocates whatever registers `plan` still needs, in place
    ///
    /// @param plan The plan receiving the registers
    ///
    /// @note Accepts partially resolved plans
    void new_iterator(plan_t &plan) const {
        for (int d = 0; d < plan.ndims(); ++d) {
            auto &ax = plan.axis(d);
            if (!ax.needs_regs()) continue;
            ax.regs = pool_.integers().pop(ax.ngpr);
            LOG_FAIL_IF(
                    ax.needs_regs(), "insufficient registers for axis %d", d);
        }
    }

private:
    iterator_impl_t proto_;
    const strategy_set_t &strategies_;
    buffer::buffer_pool_t &pool_;
};

/// The strategies of a set as optimizable levels, from best to cheapest
///
/// @details An explicit `hint` is the only level; the relief between levels
///     is the difference of the registers each resolves `request` to
struct strategy_ladder_t : public resource_optimizable_t {
    static constexpr const char *NAME = "rvjit::stream::iterator::strategy";

    /// Prices each strategy level of a request
    ///
    /// @param set The strategies, in preference order
    /// @param request What the caller asks of the iterator
    /// @param hint The only strategy to try, `any` for every one of the set
    /// @param ntmp The cap on the registers of every level, negative for none
    /// @param out Receives the strategy of the current level
    strategy_ladder_t(const strategy_set_t &set, const request_t &request,
            strategy_id_t hint, int ntmp, strategy_id_t &out)
        : hint_(hint), out_(out) {
        nlevels_ = hint_.is_any() ? set.size() : 1;
        for (int l = 0; l < nlevels_; ++l) {
            request_t r = request;
            r.strategy = hint_.is_any() ? strategy_id_t(l) : hint_;
            need_[l] = iterator_t::need(set, r, ntmp);
        }
        out_ = hint_.is_any() ? strategy_id_t(0) : hint_;
    }

    int levels() const override { return nlevels_; }

    int level() const override { return hint_.is_any() ? out_.id() : 0; }

    bool set_level(int l) override {
        CHECK(0 <= l && l < levels(), "no such level");
        out_ = hint_.is_any() ? strategy_id_t(l) : hint_;
        return true;
    }

    demand_t demand() const override {
        return demand_t::of(resource_t::integers, need_[level()]);
    }

    demand_t relief(int from, int to) const override {
        return demand_t::of(resource_t::integers, need_[from])
                - demand_t::of(resource_t::integers, need_[to]);
    }

private:
    strategy_id_t hint_;
    strategy_id_t &out_;
    int nlevels_ = 1;
    int need_[strategy_set_t::CAPACITY] = {};
};

} // namespace iterator
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
