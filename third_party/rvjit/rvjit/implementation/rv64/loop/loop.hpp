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

#ifndef RVJIT_IMPLEMENTATION_RV64_LOOP_LOOP_HPP
#define RVJIT_IMPLEMENTATION_RV64_LOOP_LOOP_HPP

#include <cstdint>

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/isa/buffer/stack.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/parameter.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/interface/loop/loop.hpp"
#include "rvjit/interface/loop/plan.hpp"
#include "rvjit/interface/loop/schedule.hpp"
#include "rvjit/interface/loop/unroll.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace loop {

using branch_t = rvjit::control_flow::branch_t;
using distance_t = rvjit::control_flow::distance_t;
using on_enter_t = rvjit::control_flow::on_enter_t;
using on_index_t = rvjit::control_flow::on_index_t;

/// RV64 loop, holding its own descriptor and emitting itself
///
/// @note A `literal` instance emits no control flow and needs no resources
/// @note A `switch_` instance uses `counter` as the dispatch case id
/// @note A `unroll` instance counts `counter` down through each `factors`
///     entry, in units of size `grain`
/// @note A `unroll_switch` instance is an `unroll` followed by `switch_`
/// @note A `unroll_literal` instance is an `unroll` followed by an
///     unconditional literal call to the remaining `factors` entry
struct loop_t : public rvjit::loop::loop_t {
    static constexpr const char *NAME = "rvjit::rv64::loop";

    using mode_t = rvjit::loop::schedule_t::mode_t;

    mode_t mode = mode_t::unroll;

    rvjit::control_flow::distance_t dist
            = rvjit::control_flow::distance_t::short_;

    /// The loop unrolling factors applied to this loop
    rvjit::loop::unroll_t factors;

    /// Work-units represented by each unrolling-factor count
    ///
    /// @note A grain of `n = 1` represents element-wise loops
    /// @note A grain of `n > 1` represents data-parallel loops
    int grain = 1;

    /// The most work-units the counter holds on entry, 0 when unknown
    ///
    /// @note Lets a stage that cannot repeat be an `if_` rather than a `while_`
    int bound = 0;

    /// Register counting down (or selecting) the dispatched work
    Reg counter {};

    /// Where `counter` is loaded from on entry
    ///
    /// @note Never written back to
    rvjit::source_t entry = rvjit::source_t::none();

    /// Where `counter` is spilled while its register is lent to the body
    rvjit::slot_t spill {};

    /// Threshold/step/dispatch temporary
    Reg tmp {};

    /// The kind of loop it was planned as
    rvjit::loop::plan_t::kind_t plan_kind
            = rvjit::loop::plan_t::kind_t::data_parallel;

    /// Whether another loop shares this loop's registers
    bool shared = false;

    /// Whether `tmp` is lent to the body
    bool lent_tmp = false;

    /// Whether `counter` is lent to the body, spilled to `spill`
    bool lent_counter = false;

    /// Creates an unresolved loop
    ///
    /// @param e The emitter instructions are emitted through
    /// @param arith The constant arithmetic component
    /// @param cf The control flow component
    /// @param pool The pool the loop draws its registers from
    /// @param stack The stack the loop draws its spill slot from
    /// @param xfer The value transfer component
    loop_t(rvjit::emitter_t e, const operation::const_arithmetic_t &arith,
            const operation::control_flow_t &cf,
            rvjit::buffer::buffer_pool_t &pool, rvjit::stack_t &stack,
            const operation::transfer_t &xfer)
        : e_(e)
        , arith_(arith)
        , cf_(cf)
        , pool_(pool)
        , stack_(stack)
        , xfer_(xfer) {}

    /// The distance branches around a body of `instrs` instructions must reach
    ///
    /// @param instrs The instructions in the loop body
    static distance_t reach(int instrs) {
        static constexpr int INSTR_BYTES = 4;
        static constexpr int SHORT_BRANCH_BYTES = 1 << 12;
        return instrs * INSTR_BYTES >= SHORT_BRANCH_BYTES ? distance_t::medium
                                                          : distance_t::short_;
    }

    rvjit::loop::plan_t::kind_t kind() const override { return plan_kind; }

    /// Resolves `p` into the cheapest valid mode, allocating nothing
    ///
    /// @param p The plan
    ///
    /// @note `entry` is a copy of `size`'s source: the counter is mutated, and
    ///     the loop may be dispatched more than once
    bool init(const rvjit::loop::plan_t &p) {
        // Restore the default descriptor, dropping any drawn register
        frame_.truncate(0);
        alias_ = nullptr;
        counter = Reg {};
        spill = rvjit::slot_t {};
        tmp = Reg {};
        shared = false;
        lent_tmp = false;
        lent_counter = false;
        drawn_ = {};

        // Take the plan's schedule, reach and entry source
        const rvjit::loop::schedule_t s = rvjit::loop::schedule_t::of(p);
        mode = s.mode;
        factors = s.factors;
        grain = s.grain;
        bound = s.bound;
        dist = p.dist;
        entry = p.size.source().to_copy();
        plan_kind = p.kind;
        CHECK(is_resolved(), "an unresolved loop");
        return true;
    }

    /// Resolves `p` as a loop sharing `src`'s registers
    ///
    /// @param p The plan
    /// @param src The loop whose registers this loop adopts on `allocate()`
    ///
    /// @note Both loops stop lending `counter`, which they now share
    bool init_alias(const rvjit::loop::plan_t &p, loop_t &src) {
        CHECK(init(p), "an unresolved loop");
        alias_ = &src;
        shared = true;
        src.shared = true;
        return true;
    }

    /// @note Adopts an alias source's registers first, and keeps the registers
    ///     assigned by the caller
    ///
    /// @pre An alias source is allocated first
    shortfall_t allocate() override {
        // Forget what an earlier call drew
        if (drawn_.has(reg_name(counter))) counter = Reg {};
        if (drawn_.has(reg_name(tmp))) tmp = Reg {};
        drawn_ = {};

        if (alias_) {
            if (is_not_zero(alias_->counter)) counter = alias_->counter;
            if (is_not_zero(alias_->tmp)) tmp = alias_->tmp;
            if (entry.is_none()) entry = alias_->entry;
        }

        // `counter` is drawn before `tmp`
        const bool draw_counter = wants_counter_() && !is_not_zero(counter);
        const bool draw_tmp = wants_tmp_() && !is_not_zero(tmp);
        const auto regs = pool_.integers().take(draw_counter + draw_tmp);
        if (!regs) return regs.miss();
        drawn_ = *regs;
        int i = 0;
        if (draw_counter) counter = Reg(regs->id(i++));
        if (draw_tmp) tmp = Reg(regs->id(i++));

        spill = lent_counter ? stack_.new_slot() : rvjit::slot_t {};
        return frame_.allocate(pool_, stack_, has_multiple_trips());
    }

    /// @note Excludes the frame's slots, which the persistence binding them
    ///     accounts for
    demand_t demand() const override {
        // A register is drawn when wanted, neither assigned by the caller nor
        // adopted from an alias source
        const auto draws = [&](const Reg &r, bool wanted, bool adopted) {
            const bool unassigned = !is_not_zero(r) || drawn_.has(reg_name(r));
            return wanted && !adopted && unassigned ? 1 : 0;
        };
        const bool adopts_counter = alias_ && alias_->wants_counter_();
        const bool adopts_tmp = alias_ && alias_->wants_tmp_();
        const int ints = draws(counter, wants_counter_(), adopts_counter)
                + draws(tmp, wants_tmp_(), adopts_tmp);
        return demand_t::of(resource_t::integers, ints)
                + demand_t::of(resource_t::stack, lent_counter ? 1 : 0);
    }

    rvjit::buffer::buffer_t supplied() const override {
        rvjit::buffer::buffer_t b;
        if (is_not_zero(counter)) b.set(reg_name(counter));
        if (is_not_zero(tmp)) b.set(reg_name(tmp));
        return b;
    }

    bool is_resolved() const override { return !need_unrolling() && grain > 0; }

    rvjit::loop::frame_t &frame() override { return frame_; }

    /// Whether this object is missing valid unroll factors
    bool need_unrolling() const { return !factors.is_valid(); }

    /// Whether the mode is `literal`
    bool is_literal() const { return mode == mode_t::literal; }

    /// Whether the mode is `switch_`
    bool is_switch() const { return mode == mode_t::switch_; }

    /// Whether the mode is `unroll`
    bool is_unroll() const { return mode == mode_t::unroll; }

    /// Whether the mode is `unroll_switch`
    bool is_unroll_switch() const { return mode == mode_t::unroll_switch; }

    /// Whether the mode is `unroll_literal`
    bool is_unroll_literal() const { return mode == mode_t::unroll_literal; }

    /// Whether the mode counts `counter` down through the unroll factors
    bool has_unroll() const {
        return is_unroll() || is_unroll_switch() || is_unroll_literal();
    }

    bool is_valid() const override {
        if (need_unrolling()) return false;
        const bool assigned = (!wants_counter_() || is_not_zero(counter))
                && (!wants_tmp_() || is_not_zero(tmp));
        const bool core = grain > 0 && assigned && tmp != counter;
        switch (mode) {
            case mode_t::literal: return assigned;
            case mode_t::switch_:
                return factors[0] <= operation::control_flow_t::NCASES
                        && assigned && counter != tmp;
            case mode_t::unroll: return core;
            case mode_t::unroll_switch:
                return core
                        && factors[0] <= operation::control_flow_t::NCASES + 1;
            case mode_t::unroll_literal: return core;
        }
        return false;
    }

    /// @note A bounded `unroll` or `unroll_switch` runs once when its stages
    ///     cannot dispatch more than once between them
    bool has_multiple_trips() const override {
        switch (mode) {
            case mode_t::literal: return factors.size() > 1;
            case mode_t::switch_: return false;
            case mode_t::unroll_literal: return true;
            case mode_t::unroll:
            case mode_t::unroll_switch: {
                if (bound <= 0) return true;
                // The switch tail dispatches once more
                int64_t trips = is_unroll_switch() ? 1 : 0;
                int64_t rem = bound;
                for (int i = 0; i < factors.size(); ++i) {
                    trips += rvjit::loop::schedule_t::stage_trips(
                            factors[i], grain, rem);
                    rem = rvjit::loop::schedule_t::stage_exit(
                            factors[i], grain, rem);
                }
                return trips > 1;
            }
        }
        return false;
    }

    /// @note A loop with `granularity > 1` always keeps the register
    bool has_remaining() const override {
        return !is_switch() && (has_multiple_trips() || grain > 1)
                && is_not_zero(counter);
    }

    /// @note Holds the work-units still to be dispatched, in elements,
    ///     including the current call's; valid within the loop's body
    /// @note Zero if the loop keeps no such register, which a loop with
    ///     `granularity > 1` always does
    reg_t remaining() const override { return counter; }

    /// @note Drives the loop's `frame` around each `cb` call
    ///
    /// @pre A `switch_` has `factors[0]` in [1, `control_flow_t::NCASES`]
    ///     and `counter / grain` in [1, `factors[0]`]
    /// @pre An `unroll_switch` has its last factor minus one in [1,
    ///     `control_flow_t::NCASES`], and a `grain > 1` dividing the count
    /// @pre An `unroll_literal` has a compile-time literal total
    bool generate(const on_index_t &cb) override {
        CHECK(is_valid(), "invalid loop descriptor");

        const bool has_frame = frame_.has_active();
        const on_index_t body = has_frame
                ? on_index_t([&](int n) { frame_(xfer_, [&] { cb(n); }); })
                : on_index_t([&](int n) { cb(n); });

        switch (mode) {
            case mode_t::literal: {
                // No control flow; a `grain > 1` loads the elements still to
                // be dispatched into `counter` before each call
                int left = 0;
                for (int i = 0; i < factors.size(); ++i)
                    left += factors[i] * grain;
                for (int i = 0; i < factors.size(); ++i) {
                    if (grain > 1)
                        xfer_.init(rvjit::variable_t(
                                counter, rvjit::source_t::immediate(left)));
                    body(factors[i]);
                    left -= factors[i] * grain;
                }
                break;
            }
            case mode_t::switch_: {
                // A `grain > 1` divides `counter` into `tmp` first; clobbers
                // `counter` and `tmp`
                const int n = factors[0];
                xfer_.init(rvjit::variable_t(counter, entry));
                if (n > 1 && grain > 1) {
                    arith_.div(tmp, counter, rvjit::const_t(grain));
                    cf_.switch_case(n, rvjit::variable_t(tmp), counter, body);
                } else {
                    cf_.switch_case(n, rvjit::variable_t(counter), tmp, body);
                }
                break;
            }
            case mode_t::unroll: unroll_(factors, body); break;
            case mode_t::unroll_switch: {
                // A `switch_case` over the remainder `unroll` leaves
                unroll_(factors, body);
                const int switch_cases = factors[factors.size() - 1] - 1;
                Reg switch_rid = counter;
                Reg switch_tmp = tmp;
                if (grain > 1) {
                    arith_.div(tmp, counter, rvjit::const_t(grain));
                    switch_rid = tmp;
                    switch_tmp = counter;
                }
                cf_.if_(branch_t::nez(switch_rid, distance_t::medium), [&] {
                    cf_.switch_case(switch_cases, switch_rid, switch_tmp, body);
                });
                break;
            }
            case mode_t::unroll_literal:
                // `unroll` by `factors[0]`, then a literal tail
                unroll_(rvjit::loop::unroll_t(factors[0]), body);
                if (factors.size() > 1 && factors[1] > 0) body(factors[1]);
                break;
        }
        return true;
    }

    int levels() const override {
        const lend_t all = lends_(2);
        return 1 + all.tmp + all.counter;
    }

    int level() const override { return lent_tmp + lent_counter; }

    /// @note Lends `tmp` first, then `counter`, spilled around each body
    bool set_level(int l) override {
        CHECK(0 <= l && l < levels(), "no such level");
        const lend_t lend = lends_(l);
        lent_tmp = lend.tmp;
        lent_counter = lend.counter;
        return true;
    }

    demand_t relief(int from, int to) const override {
        const lend_t f = lends_(from), t = lends_(to);
        return demand_t::of(resource_t::integers,
                       (t.tmp + t.counter) - (f.tmp + f.counter))
                + demand_t::of(resource_t::stack, f.counter - t.counter);
    }

    rvjit::buffer::buffer_t released() const override {
        rvjit::buffer::buffer_t b;
        if (lent_tmp) b.set(reg_name(tmp));
        if (lent_counter) b.set(reg_name(counter));
        return b;
    }

    distance_t distance() const override { return dist; }

    bool set_distance(distance_t d) override {
        dist = d;
        return true;
    }

    const rvjit::loop::unroll_t &unroll() const override { return factors; }

    int granularity() const override { return grain; }

    rvjit::loop::state_t checkpoint() const override {
        rvjit::loop::state_t m;
        m.frame = frame_.size();
        m.lent = (lent_tmp ? 1 : 0) | (lent_counter ? 2 : 0);
        m.spill = spill;
        return m;
    }

    bool rewind(const rvjit::loop::state_t &m) override {
        frame_.truncate(m.frame);
        lent_tmp = m.lent & 1;
        lent_counter = m.lent & 2;
        spill = m.spill;
        return true;
    }

private:
    /// Whether the mode requires a `counter`, assigned or not
    bool wants_counter_() const { return !is_literal() || grain > 1; }

    /// Whether the mode requires a `tmp`, assigned or not
    ///
    /// @note An `unroll` needs none when every factor is 1 and `-grain` folds
    ///     into a simm12 immediate; `unroll_literal` scans only `factors[0]`
    bool wants_tmp_() const {
        if (is_literal()) return false;
        if (is_switch() || is_unroll_switch()) return true;
        const int n = is_unroll_literal() ? 1 : factors.size();
        for (int i = 0; i < n; ++i)
            if (factors[i] > 1) return true;
        return !is_simm12(-grain);
    }

    struct lend_t {
        bool tmp = false;
        bool counter = false;
    };

    /// What level `l` lends: `tmp` first, then a `counter` not shared with
    /// another loop, which needs a spill slot
    lend_t lends_(int l) const {
        const bool can_tmp = has_unroll() && wants_tmp_();
        const bool can_counter = has_unroll() && !shared && wants_counter_();
        lend_t lend;
        lend.tmp = can_tmp && l >= 1;
        lend.counter = can_counter && l >= (can_tmp ? 2 : 1);
        return lend;
    }

    /// Emits a loop counting `counter` down through descending factors
    ///
    /// @details Each factor `ur` steps `counter` by `ur * grain`
    ///
    /// @note A set `spill` spills `counter` around `body`, freeing its
    ///     register for `body`
    /// @note A stage becomes an `if_` instead of a `while_` when it cannot
    ///     dispatch twice: the previous stage leaves less than its own width,
    ///     and `bound` caps what the first stage is entered with
    ///
    /// @pre `counter` holds the trip count on entry or `entry` is set
    /// @pre `tmp` is writeable and distinct from `counter`
    /// @pre `dist` covers the longest loop `body`
    void unroll_(const rvjit::loop::unroll_t &urs, const on_index_t &body) {
        const Reg &iter = counter;

        xfer_.init(rvjit::variable_t(iter, entry));

        rvjit::bind_t counter_bind;
        counter_bind.reg = iter;
        counter_bind.slot = spill;

        // The most work-units the counter holds entering each stage
        int64_t rem = bound > 0 ? bound : INT64_MAX / 2;

        for (int i = 0; i < urs.size(); ++i) {
            const int ur = urs[i];

            const bool as_if
                    = rvjit::loop::schedule_t::stage_trips(ur, grain, rem) <= 1;
            rem = rvjit::loop::schedule_t::stage_exit(ur, grain, rem);
            const auto emit = [&](const branch_t &b, const on_enter_t &pass) {
                if (as_if)
                    cf_.if_(b, pass);
                else
                    cf_.while_(b, pass);
            };

            if (ur == 1) {
                counter_bind.step = rvjit::step_t::macc(
                        rvjit::const_t(-1), rvjit::const_t(grain), tmp);
                emit(branch_t::gtz(iter, dist), [&] {
                    xfer_.preserve(counter_bind);
                    body(1);
                    xfer_.restore(counter_bind);
                    xfer_.advance(counter_bind);
                });
                continue;
            }

            const int width = ur * grain;
            const rvjit::variable_t threshold(
                    tmp, rvjit::source_t::immediate(width));
            xfer_.init(threshold);

            const rvjit::const_t step(width, tmp);
            const bool reload = lent_tmp && (step.has_reg() || !as_if);
            counter_bind.step = rvjit::step_t::sub(step);
            emit(branch_t::ge(iter, tmp, dist), [&] {
                xfer_.preserve(counter_bind);
                body(ur);
                xfer_.restore(counter_bind);
                if (reload) xfer_.init(threshold);
                xfer_.advance(counter_bind);
            });
        }
    }

    emitter_t e_;
    const operation::const_arithmetic_t &arith_;
    const operation::control_flow_t &cf_;
    rvjit::buffer::buffer_pool_t &pool_;
    rvjit::stack_t &stack_;
    const operation::transfer_t &xfer_;

    /// Bindings driven alongside this loop's dispatch
    rvjit::loop::frame_t frame_;

    /// The loop whose registers this one adopts on `allocate()`, if any
    loop_t *alias_ = nullptr;

    /// The registers the last `allocate()` drew
    rvjit::buffer::buffer_t drawn_;
};

} // namespace loop
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
