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

#ifndef RVJIT_IMPLEMENTATION_RV64_STREAM_ITERATOR_STRATEGIES_GUIDED_HPP
#define RVJIT_IMPLEMENTATION_RV64_STREAM_ITERATOR_STRATEGIES_GUIDED_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/stream/iterator/strategy.hpp"

namespace rvjit {
namespace rv64 {
namespace stream {
namespace iterator {

/// Moves the base pointer at each iteration for the whole axis
struct guided_t : rvjit::stream::iterator::strategy_t {
    static constexpr const char *NAME = "rvjit::rv64::stream::iterator";

    using addressing_t = rvjit::stream::iterator::addressing_t;
    using axis_t = rvjit::stream::iterator::axis_t;
    using budget_t = rvjit::stream::iterator::budget_t;
    using claim_t = rvjit::stream::iterator::claim_t;
    using plan_t = rvjit::stream::iterator::plan_t;
    using request_t = rvjit::stream::iterator::request_t;
    using update_t = rvjit::stream::iterator::update_t;
    using window_t = rvjit::stream::iterator::window_t;

    const char *name() const override { return "guided"; }

    /// Any stride and addressing: a register walks the axis
    bool supports(const axis_t &, addressing_t) const override { return true; }

    claim_t claim(const request_t &request, int d, const budget_t &budget,
            const window_t &window, plan_t &out) const override {
        claim_t c;

        // The end axis of a preserving plan keeps its register, any other
        // axis needs one only when it has more than one position and does
        // not walk the advancing origin
        const bool is_end_axis = d == request.update.axis();
        const bool preserve
                = request.update.kind() == update_t::kind_t::preserve;
        const bool multi_position = request.space.axis(d).size > 1;
        const bool takes = !is_end_axis && multi_position
                && takes_origin_(request, d, out);
        const int need
                = (is_end_axis ? preserve : multi_position && !takes) ? 1 : 0;

        // A walking origin cannot follow an outer axis that moves
        bool outer_moves = false;
        for (int i = 0; i < d; ++i)
            outer_moves = outer_moves || request.space.axis(i).size > 1;
        if (is_end_axis && !preserve && outer_moves) {
            c.status = claim_t::status_t::misfit;
            return c;
        }

        if (!budget.fits(need)) {
            c.status = claim_t::status_t::misfit;
            return c;
        }
        out.axis(d).ngpr = need;
        out.axis(d).takes_origin = takes;
        c.status = claim_t::status_t::fit;
        c.last_axis = d;
        c.regs = need;
        c.window = {0, window.reserve};
        return c;
    }

    void walk(rvjit::pointer_t &out, const plan_t &plan, int d,
            const rvjit::index_t &idx) const override {
        const auto &ap = plan.axis(d);
        if (idx[d] > 0 && ap.ngpr > 0) out.base = Reg(ap.regs.id(0));
    }

    void step(const operation::const_arithmetic_t &arith, const plan_t &plan,
            int d, const rvjit::pointer_t &prev,
            const rvjit::index_t &) const override {
        const auto &ap = plan.axis(d);
        const auto &ax = plan.space.axis(d);
        const Reg base = plan.origin.base;
        if (ap.ngpr > 0) {
            arith.add(Reg(ap.regs.id(0)), prev.base, ax.stride);
        } else if (ap.takes_origin) {
            arith.add(base, base, ax.stride);
        } else if (d == plan.update.axis()) {
            // Without a register, the origin base is the walking pointer
            end_step_(arith, plan, d);
        }
    }

    /// @note Lands the pointer at the full span, after every row transition
    ///     already stepped once
    void finish(const operation::const_arithmetic_t &arith, const plan_t &plan,
            int end_axis) const override {
        end_step_(arith, plan, end_axis);
    }

private:
    /// Whether non-end axis `d` walks the advancing origin instead of a register
    ///
    /// @note Only when the rewind of `d` folds into the end axis's immediate
    ///     step, every other axis being a single position, so the walk costs
    ///     no more instructions than a register would
    bool takes_origin_(
            const request_t &request, int d, const plan_t &out) const {
        const int end = request.update.axis();
        const auto &space = request.space;
        const bool owned = request.update.kind() != update_t::kind_t::preserve;
        if (!owned || end >= d || out.axis(end).strategy != out.axis(d).strategy
                || out.axis(end).ngpr != 0)
            return false;
        for (int i = 0; i < space.ndims(); ++i)
            if (i != d && i != end && space.axis(i).size > 1) return false;

        const auto &ax = space.axis(d);
        const auto &end_ax = space.axis(end);
        if (!ax.stride.has_imm() || !end_ax.stride.has_imm()) return false;
        if (!is_simm12(ax.stride.imm()) || !is_simm12(end_ax.stride.imm()))
            return false;

        // A zero end step is elided, so the rewind would be an extra add
        if (end_ax.stride.imm() == 0) return false;

        // Both ends fitting, so does the step of any narrower resize
        const int rewind = -(ax.size - 1) * ax.stride.imm();
        return is_simm12(rewind) && is_simm12(end_ax.stride.imm() + rewind);
    }

    /// Emits the end axis's step on the origin, rewinding the axis that walks
    /// it back to its first position
    void end_step_(const operation::const_arithmetic_t &arith,
            const plan_t &plan, int end_axis) const {
        const Reg base = plan.origin.base;
        int rewind = 0;
        for (int i = 0; i < plan.ndims(); ++i)
            if (plan.axis(i).takes_origin)
                rewind -= (plan.space.axis(i).size - 1)
                        * plan.space.axis(i).stride.imm();
        const auto &stride = plan.space.axis(end_axis).stride;
        if (rewind == 0)
            arith.add(base, base, stride);
        else
            arith.add(base, base, rvjit::const_t(stride.imm() + rewind));
    }
};

} // namespace iterator
} // namespace stream
} // namespace rv64
} // namespace rvjit

#endif
