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

#ifndef RVJIT_IMPLEMENTATION_RV64_STREAM_ITERATOR_STRATEGIES_PIVOT_HPP
#define RVJIT_IMPLEMENTATION_RV64_STREAM_ITERATOR_STRATEGIES_PIVOT_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/stream/iterator/strategy.hpp"

namespace rvjit {
namespace rv64 {
namespace stream {
namespace iterator {

/// Distributes pointers across the space to cover simm12-sized offset windows
struct pivot_t : rvjit::stream::iterator::strategy_t {
    static constexpr const char *NAME = "rvjit::rv64::stream::iterator";

    using addressing_t = rvjit::stream::iterator::addressing_t;
    using axis_plan_t = rvjit::stream::iterator::axis_plan_t;
    using axis_t = rvjit::stream::iterator::axis_t;
    using budget_t = rvjit::stream::iterator::budget_t;
    using claim_t = rvjit::stream::iterator::claim_t;
    using plan_t = rvjit::stream::iterator::plan_t;
    using request_t = rvjit::stream::iterator::request_t;
    using update_t = rvjit::stream::iterator::update_t;
    using window_t = rvjit::stream::iterator::window_t;

    const char *name() const override { return "pivot"; }

    /// Any stride; a runtime one takes one register per position
    bool supports(const axis_t &, addressing_t) const override { return true; }

    /// Resolves axis `d`, or the run of one-register-per-position axes
    /// starting at it
    ///
    /// @details A run of 2+ such axes resolves as one combined-index chain,
    ///     only when it starts at axis 0 and its register product fits the
    ///     budget; otherwise it is withheld, so a fallback must fit
    claim_t claim(const request_t &request, int d, const budget_t &budget,
            const window_t &window, plan_t &out) const override {
        claim_t c;
        const auto &space = request.space;
        const auto &ax = space.axis(d);
        const addressing_t mode = request.addressing;
        if (!supports(ax, mode)) return c;

        // An advancing origin serves as the first register of the first axis
        // owning any, when only that axis moves it: it is the end axis, or
        // the one the end axis delegates its advance to
        const int end = request.update.axis();
        const auto &end_ap = out.axis(end);
        bool first_owner = true;
        for (int i = 0; i < d; ++i)
            first_owner = first_owner && out.axis(i).nregs() == 0;
        const bool delegated = d == end + 1
                && end_ap.strategy == out.axis(d).strategy
                && end_ap.nregs() == 0 && space.axis(end).stride.has_imm()
                && !would_be_multibase_(space.axis(end), mode);
        const bool owned = request.update.kind() != update_t::kind_t::preserve;
        const bool takes = owned && first_owner && (d == end || delegated);

        // A one-register-per-position axis chains with its neighbors when it
        // can, else it takes one register per position on its own
        if (would_be_multibase_(ax, mode)) {
            // Built once from the origin: outer axes may only move by offsets
            for (int i = 0; i < d; ++i) {
                const auto &outer = out.axis(i);
                const bool offsets_only = outer.strategy == out.axis(d).strategy
                        && outer.nregs() == 0
                        && !would_be_multibase_(space.axis(i), mode);
                if (space.axis(i).size > 1 && !offsets_only) {
                    c.status = claim_t::status_t::misfit;
                    return c;
                }
            }

            // The run of chainable axes around `d`, which a runtime stride
            // never joins
            int first = d, last = d;
            const bool chains = can_chain_(ax, mode);
            while (chains && first > 0
                    && can_chain_(space.axis(first - 1), mode))
                --first;
            while (chains && last + 1 < space.ndims()
                    && can_chain_(space.axis(last + 1), mode))
                ++last;

            if (last > first) {
                long long product = 1;
                for (int i = first; i <= last; ++i)
                    product *= space.axis(i).size;
                if (d != first || first != 0 || product <= 0
                        || product > budget.remaining) {
                    c.status = claim_t::status_t::withheld;
                    return c;
                }
                // The last axis owns the registers, the others delegate
                for (int i = first; i <= last; ++i) {
                    axis_plan_t &o = out.axis(i);
                    o.ngpr = i == last ? static_cast<int>(product) : 0;
                    o.chain_start = first;
                    o.takes_origin = false;
                }
                c.last_axis = last;
                c.regs = static_cast<int>(product);
            } else {
                const int need = ax.size - (takes ? 1 : 0);
                if (!budget.fits(need)) {
                    c.status = claim_t::status_t::misfit;
                    return c;
                }
                axis_plan_t &o = out.axis(d);
                o.ngpr = need;
                o.positions_per_pivot = 0;
                o.shift = 0;
                o.chain_start = d;
                o.takes_origin = takes;
                c.last_axis = d;
                c.regs = need;
            }
            c.status = claim_t::status_t::fit;
            c.window = {0, window.reserve};
            return c;
        }

        // An axis fitting the current base needs no register (case a),
        // else it is split into pivots, each covering one simm12 window
        const int stride = ax.stride.imm();
        const int abs_stride = stride < 0 ? -stride : stride;
        const int span = (ax.size - 1) * abs_stride;
        const bool tail = d == request.update.axis()
                && request.update.kind() != update_t::kind_t::preserve;
        const bool case_a = (!tail || is_simm12(ax.size * stride))
                && is_simm12(window.group_off + span);

        int need = 0, per_pivot = 0, shift = 0;
        if (!case_a) {
            // The window starts at one end of simm12 and grows toward the
            // other, leaving `reserve` headroom at both ends
            const int reach = 2047 - window.reserve;
            shift = stride > 0 ? reach : -reach;
            per_pivot = (4094 - 2 * window.reserve) / abs_stride + 1;
            if (tail) per_pivot = per_pivot > 1 ? per_pivot - 1 : 1;
            need = (ax.size + per_pivot - 1) / per_pivot;
        }
        const bool strips_take = takes && need > 0;
        const int drawn = need - (strips_take ? 1 : 0);
        if (!budget.fits(drawn) || (need > 1 && budget.multi >= 1)) {
            c.status = claim_t::status_t::misfit;
            return c;
        }

        axis_plan_t &o = out.axis(d);
        o.ngpr = drawn;
        o.positions_per_pivot = per_pivot;
        o.shift = shift;
        o.chain_start = -1;
        o.takes_origin = strips_take;
        c.status = claim_t::status_t::fit;
        c.last_axis = d;
        c.regs = drawn;
        c.multi = need > 1;

        // A case-a axis adds its span to the window and leaves its delegate
        // headroom for the largest compensated read, which only an eager axis
        // has
        const int delegate_reserve = (d == 0 && tail) ? span : abs_stride;
        const window_t case_a_window {
                window.group_off + span, delegate_reserve};
        c.window = case_a ? case_a_window : window_t {0, window.reserve};
        return c;
    }

    /// Contributes axis `d`'s base/offset to `out` when reading `idx`
    ///
    /// @note An eager case-a end axis compensates its reads at `iter > 0` for
    ///     the step already landed on the base pointer
    void walk(rvjit::pointer_t &out, const plan_t &plan, int d,
            const rvjit::index_t &idx) const override {
        const auto &ax = plan.space.axis(d);
        const auto &ap = plan.axis(d);
        const int iter = idx[d];

        if (would_be_multibase_(ax, plan.addressing)) {
            // A chain delegate is covered by its owner's combined index
            if (ap.nregs() == 0) return;
            int combined = 0;
            for (int i = ap.chain_start; i <= d; ++i)
                combined = combined * plan.space.axis(i).size + idx[i];
            out.base = reg_(plan, ap, combined);
        } else if (ap.nregs() == 0) {
            const int compensation = (iter > 0 && is_eager_(plan, d))
                    ? ax.size * ax.stride.imm()
                    : 0;
            out.offset += iter * ax.stride.imm() - compensation;
        } else {
            const int p = iter / ap.positions_per_pivot;
            const int r = iter % ap.positions_per_pivot;
            out.base = reg_(plan, ap, p);
            out.offset += r * ax.stride.imm() - ap.shift;
        }
    }

    /// Emits the end axis's whole-span step at its first transition when eager
    ///
    /// @note A delegating eager axis steps each delegate register right after
    ///     its last read of the first row, so the steps interleave with the
    ///     first row's reads
    void step(const operation::const_arithmetic_t &arith, const plan_t &plan,
            int d, const rvjit::pointer_t &,
            const rvjit::index_t &idx) const override {
        if (idx[0] != 0 || !is_eager_(plan, 0)) return;
        if (!delegates_(plan, 0)) {
            if (d == 0) advance_(arith, plan, 0);
            return;
        }

        // The register of the delegate's position steps once that position
        // is the register's last; the row's last position ends at axis 0
        const auto &ap = plan.axis(1);
        const int pos = idx[1];
        const int per = ap.chain_start == 1 ? 1 : ap.positions_per_pivot;
        const bool last_of_reg = d == 0 || (d == 1 && (pos + 1) % per == 0);
        if (last_of_reg) bump_(arith, plan, 0, reg_(plan, ap, pos / per));
    }

    /// Emits axis `end_axis`'s end-of-space step, unless it landed eagerly
    ///
    /// @pre `end_axis` resolved to strategy `pivot`
    void finish(const operation::const_arithmetic_t &arith, const plan_t &plan,
            int end_axis) const override {
        if (!is_eager_(plan, end_axis)) advance_(arith, plan, end_axis);
    }

    /// Emits axis `d`'s own pivot-chain or grouped-strip construction
    ///
    /// @details `walked_base` is the pointer resolved through axes `[0, d)`
    ///     at `idx`, used by grouped strips; a chain starts from the origin
    ///
    /// @pre Every axis inner to `d` is at its first-visit position
    void init(rvjit::emitter_t e, const operation::const_arithmetic_t &arith,
            const operation::transfer_t &xfer, const plan_t &plan, int d,
            const rvjit::index_t &idx,
            const rvjit::pointer_t &walked_base) const override {
        const int end_axis = plan.update.axis();
        const bool pointer_moves
                = plan.update.kind() != update_t::kind_t::preserve;
        const auto &ax = plan.space.axis(d);
        const auto &ap = plan.axis(d);

        // The registers are shared by every outer position, so only the first
        // one builds them
        const int outer = ap.chain_start >= 0 ? ap.chain_start : d;
        for (int i = 0; i < outer; ++i)
            if (idx[i] != 0) return;

        if (would_be_multibase_(ax, plan.addressing)) {
            // One register per combined position of the chain `[chain_start,
            // d]`, which is `d` alone for a standalone axis, each built at
            // its own position
            int product = 1, combined = 0;
            bool last_visit = true;
            for (int i = ap.chain_start; i <= d; ++i) {
                const int size = plan.space.axis(i).size;
                product *= size;
                combined = combined * size + idx[i];
                last_visit = last_visit && idx[i] == size - 1;
            }

            const Reg cur = reg_(plan, ap, combined);
            if (combined == 0) {
                if (!ap.takes_origin)
                    arith.add(cur, plan.origin.base, rvjit::const_t(0),
                            /*noop_zero=*/false);
            } else if (!ax.stride.has_imm()) {
                arith.add(cur, reg_(plan, ap, combined - 1), ax.stride);
            } else {
                int offset = 0, rem = combined;
                for (int j = d; j >= ap.chain_start; --j) {
                    const int size = plan.space.axis(j).size;
                    offset += (rem % size) * plan.space.axis(j).stride.imm();
                    rem /= size;
                }
                const rvjit::const_t off(offset, cur);
                xfer.init(off);
                arith.add(cur, plan.origin.base, off, /*noop_zero=*/false);
            }
            if (last_visit && d == end_axis && pointer_moves
                    && !ap.takes_origin)
                arith.add(plan.origin.base, reg_(plan, ap, product - 1),
                        ax.stride);
            return;
        }

        // Grouped strips share a pivot register through simm12 offsets; each
        // is built at its first position, spreading them over the init
        if (idx[d] % ap.positions_per_pivot != 0) return;
        const int strip = idx[d] / ap.positions_per_pivot;
        const int n_strips = (ax.size + ap.positions_per_pivot - 1)
                / ap.positions_per_pivot;

        // Strip 0 sits at `shift` and every next strip a fixed step after
        // the previous one; the last strip holds an oversized step until
        // it is built
        const Reg cur = reg_(plan, ap, strip);
        if (strip == 0) {
            if (ap.takes_origin)
                arith.add(cur, cur, rvjit::const_t(ap.shift));
            else if (ap.shift == 0)
                emitter_t(e)->mv(cur, walked_base.base);
            else
                arith.add(cur, walked_base.base, rvjit::const_t(ap.shift));
        } else {
            const int step = ap.positions_per_pivot * ax.stride.imm();
            const rvjit::const_t step_c(step, reg_(plan, ap, n_strips - 1));
            if (strip == 1) xfer.init(step_c);
            arith.add(cur, reg_(plan, ap, strip - 1), step_c);
        }

        const bool last_strip = strip == n_strips - 1;
        if (last_strip && d == end_axis && pointer_moves && !ap.takes_origin) {
            const int tail_count
                    = ax.size - (n_strips - 1) * ap.positions_per_pivot;
            const int tail_imm = tail_count * ax.stride.imm() - ap.shift;
            arith.add(plan.origin.base, reg_(plan, ap, n_strips - 1),
                    rvjit::const_t(tail_imm));
        }
    }

private:
    /// Whether `ax` needs one register per position, because its stride has no
    /// immediate form or `mode` has no offset field
    bool would_be_multibase_(const axis_t &ax, addressing_t mode) const {
        return !ax.stride.has_imm() || mode == addressing_t::base_only;
    }

    /// Whether a one-register-per-position axis can join a chain, which needs
    /// a known immediate stride to host-compute every position's offset
    bool can_chain_(const axis_t &ax, addressing_t mode) const {
        return would_be_multibase_(ax, mode) && ax.stride.has_imm();
    }

    /// Whether axis `d`'s whole-span step lands at its first transition
    /// instead of at the last cell
    ///
    /// @note True for the advancing end axis at axis 0 when it is a case-a
    ///     axis with more than one position
    ///
    /// @pre Callers issue `get(idx)` immediately followed by
    ///     `produce().iteration(idx)` in ascending order, after every `init()`
    bool is_eager_(const plan_t &plan, int d) const {
        return d == 0 && d == plan.update.axis()
                && plan.update.kind() != update_t::kind_t::preserve
                && !would_be_multibase_(plan.space.axis(d), plan.addressing)
                && plan.axis(d).nregs() == 0 && plan.space.axis(d).size >= 2;
    }

    /// Register `i` of axis plan `ap`, the origin's being the first when the
    /// axis takes it
    Reg reg_(const plan_t &plan, const axis_plan_t &ap, int i) const {
        if (!ap.takes_origin) return Reg(ap.regs.id(i));
        return i == 0 ? Reg(plan.origin.base) : Reg(ap.regs.id(i - 1));
    }

    /// Whether a register-free end axis hands its step to the next axis's
    /// registers, the origin base never moving
    bool delegates_(const plan_t &plan, int end_axis) const {
        const auto &end_ap = plan.axis(end_axis);
        return end_ap.nregs() == 0 && plan.space.axis(end_axis).stride.has_imm()
                && end_axis + 1 < plan.ndims()
                && plan.axis(end_axis + 1).strategy == end_ap.strategy
                && plan.axis(end_axis + 1).nregs() > 0;
    }

    /// Emits axis `end_axis`'s whole-span step on register `r`
    void bump_(const operation::const_arithmetic_t &arith, const plan_t &plan,
            int end_axis, const Reg &r) const {
        // One add of the whole span when it fits simm12, else one per step
        const auto &end_ax = plan.space.axis(end_axis);
        const int steps = end_ax.size;
        const bool consolidate = end_ax.stride.has_imm()
                && is_simm12(steps * end_ax.stride.imm());
        const rvjit::const_t step_or_total = consolidate
                ? rvjit::const_t(steps * end_ax.stride.imm())
                : end_ax.stride;
        const int reps = consolidate ? 1 : steps;
        for (int s = 0; s < reps; ++s)
            arith.add(r, r, step_or_total);
    }

    /// Emits axis `end_axis`'s whole-span step
    ///
    /// @pre `end_axis` resolved to strategy `pivot`
    void advance_(const operation::const_arithmetic_t &arith,
            const plan_t &plan, int end_axis) const {
        const auto &end_ax = plan.space.axis(end_axis);
        const auto &end_ap = plan.axis(end_axis);

        if (delegates_(plan, end_axis)) {
            // Only the delegate registers the current shape reads; a chain
            // owner's combined positions all are
            const int d = end_axis + 1;
            const auto &ap = plan.axis(d);
            const int size = plan.space.axis(d).size;
            int used = ap.nregs();
            if (ap.chain_start == d)
                used = size;
            else if (ap.chain_start < 0)
                used = (size + ap.positions_per_pivot - 1)
                        / ap.positions_per_pivot;
            for (int i = 0; i < used; ++i)
                bump_(arith, plan, end_axis, reg_(plan, ap, i));
        } else if (end_ap.nregs() == 0 && end_ax.stride.has_imm()) {
            bump_(arith, plan, end_axis, Reg(plan.origin.base));
        } else if (end_ap.nregs() > 0) {
            const int size = end_ax.size;
            const bool chain = end_ap.chain_start == end_axis;
            const bool consolidates = end_ax.stride.has_imm()
                    && is_simm12(size * end_ax.stride.imm());
            if (chain && !consolidates) {
                // Restart the chain one stride past the last position read
                arith.add(reg_(plan, end_ap, 0), reg_(plan, end_ap, size - 1),
                        end_ax.stride);
                for (int i = 1; i < size; ++i)
                    arith.add(reg_(plan, end_ap, i), reg_(plan, end_ap, i - 1),
                            end_ax.stride);
                return;
            }

            // Every owned register stayed fixed during the pass, and none is
            // free to materialize an oversized total, so each is caught up here
            for (int i = 0; i < end_ap.nregs(); ++i)
                bump_(arith, plan, end_axis, reg_(plan, end_ap, i));
        }
    }
};

} // namespace iterator
} // namespace stream
} // namespace rv64
} // namespace rvjit

#endif
