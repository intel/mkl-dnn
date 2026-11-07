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

#ifndef RVJIT_INTERFACE_INDUCTION_INDUCTION_HPP
#define RVJIT_INTERFACE_INDUCTION_INDUCTION_HPP

#include "rvjit/interface/induction/bind.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// Aggregates `bind_t` objects manipulated uniformly around a delegate
///
/// @note Implements inductive variable optimizations based on parametrization
struct induction_t : public array_t<bind_t, 8> {
    using base_t = array_t<bind_t, 8>;

    /// Creates a set of bindings
    ///
    /// @param list The bindings
    induction_t(std::initializer_list<bind_t> list = {}) : base_t(list) {}

    /// The binding of register `r`, or nullptr if none is aggregated
    ///
    /// @param r The register of the bound value
    bind_t *find(const reg_t &r) {
        for (int i = 0; i < size(); ++i)
            if ((*this)[i].reg == r) return &(*this)[i];
        return nullptr;
    }

    /// Whether any aggregated binding takes part in the inductive routine
    bool has_active() const {
        for (int i = 0; i < size(); ++i)
            if ((*this)[i].active) return true;
        return false;
    }

    /// Preserves every active binding, runs `body`, then restores and
    /// advances them
    ///
    /// @param xfer The transfer component that emits the passes
    /// @param body The code the bindings are preserved around
    ///
    /// @note The bindings are updated in order. A step that reads a bound
    ///     register sees its updated value if that binding comes earlier, and
    ///     its restored value if it comes later
    /// @note A binding reloaded from a register and stepped is restored and
    ///     advanced in a single update from that register
    void operator()(const operation::transfer_t &xfer,
            const control_flow::on_enter_t &body) const {
        for (int i = 0; i < size(); ++i)
            if ((*this)[i].active) xfer.preserve((*this)[i]);
        if (body) body();
        for (int i = 0; i < size(); ++i)
            if ((*this)[i].active && !fuses_(i)) xfer.restore((*this)[i]);
        for (int i = 0; i < size(); ++i) {
            if (!(*this)[i].active) continue;
            if (fuses_(i))
                xfer.reload_advance((*this)[i]);
            else
                xfer.advance((*this)[i]);
        }
    }

private:
    /// Whether `step` reads register `r` as an operand
    static bool reads_(const step_t &step, const reg_t &r) {
        return (step.src1.has_reg() && step.src1.buffer() == r)
                || (step.src2.has_reg() && step.src2.buffer() == r);
    }

    /// Whether binding `i` reloads and advances in one update
    ///
    /// @note The register stays dirty until its update, so no step at or
    ///     before the binding may read it
    bool fuses_(int i) const {
        const bind_t &b = (*this)[i];
        if (!b.active || !b.is_valid() || b.step.is_none()) return false;
        if (!b.reloads_from_register()) return false;
        for (int j = 0; j <= i; ++j)
            if ((*this)[j].active && reads_((*this)[j].step, b.reg))
                return false;
        return true;
    }
};

} // namespace rvjit

#endif
