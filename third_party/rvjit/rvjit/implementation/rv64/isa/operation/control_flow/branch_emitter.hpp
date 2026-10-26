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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONTROL_FLOW_BRANCH_EMITTER_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONTROL_FLOW_BRANCH_EMITTER_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/types.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rv64 {
namespace control_flow {

/// Shared RV64 branch-instruction emission primitives for control-flow op
/// components that emit a conditional branch
struct branch_emitter_t {
    static constexpr const char *NAME = "rvjit::rv64::control_flow";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    /// @param xfer The transfer component that loads the operands
    branch_emitter_t(rvjit::emitter_t e, const operation::transfer_t &xfer)
        : e_(e), xfer_(xfer) {}

protected:
    emitter_t e_;
    const operation::transfer_t &xfer_;

    /// Loads `b.lhs` and `b.rhs` through the transfer component if needed
    void materialize_operands_(const rvjit::control_flow::branch_t &b) const {
        if (b.lhs.is_valid()) xfer_.init(b.lhs);
        if (b.rhs.is_valid()) xfer_.init(b.rhs);
    }

    /// Emits a control flow instruction described by `b`
    ///
    /// @pre Caller has already materialized the operands of `b`
    void emit_branch_(const rvjit::control_flow::branch_t &b,
            const Label &not_taken) const {
        const Reg &lhs = b.lhs.buffer();
        const Reg &rhs = b.rhs.buffer();
        const rvjit::control_flow::cond_t &c = b.cond;
        const rvjit::control_flow::distance_t &d = b.dist;

        using cg_t = rvjit::codegen_t;
        using fn_t = void (cg_t::*)(const Reg &, const Reg &, const Label &);

        // Organized in the same order elements are declared
        static constexpr fn_t branch_table[] = {
                &cg_t::beq,
                &cg_t::bne,
                &cg_t::blt,
                &cg_t::ble,
                &cg_t::bgt,
                &cg_t::bge,
        };

        // Emit a condition-agnostic branch instruction
        cg_t &cg = e_.cg();
        const auto emit = [&](const rvjit::control_flow::cond_t &cond,
                                  const Label &lbl) {
            (cg.*branch_table[static_cast<int>(cond)])(lhs, rhs, lbl);
        };

        // Get the reverse branching condition
        const auto inverse = [&]() {
            switch (c) {
                case rvjit::control_flow::cond_t::eq:
                    return rvjit::control_flow::cond_t::ne;
                case rvjit::control_flow::cond_t::ne:
                    return rvjit::control_flow::cond_t::eq;
                case rvjit::control_flow::cond_t::lt:
                    return rvjit::control_flow::cond_t::ge;
                case rvjit::control_flow::cond_t::ge:
                    return rvjit::control_flow::cond_t::lt;
                case rvjit::control_flow::cond_t::le:
                    return rvjit::control_flow::cond_t::gt;
                case rvjit::control_flow::cond_t::gt:
                    return rvjit::control_flow::cond_t::le;
                default: return rvjit::control_flow::cond_t::ne;
            }
        };

        if (d == rvjit::control_flow::distance_t::short_) {
            // emits the inverse condition as a single branch instruction
            emit(inverse(), not_taken);
        } else {
            // Use jump when "taken" path is further than the 4 KB branch range
            Label taken;
            emit(c, taken);
            e_->j_(not_taken);
            e_->L(taken);
        }
    }

    /// Whether the branch instructions emitted since `pos_before` overflows
    bool distance_overflowed_(
            const rvjit::control_flow::branch_t &b, size_t pos_before) const {
        const size_t nbytes = e_.cg().getSize() - pos_before;
        const size_t limit = b.dist == rvjit::control_flow::distance_t::short_
                ? (size_t(1) << 12)
                : (size_t(1) << 20);
        return nbytes >= limit;
    }
};

} // namespace control_flow
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
