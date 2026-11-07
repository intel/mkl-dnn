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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONTROL_FLOW_WHILE_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONTROL_FLOW_WHILE_HPP

#include "rvjit/implementation/rv64/isa/operation/control_flow/branch_emitter.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rv64 {
namespace control_flow {

struct while_t : public branch_emitter_t {
    static constexpr const char *NAME = "rvjit::rv64::control_flow";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    /// @param arith The constant arithmetic component
    /// @param xfer The transfer component that loads the operands
    explicit while_t(rvjit::emitter_t e,
            const operation::const_arithmetic_t &arith,
            const operation::transfer_t &xfer)
        : branch_emitter_t(e, xfer), arith_(arith) {}

    void operator()(const rvjit::control_flow::branch_t &b,
            const rvjit::control_flow::on_enter_t &body) const {
        if (!body) return;

        materialize_operands_(b);

        Label head, end;
        e_->L(head);

#if defined(RVJIT_DEBUG)
        const size_t pos = e_.cg().getSize();
#endif

        emit_branch_(b, end);
        body();
        e_->j_(head);
        e_->L(end);

#if defined(RVJIT_DEBUG)
        if (distance_overflowed_(b, pos))
            LOG_FAIL("while", "branch distance overflow");
#endif
    }

    void operator()(const rvjit::control_flow::branch_t &b,
            const rvjit::const_t step,
            const rvjit::control_flow::on_enter_t &cb) const {
        if (!cb) return;

        this->operator()(b, [&]() {
            arith_.add(b.lhs.buffer(), b.lhs.buffer(), step);
            cb();
        });
    }

private:
    const operation::const_arithmetic_t &arith_;
};

} // namespace control_flow
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
