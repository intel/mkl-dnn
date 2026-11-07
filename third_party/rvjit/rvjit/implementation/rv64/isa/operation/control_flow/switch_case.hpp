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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONTROL_FLOW_SWITCH_CASE_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONTROL_FLOW_SWITCH_CASE_HPP

#include "rvjit/implementation/rv64/isa/operation/control_flow/if_else.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace control_flow {

struct switch_case_t {
    static constexpr const char *NAME = "rvjit::rv64::control_flow";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    /// @param xfer The transfer component that loads the operands
    switch_case_t(rvjit::emitter_t e, const operation::transfer_t &xfer)
        : e_(e), if_else_(e, xfer), xfer_(xfer) {}

    bool operator()(int n, const rvjit::variable_t &id, const Reg &t,
            const rvjit::control_flow::on_index_t &cb) const {
        const bool valid = n > 0 && n <= operation::control_flow_t::NCASES
                && is_not_zero(id.buffer());
        CHECK(valid, "invalid parameters");

        // A single case needs no dispatch
        if (n == 1) {
            cb(1);
            return true;
        }

        // Two cases compare against the second one
        if (n == 2) {
            e_->li(t, 2);
            if_else_(rvjit::control_flow::branch_t::eq(id, t), [&] { cb(2); },
                    [&] { cb(1); });
            return true;
        }

        // Load the dispatch value, then compute the jump table entry address
        Label end;
        Label cases[operation::control_flow_t::NCASES];
        const Reg &address = t;
        const Reg &entry = id.buffer();

        if (id.is_valid()) { xfer_.init(id); }

        // Byte offset from auipc to the selected case entry (jump table
        // starts 3 instructions after auipc; entry i maps to case n-i):
        // offset = ((3 + n) * 4) - (id * 4)
        e_->li(address, (3 + n) * sizeof(uint32_t));
        e_->slli(entry, id.buffer(), 2);
        e_->sub(entry, address, entry);

        e_->auipc(address, 0);
        e_->add(address, address, entry);
        e_->jr(address);

        // Emit the jump table, then the case bodies in the same order
        for (int i = n; i > 0; --i)
            e_->j_(cases[i - 1]);

        for (int i = n; i > 0; --i) {
            e_->L(cases[i - 1]);
            cb(i);
            if (i > 1) e_->j_(end);
        }

        e_->L(end);
        return true;
    }

private:
    emitter_t e_;
    if_else_t if_else_;
    const operation::transfer_t &xfer_;
};

} // namespace control_flow
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
