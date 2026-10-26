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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONST_ARITHMETIC_ADD_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONST_ARITHMETIC_ADD_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {
namespace rv64 {
namespace const_arithmetic {

struct add_t {
    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit add_t(rvjit::emitter_t e) : e_(e) {}

    void operator()(const Reg &rd, const Reg &rs1, const rvjit::const_t &c,
            bool noop_zero = true) const {
        if (c.is_zero()) {
            if (!noop_zero) e_->mv(rd, rs1);
            return;
        }
        if (c.has_reg())
            e_->add(rd, rs1, c.buffer());
        else
            e_->addi(rd, rs1, c.imm());
    }

private:
    emitter_t e_;
};

} // namespace const_arithmetic
} // namespace rv64
} // namespace rvjit

#endif
