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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_ZERO_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_ZERO_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

struct zero_t {
    static constexpr const char *NAME = "rvjit::rv64::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit zero_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` is a unary scalar descriptor
    ///
    /// @param op The operation checked
    bool supports(const rvjit::arithmetic::op_t &op) const {
        return op.is_unary() && op.rd.is_defined() && op.rd.is_scalar();
    }

    bool operator()(const rvjit::arithmetic::op_t &op) const {
        CHECK(op.is_unary(), "expected a unary operand descriptor");
        CHECK(op.rd.is_valid(), "invalid operand descriptor");
        CHECK(supports(op), "vector zero not supported");

        if (op.rd.is_int())
            e_->mv(reg_t(op.rd.name), zero);
        else
            e_->fmv_d_x(freg_t(op.rd.name), zero);
        return true;
    }

private:
    emitter_t e_;
};

} // namespace arithmetic
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
