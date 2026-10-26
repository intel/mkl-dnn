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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_BITCAST_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_BITCAST_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

/// General integer-to-float scalar bit-pattern move
///
/// @details Emits `fmv.h.x`, `fmv.w.x` or `fmv.d.x` by the float width
struct bitcast_t {
    static constexpr const char *NAME = "rvjit::rv64::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit bitcast_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` moves an int scalar into a same-width f16/f32/f64 scalar
    ///
    /// @param op The operation checked
    bool supports(const rvjit::arithmetic::op_t &op) const {
        if (!op.is_binary() || !op.rd.is_defined() || !op.rs1.is_defined())
            return false;
        if (!op.rd.is_scalar() || !op.rs1.is_scalar()) return false;
        if (!op.rd.is_float() || !op.rs1.is_int()) return false;
        const rvjit::type_t t = op.rd.type;
        return (t == rvjit::type_t::f16 || t == rvjit::type_t::f32
                       || t == rvjit::type_t::f64)
                && op.rd.width == op.rs1.width;
    }

    bool operator()(const rvjit::arithmetic::op_t &op) const {
        CHECK(op.is_binary(), "expected a binary operand descriptor");
        CHECK(all_valid(op.rd, op.rs1), "invalid operand descriptor");
        CHECK(supports(op), "invalid descriptor");

        const FReg fd(op.rd.name);
        const Reg rs(op.rs1.name);
        switch (op.rd.width) {
            case 2: e_->fmv_h_x(fd, rs); break;
            case 4: e_->fmv_w_x(fd, rs); break;
            default: e_->fmv_d_x(fd, rs); break;
        }
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
