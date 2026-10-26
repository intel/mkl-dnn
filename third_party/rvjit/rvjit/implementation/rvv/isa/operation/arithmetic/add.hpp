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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_ADD_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_ADD_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// General vector add instruction
///
/// @details Dispatches to `vadd_vv`/`vadd_vx` on int vector forms, or
///     `vfadd_vv`/`vfadd_vf` on float vector forms
///
/// @note The scalar operand, if any, is `rs1`
///
/// @pre The caller must issue `vsetvli` prior to calling this expression
struct add_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit add_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` maps to a supported instruction
    ///
    /// @param op The operation checked
    bool supports(const rvjit::arithmetic::op_t &op) const {
        if (!op.is_ternary() || !op.rd.is_vector() || !op.is_uniform())
            return false;
        if (!op.rd.is_defined() || !op.rs1.is_defined() || !op.rs2.is_defined())
            return false;
        if (!op.rd.has_valid_width() || !op.rs2.is_vector()) return false;
        if (op.rd.type != op.rs1.type || op.rd.type != op.rs2.type)
            return false;
        if (op.rd.is_int()) return true;
        const rvjit::type_t t = op.rd.type;
        return t == rvjit::type_t::f16 || t == rvjit::type_t::f32
                || t == rvjit::type_t::f64;
    }

    bool operator()(const rvjit::arithmetic::op_t &op) const {
        CHECK(op.is_ternary(), "expected a ternary operand descriptor");
        CHECK(all_valid(op.rd, op.rs1, op.rs2), "invalid operand descriptor");
        CHECK(supports(op), "invalid descriptor");

        const VReg vd(op.rd.name), vs2(op.rs2.name);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        if (op.rd.is_int()) {
            if (op.rs1.is_scalar())
                e_->vadd_vx(vd, vs2, reg_t(op.rs1.name), vm);
            else
                e_->vadd_vv(vd, vs2, vreg_t(op.rs1.name), vm);
        } else {
            if (op.rs1.is_scalar())
                e_->vfadd_vf(vd, vs2, freg_t(op.rs1.name), vm);
            else
                e_->vfadd_vv(vd, vs2, vreg_t(op.rs1.name), vm);
        }
        return true;
    }

private:
    rv64::emitter_t e_;
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
