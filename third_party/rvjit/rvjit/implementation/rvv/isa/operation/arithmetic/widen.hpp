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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_WIDEN_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_WIDEN_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// Vector conversion into a type twice as wide
///
/// @details Dispatches to `vfwcvt_f_f_v` (f16 to f32, f32 to f64),
///     `vfwcvtbf16_f_f_v` (bf16 to f32), and `vwadd_vx` / `vwaddu_vx` of the
///     zero register on integers, which extend the sign or the zero bits
///
/// @pre The caller must issue `vsetvli` prior to calling this expression and
///     the `vtype` must be configured using the source's SEW and LMUL
/// @pre The source group is the highest-numbered part of the destination
///     group, or does not overlap it
struct widen_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit widen_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` maps to a supported instruction
    ///
    /// @param op The operation checked
    bool supports(const rvjit::arithmetic::op_t &op) const {
        return select_(op) != sel_t::none;
    }

    bool operator()(const rvjit::arithmetic::op_t &op) const {
        CHECK(op.is_binary(), "expected a binary operand descriptor");
        CHECK(all_valid(op.rd, op.rs1), "invalid operand descriptor");
        const sel_t sel = select_(op);
        CHECK(sel != sel_t::none, "invalid descriptor");

        const VReg vd(op.rd.name), vs2(op.rs1.name);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        switch (sel) {
            case sel_t::vfwcvt: e_->vfwcvt_f_f_v(vd, vs2, vm); break;
            case sel_t::vfwcvtbf16: e_->vfwcvtbf16_f_f_v(vd, vs2, vm); break;
            case sel_t::vwadd: e_->vwadd_vx(vd, vs2, zero, vm); break;
            case sel_t::vwaddu: e_->vwaddu_vx(vd, vs2, zero, vm); break;
            default: return false;
        }
        return true;
    }

private:
    rv64::emitter_t e_;

    enum class sel_t { none, vfwcvt, vfwcvtbf16, vwadd, vwaddu };

    static sel_t select_(const rvjit::arithmetic::op_t &op) {
        if (!op.is_extending() || !op.rd.is_vector() || !op.rs1.is_vector())
            return sel_t::none;
        if (!op.rd.has_valid_width() || !op.rs1.has_valid_width())
            return sel_t::none;

        const rvjit::type_t d = op.rd.type, s = op.rs1.type;
        if (op.rd.is_float() && op.rs1.is_float()) {
            if (s == rvjit::type_t::f16 && d == rvjit::type_t::f32)
                return sel_t::vfwcvt;
            if (s == rvjit::type_t::f32 && d == rvjit::type_t::f64)
                return sel_t::vfwcvt;
            if (s == rvjit::type_t::bf16 && d == rvjit::type_t::f32)
                return sel_t::vfwcvtbf16;
            return sel_t::none;
        }
        if (!op.rd.is_int() || !op.rs1.is_int()) return sel_t::none;

        // A signed source needs a signed destination; an unsigned source
        // extends with zeros into either
        if (op.rs1.is_signed())
            return op.rd.is_signed() ? sel_t::vwadd : sel_t::none;
        return sel_t::vwaddu;
    }
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
