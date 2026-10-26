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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_NARROW_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_NARROW_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// Vector conversion into a type half as wide
///
/// @details Dispatches to `vfncvt_f_f_w` (f64 to f32, f32 to f16),
///     `vfncvtbf16_f_f_w` (f32 to bf16), and the saturating `vnclip_wi` /
///     `vnclipu_wi` with a zero shift on integers of the same signedness
///
/// @note Floats round to nearest even under the default `frm`
///
/// @pre The caller must issue `vsetvli` prior to calling this expression and
///     the `vtype` must be configured using the destination's SEW and LMUL
/// @pre `rd` is `rs1` or does not overlap the source group, which is twice
///     as long as the destination's: in place, the destination takes the
///     lowest-numbered part of the source
struct narrow_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit narrow_t(rvjit::emitter_t e) : e_(e) {}

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
            case sel_t::vfncvt: e_->vfncvt_f_f_w(vd, vs2, vm); break;
            case sel_t::vfncvtbf16: e_->vfncvtbf16_f_f_w(vd, vs2, vm); break;
            case sel_t::vnclip: e_->vnclip_wi(vd, vs2, 0, vm); break;
            case sel_t::vnclipu: e_->vnclipu_wi(vd, vs2, 0, vm); break;
            default: return false;
        }
        return true;
    }

private:
    rv64::emitter_t e_;

    enum class sel_t { none, vfncvt, vfncvtbf16, vnclip, vnclipu };

    static sel_t select_(const rvjit::arithmetic::op_t &op) {
        if (!op.is_narrowing() || !op.rd.is_vector() || !op.rs1.is_vector())
            return sel_t::none;
        if (!op.rd.has_valid_width() || !op.rs1.has_valid_width())
            return sel_t::none;

        const rvjit::type_t d = op.rd.type, s = op.rs1.type;
        if (op.rd.is_float() && op.rs1.is_float()) {
            if (s == rvjit::type_t::f64 && d == rvjit::type_t::f32)
                return sel_t::vfncvt;
            if (s == rvjit::type_t::f32 && d == rvjit::type_t::f16)
                return sel_t::vfncvt;
            if (s == rvjit::type_t::f32 && d == rvjit::type_t::bf16)
                return sel_t::vfncvtbf16;
            return sel_t::none;
        }
        if (!op.rd.is_int() || !op.rs1.is_int()) return sel_t::none;

        // The clip saturates to the destination, so the signedness must match
        if (op.rd.is_signed() != op.rs1.is_signed()) return sel_t::none;
        return op.rs1.is_signed() ? sel_t::vnclip : sel_t::vnclipu;
    }
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
