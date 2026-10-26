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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_CONVERT_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_CONVERT_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

/// Conversion between an integer and a float type
///
/// @details On vectors of one width, dispatches to `vfcvt.f.x[u].v` and
///     `vfcvt.x[u].f.v`; on scalars, to the `fcvt` of the f16, f32 or f64
///     and the 32- or 64-bit integer. Both round as the dynamic rounding mode
///     says
///
/// @pre On vector forms, the caller must issue `vsetvli` prior to calling this
///     expression with the operands' SEW and LMUL
struct convert_t {
    static constexpr const char *NAME = "rvjit::rvv::arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit convert_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op` maps to a supported instruction
    ///
    /// @param op The operation checked
    bool supports(const rvjit::arithmetic::op_t &op) const {
        if (!op.is_converting() || !op.rd.has_valid_width()
                || !op.rs1.has_valid_width())
            return false;
        if (op.rd.is_vector() && op.rs1.is_vector())
            return op.rd.width == op.rs1.width && float_ok_(op);
        if (op.rd.is_scalar() && op.rs1.is_scalar())
            return float_ok_(op) && int_width_(op) >= 4;
        return false;
    }

    bool operator()(const rvjit::arithmetic::op_t &op) const {
        CHECK(op.is_binary(), "expected a binary operand descriptor");
        CHECK(all_valid(op.rd, op.rs1), "invalid operand descriptor");
        CHECK(supports(op), "invalid descriptor");
        if (op.rd.is_vector()) return vector_(op);
        return scalar_(op);
    }

private:
    rv64::emitter_t e_;

    // The float side is f16, f32 or f64; bf16 has no integer conversion
    static bool float_ok_(const rvjit::arithmetic::op_t &op) {
        const rvjit::type_t f = op.rd.is_float() ? op.rd.type : op.rs1.type;
        return f == rvjit::type_t::f16 || f == rvjit::type_t::f32
                || f == rvjit::type_t::f64;
    }

    static int int_width_(const rvjit::arithmetic::op_t &op) {
        return op.rd.is_int() ? op.rd.width : op.rs1.width;
    }

    bool vector_(const rvjit::arithmetic::op_t &op) const {
        const VReg vd(op.rd.name), vs2(op.rs1.name);
        const VM vm = rv64::isa_traits_base_t::vm(op.predicate);
        if (op.rd.is_float()) {
            if (op.rs1.is_signed())
                e_->vfcvt_f_x_v(vd, vs2, vm);
            else
                e_->vfcvt_f_xu_v(vd, vs2, vm);
        } else {
            if (op.rd.is_signed())
                e_->vfcvt_x_f_v(vd, vs2, vm);
            else
                e_->vfcvt_xu_f_v(vd, vs2, vm);
        }
        return true;
    }

    // The unsigned scalar integer is u32: rvjit has no u64
    bool scalar_(const rvjit::arithmetic::op_t &op) const {
        const bool to_float = op.rd.is_float();
        const int fw = to_float ? op.rd.width : op.rs1.width;
        const bool wide = int_width_(op) == 8;
        const bool sgn = to_float ? op.rs1.is_signed() : op.rd.is_signed();
        if (to_float) {
            const FReg fd(op.rd.name);
            const Reg rs(op.rs1.name);
            if (fw == 2)
                wide ? e_->fcvt_h_l(fd, rs)
                     : (sgn ? e_->fcvt_h_w(fd, rs) : e_->fcvt_h_wu(fd, rs));
            else if (fw == 4)
                wide ? e_->fcvt_s_l(fd, rs)
                     : (sgn ? e_->fcvt_s_w(fd, rs) : e_->fcvt_s_wu(fd, rs));
            else
                wide ? e_->fcvt_d_l(fd, rs)
                     : (sgn ? e_->fcvt_d_w(fd, rs) : e_->fcvt_d_wu(fd, rs));
            return true;
        }
        const Reg rd(op.rd.name);
        const FReg fs(op.rs1.name);
        if (fw == 2)
            wide ? e_->fcvt_l_h(rd, fs)
                 : (sgn ? e_->fcvt_w_h(rd, fs) : e_->fcvt_wu_h(rd, fs));
        else if (fw == 4)
            wide ? e_->fcvt_l_s(rd, fs)
                 : (sgn ? e_->fcvt_w_s(rd, fs) : e_->fcvt_wu_s(rd, fs));
        else
            wide ? e_->fcvt_l_d(rd, fs)
                 : (sgn ? e_->fcvt_w_d(rd, fs) : e_->fcvt_wu_d(rd, fs));
        return true;
    }
};

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
