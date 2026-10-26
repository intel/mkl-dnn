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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_ARITHMETIC_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_ARITHMETIC_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace arithmetic {

using operand_t = rvjit::operand_t;

/// Generic arithmetic instruction involving up to three operands
struct op_t : public typed_t {
    operand_t rd = absent_();
    operand_t rs1 = absent_();
    operand_t rs2 = absent_();
    predicate_t predicate = predicate_t::unmasked;

    op_t() = default;

    /// Creates a unary operation
    ///
    /// @param dst The destination operand
    explicit op_t(const operand_t &dst) : rd(dst) {}

    /// Creates a ternary operation
    ///
    /// @param dst The destination operand
    /// @param src1 The first source operand
    /// @param src2 The second source operand
    op_t(const operand_t &dst, const operand_t &src1, const operand_t &src2)
        : rd(dst), rs1(src1), rs2(src2) {}

    /// Creates a ternary operation from data types, with no register names
    ///
    /// @param dst The destination data type
    /// @param src1 The first source data type
    /// @param src2 The second source data type
    /// @param form The storage category of every operand
    op_t(type_t dst, type_t src1, type_t src2,
            rvjit::form_t form = rvjit::form_t::undefined)
        : rd(form, dst), rs1(form, src1), rs2(form, src2) {}

    /// Number of present operands
    int size() const {
        if (!is_present_(rd)) return 0;
        if (!is_present_(rs1)) return 1;
        return is_present_(rs2) ? 3 : 2;
    }

    bool is_unary() const { return size() == 1; }
    bool is_binary() const { return size() == 2; }
    bool is_ternary() const { return size() == 3; }
    bool is_scalar() const override { return rd.is_scalar(); }
    bool is_vector() const override { return rd.is_vector(); }
    bool is_int() const override { return rd.is_int(); }
    bool is_float() const override { return rd.is_float(); }
    bool is_signed() const override { return rd.is_signed(); }

    /// Whether every present operand has the same width
    bool is_uniform() const {
        return is_unary() || (rd.width == rs1.width && rs1.width == rs2.width);
    }

    /// Whether the destination is twice as wide as both equally wide sources
    bool is_widening() const {
        return !is_unary() && rd.width == 2 * rs1.width
                && rs1.width == rs2.width;
    }

    /// Whether the destination is half as wide as the only source
    bool is_narrowing() const {
        return is_binary() && 2 * rd.width == rs1.width;
    }

    /// Whether the destination is twice as wide as the only source
    bool is_extending() const {
        return is_binary() && rd.width == 2 * rs1.width;
    }

    /// Whether the destination is 2, 4 or 8 times as wide as the only source
    bool is_extending_any() const {
        if (!is_binary() || rs1.width <= 0) return false;
        const int ratio = rd.width / rs1.width;
        return rd.width % rs1.width == 0
                && (ratio == 2 || ratio == 4 || ratio == 8);
    }

    /// Whether the only source changes between integer and float
    bool is_converting() const {
        return is_binary() && rd.is_float() != rs1.is_float();
    }

    /// A copy with the destination register `d`
    ///
    /// @param d The destination register name
    op_t names(name_t d) const {
        op_t o = *this;
        o.rd.name = d;
        return o;
    }

    /// A copy with the destination `d` and first source `s1` registers
    ///
    /// @param d The destination register name
    /// @param s1 The first source register name
    op_t names(name_t d, name_t s1) const {
        op_t o = names(d);
        o.rs1.name = s1;
        return o;
    }

    /// A copy with the destination `d` and source `s1`, `s2` registers
    ///
    /// @param d The destination register name
    /// @param s1 The first source register name
    /// @param s2 The second source register name
    op_t names(name_t d, name_t s1, name_t s2) const {
        op_t o = names(d, s1);
        o.rs2.name = s2;
        return o;
    }

    /// Sets every operand to the scalar form
    op_t &scalar() {
        rd.form = rs1.form = rs2.form = rvjit::form_t::scalar;
        return *this;
    }

    /// Sets every operand to the vector form
    ///
    /// @param vm The vector mask policy
    op_t &vector(predicate_t vm = predicate_t::unmasked) {
        rd.form = rs1.form = rs2.form = rvjit::form_t::vector;
        predicate = vm;
        return *this;
    }

    /// Sets the first source to the scalar form and the rest to the vector form
    ///
    /// @param vm The vector mask policy
    op_t &vector_scalar(predicate_t vm = predicate_t::unmasked) {
        rd.form = rvjit::form_t::vector;
        rs1.form = rvjit::form_t::scalar;
        rs2.form = rvjit::form_t::vector;
        predicate = vm;
        return *this;
    }

    /// Creates a ternary operation whose operands all have the type `dt`
    ///
    /// @param dt The data type of every operand
    /// @param form The storage category of every operand
    static op_t uniform(type_t dt, rvjit::form_t form) {
        const operand_t o(form, dt);
        return op_t(o, o, o);
    }

    /// Creates a binary operation converting `src` into `dst`
    ///
    /// @param dst The data type of the destination
    /// @param src The data type of the source
    /// @param form The storage category of both operands
    static op_t convert(type_t dst, type_t src, rvjit::form_t form) {
        return op_t(operand_t(form, dst), operand_t(form, src), absent_());
    }

    /// Creates a ternary operation whose sources are `dt` and destination
    /// is the natural 2x-wide type of `dt`
    ///
    /// @param dt The data type of both sources
    /// @param form The storage category of every operand
    static op_t widening(type_t dt, rvjit::form_t form) {
        const operand_t wide(form, natural_wide(dt));
        const operand_t narrow(form, dt);
        return op_t(wide, narrow, narrow);
    }

private:
    static operand_t absent_() {
        return operand_t(rvjit::form_t::undefined, type_t::none);
    }
    static bool is_present_(const operand_t &o) {
        return o.type != type_t::none;
    }
};

} // namespace arithmetic
} // namespace rvjit

namespace rvjit {
namespace operation {

/// Arithmetic expressions
struct arithmetic_t {
    static constexpr const char *NAME = "rvjit::operation::arithmetic";

    virtual ~arithmetic_t() = default;

    /// Multiply-accumulate expression
    ///
    /// @param op The operation, `rd += rs1 * rs2`
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_ternary()` expression must be true
    /// @pre Caller must supply operand combinations supported by the ISA
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    virtual bool macc(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `macc` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_macc(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `zero` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_zero(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `mul` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_mul(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `add` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_add(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `bitcast` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_bitcast(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `narrow` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_narrow(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `widen` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_widen(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `extend` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_extend(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `convert` can emit `op`, ignoring the operands' register names
    ///
    /// @param op The operation checked
    virtual bool supports_convert(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Integer extension of `rs1` into a type 2, 4 or 8 times as wide
    ///
    /// @param op The operation, `rd = extend(rs1)`, by the sign of a signed
    ///     source and by zeros otherwise
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_extending_any()` expression is true
    /// @pre On vector forms, the vtype is configured with the destination's
    ///     SEW and LMUL, unlike `widen`
    /// @pre On vector forms, a source overlapping the destination group must
    ///     be its highest-numbered part
    virtual bool extend(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Conversion of `rs1` between an integer and a float type
    ///
    /// @param op The operation, `rd = convert(rs1)`, rounding as the dynamic
    ///     rounding mode says (round to nearest even unless changed)
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_converting()` expression is true; on vector forms both
    ///     operands have the same width
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    /// @pre On vector forms, `rd` may name `rs1`, converting in place
    virtual bool convert(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Conversion of `rs1` into a type half as wide
    ///
    /// @param op The operation, `rd = narrow(rs1)`; integers saturate
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_narrowing()` expression is true
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    /// @pre `rd` may name `rs1`, narrowing in place
    virtual bool narrow(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Conversion of `rs1` into a type twice as wide
    ///
    /// @param op The operation, `rd = widen(rs1)`
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_extending()` expression is true
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    /// @pre On vector forms, `rd` is not `rs1`: a source overlapping the
    ///     destination group must be its highest-numbered part
    virtual bool widen(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Multiplication expression
    ///
    /// @param op The operation, `rd = rs1 * rs2`
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_ternary()` and `op.is_uniform()` expressions are true
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    virtual bool mul(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Addition expression
    ///
    /// @param op The operation, `rd = rs1 + rs2`
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_ternary()` and `op.is_uniform()` expressions are true
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    virtual bool add(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Bit-pattern move from an integer scalar into a float scalar
    ///
    /// @param op The operation, from `rs1` into `rd`
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_binary()` expression is true and both widths match
    virtual bool bitcast(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Clear buffer expression
    ///
    /// @param op The operation, clearing `rd`
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre The `op.is_unary()` expression must be true
    /// @pre On vector forms, the caller must manage the ISA state (e.g. vtype)
    virtual bool zero(const arithmetic::op_t &op) const {
        (void)op;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }
};

/// Assembles an `arithmetic_t` from individual expression implementations
///
/// @details The expression must be constructible from `emitter_t`
template <typename MaccOp, typename ZeroOp, typename MulOp, typename AddOp,
        typename BitcastOp, typename NarrowOp, typename WidenOp,
        typename ExtendOp, typename ConvertOp>
struct arithmetic_compose_t : public arithmetic_t {
    /// Constructs every expression implementation
    ///
    /// @param e The emitter the expressions emit through
    explicit arithmetic_compose_t(emitter_t e)
        : macc_(e)
        , zero_(e)
        , mul_(e)
        , add_(e)
        , bitcast_(e)
        , narrow_(e)
        , widen_(e)
        , extend_(e)
        , convert_(e) {}

    bool supports_macc(const arithmetic::op_t &op) const override {
        return macc_.supports(op);
    }
    bool supports_zero(const arithmetic::op_t &op) const override {
        return zero_.supports(op);
    }
    bool supports_mul(const arithmetic::op_t &op) const override {
        return mul_.supports(op);
    }
    bool supports_add(const arithmetic::op_t &op) const override {
        return add_.supports(op);
    }
    bool supports_bitcast(const arithmetic::op_t &op) const override {
        return bitcast_.supports(op);
    }
    bool supports_narrow(const arithmetic::op_t &op) const override {
        return narrow_.supports(op);
    }
    bool supports_widen(const arithmetic::op_t &op) const override {
        return widen_.supports(op);
    }
    bool narrow(const arithmetic::op_t &op) const override {
        return narrow_(op);
    }
    bool widen(const arithmetic::op_t &op) const override { return widen_(op); }
    bool supports_extend(const arithmetic::op_t &op) const override {
        return extend_.supports(op);
    }
    bool supports_convert(const arithmetic::op_t &op) const override {
        return convert_.supports(op);
    }
    bool extend(const arithmetic::op_t &op) const override {
        return extend_(op);
    }
    bool convert(const arithmetic::op_t &op) const override {
        return convert_(op);
    }
    bool macc(const arithmetic::op_t &op) const override { return macc_(op); }
    bool zero(const arithmetic::op_t &op) const override { return zero_(op); }
    bool mul(const arithmetic::op_t &op) const override { return mul_(op); }
    bool add(const arithmetic::op_t &op) const override { return add_(op); }
    bool bitcast(const arithmetic::op_t &op) const override {
        return bitcast_(op);
    }

protected:
    MaccOp macc_;
    ZeroOp zero_;
    MulOp mul_;
    AddOp add_;
    BitcastOp bitcast_;
    NarrowOp narrow_;
    WidenOp widen_;
    ExtendOp extend_;
    ConvertOp convert_;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
