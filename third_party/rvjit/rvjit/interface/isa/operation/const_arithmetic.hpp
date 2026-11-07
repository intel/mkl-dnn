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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_CONST_ARITHMETIC_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_CONST_ARITHMETIC_HPP

#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace operation {

/// Constant-folded arithmetic expressions
struct const_arithmetic_t {
    static constexpr const char *NAME = "rvjit::operation::const_arithmetic";

    virtual ~const_arithmetic_t() = default;

    /// Constant-folded addition expression, `rd = rs1 + c`
    ///
    /// @param rd The destination register
    /// @param rs1 The source register
    /// @param c The constant added
    /// @param noop_zero Whether to emit nothing when `c` is zero, else a move
    virtual void add(const reg_t &rd, const reg_t &rs1, const const_t &c,
            bool noop_zero = true) const {
        LOG_FAIL(__func__, "unimplemented");
    }

    /// Constant-folded bitwise-and expression, `rd = rs1 & c`
    ///
    /// @param rd The destination register
    /// @param rs1 The source register
    /// @param c The constant masked with
    ///
    /// @note An immediate that does not fit simm12 is read from the register
    ///     `c` supplies
    virtual void and_(
            const reg_t &rd, const reg_t &rs1, const const_t &c) const {
        LOG_FAIL(__func__, "unimplemented");
    }

    /// Constant-folded division expression, `rd = floor(rs1 / c)`
    ///
    /// @param rd The destination register
    /// @param rs1 The dividend register
    /// @param c The constant divisor
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre `rd` must not alias `rs1` unless c is a power of two
    virtual bool div(
            const reg_t &rd, const reg_t &rs1, const const_t &c) const {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Constant-folded multiplication expression, `rd = rs1 * c`
    ///
    /// @param rd The destination register
    /// @param rs1 The multiplicand register
    /// @param c The constant multiplier
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre `rd` must not alias `rs1` unless c is a power of two
    virtual bool mul(
            const reg_t &rd, const reg_t &rs1, const const_t &c) const {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Constant-folded down-alignment expression, `rd = rs1 - (rs1 % c)`
    ///
    /// @param rd The destination register
    /// @param rs1 The register aligned
    /// @param c The constant alignment
    ///
    /// @note Returns false if code emission fails
    ///
    /// @pre `rd` must not alias `rs1` unless c is a power of two
    virtual bool align_down(
            const reg_t &rd, const reg_t &rs1, const const_t &c) const {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }
};

/// Assembles a `const_arithmetic_t` from individual expression implementations
///
/// @details The expression must be constructible from `emitter_t`
template <typename AddOp, typename MulOp, typename DivOp, typename AlignDownOp,
        typename AndOp>
struct const_arithmetic_compose_t : public const_arithmetic_t {

    /// Constructs every expression implementation
    ///
    /// @param e The emitter the expressions emit through
    explicit const_arithmetic_compose_t(emitter_t e)
        : add_(e), mul_(e), div_(e), align_down_(e), and_op_(e) {}

    void add(const reg_t &rd, const reg_t &rs1, const const_t &c,
            bool noop_zero = true) const override {
        add_(rd, rs1, c, noop_zero);
    }

    void and_(const reg_t &rd, const reg_t &rs1,
            const const_t &c) const override {
        and_op_(rd, rs1, c);
    }

    bool div(const reg_t &rd, const reg_t &rs1,
            const const_t &c) const override {
        return div_(rd, rs1, c);
    }

    bool mul(const reg_t &rd, const reg_t &rs1,
            const const_t &c) const override {
        return mul_(rd, rs1, c);
    }

    bool align_down(const reg_t &rd, const reg_t &rs1,
            const const_t &c) const override {
        return align_down_(rd, rs1, c);
    }

protected:
    AddOp add_;
    MulOp mul_;
    DivOp div_;
    AlignDownOp align_down_;
    AndOp and_op_;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
