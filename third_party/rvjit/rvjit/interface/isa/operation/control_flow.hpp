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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_CONTROL_FLOW_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_CONTROL_FLOW_HPP

#include "rvjit/interface/core/callback.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace control_flow {

/// Delegate emission of a code segment
using on_enter_t = callback_t<>;

/// Delegate emission of code segment within an indexed region
using on_index_t = callback_t<int>;

/// Branch condition types, comparing the left and right hand sides
enum class cond_t { eq, ne, lt, le, gt, ge };

/// Distance based on the addressing range of control flow instructions
///
/// @note `medium` emits a jump to the taken path, costing one more instruction
enum class distance_t {
    short_, // 12-bit simm (branch)
    medium // 20-bit simm (jump)
};

/// Description of a plan to create a control flow branch/jump
struct branch_t {
    variable_t lhs, rhs;
    cond_t cond;
    distance_t dist;

    branch_t() = default;

    /// Creates a branch comparing two variables
    ///
    /// @param lhs The left hand side
    /// @param rhs The right hand side
    /// @param cond The condition under which the branch is taken
    /// @param dist The reach of the branches around the guarded code
    branch_t(variable_t lhs, variable_t rhs, cond_t cond,
            distance_t dist = distance_t::short_)
        : lhs(std::move(lhs)), rhs(std::move(rhs)), cond(cond), dist(dist) {}

    /// Creates a branch comparing a variable against the zero register
    ///
    /// @param lhs The left hand side
    /// @param cond The condition under which the branch is taken
    /// @param dist The reach of the branches around the guarded code
    branch_t(variable_t lhs, cond_t cond, distance_t dist = distance_t::short_)
        : lhs(std::move(lhs)), rhs(reg_t::zero()), cond(cond), dist(dist) {}

    /// Creates a branch taken when `lhs` is equal to `rhs`
    ///
    /// @param lhs The left hand side
    /// @param rhs The right hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t eq(variable_t lhs, variable_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), std::move(rhs), cond_t::eq, dist);
    }

    /// Creates a branch taken when `lhs` is not equal to `rhs`
    ///
    /// @param lhs The left hand side
    /// @param rhs The right hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t ne(variable_t lhs, variable_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), std::move(rhs), cond_t::ne, dist);
    }

    /// Creates a branch taken when `lhs` is less than to `rhs`
    ///
    /// @param lhs The left hand side
    /// @param rhs The right hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t lt(variable_t lhs, variable_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), std::move(rhs), cond_t::lt, dist);
    }

    /// Creates a branch taken when `lhs` is less than or equal to `rhs`
    ///
    /// @param lhs The left hand side
    /// @param rhs The right hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t le(variable_t lhs, variable_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), std::move(rhs), cond_t::le, dist);
    }

    /// Creates a branch taken when `lhs` is greater than to `rhs`
    ///
    /// @param lhs The left hand side
    /// @param rhs The right hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t gt(variable_t lhs, variable_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), std::move(rhs), cond_t::gt, dist);
    }

    /// Creates a branch taken when `lhs` is greater than or equal to `rhs`
    ///
    /// @param lhs The left hand side
    /// @param rhs The right hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t ge(variable_t lhs, variable_t rhs,
            distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), std::move(rhs), cond_t::ge, dist);
    }

    /// Creates a branch taken when `lhs` is zero
    ///
    /// @param lhs The left hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t eqz(variable_t lhs, distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), cond_t::eq, dist);
    }

    /// Creates a branch taken when `lhs` is not zero
    ///
    /// @param lhs The left hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t nez(variable_t lhs, distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), cond_t::ne, dist);
    }

    /// Creates a branch taken when `lhs` is negative
    ///
    /// @param lhs The left hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t ltz(variable_t lhs, distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), cond_t::lt, dist);
    }

    /// Creates a branch taken when `lhs` is not positive
    ///
    /// @param lhs The left hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t lez(variable_t lhs, distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), cond_t::le, dist);
    }

    /// Creates a branch taken when `lhs` is positive
    ///
    /// @param lhs The left hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t gtz(variable_t lhs, distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), cond_t::gt, dist);
    }

    /// Creates a branch taken when `lhs` is not negative
    ///
    /// @param lhs The left hand side
    /// @param dist The reach of the branches around the guarded code
    static branch_t gez(variable_t lhs, distance_t dist = distance_t::short_) {
        return branch_t(std::move(lhs), cond_t::ge, dist);
    }
};

} // namespace control_flow
} // namespace rvjit

namespace rvjit {
namespace operation {

/// Control-flow expressions
struct control_flow_t {
    using branch_t = control_flow::branch_t;
    using on_enter_t = control_flow::on_enter_t;

    static constexpr const char *NAME = "rvjit::operation::control_flow";

    /// The maximum case count `switch_case` can dispatch over
    static constexpr int NCASES = 32;

    virtual ~control_flow_t() = default;

    /// if (b.lhs <b.cond> b.rhs) { taken(); }
    ///
    /// @param b The branch condition
    /// @param taken The code emitted for the taken path
    ///
    /// @note Loads `b.rhs` if `b.rhs.is_valid()`
    /// @note Loads `b.lhs` if `b.lhs.is_valid()`
    /// @note Emits nothing if `taken` is nullptr
    virtual void if_(const branch_t &b, const on_enter_t &taken) const {
        (void)b;
        (void)taken;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// if (b.lhs <b.cond> b.rhs) { t(); } else { f(); }
    ///
    /// @param b The branch condition
    /// @param t The code emitted for the taken path
    /// @param f The code emitted for the not-taken path
    ///
    /// @note Loads `b.rhs` if `b.rhs.is_valid()`
    /// @note Loads `b.lhs` if `b.lhs.is_valid()`
    /// @note Emits nothing for a side with a null callback
    virtual void if_else(
            const branch_t &b, const on_enter_t &t, const on_enter_t &f) const {
        (void)b;
        (void)t;
        (void)f;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// while (b.lhs <b.cond> b.rhs) { body(); }
    ///
    /// @param b The loop condition
    /// @param body The code emitted for each iteration
    ///
    /// @note Loads `b.rhs` if `b.rhs.is_valid()`
    /// @note Loads `b.lhs` if `b.lhs.is_valid()`
    /// @note Emits nothing if `body == nullptr`
    ///
    /// @pre Caller must advance the iterator (`b.lhs`) on `body`
    virtual void while_(const branch_t &b, const on_enter_t &body) const {
        (void)b;
        (void)body;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// while (b.lhs <b.cond> b.rhs) { b.lhs += step; body(); }
    ///
    /// @param b The loop condition
    /// @param step The constant added to `b.lhs` before each iteration
    /// @param body The code emitted for each iteration
    ///
    /// @note Loads `b.rhs` if `b.rhs.is_valid()`
    /// @note Loads `b.lhs` if `b.lhs.is_valid()`
    /// @note Emits nothing if `body == nullptr`
    /// @note Clobbers `b.lhs` by applying the loop step
    ///
    /// @details The loop step update happens before the `body` callback
    ///
    /// @pre Caller must ensure `step` drives the loop until its completion
    virtual void while_(const branch_t &b, const const_t step,
            const on_enter_t &body) const {
        (void)b;
        (void)step;
        (void)body;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// switch(id) { case(n-1): { cb(n-1) break; } ... case(1): cb(1); }
    ///
    /// @param n The number of cases, at most `NCASES`
    /// @param id The runtime value selecting the case, within [1, n]
    /// @param tmp The scratch register
    /// @param body The code emitted for each case
    ///
    /// @details When n == 1, emits only `body(1)`
    /// @details When n == 2, emits an if/else between cases 1 and 2
    /// @details When n >= 3, emits a jump table dispatch mechanism
    ///
    /// @note Clobbers `id` and `tmp`, except when n == 1
    /// @note Loads `id` if `id.is_valid()`, except when n == 1
    ///
    /// @pre The value in `id` must fall within [1, n]
    virtual bool switch_case(int n, const variable_t &id, const reg_t &tmp,
            const control_flow::on_index_t &body) const {
        (void)n;
        (void)id;
        (void)tmp;
        (void)body;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }
};

/// Assembles a `control_flow_t` from individual expression implementations
///
/// @details All expressions must be constructible from `emitter_t` and the
///     `while` expressions also takes a `const_arithmetic_t`
template <typename IfOp, typename IfElseOp, typename WhileOp,
        typename SwitchCaseOp>
struct control_flow_compose_t : public control_flow_t {

    /// Constructs every expression implementation
    ///
    /// @param e The emitter the expressions emit through
    /// @param arith The constant arithmetic the `while` expressions use
    /// @param xfer The value transfer that loads the branch operands
    control_flow_compose_t(emitter_t e, const const_arithmetic_t &arith,
            const transfer_t &xfer)
        : if_op_(e, xfer)
        , if_else_op_(e, xfer)
        , while_op_(e, arith, xfer)
        , switch_case_op_(e, xfer) {}

    void if_(const branch_t &b, const on_enter_t &taken) const override {
        if_op_(b, taken);
    }

    void if_else(const branch_t &b, const on_enter_t &t,
            const on_enter_t &f) const override {
        if_else_op_(b, t, f);
    }

    void while_(const branch_t &b, const on_enter_t &body) const override {
        while_op_(b, body);
    }

    void while_(const branch_t &b, const const_t step,
            const on_enter_t &cb) const override {
        while_op_(b, step, cb);
    }

    bool switch_case(int n, const variable_t &id, const reg_t &t,
            const control_flow::on_index_t &cb) const override {
        return switch_case_op_(n, id, t, cb);
    }

protected:
    IfOp if_op_;
    IfElseOp if_else_op_;
    WhileOp while_op_;
    SwitchCaseOp switch_case_op_;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
