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

#ifndef RVJIT_INTERFACE_LOOP_LOOP_HPP
#define RVJIT_INTERFACE_LOOP_LOOP_HPP

#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/induction/step.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/interface/loop/bind.hpp"
#include "rvjit/interface/loop/frame.hpp"
#include "rvjit/interface/loop/plan.hpp"
#include "rvjit/interface/loop/unroll.hpp"
#include "rvjit/interface/resource/resource_optimizable.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace loop {

/// Transaction-covered state of one loop
struct state_t {
    int frame = 0;
    int lent = 0;
    slot_t spill;
};

/// One loop, able to emit and describe itself
///
/// @details The optimizable levels are how much of the loop's own registers
///     it lends to its body: level 0 keeps them private, level 1 lends the
///     reloadable ones, and level 2 also lends the ones needing a spill
///
/// @note Created resolved; `allocate()` draws its resources in the scope the
///     loop is dispatched from, `supplied()` being the registers it holds,
///     lent or not
struct loop_t : public resource_optimizable_t {
    static constexpr const char *NAME = "rvjit::loop";

    virtual ~loop_t() = default;

    /// A loop that is never valid, handed out in place of one that failed
    static loop_t *invalid();

    /// Emits a loop structure around a delegated parametrized code segment
    ///
    /// @param cb The code emitted per dispatch, given the unroll factor
    ///
    /// @pre The loop passes its own `is_valid()` check
    virtual bool generate(const control_flow::on_index_t &cb) {
        (void)cb;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Draws the loop's registers and slots in the current scope
    ///
    /// @note Draws again on every call, so a dropped attempt leaves no
    ///     stale resource behind
    virtual shortfall_t allocate() {
        LOG_FAIL(__func__, "unimplemented");
        return shortfall_t::of(resource_t::integers);
    }

    /// Registers lent to the body at the current level
    ///
    /// @note Free during the whole body: a scope around it may be given them
    virtual buffer::buffer_t released() const { return {}; }

    /// The addressing distance the loop's own branch/jump instructions cover
    virtual control_flow::distance_t distance() const {
        LOG_FAIL(__func__, "unimplemented");
        return control_flow::distance_t::short_;
    }

    /// Changes the addressing distance the loop's branches cover
    ///
    /// @param d The distance
    ///
    /// @note Affects only the instructions emitted afterwards
    virtual bool set_distance(control_flow::distance_t d) {
        (void)d;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// The kind of loop it was planned as
    virtual plan_t::kind_t kind() const {
        LOG_FAIL(__func__, "unimplemented");
        return plan_t::kind_t::data_parallel;
    }

    /// The loop unrolling factors
    ///
    /// @note Even if unrolling is not applied, return 1
    ///
    /// @pre A valid loop has at least one factor, each at least 1
    virtual const unroll_t &unroll() const {
        LOG_FAIL(__func__, "unimplemented");
        static const unroll_t none;
        return none;
    }

    /// Work-units represented by each unrolling-factor count
    virtual int granularity() const {
        LOG_FAIL(__func__, "unimplemented");
        return 1;
    }

    /// The induction frame driven around each body call
    virtual frame_t &frame() {
        LOG_FAIL(__func__, "unimplemented");
        static frame_t none;
        return none;
    }

    /// Whether or not the loop is resolved, its resources aside
    virtual bool is_resolved() const {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether or not the loop is ready to be emitted
    bool is_valid() const override {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether or not the loop body callback is invoked more than once
    virtual bool has_multiple_trips() const {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Whether `remaining()` names a register within the loop body
    ///
    /// @pre A valid loop of `granularity() > 1` returns true
    virtual bool has_remaining() const {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Register holding the loop's own trip count still to be dispatched
    ///
    /// @note Returns the register `zero` if unsupported
    ///
    /// @pre The result is valid only within the loop body
    virtual reg_t remaining() const {
        LOG_FAIL(__func__, "unimplemented");
        return reg_t {};
    }

    /// Captures the state a transaction rolls back
    virtual state_t checkpoint() const {
        LOG_FAIL(__func__, "unimplemented");
        return {};
    }

    /// Reverts to a `checkpoint()` result: drops later bindings and lending
    ///
    /// @param m The state captured
    virtual bool rewind(const state_t &m) {
        (void)m;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }
};

/// The loop `loop_t::invalid()` hands out
struct invalid_loop_t final : public loop_t {
    bool is_resolved() const override { return false; }
    bool is_valid() const override { return false; }
};

inline loop_t *loop_t::invalid() {
    static invalid_loop_t none;
    return &none;
}

} // namespace loop
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
