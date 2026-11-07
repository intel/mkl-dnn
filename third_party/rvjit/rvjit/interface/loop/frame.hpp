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

#ifndef RVJIT_INTERFACE_LOOP_FRAME_HPP
#define RVJIT_INTERFACE_LOOP_FRAME_HPP

#include "rvjit/interface/induction/induction.hpp"
#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/induction/step.hpp"
#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/buffer/stack.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/interface/loop/bind.hpp"
#include "rvjit/interface/resource/resource.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace loop {

/// Variables saved, restored, and advanced together around a body
///
/// @details `bind()` records a request and `allocate()` draws the slots the
///     requests need in the current scope, again on every allocation attempt
///
/// @note A loop drives its frame around each body call; a standalone frame is
///     driven once around a phase of code
struct frame_t {
    static constexpr const char *NAME = "rvjit::loop::frame";

    /// Records the binding of `v` requested by `spec`
    ///
    /// @param v The variable bound
    /// @param spec The request
    ///
    /// @note Binding the same variable twice replaces the previous request
    ///
    /// @pre `v` has a buffer
    bool bind(const variable_t &v, const bind_t &spec) {
        CHECK(v.has_buffer(), "a variable without a buffer");
        rvjit::bind_t b;
        b.reg = v.buffer();
        b.origin = v.source();
        b.step = spec.step;
        b.active = false;
        const int i = find_(b.reg);
        if (i >= 0) {
            binds_[i] = b;
            specs_[i] = spec;
            return true;
        }
        CHECK(binds_.push_back(b) && specs_.push_back(spec), "a full frame");
        return true;
    }

    /// Sets the step `v` is advanced by after each body
    ///
    /// @param v The variable bound
    /// @param step The step
    ///
    /// @pre `v` is bound to the frame
    bool set_step(const variable_t &v, const step_t &step) {
        const int i = find_(v.buffer());
        CHECK(i >= 0, "a variable not bound to the frame");
        binds_[i].step = step;
        specs_[i].step = step;
        return true;
    }

    /// Draws the slots of the recorded bindings in the current scope
    ///
    /// @param pool The pool the register slots are drawn from
    /// @param stack The stack the memory slots are drawn from
    /// @param multi_trip Whether the body runs more than once
    ///
    /// @note A binding is active unless it is `off`, or `multi_trip` on a
    ///     body run once
    shortfall_t allocate(
            buffer::buffer_pool_t &pool, stack_t &stack, bool multi_trip) {
        using storage_t = bind_t::storage_t;
        const auto takes_part = [&](const bind_t &spec) {
            return spec.storage != storage_t::off
                    && (spec.when == bind_t::when_t::always || multi_trip);
        };
        for (int i = 0; i < size(); ++i) {
            rvjit::bind_t &b = binds_[i];
            const bind_t &spec = specs_[i];
            b.slot = slot_t {};
            b.active = takes_part(spec);
            if (!b.active) continue;
            switch (spec.storage) {
                case storage_t::stack: b.slot = stack.new_slot(); break;
                case storage_t::buffer: {
                    const auto r = pool.integers().take(1);
                    if (!r) {
                        // Every buffer binding left misses its register
                        int rest = 0;
                        for (int j = i; j < size(); ++j) {
                            const bool is_buffer
                                    = specs_[j].storage == storage_t::buffer;
                            rest += is_buffer && takes_part(specs_[j]);
                        }
                        return shortfall_t::of(resource_t::integers,
                                rest - pool.integers().available());
                    }
                    b.slot = slot_t::buffer(reg_t(r->id(0)));
                    break;
                }
                case storage_t::given: b.slot = spec.slot; break;
                case storage_t::none:
                case storage_t::off: break;
            }
            CHECK_OR_RET(spec.storage == storage_t::none || b.slot.is_valid(),
                    shortfall_t::of(resource_t::stack), "an invalid slot");
        }
        return {};
    }

    /// The registers of the listed variables the frame reloads after its body
    ///
    /// @param v The first variable
    /// @param vs The remaining variables
    ///
    /// @note Free from the last read of their variable until the body ends
    template <typename... Vs>
    buffer::buffer_t restored(const variable_t &v, const Vs &...vs) const {
        const variable_t list[] = {v, vs...};
        buffer::buffer_t regs;
        for (const variable_t &x : list) {
            if (!x.has_buffer()) continue;
            const int i = find_(x.buffer());
            if (i >= 0 && binds_[i].active && !binds_[i].reload().is_none())
                regs.set(reg_name(x.buffer()));
        }
        return regs;
    }

    /// Whether any binding takes part in the frame's passes
    bool has_active() const { return binds_.has_active(); }

    /// Number of bound variables
    int size() const { return binds_.size(); }

    /// Emits one preserve, `body`, restore, and advance pass
    ///
    /// @param xfer The transfer component that emits the pass
    /// @param body The code the bindings are preserved around
    void operator()(const operation::transfer_t &xfer,
            const control_flow::on_enter_t &body) const {
        binds_(xfer, body);
    }

    /// Drops the bindings recorded after the first `n`
    ///
    /// @param n The number of bindings kept
    void truncate(int n) {
        while (size() > n) {
            binds_.pop_back();
            specs_.pop_back();
        }
    }

private:
    int find_(const reg_t &r) const {
        for (int i = 0; i < size(); ++i)
            if (binds_[i].reg == r) return i;
        return -1;
    }

    induction_t binds_;
    array_t<bind_t, 8> specs_;
};

} // namespace loop
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
