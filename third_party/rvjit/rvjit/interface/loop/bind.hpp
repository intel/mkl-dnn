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

#ifndef RVJIT_INTERFACE_LOOP_BIND_HPP
#define RVJIT_INTERFACE_LOOP_BIND_HPP

#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/induction/step.hpp"

namespace rvjit {
namespace loop {

/// Declarative request to bind a variable to an induction frame
///
struct bind_t {
    /// Where the value is preserved to and restored from
    ///
    /// @note The bind is inert with `off`
    enum class storage_t {
        none, // Reloaded from the variable's origin, nothing is written
        stack, // A stack slot drawn when the frame is allocated
        buffer, // A register drawn when the frame is allocated
        given, // The slot the caller supplies
        off, // Inert
    };

    /// When the bind takes part in the frame
    enum class when_t {
        always, // On every body run
        multi_trip, // Only when the body runs more than once, else inert
    };

    storage_t storage = storage_t::none;
    when_t when = when_t::always;
    slot_t slot {};
    step_t step {};

    /// A bind reloaded from the variable's origin
    static bind_t reloaded() { return {}; }

    /// An inert bind
    static bind_t off() { return with_(storage_t::off); }

    /// A bind preserved to a stack slot
    static bind_t stack() { return with_(storage_t::stack); }

    /// A bind preserved to a register
    static bind_t buffer() { return with_(storage_t::buffer); }

    /// A bind preserved to a slot the caller supplies
    ///
    /// @param s The slot
    static bind_t given(const slot_t &s) {
        bind_t b = with_(storage_t::given);
        b.slot = s;
        return b;
    }

    /// A copy of this bind that takes part only when the body runs repeatedly
    bind_t multi_trip() const {
        bind_t b = *this;
        b.when = when_t::multi_trip;
        return b;
    }

    /// A copy of this bind that advances by `s` after each body
    ///
    /// @param s The step
    bind_t stepped(const step_t &s) const {
        bind_t b = *this;
        b.step = s;
        return b;
    }

private:
    static bind_t with_(storage_t st) {
        bind_t b;
        b.storage = st;
        return b;
    }
};

} // namespace loop
} // namespace rvjit

#endif
