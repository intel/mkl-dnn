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

#ifndef RVJIT_INTERFACE_INDUCTION_BIND_HPP
#define RVJIT_INTERFACE_INDUCTION_BIND_HPP

#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/induction/step.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// How one value is saved, restored, and advanced around a loop body
///
/// @note A value with a `slot` is written there before the body and reloaded
///     from it afterwards; one without a `slot` is reloaded from `origin`
/// @note Identifies its value by `reg`
/// @note An inactive binding is skipped by the inductive routine
struct bind_t {
    reg_t reg {};
    slot_t slot {};
    source_t origin = source_t::none();
    step_t step {};
    bool active = true;

    /// Whether the binding names a usable register
    bool is_valid() const { return is_not_zero(reg); }

    /// The source `reg` is reloaded from after the body, `none` if absent
    source_t reload() const { return slot.is_none() ? origin : slot.source(); }

    /// Whether `reload()` reads a register other than `reg`
    bool reloads_from_register() const {
        const source_t src = reload();
        return (src.is_adopt() || src.is_copy()) && src.buffer() != reg;
    }
};

} // namespace rvjit

#endif
