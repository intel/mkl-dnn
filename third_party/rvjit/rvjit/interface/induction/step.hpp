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

#ifndef RVJIT_INTERFACE_INDUCTION_STEP_HPP
#define RVJIT_INTERFACE_INDUCTION_STEP_HPP

#include "rvjit/interface/isa/buffered.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// Compile-time known value update pattern taken in predictable steps
///
/// @note Expresses inductive variable update procedures
///
/// @pre `src1` must be set on instances with kind `add` or `sub`
/// @pre `src1`/`src2` must be set on instances with kind `macc`
/// @pre The caller must fulfill the validity conditions of `src1` and `src2`
struct step_t : buffered_t {
    /// The in-place update a step performs
    enum class kind_t {
        none, // No update
        add, // Adds `src1`
        sub, // Subtracts `src1`
        macc, // Adds the product `src1 * src2`
    };

    kind_t kind = kind_t::none;
    const_t src1 {}; // constant added/subtracted, first factor in `macc`
    const_t src2 {}; // second factor constant, only used in `macc`

    step_t() = default;

    /// The scratch register a `macc` step multiplies into
    reg_t buffer() const { return reg_(); }

    /// Whether the step performs no update
    bool is_none() const { return kind == kind_t::none; }

    /// Whether the step adds `src1`
    bool is_add() const { return kind == kind_t::add; }

    /// Whether the step subtracts `src1`
    bool is_sub() const { return kind == kind_t::sub; }

    /// Whether the step adds the product `src1 * src2`
    bool is_macc() const { return kind == kind_t::macc; }

    /// Whether emitting this `macc` step requires a real multiply
    ///
    /// @details A compile-time delta that fits into the immediate field
    ///     can be lowered into a single addition using immediates
    bool macc_needs_multiply() const {
        if (!is_macc()) return false;
        if (!src1.has_imm() || !src2.has_imm()) return true;
        const int64_t delta = src1.imm() * src2.imm();
        return !is_simm12(delta);
    }

    bool is_valid() const override {
        if (!buffered_t::is_valid()) return false;
        switch (kind) {
            case kind_t::none: return true;
            case kind_t::add:
            case kind_t::sub: return src1.is_valid();
            case kind_t::macc: return src1.is_valid() && src2.is_valid();
        }
        return false;
    }

    /// Creates the step that performs no update
    static step_t none() { return {}; }

    /// Creates a step adding `c`
    ///
    /// @param c The constant added
    ///
    /// @pre The caller must supply `c` such that `c.is_valid()` is true
    static step_t plain(const const_t &c) {
        step_t s;
        s.kind = kind_t::add;
        s.src1 = c;
        return s;
    }

    /// Creates a step subtracting `c`
    ///
    /// @param c The constant subtracted
    ///
    /// @pre The caller must supply `c` such that `c.is_valid()` is true
    static step_t sub(const const_t &c) {
        step_t s;
        s.kind = kind_t::sub;
        s.src1 = c;
        return s;
    }

    /// Creates a step adding the product of two constants
    ///
    /// @param s1 The first factor
    /// @param s2 The second factor
    /// @param tmp The scratch register for the multiply, if the caller has one
    ///
    /// @pre The caller must supply `s1`/`s2` such that both are valid
    static step_t macc(
            const const_t &s1, const const_t &s2, const reg_t &tmp = reg_t()) {
        step_t s;
        s.kind = kind_t::macc;
        s.src1 = s1;
        s.src2 = s2;
        s.set_buffer(tmp);
        return s;
    }

    /// @note Only a `macc` emitting mul needs a scratch register
    int buffer_needed() const override {
        return macc_needs_multiply() && !has_buffer() ? 1 : 0;
    }
};

} // namespace rvjit

#endif
