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

#ifndef RVJIT_INTERFACE_ISA_CONST_HPP
#define RVJIT_INTERFACE_ISA_CONST_HPP

#include "rvjit/interface/isa/buffered.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// Constant represented as an immediate, a register, or both (large numbers)
///
/// @pre The constant must be consumed from `buffer()` whenever `is_reg()`
/// @pre The constant must be consumed from `imm()` whenever `is_imm()`
struct const_t : buffered_t {
    /// How a consumer reads the constant
    enum class kind_t {
        reg, // Held by a register only
        imm, // An immediate that fits `simm12`
        imm_reg, // An immediate that overflows `simm12`, held by a register
    };

    kind_t kind = kind_t::imm;

    /// The kind an immediate of value `imm` takes
    ///
    /// @param imm The integer value of the constant
    static kind_t kind_of(int imm) {
        return is_simm12(imm) ? kind_t::imm : kind_t::imm_reg;
    }

    /// Creates the immediate constant zero
    const_t() { source_ = source_t::immediate(0); }

    /// Creates a `reg` constant read in place
    ///
    /// @param reg The register holding the value
    const_t(const reg_t &reg) : kind(kind_t::reg) { supply_(reg); }

    /// Creates a `reg` constant read in place from a native register
    ///
    /// @param reg The register holding the value
    template <typename N, if_native_reg_t<N> = 0>
    const_t(const N &reg) : const_t(reg_t(reg)) {}

    /// Creates an immediate constant, which needs a register if it is large
    ///
    /// @param imm The integer value of the constant
    const_t(int imm) : kind(kind_of(imm)) {
        source_ = source_t::immediate(imm);
    }

    /// Creates an immediate constant with a register for a large value
    ///
    /// @param imm The integer value of the constant
    /// @param reg The register holding `imm`, used if it overflows `simm12`
    const_t(int imm, const reg_t &reg) : kind(kind_of(imm)) {
        source_ = source_t::immediate(imm);
        if (kind == kind_t::imm_reg) supply_(reg);
    }

    /// Creates the constant a `source_t` describes
    ///
    /// @param src The origin of the value
    ///
    /// @note An `immediate` source is an immediate constant
    /// @note An `adopt`/`copy` source is a `reg` constant read in place
    /// @note A `load` source is a `reg` constant loaded lazily
    /// @note A `none` or invalid source is an `invalid()` constant
    explicit const_t(const source_t &src) {
        if (!src.is_valid() || src.is_none())
            *this = invalid();
        else if (src.is_immediate())
            *this = const_t(src.immediate());
        else if (src.is_load())
            *this = from_memory(src.memory());
        else
            *this = const_t(src.buffer());
    }

    /// Creates a `reg` constant loaded lazily from memory
    ///
    /// @param mem The memory address the value is loaded from
    static const_t from_memory(const pointer_t &mem) {
        const_t c;
        c.kind = kind_t::reg;
        c.source_ = source_t::load(mem);
        return c;
    }

    /// Creates the constant that is never valid
    static const_t invalid() {
        const_t c;
        c.source_ = source_t();
        return c;
    }

    /// Whether the constant is consumed from a register
    bool is_reg() const { return kind != kind_t::imm; }

    /// Whether the constant is consumed from an immediate alone
    bool is_imm() const { return kind == kind_t::imm; }

    /// Whether the constant is an immediate held by a register as well
    bool is_imm_reg() const { return kind == kind_t::imm_reg; }

    /// Whether the constant has a known integer value
    bool has_imm() const { return is_imm() || is_imm_reg(); }

    /// Whether the constant has a register
    bool has_reg() const { return has_buffer(); }

    /// The integer value
    ///
    /// @pre `has_imm()`
    int imm() const { return source().immediate(); }

    /// The register holding the constant
    ///
    /// @note Returns zero if kind is `imm` or `!has_reg()`
    reg_t buffer() const { return reg_(); }

    /// This constant multiplied by `factor`
    ///
    /// @param factor The integer to multiply by
    ///
    /// @note Returns `invalid()` for a register-sourced constant
    /// @note A product past `simm12` is an `imm_reg` constant that still
    ///     needs its register
    const_t scaled(int factor) const {
        if (factor == 1) return *this;
        if (!has_imm() || has_reg()) return invalid();
        return const_t(imm() * factor);
    }

    /// This constant held in a register, loaded with its immediate value
    ///
    /// @note Returns the constant itself unless it is a plain immediate
    const_t in_reg() const {
        if (!is_imm()) return *this;
        const_t c = *this;
        c.kind = kind_t::imm_reg;
        return c;
    }

    /// Whether the constant is a register loaded from a memory source
    bool is_loaded() const { return kind == kind_t::reg && source().is_load(); }

    /// Whether the constant is known to be zero, or holds no register
    bool is_zero() const {
        if (has_imm()) return imm() == 0;
        return !has_buffer();
    }

    /// Whether both constants are immediates of the same value
    ///
    /// @param o The constant compared against
    bool is_same_imm(const const_t &o) const {
        return has_imm() && o.has_imm() && imm() == o.imm();
    }

    /// @note A large known or memory-sourced constant needs a register
    int buffer_needed() const override {
        return needs_reg_() && !has_buffer() ? 1 : 0;
    }

    bool needs_to_materialize() const override { return needs_reg_(); }

    /// @note An immediate constant without an immediate source is invalid
    bool is_valid() const override {
        return !needs_buffer() && source().is_valid()
                && !(has_imm() && source().is_none());
    }

private:
    bool needs_reg_() const { return is_imm_reg() || is_loaded(); }

    void supply_(const reg_t &r) {
        if (is_not_zero(r)) buffer_ = buffer::buffer_t({r});
    }
};

} // namespace rvjit

#endif
