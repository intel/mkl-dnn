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

#ifndef RVJIT_INTERFACE_ISA_OPERAND_HPP
#define RVJIT_INTERFACE_ISA_OPERAND_HPP

#include <cstdint>

#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// Operand data type
enum class type_t {
    s8,
    u8,
    s16,
    u16,
    s32,
    u32,
    s64,
    f16,
    bf16,
    f32,
    f64,
    none
};

/// Operand storage categories
enum class form_t {
    scalar, // Held by a scalar register
    vector, // Held by a vector register
    undefined, // Not yet chosen
};

/// Operand numeric categories
enum class kind_t { integer, floating };

/// Interface objects implement when they must answer type-traits queries
struct typed_t {
    /// Whether the object is held by a scalar register
    virtual bool is_scalar() const { return false; }

    /// Whether the object is held by a vector register
    virtual bool is_vector() const { return false; }

    /// Whether the object is an integer
    virtual bool is_int() const { return false; }

    /// Whether the object is a floating-point number
    virtual bool is_float() const { return false; }

    /// Whether the object is a signed integer
    virtual bool is_signed() const { return false; }
};

/// Maps a narrow operand type to its natural 2x-wide counterpart
///
/// @param dt The narrow operand type
///
/// @note s64 and f64 map to `none`
/// @note u16 maps to s32, which holds every u16 value
inline type_t natural_wide(const type_t &dt) {
    switch (dt) {
        case type_t::f16:
        case type_t::bf16: return type_t::f32;
        case type_t::f32: return type_t::f64;
        case type_t::s8: return type_t::s16;
        case type_t::u8: return type_t::u16;
        case type_t::s16:
        case type_t::u16: return type_t::s32;
        case type_t::s32:
        case type_t::u32: return type_t::s64;
        case type_t::s64:
        case type_t::f64: return type_t::none;
        default: return type_t::none;
    }
}

/// Whether an operand type is a signed integer
///
/// @param dt The operand type
inline bool is_signed_int(const type_t &dt) {
    switch (dt) {
        case type_t::s8:
        case type_t::s16:
        case type_t::s32:
        case type_t::s64: return true;
        default: return false;
    }
}

/// Whether an operand type is an integer type
///
/// @param dt The operand type
inline bool is_int(const type_t &dt) {
    switch (dt) {
        case type_t::s8:
        case type_t::u8:
        case type_t::s16:
        case type_t::u16:
        case type_t::s32:
        case type_t::u32:
        case type_t::s64: return true;
        default: return false;
    }
}

/// The single-element-width size, in bytes, of an operand type
///
/// @param dt The operand type
///
/// @note Returns zero for `none`
inline int sewb(const type_t &dt) {
    switch (dt) {
        case type_t::s8:
        case type_t::u8: return 1;
        case type_t::s16:
        case type_t::u16:
        case type_t::f16:
        case type_t::bf16: return 2;
        case type_t::s32:
        case type_t::u32:
        case type_t::f32: return 4;
        case type_t::s64:
        case type_t::f64: return 8;
        default: return 0;
    }
}

/// Data operand for operations with generic type, form, or numerical kind
///
/// @note The `form, kind, type` triplet determines the ISA operand traits
/// @note The `name` parameter informs the generic operand buffer id
/// @note Whether `name` is a valid name choice for the underlying ISA and
///     considering the ISA traits is not verified by the `is_valid` function
struct operand_t : public typed_t {
    form_t form = form_t::vector;
    kind_t kind = kind_t::floating;
    type_t type = type_t::f32;
    name_t name = no_name;
    int width = 4;
    bool sign = false;

    operand_t() = default;

    /// Creates an operand of a form and data type
    ///
    /// @param form The storage category
    /// @param dt The data type, which sets `kind`, `width` and `sign`
    /// @param i The buffer id, `no_name` until allocated
    operand_t(form_t form, type_t dt, name_t i = 0) : form(form), name(i) {
        set_type(dt);
    }

    /// Sets the data type and derives `kind`, `width` and `sign` from it
    ///
    /// @param dt The data type
    void set_type(type_t dt) {
        type = dt;
        kind = rvjit::is_int(dt) ? kind_t::integer : kind_t::floating;
        width = sewb(dt);
        sign = is_signed_int(dt);
    }

    bool is_scalar() const override { return form == form_t::scalar; }
    bool is_vector() const override { return form == form_t::vector; }
    bool is_int() const override { return kind == kind_t::integer; }
    bool is_float() const override { return kind == kind_t::floating; }
    bool is_signed() const override { return sign; }

    /// Whether the storage category was chosen
    bool is_defined() const { return form != form_t::undefined; }

    /// Whether the buffer id was allocated
    bool has_name() const { return name != no_name; }

    /// Whether the width is a power of two
    bool has_valid_width() const { return is_pow2(width) && width > 0; }

    /// Whether the operand has a buffer id and a valid width
    bool is_valid() const { return has_name() && has_valid_width(); }
};

} // namespace rvjit

#endif
