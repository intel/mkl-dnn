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

#ifndef RVJIT_INTERFACE_ISA_PARAMETER_HPP
#define RVJIT_INTERFACE_ISA_PARAMETER_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {

/// A numeric value parameter passed to a component
struct parameter_t {
    /// How the value reaches the generated code
    enum class kind_t {
        literal, // A compile-time known value
        runtime, // A dynamic, unbounded value
    };

    parameter_t() = default;

    /// Creates a parameter from the origin of its value
    ///
    /// @param src The origin of the value
    parameter_t(const source_t &src) : src_(src) {}

    /// Creates a `literal` parameter
    ///
    /// @param v The compile-time known value
    parameter_t(int v) : parameter_t(source_t(v)) {}

    /// Creates a `literal` parameter
    ///
    /// @param v The compile-time known value
    static parameter_t literal(int v) { return parameter_t(v); }

    /// Creates a `runtime` parameter
    ///
    /// @param r The origin of the dynamic value
    static parameter_t runtime(const source_t &r) { return parameter_t(r); }

    /// `literal` when the source is an `immediate` or `runtime` otherwise
    kind_t kind() const {
        return src_.is_immediate() ? kind_t::literal : kind_t::runtime;
    }

    /// Obtains the parameter numerical value
    ///
    /// @pre The object must not be `runtime`-kind
    int value() const { return src_.is_immediate() ? src_.immediate() : 0; }

    /// Obtains the source this parameter's value comes from
    const source_t &source() const { return src_; }

    /// Whether the value is known at code generation
    bool is_literal() const { return kind() == kind_t::literal; }

    /// Whether the value is dynamic
    bool is_runtime() const { return kind() == kind_t::runtime; }

    /// Whether or not both parameters are identical literals
    ///
    /// @param o The parameter compared against
    bool is_same_value(const parameter_t &o) const {
        return is_literal() && o.is_literal() && value() == o.value();
    }

protected:
    source_t src_ = source_t::none();
};

/// A `parameter_t` denoting a size refined with what is known about it
///
/// @details A literal is exactly its value; a runtime value may be enhanced
///     with an upper bound, if known, enabling better emission decisions.
struct size_parameter_t : public parameter_t {
    size_parameter_t() = default;

    /// Creates a size from the origin of its value
    ///
    /// @param src The origin of the value
    size_parameter_t(const source_t &src) : parameter_t(src) {}

    /// Creates a `literal` size
    ///
    /// @param v The compile-time known value
    size_parameter_t(int v) : size_parameter_t(source_t(v)) {}

    /// Creates a `literal` size
    ///
    /// @param v The compile-time known value
    static size_parameter_t literal(int v) { return size_parameter_t(v); }

    /// Creates an unbounded `runtime` size
    ///
    /// @param src The origin of the dynamic value
    static size_parameter_t runtime(const source_t &src) {
        return size_parameter_t(src);
    }

    /// Creates a `less_eq` size
    ///
    /// @param r The register holding the dynamic value
    /// @param p The upper bound of the value
    static size_parameter_t less_eq(const reg_t &r, int p) {
        return size_parameter_t(source_t(r)).as_less_eq(p);
    }

    /// Interprets the parameter value as a constant known upper bound
    ///
    /// @param v The upper bound
    size_parameter_t &as_less_eq(int v) {
        bound_ = v;
        has_bound_ = true;
        return *this;
    }

    /// Whether the value is known at code generation
    bool is_literal() const { return parameter_t::is_literal(); }

    /// Whether the value is dynamic and unbounded
    bool is_runtime() const { return !is_literal() && !has_bound_; }

    /// Whether the value is dynamic with a known upper bound
    bool is_less_eq() const { return !is_literal() && has_bound_; }

    /// Whether the value is at most a known number: a literal or `less_eq`
    bool has_bound() const { return is_literal() || has_bound_; }

    /// The largest value, or 0 when unbounded
    int bound() const {
        if (is_literal()) return parameter_t::value();
        return has_bound_ ? bound_ : 0;
    }

    /// Obtains the numerical value, or the bound for `less_eq`
    int value() const { return bound(); }

private:
    bool has_bound_ = false;
    int bound_ = 0;
};

/// A scalar parameter of a certain data type from a literal or runtime source
///
/// @note A runtime source holds the raw bit pattern of the scalar: a `load`
///     reads it from memory, an `adopt`/`copy` is a register holding it
/// @note A bare number is a literal *value* (`scalar_parameter_t(1)` is 1.0),
///     whereas an integer `source_t` is a bit pattern
struct scalar_parameter_t : public parameter_t {
    scalar_parameter_t() = default;

    /// Creates a literal scalar
    ///
    /// @param v The scalar value
    scalar_parameter_t(double v) : value_(v), has_value_(true) {}

    /// Creates a runtime scalar
    ///
    /// @param bits The origin of the scalar's bit pattern
    scalar_parameter_t(const source_t &bits) : parameter_t(bits) {}

    /// Whether the scalar is a codegen-time literal
    bool has_literal() const { return has_value_; }

    /// The literal scalar value
    ///
    /// @pre `has_literal()`
    double literal() const { return value_; }

    /// Whether a value (literal or runtime) was given at all
    bool is_set() const { return has_value_ || !src_.is_none(); }

private:
    double value_ = 0.0;
    bool has_value_ = false;
};

/// The data type of a value in memory and in the registers computing on it
struct type_parameter_t {
    type_t mem = type_t::f32; // The type the value is loaded or stored as
    type_t reg = type_t::f32; // The type the registers hold

    type_parameter_t() = default;

    /// Creates a value of one type in memory and in registers
    ///
    /// @param t The type
    type_parameter_t(type_t t) : mem(t), reg(t) {}

    /// Creates a value converted between memory and the registers
    ///
    /// @param m The type in memory
    /// @param r The type in registers
    type_parameter_t(type_t m, type_t r) : mem(m), reg(r) {}

    /// Whether the value converts between memory and the registers
    bool converts() const { return mem != reg; }

    /// Whether the registers are wider than memory
    bool widens() const { return sewb(reg) > sewb(mem); }

    /// Whether the registers are narrower than memory
    bool narrows() const { return sewb(reg) < sewb(mem); }

    /// How many times wider the registers are than memory: the widening
    /// factor, 1 unless the value widens
    int ratio() const {
        const int m = sewb(mem), r = sewb(reg);
        return m > 0 && r > m ? r / m : 1;
    }

    /// The registers between the first of a register group of `reg` and the
    /// part a widened value is read from, 0 unless the value widens
    ///
    /// @param group The registers of a group of `reg`
    ///
    /// @details A widening reads the narrow value from the highest part of
    ///     the wide group (see `conversion_t`), where it must be loaded
    int offset(int group) const { return group - group / ratio(); }

    bool operator==(const type_parameter_t &o) const {
        return mem == o.mem && reg == o.reg;
    }
    bool operator!=(const type_parameter_t &o) const { return !(*this == o); }
};

} // namespace rvjit

#endif
