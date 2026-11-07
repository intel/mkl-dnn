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

#ifndef RVJIT_INTERFACE_ISA_REGISTER_HPP
#define RVJIT_INTERFACE_ISA_REGISTER_HPP

#include <type_traits>

#include "rvjit/interface/isa/isa_traits.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/resource/resource.hpp"

namespace rvjit {

namespace detail {

// Whether `isa_traits<N>` describes a native register of any class
template <typename N, typename = void>
struct is_native_t : std::false_type {};

template <typename N>
struct is_native_t<N, decltype(void(isa_traits<N>::resource))>
    : std::true_type {};

// Whether `isa_traits<N>` describes a native register of class `C`
template <typename N, resource_t C, bool = is_native_t<N>::value>
struct is_native_reg_t : std::false_type {};

template <typename N, resource_t C>
struct is_native_reg_t<N, C, true>
    : std::integral_constant<bool, isa_traits<N>::resource == C> {};

} // namespace detail

/// Enables an overload for native registers of class `C` only
template <typename N, resource_t C = resource_t::integers>
using if_native_reg_t =
        typename std::enable_if<detail::is_native_reg_t<N, C>::value,
                int>::type;

/// A register of class `C`, identified by its name alone
///
/// @note Converts implicitly to and from the native registers an ISA
///     describes through `isa_traits`
template <resource_t C>
struct reg_of_t {
    /// Creates an absent register
    constexpr reg_of_t() = default;

    /// Creates the register named `n`
    ///
    /// @param n The register name
    constexpr explicit reg_of_t(name_t n) : name_(n) {}

    /// Creates the register a native register names
    ///
    /// @param native The native register
    template <typename N, if_native_reg_t<N, C> = 0>
    reg_of_t(const N &native) : name_(isa_traits<N>::name(native)) {}

    /// The native register this register names
    template <typename N, if_native_reg_t<N, C> = 0>
    operator N() const {
        return isa_traits<N>::make(name_);
    }

    /// The register that always reads as zero
    static constexpr reg_of_t zero() { return reg_of_t(zero_name); }

    /// The stack pointer register
    static constexpr reg_of_t stack() { return reg_of_t(stack_name); }

    /// The register name
    constexpr name_t name() const { return name_; }

    /// Whether the register names neither nothing nor the zero register
    constexpr bool is_not_zero() const {
        return name_ != no_name && name_ != zero_name;
    }

    /// Whether both registers have the same name
    ///
    /// @param o The register compared against
    constexpr bool operator==(const reg_of_t &o) const {
        return name_ == o.name_;
    }

    /// Whether the registers have different names
    ///
    /// @param o The register compared against
    constexpr bool operator!=(const reg_of_t &o) const {
        return name_ != o.name_;
    }

private:
    name_t name_ = no_name;
};

using reg_t = reg_of_t<resource_t::integers>; // A general-purpose register
using freg_t = reg_of_t<resource_t::floats>; // A floating-point register
using vreg_t = reg_of_t<resource_t::vectors>; // A vector register

/// Whether a general-purpose register names neither nothing nor zero
///
/// @param r The register
inline bool is_not_zero(const reg_t &r) {
    return r.is_not_zero();
}

/// The name of a register
///
/// @param r The register
template <resource_t C>
constexpr name_t reg_name(const reg_of_t<C> &r) {
    return r.name();
}

/// The name of a native register an ISA describes
///
/// @param r The native register
template <typename N,
        typename std::enable_if<detail::is_native_t<N>::value, int>::type = 0>
name_t reg_name(const N &r) {
    return isa_traits<N>::name(r);
}

} // namespace rvjit

#endif
