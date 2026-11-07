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

#ifndef RVJIT_INTERFACE_ISA_SOURCE_HPP
#define RVJIT_INTERFACE_ISA_SOURCE_HPP

#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// The origin of a value, naming what a consumer does to materialize it
///
/// @note A default-constructed source is the invalid sentinel; `none()` is the
///     valid source without origin
/// @pre `adopt`/`copy` instances must include valid buffer names
/// @pre `load` instances must include a valid pointer
struct source_t {
    /// What a consumer does to materialize the value
    ///
    /// @note The value is loaded lazily for every kind but `none` and `adopt`
    enum class kind_t {
        invalid, // No source was given
        none, // The object's own buffer already holds the value
        immediate, // A compile-time known integer
        adopt, // A caller register handed over to the consumer
        copy, // A caller register the consumer never writes
        load, // A caller memory address the consumer never writes
    };

    // Not a union: copying one whose active member is narrower than the
    // widest reads uninitialized bytes
    struct payload_t {
        int immediate = 0;
        reg_t buffer;
        pointer_t memory;
    };

    /// Creates the invalid sentinel source
    source_t() : kind_(kind_t::invalid) { value_.immediate = 0; }

    /// Creates an `immediate` source
    ///
    /// @param c The compile-time integer value
    source_t(int c) : kind_(kind_t::immediate) { value_.immediate = c; }

    /// Creates an `adopt` source
    ///
    /// @param b The caller register handed over
    source_t(const reg_t &b) : kind_(kind_t::adopt) { value_.buffer = b; }

    /// Creates an `adopt` source from a native register
    ///
    /// @param b The caller register handed over
    template <typename N, if_native_reg_t<N> = 0>
    source_t(const N &b) : source_t(reg_t(b)) {}

    /// Creates a `load` source
    ///
    /// @param p The memory address the value is loaded from
    source_t(const pointer_t &p) : kind_(kind_t::load) { value_.memory = p; }

    /// Creates a valid source without origin
    static source_t none() {
        source_t s;
        s.kind_ = kind_t::none;
        return s;
    }

    /// Creates an `immediate` source
    ///
    /// @param c The compile-time integer value
    static source_t immediate(int c) { return source_t(c); }

    /// Creates an `adopt` source
    ///
    /// @param b The caller register handed over
    static source_t adopt(const reg_t &b) { return source_t(b); }

    /// Creates a `copy` source
    ///
    /// @param b The caller register copied from, never written
    static source_t copy(const reg_t &b) {
        source_t s(b);
        s.kind_ = kind_t::copy;
        return s;
    }

    /// Creates a `load` source
    ///
    /// @param p The memory address the value is loaded from
    static source_t load(const pointer_t &p) { return source_t(p); }

    /// The kind of origin
    kind_t kind() const { return kind_; }

    /// The compile-time integer
    ///
    /// @pre `is_immediate()`
    int immediate() const { return value_.immediate; }

    /// The caller register
    ///
    /// @pre `is_adopt() || is_copy()`
    reg_t buffer() const { return value_.buffer; }

    /// The memory address
    ///
    /// @pre `is_load()`
    pointer_t memory() const { return value_.memory; }

    /// Whether this source is the invalid sentinel
    bool is_invalid() const { return kind_ == kind_t::invalid; }

    /// Whether this source has no origin
    bool is_none() const { return kind_ == kind_t::none; }

    /// Whether this source is a compile-time integer
    bool is_immediate() const { return kind_ == kind_t::immediate; }

    /// Whether this source is a register handed over to the consumer
    bool is_adopt() const { return kind_ == kind_t::adopt; }

    /// Whether this source is a register the consumer copies from
    bool is_copy() const { return kind_ == kind_t::copy; }

    /// Whether this source is a memory address
    bool is_load() const { return kind_ == kind_t::load; }

    /// Whether materializing from this source emits an instruction
    bool is_lazy() const { return is_immediate() || is_copy() || is_load(); }

    /// Whether the source is not the sentinel and its registers and pointer
    /// are usable
    bool is_valid() const {
        if (is_invalid()) return false;
        if (is_adopt() || is_copy()) return is_not_zero(value_.buffer);
        if (is_load()) return is_not_zero(value_.memory.base);
        return true;
    }

    /// Reinterprets an `adopt` source as a `copy` of the same register
    ///
    /// @note Any other kind is returned unchanged
    source_t to_copy() const {
        return is_adopt() ? source_t::copy(value_.buffer) : *this;
    }

private:
    kind_t kind_;
    payload_t value_;
};

} // namespace rvjit

#endif
