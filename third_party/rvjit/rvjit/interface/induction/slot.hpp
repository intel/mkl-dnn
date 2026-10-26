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

#ifndef RVJIT_INTERFACE_INDUCTION_SLOT_HPP
#define RVJIT_INTERFACE_INDUCTION_SLOT_HPP

#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// A writable location a value is saved into and restored from
struct slot_t {
    /// Where the value is saved
    enum class kind_t {
        none, // No location
        buffer, // A register
        memory, // A memory address
    };

    slot_t() = default;

    /// Creates a `buffer` slot
    ///
    /// @param b The register the value is saved into
    slot_t(const reg_t &b) : kind_(kind_t::buffer), reg_(b) {}

    /// Creates a `buffer` slot from a native register
    ///
    /// @param b The register written to and reloaded from
    template <typename N, if_native_reg_t<N> = 0>
    slot_t(const N &b) : slot_t(reg_t(b)) {}

    /// Creates a `memory` slot
    ///
    /// @param p The memory address the value is saved into
    slot_t(const pointer_t &p) : kind_(kind_t::memory), ptr_(p) {}

    /// Creates a `buffer` slot
    ///
    /// @param b The register the value is saved into
    static slot_t buffer(const reg_t &b) { return slot_t(b); }

    /// Creates a `memory` slot
    ///
    /// @param p The memory address the value is saved into
    static slot_t memory(const pointer_t &p) { return slot_t(p); }

    /// The kind of location
    kind_t kind() const { return kind_; }

    /// Whether the slot has no location
    bool is_none() const { return kind_ == kind_t::none; }

    /// Whether the slot is a register
    bool is_buffer() const { return kind_ == kind_t::buffer; }

    /// Whether the slot is a memory address
    bool is_memory() const { return kind_ == kind_t::memory; }

    /// The register
    ///
    /// @pre `is_buffer()`
    reg_t buffer() const { return reg_; }

    /// The memory address
    ///
    /// @pre `is_memory()`
    pointer_t memory() const { return ptr_; }

    /// Whether the slot names a usable location
    bool is_valid() const {
        if (is_buffer()) return is_not_zero(reg_);
        if (is_memory()) return is_not_zero(ptr_.base);
        return false;
    }

    /// The source a value saved into this slot is restored from
    source_t source() const {
        if (is_buffer()) return source_t::adopt(reg_);
        if (is_memory()) return source_t::load(ptr_);
        return source_t::none();
    }

private:
    kind_t kind_ = kind_t::none;
    reg_t reg_ {};
    pointer_t ptr_ {};
};

} // namespace rvjit

#endif
