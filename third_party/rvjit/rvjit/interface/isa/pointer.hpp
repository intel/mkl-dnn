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

#ifndef RVJIT_INTERFACE_ISA_POINTER_HPP
#define RVJIT_INTERFACE_ISA_POINTER_HPP

#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// A pointer composed of a base register and an offset
struct pointer_t {
    reg_t base;
    int offset = 0;

    pointer_t() = default;

    /// Creates a pointer
    ///
    /// @param base The base register
    /// @param off The byte offset from `base`
    pointer_t(const reg_t &base, int off = 0) : base(base), offset(off) {}

    /// Creates a pointer from a native base register
    ///
    /// @param base The base register
    /// @param off The byte offset from `base`
    template <typename N, if_native_reg_t<N> = 0>
    pointer_t(const N &base, int off = 0) : pointer_t(reg_t(base), off) {}

    /// Creates a pointer to a byte offset from the stack pointer
    ///
    /// @param offset The byte offset from `sp`
    static pointer_t stack_at(int offset) {
        return pointer_t(reg_t::stack(), offset);
    }

    /// Whether the base is usable and the offset fits `simm12`
    bool is_valid() const { return is_simm12(offset) && is_not_zero(base); }

    /// Creates a pointer with a new base, preserving `offset`
    ///
    /// @param b The new base register
    pointer_t with_base(const reg_t &b) const {
        auto out = *this;
        out.base = b;
        return out;
    }

    /// Creates a pointer with `delta` added to `offset`, reusing `base`
    ///
    /// @param delta The byte offset added
    pointer_t added_offset(int delta) const {
        auto out = *this;
        out.offset += delta;
        return out;
    }
};

} // namespace rvjit

#endif
