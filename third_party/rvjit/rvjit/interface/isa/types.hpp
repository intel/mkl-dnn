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

#ifndef RVJIT_INTERFACE_ISA_TYPES_HPP
#define RVJIT_INTERFACE_ISA_TYPES_HPP

#include <cstdint>

#include "rvjit/interface/isa/isa_traits.hpp"

namespace rvjit {

using name_t = uint8_t; // Generic register identifier

// Sentinel name of a pending or failed allocation, no register has it
constexpr name_t no_name = 255;

// Name of the register that always reads as zero
constexpr name_t zero_name = 254;

// Name of the stack pointer register
constexpr name_t stack_name = 253;

/// Whether an operation applies to every element or only to the masked ones
enum class predicate_t {
    unmasked, // Every element
    masked, // Only the elements the ISA's mask register enables
};

/// Opaque handle to an ISA code generator
///
/// @note An ISA implementation recovers the typed code generator with `as()`
class emitter_t {
public:
    /// A detached handle, only fit for objects that never emit
    emitter_t() = default;

    /// Creates a handle to `cg`, which must outlive it
    ///
    /// @param cg The code generator instructions are emitted through
    ///
    /// @note Holds `cg` as the ISA's own `isa_traits<CG>::codegen_t`, so a
    ///     class derived from it is recovered as that type
    template <typename CG, typename Base = typename isa_traits<CG>::codegen_t>
    explicit emitter_t(CG &cg)
        : cg_(static_cast<Base *>(&cg)), tag_(tag_of_<Base>()) {}

    /// Whether the handle refers to a code generator
    bool is_attached() const { return cg_ != nullptr; }

    /// The code generator, if it is a `CG`
    ///
    /// @note Returns `nullptr` when detached or when `CG` is not the type the
    ///     handle was created from
    template <typename CG>
    CG *as() const {
        return tag_ == tag_of_<CG>() ? static_cast<CG *>(cg_) : nullptr;
    }

private:
    template <typename CG>
    static const void *tag_of_() {
        static const char tag = 0;
        return &tag;
    }

    void *cg_ = nullptr;
    const void *tag_ = nullptr;
};

} // namespace rvjit

#endif
