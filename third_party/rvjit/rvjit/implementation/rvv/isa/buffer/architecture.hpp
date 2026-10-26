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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_BUFFER_ARCHITECTURE_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_BUFFER_ARCHITECTURE_HPP

#include <utility>

#include "rvjit/implementation/rv64/isa/buffer/float_register_file.hpp"
#include "rvjit/implementation/rv64/isa/buffer/int_register_file.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rvv/isa/buffer/vector_register_file.hpp"
#include "rvjit/interface/isa/buffer/architecture.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"

namespace rvjit {
namespace rvv {
namespace buffer {

/// Wires up the int/float/vector register files for an RVV-capable target
struct architecture_t : rvjit::buffer::architecture_t {
    architecture_t() { *this = standard(); }

    /// Creates the architecture with default float and vector register files
    ///
    /// @param integers The integer registers
    explicit architecture_t(rvjit::buffer::buffer_t integers)
        : architecture_t(integers, default_floats_(), default_vectors_()) {}

    /// Creates the architecture from explicit register sets
    ///
    /// @param i The integer registers
    /// @param f The float registers
    /// @param v The vector registers
    architecture_t(rvjit::buffer::buffer_t i, rvjit::buffer::buffer_t f,
            rvjit::buffer::buffer_t v) {
        integer = rvjit::buffer::buffer_file_t(
                i, rv64::buffer::int_register_file_t::order());
        floating = rvjit::buffer::buffer_file_t(
                f, rv64::buffer::float_register_file_t::standard().order());
        vector = rvjit::buffer::buffer_file_t(v);
    }

    /// The standard lists: 27 x, 20 f and 32 v registers
    ///
    /// @param include_a0 Whether `a0` is part of the integer list
    ///
    /// @note The `a0` register often carries the JIT kernel arguments, and
    ///     so it is removed from the pool unless requested with `include_a0`
    static architecture_t standard(bool include_a0 = false) {
        architecture_t a(
                rv64::buffer::int_register_file_t::standard(include_a0).file(),
                default_floats_(), default_vectors_());
        return a;
    }

    /// The same as `standard()`
    ///
    /// @param include_a0 Whether `a0` is part of the integer list
    static architecture_t default_list(bool include_a0 = false) {
        return standard(include_a0);
    }

private:
    static rvjit::buffer::buffer_t default_floats_() {
        return rv64::buffer::float_register_file_t::standard().file();
    }

    static rvjit::buffer::buffer_t default_vectors_() {
        return vector_register_file_t::standard().file();
    }
};

} // namespace buffer
} // namespace rvv
} // namespace rvjit

#endif
