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

#ifndef RVJIT_INTERFACE_ISA_VARIABLE_HPP
#define RVJIT_INTERFACE_ISA_VARIABLE_HPP

#include "rvjit/interface/isa/buffered.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// A value residing in a buffer with an optional source
///
/// @note An instance with `source().is_none()` is a materialized value
/// @note Non-materialized instances may be materialized via `transfer_t`
struct variable_t : buffered_t {
    variable_t() = default;

    /// Creates a variable residing in `r`
    ///
    /// @param r The register holding the value
    variable_t(const reg_t &r) { supply_(r); }

    /// Creates a variable residing in a native register
    ///
    /// @param r The register holding the value
    template <typename N, if_native_reg_t<N> = 0>
    variable_t(const N &r) : variable_t(reg_t(r)) {}

    /// Creates a variable residing in `r` that reloads from `src`
    ///
    /// @param r The register holding the value
    /// @param src The origin the value is reloaded from
    ///
    /// @note An `adopt` source is kept as a `copy` of the same register
    variable_t(const reg_t &r, const source_t &src) {
        supply_(r);
        source_ = src.to_copy();
    }

    /// Creates a variable that reloads from `src`, its register still pending
    ///
    /// @param src The origin the value is reloaded from
    explicit variable_t(const source_t &src) : variable_t(reg_t(), src) {}

    /// Creates the variable a `source_t` describes
    ///
    /// @param src The origin of the value
    /// @param r The register holding the value, if the caller supplies one
    ///
    /// @note An `adopt` source is taken as the buffer and keeps no origin
    static variable_t from(const source_t &src, const reg_t &r = reg_t()) {
        if (src.is_adopt()) return variable_t(src.buffer());
        return variable_t(r, src);
    }

    /// The register holding the value
    reg_t buffer() const { return reg_(); }

    int buffer_needed() const override { return has_buffer() ? 0 : 1; }

    /// @note Materializes from a lazy source into its buffer
    bool needs_to_materialize() const override {
        return source().is_lazy() && has_buffer();
    }

private:
    void supply_(const reg_t &r) {
        if (is_not_zero(r)) buffer_ = buffer::buffer_t({r});
    }
};

} // namespace rvjit

#endif
