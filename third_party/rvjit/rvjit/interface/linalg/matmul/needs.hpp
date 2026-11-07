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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_NEEDS_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_NEEDS_HPP

#include <algorithm>

namespace rvjit {
namespace linalg {
namespace matmul {

/// Resources a post-op needs while it emits, which the matmul provides
///
/// @note `accs` counts registers of the accumulators' class, in the groups the
///     output allocation draws; `accs_wanted` is how many the post-op uses
///     when they are free, which the matmul grants from what the accumulators
///     leave rather than shrinking them
struct needs_t {
    int accs = 0;
    int accs_wanted = 0;
    int floats = 0;
    int ints = 0;
    bool m_offset = false; // the byte offset of the tile's first M element
    bool args = false; // the plan's `postop_args` pointer
    bool mask = false; // the ISA's mask register, which scratch never holds

    /// Raises this need to cover `o` too
    ///
    /// @param o The need merged
    needs_t &merge(const needs_t &o) {
        accs = std::max(accs, o.accs);
        accs_wanted = std::max(accs_wanted, o.accs_wanted);
        floats = std::max(floats, o.floats);
        ints = std::max(ints, o.ints);
        m_offset = m_offset || o.m_offset;
        args = args || o.args;
        mask = mask || o.mask;
        return *this;
    }
};

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#endif
