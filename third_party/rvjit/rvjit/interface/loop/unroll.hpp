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

#ifndef RVJIT_INTERFACE_LOOP_UNROLL_HPP
#define RVJIT_INTERFACE_LOOP_UNROLL_HPP

#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace loop {

/// An ordered list of loop-unrolling factors
///
/// @pre The factors, if modified, must retain a descending order
struct unroll_t : array_t<int, 8> {
    using array_t::array_t;

    static constexpr int MAX_UNROLL = 8;

    /// Whether the factors are positive and in descending order
    bool is_valid() const {
        if (size() <= 0 || size() > MAXN) return false;
        for (int i = 0; i < size(); ++i) {
            if ((*this)[i] < 1) return false;
            if (i > 0 && (*this)[i] > (*this)[i - 1]) return false;
        }
        return true;
    }
};

} // namespace loop
} // namespace rvjit

#endif
