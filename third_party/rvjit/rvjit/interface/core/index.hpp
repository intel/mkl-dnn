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

#ifndef RVJIT_INTERFACE_CORE_INDEX_HPP
#define RVJIT_INTERFACE_CORE_INDEX_HPP

#include <array>

#include "rvjit/interface/core/callback.hpp"

namespace rvjit {

/// Default dimensionality for multi-dimensional abstractions
///
/// @note Two axes per tensor: three 2d operand tensors sharing one axis
///     pairwise cover the up-to-three register operands of an instruction
static constexpr int NDIMS = 2;

/// A multi-dimensional index into an iteration space
template <int ND = NDIMS>
using index_nd_t = std::array<int, ND>;

/// An index into an iteration space with one dimension
using index_1d_t = index_nd_t<1>;

/// An index into an iteration space with the default number of dimensions
using index_t = index_nd_t<NDIMS>;

namespace stream {

/// Callback receiving the axis positions of two entries in a stream `nest`
using on_nest_t = callback_t<const index_t &, const index_t &>;

/// Callback receiving the axis position shared by every entry of a stream
/// `interleave`
using on_interleave_t = callback_t<const index_t &>;

/// Convert from flat index to a position over the first `nd` axes of `shape`
///
/// @param flat The flat index
/// @param nd The number of axes
/// @param shape The size of each axis
inline index_t unravel(int flat, int nd, const index_t &shape) {
    index_t idx {};
    for (int d = nd - 1; d >= 0; --d) {
        idx[d] = flat % shape[d];
        flat /= shape[d];
    }
    return idx;
}

/// Convert from a position over the first `nd` axes of `shape` to flat index
///
/// @param idx The position
/// @param nd The number of axes
/// @param shape The size of each axis
///
/// @note This method is the inverse of `unravel`
inline int ravel(const index_t &idx, int nd, const index_t &shape) {
    int flat = 0;
    for (int d = 0; d < nd; ++d) {
        flat *= shape[d];
        flat += idx[d];
    }
    return flat;
}

} // namespace stream
} // namespace rvjit

#endif
