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

#ifndef RVJIT_INTERFACE_STREAM_BUFFER_MAP_HPP
#define RVJIT_INTERFACE_STREAM_BUFFER_MAP_HPP

#include <algorithm>
#include <utility>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {

/// How the positions of a dense map are packed
///
/// @details `dense` varies the last axis fastest, `transposed` the first
enum class order_t { dense, transposed };

/// Associates a register to a position on a multi-dimensional space
///
/// @details Position `idx` resolves to `storage.id((sum idx[d] * stride[d])
///     % modulus)`; constructing from a `shape` gives a densely packed map
///
/// @note `shape` is the extent of the operand, independent of `modulus`
template <int ND = NDIMS>
struct buffer_map_nd_t {
    buffer::buffer_t storage;
    index_nd_t<ND> shape {};
    index_nd_t<ND> stride {};
    int modulus = 0;

    buffer_map_nd_t() = default;

    /// Creates a densely packed map over `shape`
    ///
    /// @param storage The registers backing the map
    /// @param shape The extent of the operand
    /// @param order How the positions are packed
    buffer_map_nd_t(buffer::buffer_t storage, const index_nd_t<ND> &shape,
            order_t order = order_t::dense)
        : storage(std::move(storage)), shape(shape), order_(order) {
        pack_();
        modulus = 1;
        for (int d = 0; d < ND; ++d)
            modulus *= shape[d];
    }

    /// Constructs an unresolved map over `shape`, with no storage yet
    ///
    /// @param shape The extent of the operand
    /// @param order How the positions are packed
    explicit buffer_map_nd_t(
            const index_nd_t<ND> &shape, order_t order = order_t::dense)
        : buffer_map_nd_t(buffer::buffer_t(), shape, order) {}

    /// Wraps the positions at `depth` registers, and has `reshape` repack them
    ///
    /// @param depth The number of registers the positions wrap at
    ///
    /// @note A flat ring read in the order its positions are produced
    buffer_map_nd_t &wrapped(int depth) {
        modulus = depth;
        follows_ = true;
        return *this;
    }

    /// Repacks a wrapped map over a new `shape`, keeping its `modulus`
    ///
    /// @param shape The new extent of the operand
    ///
    /// @note Does nothing unless the map was `wrapped`
    void reshape(const index_nd_t<ND> &shape) {
        if (!follows_) return;
        this->shape = shape;
        pack_();
    }

    /// Replaces the position-to-register layout, keeping `storage`/`shape`
    ///
    /// @param stride The register distance of one step along each axis
    /// @param modulus The number of registers the positions wrap at
    buffer_map_nd_t &layout(const index_nd_t<ND> &stride, int modulus) {
        this->stride = stride;
        this->modulus = modulus;
        follows_ = false;
        return *this;
    }

    /// Replaces the layout, wrapping at the extent the strides address
    ///
    /// @param stride The register distance of one step along each axis
    buffer_map_nd_t &layout(const index_nd_t<ND> &stride) {
        int extent = 0;
        for (int d = 0; d < ND; ++d)
            extent = std::max(extent, shape[d] * stride[d]);
        return layout(stride, extent);
    }

    /// The register at `idx`
    ///
    /// @param idx The position
    ///
    /// @pre `is_valid()`
    name_t operator()(const index_nd_t<ND> &idx) const {
        int off = 0;
        for (int d = 0; d < ND; ++d)
            off += idx[d] * stride[d];
        return name_t(storage.id(off % modulus));
    }

    /// Whether the map was `wrapped` and still repacks on `reshape`
    bool is_wrapped() const { return follows_; }

    /// Number of registers the layout addresses
    int total() const { return modulus; }

    /// Whether `storage` covers every register the layout addresses
    bool is_valid() const { return modulus > 0 && storage.size() >= modulus; }

private:
    order_t order_ = order_t::dense;
    bool follows_ = false;

    void pack_() {
        int t = 1;
        if (order_ == order_t::dense) {
            for (int d = ND - 1; d >= 0; --d) {
                stride[d] = t;
                t *= shape[d];
            }
        } else {
            for (int d = 0; d < ND; ++d) {
                stride[d] = t;
                t *= shape[d];
            }
        }
    }
};

/// `buffer_map_nd_t`, fixed at `NDIMS` dimensions
using buffer_map_t = buffer_map_nd_t<NDIMS>;

} // namespace rvjit

#endif
