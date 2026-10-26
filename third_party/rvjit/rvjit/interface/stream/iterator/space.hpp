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

#ifndef RVJIT_INTERFACE_STREAM_ITERATOR_SPACE_HPP
#define RVJIT_INTERFACE_STREAM_ITERATOR_SPACE_HPP

#include <initializer_list>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace stream {
namespace iterator {

/// A stride value sourced from a register or simm12 field
using stride_t = const_t;

/// Describes an axis on a multi-dimensional pointer iteration space
///
/// @note Number of elements (`size`) and the `stride` between them
struct axis_t {
    int size = 0;
    stride_t stride {0};

    axis_t() = default;

    /// Creates an axis
    ///
    /// @param size The number of elements
    /// @param stride The distance between consecutive elements
    axis_t(int size, const stride_t &stride) : size(size), stride(stride) {}

    /// Whether or not the axis' size and stride are valid
    bool is_valid() const { return size > 0 && stride.is_valid(); }

    /// Whether or not the axis' stride is a compile-time constant value
    bool has_imm() const { return stride.has_imm(); }
};

/// Describes an N-dimensional iterator space
///
/// @note Aggregates a collection of `axis_t` elements, one for each dimension
///
/// @details Dimensions are expressed in an array from outer-to-inner order
struct space_t : array_t<axis_t, NDIMS> {
    using array_t<axis_t, NDIMS>::size;

    space_t() = default;

    /// Builds an outer-to-inner space from per-axis sizes and strides
    ///
    /// @param sizes The number of elements of each axis
    /// @param strides The distance between consecutive elements of each axis
    ///
    /// @note Axes beyond the shorter list are dropped
    space_t(std::initializer_list<int> sizes,
            std::initializer_list<stride_t> strides) {
        auto s = strides.begin();
        for (auto n = sizes.begin(); n != sizes.end() && s != strides.end();
                ++n, ++s)
            axis(*n, *s);
    }

    /// Appends the next inner axis
    ///
    /// @param size The number of elements
    /// @param stride The distance between consecutive elements
    ///
    /// @pre The caller must inform axes in outer-to-inner order
    space_t &axis(int size, const stride_t &stride) {
        if (this->n_ < NDIMS) this->e_[this->n_++] = axis_t(size, stride);
        return *this;
    }

    /// Number of dimensions on this iterator space
    int ndims() const { return size(); }

    /// Get the axis associated with a given dimension id
    ///
    /// @param dim The dimension
    const axis_t &axis(int dim) const { return this->e_[dim]; }

    /// Checks if all dimensions are valid
    bool is_valid() const {
        if (ndims() <= 0) return false;
        for (int d = 0; d < ndims(); ++d)
            if (!(*this)[d].is_valid()) return false;
        return true;
    }

    /// A copy with every axis's `size` overwritten from `sizes`
    ///
    /// @param sizes The new size of each axis
    ///
    /// @pre `sizes[d] > 0` for every `d < ndims()`
    space_t resized(const index_t &sizes) const {
        space_t out;
        for (int d = 0; d < ndims(); ++d)
            out.axis(sizes[d], (*this)[d].stride);
        return out;
    }
};

} // namespace iterator
} // namespace stream
} // namespace rvjit

#endif
