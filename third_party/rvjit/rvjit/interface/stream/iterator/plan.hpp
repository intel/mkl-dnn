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

#ifndef RVJIT_INTERFACE_STREAM_ITERATOR_PLAN_HPP
#define RVJIT_INTERFACE_STREAM_ITERATOR_PLAN_HPP

#include <array>

#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/stream/buffer_map.hpp"
#include "rvjit/interface/stream/iterator/space.hpp"
#include "rvjit/interface/stream/iterator/types.hpp"

namespace rvjit {
namespace stream {
namespace iterator {

/// Plan to iterate over one axis of a pointer iteration space
struct axis_plan_t {
    /// Which strategy this axis resolved to
    strategy_id_t strategy = strategy_id_t::any();

    /// Number of registers required to implement the strategy
    int ngpr = 0;

    /// Registers available to implement the strategy
    buffer::buffer_t regs;

    // Data owned by the `pivot` strategy, ignored by any other

    /// Index positions covered by each pivot register via offsets
    int positions_per_pivot = 0;

    /// Shift applied during pivot init to center the simm12 window
    int shift = 0;

    /// Axis index where this axis's one-register-per-position pivot chain
    /// begins
    ///
    /// @note Only meaningful for a `pivot` axis on the one-register-per-
    ///     position path; the chain's last axis (the owner) holds one
    ///     register per combined position, earlier axes have `ngpr == 0`
    ///
    /// @pre Across a `resized()` call, only the `chain_start` axis may change
    ///     size; resizing another axis of a built chain needs a fresh `init()`
    int chain_start = -1;

    /// Whether the axis's first register is the origin's, `regs` holding the
    /// others
    bool takes_origin = false;

    /// Registers the axis addresses through, the origin's included
    int nregs() const { return ngpr + (takes_origin ? 1 : 0); }

    /// Whether or not the plan needs more registers to materialize
    bool needs_regs() const { return regs.size() < ngpr; }

    /// Whether or not the plan can be correctly materialized
    bool is_valid() const { return !strategy.is_any() && !needs_regs(); }
};

/// What a caller asks of an iterator over a pointer_t iteration space
///
/// @note `strategy` is a hint: `any` lets the resolver choose per axis
struct request_t {
    /// Pointer to the first addressable element
    pointer_t origin;

    /// Multi-dimensional iterator space
    space_t space;

    /// How the underlying memory operation addresses elements (base or
    /// base+imm)
    addressing_t addressing = addressing_t::base_simm;

    /// What happens to the base pointer once the whole space is iterated
    update_t update = update_t::advance();

    /// The iterator strategy hint for this request
    strategy_id_t strategy = strategy_id_t::any();

    /// Number of dimensions
    int ndims() const { return space.ndims(); }

    /// Whether or not this request uses a constant offset across iterations
    bool is_base_only() const { return addressing == addressing_t::base_only; }

    /// Whether or not this request leverages a 12-bit signed immediate offset
    bool is_base_simm() const { return addressing == addressing_t::base_simm; }
};

/// A request resolved into one implementation plan per axis
///
/// @note Each axis's `strategy` names the strategy that resolved it
struct plan_t : request_t {
    plan_t() = default;

    /// Creates a plan with no axis resolved
    ///
    /// @param request What the caller asks of the iterator
    plan_t(const request_t &request) : request_t(request) {}

    /// Obtain the implementation plan associated to an axis
    ///
    /// @param dim The axis
    const axis_plan_t &axis(int dim) const { return axis_[dim]; }

    /// Obtain the implementation plan associated to an axis
    ///
    /// @param dim The axis
    axis_plan_t &axis(int dim) { return axis_[dim]; }

    /// Whether every axis has a resolved strategy over a valid iterator space
    bool is_valid() const {
        if (!space.is_valid()) return false;
        for (int d = 0; d < ndims(); ++d)
            if (!axis_[d].is_valid()) return false;
        return true;
    }

    /// Whether any axis still needs registers
    bool needs_regs() const {
        for (int d = 0; d < ndims(); ++d)
            if (axis_[d].needs_regs()) return true;
        return false;
    }

    /// Registers the axes need in total
    int ngpr() const {
        int n = 0;
        for (int d = 0; d < ndims(); ++d)
            n += axis_[d].ngpr;
        return n;
    }

    /// The offset an advancing origin keeps past its span, the `shift` of the
    /// axis taking it
    int origin_bias() const {
        for (int d = 0; d < ndims(); ++d)
            if (axis_[d].takes_origin) return axis_[d].shift;
        return 0;
    }

    /// A copy with `space`'s axis sizes overwritten from `sizes`
    ///
    /// @param sizes The new size of each axis
    ///
    /// @pre The plan's origin has not moved (e.g. via `update_t::advance`)
    ///     since the last `init()`
    plan_t resized(const index_t &sizes) const {
        plan_t out = *this;
        out.space = space.resized(sizes);
        return out;
    }

private:
    std::array<axis_plan_t, NDIMS> axis_ {};
};

} // namespace iterator
} // namespace stream
} // namespace rvjit

#endif
