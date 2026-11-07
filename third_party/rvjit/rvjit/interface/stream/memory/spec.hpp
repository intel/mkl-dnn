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

#ifndef RVJIT_INTERFACE_STREAM_MEMORY_SPEC_HPP
#define RVJIT_INTERFACE_STREAM_MEMORY_SPEC_HPP

#include "rvjit/interface/isa/operation/memory.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/resource/resource.hpp"
#include "rvjit/interface/stream/buffer_map.hpp"
#include "rvjit/interface/stream/iterator/plan.hpp"

namespace rvjit {
namespace stream {
namespace memory_operand {

/// What an axis of the register ring means to a resource search
enum class axis_role_t {
    fixed, // Never shrunk
    reduce, // Shrunk first
    parallel, // Shrunk after the `reduce` axes
};

/// What a memory operand is, independent of how its resources are traded
///
/// @details The memory operation, and the origin, space, addressing and
///     end-state of the pointer walking it; none of it changes across the
///     candidates of a resource search
struct spec_t {
    memory::op_t op;

    /// Role of each axis of the register ring, which shares the space's axes
    axis_role_t role[NDIMS] = {};

    /// The register class the data registers are drawn from
    resource_t buffer_resource() const {
        return op.is_vector() ? resource_t::vectors
                : op.is_int() ? resource_t::integers
                              : resource_t::floats;
    }
    pointer_t origin;
    iterator::space_t space;
    iterator::addressing_t addressing = iterator::addressing_t::base_simm;
    iterator::update_t update = iterator::update_t::advance();

    /// The pointer walk, with the strategy left for a variant to choose
    iterator::request_t request() const {
        iterator::request_t r;
        r.origin = origin;
        r.space = space;
        r.addressing = addressing;
        r.update = update;
        return r;
    }

    /// Whether the operation and the walk form a packageable operand
    ///
    /// @note Vector memops require `base_only` addressing: the vector
    ///     instructions have no offset field to bake a position into
    bool is_valid() const {
        return space.is_valid() && origin.is_valid()
                && (!op.is_vector()
                        || addressing == iterator::addressing_t::base_only);
    }
};

/// What a resource search sets on a `spec_t`
struct variant_t {
    /// Strategy of the pointer walk; `any` lets the iterator choose
    iterator::strategy_id_t strategy = iterator::strategy_id_t::any();

    /// Cap on the registers the pointer walk may take; negative takes
    /// whatever is left once the ring is allocated
    int ntmp = -1;

    /// The destination register ring; unallocated storage is drawn from the
    /// pool at creation
    buffer_map_t ring;

    /// The type the unallocated ring's registers are drawn for; none is the
    /// operation's, e.g. a wider type its data is converted to in place
    type_t ring_type = type_t::none;

    /// The registers between a ring register, drawn or supplied, and the one
    /// the operation accesses, e.g. the upper part of a wider group a narrow
    /// load lands in (`type_parameter_t::offset`)
    int offset = 0;
};

} // namespace memory_operand
} // namespace stream
} // namespace rvjit

#endif
