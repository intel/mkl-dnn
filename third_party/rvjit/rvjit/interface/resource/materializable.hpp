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

#ifndef RVJIT_INTERFACE_RESOURCE_MATERIALIZABLE_HPP
#define RVJIT_INTERFACE_RESOURCE_MATERIALIZABLE_HPP

#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/resource/resource.hpp"

namespace rvjit {

/// An object whose resources are drawn by an allocation step
///
/// @pre Every implementer answers queries from state, not cached values
struct materializable_t {
    virtual ~materializable_t() = default;

    /// Resources drawn to materialize this object
    ///
    /// @note A lower bound per class: an overstated count makes a scope
    ///     report a shortfall for an allocation that would fit
    virtual demand_t demand() const { return {}; }

    /// The buffer this object holds
    virtual buffer::buffer_t supplied() const { return {}; }

    /// Whether the object has the resources it needs and it may materialize
    virtual bool is_valid() const { return true; }

    /// Whether the materialization requires instructions to be emitted
    virtual bool needs_to_materialize() const { return false; }
};

} // namespace rvjit

#endif
