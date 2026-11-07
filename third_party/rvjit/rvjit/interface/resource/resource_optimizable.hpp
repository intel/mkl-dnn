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

#ifndef RVJIT_INTERFACE_RESOURCE_RESOURCE_OPTIMIZABLE_HPP
#define RVJIT_INTERFACE_RESOURCE_RESOURCE_OPTIMIZABLE_HPP

#include "rvjit/interface/resource/materializable.hpp"
#include "rvjit/interface/resource/resource.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {

/// A materializable object whose resource usage may optionally be reduced
///
/// @details Level 0 is the desired shape without cost reductions; every
///     later level trades resources for an unmeasured code degradation.
///     The caller is responsible for handling this tradeoff.
///
/// @pre The implementation of `level()` must derive from the object state
struct resource_optimizable_t : public materializable_t {
    static constexpr const char *NAME = "rvjit::resource_optimizable";

    /// Number of levels, 1 when there is nothing to trade
    virtual int levels() const { return 1; }

    /// Current level
    virtual int level() const { return 0; }

    /// Moves to level `l` from any current level
    ///
    /// @param l The level moved to
    ///
    /// @pre Implementors must ensure an absolute and idempotent operation
    virtual bool set_level(int l) {
        CHECK(l == 0, "no such level");
        return true;
    }

    /// Registers per class that moving from level `from` to level `to` makes
    /// available, at any point of the allocation, without changing the level
    ///
    /// @param from The level moved from
    /// @param to The level moved to
    ///
    /// @details Negative when `to` costs more; `demand_t::INF` when `from`
    ///     can never fit and `-INF` when `to` cannot; an unknown class is
    ///     unbounded, so the default knows nothing
    ///
    /// @note Must be an upper bound: an understated relief makes a search
    ///     skip a candidate that fits
    virtual demand_t relief(int from, int to) const {
        return from == to ? demand_t {} : demand_t::unknowns();
    }

    /// Resource classes some later level frees more of
    shortfall_t relieves() const {
        shortfall_t s;
        for (int l = level() + 1; l < levels(); ++l) {
            const demand_t d = relief(level(), l);
            for (int i = 0; i < demand_t::CLASSES; ++i) {
                const resource_t r = demand_t::at(i);
                if (!d.knows(r) || d[r] > 0) s = s | shortfall_t::of(r);
            }
        }
        return s;
    }

    /// Whether the object left its preferred level
    bool is_relaxed() const { return level() > 0; }
};

} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
