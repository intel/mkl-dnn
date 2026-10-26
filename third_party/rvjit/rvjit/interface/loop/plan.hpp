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

#ifndef RVJIT_INTERFACE_LOOP_PLAN_HPP
#define RVJIT_INTERFACE_LOOP_PLAN_HPP

#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/parameter.hpp"

namespace rvjit {
namespace loop {

/// Declarative plan to instantiate implementation-specific loop objects
///
struct plan_t {
    /// What the loop iterations do
    enum class kind_t {
        reduce, // The stages accumulate into shared state
        data_parallel, // The iterations are independent
    };

    kind_t kind = kind_t::data_parallel;
    control_flow::distance_t dist = control_flow::distance_t::short_;

    size_parameter_t size;
    int granularity = 1;
    int max_ur = 1;

    plan_t() = default;

    /// Creates a plan
    ///
    /// @param kind What the loop iterations do
    /// @param s The trip count
    /// @param u The largest unroll factor
    /// @param g The work units one iteration covers
    plan_t(kind_t kind, const size_parameter_t &s, int u, int g = 1)
        : kind(kind), size(s), granularity(g), max_ur(u) {}

    /// Creates a `reduce` plan
    ///
    /// @param s The trip count
    /// @param u The largest unroll factor
    /// @param g The work units one iteration covers
    static plan_t reduce(const size_parameter_t &s, int u, int g = 1) {
        return plan_t(kind_t::reduce, s, u, g);
    }

    /// Creates a `data_parallel` plan
    ///
    /// @param s The trip count
    /// @param u The largest unroll factor
    /// @param g The work units one iteration covers
    static plan_t data_parallel(const size_parameter_t &s, int u, int g = 1) {
        return plan_t(kind_t::data_parallel, s, u, g);
    }

    /// Copy of this plan whose body branches reach `d`
    ///
    /// @param d The reach of the branches around the loop body
    plan_t with_distance(control_flow::distance_t d) const {
        plan_t p = *this;
        p.dist = d;
        return p;
    }
};

} // namespace loop
} // namespace rvjit

#endif
