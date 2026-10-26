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

#ifndef RVJIT_INTERFACE_RESOURCE_OPTIMIZATION_GROUP_HPP
#define RVJIT_INTERFACE_RESOURCE_OPTIMIZATION_GROUP_HPP

#include "rvjit/interface/resource/resource_optimizable.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace optimization {

/// How a group steps through its members' levels
enum class order_t {
    depth_first, // each member is fully relaxed before the next one starts
    level_first, // every member reaches level `k` before any reaches `k + 1`
};

/// An ordered set of optimizables stepped as one
///
/// @details Levels are cumulative: level `l` of the group relaxes the first
///     `l` steps of the policy, leaving the remaining members at level 0
///
/// @note Holds non-owning references; the members must outlive the group
struct group_t : public resource_optimizable_t {
    static constexpr const char *NAME = "rvjit::optimization::group";
    static constexpr int CAPACITY = 8;

    /// Creates an empty group
    ///
    /// @param order How the group steps through its members' levels
    explicit group_t(order_t order = order_t::depth_first) : order_(order) {}

    /// Declares the members; the argument order is the relaxation order
    ///
    /// @param o The first member
    /// @param os The remaining members
    template <typename... Os>
    group_t &declare(resource_optimizable_t &o, Os &...os) {
        resource_optimizable_t *list[] = {&o, &os...};
        for (resource_optimizable_t *m : list)
            add_(*m);
        return *this;
    }

    /// Number of members
    int size() const { return members_.size(); }

    int levels() const override {
        int total = 1;
        for (int i = 0; i < size(); ++i)
            total += members_[i]->levels() - 1;
        return total;
    }

    /// @note Every step relaxes one member by one level, so the group level
    ///     is the sum of the members' levels
    int level() const override {
        int l = 0;
        for (int i = 0; i < size(); ++i)
            l += members_[i]->level();
        return l;
    }

    bool set_level(int l) override {
        CHECK(0 <= l && l < levels(), "no such level");
        bool ok = true;
        for (int i = 0; i < size(); ++i)
            ok = members_[i]->set_level(share_(i, l)) && ok;
        return ok && on_changed_();
    }

    /// @note Exact for independent members: members sharing one resource
    ///     need a group computing their joint relief instead
    demand_t relief(int from, int to) const override {
        demand_t d;
        for (int i = 0; i < size(); ++i)
            d = d + members_[i]->relief(share_(i, from), share_(i, to));
        return d;
    }

    demand_t demand() const override {
        demand_t d;
        for (int i = 0; i < size(); ++i)
            d = d + members_[i]->demand();
        return d;
    }

protected:
    void add_(resource_optimizable_t &o) {
        LOG_FAIL_IF(!members_.push_back(&o), "a full group");
    }

    /// Called after the members changed level
    virtual bool on_changed_() { return true; }

    resource_optimizable_t &member_(int i) const { return *members_[i]; }

    /// Level of member `i` when the group is at level `l`
    int share_(int i, int l) const {
        if (order_ == order_t::depth_first) {
            for (int j = 0; j < i; ++j)
                l -= members_[j]->levels() - 1;
            if (l < 0) return 0;
            const int max = members_[i]->levels() - 1;
            return l < max ? l : max;
        }
        // level_first: step `(j, k)` visits members in order, for k = 1, 2..
        int steps = 0, lvl = 0;
        for (int k = 1;; ++k) {
            bool any = false;
            for (int j = 0; j < size(); ++j) {
                if (members_[j]->levels() - 1 < k) continue;
                any = true;
                if (steps >= l) return lvl;
                ++steps;
                if (j == i) lvl = k;
            }
            if (!any) break;
        }
        return lvl;
    }

private:
    order_t order_;
    array_t<resource_optimizable_t *, CAPACITY> members_;
};

} // namespace optimization
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
