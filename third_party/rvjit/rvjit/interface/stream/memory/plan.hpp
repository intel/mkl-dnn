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

#ifndef RVJIT_INTERFACE_STREAM_MEMORY_PLAN_HPP
#define RVJIT_INTERFACE_STREAM_MEMORY_PLAN_HPP

#include <algorithm>

#include "rvjit/interface/resource/optimization/group.hpp"
#include "rvjit/interface/resource/resource_optimizable.hpp"
#include "rvjit/interface/stream/iterator/iterator.hpp"
#include "rvjit/interface/stream/memory/spec.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace memory_operand {

/// Shrinks the axes of one role of a register ring, one register at a time
///
/// @details Level 0 is the requested extents; each level takes one more
///     position off the first axis of the role still above 1, then the next
struct ring_shrink_t : public resource_optimizable_t {
    static constexpr const char *NAME = "rvjit::stream::memory_operand::ring";

    /// Creates the levels of one role of a ring
    ///
    /// @param roles The role of each axis of the ring
    /// @param which The role whose axes are shrunk
    /// @param resource The register class the ring is drawn from
    /// @param base The requested extents
    /// @param cur Receives the extents of the current level
    ring_shrink_t(const axis_role_t *roles, axis_role_t which,
            resource_t resource, const index_t &base, index_t &cur)
        : roles_(roles)
        , which_(which)
        , resource_(resource)
        , base_(base)
        , cur_(cur) {}

    /// Registers a ring of extents `ext` holds
    ///
    /// @param ext The extents
    static int count(const index_t &ext) {
        int n = 1;
        for (int d = 0; d < NDIMS; ++d)
            n *= ext[d];
        return n;
    }

    /// The extents `ext` takes with this role's axes at level `l`
    ///
    /// @param l The level
    /// @param ext The extents of the axes of the other roles
    index_t extents(int l, index_t ext) const {
        for (int d = 0; d < NDIMS; ++d) {
            if (roles_[d] != which_) continue;
            const int take = std::min(l, base_[d] - 1);
            ext[d] = base_[d] - take;
            l -= take;
        }
        return ext;
    }

    int levels() const override {
        int n = 1;
        for (int d = 0; d < NDIMS; ++d)
            if (roles_[d] == which_) n += base_[d] - 1;
        return n;
    }

    int level() const override {
        int l = 0;
        for (int d = 0; d < NDIMS; ++d)
            if (roles_[d] == which_) l += base_[d] - cur_[d];
        return l;
    }

    bool set_level(int l) override {
        CHECK(0 <= l && l < levels(), "no such level");
        cur_ = extents(l, cur_);
        return true;
    }

    demand_t demand() const override {
        return demand_t::of(resource_, count(cur_));
    }

    /// @note Exact with the other roles' axes at their current extents
    demand_t relief(int from, int to) const override {
        return demand_t::of(resource_, count(extents(from, cur_)))
                - demand_t::of(resource_, count(extents(to, cur_)));
    }

private:
    const axis_role_t *roles_;
    axis_role_t which_;
    resource_t resource_;
    const index_t &base_;
    index_t &cur_;
};

/// A memory operand as a resource search sees it: what to create and the
/// levels it can be relaxed through
///
/// @details The levers shrink the ring along its `reduce` axes, then its
///     `parallel` ones, and step the pointer walk down the strategy ladder;
///     for an integer buffer they form one lever, since pointers and data
///     compete for the same registers
///
/// @note A `wrapped` ring shrinks as axis 0, so it needs `role[0] == reduce`
///
/// @note Not copyable: the levers refer to the plan's own state
struct plan_t : public materializable_t {
    static constexpr const char *NAME = "rvjit::stream::memory_operand::plan";

    /// Creates the plan of a memory operand
    ///
    /// @param set The strategies the pointer walk resolves over
    /// @param spec What the operand is
    /// @param base The preferred variant, at level 0
    plan_t(const iterator::strategy_set_t &set, const spec_t &spec,
            const variant_t &base)
        : spec(spec)
        , base_(base)
        , wrapped_(base.ring.is_wrapped())
        , walk_(set, spec.request(), base.strategy, base.ntmp, strategy_)
        , reduce_(roles_, axis_role_t::reduce, spec.buffer_resource(), extent_,
                  cur_)
        , other_(roles_, axis_role_t::parallel, spec.buffer_resource(), extent_,
                  cur_)
        , ring_(*this)
        , all_(*this) {
        for (int d = 0; d < NDIMS; ++d) {
            roles_[d] = wrapped_ && d > 0 ? axis_role_t::fixed : spec.role[d];
            extent_[d] = wrapped_ ? (d == 0 ? base.ring.modulus : 1)
                                  : base.ring.shape[d];
        }
        cur_ = extent_;
        ring_.declare(reduce_, other_);
        all_.declare(reduce_, walk_, other_);
    }

    plan_t(const plan_t &) = delete;
    plan_t &operator=(const plan_t &) = delete;

    spec_t spec;

    /// The variant of the current levels
    variant_t variant() const {
        variant_t v = base_;
        v.strategy = strategy_;
        v.ring = wrapped_ ? buffer_map_t(base_.ring.shape).wrapped(cur_[0])
                          : buffer_map_t(cur_);
        return v;
    }

    /// The ring and the pointer walk at the current levels
    demand_t demand() const override {
        return reduce_.demand() + walk_.demand();
    }

    /// The levers to hand to a search, in preference order
    array_t<resource_optimizable_t *, 2> levers() {
        array_t<resource_optimizable_t *, 2> l;
        if (spec.op.is_int()) {
            l.push_back(&all_);
        } else {
            l.push_back(&ring_);
            l.push_back(&walk_);
        }
        return l;
    }

private:
    /// Levers of one plan stepped as one, priced jointly since the ring
    /// levers share one ring
    struct lever_t : public optimization::group_t {
        explicit lever_t(const plan_t &plan) : plan_(plan) {}

        demand_t demand() const override {
            demand_t d = plan_.reduce_.demand();
            for (int i = 0; i < size(); ++i)
                if (&member_(i) == &plan_.walk_) d = d + plan_.walk_.demand();
            return d;
        }

        demand_t relief(int from, int to) const override {
            index_t f = plan_.extent_, t = plan_.extent_;
            demand_t walk;
            for (int i = 0; i < size(); ++i) {
                const int lf = share_(i, from), lt = share_(i, to);
                if (&member_(i) == &plan_.walk_) {
                    walk = plan_.walk_.relief(lf, lt);
                    continue;
                }
                const ring_shrink_t &r = &member_(i) == &plan_.reduce_
                        ? plan_.reduce_
                        : plan_.other_;
                f = r.extents(lf, f);
                t = r.extents(lt, t);
            }
            const resource_t cls = plan_.spec.buffer_resource();
            return demand_t::of(cls, ring_shrink_t::count(f))
                    - demand_t::of(cls, ring_shrink_t::count(t)) + walk;
        }

    private:
        const plan_t &plan_;
    };

    variant_t base_;
    bool wrapped_;
    axis_role_t roles_[NDIMS] = {};
    index_t extent_ {}, cur_ {};
    iterator::strategy_id_t strategy_;
    iterator::strategy_ladder_t walk_;
    ring_shrink_t reduce_, other_;
    lever_t ring_, all_;
};

} // namespace memory_operand
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
