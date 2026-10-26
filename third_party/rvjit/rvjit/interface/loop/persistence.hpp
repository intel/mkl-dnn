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

#ifndef RVJIT_INTERFACE_LOOP_PERSISTENCE_HPP
#define RVJIT_INTERFACE_LOOP_PERSISTENCE_HPP

#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/interface/loop/bind.hpp"
#include "rvjit/interface/loop/frame.hpp"
#include "rvjit/interface/loop/loop.hpp"
#include "rvjit/interface/resource/resource_optimizable.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace loop {

/// Where a set of frame bindings keeps its values, from preferred to cheapest
///
/// @details Level `l` binds every entry with the `l`-th storage of the list
///
/// @note Holds non-owning references; the frames must outlive it
struct persistence_t : public resource_optimizable_t {
    static constexpr const char *NAME = "rvjit::loop::persistence";
    static constexpr int CAPACITY = 8;
    static constexpr int STORAGES = 3;

    using storage_t = bind_t::storage_t;

    /// A buffer-backed binding first, with the stack as the only fallback
    static persistence_t buffered_then_stack() {
        return persistence_t({storage_t::buffer, storage_t::stack});
    }

    /// Always on the stack with nothing to trade
    static persistence_t stacked() { return persistence_t({storage_t::stack}); }

    /// Unbound first, then saved on the stack, freeing the variables'
    /// registers inside the frame's body
    static persistence_t on_demand() {
        return persistence_t({storage_t::off, storage_t::stack});
    }

    /// Binds the variables on frame `f`, preserved across its multiple trips
    ///
    /// @param f The frame
    /// @param v The first variable
    /// @param vs The remaining variables
    ///
    /// @note Variables without a buffer (absent features) are skipped
    template <typename... Vs>
    persistence_t &bind(frame_t &f, const variable_t &v, const Vs &...vs) {
        return bind_(f, /*multi_trip=*/true, {v, vs...});
    }

    /// Binds the variables on the frame of loop `l`
    ///
    /// @param l The loop
    /// @param v The first variable
    /// @param vs The remaining variables
    ///
    /// @note A loop dispatched once binds nothing, and has nothing to trade
    template <typename... Vs>
    persistence_t &bind(loop_t &l, const variable_t &v, const Vs &...vs) {
        return bind_(l.frame(), l.has_multiple_trips(), {v, vs...});
    }

    int levels() const override { return nstorages_; }

    /// @note The last level set, which a dropped transaction does not rewind
    int level() const override { return level_; }

    bool set_level(int l) override {
        CHECK(0 <= l && l < levels(), "no such level");
        bind_t spec = bind_t::reloaded().multi_trip();
        spec.storage = storages_[l];
        for (int i = 0; i < entries_.size(); ++i)
            CHECK(entries_[i].frame->bind(entries_[i].var, spec),
                    "unable to bind the variable");
        level_ = l;
        return true;
    }

    /// @details A `buffer` binding draws a register and a `stack` binding a
    ///     slot, per variable whose frame runs more than once
    demand_t demand() const override {
        demand_t d;
        for (int i = 0; i < entries_.size(); ++i)
            if (entries_[i].multi_trip) d = d + of_(storages_[level_]);
        return d;
    }

    /// @details Unbinding returns each variable's register to the frame's
    ///     body once it is saved, which is counted as freed
    demand_t relief(int from, int to) const override {
        const storage_t f = storages_[from], t = storages_[to];
        const demand_t lent = demand_t::of(resource_t::integers,
                f == storage_t::off && t != storage_t::off ? 1 : 0);
        demand_t d;
        for (int i = 0; i < entries_.size(); ++i)
            if (entries_[i].multi_trip) d = d + of_(f) - of_(t) + lent;
        return d;
    }

private:
    struct entry_t {
        frame_t *frame = nullptr;
        variable_t var;
        bool multi_trip = true;

        entry_t() = default;
        entry_t(frame_t *frame, const variable_t &var, bool multi_trip)
            : frame(frame), var(var), multi_trip(multi_trip) {}
    };

    persistence_t(std::initializer_list<storage_t> s) {
        for (storage_t st : s)
            storages_[nstorages_++] = st;
    }

    persistence_t &bind_(frame_t &f, bool multi_trip,
            std::initializer_list<variable_t> vars) {
        for (const variable_t &x : vars) {
            if (!x.has_buffer()) continue;
            LOG_FAIL_IF(!entries_.push_back(entry_t {&f, x, multi_trip}),
                    "a full set");
        }
        return *this;
    }

    /// Resources one binding of storage `s` draws
    static demand_t of_(storage_t s) {
        return s == storage_t::buffer   ? demand_t::of(resource_t::integers, 1)
                : s == storage_t::stack ? demand_t::of(resource_t::stack, 1)
                                        : demand_t {};
    }

    array_t<entry_t, CAPACITY> entries_;
    storage_t storages_[STORAGES] = {};
    int nstorages_ = 0;
    int level_ = 0;
};

} // namespace loop
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
