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

#ifndef RVJIT_IMPLEMENTATION_COMMON_ISA_BUFFER_ALLOCATOR_NAMED_HPP
#define RVJIT_IMPLEMENTATION_COMMON_ISA_BUFFER_ALLOCATOR_NAMED_HPP

#include "rvjit/interface/isa/buffer/allocator.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace buffer {
namespace allocator {

/// Allocator over a candidate list of individual buffer names
///
/// @details Draws the registers given to the current scope first, then the
///     first free candidate in order
///
/// @note `allocation_t` is ignored on every method
///
/// @pre Only accepts configurations where `in_gs == out_gs == 1`
struct named_t : rvjit::buffer::allocator_t {
    static constexpr const char *NAME = "rvjit::buffer::allocator";

    using rvjit::buffer::allocator_t::configure;

    bool configure(const buffer_t &source, const order_t &order, int in_gs,
            int out_gs, const buffer_t &input_only) override {
        CHECK(in_gs == 1 && out_gs == 1, "requires in_gs == 1 && out_gs == 1");
        CHECK(input_only.is_empty(), "no input-only registers on this kind");
        candidates_ = source.ordered(order);
        candidate_set_ = buffer_t(candidates_);
        occ_ = occupancy_t {};
        return true;
    }

    name_t pop(allocation_t = allocation_t::input) override {
        CHECK_OR_RET(available() > 0, no_name, "insufficient registers");
        return draw_();
    }

    buffer_t pop(int n, allocation_t = allocation_t::input) override {
        CHECK_OR_RET(n <= available(), {}, "insufficient registers");
        name_t tmp[buffer_t::MAXN];
        for (int i = 0; i < n; ++i)
            tmp[i] = draw_();
        return buffer_t(tmp, n);
    }

    int size(allocation_t = allocation_t::input) const override {
        return candidates_.size();
    }

    int allocated(allocation_t = allocation_t::input) const override {
        return (candidate_set_ & occ_.taken).size();
    }

    /// @note Includes the registers given to the current scope
    int available(allocation_t = allocation_t::input) const override {
        return (candidate_set_ | occ_.given).filter(occ_.taken).size();
    }

    name_t id(int i) const override { return candidates_[i]; }

    bool may_reserve(int out, int in) const override {
        return available() >= out + in;
    }

private:
    /// Takes the first given register, else the first free candidate
    ///
    /// @pre `available() > 0`
    name_t draw_() {
        const buffer_t blocked = occ_.taken;
        const buffer_t gifts = occ_.given.filter(blocked);
        for (int i = 0; i < candidates_.size(); ++i) {
            const name_t c = candidates_[i];
            if (gifts.is_empty() ? !blocked.has(c) : gifts.has(c))
                return take_(c);
        }
        return gifts.is_empty() ? no_name : take_(gifts.id(0));
    }

    name_t take_(name_t r) {
        occ_.taken.set(r);
        occ_.used.set(r);
        return r;
    }

    order_t candidates_;
    buffer_t candidate_set_;
};

} // namespace allocator
} // namespace buffer
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
