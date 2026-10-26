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

#ifndef RVJIT_IMPLEMENTATION_COMMON_ISA_BUFFER_ALLOCATOR_GROUP_HPP
#define RVJIT_IMPLEMENTATION_COMMON_ISA_BUFFER_ALLOCATOR_GROUP_HPP

#include <cstdint>

#include "rvjit/interface/isa/buffer/allocator.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/types.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace buffer {
namespace allocator {

/// Allocator for register groups with different input and output group sizes
///
/// @note Input and output candidate lists share one occupancy bitmask
///
/// @details A mixed-precision kernel may safely allocate registers for both
///     inputs and outputs from the same allocator as each allocation
///     invalidates candidates in both lists
/// @details The implementation eliminates candidates whose names overlap with
///     previous allocation to input and output register groups
struct group_t : rvjit::buffer::allocator_t {
    static constexpr const char *NAME = "rvjit::buffer::allocator";

    using rvjit::buffer::allocator_t::configure;

    bool configure(const buffer_t &source, const order_t &order, int in_gs,
            int out_gs, const buffer_t &input_only) override {
        if (in_gs != out_gs)
            CHECK_OR_RET(out_gs >= in_gs, false,
                    "unimplemented support for narrowing kernels");

        const resource_t served = resource_;
        *this = group_t();
        resource_ = served;
        in_group_size_ = in_gs;
        const order_t ordered = source.ordered(order);
        in_names_ = filter_valid_groups_(ordered, in_gs, input_only, true);
        if (in_gs == out_gs && input_only.is_empty()) return true;

        is_mixed_ = true;
        out_group_size_ = out_gs;
        out_names_ = filter_valid_groups_(ordered, out_gs, input_only, false);
        scratch_names_ = out_names_;
        for (int i = 0; i < ordered.size(); ++i)
            if (ordered[i] % out_gs == 0
                    && holds_any_(ordered[i], out_gs, input_only))
                scratch_names_.push_back(ordered[i]);
        return true;
    }

    name_t pop(allocation_t kind = allocation_t::input) override {
        return pop(1, kind).id(0);
    }
    buffer_t pop(int n, allocation_t kind = allocation_t::input) override {
        return alloc_block_(names_(kind), group_size_(kind), n, __func__);
    }
    int size(allocation_t kind = allocation_t::input) const override {
        return names_(kind).size();
    }
    int allocated(allocation_t kind = allocation_t::input) const override {
        return size(kind) - available(kind);
    }
    int available(allocation_t kind = allocation_t::input) const override {
        return count_available_(names_(kind), group_size_(kind));
    }
    int group_size(allocation_t kind = allocation_t::input) const override {
        return group_size_(kind);
    }
    name_t id(int i) const override { return in_names_[i]; }

    bool may_reserve(int out, int in) const override {
        if (!is_mixed_) return available(allocation_t::input) >= out + in;
        // Equal group sizes share their registers, minus the input-only ones
        if (in_group_size_ == out_group_size_)
            return available(allocation_t::output) >= out
                    && available(allocation_t::input) >= out + in;
        return available(allocation_t::output) >= out
                && available(allocation_t::input) >= in;
    }

private:
    order_t in_names_;
    int in_group_size_ = 1;

    order_t out_names_;
    int out_group_size_ = 1;

    // The output groups followed by the ones holding input-only registers
    order_t scratch_names_;

    bool is_mixed_ = false;

    const order_t &names_(allocation_t kind) const {
        if (kind == allocation_t::input || !is_mixed_) return in_names_;
        return kind == allocation_t::output ? out_names_ : scratch_names_;
    }

    int group_size_(allocation_t kind) const {
        return kind == allocation_t::input || !is_mixed_ ? in_group_size_
                                                         : out_group_size_;
    }

    /// Whether the group of `gsize` registers from `first` holds any of `regs`
    static bool holds_any_(name_t first, int gsize, const buffer_t &regs) {
        for (int j = 0; j < gsize; ++j)
            if (regs.has(first + j)) return true;
        return false;
    }

    /// Number of candidates in `cand`, grouped by `gsize` without overlapping
    /// any blocked physical register
    int count_available_(const order_t &cand, int gsize) const {
        const uint32_t blocked = occ_.taken.mask();
        const uint32_t bits = (1u << gsize) - 1u;
        int n = 0;
        for (int i = 0; i < cand.size(); ++i) {
            const uint32_t m = bits << cand[i];
            if (!(blocked & m)) ++n;
        }
        return n;
    }

    /// General block allocation with rollback on failure
    ///
    /// @note Restores the occupancy to its pre-call value if allocation of
    ///     any of the `n` registers fails partway through
    buffer_t alloc_block_(
            const order_t &cands, int phys, int n, const char *method) {
        const occupancy_t save = occ_;
        const uint32_t bits = (1u << phys) - 1u;
        name_t picked[buffer_t::MAXN];
        for (int k = 0; k < n; ++k) {
            bool found = false;
            for (int i = 0; i < cands.size(); ++i) {
                const name_t cand = cands[i];
                const buffer_t m = buffer_t::from_mask(bits << cand);
                if ((occ_.taken & m).is_empty()) {
                    occ_.taken = occ_.taken | m;
                    occ_.used = occ_.used | m;
                    picked[k] = cand;
                    found = true;
                    break;
                }
            }
            if (!found) {
                occ_ = save;
                LOG_FAIL(method, "insufficient registers");
                return {};
            }
        }
        return buffer_t(picked, n);
    }

    /// Keeps only ids that represent a valid group start for `gsize`
    ///
    /// @details A group holding an `input_only` register is kept first for
    ///     inputs and dropped for outputs
    static order_t filter_valid_groups_(const order_t &regs, int gsize,
            const buffer_t &input_only, bool input) {
        const auto holds_input_only = [&](name_t first) {
            return holds_any_(first, gsize, input_only);
        };
        order_t r;
        if (input)
            for (int i = 0; i < regs.size(); ++i)
                if (regs[i] % gsize == 0 && holds_input_only(regs[i]))
                    r.push_back(regs[i]);
        for (int i = 0; i < regs.size(); ++i)
            if (regs[i] % gsize == 0 && !holds_input_only(regs[i]))
                r.push_back(regs[i]);
        return r;
    }
};

} // namespace allocator
} // namespace buffer
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
