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

#ifndef RVJIT_INTERFACE_ISA_BUFFER_ALLOCATOR_HPP
#define RVJIT_INTERFACE_ISA_BUFFER_ALLOCATOR_HPP

#include <cstdint>

#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/resource/resource.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace buffer {

/// Discerns between allocations for inputs, outputs (accumulator) or scratch
///
/// @note Defaults to `input`, retaining call compatibility with a consumer
///     that never draws on separate output resources. `scratch` draws groups
///     of the output size, those holding input-only registers last
enum class allocation_t { input, output, scratch };

/// The outcome of taking buffers as either buffers or a resource shortfall
struct acquired_t {
    buffer_t regs;
    shortfall_t lack;

    /// Whether the buffers were taken
    explicit operator bool() const { return lack.ok(); }

    /// The shortfall to hand back to the search when the take failed
    shortfall_t miss() const { return lack; }

    /// The buffers taken
    const buffer_t &operator*() const { return regs; }

    /// The buffers taken
    const buffer_t *operator->() const { return &regs; }
};

/// Allocation state of one allocator, copied to checkpoint it
///
/// @details `taken` holds the registers live in the current scope, `given`
///     the ones their owner handed to the current scope, and `used` every
///     register drawn so far
struct occupancy_t {
    buffer_t taken, given, used;
};

/// Buffer allocation state built from a candidate list of buffer ids
///
/// @note Allocation mutates state, so these accessors are not `const`
struct allocator_t {
    static constexpr const char *NAME = "rvjit::buffer";

    virtual ~allocator_t() = default;

    /// Configure the allocator with a source and input/output group sizes
    ///
    /// @param src The buffer describing the whole collection of resources
    /// @param order Denotes the allocation order of names (default: ascending)
    /// @param ins The input buffer group size
    /// @param outs The output buffer group size
    ///
    /// @details One occupancy view is kept when `ins == outs`; otherwise
    ///     input and output use two filtered views sharing the occupancy
    ///
    /// @param input_only Registers only input allocations may draw, and are
    ///     drawn first by them
    ///
    /// @pre `outs` must be `>= ins` since narrowing is not implemented
    virtual bool configure(const buffer_t &src, const order_t &order, int ins,
            int outs, const buffer_t &input_only)
            = 0;

    /// Configures the allocator handing out `src` in the order `order`
    ///
    /// @param src The buffer describing the whole collection of resources
    /// @param order Denotes the allocation order of names
    /// @param ins The input buffer group size
    /// @param outs The output buffer group size
    bool configure(const buffer_t &src, const order_t &order, int ins = 1,
            int outs = 1) {
        return configure(src, order, ins, outs, buffer_t {});
    }

    /// Configures the allocator handing out `src` in ascending name order
    ///
    /// @param src The buffer describing the whole collection of resources
    /// @param ins The input buffer group size
    /// @param outs The output buffer group size
    bool configure(const buffer_t &src, int ins = 1, int outs = 1) {
        return configure(src, order_t {}, ins, outs, buffer_t {});
    }

    /// Allocates a single buffer id
    ///
    /// @param kind Whether the buffer is an input or an output
    ///
    /// @note Returns `no_name` when no buffer is left
    virtual name_t pop(allocation_t kind = allocation_t::input) = 0;

    /// Allocates `n` buffers as one flat buffer
    ///
    /// @param n The number of buffers
    /// @param kind Whether the buffers are inputs or outputs
    virtual buffer_t pop(int n, allocation_t kind = allocation_t::input) = 0;

    /// Allocates `n` buffers, or reports the resource this allocator serves
    ///
    /// @param n The number of buffers
    /// @param kind Whether the buffers are inputs or outputs
    ///
    /// @note Takes nothing on a shortfall
    acquired_t take(int n, allocation_t kind = allocation_t::input) {
        acquired_t a;
        if (n <= 0) return a;
        a.regs = pop(n, kind);
        if (a.regs.size() < n) {
            a.regs = buffer_t();
            a.lack = shortfall_t::of(resource_, n - available(kind));
        }
        return a;
    }

    /// General output (accumulator) take
    ///
    /// @param n The number of buffers
    acquired_t take_acc(int n) { return take(n, allocation_t::output); }

    /// Names the resource class this allocator serves in a `shortfall_t`
    ///
    /// @param r The resource class
    void serves(resource_t r) { resource_ = r; }

    /// Total candidate count
    ///
    /// @param kind Whether the inputs or the outputs are counted
    virtual int size(allocation_t kind = allocation_t::input) const = 0;

    /// Number of buffers allocated so far
    ///
    /// @param kind Whether the inputs or the outputs are counted
    virtual int allocated(allocation_t kind = allocation_t::input) const = 0;

    /// Remaining allocation headroom
    ///
    /// @param kind Whether the inputs or the outputs are counted
    virtual int available(allocation_t kind = allocation_t::input) const = 0;

    /// Registers in one buffer of `kind`
    ///
    /// @param kind Whether the inputs or the outputs are counted
    ///
    /// @note One, unless the allocator groups its registers
    virtual int group_size(allocation_t kind = allocation_t::input) const {
        (void)kind;
        return 1;
    }

    /// The id of the `i`-th candidate, regardless of allocation state
    ///
    /// @param i The position in the candidate list
    virtual name_t id(int i) const = 0;

    /// Whether `out` output and `in` input registers may yet be reserved
    ///
    /// @param out The number of output registers
    /// @param in The number of input registers
    ///
    /// @note On uniform precision settings, the sum must fit the budget; on
    ///     mixed precision settings, each category is tested separately
    virtual bool may_reserve(int out, int in) const {
        (void)out;
        (void)in;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Snapshot of the allocation state
    using checkpoint_t = occupancy_t;

    /// Captures the current allocation state
    checkpoint_t save() const { return occ_; }

    /// Reverts the allocation state to a `save()` result
    ///
    /// @param cp The state captured by `save()`
    ///
    /// @pre `cp` came from `save()` on this allocator, with no `configure()`
    ///     call since
    void restore(const checkpoint_t &cp) { occ_ = cp; }

    /// Opens a scope whose allocations draw `given` first
    ///
    /// @param given The registers drawn before any candidate
    ///
    /// @note Returns the state `leave()` closes the scope against
    checkpoint_t enter(const buffer_t &given = {}) {
        const checkpoint_t entry = occ_;
        occ_.taken = occ_.taken.filter(given);
        occ_.given = occ_.given | given;
        return entry;
    }

    /// Closes the scope `enter()` returned `entry` for
    ///
    /// @param entry The state `enter()` returned
    ///
    /// @note What the scope drew is free again for any later allocation
    void leave(const checkpoint_t &entry) {
        occ_.taken = entry.taken;
        occ_.given = entry.given;
    }

    /// Every register drawn so far, including by closed scopes
    const buffer_t &used() const { return occ_.used; }

protected:
    resource_t resource_ = resource_t::integers;
    occupancy_t occ_;
};

} // namespace buffer
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
