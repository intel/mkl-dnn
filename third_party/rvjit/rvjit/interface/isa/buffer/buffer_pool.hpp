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

#ifndef RVJIT_INTERFACE_ISA_BUFFER_BUFFER_POOL_HPP
#define RVJIT_INTERFACE_ISA_BUFFER_BUFFER_POOL_HPP

#include <utility>
#include <type_traits>

#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/isa/buffer/allocator.hpp"
#include "rvjit/interface/isa/buffer/architecture.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/buffer/uses_transactions.hpp"
#include "rvjit/interface/isa/buffered.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace buffer {

/// Saved state of the integer, float, and vector allocators
struct pool_checkpoint_t {
    allocator_t::checkpoint_t ints, floats, vectors;
};

/// How the vector register file is grouped for a kernel
///
/// @details `in` and `out` are the input and output buffer group sizes, and
///     `input_only` holds the registers only inputs may draw
struct vector_layout_t {
    int in = 1;
    int out = 1;
    buffer_t input_only;

    vector_layout_t() = default;

    /// Creates a vector layout
    ///
    /// @param in The input register group size
    /// @param out The output register group size
    /// @param input_only The registers only inputs may draw
    vector_layout_t(int in, int out, const buffer_t &input_only = buffer_t())
        : in(in), out(out), input_only(input_only) {}

    /// Whether the groups are realizable: positive, outputs no smaller
    bool is_valid() const { return in >= 1 && out >= in; }
};

/// Pool with mutable state encapsulating buffer allocation expressions
///
/// @note Allocation mutates the pool state
struct buffer_pool_t : public uses_transactions_t<pool_checkpoint_t> {
    static constexpr const char *NAME = "rvjit::buffer";

    virtual ~buffer_pool_t() = default;

    /// The integer buffer allocator
    virtual allocator_t &integers() = 0;

    /// The float buffer allocator
    ///
    /// @pre The candidate list must not contain any callee-saved id
    virtual allocator_t &floats() = 0;

    /// The vector buffer allocator
    virtual allocator_t &vectors() = 0;

    /// Runs `f` in a scope whose allocations end with it
    ///
    /// @param f A callable returning `shortfall_t`
    ///
    /// @note What `f` allocates is free again for any later allocation
    ///
    /// @pre The scope matches the lifetime of its values in the emitted code
    template <typename F>
    shortfall_t scope(F &&f) {
        return scope(buffer_t {}, std::forward<F>(f));
    }

    /// Runs `f` in a scope whose allocations end with it, drawing `given` first
    ///
    /// @param given The integer registers drawn before any candidate
    /// @param f A callable returning `shortfall_t`
    ///
    /// @pre The owner of `given` leaves them alone during the scope
    template <typename F>
    shortfall_t scope(const buffer_t &given, F &&f) {
        const pool_checkpoint_t entry {
                integers().enter(given), floats().enter(), vectors().enter()};
        const shortfall_t s = f();
        integers().leave(entry.ints);
        floats().leave(entry.floats);
        vectors().leave(entry.vectors);
        return s;
    }

    /// The allocator of the register class suiting the `op` storage form and
    /// numeric kind
    ///
    /// @param op The object whose type traits select the allocator
    allocator_t &allocator(const typed_t &op) {
        if (op.is_vector()) return vectors();
        return op.is_int() ? integers() : floats();
    }

    /// Allocates `n` buffers suiting the `op` storage form and numeric kind
    ///
    /// @param n The number of buffers
    /// @param op The object whose type traits select the allocator
    buffer_t new_buffer(int n, const typed_t &op) {
        return allocator(op).pop(n);
    }

    /// What `d` misses of the registers the current scope may still draw
    ///
    /// @param d The demand checked
    ///
    /// @note The stack and the classes `d` does not know are never short
    shortfall_t lacks(const demand_t &d) {
        shortfall_t s;
        allocator_t *const of[] = {&integers(), &floats(), &vectors()};
        for (int i = 0; i < 3; ++i) {
            const resource_t r = demand_t::at(i);
            if (!d.knows(r)) continue;
            const int lack = d[r] >= demand_t::INF ? demand_t::INF
                                                   : d[r] - of[i]->available();
            if (lack > 0) s = s | shortfall_t::of(r, lack);
        }
        return s;
    }

    /// Allocates the integer buffers `obj` still needs, if any
    ///
    /// @param obj The object receiving the buffers
    ///
    /// @note A `drop()` returns the registers allocated since its `begin()`,
    ///     so `obj` must not be used afterwards
    bool allocate(buffered_t &obj) {
        const int n = obj.buffer_needed();
        if (n <= 0) return true;

        const buffer_t b = integers().pop(n);
        CHECK_OR_RET(b.size() == n, false, "insufficient registers");

        obj.set_buffer(b);
        return true;
    }

    /// Allocates an integer buffer and creates a slot referencing it
    ///
    /// @note Returns a slot that is not `is_valid()` on a shortfall
    slot_t new_buffer_source() {
        const name_t n = integers().pop();
        CHECK_OR_RET(n != no_name, slot_t {}, "insufficient registers");
        return slot_t::buffer(reg_t(n));
    }

    /// Creates a new constant from a source, allocating a buffer if needed
    ///
    /// @param src The origin of the value, resolved as `const_t(src)` does
    ///
    /// @note Returns a constant that is not `is_valid()` on an invalid source
    ///     or an allocation shortfall
    const_t new_constant(const source_t &src) {
        CHECK_OR_RET(src.is_valid() && !src.is_none(), const_t::invalid(),
                "an invalid source");

        const_t c(src);

        LOG_FAIL_IF(!allocate(c), "failed to allocate buffer for new constant");
        return c;
    }

    /// Creates a new variable from a source, allocating a buffer if needed
    ///
    /// @param src The origin of the value
    /// @param r The register used instead of a new allocation, if supplied
    ///
    /// @note An `adopt` source hands its register over to the variable, which
    ///     may overwrite it; a `copy` source gets a private copy instead
    /// @note Returns a variable without a buffer on an invalid source or an
    ///     allocation shortfall
    variable_t new_variable(
            const source_t &src = source_t::none(), const reg_t &r = reg_t()) {
        CHECK_OR_RET(src.is_valid(), variable_t(), "an invalid source");

        variable_t v = variable_t::from(src, r);
        LOG_FAIL_IF(!allocate(v), "failed to allocate buffer for new variable");
        return v;
    }

    /// Configures a single vector register file view for group size `gs`
    ///
    /// @param gs The input and output buffer group size
    virtual void set_vector_group_size(int gs) {
        (void)gs;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// Configures up to two vector register file views for two group sizes
    ///
    /// @param in The input buffer group size
    /// @param out The output buffer group size
    /// @param input_only The registers only input allocations may draw
    ///
    /// @note Input allocations draw `input_only` first and outputs never do
    ///
    /// @pre `out >= in`
    virtual void set_vector_group_size(
            int in, int out, const buffer_t &input_only = {}) {
        (void)in;
        (void)out;
        (void)input_only;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// Groups the vector register file as `layout` says
    ///
    /// @param layout The group sizes and the registers only inputs may draw
    ///
    /// @pre No transaction is open, since regrouping invalidates checkpoints
    bool configure_vectors(const vector_layout_t &layout) {
        CHECK(open_transactions() == 0,
                "the vector layout changed inside a transaction");
        CHECK(layout.is_valid(), "an invalid vector layout (in %d, out %d)",
                layout.in, layout.out);
        if (layout.in == layout.out && layout.input_only.is_empty())
            set_vector_group_size(layout.in);
        else
            set_vector_group_size(layout.in, layout.out, layout.input_only);
        return true;
    }

protected:
    /// @pre `set_vector_group_size()` is not called while a transaction is open
    bool on_begin_(pool_checkpoint_t &cp) override {
        cp = {integers().save(), floats().save(), vectors().save()};
        return true;
    }

    bool on_drop_(const pool_checkpoint_t &cp) override {
        integers().restore(cp.ints);
        floats().restore(cp.floats);
        vectors().restore(cp.vectors);
        return true;
    }
};

/// Assembles a buffer pool from allocator implementations
///
/// @details Templare arguments must be default-constructible `allocator_t`
template <typename IntAlloc, typename FloatAlloc, typename VecAlloc>
struct buffer_compose_t : public buffer_pool_t {
    /// Configures the allocators from the register files
    ///
    /// @param source The register files, which are copied
    explicit buffer_compose_t(architecture_t &source) {
        int_.serves(resource_t::integers);
        float_.serves(resource_t::floats);
        vec_.serves(resource_t::vectors);
        int_.configure(source.integer.file(), source.integer.order());
        float_.configure(source.floating.file(), source.floating.order());
        vector_source_ = source.vector.file();
        vector_order_ = source.vector.order();
    }

    allocator_t &integers() override { return int_; }
    allocator_t &floats() override { return float_; }
    allocator_t &vectors() override { return vec_; }

    void set_vector_group_size(int gs) override {
        vec_.configure(vector_source_, vector_order_, gs, gs);
    }

    void set_vector_group_size(
            int in, int out, const buffer_t &input_only = {}) override {
        vec_.configure(vector_source_, vector_order_, in, out, input_only);
    }

protected:
    IntAlloc int_;
    FloatAlloc float_;
    VecAlloc vec_;
    buffer_t vector_source_;
    order_t vector_order_;
};

} // namespace buffer
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
