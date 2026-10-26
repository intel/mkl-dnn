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

#ifndef RVJIT_INTERFACE_STREAM_MEMORY_MEMORY_OPERAND_HPP
#define RVJIT_INTERFACE_STREAM_MEMORY_MEMORY_OPERAND_HPP

#include <algorithm>
#include <array>
#include <utility>

#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/operation/memory.hpp"
#include "rvjit/interface/resource/resource.hpp"
#include "rvjit/interface/stream/iterator/iterator.hpp"
#include "rvjit/interface/stream/memory/plan.hpp"
#include "rvjit/interface/stream/memory/spec.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"
#include "rvjit/interface/stream/stream_table.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace memory_operand {

class memory_operand_t;

/// A stream pairing an iterator stream with `operation::memory_t` emission
struct memory_operand_impl_t : public stream_impl_t {
    static constexpr const char *NAME = "rvjit::stream::memory_operand";

    /// Creates an unconfigured memory operand
    ///
    /// @param mem The memory operations emitted
    /// @param it The pointer iterator, which the operand owns
    memory_operand_impl_t(
            const operation::memory_t &mem, iterator::iterator_impl_t it)
        : mem_(&mem), it_(std::move(it)) {}

    /// A memory operand that is never valid
    memory_operand_impl_t() = default;

    /// The stream handed out in place of one that failed to be created
    static memory_operand_impl_t *invalid() {
        static memory_operand_impl_t none;
        return &none;
    }

    /// Configures the stream, and its iterator, from a resolved pointer walk
    /// and a ring
    ///
    /// @param op The memory operation shared by every position
    /// @param walk The resolved pointer walk
    /// @param ring The destination register ring
    bool init(const memory::op_t &op, const iterator::plan_t &walk,
            const buffer_map_t &ring) {
        CHECK(ring.is_valid(), "invalid register ring");
        CHECK(it_.init(walk), "unresolved iterator plan");
        op_ = op;
        regs_ = ring;
        return true;
    }

    /// The destination register ring, for a caller reshaping its map
    buffer_map_t &regs() { return regs_; }

    /// The memory operation shared by every position
    const memory::op_t &op() const { return op_; }

    /// Re-resolves the stream with another variant of the spec it was
    /// created from
    ///
    /// @param variant The strategy, pointer budget and ring
    ///
    /// @note The registers of the previous variant stay allocated until the
    ///     enclosing transaction ends
    ///
    /// @pre The stream was created by `memory_operand_t::create`
    bool replan(const variant_t &variant);

    int ndims() const override { return it_.ndims(); }

    int size(int d) const override { return it_.size(d); }

    bool is_valid() const override {
        return it_.is_valid() && regs_.is_valid();
    }

    bool resize(const index_t &shape) override {
        CHECK(is_valid(), "invalid stream");
        regs_.reshape(shape);
        return it_.resize(shape);
    }

    name_t get(const index_t &idx) const override {
        CHECK_OR_RET(is_valid(), name_t(), "invalid stream");
        return regs_({idx[0], idx[1]});
    }

    const buffer::buffer_t &buffer() const override {
        static const buffer::buffer_t none;
        CHECK_OR_RET(is_valid(), none, "invalid stream");
        return regs_.storage;
    }

protected:
    /// Emits the memory operation at `idx`, then advances the pointer
    void produce_(const index_t &idx) override {
        memory::op_t op = op_;
        op.data.name = regs_({idx[0], idx[1]});
        op.addr = it_.pointer(idx);
        op.dir == memory::dir_t::load ? mem_->load(op) : mem_->store(op);
        it_.iteration(idx);
    }

    /// Builds whatever pivot chains the addressing needs
    void init_(const index_t &idx) override { it_.init_iteration(idx); }

private:
    friend class memory_operand_t;

    const operation::memory_t *mem_ = nullptr;
    iterator::iterator_impl_t it_;
    memory::op_t op_;
    buffer_map_t regs_;
    spec_t spec_;
    memory_operand_t *owner_ = nullptr;
};

/// Creates the memory operand streams, each with its own pointer iterator
///
/// @note Transactions reclaim the streams created since `begin()`
class memory_operand_t : public stream_table_t<memory_operand_impl_t, 8> {
public:
    static constexpr const char *NAME = "rvjit::stream::memory_operand";

    /// Creates a memory operand factory
    ///
    /// @param e The emitter instructions are emitted through
    /// @param ptr The pointer iterator factory
    /// @param mem The memory operations emitted
    /// @param pool The pool the data registers are drawn from
    memory_operand_t(emitter_t e, iterator::iterator_t &ptr,
            const operation::memory_t &mem, buffer::buffer_pool_t &pool)
        : stream_table_t<memory_operand_impl_t, 8>(mem, ptr.detached())
        , e_(e)
        , ptr_(ptr)
        , pool_(pool) {}

    /// Creates a memory operand stream, and its iterator, in one call
    ///
    /// @param spec What the operand is
    /// @param variant The strategy, pointer budget and ring
    ///
    /// @note Returns `invalid()` on failure, `miss()` naming the resource
    ///     class the attempt ran short of
    memory_operand_impl_t *create(
            const spec_t &spec, const variant_t &variant) {
        memory_operand_impl_t *m = next_();
        if (!m) {
            LOG_FAIL(__func__, "memory operand table full");
            miss_ = shortfall_t::of(resource_t::integers);
            return invalid();
        }
        if (!build_(*m, spec, variant)) return invalid();
        m->owner_ = this;
        take_();
        return m;
    }

    /// Creates a memory operand stream from the current levels of a plan
    ///
    /// @param plan The plan
    memory_operand_impl_t *create(const plan_t &plan) {
        return create(plan.spec, plan.variant());
    }

    /// The strategies the pointer walks of this factory's streams resolve over
    const iterator::strategy_set_t &strategies() const {
        return ptr_.strategies();
    }

    /// The shortfall of the last failed `create` or `replan`
    shortfall_t miss() const { return miss_; }

private:
    friend struct memory_operand_impl_t;

    /// Resolves and allocates `spec` under `variant` into `m`
    bool build_(memory_operand_impl_t &m, const spec_t &spec,
            const variant_t &variant) {
        miss_ = shortfall_t::of(resource_t::integers);
        CHECK(spec.is_valid(), "invalid spec");

        // Allocate the ring first: the pointers draw from what is left
        buffer_map_t ring = variant.ring;
        const bool drawn = !ring.is_valid();
        if (drawn) {
            memory::op_t as = spec.op;
            if (variant.ring_type != type_t::none)
                as.data.set_type(variant.ring_type);
            ring.storage = pool_.new_buffer(ring.total(), as);
        }
        if (!ring.is_valid()) {
            const int have = spec.op.is_vector() ? pool_.vectors().available()
                    : spec.op.is_int()           ? pool_.integers().available()
                                                 : pool_.floats().available();
            miss_ = shortfall_t::of(
                    spec.buffer_resource(), ring.total() - have);
            LOG_FAIL(__func__, "insufficient registers for the ring");
            return false;
        }

        // The operation accesses the ring's registers shifted by the offset
        if (variant.offset != 0) {
            buffer::buffer_t shifted;
            for (int i = 0; i < ring.storage.size(); ++i)
                shifted.set(ring.storage.id(i) + variant.offset);
            ring.storage = shifted;
        }

        // Resolve the pointer walk within the registers left
        const int left = pool_.integers().available();
        const int ntmp = variant.ntmp < 0 ? left : std::min(variant.ntmp, left);
        iterator::request_t request = spec.request();
        request.strategy = variant.strategy;
        iterator::plan_t walk = iterator::iterator_t::resolve(
                ptr_.strategies(), request, ntmp);
        if (walk.axis(0).strategy.is_any()) {
            // An explicit strategy misses its own need, which no pool may
            // host past the cap; `any` may fall back to a cheaper one
            const int need = iterator::iterator_t::need(
                    ptr_.strategies(), request, variant.ntmp);
            int missing = 0;
            if (!request.strategy.is_any())
                missing = need >= demand_t::INF ? demand_t::INF : need - left;
            miss_ = shortfall_t::of(resource_t::integers, missing);
            LOG_FAIL(__func__, "unresolved pointer walk");
            return false;
        }
        // Allocate the registers of the resolved pointer walk
        miss_ = shortfall_t::of(resource_t::integers);
        ptr_.new_iterator(walk);
        CHECK(!walk.needs_regs(), "insufficient registers for the pointers");

        CHECK(m.init(spec.op, walk, ring),
                "unable to create the memory operand");
        m.spec_ = spec;
        miss_ = {};
        return true;
    }

    emitter_t e_;
    iterator::iterator_t &ptr_;
    buffer::buffer_pool_t &pool_;
    shortfall_t miss_;
};

inline bool memory_operand_impl_t::replan(const variant_t &variant) {
    CHECK(owner_, "a stream not created by the factory");
    return owner_->build_(*this, spec_, variant);
}

} // namespace memory_operand
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
