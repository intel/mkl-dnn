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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_HPP

#include <algorithm>

#include "rvjit/interface/core/callback.hpp"
#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/induction/step.hpp"
#include "rvjit/interface/isa/buffer/stack.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/conversion.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/operation/memory.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/interface/linalg/matmul/epilogue.hpp"
#include "rvjit/interface/linalg/matmul/impl.hpp"
#include "rvjit/interface/linalg/matmul/plan.hpp"
#include "rvjit/interface/linalg/matmul/strides.hpp"
#include "rvjit/interface/loop/frame.hpp"
#include "rvjit/interface/loop/loopgen.hpp"
#include "rvjit/interface/loop/persistence.hpp"
#include "rvjit/interface/loop/reduce_stream.hpp"
#include "rvjit/interface/programming_model.hpp"
#include "rvjit/interface/resource/optimization/group.hpp"
#include "rvjit/interface/resource/optimization/search.hpp"
#include "rvjit/interface/stream/memory/input_stream.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"
#include "rvjit/interface/stream/operand/operand.hpp"
#include "rvjit/interface/stream/scheduler.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {

using buffer_stream_t = stream::operand::operand_impl_t;
using memory_stream_t = stream::memory_operand::memory_operand_impl_t;
using memory_operand_t = stream::memory_operand::memory_operand_t;
using memory_spec_t = stream::memory_operand::spec_t;
using memory_variant_t = stream::memory_operand::variant_t;
using memory_plan_t = stream::memory_operand::plan_t;
using memory_axis_role_t = stream::memory_operand::axis_role_t;
using register_operand_t = stream::operand::operand_t;
using iterator_plan_t = stream::iterator::plan_t;
using iterator_update_t = stream::iterator::update_t;
using iterator_space_t = stream::iterator::space_t;
using strategy_id_t = stream::iterator::strategy_id_t;

/// General matrix multiplication interface
///
/// @note Builds a matmul kernel based on ISA-agnostic descriptions
///
/// @pre `init()` has returned true before any other method is called
struct matmul_t {
    static constexpr const char *NAME = "rvjit::linalg::matmul";

    /// Creates the engine around an implementation
    ///
    /// @param impl The implementation, which must outlive the engine
    ///
    /// @note Everything is sourced from the programming model `impl` ships
    explicit matmul_t(impl_t &impl)
        : impl_(impl)
        , pm_(impl.model())
        , configure_([&impl](type_t t) { return impl.on_configure(t); })
        , epilogue_(pm_, pm_.streams().memory_operand(), plan_) {}

    /// Resolves `p` and returns whether or not the component succeeded
    ///
    /// @param p The plan
    bool init(const plan_t &p) {
        // The epilogue's register needs bound the accumulators the backend
        // picks
        plan_ = p;
        CHECK(epilogue_.select(), "invalid epilogue configuration");
        impl_.epilogue_needs_fn
                = [this](int mb) { return epilogue_.needs(mb, 1); };

        // Prepare the register classes the way the backend needs them
        CHECK(impl_.configure(p), "unable to configure the registers");

        // Dispatch the plan to the matmul backend implementation
        CHECK(impl_(p) && impl_.is_valid(), "invalid implementation");

        // Collect the implementation-defined granularity and unroll factors
        const int mb = impl_.mb;
        const int nb = impl_.nb;
        const int kb = impl_.kb;
        const int ktb = impl_.ktb;
        const int mg = impl_.m_grain;
        const int ng = impl_.n_grain;
        const int kg = impl_.k_grain;

        // Configure the multiply-accumulate operation from the operand forms
        macc_ = arithmetic::op_t(p.dtc.reg, p.dtb.reg, p.dta.reg);
        macc_.rd.form = impl_.c_form;
        macc_.rs1.form = impl_.b_form;
        macc_.rs2.form = impl_.a_form;
        CHECK(pm_.isa().arithmetic().supports_macc(macc_),
                "invalid macc operation");
        CHECK(!p.dta.converts() || p.dta.widens(), "a narrowed A");

        // B is extended to a wider integer by its load: the narrow load
        // sign-extends a signed type and zero-extends an unsigned one
        if (p.dtb.converts()) {
            const bool ints
                    = rvjit::is_int(p.dtb.mem) && rvjit::is_int(p.dtb.reg);
            CHECK(ints, "an extended B that is not an integer");
            CHECK(p.dtb.widens(),
                    "an extended B that is not wider than its memory type");
            CHECK(!rvjit::is_signed_int(p.dtb.mem)
                            || rvjit::is_signed_int(p.dtb.reg),
                    "a signed B extended to an unsigned type");
        }

        // The accumulators reach C's type as they are stored, through the
        // epilogue's type
        conversion_t out;
        CHECK(out.plan(p.epi_type(), p.dtc.mem, impl_.c_form,
                      pm_.isa().arithmetic()),
                "unsupported conversion of the accumulators");

        // Create the dense loop nest, whose resources are drawn later, in the
        // scope each loop is dispatched from
        auto &loops = pm_.loop();
        m_loop_ = loops.data_parallel(p.m, mb, mg);
        n_loop_ = loops.data_parallel(p.n, nb, ng);
        kt_loop_ = loops.reduce(p.k, ktb, kg);
        k_loop_ = loops.alias(*kt_loop_, p.k, kb, kg);
        CHECK(m_loop_->is_resolved() && n_loop_->is_resolved()
                        && kt_loop_->is_resolved() && k_loop_->is_resolved(),
                "unresolved loops");

        // Determine the M loop `distance` by predicting its body size
        const const_t lda_value(p.lda.source());
        int m_instrs = 0;
        const auto instrs = [&](int m, int n, int k) {
            // Predict the number of instructions per macc.
            // The current prediction uses 2 or 3 instructions per macc.
            // Two is used when pointer arithmetics are predicted to be folded.
            // Folding is predicted when the lda stride is a small immediate.
            const bool maybe_fold
                    = lda_value.has_imm() && is_simm12(lda_value.imm() * nb);
            const int per_macc = maybe_fold ? 2 : 3;

            // Predict instruction count for the post-operations.
            // The current model predicts 4 instruction per operation per acc
            const int postops = (p.flags & with_alpha ? 1 : 0)
                    + (p.flags & with_beta ? 1 : 0)
                    + (p.flags & with_bias ? 1 : 0)
                    + (p.flags & with_postops ? 1 : 0);
            const int postop_factor = m * n * 4 * postops;

            // A fixed number of operations unrelated to the macc operations
            static constexpr int bias = 64;
            return m * n * k * per_macc + bias + k + postop_factor;
        };
        // Sum the predicted intruction count of all unrolled N loops
        for (int n = 1; n <= nb; ++n) {
            // The widest N tile runs the main K loop, the others the narrow
            const auto &ku = n == nb ? k_loop_->unroll() : kt_loop_->unroll();
            for (int j = 0; j < ku.size(); ++j)
                m_instrs += instrs(impl_.mb, n, ku[j]);
        }
        CHECK(m_loop_->set_distance(loops.reach(m_instrs)), "failed M loop");

        // Allocate the values living across the whole kernel
        ptra_ = pm_.isa().buffers().new_variable(p.ptra.source());
        ptrb_ = pm_.isa().buffers().new_variable(p.ptrb.source());
        ptrc_ = pm_.isa().buffers().new_variable(p.ptrc.source());
        lda_ = pm_.isa().buffers().new_constant(p.lda.source());
        ldb_ = pm_.isa().buffers().new_constant(p.ldb.source());

        // The bytes between granules along each axis of the streams
        const strides_t strides = strides_t::of(p, mg, ng, kg);
        n_derived_ = 0;

        // The M tiles of a transposed A are `m_grain * lda` bytes apart. A
        // single tile is never stepped
        a_m_stride_ = const_t(0);
        if (p.is_transa && mb > 1) {
            a_m_stride_ = stride_(strides.a_m);
            CHECK(a_m_stride_.is_valid(),
                    "unable to allocate the A tile stride");
        }
        ldc_ = lda_;
        if (!p.lda.is_same_value(p.ldc))
            ldc_ = pm_.isa().buffers().new_constant(p.ldc.source());
        CHECK(all_valid(ptra_, ptrb_, ptrc_, lda_, ldb_, ldc_),
                "failed allocation");

        // A post-op reading the M position, e.g. the bias, finds its byte
        // offset in a variable the M loop advances; an M dispatched once
        // has none, its offset being zero
        m_off_ = variable_t();
        if (epilogue_.needs(mb, nb).m_offset && m_loop_->has_multiple_trips()) {
            m_off_ = pm_.isa().buffers().new_variable(source_t::immediate(0));
            CHECK(m_off_.is_valid(), "unable to allocate the M offset");
        }

        CHECK(impl_.allocate().ok(), "failed implementation allocation");

        // Create the C operand buffer stream with the implementation shape
        const auto acc
                = pm_.isa().buffers().allocator(macc_.rd).take_acc(mb * nb);
        CHECK(acc, "unable to allocate the accumulators");
        const auto map = buffer_map_t(*acc, {mb, nb});
        c_stream_ = pm_.streams().operand().accumulator(map, macc_.rd);
        CHECK(c_stream_->is_valid(), "failed C stream resolution");

        static constexpr int k_depth = 2; // Prefered K axis size on A and B

        // Compute strides for the memory operand streams
        if (!p.is_transa) a_m_stride_ = stride_(strides.a_m);
        const const_t a_k_stride = stride_(strides.a_k);
        const const_t b_k_stride = stride_(strides.b_k);
        b_outer_ = stride_(strides.b_n);
        c_n_stride_ = stride_(strides.c_n);
        CHECK(all_valid(a_m_stride_, a_k_stride, b_k_stride, b_outer_,
                      c_n_stride_),
                "failed stride resolution");

        // Declare the memory streams specifications and desired variants
        //
        // A search algorithm attempts to use the desired variants, attempting
        // alternatives that are less register-intensive if the streams do not
        // fit the architectural state.
        //
        // The A memory stream attempts to use a buffer with a depth of k_depth,
        // which costs more data buffers but enables pipelining by increasing
        // the distance of producer instructions (A loads) and consumers (macc)
        memory_spec_t a_spec;
        memory_variant_t a_variant;
        a_spec.op = memory::op_t::load(impl_.a_form, p.dta.mem);
        a_spec.origin = pointer_t(ptra_.buffer());
        a_spec.space = iterator_space_t({kb, mb}, {a_k_stride, a_m_stride_});
        a_spec.update = iterator_update_t::advance(0);
        a_spec.role[K_AXIS] = memory_axis_role_t::reduce;
        a_spec.role[M_AXIS] = memory_axis_role_t::parallel;
        a_variant.ring = buffer_map_t({kb, mb}).wrapped(k_depth);

        // A narrower A is loaded into the upper part of the groups the
        // multiply-accumulate reads, and converted there
        a_offset_ = 0;
        if (p.dta.converts()) {
            const operand_t a_reg(impl_.a_form, p.dta.reg);
            const int group = pm_.isa().buffers().allocator(a_reg).group_size(
                    buffer::allocation_t::input);
            a_offset_ = p.dta.offset(group);
            a_variant.ring_type = p.dta.reg;
            a_variant.offset = a_offset_;
        }
        impl_.set_a_operand(a_spec, a_variant);

        // Unlike the A stream, the B stream is not limited to a single `tmp`
        // register to materialize the iterator stream. This is motivated by
        // the use of implicit broadcast operations on ISAs which encode B in
        // `scalar` form. On most ISAs, the `scalar` load instructions
        // feature imm offset fields, allowing `pivot` B iterator streams.
        // These are more resource-intensive but may drastically reduce the
        // dynamic instruction count by cutting pointer arithmetics.
        // Therefore, more resources are staked for the B stream variant.
        memory_spec_t b_spec;
        memory_variant_t b_variant;
        b_spec.op = memory::op_t::load(impl_.b_form, p.dtb.mem);
        b_spec.origin = pointer_t(ptrb_.buffer());
        b_spec.space = iterator_space_t({kb, nb}, {b_k_stride, b_outer_});
        b_spec.update = iterator_update_t::advance(0);
        b_spec.role[K_AXIS] = memory_axis_role_t::reduce;
        b_spec.role[N_AXIS] = memory_axis_role_t::parallel;
        if (impl_.mb > 1) b_spec.role[N_AXIS] = memory_axis_role_t::fixed;
        b_variant.ring = buffer_map_t({k_depth, impl_.nb});
        impl_.set_b_operand(b_spec, b_variant);

        // Optimization search: allocate resources for all components
        //
        // Attempts to distribute registers as buffers for variables
        // and implement the proposed A and B memory operands. If resources
        // are insufficient, reclaim resources from `resource_optimizable_t`.
        //
        // Prioritize the implementation of the A and B memory operands over
        // variables on the outermost loop levels.
        //
        // Strategy 1) Move M pointer bases and bookkeeping to the stack
        auto m_persist = loop::persistence_t::buffered_then_stack();
        m_persist.bind(*m_loop_, ptra_, ptrb_, ptrc_, m_off_);

        // Strategy 2) Save the epilogue's variables around the K phase, which
        // does not read them, so that its streams may use their registers
        k_phase_ = loop::frame_t();
        auto k_borrow = loop::persistence_t::on_demand();
        k_borrow.bind(k_phase_, ptrc_, m_off_);

        // Strategy 3) Reclaim loop resources used for control flow
        optimization::group_t nest(optimization::order_t::level_first);
        nest.declare(*m_loop_, *n_loop_);

        // Strategy 4) Move N pointer bases to the stack
        auto n_persist = loop::persistence_t::buffered_then_stack();
        n_persist.bind(*n_loop_, ptra_, ptrb_, m_off_);

        // Strategy 5) Use a stream variant that costs fewer resources
        auto &ms = pm_.streams().memory_operand();
        memory_plan_t a_plan(ms.strategies(), a_spec, a_variant);
        memory_plan_t b_plan(ms.strategies(), b_spec, b_variant);

        optimization::group_t relax;
        relax.declare(m_persist, k_borrow, nest, n_persist);

        optimization::search_t search;
        search.guard(pm_.guards()); // objects to initiate transactions

        // If the intendend build fails, try:
        // - strategies 1 to 4
        // - strategies 5 reducing the A stream resources
        // - strategies 5 reducing the B stream resources
        search.axis(b_plan.levers());
        search.axis(a_plan.levers());
        search.axis(relax);

        // Each scope below matches the lifetime of its values in `operator()`
        const bool searched = search.run([&] {
            // The M loop's registers and slots live across the whole nest
            const shortfall_t m = m_loop_->allocate();
            if (!m.ok()) return m;

            // An M tile may use the registers the M loop lends its body
            const shortfall_t m_tile = pm_.scope(m_loop_->released(), [&] {
                const shortfall_t n = n_loop_->allocate();
                if (!n.ok()) return n;

                // An N tile may use the registers the N loop lends its body
                const shortfall_t n_tile = pm_.scope(n_loop_->released(), [&] {
                    const shortfall_t f = k_phase_.allocate(pm_.isa().buffers(),
                            pm_.isa().stack(), /*multi_trip=*/true);
                    if (!f.ok()) return f;

                    // The K phase holds the K loops and the A and B streams,
                    // and may use the registers `k_phase` saves around it
                    const shortfall_t k
                            = pm_.scope(k_phase_.restored(ptrc_, m_off_), [&] {
                        // Report everything the K phase misses at once
                        shortfall_t s = pm_.isa().buffers().lacks(
                                kt_loop_->demand() + k_loop_->demand()
                                + b_plan.demand() + a_plan.demand());
                        if (!s.ok()) return s;

                        s = kt_loop_->allocate();
                        if (!s.ok()) return s;
                        s = k_loop_->allocate();
                        if (!s.ok()) return s;

                        b_stream_ = ms.create(b_plan);
                        if (!b_stream_->is_valid()) return ms.miss();

                        a_stream_ = ms.create(a_plan);
                        if (!a_stream_->is_valid()) return ms.miss();
                        if (plan_.dta.converts()
                                && !a_in_.init(pm_.isa().arithmetic(),
                                        a_stream_, plan_.dta, a_offset_,
                                        configure_))
                            return shortfall_t::of(resource_t::vectors);
                        return shortfall_t {};
                    });
                    if (!k.ok()) return k;

                    // The epilogue follows the K phase, reusing its registers,
                    // and the last reads of the A and B pointers, which the N
                    // loop reloads after its body
                    return pm_.scope(
                            n_loop_->frame().restored(ptra_, ptrb_), [&] {
                        postop_env_t &env = epilogue_.env();
                        env.mb = impl_.mb;
                        env.nb = impl_.nb;
                        env.m_grain = impl_.m_grain;
                        env.sewb_c = rvjit::sewb(plan_.dtc.mem);
                        env.c_form = impl_.c_form;
                        env.configure = configure_;
                        env.tune_c = [this](memory_spec_t &spec,
                                             memory_variant_t &variant) {
                            impl_.set_c_operand(spec, variant);
                        };
                        env.ptrc = ptrc_;
                        env.c_n_stride = c_n_stride_;
                        env.m_off = m_off_;
                        return epilogue_.plan(c_stream_->buffer());
                    });
                });
                if (!n_tile.ok()) return n_tile;

                // The scratch of the N loop's frame steps, after each N tile
                return pm_.scope([&] {
                    const auto r = pm_.isa().buffers().integers().take(1);
                    if (!r) return r.miss();
                    n_tmp_ = reg_t(r->id(0));
                    return shortfall_t {};
                });
            });
            if (!m_tile.ok()) return m_tile;

            // The scratch of the M loop's frame steps, after each M tile
            return pm_.scope([&] {
                const auto r = pm_.isa().buffers().integers().take(1);
                if (!r) return r.miss();
                m_tmp_ = reg_t(r->id(0));
                return shortfall_t {};
            });
        });
        search_stats_ = search.stats();
        CHECK(searched, "failed to allocate the operand stream resources");

        return true;
    }

    /// What the resource search of the last `init()` spent
    const optimization::search_t::stats_t &search_stats() const {
        return search_stats_;
    }

    /// Emits the innermost M,N,K matmul loops
    ///
    /// @note This method does not issue `stack` preserve/restore calls
    ///
    /// @pre This object has returned true on `init()`
    void operator()() {
        pm_.isa().transfer().init({ptra_, ptrb_, ptrc_, m_off_});
        pm_.isa().transfer().init(
                {lda_, ldb_, lda_.is_same_imm(ldc_) ? const_t {} : ldc_});
        pm_.isa().transfer().init(a_m_stride_);
        for (int i = 0; i < n_derived_; ++i)
            pm_.isa().const_arithmetic().mul(derived_[i].var.buffer(),
                    derived_[i].ld.buffer(), const_t(derived_[i].factor));
        impl_.prepare();
        pm_.isa().stack().reserve_stack();

        // A's and C's per-element strides along M
        const const_t a_unit
                = plan_.is_transa ? lda_ : const_t(rvjit::sewb(plan_.dta.mem));
        const const_t c_unit(rvjit::sewb(plan_.dtc.mem));

        m_loop_->generate([&](int mb) {
            // Compute M-loop inductive updates
            const int mtile = impl_.m_grain * mb;
            const auto c_m_tile_step = step_t::macc(c_unit, mtile, m_tmp_);
            const auto a_m_tile_step = step_t::macc(a_unit, mtile, m_tmp_);

            // Configure M-loop inductive updates
            m_loop_->frame().set_step(ptra_, a_m_tile_step);
            m_loop_->frame().set_step(ptrc_, c_m_tile_step);
            if (m_off_.has_buffer())
                m_loop_->frame().set_step(m_off_, c_m_tile_step);

            impl_.on_m_tile_enter(mb, m_loop_->remaining());
            n_loop_->generate([&](int nb) {
                // Configure N-loop inductive updates
                const auto b_n_tile_step = step_t::macc(b_outer_, nb, n_tmp_);
                n_loop_->frame().set_step(ptrb_, b_n_tile_step);

                // Reshape macc operand blocks according to the current unroll
                stream::stream_impl_t &A = plan_.dta.converts()
                        ? static_cast<stream::stream_impl_t &>(a_in_)
                        : *a_stream_;
                memory_stream_t &B = *b_stream_;
                buffer_stream_t &C = *c_stream_;
                A.resize({impl_.kb, mb});
                B.resize({impl_.kb, nb});
                C.resize({mb, nb});

                loop::loop_t &k = *(nb == impl_.nb ? k_loop_ : kt_loop_);

                impl_.on_n_tile_enter(nb, n_loop_->remaining());

                // Delay consumers (macc) as far as the stream shapes tolerate
                // before data buffers are rewritten with newer iteration data
                const auto delay
                        = pm_.streams().reduce().max_delay(B.ref(), A.ref());

                // The goal is to hide the memory latency of the architecture
                // respecting the `delay` limit of the stream materialization
                const auto schedule = stream::policy_t::memory(delay);

                // The reduce stream initializes the C accumulators with zeroes
                // and initializes the A and B iterators. Then, it instantiates
                // the K reduce loop and its tail loops. An iteration space is
                // constructed using the outer product of the B and A streams,
                // defined with shape (kb, nb) x (kb, mb) = (kb, nb, mb).
                // At each iteration, before updating the streams, emit a macc
                // using the buffers mapped to that iteration on each operand.
                //
                // `k_phase` saves the variables whose registers the K phase
                // may use, if any, and reloads them for the epilogue
                k_phase_(pm_.isa().transfer(), [&] {
                    pm_.streams().reduce()(C.ref(), B.ref(), A.ref(), k,
                            schedule, [&](const index_t &a, const index_t &b) {
                        const int k = a[K_AXIS];
                        const int m = a[M_AXIS];
                        const int n = b[N_AXIS];
                        const auto rd = C.get({m, n});
                        const auto rs1 = B.get({k, n});
                        const auto rs2 = A.get({k, m});
                        pm_.isa().arithmetic().macc(macc_.names(rd, rs1, rs2));
                    }, [&] { impl_.on_k_loop_enter(mb, nb, k.remaining()); });

                    impl_.on_k_loop_exit(mb, nb, k.remaining());
                });
                epilogue_.emit(C, mb, nb);
                impl_.on_n_tile_exit(nb);
            });
            impl_.on_m_tile_exit(mb);
        });

        pm_.isa().stack().release_stack();
    }

private:
    /// The constant holding the stride `g` in bytes
    ///
    /// @note A product with a register leading dimension is computed by
    ///     `operator()` into a variable, which draws a register now
    const_t stride_(const granule_stride_t &g) {
        using base_t = granule_stride_t::base_t;
        auto &pool = pm_.isa().buffers();
        if (!g.is_ld()) return pool.new_constant(source_t::immediate(g.factor));

        const const_t ld = g.base == base_t::lda
                ? lda_
                : (g.base == base_t::ldb ? ldb_ : ldc_);
        if (g.factor == 1) return ld;
        if (ld.has_imm())
            return pool.new_constant(source_t::immediate(ld.imm() * g.factor));

        const int capacity
                = static_cast<int>(sizeof(derived_) / sizeof(*derived_));
        CHECK_OR_RET(n_derived_ < capacity, const_t::invalid(),
                "too many derived strides");
        derived_t &d = derived_[n_derived_];
        d.var = pool.new_variable(source_t::immediate(0));
        CHECK_OR_RET(d.var.is_valid(), const_t::invalid(),
                "unable to allocate a derived stride");
        d.ld = ld;
        d.factor = g.factor;
        ++n_derived_;
        return const_t(d.var.buffer());
    }

    static constexpr int K_AXIS = 0; // K axis id on A and B streams
    static constexpr int N_AXIS = 1; // N axis id on B stream
    static constexpr int M_AXIS = 1; // M axis id on A stream

    impl_t &impl_;
    programming_model_t &pm_;
    conversion::configure_t configure_; // The implementation's switch

    plan_t plan_; // The plan being generated
    epilogue_t epilogue_; // Post-ops and the C store, after each K loop

    // Variables
    variable_t ptra_, ptrb_, ptrc_;
    variable_t m_off_; // Byte offset of the M tile, when a post-op needs it

    // Constants
    const_t lda_, ldb_, ldc_;
    const_t a_m_stride_; // Bytes between the M granules of A
    const_t b_outer_; // Bytes between the N granules of B
    const_t c_n_stride_; // Bytes between the N granules of C

    // Strides computed at run time from a register leading dimension
    struct derived_t {
        variable_t var;
        const_t ld;
        int factor = 1;
    };
    derived_t derived_[5];
    int n_derived_ = 0;

    // Loops
    loop::loop_t *m_loop_ = loop::loop_t::invalid(); // M-tile loop
    loop::loop_t *n_loop_ = loop::loop_t::invalid(); // N-tile loop
    loop::loop_t *k_loop_ = loop::loop_t::invalid(); // K loop for main N loop
    loop::loop_t *kt_loop_ = loop::loop_t::invalid(); // K loop for tail N loop
    loop::frame_t k_phase_; // Saves variables around the K phase, if needed

    // Operations and operands
    arithmetic::op_t macc_;
    memory_stream_t *a_stream_ = memory_stream_t::invalid();
    stream::memory_operand::input_stream_t a_in_; // A converted, if narrower
    int a_offset_ = 0;
    memory_stream_t *b_stream_ = memory_stream_t::invalid();
    buffer_stream_t *c_stream_ = buffer_stream_t::invalid();

    reg_t m_tmp_ {}; // Scratch of the M loop's frame steps
    reg_t n_tmp_ {}; // Scratch of the N loop's frame steps

    optimization::search_t::stats_t search_stats_;
};

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
