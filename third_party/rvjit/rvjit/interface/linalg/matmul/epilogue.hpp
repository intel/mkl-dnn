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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_EPILOGUE_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_EPILOGUE_HPP

#include <algorithm>

#include "rvjit/interface/isa/conversion.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/linalg/matmul/impl.hpp"
#include "rvjit/interface/linalg/matmul/plan.hpp"
#include "rvjit/interface/linalg/matmul/postop.hpp"
#include "rvjit/interface/linalg/matmul/postop/builtin.hpp"
#include "rvjit/interface/linalg/matmul/postop/chain.hpp"
#include "rvjit/interface/stream/memory/conversion_stream.hpp"
#include "rvjit/interface/stream/memory/output_stream.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {

/// What follows the K loop of every tile
///
/// @note The conversion to the epilogue's type, the post-op chain and stores
///
/// @details The chain is alpha, beta, bias, then the plan's post-ops, each
///     present when its codegen flag is set and, if the runtime flag word is a
///     literal, set in it. A runtime flag word gates every present entry with
///     an `andi` and a branch, and a single branch over all of them first
///
/// @note Resources are planned in the registers the matmul frees at the
///     epilogue: the caller calls `plan()` in a scope sibling to the K phase
struct epilogue_t {
    static constexpr const char *NAME = "rvjit::linalg::matmul::epilogue";

    /// Creates the epilogue of a matmul
    ///
    /// @param pm The programming model
    /// @param ms The memory operand factory
    /// @param p The plan, which must outlive the epilogue
    epilogue_t(programming_model_t &pm, memory_operand_t &ms, const plan_t &p)
        : pm_(pm)
        , ms_(ms)
        , plan_(p)
        , env_(pm, ms, p)
        , store_(memory_stream_t::invalid()) {}

    /// Selects the chain entries the plan makes active
    ///
    /// @note Returns false on an inconsistent plan
    bool select() { return chain_.select(plan_); }

    /// Whether any active entry is a feature, rather than a bare store
    bool active() const { return chain_.size() > 0; }

    /// Whether the chain reads the bias pointer
    bool uses_bias() const { return chain_.mask() & with_bias; }

    /// The registers the active entries need, for tile `mb` x `nb`
    ///
    /// @param mb The M granules of the tile
    /// @param nb The columns of the tile
    needs_t needs(int mb, int nb) {
        env_.mb = mb;
        env_.nb = nb;
        return chain_.needs(env_);
    }

    /// The environment the matmul fills before `plan()`
    postop_env_t &env() { return env_; }

    /// Allocates the epilogue's registers and builds its streams
    ///
    /// @param c_regs The accumulator registers
    ///
    /// @pre `env()` is filled, and `c_regs` holds the accumulators, shaped as
    ///     the matmul's C stream
    shortfall_t plan(const buffer::buffer_t &c_regs) {
        auto &pool = pm_.isa().buffers();
        const int mb = env_.mb, nb = env_.nb;

        // C's M granules are one granule apart; an oversized stride needs a
        // register, even when a single granule leaves it unused
        env_.c_m_stride = const_t(env_.m_grain * env_.sewb_c);
        if (!pool.allocate(env_.c_m_stride))
            return shortfall_t::of(resource_t::integers);

        // The registers the entries share, the first of the accumulators' class
        const needs_t n = needs(mb, nb);
        const operand_t acc_operand(env_.c_form, plan_.dtc.reg);
        auto &acc_allocator = pool.allocator(acc_operand);
        const auto kind = n.mask ? buffer::allocation_t::output
                                 : buffer::allocation_t::scratch;
        const int granted = std::max(
                n.accs, std::min(n.accs_wanted, acc_allocator.available(kind)));
        const auto accs = acc_allocator.take(granted, kind);
        const auto floats = pool.floats().take(n.floats);
        const auto ints = pool.integers().take(n.ints);
        if (!accs) return accs.miss();
        if (!floats) return floats.miss();
        if (!ints) return ints.miss();
        env_.scratch = {*accs, *floats, *ints};
        if (n.args) {
            env_.args = pool.new_variable(plan_.postop_args.source());
            if (!env_.args.is_valid())
                return shortfall_t::of(resource_t::integers);
        }
        if (chain_.rt_runtime()) {
            rt_ = pool.new_variable(plan_.rt_flags.source());
            const auto gate = pool.integers().take(1);
            if (!rt_.is_valid()) return shortfall_t::of(resource_t::integers);
            if (!gate) return gate.miss();
            gate_ = reg_t(gate->id(0));
        }

        for (int i = 0; i < chain_.size(); ++i) {
            const shortfall_t s = chain_[i]->plan(env_);
            if (!s.ok()) return s;
        }

        // The store's registers are the accumulators indexed as {n, m}
        const auto store_regs
                = buffer_map_t(c_regs, {nb, mb}, order_t::transposed);
        store_ = env_.create_c(env_.c_spec(memory::dir_t::store,
                                       stream::iterator::update_t::advance(0)),
                env_.c_variant(store_regs));
        if (!store_->is_valid()) return ms_.miss();

        // The accumulators reach the epilogue's type in place first, e.g. an
        // s32 accumulation scaled in f32, through the store's view of them
        const type_t acc = plan_.dtc.reg, epi = plan_.epi_type();
        const auto &arith = pm_.isa().arithmetic();
        conversion_t retype;
        CHECK_OR_RET(sewb(acc) == sewb(epi),
                shortfall_t::of(resource_t::integers),
                "an epilogue type of another width");
        CHECK_OR_RET(retype.plan(acc, epi, env_.c_form, arith),
                shortfall_t::of(resource_t::integers),
                "unsupported conversion of the accumulators");
        retype_ = stream::memory_operand::conversion_stream_t();
        if (!retype.empty() && !retype_.init(arith, retype, store_, epi))
            return shortfall_t::of(resource_t::integers);

        // The accumulators are narrowed to C's type on their way out
        CHECK_OR_RET(
                out_.init(pm_.isa().arithmetic(), pm_.streams().scheduler(),
                        store_, type_parameter_t(plan_.dtc.mem, epi),
                        stream::policy_t::arithmetic(), env_.configure),
                shortfall_t::of(resource_t::integers),
                "unable to narrow the accumulators");
        return {};
    }

    /// Emits the chain over the accumulators of one dispatched tile, then
    /// stores them
    ///
    /// @param acc The accumulators
    /// @param mb The M granules of the tile
    /// @param nb The columns of the tile
    bool emit(buffer_stream_t &acc, int mb, int nb) {
        postop_tile_t tile {env_, acc, mb, nb};
        const int n = chain_.size();
        if (n > 0 && env_.args.is_valid()) pm_.isa().transfer().init(env_.args);

        // The conversion to the epilogue's type is never gated
        if (retype_.converts()) {
            retype_.resize({nb, mb});
            retype_.emit();
        }

        if (n > 0) {
            // Run the chain, consecutive entries sharing a flag sharing a gate
            bool ok = true;
            const auto run = [&] {
                for (int i = 0; i < n;) {
                    int j = i + 1;
                    while (j < n && chain_[j]->flag() == chain_[i]->flag())
                        ++j;
                    gated_(chain_[i]->flag(), [&] {
                        for (int k = i; k < j; ++k)
                            ok = chain_[k]->emit(tile) && ok;
                    });
                    i = j;
                }
            };

            // A runtime flag word gates the whole chain with a single branch
            const int mask = chain_.mask();
            if (chain_.rt_runtime()) pm_.isa().transfer().init(rt_);
            const bool many = mask & (mask - 1);
            if (chain_.rt_runtime() && many)
                gated_(mask, run);
            else
                run();
            CHECK(ok, "unable to emit the post-ops");
        }
        out_.resize({nb, mb});
        out_.emit();
        return true;
    }

private:
    /// Runs `body` when the runtime word has any bit of `bit` set
    template <typename F>
    void gated_(int bit, F &&body) {
        if (!chain_.rt_runtime()) {
            body();
            return;
        }
        pm_.isa().const_arithmetic().and_(gate_, rt_.buffer(), const_t(bit));
        const auto is_set = control_flow::branch_t::nez(
                variable_t(gate_), control_flow::distance_t::short_);
        pm_.isa().control_flow().if_(is_set, [&] { body(); });
    }

    programming_model_t &pm_;
    memory_operand_t &ms_;
    const plan_t &plan_;
    postop_env_t env_;

    postop_chain_t chain_;

    variable_t rt_; // The runtime flag word
    reg_t gate_ {}; // `rt_ & bit`

    memory_stream_t *store_;
    stream::memory_operand::conversion_stream_t retype_; // To the epilogue's
    stream::memory_operand::output_stream_t out_; // Narrows, then stores
};

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
