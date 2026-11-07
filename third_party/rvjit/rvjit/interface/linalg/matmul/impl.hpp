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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_IMPL_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_IMPL_HPP

#include <algorithm>
#include <functional>

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/linalg/matmul/needs.hpp"
#include "rvjit/interface/linalg/matmul/plan.hpp"
#include "rvjit/interface/loop/loopgen.hpp"
#include "rvjit/interface/loop/unroll.hpp"
#include "rvjit/interface/programming_model.hpp"
#include "rvjit/interface/stream/memory/spec.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {

/// ISA-specific matmul implementation
///
/// @details Decides the optimization values a `matmul_t` builds its loops,
///     operands and streams from, and supplies the ISA actions around them.
///     The block sizes it decides are final: `matmul_t` never reduces them
///
/// @note A `literal` or `runtime` size covers the whole dimension in one
///     call, the implementation chunking it into its own unrolled blocks
///
/// @note A `less_eq` size is a single chunk of at most its bound, the client
///     owning the loop around the call; the chunk need not be a multiple of a
///     loop's granularity, the tail being covered by the loop itself
///
/// @note A bounded size may steer the implementation's unroll decisions by
///     bounding them. The blocks a client pins in `plan_t::blocks` are
///     honored as they are or the plan is rejected
struct impl_t {
    static constexpr const char *NAME = "rvjit::linalg::matmul";

    using memory_spec_t = stream::memory_operand::spec_t;
    using memory_variant_t = stream::memory_operand::variant_t;

    /// Creates an implementation of the decisions of `matmul_t`
    ///
    /// @param model The programming model the engine builds its kernel with
    explicit impl_t(programming_model_t &model) : model_(model) {}

    virtual ~impl_t() = default;

    /// The programming model this implementation ships with
    programming_model_t &model() const { return model_; }

    // Loop control
    int mb = 1;
    int nb = 1; // The main N tile width
    int kb = 1;
    int ktb = 1; // The unroll of the K loop under an N-tail tile
    int m_grain = 1;
    int n_grain = 1;
    int k_grain = 1;

    /// What the epilogue needs for `mb` M granules, which `matmul_t` sets
    /// before calling `operator()`
    ///
    /// @note The epilogue runs in the registers the K loop frees, so the
    ///     implementation checks its needs against the registers the
    ///     accumulators leave instead of reserving them
    std::function<needs_t(int)> epilogue_needs_fn;

    /// What the epilogue needs for `mb` M granules
    ///
    /// @param mb The M granules of the tile
    needs_t epilogue_needs(int mb) const {
        return epilogue_needs_fn ? epilogue_needs_fn(mb) : needs_t();
    }

    // Buffer form
    form_t a_form = form_t::undefined;
    form_t b_form = form_t::undefined;
    form_t c_form = form_t::undefined;

    /// Work-units covered by one dispatch of the largest M unrolling
    int m_tile() const { return m_grain * mb; }

    /// Work-units covered by one dispatch of the largest N unrolling
    int n_tile() const { return n_grain * nb; }

    /// Work-units covered by one dispatch of the largest K unrolling
    int k_tile() const { return k_grain * kb; }

    /// Prepares the pool's register classes to run `p`, e.g. their grouping
    ///
    /// @param p The plan
    ///
    /// @note `matmul_t` calls it before `operator()` and before allocating
    ///     anything; the registers are the programming model's `isa().buffers()`
    virtual bool configure(const plan_t &p) {
        (void)p;
        return true;
    }

    /// Resolves the decisions for `p` into this object's fields
    ///
    /// @param p The plan
    ///
    /// @note Returns false unless the implementation accepted the plan
    /// @note Allocates nothing
    virtual bool operator()(const plan_t &p) {
        (void)p;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// Draws the registers the implementation keeps across the whole kernel
    ///
    /// @note Called once after `operator()`, before the operands are tuned
    virtual shortfall_t allocate() { return {}; }

    /// Whether the resolved decisions are complete and in range
    bool is_valid() const {
        if (mb < 1 || nb < 1) return false;
        if (kb < 1 || ktb < 1) return false;
        if (m_grain < 1 || n_grain < 1 || k_grain < 1) return false;
        if (a_form == form_t::undefined) return false;
        if (b_form == form_t::undefined) return false;
        return c_form != form_t::undefined;
    }

    /// Tunes the memory operand of A before the resource search
    ///
    /// @param spec The operand, with the operation shape, origin, space and
    ///     update `matmul_t` filled; the implementation sets its addressing and
    ///     may edit `spec.op`, e.g. to make the load strided
    /// @param variant The preferred variant; the implementation sets its
    ///     strategy and pointer budget `ntmp` (capped by the registers left,
    ///     negative for all of them)
    virtual void set_a_operand(memory_spec_t &spec, memory_variant_t &variant) {
        (void)spec;
        (void)variant;
    }

    /// Tunes the memory operand of B, as `set_a_operand()` does for A
    ///
    /// @param spec The operand
    /// @param variant The preferred variant
    virtual void set_b_operand(memory_spec_t &spec, memory_variant_t &variant) {
        (void)spec;
        (void)variant;
    }

    /// Tunes every C-shaped memory operand the epilogue builds, as
    /// `set_a_operand()` does for A
    ///
    /// @param spec The operand
    /// @param variant The preferred variant
    ///
    /// @note Called for the store and for each post-op reading C or an array
    ///     along M, with the shape and update they built
    virtual void set_c_operand(memory_spec_t &spec, memory_variant_t &variant) {
        (void)spec;
        (void)variant;
    }

    /// Hook called once before the loops, after `matmul_t` initialized its own
    /// variables and constants
    ///
    /// @note The place to initialize the registers the implementation holds
    virtual void prepare() {}

    /// Hook called at the start of each M-tile dispatch
    ///
    /// @param mb The M granules of the dispatched tile
    /// @param remaining The register holding the work-units left
    ///
    /// @note `remaining` is the M loop's `remaining()`: the work-units still
    ///     to be dispatched, including this tile's; zero if the loop keeps
    ///     none, i.e. a single-trip loop of granularity 1
    virtual void on_m_tile_enter(int mb, const reg_t &remaining) {
        (void)mb;
        (void)remaining;
    }

    /// Hook called at the end of each M-tile dispatch
    ///
    /// @param mb The M granules of the dispatched tile
    virtual void on_m_tile_exit(int mb) { (void)mb; }

    /// Hook called at the start of each N-tile dispatch
    ///
    /// @param nb The columns of the dispatched tile
    /// @param remaining The register holding the work-units left
    ///
    /// @note `remaining` is the N loop's `remaining()`, as for M
    virtual void on_n_tile_enter(int nb, const reg_t &remaining) {
        (void)nb;
        (void)remaining;
    }

    /// Hook called at the end of each N-tile dispatch
    ///
    /// @param nb The columns of the dispatched tile
    virtual void on_n_tile_exit(int nb) { (void)nb; }

    /// Hook called before each K-loop
    ///
    /// @param mb The M granules of the dispatched tile
    /// @param nb The columns of the dispatched tile
    /// @param remaining The register holding the work-units left
    ///
    /// @note `remaining` is the K loop's `remaining()`, valid only within its
    ///     body
    virtual void on_k_loop_enter(int mb, int nb, const reg_t &remaining) {
        (void)mb;
        (void)nb;
        (void)remaining;
    }

    /// Hook called after each K-loop
    ///
    /// @param mb The M granules of the dispatched tile
    /// @param nb The columns of the dispatched tile
    /// @param remaining The register holding the work-units left
    virtual void on_k_loop_exit(int mb, int nb, const reg_t &remaining) {
        (void)mb;
        (void)nb;
        (void)remaining;
    }

    /// Switches the vector configuration to `t`'s element width, keeping the
    /// vector length
    ///
    /// @param t The type whose width the conversions or memory operations that
    ///     follow run at
    ///
    /// @note Called by the conversion streams around the steps of a conversion
    ///     at another width, and with the accumulators' type to come back;
    ///     returns false when no configuration fits the width
    virtual bool on_configure(type_t t) {
        (void)t;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

private:
    programming_model_t &model_;
};

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
