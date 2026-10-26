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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_POSTOP_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_POSTOP_HPP

#include <algorithm>
#include <functional>

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/conversion.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/isa/variable.hpp"
#include "rvjit/interface/linalg/matmul/needs.hpp"
#include "rvjit/interface/linalg/matmul/plan.hpp"
#include "rvjit/interface/programming_model.hpp"
#include "rvjit/interface/resource/resource.hpp"
#include "rvjit/interface/stream/iterator/plan.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"
#include "rvjit/interface/stream/operand/operand.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {

using buffer_stream_t = stream::operand::operand_impl_t;
using memory_stream_t = stream::memory_operand::memory_operand_impl_t;
using memory_operand_t = stream::memory_operand::memory_operand_t;

/// Registers every post-op of a chain shares
///
/// @details They are free while the chain emits: the K-loop streams and
///     variables are dead by then, and the matmul allocates them from the
///     registers those left behind, the ones only inputs may draw included.
///     A post-op declaring `needs_t::mask` never gets those
struct scratch_t {
    rvjit::buffer::buffer_t accs, floats, ints;

    /// The `i`-th register of the accumulators' class
    ///
    /// @param i The position
    name_t acc(int i) const { return accs.id(i); }

    /// The `i`-th float register
    ///
    /// @param i The position
    freg_t f(int i) const { return freg_t(floats.id(i)); }

    /// The `i`-th integer register
    ///
    /// @param i The position
    reg_t x(int i) const { return reg_t(ints.id(i)); }
};

/// What a post-op knows about the tile when it declares its needs
///
/// @details Holds no emitter or register pool: the needs of a chain can be
///     asked of a plan alone
struct postop_shape_t {
    /// Creates the shape of a tile
    ///
    /// @param p The plan, which must outlive the shape
    explicit postop_shape_t(const plan_t &p) : plan(p) {}

    const plan_t &plan;

    int mb = 1, nb = 1; // The largest dispatched tile, in granules and columns
    int m_grain = 1; // Elements of one M granule
    int sewb_c = 0; // Bytes of one output element

    // C's storage form, as the store stream uses it
    form_t c_form = form_t::undefined;
};

/// What a post-op sees while it plans and emits
struct postop_env_t : public postop_shape_t {
    /// Creates the environment of a matmul
    ///
    /// @param pm The programming model
    /// @param ms The memory operand factory
    /// @param p The plan, which must outlive the environment
    postop_env_t(programming_model_t &pm, memory_operand_t &ms, const plan_t &p)
        : postop_shape_t(p), pm(pm), ms(ms) {}

    programming_model_t &pm;
    memory_operand_t &ms;

    // The implementation's tuning of a C-shaped memory operand
    std::function<void(stream::memory_operand::spec_t &,
            stream::memory_operand::variant_t &)>
            tune_c;

    // Switches the configuration to a type's width, keeping the vector length
    conversion::configure_t configure;

    variable_t ptrc;
    const_t c_m_stride; // Bytes between C's M granules
    const_t c_n_stride; // Bytes between C's N granules

    // Byte offset of the M tile, advanced with the M loop; absent (the zero
    // register) when M is dispatched once
    variable_t m_off;
    variable_t args; // The plan's `postop_args`

    scratch_t scratch;

    /// The memory operand over one C-shaped `{nb, mb}` tile
    ///
    /// @param dir Whether the operand loads or stores C
    /// @param update What happens to C's pointer after the tile
    stream::memory_operand::spec_t c_spec(
            memory::dir_t dir, const stream::iterator::update_t &update) const {
        stream::memory_operand::spec_t spec;
        spec.op = memory::op_t(c_form, dir, plan.dtc.mem);
        spec.origin = pointer_t(ptrc.buffer());
        spec.space
                = stream::iterator::space_t({nb, mb}, {c_n_stride, c_m_stride});
        spec.update = update;
        return spec;
    }

    /// The variant of `c_spec`, over the registers `ring`
    ///
    /// @param ring The destination or source registers
    stream::memory_operand::variant_t c_variant(
            const buffer_map_t &ring) const {
        stream::memory_operand::variant_t variant;
        variant.ring = ring;
        return variant;
    }

    /// Creates a C-shaped memory operand the implementation has tuned
    ///
    /// @param spec The operand, from `c_spec()`
    /// @param variant The preferred variant, from `c_variant()`
    stream::memory_operand::memory_operand_impl_t *create_c(
            stream::memory_operand::spec_t spec,
            stream::memory_operand::variant_t variant) const {
        if (tune_c) tune_c(spec, variant);
        return ms.create(spec, variant);
    }
};

/// One dispatched tile of accumulators a post-op operates on
struct postop_tile_t {
    postop_env_t &env;
    buffer_stream_t &acc; // The accumulators, shaped `{mb, nb}`
    int mb, nb; // The dispatched tile

    /// Calls `f(name_t, {m, n})` for every accumulator, in storage order
    ///
    /// @param f The callable
    template <typename F>
    void for_each_acc(F &&f) const {
        for (int n = 0; n < nb; ++n)
            for (int m = 0; m < mb; ++m)
                f(acc.get({m, n}), index_t {m, n});
    }

    /// Byte offset of M granule `m` within the tile
    ///
    /// @param m The M granule
    int m_bytes(int m) const { return m * env.m_grain * env.sewb_c; }

    /// The register holding the tile's M byte offset
    ///
    /// @note The zero register when M is dispatched once
    ///
    /// @pre The post-op declared `needs_t::m_offset`
    reg_t m_offset() const { return env.m_off.buffer(); }

    /// The register holding the plan's `postop_args`
    ///
    /// @pre The post-op declared `needs_t::args`
    reg_t args() const { return env.args.buffer(); }
};

/// One step applied to the accumulators of a tile before they are stored
///
/// @details A post-op declares its register needs, which the matmul resolves
///     once into the shared `scratch_t`, plans what else it keeps (streams) in
///     the registers the matmul frees at the epilogue, and emits after the K
///     loop, in the accumulators' register groups
struct postop_t {
    static constexpr const char *NAME = "rvjit::linalg::matmul::postop";

    virtual ~postop_t() = default;

    /// The `codegen_flags_t` bit gating this post-op at run time
    virtual int flag() const { return with_postops; }

    /// Registers this post-op uses while it emits
    ///
    /// @param shape The plan and the tile it is asked about
    virtual needs_t needs(const postop_shape_t &shape) const {
        (void)shape;
        return {};
    }

    /// Allocates what the post-op keeps across tiles, within `env.scratch`
    ///
    /// @param env The environment of the matmul
    ///
    /// @note Returns the resource it lacked, if any
    virtual shortfall_t plan(postop_env_t &env) {
        (void)env;
        return {};
    }

    /// Emits the post-op over the tile's accumulators
    ///
    /// @param tile The accumulators and the environment
    virtual bool emit(postop_tile_t &tile) {
        (void)tile;
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }
};

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
