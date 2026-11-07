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

#ifndef RVJIT_IMPLEMENTATION_RVV_PROGRAMMING_MODEL_HPP
#define RVJIT_IMPLEMENTATION_RVV_PROGRAMMING_MODEL_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/arithmetic.hpp"
#include "rvjit/implementation/rvv/isa/operation/memory/memory.hpp"
#include "rvjit/implementation/rvv/rvv.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/programming_model.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {

namespace rvv {

/// RVV specialization of `programming_model_t`
///
/// @details Pairs a hardware `model` with a mutable kernel state `vtype`,
///     and owns `vtype`'s emission (`set_vtype`/`set_vtype_if_mixed`),
///     lowering its housed state into a `vsetvli` instruction
struct programming_model_t
    : public programming_model_compose_t<rvv::arithmetic::arithmetic_t,
              rvv::memory::memory_t> {
    static constexpr const char *NAME = "rvjit::rvv";
    using base_t = programming_model_compose_t<rvv::arithmetic::arithmetic_t,
            rvv::memory::memory_t>;

    const vpu_t &model;
    vtype_t vtype;

    /// Creates the RVV programming model
    ///
    /// @param e The emitter instructions are emitted through
    /// @param pool The register allocation pool
    /// @param loop The loop factory
    /// @param stack The stack allocator
    /// @param xfer The value transfer operations
    /// @param arith The constant arithmetic operations
    /// @param cf The control flow operations
    /// @param memops The memory operand factory
    /// @param model The hardware model
    ///
    /// @pre `model` must outlive the programming model
    explicit programming_model_t(rv64::emitter_t e,
            rvjit::buffer::buffer_pool_t &pool, rvjit::loop::loopgen_t &loop,
            rvjit::stack_t &stack, const operation::transfer_t &xfer,
            const operation::const_arithmetic_t &arith,
            const operation::control_flow_t &cf,
            rvjit::stream::memory_operand::memory_operand_t &memops,
            const vpu_t &model)
        : base_t(e, pool, loop, stack, xfer, arith, cf, memops,
                  scheduler_t::from_params(
                          model.memory_latency(), model.vector_latency()))
        , model(model)
        , e_(e) {}

    /// General vtype configuration
    ///
    /// @param gvl The register receiving the granted vector length
    /// @param avl The register holding the application vector length
    /// @param kind Which of the input or output SEW and LMUL is applied
    void set_vtype(const Reg &gvl, const Reg &avl,
            vtype_t::cfg_t kind = vtype_t::cfg_t::input) const {
        auto sew = vtype.sewi;
        auto mul = vtype.lmuli;
        if (kind == vtype_t::cfg_t::output) {
            sew = vtype.sewo;
            mul = vtype.lmulo;
        }
        e_->vsetvli(gvl, avl, sew, mul, vtype.vt, vtype.vm);
    }

    /// General vtype configuration using the housed `vtype`'s own avl
    ///
    /// @param gvl The register receiving the granted vector length
    /// @param kind Which of the input or output SEW and LMUL is applied
    ///
    /// @pre `vtype.avl` must be assigned
    bool set_vtype(const Reg &gvl = ignore_out,
            vtype_t::cfg_t kind = vtype_t::cfg_t::input) const {
        CHECK(is_not_zero(vtype.avl), "invalid avl");
        set_vtype(gvl, vtype.avl, kind);
        return true;
    }

    /// General vtype configuration using the housed `vtype`'s own avl
    ///
    /// @param kind Which of the input or output SEW and LMUL is applied
    ///
    /// @note Uses the `x0` return register, ignoring the granted vector length
    ///
    /// @pre `vtype.avl` must be assigned
    void set_vtype(vtype_t::cfg_t kind) const { set_vtype(ignore_out, kind); }

    /// Conditionally calls `set_vtype` when `vtype` is mixed-precision
    ///
    /// @param kind Which of the input or output SEW and LMUL is applied
    void set_vtype_if_mixed(vtype_t::cfg_t kind) const {
        if (vtype.is_mixed()) set_vtype(kind);
    }

    /// General vtype configuration keeping the current vector length
    ///
    /// @param kind Which of the input or output SEW and LMUL is applied
    ///
    /// @note Needs no avl register, which may have been repurposed since
    ///
    /// @pre The vtype in effect and `kind`'s share the same VLMAX, as the
    ///     input and output settings of a mixed-precision `vtype` do
    void set_vtype_keep_vl(vtype_t::cfg_t kind) const {
        set_vtype(ignore_out, ignore_out, kind);
    }

    /// Conditionally calls `set_vtype_keep_vl` when `vtype` is mixed-precision
    ///
    /// @param kind Which of the input or output SEW and LMUL is applied
    void set_vtype_keep_vl_if_mixed(vtype_t::cfg_t kind) const {
        if (vtype.is_mixed()) set_vtype_keep_vl(kind);
    }

    /// Switches to `sew`, keeping the vector length of the output configuration
    ///
    /// @param sew The element width switched to
    ///
    /// @details The LMUL scales the output's by the ratio of the widths
    ///
    /// @note Returns false, emitting nothing, when that LMUL is fractional or
    ///     beyond m8
    bool set_vtype_keep_vl(SEW sew) const {
        const int group = vgroup_size(vtype.lmulo);
        const int from = sewb(vtype.sewo), to = sewb(sew);
        const int scaled
                = to >= from ? group * (to / from) : group / (from / to);
        const bool fits = scaled >= 1 && scaled <= 8
                && (to >= from ? true : group % (from / to) == 0);
        CHECK(fits, "a configuration beyond the register groups");
        e_->vsetvli(ignore_out, ignore_out, sew, lmul_for(scaled), vtype.vt,
                vtype.vm);
        return true;
    }

    /// Gets the maximum vector length for the host machine and vtype settings
    int maxvl() const {
        return get_maximum_vector_length(model.vlenb, vtype.sewi, vtype.lmuli);
    }

private:
    rv64::emitter_t e_;
};

} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
