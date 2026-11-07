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

#ifndef RVJIT_IMPLEMENTATION_RVV_LINALG_MATMUL_HPP
#define RVJIT_IMPLEMENTATION_RVV_LINALG_MATMUL_HPP

#include <algorithm>
#include <utility>

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rv64/stream/iterator/iterator.hpp"
#include "rvjit/implementation/rvv/linalg/tuning.hpp"
#include "rvjit/implementation/rvv/programming_model.hpp"
#include "rvjit/implementation/rvv/rvv.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/linalg/matmul/impl.hpp"
#include "rvjit/interface/loop/loop.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rvv {
namespace linalg {
namespace matmul {

/// RVV implementation decisions for `linalg::matmul::matmul_t`
struct impl_t : public rvjit::linalg::matmul::impl_t {
    static constexpr const char *NAME = "rvjit::rvv::linalg::matmul";

    /// Creates the RVV implementation
    ///
    /// @param pm The RVV programming model the engine builds its kernel with
    explicit impl_t(rvjit::rvv::programming_model_t &pm)
        : rvjit::linalg::matmul::impl_t(pm), pm_(pm), tuning_(pm.model) {}

    /// Forces the input LMUL instead of the model-driven choice
    ///
    /// @param m The input LMUL
    ///
    /// @note The accumulator LMUL is derived from it; a widening plan whose
    ///     accumulator LMUL would exceed m8 is rejected
    void set_lmul(LMUL m) { tuning_.set_lmul(m); }

    /// The tuning this implementation decides with
    const tuning_t &tuning() const { return tuning_; }

    /// @details Groups the vector register file as the tuning says
    bool configure(const rvjit::linalg::matmul::plan_t &p) override {
        const auto layout = tuning_.vector_layout(p);
        CHECK(layout.is_valid(), "no valid vector layout for the plan");
        CHECK(pm_.isa().buffers().configure_vectors(layout),
                "unable to group the vector registers");
        return true;
    }

    /// Resolves the loop unrolling, granularity and operand decisions
    ///
    /// @param p The plan
    ///
    /// @pre `configure()` has returned true
    bool operator()(const rvjit::linalg::matmul::plan_t &p) override {
        tuned_t tuned;
        CHECK(tuning_(p, pm_.isa().buffers(), epilogue_needs_fn, tuned),
                "no valid tuning");

        // Configure vtype according to the plan
        pm_.vtype.sewi = tuned.vtype.sewi;
        pm_.vtype.sewo = tuned.vtype.sewo;
        pm_.vtype.lmuli = tuned.vtype.lmuli;
        pm_.vtype.lmulo = tuned.vtype.lmulo;

        // Source A and C from vector registers
        a_form = rvjit::form_t::vector;
        c_form = rvjit::form_t::vector;

        // source B from scalar registers (the scalar component on vfmacc.vf)
        b_form = rvjit::form_t::scalar;

        // A transposed A is read with strided vector loads, whose stride
        // `lda` must live in a register, even when it is an immediate
        is_transa_ = p.is_transa;
        lda_ = p.lda;
        a_ld_ = const_t();

        // Determine the granularity of each loop (only M is accelerated)
        m_grain = tuned.m_grain;
        n_grain = tuned.n_grain;
        k_grain = tuned.k_grain;

        // The unroll of each loop
        mb = tuned.mb;
        nb = tuned.nb;
        kb = tuned.kb;
        ktb = tuned.ktb;
        return true;
    }

    /// @details A transposed A keeps `lda` in a register, even when it is an
    ///     immediate
    shortfall_t allocate() override {
        a_ld_ = const_t();
        if (!is_transa_) return {};
        a_ld_ = pm_.isa().buffers().new_constant(lda_.source());
        if (a_ld_.is_imm()) {
            a_ld_ = a_ld_.in_reg();
            if (!pm_.isa().buffers().allocate(a_ld_))
                return shortfall_t::of(resource_t::integers);
        }
        CHECK_OR_RET(a_ld_.is_valid() && a_ld_.has_reg(),
                shortfall_t::of(resource_t::integers), "no lda register");
        return {};
    }

    /// @details Vector loads address only through their base register, so a
    ///     `guided` walk is enough; a transposed A loads with the stride `lda`
    void set_a_operand(
            memory_spec_t &spec, memory_variant_t &variant) override {
        spec.addressing = rvjit::stream::iterator::addressing_t::base_only;
        variant.strategy = rv64::stream::iterator::guided;
        variant.ntmp = 2;
        if (is_transa_) spec.op.strided(a_ld_.buffer());
    }

    /// @details B is read with scalar loads, which take an immediate offset:
    ///     let the iterator trade registers for pointer arithmetic
    void set_b_operand(
            memory_spec_t &spec, memory_variant_t &variant) override {
        spec.addressing = rvjit::stream::iterator::addressing_t::base_simm;
        variant.strategy = rvjit::stream::iterator::strategy_id_t::any();
        variant.ntmp = -1;
    }

    void set_c_operand(
            memory_spec_t &spec, memory_variant_t &variant) override {
        spec.addressing = rvjit::stream::iterator::addressing_t::base_only;
        variant.strategy = rv64::stream::iterator::guided;
        variant.ntmp = 2;
    }

    void prepare() override { pm_.isa().transfer().init(a_ld_); }

    /// @note Sets the vtype using the output SEW, LMUL configurations, with
    ///     the remaining M work-units as its avl
    void on_m_tile_enter(int /*mb*/, const reg_t &remaining) override {
        pm_.vtype.avl = remaining;
        pm_.set_vtype(rvjit::rvv::vtype_t::cfg_t::output);
    }

    /// @note Sets the vtype using the input SEW, LMUL if mixed precision
    ///
    /// @details Necessary as init, or store, may use a different vtype; it
    ///     keeps the vector length, since the avl register may be lent to the
    ///     body by now
    void on_k_loop_enter(
            int /*mb*/, int /*nb*/, const reg_t & /*remaining*/) override {
        pm_.set_vtype_keep_vl_if_mixed(rvjit::rvv::vtype_t::cfg_t::input);
    }

    void on_k_loop_exit(
            int /*mb*/, int /*nb*/, const reg_t & /*remaining*/) override {
        pm_.set_vtype_keep_vl_if_mixed(rvjit::rvv::vtype_t::cfg_t::output);
    }

    bool on_configure(rvjit::type_t t) override {
        return pm_.set_vtype_keep_vl(rvjit::rvv::sew_for(t));
    }

private:
    rvjit::rvv::programming_model_t &pm_;

    bool is_transa_ = false;
    parameter_t lda_; // The plan's `lda`
    const_t a_ld_; // The register holding `lda` of a transposed A

    tuning_t tuning_;
};

} // namespace matmul
} // namespace linalg
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
