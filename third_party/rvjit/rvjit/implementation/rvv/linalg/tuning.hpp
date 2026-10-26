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

#ifndef RVJIT_IMPLEMENTATION_RVV_LINALG_TUNING_HPP
#define RVJIT_IMPLEMENTATION_RVV_LINALG_TUNING_HPP

#include <algorithm>
#include <functional>

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rvv/rvv.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/linalg/matmul/needs.hpp"
#include "rvjit/interface/linalg/matmul/plan.hpp"
#include "rvjit/interface/linalg/matmul/tuned.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rvv {
namespace linalg {
namespace matmul {

/// What `tuning_t` decides for a plan: the generic decisions, an M granule
/// being one vector, and the vtype that realizes them
struct tuned_t : public rvjit::linalg::matmul::tuned_t {
    /// The SEW and LMUL of the inputs and the accumulators
    vtype_t vtype;
};

/// The RVV decisions about a matmul plan: vector grouping and block sizes
///
/// @details A function of the plan and the hardware alone, holding no emitter
///     or programming model, so a host can ask what a kernel for a plan will
///     do before it builds one. The blocks it returns, handed back in
///     `plan_t::blocks`, are honored as they are
struct tuning_t {
    static constexpr const char *NAME = "rvjit::rvv::linalg::matmul::tuning";

    /// Creates the tuning for a machine
    ///
    /// @param model The hardware model
    explicit tuning_t(const rvjit::rvv::vpu_t &model) : model_(model) {}

    /// Forces the input LMUL instead of the model-driven choice
    ///
    /// @param m The input LMUL
    ///
    /// @note The accumulator LMUL is derived from it; a widening plan whose
    ///     accumulator LMUL would exceed m8 is rejected
    void set_lmul(LMUL m) {
        lmul_ = m;
        has_lmul_ = true;
    }

    /// The vector registers only inputs may draw, keeping `v0` (the mask
    /// register) away from accumulators and epilogue values
    static rvjit::buffer::buffer_t reserved_vectors() {
        const name_t v0 = 0;
        return rvjit::buffer::buffer_t(&v0, 1);
    }

    /// How the vector register file must be grouped to run `p`
    ///
    /// @param p The plan
    ///
    /// @note An invalid layout is returned when the plan has none
    rvjit::buffer::vector_layout_t vector_layout(
            const rvjit::linalg::matmul::plan_t &p) const {
        if (!accumulator_lmul_fits_(p.dta.reg, p.dtc.reg)) return {0, 0, {}};
        const vtype_t vt = vtype_of(p);
        return {vgroup_size(vt.lmuli), vgroup_size(vt.lmulo),
                reserved_vectors()};
    }

    /// The SEW and LMUL pair `p`'s data types run at
    ///
    /// @param p The plan
    ///
    /// @details The input group is the model's preference, widened when a
    ///     conversion of `p` runs at a width too narrow to keep the
    ///     accumulators' vector length in a whole register group, and
    ///     narrowed while a bounded M fits the half group, unless the LMUL is
    ///     forced
    ///
    /// @note Does not retry against a smaller LMUL when the derived
    ///     accumulator LMUL would exceed the model's max_lmul; `vtype_t`
    ///     does not validate this, so an unsupported combination is on
    ///     the caller to avoid
    vtype_t vtype_of(const rvjit::linalg::matmul::plan_t &p) const {
        const rvjit::type_t dta = p.dta.reg, dtc = p.dtc.reg;
        const SEW sewi = sew_for(dta);
        const SEW sewo = sew_for(dtc);
        LMUL m = input_lmul_(dta != dtc);

        // The narrowest width a conversion runs at sets the least output
        // group, which the input group times the widening ratio must reach
        if (!has_lmul_) {
            const int wo = rvjit::sewb(dtc);
            int narrowest = wo;
            const rvjit::type_t at[] = {p.dtc.mem, p.epi_type(), p.dta.mem};
            for (rvjit::type_t t : at)
                narrowest = std::min(narrowest, rvjit::sewb(t));
            if (p.flags & rvjit::linalg::matmul::with_bias)
                narrowest = std::min(narrowest, rvjit::sewb(p.bias_type()));
            const int ratio = wo / rvjit::sewb(dta);
            const int least_out = wo / narrowest;
            const int least_in = std::max(1, least_out / std::max(ratio, 1));
            if (vgroup_size(m) < least_in) m = lmul_for(least_in);

            // A bounded M that half the group still holds leaves lanes idle
            const int m_bound = p.m.bound();
            const int sewib = rvjit::sewb(dta);
            while (m_bound > 0 && vgroup_size(m) / 2 >= least_in
                    && model_.vlenb * (vgroup_size(m) / 2) / sewib >= m_bound)
                m = lmul_for(vgroup_size(m) / 2);
        }

        vtype_t vt;
        if (dta == dtc)
            vt.for_uniform(sewo, m);
        else
            vt.for_mixed(sewi, sewo, m);
        return vt;
    }

    /// Decides the vector grouping and the blocks for `p`
    ///
    /// @param p The plan; the blocks in `p.blocks` are honored as they are or
    ///     the plan is rejected
    /// @param pool The register pool, whose vector file is grouped as
    ///     `vector_layout(p)` says
    /// @param epilogue What the epilogue needs for the given M vectors, nothing
    ///     when empty
    /// @param out The decisions
    ///
    /// @note Returns false, logging the reason, when the plan has no valid
    ///     decision. The accumulators take the groups the pipeline leaves, at
    ///     most 16, or 8 from a VLEN of 2048 bits; the epilogue reuses the input groups, `v0`'s included unless it needs
    ///     the mask, and only a minimum that does not fit shrinks them. An
    ///     unpinned M takes at most `mb_max` vectors, and is unrolled only when
    ///     the integer registers left after the epilogue's reach
    ///     `m_unroll_min_ints`
    bool operator()(const rvjit::linalg::matmul::plan_t &p,
            rvjit::buffer::buffer_pool_t &pool,
            const std::function<rvjit::linalg::matmul::needs_t(int)> &epilogue,
            tuned_t &out) const {
        CHECK(accumulator_lmul_fits_(p.dta.reg, p.dtc.reg),
                "an accumulator LMUL beyond m8");

        // Configure vtype according to the plan
        out = tuned_t();
        out.vtype = vtype_of(p);
        const vtype_t &vt = out.vtype;

        // Every conversion runs at a width whose LMUL, scaled from the
        // accumulators' to keep the vector length, is a whole register group
        const auto fits = [&](rvjit::type_t t) {
            const int group = vgroup_size(out.vtype.lmulo);
            const int from = rvjit::rvv::sewb(out.vtype.sewo);
            const int to = rvjit::sewb(t);
            return to >= from || group * to % from == 0;
        };
        CHECK(fits(p.dtc.mem), "an output narrower than a register group");
        CHECK(fits(p.epi_type()), "an epilogue type narrower than a group");
        CHECK(fits(p.dta.mem), "an A narrower than a register group");
        CHECK(!(p.flags & rvjit::linalg::matmul::with_bias)
                        || fits(p.bias_type()),
                "a bias narrower than a register group");

        // Derive the MAXVL and SEWB from the vtype configurations
        const int maxvl
                = get_maximum_vector_length(model_.vlenb, vt.sewi, vt.lmuli);
        const int sewib = rvjit::rvv::sewb(vt.sewi);
        const int sewob = rvjit::rvv::sewb(vt.sewo);
        out.m_grain = maxvl;

        const auto needs = [&](int mb) {
            return epilogue ? epilogue(mb) : rvjit::linalg::matmul::needs_t();
        };

        // Determine the K loop unrolling factor (reduce loop)
        // The empirical value 8 shows good results on different micro-arch
        static constexpr int kb_max = 8;
        out.kb = kb_max;
        if (p.blocks.kb > 0) {
            CHECK(p.blocks.kb <= kb_max, "a pinned K unroll beyond %d", kb_max);
            out.kb = p.blocks.kb;
        }

        // Determine the K loop unrolling factor for the N loop tail(s)
        // Use shorter unrolling to reduce code size on lower impact code paths
        static constexpr int ktb_max = 4;
        out.ktb = ktb_max;
        while (out.ktb > 1 && out.ktb >= out.kb)
            out.ktb /= 2;

        // The accumulator groups left by the pipeline buffers (one when mixed
        // precision or m8, else two). The epilogue reuses the input groups
        // once the K loop is done, keeping `v0`'s group clear when it needs
        // the mask
        const int out_groups = get_nvgroups(vt.lmulo);
        const auto epilogue_groups = [&](int mb) {
            const rvjit::linalg::matmul::needs_t n = needs(mb);
            return n.accs + (n.mask ? 1 : 0);
        };
        CHECK(out_groups - 1 >= epilogue_groups(1),
                "the epilogue leaves no vector register for the accumulators");
        const bool one_spare = vt.is_mixed() || vt.lmulo == LMUL::m8;
        const int ceiling = std::min(out_groups - (one_spare ? 1 : 2),
                out_groups - epilogue_groups(1));
        // Long vectors (VLEN >= 2048 bits) take at most 8 accumulators
        const int nb_max = model_.vlenb * 8 >= 2048 ? 8 : 16;
        const int max_acc = std::min(nb_max, std::max(1, ceiling));
        CHECK(p.blocks.nb <= max_acc,
                "a pinned N block of %d beyond the accumulator ceiling %d",
                p.blocks.nb, max_acc);
        out.nb = max_acc;
        if (p.blocks.nb > 0)
            out.nb = p.blocks.nb;
        else if (p.n.has_bound())
            out.nb = std::min(p.n.bound(), out.nb);

        // M: unroll as many vector tiles as M and the registers left after
        // N's accumulators and A's double buffer allow, up to `mb_max`; the
        // tile pointer steps must fold into one simm12 `addi`. A pinned count
        // is honored as is. An unbounded M is unrolled as a large bound is,
        // its tails left to the M loop
        static constexpr int mb_max = 4;
        const int m_step_max = 2047 / (maxvl * std::max(sewib, sewob));
        const bool m_pinned = p.blocks.mb > 0;
        CHECK(!m_pinned || p.blocks.mb <= std::max(m_step_max, 1),
                "a pinned M block of %d vectors beyond the pointer step range",
                p.blocks.mb);
        int m_ur = 1;
        if (m_pinned) {
            m_ur = p.blocks.mb;
        } else {
            const int m_tiles = p.m.has_bound() ? p.m.bound() / maxvl : mb_max;
            m_ur = std::min({m_tiles, m_step_max, mb_max});
        }

        // An unrolled M tile addresses each M-indexed stream through its own
        // pointer, which only kernels with integer registers to spare afford
        // (empirical, from the register-pressure sweeps)
        static constexpr int m_unroll_min_ints = 20;
        while (m_ur > 1) {
            const bool fits_pool = pool.vectors().may_reserve(out.nb * m_ur, 2);
            const bool fits_epilogue
                    = out.nb * m_ur + epilogue_groups(m_ur) <= out_groups;
            const int ints = pool.integers().available() - needs(m_ur).ints;
            const bool fits_ints = m_pinned || ints >= m_unroll_min_ints;
            if (fits_pool && fits_epilogue && fits_ints) break;
            CHECK(!m_pinned, "a pinned M block of %d vectors that does not fit",
                    p.blocks.mb);
            --m_ur;
        }
        out.mb = std::max(m_ur, 1);
        return true;
    }

private:
    /// The input LMUL: the forced one, else the model's preference capped
    LMUL input_lmul_(bool widening) const {
        if (has_lmul_) return lmul_;
        const LMUL m = widening ? LMUL::m2 : LMUL::m4;
        return is_lmul_gte(m, model_.max_lmul) ? model_.max_lmul : m;
    }

    /// Whether the accumulators of `dta` x `dtc` fit a group of at most m8
    ///
    /// @note A widening output takes `sewb(dtc) / sewb(dta)` times the
    ///     input group, which `lmul_for` would otherwise silently map to m1
    bool accumulator_lmul_fits_(rvjit::type_t dta, rvjit::type_t dtc) const {
        if (dta == dtc) return true;
        const int in = vgroup_size(input_lmul_(true));
        const int out = in * (rvjit::sewb(dtc) / rvjit::sewb(dta));
        CHECK(out <= vgroup_size(LMUL::m8),
                "a widening accumulator group of %d registers (input group "
                "%d) beyond m8",
                out, in);
        return true;
    }

    rvjit::rvv::vpu_t model_;
    LMUL lmul_ = LMUL::m1; // Forced input LMUL, if has_lmul_
    bool has_lmul_ = false;
};

} // namespace matmul
} // namespace linalg
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
