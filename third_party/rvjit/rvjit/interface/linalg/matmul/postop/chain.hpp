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

#ifndef RVJIT_INTERFACE_LINALG_MATMUL_POSTOP_CHAIN_HPP
#define RVJIT_INTERFACE_LINALG_MATMUL_POSTOP_CHAIN_HPP

#include "rvjit/interface/linalg/matmul/plan.hpp"
#include "rvjit/interface/linalg/matmul/postop.hpp"
#include "rvjit/interface/linalg/matmul/postop/builtin.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace linalg {
namespace matmul {

/// The post-ops a plan makes active: alpha, beta, bias, then the plan's own
///
/// @details An entry is active when its codegen flag is set and, if the
///     runtime flag word is a literal, set in it. Needs no emitter, so the
///     registers a plan's chain asks for are known without building a kernel
///
/// @note The entries point into the chain: it is neither copied nor moved
struct postop_chain_t {
    static constexpr const char *NAME = "rvjit::linalg::matmul::postop_chain";
    static constexpr int MAX_ENTRIES = 3 + plan_t::MAX_POSTOPS;

    postop_chain_t() = default;
    postop_chain_t(const postop_chain_t &) = delete;
    postop_chain_t &operator=(const postop_chain_t &) = delete;

    /// Selects the entries `p` makes active
    ///
    /// @param p The plan, which must outlive the chain
    ///
    /// @note Returns false on an inconsistent plan
    bool select(const plan_t &p) {
        n_ = 0;
        mask_ = 0;
        const int flags = p.flags;
        CHECK(!(flags & with_alpha) || p.alpha.is_set(),
                "with_alpha without an alpha");
        CHECK(!(flags & with_beta) || p.beta.is_set(),
                "with_beta without a beta");
        CHECK(!(flags & with_bias) || !p.bias.source().is_none(),
                "with_bias without a bias pointer");
        CHECK(!(flags & with_postops) || p.n_postops > 0,
                "with_postops without post-ops");

        // The flag word of a run is known now when it is a literal
        rt_runtime_
                = !p.rt_flags.source().is_none() && !p.rt_flags.is_literal();
        int enabled = flags;
        if (p.rt_flags.is_literal()) enabled &= p.rt_flags.value();

        const auto add = [&](postop_t *e) {
            if (!(enabled & e->flag())) return true;
            CHECK(n_ < MAX_ENTRIES, "too many post-ops");
            entries_[n_++] = e;
            mask_ |= e->flag();
            return true;
        };
        if (!add(&scale_) || !add(&accumulate_) || !add(&bias_)) return false;
        for (int i = 0; i < p.n_postops; ++i)
            if (!add(p.postops[i])) return false;

        // With nothing active the flag word is never read
        if (n_ == 0) rt_runtime_ = false;
        return true;
    }

    /// The number of active entries
    int size() const { return n_; }

    /// The `i`-th active entry
    ///
    /// @param i The position
    postop_t *operator[](int i) const { return entries_[i]; }

    /// The codegen flags of the active entries
    int mask() const { return mask_; }

    /// Whether the entries are gated by a runtime flag word
    bool rt_runtime() const { return rt_runtime_; }

    /// The registers the active entries need for a tile
    ///
    /// @param shape The plan and the tile
    needs_t needs(const postop_shape_t &shape) const {
        needs_t n;
        for (int i = 0; i < n_; ++i)
            n.merge(entries_[i]->needs(shape));
        return n;
    }

private:
    postop::scale_t scale_;
    postop::accumulate_t accumulate_;
    postop::bias_add_t bias_;
    postop_t *entries_[MAX_ENTRIES] = {};
    int n_ = 0;
    int mask_ = 0;
    bool rt_runtime_ = false;
};

/// The registers the post-op chain of `p` needs for a tile of `mb` x `nb`
///
/// @param p The plan
/// @param mb The M granules of the tile
/// @param nb The columns of the tile
/// @param out The needs
///
/// @note Returns false on an inconsistent plan
inline bool chain_needs(const plan_t &p, int mb, int nb, needs_t &out) {
    postop_chain_t chain;
    if (!chain.select(p)) return false;
    postop_shape_t shape(p);
    shape.mb = mb;
    shape.nb = nb;
    out = chain.needs(shape);
    return true;
}

} // namespace matmul
} // namespace linalg
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
