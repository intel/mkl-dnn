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

#ifndef RVJIT_INTERFACE_RESOURCE_OPTIMIZATION_SEARCH_HPP
#define RVJIT_INTERFACE_RESOURCE_OPTIMIZATION_SEARCH_HPP

#include <array>
#include <functional>

#include "rvjit/interface/isa/buffer/uses_transactions.hpp"
#include "rvjit/interface/resource/resource_optimizable.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace optimization {

/// Attempts to build one candidate, reporting what it ran short of
using build_t = std::function<shortfall_t()>;

/// Looks for the first candidate, in preference order, that fits
///
/// @details Candidates are the product of the axes' levels, the first axis
///     varying slowest. Each runs inside a transaction over the guarded
///     components, committed for the first that fits and dropped otherwise
///
/// @note A candidate is skipped when the reliefs of its levels cannot cover
///     what a recorded failure missed, which is sound only while every
///     `relief()` is an upper bound and every shortfall a lower bound
///
/// @note Holds non-owning references; the axes and guards must outlive it
struct search_t {
    static constexpr const char *NAME = "rvjit::optimization::search";
    static constexpr int CAPACITY = 8;

    /// Failures remembered at once; the oldest is forgotten first
    static constexpr int FAILURES = 8;

    /// Adds every component of a ready-made guard list
    ///
    /// @param list The components whose allocation state is rolled back
    template <std::size_t N>
    search_t &guard(const std::array<transactional_t *, N> &list) {
        for (transactional_t *g : list)
            LOG_FAIL_IF(!guards_.push_back(g), "too many guards");
        return *this;
    }

    /// Declares the components whose allocation state is rolled back per
    /// candidate
    ///
    /// @param t The first component
    /// @param ts The remaining components
    template <typename... Ts>
    search_t &guard(transactional_t &t, Ts &...ts) {
        transactional_t *list[] = {&t, &ts...};
        for (transactional_t *g : list)
            LOG_FAIL_IF(!guards_.push_back(g), "too many guards");
        return *this;
    }

    /// Declares the axes; the argument order is the preference order, with
    /// the first varying slowest
    ///
    /// @param o The first axis
    /// @param os The remaining axes
    template <typename... Os>
    search_t &axis(resource_optimizable_t &o, Os &...os) {
        resource_optimizable_t *list[] = {&o, &os...};
        for (resource_optimizable_t *a : list)
            LOG_FAIL_IF(!axes_.push_back(a), "too many axes");
        return *this;
    }

    /// Declares the axes of a ready-made lever list, in its order
    ///
    /// @param list The axes
    template <int N>
    search_t &axis(const array_t<resource_optimizable_t *, N> &list) {
        for (int i = 0; i < list.size(); ++i)
            LOG_FAIL_IF(!axes_.push_back(list[i]), "too many axes");
        return *this;
    }

    /// Builds the ruled-out candidates too, committing the first candidate
    /// that fits as an exhaustive search would, and logs any that fits
    ///
    /// @param on Whether to verify
    ///
    /// @note Defaults to on when `RVJIT_SEARCH_VERIFY` is defined
    search_t &verify(bool on = true) {
        verify_ = on;
        return *this;
    }

    /// What the search spent
    struct stats_t {
        int builds = 0; // candidates built and not ruled out
        int skips = 0; // ruled-out prefixes skipped without a build
        int verified = 0; // ruled-out candidates built by `verify()`
        int mispruned = 0; // ruled-out candidates that fit
    };

    /// What the last `run()` spent
    const stats_t &stats() const { return stats_; }

    /// Runs `build` on each candidate until one fits, then commits it
    ///
    /// @param build Builds a candidate, reporting what it ran short of
    ///
    /// @note The winning levels remain set on the axes
    bool run(const build_t &build) {
        const int n = axes_.size();
        int lv[CAPACITY] = {};
        stats_ = stats_t {};
        nfailures_ = 0;
        bool more = verify_ || seek_(lv);
        while (more) {
            const bool ruled_out = !viable_(lv, n - 1);
            // Build the candidate inside a transaction over the guards
            shortfall_t s;
            bool fits = false;
            {
                transactional_t *list[CAPACITY];
                for (int i = 0; i < guards_.size(); ++i)
                    list[i] = guards_[i];
                transaction_t txn(
                        transaction_t::list_t {}, list, guards_.size());
                bool levels_set = static_cast<bool>(txn);
                for (int i = 0; i < n && levels_set; ++i)
                    levels_set = axes_[i]->set_level(lv[i]);
                if (levels_set) {
                    s = build();
                    fits = s.ok() && txn.commit();
                }
            }
            if (ruled_out) {
                ++stats_.verified;
                if (fits) {
                    ++stats_.mispruned;
                    LOG_FAIL(__func__, "a ruled-out candidate fits");
                    return true;
                }
            } else {
                ++stats_.builds;
                if (fits) return true;

                // Record the failure, the oldest forgotten first, with the
                // most each axis may relieve from it
                if (nfailures_ == FAILURES) {
                    for (int i = 1; i < FAILURES; ++i)
                        failures_[i - 1] = failures_[i];
                    --nfailures_;
                }
                failure_t &f = failures_[nfailures_++];
                f.lack = s;
                for (int j = 0; j < n; ++j) {
                    f.lv[j] = lv[j];
                    f.best[j] = demand_t {};
                    for (int l = 0; l < axes_[j]->levels(); ++l)
                        f.best[j] = f.best[j].max(axes_[j]->relief(lv[j], l));
                }
            }
            more = advance_(lv, n - 1) && (verify_ || seek_(lv));
        }
        LOG_FAIL(__func__, "no candidate fits");
        return false;
    }

private:
    /// A failed candidate and, per axis, the most any level relieves from it
    struct failure_t {
        int lv[CAPACITY] = {};
        shortfall_t lack;
        demand_t best[CAPACITY];
    };

    /// Whether candidates with the levels of axes `[0, d]` may still fit
    /// every recorded failure, whatever the later axes' levels
    bool viable_(const int *lv, int d) const {
        for (int i = 0; i < nfailures_; ++i) {
            const failure_t &f = failures_[i];
            demand_t sum;
            for (int j = 0; j < axes_.size(); ++j)
                sum = sum
                        + (j <= d ? axes_[j]->relief(f.lv[j], lv[j])
                                  : f.best[j]);
            for (int c = 0; c < demand_t::CLASSES; ++c) {
                const resource_t r = demand_t::at(c);
                if (!f.lack.intersects(shortfall_t::of(r)) || !sum.knows(r))
                    continue;
                if (sum[r] < f.lack.missing(r)) return false;
            }
        }
        return true;
    }

    /// Moves `lv` past every candidate sharing its levels of axes `[0, d]`
    bool advance_(int *lv, int d) const {
        for (int j = d + 1; j < axes_.size(); ++j)
            lv[j] = 0;
        while (d >= 0 && lv[d] + 1 >= axes_[d]->levels())
            lv[d--] = 0;
        if (d < 0) return false;
        ++lv[d];
        return true;
    }

    /// Moves `lv` to the first candidate from it on, in order, that no
    /// recorded failure rules out
    bool seek_(int *lv) {
        const int n = axes_.size();
        while (true) {
            int d = 0;
            while (d < n && viable_(lv, d))
                ++d;
            if (d == n) return n > 0 || viable_(lv, -1);
            ++stats_.skips;
            if (!advance_(lv, d)) return false;
        }
    }

    stats_t stats_;
#if defined(RVJIT_SEARCH_VERIFY)
    bool verify_ = true;
#else
    bool verify_ = false;
#endif
    failure_t failures_[FAILURES];
    int nfailures_ = 0;
    array_t<transactional_t *, 8> guards_;
    array_t<resource_optimizable_t *, CAPACITY> axes_;
};

} // namespace optimization
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
