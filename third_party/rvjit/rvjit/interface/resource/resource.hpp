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

#ifndef RVJIT_INTERFACE_RESOURCE_RESOURCE_HPP
#define RVJIT_INTERFACE_RESOURCE_RESOURCE_HPP

#include "rvjit/utils/utils.hpp"

namespace rvjit {

/// Register classes a candidate may run short of
enum class resource_t : unsigned {
    integers = 1,
    floats = 2,
    vectors = 4,
    stack = 8,
};

/// Description of a demand expressed as counts for each resource class
///
/// @details `INF` stands for a count no resource can satisfy and `-INF` for
///     its opposite; a class in `unknown` has no count
///
/// @note Arithmetic saturates: `-INF` absorbs `INF`, which absorbs any count
struct demand_t {
    // Enumerators, so binding them to a reference needs no definition
    enum : int { INF = 1 << 20, CLASSES = 4, ALL = (1 << CLASSES) - 1 };

    int n[CLASSES] = {};
    unsigned unknown = 0;

    /// Creates a demand of `count` resources of a single class
    ///
    /// @param r The resource class
    /// @param count The number of resources, saturated at `INF`
    static demand_t of(resource_t r, int count) {
        demand_t d;
        d.n[index(r)] = clamp_(count);
        return d;
    }

    /// A demand knowing nothing of any class
    static demand_t unknowns() {
        demand_t d;
        d.unknown = ALL;
        return d;
    }

    /// The position of class `r` in `n`
    ///
    /// @param r The resource class
    ///
    /// @pre `r` is one of the `resource_t` classes
    static int index(resource_t r) { return ilog2q(static_cast<unsigned>(r)); }

    /// The class of index `i`, the inverse of `index()`
    ///
    /// @param i The position in `n`
    static resource_t at(int i) { return static_cast<resource_t>(1u << i); }

    /// The demand count of class `r`
    ///
    /// @param r The resource class
    int operator[](resource_t r) const { return n[index(r)]; }

    /// Whether the demand count of class `r` is known
    ///
    /// @param r The resource class
    bool knows(resource_t r) const {
        return !(unknown & static_cast<unsigned>(r));
    }

    /// Whether nothing is demanded and every count is known
    bool is_none() const {
        for (int i = 0; i < CLASSES; ++i)
            if (n[i] != 0) return false;
        return unknown == 0;
    }

    /// Per class sum of both demands
    ///
    /// @param o The demand added
    demand_t operator+(const demand_t &o) const {
        demand_t d;
        d.unknown = unknown | o.unknown;
        for (int i = 0; i < CLASSES; ++i) {
            const int a = n[i], b = o.n[i];
            d.n[i] = a <= -INF || b <= -INF ? -INF
                    : a >= INF || b >= INF  ? INF
                                            : clamp_(a + b);
        }
        return d;
    }

    /// Per class difference of both demands
    ///
    /// @param o The demand subtracted
    ///
    /// @note Subtracting `INF` gives `-INF` whatever is subtracted from
    demand_t operator-(const demand_t &o) const {
        demand_t d;
        d.unknown = unknown | o.unknown;
        for (int i = 0; i < CLASSES; ++i) {
            const int a = n[i], b = o.n[i];
            d.n[i] = b >= INF ? -INF : a >= INF ? INF : clamp_(a - b);
        }
        return d;
    }

    /// Per class, the larger count
    ///
    /// @param o The demand compared against
    demand_t max(const demand_t &o) const {
        demand_t d;
        d.unknown = unknown | o.unknown;
        for (int i = 0; i < CLASSES; ++i)
            d.n[i] = n[i] > o.n[i] ? n[i] : o.n[i];
        return d;
    }

private:
    static int clamp_(int v) { return v > INF ? INF : v < -INF ? -INF : v; }
};

/// Set of resource classes an allocation attempt ran short of
///
/// @details `need` is a lower bound of the registers missing per class, 0
///     meaning only that at least one is, and `demand_t::INF` that no pool
///     can satisfy the attempt as it stands
///
/// @note A search rules candidates out against these bounds
struct shortfall_t {
    unsigned mask = 0;
    int need[demand_t::CLASSES] = {};

    /// Creates a shortfall of a single class
    ///
    /// @param r The resource class that fell short
    /// @param missing The registers missing, 0 when only one is known missing
    static shortfall_t of(resource_t r, int missing = 0) {
        shortfall_t s;
        s.mask = static_cast<unsigned>(r);
        s.need[demand_t::index(r)] = missing > 0 ? missing : 0;
        return s;
    }

    /// Creates the shortfall of the known classes of `lack` with a count
    ///
    /// @param lack The registers missing per class
    static shortfall_t of(const demand_t &lack) {
        shortfall_t s;
        for (int i = 0; i < demand_t::CLASSES; ++i) {
            const resource_t r = demand_t::at(i);
            if (lack.knows(r) && lack[r] > 0) s = s | of(r, lack[r]);
        }
        return s;
    }

    /// Whether no class fell short
    bool ok() const { return mask == 0; }

    /// Whether both shortfalls share a class
    ///
    /// @param o The shortfall compared against
    bool intersects(const shortfall_t &o) const { return (mask & o.mask) != 0; }

    /// Registers of class `r` missing, at least 1 if the class fell short
    ///
    /// @param r The resource class
    int missing(resource_t r) const {
        if (!(mask & static_cast<unsigned>(r))) return 0;
        const int i = demand_t::index(r);
        return need[i] > 0 ? need[i] : 1;
    }

    /// Whether the missing count of class `r` is known
    ///
    /// @param r The resource class
    bool knows(resource_t r) const {
        return (mask & static_cast<unsigned>(r))
                && need[demand_t::index(r)] > 0;
    }

    /// The union of both shortfalls, with the larger count per class
    ///
    /// @param o The shortfall merged
    shortfall_t operator|(const shortfall_t &o) const {
        shortfall_t s;
        s.mask = mask | o.mask;
        for (int i = 0; i < demand_t::CLASSES; ++i)
            s.need[i] = need[i] > o.need[i] ? need[i] : o.need[i];
        return s;
    }
};

} // namespace rvjit

#endif
