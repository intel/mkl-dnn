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

#ifndef RVJIT_INTERFACE_ISA_BUFFER_BUFFER_HPP
#define RVJIT_INTERFACE_ISA_BUFFER_BUFFER_HPP

#include <cstdint>
#include <initializer_list>

#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace buffer {

/// Constant allocation order over buffer names
using order_t = array_t<name_t, 32>;

/// Set of buffer/register names stored as a 32-bit bitmask
///
/// @note Ids are enumerated in ascending name order, see `id()`
struct buffer_t {
    static constexpr int MAXN = 32;

    buffer_t() = default;

    /// Builds a buffer from a range of registers or names
    ///
    /// @param first The first element of the range
    /// @param last One past the last element of the range
    template <typename It>
    buffer_t(It first, It last) {
        for (auto it = first; it != last; ++it)
            set(name_of_(*it));
    }

    /// Builds a buffer from the first `n` names in `d`
    ///
    /// @param d The first name
    /// @param n The number of names
    buffer_t(const name_t *d, int n) : buffer_t(d, d + n) {}

    /// Builds a buffer from a list of registers or names
    ///
    /// @param l The registers or names
    template <typename RegT>
    buffer_t(std::initializer_list<RegT> l) : buffer_t(l.begin(), l.end()) {}

    /// Builds a buffer from the names in `order`
    ///
    /// @param order The names
    explicit buffer_t(const order_t &order) {
        for (int i = 0; i < order.size(); ++i)
            set(order[i]);
    }

    /// Number of names in the buffer
    int size() const { return popcount(mask_); }

    /// Whether the buffer holds no name
    bool is_empty() const { return mask_ == 0; }

    /// Whether the buffer holds every name
    bool full() const { return mask_ == ~0u; }

    /// Whether the buffer holds `name`
    ///
    /// @param name The name looked up
    ///
    /// @note A name past `MAXN` (absent, zero, stack) is never held
    bool has(name_t name) const {
        return name < MAXN && ((mask_ >> name) & 1u);
    }

    /// Adds `name` to the buffer
    ///
    /// @param name The name added
    ///
    /// @note A name past `MAXN` is ignored
    void set(name_t name) {
        if (name < MAXN) mask_ |= 1u << name;
    }

    /// Removes `name` from the buffer
    ///
    /// @param name The name removed
    void clear(name_t name) {
        if (name < MAXN) mask_ &= ~(1u << name);
    }

    /// The name of the `i`-th smallest name in the buffer
    ///
    /// @param i The position in ascending name order
    ///
    /// @note Returns `no_name` when `i` is outside `[0, size())`
    name_t id(int i) const {
        if (i < 0) return no_name;
        uint32_t m = mask_;
        for (; i > 0 && m; --i)
            m &= m - 1;
        return m ? static_cast<name_t>(ctz(m)) : no_name;
    }

    /// The name of the `i`-th smallest name in the buffer
    ///
    /// @param i The position in ascending name order
    ///
    /// @note Returns `no_name` when `i` is outside `[0, size())`
    name_t operator[](int i) const { return id(i); }

    /// The bitmask of names, bit `n` set when the buffer holds name `n`
    uint32_t mask() const { return mask_; }

    /// The buffer whose names are the set bits of `mask`
    ///
    /// @param mask The bitmask of names
    static buffer_t from_mask(uint32_t mask) {
        buffer_t r;
        r.mask_ = mask;
        return r;
    }

    /// Returns a copy of this buffer without ids specified on `excl`
    ///
    /// @param excl The names removed
    buffer_t filter(const buffer_t &excl) const {
        buffer_t r;
        r.mask_ = mask_ & ~excl.mask_;
        return r;
    }

    /// Lists the names present in this buffer following `order`
    ///
    /// @param order The preferred order of names
    ///
    /// @note Names not in `order` are appended in ascending order
    order_t ordered(const order_t &order) const {
        order_t r;
        buffer_t rest = *this;
        for (int i = 0; i < order.size(); ++i) {
            if (!has(order[i])) continue;
            r.push_back(order[i]);
            rest.clear(order[i]);
        }
        for (int i = 0, n = rest.size(); i < n; ++i)
            r.push_back(rest.id(i));
        return r;
    }

    /// The names held by either buffer
    ///
    /// @param o The buffer merged
    buffer_t operator|(const buffer_t &o) const {
        buffer_t r;
        r.mask_ = mask_ | o.mask_;
        return r;
    }

    /// The names held by both buffers
    ///
    /// @param o The buffer intersected
    buffer_t operator&(const buffer_t &o) const {
        buffer_t r;
        r.mask_ = mask_ & o.mask_;
        return r;
    }

    /// Whether both buffers hold the same names
    ///
    /// @param o The buffer compared against
    bool operator==(const buffer_t &o) const { return mask_ == o.mask_; }

    /// Whether the buffers hold different names
    ///
    /// @param o The buffer compared against
    bool operator!=(const buffer_t &o) const { return mask_ != o.mask_; }

private:
    static name_t name_of_(name_t n) { return n; }

    template <typename RegT>
    static name_t name_of_(const RegT &r) {
        return reg_name(r);
    }

    uint32_t mask_ = 0;
};

} // namespace buffer
} // namespace rvjit

#endif
