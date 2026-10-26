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

#ifndef RVJIT_UTILS_UTILS_HPP
#define RVJIT_UTILS_UTILS_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <utility>
#include <initializer_list>
#include <type_traits>

#include "rvjit/interface/isa/types.hpp"

namespace rvjit {

/// std::make_unique equivalent
///
/// @param args The arguments forwarded to the constructor of `T`
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

/// Whether every argument is valid
///
/// @param args The objects checked, each exposing `is_valid()`
template <typename... Args>
bool all_valid(const Args &...args) {
    bool ok = true;
    const bool unused[] = {true, (ok = ok && args.is_valid())...};
    (void)unused;
    return ok;
}

/// A compile-time sequence of indices, std::index_sequence in C++11
template <std::size_t... Is>
struct index_seq_t {};

/// Builds `index_seq_t<0, ..., N - 1>` as its `type`
template <std::size_t N, std::size_t... Is>
struct make_index_seq_t {
    using type = typename make_index_seq_t<N - 1, N - 1, Is...>::type;
};

template <std::size_t... Is>
struct make_index_seq_t<0, Is...> {
    using type = index_seq_t<Is...>;
};

/// A set of values stored on an array with fixed capacity `N`
///
/// @details Thin wrapper with init and size logic for array-like rvjit types
template <typename T, int N>
struct array_t {
    static constexpr int MAXN = N;

    array_t() = default;

    /// Creates a set from a range, dropping the values past the capacity
    ///
    /// @param s The first value of the range
    /// @param l One past the last value of the range
    template <typename It>
    array_t(It s, It l) {
        for (auto i = s; i != l && n_ < N; ++i)
            e_[n_++] = *i;
    }

    /// Creates a set of a single value
    ///
    /// @param value The only value
    array_t(T value) : n_(1) { e_[0] = value; }

    /// Creates a set from a list, dropping the values past the capacity
    ///
    /// @param list The values
    array_t(std::initializer_list<T> list)
        : array_t(list.begin(), list.end()) {}

    /// Creates a set from `n` consecutive values
    ///
    /// @param d The first value
    /// @param n The number of values
    array_t(const T *d, int n) : array_t(d, d + n) {}

    /// Number of values
    int size() const { return n_; }

    /// Whether the capacity is exhausted
    bool full() const { return n_ == N; }

    /// Appends a value, returns false if the capacity is exhausted
    ///
    /// @param value The value appended
    bool push_back(T value) {
        if (full()) return false;
        e_[n_++] = value;
        return true;
    }

    /// Removes and returns the last value
    ///
    /// @pre `size() > 0`
    T pop_back() { return e_[--n_]; }

    /// Removes every value
    void clear() { n_ = 0; }

    /// The value at position `i`
    ///
    /// @param i The position
    T operator[](int i) const { return e_[i]; }

    /// The value at position `i`
    ///
    /// @param i The position
    T &operator[](int i) { return e_[i]; }

protected:
    std::array<T, N> e_ {};
    int n_ = 0;
};

/// Whether an integer value is representable as a signed 12-bit value
///
/// @param i The value
inline bool is_simm12(int i) {
    return -2048 <= i && i < 2048;
}

/// Whether a value is a power of two
///
/// @param v The value
inline bool is_pow2(int64_t v) {
    return v > 0 && (v & (v - 1)) == 0;
}

namespace detail {

// Single-return stages of the SWAR popcount, as C++11 `constexpr` requires
constexpr uint32_t popcount_pairs(uint32_t v) {
    return v - ((v >> 1) & 0x55555555u);
}
constexpr uint32_t popcount_nibbles(uint32_t v) {
    return (v & 0x33333333u) + ((v >> 2) & 0x33333333u);
}
constexpr uint32_t popcount_bytes(uint32_t v) {
    return (v + (v >> 4)) & 0x0f0f0f0fu;
}

} // namespace detail

/// Number of bits set in a 32-bit word
///
/// @param v The word
constexpr int popcount(uint32_t v) {
    return static_cast<int>((detail::popcount_bytes(detail::popcount_nibbles(
                                     detail::popcount_pairs(v)))
                                    * 0x01010101u)
            >> 24);
}

/// Index of the lowest bit set in a 32-bit word
///
/// @param v The word
///
/// @pre `v != 0`
constexpr int ctz(uint32_t v) {
    return popcount((v & (~v + 1u)) - 1u);
}

/// Base-2 logarithm of a power-of-two value
///
/// @param v The value
///
/// @pre `is_pow2(v)` is true
inline int ilog2q(int64_t v) {
    int r = 0;
    while (v > 1) {
        v >>= 1;
        ++r;
    }
    return r;
}

/// Fills `out` with a descending power-of-two breakdown of `value`
///
/// @param value The value broken down
/// @param out The destination, with room for `max_depth` entries
/// @param max_depth The most entries written
///
/// @note Returns the number of entries written
inline int pow2_breakdown(int value, int *out, int max_depth = 8) {
    if (value <= 0) return 0;

    int start = 1;
    while (start * 2 <= value)
        start *= 2;

    int n = 0;
    for (int p = start; p >= 1 && n < max_depth; p /= 2)
        out[n++] = p;

    return n;
}

} // namespace rvjit

#endif
