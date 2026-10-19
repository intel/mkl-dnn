/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GEMMSTONE_DSL_UTILS_UTILS_HPP
#define GEMMSTONE_DSL_UTILS_UTILS_HPP

#include <array>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "gemmstone/config.hpp"
#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

using ngen::utils::rounddown_pow2;
using ngen::utils::roundup_pow2;

namespace dsl {

template <typename derived_type, typename base_type>
inline derived_type downcast(base_type *base) {
    gemm_assert(dynamic_cast<derived_type>(base) == base);
    return static_cast<derived_type>(base);
}

// Implementation of std::bit_cast for interoperability with C++17 and earlier
template <typename T, typename U>
inline T bit_cast(const U &u) {
    static_assert(sizeof(T) == sizeof(U), "Bit-casting must preserve size.");
    static_assert(std::is_trivially_copyable<T>::value
                    && std::is_trivially_constructible<T>::value,
            "T must be trivially copyable and constructible.");
    static_assert(std::is_trivially_copyable<U>::value
                    && std::is_trivially_constructible<U>::value,
            "U must be trivially copyable.");
    T t;
    std::memcpy(&t, &u, sizeof(T));
    return t;
}

// Implementation of std::make_unique for C++11 interoperability
template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args &&...args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

class error_stream_t {
public:
    error_stream_t(const char *file, int line, const char *assert_msg) {
        out_ << "Assertion " << assert_msg << " failed at " << file << ":"
             << line << std::endl;
    }

    // This is to be able use a steam object in short-circuit evaluation with
    // booleans, see below.
    operator bool() const { return true; }

    template <typename T>
    error_stream_t &operator<<(const T &t) {
        out_ << t;
        return *this;
    }

    ~error_stream_t() noexcept(false) {
#if __cplusplus < 201703L || (defined(_MSVC_LANG) && _MSVC_LANG < 201703L)
        if (std::uncaught_exception()) {
#else
        if (std::uncaught_exceptions()) {
#endif
            return;
        }

        printf("%s\n", out_.str().c_str());
#ifdef GPU_ABORT_ON_ERROR
        std::abort();
#else
        throw std::runtime_error(out_.str());
#endif
    }

private:
    ostringstream_t out_;
};

#if !defined(NDEBUG) || GEMMSTONE_ASSERTIONS
#define dsl_assert(cond) \
    !(cond) && gemmstone::dsl::error_stream_t(__FILE__, __LINE__, #cond)
#else
#define dsl_assert(cond) \
    (false) && !(cond) \
            && gemmstone::dsl::error_stream_t(__FILE__, __LINE__, #cond)
#endif

#define dsl_error() dsl_assert(false) << "Not Expected. "

template <typename T>
using name_map_t = std::unordered_map<T, std::string>;
template <typename T>
const name_map_t<T> &get_name_map();

template <typename T>
const std::string &to_string(T value) {
    auto &map = get_name_map<T>();
    auto ret = map.find(value);
    if (ret == map.end()) stub();
    return ret->second;
}
template <>
inline const std::string &to_string(bool value) {
    static const std::array<std::string, 2> ret {"true", "false"};
    return value ? ret[0] : ret[1];
}

template <typename T>
T from_string(const std::string &str) {
    auto &map = get_name_map<T>();
    for (auto &entry : map) {
        if (entry.second == str) return entry.first;
    }
    stub();
    return {};
}

namespace ir {
using gemmstone::dsl::from_string;
using gemmstone::dsl::to_string;
} // namespace ir

template <>
inline const name_map_t<ngen::HW> &get_name_map() {
    static const name_map_t<ngen::HW> names {
            {ngen::HW::Unknown, "unknown"},
            {ngen::HW::Gen9, "Gen9"},
            {ngen::HW::Gen10, "Gen10"},
            {ngen::HW::Gen11, "Gen11"},
            {ngen::HW::XeLP, "XeLP"},
            {ngen::HW::XeHP, "XeHP"},
            {ngen::HW::XeHPG, "XeHPG"},
            {ngen::HW::XeHPC, "XeHPC"},
            {ngen::HW::Xe2, "Xe2"},
            {ngen::HW::Xe3, "Xe3"},
            {ngen::HW::Xe3p, "Xe3p"},
    };
    return names;
}
template <>
inline const name_map_t<ngen::ProductFamily> &get_name_map() {
    static const name_map_t<ngen::ProductFamily> names {
            {ngen::ProductFamily::Unknown, "unknown"},
            {ngen::ProductFamily::GenericGen9, "Gen9"},
            {ngen::ProductFamily::GenericGen10, "Gen10"},
            {ngen::ProductFamily::GenericGen11, "Gen11"},
            {ngen::ProductFamily::GenericXeLP, "XeLP"},
            {ngen::ProductFamily::GenericXeHP, "XeHP"},
            {ngen::ProductFamily::GenericXeHPG, "XeHPG"},
            {ngen::ProductFamily::DG2, "DG2"},
            {ngen::ProductFamily::MTL, "MTL"},
            {ngen::ProductFamily::ARL, "ARL"},
            {ngen::ProductFamily::GenericXeHPC, "XeHPC"},
            {ngen::ProductFamily::PVC, "PVC"},
            {ngen::ProductFamily::GenericXe2, "Xe2"},
            {ngen::ProductFamily::GenericXe3, "Xe3"},
            {ngen::ProductFamily::GenericXe3p, "Xe3p"},
            {ngen::ProductFamily::NVLP, "NVLP"},
            {ngen::ProductFamily::CRI, "CRI"},
    };
    return names;
}

template <>
inline const name_map_t<ngen::PlatformType> &get_name_map() {
    static const name_map_t<ngen::PlatformType> names = {
            {ngen::PlatformType::Unknown, "Unknown"},
            {ngen::PlatformType::Integrated, "Integrated"},
            {ngen::PlatformType::Discrete, "Discrete"},
    };
    return names;
}

inline std::string to_string(const ngen::Product &product) {
    return to_string(product.family) + ": platform - " + to_string(product.type)
            + ", stepping - " + std::to_string(product.stepping);
}

template <typename T>
inline void maybe_unused(const T &x) {
    (void)(x);
}

namespace utils {

template <typename T, typename U, typename = void>
struct is_equal_helper_t {
    static bool call(const T &t, const U &u) { return t == u; }
};

template <typename T, typename U>
struct is_equal_helper_t<T, U,
        decltype(std::declval<T>().is_equal(std::declval<U>()), void())> {
    static bool call(const T &t, const U &u) { return t.is_equal(u); }
};

// Checks equality of objects:
// 1. Uses t.is_equal(u) if is_equal() is available
// 2. Uses (t == u) otherwise
template <typename T, typename U>
bool is_equal(const T &t, const U &u) {
    return is_equal_helper_t<T, U>::call(t, u);
}

// Checks equality of vector elements.
template <typename T, typename U>
bool is_equal(const std::vector<T> &a, const std::vector<U> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (!utils::is_equal(a[i], b[i])) return false;
    return true;
}

// Checks identity of vector elements.
template <typename T, typename U>
bool is_same(const std::vector<T> &a, const std::vector<U> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (!a[i].is_same(b[i])) return false;
    return true;
}
template <typename ContainerT>
struct seq_print_helper_t {
    seq_print_helper_t(const ContainerT &v, const std::string &sep, int width)
        : v(v), sep(sep), width(width) {}

    const ContainerT &v;
    const std::string sep;
    int width;
};

template <typename T>
seq_print_helper_t<T> make_seq_print_helper(
        const T &v, const std::string &sep = ", ", int width = 0) {
    return seq_print_helper_t<T>(v, sep, width);
}

template <typename T>
inline std::ostream &operator<<(
        std::ostream &out, const seq_print_helper_t<T> &seq) {
    for (auto it = seq.v.begin(); it != seq.v.end(); it++) {
        out << (it != seq.v.begin() ? seq.sep : "") << std::setw(seq.width)
            << *it;
    }
    return out;
}

template <typename T>
inline T max_divisor(T n, std::initializer_list<T> divisors) {
    T ret = -1;
    for (auto d : divisors) {
        if (n % d == 0) ret = std::max(ret, d);
    }
    gemm_assert(ret != -1);
    return ret;
}

// Adapted version of magicgu function from Hacker's Delight 10-15.
inline void idiv_magicgu(uint32_t d, uint32_t &m, uint32_t &p) {
    uint32_t s32_max = std::numeric_limits<int32_t>::max();
    gemm_assert(d != 0 && d <= s32_max);
    uint64_t nc = (s32_max / d) * d - 1;
    for (p = 32; p < 64; p++) {
        uint64_t _2p = 1LL << p;
        if (_2p > nc * (d - 1 - (_2p - 1) % d)) {
            m = into<uint32_t>((_2p + d - 1 - (_2p - 1) % d) / d);
            return;
        }
    }
    stub();
}

inline uint64_t idiv_magicgu_packed(uint32_t d) {
    uint32_t m = 0, p = 0;
    if (is_pow2(d)) {
        p = ilog2(d);
    } else {
        utils::idiv_magicgu(d, m, p);
    }
    return m + (static_cast<uint64_t>(p) << 32);
}

template <typename T, typename U>
inline typename std::remove_reference<T>::type max_div(const T a, const U b) {
    U div = b;
    while (div > 1) {
        if (a % div == 0) return div;
        div--;
    }
    return static_cast<typename std::remove_reference<T>::type>(div);
}

template <typename T, typename U>
inline T safe_divide(T a, U b) {
    dsl_assert(b != 0 && a % b == 0) << "Can't divide: " << a << " / " << b;
    return a / b;
}

} // namespace utils

inline bool stream_try_match(std::istream &in, const std::string &s) {
    in >> std::ws;
    auto pos = in.tellg();
    bool ok = true;
    for (auto &c : s) {
        if (in.get() != c || in.fail()) {
            ok = false;
            break;
        }
    }
    if (!ok) {
        in.clear();
        in.seekg(pos);
    }
    return ok;
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
