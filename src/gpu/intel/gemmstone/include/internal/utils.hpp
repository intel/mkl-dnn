/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GEMMSTONE_INCLUDE_INTERNAL_UTILS_HPP
#define GEMMSTONE_INCLUDE_INTERNAL_UTILS_HPP

#include "gemmstone/config.hpp"

#include <array>
#include <functional>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <type_traits>
#include <map>

#ifdef GEMMSTONE_ENABLE_SOURCE_LOCATION
#include <source_location>
#endif

GEMMSTONE_NAMESPACE_START

template <typename T, typename U>
static inline bool one_of(T value, std::initializer_list<U> list)
{
    for(auto &v: list)
        if(value == v) return true;
    return false;
}

static inline int ilog2(size_t x)
{
#ifdef _MSC_VER
    unsigned long index = 0;
    (void) _BitScanReverse64(&index, __int64(x));
    return index;
#else
    return (sizeof(unsigned long long) * 8 - 1) - __builtin_clzll(x);
#endif
}

template <typename T> static inline constexpr bool equal(T t) { return true; }
template <typename T1, typename T2> static inline constexpr bool equal(T1 t1, T2 t2) { return (t1 == t2); }
template <typename T1, typename T2, typename... To> static inline constexpr bool equal(T1 t1, T2 t2, To... to) {
    return (t1 == t2) && equal(t2, to...);
}

template <typename T> static inline constexpr T clamp(T val, T lo, T hi)
{
    return std::min<T>(hi, std::max<T>(lo, val));
}

template <typename T> static inline T gcd(T x, T y)
{
    if (x == 0) return y;
    if (y == 0) return x;

    // Optimized path for powers of 2 (common case)
    if ((x & (x - 1)) == 0 && (y & (y - 1)) == 0)
        return std::min(x, y);

    // Euclidean algorithm for general values
    T g1 = std::max(x, y), g2 = std::min(x, y);

    for (;;) {
        T g = g1 % g2;
        if (g == 0)
            return g2;
        g1 = g2;
        g2 = g;
    }
}

template <typename T> static inline T lcm(T x, T y)
{
    if (x == 0 || y == 0) return 0;

    if ((x & (x - 1)) == 0 && (y & (y - 1)) == 0)
        return std::max(x, y);

    return (x * y) / gcd(x, y);
}

static inline int largest_pow2_divisor(int x)
{
    return x & ~(x - 1);
}

template <typename T>
inline bool is_pow2(const T &v) {
    return (v > 0) && ((v & (v - 1)) == 0);
}
template <typename T, typename U>
inline decltype(std::declval<T>()/std::declval<U>()) div_up(T a, U b) {
    return a >= 1 ? 1 + (a - 1) / b : 0;
}

template <typename T, typename U>
inline decltype(std::declval<T>()*std::declval<U>()) round_up(T a, U b) {
    return div_up(a, b) * b;
}
static inline int align_up(int a, int b) { return round_up(a, b);}

template <typename T, typename U>
constexpr decltype(std::declval<T>()*std::declval<U>()) round_down(T a, U b) {
    return (a / b) * b;
}
static inline int align_down(int a, int b) { return round_down(a, b);}

template <typename T> static inline T lshift(T x, int shift)
{
    constexpr auto bits = int(sizeof(T) * 8);
    if (shift >= bits)
        return T(0);
    else if (shift >= 0)
        return x << shift;
    else if (shift > -bits)
        return x >> -shift;
    else
        return (x >> (bits/2)) >> (bits/2);
}

template <typename T> static inline T rshift(T x, int shift)
{
    return lshift(x, -shift);
}

class stub_exception : public std::runtime_error
{
public:
    stub_exception(const char *msg = unimpl()) : std::runtime_error(msg) {}
    stub_exception(const char *msg, const char *file, size_t line)
        : std::runtime_error(std::string(msg) + " (at " + std::string(file) + ":" + std::to_string(line) + ")") {}
    stub_exception(const char *file, size_t line) : stub_exception(unimpl(), file, line) {}

protected:
    static const char *unimpl() { return "Functionality is unimplemented"; }
};

class hw_unsupported_exception : public std::runtime_error {
public:
    hw_unsupported_exception() : std::runtime_error("Unsupported in hardware") {}
};

#ifdef GEMMSTONE_ENABLE_SOURCE_LOCATION
[[noreturn]] static inline void stub(
                                     std::source_location where = std::source_location::current()) {
    throw stub_exception(where.file_name(), where.line());
}

[[noreturn]] static inline void stub(const char * msg,
                                     std::source_location where = std::source_location::current()) {
    throw stub_exception(msg, where.file_name(), where.line());
}

inline void gemm_assert(bool i, const char *msg = nullptr, std::source_location where = std::source_location::current()) {
#if !defined(NDEBUG) || GEMMSTONE_ASSERTIONS
    if(!i) msg ? stub(msg, where) : stub(where);
#elif defined __clang__
    __builtin_assume(i);
#elif defined (__GNUC__)
    if(!i) __builtin_unreachable();
#elif defined(_MSVC_LANG)
    __assume(i);
#endif
}

inline void gemm_assert(bool i, const std::string &msg, std::source_location where = std::source_location::current()) {
    return gemm_assert(i, msg.c_str(), where);
}

#else
[[noreturn]] static inline void stub() {
    throw stub_exception();
}

[[noreturn]] static inline void stub(const char * msg) {
    throw stub_exception(msg);
}

inline void gemm_assert(bool i, const char *msg = nullptr) {
#if !defined(NDEBUG) || GEMMSTONE_ASSERTIONS
    if(!i) msg ? stub(msg) : stub();
#elif defined __clang__
    __builtin_assume(i);
#elif defined (__GNUC__)
    if(!i) __builtin_unreachable();
#elif defined(_MSVC_LANG)
    __assume(i);
#endif
}

inline void gemm_assert(bool i, const std::string &msg) {
    const char *msg_c = msg.c_str();
    return gemm_assert(i, msg_c);
}

#endif
template <typename out_type, typename in_type>
inline out_type into(in_type in) {
    auto max = static_cast<typename std::make_unsigned<out_type>::type>(std::numeric_limits<out_type>::max());
    auto min = static_cast<typename std::make_signed<out_type>::type>(std::numeric_limits<out_type>::min());
    gemm_assert (in < 0 || static_cast<typename std::make_unsigned<in_type>::type>(in) <= max);
    gemm_assert (in > 0 || static_cast<typename std::make_signed<in_type>::type>(in) >= min);
    return static_cast<out_type>(in);
}

[[noreturn]] static inline void hw_unsupported()
{
    throw hw_unsupported_exception();
}

static inline void noop() {}

struct hasher_t {
    template <typename T>
    size_t operator()(const T &t) {
        return get_hash(t);
    }

    template <typename T, typename... Args> size_t operator()(const T &t, const Args &... args) {
        size_t hash0 = operator()(t);
        size_t hash1 = operator()(args...);
        return hash_combine(hash0, hash1);
    }

private:
    // The following code is derived from Boost C++ library
    // Copyright 2005-2014 Daniel James.
    // Distributed under the Boost Software License, Version 1.0. (See accompanying
    // file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
    template <typename T>
    static size_t hash_combine(size_t seed, const T &v) {
        return seed ^= std::hash<T> {}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template <typename E>
    struct enum_hash_t {
        size_t operator()(const E &e) const noexcept {
            return std::hash<size_t>()((size_t)e);
        }
    };

    template <typename T, typename = void>
    struct get_std_hash_helper_t {
        static size_t call(const T &t) { return std::hash<T>()(t); }
    };

    template <typename T>
    struct get_std_hash_helper_t<T,
                                 typename std::enable_if<std::is_enum<T>::value>::type> {
        static size_t call(const T &t) { return enum_hash_t<T>()(t); }
    };

    template <typename T, typename = void>
    struct get_hash_helper_t {
        static size_t call(const T &t) { return get_std_hash_helper_t<T>::call(t); }
    };

    template <typename T>
    struct get_hash_helper_t<T, decltype(std::declval<T>().get_hash(), void())> {
        static size_t call(const T &t) { return t.get_hash(); }
    };

    template <typename T>
    size_t get_hash(const T &t) {
        return get_hash_helper_t<T>::call(t);
    }

    template <typename T, size_t N>
    size_t get_hash(const std::array<T, N> &a) {
        size_t h = 0;
        for (auto &e : a)
            h = hash_combine(h, get_hash(e));
        return h;
    }

    template <typename T>
    size_t get_hash(const std::vector<T> &v) {
        size_t h = 0;
        for (auto &e : v)
            h = hash_combine(h, get_hash(e));
        return h;
    }

    template <typename Key, typename T, typename Compare, typename Allocator>
    size_t get_hash(const std::map<Key, T, Compare, Allocator> &m) {
        size_t h = 0;
        for (auto &kv : m) {
            h = hash_combine(h, get_hash(kv.first));
            h = hash_combine(h, get_hash(kv.second));
        }
        return h;
    }
};

template <typename... Args>
size_t hash(const Args &...args) {
    return hasher_t{}(args...);
}

struct ostringstream_t : public std::ostringstream {
    template <typename... Args>
    ostringstream_t(Args &&...args)
        : std::ostringstream(std::forward<Args>(args)...) {
        this->imbue(std::locale::classic());
    }

    ostringstream_t(const ostringstream_t &) = delete;
    ostringstream_t &operator=(const ostringstream_t &) = delete;

    ostringstream_t(ostringstream_t &&) = delete;
    ostringstream_t &operator=(ostringstream_t &&) = delete;

private:
    using std::ostringstream::imbue;
};

template <typename T>
struct stringify_t {
#ifdef __GNUC__
    __attribute__((noinline)) __attribute__((used))
#endif
    void dump() const {
        printf("%s\n", static_cast<const T *>(this)->str().c_str());
    }

    friend std::ostream &operator<<(std::ostream &out, const T& t) {
        return out << t.str();
    }
#if __cplusplus >= 202002L
    bool operator==(const stringify_t &) const = default;
#else
protected:
    bool operator==(const stringify_t &) const { return true; }
#endif

};

template <typename T, typename... Types>
struct is_one_of : std::false_type {};

template <typename T, typename Head, typename... Tail>
struct is_one_of<T, Head, Tail...>
    : std::conditional<std::is_same<T, Head>::value, std::true_type,
              is_one_of<T, Tail...>>::type {};

// std::variant functionality with pointer storage.
template <typename... Types>
struct generic_t {
    generic_t() = default;

    template <typename T>
    generic_t(T x)
        : index_(index_of<T, Types...>())
        , impl_(new storage_t<T>(std::move(x))) {
        static_assert(is_one_of<T, Types...>::value, "Unexpected type.");
    }

    generic_t(const generic_t &other)
        : index_(other.index_)
        , impl_(other.impl_ ? other.impl_->clone() : nullptr) {}

    generic_t(generic_t &&other) = default;

    generic_t &operator=(const generic_t &other) {
        if (this == &other) return *this;
        index_ = other.index_;
        impl_ = other.impl_ ? other.impl_->clone() : nullptr;
        return *this;
    }

    generic_t &operator=(generic_t &&other) = default;

    explicit operator bool() const { return (bool)impl_; }

    template <typename T>
    bool is() const {
        return impl_ && impl_->type() == typeid(T);
    }

    template <typename T>
    T &as() {
        gemm_assert(is<T>());
        return static_cast<storage_t<T> *>(impl_.get())->data;
    }

    template <typename T>
    const T &as() const {
        gemm_assert(is<T>());
        return static_cast<const storage_t<T> *>(impl_.get())->data;
    }

    template <typename T>
    T *as_ptr() {
        if (!is<T>()) return nullptr;
        return &as<T>();
    }

    template <typename T>
    const T *as_ptr() const {
        if (!is<T>()) return nullptr;
        return &as<T>();
    }

    size_t index() const { return index_; }

private:
    template <typename T>
    static constexpr size_t index_of() {
        return 1;
    }

    template <typename T, typename First, typename ... Rest>
    static constexpr size_t index_of() {
        return std::is_same<T, First>::value ? 0 : 1 + index_of<T, Rest...>();
    }

    struct storage_impl_t {
        virtual ~storage_impl_t() = default;
        virtual const std::type_info &type() const = 0;
        virtual std::unique_ptr<storage_impl_t> clone() const = 0;
    };

    template <typename T>
    struct storage_t : storage_impl_t {
        T data;
        storage_t(T x) : data(std::move(x)) {}
        const std::type_info &type() const override { return typeid(T); }
        std::unique_ptr<storage_impl_t> clone() const override {
            return std::unique_ptr<storage_impl_t>(new storage_t<T>(data));
        }
    };

    size_t index_ = sizeof...(Types);
    std::unique_ptr<storage_impl_t> impl_;

    template <typename ... Ts, typename F>
    friend auto visit(const generic_t<Ts...> &g, const F &f) -> decltype(g.invoke(f));
};

template <typename ... Ts> struct overload_t;

template <typename T, typename ... Ts>
struct overload_t<T, Ts...> : T, overload_t<Ts...> {
    using T::operator();
    using overload_t<Ts...>::operator();

    overload_t(const T &t, const Ts &...ts)
        : T(t), overload_t<Ts...>(ts...) {}
};

template <typename T>
struct overload_t<T> : T {
    using T::operator();

    overload_t(const T &t) : T(t) {}
};

template <typename ... Ts>
overload_t<Ts...> overload(const Ts &...ts) {
    return overload_t<Ts...>{ts...};
}

template <typename First, typename ... Rest, typename F>
auto visit(const generic_t<First, Rest...> &g, const F &f) -> decltype(f(std::declval<First>())) {
    using return_type_t = decltype(f(std::declval<First>()));
    gemm_assert(g.index() < 1 + sizeof...(Rest));
    std::array<std::function<return_type_t(void)>, 1 + sizeof...(Rest)> funcs = {
        [&]() { return f(g.template as<First>()); },
        [&]() { return f(g.template as<Rest>()); }...
    };
    return funcs[g.index()]();
}

GEMMSTONE_NAMESPACE_END

#endif // GEMMSTONE_INCLUDE_INTERNAL_UTILS_HPP
