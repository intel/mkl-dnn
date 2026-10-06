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

#ifndef GEMMSTONE_TOOLS_TYPES_COMPLEX_HELPERS_HPP
#define GEMMSTONE_TOOLS_TYPES_COMPLEX_HELPERS_HPP

#include <type_traits>

// Custom complex struct with more flexible type support than std::complex.
template <typename T>
struct complex;

template <typename T>
struct ComplexHelper {
    using real_type = T;
    static constexpr bool isComplex = false;
};

template <typename T>
struct ComplexHelper<complex<T>> {
    using real_type = T;
    static constexpr bool isComplex = true;
};

template <typename T> constexpr bool isComplex() { return ComplexHelper<T>::isComplex; }
template <typename T> constexpr int numComponents() { return isComplex<T>() ? 2 : 1; }

template <typename T> using real_t = typename ComplexHelper<T>::real_type;

template <typename T>
struct complex {
    T _real, _imag;

    constexpr complex() {}
    constexpr complex(const T &r, const T &i) : _real(r), _imag(i) {}

    template <typename U,
              typename _ = std::enable_if_t<!isComplex<U>()>>
    constexpr complex(const U &r) : _real(T(r)), _imag(0) {}

    template <typename U>
    constexpr complex(const complex<U> &o) : _real(o._real), _imag(o._imag) {}

    constexpr T real() const { return _real; }
    constexpr T imag() const { return _imag; }

    void real(T value) { _real = value; }
    void imag(T value) { _imag = value; }

    complex<T> operator+() const {
        return *this;
    }

    complex<T> operator-() const {
        auto result = *this;
        result._real = -result._real;
        result._imag = -result._imag;
        return result;
    }

    template <typename U>
    complex<T> &operator+=(const U &o) {
        *this = *this + o;
        return *this;
    }

    template <typename U>
    complex<T> &operator-=(const U &o) {
        *this = *this - o;
        return *this;
    }

    template <typename U>
    complex<T> &operator*=(const U &o) {
        *this = *this * o;
        return *this;
    }

    template <typename U>
    complex<T> &operator/=(const U &o) {
        *this = *this / o;
        return *this;
    }

    operator T() const { return _real; }
};

template <typename T, typename R, typename I>
T makeComplex(R real, I imag)
{
    if constexpr (isComplex<T>()) {
        using T_real = typename ComplexHelper<T>::real_type;
        return T(T_real(real), T_real(imag));
    } else
        return T(real);
}

template <typename T1, typename T2>
static inline auto operator+(const complex<T1> &z1, const complex<T2> &z2)
{
    using U = decltype(T1() + T2());
    return complex<U>(z1.real() + z2.real(), z1.imag() + z2.imag());
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T2>()>>
static inline auto operator+(const complex<T1> &z1, const T2 &r2)
{
    return z1 + complex<T2>(r2);
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T1>()>>
static inline auto operator+(const T1 &r1, const complex<T2> &z2)
{
    return complex<T1>(r1) + z2;
}

template <typename T1, typename T2>
static inline auto operator-(const complex<T1> &z1, const complex<T2> &z2)
{
    using U = decltype(T1() - T2());
    return complex<U>(z1.real() - z2.real(), z1.imag() - z2.imag());
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T2>()>>
static inline auto operator-(const complex<T1> &z1, const T2 &r2)
{
    return z1 - complex<T2>(r2);
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T1>()>>
static inline auto operator-(const T1 &r1, const complex<T2> &z2)
{
    return complex<T1>(r1) - z2;
}

template <typename T1, typename T2>
static inline auto operator*(const complex<T1> &z1, const complex<T2> &z2)
{
    using U = decltype(T1() * T2());
    return complex<U>(z1.real() * z2.real() - z1.imag() * z2.imag(),
                      z1.real() * z2.imag() + z1.imag() * z2.real());
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T2>()>>
static inline auto operator*(const complex<T1> &z1, const T2 &r2)
{
    return z1 * complex<T2>(r2);
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T1>()>>
static inline auto operator*(const T1 &r1, const complex<T2> &z2)
{
    return complex<T1>(r1) * z2;
}

template <typename T1, typename T2>
static inline auto operator/(const complex<T1> &z1, const complex<T2> &z2)
{
    // No overflow protection...
    using U = decltype(T1() / T2());
    auto mag2 = z2.real() * z2.real() + z2.imag() * z2.imag();
    return complex<U>((z1.real() * z2.real() + z1.imag() * z2.imag()) / mag2,
                      (z1.imag() * z2.real() - z1.real() * z2.imag()) / mag2);
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T2>()>>
static inline auto operator/(const complex<T1> &z1, const T2 &r2)
{
    using U = decltype(T1() / T2());
    return complex<U>(z1.real() / r2, z1.imag() / r2);
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T1>()>>
static inline auto operator/(const T1 &r1, const complex<T2> &z2)
{
    return complex<T1>(r1) / z2;
}

template <typename T1, typename T2>
static inline bool operator==(const complex<T1> &z1, const complex<T2> &z2)
{
    return (z1.real() == z2.real()) && (z1.imag() == z2.imag());
}

template <typename T1, typename T2,
          typename _ = std::enable_if_t<!isComplex<T2>()>>
static inline bool operator==(const complex<T1> &z1, T2 z2)
{
    return (z1 == complex<T1>(z2));
}

template <typename T1, typename T2>
static inline bool operator!=(const complex<T1> &z1, const T2 &z2)
{
    return !(z1 == z2);
}

namespace std {
	template <typename T>
    static inline bool isfinite(const ::complex<T> &z) {
        return isfinite(z.real()) && isfinite(z.imag());
    }

    template <typename T>
    static inline ::complex<T> conj(const ::complex<T> &z) {
        return ::complex<T>(z.real(), -z.imag());
    }

    template <typename T>
    static inline T real(const ::complex<T> &z) {
        return z.real();
    }

    template <typename T>
    static inline T imag(const ::complex<T> &z) {
        return z.imag();
    }

    template <typename T>
    static inline T abs(const ::complex<T> &z) {
        // No overflow protection...
        return std::sqrt(z.real()*z.real() + z.imag()*z.imag());
    }
}

template <typename T>
static inline std::ostream &operator<<(std::ostream &s, const complex<T> &z)
{
    s << z.real();
    if (z.imag() < 0)
        s << " - " << std::abs(z.imag()) << 'i';
    else if (z.imag() > 0)
        s << " + " << z.imag() << 'i';
    return s;
}

#endif /* header guard */
