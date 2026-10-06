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

#ifndef GEMMSTONE_TOOLS_TYPES_NF4_HPP
#define GEMMSTONE_TOOLS_TYPES_NF4_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

#include "fp16.hpp"
#include "bfloat16.hpp"
#include "bfloat8.hpp"
#include "int4.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace nf4_impl {

template <typename T> struct is_float_double         { static constexpr bool value = false; };
template <>           struct is_float_double<float>  { static constexpr bool value = true; };
template <>           struct is_float_double<double> { static constexpr bool value = true; };

} /* namespace nf4_impl */

struct nf4 {
    uint8_t raw;

    nf4(int raw_, bool) : raw(raw_) {}

    nf4() {}
    inline nf4(float f);
    nf4(double d) : nf4(float(d)) {}
    template <typename T> nf4(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr) : nf4(float(i)) {}

    operator float() const                                      { return values()[raw & 0xF]; }

    nf4 operator+() const                                       { return *this; }
    float operator-() const                                     { return -float(*this); }

    nf4 operator++()                                            { return (*this = *this + 1); }
    nf4 operator++(int)                                         { nf4 h = *this; ++*this; return h; }
    nf4 operator--()                                            { return (*this = *this - 1); }
    nf4 operator--(int)                                         { nf4 h = *this; --*this; return h; }

    friend float operator+(const nf4 &h1, const nf4 &h2)    { return float(h1) + float(h2); }
    friend float operator-(const nf4 &h1, const nf4 &h2)    { return float(h1) - float(h2); }
    friend float operator*(const nf4 &h1, const nf4 &h2)    { return float(h1) * float(h2); }
    friend float operator/(const nf4 &h1, const nf4 &h2)    { return float(h1) / float(h2); }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const nf4 &h, const T &o)           { return float(h) + float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const nf4 &h, const T &o)           { return float(h) - float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const nf4 &h, const T &o)           { return float(h) * float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const nf4 &h, const T &o)           { return float(h) / float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const T &o, const nf4 &h)           { return float(o) + float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const T &o, const nf4 &h)           { return float(o) - float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const T &o, const nf4 &h)           { return float(o) * float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const T &o, const nf4 &h)           { return float(o) / float(h); }

    template <typename T>
    friend typename std::enable_if<nf4_impl::is_float_double<T>::value, T>::type
    operator+(const nf4 &h, const T &o)           { return float(h) + o; }
    template <typename T>
    friend typename std::enable_if<nf4_impl::is_float_double<T>::value, T>::type
    operator-(const nf4 &h, const T &o)           { return float(h) - o; }
    template <typename T>
    friend typename std::enable_if<nf4_impl::is_float_double<T>::value, T>::type
    operator*(const nf4 &h, const T &o)           { return float(h) * o; }
    template <typename T>
    friend typename std::enable_if<nf4_impl::is_float_double<T>::value, T>::type
    operator/(const nf4 &h, const T &o)           { return float(h) / o; }
    template <typename T>
    friend typename std::enable_if<nf4_impl::is_float_double<T>::value, T>::type
    operator+(const T &o, const nf4 &h)           { return o + float(h); }
    template <typename T>
    friend typename std::enable_if<nf4_impl::is_float_double<T>::value, T>::type
    operator-(const T &o, const nf4 &h)           { return o - float(h); }
    template <typename T>
    friend typename std::enable_if<nf4_impl::is_float_double<T>::value, T>::type
    operator*(const T &o, const nf4 &h)           { return o * float(h); }
    template <typename T>
    friend typename std::enable_if<nf4_impl::is_float_double<T>::value, T>::type
    operator/(const T &o, const nf4 &h)           { return o / float(h); }

    template <typename T>
    nf4 operator+=(const T &o)                    { return *this = nf4(*this + o); }
    template <typename T>
    nf4 operator-=(const T &o)                    { return *this = nf4(*this - o); }
    template <typename T>
    nf4 operator*=(const T &o)                    { return *this = nf4(*this * o); }
    template <typename T>
    nf4 operator/=(const T &o)                    { return *this = nf4(*this / o); }

    static const std::array<float, 16> &values() {
        static std::array<float, 16> v = {
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
             0.0,
             0.07958029955625534,
             0.16093020141124725,
             0.24611230194568634,
             0.33791524171829224,
             0.44070982933044434,
             0.5626170039176941,
             0.7229568362236023,
             1.0,
        };
        return v;
    }
};

inline nf4::nf4(float f)
{
    auto &v = values();
    auto diff = std::abs(f - v[0]);
    raw = 0;
    for (int i = 1; i < 16; i++) {
        auto tdiff = std::abs(f - v[i]);
        if (tdiff < diff) {
            diff = tdiff;
            raw = i;
        }
    }
}

namespace std {

inline bool isfinite(nf4 h) { return true; }
inline nf4 abs(nf4 h)       { return std::abs(float(h)); }

inline nf4 real(nf4 h)      { return h; }
inline float imag(nf4 h)    { return 0.0f; }

template <>
class numeric_limits<nf4> {
public:
    static nf4 min()               { return nf4(0x8, false); }
    static nf4 lowest()            { return nf4(0x0, false); }
    static nf4 max()               { return nf4(0xF, false); }
    static nf4 epsilon()           { return nf4(0x8, false); }  /* sort of representative value */
    static nf4 round_error()       { return nf4(0.5); }
    static nf4 infinity()          { return nf4(0x0, false); }
    static nf4 quiet_NaN()         { return nf4(0x0, false); }
    static nf4 signaling_NaN()     { return nf4(0x0, false); }
    static nf4 denorm_min()        { return nf4(0x8, false); }
};

} /* namespace std */

#endif
