/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_TYPES_BFLOAT16_HPP
#define GEMMSTONE_TOOLS_TYPES_BFLOAT16_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace bfloat16_impl {

template <typename T> struct is_float_double         { static constexpr bool value = false; };
template <>           struct is_float_double<float>  { static constexpr bool value = true; };
template <>           struct is_float_double<double> { static constexpr bool value = true; };

union float_raw {
    float f;
    uint32_t i;
};

static inline uint32_t float_to_raw(float f)
{
    float_raw r;
    r.f = f;
    return r.i;
}

static inline float raw_to_float(uint32_t i)
{
    float_raw r;
    r.i = i;
    return r.f;
}

}

// bfloat16: seeeeeee'emmmmmmm

struct bfloat16 {
    uint16_t raw;

    bfloat16(int raw_, bool) : raw(raw_) {}

    bfloat16() {}
    inline bfloat16(float f);
    bfloat16(double d) : bfloat16(float(d)) {}
    template <typename T> bfloat16(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr) : bfloat16(float(i)) {}

    inline operator float() const;

    bfloat16 operator+() const                                     { return *this; }
    bfloat16 operator-() const                                     { bfloat16 h = *this; h.raw ^= 0x8000; return h; }

    bfloat16 operator++()                                          { return (*this = *this + 1); }
    bfloat16 operator++(int)                                       { bfloat16 h = *this; ++*this; return h; }
    bfloat16 operator--()                                          { return (*this = *this - 1); }
    bfloat16 operator--(int)                                       { bfloat16 h = *this; --*this; return h; }

    friend float operator+(const bfloat16 &h1, const bfloat16 &h2)      { return float(h1) + float(h2); }
    friend float operator-(const bfloat16 &h1, const bfloat16 &h2)      { return float(h1) - float(h2); }
    friend float operator*(const bfloat16 &h1, const bfloat16 &h2)      { return float(h1) * float(h2); }
    friend float operator/(const bfloat16 &h1, const bfloat16 &h2)      { return float(h1) / float(h2); }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const bfloat16 &h, const T &o)           { return float(h) + float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const bfloat16 &h, const T &o)           { return float(h) - float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const bfloat16 &h, const T &o)           { return float(h) * float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const bfloat16 &h, const T &o)           { return float(h) / float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const T &o, const bfloat16 &h)           { return float(o) + float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const T &o, const bfloat16 &h)           { return float(o) - float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const T &o, const bfloat16 &h)           { return float(o) * float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const T &o, const bfloat16 &h)           { return float(o) / float(h); }

    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type
    operator+(const bfloat16 &h, const T &o)           { return float(h) + o; }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type
    operator-(const bfloat16 &h, const T &o)           { return float(h) - o; }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type
    operator*(const bfloat16 &h, const T &o)           { return float(h) * o; }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type
    operator/(const bfloat16 &h, const T &o)           { return float(h) / o; }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type
    operator+(const T &o, const bfloat16 &h)           { return o + float(h); }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type
    operator-(const T &o, const bfloat16 &h)           { return o - float(h); }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type
    operator*(const T &o, const bfloat16 &h)           { return o * float(h); }
    template <typename T>
    friend typename std::enable_if<bfloat16_impl::is_float_double<T>::value, T>::type
    operator/(const T &o, const bfloat16 &h)           { return o / float(h); }

    template <typename T>
    bfloat16 operator+=(const T &o)                    { return *this = bfloat16(*this + o); }
    template <typename T>
    bfloat16 operator-=(const T &o)                    { return *this = bfloat16(*this - o); }
    template <typename T>
    bfloat16 operator*=(const T &o)                    { return *this = bfloat16(*this * o); }
    template <typename T>
    bfloat16 operator/=(const T &o)                    { return *this = bfloat16(*this / o); }
};

bfloat16::bfloat16(float f)
{
    uint32_t raw_f = bfloat16_impl::float_to_raw(f);

    if (std::isfinite(f)) {
        if ((raw_f & 0x1FFFF) != 0x08000)
            raw_f += 0x8000;
    }

    raw = raw_f >> 16;
}

inline bfloat16::operator float() const
{
    return bfloat16_impl::raw_to_float(raw << 16);
}


namespace std {

inline bool isfinite(bfloat16 h)
{
    return (~h.raw & 0x7F80);
}

inline bfloat16 abs(bfloat16 h)
{
    bfloat16 a = h;
    a.raw &= ~0x8000;
    return a;
}

inline bfloat16 real(bfloat16 h)
{
    return h;
}

inline float imag(bfloat16 h)
{
    return 0.0f;
}

template <>
class numeric_limits<bfloat16> {
public:
    static bfloat16 min()               { return bfloat16(0x0080, false); }
    static bfloat16 lowest()            { return bfloat16(0xFBFF, false); }
    static bfloat16 max()               { return bfloat16(0x7F7F, false); }
    static bfloat16 epsilon()           { return bfloat16(0x3C00, false); }
    static bfloat16 round_error()       { return bfloat16(0x3F00, false); }     // 0.5ulp
    static bfloat16 infinity()          { return bfloat16(0x7F80, false); }
    static bfloat16 quiet_NaN()         { return bfloat16(0x7FC0, false); }
    static bfloat16 signaling_NaN()     { return bfloat16(0x7FF0, false); }
    static bfloat16 denorm_min()        { return bfloat16(0x0001, false); }
};

} /* namespace std */

#endif
