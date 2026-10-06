/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_TYPES_BFLOAT8_HPP
#define GEMMSTONE_TOOLS_TYPES_BFLOAT8_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace bf8_impl {

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

// bf8: seeeeemm

struct bfloat8 {
    uint8_t raw;

    bfloat8(int raw_, bool) : raw(raw_) {}

    bfloat8() {}
    inline bfloat8(float f);
    bfloat8(double d) : bfloat8(float(d)) {}
    template <typename T> bfloat8(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr) : bfloat8(float(i)) {}

    inline operator float() const;

    bfloat8 operator+() const                                     { return *this; }
    bfloat8 operator-() const                                     { bfloat8 h = *this; h.raw ^= 0x80; return h; }

    bfloat8 operator++()                                          { return (*this = *this + 1); }
    bfloat8 operator++(int)                                       { bfloat8 h = *this; ++*this; return h; }
    bfloat8 operator--()                                          { return (*this = *this - 1); }
    bfloat8 operator--(int)                                       { bfloat8 h = *this; --*this; return h; }

    friend float operator+(const bfloat8 &h1, const bfloat8 &h2)      { return float(h1) + float(h2); }
    friend float operator-(const bfloat8 &h1, const bfloat8 &h2)      { return float(h1) - float(h2); }
    friend float operator*(const bfloat8 &h1, const bfloat8 &h2)      { return float(h1) * float(h2); }
    friend float operator/(const bfloat8 &h1, const bfloat8 &h2)      { return float(h1) / float(h2); }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const bfloat8 &h, const T &o)           { return float(h) + float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const bfloat8 &h, const T &o)           { return float(h) - float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const bfloat8 &h, const T &o)           { return float(h) * float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const bfloat8 &h, const T &o)           { return float(h) / float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const T &o, const bfloat8 &h)           { return float(o) + float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const T &o, const bfloat8 &h)           { return float(o) - float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const T &o, const bfloat8 &h)           { return float(o) * float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const T &o, const bfloat8 &h)           { return float(o) / float(h); }

    template <typename T>
    friend typename std::enable_if<bf8_impl::is_float_double<T>::value, T>::type
    operator+(const bfloat8 &h, const T &o)           { return float(h) + o; }
    template <typename T>
    friend typename std::enable_if<bf8_impl::is_float_double<T>::value, T>::type
    operator-(const bfloat8 &h, const T &o)           { return float(h) - o; }
    template <typename T>
    friend typename std::enable_if<bf8_impl::is_float_double<T>::value, T>::type
    operator*(const bfloat8 &h, const T &o)           { return float(h) * o; }
    template <typename T>
    friend typename std::enable_if<bf8_impl::is_float_double<T>::value, T>::type
    operator/(const bfloat8 &h, const T &o)           { return float(h) / o; }
    template <typename T>
    friend typename std::enable_if<bf8_impl::is_float_double<T>::value, T>::type
    operator+(const T &o, const bfloat8 &h)           { return o + float(h); }
    template <typename T>
    friend typename std::enable_if<bf8_impl::is_float_double<T>::value, T>::type
    operator-(const T &o, const bfloat8 &h)           { return o - float(h); }
    template <typename T>
    friend typename std::enable_if<bf8_impl::is_float_double<T>::value, T>::type
    operator*(const T &o, const bfloat8 &h)           { return o * float(h); }
    template <typename T>
    friend typename std::enable_if<bf8_impl::is_float_double<T>::value, T>::type
    operator/(const T &o, const bfloat8 &h)           { return o / float(h); }

    template <typename T>
    bfloat8 operator+=(const T &o)                    { return *this = bfloat8(*this + o); }
    template <typename T>
    bfloat8 operator-=(const T &o)                    { return *this = bfloat8(*this - o); }
    template <typename T>
    bfloat8 operator*=(const T &o)                    { return *this = bfloat8(*this * o); }
    template <typename T>
    bfloat8 operator/=(const T &o)                    { return *this = bfloat8(*this / o); }
};

inline bfloat8::bfloat8(float f)
{
    uint32_t i = bf8_impl::float_to_raw(f);
    uint32_t s = i >> 31;
    uint32_t e = (i >> 23) & 0xFF;
    uint32_t m = i & 0x7FFFFF;

    uint32_t ss = s;
    uint32_t mm = m >> 21;
    uint32_t r = m & 0x1FFFFF;
    uint32_t ee = 0;
    int32_t eee = (e - 127) + 15;

    if (e == 0) {
        // Denormal/zero floats all become zero.
        ee = 0;
        mm = 0;
    } else if (e == 0xFF) {
        // Preserve inf/nan.
        ee = 0x1F;
        if (m != 0 && mm == 0)
            mm = 1;
    } else if (eee > 0 && eee < 0x1F) {
        // Normal range. Perform round to even on mantissa.
        ee = eee;
        if (r > (0x100000 - (mm & 1))) {
            // Round up.
            mm++;
            if (mm == 4) {
                // Rounds up to next dyad (or inf).
                mm = 0;
                ee++;
            }
        }
    } else if (eee >= 0x1F) {
        // Overflow.
        ee = 0x1F;
        mm = 0;
    } else {
        // Underflow. Scale the input float, converting it
        //  into an equivalent denormal.
        float ff = f * bf8_impl::raw_to_float(0x00010000);
        uint32_t ii = bf8_impl::float_to_raw(ff);
        ee = 0;
        mm = ii;
    }

    raw = (ss << 7) | (ee << 2) | mm;
}

inline bfloat8::operator float() const
{
    uint32_t ss = raw >> 7;
    uint32_t ee = (raw >> 2) & 0x1F;
    uint32_t mm = raw & 0x3;

    uint32_t s = ss;
    uint32_t eee = ee - 15 + 127;
    uint32_t m = mm << 21;
    uint32_t e;

    if (ee == 0) {
        if (mm == 0)
            e = 0;
        else {
            // bfloat8 denormal -> float normal
            return (ss ? -1 : 1) * std::scalbn((float) mm, -16);
        }
    } else if (ee == 0x1F) {
        // inf/nan
        e = 0xFF;
    } else
        e = eee;

    uint32_t f = (s << 31) | (e << 23) | m;

    return bf8_impl::raw_to_float(f);
}


namespace std {

inline bool isfinite(bfloat8 h)
{
    return (~h.raw & 0x7C);
}

inline bfloat8 abs(bfloat8 h)
{
    bfloat8 a = h;
    a.raw &= ~0x80;
    return a;
}

inline bfloat8 real(bfloat8 h)
{
    return h;
}

inline float imag(bfloat8 h)
{
    return 0.0f;
}

template <>
class numeric_limits<bfloat8> {
public:
    static bfloat8 min()               { return bfloat8(0x04, false); }
    static bfloat8 lowest()            { return bfloat8(0xFB, false); }
    static bfloat8 max()               { return bfloat8(0x7B, false); }
    static bfloat8 epsilon()           { return bfloat8(0x34, false); }
    static bfloat8 round_error()       { return bfloat8(0x38, false); }     // 0.5ulp
    static bfloat8 infinity()          { return bfloat8(0x7C, false); }
    static bfloat8 quiet_NaN()         { return bfloat8(0x7D, false); }
    static bfloat8 signaling_NaN()     { return bfloat8(0x7E, false); }
    static bfloat8 denorm_min()        { return bfloat8(0x01, false); }
};

} /* namespace std */

#endif
