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

#ifndef GEMMSTONE_TOOLS_TYPES_HFLOAT8_HPP
#define GEMMSTONE_TOOLS_TYPES_HFLOAT8_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace hf8_impl {

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

// hf8: seeeemmm

struct hfloat8 {
    uint8_t raw;

    hfloat8(int raw_, bool) : raw(raw_) {}

    hfloat8() {}
    inline hfloat8(float f);
    hfloat8(double d) : hfloat8(float(d)) {}
    template <typename T> hfloat8(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr) : hfloat8(float(i)) {}

    inline operator float() const;

    hfloat8 operator+() const                                       { return *this; }
    hfloat8 operator-() const                                       { hfloat8 h = *this; h.raw ^= 0x80; return h; }

    hfloat8 operator++()                                            { return (*this = *this + 1); }
    hfloat8 operator++(int)                                         { hfloat8 h = *this; ++*this; return h; }
    hfloat8 operator--()                                            { return (*this = *this - 1); }
    hfloat8 operator--(int)                                         { hfloat8 h = *this; --*this; return h; }

    friend float operator+(const hfloat8 &h1, const hfloat8 &h2)    { return float(h1) + float(h2); }
    friend float operator-(const hfloat8 &h1, const hfloat8 &h2)    { return float(h1) - float(h2); }
    friend float operator*(const hfloat8 &h1, const hfloat8 &h2)    { return float(h1) * float(h2); }
    friend float operator/(const hfloat8 &h1, const hfloat8 &h2)    { return float(h1) / float(h2); }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const hfloat8 &h, const T &o)           { return float(h) + float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const hfloat8 &h, const T &o)           { return float(h) - float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const hfloat8 &h, const T &o)           { return float(h) * float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const hfloat8 &h, const T &o)           { return float(h) / float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const T &o, const hfloat8 &h)           { return float(o) + float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const T &o, const hfloat8 &h)           { return float(o) - float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const T &o, const hfloat8 &h)           { return float(o) * float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const T &o, const hfloat8 &h)           { return float(o) / float(h); }

    template <typename T>
    friend typename std::enable_if<hf8_impl::is_float_double<T>::value, T>::type
    operator+(const hfloat8 &h, const T &o)           { return float(h) + o; }
    template <typename T>
    friend typename std::enable_if<hf8_impl::is_float_double<T>::value, T>::type
    operator-(const hfloat8 &h, const T &o)           { return float(h) - o; }
    template <typename T>
    friend typename std::enable_if<hf8_impl::is_float_double<T>::value, T>::type
    operator*(const hfloat8 &h, const T &o)           { return float(h) * o; }
    template <typename T>
    friend typename std::enable_if<hf8_impl::is_float_double<T>::value, T>::type
    operator/(const hfloat8 &h, const T &o)           { return float(h) / o; }
    template <typename T>
    friend typename std::enable_if<hf8_impl::is_float_double<T>::value, T>::type
    operator+(const T &o, const hfloat8 &h)           { return o + float(h); }
    template <typename T>
    friend typename std::enable_if<hf8_impl::is_float_double<T>::value, T>::type
    operator-(const T &o, const hfloat8 &h)           { return o - float(h); }
    template <typename T>
    friend typename std::enable_if<hf8_impl::is_float_double<T>::value, T>::type
    operator*(const T &o, const hfloat8 &h)           { return o * float(h); }
    template <typename T>
    friend typename std::enable_if<hf8_impl::is_float_double<T>::value, T>::type
    operator/(const T &o, const hfloat8 &h)           { return o / float(h); }

    template <typename T>
    hfloat8 operator+=(const T &o)                    { return *this = hfloat8(*this + o); }
    template <typename T>
    hfloat8 operator-=(const T &o)                    { return *this = hfloat8(*this - o); }
    template <typename T>
    hfloat8 operator*=(const T &o)                    { return *this = hfloat8(*this * o); }
    template <typename T>
    hfloat8 operator/=(const T &o)                    { return *this = hfloat8(*this / o); }
};

inline hfloat8::hfloat8(float f)
{
    uint32_t i = hf8_impl::float_to_raw(f);
    uint32_t s = i >> 31;
    uint32_t e = (i >> 23) & 0xFF;
    uint32_t m = i & 0x7FFFFF;

    uint32_t ss = s;
    uint32_t mm = m >> 20;
    uint32_t r = m & 0xFFFFF;
    uint32_t ee = 0;
    int32_t eee = (e - 127) + 7;

    if (e == 0) {
        // Denormal/zero floats all become zero.
        ee = 0;
        mm = 0;
    } else if (eee > 0xF || std::abs(f) > 488.f) {
        // Inf/NaN/overflow.
        ee = 0xF;
        mm = 0x7;
    } else if (eee > 0) {
        // Normal range. Perform round to even on mantissa.
        ee = eee;
        if (r > (0x80000 - (mm & 1))) {
            // Round up.
            mm++;
            if (mm == 8) {
                // Rounds up to next dyad (or inf).
                mm = 0;
                ee++;
            }
        }
    } else {
        // Underflow. Scale the input float, converting it
        //  into an equivalent denormal.
        float ff = f * hf8_impl::raw_to_float(0x00000200);
        uint32_t ii = hf8_impl::float_to_raw(ff);
        ee = 0;
        mm = ii;
    }

    raw = (ss << 7) | (ee << 3) | mm;
}

inline hfloat8::operator float() const
{
    uint32_t ss = raw >> 7;
    uint32_t ee = (raw >> 3) & 0xF;
    uint32_t mm = raw & 0x7;

    uint32_t s = ss;
    uint32_t eee = ee - 7 + 127;
    uint32_t m = mm << 20;
    uint32_t e;

    if (ee == 0) {
        // hfloat8 denormal -> float normal
        return (ss ? -1 : 1) * std::scalbn((float) mm, -9);
    } else if (ee == 0xF && mm == 0x7) {
        // Inf/NaN
        return std::numeric_limits<float>::quiet_NaN();
    } else
        e = eee;

    uint32_t f = (s << 31) | (e << 23) | m;

    return hf8_impl::raw_to_float(f);
}


namespace std {

inline bool isfinite(hfloat8 h)
{
    return (~h.raw & 0x7F);
}

inline hfloat8 abs(hfloat8 h)
{
    hfloat8 a = h;
    a.raw &= ~0x80;
    return a;
}

inline hfloat8 real(hfloat8 h)
{
    return h;
}

inline float imag(hfloat8 h)
{
    return 0.0f;
}

template <>
class numeric_limits<hfloat8> {
public:
    static hfloat8 min()               { return hfloat8(0x08, false); }
    static hfloat8 lowest()            { return hfloat8(0xFE, false); }
    static hfloat8 max()               { return hfloat8(0x7E, false); }
    static hfloat8 epsilon()           { return hfloat8(0x20, false); }
    static hfloat8 round_error()       { return hfloat8(0x30, false); }     // 0.5ulp
    static hfloat8 infinity()          { return hfloat8(0x7E, false); }
    static hfloat8 quiet_NaN()         { return hfloat8(0x7E, false); }
    static hfloat8 signaling_NaN()     { return hfloat8(0x7E, false); }
    static hfloat8 denorm_min()        { return hfloat8(0x01, false); }
};

} /* namespace std */

#endif
