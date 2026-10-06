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

#ifndef GEMMSTONE_TOOLS_TYPES_FP4_HPP
#define GEMMSTONE_TOOLS_TYPES_FP4_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

#include "fp16.hpp"
#include "bfloat16.hpp"
#include "bfloat8.hpp"
#include "int4.hpp"

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace f4_impl {

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

// f4_e2m1: seem

struct f4_e2m1 {
    uint8_t raw;

    f4_e2m1(int raw_, bool) : raw(raw_) {}

    f4_e2m1() {}
    inline f4_e2m1(float f);
    f4_e2m1(double d) : f4_e2m1(float(d)) {}
    template <typename T> f4_e2m1(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr) : f4_e2m1(float(i)) {}

    inline operator float() const;

    f4_e2m1 operator+() const                                       { return *this; }
    f4_e2m1 operator-() const                                       { f4_e2m1 h = *this; h.raw ^= 0x80; return h; }

    f4_e2m1 operator++()                                            { return (*this = *this + 1); }
    f4_e2m1 operator++(int)                                         { f4_e2m1 h = *this; ++*this; return h; }
    f4_e2m1 operator--()                                            { return (*this = *this - 1); }
    f4_e2m1 operator--(int)                                         { f4_e2m1 h = *this; --*this; return h; }

    friend float operator+(const f4_e2m1 &h1, const f4_e2m1 &h2)    { return float(h1) + float(h2); }
    friend float operator-(const f4_e2m1 &h1, const f4_e2m1 &h2)    { return float(h1) - float(h2); }
    friend float operator*(const f4_e2m1 &h1, const f4_e2m1 &h2)    { return float(h1) * float(h2); }
    friend float operator/(const f4_e2m1 &h1, const f4_e2m1 &h2)    { return float(h1) / float(h2); }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const f4_e2m1 &h, const T &o)           { return float(h) + float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const f4_e2m1 &h, const T &o)           { return float(h) - float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const f4_e2m1 &h, const T &o)           { return float(h) * float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const f4_e2m1 &h, const T &o)           { return float(h) / float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const T &o, const f4_e2m1 &h)           { return float(o) + float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const T &o, const f4_e2m1 &h)           { return float(o) - float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const T &o, const f4_e2m1 &h)           { return float(o) * float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const T &o, const f4_e2m1 &h)           { return float(o) / float(h); }

    template <typename T>
    friend typename std::enable_if<f4_impl::is_float_double<T>::value, T>::type
    operator+(const f4_e2m1 &h, const T &o)           { return float(h) + o; }
    template <typename T>
    friend typename std::enable_if<f4_impl::is_float_double<T>::value, T>::type
    operator-(const f4_e2m1 &h, const T &o)           { return float(h) - o; }
    template <typename T>
    friend typename std::enable_if<f4_impl::is_float_double<T>::value, T>::type
    operator*(const f4_e2m1 &h, const T &o)           { return float(h) * o; }
    template <typename T>
    friend typename std::enable_if<f4_impl::is_float_double<T>::value, T>::type
    operator/(const f4_e2m1 &h, const T &o)           { return float(h) / o; }
    template <typename T>
    friend typename std::enable_if<f4_impl::is_float_double<T>::value, T>::type
    operator+(const T &o, const f4_e2m1 &h)           { return o + float(h); }
    template <typename T>
    friend typename std::enable_if<f4_impl::is_float_double<T>::value, T>::type
    operator-(const T &o, const f4_e2m1 &h)           { return o - float(h); }
    template <typename T>
    friend typename std::enable_if<f4_impl::is_float_double<T>::value, T>::type
    operator*(const T &o, const f4_e2m1 &h)           { return o * float(h); }
    template <typename T>
    friend typename std::enable_if<f4_impl::is_float_double<T>::value, T>::type
    operator/(const T &o, const f4_e2m1 &h)           { return o / float(h); }

    template <typename T>
    f4_e2m1 operator+=(const T &o)                    { return *this = f4_e2m1(*this + o); }
    template <typename T>
    f4_e2m1 operator-=(const T &o)                    { return *this = f4_e2m1(*this - o); }
    template <typename T>
    f4_e2m1 operator*=(const T &o)                    { return *this = f4_e2m1(*this * o); }
    template <typename T>
    f4_e2m1 operator/=(const T &o)                    { return *this = f4_e2m1(*this / o); }
};


inline f4_e2m1::f4_e2m1(float f)
{
    float thresh[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

    uint32_t i = f4_impl::float_to_raw(f);
    uint8_t s = i >> 31;

    float af = std::abs(f);

    for (raw = 0; raw < 7; raw++) {
        if (af < thresh[raw])
            break;
        if (af == thresh[raw] && !(raw & 1))
            break;
    }
    raw |= (s << 3);
}

inline f4_e2m1::operator float() const
{
    float vals[16] = { 0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                      -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
    return vals[raw & 0xF];
}


namespace std {

    inline bool isfinite(f4_e2m1 h)
    {
        return (true);
    }

    inline f4_e2m1 abs(f4_e2m1 h)
    {
        f4_e2m1 a = h;
        a.raw &= ~0x8;
        return a;
    }

    inline f4_e2m1 real(f4_e2m1 h)
    {
        return h;
    }

    inline float imag(f4_e2m1 h)
    {
        return 0.0f;
    }

    template <>
    class numeric_limits<f4_e2m1> {
    public:
        static f4_e2m1 min()               { return f4_e2m1(0x2, false); }
        static f4_e2m1 lowest()            { return f4_e2m1(0xF, false); }
        static f4_e2m1 max()               { return f4_e2m1(0x7, false); }
        static f4_e2m1 epsilon()           { return f4_e2m1(0x1, false); } // 0.5
        static f4_e2m1 round_error()       { return f4_e2m1(0x1, false); } // 0.5
        static f4_e2m1 infinity()          { return f4_e2m1(0x7, false); }
        static f4_e2m1 quiet_NaN()         { return f4_e2m1(0x7, false); }
        static f4_e2m1 signaling_NaN()     { return f4_e2m1(0x7, false); }
        static f4_e2m1 denorm_min()        { return f4_e2m1(0x1, false); } // 0.5
    };

} /* namespace std */

#endif
