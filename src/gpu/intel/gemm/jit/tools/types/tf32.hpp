/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_TYPES_TF32_HPP
#define GEMMSTONE_TOOLS_TYPES_TF32_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace tf32_impl {

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

// tf32: seeeeeee'emmmmmmm'mmm00000'00000000

struct tf32 {
    float val;

    tf32(uint32_t raw, bool, bool) : val(tf32_impl::raw_to_float(raw)) {}

    tf32() {}
    inline tf32(float f);
    tf32(double d) : tf32(float(d)) {}
    template <typename T> tf32(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr) : tf32(float(i)) {}

    inline operator float() const { return val; }

    tf32 operator+() const                                     { return *this; }
    tf32 operator-() const                                     { tf32 h = *this; h.val = -h.val; return h; }

    tf32 operator++()                                          { return (*this = *this + 1); }
    tf32 operator++(int)                                       { tf32 h = *this; ++*this; return h; }
    tf32 operator--()                                          { return (*this = *this - 1); }
    tf32 operator--(int)                                       { tf32 h = *this; --*this; return h; }

    friend float operator+(const tf32 &h1, const tf32 &h2)      { return float(h1) + float(h2); }
    friend float operator-(const tf32 &h1, const tf32 &h2)      { return float(h1) - float(h2); }
    friend float operator*(const tf32 &h1, const tf32 &h2)      { return float(h1) * float(h2); }
    friend float operator/(const tf32 &h1, const tf32 &h2)      { return float(h1) / float(h2); }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const tf32 &h, const T &o)           { return float(h) + float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const tf32 &h, const T &o)           { return float(h) - float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const tf32 &h, const T &o)           { return float(h) * float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const tf32 &h, const T &o)           { return float(h) / float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const T &o, const tf32 &h)           { return float(o) + float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const T &o, const tf32 &h)           { return float(o) - float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const T &o, const tf32 &h)           { return float(o) * float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const T &o, const tf32 &h)           { return float(o) / float(h); }

    template <typename T>
    friend typename std::enable_if<tf32_impl::is_float_double<T>::value, T>::type
    operator+(const tf32 &h, const T &o)           { return float(h) + o; }
    template <typename T>
    friend typename std::enable_if<tf32_impl::is_float_double<T>::value, T>::type
    operator-(const tf32 &h, const T &o)           { return float(h) - o; }
    template <typename T>
    friend typename std::enable_if<tf32_impl::is_float_double<T>::value, T>::type
    operator*(const tf32 &h, const T &o)           { return float(h) * o; }
    template <typename T>
    friend typename std::enable_if<tf32_impl::is_float_double<T>::value, T>::type
    operator/(const tf32 &h, const T &o)           { return float(h) / o; }
    template <typename T>
    friend typename std::enable_if<tf32_impl::is_float_double<T>::value, T>::type
    operator+(const T &o, const tf32 &h)           { return o + float(h); }
    template <typename T>
    friend typename std::enable_if<tf32_impl::is_float_double<T>::value, T>::type
    operator-(const T &o, const tf32 &h)           { return o - float(h); }
    template <typename T>
    friend typename std::enable_if<tf32_impl::is_float_double<T>::value, T>::type
    operator*(const T &o, const tf32 &h)           { return o * float(h); }
    template <typename T>
    friend typename std::enable_if<tf32_impl::is_float_double<T>::value, T>::type
    operator/(const T &o, const tf32 &h)           { return o / float(h); }

    template <typename T>
    tf32 operator+=(const T &o)                    { return *this = tf32(*this + o); }
    template <typename T>
    tf32 operator-=(const T &o)                    { return *this = tf32(*this - o); }
    template <typename T>
    tf32 operator*=(const T &o)                    { return *this = tf32(*this * o); }
    template <typename T>
    tf32 operator/=(const T &o)                    { return *this = tf32(*this / o); }
};

tf32::tf32(float f)
{
    uint32_t raw = tf32_impl::float_to_raw(f);

    if (std::isfinite(f)) {
        if ((raw & 0x3FFF) != 0x1000)
            raw += 0x1000;
    }

    raw &= ~0x1FFF;
    val = tf32_impl::raw_to_float(raw);
}


namespace std {

inline bool isfinite(tf32 h)
{
    return std::isfinite(h.val);
}

inline tf32 abs(tf32 h)
{
    tf32 a = h;
    a.val = std::abs(a.val);
    return a;
}

inline tf32 real(tf32 h)
{
    return h;
}

inline float imag(tf32 h)
{
    return 0.0f;
}

template <>
class numeric_limits<tf32> {
public:
    static tf32 min()               { return tf32(0x00800000, false, false); }
    static tf32 lowest()            { return tf32(0xFBFFE000, false, false); }
    static tf32 max()               { return tf32(0x7F7FE000, false, false); }
    static tf32 epsilon()           { return tf32(0x3A800000, false, false); }
    static tf32 round_error()       { return tf32(0x3F000000, false, false); }     // 0.5ulp
    static tf32 infinity()          { return tf32(0x7F800000, false, false); }
    static tf32 quiet_NaN()         { return tf32(0x7FC00000, false, false); }
    static tf32 signaling_NaN()     { return tf32(0x7FF00000, false, false); }
    static tf32 denorm_min()        { return tf32(0x00002000, false, false); }
};

} /* namespace std */

#endif
