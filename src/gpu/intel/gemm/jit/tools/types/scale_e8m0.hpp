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

#ifndef GEMMSTONE_TOOLS_TYPES_SCALE_E8M0_HPP
#define GEMMSTONE_TOOLS_TYPES_SCALE_E8M0_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

namespace fp8_e8m0_impl {

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



struct scale_e8m0 {
    uint8_t raw;

    scale_e8m0(int raw_, bool) : raw(raw_) {}

    scale_e8m0() {}
    inline scale_e8m0(float f);
    scale_e8m0(double d) : scale_e8m0(float(d)) {}
    template <typename T> scale_e8m0(T i, typename std::enable_if<std::is_integral<T>::value>::type *_ = nullptr) : scale_e8m0(float(i)) {}

    inline operator float() const;

    scale_e8m0 operator+() const                                       { return *this; }
    scale_e8m0 operator-() const                                       { scale_e8m0 h = *this; h.raw ^= 0x80; return h; }

    scale_e8m0 operator++()                                            { return (*this = *this + 1); }
    scale_e8m0 operator++(int)                                         { scale_e8m0 h = *this; ++*this; return h; }
    scale_e8m0 operator--()                                            { return (*this = *this - 1); }
    scale_e8m0 operator--(int)                                         { scale_e8m0 h = *this; --*this; return h; }

    friend float operator+(const scale_e8m0 &h1, const scale_e8m0 &h2)    { return float(h1) + float(h2); }
    friend float operator-(const scale_e8m0 &h1, const scale_e8m0 &h2)    { return float(h1) - float(h2); }
    friend float operator*(const scale_e8m0 &h1, const scale_e8m0 &h2)    { return float(h1) * float(h2); }
    friend float operator/(const scale_e8m0 &h1, const scale_e8m0 &h2)    { return float(h1) / float(h2); }

    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const scale_e8m0 &h, const T &o)           { return float(h) + float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const scale_e8m0 &h, const T &o)           { return float(h) - float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const scale_e8m0 &h, const T &o)           { return float(h) * float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const scale_e8m0 &h, const T &o)           { return float(h) / float(o); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator+(const T &o, const scale_e8m0 &h)           { return float(o) + float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator-(const T &o, const scale_e8m0 &h)           { return float(o) - float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator*(const T &o, const scale_e8m0 &h)           { return float(o) * float(h); }
    template <typename T>
    friend typename std::enable_if<std::is_integral<T>::value, float>::type
    operator/(const T &o, const scale_e8m0 &h)           { return float(o) / float(h); }

    template <typename T>
    friend typename std::enable_if<fp8_e8m0_impl::is_float_double<T>::value, T>::type
    operator+(const scale_e8m0 &h, const T &o)           { return float(h) + o; }
    template <typename T>
    friend typename std::enable_if<fp8_e8m0_impl::is_float_double<T>::value, T>::type
    operator-(const scale_e8m0 &h, const T &o)           { return float(h) - o; }
    template <typename T>
    friend typename std::enable_if<fp8_e8m0_impl::is_float_double<T>::value, T>::type
    operator*(const scale_e8m0 &h, const T &o)           { return float(h) * o; }
    template <typename T>
    friend typename std::enable_if<fp8_e8m0_impl::is_float_double<T>::value, T>::type
    operator/(const scale_e8m0 &h, const T &o)           { return float(h) / o; }
    template <typename T>
    friend typename std::enable_if<fp8_e8m0_impl::is_float_double<T>::value, T>::type
    operator+(const T &o, const scale_e8m0 &h)           { return o + float(h); }
    template <typename T>
    friend typename std::enable_if<fp8_e8m0_impl::is_float_double<T>::value, T>::type
    operator-(const T &o, const scale_e8m0 &h)           { return o - float(h); }
    template <typename T>
    friend typename std::enable_if<fp8_e8m0_impl::is_float_double<T>::value, T>::type
    operator*(const T &o, const scale_e8m0 &h)           { return o * float(h); }
    template <typename T>
    friend typename std::enable_if<fp8_e8m0_impl::is_float_double<T>::value, T>::type
    operator/(const T &o, const scale_e8m0 &h)           { return o / float(h); }

    template <typename T>
    scale_e8m0 operator+=(const T &o)                    { return *this = scale_e8m0(*this + o); }
    template <typename T>
    scale_e8m0 operator-=(const T &o)                    { return *this = scale_e8m0(*this - o); }
    template <typename T>
    scale_e8m0 operator*=(const T &o)                    { return *this = scale_e8m0(*this * o); }
    template <typename T>
    scale_e8m0 operator/=(const T &o)                    { return *this = scale_e8m0(*this / o); }
};

inline scale_e8m0::scale_e8m0(float f)
{
    uint32_t i = fp8_e8m0_impl::float_to_raw(f);
    uint32_t e = (i >> 23) & 0xFF;

    raw = e;

}

inline scale_e8m0::operator float() const
{
    uint32_t ee = raw  & 0xFF;
    uint32_t e;

    if (ee == 0xFF) {
        // Inf/NaN
        return std::numeric_limits<float>::quiet_NaN();
    } else
        e = ee;

    uint32_t f =  (e << 23) ;

    return fp8_e8m0_impl::raw_to_float(f);
}


namespace std {

    inline bool isfinite(scale_e8m0 h)
    {
        return (true);
    }

    inline scale_e8m0 abs(scale_e8m0 h)
    {
        return h;
    }

    inline scale_e8m0 real(scale_e8m0 h)
    {
        return h;
    }

    inline float imag(scale_e8m0 h)
    {
        return 0.0f;
    }

    template <>
    class numeric_limits<scale_e8m0> {
    public:
        static scale_e8m0 min()               { return scale_e8m0(0x01, false); }
        static scale_e8m0 lowest()            { return scale_e8m0(0x01, false); }
        static scale_e8m0 max()               { return scale_e8m0(0xFE, false); }
        static scale_e8m0 epsilon()           { return scale_e8m0(0x01, false); }
        static scale_e8m0 round_error()       { return scale_e8m0(0x7E, false); }     // 0.5ulp
        static scale_e8m0 infinity()          { return scale_e8m0(0xFE, false); }
        static scale_e8m0 quiet_NaN()         { return scale_e8m0(0xFF, false); }
        static scale_e8m0 signaling_NaN()     { return scale_e8m0(0xFF, false); }
        static scale_e8m0 denorm_min()        { return scale_e8m0(0x01, false); }
    };

} /* namespace std */

#endif
