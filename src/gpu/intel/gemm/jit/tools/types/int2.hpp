/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_TYPES_INT2_HPP
#define GEMMSTONE_TOOLS_TYPES_INT2_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

#include "fp16.hpp"
#include "bfloat16.hpp"
#include "bfloat8.hpp"
#include "hfloat8.hpp"

namespace int2_impl {

template <bool is_signed>
struct int2_base {
    int8_t val;

    int2_base() {}
    int2_base(int8_t i) : val(i & 0x3) {}
    template <typename T> int2_base(T i) : int2_base(int8_t(i)) {}

    inline operator int() const
    {
        if constexpr (is_signed)
            return int8_t(val << 6) >> 6;
        else
            return val;
    }

    inline operator const half() const                              { return int(*this); }
    inline operator const bfloat16() const                          { return int(*this); }
    inline operator const hfloat8() const                           { return int(*this); }
    inline operator const bfloat8() const                           { return int(*this); }

    int2_base operator+() const                                     { return *this; }
    int2_base operator-() const                                     { int2_base h = *this; h.val = -h.val; return h; }

    int2_base operator++()                                          { return (*this = *this + 1); }
    int2_base operator++(int)                                       { int2_base h = *this; ++*this; return h; }
    int2_base operator--()                                          { return (*this = *this - 1); }
    int2_base operator--(int)                                       { int2_base h = *this; --*this; return h; }

    using oint4 = int2_base<!is_signed>;

    friend int operator+(const int2_base &x1, const int2_base &x2)  { return int(x1) + int(x2); }
    friend int operator-(const int2_base &x1, const int2_base &x2)  { return int(x1) - int(x2); }
    friend int operator*(const int2_base &x1, const int2_base &x2)  { return int(x1) * int(x2); }
    friend int operator/(const int2_base &x1, const int2_base &x2)  { return int(x1) / int(x2); }

    friend int operator+(const int2_base &x1, const oint4 &x2)      { return int(x1) + int(x2); }
    friend int operator-(const int2_base &x1, const oint4 &x2)      { return int(x1) - int(x2); }
    friend int operator*(const int2_base &x1, const oint4 &x2)      { return int(x1) * int(x2); }
    friend int operator/(const int2_base &x1, const oint4 &x2)      { return int(x1) / int(x2); }

    template <typename T>
    friend auto operator+(const int2_base &x, const T &o)            { return int(x) + o; }
    template <typename T>
    friend auto operator-(const int2_base &x, const T &o)            { return int(x) - o; }
    template <typename T>
    friend auto operator*(const int2_base &x, const T &o)            { return int(x) * o; }
    template <typename T>
    friend auto operator/(const int2_base &x, const T &o)            { return int(x) / o; }
    template <typename T>
    friend auto operator+(const T &o, const int2_base &x)            { return o + int(x); }
    template <typename T>
    friend auto operator-(const T &o, const int2_base &x)            { return o - int(x); }
    template <typename T>
    friend auto operator*(const T &o, const int2_base &x)            { return o * int(x); }
    template <typename T>
    friend auto operator/(const T &o, const int2_base &x)            { return o / int(x); }

    template <typename T>
    int2_base operator+=(const T &o)                    { return *this = int2_base(*this + o); }
    template <typename T>
    int2_base operator-=(const T &o)                    { return *this = int2_base(*this - o); }
    template <typename T>
    int2_base operator*=(const T &o)                    { return *this = int2_base(*this * o); }
    template <typename T>
    int2_base operator/=(const T &o)                    { return *this = int2_base(*this / o); }
};

} /* namespace int2_impl */

namespace std {

template <bool is_signed> bool isfinite(int2_impl::int2_base<is_signed>)                          { return true; }
template <bool is_signed> int2_impl::int2_base<is_signed>  abs(int2_impl::int2_base<is_signed> x) { return abs(int(x)); }
template <bool is_signed> int2_impl::int2_base<is_signed> real(int2_impl::int2_base<is_signed> x) { return x; }
template <bool is_signed> int2_impl::int2_base<is_signed> imag(int2_impl::int2_base<is_signed> x) { return 0; }

} /* namespace std */

using  int2 = int2_impl::int2_base<true>;
using uint2 = int2_impl::int2_base<false>;

#endif
