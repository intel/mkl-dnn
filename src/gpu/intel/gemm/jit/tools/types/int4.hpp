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

#ifndef GEMMSTONE_TOOLS_TYPES_INT4_HPP
#define GEMMSTONE_TOOLS_TYPES_INT4_HPP

#include <cmath>
#include <cstdint>
#include <type_traits>
#include <limits>

#include "fp16.hpp"
#include "bfloat16.hpp"
#include "bfloat8.hpp"
#include "hfloat8.hpp"

namespace int4_impl {

template <bool is_signed>
struct int4_base {
    int8_t val;

    int4_base() {}
    int4_base(int8_t i) : val(i & 0xF) {}
    template <typename T> int4_base(T i) : int4_base(int8_t(i)) {}

    inline operator int() const
    {
        if constexpr (is_signed)
            return int8_t(val << 4) >> 4;
        else
            return val;
    }

    inline operator const half() const                              { return int(*this); }
    inline operator const bfloat16() const                          { return int(*this); }
    inline operator const hfloat8() const                           { return int(*this); }
    inline operator const bfloat8() const                           { return int(*this); }

    int4_base operator+() const                                     { return *this; }
    int4_base operator-() const                                     { int4_base h = *this; h.val = -h.val; return h; }

    int4_base operator++()                                          { return (*this = *this + 1); }
    int4_base operator++(int)                                       { int4_base h = *this; ++*this; return h; }
    int4_base operator--()                                          { return (*this = *this - 1); }
    int4_base operator--(int)                                       { int4_base h = *this; --*this; return h; }

    using oint4 = int4_base<!is_signed>;

    friend int operator+(const int4_base &x1, const int4_base &x2)  { return int(x1) + int(x2); }
    friend int operator-(const int4_base &x1, const int4_base &x2)  { return int(x1) - int(x2); }
    friend int operator*(const int4_base &x1, const int4_base &x2)  { return int(x1) * int(x2); }
    friend int operator/(const int4_base &x1, const int4_base &x2)  { return int(x1) / int(x2); }

    friend int operator+(const int4_base &x1, const oint4 &x2)      { return int(x1) + int(x2); }
    friend int operator-(const int4_base &x1, const oint4 &x2)      { return int(x1) - int(x2); }
    friend int operator*(const int4_base &x1, const oint4 &x2)      { return int(x1) * int(x2); }
    friend int operator/(const int4_base &x1, const oint4 &x2)      { return int(x1) / int(x2); }

    template <typename T>
    friend auto operator+(const int4_base &x, const T &o)            { return int(x) + o; }
    template <typename T>
    friend auto operator-(const int4_base &x, const T &o)            { return int(x) - o; }
    template <typename T>
    friend auto operator*(const int4_base &x, const T &o)            { return int(x) * o; }
    template <typename T>
    friend auto operator/(const int4_base &x, const T &o)            { return int(x) / o; }
    template <typename T>
    friend auto operator+(const T &o, const int4_base &x)            { return o + int(x); }
    template <typename T>
    friend auto operator-(const T &o, const int4_base &x)            { return o - int(x); }
    template <typename T>
    friend auto operator*(const T &o, const int4_base &x)            { return o * int(x); }
    template <typename T>
    friend auto operator/(const T &o, const int4_base &x)            { return o / int(x); }

    template <typename T>
    int4_base operator+=(const T &o)                    { return *this = int4_base(*this + o); }
    template <typename T>
    int4_base operator-=(const T &o)                    { return *this = int4_base(*this - o); }
    template <typename T>
    int4_base operator*=(const T &o)                    { return *this = int4_base(*this * o); }
    template <typename T>
    int4_base operator/=(const T &o)                    { return *this = int4_base(*this / o); }
};

} /* namespace int4_impl */

namespace std {

template <bool is_signed> bool isfinite(int4_impl::int4_base<is_signed>)                          { return true; }
template <bool is_signed> int4_impl::int4_base<is_signed>  abs(int4_impl::int4_base<is_signed> x) { return abs(int(x)); }
template <bool is_signed> int4_impl::int4_base<is_signed> real(int4_impl::int4_base<is_signed> x) { return x; }
template <bool is_signed> int4_impl::int4_base<is_signed> imag(int4_impl::int4_base<is_signed> x) { return 0; }

} /* namespace std */

using  int4 = int4_impl::int4_base<true>;
using uint4 = int4_impl::int4_base<false>;

#endif
