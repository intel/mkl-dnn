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

#ifndef GEMMSTONE_TOOLS_MULTIOPT_UTILS_HPP
#define GEMMSTONE_TOOLS_MULTIOPT_UTILS_HPP

#include <cmath>
#include <cstdint>
#include <utility>

#include <random>

namespace multiopt {

template <typename V> using ElementType = std::remove_reference_t<decltype(std::declval<V>()[0])>;

inline uint32_t rand32()
{
    static std::independent_bits_engine<std::mt19937, 32, uint32_t> generator(1);
    return generator();
}

inline bool randBool(float prob)
{
    if (prob >= 1.0f) return true;
    return (rand32() < std::ldexp(prob, 32));
}

// Random integer in [min, max] (inclusive) -- T should be <= 32 bits.
template <typename T>
inline T randInt(T min, T max)
{
    uint32_t r = rand32();
    return (r % (max - min + 1)) + min;     /* not totally uniform, of course */
}

// Random integer in [0, n).
template <typename T>
inline T randInt(T n)
{
    return randInt<T>(0, n - 1);
}

// Random float in [min, max).
inline float randFloat(float min, float max)
{
    float r = std::ldexp(rand32(), -32);    // [0, 1)
    return min + r * (max - min);
}

} /* namespace multiopt */

#endif
