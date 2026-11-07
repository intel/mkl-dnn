/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#ifndef COMMON_FLOAT8_HPP
#define COMMON_FLOAT8_HPP

#include <cstdint>

#include "common/bfloat16.hpp"
#include "common/float16.hpp"

namespace dnnl {
namespace impl {

struct float8_e8m0_t {
    uint8_t raw_bits_;
    float8_e8m0_t() = default;
    constexpr float8_e8m0_t(uint8_t r, bool) : raw_bits_(r) {}
    float8_e8m0_t(float f) { (*this) = f; }
    float8_e8m0_t DNNL_API &operator=(float f);
    DNNL_API operator float() const;
};
static_assert(sizeof(float8_e8m0_t) == 1, "float8_e8m0_t must be 1 byte");

struct float8_e5m2_t {
    uint8_t raw_bits_;
    float8_e5m2_t() = default;
    constexpr float8_e5m2_t(uint8_t r, bool) : raw_bits_(r) {}
    float8_e5m2_t(float f) { (*this) = f; }
    float8_e5m2_t(float16_t f) { (*this) = f; }
    float8_e5m2_t(bfloat16_t f) { (*this) = f; }

    float8_e5m2_t DNNL_API &operator=(float f);
    float8_e5m2_t DNNL_API &operator=(float16_t f);
    float8_e5m2_t DNNL_API &operator=(bfloat16_t f);

    DNNL_API operator float() const;
    DNNL_API operator float16_t() const;
    DNNL_API operator bfloat16_t() const;
    float8_e5m2_t &convert_fp32_to_fp8e5m2_RNE(float f, bool is_saturating);

    float8_e5m2_t &operator+=(const float a) {
        (*this) = float {*this} + a;
        return *this;
    }
};
static_assert(sizeof(float8_e5m2_t) == 1, "float8_e5m2_t must be 1 byte");

struct float8_e4m3_t {
    uint8_t raw_bits_;
    float8_e4m3_t() = default;
    constexpr float8_e4m3_t(uint8_t r, bool) : raw_bits_(r) {}
    float8_e4m3_t(float f) { (*this) = f; }
    float8_e4m3_t(float16_t f) { (*this) = f; }
    float8_e4m3_t(bfloat16_t f) { (*this) = f; }

    float8_e4m3_t DNNL_API &operator=(float f);
    float8_e4m3_t DNNL_API &operator=(float16_t f);
    float8_e4m3_t DNNL_API &operator=(bfloat16_t f);

    float8_e4m3_t &convert_fp32_to_fp8e4m3_RNE(float f, bool is_saturating);
    DNNL_API operator float() const;
    DNNL_API operator float16_t() const;
    DNNL_API operator bfloat16_t() const;

    float8_e4m3_t &operator+=(const float a) {
        (*this) = float {*this} + a;
        return *this;
    }
};
static_assert(sizeof(float8_e4m3_t) == 1, "float8_e4m3_t must be 1 byte");

void cvt_f8_e5m2_to_float(float *out, const float8_e5m2_t *inp, size_t nelems);
void cvt_f8_e4m3_to_float(float *out, const float8_e4m3_t *inp, size_t nelems);
void cvt_float_to_f8_e5m2(float8_e5m2_t *out, const float *inp, size_t nelems);
void cvt_float_to_f8_e4m3(float8_e4m3_t *out, const float *inp, size_t nelems);

// performs element-by-element sum of inp and add float arrays and stores
// result to f8 out array with down-conversion
void add_floats_and_cvt_to_f8_e5m2(float8_e5m2_t *out, const float *inp0,
        const float *inp1, size_t nelems);
void add_floats_and_cvt_to_f8_e4m3(float8_e4m3_t *out, const float *inp0,
        const float *inp1, size_t nelems);

} // namespace impl
} // namespace dnnl

#endif
