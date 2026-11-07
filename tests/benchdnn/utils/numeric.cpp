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

#include <algorithm>
#include <cmath>

#include "src/common/bfloat16.hpp"
#include "src/common/float16.hpp"
#include "src/common/float4.hpp"
#include "src/common/float8.hpp"
#include "src/common/nibble.hpp"
#include "src/common/nstl.hpp"

#include "common.hpp"

#include "utils/numeric.hpp"

template <>
struct prec_traits<dnnl_f4_e2m1> {
    using type = dnnl::impl::float4_e2m1_t;
};
template <>
struct prec_traits<dnnl_e8m0> {
    using type = dnnl::impl::float8_e8m0_t;
};
template <>
struct prec_traits<dnnl_f8_e5m2> {
    using type = dnnl::impl::float8_e5m2_t;
};
template <>
struct prec_traits<dnnl_f8_e4m3> {
    using type = dnnl::impl::float8_e4m3_t;
};
template <>
struct prec_traits<dnnl_bf16> {
    using type = dnnl::impl::bfloat16_t;
};
template <>
struct prec_traits<dnnl_f16> {
    using type = dnnl::impl::float16_t;
};
template <>
struct prec_traits<dnnl_f32> {
    using type = float;
};

// XXX: benchdnn infra doesn't support double yet.
// Use float's max/min/epsilon values to avoid following build warnings:
// warning C4756: overflow in constant arithmetic.
// This should be fixed once cpu reference in f64 is added.
template <>
struct prec_traits<dnnl_f64> {
    using type = float;
};
template <>
struct prec_traits<dnnl_s32> {
    using type = int32_t;
};
template <>
struct prec_traits<dnnl_s64> {
    using type = int64_t;
};
template <>
struct prec_traits<dnnl_s8> {
    using type = int8_t;
};
template <>
struct prec_traits<dnnl_u8> {
    using type = uint8_t;
};
template <>
struct prec_traits<dnnl_s4> {
    using type = dnnl::impl::int4_t;
};
template <>
struct prec_traits<dnnl_u4> {
    using type = dnnl::impl::uint4_t;
};
template <>
struct prec_traits<dnnl_u2> {
    using type = dnnl::impl::uint2_t;
};
template <>
struct prec_traits<dnnl_u3> {
    using type = dnnl::impl::uint3_t;
};
#define CASE_ALL(dt) \
    switch (dt) { \
        CASE(dnnl_f4_e2m1); \
        CASE(dnnl_e8m0); \
        CASE(dnnl_f8_e5m2); \
        CASE(dnnl_f8_e4m3); \
        CASE(dnnl_bf16); \
        CASE(dnnl_f16); \
        CASE(dnnl_f32); \
        CASE(dnnl_f64); \
        CASE(dnnl_s32); \
        CASE(dnnl_s8); \
        CASE(dnnl_u8); \
        CASE(dnnl_s4); \
        CASE(dnnl_u4); \
        CASE(dnnl_u2); \
        CASE(dnnl_u3); \
        default: assert(!"bad data_type"); SAFE_V(FAIL); \
    }

/* std::numeric_limits::digits functionality */
int digits_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::digits;

    CASE_ALL(dt);

#undef CASE
    return 0;
}

float epsilon_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::epsilon();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

float lowest_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::lowest();

    CASE_ALL(dt);

#undef CASE

    return 0;
}

float max_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::max();

    CASE_ALL(dt);

#undef CASE
    return 0;
}

// Smallest positive normal value (std::numeric_limits::min functionality).
float min_normal_dt(dnnl_data_type_t dt) {
#define CASE(dt) \
    case dt: \
        return (float)dnnl::impl::nstl::numeric_limits< \
                typename prec_traits<dt>::type>::min();

    CASE_ALL(dt);

#undef CASE
    return 0;
}

// Returns the distance between `x` (assumed representable in `dt`) and the
// next representable value of larger magnitude. Built on
// `nstl::numeric_limits`: the ULP at magnitude `|x|` is
// 2^(floor(log2|x|) - (digits - 1)). In the subnormal range the exponent is
// clamped to the one of the smallest normal value (`min()`), which gives the
// constant subnormal step. Returns 1 for integral data types and 0 for
// unsupported data types.
float ulp_dt(dnnl_data_type_t dt, float x) {
    // Integral types have a uniform grid with a step of 1.
    if (is_integral_dt(dt)) return 1.f;

    // Only floating-point types covered by `CASE_ALL` (others would assert).
    switch (dt) {
        case dnnl_f4_e2m1:
        case dnnl_f8_e5m2:
        case dnnl_f8_e4m3:
        case dnnl_bf16:
        case dnnl_f16:
        case dnnl_f32: break;
        default: return 0.f;
    }

    const float min_normal = min_normal_dt(dt);
    const int mant_bits = digits_dt(dt) - 1;
    if (min_normal <= 0.f || mant_bits < 0) return 0.f;

    int exp2 = 0;
    // frexp: |v| = m * 2^exp2, m in [0.5, 1) => floor(log2|v|) = exp2 - 1.
    std::frexp(std::max(fabsf(x), min_normal), &exp2);
    return std::ldexp(1.f, (exp2 - 1) - mant_bits);
}

template <dnnl_data_type_t dt>
float saturate_and_round(float value) {
    static const float dt_max = max_dt(dt);
    static const float dt_min = lowest_dt(dt);
    static const float max_dt_s32 = max_dt(dnnl_s32);
    if (dt == dnnl_s32 && value >= max_dt_s32) return max_dt_s32;
    if (value > dt_max) value = dt_max;
    if (value < dt_min) value = dt_min;
    return mxcsr_cvt(value);
}

bool is_integral_dt(dnnl_data_type_t dt) {
    switch (dt) {
        case dnnl_s64:
        case dnnl_s32:
        case dnnl_s8:
        case dnnl_u8:
        case dnnl_s4:
        case dnnl_u4:
        case dnnl_u2:
        case dnnl_u3: return true;
        case dnnl_f64:
        case dnnl_f32:
        case dnnl_bf16:
        case dnnl_f16:
        case dnnl_e8m0:
        case dnnl_f8_e5m2:
        case dnnl_f8_e4m3:
        case dnnl_f4_e2m1:
        case dnnl_boolean: return false;
        default: assert(!"unsupported data type"); SAFE_V(FAIL);
    }

    return false;
}

template <dnnl_data_type_t dt>
float maybe_saturate_templ(float value) {
    if (!is_integral_dt(dt)) return value;
    return saturate_and_round<dt>(value);
}

float maybe_saturate(dnnl_data_type_t dt, float value) {
#define CASE(dt) \
    case dt: return maybe_saturate_templ<dt>(value)

    CASE_ALL(dt)
#undef CASE
    return value;
}

template <dnnl_data_type_t dt>
float round_to_nearest_representable_templ(float value) {
    switch (dt) {
        case dnnl_f32: break;
        case dnnl_f64: break;
        case dnnl_f4_e2m1:
            value = (float)dnnl::impl::float4_e2m1_t(value);
            break;
        case dnnl_e8m0: value = (float)dnnl::impl::float8_e8m0_t(value); break;
        case dnnl_f8_e5m2:
            value = (float)dnnl::impl::float8_e5m2_t(value);
            break;
        case dnnl_f8_e4m3:
            value = (float)dnnl::impl::float8_e4m3_t(value);
            break;
        case dnnl_bf16: value = (float)dnnl::impl::bfloat16_t(value); break;
        case dnnl_f16: value = (float)dnnl::impl::float16_t(value); break;
        case dnnl_s32:
        case dnnl_s8:
        case dnnl_u8:
        case dnnl_s4:
        case dnnl_u4:
        case dnnl_u2:
        case dnnl_u3: value = maybe_saturate_templ<dt>(value); break;
        default: SAFE_V(FAIL);
    }

    return value;
}

float round_to_nearest_representable(dnnl_data_type_t dt, float value) {
#define CASE(dt) \
    case dt: return round_to_nearest_representable_templ<dt>(value)

    CASE_ALL(dt)
#undef CASE
    return value;
}

#undef CASE_ALL

bool is_subbyte_type(dnnl_data_type_t dt) {
    return bits_dt(dt) < 8;
}

size_t bits_dt(dnnl_data_type_t dt) {
    switch (dt) {
        case dnnl_s64:
        case dnnl_f64: return 64;
        case dnnl_s32:
        case dnnl_f32: return 32;
        case dnnl_bf16:
        case dnnl_f16: return 16;
        case dnnl_e8m0:
        case dnnl_f8_e5m2:
        case dnnl_f8_e4m3:
        case dnnl_s8:
        case dnnl_u8: return 8;
        case dnnl_f4_e2m1:
        case dnnl_s4:
        case dnnl_u4: return 4;
        case dnnl_u2: return 2;
        case dnnl_u3: return 3;
        case dnnl_boolean: return 1;
        default: assert(!"unsupported data type"); SAFE_V(FAIL);
    }

    return 0;
}

bool is_fp8_dt(dnnl_data_type_t type) {
    return type == dnnl_f8_e5m2 || type == dnnl_f8_e4m3;
}

float get_element(dnnl_data_type_t dt, int64_t idx, void *ptr) {
    float elem = 0.f;
#define CASE(dt) \
    case dt: elem = static_cast<prec_traits<dt>::type *>(ptr)[idx]; break;

    switch (dt) {
        CASE(dnnl_s8);
        CASE(dnnl_u8);
        CASE(dnnl_s32);
        CASE(dnnl_s64);
        CASE(dnnl_f32);
        CASE(dnnl_f16);
        CASE(dnnl_bf16);
        CASE(dnnl_e8m0);
        CASE(dnnl_f8_e5m2);
        CASE(dnnl_f8_e4m3);
        case dnnl_f64: elem = static_cast<double *>(ptr)[idx]; break;
        case dnnl_s4: {
            dnnl::impl::nibble2_t nibble_pair(
                    reinterpret_cast<uint8_t *>(ptr)[idx / 2]);
            elem = dnnl::impl::int4_t(nibble_pair.get(idx % 2));
            break;
        }
        case dnnl_u4: {
            dnnl::impl::nibble2_t nibble_pair(
                    reinterpret_cast<uint8_t *>(ptr)[idx / 2]);
            elem = dnnl::impl::uint4_t(nibble_pair.get(idx % 2));
            break;
        }
        case dnnl_f4_e2m1: {
            dnnl::impl::nibble2_t nibble_pair(
                    reinterpret_cast<uint8_t *>(ptr)[idx / 2]);
            elem = dnnl::impl::float4_e2m1_t(nibble_pair.get(idx % 2));
            break;
        }
        case dnnl_u3: {
            const auto nibble_octet
                    = reinterpret_cast<dnnl::impl::nibble8_t *>(ptr)[idx / 8];
            elem = dnnl::impl::uint3_t(nibble_octet.get(idx % 8));
            break;
        }
        default: assert(!"bad data type");
    }
#undef CASE

    return elem;
}

void set_element(dnnl_data_type_t dt, int64_t idx, void *ptr, float value) {
#define CASE(dt) \
    case dt: static_cast<prec_traits<dt>::type *>(ptr)[idx] = value; break;

    switch (dt) {
        CASE(dnnl_s8);
        CASE(dnnl_u8);
        CASE(dnnl_s32);
        CASE(dnnl_s64);
        CASE(dnnl_f32);
        CASE(dnnl_f16);
        CASE(dnnl_bf16);
        CASE(dnnl_e8m0);
        CASE(dnnl_f8_e5m2);
        CASE(dnnl_f8_e4m3);
        case dnnl_f64: ((double *)ptr)[idx] = value; break;
        case dnnl_s4: {
            auto dst_val = ((dnnl::impl::nibble2_t *)ptr)[idx / 2];
            dst_val.set(dnnl::impl::int4_t(value).raw_bits_, idx % 2);
            ((dnnl::impl::nibble2_t *)ptr)[idx / 2] = dst_val;
            break;
        }
        case dnnl_u4: {
            auto dst_val = ((dnnl::impl::nibble2_t *)ptr)[idx / 2];
            dst_val.set(dnnl::impl::uint4_t(value).raw_bits_, idx % 2);
            ((dnnl::impl::nibble2_t *)ptr)[idx / 2] = dst_val;
            break;
        }
        case dnnl_f4_e2m1: {
            auto dst_val = ((dnnl::impl::nibble2_t *)ptr)[idx / 2];
            dst_val.set(dnnl::impl::float4_e2m1_t(value).raw_bits_, idx % 2);
            ((dnnl::impl::nibble2_t *)ptr)[idx / 2] = dst_val;
            break;
        }
        case dnnl_u3: {
            auto dst_val = ((dnnl::impl::nibble8_t *)ptr)[idx / 8];
            dst_val.set(dnnl::impl::uint3_t(value).raw_bits_, idx % 8);
            ((dnnl::impl::nibble8_t *)ptr)[idx / 8] = dst_val;
            break;
        }
        default: assert(!"bad data type");
    }
#undef CASE
}
