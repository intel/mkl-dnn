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

#include <array>

#include "common/bfloat16.hpp"
#include "common/bit_cast.hpp"
#include "common/dnnl_thread.hpp"
#include "common/float16.hpp"
#include "common/float8.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {

float8_e8m0_t &float8_e8m0_t::operator=(float f) {
    // we keep exponent bits and discard sign bit.
    // - support only round to zero to simplify conversion.
    // - no infinities in e8m0, so those become NaN essentially
    uint32_t uf = utils::bit_cast<uint32_t>(f);
    raw_bits_ = (uf >> 23) & 0xff;
    return *this;
}

float8_e8m0_t::operator float() const {
    // no inf, only NaN in e8m0
    // we return Real Indefinite NaN
    if (raw_bits_ == 0xff) return utils::bit_cast<float>(0xffc00000);
    // no zero, return subnormal float 2^-127
    if (raw_bits_ == 0x00) return utils::bit_cast<float>(0x00400000);
    return utils::bit_cast<float>(raw_bits_ << 23);
}

float8_e5m2_t &float8_e5m2_t::operator=(float16_t f) {
    // we just need to apply rounding
    uint16_t fraw = f.raw;
    uint16_t naninf_mask = 0x7c00;

    bool is_special = (fraw & naninf_mask) == naninf_mask;
    bool is_nan = is_special && (fraw & 0x03ff); // one of the lsb is non zero

    // we always set quiet bit for NaN
    if (is_nan) {
        raw_bits_ = (fraw >> 8) | 0x02;
        return *this;
    }

    // if infinity, we just return it as is
    if (is_special) {
        raw_bits_ = fraw >> 8;
        return *this;
    }

    // otherwise we just round and return
    int16_t rounding_nudge = 0x007f + ((fraw & 0x0100) >> 8);
    fraw = fraw + rounding_nudge;
    raw_bits_ = fraw >> 8;
    return *this;
}

float8_e5m2_t &float8_e5m2_t::operator=(bfloat16_t f) {
    constexpr auto bf16_mantissa_len = 7;
    constexpr auto bf16_mantissa_mask = 0x007f;
    constexpr auto bf16_exp_mask = 0x7f80;
    constexpr auto f8_e5m2_mantissa_len = 2;

    const bool is_special = (f.raw_bits_ & bf16_exp_mask)
            == bf16_exp_mask; // exp has all bits set
    const bool is_nan = is_special
            && (f.raw_bits_ & bf16_mantissa_mask); // and non-zero mantissa

    if (is_special) {
        // sign bit and 5 exp bits = 6 bits, they are S'11111
        raw_bits_ = ((f.raw_bits_ >> 8) & 0xfc);

        // we always set quiet bit for NaN
        if (is_nan) {
            // also add 2 mantissa bits
            raw_bits_ = raw_bits_ | ((f.raw_bits_ >> 5) & 0x3) | 0x2;
            return *this;
        }

        // if infinity then mantissa is zero
        return *this;
    }

    // first we extract the sign and make the input positive
    const unsigned int s8 = (f.raw_bits_ & 0x8000) >> 8;
    const int fraw = f.raw_bits_ & 0x7fff;

    // we filter out overflow
    // Note: values in [57344;61184] round to 57344, which is representable
    // So we overflow above 61184
    constexpr auto bf16_61184_raw = 0x476f;
    if (fraw > bf16_61184_raw) {
        raw_bits_ = s8 | 0x7c; // S'11111'00
        return *this;
    }
    // we filter out underflow when f <= 2^-17
    constexpr auto bf16_2_to_minus17_raw = 0x3700;
    if (fraw <= bf16_2_to_minus17_raw) {
        raw_bits_ = s8;
        return *this;
    }

    // Here the idea is to add a large constant to the bfloat16_t to force the
    // proper rounding to f8_e5m2 accuracy.
    // Compute the rounding shifter by taking its exponent + 5.
    // It does not overflow as fraw <= 61184.
    // This allows to discard 5 bits of mantissa during addition,
    // leaving the remaining 2 bits perfectly rounded.
    constexpr bool is_bitcast = true;
    constexpr auto shifter_exp_adjustment = 5 << bf16_mantissa_len;
    bfloat16_t shifter(
            (fraw & bf16_exp_mask) + shifter_exp_adjustment, is_bitcast);
    // e8 = e16 - e16_bias + e8_bias = e16 - 127 + 15
    // e8 will be denorm if e8 <= 0 or e16 + 5 < 118
    constexpr int exp_threshold = 0x3b00; // raw bits of exponent = 118
    const bool is_denorm = shifter.raw_bits_ < exp_threshold;
    if (is_denorm) shifter.raw_bits_ = exp_threshold;

    bfloat16_t rounded = (bfloat16_t(fraw, is_bitcast) + shifter);
    // separate line to force line above to round to bf16
    rounded = rounded - shifter;

    constexpr auto exp_bias_diff = 112; // bf16_bias - f8_e5m2_bias = 127 - 15
    int e8 = ((rounded.raw_bits_ & bf16_exp_mask) >> bf16_mantissa_len)
            - exp_bias_diff;

    constexpr auto mantissa_len_diff = bf16_mantissa_len - f8_e5m2_mantissa_len;
    uint8_t m8 = (rounded.raw_bits_ & bf16_mantissa_mask) >> mantissa_len_diff;

    // we need to make the implicit mantissa bit explicit for
    // denorm f8_e5m2
    if (is_denorm) {
        m8 = (m8 | (1 << f8_e5m2_mantissa_len)) >> (-e8 + 1);
        e8 = 0;
    }

    raw_bits_ = s8 | (e8 << f8_e5m2_mantissa_len) | m8;
    return *this;
}

float8_e5m2_t &float8_e5m2_t::convert_fp32_to_fp8e5m2_RNE(
        float f, bool is_saturating) {
    // Convert f32 -> f8_e5m2 directly with a single rounding step.
    // RNE, DAZ=1, FTZ=0
    using namespace utils;
    uint32_t fraw = bit_cast<uint32_t>(f);
    const uint32_t fp32_bias = 127;
    const uint32_t bf8_bias = 15;

    // extract sign, exponent and mantissa
    uint32_t s_32 = fraw >> 31;
    uint32_t exp_32 = (fraw >> 23) & 0xff;
    uint32_t mant_32 = fraw & 0x7fffff;

    raw_bits_ = 0;

    // Check for Zero or Denormals
    if (exp_32 == 0) {
        raw_bits_ = s_32 << 7; // return signed zero
        return *this;
    }

    // Check for NaN
    if (exp_32 == 0xff && mant_32 != 0) {
        raw_bits_ = ((s_32 << 7) | 0x7c); //same sign, exponent all ones
        mant_32 |= 0x00400000;
        uint8_t mant_8 = (mant_32 >> 21) & 0x03;
        raw_bits_ |= mant_8; // set mantissa bits
        return *this;
    }

    // Check for Infinity
    if ((exp_32 == 0xff) && (mant_32 == 0)) {
        if (is_saturating) {
            raw_bits_ = (s_32 << 7) | 0x7b; // MAX BF8
        } else {
            raw_bits_ = (s_32 << 7) | 0x7c; // Infinity
        }
        return *this;
    }

    // Overflow --> make it INF or MAX BF8
    if ((exp_32 >= (fp32_bias - bf8_bias + 31))
            || ((exp_32 == (fp32_bias - bf8_bias + 30))
                    && (mant_32 >= 0x00700000))) {
        if (is_saturating) {
            raw_bits_ = (s_32 << 7) | 0x7b; // MAX BF8
        } else {
            raw_bits_ = (s_32 << 7) | 0x7c; // Infinity
        }
        return *this;
    }

    // Underflow
    if (exp_32 <= (fp32_bias - bf8_bias)) {
        // denormalized mantissa
        uint32_t mant = mant_32 | 0x00800000; // add implicit leading 1

        // subnormal shift (clamped: shifts >= 32 are UB in C++)
        const uint32_t shift = (fp32_bias - bf8_bias) - exp_32 + 1;
        mant = shift >= 32 ? 0 : (mant >> shift);

        // preserve sticky bit (some sticky bits are lost when denormalizing)
        uint32_t shift_out_sticky
                = (((mant_32 & 0x001fffff) + 0x001fffff) >> 21);
        mant |= shift_out_sticky;

        // RNE rounding
        uint32_t l_bit = (mant >> 21) & 0x1;
        mant += 0x000fffff + l_bit;

        if (((mant >> 23) & 0x1) == 1) {
            // rounding overflowed into the normal range: min normal
            raw_bits_ = (s_32 << 7) | 0x4; // exp = 1, mant = 0
        } else {
            mant = (mant >> 21) & 0x3;
            raw_bits_ = (s_32 << 7) | mant;
        }
        return *this;
    }

    uint32_t l_bit = (mant_32 >> 21) & 0x1;
    uint32_t r_nex = fraw + 0x000fffff + l_bit;
    uint32_t exp_8 = ((r_nex & 0x7f800000) >> 23);
    uint32_t mant = (r_nex & 0x00600000);
    exp_8 -= (fp32_bias - bf8_bias);
    mant >>= 21;

    raw_bits_ |= exp_8 << 2;
    raw_bits_ |= mant;
    raw_bits_ |= s_32 << 7;

    return *this;
}

float8_e5m2_t &float8_e5m2_t::operator=(float f) {
    return this->convert_fp32_to_fp8e5m2_RNE(f, false);
}

float8_e5m2_t::operator float() const {
    float16_t f16 = *this;
    return static_cast<float>(f16);
}

float8_e5m2_t::operator float16_t() const {
    uint16_t snan_mask = 0x7d;
    uint16_t qnan_qbit = 0x02;
    const bool is_snan = (raw_bits_ & snan_mask) == snan_mask;
    const uint8_t raw = is_snan ? raw_bits_ | qnan_qbit : raw_bits_;
    std::array<uint8_t, 2> iraw = {{0, raw}};
    auto f16 = utils::bit_cast<float16_t>(iraw);
    return f16;
}

float8_e5m2_t::operator bfloat16_t() const {
    const uint16_t s8 = (raw_bits_ & 0x80) >> 7;
    const uint16_t e8 = (raw_bits_ & 0x7c) >> 2;
    const uint16_t m8 = (raw_bits_ & 0x3);

    const uint16_t s_bf16 = s8;
    uint16_t e_bf16 = 0;
    uint16_t m_bf16 = 0;

    if (e8 == 0 && m8 == 0) {
        // Zero
        e_bf16 = 0;
        m_bf16 = 0;
    } else if (e8 == 0x1f && m8 != 0) {
        // NaN: convert to quiet NaN, preserve payload
        e_bf16 = 0xff;
        m_bf16 = 0x40 | (m8 << 5);
    } else if (e8 == 0x1f && m8 == 0) {
        // Infinity
        e_bf16 = 0xff;
        m_bf16 = 0;
    } else if (e8 == 0) {
        // Need to convert f8_e5m2 denormal into bf16 normal
        const int shift = (m8 & 0x2) ? 1 : 2;
        // e5m2 bias=15, bf16 bias=127
        e_bf16 = 1 - shift + 112; // 112 = 127 - 15
        m_bf16 = ((m8 << shift) & 0x3) << 5;
    } else {
        // Normal values
        e_bf16 = e8 + 112; // = (e8 - 15) + 127
        m_bf16 = m8 << 5;
    }

    const uint16_t bf16_bits = (s_bf16 << 15) | (e_bf16 << 7) | m_bf16;
    return utils::bit_cast<bfloat16_t>(bf16_bits);
}

float8_e4m3_t &float8_e4m3_t::operator=(float16_t f) {
    using namespace utils;
    // Here the idea is to add a large constant to the float16_t to force the
    // proper rounding to f8_e4m3 accuracy.
    int fraw = f.raw;

    // first we extract the sign and make the input positive
    unsigned int s8 = (fraw & 0x8000) >> 8;
    fraw = fraw & 0x7fff;

    // we filter out overflow, nan
    // Note: values in [448;464] round to 448, which is representable
    // So we overflow above 464
    if (fraw > 0x5f40) {
        raw_bits_ = s8 | 0x7f;
        return *this;
    }
    // we filter out underflow when f <= 2^-10
    if (fraw <= 0x1400) {
        raw_bits_ = s8;
        return *this;
    }

    // Compute the rounding shifter by taking its exponent + 7.
    // Lucky us, it does not overflow as fraw <= 448.
    // This allows to discard 7 bits of mantissa during addition,
    // leaving the remaining 3 bits perfectly rounded.
    constexpr bool is_bitcast = true;
    float16_t shifter((fraw & 0x7c00) + 0x1c00, is_bitcast);
    // e8 = e16 - e16_bias + e8_bias = e16 - 15 + 7
    // e8 will be denorm if e8 <= 0 or e16 + 7 < 16
    constexpr int exp_threshold = 0x4000; // raw bits of exponent = 16
    int is_denorm = shifter.raw < exp_threshold;
    if (is_denorm) shifter.raw = exp_threshold;

    float16_t rounded = (float16_t(fraw, is_bitcast) + shifter);
    // separate line to force line above to round to f16
    rounded = rounded - shifter;

    int e8 = ((rounded.raw & 0x7c00) >> 10) - 8;
    uint8_t m8 = (rounded.raw & 0x03ff) >> 7;

    // we need to make the implicit f32 mantissa bit explicit for
    // denorm f8_e4m3
    if (is_denorm) {
        m8 = (m8 | 0x00000008) >> (-e8 + 1);
        e8 = 0;
    }

    raw_bits_ = s8 | (e8 << 3) | m8;
    return *this;
}

float8_e4m3_t &float8_e4m3_t::convert_fp32_to_fp8e4m3_RNE(
        float f, bool is_saturating) {
    // Convert f32 -> f8_e4m3 directly with a single rounding step.
    // RNE, DAZ=1, FTZ=0
    using namespace utils;
    uint32_t fraw = bit_cast<uint32_t>(f);
    const uint32_t fp32_bias = 127;
    const uint32_t hf8_bias = 7;

    // extract sign, exponent and mantissa
    uint32_t s_32 = fraw >> 31;
    uint32_t exp_32 = (fraw >> 23) & 0xff;
    uint32_t mant_32 = fraw & 0x7fffff;

    raw_bits_ = 0;

    // Check for Zero or Denormals
    if (exp_32 == 0) {
        raw_bits_ = s_32 << 7; // return signed zero
        return *this;
    }

    // Check for NaN
    if (exp_32 == 0xff && mant_32 != 0) {
        raw_bits_ = ((s_32 << 7) | 0x7f); //same sign, exponent all ones
        return *this;
    }

    // Check for Infinity
    if ((exp_32 == 0xff) && (mant_32 == 0)) {
        if (is_saturating) {
            raw_bits_ = (s_32 << 7) | 0x7e; // MAX HF8
        } else {
            raw_bits_ = (s_32 << 7) | 0x7f; // NAN
        }
        return *this;
    }

    // Overflow --> make it NaN or MAX HF8
    if ((exp_32 > (fp32_bias - hf8_bias + 15))
            || ((exp_32 == (fp32_bias - hf8_bias + 15))
                    && (mant_32 > 0x00680000))) {
        if (is_saturating) {
            raw_bits_ = (s_32 << 7) | 0x7e; // MAX HF8
        } else {
            raw_bits_ = (s_32 << 7) | 0x7f; // NAN
        }
        return *this;
    }

    // Underflow
    if (exp_32 <= (fp32_bias - hf8_bias)) {
        // denormalized mantissa
        uint32_t mant = mant_32 | 0x00800000; // add implicit leading 1

        // subnormal shift (clamped: shifts >= 32 are UB in C++)
        const uint32_t shift = (fp32_bias - hf8_bias) - exp_32 + 1;
        mant = shift >= 32 ? 0 : (mant >> shift);

        // preserve sticky bit (some sticky bits are lost when denormalizing)
        uint32_t shift_out_sticky
                = (((mant_32 & 0x000fffff) + 0x000fffff) >> 20);
        mant |= shift_out_sticky;

        // RNE rounding
        uint32_t l_bit = (mant >> 20) & 0x1;
        mant += 0x0007ffff + l_bit;

        if (((mant >> 23) & 0x1) == 1) {
            // rounding overflowed into the normal range: min normal
            raw_bits_ = (s_32 << 7) | 0x8; // exp = 1, mant = 0
        } else {
            mant = (mant >> 20) & 0x7;
            raw_bits_ = (s_32 << 7) | mant;
        }
        return *this;
    }

    uint32_t l_bit = (mant_32 >> 20) & 0x1;
    uint32_t r_nex = fraw + 0x0007ffff + l_bit;
    uint32_t exp_8 = ((r_nex & 0x7f800000) >> 23);
    uint32_t mant = (r_nex & 0x00700000);
    mant = (mant >> 20) & 0x7;
    exp_8 -= (fp32_bias - hf8_bias);

    raw_bits_ |= exp_8 << 3;
    raw_bits_ |= mant;
    raw_bits_ |= s_32 << 7;

    return *this;
}

float8_e4m3_t &float8_e4m3_t::operator=(float f) {
    return convert_fp32_to_fp8e4m3_RNE(f, false);
}

float8_e4m3_t &float8_e4m3_t::operator=(bfloat16_t f) {
    constexpr auto bf16_mantissa_len = 7;
    constexpr auto bf16_mantissa_mask = 0x007f;
    constexpr auto bf16_exp_mask = 0x7f80;
    constexpr auto f8_e4m3_mantissa_len = 3;

    int fraw = f.raw_bits_;

    // first we extract the sign and make the input positive
    const unsigned int s8 = (fraw & 0x8000) >> 8;
    fraw = fraw & 0x7fff;

    // we filter out overflow, nan
    // Note: values in [448;464] round to 448, which is representable
    // So we overflow above 464
    constexpr auto bf16_464_raw = 0x43e8;
    if (fraw > bf16_464_raw) {
        raw_bits_ = s8 | 0x7f; // S'1111'111
        return *this;
    }
    // we filter out underflow when f <= 2^-10
    constexpr auto bf16_2_to_minus10_raw = 0x3a80;
    if (fraw <= bf16_2_to_minus10_raw) {
        raw_bits_ = s8;
        return *this;
    }

    // Here the idea is to add a large constant to the bfloat16_t to force the
    // proper rounding to f8_e4m3 accuracy.
    // Compute the rounding shifter by taking its exponent + 4.
    // It does not overflow as fraw <= 464.
    // This allows to discard 4 bits of mantissa during addition,
    // leaving the remaining 3 bits perfectly rounded.
    constexpr bool is_bitcast = true;
    constexpr auto shifter_exp_adjustment = 4 << bf16_mantissa_len;
    bfloat16_t shifter(
            (fraw & bf16_exp_mask) + shifter_exp_adjustment, is_bitcast);
    // e8 = e16 - e16_bias + e8_bias = e16 - 127 + 7
    // e8 will be denorm if e8 <= 0 or e16 + 4 < 125
    constexpr int exp_threshold = 0x3e80; // raw bits of exponent = 125
    const bool is_denorm = shifter.raw_bits_ < exp_threshold;
    if (is_denorm) shifter.raw_bits_ = exp_threshold;

    bfloat16_t rounded = (bfloat16_t(fraw, is_bitcast) + shifter);
    // separate line to force line above to round to bf16
    rounded = rounded - shifter;

    constexpr auto exp_bias_diff = 120; // bf16_bias - f8_e4m3_bias = 127 - 7
    int e8 = ((rounded.raw_bits_ & bf16_exp_mask) >> bf16_mantissa_len)
            - exp_bias_diff;

    constexpr auto mantissa_len_diff = bf16_mantissa_len - f8_e4m3_mantissa_len;
    uint8_t m8 = (rounded.raw_bits_ & bf16_mantissa_mask) >> mantissa_len_diff;

    // we need to make the implicit mantissa bit explicit for
    // denorm f8_e4m3
    if (is_denorm) {
        m8 = (m8 | (1 << f8_e4m3_mantissa_len)) >> (-e8 + 1);
        e8 = 0;
    }

    raw_bits_ = s8 | (e8 << f8_e4m3_mantissa_len) | m8;
    return *this;
}

float8_e4m3_t::operator float() const {
    float16_t f16 = *this;
    return static_cast<float>(f16);
}

float8_e4m3_t::operator float16_t() const {
    const uint16_t s8 = (raw_bits_ & 0x80) >> 7;
    const uint16_t e8 = (raw_bits_ & 0x78) >> 3;
    const uint16_t m8 = (raw_bits_ & 0x7);
    uint16_t s16 = s8;
    uint16_t e16 = e8 + 8; // 8 = 15 - 7 = f16_bias - f8_e4m3_bias
    uint16_t m16 = m8;

    // Need to convert f8_e4m3 denormal into f16 normal.
    if (e8 == 0 && m8 != 0) {
        uint16_t count = 2;
        count = m8 > 0x1 ? 1 : count;
        count = m8 > 0x3 ? 0 : count;
        e16 -= count;
        m16 = (m16 << (count + 1)) & 0x7;
    } else if (e8 == 0 && m8 == 0) {
        // set correct exponent for zero
        e16 = 0;
    } else if (e8 == 0xf && m8 == 0x7) {
        // set correct exponent and mantissa for NaN input
        e16 = 0x1f;
        m16 = 0x4; // Real Indefinite (a qNaN)
    }
    s16 <<= 15;
    e16 <<= 10;
    m16 <<= 7;

    const uint16_t u16 = s16 | e16 | m16;
    return utils::bit_cast<float16_t>(u16);
}

float8_e4m3_t::operator bfloat16_t() const {
    const uint16_t s8 = (raw_bits_ & 0x80) >> 7;
    const uint16_t e8 = (raw_bits_ & 0x78) >> 3;
    const uint16_t m8 = (raw_bits_ & 0x7);

    const uint16_t s_bf16 = s8;
    uint16_t e_bf16 = 0;
    uint16_t m_bf16 = 0;

    if (e8 == 0 && m8 == 0) {
        // Zero
        e_bf16 = 0;
        m_bf16 = 0;
    } else if (e8 == 0xf && m8 == 0x7) {
        // NaN (e4m3 has no infinity, only exp=15 mantissa=7 is NaN)
        // Convert to quiet NaN, preserve payload
        e_bf16 = 0xff;
        m_bf16 = 0x40 | (m8 << 4);
    } else if (e8 == 0) {
        // Need to convert f8_e4m3 denormal into bf16 normal
        const int shift = (m8 & 0x4) ? 1 : (m8 & 0x2) ? 2 : 3;
        // e4m3 bias=7, bf16 bias=127
        e_bf16 = 1 - shift + 120; // 120 = 127 - 7
        m_bf16 = ((m8 << shift) & 0x7) << 4;
    } else {
        // Normal values
        e_bf16 = e8 + 120; // = e8 - 7 + 127
        m_bf16 = m8 << 4;
    }

    const uint16_t bf16_bits = (s_bf16 << 15) | (e_bf16 << 7) | m_bf16;
    return utils::bit_cast<bfloat16_t>(bf16_bits);
}

void cvt_f8_e5m2_to_float(float *out, const float8_e5m2_t *inp, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = inp[i];
}

void cvt_f8_e4m3_to_float(float *out, const float8_e4m3_t *inp, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = inp[i];
}

void cvt_float_to_f8_e5m2(float8_e5m2_t *out, const float *inp, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float8_e5m2_t>(inp[i]);
}

void cvt_float_to_f8_e4m3(float8_e4m3_t *out, const float *inp, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float8_e4m3_t>(inp[i]);
}

void add_floats_and_cvt_to_f8_e5m2(float8_e5m2_t *out, const float *inp0,
        const float *inp1, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float8_e5m2_t>(inp0[i] + inp1[i]);
}

void add_floats_and_cvt_to_f8_e4m3(float8_e4m3_t *out, const float *inp0,
        const float *inp1, size_t nelems) {

    // TODO: implement and use jit conversion kernel for DNNL_X64

    PRAGMA_OMP_SIMD()
    for (size_t i = 0; i < nelems; ++i)
        out[i] = static_cast<float8_e4m3_t>(inp0[i] + inp1[i]);
}

} // namespace impl
} // namespace dnnl
