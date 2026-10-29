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

#ifndef GPU_INTEL_INCLUDE_PHILOX_H
#define GPU_INTEL_INCLUDE_PHILOX_H

#define DT_UNDEF 1
#include "gpu/intel/include/types.h"

uint4 philox_4x32_vec4_w_offset(ulong idx, ulong seed, ulong offset) {
/* No native 64-bit multiply: use 32-bit mul + mul_hi for each product. */
#define PHILOX_4UINT_ROUND(mul, ctr, key) \
    ({ \
        uint lo0 = ctr.s0 * mul.s0; \
        uint hi0 = mul_hi(ctr.s0, mul.s0); \
        uint lo1 = ctr.s2 * mul.s1; \
        uint hi1 = mul_hi(ctr.s2, mul.s1); \
        (uint4)(hi1 ^ ctr.s1 ^ key.s0, lo1, hi0 ^ ctr.s3 ^ key.s1, lo0); \
    })

    ulong x = (idx & ~3L);
    uint4 ctr = (uint4)((uint)offset, (uint)(offset >> 32), (uint)x,
            (uint)(x >> 32));
    uint seed_lo = (uint)seed;
    uint seed_hi = (uint)(seed >> 32);
    const ulong seeds = as_ulong((uint2)(seed_lo, seed_hi));

    const uint2 PHILOX_M4x32 = (uint2)(0xD2511F53u, 0xCD9E8D57u);
    const ulong PHILOX_W4x32 = as_ulong((uint2)(0x9E3779B9u, 0xBB67AE85u));
    const uint16 key0 = as_uint16((ulong8)(seeds))
            + as_uint16((ulong8)(PHILOX_W4x32))
                    * (uint16)(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const uint4 key1 = as_uint4((ulong2)seeds)
            + as_uint4((ulong2)(PHILOX_W4x32)) * (uint4)(8, 8, 9, 9);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s01);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s23);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s45);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s67);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.s89);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.sAB);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.sCD);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key0.sEF);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key1.s01);
    ctr = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctr, key1.s23);
    return ctr;
}

/* Two independent vec4 chains (idx&~3, idx+4) per call, for ILP. */
uint8 philox_4x32_vec4x2_w_offset(ulong idx, ulong seed, ulong offset) {
    ulong x0 = (idx & ~3L);
    ulong x1 = x0 + 4;
    uint4 ctrA = (uint4)((uint)offset, (uint)(offset >> 32), (uint)x0,
            (uint)(x0 >> 32));
    uint4 ctrB = (uint4)((uint)offset, (uint)(offset >> 32), (uint)x1,
            (uint)(x1 >> 32));
    uint seed_lo = (uint)seed;
    uint seed_hi = (uint)(seed >> 32);
    const ulong seeds = as_ulong((uint2)(seed_lo, seed_hi));

    const uint2 PHILOX_M4x32 = (uint2)(0xD2511F53u, 0xCD9E8D57u);
    const ulong PHILOX_W4x32 = as_ulong((uint2)(0x9E3779B9u, 0xBB67AE85u));
    const uint16 key0 = as_uint16((ulong8)(seeds))
            + as_uint16((ulong8)(PHILOX_W4x32))
                    * (uint16)(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const uint4 key1 = as_uint4((ulong2)seeds)
            + as_uint4((ulong2)(PHILOX_W4x32)) * (uint4)(8, 8, 9, 9);

#define PHILOX_4UINT_ROUND_X2(key) \
    do { \
        ctrA = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctrA, (key)); \
        ctrB = PHILOX_4UINT_ROUND(PHILOX_M4x32, ctrB, (key)); \
    } while (0)

    PHILOX_4UINT_ROUND_X2(key0.s01);
    PHILOX_4UINT_ROUND_X2(key0.s23);
    PHILOX_4UINT_ROUND_X2(key0.s45);
    PHILOX_4UINT_ROUND_X2(key0.s67);
    PHILOX_4UINT_ROUND_X2(key0.s89);
    PHILOX_4UINT_ROUND_X2(key0.sAB);
    PHILOX_4UINT_ROUND_X2(key0.sCD);
    PHILOX_4UINT_ROUND_X2(key0.sEF);
    PHILOX_4UINT_ROUND_X2(key1.s01);
    PHILOX_4UINT_ROUND_X2(key1.s23);
#undef PHILOX_4UINT_ROUND_X2
    return (uint8)(ctrA, ctrB);
}

uint philox_4x32_w_offset(long idx, long seed, long offset) {
    return philox_4x32_vec4_w_offset(
            (ulong)idx, (ulong)seed, (ulong)offset)[idx & 3L];
}

uint philox_4x32(long idx, long seed) {
    // Note: this is for compatibility with impls that don't support s64 rand
    // Truncate to uint first to handle negative longs correctly
    uint idx_u = (uint)idx;
    uint seed_u = (uint)seed;
    ulong x = idx_u & ~3UL;
    ulong idx_64 = ((x + 3) << 32) + (x + 2);
    ulong offset_64 = ((x + 1) << 32) + x;
    ulong seed_64 = ((ulong)seed_u << 32) + seed_u;
    return philox_4x32_w_offset(idx_64, seed_64, offset_64);
}

uint4 philox_4x32_vec4(uint idx, uint seed) {
    ulong x = idx & ~3L;
    ulong idx_64 = ((x + 3) << 32) + (x + 2);
    ulong offset_64 = ((x + 1) << 32) + x;
    ulong seed_64 = ((ulong)(seed) << 32) + seed;
    return philox_4x32_vec4_w_offset(idx_64, seed_64, offset_64);
}

ushort philox_8x16(long idx, long seed) {
    return as_ushort2(philox_4x32((idx >> 1), seed))[idx & 1];
}

uchar philox_16x8(long idx, long seed) {
    return as_uchar4(philox_4x32((idx >> 2), seed))[idx & 3];
}

#if WITH_SROUND

#if DST_DT_DIGITS > 24
#error "Invalid dst digits"
#endif

float stochastic_round_fwd(float s, long idx, uint seed) {
    if (isnan(s) || isinf(s)) return s;
    uint truncation_mask = 0xffffffff << (24 - DST_DT_DIGITS);
    uint bias_val = sizeof(DST_DATA_T) == 2 ? philox_16x8(idx, (long)seed)
                                            : philox_8x16(idx, (long)seed);
    uint rnd_bias = (uint)(bias_val & ~truncation_mask);
    float r = as_float((as_uint(s) + rnd_bias) & truncation_mask);
    r = fmin(fmax((float)DST_DATA_FLOW, r), (float)DST_DATA_FMAX);
    if (fabs(r) > 0 && fabs(r) < DST_DATA_FMIN) r = 0;
    return r;
}
#endif

#if WITH_DROPOUT
// No need to enable fp64 extensions just to compute (double)p * 0xFFFFFFFFu
uint get_dropout_threshold(float p) {
    if (p >= 1.f) return 0xFFFFFFFFu;
    char exponent = 126 - ((as_uint(p) >> 23) & 0x7F);
    if ((p <= 0.f) || (exponent > 31)) return 0u;
    uint mantissa = (as_uint(p) << 8) | 0x80000000u;
    if (!exponent) return (convert_ulong(mantissa) * 0xFFFFFFFFuL) >> 32;
    return ((convert_ulong(mantissa >> exponent) * 0xFFFFFFFFuL) >> 32)
            + !!(mantissa & ((1u << exponent) - 1u));
}
#endif
#endif
