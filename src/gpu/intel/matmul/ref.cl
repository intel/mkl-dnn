/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "gpu/intel/include/philox.h"
#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"

#define offset6D(d0, d1, d2, d3, d4, d5, s0, s1, s2, s3, s4, s5) \
    ((d0) * (s0) + (d1) * (s1) + (d2) * (s2) + (d3) * (s3) + (d4) * (s4) \
            + (d5) * (s5))

#if RUNTIME_DIMS
#define SRC_OFFSET(m, k) \
    offset6D(m, k, d0, d1, d2, d3, a_stride_m, a_stride_k, a_stride_d0, \
            a_stride_d1, a_stride_d2, a_stride_d3)
#define WEI_OFFSET(k, n) \
    offset6D(k, n, d0, d1, d2, d3, b_stride_k, b_stride_n, b_stride_d0, \
            b_stride_d1, b_stride_d2, b_stride_d3)
#define DST_OFFSET(m, n) \
    offset6D(m, n, d0, d1, d2, d3, c_stride_m, c_stride_n, c_stride_d0, \
            c_stride_d1, c_stride_d2, c_stride_d3)
#elif NDIMS == 5
#define SRC_OFFSET(m, k) SRC_OFF(d2 % SRC_D0, d1 % SRC_D1, d0 % SRC_D2, m, k)
#define WEI_OFFSET(k, n) WEI_OFF(0, d2 % WEI_D0, d1 % WEI_D1, d0 % WEI_D2, k, n)
#define DST_OFFSET(m, n) DST_OFF(d2 % DST_D0, d1 % DST_D1, d0 % DST_D2, m, n)
#elif NDIMS == 4
#define SRC_OFFSET(m, k) SRC_OFF(d1 % SRC_D0, d0 % SRC_D1, 0, m, k)
#define WEI_OFFSET(k, n) WEI_OFF(0, d1 % WEI_D0, d0 % WEI_D1, 0, k, n)
#define DST_OFFSET(m, n) DST_OFF(d1 % DST_D0, d0 % DST_D1, 0, m, n)
#elif NDIMS == 3
#define SRC_OFFSET(m, k) SRC_OFF(d0 % SRC_D0, m, 0, 0, k)
#define WEI_OFFSET(k, n) WEI_OFF(0, d0 % WEI_D0, k, 0, 0, n)
#define DST_OFFSET(m, n) DST_OFF(d0 % DST_D0, m, 0, 0, n)
#else
#define SRC_OFFSET(m, k) SRC_OFF(m, k, 0, 0, 0)
#define WEI_OFFSET(k, n) WEI_OFF(0, k, n, 0, 0, 0)
#define DST_OFFSET(m, n) DST_OFF(m, n, 0, 0, 0)
#endif

#define BATCH_OFF(pfx) \
    (pfx##_stride_d0 * d0 + pfx##_stride_d1 * d1 + pfx##_stride_d2 * d2)

__kernel void ref_matmul(__global SRC_DATA_T *A, __global WEI_DATA_T *B,
#if WITH_DYN_DST_SCALE
        __global float *C,
#else
        __global DST_DATA_T *C,
#endif
#if WITH_SUM
        __global SUM_DATA_T *C_sum,
#endif
        __global BIA_DATA_T *bia,
#if WITH_HOST_SRC_ZP
        SRC_ZP_DATA_T a0_value,
#else
        __global SRC_ZP_DATA_T *a0,
#endif
        long src_zp_stride_k, long src_zp_stride_m, long src_zp_stride_d0,
        long src_zp_stride_d1, long src_zp_stride_d2, long src_zp_group_k,
#if WITH_HOST_WEI_ZP
        WEI_ZP_DATA_T b0_value,
#else
        __global WEI_ZP_DATA_T *b0,
#endif
        long wei_zp_stride_n, long wei_zp_stride_k, long wei_zp_stride_d0,
        long wei_zp_stride_d1, long wei_zp_stride_d2, long wei_zp_group_n,
        long wei_zp_group_k,
#if WITH_HOST_DST_ZP
        int c0_value,
#else
        __global int *c0,
#endif
#if WITH_HOST_SRC_SCALE
        SRC_SCALES_DATA_T src_scale_value,
#else
        __global SRC_SCALES_DATA_T *src_scales,
#endif
        long src_scale_stride_k, long src_scale_stride_m,
        long src_scale_stride_d0, long src_scale_stride_d1,
        long src_scale_stride_d2, long src_scale_group_m,
        long src_scale_group_k,
#if WITH_HOST_WEI_SCALE
        WEI_SCALES_DATA_T wei_scale_value,
#else
        __global WEI_SCALES_DATA_T *wei_scales,
#endif
        long wei_scale_stride_n, long wei_scale_stride_k,
        long wei_scale_stride_d0, long wei_scale_stride_d1,
        long wei_scale_stride_d2, long wei_scale_group_n,
        long wei_scale_group_k,
#if WITH_HOST_DST_SCALE
        DST_SCALES_DATA_T dst_scale_value,
#else
        __global DST_SCALES_DATA_T *dst_scales,
#endif
        long dst_scale_stride_n, long dst_scale_stride_m,
        long dst_scale_stride_d0, long dst_scale_stride_d1,
        long dst_scale_stride_d2, long dst_scale_group_n,
        long dst_scale_group_m, __global SRC_GS_DATA_T *ag,
        long src_gs_stride_k, long src_gs_stride_m, long src_gs_stride_d0,
        long src_gs_stride_d1, long src_gs_stride_d2, long src_gs_group_k,
        long group_K, long K, long N, long M, long D0, long D1, long D2,
        long bia_stride_d3, long bia_stride_d2, long bia_stride_d1,
        long bia_stride_d0, long bia_stride_m, long bia_stride_n,
        long a_stride_d3, long a_stride_d2, long a_stride_d1, long a_stride_d0,
        long a_stride_m, long a_stride_k, long b_stride_d3, long b_stride_d2,
        long b_stride_d1, long b_stride_d0, long b_stride_k, long b_stride_n,
        long c_stride_d3, long c_stride_d2, long c_stride_d1, long c_stride_d0,
        long c_stride_m, long c_stride_n
#if WITH_DROPOUT
        ,
        __global uchar *dropout_mask_buf,
#if USE_HOST_SCALARS
        long dropout_seed, long dropout_offset, float dropout_p
#else
        __global SEED_DATA_T *dropout_seed_buf,
        __global long *dropout_offset_buf, __global float *dropout_p_buf
#endif
#endif
#if WITH_SROUND
        ,
        __global uint *sround_seed_buf
#endif
                POST_OP_ARGS) {

#if WITH_HOST_SRC_ZP
    SRC_ZP_DATA_T *a0 = &a0_value;
#endif
#if WITH_HOST_WEI_ZP
    WEI_ZP_DATA_T *b0 = &b0_value;
#endif
#if WITH_HOST_DST_ZP
    int *c0 = &c0_value;
#endif
#if WITH_HOST_SRC_SCALE
    SRC_SCALES_DATA_T *src_scales = &src_scale_value;
#endif
#if WITH_HOST_WEI_SCALE
    WEI_SCALES_DATA_T *wei_scales = &wei_scale_value;
#endif
#if WITH_HOST_DST_SCALE
    DST_SCALES_DATA_T *dst_scales = &dst_scale_value;
#endif

#if WITH_DROPOUT
#if !USE_HOST_SCALARS
    SEED_DATA_T dropout_seed = dropout_seed_buf[0];
    long dropout_offset = USE_OFFSET ? dropout_offset_buf[0] : 0;
    float dropout_p = dropout_p_buf[0];
#endif
    uint dropout_threshold = get_dropout_threshold(dropout_p);
    float dropout_inv_q = (dropout_p != 1.f) ? 1.f / (1.f - dropout_p) : 0.f;
#endif
#if WITH_SROUND
    uint sround_seed = sround_seed_buf[0];
#endif

    long m = get_global_id(0);
    long n = get_global_id(1);
    int mb = get_global_id(2);

#if WITH_DST_ZPOINTS
    int dst_zp = c0[WITH_DST_ZPOINTS_PER_OC ? n : 0];
#else
    int dst_zp = 0;
#endif
    // NOTE: non-standard notation
    // In matmul all dimensions above 2 are considered as batch
    // In below code that deals with broadcast of batch dimensions,
    // d0 means lowest batch dimension and d3 the highest

    // decompose mb into batch dimensions (d0..d3)
    long d3 = mb / D0 / D1 / D2;
    long d2 = (mb / D0 / D1) % D2;
    long d1 = (mb / D0) % D1;
    long d0 = mb % D0;

    // With groups, compute `k` over each group, and iterate over k_groups.
    // Inside each group, compute acc as `ACC_DATA_T` but once reduction
    // happens, convert to float and apply scales.
    long n_groups_k = K / group_K;

    FLT_ACC_DATA_T acc = 0.f;
    for (long g = 0; g < n_groups_k; g++) {
        ACC_DATA_T acc_g = 0;
        for (long k_g = 0; k_g < group_K; ++k_g) {
            auto k = k_g + g * group_K;
            long src_off = SRC_OFFSET(m, k);
            long wei_off = WEI_OFFSET(k, n);
            int wei_zp = 0;
#if WITH_WEI_ZPOINTS
            long wei_zp_off = wei_zp_stride_n * (n / wei_zp_group_n)
                    + wei_zp_stride_k * (k / wei_zp_group_k)
                    + BATCH_OFF(wei_zp);
#if !WITH_SRC_GROUP_SUMS && !WITH_WEI_SBZP
            wei_zp = WEI_ZP_TO_REF(b0, wei_zp_off);
#endif
#endif
            int src_zp = 0;
#if WITH_SRC_ZPOINTS
            long src_zp_off = src_zp_stride_k * (k / src_zp_group_k)
                    + src_zp_stride_m * m + BATCH_OFF(src_zp);
            src_zp = SRC_ZP_TO_REF(a0, src_zp_off);
#endif
#if SRC_DT_F4_E2M1
            ACC_DATA_T s
                    = TO_ACC(SRC_TO_REF(GET_HALF_BYTE(A, src_off)) - src_zp);
#else
            ACC_DATA_T s = TO_ACC(SRC_TO_REF(A[src_off]) - src_zp);
#endif
#if WEI_DT_U2
            ACC_DATA_T w_raw = WEI_TO_REF(GET_QUARTER_BYTE(B, wei_off));
#elif WEI_DT_U3
            ACC_DATA_T w_raw = WEI_TO_REF(GET_U3(B, wei_off));
#elif WEI_DT_S4 || WEI_DT_U4 || WEI_DT_F4_E2M1
            ACC_DATA_T w_raw = WEI_TO_REF(GET_HALF_BYTE(B, wei_off));
#else
            ACC_DATA_T w_raw = WEI_TO_REF(B[wei_off]);
#endif
#if WITH_WEI_SBZP
            // static_sbzp: w = q * scale - zp, scaled per weight.
            // It's sitting in a separate branch due to wei_zp is not `int`
            // in this branch. Additionally, can't reuse scales from below due
            // to acc_g type would become different.
            long wei_scale_off_k = wei_scale_stride_n * (n / wei_scale_group_n)
                    + wei_scale_stride_k * (k / wei_scale_group_k)
                    + BATCH_OFF(wei_scale);
            ACC_DATA_T w = TO_ACC(
                    w_raw * WEI_SCALES_TO_REF(wei_scales[wei_scale_off_k])
                    - WEI_ZP_TO_REF(b0, wei_zp_off));
#else
            ACC_DATA_T w = TO_ACC(w_raw - wei_zp);
#endif
            acc_g += s * w;
#if WITH_WEI_ZPOINTS && WITH_SRC_GROUP_SUMS
            wei_zp = WEI_ZP_TO_REF(b0, wei_zp_off);
            acc_g += TO_ACC(src_zp * wei_zp);
#endif
        }

        FLT_ACC_DATA_T src_scale = 1.f;
        FLT_ACC_DATA_T wei_scale = 1.f;
#if WITH_SRC_SCALES
        long src_scale_off = src_scale_stride_m * (m / src_scale_group_m)
                + src_scale_stride_k * (g * group_K / src_scale_group_k)
                + BATCH_OFF(src_scale);
        src_scale = SRC_SCALES_TO_REF(src_scales[src_scale_off]);
#endif
#if WITH_WEI_SCALES && !WITH_WEI_SBZP
        long wei_scale_off = wei_scale_stride_n * (n / wei_scale_group_n)
                + wei_scale_stride_k * (g * group_K / wei_scale_group_k)
                + BATCH_OFF(wei_scale);
        wei_scale = WEI_SCALES_TO_REF(wei_scales[wei_scale_off]);
#endif
        FLT_ACC_DATA_T acc_g_to_f = ACC_TO_REF(acc_g) * src_scale * wei_scale;
        acc += acc_g_to_f;
    }
#if WITH_SRC_GROUP_SUMS
    for (int g = 0, gend = K / src_gs_group_k; g < gend; g++) {
        FLT_ACC_DATA_T src_scale = 1.f;
        FLT_ACC_DATA_T wei_scale = 1.f;
#if WITH_SRC_SCALES
        long src_scale_g = g * src_gs_group_k / src_scale_group_k;
        long src_scale_off = src_scale_stride_m * (m / src_scale_group_m)
                + src_scale_stride_k * src_scale_g + BATCH_OFF(src_scale);
        src_scale = SRC_SCALES_TO_REF(src_scales[src_scale_off]);
#endif
#if WITH_WEI_SCALES
        long wei_scale_g = g * src_gs_group_k / wei_scale_group_k;
        long wei_scale_off = wei_scale_stride_n * (n / wei_scale_group_n)
                + wei_scale_stride_k * wei_scale_g + BATCH_OFF(wei_scale);
        wei_scale = WEI_SCALES_TO_REF(wei_scales[wei_scale_off]);
#endif
        long src_gs_off
                = src_gs_stride_m * m + src_gs_stride_k * g + BATCH_OFF(src_gs);
        int src_gs = SRC_ZP_TO_REF(ag, src_gs_off);
        long wei_zp_off = wei_zp_stride_n * (n / wei_zp_group_n)
                + wei_zp_stride_k * (g * src_gs_group_k / wei_zp_group_k)
                + BATCH_OFF(wei_zp);
        int wei_zp = WEI_ZP_TO_REF(b0, wei_zp_off);
        acc -= src_scale * wei_scale * TO_ACC(src_gs) * TO_ACC(wei_zp);
    }
#endif
    long dst_off = DST_OFFSET(m, n);

#if WITH_BIAS || WITH_DROPOUT || NON_DEFAULT_ATTRS
    POST_OP_DATA_T temp = (POST_OP_DATA_T)acc;
#if WITH_BIAS
    long bia_off = offset6D(m, n, d0, d1, d2, d3, bia_stride_m, bia_stride_n,
            bia_stride_d0, bia_stride_d1, bia_stride_d2, bia_stride_d3);
    temp += BIA_TO_REF(bia[bia_off]);
#endif // WITH_BIAS

    float dst_data;
#if WITH_SUM
#if DST_PACKED
    dst_data = SUM_TO_REF(GET_HALF_BYTE(C_sum, dst_off));
#else
    dst_data = SUM_TO_REF(C_sum[dst_off]);
#endif
#endif // WITH_SUM

    float po_acc = convert_float(temp);

#if WITH_DROPOUT
#if WITH_SEED_S64 && USE_OFFSET
    uint res = philox_4x32_w_offset(dst_off, dropout_seed, dropout_offset);
#else
    uint res = philox_4x32(dst_off, dropout_seed);
#endif
    uchar dropout = res > dropout_threshold;
    po_acc = (dropout) ? po_acc * dropout_inv_q : 0;
#if HAS_OUTPUT_MASK
    dropout_mask_buf[dst_off] = dropout;
#endif
#endif

#if WITH_SROUND
    po_acc = stochastic_round_fwd(po_acc, dst_off, sround_seed);
#endif

    if (DST_NDIMS == 2)
        APPLY_POST_OPS_SERIAL(po_acc, dst_data, m, n, 0, 0, 0, 0);
    if (DST_NDIMS == 3)
        APPLY_POST_OPS_SERIAL(po_acc, dst_data, d0, m, n, 0, 0, 0);
    if (DST_NDIMS == 4)
        APPLY_POST_OPS_SERIAL(po_acc, dst_data, d1, d0, m, n, 0, 0);
    if (DST_NDIMS == 5)
        APPLY_POST_OPS_SERIAL(po_acc, dst_data, d2, d1, d0, m, n, 0);
    if (DST_NDIMS == 6)
        APPLY_POST_OPS_SERIAL(po_acc, dst_data, d3, d2, d1, d0, m, n);

#if WITH_DST_SCALES
#if DST_SCALES_MASK == 0
    po_acc /= DST_SCALES_TO_REF(dst_scales[0]);
#elif WITH_DYN_DST_SCALE == 0
    long dst_scale_off = dst_scale_stride_n * (n / dst_scale_group_n)
            + dst_scale_stride_m * (m / dst_scale_group_m)
            + BATCH_OFF(dst_scale);
    po_acc /= DST_SCALES_TO_REF(dst_scales[dst_scale_off]);
#endif
#endif
    po_acc += dst_zp;

#if WITH_DYN_DST_SCALE
    C[dst_off] = po_acc;
#else
    C[dst_off] = TO_DST(po_acc);
#endif
#else // WITH_BIAS || NON_DEFAULT_ATTRS
#if WITH_DYN_DST_SCALE
    C[dst_off] = acc;
#else
    C[dst_off] = TO_DST(acc);
#endif
#endif // WITH_BIAS || NON_DEFAULT_ATTRS
}
