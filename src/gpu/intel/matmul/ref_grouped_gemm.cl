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

#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"

// Grouped GEMM OCL reference kernel
//
// Per group g: C[g] = A[g] * B[g] (+ scales, and bias / zero-points for 2Dx3D)
//
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// partner_offsets buffer is dst_offsets
//
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
// partner_offsets buffer is wei_offsets
//
// Tensors, scales and zero points are accessed by logical indices and strides
// of their [G, rows, cols] views: a 2D tensor has a zero group stride, and an
// index along its variable dim spans all groups
//
// get_global_id(0): group index
// get_global_id(1): output row (m) within the group
// get_global_id(2): output column (n)
//
// Supported:
//  f32/f16/bf16
// For 2Dx2D:
//  f8 src and wei with K-grouped scales, a grid over total_K
// For 2Dx3D only:
//  u8/s8 src, u8/s8/s4/u4/f8/f4 wei (incl. WOQ)
//  row/col-wise, K-grouped or KxN-grouped (wei) scales & zero-points
//  common wei zero-point (u3 wei), incl. host-side scalar
//  bias [group_count, N] and post-ops
__kernel void ref_grouped_gemm_matmul(__global const SRC_DATA_T *src,
        __global const int *src_offsets, __global const WEI_DATA_T *wei,
        __global const int *partner_offsets, __global DST_DATA_T *dst,
        const int group_count, const int is_2dby2d, const long m_fixed,
        const long k_total, const long N, const long k_block,
        const long src_total, const long k_align, const long src_stride_m,
        const long src_stride_k, const long wei_stride_g,
        const long wei_stride_k, const long wei_stride_n,
        const long dst_stride_g, const long dst_stride_m,
        const long dst_stride_n
#if WITH_BIAS
        ,
        __global const BIA_DATA_T *bias, // Bias [group_count, N]
        const long bia_stride_g, const long bia_stride_n
#endif
#if WITH_SRC_SCALES
        ,
        __global const SRC_SCALES_DATA_T *src_scales,
        const long src_scale_stride_m, const long src_scale_stride_k,
        const long src_scale_group_k
#endif
#if WITH_WEI_SCALES
        ,
        __global const WEI_SCALES_DATA_T *wei_scales,
        const long wei_scale_stride_g, const long wei_scale_stride_k,
        const long wei_scale_stride_n, const long wei_scale_group_k,
        const long wei_scale_group_n
#endif
#if WITH_SRC_ZPOINTS
        ,
        __global const SRC_ZP_DATA_T *src_zero_points,
        const long src_zp_stride_m, const long src_zp_stride_k,
        const long src_zp_group_k
#endif
#if WITH_WEI_ZPOINTS
        ,
#if WITH_HOST_WEI_ZP
        WEI_ZP_DATA_T wei_zp_value,
#else
        __global const WEI_ZP_DATA_T *wei_zero_points,
#endif
        const long wei_zp_stride_g, const long wei_zp_stride_k,
        const long wei_zp_stride_n, const long wei_zp_group_k,
        const long wei_zp_group_n
#endif
                POST_OP_ARGS) {
#if WITH_HOST_WEI_ZP
    const WEI_ZP_DATA_T *wei_zero_points = &wei_zp_value;
#endif
    const off_t group_id = get_global_id(0);
    const long m = get_global_id(1);
    const long n = get_global_id(2);

    if (group_id >= group_count) return;
    if (n >= N) return;

    // Groups with invalid offsets are skipped: the tensor split along with src
    // shares the src offsets, and 2Dx2D groups align with the coarsest scales
    // K-group
    const long start = (group_id == 0) ? 0 : src_offsets[group_id - 1];
    const long end = src_offsets[group_id];
    if (start < 0 || end < start || end > src_total || start % k_align != 0
            || end % k_align != 0 || partner_offsets[group_id] != end
            || (group_id > 0 && partner_offsets[group_id - 1] != start))
        return;

    // The group starts at `start` along the variable dim: M for 2Dx3D or K
    // for 2Dx2D
    const long M_g = is_2dby2d ? m_fixed : end - start;
    const long K_g = is_2dby2d ? end - start : k_total;
    const long m_abs = is_2dby2d ? m : start + m;
    const long k_start = is_2dby2d ? start : 0;

    if (m >= M_g) return;

    // Note, that K_g == 0 must still write zeros
    const long k_group_size = min(k_block, K_g);
    const long n_k_groups = K_g ? K_g / k_group_size : 0;

    FLT_ACC_DATA_T acc = 0;
    for (long i_group = 0; i_group < n_k_groups; i_group++) {
        // Scales and zero points are the same across the K-group
        const long k_group_start = k_start + i_group * k_group_size;
        int src_zp = 0;
#if WITH_SRC_ZPOINTS
        src_zp = SRC_ZP_TO_REF(src_zero_points,
                m_abs * src_zp_stride_m
                        + k_group_start / src_zp_group_k * src_zp_stride_k);
#endif
        int wei_zp = 0;
#if WITH_WEI_ZPOINTS
        wei_zp = WEI_ZP_TO_REF(wei_zero_points,
                group_id * wei_zp_stride_g
                        + k_group_start / wei_zp_group_k * wei_zp_stride_k
                        + n / wei_zp_group_n * wei_zp_stride_n);
#endif
        ACC_DATA_T acc_g = (ACC_DATA_T)0;
        for (long k = k_group_start; k < k_group_start + k_group_size; k++) {
            const long src_idx = m_abs * src_stride_m + k * src_stride_k;
            const long wei_idx = group_id * wei_stride_g + k * wei_stride_k
                    + n * wei_stride_n;
#if SRC_DT_F4_E2M1
            ACC_DATA_T s
                    = TO_ACC(SRC_TO_REF(GET_HALF_BYTE(src, src_idx)) - src_zp);
#else
            ACC_DATA_T s = TO_ACC(SRC_TO_REF(src[src_idx]) - src_zp);
#endif
#if WEI_DT_U3
            ACC_DATA_T w = TO_ACC(WEI_TO_REF(GET_U3(wei, wei_idx)) - wei_zp);
#elif WEI_DT_S4 || WEI_DT_U4 || WEI_DT_F4_E2M1
            ACC_DATA_T w
                    = TO_ACC(WEI_TO_REF(GET_HALF_BYTE(wei, wei_idx)) - wei_zp);
#else
            ACC_DATA_T w = TO_ACC(WEI_TO_REF(wei[wei_idx]) - wei_zp);
#endif
            acc_g += s * w;
        }
        FLT_ACC_DATA_T src_scale = 1.f;
        FLT_ACC_DATA_T wei_scale = 1.f;
#if WITH_SRC_SCALES
        src_scale = SRC_SCALES_TO_REF(src_scales[m_abs * src_scale_stride_m
                + k_group_start / src_scale_group_k * src_scale_stride_k]);
#endif
#if WITH_WEI_SCALES
        wei_scale = WEI_SCALES_TO_REF(wei_scales[group_id * wei_scale_stride_g
                + k_group_start / wei_scale_group_k * wei_scale_stride_k
                + n / wei_scale_group_n * wei_scale_stride_n]);
#endif
        acc += ACC_TO_REF(acc_g) * src_scale * wei_scale;
    }

#if WITH_BIAS
    acc += BIA_TO_REF(bias[group_id * bia_stride_g + n * bia_stride_n]);
#endif

    // Post-ops apply to the 2Dx3D pattern only
    POST_OP_DATA_T po_acc = acc;
    POST_OP_DATA_T sum_src = 0;
    APPLY_POST_OPS_SERIAL(po_acc, sum_src, group_id, m_abs, n, 0, 0, 0);
    acc = po_acc;

    const long dst_idx
            = group_id * dst_stride_g + m_abs * dst_stride_m + n * dst_stride_n;
    dst[dst_idx] = REF_TO_DST(acc);
}
