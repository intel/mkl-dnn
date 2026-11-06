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

#define USE_CUSTOM_GWS_GET_ID

// Temporary W/A for bf16 problems in HW and compiler
#undef cl_future_bf16_cvt

#include "gpu/intel/include/dispatch.h"
#include "gpu/intel/include/philox.h"
#include "gpu/intel/include/types_interop.h"
#include "gpu/intel/reorder/common.h"

#define FROM_I2 SRC_DT_U2
#define FROM_I4 (SRC_DT_U4 || SRC_DT_S4)
#define GWS_GET_THREAD_ID(index) \
    (off_t)(get_global_id(index) + offset.array[index])

KERNEL_ATTR
#if SUB_GROUP_SIZE != 1
__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
#endif
__kernel void
ref_reorder(__global SRC_DATA_T *restrict src,
        __global DST_DATA_T *restrict dst,
        __global SRC_SCALES_DATA_T *restrict src_scales,
        __global SRC_ZP_DATA_T *restrict src_zps,
        __global DST_SCALES_DATA_T *restrict dst_scales, __global int *dst_zps,
        float sum_scale, int sum_zp,
#if WITH_SROUND
        __global uint *sround_seed_buf,
#endif
        int64x3_t offset) {

    int src_zp = 0;
    const int dst_zp = GET_DST_ZP(dst_zps);
    float src_scale = 1.0f;
    float dst_scale = 1.0f;

    src += SRC_OFFSET0;
    dst += DST_OFFSET0;

    const off_t d0_blk_start = GWS_GET_D0();
    const off_t d1_blk_start = GWS_GET_D1();
    const off_t d2_blk_start = GWS_GET_D2();
    const off_t d3_blk_start = GWS_GET_D3();
    const off_t d4_blk_start = GWS_GET_D4();
    const off_t d5_blk_start = GWS_GET_D5();

    const off_t d0_blk_end = d0_blk_start + GWS_GET_D0_BLOCK();
    const off_t d1_blk_end = d1_blk_start + GWS_GET_D1_BLOCK();
    const off_t d2_blk_end = d2_blk_start + GWS_GET_D2_BLOCK();
    const off_t d3_blk_end = d3_blk_start + GWS_GET_D3_BLOCK();
    const off_t d4_blk_end = d4_blk_start + GWS_GET_D4_BLOCK();
    const off_t d5_blk_end = d5_blk_start + GWS_GET_D5_BLOCK();

#if WITH_SROUND
    const uint sround_seed = *sround_seed_buf;
#endif

    for_(off_t d0 = d0_blk_start; d0 < d0_blk_end; ++d0)
    for_(off_t d1 = d1_blk_start; d1 < d1_blk_end; ++d1)
    for_(off_t d2 = d2_blk_start; d2 < d2_blk_end; ++d2)
    for_(off_t d3 = d3_blk_start; d3 < d3_blk_end; ++d3)
    for_(off_t d4 = d4_blk_start; d4 < d4_blk_end; ++d4)
    for (off_t d5 = d5_blk_start; d5 < d5_blk_end; ++d5) {
        const off_t src_off = SRC_OFF(d0, d1, d2, d3, d4, d5);
        const off_t dst_off = DST_OFF(d0, d1, d2, d3, d4, d5);
#if PAD_FILL_ZERO == 1
        int pad_d0 = d0 >= SRC_D0;
        int pad_d1 = NDIMS > 1 && d1 >= SRC_D1;
        int pad_d2 = NDIMS > 2 && d2 >= SRC_D2;
        int pad_d3 = NDIMS > 3 && d3 >= SRC_D3;
        int pad_d4 = NDIMS > 4 && d4 >= SRC_D4;
        int pad_d5 = NDIMS > 5 && d5 >= SRC_D5;
        if (pad_d0 || pad_d1 || pad_d2 || pad_d3 || pad_d4 || pad_d5) {
            dst[dst_off] = TO_DST(0);
            continue;
        }
#endif
        // Both scales and zero-points include groups in their offsets.
        // It involves division by a group value of a correspondent dX value,
        // and also adjusting stride for a dimension with a group.
#if WITH_SRC_ZPOINT
        off_t zp_off = ZPOINT_OFF(SRC, d0, d1, d2, d3, d4, d5);
        src_zp = SRC_ZP_TO_REF(src_zps, zp_off);
#endif
#if WITH_SRC_SCALE
        off_t scale_off = SCALE_OFF(SRC, d0, d1, d2, d3, d4, d5);
        src_scale = SRC_SCALES_TO_REF(src_scales[scale_off]);
#endif
#if WITH_DST_SCALE
        off_t dst_scale_off = SCALE_OFF(DST, d0, d1, d2, d3, d4, d5);
        dst_scale = DST_SCALES_TO_REF(dst_scales[dst_scale_off]);
#endif
#if FROM_I2
        SRC_DATA_T src_value = GET_QUARTER_BYTE(src, src_off);
#elif FROM_I4 || SRC_DT_F4_E2M1
        SRC_DATA_T src_value = GET_HALF_BYTE(src, src_off);
#else
        SRC_DATA_T src_value = src[src_off];
#endif
#if WITH_SROUND
#define ROUND(f) stochastic_round_fwd(f, dst_off, sround_seed)
#else
#define ROUND DEFAULT_ROUND
#endif

        REORDER(ROUND, dst[dst_off], src_value, src_scale, dst_scale, sum_scale,
                src_zp, dst_zp, sum_zp);
    }
}
