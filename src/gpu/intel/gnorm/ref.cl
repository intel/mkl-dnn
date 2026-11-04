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

#include "gpu/intel/include/dispatch.h"
#include "gpu/intel/include/post_ops.h"
#include "gpu/intel/include/types.h"

// define data type for auxilary data like mean, variance, shift, scale and etc
#define AUX_DATA_T float

// kernel accumulator datatype.
// DEF_ACC_DATA_T can not be used - it is "int" for small datatypes
#define GNORM_ACC POST_OP_DATA_T
// accumulator type specific constants
#define GNORM_ACC_CONST_0 0.0f
#define GNORM_ACC_CONST_1 1.0f

#if IS_FWD

/**
 * @ingroup OCL_KERNELS
 * @brief Group Normalization reference implementation. Forward case.
 *
 * Reference implementation of the Group Normalization algorithm.
 * This is forward case.
 * Paralelized with "batch" and "group" dimentions
 *
 * @param [in]      src       Main input data array [batch, channel, depth, height, width]
 * @param [in,out]  mean      Mean array [batch, group]
 * @param [in,out]  variance  Variance array [batch, group]
 * @param [out]     dst       Main output data array [batch, channel, depth, height, width]
 * @param [in]      scale     Scale vector [channel]
 * @param [in]      shift     Shift vector [channel]
 * @param [in]      src_scale Input data scale [1]
 * @param [in]      dst_scale Output data scale [1]
 * @param [in]      eps       User defined constant
 *
 * POST_OP_ARGS Expands to ', const __global POST_OP_TYPE *po_0_binary_arg' where POST_OP_TYPE
 * defined by post operation type API
 *
 */
KERNEL_ATTR
__kernel void ref_gnorm_fwd(__global const SRC_DATA_T *src,
        __global AUX_DATA_T *mean, __global AUX_DATA_T *variance,
        __global DST_DATA_T *dst, dim_t C_PER_G,
        __global const AUX_DATA_T *scale, __global const AUX_DATA_T *shift,
        __global const AUX_DATA_T *src_scale,
        __global const AUX_DATA_T *dst_scale,
        const AUX_DATA_T eps POST_OP_ARGS) {

    // get parallel variables IDs
    const size_t id_batch = GWS_GET_BATCH();
    const size_t id_group = GWS_GET_NGROUPS();

    const size_t channel_start = id_group * C_PER_G;
    const size_t channel_end = channel_start + C_PER_G;

    // data conversion
    const GNORM_ACC stat_divisor = C_PER_G * D * H * W;

    const size_t stat_off = id_batch * G + id_group;

    GNORM_ACC mean_val = CALCULATE_STATS ? GNORM_ACC_CONST_0 : mean[stat_off];
    if (CALCULATE_STATS) {
        for_(size_t channel = channel_start; channel < channel_end; ++channel)
        for_(size_t depth = 0; depth < D; ++depth)
        for_(size_t height = 0; height < H; ++height)
        for (size_t width = 0; width < W; ++width) {
            const size_t idx = SRC_OFF(id_batch, channel, depth, height, width);
            const GNORM_ACC input_val = SRC_TO_REF(src[idx]);
            mean_val += input_val;
        }
        mean_val /= stat_divisor;

        if (SAVE_STATS) mean[stat_off] = mean_val;
    }

    GNORM_ACC variance_val
            = CALCULATE_STATS ? GNORM_ACC_CONST_0 : variance[stat_off];
    if (CALCULATE_STATS) {
        for_(size_t channel = channel_start; channel < channel_end; ++channel)
        for_(size_t depth = 0; depth < D; ++depth)
        for_(size_t height = 0; height < H; ++height)
        for (size_t width = 0; width < W; ++width) {
            const size_t idx = SRC_OFF(id_batch, channel, depth, height, width);
            const GNORM_ACC input_val = SRC_TO_REF(src[idx]);
            const GNORM_ACC var1 = input_val - mean_val;
            variance_val += var1 * var1;
        }
        variance_val /= stat_divisor; // use extra division to keep accuracy

        if (SAVE_STATS) variance[stat_off] = variance_val;
    }

    const GNORM_ACC src_scale_val
            = WITH_SRC_SCALES ? src_scale[0] : GNORM_ACC_CONST_1;
    const GNORM_ACC dst_scale_val
            = WITH_DST_SCALES ? dst_scale[0] : GNORM_ACC_CONST_1;
    const GNORM_ACC r_dst_scale_val = GNORM_ACC_CONST_1 / dst_scale_val;

    const GNORM_ACC variance_rsqrt = rsqrt(variance_val + eps);

    // Unroll hint is to avoid register spilling. For real workloads, D,H, and W
    // should be large enough that unrolling the outer loop is pointless from a
    // performance perspective.
    unroll_for_by(1)(size_t channel = channel_start; channel < channel_end;
                     ++channel) {
        const GNORM_ACC scale_val = scale ? scale[channel] : GNORM_ACC_CONST_1;
        const GNORM_ACC shift_val = shift ? shift[channel] : GNORM_ACC_CONST_0;

        for_(size_t depth = 0; depth < D; ++depth)
        for_(size_t height = 0; height < H; ++height)
        for (size_t width = 0; width < W; ++width) {
            const size_t idx = SRC_OFF(id_batch, channel, depth, height, width);
            const GNORM_ACC input_val = SRC_TO_REF(src[idx]);
            GNORM_ACC result_val = (input_val - mean_val) * variance_rsqrt;

            result_val = scale_val * result_val + shift_val;

            result_val *= src_scale_val;

            // post-op operation
            GNORM_ACC post_op_acc
                    = WITH_SUM ? DST_TO_REF(dst[idx]) : GNORM_ACC_CONST_0;

// the macro changes meaning of the input parameters with diffrent ndims
#if NDIMS == 3
            APPLY_POST_OPS_SERIAL(
                    result_val, post_op_acc, id_batch, channel, width, 0, 0, 0);
#elif NDIMS == 4
            APPLY_POST_OPS_SERIAL(result_val, post_op_acc, id_batch, channel,
                    height, width, 0, 0);
#elif NDIMS == 5
            APPLY_POST_OPS_SERIAL(result_val, post_op_acc, id_batch, channel,
                    depth, height, width, 0);
#else
            APPLY_POST_OPS_SERIAL(
                    result_val, post_op_acc, id_batch, channel, 0, 0, 0, 0);
#endif
            result_val *= r_dst_scale_val;

            dst[idx] = TO_DST(result_val);
        }
    }
}

#else // !IS_FWD

/**
 * @ingroup OCL_KERNELS
 * @brief Group Normalization reference implementation. Backward case.
 *
 * Reference implementation of the Group Normalization algorithm.
 * This is backward case.
 * Paralelized with "channel" dimention
 *
 * @param [in]      src             Main input data array [batch, channel, depth, height, width]
 * @param [in,out]  mean            Mean array [batch, group]
 * @param [in,out]  variance        Variance array [batch, group]
 * @param [in]      diff_dst        Main input weights array [batch, channel, depth, height, width]
 * @param [in]      scale           Scale vector [channel]
 * @param [out]     out_diff_src    Main output data array [batch, channel, depth, height, width]
 * @param [out]     out_diff_scale  Calculated scale vactor [channel]
 * @param [out]     out_diff_shift  Calculated shift vactor [channel]
 * @param [in]      eps             User defined constant
 *
 */
KERNEL_ATTR
__kernel void ref_gnorm_bwd(__global const SRC_DATA_T *src,
        __global AUX_DATA_T *mean, __global AUX_DATA_T *variance,
        __global const DST_DATA_T *diff_dst, __global const AUX_DATA_T *scale,
        __global SRC_DATA_T *out_diff_src, __global AUX_DATA_T *out_diff_scale,
        __global AUX_DATA_T *out_diff_shift, dim_t C_PER_G,
        const AUX_DATA_T eps) {

    // get parallel variables IDs
    const size_t id_group = GWS_GET_NGROUPS();

    // data conversion
    const GNORM_ACC CSP = C_PER_G * D * H * W;

    const size_t channel_start = id_group * C_PER_G;
    const size_t channel_end = channel_start + C_PER_G;

    // Pass 1: Compute diff_scale and diff_shift for each channel.
    for (size_t channel = channel_start; channel < channel_end; ++channel) {
        GNORM_ACC diff_scale = GNORM_ACC_CONST_0;
        GNORM_ACC diff_shift = GNORM_ACC_CONST_0;

        for (size_t batch = 0; batch < MB; ++batch) {
            const size_t idx_stat = batch * G + id_group;
            const GNORM_ACC mean_val = mean[idx_stat];
            const GNORM_ACC variance_val = variance[idx_stat];
            const GNORM_ACC variance_recip = rsqrt(variance_val + eps);

            for_(size_t depth = 0; depth < D; ++depth)
            for_(size_t height = 0; height < H; ++height)
            for (size_t width = 0; width < W; ++width) {
                const size_t idx
                        = SRC_OFF(batch, channel, depth, height, width);
                const GNORM_ACC src_val = SRC_TO_REF(src[idx]);
                const GNORM_ACC diff_dst_val = DST_TO_REF(diff_dst[idx]);

                diff_scale
                        += (src_val - mean_val) * diff_dst_val * variance_recip;
                diff_shift += diff_dst_val;
            }
        }

        if (out_diff_scale) out_diff_scale[channel] = diff_scale;
        if (out_diff_shift) out_diff_shift[channel] = diff_shift;
    }

    // Pass 2: Compute diff_src.
    for (size_t batch = 0; batch < MB; ++batch) {
        const size_t idx_stat = batch * G + id_group;
        const GNORM_ACC mean_val = mean[idx_stat];
        const GNORM_ACC variance_val = variance[idx_stat];
        const GNORM_ACC variance_recip = rsqrt(variance_val + eps);

        GNORM_ACC sum_dd_scaled = GNORM_ACC_CONST_0;
        GNORM_ACC sum_dd_snorm = GNORM_ACC_CONST_0;

        if (CALCULATE_STATS) {
            for (size_t channel = channel_start; channel < channel_end;
                    ++channel) {
                const GNORM_ACC scale_val
                        = scale ? scale[channel] : GNORM_ACC_CONST_1;

                for_(size_t depth = 0; depth < D; ++depth)
                for_(size_t height = 0; height < H; ++height)
                for (size_t width = 0; width < W; ++width) {
                    const size_t idx
                            = SRC_OFF(batch, channel, depth, height, width);
                    const GNORM_ACC src_val = SRC_TO_REF(src[idx]);
                    const GNORM_ACC diff_dst_val = DST_TO_REF(diff_dst[idx]);

                    const GNORM_ACC dd_scaled = diff_dst_val * scale_val;
                    const GNORM_ACC s_norm
                            = (src_val - mean_val) * variance_recip;

                    sum_dd_scaled += dd_scaled;
                    sum_dd_snorm += dd_scaled * s_norm;
                }
            }
        }

        const GNORM_ACC mean_dd_scaled = sum_dd_scaled / CSP;
        const GNORM_ACC mean_dd_snorm = sum_dd_snorm / CSP;

        for (size_t channel = channel_start; channel < channel_end; ++channel) {
            const GNORM_ACC scale_val
                    = scale ? scale[channel] : GNORM_ACC_CONST_1;

            for_(size_t depth = 0; depth < D; ++depth)
            for_(size_t height = 0; height < H; ++height)
            for (size_t width = 0; width < W; ++width) {
                const size_t idx
                        = SRC_OFF(batch, channel, depth, height, width);
                const GNORM_ACC src_val = SRC_TO_REF(src[idx]);
                GNORM_ACC dd_scaled = DST_TO_REF(diff_dst[idx]) * scale_val;

                if (CALCULATE_STATS) {
                    const GNORM_ACC s_norm
                            = (src_val - mean_val) * variance_recip;
                    dd_scaled -= mean_dd_scaled + s_norm * mean_dd_snorm;
                }

                out_diff_src[idx] = TO_SRC(dd_scaled * variance_recip);
            }
        }
    }
}

#endif // !IS_FWD
