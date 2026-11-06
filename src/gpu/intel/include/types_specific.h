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

// See ocl_types.h for details about macros naming and organization.

#ifndef GPU_INTEL_INCLUDE_TYPES_SPECIFIC_H
#define GPU_INTEL_INCLUDE_TYPES_SPECIFIC_H

#ifdef SRC_DATA_T
#define SRC_DATA2_T CONCAT2(SRC_DATA_T, 2)
#define SRC_DATA4_T CONCAT2(SRC_DATA_T, 4)
#define SRC_DATA8_T CONCAT2(SRC_DATA_T, 8)
#define SRC_DATA16_T CONCAT2(SRC_DATA_T, 16)

#define AS_SRC_DATA_T CONCAT2(as_, SRC_DATA_T)
#define AS_SRC_DATA2_T CONCAT2(as_, SRC_DATA2_T)
#define AS_SRC_DATA4_T CONCAT2(as_, SRC_DATA4_T)
#define AS_SRC_DATA8_T CONCAT2(as_, SRC_DATA8_T)
#define AS_SRC_DATA16_T CONCAT2(as_, SRC_DATA16_T)
#if SRC_DT_U8
#define SRC_TO_REF(x) convert_float(x)
#define SRC_TO_REF8(x) convert_float8(x)
#define REF_TO_SRC(x) convert_uchar(x)
#elif SRC_DT_S8
#define SRC_TO_REF(x) convert_float(x)
#define SRC_TO_REF8(x) convert_float8(x)
#define REF_TO_SRC(x) convert_char(x)
#elif SRC_DT_BF16
#define SRC_TO_REF(x) into_float(x)
#define SRC_TO_REF8(x) into_float(x)
#define REF_TO_SRC(x) into_bf16(x)
#elif SRC_DT_F16
#define SRC_TO_REF(x) convert_float(x)
#define SRC_TO_REF8(x) convert_float8(x)
#define REF_TO_SRC(x) convert_half(x)
#elif SRC_DT_BF8
#define SRC_TO_REF(x) into_float(x)
#define SRC_TO_REF8(x) into_float(x)
#define REF_TO_SRC(x) into_f8_e5m2(x)
#elif SRC_DT_HF8
#define SRC_TO_REF(x) into_float(x)
#define SRC_TO_REF8(x) into_float(x)
#define REF_TO_SRC(x) into_f8_e4m3(x)
#elif SRC_DT_F4_E2M1
#define SRC_TO_REF(x) into_float(x)
#define SRC_TO_REF8(x) into_float(x)
#define REF_TO_SRC(x) into_f4_e2m1(x)
#elif SRC_DT_E8M0
#define SRC_TO_REF(x) into_float(x)
#define SRC_TO_REF8(x) into_float(x)
#define REF_TO_SRC(x) into_e8m0(x)
#elif SRC_DT_U4
#define SRC_TO_REF(x) into_float(x)
#elif SRC_DT_S4
#define SRC_TO_REF(x) into_float(x)
#elif SRC_DT_U2
#define SRC_TO_REF(x) into_float(x)
#else
#define SRC_TO_REF(x) (x)
#define SRC_TO_REF8(x) (x)
#define REF_TO_SRC(x) (x)
#endif
#if SRC_DT_BF16
#define TO_SRC(x) into_bf16(x)
#elif SRC_DT_F16
#define TO_SRC(x) convert_half(x)
#elif SRC_DT_BF8
#define TO_SRC(x) into_f8_e5m2(x)
#elif SRC_DT_HF8
#define TO_SRC(x) into_f8_e4m3(x)
#elif SRC_DT_F4_E2M1
#define TO_SRC(x) into_f4_e2m1(x)
#elif SRC_DT_U8
#define TO_SRC(x) convert_uchar_sat_rte(x)
#elif SRC_DT_S8
#define TO_SRC(x) convert_char_sat_rte(x)
#elif SRC_DT_S32
#define TO_SRC(x) convert_int_sat_rte(x)
#else
#define TO_SRC(x) (x)
#endif
#endif

#ifdef A_DATA_T
#if A_DT_BF16
#define A_TO_REF(x) into_float(x)
#elif A_DT_BF8
#define A_TO_REF(x) into_half(x)
#elif A_DT_HF8
#define A_TO_REF(x) into_half(x)
#elif A_DT_F4_E2M1
#define A_TO_REF(x) into_float(x)
#else
#define A_TO_REF(x) (x)
#endif
#endif

#ifdef WEI_DATA_T
#if WEI_DT_BF16
#define WEI_TO_REF(x) into_float(x)
#define REF_TO_WEI(x) into_bf16(x)
#elif WEI_DT_BF8
#define WEI_TO_REF(x) into_float(x)
#define REF_TO_WEI(x) into_f8_e5m2(x)
#elif WEI_DT_HF8
#define WEI_TO_REF(x) into_float(x)
#define REF_TO_WEI(x) into_f8_e4m3(x)
#elif WEI_DT_F4_E2M1
#define WEI_TO_REF(x) into_float(x)
#define REF_TO_WEI(x) into_f4_e2m1(x)
#elif WEI_DT_S8
#define WEI_TO_REF(x) convert_int_sat_rte(x)
#define REF_TO_WEI(x) convert_char_sat_rte(x)
#elif WEI_DT_U8
#define WEI_TO_REF(x) convert_int_sat_rte(x)
#define REF_TO_WEI(x) convert_uchar_sat_rte(x)
#elif WEI_DT_S4
#define WEI_TO_REF(x) cvt_s4_to_s32(x)
#elif WEI_DT_U4
#define WEI_TO_REF(x) cvt_u4_to_s32(x)
#elif WEI_DT_U2
#define WEI_TO_REF(x) cvt_u2_to_s32(x)
#elif WEI_DT_U3
#define WEI_TO_REF(x) cvt_u3_to_s32(x)
#else
#define WEI_TO_REF(x) (x)
#define REF_TO_WEI(x) (x)
#endif
#endif

#ifdef DIFF_WEI_DATA_T
#if DIFF_WEI_DT_BF16
#define TO_DIFF_WEI(x) into_bf16(x)
#elif DIFF_WEI_DT_U8
#define TO_DIFF_WEI(x) convert_uchar_sat_rte(x)
#elif DIFF_WEI_DT_S8
#define TO_DIFF_WEI(x) convert_char_sat_rte(x)
#elif DIFF_WEI_DT_S32
#define TO_DIFF_WEI(x) convert_int_sat_rte(x)
#else
#define TO_DIFF_WEI(x) (x)
#endif
#endif

// f16/bf16 scale-shift support for normalization primitives
#ifdef WEI_DATA_T
#if WEI_DT_F32 == 1
#define AS_WEI_DATA_T as_float
#define BLOCK_WEI_READ intel_sub_group_block_read
#define BLOCK_WEI_WRITE intel_sub_group_block_write
#define BLOCK_WEI_DATA_T uint
#define AS_BLOCK_WEI_DATA_T as_uint
#define CONVERT_WEI_FLOAT_T convert_float
#define CONVERT_WEI_DATA_T convert_float
#elif WEI_DT_F16 == 1
#define AS_WEI_DATA_T as_half
#define BLOCK_WEI_READ intel_sub_group_block_read_us
#define BLOCK_WEI_WRITE intel_sub_group_block_write_us
#define BLOCK_WEI_DATA_T ushort
#define AS_BLOCK_WEI_DATA_T as_ushort
#define CONVERT_WEI_FLOAT_T convert_float
#define CONVERT_WEI_DATA_T convert_half
#elif WEI_DT_BF16 == 1
#define AS_WEI_DATA_T as_bf16
#define BLOCK_WEI_READ intel_sub_group_block_read_us
#define BLOCK_WEI_WRITE intel_sub_group_block_write_us
#define BLOCK_WEI_DATA_T ushort
#define AS_BLOCK_WEI_DATA_T(v) (v).data
#define CONVERT_WEI_FLOAT_T into_float
#define CONVERT_WEI_DATA_T into_bf16
#endif
#if VECT_DT_N == 1
#define VECT_BLOCK_WEI_READ BLOCK_WEI_READ
#define VECT_BLOCK_WEI_WRITE BLOCK_WEI_WRITE
#define AS_VECT_WEI_DATA_T AS_WEI_DATA_T
#define AS_VECT_BLOCK_WEI_DATA_T AS_BLOCK_WEI_DATA_T
#define CONVERT_VECT_WEI_FLOAT_T CONVERT_WEI_FLOAT_T
#define CONVERT_VECT_WEI_DATA_T CONVERT_WEI_DATA_T
#else
#define VECT_BLOCK_WEI_READ CONCAT2(BLOCK_WEI_READ, VECT_DT_N)
#define VECT_BLOCK_WEI_WRITE CONCAT2(BLOCK_WEI_WRITE, VECT_DT_N)
#if WEI_DT_BF16 == 1
#define AS_VECT_WEI_DATA_T CONCAT2(as_ushort, VECT_DT_N)
#define AS_VECT_BLOCK_WEI_DATA_T CONCAT2(as_ushort, VECT_DT_N)
#define CONVERT_VECT_WEI_FLOAT_T cvt_bf16_to_f32
#define CONVERT_VECT_WEI_DATA_T cvt_f32_to_bf16
#else
#define AS_VECT_WEI_DATA_T CONCAT2(AS_WEI_DATA_T, VECT_DT_N)
#define AS_VECT_BLOCK_WEI_DATA_T CONCAT2(AS_BLOCK_WEI_DATA_T, VECT_DT_N)
#define CONVERT_VECT_WEI_FLOAT_T CONCAT2(CONVERT_WEI_FLOAT_T, VECT_DT_N)
#define CONVERT_VECT_WEI_DATA_T CONCAT2(CONVERT_WEI_DATA_T, VECT_DT_N)
#endif
#endif
#define LOAD_VECT_WEI(ptr) \
    CONVERT_VECT_WEI_FLOAT_T(AS_VECT_WEI_DATA_T( \
            VECT_BLOCK_WEI_READ((const __global BLOCK_WEI_DATA_T *)(ptr))))
#define SAVE_VECT_WEI(ptr, val) \
    VECT_BLOCK_WEI_WRITE((__global BLOCK_WEI_DATA_T *)(ptr), \
            AS_VECT_BLOCK_WEI_DATA_T(CONVERT_VECT_WEI_DATA_T(val)))
#endif

#ifdef B_DATA_T
#if B_DT_BF16
#define B_TO_REF(x) into_float(x)
#elif B_DT_BF8
#define B_TO_REF(x) into_half(x)
#elif B_DT_HF8
#define B_TO_REF(x) into_half(x)
#elif B_DT_F4_E2M1
#define B_TO_REF(x) into_float(x)
#elif B_DT_U8
#define B_TO_REF(x) (x)
#elif B_DT_S8
#define B_TO_REF(x) (x)
#elif B_DT_S32
#define B_TO_REF(x) (x)
#else
#define B_TO_REF(x) (x)
#endif
#endif

#ifdef BIA_DATA_T
#if BIA_DT_BF16
#define BIA_TO_REF(x) into_float(x)
#define REF_TO_BIA(x) into_bf16(x)
#elif BIA_DT_BF8
#define BIA_TO_REF(x) into_float(x)
#define REF_TO_BIA(x) into_f8_e5m2(x)
#elif BIA_DT_HF8
#define BIA_TO_REF(x) into_float(x)
#define REF_TO_BIA(x) into_f8_e4m3(x)
#elif BIA_DT_F4_E2M1
#define BIA_TO_REF(x) into_float(x)
#define REF_TO_BIA(x) into_f4_e2m1(x)
#else
#define BIA_TO_REF(x) (x)
#define REF_TO_BIA(x) (x)
#endif

#if BIA_DT_BF16
#define TO_BIA(x) into_bf16(x)
#elif BIA_DT_BF8
#define TO_BIA(x) into_f8_e5m2(x)
#elif BIA_DT_HF8
#define TO_BIA(x) into_f8_e4m3(x)
#elif BIA_DT_F4_E2M1
#define TO_BIA(x) into_f4_e2m1(x)
#elif BIA_DT_U8
#define TO_BIA(x) convert_uchar_sat_rte(x)
#elif BIA_DT_S8
#define TO_BIA(x) convert_char_sat_rte(x)
#elif BIA_DT_S32
#define TO_BIA(x) convert_int_sat_rte(x)
#else
#define TO_BIA(x) (x)
#endif
#endif

#ifdef DST_DATA_T
#define DST_DATA2_T CONCAT2(DST_DATA_T, 2)
#define DST_DATA4_T CONCAT2(DST_DATA_T, 4)
#define DST_DATA8_T CONCAT2(DST_DATA_T, 8)
#define DST_DATA16_T CONCAT2(DST_DATA_T, 16)

#define AS_DST_DATA_T CONCAT2(as_, DST_DATA_T)
#define AS_DST_DATA2_T CONCAT2(as_, DST_DATA2_T)
#define AS_DST_DATA4_T CONCAT2(as_, DST_DATA4_T)
#define AS_DST_DATA8_T CONCAT2(as_, DST_DATA8_T)
#define AS_DST_DATA16_T CONCAT2(as_, DST_DATA16_T)

#if DST_DT_F32 || DST_DT_F16
#define CONVERT_DST_DATA_T CONCAT2(convert_, DST_DATA_T)
#define CONVERT_DST_DATA2_T CONCAT2(convert_, DST_DATA2_T)
#define CONVERT_DST_DATA4_T CONCAT2(convert_, DST_DATA4_T)
#define CONVERT_DST_DATA8_T CONCAT2(convert_, DST_DATA8_T)
#define CONVERT_DST_DATA16_T CONCAT2(convert_, DST_DATA16_T)
#elif DST_DT_BF16
#define CONVERT_DST_DATA_T TO_DST
#define CONVERT_DST_DATA2_T TO_DST2
#define CONVERT_DST_DATA4_T TO_DST4
#define CONVERT_DST_DATA8_T TO_DST8
#define CONVERT_DST_DATA16_T TO_DST16
#else
#define CONVERT_DST_DATA_T CONCAT3(convert_, DST_DATA_T, _sat_rte)
#define CONVERT_DST_DATA2_T CONCAT3(convert_, DST_DATA2_T, _sat_rte)
#define CONVERT_DST_DATA4_T CONCAT3(convert_, DST_DATA4_T, _sat_rte)
#define CONVERT_DST_DATA8_T CONCAT3(convert_, DST_DATA8_T, _sat_rte)
#define CONVERT_DST_DATA16_T CONCAT3(convert_, DST_DATA16_T, _sat_rte)
#endif

// Block read/write macros for dst.
#if DST_DT_U8 || DST_DT_S8
#define BLOCK_READ_DST(ptr) \
    AS_DST_DATA2_T(intel_sub_group_block_read_uc2((__global uchar *)ptr))
#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write_uc((__global uchar *)ptr, as_uchar(v))

#define BLOCK_READ_DST2(ptr) \
    AS_DST_DATA2_T(intel_sub_group_block_read_uc2((__global uchar *)ptr))
#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write_uc2((__global uchar *)ptr, as_uchar2(v))

#define BLOCK_READ_DST4(ptr) \
    AS_DST_DATA4_T(intel_sub_group_block_read_uc4((__global uchar *)ptr))
#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write_uc4((__global uchar *)ptr, as_uchar4(v))

#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write_uc8((__global uchar *)ptr, as_uchar8(v))

#define BLOCK_WRITE_DST16(ptr, v) \
    intel_sub_group_block_write_uc16((__global uchar *)ptr, as_uchar16(v))

#elif DST_DT_F16
#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write_us((__global ushort *)ptr, as_ushort(v))

#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write_us2((__global ushort *)ptr, as_ushort2(v))

#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write_us4((__global ushort *)ptr, as_ushort4(v))

#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write_us8((__global ushort *)ptr, as_ushort8(v))

#define BLOCK_WRITE_DST16(ptr, v) \
    do { \
        BLOCK_WRITE_DST8(ptr, (v).s01234567); \
        BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
    } while (0)

#elif DST_DT_BF16
#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write_us((__global ushort *)ptr, (v).data)

#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write_us2((__global ushort *)ptr, as_ushort2(v))

#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write_us4((__global ushort *)ptr, as_ushort4(v))

#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write_us8((__global ushort *)ptr, as_ushort8(v))

#define BLOCK_WRITE_DST16(ptr, v) \
    do { \
        BLOCK_WRITE_DST8(ptr, (v).s01234567); \
        BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
    } while (0)

#elif DST_DT_S32 || DST_DT_F32

#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write((__global uint *)ptr, as_uint(v))

#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write2((__global uint *)ptr, as_uint2(v))

#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write4((__global uint *)ptr, as_uint4(v))

#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write8((__global uint *)ptr, as_uint8(v))

#define BLOCK_WRITE_DST16(ptr, v) \
    do { \
        BLOCK_WRITE_DST8(ptr, (v).s01234567); \
        BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
    } while (0)

#elif DST_DT_F16 || DST_DT_BF16

#define BLOCK_WRITE_DST(ptr, v) \
    intel_sub_group_block_write_us((__global ushort *)ptr, as_ushort(v))

#define BLOCK_WRITE_DST2(ptr, v) \
    intel_sub_group_block_write_us2((__global ushort *)ptr, as_short2(v))

#define BLOCK_WRITE_DST4(ptr, v) \
    intel_sub_group_block_write_us4((__global ushort *)ptr, as_ushort4(v))

#define BLOCK_WRITE_DST8(ptr, v) \
    intel_sub_group_block_write_us8((__global ushort *)ptr, as_ushort8(v))

#define BLOCK_WRITE_DST16(ptr, v) \
    do { \
        BLOCK_WRITE_DST8(ptr, (v).s01234567); \
        BLOCK_WRITE_DST8(ptr + 8 * SUB_GROUP_SIZE, (v).s89abcdef); \
    } while (0)

#endif

#if DST_DT_BF16
#define DST_TO_REF(x) into_float(x)
#define DST_TO_REF8(x) into_float(x)
#define REF_TO_DST(x) into_bf16(x)
#elif DST_DT_BF8
#define DST_TO_REF(x) into_float(x)
#define DST_TO_REF8(x) into_float(x)
#define REF_TO_DST(x) into_f8_e5m2(x)
#elif DST_DT_HF8
#define DST_TO_REF(x) into_float(x)
#define DST_TO_REF8(x) into_float(x)
#define REF_TO_DST(x) into_f8_e4m3(x)
#define DST_DATA_MAX (uchar)0x7B
#define DST_DATA_MIN (uchar)0xFB
#elif DST_DT_F4_E2M1
#define DST_TO_REF(x) into_float(x)
#define DST_TO_REF8(x) into_float(x)
#define REF_TO_DST(x) into_f4_e2m1(x)
#define DST_DATA_MAX (uchar)0x07
#define DST_DATA_MIN (uchar)0x01
#elif DST_DT_E8M0
#define DST_TO_REF(x) into_float(x)
#define DST_TO_REF8(x) into_float(x)
#define REF_TO_DST(x) into_e8m0(x)
#elif DST_DT_F16
#define REF_TO_DST(x) convert_half(x)
#define DST_TO_REF(x) convert_float(x)
#define DST_TO_REF8(x) convert_float8(x)
#elif DST_DT_U8
#define DST_TO_REF(x) (x)
#define DST_TO_REF8(x) (x)
#define REF_TO_DST(x) convert_uchar(x)
#elif DST_DT_S8
#define DST_TO_REF(x) (x)
#define DST_TO_REF8(x) (x)
#define REF_TO_DST(x) convert_char(x)
#elif DST_DT_S4
#define DST_TO_REF(x) into_float(x)
#define DST_TO_REF8(x) into_float(x)
#define REF_TO_DST(x) into_s4(x)
#elif DST_DT_U4
#define DST_TO_REF(x) into_float(x)
#define DST_TO_REF8(x) into_float(x)
#define REF_TO_DST(x) into_u4(x)
#elif DST_DT_U2
#define DST_TO_REF(x) into_float(x)
#define DST_TO_REF8(x) into_float(x)
#define REF_TO_DST(x) into_u2(x)
#else
#define DST_TO_REF(x) (x)
#define DST_TO_REF8(x) (x)
#define REF_TO_DST(x) (x)
#endif

#if DST_DT_BF16
#define TO_DST(x) into_bf16(into_float(x))
#define TO_DST2(x) cvt_f32_to_bf16(convert_float2(x))
#define TO_DST4(x) cvt_f32_to_bf16(convert_float4(x))
#define TO_DST8(x) cvt_f32_to_bf16(convert_float8(x))
#define DST_DATA_FMAX into_float(as_bf16(0x7F7F))
#define DST_DATA_FMIN into_float(as_bf16(0x0080))
#define DST_DATA_FLOW into_float(as_bf16((ushort)0xFF7F))
#elif DST_DT_F16
#define TO_DST(x) convert_half(x)
#define TO_DST2(x) convert_half2(x)
#define TO_DST4(x) convert_half4(x)
#define TO_DST8(x) convert_half8(x)
#define DST_DATA_FMAX convert_float(HALF_MAX)
#define DST_DATA_FMIN convert_float(HALF_MIN)
#define DST_DATA_FLOW -DST_DATA_FMAX
#elif DST_DT_BF8
#define TO_DST(x) into_f8_e5m2(x)
#define TO_DST2(x) cvt_hf_to_f8_e5m2(convert_half2(x))
#define TO_DST4(x) cvt_hf_to_f8_e5m2(convert_half4(x))
#define TO_DST8(x) cvt_hf_to_f8_e5m2(convert_half8(x))
#define DST_DATA_FMAX into_float(as_f8_e5m2((uchar)0x7B))
#define DST_DATA_FMIN into_float(as_f8_e5m2((uchar)0x04))
#define DST_DATA_FLOW into_float(as_f8_e5m2((uchar)0xFB))
#elif DST_DT_HF8
#define TO_DST(x) into_f8_e4m3(x)
#define TO_DST2(x) cvt_hf_to_f8_e4m3(convert_half2(x))
#define TO_DST4(x) cvt_hf_to_f8_e4m3(convert_half4(x))
#define TO_DST8(x) cvt_hf_to_f8_e4m3(convert_half8(x))
#define DST_DATA_FMAX into_float(as_f8_e4m3((uchar)0x7E))
#define DST_DATA_FMIN into_float(as_f8_e4m3((uchar)0x08))
#define DST_DATA_FLOW into_float(as_f8_e4m3((uchar)0xFE))
#elif DST_DT_U4
#define TO_DST(x) into_u4(convert_float(x))
#define DST_DATA_FMAX 15.0f
#define DST_DATA_FMIN 0.0f
#define DST_DATA_FLOW DST_DATA_FMIN
#elif DST_DT_S4
#define TO_DST(x) into_s4(convert_float(x))
#define DST_DATA_FMAX 7.0f
#define DST_DATA_FMIN 1f
#define DST_DATA_FLOW -8.0f
#elif DST_DT_F4_E2M1
#define TO_DST(x) into_f4_e2m1(convert_float(x))
#define DST_DATA_FMAX 6.0f
#define DST_DATA_FMIN 1.0f
#define DST_DATA_FLOW -6.0f
#elif DST_DT_E8M0
#define TO_DST(x) into_e8m0(convert_float(x))
#elif DST_DT_U2
#undef SET_DOUBLE_HALF_BYTE
#define TO_DST(x) into_u2(convert_float(x))
#define DST_DATA_FMAX 3.0f
#define DST_DATA_FMIN 0.0f
#define DST_DATA_FLOW DST_DATA_FMIN
#elif DST_DT_U8
#define TO_DST(x) convert_uchar_sat_rte(x)
#define TO_DST2(x) convert_uchar2_sat_rte(x)
#define TO_DST4(x) convert_uchar4_sat_rte(x)
#define TO_DST8(x) convert_uchar8_sat_rte(x)
#define TO_DST16(x) convert_uchar16_sat_rte(x)
#define DST_DATA_FMAX convert_float(UCHAR_MAX)
#define DST_DATA_FMIN 1
#define DST_DATA_FLOW 0
#elif DST_DT_S8
#define TO_DST(x) convert_char_sat_rte(x)
#define TO_DST2(x) convert_char2_sat_rte(x)
#define TO_DST4(x) convert_char4_sat_rte(x)
#define TO_DST8(x) convert_char8_sat_rte(x)
#define TO_DST16(x) convert_char16_sat_rte(x)
#define DST_DATA_FMAX convert_float(CHAR_MAX)
#define DST_DATA_FMIN 1
#define DST_DATA_FLOW -DST_DATA_FMAX
#elif DST_DT_S32
#define TO_DST(x) convert_int_sat_rte(x)
#define TO_DST2(x) convert_int2_sat_rte(x)
#define TO_DST4(x) convert_int4_sat_rte(x)
#define TO_DST8(x) convert_int8_sat_rte(x)
#define DST_DATA_FMAX convert_float(INT_MAX)
#define DST_DATA_FMIN 1
#define DST_DATA_FLOW -DST_DATA_FMAX
#elif DST_DT_F32
#define TO_DST(x) convert_float(x)
#define TO_DST2(x) convert_float2(x)
#define TO_DST4(x) convert_float4(x)
#define TO_DST8(x) convert_float8(x)
#define DST_DATA_FMAX convert_float(FLT_MAX)
#define DST_DATA_FMIN convert_float(FLT_MIN)
#define DST_DATA_FLOW -DST_DATA_FMAX
#elif DST_DT_F64
#define TO_DST(x) convert_double(x)
#define TO_DST2(x) convert_double2(x)
#define TO_DST4(x) convert_double4(x)
#define TO_DST8(x) convert_double8(x)
#define DST_DATA_FMAX convert_float(FLT_MAX)
#define DST_DATA_FMIN convert_float(FLT_MIN)
#define DST_DATA_FLOW -DST_DATA_FMAX
#else
#error "Not expected"
#endif
#endif

#ifdef C_DATA_T
#if C_DT_BF16
#define C_TO_REF(x) into_float(x)
#elif C_DT_BF8
#define C_TO_REF(x) into_half(x)
#elif C_DT_HF8
#define C_TO_REF(x) into_half(x)
#elif C_DT_F4_E2M1
#define C_TO_REF(x) into_float(x)
#else
#define C_TO_REF(x) (x)
#endif
#if C_DT_BF16
#define TO_C(x) into_bf16(x)
#elif C_DT_BF8
#define TO_C(x) into_f8_e5m2(x)
#elif C_DT_HF8
#define TO_C(x) into_f8_e4m3(x)
#elif C_DT_F4_E2M1
#define TO_C(x) into_f4_e2m1(x)
#elif C_DT_F16
#define TO_C(x) convert_half(x)
#elif C_DT_U8
#define TO_C(x) convert_uchar_sat_rte(x)
#elif C_DT_S8
#define TO_C(x) convert_char_sat_rte(x)
#elif C_DT_S32
#define TO_C(x) convert_int_sat_rte(x)
#elif C_DT_F32
#define TO_C(x) convert_float(x)
#elif C_DT_F64
#define TO_C(x) convert_double(x)
#else
#error "Not expected"
#endif
#endif

#ifdef ACC_DATA_T
#if ACC_DT_F16
#define TO_ACC(x) convert_half(x)
#define ACC_TO_REF(x) convert_float(x)
#elif ACC_DT_F32
#define TO_ACC(x) convert_float(x)
#define ACC_TO_REF(x) convert_float(x)
#elif ACC_DT_F64
#define TO_ACC(x) convert_double(x)
#define ACC_TO_REF(x) convert_double(x)
#elif ACC_DT_S32
#define TO_ACC(x) convert_int(x)
#define ACC_TO_REF(x) convert_float(x)
#else
#error "Unexpected accumulation data type"
#endif
#endif

#ifdef SUM_DATA_T
#define SUM_DATA2_T CONCAT2(SUM_DATA_T, 2)
#define SUM_DATA4_T CONCAT2(SUM_DATA_T, 4)
#define SUM_DATA8_T CONCAT2(SUM_DATA_T, 8)
#define SUM_DATA16_T CONCAT2(SUM_DATA_T, 16)
#define AS_SUM_DATA_T CONCAT2(as_, SUM_DATA_T)
#define AS_SUM_DATA2_T CONCAT2(as_, SUM_DATA2_T)
#define AS_SUM_DATA4_T CONCAT2(as_, SUM_DATA4_T)
#define AS_SUM_DATA8_T CONCAT2(as_, SUM_DATA8_T)
#define AS_SUM_DATA16_T CONCAT2(as_, SUM_DATA16_T)
#if SUM_DT_BF16
#define SUM_TO_REF into_float
#elif SUM_DT_BF8
#define SUM_TO_REF(x) into_float(x)
#elif SUM_DT_HF8
#define SUM_TO_REF(x) into_float(x)
#elif SUM_DT_F4_E2M1
#define SUM_TO_REF(x) into_float(x)
#else
#define SUM_TO_REF AS_SUM_DATA_T
#endif
#endif

#ifdef DST_SCALES_DATA_T
#if DST_SCALES_DT_HF8
#define DST_SCALES_TO_REF(x) into_float(x)
#define REF_TO_DST_SCALES(x) into_f8_e4m3(x)
#define DST_SCALES_DATA_FMAX into_float(as_f8_e4m3((uchar)0x7E))
#define DST_SCALES_DATA_FEPS into_float(as_f8_e4m3((uchar)0x20))
#elif DST_SCALES_DT_BF8
#define DST_SCALES_TO_REF(x) into_float(x)
#define REF_TO_DST_SCALES(x) into_f8_e5m2(x)
#define DST_SCALES_DATA_FMAX into_float(as_f8_e5m2((uchar)0x7B))
#define DST_SCALES_DATA_FEPS into_float(as_f8_e5m2((uchar)0x34))
#elif DST_SCALES_DT_F16
#define DST_SCALES_TO_REF(x) convert_float(x)
#define REF_TO_DST_SCALES(x) convert_half(x)
#elif DST_SCALES_DT_BF16
#define DST_SCALES_TO_REF(x) into_float(x)
#define REF_TO_DST_SCALES(x) into_bf16(x)
#elif DST_SCALES_DT_E8M0
#define DST_SCALES_TO_REF(x) into_float(x)
#define REF_TO_DST_SCALES(x) into_e8m0(x)
#define DST_SCALES_DATA_FMAX into_float(as_e8m0((uchar)0xFE))
#define DST_SCALES_DATA_FEPS into_float(as_e8m0((uchar)0x80))
#else
#define DST_SCALES_TO_REF(x) (x)
#endif
#endif

#ifdef WEI_SCALES_DATA_T
#if WEI_SCALES_DT_HF8
#define WEI_SCALES_TO_REF(x) into_float(x)
#elif WEI_SCALES_DT_BF8
#define WEI_SCALES_TO_REF(x) into_float(x)
#elif WEI_SCALES_DT_F16
#define WEI_SCALES_TO_REF(x) convert_float(x)
#elif WEI_SCALES_DT_BF16
#define WEI_SCALES_TO_REF(x) into_float(x)
#elif WEI_SCALES_DT_E8M0
#define WEI_SCALES_TO_REF(x) into_float(x)
#else
#define WEI_SCALES_TO_REF(x) (x)
#endif
#endif

#ifdef SRC_SCALES_DATA_T
#if SRC_SCALES_DT_HF8
#define SRC_SCALES_TO_REF(x) into_float(x)
#elif SRC_SCALES_DT_BF8
#define SRC_SCALES_TO_REF(x) into_float(x)
#elif SRC_SCALES_DT_F16
#define SRC_SCALES_TO_REF(x) convert_float(x)
#elif SRC_SCALES_DT_BF16
#define SRC_SCALES_TO_REF(x) into_float(x)
#elif SRC_SCALES_DT_E8M0
#define SRC_SCALES_TO_REF(x) into_float(x)
#else
#define SRC_SCALES_TO_REF(x) (x)
#endif
#endif

#ifdef WEI_ZP_DATA_T
#if WEI_ZP_DT_S8
#define WEI_ZP_TO_REF(zp, off) convert_int_sat_rte(zp[off])
#elif WEI_ZP_DT_U8
#define WEI_ZP_TO_REF(zp, off) convert_int_sat_rte(zp[off])
#elif WEI_ZP_DT_S4
#define WEI_ZP_TO_REF(zp, off) cvt_s4_to_s32(GET_HALF_BYTE(zp, off))
#elif WEI_ZP_DT_U4
#define WEI_ZP_TO_REF(zp, off) cvt_u4_to_s32(GET_HALF_BYTE(zp, off))
#elif WEI_ZP_DT_F16
#define WEI_ZP_TO_REF(zp, off) convert_float(zp[off])
#elif WEI_ZP_DT_BF16
#define WEI_ZP_TO_REF(zp, off) into_float(zp[off])
#else
#define WEI_ZP_TO_REF(zp, off) (zp[off])
#endif
#endif

#ifdef SRC_ZP_DATA_T
#if SRC_ZP_DT_S8
#define SRC_ZP_TO_REF(zp, off) convert_int_sat_rte(zp[off])
#elif SRC_ZP_DT_U8
#define SRC_ZP_TO_REF(zp, off) convert_int_sat_rte(zp[off])
#elif SRC_ZP_DT_S4
#define SRC_ZP_TO_REF(zp, off) cvt_s4_to_s32(GET_HALF_BYTE(zp, off))
#elif SRC_ZP_DT_U4
#define SRC_ZP_TO_REF(zp, off) cvt_u4_to_s32(GET_HALF_BYTE(zp, off))
#else
#define SRC_ZP_TO_REF(zp, off) (zp[off])
#endif
#endif

#if SRC_NDIMS == 3
#define CONV_SRC_OFF(n, c, d, h, w) OFF_MD(SRC, n, c, w, 0, 0, 0)
#elif SRC_NDIMS == 4
#define CONV_SRC_OFF(n, c, d, h, w) OFF_MD(SRC, n, c, h, w, 0, 0)
#elif SRC_NDIMS == 5
#define CONV_SRC_OFF(n, c, d, h, w) OFF_MD(SRC, n, c, d, h, w, 0)
#endif

#if WEI_NDIMS == 3
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, o, i, w, 0, 0, 0)
#elif WEI_NDIMS == 4
#if WITH_GROUPS == 0
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, o, i, h, w, 0, 0)
#else
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, g, o, i, w, 0, 0)
#endif
#elif WEI_NDIMS == 5
#if WITH_GROUPS == 0
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, o, i, d, h, w, 0)
#else
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, g, o, i, h, w, 0)
#endif
#elif WEI_NDIMS == 6
#define CONV_WEI_OFF(g, o, i, d, h, w) OFF_MD(WEI, g, o, i, d, h, w)
#endif

#if DST_NDIMS == 3
#define CONV_DST_OFF(n, c, d, h, w) OFF_MD(DST, n, c, w, 0, 0, 0)
#elif DST_NDIMS == 4
#define CONV_DST_OFF(n, c, d, h, w) OFF_MD(DST, n, c, h, w, 0, 0)
#elif DST_NDIMS == 5
#define CONV_DST_OFF(n, c, d, h, w) OFF_MD(DST, n, c, d, h, w, 0)
#endif

#if NDIMS == 2
#define SRC_OFF(x0, x1, d, h, w) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, h, w) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2)
#else
#define WEI_OFF(g, x0, x1, d, h, w) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1)
#endif

#define DST_OFF(x0, x1, d, h, w) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1)

#define DST_SCALE_OFF(n, m, d0, d1, d2, groupSize) \
    (((m * DST_D1) / groupSize) + (n) * (1))
#elif NDIMS == 3
#define SRC_OFF(x0, x1, d, h, x2) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, h, x3) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3)
#else
#define WEI_OFF(g, x0, x1, d, h, x2) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2)
#endif

#define DST_OFF(x0, x1, d, h, x2) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2)

#define DST_SCALE_OFF(n, m, d0, d1, d2, groupSize) \
    ((d0 % DST_D0) * ((DST_D1 * DST_D2) / groupSize) \
            + ((m * DST_D2) / groupSize) + (n) * (1))
#elif NDIMS == 4
#define SRC_OFF(x0, x1, d, x2, x3) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, d, x3, x4) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3 \
            + ((x4) % WEI_B4) * WEI_SB4 + ((x4) / WEI_B4) * WEI_S4)
#else
#define WEI_OFF(g, x1, x2, d, x3, x4) \
    (((x1) % WEI_B0) * WEI_SB0 + ((x1) / WEI_B0) * WEI_S0 \
            + ((x2) % WEI_B1) * WEI_SB1 + ((x2) / WEI_B1) * WEI_S1 \
            + ((x3) % WEI_B2) * WEI_SB2 + ((x3) / WEI_B2) * WEI_S2 \
            + ((x4) % WEI_B3) * WEI_SB3 + ((x4) / WEI_B3) * WEI_S3)
#endif

#define DST_OFF(x0, x1, d, x2, x3) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3)

#define DST_SCALE_OFF(n, m, d0, d1, d2, groupSize) \
    ((d1 % DST_D0) * ((DST_D1 * DST_D2 * DST_D3) / groupSize) \
            + (d0 % DST_D1) * ((DST_D2 * DST_D3) / groupSize) \
            + ((m * DST_D3) / groupSize) + (n) * (1))
#elif NDIMS == 5
#define SRC_OFF(x0, x1, x2, x3, x4) \
    (((x0) % SRC_B0) * SRC_SB0 + ((x0) / SRC_B0) * SRC_S0 \
            + ((x1) % SRC_B1) * SRC_SB1 + ((x1) / SRC_B1) * SRC_S1 \
            + ((x2) % SRC_B2) * SRC_SB2 + ((x2) / SRC_B2) * SRC_S2 \
            + ((x3) % SRC_B3) * SRC_SB3 + ((x3) / SRC_B3) * SRC_S3 \
            + ((x4) % SRC_B4) * SRC_SB4 + ((x4) / SRC_B4) * SRC_S4)

#if WITH_GROUPS == 1
#define WEI_OFF(x0, x1, x2, x3, x4, x5) \
    (((x0) % WEI_B0) * WEI_SB0 + ((x0) / WEI_B0) * WEI_S0 \
            + ((x1) % WEI_B1) * WEI_SB1 + ((x1) / WEI_B1) * WEI_S1 \
            + ((x2) % WEI_B2) * WEI_SB2 + ((x2) / WEI_B2) * WEI_S2 \
            + ((x3) % WEI_B3) * WEI_SB3 + ((x3) / WEI_B3) * WEI_S3 \
            + ((x4) % WEI_B4) * WEI_SB4 + ((x4) / WEI_B4) * WEI_S4 \
            + ((x5) % WEI_B5) * WEI_SB5 + ((x5) / WEI_B5) * WEI_S5)
#else
#define WEI_OFF(g, x1, x2, x3, x4, x5) \
    (((x1) % WEI_B0) * WEI_SB0 + ((x1) / WEI_B0) * WEI_S0 \
            + ((x2) % WEI_B1) * WEI_SB1 + ((x2) / WEI_B1) * WEI_S1 \
            + ((x3) % WEI_B2) * WEI_SB2 + ((x3) / WEI_B2) * WEI_S2 \
            + ((x4) % WEI_B3) * WEI_SB3 + ((x4) / WEI_B3) * WEI_S3 \
            + ((x5) % WEI_B4) * WEI_SB4 + ((x5) / WEI_B4) * WEI_S4)
#endif

#define DST_OFF(x0, x1, x2, x3, x4) \
    (((x0) % DST_B0) * DST_SB0 + ((x0) / DST_B0) * DST_S0 \
            + ((x1) % DST_B1) * DST_SB1 + ((x1) / DST_B1) * DST_S1 \
            + ((x2) % DST_B2) * DST_SB2 + ((x2) / DST_B2) * DST_S2 \
            + ((x3) % DST_B3) * DST_SB3 + ((x3) / DST_B3) * DST_S3 \
            + ((x4) % DST_B4) * DST_SB4 + ((x4) / DST_B4) * DST_S4)

#define DST_SCALE_OFF(n, m, d0, d1, d2, groupSize) \
    ((d2 % DST_D0) * ((DST_D1 * DST_D2 * DST_D3 * DST_D4) / groupSize) \
            + (d1 % DST_D1) * ((DST_D2 * DST_D3 * DST_D4) / groupSize) \
            + (d0 % DST_D2) * ((DST_D3 * DST_D4) / groupSize) \
            + ((m * DST_D4) / groupSize) + (n) * (1))
#endif

#if SRC_DT_U8 == 1
#define SRC_DT_ALIAS UCHAR
#elif SRC_DT_S8 == 1
#define SRC_DT_ALIAS CHAR
#elif SRC_DT_F16 == 1
#define SRC_DT_ALIAS HALF
#elif SRC_DT_BF16 == 1
#define SRC_DT_ALIAS BFLOAT
#elif SRC_DT_F32 == 1
#define SRC_DT_ALIAS FLOAT
#elif SRC_DT_F64 == 1
#define SRC_DT_ALIAS DOUBLE
#endif

#if DST_DT_U8 == 1
#define DST_DT_ALIAS UCHAR
#elif DST_DT_S8 == 1
#define DST_DT_ALIAS CHAR
#elif DST_DT_F16 == 1
#define DST_DT_ALIAS HALF
#elif DST_DT_BF16 == 1
#define DST_DT_ALIAS BFLOAT
#elif DST_DT_F32 == 1
#define DST_DT_ALIAS FLOAT
#elif DST_DT_F64 == 1
#define DST_DT_ALIAS DOUBLE
#endif

#endif
