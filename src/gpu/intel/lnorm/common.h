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

#ifndef GPU_INTEL_LNORM_COMMON_H
#define GPU_INTEL_LNORM_COMMON_H

#undef SRC_OFF
#undef DST_OFF

#define SRC_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(SRC, x0, x1, x2, x3, x4, x5)
#define DST_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(DST, x0, x1, x2, x3, x4, x5)
#define STAT_OFF(x0, x1, x2, x3, x4, x5) OFF_MD(STAT, x0, x1, x2, x3, x4, x5)

#if SRC_DT_F64 || DST_DT_F64
#define ACC_DATA_T double
#else
#define ACC_DATA_T float
#endif

#if IS_BWD
#if USE_SCALE || USE_SHIFT
#if VECTORIZE_BWD_SCALESHIFT

#if VECTOR_SIZE_SCALESHIFT == 1
#define VECTORIZED_VERSION(x) x
#define vector_load(x) (x)
#else
#define VECTORIZED_VERSION(x) CONCAT2(x, VECTOR_SIZE_SCALESHIFT)
#define vector_load(x) CONCAT2(vload, VECTOR_SIZE_SCALESHIFT)(0, &x)
#endif

#if DT_BF16 == 1
#define convert_vector_to_float cvt_bf16_to_f32
#else
#define convert_vector_to_float VECTORIZED_VERSION(convert_float)
#endif

#if SRC_DT_BF16 == 1
#define convert_vector_src_to_float cvt_bf16_to_f32
#else
#define convert_vector_src_to_float VECTORIZED_VERSION(convert_float)
#endif

#if defined(SRC_DT_F64)
#define SRC_BLOCK_DATA_T ulong
#elif defined(SRC_DT_F32)
#define SRC_BLOCK_DATA_T uint
#elif defined(SRC_DT_F16) || defined(SRC_DT_BF16)
#define SRC_BLOCK_DATA_T ushort
#elif defined(SRC_DT_U8) || defined(SRC_DT_S8)
#define SRC_BLOCK_DATA_T uchar
#else
#error "Unexpected SRC data type"
#endif

#define as_vector_data_t VECTORIZED_VERSION(AS_DATA_T)
#define as_vector_src_data_t VECTORIZED_VERSION(AS_SRC_DATA_T)
#define sub_group_read VECTORIZED_VERSION(BLOCK_READ)
#define vector_float VECTORIZED_VERSION(float)

#endif //USE_SCALE
#endif //USE_VECTORIZE_BWD

#endif
#endif
