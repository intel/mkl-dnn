/*******************************************************************************
* Copyright 2025-2026 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_MATMUL_JIT_INT8_KERNEL_TYPES_HPP
#define CPU_AARCH64_MATMUL_JIT_INT8_KERNEL_TYPES_HPP

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

enum jit_int8_broadcast_t { none, common };

struct dyn_vals_t {
    dim_t M = 0;
    dim_t K = 0;
    dim_t N = 0;
    dim_t batch = 0;
    bool is_s8 = false;
    bool is_u8 = false;
    int mtail = 0;
    int ktail = 0;
    int ntail = 0;
    int m_blk = 0;
    int k_blk = 0;
    int n_blk = 0;
};

struct dyn_params_t {
    const int8_t *src;
    int8_t *dst;
    const int *nm;
    const int *nk;
    const int *nn;
    const bool *is_m_tail;
    const bool *is_k_tail;
    const bool *is_n_tail;
};

struct brg_int8_t {
    dim_t batch;
    int M;
    int K;
    int N;
    const int m_blk = 8;
    const int k_blk = 8;
    // Number of N columns packed per B vector load.
    // Depends on SVE vector length: isa_length_in_bytes / k_blk
    int n_blk;
    // rd_block represents the loop unroll factor along K dimension and bd_block
    // represents the blocking along M dimension
    const int rd_block = 4;
    const int bd_block = 8;
    // ld_block represents the number of vector registers used to load along the
    // N dimension.
    // ld_block is determined via pd init-time heuristics
    int ld_block;
    int m_tail;
    int n_tail;
    int k_tail;
    bool is_m_tail;
    bool is_k_tail;
    bool is_n_tail;
    bool is_zp_cal;
    data_type_t dst_dt;
    const int acc_dt_sz = sizeof(float);
    int dst_dt_sz;
    bool is_s8;
    bool is_u8_s8;
    bool is_bias;
    bool with_scales;
    bool with_src_scales;
    bool with_wei_scales;
    bool with_dst_scales;
    bool is_oc_scales;
    bool is_per_m_scales = false;
    jit_int8_broadcast_t zp_type_a = jit_int8_broadcast_t::none;
    jit_int8_broadcast_t zp_type_b = jit_int8_broadcast_t::none;
    jit_int8_broadcast_t zp_type_c = jit_int8_broadcast_t::none;
    bool is_zp_b_int8 = false;
    bool b_reo = true;
    bool is_a_plain = false;
    data_type_t zp_b_dt;
};

struct call_params_t {
    const uint8_t *src;
    const uint8_t *wei;
    uint8_t *dst;
    const float *bias;
    const float *src_scales; // optional per-row src logical-M scales
    const float *wei_scales; // optional kernel-ready weight scales
    const float *scales;
    const float *dst_scales;
    dim_t M;
    dim_t K;
    dim_t N;
    int *na;
    int *nb;
    const int32_t *src_zero_point;
    const int32_t *wei_zero_point;
    const int32_t *dst_zero_point;
    const int8_t *wei_zero_point_buf;
    float *zp_a_ptr;
    float *zp_b_ptr;
};

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
#endif
