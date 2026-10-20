/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
* Copyright 2024-2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_BRGEMM_BRGEMM_UTILS_HPP
#define CPU_AARCH64_BRGEMM_BRGEMM_UTILS_HPP

#include "cpu/aarch64/brgemm/brgemm.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/ref_io_helper.hpp"

#include "common/c_types_map.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace scale_utils {
inline size_t scales_simd_w(cpu_isa_t isa) {
    return simd_elems(data_type::f32, isa);
}

inline float load_scale(const primitive_attr_t *attr, int arg,
        const void *scales, dim_t idx = 0) {
    if (attr->scales_.has_default_values(arg)) return 1.f;

    const auto scales_dt = attr->scales_.get_data_type(arg);
    return cpu::io::load_float_value(scales_dt, scales, idx);
}

inline void precompute_oscales(cpu_isa_t isa, float *oscales,
        const primitive_attr_t *attr, const void *src_scales,
        const void *wei_scales, dim_t oc, float scale_adjust_factor) {
    if (oscales == nullptr) return;

    const float src_scale = load_scale(attr, DNNL_ARG_SRC, src_scales);
    const auto &wei_scales_attr = attr->scales_.get(DNNL_ARG_WEIGHTS);
    const bool is_oc_scale = wei_scales_attr.get_mask() > 0;

    if (is_oc_scale) {
        for (dim_t c = 0; c < oc; c++) {
            oscales[c] = src_scale
                    * load_scale(attr, DNNL_ARG_WEIGHTS, wei_scales, c)
                    * scale_adjust_factor;
        }
    } else {
        const float scale = src_scale
                * load_scale(attr, DNNL_ARG_WEIGHTS, wei_scales)
                * scale_adjust_factor;
        utils::array_set(oscales, scale, scales_simd_w(isa));
    }
}

inline void precompute_inv_dst_scales(cpu_isa_t isa,
        float *dst_scales_precomputed, const primitive_attr_t *attr,
        const void *dst_scales) {
    if (dst_scales_precomputed == nullptr) return;

    const float dst_scale = 1.f / load_scale(attr, DNNL_ARG_DST, dst_scales);
    utils::array_set(dst_scales_precomputed, dst_scale, scales_simd_w(isa));
}
} // namespace scale_utils

status_t init_kernel_datatype(
        brgemm_desc_t *brg, data_type_t dt_a, data_type_t dt_b);

namespace brgemm_utils {
inline constexpr int mmla_bd_blk() {
    return 2;
}
inline constexpr int mmla_ld_blk() {
    return 2;
}
// BFMMLA consumes K4 chunks of 8 bytes; two chunks form a K8 block.
inline constexpr int mmla_rd_chunk_bytes() {
    return 8;
}
inline constexpr int mmla_rd_block() {
    return 8;
}

inline constexpr int mmla_rd_chunk_elems(int dt_a_sz) {
    return mmla_rd_chunk_bytes() / dt_a_sz;
}
inline constexpr int mmla_rd_chunks_per_block(int dt_a_sz) {
    return mmla_rd_block() / mmla_rd_chunk_elems(dt_a_sz);
}

// Four f32 output vectors per row x six rows = 24 accumulator registers.
inline constexpr int mmla_ld_block2() {
    return 4;
}
inline constexpr int mmla_max_native_bd_block() {
    return 6;
}
inline int mmla_n_block(cpu_isa_t isa) {
    return mmla_ld_block2() * simd_elems(data_type::f32, isa);
}

// Small OpenMP workloads can favor BFDOT. Scale the measured crossover by
// the ISA-specific N tile while leaving the requested thread team unchanged.
inline bool has_sufficient_mmla_work(double work, int nthr, int n_block) {
    return DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_OMP || nthr <= 1
            || work >= 16. * 1024 * n_block * nthr;
}

status_t init_mmla_wei_md(memory_desc_t &md, int k_dim, int n_dim, int n_block);

void set_mmla_attr(brgemm_attr_t &attr, dim_t M);

status_t validate_mmla_compute(const brgemm_desc_t &brg);

inline bool is_mmla_ld_blocks_supported(dim_t ld_blocks) {
    return ld_blocks >= 1 && ld_blocks <= 12;
}

bool can_dispatch_uker(const brgemm_desc_t *brg);

void maybe_try_bf32(brgemm_desc_t *brg);

status_t set_isa_impl(brgemm_desc_t *brg);

status_t brgemm_blocking(brgemm_desc_t *brg);

status_t brdgmm_blocking(brgemm_desc_t *brg);

/* The purpose of this function is to enable initialization of brgemm values
 * and then call additional functions like blocking heuristics without
 * having to depend on BRGeMM's API. An additional feature is that this
 * function can be modified depending on needs without requiring changes
 * at the API level. */
status_t init_brgemm_conf(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, brgemm_layout_t layout, float alpha, float beta,
        dim_t LDA, dim_t LDB, dim_t LDC, dim_t M, dim_t N, dim_t K,
        const brgemm_strides_t *strides = nullptr, bool is_bf32 = false);

/* The purpose of this function is to enable initialization of brgemm values
 * and then call additional functions like blocking heuristics without
 * having to depend on BRDGeMM's API. An additional feature is that this
 * function can be modified depending on needs without requiring changes
 * at the API level. */
status_t init_brdgmm_conf(brgemm_desc_t *brg, cpu_isa_t isa,
        brgemm_batch_kind_t type, impl::data_type_t dt_a,
        impl::data_type_t dt_b, brgemm_layout_t layout, float alpha, float beta,
        dim_t LDA, dim_t LDC, dim_t M, dim_t N,
        const brgemm_strides_t *strides = nullptr);

} // namespace brgemm_utils

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_BRGEMM_BRGEMM_UTILS_HPP

//vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
