/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2026 Arm Ltd. and affiliates
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

#include <assert.h>
#include "matmul_pd.hpp"
#include "opdesc.hpp"
#include "primitive_desc_iface.hpp"
#include "primitive_desc_iterator.hpp"

#include "oneapi/dnnl/dnnl.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

using namespace dnnl::impl;
using namespace dnnl::impl::utils;
using namespace dnnl::impl::types;

#define VCHECK_MATMUL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::invalid_arguments, msg, ##__VA_ARGS__);

#define VCHECK_MATMUL_UNIMPL(cond, msg, ...) \
    VCONDCHECK(primitive, create, check, matmul, (cond), \
            status::unimplemented, msg, ##__VA_ARGS__);

namespace {
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
// Currently grouped matmul specific validation function is separated
// as the coverage of grouped gemm is experimental and limited
// Supported:
//     2Dx3D (variable M): src 2D grouped [total_M, K] (idx=0, row-major),
//                         wei dense 3D [G, K, N],
//                         dst 2D grouped [total_M, N] (idx=0, row-major)
//     2Dx2D (variable K): src 2D grouped [M, total_K] (idx=1, col-major),
//                         wei 2D grouped [total_K, N] (idx=0, row-major),
//                         dst dense 3D [G, M, N]
status_t grouped_matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc) {
    using namespace data_type;

    VCHECK_MATMUL(
            !any_null(src_desc, weights_desc, dst_desc), VERBOSE_NULL_ARG);

    VCHECK_MATMUL(!any_memory_desc_host_scalar(
                          src_desc, weights_desc, bias_desc, dst_desc),
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    auto op_d = matmul_desc_t();
    op_d.primitive_kind = primitive_kind::matmul;

    op_d.src_desc = *src_desc;
    op_d.weights_desc = *weights_desc;
    if (bias_desc) op_d.bias_desc = *bias_desc;
    op_d.dst_desc = *dst_desc;

    const memory_desc_wrapper src_d(&op_d.src_desc);
    const memory_desc_wrapper wei_d(&op_d.weights_desc);
    const memory_desc_wrapper dst_d(&op_d.dst_desc);

    VCHECK_MATMUL_UNIMPL(
            src_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const bool is_2dby3d = dst_d.is_grouped_desc() && !wei_d.is_grouped_desc();
    const bool is_2dby2d = wei_d.is_grouped_desc() && !dst_d.is_grouped_desc();

    VCHECK_MATMUL_UNIMPL(
            is_2dby3d || is_2dby2d, VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const auto &src_grouped = src_d.sparse_desc().grouped_desc;
    const dim_t group_count = src_grouped.group_count;

    // Check variable dim indices/group count for consistency
    const memory_desc_wrapper &variant_d = is_2dby3d ? dst_d : wei_d;
    const char *variant_name = is_2dby3d ? "dst" : "weights";
    const auto &variant_grouped = variant_d.sparse_desc().grouped_desc;

    VCHECK_MATMUL_UNIMPL(src_grouped.variable_dim_idx == (is_2dby3d ? 0 : 1)
                    && variant_grouped.variable_dim_idx == 0,
            VERBOSE_INCONSISTENT_DIM, "src_variable_dim_idx",
            (int)src_grouped.variable_dim_idx, variant_name,
            (int)variant_grouped.variable_dim_idx);

    VCHECK_MATMUL_UNIMPL(src_grouped.group_count == variant_grouped.group_count,
            VERBOSE_INCONSISTENT_DIM, "src_group_count",
            (int)src_grouped.group_count, variant_name,
            (int)variant_grouped.group_count);

    VCHECK_MATMUL_UNIMPL(
            src_d.metadata_type(0) == s32 && variant_d.metadata_type(0) == s32,
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    // Check number dims for consistency
    const int ndims_src = src_d.ndims();
    const int ndims_wei = wei_d.ndims();
    const int ndims_dst = dst_d.ndims();
    const int expected_ndims_wei = is_2dby3d ? 3 : 2;
    const int expected_ndims_dst = is_2dby3d ? 2 : 3;
    VCHECK_MATMUL_UNIMPL(ndims_src == 2, VERBOSE_BAD_NDIMS, "src", ndims_src);
    VCHECK_MATMUL_UNIMPL(ndims_wei == expected_ndims_wei, VERBOSE_BAD_NDIMS,
            "weights", ndims_wei);
    VCHECK_MATMUL_UNIMPL(ndims_dst == expected_ndims_dst, VERBOSE_BAD_NDIMS,
            "dst", ndims_dst);

    // Check dims for consistency
    const dim_t group_dim = is_2dby3d ? wei_d.dims()[0] : dst_d.dims()[0];
    VCHECK_MATMUL_UNIMPL(group_dim == group_count, VERBOSE_INCONSISTENT_DIM,
            is_2dby3d ? "weights_dim[0]" : "dst_dim[0]", (int)group_dim,
            "src_group_count", (int)group_count);
    VCHECK_MATMUL_UNIMPL(src_d.dims()[0] == dst_d.dims()[dst_d.ndims() - 2],
            VERBOSE_INCONSISTENT_DIM, "src", 0, "dst", dst_d.ndims() - 2);
    VCHECK_MATMUL_UNIMPL(src_d.dims()[1] == wei_d.dims()[wei_d.ndims() - 2],
            VERBOSE_INCONSISTENT_DIM, "src", 1, "weights", wei_d.ndims() - 2);
    VCHECK_MATMUL_UNIMPL(
            dst_d.dims()[dst_d.ndims() - 1] == wei_d.dims()[wei_d.ndims() - 1],
            VERBOSE_INCONSISTENT_DIM, "dst", dst_d.ndims() - 1, "weights",
            wei_d.ndims() - 1);

    if (is_2dby3d) {
        // Bias supported with shape [G, N]
        const bool with_bias = op_d.bias_desc.ndims != 0;
        if (with_bias) {
            const memory_desc_wrapper bia_d(&op_d.bias_desc);
            const dim_t N = wei_d.dims()[wei_d.ndims() - 1];

            VCHECK_MATMUL_UNIMPL(
                    !bia_d.is_sparse_desc() && !bia_d.is_grouped_desc(),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VCHECK_MATMUL_UNIMPL(
                    bia_d.ndims() == 2, VERBOSE_UNSUPPORTED_BIAS_CFG);
            VCHECK_MATMUL_UNIMPL(
                    bia_d.dims()[0] == group_count && bia_d.dims()[1] == N,
                    VERBOSE_INCONSISTENT_DIM, "bias_dim[0]",
                    (int)bia_d.dims()[0], "dst_group_count", (int)group_count);
        }
    } else {
        // Bias is not supported for the 2Dx2D
        VCHECK_MATMUL_UNIMPL(
                op_d.bias_desc.ndims == 0, VERBOSE_UNSUPPORTED_BIAS_CFG);
    }

    op_d.accum_data_type = types::default_accum_data_type(src_desc->data_type,
            weights_desc->data_type, dst_desc->data_type, prop_kind::forward);
    VCHECK_MATMUL_UNIMPL(op_d.accum_data_type != data_type::undef,
            VERBOSE_INVALID_DATATYPE, "accumulation");
    *matmul_desc = op_d;
    return status::success;
}

// Grouped matmul attribute checks.
// Separated from regular matmul as grouped gemm coverage is experimental
// and limited
//
// Note: 2Dx2D supports only default attribues for now
status_t grouped_matmul_attr_check(
        const matmul_desc_t &desc, const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;
    using namespace data_type;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Reject 2Dx2D as no support so far
    const memory_desc_wrapper src_d(&desc.src_desc);
    const memory_desc_wrapper wei_d(&desc.weights_desc);
    const bool is_2dby2d = src_d.is_grouped_desc() && wei_d.is_grouped_desc();
    VCHECK_MATMUL_UNIMPL(!is_2dby2d, VERBOSE_UNSUPPORTED_ATTR);

    // Grouped matmul supports scales, zero points, woq, and post-ops
    auto allowed_mask = smask_t::scales_data_type | smask_t::scales_groups
            | smask_t::fpmath_mode | smask_t::zero_points_data_type
            | smask_t::zero_points_groups | smask_t::post_ops;
    VCHECK_MATMUL_UNIMPL(
            attr->has_default_values(allowed_mask, desc.dst_desc.data_type),
            VERBOSE_UNSUPPORTED_ATTR);

    const auto src_dt = desc.src_desc.data_type;
    const auto wei_dt = desc.weights_desc.data_type;
    const auto &sc = attr->scales_;

    // Dissalow scales on fp src/wei/dst data as real-world applications
    // rely on low precision fp or integer (e.g., WoQ) data
    VCHECK_MATMUL_UNIMPL(IMPLICATION(utils::one_of(src_dt, f32, bf16, f16),
                                 sc.has_default_values(DNNL_ARG_SRC)),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCHECK_MATMUL_UNIMPL(IMPLICATION(utils::one_of(wei_dt, f32, bf16, f16),
                                 sc.has_default_values(DNNL_ARG_WEIGHTS)),
            VERBOSE_UNSUPPORTED_SCALES_CFG);
    VCHECK_MATMUL_UNIMPL(sc.has_default_values(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    // Post-ops: only eltwise and binary_mul are supported
    const memory_desc_wrapper dst_d(&desc.dst_desc);
    const auto &po = attr->post_ops_;
    for (int i = 0; i < po.len(); ++i) {
        const auto &e = po.entry_[i];
        VCHECK_MATMUL_UNIMPL(
                e.is_eltwise() || e.is_binary(), VERBOSE_UNSUPPORTED_POSTOP);

        if (e.is_binary()) {
            VCHECK_MATMUL_UNIMPL(e.binary.alg == alg_kind::binary_mul,
                    VERBOSE_UNSUPPORTED_POSTOP);
            const memory_desc_wrapper src1_d(e.binary.src1_desc);

            VCHECK_MATMUL(src1_d.ndims() == dst_d.ndims(),
                    VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "dst", "bin_po src1",
                    dst_d.ndims(), src1_d.ndims());

            if (src1_d.is_grouped_desc()) {
                // Validate grouped binary post-op descriptor:
                // group_count and variable_dim_idx must match dst
                const auto &bin_g = src1_d.sparse_desc().grouped_desc;
                const auto &dst_g = dst_d.sparse_desc().grouped_desc;
                VCHECK_MATMUL_UNIMPL(bin_g.group_count == dst_g.group_count
                                && bin_g.variable_dim_idx
                                        == dst_g.variable_dim_idx,
                        VERBOSE_INCONSISTENT_MDS, "binary post-op", "dst");
            } else {
                // Validate dense binary post-op descriptor:
                // only per-group (e.g., NVFP4 with [G, 1])
                // or per-row shapes are allowed
                const dim_t bin_M = src1_d.dims()[0];
                const dim_t bin_N = src1_d.dims()[src1_d.ndims() - 1];
                const dim_t total_M = dst_d.dims()[0];
                const dim_t gc = dst_d.sparse_desc().grouped_desc.group_count;
                VCHECK_MATMUL_UNIMPL(
                        bin_N == 1 || src1_d.matches_one_of_tag(format_tag::ab),
                        VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "binary post-op");
                VCHECK_MATMUL_UNIMPL(utils::one_of(bin_M, gc, total_M)
                                && IMPLICATION(bin_M == gc, bin_N == 1),
                        VERBOSE_INCONSISTENT_MDS, "binary post-op", "dst");
            }
        }
    }

    return status::success;
}
#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY

status_t matmul_attr_check(const matmul_desc_t &desc, const engine_t *engine,
        const primitive_attr_t *attr) {
    using smask_t = primitive_attr_t::skip_mask_t;
    using namespace quantization_mode;

    if (attr == nullptr) return status::success;
    if (attr->has_default_values()) return status::success;

    // Check attributes
    const data_type_t src_dt = desc.src_desc.data_type;
    const data_type_t wei_dt = desc.weights_desc.data_type;
    const data_type_t dst_dt = desc.dst_desc.data_type;

    auto attr_mask = smask_t::post_ops | smask_t::sum_dt | smask_t::dropout
            | smask_t::rounding_mode;
    // Matmul supports scales for floating point data types
    attr_mask |= smask_t::scales_data_type;
    // Matmul supports in-place post-ops, both on CPU and on GPU
    attr_mask |= smask_t::post_ops_inplace;

    const bool src_is_int8
            = utils::one_of(src_dt, data_type::s8, data_type::u8);
    const bool src_is_fp8
            = utils::one_of(src_dt, data_type::f8_e5m2, data_type::f8_e4m3);
    const bool src_is_fp4 = utils::one_of(src_dt, data_type::f4_e2m1);
    if (src_is_int8 || src_is_fp8 || src_is_fp4)
        attr_mask |= smask_t::zero_points;
    // Sub-byte weights apply zero points during upconversion, so precomputed
    // reductions are only supported for int8 WEI
    if (src_is_int8 && utils::one_of(wei_dt, data_type::s8, data_type::u8))
        attr_mask |= smask_t::precomputed_reductions;

    // Matmul supports zero points for floating point data types as part of
    // weights decompression.
    const bool wei_is_int = utils::one_of(wei_dt, data_type::s8, data_type::u8,
            data_type::s4, data_type::u4, data_type::u2, data_type::u3);
    const bool wei_is_fp8
            = utils::one_of(wei_dt, data_type::f8_e5m2, data_type::f8_e4m3);
    const bool wei_is_fp4 = utils::one_of(wei_dt, data_type::f4_e2m1);
    if (wei_is_int || wei_is_fp8 || wei_is_fp4) {
        attr_mask |= smask_t::zero_points_data_type;
        attr_mask |= smask_t::zero_points_groups;
        attr_mask |= smask_t::scales_groups;
    }

    const bool dst_is_fp8
            = utils::one_of(dst_dt, data_type::f8_e5m2, data_type::f8_e4m3);
    const bool dst_is_fp4 = utils::one_of(dst_dt, data_type::f4_e2m1);
    // grouped dst scales are supported for MXFP
    if (dst_is_fp8 || dst_is_fp4) attr_mask |= smask_t::scales_groups;

    // Matmul supports fpmath mode and accumulation mode
    attr_mask |= smask_t::fpmath_mode | smask_t::accumulation_mode;

    VCHECK_MATMUL_UNIMPL(attr->has_default_values(attr_mask, dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);

    const int ndims_src = desc.src_desc.ndims;
    const int ndims_wei = desc.weights_desc.ndims;
    const int m_idx = ndims_src - 2;
    const int k_idx_wei = m_idx;
    const int n_idx = ndims_wei - 1;
    const dim_t K = desc.weights_desc.dims[k_idx_wei];
    const dim_t N = desc.weights_desc.dims[n_idx];

    assert(ndims_src >= 2);
    assert(ndims_wei >= 2);
    int src_qmask_M = 1 << (ndims_src - 2);
    int src_qmask_K = 1 << (ndims_src - 1);

    int wei_qmask_K = 1 << (ndims_wei - 2);
    int wei_qmask_N = 1 << (ndims_wei - 1);

    int dst_qmask_M = src_qmask_M;
    int dst_qmask_N = wei_qmask_N;

    int full_tensor_mask = (1 << ndims_src) - 1;

    const auto &quant_groups_are_divisible = [](dim_t g1, dim_t g2) -> bool {
        return IMPLICATION(g1 > 1 && g2 > 1, (g1 % g2 == 0) || (g2 % g1 == 0));
    };

    const auto &mask_fits_ndims = [](int mask, int ndims) -> bool {
        return mask >= 0 && mask < (1 << ndims);
    };

    // Check scales
    if (!attr->scales_.has_default_values()) {
        const auto &sc = attr->scales_;

        // Only weights and destination scales support a non-default
        // quantization mode.
        VCHECK_MATMUL_UNIMPL(
                sc.has_default_qmode({DNNL_ARG_WEIGHTS, DNNL_ARG_DST}),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        dim_t src_scale_group_k = 1;
        if (!sc.has_default_values(DNNL_ARG_SRC)) {
            const int mask_src = sc.get_mask(DNNL_ARG_SRC);

            VCHECK_MATMUL(mask_fits_ndims(mask_src, ndims_src),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // Arbitrary masks are allowed only for 8-bit and smaller src type.
            const bool src_mask_is_scalar = mask_src == 0;
            const bool src_mask_is_full = mask_src == full_tensor_mask;
            const bool src_mask_is_mk = utils::one_of(mask_src, src_qmask_M,
                    src_qmask_K, src_qmask_M + src_qmask_K);
            if (types::data_type_bits(src_dt) > 8) {
                VCHECK_MATMUL_UNIMPL(src_mask_is_scalar || src_mask_is_mk
                                || src_mask_is_full,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }

            if (!sc.get(DNNL_ARG_SRC).has_default_groups()) {
                if (mask_src & src_qmask_K)
                    src_scale_group_k = sc.get_group(DNNL_ARG_SRC, 1);
            }

            // Due to hardware specifics, groups, when more than 1, should be
            // multiple of 16.
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(src_scale_group_k > 1 && src_scale_group_k < K,
                            src_scale_group_k % 16 == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        dim_t wei_scale_group_k = 1;
        dim_t wei_scale_group_n = 1;
        if (!sc.has_default_values(DNNL_ARG_WEIGHTS)) {
            const auto wei_sc_qmode
                    = sc.get_quantization_mode(DNNL_ARG_WEIGHTS);
            VCHECK_MATMUL_UNIMPL(
                    utils::one_of(wei_sc_qmode, static_sazp, static_sbzp),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            // static_sbzp scales are floating-point values, f16 or bf16.
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(wei_sc_qmode == static_sbzp,
                            utils::one_of(sc.get_data_type(DNNL_ARG_WEIGHTS),
                                    data_type::bf16, data_type::f16)),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            // static_sbzp is a weights decompression mode.
            VCHECK_MATMUL_UNIMPL(IMPLICATION(wei_sc_qmode == static_sbzp,
                                         attr->fpmath_.apply_to_int_),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            const int mask_wei = sc.get_mask(DNNL_ARG_WEIGHTS);

            VCHECK_MATMUL(mask_fits_ndims(mask_wei, ndims_wei),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            if (!sc.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                if (mask_wei & wei_qmask_K)
                    wei_scale_group_k = sc.get_group(DNNL_ARG_WEIGHTS, 0);
                if (mask_wei & wei_qmask_N)
                    wei_scale_group_n = sc.get_group(DNNL_ARG_WEIGHTS, 1);
            }

            // Due to hardware specifics, groups, when more than 1, should be
            // multiple of 16.
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(wei_scale_group_k > 1 && wei_scale_group_k < K,
                            wei_scale_group_k % 16 == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(wei_scale_group_n > 1 && wei_scale_group_n < N,
                            wei_scale_group_n % 16 == 0),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        if (!sc.has_default_values(DNNL_ARG_DST)) {
            VCHECK_MATMUL_UNIMPL(
                    utils::one_of(sc.get_quantization_mode(DNNL_ARG_DST),
                            static_sazp, dynamic_mx, dynamic_fp),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            const int mask_dst = sc.get_mask(DNNL_ARG_DST);
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(mask_dst > 0, dst_is_fp8 || dst_is_fp4),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            VCHECK_MATMUL_UNIMPL(
                    utils::one_of(mask_dst, 0, dst_qmask_N, dst_qmask_M,
                            dst_qmask_N + dst_qmask_M, full_tensor_mask),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        // Check dependency between scales.
        // Source scales groups are supported for int8 source and must divide
        // or be divided by weights groups when both are greater than 1.
        const bool groups_are_divisible = quant_groups_are_divisible(
                src_scale_group_k, wei_scale_group_k);
        VCHECK_MATMUL_UNIMPL(IMPLICATION(src_scale_group_k > 1,
                                     (src_is_int8 || src_is_fp8 || src_is_fp4)
                                             && groups_are_divisible),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        // For dynamic_mx scaling, we support only OCP MX flavor
        if (sc.get(DNNL_ARG_DST).is_mx()) {
            // only group size of 32
            VCHECK_MATMUL_UNIMPL(sc.get_mask(DNNL_ARG_DST) == full_tensor_mask,
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
            VCHECK_MATMUL_UNIMPL(sc.get_group(DNNL_ARG_DST, -1) == 32
                            && sc.get_group(DNNL_ARG_DST, -2) == 1,
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            // only e8m0 scales
            VCHECK_MATMUL_UNIMPL(
                    sc.get_data_type(DNNL_ARG_DST) == data_type::e8m0,
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }

        // For dynamic_fp scaling, only NVFP4 flavor is supported.
        if (sc.get(DNNL_ARG_DST).is_dynamic_fp()) {
            using namespace data_type;

            VCHECK_MATMUL_UNIMPL(sc.get_mask(DNNL_ARG_DST) == full_tensor_mask,
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            switch (sc.get_data_type(DNNL_ARG_DST)) {
                case f8_e4m3:
                    // Group sizes of 16 for NVFP4.
                    VCHECK_MATMUL_UNIMPL(sc.get_group(DNNL_ARG_DST, -1) == 16
                                    && sc.get_group(DNNL_ARG_DST, -2) == 1,
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                    break;
                default:
                    VCHECK_MATMUL_UNIMPL(false, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    break;
            }
        }
    }

    // Check zero points
    if (!attr->zero_points_.has_default_values()) {
        const auto &zp = attr->zero_points_;

        // Only weights zero-points support a non-default quantization mode.
        VCHECK_MATMUL_UNIMPL(zp.has_default_qmode({DNNL_ARG_WEIGHTS}),
                VERBOSE_UNSUPPORTED_ZP_CFG);

        dim_t src_zp_group_k = 1;
        if (!zp.has_default_values(DNNL_ARG_SRC)) {
            const int mask_src = zp.get_mask(DNNL_ARG_SRC);

            VCHECK_MATMUL_UNIMPL(
                    utils::one_of(mask_src, 0, src_qmask_K,
                            src_qmask_M + src_qmask_K, full_tensor_mask),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            // Only a unit batch dim is supported for `full_tensor_mask`, e.g.,
            // 1xMxK:BxKxN
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(mask_src == full_tensor_mask, ndims_src <= 3),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(mask_src == full_tensor_mask && ndims_src == 3,
                            !is_runtime_value(desc.src_desc.dims[0])
                                    && desc.src_desc.dims[0] == 1),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            if (!zp.get(DNNL_ARG_SRC).has_default_groups()) {
                if (mask_src & src_qmask_K)
                    src_zp_group_k = zp.get_group(DNNL_ARG_SRC, 1);
            }

            // Due to hardware specifics, groups, when more than 1, should be
            // multiple of 32.
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(src_zp_group_k > 1 && src_zp_group_k < K,
                            src_zp_group_k % 32 == 0),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }

        dim_t wei_zp_group_k = 1;
        dim_t wei_zp_group_n = 1;
        if (!zp.has_default_values(DNNL_ARG_WEIGHTS)) {
            const auto wei_zp_qmode
                    = zp.get_quantization_mode(DNNL_ARG_WEIGHTS);
            VCHECK_MATMUL_UNIMPL(
                    utils::one_of(wei_zp_qmode, static_sazp, static_sbzp),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            // static_sbzp zero-points are floating-point values, f16 or bf16.
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(wei_zp_qmode == static_sbzp,
                            utils::one_of(zp.get_data_type(DNNL_ARG_WEIGHTS),
                                    data_type::bf16, data_type::f16)),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            // Weights scales and zero-points must share the quantization mode.
            // Unset scales have the default mode, so static_sbzp zero-points
            // also require static_sbzp scales.
            // TODO: (postponed till more similar places appear) consolidate
            // cross-argument checks in a common space.
            VCHECK_MATMUL_UNIMPL(wei_zp_qmode
                            == attr->scales_.get_quantization_mode(
                                    DNNL_ARG_WEIGHTS),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            const int mask_wei = zp.get_mask(DNNL_ARG_WEIGHTS);

            VCHECK_MATMUL(mask_fits_ndims(mask_wei, ndims_wei),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            if (!zp.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                if (mask_wei & wei_qmask_K)
                    wei_zp_group_k = zp.get_group(DNNL_ARG_WEIGHTS, 0);
                if (mask_wei & wei_qmask_N)
                    wei_zp_group_n = zp.get_group(DNNL_ARG_WEIGHTS, 1);
            }

            // Groups per N are solely for weights decompression as it's
            // impossible to get performant kernel for a single `k` element in
            // chain for regular quantized case.
            VCHECK_MATMUL_UNIMPL(IMPLICATION(wei_zp_group_n > 1,
                                         attr->fpmath_.apply_to_int_),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            // Due to hardware specifics, groups, when more than 1, should be
            // multiple of 16.
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(wei_zp_group_k > 1 && wei_zp_group_k < K,
                            wei_zp_group_k % 16 == 0),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VCHECK_MATMUL_UNIMPL(
                    IMPLICATION(wei_zp_group_n > 1 && wei_zp_group_n < N,
                            wei_zp_group_n % 16 == 0),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            if (utils::one_of(zp.get_data_type(DNNL_ARG_WEIGHTS), data_type::s4,
                        data_type::u4)) {
                VCHECK_MATMUL_UNIMPL(
                        IMPLICATION(mask_wei & wei_qmask_K, K % 2 == 0),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                VCHECK_MATMUL_UNIMPL(
                        IMPLICATION(mask_wei & wei_qmask_N, N % 2 == 0),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
            }
        }

        if (!zp.has_default_values(DNNL_ARG_DST)) {
            const int mask_dst = zp.get_mask(DNNL_ARG_DST);

            VCHECK_MATMUL_UNIMPL(mask_dst == 0
                            || (desc.dst_desc.ndims == 2 && mask_dst == 1 << 1),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }

        // Check dependency between zero_points.
        // Source zero_points groups are supported for int8 source and must
        // divide or be divided by weights groups when both are greater than 1.
        const bool groups_are_divisible
                = quant_groups_are_divisible(src_zp_group_k, wei_zp_group_k);
        VCHECK_MATMUL_UNIMPL(IMPLICATION(src_zp_group_k > 1,
                                     src_is_int8 && groups_are_divisible),
                VERBOSE_UNSUPPORTED_ZP_CFG);
    }

    // Check precomputed reductions
    if (!attr->precomputed_reductions_.has_default_values()) {
        const auto &pr = attr->precomputed_reductions_;

        // Only SRC argument is supported so far.
        std::vector<int> supported_args = {DNNL_ARG_SRC};
        VCHECK_MATMUL_UNIMPL(pr.has_default_values(supported_args),
                VERBOSE_UNSUPPORTED_PR_CFG);

        if (!pr.has_default_values(DNNL_ARG_SRC)) {
            const auto &zp = attr->zero_points_;
            // Weights zero points must be specified.
            VCHECK_MATMUL_UNIMPL(!zp.get(DNNL_ARG_WEIGHTS).has_default_values(),
                    VERBOSE_UNSUPPORTED_PR_CFG);

            // Mask must be defined for a full tensor, no broadcasts.
            const int pr_mask_src = pr.get_mask(DNNL_ARG_SRC);
            VCHECK_MATMUL_UNIMPL(pr_mask_src == full_tensor_mask,
                    VERBOSE_UNSUPPORTED_PR_CFG);

            // Data type must be s32 so far.
            const auto pr_dt = pr.get_data_type(DNNL_ARG_SRC);
            VCHECK_MATMUL_UNIMPL(
                    pr_dt == data_type::s32, VERBOSE_UNSUPPORTED_PR_CFG);

            if (!pr.get(DNNL_ARG_SRC).has_default_groups()) {
                const auto &sc = attr->scales_;
                const dim_t src_pr_group_k = pr.get_group(DNNL_ARG_SRC, 1);

                // Pre-computed reductions can only be used when the PR group
                // size is the minimal amongst all group sizes along the K
                // dimension from a mathematical point of view.
                // An entry not quantized over K doesn't constrain the group.
                const auto &k_group = [&K](const quant_entry_t &e, int qmask_K,
                                              int k_idx) -> dim_t {
                    if (!(e.get_mask() & qmask_K)) return K;
                    return e.has_default_groups() ? 1 : e.get_group(k_idx);
                };
                const dim_t wei_zp_group_k
                        = k_group(zp.get(DNNL_ARG_WEIGHTS), wei_qmask_K, -2);
                const dim_t src_zp_group_k
                        = k_group(zp.get(DNNL_ARG_SRC), src_qmask_K, -1);
                const dim_t wei_sc_group_k
                        = k_group(sc.get(DNNL_ARG_WEIGHTS), wei_qmask_K, -2);
                const dim_t src_sc_group_k
                        = k_group(sc.get(DNNL_ARG_SRC), src_qmask_K, -1);
                const dim_t minimal_group
                        = std::min(std::min(wei_zp_group_k, src_zp_group_k),
                                std::min(wei_sc_group_k, src_sc_group_k));

                const bool groups_are_divisible = quant_groups_are_divisible(
                        src_pr_group_k, wei_zp_group_k);
                VCHECK_MATMUL(IMPLICATION(src_pr_group_k > 1, src_is_int8),
                        VERBOSE_UNSUPPORTED_PR_CFG);
                VCHECK_MATMUL(
                        IMPLICATION(src_pr_group_k > 1, groups_are_divisible),
                        VERBOSE_UNSUPPORTED_PR_CFG);
                VCHECK_MATMUL(IMPLICATION(src_pr_group_k > 1,
                                      src_pr_group_k <= minimal_group),
                        VERBOSE_UNSUPPORTED_PR_CFG);
            }
        }
    }

    // Check post-ops
    if (!attr->post_ops_.has_default_values()) {
        const auto &po = attr->post_ops_;
        using namespace primitive_kind;
        VCHECK_MATMUL_UNIMPL(
                po.has_default_values({binary, eltwise, prelu, sum}),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Check sum
        VCHECK_MATMUL_UNIMPL(
                po.check_sum_consistency(dst_dt, src_is_int8, true),
                VERBOSE_UNSUPPORTED_POSTOP);

        // Note: verbose support is inside the call.
        CHECK(po.validate_binary(engine->kind(), &desc.dst_desc));
    }

    return status::success;
}

} // namespace

namespace dnnl {
namespace impl {
status_t matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const memory_desc_t *reduce_desc, matmul_reduce_kind_t reduce_kind) {
    VCHECK_MATMUL(
            !any_null(src_desc, weights_desc, dst_desc), VERBOSE_NULL_ARG);

    // Note: This is an artificial limitation for the internal `reduce` feature
    // to limit the scope to what is actually used.
    VCHECK_MATMUL(
            IMPLICATION(bias_desc, !reduce_desc), VERBOSE_UNSUPPORTED_BIAS_CFG);

    VCHECK_MATMUL(!any_memory_desc_host_scalar(src_desc, weights_desc,
                          bias_desc, dst_desc, reduce_desc),
            VERBOSE_UNSUPPORTED_FORMAT_KIND);

    auto op_d = matmul_desc_t();
    op_d.primitive_kind = primitive_kind::matmul;

    op_d.src_desc = *src_desc;
    op_d.weights_desc = *weights_desc;
    if (bias_desc) op_d.bias_desc = *bias_desc;
    op_d.dst_desc = *dst_desc;
    if (reduce_desc) {
        VCHECK_MATMUL(reduce_desc->format_kind != format_kind::any,
                VERBOSE_UNSUPPORTED_FORMAT_KIND);
        op_d.reduce_desc = *reduce_desc;
        op_d.reduce_kind = reduce_kind;
        VCHECK_MATMUL(op_d.reduce_kind != matmul_reduce_kind::undef,
                VERBOSE_BAD_PARAM, "reduce_kind");
    }

    const bool with_bias = op_d.bias_desc.ndims != 0;
    const bool with_reduce = op_d.reduce_desc.ndims != 0;
    const int ndims = dst_desc->ndims;
    VCHECK_MATMUL(ndims >= 2 && ndims <= DNNL_MAX_NDIMS, VERBOSE_BAD_NDIMS,
            "dst", ndims);
    VCHECK_MATMUL(everyone_is(ndims, src_desc->ndims, weights_desc->ndims),
            VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src", "weights",
            src_desc->ndims, weights_desc->ndims);
    VCHECK_MATMUL(IMPLICATION(with_bias, op_d.bias_desc.ndims == ndims),
            VERBOSE_BAD_NDIMS, "bias", op_d.bias_desc.ndims);
    VCHECK_MATMUL(IMPLICATION(with_reduce, op_d.reduce_desc.ndims == ndims),
            VERBOSE_BAD_NDIMS, "reduce", op_d.reduce_desc.ndims);

    // check: m, n, k
    const int m_idx = ndims - 2;
    const int k_idx_src = m_idx + 1;
    const int k_idx_wei = m_idx;
    const int n_idx = ndims - 1;
    VCHECK_MATMUL(dst_desc->dims[m_idx] == src_desc->dims[m_idx],
            VERBOSE_INCONSISTENT_DIM, "dst", m_idx, "src", m_idx);
    VCHECK_MATMUL(dst_desc->dims[n_idx] == weights_desc->dims[n_idx],
            VERBOSE_INCONSISTENT_DIM, "dst", n_idx, "weights", n_idx);
    VCHECK_MATMUL(src_desc->dims[k_idx_src] == weights_desc->dims[k_idx_wei],
            VERBOSE_INCONSISTENT_DIM, "src", k_idx_src, "weights", k_idx_wei);
    VCHECK_MATMUL(IMPLICATION(with_bias,
                          one_of(op_d.bias_desc.dims[n_idx], 1,
                                  dst_desc->dims[n_idx])),
            VERBOSE_INCONSISTENT_DIM, "bias", n_idx, "dst", n_idx);
    VCHECK_MATMUL(IMPLICATION(with_bias,
                          one_of(op_d.bias_desc.dims[m_idx], 1,
                                  dst_desc->dims[m_idx])),
            VERBOSE_INCONSISTENT_DIM, "bias", m_idx, "dst", m_idx);

    VCHECK_MATMUL(IMPLICATION(with_reduce,
                          one_of(op_d.reduce_desc.dims[n_idx], 1,
                                  dst_desc->dims[n_idx])),
            VERBOSE_INCONSISTENT_DIM, "reduce", n_idx, "dst", n_idx);
    VCHECK_MATMUL(IMPLICATION(with_reduce,
                          one_of(op_d.reduce_desc.dims[m_idx], 1,
                                  dst_desc->dims[m_idx])),
            VERBOSE_INCONSISTENT_DIM, "reduce", m_idx, "dst", m_idx);

    const int bia_mask = with_bias
            ? utils::get_dims_mask(dst_desc->dims, op_d.bias_desc.dims, ndims)
            : 0;

    // check if other dims match.
    for (int d = 0; d < ndims - 2; ++d) {
        const dim_t s_dim = src_desc->dims[d];
        const dim_t w_dim = weights_desc->dims[d];
        const dim_t d_dim = dst_desc->dims[d];
        const dim_t b_dim = with_bias ? op_d.bias_desc.dims[d] : 0;
        const dim_t r_dim = with_reduce ? op_d.reduce_desc.dims[d] : 0;

        if (any_runtime_value(s_dim, w_dim, d_dim, b_dim)) {

            VCHECK_MATMUL(all_runtime_values(s_dim, w_dim, d_dim)
                            && IMPLICATION((bia_mask & (1 << d)) && with_bias,
                                    is_runtime_value(b_dim)),
                    VERBOSE_RUNTIMEDIM_INCONSISTENT, d);
        } else {
            // This follows numpy semantics of broadcasting when 0 is involved.
            VCHECK_MATMUL(IMPLICATION(!everyone_is(s_dim, w_dim, d_dim),
                                  one_of(1, s_dim, w_dim)),
                    VERBOSE_INVALID_BROADCAST, "dst", d);
            VCHECK_MATMUL(IMPLICATION(s_dim == 1, d_dim == w_dim),
                    VERBOSE_INVALID_BROADCAST, "weights", d);
            VCHECK_MATMUL(IMPLICATION(w_dim == 1, d_dim == s_dim),
                    VERBOSE_INVALID_BROADCAST, "src", d);
            VCHECK_MATMUL(IMPLICATION(with_bias, one_of(b_dim, 1, d_dim)),
                    VERBOSE_INCONSISTENT_DIM, "bias", d, "dst", d);
            VCHECK_MATMUL(IMPLICATION(with_reduce, one_of(r_dim, 1, d_dim)),
                    VERBOSE_INCONSISTENT_DIM, "reduce", d, "dst", d);
        }
    }

    op_d.accum_data_type = types::default_accum_data_type(src_desc->data_type,
            weights_desc->data_type, dst_desc->data_type, prop_kind::forward);
    VCHECK_MATMUL(op_d.accum_data_type != data_type::undef,
            VERBOSE_INVALID_DATATYPE, "accumulation");
    *matmul_desc = op_d;
    return status::success;
}

status_t matmul_desc_init(matmul_desc_t *matmul_desc,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc) {
    return matmul_desc_init(matmul_desc, src_desc, weights_desc, bias_desc,
            dst_desc, nullptr, matmul_reduce_kind::undef);
}

status_t create_matmul_pd(std::shared_ptr<primitive_desc_t> &matmul_pd,
        const engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *bias_md,
        const memory_desc_t *dst_md, const primitive_attr_t *attr,
        const memory_desc_t *reduce_md, matmul_reduce_kind_t reduce_kind,
        data_type_t acc_dt) {
    matmul_pd = nullptr;
    matmul_desc_t matmul_desc;
    CHECK(matmul_desc_init(&matmul_desc, src_md, wei_md, bias_md, dst_md,
            reduce_md, reduce_kind));
    if (acc_dt != data_type::undef) matmul_desc.accum_data_type = acc_dt;

    primitive_attr_t matmul_attr = *attr;

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&matmul_desc, &matmul_attr, nullptr);
    if (!it.is_initialized()) return status::out_of_memory;

    while (++it != it.end()) {
        if (*it) {
            matmul_pd = *it;
            return status::success;
        }
    }
    VCHECK_MATMUL_UNIMPL(false, VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");
    return status::unimplemented;
}

} // namespace impl
} // namespace dnnl

status_t dnnl_matmul_primitive_desc_create(
        primitive_desc_iface_t **primitive_desc_iface, engine_t *engine,
        const memory_desc_t *src_desc, const memory_desc_t *weights_desc,
        const memory_desc_t *bias_desc, const memory_desc_t *dst_desc,
        const primitive_attr_t *attr) {
    auto matmul_desc = matmul_desc_t();

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
    const memory_desc_wrapper src_d(src_desc);
    if (src_d.is_grouped_desc()) {
        // Use grouped-specific validation functions
        CHECK(grouped_matmul_desc_init(
                &matmul_desc, src_desc, weights_desc, bias_desc, dst_desc));
        CHECK(grouped_matmul_attr_check(matmul_desc, attr));
        return primitive_desc_create(primitive_desc_iface, engine,
                (const op_desc_t *)&matmul_desc, nullptr, attr);
    }
#endif

    // Regular matmul validation
    CHECK(matmul_desc_init(
            &matmul_desc, src_desc, weights_desc, bias_desc, dst_desc));
    CHECK(matmul_attr_check(matmul_desc, engine, attr));
    return primitive_desc_create(primitive_desc_iface, engine,
            (const op_desc_t *)&matmul_desc, nullptr, attr);
}
