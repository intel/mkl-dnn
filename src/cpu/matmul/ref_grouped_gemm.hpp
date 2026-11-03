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

#ifndef CPU_MATMUL_REF_GROUPED_GEMM_HPP
#define CPU_MATMUL_REF_GROUPED_GEMM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#if DNNL_EXPERIMENTAL_GROUPED_MEMORY

#include "common/c_types_map.hpp"
#include "common/memory_desc_wrapper.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

// Two grouped matmul patterns are supported:
// 2D grouped src (variable M) x 3D dense wei -> 2D grouped dst (variable M)
// 2D grouped src (variable K) x 2D grouped wei (variable M) -> dense 3D dst
struct ref_grouped_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref_grouped:any", ref_grouped_t);

        // For 3D weights [G, K, N], override masks to include 0th expert dim
        int wei_qmask_K() const { return (1 << 0) | (1 << 1); }
        int wei_qmask_N() const { return (1 << 0) | (1 << 2); }

        bool is_2dby2d() const { return is_2dby2d_; }

        status_t init(const engine_t *engine) {
            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            // Detect pattern (2Dx3D vs 2Dx2D) and initialize
            VDISPATCH_MATMUL(
                    src_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            is_2dby2d_ = wei_d.is_grouped_desc();
            return is_2dby2d_ ? init_2dby2d(engine) : init_2dby3d(engine);
        }

    private:
        bool is_2dby2d_ = false;

        status_t init_2dby3d(const engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            // Resolve format_any to plain dense
            if (memory_desc_wrapper(weights_md(0)).format_any())
                CHECK(memory_desc_init_by_strides(weights_md_, nullptr));

            memory_desc_wrapper wei_d(weights_md(0));
            memory_desc_wrapper dst_d(dst_md());

            // Supported configurations: grouped src/dst, abc/acb 3D weights
            VDISPATCH_MATMUL(
                    dst_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(
                    wei_d.matches_one_of_tag(format_tag::abc, format_tag::acb),
                    VERBOSE_UNSUPPORTED_TAG);

            // Supported data types: fp and int for src/wei
            const bool is_fp_src = utils::one_of(
                    src_type, f32, bf16, f16, f8_e5m2, f8_e4m3, f4_e2m1);
            const bool is_fp_wei = utils::one_of(
                    wei_type, f32, bf16, f16, f8_e5m2, f8_e4m3, f4_e2m1);
            const bool is_int_src = utils::one_of(src_type, u8, s8);
            const bool is_int_wei = utils::one_of(wei_type, u8, s8, s4, u4, u3);

            // Supported configurations: fp src + int wei (weight-only quantization),
            // int src + int wei, fp src + fp wei
            VDISPATCH_MATMUL(is_fp_src || is_int_src, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(is_fp_wei || is_int_wei, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(utils::one_of(dst_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(IMPLICATION(is_int_src, is_int_wei),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            // WOQ requires weight scales and fpmath with apply_to_int
            VDISPATCH_MATMUL(IMPLICATION(is_fp_src && is_int_wei,
                                     !attr()->scales_.has_default_values(
                                             DNNL_ARG_WEIGHTS)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(IMPLICATION(is_fp_src && is_int_wei,
                                     attr()->fpmath_.apply_to_int_),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            // Note: f4_e2m1 passes the data type checks above but is rejected
            // here on all ISAs, see `platform::has_data_type_support()`.
            VDISPATCH_MATMUL_SC(platform::data_type_support_status(src_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL_SC(platform::data_type_support_status(wei_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL_SC(platform::data_type_support_status(dst_type),
                    VERBOSE_UNSUPPORTED_DT);

            // Check for supported quantization schemes
            const auto &attr_scales = attr()->scales_;
            if (!attr_scales.has_default_values(DNNL_ARG_SRC)) {
                const int src_mask = attr_scales.get_mask(DNNL_ARG_SRC);
                const int rowwise_mask = src_qmask_M();
                const int blocked_mask = src_qmask_M() | src_qmask_K();
                // Allow row-wise or blocked (K-grouping) scales for src
                VDISPATCH_MATMUL(
                        src_mask == rowwise_mask || src_mask == blocked_mask,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        utils::one_of(attr_scales.get_data_type(DNNL_ARG_SRC),
                                f32, bf16, f16, e8m0, f8_e4m3, f8_e5m2),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                if (!attr_scales.get(DNNL_ARG_SRC).has_default_groups()) {
                    // K-grouped src scales supported with int and fp types
                    VDISPATCH_MATMUL(is_int_src || is_fp_src,
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                    const auto gM = attr_scales.get_group(DNNL_ARG_SRC, -2);
                    VDISPATCH_MATMUL(gM == 1, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    const auto gK = attr_scales.get_group(DNNL_ARG_SRC, -1);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    VDISPATCH_MATMUL(
                            K() % gK == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
            }
            if (!attr_scales.has_default_values(DNNL_ARG_WEIGHTS)) {
                const int wei_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
                const int colwise_mask = wei_qmask_N();
                const int blocked_mask = wei_qmask_K() | wei_qmask_N();
                // Allow column-wise or blocked (K or KxN grouping) wei scales
                VDISPATCH_MATMUL(
                        utils::one_of(
                                attr_scales.get_data_type(DNNL_ARG_WEIGHTS),
                                f32, bf16, f16, e8m0, f8_e4m3, f8_e5m2),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                VDISPATCH_MATMUL(
                        wei_mask == colwise_mask || wei_mask == blocked_mask,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                if (!attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                    // K-grouped wei scales supported with int and fp types
                    VDISPATCH_MATMUL(is_int_wei || is_fp_wei,
                            VERBOSE_UNSUPPORTED_SCALES_CFG);
                    const auto gK = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    VDISPATCH_MATMUL(
                            K() % gK == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
                    const auto gN = attr_scales.get_group(DNNL_ARG_WEIGHTS, -1);
                    VDISPATCH_MATMUL(
                            N() % gN == 0, VERBOSE_UNSUPPORTED_SCALES_CFG);
                }
            }
            // Zero-points are supported for src and wei: for WOQ (fp src) and
            // for int arithmetic (int src)
            const auto &attr_zps = attr()->zero_points_;
            VDISPATCH_MATMUL(attr_zps.has_default_values(DNNL_ARG_DST),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            // Allow row-wise or blocked (K grouping) zps for src
            if (!attr_zps.has_default_values(DNNL_ARG_SRC)) {
                VDISPATCH_MATMUL(is_int_src, VERBOSE_UNSUPPORTED_ZP_CFG);
                VDISPATCH_MATMUL(
                        utils::one_of(attr_zps.get_data_type(DNNL_ARG_SRC), u8,
                                s8, s32),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                const int zp_mask = attr_zps.get_mask(DNNL_ARG_SRC);
                const int rowwise_mask = src_qmask_M();
                const int blocked_mask = src_qmask_M() | src_qmask_K();
                VDISPATCH_MATMUL(
                        zp_mask == rowwise_mask || zp_mask == blocked_mask,
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                if (!attr_zps.get(DNNL_ARG_SRC).has_default_groups()) {
                    const auto gM = attr_zps.get_group(DNNL_ARG_SRC, -2);
                    VDISPATCH_MATMUL(gM == 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                    const auto gK = attr_zps.get_group(DNNL_ARG_SRC, -1);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                    VDISPATCH_MATMUL(K() % gK == 0, VERBOSE_UNSUPPORTED_ZP_CFG);
                }
            }

            // Allow column-wise or blocked (K or KxN grouping) zps for weights,
            // and common zps for u3 wei only
            if (!attr_zps.has_default_values(DNNL_ARG_WEIGHTS)) {
                VDISPATCH_MATMUL(is_int_wei, VERBOSE_UNSUPPORTED_ZP_CFG);
                VDISPATCH_MATMUL(
                        utils::one_of(attr_zps.get_data_type(DNNL_ARG_WEIGHTS),
                                u8, s8, u4, s4, s32),
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                const int zp_mask = attr_zps.get_mask(DNNL_ARG_WEIGHTS);
                const int colwise_mask = wei_qmask_N();
                const int blocked_mask = wei_qmask_K() | wei_qmask_N();
                const bool common_ok = zp_mask == 0 && wei_type == u3;
                VDISPATCH_MATMUL(zp_mask == colwise_mask
                                || zp_mask == blocked_mask || common_ok,
                        VERBOSE_UNSUPPORTED_ZP_CFG);
                if (!attr_zps.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                    const auto gK = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
                    VDISPATCH_MATMUL(gK > 1, VERBOSE_UNSUPPORTED_ZP_CFG);
                    VDISPATCH_MATMUL(K() % gK == 0, VERBOSE_UNSUPPORTED_ZP_CFG);
                    const auto gN = attr_zps.get_group(DNNL_ARG_WEIGHTS, -1);
                    VDISPATCH_MATMUL(N() % gN == 0, VERBOSE_UNSUPPORTED_ZP_CFG);
                }
            }

            // for src/wei scales group size in case of K grouping,
            // one must be a multiple of the other
            if (!attr_scales.get(DNNL_ARG_SRC).has_default_groups()
                    && !attr_scales.get(DNNL_ARG_WEIGHTS)
                                .has_default_groups()) {
                const auto src_gK = attr_scales.get_group(DNNL_ARG_SRC, -1);
                const auto wei_gK = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
                VDISPATCH_MATMUL(src_gK % wei_gK == 0 || wei_gK % src_gK == 0,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            }

            // Scales and ZPs groups must match
            if (!attr_scales.has_default_values(DNNL_ARG_WEIGHTS)
                    && !attr_zps.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                const auto scale_gK
                        = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
                const auto zp_gK = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
                VDISPATCH_MATMUL(scale_gK == zp_gK, VERBOSE_INCONSISTENT_DIM,
                        "wei_scale_group_k", (int)scale_gK, "wei_zp_group_k",
                        (int)zp_gK);
            }

            // Resolve format_any for dense binary post-ops
            const auto &po = attr()->post_ops_;
            for (int i = 0; i < po.len(); ++i) {
                auto &e = attr_.post_ops_.entry_[i];
                if (e.is_binary()) {
                    const memory_desc_wrapper src1_d(e.binary.src1_desc);
                    if (src1_d.format_any()) {
                        CHECK(memory_desc_init_by_strides(
                                e.binary.src1_desc, nullptr));
                    }
                }
            }

            return status::success;
        }

        status_t init_2dby2d(const engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper dst_d(dst_md());

            // Resolve format_any to plain dense
            if (dst_d.format_any())
                CHECK(memory_desc_init_by_strides(dst_md_, nullptr));

            // Only plain 3D dst is supported
            VDISPATCH_MATMUL(dst_d.is_plain(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(dst_d.ndims() == 3, VERBOSE_BAD_NDIMS, "dst",
                    dst_d.ndims());

            VDISPATCH_MATMUL(src_type == wei_type && src_type == dst_type
                            && utils::one_of(src_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL_SC(platform::data_type_support_status(src_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL_SC(platform::data_type_support_status(wei_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL_SC(platform::data_type_support_status(dst_type),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_MATMUL(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(!with_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

            return status::success;
        }
    };

    ref_grouped_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        const auto &po = pd()->attr()->post_ops_;
        for (int i = 0; i < po.len(); ++i) {
            const auto &e = po.entry_[i];
            if (e.is_eltwise())
                eltwise_po_.emplace_back(e.eltwise);
            else if (e.is_binary())
                binary_po_.emplace_back(e.binary);
        }
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::vector<ref_eltwise_scalar_fwd_t> eltwise_po_;
    std::vector<ref_binary_scalar_t> binary_po_;
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // DNNL_EXPERIMENTAL_GROUPED_MEMORY
#endif // CPU_MATMUL_REF_GROUPED_GEMM_HPP
