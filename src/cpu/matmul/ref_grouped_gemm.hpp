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

        bool is_2dby2d() const { return is_2dby2d_; }

        status_t init(const engine_t *engine) {
            using namespace data_type;
            memory_desc_wrapper src_d(src_md());
            memory_desc_wrapper wei_d(weights_md(0));

            // Detect pattern (2Dx3D vs 2Dx2D) and initialize
            VDISPATCH_MATMUL(
                    src_d.is_grouped_desc(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            is_2dby2d_ = wei_d.is_grouped_desc();
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            // Scales and zero points layouts are checked by
            // grouped_matmul_attr_check(), their data types here
            const auto &attr_scales = attr()->scales_;
            const auto &attr_zps = attr()->zero_points_;
            for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS})
                VDISPATCH_MATMUL(attr_scales.has_default_values(arg)
                                || utils::one_of(attr_scales.get_data_type(arg),
                                        f32, bf16, f16, e8m0, f8_e4m3, f8_e5m2),
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_MATMUL(attr_zps.has_default_values(DNNL_ARG_SRC)
                            || utils::one_of(
                                    attr_zps.get_data_type(DNNL_ARG_SRC), u8,
                                    s8, s32),
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            VDISPATCH_MATMUL(attr_zps.has_default_values(DNNL_ARG_WEIGHTS)
                            || utils::one_of(
                                    attr_zps.get_data_type(DNNL_ARG_WEIGHTS),
                                    u8, s8, u4, s4, s32),
                    VERBOSE_UNSUPPORTED_ZP_CFG);

            return is_2dby2d_ ? init_2dby2d(engine) : init_2dby3d(engine);
        }

    private:
        bool is_2dby2d_ = false;

        status_t init_2dby3d(const engine_t *engine) {
            using namespace data_type;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            memory_desc_wrapper wei_d(weights_md(0));

            // Supported configurations: abc/acb 3D weights
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

            // Supported configurations: f32/bf16/f16 src + int wei (weight-only
            // quantization), int src + int wei but u3, fp src + fp wei
            VDISPATCH_MATMUL(is_fp_src || is_int_src, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(is_fp_wei || is_int_wei, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(utils::one_of(dst_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(
                    IMPLICATION(is_int_src, is_int_wei && wei_type != u3),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(IMPLICATION(is_fp_src && is_int_wei,
                                     utils::one_of(src_type, f32, bf16, f16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(platform::has_data_type_support(src_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(platform::has_data_type_support(wei_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(platform::has_data_type_support(dst_type),
                    VERBOSE_UNSUPPORTED_DT);

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

            // Only plain dst is supported
            VDISPATCH_MATMUL(dst_d.is_plain(), VERBOSE_UNSUPPORTED_TAG);

            // Supported data types: f32/bf16/f16 for all tensors, or fp8 src
            // and wei with f32/bf16/f16 dst
            const bool is_f8_src = utils::one_of(src_type, f8_e5m2, f8_e4m3);
            const bool is_f8_wei = utils::one_of(wei_type, f8_e5m2, f8_e4m3);
            VDISPATCH_MATMUL(utils::one_of(dst_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL((is_f8_src && is_f8_wei)
                            || utils::everyone_is(dst_type, src_type, wei_type),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL(platform::has_data_type_support(src_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(platform::has_data_type_support(wei_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(platform::has_data_type_support(dst_type),
                    VERBOSE_UNSUPPORTED_DT);

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
