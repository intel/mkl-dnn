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

#ifndef CPU_MATMUL_REF_MATMUL_HPP
#define CPU_MATMUL_REF_MATMUL_HPP

#include <assert.h>

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"

#include "cpu/matmul/cpu_matmul_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

struct ref_matmul_t : public primitive_t {
    struct pd_t : public cpu_matmul_pd_t {
        using cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_matmul_t);

        status_t init(const engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(utils::one_of(src_type, f32, bf16, f16, f8_e5m2,
                                     f8_e4m3, f4_e2m1),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(utils::one_of(wei_type, f32, bf16, f16, f8_e5m2,
                                     f8_e4m3, f4_e2m1, u8, s8, u4, s4, u2, u3),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(utils::one_of(dst_type, f32, bf16, f16, f8_e5m2,
                                     f8_e4m3, f4_e2m1, u8, s8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(
                    (src_type == wei_type
                            || utils::one_of(wei_type, bf16, f16, f8_e5m2,
                                    f8_e4m3, u8, s8, u4, s4, u2, u3, f4_e2m1)),
                    VERBOSE_UNSUPPORTED_DT);
            /* int weights decompression support */
            VDISPATCH_MATMUL(
                    IMPLICATION(utils::one_of(wei_type, u8, s8, u4, s4, u2, u3),
                            attr_.mayiconvert(wei_type, src_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(IMPLICATION(src_type == f16,
                                     utils::one_of(dst_type, f32, f16, u8, s8)),
                    VERBOSE_UNSUPPORTED_DT);
            // TODO: any implication on allowed dst data type for fp8?
            VDISPATCH_MATMUL(
                    IMPLICATION(with_bias(),
                            utils::one_of(
                                    bia_type, f32, bf16, f16, f8_e5m2, f8_e4m3)
                                    && IMPLICATION(
                                            wei_type == f32, bia_type == f32)
                                    && IMPLICATION(wei_type == f16,
                                            utils::one_of(bia_type, f32, f16))
                                    && IMPLICATION(wei_type == bf16,
                                            utils::one_of(bia_type, f32, bf16))
                            // TODO: any implication on allowed bias
                            // data type for fp8?
                            ),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_MATMUL(platform::has_data_type_support(src_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(platform::has_data_type_support(wei_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(platform::has_data_type_support(dst_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(IMPLICATION(with_bias(),
                                     platform::has_data_type_support(bia_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_MATMUL(
                    attr()->has_default_values(smask_t::scales_data_type
                                    | smask_t::scales_groups
                                    | smask_t::zero_points_data_type
                                    | smask_t::zero_points_groups
                                    | smask_t::post_ops | smask_t::sum_dt
                                    | smask_t::fpmath_mode | smask_t::dropout
                                    | smask_t::rounding_mode
                                    | smask_t::post_ops_inplace,
                            dst_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_MATMUL(
                    attr_.post_ops_.check_sum_consistency(dst_type,
                            /* is_int8 */ types::is_integral_dt(dst_type)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(ref_post_ops_t::post_ops_ok(attr()->post_ops_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            CHECK(attr_scales_ok(engine,
                    {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST},
                    {quantization_mode::static_sazp,
                            quantization_mode::static_sbzp,
                            quantization_mode::dynamic_mx,
                            quantization_mode::dynamic_fp},
                    {{DNNL_ARG_SRC, {any_mask}}}));
            CHECK(attr_zero_points_ok(engine, {DNNL_ARG_WEIGHTS, DNNL_ARG_DST},
                    {quantization_mode::static_sazp,
                            quantization_mode::static_sbzp}));
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);
            CHECK(dropout_ok(engine));

            init_scratchpad();

            return status::success;
        }

        int nthr_ = 0;
        int ntasks_ = 0;

    private:
        void init_scratchpad();

        status_t dropout_ok(const engine_t *engine) const {
            if (attr_.dropout_.has_default_values()) return status::success;

            assert(memory_desc_wrapper(dst_md(0)).format_kind()
                    == format_kind::blocked);

            using namespace format_tag;
            // See `ref_dropout(...)` comment which explains the requirement.
            VDISPATCH_MATMUL(memory_desc_matches_one_of_tag(
                                     *dst_md(0), ncdhw, nchw, ncw, nc)
                            && IMPLICATION(attr_.dropout_.has_output_mask(),
                                    memory_desc_wrapper(dst_md(0)).similar_to(
                                            attr_.dropout_.dropout_desc_, true,
                                            false)),
                    VERBOSE_UNSUPPORTED_DROPOUT);

            return status::success;
        }
    };

    ref_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
