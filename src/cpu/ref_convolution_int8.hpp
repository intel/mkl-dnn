/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef CPU_REF_CONVOLUTION_INT8_HPP
#define CPU_REF_CONVOLUTION_INT8_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/primitive_attr_postops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_convolution_int8_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_int8_fwd_t);

        status_t init(const engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;
            const auto src_type = src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto bia_type = weights_md(1)->data_type;
            const auto dst_type = dst_md(0)->data_type;

            VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(
                    utils::one_of(src_type, s8, u8), VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(wei_type == s8, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(IMPLICATION(with_bias(),
                                   utils::one_of(bia_type, f32, bf16, f16, s32,
                                           s8, u8)),
                    VERBOSE_UNSUPPORTED_BIAS_CFG);
            VDISPATCH_CONV(utils::one_of(dst_type, f32, bf16, f16, s32, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(
                    attr()->has_default_values(smask_t::scales
                                    | smask_t::zero_points | smask_t::post_ops
                                    | smask_t::sum_dt,
                            dst_type),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_CONV(attr()->post_ops_.check_sum_consistency(dst_type,
                                   /* is_int8 */ true),
                    VERBOSE_UNSUPPORTED_POSTOP);
            CHECK(attr_scales_ok());
            CHECK(attr_zero_points_ok(
                    {{DNNL_ARG_SRC, {0, 2}}, {DNNL_ARG_DST, {0, 2}}}));
            VDISPATCH_CONV(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_CONV(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);
            return status::success;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool post_ops_ok() const {
            return ref_post_ops_t::post_ops_ok(attr()->post_ops_);
        }
    };

    ref_convolution_int8_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

struct ref_convolution_int8_bwd_data_t : public primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        using cpu_convolution_bwd_data_pd_t::cpu_convolution_bwd_data_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_convolution_int8_bwd_data_t);

        status_t init(const engine_t *engine) {
            using namespace data_type;
            const auto diff_src_type = diff_src_md(0)->data_type;
            const auto wei_type = weights_md(0)->data_type;
            const auto diff_dst_type = diff_dst_md(0)->data_type;

            VDISPATCH_CONV(desc()->prop_kind == prop_kind::backward_data,
                    VERBOSE_BAD_PROPKIND);
            VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
                    VERBOSE_BAD_ALGORITHM);
            VDISPATCH_CONV(utils::one_of(diff_dst_type, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(wei_type == s8, VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(utils::one_of(diff_src_type, f32, bf16, s32, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_CONV(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_CONV(attr()->has_default_values(
                                   primitive_attr_t::skip_mask_t::scales),
                    VERBOSE_UNSUPPORTED_ATTR);
            CHECK(attr_scales_ok());

            return status::success;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = utils::pick(ndims() - 3, nwc, nhwc, ndhwc);
            auto wei_tag = with_groups()
                    ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                    : utils::pick(ndims() - 3, oiw, oihw, oidhw);
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    ref_convolution_int8_bwd_data_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_data(ctx);
    }

private:
    status_t execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
