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

#ifndef GPU_INTEL_BINARY_SIMPLE_HPP
#define GPU_INTEL_BINARY_SIMPLE_HPP

#include "common/c_types_map.hpp"
#include "gpu/intel/binary/config.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace binary {

struct simple_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public binary::pd_t {
        using binary::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:simple:any", simple_t);

        status_t init(const impl::engine_t *engine) {
            using namespace data_type;
            using sm = primitive_attr_t::skip_mask_t;

            const auto attr_skip_mask = sm::post_ops | sm::scales;

            VDISPATCH_BINARY_SC(set_default_params(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_BINARY(
                    ((utils::everyone_is(
                              bf16, src_md(0)->data_type, src_md(1)->data_type)
                             && utils::one_of(
                                     dst_md()->data_type, bf16, u8, f32))
                            || (utils::one_of(src_md(0)->data_type, f16, f32,
                                        s8, u8, s32)
                                    && utils::one_of(src_md(1)->data_type, f16,
                                            f32, s8, u8, s32)
                                    && utils::one_of(dst_md()->data_type, f16,
                                            f32, s8, u8, s32))
                            || (src_md(0)->data_type == f32
                                    && src_md(1)->data_type == bf16
                                    && utils::one_of(
                                            dst_md()->data_type, bf16, f32))
                            || (src_md(0)->data_type == bf16
                                    && src_md(1)->data_type == f32
                                    && dst_md()->data_type == bf16)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BINARY(IMPLICATION(is_ternary_op(),
                                     utils::one_of(src_md(2)->data_type, s8)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_BINARY(
                    memory_desc_ndims_ok(src_md(0), src_md(1), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src_0, src_1", "dst");

            VDISPATCH_BINARY(IMPLICATION(!attr()->scales_.has_default_values(),
                                     check_scales_mask()),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);

            VDISPATCH_BINARY(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_BINARY(
                    post_ops_with_binary_ok(attr(), *dst_md(), MAX_NDIMS),
                    VERBOSE_UNSUPPORTED_POSTOP);

            VDISPATCH_BINARY_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);

            CHECK(init_conf(engine));
            return status::success;
        }

        status_t init_conf(const impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool with_scales(int position) const {
            return !attr()->scales_.has_default_values(position);
        }

        bool with_scales() const {
            return with_scales(DNNL_ARG_SRC_0) || with_scales(DNNL_ARG_SRC_1);
        }

        bool with_eltwise(int position) const {
            return attr()->post_ops_.contain(primitive_kind::eltwise, position);
        }

        bool with_sum() const {
            return attr()->post_ops_.find(primitive_kind::sum) != -1;
        }

        float eltwise_alpha() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alpha
                    : 1.0f;
        }

        float eltwise_beta() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.beta
                    : 0.0f;
        }

        float eltwise_scale() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.scale
                    : 1.0f;
        }

        float sum_scale() const {
            const int sum_idx = attr()->post_ops_.find(primitive_kind::sum);
            return sum_idx != -1 ? attr()->post_ops_.entry_[sum_idx].sum.scale
                                 : 0.0f;
        }

        alg_kind_t eltwise_alg_kind() const {
            const int eltwise_idx
                    = attr()->post_ops_.find(primitive_kind::eltwise);
            return eltwise_idx != -1
                    ? attr()->post_ops_.entry_[eltwise_idx].eltwise.alg
                    : dnnl_alg_kind_undef;
        }

        conf_t conf;

    private:
        bool check_scales_mask() const {
            const std::vector<int> supported_args
                    = {DNNL_ARG_SRC_0, DNNL_ARG_SRC_1};
            return attr_scales_ok(supported_args);
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        CHECK(create_kernel(engine, &kernel_, "simple_binary", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_simple(ctx);
    }

private:
    status_t execute_simple(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace binary
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
