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

#ifndef GPU_INTEL_ELTWISE_REF_HPP
#define GPU_INTEL_ELTWISE_REF_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "gpu/gpu_eltwise_pd.hpp"
#include "gpu/intel/eltwise/config.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace eltwise {

struct ref_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gpu_eltwise_fwd_pd_t {
        using gpu_eltwise_fwd_pd_t::gpu_eltwise_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_fwd_t);
        status_t dropout_ok() const {
            if (attr_.dropout_.has_default_values()) return status::success;

            assert(memory_desc_wrapper(dst_md(0)).format_kind()
                    == format_kind::blocked);

            using namespace format_tag;
            // See `ref_dropout(...)` comment which explains the requirement.
            VDISPATCH_ELTWISE_IC(memory_desc_matches_one_of_tag(
                                         *dst_md(0), ncdhw, nchw, ncw, nc)
                            && IMPLICATION(attr_.dropout_.has_output_mask(),
                                    memory_desc_wrapper(dst_md(0)).similar_to(
                                            attr_.dropout_.dropout_desc_, true,
                                            false)),
                    VERBOSE_UNSUPPORTED_DROPOUT);

            return status::success;
        }

        status_t init(const impl::engine_t *engine) {
            const auto *intel_engine
                    = utils::downcast<const intel::engine_t *>(engine);

            const auto attr_skip_mask = (primitive_attr_t::skip_mask_t::post_ops
                    | primitive_attr_t::skip_mask_t::dropout);

            using namespace alg_kind;
            VDISPATCH_ELTWISE(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(src_md()->data_type == dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "src", "dst");

            VDISPATCH_ELTWISE(memory_desc_ndims_ok(dst_md()), VERBOSE_BAD_NDIMS,
                    "dst_md", dst_md()->ndims);
            VDISPATCH_ELTWISE(attr()->has_default_values(attr_skip_mask),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(memory_desc_wrapper(src_md())
                            == memory_desc_wrapper(dst_md()),
                    VERBOSE_INCONSISTENT_MDS, "src", "dst");
            VDISPATCH_ELTWISE(
                    post_ops_with_binary_ok(attr(), *dst_md(), MAX_NDIMS),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_ELTWISE_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::f64,
                                      intel_engine->mayiuse(
                                              compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(IMPLICATION(src_md()->data_type == data_type::f16,
                                      intel_engine->mayiuse(
                                              compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            CHECK(dropout_ok());
            CHECK(init_conf(engine));
            return status::success;
        }

        status_t init_conf(const impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        CHECK(create_kernel(engine, &kernel_, "ref_eltwise_fwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward_dense(ctx);
    }

private:
    status_t execute_forward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

struct ref_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gpu_eltwise_bwd_pd_t {
        using gpu_eltwise_bwd_pd_t::gpu_eltwise_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_bwd_t);

        status_t init(const impl::engine_t *engine) {
            using namespace prop_kind;
            using namespace utils;
            assert(engine->kind() == engine_kind::gpu);
            const auto *intel_engine
                    = utils::downcast<const intel::engine_t *>(engine);

            using namespace alg_kind;
            VDISPATCH_ELTWISE(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_ELTWISE(memory_desc_ndims_ok(data_md(), diff_dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "data_md",
                    "diff_dst_md", data_md()->ndims, diff_dst_md()->ndims);
            VDISPATCH_ELTWISE(
                    utils::one_of(data_md()->data_type, data_type::f32,
                            data_type::f16, data_type::bf16, data_type::f64,
                            data_type::f8_e5m2, data_type::f8_e4m3),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    utils::everyone_is(data_md()->data_type,
                            diff_src_md()->data_type, diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_ELTWISE(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_ELTWISE(
                    IMPLICATION(data_md()->data_type == data_type::f64,
                            intel_engine->mayiuse(
                                    compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_ELTWISE(
                    IMPLICATION(data_md()->data_type == data_type::f16,
                            intel_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_ELTWISE(memory_desc_wrapper(diff_dst_md())
                            == memory_desc_wrapper(diff_src_md()),
                    VERBOSE_INCONSISTENT_MDS, "diff_src", "diff_dst");
            CHECK(init_conf(engine));
            return status::success;
        }

        status_t init_conf(const impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        CHECK(create_kernel(engine, &kernel_, "ref_eltwise_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward_dense(ctx);
    }

private:
    status_t execute_backward_dense(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace eltwise
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
