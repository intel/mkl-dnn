/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GPU_INTEL_GEMM_WITH_POST_OPS_HPP
#define GPU_INTEL_GEMM_WITH_POST_OPS_HPP

#include "gpu/intel/gemm/config.hpp"
#include "gpu/intel/gemm/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/subbyte_pack.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

struct with_post_ops_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public gemm::pd_t {
        using gemm::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:with_po:any", with_post_ops_t);

        status_t init(const impl::engine_t *engine);

        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        bool use_scratchpad() const {
            return use_scratchpad_with_post_op_worker;
        }
        status_t query(query_t what, int idx, void *result) const override {
            // Ranks are rejected here, so do not report the nested count.
            if (!pd_ || what == query::gpu_kernel_count)
                return gemm::pd_t::query(what, idx, result);
            return pd_->query(what, idx, result);
        }

        std::shared_ptr<primitive_desc_t> pd_;
        bool use_scratchpad_with_post_op_worker = false;
        bool use_reorder = false;
        compute::dispatch_t dispatch_;
        attr_info_t attr_info_;
        subbyte_pack_desc_t pack_desc_;
        bool dynamic_scales_ = false;
        bool with_dropout = false;
        bool dropout_use_host_scalars = false;
        bool dropout_use_offset = false;
        bool dropout_has_output_mask = false;
        data_type_t dst_type_ = data_type::undef;
        data_type_t acc_type_ = data_type::undef;
    };

    status_t init(impl::engine_t *engine) override {
        auto ret_status = create_nested_primitive(prim_, pd()->pd_, engine);
        CHECK(ret_status);
        primitive_attr_t attr;
        int grf_per_thread = 0;
        if (status::success
                == pd()->pd_->query(query::preferred_gpu_grf_per_thread, 0,
                        &grf_per_thread)) {
            CHECK(attr.set_gpu_attr(gpu_primitive_attr_t(grf_per_thread)));
        }
        compute::kernel_ctx_t kernel_ctx(&attr);
        CHECK(pd()->init_kernel_ctx(kernel_ctx));
        CHECK(create_kernel(engine, &kernels_[0], "gemm_post_ops", kernel_ctx));
        const bool dyn_scales = pd()->dynamic_scales_;
        if (dyn_scales) {
            compute::kernel_ctx_t alt_ctx(pd()->attr());
            const auto src_info = memory_desc_info_t::create(pd_->dst_md(0));
            dnnl_memory_desc dst_md(*(pd()->dst_md(0)));
            dst_md.data_type = pd()->dst_type_;
            memory_desc_wrapper dst_d(dst_md);
            def_memory_desc_info(alt_ctx, src_info, "SRC");
            def_memory_desc_info(
                    alt_ctx, memory_desc_info_t::create(dst_d), "DST");
            def_data_type(alt_ctx,
                    pd()->attr()->scales_.get_data_type(DNNL_ARG_DST),
                    "DST_SCALES");
            const int ndims = dst_d.ndims();
            bool runtime_dims
                    = pd()->has_runtime_dims_or_strides() || ndims > 5;
            if (!runtime_dims) {
                offsets_t off;
                set_offsets(dst_d, off.dst_off);
                def_offsets(off.dst_off, alt_ctx, "DST", ndims);
                alt_ctx.define_int("NDIMS", ndims);
            }
            CHECK(create_kernel(
                    engine, &kernels_[1], "dynamic_scale_dst", alt_ctx));
        }
        if (pd()->pack_desc_)
            CHECK(pack_.create(pd()->pack_desc_, *this, engine));
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    std::shared_ptr<impl::primitive_t> prim_;
    std::array<compute::kernel_t, 2> kernels_ = {};
    subbyte_pack_t pack_;
};

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
