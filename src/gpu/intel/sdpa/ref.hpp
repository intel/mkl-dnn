/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GPU_INTEL_SDPA_REF_HPP
#define GPU_INTEL_SDPA_REF_HPP

#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/sdpa/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

struct ref_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public sdpa_fwd_pd_t {
        using sdpa_fwd_pd_t::sdpa_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_fwd_t);

        status_t init(const impl::engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            /* Reference SDPA is only enabled on-demand, for testing. */
            bool enable_ref = gpu_utils::dev_getenv("enable_ref_sdpa", false);
            VDISPATCH_SDPA(enable_ref, VERBOSE_SKIP_PRIMITIVE_IMPL);

            VDISPATCH_SDPA(attr()->has_default_values(smask_t::scales),
                    VERBOSE_UNSUPPORTED_ATTR);

            VDISPATCH_SDPA(!with_query_scales(), VERBOSE_UNSUPPORTED_ATTR);

            const auto is_fp8 = [](data_type_t dt) {
                return utils::one_of(dt, f8_e4m3, f8_e5m2);
            };
            VDISPATCH_SDPA(!is_fp8(desc()->qry_md()->data_type)
                            && !is_fp8(desc()->key_md()->data_type)
                            && !is_fp8(desc()->val_md()->data_type)
                            && !is_fp8(dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);

            VDISPATCH_SDPA(utils::everyone_is(4, desc()->qry_md()->ndims,
                                   desc()->key_md()->ndims,
                                   desc()->val_md()->ndims, dst_md()->ndims),
                    VERBOSE_SHAPE_RESTRICTION
                    ": qry(%d) key(%d) val(%d) dst(%d) must be 4d",
                    desc()->qry_md()->ndims, desc()->key_md()->ndims,
                    desc()->val_md()->ndims, dst_md()->ndims);
            if (with_attn_mask()) {
                VDISPATCH_SDPA(desc()->attn_mask_md()->ndims == 4,
                        VERBOSE_SHAPE_RESTRICTION ": attn_mask(%d) must be 4d",
                        desc()->attn_mask_md()->ndims);
            }
            VDISPATCH_SDPA(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        kernel_ctx.set_data_type(pd()->dst_md()->data_type);
        kernel_ctx.require_stateless_addressing(pd()->has_large_buffers());

        int ndims = 4;

        const memory_desc_wrapper qry_mdw(pd()->desc()->qry_md());
        const memory_desc_wrapper key_mdw(pd()->desc()->key_md());
        const memory_desc_wrapper val_mdw(pd()->desc()->val_md());
        const memory_desc_wrapper dst_mdw(pd()->dst_md());
        const memory_desc_wrapper msk_mdw(pd()->desc()->attn_mask_md());
        using offset_t = decltype(offsets_t().src_off);
        offset_t qry_off, key_off, val_off, dst_off, msk_off;
        set_offsets(qry_mdw, qry_off);
        set_offsets(key_mdw, key_off);
        set_offsets(val_mdw, val_off);
        set_offsets(dst_mdw, dst_off);
        set_offsets(msk_mdw, msk_off);
        def_offsets(qry_off, kernel_ctx, "QRY", ndims);
        def_offsets(key_off, kernel_ctx, "KEY", ndims);
        def_offsets(val_off, kernel_ctx, "VAL", ndims);
        def_offsets(dst_off, kernel_ctx, "DST", ndims);
        def_offsets(msk_off, kernel_ctx, "MSK", ndims);
        kernel_ctx.define_int("NDIMS", ndims);

        kernel_ctx.define_int("SIZE_K", pd()->desc()->keys());
        kernel_ctx.define_int("INVERT_SCALE", pd()->desc()->invert_scale);
        kernel_ctx.define_int("WITH_ATTN_SCALE", pd()->with_attn_scale());
        kernel_ctx.define_int("WITH_ATTN_MASK", pd()->with_attn_mask());

        def_data_type(kernel_ctx, pd()->desc()->qry_md()->data_type, "QRY");
        def_data_type(kernel_ctx, pd()->desc()->key_md()->data_type, "KEY");
        def_data_type(kernel_ctx, pd()->desc()->val_md()->data_type, "VAL");
        def_data_type(kernel_ctx, pd()->dst_md()->data_type, "DST");
        def_data_type(
                kernel_ctx, pd()->desc()->attn_mask_md()->data_type, "MSK");
        def_data_type(kernel_ctx, pd()->desc()->scale_md()->data_type, "SCALE");
        CHECK(create_kernel(engine, &kernel_, "ref_sdpa", kernel_ctx));
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    compute::kernel_t kernel_;
};

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
