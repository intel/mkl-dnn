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

#include "gpu/intel/gnorm/ref.hpp"

#include "gpu/intel/primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gnorm {

// define kernel runtime common environment variables
static status_t init_kernel_ctx_common(
        compute::kernel_ctx_t &kernel_ctx, const pd_t *pd) {

    const memory_desc_wrapper input_data_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    const memory_desc_wrapper output_data_mdw(
            pd->is_fwd() ? pd->dst_md() : pd->diff_dst_md());

    kernel_ctx.set_data_type(input_data_mdw.data_type());
    kernel_ctx.require_stateless_addressing(pd->has_large_buffers());

    kernel_ctx.define_int("NDIMS", input_data_mdw.ndims()); // for SRC_OFF macro
    kernel_ctx.define_int("G", pd->desc()->groups);
    kernel_ctx.define_int("MB", pd->MB());
    kernel_ctx.define_int("C", pd->C());
    kernel_ctx.define_int("D", pd->D());
    kernel_ctx.define_int("H", pd->H());
    kernel_ctx.define_int("W", pd->W());
    kernel_ctx.define_int("CALCULATE_STATS", !pd->stats_is_src());
    kernel_ctx.define_int("SAVE_STATS", pd->is_training());
    kernel_ctx.define_int("IS_FWD", pd->is_fwd());

    // used by SRC_OFF() macro in OpenCL
    const memory_desc_info_t input_data_mdw_info
            = memory_desc_info_t::create(input_data_mdw);
    const memory_desc_info_t output_data_mdw_info
            = memory_desc_info_t::create(output_data_mdw);
    def_memory_desc_info(kernel_ctx, input_data_mdw_info, "SRC");
    def_memory_desc_info(kernel_ctx, output_data_mdw_info, "DST");

    // create post-op macro required definitions
    CHECK(def_attr_info(kernel_ctx, attr_info_t::create(pd->attr()),
            pd->attr()->post_ops_, *pd->invariant_dst_md()));

    offsets_t off;
    set_offsets(input_data_mdw, off.src_off);
    def_offsets(off.src_off, kernel_ctx, "SRC", input_data_mdw.ndims());

    return status::success;
}

status_t ref_fwd_t::pd_t::init(const impl::engine_t *engine) {
    using namespace data_type;

    using skip_mask_t = primitive_attr_t::skip_mask_t;

    data_type_t src_dt = src_md()->data_type;
    data_type_t dst_dt = dst_md()->data_type;

    VDISPATCH_GNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_GNORM(utils::one_of(src_dt, f32, bf16, f16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_GNORM(utils::one_of(dst_dt, f32, bf16, f16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);

    const skip_mask_t attr_mask = skip_mask_t::scales | skip_mask_t::post_ops;
    VDISPATCH_GNORM(
            attr()->has_default_values(attr_mask), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_GNORM(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);

    // post-op related checks and adjustments
    VDISPATCH_GNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_GNORM(post_ops_with_binary_ok(attr(), *dst_md(), MAX_NDIMS),
            VERBOSE_UNSUPPORTED_TAG);
    CHECK(attr_.set_default_formats(
            dst_md(0))); // can't use attr() due to it is const

    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    dispatch = intel_engine->create_dispatch(src_md());

    dispatch.define_dim("BATCH", MB());
    // number of goups provided by the user
    dispatch.define_dim("NGROUPS", desc()->groups);
    dispatch.generate();

    return status::success;
}

status_t ref_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {

    kernel_ctx.define_int("WITH_SRC_SCALES",
            !attr()->scales_.has_default_values(DNNL_ARG_SRC));
    kernel_ctx.define_int("WITH_DST_SCALES",
            !attr()->scales_.has_default_values(DNNL_ARG_DST));
    CHECK(init_kernel_ctx_common(kernel_ctx, this));

    // promote macros defined by parameters to OpenCL command line
    def_dispatch(kernel_ctx, dispatch);

    return status::success;
}

status_t ref_fwd_t::execute(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    memory_storage_t &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    memory_storage_t &src = CTX_IN_STORAGE(DNNL_ARG_SRC);

    memory_storage_t &mean = pd()->stats_is_src()
            ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
            : CTX_OUT_STORAGE(DNNL_ARG_MEAN);
    memory_storage_t &variance = pd()->stats_is_src()
            ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
            : CTX_OUT_STORAGE(DNNL_ARG_VARIANCE);

    memory_storage_t &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    memory_storage_t &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);

    memory_storage_t &src_scale
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    memory_storage_t &dst_scale
            = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src);
    arg_list.append(mean);
    arg_list.append(variance);
    arg_list.append(dst);
    arg_list.append(pd()->C() / pd()->G());
    arg_list.append(scale);
    arg_list.append(shift);
    arg_list.append(src_scale);
    arg_list.append(dst_scale);
    arg_list.append(pd()->desc()->group_norm_epsilon);

    append_post_ops_to_arg_list(ctx, arg_list, arg_list.nargs(),
            pd()->attr()->post_ops_, *pd()->dst_md());

    const compute::nd_range_t &nd_range_kernel = pd()->dispatch.nd_range();
    status_t status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

status_t ref_bwd_t::pd_t::init(const impl::engine_t *engine) {
    using namespace data_type;

    const data_type_t src_dt = src_md()->data_type;
    const data_type_t diff_dst_dt = diff_dst_md()->data_type;
    const data_type_t diff_src_dt = diff_src_md()->data_type;

    VDISPATCH_GNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);

    VDISPATCH_GNORM(
            utils::one_of(src_dt, f32, bf16, f16), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_GNORM(
            utils::one_of(diff_dst_dt, f32, bf16, f16), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_GNORM(
            utils::one_of(diff_src_dt, f32, bf16, f16), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_GNORM(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);

    VDISPATCH_GNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);

    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    dispatch = intel_engine->create_dispatch(diff_src_md());
    dispatch.define_dim("NGROUPS", desc()->groups);
    dispatch.generate();

    return status::success;
}

status_t ref_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {

    CHECK(init_kernel_ctx_common(kernel_ctx, this));

    // promote macros defined by parameters to OpenCL command line
    def_dispatch(kernel_ctx, dispatch);

    return status::success;
}

status_t ref_bwd_t::execute(const exec_ctx_t &ctx) const {

    if (pd()->has_zero_dim_memory()) { return status::success; }

    status_t status = status::success;

    memory_storage_t &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    memory_storage_t &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    memory_storage_t &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    memory_storage_t &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    memory_storage_t &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    memory_storage_t &diff_src = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SRC);
    memory_storage_t &diff_scale = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE);
    memory_storage_t &diff_shift = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SHIFT);

    compute::kernel_arg_list_t arg_list;
    arg_list.append(src);
    arg_list.append(mean);
    arg_list.append(variance);
    arg_list.append(diff_dst);
    arg_list.append(scale);
    arg_list.append(diff_src);
    arg_list.append(diff_scale);
    arg_list.append(diff_shift);
    arg_list.append(pd()->C() / pd()->G());
    arg_list.append(pd()->desc()->group_norm_epsilon);

    const compute::nd_range_t &nd_range_kernel = pd()->dispatch.nd_range();
    status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

} // namespace gnorm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
