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

#include "gpu/intel/lnorm/ref.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_exec_types.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace lnorm {

static status_t init_conf_common(
        conf_t &conf, const pd_t *pd, const impl::engine_t *engine) {
    using namespace dnnl::impl::format_tag;

    memory_desc_wrapper src_mdw(
            pd->is_fwd() ? pd->src_md() : pd->diff_src_md());
    memory_desc_wrapper stat_mdw(pd->stat_md());
    memory_desc_wrapper dst_mdw(
            pd->is_fwd() ? pd->dst_md() : pd->diff_dst_md());

    int ndims = src_mdw.ndims();

    conf.src_dt = src_mdw.data_type();
    conf.dst_dt = dst_mdw.data_type();
    conf.require_stateless_addressing = pd->has_large_buffers();
    conf.ndims = ndims;
    conf.norm_axis = pd->norm_axis();
    conf.use_scale = pd->use_scale();
    conf.use_shift = pd->use_shift();
    conf.calculate_stats = !pd->stats_are_src();
    conf.save_stats = pd->is_training();
    conf.eps = pd->desc()->layer_norm_epsilon;
    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.stat_md_info = memory_desc_info_t::create(stat_mdw);
    conf.is_fwd = pd->is_fwd();
    conf.skip_mean = pd->skip_mean();
    conf.attr_info = attr_info_t::create(pd->attr());

    if (conf.use_shift || conf.use_scale) {
        memory_desc_wrapper weights_mdw(
                pd->is_fwd() ? pd->weights_md() : pd->diff_weights_md());
        conf.weights_data_type = weights_mdw.data_type();
    }

    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    conf.dispatch_scaleshift = intel_engine->create_dispatch();
    conf.dispatch = intel_engine->create_dispatch(
            pd->is_fwd() ? dst_mdw.md_ : src_mdw.md_);

    const auto &dims = pd->is_fwd() ? src_mdw.padded_dims() : dst_mdw.dims();

    if (pd->is_fwd()) {
        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            dim_t dim = (i < ndims - 1) ? dims[i] : 1;
            conf.dispatch.define_dim(utils::format("X%d", i), md_hint_idx, dim);
        }
    } else {
        conf.dispatch_scaleshift.define_dim("C", pd->norm_axis());
        for (int i = 0; i < 4; i++) {
            int md_hint_idx = nstl::min(i, ndims - 1);
            dim_t dim = (i < ndims - 1) ? dims[i] : 1;
            conf.dispatch.define_dim(utils::format("X%d", i), md_hint_idx, dim);
        }
        conf.dispatch_scaleshift.set_kernel_attr_suffix("SCALESHIFT");
        conf.dispatch_scaleshift.generate();
    }

    conf.dispatch.generate();
    return status::success;
}

static status_t init_kernel_ctx_common(compute::kernel_ctx_t &kernel_ctx,
        const conf_t &conf, const post_ops_t &post_ops = post_ops_t(),
        const memory_desc_t *dst_md = &glob_zero_md) {
    kernel_ctx.set_data_type(conf.is_fwd ? conf.src_dt : conf.dst_dt);
    def_data_type(kernel_ctx, conf.weights_data_type, "WEI");
    kernel_ctx.require_stateless_addressing(conf.require_stateless_addressing);

    kernel_ctx.define_int("C", conf.norm_axis);
    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.define_int("USE_SCALE", conf.use_scale);
    kernel_ctx.define_int("USE_SHIFT", conf.use_shift);
    kernel_ctx.define_int("CALCULATE_STATS", conf.calculate_stats);
    kernel_ctx.define_int("SAVE_STATS", conf.save_stats);
    kernel_ctx.define_int("IS_FWD", conf.is_fwd);
    kernel_ctx.define_int("IS_BWD", !conf.is_fwd);
    kernel_ctx.define_int("VECT_DT_N", conf.vect_dt_n);
    kernel_ctx.define_int("SKIP_MEAN", conf.skip_mean);

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");
    def_memory_desc_info(kernel_ctx, conf.stat_md_info, "STAT");

    def_dispatch(kernel_ctx, conf.dispatch);
    if (!conf.is_fwd) def_dispatch(kernel_ctx, conf.dispatch_scaleshift);

    CHECK(def_attr_info(kernel_ctx, conf.attr_info, post_ops, *dst_md));

    return status::success;
}

status_t ref_fwd_t::pd_t::init_conf(const impl::engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_fwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(
            kernel_ctx, conf, attr()->post_ops_, dst_md());
}

status_t ref_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    const auto &conf = pd()->conf;
    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_MEAN)
                                       : CTX_OUT_STORAGE(DNNL_ARG_MEAN);

    auto &variance = pd()->stats_are_src() ? CTX_IN_STORAGE(DNNL_ARG_VARIANCE)
                                           : CTX_OUT_STORAGE(DNNL_ARG_VARIANCE);

    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);
    auto &shift = CTX_IN_STORAGE(DNNL_ARG_SHIFT);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_DST);
    auto &src_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    auto &dst_scale = CTX_IN_STORAGE(DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, dst);
    arg_list.set(4, scale);
    arg_list.set(5, shift);
    arg_list.set(6, conf.eps);
    arg_list.set(7, src_scale);
    arg_list.set(8, dst_scale);

    append_post_ops_to_arg_list(
            ctx, arg_list, 9, pd()->attr()->post_ops_, *pd()->dst_md());

    auto nd_range_kernel = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

status_t ref_bwd_t::pd_t::init_conf(const impl::engine_t *engine) {
    return init_conf_common(conf, this, engine);
}

status_t ref_bwd_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    return init_kernel_ctx_common(kernel_ctx, conf);
}

status_t ref_bwd_t::execute_backward(const exec_ctx_t &ctx) const {
    if (pd()->has_zero_dim_memory()) return status::success;

    status_t status = status::success;

    const auto &conf = pd()->conf;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_SRC);
    auto &mean = CTX_IN_STORAGE(DNNL_ARG_MEAN);
    auto &variance = CTX_IN_STORAGE(DNNL_ARG_VARIANCE);
    auto &diff_dst = CTX_IN_STORAGE(DNNL_ARG_DIFF_DST);
    auto &scale = CTX_IN_STORAGE(DNNL_ARG_SCALE);

    auto &diff_src = CTX_OUT_CLEAN_STORAGE(DNNL_ARG_DIFF_SRC, status);
    CHECK(status);
    auto &diff_scale = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SCALE);
    auto &diff_shift = CTX_OUT_STORAGE(DNNL_ARG_DIFF_SHIFT);

    if (conf.use_scale || conf.use_shift) {
        compute::kernel_arg_list_t arg_list;
        arg_list.set(0, src);
        arg_list.set(1, mean);
        arg_list.set(2, variance);
        arg_list.set(3, diff_dst);
        arg_list.set(4, diff_scale);
        arg_list.set(5, diff_shift);
        arg_list.set(6, conf.eps);

        auto nd_range = conf.dispatch_scaleshift.nd_range();
        status = parallel_for(ctx, nd_range, kernel_scaleshift_, arg_list);
        if (status != status::success) return status;
    }

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, mean);
    arg_list.set(2, variance);
    arg_list.set(3, diff_dst);
    arg_list.set(4, scale);
    arg_list.set(5, diff_src);
    arg_list.set(6, conf.eps);

    auto nd_range_kernel = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range_kernel, kernel_, arg_list);

    return status;
}

} // namespace lnorm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
