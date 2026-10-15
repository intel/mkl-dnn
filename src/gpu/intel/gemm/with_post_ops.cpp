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

#include "gpu/intel/gemm/with_post_ops.hpp"
#include "gpu/intel/gemm/host_scalars.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

status_t with_post_ops_t::pd_t::init(const impl::engine_t *engine) {
    using namespace data_type;

    const auto &d = desc();
    using smask_t = primitive_attr_t::skip_mask_t;
    const auto attr_skip_mask = smask_t::scales_data_type
            | smask_t::scales_groups | smask_t::post_ops
            | smask_t::accumulation_mode | smask_t::fpmath_mode
            | smask_t::zero_points_data_type | smask_t::dropout
            | smask_t::gpu_attr;

    bool wei_decomp = (utils::one_of(d->c_type(), f32, f16, bf16)
                              && utils::one_of(d->a_type(), u8, s8, u4, s4, u2)
                              && utils::one_of(d->b_type(), f16, f32, bf16))
            && attr()->mayiconvert(d->a_type(), f32);
    VDISPATCH_GEMM(
            d->c_desc.ndims <= 6, VERBOSE_UNSUPPORTED_MD_FLAG, "c_desc.ndims");
    VDISPATCH_GEMM(!utils::one_of(DNNL_RUNTIME_DIM_VAL, d->m(), d->n(), d->k()),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_GEMM(attr()->has_default_values(attr_skip_mask),
            VERBOSE_UNSUPPORTED_ATTR);
    // A rank indexes this problem's candidates, not the nested gemm's.
    VDISPATCH_GEMM(kernel_rank(attr()) < 0,
            "kernel rank override through post-op decomposition");
    VDISPATCH_GEMM(
            !utils::one_of(d->c_type(), u4, s4, u2), VERBOSE_UNSUPPORTED_DT);

    // gemm_post_ops kernel supports only dst zero-point,
    // host scalar is also supported for dst zp
    const auto &zps = attr()->zero_points_;
    VDISPATCH_GEMM(!(zps.get(DNNL_ARG_SRC).is_host_scalar()
                           || zps.get(DNNL_ARG_WEIGHTS).is_host_scalar()),
            VERBOSE_UNSUPPORTED_ZP_CFG);
    const auto &dst_zps = zps.get(DNNL_ARG_DST);
    VDISPATCH_GEMM(
            IMPLICATION(!dst_zps.has_default_values(), dst_zps.get_mask() == 0),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    const primitive_attr_t *attributes_with_po = attr();
    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST}) {
        if (attr()->scales_.has_default_values(arg)) continue;

        const auto &mask = attr()->scales_.get_mask(arg);
        if (arg == DNNL_ARG_WEIGHTS && !wei_decomp) {
            VDISPATCH_GEMM((mask == 0 || mask == (1 << (dst_md()->ndims - 1))),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        } else if (arg == DNNL_ARG_DST
                && attr()->scales_.get(arg).is_dynamic()) {
            VDISPATCH_GEMM(utils::one_of(d->a_type(), f4_e2m1, f8_e5m2, f8_e4m3)
                            && utils::one_of(
                                    d->b_type(), f4_e2m1, f8_e5m2, f8_e4m3),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        } else
            VDISPATCH_GEMM((mask == 0), VERBOSE_UNSUPPORTED_SCALES_CFG);
    }
    attr_info_ = attr_info_t::create(attributes_with_po);

    const auto &po = attributes_with_po->post_ops_;
    for (auto i = 0; i < po.len(); ++i)
        VDISPATCH_GEMM(!po.entry_[i].is_binary_with_ternary_op(),
                VERBOSE_UNSUPPORTED_POSTOP);

    VDISPATCH_GEMM(d->sum_ab == sum_ab::sum_none, VERBOSE_UNSUPPORTED_FEATURE,
            "bias reduction");

    const auto impl_list = engine->get_implementation_list(op_desc());
    int current_impl_idx
            = impl_list_item_t::find<with_post_ops_t::pd_t>(impl_list);

    primitive_desc_iterator_t it_with_po(engine, op_desc(), attributes_with_po,
            nullptr, current_impl_idx /* skip implementation */);
    if (!it_with_po.is_initialized()) return status::invalid_arguments;
    pd_ = *(++it_with_po);
    // exit if gemm kernel support post ops
    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);
    auto arch = intel_engine->device_info()->gpu_arch();
    bool is_xe_hp = arch >= compute::gpu_arch_t::xe_hp;
    auto skip_impl = is_xe_hp ? "ocl" : "ref";
    VDISPATCH_GEMM(!(pd_ && strstr(pd_->name(), skip_impl) == nullptr),
            VERBOSE_SKIP_PRIMITIVE_IMPL);
    auto desc = *this->desc();
    dst_type_ = desc.c_desc.data_type;
    // Force f32 destination even if f16 accumulation mode is used.
    desc.c_desc.data_type
            = desc.acc_type == data_type::f16 ? data_type::f32 : desc.acc_type;
    acc_type_ = desc.c_desc.data_type;
    use_reorder = dst_md(0)->data_type != desc.c_desc.data_type;
    desc.bias_desc = glob_zero_md;
    // Setup empty attributes but keep zero points for gemm.
    primitive_attr_t attributes_without_po = *attr();
    CHECK(attributes_without_po.set_post_ops(post_ops_t()));
    attributes_without_po.scales_ = scales_t();
    attributes_without_po.zero_points_ = zero_points_t();
    attributes_without_po.dropout_ = dropout_t();
    const auto &zp = attributes_with_po->zero_points_;
    int src_mask = zp.get_mask(DNNL_ARG_SRC);
    int wei_mask = zp.get_mask(DNNL_ARG_WEIGHTS);
    if (!zp.has_default_values(DNNL_ARG_SRC)) {
        const auto dt = attr()->zero_points_.get_data_type(DNNL_ARG_SRC);
        CHECK(attributes_without_po.zero_points_.set(
                DNNL_ARG_SRC, src_mask, dt, 0, {}));
    }
    if (!zp.has_default_values(DNNL_ARG_WEIGHTS)) {
        const auto dt = attr()->zero_points_.get_data_type(DNNL_ARG_WEIGHTS);
        CHECK(attributes_without_po.zero_points_.set(
                DNNL_ARG_WEIGHTS, wei_mask, dt, 0, {}));
    }

    primitive_desc_iterator_t it_without_po(engine,
            reinterpret_cast<const op_desc_t *>(&desc), &attributes_without_po,
            nullptr, current_impl_idx /* skip implementation */);
    if (!it_without_po.is_initialized()) return status::invalid_arguments;
    pd_ = *(++it_without_po);
    VDISPATCH_GEMM(!(!pd_ || strstr(pd_->name(), skip_impl) != nullptr),
            VERBOSE_PRIMITIVE_CREATION_FAIL, pd_ ? pd_->name() : "");

    //set tags for end user
    desc_.a_desc = *pd_->arg_md(DNNL_ARG_SRC_0);
    desc_.b_desc = *pd_->arg_md(DNNL_ARG_SRC_1);
    desc_.c_desc = *pd_->arg_md(DNNL_ARG_DST);
    desc_.c_desc.data_type = dst_type_;
    desc_.acc_type = desc.c_desc.data_type;
    CHECK(attr_.set_default_formats(dst_md(0)));
    VDISPATCH_GEMM(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);

    CHECK(pack_desc_.init(*dst_md(0)));
    dynamic_scales_ = attr()->scales_.get(DNNL_ARG_DST).is_dynamic();
    VDISPATCH_GEMM(IMPLICATION(bool(pack_desc_) || dynamic_scales_,
                           attr()->post_ops_.find(primitive_kind::sum) == -1),
            VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_GEMM(IMPLICATION(dynamic_scales_,
                           !memory_desc_wrapper(dst_md(0))
                                    .has_runtime_dims_or_strides()),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    use_scratchpad_with_post_op_worker = use_reorder
            || attributes_with_po->post_ops_.find(primitive_kind_t::dnnl_sum)
                    != -1;
    with_dropout = !attr()->dropout_.has_default_values();
    if (with_dropout) {
        dropout_use_host_scalars = attr()->dropout_.use_host_scalars_;
        dropout_use_offset = attr()->dropout_.use_offset_;
        dropout_has_output_mask = attr()->dropout_.has_output_mask();
        assert(memory_desc_wrapper(dst_md(0)).format_kind()
                == format_kind::blocked);
        using namespace format_tag;
        // Note: for `offset = 0` keep the legacy logic without the `offset`.
        VDISPATCH_GEMM_IC(
                memory_desc_matches_one_of_tag(*dst_md(0), ncdhw, nchw, ncw, nc)
                        && IMPLICATION(dropout_has_output_mask,
                                memory_desc_wrapper(dst_md(0)).similar_to(
                                        attr()->dropout_.dropout_desc_, true,
                                        false)),
                VERBOSE_UNSUPPORTED_DROPOUT);
    }
    auto ndims = pd_->dst_md()->ndims;
    dispatch_ = intel_engine->create_dispatch(pd_->dst_md());
    dispatch_.define_dim("D0", 0, pd_->dst_md()->padded_dims[0]);
    dispatch_.define_dim("D1", 1, pd_->dst_md()->padded_dims[1]);
    dispatch_.define_dim("D2", ndims > 2 ? 2 : 0,
            ndims > 2 ? pd_->dst_md()->padded_dims[2] : 1);
    dispatch_.define_dim("D3", ndims > 3 ? 3 : 0,
            ndims > 3 ? pd_->dst_md()->padded_dims[3] : 1);
    dispatch_.define_dim("D4", ndims > 4 ? 4 : 0,
            ndims > 4 ? pd_->dst_md()->padded_dims[4] : 1);
    dispatch_.define_dim("D5", ndims > 5 ? 5 : 0,
            ndims > 5 ? pd_->dst_md()->padded_dims[5] : 1);
    dispatch_.generate();

    init_scratchpad();

    return status::success;
}
status_t with_post_ops_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    auto c_type = dst_md(0)->data_type;
    const auto src_info = memory_desc_info_t::create(pd_->dst_md(0));
    const auto bias_info = [&]() {
        // If no bias, just default to same layout as dst - any valid layout will work, it's just a dummy
        auto info = memory_desc_info_t::create(
                with_bias() ? src_md(2) : dst_md(0));
        if (info.data_type == data_type::undef) info.data_type = data_type::f32;
        return info;
    }();

    def_memory_desc_info(kernel_ctx, src_info, "SRC");
    def_memory_desc_info(kernel_ctx, bias_info, "BIAS");
    if (dynamic_scales_) {
        dnnl_memory_desc d_md(*dst_md(0));
        d_md.data_type = acc_type_;
        memory_desc_wrapper d_mdw(d_md);
        def_memory_desc_info(
                kernel_ctx, memory_desc_info_t::create(d_mdw), "DST");
    } else {
        def_memory_desc_info(
                kernel_ctx, memory_desc_info_t::create(dst_md(0)), "DST");
    }

    int ndims = src_info.ndims;
    kernel_ctx.set_data_type(dynamic_scales_ ? acc_type_ : c_type);
    kernel_ctx.require_stateless_addressing(has_large_buffers());
    kernel_ctx.register_buffer_size(pack_desc_.span(), pack_desc_.span());

    const auto &attr_scales = attr()->scales_;
    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const bool with_dst_scales = !attr_scales.has_default_values(DNNL_ARG_DST);
    auto is_int_type = [](data_type_t t) {
        return utils::one_of(t, data_type::s8, data_type::u8, data_type::s32);
    };
    data_type_t acc_type = desc_.acc_type;
    if (desc_.acc_type == data_type::s32) {
        if (with_src_scales || with_wei_scales
                || !is_int_type(bias_info.data_type)
                || !is_int_type(dst_md(0)->data_type)) {
            acc_type = data_type::f32;
        }
    }
    data_type_t seed_type = attr()->dropout_.seed_dt_;
    bool with_seed_s64 = attr()->dropout_.seed_dt_ == data_type::s64;

    def_data_type(kernel_ctx, acc_type, "ACC");
    def_data_type(kernel_ctx, seed_type, "SEED");

    kernel_ctx.define_int("NDIMS", ndims);
    CHECK(def_attr_info(
            kernel_ctx, attr_info_, attr()->post_ops_, *pd_->dst_md()));
    kernel_ctx.define_int("A_SCALES", with_src_scales);
    kernel_ctx.define_int("B_SCALES", with_wei_scales);
    kernel_ctx.define_int("C_SCALES", with_dst_scales);
    kernel_ctx.define_int("DST_ZERO_POINT",
            !attr()->zero_points_.has_default_values(DNNL_ARG_DST));
    kernel_ctx.define_int("WITH_DROPOUT", with_dropout);
    kernel_ctx.define_int("WITH_SEED_S64", with_seed_s64);
    kernel_ctx.define_int("DROPOUT_USE_HOST_SCALARS", dropout_use_host_scalars);
    kernel_ctx.define_int("DROPOUT_USE_OFFSET", dropout_use_offset);
    kernel_ctx.define_int("DROPOUT_HAS_OUTPUT_MASK", dropout_has_output_mask);
    def_dispatch(kernel_ctx, dispatch_);
    return status::success;
}

void with_post_ops_t::pd_t::init_scratchpad() {
    auto scratchpad = scratchpad_registry().registrar();
    const size_t dst_span = memory_desc_wrapper(dst_md(0)).span();
    if (pack_desc_) {
        scratchpad.book(memory_tracking::names::key_matmul_pack_space,
                pack_desc_.span(), sizeof(char), OCL_BUFFER_ALIGNMENT);
    }
    if (dynamic_scales_) {
        scratchpad.book(memory_tracking::names::key_matmul_dyn_scale_space,
                dst_span, sizeof(float), OCL_BUFFER_ALIGNMENT);
    }
    if (use_scratchpad_with_post_op_worker) {
        scratchpad.book(memory_tracking::names::key_gemm_tmp_buffer, dst_span,
                types::data_type_size(desc_.acc_type), OCL_BUFFER_ALIGNMENT);
    }
    scratchpad.book(memory_tracking::names::key_nested_multiple,
            pd_->scratchpad_registry());
}

status_t with_post_ops_t::execute(const exec_ctx_t &ctx) const {
    std::unique_ptr<memory_t, memory_deleter_t> c_mem_before_po_worker;
    exec_args_t nested_args(ctx.args());

    if (pd()->use_scratchpad()) {
        auto scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_tmp_buffer);
        auto tmp_md = *(pd()->dst_md(0));
        tmp_md.data_type = pd()->desc()->acc_type;
        CHECK(safe_ptr_assign(c_mem_before_po_worker,
                new memory_t(ctx.stream()->engine(), &tmp_md,
                        std::move(scratchpad))));

        nested_args.c = c_mem_before_po_worker->memory_storage();
    }

    exec_ctx_t nested_ctx(ctx, nested_args, ctx.desc());
    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            memory_tracking::names::key_nested_multiple,
            prim_->pd()->scratchpad_registry());
    nested_ctx.set_scratchpad_grantor(nested_grantor);

    CHECK(gemm(prim_)->execute(nested_ctx));

    const bool dyn_scales = pd()->dynamic_scales_;

    auto tmp = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_matmul_pack_space);

    auto tmp_ds = ctx.get_scratchpad_grantor().get_memory_storage(
            memory_tracking::names::key_matmul_dyn_scale_space);

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0,
            pd()->use_scratchpad() ? *c_mem_before_po_worker->memory_storage()
                                   : GEMM_CTX_ARG_STORAGE(c));
    arg_list.set(1, GEMM_CTX_ARG_STORAGE(bias));
    arg_list.set(2,
            dyn_scales      ? *tmp_ds
                    : pack_ ? *tmp
                            : GEMM_CTX_ARG_STORAGE(c));
    const auto &args = ctx.args();
    int idx = append_post_ops_to_arg_list(args.exec_args, arg_list, 3,
            pd()->attr()->post_ops_, *pd()->dst_md());
    //a/b tensors are swapped for gemm
    arg_list.set(idx++, GEMM_CTX_ARG_STORAGE(b_scales));
    arg_list.set(idx++, GEMM_CTX_ARG_STORAGE(a_scales));
    arg_list.set(idx++, GEMM_CTX_ARG_STORAGE(c_scales));
    arg_list.set(idx++,
            pd()->attr()->scales_.get_mask(DNNL_ARG_WEIGHTS) > 0 ? 1 : 0);
    arg_list.set(idx, GEMM_CTX_ARG_STORAGE(c_zero_point));
    if (pd()->with_dropout) {
        const auto mem_dropout_seed = &GEMM_CTX_ARG_STORAGE(dropout_seed);
        const auto mem_dropout_offset = &GEMM_CTX_ARG_STORAGE(dropout_offset);
        const auto mem_dropout_prob = &GEMM_CTX_ARG_STORAGE(dropout_prob);
        idx++;
        arg_list.set(idx++, GEMM_CTX_ARG_STORAGE(dropout_mask));
        if (pd()->dropout_use_host_scalars) {
            int64_t scalar_dropout_seed = 0;
            int64_t scalar_dropout_offset = 0;
            float scalar_dropout_prob = 0.f;
            CHECK(maybe_get_host_scalar_value(
                    *mem_dropout_seed, scalar_dropout_seed));
            if (pd()->dropout_use_offset) {
                CHECK(maybe_get_host_scalar_value(
                        *mem_dropout_offset, scalar_dropout_offset));
            }
            CHECK(maybe_get_host_scalar_value(
                    *mem_dropout_prob, scalar_dropout_prob));
            arg_list.set(idx++, scalar_dropout_seed);
            arg_list.set(idx++, scalar_dropout_offset);
            arg_list.set(idx, scalar_dropout_prob);
        } else {
            arg_list.set(idx++, *mem_dropout_seed);
            arg_list.set(idx++, *mem_dropout_offset);
            arg_list.set(idx, *mem_dropout_prob);
        }
    }
    auto nd_range = pd()->dispatch_.nd_range();
    CHECK(parallel_for(ctx, nd_range, kernels_[0], arg_list));

    if (dyn_scales) {
        const auto group_size
                = pd()->attr()->scales_.get_group(DNNL_ARG_DST, -1);
        const auto c_d = nested_ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
        const int last = c_d.ndims() - 1;
        const dim_t D3 = c_d.ndims() > 5 ? c_d.dims()[last - 5] : 1;
        const dim_t D2 = c_d.ndims() > 4 ? c_d.dims()[last - 4] : 1;
        const dim_t D1 = c_d.ndims() > 3 ? c_d.dims()[last - 3] : 1;
        const dim_t D0 = c_d.ndims() > 2 ? c_d.dims()[last - 2] : 1;
        const dim_t M = c_d.dims()[last - 1];
        const dim_t N = c_d.dims()[last];
        dnnl_dims_t c_stride {0};
        const auto &c_strides = c_d.blocking_desc().strides;
        for (int i = 0; i < c_d.ndims(); i++)
            if (c_d.dims()[last - i] > 1) { c_stride[i] = c_strides[last - i]; }

        compute::kernel_arg_list_t arg_list;
        int arg_idx = 0;
        arg_list.set(arg_idx++, *tmp_ds);
        arg_list.set(arg_idx++, pack_ ? *tmp : GEMM_CTX_ARG_STORAGE(c));
        arg_list.set(arg_idx++, GEMM_CTX_ARG_STORAGE(c_scales));
        arg_list.set(arg_idx++, group_size);
        arg_list.set(arg_idx++, D0);
        arg_list.set(arg_idx++, D1);
        arg_list.set(arg_idx++, D2);
        arg_list.set(arg_idx++, c_stride[5]);
        arg_list.set(arg_idx++, c_stride[4]);
        arg_list.set(arg_idx++, c_stride[3]);
        arg_list.set(arg_idx++, c_stride[2]);
        arg_list.set(arg_idx++, c_stride[1]);
        arg_list.set(arg_idx++, c_stride[0]);
        compute::range_t gws({(size_t)M, (size_t)N / group_size,
                (size_t)(D0 * D1 * D2 * D3)});
        compute::nd_range_t nd_range(gws);
        CHECK(parallel_for(ctx, nd_range, kernels_[1], arg_list));
    }
    if (!pack_) return status::success;
    return pack_(impl::exec_ctx_t(ctx.stream()), *tmp, GEMM_CTX_ARG_STORAGE(c));
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
