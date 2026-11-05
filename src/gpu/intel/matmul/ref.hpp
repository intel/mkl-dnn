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

#ifndef GPU_INTEL_MATMUL_REF_HPP
#define GPU_INTEL_MATMUL_REF_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/host_scalar_memory_storage.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "gpu/intel/matmul/config.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/primitive_conf.hpp"
#include "gpu/intel/subbyte_pack.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace matmul {

struct ref_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public matmul::pd_t {
        using matmul::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:ref:any", ref_t);

        status_t init(const impl::engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            src_dt_ = src_md()->data_type;
            dst_dt_ = dst_md()->data_type;
            wei_dt_ = weights_md(0)->data_type;
            bia_dt_ = with_bias() ? weights_md(1)->data_type : data_type::f32;
            const auto *intel_engine
                    = utils::downcast<const intel::engine_t *>(engine);

            auto dev_info_ = intel_engine->device_info();

            VDISPATCH_MATMUL(
                    is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);
            VDISPATCH_MATMUL(
                    !with_reduce(), VERBOSE_UNSUPPORTED_FEATURE, "reduce");
            VDISPATCH_MATMUL(
                    attr()->has_default_values(smask_t::scales_data_type
                            | smask_t::scales_groups | smask_t::dropout
                            | smask_t::zero_points_data_type
                            | smask_t::zero_points_groups | smask_t::post_ops
                            | smask_t::sum_dt | smask_t::accumulation_mode
                            | smask_t::fpmath_mode | smask_t::rounding_mode
                            | smask_t::precomputed_reductions
                            | smask_t::post_ops_inplace),
                    VERBOSE_UNSUPPORTED_ATTR);
            CHECK(attr_scales_ok(engine,
                    {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST},
                    {quantization_mode::static_sazp,
                            quantization_mode::static_sbzp,
                            quantization_mode::dynamic_mx,
                            quantization_mode::dynamic_fp},
                    {{DNNL_ARG_SRC, {any_mask}}}));
            CHECK(attr_zero_points_ok(engine,
                    {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST},
                    {quantization_mode::static_sazp,
                            quantization_mode::static_sbzp}));
            VDISPATCH_MATMUL(
                    precomputed_reductions_ok(), VERBOSE_UNSUPPORTED_PR_CFG);
            VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_MATMUL(
                    IMPLICATION(has_blocks() || !attr()->has_default_values(),
                            dst_md()->ndims < 6),
                    VERBOSE_BAD_NDIMS, "dst", dst_md()->ndims);
            VDISPATCH_MATMUL(dst_md()->ndims <= 6, VERBOSE_BAD_NDIMS, "dst",
                    dst_md()->ndims);

            const bool is_f64
                    = utils::everyone_is(f64, src_dt_, wei_dt_, dst_dt_);
            const bool is_f32 = src_dt_ == f32
                    && utils::one_of(wei_dt_, f32, s8, u8, s4, u4, u2, u3);
            const bool is_f16 = src_dt_ == f16
                    && utils::one_of(wei_dt_, f16, s8, u8, s4, u4, u2, u3);
            const bool is_bf16 = src_dt_ == bf16
                    && utils::one_of(wei_dt_, bf16, s8, u8, s4, u4, u2, u3);

            const bool is_f8 = utils::one_of(src_dt_, f8_e5m2, f8_e4m3)
                    || utils::one_of(wei_dt_, f8_e5m2, f8_e4m3);
            const bool is_f4 = utils::one_of(src_dt_, f4_e2m1, f32, bf16, f16)
                    || utils::one_of(wei_dt_, f4_e2m1);
            const bool is_int8 = utils::one_of(src_dt_, u8, s8)
                    && utils::one_of(wei_dt_, u8, s8, u4, s4, u2, u3);
            // Note: fp4 bias will require sub-byte reads in the kernel.
            VDISPATCH_MATMUL(
                    (is_int8
                            || ((is_f32 || is_f64 || is_f16 || is_f8 || is_f4
                                        || is_bf16)
                                    && IMPLICATION(with_bias(),
                                            utils::one_of(bia_dt_, f32, f16,
                                                    bf16, f8_e5m2, f8_e4m3)))),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_MATMUL_SC(attr_.set_default_formats(dst_md(0)),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(post_ops_with_binary_ok(attr(), *dst_md(), 6),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(
                    IMPLICATION(utils::one_of(f64, src_dt_, wei_dt_, dst_dt_),
                            dev_info_->has_native(f64)),
                    VERBOSE_UNSUPPORTED_DT);
            CHECK(dropout_ok());
            CHECK(pack_desc_.init(*dst_md(0)));
            dynamic_scales_ = attr()->scales_.get(DNNL_ARG_DST).is_dynamic();
            VDISPATCH_MATMUL(
                    IMPLICATION(bool(pack_desc_),
                            attr()->post_ops_.get_sum_dt(dst_dt_) == dst_dt_),
                    VERBOSE_UNSUPPORTED_POSTOP);
            VDISPATCH_MATMUL(IMPLICATION(dynamic_scales_,
                                     !memory_desc_wrapper(dst_md(0))
                                              .has_runtime_dims_or_strides()),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);
            const size_t dst_span = memory_desc_wrapper(dst_md(0)).span();
            auto scratchpad = scratchpad_registry().registrar();
            if (dynamic_scales_) {
                scratchpad.book(
                        memory_tracking::names::key_matmul_dyn_scale_space,
                        dst_span, sizeof(float), OCL_BUFFER_ALIGNMENT);
            }
            if (pack_desc_) {
                scratchpad.book(memory_tracking::names::key_matmul_pack_space,
                        pack_desc_.span(), sizeof(char), OCL_BUFFER_ALIGNMENT);
            }

            non_default_attrs_ = !attr()->has_default_values();
            attr_info_ = attr_info_t::create(attr());

            return status::success;
        }

        bool non_default_attrs_ = false;
        subbyte_pack_desc_t pack_desc_;
        bool dynamic_scales_ = false;
        data_type_t bia_dt_ = data_type::undef;
        data_type_t src_dt_ = data_type::undef;
        data_type_t dst_dt_ = data_type::undef;
        data_type_t wei_dt_ = data_type::undef;

        attr_info_t attr_info_ = {};

    private:
        status_t dropout_ok() const {
            if (attr_.dropout_.has_default_values()) return status::success;

            assert(memory_desc_wrapper(dst_md(0)).format_kind()
                    == format_kind::blocked);

            using namespace format_tag;
            // Note: for `offset = 0` keep the legacy logic without the `offset`.
            VDISPATCH_MATMUL_IC(memory_desc_matches_one_of_tag(
                                        *dst_md(0), ncdhw, nchw, ncw, nc)
                            && IMPLICATION(attr_.dropout_.has_output_mask(),
                                    memory_desc_wrapper(dst_md(0)).similar_to(
                                            attr_.dropout_.dropout_desc_, true,
                                            false)),
                    VERBOSE_UNSUPPORTED_DROPOUT);

            return status::success;
        }
        bool precomputed_reductions_ok() const {
            const auto &pr = attr()->precomputed_reductions_;
            if (pr.has_default_values(DNNL_ARG_SRC)) return true;

            const auto &sc = attr()->scales_;
            const auto &zp = attr()->zero_points_;
            auto sgw = (!sc.has_default_groups(DNNL_ARG_WEIGHTS))
                    ? sc.get(DNNL_ARG_WEIGHTS).get_group(0)
                    : K();
            auto sgs = (!sc.has_default_groups(DNNL_ARG_SRC))
                    ? sc.get(DNNL_ARG_SRC).get_group(1)
                    : K();
            auto zgw = (!zp.has_default_groups(DNNL_ARG_WEIGHTS))
                    ? zp.get(DNNL_ARG_WEIGHTS).get_group(0)
                    : K();
            auto pgs = (!pr.has_default_groups(DNNL_ARG_SRC))
                    ? pr.get(DNNL_ARG_SRC).get_group(1)
                    : K();
            // all other groups should be divisible by the precomp group
            return (sgw % pgs == 0) && (sgs % pgs == 0) && (zgw % pgs == 0);
        }
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        bool with_seed_s64
                = (pd()->attr()->dropout_.seed_dt_) == data_type::s64;
        int ndims = pd()->dst_md()->ndims;
        kernel_ctx.define_int("DST_NDIMS", ndims);
        kernel_ctx.define_int("WITH_BIAS", pd()->with_bias());
        kernel_ctx.define_int("WITH_SEED_S64", with_seed_s64);
        kernel_ctx.define_int(
                "WITH_DROPOUT", !pd()->attr()->dropout_.has_default_values());
        kernel_ctx.define_int(
                "USE_HOST_SCALARS", pd()->attr()->dropout_.use_host_scalars_);
        kernel_ctx.define_int("USE_OFFSET", pd()->attr()->dropout_.use_offset_);
        kernel_ctx.define_int(
                "HAS_OUTPUT_MASK", pd()->attr()->dropout_.has_output_mask());
        kernel_ctx.define_int("NON_DEFAULT_ATTRS", pd()->non_default_attrs_);

        auto dst_rnd_mode = pd()->attr()->rounding_mode_.get(DNNL_ARG_DST);
        kernel_ctx.define_int(
                "WITH_SROUND", dst_rnd_mode == rounding_mode::stochastic);
        kernel_ctx.define_int("DST_DT_DIGITS",
                dnnl::impl::types::digits<uint32_t>(pd()->dst_dt_));

        kernel_ctx.set_data_type(pd()->dst_dt_);
        CHECK(def_attr_info(kernel_ctx, pd()->attr_info_,
                pd()->attr()->post_ops_, *pd()->dst_md()));
        kernel_ctx.require_stateless_addressing(pd()->has_large_buffers());
        kernel_ctx.register_buffer_size(
                pd()->pack_desc_.span(), pd()->pack_desc_.span());

        if (!pd()->attr()->precomputed_reductions_.has_default_values(
                    DNNL_ARG_SRC))
            kernel_ctx.define_int("WITH_SRC_GROUP_SUMS", 1);

        bool dyn_scales = pd()->attr()->scales_.get(DNNL_ARG_DST).is_dynamic();
        kernel_ctx.define_int("DYN_SCALES", dyn_scales);
        kernel_ctx.define_int("DST_PACKED", bool(pd()->pack_desc_));

        const memory_desc_wrapper src_d(pd()->src_md(0));
        const memory_desc_wrapper wei_d(pd()->weights_md(0));
        const memory_desc_wrapper dst_d(pd()->dst_md(0));
        // Plain layouts use runtime strides to maximize kernel reuse; blocked
        // layouts require compile-time offset macros.
        const bool all_plain
                = src_d.is_plain() && wei_d.is_plain() && dst_d.is_plain();
        bool runtime_dims
                = pd()->has_runtime_dims_or_strides() || ndims > 5 || all_plain;
        if (!runtime_dims) {
            offsets_t off;
            set_offsets(src_d, off.src_off);
            set_offsets(wei_d, off.wei_off);
            set_offsets(dst_d, off.dst_off);
            def_offsets(off.src_off, kernel_ctx, "SRC", ndims);
            def_offsets(off.wei_off, kernel_ctx, "WEI", ndims);
            def_offsets(off.dst_off, kernel_ctx, "DST", ndims);
            kernel_ctx.define_int("NDIMS", ndims);
        }
        kernel_ctx.define_int("RUNTIME_DIMS", runtime_dims);

        def_data_type(kernel_ctx, pd()->src_dt_, "SRC");
        def_data_type(kernel_ctx, pd()->wei_dt_, "WEI");
        def_data_type(kernel_ctx, pd()->dst_dt_, "DST");
        def_data_type(kernel_ctx, pd()->bia_dt_, "BIA");
        data_type_t acc_type = pd()->desc()->accum_data_type;
        data_type_t seed_type = pd()->attr()->dropout_.seed_dt_;
        switch (pd()->attr()->acc_mode_) {
            case accumulation_mode::strict:
            case accumulation_mode::relaxed:
            case accumulation_mode::any: break;
            case accumulation_mode::f16: acc_type = data_type::f16; break;
            case accumulation_mode::f32: acc_type = data_type::f32; break;
            case accumulation_mode::s32: acc_type = data_type::s32; break;
            default: break;
        }
        def_data_type(kernel_ctx, acc_type, "ACC");
        def_data_type(kernel_ctx, seed_type, "SEED");
        def_data_type(kernel_ctx,
                pd()->attr()->scales_.get_data_type(DNNL_ARG_WEIGHTS),
                "WEI_SCALES");
        def_data_type(kernel_ctx,
                pd()->attr()->zero_points_.get_data_type(DNNL_ARG_WEIGHTS),
                "WEI_ZP");
        // Weights scales applied before zero-points.
        kernel_ctx.define_int("WITH_WEI_SBZP",
                pd()->attr()->zero_points_.get(DNNL_ARG_WEIGHTS).is_sbzp());
        def_data_type(kernel_ctx,
                pd()->attr()->scales_.get_data_type(DNNL_ARG_SRC),
                "SRC_SCALES");
        def_data_type(kernel_ctx,
                pd()->attr()->zero_points_.get_data_type(DNNL_ARG_SRC),
                "SRC_ZP");
        def_data_type(kernel_ctx,
                pd()->attr()->precomputed_reductions_.get_data_type(
                        DNNL_ARG_SRC),
                "SRC_GS");
        def_data_type(kernel_ctx,
                pd()->attr()->scales_.get_data_type(DNNL_ARG_DST),
                "DST_SCALES");
        def_data_type(kernel_ctx,
                pd()->attr_info_.sum_data_type == dnnl_data_type_undef
                        ? pd()->dst_dt_
                        : pd()->attr_info_.sum_data_type,
                "SUM");
        CHECK(create_kernel(engine, &kernels_[0], "ref_matmul", kernel_ctx));
        if (pd()->dynamic_scales_)
            CHECK(create_kernel(
                    engine, &kernels_[1], "dynamic_scale_dst", kernel_ctx));
        if (pd()->pack_desc_)
            CHECK(pack_.create(pd()->pack_desc_, *this, engine));
        if (!kernels_[0]) return status::runtime_error;
        if (pd()->dynamic_scales_ && !kernels_[1]) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_ref(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_ref(const exec_ctx_t &ctx) const;
    std::array<compute::kernel_t, 2> kernels_ = {};
    subbyte_pack_t pack_;
};

} // namespace matmul
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
