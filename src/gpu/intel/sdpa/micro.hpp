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

#ifndef GPU_INTEL_SDPA_MICRO_HPP
#define GPU_INTEL_SDPA_MICRO_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/gemm_types.hpp"
#include "common/math_utils.hpp"
#include "common/primitive.hpp"
#include "common/sdpa_pd.hpp"
#include "common/utils.hpp"
#include "gpu/intel/gemm/utils.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/sdpa/configs.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace sdpa {

#define VDISPATCH_SDPA(cond, msg, ...) \
    VCONDCHECK(primitive, create, dispatch, sdpa, (cond), \
            status::unimplemented, "%s," msg, this->info(engine), \
            ##__VA_ARGS__)

struct micro_fwd_params_t : trivially_serializable_t<micro_fwd_params_t> {

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names_fwd
                = {"micro_sdpa"};
        return kernel_names_fwd;
    }

    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t kernel_ctx;
        CHECK(get_kernel_ctx(kernel_ctx));
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
        return status;
    }

    status_t get_kernel_ctx(compute::kernel_ctx_t &) const;

    int ndims;
    data_type_t data_t;
    data_type_t dst_data_t, key_data_t, qry_data_t, val_data_t, msk_data_t;
    data_type_t key_scales_data_t, value_scales_data_t;
    data_type_t key_zp_data_t, value_zp_data_t;
    int kv_group_size;

    int q_align, k_align, v_align, a_align;
    bool transpose_k;
    uint8_t padding0[3] = {0};

    int kq_scale_mask, vs_scale_mask, kq_zp_mask, vs_zp_mask;
    int key_elements_per_byte, key_zp_elements_per_byte, val_elements_per_byte,
            val_zp_elements_per_byte;

    int key_group_size, val_group_size;
    data_type_t scale_data_t;

    int attn_mask_undef, attn_mask_buffer, attn_mask_top_left,
            attn_mask_bottom_right;
    bool invert_scale, with_attn_scale, with_host_scale, with_attn_mask,
            broadcast_mask_q, with_causal_mask;
    uint8_t padding1[2] = {0};
    int subgroup_size, d_max_kq, d_max_v;

    bool d_full, arch_gte_hpc;
    bool block_q, block_a, block_2d_a;
    bool prefetch_mask, prefetch_k0, prefetch_k, prefetch_v, prefetch_remainder;
    bool remainder_q;
    uint8_t padding2[5] = {0};
    int prefetch_d_max;
    int prefetch_v_max;

    bool softmax_inf_as_zero;
    bool q_arrive_await_barrier;
    bool use_systolic_ukernel;
    bool kq_f16_accumulate, vs_f16_accumulate;
    bool require_stateless_addressing;
    bool is_training;
    bool dropout, dropout_output_mask, dropout_offset, dropout_host_scalars;
    uint8_t padding3[1] = {0};

    micro_fwd_ukernel_params_t ukernel_config;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(micro_fwd_params_t);

struct micro_bwd_params_t : trivially_serializable_t<micro_bwd_params_t> {

    const std::vector<const char *> &get_kernel_names() const {
        static const std::vector<const char *> kernel_names_bwd
                = {"preprocess_Di", "micro_sdpa_bwd", "postprocess_dQ"};
        return kernel_names_bwd;
    }

    status_t create_generator(const intel::engine_t &engine,
            compute::kernel_bundle_t &bundle) const {
        compute::kernel_ctx_t kernel_ctx;
        CHECK(get_kernel_ctx(kernel_ctx));
        auto status = engine.create_kernel_bundle(
                bundle, get_kernel_names(), kernel_ctx);
        return status;
    }

    status_t get_kernel_ctx(compute::kernel_ctx_t &) const;

    int ndims;
    int kv_group_size;
    data_type_t data_t;
    data_type_t dst_data_t, key_data_t, qry_data_t, val_data_t, msk_data_t;

    int q_align, k_align, v_align, a_align;
    bool transpose_k;
    uint8_t padding0[3] = {0};

    int key_group_size, val_group_size;
    data_type_t scale_data_t;

    int attn_mask_undef, attn_mask_buffer, attn_mask_top_left,
            attn_mask_bottom_right;
    bool invert_scale, with_attn_scale, with_host_scale, with_attn_mask,
            broadcast_mask_q, with_causal_mask;
    uint8_t padding1[2] = {0};
    int subgroup_size, d_max;

    bool d_full, arch_gte_hpc;
    bool block_k, block_dK, block_dV;
    bool remainder_q;
    bool use_systolic_ukernel;
    bool with_dS;
    bool require_stateless_addressing;
    bool dropout, dropout_output_mask, dropout_offset, dropout_host_scalars;
    bool k_in_slm;
    bool direct_dQ;
    uint8_t padding2[1] = {0};

    micro_bwd_ukernel_params_t ukernel_config;
};
DNNL_ASSERT_TRIVIALLY_SERIALIZABLE(micro_bwd_params_t);

struct micro_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public sdpa_fwd_pd_t {
        using sdpa_fwd_pd_t::sdpa_fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:micro:reusable", micro_fwd_t);

        status_t init(const impl::engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            const auto *intel_engine
                    = utils::downcast<const intel::engine_t *>(engine);
            auto *dev_info = intel_engine->device_info();
            arch_ = dev_info->gpu_arch();
            sg_size_ = dev_info->min_subgroup_size();

            VDISPATCH_SDPA(is_fwd(), VERBOSE_BAD_PROPKIND);
            memory_desc_wrapper qry_mdw(desc()->qry_md());
            memory_desc_wrapper key_mdw(desc()->key_md());
            memory_desc_wrapper val_mdw(desc()->val_md());
            memory_desc_wrapper dst_mdw(dst_md());

            bool is_f32 = (qry_mdw.data_type() == data_type::f32);
            use_systolic_ukernel_
                    = intel_engine->mayiuse(compute::device_ext_t::
                                      intel_subgroup_matrix_multiply_accumulate)
                    && !is_f32; // f32 -> non-systolic kernel only

            VDISPATCH_SDPA(
                    utils::everyone_is(4, qry_mdw.ndims(), key_mdw.ndims(),
                            val_mdw.ndims(), dst_mdw.ndims()),
                    VERBOSE_SHAPE_RESTRICTION
                    ": qry(%d) key(%d) val(%d) and dst(%d) must be 4d",
                    qry_mdw.ndims(), key_mdw.ndims(), val_mdw.ndims(),
                    dst_mdw.ndims());

            VDISPATCH_SDPA(utils::everyone_is(true, qry_mdw.is_plain(),
                                   key_mdw.is_plain(), val_mdw.is_plain(),
                                   dst_mdw.is_plain()),
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SDPA(attr()->has_default_values(smask_t::dropout),
                    VERBOSE_UNSUPPORTED_DROPOUT);
            VDISPATCH_SDPA(!with_select_mask(),
                    "Select attention mask is not supported");
            if (with_attn_mask()) {
                VDISPATCH_SDPA(desc()->attn_mask_md()->ndims == 4,
                        VERBOSE_SHAPE_RESTRICTION ": attn_mask(%d) must be 4d",
                        desc()->attn_mask_md()->ndims);
                VDISPATCH_SDPA(
                        utils::one_of(
                                desc()->attn_mask_md()->dims[mask_q_index],
                                desc()->queries(), 1),
                        VERBOSE_INVALID_BROADCAST, "attn_mask", mask_q_index);
                VDISPATCH_SDPA(desc()->attn_mask_md()->dims[mask_k_index]
                                == desc()->keys(),
                        VERBOSE_INVALID_BROADCAST, "attn_mask", mask_k_index);
                if (desc()->qry_md()->data_type == data_type::f32) {
                    VDISPATCH_SDPA(desc()->attn_mask_md()->data_type
                                    == desc()->qry_md()->data_type,
                            "Mask data type(%s) should match Qry/Dst data "
                            "type(%s).",
                            dnnl_dt2str(desc()->attn_mask_md()->data_type),
                            dnnl_dt2str(desc()->qry_md()->data_type));
                } else {
                    VDISPATCH_SDPA((desc()->attn_mask_md()->data_type
                                           == desc()->qry_md()->data_type)
                                    || (desc()->attn_mask_md()->data_type
                                            == data_type::f32),
                            "Mask data type(%s) should be xf16 or f32 when "
                            "Qry/Dst(%s) is xf16.",
                            dnnl_dt2str(desc()->attn_mask_md()->data_type),
                            dnnl_dt2str(desc()->qry_md()->data_type));
                }
            }
            VDISPATCH_SDPA(
                    (utils::everyone_is(data_type::f16,
                             desc()->qry_md()->data_type, dst_md()->data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->qry_md()->data_type,
                                    dst_md()->data_type)
                            || utils::everyone_is(data_type::f32,
                                    desc()->qry_md()->data_type,
                                    dst_md()->data_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(utils::one_of(desc()->key_md()->data_type, f32, bf16,
                                   f16, u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(utils::one_of(desc()->val_md()->data_type, f32, bf16,
                                   f16, u8, s8, u4, s4),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            if (utils::one_of(desc()->key_md()->data_type, u4, s4)) {
                VDISPATCH_SDPA(desc()->keys() % 2 == 0,
                        "The number of keys must be an even size with the data "
                        "type is u4 or s4.");
            }

            if (utils::one_of(desc()->val_md()->data_type, u4, s4)) {
                VDISPATCH_SDPA(desc()->values() % 2 == 0,
                        "The number of values must be an even size with the "
                        "data type is u4 or s4.");
            }

            VDISPATCH_SDPA(
                    desc()->qry_md()->dims[1] >= desc()->key_md()->dims[1]
                            && desc()->qry_md()->dims[1]
                                    >= desc()->val_md()->dims[1],
                    "number of heads in query tensor(%ld) must be greater "
                    "than the number of heads in the key(%ld) and value(%ld) "
                    "tensors",
                    static_cast<long int>(desc()->qry_md()->dims[1]),
                    static_cast<long int>(desc()->key_md()->dims[1]),
                    static_cast<long int>(desc()->val_md()->dims[1]));

            // Asymmetric heads are only supported when the QK head size is at
            // least the V head size (d_qk >= d_v). The kernel config is chosen
            // from d_qk, so its V-side tile can only hold d_v when d_v <= d_qk.
            VDISPATCH_SDPA(desc()->head_size() >= desc()->values(),
                    "the QK head size(%ld) must be greater than or equal to"
                    " the V head size(%ld)",
                    static_cast<long int>(desc()->head_size()),
                    static_cast<long int>(desc()->values()));

            VDISPATCH_SDPA(utils::one_of(kq_acc_dt(), f16, f32),
                    "KQ accumulation data type should be f16 or f32");
            VDISPATCH_SDPA(utils::one_of(vs_acc_dt(), f16, f32),
                    "VS accumulation data type should be f16 or f32");

            int kq_scales_mask = desc()->kq_scales.get_mask();
            int kq_zp_mask = desc()->kq_zero_points.get_mask();
            if (!desc()->kq_scales.has_default_values()
                    && !desc()->kq_zero_points.has_default_values())
                VDISPATCH_SDPA(kq_scales_mask == kq_zp_mask,
                        "kq scales mask(%d) must equal kq zero point(%d) "
                        "mask",
                        kq_scales_mask, kq_zp_mask);
            if (!desc()->kq_scales.has_default_values())
                VDISPATCH_SDPA(utils::one_of(kq_scales_mask, 0, 1, 3, 11, 15),
                        "unsupported mask for kq matmul(%d). must be 0, 1, 3, "
                        "11, or 15",
                        kq_scales_mask);
            if (!desc()->kq_zero_points.has_default_values())
                VDISPATCH_SDPA(utils::one_of(kq_zp_mask, 0, 1, 3, 11, 15),
                        "unsupported mask for kq matmul(%d). must be 0, 1, 3, "
                        "11, or 15",
                        kq_zp_mask);

            int vs_scales_mask = desc()->vs_scales.get_mask();
            int vs_zp_mask = desc()->vs_zero_points.get_mask();
            if (!desc()->vs_scales.has_default_values()
                    && !desc()->vs_zero_points.has_default_values())
                VDISPATCH_SDPA(vs_scales_mask == vs_zp_mask,
                        "vs scales mask(%d) must equal vs zero point(%d) "
                        "mask",
                        vs_scales_mask, vs_zp_mask);
            if (!desc()->vs_scales.has_default_values())
                VDISPATCH_SDPA(utils::one_of(vs_scales_mask, 0, 1, 3, 7, 15),
                        "unsupported mask for vs matmul(%d). must be 0, 1, 3, "
                        "7, or 15",
                        vs_scales_mask);
            if (!desc()->vs_zero_points.has_default_values())
                VDISPATCH_SDPA(utils::one_of(vs_zp_mask, 0, 1, 3, 7, 15),
                        "unsupported mask for vs matmul(%d). must be 0, 1, 3, "
                        "7, or 15",
                        vs_zp_mask);

            /// NOTE: Limitation of microkernels
            if (utils::one_of(desc()->vs_zero_points.get_data_type(), s4, u4)) {
                VDISPATCH_SDPA(value_group_size() == 16,
                        "if vs zero points data type is s4 or u4 then the "
                        "group size(%d) must be 16.",
                        value_group_size());
            }

            if (!desc()->vs_scales.has_default_values()
                    || !desc()->vs_zero_points.has_default_values()) {
                int vgs = value_group_size();
                VDISPATCH_SDPA(utils::one_of(vs_scales_mask, 0, 1, 3)
                                || (math::is_pow2<int>(vgs)
                                        || vgs == desc()->val_md()->dims[3]),
                        "the value group size(%d) must be a power of 2 or "
                        "equal to the number of values(%ld).",
                        vgs, static_cast<long int>(desc()->val_md()->dims[3]));
            }

            VDISPATCH_SDPA(IMPLICATION(is_f32, with_causal_mask()),
                    "fused f32 SDPA only optimized for causal mask"); //TODO: update when performance improved
            VDISPATCH_SDPA(
                    IMPLICATION(arch() == compute::gpu_arch_t::xe_hpg
                                    && !attr()->dropout_.has_default_values(),
                            attr()->dropout_.use_host_scalars_),
                    "fused SDPA FWD with device dropout not supported "
                    "for xe_hpg");

            if (desc()->prop_kind == prop_kind::forward_training) {
                const memory_desc_wrapper stats_mdw(desc()->stats_md());
                if (!stats_mdw.is_zero()) {
                    bool dense = stats_mdw.is_plain();
                    if (dense) {
                        const auto &strides = stats_mdw.strides();
                        const auto *sdims = stats_mdw.dims();
                        dim_t expected = 1;
                        for (int i = stats_mdw.ndims() - 1; i >= 0; --i) {
                            if (strides[i] != expected) {
                                dense = false;
                                break;
                            }
                            expected *= (sdims[i] > 0 ? sdims[i] : 1);
                        }
                    }
                    VDISPATCH_SDPA(dense,
                            "fused sdpa training requires densely packed "
                            "softmax stats");
                }
            }

            CHECK(init_conf_microkernels(engine));
            CHECK(init_conf(engine));

            return status::success;
        }

        status_t set_default_formats() {
            CHECK(set_default_format(desc_.q_desc, false));
            CHECK(set_default_format(desc_.k_desc, true));
            CHECK(set_default_format(desc_.v_desc, false));
            CHECK(set_default_format(desc_.dst_desc, false));
            return status::success;
        }

        int sg_size() const { return sg_size_; }
        bool use_systolic_ukernel() const { return use_systolic_ukernel_; }

        // Block size for the Q/K head dim, baked into the kernel.
        int d_max_kq() const {
            int head_size = into<int>(desc()->head_size());
            for (int i = 32; i <= 1024; i *= 2)
                if (head_size <= i) return i;
            return head_size;
        }
        // Block size for the V/output head dim, baked into the kernel.
        int d_max_v() const {
            int v_size = into<int>(desc()->values());
            for (int i = 32; i <= 1024; i *= 2)
                if (v_size <= i) return i;
            return v_size;
        }
        // Alias kept for ukernel tileC/tileR sites that use the KQ dimension.
        int d_max() const { return d_max_kq(); }

        compute::gpu_arch_t arch() const { return arch_; }
        micro_fwd_params_t conf;

    private:
        int sg_size_ = 0;
        bool use_systolic_ukernel_ = true;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        status_t init_conf_microkernels(const impl::engine_t *engine);
        status_t init_conf(const impl::engine_t *engine);

        status_t set_default_format(memory_desc_t &md, bool allow_transpose) {
            using namespace format_tag;
            memory_desc_wrapper mdw(md);
            VCHECK_SDPA_UNIMPL(!mdw.format_any(), VERBOSE_UNSUPPORTED_TAG);
            VCHECK_SDPA_UNIMPL(is_md_gemm_compatible_plain_format(&md),
                    VERBOSE_UNSUPPORTED_TAG);
            VCHECK_SDPA_UNIMPL(
                    IMPLICATION(gemm_desc_t::get_trans(md) == dnnl_trans,
                            allow_transpose),
                    VERBOSE_UNSUPPORTED_TAG);
            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;

    compute::kernel_t kernel_;
};

struct micro_bwd_t : public primitive_t {

    using primitive_t::primitive_t;
    struct pd_t : public sdpa_bwd_pd_t {
        using sdpa_bwd_pd_t::sdpa_bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:micro:reusable", micro_bwd_t);

        status_t init(const impl::engine_t *engine) {
            using namespace data_type;

            const auto *intel_engine
                    = utils::downcast<const intel::engine_t *>(engine);
            auto *dev_info = intel_engine->device_info();
            arch_ = dev_info->gpu_arch();
            sg_size_ = dev_info->min_subgroup_size();

            VDISPATCH_SDPA(!is_fwd(), VERBOSE_BAD_PROPKIND);

            VDISPATCH_SDPA(arch_ != compute::gpu_arch_t::xe_hpg,
                    "fused SDPA BWD not supported for xe_hpg");

            bool is_f32 = (desc()->qry_md()->data_type == data_type::f32);
            use_systolic_ukernel_
                    = intel_engine->mayiuse(compute::device_ext_t::
                                      intel_subgroup_matrix_multiply_accumulate)
                    && !is_f32; // f32 -> non-systolic kernel only

            VDISPATCH_SDPA(utils::everyone_is(4, desc()->qry_md()->ndims,
                                   desc()->key_md()->ndims,
                                   desc()->val_md()->ndims, dst_md()->ndims),
                    VERBOSE_SHAPE_RESTRICTION
                    ": qry(%d) key(%d) val(%d) dst(%d) must be 4d",
                    desc()->qry_md()->ndims, desc()->key_md()->ndims,
                    desc()->val_md()->ndims, dst_md()->ndims);
            VDISPATCH_SDPA(
                    utils::everyone_is(4, desc()->diff_qry_md()->ndims,
                            desc()->diff_key_md()->ndims,
                            desc()->diff_val_md()->ndims, diff_dst_md()->ndims),
                    VERBOSE_SHAPE_RESTRICTION
                    ": diff_qry(%d) diff_key(%d) diff_val(%d) diff_dst(%d) "
                    "must be 4d",
                    desc()->diff_qry_md()->ndims, desc()->diff_key_md()->ndims,
                    desc()->diff_val_md()->ndims, diff_dst_md()->ndims);
            VDISPATCH_SDPA(!with_select_mask(),
                    "Select attention mask is not supported");
            if (with_attn_mask()) {
                VDISPATCH_SDPA(desc()->attn_mask_md()->ndims == 4,
                        VERBOSE_SHAPE_RESTRICTION ": attn_mask(%d) must be 4d",
                        desc()->attn_mask_md()->ndims);
                VDISPATCH_SDPA(
                        utils::one_of(
                                desc()->attn_mask_md()->dims[mask_q_index],
                                desc()->queries(), 1),
                        VERBOSE_INVALID_BROADCAST, "attn_mask", mask_q_index);
                VDISPATCH_SDPA(desc()->attn_mask_md()->dims[mask_k_index]
                                == desc()->keys(),
                        VERBOSE_INVALID_BROADCAST, "attn_mask", mask_k_index);
                VDISPATCH_SDPA(desc()->attn_mask_md()->data_type
                                == desc()->qry_md()->data_type,
                        "Mask data type should match Qry/Dst data type.");
            }
            VDISPATCH_SDPA(
                    (utils::everyone_is(data_type::f16,
                             desc()->qry_md()->data_type, dst_md()->data_type)
                            || utils::everyone_is(data_type::bf16,
                                    desc()->qry_md()->data_type,
                                    dst_md()->data_type)
                            || utils::everyone_is(data_type::f32,
                                    desc()->qry_md()->data_type,
                                    dst_md()->data_type)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(
                    utils::one_of(desc()->key_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(
                    utils::one_of(desc()->val_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SDPA(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SDPA(desc()->values() == desc()->head_size(),
                    "values does not match head size");

            VDISPATCH_SDPA(
                    desc()->qry_md()->dims[1] >= desc()->key_md()->dims[1]
                            && desc()->qry_md()->dims[1]
                                    >= desc()->val_md()->dims[1],
                    "number of heads in query tensor(%ld) must be greater "
                    "than the number of heads in the key(%ld) and value(%ld) "
                    "tensors",
                    static_cast<long int>(desc()->qry_md()->dims[1]),
                    static_cast<long int>(desc()->key_md()->dims[1]),
                    static_cast<long int>(desc()->val_md()->dims[1]));
            {
                memory_desc_wrapper diff_qry_mdw(desc()->diff_qry_md());
                memory_desc_wrapper diff_key_mdw(desc()->diff_key_md());
                memory_desc_wrapper diff_val_mdw(desc()->diff_val_md());
                memory_desc_wrapper diff_dst_mdw(diff_dst_md());
                VDISPATCH_SDPA(utils::everyone_is(true, diff_qry_mdw.is_plain(),
                                       diff_key_mdw.is_plain(),
                                       diff_val_mdw.is_plain(),
                                       diff_dst_mdw.is_plain()),
                        VERBOSE_UNSUPPORTED_TAG);
            }

            VDISPATCH_SDPA(utils::everyone_is(desc()->qry_md()->data_type,
                                   desc()->diff_qry_md()->data_type,
                                   desc()->diff_key_md()->data_type,
                                   desc()->diff_val_md()->data_type,
                                   diff_dst_md()->data_type),
                    "diff tensor data types must match qry data type(%s) "
                    " ?= dQ(%s), dK(%s), dV(%s), dO(%s)",
                    dnnl_dt2str(desc()->qry_md()->data_type),
                    dnnl_dt2str(desc()->diff_qry_md()->data_type),
                    dnnl_dt2str(desc()->diff_key_md()->data_type),
                    dnnl_dt2str(desc()->diff_val_md()->data_type),
                    dnnl_dt2str(diff_dst_md()->data_type));

            CHECK(init_default_ws());
            VDISPATCH_SDPA(compare_ws(hint_fwd_pd_), VERBOSE_WS_MISMATCH);

            CHECK(init_conf_microkernels(engine));
            CHECK(init_conf(engine));
            CHECK(init_scratchpad(engine));

            return status::success;
        }

        status_t set_default_formats() {
            CHECK(set_default_format(desc_.q_desc, false));
            CHECK(set_default_format(desc_.k_desc, true));
            CHECK(set_default_format(desc_.v_desc, false));
            CHECK(set_default_format(desc_.dst_desc, false));
            CHECK(set_default_format(desc_.diff_dst_desc, false));
            CHECK(set_default_format(desc_.diff_q_desc, false));
            CHECK(set_default_format(desc_.diff_k_desc, true));
            CHECK(set_default_format(desc_.diff_v_desc, false));
            return status::success;
        }

        int sg_size() const { return sg_size_; }
        bool use_systolic_ukernel() const { return use_systolic_ukernel_; }

        // Block size for head_size, which must be hard-coded into the kernel.
        int d_max() const {
            int head_size = into<int>(desc()->head_size());
            for (int i = 32; i <= 1024; i *= 2)
                if (head_size <= i) return i;
            return head_size;
        }

        compute::gpu_arch_t arch() const { return arch_; }
        micro_bwd_params_t conf;

    private:
        int sg_size_ = 0;
        bool use_systolic_ukernel_ = true;
        compute::gpu_arch_t arch_ = compute::gpu_arch_t::unknown;

        status_t init_scratchpad(const impl::engine_t *engine);
        status_t init_conf_microkernels(const impl::engine_t *engine);
        status_t init_conf(const impl::engine_t *engine);

        status_t set_default_format(memory_desc_t &md, bool allow_transpose) {
            using namespace format_tag;
            memory_desc_wrapper mdw(md);
            VCHECK_SDPA_UNIMPL(!mdw.format_any(), VERBOSE_UNSUPPORTED_TAG);
            VCHECK_SDPA_UNIMPL(is_md_gemm_compatible_plain_format(&md),
                    VERBOSE_UNSUPPORTED_TAG);
            VCHECK_SDPA_UNIMPL(
                    IMPLICATION(gemm_desc_t::get_trans(md) == dnnl_trans,
                            allow_transpose),
                    VERBOSE_UNSUPPORTED_TAG);
            return status::success;
        }
    };

    status_t init(impl::engine_t *engine) override;

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_backward(const exec_ctx_t &ctx) const;

    compute::kernel_t kernel_, preprocess_, postprocess_;
};

} // namespace sdpa
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
