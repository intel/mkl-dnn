/*******************************************************************************
* Copyright 2016 Intel Corporation
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

#ifndef CPU_REF_SOFTMAX_HPP
#define CPU_REF_SOFTMAX_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/primitive_attr_postops.hpp"

#include "cpu/cpu_softmax_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

struct ref_softmax_fwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_fwd_t);

        status_t init(const engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;

            VDISPATCH_SOFTMAX(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_SOFTMAX_SC(
                    platform::data_type_support_status(src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX_SC(
                    platform::data_type_support_status(dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(utils::one_of(src_md()->data_type, f32, bf16, f16,
                                      f8_e5m2, f8_e4m3, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(utils::one_of(dst_md()->data_type, f32, bf16, f16,
                                      f8_e5m2, f8_e4m3, s8, u8),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(skip_mask_t::scales
                            | skip_mask_t::post_ops | skip_mask_t::dropout),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
            VDISPATCH_SOFTMAX_SC(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);

            VDISPATCH_SOFTMAX(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);
            VDISPATCH_SOFTMAX(
                    attr_.set_default_formats(dst_md(0)) == status::success,
                    VERBOSE_UNSUPPORTED_POSTOP);
            CHECK(dropout_ok());

            nthr_ = 0;
            init_scratchpad();

            return status::success;
        }

        int nthr_; // To not exceed the limit in execute used for set up.

        bool need_intermediate_scratchpad() const {
            const auto src_dt = src_md()->data_type;
            const auto dst_dt = dst_md()->data_type;
            // Relaxed accumulation allows to downconvert intermediate results
            // directly from xf16 or xf8 to dst avoiding scratchpad memory.
            const bool relaxed_acc = src_dt == dst_dt
                    && !types::is_integral_dt(dst_dt)
                    && utils::one_of(attr()->acc_mode_,
                            accumulation_mode::relaxed, accumulation_mode::any);
            const bool need_scratchpad = dst_md()->data_type
                            != types::default_accum_data_type(
                                    src_md()->data_type, dst_md()->data_type)
                    && !relaxed_acc;
            return need_scratchpad;
        }

    private:
        void init_scratchpad() {
            auto scratchpad = scratchpad_registry().registrar();
            const dim_t in_s = inner_size();

            if (in_s > 1) {
                const dim_t ou_s = outer_size();
                scratchpad.template book<float>(
                        memory_tracking::names::key_softmax_reduction,
                        2 * in_s * ou_s);
            }

            if (need_intermediate_scratchpad()) {
                nthr_ = dnnl_get_max_threads();
                scratchpad.template book<char>(
                        memory_tracking::names::key_softmax_interim_store,
                        axis_size(true) * sizeof(float) * nthr_);
            }
        }

        status_t post_ops_ok() const {
            return ref_post_ops_t::post_ops_status(attr()->post_ops_);
        }

        status_t dropout_ok() const {
            if (attr_.dropout_.has_default_values()) return status::success;

            assert(memory_desc_wrapper(dst_md(0)).format_kind()
                    == format_kind::blocked);

            using namespace format_tag;
            // See `ref_dropout(...)` comment which explains the requirement.
            VDISPATCH_SOFTMAX_IC(memory_desc_matches_one_of_tag(
                                         *dst_md(0), ncdhw, nchw, ncw, nc)
                            && IMPLICATION(attr_.dropout_.has_output_mask(),
                                    memory_desc_wrapper(dst_md(0)).similar_to(
                                            attr_.dropout_.dropout_desc_, true,
                                            false)),
                    VERBOSE_UNSUPPORTED_DROPOUT);

            return status::success;
        }
    };

    ref_softmax_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        outer_size_ = pd()->outer_size();
        channels_ = pd()->axis_size();
        inner_size_ = pd()->inner_size();

        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper dst_d(pd()->dst_md());
        const auto &bd = src_d.blocking_desc();

        auto axis = pd()->axis();
        dim_t axis_blk_size = 1;
        for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
            if (bd.inner_idxs[iblk] == axis)
                axis_blk_size *= bd.inner_blks[iblk];

        use_dense_ = inner_size_ == 1 && src_d == dst_d && src_d.is_dense(true)
                && src_d.only_padded_dim(axis)
                && bd.strides[axis] == axis_blk_size;

        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        CHECK(ref_post_ops->init(pd()->dst_md()));

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (use_dense_)
            return execute_forward_dense(ctx);
        else
            return execute_forward_generic(ctx);
    }

private:
    status_t execute_forward_dense(const exec_ctx_t &ctx) const;
    status_t execute_forward_generic(const exec_ctx_t &ctx) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    std::unique_ptr<ref_post_ops_t> ref_post_ops;
    bool use_dense_;
    dim_t outer_size_, channels_, inner_size_;
};

struct ref_softmax_bwd_t : public primitive_t {
    struct pd_t : public cpu_softmax_bwd_pd_t {
        using cpu_softmax_bwd_pd_t::cpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_bwd_t);

        status_t init(const engine_t *engine) {
            using namespace data_type;
            VDISPATCH_SOFTMAX(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_SOFTMAX(
                    utils::one_of(dst_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    utils::one_of(diff_dst_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(
                    utils::one_of(diff_src_md()->data_type, f32, bf16, f16),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX_SC(
                    platform::data_type_support_status(dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX_SC(platform::data_type_support_status(
                                         diff_dst_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX_SC(platform::data_type_support_status(
                                         diff_src_md()->data_type),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_SOFTMAX(dst_md()->data_type == diff_dst_md()->data_type,
                    VERBOSE_INCONSISTENT_DT, "dst", "diff_dst");
            VDISPATCH_SOFTMAX(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_SOFTMAX(set_default_formats() == status::success,
                    VERBOSE_UNSUPPORTED_TAG);

            return status::success;
        }
    };

    ref_softmax_bwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        outer_size_ = pd()->outer_size();
        channels_ = pd()->axis_size();
        inner_size_ = pd()->inner_size();

        const memory_desc_wrapper data_d(pd()->dst_md());
        const memory_desc_wrapper diff_d(pd()->diff_dst_md());
        const auto &bd = diff_d.blocking_desc();

        auto axis = pd()->axis();
        dim_t axis_blk_size = 1;
        for (int iblk = 0; iblk < bd.inner_nblks; ++iblk)
            if (bd.inner_idxs[iblk] == axis)
                axis_blk_size *= bd.inner_blks[iblk];

        use_dense_ = inner_size_ == 1 && diff_d == data_d && diff_d.is_dense()
                && bd.strides[axis] == axis_blk_size;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        if (use_dense_)
            return execute_backward_dense(ctx);
        else
            return execute_backward_generic(ctx);
    }

private:
    status_t execute_backward_dense(const exec_ctx_t &ctx) const;
    status_t execute_backward_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    bool use_dense_;
    dim_t outer_size_, channels_, inner_size_;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
