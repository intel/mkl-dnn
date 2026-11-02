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

#include <assert.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/reorder.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"

#include "cpu/simple_layer_normalization.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;
using namespace data_type;

status_t simple_layer_normalization_fwd_t::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    const memory_desc_wrapper src_d(src_md());

    VDISPATCH_LNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_LNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
    VDISPATCH_LNORM(utils::one_of(src_md()->data_type, f32, bf16, f16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(utils::one_of(dst_md()->data_type, f32, bf16, f16, s8, u8),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM_SC(platform::data_type_support_status(src_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM_SC(platform::data_type_support_status(dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_FEATURE,
            "unsupported scale or shift data type");
    VDISPATCH_LNORM(attr()->has_default_values(
                            skip_mask_t::scales | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_LNORM(attr_scales_ok(), VERBOSE_UNSUPPORTED_SCALES_CFG);
    VDISPATCH_LNORM_SC(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_LNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LNORM(src_d.is_blocking_desc(), VERBOSE_BLOCKING_FAIL,
            "blocking descriptor fail");
    // plain format, last logical dim is last physical
    VDISPATCH_LNORM(src_d.blocking_desc().strides[ndims() - 1] == 1,
            VERBOSE_BLOCKING_FAIL, "bad stride value");
    VDISPATCH_LNORM(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    CHECK(fill_compatible_stats_md(*src_md(), reordered_stat_md_));

    if (reordered_stat_md_ != *stat_md() && !stats_are_tmp()) {
        CHECK(reorder_primitive_desc_create(reorder_pd_, engine,
                stats_are_src() ? stat_md() : &reordered_stat_md_,
                stats_are_src() ? &reordered_stat_md_ : stat_md()));
    }

    bool ok = attr_.set_default_formats(dst_md(0)) == status::success;
    VDISPATCH_LNORM(ok, VERBOSE_UNSUPPORTED_POSTOP);

    init_scratchpad();
    return status::success;
}

status_t simple_layer_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const bool use_scale = pd()->use_scale();
    const bool use_shift = pd()->use_shift();
    const bool skip_mean = pd()->skip_mean();

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(void *, DNNL_ARG_DST);

    auto scale = CTX_IN_MEM(const float *, DNNL_ARG_SCALE);
    auto shift = CTX_IN_MEM(const float *, DNNL_ARG_SHIFT);

    float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        mean = skip_mean ? nullptr
                         : scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = pd()->stats_are_src()
                ? const_cast<float *>(CTX_IN_MEM(const float *, DNNL_ARG_MEAN))
                : CTX_OUT_MEM(float *, DNNL_ARG_MEAN);
        variance = pd()->stats_are_src()
                ? const_cast<float *>(
                          CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE))
                : CTX_OUT_MEM(float *, DNNL_ARG_VARIANCE);
    }

    const float *src_scales
            = CTX_IN_MEM(const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const float *dst_scales
            = CTX_IN_MEM(const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const bool with_src_scales
            = !pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC);
    const bool with_dst_scales
            = !pd()->attr()->scales_.has_default_values(DNNL_ARG_DST);

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const float C_f = static_cast<float>(C);
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    const auto calculate_stats = !pd()->stats_are_src();
    const auto src_dt = pd()->src_md()->data_type;
    const auto dst_dt = pd()->dst_md()->data_type;
    const auto eps = pd()->desc()->layer_norm_epsilon;
    const auto save_stats = pd()->is_training();

    parallel(0, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        dim_t N_start = 0, N_end = 0;
        balance211(N, nthr, ithr, N_start, N_end);
        const char *const __restrict src_ptr
                = reinterpret_cast<const char *>(src)
                + N_start * C_padded * src_d.data_type_size();
        char *const __restrict dst_ptr = reinterpret_cast<char *>(dst)
                + N_start * C_padded * dst_d.data_type_size();
        float *const __restrict mean_ptr = skip_mean ? nullptr : &mean[N_start];
        float *const __restrict var_ptr = &variance[N_start];
        const size_t block_size = N_end - N_start;
        // Note: manual unrolling for scale and shift due to clang issue.
        //       see: CLANG_WA_01_SAFE_TO_USE_OMP_SIMD
        for (size_t offset = 0; offset < block_size; offset++) {
            float v_mean = 0, v_variance = 0;
            if (calculate_stats) {
                if (!skip_mean) {
                    PRAGMA_OMP_SIMD(reduction(+ : v_mean))
                    for (dim_t c = 0; c < C; ++c) {
                        float s = io::load_float_value(
                                src_dt, src_ptr, c + C * offset);
                        v_mean += s;
                    }
                    v_mean /= C_f;
                }

                PRAGMA_OMP_SIMD(reduction(+ : v_variance))
                for (dim_t c = 0; c < C; ++c) {
                    float s = io::load_float_value(
                            src_dt, src_ptr, c + C * offset);
                    float src_sub_mean = s - v_mean;
                    v_variance += src_sub_mean * src_sub_mean;
                }
                v_variance /= C_f;
            } else {
                if (!skip_mean) { v_mean = mean_ptr[offset]; }
                v_variance = var_ptr[offset];
            }

            const float inv_sqrtvar = 1.f / sqrtf(v_variance + eps);
            if (use_scale && use_shift) {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < C; ++c) {
                    const float sm = scale[c] * inv_sqrtvar;
                    const float sv = shift[c];
                    const size_t off = c + C * offset;
                    float s = io::load_float_value(src_dt, src_ptr, off);
                    float d = sm * (s - v_mean) + sv;
                    if (with_src_scales) d *= src_scales[0];
                    ref_post_ops_t::args_t args;
                    args.ctx = &ctx;
                    args.l_offset = N_start * C_padded + off;
                    args.dst_md = pd()->dst_md();
                    ref_post_ops->execute(d, args);
                    if (with_dst_scales) d /= dst_scales[0];
                    io::store_float_value(dst_dt, d, dst_ptr, off);
                }
            } else if (use_scale) {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < C; ++c) {
                    const float sm = scale[c] * inv_sqrtvar;
                    const size_t off = c + C * offset;
                    float s = io::load_float_value(src_dt, src_ptr, off);
                    float d = sm * (s - v_mean);
                    if (with_src_scales) d *= src_scales[0];
                    ref_post_ops_t::args_t args;
                    args.ctx = &ctx;
                    args.l_offset = N_start * C_padded + off;
                    args.dst_md = pd()->dst_md();
                    ref_post_ops->execute(d, args);
                    if (with_dst_scales) d /= dst_scales[0];
                    io::store_float_value(dst_dt, d, dst_ptr, off);
                }
            } else if (use_shift) {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < C; ++c) {
                    const float sm = 1.f * inv_sqrtvar;
                    const float sv = shift[c];
                    const size_t off = c + C * offset;
                    float s = io::load_float_value(src_dt, src_ptr, off);
                    float d = sm * (s - v_mean) + sv;
                    if (with_src_scales) d *= src_scales[0];
                    ref_post_ops_t::args_t args;
                    args.ctx = &ctx;
                    args.l_offset = N_start * C_padded + off;
                    args.dst_md = pd()->dst_md();
                    ref_post_ops->execute(d, args);
                    if (with_dst_scales) d /= dst_scales[0];
                    io::store_float_value(dst_dt, d, dst_ptr, off);
                }
            } else {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < C; ++c) {
                    const float sm = 1.f * inv_sqrtvar;
                    const size_t off = c + C * offset;
                    float s = io::load_float_value(src_dt, src_ptr, off);
                    float d = sm * (s - v_mean);
                    if (with_src_scales) d *= src_scales[0];
                    ref_post_ops_t::args_t args;
                    args.ctx = &ctx;
                    args.l_offset = N_start * C_padded + off;
                    args.dst_md = pd()->dst_md();
                    ref_post_ops->execute(d, args);
                    if (with_dst_scales) d /= dst_scales[0];
                    io::store_float_value(dst_dt, d, dst_ptr, off);
                }
            }
            if (calculate_stats && save_stats) {
                if (!skip_mean) { mean_ptr[offset] = v_mean; }
                var_ptr[offset] = v_variance;
            }
        }
    });
    return status::success;
}

status_t simple_layer_normalization_bwd_t::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    const memory_desc_wrapper src_d(src_md());

    VDISPATCH_LNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_LNORM(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
    VDISPATCH_LNORM(utils::one_of(src_md()->data_type, f32, bf16, f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(utils::one_of(diff_dst_md()->data_type, f32, bf16, f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(utils::one_of(diff_src_md()->data_type, f32, bf16, f16),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM_SC(platform::data_type_support_status(src_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM_SC(
            platform::data_type_support_status(diff_dst_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM_SC(
            platform::data_type_support_status(diff_src_md()->data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_LNORM(check_scale_shift_data_type(), VERBOSE_UNSUPPORTED_FEATURE,
            "unsupported scale or shift data type");
    VDISPATCH_LNORM(attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_LNORM(set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_LNORM(src_d.is_blocking_desc(), VERBOSE_BLOCKING_FAIL,
            "blocking descriptor fail");
    // plain format, last logical dim is last physical
    VDISPATCH_LNORM(src_d.blocking_desc().strides[ndims() - 1] == 1,
            VERBOSE_BLOCKING_FAIL, "bad stride value");
    VDISPATCH_LNORM(impl::is_dense_format_kind(
                            {src_md(), diff_src_md(), dst_md(), diff_dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    CHECK(fill_compatible_stats_md(*src_md(), reordered_stat_md_));

    if (reordered_stat_md_ != *stat_md()) {
        CHECK(reorder_primitive_desc_create(
                reorder_pd_, engine, stat_md(), &reordered_stat_md_));
    }

    nthr_ = dnnl_get_max_threads();
    init_scratchpad();
    return status::success;
}

status_t simple_layer_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    status_t status = status::success;

    const bool use_scale = pd()->use_scale();

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const void *, DNNL_ARG_DIFF_DST);
    auto scale = CTX_IN_MEM(float *, DNNL_ARG_SCALE);
    auto diff_src = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DIFF_SRC, status);

    auto diff_scale = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SCALE, status);
    CHECK(status);
    auto diff_shift = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DIFF_SHIFT, status);
    CHECK(status);

    const bool skip_mean = pd()->skip_mean();

    const float *mean, *variance;
    if (pd()->use_tmp_stats()) {
        mean = skip_mean ? nullptr
                         : scratchpad.template get<float>(key_lnorm_tmp_mean);
        variance = scratchpad.template get<float>(key_lnorm_tmp_var);
    } else {
        mean = CTX_IN_MEM(const float *, DNNL_ARG_MEAN);
        variance = CTX_IN_MEM(const float *, DNNL_ARG_VARIANCE);
    }

    float *const inv_sqrtvar
            = scratchpad.template get<float>(key_lnorm_inv_sqrtvar);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());

    const dim_t N = pd()->across_axis();
    const dim_t C = pd()->norm_axis();
    const float C_f = static_cast<float>(C);
    const dim_t C_padded = src_d.padded_dims()[pd()->ndims() - 1];

    float *reduce = scratchpad.template get<float>(key_lnorm_reduction);
    if (diff_scale == nullptr)
        diff_scale = scratchpad.template get<float>(key_lnorm_tmp_diff_ss);
    if (diff_shift == nullptr) {
        diff_shift = scratchpad.template get<float>(key_lnorm_tmp_diff_ss);
    }

    const int max_nthr = pd()->nthr_;

    const auto src_dt = pd()->src_md()->data_type;
    const auto diff_dst_dt = pd()->diff_dst_md()->data_type;
    const auto diff_src_dt = pd()->diff_src_md()->data_type;
    const auto eps = pd()->desc()->layer_norm_epsilon;
    const auto calculate_diff_stats = !pd()->stats_are_src();

    parallel(max_nthr, [=](int ithr, int nthr) {
        dim_t N_start = 0, N_end = 0;
        balance211(N, nthr, ithr, N_start, N_end);
        const size_t block_size = N_end - N_start;
        const char *const __restrict src_ptr
                = reinterpret_cast<const char *>(src)
                + N_start * C_padded * src_d.data_type_size();
        const char *const __restrict diff_dst_ptr
                = reinterpret_cast<const char *>(diff_dst)
                + N_start * C_padded * diff_dst_d.data_type_size();
        const float *mean_ptr = skip_mean ? nullptr : &mean[N_start];
        const float *var_ptr = &variance[N_start];
        float *const inv_sqrtvar_ptr = &inv_sqrtvar[N_start];

        float *my_diff_gamma = reduce + C * ithr;
        float *my_diff_beta = reduce + C * nthr + C * ithr;

        PRAGMA_OMP_SIMD()
        for (dim_t c = 0; c < C; c++) {
            my_diff_gamma[c] = 0.f;
            my_diff_beta[c] = 0.f;
        }

        for (size_t offset = 0; offset < block_size; offset++) {
            inv_sqrtvar_ptr[offset] = 1.f / sqrtf(var_ptr[offset] + eps);
            float mean_val = skip_mean ? 0.f : mean_ptr[offset];

            PRAGMA_OMP_SIMD()
            for (dim_t c = 0; c < C; c++) {
                const size_t off = c + C * offset;
                float s = io::load_float_value(src_dt, src_ptr, off);
                float dd = io::load_float_value(diff_dst_dt, diff_dst_ptr, off);
                my_diff_gamma[c]
                        += (s - mean_val) * dd * inv_sqrtvar_ptr[offset];
                my_diff_beta[c] += dd;
            }
        }
    });

    parallel_nd(C, [=](dim_t c) {
        float diff_gamma = 0, diff_beta = 0;
        for (dim_t n = 0; n < max_nthr; n++) {
            diff_gamma += reduce[C * n + c];
            diff_beta += reduce[C * max_nthr + C * n + c];
        }
        diff_scale[c] = diff_gamma;
        diff_shift[c] = diff_beta;
    });

    parallel(max_nthr, [=](int ithr, int nthr) {
        dim_t N_start = 0, N_end = 0;
        balance211(N, nthr, ithr, N_start, N_end);
        const size_t block_size = N_end - N_start;
        const char *const __restrict src_ptr
                = reinterpret_cast<const char *>(src)
                + N_start * C_padded * src_d.data_type_size();
        const char *const __restrict diff_dst_ptr
                = reinterpret_cast<const char *>(diff_dst)
                + N_start * C_padded * diff_dst_d.data_type_size();
        char *const __restrict diff_src_ptr = reinterpret_cast<char *>(diff_src)
                + N_start * C_padded * diff_src_d.data_type_size();
        const float *mean_ptr = skip_mean ? nullptr : &mean[N_start];
        float *const inv_sqrtvar_ptr = &inv_sqrtvar[N_start];

        // Note: manual unrolling for scale and shift due to clang issue.
        //       see: CLANG_WA_01_SAFE_TO_USE_OMP_SIMD
        float dd_gamma, dd_gamma_x, mean_val;
        for (size_t offset = 0; offset < block_size; offset++) {
            // reduce gamma
            dd_gamma = dd_gamma_x = 0;
            mean_val = skip_mean ? 0.f : mean_ptr[offset];
            if (calculate_diff_stats) {
                if (use_scale) {
                    PRAGMA_OMP_SIMD(reduction(+ : dd_gamma, dd_gamma_x))
                    for (dim_t c = 0; c < C; c++) {
                        const size_t off = c + C * offset;
                        float s = io::load_float_value(src_dt, src_ptr, off);
                        float dd = io::load_float_value(
                                diff_dst_dt, diff_dst_ptr, off);
                        dd_gamma += dd * scale[c];
                        dd_gamma_x += dd * scale[c] * (s - mean_val);
                    }
                } else {
                    PRAGMA_OMP_SIMD(reduction(+ : dd_gamma, dd_gamma_x))
                    for (dim_t c = 0; c < C; c++) {
                        const size_t off = c + C * offset;
                        float s = io::load_float_value(src_dt, src_ptr, off);
                        float dd = io::load_float_value(
                                diff_dst_dt, diff_dst_ptr, off);
                        dd_gamma += dd;
                        dd_gamma_x += dd * (s - mean_val);
                    }
                }
                dd_gamma_x *= inv_sqrtvar_ptr[offset];
            }

            // calculate diff_dst
            if (use_scale) {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < C; c++) {
                    const size_t off = c + C * offset;
                    float dd = io::load_float_value(
                            diff_dst_dt, diff_dst_ptr, off);
                    float ds = dd * scale[c];
                    if (calculate_diff_stats) {
                        float s = io::load_float_value(src_dt, src_ptr, off);
                        ds -= dd_gamma / C_f;
                        ds -= (s - mean_val) * dd_gamma_x
                                * inv_sqrtvar_ptr[offset] / C_f;
                    }
                    ds *= inv_sqrtvar_ptr[offset];
                    io::store_float_value(diff_src_dt, ds, diff_src_ptr, off);
                }
            } else {
                PRAGMA_OMP_SIMD()
                for (dim_t c = 0; c < C; c++) {
                    const size_t off = c + C * offset;
                    float dd = io::load_float_value(
                            diff_dst_dt, diff_dst_ptr, off);
                    float ds = dd;
                    if (calculate_diff_stats) {
                        float s = io::load_float_value(src_dt, src_ptr, off);
                        ds -= dd_gamma / C_f;
                        ds -= (s - mean_val) * dd_gamma_x
                                * inv_sqrtvar_ptr[offset] / C_f;
                    }
                    ds *= inv_sqrtvar_ptr[offset];
                    io::store_float_value(diff_src_dt, ds, diff_src_ptr, off);
                }
            }
        }
    });
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
