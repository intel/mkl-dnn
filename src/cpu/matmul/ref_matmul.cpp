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

#include <algorithm>
#include <assert.h>
#include <float.h>
#include <math.h>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"

#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/matmul/ref_matmul.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace matmul {

void ref_matmul_t::pd_t::init_scratchpad() {
    using namespace memory_tracking::names;
    nthr_ = dnnl_get_max_threads();
    ntasks_ = nthr_;
    auto dst_scales = attr()->scales_.get(DNNL_ARG_DST);
    if (dst_scales.is_dynamic()) {
        auto scratchpad = scratchpad_registry().registrar();
        const memory_desc_wrapper dst_d(dst_md());
        const dim_t group_size = dst_scales.get_group_size();
        const dim_t work_amount = dst_d.nelems() / group_size;
        ntasks_ = static_cast<int>(nstl::min<dim_t>(nthr_, work_amount));
        scratchpad.template book<float>(
                key_matmul_dst_in_acc_dt, ntasks_ * group_size);
    }
}

status_t ref_matmul_t::execute_ref(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    const auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);

    const auto dropout_p
            = CTX_IN_MEM(const float *, DNNL_ARG_ATTR_DROPOUT_PROBABILITY);
    const auto dropout_seed
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_DROPOUT_SEED);
    const auto dropout_offset
            = CTX_IN_MEM(const int64_t *, DNNL_ARG_ATTR_DROPOUT_OFFSET);
    auto dropout_mask = CTX_OUT_CLEAN_MEM(
            unsigned char *, DNNL_ARG_ATTR_DROPOUT_MASK, status);
    CHECK(status);

    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
    // No need to zero-pad dynamic scales as they should have plain format.
    auto dst_dynamic_scales
            = CTX_OUT_MEM(float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const void *wei_zero_points = CTX_IN_MEM(
            const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);

    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    if (src_d.has_zero_dim() || weights_d.has_zero_dim()
            || dst_d.has_zero_dim())
        return status::success;

    const bool non_default_attrs = !pd()->attr()->has_default_values();

    matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const int batch_ndims = ndims - 2;
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();

    // Weights decompression
    // TODO: consolidate the weights-decompression detection into a pd() helper
    // shared across the library instead of recomputing it per implementation.
    const bool wei_is_int
            = utils::one_of(weights_d.data_type(), data_type::s8, data_type::u8,
                    data_type::s4, data_type::u4, data_type::u2, data_type::u3);
    // Low-precision floating-point weights (fp8/fp4) are decompressed whenever
    // weights scales or zero-points must be folded into the weights before the
    // multiply, e.g. the static_sbzp mode.
    const bool wei_is_fp8_fp4 = utils::one_of(weights_d.data_type(),
            data_type::f8_e5m2, data_type::f8_e4m3, data_type::f4_e2m1);
    const bool with_wei_quant_entries
            = !pd()->attr()->scales_.has_default_values(DNNL_ARG_WEIGHTS)
            || !pd()->attr()->zero_points_.has_default_values(DNNL_ARG_WEIGHTS);
    const bool with_wei_decompression
            = (wei_is_int && pd()->attr()->fpmath_.apply_to_int_)
            || (wei_is_fp8_fp4 && with_wei_quant_entries);
    const auto &attr_zps = pd()->attr()->zero_points_;
    const bool with_wei_zero_points
            = !attr_zps.has_default_values(DNNL_ARG_WEIGHTS);
    int wei_zp_mask = attr_zps.get_mask(DNNL_ARG_WEIGHTS);
    const auto &wei_zp_dt = attr_zps.get_data_type(DNNL_ARG_WEIGHTS);
    const auto wei_zp_group_k = attr_zps.get_group(DNNL_ARG_WEIGHTS, -2);
    const auto wei_zp_group_n = attr_zps.get_group(DNNL_ARG_WEIGHTS, -1);
    // Initialize a memory desc for quant entries for easier offset calculation.
    memory_desc_t wei_zp_md {};
    CHECK(attr_zps.get(DNNL_ARG_WEIGHTS).get_md(wei_zp_md, *weights_d.md_));

    const int src_mask
            = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
    const int wei_mask
            = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
    const int bia_mask
            = utils::get_dims_mask(dst_d.dims(), bia_d.dims(), ndims);

    // Scales section
    const auto &attr_scales = pd()->attr()->scales_;

    const bool with_src_scales = !attr_scales.has_default_values(DNNL_ARG_SRC);
    const auto src_scale_mask = attr_scales.get_mask(DNNL_ARG_SRC);
    const auto src_scale_dt = attr_scales.get_data_type(DNNL_ARG_SRC);
    const auto src_scale_group_m = attr_scales.get_group(DNNL_ARG_SRC, -2);
    const auto src_scale_group_k = attr_scales.get_group(DNNL_ARG_SRC, -1);
    const auto src_scale_ngroups_k = K / src_scale_group_k;
    // Initialize a memory desc for quant entries for easier offset calculation.
    memory_desc_t src_scale_md {};
    CHECK(attr_scales.get(DNNL_ARG_SRC).get_md(src_scale_md, *src_d.md_));

    const bool with_wei_scales
            = !attr_scales.has_default_values(DNNL_ARG_WEIGHTS);
    const auto wei_scale_mask = attr_scales.get_mask(DNNL_ARG_WEIGHTS);
    const auto wei_scale_dt = attr_scales.get_data_type(DNNL_ARG_WEIGHTS);
    const auto wei_scale_group_k = attr_scales.get_group(DNNL_ARG_WEIGHTS, -2);
    const auto wei_scale_ngroups_k = K / wei_scale_group_k;
    const auto wei_scale_group_n = attr_scales.get_group(DNNL_ARG_WEIGHTS, -1);
    // Initialize a memory desc for quant entries for easier offset calculation.
    memory_desc_t wei_scale_md {};
    CHECK(attr_scales.get(DNNL_ARG_WEIGHTS)
                    .get_md(wei_scale_md, *weights_d.md_));
    // Scales applied before zero-points (both entries use the same mode).
    const bool with_wei_sbzp
            = with_wei_scales && attr_scales.get(DNNL_ARG_WEIGHTS).is_sbzp();

    const bool with_dst_scales = !attr_scales.has_default_values(DNNL_ARG_DST);
    const auto dst_scale_dt = attr_scales.get_data_type(DNNL_ARG_DST);
    const auto dst_scale_mask = attr_scales.get_mask(DNNL_ARG_DST);
    const auto dst_scale_group_m = attr_scales.get_group(DNNL_ARG_DST, -2);
    const auto dst_scale_group_n = attr_scales.get_group(DNNL_ARG_DST, -1);
    memory_desc_t dst_scales_md {};
    CHECK(attr_scales.get(DNNL_ARG_DST).get_md(dst_scales_md, *dst_d.md_));

    // For compute kernel, the minimal group is picked.
    const auto ngroups_k = std::max(src_scale_ngroups_k, wei_scale_ngroups_k);
    const auto group_k = K / ngroups_k;

    auto dst_rnd_mode = pd()->attr()->rounding_mode_.get(DNNL_ARG_DST);

    // mm kernel
    auto ker = [=](const dims_t dst_dims_idx, dim_t m, dim_t n) {
        dims_t src_dims_idx, weights_dims_idx;
        utils::copy_dims_with_mask(src_dims_idx, dst_dims_idx, ndims, src_mask);
        utils::copy_dims_with_mask(
                weights_dims_idx, dst_dims_idx, ndims, wei_mask);
        src_dims_idx[ndims - 2] = m;
        weights_dims_idx[ndims - 1] = n;
        auto &src_k_dim = src_dims_idx[ndims - 1];
        auto &wei_k_dim = weights_dims_idx[ndims - 2];
        float res = 0.0f;
        for (dim_t i_group = 0; i_group < ngroups_k; i_group++) {
            float acc = 0.0f;
            for (dim_t k = 0; k < group_k; ++k) {
                src_k_dim = k + i_group * group_k;
                wei_k_dim = k + i_group * group_k;

                const auto src_off = src_d.off_v(src_dims_idx);
                const auto weights_off = weights_d.off_v(weights_dims_idx);
                const float s
                        = io::load_float_value(src_d.data_type(), src, src_off);
                float w = io::load_float_value(
                        weights_d.data_type(), weights, weights_off);

                // weights decompression should happen before the operation
                if (with_wei_decompression) {
                    float wei_zp = 0.f;
                    if (with_wei_zero_points) {
                        const dim_t wei_zp_offset
                                = matmul_helper_t::get_quant_off(
                                        weights_dims_idx, ndims, wei_zp_mask,
                                        wei_zp_group_k, wei_zp_group_n,
                                        wei_zp_md);
                        wei_zp = io::load_float_value(
                                wei_zp_dt, wei_zero_points, wei_zp_offset);
                    }
                    float wei_scale = 1.f;
                    if (with_wei_scales) {
                        const dim_t wei_scale_offset
                                = matmul_helper_t::get_quant_off(
                                        weights_dims_idx, ndims, wei_scale_mask,
                                        wei_scale_group_k, wei_scale_group_n,
                                        wei_scale_md);
                        wei_scale = io::load_float_value(
                                wei_scale_dt, wei_scales, wei_scale_offset);
                    }
                    // static_sbzp applies the scale first, static_sazp the
                    // zero-point first.
                    w = with_wei_sbzp ? w * wei_scale - wei_zp
                                      : (w - wei_zp) * wei_scale;
                }
                acc += s * w;
            }
            // apply scales after computing a group along K
            if (with_src_scales) {
                const dim_t src_scale_offset = matmul_helper_t::get_quant_off(
                        src_dims_idx, ndims, src_scale_mask, src_scale_group_m,
                        src_scale_group_k, src_scale_md);
                float src_scale = io::load_float_value(
                        src_scale_dt, src_scales, src_scale_offset);
                acc *= src_scale;
            }
            if (with_wei_scales && !with_wei_decompression) {
                const dim_t wei_scale_offset = matmul_helper_t::get_quant_off(
                        weights_dims_idx, ndims, wei_scale_mask,
                        wei_scale_group_k, wei_scale_group_n, wei_scale_md);
                const float wei_scale = io::load_float_value(
                        wei_scale_dt, wei_scales, wei_scale_offset);
                acc *= wei_scale;
            }
            res += acc;
        }
        return res;
    };

    // bias section
    auto ker_bias = [=](const dims_t &dst_dims_idx) -> float {
        dims_t bia_dims_idx;
        utils::copy_dims_with_mask(bia_dims_idx, dst_dims_idx, ndims, bia_mask);
        const auto bias_off = bia_d.off_v(bia_dims_idx);
        return io::load_float_value(bia_d.data_type(), bias, bias_off);
    };

    auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());
    bool with_dropout = !pd()->attr()->dropout_.has_default_values();

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    float *temp_dst = scratchpad.template get<float>(
            memory_tracking::names::key_matmul_dst_in_acc_dt);

    // Note 1: If dst type is < 8 bits, we cannot split a byte during
    //         store or we get a race condition. To simplify logic, we limit
    //         parallelization on M and N by a factor of 2.
    // Note 2: dynamic quantization requires to compute a reduction
    //         along dst groups, so we split parallelization accordingly
    dim_t M_chunk_size = std::max<dim_t>(2, dst_scale_group_m);
    dim_t N_chunk_size = std::max<dim_t>(2, dst_scale_group_n);
    dim_t M_chunks = utils::div_up(M, M_chunk_size);
    dim_t N_chunks = utils::div_up(N, N_chunk_size);
    parallel_nd_ext(pd()->nthr_, batch, M_chunks, N_chunks,
            [= COMPAT_THIS_CAPTURE](
                    int ithr, int nthr, dim_t mb, dim_t mc, dim_t nc) {
        if (ithr >= pd()->ntasks_) return;

        int64_t dropout_seed_val = with_dropout
                ? io::load_int64_value(
                          pd()->attr()->dropout_.seed_dt_, dropout_seed, 0)
                : 0;
        float dropout_p_val = with_dropout ? dropout_p[0] : 0.0f;
        int64_t dropout_offset_val
                = with_dropout && pd()->attr()->dropout_.use_offset_
                ? dropout_offset[0]
                : 0;

        for_(dim_t m_ = mc * M_chunk_size;
                m_ < std::min<dim_t>((mc + 1) * M_chunk_size, M);
                m_ += dst_scale_group_m)
        for (dim_t n_ = nc * N_chunk_size;
                n_ < std::min<dim_t>((nc + 1) * N_chunk_size, N);
                n_ += dst_scale_group_n) {
            float max_dst_group = 0.0f;
            for_(dim_t m_gidx = 0; m_gidx < dst_scale_group_m; m_gidx++)
            for (dim_t n_gidx = 0; n_gidx < dst_scale_group_n; n_gidx++) {
                dim_t m = m_ + m_gidx;
                dim_t n = n_ + n_gidx;
                dims_t dst_dims_idx;
                const size_t offset = mb * M * N + m * N + n;
                utils::l_dims_by_l_offset(
                        dst_dims_idx, offset, dst_d.dims(), ndims);

                float d = ker(dst_dims_idx, m, n);
                if (bias) d += ker_bias(dst_dims_idx);

                const auto dst_off = dst_d.off_v(dst_dims_idx);
                if (non_default_attrs) {
                    if (with_dropout)
                        d = ref_dropout(d, dropout_mask, dst_off, dropout_p_val,
                                dropout_seed_val, dropout_offset_val);
                    ref_post_ops_t::args_t args;
                    args.dst_val = io::load_float_value(sum_dt, dst, dst_off);
                    args.ctx = &ctx;
                    args.l_offset = offset;
                    args.dst_md = pd()->dst_md();
                    ref_post_ops->execute(d, args);
                }
                if (attr_scales.get(DNNL_ARG_DST).is_dynamic()) {
                    max_dst_group = std::max(max_dst_group, ::fabsf(d));
                    auto temp_dst_off = (ithr * dst_scale_group_m + m_gidx)
                                    * dst_scale_group_n
                            + n_gidx;
                    io::store_float_value(
                            data_type::f32, d, temp_dst, temp_dst_off);
                } else {
                    if (with_dst_scales) {
                        const dim_t dst_scale_off
                                = matmul_helper_t::get_quant_off(dst_dims_idx,
                                        ndims, dst_scale_mask,
                                        dst_scale_group_m, dst_scale_group_n,
                                        dst_scales_md);
                        const float dst_scale = io::load_float_value(
                                dst_scale_dt, dst_scales, dst_scale_off);
                        d /= dst_scale;
                    }
                    if (dst_rnd_mode == rounding_mode::stochastic)
                        // NOTE: dst_off is truncated to 32 bits here. For
                        // tensors with > 2^32 elements, different offsets
                        // may alias to the same random seed --
                        // stochastic_round_fwd does not support a 64-bit
                        // index space.
                        d = math::stochastic_round_fwd(d,
                                static_cast<uint32_t>(dst_off),
                                static_cast<uint32_t>(dropout_seed_val),
                                dst_d.data_type());
                    io::store_float_value(dst_d.data_type(), d, dst, dst_off);
                    utils::dim_iterator(
                            dst_d.dims(), dst_dims_idx, batch_ndims);
                }
            }

            if (attr_scales.get(DNNL_ARG_DST).is_dynamic()) {
                float dst_group_scale = 1.0f;
                if (attr_scales.get(DNNL_ARG_DST).is_mx()) {
                    // MXSPEC does round_down_pow2(dst_d.data_type() /
                    // round_down_pow2(max_dst_group) so the rounding
                    // to a power of two happens before the division,
                    // and not after.
                    dst_group_scale
                            = types::round_to_dt(dst_scale_dt, max_dst_group)
                            / types::round_to_dt(dst_scale_dt,
                                    types::max_value<float>(dst_d.data_type()));
                    dst_group_scale
                            = types::round_to_dt(dst_scale_dt, dst_group_scale);
                }
                if (attr_scales.get(DNNL_ARG_DST).is_dynamic_fp()) {
                    // Clamp the scale value to epsilon(dst_scale_dt) from the
                    // down and to max(dst_scale_dt) from the up.
                    // This is done to deal with zero block and small values
                    // obtained after dividing a group max value.
                    dst_group_scale = max_dst_group
                            / types::max_value<float>(dst_d.data_type());
                    dst_group_scale = nstl::min(dst_group_scale,
                            types::max_value<float>(dst_scale_dt));
                    dst_group_scale = nstl::max(dst_group_scale,
                            types::epsilon_value<float>(dst_scale_dt));
                    dst_group_scale
                            = types::round_to_dt(dst_scale_dt, dst_group_scale);
                }

                dims_t dst_dims_idx;
                const size_t offset = mb * M * N + m_ * N + n_;
                utils::l_dims_by_l_offset(
                        dst_dims_idx, offset, dst_d.dims(), ndims);

                const dim_t dst_scale_off = matmul_helper_t::get_quant_off(
                        dst_dims_idx, ndims, dst_scale_mask, dst_scale_group_m,
                        dst_scale_group_n, dst_scales_md);

                io::store_float_value(dst_scale_dt, dst_group_scale,
                        dst_dynamic_scales, dst_scale_off);
                // Pre-invert the scale to apply it as a multiplier for the
                // group. Can't be zero.
                dst_group_scale = 1.f / dst_group_scale;
                for_(dim_t m_gidx = 0; m_gidx < dst_scale_group_m; m_gidx++)
                for (dim_t n_gidx = 0; n_gidx < dst_scale_group_n; n_gidx++) {
                    dim_t m = m_ + m_gidx;
                    dim_t n = n_ + n_gidx;
                    dims_t dst_dims_idx;
                    const size_t offset = mb * M * N + m * N + n;
                    utils::l_dims_by_l_offset(
                            dst_dims_idx, offset, dst_d.dims(), ndims);
                    const auto dst_off = dst_d.off_v(dst_dims_idx);

                    auto temp_dst_off = (ithr * dst_scale_group_m + m_gidx)
                                    * dst_scale_group_n
                            + n_gidx;
                    float d = io::load_float_value(
                            data_type::f32, temp_dst, temp_dst_off);
                    d *= dst_group_scale;

                    // A dynamically-computed scale is rounded down to the
                    // nearest representable power-of-two (e8m0). When the
                    // resulting (pre-inverted) scale is less than one, its
                    // inverse applied here is greater than one and may push
                    // the value beyond the representable range of the
                    // destination data type. Saturate to the max/min
                    // representable value instead of overflowing to NaN.
                    const float dst_max
                            = types::max_value<float>(dst_d.data_type());
                    d = nstl::min(nstl::max(d, -dst_max), dst_max);

                    if (dst_rnd_mode == rounding_mode::stochastic)
                        // NOTE: dst_off is truncated to 32 bits here. For
                        // tensors with > 2^32 elements, different offsets
                        // may alias to the same random seed --
                        // stochastic_round_fwd does not support a 64-bit
                        // index space.
                        d = math::stochastic_round_fwd(d,
                                static_cast<uint32_t>(dst_off),
                                static_cast<uint32_t>(dropout_seed_val),
                                dst_d.data_type());
                    io::store_float_value(dst_d.data_type(), d, dst, dst_off);
                    utils::dim_iterator(
                            dst_d.dims(), dst_dims_idx, batch_ndims);
                }
            }
        }
    });

    return status::success;
}

} // namespace matmul
} // namespace cpu
} // namespace impl
} // namespace dnnl
