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

#include <cmath>

#include "cpu/platform.hpp"
#include "cpu/primitive_attr_postops.hpp"
#include "cpu/ref_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

using namespace alg_kind;
using namespace math;

float compute_binary_scalar(alg_kind_t alg, float x, float y, bool c) {
    switch (alg) {
        case binary_add: return x + y;
        case binary_div: return x / y;
        case binary_max: return nstl::max(x, y);
        case binary_min: return nstl::min(x, y);
        case binary_mul: return x * y;
        case binary_sub: return x - y;
        case binary_ge: return x >= y;
        case binary_gt: return x > y;
        case binary_le: return x <= y;
        case binary_lt: return x < y;
        case binary_eq: return x == y;
        case binary_ne: return x != y;
        case binary_select: return c ? x : y;
        case binary_mul_inplace: return x * y;
        default: assert(!"unsupported operation!"); return NAN;
    }
}

float compute_eltwise_scalar_fwd(
        const alg_kind_t alg, float s, float alpha, float beta) {
    float d = 0.f;
    switch (alg) {
        case eltwise_relu: d = relu_fwd(s, alpha); break;
        case eltwise_tanh: d = tanh_fwd(s); break;
        case eltwise_elu: d = elu_fwd(s, alpha); break;
        case eltwise_square: d = square_fwd(s); break;
        case eltwise_abs: d = abs_fwd(s); break;
        case eltwise_sqrt: d = sqrt_fwd(s); break;
        case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
        case eltwise_soft_relu: d = soft_relu_fwd(s, alpha); break;
        case eltwise_logistic: d = logistic_fwd(s); break;
        case eltwise_exp: d = exp_fwd(s); break;
        case eltwise_gelu_tanh: d = gelu_tanh_fwd(s); break;
        case eltwise_swish: d = swish_fwd(s, alpha); break;
        case eltwise_log: d = log_fwd(s); break;
        case eltwise_clip: d = clip_fwd(s, alpha, beta); break;
        case eltwise_clip_v2: d = clip_v2_fwd(s, alpha, beta); break;
        case eltwise_pow: d = pow_fwd(s, alpha, beta); break;
        case eltwise_gelu_erf: d = gelu_erf_fwd(s); break;
        case eltwise_round: d = round_fwd(s); break;
        case eltwise_mish: d = mish_fwd(s); break;
        case eltwise_hardsigmoid: d = hardsigmoid_fwd(s, alpha, beta); break;
        case eltwise_hardswish: d = hardswish_fwd(s, alpha, beta); break;
        case eltwise_relu_use_dst_for_bwd: d = relu_fwd(s, alpha); break;
        case eltwise_tanh_use_dst_for_bwd: d = tanh_fwd(s); break;
        case eltwise_elu_use_dst_for_bwd: d = elu_fwd(s, alpha); break;
        case eltwise_sqrt_use_dst_for_bwd: d = sqrt_fwd(s); break;
        case eltwise_logistic_use_dst_for_bwd: d = logistic_fwd(s); break;
        case eltwise_exp_use_dst_for_bwd: d = exp_fwd(s); break;
        case eltwise_clip_v2_use_dst_for_bwd:
            d = clip_v2_fwd(s, alpha, beta);
            break;

        default: assert(!"unknown eltwise alg_kind");
    }
    return d;
}

float compute_eltwise_scalar_bwd(
        const alg_kind_t alg, float dd, float s, float alpha, float beta) {
    float ds = 0.f;
    switch (alg) {
        case eltwise_relu: ds = relu_bwd(dd, s, alpha); break;
        case eltwise_tanh: ds = tanh_bwd(dd, s); break;
        case eltwise_elu: ds = elu_bwd(dd, s, alpha); break;
        case eltwise_square: ds = square_bwd(dd, s); break;
        case eltwise_abs: ds = abs_bwd(dd, s); break;
        case eltwise_sqrt: ds = sqrt_bwd(dd, s); break;
        case eltwise_linear: ds = linear_bwd(dd, s, alpha, beta); break;
        case eltwise_soft_relu: ds = soft_relu_bwd(dd, s, alpha); break;
        case eltwise_logistic: ds = logistic_bwd(dd, s); break;
        case eltwise_exp: ds = exp_bwd(dd, s); break;
        case eltwise_gelu_tanh: ds = gelu_tanh_bwd(dd, s); break;
        case eltwise_swish: ds = swish_bwd(dd, s, alpha); break;
        case eltwise_log: ds = log_bwd(dd, s); break;
        case eltwise_clip: ds = clip_bwd(dd, s, alpha, beta); break;
        case eltwise_clip_v2: ds = clip_v2_bwd(dd, s, alpha, beta); break;
        case eltwise_pow: ds = pow_bwd(dd, s, alpha, beta); break;
        case eltwise_gelu_erf: ds = gelu_erf_bwd(dd, s); break;
        case eltwise_mish: ds = mish_bwd(dd, s); break;
        case eltwise_hardsigmoid:
            ds = hardsigmoid_bwd(dd, s, alpha, beta);
            break;
        case eltwise_hardswish: ds = hardswish_bwd(dd, s, alpha, beta); break;
        case eltwise_relu_use_dst_for_bwd:
            ds = relu_bwd_use_dst(dd, s, alpha);
            break;
        case eltwise_tanh_use_dst_for_bwd: ds = tanh_bwd_use_dst(dd, s); break;
        case eltwise_elu_use_dst_for_bwd:
            ds = elu_bwd_use_dst(dd, s, alpha);
            break;
        case eltwise_sqrt_use_dst_for_bwd: ds = sqrt_bwd_use_dst(dd, s); break;
        case eltwise_logistic_use_dst_for_bwd:
            ds = logistic_bwd_use_dst(dd, s);
            break;
        case eltwise_exp_use_dst_for_bwd: ds = exp_bwd_use_dst(dd, s); break;
        case eltwise_clip_v2_use_dst_for_bwd:
            ds = clip_v2_bwd_use_dst(dd, s, alpha, beta);
            break;

        default: assert(!"unknown eltwise alg_kind");
    }
    return ds;
}

ref_binary_scalar_t::ref_binary_scalar_t(alg_kind_t alg) : alg_(alg) {
    assert(utils::one_of(alg_, alg_kind::binary_add, alg_kind::binary_max,
            alg_kind::binary_min, alg_kind::binary_mul, alg_kind::binary_div,
            alg_kind::binary_sub, alg_kind::binary_ge, alg_kind::binary_gt,
            alg_kind::binary_le, alg_kind::binary_lt, alg_kind::binary_eq,
            alg_kind::binary_ne, alg_kind::binary_select,
            alg_kind::binary_mul_inplace));
}

ref_binary_scalar_t::ref_binary_scalar_t(
        const post_ops_t::entry_t::binary_t &binary)
    : ref_binary_scalar_t(binary.alg) {}

float ref_binary_scalar_t::compute_scalar(
        float src0, float src1, bool src2) const {
    return compute_binary_scalar(alg_, src0, src1, src2);
}

bool ref_binary_scalar_t::data_type_ok(const post_ops_t::entry_t &entry) {
    const auto &binary = entry.binary;
    bool src1_ok = platform::has_data_type_support(binary.src1_desc.data_type);

    if (entry.is_binary_with_ternary_op()) {
        bool src2_ok
                = platform::has_data_type_support(binary.src2_desc.data_type);
        return src1_ok && src2_ok;
    }

    return src1_ok;
}

ref_eltwise_scalar_fwd_t::ref_eltwise_scalar_fwd_t(
        alg_kind_t alg, float alpha, float beta, float scale)
    : alg_(alg), alpha_(alpha), beta_(beta), scale_(scale) {
    assert(utils::one_of(alg_, eltwise_relu, eltwise_tanh, eltwise_elu,
            eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
            eltwise_soft_relu, eltwise_mish, eltwise_logistic, eltwise_exp,
            eltwise_gelu_tanh, eltwise_swish, eltwise_log, eltwise_clip,
            eltwise_clip_v2, eltwise_pow, eltwise_gelu_erf, eltwise_round,
            eltwise_hardsigmoid, eltwise_hardswish,
            eltwise_relu_use_dst_for_bwd, eltwise_tanh_use_dst_for_bwd,
            eltwise_elu_use_dst_for_bwd, eltwise_sqrt_use_dst_for_bwd,
            eltwise_logistic_use_dst_for_bwd, eltwise_exp_use_dst_for_bwd,
            eltwise_clip_v2_use_dst_for_bwd));
}

ref_eltwise_scalar_fwd_t::ref_eltwise_scalar_fwd_t(
        const post_ops_t::entry_t::eltwise_t &eltwise)
    : ref_eltwise_scalar_fwd_t(
              eltwise.alg, eltwise.alpha, eltwise.beta, eltwise.scale) {}

float ref_eltwise_scalar_fwd_t::compute_scalar(float s) const {
    return compute_eltwise_scalar_fwd(alg_, s, alpha_, beta_) * scale_;
}

ref_sum_scalar_t::ref_sum_scalar_t(bool skip_sum) : skip_sum_(skip_sum) {}

void ref_sum_scalar_t::execute(
        float &res, float dst_val, float scale, int32_t zero_point) const {
    if (skip_sum_) return;

    res += scale * (dst_val - static_cast<float>(zero_point));
}

bool ref_sum_scalar_t::data_type_ok(const post_ops_t::entry_t &entry) {
    const auto &sum = entry.sum;
    return sum.dt == data_type::undef
            || platform::has_data_type_support(sum.dt);
}

ref_post_ops_t::ref_post_ops_t(const post_ops_t &po, bool skip_sum)
    : po_(po), sum_po_(skip_sum) {
    for (auto idx = 0; idx < po_.len(); ++idx) {
        const auto &e = po_.entry_[idx];
        if (po_.contain(primitive_kind::eltwise, idx)) {
            eltwise_po_.emplace_back(e.eltwise);
        } else if (po_.contain(primitive_kind::binary, idx)) {
            binary_po_.emplace_back(e.binary);
        }
    }
}

namespace {

format_tag_t get_prelu_weights_format(const dim_t n_dims) {
    switch (n_dims) {
        case 1: return format_tag::a;
        case 2: return format_tag::ab;
        case 3: return format_tag::acb;
        case 4: return format_tag::acdb;
        case 5: return format_tag::acdeb;
    }

    return format_tag::undef;
}

status_t get_prelu_memory_desc(memory_desc_t &weights_md,
        const dims_t &dst_dims, const int dst_ndims, int weights_mask) {

    weights_md.data_type = data_type::f32;
    weights_md.ndims = dst_ndims;
    utils::copy_dims_with_mask(
            weights_md.dims, dst_dims, dst_ndims, weights_mask);
    CHECK(memory_desc_init_by_tag(
            weights_md, get_prelu_weights_format(dst_ndims)));

    return status::success;
}

void get_l_dims_po(dims_t &l_dims_po, const dim_t l_offset,
        const dims_t &dst_dims, const int dst_ndims, int mask) {
    utils::l_dims_by_l_offset(l_dims_po, l_offset, dst_dims, dst_ndims);
    utils::apply_mask_on_dims(l_dims_po, dst_ndims, mask);
}

dim_t get_po_tensor_off(const memory_desc_t &tensor_md, const dim_t l_offset,
        const dims_t &dst_dims, const int dst_ndims, int mask) {

    dims_t l_dims_po {};
    get_l_dims_po(l_dims_po, l_offset, dst_dims, dst_ndims, mask);

    return memory_desc_wrapper(tensor_md).off_v(l_dims_po);
}

dim_t get_prelu_weights_off(const memory_desc_t &weights_md,
        const dim_t l_offset, const dims_t &dst_dims, const int dst_ndims,
        int weights_mask) {

    return get_po_tensor_off(
            weights_md, l_offset, dst_dims, dst_ndims, weights_mask);
}

// Note: src_md is either src1_md or src2_md
dim_t get_binary_src_off(const memory_desc_t &src_md, const dim_t l_offset,
        const dims_t &dst_dims, const int dst_ndims) {

    const int mask_binary_po
            = utils::get_dims_mask(dst_dims, src_md.dims, dst_ndims);

    return get_po_tensor_off(
            src_md, l_offset, dst_dims, dst_ndims, mask_binary_po);
}

} // namespace

status_t ref_post_ops_t::init(const memory_desc_t *dst_md) {
    if (!dst_md) return status::invalid_arguments;

    for (auto idx = 0; idx < po_.len(); ++idx) {
        const auto &e = po_.entry_[idx];

        if (e.is_prelu()) {
            memory_desc_t weights_md;
            CHECK(get_prelu_memory_desc(
                    weights_md, dst_md->dims, dst_md->ndims, e.prelu.mask));
            prelu_md_.emplace_back(weights_md);
        }
    }
    return status::success;
}

float ref_dropout(float src, uint8_t *mask, dim_t idx, float p, int64_t seed,
        int64_t offset) {
    // Note: as this is a reference implementation, it's not intended to be
    // efficient. For optimized versions, `1.f / (1.f - p)` should be passed as
    // a single value computed once to avoid division for every element.
    //
    // Note: for `offset = 0` keep the legacy logic without the `offset`.
    float inv_q = (p != 1.f) ? 1.f / (1.f - p) : 0.f;
    uint32_t r = offset ? philox4x32(idx, seed, offset)
                        : philox4x32(uint32_t(idx), uint32_t(seed));
    p = std::max(std::min(p, 1.f), 0.f);
    uint8_t m = (r > double(std::numeric_limits<uint32_t>::max()) * p);
    if (mask) mask[idx] = m;
    return (m) ? src * inv_q : 0;
}

void ref_post_ops_t::execute(float &res, const args_t &args) const {
    if (po_.len() == 0) return;

    auto it_eltwise_po = eltwise_po_.begin();
    auto it_binary_po = binary_po_.begin();
    auto it_prelu_md = prelu_md_.begin();
    for (auto idx = 0; idx < po_.len(); ++idx) {
        const auto &e = po_.entry_[idx];
        switch (e.kind) {
            case primitive_kind::sum:
                sum_po_.execute(
                        res, args.dst_val, e.sum.scale, e.sum.zero_point);
                break;
            case primitive_kind::eltwise:
                res = it_eltwise_po->compute_scalar(res);
                it_eltwise_po++;
                break;
            case primitive_kind::binary: {
                assert(args.ctx);
                assert(args.l_offset >= 0);
                assert(args.dst_md);

                const exec_ctx_t &ctx = *args.ctx;
                const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, args.dst_md);
                const auto &src1_desc = e.binary.src1_desc;

                // An in-place binary post-op aliases the destination memory,
                // so its right-hand side value is the current destination value
                // at the same logical offset, already loaded into `dst_val`.
                float val_po = args.dst_val;
                if (!e.is_inplace_binary()) {
                    const auto off = get_binary_src_off(src1_desc,
                            args.l_offset, dst_d.dims(), dst_d.ndims());
                    const auto src1_binary_po = CTX_IN_MEM(const void *,
                            (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                    | DNNL_ARG_SRC_1));
                    val_po = io::load_float_value(
                            src1_desc.data_type, src1_binary_po, off);
                }

                bool src2_val = false;
                if (e.is_binary_with_ternary_op()
                        && e.binary.alg
                                == dnnl::impl::alg_kind::binary_select) {
                    const auto src2_binary_po = CTX_IN_MEM(const void *,
                            (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                    | DNNL_ARG_SRC_2));
                    const auto &src2_desc = e.binary.src2_desc;
                    const auto src2_off = get_binary_src_off(src2_desc,
                            args.l_offset, dst_d.dims(), dst_d.ndims());
                    src2_val = io::load_int_value(src2_desc.data_type,
                                       src2_binary_po, src2_off)
                            > 0;
                }

                res = it_binary_po->compute_scalar(res, val_po, src2_val);
                ++it_binary_po;
            } break;
            case primitive_kind::prelu: {
                if (res >= 0) break;

                assert(args.ctx);
                assert(args.l_offset >= 0);
                assert(args.dst_md);

                const exec_ctx_t &ctx = *args.ctx;
                const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, args.dst_md);
                auto prelu_weights_md = *it_prelu_md;

                // Handle for runtime dimensions.
                const bool has_runtime_dims
                        = memory_desc_wrapper(args.dst_md).has_runtime_dims();
                if (has_runtime_dims)
                    get_prelu_memory_desc(prelu_weights_md, dst_d.dims(),
                            dst_d.ndims(), e.prelu.mask);

                const auto prelu_weights = CTX_IN_MEM(const float *,
                        (DNNL_ARG_ATTR_MULTIPLE_POST_OP(idx)
                                | DNNL_ARG_WEIGHTS));
                const auto off
                        = get_prelu_weights_off(prelu_weights_md, args.l_offset,
                                dst_d.dims(), dst_d.ndims(), e.prelu.mask);
                const auto &weights_value = prelu_weights[off];
                res = weights_value * res;
                ++it_prelu_md;
            } break;
            default: assert(!"unsupported post op primitive kind!");
        }
    }
}

bool ref_post_ops_t::post_ops_ok(const post_ops_t &po) {

    if (!primitive_kind_ok(po)) return false;

    for (const auto &e : po.entry_) {
        switch (e.kind) {
            case primitive_kind::sum:
                if (!ref_sum_scalar_t::data_type_ok(e)) return false;
                break;
            case primitive_kind::binary:
                if (!ref_binary_scalar_t::data_type_ok(e)) return false;
                break;
            case primitive_kind::eltwise:
            case primitive_kind::prelu: break;
            default: return false;
        }
    }
    return true;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
