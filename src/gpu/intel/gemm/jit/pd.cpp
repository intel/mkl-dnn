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

#include "gpu/intel/gemm/jit/pd.hpp"
#include "common/c_types_map.hpp"
#include "common/primitive_attr_quant.hpp"
#include "gpu/intel/gemm/exec_types.hpp"
#include "gpu/intel/gemm/jit/gen_kernel.hpp"
#include "gpu/intel/gemm/utils.hpp"
#include "gpu/intel/jit/eltwise_injector.hpp"
#include "gpu/intel/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {
namespace jit {

using namespace intel::jit;

namespace {

// Obtain dimension count for gemmstone (common scales give count 0).
int quant_entry_ndims(
        const quant_entry_t &entry, const memory_desc_t &qmd, int k_idx) {
    if (entry.has_default_values()) return -1;
    if (qmd.ndims < 2) return 0;

    // Count the number of nontrivial (dim > 1) dimensions present
    int count = 0;
    for (int i = qmd.ndims - 2; i < qmd.ndims; ++i) {
        if (qmd.dims[i] > 1) { count++; }
    }

    if (count == 0) return 0;

    // for gemmstone, 1D quantization implies a full column vector
    // (i.e. not on the K dimension). If quantization varies over K,
    // we have to send these as 2D
    if (k_idx >= 0 && count == 1 && qmd.dims[k_idx] > 1) return 2;

    // If M/N quantization requires broadcast it is unsupported
    // by post-ops and must be submitted as 2D.
    int gcount = 0;
    for (int i = 0; i < 2; ++i) {
        if (entry.get_group(i) > 1) { gcount++; }
    }

    if (gcount == 2) return 2;

    return count;
}
} // anonymous namespace

status_t pd_t::init_post_ops(const impl::engine_t *engine) {
    using namespace primitive_kind;
    using namespace alg_kind;
    using namespace data_type;

    const auto d = desc();

    // Examine post-ops and remember binary srcs.
    post_ops_ = attr()->post_ops_;
    binary_srcs_.reserve(post_ops_.len() + 4);

    bool ok = true;
    int prelu_count = 0;
    for (int i = 0; i < post_ops_.len(); i++) {
        const auto &e = post_ops_.entry_[i];
        switch (e.kind) {
            case binary:
                ok &= supported_binary_op(e.binary.alg)
                        && is_md_gemm_compatible_plain_format(
                                &e.binary.src1_desc, false, true);
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::binary, int(i)});
                non_scale_po_ = true;
                break;
            case sum:
                ok &= !with_sum_;
                with_sum_ = true;
                sum_at_begin_ = (i == 0);
                binary_srcs_.push_back(binary_src_t {binary_src_t::none, 0});
                beta_ = e.sum.scale;
                break;
            case eltwise:
                ok &= eltwise_injector_f32_is_supported(e.eltwise.alg);
                binary_srcs_.push_back(binary_src_t {binary_src_t::none, 0});
                non_scale_po_ = true;
                break;
            case prelu:
                binary_srcs_.push_back(
                        binary_src_t {binary_src_t::prelu, int(i)});
                ok &= get_prelu_md(e.prelu.mask, dst_md()->dims, prelu_wei_md,
                              dst_md()->ndims)
                        == status::success;
                prelu_count++;
                ok &= prelu_count <= 1;
                non_scale_po_ = true;
                break;
            default: VDISPATCH_GEMM(false, VERBOSE_UNSUPPORTED_POSTOP);
        }
    }

    VDISPATCH_GEMM(ok, VERBOSE_UNSUPPORTED_POSTOP);

    if (with_sum_) {
        const auto idx = post_ops_.find(primitive_kind::sum);
        const auto &sum = post_ops_.entry_[idx].sum;
        if (beta_ * sum.zero_point != 0) {
            post_ops_t::entry_t::eltwise_t linear {
                    eltwise_linear, 1.0f, 1.0f, -beta_ * sum.zero_point};
            post_ops_t::entry_t linear_entry;
            linear_entry.kind = primitive_kind::eltwise;
            linear_entry.eltwise = linear;
            post_ops_.entry_.insert(
                    post_ops_.entry_.begin() + idx, linear_entry);
            binary_srcs_.insert(binary_srcs_.begin() + idx,
                    binary_src_t {binary_src_t::none, idx});
        }
    }

    // If scales are present, convert them and any bias to binary post-ops.
    //   Exception: 2D scales.
    // Also convert bias to binary post-op if dst zp are present.
    const auto &a_scales = attr()->scales_.get(DNNL_ARG_A);
    const auto &b_scales = attr()->scales_.get(DNNL_ARG_B);
    const auto &c_scales = attr()->scales_.get(DNNL_ARG_C);

    bias_via_binary_ = (desc()->bias_type() != data_type::undef)
            && (d->bias_desc.ndims >= 1 || !a_scales.has_default_values()
                    || !b_scales.has_default_values()
                    || !attr()->zero_points_.has_default_values(DNNL_ARG_C));
    if (bias_via_binary_) {
        VDISPATCH_GEMM_SC(post_ops_.prepend_binary(binary_add, &d->bias_desc),
                "%s: bias via binary post-op", VERBOSE_UNSUPPORTED_POSTOP);
        binary_srcs_.emplace(binary_srcs_.begin(), binary_src_t::bias, 0);
    }
    non_scale_po_ |= bias_via_binary_;

    auto maybe_convert_scales_to_postop
            = [this, engine](const memory_desc_t &scale_md, int arg,
                      int scale_ndims, bool mx, bool &converted) -> status_t {
        auto ndims = desc()->c_desc.ndims;
        // Scales on A/B can be converted to postops if
        // the scales md has K=1 and M/N is not bcast.
        converted = false;
        if (scale_ndims > 1) return status::success;
        int inner_dim = (arg == DNNL_ARG_A ? ndims - 2 : ndims - 1);
        bool convert = (scale_md.dims[inner_dim] <= 1) || (arg == DNNL_ARG_C);
        convert &= !mx;
        if (convert) {
            if (arg == DNNL_ARG_C) {
                VDISPATCH_GEMM_SC(
                        post_ops_.append_binary(binary_div, &scale_md),
                        "%s: %s scales via binary post-op",
                        VERBOSE_UNSUPPORTED_POSTOP, arg2str(arg).c_str());
                binary_srcs_.emplace_back(binary_src_t::scales, arg);
            } else {
                VDISPATCH_GEMM_SC(
                        post_ops_.prepend_binary(binary_mul, &scale_md),
                        "%s: %s scales via binary post-op",
                        VERBOSE_UNSUPPORTED_POSTOP, arg2str(arg).c_str());
                binary_srcs_.emplace(
                        binary_srcs_.begin(), binary_src_t::scales, arg);
            }
            converted = true;
        }
        return status::success;
    };

    if (!a_scales.has_default_values() && !a_scales.is_host_scalar()) {
        // Host scalar scale will be converted to Alpha
        bool converted;
        CHECK(maybe_convert_scales_to_postop(a_scale_md_, DNNL_ARG_A,
                a_quant.scale_ndims, a_scales.is_mx(), converted));
        if (converted) a_quant.scale_ndims = -1;
    }

    if (!b_scales.has_default_values() && !b_scales.is_host_scalar()) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(b_scale_md_, DNNL_ARG_B,
                b_quant.scale_ndims, b_scales.is_mx(), converted));
        if (converted) b_quant.scale_ndims = -1;
    }

    bool try_c_scale = !c_scales.is_host_scalar() || !with_inlined_c_scale();
    if (!c_scales.has_default_values() && try_c_scale) {
        bool converted;
        CHECK(maybe_convert_scales_to_postop(c_scale_md_, DNNL_ARG_C,
                c_quant.scale_ndims, c_scales.is_mx(), converted));
        // Conversion of dst scales to post ops is currently supported for all
        // cases supported in the library.
        gpu_assert(converted || c_scales.is_mx())
                << "Unable to convert dst scales to a post op";
    }

    return status::success;
}

bool pd_t::with_inlined_c_scale() const {
    if (!with_c_scales()) return false;
    const auto &c_scales = attr()->scales_.get(DNNL_ARG_C);
    if (!c_scales.is_host_scalar()) return false;
    return !non_scale_po_ && !with_sum_;
}

bool pd_t::dy_quant_enabled() const {
    const auto d = desc();
    using namespace data_type;
    bool all_f8 = (utils::one_of(d->a_type(), f8_e5m2, f8_e4m3)
            && utils::one_of(d->b_type(), f8_e5m2, f8_e4m3)
            && utils::one_of(d->c_type(), f8_e5m2, f8_e4m3, f16, bf16, f32));
    return (utils::one_of(d->c_type(), f32, f16, bf16, u8, s8)
                   && utils::one_of(d->a_type(), u8, s8, s4, u4, u2)
                   && utils::one_of(d->b_type(), u8, s8))
            || all_f8;
}

bool pd_t::wei_decomp() const {
    const auto d = desc();
    using namespace data_type;
    return (utils::one_of(d->c_type(), f32, f16, bf16, f8_e5m2, f8_e4m3)
                   && utils::one_of(d->a_type(), u8, s8, s4, u4, u2, u3,
                           f8_e4m3, f8_e5m2, f4_e2m1)
                   && utils::one_of(
                           d->b_type(), f16, f32, bf16, f8_e5m2, f8_e4m3))
            && types::data_type_bits(d->a_type())
            < types::data_type_bits(d->b_type())
            && attr()->mayiconvert(d->a_type(), f32);
}

bool pd_t::quant_enabled() const {
    return wei_decomp() || dy_quant_enabled();
}

status_t pd_t::init_attrs(const impl::engine_t *engine) {
    wei_decomp_ = wei_decomp();
    dy_quant_enabled_ = dy_quant_enabled();
    quant_enabled_ = quant_enabled();
    const auto &d = desc();

    const auto &attr_zps = attr()->zero_points_;
    const auto a_zps = attr_zps.get(DNNL_ARG_A);
    const auto b_zps = attr_zps.get(DNNL_ARG_B);
    const auto c_zps = attr_zps.get(DNNL_ARG_C);

    const auto &attr_gs = attr()->precomputed_reductions_;
    const auto a_gs = attr_gs.get(DNNL_ARG_A);
    const auto b_gs = attr_gs.get(DNNL_ARG_B);

    const auto &scales = attr()->scales_;
    const auto a_scales = scales.get(DNNL_ARG_A);
    const auto b_scales = scales.get(DNNL_ARG_B);
    const auto c_scales = scales.get(DNNL_ARG_C);

    cmask_a_ = a_zps.get_mask();
    cmask_b_ = b_zps.get_mask();
    cmask_c_ = c_zps.get_mask();

    // Swap descriptors to follow column major format
    VDISPATCH_GEMM_SC(a_zps.get_md(a_zp_md_, d->b_desc),
            VERBOSE_DESC_CREATION_FAIL, "A zero points");
    VDISPATCH_GEMM_SC(b_zps.get_md(b_zp_md_, d->a_desc),
            VERBOSE_DESC_CREATION_FAIL, "B zero points");
    VDISPATCH_GEMM_SC(c_zps.get_md(c_zp_md_, d->c_desc),
            VERBOSE_DESC_CREATION_FAIL, "C zero points");
    VDISPATCH_GEMM_SC(a_gs.get_md(a_gs_md_, d->b_desc),
            VERBOSE_DESC_CREATION_FAIL, "A group sums");
    VDISPATCH_GEMM_SC(b_gs.get_md(b_gs_md_, d->a_desc),
            VERBOSE_DESC_CREATION_FAIL, "B group sums");
    VDISPATCH_GEMM_SC(a_scales.get_md(a_scale_md_, desc_.b_desc),
            VERBOSE_DESC_CREATION_FAIL, "A scales");
    VDISPATCH_GEMM_SC(b_scales.get_md(b_scale_md_, desc_.a_desc),
            VERBOSE_DESC_CREATION_FAIL, "B scales");
    VDISPATCH_GEMM_SC(c_scales.get_md(c_scale_md_, desc_.c_desc),
            VERBOSE_DESC_CREATION_FAIL, "C scales");

    auto ndims = d->c_desc.ndims;
    a_quant.zp_ndims = quant_entry_ndims(a_zps, a_zp_md_, ndims - 2);
    b_quant.zp_ndims = quant_entry_ndims(b_zps, b_zp_md_, ndims - 1);
    a_quant.zp_after_scale = a_zps.is_sbzp();
    b_quant.zp_after_scale = b_zps.is_sbzp();
    c_quant.zp_ndims = quant_entry_ndims(c_zps, c_zp_md_, -1);
    a_quant.gs_ndims = quant_entry_ndims(a_gs, a_gs_md_, ndims - 2);
    b_quant.gs_ndims = quant_entry_ndims(b_gs, b_gs_md_, ndims - 1);
    a_quant.scale_ndims = quant_entry_ndims(a_scales, a_scale_md_, ndims - 2);
    b_quant.scale_ndims = quant_entry_ndims(b_scales, b_scale_md_, ndims - 1);
    c_quant.scale_ndims = quant_entry_ndims(c_scales, c_scale_md_, -1);

    a_quant.scales_type = a_scales.get_data_type();
    a_quant.zp_type = a_zps.get_data_type();
    a_quant.gs_type = a_gs.get_data_type();
    a_quant.force_gs = !a_gs.has_default_values();
    a_quant.zp_host_scalar = a_zp_host_scalar();
    // XXX, gemmstone support: if multiple grouped quantization attributes exist
    // for one matrix, they must have the same group size (default/unset is 0)
    const auto &set_if_consistent
            = [this, engine](int &quant_dim, int new_dim, int arg) -> status_t {
        VDISPATCH_GEMM(utils::one_of(quant_dim, 0, new_dim),
                "%s: %s quantization attrs with different group sizes",
                VERBOSE_UNSUPPORTED_ATTR, arg2str(arg).c_str());
        quant_dim = new_dim;
        return status::success;
    };
    const auto &set_a_groups = [&set_if_consistent](quant_params &quant,
                                       const quant_entry_t &entry) -> status_t {
        if (entry.has_default_groups()) return status::success;
        CHECK(set_if_consistent(
                quant.group_k, into<int>(entry.get_group(0)), DNNL_ARG_A));
        CHECK(set_if_consistent(
                quant.group_m, into<int>(entry.get_group(1)), DNNL_ARG_A));
        return status::success;
    };
    CHECK(set_a_groups(a_quant, a_zps));
    CHECK(set_a_groups(a_quant, a_gs));
    CHECK(set_a_groups(a_quant, a_scales));

    // static_sbzp needs 2D quantization. Parameters that do not vary over K are
    // passed as small groups with a zero K stride: one group spanning K is read
    // past its row and slows down kernel generation. A fallback: the row is
    // reloaded per group, which source reductions would avoid.
    const dim_t k_broadcast_group = 32;
    const dim_t k = desc()->k();
    if (a_quant.zp_after_scale && a_quant.zp_ndims > 0
            && k != DNNL_RUNTIME_DIM_VAL
            && utils::one_of(dim_t(a_quant.group_k), dim_t(0), k)) {
        // gemmstone expects the K group to divide K.
        a_quant.group_k = into<int>(math::gcd(k, k_broadcast_group));
        a_quant.group_m = std::max(a_quant.group_m, 1);
        a_quant.k_broadcast = true;
        a_quant.zp_ndims = 2;
        if (a_quant.scale_ndims > 0) a_quant.scale_ndims = 2;
    }

    b_quant.scales_type = b_scales.get_data_type();
    b_quant.zp_type = b_zps.get_data_type();
    b_quant.gs_type = b_gs.get_data_type();
    b_quant.force_gs = !b_gs.has_default_values();
    b_quant.zp_host_scalar = b_zp_host_scalar();
    const auto &set_b_groups = [&set_if_consistent](quant_params &quant,
                                       const quant_entry_t &entry) -> status_t {
        if (entry.has_default_groups()) return status::success;
        CHECK(set_if_consistent(
                quant.group_n, into<int>(entry.get_group(0)), DNNL_ARG_B));
        CHECK(set_if_consistent(
                quant.group_k, into<int>(entry.get_group(1)), DNNL_ARG_B));
        return status::success;
    };
    CHECK(set_b_groups(b_quant, b_zps));
    CHECK(set_b_groups(b_quant, b_gs));
    CHECK(set_b_groups(b_quant, b_scales));

    c_quant.scales_type = c_scales.get_data_type();
    c_quant.zp_type = c_zps.get_data_type();
    if (!c_scales.has_default_groups()) {
        c_quant.group_m = into<int>(c_scales.get_group(1));
        c_quant.group_n = into<int>(c_scales.get_group(0));
        with_mx_scale_ = c_scales.is_mx();
    }
    c_quant.zp_host_scalar = c_zp_host_scalar();

    return status::success;
}

status_t pd_t::zp_ok(const impl::engine_t *engine) {
    using namespace data_type;
    auto &attr_zps = attr()->zero_points_;
    if (attr_zps.has_default_values()) return status::success;
    auto &a_zps = attr_zps.get(DNNL_ARG_A);
    auto &b_zps = attr_zps.get(DNNL_ARG_B);
    auto &c_zps = attr_zps.get(DNNL_ARG_C);

    int ndims = desc()->a_desc.ndims;
    const bool a_smallint = utils::one_of(desc()->a_type(), s4, u4, u2);
    const bool b_smallint = utils::one_of(desc()->b_type(), s4, u4, u2);
    const bool weights_upconversion
            = wei_decomp_ || (a_smallint && dy_quant_enabled_);

    // static_sbzp: weights decompression with grouped scales and zero-points.
    VDISPATCH_GEMM(!b_quant.zp_after_scale, VERBOSE_UNSUPPORTED_ZP_CFG);
    VDISPATCH_GEMM(IMPLICATION(a_quant.zp_after_scale,
                           wei_decomp_ && a_zp_2d() && a_scales_2d()),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    if (!a_zps.has_default_values()) {
        VDISPATCH_GEMM(a_zps.get_data_type() != u3,
                "%s: int3 zero points are unsupported",
                VERBOSE_UNSUPPORTED_ZP_CFG);
        const bool has_prB
                = !attr()->precomputed_reductions_.has_default_values(
                        DNNL_ARG_B);
        // The src_zp x wei_zp term is unaccounted for in the generator.
        // Disable this path.
        VDISPATCH_GEMM(IMPLICATION(has_prB, b_zps.has_default_values()),
                VERBOSE_UNSUPPORTED_ZP_CFG);
        // Groups determine supported masks.
        if (!a_zps.has_default_groups()) {
            VDISPATCH_GEMM(valid_2d_mask(cmask_a_, ndims, weights_upconversion),
                    "%s: unsupported A mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            const auto a_q2d_group_n = a_zps.get_group(1);
            // Non-trivial N group unsupported.
            VDISPATCH_GEMM(a_q2d_group_n == 1,
                    "%s: Grouped N dimension on A matrix",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            // Zero points with non-trivial groups only supported with
            // precomputed reductions or when target tensor is being dequantized.
            // TODO: Re-examine this condition
            bool is_dequantized
                    = !dy_quant_enabled_ || !b_smallint || a_smallint;
            VDISPATCH_GEMM(IMPLICATION(a_zp_2d(), is_dequantized || has_prB),
                    "%s: Nontrivial groups on A matrix, and no precomputed "
                    "reductions or dequantization",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        } else {
            VDISPATCH_GEMM(utils::one_of(cmask_a_, 0, mask_per_oc, mask_per_ic),
                    "%s: unsupported A mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            // Weights zp can only be performantly enabled during upconversion
            // for cases that perform decompression.
            VDISPATCH_GEMM(
                    IMPLICATION(a_scales_2d(),
                            !(b_smallint && !wei_decomp_ && !a_smallint)),
                    "%s: 2D scales on A matrix, but no weights "
                    "decompression",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    if (!b_zps.has_default_values()) {
        VDISPATCH_GEMM(b_zps.get_data_type() != u3,
                "%s: int3 zero points are unsupported",
                VERBOSE_UNSUPPORTED_ZP_CFG);
        // INT4/2 ZPs on SRC do not expand the range in a meaningful way, skipping
        VDISPATCH_GEMM(!utils::one_of(b_zps.get_data_type(), s4, u4, u2),
                VERBOSE_UNSUPPORTED_ZP_CFG);

        // Groups determine supported masks.
        if (!b_zps.has_default_groups()) {
            VDISPATCH_GEMM(valid_2d_mask(cmask_b_, ndims, false),
                    "%s: unsupported B mask", VERBOSE_UNSUPPORTED_ZP_CFG);
            const auto b_q2d_group_n = b_zps.get_group(0);
            // Non-trivial M group unsupported.
            VDISPATCH_GEMM(utils::one_of(b_q2d_group_n, 1, desc()->n()),
                    "%s: Nontrivial N groups on B matrix",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
            // Zero points with non-trivial groups only supported
            // when target tensor is being dequantized.
            // TODO: Re-examine this condition
            bool is_dequantized
                    = !dy_quant_enabled_ || !a_smallint || b_smallint;
            VDISPATCH_GEMM(IMPLICATION(b_zp_2d(), is_dequantized),
                    "%s: Grouped B zero points, and no dequantization",
                    VERBOSE_UNSUPPORTED_ZP_CFG);
        } else {
            VDISPATCH_GEMM(utils::one_of(cmask_b_, 0, mask_scalar,
                                   mask_per_oc | mask_per_ic),
                    "%s: unsupported B mask", VERBOSE_UNSUPPORTED_ZP_CFG);
        }
    }

    if (!attr_zps.has_default_values(DNNL_ARG_C)) {
        VDISPATCH_GEMM(c_zps.get_data_type() != u3,
                "%s: int3 zero points are unsupported",
                VERBOSE_UNSUPPORTED_ZP_CFG);
        VDISPATCH_GEMM(
                IMPLICATION(!c_zps.is_host_scalar(),
                        utils::one_of(cmask_c_, 0, mask_scalar, mask_per_oc)),
                "%s: unsupported C mask", VERBOSE_UNSUPPORTED_ZP_CFG);
    }

    return status::success;
}

status_t pd_t::gs_ok(const impl::engine_t *engine) {
    auto &attr_gs = attr()->precomputed_reductions_;
    if (attr_gs.has_default_values()) return status::success;

    VDISPATCH_GEMM(attr_gs.has_default_values(DNNL_ARG_DST),
            VERBOSE_UNSUPPORTED_PR_CFG);

    bool with_a_group_sums_ = !attr_gs.has_default_values(DNNL_ARG_A);
    bool with_b_group_sums_ = !attr_gs.has_default_values(DNNL_ARG_B);

    VDISPATCH_GEMM(IMPLICATION(with_a_group_sums_,
                           attr_gs.get_data_type(DNNL_ARG_A) == data_type::s32),
            VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_GEMM(IMPLICATION(with_b_group_sums_,
                           attr_gs.get_data_type(DNNL_ARG_B) == data_type::s32),
            VERBOSE_UNSUPPORTED_DT_CFG);

    return status::success;
}

status_t pd_t::scales_ok(const impl::engine_t *engine) {
    const auto &scales = attr()->scales_;
    if (scales.has_default_values()) return status::success;
    int ndims = desc()->a_desc.ndims;
    using namespace data_type;

    for (auto s : {DNNL_ARG_A, DNNL_ARG_B}) {
        if (scales.has_default_values(s) || scales.get(s).is_host_scalar())
            continue;
        const auto &x_scales = scales.get(s);

        auto mask = x_scales.get_mask();
        bool supportedMask
                = utils::one_of(mask, 0, mask_scalar, mask_per_oc, mask_per_ic)
                || (!x_scales.has_default_groups()
                        && valid_2d_mask(mask, ndims));
        VDISPATCH_GEMM(supportedMask, "%s: unsupported A/B mask",
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        // Quantization over K requires explicit groups, otherwise the group
        // size stays unset (0). A is K x N, B is M x K.
        const int k_mask
                = s == DNNL_ARG_A ? (1 << (ndims - 2)) : (1 << (ndims - 1));
        VDISPATCH_GEMM(
                IMPLICATION((mask & k_mask), !x_scales.has_default_groups()),
                "%s: K quantization without groups",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    const auto &dst_scales = scales.get(DNNL_ARG_C);
    if (!dst_scales.has_default_values() && !dst_scales.is_host_scalar()) {
        auto mask = dst_scales.get_mask();
        bool supportedMask
                = utils::one_of(mask, 0, mask_scalar, mask_per_oc, mask_per_ic)
                || (!dst_scales.has_default_groups() && with_mx_scale()
                        && valid_2d_mask(mask, ndims));
        VDISPATCH_GEMM(supportedMask, "%s: unsupported C mask",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    if (!dst_scales.has_default_values() && with_mx_scale()) {
        // Dynamic Dst Quant only supported with `1x32` groups.
        VDISPATCH_GEMM(dst_scales.get_group(0) == 1
                        && dst_scales.get_group(1) == 32
                        && arch_ >= compute::gpu_arch_t::xe_hpc,
                "%s: unsupported mx_scale groups",
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        // M+N dimensions must have trivial strides for Dynamic Dst Quant
        auto md = &desc()->c_desc;
        auto strides = md->format_desc.blocking.strides;
        VDISPATCH_GEMM(strides[md->ndims - 1] == 1,
                "%s: unsupported mx_scale strides",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        VDISPATCH_GEMM(strides[md->ndims - 2] == md->dims[md->ndims - 1],
                "%s: unsupported mx_scale strides",
                VERBOSE_UNSUPPORTED_SCALES_CFG);
    }

    return status::success;
}

bool pd_t::valid_2d_mask(int mask, int ndims, bool per_tensor_ok) {
    return (mask == full_tensor_mask() && per_tensor_ok)
            || utils::one_of(mask, (1 << (ndims - 1)),
                    (1 << (ndims - 1)) + (1 << (ndims - 2)));
}

status_t transfer_post_ops(
        gemmstone::GEMMProblem &problem, gpu_post_ops_t &&post_ops_) {
    using namespace gemmstone;
    problem.postOps = std::move(post_ops_);
    const auto &post_ops = problem.postOps;

    if (post_ops.len() > 0) {

        size_t po_count = post_ops.len();
        problem.Tbinary.reserve(po_count);
        problem.binary.reserve(po_count);
        problem.postOps.binaryRow = {};
        problem.postOps.binaryCol = {};
        problem.postOps.binaryBatch = {};
        problem.postOps.binaryTrans = {};

        if (problem.Ta == Type::f16) problem.Ts = Type::f32;
        if (problem.Ta.isF8() || problem.Tb.isF8()) problem.Ts = Type::f32;

        for (size_t i = 0; i < po_count; i++) {
            const auto &entry = post_ops[i];
            if (!entry.is_binary()) {
                problem.Tbinary.push_back(Type::invalid);
                problem.binary.push_back(MatrixAddressing {});
                continue;
            }

            auto &src_rmd = entry.as_binary().src1_desc;

            auto T = convert_dnnl_to_kernel_type(src_rmd.dt);
            bool is_multi_row = (src_rmd.broadcast_mask & 1) == 0;
            bool is_multi_col = (src_rmd.broadcast_mask & 2) == 0;

            bool is_compatible = src_rmd.inner_layout.empty();
            if (!is_compatible) return status::unimplemented;

            bool trans = is_multi_row && !src_rmd.inner_dim.is_innermost();

            problem.Tbinary.push_back(T);
            problem.postOps.binaryRow[i] = is_multi_row;
            problem.postOps.binaryCol[i] = is_multi_col;
            problem.postOps.binaryBatch[i] = src_rmd.ndims() >= 3;
            problem.postOps.binaryTrans[i] = trans;

            MatrixAddressing atype;
            atype.layout = trans ? MatrixLayout::T : MatrixLayout::N;
            atype.crosspack = 1;
            atype.packSize = 0;
            atype.setAlignment(T.size());

            problem.binary.push_back(atype);
        }
    }

    return status::success;
}

status_t pd_t::init_GEMMProblem(
        gemmstone::GEMMProblem &problem, const intel::engine_t *engine) const {
    // Set up problem structure.
    using namespace gemmstone;
    problem = {};

    problem.product = engine->device_info()->product();
    bool has_systolic
            = engine->mayiuse(compute::device_ext_t::
                              intel_subgroup_matrix_multiply_accumulate)
            || engine->mayiuse(compute::device_ext_t::
                            intel_subgroup_split_matrix_multiply_accumulate);

    auto a_type = get_type(DNNL_ARG_A);
    auto b_type = get_type(DNNL_ARG_B);

    auto m = desc()->m();
    auto n = desc()->n();
    auto k = desc()->k();

    auto align_a = align(DNNL_ARG_A);
    auto align_b = align(DNNL_ARG_B);

    auto lda = ld(DNNL_ARG_A);
    auto ldb = ld(DNNL_ARG_B);

    auto trans_a = this->trans_a();
    auto trans_b = this->trans_b();

    if (swap_ab_) {
        std::swap(a_type, b_type);
        std::swap(m, n);
        std::swap(align_a, align_b);
        std::swap(lda, ldb);
        std::swap(trans_a, trans_b);
        trans_a = !trans_a;
        trans_b = !trans_b;
    }

    align_a = nstl::max(align_a, (int)types::data_type_size(a_type));
    auto a_size = (trans_a ? m : k) * lda * types::data_type_size(a_type);

    align_b = nstl::max(align_b, (int)types::data_type_size(b_type));
    auto b_size = (trans_b ? k : n) * ldb * types::data_type_size(b_type);

    bool int_acc = utils::one_of(a_type, data_type::s8, data_type::u8)
            || (types::is_integral_dt(a_type) && types::is_integral_dt(b_type));
    int_acc &= !(a_grouped() || b_grouped());
    auto c_type = desc()->c_type();
    auto align_c
            = nstl::max(align(DNNL_ARG_C), (int)types::data_type_size(c_type));
    auto ldc = desc()->ldc();
    auto c_size = n * ldc * types::data_type_size(c_type);

    auto co_type = with_bias() ? desc()->bias_type()
            : with_sum_ab()    ? desc()->sum_ab_type
            : int_acc          ? data_type::s32
                               : desc()->c_type();

    // Choose accumulation data type.
    auto acc_type = int_acc
            ? data_type::s32
            : (utils::one_of(data_type::f64, a_type, b_type) ? data_type::f64
                                                             : data_type::f32);

    bool with_binary = (post_ops_.find(primitive_kind::binary) != -1)
            || (post_ops_.find(primitive_kind::prelu) != -1);

    bool need_x32_acc = with_binary || !IMPLICATION(with_sum_, sum_at_begin_);
    auto acc_mode = attr()->acc_mode_;

    // Strict acc mode default.
    auto Tacc_mode
            = types::is_integral_dt(c_type) ? data_type::s32 : data_type::f32;

    // Initialize non-strict acc mode.
    switch (acc_mode) {
        case accumulation_mode::any:
            if (!need_x32_acc) Tacc_mode = data_type::undef;
            break;
        case accumulation_mode::f16: Tacc_mode = data_type::f16; break;
        case accumulation_mode::f32: Tacc_mode = data_type::f32; break;
        case accumulation_mode::s32: Tacc_mode = data_type::s32; break;
        default: break;
    }

    // Minimum precision type for applying post-ops based on acc mode.
    // Limits use of atomic add.
    problem.Tacc_min = convert_dnnl_to_kernel_type(Tacc_mode);

    if (acc_mode != accumulation_mode::strict) acc_type = Tacc_mode;

    if (wei_decomp_) { acc_type = data_type::f32; }

    auto trans_co = trans_bias();
    if (swap_ab_) trans_co = !trans_co;
    auto dst_sround = with_sround_;
    bool c_offset = with_c_zero_points();
    bool bias = with_bias();

    jit::quant_params a_quant = this->a_quant;
    jit::quant_params b_quant = this->b_quant;

    if (swap_ab()) {
        std::swap(a_quant, b_quant);
        std::swap(a_quant.group_m, a_quant.group_n);
        std::swap(b_quant.group_m, b_quant.group_n);
    }

    problem.Ta = problem.Ta_ext = convert_dnnl_to_kernel_type(a_type);
    problem.Tb = problem.Tb_ext = convert_dnnl_to_kernel_type(b_type);
    problem.Tc = convert_dnnl_to_kernel_type(acc_type);
    problem.Tc_ext = convert_dnnl_to_kernel_type(c_type);
    problem.Ts = problem.Tc;
    problem.Tao = convert_dnnl_to_kernel_type(a_quant.zp_type);
    problem.Tbo = convert_dnnl_to_kernel_type(b_quant.zp_type);
    problem.Tco = convert_dnnl_to_kernel_type(co_type);
    problem.A.layout = trans_a ? MatrixLayout::T : MatrixLayout::N;
    problem.B.layout = trans_b ? MatrixLayout::T : MatrixLayout::N;
    problem.C.layout = MatrixLayout::N;
    problem.A.crosspack = problem.B.crosspack = problem.C.crosspack = 1;
    problem.A.packSize = problem.B.packSize = problem.C.packSize = 0;
    problem.A.setAlignment(align_a);
    problem.B.setAlignment(align_b);
    problem.C.setAlignment(align_c);

    // Consolidate specialization logic to limit large buffer configurations
    bool needA64 = std::max({a_size, b_size, c_size})
            > std::numeric_limits<uint32_t>::max();
    problem.A.needA64 = needA64;
    problem.B.needA64 = needA64;
    problem.C.needA64 = needA64;

    if (batch_dims() > 0) {
        problem.batch = BatchMode::Strided;
        problem.batchDims = batch_dims();
    }
    if (a_quant.zp_ndims >= 0 || a_quant.zp_host_scalar)
        problem.aOffset = ABOffset::Calc;
    if (b_quant.zp_ndims >= 0 || b_quant.zp_host_scalar)
        problem.bOffset = ABOffset::Calc;
    problem.aoPtrDims = a_quant.zp_host_scalar ? -1 : a_quant.zp_ndims;
    problem.boPtrDims = b_quant.zp_host_scalar ? -1 : b_quant.zp_ndims;
    problem.aOffsetAfterScale = a_quant.zp_after_scale;
    problem.bOffsetAfterScale = b_quant.zp_after_scale;
    problem.asPtrDims = a_quant.scale_ndims;
    problem.bsPtrDims = b_quant.scale_ndims;

    problem.AO.layout = problem.BO.layout = MatrixLayout::N;
    problem.AO.crosspack = problem.BO.crosspack = 1;
    problem.AO.packSize = problem.BO.packSize = 0;
    problem.A_scale = problem.Ag = problem.AO;
    problem.B_scale = problem.Bg = problem.BO;

    // 1D tensors can be treated as either transposition - choose the one that
    // allows block loads (i.e. A -> N and B -> T)
    if (!problem.bOffset2D()) problem.BO.layout = MatrixLayout::T;
    if (!problem.bScale2D()) problem.B_scale.layout = MatrixLayout::T;
    if (b_quant.gs_ndims < 2) problem.Bg.layout = MatrixLayout::T;

    if (a_quant.zp_type != data_type::undef)
        problem.AO.setAlignment(int(types::data_type_size(a_quant.zp_type)));
    if (b_quant.zp_type != data_type::undef)
        problem.BO.setAlignment(int(types::data_type_size(b_quant.zp_type)));
    problem.aqGroupK = a_quant.group_k;
    problem.bqGroupK = b_quant.group_k;
    problem.aqGroupM = a_quant.group_m;
    problem.bqGroupN = b_quant.group_n;
    if (a_quant.scales_type != data_type::undef) {
        problem.Ta_scale = convert_dnnl_to_kernel_type(a_quant.scales_type);
        problem.A_scale.setAlignment(
                int(types::data_type_size(a_quant.scales_type)));
    }
    if (b_quant.scales_type != data_type::undef) {
        problem.Tb_scale = convert_dnnl_to_kernel_type(b_quant.scales_type);
        problem.B_scale.setAlignment(
                int(types::data_type_size(b_quant.scales_type)));
    }

    if (c_quant.scales_type != data_type::undef) {
        problem.csPtrDims = c_quant.scale_ndims;
        problem.cMXScale = with_mx_scale_;
        problem.Tc_scale = convert_dnnl_to_kernel_type(c_quant.scales_type);
        problem.cqGroupM = c_quant.group_m;
        problem.cqGroupN = c_quant.group_n;
    }

    // Mixed s8/s4 DPAS support:
    // - Xe3p: Not supported, require s4->s8 upconversion
    // - pre-Xe3p: supported, but only when s4 matrix doesn't have zero points
    // There is no native s8/s3 or s8/s2 DPAS support, so int3 and int2 are
    // unconditionally upconverted to s8 when paired with an int8 matrix.
    bool has_s8s4_dpas = getCore(problem.product.family) != ngen::HW::Xe3p;
    if ((problem.Ta_ext.is3()
                || (problem.Ta_ext.is2() && problem.Ta_ext.isInteger()))
            && problem.Tb_ext.isInt8()) {
        problem.Ta = Type::s8;
    } else if (problem.Ta_ext.isInt4() && problem.Tb_ext.isInt8()) {
        bool s8s4_dpas_ok = has_s8s4_dpas && (a_quant.zp_ndims < 0);
        if (!s8s4_dpas_ok) problem.Ta = Type::s8;
    }
    if ((problem.Tb_ext.is3()
                || (problem.Tb_ext.is2() && problem.Tb_ext.isInteger()))
            && problem.Ta_ext.isInt8()) {
        problem.Tb = Type::s8;
    } else if (problem.Tb_ext.isInt4() && problem.Ta_ext.isInt8()) {
        bool s8s4_dpas_ok = has_s8s4_dpas && (b_quant.zp_ndims < 0);
        if (!s8s4_dpas_ok) problem.Tb = Type::s8;
    }

    if (problem.Ta.isInteger()) problem.Ts = Type::f32;

    if (alpha() == 1.0f) problem.alpha = (int)alpha();
    if (beta() == 0.0f || beta() == 1.0f) problem.beta = (int)beta();

    gpu_post_ops_t gpu_post_ops;
    CHECK(gpu_post_ops_t::make(
            gpu_post_ops, post_ops_, dst_md(), get_post_op_specializations()));

    CHECK(transfer_post_ops(problem, std::move(gpu_post_ops)));
    if (swap_ab()) {
        problem.postOps.transpose();
        for (auto &b : problem.binary)
            b.transpose();
    }

    auto reduce_ab = sum_ab();
    if (c_offset || bias || reduce_ab != sum_ab::sum_none) {
        assert(!(c_offset && bias));
        if (bias) problem.cOffset = COffset::Pre;
        if (c_offset) problem.cOffset = COffset::Post;
        problem.CO.crosspack = 1;
        problem.CO.alignment = problem.C.alignment;
        problem.CO.layout = trans_co ? MatrixLayout::T : MatrixLayout::N;
        problem.coPtrDims = c_quant.zp_host_scalar ? -1 : c_quant.zp_ndims;
    }

    problem.sumA = (reduce_ab == sum_ab::sum_b_col);
    problem.sumB = (reduce_ab == sum_ab::sum_a_row);
    if (swap_ab_) std::swap(problem.sumA, problem.sumB);
    problem.hasGroupSumsA = a_quant.force_gs;
    problem.hasGroupSumsB = b_quant.force_gs;

    problem.postOps.cStochasticRound = dst_sround;

    if (problem.needsAGroupSums() || problem.needsBGroupSums())
        problem.autoTypeConversions(has_systolic);

    if (problem.needsAGroupSums()) {
        VDISPATCH_GEMM(problem.hasGroupSumsA, VERBOSE_UNSUPPORTED_ZP_CFG);
        data_type_t gs_dt = a_quant.gs_type == data_type::undef
                ? data_type::s32
                : a_quant.gs_type;
        problem.Tag = convert_dnnl_to_kernel_type(gs_dt);
        problem.Ag.setAlignment(problem.Tag.paddedSize());
        if (problem.bqGroupK == 0) problem.bqGroupK = problem.aqGroupK;
        if (problem.aqGroupK == 0) problem.aqGroupK = problem.bqGroupK;
    }
    if (problem.needsBGroupSums()) {
        VDISPATCH_GEMM(problem.hasGroupSumsB, VERBOSE_UNSUPPORTED_ZP_CFG);
        data_type_t gs_dt = b_quant.gs_type == data_type::undef
                ? data_type::s32
                : b_quant.gs_type;
        problem.Tbg = convert_dnnl_to_kernel_type(gs_dt);
        problem.Bg.setAlignment(problem.Tbg.paddedSize());
        if (problem.aqGroupK == 0) problem.aqGroupK = problem.bqGroupK;
        if (problem.bqGroupK == 0) problem.bqGroupK = problem.aqGroupK;
    }
    // Disable bdpas with unsupported k dim.
    // TODO: Enable 2D block, masking scale loads.
    if (problem.nativeBDPAS()) {
        if (((!problem.Ta.isF4() || !problem.Tb.isF4()) || k % 64 == 0))
            problem.bdpasEnabled = true;
    }

    if (swap_ab_) {
        // problem.A and problem.B are already swapped + transposed
        // TODO: use problem.transpose() instead
        problem.AO.transpose();
        problem.BO.transpose();
        problem.A_scale.transpose();
        problem.B_scale.transpose();
        problem.Ag.transpose();
        problem.Bg.transpose();
    }

    // A zero K stride needs the quantization rows laid out along M (A) or N (B).
    VDISPATCH_GEMM(
            IMPLICATION(a_quant.k_broadcast, isColMajor(problem.AO.layout)),
            VERBOSE_UNSUPPORTED_ZP_CFG);
    VDISPATCH_GEMM(
            IMPLICATION(b_quant.k_broadcast, !isColMajor(problem.BO.layout)),
            VERBOSE_UNSUPPORTED_ZP_CFG);

    return status::success;
}

dim_t pd_t::ld_binary(int idx) const {
    switch (binary_srcs_[idx].type) {
        case binary_src_t::binary: {
            const auto &entry = post_ops_.entry_[idx];
            assert(entry.kind == primitive_kind::binary);
            return gemm_desc_t::get_ld(entry.binary.src1_desc);
        }
        case binary_src_t::bias: return desc()->ld_bias();
        case binary_src_t::prelu: {
            return gemm_desc_t::get_ld(prelu_wei_md);
        }

        default: return 1;
    }
}

dim_t pd_t::stride_binary(int idx, int stride) const {
    switch (binary_srcs_[idx].type) {
        case binary_src_t::binary:
        case binary_src_t::scales:
        case binary_src_t::bias: {
            const auto &entry = post_ops_.entry_[idx];
            assert(entry.kind == primitive_kind::binary);
            return gemm_desc_t::get_stride(entry.binary.src1_desc, stride);
        }
        case binary_src_t::prelu: {
            return gemm_desc_t::get_stride(prelu_wei_md, stride);
        }
        default: return 0;
    }
}

dim_t pd_t::scale_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, DNNL_ARG_A, DNNL_ARG_B));
    const memory_desc_t *md_ptr
            = (arg == DNNL_ARG_A) ? &a_scale_md_ : &b_scale_md_;
    const memory_desc_wrapper mdw(md_ptr);
    if (mdw.is_host_scalar_desc()) return 0;
    gpu_assert(mdw.is_plain()) << "Expected plain scale_md_";
    if (mdw.dims()[idx] == 1) return 0;
    return mdw.strides()[idx];
}

dim_t pd_t::zp_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, DNNL_ARG_A, DNNL_ARG_B));
    const memory_desc_t *md_ptr = (arg == DNNL_ARG_A) ? &a_zp_md_ : &b_zp_md_;
    const memory_desc_wrapper mdw(md_ptr);
    if (mdw.is_host_scalar_desc()) return 0;
    gpu_assert(mdw.is_plain()) << "Expected plain zp_md_";
    if (mdw.dims()[idx] == 1) return 0;
    return mdw.strides()[idx];
}

dim_t pd_t::gs_stride(int idx, int arg) const {
    gpu_assert(utils::one_of(arg, DNNL_ARG_A, DNNL_ARG_B));
    const memory_desc_t *md_ptr = (arg == DNNL_ARG_A) ? &a_gs_md_ : &b_gs_md_;
    const memory_desc_wrapper mdw(md_ptr);
    if (mdw.is_host_scalar_desc()) return 0;
    gpu_assert(mdw.is_plain()) << "Expected plain gs_md_";
    if (mdw.dims()[idx] == 1) return 0;
    return mdw.strides()[idx];
}

} // namespace jit
} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
