/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include <map>

#include "cpu/x64/matmul_inner_product.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

bool is_desired_mm_impl(const std::shared_ptr<primitive_desc_t> &matmul_pd) {
    // Only the brgemm-based matmul is used. Otherwise fall back to the
    // gemm-based inner product.
    return std::string(matmul_pd->name()).find("brg_matmul")
            != std::string::npos;
}

status_t create_matmul_pd(std::shared_ptr<primitive_desc_t> &matmul_pd,
        const engine_t *engine, const memory_desc_t *src_md,
        const memory_desc_t *wei_md, const memory_desc_t *dst_md,
        const memory_desc_t *bia_md, const memory_desc_t *reduce_md,
        const primitive_attr_t *attr) {
    auto matmul_desc = matmul_desc_t();

    CHECK(matmul_desc_init(&matmul_desc, src_md, wei_md, bia_md, dst_md,
            reduce_md, matmul_reduce_kind::src));

    primitive_desc_iterator_t it(
            engine, (op_desc_t *)&matmul_desc, attr, nullptr);

    while (it != it.end()) {
        matmul_pd = *(++it);
        if (!matmul_pd) return it.no_impl_status();
        if (is_desired_mm_impl(matmul_pd)) break;
    }

    return status::success;
}

status_t init_matmul_md(memory_desc_t &mm_md, const memory_desc_t &ip_md,
        format_tag_t tag, bool swap_dims) {
    auto p_dims = ip_md.dims;
    auto p_dim1 = utils::array_product(p_dims + 1, ip_md.ndims - 1);

    if (swap_dims) {
        dims_t dims_2d = {p_dim1, p_dims[0]};
        return memory_desc_init_by_tag(mm_md, 2, dims_2d, ip_md.data_type, tag);
    } else {
        dims_t dims_2d = {p_dims[0], p_dim1};
        return memory_desc_init_by_tag(mm_md, 2, dims_2d, ip_md.data_type, tag);
    }
}

static bool check_training_formats(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &bias_d,
        const memory_desc_wrapper &dst_d, prop_kind_t prop_kind) {
    using namespace format_tag;
    using namespace utils;

    bool ok = src_d.matches_one_of_tag(ab, acb, acdb, acdeb)
            && dst_d.matches_tag(ab);

    if (!bias_d.is_zero()) ok = ok && bias_d.matches_tag(x);

    auto tag = utils::pick(src_d.ndims() - 2, ab, acb, acdb, acdeb);
    bool tag_matches
            = IMPLICATION(src_d.matches_tag(tag), wei_d.matches_tag(tag));
    // Note: allow forward_training weights' OC to be the innermost.
    if (!tag_matches && prop_kind == prop_kind::forward_training) {
        tag = utils::pick(src_d.ndims() - 2, ba, bca, bcda, bcdea);
        tag_matches
                = IMPLICATION(src_d.matches_tag(tag), wei_d.matches_tag(tag));
    }

    ok = ok && tag_matches && src_d.is_dense() && wei_d.is_dense()
            && dst_d.is_dense();
    return ok;
}

status_t set_training_formats(memory_desc_t *src_md, memory_desc_t *wei_md,
        memory_desc_t *bias_md, memory_desc_t *dst_md, prop_kind_t prop_kind) {
    assert(prop_kind != prop_kind::forward_inference);

    using namespace format_tag;

    const int ndims = src_md->ndims;
    const auto tag = utils::pick(ndims - 2, ab, acb, acdb, acdeb);
    if (src_md->format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(*src_md, tag));

    if (wei_md->format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(*wei_md, tag));

    if (dst_md->format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(*dst_md, ab));

    if (bias_md && bias_md->format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(*bias_md, x));

    return check_training_formats(src_md, wei_md, bias_md, dst_md, prop_kind)
            ? status::success
            : status::unimplemented;
}

int matmul_inner_product_fwd_t::pd_t::get_k_blk(format_tag_t tag) const {
    using namespace format_tag;
    switch (tag) {
        case ab:
        case ba: return 0;
        case BA8a8b:
        case BA8a24b: return 8;
        case BA16a16b:
        case BA16a32b:
        case BA16a48b:
        case BA16a64b: return 16;
        case BA16a16b2a:
        case BA16a32b2a:
        case BA16a48b2a:
        case BA16a64b2a: return 32;
        case BA16a16b4a:
        case BA16a32b4a:
        case BA16a48b4a:
        case BA16a64b4a: return 64;
        default: assert(!"unsupported tag"); return -1;
    }
}

// This implementation is completely based on the MatMul primitive and is
// currently enabled only for `forward_inference` propagation kind.
//
// The implementation allows using blocked weights layouts directly or via
// the special tag `any`.
// The Inner Product weights must meet **ONE** of the following requirements to
// enable using the blocked layouts:
//   - Weights don't have spatial.
//   - Weights have unit spatial.
//   - Weights have non-unit spatial but the number of input channels is a
//     multiple of K block (returned by `get_k_blk()`).
//
// If none of the above requirements are met then a plain layout will be
// used.
//
// Note: this implementation is only guranteed to work with a set of the
// pre-defined layouts therefore there is no need to implement a generic
// mechanism to map inner product weights layouts to the matmul ones and
// vice versa.
status_t matmul_inner_product_fwd_t::pd_t::init_matmul_params(
        const engine_t *engine) {
    using namespace format_tag;

    // clang-format off
    static const std::map<format_tag_t, std::vector<format_tag_t>> mm_wei_to_ip_wei = {
        { ba, {ab, acb, acdb, acdeb}},
        { ab, {ba, bca, bcda, bcdea}},
        { BA8a8b, {AB8b8a, AcB8b8a, AcdB8b8a, AcdeB8b8a}},
        { BA8a24b, {AB8b24a, AcB8b24a, AcdB8b24a, AcdeB8b24a}},
        { BA16a16b, {AB16b16a, AcB16b16a, AcdB16b16a, AcdeB16b16a}},
        { BA16a32b, {AB16b32a, AcB16b32a, AcdB16b32a, AcdeB16b32a}},
        { BA16a48b, {AB16b48a, AcB16b48a, AcdB16b48a, AcdeB16b48a}},
        { BA16a64b, {AB16b64a, AcB16b64a, AcdB16b64a, AcdeB16b64a}},
        { BA16a16b2a, {AB16b16a2b, AcB16b16a2b, AcdB16b16a2b, AcdeB16b16a2b}},
        { BA16a32b2a, {AB16b32a2b, AcB16b32a2b, AcdB16b32a2b, AcdeB16b32a2b}},
        { BA16a48b2a, {AB16b48a2b, AcB16b48a2b, AcdB16b48a2b, AcdeB16b48a2b}},
        { BA16a64b2a, {AB16b64a2b, AcB16b64a2b, AcdB16b64a2b, AcdeB16b64a2b}},
        { BA16a16b4a, {AB16b16a4b, AcB16b16a4b, AcdB16b16a4b, AcdeB16b16a4b}},
        { BA16a32b4a, {AB16b32a4b, AcB16b32a4b, AcdB16b32a4b, AcdeB16b32a4b}},
        { BA16a48b4a, {AB16b48a4b, AcB16b48a4b, AcdB16b48a4b, AcdeB16b48a4b}},
        { BA16a64b4a, {AB16b64a4b, AcB16b64a4b, AcdB16b64a4b, AcdeB16b64a4b}}};
    // clang-format on

    auto mm_wei_tag = format_tag::undef;
    // Try to initialize Inner Product weights layout based on the user-provided
    // layout.
    if (weights_md()->format_kind != format_kind::any) {
        for (const auto &v : mm_wei_to_ip_wei) {
            if (memory_desc_matches_tag(
                        *weights_md(), v.second[weights_md()->ndims - 2])) {
                mm_wei_tag = v.first;
                // Check if the user-provided blocked layout can be handled.
                const bool has_spatial = KD() + KH() + KW() > 3;
                const int k_blk = get_k_blk(mm_wei_tag);
                const bool is_wtag_supported = !(weights_md()->ndims > 2
                        && has_spatial && k_blk > 0 && IC() % k_blk != 0);
                VDISPATCH_INNER_PRODUCT(is_wtag_supported,
                        VERBOSE_UNSUPPORTED_TAG_S, "weights");
                break;
            }
        }
    } else {
        mm_wei_tag = format_tag::any;
    }

    VDISPATCH_INNER_PRODUCT(mm_wei_tag != format_tag::undef,
            VERBOSE_UNSUPPORTED_TAG_S, "weights");

    memory_desc_t mm_src_md {};
    memory_desc_t mm_wei_md {};
    memory_desc_t mm_dst_md {};

    if (bias_md_.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(bias_md_, x));

    CHECK(init_matmul_md(mm_src_md, *src_md(), format_tag::ab));
    CHECK(init_matmul_md(mm_wei_md, *weights_md(), mm_wei_tag, true));
    CHECK(init_matmul_md(mm_dst_md, *dst_md(), format_tag::ab));

    const auto src_tag = utils::pick(src_md()->ndims - 2, ab, acb, acdb, acdeb);
    if (src_md()->format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md_, src_md_.ndims, src_md_.dims,
                src_md_.data_type, src_tag));
    else
        VDISPATCH_INNER_PRODUCT(memory_desc_matches_tag(*src_md(), src_tag),
                VERBOSE_UNSUPPORTED_TAG_S, "src");

    const auto dst_tag = ab;
    if (dst_md()->format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md_, dst_md_.ndims, dst_md_.dims,
                dst_md_.data_type, dst_tag));
    else
        VDISPATCH_INNER_PRODUCT(memory_desc_matches_tag(*dst_md(), dst_tag),
                VERBOSE_UNSUPPORTED_TAG_S, "dst");

    VDISPATCH_INNER_PRODUCT_SC(
            attr_.set_default_formats(dst_md(0)), VERBOSE_UNSUPPORTED_POSTOP);

    primitive_attr_t matmul_attr = *attr();
    if (!matmul_attr.scales_.has_default_values(DNNL_ARG_WEIGHTS)) {
        const auto wei_mask = matmul_attr.scales_.get_mask(DNNL_ARG_WEIGHTS);
        if (wei_mask == 1) {
            VDISPATCH_INNER_PRODUCT_SC(matmul_attr.scales_.set(DNNL_ARG_WEIGHTS,
                                               1 << (mm_wei_md.ndims - 1)),
                    VERBOSE_UNSUPPORTED_ATTR);
        } else if (wei_mask > 0) {
            VDISPATCH_INNER_PRODUCT(false, VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
    }

    memory_desc_t mm_bia_md {};
    // Inner Product bias is always a vector while MatMul requires bias to have
    // the same number of dimensions as that of the output tensor, therefore an
    // adjustment is required.
    if (with_bias()) {
        assert(weights_md(1)->ndims == 1);
        dims_t mm_bia_dims = {1, weights_md(1)->dims[0]};
        CHECK(memory_desc_init_by_tag(mm_bia_md, 2, mm_bia_dims,
                weights_md(1)->data_type, format_tag::ab));
    }

    VDISPATCH_INNER_PRODUCT_SC(
            create_matmul_pd(matmul_pd_, engine, &mm_src_md, &mm_wei_md,
                    &mm_dst_md, with_bias() ? &mm_bia_md : nullptr, nullptr,
                    &matmul_attr),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");

    // Try to initialize Inner Product weights layout based on the MatMul's one.
    if (weights_md()->format_kind == format_kind::any) {
        // If the table doesn't have the required layout then fallback
        // is needed.
        bool is_fallback_required = true;
        format_tag_t ip_wei_tag = format_tag::undef;
        const auto &mm_queried_wei_md = *matmul_pd_->weights_md();
        for (const auto &v : mm_wei_to_ip_wei) {
            if (memory_desc_matches_tag(mm_queried_wei_md, v.first)) {
                // Check if the implementation defined blocked layout can be
                // handled.
                const bool has_spatial = KD() + KH() + KW() > 3;
                const int k_blk = get_k_blk(v.first);
                is_fallback_required = weights_md()->ndims > 2 && has_spatial
                        && k_blk > 0 && IC() % k_blk != 0;

                if (!is_fallback_required)
                    ip_wei_tag = v.second[weights_md()->ndims - 2];
                break;
            }
        }
        if (is_fallback_required) {
            // Re-initialize MatMul weights memory descriptor with a plain
            // layout.
            CHECK(init_matmul_md(
                    mm_wei_md, *weights_md(), format_tag::ba, true));
            // Re-create MatMul primitive descriptor.
            VDISPATCH_INNER_PRODUCT_SC(
                    create_matmul_pd(matmul_pd_, engine, &mm_src_md, &mm_wei_md,
                            &mm_dst_md, with_bias() ? &mm_bia_md : nullptr,
                            nullptr, &matmul_attr),
                    VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");
            ip_wei_tag = utils::pick(
                    weights_md()->ndims - 2, ab, acb, acdb, acdeb);
        }
        CHECK(memory_desc_init_by_tag(weights_md_, weights_md_.ndims,
                weights_md_.dims, weights_md_.data_type, ip_wei_tag));
        // Carry over the extra info from MatMul weights memory descriptor.
        if (!is_fallback_required && mm_queried_wei_md.extra.flags != 0) {
            weights_md_.extra = mm_queried_wei_md.extra;
            // Since IP weights are transposed we need to swap bits
            // (mask: 2 -> 1).
            weights_md_.extra.compensation_mask = 1;
        }
    }

    return status::success;
}

status_t matmul_inner_product_bwd_data_t::pd_t::init_matmul_params(
        const engine_t *engine) {
    memory_desc_t mm_src_md {};
    memory_desc_t mm_wei_md {};
    memory_desc_t mm_dst_md {};

    CHECK(init_matmul_md(mm_src_md, *diff_dst_md(), format_tag::ab));
    CHECK(init_matmul_md(mm_wei_md, *weights_md(), format_tag::ab));
    CHECK(init_matmul_md(mm_dst_md, *diff_src_md(), format_tag::ab));

    VDISPATCH_INNER_PRODUCT_SC(
            create_matmul_pd(matmul_pd_, engine, &mm_src_md, &mm_wei_md,
                    &mm_dst_md, nullptr, nullptr, attr()),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");

    return status::success;
}

status_t matmul_inner_product_bwd_weights_t::pd_t::init_matmul_params(
        const engine_t *engine) {
    memory_desc_t mm_src_md {};
    memory_desc_t mm_wei_md {};
    memory_desc_t mm_dst_md {};

    CHECK(init_matmul_md(mm_src_md, *diff_dst_md(), format_tag::ba, true));
    CHECK(init_matmul_md(mm_wei_md, *src_md(), format_tag::ab));
    CHECK(init_matmul_md(mm_dst_md, *diff_weights_md(), format_tag::ab));

    memory_desc_t reduce_md {};

    if (with_bias()) {
        const memory_desc_t &diff_bias_md = *diff_weights_md(1);
        dims_t reduce_dims {};
        reduce_dims[0] = diff_bias_md.dims[0];
        reduce_dims[1] = 1;

        CHECK(memory_desc_reshape(reduce_md, diff_bias_md, 2, reduce_dims));
    }

    VDISPATCH_INNER_PRODUCT_SC(
            create_matmul_pd(matmul_pd_, engine, &mm_src_md, &mm_wei_md,
                    &mm_dst_md, nullptr, with_bias() ? &reduce_md : nullptr,
                    attr()),
            VERBOSE_PRIMITIVE_CREATION_FAIL, "matmul");

    return status::success;
}

status_t matmul_inner_product_fwd_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t matmul_args = ctx.args();
    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);

    return matmul_->execute(matmul_ctx);
}

status_t matmul_inner_product_bwd_data_t::execute(const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t matmul_args;
    matmul_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_DIFF_DST);
    matmul_args[DNNL_ARG_WEIGHTS] = ctx.args().at(DNNL_ARG_WEIGHTS);
    matmul_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DIFF_SRC);

    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);

    return matmul_->execute(matmul_ctx);
}

status_t matmul_inner_product_bwd_weights_t::execute(
        const exec_ctx_t &ctx) const {
    using namespace memory_tracking::names;

    exec_args_t matmul_args;
    matmul_args[DNNL_ARG_SRC] = ctx.args().at(DNNL_ARG_DIFF_DST);
    matmul_args[DNNL_ARG_WEIGHTS] = ctx.args().at(DNNL_ARG_SRC);
    matmul_args[DNNL_ARG_DST] = ctx.args().at(DNNL_ARG_DIFF_WEIGHTS);

    if (pd()->with_bias())
        matmul_args[DNNL_ARG_REDUCE] = ctx.args().at(DNNL_ARG_DIFF_BIAS);

    exec_ctx_t matmul_ctx(ctx, std::move(matmul_args));

    auto *nested_grantor = create_nested_grantor(ctx.get_scratchpad_grantor(),
            key_nested, matmul_->pd()->scratchpad_registry());
    matmul_ctx.set_scratchpad_grantor(nested_grantor);
    return matmul_->execute(matmul_ctx);
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
