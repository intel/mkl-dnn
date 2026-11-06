/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "cpu/reorder/simple_reorder.hpp"

#include <cstring>
#include <vector>

#include "cpu/ref_io_helper.hpp"
#include "cpu/simple_q10n.hpp"

#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
// Everything past dim 0 is dense when the span of those dimensions equals the
// number of elements they hold, leaving dim 0 as the only dimension allowed to
// carry a stride gap.
// TODO: consider merging with `memory_desc_wrapper::is_dense`. It is kept
// separate for now because its implementation differs and unifying them would
// require a whole-library validation that is out of scope here.
bool is_dense_no_dim_0(const memory_desc_wrapper &data_d) {
    return static_cast<dim_t>(data_d.span(1)) == data_d.nelems(false, 1);
}
} // namespace

status_t simple_reorder_t::pd_t::init(const engine_t *engine,
        const engine_t *src_engine, const engine_t *dst_engine) {
    using namespace data_type;

    VDISPATCH_REORDER(impl::is_dense_format_kind({src_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    const memory_desc_wrapper input_d(src_md());
    const memory_desc_wrapper output_d(dst_md());

    // TODO: enable checks once ready.
    // VDISPATCH_REORDER(platform::has_data_type_support(input_d.data_type()),
    //         VERBOSE_UNSUPPORTED_DT);
    // VDISPATCH_REORDER(platform::has_data_type_support(output_d.data_type()),
    //         VERBOSE_UNSUPPORTED_DT);

    // These types are not supported in this implementation as they would
    // require special handling on a common infrastructure side (loading and
    // storing 64-bit values for int64_t and double properly), which doesn't
    // seem to bring any value to the library at this point, thus, just making
    // sure they are not expected here.
    VDISPATCH_REORDER(!utils::one_of(input_d.data_type(), f64, s64),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(!utils::one_of(output_d.data_type(), f64, s64),
            VERBOSE_UNSUPPORTED_DT);

    // Sub-byte data types require dedicated bit-level handling which is a
    // prerogative of `ref_reorder` only. Reject them before any kernel is
    // selected so no specialization has to worry about them.
    VDISPATCH_REORDER(types::data_type_bits(input_d.data_type()) >= 8,
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_REORDER(types::data_type_bits(output_d.data_type()) >= 8,
            VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_REORDER(
            input_d.is_blocking_desc() && !input_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    // The dst may carry an additional (compensation) buffer; only the
    // `plain_to_dense_comp` kernel consumes it, the other kernels reject it
    // in their own `is_*` checks.
    VDISPATCH_REORDER(output_d.is_blocking_desc(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    auto gpu_zp = memory_extra_flags::compensation_gpu_conv_asymmetric_src;
    VDISPATCH_REORDER(!(dst_md()->extra.flags & gpu_zp),
            VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    // Select a concrete kernel by probing each specialization's applicability.
    // The first match wins; sub-calls dispatch on `kernel_kind_` at execute
    // time. If nothing matches, the problem is not supported.
    kernel_kind_ = kernel_kind_t::unknown;
    if (is_direct_copy(engine, input_d, output_d, attr()) == status::success)
        kernel_kind_ = kernel_kind_t::direct_copy;
    else if (is_plain_to_blocked(engine, input_d, output_d, attr())
            == status::success)
        kernel_kind_ = kernel_kind_t::plain_to_blocked;
    else if (is_blocked_to_blocked(engine, input_d, output_d, attr())
            == status::success)
        kernel_kind_ = kernel_kind_t::blocked_to_blocked;
    else if (is_plain_to_dense_comp(engine, input_d, output_d, attr())
            == status::success)
        kernel_kind_ = kernel_kind_t::plain_to_dense_comp;

    VDISPATCH_REORDER(kernel_kind_ != kernel_kind_t::unknown,
            VERBOSE_IMPL_HEURISTIC_FAIL, "no applicable kernel kind");

    // The compensation kernel caches per-dim src/dst offset tables in the
    // scratchpad so they outlive `execute()` under an asynchronous threadpool
    // (where the compute closure runs after the call returns). The tables hold
    // one offset per index of every (real) dim, for both src and dst.
    if (kernel_kind_ == kernel_kind_t::plain_to_dense_comp) {
        dim_t tab_total = 0;
        for (int d = 0; d < output_d.ndims(); ++d)
            tab_total += output_d.dims()[d];
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book<dim_t>(
                memory_tracking::names::key_reorder_space, 2 * tab_total);
    }

    return status::success;
}

status_t simple_reorder_t::pd_t::is_direct_copy(const engine_t *engine,
        const memory_desc_wrapper &input_d, const memory_desc_wrapper &output_d,
        const primitive_attr_t *attr) {
    VDISPATCH_REORDER(!input_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Only per-tensor (mask 0), default data type and default groups scales,
    // plus an optional sum post-op, are supported; zero points are not. The
    // skip mask implicitly rejects non-default zero points.
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(attr->has_default_values(
                              skip_mask_t::scales | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    const auto &post_ops = attr->post_ops_;
    VDISPATCH_REORDER(
            IMPLICATION(post_ops.len() != 0,
                    post_ops.len() == 1 && post_ops.entry_[0].is_sum(false)),
            VERBOSE_UNSUPPORTED_POSTOP);

    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        VDISPATCH_REORDER(attr->scales_.has_default_data_type(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_REORDER(attr->scales_.has_default_groups(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        if (!attr->scales_.has_default_values(arg))
            VDISPATCH_REORDER(
                    attr->scales_.get_mask(arg) == 0, VERBOSE_UNSUPPORTED_ATTR);
    }

    // An e8m0 source or destination is copied through a dedicated raw-conversion
    // path (see execute_direct_copy) that must not apply any floating-point
    // operation around the value, thus, no attributes can be supported.
    VDISPATCH_REORDER(
            IMPLICATION(utils::one_of(data_type::e8m0, input_d.data_type(),
                                output_d.data_type()),
                    attr->has_default_values()),
            VERBOSE_UNSUPPORTED_ATTR);

    // `direct_copy` requires identical src/dst layouts for every dimension but
    // the outermost one, which may carry a stride gap (e.g. a padded dim 0).
    // The dim-0-contiguous case is naturally covered as a sub-case.
    VDISPATCH_REORDER(is_dense_no_dim_0(input_d),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    VDISPATCH_REORDER(is_dense_no_dim_0(output_d),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");
    VDISPATCH_REORDER(input_d.similar_to(output_d, true, false, 1),
            VERBOSE_TENSOR_FORMAT_MISMATCH, "src", "dst");

    // Compensation buffers are handled by `plain_to_dense_comp` only.
    VDISPATCH_REORDER(!output_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    return status::success;
}

status_t simple_reorder_t::pd_t::is_plain_to_blocked(const engine_t *engine,
        const memory_desc_wrapper &input_d, const memory_desc_wrapper &output_d,
        const primitive_attr_t *attr) {
    VDISPATCH_REORDER(!input_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Only per-tensor (mask 0), default data type and default groups scales,
    // plus an optional sum post-op, are supported; zero points are not. The
    // skip mask implicitly rejects non-default zero points.
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(attr->has_default_values(
                              skip_mask_t::scales | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    const auto &post_ops = attr->post_ops_;
    VDISPATCH_REORDER(
            IMPLICATION(post_ops.len() != 0,
                    post_ops.len() == 1 && post_ops.entry_[0].is_sum(false)),
            VERBOSE_UNSUPPORTED_POSTOP);

    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        VDISPATCH_REORDER(attr->scales_.has_default_data_type(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_REORDER(attr->scales_.has_default_groups(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        if (!attr->scales_.has_default_values(arg))
            VDISPATCH_REORDER(
                    attr->scales_.get_mask(arg) == 0, VERBOSE_UNSUPPORTED_ATTR);
    }

    // `plain_to_blocked` converts between a plain layout and a blocked one
    // whose inner blocks cover one or two distinct logical dims (a dim may be
    // blocked more than once, e.g. `8a16b2a`). Exactly one side must be plain.
    const bool src_plain = input_d.is_plain();
    const bool dst_plain = output_d.is_plain();
    VDISPATCH_REORDER(src_plain != dst_plain, VERBOSE_TENSOR_FORMAT_MISMATCH,
            "src", "dst");

    const auto &blocked_d = src_plain ? output_d : input_d;
    const char *blocked_arg = src_plain ? "dst" : "src";
    const auto &blk = blocked_d.blocking_desc();
    VDISPATCH_REORDER(blk.inner_nblks >= 1, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT,
            blocked_arg);

    // Collect the distinct blocked logical dims. One or two are supported;
    // three or more (`d1 == -2`) are not.
    dim_t d0 = blk.inner_idxs[0], d1 = -1;
    for (int i = 1; i < blk.inner_nblks; ++i) {
        const dim_t d = blk.inner_idxs[i];
        if (d == d0 || d == d1) continue;
        d1 = d1 == -1 ? d : -2;
    }
    VDISPATCH_REORDER(d1 != -2, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
    if (d1 >= 0 && d0 > d1) nstl::swap(d0, d1);

    const int ndims = blocked_d.ndims();
    if (d1 == -1) {
        // A single inner block on dim 0 or dim 1 (e.g. nchw <-> nChw16c).
        // `ndims >= 3` guarantees the innermost logical dim is distinct from
        // the blocked one, which the kernel iterates as its contiguous inner
        // loop.
        VDISPATCH_REORDER(utils::one_of(d0, 0, 1),
                VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
        VDISPATCH_REORDER(ndims >= 3 && ndims <= 6,
                VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
    } else {
        // Two blocked dims anchored at dims {0,1} (plain weights, e.g.
        // OIhw16i16o) or at dims {1,2} with a leading group dim (grouped
        // weights, e.g. gOIhw16i16o). This mirrors the `_AB`/`_BC` layouts.
        // `ndims >= d1 + 2` guarantees a trailing non-blocked innermost dim.
        const bool ab = d0 == 0 && d1 == 1;
        const bool bc = d0 == 1 && d1 == 2;
        VDISPATCH_REORDER(
                ab || bc, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
        VDISPATCH_REORDER(ndims >= d1 + 2 && ndims <= (ab ? 5 : 6),
                VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, blocked_arg);
    }

    // Compensation buffers are handled by `plain_to_dense_comp` only.
    VDISPATCH_REORDER(!output_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    return status::success;
}

// TODO: remove this kernel kind, along with its `execute_*` counterpart and the
// dispatch wiring in `init`/`execute`, once blocked activation layouts are
// dropped from the library.
status_t simple_reorder_t::pd_t::is_blocked_to_blocked(const engine_t *engine,
        const memory_desc_wrapper &input_d, const memory_desc_wrapper &output_d,
        const primitive_attr_t *attr) {
    VDISPATCH_REORDER(!input_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // Only per-tensor (mask 0), default data type and default groups scales,
    // plus an optional sum post-op, are supported; zero points are not. The
    // skip mask implicitly rejects non-default zero points.
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(attr->has_default_values(
                              skip_mask_t::scales | skip_mask_t::post_ops),
            VERBOSE_UNSUPPORTED_ATTR);

    const auto &post_ops = attr->post_ops_;
    VDISPATCH_REORDER(
            IMPLICATION(post_ops.len() != 0,
                    post_ops.len() == 1 && post_ops.entry_[0].is_sum(false)),
            VERBOSE_UNSUPPORTED_POSTOP);

    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        VDISPATCH_REORDER(attr->scales_.has_default_data_type(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_REORDER(attr->scales_.has_default_groups(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        if (!attr->scales_.has_default_values(arg))
            VDISPATCH_REORDER(
                    attr->scales_.get_mask(arg) == 0, VERBOSE_UNSUPPORTED_ATTR);
    }

    // `blocked_to_blocked` re-blocks a single blocked logical dim (the channel
    // dim) between two blocked activation layouts whose inner blocks differ in
    // size (e.g. nChw8c <-> nChw16c). Both sides must be blocked exactly once,
    // over the same single logical dim, with different block sizes. Equal block
    // sizes would make the layouts identical and are served by `direct_copy`.
    VDISPATCH_REORDER(!input_d.is_plain() && !output_d.is_plain(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    const auto &iblk = input_d.blocking_desc();
    const auto &oblk = output_d.blocking_desc();
    VDISPATCH_REORDER(iblk.inner_nblks == 1 && oblk.inner_nblks == 1,
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    // The channel dim is dim 1 for every blocked activation layout (nCw/nChw/
    // nCdhw). Anchoring on it keeps this kernel scoped to activations.
    VDISPATCH_REORDER(iblk.inner_idxs[0] == 1 && oblk.inner_idxs[0] == 1,
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    VDISPATCH_REORDER(iblk.inner_blks[0] != oblk.inner_blks[0],
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    VDISPATCH_REORDER(input_d.ndims() == output_d.ndims(),
            VERBOSE_INCONSISTENT_NDIMS, "src", "dst");
    VDISPATCH_REORDER(input_d.ndims() >= 3 && input_d.ndims() <= 5,
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");

    // Compensation buffers are handled by `plain_to_dense_comp` only.
    VDISPATCH_REORDER(!output_d.is_additional_buffer(),
            VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");

    return status::success;
}

status_t simple_reorder_t::pd_t::is_plain_to_dense_comp(const engine_t *engine,
        const memory_desc_wrapper &input_d, const memory_desc_wrapper &output_d,
        const primitive_attr_t *attr) {
    using namespace data_type;
    using namespace memory_extra_flags;

    VDISPATCH_REORDER(!input_d.has_runtime_dims_or_strides(),
            VERBOSE_RUNTIMEDIM_UNSUPPORTED);

    // This kernel owns the s8 convolution-weights reorders that must fill an
    // appended compensation buffer: s8s8 and/or asymmetric-src (zero-point).
    // An optional scale adjustment may accompany them; any other extra flag is
    // out of scope.
    //
    // Dispatch is purely on runtime blocking, so the canonical forms are
    // handled directly.
    const auto &extra = output_d.extra();
    const bool req_s8s8 = extra.flags & compensation_conv_s8s8;
    const bool req_asymm = extra.flags & compensation_conv_asymmetric_src;
    VDISPATCH_REORDER(
            req_s8s8 || req_asymm, VERBOSE_UNSUPPORTED_MD_FLAG, "extra");
    const uint64_t supported = compensation_conv_s8s8
            | compensation_conv_asymmetric_src | scale_adjust;
    VDISPATCH_REORDER(
            !(extra.flags & ~supported), VERBOSE_UNSUPPORTED_MD_FLAG, "extra");
    // When both compensations are present they must share the same mask so a
    // single comp index space addresses both buffers.
    VDISPATCH_REORDER(
            IMPLICATION(req_s8s8 && req_asymm,
                    extra.compensation_mask == extra.asymm_compensation_mask),
            VERBOSE_UNSUPPORTED_MD_FLAG, "extra");

    VDISPATCH_REORDER(
            input_d.is_plain(), VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "src");
    // The dst may be plain (e.g. `wigo`) or blocked (e.g. `OIhw4i16o4i`); both
    // reduce to the same dense `off_v` addressing. Only the data type is pinned.
    VDISPATCH_REORDER(output_d.data_type() == s8, VERBOSE_UNSUPPORTED_DT);

    // Only scales are supported (any mask); no post-ops / zero points.
    using skip_mask_t = primitive_attr_t::skip_mask_t;
    VDISPATCH_REORDER(attr->has_default_values(skip_mask_t::scales),
            VERBOSE_UNSUPPORTED_ATTR);
    for (int arg : {DNNL_ARG_SRC, DNNL_ARG_DST}) {
        VDISPATCH_REORDER(attr->scales_.has_default_data_type(arg),
                VERBOSE_UNSUPPORTED_ATTR);
        VDISPATCH_REORDER(attr->scales_.has_default_groups(arg),
                VERBOSE_UNSUPPORTED_ATTR);
    }

    // The compensation buffer uses the dst tag trimmed to the masked logical
    // dims (a dense layout, plain or blocked; see `execute_plain_to_dense_comp`).
    // That is only well-defined when each masked dim is blocked at most once (so
    // it contributes a single outer/inner stride pair); for a plain dst no dim
    // is blocked and the layout collapses to row-major. The reduction over the
    // remaining dims is race-free because a single thread owns each comp index.
    const int comp_mask = req_s8s8 ? extra.compensation_mask
                                   : extra.asymm_compensation_mask;
    const auto &blk = output_d.blocking_desc();
    for (int d = 0; d < output_d.ndims(); ++d) {
        if (!((comp_mask >> d) & 1)) continue;
        int nblk = 0;
        for (int i = 0; i < blk.inner_nblks; ++i)
            if (blk.inner_idxs[i] == d) ++nblk;
        VDISPATCH_REORDER(nblk <= 1, VERBOSE_UNSUPPORTED_TENSOR_LAYOUT, "dst");
    }

    return status::success;
}

status_t simple_reorder_t::execute(const exec_ctx_t &ctx) const {
    switch (pd()->kernel_kind_) {
        case pd_t::kernel_kind_t::direct_copy: return execute_direct_copy(ctx);
        case pd_t::kernel_kind_t::plain_to_blocked:
            return execute_plain_to_blocked(ctx);
        case pd_t::kernel_kind_t::blocked_to_blocked:
            return execute_blocked_to_blocked(ctx);
        case pd_t::kernel_kind_t::plain_to_dense_comp:
            return execute_plain_to_dense_comp(ctx);
        case pd_t::kernel_kind_t::unknown: break;
    }
    assert(!"unknown kernel kind");
    return status::runtime_error;
}

status_t simple_reorder_t::execute_direct_copy(const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const auto dst_dt = output_d.data_type();

    // This kernel can be used for tensors with multiple inner blocks for
    // which generic zero padding must be used.
    ctx.zero_pad_output(DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM).data_type();
    const auto dst_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO).data_type();
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const float beta = pd()->beta();

    // Every dimension but dim 0 is dense and shares the same layout across
    // src/dst, so each dim-0 slab is addressed by its own stride (`is`/`os`)
    // while the inner `nelems_no_d0` elements are contiguous. The fully dense
    // case falls out naturally with `is == os == nelems_no_d0`.
    const dim_t N = input_d.dims()[0];
    const dim_t is = input_d.blocking_desc().strides[0];
    const dim_t os = output_d.blocking_desc().strides[0];
    const dim_t nelems_no_d0 = input_d.nelems(false, 1);
    const dim_t work_amount = N * nelems_no_d0;
    if (work_amount == 0) return status::success;

    // Advance each base to its logical element 0.
    input += input_d.blk_off(0) * input_d.data_type_size();
    output += output_d.blk_off(0) * output_d.data_type_size();

    // Pure copy (identical src/dst data type, no scales, no sum): the inner
    // `nelems_no_d0` elements of every dim-0 slab are contiguous and bitwise
    // identical, so stream them with `memcpy`. This avoids the per-element
    // float load/convert/store of the generic path below, which cannot match
    // `memcpy` throughput for a bitwise copy.
    if (src_dt == dst_dt && !with_src_scales && !with_dst_scales
            && beta == 0.f) {
        const dim_t dt_sz = input_d.data_type_size();
        parallel(0, [=](const int ithr, const int nthr) {
            dim_t n {0}, dim1_s {0};
            dim_t start {0}, end {0};
            balance211(work_amount, nthr, ithr, start, end);
            utils::nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
            while (start < end) {
                const dim_t work_rem = end - start;
                const dim_t dim1_e = nstl::min(dim1_s + work_rem, nelems_no_d0);
                const dim_t count = dim1_e - dim1_s;
                std::memcpy(output + (os * n + dim1_s) * dt_sz,
                        input + (is * n + dim1_s) * dt_sz, count * dt_sz);
                utils::nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
            }
        });
        return status::success;
    }

    parallel(0, [=](const int ithr, const int nthr) {
        const float src_scale = with_src_scales
                ? io::load_float_value(src_scales_dt, src_scales, 0)
                : 1.f;
        const float dst_scale = with_dst_scales
                ? io::load_float_value(dst_scales_dt, dst_scales, 0)
                : 1.f;
        const float alpha = src_scale / dst_scale;
        // Folding `1 / dst_scale` into `alpha` requires scaling the sum term
        // the same way, since `(src_scale * src + beta * dst) / dst_scale !=
        // alpha * src + beta * dst`.
        const float beta_scaled = with_dst_scales ? beta / dst_scale : beta;

        auto convert = [&](dim_t in_off, dim_t out_off) {
            if (src_dt == data_type::e8m0) {
                // Reorder from e8m0 to f32 is used for benchdnn correctness
                // validation purpose only. A dedicated path is required to
                // preserve a minimal e8m0 value which gets converted by any
                // compiler to 0 if any floating-point operation or comparison
                // is applied around the value.
                const float s = io::load_float_value(src_dt, input, in_off);
                io::store_float_value(dst_dt, s, output, out_off);
            } else {
                float d = alpha * io::load_float_value(src_dt, input, in_off);
                if (beta_scaled != 0.f)
                    d += beta_scaled
                            * io::load_float_value(dst_dt, output, out_off);
                io::store_float_value(dst_dt, d, output, out_off);
            }
        };

        dim_t n {0}, dim1_s {0};
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        utils::nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
        while (start < end) {
            const dim_t work_rem = end - start;
            const dim_t dim1_e = nstl::min(dim1_s + work_rem, nelems_no_d0);
            for (dim_t e = dim1_s; e < dim1_e; ++e)
                convert(is * n + e, os * n + e);
            utils::nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
        }
    });
    return status::success;
}

status_t simple_reorder_t::execute_plain_to_blocked(
        const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const auto dst_dt = output_d.data_type();

    // The blocked side carries the inner blocks; the other side is plain.
    const bool to_blocked = !output_d.is_plain();
    const auto &blocked_d = to_blocked ? output_d : input_d;
    const auto &flat_d = to_blocked ? input_d : output_d;

    const auto &blk = blocked_d.blocking_desc();
    const int nblks = blk.inner_nblks;
    const int ndims = blocked_d.ndims();
    const int last = ndims - 1;

    // Collect the distinct blocked logical dims (`d0`, and `d1 >= 0` when a
    // second one exists) and their total block sizes (product of the inner
    // blocks mapped onto each dim; a dim may be blocked more than once, e.g.
    // `8a16b2a`).
    dim_t d0 = blk.inner_idxs[0], d1 = -1;
    for (int i = 1; i < nblks; ++i)
        if (blk.inner_idxs[i] != d0) d1 = blk.inner_idxs[i];
    if (d1 >= 0 && d0 > d1) nstl::swap(d0, d1);
    dim_t bs0 = 1, bs1 = 1;
    dim_t in_idxs[DNNL_MAX_NDIMS] = {0};
    dim_t in_blks[DNNL_MAX_NDIMS] = {0};
    for (int i = 0; i < nblks; ++i) {
        in_idxs[i] = blk.inner_idxs[i];
        in_blks[i] = blk.inner_blks[i];
        if (blk.inner_idxs[i] == d0)
            bs0 *= blk.inner_blks[i];
        else
            bs1 *= blk.inner_blks[i];
    }

    // The blocked dst holds zero padding in the inner block tail; let the common
    // infrastructure handle it generically.
    ctx.zero_pad_output(DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM).data_type();
    const auto dst_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO).data_type();
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const float beta = pd()->beta();

    const dims_t &dims = blocked_d.dims();
    const dims_t &pdims = blocked_d.padded_dims();
    const dim_t *flat_strides = flat_d.blocking_desc().strides;
    const dim_t *blkd_strides = blk.strides;

    // The innermost logical dim is non-blocked and iterated as the contiguous
    // inner loop; the block positions are iterated inside it.
    const dim_t L = dims[last];
    const dim_t l_flat_stride = flat_strides[last];
    const dim_t l_blk_stride = blkd_strides[last];
    const dim_t d0_flat_stride = flat_strides[d0];
    const dim_t d1_flat_stride = d1 >= 0 ? flat_strides[d1] : 0;

    // Outer iteration space: every dim but the innermost one. Blocked dims
    // iterate over their outer blocks; the rest over their full extent.
    dims_t outer_dims = {0};
    dim_t outer_work = 1;
    for (int d = 0; d < last; ++d) {
        if (d == d0)
            outer_dims[d] = pdims[d] / bs0;
        else if (d == d1)
            outer_dims[d] = pdims[d] / bs1;
        else
            outer_dims[d] = dims[d];
        outer_work *= outer_dims[d];
    }
    if (outer_work * L == 0) return status::success;

    // Offsets below are computed from strides only, so each base is advanced to
    // its own logical element 0 here.
    input += input_d.offset0() * input_d.data_type_size();
    output += output_d.offset0() * output_d.data_type_size();

    parallel(0, [=](const int ithr, const int nthr) {
        const float src_scale = with_src_scales
                ? io::load_float_value(src_scales_dt, src_scales, 0)
                : 1.f;
        const float dst_scale = with_dst_scales
                ? io::load_float_value(dst_scales_dt, dst_scales, 0)
                : 1.f;
        const float alpha = src_scale / dst_scale;
        // Folding `1 / dst_scale` into `alpha` requires scaling the sum term
        // the same way, since `(src_scale * src + beta * dst) / dst_scale !=
        // alpha * src + beta * dst`.
        const float beta_scaled = with_dst_scales ? beta / dst_scale : beta;

        auto convert = [&](dim_t in_off, dim_t out_off) {
            float d = alpha * io::load_float_value(src_dt, input, in_off);
            if (beta_scaled != 0.f)
                d += beta_scaled
                        * io::load_float_value(dst_dt, output, out_off);
            io::store_float_value(dst_dt, d, output, out_off);
        };

        // With two blocked dims the within-block offset is a nested function of
        // the local positions `(p0, p1)` (it may interleave repeated blocks).
        // It is additive with the outer-block offset (both are terms of
        // `off_v`), so it is tabulated once over the whole block and reused,
        // keeping the hot loop free of the per-element modulo/division the inner
        // blocks require. The table is built per thread so its storage never
        // outlives the (possibly asynchronous) parallel region. A single
        // blocked dim is contiguous (`within(p0) == p0`), so no table is built
        // and the fast path indexes the block position directly.
        std::vector<dim_t> within_tbl;
        if (d1 >= 0) {
            within_tbl.resize((size_t)bs0 * bs1);
            for_(dim_t p0 = 0; p0 < bs0; ++p0)
            for (dim_t p1 = 0; p1 < bs1; ++p1) {
                dim_t lpos[DNNL_MAX_NDIMS] = {0};
                lpos[d0] = p0;
                lpos[d1] = p1;
                dim_t off = 0, stride = 1;
                for (int i = nblks - 1; i >= 0; --i) {
                    const dim_t d = in_idxs[i];
                    const dim_t b = in_blks[i];
                    off += (lpos[d] % b) * stride;
                    lpos[d] /= b;
                    stride *= b;
                }
                within_tbl[(size_t)p0 * bs1 + p1] = off;
            }
        }

        dim_t start {0}, end {0};
        balance211(outer_work, nthr, ithr, start, end);
        for (dim_t w = start; w < end; ++w) {
            // Decode the flat outer index into per-dimension coordinates.
            dims_t cur = {0};
            dim_t rem = w;
            for (int d = last - 1; d >= 0; --d) {
                cur[d] = rem % outer_dims[d];
                rem /= outer_dims[d];
            }

            dim_t base_flat = 0, base_blk = 0;
            for (int d = 0; d < last; ++d) {
                dim_t flat_coord;
                if (d == d0)
                    flat_coord = cur[d] * bs0;
                else if (d == d1)
                    flat_coord = cur[d] * bs1;
                else
                    flat_coord = cur[d];
                base_flat += flat_coord * flat_strides[d];
                base_blk += cur[d] * blkd_strides[d];
            }
            // Trailing blocks may be partial when a blocked dim is padded.
            const dim_t block0
                    = nstl::min<dim_t>(bs0, dims[d0] - cur[d0] * bs0);
            const dim_t block1 = d1 >= 0
                    ? nstl::min<dim_t>(bs1, dims[d1] - cur[d1] * bs1)
                    : 1;

            for (dim_t l = 0; l < L; ++l) {
                const dim_t flat_l = base_flat + l * l_flat_stride;
                const dim_t blk_l = base_blk + l * l_blk_stride;
                for_(dim_t p0 = 0; p0 < block0; ++p0)
                for (dim_t p1 = 0; p1 < block1; ++p1) {
                    const dim_t flat_off = flat_l + p0 * d0_flat_stride
                            + p1 * d1_flat_stride;
                    const dim_t blkd_off = blk_l
                            + (d1 >= 0 ? within_tbl[(size_t)p0 * bs1 + p1]
                                       : p0);
                    if (to_blocked)
                        convert(flat_off, blkd_off);
                    else
                        convert(blkd_off, flat_off);
                }
            }
        }
    });
    return status::success;
}

// TODO: remove this kernel, along with its `is_*` counterpart and the dispatch
// wiring in `init`/`execute`, once blocked activation layouts are dropped from
// the library.
status_t simple_reorder_t::execute_blocked_to_blocked(
        const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const auto dst_dt = output_d.data_type();

    const auto &iblk = input_d.blocking_desc();
    const auto &oblk = output_d.blocking_desc();
    // Both sides block the same single logical dim (channels) with different
    // block sizes; everything else shares the same logical iteration space.
    const dim_t cdim = iblk.inner_idxs[0];
    const dim_t bs_src = iblk.inner_blks[0];
    const dim_t bs_dst = oblk.inner_blks[0];
    const int ndims = input_d.ndims();

    // The blocked dst holds zero padding in the channel block tail; let the
    // common infrastructure handle it generically.
    ctx.zero_pad_output(DNNL_ARG_TO);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM).data_type();
    const auto dst_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO).data_type();
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const float beta = pd()->beta();

    const dims_t &dims = input_d.dims();
    const dim_t *src_strides = iblk.strides;
    const dim_t *dst_strides = oblk.strides;
    const dim_t C = dims[cdim];
    const dim_t src_cstride = src_strides[cdim];
    const dim_t dst_cstride = dst_strides[cdim];

    // Outer iteration space: every dim but the channel one (set to extent 1 so
    // it collapses in the mixed-radix decode). The channel dim is iterated as
    // the contiguous inner loop.
    dims_t outer_dims = {0};
    dim_t outer_work = 1;
    for (int d = 0; d < ndims; ++d) {
        outer_dims[d] = d == cdim ? 1 : dims[d];
        outer_work *= outer_dims[d];
    }
    if (outer_work * C == 0) return status::success;

    // Offsets below are computed from strides only, so each base is advanced to
    // its own logical element 0 here.
    input += input_d.offset0() * input_d.data_type_size();
    output += output_d.offset0() * output_d.data_type_size();

    parallel(0, [=](const int ithr, const int nthr) {
        const float src_scale = with_src_scales
                ? io::load_float_value(src_scales_dt, src_scales, 0)
                : 1.f;
        const float dst_scale = with_dst_scales
                ? io::load_float_value(dst_scales_dt, dst_scales, 0)
                : 1.f;
        const float alpha = src_scale / dst_scale;
        // Folding `1 / dst_scale` into `alpha` requires scaling the sum term
        // the same way, since `(src_scale * src + beta * dst) / dst_scale !=
        // alpha * src + beta * dst`.
        const float beta_scaled = with_dst_scales ? beta / dst_scale : beta;

        auto convert = [&](dim_t in_off, dim_t out_off) {
            float d = alpha * io::load_float_value(src_dt, input, in_off);
            if (beta_scaled != 0.f)
                d += beta_scaled
                        * io::load_float_value(dst_dt, output, out_off);
            io::store_float_value(dst_dt, d, output, out_off);
        };

        dim_t start {0}, end {0};
        balance211(outer_work, nthr, ithr, start, end);
        for (dim_t w = start; w < end; ++w) {
            // Decode the flat outer index into per-dimension coordinates.
            dims_t cur = {0};
            dim_t rem = w;
            for (int d = ndims - 1; d >= 0; --d) {
                cur[d] = rem % outer_dims[d];
                rem /= outer_dims[d];
            }

            // Both offsets decompose additively (`memory_desc_wrapper::off_v`):
            // the channel contribution `(c / bs) * cstride + (c % bs)` adds onto
            // the shared outer base.
            dim_t base_src = 0, base_dst = 0;
            for (int d = 0; d < ndims; ++d) {
                if (d == cdim) continue;
                base_src += cur[d] * src_strides[d];
                base_dst += cur[d] * dst_strides[d];
            }

            for (dim_t c = 0; c < C; ++c) {
                const dim_t src_off
                        = base_src + (c / bs_src) * src_cstride + (c % bs_src);
                const dim_t dst_off
                        = base_dst + (c / bs_dst) * dst_cstride + (c % bs_dst);
                convert(src_off, dst_off);
            }
        }
    });
    return status::success;
}

// TODO: once blocked convolution-weights layouts are reworked, revisit whether
// this compensation kernel can be folded back into `plain_to_blocked`.
namespace {
// Shared, derived state for one `execute_plain_to_dense_comp` invocation. Holds
// only information that is not directly available from the src/dst
// `memory_desc_wrapper`s (those are passed to the kernels separately). This
// struct is self-contained and captured *by value* into the compute closures:
// the fixed-size arrays are embedded (not pointers to caller-stack buffers), so
// a copy survives an asynchronous threadpool run that executes after
// `execute()` returns. The only external storage it points to is the
// `in_tab`/`out_tab` offset tables, which live in the scratchpad (kept alive by
// the execution context) rather than on the stack.
struct plain_to_dense_comp_ctx_t {
    const char *input;
    char *output;
    int32_t *cp; // s8s8 compensation buffer (may be null)
    int32_t *zp; // asymmetric zero-point compensation buffer (may be null)

    // Masked (compensated) dims index the comp buffer.
    int n_comp;
    int comp_idx_buf[DNNL_MAX_NDIMS];
    dim_t comp_sz[DNNL_MAX_NDIMS]; // padded sizes
    dim_t comp_work;
    // Reduction dims are summed per masked coordinate.
    int n_red;
    int red_idx[DNNL_MAX_NDIMS];
    dim_t red_sz[DNNL_MAX_NDIMS]; // real sizes
    dim_t red_work;

    // Blocked comp-buffer strides, indexed by real dim.
    dim_t comp_blk[DNNL_MAX_NDIMS];
    dim_t comp_inner_str[DNNL_MAX_NDIMS];
    dim_t comp_outer_str[DNNL_MAX_NDIMS];

    // Per-dim offset tables, indexed `[dim][index]` with index in [0, dims[d]).
    // `off_v` is separable across dims, so a position's offset is the sum of
    // per-dim contributions; these tables cache them so the hot loop walks
    // constant strides instead of recomputing `off_v` per element. The pointed
    // storage lives in the scratchpad (see `execute_plain_to_dense_comp`).
    const dim_t *in_tab[DNNL_MAX_NDIMS];
    const dim_t *out_tab[DNNL_MAX_NDIMS];

    bool with_src_scales;
    bool with_dst_scales;
    const void *src_scales;
    const void *dst_scales;
    data_type_t src_scales_dt;
    data_type_t dst_scales_dt;
    int src_scales_mask;
    int dst_scales_mask;
    // True when no reduction dim is masked by any scale, so the scale factor is
    // constant across the reduction and can be hoisted out of the hot loop.
    bool scales_hoistable;
    float adj_scale;
    // When >= 0, the stride-1 (contiguous) input dim is a compensated dim.
    // Streaming the reduction per masked coordinate would then stride through
    // memory, so the cache-blocked kernel streams this dim in the inner loop
    // instead. -1 selects the reduction-inner kernel.
    int inner_comp_dim;
};

// Typed source load. Specialized on the source data type so the conversion
// inlines and the hot loop can vectorize. Byte-and-wider types index a typed
// pointer directly; everything else (including sub-byte packed types) uses the
// generic, correctness-first `undef` fallback below.
template <data_type_t src_dt>
inline float load_src_val(
        const plain_to_dense_comp_ctx_t &c, data_type_t, dim_t off) {
    using T = typename prec_traits_t<src_dt>::type;
    return static_cast<float>(reinterpret_cast<const T *>(c.input)[off]);
}
template <>
inline float load_src_val<data_type::undef>(
        const plain_to_dense_comp_ctx_t &c, data_type_t src_dt, dim_t off) {
    return io::load_float_value(src_dt, c.input, off);
}

// Row-major scale index for `pos` over a mask, across the actual dims.
inline dim_t scale_off(
        int ndims, const dims_t &dims, const dims_t &pos, int mask) {
    dim_t off = 0;
    for (int d = 0; d < ndims; ++d)
        if ((mask >> d) & 1) off = off * dims[d] + pos[d];
    return off;
}

// Combined src/dst scale factor for `pos`, starting from `adj_scale`.
inline float scale_factor(const plain_to_dense_comp_ctx_t &c, int ndims,
        const dims_t &dims, const dims_t &pos) {
    float alpha = c.adj_scale;
    if (c.with_src_scales)
        alpha *= io::load_float_value(c.src_scales_dt, c.src_scales,
                scale_off(ndims, dims, pos, c.src_scales_mask));
    if (c.with_dst_scales)
        alpha /= io::load_float_value(c.dst_scales_dt, c.dst_scales,
                scale_off(ndims, dims, pos, c.dst_scales_mask));
    return alpha;
}

// Advance the reduction odometer by one element, keeping the running
// `in_red`/`out_red` offsets in sync. For each digit (fastest first) subtract
// its current contribution unconditionally -- the old `pos[d]` entry must leave
// whether this digit increments or overflows and carries -- then add the new
// contribution and stop only when it stays in range; otherwise wrap `pos[d]` to
// 0, whose table offset is 0 (nothing to add), and carry into the next digit.
inline void advance_reduction(const plain_to_dense_comp_ctx_t &c, dims_t &pos,
        dim_t &in_red, dim_t &out_red) {
    for (int i = c.n_red - 1; i >= 0; --i) {
        const int d = c.red_idx[i];
        in_red -= c.in_tab[d][pos[d]];
        out_red -= c.out_tab[d][pos[d]];
        if (++pos[d] < c.red_sz[i]) {
            in_red += c.in_tab[d][pos[d]];
            out_red += c.out_tab[d][pos[d]];
            return;
        }
        pos[d] = 0;
    }
}

// Reduction-inner compensation kernel.
//
// Compensation (s8s8 and/or zero-point) needs one correction value per
// "compensated" (masked) coordinate, formed by summing the quantized source
// over the remaining "reduction" dims:
//
//     comp[masked] = -128 * sum_over_reduction( q(alpha * src) )   (s8s8)
//     comp[masked] =        -sum_over_reduction( q(alpha * src) )  (zero-point)
//
// The dims therefore split into two disjoint groups:
//   * compensated dims (`comp_idx_buf`) -- one output comp slot each; the outer
//     loop walks `comp_work` of them and is parallelized across threads.
//   * reduction dims (`red_idx`) -- summed away; the inner loop walks `red_work`
//     of them, accumulating `acc` and writing every quantized dst element once.
//
// A layout offset is separable (a sum of independent per-dim strides), so any
// element's src/dst offset is `comp_part + red_part`. The masked part is fixed
// per outer iteration; the reduction part is enumerated with an odometer: `pos`
// over the reduction dims is a mixed-radix counter, and running `in_red`/
// `out_red` offsets are kept in sync via precomputed per-dim offset tables
// (`in_tab`/`out_tab`), so advancing one element is a table swap rather than a
// full `off_v` recompute (see `advance_reduction`). Enumeration order is
// irrelevant: `acc` is a sum and each dst element is written exactly once.
// (When the stride-1 src dim is itself compensated, the cache-blocked sibling
// kernel below is used instead.)
template <data_type_t src_dt>
void run_plain_to_dense_comp(const plain_to_dense_comp_ctx_t &c,
        const memory_desc_wrapper &src_mdw,
        const memory_desc_wrapper &dst_mdw) {
    const int ndims = dst_mdw.ndims();
    const dims_t &dims = dst_mdw.dims(); // real (non-padded) dims
    const data_type_t src_dt_rt = src_mdw.data_type();
    parallel(0, [=](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(c.comp_work, nthr, ithr, start, end);
        int8_t *const out8 = reinterpret_cast<int8_t *>(c.output);
        dims_t pos = {0};
        for (dim_t comp_idx = start; comp_idx < end; ++comp_idx) {
            // Decode the masked coordinates (padded) from the comp index.
            dim_t linear_idx = comp_idx;
            bool is_padded = false;
            for (int i = c.n_comp - 1; i >= 0; --i) {
                const int d = c.comp_idx_buf[i];
                const dim_t cc = linear_idx % c.comp_sz[i];
                linear_idx /= c.comp_sz[i];
                pos[d] = cc;
                if (cc >= dims[d]) is_padded = true;
            }
            // A padded channel contributes nothing; its comp stays zero (the
            // buffer was pre-zeroed) and its blocked slot is never read.
            if (is_padded) continue;

            // Blocked comp-buffer offset and the masked-dim part of the running
            // src/dst offsets for these coordinates.
            dim_t comp_off = 0, in_comp = 0, out_comp = 0;
            for (int i = 0; i < c.n_comp; ++i) {
                const int d = c.comp_idx_buf[i];
                comp_off += (pos[d] / c.comp_blk[d]) * c.comp_outer_str[d]
                        + (pos[d] % c.comp_blk[d]) * c.comp_inner_str[d];
                in_comp += c.in_tab[d][pos[d]];
                out_comp += c.out_tab[d][pos[d]];
            }

            // When no reduction dim is masked the scale is constant across the
            // reduction; compute it once per masked coordinate.
            const float alpha0 = c.scales_hoistable
                    ? scale_factor(c, ndims, dims, pos)
                    : c.adj_scale;

            int32_t acc = 0;
            // Enumerate the reduction with an odometer (see `advance_reduction`):
            // maintain running src/dst offsets via table lookups instead of
            // recomputing `off_v` per element. Enumeration order is irrelevant:
            // `acc` is order-independent and every output element is written
            // exactly once.
            for (int i = 0; i < c.n_red; ++i)
                pos[c.red_idx[i]] = 0;
            dim_t in_red = 0, out_red = 0;
            for (dim_t ri = 0; ri < c.red_work; ++ri) {
                const float alpha = c.scales_hoistable
                        ? alpha0
                        : scale_factor(c, ndims, dims, pos);
                const float v = alpha
                        * load_src_val<src_dt>(c, src_dt_rt, in_comp + in_red);
                // `dst_dt` is pinned to s8 (see `is_plain_to_dense_comp`), so
                // quantize once in-register, store the byte, and accumulate the
                // same value instead of storing and reloading it from memory.
                const int8_t q = q10n::saturate_and_round<int8_t>(v);
                out8[out_comp + out_red] = q;
                acc += q;

                advance_reduction(c, pos, in_red, out_red);
            }
            // Zero-point compensation is -sum(q); s8s8 compensation is
            // -128 * sum(q).
            if (c.zp) c.zp[comp_off] = -acc;
            if (c.cp) c.cp[comp_off] = -128 * acc;
        }
    });
}

// Cache-blocked variant for when the contiguous (stride-1) input dim is a
// compensated dim (e.g. int8 matmul weights `ab -> BA16a64b4a`): the reduction
// would otherwise stride through memory for each masked coordinate. Instead
// stream a tile of the contiguous masked dim in the inner loop (sequential
// reads) and keep one accumulator per masked coordinate in the tile.
// Tile width for streaming the contiguous compensated dim in the tiled kernel.
// Bounds the per-thread stack arrays (`alpha_tile`/`acc`) and the number of
// live accumulators; 64 balances cache footprint against inner-loop length.
// Namespace-scoped so the compute closure uses it as a true compile-time
// constant without capturing it.
constexpr dim_t block_columns = 64;

template <data_type_t src_dt>
void run_plain_to_dense_comp_tiled(const plain_to_dense_comp_ctx_t &c,
        const memory_desc_wrapper &src_mdw,
        const memory_desc_wrapper &dst_mdw) {
    const int ndims = dst_mdw.ndims();
    const dims_t &dims = dst_mdw.dims(); // real (non-padded) dims
    const data_type_t src_dt_rt = src_mdw.data_type();
    const int icd = c.inner_comp_dim;
    const dim_t icd_sz = dims[icd];
    const dim_t n_tiles = (icd_sz + block_columns - 1) / block_columns;

    // Masked dims other than the streamed one form the outer (parallel) work.
    int outer_idx[DNNL_MAX_NDIMS];
    dim_t outer_sz[DNNL_MAX_NDIMS];
    int n_outer = 0;
    dim_t outer_work = 1;
    for (int i = 0; i < c.n_comp; ++i) {
        if (c.comp_idx_buf[i] == icd) continue;
        outer_idx[n_outer] = c.comp_idx_buf[i];
        outer_sz[n_outer] = c.comp_sz[i];
        outer_work *= c.comp_sz[i];
        ++n_outer;
    }
    const dim_t work = outer_work * n_tiles;

    parallel(0, [=](const int ithr, const int nthr) {
        dim_t start {0}, end {0};
        balance211(work, nthr, ithr, start, end);
        int8_t *const out8 = reinterpret_cast<int8_t *>(c.output);
        dims_t pos = {0};
        float alpha_tile[block_columns];
        int32_t acc[block_columns];
        for (dim_t w = start; w < end; ++w) {
            dim_t rem = w;
            const dim_t tile = rem % n_tiles;
            rem /= n_tiles;
            // Decode the outer masked coordinates (padded); skip if padded.
            bool is_padded = false;
            for (int i = n_outer - 1; i >= 0; --i) {
                const int d = outer_idx[i];
                const dim_t cc = rem % outer_sz[i];
                rem /= outer_sz[i];
                pos[d] = cc;
                if (cc >= dims[d]) is_padded = true;
            }
            if (is_padded) continue;

            const dim_t j0 = tile * block_columns;
            dim_t jend = icd_sz - j0;
            if (jend > block_columns) jend = block_columns;

            // Offsets contributed by the outer masked coordinates.
            dim_t in_base = 0, out_base = 0, comp_base = 0;
            for (int i = 0; i < n_outer; ++i) {
                const int d = outer_idx[i];
                in_base += c.in_tab[d][pos[d]];
                out_base += c.out_tab[d][pos[d]];
                comp_base += (pos[d] / c.comp_blk[d]) * c.comp_outer_str[d]
                        + (pos[d] % c.comp_blk[d]) * c.comp_inner_str[d];
            }

            for (dim_t j = 0; j < jend; ++j)
                acc[j] = 0;
            // Scales that don't depend on a reduction dim are constant across
            // the reduction; precompute one factor per tile column.
            if (c.scales_hoistable) {
                for (dim_t j = 0; j < jend; ++j) {
                    pos[icd] = j0 + j;
                    alpha_tile[j] = scale_factor(c, ndims, dims, pos);
                }
            }

            // Reduction is the outer loop; the inner loop streams the
            // contiguous masked dim with running reduction offsets.
            for (int i = 0; i < c.n_red; ++i)
                pos[c.red_idx[i]] = 0;
            dim_t in_red = 0, out_red = 0;
            for (dim_t ri = 0; ri < c.red_work; ++ri) {
                const dim_t in_r = in_base + in_red;
                const dim_t out_r = out_base + out_red;
                for (dim_t j = 0; j < jend; ++j) {
                    const dim_t col = j0 + j;
                    float alpha;
                    if (c.scales_hoistable) {
                        alpha = alpha_tile[j];
                    } else {
                        pos[icd] = col;
                        alpha = scale_factor(c, ndims, dims, pos);
                    }
                    const float v = alpha
                            * load_src_val<src_dt>(
                                    c, src_dt_rt, in_r + c.in_tab[icd][col]);
                    const int8_t q = q10n::saturate_and_round<int8_t>(v);
                    out8[out_r + c.out_tab[icd][col]] = q;
                    acc[j] += q;
                }
                advance_reduction(c, pos, in_red, out_red);
            }

            for (dim_t j = 0; j < jend; ++j) {
                const dim_t col = j0 + j;
                const dim_t comp_off = comp_base
                        + (col / c.comp_blk[icd]) * c.comp_outer_str[icd]
                        + (col % c.comp_blk[icd]) * c.comp_inner_str[icd];
                if (c.zp) c.zp[comp_off] = -acc[j];
                if (c.cp) c.cp[comp_off] = -128 * acc[j];
            }
        }
    });
}
} // namespace

status_t simple_reorder_t::execute_plain_to_dense_comp(
        const exec_ctx_t &ctx) const {
    auto input = CTX_IN_MEM(const char *, DNNL_ARG_FROM);
    auto output = CTX_OUT_MEM(char *, DNNL_ARG_TO);
    const auto input_d = ctx.memory_mdw(DNNL_ARG_FROM, pd()->src_md());
    const auto output_d = ctx.memory_mdw(DNNL_ARG_TO, pd()->dst_md());
    const auto src_dt = input_d.data_type();
    const int ndims = output_d.ndims();

    const auto &extra = output_d.extra();
    const bool req_s8s8
            = extra.flags & memory_extra_flags::compensation_conv_s8s8;
    const bool req_asymm = extra.flags
            & memory_extra_flags::compensation_conv_asymmetric_src;
    const int comp_mask = req_s8s8 ? extra.compensation_mask
                                   : extra.asymm_compensation_mask;
    // The s8s8 compensation absorbs an extra `* 128` factor (see below); both
    // buffers share the same per-channel accumulator otherwise.
    const float adj_scale = (extra.flags & memory_extra_flags::scale_adjust)
            ? extra.scale_adjust
            : 1.f;

    // A zero-dim memory has no storage at all, compensation buffers included
    // (`size()` is 0), while `additional_buffer_size()` still counts the masked
    // dims, so the offsets below would point outside of the allocation.
    if (output_d.has_zero_dim()) return status::success;

    // Compensation buffer(s) are appended right after the main data blob; the
    // s8s8 buffer (when present) precedes the zero-point one.
    const size_t comp_base_off
            = output_d.size() - output_d.additional_buffer_size();
    const size_t s8s8_sz = output_d.additional_buffer_size(
            memory_extra_flags::compensation_conv_s8s8);
    int32_t *cp = req_s8s8 ? reinterpret_cast<int32_t *>(output + comp_base_off)
                           : nullptr;
    int32_t *zp = req_asymm ? reinterpret_cast<int32_t *>(output + comp_base_off
                                      + (req_s8s8 ? s8s8_sz : 0))
                            : nullptr;

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO);
    const auto src_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_FROM).data_type();
    const auto dst_scales_dt
            = ctx.memory_mdw(DNNL_ARG_ATTR_SCALES | DNNL_ARG_TO).data_type();
    const auto &scales = pd()->attr()->scales_;
    const bool with_src_scales = !scales.has_default_values(DNNL_ARG_FROM);
    const bool with_dst_scales = !scales.has_default_values(DNNL_ARG_TO);
    const int src_scales_mask
            = with_src_scales ? scales.get_mask(DNNL_ARG_FROM) : 0;
    const int dst_scales_mask
            = with_dst_scales ? scales.get_mask(DNNL_ARG_TO) : 0;

    const dims_t &dims = output_d.dims();
    const dims_t &pdims = output_d.padded_dims();

    // Split the logical dims into the compensated (masked) set, which indexes
    // the comp buffer, and the reduction set. The outer loop enumerates every
    // padded masked coordinate (`comp_work` slots == the comp buffer size).
    int comp_idx[DNNL_MAX_NDIMS], red_idx[DNNL_MAX_NDIMS];
    dim_t comp_sz[DNNL_MAX_NDIMS], red_sz[DNNL_MAX_NDIMS];
    int n_comp = 0, n_red = 0;
    dim_t comp_work = 1, red_work = 1;
    for (int d = 0; d < ndims; ++d) {
        if ((comp_mask >> d) & 1) {
            comp_idx[n_comp] = d;
            comp_sz[n_comp] = pdims[d];
            comp_work *= pdims[d];
            ++n_comp;
        } else {
            red_idx[n_red] = d;
            red_sz[n_red] = dims[d];
            red_work *= dims[d];
            ++n_red;
        }
    }

    // The comp buffer is laid out as the dst tag trimmed to the masked dims,
    // i.e. a *blocked* layout, not a flat row-major one: a masked dim that is
    // blocked in the dst keeps its inner block in the comp buffer too.
    // Row-major indexing is only correct when the blocked masked dim is the
    // innermost masked dim; when an outer masked dim is blocked (e.g. `Abx16a`
    // with mask over dims 0 and 1) the inner block is interleaved after the
    // trailing masked dims and the offsets differ. Build per-masked-dim blocked
    // strides so the comp offset matches the library convention. The `is_`
    // guard guarantees each masked dim is blocked at most once, so a dim
    // contributes a single (outer, inner) stride pair. For a plain dst no
    // masked dim is blocked, so this degenerates to a plain row-major layout.
    dim_t comp_blk[DNNL_MAX_NDIMS], comp_inner_str[DNNL_MAX_NDIMS];
    dim_t comp_outer_str[DNNL_MAX_NDIMS];
    for (int d = 0; d < ndims; ++d) {
        comp_blk[d] = 1;
        comp_inner_str[d] = 0;
    }
    const auto &oblk = output_d.blocking_desc();
    dim_t inner_tot = 1;
    for (int i = oblk.inner_nblks - 1; i >= 0; --i) {
        const dim_t d = oblk.inner_idxs[i];
        if (!((comp_mask >> d) & 1)) continue;
        comp_inner_str[d] = inner_tot;
        comp_blk[d] = oblk.inner_blks[i];
        inner_tot *= oblk.inner_blks[i];
    }
    dim_t outer_acc = inner_tot;
    for (int i = n_comp - 1; i >= 0; --i) {
        const int d = comp_idx[i];
        comp_outer_str[d] = outer_acc;
        outer_acc *= pdims[d] / comp_blk[d];
    }

    // Per-dim offset tables: `off_v` is separable across dims, so a position's
    // offset is the sum of per-dim contributions. Cache every dim's
    // contribution for all of its indices once (built via `off_v` on
    // single-dim positions, so arbitrary blocking/padding is handled exactly);
    // the reduction loop then walks these tables instead of recomputing
    // `off_v` per element. The tables live in the scratchpad so they outlive
    // `execute()` under an asynchronous threadpool; only the per-dim slice
    // pointers are set up here, the fill is submitted as a task below.
    dim_t tab_total = 0;
    for (int d = 0; d < ndims; ++d)
        tab_total += dims[d];
    const auto &scratchpad = ctx.get_scratchpad_grantor();
    dim_t *const tab_pool = scratchpad.template get<dim_t>(
            memory_tracking::names::key_reorder_space);
    dim_t *const in_tab_pool = tab_pool;
    dim_t *const out_tab_pool = tab_pool + tab_total;
    const dim_t *in_tab[DNNL_MAX_NDIMS], *out_tab[DNNL_MAX_NDIMS];
    {
        dim_t base = 0;
        for (int d = 0; d < ndims; ++d) {
            in_tab[d] = in_tab_pool + base;
            out_tab[d] = out_tab_pool + base;
            base += dims[d];
        }
    }

    // Scales can be hoisted out of the reduction unless a scale masks a
    // reduction dim (i.e. its value varies within the reduction).
    const int all_mask = (ndims >= 31) ? -1 : ((1 << ndims) - 1);
    const int red_mask = (~comp_mask) & all_mask;
    const bool scales_hoistable
            = ((src_scales_mask | dst_scales_mask) & red_mask) == 0;

    // The blocked dst carries zero padding in the inner block tails; let the
    // common infrastructure zero it so the kernel only writes real elements.
    ctx.zero_pad_output(DNNL_ARG_TO);

    parallel(1, [=](const int ithr, const int nthr) {
        if (cp) std::memset(cp, 0, comp_work * sizeof(int32_t));
        if (zp) std::memset(zp, 0, comp_work * sizeof(int32_t));

        dim_t base = 0;
        dims_t upos = {0};
        for (int d = 0; d < ndims; ++d) {
            for (int dd = 0; dd < ndims; ++dd)
                upos[dd] = 0;
            for (dim_t x = 0; x < dims[d]; ++x) {
                upos[d] = x;
                in_tab_pool[base + x] = input_d.off_v(upos);
                out_tab_pool[base + x] = output_d.off_v(upos);
            }
            base += dims[d];
        }
    });

    plain_to_dense_comp_ctx_t c;
    c.input = input;
    c.output = output;
    c.cp = cp;
    c.zp = zp;
    c.n_comp = n_comp;
    std::memcpy(c.comp_idx_buf, comp_idx, sizeof(c.comp_idx_buf));
    std::memcpy(c.comp_sz, comp_sz, sizeof(c.comp_sz));
    c.comp_work = comp_work;
    c.n_red = n_red;
    std::memcpy(c.red_idx, red_idx, sizeof(c.red_idx));
    std::memcpy(c.red_sz, red_sz, sizeof(c.red_sz));
    c.red_work = red_work;
    std::memcpy(c.comp_blk, comp_blk, sizeof(c.comp_blk));
    std::memcpy(c.comp_inner_str, comp_inner_str, sizeof(c.comp_inner_str));
    std::memcpy(c.comp_outer_str, comp_outer_str, sizeof(c.comp_outer_str));
    std::memcpy(c.in_tab, in_tab, sizeof(c.in_tab));
    std::memcpy(c.out_tab, out_tab, sizeof(c.out_tab));
    c.with_src_scales = with_src_scales;
    c.with_dst_scales = with_dst_scales;
    c.src_scales = src_scales;
    c.dst_scales = dst_scales;
    c.src_scales_dt = src_scales_dt;
    c.dst_scales_dt = dst_scales_dt;
    c.src_scales_mask = src_scales_mask;
    c.dst_scales_mask = dst_scales_mask;
    c.scales_hoistable = scales_hoistable;
    c.adj_scale = adj_scale;

    // Pick the cache-blocked kernel when the contiguous (stride-1) input dim is
    // a compensated dim with more than one element: its reduction would stride
    // through memory otherwise. Any other case (notably conv weights, whose
    // contiguous dim is a reduction dim) streams fine with the default kernel.
    const auto &istrides = input_d.blocking_desc().strides;
    int inner_comp_dim = -1;
    for (int i = 0; i < n_comp; ++i) {
        const int d = comp_idx[i];
        if (istrides[d] == 1 && dims[d] > 1) {
            inner_comp_dim = d;
            break;
        }
    }
    c.inner_comp_dim = inner_comp_dim;

    using namespace data_type;
#define DISPATCH_COMP(dt) \
    do { \
        if (inner_comp_dim >= 0) \
            run_plain_to_dense_comp_tiled<dt>(c, input_d, output_d); \
        else \
            run_plain_to_dense_comp<dt>(c, input_d, output_d); \
    } while (0)
    switch (src_dt) {
        case f32: DISPATCH_COMP(f32); break;
        case bf16: DISPATCH_COMP(bf16); break;
        case f16: DISPATCH_COMP(f16); break;
        case s8: DISPATCH_COMP(s8); break;
        case u8: DISPATCH_COMP(u8); break;
        case s32: DISPATCH_COMP(s32); break;
        case f8_e5m2: DISPATCH_COMP(f8_e5m2); break;
        case f8_e4m3: DISPATCH_COMP(f8_e4m3); break;
        default: DISPATCH_COMP(undef); break;
    }
#undef DISPATCH_COMP
    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
