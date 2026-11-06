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

#include "common/c_types_map.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/reorder/config.hpp"
#include "gpu/intel/reorder/custom.hpp"

#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {

using namespace dnnl::impl::memory_tracking::names;

using dimension = struct {
    dim_t size;
    dim_idx_t idx;
};

struct stride_t {
    dim_t stride = 1;
    dim_t size = 1;
    dim_idx_t idx = 0;
};

// Stride sorter. Smaller stride = inner dim, bigger stride = outer dim.
// Dimensions of size 1 are considered outermost regardless of strides and
// they are sorted by index.
bool stride_cmp(const stride_t &a, const stride_t &b) {
    if (a.size == 1 && b.size == 1) {
        return a.idx > b.idx;
    } else if (a.size != 1 && b.size == 1) {
        return true;
    } else if (a.size == 1 && b.size != 1) {
        return false;
    } else {
        return a.stride < b.stride;
    }
}

// Returns size and index of dimension or block that's last or at given
// distance from end. Blocks, if exist, take precedence before dimensions.
// Order of dimensions is determined by sorting strides; smallest stride is
// last dimension. Dimensions of size 1 are 'ignored' (treated as first).
// Due to ignoring dims of size 1 and treating them as outermost regardless of
// their original position in tensor tag, now it is possible to arrive at
// illegal tensor tag where last dim is the same as first block:
// 8x8x1x1 aBcd8b becomes cdaB8b (notice the "B8b").
// In such cases this function combines last dim with first block (xB8b := xb)
dimension get_Nth_last_dim_or_block(
        const memory_desc_wrapper &md, dim_idx_t distance = 0) {
    dim_idx_t nblks = into<dim_idx_t>(md.blocking_desc().inner_nblks);
    dimension ret;
    dim_idx_t ndims = into<dim_idx_t>(md.ndims());

    std::vector<stride_t> strides(ndims);
    for (dim_idx_t d = 0; d < ndims; ++d) {
        strides[d].idx = d;
        strides[d].stride = md.blocking_desc().strides[d];
        strides[d].size = md.padded_dims()[d];
    }
    std::sort(strides.begin(), strides.end(), stride_cmp);
    for (dim_idx_t i = 0; i < nblks; i++) {
        stride_t blk;
        blk.idx = into<dim_idx_t>(md.blocking_desc().inner_idxs[i]);
        blk.size = md.blocking_desc().inner_blks[i];
        if (i == 0 && blk.idx == strides[0].idx) { continue; }
        strides.insert(strides.begin(), blk);
    }
    ret.idx = strides[distance].idx;
    ret.size = strides[distance].size;
    return ret;
}

int innermost_block(const blocking_desc_t &blk) {
    int last = blk.inner_nblks - 1;
    return into<int>(blk.inner_blks[last]);
}

bool is_alt_faster_than_ref(const memory_desc_wrapper &src_mdw,
        const memory_desc_wrapper &dst_mdw,
        const compute::device_info_t *dev_info) {
    return false; // alt > ref is no more the case after gen9
}

bool matches_one_NxN_layout(const memory_desc_wrapper &src,
        const memory_desc_wrapper &dst, int n, int scale_mask) {
    if (dst.ndims() < 2) { return false; }
    if (!src.is_blocking_desc() || !dst.is_blocking_desc()) { return false; }
    auto dst_last = get_Nth_last_dim_or_block(dst, 0);
    auto src_last = get_Nth_last_dim_or_block(src, 0);

    if (dst_last.size % n != 0) { return false; }
    if (src_last.size % n != 0) { return false; }
    if (dst_last.idx == src_last.idx) { return false; }
    // no padding allowed on dimensions that are last in src or last in dst
    if (src.padded_dims()[src_last.idx] != src.dims()[src_last.idx]) {
        return false;
    }
    if (src.padded_dims()[dst_last.idx] != src.dims()[dst_last.idx]) {
        return false;
    }
    if (dst.padded_dims()[src_last.idx] != dst.dims()[src_last.idx]) {
        return false;
    }
    if (dst.padded_dims()[dst_last.idx] != dst.dims()[dst_last.idx]) {
        return false;
    }
    // no scale mask allowed on src's or dst's innermost dimension
    if (scale_mask & (1 << src_last.idx)) { return false; }
    if (scale_mask & (1 << dst_last.idx)) { return false; }

    return true;
}

// For cases like ab -> Ba4b, BA4b8a8b2a etc. Takes dst's last two dimensions
// and groups them together into packets of size 16 so that it can be written
// to in bursts.
bool fill_conf_xab_xba(const memory_desc_wrapper &src,
        const memory_desc_wrapper &dst, int scale_mask, xb_to_xab_xba_t &cfg,
        dim_idx_t &vect_dim, int &vect_size, dim_t *blocks) {

    vect_size = 16;
    if (dst.ndims() < 2) { return false; }
    dimension src_last = get_Nth_last_dim_or_block(src);
    dimension dst_last = get_Nth_last_dim_or_block(dst);
    dimension dst_next_last = get_Nth_last_dim_or_block(dst, 1);

    if (src_last.idx != dst_last.idx && src_last.idx != dst_next_last.idx) {
        return false;
    }
    bool xb_to_xab = (src_last.idx == dst_last.idx);

    // no padding on dims that are src's or dst's innermost
    if (src.padded_dims()[src_last.idx] != src.dims()[src_last.idx]) {
        return false;
    }
    if (src.padded_dims()[dst_last.idx] != src.dims()[dst_last.idx]) {
        return false;
    }
    if (dst.padded_dims()[src_last.idx] != dst.dims()[src_last.idx]) {
        return false;
    }
    if (dst.padded_dims()[dst_last.idx] != dst.dims()[dst_last.idx]) {
        return false;
    }
    if (src.offset0() != 0) { return false; }
    if (dst.offset0() != 0) { return false; }

    // no scale mask allowed on src's or dst's innermost dimension
    if (scale_mask & (1 << src_last.idx)) { return false; }
    if (scale_mask & (1 << dst_last.idx)) { return false; }

    if (src_last.size < 16) { return false; }
    if (src_last.size % 16 != 0) { return false; }
    if (vect_size % dst_last.size != 0) { return false; }
    if (dst_next_last.size % (vect_size / dst_last.size) != 0) { return false; }

    dimension src_burst;
    dimension dst_burst[2];
    dimension dst_loop;
    dimension src_loop;

    src_burst.idx = src_last.idx;
    src_burst.size = vect_size;
    dst_burst[0] = dst_last;
    dst_burst[1].idx = dst_next_last.idx;
    dst_burst[1].size = 16 / dst_last.size;

    int dst_src_idx = xb_to_xab ? 0 : 1;
    dst_loop.size = src_burst.size / dst_burst[dst_src_idx].size;
    dst_loop.idx = src_burst.idx;

    src_loop.size = dst_burst[1 - dst_src_idx].size;
    src_loop.idx = dst_burst[1 - dst_src_idx].idx;

    vect_dim = src_last.idx;

    if ((dst_last.size * dst_next_last.size) % vect_size != 0) { return false; }
    blocks[src_loop.idx] = src_loop.size;
    cfg.blk_size = src_loop.size;
    cfg.src_blk_dim = src_loop.idx;
    cfg.src_blk_coeff = 1;
    cfg.dst_blk_dim = dst_loop.idx;

    if (dst_loop.idx == dst_burst[0].idx) {
        cfg.dst_blk_coeff = dst_burst[0].size;
    } else if (dst_loop.idx == dst_burst[1].idx) {
        cfg.dst_blk_coeff = dst_burst[1].size;
    } else {
        cfg.dst_blk_coeff = 1;
    }
    cfg.vd = xb_to_xab;
    return true;
}

bool fits_xab_xba(const memory_desc_wrapper &src,
        const memory_desc_wrapper &dst, int scale_mask) {
    xb_to_xab_xba_t cfg;
    dim_idx_t vect_dim;
    int vect_size;
    dim_t blocks[6];

    return fill_conf_xab_xba(
            src, dst, scale_mask, cfg, vect_dim, vect_size, &blocks[0]);
}

bool matches_ABxxxx8ayb_layout(const blocking_desc_t &blk, int ndims) {
    if (ndims > 2 || blk.inner_nblks < 2) { return false; }
    int last = blk.inner_nblks - 1;
    // Don't allow this kernel when two adjacent blocks by b create
    // total block size smaller than 16 - in that situation macros
    // used for calculation of dst address return wrong values.
    for (int d = last - 2; d >= 0; d--) {
        if (blk.inner_idxs[d] == ndims - 1) {
            int double_block
                    = into<int>(blk.inner_blks[last] * blk.inner_blks[d]);
            if (double_block < 16) {
                return false;
            } else {
                break;
            }
        }
    }
    return ((blk.inner_blks[last] == 4 || blk.inner_blks[last] == 2)
            && blk.inner_idxs[last] == ndims - 1
            && blk.inner_blks[last - 1] == 8
            && blk.inner_idxs[last - 1] == ndims - 2);
}

bool dim_is_div_by_16_or_less_than_16(
        const memory_desc_wrapper &src, int dim_index) {
    const auto &padded_dims = src.padded_dims();
    assert(dim_index < src.ndims());
    return (padded_dims[dim_index] % 16 == 0 || padded_dims[dim_index] < 16);
}

bool is_broadcast_by_strides(const memory_desc_wrapper &mdw) {
    if (mdw.is_blocking_desc()) {
        for (int i = 0; i < mdw.ndims(); i++) {
            if (mdw.blocking_desc().strides[i] == 0) { return true; }
        }
    }
    return false;
}

bool is_padded(const memory_desc_wrapper &mdw, int dim) {
    return (mdw.dims()[dim] != mdw.padded_dims()[dim]);
}

bool fits_3ch(const memory_desc_wrapper &src_mdw,
        const memory_desc_wrapper &dst_mdw, int scale_mask) {
    // TODO: make it more generic, now it works only for dense->padded case
    if (src_mdw.ndims() < 2 || dst_mdw.ndims() < 2) { return false; }

    auto last_dim_src = get_Nth_last_dim_or_block(src_mdw);
    auto nextlast_dim_src = get_Nth_last_dim_or_block(src_mdw, 1);
    auto last_dim_dst = get_Nth_last_dim_or_block(dst_mdw);
    auto nextlast_dim_dst = get_Nth_last_dim_or_block(dst_mdw, 1);

    bool same_nextlast_dims = (nextlast_dim_src.idx == nextlast_dim_dst.idx);

    // src's innermost dim is assumed to be contiguous, it'll be read from
    // adjacent mem addresses
    auto src_innermost_stride
            = src_mdw.blocking_desc().strides[last_dim_src.idx];
    if (last_dim_src.size != src_mdw.padded_dims()[last_dim_src.idx]
            || (src_innermost_stride > 1 && src_mdw.is_plain())) {
        return false;
    }
    if (last_dim_src.idx != last_dim_dst.idx) { return false; }
    if (last_dim_src.size > last_dim_dst.size) { return false; }
    if (last_dim_dst.size > 16 || last_dim_dst.size % 8 != 0) { return false; }
    if (last_dim_src.idx == nextlast_dim_src.idx) { return false; }
    if (last_dim_src.idx == nextlast_dim_dst.idx) { return false; }
    if (nextlast_dim_src.size % 2 != 0) { return false; }
    if (nextlast_dim_dst.size % 2 != 0) { return false; }
    if (is_padded(src_mdw, last_dim_src.idx)) { return false; }
    // If src's and dst's nextlast dims are not the same, CL code will use
    // nested loops. In inner loop, data offset is incremented and can't be
    // converted back into tensor coordinates. This means there can't be
    // operations that depend on nextlast dim's coordinates in inner loop -
    // so no padding and no scale quant.
    if (!same_nextlast_dims) {
        if (is_padded(src_mdw, nextlast_dim_src.idx)) { return false; }
        if (is_padded(src_mdw, nextlast_dim_dst.idx)) { return false; }
        if (is_padded(dst_mdw, nextlast_dim_src.idx)) { return false; }
        if (is_padded(dst_mdw, nextlast_dim_dst.idx)) { return false; }
        if (scale_mask & (1 << nextlast_dim_src.idx)) { return false; }
        if (scale_mask & (1 << nextlast_dim_dst.idx)) { return false; }
    }
    if (scale_mask & (1 << last_dim_src.idx)) { return false; }
    // no 2nd layer of block on innermost dim in dst
    if (dst_mdw.padded_dims()[last_dim_src.idx] != last_dim_dst.size) {
        return false;
    }
    return true;
}

custom_kernel_t select_kernel(const conf_t &conf,
        const memory_desc_wrapper &src_mdw, const memory_desc_wrapper &dst_mdw,
        const compute::device_info_t *dev_info) {
    using namespace format_tag;

    const auto &padded_dims = dst_mdw.padded_dims();

    int last = conf.ndims - 1;
    size_t last_dim = padded_dims[last];

    const bool multi_scale_quant
            = (conf.src_quant.with_scale() && conf.src_quant.num_scales() > 1)
            || (conf.dst_quant.with_scale() && conf.dst_quant.num_scales() > 1);
    const bool has_padding_or_multi_scale_quant
            = conf.has_padding || multi_scale_quant;

    const bool type_s8_u8 = utils::one_of(src_mdw.data_type(), dnnl_s8, dnnl_u8)
            || utils::one_of(dst_mdw.data_type(), dnnl_s8, dnnl_u8);

    const bool allow_unroll
            = !conf.has_padding && !multi_scale_quant && !type_s8_u8;

    if (is_broadcast_by_strides(src_mdw) || is_broadcast_by_strides(dst_mdw)) {
        return custom_kernel_t::none;
    }

    // This kernel fails to compile with f8 data types
    if (utils::one_of(src_mdw.data_type(), dnnl_f8_e4m3, dnnl_f8_e5m2)
            || utils::one_of(dst_mdw.data_type(), dnnl_f8_e4m3, dnnl_f8_e5m2)) {
        return custom_kernel_t::none;
    }

    int mask = conf.src_quant.scale_mask() | conf.src_quant.zp_mask()
            | conf.dst_quant.scale_mask() | conf.dst_quant.zp_mask();

    if (matches_one_NxN_layout(src_mdw, dst_mdw, 16, mask)) {
        // W/A for compiler bug: avoid using intel_sub_group_shuffle with
        // SIMD16 on gen12lp
        if (dev_info->gpu_arch() == compute::gpu_arch_t::xe_lp) {
            return custom_kernel_t::transpose8x8;
        }
        return custom_kernel_t::transpose16x16;
    }
    if (matches_one_NxN_layout(src_mdw, dst_mdw, 8, mask)) {
        return custom_kernel_t::transpose8x8;
    }
    // This kernel reads a single scale per work-item; defer a per-dim scale to
    // the unaligned path below, which reloads it per element.
    if (src_mdw.matches_one_of_tag(nhwc) && dst_mdw.matches_one_of_tag(nchw)
            && padded_dims[last] % 16 == 0
            && dim_is_div_by_16_or_less_than_16(dst_mdw, 1)
            && !multi_scale_quant) {
        return custom_kernel_t::nchw;
    }
    if (src_mdw.matches_one_of_tag(nhwc) && dst_mdw.matches_one_of_tag(nchw)
            && dim_is_div_by_16_or_less_than_16(dst_mdw, 1)) {
        return custom_kernel_t::unaligned_sizes;
    }

    if (!has_padding_or_multi_scale_quant && (conf.nelems % 256 == 0)
            && src_mdw.similar_to(dst_mdw, true, false, 0)) {
        return custom_kernel_t::dense_vector;
    }

    if (fits_3ch(src_mdw, dst_mdw, mask)) {
        return custom_kernel_t::pad_innermost;
    }
    if (fits_xab_xba(src_mdw, dst_mdw, mask)) {
        return custom_kernel_t::xb_to_xab_xba;
    }
    // This kernel works on tensors that have common innermost dim. Tries to
    // access mem using large enough chunks to utilize whole cache lines.
    auto src_last = get_Nth_last_dim_or_block(src_mdw);
    auto dst_last = get_Nth_last_dim_or_block(dst_mdw);
    auto inner_dim = dst_last.idx;
    if (src_last.idx == dst_last.idx
            && (src_last.size <= 16 || dst_last.size <= 16)
            && src_last.size % 8 == 0 && dst_last.size % 8 == 0
            && conf.ndims <= MAX_NDIMS
            && (src_last.size % (2 * dst_last.size) == 0
                    || dst_last.size % (2 * src_last.size) == 0)
            && src_mdw.offset0() == 0 && dst_mdw.offset0() == 0
            && !(mask & (1 << inner_dim))
            && dst_mdw.dims()[inner_dim] == dst_mdw.padded_dims()[inner_dim]
            && src_mdw.dims()[inner_dim] == src_mdw.padded_dims()[inner_dim]) {
        return custom_kernel_t::vectorize_groups;
    }

    if (allow_unroll) {
        if (src_mdw.matches_one_of_tag(ABc16a16b, ABc16b16a, ABcd16a16b,
                    ABcd16b16a, ABcde16a16b, ABcde16b16a, BAc16a16b, BAc16b16a,
                    BAcd16a16b, BAcd16b16a, BAcde16b16a)
                || dst_mdw.matches_one_of_tag(ABc16a16b, ABc16b16a, ABcd16a16b,
                        ABcd16b16a, ABcde16a16b, ABcde16b16a, BAc16a16b,
                        BAc16b16a, BAcd16a16b, BAcd16b16a, BAcde16b16a)) {
            return custom_kernel_t::unroll_16a16b;
        }
        if (src_mdw.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)
                || dst_mdw.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)) {
            return custom_kernel_t::unroll_16b;
        }
        if (src_mdw.matches_one_of_tag(aBCd16b16c, aBCd16c16b, aBCde16b16c,
                    aBCde16c16b, aBCdef16b16c, aBCdef16c16b, aCBd16b16c,
                    aCBd16c16b, aCBde16b16c, aCBde16c16b, aCBdef16c16b)
                || dst_mdw.matches_one_of_tag(aBCd16b16c, aBCd16c16b,
                        aBCde16b16c, aBCde16c16b, aBCdef16b16c, aBCdef16c16b,
                        aCBd16b16c, aCBd16c16b, aCBde16b16c, aCBde16c16b,
                        aCBdef16c16b)) {
            return custom_kernel_t::unroll_16b16c;
        }
    }

    // The kernel vectorizes and shuffles the two innermost dims, so it reads a
    // single scale per work-item; a per-dim scale/zp on those dims would vary
    // within a work-item and cannot be indexed correctly here.
    const int xFxE_inner_mask
            = (1 << (conf.ndims - 1)) | (1 << (conf.ndims - 2));
    if (src_mdw.matches_one_of_tag(abdfce) && dst_mdw.matches_one_of_tag(abcdef)
            && ((padded_dims[conf.ndims - 2] % 16) == 0)
            && dim_is_div_by_16_or_less_than_16(dst_mdw, last)
            && !(mask & xFxE_inner_mask)) {
        return custom_kernel_t::plain_xFxE_to_abcdef;
    }

    // This kernel reads a single scale per work-item while a per-dim scale on
    // the reordered a/b dims varies within a work-item, so defer those cases to
    // a more generic kernel that indexes the scale per element.
    if ((src_mdw.matches_one_of_tag(abcd, acdb))
            && dst_mdw.matches_one_of_tag(
                    ABcd4a2b, ABcd4a4b, ABcd8a2b, ABcd8a4b)
            && src_mdw.is_dense() && dst_mdw.is_dense(true)
            && !multi_scale_quant) {
        return custom_kernel_t::plain_to_ABcd84a42b;
    }

    // This kernel will be used where last dimension is not reordered.
    // It will vectorize that dimension.
    if (!has_padding_or_multi_scale_quant && src_mdw.is_dense()
            && dst_mdw.is_dense() && last_dim % 8 == 0
            && dst_mdw.md_->format_desc.blocking.strides[last] == 1
            && src_mdw.md_->format_desc.blocking.strides[last] == 1
            && conf.ndims <= MAX_NDIMS) {
        return custom_kernel_t::vectorize_last_dim;
    }

    // This kernel supports 2D reorders into blocked formats that
    // end in 8a4b or 8a2b, no matter how many block layers, but no padding.
    if (!has_padding_or_multi_scale_quant && src_mdw.matches_one_of_tag(ab)
            && matches_ABxxxx8ayb_layout(
                    dst_mdw.md_->format_desc.blocking, conf.ndims)
            && padded_dims[last] % 16 == 0) {
        return custom_kernel_t::plain_to_ABxx8ayb;
    }

    if (conf.ndims >= 2 && conf.ndims <= 4
            && src_mdw.md_->format_desc.blocking.inner_nblks == 0
            && dst_mdw.md_->format_desc.blocking.inner_nblks == 0
            && src_mdw.offset0() == 0 && dst_mdw.offset0() == 0
            && is_alt_faster_than_ref(src_mdw, dst_mdw, dev_info)
            && !has_padding_or_multi_scale_quant) {
        return custom_kernel_t::alt;
    }

    return custom_kernel_t::none;
}

void custom_t::pd_t::alt_defines(compute::kernel_ctx_t &kernel_ctx) const {
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    size_t ndims = src_mdw.ndims();
    size_t last = ndims - 1;

    auto sdim = src_mdw.dims();
    auto sstr = src_mdw.blocking_desc().strides;
    auto dstr = dst_mdw.blocking_desc().strides;
    kernel_ctx.define_int("ALT_OFFSETS", 1);
    // LIMIT_MAX_D0 is necessary to avoid buffer overwritte.
    if (conf.dispatch.nd_range().global_range()[0] != (size_t)sdim[last]) {
        kernel_ctx.define_int("LIMIT_MAX_D0", sdim[last]);
    }
    kernel_ctx.define_int("S0", sstr[last]);
    kernel_ctx.define_int("S1", sstr[last - 1]);
    kernel_ctx.define_int("S2", ndims > 2 ? sstr[last - 2] : 1);
    kernel_ctx.define_int("SB", ndims > 3 ? sstr[last - 3] : 1);
    kernel_ctx.define_int("D0", dstr[last]);
    kernel_ctx.define_int("D1", dstr[last - 1]);
    kernel_ctx.define_int("D2", ndims > 2 ? dstr[last - 2] : 1);
    kernel_ctx.define_int("DB", ndims > 3 ? dstr[last - 3] : 1);
    kernel_ctx.define_int("BLK", ndims > 3 ? sdim[last - 3] : 1);
}

void custom_t::pd_t::alt_gen() {
    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());
    auto sdim = src_mdw.dims();

    size_t last = src_mdw.ndims() - 1;
    size_t gws3 = src_mdw.ndims() > 2 ? sdim[last - 2] : 1;
    compute::range_t gws((size_t)sdim[last], (size_t)sdim[last - 1], gws3);
    size_t work_group_size = 32;
    if (sdim[last] <= 16) { work_group_size = 16; }
    if (sdim[last] <= 8) { work_group_size = 8; }
    const compute::range_t lws(work_group_size, 1, 1);
    // Don't use nonuniform work groups, round up number work items if needed.
    size_t mod = gws[0] % lws[0];
    if (mod != 0) { gws[0] += lws[0] - mod; }
    conf.dispatch.generate_override(gws, lws);
}

status_t custom_t::pd_t::init_conf(const impl::engine_t *engine) {
    using namespace format_tag;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    conf.src_md_info = memory_desc_info_t::create(src_mdw);
    conf.dst_md_info = memory_desc_info_t::create(dst_mdw);
    conf.require_stateless_addressing = has_large_buffers();

    status_t status = status::success;

    const auto &padded_dims = dst_mdw.padded_dims();
    conf.src_quant = {attr(), src_mdw, DNNL_ARG_SRC};
    conf.dst_quant = {attr(), dst_mdw, DNNL_ARG_DST};
    conf.sum_quant = {attr()};
    conf.has_padding = !src_mdw.is_dense() || !dst_mdw.is_dense();
    conf.ndims = src_mdw.ndims();
    conf.nelems = utils::array_product(padded_dims, conf.ndims);

    conf.sub_group_size = 1;

    if (conf.nelems == 0) return status::success;

    dim_idx_t last = conf.ndims - 1;
    size_t last_dim = padded_dims[last];

    const auto *intel_engine = utils::downcast<const intel::engine_t *>(engine);

    conf.implementation = select_kernel(
            conf, src_mdw, dst_mdw, intel_engine->device_info());

    dim_t blocks[MAX_NDIMS] = {1, 1, 1, 1, 1, 1};
    int vect_size = 1;
    dim_idx_t vect_dim = 0;

    conf.dispatch = intel_engine->create_dispatch(dst_mdw.md_);
    int temp_block = 1;

    const bool may_use_sg8 = intel_engine->mayiuse_sub_group(8);
    int mask = conf.src_quant.scale_mask() | conf.src_quant.zp_mask()
            | conf.dst_quant.scale_mask() | conf.dst_quant.zp_mask();

    switch (conf.implementation) {
        case custom_kernel_t::none:
            VDISPATCH_REORDER_IC(false, "undefined kernel implementation");
        case custom_kernel_t::alt:
            // special handling with dispatcher override
            conf.sub_group_size = 16;
            break;
        case custom_kernel_t::dense_vector:
            // see special handling below
            conf.sub_group_size = 16;
            break;
        case custom_kernel_t::unroll_16b:
            conf.sub_group_size = 16;
            vect_dim = 1;
            vect_size = 16;
            break;
        case custom_kernel_t::unroll_16b16c:
            conf.sub_group_size = 16;
            blocks[2] = 16;
            vect_dim = 1;
            vect_size = 16;
            break;
        case custom_kernel_t::unroll_16a16b:
            conf.sub_group_size = 16;
            blocks[0] = 16;
            vect_dim = 1;
            vect_size = 16;
            break;
        case custom_kernel_t::plain_to_ABcd84a42b: {
            auto &blk = dst_mdw.blocking_desc();
            int inner_block = into<int>(blk.inner_blks[blk.inner_nblks - 1]);
            int outer_block = into<int>(blk.inner_blks[blk.inner_nblks - 2]);
            conf.sub_group_size = inner_block * outer_block;
            blocks[0] = outer_block;
            blocks[1] = inner_block;
            vect_dim = 3;
            vect_size = conf.sub_group_size;
        } break;
        case custom_kernel_t::xb_to_xab_xba:
            fill_conf_xab_xba(src_mdw, dst_mdw, mask, conf.aux_data.ab,
                    vect_dim, vect_size, &blocks[0]);
            conf.sub_group_size = vect_size;
            break;
        case custom_kernel_t::vectorize_last_dim:
            vect_dim = last;
            vect_size = (last_dim % 16 == 0) ? 16 : 8;
            VDISPATCH_REORDER_IC(may_use_sg8 || vect_size != 8,
                    VERBOSE_BAD_PARAM, "vect_size");
            for (dim_idx_t dim = last - 1; dim < MAX_NDIMS && temp_block == 1;
                    dim--) {
                if (padded_dims[dim] % 4 == 0) { temp_block = 4; }
                if (padded_dims[dim] % 8 == 0) { temp_block = 8; }
                if (padded_dims[dim] % 16 == 0) { temp_block = 16; }
                blocks[dim] = temp_block;
            }
            break;
        case custom_kernel_t::pad_innermost: {
            auto last_dim_src = get_Nth_last_dim_or_block(src_mdw);
            auto nextlast_dim_src = get_Nth_last_dim_or_block(src_mdw, 1);
            auto last_dim_dst = get_Nth_last_dim_or_block(dst_mdw);
            auto nextlast_dim_dst = get_Nth_last_dim_or_block(dst_mdw, 1);

            dim_t min_common_size
                    = std::min(last_dim_src.size, last_dim_dst.size);
            dim_t max_common_size
                    = std::max(last_dim_src.size, last_dim_dst.size);
            conf.sub_group_size = into<int>(max_common_size);
            VDISPATCH_REORDER_IC(may_use_sg8 || conf.sub_group_size != 8,
                    VERBOSE_BAD_PARAM, "sub_group_size");

            // Group size bigger than 4 would need too much private mem;
            // group size 1 will give worse perf than reference kernel.
            int max_group_size = 4;
            while (nextlast_dim_src.size % max_group_size != 0
                    || nextlast_dim_dst.size % max_group_size != 0) {
                max_group_size--;
            }

            conf.aux_data.vg.vector_dim = last_dim_src.idx;
            conf.aux_data.vg.src_loop_dim = nextlast_dim_dst.idx;
            conf.aux_data.vg.dst_loop_dim = nextlast_dim_src.idx;
            conf.aux_data.vg.innermost_size = into<int>(min_common_size);

            blocks[conf.aux_data.vg.src_loop_dim] = max_group_size;
            blocks[conf.aux_data.vg.dst_loop_dim] = max_group_size;
            // if src loop and dst loop dims are the same, CL code would iterate
            // over the same dimension in inner and outer loop, effectively doing
            // redundant operations; set inner loop counter to 1 in that case.
            conf.aux_data.vg.group_size
                    = (nextlast_dim_dst.idx == nextlast_dim_src.idx)
                    ? 1
                    : max_group_size;

            vect_dim = conf.aux_data.vg.vector_dim;
            vect_size = conf.sub_group_size;
        } break;

        case custom_kernel_t::vectorize_groups: {
            auto last_dim_src = get_Nth_last_dim_or_block(src_mdw);
            auto nextlast_dim_src = get_Nth_last_dim_or_block(src_mdw, 1);
            auto last_dim_dst = get_Nth_last_dim_or_block(dst_mdw);
            auto nextlast_dim_dst = get_Nth_last_dim_or_block(dst_mdw, 1);
            dim_t min_common_size
                    = std::min(last_dim_src.size, last_dim_dst.size);
            vect_size = (min_common_size % 16 == 0) ? 16 : 8;
            vect_dim = last_dim_src.idx;
            VDISPATCH_REORDER_IC(may_use_sg8 || vect_size != 8,
                    VERBOSE_BAD_PARAM, "vect_size");

            assert(last_dim_src.size % vect_size == 0
                    && last_dim_dst.size % vect_size == 0);
            assert(last_dim_src.idx == last_dim_dst.idx);
            dim_t src_chunks;
            if (last_dim_src.size / vect_size > 1) {
                src_chunks = last_dim_src.size / vect_size;
                conf.aux_data.vg.dst_loop_dim = last_dim_src.idx;
            } else {
                src_chunks = nextlast_dim_src.size;
                conf.aux_data.vg.dst_loop_dim = nextlast_dim_src.idx;
            }
            dim_t dst_chunks;
            if (last_dim_dst.size / vect_size > 1) {
                dst_chunks = last_dim_dst.size / vect_size;
                conf.aux_data.vg.src_loop_dim = last_dim_dst.idx;
            } else {
                dst_chunks = nextlast_dim_dst.size;
                conf.aux_data.vg.src_loop_dim = nextlast_dim_dst.idx;
            }
            // TODO:
            // Final algorithm for selecting group size should consider:
            // 1. Group size must be small enough to guarantee no spill.
            // 2. Group size should be large enough to fill whole cache lines
            //    on both reads and writes, with line size determined by HW
            // 3. If there's not enough data to feed all EUs, ignore (2) and
            //    decrease group size.
            int max_data_size = (int)std::max(
                    src_mdw.data_type_size(), dst_mdw.data_type_size());

            int group = 16 / max_data_size;
            while (group > 1) {
                if (src_chunks % group == 0 && dst_chunks % group == 0) {
                    break;
                }
                group--;
            }
            assert(group >= 1);

            conf.aux_data.vg.vector_dim = last_dim_src.idx;
            conf.aux_data.vg.group_size = group;
            conf.sub_group_size = vect_size;

            blocks[conf.aux_data.vg.src_loop_dim] = group;
            blocks[conf.aux_data.vg.dst_loop_dim] = group;
        } break;
        case custom_kernel_t::plain_to_ABxx8ayb:
            conf.sub_group_size = 16;
            blocks[0] = 8;
            vect_dim = last;
            vect_size = 16;
            break;
        case custom_kernel_t::plain_xFxE_to_abcdef:
            conf.sub_group_size = 16;
            blocks[5] = nstl::min(padded_dims[conf.ndims - 1], dnnl_dim_t(16));
            vect_dim = 4;
            vect_size = 16;
            break;
        case custom_kernel_t::transpose8x8:
            VDISPATCH_REORDER_IC(
                    may_use_sg8, VERBOSE_BAD_PARAM, "sub_group_size");
            conf.sub_group_size = 8;
            blocks[get_Nth_last_dim_or_block(dst_mdw).idx] = 8;
            vect_dim = get_Nth_last_dim_or_block(src_mdw).idx;
            vect_size = 8;
            break;
        case custom_kernel_t::transpose16x16:
            conf.sub_group_size = 16;
            blocks[get_Nth_last_dim_or_block(dst_mdw).idx] = 16;
            vect_dim = get_Nth_last_dim_or_block(src_mdw).idx;
            vect_size = 16;
            break;
        case custom_kernel_t::nchw:
            conf.sub_group_size = 16;
            blocks[1] = nstl::min(padded_dims[1], dnnl_dim_t(16));
            vect_dim = 3;
            vect_size = 16;
            break;
        case custom_kernel_t::unaligned_sizes:
            blocks[1] = padded_dims[1];
            break;
    }

    // special case for dense_vector kernel - treat tensors as flat 1D vectors
    if (conf.implementation == custom_kernel_t::dense_vector) {
        conf.dispatch.define_dim("D0", 0, conf.nelems, 16);
        CHECK(conf.dispatch.vectorize_dim("D0", 16));
    } else {
        for (dim_idx_t i = 0; i < MAX_NDIMS; ++i) {
            auto dim_str = utils::format("D%d", i);
            if (i < into<dim_idx_t>(dst_mdw.ndims())) {
                dim_t dim = padded_dims[i];
                // if needed to align vectorized dim with vector size, pad that dim again
                if (i == vect_dim) { dim = utils::rnd_up(dim, vect_size); }
                conf.dispatch.define_dim(dim_str, i, dim, blocks[i]);
            } else {
                conf.dispatch.define_dim(dim_str, 1);
            }
        }
        if (vect_size != 1) {
            auto dim_str = utils::format("D%d", vect_dim);
            CHECK(conf.dispatch.vectorize_dim(dim_str, vect_size));
        }
    }

    if (conf.implementation == custom_kernel_t::alt) {
        alt_gen();
    } else {
        conf.dispatch.generate();
    }
    return status;
}

status_t custom_t::pd_t::init_kernel_ctx(
        compute::kernel_ctx_t &kernel_ctx) const {
    using namespace format_tag;

    const memory_desc_wrapper src_mdw(src_md());
    const memory_desc_wrapper dst_mdw(dst_md());

    if (conf.nelems == 0) return status::success;

    kernel_ctx.define_int("NDIMS", conf.ndims);
    kernel_ctx.add_option("-cl-std=CL2.0");

    conf.src_quant.define_macros(kernel_ctx, "SRC");
    conf.dst_quant.define_macros(kernel_ctx, "DST");
    conf.sum_quant.define_macros(kernel_ctx, "SUM");

    def_dispatch(kernel_ctx, conf.dispatch);
    kernel_ctx.require_stateless_addressing(conf.require_stateless_addressing);

    // the 'unaligned_sizes' kernel uses the same implementation in .cl
    // the difference is in sizes of blocks[]
    if (conf.implementation == custom_kernel_t::unaligned_sizes) {
        kernel_ctx.define_int("UNALIGNED", 1);
    }
    kernel_ctx.define_int("SUB_GROUP_SIZE", conf.sub_group_size);

    kernel_ctx.define_int("PAD_FILL_ZERO", conf.has_padding);
    if (conf.implementation == custom_kernel_t::dense_vector) {
        kernel_ctx.add_option("-Dcl_intel_subgroups_char");
        kernel_ctx.define_int("USE_DENSE_VECT", 1);
    }

    def_memory_desc_info(kernel_ctx, conf.src_md_info, "SRC");
    def_memory_desc_info(kernel_ctx, conf.dst_md_info, "DST");

    // distinguish between various flavors of unroll kernel
    if (src_mdw.matches_one_of_tag(
                ABc16a16b, ABcd16a16b, ABcde16a16b, BAc16a16b, BAcd16a16b)) {
        kernel_ctx.define_int("SRC_16A16B", 1);
    } else if (src_mdw.matches_one_of_tag(ABc16b16a, ABcd16b16a, ABcde16b16a,
                       BAc16b16a, BAcd16b16a, BAcde16b16a)) {
        kernel_ctx.define_int("SRC_16B16A", 1);
    } else if (src_mdw.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)) {
        kernel_ctx.define_int("SRC_16B", 1);
    } else if (src_mdw.matches_one_of_tag(aBCd16b16c, aBCde16b16c, aBCdef16b16c,
                       aCBd16b16c, aCBde16b16c)) {
        kernel_ctx.define_int("SRC_16B16C", 1);
    } else if (src_mdw.matches_one_of_tag(aBCd16c16b, aBCde16c16b, aBCdef16c16b,
                       aCBd16c16b, aCBde16c16b, aCBdef16c16b)) {
        kernel_ctx.define_int("SRC_16C16B", 1);
    }
    if (dst_mdw.matches_one_of_tag(
                ABc16a16b, ABcd16a16b, ABcde16a16b, BAc16a16b, BAcd16a16b)) {
        kernel_ctx.define_int("DST_16A16B", 1);
    } else if (dst_mdw.matches_one_of_tag(ABc16b16a, ABcd16b16a, ABcde16b16a,
                       BAc16b16a, BAcd16b16a, BAcde16b16a)) {
        kernel_ctx.define_int("DST_16B16A", 1);
    } else if (dst_mdw.matches_one_of_tag(aBc16b, aBcd16b, aBcde16b)) {
        kernel_ctx.define_int("DST_16B", 1);
    } else if (dst_mdw.matches_one_of_tag(aBCd16b16c, aBCde16b16c, aBCdef16b16c,
                       aCBd16b16c, aCBde16b16c)) {
        kernel_ctx.define_int("DST_16B16C", 1);
    } else if (dst_mdw.matches_one_of_tag(aBCd16c16b, aBCde16c16b, aBCdef16c16b,
                       aCBd16c16b, aCBde16c16b, aCBdef16c16b)) {
        kernel_ctx.define_int("DST_16C16B", 1);
    }

    if (conf.implementation == custom_kernel_t::alt) {
        alt_defines(kernel_ctx);
    }
    if (conf.implementation == custom_kernel_t::plain_xFxE_to_abcdef)
        kernel_ctx.define_int("PLAIN_xFxE_TO_ABCDEF", 1);

    if (conf.implementation == custom_kernel_t::plain_to_ABcd84a42b) {
        kernel_ctx.define_int("PLAIN_TO_ABCD84A42B", 1);
        compute::nd_range_t nd_range = conf.dispatch.nd_range();
        const auto &lws = nd_range.local_range();
        if (!lws) return status::runtime_error;
        kernel_ctx.define_int("SG_PER_WG", lws.nelems() / conf.sub_group_size);
    }
    if (conf.implementation == custom_kernel_t::xb_to_xab_xba) {
        kernel_ctx.define_int("XAB_XBA", 1);
        compute::nd_range_t nd_range = conf.dispatch.nd_range();
        const auto &lws = nd_range.local_range();
        if (!lws) return status::runtime_error;
        kernel_ctx.define_int("SG_PER_WG", lws.nelems() / conf.sub_group_size);
        kernel_ctx.define_int("BLOCK_SIZE", conf.aux_data.ab.blk_size);
        kernel_ctx.define_int("SRC_BLK_DIM", conf.aux_data.ab.src_blk_dim);
        kernel_ctx.define_int("SRC_OFF_COEFF", conf.aux_data.ab.src_blk_coeff);
        kernel_ctx.define_int("DST_BLK_DIM", conf.aux_data.ab.dst_blk_dim);
        kernel_ctx.define_int("DST_OFF_COEFF", conf.aux_data.ab.dst_blk_coeff);
        kernel_ctx.define_int("XB_TO_XAB", conf.aux_data.ab.vd);
    }

    if (conf.implementation == custom_kernel_t::vectorize_last_dim) {
        kernel_ctx.define_int("VECTORIZE_LAST_DIM", 1);
    }

    if (conf.implementation == custom_kernel_t::pad_innermost) {
        kernel_ctx.define_int("PAD_INNERMOST", 1);
        kernel_ctx.define_int(
                "VECT_DIM", conf.aux_data.vg.vector_dim); //useless
        kernel_ctx.define_int("SRC_LOOP_DIM", conf.aux_data.vg.src_loop_dim);
        kernel_ctx.define_int("DST_LOOP_DIM", conf.aux_data.vg.dst_loop_dim);
        kernel_ctx.define_int("GROUP", conf.aux_data.vg.group_size);
        compute::nd_range_t nd_range = conf.dispatch.nd_range();
        const auto &lws = nd_range.local_range();
        if (!lws) return status::runtime_error;
        kernel_ctx.define_int("SG_PER_WG", lws.nelems() / conf.sub_group_size);
        kernel_ctx.define_int(
                "INNERMOST_SIZE", conf.aux_data.vg.innermost_size);
        kernel_ctx.define_int("VECT_SIZE", conf.sub_group_size);
        bool has_non_innermost_padding = false;
        for (dim_idx_t i = 0; i < MAX_NDIMS; i++) {
            if (i == conf.aux_data.vg.vector_dim) { continue; }
            has_non_innermost_padding
                    |= (dst_mdw.dims()[i] != dst_mdw.padded_dims()[i]);
        }
        kernel_ctx.define_int(
                "NON_INNERMOST_PADDING", has_non_innermost_padding);
        auto last_dim_dst = get_Nth_last_dim_or_block(dst_mdw);
        kernel_ctx.define_int("DST_INNERMOST_STRIDE",
                dst_mdw.is_plain()
                        ? dst_mdw.blocking_desc().strides[last_dim_dst.idx]
                        : 1);
    }
    if (conf.implementation == custom_kernel_t::vectorize_groups) {
        kernel_ctx.define_int("VECTORIZE_GROUPS", 1);
        kernel_ctx.define_int("VECT_DIM", conf.aux_data.vg.vector_dim);
        kernel_ctx.define_int("SRC_LOOP_DIM", conf.aux_data.vg.src_loop_dim);
        kernel_ctx.define_int("DST_LOOP_DIM", conf.aux_data.vg.dst_loop_dim);
        kernel_ctx.define_int("GROUP", conf.aux_data.vg.group_size);
    }
    if (conf.implementation == custom_kernel_t::plain_to_ABxx8ayb) {
        kernel_ctx.define_int("PLAIN_TO_AB_XX_8AYB", 1);
        kernel_ctx.define_int(
                "BLK_L", innermost_block(dst_mdw.md_->format_desc.blocking));
    }

    if (conf.implementation == custom_kernel_t::transpose8x8
            || conf.implementation == custom_kernel_t::transpose16x16) {
        kernel_ctx.define_int("TRANSPOSE_NXN", 1);
        kernel_ctx.define_int(
                "DST_BLOCK_DIM", get_Nth_last_dim_or_block(src_mdw).idx);
    }

    if (conf.implementation == custom_kernel_t::nchw) {
        kernel_ctx.define_int("REORDER_NCHW", 1);
    }

    return status::success;
}

void custom_t::pd_t::init_scratchpad() {
    if (conf.src_quant.with_scale()) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_src_scales,
                conf.src_quant.num_scales(), sizeof(float),
                OCL_BUFFER_ALIGNMENT);
    }
    if (conf.dst_quant.with_scale()) {
        auto scratchpad = scratchpad_registry().registrar();
        scratchpad.book(memory_tracking::names::key_reorder_dst_scales,
                conf.dst_quant.num_scales(), sizeof(float),
                OCL_BUFFER_ALIGNMENT);
    }
}

status_t custom_t::execute(const exec_ctx_t &ctx) const {

    status_t status = status::success;

    auto &src = CTX_IN_STORAGE(DNNL_ARG_FROM);
    auto &dst = CTX_OUT_STORAGE(DNNL_ARG_TO);
    CHECK(status);

    const auto &conf = pd()->conf;
    if (conf.nelems == 0) return status::success;

    compute::kernel_arg_list_t arg_list;
    arg_list.set(0, src);
    arg_list.set(1, dst);

    arg_list.set(2, conf.src_quant.scales(ctx));
    arg_list.set(3, conf.src_quant.zero_points(ctx));
    arg_list.set(4, conf.dst_quant.scales(ctx));
    arg_list.set(5, conf.dst_quant.zero_points(ctx));

    arg_list.set(6, conf.sum_quant.scales());
    arg_list.set(7, conf.sum_quant.zero_points());

    auto nd_range = conf.dispatch.nd_range();

    status = parallel_for(ctx, nd_range, kernel_, arg_list);

    return status;
}

} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
