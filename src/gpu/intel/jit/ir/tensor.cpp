/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include <cctype>

#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/jit/utils/type_bridge.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

tile_coord_t split(const layout_t &layout, const grid_info_t &grid_info,
        grid_info_t *out_grid) {
    tile_coord_t min_tile_coord = tile_coord_t::invalid();
    std::vector<dim_t> cur_dims(grid_info.ndims(), 1);

    for (dim_t iter = 0; iter < grid_info.elems(); iter++) {
        for (dim_idx_t i = 0; i < grid_info.ndims(); i++) {
            if (++cur_dims[i] <= grid_info.dim(i)) break;
            cur_dims[i] = 1;
        }
        auto sub_grid = grid_info.resize(cur_dims);
        auto tile_coord = split_exact(layout, sub_grid);
        if (tile_coord.is_invalid()) continue;
        if (min_tile_coord.is_invalid()
                || tile_coord.elems() < min_tile_coord.elems()) {
            min_tile_coord = std::move(tile_coord);
            if (out_grid) { *out_grid = std::move(sub_grid); }
        }
    }
    return min_tile_coord;
}

tile_coord_t split_exact(const layout_t &layout, const grid_info_t &grid) {
    tile_t tile;
    if (layout.elems() % grid.elems() != 0) return tile_coord_t::invalid();

    dim_t cur_elems_per_tile = 1;
    dim_t elems_per_tile = layout.elems() / grid.elems();
    for (auto &b : layout.blocks()) {
        dim_t block = std::min(b.size, elems_per_tile / cur_elems_per_tile);
        tile[b.idx] = tile.get(b.idx, 1) * block;
        cur_elems_per_tile *= block;
    }
    if (cur_elems_per_tile != elems_per_tile) return tile_coord_t::invalid();

    return split(layout, tile, grid);
}

tile_coord_t split_exact(const layout_t &layout, int factor) {
    if (factor == 1) return tile_coord_t(layout.tile());
    if (layout.elems() % factor != 0) return tile_coord_t::invalid();
    dim_t cur_elems = 1;
    dim_t split_elems = layout.elems() / factor;
    std::vector<layout_block_t> split_blocks;
    for (auto &b : layout.blocks()) {
        if (cur_elems * b.size > split_elems) {
            if (split_elems % cur_elems != 0) return tile_coord_t::invalid();
            auto bb = b;
            bb.size = split_elems / cur_elems;
            if (b.size % bb.size != 0) return tile_coord_t::invalid();
            split_blocks.push_back(bb);
        } else {
            split_blocks.push_back(b);
        }
        cur_elems *= split_blocks.back().size;
        if (cur_elems == split_elems) break;
    }
    tile_t split_tile;
    for (auto &b : split_blocks)
        split_tile[b.idx] = split_tile.get(b.idx, 1) * b.size;
    return tile_coord_t(split_tile);
}

tile_coord_t split(const layout_t &layout, const tile_t &tile,
        const grid_info_t &grid, std::vector<layout_block_t> *outer_blocks) {
    if (outer_blocks) outer_blocks->resize(0);

    if (grid.elems() == 1) return tile_coord_t(tile);

    dim_t total_elems = layout.elems();
    dim_t tile_elems = tile.elems();

    grid_splitter_t grid_splitter(grid);
    gpu_assert(tile_elems * grid.elems() == total_elems)
            << "Tile/grid dimensions do not match.";
    MAYBE_UNUSED(total_elems);
    MAYBE_UNUSED(tile_elems);

    tile_t dims;
    coord_t start;
    auto rem_tile = tile;
    for (auto &b : layout.blocks()) {
        if (b.size == 1) continue;

        dim_t &e = rem_tile[b.idx];
        if (e > 1) {
            if (e % b.size == 0) {
                e /= b.size;
            } else if (b.size % e == 0) {
                auto tmp_layout = layout.split_block(b, e, b.size / e);
                return split(tmp_layout, tile, grid, outer_blocks);
            } else {
                return tile_coord_t::invalid();
            }
        } else {
            dim_t next_chunk = math::gcd(b.size, grid_splitter.cur_block());
            if (b.size == next_chunk) {
                auto idx = grid_splitter.pop_block(next_chunk);
                start[b.idx] += idx * dims[b.idx];
                if (outer_blocks) outer_blocks->push_back(b);
            } else if (b.size % next_chunk == 0 && next_chunk != 1) {
                auto tmp_layout = layout.split_block(
                        b, next_chunk, b.size / next_chunk);
                return split(tmp_layout, tile, grid, outer_blocks);
            } else {
                return tile_coord_t::invalid();
            }
        }
        dims[b.idx] *= b.size;
    }
    return tile_coord_t(tile, start);
}

memory_desc_t to_md(const layout_t &l, const memory_desc_t &md_hint) {
    auto dims_hint = md_hint.dims;
    auto ndims = md_hint.ndims;
    memory_desc_t md = {};
    md.ndims = ndims;
    std::copy(dims_hint, dims_hint + ndims, md.dims);
    md.data_type = jit::to_dnnl(l.type());
    md.offset0 = to_cpp<dim_t>(l.offset());
    md.format_kind = format_kind::blocked;

    auto &blk = md.format_desc.blocking;
    bool seen[DNNL_MAX_NDIMS] = {};

    bool in_inner_block = false;
    dim_t prev_stride = 0;

    for (auto it = l.blocks().rbegin(); it != l.blocks().rend(); ++it) {
        auto &b = *it;
        if (!seen[b.idx]) {
            // Outer block.
            gpu_assert(!in_inner_block);
            MAYBE_UNUSED(in_inner_block);
            blk.strides[b.idx] = dim_t(b.stride);
            md.padded_dims[b.idx] = b.size;
        } else {
            // Inner block.
            md.padded_dims[b.idx] *= b.size;
            blk.inner_idxs[blk.inner_nblks] = b.idx;
            blk.inner_blks[blk.inner_nblks] = b.size;
            blk.inner_nblks++;
            if (prev_stride > 0) {
                // Inner block must be dense.
                gpu_assert(prev_stride == b.size * dim_t(b.stride));
            }
            prev_stride = dim_t(b.stride);
            in_inner_block = true;
        }
        seen[b.idx] = true;
    }

    for (int i = 0; i < ndims; i++) {
        if (seen[i]) continue;
        gpu_assert(md.dims[i] == 1);
        md.padded_dims[i] = md.dims[i];
        blk.strides[i] = l.elems();
    }

    return md;
}

void align_layouts(layout_t &a, layout_t &b) {
    for (auto &d : a.tile()) {
        auto a_blocks = a.blocks();
        auto b_blocks = b.blocks();

        int a_max = int(a_blocks.size());
        int b_max = int(b_blocks.size());
        int a_idx = 0;
        int b_idx = 0;

        for (;;) {
            while (a_idx < a_max && a_blocks[a_idx].idx != d)
                a_idx++;
            while (b_idx < b_max && b_blocks[b_idx].idx != d)
                b_idx++;

            if (a_idx >= a_max || b_idx >= b_max) break;

            auto &ab = a_blocks[a_idx];
            auto &bb = b_blocks[b_idx];
            dim_t common_size = math::gcd(ab.size, bb.size);
            if (ab.size == common_size && bb.size == common_size) {
                a_idx++;
                b_idx++;
                continue;
            }

            if (ab.size != common_size) {
                a = a.split_block(a[a_idx], common_size, ab.size / common_size);
            }
            if (bb.size != common_size) {
                b = b.split_block(b[b_idx], common_size, bb.size / common_size);
            }
            break;
        }
    }
}

expr_t grid_splitter_t::pop_block(dim_t size) {
    gpu_assert(size > 1);
    gpu_assert(can_pop_block(size));

    dim_t new_stride = cur_stride_ * size;

    auto idx_expr = grid_.idx(cur_idx_);
    if (cur_stride_ != 1) idx_expr /= cur_stride_;
    if (new_stride != grid_.dim(cur_idx_)) idx_expr %= size;

    cur_stride_ = new_stride;
    if (cur_stride_ == grid_.dim(cur_idx_)) {
        // Move to the next dimension.
        cur_idx_--;
        skip_size_1_dims();
        cur_stride_ = 1;
    }
    return idx_expr;
}

stride_t tdim_t::compute_stride(
        const expr_t &e, dim_idx_t idx, const expr_t &var) {
    // e == var -> fixed stride.
    if (e.is_same(var)) return stride_t(1);

    auto e1 = substitute(e, var, var + 1);
    auto e_stride = simplify(e1 - e);

    if (is_const(e_stride)) return stride_t(to_cpp<dim_t>(e_stride));

    // Stride is not a constant.
    return stride_t::unknown();
}

view_t view_t::create_sub_view(const tile_t &tile, const coord_t &coord) const {
    auto ret = *this;
    for (dim_idx_t i = 0; i < nvdims(); i++) {
        ret.vdims_[i] = tile.get(i);
        if (!coord.has(i) || coord[i].is(0)) continue;
        auto &i_start = coord[i];
        auto &s = ret.vstart_[i];
        s += i_start;
        s = simplify(s);
    }
    return ret;
}

view_t view_t::substitute(const expr_t &from, const expr_t &to) const {
    view_t ret = *this;
    for (dim_idx_t i = 0; i < nvdims(); i++) {
        ret.vstart_[i] = ir::substitute(ret.vstart_[i], from, to);
        ret.vstart_[i] = simplify(ret.vstart_[i]);
    }
    return ret;
}

std::vector<expr_t> view_t::create_vvars(dim_idx_t nvdims) {
    static const int max_nvdims = 128;
    static thread_local std::vector<expr_t> _vvars([] {
        std::vector<expr_t> ret;
        ret.reserve(max_nvdims);
        for (int i = 0; i < max_nvdims; i++)
            ret.push_back(
                    var_t::make(dsl::type_t::s32(), "_" + std::to_string(i)));
        return ret;
    }());

    gpu_assert(nvdims <= max_nvdims) << "Too many dimensions: " << nvdims;
    return std::vector<expr_t>(_vvars.begin(), _vvars.begin() + nvdims);
}

layout_t view_t::create_pseudo_vlayout(
        const layout_t &tlayout, bool init_offset) const {
    gpu_assert(!tlayout.is_empty());

    auto rem_vdims = vdims_;
    std::vector<layout_block_t> blocks;

    for (auto &tb : tlayout.blocks()) {
        bool tb_is_outermost = tlayout.is_outermost(tb);
        dim_t tsize = tb.size;

        auto &tinfo = tdims_[tb.idx];
        if (tb_is_outermost) {
            // Use innermost dimension with maximum remaining size for first
            // block
            dim_idx_t max_idx = dim_idx::invalid;
            dim_idx_t max_vidx = dim_idx::invalid;
            dim_t max_vdim = 1;
            for (int i = tinfo.nvargs() - 1; i >= 0; i--) {
                dim_idx_t vidx = tinfo.vidx(i);
                if (rem_vdims[vidx] > max_vdim) {
                    max_idx = i;
                    max_vidx = vidx;
                    max_vdim = rem_vdims[vidx];
                }
            }

            if (max_vdim > 1) {
                stride_t stride = tinfo.vstride(max_idx);
                blocks.emplace_back(
                        max_vidx, max_vdim, stride * stride_t(tb.stride));
                rem_vdims[max_vidx] = 1;
            }

            for (int i = tinfo.nvargs() - 1; i >= 0; i--) {
                dim_idx_t vidx = tinfo.vidx(i);
                if (rem_vdims[vidx] == 1) continue;

                stride_t stride = tinfo.vstride(i) * tb.stride;
                blocks.emplace_back(vidx, rem_vdims[vidx], stride);
                rem_vdims[vidx] = 1;
            }
            continue;
        }

        gpu_assert(tinfo.is_identity()) << "Can't create pseudo-layout.";

        int vidx = tinfo.vidx(0);
        dim_t &rem_vdim = rem_vdims[vidx];
        if (rem_vdim == 1) continue;

        if (rem_vdim % tsize == 0) {
            rem_vdim /= tsize;
        } else if (rem_vdim % tsize != 0) {
            // Try to split the current block and start from scratch.
            if (tsize % rem_vdim == 0) {
                auto tmp_layout
                        = tlayout.split_block(tb, rem_vdim, tsize / rem_vdim);
                return create_pseudo_vlayout(tmp_layout, init_offset);
            }

            // TODO: Remove exception usage.
            gpu_except_not_implemented("Can't create pseudo-layout.");
        }
        blocks.emplace_back(tb.idx, tsize, tb.stride);
    }

    for (auto &d : rem_vdims) {
        gpu_assert(rem_vdims[d] == 1) << "Can't create pseudo-layout.";
    }

    layout_t ret(tlayout.type(), blocks, 0, nvdims());
    if (!init_offset) return ret;

    auto targs = cvt_vargs_to_targs();
    auto off = tlayout.offset(targs);
    return layout_t(tlayout.type(), blocks, off, nvdims());
}

layout_t dim_assignment_t::map(const layout_t &layout) const {
    std::vector<layout_block_t> new_blocks;
    for (auto &b : layout.blocks()) {
        size_t new_idx = assignments_[b.idx];
        if (new_idx == dim_idx::invalid) continue; // Drop this block.
        auto new_b = b;
        new_b.idx = new_idx;
        new_blocks.push_back(new_b);
    }
    new_blocks = merge_blocks(new_blocks);
    auto ret = layout_t(layout.type(), new_blocks, layout.offset(), new_ndims(),
            /*do_normalize=*/false);
    gpu_assert(layout.elems() == ret.elems())
            << "Assignment doesn't preserve number of elements.";
    return ret;
}

layout_t spatials_to_3d(const layout_t &layout, bool with_groups,
        const std::array<int, 3> &dhw_map) {
    const size_t old_ndims = layout.ndims();
    const size_t old_sp_ndims = old_ndims - (with_groups ? 3 : 2);
    const size_t new_ndims = old_ndims - old_sp_ndims + 3;

    dim_assignment_t to_3d(old_ndims, new_ndims);
    for (size_t i = 0; i < old_ndims; i++) {
        if (i < old_ndims - old_sp_ndims) {
            // Non-spatial dimensions.
            to_3d.assign(i, i);
        } else {
            // Spatial dimensions.
            size_t old_sp_idx = 3 - (old_ndims - i);
            size_t new_sp_idx = dhw_map[old_sp_idx];
            to_3d.assign(i, new_ndims - (3 - new_sp_idx));
        }
    }
    return to_3d.map(layout);
}

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
