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

#ifndef GPU_INTEL_JIT_IR_TENSOR_HPP
#define GPU_INTEL_JIT_IR_TENSOR_HPP

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <unordered_map>

#include "gemmstone/../../dsl/ir/pass/simplify.hpp"
#include "gpu/intel/block_structure.hpp"
#include "gpu/intel/jit/ir/legacy.hpp"
#include "gpu/intel/jit/ir/problem.hpp"
#include "gpu/intel/jit/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

class grid_info_t {
public:
    grid_info_t() = default;
    grid_info_t(dim_idx_t ndims) : dims_(ndims), offs_(ndims), idxs_(ndims) {}
    grid_info_t(const std::vector<dim_t> &dims, const std::vector<expr_t> &idxs)
        : grid_info_t(dims, {}, idxs) {}
    grid_info_t(
            const std::vector<dim_t> &dims, const std::string &(*genname)(int))
        : grid_info_t(dims, make_idxs(genname, into<dim_idx_t>(dims.size()))) {}
    grid_info_t(const std::vector<dim_t> &dims, const std::vector<dim_t> &offs,
            const std::vector<expr_t> &idxs)
        : dims_(dims), offs_(offs), idxs_(idxs) {
        if (offs_.empty()) offs_.resize(dims.size());
        gpu_assert(dims_.size() == offs_.size());
        gpu_assert(dims_.size() == idxs_.size());
    }

    bool operator==(const grid_info_t &other) const {
        if (ndims() != other.ndims()) return false;
        for (dim_idx_t i = 0; i < ndims(); i++) {
            if (dim(i) != other.dim(i)) return false;
            if (off(i) != other.off(i)) return false;
            if (!idx(i).is_equal(other.idx(i))) return false;
        }
        return true;
    }

    bool is_empty() const { return dims_.empty(); }

    dim_t &dim(dim_idx_t dim_idx) { return dims_[dim_idx]; }
    dim_t &off(dim_idx_t dim_idx) { return offs_[dim_idx]; }
    expr_t &idx(dim_idx_t dim_idx) { return idxs_[dim_idx]; }
    dim_idx_t dim_idx(const expr_t &idx_var) const {
        for (dim_idx_t i = 0; i < ndims(); i++) {
            if (idx(i).is_same(idx_var)) return i;
        }
        gpu_error_not_expected() << "Index not found: " << idx_var;
        return dim_idx::invalid;
    }

    const dim_t &dim(dim_idx_t dim_idx) const { return dims_[dim_idx]; }
    const dim_t &dim(const expr_t &idx_var) const {
        return dims_[dim_idx(idx_var)];
    }
    const dim_t &off(dim_idx_t dim_idx) const { return offs_[dim_idx]; }
    const expr_t &idx(dim_idx_t dim_idx) const { return idxs_[dim_idx]; }

    dim_t &operator[](dim_idx_t dim_idx) { return dim(dim_idx); }
    const dim_t &operator[](dim_idx_t dim_idx) const { return dim(dim_idx); }

    dim_idx_t ndims() const { return into<dim_idx_t>(dims_.size()); }
    dim_t elems() const {
        return utils::array_product(dims_.data(), dims_.size());
    }

    grid_info_t sub_grid(std::initializer_list<dim_idx_t> old_dim_idxs) const {
        grid_info_t ret(into<dim_idx_t>(old_dim_idxs.size()));
        dim_idx_t new_dim_idx = 0;
        for (auto old_dim_idx : old_dim_idxs) {
            ret.dim(new_dim_idx) = dim(old_dim_idx);
            ret.off(new_dim_idx) = off(old_dim_idx);
            ret.idx(new_dim_idx) = idx(old_dim_idx);
            new_dim_idx++;
        }
        return ret;
    }

    grid_info_t resize(const std::vector<dim_t> &new_dims) const {
        grid_info_t ret = *this;
        ret.dims_ = new_dims;
        return ret;
    }

    grid_info_t slice(dim_idx_t dim_idx, dim_t new_off, dim_t new_dim,
            const expr_t &new_idx, expr_t &new_idx_value) const {
        gpu_assert(dim_idx >= 0 && dim_idx < ndims());
        gpu_assert(new_dim > 0 && new_off >= 0);
        gpu_assert(new_off + new_dim <= dims_[dim_idx]);

        grid_info_t ret = *this;
        ret.offs_[dim_idx] += new_off;
        ret.dims_[dim_idx] = new_dim;
        if (new_off > 0) {
            new_idx_value = ret.idxs_[dim_idx] - new_off;
            ret.idxs_[dim_idx] = new_idx;
        } else {
            new_idx_value = expr_t();
        }
        ret.parent_dims_ = (parent_dims_.empty() ? dims_ : parent_dims_);
        return ret;
    }

    grid_info_t halven(const expr_t &new_idx, dim_idx_t &dim_idx,
            expr_t &new_idx_value, bool first = true) const {
        for (int i = ndims() - 1; i >= 0; i--) {
            if (dim(i) == 1 || dim(i) % 2 != 0) continue;
            dim_idx = i;
            if (first) return slice(i, 0, dim(i) / 2, new_idx, new_idx_value);
            return slice(i, dim(i) / 2, dim(i) / 2, new_idx, new_idx_value);
        }
        return grid_info_t();
    }

    expr_t slice_condition() const {
        if (parent_dims_.empty()) return expr_t();
        expr_t ret(true);
        for (dim_idx_t i = 0; i < ndims(); i++) {
            auto &idx = idxs_[i];
            if (offs_[i] > 0) ret &= (idx >= 0);
            if (offs_[i] + dims_[i] < parent_dims_[i]) ret &= (idx < dims_[i]);
        }
        if (ret.is_equal(expr_t(true))) return expr_t();
        return ret;
    }

    std::string str() const {
        ostringstream_t oss;
        oss << ir_utils::make_seq_print_helper(dims_, " x ");
        return oss.str();
    }

    XE_DEFINE_DUMP()

private:
    static std::vector<expr_t> make_idxs(
            const std::string &(*genname)(int), int n) {
        std::vector<expr_t> ret;
        ret.reserve(n);
        for (int i = 0; i < n; i++)
            ret.push_back(var_t::make(dsl::type_t::s32(), genname(i)));
        return ret;
    }

    std::vector<dim_t> dims_;
    std::vector<dim_t> offs_;
    std::vector<expr_t> idxs_;

    std::vector<dim_t> parent_dims_;
};

class grid_splitter_t {
public:
    grid_splitter_t(const grid_info_t &grid)
        : grid_(grid), cur_idx_(grid.ndims() - 1), cur_stride_(1) {
        skip_size_1_dims();
        gpu_assert(cur_idx_ != dim_idx::invalid);
    }

    dim_t cur_block() const {
        if (is_empty()) return 1;

        return grid_.dim(cur_idx_) / cur_stride_;
    }

    bool is_empty() const { return cur_idx_ == dim_idx::invalid; }

    bool can_pop_block(dim_t size) const {
        if (is_empty()) return false;
        return cur_block() % size == 0;
    }

    expr_t pop_block(dim_t size);

private:
    void skip_size_1_dims() {
        while (cur_idx_ != dim_idx::invalid && grid_.dim(cur_idx_) == 1)
            cur_idx_--;
    }

    grid_info_t grid_;

    dim_idx_t cur_idx_;
    dim_t cur_stride_;
};

using layout_block_t = dsl::layout::block_t;
using layout_t = dsl::layout_t;

inline dim_t inner_block(const layout_t &layout, const pvar_t &idx,
        bool skip_outer = true, bool inner_only = true) {
    std::vector<dim_t> dim_blocks;
    for (auto &b : layout.blocks()) {
        if (b.idx == idx) dim_blocks.push_back(b.size);
    }
    dim_t ret = 1;
    int nblocks = (int)dim_blocks.size();
    int lo = 0;
    int hi = skip_outer ? nblocks - 1 : nblocks;
    if (inner_only) hi = std::min(hi, 1);
    for (int i = lo; i < hi; i++)
        ret *= dim_blocks[i];
    return ret;
}

// Storage size in bytes.
inline dim_t size_bytes(const layout_t &layout, dim_t alignment = 1) {
    if (layout.is_empty()) return 0;
    dim_t max_off = 0;
    dim_t max_block_size = 0;
    for (auto &b : layout.blocks()) {
        max_off += (b.size - 1) * (dim_t)b.stride;
        max_block_size = std::max(max_block_size, b.size * (dim_t)b.stride);
    }
    dim_t max_elems = std::max(max_off + 1, max_block_size);
    return utils::rnd_up(
            max_elems * layout.type().size() / layout.type().packing(),
            alignment);
}

template <typename T = expr_t>
T offset_bytes(const layout_t &layout, const coord_t &coord = {},
        bool ignore_offset = false) {
    return layout.offset<T>(coord, ignore_offset) * layout.type().size()
            / layout.type().packing();
}

inline layout_t make_strided(
        const layout_t &layout, int _stride, int block_idx = 0) {
    auto new_blocks = layout.blocks();
    int64_t factor = 1;
    for (int i = 0; i < (int)new_blocks.size(); i++) {
        auto &b = new_blocks[i];
        if (i == block_idx) {
            auto i_stride = int64_t(b.stride);
            if (_stride % i_stride == 0) {
                factor = (_stride / i_stride);
            } else if (i_stride % _stride == 0) {
                factor = -(i_stride / _stride);
            } else {
                gpu_error_not_expected();
            }
        }
        if (factor > 0) {
            b.stride *= factor;
        } else {
            b.stride = ir_utils::safe_divide(int64_t(b.stride), -factor);
        }
    }
    return layout.with(new_blocks);
}

tile_coord_t split(const layout_t &layout, const grid_info_t &grid_info,
        grid_info_t *out_grid = nullptr);
tile_coord_t split_exact(const layout_t &layout, const grid_info_t &grid);
tile_coord_t split_exact(const layout_t &layout, int factor);
tile_coord_t split(const layout_t &layout, const tile_t &tile,
        const grid_info_t &grid,
        std::vector<layout_block_t> *outer_blocks = nullptr);

void align_layouts(layout_t &a, layout_t &b);

memory_desc_t to_md(const layout_t &layout, const memory_desc_t &md_hint);

// Helper class to incrementally increase a sub-layout of the given layout.
// One step - adding the minimal factor of the next remaining block. Used
// to find the minimal tile between two layouts that is innermost for both
// layouts.
class layout_iterator_t {
public:
    layout_iterator_t(const layout_t &l) : l_(l), block_idx_(-1), block_(1) {}

    bool has_next() const {
        dim_t b = block_;
        int b_idx = block_idx_;
        while (b == 1) {
            b_idx++;
            if (b_idx >= int(l_.blocks().size())) return false;
            b = int(l_[b_idx].size);
        }
        return true;
    }

    layout_iterator_t &operator++() {
        gpu_assert(has_next());
        while (block_ == 1) {
            block_idx_++;
            block_ = int(l_[block_idx_].size);
        }
        // Find smallest factor.
        for (int factor = 2; factor <= int(block_); factor++) {
            if (block_ % factor == 0) {
                block_ /= factor;
                return *this;
            }
        }

        gpu_error_not_expected();
        return *this;
    }

    tile_t tile() const {
        tile_t ret;
        for (int i = 0; i <= block_idx_; i++) {
            auto &b = l_[i];
            dim_t b_block = b.size;
            if (i == block_idx_) b_block /= block_;
            ret[b.idx] *= b_block;
        }
        return ret;
    }

    int nblocks() const { return block_idx_ + 1; }

    layout_t outer_layout() const {
        auto &blocks = l_.blocks();
        std::vector<layout_block_t> outer_blocks;
        if (block_ > 1) {
            auto &b = blocks[block_idx_];
            outer_blocks.push_back(b);
            outer_blocks[0].size = block_;
            outer_blocks[0].stride = b.stride * (b.size / block_);
        }
        outer_blocks.insert(outer_blocks.end(),
                blocks.begin() + (block_idx_ + 1), blocks.end());
        return l_.with(outer_blocks);
    }

private:
    const layout_t &l_;

    int block_idx_;
    dim_t block_;
};

// Some complex expressions need more than one simplify() call.
inline expr_t simplify_repeat(
        const expr_t &e, const constraint_set_t &cset, int max_tries = 5) {
    auto ret = e;
    auto next = ir::simplify(ret, cset);
    for (int i = 0; i < max_tries; i++) {
        ret = next;
        next = ir::simplify(next, cset);
        if (next.is_equal(ret)) break;
    }
    return ret;
}

class mask_tensor_t {
public:
    mask_tensor_t() = default;

    mask_tensor_t(const layout_t &layout)
        : layout_(layout), masks_(layout.elems(), -1) {
        gpu_assert(layout.is_dense());
    }

    mask_tensor_t(const layout_t &layout, const std::vector<int> &masks,
            const object_eq_map_t<expr_t, int> &mask2ids,
            const std::vector<expr_t> &id2masks)
        : layout_(layout)
        , masks_(masks)
        , mask2ids_(mask2ids)
        , id2masks_(id2masks) {
        gpu_assert(int(masks.size()) == elems()) << "Incompatible size.";
    }

    const dsl::type_t &type() const { return layout_.type(); }

    const layout_t &layout() const { return layout_; }

    dim_t elems() const { return layout_.elems(); }

    void set_mask(dim_t off, const expr_t &mask) {
        gpu_assert(0 <= off && off < elems()) << "Incorrect offset.";
        if (mask.is_empty()) return;

        auto ret = mask2ids_.insert({mask, int(mask2ids_.size())});
        int id = ret.first->second;
        masks_[off] = id;

        if (ret.second) id2masks_.push_back(mask);
    }

    const expr_t &mask(dim_t off) const {
        gpu_assert(0 <= off && off < elems());
        return id2masks_[masks_[off]];
    }

    void simplify(const constraint_set_t &cset) {
        for (auto &mask : id2masks_)
            mask = simplify_repeat(mask, cset);
        mask2ids_.clear();
        for (int i = 0; i < int(id2masks_.size()); i++) {
            auto ret = mask2ids_.insert({id2masks_[i], i});
            if (!ret.second) {
                for (auto &m : masks_)
                    if (m == i) m = ret.first->second;
            }
        }
    }

    mask_tensor_t sub(const tile_t &tile, const coord_t &start) const {
        coord_t tile_start(start);
        auto sub_layout = layout_.sub(tile);
        mask_tensor_t sub_mask(sub_layout);
        for_each(tile, [&](const icoord_t &sub_start) {
            dim_t sub_off = sub_layout.offset<dim_t>(sub_start);
            dim_t off = layout_.offset<dim_t>(tile_start)
                    + layout_.offset<dim_t>(sub_start);
            sub_mask.set_mask(sub_off, mask(off));
        });
        return sub_mask;
    }

    mask_tensor_t reinterpret(const dsl::type_t &new_type) const {
        gpu_assert(!is_empty()) << "Can't reinterpret.";
        dim_t bytes = elems() * type().size();
        if (bytes % new_type.size() != 0 && bytes > new_type.size())
            return mask_tensor_t();
        int new_mask_size = std::max((int)(bytes / new_type.size()), 1);
        std::vector<int> new_masks(new_mask_size);
        for (dim_t i = 0; i < bytes; i += new_type.size()) {
            int mask_id = std::numeric_limits<int>::max();
            for (int j = 0; j < new_type.size() && j < bytes; j++) {
                int cur_mask_id = masks_[(i + j) / type().size()];
                if (mask_id >= int(masks_.size())) {
                    mask_id = cur_mask_id;
                } else if (mask_id != cur_mask_id) {
                    // Mask is not consistent, can't reinterpret.
                    return mask_tensor_t();
                }
            }
            gpu_assert(0 <= mask_id && mask_id < int(masks_.size()));
            new_masks[i / new_type.size()] = mask_id;
        }
        dim_t new_elems = utils::div_up(bytes, new_type.size());
        layout_t _1d_layout(new_type, std::vector<dim_t> {new_elems});
        return mask_tensor_t(_1d_layout, new_masks, mask2ids_, id2masks_);
    }

    expr_t to_expr(dim_t nmasks) const {
        if (elems() % nmasks != 0) return expr_t();

        std::vector<expr_t> vec(nmasks);
        for (dim_t i = 0; i < elems(); i++) {
            auto &channel_mask = vec[i % nmasks];
            auto &cur_mask = id2masks_[masks_[i]];
            if (channel_mask.is_empty()) {
                channel_mask = cur_mask;
                continue;
            }
            if (!channel_mask.is_equal(cur_mask)) return expr_t();
        }
        auto e = shuffle_t::make(vec);
        e = ir::simplify(e);
        e = ir::simplify_propagate_shuffle(e);
        return e;
    }

    bool is_empty() const { return layout_.is_empty(); }

    std::string str() const {
        ostringstream_t oss;
        for (int i = 0; i < int(elems()); i++) {
            if (i != 0) oss << std::endl;
            oss << "mask #" << i << ": ";
            if (masks_[i] == -1) {
                oss << "(nil)";
            } else {
                oss << id2masks_[masks_[i]];
            }
        }
        return oss.str();
    }

    XE_DEFINE_DUMP()

private:
    layout_t layout_;
    std::vector<int> masks_;

    object_eq_map_t<expr_t, int> mask2ids_;
    std::vector<expr_t> id2masks_;
};

class tdim_t {
public:
    tdim_t() = default;

    tdim_t(const expr_t &expr, const expr_t &mask) : expr_(expr), mask_(mask) {}

    dim_idx_t nvargs() const { return nvargs_; }

    const expr_t &expr() const { return expr_; }

    const expr_t &mask() const { return mask_; }

    void set_mask(const expr_t &value) { mask_ = value; }

    expr_t mask(const expr_t &tvalue, const std::vector<expr_t> &vvars,
            const coord_t &vvalues) const {
        auto ret = substitute(mask_, placeholder_var(), tvalue);
        for (dim_idx_t i = 0; i < vvars.size(); i++) {
            if (contains_object(ret, vvars[i])) {
                ret = substitute(ret, vvars[i], vvalues[i]);
            }
        }
        return ret;
    }

    dim_idx_t vidx(dim_idx_t arg_idx) const {
        gpu_assert(arg_idx < nvargs());
        return vidxs_[arg_idx];
    }

    stride_t vstride(dim_idx_t arg_idx) const {
        gpu_assert(arg_idx < nvargs());
        return vstrides_[arg_idx];
    }

    bool is_empty() const { return expr_.is_empty(); }

    bool is_identity() const { return is_var(expr_); }

    bool is_fixed_stride(dim_idx_t arg_idx) const {
        gpu_assert(arg_idx < nvargs());
        return vstrides_[arg_idx].is_fixed();
    }

    void add_vvar(dim_idx_t vidx, const expr_t &varg) {
        gpu_assert(nvargs_ + 1 <= max_nvargs);
        vidxs_[nvargs_] = vidx;
        vstrides_[nvargs_] = compute_stride(expr_, nvargs_, varg);
        nvargs_++;
    }

    static const expr_t &placeholder_var() {
        static thread_local expr_t ph_var
                = var_t::make(dsl::type_t::s32(), "_ph");
        return ph_var;
    }

    std::string str() const {
        ostringstream_t oss;
        oss << expr_;
        if (mask_) oss << " mask: " << mask_;
        return oss.str();
    }

    XE_DEFINE_DUMP()

private:
    static const dim_idx_t max_nvargs = 2;

    static stride_t compute_stride(
            const expr_t &e, dim_idx_t idx, const expr_t &var);

    expr_t expr_;

    dim_idx_t nvargs_ = 0;
    std::array<stride_t, max_nvargs> vstrides_;
    std::array<dim_idx_t, max_nvargs> vidxs_;
    expr_t mask_;
};

class view_t {
public:
    view_t() = default;

    view_t(const std::vector<expr_t> &vvars, dim_idx_t ntdims)
        : vvars_(vvars), vstart_(vvars.size()), tdims_(ntdims) {}

    // Constructs view from a layout.
    explicit view_t(const layout_t &layout,
            const std::vector<expr_t> &_vvars = {},
            uint32_t bound_check_mask = 0)
        : view_t(layout, _vvars, layout.tile(), bound_check_mask) {}

    view_t(const layout_t &layout, const std::vector<expr_t> &_vvars,
            const tile_t &_vdims, uint32_t bound_check_mask)
        : vvars_(_vvars)
        , vdims_(_vdims)
        , vstart_(layout.ndims())
        , tdims_(layout.ndims())
        , tlayout_(layout) {
        if (vvars_.empty())
            vvars_ = create_vvars(into<dim_idx_t>(layout.ndims()));
        for (dim_idx_t i = 0; i < nvdims(); i++) {
            expr_t i_mask;
            if ((bound_check_mask & (1 << i)) != 0)
                i_mask = (placeholder_var() < layout.elems(i));
            set_tdim(i, vvars_[i], i_mask);
        }
    }

    const std::vector<expr_t> &vvars() const { return vvars_; }

    const tile_t &vdims() const { return vdims_; }

    const coord_t &vstart() const { return vstart_; }

    tile_coord_t vtile_coord() const { return tile_coord_t(vdims_, vstart_); }

    const layout_t &tlayout() const { return tlayout_; }

    dim_idx_t nvdims() const { return into<dim_idx_t>(vvars_.size()); }

    dim_idx_t ntdims() const { return into<dim_idx_t>(tdims_.size()); }

    dim_t velems() const {
        dim_t ret = 1;
        for (dim_idx_t i = 0; i < nvdims(); i++)
            ret *= vdims_[i];
        return ret;
    }

    const expr_t &vvar(size_t idx) const {
        gpu_assert(idx < nvdims());
        return vvars_[idx];
    }

    const expr_t &vvar(const std::string &name) const {
        for (auto &v : vvars_)
            if (v.as<var_t>().name == name) return v;
        gpu_error_not_expected() << name;
        return vvars_[0];
    }

    const tdim_t &tdim(size_t idx) const {
        gpu_assert(idx < ntdims());
        return tdims_[idx];
    }

    void set_tdim(
            dim_idx_t tidx, const expr_t &_texpr, const expr_t &mask = {}) {
        gpu_assert(tdims_[tidx].is_empty());

        auto texpr = simplify(_texpr);

        tdim_t tdim(texpr, mask);
        for (dim_idx_t i = 0; i < nvdims(); i++) {
            if (contains_object(texpr, vvars_[i])) tdim.add_vvar(i, vvars_[i]);
        }
        if (!is_const(texpr)) {
            gpu_assert(tdim.nvargs() > 0)
                    << "Tensor dimension must have at least one view dimension "
                       "that maps to it.";
        }
        tdims_[tidx] = std::move(tdim);
    }

    void set_vdim(const expr_t &varg, dim_t vdim,
            const expr_t &vstart = expr_t(0), bool overwrite = false) {
        dim_idx_t vidx = vvar_index(varg);
        if (!overwrite) gpu_assert(vstart_[vidx].is(0));
        vstart_[vidx] = vstart;
        vdims_[vidx] = vdim;
    }

    void set_tlayout(const layout_t &tlayout) { tlayout_ = tlayout; }

    void set_tmasks(const std::unordered_map<std::string, dim_t> &padded_dims) {
        using namespace ir_utils;
        auto &x = placeholder_var();
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            auto &tdim = tdims_[i];
            if (!tdim.is_identity() || tdim.mask()) continue;
            dim_idx_t vidx = tdim.vidx(0);
            dim_t dim = tlayout_.elems(i);
            auto &dim_name = vvars_[vidx].as<var_t>().name;
            dim_t padded_dim = get_or_default(padded_dims, dim_name, dim_t(1));
            if (dim >= padded_dim) continue;
            dim_t inner_blk = ir_utils::max_pow2_divisor(dim);
            dim_t dim_blk = ir_utils::max_pow2_divisor(inner_block(
                    tlayout_, i, /*skip_outer=*/true, /*inner_only=*/false));
            inner_blk = std::min(inner_blk, dim_blk);
            auto tmask = (inner_blk == 1) ? (x < dim)
                                          : (x / inner_blk < dim / inner_blk);
            tdim.set_mask(tmask);
        }
    }

    void set_tmasks(const std::vector<dim_t> &padded_dims) {
        gpu_assert(padded_dims.size() == ntdims());
        std::unordered_map<std::string, dim_t> pd_map;
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            auto &dim_name = vvars_[tdims_[i].vidx(0)].as<var_t>().name;
            pd_map.emplace(dim_name, padded_dims[i]);
        }
        set_tmasks(pd_map);
    }

    std::string str() const {
        using ir_utils::operator<<;

        if (is_empty()) return "(nil)";
        ostringstream_t oss;
        oss << vdims_.str();
        if (!has_zero_vstart()) oss << " vstart: [" << vstart_ << "]";
        oss << " tlayout: " << tlayout_;
        return oss.str();
    }

    XE_DEFINE_DUMP()

    bool is_empty() const { return vdims_.is_empty(); }

    bool has_zero_vstart() const {
        for (dim_idx_t i = 0; i < nvdims(); i++)
            if (!vstart_[i].is(0)) return false;
        return true;
    }

    bool has_tmask(dim_idx_t tidx) const {
        gpu_assert(tidx != dim_idx::invalid && tidx < ntdims());
        return bool(tdims_[tidx].mask());
    }

    const dsl::type_t &type() const { return tlayout_.type(); }

    expr_t offset(const coord_t &vargs = {}, bool ignore_offset = false) const {
        auto targs = cvt_vargs_to_targs(vargs);
        return tlayout_.offset(targs, ignore_offset);
    }

    expr_t offset_bytes(
            const coord_t &vargs = {}, bool ignore_offset = false) const {
        return offset(vargs, ignore_offset) * type().size() / type().packing();
    }

    int get_alignment(const constraint_set_t &cset) const {
        // Alignment must be a power of 2.
        const dim_t base_alignment = 128;
        int64_t f = get_max_const_factor(this->offset_bytes(), cset);
        dim_t alignment = f ? ir_utils::max_pow2_divisor(f) : base_alignment;
        return static_cast<int>(std::min(base_alignment, alignment));
    }

    dim_idx_t vvar_index(const expr_t &vvar) const {
        for (dim_idx_t i = 0; i < vvars_.size(); i++)
            if (vvar.is_same(vvars_[i])) return i;
        gpu_error_not_expected() << "Can't find view dimension.";
        return dim_idx::invalid;
    }

    view_t create_sub_view(const tile_t &tile, const coord_t &coord) const;

    view_t create_sub_view(const tile_coord_t &tile_coord) const {
        return create_sub_view(tile_coord.tile, tile_coord.coord);
    }

    view_t retype(const dsl::type_t &new_type) const {
        auto ret = *this;
        ret.tlayout_ = tlayout_.with(new_type);
        return ret;
    }

    view_t make_dense() const {
        auto ret = *this;
        ret.tlayout_ = tlayout_.make_dense();
        return ret;
    }

    bool is_masked_vdim(dim_idx_t vidx) const {
        gpu_assert(vidx != dim_idx::invalid && vidx < nvdims());
        gpu_assert(has_zero_vstart())
                << "Can't be reliably determined if the view is a sub-view.";
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            auto &tdim = tdims_[i];
            if (tdim.expr().is_equal(vvars_[vidx])) {
                if (vdims_[vidx] != tlayout_.elems(i)) return true;
            }
            if (has_tmask(i)) {
                for (dim_idx_t j = 0; j < tdim.nvargs(); j++) {
                    if (tdim.vidx(j) == vidx) return true;
                }
            }
        }
        return false;
    }

    // Returns the mask corresponding to `vargs` view indices. The mask is
    // based on:
    // 1) combined tensor masks for the given indices
    // 2) Bounds-based masks for those view dimensions that are used directly
    //    in the tensor
    //    - Example: 32a layout when 'a' dimension is A < 32. In general it's
    //      fine to load/store elements with indices in the range [A, 31]
    //      assuming the zero padding invariant. However in some cases we need
    //      to generate the exact bound condition based on the logical indices.
    expr_t vmask(const coord_t &vargs) const {
        gpu_assert(vargs.size() == nvdims()) << "Incompatible dimensions.";
        gpu_assert(has_zero_vstart())
                << "Can't be reliably determined if the view is a sub-view.";
        auto targs = cvt_vargs_to_targs(vargs);
        auto mask = bool_imm_t::make(true);
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            for (dim_idx_t j = 0; j < nvdims(); j++) {
                if (!tdims_[i].expr().is_equal(vvars_[j])) continue;
                if (vdims_[j] != tlayout_.elems(i)) {
                    mask &= (vargs[j] < vdims_[j]);
                }
            }
            if (has_tmask(i)) {
                auto &tdim = tdims_[i];
                mask &= tdim.mask(targs[i], vvars_, vargs);
            }
        }
        return mask;
    }

    bool can_convert_to_vlayout() const {
        if (nvdims() != ntdims()) return false;
        for (dim_idx_t i = 0; i < nvdims(); i++) {
            if (!tdims_[i].expr().is_same(vvars_[i])) return false;
            if (!tdims_[i].is_fixed_stride(0)) return false;
        }
        return true;
    }

    // Returns the view-based layout constructed based on the view mapping from
    // the tensor layout to the view dimensions. In general such a layout may
    // include "unknown" strides which means it can't be used for offset
    // calculation.
    // However in many cases it's possible to construct a valid "view" layout
    // fully representative of the view and the underlying tensor layout.
    // Mainly it depends on whether each view dimension is a linear combination
    // of tensor layout dimensions.
    // If 1) init_offset is true and 2) the returned layout doesn't contain
    // "unknown" strides then it can be directly used for offset calculation.
    layout_t create_pseudo_vlayout(bool init_offset = false) const {
        return create_pseudo_vlayout(normalized_tlayout(), init_offset);
    }

    layout_t normalized_tlayout() const {
        auto blocks = move_size_1_blocks_outer();
        blocks = dsl::layout::merge_blocks(blocks);
        auto layout = tlayout_.with(blocks, false);
        return layout;
    }

    layout_t create_dense_vlayout() const {
        return create_pseudo_vlayout().make_dense();
    }

    layout_t create_vlayout(bool force_zero_offset = false) const {
        gpu_assert(can_convert_to_vlayout()) << "Can't convert view to layout.";
        if (force_zero_offset) return tlayout_.sub(vdims_);
        return tlayout_.sub(vdims_, vstart_);
    }

    dim_t vlayout_size() const { return size_bytes(create_vlayout()); }

    bool has_same_vlayout(
            const view_t &other, bool compare_offset = true) const {
        return create_vlayout().is_equal_normalized(
                other.create_vlayout(), compare_offset);
    }

    view_t split(const grid_info_t &grid, tile_coord_t &vtile_coord,
            grid_info_t *out_grid = nullptr) const {
        auto vlayout = create_pseudo_vlayout();
        vtile_coord
                = dnnl::impl::gpu::intel::jit::split(vlayout, grid, out_grid);
        return create_sub_view(vtile_coord.tile, vtile_coord.coord);
    }

    view_t split(
            const grid_info_t &grid, grid_info_t *out_grid = nullptr) const {
        tile_coord_t vtile_coord;
        return split(grid, vtile_coord, out_grid);
    }

    // Returns a tensor corresponding to the biggest innermost sub-layout so that
    // 1) It consists of consecutive blocks only.
    // 2) It contains less or equal than max_tile_elems elements.
    // 3) It is dense if is_dense_tile is true.
    tile_t split_into_max_tile(dim_t max_tile_elems, bool is_dense_tile) const {
        auto vlayout = create_pseudo_vlayout();
        return vlayout.max_subtile(max_tile_elems, is_dense_tile);
    }

    template <typename F>
    void for_each_tile(const tile_t &tile, const F &f) const {
        auto vlayout = create_dense_vlayout();
        vlayout.for_each_tile(tile, f);
    }

    view_t substitute(const expr_t &from, const expr_t &to) const;

    mask_tensor_t create_mask_tensor(
            const constraint_set_t &cset, uint32_t tmask = 0xFFFFFFFF) const {
        auto _vlayout = create_dense_vlayout();
        mask_tensor_t mask_tensor(_vlayout);
        icoord_t vargs(nvdims());
        // Distinct per-tdim mask factors are far fewer than elements.
        std::vector<object_eq_map_t<expr_t, expr_t>> memo(ntdims());
        create_mask_tensor(mask_tensor, _vlayout, 0, vargs, tmask, cset, memo);
        return mask_tensor;
    }

    void try_create_buffer_view(view_t &buf_view, view_t &inv_view) const {
        buf_view = view_t(create_vvars(ntdims()), ntdims());
        inv_view = view_t(vvars(), ntdims());
        for (dim_idx_t i = 0; i < nvdims(); i++) {
            inv_view.set_vdim(vvars()[i], vdims()[i]);
        }
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            auto &tdim = tdims_[i];
            auto &buf_vvar = buf_view.vvars()[i];
            if (tdim.is_identity()) {
                dim_idx_t vidx = tdim.vidx(0);
                buf_view.set_vdim(buf_vvar, vdims()[vidx], vstart()[vidx]);
                buf_view.set_tdim(i, buf_vvar, tdim.mask());
                inv_view.set_tdim(i, tdim.expr());
                continue;
            }
            int64_t buf_vdim = 0;
            bool ok = true;
            for (dim_idx_t j = 0; j < tdim.nvargs(); j++) {
                dim_idx_t vidx = tdim.vidx(j);
                auto &vvar = vvars()[vidx];
                dim_t vdim = vdims()[vidx];
                if (vdim == 1) continue;
                const auto &A = tdim.expr();
                auto B = ir::substitute(A, vvar, vvar + 1);
                auto C = simplify(B - A);
                if (!is_const(C)) {
                    ok = false;
                    break;
                }
                buf_vdim += to_cpp<int64_t>(C) * (vdim - 1);
            }
            buf_vdim++;

            if (!ok) {
                buf_view = view_t();
                inv_view = view_t();
                return;
            }

            auto buf_vstart = tdim.expr();
            auto inv_vstart = tdim.expr();
            for (dim_idx_t j = 0; j < tdim.nvargs(); j++) {
                dim_idx_t vidx = tdim.vidx(j);
                buf_vstart = ir::substitute(
                        buf_vstart, vvars()[vidx], vstart()[vidx]);
                inv_vstart
                        = ir::substitute(inv_vstart, vvars()[vidx], expr_t(0));
            }
            buf_vstart = simplify(buf_vstart);
            inv_vstart = simplify(inv_vstart);

            if (!is_const(inv_vstart)) {
                buf_view = view_t();
                inv_view = view_t();
                return;
            }

            buf_view.set_vdim(buf_vvar, buf_vdim, buf_vstart);

            // Check that mask doesn't contain vvars - they can't be accessed
            // in the buffered view.
            auto &tmask = tdim.mask();
            for (auto &vvar : vvars()) {
                if (contains_object(tmask, vvar)) {
                    buf_view = view_t();
                    inv_view = view_t();
                    return;
                }
            }

            buf_view.set_tdim(i, buf_vvar, tmask);
            inv_view.set_tdim(i, tdim.expr() - inv_vstart);
        }
        buf_view.set_tlayout(tlayout_);
    }

    static const expr_t &placeholder_var() { return tdim_t::placeholder_var(); }

    static std::vector<expr_t> create_vvars(dim_idx_t nvdims);

    coord_t cvt_vargs_to_targs(
            coord_t vcoord = {}, bool ignore_vstart = false) const {
        if (vcoord.is_empty()) vcoord = coord_t(nvdims());

        if (!ignore_vstart) {
            for (dim_idx_t i = 0; i < nvdims(); i++) {
                if (!vstart_[i].is(0)) vcoord[i] += vstart_[i];
            }
        }

        coord_t tcoord(ntdims());
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            tcoord[i] = tdims_[i].expr();
            for (dim_idx_t j = 0; j < nvdims(); j++) {
                tcoord[i] = ir::substitute(tcoord[i], vvars_[j], vcoord[j]);
            }
        }
        for (dim_idx_t i = 0; i < ntdims(); i++) {
            tcoord[i] = const_fold(tcoord[i]);
        }
        return tcoord;
    }

private:
    layout_t create_pseudo_vlayout(
            const layout_t &tlayout, bool init_offset) const;

    void create_mask_tensor(mask_tensor_t &mask_tensor,
            const layout_t &_vlayout, dim_idx_t vidx, icoord_t &vargs,
            uint32_t tmask, const constraint_set_t &cset,
            std::vector<object_eq_map_t<expr_t, expr_t>> &memo) const {
        if (vidx == _vlayout.ndims()) {
            bool is_init = false;
            coord_t vvalues;
            coord_t targs;
            expr_t mask;
            for (dim_idx_t i = 0; i < ntdims(); i++) {
                auto &tdim = tdims_[i];
                if ((tmask & (1 << i)) == 0) continue;
                if (tdim.mask().is_empty()) continue;
                if (!is_init) {
                    // Lazily initialize values
                    vvalues = vstart_.values();
                    for (dim_idx_t i = 0; i < nvdims(); i++)
                        vvalues[i] += vargs[i];
                    targs = cvt_vargs_to_targs(vargs);
                    is_init = true;
                }
                auto f = tdim.mask(targs[i], vvars_, vvalues);
                expr_t fs;
                auto it = memo[i].find(f);
                if (it != memo[i].end()) {
                    fs = it->second;
                } else {
                    fs = simplify_repeat(f, cset);
                    memo[i].emplace(f, fs);
                }
                // Fold the conjunction as simplify_rewrite_and() would.
                if (auto *imm = fs.as_ptr<bool_imm_t>()) {
                    if (imm->value) continue;
                    mask = fs;
                    break;
                }
                mask = mask.is_empty() ? fs : (mask & fs);
            }
            if (mask.is_empty()) mask = bool_imm_t::make(true);
            mask_tensor.set_mask(_vlayout.offset<dim_t>(vargs), mask);
            return;
        }

        for (dim_idx_t i = 0; i < vdims()[vidx]; i++) {
            vargs[vidx] = i;
            create_mask_tensor(
                    mask_tensor, _vlayout, vidx + 1, vargs, tmask, cset, memo);
        }
    }

    bool all_vdims_are_1(const pvar_t &tidx) const {
        auto &tinfo = tdims_[tidx];
        for (dim_idx_t i = 0; i < tinfo.nvargs(); i++) {
            if (vdims_[tinfo.vidx(i)] != 1) return false;
        }
        return true;
    }

    std::vector<layout_block_t> move_size_1_blocks_outer() const {
        std::vector<layout_block_t> new_blocks;
        std::vector<layout_block_t> size_1_blocks;
        for (auto &b : tlayout_.blocks()) {
            if (b.size == 1 && all_vdims_are_1(b.idx)) {
                size_1_blocks.emplace_back(b);
            } else {
                new_blocks.emplace_back(b);
            }
        }
        stride_t stride = new_blocks.empty()
                ? stride_t(1)
                : new_blocks.back().size * new_blocks.back().stride;
        for (auto &b : size_1_blocks) {
            b.stride = stride;
            new_blocks.emplace_back(b);
        }
        return new_blocks;
    }

    std::vector<expr_t> vvars_;
    tile_t vdims_;
    coord_t vstart_;

    std::vector<tdim_t> tdims_;
    layout_t tlayout_;
};

class dim_assignment_t {
public:
    dim_assignment_t() = default;

    dim_assignment_t(size_t old_ndims, size_t new_ndims)
        : old_ndims_(old_ndims)
        , new_ndims_(new_ndims)
        , assignments_(old_ndims, dim_idx::invalid) {}

    void assign(size_t old_idx, size_t new_idx) {
        gpu_assert(old_idx != dim_idx::invalid && old_idx < old_ndims_);
        gpu_assert(new_idx != dim_idx::invalid && new_idx < new_ndims_);
        assignments_[old_idx] = new_idx;
    }

    void assign(const std::vector<size_t> &old_idxes, size_t new_idx) {
        for (auto old_idx : old_idxes) {
            assign(old_idx, new_idx);
        }
    }

    size_t operator[](size_t old_idx) const {
        gpu_assert(old_idx >= 0 && old_idx < old_ndims());
        return assignments_[old_idx];
    }

    size_t old_ndims() const { return old_ndims_; }

    size_t new_ndims() const { return new_ndims_; }

    bool is_empty() const { return old_ndims_ == 0 && new_ndims_ == 0; }

    layout_t map(const layout_t &layout) const;

private:
    size_t old_ndims_ = 0;
    size_t new_ndims_ = 0;

    // assignments_[old_idx] = new_idx.
    std::vector<size_t> assignments_;
};

// Adds size one spatial dimensions according to input parameters. Spatial
// dimensions are assumed to be the last dimensions.
layout_t spatials_to_3d(const layout_t &layout, bool with_groups,
        const std::array<int, 3> &dhw_map);

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif
