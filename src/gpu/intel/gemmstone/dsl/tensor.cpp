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

#include "gemmstone/dsl/tensor.hpp"
#include "dsl/ir/ir.hpp"
#include "dsl/ir/pass/simplify.hpp"

#include <algorithm>

GEMMSTONE_NAMESPACE_START
namespace dsl {

namespace layout {

std::vector<block_t> merge_blocks(const std::vector<block_t> &blocks) {
    if (blocks.empty()) return {};

    std::vector<block_t> res;
    for (const block_t &b : blocks) {
        auto can_merge = [](const block_t &a, const block_t &b) {
            return a.idx == b.idx && a.stride * a.size == b.stride;
        };
        if (!res.empty() && can_merge(res.back(), b)) {
            res.back().size *= b.size;
        } else {
            res.emplace_back(b);
        }
    }
    return res;
}

// Removes size-1 blocks and pre-merges stride-contiguous blocks sharing the
// same index, regardless of their position in the input.
std::vector<block_t> prepare_blocks(const std::vector<block_t> &blocks) {
    std::vector<block_t> sorted;
    sorted.reserve(blocks.size());
    for (auto &b : blocks) {
        if (b.size == 1) continue;
        sorted.push_back(b);
    }
    if (sorted.size() <= 1) return sorted;

    std::sort(sorted.begin(), sorted.end(),
            [](const block_t &a, const block_t &b) {
        if (a.idx != b.idx) return a.idx < b.idx;
        if (a.stride.is_fixed() != b.stride.is_fixed())
            return a.stride.is_fixed();
        if (!a.stride.is_fixed()) return false;
        return (int64_t)a.stride < (int64_t)b.stride;
    });

    std::vector<block_t> res;
    for (const auto &b : sorted) {
        if (!res.empty() && res.back().idx == b.idx
                && res.back().stride.is_fixed() && b.stride.is_fixed()
                && res.back().stride * res.back().size == b.stride) {
            res.back().size *= b.size;
        } else {
            res.push_back(b);
        }
    }
    return res;
}

std::vector<block_t> normalize_blocks(const std::vector<block_t> &_blocks) {
    if (_blocks.empty()) return {};

    auto blocks = prepare_blocks(_blocks);

    // Normalize blocks order: select min-stride block and put all preceding
    // same-index blocks before it.
    std::vector<block_t> ordered;
    ordered.reserve(blocks.size());
    std::vector<bool> used(blocks.size(), false);
    size_t remaining = blocks.size();

    auto less = [&](const layout::block_t &a, const layout::block_t &b) {
        auto prev_idx = (ordered.empty() ? idx_t() : ordered.back().idx);
        auto &as = a.stride;
        auto &bs = b.stride;
        dsl_assert(!as.is_undefined() && !bs.is_undefined());
        // Same stride -> prefer contiguous idx, otherwise order by idx.
        if (as == bs) {
            if (a.idx == b.idx) return false;
            if (a.idx == prev_idx) return true;
            if (b.idx == prev_idx) return false;
            return a.idx < b.idx;
        }
        // Fixed stride < unknown stride.
        if (as.is_unknown() != bs.is_unknown()) return bs.is_unknown();
        // Use stride values otherwise.
        return (int64_t)as < (int64_t)bs;
    };

    while (remaining > 0) {
        // Find min-stride block.
        size_t min_pos = blocks.size();
        for (size_t i = 0; i < blocks.size(); i++) {
            if (used[i]) continue;
            if (min_pos == blocks.size() || less(blocks[i], blocks[min_pos]))
                min_pos = i;
        }

        // Add all same-index blocks from start up to and including min_pos.
        idx_t target_idx = blocks[min_pos].idx;
        for (size_t i = 0; i <= min_pos; i++) {
            if (used[i] || blocks[i].idx != target_idx) continue;
            ordered.push_back(blocks[i]);
            used[i] = true;
            remaining--;
        }
    }

    return ordered;
}

tile_iterator_t &tile_iterator_t::operator++() {
    for (size_t i = 0; i < d_.size(); i++) {
        if (d_[i].i < d_[i].end - 1) {
            coord_[d_[i].idx] += d_[i].stride;
            d_[i].i++;
            return *this;
        }
        coord_[d_[i].idx] -= d_[i].i * d_[i].stride;
        d_[i].i = 0;
    }
    *this = end();
    return *this;
}

tile_iterator_t::tile_iterator_t(const layout_t &layout, const tile_t &tile) {
    tile_t strides;
    for (auto &b : layout.blocks()) {
        auto &stride = strides[b.idx];
        auto tile_dim = tile.get(b.idx);
        if (tile_dim == 0) {
            d_.clear();
            return;
        }
        d_.emplace_back(b.idx, b.size, stride, tile_dim);
        stride = d_.back().stride * d_.back().end;
        coord_[b.idx] = 0;
    }
    if (!layout.is_empty() && layout.blocks().empty()) {
        auto idx = tile.is_empty() ? idx_t() : *tile.begin();
        d_.emplace_back(idx, 1, 1, 1);
        coord_[idx] = 0;
    }
}

} // namespace layout

layout_t::layout_t(const type_t &type, const std::vector<int64_t> &dims,
        const expr_t &offset, bool do_normalize)
    : type_(type), ndims_(dims.size()), offset_(offset) {
    if (type.is_undef()) {
        *this = layout_t();
        return;
    }

    int64_t stride = 1;
    for (int64_t i = ndims_ - 1; i >= 0; i--) {
        blocks_.emplace_back(i, dims[i], stride);
        stride *= dims[i];
    }
    if (do_normalize) blocks_ = normalize_blocks(blocks_);
    sanity_check();
}

layout_t::layout_t(const type_t &type, const std::vector<block_t> &blocks,
        const expr_t &offset, size_t ndims, bool do_normalize)
    : type_(type), ndims_(ndims), offset_(offset), blocks_(blocks) {
    if (type.is_undef()) {
        *this = layout_t();
        return;
    }

    stride_t stride(1);
    for (auto &b : blocks_) {
        if (b.stride.is_undefined()) {
            b.stride = stride;
        } else {
            stride = b.size;
        }
        stride *= b.size;
    }
    if (do_normalize) blocks_ = normalize_blocks(blocks_);
    sanity_check();
}

layout_t layout_t::with_block(block_t block) const {
    auto new_blocks = blocks();
    if (block.stride.is_unknown()) {
        new_blocks.emplace_back(block);
    } else if (block.stride.is_undefined()) {
        block.stride = !new_blocks.empty()
                ? new_blocks.back().stride * new_blocks.back().size
                : stride_t(1);
        new_blocks.emplace_back(block);
    } else {
        auto it = new_blocks.begin();
        while (it != new_blocks.end() && it->stride <= block.stride) {
            it++;
        }
        new_blocks.insert(it, block);
    }

    auto ret = with(new_blocks);
    if (ret.has_ndims()) {
        if (block.idx.index() == ret.ndims()) ret.ndims_++;
        dsl_assert(has_ndims());
    }
    return ret;
}

template <typename T>
T layout_t::offset(const coord_t &args, bool ignore_offset) const {
    if (args.is_empty()) return ir::expr_cast<T>(ignore_offset ? 0 : offset_);

    expr_t off = 0;
    auto _args = args;
    for (auto &b : blocks()) {
        if (!_args.has(b.idx)) continue;
        auto &idx = _args[b.idx];
        if (idx.is(0)) continue;

        // Do not use modulus for outermost blocks.
        auto i = is_outermost(b) ? idx : (idx % b.size);
        off = i * int64_t(b.stride) + off;
        idx /= b.size;
    }
    if (ignore_offset) return ir::expr_cast<T>(off);
    return ir::expr_cast<T>(offset_ + off);
}

template expr_t layout_t::offset<expr_t>(
        const coord_t &args, bool ignore_offset) const;
template int layout_t::offset<int>(
        const coord_t &args, bool ignore_offset) const;
template int64_t layout_t::offset<int64_t>(
        const coord_t &args, bool ignore_offset) const;

bool layout_t::is_strictly_equal(const layout_t &other, bool compare_offset,
        bool compare_strides) const {
    if (type_ != other.type_) return false;
    if (compare_offset && !offset_.is_equal(other.offset_)) return false;
    if (blocks_.size() != other.blocks_.size()) return false;
    for (size_t i = 0; i < blocks_.size(); i++) {
        auto &b0 = blocks_[i];
        auto &b1 = other.blocks_[i];
        if (b0.idx != b1.idx) return false;
        if (b0.size != b1.size) return false;
        if (compare_strides && b0.stride != b1.stride) return false;
    }
    return true;
}

bool layout_t::operator<=(const layout_t &other) const {
    if (type_ != other.type_) return false;
    auto other_blocks = other.normalize().blocks();
    auto self_blocks = normalize().blocks();
    if (self_blocks.size() > other_blocks.size()) return false;
    if (self_blocks.empty()) return true;

    int i = 0;
    for (; i < (int)self_blocks.size() - 1; i++) {
        if (self_blocks[i] != other_blocks[i]) return false;
    }
    return (self_blocks[i].idx == other_blocks[i].idx
            && self_blocks[i].stride == other_blocks[i].stride
            && other_blocks[i].size % self_blocks[i].size == 0);
}

layout_t layout_t::sub(const tile_t &tile, const coord_t &start) const {
    auto remaining_tile = tile;
    std::vector<block_t> mapped_blocks;

    for (auto &b : blocks()) {
        bool b_is_outermost = is_outermost(b);

        int64_t size = b.size;
        if (!remaining_tile.has(b.idx)) remaining_tile[b.idx] = 1;
        int64_t &rem_dim = remaining_tile[b.idx];
        if (rem_dim == 1) {
            if (b_is_outermost) {
                // This is to have similarity between the current and
                // mapped layouts.
                mapped_blocks.emplace_back(b.idx, 1, b.stride);
            }
            continue;
        }
        if (b_is_outermost) {
            size = rem_dim;
        } else if (rem_dim % size != 0) {
            // Try to split the current block and start mapping from
            // scratch.
            if (size % rem_dim == 0)
                return split_block(b, rem_dim, size / rem_dim).sub(tile, start);

            // TODO: Remove exception usage.
            stub("Can't map tensor layout.");
        }
        rem_dim /= size;
        mapped_blocks.emplace_back(b.idx, size, b.stride);
    }

    return layout_t(type(), mapped_blocks,
            start.is_empty() ? offset_ : operator()(start), ndims_);
}

layout_t layout_t::split_block(
        const block_t &b, int64_t size0, int64_t size1) const {
    size_t block_idx = get_idx(b);
    dsl_assert(b.size == size0 * size1) << "Incompatible block sizes.";
    maybe_unused(b);

    auto new_blocks = blocks_;

    block_t &b0 = new_blocks[block_idx];
    block_t b1 = b0;

    b0.size = size0;
    b1.size = size1;
    b1.stride = b0.stride * size0;

    new_blocks.insert(new_blocks.begin() + block_idx + 1, b1);

    return with(new_blocks, false);
}

tile_t layout_t::max_subtile(
        int64_t max, bool is_dense, bool perfectly_divides) const {
    tile_t subtile;
    int64_t elems = 1;
    for (size_t i = 0; i < nblocks(); i++) {
        auto &b = blocks()[i];
        dsl_assert(!b.stride.is_undefined());
        if (is_dense) {
            if (b.stride.is_unknown()) return subtile;
            if (i > 0) {
                auto &b0 = blocks()[i - 1];
                if (b.stride != b0.size * b0.stride) break;
            }
        }
        if (b.size * elems >= max) {
            if (perfectly_divides)
                subtile[b.idx] *= utils::max_div(b.size, max / elems);
            else
                subtile[b.idx] *= max / elems;
            break;
        }
        subtile[b.idx] *= b.size;
        elems *= b.size;
    }
    return subtile;
}

std::string layout_t::desc_str(bool dnnl_style) const {
    if (is_empty()) return "(nil)";
    if (!dnnl_style && blocks_.empty()) return "(scalar:" + type().str() + ")";

    auto to_str = [](const idx_t &idx, bool is_outer) {
        auto ret = idx.str();
        if (ret.length() == 1) {
            if (is_outer) ret[0] -= 'a' - 'A';
            return ret;
        }
        return "<" + ret + ">";
    };
    std::string ret;
    stride_t dense_stride(1);
    idx_map_t<bool> seen;
    bool is_dense = true;
    for (auto &b : blocks()) {
        std::string b_str;
        if (dnnl_style && is_outermost(b)) {
            b_str += to_str(b.idx, seen.get(b.idx, false));
        } else {
            b_str = std::to_string(b.size);
            b_str += to_str(b.idx, false);
        }
        if (!dnnl_style) {
            if (b.stride.is_unknown()) {
                b_str.append(1, '?');
            } else if (b.stride != dense_stride) {
                b_str.append(1, '*');
                is_dense = false;
            }
        }
        b_str += ret;
        std::swap(ret, b_str);
        dense_stride = b.stride * b.size;
        seen[b.idx] = true;
    }
    if (!is_dense) {
        ret += ":";
        for (size_t i = nblocks(); i > 0; i--) {
            ret += std::to_string((int)blocks_[i - 1].stride);
            if (i != 1) ret += "x";
        }
    }
    ret += ":" + type().str();
    return ret;
}

void layout_t::sanity_check() const {
#if !defined(NDEBUG) || GEMMSTONE_ASSERTIONS
    for (size_t i = 0; i < blocks_.size(); i++) {
        dsl_assert(blocks_[i].size > 0) << "Incorrect block size.";
    }
    dsl_assert(has_ndims() || ndims_ == max_ndims);
#endif
}

expr_t tensor_t::subvec(const icoord_t &coord, int elems) const {
    int off = base_layout_.offset<int>(coord_ + coord);
    return ir::ref_t::make(buf(), off, elems);
}

expr_t global_tensor_t::offset(const coord_t &sub_coord) const {
    expr_t ret = base_offset_ ? base_offset_ : 0;
    for (auto &c : strides_) {
        ret += (coord_.get(c) + sub_coord.get(c)) * strides_[c];
    }
    return simplify(ret);
}

expr_t global_tensor_t::buf_u64(const expr_t &off) const {
    auto ret = cast(u64, buf_);
    if (base_offset_)
        ret += base_offset_ * scalar_type().size() / scalar_type().packing();
    if (off) ret += off * scalar_type().size() / scalar_type().packing();
    return ret;
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END
