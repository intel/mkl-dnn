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

#include "dsl/ir/reorder.hpp"
#include "gemmstone/dsl/tensor.hpp"

#include "dsl/utils/range.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

struct normalization_stage_t {
    int idx;
    layout::block_t curr, last;
    std::array<int64_t, 2> tile;

    bool is_dense() const { return curr.stride == last.stride * last.size; }

    int64_t elems() const { return tile[0]; }

    normalization_stage_t() = default;
    normalization_stage_t(int idx, const layout::block_t &curr,
            const layout::block_t &last, std::vector<int64_t> tile)
        : idx(idx)
        , curr(curr)
        , last(last)
        , tile({tile[curr.idx], tile[last.idx]}) {}
};

struct merge_info_t {
    enum class merge_direction_t { none = 0, forward, backward };

    int iter_idx;
    merge_direction_t direction;

    merge_info_t(int iter_idx, merge_direction_t direction)
        : iter_idx(iter_idx), direction(direction) {}
};

merge_info_t::merge_direction_t merge_direction(
        const normalization_stage_t &l, const normalization_stage_t &r) {
    using direction_t = merge_info_t::merge_direction_t;
    if (l.curr.idx != r.curr.idx) return direction_t::none;
    if (l.last.idx != r.last.idx) return direction_t::none;
    if (l.tile[0] != r.tile[0]) return direction_t::none;
    if (l.curr.size == r.curr.size
            && l.tile[1] * l.last.size == r.tile[1] * r.last.size)
        return direction_t::backward;
    if (l.tile[1] == r.tile[1] && l.last.size == r.last.size)
        return direction_t::forward;
    return direction_t::none;
}

struct layout_normalization_t {
    using blocks_t = std::vector<layout::block_t>;
    using block_iterator_t = typename blocks_t::const_iterator;
    using stage_t = normalization_stage_t;

    struct iterator_t {
        bool operator==(const iterator_t &o) const { return curr_ == o.curr_; }
        bool operator!=(const iterator_t &o) const { return !operator==(o); }
        stage_t operator*() const { return {idx_, *curr_, *last_, tile_}; }
        iterator_t &operator++() {
            if (curr_ == end_) return *this;
            auto blk = *last_;
            tile_[blk.idx] *= blk.size;
            last_ = curr_;
            ++curr_;
            ++idx_;
            return *this;
        }

        iterator_t(size_t ndims, block_iterator_t it, block_iterator_t end)
            : curr_(it == end ? end : it + 1)
            , last_(it)
            , end_(end)
            , idx_(0)
            , tile_(ndims, 1) {}

    private:
        block_iterator_t curr_, last_, end_;
        int idx_;
        std::vector<int64_t> tile_;
    };

    size_t ndims() const { return ndims_; }
    const blocks_t &blocks() const { return blocks_; }

    bool empty() const { return begin() == end(); }
    bool contains_dim(const idx_t &idx) const {
        for (auto &blk : blocks_)
            if (blk.idx == idx) return true;
        return false;
    }

    void merge(std::vector<merge_info_t> merges) {
        using direction_t = merge_info_t::merge_direction_t;
        if (empty()) {
            if (blocks_.empty()) blocks_.emplace_back(0, 1, 1);
            return;
        }

        std::sort(merges.begin(), merges.end(),
                [](const merge_info_t &l, const merge_info_t &r) {
            return l.iter_idx < r.iter_idx;
        });
        auto merge_it = merges.begin();
        auto merge_end = merges.end();
        std::vector<layout::block_t> blocks;
        layout::block_t last = (*begin()).last;
        for (auto s : *this) {
            if (merge_it != merge_end && merge_it->iter_idx == s.idx) {
                if (merge_it->direction == direction_t::backward)
                    s.curr.idx = last.idx;
                s.curr.size *= last.size;
                s.curr.stride = last.stride;
                ++merge_it;
            } else
                blocks.push_back(last);
            last = s.curr;
        }
        blocks.push_back(last);
        blocks_ = std::move(blocks);
    }

    void reindex(int ndims, const std::vector<int> &map) {
        ndims_ = ndims;
        for (auto &blk : blocks_)
            blk.idx = map[blk.idx];
    }

    layout_t layout() const {
        return {type_, blocks_, offset_, ndims_, /*do_normalize=*/false};
    }

    iterator_t begin() const {
        return {ndims_, blocks_.begin(), blocks_.end()};
    }
    iterator_t end() const { return {ndims_, blocks_.end(), blocks_.end()}; }

    layout_normalization_t(
            const layout_t &layout, const std::vector<bool> &dim_empty)
        : type_(layout.type())
        , ndims_(layout.ndims())
        , offset_(layout.offset())
        , blocks_(normalized_blocks(layout, dim_empty)) {}

private:
    static bool can_combine(
            const layout::block_t &last, const layout::block_t &next) {
        if (last.idx != next.idx) return false;
        if (last.stride * last.size != next.stride) return false;
        return true;
    }

    static std::vector<layout::block_t> normalized_blocks(
            const layout_t &layout, std::vector<bool> dim_empty) {
        std::vector<layout::block_t> normalized_blocks;
        for (auto &blk : layout.blocks()) {
            if (blk.size != 1
                    || (layout.is_outermost(blk) && !dim_empty[blk.idx])) {
                if (normalized_blocks.empty()
                        || !can_combine(normalized_blocks.back(), blk)) {
                    normalized_blocks.push_back(blk);
                    dim_empty[blk.idx] = true;
                } else {
                    normalized_blocks.back().size *= blk.size;
                }
            }
        }
        return normalized_blocks;
    }

    type_t type_;
    size_t ndims_;
    expr_t offset_;
    blocks_t blocks_;
};

// Given two layouts, finds an equivalent pair of simpler layouts by attempting
// to combine consecutive blocks that appear in both layouts at the same level
// of nesting for the dimensions to which the blocks belong. E.g.,
//
//             1.          2.
// 16a16b16c ---> 256a16c ---> 256a16b
// 16c16a16b ---> 16c256a ---> 16b256a
//
// 1. The consecutive blocks 16a16b are repeated. For the first layout it
//    appears with an inner tile 1x1x16, and 1x1x1 for the second. Because the
//    ab-subtile is 1x1 for both and  the inner block (16b) is the same for
//    both, we can combine these blocks.
// 2. The b dimension no longer appears, so we can remove it from the layout and
//    re-index the dimensions so that the new layouts are 2D.
void reorder_t::normalize(layout_t &a, layout_t &b) {
    using direction_t = merge_info_t::merge_direction_t;
    auto ndims = a.ndims();
    auto cmp = [](const normalization_stage_t &a,
                       const normalization_stage_t &b) {
        return a.elems() <= b.elems();
    };
    auto dim_blocks = [](const idx_t &idx) {
        return [=](const normalization_stage_t &s) {
            return s.curr.idx == idx;
        };
    };

    std::vector<bool> empty_dimension(ndims, true);
    for (auto &blk : a.blocks())
        if (blk.size != 1) empty_dimension[blk.idx] = false;
    for (auto &blk : b.blocks())
        if (blk.size != 1) empty_dimension[blk.idx] = false;

    layout_normalization_t a_normalization {a, empty_dimension};
    layout_normalization_t b_normalization {b, empty_dimension};

    std::vector<merge_info_t> a_merges;
    std::vector<merge_info_t> b_merges;
    // Find pairs of consecutive blocks which can be combined
    for (size_t i = 0; i < ndims; ++i) {
        auto dim_i_blocks = dim_blocks(i);
        auto a_stages = a_normalization | filter(dim_i_blocks);
        auto b_stages = b_normalization | filter(dim_i_blocks);
        for (auto p : merge(a_stages, b_stages, cmp)) {
            if (!p[0].is_dense() || !p[1].is_dense()) continue;
            direction_t direction = merge_direction(p[0], p[1]);
            if (direction == direction_t::none) continue;
            a_merges.emplace_back(p[0].idx, direction);
            b_merges.emplace_back(p[1].idx, direction);
        }
    }
    a_normalization.merge(std::move(a_merges));
    b_normalization.merge(std::move(b_merges));

    // Find dimensions present in either normalized layout and construct map of
    // new dimension indices
    int curr_dim = 0;
    std::vector<int> dim_map(ndims);
    for (size_t i = 0; i < ndims; ++i)
        if (a_normalization.contains_dim(i) || b_normalization.contains_dim(i))
            dim_map[i] = curr_dim++;
    a_normalization.reindex(curr_dim, dim_map);
    b_normalization.reindex(curr_dim, dim_map);

    a = a_normalization.layout();
    b = b_normalization.layout();
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
