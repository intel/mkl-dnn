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

#include "gemmstone/dsl/partition.hpp"
#include "dsl/ir/ir.hpp"
#include "dsl/utils/utils.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

expr_t subgroup_range_t::has(const expr_t &sg_id) const {
    using namespace ir;
    if (count == 1) return sg_id == start;
    return (sg_id >= start) & (sg_id < start + count);
}

expr_t subgroup_range_t::index(const expr_t &sg_id) const {
    using namespace ir;
    return sg_id - start;
}

void tile_by(const layout_t &layout, int64_t inner_elems,
        layout_t *inner = nullptr, layout_t *outer = nullptr) {
    int64_t cur_inner_elems = 1;
    std::vector<layout::block_t> inner_blocks;
    std::vector<layout::block_t> outer_blocks;
    for (size_t i = 0; i < layout.nblocks(); i++) {
        auto b = layout.blocks()[i];
        if (cur_inner_elems < inner_elems) {
            int64_t take = std::min(inner_elems / cur_inner_elems, b.size);
            if (inner) inner_blocks.emplace_back(b.idx, take);
            cur_inner_elems *= take;
            b.size = utils::safe_divide(b.size, take);
        }
        if (outer && b.size > 1) outer_blocks.emplace_back(b.idx, b.size);
    }
    if (inner) *inner = layout_t(layout.type(), inner_blocks);
    if (outer) *outer = layout_t(layout.type(), outer_blocks);
}

layout_t tile_by(const layout_t &layout, int64_t tile_elems) {
    layout_t inner;
    tile_by(layout, tile_elems, &inner, nullptr);
    return inner;
}

std::pair<tile_t, coord_t> split_to(
        const layout_t &layout, int64_t size, expr_t id) {
    dsl_assert(layout.nblocks() > 0);
    dsl_assert(layout.elems() % size == 0);
    layout_t inner, outer;
    tile_by(layout, layout.elems() / size, &inner, &outer);
    auto tile = inner.tile();
    auto rem_tile = layout.tile();
    int64_t rem_size = size;
    coord_t coord;
    for (auto &b : outer.blocks()) {
        rem_tile[b.idx] /= b.size;
        rem_size /= b.size;
        auto cur_idx = (rem_size == 1 ? id : id % b.size);
        coord[b.idx] = coord.get(b.idx) + cur_idx * rem_tile[b.idx];
        id /= b.size;
    }
    return std::make_pair(tile, coord);
}

namespace detail {

std::pair<tile_t, coord_t> for_each_subgroup_split_layout(
        const expr_t &subgroup_id, const subgroup_range_t &sg_range,
        const layout_t &layout) {
    // TODO: Add using aliases for binary operators.
    using namespace ir;
    auto tmp_id
            = def("tmp_id", s32, ir::cast(subgroup_id, s32) - sg_range.start);
    return split_to(layout, sg_range.count, tmp_id);
}

std::pair<tile_t, coord_t> for_each_subgroup_split_layout(
        const expr_t &subgroup_id, const subgroup_range_t &sg_range,
        const layout_t &layout, expr_t &condition) {
    // TODO: Add using aliases for binary operators.
    using namespace ir;
    auto tmp_id
            = def("tmp_id", s32, ir::cast(subgroup_id, s32) - sg_range.start);
    condition = (tmp_id >= 0) & (tmp_id < sg_range.count);
    return split_to(layout, sg_range.count, tmp_id);
}

} // namespace detail

} // namespace dsl
GEMMSTONE_NAMESPACE_END
