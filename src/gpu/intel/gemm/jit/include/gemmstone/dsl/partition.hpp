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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_DSL_PARTITION_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_DSL_PARTITION_HPP

#include "gemmstone/dsl/dsl.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

struct subgroup_range_t {
    int start = 0;
    int count = 0;

    subgroup_range_t() = default;
    subgroup_range_t(int start, int count) : start(start), count(count) {}

    // Check if subgroup ID is within this range.
    expr_t has(const expr_t &sg_id) const;

    // Get 0-based index within this range.
    expr_t index(const expr_t &sg_id) const;
};

namespace detail {
std::pair<tile_t, coord_t> for_each_subgroup_split_layout(
        const expr_t &subgroup_id, const subgroup_range_t &sg_range,
        const layout_t &layout);

std::pair<tile_t, coord_t> for_each_subgroup_split_layout(
        const expr_t &subgroup_id, const subgroup_range_t &sg_range,
        const layout_t &layout, expr_t &condition);

template <typename Func>
void for_each(const layout_t &layout, const tile_t &subtile, const Func &func) {
    for (auto &idx : layout.iter(subtile)) {
        func(idx);
    }
}
} // namespace detail

// Partitions the tensor's layout across the range of subgroups
// [start_subgroup_idx, stop_subgroup_idx). The partitioning scheme uses
// **outer layout blocks** to distribute work, ensuring each subgroup processes
// the **innermost portion** of the tensor's layout. It calculates the tile
// (same for all participating subgroups) and coordinate (dependent on subgroup
// ID), then calls `func(tile, coord)` to delegate handling of the subgroup's
// subtensor.
// Primary use case: partitioning an SLM layout within a workgroup to
// parallelize data loading and processing.
// `subgroup_dim_idx` specifies the ND dimension index for the subgroup range.
template <typename Func>
void for_each_subgroup(const subgroup_range_t &sg_range, const tensor_t &tensor,
        const Func &func, int subgroup_dim_idx = 0) {
    expr_t cond;
    auto region = detail::for_each_subgroup_split_layout(
            subgroup_id(subgroup_dim_idx), sg_range, tensor.layout(), cond);
    _if(cond, std::bind(func, region.first, region.second));
}

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
