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

#ifndef GEMMSTONE_DSL_PARTITION_IMPL_HPP
#define GEMMSTONE_DSL_PARTITION_IMPL_HPP

#include "gemmstone/dsl/dsl.hpp"
#include "gemmstone/dsl/tensor.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {

std::pair<tile_t, coord_t> split_to(
        const layout_t &layout, int64_t size, expr_t id);
layout_t tile_by(const layout_t &layout, int64_t tile_elems);

} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
