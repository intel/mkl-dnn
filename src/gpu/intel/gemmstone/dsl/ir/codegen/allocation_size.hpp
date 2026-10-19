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

#ifndef GEMMSTONE_DSL_IR_CODEGEN_ALLOCATION_SIZE_HPP
#define GEMMSTONE_DSL_IR_CODEGEN_ALLOCATION_SIZE_HPP

#include "gemmstone/../../dsl/ir/core.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

static const int ngen_alloc_granularity = 4;
inline int register_size(const alloc_t &obj, int grf_size) {
    return (obj.kind == alloc_kind_t::grf)
            ? into<int>(round_up(obj.size, grf_size))
            : 0;
}

inline int register_size(const let_t &obj) {
    // Empty objects are allocated in reserved space
    // nGEN only claims subregisters at dword granularity
    if (obj.value.is_empty()) return 0;
    return round_up(obj.var.type().size(), ngen_alloc_granularity);
}

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
