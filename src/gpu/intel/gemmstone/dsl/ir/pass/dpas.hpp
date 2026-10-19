/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#ifndef GEMMSTONE_DSL_IR_PASS_DPAS_HPP
#define GEMMSTONE_DSL_IR_PASS_DPAS_HPP

#include "gemmstone/dsl/ir/object.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

// Adds {Atomic} modifier to dpas/dpasw instructions when applicable.
stmt_t inject_dpas_atomic(const stmt_t &stmt, bool filter_by_label = true);
// Adds {Fwd} modifier to dpas/dpasw instructions when applicable.
stmt_t inject_dpas_fwd(const stmt_t &stmt);

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
