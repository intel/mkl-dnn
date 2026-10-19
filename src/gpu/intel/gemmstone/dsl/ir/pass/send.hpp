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

#ifndef GEMMSTONE_DSL_IR_PASS_SEND_HPP
#define GEMMSTONE_DSL_IR_PASS_SEND_HPP

#include "gemmstone/dsl/ir/object.hpp"
#include "gemmstone/dsl/kernel.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {
class ir_context_t;

stmt_t lift_buffer_offsets_in_send(const stmt_t &s, ir_context_t &ir_ctx);

void inject_send(kernel_t &k);
stmt_t inject_send(const stmt_t &s, ir_context_t &ir_ctx);

// Lifts loop-invariant header assignments related to block 2D messages.
stmt_t lift_send_2d_header_store(const stmt_t &s, ir_context_t &ir_ctx);

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
