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

#ifndef GEMMSTONE_DSL_IR_PASS_TRACE_HPP
#define GEMMSTONE_DSL_IR_PASS_TRACE_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/dsl/kernel.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

class stmt_t;
class ir_context_t;

// Trace for debugging purposes.
#if GEMMSTONE_ASSERTIONS
void trace_start();
void trace_reset();
void trace_stamp(const char *pass_name);
void trace_stop(const char *pass_name);
void trace_perf();
void trace_pass(
        const char *pass_name, const stmt_t &stmt, const ir_context_t &ir_ctx);
void trace_pass(const char *pass_name, const kernel_t &kernel);
void trace_pass(
        const char *pass_name, const stmt_t &stmt, const ir_context_t &ir_ctx);
#else
inline void trace_start() {};
inline void trace_reset() {};
inline void trace_stamp(const char *) {};
inline void trace_stop(const char *) {};
inline void trace_perf() {};
inline void trace_pass(const char *pass_name, const kernel_t &kernel) {}
inline void trace_pass(const char *pass_name, const stmt_t &stmt,
        const ir_context_t &ir_ctx) {};
#endif

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END

#endif
