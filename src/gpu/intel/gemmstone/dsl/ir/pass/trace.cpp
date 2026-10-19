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

#include "dsl/ir/pass/trace.hpp"

#include "dsl/ir/ir.hpp"
#include "dsl/utils/logging.hpp"
#include "dsl/utils/profiler.hpp"

GEMMSTONE_NAMESPACE_START
namespace dsl {
namespace ir {

#if GEMMSTONE_ASSERTIONS
profiler_t *get_trace_profiler() {
    static thread_local auto profiler = getVerbose(GEMMVerbose::DebugInfo)
                    >= static_cast<int>(log_level_t::perf)
            ? std::unique_ptr<profiler_t>(new profiler_t("Trace Profile"))
            : std::unique_ptr<profiler_t>(nullptr);
    return profiler.get();
}

void trace_start() {
    if (get_trace_profiler()) get_trace_profiler()->start();
}
void trace_reset() {
    if (get_trace_profiler()) get_trace_profiler()->reset();
}
void trace_stamp(const char *pass_name) {
    if (get_trace_profiler()) get_trace_profiler()->stamp(pass_name);
}
void trace_stop(const char *pass_name) {
    if (get_trace_profiler()) get_trace_profiler()->stop(pass_name);
}
void trace_perf() {
    if (get_trace_profiler()) dsl_perf() << *get_trace_profiler();
}

void trace_pass(const char *pass_name, const kernel_t &kernel) {
    trace_stop(pass_name);
    dsl_trace() << "=== After " << pass_name << "\n" << kernel;
}

void trace_pass(
        const char *pass_name, const stmt_t &stmt, const ir_context_t &ir_ctx) {
    trace_stop(pass_name);
    dsl_trace() << "=== After " << pass_name << "\n" << stmt;
    dsl_trace() << ir_ctx.cset();
}
#endif

} // namespace ir
} // namespace dsl
GEMMSTONE_NAMESPACE_END
