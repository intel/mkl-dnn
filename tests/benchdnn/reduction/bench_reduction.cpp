/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "reduction/reduction.hpp"

namespace reduction {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_ddt : s.ddt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_p : s.p)
    for_(const auto &i_eps : s.eps)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for (const auto &i_ctx_exe : s.ctx_exe) {
        auto prb = std::make_shared<prb_t>(s.prb_vdims, i_sdt, i_ddt, i_stag,
                i_dtag, i_alg, i_p, i_eps, i_attr, i_ctx_init, i_ctx_exe,
                s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s) {
    // Expect exactly two inputs for problem dimensions.
    static constexpr int n_inputs = 2;

    if (s.prb_vdims.n_inputs() != n_inputs) {
        BENCHDNN_PRINT(0, "%s\n",
                "Error: input tensors were specified in wrong format. "
                "Please use NxNxNxNxN:MxMxMxMxM as a problem description "
                "format.");
        SAFE_V(FAIL);
    }

    return OK;
}

static const std::string help_p
        = "FLOAT    (Default: `1.f`)\n    Specifies algorithm parameter "
          "extension where applicable.\n";

static const std::string help_eps
        = "FLOAT    (Default: `0.f`)\n    Specifies algorithm parameter "
          "extension where applicable.\n";

int bench(int argc, char **argv) {
    driver_name = "reduction";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dt(s.sdt, def.sdt, argv[0], "sdt")
                || parse_dt(s.ddt, def.ddt, argv[0], "ddt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_alg(s.alg, def.alg, str2alg, argv[0])
                || parse_vector_option(s.p, def.p, atof, argv[0], "p", help_p)
                || parse_vector_option(
                        s.eps, def.eps, atof, argv[0], "eps", help_eps)
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0]);

            SAFE(verify_input(s), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace reduction
