/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "sum/sum.hpp"

namespace sum {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_sdt : s.sdt)
    for_(const auto &i_ddt : s.ddt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_input_scales : s.input_scales)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        auto prb = std::make_shared<prb_t>(s.prb_dims, i_sdt, i_ddt, i_stag,
                i_dtag, i_input_scales, i_inplace, i_attr, i_ctx_init,
                i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s) {
    for_(const auto &i_sdt : s.sdt)
    for (const auto &i_stag : s.stag) {
        const int n_inputs = static_cast<int>(i_sdt.size());
        const int n_stags = static_cast<int>(i_stag.size());
        if (n_stags != n_inputs && n_stags != 1) {
            BENCHDNN_PRINT(0,
                    "ERROR: Expected number of stag arguments is `1` or `%d`, "
                    "provided `%d`.\n",
                    n_inputs, n_stags);
            SAFE_V(FAIL);
        }
    }

    for_(const auto &i_sdt : s.sdt)
    for (const auto &i_input_scales : s.input_scales) {
        const int n_inputs = static_cast<int>(i_sdt.size());
        const int n_input_scales = static_cast<int>(i_input_scales.size());
        if (n_input_scales != n_inputs && n_input_scales != 1) {
            BENCHDNN_PRINT(0,
                    "ERROR: Expected number of scales arguments is `1` or "
                    "`%d`, provided `%d`.\n",
                    n_inputs, n_input_scales);
            SAFE_V(FAIL);
        }
    }

    return OK;
}

static const std::string help_scales
        = "FLOAT[:FLOAT...]    (Default: `1.f`)\n    Input scales for source "
          "values.\n    If a single value is specified, will be broadcasted "
          "for all sources, otherwise number of scales should match number of "
          "inputs.\n";

int bench(int argc, char **argv) {
    driver_name = "sum";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.sdt, def.sdt, argv[0])
                || parse_dt(s.ddt, def.ddt, argv[0], "ddt")
                || parse_multi_tag(s.stag, def.stag, argv[0])
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_multivector_option(s.input_scales, def.input_scales,
                        atof, argv[0], "scales", help_scales)
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_dims(s.prb_dims, argv[0]);

            SAFE(verify_input(s), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace sum
