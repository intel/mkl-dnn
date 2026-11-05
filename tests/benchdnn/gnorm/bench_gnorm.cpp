/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "bnorm/bnorm.hpp"
#include "gnorm/gnorm.hpp"

using namespace bnorm;

namespace gnorm {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_tag : s.tag)
    for_(const auto &i_flags : s.flags)
    for_(const auto &i_mb : s.mb)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        auto prb = std::make_shared<prb_t>(s.desc, i_dir, i_dt, i_tag, i_flags,
                s.check_alg, i_mb, i_inplace, i_attr, i_ctx_init, i_ctx_exe,
                s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s) {
    for_(const auto &i_scales : s.scales)
    for (const auto &e : i_scales.scales) {
        if (!e.second.has_single_element()) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: scales support only a single element input.");
            return FAIL;
        }
    }

    static constexpr int n_inputs = 2;
    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s%d%s%ld%s\n",
                    "ERROR: `dt` option expects either 1 or ", n_inputs,
                    " inputs in SRC:DST format. Current size is: \"",
                    (long)i_dt.size(), "\".");
            return FAIL;
        }
    }

    for (const auto &i_tag : s.tag) {
        if (i_tag.size() != 1 && i_tag.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s%d%s%ld%s\n",
                    "ERROR: `tag` option expects either 1 or ", n_inputs,
                    " inputs in SRC:DST format. Current size is: \"",
                    (long)i_tag.size(), "\".");
            return FAIL;
        }
    }
    return OK;
}

static const std::string help_flags
        = "FLAGS    (Default: not specified)\n    Specifies normalization "
          "flags. `FLAGS` values are:\n    * `G` for global_stats.\n    * `C` "
          "for scale.\n    * `H` for shift.\n";

int bench(int argc, char **argv) {
    driver_name = "gnorm";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_multi_tag(s.tag, def.tag, argv[0], "tag")
                || parse_vector_option(s.flags, def.flags, str2flags, argv[0],
                        "flags", help_flags)
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            SAFE(str2desc(&s.desc, argv[0]), CRIT);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace gnorm
