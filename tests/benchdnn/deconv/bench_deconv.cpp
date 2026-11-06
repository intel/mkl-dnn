/*******************************************************************************
* Copyright 2018 Intel Corporation
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
#include <string.h>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/task_executor.hpp"

#include "deconv/deconv.hpp"

namespace deconv {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_bia_dt_ : s.bia_dt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (const auto &i_mb : s.mb) {
        auto i_bia_dt = i_bia_dt_;
        if (i_dir & FLAG_BIA) {
            if (i_bia_dt != dnnl_data_type_undef) {
                BENCHDNN_PRINT(0, "%s\n",
                        "Warning: `--dir=FWD_B,BWD_WB` options are "
                        "incompatible with `--bia-dt` option. To specify a "
                        "bias data type, use `--dir=FWD_D,FWD_I,BWD_W` values "
                        "intead.");
            }
            // The f32/f64 data type should be used as the default for bias with
            // directions that include a bias.
            const bool is_f64 = (i_dt.size() == 1 && i_dt[0] == dnnl_f64)
                    || (i_dt.size() > 1 && i_dt[1] == dnnl_f64);
            i_bia_dt = is_f64 ? dnnl_f64 : dnnl_f32;
        }
        auto prb = std::make_shared<prb_t>(s.desc, i_dir, i_dt, i_bia_dt,
                i_stag, i_wtag, i_dtag, i_alg, i_mb, i_attr, i_ctx_init,
                i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s) {
    static constexpr int n_inputs = 3;
    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            BENCHDNN_PRINT(0, "%s%d.\n",
                    "ERROR: `dt` option expects either a single input or three "
                    "inputs in SRC, WEI, DST order. Current size is: ",
                    static_cast<int>(i_dt.size()));
            return FAIL;
        }
    }

    return OK;
}

int bench(int argc, char **argv) {
    driver_name = "deconv";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_dt(s.bia_dt, def.bia_dt, argv[0], "bia-dt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_alg(s.alg, def.alg, str2alg, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            SAFE(str2desc(&s.desc, argv[0]), CRIT);

            SAFE(verify_input(s), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace deconv
