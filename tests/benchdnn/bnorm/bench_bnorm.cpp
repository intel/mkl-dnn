/*******************************************************************************
* Copyright 2017 Intel Corporation
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
#include "utils/stringstream.hpp"
#include "utils/task_executor.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_tag : s.tag)
    for_(const auto &i_strides : s.strides)
    for_(const auto &i_flags : s.flags)
    for_(const auto &i_mb : s.mb)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        auto prb = std::make_shared<prb_t>(s.desc, i_dir, i_dt, i_tag,
                i_strides, i_flags, s.check_alg, s.debug_check_ws, i_mb,
                i_inplace, i_attr, i_ctx_init, i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s, const settings_t &def) {
    static constexpr int n_inputs = 2;

    for (const auto &i_strides : s.strides) {
        if (i_strides.size() != n_inputs) {
            stringstream_t ss;
            ss << vdims2str(i_strides);
            BENCHDNN_PRINT(0,
                    "Error: `strides` option expects two inputs in format "
                    "`[SRC]:[DST]` (colon must present). Current input is: "
                    "\"%s\"\n",
                    ss.str().c_str());
            SAFE_V(FAIL);
        }
        for (int i = 0; i < n_inputs; i++) {
            if (i_strides[i].size() != static_cast<size_t>(s.desc.ndims)
                    && !i_strides[i].empty()) {
                stringstream_t ss;
                ss << vdims2str(i_strides);
                BENCHDNN_PRINT(0,
                        "Error: number of dimensions in the `strides` option "
                        "doesn't match the number of dimensions in the "
                        "original "
                        "problem. Current output is: \"%s\"\n",
                        ss.str().c_str());
                SAFE_V(FAIL);
            }
        }
    }

    for_(const auto &i_strides : s.strides)
    for (const auto &i_tag : s.tag) {
        const bool strided_input
                = !i_strides[0].empty() || !i_strides[1].empty();
        if (strided_input) {
            const bool no_stride_with_tag = IMPLICATION(i_tag != def.tag[0],
                    i_strides[0].empty() && i_strides[1].empty());

            if (!no_stride_with_tag) {
                BENCHDNN_PRINT(0, "%s\n",
                        "Error: both `strides` and `tag` knobs can't be used "
                        "with either of `src`, or `dst` tensors.");
                SAFE_V(FAIL);
            }
        }
    }

    return OK;
}

static const std::string help_flags
        = "FLAGS    (Default: not specified)\n    Specifies normalization "
          "flags. `FLAGS` values are:\n    * `G` for global_stats.\n    * `C` "
          "for scale.\n    * `H` for shift.\n    * `R` for fuse_norm_relu.\n   "
          " * `A` for fuse_norm_add_relu.\n";

static const std::string help_check_alg
        = "CHECK_ALG\n    Dev debug setting to validate output for different "
          "inputs. Overrides driver's automatic choice.\n    `CHECK_ALG` "
          "values are `alg_0` or `alg_1`.\n";

static const std::string help_debug_check_ws
        = "BOOL    (Default: `false`)\n    Instructs the driver to validates "
          "workspace correctness on forward prop kind when set to `true`.\n";

int bench(int argc, char **argv) {
    driver_name = "bnorm";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_dt(s.dt, def.dt, argv[0])
                || parse_tag(s.tag, def.tag, argv[0])
                || parse_strides(s.strides, def.strides, argv[0], "strides")
                || parse_vector_option(s.flags, def.flags, str2flags, argv[0],
                        "flags", help_flags)
                || parse_single_value_option(s.check_alg, def.check_alg,
                        str2check_alg, argv[0], "check-alg", help_check_alg)
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
                || parse_single_value_option(s.debug_check_ws,
                        def.debug_check_ws, parsers::str2bool, argv[0],
                        "debug-check-ws", help_debug_check_ws)
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            SAFE(str2desc(&s.desc, argv[0]), CRIT);

            SAFE(verify_input(s, def), WARN);
            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace bnorm
