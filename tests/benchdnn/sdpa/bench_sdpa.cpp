/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "sdpa/sdpa.hpp"

namespace sdpa {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_qtag : s.qtag)
    for_(const auto &i_ktag : s.ktag)
    for_(const auto &i_vtag : s.vtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_mdt : s.mdt)
    for_(const auto &i_mask_type : s.mask_type)
    for_(const auto &i_scale_type : s.scale_type)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for (const auto &i_ctx_exe : s.ctx_exe) {
        auto prb = std::make_shared<prb_t>(s.prb_vdims, i_dir, i_dt, i_qtag,
                i_ktag, i_vtag, i_dtag, i_mdt, i_mask_type, i_scale_type,
                i_attr, i_ctx_init, i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s) {
    static constexpr int n_inputs = 4; // Q, K, V, DST dt

    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            BENCHDNN_PRINT(0,
                    "ERROR: sdpa driver: `dt` option expects either a single "
                    "input or four inputs in Q, K, V, DST order. Current "
                    "size is: \"%ld\"\n",
                    (long)i_dt.size());
            SAFE_V(FAIL);
        }
    }

    static constexpr int n_vdims_inputs = 3; // Q, K, V dims
    if (s.prb_vdims.n_inputs() != n_vdims_inputs) {
        BENCHDNN_PRINT(0,
                "ERROR: Expected number of dims arguments is `%d`, provided "
                "`%d`.\n",
                n_vdims_inputs, s.prb_vdims.n_inputs());
        SAFE_V(FAIL);
    }
    return OK;
}

static const char *help_mask
        = "MASK    (Default: `none`)\n    Specifies the attention mask "
          "type for SDPA.\n    `none` - no mask, `buffer` - explicit mask "
          "buffer, `buffer_1d` - broadcast [1,1,1,S_kv],\n"
          "    `buffer_2d` - broadcast [1,1,S_q,S_kv],\n"
          "    `causal_top_left` - causal mask from top-left,\n"
          "    `causal_bottom_right` - causal mask from bottom-right.\n";

static const char *help_scale
        = "SCALE    (Default: `library`)\n    Specifies the attention "
          "scale type for SDPA.\n    `library` - library computes "
          "1/sqrt(head_size),\n    `mul` - multiply by scale, `div` - divide "
          "by scale.\n";

int bench(int argc, char **argv) {
    driver_name = "sdpa";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_dir(s.dir, def.dir, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_tag(s.qtag, def.qtag, argv[0], "qtag")
                || parse_tag(s.ktag, def.ktag, argv[0], "ktag")
                || parse_tag(s.vtag, def.vtag, argv[0], "vtag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_dt(s.mdt, def.mdt, argv[0], "mdt")
                || parse_vector_option(s.mask_type, def.mask_type,
                        str2mask_type, argv[0], "mask", help_mask)
                || parse_vector_option(s.scale_type, def.scale_type,
                        str2scale_type, argv[0], "scale", help_scale)
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0], /* min_inputs = */ 3);

            SAFE(verify_input(s), WARN);

            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace sdpa
