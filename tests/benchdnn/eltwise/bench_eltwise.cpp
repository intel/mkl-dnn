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
#include "utils/stringstream.hpp"
#include "utils/task_executor.hpp"

#include "eltwise/eltwise.hpp"

namespace eltwise {

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    for_(const auto &i_dir : s.dir)
    for_(const auto &i_dt : s.dt)
    for_(const auto &i_tag : s.tag)
    for_(const auto &i_alg : s.alg)
    for_(const auto &i_alpha : s.alpha)
    for_(const auto &i_beta : s.beta)
    for_(const auto &i_mb : s.mb)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (auto i_inplace : s.inplace) {
        auto prb = std::make_shared<prb_t>(s.prb_dims, i_dir, i_dt, i_tag,
                i_alg, i_alpha, i_beta, i_mb, i_inplace, i_attr, i_ctx_init,
                i_ctx_exe, s.impl_filter);
        if (s.pattern && !match_regex(prb->str(), s.pattern)) return;

        task_executor.submit(prb, s.perf_template, createit, checkit, doit);
    }
}

int verify_input(const settings_t &s) {
    for (const auto &i_alg : s.alg) {
        bool ok = i_alg > alg_t::ELTWISE_START && i_alg < alg_t::ELTWISE_END;
        if (!ok) {
            stringstream_t ss;
            ss << i_alg;
            BENCHDNN_PRINT(0, "%s%s%s\n", "ERROR: unknown algorithm `",
                    ss.str().c_str(), "`.");
            SAFE_V(FAIL);
        }
    }

    for_(const auto &i_alg : s.alg)
    for_(const auto &i_alpha : s.alpha)
    for (const auto &i_beta : s.beta) {
        // iterator over alpha and beta (alphabetic order!)
        switch (i_alg) {
            case alg_t::ABS:
            case alg_t::EXP:
            case alg_t::EXP_DST:
            case alg_t::GELU_ERF:
            case alg_t::GELU_TANH:
            case alg_t::LOG:
            case alg_t::LOGISTIC:
            case alg_t::LOGISTIC_DST:
            case alg_t::MISH:
            case alg_t::ROUND:
            case alg_t::SQRT:
            case alg_t::SQRT_DST:
            case alg_t::SQUARE:
            case alg_t::TANH:
            case alg_t::TANH_DST:
                if (i_alpha != 0)
                    BENCHDNN_PRINT(1, "%s\n",
                            "WARNING: non-zero alpha is ignored. "
                            "Consider adding --alpha=0 to a command line.");
                if (i_beta != 0)
                    BENCHDNN_PRINT(1, "%s\n",
                            "WARNING: non-zero beta is ignored. "
                            "Consider adding --beta=0 to a command line.");
                break;
            case alg_t::ELU:
            case alg_t::ELU_DST:
            case alg_t::RELU:
            case alg_t::RELU_DST:
            case alg_t::SRELU:
            case alg_t::SWISH:
                if (i_beta != 0)
                    BENCHDNN_PRINT(1, "%s\n",
                            "WARNING: non-zero beta is ignored. "
                            "Consider adding --beta=0 to a command line.");
                break;
            default:;
        }
    }

    return OK;
}

static const std::string help_alpha_beta
        = "FLOAT    (Default: 0.f)\n    Specifies algorithm parameter "
          "extension where applicable.\n";

int bench(int argc, char **argv) {
    driver_name = "eltwise";
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
                || parse_vector_option(s.alpha, def.alpha, atof, argv[0],
                        "alpha", help_alpha_beta)
                || parse_vector_option(s.beta, def.beta, atof, argv[0], "beta",
                        help_alpha_beta)
                || parse_alg(
                        s.alg, def.alg, attr_t::post_ops_t::str2kind, argv[0])
                || parse_inplace(s.inplace, def.inplace, argv[0])
                || parse_mb(s.mb, def.mb, argv[0])
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
} // namespace eltwise
