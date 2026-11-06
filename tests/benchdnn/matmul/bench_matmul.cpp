/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2025 Arm Ltd. and affiliates
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

#include <algorithm>
#include <limits>

#include "dnnl_common.hpp"
#include "utils/parser.hpp"
#include "utils/stringstream.hpp"
#include "utils/task_executor.hpp"

#include "matmul/matmul.hpp"

namespace matmul {

#ifdef DNNL_DEV_MODE
// Parses `A-B`; false for any other string.
static bool parse_range(const std::string &v, int &beg, int &end) {
    auto is_num = [](const std::string &s) {
        return s.size() <= 9 && parser::utils::has_only_digits(s);
    };
    auto dash = v.find('-');
    if (dash == std::string::npos || !is_num(v.substr(0, dash))
            || !is_num(v.substr(dash + 1)))
        return false;
    beg = std::stoi(v.substr(0, dash));
    end = std::stoi(v.substr(dash + 1));
    return true;
}

// Expands `all`/`A-B` into the problem's valid ranks; no override if there is
// no jit:gemm candidate count; no ranks if the range is empty.
template <typename make_prb_t>
static std::vector<std::string> expand_gpu_kernel(
        const std::string &v, const make_prb_t &make_prb) {
    if (!is_gpu()) return {""};
    int beg = 0, end = std::numeric_limits<int>::max();
    if (v != "all" && !parse_range(v, beg, end)) return {v};
    auto prb = make_prb("");
    int count = query_gpu_kernel_count(prb.get());
    // Systolic takes no overrides, so a pinned rank reaches jit:gemm.
    if (count < 0) count = query_gpu_kernel_count(make_prb("0").get());
    if (count < 0) {
        BENCHDNN_PRINT(1,
                "Warning: matmul driver: --gpu-kernel=%s: no jit:gemm kernel "
                "count, running without override: %s\n",
                v.c_str(), prb->str());
        return {""};
    }
    std::vector<std::string> ranks;
    for (int r = beg; r <= std::min(end, count - 1); r++)
        ranks.push_back(std::to_string(r));
    if (ranks.empty()) {
        BENCHDNN_PRINT(0,
                "Warning: matmul driver: --gpu-kernel=%s: no jit:gemm "
                "candidates in range (count: %d), skipping: %s\n",
                v.c_str(), count, prb->str());
    }
    return ranks;
}
#endif // DNNL_DEV_MODE

void check_correctness(const settings_t &s, task_executor_t &task_executor) {
    std::vector<std::pair<dnnl_data_type_t, int>> bia_cfg;
    for (const auto &i_bia_dt : s.bia_dt) {
        if (i_bia_dt == dnnl_data_type_undef) {
            bia_cfg.emplace_back(i_bia_dt, 0);
            continue;
        }
        for (const auto &i_bia_mask : s.bia_mask)
            bia_cfg.emplace_back(i_bia_dt, i_bia_mask);
    }

    for_(const auto &i_dt : s.dt)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for_(const auto &i_dtag : s.dtag)
    for_(const auto &i_sparse_options : s.sparse_options)
    for_(const auto &i_strides : s.strides)
    for_(const auto &i_rt_dims_masks : s.rt_dims_masks)
    for_(const auto &i_gpu_kernel : s.gpu_kernel)
    for_(const auto &i_attr : s.attributes)
    for_(const auto &i_ctx_init : s.ctx_init)
    for_(const auto &i_ctx_exe : s.ctx_exe)
    for (const auto &i_bia_cfg : bia_cfg) {
        auto make_prb = [&](const std::string &gpu_kernel) {
            return std::make_shared<prb_t>(s.prb_vdims, i_dt, i_stag, i_wtag,
                    i_dtag, i_strides, i_bia_cfg.first, i_bia_cfg.second,
                    i_rt_dims_masks, i_sparse_options, i_attr, i_ctx_init,
                    i_ctx_exe, s.impl_filter, gpu_kernel);
        };
        std::vector<std::string> gpu_kernels {i_gpu_kernel};
#ifdef DNNL_DEV_MODE
        // `all` and `A-B` depend on the problem's candidate count, so expand
        // them per problem; make_prb builds the problem to query that count.
        gpu_kernels = expand_gpu_kernel(i_gpu_kernel, make_prb);
#endif
        for (const auto &gpu_kernel : gpu_kernels) {
            auto prb = make_prb(gpu_kernel);
            if (s.pattern && !match_regex(prb->str(), s.pattern)) continue;

            task_executor.submit(prb, s.perf_template, createit, checkit, doit);
        }
    }
}

int verify_input(const settings_t &s, const settings_t &def) {
    static constexpr int n_inputs = 3;

    for (const auto &i_dt : s.dt) {
        if (i_dt.size() != 1 && i_dt.size() != n_inputs) {
            fprintf(stderr,
                    "ERROR: matmul driver: `dt` option expects either a single "
                    "input or three inputs in SRC, WEI, DST order. Current "
                    "size is: \"%ld\"\n",
                    (long)i_dt.size()),
                    fflush(stderr);
            SAFE_V(FAIL);
        }
    }

    for (const auto &i_strides : s.strides) {
        if (i_strides.size() != n_inputs) {
            stringstream_t ss;
            ss << i_strides;
            fprintf(stderr,
                    "ERROR: matmul driver: `strides` option expects three "
                    "inputs in format `[SRC]:[WEI]:[DST]` (two colons must "
                    "present). Current input is: \"%s\"\n",
                    ss.str().c_str()),
                    fflush(stderr);
            SAFE_V(FAIL);
        }
    }

    for_(const auto &i_strides : s.strides)
    for_(const auto &i_stag : s.stag)
    for_(const auto &i_wtag : s.wtag)
    for (const auto &i_dtag : s.dtag) {
        const bool strided_input = !i_strides[STRIDES_SRC].empty()
                || !i_strides[STRIDES_WEI].empty()
                || !i_strides[STRIDES_DST].empty();
        if (strided_input) {
            const bool no_stride_with_tag
                    = IMPLICATION(i_stag != def.stag[0],
                              i_strides[STRIDES_SRC].empty())
                    && IMPLICATION(i_wtag != def.wtag[0],
                            i_strides[STRIDES_WEI].empty())
                    && IMPLICATION(i_dtag != def.dtag[0],
                            i_strides[STRIDES_DST].empty());

            if (!no_stride_with_tag) {
                fprintf(stderr,
                        "ERROR: matmul driver: both `strides` and `tag` knobs "
                        "can not be used with either of `src`, `wei`, and `dst`"
                        " tensors.\n"),
                        fflush(stderr);
                SAFE_V(FAIL);
            }
        }
    }

    static constexpr int n_vdims_inputs = 2;
    if (s.prb_vdims.n_inputs() != n_vdims_inputs) {
        BENCHDNN_PRINT(0,
                "ERROR: Expected number of dims arguments is `%d`, provided "
                "`%d`.\n",
                n_vdims_inputs, s.prb_vdims.n_inputs());
        SAFE_V(FAIL);
    }
    return OK;
}

// Validate input consistency for grouped encoding
//
// Two configurations are supported:
//  - 2Dx3D with variable_dim_idx=0: src is [total_M, K] with dim 0
//    variable (row-major), wei is dense 3D [G, K, N],
//    dst is [total_M, N] with dim 0 variable,
//  - 2Dx2D with variable_dim_idx=1: src is [M, total_K] with dim 1
//    variable (col-major), wei is [total_K, N] with dim 0
//    variable (row-major), dst is dense 3D [G, M, N]
int verify_grouped_input(const settings_t &s) {
    if (!s.sparse_options[0].is_grouped(DNNL_ARG_SRC)) return OK;

    const int variable_dim_idx
            = s.sparse_options[0].get_variable_dim_idx(DNNL_ARG_SRC);

    if (variable_dim_idx != 0 && variable_dim_idx != 1) {
        BENCHDNN_PRINT(0,
                "ERROR: grouped encoding supports dim 0 (2Dx3D) or "
                "dim 1 (2Dx2D , got dim %d\n",
                variable_dim_idx);
        SAFE_V(FAIL);
    }

    // The variable dim is encoded in the src tensor at variable_dim_idx
    const auto &src_dims = s.prb_vdims.vdims[0];

    // The 2Dx2D variant additionally requires wei dim 0 == src dim 1
    // (i.e., shared contraction dim)
    if (variable_dim_idx == 1) {
        const auto &wei_dims = s.prb_vdims.vdims[1];
        if (wei_dims.size() != 2) {
            BENCHDNN_PRINT(0, "%s\n",
                    "ERROR: 2Dx2D variant (variable_dim_idx=1) requires 2D "
                    "weights with shape [total_K, N]");
            SAFE_V(FAIL);
        }
        if (wei_dims[0] != src_dims[1]) {
            BENCHDNN_PRINT(0,
                    "ERROR: 2Dx2D variant wei dim 0 (%lld) must match src "
                    "dim 1 (%lld)\n",
                    (long long)wei_dims[0], (long long)src_dims[1]);
            SAFE_V(FAIL);
        }
    }

    // Deferred profiles build their sizes later in the prb_t ctor, so explicit
    // checks below don't apply
    if (s.sparse_options[0].has_deferred_profile()) return OK;

    // The library cannot validate group sizes at primitive creation time,
    // so benchdnn checks them here
    const int64_t expected = src_dims[variable_dim_idx];
    int64_t total_var = 0;
    const auto &group_sizes = s.sparse_options[0].get_group_sizes(DNNL_ARG_SRC);
    for (size_t i = 0; i < group_sizes.size(); i++) {
        if (group_sizes[i] < 0) {
            BENCHDNN_PRINT(0,
                    "ERROR: group_sizes[%zu] must be non-negative, got %lld\n",
                    i, (long long)group_sizes[i]);
            SAFE_V(FAIL);
        }
        total_var += group_sizes[i];
    }

    if (total_var != expected) {
        BENCHDNN_PRINT(0,
                "ERROR: sum of group sizes (%lld) doesn't match src "
                "variable dim (%lld)\n",
                (long long)total_var, (long long)expected);
        SAFE_V(FAIL);
    }
    return OK;
}

static const std::string help_bia_mask
        = "UINT    (Default: `2`)\n    Specifies a bit-mask that indicates "
          "which bias dimensions coincide with C matrix dimensions, when `1` "
          "is on a correspondent dimension.\n    Note: for grouped matmul "
          "(--grouped) only `bia_mask=2` (N-only) is supported; the bias "
          "shape becomes `[G, N]` (one value per column per expert).\n";

static const std::string help_runtime_dims_masks
        = "UINT:UINT    (Default: `0:0`)\n    Specifies a bit-mask for "
          "matrices A and B that indicates whether a dimension is "
          "`DNNL_RUNTIME_DIM_VAL` if `1` on a correspondent dimension.\n    "
          "For tensors with runtime dimensions specified a correspondent "
          "memory format must be specified, too.\n";

#ifdef DNNL_DEV_MODE
static const std::string help_gpu_kernel
        = "STR    (Default: empty)\n    Specifies a GPU kernel override "
          "passed via a primitive attribute. `N` runs the Nth candidate "
          "after sort (rank); `A-B` and `all` run ranks `A` through `B` or "
          "every rank, clamped to the problem's candidate count (a problem "
          "with an empty range is skipped); problems with no jit:gemm "
          "candidate count run without an override. Any other string is a "
          "strategy override. An empty string means no override. Ignored on "
          "non-GPU engines. Requires a dev-mode library "
          "(ONEDNN_DEV_MODE=ON).\n";

// Validates `A-B`; ranks are expanded per problem in check_correctness.
static std::vector<std::string> str2gpu_kernel(const char *str) {
    int beg = 0, end = 0;
    if (parse_range(str, beg, end) && beg > end) {
        BENCHDNN_PRINT(0, "Error: --gpu-kernel=%s: empty range.\n", str);
        SAFE_V(FAIL);
    }
    std::string s(str);
    if (s.find(',') != std::string::npos
            && s.find_first_not_of("0123456789,") == std::string::npos) {
        BENCHDNN_PRINT(0,
                "Error: --gpu-kernel=%s: one value only; use A-B for several "
                "ranks.\n",
                str);
        SAFE_V(FAIL);
    }
    return {str};
}
#endif // DNNL_DEV_MODE

bool parse_legacy_dt(std::vector<dnnl_data_type_t> &dt,
        const std::vector<dnnl_data_type_t> &def_dt, const char *str,
        const std::string &option_name /* = "dt"*/) {
    if (parser::parse_dt(dt, def_dt, str, option_name)) {
        BENCHDNN_PRINT(0, "%s\n",
                "Warning: \'--bia_dt\' option is deprecated. Please use the "
                "\'--bia-dt\' one.");
        return true;
    }
    return false;
}

int bench(int argc, char **argv) {
    driver_name = "matmul";
    using namespace parser;
    static settings_t s;
    static const settings_t def {};
    static task_executor_t task_executor;
    for (; argc > 0; --argc, ++argv) {
        const bool parsed_options = parse_bench_settings(argv[0])
                || parse_batch(bench, argv[0])
                || parse_multi_dt(s.dt, def.dt, argv[0], "dt")
                || parse_tag(s.stag, def.stag, argv[0], "stag")
                || parse_tag(s.wtag, def.wtag, argv[0], "wtag")
                || parse_tag(s.dtag, def.dtag, argv[0], "dtag")
                || parse_encoding(s.sparse_options, argv[0], "encoding")
                || parse_grouped(s.sparse_options, argv[0], "grouped")
                || parse_strides(s.strides, def.strides, argv[0], "strides")
                || parse_dt(s.bia_dt, def.bia_dt, argv[0], "bia-dt")
                // TODO: remove this later
                || parse_legacy_dt(s.bia_dt, def.bia_dt, argv[0], "bia_dt")
                || parse_vector_option(s.bia_mask, def.bia_mask, atoi, argv[0],
                        "bia_mask", help_bia_mask)
                || parse_multivector_option(s.rt_dims_masks, def.rt_dims_masks,
                        atoi, argv[0], "runtime_dims_masks",
                        help_runtime_dims_masks)
#ifdef DNNL_DEV_MODE
                // Single-value parse: kernel strings may contain commas.
                || parse_single_value_option(s.gpu_kernel, def.gpu_kernel,
                        str2gpu_kernel, argv[0], "gpu-kernel", help_gpu_kernel)
#endif
                || parse_driver_shared_settings(s, def, argv[0]);
        if (!parsed_options) {
            catch_unknown_options(argv[0]);

            parse_prb_vdims(s.prb_vdims, argv[0]);

            SAFE(verify_input(s, def), WARN);
            SAFE(verify_grouped_input(s), WARN);

            s.finalize();
            check_correctness(s, task_executor);
        }
    }

    task_executor.flush();

    return parse_last_argument();
}

} // namespace matmul
