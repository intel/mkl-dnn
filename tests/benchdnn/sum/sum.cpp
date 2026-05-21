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

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>

#include "oneapi/dnnl/dnnl.h"

#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sum/sum.hpp"

namespace sum {

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    std::vector<benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t>> src_d_wrappers(
            prb->n_inputs());

    for (int i_input = 0; i_input < prb->n_inputs(); ++i_input)
        src_d_wrappers[i_input] = dnn_mem_t::init_md(prb->ndims,
                prb->dims.data(), force_f32_dt ? dnnl_f32 : prb->sdt[i_input],
                prb->stag[i_input]);

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d {};
    if (prb->dtag != tag::undef) {
        dst_d = dnn_mem_t::init_md(prb->ndims, prb->dims.data(),
                force_f32_dt ? dnnl_f32 : prb->ddt, prb->dtag);
    }

    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args_t(), prb->ndims));

    std::vector<dnnl_memory_desc_t> src_d(
            src_d_wrappers.begin(), src_d_wrappers.end());
    init_pd_args.is_iterator_supported = false;
    TIME_C_PD(DNN_SAFE_STATUS(dnnl_sum_primitive_desc_create(&init_pd_args.pd,
            init_pd_args.engine, dst_d, prb->n_inputs(),
            prb->input_scales.data(), src_d.data(), dnnl_attr)));

    return dnnl_success;
}

int fill_src(int exec_arg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_dt.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    const auto dt = mem_dt.dt();
    const int range = 16;
    const int f_min = dt == dnnl_u8 ? 0 : -range / 2;

    benchdnn_parallel_nd(nelems, [&](int64_t i) {
        const float gen = ((97 * i) - 17 * exec_arg + 101) % range;
        const float value = (dt == dnnl_bf16 || dt == dnnl_f16)
                ? (f_min + gen) / range
                : (f_min + gen) * (1.0f + 4.0f / range);
        mem_fp.set_f32_elem(i, round_to_nearest_representable(dt, value));
    });

    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    std::vector<dnnl_data_type_t> dts = prb->sdt;
    dts.push_back(prb->ddt);
    skip_unimplemented_data_type(dts, prb->dir, res);
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(
                res, prb->sdt[0], prb->ddt, prb->stag[0], prb->dtag);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    cmp.set_threshold(epsilon_dt(prb->ddt) * prb->n_inputs());
}

std::vector<int> prb_t::supported_exec_args(bool) const {
    static const std::vector<int> exec_args = {
            DNNL_ARG_MULTIPLE_SRC,
            DNNL_ARG_DST,
    };
    return exec_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    const auto *prb = prb_t::from(base_prb);
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                            /* prefill = */ false));
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        bool is_src_arg = (exec_arg & DNNL_ARG_MULTIPLE_SRC);
        if (is_src_arg) {
            SAFE(fill_src(exec_arg, mem, ref_mem, res), WARN);
            // Need a copy of source data for inplace mode for bitwise testing.
            // For multiple args, only the first one requires a copy.
            if (has_bench_mode_bit(mode_bit_t::bitwise) && prb->inplace
                    && exec_arg == DNNL_ARG_MULTIPLE_SRC) {
                auto &src_copy = mem_map.at(-exec_arg);
                SAFE(bool(src_copy) ? OK : FAIL, WARN);
                SAFE(src_copy.reorder(mem, res), WARN);
            }
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    v_prim.resize(1);
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        SAFE(check_total_size(res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
    }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = v_prim[0];

    dnn_mem_map_t mem_map, ref_mem_map;
    SAFE(init_memory_args(mem_map, prb, prim, res), WARN);
    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, prim, prb, res), WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(prim, args, res), WARN);

    check_correctness(
            prb, {DST}, args, ref_args, compute_ref, setup_cmp, res, prb->dir);
    SAFE(check_bitwise(prim, {DST}, args, prb->attr, prb->inplace, res), WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace sum
