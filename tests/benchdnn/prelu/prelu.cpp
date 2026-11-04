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

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "prelu/prelu.hpp"

namespace prelu {

int fill_data(data_kind_t kind, int exec_arg, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
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

    // Do fixed partitioning to have same filling for any number of threads.
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note 1: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // we avoid it for two reasons:
        //   a. it has a complexity in O(idx_start).
        //   b. igen below might require more than 1 sample
        //   per idx, so the we cannot deterministically compute the
        //   number of states we need to discard
        // Note 2: We also advance the state to avoid having only
        // small values as first chunk input.  The +1 is necessary to
        // avoid generating zeros in first chunk.
        // Note 3: we multiply by kind + 1 to have different values in
        // src/dst and diff_dst. The +1 is to avoid 0 again.
        std::minstd_rand msr((idx_start + 1) * (kind + 1));
        msr.discard(1);
        std::uniform_int_distribution<> igen_02(0, 2), igen_05(0, 5),
                igen_06(0, 6);
        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = 0;
            if (is_integral_dt(mem_dt.dt())) {
                value = igen_05(msr);
            } else {
                // TODO: amount of negative values should depend on number of points
                // to reduce as summation becomes inaccurate.
                switch (kind) {
                    case SRC: value = igen_02(msr); break;
                    case WEI:
                        value = (64 >> igen_06(msr)) / 8.f; // pow2 [0.125f, 8f]
                        break;
                    case DST: value = igen_02(msr) / 16.f; break;
                    default: assert(!"unexpected"); break;
                }
            }
            float sign = mem_dt.dt() == dnnl_u8 ? 1.f
                    : flip_coin(idx, 0.1f)      ? -1.f
                                                : 1.f;
            value = round_to_nearest_representable(mem_dt.dt(), sign * value);
            mem_fp.set_f32_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    const auto &src_dims = prb->vdims[0];
    const auto &weight_dims = prb->vdims[1];

    auto src_d = dnn_mem_t::init_md(prb->ndims, src_dims.data(),
            force_f32_dt ? dnnl_f32 : prb->sdt[0], prb->stag[0]);
    auto weights_d = dnn_mem_t::init_md(prb->ndims, weight_dims.data(),
            force_f32_dt ? dnnl_f32 : prb->sdt[1], prb->stag[1]);

    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args_t(), prb->ndims));

    if (prb->dir & FLAG_FWD) {
        auto dst_d = dnn_mem_t::init_md(prb->ndims, src_dims.data(),
                force_f32_dt ? dnnl_f32 : prb->sdt[0], tag::any);

        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        TIME_C_PD(DNN_SAFE_STATUS(dnnl_prelu_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop,
                init_pd_args.src_md ? init_pd_args.src_md : src_d, weights_d,
                dst_d, dnnl_attr)));
    } else {
        auto diff_src_d = dnn_mem_t::init_md(prb->ndims, src_dims.data(),
                force_f32_dt ? dnnl_f32 : prb->sdt[0], tag::any);
        auto diff_weights_d = dnn_mem_t::init_md(prb->ndims, weight_dims.data(),
                force_f32_dt ? dnnl_f32 : prb->sdt[1], tag::any);
        auto diff_dst_d = dnn_mem_t::init_md(prb->ndims, src_dims.data(),
                force_f32_dt ? dnnl_f32 : prb->sdt[0], tag::any);

        TIME_C_PD(DNN_SAFE_STATUS(dnnl_prelu_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, src_d, weights_d,
                diff_src_d, diff_weights_d, diff_dst_d, init_pd_args.hint,
                dnnl_attr)));
    }

    return dnnl_success;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    skip_unimplemented_data_type(prb->sdt, FWD_D, res);
}

void prb_t::skip_invalid(res_t *res) const {}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    const auto trh_dt = kind == WEI ? prb->sdt[1] : prb->sdt[0];
    cmp.set_threshold(2 * epsilon_dt(trh_dt));

    // Weights are very sparse, no sense to test for trust, otherwise filling
    // is specific to cover half non-zeros only.
    const float zero_trust_percent = kind == WEI ? 99.f : 50.f;
    cmp.set_zero_trust_percent(zero_trust_percent);
}

std::vector<int> prb_t::supported_exec_args(bool override_dir_with_fwd) const {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_DIFF_WEIGHTS,
            DNNL_ARG_DIFF_DST,
    };
    return (override_dir_with_fwd || (dir & FLAG_FWD)) ? exec_fwd_args
                                                       : exec_bwd_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
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

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(SRC, exec_arg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_data(WEI, exec_arg, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_data(DST, exec_arg, mem, ref_mem, res), WARN);
                break;
            default: break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        check_kinds = {DST};
    } else if (prb->dir & FLAG_BWD) {
        check_kinds = {SRC, WEI};
    } else {
        assert(!"unexpected!");
        SAFE_V(FAIL);
    }
    assert(!check_kinds.empty());
    return check_kinds;
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

    check_correctness(prb, get_kinds_to_check(prb), args, ref_args, compute_ref,
            setup_cmp, res, prb->dir);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->attr,
                 prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace prelu
