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
#include <sstream>

#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "reduction/reduction.hpp"

namespace reduction {

// first: nonneutral elements
// second: maximum range
using problem_bounds = std::pair<const int, const int>;

// acc | acc | elems | value_range | worst case
// s32 | mul |  10   |       3     |     3^10=2^16, out of 2^30 (max integer)
// f16 | mul |  10   |       1     | (2^1)^10=2^10, out of 2^16 (max exponent)
// f32 | mul |  30   |       3     | (2^3)^30=2^90, out of 2^128 (max exponent)
// s32 | sum | 10000 |      50     | 10000*50=2^19, out of 2^30 (max integer)
// f16 | sum | 1000  |       8     | 1000*8=2^13, out of 2^10 (max mantissa/integer)
// f32 | sum | 10000 |      16     | 10000*16=2^18, out of 2^23 (max mantissa/integer)
//  min/max  |  all  |    1000     | no limits on accumulation chain

// In f16 cases, the worst case exceeds the data type bounds, however it's rare
// to reach these extreme cases as long as they're close (can't just use f32 bounds)
const problem_bounds MUL_INT = problem_bounds(10, 3);
const problem_bounds MUL_F16 = problem_bounds(10, 1);
const problem_bounds MUL_F32 = problem_bounds(30, 3);
const problem_bounds SUM_INT = problem_bounds(10000, 50);
const problem_bounds SUM_F16 = problem_bounds(1000, 8);
const problem_bounds SUM_F32 = problem_bounds(10000, 16);
const problem_bounds MINMAX_INT = problem_bounds(-1, 1000);
const problem_bounds MINMAX_FP = problem_bounds(-1, 1000);

problem_bounds get_problem_bounds(alg_t alg, dnnl_data_type_t dt) {
    const bool is_int = is_integral_dt(dt);

    // Integer cases
    if (is_int) {
        switch (alg) {
            case alg_t::max:
            case alg_t::min: return MINMAX_INT;
            case alg_t::mul: return MUL_INT;
            // All remaining cases accumulate via sum
            default: return SUM_INT;
        }
    }

    // Floating-point cases
    const bool is_f16 = (dt == dnnl_f16);
    switch (alg) {
        case alg_t::max:
        case alg_t::min: return MINMAX_FP;
        case alg_t::mul: return is_f16 ? MUL_F16 : MUL_F32;
        // All remaining cases accumulate via sum
        default: return is_f16 ? SUM_F16 : SUM_F32;
    }
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_d, prb->ndims, prb->vdims[0].data(),
            force_f32_dt ? dnnl_f32 : prb->sdt, prb->stag));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_d, prb->ndims, prb->vdims[1].data(),
            force_f32_dt ? dnnl_f32 : prb->ddt, prb->dtag));

    attr_args_t attr_args;
    DNN_SAFE_STATUS(attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->vdims[1].data()));
    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args, prb->ndims));

    TIME_C_PD(DNN_SAFE_STATUS(dnnl_reduction_primitive_desc_create(
            &init_pd_args.pd, init_pd_args.engine, alg2alg_kind(prb->alg),
            init_pd_args.src_md ? init_pd_args.src_md : src_d, dst_d, prb->p,
            prb->eps, dnnl_attr)));

    return dnnl_success;
}

bool is_norm_alg(const alg_t alg) {
    return alg == alg_t::norm_lp_max || alg == alg_t::norm_lp_sum
            || alg == alg_t::norm_lp_power_p_max
            || alg == alg_t::norm_lp_power_p_sum;
}

int fill_mem(const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp,
        float non_neutral_prob, bool expanded_range, bool only_positive_values,
        res_t *res) {
    // Refer to modes documentation for filling principles.
    // Multiply alg overflows extremely fast. Exclude it from validation.
    if (has_bench_mode_bit(mode_bit_t::bitwise) && prb->alg != alg_t::mul) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    const auto sdt = mem_dt.dt();
    const auto ddt = prb->ddt;
    const auto nelems = mem_fp.nelems();
    const float neutral_value = prb->alg == alg_t::mul ? 1.0f : 0.0f;
    // include ddt in is_signed to avoid mistrusted rounding negative -> 0
    const bool is_signed = sdt != dnnl_u8 && ddt != dnnl_u8;
    float shift = 0.0f;
    if (prb->alg == alg_t::mean || (prb->alg == alg_t::min && !is_signed))
        shift = 1.0f;
    const bool is_int = is_integral_dt(sdt);

    // Follow table in comments of fill_src
    int value_range = get_problem_bounds(prb->alg, sdt).second;

    const bool is_mul_fp = prb->alg == alg_t::mul && !is_int;
    const int min_range = is_mul_fp ? -value_range : 1;

    bool fill_with_powers_of_two = is_mul_fp;
    if (expanded_range) {
        // when using the expanded range, never fill with powers of 2
        fill_with_powers_of_two = false;
        value_range = 1000;
    }

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        const int64_t idx_start = idx_chunk * chunk_size;
        const int64_t idx_end = MIN2(idx_start + chunk_size, nelems);

        std::minstd_rand msr(idx_start + 1);
        msr.discard(1);
        std::uniform_int_distribution<> igen(min_range, value_range);
        std::uniform_int_distribution<> fifty_fifty(0, 1);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = neutral_value;
            if (flip_coin(idx, non_neutral_prob)) {
                const int gen = igen(msr);
                value = fill_with_powers_of_two ? std::pow(2, gen) : gen;

                if (!only_positive_values && is_signed && fifty_fifty(msr) == 1)
                    value = -value;
            }
            value += shift;
            mem_fp.set_f32_elem(
                    idx, round_to_nearest_representable(sdt, value));
        }
    });
    SAFE(mem_dt.reorder(mem_fp, res), WARN);
    return OK;
}

int fill_src(int exec_arg, const prb_t *prb, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (!nelems) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    int nelems_to_reduce = 1;
    for (int dim = 0; dim < prb->ndims; dim++) {
        if (prb->vdims[0][dim] != prb->vdims[1][dim]) {
            nelems_to_reduce *= prb->vdims[0][dim];
        }
    }

    // Determine number of non-neutral elements to have in the reduction chain
    int safe_to_reduce_elems = get_problem_bounds(prb->alg, prb->sdt).first;
    if (safe_to_reduce_elems == -1) safe_to_reduce_elems = nelems_to_reduce;

    const float non_neutral_prob
            = 1.f * safe_to_reduce_elems / nelems_to_reduce;

    return fill_mem(prb, mem_dt, mem_fp, non_neutral_prob, false, false, res);
}

int fill_dst(int exec_arg, const prb_t *prb, dnn_mem_t &mem_dt,
        dnn_mem_t &mem_fp, res_t *res) {
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;
    const bool only_positive_values = is_norm_alg(prb->alg);
    return fill_mem(prb, mem_dt, mem_fp, 1.0f, true, only_positive_values, res);
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    skip_unimplemented_data_type({prb->sdt, prb->ddt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_reduction, prb->sdt);
    skip_unimplemented_binary_po(prb->attr, res);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_reduction);
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    // Normalization algorithms don't make sense for integer data type.
    // They also can't have `p` parameter less than one.
    const bool is_invalid = is_norm_alg(prb->alg)
            && (is_integral_dt(prb->sdt) || prb->p < 1.f);

    if (is_invalid) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    // accounts for inaccurate rootn/pow functions in norm algs.
    float scale = is_norm_alg(prb->alg) ? 5.0f : 1.0f;
    cmp.set_threshold(scale * epsilon_dt(prb->ddt));

    if (is_amd_gpu()) {
        // MIOpen implementation is less accurate for f16 data type therefore
        // adjust the threshold.
        if (prb->sdt == dnnl_f16 || prb->ddt == dnnl_f16)
            cmp.set_threshold(1.5e-4f * 4.f);
    }
}

std::vector<int> prb_t::supported_exec_args(bool) const {
    static const std::vector<int> exec_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DST,
    };
    return exec_args;
}

void binary_po_fill_cfg(std::unordered_map<int, fill_cfg_t> &fill_cfg_map,
        int exec_arg, const dnn_mem_t &mem, const attr_t &attr) {
    fill_cfg_t cfg;
    const int post_ops_range = DNNL_ARG_ATTR_MULTIPLE_POST_OP(31)
            - DNNL_ARG_ATTR_MULTIPLE_POST_OP(0);
    const bool is_post_ops_arg = (exec_arg & post_ops_range);
    if (is_post_ops_arg) {
        // Config secures only positive values since reduction output is
        // positive in several scenarios, and using negative values leads to the
        // cancellation effect.
        const int bin_po_idx
                = exec_arg / DNNL_ARG_ATTR_MULTIPLE_POST_OP_BASE - 1;
        assert(bin_po_idx < attr.post_ops.len());
        const auto alg = attr.post_ops.entry[bin_po_idx].kind;
        const bool is_src1_arg = !(exec_arg
                ^ (DNNL_ARG_ATTR_MULTIPLE_POST_OP(bin_po_idx)
                        | DNNL_ARG_SRC_1));

        if (is_src1_arg) {
            cfg = fill_cfg_t(mem.dt(), 0.f, 16.f, /* int = */ true, alg,
                    "reduction_binary_post_op");
            fill_cfg_map.insert({DNNL_ARG_SRC_1, cfg});
        }
    }
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

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_src(exec_arg, prb, mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM)
                        >= 0) {
                    SAFE(fill_dst(exec_arg, prb, mem, ref_mem, res), WARN);
                    // Bitwise mode for sum requires a copy due to data for
                    // post-op will be overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem, res), WARN);
                    }
                }
                break;
            default: {
                std::unordered_map<int, fill_cfg_t> fill_cfg_map;
                binary_po_fill_cfg(fill_cfg_map, exec_arg, mem, prb->attr);
                SAFE(init_ref_memory_args_default_case(exec_arg, mem, ref_mem,
                             prb->attr, res, fill_cfg_map),
                        WARN);
            } break;
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

} // namespace reduction
