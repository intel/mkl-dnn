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

#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

#include "utils/dnnl_query.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "eltwise/eltwise.hpp"

namespace eltwise {

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    const dir_t dir = init_pd_args.dir;
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_d, prb->ndims, prb->dims.data(),
            force_f32_dt ? dnnl_f32 : prb->dt, prb->tag));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_d, prb->ndims, prb->dims.data(),
            force_f32_dt ? dnnl_f32 : prb->dt, tag::any));

    dnnl_alg_kind_t alg = attr_t::post_ops_t::kind2dnnl_kind(prb->alg);

    attr_args_t attr_args;
    DNN_SAFE_STATUS(attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dims.data()));
    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args, prb->ndims));

    if (dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;

        TIME_C_PD(DNN_SAFE_STATUS(dnnl_eltwise_forward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, prop, alg,
                init_pd_args.src_md ? init_pd_args.src_md : src_d, dst_d,
                prb->alpha, prb->beta, dnnl_attr)));
    } else {
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_src_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_src_d, prb->ndims,
                prb->dims.data(), force_f32_dt ? dnnl_f32 : prb->dt, tag::any));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_dst_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_dst_d, prb->ndims,
                prb->dims.data(), force_f32_dt ? dnnl_f32 : prb->dt, tag::any));
        if (prb->use_dst()) // Need to create with proper tag
            DNN_SAFE_STATUS(
                    dnn_mem_t::init_md(dst_d, prb->ndims, prb->dims.data(),
                            force_f32_dt ? dnnl_f32 : prb->dt, prb->tag));
        auto &data_d = prb->use_dst() ? dst_d : src_d;

        TIME_C_PD(DNN_SAFE_STATUS(dnnl_eltwise_backward_primitive_desc_create(
                &init_pd_args.pd, init_pd_args.engine, alg, diff_src_d,
                diff_dst_d, data_d, prb->alpha, prb->beta, init_pd_args.hint,
                dnnl_attr)));
    }

    return dnnl_success;
}

static bool check_abs_err(const prb_t *prb, const float &s, const float &trh) {
    const float approx_machine_eps = 2 * epsilon_dt(dnnl_f32);
    const float comp_err = approx_machine_eps / trh;

    switch (prb->alg) {
        case alg_t::ELU:
        case alg_t::ELU_DST:
            // catch catastrophic cancellation when (exp(s) - 1), s < 0 and
            // s is close to zero.
            return (prb->dir & FLAG_FWD) && std::signbit(s)
                    && (fabsf(expf(s) - 1.f) <= comp_err);
        case alg_t::GELU_TANH: {
            // catch catastrophic cancellation
            // (4.f is magic scale for f32)
            const float sqrt_2_over_pi = 0.797884f;
            const float fitting_const = 0.044715f;
            float v = tanhf(sqrt_2_over_pi * s * (1.f + fitting_const * s * s));
            float dg = sqrt_2_over_pi * (1.f + 3.f * fitting_const * s * s);
            if (fabsf(1.f + v) <= comp_err) return true;
            return (prb->dir & FLAG_BWD) && std::signbit(s)
                    && fabsf(1.f + s * (1.f - v) * dg) <= 4.f * comp_err;
        }
        case alg_t::GELU_ERF: {
            // Catch catastrophic cancellation
            // which occurs at large negative s.
            // Factor 2 (in bwd) is to account for the fact that error is
            // accumulated for each summand (except the 1) when they
            // are of the same order of magnitude.
            const float sqrt_2_over_2 = 0.707106769084930419921875f;
            const float two_over_sqrt_pi = 1.12837922573089599609375f;
            float v = s * sqrt_2_over_2;
            if (prb->dir & FLAG_FWD)
                return fabsf(1.f + erff(v)) <= comp_err;
            else
                return fabsf(1.f + erff(v)
                               + v * two_over_sqrt_pi * expf(-v * v))
                        <= comp_err * 2.f;
        }
        case alg_t::TANH:
            // catch catastrophic cancellation, which occurs when err in tanh(s)
            // is high and tanh(s) is close to 1.
            return (prb->dir & FLAG_BWD) && (1.f - tanhf(fabsf(s))) <= comp_err;
        case alg_t::TANH_DST: // sse41 can't do fma
            // catch catastrophic cancellation, which occurs when err in tanh(s)
            // is high and tanh(s) is close to 1.
            return (prb->dir & FLAG_BWD) && (1.f - s * s) <= comp_err;
        case alg_t::SRELU:
            // when `alpha * s` is negative, expf(alpha * s) -> 0 rapidly
            // which leads to log1pf(expf(alpha * s)) -> 0
            // which leads to high relative error,
            // while abs error is still low.
            // (10.f is magic scale for bf16)
            return (prb->dir & FLAG_FWD) && std::signbit(prb->alpha * s)
                    && log1pf(expf(prb->alpha * s)) <= 10.f * comp_err;
        case alg_t::MISH:
            // same situation like in SRELU
            return (prb->dir & FLAG_FWD) && std::signbit(s)
                    && s * tanh(log1pf(expf(s))) <= 10.f * comp_err;
        case alg_t::LOGISTIC:
            // when s >= 4, logistic(s) -> 0 rapidly, which leads to high
            // relative error of logistic(s) * (1 - logistic(s)) due to
            // catastrohic cancellation.
            return (prb->dir & FLAG_BWD) && !std::signbit(s)
                    && (1.f / (1.f + expf(s))) <= comp_err;
        case alg_t::LOGISTIC_DST:
            // when s = logistic(x) ~~ 1, it leads to high relative error of
            // s * (1 - s) due to catastrohic cancellation.
            return (prb->dir & FLAG_BWD)
                    && ((1.f - s) <= comp_err || s <= comp_err);
        case alg_t::SWISH: {
            // catch cancellation happening when W(s) ~~ -1 in (1 + W(s))
            // formula part on backward.
            const float alpha_s = prb->alpha * s;
            return (prb->dir & FLAG_BWD)
                    && (alpha_s * (1.f - 1.f / (1.f + expf(-alpha_s)))
                            <= comp_err);
        }
        default: return false;
    }
}

float get_eltwise_threshold(dnnl_data_type_t dt, alg_t alg, bool is_fwd) {
    // Tolerate only rounding error (1 ulp) for other than fp32 precisions.
    float trh = (dt == dnnl_f32 || dt == dnnl_f64) ? 4e-6f : epsilon_dt(dt);
    // Tolerate bigger compute errors for complex algorithms.
    const bool alg_has_higher_tolerance = alg == alg_t::GELU_TANH
            || alg == alg_t::ELU || alg == alg_t::SWISH || alg == alg_t::TANH
            || alg == alg_t::SRELU || alg == alg_t::MISH || alg == alg_t::LOG
            || (is_nvidia_gpu() && alg == alg_t::POW)
            || ((alg == alg_t::ELU_DST || alg == alg_t::TANH_DST) && is_fwd);
    if ((dt == dnnl_f32 || dt == dnnl_f64) && alg_has_higher_tolerance)
        trh = 4e-5f;
    return trh;
}

static float get_eltwise_zero_trust_percent(const prb_t *prb) {
    float ztp = 65.f; // default for eltwise due to filling.
    switch (prb->alg) {
        case alg_t::LINEAR:
            if (prb->alpha == 0) ztp = 100.f;
            break;
        case alg_t::CLIP:
        case alg_t::CLIP_V2:
        case alg_t::CLIP_V2_DST:
            if ((prb->alpha == 0 && prb->beta == 0) || (prb->dir & FLAG_BWD))
                ztp = 100.f;
            break;
        case alg_t::POW:
            if (prb->alpha == 0 || ((prb->dir & FLAG_BWD) && prb->beta == 0))
                ztp = 100.f;
            break;
        default: break;
    }
    // Integral data types with small float values will produce most zeros.
    // u8 with negative alpha will produce only zeros.
    if (is_integral_dt(prb->dt)) ztp = 100.f;
    return ztp;
}

int fill_data(int exec_arg, const prb_t *prb, data_kind_t kind,
        dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        // Some algorithms mandate positive filling.
        const std::vector<alg_t> alg_list {
                alg_t::LOG, alg_t::POW, alg_t::SQRT, alg_t::SQRT_DST};
        const bool use_zero_min_val = std::any_of(alg_list.begin(),
                alg_list.end(), [&](alg_t alg) { return alg == prb->alg; });
        const float range_min_val = use_zero_min_val ? 0.f : -16.f;
        fill_cfg_t fill_cfg(mem_dt.dt(), range_min_val, 16.f, /* int = */ false,
                prb->alg, "eltwise");
        return fill_random_real(mem_dt, mem_fp, nullptr, fill_cfg);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);
    const bool is_log = prb->alg == alg_t::LOG;

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note 1: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // we avoid it for two reasons:
        //   a. it has a complexity in O(idx_start).
        //   b. igen and fgen below might require more than 1 sample
        //   per idx, so the we cannot deterministically compute the
        //   number of states we need to discard
        // Note 2: We also advance the state to avoid having only
        // small values as first chunk input.  The +1 is necessary to
        // avoid generating zeros in first chunk.
        // Note 3: we multiply by kind + 1 to have different values in
        // src/dst and diff_dst. The +1 is to avoid 0 again.
        std::minstd_rand msr((idx_start + 1) * (kind + 1));
        msr.discard(1);
        std::uniform_int_distribution<> igen(0, 10);
        // TODO: 0.09 due to log impl doesn't give good accuracy in 0.99 points
        std::uniform_real_distribution<> fgen(0.f, 0.09f);

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            const int64_t num_of_generation_variants
                    = 13 + (2 * static_cast<int64_t>(is_log));
            float value = FLT_MAX;
            switch (idx % num_of_generation_variants) {
                case 0: value = (float)igen(msr); break; // [0-10] pos
                case 1: value = -(float)igen(msr); break; // [0-10] neg
                case 2: value = fgen(msr); break; // [0.-0.1) pos
                case 3: value = -fgen(msr); break; // [0.-0.1) neg
                case 4: value = 10.f * igen(msr); break; // [0-100] pos
                case 5: value = -10.f * igen(msr); break; // [0-100] neg
                case 6: value = 10.f * fgen(msr); break; // [0.-1.) pos
                case 7: value = -10.f * fgen(msr); break; // [0.-1.) neg
                case 8:
                    value = 88.f + 10.f * fgen(msr);
                    break; // values close to logf(FLT_MAX) for exp alg testing
                case 9:
                    value = 22.f + 10.f * fgen(msr);
                    break; // values close to logf(FLT_MAX)/4.0 for bwd mish alg testing
                case 10:
                    value = 44.f + 10.f * fgen(msr);
                    break; // values close to logf(FLT_MAX)/2.0 for fwd mish alg testing
                case 11: value = prb->alpha; break; // `x = alpha` corner cases
                case 12: value = prb->beta; break; // `x = beta` corner cases
                case 13: value = INFINITY; break; // used in LOG alg only
                case 14: value = -INFINITY; break; // used in LOG alg only
            }
            value = round_to_nearest_representable(prb->dt, value);

            // Hack: -0 may lead to different sign in the answer since input
            // passes through simple reorder which converts -0 into +0.
            if (value == -0.f) value = 0.f;

            mem_fp.set_f32_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    skip_unimplemented_data_type({prb->dt}, prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_eltwise, prb->dt);
    skip_unimplemented_binary_po(prb->attr, res);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_eltwise);

    if (is_gpu() && (prb->dt == dnnl_f8_e5m2 || prb->dt == dnnl_f8_e4m3)
            && prb->dir == BWD_D) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_data_type;
        return;
    }
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    bool is_invalid = false;
    switch (prb->alg) {
        case alg_t::CLIP:
        case alg_t::CLIP_V2:
        case alg_t::CLIP_V2_DST: is_invalid = prb->beta < prb->alpha; break;
        case alg_t::ELU_DST:
        case alg_t::RELU_DST: is_invalid = prb->alpha < 0; break;
        case alg_t::ROUND:
            is_invalid = prb->dt != dnnl_f32 || prb->dir & FLAG_BWD;
            break;
        default: break;
    };
    if (is_invalid) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // Since source is needed for non-use-dst algorithms, it is incorrect to
    // let forward path overwrite it.
    is_invalid = (prb->dir & FLAG_BWD) && !prb->use_dst() && prb->inplace;
    if (is_invalid) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(res, prb->dt, prb->dt, prb->tag, prb->tag);
        if (res->state == SKIPPED) return;
    }
}

bool eltwise_alg_returns_nan_or_inf(alg_t alg) {
    static const std::vector<alg_t> nan_inf_alg
            = {alg_t::EXP, alg_t::EXP_DST, alg_t::LOG, alg_t::POW, alg_t::SQRT,
                    alg_t::SQRT_DST, alg_t::SQUARE};
    return std::any_of(nan_inf_alg.cbegin(), nan_inf_alg.cend(),
            [alg](const alg_t _alg) { return (_alg == alg); });
}

bool eltwise_alg_returns_nan_or_inf(const attr_t &attr) {
    const auto &po = attr.post_ops;
    for (int i = 0; i < po.len(); i++) {
        if (eltwise_alg_returns_nan_or_inf(po.entry[i].kind)) return true;
    }
    return false;
}

bool miopen_check_correctness(const prb_t *prb,
        const compare::compare_t::driver_check_func_args_t &args) {
    if (!is_amd_gpu()) return false;

    // MIOpen generates outputs that are 1ulp in some cases
    // this extra case needs to be addressed
    if ((prb->alg == alg_t::ELU || prb->alg == alg_t::LOGISTIC
                || prb->alg == alg_t::SRELU)
            && ((prb->dir & FLAG_FWD) && (prb->dt == dnnl_f16))) {
        return args.diff <= epsilon_dt(args.dt);
    }
    return false;
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    const float trh
            = get_eltwise_threshold(prb->dt, prb->alg, prb->dir & FLAG_FWD);
    cmp.set_threshold(trh);

    cmp.set_zero_trust_percent(get_eltwise_zero_trust_percent(prb));
    cmp.set_op_output_has_nans(eltwise_alg_returns_nan_or_inf(prb->alg));

    // Since lambda is called when stack is unavailable, need to capture `prb`
    // by value to avoid using dangling references.
    const auto eltwise_add_check =
            [&, prb](const compare::compare_t::driver_check_func_args_t &args) {
        // Some algorithms require absolute value comparison for inputs
        // where catastrophic cancellation may happen.
        const auto &src = ref_args.find(DNNL_ARG_SRC);
        const auto &dst = ref_args.find(DNNL_ARG_DST);
        const auto &source
                = ((prb->dir & FLAG_BWD) && prb->use_dst()) ? dst : src;
        const float s = source.get_f32_elem(args.idx);
        if (check_abs_err(prb, s, args.trh)) return args.diff <= args.trh;
        if (prb->attr.post_ops.binary_index() != -1)
            return args.diff <= args.trh;
        return miopen_check_correctness(prb, args);
    };
    cmp.set_driver_check_function(eltwise_add_check);
}

std::vector<int> prb_t::supported_exec_args(bool override_dir_with_fwd) const {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DST,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SRC,
    };
    // Use these args for the `use_dst` case on backward.
    static const std::vector<int> exec_bwd_use_dst_args = {
            DNNL_ARG_DST,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SRC,
    };
    return (override_dir_with_fwd || (dir & FLAG_FWD)) ? exec_fwd_args
            : use_dst()                                ? exec_bwd_use_dst_args
                                                       : exec_bwd_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    const auto *prb = prb_t::from(base_prb);
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    if (!ref_mem_map.empty()) { erase_unused_args(ref_mem_map, mem_map); }

    const auto &ref_engine = get_cpu_engine();

    const bool is_fwd_prim = is_fwd_prop_kind(query_prop_kind(query_pd(prim)));

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
                if (is_fwd_prim)
                    SAFE(fill_data(exec_arg, prb, SRC, mem, ref_mem, res),
                            WARN);
                // Need a copy of source data for inplace mode for bitwise
                // testing.
                if (has_bench_mode_bit(mode_bit_t::bitwise) && prb->inplace) {
                    auto &src_copy = mem_map.at(-exec_arg);
                    SAFE(bool(src_copy) ? OK : FAIL, WARN);
                    SAFE(src_copy.reorder(mem, res), WARN);
                }
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_data(exec_arg, prb, DST, mem, ref_mem, res), WARN);
                // Need a copy of source data for inplace mode for bitwise
                // testing.
                if (has_bench_mode_bit(mode_bit_t::bitwise) && prb->inplace) {
                    auto &diff_dst_copy = mem_map.at(-exec_arg);
                    SAFE(bool(diff_dst_copy) ? OK : FAIL, WARN);
                    SAFE(diff_dst_copy.reorder(mem, res), WARN);
                }
                break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
                break;
        }
        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    // Unique scenario when driver can utilize inplace but original source is
    // used for cancellation validation, thus, can't be overwritten. Need to
    // create proper DST memory if it was dropped from original map.
    const bool inplace_fwd = prb->inplace && (prb->dir & FLAG_FWD);
    if (inplace_fwd) {
        const auto &dst_md = mem_map.at(DNNL_ARG_SRC).md_;
        ref_mem_map[DNNL_ARG_DST] = dnn_mem_t(
                dst_md, dnnl_f32, tag::abx, ref_engine, /* prefill = */ false);
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb, dir_t dir) {
    std::vector<data_kind_t> check_kinds;
    if ((prb->dir & FLAG_FWD) && (dir & FLAG_FWD)) {
        check_kinds = {DST};
    } else if ((prb->dir & FLAG_BWD) && (dir & FLAG_BWD)) {
        check_kinds = {SRC};
    }
    // `check_kinds` is empty for `(prb->dir & FLAG_BWD) && (dir & FLAG_FWD)`.
    return check_kinds;
}

int createit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    v_prim.resize(2); // just fwd or fwd + bwd.
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res, FLAG_FWD,
                 nullptr, /* is_service_prim = */ prb->dir & FLAG_BWD),
            WARN);
    if (prb->dir & FLAG_BWD) {
        SAFE(init_prim(prb->ctx_init, v_prim[1], init_pd, prb, res, FLAG_BWD,
                     query_pd(v_prim[0])),
                WARN);
    }
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        SAFE(check_total_size(res), WARN);
        if (v_prim[1]) SAFE(check_total_size(res), WARN);
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
        if (v_prim[1]) {
            SAFE(check_caches(v_prim[1], prb->ctx_init, res), WARN);
        }
    }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

    const auto &prim = prb->dir & FLAG_FWD ? v_prim[0] : v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    SAFE(init_memory_args(
                 mem_map, prb, v_prim[0], res, /*override_dir_with_fwd=*/true),
            WARN);
    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, v_prim[0], prb, res),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(v_prim[0], args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb, FLAG_FWD), args, ref_args,
            compute_ref, setup_cmp, res, FLAG_FWD);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb, FLAG_FWD), args, prb->attr,
                 prb->inplace, res),
            WARN);

    if (prb->dir & FLAG_BWD) {
        // Pass same memory map as we need data from forward on backward.
        SAFE(init_memory_args(mem_map, prb, v_prim[1], res), WARN);
        TIME_FILL(SAFE(
                init_ref_memory_args(ref_mem_map, mem_map, v_prim[1], prb, res),
                WARN));

        args = args_t(mem_map);
        ref_args = args_t(ref_mem_map);

        SAFE(run_execution(v_prim[1], args, res), WARN);

        check_correctness(prb, get_kinds_to_check(prb, FLAG_BWD), args,
                ref_args, compute_ref, setup_cmp, res, FLAG_BWD);
        SAFE(check_bitwise(prim, get_kinds_to_check(prb, FLAG_BWD), args,
                     prb->attr, prb->inplace, res),
                WARN);
    }

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace eltwise
