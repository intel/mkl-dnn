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

#include <cmath>
#include <limits>
#include <math.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>

#include "oneapi/dnnl/dnnl.h"

// Internal alg_kind used by the GPU SDPA kernel. Must be removed once
// softmax_accurate_inf_as_zero is promoted to a public value.
#include "src/common/c_types_map.hpp"

#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "sdpa/sdpa.hpp"

namespace sdpa {

benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> create_md(int ndims,
        const dims_t &dims, dnnl_data_type_t dt, const std::string &tag) {
    return dnn_mem_t::init_md(ndims, dims.data(), dt, tag);
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;
    const dir_t dir = init_pd_args.dir;

    auto q_dt = force_f32_dt ? dnnl_f32 : prb->q_dt();
    auto k_dt = force_f32_dt ? dnnl_f32 : prb->k_dt();
    auto v_dt = force_f32_dt ? dnnl_f32 : prb->v_dt();
    auto dst_dt = force_f32_dt ? dnnl_f32 : prb->dst_dt();

    auto q_d = create_md(prb->ndims, prb->q_dims(), q_dt, prb->qtag);
    auto k_d = create_md(prb->ndims, prb->k_dims(), k_dt, prb->ktag);
    auto v_d = create_md(prb->ndims, prb->v_dims(), v_dt, prb->vtag);
    auto dst_d = create_md(prb->ndims, prb->dst_dims, dst_dt, prb->dtag);

    // Attention mask (optional).
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> mask_d {};
    if (prb->with_mask()) {
        mask_d = create_md(prb->ndims, prb->msk_dims, prb->mdt, tag::abx);
    }

    // Always pass a valid md — the API unconditionally dereferences it.
    auto scale_d = dnn_mem_t::init_host_scalar_md(dnnl_f32);

    // Map benchdnn mask_type_t to API's dnnl_attn_mask_type_t values.
    // All buffer variants (full, 1D, 2D) use dnnl_attn_mask_buffer (1).
    int attn_mask_type_val = [](mask_type_t mt) {
        switch (mt) {
            case MASK_NONE: return 0; // dnnl_attn_mask_undef
            case MASK_BUFFER:
            case MASK_BUFFER_1D:
            case MASK_BUFFER_2D: return 1; // dnnl_attn_mask_buffer
            case MASK_CAUSAL_TOP_LEFT: return 2; // dnnl_attn_mask_top_left
            case MASK_CAUSAL_BOTTOM_RIGHT:
                return 3; // dnnl_attn_mask_bottom_right
            default: return 0;
        }
    }(prb->mask_type);
    dnnl_alg_kind_t softmax_alg = static_cast<dnnl_alg_kind_t>(
            dnnl::impl::alg_kind::softmax_accurate_inf_as_zero);

    // KV head count is always derived from the K tensor's head dimension.
    dnnl_dim_t kv_hn = prb->k_dims()[prb->ndims - 3];

    // Default to invert_scale=false; scale is filled in init_ref_memory_args.
    bool invert = prb->with_scale() ? prb->invert_scale() : false;

    // Build attr_args for dropout mask md if configured.
    attr_args_t attr_args;
    if (!prb->attr.dropout.is_def()) {
        attr_args.prepare_post_ops_mds(
                prb->attr, prb->ndims, prb->score_dims.data());
    }
    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args, prb->ndims));

    const auto mask_ptr
            = prb->with_mask() ? (const_dnnl_memory_desc_t)mask_d : nullptr;

    if (dir & FLAG_FWD) {
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;

        TIME_C_PD(DNN_SAFE_STATUS(sdpa_primitive_desc_create(&init_pd_args.pd,
                init_pd_args.engine, q_d, k_d, v_d, dst_d, mask_ptr, scale_d,
                invert, kv_hn, attn_mask_type_val, softmax_alg, prop, dnnl_attr,
                /* kq_attr = */ nullptr,
                /* vs_attr = */ nullptr)));
    } else {
        auto diff_q_d = create_md(prb->ndims, prb->q_dims(), q_dt, prb->qtag);
        auto diff_k_d = create_md(prb->ndims, prb->k_dims(), k_dt, prb->ktag);
        auto diff_v_d = create_md(prb->ndims, prb->v_dims(), v_dt, prb->vtag);
        auto diff_dst_d
                = create_md(prb->ndims, prb->dst_dims, dst_dt, prb->dtag);

        TIME_C_PD(DNN_SAFE_STATUS(sdpa_primitive_desc_create(&init_pd_args.pd,
                init_pd_args.engine, q_d, k_d, v_d, dst_d, mask_ptr, scale_d,
                diff_q_d, diff_k_d, diff_v_d, diff_dst_d,
                /* dS = */ nullptr, invert, kv_hn, attn_mask_type_val,
                softmax_alg, dnnl_attr, init_pd_args.hint)));
    }

    return dnnl_success;
}

int fill_data(int exec_arg, data_kind_t kind, const prb_t *prb,
        const cfg_t &cfg, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(exec_arg, mem_dt, mem_fp, res)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, res);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, res, get_perf_fill_cfg(mem_dt.dt()));
    }

    cfg_t::density_args_t density_args;
    density_args.data_kind = kind;
    density_args.n_acc = prb->head_size;
    const auto density = cfg.get_density(density_args);

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::bernoulli_distribution b_dist(density);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));

        // Make sure the first element is positive.
        if (idx_start == 0) {
            float val = 0;
            while (val <= 0)
                val = gen(int_seed);
            mem_fp.set_f32_elem(
                    0, round_to_nearest_representable(cfg.get_dt(kind), val));
            idx_start += 1;
        }

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            bool is_one = density == 1.f ? true : b_dist(b_seed);
            if (!is_one) {
                mem_fp.set_f32_elem(idx, 0.f);
                continue;
            }
            float val = gen(int_seed);
            mem_fp.set_f32_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    SAFE(mem_dt.reorder(mem_fp, res, cfg.get_swapped_dt(kind)), WARN);
    return OK;
}

// Fill attention mask with realistic 0 / -inf values (~25 % masked).
int fill_mask(dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;

    const float neg_inf = -std::numeric_limits<float>::infinity();
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        std::minstd_rand rng(nelems + idx_start + 1);
        rng.discard(1);

        for (int64_t idx = idx_start; idx < idx_end; ++idx)
            mem_fp.set_f32_elem(idx, (rng() % 4 == 0) ? neg_inf : 0.f);
    });

    SAFE(mem_dt.reorder(mem_fp, res), WARN);
    return OK;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    skip_unimplemented_data_type(
            {prb->q_dt(), prb->k_dt(), prb->v_dt(), prb->dst_dt()}, prb->dir,
            res);

    // SDPA is currently only implemented for GPU.
    if (is_cpu()) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    // No fallback implementation: skip cases the device does not support
    res->state = SKIPPED;
    res->reason = reason_t::skip_not_supported;
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    // SDPA API requires 4D tensors (batch x heads x seq x head_dim).
    // Guard must be here (not skip_unimplemented) because init_pd accesses
    // ndims-3 which would be OOB for 2D/3D inputs.
    if (prb->ndims < 4) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }

    // Validate dimension consistency.
    const auto &qdims = prb->q_dims();
    const auto &kdims = prb->k_dims();
    const auto &vdims = prb->v_dims();
    int nd = prb->ndims;

    // Q head_size must match K row dim.
    if (qdims[nd - 1] != kdims[nd - 2]) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
    // K col dim must match V row dim.
    if (kdims[nd - 1] != vdims[nd - 2]) {
        res->state = SKIPPED;
        res->reason = reason_t::invalid;
        return;
    }
}

// Intermediates (P, dS, O) are rounded to the dst type.
float intermediate_min_normal(const prb_t *prb) {
    switch (prb->dst_dt()) {
        case dnnl_f32:
        case dnnl_bf16: return std::numeric_limits<float>::min();
        case dnnl_f16:
        case dnnl_f8_e5m2: return std::ldexp(1.f, -14);
        case dnnl_f8_e4m3: return std::ldexp(1.f, -6);
        default: return 0.f;
    }
}

static float cmp_threshold(const prb_t *prb, data_kind_t kind) {
    const bool is_bwd = (kind == SRC || kind == SRC_1 || kind == SRC_2);
    const int nd = prb->ndims;
    const bool is_gqa = prb->k_dims()[nd - 3] != prb->q_dims()[nd - 3];

    // Roundings (eps/2 each): DST/dV: P, output; dQ/dK: O in D_i (eps), dS,
    // output; GQA dK/dV: per-head partials.
    float n_eps = (kind == SRC || kind == SRC_1) ? 2.f : 1.f;
    if (is_gqa && (kind == SRC_1 || kind == SRC_2)) n_eps += 0.5f;

    // f32 accumulation order differs from the reference. Backward's dP and
    // D_i sums over the V head dim feed directly into the dS magnitude.
    const int64_t n_acc = is_bwd
            ? 1 + prb->n_values + prb->n_keys + prb->n_queries
            : 1 + prb->n_keys;
    return n_eps * epsilon_dt(prb->dst_dt())
            + 8.f * n_acc * epsilon_dt(dnnl_f32);
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    const bool is_bwd = (kind == SRC || kind == SRC_1 || kind == SRC_2);
    const float eps = epsilon_dt(prb->dst_dt());
    const float err_per_mag = cmp_threshold(prb, kind);

    // Subsumed by the magnitude check since mag >= |exp|.
    cmp.set_threshold(err_per_mag);

    // Cancellation can drive |exp| -> 0, so bound the error by the same
    // contraction over absolute values (see compute_ref).
    int mag_arg = SDPA_REF_ARG_OUT_ABSMAG;
    switch (kind) {
        case SRC: mag_arg = SDPA_REF_ARG_DQ_ABSMAG; break;
        case SRC_1: mag_arg = SDPA_REF_ARG_DK_ABSMAG; break;
        case SRC_2: mag_arg = SDPA_REF_ARG_DV_ABSMAG; break;
        default: break;
    }
    const dnn_mem_t *mag = &ref_args.find(mag_arg);
    // Subnormal outputs may round in opposite directions.
    const float subnormal_ulp = intermediate_min_normal(prb) * eps;
    cmp.set_driver_check_function(
            [err_per_mag, mag, subnormal_ulp](
                    const compare::compare_t::driver_check_func_args_t &a)
                    -> bool {
        return a.diff <= err_per_mag * mag->get_f32_elem(a.idx) + subnormal_ulp;
    });

    cmp.set_zero_trust_percent(is_bwd ? 70.f : 90.f);
}

std::vector<int> prb_t::supported_exec_args(bool override_dir_with_fwd) const {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_QUERIES,
            DNNL_ARG_KEYS,
            DNNL_ARG_VALUES,
            DNNL_ARG_DST,
            DNNL_ARG_ATTN_MASK,
            DNNL_ARG_WORKSPACE,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_QUERIES,
            DNNL_ARG_KEYS,
            DNNL_ARG_VALUES,
            DNNL_ARG_DST,
            DNNL_ARG_ATTN_MASK,
            DNNL_ARG_DIFF_QUERIES,
            DNNL_ARG_DIFF_KEYS,
            DNNL_ARG_DIFF_VALUES,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DS,
            DNNL_ARG_WORKSPACE,
    };
    return (override_dir_with_fwd || (dir & FLAG_FWD)) ? exec_fwd_args
                                                       : exec_bwd_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    const auto *prb = prb_t::from(base_prb);
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, DST});

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg == DNNL_ARG_SCRATCHPAD) continue;

        // Workspace connects forward and backward; not filled manually.
        if (exec_arg == DNNL_ARG_WORKSPACE) continue;

        if (exec_arg == DNNL_ARG_SCALE) {
            dnnl_dims_t s_dims = {1};
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(1, s_dims, dnnl_f32, tag::x, ref_engine,
                            /* prefill = */ false));
            auto &ref_mem = ref_mem_map[exec_arg];
            float scale_val = prb->invert_scale()
                    ? sqrtf(static_cast<float>(prb->head_size))
                    : 1.0f / sqrtf(static_cast<float>(prb->head_size));
            ref_mem.set_f32_elem(0, scale_val);
            mem.set_f32_elem(0, scale_val);
            continue;
        }

        ref_mem_map.emplace(exec_arg,
                dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                        /* prefill = */ false));
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_QUERIES:
                SAFE(fill_data(exec_arg, SRC, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_KEYS:
            case DNNL_ARG_VALUES:
                SAFE(fill_data(exec_arg, WEI, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DST: break;
            case DNNL_ARG_ATTN_MASK:
                SAFE(fill_mask(mem, ref_mem, res), WARN);
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_data(exec_arg, DST, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DIFF_QUERIES:
            case DNNL_ARG_DIFF_KEYS:
            case DNNL_ARG_DIFF_VALUES:
            case DNNL_ARG_DS: break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
                break;
        }

        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
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
        // Set memory used for reference.
        auto &mem_args = res->mem_size_args;
        if (has_bench_mode_bit(mode_bit_t::corr)) {
            mem_args.extra_size_driver = get_ref_extra_size(prb, FLAG_FWD);
        }

        SAFE(check_total_size(res), WARN);

        if (v_prim[1]) {
            // Forward reference runs as a service step, its buffers are freed
            // before the backward one allocates.
            if (has_bench_mode_bit(mode_bit_t::corr)) {
                mem_args.extra_size_driver = MAX2(mem_args.extra_size_driver,
                        get_ref_extra_size(prb, FLAG_BWD));
            }

            SAFE(check_total_size(res), WARN);
        }
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
        if (v_prim[1]) {
            SAFE(check_caches(v_prim[1], prb->ctx_init, res), WARN);
        }
    }
    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb, dir_t dir) {
    std::vector<data_kind_t> check_kinds;
    if (dir & FLAG_FWD) {
        // Skip forward validation when it runs as service for backward.
        if (!(prb->dir & FLAG_BWD)) check_kinds = {DST};
    } else {
        // Backward: check diff_Q (SRC), diff_K (SRC_1), diff_V (SRC_2).
        check_kinds = {SRC, SRC_1, SRC_2};
    }
    get_kinds_to_check_shared(check_kinds, prb->attr);
    return check_kinds;
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

    {
        auto scale_md = dnn_mem_t::init_host_scalar_md(dnnl_f32);
        mem_map.emplace(DNNL_ARG_SCALE, dnn_mem_t(scale_md));
    }

    TIME_FILL(SAFE(
            init_ref_memory_args(ref_mem_map, mem_map, v_prim[0], prb, res),
            WARN));
    // DST isn't checked for backward problems.
    if (has_bench_mode_bit(mode_bit_t::corr)
            && !has_bench_mode_modifier(mode_modifier_t::no_ref_memory)
            && !(prb->dir & FLAG_BWD)) {
        // Reference-only buffer holding the per-element conditioning magnitude
        // sum_k prob_k*|V_k| (same layout as the reference DST). compute_ref fills
        // it; setup_cmp reads it to size a per-element DST threshold. Forward only.
        const auto &ref_dst = ref_mem_map.at(DNNL_ARG_DST);
        ref_mem_map.emplace(SDPA_REF_ARG_OUT_ABSMAG,
                dnn_mem_t(
                        ref_dst.md_, get_cpu_engine(), /* prefill = */ false));
    }

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(v_prim[0], args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb, FLAG_FWD), args, ref_args,
            compute_ref, setup_cmp, res, FLAG_FWD);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb, FLAG_FWD), args, prb->attr,
                 prb->inplace, res),
            WARN);

    if (prb->dir & FLAG_BWD) {
        // Extend memory map with backward args.
        SAFE(init_memory_args(mem_map, prb, v_prim[1], res), WARN);

        // Re-add scale (pruned by init_memory_args since SCALE is not in
        // exec_bwd_args; init_ref_memory_args will fill the value).
        {
            auto scale_md = dnn_mem_t::init_host_scalar_md(dnnl_f32);
            mem_map.emplace(DNNL_ARG_SCALE, dnn_mem_t(scale_md));
        }

        TIME_FILL(SAFE(
                init_ref_memory_args(ref_mem_map, mem_map, v_prim[1], prb, res),
                WARN));

        if (has_bench_mode_bit(mode_bit_t::corr)
                && !has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) {
            const std::pair<int, int> mag_args[] = {
                    {SDPA_REF_ARG_DQ_ABSMAG, DNNL_ARG_DIFF_QUERIES},
                    {SDPA_REF_ARG_DK_ABSMAG, DNNL_ARG_DIFF_KEYS},
                    {SDPA_REF_ARG_DV_ABSMAG, DNNL_ARG_DIFF_VALUES},
            };
            for (const auto &e : mag_args) {
                const auto &ref_diff = ref_mem_map.at(e.second);
                ref_mem_map.emplace(e.first,
                        dnn_mem_t(ref_diff.md_, get_cpu_engine(),
                                /* prefill = */ false));
            }
        }

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

} // namespace sdpa
