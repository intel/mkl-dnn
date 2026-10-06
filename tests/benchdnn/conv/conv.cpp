/*******************************************************************************
* Copyright 2017 Intel Corporation
* Copyright 2021 FUJITSU LIMITED
* Copyright 2021 Arm Ltd. and affiliates
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

#include <algorithm>
#include <cstring>
#include <random>
#include <vector>

#include "oneapi/dnnl/dnnl.h"

#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "conv/conv.hpp"

namespace conv {

double get_non_zero_trust_percent(const prb_t *prb, data_kind_t kind) {
    // TODO: duplicated with compare internal logic.
    auto negative_to_zero = [&]() {
        using pk = attr_t::post_ops_t::kind_t;
        const auto &po = prb->attr.post_ops;
        int count = 0;

        // Check for all post-ops that convert negative to zero
        std::vector<pk> non_neg_po {pk::ABS};
        std::vector<pk> non_neg_alpha_0_po {
                pk::CLIP, pk::CLIP_V2, pk::ELU, pk::RELU};
        for (int i = 0; i < po.len(); ++i) {
            const auto &e = po.entry[i];
            if (!e.is_eltwise_kind()) continue;

            auto k = e.kind;
            auto alpha = e.eltwise.alpha;

            count += std::any_of(non_neg_po.cbegin(), non_neg_po.cend(),
                    [k](const pk alg) { return alg == k; });
            count += std::any_of(non_neg_alpha_0_po.cbegin(),
                    non_neg_alpha_0_po.cend(), [k, alpha](const pk alg) {
                return alg == k && alpha == 0;
            });
        }
        // Check for u8 dst
        count += prb->get_dt(DST) == dnnl_u8;
        // Check for physically padded area in the output
        count += prb->od > prb->id || prb->oh > prb->ih || prb->ow > prb->iw;

        return !!count;
    };

    double trust = 0.3; /* why? */
    switch (kind) {
        case SRC: trust /= prb->sd * prb->sh * prb->sw; break;
        case WEI:
            trust /= 1. * prb->kd * prb->kh * prb->kw
                    / MIN3(prb->kd * prb->kh * prb->kw,
                            prb->id * prb->ih * prb->iw,
                            prb->od * prb->oh * prb->ow);
            break;
        case BIA: trust = 0.8; break;
        case DST: trust /= (1.f + negative_to_zero()); break;
        default: assert(!"unsupported data kind");
    }

    return trust;
}

int check_reorder_presence(
        const prb_t *prb, dnn_mem_t &mem_dt, dnn_mem_t &mem_fp, res_t *res) {
    if (!is_cpu()) return OK;

    dnnl_data_type_t dt_check = dnnl_s8;
#if defined(DNNL_AARCH64) && (DNNL_AARCH64 == 1)
    /* Note for x64:
    Both data types of src and weight are s8, oneDNN addds 128 to one of the s8
    input to make it of type u8 instead, as explained in
    https://uxlfoundation.github.io/oneDNN/dev_guide_int8_computations.html or
    doc/advanced/int8_computations.md
    It is because `VPDPBUSD` instruction uses the combination of s8 and u8 as
    input.

    Note for AArch64:
    Because dot product instructions of AArch64 "SDOT" receives s8 input
    for both src and weight, the addition (and its counterpart of subtraction)
    is not required for AArch64.
    */
    if (res->impl_name.find("jit", 0) == 0) dt_check = dnnl_u8;
#endif

    const bool wei_x8x8
            = prb->get_dt(WEI) == dnnl_s8 && prb->get_dt(SRC) == dt_check;
    const bool is_def_zp = prb->attr.zero_points.is_def(DNNL_ARG_SRC);
    if (wei_x8x8 || !is_def_zp) {
        // A work around zmalloc registry checker: temporarily increase the
        // capacity just for this check since there's no simple way to account
        // for memory allocated here to verify an extra reorder.
        size_t extra_reorder_mem_size
                = (dnnl_memory_desc_get_size(mem_fp.md_) / 4)
                + dnnl_memory_desc_get_size(mem_dt.md_);
        res->mem_size_args.zmalloc_expected_size += extra_reorder_mem_size;
        set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);

        // Check that s8 -> s8_comp exists in the library since users may have
        // already quantized data.
        dnn_mem_t mem_fp_s8(mem_fp.md_, dnnl_s8, tag::abx, get_cpu_engine(),
                /* prefill = */ true);
        dnn_mem_t mem_dt_s8(
                mem_dt.md_, get_test_engine(), /* prefill = */ true);
        SAFE(mem_fp_s8.reorder(mem_fp, res), WARN);
        SAFE(mem_dt_s8.reorder(mem_fp_s8, res), WARN);
        SAFE(mem_dt.size() == mem_dt_s8.size() ? OK : FAIL, WARN);
        int rc = std::memcmp((void *)mem_dt, (void *)mem_dt_s8, mem_dt.size());
        SAFE(rc == 0 ? OK : FAIL, WARN);

        // Subtract to restore the original size.
        res->mem_size_args.zmalloc_expected_size -= extra_reorder_mem_size;
    }
    // Must be done in a separate scope to have extra memory objects destroyed
    // before updating the limit to an original value.
    if (wei_x8x8 || !is_def_zp) {
        set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);
    }

    return OK;
}

int fill_data(data_kind_t kind, int exec_arg, const prb_t *prb,
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
    density_args.n_acc = prb->count_n_acc();
    const auto density = cfg.get_density(density_args);

    // Apply the adjustments for weights only, they need to be even.
    // See `cfg.cpp` for more comments.
    const bool is_s8s8 = kind == WEI && cfg.get_dt(SRC) == dnnl_s8
            && cfg.get_dt(WEI) == dnnl_s8;

    const auto &e_zp_src = prb->attr.zero_points.get(DNNL_ARG_SRC);
    const bool has_src_zp = !e_zp_src.is_def();
    const int src_zp_mask = prb->attr.zero_points.get_mask(
            DNNL_ARG_SRC, dnnl_convolution, prb->ndims, prb->has_groups);
    // Apply src_zp for source tensor only.
    int src_zp = kind == SRC && has_src_zp && src_zp_mask == 0 ? e_zp_src.value
                                                               : 0;

    const auto &e_zp_wei = prb->attr.zero_points.get(DNNL_ARG_WEIGHTS);
    const bool has_wei_zp = !e_zp_wei.is_def();
    const int wei_zp_mask = prb->attr.zero_points.get_mask(
            DNNL_ARG_WEIGHTS, dnnl_convolution, prb->ndims, prb->has_groups);
    // Apply wei_zp for weights tensor only.
    int wei_zp = kind == WEI && has_wei_zp && wei_zp_mask == 0 ? e_zp_wei.value
                                                               : 0;

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t chunk_size = 64;
    const int64_t n_chunks = div_up(nelems, chunk_size);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);
        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand int_seed(kind * nelems + idx_start + 1);
        int_seed.discard(1);
        std::minstd_rand b_seed(kind * nelems + idx_start + 1);
        b_seed.discard(10);

        std::uniform_int_distribution<> gen(
                cfg.get_range_min(kind), cfg.get_range_max(kind));
        std::bernoulli_distribution b_dist(density);

        // make sure the first element is positive
        if (idx_start == 0) {
            float gen_val = 0;
            while (gen_val <= 0)
                gen_val = gen(int_seed);
            float val = gen_val * (1.f + is_s8s8);
            val += src_zp + wei_zp; // Add zp so that it will be subtracted.
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
            float val = gen(int_seed) * (1.f + is_s8s8);
            val += src_zp + wei_zp; // Add zp so that it will be subtracted.
            mem_fp.set_f32_elem(
                    idx, round_to_nearest_representable(cfg.get_dt(kind), val));
        }
    });

    SAFE(mem_dt.reorder(mem_fp, res, cfg.get_swapped_dt(kind)), WARN);

    if (kind == WEI)
        SAFE(check_reorder_presence(prb, mem_dt, mem_fp, res), WARN);

    return OK;
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_d, prb->ndims,
            prb->src_dims().data(), force_f32_dt ? dnnl_f32 : prb->get_dt(SRC),
            prb->stag, prb->strides[STRIDES_SRC]));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> wei_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(wei_d, prb->ndims + prb->has_groups,
            prb->wei_dims().data(), force_f32_dt ? dnnl_f32 : prb->get_dt(WEI),
            prb->wtag, prb->strides[STRIDES_WEI]));
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> bia_d {};
    if (prb->bia_dt() != dnnl_data_type_undef) {
        DNN_SAFE_STATUS(dnn_mem_t::init_md(bia_d, 1, prb->bia_dims().data(),
                force_f32_dt ? dnnl_f32 : prb->get_dt(BIA), tag::any));
    }
    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_d, prb->ndims,
            prb->dst_dims().data(), force_f32_dt ? dnnl_f32 : prb->get_dt(DST),
            prb->dtag, prb->strides[STRIDES_DST]));

    dnnl_alg_kind_t alg = dnnl_convolution_direct;
    if (prb->alg == WINO) alg = dnnl_convolution_winograd;
    if (prb->alg == AUTO) alg = dnnl_convolution_auto;

    attr_args_t attr_args;
    DNN_SAFE_STATUS(attr_args.prepare_post_ops_mds(
            prb->attr, prb->ndims, prb->dst_dims().data()));
    auto wei_scale = prb->attr.scales.get(DNNL_ARG_WEIGHTS);
    if (wei_scale.policy == policy_t::PER_OC) {
        // oihw: per_oc: 1 << 0 -> 1
        // goihw: per_oc: 1 << 1 + 1 << 0 -> 3
        auto wei_mask = prb->has_groups ? 3 : 1;
        attr_args.prepare_quant(
                prb->attr, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_mask);
    }
    auto dw_wei_scale
            = prb->attr.scales.get(DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS);
    if (dw_wei_scale.policy == policy_t::PER_OC) {
        // dw fusion always has groups.
        auto wei_mask = 3;
        attr_args.prepare_quant(prb->attr,
                DNNL_ARG_ATTR_SCALES | DNNL_ARG_ATTR_POST_OP_DW
                        | DNNL_ARG_WEIGHTS,
                wei_mask);
    }
    attr_args.prepare_dw_post_op(prb->attr, prb->get_dt(WEI), prb->bia_dt());
    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args, prb->ndims));

    switch (prb->dir) {
        case FWD_D:
        case FWD_B:
        case FWD_I:
            TIME_C_PD(DNN_SAFE_STATUS(
                    dnnl_convolution_forward_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine,
                            prb->dir == FWD_I ? dnnl_forward_inference
                                              : dnnl_forward_training,
                            alg,
                            init_pd_args.src_md ? init_pd_args.src_md : src_d,
                            wei_d, bia_d, dst_d, prb->kstrides().data(),
                            prb->dilations().data(), prb->padding().data(),
                            prb->padding_r().data(), dnnl_attr)));
            break;
        case BWD_D:
            TIME_C_PD(DNN_SAFE_STATUS(
                    dnnl_convolution_backward_data_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, alg, src_d,
                            wei_d, dst_d, prb->kstrides().data(),
                            prb->dilations().data(), prb->padding().data(),
                            prb->padding_r().data(), init_pd_args.hint,
                            dnnl_attr)));
            break;
        case BWD_W:
        case BWD_WB:
            TIME_C_PD(DNN_SAFE_STATUS(
                    dnnl_convolution_backward_weights_primitive_desc_create(
                            &init_pd_args.pd, init_pd_args.engine, alg, src_d,
                            wei_d, bia_d, dst_d, prb->kstrides().data(),
                            prb->dilations().data(), prb->padding().data(),
                            prb->padding_r().data(), init_pd_args.hint,
                            dnnl_attr)));
            break;
        default: DNN_SAFE_STATUS(dnnl_invalid_arguments);
    }

    // TODO: add query for acc type in pd.
    //DNN_SAFE_STATUS(cd.accum_data_type == prb->get_dt(ACC)
    //                ? dnnl_success
    //                : dnnl_unimplemented);
    return dnnl_success;
}

int init_prim_ref(benchdnn_dnnl_wrapper_t<dnnl_primitive_t> &prim_ref,
        const prb_t *prb, res_t *res) {
    if (!(has_bench_mode_bit(mode_bit_t::corr) && fast_ref)) return OK;
    // Create prim_ref if only original prim was successfully created.
    if (res->state != INITIALIZED) return OK;

    // f32 cases should go through reference no matter what.
    if (is_cpu() && (prb->src_dt() == dnnl_f32 && prb->wei_dt() == dnnl_f32))
        return OK;

    // Wino inputs doesn't suit optimized CPU implementation.
    if (prb->alg == WINO) return OK;

    std::vector<std::vector<dnnl_data_type_t>> prim_ref_dt {
            prb->dt, {dnnl_f32}};
    // If there's no bias, undef data type should be used for prim_ref as well.
    dnnl_data_type_t cpu_bia_dt
            = prb->bia_dt() == dnnl_data_type_undef ? prb->bia_dt() : dnnl_f32;
    std::vector<dnnl_data_type_t> prim_ref_bia_dt {prb->bia_dt(), cpu_bia_dt};
    if (is_cpu()) {
        prim_ref_dt.erase(prim_ref_dt.begin());
        prim_ref_bia_dt.erase(prim_ref_bia_dt.begin());
    }

    for_(const auto &prim_ref_dt_i : prim_ref_dt)
    for (const auto &prim_ref_bia_dt_i : prim_ref_bia_dt) {
        auto cpu_attr = prb->attr;
        update_cpu_ref_attrs(cpu_attr, prim_ref_dt_i.back());

        // Create a new copy of prb to avoid potentially corrupting the test by
        // modifying prb in place.
        // `DIRECT` algorithm is used to prevent fallback to the slow benchdnn
        // reference implementation.
        prb_t prb_cpu {*prb, prb->dir, prim_ref_dt_i, prim_ref_bia_dt_i,
                tag::any, tag::any, tag::any, {vdims_t(STRIDES_SIZE)}, DIRECT,
                prb->mb, cpu_attr, prb->ctx_init, prb->ctx_exe,
                prb->impl_filter};

        auto st = init_prim_ref_common(prim_ref, &prb_cpu, res, init_pd);
        if (st == OK) return OK;
    }

    prim_ref.reset(nullptr);
    return OK;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    skip_unimplemented_data_type({prb->get_dt(SRC), prb->get_dt(WEI),
                                         prb->get_dt(BIA), prb->get_dt(DST)},
            prb->dir, res);
    skip_unimplemented_sum_po(prb->attr, res, dnnl_convolution,
            prb->get_dt(SRC), prb->get_dt(DST));
    skip_unimplemented_binary_po(prb->attr, res);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_convolution);

    if (is_cpu()) {
        // Specific configurations are not supported.
        const bool is_f32_src = prb->get_dt(SRC) == dnnl_f32;
        const bool is_f32_wei = prb->get_dt(WEI) == dnnl_f32;
        const bool is_f16 = prb->get_dt(WEI) == dnnl_f16;
        const bool is_bf16_src = prb->get_dt(SRC) == dnnl_bf16;
        const bool is_bf16_wei = prb->get_dt(WEI) == dnnl_bf16;
        const bool is_int8_dst
                = prb->get_dt(DST) == dnnl_s8 || prb->get_dt(DST) == dnnl_u8;
        const bool is_f32f32x8 = is_f32_src && is_f32_wei && is_int8_dst;
        const bool is_bf16bf16x8 = is_bf16_src && is_bf16_wei && is_int8_dst;
        const bool is_valid_f16 = IMPLICATION(is_f16,
                prb->get_dt(DST) == dnnl_f32 || prb->get_dt(DST) == dnnl_f16);
        const bool is_int8_src
                = prb->get_dt(SRC) == dnnl_s8 || prb->get_dt(SRC) == dnnl_u8;
        const bool is_int8_wei
                = prb->get_dt(WEI) == dnnl_s8 || prb->get_dt(WEI) == dnnl_u8;
        const bool is_f16_dst = prb->get_dt(DST) == dnnl_f16;
        const bool is_x8x8f16 = is_int8_src && is_int8_wei && is_f16_dst;
        const bool is_wei_zp = !prb->attr.zero_points.is_def(DNNL_ARG_WEIGHTS);
        const bool is_non_s32_src_zp
                = prb->attr.zero_points.get(DNNL_ARG_SRC).dt != dnnl_s32;
        const bool is_non_unit_dst_scale
                = !prb->attr.scales.is_def(DNNL_ARG_DST)
                && prb->attr.scales.get_mask(DNNL_ARG_DST, dnnl_convolution)
                        > 0;

        if (is_f32f32x8 || is_bf16bf16x8 || is_x8x8f16 || !is_valid_f16
                || is_wei_zp || is_non_s32_src_zp || is_non_unit_dst_scale) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
            return;
        }
    }

    // Winograd implementation has very limited scope and support. It doesn't
    // make sense to list all of them, just convert all unimplemented Winograd
    // problems into not supported.
    if (prb->alg == WINO) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }

    // GPU does not support depthwise fusion
    if (is_gpu() && prb->attr.post_ops.convolution_index() != -1) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
        return;
    }
}

void prb_t::skip_invalid(res_t *res) const {}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    const bool allow_norm_check = (prb->alg & WINO);
    cmp.set_allow_norm_check(allow_norm_check);

    float trh = 0.f;
    if (prb->alg & WINO) {
        trh = prb->dt[1] == dnnl_f16 ? 7e-3f : 2e-5f;
        if (prb->dir & FLAG_WEI) {
            // This is an empirical equation derived by observing growth error
            // with increasing 'k' dimension in gemm of winograd
            const float log_const = log10(0.125 * prb->mb * prb->oh * prb->ow);
            trh *= (MAX2(1, pow(10, 0.4 * log_const)));
        }
    }
    cmp.set_threshold(trh);

    const float zpp = (1.f - get_non_zero_trust_percent(prb, kind)) * 100.f;
    cmp.set_zero_trust_percent(zpp);
}

std::vector<int> prb_t::supported_exec_args(bool override_dir_with_fwd) const {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DST,
            DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_WEIGHTS,
            DNNL_ARG_ATTR_POST_OP_DW | DNNL_ARG_BIAS,
    };
    static const std::vector<int> exec_bwd_d_args = {
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_WEIGHTS,
            DNNL_ARG_BIAS,
            DNNL_ARG_DIFF_DST,
    };
    static const std::vector<int> exec_bwd_w_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_DIFF_WEIGHTS,
            DNNL_ARG_DIFF_BIAS,
            DNNL_ARG_DIFF_DST,
    };
    return (override_dir_with_fwd || (dir & FLAG_FWD)) ? exec_fwd_args
            : (dir & FLAG_WEI)                         ? exec_bwd_w_args
                                                       : exec_bwd_d_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    const auto *prb = prb_t::from(base_prb);
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    const auto &ref_engine = get_cpu_engine();

    // Memory filling is the first one who uses updated problem alg.
    if (prb->alg == AUTO)
        prb->alg = alg_kind2alg(query_alg_kind(query_pd(prim)));

    // Move cfg out of filling since its creation is not free.
    cfg_t cfg(prb, {SRC, WEI, BIA, DST});

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second; // `mem` is modified by filler (reorder).

        if (exec_arg == DNNL_ARG_SRC && (prb->dir & FLAG_FWD)
                && prb->alg != WINO) {
            // Switch the format tag from "abx" to "axb" for forward problems.
            // Winograd is dead for optimizations.
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::axb, ref_engine,
                            /* prefill = */ false));
        } else if (exec_arg == DNNL_ARG_WEIGHTS && (prb->dir & FLAG_FWD)
                && prb->alg != WINO) {
            // Switch the format tag from "abx" to "axb" for forward problems.
            // Handle groups with "abxc" meta-tag.
            // Winograd is dead for optimizations.
            std::string axb_tag = prb->has_groups ? "abxc" : "axb";
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, axb_tag, ref_engine,
                            /* prefill = */ false));
        } else if (exec_arg != DNNL_ARG_SCRATCHPAD) {
            // Scratchpad memory relates to a primitive. If reference needs it,
            // use switch below to define a memory desc for it.
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                            /* prefill = */ false));
        }
        auto &ref_mem = ref_mem_map[exec_arg];

        switch (exec_arg) {
            case DNNL_ARG_SRC:
                SAFE(fill_data(SRC, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_WEIGHTS:
                SAFE(fill_data(WEI, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_BIAS:
                SAFE(fill_data(BIA, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            case DNNL_ARG_DST:
                if (prb->attr.post_ops.find(attr_t::post_ops_t::kind_t::SUM)
                        >= 0) {
                    SAFE(fill_data(DST, exec_arg, prb, cfg, mem, ref_mem, res),
                            WARN);
                    // Bitwise mode for sum requires a copy due to data for
                    // post-op will be overwritten and it must be refreshed.
                    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
                        SAFE(mem_map.at(-exec_arg).reorder(ref_mem, res), WARN);
                    }
                }
                break;
            case DNNL_ARG_DIFF_DST:
                SAFE(fill_data(DST, exec_arg, prb, cfg, mem, ref_mem, res),
                        WARN);
                break;
            default:
                SAFE(init_ref_memory_args_default_case(
                             exec_arg, mem, ref_mem, prb->attr, res),
                        WARN);
                break;
        }

        update_ref_mem_map_from_prim(prim_ref, mem, ref_mem_map, exec_arg,
                cfg.get_swapped_dt(exec_arg2data_kind(exec_arg)), res);

        // Don't keep reference memory if it is not used further.
        if (!has_bench_mode_bit(mode_bit_t::corr)) ref_mem_map.clear();
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb) {
    std::vector<data_kind_t> check_kinds;
    if (prb->dir & FLAG_FWD) {
        check_kinds = {DST};
    } else if (prb->dir == BWD_D) {
        check_kinds = {SRC};
    } else if (prb->dir & FLAG_BWD && prb->dir & FLAG_WEI) {
        check_kinds = {WEI};
        if (prb->bia_dt() != dnnl_data_type_undef) check_kinds.push_back(BIA);
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
    v_prim.resize(2); // regular + cpu_ref
    SAFE(init_prim(prb->ctx_init, v_prim[0], init_pd, prb, res), WARN);
    // Use CPU prim as the reference in GPU testing to reduce testing time.
    SAFE(init_prim_ref(v_prim[1], prb, res), WARN);
    return OK;
}

int checkit(std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    if (has_bench_mode_bit(mode_bit_t::exec)) {
        const auto &prim_ref = v_prim[1];
        if (prim_ref) {
            // Copy res to avoid save/restore state and reason.
            res_t res_copy = *res;
            SAFE(check_total_size(&res_copy, prim_ref), WARN);
            if (res_copy.state == SKIPPED) {
                v_prim[1].reset(nullptr);
                SAFE(check_total_size(res), WARN);
            } else {
                // Copy estimations back to original `res`.
                *res = res_copy;
            }
        } else {
            SAFE(check_total_size(res), WARN);
        }
    }
    if (has_bench_mode_bit(mode_bit_t::corr)) {
        SAFE(check_caches(v_prim[0], prb->ctx_init, res), WARN);
        // Don't check caches for CPU prim as the reference.
    }
    return OK;
}

int doit(const std::vector<benchdnn_dnnl_wrapper_t<dnnl_primitive_t>> &v_prim,
        const base_prb_t *base_prb, res_t *res) {
    const prb_t *prb = prb_t::from(base_prb);
    set_zmalloc_max_expected_size(res->mem_size_args.zmalloc_expected_size);
    // TODO: move Winograd's reference implementation scratchpad to a dedicated
    // class for ability to query sizes.
    // So far, just increase the size twice and let it roll.
    if (prb->alg == WINO) {
        set_zmalloc_max_expected_size(
                2 * res->mem_size_args.zmalloc_expected_size);
    }

    const auto &prim = v_prim[0];
    const auto &prim_ref = v_prim[1];

    dnn_mem_map_t mem_map, ref_mem_map;
    SAFE(init_memory_args(mem_map, prb, prim, res), WARN);
    TIME_FILL(SAFE(init_ref_memory_args(
                           ref_mem_map, mem_map, prim, prb, res, prim_ref),
            WARN));

    args_t args(mem_map), ref_args(ref_mem_map);

    SAFE(run_execution(prim, args, res), WARN);

    check_correctness(prb, get_kinds_to_check(prb), args, ref_args, compute_ref,
            setup_cmp, res, prb->dir, prim_ref);
    SAFE(check_bitwise(prim, get_kinds_to_check(prb), args, prb->attr,
                 prb->inplace, res),
            WARN);

    return measure_perf(prb->ctx_exe, res, prim, args);
}

} // namespace conv
