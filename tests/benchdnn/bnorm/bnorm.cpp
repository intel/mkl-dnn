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

#include <float.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <sstream>

#include "oneapi/dnnl/dnnl.h"

#include "utils/dnnl_query.hpp"
#include "utils/fill.hpp"
#include "utils/memory.hpp"
#include "utils/parallel.hpp"

#include "dnnl_common.hpp"
#include "dnnl_memory.hpp"

#include "bnorm/bnorm.hpp"

namespace bnorm {

int fill_mean(const prb_t *prb, const cfg_t &cfg, dnn_mem_t &mem_fp,
        dnn_mem_t &mem_dt, res_t *res) {
    if (fill_from_file(DNNL_ARG_MEAN, mem_dt, mem_fp, res)) return OK;
    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        // If the library doesn't expect input mean, don't fill it.
        if (mem_dt.nelems() == 0) return OK;
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        // If the library doesn't expect input mean, don't fill it.
        if (mem_dt.nelems() == 0) return OK;
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        float val = 0.f;
        if (cfg.check_alg_ != ALG_0 || (prb->flags & GLOB_STATS))
            val = 0.25f * (1 << (c % 7));
        mem_fp.set_f32_elem(c, val);
    });

    if (mem_dt && prb->use_stats()) SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

int fill_src(const prb_t *prb, const cfg_t &cfg, dnn_mem_t &mem_fp,
        dnn_mem_t &mem_dt, const dnn_mem_t &ref_mean, res_t *res) {
    if (fill_from_file(DNNL_ARG_SRC, mem_dt, mem_fp, res)) return OK;
    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, res);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, res, get_perf_fill_cfg(mem_dt.dt()));
    }

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        const float m = ref_mean.get_f32_elem(c);
        for (int64_t mb = 0; mb < prb->mb; ++mb) {
            // l[0] must be even
            int64_t l_base = mb * prb->id * prb->ih * prb->iw + c * 239 * 2;
            int64_t off = data_off(prb, mb, c, 0, 0, 0);

            for_(int64_t d = 0; d < prb->id; ++d)
            for_(int64_t h = 0; h < prb->ih; ++h)
            for (int64_t w = 0; w < prb->iw; ++w) {
                const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                const int64_t l = l_base + sp;

                // Shortcut for zero values.
                if (cfg.check_alg_ == ALG_0
                        && !flip_coin(l / 2 * 257ULL, cfg.density_)) {
                    mem_fp.set_f32_elem(off + sp, 0);
                    continue;
                }

                const int64_t gen = (l / 2 * 1637) & cfg.flex_mask_;
                const int sgn = l % 2 == 0 ? 1 : -1; // s_{i} + s_{i+1} = 2 * m
                const float f = 1.f * sgn * gen / (1 << cfg.flex_bits_);

                float val = cfg.check_alg_ == ALG_0 ? f : m * (1.f + f);
                if (prb->flags & GLOB_STATS) {
                    val = (l % 65) - 32;
                } else if (cfg.L_ % 2
                        && (mb * prb->id * prb->ih * prb->iw + sp
                                == cfg.L_ - 1)) {
                    val = m;
                }
                mem_fp.set_f32_elem(
                        off + sp, round_to_nearest_representable(prb->dt, val));
            }
        }
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

int fill_variance(const prb_t *prb, const cfg_t &cfg, dnn_mem_t &mem_fp,
        dnn_mem_t &mem_dt, const dnn_mem_t &ref_src, const dnn_mem_t &ref_mean,
        res_t *res) {
    if (fill_from_file(DNNL_ARG_VARIANCE, mem_dt, mem_fp, res)) return OK;
    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        // If the library doesn't expect input variance, don't fill it.
        if (mem_dt.nelems() == 0) return OK;
        // Variance must be always positive by definition.
        fill_cfg_t fill_cfg(dnnl_f32, 0.f, 16.f, /* int = */ false,
                attr_t::post_ops_t::kind_t::ADD, "variance");
        return fill_random_real(mem_dt, mem_fp, nullptr, fill_cfg);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        // If the library doesn't expect input variance, don't fill it.
        if (mem_dt.nelems() == 0) return OK;
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        float val = 0.f;
        if (prb->flags & GLOB_STATS) {
            val = ((c % 7) << 1);
        } else {
            const float m = ref_mean.get_f32_elem(c);
            for (int64_t mb = 0; mb < prb->mb; ++mb) {
                int64_t off = data_off(prb, mb, c, 0, 0, 0);

                for_(int64_t d = 0; d < prb->id; ++d)
                for_(int64_t h = 0; h < prb->ih; ++h)
                for (int64_t w = 0; w < prb->iw; ++w) {
                    const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
                    const float s = ref_src.get_f32_elem(sp + off);
                    val += (s - m) * (s - m);
                }
            }
            val /= cfg.L_;
        }
        mem_fp.set_f32_elem(c, val);
    });

    if (mem_dt && prb->use_stats()) SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

int fill_src_add(const prb_t *prb, const cfg_t &cfg, dnn_mem_t &mem_fp,
        dnn_mem_t &mem_dt, const dnn_mem_t &ref_src, const dnn_mem_t &ref_mean,
        res_t *res) {
    const bool fill_src_add = prb->fuse_add_relu();
    if (!fill_src_add) return OK;
    if (fill_from_file(DNNL_ARG_SRC_1, mem_dt, mem_fp, res)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    benchdnn_parallel_nd(prb->ic, prb->mb, prb->id, prb->ih, prb->iw,
            [&](int64_t c, int64_t mb, int64_t d, int64_t h, int64_t w) {
        const int64_t l_base = mb * prb->id * prb->ih * prb->iw + c * 239 * 2;
        const int64_t sp = d * prb->ih * prb->iw + h * prb->iw + w;
        const int64_t offset = data_off(prb, mb, c, 0, 0, 0) + sp;
        const int64_t l = l_base + sp;
        const float s = ref_src.get_f32_elem(offset);
        const float m = ref_mean.get_f32_elem(c);

        if (!(prb->flags & GLOB_STATS) && s == 0) {
            mem_fp.set_f32_elem(offset, 1.f);
            return;
        }

        float val = 0.f;
        if (prb->flags & GLOB_STATS) {
            val = (l % 17) - 8;
        } else {
            // The main purpose of such filling is to avoid catastrophic
            // cancellation. To do that, the sign of `Add` tensor final
            // values is kept the same as it would be after applying
            // bnorm: what's below mean, that has negative sign, what's
            // equal or higher - positive.
            const int64_t mod2_base = (mb + c + d + h + w) % 5;
            const float mod2_val = 1.f / (2LL << mod2_base);
            const int64_t sign_val = s < m ? -1 : 1;
            val = mod2_val * sign_val;
        }
        mem_fp.set_f32_elem(
                offset, round_to_nearest_representable(prb->dt, val));
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

int fill_scale(
        const prb_t *prb, dnn_mem_t &mem_fp, dnn_mem_t &mem_dt, res_t *res) {
    const bool use_sc = prb->use_sc();
    if (!use_sc) return OK;
    if (fill_from_file(DNNL_ARG_SCALE, mem_dt, mem_fp, res)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        float val = (1.f / 8) * (1 << (c % 7));
        if (prb->flags & GLOB_STATS) val *= 8.f;
        mem_fp.set_f32_elem(c, val);
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

int fill_shift(
        const prb_t *prb, dnn_mem_t &mem_fp, dnn_mem_t &mem_dt, res_t *res) {
    const bool use_sh = prb->use_sh();
    if (!use_sh) return OK;
    if (fill_from_file(DNNL_ARG_SHIFT, mem_dt, mem_fp, res)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        return fill_random_real(mem_dt, mem_fp, nullptr);
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    benchdnn_parallel_nd(prb->ic, [&](int64_t c) {
        float val = ((c % 3) - 1) * (1.f / 512 * (1 << (c % 7)));
        if (prb->flags & GLOB_STATS) val *= 512.f;
        mem_fp.set_f32_elem(c, val);
    });

    if (mem_dt) SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

int prepare_fwd(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    cfg_t cfg(prb);

    auto &mean = mem_map.at(DNNL_ARG_MEAN);
    auto &ref_mean = ref_mem_map.at(DNNL_ARG_MEAN);
    SAFE(fill_mean(prb, cfg, ref_mean, mean, res), WARN);

    auto &src = mem_map.at(DNNL_ARG_SRC);
    auto &ref_src = ref_mem_map.at(DNNL_ARG_SRC);
    SAFE(fill_src(prb, cfg, ref_src, src, ref_mean, res), WARN);

    // Need a copy of source data for inplace mode for bitwise testing.
    if (has_bench_mode_bit(mode_bit_t::bitwise) && prb->inplace) {
        auto &src_copy = mem_map.at(-DNNL_ARG_SRC);
        SAFE(bool(src_copy) ? OK : FAIL, WARN);
        SAFE(src_copy.reorder(src, res), WARN);
    }

    auto &var = mem_map.at(DNNL_ARG_VARIANCE);
    auto &ref_var = ref_mem_map.at(DNNL_ARG_VARIANCE);
    SAFE(fill_variance(prb, cfg, ref_var, var, ref_src, ref_mean, res), WARN);

    auto &src_add = mem_map.at(DNNL_ARG_SRC_1);
    auto &ref_src_add = ref_mem_map.at(DNNL_ARG_SRC_1);
    SAFE(fill_src_add(prb, cfg, ref_src_add, src_add, ref_src, ref_mean, res),
            WARN);

    auto &scale = mem_map.at(DNNL_ARG_SCALE);
    auto &ref_scale = ref_mem_map.at(DNNL_ARG_SCALE);
    SAFE(fill_scale(prb, ref_scale, scale, res), WARN);

    auto &shift = mem_map.at(DNNL_ARG_SHIFT);
    auto &ref_shift = ref_mem_map.at(DNNL_ARG_SHIFT);
    SAFE(fill_shift(prb, ref_shift, shift, res), WARN);

    return OK;
}

int prepare_bwd(const prb_t *prb, dnn_mem_map_t &mem_map,
        dnn_mem_map_t &ref_mem_map, res_t *res) {
    auto &mem_fp = ref_mem_map.at(DNNL_ARG_DIFF_DST);
    auto &mem_dt = mem_map.at(DNNL_ARG_DIFF_DST);

    const auto nelems = mem_fp.nelems();
    if (nelems == 0) return OK;
    if (fill_from_file(DNNL_ARG_DIFF_DST, mem_dt, mem_fp, res)) return OK;

    // Refer to modes documentation for filling principles.
    if (has_bench_mode_bit(mode_bit_t::bitwise)) {
        SAFE(fill_random_real(mem_dt, mem_fp, nullptr), WARN);

        // Need a copy of source data for inplace mode for bitwise testing.
        if (prb->inplace) {
            auto &diff_dst_copy = mem_map.at(-DNNL_ARG_DIFF_DST);
            SAFE(bool(diff_dst_copy) ? OK : FAIL, WARN);
            SAFE(diff_dst_copy.reorder(mem_dt, res), WARN);
        }

        return OK;
    }
    if (has_bench_mode_bit(mode_bit_t::perf)) {
        return fill_random_real(
                mem_dt, mem_fp, nullptr, get_perf_fill_cfg(mem_dt.dt()));
    }

    // Idea behind filling: integer diff_dst values decrease norms unlike fp32
    // values in [-1.f, 1.f] range. To decrease norms more, make data pretty
    // sparse as answers sum all diff_dst values.

    /* Do fixed partitioning to have same filling for any number of threads */
    const int64_t n_chunks = 16;
    const int64_t chunk_size = div_up(nelems, n_chunks);

    benchdnn_parallel_nd(n_chunks, [&](int64_t idx_chunk) {
        int64_t idx_start = idx_chunk * chunk_size;
        int64_t idx_end = MIN2(idx_start + chunk_size, nelems);

        // Note: we use a different seed for each chunk to avoid
        // repeating patterns. We could use discard(idx_start) too but
        // it has a complexity in O(idx_start). We also add 1 to avoid
        // seeding with 0.
        std::minstd_rand msr(idx_start + 1);
        msr.discard(1);

        std::uniform_int_distribution<> igen_val(-2, 2);
        std::uniform_int_distribution<> igen_coin(0, 256 * 1024);

        // at least 20 non-zero elems
        float sparsity = MAX2(0.05f, MIN2(1.f, 20.f / nelems));

        for (int64_t idx = idx_start; idx < idx_end; ++idx) {
            float value = flip_coin(igen_coin(msr), sparsity)
                    ? round_to_nearest_representable(prb->dt, igen_val(msr))
                    : 0;
            mem_fp.set_f32_elem(idx, value);
        }
    });

    SAFE(mem_dt.reorder(mem_fp, res), WARN);

    return OK;
}

int check_fwd_ws(dnn_mem_map_t &mem_map, res_t *res) {
    const auto &dst_dt = mem_map.at(DNNL_ARG_DST);
    const auto &ws_dt = mem_map.at(DNNL_ARG_WORKSPACE);

    if (ws_dt.ndims() == 0) return OK;

    /* so far we know ws is just bit-mask of whether value was negative or
     * positive */
    const auto nelems = dst_dt.nelems(true);
    const uint8_t *ws = (const uint8_t *)ws_dt;

    /* some internal knowledge: flags in ws are either stored as bytes (e.g.
     * for the ref implementation) or as bits (e.g. for the jitted one); in
     * the latter case the ws memory has fewer elements than the data memory */
    enum { ws_byte, ws_bit } ws_type;
    ws_type = ws_dt.nelems(true) < nelems ? ws_bit : ws_byte;

    /* more internal knowledge: dst_dt and ws_dt are expected to have exactly
     * the same data layout, and dst_dt padded regions are expected to be
     * zero, and the respective ws_dt elements should be set accordingly */
    for (int64_t i = 0; i < nelems; i += 8) {
        for (int64_t j = 0; j < MIN2(8, nelems - i); ++j) {
            const float data = dst_dt.get_f32_elem(i + j);
            const bool want = data > 0.f;
            const bool bit_set = ws_type == ws_byte ? *ws : !!(*ws & (1 << j));

            const bool ok = bit_set == want;
            res->errors += !ok;

            bool dump = false || (!ok && (res->errors < 10 || verbose >= 10))
                    || (verbose >= 50 && i < 30);
            if (dump) {
                BENCHDNN_PRINT(0, "[%4ld] ws exp:%d got:%d (data:%g:%a)\n",
                        (long)(i + j), want, bit_set, data, data);
            }

            if (ws_type == ws_byte) ++ws;
        }
        if (ws_type == ws_bit) ++ws;
    }

    if (res->errors) res->state = FAILED;
    if (res->state == EXECUTED) res->state = PASSED;

    return res->state == FAILED ? FAIL : OK;
}

dnnl_status_t init_pd(init_pd_args_t &init_pd_args) {
    const prb_t *prb = prb_t::from(init_pd_args.base_prb);
    const dir_t dir = init_pd_args.dir;
    res_t *res = init_pd_args.res;
    bool force_f32_dt = init_pd_args.force_f32_dt;

    benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> src_d;
    DNN_SAFE_STATUS(dnn_mem_t::init_md(src_d, prb->ndims,
            prb->data_dims().data(), force_f32_dt ? dnnl_f32 : prb->dt,
            prb->tag, prb->strides[0]));

    benchdnn_dnnl_wrapper_t<dnnl_primitive_attr_t> dnnl_attr;
    DNN_SAFE_STATUS(
            create_dnnl_attr(dnnl_attr, prb->attr, attr_args_t(), prb->ndims));

    auto flags = (dnnl_normalization_flags_t)prb->flags;
    if (dir & FLAG_FWD) {
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> dst_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(dst_d, prb->ndims,
                prb->data_dims().data(), force_f32_dt ? dnnl_f32 : prb->dt,
                tag::any, prb->strides[1]));
        auto prop = prb->dir & FLAG_INF ? dnnl_forward_inference
                                        : dnnl_forward_training;
        TIME_C_PD(DNN_SAFE_STATUS(
                dnnl_batch_normalization_forward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, prop,
                        init_pd_args.src_md ? init_pd_args.src_md : src_d,
                        dst_d, prb->eps, flags, dnnl_attr)));
    } else {
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_src_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_src_d, prb->ndims,
                prb->data_dims().data(), force_f32_dt ? dnnl_f32 : prb->dt,
                tag::any, prb->strides[0]));
        benchdnn_dnnl_wrapper_t<dnnl_memory_desc_t> diff_dst_d;
        DNN_SAFE_STATUS(dnn_mem_t::init_md(diff_dst_d, prb->ndims,
                prb->data_dims().data(), force_f32_dt ? dnnl_f32 : prb->dt,
                tag::any, prb->strides[1]));
        auto prop = prb->dir & FLAG_WEI ? dnnl_backward : dnnl_backward_data;
        TIME_C_PD(DNN_SAFE_STATUS(
                dnnl_batch_normalization_backward_primitive_desc_create(
                        &init_pd_args.pd, init_pd_args.engine, prop, diff_src_d,
                        diff_dst_d, src_d, prb->eps, flags, init_pd_args.hint,
                        dnnl_attr)));
    }

    return dnnl_success;
}

void prb_t::skip_unimplemented(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    skip_unimplemented_data_type({prb->dt}, prb->dir, res);
    skip_unimplemented_sum_po(
            prb->attr, res, dnnl_batch_normalization, prb->dt);
    skip_unimplemented_binary_po(prb->attr, res);
    skip_unimplemented_prelu_po(prb->attr, res, dnnl_batch_normalization);

    // Non-zero alpha is not supported for training in general.
    const auto &po = prb->attr.post_ops;
    const auto relu_idx = po.find(attr_t::post_ops_t::kind_t::RELU);
    if (relu_idx >= 0) {
        const auto &e = po.entry[relu_idx];
        float alpha = e.eltwise.alpha;
        bool alpha_ok = IMPLICATION(alpha != 0.f, (prb->dir & FLAG_INF));
        if (!alpha_ok) {
            res->state = SKIPPED;
            res->reason = reason_t::skip_not_supported;
        }
    }
    // BN+Add+ReLU fusion is not supported on CPU
    if (is_cpu() && prb->fuse_add_relu()) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
    }
    // int8 only supports forward s8 w/ global stats
    const bool u8_not_ok = prb->dt == dnnl_u8;
    const bool s8_not_ok = prb->dt == dnnl_s8
            && ((prb->dir & FLAG_BWD) || (prb->flags & GLOB_STATS) == 0);
    if (s8_not_ok || u8_not_ok) {
        res->state = SKIPPED;
        res->reason = reason_t::skip_not_supported;
    }
}

void prb_t::skip_invalid(res_t *res) const {
    const prb_t *prb = this; // Kept to avoid mass update
    // MIOpen cannot handle inplace cases correctly.
    if (is_amd_gpu() && prb->inplace) {
        res->state = SKIPPED;
        return;
    }

    // See `skip_invalid_inplace` for details.
    if (prb->inplace) {
        skip_invalid_inplace(res, prb->dt, prb->dt, prb->tag, prb->tag);
        if (res->state == SKIPPED) return;
    }
}

void setup_cmp(compare::compare_t &cmp, const base_prb_t *base_prb,
        data_kind_t kind, const args_t &ref_args) {
    const prb_t *prb = prb_t::from(base_prb);
    // Since bwd testing is done using results from forward which are random
    // fp32 values, diff_scale starts fluctuating, so we check norm for both
    // data, SC, and SH.
    const bool allow_norm_check = (prb->dir & FLAG_BWD);
    cmp.set_allow_norm_check(allow_norm_check);

    // Digits must be non-negative for safe left-shifting when `digits_dt`
    // exceeds `digits_f32`.
    const int safe_digits = MAX2(0, digits_dt(dnnl_f32) - digits_dt(prb->dt));
    const float trh_coeff = (1 << safe_digits);
    float trh = trh_coeff
            * ((kind == SRC || kind == DST || kind == SRC_1) ? 6e-7f : 0);
    if ((kind == SC || kind == SH) && prb->dir & FLAG_BWD)
        trh = trh_coeff * 5e-6f;

#ifdef DNNL_EXPERIMENTAL
    const bool bnorm_single_pass
            = dnnl::impl::experimental::use_bnorm_stats_one_pass();
#else
    const bool bnorm_single_pass = false;
#endif

    const bool use_relaxed_validation
            = is_nvidia_gpu() || is_amd_gpu() || bnorm_single_pass;
    if (use_relaxed_validation) {
        // Nvidia (cuDNN) and AMD (MIOpen): store unbiased variance which
        // requires rescaling by `(N - 1) / N`, where `N = MB * Spatial`.
        // Hence, we cannot set the threshold to 0...
        // Also mean could be computed using a single pass formula.
        //
        // On Intel GPUs mean and variance could be rounded incorrectly because
        // they are calculated using fast but potentially unstable formula.
        if (kind == MEAN) trh = 1e-7f;
        if (kind == VAR) trh = 5e-7f;
        if (kind == DST) trh = 2e-6f;
    }
    cmp.set_threshold(trh);

    // TODO: improve bf16 filling
    if (prb->dt == dnnl_bf16) cmp.set_zero_trust_percent(99.f);

    // When the error is larger than `trh`, it could be due to a catastrophic
    // cancellation in final result which is computed as `Y = a * X + b`.
    // When `a * X` is close to `b` and their signs are opposite, then large
    // error in `a * X` could result in a final result (which has a cancellation
    // i.e. `|Y| = |a*X - (-b)|`), which has no meaningful digits left in
    // mantissa.
    //
    // Since lambda is called when stack is unavailable, need to capture `prb`
    // and `kind` by value to avoid using dangling references.
    const auto bnorm_add_check =
            [&, kind, prb](
                    const compare::compare_t::driver_check_func_args_t &args) {
        bool ok = is_nvidia_gpu() && args.diff < args.trh;
        if (ok) return true;

        if (!((prb->dir & FLAG_FWD) && kind == DST && prb->use_sh()))
            return false;

        const auto &sh = ref_args.find(DNNL_ARG_SHIFT);
        const auto &dst = ref_args.find(DNNL_ARG_DST);
        const int64_t c = dst.get_idx(args.idx, 1 << 1 /* channel_mask */);
        const float beta = sh.get_f32_elem(c);
        // Using an empirically derived threshold, check if
        // cancellation error in `|Y| = |a*X - (-b)|` is huge.
        const float abs_exp = fabsf(args.exp);
        const float norm_denom = abs_exp > FLT_MIN ? abs_exp : 1.f;
        const float abs_exp_delta = fabsf(args.exp - beta);
        bool maybe_cancel_error = abs_exp_delta / norm_denom > 1.f;
        if (!maybe_cancel_error) return false;

        // Check for error in `a * X`
        float diff_aX = fabsf((args.exp - beta) - (args.got - beta));
        float rel_diff_aX
                = diff_aX / (abs_exp_delta > FLT_MIN ? abs_exp_delta : 1.f);
        return rel_diff_aX <= args.trh;
    };
    cmp.set_driver_check_function(bnorm_add_check);
}

std::vector<int> prb_t::supported_exec_args(bool override_dir_with_fwd) const {
    static const std::vector<int> exec_fwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_SRC_1,
            DNNL_ARG_MEAN,
            DNNL_ARG_VARIANCE,
            DNNL_ARG_SCALE,
            DNNL_ARG_SHIFT,
            DNNL_ARG_DST,
            DNNL_ARG_WORKSPACE,
    };
    static const std::vector<int> exec_bwd_args = {
            DNNL_ARG_SRC,
            DNNL_ARG_SRC_1,
            DNNL_ARG_MEAN,
            DNNL_ARG_VARIANCE,
            DNNL_ARG_SCALE,
            DNNL_ARG_WORKSPACE,
            DNNL_ARG_DIFF_DST,
            DNNL_ARG_DIFF_SCALE,
            DNNL_ARG_DIFF_SHIFT,
            DNNL_ARG_DIFF_SRC,
            DNNL_ARG_DIFF_SRC_1,
    };
    return (override_dir_with_fwd || (dir & FLAG_FWD)) ? exec_fwd_args
                                                       : exec_bwd_args;
}

int init_ref_memory_args(dnn_mem_map_t &ref_mem_map, dnn_mem_map_t &mem_map,
        dnnl_primitive_t prim, const base_prb_t *base_prb, res_t *res,
        dnnl_primitive_t prim_ref) {
    const auto *prb = prb_t::from(base_prb);
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return OK;

    if (!ref_mem_map.empty()) { erase_unused_args(ref_mem_map, mem_map); }

    // TODO: this function still allocates the full memory print needed to fill
    // the data and each argument can't be destroyed right away since filling
    // requires all of them at a time.
    const auto &ref_engine = get_cpu_engine();
    const bool is_fwd_prim = is_fwd_prop_kind(query_prop_kind(query_pd(prim)));

    for (auto &entry : mem_map) {
        const int exec_arg = entry.first;
        // The function targets regular exec_args that are positive.
        // Negative args are used by bitwise and are broken in the `default`
        // branch due to `&` always returns `true`.
        if (exec_arg <= 0) continue;

        auto &mem = entry.second;

        // Scratchpad memory relates to a primitive. If reference needs it,
        // use switch below to define a memory desc for it.
        if (exec_arg != DNNL_ARG_SCRATCHPAD && exec_arg != DNNL_ARG_WORKSPACE) {
            ref_mem_map.emplace(exec_arg,
                    dnn_mem_t(mem.md_, dnnl_f32, tag::abx, ref_engine,
                            /* prefill = */ false));
        }

        switch (exec_arg) {
            case DNNL_ARG_DST:
            case DNNL_ARG_DIFF_SRC: break; // Skip on backward.
            case DNNL_ARG_MEAN:
            case DNNL_ARG_VARIANCE:
                if (prb->dir & FLAG_INF) {
                    const dnnl_dims_t dims1d = {prb->ic};
                    ref_mem_map[exec_arg] = dnn_mem_t(1, dims1d, dnnl_f32,
                            tag::abx, ref_engine, /* prefill = */ false);
                }
                break;
            default: break;
        }
    }

    if (is_fwd_prim) {
        SAFE(prepare_fwd(prb, mem_map, ref_mem_map, res), WARN);
    } else {
        SAFE(prepare_bwd(prb, mem_map, ref_mem_map, res), WARN);
    }

    // Don't keep reference memory if it is not used further.
    if (!has_bench_mode_bit(mode_bit_t::corr)) {
        ref_mem_map.clear();
        return OK;
    }

    // Reference code uses different kind of workspace. Adjust to ref needs.
    // Insert a reference workspace if it comes non-empty from the library side.
    if (query_md_ndims(mem_map.at(DNNL_ARG_WORKSPACE).md_) > 0 && is_fwd_prim) {
        const auto &src_md = ref_mem_map[DNNL_ARG_SRC].md_;
        ref_mem_map[DNNL_ARG_WORKSPACE] = dnn_mem_t(
                src_md, dnnl_u8, tag::abx, ref_engine, /* prefill = */ false);
    }

    return OK;
}

std::vector<data_kind_t> get_kinds_to_check(const prb_t *prb, dir_t dir) {
    std::vector<data_kind_t> check_kinds;
    if ((prb->dir & FLAG_FWD) && (dir & FLAG_FWD)) {
        if (!(prb->flags & GLOB_STATS) && !(prb->dir & FLAG_INF)) {
            check_kinds.push_back(MEAN);
            check_kinds.push_back(VAR);
        }
        check_kinds.push_back(DST);
    } else if ((prb->dir & FLAG_BWD) && (dir & FLAG_BWD)) {
        if (prb->dir & FLAG_WEI) {
            if (prb->use_sc()) check_kinds.push_back(SC);
            if (prb->use_sh()) check_kinds.push_back(SH);
        }
        check_kinds.push_back(SRC);
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
    if (prb->debug_check_ws && has_bench_mode_bit(mode_bit_t::corr))
        check_fwd_ws(mem_map, res);

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

} // namespace bnorm
