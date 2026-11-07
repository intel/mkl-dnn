/*******************************************************************************
* Copyright 2026 Intel Corporation
* Copyright 2025 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
* Copyright 2026 Ampere Computing LLC
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
#include <cassert>
#include <cmath>
#include <limits>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "common/verbose.hpp"
#include "common/verbose_msg.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/platform.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/matmul/jit_int8_kernel_types.hpp"
#include "cpu/aarch64/matmul/jit_int8_matmul.hpp"
#include "cpu/aarch64/matmul/jit_int8_matmul_utils.hpp"

#define GET_OFF(field) (uint32_t) offsetof(call_params_t, field)

#define VCHECK_BG(f, msg, ...) \
    VCHECK(primitive, create, dispatch, brgemm_matmul, f, msg, ##__VA_ARGS__);

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace matmul {

using namespace Xbyak_aarch64;
using namespace dnnl::impl::cpu::matmul;
using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;

namespace {
struct jit_int8_matmul_tile_hint_t {
    int m_block_sz = 32;
    int n_block_factor = 0;
};

struct jit_int8_matmul_balance_t {
    float thread_eff = 0.f;
    float active_threads = 0.f;
    dim_t first_thread_jobs = 0;
    double first_thread_work = 0.;
};

jit_int8_matmul_balance_t get_thread_balance(
        dim_t m, dim_t n, int m_block, int n_block, int nthr) {
    jit_int8_matmul_balance_t balance;
    const dim_t m_tiles = div_up(m, m_block);
    const dim_t n_tiles = div_up(n, n_block);
    const dim_t total_tiles = m_tiles * n_tiles;

    balance.first_thread_jobs = div_up(total_tiles, nthr);
    dim_t remaining = balance.first_thread_jobs;

    const dim_t full_n_tiles = remaining / m_tiles;
    remaining = remaining % m_tiles;
    const dim_t full_n_work = nstl::min(n, full_n_tiles * n_block);
    balance.first_thread_work += static_cast<double>(m) * full_n_work;

    if (remaining > 0) {
        const dim_t cur_n
                = nstl::min(static_cast<dim_t>(n_block), n - full_n_work);
        const dim_t cur_m = nstl::min(m, remaining * m_block);
        balance.first_thread_work += static_cast<double>(cur_m) * cur_n;
    }

    const double total_work = static_cast<double>(m) * n;
    balance.thread_eff = static_cast<float>(
            total_work / (nthr * balance.first_thread_work));
    balance.active_threads = static_cast<float>(nstl::min(
                                     static_cast<dim_t>(nthr), total_tiles))
            / nthr;
    return balance;
}

constexpr double cache_fit_midpoint = 0.7;
constexpr double l2_stream_fraction = 0.75;
constexpr double cache_partition_a_l1_fraction = 0.5;
constexpr double cache_partition_b_l2_fraction = 0.25;
constexpr double cache_partition_min_b_l2_fraction = 0.5;

float soft_cache_fit(double bytes, double capacity) {
    // Use 70% as the cache-residency crossover to leave room for unrelated
    // data. Clamp the positive exponent once the fit is negligible so exp()
    // remains bounded.
    if (capacity <= 0.) return 0.f;
    constexpr double cache_fit_slope = 5.;
    constexpr double max_exp_argument = 20.;
    double x = cache_fit_slope * (bytes / capacity - cache_fit_midpoint);
    x = nstl::min(max_exp_argument, x);
    return static_cast<float>(1. / (1. + std::exp(x)));
}

float score_thread_tiles(const brg_int8_t &brg, int m_block, int n_block,
        int micro_n, int nthr, dim_t padded_k, double l1_cache_size,
        double l2_cache_size, bool model_outer_b_panel) {
    const auto balance
            = get_thread_balance(brg.M, brg.N, m_block, n_block, nthr);

    const int actual_m = nstl::min(brg.M, m_block);
    const int actual_n = nstl::min(brg.N, n_block);

    constexpr double job_cost_macs = 128. * 1024.;
    const double first_compute = balance.first_thread_work * padded_k;
    const double job_eff = first_compute
            / (first_compute + balance.first_thread_jobs * job_cost_macs);

    const double a_bytes = static_cast<double>(actual_m) * padded_k;
    const int cache_model_n
            = model_outer_b_panel ? actual_n : nstl::min(brg.N, micro_n);
    const double b_bytes = static_cast<double>(cache_model_n) * padded_k;
    const float a_l1_fit = soft_cache_fit(a_bytes, l1_cache_size);
    const float ab_l2_fit = soft_cache_fit(a_bytes + b_bytes, l2_cache_size);

    const double m_reuse = std::log2(
            nstl::max(1., static_cast<double>(actual_m) / brg.m_blk));
    const double n_reuse
            = std::log2(nstl::max(1., static_cast<double>(actual_n) / micro_n));

    // Prioritize thread balance and job amortization; use cache fit and reuse
    // as secondary signals. weights are empirical for SVE128
    constexpr float thread_eff_weight = 4.f;
    constexpr float job_eff_weight = 4.f;
    constexpr float active_threads_weight = 0.5f;
    constexpr float a_l1_fit_weight = 3.f;
    constexpr float ab_l2_fit_weight = 0.8f;
    constexpr float m_reuse_weight = 0.25f;
    constexpr float n_reuse_weight = 0.05f;
    return thread_eff_weight * balance.thread_eff
            + job_eff_weight * static_cast<float>(job_eff)
            + active_threads_weight * balance.active_threads
            + a_l1_fit_weight * a_l1_fit + ab_l2_fit_weight * ab_l2_fit
            + m_reuse_weight * static_cast<float>(m_reuse) * (1.f - a_l1_fit)
            + n_reuse_weight * static_cast<float>(n_reuse);
}

constexpr size_t max_m_tile_candidates = 8;
constexpr size_t max_n_tile_candidates = 12;
constexpr int max_tuned_m_block = 128;

template <size_t capacity>
void append_unique_candidate(
        std::array<int, capacity> &candidates, size_t &count, int value) {
    const auto end = candidates.begin() + count;
    if (std::find(candidates.begin(), end, value) != end) return;
    candidates[count++] = value;
}

jit_int8_matmul_tile_hint_t get_thread_tile_hint(const brg_int8_t &brg,
        int nthr, int micro_n, int max_search_m_block,
        bool allow_wide_n_candidates, bool use_long_reduction_model) {
    jit_int8_matmul_tile_hint_t hint;

    const dim_t max_int = std::numeric_limits<int>::max();
    constexpr int short_reduction_k = 256;
    constexpr int short_reduction_max_m_block = 32;
    // Empirical search bounds keep short reductions at or below 32 rows and
    // allow longer reductions to amortize blocks of up to 128 rows.
    const int reduction_m_block_limit = brg.K < short_reduction_k
            ? short_reduction_max_m_block
            : max_tuned_m_block;
    const int max_m_block
            = nstl::min(max_search_m_block, reduction_m_block_limit);

    // Sample power-of-two M blocks and 1.5x intermediate sizes, plus M rounded
    // to the microtile and capped by max_m_block.
    constexpr std::array<int, max_m_tile_candidates> m_block_values {
            {8, 16, 24, 32, 48, 64, 96, 128}};
    std::array<int, max_m_tile_candidates> m_candidates {};
    size_t m_count = 0;
    const int rounded_m = static_cast<int>(nstl::min(
            rnd_up(static_cast<dim_t>(brg.M), static_cast<dim_t>(brg.m_blk)),
            static_cast<dim_t>(max_m_block)));
    for (const int value : m_block_values) {
        if (value <= rounded_m)
            append_unique_candidate(m_candidates, m_count, value);
    }
    append_unique_candidate(m_candidates, m_count, rounded_m);

    const int full_n_factor = static_cast<int>(
            div_up(static_cast<dim_t>(brg.N), static_cast<dim_t>(micro_n)));
    const dim_t padded_k
            = rnd_up(static_cast<dim_t>(brg.K), static_cast<dim_t>(brg.k_blk));
    const double l1_cache_size = platform::get_per_core_cache_size(1);
    const double l2_cache_size = platform::get_per_core_cache_size(2);
    const double b_stream_limit = l2_stream_fraction * l2_cache_size;
    const double wide_n_panel_limit = cache_fit_midpoint * l2_cache_size;

    float best_score = nstl::numeric_limits<float>::lowest();
    for (size_t m_idx = 0; m_idx < m_count; m_idx++) {
        const int m_block = m_candidates[m_idx];
        const int actual_m = nstl::min(brg.M, m_block);
        std::array<int, max_n_tile_candidates> n_factors {};
        size_t n_count = 0;

        constexpr int max_base_n_factor = 4;
        constexpr std::array<int, 7> preset_n_factors {{1, 2, 4, 6, 8, 9, 16}};
        for (const int factor : preset_n_factors) {
            if (factor > max_base_n_factor) {
                const bool long_reduction_candidate = use_long_reduction_model
                        && utils::one_of(factor, 6, 8, 9);
                const bool wide_n_candidate = allow_wide_n_candidates
                        && utils::one_of(factor, 8, 16);
                if (!long_reduction_candidate && !wide_n_candidate) continue;
                const int actual_n = nstl::min(brg.N, factor * micro_n);
                const double outer_panel_bytes
                        = static_cast<double>(actual_m) * padded_k
                        + static_cast<double>(actual_n) * padded_k;
                if (outer_panel_bytes > wide_n_panel_limit) continue;
            }
            append_unique_candidate(
                    n_factors, n_count, nstl::min(factor, full_n_factor));
        }
        append_unique_candidate(n_factors, n_count, full_n_factor);

        const dim_t m_tiles = div_up(
                static_cast<dim_t>(brg.M), static_cast<dim_t>(m_block));
        const dim_t m_jobs = m_tiles;
        // Include candidates targeting roughly one through four jobs/thread.
        constexpr int max_jobs_per_thread = 4;
        for (int job_multiplier = 1; job_multiplier <= max_jobs_per_thread;
                job_multiplier++) {
            const dim_t target_jobs = static_cast<dim_t>(nthr) * job_multiplier;
            const dim_t target_n_jobs = nstl::max(
                    static_cast<dim_t>(1), div_up(target_jobs, m_jobs));
            const dim_t target_n_block
                    = rnd_up(div_up(static_cast<dim_t>(brg.N), target_n_jobs),
                            static_cast<dim_t>(micro_n));
            const int factor = static_cast<int>(
                    nstl::min(div_up(target_n_block, micro_n),
                            static_cast<dim_t>(full_n_factor)));
            append_unique_candidate(n_factors, n_count, factor);
        }

        for (size_t n_idx = 0; n_idx < n_count; n_idx++) {
            const int factor = n_factors[n_idx];
            const dim_t n_block_dim = static_cast<dim_t>(factor) * micro_n;
            if (n_block_dim > max_int) continue;
            const dim_t n_tiles
                    = div_up(static_cast<dim_t>(brg.N), n_block_dim);
            if (m_jobs > max_int / n_tiles) continue;
            const int n_block = static_cast<int>(n_block_dim);
            const int actual_n = nstl::min(brg.N, n_block);
            const double b_block_bytes
                    = static_cast<double>(actual_n) * padded_k;
            const double b_bytes_per_m_micro
                    = b_block_bytes / div_up(actual_m, brg.m_blk);
            if (factor > 1 && b_bytes_per_m_micro > b_stream_limit) continue;

            const float score = score_thread_tiles(brg, m_block, n_block,
                    micro_n, nthr, padded_k, l1_cache_size, l2_cache_size,
                    use_long_reduction_model);
            if (score > best_score) {
                best_score = score;
                hint.m_block_sz = m_block;
                hint.n_block_factor = factor;
            }
        }
    }

    if (hint.n_block_factor == 0) return hint;

    // Short reductions do not amortize an imbalanced wave of outer jobs.
    // Preserve the established blocking if tuning leaves fewer than two
    // jobs per thread.
    constexpr int min_short_reduction_jobs_per_thread = 2;
    if (brg.K < short_reduction_k) {
        const dim_t tuned_jobs = div_up(static_cast<dim_t>(brg.M),
                                         static_cast<dim_t>(hint.m_block_sz))
                * div_up(static_cast<dim_t>(brg.N),
                        static_cast<dim_t>(hint.n_block_factor) * micro_n);
        if (tuned_jobs < static_cast<dim_t>(nthr)
                        * min_short_reduction_jobs_per_thread)
            return {};
    }
    return hint;
}

jit_int8_matmul_tile_hint_t get_partitioned_n_tile_hint(
        const brg_int8_t &brg, int micro_n, int m_block, int target_n_tiles) {
    const dim_t n_block
            = rnd_up(div_up(static_cast<dim_t>(brg.N), target_n_tiles),
                    static_cast<dim_t>(micro_n));
    jit_int8_matmul_tile_hint_t hint;
    hint.m_block_sz = m_block;
    hint.n_block_factor = static_cast<int>(n_block / micro_n);
    return hint;
}

jit_int8_matmul_tile_hint_t get_cache_partitioned_tile_hint(
        const brg_int8_t &brg, int micro_n, double l1_cache_size,
        double l2_cache_size) {
    if (l1_cache_size <= 0. || l2_cache_size <= 0.) return {};

    const dim_t padded_k
            = rnd_up(static_cast<dim_t>(brg.K), static_cast<dim_t>(brg.k_blk));
    const double a_tile_budget = cache_partition_a_l1_fraction * l1_cache_size;
    const dim_t cache_m_rows = static_cast<dim_t>(a_tile_budget / padded_k);
    const dim_t aligned_cache_m_rows
            = rnd_dn(cache_m_rows, static_cast<dim_t>(brg.m_blk));
    if (aligned_cache_m_rows < brg.m_blk) return {};

    const dim_t rounded_m
            = rnd_up(static_cast<dim_t>(brg.M), static_cast<dim_t>(brg.m_blk));
    const dim_t m_block_dim = nstl::min(rounded_m,
            nstl::min(aligned_cache_m_rows,
                    static_cast<dim_t>(max_tuned_m_block)));
    const int m_block = static_cast<int>(m_block_dim);

    const double b_panel_budget = cache_partition_b_l2_fraction * l2_cache_size;
    const double total_b_bytes = static_cast<double>(brg.N) * padded_k;
    const dim_t max_n_tiles
            = div_up(static_cast<dim_t>(brg.N), static_cast<dim_t>(micro_n));
    dim_t target_n_tiles
            = static_cast<dim_t>(std::ceil(total_b_bytes / b_panel_budget));
    target_n_tiles = nstl::max(
            static_cast<dim_t>(1), nstl::min(target_n_tiles, max_n_tiles));
    if (brg.N >= brg.K && target_n_tiles < max_n_tiles) ++target_n_tiles;

    return get_partitioned_n_tile_hint(
            brg, micro_n, m_block, static_cast<int>(target_n_tiles));
}

struct jit_int8_matmul_thread_tiles_t {
    int m_block_sz;
    int n_block_sz;
};

jit_int8_matmul_thread_tiles_t get_thread_tiles(const brg_int8_t &brg,
        int num_threads, size_t simd_bytes, int micro_n,
        const jit_int8_matmul_tile_hint_t &hint) {
    jit_int8_matmul_thread_tiles_t tiles;
    tiles.m_block_sz = hint.m_block_sz;
    if (hint.n_block_factor > 0) {
        tiles.n_block_sz = hint.n_block_factor * micro_n;
        return tiles;
    }

    tiles.n_block_sz = (brg.ld_block == 6) ? micro_n : 2 * micro_n;

    if (simd_bytes == 16 && brg.ld_block == 6) {
        int best_n_block_sz = micro_n;
        const int num_m_tiles = div_up(brg.M, tiles.m_block_sz);
        const bool mid_m
                = (brg.M > tiles.m_block_sz) && (brg.M <= 4 * tiles.m_block_sz);
        const bool small_m = num_m_tiles <= 4;

        const int preferred_factor = (mid_m || small_m) ? 2 : 4;
        for (int f = preferred_factor; f >= 1; f /= 2) {
            const int temp_n_block_sz = f * micro_n;
            const int num_n_tiles = div_up(brg.N, temp_n_block_sz);
            const dim_t work
                    = brg.batch * num_m_tiles * static_cast<dim_t>(num_n_tiles);
            if (work >= num_threads) {
                best_n_block_sz = temp_n_block_sz;
                break;
            }
        }
        tiles.n_block_sz = nstl::min(brg.N, best_n_block_sz);
        if (tiles.n_block_sz % micro_n != 0)
            tiles.n_block_sz = div_up(tiles.n_block_sz, micro_n) * micro_n;
    }

    return tiles;
}
} // namespace

template <cpu_isa_t isa>
struct jit_int8_matmul_kernel_t : public jit_generator_t {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_int8_matmul_kernel_t)

    XReg reg_param = abi_param1;
    XReg reg_a = x3;
    XReg reg_b = x4;
    XReg reg_c = x5;
    XReg reg_aux_a = x6;
    XReg reg_aux_b = x7;
    XReg reg_aux_c = x8;
    XReg reg_aux_a1 = x9;
    XReg reg_zp_aux_b_buf = x10;
    XReg reg_aux_c1 = x11;
    XReg reg_ld_loop = x12;
    XReg reg_rd_loop = x13;
    XReg reg_bd_loop = x14;
    XReg reg_tmp = x15;
    XReg reg_tmp_1 = x16;
    XReg reg_bias = x17;
    XReg reg_zp_a = x18;

    XReg reg_src_scales = x20;
    XReg reg_aux_src_scales = x1;
    XReg reg_wei_scales = x19;

    XReg reg_aux_scales = x24; //used X_TMP_1
    XReg reg_na = x25; //used X_TMP_2
    XReg reg_zp_b = x26; //used X_TMP_3
    XReg reg_zp_aux_b = x27; //used X_TMP_4
    PReg prd_ld = p1;
    PReg prd_st = p2;
    PReg prd_b = p3;
    PReg prd_8 = p4;
    PReg prd_zp_b_tl = p5;
    XReg reg_zp_val_c = x2;

    XReg reg_zp_val_a = reg_src_scales;

    XReg reg_zp_val_b = reg_bias;

    call_params_t inp;

    void cvt_f32_to_xf16(const ZReg &zdata, const PReg &pg, data_type_t xf16) {
        assert(xf16 == f16 || xf16 == bf16);
        if (xf16 == bf16) {
            bfcvt(zdata.h, pg / T_m, zdata.s);
        } else {
            fcvt(zdata.h, pg / T_m, zdata.s);
        }
    }

    void operator()(const call_params_t *p) {
        return jit_generator_t::operator()(p);
    }

    ZReg loadb(int ld) { return ZReg(ld + 1); }
    ZReg acc(int bd, int ld) const {
        return ZReg(bd * brg_.ld_block + ld + brg_.ld_block + 1);
    }
    void zero_regs() {
        for (int a = 0; a < brg_.bd_block / 2; a++)
            for (int b = 0; b < brg_.ld_block; b++)
                uni_clear(acc(a, b));
    }
    void store_regs(int bdb, int ldb, int tail) {
        // Plain s32 dst stores the accumulators directly
        //
        // Convert to f32 when zero-point compensation, scales, bias addition,
        // or eltwise post-op and then convert back to s32 when dst is s32
        const bool can_store_s32_acc_directly = brg_.dst_dt == s32
                && everyone_is(jit_int8_broadcast_t::none, brg_.zp_type_a,
                        brg_.zp_type_b, brg_.zp_type_c)
                && !brg_.with_scales && !brg_.is_bias
                && eltwise_injectors_.empty() && !brg_.with_dst_scales;

        if (!can_store_s32_acc_directly) {
            for (int a = 0; a < bdb; a++) {
                for (int b = 0; b < ldb; b++) {
                    if (brg_.is_s8 || brg_.is_u8_s8)
                        scvtf(acc(a, b).s, P_ALL_ONE, acc(a, b).s);
                    else
                        ucvtf(acc(a, b).s, P_ALL_ONE, acc(a, b).s);
                }
            }
        }

        for (int a = 0; a < bdb; a++) {
            for (int b = 0; b < ldb; b += 2) {
                if (b + 1 < ldb) {
                    uzp1(z31.d, acc(a, b).d, acc(a, b + 1).d);
                    uzp2(acc(a, b + 1).d, acc(a, b).d, acc(a, b + 1).d);
                    mov(acc(a, b).d, z31.d);
                } else {
                    uzp1(z31.d, acc(a, b).d, acc(a, b).d);
                    uzp2(acc(a, b + 1).d, acc(a, b).d, acc(a, b).d);
                    mov(acc(a, b).d, z31.d);
                }
            }
        }

        if (brg_.zp_type_a != jit_int8_broadcast_t::none) {
            for (int b = 0; b < ldb; b += 2) {
                PReg p = (brg_.is_n_tail && b >= ldb - 2) ? prd_b : P_ALL_ONE;
                ld1w(z31.s, p, ptr(reg_zp_a, b / 2, MUL_VL));
                for (int a = 0; a < bdb; a++) {
                    fsub(acc(a, b).s, acc(a, b).s, z31.s);
                    fsub(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
                }
            }
        }

        if (brg_.zp_type_b != jit_int8_broadcast_t::none) {
            int ao = 0;
            const ZReg zp_b_val = ZReg(1);
            if (brg_.is_zp_b_int8) {
                // Common int8/u8 weights zero-point is provided as a single
                // scalar runtime value.
                if (brg_.zp_b_dt == data_type::u8) {
                    ld1rb(zp_b_val.s, P_ALL_ONE, ptr(reg_zp_aux_b_buf));
                    ucvtf(zp_b_val.s, P_ALL_ONE, zp_b_val.s);
                } else {
                    ld1rsb(zp_b_val.s, P_ALL_ONE, ptr(reg_zp_aux_b_buf));
                    scvtf(zp_b_val.s, P_ALL_ONE, zp_b_val.s);
                }
            }
            for (int a = 0; a < bdb; a++) {
                // Fetch row-wise src reduction
                ld1rw(z31.s, P_ALL_ONE, ptr(reg_zp_aux_b, ao * 4));
                ld1rw(z0.s, P_ALL_ONE, ptr(reg_zp_aux_b, (ao + 1) * 4));

                if (brg_.is_zp_b_int8) {
                    fmul(z31.s, z31.s, zp_b_val.s);
                    fmul(z0.s, z0.s, zp_b_val.s);
                }

                for (int b = 0; b < ldb; b += 2) {
                    fsub(acc(a, b).s, acc(a, b).s, z31.s);
                    fsub(acc(a, b + 1).s, acc(a, b + 1).s, z0.s);
                }
                ao += 2;
            }
        }

        if (brg_.with_scales) {
            for (int b = 0; b < ldb; b += 2) {
                PReg p = (brg_.is_n_tail && b >= ldb - 2) ? prd_b : P_ALL_ONE;
                if (brg_.is_oc_scales) {
                    ld1w(z31.s, p, ptr(reg_wei_scales, b / 2, MUL_VL));
                } else {
                    ld1w(z31.s, p, ptr(reg_wei_scales));
                }

                if (!brg_.is_per_m_scales) {
                    if (brg_.with_src_scales) {
                        ld1rw(z0.s, P_ALL_ONE, ptr(reg_src_scales));
                        fmul(z31.s, z31.s, z0.s);
                    }

                    for (int a = 0; a < bdb; a++) {
                        fmul(acc(a, b).s, acc(a, b).s, z31.s);
                        fmul(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
                    }
                } else {
                    for (int a = 0; a < bdb; a++) {
                        auto apply_row_scale
                                = [&](int row_off, const ZReg &scale,
                                          const ZReg &acc) {
                            add_imm(reg_tmp, reg_aux_src_scales, row_off,
                                    X_TMP_0);
                            ld1rw(scale.s, P_ALL_ONE, ptr(reg_tmp));
                            fmul(scale.s, scale.s, z31.s);
                            fmul(acc.s, acc.s, scale.s);
                        };

                        const bool has_second_row
                                = !brg_.is_m_tail || (2 * a + 1) < brg_.m_tail;
                        const int row_off = (2 * a) * sizeof(float);

                        apply_row_scale(row_off, z0, acc(a, b));
                        if (has_second_row)
                            apply_row_scale(
                                    row_off + sizeof(float), z2, acc(a, b + 1));
                    }
                }
            }
        }

        if (brg_.is_bias) {
            for (int b = 0; b < ldb; b += 2) {
                PReg p = (brg_.is_n_tail && b >= ldb - 2) ? prd_b : P_ALL_ONE;
                ld1w(z31.s, p, ptr(reg_bias, b / 2, MUL_VL));
                for (int a = 0; a < bdb; a++) {
                    fadd(acc(a, b).s, acc(a, b).s, z31.s);
                    fadd(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
                }
            }
        }

        if (!eltwise_injectors_.empty()) {
            injector_utils::vmm_index_set_t accumulator_idxs;
            for (int a = 0; a < bdb; a++) {
                for (int b = 0; b < ldb; b += 2) {
                    accumulator_idxs.insert(acc(a, b).getIdx());
                    accumulator_idxs.insert(acc(a, b + 1).getIdx());
                }
            }
            for (const auto &injector : eltwise_injectors_) {
                injector->load_table_addr();
                injector->compute_vector_range(accumulator_idxs);
            }
        }

        if (brg_.with_dst_scales) {
            ld1rw(z31.s, P_ALL_ONE, ptr(reg_aux_scales));
            for (int b = 0; b < ldb; b += 2) {
                for (int a = 0; a < bdb; a++) {
                    fmul(acc(a, b).s, acc(a, b).s, z31.s);
                    fmul(acc(a, b + 1).s, acc(a, b + 1).s, z31.s);
                }
            }
        }

        if (brg_.zp_type_c != jit_int8_broadcast_t::none) {
            ldr_imm(reg_zp_val_c, reg_param, GET_OFF(dst_zero_point));
            ldr(W_TMP_0, ptr(reg_zp_val_c));
            dup(z0.s, W_TMP_0);
            scvtf(z0.s, P_ALL_ONE, z0.s);
            for (int b = 0; b < ldb; b += 2) {
                for (int a = 0; a < bdb; a++) {
                    fadd(acc(a, b).s, acc(a, b).s, z0.s);
                    fadd(acc(a, b + 1).s, acc(a, b + 1).s, z0.s);
                }
            }
        }

        mov(reg_tmp, reg_aux_c);
        add_imm(reg_tmp_1, reg_aux_c, brg_.N * brg_.dst_dt_sz, X_TMP_0);
        for (int a = 0; a < bdb; a++) {
            const bool store_second_row
                    = !brg_.is_m_tail || (2 * a + 1) < brg_.m_tail;
            switch (brg_.dst_dt) {
                case f32:
                    for (int b = 0; b < ldb; b += 2) {
                        PReg p = (brg_.is_n_tail && b >= ldb - 2) ? prd_st
                                                                  : P_ALL_ONE;
                        const int vl = b / 2;
                        st1w(acc(a, b).s, p, ptr(reg_tmp, vl, MUL_VL));
                        if (store_second_row)
                            st1w(acc(a, b + 1).s, p,
                                    ptr(reg_tmp_1, vl, MUL_VL));
                    }
                    break;
                case data_type::s8:
                case u8:
                case s32: {
                    const bool need_saturation = !can_store_s32_acc_directly
                            || one_of(brg_.dst_dt, data_type::s8, u8);

                    constexpr bool force_lbound = true;
                    if (need_saturation) {
                        init_saturate_f32(z31, z0, reg_aux_a, f32, brg_.dst_dt,
                                force_lbound);
                    }

                    auto maybe_saturate_and_store
                            = [&](const ZReg &zreg, const XReg &dst,
                                      const PReg &p, int vl, data_type_t dt) {
                        if (need_saturation) {
                            saturate_f32(zreg, z31, z0, dt, p, force_lbound);
                            frinti(zreg.s, p, zreg.s);

                            // The choice of rounding instruction is a function
                            // of signedness, so s8, and s32 are handled
                            // together, with u8 on its own.
                            if (dt == u8) {
                                fcvtzu(zreg.s, p, zreg.s);
                            } else {
                                fcvtzs(zreg.s, p, zreg.s);
                            }
                        }

                        // The choice of store instruction is a function of data
                        // type size. So u8, s8 are handled together, with s32
                        // on its own.
                        switch (brg_.dst_dt_sz) {
                            case 4:
                                st1w(zreg.s, p, ptr(dst, vl, MUL_VL));
                                break;
                            case 1:
                                st1b(zreg.s, p, ptr(dst, vl, MUL_VL));
                                break;
                            default: assert(!"unreachable dt");
                        }
                    };

                    for (int b = 0; b < ldb; b += 2) {
                        PReg p = (brg_.is_n_tail && b >= ldb - 2) ? prd_st
                                                                  : P_ALL_ONE;
                        const int vl = b / 2;
                        maybe_saturate_and_store(
                                acc(a, b), reg_tmp, p, vl, brg_.dst_dt);
                        if (store_second_row)
                            maybe_saturate_and_store(acc(a, b + 1), reg_tmp_1,
                                    p, vl, brg_.dst_dt);
                    }

                    break;
                }
                case f16:
                case bf16:
                    for (int b = 0; b < ldb; b += 2) {
                        PReg p = (brg_.is_n_tail && b >= ldb - 2) ? prd_st
                                                                  : P_ALL_ONE;
                        cvt_f32_to_xf16(acc(a, b), p, brg_.dst_dt);
                        if (store_second_row)
                            cvt_f32_to_xf16(acc(a, b + 1), p, brg_.dst_dt);
                    }

                    for (int b = 0; b < ldb; b += 2) {
                        PReg p = (brg_.is_n_tail && b >= ldb - 2) ? prd_st
                                                                  : P_ALL_ONE;
                        const int vl = b / 2;
                        st1h(acc(a, b).s, p, ptr(reg_tmp, vl, MUL_VL));
                        if (store_second_row)
                            st1h(acc(a, b + 1).s, p,
                                    ptr(reg_tmp_1, vl, MUL_VL));
                    }
                    break;
                default: assert(!"unsupported dst_dt");
            }
            add_imm(reg_tmp, reg_tmp, 2 * brg_.N * brg_.dst_dt_sz, X_TMP_0);
            add_imm(reg_tmp_1, reg_tmp_1, 2 * brg_.N * brg_.dst_dt_sz, X_TMP_0);
        }
    }

    void microkernel(int rdb, int bdb, int ldb, int tail) {
        int a_off = 0, rd, ld, bd;
        mov(reg_tmp, reg_aux_b);
        for (rd = 0; rd < rdb; rd++) {
            int ao = 0;

            // For SVE128 and no N-tail, use NEON loads (v/z regs overlap) to
            // reduce predicated SVE load overhead in the hot loop.
            if (simd_bytes(isa) == 16 && !brg_.is_n_tail) {
                constexpr int bytes_per_vec = 16;
                for (ld = 0; ld + 1 < ldb; ld += 2) {
                    QReg q0(loadb(ld).getIdx());
                    QReg q1(loadb(ld + 1).getIdx());
                    ldp(q0, q1, ptr(reg_tmp, ld * bytes_per_vec));
                }
                if (ld < ldb) {
                    QReg q(loadb(ld).getIdx());
                    ldr(q, ptr(reg_tmp, ld * bytes_per_vec));
                }
            } else {
                for (ld = 0; ld < ldb; ld++) {
                    PReg p = (brg_.is_n_tail && ld == ldb - 1) ? prd_ld
                                                               : P_ALL_ONE;
                    ld1b(loadb(ld).b, p, ptr(reg_tmp, ld, MUL_VL));
                }
            }
            for (bd = 0; bd < bdb;) {
                const int a_inc = brg_.m_blk * 2;
                const int32_t a_off_bytes = a_off + ao;
                if (simd_bytes(isa) == 16 && bd + 1 < bdb) {
                    // Load two consecutive A blocks (each 16 bytes for SVE128)
                    // and compute two rows to amortize load overhead.
                    if ((a_off_bytes & 0xf) == 0 && a_off_bytes >= -1024
                            && a_off_bytes <= 1008) {
                        ldp(QReg(0), QReg(31), ptr(reg_aux_a, a_off_bytes));
                    } else {
                        add_imm(X_DEFAULT_ADDR, reg_aux_a, a_off_bytes,
                                X_TMP_0);
                        ldp(QReg(0), QReg(31), ptr(X_DEFAULT_ADDR));
                    }
                    ao += 2 * a_inc;

                    for (ld = 0; ld < ldb; ld++) {
                        if (brg_.is_s8) {
                            smmla(acc(bd, ld).s, z0.b, loadb(ld).b);
                            smmla(acc(bd + 1, ld).s, z31.b, loadb(ld).b);
                        } else if (brg_.is_u8_s8) {
                            usmmla(acc(bd, ld).s, z0.b, loadb(ld).b);
                            usmmla(acc(bd + 1, ld).s, z31.b, loadb(ld).b);
                        } else {
                            ummla(acc(bd, ld).s, z0.b, loadb(ld).b);
                            ummla(acc(bd + 1, ld).s, z31.b, loadb(ld).b);
                        }
                    }
                    bd += 2;
                } else {
                    if (simd_bytes(isa) == 16) {
                        ldr(QReg(0), ptr(reg_aux_a, a_off_bytes));
                    } else {
                        add_imm(X_DEFAULT_ADDR, reg_aux_a, a_off_bytes,
                                X_TMP_0);
                        ld1rqb(z0.b, P_ALL_ONE, ptr(X_DEFAULT_ADDR));
                    }
                    ao += a_inc;

                    for (ld = 0; ld < ldb; ld++) {
                        if (brg_.is_s8)
                            smmla(acc(bd, ld).s, z0.b, loadb(ld).b);
                        else if (brg_.is_u8_s8)
                            usmmla(acc(bd, ld).s, z0.b, loadb(ld).b);
                        else
                            ummla(acc(bd, ld).s, z0.b, loadb(ld).b);
                    }
                    bd += 1;
                }
            }
            a_off += brg_.m_blk * brg_.k_blk;
            add_imm(reg_tmp, reg_tmp, brg_.k_blk * brg_.n_blk * brg_.ld_block,
                    X_TMP_0);
        }
    }

    void loop_k(int bdb, int ldb, int tail) {
        zero_regs();
        mov(reg_aux_a, reg_aux_a1);
        mov(reg_aux_b, reg_b);
        if (k_full_blks > 0) {
            mov(reg_rd_loop, k_full_blks);

            asm_do_while(reg_rd_loop, [&]() {
                microkernel(brg_.rd_block, bdb, ldb, tail);
                add_imm(reg_aux_a, reg_aux_a,
                        brg_.m_blk * brg_.k_blk * brg_.rd_block, X_TMP_0);
                add_imm(reg_aux_b, reg_aux_b,
                        brg_.k_blk * brg_.n_blk * brg_.ld_block * brg_.rd_block,
                        X_TMP_0);
            });
        }
        if (k_tail_blk > 0) {
            microkernel(k_tail_blk, bdb, ldb, tail);
            add_imm(reg_aux_a, reg_aux_a, brg_.m_blk * brg_.k_blk * k_tail_blk,
                    X_TMP_0);
            add_imm(reg_aux_b, reg_aux_b,
                    brg_.k_blk * brg_.n_blk * brg_.ld_block * k_tail_blk,
                    X_TMP_0);
        }
        if (k_residual_blk > 0) { microkernel(1, bdb, ldb, tail); }
        store_regs(bdb, ldb, tail);
    }

    void loop_k_zp(int bdb, int ldb, int is_a, int is_b) {
        const ZReg zp_b_sum0 = ZReg(3);
        const ZReg zp_b_sum1 = ZReg(4);
        const ZReg zp_b_sum2 = ZReg(5);
        const ZReg zp_b_sum3 = ZReg(6);

        eor(zp_b_sum0.d, zp_b_sum0.d, zp_b_sum0.d);
        eor(zp_b_sum1.d, zp_b_sum1.d, zp_b_sum1.d);
        if (simd_bytes(isa) == 16) {
            eor(zp_b_sum2.d, zp_b_sum2.d, zp_b_sum2.d);
            eor(zp_b_sum3.d, zp_b_sum3.d, zp_b_sum3.d);
        }
        for (int i = 0; i < 6; i++)
            eor(acc(2, i).d, acc(2, i).d, acc(2, i).d);
        mov(reg_aux_a, reg_aux_a1);
        mov(reg_aux_b, reg_b);
        if (k_full_blks > 0) {
            mov(reg_rd_loop, k_full_blks);

            asm_do_while(reg_rd_loop, [&]() {
                zp_comp(brg_.rd_block, bdb, ldb, is_a, is_b);
                add_imm(reg_aux_a, reg_aux_a,
                        brg_.m_blk * brg_.k_blk * brg_.rd_block, X_TMP_0);
                add_imm(reg_aux_b, reg_aux_b,
                        brg_.k_blk * brg_.n_blk * brg_.ld_block * brg_.rd_block,
                        X_TMP_0);
            });
        }
        if (k_tail_blk > 0) {
            zp_comp(k_tail_blk, bdb, ldb, is_a, is_b);
            add_imm(reg_aux_a, reg_aux_a, brg_.m_blk * brg_.k_blk * k_tail_blk,
                    X_TMP_0);
            add_imm(reg_aux_b, reg_aux_b,
                    brg_.k_blk * brg_.n_blk * brg_.ld_block * k_tail_blk,
                    X_TMP_0);
        }
        if (k_residual_blk > 0) { zp_comp(1, bdb, ldb, is_a, is_b); }

        if (brg_.zp_type_b != jit_int8_broadcast_t::none && is_b == 1) {
            if (simd_bytes(isa) == 16) {
                uzp1(zp_b_sum0.d, zp_b_sum0.d, zp_b_sum1.d);
                uzp1(zp_b_sum2.d, zp_b_sum2.d, zp_b_sum3.d);
                scvtf(zp_b_sum0.s, P_ALL_ONE, zp_b_sum0.s);
                scvtf(zp_b_sum2.s, P_ALL_ONE, zp_b_sum2.s);
            } else {
                uzp1(zp_b_sum0.d, zp_b_sum0.d, zp_b_sum1.d);
                scvtf(zp_b_sum0.s, P_ALL_ONE, zp_b_sum0.s);
            }
            if (!brg_.is_zp_b_int8) {
                ldr(W_TMP_0, ptr(reg_zp_val_b));
                dup(z0.s, W_TMP_0);
                scvtf(z0.s, P_ALL_ONE, z0.s);
                fmul(zp_b_sum0.s, P_ALL_ONE, z0.s);
                if (simd_bytes(isa) == 16) {
                    fmul(zp_b_sum2.s, P_ALL_ONE, z0.s);
                }
            } else {
                if (brg_.zp_type_a != jit_int8_broadcast_t::none) {
                    ldr(W_TMP_0, ptr(reg_zp_val_a));
                    dup(z0.s, W_TMP_0);
                    mov_imm(W_TMP_0, brg_.K);
                    dup(z1.s, W_TMP_0);
                    scvtf(z0.s, P_ALL_ONE, z0.s);
                    scvtf(z1.s, P_ALL_ONE, z1.s);
                    fmul(z0.s, z1.s, z0.s);
                    fsub(zp_b_sum0.s, zp_b_sum0.s, z0.s);
                    if (simd_bytes(isa) == 16) {
                        fsub(zp_b_sum2.s, zp_b_sum2.s, z0.s);
                    }
                }
            }
            st1w(zp_b_sum0.s, P_ALL_ONE, ptr(reg_zp_b));
            if (simd_bytes(isa) == 16) {
                st1w(zp_b_sum2.s, P_ALL_ONE, ptr(reg_zp_b, 1, MUL_VL));
            }
        }

        if ((brg_.zp_type_a != jit_int8_broadcast_t::none) && is_a == 1) {
            ldr(W_TMP_0, ptr(reg_zp_val_a));
            dup(z2.s, W_TMP_0);
            scvtf(z2.s, P_ALL_ONE, z2.s);
            uzp1(acc(2, 0).d, acc(2, 0).d, acc(2, 1).d);
            if (brg_.ld_block > 2) uzp1(acc(2, 2).d, acc(2, 2).d, acc(2, 3).d);
            if (brg_.ld_block > 4) uzp1(acc(2, 4).d, acc(2, 4).d, acc(2, 5).d);

            scvtf(acc(2, 0).s, P_ALL_ONE, acc(2, 0).s);
            if (brg_.ld_block > 2) scvtf(acc(2, 2).s, P_ALL_ONE, acc(2, 2).s);
            if (brg_.ld_block > 4) scvtf(acc(2, 4).s, P_ALL_ONE, acc(2, 4).s);
            if (brg_.zp_type_b != jit_int8_broadcast_t::none
                    && !brg_.is_zp_b_int8) {
                ldr(W_TMP_0, ptr(reg_zp_val_b));
                dup(z0.s, W_TMP_0);
                mov_imm(W_TMP_0, brg_.K);
                dup(z1.s, W_TMP_0);
                scvtf(z0.s, P_ALL_ONE, z0.s);
                scvtf(z1.s, P_ALL_ONE, z1.s);
                fmul(z0.s, z1.s, z0.s);
                fsub(acc(2, 0).s, acc(2, 0).s, z0.s);
                if (brg_.ld_block > 2) fsub(acc(2, 2).s, acc(2, 2).s, z0.s);
                if (brg_.ld_block > 4) fsub(acc(2, 4).s, acc(2, 4).s, z0.s);
            }
            fmul(acc(2, 0).s, P_ALL_ONE, z2.s);
            if (brg_.ld_block > 2) fmul(acc(2, 2).s, P_ALL_ONE, z2.s);
            if (brg_.ld_block > 4) fmul(acc(2, 4).s, P_ALL_ONE, z2.s);

            st1w(acc(2, 0).s, P_ALL_ONE, ptr(reg_zp_a));
            if (brg_.ld_block > 2)
                st1w(acc(2, 2).s, P_ALL_ONE, ptr(reg_zp_a, 1, MUL_VL));
            if (brg_.ld_block > 4)
                st1w(acc(2, 4).s, P_ALL_ONE, ptr(reg_zp_a, 2, MUL_VL));
        }
    }

    void han_blk() {
        ldr_imm(reg_tmp, reg_param, GET_OFF(nb));
        ldr_imm(reg_na, reg_param, GET_OFF(na));
        ldr(WReg(reg_ld_loop.getIdx()), ptr(reg_tmp));
        mov(reg_aux_a1, reg_a);
        mov(reg_aux_c1, reg_c);
        mov(reg_aux_c, reg_aux_c1);
        mov(reg_zp_aux_b, reg_zp_b);

        asm_do_while(reg_ld_loop, [&]() {
            if (brg_.is_per_m_scales)
                ldr_imm(reg_aux_src_scales, reg_param, GET_OFF(src_scales));
            ldr(WReg(reg_bd_loop.getIdx()), ptr(reg_na));
            asm_do_while(reg_bd_loop, [&]() {
                loop_k(bdb, ldb, 0);

                add_imm(reg_aux_a1, reg_aux_a1,
                        div_up(brg_.K, brg_.k_blk) * brg_.k_blk * brg_.bd_block,
                        X_TMP_0);
                add_imm(reg_aux_c, reg_aux_c,
                        brg_.N * brg_.bd_block * brg_.dst_dt_sz, X_TMP_0);
                add_imm(reg_zp_aux_b, reg_zp_aux_b, brg_.m_blk * brg_.acc_dt_sz,
                        X_TMP_0);
                if (brg_.is_per_m_scales)
                    add_imm(reg_aux_src_scales, reg_aux_src_scales,
                            brg_.m_blk * sizeof(float), X_TMP_0);
            });

            mov(reg_aux_a1, reg_a);
            mov(reg_zp_aux_b, reg_zp_b);
            add_imm(reg_b, reg_b,
                    (brg_.n_blk * brg_.ld_block) * div_up(brg_.K, brg_.k_blk)
                            * brg_.k_blk,
                    X_TMP_0);
            add_imm(reg_aux_c1, reg_aux_c1,
                    brg_.dst_dt_sz * (brg_.n_blk * brg_.ld_block), X_TMP_0);
            add_imm(reg_zp_a, reg_zp_a,
                    brg_.n_blk * brg_.ld_block * brg_.acc_dt_sz, X_TMP_0);
            if (brg_.is_oc_scales)
                add_imm(reg_wei_scales, reg_wei_scales,
                        brg_.acc_dt_sz * (brg_.n_blk * brg_.ld_block), X_TMP_0);
            add_imm(reg_bias, reg_bias,
                    brg_.acc_dt_sz * (brg_.n_blk * brg_.ld_block), X_TMP_0);
            mov(reg_aux_c, reg_aux_c1);
        });
    }

    void han_blk_zp() {
        ldr_imm(reg_tmp, reg_param, GET_OFF(nb));
        ldr_imm(reg_na, reg_param, GET_OFF(na));
        mov(reg_aux_a1, reg_a);
        ldr(WReg(reg_bd_loop.getIdx()), ptr(reg_na));
        ldr(WReg(reg_ld_loop.getIdx()), ptr(reg_tmp));

        if (brg_.zp_type_b != jit_int8_broadcast_t::none) {
            asm_for(reg_bd_loop, reg_bd_loop, [&]() {
                loop_k_zp(bdb, ldb, 0, 1);

                add_imm(reg_aux_a1, reg_aux_a1,
                        div_up(brg_.K, brg_.k_blk) * brg_.k_blk * brg_.bd_block,
                        X_TMP_0);
                add_imm(reg_zp_b, reg_zp_b, brg_.m_blk * brg_.acc_dt_sz,
                        X_TMP_0);
            });
        }
        if (brg_.zp_type_a != jit_int8_broadcast_t::none) {
            asm_for(reg_ld_loop, reg_ld_loop, [&]() {
                loop_k_zp(bdb, ldb, 1, 0);

                add_imm(reg_zp_a, reg_zp_a,
                        brg_.n_blk * brg_.ld_block * brg_.acc_dt_sz, X_TMP_0);
                add_imm(reg_b, reg_b,
                        (brg_.n_blk * brg_.ld_block)
                                * div_up(brg_.K, brg_.k_blk) * brg_.k_blk,
                        X_TMP_0);
            });
        }
    }
    void zp_comp(int rdb, int bdb, int ldb, int is_a, int is_b) {
        const ZReg zp_b_sum0 = ZReg(3);
        const ZReg zp_b_sum1 = ZReg(4);
        const ZReg zp_b_sum2 = ZReg(5);
        const ZReg zp_b_sum3 = ZReg(6);

        dup(z0.b, 1);
        int rd, ld;
        if (brg_.zp_type_b != jit_int8_broadcast_t::none && is_b == 1) {
            mov(reg_tmp, reg_aux_a);
            for (rd = 0; rd < rdb; rd++) {
                if (simd_bytes(isa) == 16) {
                    ld1b(z1.b, P_ALL_ONE / T_z, ptr(reg_tmp));
                    if (brg_.is_s8) {
                        smmla(zp_b_sum0.s, z0.b, z1.b);
                    } else {
                        ummla(zp_b_sum0.s, z0.b, z1.b);
                    }
                    ld1b(z1.b, P_ALL_ONE / T_z, ptr(reg_tmp, 1, MUL_VL));
                    if (brg_.is_s8) {
                        smmla(zp_b_sum1.s, z0.b, z1.b);
                    } else {
                        ummla(zp_b_sum1.s, z0.b, z1.b);
                    }
                    ld1b(z1.b, P_ALL_ONE / T_z, ptr(reg_tmp, 2, MUL_VL));
                    if (brg_.is_s8) {
                        smmla(zp_b_sum2.s, z0.b, z1.b);
                    } else {
                        ummla(zp_b_sum2.s, z0.b, z1.b);
                    }
                    ld1b(z1.b, P_ALL_ONE / T_z, ptr(reg_tmp, 3, MUL_VL));
                    if (brg_.is_s8) {
                        smmla(zp_b_sum3.s, z0.b, z1.b);
                    } else {
                        ummla(zp_b_sum3.s, z0.b, z1.b);
                    }
                } else {
                    ld1b(z1.b, P_ALL_ONE / T_z, ptr(reg_tmp));
                    ld1b(z2.b, P_ALL_ONE / T_z, ptr(reg_tmp, 1, MUL_VL));
                    if (brg_.is_s8) {
                        smmla(zp_b_sum0.s, z0.b, z1.b);
                        smmla(zp_b_sum1.s, z0.b, z2.b);
                    } else {
                        ummla(zp_b_sum0.s, z0.b, z1.b);
                        ummla(zp_b_sum1.s, z0.b, z2.b);
                    }
                }
                add_imm(reg_tmp, reg_tmp, brg_.k_blk * brg_.m_blk, X_TMP_0);
            }
        }
        if ((brg_.zp_type_a != jit_int8_broadcast_t::none) && is_a == 1) {
            mov(reg_tmp, reg_aux_b);

            for (rd = 0; rd < rdb; rd++) {
                for (ld = 0; ld < ldb; ld++) {
                    PReg p = (brg_.is_n_tail && ld == ldb - 1) ? prd_ld
                                                               : P_ALL_ONE;
                    ld1b(acc(1, ld).b, p, ptr(reg_tmp, ld, MUL_VL));
                }
                add_imm(reg_tmp, reg_tmp,
                        brg_.k_blk * brg_.n_blk * brg_.ld_block, X_TMP_0);
                for (ld = 0; ld < ldb; ld++) {
                    if (brg_.is_s8 || brg_.is_u8_s8) {
                        smmla(acc(2, ld).s, z0.b, acc(1, ld).b);
                    } else {
                        ummla(acc(2, ld).s, z0.b, acc(1, ld).b);
                    }
                }
            }
        }
    }

    void config() {
        const int vlen_bytes = simd_bytes(isa);
        const int sv_len = vlen_bytes / brg_.acc_dt_sz;
        int m, pred_st = 0, pred_ld = 0, pred_b = sv_len;
        const int n_full = brg_.n_blk * brg_.ld_block;
        const int n_cols = brg_.is_n_tail ? brg_.n_tail : n_full;
        n_blks = div_up(n_cols, sv_len);
        k_full_blks = brg_.K / (brg_.k_blk * brg_.rd_block);
        m = brg_.K % (brg_.k_blk * brg_.rd_block);
        k_tail_blk = m / brg_.k_blk;
        k_residual_blk = m % brg_.k_blk;
        ldb = (brg_.is_n_tail) ? div_up(brg_.n_tail, brg_.n_blk)
                               : brg_.ld_block;
        bdb = (brg_.is_m_tail) ? div_up(brg_.m_tail, 2) : brg_.bd_block / 2;
        rdb = (brg_.is_k_tail) ? div_up(brg_.k_tail, brg_.k_blk) : 4;

        const int pred_zp_b_tl
                = (n_cols % sv_len == 0) ? sv_len : n_cols % sv_len;
        set_preg(prd_8.b, sv_len, X_TMP_0);
        set_preg(prd_zp_b_tl.b, pred_zp_b_tl, X_TMP_0);

        if (brg_.is_n_tail) {
            pred_b = (brg_.n_tail % sv_len == 0) ? sv_len
                                                 : (brg_.n_tail % sv_len);
            if (brg_.n_tail % brg_.n_blk == 0) {
                pred_st = (brg_.n_tail % (brg_.n_blk * 2) == 0) ? sv_len
                                                                : sv_len / 2;
                pred_ld = vlen_bytes;
            } else {
                pred_ld = (brg_.n_tail % brg_.n_blk) * brg_.k_blk;
                pred_st = (ldb % 2 == 0)
                        ? (sv_len / 2) + (brg_.n_tail % brg_.n_blk)
                        : (brg_.n_tail % brg_.n_blk);
            }
        }
        set_preg(prd_ld.b, pred_ld, X_TMP_0);
        set_preg(prd_st.s, pred_st, X_TMP_0);
        set_preg(prd_b.s, pred_b, X_TMP_0);
    }

    void generate() override {
        auto live_gprs = claim_regs<XReg>({reg_aux_src_scales, reg_zp_val_c,
                reg_a, reg_b, reg_c, reg_aux_a, reg_aux_b, reg_aux_c,
                reg_aux_a1, reg_zp_aux_b_buf, reg_aux_c1, reg_ld_loop,
                reg_rd_loop, reg_bd_loop, reg_tmp, reg_wei_scales,
                reg_src_scales, reg_aux_scales, reg_na, reg_bias, reg_zp_a,
                reg_zp_b, reg_zp_aux_b});
        auto live_preds
                = claim_regs<PReg>({prd_ld, prd_st, prd_b, prd_8, prd_zp_b_tl});
        auto live_accumulators = claim_reg_range<ZReg>(
                brg_.ld_block + 1, (brg_.bd_block / 2) * brg_.ld_block);
        auto eltwise_table = alloc_reg<XReg>();
        auto eltwise_mask = alloc_reg<PReg>();
        for (auto &e : eltwise_.entry_) {
            eltwise_injectors_.emplace_back(utils::make_unique<
                    jit_uni_eltwise_injector_t<to_vla_sve(isa)>>(this,
                    reg_pool(), e.eltwise, eltwise_table.get(),
                    eltwise_mask.get()));
        }

        preamble();
        config();

        ldr_imm(reg_a, reg_param, GET_OFF(src));
        ldr_imm(reg_b, reg_param, GET_OFF(wei));
        ldr_imm(reg_c, reg_param, GET_OFF(dst));
        ldr_imm(reg_zp_b, reg_param, GET_OFF(zp_b_ptr));
        ldr_imm(reg_zp_a, reg_param, GET_OFF(zp_a_ptr));
        if (brg_.is_zp_cal) {
            ldr_imm(reg_zp_val_b, reg_param, GET_OFF(wei_zero_point));
            ldr_imm(reg_zp_val_a, reg_param, GET_OFF(src_zero_point));
            han_blk_zp();
        } else {
            ldr_imm(reg_bias, reg_param, GET_OFF(bias));
            ldr_imm(reg_src_scales, reg_param, GET_OFF(scales));
            ldr_imm(reg_wei_scales, reg_param, GET_OFF(wei_scales));
            ldr_imm(reg_aux_scales, reg_param, GET_OFF(dst_scales));
            ldr_imm(reg_zp_aux_b_buf, reg_param, GET_OFF(wei_zero_point_buf));
            han_blk();
        }

        postamble();
        if (!eltwise_injectors_.empty()) {
            for (int i = 0; i < (int)eltwise_injectors_.size(); i++) {
                eltwise_injectors_[i]->prepare_table();
            }
        }
    }

    jit_int8_matmul_kernel_t(
            const brg_int8_t &k, const dnnl::impl::post_ops_t &eltwise = {})
        : brg_(k), eltwise_(eltwise) {}
    ~jit_int8_matmul_kernel_t() override = default;

private:
    brg_int8_t brg_;
    const dnnl::impl::post_ops_t eltwise_;
    int ldb;
    int bdb;
    int rdb;
    int k_full_blks;
    int k_tail_blk;
    int k_residual_blk;
    int n_blks;
    std::vector<std::unique_ptr<jit_uni_eltwise_injector_t<to_vla_sve(isa)>>>
            eltwise_injectors_;
};

template <cpu_isa_t isa>
status_t jit_int8_matmul_t<isa>::pd_t::init(const engine_t *engine) {

    const auto src_type = src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;

    const memory_desc_wrapper src_d(src_md_);
    const memory_desc_wrapper weights_d(weights_md_);
    const memory_desc_wrapper dst_d(dst_md_);
    const memory_desc_wrapper bias_d(bias_md_);

    const bool no_runtime_dims_or_strides
            = !(src_d.has_runtime_dims_or_strides()
                    || weights_d.has_runtime_dims_or_strides());

    VDISPATCH_MATMUL(
            no_runtime_dims_or_strides, VERBOSE_RUNTIMEDIM_UNSUPPORTED);
    VDISPATCH_MATMUL(DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL,
            VERBOSE_UNSUPPORTED_THREADPOOL_RUNTIME);
    VDISPATCH_MATMUL(is_dense_format_kind(), VERBOSE_UNSUPPORTED_SPARSE_CFG);

    bool is_u8 = utils::everyone_is(u8, src_type, wei_type);
    bool is_s8 = utils::everyone_is(s8, src_type, wei_type);
    bool is_u8_s8 = utils::everyone_is(u8, src_type)
            && utils::everyone_is(s8, wei_type);
    const bool is_per_m_scales
            = attr()->scales_.get_mask(DNNL_ARG_SRC) == src_qmask_M();

    int dims = src_d.ndims();

    auto check_attr_scales = [&]() -> status_t {
        const std::vector<int> supported_args
                = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        const auto &scales = attr()->scales_;
        const auto &src_scales = scales.get(DNNL_ARG_SRC);
        const auto &wei_scales = scales.get(DNNL_ARG_WEIGHTS);
        auto is_src_scl = !scales.get(DNNL_ARG_SRC).has_default_values();
        auto is_wei_scl = !scales.get(DNNL_ARG_WEIGHTS).has_default_values();
        auto dst_scl_msk = scales.get(DNNL_ARG_DST).get_mask();
        auto wei_scl_msk = wei_scales.get_mask();
        auto src_scl_msk = src_scales.get_mask();
        const bool is_src_per_m = src_scl_msk == src_qmask_M();

        CHECK(attr_scales_ok(engine, supported_args,
                {quantization_mode::static_sazp},
                {{DNNL_ARG_SRC, {src_qmask_M()}}}));
        VDISPATCH_MATMUL(
                IMPLICATION(is_src_per_m, src_scales.has_default_groups()),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        VDISPATCH_MATMUL(IMPLICATION(is_src_scl,
                                 scales.has_default_data_type(DNNL_ARG_SRC)),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        VDISPATCH_MATMUL(
                !((src_scl_msk > 0 && !is_src_per_m)
                        || (wei_scl_msk > 0 && wei_scl_msk != 1 << (dims - 1))
                        || dst_scl_msk > 0),
                VERBOSE_UNSUPPORTED_SCALES_CFG);

        if (is_src_scl && is_wei_scl && wei_scl_msk > 0) {
            // This case requires scratchpad.
            VDISPATCH_MATMUL(
                    !is_runtime_value(N()), VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
        return status::success;
    };

    auto check_bias = [&]() -> bool {
        if (bias_d.format_any()) {
            if (bias_d.has_runtime_dims_or_strides()) return false;
            status_t status = memory_desc_init_by_strides(bias_md_, nullptr);
            if (status != status::success) return false;
        }

        const auto bia_dt = weights_md(1)->data_type;
        return IMPLICATION(with_bias(), bia_dt == f32 && is_bias_1xN());
    };

    auto init_zp_type = [&](brg_int8_t *brg_) -> bool {
        auto zero_points = attr()->zero_points_;

        auto wt_int8 = zero_points.get_data_type(DNNL_ARG_WEIGHTS) == u8
                || zero_points.get_data_type(DNNL_ARG_WEIGHTS) == s8;
        if (!zero_points.has_default_data_type(DNNL_ARG_SRC)
                || !zero_points.has_default_data_type(DNNL_ARG_DST)
                || (!zero_points.has_default_data_type(DNNL_ARG_WEIGHTS)
                        && !wt_int8))
            return false;

        if (!zero_points.has_default_data_type(DNNL_ARG_WEIGHTS)) {
            switch (zero_points.get_data_type(DNNL_ARG_WEIGHTS)) {
                case u8: {
                    brg_->zp_b_dt = u8;
                    brg_->is_zp_b_int8 = true;
                    break;
                }
                case s8: {
                    brg_->zp_b_dt = s8;
                    brg_->is_zp_b_int8 = true;
                    break;
                }
                case s32: {
                    brg_->is_zp_b_int8 = false;
                    break;
                }
                default: return false;
            }
        }

        if (zero_points.get_mask(DNNL_ARG_SRC) > 0
                || zero_points.get_mask(DNNL_ARG_DST) > 0
                || zero_points.get_mask(DNNL_ARG_WEIGHTS) > 0)
            return false;

        brg_->zp_type_a = zero_points.has_default_values(DNNL_ARG_SRC)
                ? jit_int8_broadcast_t::none
                : jit_int8_broadcast_t::common;

        brg_->zp_type_b = zero_points.has_default_values(DNNL_ARG_WEIGHTS)
                ? jit_int8_broadcast_t::none
                : jit_int8_broadcast_t::common;

        brg_->zp_type_c = zero_points.has_default_values(DNNL_ARG_DST)
                ? jit_int8_broadcast_t::none
                : jit_int8_broadcast_t::common;

        return true;
    };

    VDISPATCH_MATMUL(init_zp_type(&brg_int8_conf), VERBOSE_UNSUPPORTED_ZP_CFG);

    VDISPATCH_MATMUL(check_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);

    CHECK(check_attr_scales());

    const bool problem_dt_correct = (is_s8 || is_u8 || is_u8_s8)
            && utils::one_of(dst_type, f32, s32, f16, bf16, s8, u8)
            && platform::has_data_type_support(dst_type);

    VDISPATCH_MATMUL(problem_dt_correct, VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_MATMUL(post_ops_ok(), VERBOSE_UNSUPPORTED_POSTOP);
    VDISPATCH_MATMUL(formats_ok(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(mayiuse(sve) && utils::one_of(simd_bytes(isa), 16UL, 32UL),
            VERBOSE_UNSUPPORTED_ISA);

    auto is_src_any = src_d.format_kind() == format_kind::any;
    auto is_dst_any = dst_d.format_kind() == format_kind::any;

    matmul_helper_t helper(src_d, weights_d, dst_d);
    brg_int8_conf.K = helper.K();
    brg_int8_conf.M = helper.M();
    brg_int8_conf.N = helper.N();
    brg_int8_conf.batch = batch();

    brg_int8_conf.n_blk = simd_bytes(isa) / brg_int8_conf.k_blk;

    int num_threads = dnnl_get_current_num_threads();

    if (brg_int8_conf.N <= brg_int8_conf.n_blk * 2)
        brg_int8_conf.ld_block = 2;
    else if (brg_int8_conf.N <= brg_int8_conf.n_blk * 4)
        brg_int8_conf.ld_block = 4;
    else
        brg_int8_conf.ld_block = 6;
    const int micro_n = brg_int8_conf.n_blk * brg_int8_conf.ld_block;
    const bool tunable_weights_layout = dims == 2
            && (weights_d.format_any()
                    || weights_d.matches_tag(format_tag::BA12b8a));
    const bool no_zero_points = utils::everyone_is(jit_int8_broadcast_t::none,
            brg_int8_conf.zp_type_a, brg_int8_conf.zp_type_b,
            brg_int8_conf.zp_type_c);

    // Empirical boundaries from SVE128
    constexpr int min_tuning_threads = 32;
    constexpr int high_thread_count = 96;
    constexpr dim_t min_tuning_k = 128;
    constexpr dim_t min_float_tuning_channel = 2048;
    constexpr dim_t min_cache_partition_channel = 1024;
    constexpr dim_t min_narrow_f32_tuning_channel = 1280;
    constexpr dim_t f32_long_reduction_model_min_m = 2048;
    constexpr dim_t f32_wide_n_candidate_min_m = 2048;
    constexpr dim_t high_thread_wide_n_min_m = 128;
    constexpr dim_t low_thread_wide_n_min_m = 24;
    constexpr dim_t small_m_near_square_max = 64;
    constexpr dim_t near_square_n_slack_divisor = 15;
    constexpr dim_t medium_m_min = 128;
    constexpr dim_t large_m_min = 4096;
    constexpr dim_t long_reduction_k = 8192;
    constexpr dim_t wide_n_to_k_ratio = 4;
    constexpr dim_t reduction_k_to_n_ratio = 2;
    constexpr dim_t large_reduction_k_to_n_ratio = 3;
    const dim_t m = brg_int8_conf.M;
    const dim_t n = brg_int8_conf.N;
    const dim_t k = brg_int8_conf.K;
    const dim_t padded_k = rnd_up(k, static_cast<dim_t>(brg_int8_conf.k_blk));
    const double l1_cache_size = platform::get_per_core_cache_size(1);
    const double l2_cache_size = platform::get_per_core_cache_size(2);
    const double max_a_tile_bytes
            = static_cast<double>(
                      nstl::min(m, static_cast<dim_t>(max_tuned_m_block)))
            * padded_k;
    const dim_t wide_n_min_m = num_threads >= high_thread_count
            ? high_thread_wide_n_min_m
            : low_thread_wide_n_min_m;
    const bool wide_n = m >= wide_n_min_m && n >= wide_n_to_k_ratio * k;
    const bool medium_m_reduction = m >= medium_m_min && m < large_m_min
            && k >= reduction_k_to_n_ratio * n;
    const bool large_prefill = m >= large_m_min && k < long_reduction_k
            && (num_threads < high_thread_count || n <= k
                    || n >= wide_n_to_k_ratio * k);
    const dim_t min_channel = nstl::min(k, n);
    const bool fp32_tune_threshold = min_channel >= min_float_tuning_channel;
    const bool fp32_tune_shape = fp32_tune_threshold
            && (wide_n || medium_m_reduction || large_prefill);
    const bool small_gemv = m == 1 && k < long_reduction_k;

    const bool large_m_cache_pressure = m >= large_m_min
            && k >= large_reduction_k_to_n_ratio * n
            && max_a_tile_bytes > l2_stream_fraction * l2_cache_size;

    bool tune_dst = false;
    bool use_cache_partition = false;
    bool use_micro_m_block_limit = false;
    bool use_long_reduction_model = false;
    bool allow_wide_n_candidates = false;
    jit_int8_matmul_tile_hint_t cache_partition_hint;
    switch (dst_type) {
        case f32: {
            const dim_t max_channel = nstl::max(k, n);
            const bool cache_partition_shape = large_prefill
                    && min_channel >= min_cache_partition_channel
                    && min_channel < min_narrow_f32_tuning_channel
                    && max_channel <= wide_n_to_k_ratio * min_channel
                    && l1_cache_size > 0. && l2_cache_size > 0.
                    && static_cast<double>(brg_int8_conf.m_blk) * padded_k
                            <= cache_partition_a_l1_fraction * l1_cache_size
                    && static_cast<double>(n) * padded_k
                            >= cache_partition_min_b_l2_fraction
                                    * l2_cache_size;
            if (cache_partition_shape)
                cache_partition_hint = get_cache_partitioned_tile_hint(
                        brg_int8_conf, micro_n, l1_cache_size, l2_cache_size);
            use_cache_partition = cache_partition_hint.n_block_factor > 0
                    && div_up(m,
                               static_cast<dim_t>(
                                       cache_partition_hint.m_block_sz))
                            >= num_threads;

            const bool tune_narrow_large_prefill = large_prefill
                    && min_channel >= min_narrow_f32_tuning_channel
                    && min_channel < min_float_tuning_channel;
            const bool small_m_near_square = m >= 16
                    && m <= small_m_near_square_max && n >= k
                    && n - k <= k / near_square_n_slack_divisor;
            const bool tune_small_m_near_square
                    = fp32_tune_threshold && small_m_near_square;
            const bool tune_cache_pressure
                    = fp32_tune_threshold && large_m_cache_pressure;
            const bool k_dominant_long_reduction = k >= long_reduction_k
                    && k >= large_reduction_k_to_n_ratio * n;
            // Use the long-reduction search and scoring model from M=2048 onward
            // when K is at least 8192.
            use_long_reduction_model = fp32_tune_threshold
                    && m >= f32_long_reduction_model_min_m
                    && k_dominant_long_reduction;
            allow_wide_n_candidates = m >= f32_wide_n_candidate_min_m
                    && (wide_n || large_prefill);
            use_micro_m_block_limit
                    = tune_cache_pressure || use_long_reduction_model;
            tune_dst = (!large_m_cache_pressure || tune_cache_pressure)
                    && (fp32_tune_shape || tune_narrow_large_prefill
                            || tune_small_m_near_square || tune_cache_pressure
                            || use_long_reduction_model || use_cache_partition);
            break;
        }
        case bf16: tune_dst = !large_m_cache_pressure && fp32_tune_shape; break;
        case s32: tune_dst = !large_m_cache_pressure; break;
        default: break;
    }

    // Tune only outer M/N partitioning; preserve the established micro-N and
    // packed-weight layout.
    const bool use_tile_tuner = m > 0 && num_threads >= min_tuning_threads
            && simd_bytes(isa) == 16 && brg_int8_conf.ld_block == 6
            && k >= min_tuning_k && tunable_weights_layout && no_zero_points
            && attr()->post_ops_.len() == 0 && !with_bias() && is_s8
            && !small_gemv && tune_dst;
    jit_int8_matmul_tile_hint_t tile_hint;
    if (use_tile_tuner) {
        if (use_cache_partition) {
            tile_hint = cache_partition_hint;
        } else {
            const int max_search_m_block = use_micro_m_block_limit
                    ? brg_int8_conf.m_blk
                    : max_tuned_m_block;
            tile_hint = get_thread_tile_hint(brg_int8_conf, num_threads,
                    micro_n, max_search_m_block, allow_wide_n_candidates,
                    use_long_reduction_model);
        }
    }
    const auto thread_tiles = get_thread_tiles(
            brg_int8_conf, num_threads, simd_bytes(isa), micro_n, tile_hint);
    m_block_sz = thread_tiles.m_block_sz;
    n_block_sz = thread_tiles.n_block_sz;
    if (use_tile_tuner && tile_hint.n_block_factor > 0) {
        VDEBUGINFO(1, primitive, matmul,
                "jit_int8_matmul autotuner: policy=%s M=%ld N=%ld K=%ld "
                "nthr=%d m_block=%d n_block=%d",
                use_cache_partition ? "cache_partition" : "scored", (long)m,
                (long)n, (long)k, num_threads, m_block_sz, n_block_sz);
    }

    int num_a_blocks = div_up(brg_int8_conf.M, m_block_sz);
    int num_b_blocks = div_up(brg_int8_conf.N, n_block_sz);
    mm_parallel_work = brg_int8_conf.batch * num_a_blocks * num_b_blocks;
    if (mm_parallel_work < num_threads && brg_int8_conf.ld_block == 4)
        n_block_sz = micro_n;
    num_b_blocks = div_up(brg_int8_conf.N, n_block_sz);
    mm_parallel_work = brg_int8_conf.batch * num_a_blocks * num_b_blocks;

    auto b_tag_2d = format_tag::ab;
    auto b_tag_3d = format_tag::abc;
    auto b_tag_4d = format_tag::abcd;

    switch (micro_n) {
        case 4: {
            b_tag_2d = format_tag::BA4b8a;
            b_tag_3d = format_tag::aCB4c8b;
            b_tag_4d = format_tag::abDC4d8c;
            break;
        }
        case 8: {
            b_tag_2d = format_tag::BA8b8a;
            b_tag_3d = format_tag::aCB8c8b;
            b_tag_4d = format_tag::abDC8d8c;
            break;
        }
        case 12: {
            b_tag_2d = format_tag::BA12b8a;
            b_tag_3d = format_tag::aCB12c8b;
            b_tag_4d = format_tag::abDC12d8c;
            break;
        }
        case 16: {
            b_tag_2d = format_tag::BA16b8a;
            b_tag_3d = format_tag::aCB16c8b;
            b_tag_4d = format_tag::abDC16d8c;
            break;
        }
        case 24: {
            b_tag_2d = format_tag::BA24b8a;
            b_tag_3d = format_tag::aCB24c8b;
            b_tag_4d = format_tag::abDC24d8c;
            break;
        }
        default: return status::unimplemented;
    }

    switch (dims) {
        case 2: {
            if (is_src_any)
                VCHECK_BG(memory_desc_init_by_tag(src_md_, format_tag::ab),
                        VERBOSE_UNSUPPORTED_TAG);
            if (is_dst_any)
                VCHECK_BG(memory_desc_init_by_tag(dst_md_, format_tag::ab),
                        VERBOSE_UNSUPPORTED_TAG);
            if (!weights_d.matches_tag(format_tag::ab)) {
                brg_int8_conf.b_reo = false;
                VCHECK_BG(memory_desc_init_by_tag(weights_md_, b_tag_2d),
                        VERBOSE_UNSUPPORTED_TAG);
            } else {
                VCHECK_BG(memory_desc_init_by_tag(weights_md_, format_tag::ab),
                        VERBOSE_UNSUPPORTED_TAG);
            }
            break;
        }
        case 3: {
            if (is_src_any)
                VCHECK_BG(memory_desc_init_by_tag(src_md_, format_tag::abc),
                        VERBOSE_UNSUPPORTED_TAG);
            if (is_dst_any)
                VCHECK_BG(memory_desc_init_by_tag(dst_md_, format_tag::abc),
                        VERBOSE_UNSUPPORTED_TAG);
            if (!weights_d.matches_tag(format_tag::abc)) {
                brg_int8_conf.b_reo = false;
                VCHECK_BG(memory_desc_init_by_tag(weights_md_, b_tag_3d),
                        VERBOSE_UNSUPPORTED_TAG);
            } else {
                VCHECK_BG(memory_desc_init_by_tag(weights_md_, format_tag::abc),
                        VERBOSE_UNSUPPORTED_TAG);
            }
            if (src_d.dims()[0] != weights_d.dims()[0])
                return status::unimplemented;
            break;
        }
        case 4: {
            if (is_src_any)
                VCHECK_BG(memory_desc_init_by_tag(src_md_, format_tag::abcd),
                        VERBOSE_UNSUPPORTED_TAG);
            if (is_dst_any)
                VCHECK_BG(memory_desc_init_by_tag(dst_md_, format_tag::abcd),
                        VERBOSE_UNSUPPORTED_TAG);
            if (!weights_d.matches_tag(format_tag::abcd)) {
                brg_int8_conf.b_reo = false;
                VCHECK_BG(memory_desc_init_by_tag(weights_md_, b_tag_4d),
                        VERBOSE_UNSUPPORTED_TAG);
            } else {
                VCHECK_BG(
                        memory_desc_init_by_tag(weights_md_, format_tag::abcd),
                        VERBOSE_UNSUPPORTED_TAG);
            }
            if (src_d.dims()[0] != weights_d.dims()[0]
                    || src_d.dims()[1] != weights_d.dims()[1])
                return status::unimplemented;
            break;
        }
        default: return status::unimplemented;
    }

    const bool has_src_scales
            = !attr()->scales_.get(DNNL_ARG_SRC).has_default_values();
    const bool has_wei_scales
            = !attr()->scales_.get(DNNL_ARG_WEIGHTS).has_default_values();
    bool is_scales = has_src_scales || has_wei_scales;

    bool is_dst_scales
            = !attr()->scales_.get(DNNL_ARG_DST).has_default_values();

    const auto &wei_scales = attr()->scales_.get(DNNL_ARG_WEIGHTS);

    brg_int8_conf.dst_dt = dst_type;
    brg_int8_conf.dst_dt_sz = types::data_type_size(dst_type);
    brg_int8_conf.m_tail = brg_int8_conf.M % brg_int8_conf.m_blk;
    brg_int8_conf.k_tail
            = brg_int8_conf.K % (brg_int8_conf.k_blk * brg_int8_conf.rd_block);
    brg_int8_conf.n_tail
            = brg_int8_conf.N % (brg_int8_conf.n_blk * brg_int8_conf.ld_block);
    brg_int8_conf.is_s8 = is_s8;
    brg_int8_conf.is_u8_s8 = is_u8_s8;
    brg_int8_conf.is_bias = with_bias();
    brg_int8_conf.with_scales = is_scales;
    brg_int8_conf.with_src_scales = has_src_scales;
    brg_int8_conf.with_wei_scales = has_wei_scales;
    brg_int8_conf.with_dst_scales = is_dst_scales;
    brg_int8_conf.is_oc_scales = wei_scales.get_mask() > 0;
    brg_int8_conf.is_per_m_scales = is_per_m_scales;
    dyn_vals.K = brg_int8_conf.K;
    dyn_vals.N = brg_int8_conf.N;
    dyn_vals.M = brg_int8_conf.M;
    dyn_vals.batch = brg_int8_conf.batch;
    dyn_vals.mtail = brg_int8_conf.m_tail;
    dyn_vals.m_blk = brg_int8_conf.m_blk;
    dyn_vals.k_blk = brg_int8_conf.k_blk;
    dyn_vals.n_blk = brg_int8_conf.n_blk * brg_int8_conf.ld_block;
    dyn_vals.ntail = brg_int8_conf.n_tail;
    dyn_vals.ktail = dyn_vals.K % brg_int8_conf.k_blk;

    auto scratchpad = scratchpad_registry().registrar();
    if (brg_int8_conf.zp_type_a != jit_int8_broadcast_t::none)
        scratchpad.book(key_brgemm_primitive_zp_comp_a,
                div_up(brg_int8_conf.N,
                        (brg_int8_conf.n_blk * brg_int8_conf.ld_block))
                        * (brg_int8_conf.n_blk * brg_int8_conf.ld_block)
                        * brg_int8_conf.acc_dt_sz * brg_int8_conf.batch,
                sizeof(char));
    if (brg_int8_conf.zp_type_b != jit_int8_broadcast_t::none)
        scratchpad.book(key_brgemm_primitive_zp_comp_b,
                div_up(brg_int8_conf.M, brg_int8_conf.m_blk)
                        * brg_int8_conf.m_blk * brg_int8_conf.acc_dt_sz
                        * brg_int8_conf.batch,
                sizeof(char));
    scratchpad.book(key_brgemm_primitive_buffer_a,
            brg_int8_conf.batch * div_up(brg_int8_conf.M, brg_int8_conf.m_blk)
                    * div_up(brg_int8_conf.K, brg_int8_conf.k_blk)
                    * brg_int8_conf.m_blk * brg_int8_conf.k_blk,
            sizeof(char));
    if (brg_int8_conf.b_reo)
        scratchpad.book(key_gemm_blocked_b,
                brg_int8_conf.batch
                        * div_up(brg_int8_conf.N,
                                (brg_int8_conf.n_blk * brg_int8_conf.ld_block))
                        * (brg_int8_conf.n_blk * brg_int8_conf.ld_block)
                        * div_up(brg_int8_conf.K, brg_int8_conf.k_blk)
                        * brg_int8_conf.k_blk,
                sizeof(char));
    book_precomputed_scales(scratchpad, attr()->scales_, N());

    return status::success;
}

template <cpu_isa_t isa>
bool jit_int8_matmul_t<isa>::pd_t::formats_ok() const {
    const memory_desc_wrapper src_d(src_md_);
    const memory_desc_wrapper weights_d(weights_md_);
    const memory_desc_wrapper dst_d(dst_md_);
    const bool is_dst = dst_d.matches_one_of_tag(format_tag::ab,
                                format_tag::abc, format_tag::abcd)
                    != format_tag::undef
            || dst_d.format_kind() == format_kind::any;
    const bool is_wei
            = weights_d.matches_one_of_tag(format_tag::ab, format_tag::abc,
                      format_tag::abcd, format_tag::BA24b8a,
                      format_tag::aCB24c8b, format_tag::abDC24d8c,
                      format_tag::BA16b8a, format_tag::aCB16c8b,
                      format_tag::abDC16d8c, format_tag::BA12b8a,
                      format_tag::aCB12c8b, format_tag::abDC12d8c,
                      format_tag::BA4b8a, format_tag::aCB4c8b,
                      format_tag::abDC4d8c, format_tag::BA8b8a,
                      format_tag::aCB8c8b, format_tag::abDC8d8c)
                    != format_tag::undef
            || weights_d.format_kind() == format_kind::any;
    const bool is_src = src_d.matches_one_of_tag(format_tag::ab,
                                format_tag::abc, format_tag::abcd)
                    != format_tag::undef
            || src_d.format_kind() == format_kind::any;
    return is_dst && is_wei && is_src;
}

template <cpu_isa_t isa>
bool jit_int8_matmul_t<isa>::pd_t::post_ops_ok() const {
    if (attr()->post_ops_.has_default_values()) { return true; }

    const auto &eltwise_ok = [&](dnnl_post_ops::entry_t op) -> bool {
        return op.is_eltwise()
                && eltwise_injector::is_supported(isa, op.eltwise.alg);
    };

    const auto &post_ops = attr()->post_ops_.entry_;

    return std::all_of(post_ops.begin(), post_ops.end(), eltwise_ok);
}

template <cpu_isa_t isa>
int jit_int8_matmul_t<isa>::pd_t::get_idx(bool is_zp_cal, bool is_m_tail,
        bool is_k_tail, bool is_n_tail, const brg_int8_t &brg) const {
    if (brg.zp_type_a == jit_int8_broadcast_t::none
            && brg.zp_type_b == jit_int8_broadcast_t::none && is_zp_cal) {
        return -1;
    }

    int m_tail = brg.M % brg.m_blk;
    int n_tail = brg.N % (brg.n_blk * brg.ld_block);
    int k_tail = brg.K % (brg.k_blk * 4);

    if ((is_m_tail && m_tail == 0) || (is_k_tail && k_tail == 0)
            || (is_n_tail && n_tail == 0) || (!is_k_tail && k_tail == 1)) {
        return -1;
    }

    return static_cast<int>(is_k_tail) + static_cast<int>(is_n_tail) * 2
            + static_cast<int>(is_m_tail) * 2 * 2
            + static_cast<int>(is_zp_cal) * 2 * 2 * 2;
}

template <cpu_isa_t isa>
status_t jit_int8_matmul_t<isa>::init(engine_t *engine) {
    dyn_vals_t dyn_vals = pd()->dyn_vals;
    brg_int8_t brg_int8_conf = pd()->brg_int8_conf;

    const bool has_eltwise
            = pd()->attr()->post_ops_.find(primitive_kind_t::dnnl_eltwise) >= 0;

    // Iterate through every combination of false (0) and true (1)
    for (bool is_zp_cal : {false, true})
        for (bool is_m_tail : {false, true})
            for (bool is_n_tail : {false, true})
                for (bool is_k_tail : {false, true}) {
                    int idx = pd()->get_idx(is_zp_cal, is_m_tail, is_k_tail,
                            is_n_tail, brg_int8_conf);
                    if (idx == -1) continue;
                    brg_int8_conf.is_m_tail = is_m_tail;
                    brg_int8_conf.is_k_tail = is_k_tail;
                    brg_int8_conf.is_n_tail = is_n_tail;
                    brg_int8_conf.is_zp_cal = is_zp_cal;

                    const auto post_ops = has_eltwise ? pd()->attr()->post_ops_
                                                      : post_ops_t {};

                    safe_ptr_assign(int8_kernels_[idx],
                            new jit_int8_matmul_kernel_t<isa>(
                                    brg_int8_conf, post_ops));

                    CHECK(int8_kernels_[idx]->create_kernel());
                }

    safe_ptr_assign(reo_ker_a_,
            new jit_int8_matmul_utils_kernel_t(dyn_vals, isa,
                    jit_int8_matmul_utils_kernel_t::alg::reorder_src));
    CHECK(reo_ker_a_->create_kernel());

    safe_ptr_assign(reo_ker_b_,
            new jit_int8_matmul_utils_kernel_t(dyn_vals, isa,
                    jit_int8_matmul_utils_kernel_t::alg::reorder_wei));
    CHECK(reo_ker_b_->create_kernel());

    return status::success;
}

template <cpu_isa_t isa>
jit_int8_matmul_t<isa>::jit_int8_matmul_t(const pd_t *apd) : primitive_t(apd) {}
template <cpu_isa_t isa>
jit_int8_matmul_t<isa>::~jit_int8_matmul_t() = default;

template <cpu_isa_t isa>
status_t jit_int8_matmul_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto *wei = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    const auto *src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(uint8_t *, DNNL_ARG_DST);
    const auto *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const int32_t *wei_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const auto &b = pd()->brg_int8_conf;
    const auto &d = pd()->dyn_vals;

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    int num_threads = dnnl_get_current_num_threads();

    char *src_reorder_buffer
            = scratchpad.template get<char>(key_brgemm_primitive_buffer_a);
    char *wei_reorder_buffer = (b.b_reo)
            ? scratchpad.template get<char>(key_gemm_blocked_b)
            : (char *)wei;

    float *zp_ptr_a = nullptr;
    if (b.zp_type_a != jit_int8_broadcast_t::none)
        zp_ptr_a = scratchpad.template get<float>(
                key_brgemm_primitive_zp_comp_a);

    float *zp_ptr_b = nullptr;
    if (b.zp_type_b != jit_int8_broadcast_t::none)
        zp_ptr_b = scratchpad.template get<float>(
                key_brgemm_primitive_zp_comp_b);

    alignas(16) float unit_scale_buf[16] = {};
    utils::array_set(unit_scale_buf, 1.0f, 16);
    const auto &attr_scales = pd()->attr()->scales_;
    const bool need_wei_scale_copy = b.with_wei_scales
            && (!attr_scales.has_default_data_type(DNNL_ARG_WEIGHTS)
                    || !attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups());
    const float *kernel_wei_scales = need_wei_scale_copy
            ? precompute_scales(scratchpad, unit_scale_buf, wei_scales,
                      pd()->N(), pd()->attr())
            : wei_scales;

    const dim_t batch = b.batch;
    const dim_t M = b.M;
    const dim_t N = b.N;
    const dim_t K = b.K;

    auto reorder_a = [&]() {
        int num_m_blks = div_up(M, b.m_blk);
        int num_k_blks = div_up(K, b.k_blk);
        int blks_per_batch = num_m_blks * num_k_blks;
        int parallel_work = batch * blks_per_batch;
        // If parallel_work == 1, we limit num threads to 1 as parallel(1, ...)
        // does not create a parallel section. We do not limit number of threads
        // for case 1 < parallel_work_amount_ < dnnl_get_max_threads() to avoid
        // potential overhead on spawning different number of OMP threads from
        // layer to layer.
        int adjusted_threads = parallel_work > 1 ? num_threads : 1;

        parallel(adjusted_threads, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            balance211(parallel_work, adjusted_threads, ithr, start, end);

            // batch within which this thread's work range begins
            const int start_batch = start / blks_per_batch;
            // how far into the current batch this thread starts its work
            int thread_start_offset = start % blks_per_batch;
            const int thread_work_range = end - start;
            int batches_of_this_thread = 1;
            int work_for_first_batch = end - start;
            int work_for_last_batch = 0;

            // If this thread's work straddles more than one batch, split it up
            // as:
            // first batch work + n * full batches + last batch work
            const auto spillover
                    = thread_start_offset + thread_work_range - blks_per_batch;
            if (spillover > 0) {
                batches_of_this_thread += div_up(spillover, blks_per_batch);

                work_for_first_batch = blks_per_batch - thread_start_offset;
                work_for_last_batch = spillover % blks_per_batch;

                if (work_for_last_batch == 0)
                    work_for_last_batch = blks_per_batch;
            }

            int work_this_batch {};
            for (int i = 0; i < batches_of_this_thread; i++) {
                if (i == 0) {
                    work_this_batch = work_for_first_batch;
                } else if (i == (batches_of_this_thread - 1)) {
                    work_this_batch = work_for_last_batch;
                } else {
                    work_this_batch = blks_per_batch;
                }

                thread_start_offset = start % blks_per_batch;
                const int current_batch = start_batch + i;
                const int m_blk_src = thread_start_offset / num_k_blks;
                const int k_blk_src = thread_start_offset % num_k_blks;
                const int m_blk_dst = thread_start_offset / num_k_blks;
                const int k_blk_dst = thread_start_offset % num_k_blks;

                int k1 = std::min(num_k_blks - k_blk_src, work_this_batch);
                int k_tmp = work_this_batch - k1;
                int m1 = (k_tmp > 0) ? k_tmp / num_k_blks : 0;
                int k2 = (k_tmp > 0) ? k_tmp % num_k_blks : 0;
                int src_ad = (current_batch * M * K) + (m_blk_src * b.m_blk * K)
                        + (k_blk_src * b.k_blk);
                int dst_ad = (current_batch * num_m_blks * num_k_blks * b.m_blk
                                     * b.k_blk)
                        + (m_blk_dst * num_k_blks * b.m_blk * b.k_blk)
                        + (k_blk_dst * b.m_blk * b.k_blk);
                int src_new = src_ad, dst_new = dst_ad;

                dyn_params_t k;

                if (k1 > 0) {
                    int a = 1;
                    bool is_m_tail
                            = (d.mtail > 0) && !(m_blk_src < num_m_blks - 1);
                    bool is_k_tail
                            = (d.ktail > 0) && !(k1 + k_blk_src < num_k_blks);
                    k.src = (int8_t *)src + src_ad;
                    k.dst = (int8_t *)src_reorder_buffer + dst_ad;
                    k.nm = &a;
                    k.nk = &k1;
                    k.is_k_tail = &is_k_tail;
                    k.is_m_tail = &is_m_tail;
                    (*reo_ker_a_)(&k);
                }

                if (m1 > 0) {
                    bool is_m_tail = (d.mtail > 0)
                            && !(m1 + m_blk_src < num_m_blks - 1);
                    bool is_k_tail = (d.ktail > 0);
                    if (k1 != num_k_blks) {
                        src_new = src_ad - b.k_blk * (num_k_blks - k1)
                                + b.m_blk * K;
                    } else {
                        src_new = src_ad + b.m_blk * K;
                    }
                    dst_new = dst_ad + b.m_blk * b.k_blk * k1;
                    k.src = (int8_t *)src + src_new;
                    k.dst = (int8_t *)src_reorder_buffer + dst_new;
                    k.nm = &m1;
                    k.nk = &num_k_blks;
                    k.is_k_tail = &is_k_tail;
                    k.is_m_tail = &is_m_tail;
                    (*reo_ker_a_)(&k);
                }
                if (k2 > 0) {
                    int a = 1;
                    bool is_m_tail = (d.mtail > 0)
                            && !(m1 + m_blk_src < num_m_blks - 2);
                    bool is_k_tail = false;
                    if (m1 < 1) {
                        src_new = src_ad - b.k_blk * (num_k_blks - k1)
                                + (b.m_blk * K);
                        dst_new = dst_ad + b.m_blk * b.k_blk * k1;
                    } else {
                        src_new += K * m1 * b.m_blk;
                        dst_new += b.m_blk * b.k_blk * num_k_blks * m1;
                    }
                    k.src = (int8_t *)src + src_new;
                    k.dst = (int8_t *)src_reorder_buffer + dst_new;
                    k.nm = &a;
                    k.nk = &k2;
                    k.is_k_tail = &is_k_tail;
                    k.is_m_tail = &is_m_tail;
                    (*reo_ker_a_)(&k);
                }

                start += work_this_batch;
            }
        });
    };

    auto reorder_b = [&]() {
        const int num_k_blks = div_up(K, d.k_blk);
        const int num_n_blks = div_up(N, d.n_blk);
        const int blks_per_batch = num_n_blks * num_k_blks;
        const int parallel_work = batch * blks_per_batch;
        // If parallel_work == 1, we limit num threads to 1 as parallel(1, ...)
        // does not create a parallel section. We do not limit number of threads
        // for case 1 < parallel_work_amount_ < dnnl_get_max_threads() to avoid
        // potential overhead on spawning different number of OMP threads from
        // layer to layer.
        const int adjusted_threads = parallel_work > 1 ? num_threads : 1;

        parallel(adjusted_threads, [&](const int ithr, const int nthr) {
            int start {0}, end {0};
            balance211(parallel_work, adjusted_threads, ithr, start, end);
            // batch within which this thread's work range begins
            const int start_batch = start / blks_per_batch;
            // how far into the current batch this thread starts its work
            int thread_start_offset = start % blks_per_batch;
            const int thread_work_range = end - start;
            int batches_of_this_thread = 1;
            int work_for_first_batch = end - start;
            int work_for_last_batch = 0;

            // If this thread's work straddles more than one batch, split it up
            // as:
            // first batch work + n * full batches + last batch work
            const auto spillover
                    = thread_start_offset + thread_work_range - blks_per_batch;
            if (spillover > 0) {
                batches_of_this_thread += div_up(spillover, blks_per_batch);

                work_for_first_batch = blks_per_batch - thread_start_offset;
                work_for_last_batch = spillover % blks_per_batch;

                if (work_for_last_batch == 0)
                    work_for_last_batch = blks_per_batch;
            }

            int work_this_batch {};
            for (int i = 0; i < batches_of_this_thread; i++) {
                if (i == 0) {
                    work_this_batch = work_for_first_batch;
                } else if (i == (batches_of_this_thread - 1)) {
                    work_this_batch = work_for_last_batch;
                } else {
                    work_this_batch = blks_per_batch;
                }

                thread_start_offset = start % blks_per_batch;
                const int current_batch = start_batch + i;
                const int n_blk_src = thread_start_offset / num_k_blks;
                const int k_blk_src = thread_start_offset % num_k_blks;
                const int n_blk_dst = thread_start_offset / num_k_blks;
                const int k_blk_dst = thread_start_offset % num_k_blks;

                int k1 = std::min(num_k_blks - k_blk_src, work_this_batch);
                int k_tmp = work_this_batch - k1;
                int n1 = (k_tmp > 0) ? k_tmp / num_k_blks : 0;
                int k2 = (k_tmp > 0) ? k_tmp % num_k_blks : 0;
                int src_ad = (current_batch * N * K) + (n_blk_src * d.n_blk)
                        + (k_blk_src * d.k_blk * N);
                int dst_ad = (current_batch * num_n_blks * num_k_blks * d.k_blk
                                     * d.n_blk)
                        + (n_blk_dst * num_k_blks * d.k_blk * d.n_blk)
                        + (k_blk_dst * d.k_blk * d.n_blk);
                int src_new = src_ad, dst_new = dst_ad;

                if (k1 > 0) {
                    int a = 1;
                    bool is_n_tail
                            = (d.ntail > 0) && !(n_blk_src < num_n_blks - 1);
                    bool is_k_tail
                            = (d.ktail > 0) && !(k1 + k_blk_src < num_k_blks);

                    dyn_params_t k;
                    k.src = (int8_t *)wei + src_ad;
                    k.dst = (int8_t *)wei_reorder_buffer + dst_ad;
                    k.nn = &a;
                    k.nk = &k1;
                    k.is_k_tail = &is_k_tail;
                    k.is_n_tail = &is_n_tail;
                    (*reo_ker_b_)(&k);
                }

                if (n1 > 0) {
                    bool is_n_tail = (d.ntail > 0)
                            && !(n1 + n_blk_src < num_n_blks - 1);
                    bool is_k_tail = (d.ktail > 0);

                    if (k1 != num_k_blks) {
                        src_new = src_ad - d.k_blk * N * (num_k_blks - k1)
                                + d.n_blk;
                    } else {
                        src_new = src_ad + d.n_blk;
                    }
                    dst_new = dst_ad + d.k_blk * d.n_blk * k1;

                    dyn_params_t k;
                    k.src = (int8_t *)wei + src_new;
                    k.dst = (int8_t *)wei_reorder_buffer + dst_new;
                    k.nn = &n1;
                    k.nk = &num_k_blks;
                    k.is_k_tail = &is_k_tail;
                    k.is_n_tail = &is_n_tail;
                    (*reo_ker_b_)(&k);
                }
                if (k2 > 0) {
                    int a = 1;
                    bool is_n_tail = (d.ntail > 0)
                            && !(n1 + n_blk_src < num_n_blks - 2);
                    bool is_k_tail = false;
                    if (n1 < 1) {
                        src_new = src_ad - d.k_blk * N * (num_k_blks - k1)
                                + d.n_blk;
                        dst_new = dst_ad + d.k_blk * d.n_blk * k1;
                    } else {
                        src_new += n1 * d.n_blk;
                        dst_new += d.k_blk * d.n_blk * num_k_blks * n1;
                    }

                    dyn_params_t k;
                    k.src = (int8_t *)wei + src_new;
                    k.dst = (int8_t *)wei_reorder_buffer + dst_new;
                    k.nn = &a;
                    k.nk = &k2;
                    k.is_k_tail = &is_k_tail;
                    k.is_n_tail = &is_n_tail;
                    (*reo_ker_b_)(&k);
                }

                start += work_this_batch;
            }
        });
    };

    auto kernel_execute
            = [&](int idx, int na, int nb, int m_blk_adr, int n_blk_adr,
                      dim_t dst_byte_offset, int bias_addr, int scl_addr,
                      int zp_ptr_a_adr, int zp_ptr_b_adr, int zp_b_buf,
                      int m_row_start) {
        call_params_t p;
        p.na = &na;
        p.nb = &nb;
        p.src = (uint8_t *)src_reorder_buffer + m_blk_adr;
        p.wei = (uint8_t *)wei_reorder_buffer + n_blk_adr;
        p.dst = dst + dst_byte_offset;
        p.bias = (float *)bias + bias_addr;
        p.scales = src_scales;
        p.wei_scales = kernel_wei_scales + scl_addr;
        p.src_scales = b.is_per_m_scales ? src_scales + m_row_start : nullptr;
        p.dst_scales = dst_scales;
        p.src_zero_point = src_zero_points;
        if (b.is_zp_b_int8)
            p.wei_zero_point_buf = (const int8_t *)wei_zero_points;
        else
            p.wei_zero_point = wei_zero_points;
        p.dst_zero_point = dst_zero_points;
        p.M = M;
        p.N = N;
        p.K = K;
        p.zp_a_ptr = (float *)zp_ptr_a + zp_ptr_a_adr;
        p.zp_b_ptr = (float *)zp_ptr_b + zp_ptr_b_adr;
        (*int8_kernels_[idx])(&p);
    };

    auto kernel_execute_zp = [&]() {
        int num_a_blocks = div_up(M, b.m_blk);
        int num_b_blocks = div_up(N, (b.n_blk * b.ld_block));
        int ktail = (b.k_tail == 0) ? 0 : 1;
        int parallel_work = batch * num_a_blocks;
        int nt = parallel_work > 1 ? num_threads : 1;
        if (b.zp_type_b != jit_int8_broadcast_t::none) {
            parallel(nt, [&](const int ithr, const int nthr) {
                int start {0}, end {0};
                balance211(parallel_work, nt, ithr, start, end);
                int batch = start / num_a_blocks;
                int m_st = start % num_a_blocks;
                int m_ed = end - start + m_st;
                int mtail
                        = (m_ed == num_a_blocks) ? ((b.m_tail > 0) ? 1 : 0) : 0;
                int m_blk_adr = (batch
                                        * (num_a_blocks * b.m_blk
                                                * div_up(K, b.k_blk) * b.k_blk))
                        + m_st * b.m_blk * div_up(K, b.k_blk) * b.k_blk;
                int zp_ptr_b_adr
                        = (batch * (num_a_blocks * b.m_blk)) + m_st * b.m_blk;

                int idx = pd()->get_idx(1, 0, ktail, 0, b);
                if (idx < 0) {
                    assert(!"Requested int8 matmul kernel was not created.");
                    return;
                }
                int n_a = m_ed - m_st;
                if (mtail) n_a -= 1;
                kernel_execute(idx, n_a, 0, m_blk_adr, 0, 0, 0, 0, 0,
                        zp_ptr_b_adr, 0, 0);

                if (mtail) {
                    idx = pd()->get_idx(1, mtail, ktail, 0, b);
                    if (idx < 0) {
                        assert(!"Requested int8 matmul kernel was not "
                                "created.");
                        return;
                    }
                    m_blk_adr += n_a * b.m_blk * div_up(K, b.k_blk) * b.k_blk;
                    zp_ptr_b_adr += n_a * b.m_blk;
                    kernel_execute(idx, 1, 0, m_blk_adr, 0, 0, 0, 0, 0,
                            zp_ptr_b_adr, 0, 0);
                }
                start++;
            });
        }

        parallel_work = batch * num_b_blocks;
        nt = parallel_work > 1 ? num_threads : 1;
        if (b.zp_type_a != jit_int8_broadcast_t::none) {
            parallel(nt, [&](const int ithr, const int nthr) {
                int start {0}, end {0};
                balance211(parallel_work, nt, ithr, start, end);
                int batch = start / num_b_blocks;
                int n_st = start % num_b_blocks;
                int n_ed = n_st + end - start;
                int ntail
                        = (n_ed == num_b_blocks) ? ((b.n_tail > 0) ? 1 : 0) : 0;
                int n_blk_adr = (batch
                                        * (num_b_blocks * (b.n_blk * b.ld_block)
                                                * div_up(K, b.k_blk) * b.k_blk))
                        + n_st * (b.n_blk * b.ld_block) * div_up(K, b.k_blk)
                                * b.k_blk;
                int zp_ptr_a_adr
                        = (batch * num_b_blocks * (b.n_blk * b.ld_block))
                        + n_st * (b.n_blk * b.ld_block);

                int idx = pd()->get_idx(1, 0, ktail, 0, b);
                if (idx < 0) {
                    assert(!"Requested int8 matmul kernel was not created.");
                    return;
                }
                int n_b = n_ed - n_st;
                if (ntail == 1) n_b -= 1;

                kernel_execute(idx, 0, n_b, 0, n_blk_adr, 0, 0, 0, zp_ptr_a_adr,
                        0, 0, 0);

                if (ntail) {
                    idx = pd()->get_idx(1, 0, ktail, 1, b);
                    if (idx < 0) {
                        assert(!"Requested int8 matmul kernel was not "
                                "created.");
                        return;
                    }
                    n_blk_adr += n_b * (b.n_blk * b.ld_block)
                            * div_up(K, b.k_blk) * b.k_blk;
                    zp_ptr_a_adr += n_b * (b.n_blk * b.ld_block);
                    kernel_execute(idx, 0, 1, 0, n_blk_adr, 0, 0, 0,
                            zp_ptr_a_adr, 0, 0, 0);
                }

                start++;
            });
        }
    };

    if (b.b_reo) reorder_b();

    reorder_a();

    if (b.zp_type_a != jit_int8_broadcast_t::none
            || b.zp_type_b != jit_int8_broadcast_t::none)
        kernel_execute_zp();

    int m_block1, n_block1, m_block1_rs, n_block1_rs, num_a_blocks_act,
            num_b_blocks_act, num_a_blocks, num_b_blocks;
    int m_block_sz = pd()->m_block_sz;
    int n_block_sz = pd()->n_block_sz;

    m_block1 = div_up(m_block_sz, b.m_blk);
    n_block1 = div_up(n_block_sz, (b.n_blk * b.ld_block));
    m_block1_rs = div_up(M % m_block_sz, b.m_blk);
    n_block1_rs = div_up(N % n_block_sz, (b.n_blk * b.ld_block));

    num_a_blocks_act = div_up(M, b.m_blk);
    num_b_blocks_act = div_up(N, (b.n_blk * b.ld_block));
    num_a_blocks = div_up(M, m_block_sz);
    num_b_blocks = div_up(N, n_block_sz);
    int ktail = (b.k_tail == 0) ? 0 : 1;
    int parallel_work = pd()->mm_parallel_work;
    int nt = parallel_work > 1 ? num_threads : 1;
    parallel(nt, [&](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(parallel_work, nt, ithr, start, end);
        while (start < end) {
            int batch = start / (num_a_blocks * num_b_blocks);
            int batch_start = start % (num_a_blocks * num_b_blocks);
            int m_block = batch_start % num_a_blocks;
            int n_block = batch_start / num_a_blocks;
            int mtail
                    = (m_block1_rs != 0 && m_block == num_a_blocks - 1) ? 1 : 0;
            int ntail
                    = (n_block1_rs != 0 && n_block == num_b_blocks - 1) ? 1 : 0;
            dim_t dst_byte_offset
                    = ((batch * M * N) + m_block * b.m_blk * m_block1 * N
                              + n_block * (b.n_blk * b.ld_block) * n_block1)
                    * b.dst_dt_sz;
            int m_blk_adr = (batch
                                    * (num_a_blocks_act * b.m_blk
                                            * div_up(K, b.k_blk) * b.k_blk))
                    + m_block * b.m_blk * m_block1 * div_up(K, b.k_blk)
                            * b.k_blk;
            int n_blk_adr = (batch
                                    * (num_b_blocks_act * (b.n_blk * b.ld_block)
                                            * div_up(K, b.k_blk) * b.k_blk))
                    + n_block * (b.n_blk * b.ld_block) * n_block1
                            * div_up(K, b.k_blk) * b.k_blk;
            int zp_ptr_a_adr
                    = (batch * (num_b_blocks_act * (b.n_blk * b.ld_block)))
                    + n_block * (b.n_blk * b.ld_block) * n_block1;
            int zp_ptr_b_adr = (batch * (num_a_blocks_act * b.m_blk))
                    + m_block * b.m_blk * m_block1;
            int bias_addr = n_block * (b.n_blk * b.ld_block) * n_block1;
            int zp_b_buf = n_block * (b.n_blk * b.ld_block) * n_block1;
            int scl_addr = (b.is_oc_scales)
                    ? (n_block * (b.n_blk * b.ld_block) * n_block1)
                    : 0;
            int idx = pd()->get_idx(0, 0, ktail, 0, b);
            if (idx < 0) {
                assert(!"Requested int8 matmul kernel was not created.");
                return;
            }
            int n_a = m_block1, n_b = n_block1;
            n_a = (mtail) ? ((b.m_tail) ? m_block1_rs - 1 : m_block1_rs)
                          : m_block1;
            n_b = (ntail) ? ((b.n_tail) ? n_block1_rs - 1 : n_block1_rs)
                          : n_block1;

            if (n_a > 0 && n_b > 0) {

                kernel_execute(idx, n_a, n_b, m_blk_adr, n_blk_adr,
                        dst_byte_offset, bias_addr, scl_addr, zp_ptr_a_adr,
                        zp_ptr_b_adr, zp_b_buf, m_block * m_block_sz);
            }

            if (mtail && b.m_tail > 0 && n_b > 0) {
                dim_t new_dst_byte_offset
                        = dst_byte_offset + b.m_blk * n_a * N * b.dst_dt_sz;
                int new_m_blk_adr = m_blk_adr
                        + b.m_blk * n_a * div_up(K, b.k_blk) * b.k_blk;
                int new_zp_ptr_b_adr = zp_ptr_b_adr + b.m_blk * n_a;
                idx = pd()->get_idx(0, 1, ktail, 0, b);
                if (idx < 0) {
                    assert(!"Requested int8 matmul kernel was not created.");
                    return;
                }
                int na = 1;
                kernel_execute(idx, na, n_b, new_m_blk_adr, n_blk_adr,
                        new_dst_byte_offset, bias_addr, scl_addr, zp_ptr_a_adr,
                        new_zp_ptr_b_adr, zp_b_buf,
                        m_block * m_block_sz + n_a * b.m_blk);
            }

            if (ntail && b.n_tail > 0 && n_a > 0) {
                dim_t new_dst_byte_offset = dst_byte_offset
                        + (b.n_blk * b.ld_block) * n_b * b.dst_dt_sz;
                int new_n_blk_adr = n_blk_adr
                        + (b.n_blk * b.ld_block) * n_b * div_up(K, b.k_blk)
                                * b.k_blk;
                int new_zp_b_buf = zp_b_buf + (b.n_blk * b.ld_block) * n_b;
                int new_zp_ptr_a_adr
                        = zp_ptr_a_adr + (b.n_blk * b.ld_block) * n_b;
                int new_bias_addr = bias_addr + (b.n_blk * b.ld_block) * n_b;
                int new_scl_addr = scl_addr
                        + ((b.is_oc_scales) ? ((b.n_blk * b.ld_block) * n_b)
                                            : 0);
                idx = pd()->get_idx(0, 0, ktail, 1, b);
                if (idx < 0) {
                    assert(!"Requested int8 matmul kernel was not created.");
                    return;
                }
                int nb = 1;

                kernel_execute(idx, n_a, nb, m_blk_adr, new_n_blk_adr,
                        new_dst_byte_offset, new_bias_addr, new_scl_addr,
                        new_zp_ptr_a_adr, zp_ptr_b_adr, new_zp_b_buf,
                        m_block * m_block_sz);
            }

            if (mtail && b.m_tail > 0 && ntail && b.n_tail > 0) {
                dim_t new_dst_byte_offset = dst_byte_offset
                        + ((b.n_blk * b.ld_block) * n_b + b.m_blk * n_a * N)
                                * b.dst_dt_sz;
                int new_m_blk_adr = m_blk_adr
                        + b.m_blk * n_a * div_up(K, b.k_blk) * b.k_blk;
                int new_n_blk_adr = n_blk_adr
                        + (b.n_blk * b.ld_block) * n_b * div_up(K, b.k_blk)
                                * b.k_blk;
                int new_zp_b_buf = zp_b_buf + (b.n_blk * b.ld_block) * n_b;
                int new_zp_ptr_a_adr
                        = zp_ptr_a_adr + (b.n_blk * b.ld_block) * n_b;
                int new_zp_ptr_b_adr = zp_ptr_b_adr + b.m_blk * n_a;
                int new_bias_addr = bias_addr + (b.n_blk * b.ld_block) * n_b;
                int new_scl_addr = scl_addr
                        + ((b.is_oc_scales) ? ((b.n_blk * b.ld_block) * n_b)
                                            : 0);
                idx = pd()->get_idx(0, 1, ktail, 1, b);
                if (idx < 0) {
                    assert(!"Requested int8 matmul kernel was not created.");
                    return;
                }
                int nb = 1, na = 1;
                kernel_execute(idx, na, nb, new_m_blk_adr, new_n_blk_adr,
                        new_dst_byte_offset, new_bias_addr, new_scl_addr,
                        new_zp_ptr_a_adr, new_zp_ptr_b_adr, new_zp_b_buf,
                        m_block * m_block_sz + n_a * b.m_blk);
            }
            start++;
        }
    });

    return status::success;
}

template struct jit_int8_matmul_t<sve>;

} // namespace matmul
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
