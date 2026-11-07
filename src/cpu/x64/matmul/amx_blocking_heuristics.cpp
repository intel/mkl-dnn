/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "cpu/x64/matmul/amx_blocking_heuristics.hpp"
#include "cpu/matmul/gemm_based_common.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::cpu::matmul;

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace data_type;
using namespace format_tag;
void matmul_amx_blocking_params_t::update_configuration(
        brgemm_matmul_conf_t &bgmmc) const {
    bgmmc.nthr_k = nthr_k_;
    bgmmc.nthr_m = nthr_m_;
    bgmmc.nthr_n = nthr_n_;
    bgmmc.nthr_b = nthr_b_;
    bgmmc.nthr = nthr_;
    bgmmc.M_blk = m_blk_;
    bgmmc.M_chunk_size = static_cast<int>(m_chunk_size_);
    bgmmc.N_blk = n_blk_;
    bgmmc.N_chunk_size = static_cast<int>(n_chunk_size_);

    bgmmc.K_blk = k_blk_;
    bgmmc.K_chunk_size = static_cast<int>(k_chunk_size_);
    bgmmc.brgemm_batch_size = static_cast<int>(brgemm_batch_size_);

    bgmmc.use_buffer_c = need_buf_c_;
    bgmmc.use_buffer_a = need_buf_a_;
    bgmmc.extendable_k = extendable_k_;
    bgmmc.LDA = current_lda_;

    bgmmc.is_a_nt = is_a_nt_;
    bgmmc.is_b_nt = is_b_nt_;
    bgmmc.set_nt = set_nt_;
    bgmmc.need_prefetch_a = need_prefetch_a_;
    bgmmc.need_prefetch_b = need_prefetch_b_;
    bgmmc.use_fused_copy_a = use_fused_copy_a_;
    bgmmc.is_macro_heuristics
            = dynamic_cast<const matmul_amx_blocking_params_macro_t *>(this)
            != nullptr;
}

dim_t matmul_amx_blocking_params_t::get_actual_lda() const {
    if (!need_buf_a_)
        return treat_A_as_plain ? K : A_strides[1 - transposed_A] / a_dt_sz;

    constexpr int bytes_in_cacheline = 64;
    const dim_t elems_in_cacheline = bytes_in_cacheline / a_dt_sz;
    dim_t lda = rnd_up(k_blk_, elems_in_cacheline);
    const bool is_big_2_pow = lda >= 512 && math::is_pow2(lda);
    if (is_big_2_pow) lda += elems_in_cacheline;
    return lda;
}

bool matmul_amx_blocking_params_t::is_buffer_c_required() const {
    if (nthr_k_ > 1 && K > k_chunk_elems_) return true;

    return ((acc_dt != dst_dt || with_sum)
            && (K > k_chunk_elems_ || K % k_blk_ > 0));
}

dim_t matmul_amx_blocking_params_t::effective_c_buf_dt_sz(
        int nthr_k_cand) const {
    if (!is_c_buf_dst_dt) return acc_dt_sz;
    // Bypass (nthr_k<=1) has no scratch C buffer -> return 0 for footprint.
    // D RMW traffic is modeled in bw() separately to avoid perturbing
    // N_blk selection for single-K-chunk shapes.
    if (nthr_k_cand <= 1 && !with_sum) return 0;
    return c_dt_sz;
}

// The narrow-dt (bf16/f16) C discount only applies where the C buffer
// actually disappears: nthr_k<=1 with several K-chunks. Elsewhere the
// saving is dwarfed by the reduction and the M/N parallelism given up,
// and letting it skew the score can cost an order of magnitude.
bool matmul_amx_blocking_params_t::narrow_c_scoring_active() const {
    const bool multiple_k_chunks = k_chunk_elems_ > 0 && K > k_chunk_elems_;
    return is_c_buf_dst_dt && nthr_k_ <= 1 && multiple_k_chunks;
}

size_t matmul_amx_blocking_params_t::L2_threshold() {
    return 3 * platform::get_per_core_cache_size(2) / 4;
}

size_t matmul_amx_blocking_params_t::L2_ways_threshold() {
    return 3 * platform::get_num_ways_in_cache(2) / 4;
}

size_t matmul_amx_blocking_params_t::L1_threshold() {
    return 5 * platform::get_per_core_cache_size(1) / 6;
}

bool matmul_amx_blocking_params_macro_t::is_supported(
        const brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils) {

    bool a_dt_ok
            = one_of(bgmmc.orig_src_dt, dnnl_s8, dnnl_u8, dnnl_bf16, dnnl_f16);
    bool b_dt_ok
            = one_of(bgmmc.orig_wei_dt, dnnl_s8, dnnl_u8, dnnl_bf16, dnnl_f16)
            || bgmmc.is_xf16_fp8;

    bool a_tag_ok = bgmmc.src_tag == dnnl_format_tag_any
            || bm_conf_utils.check_is_plain(bgmmc.src_tag);

    bool b_tag_ok = IMPLICATION(
            bm_conf_utils.check_b_layout_blocked_by_n(bgmmc.wei_tag),
            bm_conf_utils.check_b_layout_blocked_32_by_n(bgmmc.wei_tag));

    return bgmmc.orig_src_dt == bgmmc.src_dt
            && (bgmmc.orig_wei_dt == bgmmc.wei_dt || bgmmc.is_xf16_fp8)
            && bgmmc.is_amx && !bgmmc.is_runtime_N && !bgmmc.is_runtime_M
            && a_dt_ok && a_tag_ok && b_dt_ok && b_tag_ok
            && (bgmmc.reduce_kind == matmul_reduce_kind::undef)
            && !bgmmc.packed_sparse_weights;
}

bool matmul_amx_blocking_params_macro_t::divs_are_acceptable() const {
    bool unacceptable_m_div
            = rnd_up(m_per_thread, m_tmul) < min_mn_elem && nthr_m_ > 1;
    bool unacceptable_k_div = k_per_thread < min_k_elem && nthr_k_ > 1;
    bool unacceptable_n_div;
    if (nthr_k_ == 1 && k_per_thread < k_threshold_write_bound_layer_elem) {
        // The layer is write-bound (small K) and no reduction (C becomes non-consecutive)
        unacceptable_n_div = rnd_up(n_per_thread, n_tmul)
                        < min_n_dim_write_bound_layer_elem
                && nthr_n_ > 1;
    } else {
        unacceptable_n_div
                = rnd_up(n_per_thread, n_tmul) < min_mn_elem && nthr_n_ > 1;
    }

    bool unacceptable_b_div = nthr_b_ > batch;
    if (nthr_k_ > 1 && calculate_compensations_in_copy_routines()) {
        // If we have to calculate compensations in copy routines,
        // we cannot split K dimension
        unacceptable_k_div = true;
    }

    return !unacceptable_m_div && !unacceptable_k_div && !unacceptable_n_div
            && !unacceptable_b_div;
}

size_t determine_tmul_size(size_t num_elements, int full_tile_size) {
    size_t tmul_tiles = div_up(num_elements, full_tile_size);
    size_t tmul_size = div_up(num_elements, tmul_tiles);
    return tmul_size;
}

/*
 * When one of the dimensions (M, N, or K) is significantly smaller than the others,
 * one of the matrices will be much larger in comparison:
 *    1. M is small:  Matrix B >> A, C
 *    2. N is small:  Matrix A >> B, C
 *    3. K is small:  Matrix C >> A, B
 *
 * In such cases, the layer becomes memory-bound, and the key performance factor
 * is efficient reading of the largest matrix.
 *
 * To optimize this, we divide the workload across cores so that each core reads
 * a consecutive chunk of the large matrix. Also ensure that each core follows
 * a sequential memory access pattern, improving bandwidth efficiency:
 *    1. M is small:  Matrix B layout is [N, K, 16K, 32N, (vnni)K] → divide along N and read in K direction before N
 *    2. N is small:  Matrix A layout is [M, K] → divide along M and read in K direction before M
 *    3. K is small:  Matrix C layout is [M, N] → divide along M and read in N direction before M (horizontal access)
 */
bool matmul_amx_blocking_params_macro_t::maybe_small_dims_heuristics(
        const brgemm_matmul_conf_t &bgmmc,
        matmul_amx_blocking_params_macro_t &best_blocking) {

    auto set_common = [&]() {
        best_blocking.k_blk_ = bgmmc.K;
        best_blocking.k_chunk_size_ = 1;

        best_blocking.brgemm_batch_size_ = 1;
        best_blocking.need_buf_c_ = false;
        best_blocking.need_buf_a_ = best_blocking.use_buffer_a;

        // extendable_k requires A to be contiguous along K (LDA == K).
        best_blocking.extendable_k_ = bgmmc.K % best_blocking.wei_k_blk != 0
                && !best_blocking.skip_extendable_k()
                && best_blocking.get_actual_lda() == bgmmc.K;

        best_blocking.is_a_nt_ = true;
        best_blocking.is_b_nt = true;
        best_blocking.set_nt_ = true;
        best_blocking.need_prefetch_a_ = false;
        best_blocking.need_prefetch_b_ = false;
        best_blocking.use_fused_copy_a_ = false;
        best_blocking.efficiency_score_ = 1;
        best_blocking.current_lda_ = best_blocking.get_actual_lda();
    };

    const float core_utilization_threshold = 0.75f;

    const auto b_is_not_flat
            = IMPLICATION(bgmmc.use_buffer_b, bgmmc.transposed_B);

    if (bgmmc.M <= 32 && b_is_not_flat && bgmmc.batch == 1) {

        best_blocking.n_decomposition
                = nstl::min(bgmmc.N, (dim_t)bgmmc.wei_n_blk);
        best_blocking.m_decomposition = bgmmc.M;
        dim_t n_per_core = div_up(bgmmc.N, bgmmc.nthr);
        n_per_core = rnd_up(n_per_core, best_blocking.n_decomposition);
        best_blocking.set_core_divs(
                1, 1, 1, static_cast<int>(div_up(bgmmc.N, n_per_core)));

        if (best_blocking.nthr_n_
                < core_utilization_threshold * best_blocking.nthr) {
            return false;
        }
        best_blocking.m_blk_ = bgmmc.M;
        best_blocking.m_chunk_size_ = 1;
        best_blocking.n_blk_ = best_blocking.n_decomposition;
        best_blocking.n_chunk_size_
                = n_per_core / best_blocking.n_decomposition;

        if (bgmmc.use_buffer_b) {
            const size_t brg_b_size = best_blocking.n_decomposition * bgmmc.K;
            const size_t brg_a_size = best_blocking.M * bgmmc.K;
            if (brg_a_size > L2_threshold()) { return false; }
            best_blocking.k_chunk_size_
                    = div_up(2 * brg_b_size, L2_threshold() - brg_a_size);
            best_blocking.K_blk = rnd_up(
                    div_up(best_blocking.K, best_blocking.k_chunk_size_),
                    best_blocking.wei_k_blk);
        } else {
            best_blocking.k_blk_ = bgmmc.K;
            best_blocking.k_chunk_size_ = 1;
        }

        set_common();
        return true;

    } else if (bgmmc.K <= best_blocking.wei_k_blk && bgmmc.batch == 1) {

        dim_t m_per_core = div_up(bgmmc.M, bgmmc.nthr);
        if (bgmmc.is_ace) {
            const dim_t ace_m_dec = ace_m_decomposition;
            m_per_core = nstl::max(
                    m_per_core, nstl::min(ace_m_dec, (dim_t)bgmmc.M));
        }
        best_blocking.set_core_divs(
                1, static_cast<int>(div_up(bgmmc.M, m_per_core)), 1, 1);
        best_blocking.set_tmul_sizes();
        best_blocking.set_decomposition();

        if (best_blocking.nthr_m_
                < core_utilization_threshold * best_blocking.nthr) {
            return false;
        }
        best_blocking.m_blk_ = best_blocking.m_decomposition;
        best_blocking.m_chunk_size_
                = div_up(m_per_core, best_blocking.m_decomposition);
        best_blocking.n_blk_ = bgmmc.N;
        best_blocking.n_chunk_size_ = 1;

        if (bgmmc.use_buffer_b) {
            const size_t brg_b_size = best_blocking.N * bgmmc.K;
            const size_t brg_b_size_tr
                    = best_blocking.N * rnd_up(bgmmc.K, bgmmc.wei_k_blk);
            const size_t brg_d_size
                    = best_blocking.N * best_blocking.m_decomposition;
            best_blocking.n_chunk_size_ = div_up(
                    brg_b_size + brg_b_size_tr + brg_d_size, L2_threshold());
            best_blocking.n_blk_ = rnd_up(
                    div_up(best_blocking.N, best_blocking.n_chunk_size_),
                    best_blocking.wei_n_blk);
        } else {
            best_blocking.n_blk_ = bgmmc.N;
            best_blocking.n_chunk_size_ = 1;
        }
        set_common();
        return true;

    } else if (bgmmc.N <= 32 && bgmmc.batch == 1) {

        const dim_t m_per_core = div_up(bgmmc.M, bgmmc.nthr);

        best_blocking.m_per_thread = m_per_core;
        // in this case 2 full are preferable
        // On ACE the two tiles are a fixed ace_m_decomposition rows; there is
        // no reduced-row form of the accumulator to fall back to.
        if (bgmmc.is_ace) {
            const dim_t ace_m_dec = ace_m_decomposition;
            best_blocking.m_decomposition
                    = nstl::min(ace_m_dec, (dim_t)bgmmc.M);
        } else {
            best_blocking.m_decomposition
                    = determine_tmul_size(best_blocking.m_per_thread, 2 * 16);
        }
        best_blocking.n_tmul = 16; // B blocked layout is a multiply of 16
        best_blocking.n_decomposition = 2 * best_blocking.n_tmul;
        best_blocking.k_tmul = nstl::min(
                (size_t)best_blocking.wei_k_blk, (size_t)best_blocking.K);

        const size_t m_per_core_actual = rnd_up(
                best_blocking.m_per_thread, best_blocking.m_decomposition);
        best_blocking.set_core_divs(
                1, static_cast<int>(div_up(bgmmc.M, m_per_core_actual)), 1, 1);

        if (best_blocking.nthr_m_
                < core_utilization_threshold * best_blocking.nthr) {
            return false;
        }

        best_blocking.m_blk_ = best_blocking.m_decomposition;
        best_blocking.m_chunk_size_
                = div_up(m_per_core, best_blocking.m_decomposition);
        best_blocking.n_blk_ = bgmmc.N;
        best_blocking.n_chunk_size_ = 1;

        const size_t brg_b_size = best_blocking.n_decomposition * bgmmc.K;
        const size_t brg_a_size = best_blocking.m_decomposition * bgmmc.K;
        best_blocking.k_chunk_size_
                = div_up(2 * brg_b_size + brg_a_size, L2_threshold());
        best_blocking.k_blk_
                = rnd_up(div_up(best_blocking.K, best_blocking.k_chunk_size_),
                        best_blocking.wei_k_blk);
        set_common();
        return true;
    } else {
        return false;
    }
}

bool matmul_amx_blocking_params_macro_t::find_best_blocking(
        const brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        matmul_amx_blocking_params_macro_t &best_blocking) {

    if (!matmul_amx_blocking_params_macro_t::is_supported(
                bgmmc, bm_conf_utils)) {
        return false;
    }

    best_blocking = matmul_amx_blocking_params_macro_t(bgmmc);

    matmul_amx_blocking_params_macro_t current_blocking(bgmmc);

    if (maybe_small_dims_heuristics(bgmmc, best_blocking)) { return true; }

    for (int nthr_to_check = bgmmc.nthr; nthr_to_check > 0; nthr_to_check--) {
        current_blocking.nthr_ = nthr_to_check;

        for (int b_div = 1; b_div <= current_blocking.nthr_; ++b_div) {
            if (current_blocking.nthr_ % b_div != 0) continue;
            for (int m_div = 1; m_div <= current_blocking.nthr_ / b_div;
                    ++m_div) {
                if ((current_blocking.nthr_ / b_div) % m_div != 0) continue;
                for (int k_div = 1;
                        k_div <= (current_blocking.nthr_ / b_div) / m_div;
                        ++k_div) {
                    if (((current_blocking.nthr_ / b_div) / m_div) % k_div != 0)
                        continue;
                    int n_div = ((current_blocking.nthr_ / b_div) / m_div)
                            / k_div;
                    current_blocking.set_core_divs(b_div, m_div, k_div, n_div);
                    current_blocking.set_tmul_sizes();
                    current_blocking.set_decomposition();
                    if (current_blocking.divs_are_acceptable()
                            && current_blocking.set_blocking_parameters()) {
                        if (current_blocking > best_blocking) {
                            best_blocking = current_blocking;
                        }
                    }
                }
            }
        }
    }

    if (best_blocking.use_buffer_b && !best_blocking.transposed_B
            && !best_blocking.is_horizontal) {
        current_blocking.nthr_ = best_blocking.nthr_b_ * best_blocking.nthr_m_
                * best_blocking.nthr_n_ * best_blocking.nthr_k_;
        current_blocking.set_core_divs(best_blocking.nthr_b_,
                best_blocking.nthr_m_, best_blocking.nthr_k_,
                best_blocking.nthr_n_);
        if (current_blocking.set_blocking_parameters(true)
                && current_blocking > best_blocking) {
            best_blocking = current_blocking;
        }
        if (current_blocking.set_blocking_parameters(true, true)
                && current_blocking > best_blocking) {
            best_blocking = current_blocking;
        }
    }
    return true;
}

bool matmul_amx_blocking_params_macro_t::b_transform_fits_in_l2() const {
    bool b_transform_in_l2;
    auto max_l2_usage = l2_matrix_usage(
            k_chunk_size_, n_blk_, k_blk_, true, true); /* B transform size */
    b_transform_in_l2 = max_l2_usage < L2_threshold();
    return b_transform_in_l2;
}

void matmul_amx_blocking_params_macro_t::init_copy_characteristics(
        layer_perf_characteristics_t &layer_perf_characteristics) const {
    layer_perf_characteristics.strip1_b_tranform_h
            = is_horizontal && use_buffer_b;
    layer_perf_characteristics.strips_b_tranform_v
            = !is_horizontal && use_buffer_b;
    layer_perf_characteristics.strip1_b_in_mlc_h
            = layer_perf_characteristics.strip1_b_tranform_h
            && b_transform_fits_in_l2();

    layer_perf_characteristics.strips_a_read_h = is_horizontal && use_buffer_a;
    layer_perf_characteristics.strip1_a_read_v = !is_horizontal && use_buffer_a;
}

void matmul_amx_blocking_params_macro_t::calculate_layer_sizes(
        layer_perf_characteristics_t &layer_perf_characteristics) const {

    int macs_per_cycle_base = 1024;
    int max_k_tmul = 64;
    int max_n_tmul = 16;
    // Reducing k-tmul or n-tmul does not shorten the cycles.
    // However, reducing mtmul reduces the number of cycles required to execute a single tmul instruction.
    layer_perf_characteristics.num_cycles_per_tmul
            = m_tmul * max_k_tmul * max_n_tmul / macs_per_cycle_base;

    layer_perf_characteristics.a_size
            = m_per_thread * k_per_thread * gemm_dt_sz;
    layer_perf_characteristics.b_size
            = n_per_thread * k_per_thread * gemm_dt_sz;
    layer_perf_characteristics.d_size = m_per_thread * n_per_thread * c_dt_sz;

    if (is_horizontal) {
        // Amount of C/D bytes that are written per core
        size_t strip_dst_size = m_decomposition * n_per_thread
                * ((nthr_k_ == 1 || narrow_c_scoring_active()) ? c_dt_sz
                                                               : acc_dt_sz);
        // Amount of compute
        layer_perf_characteristics.num_tmuls_per_strip
                = static_cast<float>(m_decomposition * k_per_thread
                        * n_per_thread / (m_tmul * k_tmul * n_tmul));
        // Amount of strips in the execution
        layer_perf_characteristics.num_strip
                = static_cast<float>(div_up(m_per_thread, m_decomposition));
        // B is blocked to the L2 in horizontal traversal, its loads are NT
        layer_perf_characteristics.nt_mat_l1_miss
                = static_cast<float>(layer_perf_characteristics.b_size);
        // Number of times A is reused from L1 in a strip
        layer_perf_characteristics.l1_reuse
                = static_cast<float>(div_up(n_blk_, n_decomposition));

        // In horizontal multiple cores load the same B to L2
        layer_perf_characteristics.strip_1_size_shared
                = static_cast<float>(layer_perf_characteristics.b_size);
        // In strip 1 there is no sharing of A since there are no prefetches
        size_t strip_1_size_private_a
                = m_decomposition * k_per_thread * gemm_dt_sz;
        layer_perf_characteristics.strip_1_size_private
                = static_cast<float>(strip_1_size_private_a + strip_dst_size);
        // The cores that share B
        layer_perf_characteristics.strip_1_share_coef
                = static_cast<float>(nthr_m_);

        // In the mid strips B is reused from L2 and
        // A is prefetched by multiple cores.
        layer_perf_characteristics.strip_mid_size_shared
                = static_cast<float>(m_decomposition * k_per_thread
                        * gemm_dt_sz); // A size per strip
        // C is private to a core, since each core writes to a distinct buffer
        layer_perf_characteristics.strip_mid_size_private
                = static_cast<float>(strip_dst_size);
        // share_coeff - the cores that share A
        layer_perf_characteristics.strip_mid_share_coef
                = static_cast<float>(std::max(1, nthr_n_));

        // Calculate the number of cache lines to be processed in AVX postops
        layer_perf_characteristics.num_postop_cache_lines = static_cast<float>(
                m_decomposition * div_up(n_per_thread, n_tmul));

    } else {
        // Amount of C/D bytes that are written per core
        size_t strip_dst_size = n_decomposition * m_per_thread
                * ((nthr_k_ == 1 || narrow_c_scoring_active()) ? c_dt_sz
                                                               : acc_dt_sz);
        // Amount of compute
        layer_perf_characteristics.num_tmuls_per_strip
                = static_cast<float>(n_decomposition * k_per_thread
                        * m_per_thread / (m_tmul * k_tmul * n_tmul));
        // Amount of strips in the execution
        layer_perf_characteristics.num_strip
                = static_cast<float>(div_up(n_per_thread, n_decomposition));
        // A is blocked to the L2 in vertical traversal, its loads are NT
        layer_perf_characteristics.nt_mat_l1_miss
                = static_cast<float>(layer_perf_characteristics.a_size);
        // Number of times B is reused from L1 in a strip
        layer_perf_characteristics.l1_reuse
                = static_cast<float>(div_up(m_blk_, m_decomposition));

        // In vertical multiple cores load the same A to L2
        layer_perf_characteristics.strip_1_size_shared
                = static_cast<float>(layer_perf_characteristics.a_size);
        // In strip 1 there is no sharing of B since there are no prefetches
        size_t strip_1_size_private_b
                = n_decomposition * k_per_thread * gemm_dt_sz;
        layer_perf_characteristics.strip_1_size_private
                = static_cast<float>(strip_1_size_private_b + strip_dst_size);
        // The cores that share A
        layer_perf_characteristics.strip_1_share_coef
                = static_cast<float>(nthr_n_);

        // In the mid strips A is reused from L2 and
        // B is prefetched by multiple cores.
        layer_perf_characteristics.strip_mid_size_shared
                = static_cast<float>(n_decomposition * k_per_thread
                        * gemm_dt_sz); // B size per strip
        // C is private to a core, since each core writes to a distinct buffer
        layer_perf_characteristics.strip_mid_size_private
                = static_cast<float>(strip_dst_size);
        // share_coeff - the cores that share B
        layer_perf_characteristics.strip_mid_share_coef
                = static_cast<float>(std::max(1, nthr_m_));

        // Calculate the number of cache lines to be processed in AVX postops
        layer_perf_characteristics.num_postop_cache_lines = static_cast<float>(
                m_per_thread * div_up(n_decomposition, n_tmul));
    }
}

float matmul_amx_blocking_params_macro_t::calculate_strip_mid_cycles(
        layer_perf_characteristics_t &layer_perf_characteristics,
        bw_map_t &bw_interpolator) const {
    // There are 2 L1 misses for the L1 matrix:
    //   1. For the prefetch to the L2 (==L1 miss)
    //   2. For the read from L2
    float temporal_matrix_l1_miss
            = layer_perf_characteristics.strip_mid_size_shared * 2;
    float temporal_matrix_l1_hit
            = layer_perf_characteristics.strip_mid_size_shared
            * (layer_perf_characteristics.l1_reuse - 1);

    float c_elem_per_strip = static_cast<float>(m_blk_ * n_blk_);

    // C post write miss in bytes = m_blk_ * (#n_decompositions in BRGEMM) * (#cache lines per n_decomposition) * 64
    float c_post_write_miss
            = static_cast<float>(m_blk_ * div_up(n_blk_, n_decomposition)
                    * rnd_up(n_decomposition * c_dt_sz, 64));
    // C post write total in bytes = m_blk_ * (#n_decompositions in BRGEMM) * (#writes per n_decomposition) * 64
    float c_post_write_total
            = static_cast<float>(m_blk_ * div_up(n_blk_, n_decomposition)
                    * div_up(n_decomposition, 16) * 64);
    float c_post_write_hit = c_post_write_total - c_post_write_miss;

    float c_post_read_c_tmp = c_elem_per_strip
            * (narrow_c_scoring_active() ? c_dt_sz : acc_dt_sz);

    float c_tmp_l1_cycles;
    if (k_blk_ == K) {
        // tile stores into wsp (hot in l1)
        c_tmp_l1_cycles = acc_dt_sz * c_elem_per_strip * k_chunk_size_
                / bw_interpolator.l1_store_hit_bw;
    } else {
        if (narrow_c_scoring_active()) {
            // tile stores into wsp (hot in l1)
            c_tmp_l1_cycles = acc_dt_sz * c_elem_per_strip * k_chunk_size_
                    / bw_interpolator.l1_store_hit_bw;
            // load from wsp to zmm
            c_tmp_l1_cycles += acc_dt_sz * c_elem_per_strip * k_chunk_size_
                    / bw_interpolator.l1_load_hit_bw;

            assert(k_chunk_size_ > 0);
            // load previous partial results that are stored in c_dt_sz
            c_tmp_l1_cycles += c_dt_sz * c_elem_per_strip * (k_chunk_size_ - 1)
                    / bw_interpolator.l1_load_miss_bw;
            // store new partial results in c_dt_sz
            c_tmp_l1_cycles += c_dt_sz * c_elem_per_strip * (k_chunk_size_ - 1)
                    / bw_interpolator.l1_store_hit_bw;
        } else {
            // tile loads from previous partial results
            c_tmp_l1_cycles = acc_dt_sz * c_elem_per_strip * k_chunk_size_
                    / bw_interpolator.l1_load_miss_bw;
            // tile stores of new partial results
            c_tmp_l1_cycles += acc_dt_sz * c_elem_per_strip * k_chunk_size_
                    / bw_interpolator.l1_store_hit_bw;
        }
    }

    float c_l1_cycles = c_post_write_miss / bw_interpolator.l1_store_miss_bw
            + c_post_write_hit / bw_interpolator.l1_store_hit_bw
            + c_post_read_c_tmp / bw_interpolator.l1_load_hit_bw
            + c_tmp_l1_cycles;

    float l1_cycles = temporal_matrix_l1_miss / bw_interpolator.l1_load_miss_bw
            + temporal_matrix_l1_hit / bw_interpolator.l1_load_hit_bw
            + layer_perf_characteristics.nt_mat_l1_miss
                    / bw_interpolator.l1_load_miss_bw
            + c_l1_cycles;

    float strip_mid_dram;
    float strip_mid_llc;
    if (layer_perf_characteristics.strips_b_tranform_v
            || layer_perf_characteristics.strips_a_read_h) {
        strip_mid_dram = layer_perf_characteristics.strip_mid_size_private
                / bw_interpolator.get_bw(1);
        strip_mid_llc = layer_perf_characteristics.strip_mid_size_private
                / bw_interpolator.llc_bw;
    } else {
        strip_mid_dram = layer_perf_characteristics.strip_mid_size_shared
                        / bw_interpolator.get_bw(
                                static_cast<int>(layer_perf_characteristics
                                                         .strip_mid_share_coef))
                + layer_perf_characteristics.strip_mid_size_private
                        / bw_interpolator.get_bw(1);

        strip_mid_llc
                = (layer_perf_characteristics.strip_mid_size_private
                          + layer_perf_characteristics.strip_mid_size_shared)
                / bw_interpolator.llc_bw;
    }

    float strip_tmul = layer_perf_characteristics.num_tmuls_per_strip
            * layer_perf_characteristics.num_cycles_per_tmul;
    float avx_insts_per_cache_line
            = calculate_avx_insts_per_cache_line(layer_perf_characteristics);
    float strip_avx = avx_insts_per_cache_line
            * layer_perf_characteristics.num_postop_cache_lines / avx_ipc;
    float strip_mid_cycles = std::max(
            {strip_mid_dram, strip_mid_llc, l1_cycles, strip_tmul, strip_avx});

    return strip_mid_cycles;
}

float matmul_amx_blocking_params_macro_t::calculate_strip_1_cycles(
        layer_perf_characteristics_t &layer_perf_characteristics,
        bw_map_t &bw_interpolator, float strip_mid_cycles) const {
    float strip_1_cycles;
    if (!layer_perf_characteristics.strip1_b_tranform_h
            && !layer_perf_characteristics.strip1_a_read_v) {
        // default for vertical and horizontal without b transform
        strip_1_cycles = layer_perf_characteristics.strip_1_size_shared
                        / bw_interpolator.get_bw(static_cast<int>(
                                layer_perf_characteristics.strip_1_share_coef))
                + layer_perf_characteristics.strip_1_size_private
                        / bw_interpolator.get_bw(1);
    } else if (layer_perf_characteristics.strip1_b_in_mlc_h
            || layer_perf_characteristics.strip1_a_read_v)
        strip_1_cycles = strip_mid_cycles; // strip 1 is regular strip;
    else
        strip_1_cycles = layer_perf_characteristics.strip_1_size_shared
                / bw_interpolator.llc_bw;
    return strip_1_cycles;
}

float matmul_amx_blocking_params_macro_t::calculate_reduction_cycles(
        layer_perf_characteristics_t &layer_perf_characteristics,
        bw_map_t &bw_interpolator) const {
    // Calculate reduction cycles
    float reduction_cycles;
    // Score the reduction as plain f32 accumulation: the narrower partials do
    // not make a K-split cheap enough to justify the M/N parallelism it costs.
    const dim_t reduction_c_dt_sz = narrow_c_scoring_active()
            ? effective_c_buf_dt_sz(nthr_k_)
            : acc_dt_sz;
    size_t c_size_per_core = m_per_thread * n_per_thread * reduction_c_dt_sz;

    if (nthr_k_ != 1) {
        if (c_size_per_core * 2 < L2_threshold() && batch == 1) {
            float reduction_read_bytes
                    = static_cast<float>((M * rnd_up(N, 16) * reduction_c_dt_sz)
                            * ((nthr_k_ - 1)) / (nthr_m_ * nthr_n_));
            float reduction_read_cycles;
            if (layer_perf_characteristics.a_size
                            + layer_perf_characteristics.b_size
                            + layer_perf_characteristics.d_size
                    < L2_threshold()) {
                reduction_read_cycles
                        = reduction_read_bytes / bw_interpolator.get_bw(2);
            } else {
                reduction_read_cycles
                        = reduction_read_bytes / bw_interpolator.llc_bw;
            }

            float reduction_write_bytes = static_cast<float>(
                    (M * N * c_dt_sz) / (nthr_m_ * nthr_n_));
            float reduction_write_cycles
                    = reduction_write_bytes / bw_interpolator.get_bw(1);
            // Add reduction const overhead - measured
            float reduction_const_overhead = 25000;
            reduction_cycles = reduction_read_cycles + reduction_write_cycles
                    + reduction_const_overhead;
        } else {
            // Don't do reduction if c tmp doesn't fit
            // Also parallel reduction is not supported for large batch
            return -1;
        }
    } else {
        reduction_cycles = 0;
    }
    return reduction_cycles;
}
float matmul_amx_blocking_params_macro_t::calculate_avx_insts_per_cache_line(
        layer_perf_characteristics_t &layer_perf_characteristics) const {
    const float down_convert_inst = 4;
    float avx_insts_per_cache_line
            = down_convert_inst + this->postops_inst_count;

    bool has_zp = this->src_zp_type != brgemm_broadcast_t::none
            || this->wei_zp_type != brgemm_broadcast_t::none
            || this->dst_zp_type != brgemm_broadcast_t::none;

    bool has_scales = this->with_dst_scales || this->with_src_scales
            || this->with_wei_scales;

    if (has_scales) avx_insts_per_cache_line += 1;
    if (has_zp) avx_insts_per_cache_line += 1;
    return avx_insts_per_cache_line;
}

float matmul_amx_blocking_params_macro_t::calculate_blocking_scores() const {
    layer_perf_characteristics_t layer_perf_characteristics;
    init_copy_characteristics(layer_perf_characteristics);
    calculate_layer_sizes(layer_perf_characteristics);

    bw_map_t bw_interpolator;

    float b_transform_cycles_v = layer_perf_characteristics.strips_b_tranform_v
            ? layer_perf_characteristics.strip_mid_size_shared
                    / bw_interpolator.get_bw(static_cast<int>(
                            layer_perf_characteristics.strip_mid_share_coef))
            : 0;

    float b_transform_cycles_h = layer_perf_characteristics.strip1_b_tranform_h
            ? layer_perf_characteristics.strip_1_size_shared
                    / bw_interpolator.get_bw(static_cast<int>(
                            layer_perf_characteristics.strip_1_share_coef))
            : 0;

    float a_read_cycles_v = layer_perf_characteristics.strip1_a_read_v
            ? layer_perf_characteristics.strip_1_size_shared
                    / bw_interpolator.get_bw(static_cast<int>(
                            layer_perf_characteristics.strip_1_share_coef))
            : 0;

    float a_read_cycles_h = layer_perf_characteristics.strips_a_read_h
            ? layer_perf_characteristics.strip_mid_size_shared
                    / bw_interpolator.get_bw(static_cast<int>(
                            layer_perf_characteristics.strip_mid_share_coef))
            : 0;

    float strip_mid_cycles = calculate_strip_mid_cycles(
            layer_perf_characteristics, bw_interpolator);
    float strip_1_cycles = calculate_strip_1_cycles(
            layer_perf_characteristics, bw_interpolator, strip_mid_cycles);

    float gemm_cycles = a_read_cycles_v + b_transform_cycles_h + strip_1_cycles
            + (layer_perf_characteristics.num_strip - 1)
                    * (b_transform_cycles_v + a_read_cycles_h
                            + strip_mid_cycles);

    float reduction_cycles = calculate_reduction_cycles(
            layer_perf_characteristics, bw_interpolator);

    if (reduction_cycles < 0)
        return 0; //reduction is not feasible, so the score is 0

    float total_macs = static_cast<float>(M * K * N * batch);
    float total_cycles = (gemm_cycles + reduction_cycles) * b_per_thread;

    int macs_per_cycle_base = 1024;
    float peak_macs_per_cycle
            = static_cast<float>((macs_per_cycle_base / gemm_dt_sz) * nthr);
    float peak_cycles = total_macs / peak_macs_per_cycle;
    return peak_cycles / total_cycles;
}

bool matmul_amx_blocking_params_macro_t::operator==(
        const matmul_amx_blocking_params_macro_t &other) const {
    bool same_score = other.efficiency_score_ == this->efficiency_score_;
    bool same_direction = this->is_horizontal == other.is_horizontal;
    bool same_l2_reuse = this->m_chunk_size_ * this->n_chunk_size_
            == other.m_chunk_size_ * other.n_chunk_size_;
    return same_score && same_direction && same_l2_reuse;
}

bool matmul_amx_blocking_params_macro_t::operator>(
        const matmul_amx_blocking_params_macro_t &other) const {
    if (other.efficiency_score_ > this->efficiency_score_) { return false; }
    if (other.efficiency_score_ < this->efficiency_score_) { return true; }
    // Both efficiency scores are equal
    if (!this->is_horizontal && other.is_horizontal) {
        if (this->m_per_thread * K + (size_t)(this->m_per_thread * N)
                < L2_threshold()) {
            // Vertical is an option. No l2 set issues for A
            if (other.is_a_nt_) {
                // Horizontal doesn't use the L1
                return true;
            }
        }
        return false;
    } else if (this->is_horizontal && !other.is_horizontal) {
        if (other.m_per_thread * K + (size_t)(other.m_per_thread * N)
                < L2_threshold()) {
            // Vertical is an option. No L2 set issues for A
            if (this->is_a_nt_) {
                // Horizontal doesn't use the L1
                return false;
            }
        }
        return true;
    } else {
        // Both are vertical or both are horizontal
        // Pick by L2 reuse - the one with the largest m/n_chunk
        // One of m_chunk_size_ and n_chunk_size is always 1
        return this->m_chunk_size_ * this->n_chunk_size_
                > other.m_chunk_size_ * other.n_chunk_size_;
    }
}

bool matmul_amx_blocking_params_macro_t::operator!=(
        const matmul_amx_blocking_params_macro_t &other) const {
    return !(*this == other);
}

bool matmul_amx_blocking_params_macro_t::operator<(
        const matmul_amx_blocking_params_macro_t &other) const {
    return *this != other && !(*this > other);
}

dim_t matmul_amx_blocking_params_macro_t::calc_k_blk(size_t l1_dim) const {
    // Assuming 2x2 decomposition
    const size_t c_tiles = m_decomposition * n_decomposition * acc_dt_sz;
    const size_t d_tiles = m_decomposition
            * rnd_up(n_decomposition * c_dt_sz,
                    64); // Rounded up to cache line size
    const size_t available_space_in_l1
            = L1_threshold() - (c_tiles * 2 + d_tiles);

    const dim_t largest_k = available_space_in_l1 / (l1_dim * gemm_dt_sz);
    const dim_t largest_k_tiles = largest_k / this->k_tmul;
    const dim_t k_tiles = div_up(K, this->k_tmul);
    const dim_t k_per_thread_tiles = div_up(k_tiles, nthr_k_);
    const dim_t num_K_blocks = div_up(k_per_thread_tiles, largest_k_tiles);
    return nstl::min(
            (dim_t)(div_up(k_per_thread_tiles, num_K_blocks) * this->k_tmul),
            K);
}

std::set<dim_t> matmul_amx_blocking_params_macro_t::blk_candidates(
        dim_t dim_per_thread, dim_t decomposition) const {
    dim_t num_inner_blocks = div_up(dim_per_thread, decomposition);
    std::set<dim_t> dim_set;
    for (int num_groups = 1; num_groups <= num_inner_blocks; ++num_groups) {
        dim_t group_size = div_up(num_inner_blocks, num_groups);
        dim_set.insert(group_size);
    }

    return dim_set;
}

size_t matmul_amx_blocking_params_macro_t::l2_matrix_usage(size_t k_chunk_size,
        size_t m_or_n_blk, size_t k_blk, bool is_horizontal,
        bool force_transform_matrix_to_l2) const {
    dim_t decomposition = is_horizontal ? m_decomposition : n_decomposition;
    const dim_t k_iter_elems
            = nstl::min((dim_t)(k_blk * k_chunk_size), (dim_t)k_per_thread);
    size_t l1_matrix_size
            = 2 * decomposition * k_iter_elems * gemm_dt_sz; // 2 for prefetch
    size_t l2_matrix_size = m_or_n_blk * k_iter_elems * gemm_dt_sz;
    if (force_transform_matrix_to_l2) {
        /* L2 matrix orig size */
        l2_matrix_size += m_or_n_blk * k_blk * k_chunk_size
                * (is_horizontal ? b_dt_sz : a_dt_sz);
    }
    // Calculate C post size (output buffer)
    size_t c_post_size = 0;
    if (nthr_k_ == 1 || (acc_dt == dst_dt)) {
        // With is_c_buf_dst_dt the final dst buffer holds the intermediate
        // results; with reduction on top of it a similarly sized buffer is
        // used instead.
        if (is_horizontal) {
            c_post_size
                    = 2 * m_decomposition * rnd_up(m_or_n_blk * c_dt_sz, 64);
        } else {
            c_post_size
                    = 2 * rnd_up(n_decomposition * c_dt_sz, 64) * m_or_n_blk;
        }
    }
    // Calculate C tmp size (partial results buffer)
    size_t c_tmp_size;
    if (k_blk == (size_t)K || (acc_dt == dst_dt && nthr_k_ == 1)) {
        c_tmp_size = 2 * m_decomposition * n_decomposition * acc_dt_sz;
    } else {
        c_tmp_size = 2 * decomposition * m_or_n_blk * acc_dt_sz;
    }

    return l1_matrix_size + l2_matrix_size + c_tmp_size + c_post_size;
}

size_t matmul_amx_blocking_params_macro_t::l2_matrix_and_c_usage(
        size_t k_chunk_size, size_t m_or_n_blk, size_t k_blk,
        bool is_horizontal) const {
    // NOTE: this, l2_matrix_usage() and bw() run during candidate generation
    // with trial k_blk/k_chunk, so they always price C as f32: keying them on
    // is_c_buf_dst_dt would make relaxed enumerate a different candidate set.
    size_t per_thread_for_l1_matrix
            = is_horizontal ? m_per_thread : n_per_thread;
    const dim_t k_iter_elems
            = nstl::min((dim_t)(k_blk * k_chunk_size), (dim_t)k_per_thread);
    size_t l1_matrix_size = 2 * per_thread_for_l1_matrix * k_iter_elems
            * gemm_dt_sz; // 2x factor to make sure C is fresher than A,B in LRU
    size_t l2_matrix_size = 2 * m_or_n_blk * k_iter_elems
            * gemm_dt_sz; // 2x factor to make sure C is fresher than A,B in LRU
    size_t c_size
            = per_thread_for_l1_matrix * m_or_n_blk * acc_dt_sz; // C in L2
    return l1_matrix_size + l2_matrix_size + c_size;
}

size_t matmul_amx_blocking_params_macro_t::bw(size_t m_blk, size_t k_chunk_size,
        size_t k_blk, size_t n_blk, bool is_horizontal) const {
    const dim_t k_iter_elems
            = nstl::min((dim_t)(k_blk * k_chunk_size), (dim_t)k_per_thread);
    size_t a_bw = m_blk * k_iter_elems * gemm_dt_sz;
    size_t b_bw = n_blk * k_iter_elems * gemm_dt_sz;
    size_t c_bw;

    // is_c_buf_dst_dt bypass: D RMW'd per K-chunk (vs f32-scratch). CRUCIAL: RMW cost only
    // when K>1 chunk; single K-chunk has c_bw=0 for wide-N parallelism on small shapes.
    const bool single_k_chunk = k_iter_elems == K;
    const bool c_bypass_to_dst
            = is_c_buf_dst_dt && nthr_k_ <= 1 && !with_sum && !single_k_chunk;

    if (!c_bypass_to_dst
            && (l2_matrix_and_c_usage(k_chunk_size,
                        is_horizontal ? n_blk : m_blk, k_blk, is_horizontal)
                            < L2_threshold()
                    || single_k_chunk)
            && nthr_k_ == 1) {
        c_bw = 0;
    } else {
        c_bw = m_blk * n_blk * acc_dt_sz;
    }
    return a_bw + b_bw + c_bw;
}

size_t matmul_amx_blocking_params_macro_t::compute(
        size_t m_blk, size_t k_chunk_size, size_t k_blk, size_t n_blk) const {
    const dim_t k_iter_elems
            = nstl::min((dim_t)(k_blk * k_chunk_size), (dim_t)k_per_thread);
    return m_blk * k_iter_elems * n_blk;
}

float matmul_amx_blocking_params_macro_t::ratio(size_t m_blk,
        size_t k_chunk_size, size_t k_blk, size_t n_blk,
        bool is_horizontal) const {
    return static_cast<float>(compute(m_blk, k_chunk_size, k_blk, n_blk))
            / bw(m_blk, k_chunk_size, k_blk, n_blk, is_horizontal);
}

float matmul_amx_blocking_params_macro_t::evaluate_single_core_blocking(
        size_t k_chunk_size, size_t m_or_n_blk, size_t k_blk,
        bool is_horizontal, bool force_transform_matrix_to_l2) const {
    if (l2_matrix_usage(k_chunk_size, m_or_n_blk, k_blk, is_horizontal,
                force_transform_matrix_to_l2)
            <= L2_threshold()) {
        size_t m_blk, n_blk;
        if (is_horizontal) {
            m_blk = m_decomposition;
            n_blk = m_or_n_blk;
        } else {
            m_blk = m_or_n_blk;
            n_blk = n_decomposition;
        }
        float ratio_score
                = ratio(m_blk, k_chunk_size, k_blk, n_blk, is_horizontal);
        return ratio_score;
    }
    return 0;
}

void matmul_amx_blocking_params_macro_t::set_tmul_sizes() {
    // determine_tmul_size() models TMUL's ability to compute on a reduced
    // number of rows. ACE has no such degree of freedom: an accumulator tile
    // always spans ace_m_tmul rows.
    this->m_tmul = is_ace ? static_cast<size_t>(ace_m_tmul)
                          : determine_tmul_size(this->m_per_thread, 16);
    this->n_tmul = 16; // B blocked layout is a multiply of 16
    this->k_tmul = nstl::min((size_t)wei_k_blk, (size_t)K);
}

void matmul_amx_blocking_params_macro_t::set_decomposition() {
    // The M multiplier is the same on both paths (ace_m_tiles == 2); only the
    // N direction differs, ACE covering ace_n_tiles tiles per iteration.
    const size_t n_tiles = is_ace ? static_cast<size_t>(ace_n_tiles) : 2;
    m_decomposition = nstl::min((size_t)M, 2 * m_tmul);
    n_decomposition = nstl::min((size_t)N, n_tiles * n_tmul);
}

bool matmul_amx_blocking_params_macro_t::is_horizontal_selected(
        bool horizontal_not_possible, bool vertical_not_possible,
        size_t best_m_v, size_t best_k_v, size_t k_blk_v) const {
    // Choose between horizontal and vertical

    bool is_horizontal_local;

    if (horizontal_not_possible) {
        is_horizontal_local = false;
    } else if (vertical_not_possible) {
        is_horizontal_local = true;
    } else if ((size_t)m_per_thread < m_tmul * 2) {
        // There are not enough tiles in M direction to go vertical
        is_horizontal_local = true;
    } else if ((size_t)n_per_thread < n_tmul * 2) {
        // There are not enough tiles in N direction to go horizontal
        is_horizontal_local = false;
    } else if (m_per_thread >= n_per_thread) {
        // Choose horizontal
        is_horizontal_local = true;
    } else {
        // Choose vertical
        is_horizontal_local = false;
    }
    return is_horizontal_local;
}

bool matmul_amx_blocking_params_macro_t::set_blocking_parameters(
        bool force_horizontal, bool force_b_transform_to_l2) {

    std::set<dim_t> m_candidates
            = blk_candidates(m_per_thread, m_decomposition);
    std::set<dim_t> n_candidates
            = blk_candidates(n_per_thread, n_decomposition);
    dim_t best_k_h = 0, best_n_h = 0;
    dim_t best_m_v = 0, best_k_v = 0;
    float best_score_h = 0, best_score_v = 0;
    bool horizontal_not_possible = false;
    bool vertical_not_possible = force_horizontal;

    auto calc_horizontal = [&](size_t k_blk_h, dim_t min_k_chunk_size = 0) {
        if (rnd_up(m_per_thread, m_decomposition) * (nthr_m_ - 1) >= M) {
            horizontal_not_possible = true;
        } else if (rnd_up(k_per_thread, k_blk_h) * (nthr_k_ - 1) >= K) {
            // Early exit: There is no possible division of work for nthr_k threads
            horizontal_not_possible = true;
        } else {
            std::set<dim_t> k_candidates_h
                    = blk_candidates(k_per_thread, k_blk_h);
            best_n_h = 0;
            for (std::set<dim_t>::reverse_iterator it_n = n_candidates.rbegin();
                    it_n != n_candidates.rend(); it_n++) {
                for (std::set<dim_t>::reverse_iterator it_k
                        = k_candidates_h.rbegin();
                        it_k != k_candidates_h.rend(); it_k++) {
                    float cur_score = evaluate_single_core_blocking(*it_k,
                            *it_n * n_decomposition, k_blk_h, true,
                            force_b_transform_to_l2);
                    if (cur_score > best_score_h && *it_k >= min_k_chunk_size) {
                        best_score_h = cur_score;
                        best_k_h = *it_k;
                        best_n_h = *it_n;
                    }
                }
            }

            if (best_n_h == 0) {
                horizontal_not_possible = true;
                return;
            }

            if (rnd_up(n_per_thread, best_n_h * n_decomposition) * (nthr_n_ - 1)
                    >= N) {
                horizontal_not_possible = true;
            }
            if (rnd_up(k_per_thread, best_k_h * k_blk_h) * (nthr_k_ - 1) >= K) {
                // There is not enough work for nthr_k threads
                horizontal_not_possible = true;
            }
        }
    };
    // Calculate best score for horizontal traversal
    dim_t k_blk_h = calc_k_blk(m_decomposition);
    calc_horizontal(k_blk_h);

    auto calc_vertical = [&](size_t k_blk_v, dim_t min_k_chunk_size = 0) {
        if (rnd_up(n_per_thread, n_decomposition) * (nthr_n_ - 1) >= N) {
            vertical_not_possible = true;
        } else if (rnd_up(k_per_thread, k_blk_v) * (nthr_k_ - 1) >= K) {
            // Early exit: There is no possible division of work for nthr_k threads
            vertical_not_possible = true;
        } else {
            // Calculate best score for vertical traversal
            std::set<dim_t> k_candidates_v
                    = blk_candidates(k_per_thread, k_blk_v);
            for (std::set<dim_t>::reverse_iterator it_m = m_candidates.rbegin();
                    it_m != m_candidates.rend(); it_m++) {
                for (std::set<dim_t>::reverse_iterator it_k
                        = k_candidates_v.rbegin();
                        it_k != k_candidates_v.rend(); it_k++) {
                    float cur_score = evaluate_single_core_blocking(
                            *it_k, *it_m * m_decomposition, k_blk_v, false);
                    if (cur_score > best_score_v && *it_k >= min_k_chunk_size) {
                        best_score_v = cur_score;
                        best_k_v = *it_k;
                        best_m_v = *it_m;
                    }
                }
            }

            if (best_m_v == 0) {
                vertical_not_possible = true;
                return;
            }

            if (rnd_up(m_per_thread, best_m_v * m_decomposition) * (nthr_m_ - 1)
                    >= M) {
                vertical_not_possible = true;
            }
            if (rnd_up(k_per_thread, best_k_v * k_blk_v) * (nthr_k_ - 1) >= K) {
                // There is not enough work for nthr_k threads
                vertical_not_possible = true;
            }
            size_t l2_util_v = 0;

            if (!vertical_not_possible) {
                // Figure out if vertical is an option wrt L2 usage
                l2_util_v = l2_matrix_and_c_usage(
                        best_k_v, best_m_v * m_decomposition, k_blk_v, false);
                if (l2_util_v > L2_threshold()) {
                    l2_util_v = l2_matrix_usage(best_k_v,
                            best_m_v * m_decomposition, k_blk_v, false);
                }
            }
            bool repeat_loop_over_k = div_up(K, k_blk_v * best_k_v) != 1;
            bool critical_l2_set_issues_a
                    = div_up(K, k_blk_v * best_k_v) != nthr_k_
                    || (size_t)((l2_util_v * nthr_k_)) >= L2_threshold();

            if (repeat_loop_over_k && critical_l2_set_issues_a)
                vertical_not_possible = true;

            uint32_t l2_sets = platform::get_num_sets_in_cache(2);
            size_t l2_line_size = l2_sets * platform::get_cache_line_size();
            if (l2_line_size == 0) {
                vertical_not_possible = true;
                return;
            }

            // If output matrix in vertical traversal suffers from set issues in
            // the L2, then go horizontally.
            // First, calculate the stride on the N dim within a page
            size_t cache_stride = N * c_dt_sz % PAGE_4K;
            if (cache_stride == 0) { cache_stride = PAGE_4K; }
            // Second, calculate the number of ways the output matrix will use
            // in each set. If higher than the threshold, useful cache lines in
            // L2 will be evicted.
            if (l2_line_size % cache_stride == 0) {
                size_t num_ways_c
                        = div_up(cache_stride * m_per_thread, l2_line_size);
                if (num_ways_c >= L2_ways_threshold()) {
                    vertical_not_possible = true;
                }
            }
        }
    };

    dim_t k_blk_v = calc_k_blk(n_decomposition);
    calc_vertical(k_blk_v);

    if (vertical_not_possible && horizontal_not_possible) { return false; }

    is_horizontal = is_horizontal_selected(horizontal_not_possible,
            vertical_not_possible, best_m_v, best_k_v, k_blk_v);

    auto is_postops_bound = [&](size_t k_blk) {
        // If the number of cycles for postops per cache line is larger than
        // the number of AMX cycles per cache line, then the calculation is
        // postops bound.
        return postops_inst_count / avx_ipc > div_up(k_blk, k_tmul);
    };

    if (is_horizontal) {
        size_t l1_eff_factor = div_up(K, k_blk_h);
        // This works for M > 32 in this case k_blk_h << 4096 =~ 512
        // For M <= the problem is heavily memory bound ==> don't care about the L1 and work completely from the L2

        size_t a_l1 = k_blk_h * m_decomposition * gemm_dt_sz;
        size_t c_l1 = n_decomposition * m_decomposition * acc_dt_sz;
        size_t d_post = m_decomposition * rnd_up(n_decomposition * c_dt_sz, 64);
        is_a_nt_ = false;
        is_b_nt_ = true;
        bool l1_set_issues = k_blk_h < K
                && l1_eff_factor * a_l1 + 2 * c_l1 + d_post > L1_threshold();

        // ACE always takes this path.
        // There are no NT loads in AVX instructions for ACE => l1 blocking is not possible.
        // Select the L2-level blocking directly instead.
        if (l1_set_issues || is_postops_bound(k_blk_h) || is_ace) {
            assert(best_n_h != 0);
            // Give up on the L1 blocking
            best_score_h = 0;
            // Calculate k_blk_h and n_blk_h that can fit in the L2 when k_blk is wei_k_blk
            calc_horizontal(wei_k_blk, k_blk_h / wei_k_blk);
            // Give up on the L1.
            k_blk_h = nstl::min(wei_k_blk * best_k_h, K);
            best_k_h = 1;
            // No effect on ACE: the NT hints are only read by the tile-load
            // path, which ACE does not use (it loads A/B through ZMMs).
            is_a_nt_ = true;
        }

        // Re-check A's L1 residency after the "give up on L1" branch above
        // shrinks k_blk_h. That branch unconditionally sets is_a_nt_ = true,
        // but the smaller k_blk_h often leaves A's per-iter panel
        // (k_blk_h * m_decomposition * gemm_dt_sz) fitting in L1 alongside
        // C/D. In that case prefer caching A in L1 (T0): the same A panel
        // is reused by every AMX N-tile within one k_blk iteration and
        // avoiding the NT hint also prevents needless eviction of B/C from
        // L1/L2.
        //
        // The original L1-residency predicate (l1_set_issues) over-estimated
        // A's footprint by scaling it with l1_eff_factor = div_up(K, k_blk_h),
        // i.e. counting all K-blocks as co-resident, which forced that branch
        // to trigger more often than warranted.
        const size_t a_l1_final
                = (size_t)k_blk_h * m_decomposition * gemm_dt_sz;
        // Per-iter L1 budget mirrors the original predicate above.
        const bool a_fits_l1 = a_l1_final + 2 * c_l1 + d_post <= L1_threshold();
        // Skip the override for write-bound layers (small K). On such layers
        // the bottleneck is C writes. Pulling A into L1 evicts the C
        // accumulator and hurts performance even when A nominally fits.
        const bool is_write_bound_layer = nthr_k_ == 1
                && k_per_thread < k_threshold_write_bound_layer_elem;
        if (a_fits_l1 && !is_write_bound_layer) is_a_nt_ = false;

        k_blk_ = k_blk_h;
        k_chunk_size_ = best_k_h;
        n_blk_ = nstl::min(best_n_h * n_decomposition, N);
        n_chunk_size_ = 1;
        m_blk_ = m_decomposition;
        m_chunk_size_ = div_up(m_per_thread, m_blk_);
        need_prefetch_a_ = (m_per_thread / m_blk_) >= 2;
        need_prefetch_b_ = false;
        // use fused copy A if A is unaligned and reused at least 2 times and not completely write bound.
        use_fused_copy_a_ = (K % wei_k_blk != 0)
                && (n_blk_ >= n_decomposition * 3)
                && ((size_t)k_blk_ >= (64 / gemm_dt_sz) * 4);

        // extendable_k reads A's K dimension rounded up to the tile
        // granularity directly from memory, so it is only valid when A is
        // contiguous along K (leading dimension equal to K). A padded LDA
        // would make the kernel stride rows by the padded value while reading
        // the extended K, corrupting the result.
        extendable_k_ = K % wei_k_blk != 0 && !skip_extendable_k()
                && !use_fused_copy_a_ && get_actual_lda() == K;

    } else {
        is_a_nt_ = true;
        is_b_nt_ = false;

        // ACE always takes this path, for the same reason as the horizontal
        // branch above
        if (is_postops_bound(k_blk_v) || is_ace) {
            assert(best_m_v != 0);
            // Give up on the L1 blocking
            best_score_v = 0;
            // Calculate k_blk_h and n_blk_h that can fit in the L2 when k_blk is wei_k_blk
            calc_vertical(wei_k_blk, k_blk_v / wei_k_blk);
            // Give up on the L1.
            k_blk_v = nstl::min(wei_k_blk * best_k_v, K);
            best_k_v = 1;
            // No effect on ACE, see the horizontal branch.
            is_b_nt_ = true;
        }

        k_blk_ = k_blk_v;
        k_chunk_size_ = best_k_v;
        n_blk_ = n_decomposition;
        n_chunk_size_ = div_up(n_per_thread, n_blk_);
        m_blk_ = nstl::min(best_m_v * m_decomposition, M);
        m_chunk_size_ = 1;
        need_prefetch_a_ = false;
        need_prefetch_b_ = ((n_per_thread / n_blk_) >= 2) && !use_buffer_b;
        use_fused_copy_a_ = false;

        // extendable_k requires A to be contiguous along K.
        extendable_k_ = K % wei_k_blk != 0 && !skip_extendable_k()
                && get_actual_lda() == K;
    }

    brgemm_batch_size_ = 1;

    n_chunk_elems_ = nstl::min(n_per_thread, n_blk_ * n_chunk_size_);
    m_chunk_elems_ = nstl::min(m_per_thread, m_blk_ * m_chunk_size_);
    k_chunk_elems_ = nstl::min(k_per_thread, k_blk_ * k_chunk_size_);

    set_nt_ = true;

    current_lda_ = get_actual_lda();

    // Need a temp C buffer if a BRGEMM creates partial results
    need_buf_c_
            = (nthr_k_ != 1) || (k_blk_ != K && (with_sum || acc_dt != dst_dt));

    efficiency_score_ = calculate_blocking_scores();

    return true;
}
bool matmul_amx_blocking_params_macro_t::skip_extendable_k() const {

    auto skip_for_k_blk = [&](size_t k_to_check) {
        size_t num_amx_ops_over_k = div_up(k_to_check, 64 / gemm_dt_sz);
        bool common_k_tile_exists = k_to_check % num_amx_ops_over_k == 0;
        size_t k_tile = k_to_check / num_amx_ops_over_k;
        bool k_tile_size_fits = k_tile <= 64 / gemm_dt_sz;
        bool k_tile_is_vnni = k_tile % data_type_vnni_granularity(wei_dt) == 0;

        return common_k_tile_exists && k_tile_size_fits && k_tile_is_vnni;
    };

    size_t k_tail = K % k_blk_;
    if (k_tail) {
        return skip_for_k_blk(k_blk_) && skip_for_k_blk(k_tail);
    } else {
        return skip_for_k_blk(k_blk_);
    }
}

void matmul_amx_blocking_params_macro_t::set_core_divs(
        int nthr_b, int nthr_m, int nthr_k, int nthr_n) {
    nthr_b_ = nthr_b;
    nthr_m_ = nthr_m;
    nthr_k_ = nthr_k;
    nthr_n_ = nthr_n;
    m_per_thread = div_up(M, nthr_m_);
    k_per_thread = div_up(K, nthr_k_);
    n_per_thread = div_up(N, nthr_n_);
    b_per_thread = div_up(this->batch, nthr_b_);

    nthr_mnb_ = nthr_ / nthr_k_;
}

void matmul_amx_blocking_params_micro_t::find_best_blocking(
        const brgemm_matmul_conf_t &bgmmc,
        const brgemm_matmul_conf_utils_t &bm_conf_utils,
        matmul_amx_blocking_params_micro_t &best_blocking) {

    matmul_amx_blocking_params_micro_t current_blocking(bgmmc);

    const int min_k_per_thread = 1024;
    const int max_k_parallel_work
            = div_up(static_cast<int>(bgmmc.K), min_k_per_thread);
    const bool is_amx_xf16 = bgmmc.is_amx
            && (bm_conf_utils.is_bf16() || bm_conf_utils.is_f16()
                    || bm_conf_utils.is_f32_f16() || bm_conf_utils.is_f32_bf16()
                    || bm_conf_utils.is_bf32()
                    || bm_conf_utils.is_bf16_with_int_wei()
                    || bm_conf_utils.is_f16_with_int_wei()
                    || bm_conf_utils.is_bf16_fp8()
                    || bm_conf_utils.is_f16_fp8());
    const bool is_amx_int8 = bgmmc.is_amx && bm_conf_utils.is_int8();

    const bool runtime_dims
            = bgmmc.is_runtime_M || bgmmc.is_runtime_N || bgmmc.is_runtime_K;
    const int max_nthr_k = !runtime_dims && is_amx_xf16 && bgmmc.batch == 1
            ? nstl::min<int>(
                      saturate<int>(1, 7, bgmmc.nthr / 8), max_k_parallel_work)
            : 1;
    int iter = 0;
    const int runtime_M_chunk = bgmmc.lda_big_pow2() ? 2 : 4;
    const int runtime_N_chunk = 2;

    // Disable skip configuration due to regressions for some cases.
    const bool disable_skip_config = bgmmc.M == 4
            && utils::one_of(true, bgmmc.N == 4096 && bgmmc.K == 4096,
                    bgmmc.N == 11008 && bgmmc.K == 4096,
                    bgmmc.N == 4096 && bgmmc.K == 11008);

    for (int nthr_k = 1; nthr_k <= max_nthr_k; nthr_k++) {
        int nthr_bmn = bgmmc.nthr / nthr_k;

        dim_t num_M_blk = bgmmc.is_runtime_M ? 1 : div_up(bgmmc.M, bgmmc.M_blk);
        dim_t num_N_blk = bgmmc.is_runtime_N ? 1 : div_up(bgmmc.N, bgmmc.N_blk);
        dim_t k_parallel_work = nstl::min<dim_t>(max_k_parallel_work, nthr_k);
        dim_t num_parallel_work
                = bgmmc.batch * num_M_blk * num_N_blk * k_parallel_work;
        const bool a_lot_of_parallel_work_lvl2
                = num_parallel_work > 16 * bgmmc.nthr;
        const bool low_parallelism
                = static_cast<float>(num_parallel_work) < 1.5f * bgmmc.nthr;
        const bool maybe_low_blocking
                = is_amx_int8 && bm_conf_utils.maybe_low_brg_blocking();
        const dim_t min_M_blk = !bgmmc.is_runtime_M
                        && (maybe_low_blocking || low_parallelism)
                        && bgmmc.M_blk > 32
                ? div_up(bgmmc.M_blk, 2)
                : bgmmc.M_blk;
        const dim_t min_N_blk = !bgmmc.is_runtime_N && low_parallelism
                        && is_amx_xf16 && !bm_conf_utils.check_n_blk_fixed()
                        && bgmmc.N_blk > 32 && !runtime_dims
                        && !bgmmc.transposed_B // Transposed copy B kernel doesn't support adjusting N_blk
                ? 32
                : bgmmc.N_blk;
        const dim_t desired_M_chunk = bgmmc.is_runtime_M
                ? runtime_M_chunk
                : nstl::min<dim_t>(4, num_M_blk);
        const dim_t desired_N_chunk = bgmmc.is_runtime_N
                ? runtime_N_chunk
                : nstl::min<dim_t>(
                          a_lot_of_parallel_work_lvl2 ? 6 : 4, num_N_blk);

        std::unordered_set<dim_t> mblk_candidates;
        for (dim_t m_blk = bgmmc.M_blk; m_blk >= min_M_blk;
                m_blk = m_blk > 1 ? div_up(m_blk, 2) : m_blk - 1) {
            if (IMPLICATION(maybe_low_blocking, m_blk != bgmmc.M_blk))
                mblk_candidates.insert(m_blk);
        }

        if (!bgmmc.is_runtime_M && bgmmc.M > 16) {
            // Add multiple of 16 M block sizes for consideration
            const dim_t mul16_m_blk_max
                    = nstl::min<dim_t>(rnd_dn(bgmmc.M, (dim_t)16), 64);
            const dim_t mul16_m_blk_min = rnd_up(min_M_blk, (dim_t)16);
            for (dim_t m_blk = mul16_m_blk_max; m_blk >= mul16_m_blk_min;
                    m_blk -= 16) {
                mblk_candidates.insert(m_blk);
            }
        }

        bool found_best_blocking = false;
        for_(dim_t n_blk = bgmmc.N_blk; n_blk >= min_N_blk; n_blk -= 16)
        for_(dim_t m_blk : mblk_candidates)
        for_(dim_t n_ch_sz = desired_N_chunk; n_ch_sz >= 1; n_ch_sz--)
        for (dim_t m_ch_sz = desired_M_chunk; m_ch_sz >= 1; m_ch_sz--, iter++) {
            current_blocking.set_blocking_parameters(
                    nthr_k, n_blk, n_ch_sz, m_blk, m_ch_sz);

            float cur_score = current_blocking.get_blocking_scores();
            float bst_score = best_blocking.get_blocking_scores();

            // TODO: Verify whether using a chunk count of 1 for runtime M and N is optimal for
            // this heuristic. The previous implementation inadvertently used DNNL_RUNTIME_DIM_VAL
            // for M and N in arithmetic, producing incorrect work_amount values.
            dim_t m_chunks
                    = bgmmc.is_runtime_M ? 1 : div_up(bgmmc.M, m_blk * m_ch_sz);
            dim_t n_chunks
                    = bgmmc.is_runtime_N ? 1 : div_up(bgmmc.N, n_blk * n_ch_sz);
            dim_t work_amount = bgmmc.batch * m_chunks * n_chunks;

            bool skip_config = work_amount < nthr_bmn * 3
                    && work_amount % nthr_bmn != 0 && max_nthr_k == 1;
            if (skip_config && !disable_skip_config) continue;

            if (cur_score > bst_score) {
                best_blocking = current_blocking;
                found_best_blocking = true;
            }
        }

        if (!found_best_blocking) {
            current_blocking.set_blocking_parameters(
                    nthr_k, min_N_blk, 1, min_M_blk, 1);

            float cur_score = current_blocking.get_blocking_scores();
            float bst_score = best_blocking.get_blocking_scores();
            if (cur_score > bst_score) best_blocking = current_blocking;
        }
    }
}

void matmul_amx_blocking_params_micro_t::update_k_blocking_dependent_params() {
    k_chunk_elems_ = k_blk_ * k_chunk_size_ * brgemm_batch_size_;
    current_lda_ = get_actual_lda();
    need_buf_c_ = is_buffer_c_required();
}

void matmul_amx_blocking_params_micro_t::set_blocking_parameters(int nthr_k,
        dim_t n_blk, dim_t n_chunk_size, dim_t m_blk, dim_t m_chunk_size) {
    nthr_k_ = nstl::max(1, nthr_k);
    nthr_mnb_ = nthr / nthr_k_;
    nthr_ = nthr_mnb_ * nthr_k_;
    n_blk_ = n_blk;
    n_chunk_size_ = n_chunk_size;
    m_blk_ = m_blk;
    m_chunk_size_ = m_chunk_size;

    if (one_of(0, n_blk_, n_chunk_size_, m_blk_, m_chunk_size_)) {
        k_blk_ = k_chunk_size_ = k_chunk_elems_ = brgemm_batch_size_ = 0;
        efficiency_score_ = 0.0f;
        return;
    }

    n_chunk_elems_ = n_blk_ * n_chunk_size_;
    m_chunk_elems_ = m_blk_ * m_chunk_size_;

    if (K < wei_k_blk) {
        k_blk_ = is_amx ? rnd_up(K, required_k_granularity) : K;
        brgemm_batch_size_ = 1;
    } else {
        dim_t k_per_thr = div_up(K, nthr_k_);
        k_blk_ = nstl::min(rnd_up(k_per_thr, required_k_granularity),
                static_cast<dim_t>(wei_k_blk));
        const dim_t num_k_blk = div_up(K, k_blk_);
        const dim_t num_k_blk_per_thread = div_up(num_k_blk, nthr_k_);
        brgemm_batch_size_ = num_k_blk_per_thread;

        auto chunk_sz = calculate_chunk_memory_size();
        const dim_t div_min = chunk_sz / L2_threshold();
        const dim_t div_max = div_up(chunk_sz, L2_threshold());
        // For big pow2 lda prefer to increase area of linear memory access
        const dim_t adjust_k_divisor_threshold = lda_big_pow2() ? 2 : 0;
        // Adjust K blocking values to fit into L2 cache
        if (div_min > adjust_k_divisor_threshold && brgemm_batch_size_ > 1) {
            const auto kc1 = nstl::max(
                    brgemm_batch_size_ / div_min, static_cast<dim_t>(1));
            const auto kc2 = div_up(brgemm_batch_size_, div_max);
            const auto tail1 = num_k_blk_per_thread % kc1;
            const auto tail2 = num_k_blk_per_thread % kc2;
            // Prefer adjusted chunk size with more equal work distribution
            // Across iterations
            brgemm_batch_size_
                    = IMPLICATION(tail1 == 0 || tail2 < tail1, tail2 == 0)
                    ? kc2
                    : kc1;
        }

        k_chunk_elems_ = k_blk_ * brgemm_batch_size_ * k_chunk_size_;
        dim_t brgemm_k_elems = k_blk_ * brgemm_batch_size_;
        const dim_t current_k_tail = K % k_blk_;

        // TODO: review extendable_k_ condition to cover more cases
        extendable_k_ = (K % wei_k_blk != 0) && (brgemm_k_elems > wei_k_blk)
                && wei_zp_type == none && !apply_scales_in_buffer_b
                && !use_buffer_a && !packed_sparse_weights && current_lda_ == K;

        if (extendable_k_) {
            if (brgemm_k_elems >= K) {
                k_blk_ = K;
                k_chunk_size_ = 1;
                brgemm_batch_size_ = 1;
            } else {
                k_blk_ = brgemm_k_elems;
                k_chunk_size_ = 1;
                brgemm_batch_size_ = 1;
            }
        } else if (current_k_tail == 0
                && K % (k_blk_ * brgemm_batch_size_) == 0) {
            k_blk_ = brgemm_k_elems;
            brgemm_batch_size_ = 1;
        } else if (nthr_k_ == 1
                && K == k_blk_ * brgemm_batch_size_ + current_k_tail) {
            k_blk_ = brgemm_k_elems;
            brgemm_batch_size_ = 2;
        }
    }
    need_buf_a_
            = use_buffer_a || (!extendable_k_ && K % required_k_granularity);

    blocking_chunk_mem_size_ = calculate_chunk_memory_size();

    efficiency_score_ = calculate_blocking_scores();
}

// Returns score for current blocking parameters' values in range [0, 1]
// for parallel work over threads distribution score. Maximum scores - when
// all threads have the same work amount w/o tails
float matmul_amx_blocking_params_micro_t::get_thread_balance_scores() const {
    assert(!(is_runtime_M && is_runtime_N)
            && "single runtime dim is supported");
    // Ignore M sizes in thread balance computation as actual M size is unknown
    if (is_runtime_M) return (float)N / rnd_up(N, n_chunk_elems_);
    // Ignore N sizes in thread balance computation as actual N size is unknown
    if (is_runtime_N) return (float)M / rnd_up(M, m_chunk_elems_);

    const dim_t num_M_chunks = div_up(M, m_chunk_elems_);
    const dim_t num_N_chunks = div_up(N, n_chunk_elems_);
    float mnb_parallel_score = batch * ((float)M / m_chunk_elems_)
            * ((float)N / n_chunk_elems_)
            / rnd_up(batch * num_M_chunks * num_N_chunks, nthr_mnb_)
            * nthr_mnb_;
    float k_parallel_score = 1.0f;
    if (nthr_k_ > 1) {
        const dim_t num_K_chunks = div_up(K, k_chunk_elems_);
        const float parallel_reduction_penalty = 0.8f;
        k_parallel_score = parallel_reduction_penalty
                * ((float)K / k_chunk_elems_) / rnd_up(num_K_chunks, nthr_k_)
                * nthr_k_;
    }

    return mnb_parallel_score * k_parallel_score / nthr;
}

// Returns score for current blocking parameters' values in range [0, 1]
// for copied data reusage
float matmul_amx_blocking_params_micro_t::get_copied_data_reusage_scores()
        const {
    const dim_t effective_m_chunk_sz = 64 * 4;
    const dim_t desired_M_chunk_size = is_runtime_M
            ? effective_m_chunk_sz
            : nstl::min(M, effective_m_chunk_sz);
    const dim_t effective_n_chunk_sz = 64 * (need_buf_a_ ? 4 : 1);
    const dim_t desired_N_chunk_size = is_runtime_N
            ? effective_n_chunk_sz
            : nstl::min(N, effective_n_chunk_sz);
    const float coef_M = nstl::min(
            static_cast<float>(m_chunk_elems_) / desired_M_chunk_size, 1.0f);
    const float coef_N = nstl::min(
            static_cast<float>(n_chunk_elems_) / desired_N_chunk_size, 1.0f);
    return 0.5f * (coef_M + coef_N);
}

// Returns score for current blocking parameters' values in range [0, 1]
// for L2 utilization
float matmul_amx_blocking_params_micro_t::get_L2_utilization_scores() const {
    const float relative_difference_with_L2
            = fabsf((float)L2_threshold() - blocking_chunk_mem_size_)
            / nstl::max(L2_threshold(), blocking_chunk_mem_size_);
    return 1.0f - relative_difference_with_L2;
}

// Returns score for current blocking parameters' values in range [0, 1]
// consists of 3 parts with its own weights:
// 	1) parallel work over threads distribution score
// 	2) L2 utilization score
// 	3) copied data re-usage score
float matmul_amx_blocking_params_micro_t::calculate_blocking_scores() const {
    if (one_of(0, n_blk_, n_chunk_size_, m_blk_, m_chunk_size_, k_blk_,
                brgemm_batch_size_))
        return 0.0f;

    const float nthr_coeff = static_cast<float>(nstl::min(nthr, 100));
    const float reusage_factor = 1.0f;
    // For runtume M the actual size is unknown, use independent on num_threads
    // balance factors
    const float balance_factor
            = is_runtime_M ? 1.0f : (nthr_coeff - 1.0f) / nthr_coeff;
    const float cache_utilization_factor
            = is_runtime_M ? 1.0f : 1.0f / nthr_coeff;

    float scores = cache_utilization_factor * get_L2_utilization_scores()
            + reusage_factor * get_copied_data_reusage_scores();
    if (balance_factor > 0.0f)
        scores += balance_factor * get_thread_balance_scores();
    return scores
            / (reusage_factor + balance_factor + cache_utilization_factor);
}

size_t matmul_amx_blocking_params_micro_t::calculate_chunk_memory_size() {
    update_k_blocking_dependent_params();

    const size_t A_chunk_sz = a_dt_sz * k_chunk_elems_ * m_chunk_elems_;
    const size_t A_buf_sz = need_buf_a_
            ? tr_a_dt_sz * current_lda_ * brgemm_batch_size_ * m_chunk_elems_
            : 0;
    const size_t B_chunk_sz = b_dt_sz * k_chunk_elems_ * n_chunk_elems_;
    const size_t B_buf_sz
            = use_buffer_b ? tr_b_dt_sz * n_blk_ * k_chunk_elems_ : 0;
    const size_t C_chunk_sz = c_dt_sz * m_chunk_elems_ * n_chunk_elems_;
    const size_t C_buf_sz = need_buf_c_
            ? effective_c_buf_dt_sz(nthr_k_) * m_chunk_elems_ * n_chunk_elems_
            : 0;
    return A_chunk_sz + A_buf_sz + B_chunk_sz + B_buf_sz + C_chunk_sz
            + C_buf_sz;
}

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
