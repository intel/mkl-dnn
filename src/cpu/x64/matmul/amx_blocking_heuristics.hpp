/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef CPU_X64_MATMUL_AMX_BLOCKING_HEURISTICS_HPP
#define CPU_X64_MATMUL_AMX_BLOCKING_HEURISTICS_HPP

#include "cpu/x64/matmul/brgemm_matmul_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {
struct layer_perf_characteristics_t {
    size_t a_size {0}, b_size {0}, d_size {0};
    float strip_1_size_shared {0}, strip_1_size_private {0},
            strip_1_share_coef {0};
    float strip_mid_size_shared {0}, strip_mid_size_private {0};
    float num_tmuls_per_strip {0}, strip_mid_share_coef {0}, num_strip {0},
            nt_mat_l1_miss {0};
    float l1_reuse {0};
    float num_postop_cache_lines {0};
    dim_t num_cycles_per_tmul {0};

    bool strip1_b_tranform_h {false};
    bool strips_b_tranform_v {false};
    bool strip1_b_in_mlc_h {false};

    bool strip1_a_read_v {false};
    bool strips_a_read_h {false};
};

class bw_map_t {
public:
    bw_map_t() = default;

    float get_bw(int x) const { return linear_interpolation(multicore_bw, x); }

    // All the following bandwidth measurements were taken on an
    // EMR machine with two NUMA domains, each containing 32 cores.

    // This BW is the BW for read/store when hitting the L1
    const float l1_load_hit_bw = (float)106.41;
    const float l1_store_hit_bw = l1_load_hit_bw;

    // This l1 BW is the BW for read when missing the L1
    const float l1_load_miss_bw = (float)(106.41 / 2.28);
    // This l1 BW is the BW for store when missing the L1
    const float l1_store_miss_bw = (float)(106.41 / 2.85);
    // LLC BW
    const float llc_bw = (float)6.0;

private:
    // This dictionary includes DRAM bandwidth for cores that share data.
    // The key represents the number of cores sharing, and the value is the bandwidth.
    const std::map<int, float> multicore_bw = {{32, 4.06f}, {16, 3.31f},
            {8, 2.98f}, {4, 2.39f}, {2, 0.9f}, {1, 2.28f}};

    float linear_interpolation(
            const std::map<int, float> &points, int x) const {
        // Find the interval [x0, x1] where x0 <= x <= x1
        auto it = points.lower_bound(x);
        if (it == points.end()) {
            return points.rbegin()
                    ->second; // x is greater than the largest x in the map
        }
        if (it == points.begin()) {
            return it->second; // x is less than the smallest x in the map
        }

        auto it1 = it;
        auto it0 = std::prev(it);

        int x0 = it0->first;
        float y0 = it0->second;
        int x1 = it1->first;
        float y1 = it1->second;

        // Perform linear interpolation
        return y0
                + (y1 - y0) * static_cast<float>(x - x0)
                / static_cast<float>(x1 - x0);
    }
};

class matmul_amx_blocking_params_t : public brgemm_matmul_conf_t {
public:
    matmul_amx_blocking_params_t(const brgemm_matmul_conf_t &bgmmc)
        : brgemm_matmul_conf_t(bgmmc)
        , nthr_m_(nstl::max(nthr_m, 1))
        , nthr_n_(nstl::max(nthr_n, 1))
        , nthr_k_(nstl::max(nthr_k, 1))
        , nthr_b_(nstl::max(nthr_b, 1))
        , nthr_mnb_(nthr / nthr_k_)
        , nthr_(nthr_mnb_ * nthr_k_)
        , n_blk_(N_blk)
        , n_chunk_size_(N_chunk_size)
        , n_chunk_elems_(n_blk_ * n_chunk_size_)
        , m_blk_(M_blk)
        , m_chunk_size_(M_chunk_size)
        , m_chunk_elems_(m_blk_ * m_chunk_size_)
        , k_blk_(K_blk)
        , k_chunk_size_(K_chunk_size)
        , k_chunk_elems_(k_blk_ * k_chunk_size_ * brgemm_batch_size)
        , is_a_nt_(is_a_nt)
        , is_b_nt_(is_b_nt)
        , set_nt_(set_nt)
        , need_prefetch_a_(need_prefetch_a)
        , need_prefetch_b_(need_prefetch_b)
        , use_fused_copy_a_(use_fused_copy_a)
        , brgemm_batch_size_(brgemm_batch_size)
        , current_lda_(LDA)
        , need_buf_c_(use_buffer_c)
        , need_buf_a_(use_buffer_a)
        , extendable_k_(extendable_k)
        , blocking_chunk_mem_size_(0)
        , efficiency_score_(0.0f) {}

    void update_configuration(brgemm_matmul_conf_t &bgmmc) const;
    float get_blocking_scores() const { return efficiency_score_; }

    static size_t L1_threshold();
    static size_t L2_threshold();
    static size_t L2_ways_threshold();

protected:
    virtual float calculate_blocking_scores() const = 0;
    virtual dim_t get_actual_lda() const;
    // Size of one C buffer element as seen by the scratchpad footprint.
    dim_t effective_c_buf_dt_sz(int nthr_k_cand) const;
    // Whether scoring may price C in dst dt instead of the accumulator dt.
    bool narrow_c_scoring_active() const;

    // Num threads for parallelism wrt K dimension
    int nthr_m_ {0}, nthr_n_ {0}, nthr_k_ {0}, nthr_b_ {0};
    // Num threads for parallelism wrt M, N and batch dimensions
    int nthr_mnb_ {0};
    int nthr_ {0};
    dim_t n_blk_ {0}, n_chunk_size_ {0}, n_chunk_elems_ {0};
    dim_t m_blk_ {0}, m_chunk_size_ {0}, m_chunk_elems_ {0};
    dim_t k_blk_ {0}, k_chunk_size_ {0}, k_chunk_elems_ {0};

    bool is_a_nt_ {true}, is_b_nt_ {true};
    bool set_nt_ {false};
    bool need_prefetch_a_ {false}, need_prefetch_b_ {false};
    bool use_fused_copy_a_ {false};
    dim_t brgemm_batch_size_ {0};
    dim_t current_lda_ {0};
    bool need_buf_c_ {false}, need_buf_a_ {false};
    bool extendable_k_ {false};
    size_t blocking_chunk_mem_size_ {0};
    float efficiency_score_ {0.0f};
    static constexpr float avx_ipc {1.2f};
    bool is_buffer_c_required() const;
};

class matmul_amx_blocking_params_macro_t : public matmul_amx_blocking_params_t {
public:
    matmul_amx_blocking_params_macro_t(const brgemm_matmul_conf_t &bgmmc)
        : matmul_amx_blocking_params_t(bgmmc) {
        assert(bgmmc.tr_a_dt_sz == bgmmc.tr_b_dt_sz);
        gemm_dt_sz = bgmmc.tr_a_dt_sz;
        min_k_elem = matmul_amx_blocking_params_macro_t::min_k_dim / gemm_dt_sz;
        min_mn_elem = matmul_amx_blocking_params_macro_t::min_mn_dim
                / bgmmc.c_dt_sz;
        k_threshold_write_bound_layer_elem
                = matmul_amx_blocking_params_macro_t::
                          k_threshold_write_bound_layer
                / gemm_dt_sz;
        min_n_dim_write_bound_layer_elem = matmul_amx_blocking_params_macro_t::
                                                   min_n_dim_write_bound_layer
                / gemm_dt_sz;
    }
    static bool is_supported(const brgemm_matmul_conf_t &bgmmc,
            const brgemm_matmul_conf_utils_t &bm_conf_utils);
    static bool find_best_blocking(const brgemm_matmul_conf_t &bgmmc,
            const brgemm_matmul_conf_utils_t &bm_conf_utils,
            matmul_amx_blocking_params_macro_t &best_blocking);
    static bool maybe_small_dims_heuristics(const brgemm_matmul_conf_t &bgmmc,
            matmul_amx_blocking_params_macro_t &best_blocking);

protected:
    float calculate_blocking_scores() const override;

private:
    static const dim_t min_k_dim = 256;
    static const dim_t min_mn_dim = 64;
    static const dim_t k_threshold_write_bound_layer = 256;
    static const dim_t min_n_dim_write_bound_layer = 256;

    static const dim_t ace_m_tmul = 16;
    static const dim_t ace_m_tiles = 2;
    static const dim_t ace_n_tiles = 4;
    static const dim_t ace_m_decomposition = ace_m_tiles * ace_m_tmul;

    dim_t n_decomposition = 32;
    dim_t m_decomposition = 32;
    size_t gemm_dt_sz {};
    dim_t m_per_thread {}, k_per_thread {}, n_per_thread {}, b_per_thread {};
    bool is_horizontal {};
    dim_t min_k_elem {}, min_mn_elem {};
    dim_t k_threshold_write_bound_layer_elem {},
            min_n_dim_write_bound_layer_elem {};

    size_t m_tmul {}, n_tmul {}, k_tmul {};
    bool set_blocking_parameters(bool force_horizontal = false,
            bool force_transform_matrix_to_l2 = false);
    bool is_horizontal_selected(bool horizontal_not_possible,
            bool vertical_not_possible, size_t best_m_v, size_t best_k_v,
            size_t k_blk_v) const;
    void set_tmul_sizes();
    void set_decomposition();
    size_t l2_matrix_usage(size_t k_chunk_size, size_t m_or_n_blk, size_t k_blk,
            bool is_horizontal,
            bool force_transform_matrix_to_l2 = false) const;
    size_t l2_matrix_and_c_usage(size_t k_chunk_size, size_t m_or_n_blk,
            size_t k_blk, bool is_horizontal) const;
    void set_core_divs(int nthr_b, int nthr_m, int nthr_k, int nthr_n);
    size_t bw(size_t m_blk, size_t k_chunk_size, size_t k_blk, size_t n_blk,
            bool is_horizontal) const;
    size_t compute(size_t m_blk, size_t k_chunk_size, size_t k_blk,
            size_t n_blk) const;
    float ratio(size_t m_blk, size_t k_chunk_size, size_t k_blk, size_t n_blk,
            bool is_horizontal) const;
    std::set<dim_t> blk_candidates(
            dim_t dim_per_thread, dim_t decomposition) const;
    float evaluate_single_core_blocking(size_t k_chunk_size, size_t m_or_n_blk,
            size_t k_blk, bool is_horizontal,
            bool force_transform_matrix_to_l2 = false) const;
    dim_t calc_k_blk(size_t l1_dim) const;
    bool divs_are_acceptable() const;
    bool operator==(const matmul_amx_blocking_params_macro_t &other) const;
    bool operator>(const matmul_amx_blocking_params_macro_t &other) const;
    bool operator!=(const matmul_amx_blocking_params_macro_t &other) const;
    bool operator<(const matmul_amx_blocking_params_macro_t &other) const;
    bool skip_extendable_k() const;
    bool b_transform_fits_in_l2() const;

    void init_copy_characteristics(
            layer_perf_characteristics_t &layer_perf_characteristics) const;
    void calculate_layer_sizes(
            layer_perf_characteristics_t &layer_perf_characteristics) const;
    float calculate_strip_mid_cycles(
            layer_perf_characteristics_t &layer_perf_characteristics,
            bw_map_t &bw_interpolator) const;
    float calculate_strip_1_cycles(
            layer_perf_characteristics_t &layer_perf_characteristics,
            bw_map_t &bw_interpolator, float strip_mid_cycles) const;
    float calculate_reduction_cycles(
            layer_perf_characteristics_t &layer_perf_characteristics,
            bw_map_t &bw_interpolator) const;
    float calculate_avx_insts_per_cache_line(
            layer_perf_characteristics_t &layer_perf_characteristics) const;
};

class matmul_amx_blocking_params_micro_t : public matmul_amx_blocking_params_t {
public:
    matmul_amx_blocking_params_micro_t(const brgemm_matmul_conf_t &bgmmc)
        : matmul_amx_blocking_params_t(bgmmc) {}

    void set_blocking_parameters(int nthr_k, dim_t n_blk, dim_t n_chunk_size,
            dim_t m_blk, dim_t m_chunk_size);

    static void find_best_blocking(const brgemm_matmul_conf_t &bgmmc,
            const brgemm_matmul_conf_utils_t &bm_conf_utils,
            matmul_amx_blocking_params_micro_t &best_blocking);

protected:
    float calculate_blocking_scores() const override;

private:
    float get_thread_balance_scores() const;
    void update_k_blocking_dependent_params();
    size_t calculate_chunk_memory_size();
    float get_copied_data_reusage_scores() const;
    float get_L2_utilization_scores() const;
};

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
