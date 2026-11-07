/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2026 FUJITSU LIMITED
* Copyright 2026 Arm Ltd. and affiliates
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

#ifndef CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_FWD_HPP
#define CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_FWD_HPP

#include <functional>

#include "cpu/aarch64/rnn/rnn_brgemm_utils.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
class brgemm_dst_layer_iter_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;
    using postgemm_fused_t = std::function<void(
            dim_t, dim_t, dim_t, const src_t *, scratch_t *, scratch_t *, int)>;

    brgemm_dst_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const src_t *src_iter,
            const src_t *src_layer, weights_t *w_iter, weights_t *w_layer,
            scratch_t *scratch_gates, scratch_t *scratch_cell_,
            gemm_acc_t *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global,
            const postgemm_fused_t &fused_postgemm);

    void execute() const;

private:
    int calculate_nthr() const;
    void kernel(const int ithr, const int nthr) const;
    void kernel_fused_iter_layer(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;

    const bool need_gemm_layer_;

    const dim_t layer_desc_idx_;
    const dim_t iter_desc_idx_;

    const src_t *const A_layer_;
    const src_t *const A_iter_;

    const weights_t *const B_layer_;
    const weights_t *const B_iter_;

    scratch_t *const C_gates_;
    scratch_t *const C_cell_;

    const dim_t LDA_layer_;
    const dim_t LDA_iter_;

    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const int work_amount_;

    const dim_t B_layer_n_offset_;
    const dim_t B_iter_n_offset_;
    const dim_t B_layer_g_offset_;
    const dim_t B_iter_g_offset_;
    const dim_t A_layer_k_tail_offset_;
    const dim_t A_iter_k_tail_offset_;
    const dim_t B_layer_kb_offset_;
    const dim_t B_iter_kb_offset_;
    const dim_t B_layer_k_tail_offset_;
    const dim_t B_iter_k_tail_offset_;

    const dim_t n_gates_;

    const brgemm_kernel_t *const brgemm_kernel_iter_main_;
    const brgemm_kernel_t *const brgemm_kernel_iter_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_nk_tail_;

    const brgemm_kernel_t *const brgemm_kernel_layer_main_;
    const brgemm_kernel_t *const brgemm_kernel_layer_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_nk_tail_;

    gemm_acc_t *const gemm_acc_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;
    const postgemm_fused_t fused_postgemm_;

    const bool is_fused_layer_iter_brgemm_;

    const int max_nthr_;
};

template <typename src_t, typename weights_t, typename gemm_acc_t>
class brgemm_dst_proj_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;

    using postgemm_fused_t
            = std::function<void(dim_t, dim_t, gemm_acc_t *, int)>;

    brgemm_dst_proj_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const src_t *proj_ht,
            const weights_t *w_projection, gemm_acc_t *output,
            gemm_acc_t *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global,
            const postgemm_fused_t &fused_postgemm);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;

    const int proj_desc_idx_;

    const src_t *const A_;
    const weights_t *const B_;
    gemm_acc_t *const C_;

    const dim_t LDC_;
    const dim_t max_nthr_;
    const int work_amount_proj_;

    const dim_t B_n_offset_;
    const dim_t Bp_kb_offset_;

    gemm_acc_t *const gemm_acc_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;

    const brgemm_kernel_t *const brgemm_kernel_main_;
    const brgemm_kernel_t *const brgemm_kernel_n_tail_;

    const postgemm_fused_t fused_postgemm_;
};

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
class brgemm_gru_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;
    using postgemm_fused_t = std::function<void(
            dim_t, dim_t, dim_t, const src_t *, scratch_t *, scratch_t *, int)>;

    brgemm_gru_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const src_t *src_iter,
            const src_t *src_layer, weights_t *w_iter, weights_t *w_iter1,
            weights_t *w_layer, src_t *d_layer, scratch_t *scratch_gates,
            scratch_t *scratch_cell, gemm_acc_t *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global,
            const postgemm_fused_t &fused_postgemm_part1,
            const postgemm_fused_t &fused_postgemm_part2);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;

    const bool need_gemm_layer_;

    const dim_t layer_desc_idx_;
    const dim_t iter_desc_idx_;
    const dim_t iter_part2_desc_idx_;

    const src_t *const A_layer_;
    const src_t *const A_iter_;

    const weights_t *const B_layer_;
    const weights_t *const B_iter_;
    const weights_t *const B_iter_part2_;

    scratch_t *const C_gates_;
    scratch_t *const C_cell_;
    src_t *const D_layer_;

    const dim_t LDA_layer_;
    const dim_t LDA_iter_part1_;
    const dim_t LDA_iter_part2_;

    const dim_t max_nthr_;
    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const int work_amount_;

    const dim_t B_layer_n_offset_;
    const dim_t B_iter_n_offset_;
    const dim_t B_layer_g_offset_;
    const dim_t B_iter_g_offset_;
    const dim_t A_layer_k_tail_offset_;
    const dim_t A_iter_k_tail_offset_;
    const dim_t B_layer_kb_offset_;
    const dim_t B_iter_kb_offset_;
    const dim_t B_layer_k_tail_offset_;
    const dim_t B_iter_k_tail_offset_;

    const dim_t n_gates_;

    const brgemm_kernel_t *const brgemm_kernel_iter_p0_main_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p0_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p0_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p0_nk_tail_;

    const brgemm_kernel_t *const brgemm_kernel_iter_p1_main_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p1_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p1_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_iter_p1_nk_tail_;

    const brgemm_kernel_t *const brgemm_kernel_layer_main_;
    const brgemm_kernel_t *const brgemm_kernel_layer_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_nk_tail_;

    gemm_acc_t *const gemm_acc_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;

    const postgemm_fused_t fused_postgemm_part1_;
    const postgemm_fused_t fused_postgemm_part2_;

    const bool is_fused_layer_iter_brgemm_;
};

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
class brgemm_merged_layer_t {
public:
    using ref_rnn_brgemm_t = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::forward>;

    brgemm_merged_layer_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, const src_t *src_layer,
            weights_t *w_layer, scratch_t *scratch_gates,
            gemm_acc_t *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;

    const dim_t layer_desc_idx_;

    const src_t *const A_layer_;
    const weights_t *const B_layer_;
    scratch_t *const C_;

    const dim_t LDA_layer_;
    const dim_t max_nthr_;

    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const int work_amount_;

    const dim_t B_layer_n_offset_;
    const dim_t B_layer_g_offset_;
    const dim_t A_layer_k_tail_offset_;
    const dim_t B_layer_kb_offset_;
    const dim_t B_layer_k_tail_offset_;

    const brgemm_kernel_t *const brgemm_kernel_layer_main_;
    const brgemm_kernel_t *const brgemm_kernel_layer_n_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_k_tail_;
    const brgemm_kernel_t *const brgemm_kernel_layer_nk_tail_;

    gemm_acc_t *const gemm_acc_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_FWD_HPP
