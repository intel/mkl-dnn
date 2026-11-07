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

#ifndef CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_BWD_HPP
#define CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_BWD_HPP

#include "common/bfloat16.hpp"
#include "cpu/aarch64/rnn/rnn_brgemm_utils.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

/*
 * Calculations:
 * scratch * w_iter = dff_src_iter
 * A (mb x rnn.n_gates * rnn.dhc) * B (rnn.n_gates * rnn.dhc, rnn.sic) =
 * C (mb x rnn.sic)
 *
 * scratch * w_layer = dff_src_layer
 * A (mb x rnn.n_gates * rnn.dhc) * B (rnn.n_gates * rnn.dhc, rnn.slc) =
 * C (mb x rnn.slc)
 *
 * Data formats:
 * scratch = igo (mb, n_gates, rnn.dhc)
 * w_iter = gIo32i (n_gates, rnn.sic, rnn.dhc)
 * w_layer = gIo32i (n_gates, rnn.slc, rnn.dhc)
 * diff_src_layer = io (mb, rnn.slc)
 * diff_src_iter = io (mb, rnn.sic)
 */

template <typename weights_t, typename scratch_t, typename gemm_acc_t>
class brgemm_diff_src_layer_iter_t {
public:
    using ref_rnn_brgemm_t
            = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::backward>;

    brgemm_diff_src_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position, scratch_t *scratch_gates,
            weights_t *w_iter, weights_t *w_layer, gemm_acc_t *diff_src_iter,
            gemm_acc_t *diff_src_layer, gemm_acc_t *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;

    const scratch_t *const A_;

    const weights_t *const B_wei_iter_;
    const weights_t *const B_wei_layer_;

    gemm_acc_t *const C_diff_iter_;
    gemm_acc_t *const C_diff_layer_;

    const dim_t k_blocks_n_gates_;
    const dim_t k_blocks_;
    const dim_t k_tail_;
    const dim_t k_block_;

    const dim_t A_k_tail_offset_;
    const dim_t B_k_tail_offset_;
    const dim_t B_nb_offset_;
    const dim_t B_kb_offset_;
    const dim_t B_gb_iter_offset_;
    const dim_t B_gb_layer_offset_;

    const dim_t LDA_;
    const dim_t LDC_;
    const dim_t max_nthr_;

    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const int work_amount_;

    const dim_t max_n_layer_blocks_;
    const dim_t max_n_iter_blocks_;

    const bool gemm_layer_needed_;

    const brgemm_kernel_t *const kernel_iter_full_blocks_b0_;
    const brgemm_kernel_t *const kernel_iter_n_tail_b0_;
    const brgemm_kernel_t *const kernel_iter_k_tail_;
    const brgemm_kernel_t *const kernel_iter_nk_tail_;

    const brgemm_kernel_t *const kernel_layer_full_blocks_b0_;
    const brgemm_kernel_t *const kernel_layer_n_tail_b0_;
    const brgemm_kernel_t *const kernel_layer_k_tail_;
    const brgemm_kernel_t *const kernel_layer_nk_tail_;

    gemm_acc_t *const gemm_acc_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;
};

/*
 * Calculations:
 * src_layer^T * scratch = dff_weights_layer
 * A before transpose (rnn.mb, rnn.slc) - layout in memory
 * A (rnn.slc, rnn.mb) * B (rnn.mb, rnn.n_gates * rnn.dhc) =
 * C (rnn.slc, rnn.n_gates * rnn.dhc)
 * src_iter^T * scratch = dff_weights_iter
 * A (rnn.sic, rnn.mb) * B (rnn.mb, rnn.n_gates * rnn.dhc) =
 * C (rnn.sic, rnn.n_gates * rnn.dhc)
 *
 * Data formats:
 * src_iter  = io (mb,  rnn.sic) -> transposed oi (rnn.sic, mb)
 * src_layer = io (mb, rnn.slc) -> transposed oi (rnn.sic, mb)
 *
 * scratch = igo (mb, n_gates, rnn.dhc)
 *
 * dff_weights_iter = igo (rnn.sic, rnn.n_gates, rnn.dhc)
 * dff_weights_layer = igo (rnn.slc, rnn.n_gates, rnn.dhc)
 */

template <typename src_layer_t, typename src_iter_t, typename scratch_t,
        typename gemm_acc_t>
class brgemm_diff_weights_layer_iter_t {
public:
    using ref_rnn_brgemm_t
            = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::backward>;

    brgemm_diff_weights_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm_,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position,
            const src_layer_t *src_iter,
            scratch_t *const A_iter_transposed_scratch,
            const src_iter_t *src_layer,
            scratch_t *const A_layer_transposed_scratch,
            const scratch_t *scratch, scratch_t *scratch_gates_blocked,
            gemm_acc_t *diff_weights_iter, gemm_acc_t *diff_weights_layer,
            gemm_acc_t *diff_bias, gemm_acc_t *gemm_acc_scratchpad,
            brgemm_batch_element_t *addr_batch_global);

    void execute() const;

private:
    const ref_rnn_brgemm_t &rnn_brgemm_;
    const rnn_utils::rnn_conf_t &rnn_;

    const src_iter_t *const A_iter_;
    scratch_t *const A_iter_transposed_scratch_;
    const src_layer_t *const A_layer_;
    scratch_t *const A_layer_transposed_scratch_;

    const scratch_t *const B_;
    scratch_t *const B_blocked_scratch_;

    gemm_acc_t *const C_iter_;
    gemm_acc_t *const C_layer_;
    gemm_acc_t *const diff_bias_;

    const dim_t LDA_iter_;
    const dim_t LDA_layer_;
    const dim_t LDC_iter_;
    const dim_t LDC_layer_;

    const dim_t max_nthr_;

    const dim_t n_blocking_;
    const dim_t m_blocking_;
    const dim_t k_blocks_;
    const dim_t k_tail_;
    const dim_t k_block_;
    const dim_t m_iter_block_;
    const dim_t m_layer_block_;

    const dim_t A_k_iter_tail_offset_;
    const dim_t A_k_layer_tail_offset_;
    const dim_t B_kb_offset_;
    const dim_t B_k_tail_offset_;
    const dim_t B_k_tail_offset_blocked_;
    const int work_amount_;

    const brgemm_kernel_t *const kernel_iter_full_blocks_;
    const brgemm_kernel_t *const kernel_iter_n_tail_;
    const brgemm_kernel_t *const kernel_iter_k_tail_;
    const brgemm_kernel_t *const kernel_iter_nk_tail_;
    const brgemm_kernel_t *const kernel_layer_full_blocks_;
    const brgemm_kernel_t *const kernel_layer_n_tail_;
    const brgemm_kernel_t *const kernel_layer_k_tail_;
    const brgemm_kernel_t *const kernel_layer_nk_tail_;

    const rnn_utils::cell_position_t cell_position_;

    gemm_acc_t *const gemm_acc_scratchpad_;
    brgemm_batch_element_t *const addr_batch_global_;

    void kernel(const int ithr, const int nthr) const;
    void reorder_scratch_gates(
            const scratch_t *src, scratch_t *dst, const bool do_n_tail) const;
};

template <typename scratch_t>
class brgemm_diff_wei_peep_t {
public:
    using ref_rnn_brgemm_t
            = rnn_brgemm_utils::rnn_brgemm_t<prop_kind::backward>;

    brgemm_diff_wei_peep_t(const ref_rnn_brgemm_t &rnn_brgemm,
            const rnn_utils::rnn_conf_t &rnn,
            rnn_utils::cell_position_t cell_position,
            const scratch_t *scratch_gates, const void *src_iter_c,
            const void *dst_iter_c, float *diff_weights_peephole);

    void execute() const;

private:
    void kernel(const int ithr, const int nthr) const;

    const int n_gates_ = 3;
    const rnn_utils::rnn_conf_t &rnn_;
    const scratch_t *scratch_gates_;
    const void *src_iter_c_;
    const void *dst_iter_c_;
    float *diff_weights_peephole_;
    const int work_amount_;
    const int dst_iter_c_ld_;
    const int src_iter_c_ld_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_AARCH64_RNN_BRGEMM_CELL_COMMON_BWD_HPP
