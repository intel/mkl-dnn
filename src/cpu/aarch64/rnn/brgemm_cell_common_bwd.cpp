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

#include "cpu/aarch64/rnn/brgemm_cell_common_bwd.hpp"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;

template <typename weights_t, typename scratch_t, typename gemm_acc_t>
brgemm_diff_src_layer_iter_t<weights_t, scratch_t,
        gemm_acc_t>::brgemm_diff_src_layer_iter_t(const ref_rnn_brgemm_t
                                                          &rnn_brgemm,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, scratch_t *scratch_gates,
        weights_t *w_iter, weights_t *w_layer, gemm_acc_t *diff_src_iter,
        gemm_acc_t *diff_src_layer, gemm_acc_t *gemm_acc_scratchpad,
        brgemm_batch_element_t *addr_batch_global)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , A_(scratch_gates)
    , B_wei_iter_(w_iter)
    , B_wei_layer_(w_layer)
    , C_diff_iter_(diff_src_iter)
    , C_diff_layer_(diff_src_layer)
    , k_blocks_n_gates_(rnn.diff_src_brgemm.K_blocks)
    , k_blocks_(rnn.diff_src_brgemm.K_blocks / rnn.n_gates)
    , k_tail_(rnn.diff_src_brgemm.k_tail)
    , k_block_(rnn.diff_src_brgemm.k_block)
    , A_k_tail_offset_(k_blocks_ * k_block_)
    , B_k_tail_offset_(A_k_tail_offset_ * rnn.diff_src_brgemm.n_block)
    , B_nb_offset_(rnn.diff_src_brgemm.Kpadded * rnn.diff_src_brgemm.n_block)
    , B_kb_offset_(k_block_ * rnn.diff_src_brgemm.n_block)
    , B_gb_iter_offset_(rnn.diff_src_brgemm.Kpadded
              * rnn.diff_src_brgemm.n_block * rnn.diff_src_brgemm.N_iter_blocks)
    , B_gb_layer_offset_(rnn.diff_src_brgemm.Kpadded
              * rnn.diff_src_brgemm.n_block
              * rnn.diff_src_brgemm.N_layer_blocks)
    , LDA_(rnn.diff_src_brgemm.LDA)
    , LDC_(rnn.diff_src_brgemm.LDC)
    , max_nthr_(nstl::min(dnnl_get_current_num_threads(), rnn.nthr))
    , n_blocking_(rnn.diff_src_brgemm.N_blocks)
    , m_blocking_(rnn.diff_src_brgemm.M_blocks)
    , work_amount_(n_blocking_ * m_blocking_)
    , max_n_layer_blocks_(rnn.diff_src_brgemm.N_layer_blocks)
    , max_n_iter_blocks_(rnn.diff_src_brgemm.N_iter_blocks)
    , gemm_layer_needed_(rnn.need_gemm_layer(cell_position))
    , kernel_iter_full_blocks_b0_(
              rnn_brgemm_.diff_src_.kernel_iter_layer_beta0_.get())
    , kernel_iter_n_tail_b0_(
              rnn_brgemm_.diff_src_.kernel_iter_N_tail_beta0_.get())
    , kernel_iter_k_tail_(
              rnn_brgemm_.diff_src_.kernel_iter_layer_K_tail_beta1_.get())
    , kernel_iter_nk_tail_(
              rnn_brgemm_.diff_src_.kernel_iter_NK_tail_beta1_.get())
    , kernel_layer_full_blocks_b0_(
              rnn_brgemm_.diff_src_.kernel_iter_layer_beta0_.get())
    , kernel_layer_n_tail_b0_(
              rnn_brgemm_.diff_src_.kernel_layer_N_tail_beta0_.get())
    , kernel_layer_k_tail_(
              rnn_brgemm_.diff_src_.kernel_iter_layer_K_tail_beta1_.get())
    , kernel_layer_nk_tail_(
              rnn_brgemm_.diff_src_.kernel_layer_NK_tail_beta1_.get())
    , gemm_acc_scratchpad_(gemm_acc_scratchpad)
    , addr_batch_global_(addr_batch_global) {}

template <typename weights_t, typename scratch_t, typename gemm_acc_t>
void brgemm_diff_src_layer_iter_t<weights_t, scratch_t, gemm_acc_t>::execute()
        const {
    parallel(max_nthr_, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

template <typename weights_t, typename scratch_t, typename gemm_acc_t>
void brgemm_diff_src_layer_iter_t<weights_t, scratch_t, gemm_acc_t>::kernel(
        const int ithr, const int nthr) const {

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    int n_block_id = 0, m_block_id = 0;
    nd_iterator_init(start, n_block_id, n_blocking_, m_block_id, m_blocking_);

    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * (k_blocks_n_gates_ + 1);

    const auto n_gates = rnn_.n_gates;

    while (start < end) {
        const int m = m_block_id * rnn_.diff_src_brgemm.m_block;
        const int n = n_block_id * rnn_.diff_src_brgemm.n_block;

        const scratch_t *const A_m = A_ + m * LDA_;
        const auto B_n_offset = n_block_id * B_nb_offset_;
        const weights_t *const B_wei_iter_n = B_wei_iter_ + B_n_offset;
        const weights_t *const B_wei_layer_n = B_wei_layer_ + B_n_offset;

        const auto C_offset = m * LDC_ + n;
        gemm_acc_t *const C_diff_iter_n = C_diff_iter_ + C_offset;
        gemm_acc_t *const C_diff_layer_n = C_diff_layer_ + C_offset;

        const brgemm_kernel_t *kernel_iter = kernel_iter_full_blocks_b0_;
        const brgemm_kernel_t *kernel_iter_k_tail = kernel_iter_k_tail_;
        const brgemm_kernel_t *kernel_layer = kernel_layer_full_blocks_b0_;
        const brgemm_kernel_t *kernel_layer_k_tail = kernel_layer_k_tail_;

        const bool should_calc_diff_src_layer
                = gemm_layer_needed_ && n_block_id < max_n_layer_blocks_;
        const bool should_calc_diff_src_iter = n_block_id < max_n_iter_blocks_;

        if (should_calc_diff_src_iter) {
            const bool do_n_iter_tail = (n + rnn_.diff_src_brgemm.n_block)
                    > rnn_.diff_src_brgemm.N_iter;
            if (do_n_iter_tail) {
                kernel_iter = kernel_iter_n_tail_b0_;
                kernel_iter_k_tail = kernel_iter_nk_tail_;
            }

            for (int gate_id = 0; gate_id < n_gates; gate_id++) {
                const auto g_block_id = gate_id * k_blocks_;
                const auto A_gb_offset = gate_id * rnn_.diff_src_brgemm.K;
                const auto B_gb_offset = gate_id * B_gb_iter_offset_;
                const auto A_gm = A_m + A_gb_offset;
                const auto B_wei_iter_gn = B_wei_iter_n + B_gb_offset;

                for (int k_block_id = 0; k_block_id < k_blocks_; k_block_id++) {
                    addr_batch[g_block_id + k_block_id].ptr.A
                            = A_gm + k_block_id * k_block_;
                    addr_batch[g_block_id + k_block_id].ptr.B
                            = B_wei_iter_gn + k_block_id * B_kb_offset_;
                }
            }
            brgemm_kernel_execute(kernel_iter, k_blocks_n_gates_, addr_batch,
                    reinterpret_cast<void *>(C_diff_iter_n), nullptr);
        }

        if (should_calc_diff_src_layer) {
            const bool do_n_layer_tail = (n + rnn_.diff_src_brgemm.n_block)
                    > rnn_.diff_src_brgemm.N_layer;
            if (do_n_layer_tail) {
                kernel_layer = kernel_layer_n_tail_b0_;
                kernel_layer_k_tail = kernel_layer_nk_tail_;
            }

            for (int gate_id = 0; gate_id < n_gates; gate_id++) {
                const auto g_block_id = gate_id * k_blocks_;
                const auto A_gb_offset = gate_id * rnn_.diff_src_brgemm.K;
                const auto B_gb_offset = gate_id * B_gb_layer_offset_;
                const auto A_gm = A_m + A_gb_offset;
                const auto B_wei_layer_gn = B_wei_layer_n + B_gb_offset;

                for (int k_block_id = 0; k_block_id < k_blocks_; k_block_id++) {
                    addr_batch[g_block_id + k_block_id].ptr.A
                            = A_gm + k_block_id * k_block_;
                    addr_batch[g_block_id + k_block_id].ptr.B
                            = B_wei_layer_gn + k_block_id * B_kb_offset_;
                }
            }
            brgemm_kernel_execute(kernel_layer, k_blocks_n_gates_, addr_batch,
                    reinterpret_cast<void *>(C_diff_layer_n), nullptr);
        }

        if (should_calc_diff_src_iter && k_tail_) {
            for (int gate_id = 0; gate_id < n_gates; gate_id++) {
                const auto A_gb_offset = gate_id * rnn_.diff_src_brgemm.K;
                const auto B_gb_offset = gate_id * B_gb_iter_offset_;
                addr_batch[gate_id].ptr.A
                        = A_m + A_gb_offset + A_k_tail_offset_;
                addr_batch[gate_id].ptr.B
                        = B_wei_iter_n + B_gb_offset + B_k_tail_offset_;
            }
            brgemm_kernel_execute(kernel_iter_k_tail, n_gates, addr_batch,
                    reinterpret_cast<void *>(C_diff_iter_n), nullptr);
        }

        if (should_calc_diff_src_layer && k_tail_) {
            for (int gate_id = 0; gate_id < n_gates; gate_id++) {
                const auto A_gb_offset = gate_id * rnn_.diff_src_brgemm.K;
                const auto B_gb_offset = gate_id * B_gb_layer_offset_;
                addr_batch[gate_id].ptr.A
                        = A_m + A_gb_offset + A_k_tail_offset_;
                addr_batch[gate_id].ptr.B
                        = B_wei_layer_n + B_gb_offset + B_k_tail_offset_;
            }
            brgemm_kernel_execute(kernel_layer_k_tail, n_gates, addr_batch,
                    reinterpret_cast<void *>(C_diff_layer_n), nullptr);
        }

        ++start;
        nd_iterator_step(n_block_id, n_blocking_, m_block_id, m_blocking_);
    }
}

template <typename src_layer_t, typename src_iter_t, typename scratch_t,
        typename gemm_acc_t>
brgemm_diff_weights_layer_iter_t<src_layer_t, src_iter_t, scratch_t,
        gemm_acc_t>::brgemm_diff_weights_layer_iter_t(const ref_rnn_brgemm_t
                                                              &rnn_brgemm,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const src_layer_t *src_iter,
        scratch_t *const A_iter_transposed_scratch, const src_iter_t *src_layer,
        scratch_t *const A_layer_transposed_scratch, const scratch_t *scratch,
        scratch_t *scratch_gates_blocked, gemm_acc_t *diff_weights_iter,
        gemm_acc_t *diff_weights_layer, gemm_acc_t *diff_bias,
        gemm_acc_t *gemm_acc_scratchpad,
        brgemm_batch_element_t *addr_batch_global)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , A_iter_(src_iter)
    , A_iter_transposed_scratch_(A_iter_transposed_scratch)
    , A_layer_(src_layer)
    , A_layer_transposed_scratch_(A_layer_transposed_scratch)
    , B_(scratch)
    , B_blocked_scratch_(scratch_gates_blocked)
    , C_iter_(diff_weights_iter)
    , C_layer_(diff_weights_layer)
    , diff_bias_(diff_bias)
    , LDA_iter_(rnn.diff_wei_brgemm.LDA_iter)
    , LDA_layer_(rnn.diff_wei_brgemm.LDA_layer)
    , LDC_iter_(rnn.diff_wei_brgemm.LDC_iter)
    , LDC_layer_(rnn.diff_wei_brgemm.LDC_layer)
    , max_nthr_(nstl::min(dnnl_get_current_num_threads(), rnn.nthr))
    , n_blocking_(rnn.diff_wei_brgemm.N_blocks)
    , m_blocking_(rnn.diff_wei_brgemm.M_blocks)
    , k_blocks_(rnn.diff_wei_brgemm.K_blocks)
    , k_tail_(rnn.diff_wei_brgemm.k_tail)
    , k_block_(rnn.diff_wei_brgemm.k_block)
    , m_iter_block_(rnn.slc == rnn.sic ? rnn.diff_wei_brgemm.m_block
                                       : rnn.diff_wei_brgemm.M_iter)
    , m_layer_block_(rnn.slc == rnn.sic ? rnn.diff_wei_brgemm.m_block
                                        : rnn.diff_wei_brgemm.M_layer)
    , A_k_iter_tail_offset_(k_blocks_ * k_block_)
    , A_k_layer_tail_offset_(k_blocks_ * k_block_)
    , B_kb_offset_(k_block_ * rnn.diff_wei_brgemm.n_block)
    , B_k_tail_offset_(k_blocks_ * k_block_ * rnn.scratch_gates_ld)
    , B_k_tail_offset_blocked_(
              k_blocks_ * k_block_ * rnn.diff_wei_brgemm.n_block)
    , work_amount_(n_blocking_ * m_blocking_)
    , kernel_iter_full_blocks_(rnn_brgemm.diff_wei_.kernel_iter_beta1_.get())
    , kernel_iter_n_tail_(rnn_brgemm.diff_wei_.kernel_iter_N_tail_beta1_.get())
    , kernel_iter_k_tail_(rnn_brgemm.diff_wei_.kernel_iter_K_tail_beta1_.get())
    , kernel_iter_nk_tail_(
              rnn_brgemm.diff_wei_.kernel_iter_NK_tail_beta1_.get())
    , kernel_layer_full_blocks_(rnn_brgemm.diff_wei_.kernel_layer_beta1_.get())
    , kernel_layer_n_tail_(
              rnn_brgemm.diff_wei_.kernel_layer_N_tail_beta1_.get())
    , kernel_layer_k_tail_(
              rnn_brgemm.diff_wei_.kernel_layer_K_tail_beta1_.get())
    , kernel_layer_nk_tail_(
              rnn_brgemm.diff_wei_.kernel_layer_NK_tail_beta1_.get())
    , cell_position_(cell_position)
    , gemm_acc_scratchpad_(gemm_acc_scratchpad)
    , addr_batch_global_(addr_batch_global) {}

template <typename src_layer_t, typename src_iter_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_diff_weights_layer_iter_t<src_layer_t, src_iter_t, scratch_t,
        gemm_acc_t>::execute() const {
    parallel(max_nthr_, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

template <typename src_layer_t, typename src_iter_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_diff_weights_layer_iter_t<src_layer_t, src_iter_t, scratch_t,
        gemm_acc_t>::reorder_scratch_gates(const scratch_t *src, scratch_t *dst,
        const bool do_n_tail) const {

    assert(!"AArch64 JIT scratch gates reorder not available in Phase A");
}

template <typename src_layer_t, typename src_iter_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_diff_weights_layer_iter_t<src_layer_t, src_iter_t, scratch_t,
        gemm_acc_t>::kernel(const int ithr, const int nthr) const {

    const bool global_transpose = rnn_.diff_wei_brgemm.global_transpose;

    scratch_t *const B_blocked = B_blocked_scratch_
            + ithr * rnn_.diff_wei_brgemm.Kpadded
                    * rnn_.diff_wei_brgemm.n_block;

    scratch_t *const A_iter_transposed_ithr = global_transpose
            ? A_iter_transposed_scratch_
            : (A_iter_transposed_scratch_
                      + ithr * rnn_.diff_wei_brgemm.Kpadded * m_iter_block_);

    scratch_t *const A_layer_transposed_ithr = global_transpose
            ? A_layer_transposed_scratch_
            : A_layer_transposed_scratch_
                    + ithr * rnn_.diff_wei_brgemm.Kpadded * m_layer_block_;

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    int n_block_id = 0, m_block_id = 0, last_n_block_id = -1,
        last_m_block_id = -1;
    nd_iterator_init(start, n_block_id, n_blocking_, m_block_id, m_blocking_);

    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * (k_blocks_ + 1);

    while (start < end) {
        const bool should_reorder_gates = last_n_block_id != n_block_id;
        const bool transpose_needed
                = !(rnn_.mb == 1 && std::is_same<float, src_iter_t>::value);
        const bool should_transpose_src = transpose_needed && !global_transpose
                && (last_m_block_id != m_block_id);

        const int m_iter = m_block_id * m_iter_block_;
        const int m_layer = m_block_id * m_layer_block_;

        const src_iter_t *const A_iter_m = global_transpose
                ? A_iter_transposed_ithr + m_iter * LDA_iter_
                : A_iter_ + m_iter;
        const src_layer_t *const A_layer_m = global_transpose
                ? A_layer_transposed_ithr + m_layer * LDA_layer_
                : A_layer_ + m_layer;

        src_iter_t *const A_iter_transposed
                = (global_transpose || !transpose_needed)
                ? const_cast<src_iter_t *>(A_iter_m)
                : A_iter_transposed_ithr;
        src_layer_t *const A_layer_transposed
                = (global_transpose || !transpose_needed)
                ? const_cast<src_layer_t *>(A_layer_m)
                : A_layer_transposed_ithr;

        const int n = n_block_id * rnn_.diff_wei_brgemm.n_block;
        const scratch_t *const B_n = B_ + n;
        const auto C_iter_offset = m_iter * LDC_iter_ + n;
        const auto C_layer_offset = m_layer * LDC_layer_ + n;
        gemm_acc_t *const C_diff_iter_n = C_iter_ + C_iter_offset;
        gemm_acc_t *const C_diff_layer_n = C_layer_ + C_layer_offset;

        const bool do_n_tail
                = (n + rnn_.diff_wei_brgemm.n_block) > rnn_.diff_wei_brgemm.N;

        const brgemm_kernel_t *kernel_iter = kernel_iter_full_blocks_;
        const brgemm_kernel_t *kernel_iter_k_tail = kernel_iter_k_tail_;
        const brgemm_kernel_t *kernel_layer = kernel_layer_full_blocks_;
        const brgemm_kernel_t *kernel_layer_k_tail = kernel_layer_k_tail_;

        if (do_n_tail) {
            kernel_iter = kernel_iter_n_tail_;
            kernel_iter_k_tail = kernel_iter_nk_tail_;
            kernel_layer = kernel_layer_n_tail_;
            kernel_layer_k_tail = kernel_layer_nk_tail_;
        }

        if (should_reorder_gates) {
            reorder_scratch_gates(B_n, B_blocked, do_n_tail);

            if (m_block_id == 0) {

                // JIT gates reduction not yet implemented
                // TODO: Add aarch64 jit_gates_reduction_t
                assert(!"JIT gates reduction kernel not available till now");
            }
        }

        for (int k_block_id = 0; k_block_id < k_blocks_; k_block_id++) {
            addr_batch[k_block_id].ptr.A
                    = A_iter_transposed + k_block_id * k_block_;
            addr_batch[k_block_id].ptr.B
                    = B_blocked + k_block_id * B_kb_offset_;
        }

        if (should_transpose_src) {

            // JIT transpose not yet implemented
            // TODO: Add aarch64 jit_brgemm_transpose_single_row_t
            assert(!"JIT transpose iter kernel not available till now");
        }

        brgemm_kernel_execute(kernel_iter, k_blocks_, addr_batch,
                reinterpret_cast<void *>(C_diff_iter_n), nullptr);

        for (int k_block_id = 0; k_block_id < k_blocks_; k_block_id++) {
            addr_batch[k_block_id].ptr.A
                    = A_layer_transposed + k_block_id * k_block_;
            addr_batch[k_block_id].ptr.B
                    = B_blocked + k_block_id * B_kb_offset_;
        }

        if (should_transpose_src) {

            // JIT transpose not yet implemented
            // TODO: Add aarch64 jit_brgemm_transpose_single_row_t
            assert(!"JIT transpose layer kernel not available till now");
        }

        brgemm_kernel_execute(kernel_layer, k_blocks_, addr_batch,
                reinterpret_cast<void *>(C_diff_layer_n), nullptr);

        if (k_tail_) {
            const auto B_blocked_k_tail = B_blocked + B_k_tail_offset_blocked_;

            addr_batch[0].ptr.A = A_iter_transposed + A_k_iter_tail_offset_;
            addr_batch[0].ptr.B = B_blocked_k_tail;
            brgemm_kernel_execute(kernel_iter_k_tail, 1, addr_batch,
                    reinterpret_cast<void *>(C_diff_iter_n), nullptr);

            addr_batch[0].ptr.A = A_layer_transposed + A_k_layer_tail_offset_;
            addr_batch[0].ptr.B = B_blocked_k_tail;
            brgemm_kernel_execute(kernel_layer_k_tail, 1, addr_batch,
                    reinterpret_cast<void *>(C_diff_layer_n), nullptr);
        }

        if (should_reorder_gates) { last_n_block_id = n_block_id; }
        if (should_transpose_src) { last_m_block_id = m_block_id; }

        ++start;
        nd_iterator_step(n_block_id, n_blocking_, m_block_id, m_blocking_);
    }
}

template <typename scratch_t>
brgemm_diff_wei_peep_t<scratch_t>::brgemm_diff_wei_peep_t(
        const ref_rnn_brgemm_t &rnn_brgemm, const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position,
        const scratch_t *scratch_gates, const void *src_iter_c,
        const void *dst_iter_c, float *diff_weights_peephole)
    : rnn_(rnn)
    , scratch_gates_(scratch_gates)
    , src_iter_c_(src_iter_c)
    , dst_iter_c_(dst_iter_c)
    , diff_weights_peephole_(diff_weights_peephole)
    , work_amount_(n_gates_ * rnn_.dhc_blocks_peephole)
    , dst_iter_c_ld_(rnn.dst_iter_c_ld(cell_position))
    , src_iter_c_ld_(rnn.src_iter_c_ld(cell_position)) {}

template <typename scratch_t>
void brgemm_diff_wei_peep_t<scratch_t>::execute() const {
    parallel(rnn_.nthr, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

template <typename scratch_t>
void brgemm_diff_wei_peep_t<scratch_t>::kernel(
        const int ithr, const int nthr) const {

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    int g = 0, dhc_block_id = 0;
    nd_iterator_init(
            start, g, n_gates_, dhc_block_id, rnn_.dhc_blocks_peephole);

    const auto dst_iter_c = rnn_utils::make_raw_aoc(dst_iter_c_,
            types::data_type_size(rnn_.dst_iter_c_dt),
            rnn_.ws_states_iter_c_nld, dst_iter_c_ld_);
    const auto src_iter_c = rnn_utils::make_raw_aoc(src_iter_c_,
            types::data_type_size(rnn_.src_iter_c_dt),
            rnn_.ws_states_iter_c_nld, src_iter_c_ld_);

    const rnn_utils::scratch_gates_aoc_t<const scratch_t> scratch_gates(
            rnn_, scratch_gates_);
    const rnn_utils::weights_peephole_aoc_t<float> diff_weights_peephole(
            rnn_, diff_weights_peephole_);

    MAYBE_UNUSED(dst_iter_c);
    MAYBE_UNUSED(src_iter_c);
    MAYBE_UNUSED(scratch_gates);
    MAYBE_UNUSED(diff_weights_peephole);

    while (start < end) {

        // JIT peephole not yet implemented
        // TODO: Add aarch64 jit_diff_weights_peephole_t
        assert(!"JIT peephole kernel not available till now");

        ++start;
        nd_iterator_step(g, n_gates_, dhc_block_id, rnn_.dhc_blocks_peephole);
    }
}

template class brgemm_diff_src_layer_iter_t<float, float, float>;
template class brgemm_diff_src_layer_iter_t<bfloat16_t, bfloat16_t, float>;
template class brgemm_diff_src_layer_iter_t<float16_t, float16_t, float>;

template class brgemm_diff_weights_layer_iter_t<float, float, float, float>;
template class brgemm_diff_weights_layer_iter_t<bfloat16_t, bfloat16_t,
        bfloat16_t, float>;
template class brgemm_diff_weights_layer_iter_t<float16_t, float16_t, float16_t,
        float>;

template class brgemm_diff_wei_peep_t<bfloat16_t>;
template class brgemm_diff_wei_peep_t<float16_t>;
template class brgemm_diff_wei_peep_t<float>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
