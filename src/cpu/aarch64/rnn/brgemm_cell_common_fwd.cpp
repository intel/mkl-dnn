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

#include "cpu/aarch64/rnn/brgemm_cell_common_fwd.hpp"

#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t,
        gemm_acc_t>::brgemm_dst_layer_iter_t(const ref_rnn_brgemm_t &rnn_brgemm,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const src_t *src_iter,
        const src_t *src_layer, weights_t *w_iter, weights_t *w_layer,
        scratch_t *scratch_gates, scratch_t *scratch_cell,
        gemm_acc_t *gemm_acc_scratchpad,
        brgemm_batch_element_t *addr_batch_global,
        const postgemm_fused_t &fused_postgemm)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , need_gemm_layer_(rnn_.need_gemm_layer(cell_position))
    , layer_desc_idx_(rnn_.layer_brgemm_desc(cell_position))
    , iter_desc_idx_(rnn_.iter_brgemm_desc(cell_position))
    , A_layer_(src_layer)
    , A_iter_(src_iter)
    , B_layer_(w_layer)
    , B_iter_(w_iter)
    , C_gates_(scratch_gates)
    , C_cell_(scratch_cell)
    , LDA_layer_(rnn_.src_layer_ld(cell_position))
    , LDA_iter_(rnn_.src_iter_ld(cell_position))
    , n_blocking_((rnn_.unfused_post_gemm) ? rnn_.N_blocks * rnn_.n_gates
                                           : rnn_.N_blocks)
    , m_blocking_(rnn_.M_blocks)
    , work_amount_(n_blocking_ * m_blocking_)
    , B_layer_n_offset_(rnn_.K1padded * rnn_.n_block)
    , B_iter_n_offset_(rnn_.K2padded * rnn_.n_block)
    , B_layer_g_offset_(rnn_.N_blocks * B_layer_n_offset_)
    , B_iter_g_offset_(rnn_.N_blocks * B_iter_n_offset_)
    , A_layer_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block)
    , A_iter_k_tail_offset_(rnn_.KB2_blocks * rnn_.k2_block)
    , B_layer_kb_offset_(rnn_.k1_block * rnn_.n_block)
    , B_iter_kb_offset_(rnn_.k2_block * rnn_.n_block)
    , B_layer_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block * rnn_.n_block)
    , B_iter_k_tail_offset_(rnn_.KB2_blocks * rnn_.k2_block * rnn_.n_block)
    , n_gates_(rnn.unfused_post_gemm ? 1 : rnn.n_gates)
    , brgemm_kernel_iter_main_(
              rnn_brgemm_.kernel_iter_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_n_tail_(
              rnn_brgemm_.kernel_iter_N_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_k_tail_(
              rnn_brgemm_.kernel_iter_K2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_nk_tail_(
              rnn_brgemm_.kernel_iter_NK2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_layer_main_(
              rnn_brgemm_.kernel_layer_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_n_tail_(
              rnn_brgemm_.kernel_layer_N_tail_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_k_tail_(
              rnn_brgemm_.kernel_layer_K1_tail_b1_[layer_desc_idx_].get())
    , brgemm_kernel_layer_nk_tail_(
              rnn_brgemm_.kernel_layer_NK1_tail_b1_[layer_desc_idx_].get())
    , gemm_acc_scratchpad_(gemm_acc_scratchpad)
    , addr_batch_global_(addr_batch_global)
    , fused_postgemm_(fused_postgemm)
    , is_fused_layer_iter_brgemm_(!rnn_.is_lbr && rnn_.sic == rnn_.slc
              && LDA_iter_ == LDA_layer_ && need_gemm_layer_)
    , max_nthr_(calculate_nthr()) {}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t, gemm_acc_t>::execute()
        const {
    if (is_fused_layer_iter_brgemm_) {
        parallel(max_nthr_, [this](const int ithr, const int nthr) {
            this->kernel_fused_iter_layer(ithr, nthr);
        });
    } else {
        parallel(max_nthr_, [this](const int ithr, const int nthr) {
            this->kernel(ithr, nthr);
        });
    }
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
int brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t,
        gemm_acc_t>::calculate_nthr() const {
    // TODO(aarch64): the serial/parallel crossover for small problems
    // (M == 1, no layer GEMM, LSTM/LBR-GRU) has not been measured on
    // AArch64. The x64 path returned 1 below an ISA-specific K2 threshold
    // tuned on AVX2; those constants do not transfer to AArch64, so no
    // small-problem heuristic is applied here yet. Until a sweep is done
    // on target hardware, always use the full thread count.
    return nstl::min(dnnl_get_current_num_threads(), rnn_.nthr);
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t, gemm_acc_t>::kernel(
        const int ithr, const int nthr) const {

    using namespace cpu::rnn_utils;

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const int max_K_Block = nstl::max(rnn_.KB1_blocks + 1,
            nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    dim_t nb_i = 0, mb = 0;
    switch (rnn_.loop_order) {
        case brgemm_rnn_execute_loop_order_t::mblk_nblk:
            nd_iterator_init(start, mb, m_blocking_, nb_i, n_blocking_);
            break;
        case brgemm_rnn_execute_loop_order_t::nblk_mblk:
            nd_iterator_init(start, nb_i, n_blocking_, mb, m_blocking_);
            break;
        default: assert(!"unsupported loop order");
    }

    while (start < end) {
        const auto m = mb * rnn_.m_block;
        const auto nb = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
        const auto n = nb * rnn_.n_block;
        const auto g_unfused
                = (rnn_.unfused_post_gemm) ? nb_i % rnn_.n_gates : 0;

        const auto *const A_layer_m = A_layer_ + m * LDA_layer_;
        const auto *const A_iter_m = A_iter_ + m * LDA_iter_;
        const auto *const B_layer_n = B_layer_ + nb * B_layer_n_offset_;
        const auto *const B_iter_n = B_iter_ + nb * B_iter_n_offset_;
        auto *const C_n = C_gates_ + m * rnn_.LDC + n;

        const auto cell_stride = rnn_.LDC;
        auto *const C_cell_i
                = C_cell_ ? C_cell_ + m * cell_stride + n : C_cell_;

        assert(rnn_.LDC == rnn_.scratch_gates_ld);

        const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;

        const brgemm_kernel_t *brgemm_kernel_layer_b0 = do_n_tail
                ? brgemm_kernel_layer_n_tail_
                : brgemm_kernel_layer_main_;
        const brgemm_kernel_t *brgemm_kernel_iter = do_n_tail
                ? brgemm_kernel_iter_n_tail_
                : brgemm_kernel_iter_main_;
        const brgemm_kernel_t *brgemm_kernel_layer_k_tail = do_n_tail
                ? brgemm_kernel_layer_nk_tail_
                : brgemm_kernel_layer_k_tail_;
        const brgemm_kernel_t *brgemm_kernel_iter_k_tail = do_n_tail
                ? brgemm_kernel_iter_nk_tail_
                : brgemm_kernel_iter_k_tail_;

        if (rnn_.is_lbr) {
            for (dim_t i = 0; i < rnn_.m_block; ++i) {
                auto *C_o = C_cell_i + i * cell_stride;
                PRAGMA_OMP_SIMD()
                for (dim_t j = 0; j < nstl::min(rnn_.n_block, cell_stride); ++j)
                    C_o[j] = 0;
            }
        }

        for (int g = 0; g < n_gates_; g++) {
            const int lg = g + g_unfused;
            const auto *const B_layer_g = B_layer_n + lg * B_layer_g_offset_;
            const auto *const B_iter_g = B_iter_n + lg * B_iter_g_offset_;
            auto *C_g = C_n + lg * rnn_.N;

            if (need_gemm_layer_) {
                for (int i = 0; i < rnn_.KB1_blocks; i++) {
                    addr_batch[i].ptr.A = A_layer_m + i * rnn_.k1_block;
                    addr_batch[i].ptr.B = B_layer_g + i * B_layer_kb_offset_;
                }
                brgemm_kernel_execute(brgemm_kernel_layer_b0, rnn_.KB1_blocks,
                        addr_batch, reinterpret_cast<void *>(C_g), nullptr);
            }

            if (rnn_.is_lbr && g == n_gates_ - 1) C_g = C_cell_i;

            for (int i = 0; i < rnn_.KB2_blocks; i++) {
                addr_batch[i].ptr.A = A_iter_m + i * rnn_.k2_block;
                addr_batch[i].ptr.B = B_iter_g + i * B_iter_kb_offset_;
            }
            brgemm_kernel_execute(brgemm_kernel_iter, rnn_.KB2_blocks,
                    addr_batch, reinterpret_cast<void *>(C_g), nullptr);
        }

        if (rnn_.k1_tail && need_gemm_layer_) {
            for (int g = 0; g < n_gates_; g++) {
                const int lg = g + g_unfused;
                const auto *const B_layer_g
                        = B_layer_n + lg * B_layer_g_offset_;
                auto *const C_g = C_n + lg * rnn_.N;
                addr_batch[0].ptr.A = A_layer_m + A_layer_k_tail_offset_;
                addr_batch[0].ptr.B = B_layer_g + B_layer_k_tail_offset_;
                brgemm_kernel_execute(brgemm_kernel_layer_k_tail, 1, addr_batch,
                        reinterpret_cast<void *>(C_g), nullptr);
            }
        }

        if (rnn_.k2_tail) {
            for (int g = 0; g < n_gates_; g++) {
                const int lg = g + g_unfused;
                const auto *const B_iter_g = B_iter_n + lg * B_iter_g_offset_;
                auto *C_g = C_n + lg * rnn_.N;
                if (rnn_.is_lbr && g == n_gates_ - 1) C_g = C_cell_i;
                addr_batch[0].ptr.A = A_iter_m + A_iter_k_tail_offset_;
                addr_batch[0].ptr.B = B_iter_g + B_iter_k_tail_offset_;
                brgemm_kernel_execute(brgemm_kernel_iter_k_tail, 1, addr_batch,
                        reinterpret_cast<void *>(C_g), nullptr);
            }
        }

        if (!rnn_.unfused_post_gemm) {
            const int n_elems = do_n_tail ? rnn_.n_tail : rnn_.n_block;
            const int block_step_bytes = n_elems * (int)sizeof(scratch_t);
            fused_postgemm_(
                    m, n, nb_i, A_iter_m, C_n, C_cell_i, block_step_bytes);
        }

        ++start;
        switch (rnn_.loop_order) {
            case brgemm_rnn_execute_loop_order_t::mblk_nblk:
                nd_iterator_step(mb, m_blocking_, nb_i, n_blocking_);
                break;
            case brgemm_rnn_execute_loop_order_t::nblk_mblk:
                nd_iterator_step(nb_i, n_blocking_, mb, m_blocking_);
                break;
            default: assert(!"unsupported loop order");
        }
    }
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_dst_layer_iter_t<src_t, weights_t, scratch_t,
        gemm_acc_t>::kernel_fused_iter_layer(const int ithr,
        const int nthr) const {

    using namespace cpu::rnn_utils;

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const int max_K_Block = 2
            * nstl::max(rnn_.KB1_blocks + 1,
                    nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    dim_t nb_i = 0, mb = 0;
    switch (rnn_.loop_order) {
        case brgemm_rnn_execute_loop_order_t::mblk_nblk:
            nd_iterator_init(start, mb, m_blocking_, nb_i, n_blocking_);
            break;
        case brgemm_rnn_execute_loop_order_t::nblk_mblk:
            nd_iterator_init(start, nb_i, n_blocking_, mb, m_blocking_);
            break;
        default: assert(!"unsupported loop order");
    }

    const auto LDA = LDA_layer_;
    const auto B_n_offset = B_layer_n_offset_;
    const auto B_g_offset = B_layer_g_offset_;
    const auto B_kb_offset = B_layer_kb_offset_;
    const auto KB_blocks
            = (need_gemm_layer_ ? rnn_.KB1_blocks : 0) + rnn_.KB2_blocks;
    const auto KB_blocks_tail = (need_gemm_layer_ ? 1 : 0) + 1;
    const auto A_k_tail_offset = A_layer_k_tail_offset_;
    const auto B_k_tail_offset = B_layer_k_tail_offset_;

    while (start < end) {
        const auto m = mb * rnn_.m_block;
        const auto nb = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
        const auto n = nb * rnn_.n_block;
        const auto g_unfused
                = (rnn_.unfused_post_gemm) ? nb_i % rnn_.n_gates : 0;

        const auto *const A_layer_m = A_layer_ + m * LDA;
        const auto *const A_iter_m = A_iter_ + m * LDA;
        const auto *const B_layer_n = B_layer_ + nb * B_n_offset;
        const auto *const B_iter_n = B_iter_ + nb * B_n_offset;
        auto *const C_n = C_gates_ + m * rnn_.LDC + n;

        const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;

        const brgemm_kernel_t *brgemm_kernel = do_n_tail
                ? brgemm_kernel_layer_n_tail_
                : brgemm_kernel_layer_main_;
        const brgemm_kernel_t *brgemm_kernel_k_tail = do_n_tail
                ? brgemm_kernel_layer_nk_tail_
                : brgemm_kernel_layer_k_tail_;

        for (int g = 0; g < n_gates_; g++) {
            const int lg = g + g_unfused;
            const auto *const B_layer_g = B_layer_n + lg * B_g_offset;
            const auto *const B_iter_g = B_iter_n + lg * B_g_offset;
            auto *const C_g = C_n + lg * rnn_.N;

            int batch_idx = 0;
            if (need_gemm_layer_) {
                for (; batch_idx < rnn_.KB1_blocks; batch_idx++) {
                    addr_batch[batch_idx].ptr.A
                            = A_layer_m + batch_idx * rnn_.k1_block;
                    addr_batch[batch_idx].ptr.B
                            = B_layer_g + batch_idx * B_kb_offset;
                }
            }

            int iter_idx = 0;
            for (; batch_idx < KB_blocks; batch_idx++) {
                addr_batch[batch_idx].ptr.A
                        = A_iter_m + iter_idx * rnn_.k2_block;
                addr_batch[batch_idx].ptr.B = B_iter_g + iter_idx * B_kb_offset;
                iter_idx++;
            }

            brgemm_kernel_execute(brgemm_kernel, KB_blocks, addr_batch,
                    reinterpret_cast<void *>(C_g), nullptr);
        }

        if (rnn_.k2_tail) {
            for (int g = 0; g < n_gates_; g++) {
                const int lg = g + g_unfused;
                auto *const C_g = C_n + lg * rnn_.N;

                int batch_idx = 0;
                if (need_gemm_layer_) {
                    const auto *const B_layer_g = B_layer_n + lg * B_g_offset;
                    addr_batch[batch_idx].ptr.A = A_layer_m + A_k_tail_offset;
                    addr_batch[batch_idx].ptr.B = B_layer_g + B_k_tail_offset;
                    batch_idx++;
                }
                const auto *const B_iter_g = B_iter_n + lg * B_g_offset;
                addr_batch[batch_idx].ptr.A = A_iter_m + A_k_tail_offset;
                addr_batch[batch_idx].ptr.B = B_iter_g + B_k_tail_offset;

                brgemm_kernel_execute(brgemm_kernel_k_tail, KB_blocks_tail,
                        addr_batch, reinterpret_cast<void *>(C_g), nullptr);
            }
        }

        if (!rnn_.unfused_post_gemm) {
            const auto block_step = (do_n_tail ? rnn_.n_tail : rnn_.n_block)
                    * sizeof(scratch_t);
            fused_postgemm_(m, n, nb_i, A_iter_m, C_n, C_cell_, block_step);
        }

        ++start;
        switch (rnn_.loop_order) {
            case brgemm_rnn_execute_loop_order_t::mblk_nblk:
                nd_iterator_step(mb, m_blocking_, nb_i, n_blocking_);
                break;
            case brgemm_rnn_execute_loop_order_t::nblk_mblk:
                nd_iterator_step(nb_i, n_blocking_, mb, m_blocking_);
                break;
            default: assert(!"unsupported loop order");
        }
    }
}

template <typename src_t, typename weights_t, typename gemm_acc_t>
brgemm_dst_proj_t<src_t, weights_t, gemm_acc_t>::brgemm_dst_proj_t(
        const ref_rnn_brgemm_t &rnn_brgemm, const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const src_t *proj_ht,
        const weights_t *w_projection, gemm_acc_t *output,
        gemm_acc_t *gemm_acc_scratchpad,
        brgemm_batch_element_t *addr_batch_global,
        const postgemm_fused_t &fused_postgemm)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , proj_desc_idx_(rnn_.is_cell_dt_f32()
                      ? rnn_.dst_brgemm_desc(cell_position, true)
                      : 0)
    , A_(proj_ht)
    , B_(w_projection)
    , C_(output)
    , LDC_(rnn_.is_cell_dt_f32() ? rnn_.dst_layer_ld(cell_position, true)
                                 : rnn_.scratch_gates_ld)
    , max_nthr_(nstl::min(dnnl_get_current_num_threads(), rnn_.nthr))
    , work_amount_proj_(rnn_.Nproj_blocks * rnn_.M_blocks)
    , B_n_offset_(rnn_.Kprojpadded * rnn_.n_block)
    , Bp_kb_offset_(rnn_.kproj_block * rnn_.n_block)
    , gemm_acc_scratchpad_(gemm_acc_scratchpad)
    , addr_batch_global_(addr_batch_global)
    , brgemm_kernel_main_(rnn_brgemm_.kernel_proj_b0_[proj_desc_idx_].get())
    , brgemm_kernel_n_tail_(
              rnn_brgemm_.kernel_proj_N_tail_b0_[proj_desc_idx_].get())
    , fused_postgemm_(fused_postgemm) {}

template <typename src_t, typename weights_t, typename gemm_acc_t>
void brgemm_dst_proj_t<src_t, weights_t, gemm_acc_t>::execute() const {
    parallel(max_nthr_, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

template <typename src_t, typename weights_t, typename gemm_acc_t>
void brgemm_dst_proj_t<src_t, weights_t, gemm_acc_t>::kernel(
        const int ithr, const int nthr) const {

    using namespace cpu::rnn_utils;

    int start = 0, end = 0;
    balance211(work_amount_proj_, nthr, ithr, start, end);

    auto *const addr_batch = addr_batch_global_ + ithr;

    int nb = 0, mb = 0;
    switch (rnn_.loop_order) {
        case brgemm_rnn_execute_loop_order_t::mblk_nblk:
            nd_iterator_init(start, mb, rnn_.M_blocks, nb, rnn_.Nproj_blocks);
            break;
        case brgemm_rnn_execute_loop_order_t::nblk_mblk:
            nd_iterator_init(start, nb, rnn_.Nproj_blocks, mb, rnn_.M_blocks);
            break;
        default: assert(!"unsupported loop order");
    }

    while (start < end) {
        const int n = nb * rnn_.n_block;
        const int m = mb * rnn_.m_block;
        const bool do_n_tail = (n + rnn_.n_block) > rnn_.Nproj;
        const int block_step = ((do_n_tail) ? rnn_.nproj_tail : rnn_.n_block)
                * sizeof(src_t);

        const auto *const Ap_m = A_ + m * rnn_.LDAproj;
        const auto *const Bp_n = B_ + nb * B_n_offset_;
        auto *const Cp_n = C_ + m * LDC_ + n;

        const brgemm_kernel_t *const brgemm_kernel_proj_b0
                = do_n_tail ? brgemm_kernel_n_tail_ : brgemm_kernel_main_;

        addr_batch[0].ptr.A = Ap_m;
        addr_batch[0].ptr.B = Bp_n;
        brgemm_kernel_execute(brgemm_kernel_proj_b0, 1, addr_batch,
                reinterpret_cast<void *>(Cp_n), nullptr);

        if (!rnn_.unfused_post_gemm) {
            fused_postgemm_(m, n, Cp_n, block_step);
        }

        ++start;
        switch (rnn_.loop_order) {
            case brgemm_rnn_execute_loop_order_t::mblk_nblk:
                nd_iterator_step(mb, rnn_.M_blocks, nb, rnn_.Nproj_blocks);
                break;
            case brgemm_rnn_execute_loop_order_t::nblk_mblk:
                nd_iterator_step(nb, rnn_.Nproj_blocks, mb, rnn_.M_blocks);
                break;
            default: assert(!"unsupported loop order");
        }
    }
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
brgemm_gru_t<src_t, weights_t, scratch_t, gemm_acc_t>::brgemm_gru_t(
        const ref_rnn_brgemm_t &rnn_brgemm, const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const src_t *src_iter,
        const src_t *src_layer, weights_t *w_iter0, weights_t *w_iter1,
        weights_t *w_layer, src_t *d_layer, scratch_t *scratch_gates,
        scratch_t *scratch_cell, gemm_acc_t *gemm_acc_scratchpad,
        brgemm_batch_element_t *addr_batch_global,
        const postgemm_fused_t &fused_postgemm_part1,
        const postgemm_fused_t &fused_postgemm_part2)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , need_gemm_layer_(rnn_.need_gemm_layer(cell_position))
    , layer_desc_idx_(rnn_.layer_brgemm_desc(cell_position))
    , iter_desc_idx_(rnn_.iter_brgemm_desc(cell_position))
    , iter_part2_desc_idx_(rnn_.iter_part2_brgemm_desc(cell_position))
    , A_layer_(src_layer)
    , A_iter_(src_iter)
    , B_layer_(w_layer)
    , B_iter_(w_iter0)
    , B_iter_part2_(w_iter1)
    , C_gates_(scratch_gates)
    , C_cell_(scratch_cell)
    , D_layer_(d_layer)
    , LDA_layer_(rnn_.src_layer_ld(cell_position))
    , LDA_iter_part1_(rnn_.src_iter_ld(cell_position))
    , LDA_iter_part2_(rnn_.dst_iter_part2_ld(cell_position))
    , max_nthr_(nstl::min(dnnl_get_current_num_threads(), rnn_.nthr))
    , n_blocking_((rnn_.unfused_post_gemm) ? rnn_.N_blocks * rnn_.n_gates
                                           : rnn_.N_blocks)
    , m_blocking_(rnn_.M_blocks)
    , work_amount_(m_blocking_)
    , B_layer_n_offset_(rnn_.K1padded * rnn_.n_block)
    , B_iter_n_offset_(rnn_.K2padded * rnn_.n_block)
    , B_layer_g_offset_(rnn_.N_blocks * B_layer_n_offset_)
    , B_iter_g_offset_(rnn_.N_blocks * B_iter_n_offset_)
    , A_layer_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block)
    , A_iter_k_tail_offset_(rnn_.KB2_blocks * rnn_.k2_block)
    , B_layer_kb_offset_(rnn_.k1_block * rnn_.n_block)
    , B_iter_kb_offset_(rnn_.k2_block * rnn_.n_block)
    , B_layer_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block * rnn_.n_block)
    , B_iter_k_tail_offset_(rnn_.KB2_blocks * rnn_.k2_block * rnn_.n_block)
    , n_gates_(rnn.unfused_post_gemm ? 1 : rnn.n_gates)
    , brgemm_kernel_iter_p0_main_(need_gemm_layer_
                      ? rnn_brgemm_.kernel_iter_b1_[iter_desc_idx_].get()
                      : rnn_brgemm_.kernel_iter_b0_[iter_desc_idx_].get())
    , brgemm_kernel_iter_p0_n_tail_(need_gemm_layer_
                      ? rnn_brgemm_.kernel_iter_N_tail_b1_[iter_desc_idx_].get()
                      : rnn_brgemm_.kernel_iter_N_tail_b0_[iter_desc_idx_]
                                .get())
    , brgemm_kernel_iter_p0_k_tail_(
              rnn_brgemm_.kernel_iter_K2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_p0_nk_tail_(
              rnn_brgemm_.kernel_iter_NK2_tail_b1_[iter_desc_idx_].get())
    , brgemm_kernel_iter_p1_main_(
              rnn_brgemm_.kernel_iter_p2_b1_[iter_part2_desc_idx_].get())
    , brgemm_kernel_iter_p1_n_tail_(
              rnn_brgemm_.kernel_iter_p2_N_tail_b1_[iter_part2_desc_idx_].get())
    , brgemm_kernel_iter_p1_k_tail_(
              rnn_brgemm_.kernel_iter_p2_K2_tail_b1_[iter_part2_desc_idx_]
                      .get())
    , brgemm_kernel_iter_p1_nk_tail_(
              rnn_brgemm_.kernel_iter_p2_NK2_tail_b1_[iter_part2_desc_idx_]
                      .get())
    , brgemm_kernel_layer_main_(
              rnn_brgemm_.kernel_layer_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_n_tail_(
              rnn_brgemm_.kernel_layer_N_tail_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_k_tail_(
              rnn_brgemm_.kernel_layer_K1_tail_b1_[layer_desc_idx_].get())
    , brgemm_kernel_layer_nk_tail_(
              rnn_brgemm_.kernel_layer_NK1_tail_b1_[layer_desc_idx_].get())
    , gemm_acc_scratchpad_(gemm_acc_scratchpad)
    , addr_batch_global_(addr_batch_global)
    , fused_postgemm_part1_(fused_postgemm_part1)
    , fused_postgemm_part2_(fused_postgemm_part2)
    , is_fused_layer_iter_brgemm_(true) {}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_gru_t<src_t, weights_t, scratch_t, gemm_acc_t>::execute() const {
    assert(is_fused_layer_iter_brgemm_);
    parallel(max_nthr_, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_gru_t<src_t, weights_t, scratch_t, gemm_acc_t>::kernel(
        const int ithr, const int nthr) const {

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const int max_K_Block = nstl::max(rnn_.KB1_blocks + 1,
            nstl::max(rnn_.KBproj_blocks + 1, rnn_.KB2_blocks + 1));
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    while (start < end) {
        dim_t mb = start;
        const auto m = mb * rnn_.m_block;

        const auto *const A_layer_m = A_layer_ + m * LDA_layer_;
        const auto *const A_iter_m = A_iter_ + m * LDA_iter_part1_;
        const auto *const A_iter_part2_m = D_layer_ + m * LDA_iter_part2_;

        for (dim_t nb_i = 0; nb_i < n_blocking_; nb_i++) {
            const auto nb
                    = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
            const auto n = nb * rnn_.n_block;
            const auto *const B_layer_n = B_layer_ + nb * B_layer_n_offset_;
            const auto *const B_iter_n = B_iter_ + nb * B_iter_n_offset_;
            auto *const C_gates_n = C_gates_ + m * rnn_.LDC + n;
            auto *const C_cell_n = C_cell_ + m * rnn_.LDC + n;

            const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;

            const brgemm_kernel_t *brgemm_kernel_layer = do_n_tail
                    ? brgemm_kernel_layer_n_tail_
                    : brgemm_kernel_layer_main_;
            const brgemm_kernel_t *brgemm_kernel_layer_k_tail = do_n_tail
                    ? brgemm_kernel_layer_nk_tail_
                    : brgemm_kernel_layer_k_tail_;
            const brgemm_kernel_t *brgemm_kernel_iter_p0 = do_n_tail
                    ? brgemm_kernel_iter_p0_n_tail_
                    : brgemm_kernel_iter_p0_main_;
            const brgemm_kernel_t *brgemm_kernel_iter_p0_k_tail = do_n_tail
                    ? brgemm_kernel_iter_p0_nk_tail_
                    : brgemm_kernel_iter_p0_k_tail_;

            if (need_gemm_layer_) {
                for (int g = 0; g < n_gates_; g++) {
                    const auto *const B_layer_g
                            = B_layer_n + g * B_layer_g_offset_;
                    auto *const C_gates_g = C_gates_n + g * rnn_.N;
                    for (int batch_idx = 0; batch_idx < rnn_.KB1_blocks;
                            batch_idx++) {
                        addr_batch[batch_idx].ptr.A
                                = A_layer_m + batch_idx * rnn_.k1_block;
                        addr_batch[batch_idx].ptr.B
                                = B_layer_g + batch_idx * B_layer_kb_offset_;
                    }
                    brgemm_kernel_execute(brgemm_kernel_layer, rnn_.KB1_blocks,
                            addr_batch, reinterpret_cast<void *>(C_gates_g),
                            nullptr);
                }
            }

            if (need_gemm_layer_ && rnn_.k1_tail > 0) {
                for (int g = 0; g < n_gates_; g++) {
                    const auto *const B_layer_g
                            = B_layer_n + g * B_layer_g_offset_;
                    auto *const C_gates_g = C_gates_n + g * rnn_.N;
                    addr_batch[0].ptr.A
                            = A_layer_m + rnn_.KB1_blocks * rnn_.k1_block;
                    addr_batch[0].ptr.B
                            = B_layer_g + rnn_.KB1_blocks * B_layer_kb_offset_;
                    brgemm_kernel_execute(brgemm_kernel_layer_k_tail, 1,
                            addr_batch, reinterpret_cast<void *>(C_gates_g),
                            nullptr);
                }
            }

            for (int g = 0; g < n_gates_ - 1; g++) {
                const auto *const B_iter_g = B_iter_n + g * B_iter_g_offset_;
                auto *const C_gates_g = C_gates_n + g * rnn_.N;
                for (int batch_idx = 0; batch_idx < rnn_.KB2_blocks;
                        batch_idx++) {
                    addr_batch[batch_idx].ptr.A
                            = A_iter_m + batch_idx * rnn_.k2_block;
                    addr_batch[batch_idx].ptr.B
                            = B_iter_g + batch_idx * B_iter_kb_offset_;
                }
                brgemm_kernel_execute(brgemm_kernel_iter_p0, rnn_.KB2_blocks,
                        addr_batch, reinterpret_cast<void *>(C_gates_g),
                        nullptr);
            }

            if (rnn_.k2_tail > 0) {
                for (int g = 0; g < n_gates_ - 1; g++) {
                    const auto *const B_iter_g
                            = B_iter_n + g * B_iter_g_offset_;
                    auto *const C_gates_g = C_gates_n + g * rnn_.N;
                    addr_batch[0].ptr.A
                            = A_iter_m + rnn_.KB2_blocks * rnn_.k2_block;
                    addr_batch[0].ptr.B
                            = B_iter_g + rnn_.KB2_blocks * B_iter_kb_offset_;
                    brgemm_kernel_execute(brgemm_kernel_iter_p0_k_tail, 1,
                            addr_batch, reinterpret_cast<void *>(C_gates_g),
                            nullptr);
                }
            }

            if (!rnn_.unfused_post_gemm) {
                const int n_elems = do_n_tail ? rnn_.n_tail : rnn_.n_block;
                const int block_step_bytes = n_elems * (int)sizeof(scratch_t);
                fused_postgemm_part1_(m, n, nb_i, A_iter_m, C_gates_n, C_cell_n,
                        block_step_bytes);
            }
        }

        for (dim_t nb_i = 0; nb_i < n_blocking_; nb_i++) {
            const auto nb
                    = (rnn_.unfused_post_gemm) ? nb_i / rnn_.n_gates : nb_i;
            const auto n = nb * rnn_.n_block;
            const auto *const B_iter_part2_n
                    = B_iter_part2_ + nb * B_iter_n_offset_;
            auto *const C_gates_n = C_gates_ + m * rnn_.LDC + n;

            const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;

            const brgemm_kernel_t *brgemm_kernel_iter_p1 = do_n_tail
                    ? brgemm_kernel_iter_p1_n_tail_
                    : brgemm_kernel_iter_p1_main_;
            const brgemm_kernel_t *brgemm_kernel_iter_p1_k_tail = do_n_tail
                    ? brgemm_kernel_iter_p1_nk_tail_
                    : brgemm_kernel_iter_p1_k_tail_;

            for (int g = 0; g < 1; g++) {
                const auto *const B_iter_part2_g
                        = B_iter_part2_n + g * B_iter_g_offset_;
                auto *const C_gates_g = C_gates_n + (n_gates_ - 1) * rnn_.N;
                for (int batch_idx = 0; batch_idx < rnn_.KB2_blocks;
                        batch_idx++) {
                    addr_batch[batch_idx].ptr.A
                            = A_iter_part2_m + batch_idx * rnn_.k2_block;
                    addr_batch[batch_idx].ptr.B
                            = B_iter_part2_g + batch_idx * B_iter_kb_offset_;
                }
                brgemm_kernel_execute(brgemm_kernel_iter_p1, rnn_.KB2_blocks,
                        addr_batch, reinterpret_cast<void *>(C_gates_g),
                        nullptr);
            }

            if (rnn_.k2_tail > 0) {
                for (int g = 0; g < 1; g++) {
                    const auto *const B_iter_part2_g
                            = B_iter_part2_n + g * B_iter_g_offset_;
                    auto *const C_gates_g = C_gates_n + (n_gates_ - 1) * rnn_.N;
                    addr_batch[0].ptr.A
                            = A_iter_part2_m + rnn_.KB2_blocks * rnn_.k2_block;
                    addr_batch[0].ptr.B = B_iter_part2_g
                            + rnn_.KB2_blocks * B_iter_kb_offset_;
                    brgemm_kernel_execute(brgemm_kernel_iter_p1_k_tail, 1,
                            addr_batch, reinterpret_cast<void *>(C_gates_g),
                            nullptr);
                }
            }

            if (!rnn_.unfused_post_gemm && nb_i == n_blocking_ - 1) {
                fused_postgemm_part2_(m, 0, 0, A_iter_m,
                        C_gates_ + m * rnn_.LDC, C_cell_ + m * rnn_.LDC,
                        rnn_.N);
            }
        }

        ++start;
    }
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
brgemm_merged_layer_t<src_t, weights_t, scratch_t,
        gemm_acc_t>::brgemm_merged_layer_t(const ref_rnn_brgemm_t &rnn_brgemm,
        const rnn_utils::rnn_conf_t &rnn,
        rnn_utils::cell_position_t cell_position, const src_t *src_layer,
        weights_t *w_layer, scratch_t *scratch_gates,
        gemm_acc_t *gemm_acc_scratchpad,
        brgemm_batch_element_t *addr_batch_global)
    : rnn_brgemm_(rnn_brgemm)
    , rnn_(rnn)
    , layer_desc_idx_(rnn_.layer_brgemm_desc(cell_position))
    , A_layer_(src_layer)
    , B_layer_(w_layer)
    , C_(scratch_gates)
    , LDA_layer_(rnn_.src_layer_ld(cell_position))
    , max_nthr_(nstl::min(dnnl_get_current_num_threads(), rnn_.nthr))
    , n_blocking_(rnn_.N_blocks * rnn_.n_gates)
    , m_blocking_(rnn_.Mlayermerged_blocks)
    , work_amount_(n_blocking_ * m_blocking_)
    , B_layer_n_offset_(rnn_.K1padded * rnn_.n_block)
    , B_layer_g_offset_(rnn_.N_blocks * B_layer_n_offset_)
    , A_layer_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block)
    , B_layer_kb_offset_(rnn_.k1_block * rnn_.n_block)
    , B_layer_k_tail_offset_(rnn_.KB1_blocks * rnn_.k1_block * rnn_.n_block)
    , brgemm_kernel_layer_main_(
              rnn_brgemm_.kernel_layermerged_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_n_tail_(
              rnn_brgemm_.kernel_layermerged_N_tail_b0_[layer_desc_idx_].get())
    , brgemm_kernel_layer_k_tail_(
              rnn_brgemm_.kernel_layermerged_K1_tail_b1_[layer_desc_idx_].get())
    , brgemm_kernel_layer_nk_tail_(
              rnn_brgemm_.kernel_layermerged_NK1_tail_b1_[layer_desc_idx_]
                      .get())
    , gemm_acc_scratchpad_(gemm_acc_scratchpad)
    , addr_batch_global_(addr_batch_global) {}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_merged_layer_t<src_t, weights_t, scratch_t, gemm_acc_t>::execute()
        const {
    parallel(max_nthr_, [this](const int ithr, const int nthr) {
        this->kernel(ithr, nthr);
    });
}

template <typename src_t, typename weights_t, typename scratch_t,
        typename gemm_acc_t>
void brgemm_merged_layer_t<src_t, weights_t, scratch_t, gemm_acc_t>::kernel(
        const int ithr, const int nthr) const {

    using namespace cpu::rnn_utils;

    int start = 0, end = 0;
    balance211(work_amount_, nthr, ithr, start, end);

    const auto m_block = rnn_.mlayermerged_block;

    const int max_K_Block = rnn_.KB1_blocks + 1;
    brgemm_batch_element_t *const addr_batch
            = addr_batch_global_ + ithr * max_K_Block;

    dim_t nb_i = 0, mb = 0;
    switch (rnn_.loop_order) {
        case brgemm_rnn_execute_loop_order_t::mblk_nblk:
            nd_iterator_init(start, mb, m_blocking_, nb_i, n_blocking_);
            break;
        case brgemm_rnn_execute_loop_order_t::nblk_mblk:
            nd_iterator_init(start, nb_i, n_blocking_, mb, m_blocking_);
            break;
        default: assert(!"unsupported loop order");
    }

    while (start < end) {
        const auto m = mb * m_block;
        const auto nb = nb_i / rnn_.n_gates;
        const auto n = nb * rnn_.n_block;
        const auto g = nb_i % rnn_.n_gates;

        const auto *const A_layer_m = A_layer_ + m * LDA_layer_;
        const auto *const B_layer_n = B_layer_ + nb * B_layer_n_offset_;
        auto *const C_n = C_ + m * rnn_.LDC + n;

        const bool do_n_tail = (n + rnn_.n_block) > rnn_.N;

        const brgemm_kernel_t *brgemm_kernel_layer_b0 = do_n_tail
                ? brgemm_kernel_layer_n_tail_
                : brgemm_kernel_layer_main_;
        const brgemm_kernel_t *brgemm_kernel_layer_k_tail = do_n_tail
                ? brgemm_kernel_layer_nk_tail_
                : brgemm_kernel_layer_k_tail_;

        const auto *const B_layer_g = B_layer_n + g * B_layer_g_offset_;
        auto *const C_g = C_n + g * rnn_.N;

        for (int i = 0; i < rnn_.KB1_blocks; i++) {
            addr_batch[i].ptr.A = A_layer_m + i * rnn_.k1_block;
            addr_batch[i].ptr.B = B_layer_g + i * B_layer_kb_offset_;
        }
        brgemm_kernel_execute(brgemm_kernel_layer_b0, rnn_.KB1_blocks,
                addr_batch, reinterpret_cast<void *>(C_g), nullptr);

        if (rnn_.k1_tail) {
            const auto *const B_layer_g_tail
                    = B_layer_n + g * B_layer_g_offset_;
            auto *const C_g_tail = C_n + g * rnn_.N;
            addr_batch[0].ptr.A = A_layer_m + A_layer_k_tail_offset_;
            addr_batch[0].ptr.B = B_layer_g_tail + B_layer_k_tail_offset_;
            brgemm_kernel_execute(brgemm_kernel_layer_k_tail, 1, addr_batch,
                    reinterpret_cast<void *>(C_g_tail), nullptr);
        }

        ++start;
        switch (rnn_.loop_order) {
            case brgemm_rnn_execute_loop_order_t::mblk_nblk:
                nd_iterator_step(mb, m_blocking_, nb_i, n_blocking_);
                break;
            case brgemm_rnn_execute_loop_order_t::nblk_mblk:
                nd_iterator_step(nb_i, n_blocking_, mb, m_blocking_);
                break;
            default: assert(!"unsupported loop order");
        }
    }
}

template class brgemm_dst_layer_iter_t<uint8_t, int8_t, int32_t, int32_t>;
template class brgemm_dst_layer_iter_t<int8_t, int8_t, int32_t, int32_t>;
template class brgemm_dst_layer_iter_t<float, float, float, float>;
template class brgemm_dst_layer_iter_t<bfloat16_t, bfloat16_t, float, float>;
template class brgemm_dst_layer_iter_t<float16_t, float16_t, float, float>;

template class brgemm_dst_proj_t<float, float, float>;
template class brgemm_dst_proj_t<bfloat16_t, bfloat16_t, float>;
template class brgemm_dst_proj_t<float16_t, float16_t, float>;
template class brgemm_dst_proj_t<int8_t, int8_t, int32_t>;
template class brgemm_dst_proj_t<uint8_t, int8_t, int32_t>;

template class brgemm_gru_t<uint8_t, int8_t, int32_t, int32_t>;
template class brgemm_gru_t<int8_t, int8_t, int32_t, int32_t>;
template class brgemm_gru_t<float, float, float, float>;
template class brgemm_gru_t<bfloat16_t, bfloat16_t, float, float>;
template class brgemm_gru_t<float16_t, float16_t, float, float>;

template class brgemm_merged_layer_t<uint8_t, int8_t, int32_t, int32_t>;
template class brgemm_merged_layer_t<int8_t, int8_t, int32_t, int32_t>;
template class brgemm_merged_layer_t<float, float, float, float>;
template class brgemm_merged_layer_t<bfloat16_t, bfloat16_t, float, float>;
template class brgemm_merged_layer_t<float16_t, float16_t, float, float>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
