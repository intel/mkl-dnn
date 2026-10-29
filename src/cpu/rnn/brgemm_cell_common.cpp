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

/*
 * Common for RNN and LSTM cell execution
 */

#include "common/bfloat16.hpp"
#include "cpu/rnn/ref_rnn.hpp"

#if DNNL_X64
#include <cassert>
#include <functional>
#include "cpu/x64/rnn/brgemm_cell_common_bwd.hpp"
#include "cpu/x64/rnn/brgemm_cell_common_fwd.hpp"
#include "cpu/x64/rnn/brgemm_cell_common_reorders.hpp"
#include "cpu/x64/rnn/brgemm_cell_common_utils.hpp"
#endif

namespace dnnl {
namespace impl {
namespace cpu {

using namespace rnn_utils;
using namespace dnnl::impl::utils;
#if DNNL_X64
using namespace dnnl::impl::cpu::x64;
#endif

template <data_type_t src_type, data_type_t weights_type, data_type_t acc_type>
rnn_merged_layer_execution_sig((
        ref_rnn_fwd_t<src_type, weights_type, acc_type>::merged_layer_brgemm)) {
#if DNNL_X64
    using brgemm_merged_layer_t = x64::brgemm_merged_layer_t<src_iter_t,
            weights_t, scratch_t, gemm_acc_t>;
    const brgemm_merged_layer_t layer_calc(this->rnn_brgemm_, rnn,
            cell_position, src_layer_, w_layer_[0], scratch_gates_,
            amx_scratchpad, addr_batch_global);

    layer_calc.execute();
#endif
    return dnnl_success;
}

template rnn_merged_layer_execution_sig(ref_rnn_fwd_f32_t::merged_layer_brgemm);
template rnn_merged_layer_execution_sig(
        ref_rnn_fwd_bf16_t::merged_layer_brgemm);
template rnn_merged_layer_execution_sig(ref_rnn_fwd_f16_t::merged_layer_brgemm);
template rnn_merged_layer_execution_sig(
        ref_rnn_fwd_u8s8_t::merged_layer_brgemm);
template rnn_merged_layer_execution_sig(
        ref_rnn_fwd_s8s8_t::merged_layer_brgemm);

template <data_type_t src_type, data_type_t weights_type, data_type_t acc_type>
rnn_cell_execution_sig((ref_rnn_fwd_t<src_type, weights_type,
        acc_type>::cell_execution_brgemm)) {
#if DNNL_X64

    const auto weights_scales = this->pd_->attr()->rnn_weights_qparams_.scales_;
    const int mask = this->pd()->attr()->rnn_weights_qparams_.mask_;
    const auto dst_postgemm = rnn.is_lstm_projection ? proj_ht_ : dst_layer_;
    const auto dst_iter_postgemm = rnn.is_lstm_projection ? nullptr : dst_iter_;

    const auto LDDl = rnn.dst_layer_ld(cell_position);
    const auto LDDi = rnn.dst_iter_ld(cell_position);
    const auto LDDic = rnn.dst_iter_c_ld(cell_position);
    const auto LDAic = rnn.src_iter_c_ld(cell_position);

    using brgemm_dst_layer_iter_t = x64::brgemm_dst_layer_iter_t<src_iter_t,
            weights_t, scratch_t, gemm_acc_t>;

    typename brgemm_dst_layer_iter_t::postgemm_fused_t fused_postgemm;

    if (!rnn.unfused_post_gemm) {
        fused_postgemm = [&](dim_t m, dim_t n, dim_t nb_i,
                                 const src_iter_t *Ai_m, scratch_t *C_n,
                                 scratch_t *C_cell_n, dim_t block_step) {
            const auto Dpg_n = (dst_postgemm != nullptr)
                    ? dst_postgemm + m * LDDl + n
                    : nullptr;
            const auto Di_n = (dst_iter_postgemm != nullptr)
                    ? dst_iter_postgemm + m * LDDi + n
                    : nullptr;
            const auto Dic_n = (dst_iter_c_ != nullptr)
                    ? inc_ptr(dst_iter_c_, rnn.dst_iter_c_dt, m * LDDic + n)
                    : nullptr;

            const auto curr_ws_gates_
                    = ws_gates_ + (m * rnn.ws_gates_ld) + nb_i * rnn.n_block;
            const float *weights_peephole_n = weights_peephole_
                    ? weights_peephole_ + n
                    : weights_peephole_;
            auto weights_scales_n = weights_scales + (mask ? n : 0);
            const auto Aic_n
                    = inc_ptr(src_iter_c_, rnn.src_iter_c_dt, m * LDAic + n);
            const auto bias_n = inc_ptr(bias_[0], rnn.bias_dt, n);
            const auto augru_attention_m
                    = augru_attention_ ? augru_attention_ + m : nullptr;
            this->rnn_postgemm_->execute(rnn, cell_position, curr_ws_gates_,
                    C_n, augru_attention_m, Dpg_n, Dic_n, Ai_m, Aic_n,
                    diff_src_layer_, diff_augru_attention_, diff_src_iter_,
                    diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
                    diff_dst_iter_c_, weights_peephole_n, bias_n, ws_grid_,
                    C_cell_n, Di_n, weights_scales_n, block_step);
        };
    }

    if (rnn.is_orig_gru) {
        using brgemm_gru_t = x64::brgemm_gru_t<src_iter_t, weights_t, scratch_t,
                gemm_acc_t>;
        typename brgemm_gru_t::postgemm_fused_t fused_postgemm_gru_part1,
                fused_postgemm_gru_part2;
        if (!rnn.unfused_post_gemm) {
            fused_postgemm_gru_part1
                    = [&](dim_t m, dim_t n, dim_t nb_i, const src_iter_t *Ai_m,
                              scratch_t *C_gates_n, scratch_t *C_cell_n,
                              dim_t block_step) {
                const auto Dpg_n = (dst_postgemm != nullptr)
                        ? dst_postgemm + m * LDDl + n
                        : nullptr;
                const auto Di_n = (dst_iter_postgemm != nullptr)
                        ? dst_iter_postgemm + m * LDDi + n
                        : nullptr;
                const auto Dic_n = (dst_iter_c_ != nullptr)
                        ? inc_ptr(dst_iter_c_, rnn.dst_iter_c_dt, m * LDDic + n)
                        : nullptr;

                const auto curr_ws_gates_ = ws_gates_ + (m * rnn.ws_gates_ld)
                        + nb_i * rnn.n_block;
                const auto Aic_n = inc_ptr(
                        src_iter_c_, rnn.src_iter_c_dt, m * LDAic + n);
                const auto bias_n = inc_ptr(bias_[0], rnn.bias_dt, n);
                auto weights_scales_n = weights_scales + (mask ? n : 0);
                const auto augru_attention_m
                        = augru_attention_ ? augru_attention_ + m : nullptr;
                this->rnn_postgemm_->execute(rnn, cell_position, curr_ws_gates_,
                        C_gates_n, augru_attention_m, Dpg_n, Dic_n, Ai_m, Aic_n,
                        diff_src_layer_, diff_augru_attention_, diff_src_iter_,
                        diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_,
                        nullptr, nullptr, bias_n, ws_grid_, C_cell_n, Di_n,
                        weights_scales_n, block_step);
            };
            fused_postgemm_gru_part2
                    = [&](dim_t m, dim_t n, dim_t nb_i, const src_iter_t *Ai_m,
                              scratch_t *C_gates_n, scratch_t *C_cell_n,
                              dim_t block_step) {
                const auto Dpg_n = (dst_postgemm != nullptr)
                        ? dst_postgemm + m * LDDl + n
                        : nullptr;
                const auto Di_n = (dst_iter_postgemm != nullptr)
                        ? dst_iter_postgemm + m * LDDi + n
                        : nullptr;
                const auto Dic_n = (dst_iter_c_ != nullptr)
                        ? inc_ptr(dst_iter_c_, rnn.dst_iter_c_dt, m * LDDic + n)
                        : nullptr;

                const auto curr_ws_gates_ = ws_gates_ + (m * rnn.ws_gates_ld)
                        + nb_i * rnn.n_block;
                const auto Aic_n = inc_ptr(
                        src_iter_c_, rnn.src_iter_c_dt, m * LDAic + n);
                const auto bias_n = inc_ptr(bias_[0], rnn.bias_dt, n);
                auto weights_scales_n = weights_scales + (mask ? n : 0);
                const auto augru_attention_m
                        = augru_attention_ ? augru_attention_ + m : nullptr;
                this->rnn_postgemm_->execute_part2(rnn, cell_position,
                        curr_ws_gates_, C_gates_n, augru_attention_m, Dpg_n,
                        Dic_n, Ai_m, Aic_n, diff_src_layer_,
                        diff_augru_attention_, diff_src_iter_, diff_src_iter_c_,
                        diff_dst_layer_, diff_dst_iter_, nullptr, nullptr,
                        bias_n, ws_grid_, C_cell_n, Di_n, weights_scales_n,
                        block_step);
            };
        }

        const brgemm_gru_t dst_calc(this->rnn_brgemm_, rnn, cell_position,
                src_iter_, src_layer_, w_iter_[0], w_iter_[1], w_layer_[0],
                dst_postgemm, scratch_gates_, scratch_cell_, amx_scratchpad,
                addr_batch_global, fused_postgemm_gru_part1,
                fused_postgemm_gru_part2);
        dst_calc.execute();
    } else {
        // calculate
        // scratch_gates_ = src_layer_ * w_layer_ + src_iter_ * w_iter_
        const brgemm_dst_layer_iter_t dst_calc(this->rnn_brgemm_, rnn,
                cell_position, src_iter_, src_layer_, w_iter_[0], w_layer_[0],
                scratch_gates_, scratch_cell_, amx_scratchpad,
                addr_batch_global, fused_postgemm);
        dst_calc.execute();
    }

    if (rnn.unfused_post_gemm) {
        const auto wscales_postgemm
                = this->pd_->attr()->rnn_weights_qparams_.scales_;

        this->rnn_postgemm_->execute(rnn, cell_position, ws_gates_,
                scratch_gates_, augru_attention_, dst_postgemm, dst_iter_c_,
                src_iter_, src_iter_c_, diff_src_layer_, diff_augru_attention_,
                diff_src_iter_, diff_src_iter_c_, diff_dst_layer_,
                diff_dst_iter_, diff_dst_iter_c_, weights_peephole_, bias_[0],
                ws_grid_, scratch_cell_, dst_iter_postgemm, wscales_postgemm,
                rnn.dhc * sizeof(scratch_t));
    }

    if (rnn.is_lstm_projection) {
        const auto wscales_proj_postgemm
                = this->pd_->attr()->rnn_weights_projection_qparams_.scales_;
        gemm_acc_t *const Cp = (rnn.dt_conf == all_f32)
                ? reinterpret_cast<gemm_acc_t *>(dst_layer_)
                : scratch_gates_;
        const dim_t pLDDl = rnn.dst_layer_ld(cell_position, true);
        const int pmask
                = this->pd_->attr()->rnn_weights_projection_qparams_.mask_;

        using brgemm_dst_proj_t
                = x64::brgemm_dst_proj_t<ht_t, weights_t, gemm_acc_t>;
        typename brgemm_dst_proj_t::postgemm_fused_t fused_postgemm_proj;

        if (!rnn.unfused_post_gemm) {
            fused_postgemm_proj = [&](dim_t m, dim_t n, gemm_acc_t *Cp_n,
                                          dim_t block_step) {
                const auto weights_scales_n
                        = wscales_proj_postgemm + (pmask ? n : 0);
                const auto Di_n = (dst_iter_ != nullptr)
                        ? dst_iter_ + m * LDDi + n
                        : nullptr;
                const auto Dl_n = (dst_layer_ != nullptr)
                        ? dst_layer_ + m * pLDDl + n
                        : nullptr;
                const auto Wp_comp_n = w_proj_comp + n;
                this->rnn_postgemm_->execute_part2(rnn, cell_position, nullptr,
                        Cp_n, nullptr, Dl_n, nullptr, nullptr, Wp_comp_n,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, Di_n,
                        weights_scales_n, block_step);
            };
        }

        // calculate
        // output = proj_ht_ * w_projection_
        const brgemm_dst_proj_t dst_proj_calc(this->rnn_brgemm_, rnn,
                cell_position, proj_ht_, w_projection_[0], Cp, amx_scratchpad,
                addr_batch_global, fused_postgemm_proj);
        dst_proj_calc.execute();

        if (rnn.unfused_post_gemm) {
            // we have to downconvert the output to dst_layer_t and copy to
            // dst_iter if needed
            this->rnn_postgemm_->execute_part2(rnn, cell_position, nullptr, Cp,
                    nullptr, dst_layer_, nullptr, nullptr, w_proj_comp, nullptr,
                    nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr, nullptr, dst_iter_,
                    wscales_proj_postgemm, rnn.dlc * sizeof(dst_layer_t));
        }
    }

#endif
    return dnnl_success;
}

template rnn_cell_execution_sig(ref_rnn_fwd_f32_t::cell_execution_brgemm);
template rnn_cell_execution_sig(ref_rnn_fwd_bf16_t::cell_execution_brgemm);
template rnn_cell_execution_sig(ref_rnn_fwd_f16_t::cell_execution_brgemm);
template rnn_cell_execution_sig(ref_rnn_fwd_u8s8_t::cell_execution_brgemm);
template rnn_cell_execution_sig(ref_rnn_fwd_s8s8_t::cell_execution_brgemm);

template <data_type_t src_type, data_type_t weights_type, data_type_t acc_type>
rnn_cell_execution_sig((ref_rnn_bwd_t<src_type, weights_type,
        acc_type>::cell_execution_brgemm)) {

#if DNNL_X64
    this->rnn_postgemm_->execute(rnn, cell_position, ws_gates_, scratch_gates_,
            augru_attention_, dst_layer_, dst_iter_c_, src_iter_, src_iter_c_,
            diff_src_layer_, diff_augru_attention_, diff_src_iter_,
            diff_src_iter_c_, diff_dst_layer_, diff_dst_iter_, diff_dst_iter_c_,
            weights_peephole_, bias_[0], ws_grid_, scratch_cell_, dst_iter_,
            nullptr, 0);

    using brgemm_diff_src_calc_t = x64::brgemm_diff_src_layer_iter_t<weights_t,
            scratch_t, gemm_acc_t>;
    using brgemm_diff_weights_calc_t
            = x64::brgemm_diff_weights_layer_iter_t<src_layer_t, src_iter_t,
                    scratch_t, gemm_acc_t>;

    const brgemm_diff_src_calc_t diff_src_calc(this->rnn_brgemm_, rnn,
            cell_position, scratch_gates_, w_iter_[0], w_layer_[0],
            diff_src_iter_, diff_src_layer_, amx_scratchpad, addr_batch_global);
    const brgemm_diff_weights_calc_t diff_weights_calc(this->rnn_brgemm_, rnn,
            cell_position, src_iter_, scratch_src_iter_, src_layer_,
            scratch_src_layer_, scratch_gates_, scratch_gates_blocked_,
            diff_w_iter_, diff_w_layer_, diff_bias_, amx_scratchpad,
            addr_batch_global);

    // calculate
    // dff_src_iter = scratch * w_iter
    // dff_src_layer = scratch * w_layer
    diff_src_calc.execute();

    if (rnn.diff_wei_brgemm.global_transpose) {
        const auto src_layer_ld = rnn.src_layer_ld(cell_position);
        const auto src_iter_ld = rnn.src_iter_ld(cell_position);
        const auto src_layer_ld_nb = rnn.layer_brgemm_desc(cell_position);
        const auto src_iter_ld_nb = rnn.iter_brgemm_desc(cell_position);
        const auto rnd_up_size = data_type_vnni_granularity(src_type);
        const auto dst_ld = utils::rnd_up(rnn.mb, rnd_up_size);

        const auto layer_transpose = src_layer_iter_transpose_t(src_layer_ld,
                dst_ld, rnn.mb, rnn.slc,
                this->rnn_brgemm_.kernel_transpose_layer_[src_layer_ld_nb]
                        .get());
        const auto iter_transpose = src_layer_iter_transpose_t(src_iter_ld,
                dst_ld, rnn.mb, rnn.sic,
                this->rnn_brgemm_.kernel_transpose_iter_[src_iter_ld_nb].get());
        layer_transpose.execute(src_layer_, scratch_src_layer_);
        iter_transpose.execute(src_iter_, scratch_src_iter_);
    }
    // calculate
    // dff_weights_layer = src_layer^T * scratch
    // dff_weights_iter = src_iter^T * scratch
    // performs gates reductions
    // diff_bias = scratch reduction over mb
    diff_weights_calc.execute();

    if (rnn.is_lstm_peephole) {
        using brgemm_diff_wei_peep_t = x64::brgemm_diff_wei_peep_t<scratch_t>;
        const brgemm_diff_wei_peep_t diff_wei_peep_calc(this->rnn_brgemm_, rnn,
                cell_position, scratch_gates_, src_iter_c_, dst_iter_c_,
                diff_weights_peephole_);

        diff_wei_peep_calc.execute();
    }

#endif

    return dnnl_success;
}
template rnn_cell_execution_sig(ref_rnn_bwd_f32_t::cell_execution_brgemm);
template rnn_cell_execution_sig(ref_rnn_bwd_bf16_t::cell_execution_brgemm);
template rnn_cell_execution_sig(ref_rnn_bwd_f16_t::cell_execution_brgemm);

} // namespace cpu
} // namespace impl
} // namespace dnnl
