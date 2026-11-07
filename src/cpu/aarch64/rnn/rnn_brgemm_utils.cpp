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

#include <algorithm>
#include <climits>
#include <cmath>
#include <tuple>
#include <utility>

#include "common/dnnl_thread.hpp"
#include "cpu/aarch64/brgemm/brgemm.hpp"
#include "cpu/aarch64/rnn/rnn_brgemm_utils.hpp"
#include "cpu/rnn/rnn_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace rnn_brgemm_utils {

namespace {

aarch64::cpu_isa_t brgemm_calc_isa(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t K1, dim_t K2);

std::pair<dim_t, dim_t> brgemm_calc_k_block(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t K1, dim_t K2, dim_t M,
        dim_t n_block, alg_kind_t cell_kind, dim_t src_layer_type_size,
        dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size,
        aarch64::cpu_isa_t isa);

std::pair<dim_t, dim_t> brgemm_calc_k_block_vanilla_rnn(dim_t K1, dim_t K2,
        dim_t M, dim_t n_block, dim_t src_layer_type_size, dim_t As, dim_t Bs,
        dim_t Cs, dim_t l2_cache_size, bool is_xf16, aarch64::cpu_isa_t isa);

dim_t brgemm_calc_m_block(alg_kind_t cell_kind, prop_kind_t aprop, dim_t nthr,
        dim_t M, dim_t N_blocks, bool is_f32, float work_by_N, dim_t As,
        dim_t Bs, dim_t Cs, dim_t l2_cache_size);

dim_t brgemm_calc_m_block_vanilla_rnn(dim_t nthr, dim_t M, dim_t N_blocks,
        float work_by_N, dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size);

dim_t brgemm_calc_m_block_lstm(dim_t nthr, dim_t M, dim_t N_blocks, bool is_f32,
        float work_by_N, dim_t As, dim_t Cs, dim_t l2_cache_size);

dim_t adjust_m_block_lstm(dim_t nthr, dim_t M, dim_t N_blocks);

dim_t brgemm_calc_n_block(
        const cpu::rnn_utils::rnn_conf_t &rnn, alg_kind_t cell_kind);

aarch64::cpu_isa_t brgemm_calc_isa(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t K1, dim_t K2) {

    // Feature gating: reject early if HW lacks required dtype support
    if (rnn.is_cell_dt_bf16() && !mayiuse_bf16()) return isa_undef;
    // TODO: Add int8 dotprod / f16 gating if brgemm_desc_init
    //       doesn't handle it robustly enough downstream.

    if (mayiuse(sve_512)) return sve_512;
    if (mayiuse(sve_256)) return sve_256;
    if (mayiuse(sve_128)) return sve_128;
    return isa_undef;
}

std::pair<dim_t, dim_t> brgemm_calc_k_block(
        const cpu::rnn_utils::rnn_conf_t &rnn, dim_t K1, dim_t K2, dim_t M,
        dim_t n_block, alg_kind_t cell_kind, dim_t src_layer_type_size,
        dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size,
        aarch64::cpu_isa_t isa) {

    if (cell_kind == alg_kind::vanilla_rnn)
        return brgemm_calc_k_block_vanilla_rnn(K1, K2, M, n_block,
                src_layer_type_size, As, Bs, Cs, l2_cache_size,
                rnn.is_cell_dt_xf16(), isa);

    return std::make_pair(K1, K2);
}

std::pair<dim_t, dim_t> brgemm_calc_k_block_vanilla_rnn(dim_t K1, dim_t K2,
        dim_t M, dim_t n_block, dim_t src_layer_type_size, dim_t As, dim_t Bs,
        dim_t Cs, dim_t l2_cache_size, bool is_xf16, aarch64::cpu_isa_t isa) {

    const bool is_sve = is_superset(isa, aarch64::sve_128);
    // Fraction of L2 that the A+B+C working set may occupy before we start
    // K-blocking. This multiplies the runtime-queried L2 size, so it is a plain
    // fraction and does not depend on vector length or ISA -- a single constant.
    //
    // The SVE value (0.50) was tuned on Graviton3 (Neoverse V1, sve_256, 1 MiB L2)
    // with a benchdnn VANILLA_RNN sweep over 15 shapes. It cut worst-case
    // per-shape regret from ~13% to ~4% (warm and cold).
    // Similarly the NEON value is inherited from Graviton2 machine.
    const float l2_occupancy = is_sve ? 0.50f : 0.70f;

    const bool should_adjust_by_l2 = static_cast<float>(As + Bs + Cs)
            >= l2_occupancy * static_cast<float>(l2_cache_size);

    dim_t k1_block = K1;
    dim_t k2_block = K2;

    if (should_adjust_by_l2) {
        int block_size = (l2_cache_size * l2_occupancy)
                / ((M + n_block) * src_layer_type_size);

        if (is_xf16) {
            block_size -= (block_size % 2);
            block_size = nstl::max(block_size, 0);
        }

        if (block_size) {
            k1_block = nstl::min(K1, static_cast<dim_t>(block_size));
            k2_block = nstl::min(K2, static_cast<dim_t>(block_size));
        }
    }

    return std::make_pair(k1_block, k2_block);
}

dim_t brgemm_calc_m_block(alg_kind_t cell_kind, prop_kind_t aprop, dim_t nthr,
        dim_t M, dim_t N_blocks, bool is_f32, float work_by_N, dim_t As,
        dim_t Bs, dim_t Cs, dim_t l2_cache_size) {

    if (cell_kind == alg_kind::vanilla_rnn
            || (cell_kind == alg_kind::vanilla_lstm
                    && aprop == prop_kind::backward))
        return brgemm_calc_m_block_vanilla_rnn(
                nthr, M, N_blocks, work_by_N, As, Bs, Cs, l2_cache_size);
    else
        return brgemm_calc_m_block_lstm(
                nthr, M, N_blocks, is_f32, work_by_N, As, Cs, l2_cache_size);
}

dim_t brgemm_calc_m_block_vanilla_rnn(dim_t nthr, dim_t M, dim_t N_blocks,
        float work_by_N, dim_t As, dim_t Bs, dim_t Cs, dim_t l2_cache_size) {

    const float decimal_n_factor = work_by_N - std::floor(work_by_N);
    static constexpr float thread_balance_threashold = 0.9f;

    dim_t m_block = M;

    if (work_by_N < 1.0f)
        return adjust_m_block_lstm(nthr, M, N_blocks);
    else if (decimal_n_factor < thread_balance_threashold
            && decimal_n_factor != 0.0f) {

        const dim_t m_block_start = M / 2;
        const dim_t m_block_end = 8;
        float max_decimal_mn = 0.0f;
        dim_t best_candidate = m_block_start;
        bool found_best_solution = false;

        for (dim_t m_block_it = m_block_start; m_block_it >= m_block_end;
                m_block_it--) {
            if (M % m_block_it == 0) {
                const auto m_blocks = M / m_block_it;
                const auto work_by_MN
                        = static_cast<float>(m_blocks * N_blocks) / nthr;
                const float work_by_MN_decimal
                        = work_by_MN - std::floor(work_by_MN);

                static constexpr float tolerance = 0.01f;
                if (work_by_MN_decimal > (max_decimal_mn + tolerance)) {
                    best_candidate = m_block_it;
                    max_decimal_mn = work_by_MN_decimal;
                }

                if (work_by_MN_decimal >= thread_balance_threashold
                        || work_by_MN_decimal == 0.0f) {
                    m_block = m_block_it;
                    found_best_solution = true;
                    break;
                }
            }
        }

        if (!found_best_solution) {
            if ((decimal_n_factor < max_decimal_mn)
                    || (static_cast<float>(As)
                            > (0.5f * static_cast<float>(l2_cache_size)))) {
                m_block = best_candidate;
            }
        }
    }

    return m_block;
}

dim_t brgemm_calc_m_block_lstm(dim_t nthr, dim_t M, dim_t N_blocks, bool is_f32,
        float work_by_N, dim_t As, dim_t Cs, dim_t l2_cache_size) {

    const bool adj_by_l2 = is_f32
            ? true
            : (static_cast<float>(As + Cs)
                      < 0.6f * static_cast<float>(l2_cache_size));

    if (work_by_N > 2.0f || (work_by_N > 1.0f && adj_by_l2))
        return M;
    else
        return adjust_m_block_lstm(nthr, M, N_blocks);
}

dim_t adjust_m_block_lstm(dim_t nthr, dim_t M, dim_t N_blocks) {

    // Target up to 4x the minimum number of M-blocks needed to cover all threads,
    // giving the scheduler slack to balance load.
    const dim_t max_m_blocks = 4 * utils::div_up(nthr, N_blocks);

    // Upper bound on rows per M-block. This is thread-coupled, not vector-length
    // coupled: it only takes effect through max_m_blocks (above), which scales with
    // nthr. A benchdnn VANILLA_LSTM sweep on Graviton3 confirmed this -- the best
    // value shifts with core count (the cap is near-inert at 64 threads but the
    // inherited 32 degrades badly at 32). 40 is the most robust single value across
    // the core counts tested.
    const dim_t max_m_value = 40;
    const dim_t max_M
            = nstl::min(max_m_value, nstl::max((dim_t)1, M / max_m_blocks));
    const dim_t min_M = 4;

    dim_t m_block = 1;
    for (dim_t m = max_M; m >= min_M; m--)
        if (M % m == 0) {
            m_block = m;
            break;
        }

    if (m_block == 1) m_block = M;
    return m_block;
}

dim_t brgemm_calc_n_block(
        const cpu::rnn_utils::rnn_conf_t &rnn, alg_kind_t cell_kind) {

    const int simd_elems = isa_max_vlen(rnn.brgemm_isa) / (int)sizeof(float);

    if (rnn.brgemm_isa == asimd && rnn.M == 1
            && utils::one_of(
                    cell_kind, alg_kind::vanilla_lstm, alg_kind::lbr_gru))
        return 4 * simd_elems;
    else
        return 2 * simd_elems;
}

} // namespace

void rnn_brgemm_base_t::init_scratchpad(const cpu::rnn_utils::rnn_conf_t &rnn,
        memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
        dim_t gemm_acc_align) {

    MAYBE_UNUSED(gemm_acc_type_size);
    MAYBE_UNUSED(gemm_acc_align);

    using namespace memory_tracking::names;
    const int max_K_Block
            = nstl::max(rnn.KB1_blocks + 1,
                      nstl::max(rnn.KBproj_blocks + 1, rnn.KB2_blocks + 1))
            * (rnn.brgemm_fwd_iter_layer_fuse_possible ? 2 : 1);

    scratchpad.template book<aarch64::brgemm_batch_element_t>(
            key_brgemm_primitive_batch, max_K_Block * rnn.nthr);
}

status_t rnn_brgemm_t<prop_kind::forward>::configure_brgemm(
        cpu::rnn_utils::rnn_conf_t &rnn, alg_kind_t cell_kind,
        dim_t src_layer_type_size, dim_t scratch_type_size) {

    using namespace cpu::rnn_utils;

    // no asimd support for now
    if (!mayiuse(sve)) return status::unimplemented;

    // no int8 brgemm RNN support for now
    if (rnn.is_int8_conf() || rnn.is_cell_dt_int8())
        return status::unimplemented;

    // For now only support vanilla RNN/LSTM/GRU/AUGRU
    if (!utils::one_of(cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm,
                alg_kind::vanilla_gru, alg_kind::vanilla_augru))
        return status::unimplemented;

    rnn.M = rnn.mb;
    rnn.N = rnn.dhc;
    rnn.K1 = rnn.slc;
    rnn.K2 = rnn.sic;

    const auto is_int8 = rnn.is_cell_dt_int8();
    const auto is_xf16 = rnn.is_cell_dt_xf16();

    const dim_t padding = (is_int8 ? 4 : (is_xf16 ? 2 : 1));
    rnn.K1padded = utils::rnd_up(rnn.K1, padding);
    rnn.K2padded = utils::rnd_up(rnn.K2, padding);

    rnn.brgemm_isa = brgemm_calc_isa(rnn, rnn.K1, rnn.K2);
    if (rnn.brgemm_isa == isa_undef) return status::unimplemented;

    if (rnn.brgemm_isa == asimd
            && (rnn.is_lstm_peephole || rnn.is_lstm_projection))
        return status::unimplemented;

    if (!IMPLICATION(rnn.is_f32_conf(), rnn.is_cell_dt_f32()))
        return status::unimplemented;

    rnn.nthr = dnnl_get_max_threads();

    rnn.n_block = brgemm_calc_n_block(rnn, cell_kind);
    rnn.N_blocks = utils::div_up(rnn.N, rnn.n_block);
    rnn.n_tail = rnn.N % rnn.n_block;

    const float work_by_N
            = static_cast<float>(rnn.N_blocks) / static_cast<float>(rnn.nthr);

    const dim_t l2_cache_size = platform::get_per_core_cache_size(2);

    const dim_t As = src_layer_type_size * rnn.M * (nstl::max(rnn.K1, rnn.K2));
    const dim_t Bs
            = src_layer_type_size * (nstl::max(rnn.K1, rnn.K2)) * rnn.n_block;
    const dim_t Cs
            = scratch_type_size * (rnn.n_gates + 1) * (rnn.M * rnn.n_block);

    std::tie(rnn.k1_block, rnn.k2_block) = brgemm_calc_k_block(rnn, rnn.K1,
            rnn.K2, rnn.M, rnn.n_block, cell_kind, src_layer_type_size, As, Bs,
            Cs, l2_cache_size, rnn.brgemm_isa);

    rnn.KB1_blocks = rnn.K1 / rnn.k1_block;
    rnn.k1_tail = rnn.K1 % rnn.k1_block;
    rnn.KB2_blocks = rnn.K2 / rnn.k2_block;
    rnn.k2_tail = rnn.K2 % rnn.k2_block;

    rnn.m_block = brgemm_calc_m_block(cell_kind, prop_kind::forward, rnn.nthr,
            rnn.M, rnn.N_blocks, rnn.is_cell_dt_f32(), work_by_N, As, Bs, Cs,
            l2_cache_size);
    rnn.M_blocks = rnn.M / rnn.m_block;

    rnn.unfused_post_gemm = false;

    rnn.LDA1[0] = rnn.src_layer_ld_;
    rnn.LDA1[1] = rnn.dst_iter_ld_;
    rnn.LDA1[2] = rnn.ws_states_layer_ld;

    rnn.LDA2[0] = rnn.src_iter_ld_;
    rnn.LDA2[1] = rnn.dst_layer_ld_;
    rnn.LDA2[2] = rnn.ws_states_iter_ld;

    rnn.LDA2_2[0] = rnn.dst_layer_ld_;
    rnn.LDA2_2[1] = rnn.dst_iter_ld_;
    rnn.LDA2_2[2] = rnn.ws_states_layer_ld;
    rnn.LDA2_2[3] = rnn.ws_states_iter_ld;

    rnn.LDB1 = rnn.n_block;
    rnn.LDB2 = rnn.n_block;
    rnn.LDC = rnn.scratch_gates_ld;

    auto get_dim = [&](dim_t block, dim_t tail) {
        return (block == 0) ? tail : block;
    };

    dim_t n_block = nstl::min(rnn.N, rnn.n_block);
    dim_t n_tail = nstl::min(rnn.N, rnn.n_tail);

    if (rnn.LDA1[0] < rnn.k1_block || rnn.LDA1[1] < rnn.k1_block
            || rnn.LDA1[2] < rnn.k1_block)
        return status::unimplemented;

    if (rnn.LDA2[0] < rnn.k2_block || rnn.LDA2[1] < rnn.k2_block
            || rnn.LDA2[2] < rnn.k2_block)
        return status::unimplemented;

    if (rnn.LDB1 < get_dim(n_block, n_tail)
            && rnn.LDB2 < get_dim(n_block, n_tail))
        return status::unimplemented;

    if (rnn.LDC < get_dim(n_block, n_tail)) return status::unimplemented;

    rnn.KBproj_blocks = 0;
    rnn.kproj_tail = 0;
    rnn.kproj_block = 0;

    if (rnn.is_lstm_projection) {
        rnn.Nproj = rnn.dic;
        rnn.Nproj_blocks = utils::div_up(rnn.Nproj, rnn.n_block);
        rnn.nproj_tail = rnn.Nproj % rnn.n_block;

        rnn.Kproj = rnn.dhc;
        rnn.Kprojpadded = utils::rnd_up(rnn.Kproj, padding);

        rnn.kproj_block = rnn.Kproj;
        rnn.KBproj_blocks = rnn.Kproj / rnn.kproj_block;

        rnn.LDAproj = rnn.proj_ht_ld;
        rnn.LDBproj = rnn.n_block;

        if (rnn.dt_conf != cpu::rnn_utils::all_f32) {
            rnn.LDCproj[0] = rnn.scratch_gates_ld;
        } else {
            rnn.LDCproj[0] = rnn.scratch_ht_ld;
            rnn.LDCproj[1] = rnn.dst_layer_ld_;
            rnn.LDCproj[2] = rnn.dst_iter_ld_;
            rnn.LDCproj[3] = rnn.ws_states_layer_ld;
        }

        dim_t n_block = nstl::min(rnn.Nproj, rnn.n_block);
        dim_t n_tail = nstl::min(rnn.Nproj, rnn.nproj_tail);

        bool check_LDC = false;
        if (rnn.dt_conf != cpu::rnn_utils::all_f32) {
            check_LDC = rnn.LDCproj[0] < get_dim(n_block, n_tail);
        } else {
            check_LDC = rnn.LDCproj[0] < get_dim(n_block, n_tail)
                    && rnn.LDCproj[1] < get_dim(n_block, n_tail)
                    && rnn.LDCproj[2] < get_dim(n_block, n_tail)
                    && rnn.LDCproj[3] < get_dim(n_block, n_tail);
        }

        if (rnn.LDAproj < rnn.kproj_block
                || rnn.LDBproj < get_dim(n_block, n_tail) || check_LDC)
            return status::unimplemented;
    }

    const bool mlc_cell_type_ok
            = (cell_kind == alg_kind::vanilla_lstm && !rnn.is_lstm_projection
                      && !rnn.is_lstm_peephole)
            || (cell_kind == alg_kind::lbr_gru && rnn.brgemm_isa == asimd);

    const int mlc_mb_max_threshold = 1;
    const int mlc_n_iter_min_threshold = 2;
    const int mlc_n_layer_max_threshold = 1;

    const bool mlc_problem_shape_ok = rnn.mb <= mlc_mb_max_threshold
            && rnn.n_iter >= mlc_n_iter_min_threshold
            && rnn.n_layer <= mlc_n_layer_max_threshold;

    const bool mlc_m_dim_adjustment_not_required
            = IMPLICATION(rnn.skip_dst_iter_copy(),
                    rnn.skip_src_layer_copy() && rnn.n_layer == 1);

    const bool merged_layer_compute_applicable = rnn.src_layer_is_trivial_stride
            && mlc_cell_type_ok && mlc_problem_shape_ok
            && mlc_m_dim_adjustment_not_required;

    if (merged_layer_compute_applicable) {
        rnn.merge_gemm_layer = true;
        const int n_iters_to_merge = rnn.n_iter;
        rnn.Mlayermerged = rnn.mb * n_iters_to_merge;
        rnn.mlayermerged_block = brgemm_calc_m_block(cell_kind,
                prop_kind::forward, rnn.nthr, rnn.Mlayermerged, rnn.N_blocks,
                rnn.is_cell_dt_f32(), work_by_N, As, Bs, Cs, l2_cache_size);
        rnn.Mlayermerged_blocks = rnn.Mlayermerged / rnn.mlayermerged_block;
    }

    rnn.brgemm_fwd_iter_layer_fuse_possible
            = rnn.slc == rnn.sic && !rnn.merge_gemm_layer;

    if (!rnn.is_orig_gru) {
        rnn.loop_order = brgemm_rnn_execute_loop_order_t::nblk_mblk;
    }

    return status::success;
}

status_t init_brgemm_kernel(aarch64::brgemm_desc_t *desc,
        aarch64::cpu_isa_t isa, impl::data_type_t src_type,
        impl::data_type_t weights_type,
        std::unique_ptr<aarch64::brgemm_kernel_t> &ker, dim_t M, dim_t N,
        dim_t K, dim_t LDA, dim_t LDB, dim_t LDC, float beta, dim_t max_bs,
        dim_t hint_expected_A_size = LLONG_MAX,
        dim_t hint_expected_B_size = LLONG_MAX,
        dim_t hint_expected_C_size = LLONG_MAX) {

    bool transA = false;
    bool transB = false;
    aarch64::brgemm_layout_t layout = aarch64::brgemm_row_major;

    CHECK(brgemm_desc_init(desc, isa, aarch64::brgemm_addr, src_type,
            weights_type, transA, transB, layout, 1.0f, beta, LDA, LDB, LDC, M,
            N, K));

    aarch64::brgemm_attr_t brgattr {};
    brgattr.hint_expected_A_size = hint_expected_A_size;
    brgattr.hint_expected_B_size = hint_expected_B_size;
    brgattr.hint_expected_C_size = hint_expected_C_size;
    brgattr.max_bs = static_cast<int>(max_bs);
    brgattr.max_top_vpad = 0;
    brgattr.max_bottom_vpad = 0;
    brgattr.use_uker = false;
    brgattr.var_bs = true;

    CHECK(brgemm_desc_set_attr(desc, brgattr));
    CHECK(brgemm_desc_finalize(desc));

    aarch64::brgemm_kernel_t *_t_ptr = nullptr;
    CHECK(brgemm_kernel_create(&_t_ptr, *desc));
    CHECK(safe_ptr_assign<aarch64::brgemm_kernel_t>(ker, _t_ptr));

    return status::success;
}

status_t rnn_brgemm_t<prop_kind::forward>::init_kernels(
        const cpu::rnn_utils::rnn_conf_t &rnn, data_type_t src_type,
        data_type_t weights_type) {

    const auto init_brgemm
            = [&](aarch64::brgemm_desc_t *desc, aarch64::cpu_isa_t isa,
                      std::unique_ptr<aarch64::brgemm_kernel_t> &ker, dim_t M,
                      dim_t N, dim_t K, dim_t LDA, dim_t LDB, dim_t LDC,
                      float beta, dim_t max_bs) {
        return init_brgemm_kernel(desc, isa, src_type, weights_type, ker, M, N,
                K, LDA, LDB, LDC, beta, max_bs);
    };

    const int brgemm_n = nstl::min(rnn.N, rnn.n_block);
    const int brgemm_n_tail = nstl::min(rnn.N, rnn.n_tail);
    const int max_bs_factor = rnn.brgemm_fwd_iter_layer_fuse_possible ? 2 : 1;

    for (int i = 0; i < num_base_kernels_; i++) {
        if (rnn.merge_gemm_layer) {
            CHECK(init_brgemm(&desc_layermerged_b0_[i], rnn.brgemm_isa,
                    kernel_layermerged_b0_[i], rnn.mlayermerged_block, brgemm_n,
                    rnn.k1_block, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0f,
                    rnn.KB1_blocks));
        } else {
            CHECK(init_brgemm(&desc_layer_b0_[i], rnn.brgemm_isa,
                    kernel_layer_b0_[i], rnn.m_block, brgemm_n, rnn.k1_block,
                    rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0f,
                    max_bs_factor * rnn.KB1_blocks));
        }

        CHECK(init_brgemm(&desc_iter_b0_[i], rnn.brgemm_isa, kernel_iter_b0_[i],
                rnn.m_block, brgemm_n, rnn.k2_block, rnn.LDA2[i], rnn.LDB2,
                rnn.LDC, 0.0f, rnn.KB2_blocks));

        CHECK(init_brgemm(&desc_iter_b1_[i], rnn.brgemm_isa, kernel_iter_b1_[i],
                rnn.m_block, brgemm_n, rnn.k2_block, rnn.LDA2[i], rnn.LDB2,
                rnn.LDC, 1.0f, rnn.KB2_blocks));

        if (rnn.n_tail) {
            if (rnn.merge_gemm_layer) {
                CHECK(init_brgemm(&desc_layermerged_N_tail_b0_[i],
                        rnn.brgemm_isa, kernel_layermerged_N_tail_b0_[i],
                        rnn.mlayermerged_block, brgemm_n_tail, rnn.k1_block,
                        rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0f, rnn.KB1_blocks));
            } else {
                CHECK(init_brgemm(&desc_layer_N_tail_b0_[i], rnn.brgemm_isa,
                        kernel_layer_N_tail_b0_[i], rnn.m_block, brgemm_n_tail,
                        rnn.k1_block, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 0.0f,
                        max_bs_factor * rnn.KB1_blocks));
            }

            CHECK(init_brgemm(&desc_iter_N_tail_b0_[i], rnn.brgemm_isa,
                    kernel_iter_N_tail_b0_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_block, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 0.0f,
                    rnn.KB2_blocks));

            CHECK(init_brgemm(&desc_iter_N_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_N_tail_b1_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_block, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0f,
                    rnn.KB2_blocks));
        }

        if (rnn.k1_tail) {
            if (rnn.merge_gemm_layer) {
                CHECK(init_brgemm(&desc_layermerged_K1_tail_b1_[i],
                        rnn.brgemm_isa, kernel_layermerged_K1_tail_b1_[i],
                        rnn.mlayermerged_block, brgemm_n, rnn.k1_tail,
                        rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0f, 1));
            } else {
                CHECK(init_brgemm(&desc_layer_K1_tail_b1_[i], rnn.brgemm_isa,
                        kernel_layer_K1_tail_b1_[i], rnn.m_block, brgemm_n,
                        rnn.k1_tail, rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0f,
                        max_bs_factor * 1));
            }
        }

        if (rnn.k2_tail) {
            CHECK(init_brgemm(&desc_iter_K2_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_K2_tail_b1_[i], rnn.m_block, brgemm_n,
                    rnn.k2_tail, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0f, 1));
        }

        if (rnn.k1_tail && rnn.n_tail) {
            if (rnn.merge_gemm_layer) {
                CHECK(init_brgemm(&desc_layermerged_NK1_tail_b1_[i],
                        rnn.brgemm_isa, kernel_layermerged_NK1_tail_b1_[i],
                        rnn.mlayermerged_block, brgemm_n_tail, rnn.k1_tail,
                        rnn.LDA1[i], rnn.LDB1, rnn.LDC, 1.0f, 1));
            } else {
                CHECK(init_brgemm(&desc_layer_NK1_tail_b1_[i], rnn.brgemm_isa,
                        kernel_layer_NK1_tail_b1_[i], rnn.m_block,
                        brgemm_n_tail, rnn.k1_tail, rnn.LDA1[i], rnn.LDB1,
                        rnn.LDC, 1.0f, max_bs_factor * 1));
            }
        }

        if (rnn.k2_tail && rnn.n_tail) {
            CHECK(init_brgemm(&desc_iter_NK2_tail_b1_[i], rnn.brgemm_isa,
                    kernel_iter_NK2_tail_b1_[i], rnn.m_block, brgemm_n_tail,
                    rnn.k2_tail, rnn.LDA2[i], rnn.LDB2, rnn.LDC, 1.0f, 1));
        }
    }

    if (rnn.is_orig_gru) {
        for (int i = 0; i < num_vanilla_gru_iter_part2_kernels_; i++) {
            CHECK(init_brgemm(&desc_iter_p2_b1_[i], rnn.brgemm_isa,
                    kernel_iter_p2_b1_[i], rnn.m_block, brgemm_n, rnn.k2_block,
                    rnn.LDA2_2[i], rnn.LDB2, rnn.LDC, 1.0f, rnn.KB2_blocks));
            if (rnn.n_tail) {
                CHECK(init_brgemm(&desc_iter_p2_N_tail_b1_[i], rnn.brgemm_isa,
                        kernel_iter_p2_N_tail_b1_[i], rnn.m_block,
                        brgemm_n_tail, rnn.k2_block, rnn.LDA2_2[i], rnn.LDB2,
                        rnn.LDC, 1.0f, rnn.KB2_blocks));
            }
            if (rnn.k2_tail) {
                CHECK(init_brgemm(&desc_iter_p2_K2_tail_b1_[i], rnn.brgemm_isa,
                        kernel_iter_p2_K2_tail_b1_[i], rnn.m_block, brgemm_n,
                        rnn.k2_tail, rnn.LDA2_2[i], rnn.LDB2, rnn.LDC, 1.0f,
                        1));
            }
            if (rnn.k2_tail && rnn.n_tail) {
                CHECK(init_brgemm(&desc_iter_p2_NK2_tail_b1_[i], rnn.brgemm_isa,
                        kernel_iter_p2_NK2_tail_b1_[i], rnn.m_block,
                        brgemm_n_tail, rnn.k2_tail, rnn.LDA2_2[i], rnn.LDB2,
                        rnn.LDC, 1.0f, 1));
            }
        }
    }

    if (rnn.is_lstm_projection) {
        const dim_t brgemm_np = nstl::min(rnn.Nproj, rnn.n_block);
        const dim_t brgemm_np_tail = nstl::min(rnn.Nproj, rnn.nproj_tail);
        const int n_kernel = (rnn.dt_conf == cpu::rnn_utils::all_f32)
                ? num_proj_kernels_
                : 1;

        for (int i = 0; i < n_kernel; i++) {
            CHECK(init_brgemm(&desc_proj_b0_[i], rnn.brgemm_isa,
                    kernel_proj_b0_[i], rnn.m_block, brgemm_np, rnn.kproj_block,
                    rnn.LDAproj, rnn.LDBproj, rnn.LDCproj[i], 0.0f,
                    rnn.KBproj_blocks));

            if (rnn.nproj_tail) {
                CHECK(init_brgemm(&desc_proj_N_tail_b0_[i], rnn.brgemm_isa,
                        kernel_proj_N_tail_b0_[i], rnn.m_block, brgemm_np_tail,
                        rnn.kproj_block, rnn.LDAproj, rnn.LDBproj,
                        rnn.LDCproj[i], 0.0f, rnn.KBproj_blocks));

                CHECK(init_brgemm(&desc_proj_N_tail_b1_[i], rnn.brgemm_isa,
                        kernel_proj_N_tail_b1_[i], rnn.m_block, brgemm_np_tail,
                        rnn.kproj_block, rnn.LDAproj, rnn.LDBproj,
                        rnn.LDCproj[i], 1.0f, rnn.KBproj_blocks));
            }
        }
    }

    return status::success;
}

void rnn_brgemm_t<prop_kind::backward>::init_scratchpad(
        const cpu::rnn_utils::rnn_conf_t &rnn,
        memory_tracking::registrar_t &scratchpad, dim_t gemm_acc_type_size,
        dim_t gemm_acc_align) {

    rnn_brgemm_base_t::init_scratchpad(
            rnn, scratchpad, gemm_acc_type_size, gemm_acc_align);

    using namespace memory_tracking::names;

    const auto data_size
            = rnn.is_xf16_conf() ? sizeof(bfloat16_t) : sizeof(float);
    const auto &d_wei = rnn.diff_wei_brgemm;

    const auto scratch_gates_blocked_per_thr = d_wei.Kpadded * d_wei.n_block;
    const auto scratch_gates_blocked_size
            = rnn.nthr * scratch_gates_blocked_per_thr;
    scratchpad.book(key_rnn_gates_blocked, scratch_gates_blocked_size,
            data_size, gemm_acc_align);

    const auto scratch_src_layer_size = d_wei.global_transpose
            ? d_wei.M_layer * d_wei.Kpadded
            : rnn.nthr * nstl::min(d_wei.m_block, d_wei.M_layer)
                    * d_wei.Kpadded;
    scratchpad.book(key_rnn_src_layer_trans, scratch_src_layer_size, data_size,
            gemm_acc_align);

    const auto scratch_src_iter_size = d_wei.global_transpose
            ? d_wei.M_iter * d_wei.Kpadded
            : rnn.nthr * nstl::min(d_wei.m_block, d_wei.M_iter) * d_wei.Kpadded;
    scratchpad.book(key_rnn_src_iter_trans, scratch_src_iter_size, data_size,
            gemm_acc_align);
}

status_t rnn_brgemm_t<prop_kind::backward>::configure_brgemm(
        cpu::rnn_utils::rnn_conf_t &rnn, alg_kind_t cell_kind,
        dim_t src_layer_type_size, dim_t scratch_type_size) {

    using namespace cpu::rnn_utils;

    // no asimd support for now
    if (!mayiuse(sve)) return status::unimplemented;

    if (rnn.is_cell_dt_xf16()) return status::unimplemented;

    // no int8 brgemm RNN support for now
    if (rnn.is_int8_conf() || rnn.is_cell_dt_int8())
        return status::unimplemented;

    // FOr now only support vanilla RNN/LSTM/GRU/AUGRU
    if (!utils::one_of(cell_kind, alg_kind::vanilla_rnn, alg_kind::vanilla_lstm,
                alg_kind::vanilla_gru, alg_kind::vanilla_augru))
        return status::unimplemented;

    auto &diff_src_conf = rnn.diff_src_brgemm;
    diff_src_conf.M = rnn.mb;
    diff_src_conf.N_iter = rnn.sic;
    diff_src_conf.N_layer = rnn.slc;
    diff_src_conf.N = nstl::max(diff_src_conf.N_iter, diff_src_conf.N_layer);
    diff_src_conf.K = rnn.dhc;

    rnn.nthr = dnnl_get_max_threads();

    diff_src_conf.n_block = 32;
    diff_src_conf.N_blocks
            = utils::div_up(diff_src_conf.N, diff_src_conf.n_block);
    diff_src_conf.n_tail = diff_src_conf.N % diff_src_conf.n_block;

    diff_src_conf.N_layer_blocks
            = utils::div_up(diff_src_conf.N_layer, diff_src_conf.n_block);
    diff_src_conf.n_layer_tail = diff_src_conf.N_layer % diff_src_conf.n_block;

    diff_src_conf.N_iter_blocks
            = utils::div_up(diff_src_conf.N_iter, diff_src_conf.n_block);
    diff_src_conf.n_iter_tail = diff_src_conf.N_iter % diff_src_conf.n_block;

    const float work_by_N = static_cast<float>(diff_src_conf.N_blocks)
            / static_cast<float>(rnn.nthr);

    const dim_t l2_cache_size = platform::get_per_core_cache_size(2);

    const dim_t As = src_layer_type_size * diff_src_conf.M * diff_src_conf.K;
    const dim_t Bs
            = src_layer_type_size * diff_src_conf.K * diff_src_conf.n_block;
    const dim_t Cs = scratch_type_size * (rnn.n_gates + 1)
            * (diff_src_conf.M * diff_src_conf.n_block);

    const auto is_xf16 = rnn.is_cell_dt_xf16();
    const dim_t padding = is_xf16 ? 2 : 1;

    diff_src_conf.Kpadded = utils::rnd_up(diff_src_conf.K, padding);
    diff_src_conf.isa = brgemm_calc_isa(rnn, diff_src_conf.K, diff_src_conf.K);
    if (diff_src_conf.isa == isa_undef) return status::unimplemented;

    diff_src_conf.gates_block = rnn.n_gates;

    std::tie(diff_src_conf.k_block, std::ignore) = brgemm_calc_k_block(rnn,
            diff_src_conf.K, diff_src_conf.K, diff_src_conf.M,
            diff_src_conf.n_block, cell_kind, src_layer_type_size, As, Bs, Cs,
            l2_cache_size, diff_src_conf.isa);

    const dim_t K_blocks_per_gate = diff_src_conf.K / diff_src_conf.k_block;
    diff_src_conf.K_blocks
            = K_blocks_per_gate * rnn.n_gates; // total across gates
    diff_src_conf.k_tail = diff_src_conf.K % diff_src_conf.k_block;

    diff_src_conf.m_block = brgemm_calc_m_block(cell_kind, prop_kind::backward,
            rnn.nthr, diff_src_conf.M, diff_src_conf.N_blocks,
            rnn.is_cell_dt_f32(), work_by_N, As, Bs, Cs, l2_cache_size);
    diff_src_conf.M_blocks = diff_src_conf.M / diff_src_conf.m_block;

    diff_src_conf.LDA = rnn.scratch_gates_ld;
    diff_src_conf.LDB = diff_src_conf.n_block;
    diff_src_conf.LDC = rnn.ws_diff_states_iter_ld;

    if (diff_src_conf.LDA < diff_src_conf.k_block) return status::unimplemented;
    const dim_t n_block = nstl::min(diff_src_conf.N, diff_src_conf.n_block);
    if (diff_src_conf.LDB < n_block) return status::unimplemented;
    if (diff_src_conf.LDC < n_block) return status::unimplemented;

    rnn.KBproj_blocks = 0;
    rnn.kproj_tail = 0;
    rnn.kproj_block = 0;

    auto &diff_wei_conf = rnn.diff_wei_brgemm;
    diff_wei_conf.global_transpose = rnn.mb > 1;
    diff_wei_conf.M_iter = rnn.sic;
    diff_wei_conf.M_layer = rnn.slc;
    diff_wei_conf.M = nstl::max(rnn.sic, rnn.slc);
    diff_wei_conf.N = rnn.dhc * rnn.n_gates;
    diff_wei_conf.K = (scratch_type_size != (dim_t)sizeof(float))
            ? utils::rnd_up(rnn.mb, 2)
            : rnn.mb;
    diff_wei_conf.Kpadded = utils::rnd_up(diff_wei_conf.K, padding);
    diff_wei_conf.isa = brgemm_calc_isa(rnn, diff_wei_conf.K, diff_wei_conf.K);
    if (diff_wei_conf.isa == isa_undef) return status::unimplemented;

    diff_wei_conf.n_block = 32;
    diff_wei_conf.N_blocks
            = utils::div_up(diff_wei_conf.N, diff_wei_conf.n_block);
    diff_wei_conf.n_tail = diff_wei_conf.N % diff_wei_conf.n_block;

    const dim_t As_wei
            = src_layer_type_size * diff_wei_conf.M * diff_wei_conf.K;
    const dim_t Bs_wei
            = src_layer_type_size * diff_wei_conf.K * diff_wei_conf.n_block;
    const dim_t Cs_wei = scratch_type_size * (rnn.n_gates + 1)
            * (diff_wei_conf.M * diff_wei_conf.n_block);

    std::tie(diff_wei_conf.k_block, std::ignore) = brgemm_calc_k_block(rnn,
            diff_wei_conf.K, diff_wei_conf.K, diff_wei_conf.M,
            diff_wei_conf.n_block, cell_kind, src_layer_type_size, As_wei,
            Bs_wei, Cs_wei, l2_cache_size, diff_wei_conf.isa);

    diff_wei_conf.K_blocks = diff_wei_conf.K / diff_wei_conf.k_block;
    diff_wei_conf.k_tail = diff_wei_conf.K % diff_wei_conf.k_block;

    if (diff_wei_conf.M_iter != diff_wei_conf.M_layer) {
        diff_wei_conf.m_block = diff_wei_conf.M;
        diff_wei_conf.M_blocks = 1;
    } else {
        const float work_by_N_wei = static_cast<float>(diff_wei_conf.N_blocks)
                / static_cast<float>(rnn.nthr);
        diff_wei_conf.m_block = brgemm_calc_m_block(cell_kind,
                prop_kind::backward, rnn.nthr, diff_wei_conf.M,
                diff_wei_conf.N_blocks, rnn.is_cell_dt_f32(), work_by_N_wei,
                As_wei, Bs_wei, Cs_wei, l2_cache_size);
        diff_wei_conf.M_blocks = diff_wei_conf.M / diff_wei_conf.m_block;
    }

    diff_wei_conf.LDA_layer = diff_wei_conf.K;
    diff_wei_conf.LDA_iter = diff_wei_conf.K;
    diff_wei_conf.LDB = diff_wei_conf.n_block;
    diff_wei_conf.LDC_iter = rnn.diff_weights_iter_ld;
    diff_wei_conf.LDC_layer = rnn.diff_weights_layer_ld;

    if (diff_wei_conf.LDA_layer < diff_wei_conf.k_block
            || diff_wei_conf.LDA_iter < diff_wei_conf.k_block)
        return status::unimplemented;

    if (rnn.is_lstm_peephole) { configure_brgemm_peephole(rnn); }

    rnn.M = nstl::max(diff_wei_conf.M, diff_src_conf.M);
    rnn.N = nstl::max(diff_wei_conf.N, diff_src_conf.N);
    rnn.K1 = nstl::max(diff_wei_conf.K, diff_src_conf.K);
    rnn.K2 = rnn.K1;
    rnn.m_block = nstl::max(diff_wei_conf.m_block, diff_src_conf.m_block);
    rnn.M_blocks = nstl::max(diff_wei_conf.M_blocks, diff_src_conf.M_blocks);
    rnn.n_block = nstl::max(diff_wei_conf.n_block, diff_src_conf.n_block);
    rnn.N_blocks = nstl::max(diff_wei_conf.N_blocks, diff_src_conf.N_blocks);
    rnn.n_tail = nstl::max(diff_wei_conf.n_tail, diff_src_conf.n_tail);
    rnn.k1_block = nstl::max(diff_wei_conf.k_block, diff_src_conf.k_block);
    rnn.k2_block = rnn.k1_block;
    rnn.k1_tail = nstl::max(diff_wei_conf.k_tail, diff_src_conf.k_tail);
    rnn.k2_tail = rnn.k1_tail;
    rnn.KB1_blocks = nstl::max(diff_wei_conf.K_blocks, diff_src_conf.K_blocks);
    rnn.KB2_blocks = rnn.KB1_blocks;
    rnn.K1padded = nstl::max(diff_wei_conf.Kpadded, diff_src_conf.Kpadded);
    rnn.K2padded = rnn.K1padded;
    rnn.unfused_post_gemm = true;

    rnn.brgemm_isa = is_superset(diff_wei_conf.isa, diff_src_conf.isa)
            ? diff_wei_conf.isa
            : diff_src_conf.isa;

    if (!rnn.is_orig_gru) {
        rnn.diff_src_brgemm.loop_order
                = brgemm_rnn_execute_loop_order_t::nblk_mblk;
        rnn.diff_wei_brgemm.loop_order
                = brgemm_rnn_execute_loop_order_t::nblk_mblk;
    }

    return status::success;
}

static dim_t divide_block_to_improve_thread_balance(
        const dim_t initial_work_amount, const dim_t division_block,
        const dim_t nthr) {

    const float nthr_f = static_cast<float>(nthr);
    const float initial_work = static_cast<float>(initial_work_amount) / nthr_f;
    const float decimal_initial_factor
            = initial_work - std::floor(initial_work);

    static constexpr float thread_balance_threashold = 0.8f;
    static constexpr float tolerance = 0.01f;

    float max_decimal_factor = -1.0f;
    dim_t best_candidate = -1;
    bool found_best_solution = false;

    if (decimal_initial_factor < thread_balance_threashold
            && decimal_initial_factor != 0.0f) {
        for (const int block_size : {4096, 2048, 1024, 512, 256, 128, 64, 32}) {
            if (division_block <= block_size) continue;
            const auto blocks = utils::div_up(division_block, block_size);
            const float work
                    = static_cast<float>(initial_work_amount * blocks) / nthr_f;
            const float work_decimal = work - std::floor(work);

            if (work_decimal == 0.0f
                    || (max_decimal_factor != 0.0f ? work_decimal
                                            > (max_decimal_factor + tolerance)
                                                   : work_decimal
                                            >= thread_balance_threashold)) {
                best_candidate = block_size;
                max_decimal_factor = work_decimal;
            }

            if (work >= nthr_f
                    && (work_decimal >= thread_balance_threashold
                            || work_decimal == 0.0f)) {
                found_best_solution = true;
                break;
            }
        }
    }

    if (found_best_solution
            || (!found_best_solution
                    && max_decimal_factor
                            > decimal_initial_factor + tolerance)) {
        return best_candidate;
    }

    return division_block;
}

void rnn_brgemm_t<prop_kind::backward>::configure_brgemm_peephole(
        cpu::rnn_utils::rnn_conf_t &rnn) {
    static constexpr dim_t n_gates = 3;
    rnn.dhc_block_peephole = divide_block_to_improve_thread_balance(
            n_gates, rnn.dhc, rnn.nthr);
    rnn.dhc_blocks_peephole = utils::div_up(rnn.dhc, rnn.dhc_block_peephole);
    rnn.dhc_tail_peephole = rnn.dhc % rnn.dhc_block_peephole;
}

status_t rnn_brgemm_t<prop_kind::backward>::init_kernels(
        const cpu::rnn_utils::rnn_conf_t &rnn, data_type_t src_type,
        data_type_t weights_type) {
    MAYBE_UNUSED(rnn);
    MAYBE_UNUSED(src_type);
    MAYBE_UNUSED(weights_type);

    return status::unimplemented;
}

status_t rnn_brgemm_t<prop_kind::backward>::init_peephole_kernels(
        const cpu::rnn_utils::rnn_conf_t &rnn) {

    MAYBE_UNUSED(rnn);

    return status::unimplemented;
}

} // namespace rnn_brgemm_utils
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
