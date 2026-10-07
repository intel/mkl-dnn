/*******************************************************************************
* Copyright 2026 Intel Corporation
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
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/sdpa/sdpa_online_softmax.hpp"
#include "cpu/x64/sdpa/sdpa_online_softmax_ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
inline size_t align64(size_t n) {
    return utils::rnd_up(n, size_t(64));
}
} // namespace

sdpa_online_softmax_kernels_t::sdpa_online_softmax_kernels_t() = default;

sdpa_online_softmax_kernels_t::~sdpa_online_softmax_kernels_t() {
    for (int qi = 0; qi < 2; ++qi)
        for (int ki = 0; ki < 2; ++ki) {
            if (mm1_kernel[qi][ki]) brgemm_kernel_destroy(mm1_kernel[qi][ki]);
            if (mm2_kernel[qi][ki]) brgemm_kernel_destroy(mm2_kernel[qi][ki]);
        }
}

namespace sdpa_online_softmax {

status_t configure(const sdpa_online_softmax_params_t &params,
        sdpa_online_softmax_conf_t &conf) {
    conf.params = params;

    const dim_t seq_q = params.seq_q;
    const dim_t seq_kv = params.seq_kv;
    const dim_t hs_qk = params.head_size_qk;
    const dim_t hs_v = params.head_size_v;

    const dim_t l2_budget_bytes
            = static_cast<dim_t>(3 * platform::get_per_core_cache_size(2) / 4);

    // KV tile width. Each KV tile runs mm1 -> softmax -> mm2 over a kv_blk-wide
    // column slice. The two matmul B operands -- mm1's K panel [hs_qk x kv_blk]
    // and mm2's V panel [kv_blk x hs_v] -- are each re-read once per q_blk
    // M-block of their brgemm, so they must stay L2-resident across those
    // re-reads or the inner loop goes memory-bound. Their footprint is
    // (hs_qk|hs_v) * kv_blk, i.e. it grows with the head size; to hold the
    // footprint (and thus the L2 pressure) constant across head sizes, kv_blk
    // must scale as 1/head_size. Pin the wider panel to L2/8 (~192 KiB here) so
    // the two panels together take ~L2/4 and leave ~3/4 L2 for the q_blk-scaled
    // state sized below. This 1/head-size law is empirically the per-head
    // optimum (hs 512 -> kv 64, hs 128 -> kv 384, hs 64 -> kv 768); at the small
    // end the panel also drops into L1. Round DOWN to a 64-column granule (4
    // AVX512 f32 vectors / cachelines; this path is always f32, never AMX); if
    // one block already covers the axis (short context) run it untiled.
    const dim_t b_panel_budget_bytes = l2_budget_bytes / 8;
    const dim_t b_panel_rows = nstl::max(hs_qk, hs_v);
    dim_t kv_block_width = b_panel_rows > 0
            ? static_cast<dim_t>(b_panel_budget_bytes
                      / (b_panel_rows * static_cast<dim_t>(sizeof(float))))
            : seq_kv;
    kv_block_width = utils::rnd_dn(kv_block_width, (dim_t)64);
    if (kv_block_width < 64) kv_block_width = seq_kv;
    conf.kv_blk = nstl::min<dim_t>(seq_kv, kv_block_width);

    // Query block. Per KV tile mm1 re-reads the Q slice [q_blk x hs_qk] and the
    // acc-renorm re-reads+writes the running acc [q_blk x hs_v]; these carry
    // across the whole KV sweep, so holding them cache-resident is what turns
    // mm1 from memory-bound (re-reading Q every tile) into compute-bound, and a
    // larger q_blk also amortizes the once-per-block K/V stream. Size q_blk from
    // the per-row state -- Q [hs_qk], acc + pv [hs_v] each, one live scores row
    // [kv_blk] -- against the full 3/4-L2 budget. The B panels (another ~L2/4,
    // reserved above via kv_blk) are deliberately NOT subtracted here: acc/pv/Q
    // are swept sequentially and tolerate spilling to L3, whereas the panels are
    // re-read and must stay put -- so the panels get first claim on L2 and q_blk
    // is allowed to overcommit the remainder. Clamp to [1, seq_q].
    const dim_t per_row_bytes = static_cast<dim_t>(
            (hs_qk + 2 * hs_v + conf.kv_blk) * sizeof(float));
    conf.q_blk = utils::saturate((dim_t)1, seq_q,
            static_cast<dim_t>(l2_budget_bytes / per_row_bytes));
    conf.q_tail = seq_q % conf.q_blk;

    // Per-thread scratch: scores, acc, pv, row_max, row_denom, old_coef, each
    // 64-byte aligned so no buffer straddles a cacheline shared with the next.
    // Sized for one query block (q_blk rows), not the full seq_q. Pure
    // arithmetic (no kernel dependency), so a primitive_desc can size its
    // scratchpad from configure() alone.
    const size_t fsz = sizeof(float);
    const size_t scores_bytes
            = align64(static_cast<size_t>(conf.q_blk) * conf.kv_blk * fsz);
    const size_t acc_bytes
            = align64(static_cast<size_t>(conf.q_blk) * hs_v * fsz);
    const size_t pv_bytes
            = align64(static_cast<size_t>(conf.q_blk) * hs_v * fsz);
    const size_t row_bytes = align64(static_cast<size_t>(conf.q_blk) * fsz);
    conf.off_scores = 0;
    conf.off_acc = conf.off_scores + scores_bytes;
    conf.off_pv = conf.off_acc + acc_bytes;
    conf.off_row_max = conf.off_pv + pv_bytes;
    conf.off_row_denom = conf.off_row_max + row_bytes;
    conf.off_old_coef = conf.off_row_denom + row_bytes;
    conf.scratch_per_thread = conf.off_old_coef + row_bytes;

    conf.nthr = dnnl_get_max_threads();

    return status::success;
}

status_t create_kernels(const sdpa_online_softmax_conf_t &conf,
        engine_t *engine, sdpa_online_softmax_kernels_t &kernels) {
    UNUSED(engine);
    const sdpa_online_softmax_params_t &p_ = conf.params;
    const dim_t kv_blk_ = conf.kv_blk;

    const int ndims = p_.ndims;
    const dim_t seq_kv = p_.seq_kv;
    const dim_t hs_qk = p_.head_size_qk;
    const dim_t hs_v = p_.head_size_v;
    const dim_t row_dim = ndims - 2;
    const dim_t kv_tail = seq_kv % kv_blk_;
    const dim_t q_blk_ = conf.q_blk;
    const dim_t q_tail = conf.q_tail;
    // Query-block row counts indexed by the q-tail flag: [0] full block, [1]
    // ragged last block (only built when q_tail != 0).
    const dim_t q_rows[2] = {q_blk_, q_tail};
    // KV tile widths indexed by the kv-tail flag.
    const dim_t kv_w[2] = {kv_blk_, kv_tail};

    // Create the BRGEMM kernels. Shapes differ only in the M (query-block rows)
    // and N/K (KV tile width) dims, so one kernel per (q full/tail, kv
    // full/tail) combination suffices.
    //   mm1 (beta=0): scores_tile[m, w] = Q[m, hs_qk] * K[hs_qk, w]
    //   mm2 (beta=0): pv_tile[m, hs_v]  = P_tile[m, w] * V[w, hs_v]
    // where m is the query-block rows (q_blk_ or the seq_q % q_blk_ remainder)
    // and w is the KV tile width (kv_blk_ or the seq_kv % kv_blk_ remainder).
    auto create_brgemm
            = [&](brgemm_kernel_t **out, float beta, dim_t M, dim_t N, dim_t K,
                      dim_t lda, dim_t ldb, dim_t ldc) -> status_t {
        brgemm_desc_t brg;
        CHECK(brgemm_desc_init(&brg, isa_undef, brgemm_addr,
                dnnl::impl::data_type::f32, dnnl::impl::data_type::f32,
                /*transA=*/false, /*transB=*/false, brgemm_row_major,
                /*alpha=*/1.0f, beta, lda, ldb, ldc, M, N, K,
                /*strides=*/nullptr));
        CHECK(brgemm_desc_finalize(&brg));
        brgemm_kernel_t *k = nullptr;
        CHECK(brgemm_kernel_create(&k, brg));
        *out = k;
        return status::success;
    };

    // mm1 writes a dense [m, w] tile (ldc = w); mm2 multiplies that dense tile
    // by V into a dense [m, hs_v] per-tile buffer (beta=0). The running
    // normalized output is combined in the epilogue, so magnitudes stay O(|V|)
    // (matches the full-softmax path's normalize-before accuracy).
    auto create_tile_kernels = [&](brgemm_kernel_t **mm1, brgemm_kernel_t **mm2,
                                       dim_t m, dim_t w) {
        CHECK(create_brgemm(mm1, /*beta=*/0.0f, m, w, hs_qk,
                /*lda=*/p_.q_strides[row_dim],
                /*ldb=*/p_.k_strides[row_dim],
                /*ldc=*/w));
        CHECK(create_brgemm(mm2, /*beta=*/0.0f, m, hs_v, w,
                /*lda=*/w, /*ldb=*/p_.v_strides[row_dim], /*ldc=*/hs_v));
        return status::success;
    };

    const int n_qi = q_tail != 0 ? 2 : 1;
    const int n_ki = kv_tail != 0 ? 2 : 1;
    for (int qi = 0; qi < n_qi; ++qi)
        for (int ki = 0; ki < n_ki; ++ki)
            CHECK(create_tile_kernels(&kernels.mm1_kernel[qi][ki],
                    &kernels.mm2_kernel[qi][ki], q_rows[qi], kv_w[ki]));

    // Build the JIT online-softmax epilogue (AVX2 IR). One softmax kernel per
    // (q full/tail, kv full/tail) combination, plus one acc-renormalization
    // kernel per query-block row count. If AVX2 is unavailable the execute path
    // falls back to the scalar epilogue.
    if (mayiuse(avx2)) {
        using namespace sdpa_softmax_ir;
        // Condition tensor row stride in elements; columns are contiguous. A
        // seq_q axis of extent 1 is a broadcast axis (meaningless stride), so
        // every query row reads the same condition row -> stride 0.
        const int cond_stride = p_.has_select && p_.cond_dims[row_dim] != 1
                ? static_cast<int>(p_.cond_strides[row_dim])
                : 0;
        auto build_ir_kernel = [](std::unique_ptr<softmax_ir_kernel_t> &slot,
                                       ir_t ir) -> status_t {
            std::unique_ptr<softmax_ir_kernel_t> k(
                    new softmax_ir_kernel_t(std::move(ir)));
            CHECK(k->create_kernel());
            slot = std::move(k);
            return status::success;
        };
        status_t st = status::success;
        for (int qi = 0; qi < n_qi && st == status::success; ++qi) {
            const int m = static_cast<int>(q_rows[qi]);
            for (int ki = 0; ki < n_ki && st == status::success; ++ki)
                st = build_ir_kernel(kernels.softmax_ir_kernel[qi][ki],
                        build_softmax_tile_ir(m, static_cast<int>(kv_w[ki]),
                                p_.has_select, p_.select_fusiable,
                                cond_stride));
            if (st == status::success)
                st = build_ir_kernel(kernels.acc_renorm_ir_kernel[qi],
                        build_acc_renorm_ir(m, static_cast<int>(hs_v)));
        }
        kernels.use_ir_epilogue = st == status::success;
    }

    return status::success;
}

status_t execute(const sdpa_online_softmax_conf_t &conf,
        const sdpa_online_softmax_kernels_t &kernels,
        const sdpa_online_softmax_run_args_t &args, void *scratch_base,
        int nthr) {
    // Aliases so the loop below reads the pd-owned conf and primitive-owned
    // kernels directly, without copying any state into this call.
    const sdpa_online_softmax_params_t &p_ = conf.params;
    const dim_t kv_blk_ = conf.kv_blk;
    const dim_t q_blk_ = conf.q_blk;
    const size_t off_scores_ = conf.off_scores, off_acc_ = conf.off_acc,
                 off_pv_ = conf.off_pv, off_row_max_ = conf.off_row_max,
                 off_row_denom_ = conf.off_row_denom,
                 off_old_coef_ = conf.off_old_coef;
    const size_t scratch_per_thread_ = conf.scratch_per_thread;
    const bool use_ir_epilogue_ = kernels.use_ir_epilogue;

    auto *q_base = static_cast<const char *>(args.q);
    auto *k_base = static_cast<const char *>(args.k);
    auto *v_base = static_cast<const char *>(args.v);
    auto *o_base = static_cast<char *>(args.out);
    const char *cond_base = static_cast<const char *>(args.cond);
    const float scale_val = args.scale;
    const float fill_val = args.fill;

    const dim_t seq_q = p_.seq_q, seq_kv = p_.seq_kv, hs_v = p_.head_size_v;
    const dim_t kv_blk = kv_blk_;
    const dim_t group = p_.group_head;
    const int ndims = p_.ndims;
    const int row_dim = ndims - 2;
    // Element strides for addressing a KV tile within K / V.
    const dim_t k_col = p_.k_strides[ndims - 1]; // K[.., hs, seq_kv]: seq step
    const dim_t v_row = p_.v_strides[row_dim]; // V[.., seq_kv, hs_v]: kv step
    const dim_t o_row = p_.o_strides[row_dim];
    const dim_t o_col = p_.o_strides[ndims - 1];
    // Broadcast-aware select-condition strides: an axis with extent 1 is a
    // broadcast axis whose stride is meaningless and must contribute 0.
    std::vector<dim_t> eff_cond_strides;
    if (p_.has_select) {
        eff_cond_strides = p_.cond_strides;
        for (int d = 0; d < ndims; ++d)
            if (p_.cond_dims[d] == 1) eff_cond_strides[d] = 0;
    }
    const dim_t cond_row = p_.has_select ? eff_cond_strides[row_dim] : 0;
    const dim_t cond_col = p_.has_select ? eff_cond_strides[ndims - 1] : 0;
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    // Q, output, and select-condition tensors use query-head/group coordinates;
    // K and V use batch/KV-head coordinates shared by each GQA group.
    const auto query_tensor_base_offset
            = [&](const std::vector<dim_t> &strides, dim_t batch_idx,
                      dim_t query_head_idx, dim_t kv_head_idx,
                      dim_t group_idx) -> dim_t {
        return ndims == 4 ? batch_idx * strides[0] + query_head_idx * strides[1]
                          : batch_idx * strides[0] + kv_head_idx * strides[1]
                        + group_idx * strides[2];
    };
    // K and V are shared by the query heads in each GQA group.
    const auto kv_tensor_base_offset
            = [&](const std::vector<dim_t> &strides, dim_t batch_idx,
                      dim_t kv_head_idx) -> dim_t {
        return batch_idx * strides[0] + kv_head_idx * strides[1];
    };

    const size_t block_size = scratch_per_thread_;
    auto *scratch = static_cast<char *>(scratch_base);

    // Query-axis stride and block count: each work item owns one query block of
    // up to q_blk_ rows, so the per-thread Q slice, scores tile and running
    // accumulator stay L2-resident across the KV sweep.
    const dim_t q_row = p_.q_strides[row_dim];
    const dim_t n_qblk = utils::div_up(seq_q, q_blk_);

    parallel_nd_ext(nthr, p_.batch, p_.num_head_q, n_qblk,
            [&](int tid, int, dim_t bo, dim_t bi, dim_t qblk) {
        const dim_t kvh = bi / group;
        const dim_t gid = bi % group;
        const dim_t q0 = qblk * q_blk_;
        const dim_t m = nstl::min(q_blk_, seq_q - q0);
        const int qi = m != q_blk_ ? 1 : 0;

        const float *q_ptr = reinterpret_cast<const float *>(q_base
                + (query_tensor_base_offset(p_.q_strides, bo, bi, kvh, gid)
                          + q0 * q_row)
                        * sizeof(float));
        const float *k_ptr = reinterpret_cast<const float *>(k_base
                + kv_tensor_base_offset(p_.k_strides, bo, kvh) * sizeof(float));
        const float *v_ptr = reinterpret_cast<const float *>(v_base
                + kv_tensor_base_offset(p_.v_strides, bo, kvh) * sizeof(float));
        float *o_ptr = reinterpret_cast<float *>(o_base
                + (query_tensor_base_offset(p_.o_strides, bo, bi, kvh, gid)
                          + q0 * o_row)
                        * sizeof(float));
        const uint8_t *c_ptr = p_.has_select
                ? reinterpret_cast<const uint8_t *>(cond_base
                          + (query_tensor_base_offset(
                                     eff_cond_strides, bo, bi, kvh, gid)
                                    + q0 * cond_row)
                                  * sizeof(uint8_t))
                : nullptr;

        // Online-softmax running state kept in a numerically stable form: the
        // accumulator (acc) holds the *normalized* output so far, so its
        // magnitude stays O(|V|). Per tile, mm2 produces the raw P_tile*V_tile
        // into pv, then acc is renormalized. row_max (m) and row_denom (l) are
        // the running max and denominator. Buffers are per-thread slices of
        // the scratchpad (see init()), sized for one query block (q_blk_ rows).
        char *tblock = scratch + static_cast<size_t>(tid) * block_size;
        float *scores = reinterpret_cast<float *>(tblock + off_scores_);
        float *acc = reinterpret_cast<float *>(tblock + off_acc_);
        float *pv = reinterpret_cast<float *>(tblock + off_pv_);
        float *row_max = reinterpret_cast<float *>(tblock + off_row_max_);
        float *row_denom = reinterpret_cast<float *>(tblock + off_row_denom_);
        // Per-row renormalization coefficient for the current tile.
        float *old_coef = reinterpret_cast<float *>(tblock + off_old_coef_);
        // acc/row_max/row_denom carry running state across tiles, so they must
        // be initialized (scratchpad memory is uninitialized).
        std::fill(row_max, row_max + m, neg_inf);
        std::fill(row_denom, row_denom + m, 0.0f);
        std::fill(acc, acc + static_cast<size_t>(m) * hs_v, 0.0f);

        for (dim_t kv0 = 0; kv0 < seq_kv; kv0 += kv_blk) {
            const dim_t w = nstl::min(kv_blk, seq_kv - kv0);
            const int ki = w != kv_blk ? 1 : 0;
            const auto *mm1 = kernels.mm1_kernel[qi][ki];
            const auto *mm2 = kernels.mm2_kernel[qi][ki];

            // mm1: scores_tile[m, w] = Q * K[:, kv0 : kv0 + w].
            brgemm_batch_element_t batch1;
            batch1.ptr.A = q_ptr;
            batch1.ptr.B = k_ptr + kv0 * k_col;
            brgemm_kernel_execute(mm1, 1, &batch1, scores, nullptr);

            // Online-softmax epilogue over this KV tile: apply scale + mask,
            // update the running max/denom, and form P_tile = exp(s - m_new).
            if (use_ir_epilogue_) {
                const auto &sm = kernels.softmax_ir_kernel[qi][ki];
                sdpa_softmax_ir::softmax_row_args_t sargs;
                sargs.scores = scores;
                sargs.scale = &scale_val;
                sargs.m = row_max;
                sargs.l = row_denom;
                sargs.old_coef = old_coef;
                // cond points at this tile's first column (row 0); the kernel
                // advances by the compiled cond row stride per row.
                sargs.cond = c_ptr ? c_ptr + kv0 * cond_col : nullptr;
                sargs.fill = &fill_val;
                (*sm)(&sargs);
            } else {
                for (dim_t i = 0; i < m; ++i) {
                    float *srow = scores + i * w;
                    const uint8_t *crow
                            = c_ptr ? c_ptr + i * cond_row : nullptr;
                    float tile_max = neg_inf;
                    for (dim_t j = 0; j < w; ++j) {
                        float v = srow[j] * scale_val;
                        if (crow) {
                            const bool cond = crow[(kv0 + j) * cond_col] != 0;
                            // not-fusiable (p1): cond ? fill : scores
                            // fusiable    (p2): cond ? scores : fill
                            const bool keep = p_.select_fusiable ? cond : !cond;
                            if (!keep) v = fill_val;
                        }
                        srow[j] = v;
                        if (v > tile_max) tile_max = v;
                    }
                    const float m_old = row_max[i];
                    const float l_old = row_denom[i];
                    const float m_new = nstl::max(m_old, tile_max);
                    // corr rescales the old contributions to the new max; it is
                    // 0 for the first (m_old == -inf) tile.
                    const float corr
                            = m_old == neg_inf ? 0.0f : expf(m_old - m_new);
                    float tile_sum = 0.0f;
                    for (dim_t j = 0; j < w; ++j) {
                        const float e = expf(srow[j] - m_new);
                        srow[j] = e;
                        tile_sum += e;
                    }
                    const float l_new = l_old * corr + tile_sum;
                    const float inv = l_new > 0.0f ? 1.0f / l_new : 0.0f;
                    row_denom[i] = l_new;
                    row_max[i] = m_new;
                    // Pre-normalize P by the running denominator so mm2
                    // accumulates O(1) magnitudes (matches the full-softmax path's
                    // accuracy). acc then holds U/l; refresh it with old_coef =
                    // corr*l_old/l_new.
                    for (dim_t j = 0; j < w; ++j)
                        srow[j] *= inv;
                    old_coef[i] = corr * l_old * inv;
                }
            }

            // mm2: pv[m, hs_v] = P_norm_tile * V[kv0 : kv0 + w, :].
            brgemm_batch_element_t batch2;
            batch2.ptr.A = scores;
            batch2.ptr.B = v_ptr + kv0 * v_row;
            brgemm_kernel_execute(mm2, 1, &batch2, pv, nullptr);

            // Renormalize the running output: acc = old_coef*acc + pv.
            if (use_ir_epilogue_) {
                sdpa_softmax_ir::acc_renorm_args_t aargs;
                aargs.acc = acc;
                aargs.pv = pv;
                aargs.old_coef = old_coef;
                (*kernels.acc_renorm_ir_kernel[qi])(&aargs);
            } else {
                for (dim_t i = 0; i < m; ++i) {
                    float *arow = acc + i * hs_v;
                    const float *prow = pv + i * hs_v;
                    const float a = old_coef[i];
                    for (dim_t d = 0; d < hs_v; ++d)
                        arow[d] = a * arow[d] + prow[d];
                }
            }
        }

        // acc already holds the normalized output; scatter to user output.
        for (dim_t i = 0; i < m; ++i) {
            const float *arow = acc + i * hs_v;
            float *out_row = o_ptr + i * o_row;
            for (dim_t d = 0; d < hs_v; ++d)
                out_row[d * o_col] = arow[d];
        }
    });

    return status::success;
}

} // namespace sdpa_online_softmax

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
