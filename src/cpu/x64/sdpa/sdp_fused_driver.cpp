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

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/sdpa/sdp_fused_driver.hpp"
#include "cpu/x64/sdpa/sdp_fused_softmax_ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
inline size_t align64(size_t n) {
    return utils::rnd_up(n, size_t(64));
}
} // namespace

sdp_fused_driver_t::sdp_fused_driver_t() = default;

sdp_fused_driver_t::~sdp_fused_driver_t() {
    for (auto *k :
            {mm1_kernel_, mm2_kernel_, mm1_tail_kernel_, mm2_tail_kernel_}) {
        if (k) brgemm_kernel_destroy(k);
    }
}

status_t sdp_fused_driver_t::init(
        const sdp_fused_params_t &params, engine_t *engine) {
    CHECK(configure(params));
    return create_kernels(engine);
}

status_t sdp_fused_driver_t::configure(const sdp_fused_params_t &params) {
    p_ = params;

    const dim_t seq_q = p_.seq_q;
    const dim_t seq_kv = p_.seq_kv;
    const dim_t hs_v = p_.head_size_v;

    // KV tiling width for the streaming softmax: K/V are processed in chunks of
    // up to this many columns, bounding the per-thread scores tile
    // ([seq_q, kv_blk]).
    // TODO: this is a fixed heuristic; it should be derived from the cache
    // size, seq_q and head size so the scores/pv tiles stay cache-resident.
    constexpr dim_t kv_block_width = 512;
    kv_blk_ = nstl::min<dim_t>(seq_kv, kv_block_width);

    // Per-thread scratch: scores, acc, pv, row_max, row_denom, old_coef, each
    // 64-byte aligned so no buffer straddles a cacheline shared with the next.
    // Pure arithmetic (no kernel dependency), so a primitive_desc can size its
    // scratchpad from configure() alone.
    const size_t fsz = sizeof(float);
    const size_t scores_bytes
            = align64(static_cast<size_t>(seq_q) * kv_blk_ * fsz);
    const size_t acc_bytes = align64(static_cast<size_t>(seq_q) * hs_v * fsz);
    const size_t pv_bytes = align64(static_cast<size_t>(seq_q) * hs_v * fsz);
    const size_t row_bytes = align64(static_cast<size_t>(seq_q) * fsz);
    off_scores_ = 0;
    off_acc_ = off_scores_ + scores_bytes;
    off_pv_ = off_acc_ + acc_bytes;
    off_row_max_ = off_pv_ + pv_bytes;
    off_row_denom_ = off_row_max_ + row_bytes;
    off_old_coef_ = off_row_denom_ + row_bytes;
    scratch_per_thread_ = off_old_coef_ + row_bytes;

    nthr_ = dnnl_get_max_threads();

    return status::success;
}

status_t sdp_fused_driver_t::create_kernels(engine_t *engine) {
    UNUSED(engine);

    const int ndims = p_.ndims;
    const dim_t seq_q = p_.seq_q;
    const dim_t seq_kv = p_.seq_kv;
    const dim_t hs_qk = p_.head_size_qk;
    const dim_t hs_v = p_.head_size_v;
    const dim_t row_dim = ndims - 2;
    const dim_t kv_tail = seq_kv % kv_blk_;

    // Create the BRGEMM kernels. Shapes/leading dims are identical for every
    // slice, so one kernel per (full/tail) tile width suffices.
    //   mm1 (beta=0): scores_tile[seq_q, w] = Q[seq_q, hs_qk] * K[hs_qk, w]
    //   mm2 (beta=0): pv_tile[seq_q, hs_v]  = P_tile[seq_q, w] * V[w, hs_v]
    // where w is the KV tile width: kv_blk_ for full tiles, and the
    // seq_kv % kv_blk_ remainder for the last tile.
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

    // mm1 writes a dense [seq_q, w] tile (ldc = w); mm2 multiplies that dense
    // tile by V into a dense [seq_q, hs_v] per-tile buffer (beta=0). The
    // running normalized output is combined in the epilogue, so magnitudes
    // stay O(|V|) (matches the blocked driver's normalize-before accuracy).
    auto create_tile_kernels
            = [&](brgemm_kernel_t **mm1, brgemm_kernel_t **mm2, dim_t w) {
        CHECK(create_brgemm(mm1, /*beta=*/0.0f, seq_q, w, hs_qk,
                /*lda=*/p_.q_strides[row_dim],
                /*ldb=*/p_.k_strides[row_dim],
                /*ldc=*/w));
        CHECK(create_brgemm(mm2, /*beta=*/0.0f, seq_q, hs_v, w,
                /*lda=*/w, /*ldb=*/p_.v_strides[row_dim], /*ldc=*/hs_v));
        return status::success;
    };

    CHECK(create_tile_kernels(&mm1_kernel_, &mm2_kernel_, kv_blk_));
    if (kv_tail != 0)
        CHECK(create_tile_kernels(
                &mm1_tail_kernel_, &mm2_tail_kernel_, kv_tail));

    // Build the JIT online-softmax epilogue (AVX2 IR). One softmax kernel per
    // tile width (full/tail), plus one acc-renormalization kernel. If AVX2 is
    // unavailable the execute path falls back to the scalar epilogue.
    if (mayiuse(avx2)) {
        using namespace sdp_softmax_ir;
        // Condition tensor row stride in elements; columns are contiguous. A
        // seq_q axis of extent 1 is a broadcast axis (meaningless stride), so
        // every query row reads the same condition row -> stride 0.
        const int cond_stride = p_.has_select && p_.cond_dims[row_dim] != 1
                ? static_cast<int>(p_.cond_strides[row_dim])
                : 0;
        const int sq = static_cast<int>(seq_q);
        auto build_ir_kernel = [](std::unique_ptr<softmax_ir_kernel_t> &slot,
                                       ir_t ir) -> status_t {
            std::unique_ptr<softmax_ir_kernel_t> k(
                    new softmax_ir_kernel_t(std::move(ir)));
            CHECK(k->create_kernel());
            slot = std::move(k);
            return status::success;
        };
        status_t st = build_ir_kernel(softmax_ir_kernel_,
                build_softmax_tile_ir(sq, static_cast<int>(kv_blk_),
                        p_.has_select, p_.select_fusiable, cond_stride));
        if (st == status::success && kv_tail != 0)
            st = build_ir_kernel(softmax_tail_ir_kernel_,
                    build_softmax_tile_ir(sq, static_cast<int>(kv_tail),
                            p_.has_select, p_.select_fusiable, cond_stride));
        if (st == status::success)
            st = build_ir_kernel(acc_renorm_ir_kernel_,
                    build_acc_renorm_ir(sq, static_cast<int>(hs_v)));
        use_ir_epilogue_ = st == status::success;
    }

    return status::success;
}

status_t sdp_fused_driver_t::execute(
        const sdp_fused_run_args_t &args, void *scratch_base, int nthr) const {
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

    // Query-side offset (Q / out / select-cond carry the group axis).
    const auto q_side_off = [&](const std::vector<dim_t> &s, dim_t bo, dim_t bi,
                                    dim_t kvh, dim_t gid) -> dim_t {
        return ndims == 4 ? bo * s[0] + bi * s[1]
                          : bo * s[0] + kvh * s[1] + gid * s[2];
    };
    // KV-side offset (K / V; the group axis has extent 1).
    const auto kv_side_off
            = [&](const std::vector<dim_t> &s, dim_t bo, dim_t kvh) -> dim_t {
        return bo * s[0] + kvh * s[1];
    };

    const size_t block_size = scratch_per_thread_;
    auto *scratch = static_cast<char *>(scratch_base);

    parallel_nd_ext(nthr, p_.batch, p_.num_head_q,
            [&](int tid, int, dim_t bo, dim_t bi) {
        const dim_t kvh = bi / group;
        const dim_t gid = bi % group;

        const float *q_ptr = reinterpret_cast<const float *>(q_base
                + q_side_off(p_.q_strides, bo, bi, kvh, gid) * sizeof(float));
        const float *k_ptr = reinterpret_cast<const float *>(
                k_base + kv_side_off(p_.k_strides, bo, kvh) * sizeof(float));
        const float *v_ptr = reinterpret_cast<const float *>(
                v_base + kv_side_off(p_.v_strides, bo, kvh) * sizeof(float));
        float *o_ptr = reinterpret_cast<float *>(o_base
                + q_side_off(p_.o_strides, bo, bi, kvh, gid) * sizeof(float));
        const uint8_t *c_ptr = p_.has_select
                ? reinterpret_cast<const uint8_t *>(cond_base
                          + q_side_off(eff_cond_strides, bo, bi, kvh, gid)
                                  * sizeof(uint8_t))
                : nullptr;

        // Online-softmax running state kept in a numerically stable form: the
        // accumulator (acc) holds the *normalized* output so far, so its
        // magnitude stays O(|V|). Per tile, mm2 produces the raw P_tile*V_tile
        // into pv, then acc is renormalized. row_max (m) and row_denom (l) are
        // the running max and denominator. Buffers are per-thread slices of
        // the scratchpad (see init()).
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
        std::fill(row_max, row_max + seq_q, neg_inf);
        std::fill(row_denom, row_denom + seq_q, 0.0f);
        std::fill(acc, acc + static_cast<size_t>(seq_q) * hs_v, 0.0f);

        for (dim_t kv0 = 0; kv0 < seq_kv; kv0 += kv_blk) {
            const dim_t w = nstl::min(kv_blk, seq_kv - kv0);
            const bool is_tail = w != kv_blk;
            const auto *mm1 = is_tail ? mm1_tail_kernel_ : mm1_kernel_;
            const auto *mm2 = is_tail ? mm2_tail_kernel_ : mm2_kernel_;

            // mm1: scores_tile[seq_q, w] = Q * K[:, kv0 : kv0 + w].
            brgemm_batch_element_t batch1;
            batch1.ptr.A = q_ptr;
            batch1.ptr.B = k_ptr + kv0 * k_col;
            brgemm_kernel_execute(mm1, 1, &batch1, scores, nullptr);

            // Online-softmax epilogue over this KV tile: apply scale + mask,
            // update the running max/denom, and form P_tile = exp(s - m_new).
            if (use_ir_epilogue_) {
                const auto &sm = is_tail ? softmax_tail_ir_kernel_
                                         : softmax_ir_kernel_;
                sdp_softmax_ir::softmax_row_args_t sargs;
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
                for (dim_t i = 0; i < seq_q; ++i) {
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
                    // accumulates O(1) magnitudes (matches the blocked driver's
                    // accuracy). acc then holds U/l; refresh it with old_coef =
                    // corr*l_old/l_new.
                    for (dim_t j = 0; j < w; ++j)
                        srow[j] *= inv;
                    old_coef[i] = corr * l_old * inv;
                }
            }

            // mm2: pv[seq_q, hs_v] = P_norm_tile * V[kv0 : kv0 + w, :].
            brgemm_batch_element_t batch2;
            batch2.ptr.A = scores;
            batch2.ptr.B = v_ptr + kv0 * v_row;
            brgemm_kernel_execute(mm2, 1, &batch2, pv, nullptr);

            // Renormalize the running output: acc = old_coef*acc + pv.
            if (use_ir_epilogue_) {
                sdp_softmax_ir::acc_renorm_args_t aargs;
                aargs.acc = acc;
                aargs.pv = pv;
                aargs.old_coef = old_coef;
                (*acc_renorm_ir_kernel_)(&aargs);
            } else {
                for (dim_t i = 0; i < seq_q; ++i) {
                    float *arow = acc + i * hs_v;
                    const float *prow = pv + i * hs_v;
                    const float a = old_coef[i];
                    for (dim_t d = 0; d < hs_v; ++d)
                        arow[d] = a * arow[d] + prow[d];
                }
            }
        }

        // acc already holds the normalized output; scatter to user output.
        for (dim_t i = 0; i < seq_q; ++i) {
            const float *arow = acc + i * hs_v;
            float *out_row = o_ptr + i * o_row;
            for (dim_t d = 0; d < hs_v; ++d)
                out_row[d * o_col] = arow[d];
        }
    });

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
