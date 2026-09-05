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
#include <limits>

#include "common/dnnl_thread.hpp"
#include "common/memory_desc.hpp"
#include "common/opdesc.hpp"
#include "common/primitive_attr.hpp"
#include "common/primitive_desc.hpp"
#include "common/primitive_desc_iterator.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "graph/backend/dnnl/kernels/sdp_blocked_driver.hpp"

#if DNNL_X64
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_uni_softmax.hpp"
#endif

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {

#if DNNL_X64

using namespace dnnl::impl::cpu::x64;

namespace {
// Round a byte count up to a 64-byte boundary so each thread's scratch block
// starts cache-line aligned when carved from a cache-line-aligned base.
inline size_t align64(size_t n) {
    return (n + 63) & ~static_cast<size_t>(63);
}

status_t create_brgemm(brgemm_kernel_t **out, float beta, dim_t M, dim_t N,
        dim_t K, dim_t lda, dim_t ldb, dim_t ldc,
        const std::vector<sdp_mm1_post_op_t> *post_ops = nullptr,
        bool select_postop = false, bool transB = false) {
    brgemm_desc_t brg;
    CHECK(brgemm_desc_init(&brg, isa_undef, brgemm_addr,
            dnnl::impl::data_type::f32, dnnl::impl::data_type::f32,
            /*transA=*/false, transB, brgemm_row_major,
            /*alpha=*/1.0f, beta, lda, ldb, ldc, M, N, K, /*strides=*/nullptr));
    // Fold the mm1 post-op chain (scale / soft-cap / attention-mask) and the
    // select-mask into the GEMM store as binary/eltwise post-ops, mirroring the
    // decomp path. A scalar binary rhs is a [1 x 1] broadcast; a tensor rhs is
    // its per-tile [rows x cols] slice (a broadcast axis stays 1); select is
    // binary_select with a scalar fill rhs and a dense [M x N] condition rhs.
    // Runtime pointers are supplied per execute call.
    const bool has_chain = post_ops && !post_ops->empty();
    if (has_chain || select_postop) {
        primitive_attr_t attr;
        post_ops_t po;
        if (has_chain) {
            for (const auto &pop : *post_ops) {
                if (!pop.is_binary) {
                    CHECK(po.append_eltwise(
                            /*scale=*/1.0f, pop.alg, pop.alpha, pop.beta));
                    continue;
                }
                memory_desc_t rhs_md;
                if (pop.rhs_is_scalar) {
                    dims_t rhs_dims = {1, 1};
                    CHECK(memory_desc_init_by_tag(
                            rhs_md, 2, rhs_dims, pop.rhs_dt, format_tag::ab));
                } else {
                    // Per-tile slice of the user rhs: keep broadcast axes at 1
                    // and carry the real row/column strides.
                    const int rn = static_cast<int>(pop.rhs_dims.size());
                    const dim_t rows = pop.rhs_dims[rn - 2] == 1 ? 1 : M;
                    const dim_t cols = pop.rhs_dims[rn - 1] == 1 ? 1 : N;
                    dims_t rhs_dims = {rows, cols};
                    dims_t rhs_str = {
                            pop.rhs_strides[rn - 2], pop.rhs_strides[rn - 1]};
                    CHECK(memory_desc_init_by_strides(
                            rhs_md, 2, rhs_dims, pop.rhs_dt, rhs_str));
                }
                CHECK(po.append_binary(pop.alg, &rhs_md));
            }
        }
        if (select_postop) {
            memory_desc_t fill_md, cond_md;
            dims_t fl_dims = {1, 1};
            CHECK(memory_desc_init_by_tag(
                    fill_md, 2, fl_dims, data_type::f32, format_tag::ab));
            dims_t cd_dims = {M, N};
            CHECK(memory_desc_init_by_tag(
                    cond_md, 2, cd_dims, data_type::u8, format_tag::ab));
            CHECK(po.append_binary(
                    alg_kind::binary_select, &fill_md, &cond_md));
        }
        CHECK(attr.set_post_ops(po));
        memory_desc_t dst_md;
        dims_t d_dims = {M, N};
        CHECK(memory_desc_init_by_tag(
                dst_md, 2, d_dims, data_type::f32, format_tag::ab));
        CHECK(brgemm_desc_set_postops(&brg, &attr, &dst_md, /*LDD=*/ldc));
    }
    CHECK(brgemm_desc_finalize(&brg));
    brgemm_kernel_t *k = nullptr;
    CHECK(brgemm_kernel_create(&k, brg));
    *out = k;
    return status::success;
}
} // namespace

sdp_blocked_driver_t::~sdp_blocked_driver_t() {
    for (auto *k :
            {mm1_kernel_, mm2_kernel_, mm1_tail_kernel_, mm2_tail_kernel_}) {
        if (k) brgemm_kernel_destroy(k);
    }
}

status_t sdp_blocked_driver_t::init(
        const sdp_blocked_params_t &params, engine_t *engine) {
    p_ = params;

    const dim_t seq_q = p_.seq_q;
    const dim_t seq_kv = p_.seq_kv;
    const dim_t hs_qk = p_.head_size_qk;
    const dim_t hs_v = p_.head_size_v;
    const int row_dim = p_.ndims - 2;

    // Choose the query block so the [q_block x seq_kv] fp32 score tile stays
    // L2-resident. GNR has ~2MB L2/core; budget ~half of it for the scores
    // tile (the rest holds Q/K/V/pv working set and other live data).
    // TODO: query the real cache size from the platform instead of a literal.
    constexpr size_t l2_budget_bytes = 1024 * 1024;
    const size_t row_bytes
            = static_cast<size_t>(seq_kv) * sizeof(float); // one score row
    dim_t q_block = row_bytes > 0
            ? static_cast<dim_t>(l2_budget_bytes / row_bytes)
            : seq_q;
    q_block = nstl::max<dim_t>(q_block, 1);
    q_block = nstl::min<dim_t>(q_block, seq_q);
    q_block_ = q_block;
    q_tail_ = seq_q % q_block;

    // BRGEMM leading dims mirror the online kernel: A/B leading dims come from
    // the user strides (row_dim = the M/K row axis), scores/pv are dense.
    //   mm1: scores[m, seq_kv] = Q[m, hs_qk] * K[hs_qk, seq_kv]
    //   mm2: pv[m, hs_v]       = P[m, seq_kv] * V[seq_kv, hs_v]
    // The brgemm ukernel does not support a transposed B operand and needs mm1
    // B row-major as [hs_qk, seq_kv] (seq_kv unit-stride). Decide the transpose
    // from the K tensor's PHYSICAL strides, not the graph transpose_b attr:
    //   * transpose_b == true  : K is [.., seq_kv, hs_qk] -> seq_kv axis is
    //                            row_dim, hs_qk axis is the last dim.
    //   * transpose_b == false : K is [.., hs_qk, seq_kv] -> hs_qk axis is
    //                            row_dim, seq_kv axis is the last dim.
    // In either orientation, if the seq_kv axis is not unit-stride the driver
    // materialises a dense [hs_qk, seq_kv] transpose once (see execute); a K
    // that is logically pre-transposed but physically stored with hs_qk
    // contiguous therefore still gets transposed here.
    k_seq_stride_ = p_.mm1_transpose_b ? p_.k_strides[row_dim]
                                       : p_.k_strides[p_.ndims - 1];
    k_hs_stride_ = p_.mm1_transpose_b ? p_.k_strides[p_.ndims - 1]
                                      : p_.k_strides[row_dim];
    mm1_transpose_k_ = k_seq_stride_ != 1;
    const dim_t mm1_ldb = mm1_transpose_k_ ? seq_kv : k_hs_stride_;

    auto create_tile_kernels
            = [&](brgemm_kernel_t **mm1, brgemm_kernel_t **mm2, dim_t m,
                      bool select_postop) -> status_t {
        CHECK(create_brgemm(mm1, /*beta=*/0.0f, m, seq_kv, hs_qk,
                /*lda=*/p_.q_strides[row_dim], /*ldb=*/mm1_ldb,
                /*ldc=*/seq_kv, &p_.mm1_post_ops, select_postop));
        CHECK(create_brgemm(mm2, /*beta=*/0.0f, m, hs_v, seq_kv,
                /*lda=*/seq_kv, /*ldb=*/p_.v_strides[row_dim], /*ldc=*/hs_v));
        return status::success;
    };

    // The select-mask can be fused into mm1 as a binary_select post-op only
    // when it is fusiable (keep-where-cond, not the inverted form) and the
    // user condition tile is dense [m x seq_kv] -- the ukernel addresses the
    // condition via the dst tile offsets, so its row stride must equal seq_kv
    // and its column stride must be 1. Otherwise fall back to the pre-pass.
    const dim_t cond_row_stride = p_.has_select ? p_.cond_strides[row_dim] : 0;
    const dim_t cond_col_stride
            = p_.has_select ? p_.cond_strides[p_.ndims - 1] : 0;
    const bool want_select_postop = p_.has_select && p_.select_fusiable
            && cond_row_stride == seq_kv && cond_col_stride == 1;

    mm1_select_postop_ = want_select_postop;

    auto build_kernels = [&](bool select_postop) -> status_t {
        CHECK(create_tile_kernels(
                &mm1_kernel_, &mm2_kernel_, q_block_, select_postop));
        if (q_tail_ != 0)
            CHECK(create_tile_kernels(&mm1_tail_kernel_, &mm2_tail_kernel_,
                    q_tail_, select_postop));
        return status::success;
    };
    auto destroy_kernels = [&]() {
        for (auto **k : {&mm1_kernel_, &mm2_kernel_, &mm1_tail_kernel_,
                     &mm2_tail_kernel_}) {
            if (*k) {
                brgemm_kernel_destroy(*k);
                *k = nullptr;
            }
        }
    };

    if (build_kernels(want_select_postop) != status::success) {
        // A ukernel config may reject the select post-op; drop it and let the
        // pre-pass apply the mask instead.
        destroy_kernels();
        mm1_select_postop_ = false;
        CHECK(build_kernels(false));
    }

    // Per-thread scratch: one score tile [q_block x seq_kv] plus one pv tile
    // [q_block x hs_v]. The pv tile lets mm2 write a dense output that is then
    // scattered to the (possibly strided) user output.
    const size_t scores_bytes
            = static_cast<size_t>(q_block_) * seq_kv * sizeof(float);
    const size_t pv_bytes
            = static_cast<size_t>(q_block_) * hs_v * sizeof(float);
    scratch_per_thread_ = align64(align64(scores_bytes) + pv_bytes);

    nthr_ = dnnl_get_max_threads();

    // transpose_b QK^T: mm1 needs a dense [hs_qk, seq_kv] B tile per head, but
    // the brgemm ukernel has no transposed-B mode. Transpose the whole K tensor
    // ONCE (see execute) into a shared [batch x num_head_kv x hs_qk x seq_kv]
    // buffer that follows the per-thread blocks; kt_global_bytes_ sizes it.
    num_head_kv_ = p_.group_head > 0 ? p_.num_head_q / p_.group_head : 0;
    if (mm1_transpose_k_ && num_head_kv_ > 0)
        kt_global_bytes_ = static_cast<size_t>(p_.batch) * num_head_kv_ * hs_qk
                * seq_kv * sizeof(float);

    // Reuse the vectorized jit softmax kernel for the max/exp/normalize over
    // the seq_kv axis. Build a plain 2D [q_block x seq_kv] f32 softmax pd
    // (axis = 1, the contiguous seq_kv axis) and extract its jit kernel. The
    // kernel is per-row, so the same instance serves the query-tail block too.
    // If no jit impl is selected (e.g. non-AVX2), fall back to scalar softmax.
    {
        memory_desc_t sm_md;
        dims_t sm_dims = {q_block_, seq_kv};
        if (memory_desc_init_by_tag(
                    sm_md, 2, sm_dims, data_type::f32, format_tag::ab)
                == status::success) {
            softmax_desc_t sd {};
            sd.primitive_kind = primitive_kind::softmax;
            sd.prop_kind = prop_kind::forward_inference;
            // inf_as_zero: a fully-masked row (all -inf) must produce an
            // all-zero row instead of NaN. The jit softmax kernel implements
            // this natively for the softmax_accurate_inf_as_zero alg.
            sd.alg_kind = p_.softmax_inf_as_zero
                    ? alg_kind::softmax_accurate_inf_as_zero
                    : alg_kind::softmax_accurate;
            sd.src_desc = sm_md;
            sd.dst_desc = sm_md;
            sd.softmax_axis = 1;

            primitive_attr_t attr;
            primitive_desc_iterator_t it(engine,
                    reinterpret_cast<const op_desc_t *>(&sd), &attr, nullptr);
            if (it.is_initialized() && ++it != it.end()) {
                softmax_pd_ = *it;
                auto *jit_pd = dynamic_cast<jit_uni_softmax_fwd_t::pd_t *>(
                        softmax_pd_.get());
                if (jit_pd) {
                    softmax_impl::jit_softmax_kernel_base_t *k
                            = softmax_impl::jit_softmax_kernel_base_t::create(
                                    jit_pd, jit_pd->isa_,
                                    jit_pd->axis_is_plain_and_strided_);
                    if (k && k->create_kernel() == status::success) {
                        softmax_kernel_.reset(k);
                        use_jit_softmax_ = true;
                    } else {
                        delete k;
                    }
                }
            }
        }
    }

    return status::success;
}

status_t sdp_blocked_driver_t::execute(const sdp_blocked_run_args_t &args,
        void *scratch_base, int nthr) const {
    const int ndims = p_.ndims;
    const int row_dim = ndims - 2;
    const dim_t seq_q = p_.seq_q;
    const dim_t seq_kv = p_.seq_kv;
    const dim_t hs_v = p_.head_size_v;
    const dim_t group = p_.group_head;
    const dim_t q_block = q_block_;
    const bool has_select = p_.has_select;
    const bool select_fusiable = p_.select_fusiable;
    const bool use_jit = use_jit_softmax_;
    const bool select_in_mm1 = mm1_select_postop_;
    const bool has_mm1_postops = !p_.mm1_post_ops.empty() || select_in_mm1;
    const float fill = args.fill;
    constexpr float neg_inf = -std::numeric_limits<float>::infinity();

    auto *q_base = static_cast<const char *>(args.q);
    auto *k_base = static_cast<const char *>(args.k);
    auto *v_base = static_cast<const char *>(args.v);
    auto *o_base = static_cast<char *>(args.out);
    auto *cond_base = static_cast<const char *>(args.cond);

    // Query-side offset (Q / out / select-cond carry the group axis); KV-side
    // offset (K / V; the group axis has extent 1). Mirrors the online kernel.
    const auto q_side_off = [&](const std::vector<dim_t> &s, dim_t bo, dim_t bi,
                                    dim_t kvh, dim_t gid) -> dim_t {
        return ndims == 4 ? bo * s[0] + bi * s[1]
                          : bo * s[0] + kvh * s[1] + gid * s[2];
    };
    const auto kv_side_off
            = [&](const std::vector<dim_t> &s, dim_t bo, dim_t kvh) -> dim_t {
        return bo * s[0] + kvh * s[1];
    };

    const dim_t q_row = p_.q_strides[row_dim];
    const dim_t o_row = p_.o_strides[row_dim];
    const dim_t o_col = p_.o_strides[ndims - 1];
    const dim_t cond_row = has_select ? p_.cond_strides[row_dim] : 0;

    const dim_t n_qblk = utils::div_up(seq_q, q_block);
    const size_t block_size = scratch_per_thread_;
    const size_t scores_bytes
            = align64(static_cast<size_t>(q_block) * seq_kv * sizeof(float));
    const bool transpose_b = mm1_transpose_k_;
    const dim_t hs_qk = p_.head_size_qk;
    // Element strides to walk the K tensor's seq_kv / hs_qk axes when
    // materialising the dense [hs_qk, seq_kv] transpose (see init: derived from
    // the physical layout, so this handles both natural and pre-transposed K).
    const dim_t k_row = k_seq_stride_;
    const dim_t k_col = k_hs_stride_;

    // transpose_b QK^T: the mm1 B tile must be dense [hs_qk, seq_kv], but the
    // brgemm ukernel has no transposed-B mode. Transpose the whole K tensor
    // ONCE, up front, into the shared buffer that follows the per-thread blocks
    // (each head becomes a dense [hs_qk, seq_kv] tile, mm1 reads it ldb=seq_kv).
    // Doing it once per (batch, kv_head) avoids the redundant per-query-tile
    // transpose. TODO(S3): swap this scalar pass for create_brgemm_matmul_copy_b
    // (transposed_B) once VNNI-blocked B is needed for bf16/f16/int8/fp8 on AMX.
    const dim_t num_head_kv = num_head_kv_;
    const size_t kt_head_elems = static_cast<size_t>(hs_qk) * seq_kv;
    float *kt_all = transpose_b
            ? reinterpret_cast<float *>(static_cast<char *>(scratch_base)
                      + static_cast<size_t>(nthr) * block_size)
            : nullptr;
    if (transpose_b) {
        parallel_nd(p_.batch, num_head_kv, [&](dim_t bo, dim_t kvh) {
            const float *kp = reinterpret_cast<const float *>(k_base
                    + kv_side_off(p_.k_strides, bo, kvh) * sizeof(float));
            float *kt = kt_all
                    + (static_cast<size_t>(bo) * num_head_kv + kvh)
                            * kt_head_elems;
            for (dim_t kv = 0; kv < seq_kv; ++kv)
                for (dim_t h = 0; h < hs_qk; ++h)
                    kt[h * seq_kv + kv] = kp[kv * k_row + h * k_col];
        });
    }

    parallel_nd_ext(nthr, p_.batch, p_.num_head_q, n_qblk,
            [&](int tid, int, dim_t bo, dim_t bi, dim_t qb) {
        const dim_t kvh = bi / group;
        const dim_t gid = bi % group;
        const dim_t q0 = qb * q_block;
        const dim_t m = nstl::min(q_block, seq_q - q0);
        const bool is_tail = m != q_block;

        const float *q_ptr = reinterpret_cast<const float *>(q_base
                + (q_side_off(p_.q_strides, bo, bi, kvh, gid) + q0 * q_row)
                        * sizeof(float));
        const float *k_ptr = reinterpret_cast<const float *>(
                k_base + kv_side_off(p_.k_strides, bo, kvh) * sizeof(float));
        const float *v_ptr = reinterpret_cast<const float *>(
                v_base + kv_side_off(p_.v_strides, bo, kvh) * sizeof(float));
        float *o_ptr = reinterpret_cast<float *>(o_base
                + (q_side_off(p_.o_strides, bo, bi, kvh, gid) + q0 * o_row)
                        * sizeof(float));
        const uint8_t *c_ptr = has_select
                ? reinterpret_cast<const uint8_t *>(cond_base
                          + (q_side_off(p_.cond_strides, bo, bi, kvh, gid)
                                    + q0 * cond_row)
                                  * sizeof(uint8_t))
                : nullptr;

        char *my_scratch = static_cast<char *>(scratch_base) + tid * block_size;
        float *scores = reinterpret_cast<float *>(my_scratch);
        float *pv = reinterpret_cast<float *>(my_scratch + scores_bytes);

        const auto *mm1 = is_tail ? mm1_tail_kernel_ : mm1_kernel_;
        const auto *mm2 = is_tail ? mm2_tail_kernel_ : mm2_kernel_;

        // mm1: scores[m, seq_kv] = Q[m, hs_qk] * K[hs_qk, seq_kv].
        // For a transpose_b QK^T, K was transposed up front into kt_all; point
        // B at this (batch, kv_head)'s dense [hs_qk, seq_kv] tile. Otherwise K
        // is already [hs_qk, seq_kv] and read in place.
        const float *b_ptr = k_ptr;
        if (transpose_b)
            b_ptr = kt_all
                    + (static_cast<size_t>(bo) * num_head_kv + kvh)
                            * kt_head_elems;

        brgemm_batch_element_t batch1;
        batch1.ptr.A = q_ptr;
        batch1.ptr.B = b_ptr;
        if (has_mm1_postops) {
            // Build the binary post-op rhs table in chain order: one
            // entry per binary in mm1_post_ops (a scalar rhs is used as
            // is; a tensor rhs is offset per batch/head/query-tile),
            // then the fill scalar + dense condition for a fused select.
            const void *rhs[8];
            int n = 0;
            for (size_t pi = 0; pi < p_.mm1_post_ops.size(); ++pi) {
                const auto &pop = p_.mm1_post_ops[pi];
                if (!pop.is_binary) continue; // eltwise: no rhs
                const char *base
                        = static_cast<const char *>(args.mm1_post_op_rhs[pi]);
                if (!pop.rhs_is_scalar) {
                    // Offset the rhs base by (batch, head, query-tile) using
                    // its OWN rank: a 4D rhs indexes the flat head bi, a 5D rhs
                    // splits it into (kv_head, group); broadcast axes (dim==1)
                    // contribute nothing. The query row is offset by q0.
                    const auto &d = pop.rhs_dims;
                    const auto &s = pop.rhs_strides;
                    const int rn = static_cast<int>(d.size());
                    dim_t off = 0;
                    if (rn == 4) {
                        if (d[0] != 1) off += bo * s[0];
                        if (d[1] != 1) off += bi * s[1];
                    } else {
                        if (d[0] != 1) off += bo * s[0];
                        if (d[1] != 1) off += kvh * s[1];
                        if (d[2] != 1) off += gid * s[2];
                    }
                    if (d[rn - 2] != 1) off += q0 * s[rn - 2];
                    base += off * types::data_type_size(pop.rhs_dt);
                }
                rhs[n++] = base;
            }
            if (select_in_mm1) {
                rhs[n++] = &fill;
                rhs[n++] = c_ptr;
            }
            // Position-dependent binary broadcasts (e.g. a per-key attention
            // mask, or the dense select condition) address their rhs from the
            // output element's logical offset, computed by the injector as
            // (dst_element - data_C_ptr_) / dt_size. The scores tile is dense
            // [M, seq_kv] and starts at logical (0, 0), so data_C_ptr_ is the
            // tile base and the remaining logical offsets are zero.
            brgemm_post_ops_data_t pod(
                    /*bias=*/nullptr, /*binary_post_ops_rhs=*/rhs,
                    /*oc_logical_off=*/0, /*dst_row_logical_off=*/0,
                    /*data_C_ptr_=*/reinterpret_cast<const char *>(scores),
                    /*first_mb_matrix_addr_off=*/0);
            brgemm_kernel_execute_postops(mm1, 1, &batch1,
                    /*ptr_C=*/scores, /*ptr_D=*/scores, pod, nullptr);
        } else {
            brgemm_kernel_execute(mm1, 1, &batch1, scores, nullptr);
        }

        // Softmax over the full seq_kv axis per row. Each query row
        // sees every key, so the result is exact (no online recurrence).
        // When the jit softmax kernel is available it does the max/exp/
        // normalize; any scale/select not already fused into mm1 is
        // applied in a cheap pre-pass (skipped when both are fused or
        // absent). Otherwise fall back to a scalar two-pass softmax.
        const bool prepass_select = has_select && !select_in_mm1;
        if (use_jit) {
            if (prepass_select) {
                for (dim_t i = 0; i < m; ++i) {
                    float *srow = scores + i * seq_kv;
                    const uint8_t *crow
                            = c_ptr ? c_ptr + i * cond_row : nullptr;
                    for (dim_t j = 0; j < seq_kv; ++j) {
                        float v = srow[j];
                        if (crow) {
                            const bool cond = crow[j] != 0;
                            const bool keep = select_fusiable ? cond : !cond;
                            if (!keep) v = fill;
                        }
                        srow[j] = v;
                    }
                }
            }
            for (dim_t i = 0; i < m; ++i) {
                float *srow = scores + i * seq_kv;
                softmax_impl::jit_softmax_kernel_base_t::call_params_t sp;
                sp.src = srow;
                sp.dst = srow;
                sp.diff_dst = nullptr;
                sp.interim = nullptr;
                sp.src_scales = nullptr;
                sp.dst_scales = nullptr;
                sp.process_n_elems = static_cast<size_t>(seq_kv);
                sp.dst_orig = srow;
                sp.post_ops_binary_rhs_arg_vec = nullptr;
                (*softmax_kernel_)(&sp);
            }
        } else {
            for (dim_t i = 0; i < m; ++i) {
                float *srow = scores + i * seq_kv;
                const uint8_t *crow = prepass_select && c_ptr
                        ? c_ptr + i * cond_row
                        : nullptr;

                // Pass 1: optional select-mask, track the max (the QK
                // scale and any other mm1 post-ops are already folded
                // into the scores by the BRGEMM store).
                float row_max = neg_inf;
                for (dim_t j = 0; j < seq_kv; ++j) {
                    float v = srow[j];
                    if (crow) {
                        const bool cond = crow[j] != 0;
                        const bool keep = select_fusiable ? cond : !cond;
                        if (!keep) v = fill;
                    }
                    srow[j] = v;
                    if (v > row_max) row_max = v;
                }

                // Pass 2: exponentiate around the max, accumulate sum.
                float row_sum = 0.0f;
                for (dim_t j = 0; j < seq_kv; ++j) {
                    const float e = expf(srow[j] - row_max);
                    srow[j] = e;
                    row_sum += e;
                }
                // A fully-masked row (row_max == -inf) has row_sum == 0. Under
                // inf_as_zero it becomes an all-zero row; otherwise standard
                // softmax yields NaN. Any finite row_max gives row_sum >= 1.
                const float inv = row_sum > 0.0f
                        ? 1.0f / row_sum
                        : (p_.softmax_inf_as_zero
                                          ? 0.0f
                                          : std::numeric_limits<
                                                    float>::quiet_NaN());
                for (dim_t j = 0; j < seq_kv; ++j)
                    srow[j] *= inv;
            }
        }

        // mm2: pv[m, hs_v] = P[m, seq_kv] * V[seq_kv, hs_v].
        brgemm_batch_element_t batch2;
        batch2.ptr.A = scores;
        batch2.ptr.B = v_ptr;
        brgemm_kernel_execute(mm2, 1, &batch2, pv, nullptr);

        // Scatter the dense pv tile to the (possibly strided) output.
        for (dim_t i = 0; i < m; ++i) {
            const float *prow = pv + i * hs_v;
            float *out_row = o_ptr + i * o_row;
            for (dim_t d = 0; d < hs_v; ++d)
                out_row[d * o_col] = prow[d];
        }
    });

    return status::success;
}

#else // !DNNL_X64

sdp_blocked_driver_t::~sdp_blocked_driver_t() = default;

status_t sdp_blocked_driver_t::init(
        const sdp_blocked_params_t &params, engine_t *engine) {
    UNUSED(params);
    UNUSED(engine);
    return status::unimplemented;
}

status_t sdp_blocked_driver_t::execute(const sdp_blocked_run_args_t &args,
        void *scratch_base, int nthr) const {
    UNUSED(args);
    UNUSED(scratch_base);
    UNUSED(nthr);
    return status::unimplemented;
}

#endif // DNNL_X64

} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl
