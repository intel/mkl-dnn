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
#include <cstring>
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

#include "cpu/x64/sdpa/sdpa_full_softmax.hpp"

#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_uni_softmax.hpp"
#include "cpu/x64/sdpa/sdpa_full_softmax_select_ir.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace {
// Round a byte count up to a 64-byte boundary so each thread's scratch block
// starts cache-line aligned when carved from a cache-line-aligned base.
inline size_t align64(size_t n) {
    return (n + 63) & ~static_cast<size_t>(63);
}

// Build (but do not JIT) a BRGEMM descriptor. Kept JIT-free so it can be
// used by configure() and by the select fallback in create_kernels().
status_t build_brgemm_desc(brgemm_desc_t &brg, data_type_t dt, float beta,
        dim_t M, dim_t N, dim_t K, dim_t lda, dim_t ldb, dim_t ldc,
        const std::vector<sdpa_mm1_post_op_t> *post_ops = nullptr,
        bool select_as_postop = false, dim_t full_output_width = 0) {
    CHECK(brgemm_desc_init(&brg, isa_undef, brgemm_addr, dt, dt,
            /*transA=*/false, /*transB=*/false, brgemm_row_major,
            /*alpha=*/1.0f, beta, lda, ldb, ldc, M, N, K, /*strides=*/nullptr));

    // The full output width can be larger than this BRGEMM's N. For tiled
    // mm1, N is the kv-block width, but post-op descriptors cover the full
    // [M, seq_kv] score tile. At execution, data_C_ptr_ anchors that full tile
    // while ptr_D points at the current N-block; the binary injector uses
    // their difference to recover each element's row and global column.
    // This handles per-element and broadcast masks. The full output width
    // defaults to N when the full output fits one tile.
    const dim_t full_tile_width = full_output_width > 0 ? full_output_width : N;

    // Append the supported mm1 post-ops (scale / attention-mask) to the GEMM
    // post-op list. A scalar binary rhs is a [1 x 1] broadcast. For a tensor
    // rhs, axes with extent 1 stay 1 in the descriptor and broadcast; other
    // axes cover this tile and retain their user strides. The optional select
    // uses a scalar fill rhs and a dense [M x full_tile_width] condition.
    // Runtime pointers are supplied per execute call.
    const bool has_post_ops = post_ops && !post_ops->empty();
    if (has_post_ops || select_as_postop) {
        primitive_attr_t attr;
        post_ops_t po;
        if (has_post_ops) {
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
                    const dim_t cols
                            = pop.rhs_dims[rn - 1] == 1 ? 1 : full_tile_width;
                    dims_t rhs_dims = {rows, cols};
                    dims_t rhs_str = {
                            pop.rhs_strides[rn - 2], pop.rhs_strides[rn - 1]};
                    CHECK(memory_desc_init_by_strides(
                            rhs_md, 2, rhs_dims, pop.rhs_dt, rhs_str));
                }
                CHECK(po.append_binary(pop.alg, &rhs_md));
            }
        }
        if (select_as_postop) {
            memory_desc_t fill_md, cond_md;
            dims_t fl_dims = {1, 1};
            CHECK(memory_desc_init_by_tag(
                    fill_md, 2, fl_dims, data_type::f32, format_tag::ab));
            dims_t cd_dims = {M, full_tile_width};
            CHECK(memory_desc_init_by_tag(
                    cond_md, 2, cd_dims, data_type::u8, format_tag::ab));
            CHECK(po.append_binary(
                    alg_kind::binary_select, &fill_md, &cond_md));
        }
        CHECK(attr.set_post_ops(po));
        memory_desc_t dst_md;
        dims_t d_dims = {M, full_tile_width};
        CHECK(memory_desc_init_by_tag(
                dst_md, 2, d_dims, data_type::f32, format_tag::ab));
        CHECK(brgemm_desc_set_postops(&brg, &attr, &dst_md, /*LDD=*/ldc));
    }
    CHECK(brgemm_desc_finalize(&brg));
    return status::success;
}
} // namespace

sdpa_full_softmax_kernels_t::sdpa_full_softmax_kernels_t() = default;

sdpa_full_softmax_kernels_t::~sdpa_full_softmax_kernels_t() {
    for (int qi = 0; qi < 2; ++qi) {
        for (int ki = 0; ki < 2; ++ki) {
            if (mm1_kernels[qi][ki]) brgemm_kernel_destroy(mm1_kernels[qi][ki]);
            if (mm2_kernels[qi][ki]) brgemm_kernel_destroy(mm2_kernels[qi][ki]);
        }
        if (mm2_kernels_beta0[qi]) brgemm_kernel_destroy(mm2_kernels_beta0[qi]);
    }
}

namespace sdpa_full_softmax {

status_t configure(const sdpa_full_softmax_params_t &params,
        sdpa_full_softmax_conf_t &conf, sdpa_full_softmax_descs_t &descs) {
    conf.params = params;
    const sdpa_full_softmax_params_t &p_ = conf.params;

    if (p_.mm_dt != data_type::f32 || p_.out_dt != data_type::f32)
        return status::unimplemented;

    if (p_.o_strides[p_.ndims - 1] != 1) return status::unimplemented;

    const dim_t seq_q = p_.seq_q;
    const dim_t seq_kv = p_.seq_kv;
    const dim_t hs_qk = p_.head_size_qk;
    const dim_t hs_v = p_.head_size_v;
    const int row_dim = p_.ndims - 2;

    const dim_t l2_budget_bytes
            = static_cast<dim_t>(3 * platform::get_per_core_cache_size(2) / 4);

    // Choose the query block so the [q_block x seq_kv] fp32 score tile fits
    // within its L2 budget across mm1 -> softmax -> mm2, while leaving capacity
    // for each matmul's weight panel alongside it.
    const dim_t score_tile_budget_bytes = l2_budget_bytes / 2;
    const dim_t row_bytes = seq_kv * sizeof(float); // one score row
    const dim_t q_block = utils::saturate((dim_t)1, seq_q,
            static_cast<dim_t>(score_tile_budget_bytes / row_bytes));
    conf.q_block = q_block;
    const dim_t q_tail = seq_q % q_block;
    conf.q_tail = q_tail;

    // Tile the seq_kv axis of both matmuls so each brgemm call's weight panel
    // (mm1's K slice [hs_qk x kv_block], mm2's V slice [kv_block x hs_v]) stays
    // L2-resident; an untiled matmul re-streams its whole panel per M-chunk and
    // turns memory-bound once that panel exceeds ~L2. Size kv_block so the wider
    // panel fits within its L2 budget, then round DOWN to a column granule: a
    // 64-wide block when the budget allows, else 32, else a 16-column floor.
    // A large head or small L2 shrinks the block rather than collapsing it to
    // the whole (untiled) axis. Clamp to seq_kv:
    // short context runs as one untiled block (kv_block == seq_kv -> no kv tail,
    // mm2 keeps beta = 0).
    const dim_t b_panel_budget_bytes = l2_budget_bytes / 8;
    const dim_t b_panel_rows = nstl::max(hs_qk, hs_v);
    dim_t kv_block = b_panel_rows > 0
            ? static_cast<dim_t>(
                      b_panel_budget_bytes / (b_panel_rows * sizeof(float)))
            : seq_kv;
    const dim_t granule = kv_block >= 64 ? 64 : (kv_block >= 32 ? 32 : 16);
    kv_block = nstl::max<dim_t>(utils::rnd_dn(kv_block, granule), (dim_t)16);
    kv_block = nstl::min<dim_t>(kv_block, seq_kv);
    conf.kv_block = kv_block;
    const dim_t kv_tail = kv_block < seq_kv ? seq_kv % kv_block : 0;
    conf.kv_tail = kv_tail;
    // >1 block -> mm2 accumulates across blocks: the first block uses a beta = 0
    // kernel (fresh C) and the rest beta = 1, so no destination pre-zeroing is
    // needed. A single block keeps the original fresh-C (beta = 0) path.
    const float mm2_beta = kv_block < seq_kv ? 1.0f : 0.0f;
    conf.mm2_beta = mm2_beta;

    // mm2 writes directly to the contiguous-column user output.
    const dim_t o_row = p_.o_strides[row_dim];

    // BRGEMM leading dims mirror the online kernel: A/B leading dims come from
    // the user strides (row_dim = the M/K row axis); scores are dense.
    //   mm1: scores[m, seq_kv] = Q[m, hs_qk] * K[hs_qk, seq_kv]
    //   mm2: output[m, hs_v]   = P[m, seq_kv] * V[seq_kv, hs_v]
    // BRGEMM reads K directly as [hs_qk, seq_kv], with a contiguous key axis.
    if (p_.k_strides[p_.ndims - 1] != 1) return status::unimplemented;
    const dim_t mm1_ldb = p_.k_strides[row_dim];

    // Build the mm1/mm2 BRGEMM descriptors (no JIT) into `descs` for
    // create_kernels(). kv is the seq_kv sub-block width (mm1 N / mm2 K);
    // the mm1 post-op descriptors keep the full seq_kv width
    // (full_output_width) so the folded mask/select is addressed by global column.
    auto build_tile_descs = [&](dim_t m, dim_t kv, bool select_postop, int qi,
                                    int ki) -> status_t {
        CHECK(build_brgemm_desc(descs.mm1_desc[qi][ki], p_.mm_dt, /*beta=*/0.0f,
                m, kv, hs_qk, /*lda=*/p_.q_strides[row_dim], /*ldb=*/mm1_ldb,
                /*ldc=*/seq_kv, &p_.mm1_post_ops, select_postop,
                /*full_output_width=*/seq_kv));
        // mm2 B is the user V in place (ldb = its row stride).
        // K = kv (one kv-block of the reduction); beta accumulates across them.
        CHECK(build_brgemm_desc(descs.mm2_desc[qi][ki], p_.mm_dt, mm2_beta, m,
                hs_v, kv, /*lda=*/seq_kv,
                /*ldb=*/p_.v_strides[row_dim], /*ldc=*/o_row,
                /*post_ops=*/nullptr, /*select_postop=*/false));
        descs.mm_valid[qi][ki] = true;
        return status::success;
    };

    // Fold select into mm1 only when it is not inverted and its query/key axes
    // are dense [m x seq_kv]. The injector indexes the condition through the
    // destination offsets, so the row/column strides must be [seq_kv, 1]. A
    // broadcast query or key axis has effective stride 0 and uses the pre-pass.
    const dim_t cond_row_stride = p_.has_select
            ? (p_.cond_dims[row_dim] == 1 ? 0 : p_.cond_strides[row_dim])
            : 0;
    const dim_t cond_col_stride = p_.has_select
            ? (p_.cond_dims[p_.ndims - 1] == 1 ? 0
                                               : p_.cond_strides[p_.ndims - 1])
            : 0;
    const bool try_select_postop = p_.has_select && !p_.invert_select
            && cond_row_stride == seq_kv && cond_col_stride == 1;

    conf.mm1_select_postop = try_select_postop;

    // Descriptor grid: [is_q_tail][is_kv_tail]. The query dimension splits into
    // q_block (+ q_tail); the key dimension into kv_block (+ kv_tail). Tail
    // slots are only built when the corresponding remainder is non-zero. The
    // beta = 0 full-kv-block mm2 for the first block (when seq_kv is tiled) is
    // built per q-tile.
    auto build_all_descs = [&](bool select_postop) -> status_t {
        const dim_t ms[2] = {q_block, q_tail};
        const dim_t kvs[2] = {kv_block, kv_tail};
        for (int qi = 0; qi < 2; ++qi) {
            if (qi == 1 && q_tail == 0) continue;
            for (int ki = 0; ki < 2; ++ki) {
                if (ki == 1 && kv_tail == 0) continue;
                CHECK(build_tile_descs(ms[qi], kvs[ki], select_postop, qi, ki));
            }
            if (mm2_beta != 0.0f) {
                CHECK(build_brgemm_desc(descs.mm2_desc_beta0[qi], p_.mm_dt,
                        /*beta=*/0.0f, ms[qi], hs_v, kv_block, /*lda=*/seq_kv,
                        /*ldb=*/p_.v_strides[row_dim],
                        /*ldc=*/o_row));
                descs.beta0_valid[qi] = true;
            }
        }
        return status::success;
    };

    auto clear_descs = [&]() {
        for (int qi = 0; qi < 2; ++qi) {
            for (int ki = 0; ki < 2; ++ki)
                descs.mm_valid[qi][ki] = false;
            descs.beta0_valid[qi] = false;
        }
    };

    if (build_all_descs(try_select_postop) != status::success) {
        // A ukernel config may reject the select post-op at desc build; drop it
        // and let the pre-pass apply the mask instead. create_kernels() mirrors
        // this decision and additionally retries if the compiled kernel itself
        // rejects the post-op.
        clear_descs();
        conf.mm1_select_postop = false;
        CHECK(build_all_descs(false));
    }

    // Per-thread scratch: the f32 score tile also holds P for mm2.
    const size_t scores_bytes
            = align64(static_cast<size_t>(q_block) * seq_kv * sizeof(float));
    conf.scratch_per_thread = scores_bytes;

    conf.nthr = dnnl_get_max_threads();

    return status::success;
}

status_t create_kernels(const sdpa_full_softmax_conf_t &conf,
        const sdpa_full_softmax_descs_t &descs, engine_t *engine,
        sdpa_full_softmax_kernels_t &kernels) {
    const sdpa_full_softmax_params_t &p_ = conf.params;
    // Bind conf scalars + kernel slots to the member names the body uses.
    const dim_t q_block_ = conf.q_block, q_tail_ = conf.q_tail;
    const dim_t kv_block_ = conf.kv_block, kv_tail_ = conf.kv_tail;
    auto &mm1_kernels_ = kernels.mm1_kernels;
    auto &mm2_kernels_ = kernels.mm2_kernels;
    auto &mm2_kernels_beta0_ = kernels.mm2_kernels_beta0;
    auto &softmax_pd_ = kernels.softmax_pd;
    auto &softmax_kernel_ = kernels.softmax_kernel;
    auto &select_kernel_ = kernels.select_kernel;
    bool &mm1_select_postop_ = kernels.mm1_select_postop;
    mm1_select_postop_ = conf.mm1_select_postop;

    const dim_t seq_kv = p_.seq_kv;
    const dim_t hs_qk = p_.head_size_qk;
    const int row_dim = p_.ndims - 2;
    // The select-post-op fallback keeps the direct-K leading dimension.
    const dim_t mm1_ldb = p_.k_strides[row_dim];

    auto destroy_kernels = [&]() {
        for (int qi = 0; qi < 2; ++qi) {
            for (int ki = 0; ki < 2; ++ki) {
                if (mm1_kernels_[qi][ki]) {
                    brgemm_kernel_destroy(mm1_kernels_[qi][ki]);
                    mm1_kernels_[qi][ki] = nullptr;
                }
                if (mm2_kernels_[qi][ki]) {
                    brgemm_kernel_destroy(mm2_kernels_[qi][ki]);
                    mm2_kernels_[qi][ki] = nullptr;
                }
            }
            if (mm2_kernels_beta0_[qi]) {
                brgemm_kernel_destroy(mm2_kernels_beta0_[qi]);
                mm2_kernels_beta0_[qi] = nullptr;
            }
        }
    };

    // Compile the mm1/mm2 (+ beta-0) kernels directly from the finalized
    // descriptors configure() stored in `descs` -- no rebuild. mm2 descriptors
    // carry no mm1 post-op, so only mm1 is affected by the select fallback.
    auto compile_from_conf = [&]() -> status_t {
        for (int qi = 0; qi < 2; ++qi) {
            for (int ki = 0; ki < 2; ++ki) {
                if (!descs.mm_valid[qi][ki]) continue;
                CHECK(brgemm_kernel_create(
                        &mm1_kernels_[qi][ki], descs.mm1_desc[qi][ki]));
                CHECK(brgemm_kernel_create(
                        &mm2_kernels_[qi][ki], descs.mm2_desc[qi][ki]));
            }
            if (descs.beta0_valid[qi])
                CHECK(brgemm_kernel_create(
                        &mm2_kernels_beta0_[qi], descs.mm2_desc_beta0[qi]));
        }
        return status::success;
    };

    if (compile_from_conf() != status::success) {
        // The compiled ukernel rejected the folded select post-op even though
        // the descriptor built. Drop it and let the pre-pass apply the mask:
        // rebuild the mm1 descriptors without the post-op and recompile; mm2 +
        // beta-0 are unaffected, so recompile those from conf.
        destroy_kernels();
        mm1_select_postop_ = false;
        const dim_t ms[2] = {q_block_, q_tail_};
        const dim_t kvs[2] = {kv_block_, kv_tail_};
        for (int qi = 0; qi < 2; ++qi) {
            if (qi == 1 && q_tail_ == 0) continue;
            for (int ki = 0; ki < 2; ++ki) {
                if (ki == 1 && kv_tail_ == 0) continue;
                brgemm_desc_t mm1_brg;
                CHECK(build_brgemm_desc(mm1_brg, p_.mm_dt, /*beta=*/0.0f,
                        ms[qi], kvs[ki], hs_qk, /*lda=*/p_.q_strides[row_dim],
                        /*ldb=*/mm1_ldb, /*ldc=*/seq_kv, &p_.mm1_post_ops,
                        /*select_postop=*/false,
                        /*full_output_width=*/seq_kv));
                CHECK(brgemm_kernel_create(&mm1_kernels_[qi][ki], mm1_brg));
                CHECK(brgemm_kernel_create(
                        &mm2_kernels_[qi][ki], descs.mm2_desc[qi][ki]));
            }
            if (descs.beta0_valid[qi])
                CHECK(brgemm_kernel_create(
                        &mm2_kernels_beta0_[qi], descs.mm2_desc_beta0[qi]));
        }
    }

    // Reuse the vectorized jit softmax kernel for the max/exp/normalize over
    // the seq_kv axis: build a plain 2D [q_block x seq_kv] f32 softmax pd
    // (axis = 1, the contiguous seq_kv axis) and extract its jit kernel. The
    // kernel is per-row, so the same instance serves the query-tail block too.
    {
        memory_desc_t sm_md;
        dims_t sm_dims = {q_block_, seq_kv};
        CHECK(memory_desc_init_by_tag(
                sm_md, 2, sm_dims, data_type::f32, format_tag::ab));
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
        VCONDCHECK(primitive, create, dispatch, sdpa, it.is_initialized(),
                status::unimplemented, VERBOSE_PRIMITIVE_CREATION_FAIL,
                "softmax");
        // Walk the dispatched impls (order-independent) for the jit softmax pd;
        // its kernel is extracted below and called directly per row.
        jit_uni_softmax_fwd_t::pd_t *jit_pd = nullptr;
        while (++it != it.end()) {
            if (auto *p = dynamic_cast<jit_uni_softmax_fwd_t::pd_t *>(
                        (*it).get())) {
                softmax_pd_ = *it;
                jit_pd = p;
                break;
            }
        }
        VCONDCHECK(primitive, create, dispatch, sdpa, jit_pd != nullptr,
                status::unimplemented, VERBOSE_PRIMITIVE_CREATION_FAIL,
                "softmax");
        softmax_impl::jit_softmax_kernel_base_t *k
                = softmax_impl::jit_softmax_kernel_base_t::create(jit_pd,
                        jit_pd->isa_, jit_pd->axis_is_plain_and_strided_);
        if (!k) return status::unimplemented;
        status_t st = k->create_kernel();
        if (st != status::success) {
            delete k;
            return st;
        }
        softmax_kernel_.reset(k);
    }

    // Build the standalone select-mask pre-pass kernel when a select mask is
    // present and is not already folded into mm1. Polarity, the seq_kv width,
    // and the score/condition row strides are baked in; the row count is a
    // runtime argument, so one instance serves both the full and query-tail
    // tiles. The condition is dense along seq_kv (cond_col == 1) because the
    // pd requires the key axis to be full and plain.
    if (p_.has_select && !mm1_select_postop_) {
        const int ndims = p_.ndims;
        std::vector<dim_t> eff = p_.cond_strides;
        for (int d = 0; d < ndims; ++d)
            if (p_.cond_dims[d] == 1) eff[d] = 0;
        const dim_t cond_row = eff[ndims - 2];
        auto k = std::make_shared<
                sdpa_full_softmax_select_ir::select_ir_kernel_t>(
                /*wseq_kv=*/seq_kv,
                /*invert_select=*/p_.invert_select,
                /*scores_row_stride=*/seq_kv,
                /*cond_row_stride=*/cond_row);
        CHECK(k->create_kernel());
        select_kernel_ = std::move(k);
    }

    return status::success;
}

status_t execute(const sdpa_full_softmax_conf_t &conf,
        const sdpa_full_softmax_kernels_t &kernels,
        const sdpa_full_softmax_run_args_t &args, void *scratch_base,
        int nthr) {
    const sdpa_full_softmax_params_t &p_ = conf.params;
    // Bind conf scalars + compiled kernels to the member names the body uses.
    const dim_t q_block_ = conf.q_block;
    const dim_t kv_block_ = conf.kv_block;
    const size_t scratch_per_thread_ = conf.scratch_per_thread;
    const auto &mm1_kernels_ = kernels.mm1_kernels;
    const auto &mm2_kernels_ = kernels.mm2_kernels;
    const auto &mm2_kernels_beta0_ = kernels.mm2_kernels_beta0;
    const auto &softmax_kernel_ = kernels.softmax_kernel;
    const auto &select_kernel_ = kernels.select_kernel;
    const bool mm1_select_postop_ = kernels.mm1_select_postop;

    const int ndims = p_.ndims;
    const int row_dim = ndims - 2;
    const dim_t seq_q = p_.seq_q;
    const dim_t seq_kv = p_.seq_kv;
    const dim_t group = p_.group_head;
    const dim_t q_block = q_block_;
    const bool has_select = p_.has_select;
    const bool select_in_mm1 = mm1_select_postop_;
    const bool has_mm1_postops = !p_.mm1_post_ops.empty() || select_in_mm1;
    const float fill = args.fill;

    auto *q_base = static_cast<const char *>(args.q);
    auto *k_base = static_cast<const char *>(args.k);
    auto *v_base = static_cast<const char *>(args.v);
    auto *o_base = static_cast<char *>(args.out);
    auto *cond_base = static_cast<const char *>(args.cond);

    // Q, output, and select-condition tensors use query-head coordinates;
    // K and V use batch/KV-head coordinates shared by each GQA group.
    const auto query_tensor_base_offset
            = [&](const std::vector<dim_t> &strides, dim_t batch_idx,
                      dim_t query_head_idx) -> dim_t {
        return batch_idx * strides[0] + query_head_idx * strides[1];
    };
    const auto kv_tensor_base_offset
            = [&](const std::vector<dim_t> &strides, dim_t batch_idx,
                      dim_t kv_head_idx) -> dim_t {
        return batch_idx * strides[0] + kv_head_idx * strides[1];
    };

    const dim_t q_row = p_.q_strides[row_dim];
    const dim_t o_row = p_.o_strides[row_dim];
    // Broadcast-aware select-condition strides: an axis with extent 1 is a
    // broadcast axis whose stride is meaningless (set to the collapsed extent), so it must contribute 0. distill_bert's condition is
    // [1,1,1,seq_kv] -- broadcast over head and seq_q -- so without this the
    // per-head base offset and the per-row (cond_row) advance both overrun the
    // seq_kv-element buffer.
    std::vector<dim_t> eff_cond_strides;
    if (has_select) {
        eff_cond_strides = p_.cond_strides;
        for (int d = 0; d < ndims; ++d)
            if (p_.cond_dims[d] == 1) eff_cond_strides[d] = 0;
    }
    const dim_t cond_row = has_select ? eff_cond_strides[row_dim] : 0;

    const dim_t n_qblk = utils::div_up(seq_q, q_block);
    const size_t block_size = scratch_per_thread_;
    // Each work item computes one query block across the full seq_kv axis,
    // then writes its [m, hs_v] output block.
    parallel_nd_ext(nthr, p_.batch, p_.num_head_q, n_qblk,
            [&](int tid, int, dim_t batch_idx, dim_t query_head_idx,
                    dim_t query_block_idx) {
        const dim_t kv_head_idx = query_head_idx / group;
        const dim_t q0 = query_block_idx * q_block;
        const dim_t m = nstl::min(q_block, seq_q - q0);
        const bool is_tail = m != q_block;

        const char *q_ptr = q_base
                + (query_tensor_base_offset(
                           p_.q_strides, batch_idx, query_head_idx)
                          + q0 * q_row)
                        * sizeof(float);
        const char *k_ptr = k_base
                + kv_tensor_base_offset(p_.k_strides, batch_idx, kv_head_idx)
                        * sizeof(float);
        const char *v_ptr = v_base
                + kv_tensor_base_offset(p_.v_strides, batch_idx, kv_head_idx)
                        * sizeof(float);
        char *o_ptr = o_base
                + (query_tensor_base_offset(
                           p_.o_strides, batch_idx, query_head_idx)
                          + q0 * o_row)
                        * sizeof(float);
        const uint8_t *c_ptr = has_select
                ? reinterpret_cast<const uint8_t *>(cond_base
                          + (query_tensor_base_offset(eff_cond_strides,
                                     batch_idx, query_head_idx)
                                    + q0 * cond_row)
                                  * sizeof(uint8_t))
                : nullptr;

        char *my_scratch = static_cast<char *>(scratch_base) + tid * block_size;
        float *scores = reinterpret_cast<float *>(my_scratch);
        // P (mm2's A operand) is the f32 softmax output in scores.
        void *prob = scores;

        const int qi = is_tail ? 1 : 0;
        // seq_kv tiling: n_kv_full full kv_block-wide blocks + an optional
        // kv_tail remainder block. A single block (kv_block_ == seq_kv) leaves
        // n_kv_full == 1, kv_tail == 0.
        const dim_t kv_block = kv_block_;
        const dim_t n_kv_full = seq_kv / kv_block;
        const dim_t kv_tail = seq_kv - n_kv_full * kv_block;

        // mm1: scores[m, seq_kv] = Q_tile[m, hs_qk] * K_tile[hs_qk, seq_kv].
        // Each call fills one kv_block-wide slice of the full score tile.
        // K is already [hs_qk, seq_kv] with a contiguous key axis.
        const dim_t v_row = p_.v_strides[row_dim];

        // Build the binary post-op rhs table ONCE in chain order (it does not
        // depend on the kv-block): one entry per binary in mm1_post_ops (a
        // scalar rhs is used as is; a tensor rhs is offset per
        // batch/head/query-tile), then the fill scalar + dense condition for a
        // folded select. Each kv-block reuses it; the output pointer identifies
        // the block's columns relative to the full score tile.
        std::vector<const void *> rhs;
        if (has_mm1_postops) {
            rhs.reserve(p_.mm1_post_ops.size() + (select_in_mm1 ? 2 : 0));
            for (size_t pi = 0; pi < p_.mm1_post_ops.size(); ++pi) {
                const auto &pop = p_.mm1_post_ops[pi];
                if (!pop.is_binary) continue; // eltwise: no rhs
                const char *base
                        = static_cast<const char *>(args.mm1_post_op_rhs[pi]);
                if (!pop.rhs_is_scalar) {
                    // Offset the 4D mask by (batch, query head, query tile).
                    // Broadcast axes contribute nothing; the kv-block column
                    // comes from the output pointer, not here.
                    const auto &d = pop.rhs_dims;
                    const auto &s = pop.rhs_strides;
                    dim_t off = 0;
                    if (d[0] != 1) off += batch_idx * s[0];
                    if (d[1] != 1) off += query_head_idx * s[1];
                    if (d[2] != 1) off += q0 * s[2];
                    base += off * types::data_type_size(pop.rhs_dt);
                }
                rhs.push_back(base);
            }
            if (select_in_mm1) {
                rhs.push_back(&fill);
                rhs.push_back(c_ptr);
            }
        }

        // Run mm1 for the kv-block [kv0, kv0+N): B slice offset by kv0 columns,
        // output written into the full [m, seq_kv] score tile at column kv0
        // (LDC/LDD = seq_kv). The post-op injector addresses the mask or select
        // condition at the global column: data_C_ptr_ is the full-tile base,
        // while c_blk is offset by kv0. The rhs table carries no kv0 offset;
        // the injector derives it from those pointers.
        auto run_mm1 = [&](const brgemm_kernel_t *k, dim_t kv0) {
            brgemm_batch_element_t be;
            be.ptr.A = q_ptr;
            be.ptr.B = k_ptr + static_cast<size_t>(kv0) * sizeof(float);
            float *c_blk = scores + kv0;
            if (has_mm1_postops) {
                brgemm_post_ops_data_t pod(
                        /*bias=*/nullptr, /*binary_post_ops_rhs=*/rhs.data(),
                        /*oc_logical_off=*/0,
                        /*dst_row_logical_off=*/0,
                        /*data_C_ptr_=*/reinterpret_cast<const char *>(scores),
                        /*first_mb_matrix_addr_off=*/0);
                brgemm_kernel_execute_postops(
                        k, 1, &be, c_blk, c_blk, pod, nullptr);
            } else {
                brgemm_kernel_execute(k, 1, &be, c_blk, nullptr);
            }
        };

        for (dim_t b = 0; b < n_kv_full; ++b)
            run_mm1(mm1_kernels_[qi][0], b * kv_block);
        if (kv_tail) { run_mm1(mm1_kernels_[qi][1], n_kv_full * kv_block); }

        // Full-axis softmax: scores[m, seq_kv] -> P[m, seq_kv], normalized
        // over all keys independently for each query row. The jit softmax
        // kernel does the max/exp/normalize; any select not already folded
        // into mm1 is applied in a cheap IR pre-pass first.
        const bool prepass_select = has_select && !select_in_mm1;
        if (prepass_select && c_ptr) {
            // Standalone IR select kernel: dense condition (column stride 1),
            // both polarities and the broadcast-over-rows (cond_row == 0) case
            // baked in at build time.
            sdpa_full_softmax_select_ir::select_row_args_t sa;
            sa.scores = scores;
            sa.cond = c_ptr;
            sa.fill = &fill;
            sa.n_rows = m;
            (*select_kernel_)(&sa);
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

        // mm2: out[m, hs_v] = P[m, seq_kv] * V[seq_kv, hs_v], tiled over the
        // seq_kv reduction so each call's B panel [kv_block, hs_v] is
        // L2-resident. The blocks accumulate: when tiled into >1 block the
        // first block uses a beta = 0 kernel (fresh C) and the rest beta = 1, so
        // no destination pre-zeroing is needed; a single block uses beta = 0
        // (identical to the untiled path). P is the f32 scores tile; mm2
        // writes directly to the user output.
        float *mm2_c = reinterpret_cast<float *>(o_ptr);

        // Run mm2 for the kv-block [kv0, kv0+K): A is P's column slice
        // (lda = seq_kv), B is V's panel at kv0 rows down.
        auto run_mm2 = [&](const brgemm_kernel_t *k, dim_t kv0) {
            brgemm_batch_element_t be;
            be.ptr.A = static_cast<const char *>(prob)
                    + static_cast<size_t>(kv0) * sizeof(float);
            be.ptr.B = v_ptr + static_cast<size_t>(kv0) * v_row * sizeof(float);
            brgemm_kernel_execute(k, 1, &be, mm2_c, nullptr);
        };

        // The first kv-block writes fresh C (beta = 0); the rest accumulate
        // with beta = 1. The first block is always a FULL kv_block (the tail,
        // if any, is never first), so mm2_kernels_beta0_ (sized for kv_block_)
        // is the right kernel.
        const bool multi_block = kv_block < seq_kv;
        run_mm2(multi_block ? mm2_kernels_beta0_[qi] : mm2_kernels_[qi][0], 0);
        for (dim_t b = 1; b < n_kv_full; ++b)
            run_mm2(mm2_kernels_[qi][0], b * kv_block);
        if (kv_tail) { run_mm2(mm2_kernels_[qi][1], n_kv_full * kv_block); }
    });

    return status::success;
}

} // namespace sdpa_full_softmax

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
