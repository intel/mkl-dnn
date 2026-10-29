/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "gpu/intel/gemm/jit.hpp"
#include "common/c_types_map.hpp"
#include "common/type_helpers.hpp"
#include "gemmstone/driver_info.hpp"
#include "gpu/intel/compute/utils.hpp"
#include "gpu/intel/gemm/host_scalars.hpp"
#include "gpu/intel/gemm/jit/walk_orders.hpp"
#include "gpu/intel/jit/ir/block_2d_utils.hpp"
#include "gpu/intel/jit/utils/utils.hpp"
#include "gpu/intel/logging.hpp"

#ifdef DNNL_WITH_SYCL
#include "gpu/intel/sycl/sycl_stream.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace gemm {

bool check_memory_storage(const memory_storage_t *storage, const char *name) {
    if (storage && *storage) return true;
    VERROR(primitive, gpu, "%s,%s: %s", "jit::gemm", "argument is not set",
            name);
    return false;
}

namespace {

// Offset of the 2D quantization parameters (zero points, scales and group
// sums) for the block starting at (mn, k). They are laid out for the full
// matrix with leading dimension ld, and mn_major is set when m (A) or n (B)
// is the contiguous dimension.
dim_t quant_offset(
        bool mn_major, dim_t ld, int group_mn, int group_k, dim_t mn, dim_t k) {
    dim_t i = mn / std::max(group_mn, 1);
    dim_t j = k / std::max(group_k, 1);
    return mn_major ? i + j * ld : i * ld + j;
}

} // namespace

status_t gen_t::launch_nocopy(const exec_ctx_t &ctx,
        intel::stream_t *compute_stream, zero_pool_t *zero_pool,
        const memory_storage_t &a, const memory_storage_t &b,
        const memory_storage_t &c, const memory_storage_t *ao,
        const memory_storage_t *bo, int16_t ao_host_scalar,
        int16_t bo_host_scalar, const memory_storage_t *a_scales,
        const memory_storage_t *b_scales, const memory_storage_t *c_scales,
        const memory_storage_t *ag, const memory_storage_t *bg,
        const memory_storage_t &co, int16_t co_host_scalar,
        const memory_storage_t *c_temp,
        const memory_storage_t *zero_buf_scratchpad,
        const memory_storage_t *sround_seed, int po_count,
        const memory_storage_t **po_srcs, int64_t offset_a, int64_t offset_b,
        int64_t offset_c, int64_t offset_ao, int64_t offset_bo,
        int64_t offset_aq, int64_t offset_bq, int64_t offset_co,
        int64_t *offset_po_src, int32_t lda, int32_t ldb, int32_t ldc,
        int32_t ldaq, int32_t ldbq, int32_t m, int32_t n, int32_t k, int32_t k0,
        float alpha, float beta, int32_t cmask, bool last_k_block, bool swap_ab,
        bool disable_hilbert) const {
    if (pd()->desc()->batch() == 0) return status::success;

    std::unique_ptr<memory_storage_t> zeros;
    const memory_storage_t *zero_buf = nullptr;
    int zp_token = 0;
    if (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps()) {
        if (zero_pool) {
            CHECK(zero_pool->claim(
                    compute_stream, zero_buffer_bytes_, zeros, &zp_token));
            zero_buf = zeros.get();
        } else {
            zero_buf = zero_buf_scratchpad;
            auto nelems = uint32_t(zero_buffer_bytes_ / sizeof(uint32_t));
            compute::kernel_arg_list_t zero_arg_list;
            zero_arg_list.set(0, *zero_buf);
            zero_arg_list.set(1, nelems);
            compute::range_t zero_gws = {size_t(nelems)};
            compute::range_t zero_lws
                    = {size_t(std::min(256, (int)utils::max_pow2_div(nelems)))};
            auto zero_nd_range = compute::nd_range_t(zero_gws, zero_lws);
            CHECK(parallel_for(
                    ctx, zero_nd_range, zero_fill_kernel_, zero_arg_list));
        }
    }

    uint32_t flags = 0;
    bool k_parallel_fixed
            = (nocopy_info()->kParallel() || nocopy_info()->kParallelLocal())
            && !nocopy_info()->kParallelVariable();

    auto problem = pd()->kernel_desc()->problem();

    if (!last_k_block) flags |= gemmstone::FlagNonfinalKBlock;
    if (cmask & 1) flags |= gemmstone::FlagCOColumn;
    if (cmask & 2) flags |= gemmstone::FlagCORow;

    compute::kernel_arg_list_t arg_list;
    int argn = 0;

    arg_list.set(argn++, a);
    arg_list.set(argn++, b);
    arg_list.set(argn++, c);
    arg_list.set(argn++, offset_a);
    arg_list.set(argn++, offset_b);
    arg_list.set(argn++, offset_c);
    arg_list.set(argn++, lda);
    arg_list.set(argn++, ldb);
    arg_list.set(argn++, ldc);
    arg_list.set(argn++, m);
    arg_list.set(argn++, n);
    arg_list.set(argn++, k);

    set_scalar_arg_cvt(arg_list, argn++, alpha, scalar_type_);
    set_scalar_arg_cvt(arg_list, argn++, beta, scalar_type_);

    bool with_a_zp = pd()->with_a_zero_points();
    bool with_b_zp = pd()->with_b_zero_points();
    if (swap_ab) std::swap(with_a_zp, with_b_zp);
    bool a_zp_ptr = with_a_zp && !problem->aOffsetHostScalar();
    bool b_zp_ptr = with_b_zp && !problem->bOffsetHostScalar();

    if (a_zp_ptr) arg_list.set(argn++, *ao);
    if (b_zp_ptr) arg_list.set(argn++, *bo);
    if (problem->aOffsetHostScalar()) arg_list.set(argn++, ao_host_scalar);
    if (problem->bOffsetHostScalar()) arg_list.set(argn++, bo_host_scalar);
    if (problem->aScale2D()) arg_list.set(argn++, *a_scales);
    if (problem->bScale2D()) arg_list.set(argn++, *b_scales);
    if (pd()->with_mx_scale()) arg_list.set(argn++, *c_scales);
    if (problem->needsAGroupSums()) {
        if (!check_memory_storage(ag, "ag")) return status::runtime_error;
        arg_list.set(argn++, *ag);
    }
    if (problem->needsBGroupSums()) {
        if (!check_memory_storage(bg, "bg")) return status::runtime_error;
        arg_list.set(argn++, *bg);
    }

    if (problem->aOffset2D() || problem->aScale2D()
            || problem->needsAGroupSums())
        arg_list.set(argn++, ldaq);
    if (problem->bOffset2D() || problem->bScale2D()
            || problem->needsBGroupSums())
        arg_list.set(argn++, ldbq);
    if (pd()->with_mx_scale()) {
        auto ldcq = pd()->desc()->m() / problem->cqGroupM;
        arg_list.set(argn++, ldcq);
    }
    if (problem->usesCOPtr()) {
        if (co.is_null()) return status::runtime_error;
        arg_list.set(argn++, co);
        arg_list.set(argn++, offset_co);
        if (pd()->with_bias()) {
            auto ldco = into<int32_t>(pd()->desc()->ld_bias());
            arg_list.set(argn++, ldco);
        }
    } else if (problem->cOffsetHostScalar()) {
        arg_list.set(argn++, co_host_scalar);
    }
    if (nocopy_info()->needsTempC()) arg_list.set(argn++, *c_temp);
    if (problem->postOps.cStochasticRound) {
        arg_list.set(argn++, *sround_seed);
    }
    arg_list.set(argn++, flags);
    if (k_parallel_fixed) arg_list.set(argn++, k0);

    for (int i = 0; i < po_count; i++) {
        if (!po_srcs[i]) continue;
        arg_list.set(argn++, *po_srcs[i]);
        arg_list.set(argn++, offset_po_src[i]);

        if (problem->postOps.binaryRow[i] && problem->postOps.binaryCol[i])
            arg_list.set(argn++, int32_t(pd()->ld_binary(i)));
    }

    if (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps()) {
        arg_list.set(argn++, *zero_buf);
    }

    if (pd()->batch_dims() >= 1) {
        for (int i = pd()->batch_dims() - 1; i >= 0; i--) {
            auto stride_a = int64_t(pd()->stride(DNNL_ARG_A, i));
            auto stride_b = int64_t(pd()->stride(DNNL_ARG_B, i));
            if (swap_ab) std::swap(stride_a, stride_b);
            auto stride_c = int64_t(pd()->desc()->stride_c(i));
            if (jit::enable_generator_dsl()) {
                auto *intel_engine = utils::downcast<intel::engine_t *>(
                        compute_stream->engine());
                auto hw = ngen::getCore(
                        intel_engine->device_info()->product().family);

                // 2d Surface pointer needs to be 64 byte aligned. When negative
                // bounds checking is unnecessary, this restriction can be
                // relaxed by rounding down the surface pointer and adjusting
                // the width accordingly.
                auto base_alignment = intel::jit::block_2d_base_alignment(hw);
                auto a_type = pd()->get_type(DNNL_ARG_A);
                auto b_type = pd()->get_type(DNNL_ARG_B);
                if (swap_ab) std::swap(a_type, b_type);
                auto a_size = types::data_type_size(a_type);
                if (stride_a * a_size % base_alignment) {
                    gpu_warning() << "Unimplemented load transform";
                    return status::runtime_error;
                }
                auto b_size = types::data_type_size(b_type);
                if (stride_b * b_size % base_alignment) {
                    gpu_warning() << "Unimplemented load transform";
                    return status::runtime_error;
                }
            }
            arg_list.set(argn++, stride_a);
            arg_list.set(argn++, stride_b);
            arg_list.set(argn++, stride_c);
            int eff_a_arg = DNNL_ARG_A;
            int eff_b_arg = DNNL_ARG_B;
            if (swap_ab) std::swap(eff_a_arg, eff_b_arg);
            if (problem->hasAScalePtr()) {
                arg_list.set(argn++, pd()->scale_stride(i, eff_a_arg));
            }
            if (problem->hasBScalePtr()) {
                arg_list.set(argn++, pd()->scale_stride(i, eff_b_arg));
            }
            if (problem->hasCMXScale()) {
                arg_list.set(argn++, stride_c / problem->cqGroupM);
            }
            if (problem->hasAOffsetPtr()) {
                arg_list.set(argn++, pd()->zp_stride(i, eff_a_arg));
            }
            if (problem->hasBOffsetPtr()) {
                arg_list.set(argn++, pd()->zp_stride(i, eff_b_arg));
            }
            if (problem->needsAGroupSums()) {
                arg_list.set(argn++, pd()->gs_stride(i, eff_a_arg));
            }
            if (problem->needsBGroupSums()) {
                arg_list.set(argn++, pd()->gs_stride(i, eff_b_arg));
            }
        }
        for (int i = 0; i < po_count; i++) {
            if (problem->postOps.binaryBatch[i]) {
                for (int b = pd()->batch_dims() - 1; b >= 0; b--) {
                    arg_list.set(argn++, int64_t(pd()->stride_binary(i, b)));
                }
            }
        }
        for (int i = 1; i < pd()->batch_dims(); i++) {
            auto batchSize = uint32_t(pd()->desc()->c_desc.dims[i]);
            arg_list.set(argn++, batchSize);
            if (jit::enable_generator_dsl()) {
                uint64_t magic = dnnl::impl::gpu::intel::jit::ir_utils::
                        idiv_magicgu_packed(batchSize);
                arg_list.set(argn++, magic);
            } else {
                uint32_t recipBatchSize = jit::uint32_reciprocal(batchSize);
                arg_list.set(argn++, recipBatchSize);
            }
        }
    }

    auto lws_k = pd()->kernel_desc()->aux_params()->wgK;

    compute::range_t gws = compute::range_t::empty();

    gws[0] = utils::div_up(m, nocopy_info()->unroll[gemmstone::LoopM]);
    gws[1] = utils::div_up(n, nocopy_info()->unroll[gemmstone::LoopN]);
    gws[2] = nocopy_info()->kParallel() ? nstl::max(1, utils::div_up(k, k0))
                                        : lws_k;

    compute::range_t lws = {size_t(nocopy_info()->wg[gemmstone::LoopM]),
            size_t(nocopy_info()->wg[gemmstone::LoopN]), size_t(lws_k)};

    // C Interleave: pad up gws[N] to a multiple of the chunk size and add to gws[M] if misaligned ldc
    auto info = nocopy_info();
    gws[1] = utils::rnd_up(
            gws[1], info->cInterleaveChunk(problem->Tc_ext) * lws[1]);
    if (info->cInterleaveEnabled()
            && (offset_c % 64 > 0 || ldc * problem->Tc % 64 > 0)) {
        auto wgTileM = info->wgTile(gemmstone::LoopM);
        auto maxShift = 64 / problem->Tc_ext.size() - 1;
        gws[0] += lws[0] * utils::div_up(wgTileM + maxShift, wgTileM);
    }

    if (nocopy_info()->isNMK()) {
        std::swap(lws[0], lws[1]);
        std::swap(gws[0], gws[1]);
    }

    if (nocopy_info()->fusedEUs() && (lws[0] > 1))
        gws[0] = utils::rnd_up(gws[0], 2);

    lws[2] = nstl::min(lws[2], gws[2]);

    if (nocopy_info()->kParallel() && nocopy_info()->kPadding())
        gws[2] += lws[2];

    int last_non_1 = 2;
    for (; last_non_1 >= 0 && (gws[last_non_1] == 1 || lws[last_non_1] == 1);
            last_non_1--)
        ;

    for (int d = 0; d < 3; d++) {
        if (nocopy_info()->fixedWG() || (gws[d] > lws[d]))
            gws[d] = utils::rnd_up(gws[d], lws[d]);
        else {
            // Workaround to avoid local ID reordering until reqd_walk_group_order implemented in UMD.
            if (pd()->arch_ >= compute::gpu_arch_t::xe_hp && d < last_non_1)
                gws[d] = utils::rnd_up_pow2(gws[d]);
            lws[d] = gws[d];
        }
    }

    lws[1] *= nocopy_info()->wgExpand;
    gws[1] *= nocopy_info()->wgExpand;

    gws[2] *= pd()->desc()->batch();

    jit::linear_order_args(arg_list, argn, lws, gws, m, n, k, disable_hilbert,
            *nocopy_info(), pd()->kernel_desc()->aux_params(), pd()->dev_info_);

    if (nocopy_info()->perKSLM > 0) {
        size_t slm = nocopy_info()->slm;
        if (lws[2] > 1) slm = nstl::max(slm, nocopy_info()->perKSLM * lws[2]);
        arg_list.set(argn++, slm, nullptr);
    }

    if (problem->aoPtrDims >= 1) arg_list.set(argn++, offset_ao);
    if (problem->boPtrDims >= 1) arg_list.set(argn++, offset_bo);
    if (problem->aScale2D()) arg_list.set(argn++, offset_aq);
    if (problem->bScale2D()) arg_list.set(argn++, offset_bq);
    if (problem->needsAGroupSums()) arg_list.set(argn++, offset_aq);
    if (problem->needsBGroupSums()) arg_list.set(argn++, offset_bq);

    lws[0] *= nocopy_info()->subgroupSize;
    gws[0] *= nocopy_info()->subgroupSize;

    auto nd_range = compute::nd_range_t(gws, lws);
    auto status = parallel_for(ctx, nd_range, nocopy_kernel_, arg_list);

    if (zero_pool
            && (nocopy_info()->fusedBeta() || nocopy_info()->fusedPostOps()))
        zero_pool->async_release(zp_token, compute_stream->ctx().get_deps());

    return status;
}

status_t gen_t::execute(const exec_ctx_t &ctx) const {
    auto *compute_stream = utils::downcast<intel::stream_t *>(ctx.stream());

    auto zero_pool = zero_pool_;

#ifdef DNNL_WITH_SYCL
    const auto *sycl_stream
            = utils::downcast<const gpu::intel::sycl::stream_t *>(
                    compute_stream);
    // The zero-pool global state management is not compatible with SYCL graph;
    // fall back to the graph-safe zero-buffer scratchpad path.
    if (sycl_stream->recording()) zero_pool = nullptr;
#endif

    const auto d = pd()->desc();
    const auto &problem = *pd()->kernel_desc()->problem();

    const bool swap_ab = pd()->swap_ab();

    auto a_type = pd()->get_type(DNNL_ARG_A);
    auto b_type = pd()->get_type(DNNL_ARG_B);
    auto c_type = d->c_type();

    auto m = into<int32_t>(pd()->desc()->m());
    auto n = into<int32_t>(pd()->desc()->n());
    auto k = into<int32_t>(d->k());

    bool trans_a = pd()->trans_a();
    bool trans_b = pd()->trans_b();

    auto lda = into<int32_t>(pd()->ld(DNNL_ARG_A));
    auto ldb = into<int32_t>(pd()->ld(DNNL_ARG_B));
    auto ldc = into<int32_t>(d->ldc());
    auto ldco = into<int32_t>(pd()->with_bias() ? d->ld_bias() : 0);

    if (swap_ab) {
        std::swap(a_type, b_type);
        std::swap(m, n);
        std::swap(lda, ldb);
        std::swap(trans_a, trans_b);
        trans_a = !trans_a;
        trans_b = !trans_b;
    }

    auto alpha = pd()->alpha();
    auto beta = pd()->beta();

    bool k_parallel_global = nocopy_info()->kParallel();
    bool k_parallel_fixed
            = (nocopy_info()->kParallel() || nocopy_info()->kParallelLocal())
            && !nocopy_info()->kParallelVariable();

    auto &a = swap_ab ? GEMM_CTX_ARG_STORAGE(a) : GEMM_CTX_ARG_STORAGE(b);
    auto &b = swap_ab ? GEMM_CTX_ARG_STORAGE(b) : GEMM_CTX_ARG_STORAGE(a);
    auto &c = GEMM_CTX_ARG_STORAGE(c);
    auto &c_zp = GEMM_CTX_ARG_STORAGE(c_zero_point);
    auto &bias = GEMM_CTX_ARG_STORAGE(bias);
    auto &sum_ab = GEMM_CTX_ARG_STORAGE(sum_ab);
    auto *sround_seed = &GEMM_CTX_ARG_STORAGE(sround_seed);
    auto *co = &c_zp;
    const memory_storage_t *ao = nullptr, *bo = nullptr;
    const memory_storage_t *a_scales = nullptr, *b_scales = nullptr;
    const memory_storage_t *c_scales = nullptr;
    const memory_storage_t *ag = nullptr, *bg = nullptr;
    int16_t ao_host_scalar = 0;
    int16_t bo_host_scalar = 0;
    int16_t co_host_scalar = 0;

    std::unique_ptr<memory_storage_t> c_temp;
    if (nocopy_info()->needsTempC()) {
        c_temp = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_accumulator);
    }

    std::unique_ptr<memory_storage_t> zero_buf_scratchpad;
    if (need_zero_buffer() && !zero_pool) {
        zero_buf_scratchpad = ctx.get_scratchpad_grantor().get_memory_storage(
                memory_tracking::names::key_gemm_zero_buffer);
    }

    const memory_storage_t *po_srcs[GEMM_MAX_PO];

    int po_count = pd()->post_ops()->len();
    assert(po_count <= GEMM_MAX_PO);

    for (int i = 0; i < po_count; i++) {
        auto &src = pd()->binary_srcs()[i];
        switch (src.type) {
            case pd_t::binary_src_t::binary:
                po_srcs[i]
                        = ctx.args()
                                  .exec_args
                                  .at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(src.index)
                                          | DNNL_ARG_SRC_1)
                                  .mem()
                                  ->memory_storage();
                break;
            case pd_t::binary_src_t::prelu:
                po_srcs[i]
                        = ctx.args()
                                  .exec_args
                                  .at(DNNL_ARG_ATTR_MULTIPLE_POST_OP(src.index)
                                          | DNNL_ARG_WEIGHTS)
                                  .mem()
                                  ->memory_storage();
                break;
            case pd_t::binary_src_t::bias: po_srcs[i] = &bias; break;
            case pd_t::binary_src_t::scales:
                switch (src.index) {
                    case DNNL_ARG_WEIGHTS:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(a_scales);
                        break;
                    case DNNL_ARG_SRC:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(b_scales);
                        break;
                    case DNNL_ARG_DST:
                        po_srcs[i] = &GEMM_CTX_ARG_STORAGE(c_scales);
                        break;
                    default:
                        po_srcs[i] = nullptr;
                        assert(!"invalid scale type");
                        break;
                }
                break;
            default: po_srcs[i] = nullptr; break;
        }
    }

    size_t off_a0
            = types::bytes_to_elements(a_type, a.offset()) + pd()->dyn_offset_a;
    size_t off_b0
            = types::bytes_to_elements(b_type, b.offset()) + pd()->dyn_offset_b;
    size_t off_c0
            = types::bytes_to_elements(c_type, c.offset()) + pd()->dyn_offset_c;
    int64_t off_aq0 = 0, off_bq0 = 0, off_co0 = 0;

    int64_t po_offsets0[GEMM_MAX_PO] = {0}, po_offsets[GEMM_MAX_PO] = {0};
    for (int i = 0; i < po_count; i++)
        if (po_srcs[i])
            po_offsets0[i] = po_srcs[i]->offset() / problem.Tbinary[i];

    int cmask = 0;
    if (pd()->with_c_zero_points()) {
        off_co0 = types::bytes_to_elements(c_type, co->offset())
                + pd()->dyn_offset_co;
        cmask = pd()->attr()->zero_points_.get_mask(DNNL_ARG_DST);
        int co_host_scalar_val = 0;
        if (co->is_host_scalar()) {
            CHECK(maybe_get_host_scalar_value(*co, co_host_scalar_val));
        }
        // DST zero point is added to result (not subtracted like SRC/WEI)
        co_host_scalar = static_cast<int16_t>(co_host_scalar_val);
    } else if (pd()->with_bias()) {
        off_co0 = types::bytes_to_elements(c_type, bias.offset());
        co = &bias;
        cmask = pd()->bias_cmask();
    } else if (pd()->with_sum_ab()) {
        off_co0 = types::bytes_to_elements(c_type, sum_ab.offset());
        co = &sum_ab;
        cmask = pd()->sum_ab_cmask();
        // TODO: Check if this swapping is still needed and correct with logic below
        if (swap_ab) {
            uint8_t swap_table[4] = {0, 2, 1, 3};
            cmask = (cmask & ~3) | swap_table[cmask & 3];
        }
    }

    // Get host scalar zero-point values
    if (pd()->with_a_zero_points() || pd()->with_b_zero_points()) {
        ao = &GEMM_CTX_ARG_STORAGE(a_zero_point);
        bo = &GEMM_CTX_ARG_STORAGE(b_zero_point);
        int a_host_scalar_val = 0;
        int b_host_scalar_val = 0;
        if (ao->is_host_scalar())
            CHECK(maybe_get_host_scalar_value(*ao, a_host_scalar_val));
        if (bo->is_host_scalar())
            CHECK(maybe_get_host_scalar_value(*bo, b_host_scalar_val));

        ao_host_scalar = static_cast<int16_t>(a_host_scalar_val);
        bo_host_scalar = static_cast<int16_t>(b_host_scalar_val);
    }

    // Convert host scalar scales to Alpha
    if (pd()->attr()->scales_.has_host_scalars()) {
        const auto &a_scales = pd()->attr()->scales_.get(DNNL_ARG_A);
        const auto &b_scales = pd()->attr()->scales_.get(DNNL_ARG_B);
        const auto &a_scales_storage = GEMM_CTX_ARG_STORAGE(a_scales);
        const auto &b_scales_storage = GEMM_CTX_ARG_STORAGE(b_scales);
        const auto &c_scales_storage = GEMM_CTX_ARG_STORAGE(c_scales);
        alpha = 1.0f;
        float scale_val = 0;
        if (a_scales.is_host_scalar()) {
            CHECK(maybe_get_host_scalar_value(a_scales_storage, scale_val));
            alpha *= scale_val;
        }
        if (b_scales.is_host_scalar()) {
            CHECK(maybe_get_host_scalar_value(b_scales_storage, scale_val));
            alpha *= scale_val;
        }
        // Limited support of host scalar dst scales
        if (pd()->with_inlined_c_scale()) {
            CHECK(maybe_get_host_scalar_value(c_scales_storage, scale_val));
            gpu_assert(scale_val != 0);
            alpha /= scale_val;
        }
    }

    if (pd()->a_scales_2d()) { a_scales = &GEMM_CTX_ARG_STORAGE(a_scales); }
    if (pd()->b_scales_2d()) { b_scales = &GEMM_CTX_ARG_STORAGE(b_scales); }
    if (pd()->with_mx_scale()) { c_scales = &GEMM_CTX_ARG_STORAGE(c_scales); }

    if (swap_ab) {
        if (problem.needsAGroupSums()) ag = &GEMM_CTX_ARG_STORAGE(b_group_sums);
        if (problem.needsBGroupSums()) bg = &GEMM_CTX_ARG_STORAGE(a_group_sums);
    } else {
        if (problem.needsAGroupSums()) ag = &GEMM_CTX_ARG_STORAGE(a_group_sums);
        if (problem.needsBGroupSums()) bg = &GEMM_CTX_ARG_STORAGE(b_group_sums);
    }

    if (swap_ab) {
        std::swap(ao, bo);
        std::swap(ao_host_scalar, bo_host_scalar);
        std::swap(a_scales, b_scales);

        uint8_t swap_table[4] = {0, 2, 1, 3};
        cmask = (cmask & ~3) | swap_table[cmask & 3];
    }

    status_t status;

    // The 2D quantization parameters are laid out for the full matrix, so
    // their leading dimension and block offsets use the full problem size.
    bool aq_2d = problem.aOffset2D() || problem.aScale2D()
            || problem.needsAGroupSums();
    bool bq_2d = problem.bOffset2D() || problem.bScale2D()
            || problem.needsBGroupSums();
    bool aq_m_major = isColMajor(problem.needsAGroupSums() ? problem.Ag.layout
                    : problem.aScale2D() ? problem.A_scale.layout
                                         : problem.AO.layout);
    bool bq_n_major = !isColMajor(problem.needsBGroupSums() ? problem.Bg.layout
                    : problem.bScale2D() ? problem.B_scale.layout
                                         : problem.BO.layout);
    int32_t ldaq = 0, ldbq = 0;
    if (aq_2d)
        ldaq = into<int32_t>(aq_m_major ? utils::div_up(m, problem.aqGroupM)
                                        : utils::div_up(k, problem.aqGroupK));
    if (bq_2d)
        ldbq = into<int32_t>(bq_n_major ? utils::div_up(n, problem.bqGroupN)
                                        : utils::div_up(k, problem.bqGroupK));
    if (swap_ab ? pd()->b_q2d_k_broadcast() : pd()->a_q2d_k_broadcast())
        ldaq = 0;
    if (swap_ab ? pd()->a_q2d_k_broadcast() : pd()->b_q2d_k_broadcast())
        ldbq = 0;

    auto block_m = nocopy_info()->blocking[0];
    auto block_n = nocopy_info()->blocking[1];
    auto block_k = nocopy_info()->blocking[2];

    bool disable_hilbert = (k <= 64) && nocopy_info()->isHilbert();
    if (disable_hilbert) {
        block_m = nocopy_info()->blockingAlt[0];
        block_n = nocopy_info()->blockingAlt[1];
    }

    if (!utils::one_of(pd()->desc()->c_type(), data_type::f32, data_type::f16))
        block_k = k;
    if (pd()->post_ops()->len() > 0
            && pd()->post_ops()->entry_[0].kind != primitive_kind::sum)
        block_k = k;

    if (k_parallel_fixed)
        block_k = into<int32_t>(pd()->kernel_desc()->aux_params()->k0);

    block_m = utils::rnd_up(block_m, nocopy_info()->wgTile(gemmstone::LoopM));
    block_n = utils::rnd_up(block_n, nocopy_info()->wgTile(gemmstone::LoopN));

    int32_t k0 = 1;
    if (k_parallel_fixed) {
        k0 = block_k;
        block_k = std::max(k, 1);

        if (k_parallel_global && !nocopy_info()->fusedBeta() && beta != 1.0f
                && (k > k0 * pd()->kernel_desc()->aux_params()->wgK)) {
            status = launch_nocopy(ctx, compute_stream, zero_pool, a, b, c, ao,
                    bo, ao_host_scalar, bo_host_scalar, a_scales, b_scales,
                    c_scales, ag, bg, *co, co_host_scalar, nullptr,
                    zero_buf_scratchpad.get(), sround_seed, po_count, po_srcs,
                    off_a0, off_b0, off_c0, off_aq0, off_bq0, off_aq0, off_bq0,
                    off_co0, po_offsets0, lda, ldb, ldc, ldaq, ldbq, m, n, 0, 1,
                    1.0f, beta, 0, false, swap_ab, true);
            if (status) return status;
            beta = 1.0f;
        }
    }

    for (int64_t Bk = 0; Bk < nstl::max<dim_t>(k, 1); Bk += block_k) {
        int64_t size_k = k - Bk;
        bool last_k_block = (size_k <= block_k);
        if (!last_k_block) size_k = block_k;

        for (int64_t Bm = 0; Bm < m; Bm += block_m) {
            int64_t size_m = m - Bm;
            if (size_m > block_m) size_m = block_m;

            auto off_a_src
                    = off_a0 + (!trans_a ? (Bm + Bk * lda) : (Bk + Bm * lda));

            for (int64_t Bn = 0; Bn < n; Bn += block_n) {
                int64_t size_n = n - Bn;
                if (size_n > block_n) size_n = block_n;

                auto off_b_src = off_b0
                        + (!trans_b ? (Bk + Bn * ldb) : (Bn + Bk * ldb));

                auto off_c = off_c0 + Bm + Bn * ldc;

                auto off_aq = off_aq0;
                auto off_bq = off_bq0;
                if (aq_2d)
                    off_aq += quant_offset(aq_m_major, ldaq, problem.aqGroupM,
                            problem.aqGroupK, Bm, Bk);
                if (bq_2d)
                    off_bq += quant_offset(bq_n_major, ldbq, problem.bqGroupN,
                            problem.bqGroupK, Bn, Bk);

                // Zero point vectors are indexed by m (A) or n (B) only.
                auto off_ao = problem.aOffset2D() ? off_aq : off_aq0 + Bm;
                auto off_bo = problem.bOffset2D() ? off_bq : off_bq0 + Bn;

                auto off_co = off_co0;
                switch (cmask & 3) {
                    case 1: off_co += Bn; break;
                    case 2: off_co += Bm; break;
                    case 3:
                        off_co += isColMajor(problem.CO.layout)
                                ? (Bn * ldco + Bm)
                                : (Bm * ldco + Bn);
                        break;
                }

                for (int i = 0; i < po_count; i++) {
                    po_offsets[i] = po_offsets0[i];
                    bool row = problem.postOps.binaryRow[i],
                         col = problem.postOps.binaryCol[i];
                    if (row && col) {
                        auto ld = pd()->ld_binary(i);
                        po_offsets[i] += isColMajor(problem.binary[i].layout)
                                ? (Bn * ld + Bm)
                                : (Bm * ld + Bn);
                    } else if (row)
                        po_offsets[i] += Bm;
                    else if (col)
                        po_offsets[i] += Bn;
                }

                float eff_beta = (Bk == 0) ? beta : 1.0f;
                status = launch_nocopy(ctx, compute_stream, zero_pool, a, b, c,
                        ao, bo, ao_host_scalar, bo_host_scalar, a_scales,
                        b_scales, c_scales, ag, bg, *co, co_host_scalar,
                        c_temp.get(), zero_buf_scratchpad.get(), sround_seed,
                        po_count, po_srcs, off_a_src, off_b_src, off_c, off_ao,
                        off_bo, off_aq, off_bq, off_co, po_offsets, lda, ldb,
                        ldc, ldaq, ldbq, into<int32_t>(size_m),
                        into<int32_t>(size_n), into<int32_t>(size_k), k0, alpha,
                        eff_beta, cmask, last_k_block, swap_ab,
                        disable_hilbert);

                if (status) return status;
            }
        }
    }

    return status::success;
}

} // namespace gemm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
