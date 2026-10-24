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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/tag_traits.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/matmul/brgemm_matmul.hpp"
#include "cpu/x64/matmul/jit_brgemm_matmul_reduce.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {

using namespace dnnl::impl::cpu::matmul;

using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;

using namespace data_type;
namespace {

dim_t get_brg_batchsize(
        const brgemm_matmul_conf_t &bgmmc, bool is_bs_tail, bool is_K_tail) {
    auto bs = is_K_tail  ? 1
            : is_bs_tail ? bgmmc.brgemm_batch_tail_size
                         : bgmmc.brgemm_batch_size;
    return bs;
}

int get_brg_kernel_index(const brgemm_matmul_conf_t &bgmmc, bool is_bs_tail,
        bool do_initialization, int m_ker_idx, int n_ker_idx, bool is_K_tail,
        dim_t bs, bool is_prefetching) {
    const int max_m_ker_idx
            = bgmmc.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;
    if (m_ker_idx >= max_m_ker_idx) return -1;

    auto vM = m_ker_idx > 0
            ? (bgmmc.is_runtime_M ? dynamic_m_tails[m_ker_idx - 1]
                                  : bgmmc.M_tail)
            : bgmmc.M_blk;
    const int max_n_ker_idx
            = bgmmc.is_runtime_N ? max_num_dynamic_n_tails + 1 : 2;
    if (n_ker_idx >= max_n_ker_idx) return -1;

    auto vN = n_ker_idx > 0
            ? (bgmmc.is_runtime_N ? dynamic_n_tails[n_ker_idx - 1]
                                  : bgmmc.N_tail)
            : bgmmc.N_blk;

    if (bgmmc.gemv_swap_a_b) std::swap(vM, vN);

    auto vK = (is_K_tail) ? bgmmc.K_tail : bgmmc.K_blk;

    if (vM == 0 || vN == 0 || vK == 0 || bs == 0) return -1;

    if (bgmmc.is_gemv) {
        assert(bgmmc.gemv_strategy != gemv_strategy_t::none);

        const bool is_transA_strategy = utils::one_of(bgmmc.gemv_strategy,
                gemv_strategy_t::n1_A_trans, gemv_strategy_t::m1_B_plain);

        // vM/vN are already normalized above for GEMV swap strategies.
        // In the normalized GEMV view:
        //   - vM is the current GEMV output block size
        //   - vK is the current reduction block size
        //
        // For GEMV, bgmmc.LDA stores the inferred LDA for BRGEMV.
        // Validate LDA:
        //   - non-transA strategies need LDA to cover vK
        //   - transA strategies need LDA to cover vM
        const dim_t gemv_lda_req = is_transA_strategy ? vM : vK;
        if (bgmmc.LDA < gemv_lda_req) return -1;

        // In BRGEMV, LDC is used as INCY. For transA strategies INCY must
        // be 1. Cases with INCY > 1 are rejected earlier by
        // `is_gemv_applicable()`. Keep a defensive check here as well.
        if (is_transA_strategy && !is_runtime_value(bgmmc.LDC) && bgmmc.LDC > 1)
            return -1;
    } else {
        if (bgmmc.LDA < vK || (bgmmc.LDB < vN && !bgmmc.is_amx)
                || ((bgmmc.LDC < vN && !bgmmc.is_amx)
                        && !is_runtime_value(bgmmc.LDC)))
            return -1;
    }

    if (is_prefetching && !bgmmc.need_prefetch_a && !bgmmc.need_prefetch_b) {
        return -1;
    }
    int idx = 2 * 2 * 2 * max_n_ker_idx * max_m_ker_idx * (int)is_prefetching
            + 2 * 2 * 2 * max_n_ker_idx * m_ker_idx
            + 2 * 2 * max_n_ker_idx * (int)is_bs_tail
            + 2 * max_n_ker_idx * (int)do_initialization + 2 * n_ker_idx
            + (int)is_K_tail;
    assert(idx < max_num_brg_kernels_matmul);
    return idx;
}

} // anonymous namespace

template <cpu_isa_t isa>
int brgemm_matmul_t<isa>::pd_t::get_brg_kernel_idx(bool is_bs_tail,
        bool do_initialization, int m_ker_idx, int n_ker_idx, bool is_K_tail,
        bool is_prefetching) const {
    dim_t bs = get_brg_batchsize(bgmmc_, is_bs_tail, is_K_tail);
    return get_brg_kernel_index(bgmmc_, is_bs_tail, do_initialization,
            m_ker_idx, n_ker_idx, is_K_tail, bs, is_prefetching);
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::pd_t::init(const engine_t *engine) {
    const auto src_dt = src_md_.data_type;
    const auto wei_dt = weights_md_.data_type;
    const auto dst_dt = dst_md_.data_type;

    const bool is_f32 = everyone_is(f32, src_dt, wei_dt, dst_dt);
    const bool is_int8 = one_of(src_dt, u8, s8) && wei_dt == s8
            && one_of(dst_dt, u8, s8, s32, f32, f16, bf16);
    const bool is_f8 = one_of(src_dt, f8_e5m2, f8_e4m3)
            && one_of(wei_dt, f8_e5m2, f8_e4m3)
            && one_of(dst_dt, f32, f16, bf16, f8_e5m2, f8_e4m3);
    const bool is_xf16_fp8 = one_of(src_dt, bf16, f16)
            && one_of(wei_dt, f8_e5m2, f8_e4m3)
            && one_of(dst_dt, f32, f16, bf16, f8_e5m2, f8_e4m3);
    const bool is_bf16
            = everyone_is(bf16, src_dt, wei_dt) && one_of(dst_dt, bf16, f32);
    const bool is_f16
            = everyone_is(f16, src_dt, wei_dt) && one_of(dst_dt, f16, f32);
    const bool is_f32_f16
            = src_dt == f32 && wei_dt == f16 && one_of(dst_dt, f16, f32);
    const bool is_f32_bf16
            = src_dt == f32 && wei_dt == bf16 && one_of(dst_dt, bf16, f32);
    const bool is_bf16_with_int_wei = src_dt == bf16
            && one_of(wei_dt, s8, u8, s4, u4) && one_of(dst_dt, bf16, f32);
    const bool is_f16_with_int_wei = src_dt == f16
            && one_of(wei_dt, s8, u8, s4, u4) && one_of(dst_dt, f16, f32);
    const bool is_f4 = wei_dt == f4_e2m1;
    const bool is_f32_with_int_wei
            = src_dt == f32 && one_of(wei_dt, s8, u8, s4, u4) && dst_dt == f32;
    // int8 grouped (dynamic) quantization: int8 src x {s4,u4,s8,u8} wei,
    // dst in {f16,bf16,f32}, with any grouped scales/ZP attribute (or int4
    // weights, which are always per-OC grouped).
    const auto &attr_scales = attr()->scales_;
    const auto &attr_zps = attr()->zero_points_;
    const bool has_grouped_quant_attrs = utils::one_of(wei_dt, s4, u4)
            || !attr_scales.get(DNNL_ARG_SRC).has_default_groups()
            || !attr_scales.get(DNNL_ARG_WEIGHTS).has_default_groups()
            || !attr_zps.get(DNNL_ARG_SRC).has_default_groups()
            || !attr_zps.get(DNNL_ARG_WEIGHTS).has_default_groups();
    const bool with_int8_grouped_quantization = one_of(src_dt, u8, s8)
            && one_of(wei_dt, s4, u4, s8, u8) && one_of(dst_dt, f16, bf16, f32)
            && has_grouped_quant_attrs;

    auto check_bias = [&]() -> bool {
        const auto bia_dt = weights_md(1)->data_type;
        // The cause in IMPLICATION should be an expression to work around
        // ICE in GCC 7.4.
        const bool is_bia_dt_correct
                = IMPLICATION(is_int8 == true,
                          one_of(bia_dt, f32, s32, s8, u8, f16, bf16))
                && IMPLICATION((is_f8 || is_xf16_fp8),
                        one_of(bia_dt, f32, f16, bf16, src_dt))
                && IMPLICATION(!(is_int8 || is_f8 || is_xf16_fp8),
                        one_of(bia_dt, f32, src_dt));
        return IMPLICATION(with_bias(), is_bia_dt_correct && is_bias_1xN());
    };

    auto check_reduce = [&]() -> bool {
        if (!with_reduce()) return true;

        bool ok = reduce_kind() == matmul_reduce_kind::src;
        ok = ok && src_md()->ndims == 2;
        ok = ok && one_of(src_dt, f32, bf16, f16);

        const memory_desc_wrapper src_mdw(src_md_);
        ok = ok && !src_mdw.has_runtime_dims();
        ok = ok && src_mdw.matches_tag(format_tag::ba);

        const auto skip_mask = primitive_attr_t::skip_mask_t::fpmath_mode;
        ok = ok && attr()->has_default_values(skip_mask);

        return ok;
    };

    auto check_attr_scales = [&]() -> status_t {
        const std::vector<int> supported_args
                = {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS, DNNL_ARG_DST};
        CHECK(attr_scales_ok(engine, supported_args));
        const auto &asc = attr()->scales_;
        if (!asc.has_default_values(DNNL_ARG_SRC)
                && !asc.has_default_values(DNNL_ARG_WEIGHTS)
                && asc.get_mask(DNNL_ARG_WEIGHTS) > 0) {
            // This case requires scratchpad
            VDISPATCH_MATMUL(
                    !is_runtime_value(N()), VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
        // Decompression and grouped quantization also support xf16 source and
        // weights scales. ISA-specific restrictions are checked in init_conf.
        const bool supports_xf16_scales = is_bf16_with_int_wei
                || is_f16_with_int_wei || is_f32_with_int_wei
                || with_int8_grouped_quantization;
        for (const int arg : {DNNL_ARG_SRC, DNNL_ARG_WEIGHTS}) {
            const auto scales_dt = asc.get_data_type(arg);
            VDISPATCH_MATMUL(one_of(scales_dt, undef, f32)
                            || (supports_xf16_scales
                                    && one_of(scales_dt, bf16, f16)),
                    VERBOSE_UNSUPPORTED_SCALES_CFG);
        }
        // The driver reads destination scales as float to compute the reciprocal.
        VDISPATCH_MATMUL(one_of(asc.get_data_type(DNNL_ARG_DST), undef, f32),
                VERBOSE_UNSUPPORTED_SCALES_CFG);
        // Implementation has limited support w.r.t. scales groups.
        if (!asc.has_default_values(DNNL_ARG_WEIGHTS)) {
            if (!asc.get(DNNL_ARG_WEIGHTS).has_default_groups()) {
                // Only grouping over K is supported.
                VDISPATCH_MATMUL(asc.get_group(DNNL_ARG_WEIGHTS, 1) == 1,
                        VERBOSE_UNSUPPORTED_SCALES_CFG);
                // masks 3 per-(k,n) and 7 per-(b,k,n) are supported
                const int mask = asc.get_mask(DNNL_ARG_WEIGHTS);
                const int kn_mask = wei_qmask_N() + wei_qmask_K();
                const int batch_mask = full_tensor_mask() & ~kn_mask;
                const bool mask_ok = (mask & ~(kn_mask | batch_mask)) == 0;
                VDISPATCH_MATMUL(mask_ok, VERBOSE_UNSUPPORTED_SCALES_CFG);
            }
        }
        return status::success;
    };

    auto check_attr_zero_points
            = [&](bool allow_multiple_wei_zp,
                      bool allow_grouped_src_zp = false) -> bool {
        bool ok = true;
        const auto &zp = attr()->zero_points_;
        // Destination zero-points support a common value only.
        if (!zp.has_default_values(DNNL_ARG_DST))
            ok = ok && zp.get_mask(DNNL_ARG_DST) == 0;
        // Source zero-points support a common value only, unless grouping is
        // allowed; then only grouping over M/K (not batch) is supported.
        if (!zp.has_default_values(DNNL_ARG_SRC)) {
            const auto mask = zp.get_mask(DNNL_ARG_SRC);
            if (allow_grouped_src_zp) {
                const int mk_mask = src_qmask_M() + src_qmask_K();
                const int batch_mask = full_tensor_mask() & ~mk_mask;
                ok = ok && (mask & ~(mk_mask | batch_mask)) == 0;
            } else {
                ok = ok && mask == 0;
            }
        }
        // Weights zero-points support a common value only, unless multiple
        // values are allowed; then masks 3 and 7 are supported.
        if (!zp.has_default_values(DNNL_ARG_WEIGHTS)) {
            const auto mask = zp.get_mask(DNNL_ARG_WEIGHTS);
            if (allow_multiple_wei_zp) {
                const auto kn_mask = wei_qmask_N() + wei_qmask_K();
                const int batch_mask = full_tensor_mask() & ~kn_mask;
                ok = ok && (mask & ~(kn_mask | batch_mask)) == 0;
            } else {
                ok = ok && mask == 0;
            }
        }
        return ok;
    };
    const bool problem_dt_correct = one_of(true, is_f4, is_int8, is_f8, is_bf16,
            is_f32, is_f16, is_f32_f16, is_f32_bf16, is_bf16_with_int_wei,
            is_f16_with_int_wei, is_f32_with_int_wei, is_xf16_fp8,
            with_int8_grouped_quantization);

    auto src_d = memory_desc_wrapper(src_md_);
    auto weights_d = memory_desc_wrapper(weights_md_);
    auto bias_d = memory_desc_wrapper(bias_md_);
    auto dst_d = memory_desc_wrapper(dst_md_);
    const bool is_sparse_ok = is_dense_format_kind()
            || (!src_d.is_sparse_desc() && !bias_d.is_sparse_desc()
                    && !dst_d.is_sparse_desc()
                    && weights_d.is_sparse_packed_desc());
    // Disabling verbose dispatch messages for unsupported isa for better
    // readability.
    if (!mayiuse(isa)) return status::unimplemented;

    VDISPATCH_MATMUL(is_sparse_ok, VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_MATMUL(problem_dt_correct, VERBOSE_UNSUPPORTED_DT_CFG);
    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_MATMUL(
            attr()->has_default_values(
                    primitive_attr_t::skip_mask_t::scales_data_type
                            | primitive_attr_t::skip_mask_t::scales_groups
                            | primitive_attr_t::skip_mask_t::
                                    zero_points_data_type
                            | primitive_attr_t::skip_mask_t::zero_points_groups
                            | primitive_attr_t::skip_mask_t::post_ops
                            | primitive_attr_t::skip_mask_t::sum_dt
                            | primitive_attr_t::skip_mask_t::fpmath_mode,
                    dst_dt),
            VERBOSE_UNSUPPORTED_ATTR);
    const auto &po = attr()->post_ops_;

    VDISPATCH_MATMUL(po.check_sum_consistency(dst_dt, is_int8),
            VERBOSE_UNSUPPORTED_POSTOP);

    CHECK(check_attr_scales());
    VDISPATCH_MATMUL(
            check_attr_zero_points(is_bf16_with_int_wei || is_f16_with_int_wei
                            || is_f32_with_int_wei
                            || with_int8_grouped_quantization,
                    with_int8_grouped_quantization),
            VERBOSE_UNSUPPORTED_ZP_CFG);
    VDISPATCH_MATMUL(check_bias(), VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_MATMUL(check_reduce(), VERBOSE_UNSUPPORTED_FEATURE,
            "reduce is not supported");

    CHECK(init_brgemm_matmul_conf(isa, bgmmc_, *desc(), src_md_, weights_md_,
            dst_md_, bias_md_, attr_));

    // f32:f16 configuration on AVX2 doesn't support tails with proper
    // instruction sequence in copy routines. Anchor: F32_F16_AVX2_NO_TAIL.
    VDISPATCH_MATMUL(IMPLICATION((is_f32_f16 || is_f32_bf16) && isa == avx2,
                             bgmmc_.N % 8 == 0),
            "unsupported configuration");

    // ZP per M,N compensation doesn't support strided cases yet.
    if (bgmmc_.with_per_mn_compensation) {
        VDISPATCH_MATMUL(bgmmc_.A_strides[0] == 1 && bgmmc_.B_strides[0] == 1,
                VERBOSE_UNSUPPORTED_TAG);
        const bool wei_is_subbyte = utils::one_of(wei_dt, s4, u4);
        if (wei_is_subbyte) {
            VDISPATCH_MATMUL(
                    bgmmc_.N_blk % 2 == 0 && bgmmc_.B_strides[1] % 2 == 0,
                    VERBOSE_UNSUPPORTED_TAG);
        }
        // Pre-validate the per-(M, N) JIT kernel constraints
        VDISPATCH_MATMUL(
                per_mn_comp_kernel_t::is_applicable(&bgmmc_) == status::success,
                VERBOSE_UNSUPPORTED_FEATURE,
                "per-mn-comp jit kernel not applicable");
    }

    const float alpha = 1.0f;
    const float beta = 1.0f;
    const float beta_init = 0.0f;

    const int max_m_ker_idx
            = bgmmc_.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;
    const int max_n_ker_idx
            = bgmmc_.is_runtime_N ? max_num_dynamic_n_tails + 1 : 2;

    const bool is_amx = isa_has_tile_accumulators(isa);
    const bool is_s8s8 = src_dt == s8 && wei_dt == s8;
    // In the case of dynamic M for amx the last tail kernel generate using
    // non-amx isa. s8s8 proplem type is exception to avoid compensations
    // processing for tail kernel
    const auto backup_isa = is_amx && bgmmc_.is_runtime_M && !is_s8s8
            ? (is_f16 || is_f32_f16 || is_f16_with_int_wei
                              ? avx512_core_fp16
                              : (is_bf16 || is_f32_bf16 || is_bf16_with_int_wei
                                                ? avx512_core_bf16
                                                : (is_int8 ? avx512_core_vnni
                                                           : avx512_core)))
            : isa;

    const int i_bs_end = bgmmc_.brgemm_batch_tail_size ? 2 : 1;
    const int i_init_start = bgmmc_.K_blk != bgmmc_.K ? 0 : 1;
    const int i_K_end = bgmmc_.K_tail ? 2 : 1;

    for_(int i_bs = 0; i_bs < i_bs_end; i_bs++)
    for_(int i_init = i_init_start; i_init < 2; i_init++)
    for_(int i_M = 0; i_M < max_m_ker_idx; i_M++)
    for_(int i_N = 0; i_N < max_n_ker_idx; i_N++)
    for_(int i_K = 0; i_K < i_K_end; i_K++)
    for (int prefetching = 0; prefetching < 2; prefetching++) {
        auto vbeta = (i_init) ? beta_init : beta;
        auto vM = (i_M) == 0 ? bgmmc_.M_blk
                             : (bgmmc_.is_runtime_M ? dynamic_m_tails[i_M - 1]
                                                    : bgmmc_.M_tail);
        auto vN = (i_N) == 0 ? bgmmc_.N_blk
                             : (bgmmc_.is_runtime_N ? dynamic_n_tails[i_N - 1]
                                                    : bgmmc_.N_tail);
        auto vK = (i_K) ? bgmmc_.K_tail : bgmmc_.K_blk;

        dim_t bs = get_brg_batchsize(bgmmc_, i_bs, i_K);
        int idx = get_brg_kernel_idx(i_bs, i_init, i_M, i_N, i_K, prefetching);
        if (idx < 0) continue;

        brgemm_desc_t &brg = brg_descs_[idx];
        const auto kernel_isa = i_M == max_m_ker_idx - 1 ? backup_isa : isa;

        brg.with_per_mn_compensation = bgmmc_.with_per_mn_compensation;
        if (bgmmc_.with_per_mn_compensation) {
            brg.skip_zp_a_compensation = true;
            brg.skip_zp_b_compensation = true;
        }
        if (bgmmc_.with_wei_decompression && bgmmc_.has_zero_point_b)
            brg.skip_zp_b_compensation = true;

        if (bgmmc_.is_gemv) {
            const bool swap_a_b = bgmmc_.gemv_swap_a_b;
            const dim_t gemv_m = swap_a_b ? vN : vM;
            const bool treat_y_as_row = swap_a_b;
            const bool transA = utils::one_of(bgmmc_.gemv_strategy,
                    gemv_strategy_t::n1_A_trans, gemv_strategy_t::m1_B_plain);
            const auto dt_a = swap_a_b ? bgmmc_.wei_dt : bgmmc_.src_dt;
            const auto dt_x = swap_a_b ? bgmmc_.src_dt : bgmmc_.wei_dt;

            CHECK(brgemv_desc_init(&brg, kernel_isa, bgmmc_.brg_type, dt_a,
                    dt_x, transA, alpha, vbeta, bgmmc_.LDA, bgmmc_.LDC, gemv_m,
                    vK, treat_y_as_row));
        } else {
            const dim_t LDA = i_K && bgmmc_.use_buffer_a_tail_only
                    ? bgmmc_.wei_k_blk
                    : bgmmc_.LDA;

            CHECK(brgemm_desc_init(&brg, kernel_isa, bgmmc_.brg_type,
                    bgmmc_.src_dt, bgmmc_.wei_dt, false, false,
                    brgemm_row_major, alpha, vbeta, LDA, bgmmc_.LDB, bgmmc_.LDC,
                    vM, vN, vK, nullptr));
        }

        auto LDD = bgmmc_.LDD;
        brg.skip_wei_scales = bgmmc_.apply_scales_in_buffer_b;
        // Fill up the scales info in case it's computing in brgemm
        if (!brg.skip_wei_scales && bgmmc_.with_wei_scales) {
            brg.is_single_wei_scale = bgmmc_.is_single_wei_scale();
            brg.is_per_n_wei_scales = bgmmc_.is_wei_scale_per_n;
            // For grouped (per-K) wei scales: K_blk == group_size and
            // brgemm_batch_size == 1, so each brgemm call covers exactly one
            // K-group. The matmul driver shifts `ptr_wei_scales` to the
            // current group's per-N vector before each call. Brgemm just
            // applies it as per-N at C-accumulation time.
            brg.is_per_k_wei_scales = bgmmc_.is_wei_scale_per_k;
            brg.dt_wei_scales = bgmmc_.wei_scales_dt;
        }
        if (bgmmc_.with_src_scales) {
            brg.dt_src_scales = bgmmc_.src_scales_dt;
            if (bgmmc_.is_src_scale_per_k) {
                brg.is_per_k_src_scales = true;
                const auto num_k_groups
                        = utils::div_up(bgmmc_.K, bgmmc_.src_scales_k_gsize);
                brg.src_scale_m_stride = num_k_groups * bgmmc_.src_scales_dt_sz;
            }
        }
        CHECK(brgemm_desc_set_postops(
                &brg, attr(), &dst_md_, LDD, bgmmc_.bia_dt));

        // is_c_buf_dst_dt path: keep partial K-chunk accumulation in dst dtype.
        // nthr_k<=1 bypasses the C buffer (accumulate into D, beta=1);
        // nthr_k>1 keeps a C buffer in dst dtype and reduces across K-threads.
        // Either way set dt_c=dst_dt to activate the kernel's narrow-dt_c
        // store/beta paths (see jit_brgemm_amx_uker.cpp).
        if (bgmmc_.is_c_buf_dst_dt) {
            brg.dt_c = brg.dt_d;
            brg.typesize_C = static_cast<int>(types::data_type_size(brg.dt_c));
        }

        brgemm_attr_t brgattr;
        brgattr.generate_skip_accumulation
                = bgmmc_.post_ops_applicable && bgmmc_.nthr_k > 1;
        // With dst-dtype C buffer (nthr_k>1), force intermediate-C-buffer
        // store path even if dt_c==dt_d. Otherwise per-K-thread brgemm
        // writes to reg_D, overrunning dst and racing across K-threads.
        brgattr.use_intermediate_c_buffer
                = bgmmc_.is_c_buf_dst_dt && bgmmc_.use_buffer_c;
        brgattr.mem_advice = bgmmc_.mem_advice;
        brgattr.max_bs = bs;
        brgattr.hint_prefetchw = bgmmc_.hint_prefetchw;
        if (is_superset(kernel_isa, avx512_core_amx)
                || is_superset(kernel_isa, avx10_2_ace)) {
            // For ACE the kernel flavor comes from a single policy helper.
            brgattr.use_uker = IMPLICATION(
                    bgmmc_.is_ace, brgemm_utils::ace_prefer_uker());
            brgattr.use_ace = bgmmc_.is_ace;
            brgattr.use_interleave_stores = !bgmmc_.is_ace;
            brgattr.max_bs = bs;
            brgattr.wary_A_k_tail_read = bgmmc_.extendable_k;
            brgattr.extendable_k = bgmmc_.extendable_k;
            // TODO: change expected sizes to local chunks wrt L2 blocking
            brgattr.hint_expected_A_size = vM * vK * bs;
            brgattr.hint_expected_B_size = vN * vK * bs;
            brgattr.hint_expected_C_size = vM * vN * bs;
            if (bgmmc_.LDB2 != 0) brgattr.LDB2 = bgmmc_.LDB2;

            brgattr.LDC2_N = bgmmc_.M_blk * bgmmc_.LDC;

            brgattr.hint_innermost_loop = brgemm_innermost_undef;
            brgattr.hint_prefetching = brgemm_kernel_prefetching_t::brgemm_prf0;

            brgattr.hint_prfA.sprinkled = bgmmc_.need_prefetch_a && prefetching;
            brgattr.hint_prfB.sprinkled = bgmmc_.need_prefetch_b && prefetching;
            brgattr.hint_fused_copy_a = bgmmc_.use_fused_copy_a;

            if (bgmmc_.set_nt) {
                brgattr.hint_load_nt_A = bgmmc_.is_a_nt ? brgemm_hint_nt_true
                                                        : brgemm_hint_nt_false;
                brgattr.hint_load_nt_B = bgmmc_.is_b_nt ? brgemm_hint_nt_true
                                                        : brgemm_hint_nt_false;
            }
        }

        CHECK(brgemm_desc_set_attr(&brg, brgattr));
        CHECK(brgemm_desc_finalize(&brg));

        // Narrow dt_c is implemented by the AMX uker only, so every kernel in
        // the set has to actually dispatch to it. Checked after the attributes
        // are applied because that is what can_dispatch_uker() reads.
        assert(IMPLICATION(bgmmc_.is_c_buf_dst_dt, brg.can_dispatch_uker()));

        bgmmc_.wsp_tile_per_thr_bytes = nstl::max<dim_t>(
                brg.get_wsp_buffer_size(), bgmmc_.wsp_tile_per_thr_bytes);
    }

    auto scratchpad = scratchpad_registry().registrar();
    init_scratchpad(scratchpad, bgmmc_);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::init(engine_t *engine) {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const int max_m_ker_idx
            = bgmmc.is_runtime_M ? max_num_dynamic_m_tails + 1 : 2;
    const int max_n_ker_idx
            = bgmmc.is_runtime_N ? max_num_dynamic_n_tails + 1 : 2;

    const int i_bs_end = bgmmc.brgemm_batch_tail_size ? 2 : 1;
    const int i_init_start = bgmmc.K_blk != bgmmc.K ? 0 : 1;
    const int i_K_end = bgmmc.K_tail ? 2 : 1;

    for_(int i_bs = 0; i_bs < i_bs_end; i_bs++)
    for_(int i_M = 0; i_M < max_m_ker_idx; i_M++)
    for_(int i_N = 0; i_N < max_n_ker_idx; i_N++)
    for_(int i_K = 0; i_K < i_K_end; i_K++)
    for_(int i_init = i_init_start; i_init < 2; i_init++)
    for (int prefetching = 0; prefetching < 2; prefetching++) {
        int idx = pd()->get_brg_kernel_idx(
                i_bs, i_init, i_M, i_N, i_K, prefetching);
        if (idx < 0) continue;

        brgemm_kernel_t *ker = nullptr;
        CHECK(brgemm_kernel_create(&ker, pd()->get_brg_desc(idx)));
        CHECK(safe_ptr_assign(brg_kernels_[idx], ker));
        if (isa_has_tile_accumulators(pd()->get_brg_desc(idx).isa_impl))
            brgemm_palettes_.insert(idx, pd()->get_brg_desc(idx));

        if (pd()->with_reduce()) {
            if (pd()->reduce_kind() == matmul_reduce_kind::src) {
                if (i_N == 0 && i_init == i_init_start) {
                    reducers_[i_M][i_K] = nullptr;
                    auto db_desc = pd()->get_brg_desc(idx);
                    db_desc.reduce_dim = i_K ? bgmmc.K_tail : bgmmc.K_blk;
                    db_desc.load_dim = i_M ? bgmmc.M_tail : bgmmc.M_blk;

                    if (db_desc.reduce_dim > 0 && db_desc.load_dim > 0) {
                        CHECK(safe_ptr_assign(reducers_[i_M][i_K],
                                new reducer_t(bgmmc, db_desc)));
                        CHECK(reducers_[i_M][i_K]->create_kernel());
                    }
                }
            } else {
                assert(!"unsupported reduce kind");
            }
        }
    }

    if (bgmmc.use_buffer_b && !bgmmc.packed_sparse_weights)
        CHECK(create_brgemm_matmul_copy_b(copy_B_kernel_, &bgmmc));

    if (bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only)
        CHECK(create_brgemm_matmul_copy_a(copy_A_kernel_, &bgmmc));

    // C-buffer dtype for cross-K reduction: by default this is acc_dt
    // (f32 / s32). When relaxed accumulation is enabled and nthr_k > 1
    // the per-thread brgemm writes bf16/f16 into the C buffer, so the
    // reducer must work on bf16/f16 too.
    const bool c_buf_is_dst_dt = bgmmc.is_c_buf_dst_dt && bgmmc.nthr_k > 1;
    if (pd()->with_reduce()
            || (bgmmc.nthr_k > 1 && bgmmc.acc_dt == f32 && !c_buf_is_dst_dt)) {
        CHECK(safe_ptr_assign(
                acc_ker_f32_, new cpu_accumulator_1d_t<data_type::f32>()));
        CHECK(acc_ker_f32_->create_kernel());
    } else if (bgmmc.nthr_k > 1 && bgmmc.acc_dt == s32 && !c_buf_is_dst_dt) {
        CHECK(safe_ptr_assign(
                acc_ker_s32_, new cpu_accumulator_1d_t<data_type::s32>()));
        CHECK(acc_ker_s32_->create_kernel());
    } else if (c_buf_is_dst_dt && bgmmc.dst_dt == data_type::bf16) {
        CHECK(safe_ptr_assign(
                acc_ker_bf16_, new cpu_accumulator_1d_t<data_type::bf16>()));
        CHECK(acc_ker_bf16_->create_kernel());
    } else if (c_buf_is_dst_dt && bgmmc.dst_dt == data_type::f16) {
        CHECK(safe_ptr_assign(
                acc_ker_f16_, new cpu_accumulator_1d_t<data_type::f16>()));
        CHECK(acc_ker_f16_->create_kernel());
    }

    if (bgmmc.packed_sparse_weights) {
        CHECK(safe_ptr_assign(sparse_decompress_kernel_,
                new jit_avx512_sparse_decompress_kernel_t(bgmmc)));
        CHECK(sparse_decompress_kernel_->create_kernel());
    }

    if (bgmmc.with_per_mn_compensation) {
        CHECK(per_mn_comp_kernel_t::create(per_mn_comp_kernel_, &bgmmc));
    }

    return status::success;
}

template <cpu_isa_t isa>
bool brgemm_matmul_t<isa>::determine_prefetch(const dim_t mb, const dim_t m_end,
        const dim_t nb, const dim_t n_end, const brgemm_matmul_conf_t &bgmmc,
        const brg_matmul_exec_ctx_t &brgmm_ctx) const {
    // Prefetch if not the last BRGEMM in the chunk and
    // if the next BRGEMM is identical to the current one.

    assert(!(bgmmc.need_prefetch_a && bgmmc.need_prefetch_b));
    bool do_prefetch = false;

    if (bgmmc.need_prefetch_a) {
        do_prefetch = mb != m_end - 1 && //not last
                brgmm_ctx.get_M_kernel_idx(mb)
                        == brgmm_ctx.get_M_kernel_idx(mb + 1);
    }

    if (bgmmc.need_prefetch_b) {
        do_prefetch = nb != n_end - 1 && //not last
                brgmm_ctx.get_N_kernel_idx(nb)
                        == brgmm_ctx.get_N_kernel_idx(nb + 1);
    }

    return do_prefetch;
}

template <cpu_isa_t isa>
status_t brgemm_matmul_t<isa>::execute_body(const exec_ctx_t &ctx) const {
    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    matmul_helper_t helper(src_d, weights_d, dst_d);

    auto brgmm_ctx_ptr
            = std::make_shared<brg_matmul_exec_ctx_t>(ctx, pd(), helper);

    const int num_threads
            = brgmm_ctx_ptr->get_num_threads_for_parallelization();
    parallel(num_threads,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &brgmm_ctx = *brgmm_ctx_ptr;

        const auto &bgmmc = pd()->get_brgemm_matmul_conf();
        const bool use_buffer_a
                = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;
        const bool is_amx = isa_has_tile_accumulators(isa);
        const dim_t M_chunks = brgmm_ctx.get_M_chunks();
        const int M_chunk_size = brgmm_ctx.get_M_chunk_size();
        const dim_t M_chunk_tail = brgmm_ctx.get_M_chunk_tail();

        const dim_t K_chunks = brgmm_ctx.get_K_chunks();
        const int K_chunk_size = brgmm_ctx.get_K_chunk_size();
        const dim_t K_chunk_tail = brgmm_ctx.get_K_chunk_tail();

        const dim_t N_chunks = brgmm_ctx.get_N_chunks();
        const dim_t N_chunk_tail = brgmm_ctx.get_N_chunk_tail();

        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn_gemm(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;
        dim_t start {0}, end {0};
        balance211(brgmm_ctx.get_parallel_work_amount_gemm(),
                brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn, start, end);
        dim_t kc_start {0}, kc_end {bgmmc.K_chunks};
        if (brgmm_ctx.parallel_reduction_is_used())
            balance211(bgmmc.K_chunks, brgmm_ctx.get_num_threads_for_k(),
                    ithr_k, kc_start, kc_end);

        int prev_ker_idx = -1;
        brgemm_palettes_.maybe_tile_configure(
                is_amx, prev_ker_idx, brgmm_ctx.get_base_brgemm_kernel_idx());

        if (bgmmc.with_dst_scales) {
            const float *dst_scales_ptr = static_cast<const float *>(
                    brgmm_ctx.get_dst_scales_ptr());
            float *dst_scales_inv_ptr = static_cast<float *>(
                    const_cast<void *>(brgmm_ctx.get_dst_scales_inv_ptr(ithr)));
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        dim_t b {0}, mc {0}, nc {0}, b_per_t {0}, mc_per_t {0}, nc_per_t {0},
                bt {0}, mt {0}, nt {0};
        dim_t m_chunks_per_thread = div_up(M_chunks, bgmmc.nthr_m);
        dim_t n_chunks_per_thread = div_up(N_chunks, bgmmc.nthr_n);
        dim_t batch_per_thread = div_up(bgmmc.batch, bgmmc.nthr_b);
        if (brgmm_ctx.is_chunks_horizontal_process_order())
            nd_iterator_init(start, bt, bgmmc.nthr_b, mt, bgmmc.nthr_m, nt,
                    bgmmc.nthr_n, b_per_t, batch_per_thread, mc_per_t,
                    m_chunks_per_thread, nc_per_t, n_chunks_per_thread);
        else
            nd_iterator_init(start, bt, bgmmc.nthr_b, nt, bgmmc.nthr_n, mt,
                    bgmmc.nthr_m, b_per_t, batch_per_thread, nc_per_t,
                    n_chunks_per_thread, mc_per_t, m_chunks_per_thread);
        mc = mt * m_chunks_per_thread + mc_per_t;
        nc = nt * n_chunks_per_thread + nc_per_t;
        b = bt * batch_per_thread + b_per_t;

        auto advance_func = [&]() {
            ++start;
            if (brgmm_ctx.is_chunks_horizontal_process_order())
                nd_iterator_step(bt, bgmmc.nthr_b, mt, bgmmc.nthr_m, nt,
                        bgmmc.nthr_n, b_per_t, batch_per_thread, mc_per_t,
                        m_chunks_per_thread, nc_per_t, n_chunks_per_thread);
            else
                nd_iterator_step(bt, bgmmc.nthr_b, nt, bgmmc.nthr_n, mt,
                        bgmmc.nthr_m, b_per_t, batch_per_thread, nc_per_t,
                        n_chunks_per_thread, mc_per_t, m_chunks_per_thread);
            mc = mt * m_chunks_per_thread + mc_per_t;
            nc = nt * n_chunks_per_thread + nc_per_t;
            b = bt * batch_per_thread + b_per_t;
        };

        dim_t mc_prev = -1;
        dim_t nb_prev = -1;
        dim_t b_prev = -1;
        const char *a_batch_ptr = nullptr;
        const char *b_batch_ptr = nullptr;

        while (start < end) {
            if (mc >= M_chunks || nc >= N_chunks || b >= bgmmc.batch) {
                advance_func();
                continue;
            }

            auto m_start = mc * M_chunk_size;
            const bool m_chunk_tail = mc == M_chunks - 1 && M_chunk_tail > 0;
            auto m_end = m_start + (m_chunk_tail ? M_chunk_tail : M_chunk_size);
            auto n_start = nc * bgmmc.N_chunk_size;
            const bool n_chunk_tail = nc == N_chunks - 1 && N_chunk_tail > 0;
            auto n_end = n_start
                    + (n_chunk_tail ? N_chunk_tail : bgmmc.N_chunk_size);
            dim_t kc_prev = -1;
            if (b != b_prev) {
                a_batch_ptr = brgmm_ctx.get_data_A_batch_ptr(b);
                b_batch_ptr = brgmm_ctx.get_data_B_batch_ptr(b);
            }
            for_(dim_t kc = kc_start; kc < kc_end; kc++)
            {
                const bool k_chunk_tail
                        = kc == K_chunks - 1 && K_chunk_tail > 0;
                auto kb_start = kc * K_chunk_size;
                auto kb_end = kb_start
                        + (k_chunk_tail ? K_chunk_tail : K_chunk_size);

                for (dim_t nb = n_start; nb < n_end; nb++) {
                    const bool bcast_across_all_batch_dims
                            = bgmmc.bcast_B_desc.bcast_across_all_batch_dims;
                    const bool skip_copy_b
                            = (nb_prev == nb && kc_prev == kc
                                      && (b_prev == b
                                              || bcast_across_all_batch_dims))
                            && !bgmmc.packed_sparse_weights;

                    for (dim_t mb = m_start; mb < m_end; mb++) {
                        const bool skip_copy_a = mc_prev == mc && kc_prev == kc
                                && (b_prev == b
                                        || bgmmc.bcast_A_desc
                                                   .bcast_across_all_batch_dims);
                        bool prefetch = determine_prefetch(
                                mb, m_end, nb, n_end, bgmmc, brgmm_ctx);
                        for (dim_t kb = kb_start; kb < kb_end; kb++) {

                            if (bgmmc.use_buffer_b && mb == m_start
                                    && !skip_copy_b)
                                copy_b_chunk_in_buffer(brgmm_ctx, b_batch_ptr,
                                        ithr, b, nb, kb);

                            if (use_buffer_a && nb == n_start && !skip_copy_a)
                                copy_a_chunk_in_buffer(
                                        brgmm_ctx, a_batch_ptr, ithr, mb, kb);

                            compute_kernel(brgmm_ctx, a_batch_ptr, b_batch_ptr,
                                    ithr, b, mb, nb, kb,
                                    kc == kc_start && kb == kb_start,
                                    prev_ker_idx, prefetch);
                        }
                    }
                    kc_prev = kc;
                    nb_prev = nb;
                }
            }
            mc_prev = mc;
            b_prev = b;

            advance_func();
        }
        if (is_amx) { amx_tile_release(); }
    });

    maybe_reduce_and_convert_partial_results_A(brgmm_ctx_ptr);
    maybe_reduce_partial_results_and_apply_postops(brgmm_ctx_ptr);

    return status::success;
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::compute_kernel(
        const brg_matmul_exec_ctx_t &brgmm_ctx, const char *A_data_batch_ptr,
        const char *B_data_batch_ptr, int ithr, dim_t b_idx, dim_t m_blk_idx,
        dim_t n_blk_idx, dim_t k_blk_idx, bool do_init, int &prev_ker_idx,
        bool prefetch) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const auto addr_batch = brgmm_ctx.get_batch_elem_ptr(ithr);

    const auto wsp_tile = brgmm_ctx.get_tile_workspace(ithr);

    const dim_t n = brgmm_ctx.get_N_idx(n_blk_idx, true);

    const dim_t M = brgmm_ctx.get_M();
    const dim_t N = brgmm_ctx.get_N();
    const int m_ker_idx = brgmm_ctx.get_M_kernel_idx(m_blk_idx);
    const int n_ker_idx = brgmm_ctx.get_N_kernel_idx(n_blk_idx);
    const bool is_last_K_blk = brgmm_ctx.is_last_K_blk(k_blk_idx);

    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_blk_idx);
    const dim_t remaining_k_blks
            = (bgmmc.use_buffer_a ? utils::rnd_up(bgmmc.K, bgmmc.K_blk)
                                  : bgmmc.K)
            - k_blk_idx * bgmmc.K_blk * bgmmc.brgemm_batch_size;
    const bool is_K_tail
            = is_last_K_blk && (gemm_batch * bgmmc.K_blk) != remaining_k_blks;

    auto is_bs_tail = (gemm_batch != bgmmc.brgemm_batch_size);
    const int brg_ker_idx = pd()->get_brg_kernel_idx(
            is_bs_tail, do_init, m_ker_idx, n_ker_idx, false, prefetch);
    const auto ptr_bias = brgmm_ctx.get_bias_ptr(n);
    auto ptr_D = brgmm_ctx.get_data_C_ptr(
            b_idx, brgmm_ctx.get_M_idx(m_blk_idx, true), n);
    auto ptr_C = (bgmmc.use_buffer_c)
            ? brgmm_ctx.get_buf_C_ptr(ithr, m_blk_idx, n_blk_idx)
            : ptr_D;

    const auto zp_comp_a
            = brgmm_ctx.get_zp_a_compensation_ptr(ithr, b_idx, n_blk_idx);
    const auto zp_comp_b
            = brgmm_ctx.get_zp_b_compensation_result_ptr(ithr, m_blk_idx);
    const auto &post_ops_binary_rhs_arg_vec
            = brgmm_ctx.get_post_ops_binary_rhs_arg_vec();
    const bool post_ops_applicable = bgmmc.post_ops_applicable
            && (brgmm_ctx.get_num_threads_for_k() <= 1 || bgmmc.K_chunks == 1);

    brgemm_dynamic_values_t leading_dimensions(
            bgmmc.LDA, bgmmc.LDB, brgmm_ctx.get_LDC(), brgmm_ctx.get_LDD());

    brgmm_ctx.maybe_backup_dst_values_to_buffer(
            ithr, b_idx, m_blk_idx, n_blk_idx);

    auto is_brg_amx = [&](const int brg_idx) {
        // Tile accumulators (and therefore the tile workspace) are used by
        // both AMX and ACE, so this must not test for TMUL support alone.
        return isa_has_tile_accumulators(pd()->get_brg_desc(brg_idx).isa_impl);
    };

    // Execute the kernel for one K-block. Pure accumulation into C goes through
    // the plain execute API. Accumulation-time scales / grouped s8s8
    // compensation and the final-block epilogue go through the post-ops API.
    // The `do_only_*` flags map to {do_post_ops, do_apply_comp} as {0,1} (comp
    // only) and {1,1} (epilogue).
    auto execute_brgemm = [&](const brgemm_kernel_t *kernel, const int brg_idx,
                                  const dim_t bs, const bool need_postops,
                                  const bool is_tail) {
        const bool per_k_scales = bgmmc.is_src_scale_per_k
                || (bgmmc.is_wei_scale_per_k
                        && !bgmmc.apply_scales_in_buffer_b);
        const auto k = k_blk_idx * bgmmc.K_blk * bgmmc.brgemm_batch_size;
        const auto m = brgmm_ctx.get_M_idx(m_blk_idx, true);

        // Symmetric src/wei zp + 128-shift compensation tile for this
        // brgemm K-block call
        if (bgmmc.with_per_mn_compensation) {
            fill_per_mn_compensation(brgmm_ctx, ithr, b_idx, m_blk_idx,
                    n_blk_idx, A_data_batch_ptr, B_data_batch_ptr,
                    brgmm_ctx.get_M_kernel_size(m_blk_idx),
                    brgmm_ctx.get_N_kernel_size(n_blk_idx), k_blk_idx, is_tail);
        }
        const void *per_mn_comp = bgmmc.with_per_mn_compensation
                ? static_cast<const void *>(brgmm_ctx.get_per_mn_comp_ptr(ithr))
                : nullptr;

        if (!need_postops) {
            // Intermediate K-block: plain accumulation unless per-K scales
            // or per-MN compensation must be applied during accumulation.
            if (!per_k_scales && !bgmmc.with_per_mn_compensation) {
                void *scratch
                        = is_brg_amx(brg_idx) ? (void *)wsp_tile : nullptr;
                brgemm_kernel_execute(kernel, bs, addr_batch, (void *)ptr_C,
                        scratch, &leading_dimensions);
                return;
            }
            const void *src_scales = bgmmc.is_src_scale_per_k
                    ? brgmm_ctx.get_src_scales_ptr(b_idx, k, m)
                    : (bgmmc.is_wei_scale_per_k && bgmmc.with_src_scales
                                      ? brgmm_ctx.get_src_scales_ptr(b_idx)
                                      : nullptr);
            const void *wei_scales = bgmmc.is_wei_scale_per_k
                            && !bgmmc.apply_scales_in_buffer_b
                    ? brgmm_ctx.get_wei_scales_ptr(b_idx, k, n)
                    : nullptr;
            void *scratch = is_brg_amx(brg_idx) ? (void *)wsp_tile : nullptr;
            const brgemm_post_ops_data_t post_ops_data {/*bias=*/nullptr,
                    /*binary_post_ops_rhs=*/nullptr, /*oc_logical_off=*/0,
                    /*dst_row_logical_off=*/0, /*data_C_ptr_=*/nullptr,
                    /*first_mb_matrix_addr_off=*/0,
                    /*a_zp_compensations=*/nullptr,
                    /*b_zp_compensations=*/nullptr, /*c_zp_values=*/nullptr,
                    /*skip_accumulation=*/false, /*zp_a_val=*/1,
                    /*do_only_comp=*/false,
                    /*do_only_zp_a_val=*/true, src_scales, wei_scales,
                    /*dst_scales=*/nullptr, /*a_zp_values=*/nullptr,
                    /*per_mn_compensation=*/per_mn_comp};
            brgemm_kernel_execute_postops(kernel, bs, addr_batch, (void *)ptr_C,
                    (void *)ptr_D, post_ops_data, scratch, &leading_dimensions);
            return;
        }

        // Final K-block: apply scales, compensation and post-ops. Per-K scales
        // take precedence over the common ones.
        const void *src_scales = bgmmc.is_src_scale_per_k
                ? brgmm_ctx.get_src_scales_ptr(b_idx, k, m)
                : brgmm_ctx.get_src_scales_ptr(b_idx);
        const void *wei_scales
                = bgmmc.is_wei_scale_per_k && !bgmmc.apply_scales_in_buffer_b
                ? brgmm_ctx.get_wei_scales_ptr(b_idx, k, n)
                : brgmm_ctx.get_wei_scales_ptr(b_idx, /*k=*/0, n);
        const void *dst_scales = brgmm_ctx.get_dst_scales_inv_ptr(ithr);
        void *scratch = is_brg_amx(brg_idx)
                ? (void *)wsp_tile
                : (void *)brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx);

        const size_t dst_row_logical_off = m;
        const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                ? b_idx / bgmmc.batch_without_first_dim
                : 0;
        const size_t first_mb_matrix_addr_off
                = batch_first_dim_idx * (M * N) + (dst_row_logical_off * N + n);

        const brgemm_post_ops_data_t post_ops_data {
                /*bias=*/static_cast<const void *>(ptr_bias),
                /*binary_post_ops_rhs=*/post_ops_binary_rhs_arg_vec.data(),
                /*oc_logical_off=*/static_cast<size_t>(n), dst_row_logical_off,
                /*data_C_ptr_=*/brgmm_ctx.get_data_C_ptr(0, 0, 0),
                first_mb_matrix_addr_off,
                /*a_zp_compensations=*/static_cast<const void *>(zp_comp_a),
                /*b_zp_compensations=*/static_cast<const void *>(zp_comp_b),
                /*c_zp_values=*/brgmm_ctx.get_zp_c_ptr(),
                /*skip_accumulation=*/false, /*zp_a_val=*/1,
                /*do_only_comp=*/false, /*do_only_zp_a_val=*/false, src_scales,
                wei_scales, dst_scales, /*a_zp_values=*/nullptr,
                /*per_mn_compensation=*/per_mn_comp};

        brgemm_kernel_execute_postops(kernel, bs, addr_batch, (void *)ptr_C,
                (void *)ptr_D, post_ops_data, scratch, &leading_dimensions);
    };

    if (gemm_batch > 0 && brg_ker_idx >= 0) {
        const auto brg_kernel = brg_kernels_[brg_ker_idx].get();
        assert(brg_kernel != nullptr);
        brgemm_palettes_.maybe_tile_configure(
                is_brg_amx(brg_ker_idx), prev_ker_idx, brg_ker_idx);

        brgmm_ctx.init_brgemm_batch_elements_values(ithr, 0, gemm_batch,
                A_data_batch_ptr, B_data_batch_ptr, b_idx, m_blk_idx, k_blk_idx,
                n_blk_idx);
        const bool need_postops
                = post_ops_applicable && is_last_K_blk && !is_K_tail;
        execute_brgemm(brg_kernel, brg_ker_idx, gemm_batch, need_postops,
                /*is_tail=*/false);

        maybe_reduce_A(brgmm_ctx, ithr, gemm_batch, m_blk_idx, n_blk_idx,
                k_blk_idx, do_init, is_K_tail, /* do_K_tail */ false);
    }
    if (is_K_tail) {
        brgmm_ctx.init_brgemm_batch_elements_values(ithr, gemm_batch, 1,
                A_data_batch_ptr, B_data_batch_ptr, b_idx, m_blk_idx, k_blk_idx,
                n_blk_idx);

        const bool use_init_ker = (do_init && gemm_batch == 0);
        const int brg_ker_idx = pd()->get_brg_kernel_idx(
                false, use_init_ker, m_ker_idx, n_ker_idx, true, prefetch);
        if (brg_ker_idx < 0) {
            assert(!"Requested brgemm kernel was not created.");
            return;
        }
        brgemm_palettes_.maybe_tile_configure(
                is_brg_amx(brg_ker_idx), prev_ker_idx, brg_ker_idx);
        const auto brg_kernel_k_tail = brg_kernels_[brg_ker_idx].get();

        execute_brgemm(brg_kernel_k_tail, brg_ker_idx, /*bs=*/1,
                post_ops_applicable, /*is_tail=*/true);

        maybe_reduce_A(brgmm_ctx, ithr, gemm_batch, m_blk_idx, n_blk_idx,
                k_blk_idx, do_init, is_K_tail,
                /* do_K_tail */ true);
    }

    brgmm_ctx.maybe_restore_dst_values_from_buffer(
            ithr, b_idx, m_blk_idx, n_blk_idx);
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::fill_per_mn_compensation(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, dim_t b_idx,
        dim_t m_blk_idx, dim_t n_blk_idx, const char *A_data_batch_ptr,
        const char *B_data_batch_ptr, dim_t m_kernel_size, dim_t n_kernel_size,
        dim_t k_blk_idx, bool is_tail) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    if (!bgmmc.with_per_mn_compensation) return;
    assert(per_mn_comp_kernel_ != nullptr);

    const int batch_size = brgmm_ctx.get_brgemm_batch_size(k_blk_idx);
    const dim_t k_start = static_cast<dim_t>(k_blk_idx) * bgmmc.K_blk
                    * bgmmc.brgemm_batch_size
            + (is_tail ? batch_size * bgmmc.K_blk : dim_t {0});
    const dim_t k_len
            = nstl::min(is_tail ? bgmmc.K_tail : batch_size * bgmmc.K_blk,
                    bgmmc.K - k_start);

    const dim_t m = brgmm_ctx.get_M_idx(m_blk_idx, true);
    const dim_t n = brgmm_ctx.get_N_idx(n_blk_idx, true);

    // Resolve per-N / per-K-group base pointers via existing getters.
    // For sub-byte src ZP (s4/u4), get_src_zp_ptr byte-aligns the pointer
    // and returns the nibble index so the kernel picks the right nibble.
    int src_zp_nibble_idx = 0;
    const void *src_zp_ptr
            = brgmm_ctx.get_src_zp_ptr(b_idx, k_start, m, &src_zp_nibble_idx);
    const void *wei_zp_ptr = brgmm_ctx.get_wei_zp_ptr(b_idx, k_start, n);

    const bool wei_is_subbyte
            = utils::one_of(bgmmc.orig_wei_dt, data_type::s4, data_type::u4);
    const dim_t wei_k_stride
            = wei_is_subbyte ? bgmmc.B_strides[1] / 2 : bgmmc.B_strides[1];

    per_mn_comp_kernel_t::ctx_t kctx;
    kctx.src_batch_ptr = A_data_batch_ptr + k_start * bgmmc.A_strides[0];
    kctx.wei_batch_ptr = B_data_batch_ptr + k_start * wei_k_stride;
    kctx.src_zp_ptr = src_zp_ptr;
    kctx.src_zp_nibble_idx = src_zp_nibble_idx;
    kctx.wei_zp_ptr = wei_zp_ptr;
    kctx.delta_ptr = brgmm_ctx.get_per_mn_comp_ptr(ithr);
    kctx.m_base = m;
    kctx.n_base = n;
    kctx.M_blk = m_kernel_size;
    kctx.N_blk = n_kernel_size;
    kctx.k_len = k_len;

    // Zero the per-thread delta tile to guard the brgemm consumer
    // against reading unwritten rows when bd_block > M_blk (the AMX
    // tile iterates over bd_block, not M_blk).
    const size_t per_thr_elems = static_cast<size_t>(bgmmc.M_blk)
            * static_cast<size_t>(rnd_up(bgmmc.N_blk, bgmmc.LDC));
    std::memset(kctx.delta_ptr, 0, per_thr_elems * sizeof(float));

    const bool has_src_zp
            = bgmmc.has_zero_point_a && kctx.src_zp_ptr != nullptr;
    const bool has_wei_zp
            = bgmmc.has_zero_point_b && kctx.wei_zp_ptr != nullptr;
    const bool wei_zp_per_n = has_wei_zp && bgmmc.is_wei_zp_per_n;
    const bool src_zp_per_m = has_src_zp && bgmmc.is_src_zp_per_m;
    const bool src_zp_per_k = has_src_zp && bgmmc.is_src_zp_per_k;
    const dim_t src_zp_m_stride_elems = src_zp_per_m
            ? (src_zp_per_k ? utils::div_up(bgmmc.K, bgmmc.src_zp_k_gsize)
                            : dim_t {1})
            : dim_t {0};
    const bool wei_zp_scalar = has_wei_zp && !wei_zp_per_n;
    const float wei_zp_f32 = wei_zp_scalar
            ? io::load_float_value(bgmmc.wei_zp_dt, kctx.wei_zp_ptr, 0)
            : 0.0f;

    const dim_t num_stripes = utils::div_up(kctx.N_blk, bgmmc.LDC);
    const dim_t m_step = src_zp_per_m ? dim_t {1} : kctx.M_blk;

    for (dim_t s = 0; s < num_stripes; ++s) {
        const dim_t n_origin = s * bgmmc.LDC;
        const dim_t stripe_w = nstl::min(bgmmc.LDC, kctx.N_blk - n_origin);
        float *const delta_stripe
                = kctx.delta_ptr + s * bgmmc.M_blk * bgmmc.LDC;

        for (dim_t m_off = 0; m_off < kctx.M_blk; m_off += m_step) {
            per_mn_comp_kernel_t::ctx_t call_ctx = kctx;
            const dim_t rows = nstl::min(m_step, kctx.M_blk - m_off);

            call_ctx.m_base = kctx.m_base + m_off;
            call_ctx.n_base = kctx.n_base + n_origin;
            call_ctx.M_blk = rows;
            call_ctx.delta_ptr = delta_stripe + m_off * bgmmc.LDC;
            call_ctx.stripe_w = stripe_w;
            call_ctx.wei_zp_ptr = wei_zp_per_n
                    ? reinterpret_cast<const uint8_t *>(kctx.wei_zp_ptr)
                            + types::elements_to_bytes(
                                    bgmmc.wei_zp_dt, n_origin)
                    : nullptr;
            call_ctx.wei_zp_f32 = wei_zp_f32;
            call_ctx.src_zp_f32 = has_src_zp
                    ? io::load_float_value(bgmmc.src_zp_dt, kctx.src_zp_ptr,
                              kctx.src_zp_nibble_idx
                                      + m_off * src_zp_m_stride_elems)
                    : 0.0f;
            call_ctx.g_sz_f32 = (has_src_zp && has_wei_zp)
                    ? call_ctx.src_zp_f32 * static_cast<float>(k_len)
                    : 0.0f;

            (*per_mn_comp_kernel_)(&call_ctx);
        }
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::maybe_reduce_A(
        const brg_matmul_exec_ctx_t &brgmm_ctx, int ithr, dim_t gemm_batch,
        dim_t m_blk_idx, dim_t n_blk_idx, dim_t k_chunk_idx, bool do_init,
        bool has_K_tail, bool do_K_tail) const {

    if (!pd()->with_reduce()) return;
    //current state macro heuristics don't support reduce_A -> kb =1 -> kb == kc
    assert(!pd()->get_brgemm_matmul_conf().is_macro_heuristics);
    const bool reduce_a = pd()->reduce_kind() == matmul_reduce_kind::src;
    // Only `matmul_reduce_kind::src` is supported for now.
    assert(reduce_a);

    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const auto *addr_batch = brgmm_ctx.get_batch_elem_ptr(ithr);

    if (reduce_a && n_blk_idx == 0) {
        const dim_t m = brgmm_ctx.get_M_idx(m_blk_idx, true);

        auto *reduce_ptr = bgmmc.use_buffer_reduce
                ? brgmm_ctx.get_buf_reduce_ptr(ithr, m)
                : brgmm_ctx.get_data_reduce_ptr(m);

        brgemm_kernel_reduce_args_t p;

        p.ptr_acc = (void *)reduce_ptr;
        p.ptr_out = (void *)brgmm_ctx.get_data_reduce_ptr(m);

        const int m_ker_idx = brgmm_ctx.get_M_kernel_idx(m_blk_idx);

        if (!do_K_tail) {
            for (dim_t gb = 0; gb < gemm_batch; gb++) {
                p.ptr_in = (void *)addr_batch[gb].ptr.A;

                const bool is_first = do_init && gb == 0;
                const bool is_last = (bgmmc.nthr_k == 1 || bgmmc.K_chunks == 1)
                        && k_chunk_idx == bgmmc.K_chunks - 1
                        && gb == gemm_batch - 1 && !has_K_tail;

                p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                        | (is_last ? FLAG_REDUCE_LAST : 0);

                (*reducers_[m_ker_idx][do_K_tail])(&p);
            }
        } else {
            p.ptr_in = (void *)addr_batch[0].ptr.A;

            const bool is_first = do_init && gemm_batch == 0;
            const bool is_last = (bgmmc.nthr_k == 1 || bgmmc.K_chunks == 1)
                    && k_chunk_idx == bgmmc.K_chunks - 1;

            p.flags = 0 | (is_first ? FLAG_REDUCE_FIRST : 0)
                    | (is_last ? FLAG_REDUCE_LAST : 0);

            (*reducers_[m_ker_idx][do_K_tail])(&p);
        }
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::maybe_reduce_and_convert_partial_results_A(
        const std::shared_ptr<brg_matmul_exec_ctx_t> &brgmm_ctx_ptr) const {
    // Partial results appear when parallel reduction is used.
    //
    // There are two cases that require slightly different handling.
    // - (Figure 1): when reduce data type is not f32. In this case there are
    //   three steps:
    //     * Step 1: add partial results from all reduce buffers except the last
    //       one to the first reduce buffer (reduce_buf_0).
    //     * Step 2 and step 3: add partial results from the first and the last
    //       reduce buffers, convert the result to the reduce data type and
    //       store it to the user provided reduce buffer.
    //
    // - (Figure 2): when reduce data type is f32. In this case the user
    //   provided reduce buffer is used as one of the reduce buffers and there
    //   is only 1 step:
    //     * Step 1: add partial results from all reduce buffers to the user
    //     provided reduce buffer.
    //       buffer.
    //
    //                    Figure 1.
    //             +--------------------+
    //             | reduce (bf16/f16)  |<------+ Step 3.
    //             +--------------------+       |
    //             +--------------------+       |
    //         +-->| reduce_buf_0 (f32) |--->   |
    // Step 1. |   +--------------------+   |   |
    //         |   +--------------------+   |   |
    //         +<--| reduce_buf_1 (f32) |   +---> Step 2.
    //             +--------------------+   |
    //             +--------------------+   |
    //             | reduce_buf_2 (f32) |--->
    //             +--------------------+
    //
    //                    Figure 2.
    //             +--------------------+
    //         +-->|    reduce (f32)    |
    //         |   +--------------------+
    //         |   +--------------------+
    // Step 1. +<--| reduce_buf_0 (f32) |
    //         |   +--------------------+
    //         |   +--------------------+
    //         +<--| reduce_buf_1 (f32) |
    //             +--------------------+

    if (!pd()->with_reduce() || !brgmm_ctx_ptr->parallel_reduction_is_used())
        return;

    const int num_threads
            = brgmm_ctx_ptr->get_num_threads_for_parallelization();
    parallel(num_threads,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &brgmm_ctx = *brgmm_ctx_ptr;

        const auto &bgmmc = pd()->get_brgemm_matmul_conf();
        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;

        const dim_t M_chunks = brgmm_ctx.get_M_chunks();

        dim_t start_mc {0}, end_mc {0};
        balance211(M_chunks, brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn,
                start_mc, end_mc);
        if (start_mc != end_mc && ithr_k == 0) {
            const size_t m = start_mc * bgmmc.M_chunk_elems;
            const size_t mc_work = end_mc - start_mc;
            const size_t acc_size
                    = std::min(mc_work * bgmmc.M_chunk_elems, bgmmc.M - m);

            const bool is_reduce_f32 = bgmmc.reduce_dt == f32;

            float *reduce_acc = is_reduce_f32
                    ? (float *)brgmm_ctx.get_data_reduce_ptr(m)
                    : (float *)brgmm_ctx.get_buf_reduce_ptr_by_index(0, m);

            int ibuf = !is_reduce_f32;
            for (; ibuf < bgmmc.nthr_k - 1; ibuf++) {
                float *reduce_buf
                        = (float *)brgmm_ctx.get_buf_reduce_ptr_by_index(
                                ibuf, m);
                acc_ker_f32_->accumulate(reduce_acc, reduce_buf, acc_size);
            }

            if (!is_reduce_f32) {
                float *reduce_buf
                        = (float *)brgmm_ctx.get_buf_reduce_ptr_by_index(
                                ibuf, m);
                switch (bgmmc.reduce_dt) {
                    case data_type::bf16:
                        add_floats_and_cvt_to_bfloat16(
                                (bfloat16_t *)brgmm_ctx.get_data_reduce_ptr(m),
                                reduce_acc, reduce_buf, acc_size);
                        break;
                    case data_type::f16:
                        add_floats_and_cvt_to_float16(
                                (float16_t *)brgmm_ctx.get_data_reduce_ptr(m),
                                reduce_acc, reduce_buf, acc_size);
                        break;
                    default: assert(!"invalid data type");
                }
            }
        }
    });
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::maybe_reduce_partial_results_and_apply_postops(
        const std::shared_ptr<brg_matmul_exec_ctx_t> &brgmm_ctx_ptr) const {
    if (!brgmm_ctx_ptr->parallel_reduction_is_used()) return;

    const int num_threads
            = brgmm_ctx_ptr->get_num_threads_for_parallelization();
    parallel(num_threads,
            [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        const auto &brgmm_ctx = *brgmm_ctx_ptr;

        const auto &bgmmc = pd()->get_brgemm_matmul_conf();
        const int nthr_k = brgmm_ctx.get_num_threads_for_k();
        const int ithr_bmn = brgmm_ctx.get_thread_idx_for_bmn_gemm(ithr);
        const int ithr_k = brgmm_ctx.get_thread_idx_for_k(ithr);
        if (ithr_bmn < 0 || ithr_k < 0) return;

        const dim_t num_reduction_buffers
                = nstl::min<dim_t>(nthr_k, bgmmc.K_chunks);
        brgemm_dynamic_values_t leading_dimensions(
                bgmmc.LDA, bgmmc.LDB, brgmm_ctx.get_LDC(), brgmm_ctx.get_LDD());
        const dim_t M = brgmm_ctx.get_M();
        const dim_t M_chunks = brgmm_ctx.get_M_chunks();
        const int M_chunk_size = brgmm_ctx.get_M_chunk_size();
        const dim_t M_chunk_tail = brgmm_ctx.get_M_chunk_tail();
        const dim_t N_chunks = brgmm_ctx.get_N_chunks();
        const dim_t N_chunk_tail = brgmm_ctx.get_N_chunk_tail();

        dim_t start {0}, end {0};
        balance211(brgmm_ctx.get_parallel_work_amount_gemm(),
                brgmm_ctx.get_num_threads_for_bmn(), ithr_bmn, start, end);
        dim_t b {0}, mc {0}, nc {0}, b_per_t {0}, mc_per_t {0}, nc_per_t {0},
                bt {0}, mt {0}, nt {0};
        dim_t m_chunks_per_thread = div_up(M_chunks, bgmmc.nthr_m);
        dim_t n_chunks_per_thread = div_up(N_chunks, bgmmc.nthr_n);
        dim_t batch_per_thread = div_up(bgmmc.batch, bgmmc.nthr_b);
        if (brgmm_ctx.is_chunks_horizontal_process_order())
            nd_iterator_init(start, bt, bgmmc.nthr_b, mt, bgmmc.nthr_m, nt,
                    bgmmc.nthr_n, b_per_t, batch_per_thread, mc_per_t,
                    m_chunks_per_thread, nc_per_t, n_chunks_per_thread);
        else
            nd_iterator_init(start, bt, bgmmc.nthr_b, nt, bgmmc.nthr_n, mt,
                    bgmmc.nthr_m, b_per_t, batch_per_thread, nc_per_t,
                    n_chunks_per_thread, mc_per_t, m_chunks_per_thread);
        mc = mt * m_chunks_per_thread + mc_per_t;
        nc = nt * n_chunks_per_thread + nc_per_t;
        b = bt * batch_per_thread + b_per_t;

        auto advance_func = [&]() {
            ++start;
            if (brgmm_ctx.is_chunks_horizontal_process_order())
                nd_iterator_step(bt, bgmmc.nthr_b, mt, bgmmc.nthr_m, nt,
                        bgmmc.nthr_n, b_per_t, batch_per_thread, mc_per_t,
                        m_chunks_per_thread, nc_per_t, n_chunks_per_thread);
            else
                nd_iterator_step(bt, bgmmc.nthr_b, nt, bgmmc.nthr_n, mt,
                        bgmmc.nthr_m, b_per_t, batch_per_thread, nc_per_t,
                        n_chunks_per_thread, mc_per_t, m_chunks_per_thread);
            mc = mt * m_chunks_per_thread + mc_per_t;
            nc = nt * n_chunks_per_thread + nc_per_t;
            b = bt * batch_per_thread + b_per_t;
        };

        assert(bgmmc.batch == 1);

        while (start < end) {
            if (mc >= M_chunks || nc >= N_chunks || b >= bgmmc.batch) {
                advance_func();
                continue;
            }
            auto mb_start_total = mc * M_chunk_size;
            const bool m_chunk_tail = mc == M_chunks - 1 && M_chunk_tail > 0;
            auto mb_end_total = mb_start_total
                    + (m_chunk_tail ? M_chunk_tail : M_chunk_size);
            auto nb_start_total = nc * bgmmc.N_chunk_size;
            const bool n_chunk_tail = nc == N_chunks - 1 && N_chunk_tail > 0;
            auto nb_end_total = nb_start_total
                    + (n_chunk_tail ? N_chunk_tail : bgmmc.N_chunk_size);

            dim_t total_m_work = mb_end_total - mb_start_total;
            dim_t total_n_work = nb_end_total - nb_start_total;
            dim_t mn_start, mn_end;
            balance211(total_m_work * total_n_work, bgmmc.nthr_k, ithr_k,
                    mn_start, mn_end);

            dim_t mb_in_chunk, nb_in_chunk;
            nd_iterator_init(mn_start, mb_in_chunk, total_m_work, nb_in_chunk,
                    total_n_work);
            while (mn_start < mn_end) {
                dim_t mb = mc * M_chunk_size + mb_in_chunk;
                dim_t nb = nc * bgmmc.N_chunk_size + nb_in_chunk;
                const dim_t curr_M_blk = brgmm_ctx.get_M_kernel_size(mb);
                const int m_ker_idx = brgmm_ctx.get_M_kernel_idx(mb);
                const dim_t curr_N_blk = brgmm_ctx.get_N_kernel_size(nb);
                char *buf_reduced_base
                        = brgmm_ctx.get_buf_C_par_reduction_ptr(0, mb, nb);

                const bool c_buf_is_dst_dt
                        = bgmmc.is_c_buf_dst_dt && bgmmc.nthr_k > 1;
                const dim_t c_buf_dt_sz
                        = c_buf_is_dst_dt ? bgmmc.c_dt_sz : bgmmc.acc_dt_sz;
                const size_t m_offset = brgmm_ctx.get_LDC() * c_buf_dt_sz;
                for (int r = 1; r < num_reduction_buffers; r++) {
                    const char *buf_to_reduce_base
                            = brgmm_ctx.get_buf_C_par_reduction_ptr(r, mb, nb);
                    for (int m = 0; m < curr_M_blk; m++) {
                        accumulate(buf_reduced_base + m * m_offset,
                                buf_to_reduce_base + m * m_offset, curr_N_blk);
                    }
                }
                if (bgmmc.post_ops_applicable) {
                    const int n_ker_idx = brgmm_ctx.get_N_kernel_idx(nb);

                    const int brg_ker_idx = pd()->get_brg_kernel_idx(
                            false, false, m_ker_idx, n_ker_idx, false, false);
                    if (brg_ker_idx == -1) {
                        assert(!"Requested brgemm kernel was not created.");
                        return;
                    }

                    const auto brg_kernel = brg_kernels_[brg_ker_idx].get();
                    const dim_t m = brgmm_ctx.get_M_idx(mb);
                    const dim_t n = nb * bgmmc.N_blk;
                    const auto ptr_bias = brgmm_ctx.get_bias_ptr(n);
                    auto ptr_D = brgmm_ctx.get_data_C_ptr(b, m, n);
                    auto ptr_C
                            = brgmm_ctx.get_buf_C_par_reduction_ptr(0, mb, nb);

                    // TODO: support reduction for zp/s8s8 compensations
                    // computed in copy routines
                    const auto zp_comp_a
                            = brgmm_ctx.get_zp_a_compensation_ptr(ithr, b, nb);
                    const auto zp_comp_b
                            = brgmm_ctx.get_zp_b_compensation_result_ptr(
                                    ithr, mb);
                    const auto &post_ops_binary_rhs_arg_vec
                            = brgmm_ctx.get_post_ops_binary_rhs_arg_vec();

                    const size_t dst_row_logical_off
                            = brgmm_ctx.get_M_idx(mb, true);
                    const size_t batch_first_dim_idx = bgmmc.batch_ndims > 1
                            ? b / bgmmc.batch_without_first_dim
                            : 0;
                    const size_t first_mb_matrix_addr_off
                            = batch_first_dim_idx * (M * bgmmc.N)
                            + (m * bgmmc.N + n);
                    // apply post-ops and convert to dst data type only
                    constexpr bool skip_accumulation = true;
                    const char *dst_anchor_point
                            = brgmm_ctx.get_data_C_ptr(0, 0, 0);
                    const brgemm_post_ops_data_t post_ops_data {
                            static_cast<const void *>(ptr_bias),
                            post_ops_binary_rhs_arg_vec.data(),
                            static_cast<size_t>(n), dst_row_logical_off,
                            dst_anchor_point, first_mb_matrix_addr_off,
                            static_cast<const void *>(zp_comp_a),
                            static_cast<const void *>(zp_comp_b),
                            brgmm_ctx.get_zp_c_ptr(), skip_accumulation, 1,
                            false, false, brgmm_ctx.get_src_scales_ptr(b),
                            brgmm_ctx.get_wei_scales_ptr(b, 0, n),
                            brgmm_ctx.get_dst_scales_inv_ptr(ithr)};

                    brgemm_kernel_execute_postops(brg_kernel, 0, nullptr,
                            (void *)ptr_C, (void *)ptr_D, post_ops_data,
                            nullptr, &leading_dimensions);
                }

                nd_iterator_step(
                        mb_in_chunk, total_m_work, nb_in_chunk, total_n_work);
                mn_start++;
            }

            advance_func();
        }
    });
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::copy_a_chunk_in_buffer(
        const brg_matmul_exec_ctx_t &brgmm_ctx, const char *A_data_batch_ptr,
        int ithr, dim_t m_blk_idx, dim_t k_blk_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();

    auto ctx = jit_brgemm_matmul_copy_a_t::ctx_t();
    const dim_t k_start = k_blk_idx * bgmmc.K_blk * bgmmc.brgemm_batch_size;
    const bool is_K_tail
            = brgmm_ctx.is_last_K_blk(k_blk_idx) && bgmmc.K_tail > 0;

    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_blk_idx);
    const dim_t gemm_batch_iters
            = bgmmc.use_buffer_a_tail_only ? 0 : gemm_batch;

    const dim_t m = brgmm_ctx.get_M_idx(m_blk_idx, true);

    ctx.current_M_blk = brgmm_ctx.get_M_kernel_size(m_blk_idx);
    ctx.zp_b_compensation_buffer_ptr
            = (void *)brgmm_ctx.get_zp_b_compensation_buffer_ptr(
                    ithr, m_blk_idx);
    ctx.zp_a_compensation_result_ptr
            = (void *)brgmm_ctx.get_zp_b_compensation_result_ptr(
                    ithr, m_blk_idx);
    ctx.dynamic_src_ld = brgmm_ctx.get_src_stride();

    // Note: instead of passing an address to a stack variable, a kernel may be
    // changed to take just zp_b value and perform negation itself, but updating
    // kernels is not straightforward for all platforms.
    const bool use_legacy_zp_a_path
            = !bgmmc.with_wei_decompression && !bgmmc.with_per_mn_compensation;
    int32_t neg_zp_b = use_legacy_zp_a_path ? brgmm_ctx.get_neg_zp_b() : 0;
    int32_t neg_zp_ab_comp = static_cast<int32_t>(
            use_legacy_zp_a_path ? bgmmc.K * brgmm_ctx.get_neg_zp_a() : 0);
    ctx.zp_b_neg_val_ptr = &neg_zp_b;
    ctx.zp_ab_comp_ptr = &neg_zp_ab_comp;

    for (dim_t gb = 0; gb < gemm_batch_iters; gb++) {
        const dim_t k = k_start + gb * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_A_mk_ptr(A_data_batch_ptr, m, k);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_A_ptr(
                ithr, m_blk_idx, k_blk_idx, gb);
        ctx.current_K_blk = nstl::min(bgmmc.K_blk, bgmmc.K);
        ctx.current_K_start = k;

        (*copy_A_kernel_)(&ctx);
    }
    if (is_K_tail) {
        const auto K_tail = bgmmc.K % bgmmc.K_blk;
        const dim_t k = k_start + gemm_batch * bgmmc.K_blk;
        ctx.src = (void *)brgmm_ctx.get_data_A_mk_ptr(A_data_batch_ptr, m, k);
        ctx.tr_src = (void *)brgmm_ctx.get_buf_A_ptr(
                ithr, m_blk_idx, k_blk_idx, gemm_batch_iters);
        ctx.current_K_blk = K_tail;
        ctx.current_K_start = k;

        (*copy_A_kernel_)(&ctx);
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::copy_b_chunk_in_buffer(
        const brg_matmul_exec_ctx_t &brgmm_ctx, const char *B_data_batch_ptr,
        int ithr, dim_t b_idx, dim_t n_blk_idx, dim_t k_blk_idx) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();

    const dim_t k_start = k_blk_idx * bgmmc.K_blk * bgmmc.brgemm_batch_size;
    const bool is_K_tail
            = brgmm_ctx.is_last_K_blk(k_blk_idx) && bgmmc.K_tail > 0;
    const int gemm_batch = brgmm_ctx.get_brgemm_batch_size(k_blk_idx);

    const dim_t n = brgmm_ctx.get_N_idx(n_blk_idx, true);

    if (brgmm_ctx.packed_sparse_weights()) {
        for (dim_t gb = 0; gb < gemm_batch + is_K_tail; gb++) {
            const dim_t k = k_start + gb * bgmmc.K_blk;
            auto p = jit_avx512_sparse_decompress_kernel_t::call_params_t();
            const char *B_data_ptr
                    = brgmm_ctx.get_data_B_kn_ptr(B_data_batch_ptr, k, n);
            p.src_ptr = (void *)B_data_ptr;
            p.bitmask_ptr
                    = (void *)brgmm_ctx.get_data_B_bitmask_ptr(b_idx, k, n);
            p.dst_ptr = (void *)brgmm_ctx.get_buf_B_ptr(
                    ithr, k_blk_idx, n_blk_idx, gb);
            (*sparse_decompress_kernel_)(&p);
        }
        return;
    }

    auto ctx = jit_brgemm_matmul_copy_b_t::ctx_t();
    ctx.current_N_blk = brgmm_ctx.get_N_kernel_size(n_blk_idx);

    ctx.zp_a_compensation_ptr = (void *)brgmm_ctx.get_zp_a_compensation_ptr(
            ithr, b_idx, n_blk_idx);

    // Note: instead of passing an address to a stack variable, a kernel may be
    // changed to take just zp_a value and perform negation itself, but updating
    // kernels is not straightforward for all platforms.
    int32_t neg_zp_a = brgmm_ctx.get_neg_zp_a();
    ctx.zp_a_neg_value_ptr = &neg_zp_a;

    ctx.compensation_ptr
            = (void *)brgmm_ctx.get_s8s8_comp_ptr(ithr, b_idx, n_blk_idx);

    ctx.dynamic_src_stride = brgmm_ctx.copy_B_wei_stride();

    // For the grouped Zero points/scales need to vary k-block size
    // For this case need to call copy kernel with unaligned (k, k_iters)
    auto call_copy_kernel = [&](dim_t k, dim_t k_iters, dim_t gb,
                                    bool aligned_blocks = false) {
        ctx.src = (void *)brgmm_ctx.get_data_B_kn_ptr(B_data_batch_ptr, k, n);
        // Use k for buffer locating only when the block is unaligned
        if (aligned_blocks)
            ctx.tr_src = (void *)brgmm_ctx.get_buf_B_ptr(
                    ithr, k_blk_idx, n_blk_idx, gb);
        else
            ctx.tr_src = (void *)brgmm_ctx.get_buf_B_k_ptr(ithr, k);

        ctx.current_K_start = k;
        ctx.current_K_iters = k_iters;
        ctx.current_K_pad = brgmm_ctx.get_current_K_pad(k_iters);
        ctx.src_scales_ptr = brgmm_ctx.get_src_scales_ptr(b_idx);
        ctx.wei_scales_ptr = brgmm_ctx.get_wei_scales_ptr(b_idx, k, n);
        ctx.zp_b_value_ptr = brgmm_ctx.get_wei_zp_ptr(b_idx, k, n);
        if (bgmmc.blocked_B && !bgmmc.is_f16_with_int_wei
                && isa == avx512_core_fp16) {
            cvt_float16_to_float((float *)ctx.tr_src, (float16_t *)ctx.src,
                    bgmmc.wei_n_blk * ctx.current_K_iters);
        } else {
            (*copy_B_kernel_)(&ctx);
        }
    };

    // grouped zero points &-or scales
    if (bgmmc.is_wei_zp_per_k || bgmmc.is_wei_scale_per_k) {
        const auto &k_group = bgmmc.is_wei_zp_per_k ? bgmmc.wei_zp_k_gsize
                                                    : bgmmc.wei_scales_k_gsize;
        const auto brgemm_k_blk = nstl::min(bgmmc.K, bgmmc.K_blk);
        const auto adj_k_blk = nstl::min(brgemm_k_blk, k_group);
        assert(adj_k_blk > 0);
        auto k = k_start;
        // is_K_tail behaves incorrectly for the case K < K_blk
        // Should be is_K_tail = true && gemm_batch = 0
        // Now: is_K_tail = false, gemm_batch = 1.
        // Causes Segfault when the `group over k` < K_blk for blocked formats.
        const auto work_amount = bgmmc.K < bgmmc.K_blk
                ? bgmmc.K
                : gemm_batch * bgmmc.K_blk
                        + is_K_tail * (bgmmc.K % bgmmc.K_blk);
        const auto k_end = k_start + work_amount;

        // Handle first block
        if (k_start % adj_k_blk > 0) {
            const auto first_blk_size = adj_k_blk - (k_start % adj_k_blk);
            call_copy_kernel(k_start, first_blk_size, 0);
            k += first_blk_size;
        }
        // Handle full blocks
        for (; (k + adj_k_blk) <= k_end; k += adj_k_blk) {
            const auto gb = (k - k_start) / bgmmc.K_blk;
            call_copy_kernel(k, adj_k_blk, gb);
        }
        // Handle last block
        if (k_end > k) {
            const auto gb = (k - k_start) / bgmmc.K_blk;
            call_copy_kernel(k, k_end - k, gb);
        }
    } else { // Default case with k_blk blocking
        for (dim_t gb = 0; gb < gemm_batch; ++gb) {
            const auto k = k_start + gb * bgmmc.K_blk;
            const auto k_iters = nstl::min(bgmmc.K_blk, bgmmc.K);
            call_copy_kernel(k, k_iters, gb, /*aligned_blocks=*/true);
        }
        if (is_K_tail) {
            const auto k = k_start + gemm_batch * bgmmc.K_blk;
            const auto k_iters = bgmmc.K % bgmmc.K_blk;
            call_copy_kernel(k, k_iters, gemm_batch, /*aligned_blocks=*/true);
        }
    }
}

template <cpu_isa_t isa>
void brgemm_matmul_t<isa>::accumulate(
        char *result_ptr, const char *reduce_ptr, size_t size) const {
    const auto &bgmmc = pd()->get_brgemm_matmul_conf();
    const bool c_buf_is_dst_dt = bgmmc.is_c_buf_dst_dt && bgmmc.nthr_k > 1;
    if (c_buf_is_dst_dt && bgmmc.dst_dt == data_type::bf16)
        acc_ker_bf16_->accumulate(
                (bfloat16_t *)result_ptr, (const bfloat16_t *)reduce_ptr, size);
    else if (c_buf_is_dst_dt && bgmmc.dst_dt == data_type::f16)
        acc_ker_f16_->accumulate(
                (float16_t *)result_ptr, (const float16_t *)reduce_ptr, size);
    else if (bgmmc.acc_dt == f32)
        acc_ker_f32_->accumulate(
                (float *)result_ptr, (const float *)reduce_ptr, size);
    else if (bgmmc.acc_dt == s32)
        acc_ker_s32_->accumulate(
                (int32_t *)result_ptr, (const int32_t *)reduce_ptr, size);
    else
        assert(!"unsupported accumulation data type");
}

template <cpu_isa_t isa>
struct brgemm_matmul_t<isa>::brg_matmul_exec_ctx_t {
    brg_matmul_exec_ctx_t(
            const exec_ctx_t &ctx, const pd_t *pd, matmul_helper_t &helper)
        : bgmmc_(pd->get_brgemm_matmul_conf())
        , src_d_(pd->src_md())
        , wei_d_(pd->weights_md())
        , dst_d_(pd->dst_md())
        , data_A_ptr_(CTX_IN_MEM(const char *, DNNL_ARG_SRC))
        , data_B_ptr_(CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS))
        , data_C_ptr_(CTX_OUT_MEM(char *, DNNL_ARG_DST))
        , data_reduce_ptr_(CTX_OUT_MEM(char *, DNNL_ARG_REDUCE))
        , is_thread_chunks_exec_order_horizontal_(true) {

        const memory_desc_wrapper weights_d(pd->weights_md(0));
        if (bgmmc_.packed_sparse_weights) {
            data_B_offsets_ptr_
                    = CTX_IN_MEM(const int64_t *, DNNL_ARG_WEIGHTS, 1);
            data_B_bitmask_ptr_ = CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS, 2);
            B_packed_sparse_block_size_ = weights_d.blk_size();
        }

        bias_ptr_ = CTX_IN_MEM(const char *, DNNL_ARG_BIAS);

        // setup scales / zp pointers
        src_zp_ptr_ = CTX_IN_MEM(
                const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
        wei_zp_ptr_ = CTX_IN_MEM(
                const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS);
        dst_zp_ptr_ = CTX_IN_MEM(
                const void *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

        const auto &scratchpad = ctx.get_scratchpad_grantor();

        const auto &bgmmc = pd->get_brgemm_matmul_conf();

        src_scales_ = CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
        wei_scales_ = CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
        dst_scales_ = CTX_IN_MEM(
                const float *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);
        dst_scales_inv_ = scratchpad.template get<float>(key_matmul_dst_scales);

        batch_element_ptr_ = scratchpad.template get<brgemm_batch_element_t>(
                key_brgemm_primitive_batch);

        const bool use_buffer_a
                = bgmmc.use_buffer_a || bgmmc.use_buffer_a_tail_only;
        buf_A_ptr_ = (use_buffer_a)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_a)
                : nullptr;

        buf_B_ptr_ = (bgmmc.use_buffer_b)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_b)
                : nullptr;

        buf_C_ptr_ = (bgmmc.use_buffer_c)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
                : nullptr;

        buf_D_ptr_ = (bgmmc.is_runtime_M || bgmmc.is_runtime_N)
                ? scratchpad.template get<char>(key_brgemm_primitive_buffer_d)
                : nullptr;

        buf_reduce_ptr_ = bgmmc.use_buffer_reduce
                ? scratchpad.template get<char>(
                          key_brgemm_primitive_buffer_reduce)
                : nullptr;

        is_amx_ = isa_has_tile_accumulators(isa);
        wsp_tile_ptr_ = is_amx_
                ? ctx.get_scratchpad_grantor().template get<char>(
                          key_conv_amx_tile_buffer)
                : nullptr;

        const dim_t comp_offset = bgmmc_.b_dt_sz
                * (weights_d.size() - weights_d.additional_buffer_size());
        s8s8_compensation_ptr_ = (bgmmc.s8s8_compensation_required)
                ? ((bgmmc.use_buffer_b)
                                  ? scratchpad.template get<int32_t>(
                                            key_brgemm_primitive_buffer_comp)
                                  : const_cast<int32_t *>(
                                            reinterpret_cast<const int32_t *>(
                                                    &data_B_ptr_[comp_offset])))
                : nullptr;
        assert(IMPLICATION(bgmmc.s8s8_compensation_required,
                bgmmc_.b_dt_sz == bgmmc_.tr_b_dt_sz));

        zero_point_a_compensations_ptr_ = bgmmc.has_zero_point_a
                ? scratchpad.template get<int32_t>(
                          key_brgemm_primitive_zp_comp_a)
                : nullptr;
        zero_point_b_compensations_ptr_ = bgmmc.has_zero_point_b
                ? scratchpad.template get<int32_t>(
                          key_brgemm_primitive_zp_comp_b)
                : nullptr;

        per_mn_comp_ptr_ = bgmmc.with_per_mn_compensation
                ? scratchpad.template get<float>(
                          key_brgemm_primitive_per_mn_comp)
                : nullptr;

        post_ops_binary_rhs_arg_vec_ = binary_injector::prepare_binary_args(
                pd->attr()->post_ops_, ctx);
        base_brg_ker_idx_
                = pd->get_brg_kernel_idx(false, true, 0, 0, false, false);
        vnni_factor = data_type_vnni_granularity(bgmmc.wei_dt);

        reorder_zp_a_comp_ptr_ = nullptr;
        if (bgmmc_.has_zero_point_a && bgmmc_.blocked_B) {
            // Store the pointer to computed in reorder compensation values to
            // scale them locally by zp_a value just before usage in post-ops.
            // Using the single global scaling before parallel section might
            // produce significant overhead for small problems running in
            // multitreaded execution mode
            const size_t reorder_zp_a_comp_offset
                    = weights_d.size() - weights_d.additional_buffer_size();
            const size_t b_batch
                    = get_bb_idx(bgmmc.batch - 1, bgmmc_.bcast_B_desc) + 1;
            assert(IMPLICATION(bgmmc.s8s8_compensation_required,
                    !is_runtime_value(bgmmc.s8s8_comp_b_str)));
            const size_t s8s8_buffer_sz = bgmmc.s8s8_compensation_required
                    ? sizeof(int32_t) * b_batch * bgmmc.s8s8_comp_b_str
                    : 0;
            reorder_zp_a_comp_ptr_
                    = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(
                            &data_B_ptr_[reorder_zp_a_comp_offset
                                    + s8s8_buffer_sz]));
        }

        // Set last_brgemm_batch_size_ to brgemm_batch_size
        // when K_tail = 0 and brgemm_batch_tail_size = 0
        last_brgemm_batch_size_ = bgmmc.brgemm_batch_tail_size;
        if (bgmmc.K_tail == 0 && last_brgemm_batch_size_ == 0)
            last_brgemm_batch_size_ = bgmmc.brgemm_batch_size;

        LDD_ = is_runtime_value(bgmmc_.LDD) ? helper.ldc() : bgmmc_.LDD;
        LDC_ = is_runtime_value(bgmmc_.LDC) ? LDD_ : bgmmc_.LDC;
        copy_A_src_stride_ = bgmmc.copy_A_src_stride;
        is_A_batch_layout_trivial_ = bgmmc_.is_src_batch_layout_trivial;
        is_B_batch_layout_trivial_ = bgmmc_.is_wei_batch_layout_trivial;
        is_C_batch_layout_trivial_ = bgmmc_.is_dst_batch_layout_trivial;
        is_thread_chunks_exec_order_horizontal_
                = bgmmc_.is_thread_chunks_exec_order_horizontal;
        K_ = bgmmc.K;
        K_chunks_ = bgmmc.K_chunks;
        K_chunk_tail_ = bgmmc.num_K_blocks % get_K_chunk_size();
        K_chunk_tail_elements_ = K_ % bgmmc.K_chunk_elems;

        const bool avoid_overlap_of_tail_and_non_tail_kernels
                = bgmmc.nthr > 1 && bgmmc.with_sum;

        if (bgmmc.is_runtime_M) {
            M_ = helper.M();
            M_chunks_ = M_ / bgmmc.M_chunk_elems;
            M_chunk_tail_elements_ = M_ % bgmmc.M_chunk_elems;
            dim_t tail = M_chunk_tail_elements_;
            dim_t m_idx = M_ - tail;
            int tail_idx = 0;
            dim_t m_c_buf_idx = 0;
            while (tail > 0) {
                int tail_ker_size = dynamic_m_tails[tail_idx];
                int ker_idx = tail_idx + 1;
                int prev_tail_ker_size = tail_idx > 0
                        ? dynamic_m_tails[tail_idx - 1]
                        : (int)bgmmc.M_blk;
                bool last_tail_kernel = tail_idx == max_num_dynamic_m_tails - 1;
                if (tail > tail_ker_size) {
                    const auto max_ker_size = m_tail_processing_.empty()
                            ? (avoid_overlap_of_tail_and_non_tail_kernels ? tail
                                                                          : M_)
                            : m_tail_processing_.back().kernel_size;
                    if (max_ker_size >= prev_tail_ker_size) {
                        tail_ker_size = prev_tail_ker_size;
                        ker_idx--;
                    }
                } else if (tail < tail_ker_size && !last_tail_kernel) {
                    // skip this tail kernel, try the next one
                    tail_idx++;
                    continue;
                }
                dim_t kernel_m_shift
                        = nstl::max<dim_t>(tail_ker_size - tail, 0);

                m_tail_processing_.push_back({m_idx, ker_idx, tail_ker_size,
                        kernel_m_shift, m_c_buf_idx});
                tail -= tail_ker_size;
                m_idx += tail_ker_size - kernel_m_shift;
                m_c_buf_idx += tail_ker_size;
                if (!last_tail_kernel && tail_ker_size != bgmmc.M_blk)
                    tail_idx++;
            }

            M_tail_block_start_ = M_chunks_ * get_M_chunk_size();
            M_chunk_tail_ = m_tail_processing_.size();
            if (M_chunk_tail_ > 0) M_chunks_++;
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                A_strides_[dim_idx] = bgmmc.a_dt_sz
                        * helper.get_a_stride(bgmmc.ndims - 1 - dim_idx);
            A_ptr_shift_b_ = bgmmc.A_ptr_shift_b;
            if (bgmmc.transposed_A)
                copy_A_src_stride_
                        = helper.get_a_stride(bgmmc.ndims - 1) * bgmmc.a_dt_sz;

            is_A_batch_layout_trivial_ = is_batch_layout_trivial(src_d_);
            is_C_batch_layout_trivial_ = is_batch_layout_trivial(dst_d_);
        } else {
            M_ = bgmmc.M;
            M_chunks_ = bgmmc.M_chunks;
            M_chunk_tail_ = bgmmc.num_M_blocks % get_M_chunk_size();
            M_chunk_tail_elements_ = M_ % bgmmc.M_chunk_elems;
            M_tail_block_start_ = bgmmc.num_M_blocks - (bgmmc.M_tail > 0);
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                A_strides_[dim_idx] = bgmmc.A_strides[dim_idx];
            A_ptr_shift_b_ = bgmmc.A_ptr_shift_b;
        }

        if (bgmmc.is_runtime_N) {
            N_ = helper.N();
            N_chunks_ = N_ / bgmmc.N_chunk_elems;
            N_chunk_tail_elems_ = N_ % bgmmc.N_chunk_elems;
            dim_t tail = N_chunk_tail_elems_;
            dim_t n_idx = N_ - tail;
            int tail_idx = 0;
            dim_t n_c_buf_idx = 0;
            while (tail > 0) {
                int tail_ker_size = dynamic_n_tails[tail_idx];
                int ker_idx = tail_idx + 1;
                int prev_tail_ker_size = tail_idx > 0
                        ? dynamic_n_tails[tail_idx - 1]
                        : (int)bgmmc.N_blk;
                bool last_tail_kernel = tail_idx == max_num_dynamic_n_tails - 1;
                if (tail > tail_ker_size) {
                    const auto max_ker_size = n_tail_processing_.empty()
                            ? (avoid_overlap_of_tail_and_non_tail_kernels ? tail
                                                                          : N_)
                            : n_tail_processing_.back().kernel_size;
                    if (max_ker_size >= prev_tail_ker_size) {
                        tail_ker_size = prev_tail_ker_size;
                        ker_idx--;
                    }
                } else if (tail < tail_ker_size && !last_tail_kernel) {
                    // skip this tail kernel, try the next one
                    tail_idx++;
                    continue;
                }
                dim_t kernel_n_shift
                        = nstl::max<dim_t>(tail_ker_size - tail, 0);

                n_tail_processing_.push_back({n_idx, ker_idx, tail_ker_size,
                        kernel_n_shift, n_c_buf_idx});
                tail -= tail_ker_size;
                n_idx += tail_ker_size - kernel_n_shift;
                n_c_buf_idx += tail_ker_size;
                if (!last_tail_kernel && tail_ker_size != bgmmc.N_blk)
                    tail_idx++;
            }

            N_tail_block_start_ = N_chunks_ * bgmmc.N_chunk_size;
            N_chunk_tail_ = n_tail_processing_.size();
            if (N_chunk_tail_ > 0) N_chunks_++;

            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                B_strides_[dim_idx] = bgmmc.b_dt_sz
                        * helper.get_b_stride(bgmmc.ndims - 1 - dim_idx);

            is_B_batch_layout_trivial_ = is_batch_layout_trivial(wei_d_);
            is_C_batch_layout_trivial_ = is_batch_layout_trivial(dst_d_);
        } else {
            N_ = bgmmc.N;
            N_chunks_ = bgmmc.N_chunks;
            N_chunk_tail_ = bgmmc.num_N_blocks % bgmmc.N_chunk_size;
            N_chunk_tail_elems_ = N_ % bgmmc.N_chunk_elems;
            N_tail_block_start_ = bgmmc.num_N_blocks - (bgmmc.N_tail > 0);
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                B_strides_[dim_idx] = bgmmc.B_strides[dim_idx];
        }

        B_ptr_shift_b_ = bgmmc.B_ptr_shift_b;
        copy_B_wei_stride_ = is_runtime_value(bgmmc_.copy_B_wei_stride)
                ? helper.get_b_stride(bgmmc.ndims - 2) * bgmmc.b_dt_sz
                : bgmmc_.copy_B_wei_stride;
        if (bgmmc.is_runtime_M || bgmmc.is_runtime_N) {
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                C_strides_[dim_idx] = bgmmc.c_dt_sz
                        * helper.get_c_stride(bgmmc.ndims - 1 - dim_idx);
        } else {
            for (int dim_idx = 0; dim_idx < 3; dim_idx++)
                C_strides_[dim_idx] = bgmmc.C_strides[dim_idx];
        }
        C_ptr_shift_b_ = bgmmc_.C_ptr_shift_b;

        // create parallel work of amount that is divisible by nthr_m and nthr_n,
        // the chunks that do not exist will be ignored in gemm execution
        // In case of micro heuristics, nthr_m==nthr_n==nthr_b==1 (round up has no effect)
        dim_t m_chunks_per_thread = rnd_up(M_chunks_, bgmmc.nthr_m);
        dim_t n_chunks_per_thread = rnd_up(N_chunks_, bgmmc.nthr_n);
        dim_t b_per_thread = rnd_up(bgmmc.batch, bgmmc.nthr_b);
        parallel_work_amount_gemm_
                = b_per_thread * m_chunks_per_thread * n_chunks_per_thread;

        // parallelization
        parallel_work_amount_ = bgmmc.batch * M_chunks_ * N_chunks_;

        // The number of threads available during primitive execution may
        // increase (ex. Eigen threadpool implementation) or decrease
        // (ex. nested parallelism) compared to the
        // number of threads available during primitive creation.
        // So we limit the total number of threads to the
        // minimum of these two values to prevent potential OOM issues.
        nthr_ = nstl::min(dnnl_get_current_num_threads(), bgmmc.nthr);

        nthr_k_ = bgmmc.nthr_k > 0 && bgmmc.nthr_k <= nthr_ ? bgmmc.nthr_k : 1;
        nthr_bmn_ = nthr_ / nthr_k_;

        // If parallel_work_amount_ == 1 and parallel reduction is not used, we
        // limit num threads to 1 as parallel(1, ...) does not create parallel
        // section at all. We do not limit number of threads for case
        // 1 < parallel_work_amount_ < dnnl_get_max_threads() to avoid potential
        // overhead on spawning different number of OMP threads from layer to
        // layer.
        // force_seq_execution forces single-threaded execution for performance
        // reasons regardless of parallel_work_amount_
        if ((parallel_work_amount_ == 1 && !parallel_reduction_is_used())
                || bgmmc.force_seq_execution)
            nthr_ = nthr_bmn_ = nthr_k_ = 1;

        // For Eigen threadpool there is significant advantage to not spawn
        // useless threads.
        if (!dnnl_thr_syncable()) {
            nthr_bmn_ = static_cast<int>(
                    nstl::min<dim_t>(nthr_bmn_, parallel_work_amount_));
        }

        num_threads_used_ = nthr_k_ * nthr_bmn_;

        // currently parallel reduction is supported only for case of
        // non-batched problems without computation of any compensations in
        // copy routines
        assert(IMPLICATION(parallel_reduction_is_used(),
                bgmmc.batch == 1
                        && !bgmmc.calculate_compensations_in_copy_routines()));
    }

    // NOTE: gb --> generalized batch, bb --> broadcast batch
    dim_t get_bb_idx(dim_t gb_idx, const brgemm_matmul_bcast_desc_t &bd) const {
        if (!bd.bcast_mask) // no broadcast
            return gb_idx;

        if (bd.bcast_across_all_batch_dims) return 0;

        dim_t gb_off_before_bcast = utils::rnd_dn(
                gb_idx, bd.first_bcast_dim_to_last_batch_dim_prod);
        dim_t bb_idx = gb_off_before_bcast / (bd.bcast_dims_prod);

        dim_t cur_bcast_dims_prod = bd.bcast_dims_prod;
        int mask = 1 << (bgmmc_.batch_ndims - bd.first_bcast_dim - 1);
        for (int d = bd.first_bcast_dim; d < bd.last_bcast_dim; ++d) {
            if (bd.bcast_mask & mask) // broadcast
                cur_bcast_dims_prod /= bd.batch_dims[d];
            else {
                dim_t cur_b = (gb_idx / bd.gb_off[d]) % bd.batch_dims[d];
                bb_idx += cur_b * (bd.gb_off[d] / cur_bcast_dims_prod);
            }
            mask >>= 1;
        }
        bb_idx += gb_idx % bd.gb_off[bd.last_bcast_dim];
        return bb_idx;
    }

    // Note: Minimize the calls to `X_batch_ptr` to reduce overhead.
    // Call it once the batch changes, later use get_data_mk_off
    const char *get_data_A_batch_ptr(dim_t b_idx) const {
        using namespace format_tag;
        const dim_t b = get_bb_idx(b_idx, bgmmc_.bcast_A_desc);
        dim_t b_off = 0;
        if (one_of(bgmmc_.src_tag, acbd, adbc)
                /* this is a special case when src can be represented
                   by plain and transposed tags due to a batch dim equal to 1 */
                || (one_of(bgmmc_.src_tag, abcd, abdc)
                        && bgmmc_.A_ptr_shift_b != 0)) {
            if (!bgmmc_.bcast_A_desc.bcast_mask) { // no broadcast
                const dim_t batch_dim1 = bgmmc_.bcast_A_desc.batch_dims[1];
                b_off = A_strides_[2] * (b % batch_dim1)
                        + (b / batch_dim1) * A_ptr_shift_b_;
            } else {
                b_off = b * A_ptr_shift_b_;
            }
        } else if (is_A_batch_layout_trivial_) {
            b_off = A_strides_[2] * b;
        } else {
            // slow code path
            b_off = src_d_.off_l(b * bgmmc_.M * bgmmc_.K) * bgmmc_.a_dt_sz;
        }
        return data_A_ptr_ + b_off;
    }

    const char *get_data_A_mk_ptr(
            const char *batch_ptr, dim_t m, dim_t k) const {
        return batch_ptr + A_strides_[1] * m + A_strides_[0] * k;
    }

    dim_t get_data_B_k_blk() const {
        if (!bgmmc_.blocked_B) return bgmmc_.wei_k_blk;

        const auto &blocking = wei_d_.blocking_desc();
        const int k_dim = bgmmc_.ndims - 2;
        dim_t k_blk = 1;
        bool has_k_block = false;
        for (int iblk = 0; iblk < blocking.inner_nblks; ++iblk) {
            if (blocking.inner_idxs[iblk] == k_dim) {
                k_blk *= blocking.inner_blks[iblk];
                has_k_block = true;
            }
        }
        return has_k_block ? k_blk : bgmmc_.wei_k_blk;
    }

    dim_t get_data_B_vnni_factor() const {
        if (!bgmmc_.blocked_B) return vnni_factor;
        return get_data_B_k_blk() / get_wei_k_blk(f32);
    }

    dim_t get_data_B_kn_off(dim_t k, dim_t n) const {
        const dim_t wei_k_blk = get_data_B_k_blk();
        const dim_t k_idx = bgmmc_.blocked_B ? k / wei_k_blk : k;
        const dim_t n_idx = bgmmc_.blocked_B ? n / bgmmc_.wei_n_blk : n;
        const int int4_fac = bgmmc_.is_int4_weights ? 2 : 1;
        return (B_strides_[1] * k_idx + B_strides_[0] * n_idx
                       + get_data_B_off_within_block(k, n))
                / int4_fac;
    }

    const char *get_data_B_kn_ptr(
            const char *batch_ptr, dim_t k, dim_t n) const {
        const char *b_ptr = batch_ptr + get_data_B_kn_off(k, n);
        if (bgmmc_.packed_sparse_weights) {
            const dim_t blk_num
                    = (b_ptr - data_B_ptr_) / B_packed_sparse_block_size_;
            const auto blk_off = data_B_offsets_ptr_[blk_num];
            return data_B_ptr_ + blk_off;
        }
        return b_ptr;
    }

    dim_t get_data_B_batch_off(dim_t b) const {
        using namespace format_tag;
        dim_t b_off = 0;
        if (one_of(bgmmc_.wei_tag, acbd, adbc)
                /* this is a special case when weights can be represented
                   by plain and transposed tags due to a batch dim equal to 1 */
                || (one_of(bgmmc_.wei_tag, abcd, abdc)
                        && bgmmc_.B_ptr_shift_b != 0)) {
            if (!bgmmc_.bcast_B_desc.bcast_mask) { // no broadcast
                const dim_t batch_dim1 = bgmmc_.bcast_B_desc.batch_dims[1];
                b_off = B_strides_[2] * (b % batch_dim1)
                        + (b / batch_dim1) * B_ptr_shift_b_;
            } else {
                b_off = b * B_ptr_shift_b_;
            }
        } else if (is_B_batch_layout_trivial_) {
            b_off = B_strides_[2] * b;
        } else {
            b_off = wei_d_.off_l(b * bgmmc_.K * bgmmc_.N) * bgmmc_.b_dt_sz;
        }
        if (bgmmc_.is_int4_weights) b_off = b_off / 2;
        return b_off;
    }

    const char *get_data_B_batch_ptr(dim_t b_idx) const {
        const dim_t b = get_bb_idx(b_idx, bgmmc_.bcast_B_desc);
        return data_B_ptr_ + get_data_B_batch_off(b);
    }

    const char *get_data_B_bitmask_ptr(dim_t b, dim_t k, dim_t n) const {
        assert(bgmmc_.packed_sparse_weights);
        const dim_t cur_data_B_off
                = get_data_B_batch_off(b) + get_data_B_kn_off(k, n);
        const auto bitmask_off = cur_data_B_off / CHAR_BIT;
        return data_B_bitmask_ptr_ + bitmask_off;
    }

    char *get_data_C_ptr(dim_t b, dim_t m, dim_t n) const {
        return data_C_ptr_ + get_data_C_off(b, m, n);
    }

    brgemm_batch_element_t *get_batch_elem_ptr(int ithr) const {
        return batch_element_ptr_
                + ithr * bgmmc_.brgemm_batch_element_per_thr_sz;
    }

    void init_brgemm_batch_elements_values(int ithr, dim_t brg_batch_start,
            dim_t brg_batch_iters, const char *A_data_batch_ptr,
            const char *B_data_batch_ptr, dim_t b_idx, dim_t m_blk_idx,
            dim_t k_blk_idx, dim_t n_blk_idx) const {
        auto addr_batch = get_batch_elem_ptr(ithr);

        const dim_t m = get_M_idx(m_blk_idx, true);
        const dim_t n = n_blk_idx * bgmmc_.N_blk;

        for (int b_iter = 0; b_iter < brg_batch_iters; b_iter++) {
            const dim_t brg_batch_idx = brg_batch_start + b_iter;
            const dim_t k = k_blk_idx * bgmmc_.K_blk * bgmmc_.brgemm_batch_size
                    + brg_batch_idx * bgmmc_.K_blk;
            addr_batch[b_iter].ptr.A = bgmmc_.use_buffer_a
                    ? get_buf_A_ptr(ithr, m_blk_idx, k_blk_idx, brg_batch_idx)
                    : get_data_A_mk_ptr(A_data_batch_ptr, m, k);
            addr_batch[b_iter].ptr.B = (bgmmc_.use_buffer_b)
                    ? get_buf_B_ptr(ithr, k_blk_idx, n_blk_idx, brg_batch_idx)
                    : get_data_B_kn_ptr(B_data_batch_ptr, k, n);

            if (bgmmc_.gemv_swap_a_b)
                std::swap(addr_batch[b_iter].ptr.A, addr_batch[b_iter].ptr.B);
        }
    }

    char *get_buf_A_ptr(
            int ithr, dim_t m_blk_idx, dim_t k_blk_idx, dim_t gb) const {
        if (!bgmmc_.use_buffer_a && !bgmmc_.use_buffer_a_tail_only)
            return nullptr;

        dim_t k_blk_local = bgmmc_.use_buffer_a_tail_only ? 0 : k_blk_idx;
        k_blk_local = k_blk_local % get_K_chunk_size();
        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const dim_t tail_idx = get_M_tail_block_idx(m_blk_idx);
            const dim_t curr_m_block_size
                    = m_tail_processing_[tail_idx].kernel_size;
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[tail_idx].buf_dim_idx;
            const dim_t ld = bgmmc_.tr_a_dt_sz
                    * (bgmmc_.use_buffer_a_tail_only ? bgmmc_.wei_k_blk
                                                     : bgmmc_.LDA);
            const dim_t batch = bgmmc_.use_buffer_a_tail_only
                    ? 1
                    : bgmmc_.brgemm_batch_size;
            const dim_t offset = ithr * bgmmc_.buffer_a_per_thread_sz
                    + curr_m_buf_shift * ld * batch * bgmmc_.K_chunk_size
                    + k_blk_local * batch * ld * curr_m_block_size
                    + gb * ld * curr_m_block_size;
            return buf_A_ptr_ + offset;
        }

        const dim_t m_blk_local = m_blk_idx % get_M_chunk_size();
        return buf_A_ptr_ + ithr * bgmmc_.buffer_a_per_thread_sz
                + m_blk_local * bgmmc_.buffer_a_m_stride
                + k_blk_local * bgmmc_.buffer_a_k_stride
                + gb * bgmmc_.buffer_a_gb_stride;
    }

    // Returns a pointer to a correspondent place in blocked matrix B.
    // The pointer position depends on the following arguments:
    //   `ithr` defines the offset per thread,
    //   `k_blk_idx` defines the offset between portions of blocks over K inside
    //     a block of memory per thread.
    //   `gb` defines an offset to a specific block over K inside a portion of
    //     blocks.
    char *get_buf_B_ptr(
            int ithr, dim_t k_blk_idx, dim_t n_blk_idx, dim_t gb) const {
        UNUSED(n_blk_idx);
        if (!bgmmc_.use_buffer_b) return nullptr;
        dim_t k_blk_local = k_blk_idx % get_K_chunk_size();
        const auto offset = ithr * bgmmc_.buffer_b_per_thread_sz
                + k_blk_local * bgmmc_.buffer_b_k_brg_stride
                + gb * bgmmc_.buffer_b_gb_stride;
        return buf_B_ptr_ + offset;
    }

    /* Returns a pointer to buffer B based on unaligned K inside
    *  a thread buffer. Used for copy kernels including grouped ZP/Scales.
    *  For the vnni granularity > 1 it returns a pointer to a start of vnni block.
    *  Functionality intersects with get_buf_B_ptr(). TODO: make combined solution.
    */
    char *get_buf_B_k_ptr(const int ithr, const dim_t k) const {
        if (!bgmmc_.use_buffer_b) return nullptr;

        const dim_t batch_block_size = bgmmc_.K_blk * bgmmc_.brgemm_batch_size;
        const auto batch_blocking = std::div(k, batch_block_size);
        const auto k_blk_idx = batch_blocking.quot;
        const auto k_blk_local = k_blk_idx % get_K_chunk_size();
        const auto k_in_batch = batch_blocking.rem;

        auto offset = ithr * bgmmc_.buffer_b_per_thread_sz;
        offset += k_blk_local * bgmmc_.buffer_b_k_brg_stride;
        // div down to the start of the vnni block
        const auto k_outer = (k_in_batch / vnni_factor) * vnni_factor;
        offset += k_outer * bgmmc_.buffer_b_k_stride;
        return buf_B_ptr_ + offset;
    }

    char *get_buf_C_ptr(int ithr, dim_t m_blk_idx, dim_t n_blk_idx) const {
        if (!bgmmc_.use_buffer_c) return nullptr;

        if (bgmmc_.nthr_k > 1) {
            const int ithr_k = get_thread_idx_for_k(ithr);
            return get_buf_C_par_reduction_ptr(ithr_k, m_blk_idx, n_blk_idx);
        }
        char *buf_C_ptr_local
                = buf_C_ptr_ + ithr * bgmmc_.buffer_c_per_thread_sz;

        dim_t n_blk_local = 0;
        dim_t m_blk_local = 0;

        if (bgmmc_.is_runtime_N || bgmmc_.is_runtime_M
                || bgmmc_.K_chunk_elems < bgmmc_.K) {
            n_blk_local = n_blk_idx % bgmmc_.N_chunk_size;
            m_blk_local = m_blk_idx % get_M_chunk_size();
        }

        const bool runtime_M_tail = is_runtime_M_tail_chunk(m_blk_idx);
        const bool runtime_N_tail = is_runtime_N_tail_chunk(n_blk_idx);

        if (runtime_M_tail || runtime_N_tail) {
            const dim_t curr_m_block_size = get_M_kernel_size(m_blk_idx);
            const dim_t curr_m_buf_shift = runtime_M_tail
                    ? m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                              .buf_dim_idx
                    : m_blk_local;
            const dim_t curr_n_buf_shift = runtime_N_tail
                    ? n_tail_processing_[get_N_tail_block_idx(n_blk_idx)]
                              .buf_dim_idx
                    : n_blk_local;
            const dim_t m_elems_shift = curr_m_buf_shift * bgmmc_.N_chunk_elems;
            const dim_t n_elems_shift = curr_n_buf_shift
                    * (bgmmc_.is_runtime_N ? 1
                                           : curr_m_block_size * bgmmc_.N_blk);

            const dim_t offset
                    = bgmmc_.acc_dt_sz * (m_elems_shift + n_elems_shift);
            return buf_C_ptr_local + offset;
        }
        const dim_t m_shift
                = bgmmc_.N_chunk_size * m_blk_local * bgmmc_.buffer_c_chunk_sz;
        const dim_t n_shift = n_blk_local
                * (bgmmc_.is_runtime_N ? bgmmc_.acc_dt_sz * bgmmc_.N_blk
                                       : bgmmc_.buffer_c_chunk_sz);
        return buf_C_ptr_local + m_shift + n_shift;
    }

    char *get_buf_C_par_reduction_ptr(
            int ithr_k, dim_t m_blk_idx, dim_t n_blk_idx) const {
        if (bgmmc_.nthr_k <= 1) return nullptr;

        const dim_t m = m_blk_idx * bgmmc_.M_blk;
        const dim_t n = n_blk_idx * bgmmc_.N_blk;

        if (!bgmmc_.post_ops_applicable && ithr_k == 0)
            return get_data_C_ptr(0, m, n);

        int k_buf_idx = ithr_k - (!bgmmc_.post_ops_applicable ? 1 : 0);
        const dim_t c_buf_dt_sz = (bgmmc_.is_c_buf_dst_dt && bgmmc_.nthr_k > 1)
                ? bgmmc_.c_dt_sz
                : bgmmc_.acc_dt_sz;
        return buf_C_ptr_ + k_buf_idx * bgmmc_.buffer_c_per_thread_sz
                + get_data_C_off(0, m, n) * c_buf_dt_sz / bgmmc_.c_dt_sz;
    }

    dim_t get_data_B_off_within_block(dim_t k, dim_t n) const {
        using namespace format_tag;

        if (!bgmmc_.blocked_B) return 0;

        const dim_t wei_k_blk = get_data_B_k_blk();
        dim_t x0 = k % wei_k_blk;
        dim_t x1 = n % bgmmc_.wei_n_blk;
        const dim_t b_vnni_factor = get_data_B_vnni_factor();
        dim_t offset = (x0 / b_vnni_factor) * b_vnni_factor * bgmmc_.wei_n_blk
                + x1 * b_vnni_factor + x0 % b_vnni_factor;
        return bgmmc_.b_dt_sz * offset;
    }

    dim_t get_data_C_off(dim_t b, dim_t m, dim_t n) const {
        using namespace format_tag;
        assert(bgmmc_.dst_tag != adbc);
        dim_t off = 0;
        if (bgmmc_.dst_tag == acbd
                || (one_of(bgmmc_.dst_tag, abcd, abdc)
                        && bgmmc_.C_ptr_shift_b != 0)) {
            const dim_t batch_dim1 = bgmmc_.bcast_A_desc.batch_dims[1];
            dim_t b_off = C_strides_[2] * (b % batch_dim1)
                    + (b / batch_dim1) * C_ptr_shift_b_;
            off = b_off + C_strides_[1] * m + C_strides_[0] * n;
        } else if (is_C_batch_layout_trivial_) {
            off = C_strides_[2] * b + C_strides_[1] * m + C_strides_[0] * n;
        } else {
            // slow code
            off = dst_d_.off_l(b * bgmmc_.M * bgmmc_.N) * bgmmc_.c_dt_sz
                    + C_strides_[1] * m + C_strides_[0] * n;
        }
        return off;
    }

    // Returns a pointer to the user-provided reduce buffer, shifted by
    // the specified offset @p off.
    char *get_data_reduce_ptr(dim_t off) const {
        if (!bgmmc_.with_reduce) return nullptr;
        return data_reduce_ptr_ + off * bgmmc_.reduce_dt_sz;
    }

    // Returns a pointer to the scratchpad reduce buffer for the
    // corresponding @p ithr, shifted by the specified offset @p off.
    char *get_buf_reduce_ptr(int ithr, dim_t off) const {
        if (!bgmmc_.with_reduce) return nullptr;
        assert(bgmmc_.acc_dt == f32);
        const int ithr_k = get_thread_idx_for_k(ithr);
        // Use the user-provided reduce buffer as one of the reduce buffers.
        const bool is_reduce_f32 = bgmmc_.reduce_dt == f32;
        if (is_reduce_f32 && ithr_k == 0) return get_data_reduce_ptr(off);

        return buf_reduce_ptr_
                + (ithr_k - is_reduce_f32) * bgmmc_.buffer_reduce_per_thread_sz
                + off * bgmmc_.acc_dt_sz;
    }

    // Returns a pointer to the scratchpad reduce buffer for the
    // corresponding index @p ibuf, shifted by the specified offset @p off.
    char *get_buf_reduce_ptr_by_index(int ibuf, dim_t off) const {
        if (!bgmmc_.with_reduce) return nullptr;
        const size_t _off = bgmmc_.M * ibuf + off;
        return buf_reduce_ptr_ + _off * bgmmc_.acc_dt_sz;
    }

    const char *get_bias_ptr(dim_t n) const {
        if (!bgmmc_.with_bias) return nullptr;

        return bias_ptr_ + n * bgmmc_.bias_dt_sz;
    }

    int32_t *get_s8s8_comp_ptr(int ithr, dim_t b, dim_t n_blk_idx) const {
        if (!bgmmc_.s8s8_compensation_required) return nullptr;

        const dim_t n_blk_local = bgmmc_.use_buffer_b
                ? n_blk_idx % bgmmc_.N_chunk_size
                : n_blk_idx;
        assert(!is_runtime_value(bgmmc_.s8s8_comp_b_str));
        return s8s8_compensation_ptr_ + ithr * bgmmc_.s8s8_comp_ithr_str
                + get_bb_idx(b, bgmmc_.bcast_B_desc) * bgmmc_.s8s8_comp_b_str
                + n_blk_local * bgmmc_.s8s8_comp_n_str;
    }

    const void *get_src_scales_ptr(
            dim_t b_idx = 0, dim_t k = 0, dim_t m = 0) const {
        if (!bgmmc_.is_src_scale_per_k && !bgmmc_.is_src_scale_per_m) {
            // Pure per-batch addressing (e.g. 3D/4D per_tensor with batch
            // bits in the scales mask).
            if (bgmmc_.src_scales_batch_stride > 0) {
                const dim_t b = get_bb_idx(b_idx, bgmmc_.bcast_A_desc);
                const auto offset = b * bgmmc_.src_scales_batch_stride
                        * bgmmc_.src_scales_dt_sz;
                return (const char *)src_scales_ + offset;
            }
            return src_scales_;
        }

        const auto k_group_sz = bgmmc_.is_src_scale_per_k
                ? bgmmc_.src_scales_k_gsize
                : bgmmc_.K;
        const auto k_idx
                = bgmmc_.is_src_scale_per_k ? (k / k_group_sz) : dim_t {0};
        const auto num_k_groups = bgmmc_.is_src_scale_per_k
                ? utils::div_up(bgmmc_.K, k_group_sz)
                : dim_t {1};
        const auto m_idx = bgmmc_.is_src_scale_per_m ? m : dim_t {0};
        auto offset = m_idx * num_k_groups + k_idx;
        if (bgmmc_.src_scales_batch_stride > 0)
            offset += get_bb_idx(b_idx, bgmmc_.bcast_A_desc)
                    * bgmmc_.src_scales_batch_stride;
        offset *= bgmmc_.src_scales_dt_sz;
        return ((const char *)src_scales_ + offset);
    }

    // Returns a pointer to the weights scales for the correspondent block based
    // on @p b_idx, @p k and @p n.
    const void *get_wei_scales_ptr(
            dim_t b_idx = 0, dim_t k = 0, dim_t n = 0) const {
        dim_t offset = 0;
        if (bgmmc_.is_wei_scale_per_n) offset += n;
        if (bgmmc_.is_wei_scale_per_k) {
            const auto &k_group_sz = bgmmc_.wei_scales_k_gsize;
            const auto k_idx = k / k_group_sz;
            offset += k_idx * bgmmc_.N;
        }

        if (bgmmc_.wei_scales_batch_stride > 0) {
            offset += get_bb_idx(b_idx, bgmmc_.bcast_B_desc)
                    * bgmmc_.wei_scales_batch_stride;
        }

        offset = offset * bgmmc_.wei_scales_dt_sz;
        return ((char *)wei_scales_ + offset);
    }

    const void *get_dst_scales_ptr() const { return dst_scales_; }

    // Since `dst_scales_inv_` is a scratchpad memory, @p ithr points to the
    // correspondent piece of that memory.
    const void *get_dst_scales_inv_ptr(int ithr) const {
        if (!bgmmc_.with_dst_scales) return nullptr;

        return reinterpret_cast<const char *const>(dst_scales_inv_)
                + ithr * sizeof(float);
    }

    int32_t get_neg_zp_a() const {
        if (!bgmmc_.has_zero_point_a) return 0;
        return -cpu::io::load_int_value(bgmmc_.src_zp_dt, src_zp_ptr_, 0);
    }

    // Used to compute compensation. Can't initialize the value at construction
    // time as memory buffers must be accessed inside a parallel task
    // (asynchronous runtime requirement).
    int32_t get_neg_zp_b() const {
        if (!bgmmc_.has_zero_point_b) return 0;
        assert(bgmmc_.is_wei_zp_common);
        return -cpu::io::load_int_value(bgmmc_.wei_zp_dt, wei_zp_ptr_, 0);
    }

    const void *get_src_zp_ptr(dim_t b_idx = 0, dim_t k = 0, dim_t m = 0,
            int *nibble_idx = nullptr) const {
        if (!bgmmc_.has_zero_point_a) return nullptr;

        dim_t elem_idx = 0;
        if (bgmmc_.is_src_zp_per_k || bgmmc_.is_src_zp_per_m) {
            const auto k_group_sz
                    = bgmmc_.is_src_zp_per_k ? bgmmc_.src_zp_k_gsize : bgmmc_.K;
            const auto k_idx
                    = bgmmc_.is_src_zp_per_k ? (k / k_group_sz) : dim_t {0};
            const auto num_k_groups = bgmmc_.is_src_zp_per_k
                    ? utils::div_up(bgmmc_.K, k_group_sz)
                    : dim_t {1};
            const auto m_idx = bgmmc_.is_src_zp_per_m ? m : dim_t {0};
            elem_idx = m_idx * num_k_groups + k_idx;
        }
        if (bgmmc_.src_zp_batch_stride > 0)
            elem_idx += get_bb_idx(b_idx, bgmmc_.bcast_A_desc)
                    * bgmmc_.src_zp_batch_stride;

        const auto dt_sz = types::data_type_size(bgmmc_.src_zp_dt);
        const auto elems_per_byte
                = one_of(bgmmc_.src_zp_dt, data_type::s4, data_type::u4) ? 2
                                                                         : 1;
        // Expose nibble parity so the caller can pass the correct element
        // index to load_float_value() for sub-byte (s4/u4) types.
        if (nibble_idx)
            *nibble_idx = static_cast<int>(elem_idx % elems_per_byte);
        const auto byte_off = elem_idx * dt_sz / elems_per_byte;
        return (const char *)src_zp_ptr_ + byte_off;
    }

    const void *get_wei_zp_ptr(
            dim_t b_idx = 0, dim_t k = 0, dim_t n = 0) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;
        if (bgmmc_.is_wei_zp_common)
            return wei_zp_ptr_; // single zero point value
        // Locate the group based on (b_idx, k, n)
        dim_t offset = 0;
        if (bgmmc_.is_wei_zp_per_n) offset += n;

        if (bgmmc_.is_wei_zp_per_k) {
            const auto &k_group_sz = bgmmc_.wei_zp_k_gsize;
            const auto k_idx = k / k_group_sz;
            const dim_t n_factor = bgmmc_.is_wei_zp_per_n ? bgmmc_.N : 1;
            offset += k_idx * n_factor;
        }

        if (bgmmc_.wei_zp_batch_stride > 0) {
            offset += get_bb_idx(b_idx, bgmmc_.bcast_B_desc)
                    * bgmmc_.wei_zp_batch_stride;
        }

        const auto dt_sz = types::data_type_size(bgmmc_.wei_zp_dt);
        const auto elems_per_byte
                = one_of(bgmmc_.wei_zp_dt, data_type::s4, data_type::u4) ? 2
                                                                         : 1;
        offset = offset * dt_sz / elems_per_byte;
        return (char *)wei_zp_ptr_ + offset;
    }

    const void *get_zp_c_ptr() const { return dst_zp_ptr_; }

    // Returns the per-thread f32 tile (M_blk × LDC elements) used by the
    // per-(M, N) compensation kernel hook. Caller fills it just before the
    // final K-block brgemm call.
    float *get_per_mn_comp_ptr(int ithr) const {
        if (!bgmmc_.with_per_mn_compensation) return nullptr;
        // Per-thread tile mirrors buffer_c's stripe layout for AMX:
        // ceil(N_blk / LDC) stripes of M_blk × LDC floats.
        const size_t per_thr_elems = static_cast<size_t>(bgmmc_.M_blk)
                * static_cast<size_t>(rnd_up(bgmmc_.N_blk, bgmmc_.LDC));
        return per_mn_comp_ptr_ + ithr * per_thr_elems;
    }

    int32_t *get_zp_a_compensation_ptr(
            int ithr, dim_t b_idx, dim_t n_blk_idx) const {
        if (!bgmmc_.has_zero_point_a) return nullptr;

        const dim_t n_blk_local = n_blk_idx % bgmmc_.N_chunk_size;
        int32_t *zp_comp = zero_point_a_compensations_ptr_
                + ithr * bgmmc_.zp_a_comp_elems_per_thr
                + n_blk_local * bgmmc_.zp_a_comp_shift_n;

        if (bgmmc_.blocked_B) {
            // Scale computed in reorder compensation values by zp_a value
            // locally just before usage. Using the single global scaling before
            // parallel section might produce significant overhead for small
            // problems running in multitreaded execution mode
            const dim_t base_offset = get_bb_idx(b_idx, bgmmc_.bcast_B_desc)
                            * rnd_up(bgmmc_.N, bgmmc_.wei_n_blk)
                    + n_blk_idx * bgmmc_.N_blk;

            const dim_t curr_N_blk = get_N_kernel_size(n_blk_idx);
            PRAGMA_OMP_SIMD()
            for (dim_t b = 0; b < curr_N_blk; b++)
                zp_comp[b] = -get_neg_zp_a()
                        * reorder_zp_a_comp_ptr_[base_offset + b];
        }
        return zp_comp;
    }

    int32_t *get_zp_b_compensation_result_ptr(int ithr, dim_t m_blk_idx) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;

        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                              .buf_dim_idx;
            return zero_point_b_compensations_ptr_
                    + ithr * bgmmc_.zp_b_comp_elems_per_thr + curr_m_buf_shift;
        }

        const dim_t m_blk_local = m_blk_idx % get_M_chunk_size();
        return zero_point_b_compensations_ptr_
                + ithr * bgmmc_.zp_b_comp_elems_per_thr
                + m_blk_local * bgmmc_.zp_b_comp_result_shift_m;
    }

    int32_t *get_zp_b_compensation_buffer_ptr(int ithr, dim_t m_blk_idx) const {
        if (!bgmmc_.has_zero_point_b) return nullptr;

        if (is_runtime_M_tail_chunk(m_blk_idx)) {
            const dim_t curr_m_buf_shift
                    = m_tail_processing_[get_M_tail_block_idx(m_blk_idx)]
                              .buf_dim_idx;
            return get_zp_b_compensation_result_ptr(ithr, 0)
                    + bgmmc_.zp_b_comp_buffer_start + curr_m_buf_shift;
        }

        const dim_t m_blk_local = m_blk_idx % get_M_chunk_size();
        return get_zp_b_compensation_result_ptr(ithr, 0)
                + bgmmc_.zp_b_comp_buffer_start
                + m_blk_local * bgmmc_.zp_b_comp_buffer_shift_m;
    }

    char *get_tile_workspace(int ithr) const {
        return is_amx_ ? wsp_tile_ptr_ + ithr * bgmmc_.wsp_tile_per_thr_bytes
                       : nullptr;
    }

    const std::vector<const void *> &get_post_ops_binary_rhs_arg_vec() const {
        return post_ops_binary_rhs_arg_vec_;
    }

    int get_base_brgemm_kernel_idx() const { return base_brg_ker_idx_; }

    bool is_last_K_blk(dim_t k_blk_idx) const {
        return k_blk_idx == bgmmc_.num_K_blocks - 1;
    }

    int get_brgemm_batch_size(dim_t k_chunk_idx) const {
        return is_last_K_blk(k_chunk_idx) ? last_brgemm_batch_size_
                                          : bgmmc_.brgemm_batch_size;
    }

    dim_t get_parallel_work_amount() const { return parallel_work_amount_; }
    dim_t get_parallel_work_amount_gemm() const {
        return parallel_work_amount_gemm_;
    }
    int get_num_threads_for_k() const { return nthr_k_; }
    bool parallel_reduction_is_used() const {
        return nthr_k_ > 1 && bgmmc_.K_chunks > 1;
    }
    int get_num_threads_for_bmn() const { return nthr_bmn_; }
    // ithr = ithr_k * nthr_bmn + ithr_bmn
    int get_thread_idx_for_k(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_k = ithr / nthr_bmn_;
        return ithr_k < bgmmc_.K_chunks ? ithr_k : -1;
    }

    int get_thread_idx_for_bmn_gemm(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_bmn = ithr % nthr_bmn_;
        return ithr_bmn < parallel_work_amount_gemm_ ? ithr_bmn : -1;
    }

    int get_thread_idx_for_bmn(int ithr) const {
        if (ithr >= num_threads_used_) return -1;
        const int ithr_bmn = ithr % nthr_bmn_;
        return ithr_bmn < parallel_work_amount_ ? ithr_bmn : -1;
    }

    int get_num_threads_for_parallelization() const {
        return num_threads_used_;
    }
    dim_t get_M() const { return M_; }
    dim_t get_M_chunks() const { return M_chunks_; }
    int get_M_chunk_size() const { return bgmmc_.M_chunk_size; }
    dim_t get_M_chunk_tail() const { return M_chunk_tail_; }

    dim_t get_K_chunks() const { return K_chunks_; }
    int get_K_chunk_size() const { return bgmmc_.K_chunk_size; }
    dim_t get_K_chunk_tail() const { return K_chunk_tail_; }

    int get_M_kernel_idx(dim_t m_block_idx) const {
        if (!is_M_tail_processing(m_block_idx))
            return 0;
        else if (!bgmmc_.is_runtime_M)
            return 1;

        assert(is_runtime_M_tail_chunk(m_block_idx)
                && !m_tail_processing_.empty());
        return m_tail_processing_[get_M_tail_block_idx(m_block_idx)].kernel_idx;
    }

    dim_t get_M_kernel_size(dim_t m_block_idx) const {
        if (!is_M_tail_processing(m_block_idx))
            return bgmmc_.M_blk;
        else if (!bgmmc_.is_runtime_M)
            return bgmmc_.M_tail;

        assert(is_runtime_M_tail_chunk(m_block_idx)
                && !m_tail_processing_.empty());
        return m_tail_processing_[get_M_tail_block_idx(m_block_idx)]
                .kernel_size;
    }

    dim_t get_M_idx(
            dim_t m_block_idx, bool adjust_for_kernel_overlap = false) const {
        if (is_runtime_M_tail_chunk(m_block_idx)) {
            const dim_t tail_idx = get_M_tail_block_idx(m_block_idx);
            const dim_t shift = adjust_for_kernel_overlap
                    ? m_tail_processing_[tail_idx].shift
                    : 0;
            return m_tail_processing_[tail_idx].idx - shift;
        }
        return m_block_idx * bgmmc_.M_blk;
    }

    dim_t get_N() const { return N_; }
    dim_t get_N_chunks() const { return N_chunks_; }
    dim_t get_N_chunk_tail() const { return N_chunk_tail_; }
    dim_t get_N_chunk_tail_elems() const { return N_chunk_tail_elems_; }

    int get_N_kernel_idx(dim_t n_block_idx) const {
        if (!is_N_tail_processing(n_block_idx))
            return 0;
        else if (!bgmmc_.is_runtime_N)
            return 1;

        assert(is_runtime_N_tail_chunk(n_block_idx)
                && !n_tail_processing_.empty());
        return n_tail_processing_[get_N_tail_block_idx(n_block_idx)].kernel_idx;
    }

    dim_t get_N_kernel_size(dim_t n_block_idx) const {
        if (!is_N_tail_processing(n_block_idx))
            return bgmmc_.N_blk;
        else if (!bgmmc_.is_runtime_N)
            return bgmmc_.N_tail;

        assert(is_runtime_N_tail_chunk(n_block_idx)
                && !n_tail_processing_.empty());
        return n_tail_processing_[get_N_tail_block_idx(n_block_idx)]
                .kernel_size;
    }

    dim_t get_N_idx(
            dim_t n_block_idx, bool adjust_for_kernel_overlap = false) const {
        if (is_runtime_N_tail_chunk(n_block_idx)) {
            const dim_t tail_idx = get_N_tail_block_idx(n_block_idx);
            const dim_t shift = adjust_for_kernel_overlap
                    ? n_tail_processing_[tail_idx].shift
                    : 0;
            return n_tail_processing_[tail_idx].idx - shift;
        }
        return n_block_idx * bgmmc_.N_blk;
    }

    dim_t get_src_stride() const { return copy_A_src_stride_; }

    // For tail processing in the case of dynamic dimensions, it's possible to
    // have kernels overlap on the dst tensor when two different kernels
    // compute dst values for the same area. We need to backup/restore values
    // for overlapped area to avoid correctness issues.
    void maybe_backup_dst_values_to_buffer(
            int ithr, dim_t b_idx, dim_t m_blk_idx, dim_t n_blk_idx) const {
        if (!copy_d_required(m_blk_idx, n_blk_idx)) return;

        const bool m_tail_overlapping = is_m_tail_overlap(m_blk_idx);
        dim_t m_start = m_tail_overlapping ? get_M_idx(m_blk_idx, true)
                                           : get_M_idx(m_blk_idx);
        const dim_t rows_to_copy = m_tail_overlapping
                ? m_tail_processing_[get_M_tail_block_idx(m_blk_idx)].shift
                : get_M_kernel_size(m_blk_idx);

        const bool n_tail_overlapping = is_n_tail_overlap(n_blk_idx);
        dim_t n_start = n_tail_overlapping ? get_N_idx(n_blk_idx, true)
                                           : get_N_idx(n_blk_idx);
        const dim_t row_elems = n_tail_overlapping
                ? n_tail_processing_[get_N_tail_block_idx(n_blk_idx)].shift
                : get_N_kernel_size(n_blk_idx);
        const dim_t bytes_to_copy = bgmmc_.c_dt_sz * row_elems;
        assert(!(n_tail_overlapping && m_tail_overlapping)
                && "dynamic tail processing for both M/N is not supported");

        auto copy_from = get_data_C_ptr(b_idx, m_start, n_start);
        auto copy_to = get_buf_D_ptr(ithr);
        const dim_t dst_ld = get_LDD() * bgmmc_.c_dt_sz;
        const dim_t buf_ld = bgmmc_.N_blk * bgmmc_.c_dt_sz;
        for (int r = 0; r < rows_to_copy; r++) {
            utils::array_copy(copy_to, copy_from, bytes_to_copy);
            copy_from += dst_ld;
            copy_to += buf_ld;
        }
    }

    void maybe_restore_dst_values_from_buffer(
            int ithr, dim_t b_idx, dim_t m_blk_idx, dim_t n_blk_idx) const {
        if (!copy_d_required(m_blk_idx, n_blk_idx)) return;

        const bool m_tail_overlapping = is_m_tail_overlap(m_blk_idx);
        dim_t m_start = m_tail_overlapping ? get_M_idx(m_blk_idx, true)
                                           : get_M_idx(m_blk_idx);
        const dim_t rows_to_copy = m_tail_overlapping
                ? m_tail_processing_[get_M_tail_block_idx(m_blk_idx)].shift
                : get_M_kernel_size(m_blk_idx);

        const bool n_tail_overlapping = is_n_tail_overlap(n_blk_idx);
        dim_t n_start = n_tail_overlapping ? get_N_idx(n_blk_idx, true)
                                           : get_N_idx(n_blk_idx);
        const dim_t row_elems = n_tail_overlapping
                ? n_tail_processing_[get_N_tail_block_idx(n_blk_idx)].shift
                : get_N_kernel_size(n_blk_idx);
        const dim_t bytes_to_copy = bgmmc_.c_dt_sz * row_elems;

        assert(!(n_tail_overlapping && m_tail_overlapping)
                && "dynamic tail processing for both M/N is not supported");

        auto copy_from = get_buf_D_ptr(ithr);
        auto copy_to = get_data_C_ptr(b_idx, m_start, n_start);
        const dim_t dst_ld = get_LDD() * bgmmc_.c_dt_sz;
        const dim_t buf_ld = bgmmc_.N_blk * bgmmc_.c_dt_sz;
        for (int r = 0; r < rows_to_copy; r++) {
            utils::array_copy(copy_to, copy_from, bytes_to_copy);
            copy_from += buf_ld;
            copy_to += dst_ld;
        }
    }

    dim_t get_LDC() const { return LDC_; }

    dim_t get_LDD() const { return LDD_; }

    bool is_chunks_horizontal_process_order() const {
        return is_thread_chunks_exec_order_horizontal_;
    }

    dim_t copy_B_wei_stride() const { return copy_B_wei_stride_; }

    bool packed_sparse_weights() const { return bgmmc_.packed_sparse_weights; }

    dim_t get_current_K_pad(dim_t current_K_iters) const {
        if (bgmmc_.is_wei_zp_per_k || bgmmc_.is_wei_scale_per_k) return 0;
        if (current_K_iters % bgmmc_.wei_k_blk == 0) return 0;
        return (bgmmc_.extendable_k || bgmmc_.use_fused_copy_a)
                ? bgmmc_.wei_k_blk
                        - rnd_up(
                                current_K_iters % bgmmc_.wei_k_blk, vnni_factor)
                : 0;
    }

private:
    struct tail_processing_t {
        // dimension index kernel is applied to
        dim_t idx;
        // index of tail processing kernel, 0 is reserved for main block
        int kernel_idx;
        // block size of tail kernel
        dim_t kernel_size;
        // shift wrt dimension index when kernel is applied w/ computational
        // overlapping with other kernel, dim_idx_to_apply_kernel = idx - shift
        dim_t shift;
        // if shift > 0 (computational overlapping case) we have to use buffer
        // for kernel dst to avoid result values spoiling, this value
        // represents dimensional idx for dst buffer
        dim_t buf_dim_idx;
    };

    bool is_amx_;
    bool is_A_batch_layout_trivial_;
    bool is_B_batch_layout_trivial_;
    bool is_C_batch_layout_trivial_;
    const brgemm_matmul_conf_t &bgmmc_;
    const memory_desc_wrapper src_d_;
    const memory_desc_wrapper wei_d_;
    const memory_desc_wrapper dst_d_;
    const char *data_A_ptr_;
    const char *data_B_ptr_;
    // The offsets and bitmask pointers are only available when the weights
    // are sparse and packed.
    const dim_t *data_B_offsets_ptr_;
    const char *data_B_bitmask_ptr_;
    // The size of a packed saprse block. E.g. the block
    // for a tag 'BA16a64b4a' is 4096.
    dim_t B_packed_sparse_block_size_;

    char *data_C_ptr_;
    char *data_reduce_ptr_;
    brgemm_batch_element_t *batch_element_ptr_;

    char *buf_A_ptr_;
    char *buf_B_ptr_;
    char *buf_C_ptr_;
    char *buf_D_ptr_;
    char *buf_reduce_ptr_;

    char *wsp_tile_ptr_;
    const char *bias_ptr_;
    const void *src_scales_;
    const void *wei_scales_;

    const void *dst_scales_;
    const void *dst_scales_inv_;
    int32_t *s8s8_compensation_ptr_;

    int32_t *zero_point_a_compensations_ptr_;
    int32_t *zero_point_b_compensations_ptr_;
    int32_t *reorder_zp_a_comp_ptr_;
    // Per-thread f32 tile (size M_blk × LDC)
    float *per_mn_comp_ptr_;

    const void *src_zp_ptr_;
    const void *wei_zp_ptr_;
    const void *dst_zp_ptr_;
    std::vector<const void *> post_ops_binary_rhs_arg_vec_;

    int base_brg_ker_idx_;
    dim_t vnni_factor;

    // parallelization parameters
    dim_t parallel_work_amount_;
    dim_t parallel_work_amount_gemm_;
    int nthr_, nthr_k_, nthr_bmn_, num_threads_used_;
    // Horizontal order means first process N (load) dim then M (bcast) dim.
    bool is_thread_chunks_exec_order_horizontal_;
    int last_brgemm_batch_size_;

    dim_t M_;
    dim_t M_chunks_;
    dim_t M_chunk_tail_;
    dim_t M_chunk_tail_elements_;
    dim_t M_tail_block_start_;

    dim_t K_;
    dim_t K_chunks_;
    dim_t K_chunk_tail_;
    dim_t K_chunk_tail_elements_;

    dim_t N_;
    dim_t N_chunks_;
    dim_t N_chunk_tail_;
    dim_t N_chunk_tail_elems_;

    dim_t N_tail_block_start_;
    dim_t A_strides_[3];
    dim_t A_ptr_shift_b_;
    dim_t copy_A_src_stride_;
    dim_t B_strides_[3];
    dim_t B_ptr_shift_b_;
    dim_t C_strides_[3];
    dim_t C_ptr_shift_b_;
    dim_t LDC_, LDD_;
    dim_t copy_B_wei_stride_;
    std::vector<tail_processing_t> m_tail_processing_;
    std::vector<tail_processing_t> n_tail_processing_;

    char *get_buf_D_ptr(int ithr) const {
        return buf_D_ptr_ + bgmmc_.c_dt_sz * bgmmc_.M_blk * bgmmc_.N_blk * ithr;
    }

    dim_t get_M_tail_block_idx(dim_t m_block_idx) const {
        const dim_t tail_idx = m_block_idx - M_tail_block_start_;
        if (!bgmmc_.is_runtime_M) return tail_idx;
        return tail_idx < (dim_t)m_tail_processing_.size() ? tail_idx : -1;
    }
    bool is_M_tail_processing(dim_t m_block_idx) const {
        return get_M_tail_block_idx(m_block_idx) >= 0;
    }
    bool is_runtime_M_tail_chunk(dim_t m_block_idx) const {
        return bgmmc_.is_runtime_M && is_M_tail_processing(m_block_idx);
    }

    bool is_m_tail_overlap(dim_t m_block_idx) const {
        return is_runtime_M_tail_chunk(m_block_idx)
                && m_tail_processing_[get_M_tail_block_idx(m_block_idx)].shift
                > 0;
    }

    dim_t get_N_tail_block_idx(dim_t n_block_idx) const {
        const dim_t tail_idx = n_block_idx - N_tail_block_start_;
        if (!bgmmc_.is_runtime_N) return tail_idx;
        return tail_idx < (dim_t)n_tail_processing_.size() ? tail_idx : -1;
    }
    bool is_N_tail_processing(dim_t n_block_idx) const {
        return get_N_tail_block_idx(n_block_idx) >= 0;
    }

    bool is_runtime_N_tail_chunk(dim_t n_block_idx) const {
        return bgmmc_.is_runtime_N && is_N_tail_processing(n_block_idx);
    }

    bool is_n_tail_overlap(dim_t n_block_idx) const {
        return is_runtime_N_tail_chunk(n_block_idx)
                && n_tail_processing_[get_N_tail_block_idx(n_block_idx)].shift
                > 0;
    }

    bool copy_d_required(dim_t m_block_idx, dim_t n_block_idx) const {
        if (!bgmmc_.with_sum) return false;
        return is_m_tail_overlap(m_block_idx) || is_n_tail_overlap(n_block_idx);
    }
};

template struct brgemm_matmul_t<avx10_2_amx_2>;
template struct brgemm_matmul_t<avx512_core_amx_fp16>;
template struct brgemm_matmul_t<avx512_core_amx>;
template struct brgemm_matmul_t<avx10_2>;
template struct brgemm_matmul_t<avx10_2_ace>;
template struct brgemm_matmul_t<avx512_core_fp16>;
template struct brgemm_matmul_t<avx512_core_bf16>;
template struct brgemm_matmul_t<avx512_core_vnni>;
template struct brgemm_matmul_t<avx2_vnni_2>;
template struct brgemm_matmul_t<avx2_vnni>;
template struct brgemm_matmul_t<avx2>;
template struct brgemm_matmul_t<avx512_core>;

} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
