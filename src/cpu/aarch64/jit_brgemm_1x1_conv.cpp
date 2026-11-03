/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2024-2025 FUJITSU LIMITED
* Copyright 2025-2026 Arm Ltd. and affiliates
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
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/scale_utils.hpp"

#include "cpu/aarch64/brgemm/brgemm_utils.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/aarch64/jit_brgemm_1x1_conv.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using namespace utils;

    const auto src_type = src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;
    const bool is_int8 = one_of(src_type, u8, s8);

    using skip_mask_t = primitive_attr_t::skip_mask_t;
    auto skip_mask = skip_mask_t::post_ops | skip_mask_t::sum_dt
            | skip_mask_t::zero_points;
    if (one_of(src_type, u8, s8)) skip_mask |= skip_mask_t::scales;

    VDISPATCH_CONV(DNNL_CPU_THREADING_RUNTIME != DNNL_RUNTIME_THREADPOOL,
            VERBOSE_UNSUPPORTED_THREADPOOL_RUNTIME);

    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && expect_data_types(src_type, wei_type, data_type::undef, dst_type,
                    data_type::undef)
            && IMPLICATION(is_int8,
                    one_of(dst_type, u8, bf16, f32)
                            && one_of(bias_md_.data_type, data_type::undef, f32,
                                    s32, s8, u8))
            && IMPLICATION(!is_int8,
                    one_of(bias_md_.data_type, data_type::undef, f32, src_type))
            && attr()->has_default_values(skip_mask, dst_type)
            && attr()->post_ops_.check_sum_consistency(dst_type, is_int8)
            && !has_zero_dim_memory() && zero_points_ok()
            && impl::is_dense_format_kind({src_md(), weights_md(), dst_md()});
    if (!ok) return status::unimplemented;

    CHECK(attr_scales_ok());

    CHECK(brgemm_convolution_utils::init_1x1_conf(jcp_, isa, *desc(), src_md_,
            weights_md_, dst_md_, bias_md_, attr_, dnnl_get_max_threads()));

    // brgemm is slower than jit_sve when combined with reorders for shapes where strides < 2
    const convolution_desc_t &cd = *desc();
    if (!cd.use_inversion && one_of(data_type::f32, src_type, wei_type)
            && (jcp_.stride_w < 2 || jcp_.stride_h < 2)) {
        return status::unimplemented;
    }

    brgs_ = std::make_shared<brgemm_containers::brgemm_desc_container_t>(16);

    const float alpha = 1.0;
    const float beta = 1.0;
    const auto &p = attr()->post_ops_;

    // TODO: fix failing post ops for bf16 on sve 128
    const bool is_bf16
            = src_type == data_type::bf16 && wei_type == data_type::bf16;
    if (is_bf16 && get_max_cpu_isa() == sve_128) {
        for (auto const &entry : p.entry_) {
            const bool is_failing_po = entry.is_eltwise()
                    && one_of(entry.eltwise.alg,
                            // these fail due to label offset being too large
                            alg_kind::eltwise_tanh, alg_kind::eltwise_gelu_tanh,
                            alg_kind::eltwise_gelu_erf);
            VDISPATCH_CONV(!is_failing_po, VERBOSE_BAD_ALGORITHM);
        }
    }

    const int sum_idx = p.find(primitive_kind::sum);
    with_sum = (sum_idx != -1);
    // Check if postop sum datatype is supported
    if (with_sum) {
        const auto &sum_po = p.entry_[sum_idx];
        if (!one_of(sum_po.sum.dt, data_type::undef, data_type::f32,
                    data_type::s32, data_type::u8, data_type::s8,
                    data_type::bf16))
            return status::unimplemented;
    }
    sum_scale = with_sum ? p.entry_[sum_idx].sum.scale : 0.0;

    ic_chunks = div_up(jcp_.nb_ic, jcp_.nb_ic_blocking);
    need_postwork = jcp_.with_bias || jcp_.with_eltwise || jcp_.with_binary
            || (one_of(src_type, u8, s8) && wei_type == s8) // oscales needed
            || (jcp_.dst_dt != jcp_.acc_dt) || jcp_.with_sum;

    int i_init_begin = (ic_chunks == 1) ? 1 : 0;
    int i_init_end = 2;

    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for_(int i_K = 0; i_K < 2; i_K++)
    for (int i_init = i_init_begin; i_init < i_init_end; i_init++) {
        auto vbeta = (i_init) ? 0 : beta;
        auto vM = (i_M) ? jcp_.M_tail : jcp_.M;
        auto vN = (i_N) ? jcp_.N_tail : jcp_.N;
        auto vK = (i_K) ? jcp_.K_tail : jcp_.K;
        const auto brg_idx = get_brg_idx(i_init, i_M, i_N, i_K);
        if (vM == 0 || vN == 0 || vK == 0) continue;
        brgemm_desc_t brg;
        brgemm_strides_t brg_strides;
        brg_strides.stride_a = jcp_.brg_stride_a;
        brg_strides.stride_b = jcp_.brg_stride_b;
        const auto strides_ptr
                = (jcp_.brg_type == brgemm_strd) ? &brg_strides : nullptr;
        CHECK(brgemm_desc_init(&brg, isa, jcp_.brg_type, src_type, wei_type,
                false, false, brgemm_row_major, alpha, vbeta, jcp_.LDA,
                jcp_.LDB, jcp_.LDC, vM, vN, vK, strides_ptr));

        if (jcp_.use_mmla) {
            brgemm_attr_t brgattr;
            brgemm_utils::set_mmla_attr(brgattr, vM);
            brgattr.max_bs = jcp_.max_batch;
            CHECK(brgemm_desc_set_attr(&brg, brgattr));
        }

        auto LDD = jcp_.oc_without_padding;
        brg.with_sum = with_sum;
        brg.with_weights_scale_adjust = jcp_.scale_adjust_factor != 1.0f;
        CHECK(brgemm_desc_set_postops(
                &brg, attr(), &dst_md_, LDD, jcp_.bia_dt));
        CHECK(brgemm_desc_finalize(&brg));
        brgs_->insert(brg_idx, brg);
    }

    auto scratchpad = scratchpad_registry().registrar();
    brgemm_convolution_utils::init_scratchpad(scratchpad, jcp_);
    if (jcp_.with_scales)
        book_precomputed_scales(scratchpad, attr()->scales_, OC(),
                jcp_.scale_adjust_factor != 1.0f);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::init(engine_t *engine) {
    auto ndims = pd()->ndims();
    if (ndims < 3 || ndims > 5) assert(!"Invalid ndims!");

    const auto &jcp = pd()->jcp_;

    ID = ndims_pick(jcp.id, 1, 1);
    IH = ndims_pick(jcp.ih, jcp.ih, 1);
    IW = jcp.iw;

    OD = ndims_pick(jcp.od, 1, 1);
    OH = ndims_pick(jcp.oh, jcp.oh, 1);
    OW = jcp.ow;

    SD = ndims_pick(jcp.stride_d, 1, 1);
    SH = ndims_pick(jcp.stride_h, jcp.stride_h, 1);
    SW = jcp.stride_w;

    bia_dsz = jcp.bia_dsz;
    acc_dsz = jcp.acc_dsz;
    src_dsz = jcp.src_dsz;
    wei_dsz = jcp.wei_dsz;

    // const variables used for address calculations
    src_w_sz = (dim_t)IW * jcp.ngroups * jcp.ic_without_padding;
    src_h_sz = IH * src_w_sz;
    src_d_sz = ID * src_h_sz;
    dst_w_sz = (dim_t)OW * jcp.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;
    dst_d_sz = OD * dst_h_sz;

    const auto src_type = pd()->src_md(0)->data_type;

    const auto last_ic_block = jcp.use_mmla
            ? brgemm_utils::mmla_wei_k_block()
            : data_type_vnni_granularity(src_type);

    wei_ic_stride = jcp.wei_plain ? jcp.oc_without_padding : jcp.oc_block;
    wei_ocb_stride = jcp.wei_plain
            ? jcp.oc_block
            : (dim_t)rnd_up(jcp.ic, last_ic_block) * jcp.oc_block;
    wei_g_stride = jcp.wei_plain ? jcp.oc : jcp.nb_oc * wei_ocb_stride;

    if (jcp.is_rtus) {
        CHECK(safe_ptr_assign(rtus_kernel_,
                new jit_sve_core_brgemm_conv_trans_kernel::
                        jit_sve_core_brgemm_conv_rtus_kernel_t(jcp)));
        CHECK(rtus_kernel_->create_kernel());
    }
    int i_init_begin = (pd()->ic_chunks == 1) ? 1 : 0;
    int i_init_end = 2;

    const auto &brgs = *(pd()->brgs_);

    for_(int i_M = 0; i_M < 2; i_M++)
    for_(int i_N = 0; i_N < 2; i_N++)
    for_(int i_K = 0; i_K < 2; i_K++)
    for (int i_init = i_init_begin; i_init < i_init_end; i_init++) {
        auto brg_idx = get_brg_idx(i_init, i_M, i_N, i_K);
        auto brg = brgs[brg_idx];
        if (brg != nullptr && brg->bcast_dim > 0 && brg->load_dim > 0
                && brg->reduce_dim > 0 && !brg_kernels_[brg_idx]) {
            CHECK(brg_kernels_.insert(brg_idx, brg));
        }
    }
    return status::success;
}

template <cpu_isa_t isa>
void brgemm_1x1_convolution_fwd_t<isa>::maybe_rtus(int ithr,
        const char *__restrict src, char *__restrict inp_buffer,
        uint8_t *__restrict inp_buffer_mask, int g, int n, int icc, int od,
        int oh, int ow) const {
    const auto &jcp = pd()->jcp_;
    if (!jcp.is_rtus) return;
    assert(jcp.is_os_blocking);
    const size_t src_dt_size = jcp.src_dsz;

    const auto os = (od * OH + oh) * OW + ow;
    const auto osb = os / jcp.os_block;

    uint8_t *bmask = &inp_buffer_mask[icc * jcp.nb_os + osb];
    if (bmask && *bmask) return; // skip if already masked
    if (bmask) *bmask = 1; // set mask to skip next time

    const auto g_ic = g * jcp.ic_without_padding
            + icc * jcp.nb_ic_blocking * jcp.ic_block;

    auto call_kernel = [&](int nh, int nw, int od, int oh, int ow) {
        assert(nh == 0 || (nw == 0 && ow == 0));
        if (utils::everyone_is(0, nh, nw)) return;
        const int id = od * jcp.stride_d;
        const int ih = oh * jcp.stride_h;
        const int iw = ow * jcp.stride_w;
        const auto inp_offset = n * src_d_sz + id * src_h_sz + ih * src_w_sz
                + iw * jcp.ngroups * jcp.ic_without_padding + g_ic;
        auto p = jit_sve_core_brgemm_conv_trans_kernel::
                jit_brgemm_conv_trans_kernel_args_t();
        p.h_count = nh;
        p.owb = nw;
        p.src = src + src_dt_size * inp_offset;
        p.dst = inp_buffer;
        (*rtus_kernel_)(&p);
        inp_buffer += src_dt_size * (nh * jcp.ow + nw) * jcp.LDA;
    };

    const bool is_os_tail = jcp.os - os < jcp.os_block;
    int count = is_os_tail ? jcp.M_tail : jcp.M;

    if (count < OW || ow > 0) {
        // copy to end of row
        const auto nw = nstl::min(count, OW - ow);
        call_kernel(0, nw, od, oh, ow);
        count -= nw;
        if (count == 0) return;
        ow = 0;
        oh = (oh + 1) % OH;
        if (oh == 0) od++;
    }

    while (od < OD) {
        // copy to end of column
        const auto nh = nstl::min(count / OW, OH - oh);
        call_kernel(nh, 0, od, oh, ow);
        count -= nh * OW;
        if (count == 0) return;
        oh = (oh + nh) % OH;
        if (oh == 0) od++;
        if (count < OW) {
            // copy partial row
            const auto nw = count;
            call_kernel(0, nw, od, oh, ow);
            return;
        }
    }
}

template <cpu_isa_t isa>
void brgemm_1x1_convolution_fwd_t<isa>::exec_ker(
        const brgemm_exec_ctx_t &brgemm_ctx, int ithr,
        brgemm_batch_element_t *const __restrict brg_batch,
        char *const c_buffer, const char *inp_buffer, int g, int n, int ocb,
        int od, int oh, int ow, int icc, int *last_brg_idx,
        const float *oscales, int32_t src_zp_vals, int32_t *src_zp_comp,
        const int32_t *dst_zero_points, int32_t *s8s8_compensation,
        const float *dst_scales) const {

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const size_t src_dt_size = types::data_type_size(src_d.data_type());
    const size_t wei_dt_size = types::data_type_size(weights_d.data_type());
    const size_t dst_dt_size = types::data_type_size(dst_d.data_type());

    const char *const __restrict src = brgemm_ctx.src;
    const char *const __restrict weights = brgemm_ctx.weights;
    const char *const __restrict bias = brgemm_ctx.bias;
    char *const __restrict dst = brgemm_ctx.dst;
    const std::vector<const void *> &post_ops_binary_rhs_arg_vec
            = brgemm_ctx.post_ops_binary_rhs_arg_vec;

    const auto &jcp = pd()->jcp_;
    auto ndims = pd()->ndims();

    const int id = ndims_pick(od * SD, 0, 0);
    const int ih = ndims_pick(oh * SH, oh * SH, 0);
    const int iw = ow * SW;

    const int oc = ocb * jcp.oc_block;
    const int g_oc = g * jcp.oc + oc;

    const int icb = icc * jcp.nb_ic_blocking;
    const int ic = icb * jcp.ic_block;
    const int g_ic = g * jcp.ic + ic;

    const bool kernel_init = (icc == 0);

    const auto os = (od * OH + oh) * OW + ow;

    const bool is_os_tail = jcp.is_os_blocking ? (jcp.os - os < jcp.os_block)
                                               : (OW - ow < jcp.ow_block);
    const bool is_oc_tail = (jcp.oc - oc < jcp.oc_block);
    const bool is_ic_tail = (icc == pd()->ic_chunks - 1
            && ((jcp.ic - ic) % jcp.ic_block != 0));

    const auto src_offset = n * src_d_sz + id * src_h_sz + ih * src_w_sz
            + iw * jcp.ngroups * jcp.ic_without_padding + g_ic;
    const auto src_base
            = jcp.is_rtus ? inp_buffer : src + src_dt_size * src_offset;
    const auto wei_offset = g * wei_g_stride + ocb * wei_ocb_stride;
    const auto wei_base = weights + wei_dt_size * wei_offset;
    const auto ptr_D = dst
            + dst_dt_size
                    * (n * dst_d_sz + od * dst_h_sz + oh * dst_w_sz
                            + ow * jcp.oc_without_padding + g_oc);
    char *const ptr_C = (jcp.use_buffer) ? c_buffer : (char *)ptr_D;

    const auto bias_w
            = bias ? bias + (bias_d.blk_off(g_oc) * bia_dsz) : nullptr;
    const auto nb_ic_b = nstl::min(jcp.nb_ic_blocking, jcp.nb_ic - icb)
            - (is_ic_tail ? 1 : 0);

    const auto comp_offset = (g * jcp.nb_oc + ocb) * jcp.oc_block;
    int32_t *src_zp_comp_ptr
            = (jcp.src_zero_point && icc == pd()->ic_chunks - 1)
            ? &src_zp_comp[comp_offset]
            : nullptr;
    int32_t *s8s8_comp_ptr
            = (jcp.s8s8_compensation_required && icc == pd()->ic_chunks - 1)
            ? &s8s8_compensation[comp_offset]
            : nullptr;

    const auto call_brgemm = [=](int brg_idx, int ic_block_s, int n_ic_blocks,
                                     bool do_postops) {
        for (int k = 0; k < n_ic_blocks; k++) {
            const auto ic_off = (ic_block_s + k) * jcp.ic_block;
            const auto src_ic = ic_off;
            const auto wei_ic = ic + ic_off;
            const auto ptr_A = src_base + src_dt_size * src_ic;
            const auto ptr_B = wei_base + wei_dt_size * wei_ic * wei_ic_stride;
            brg_batch[k].ptr.A = ptr_A;
            brg_batch[k].ptr.B = ptr_B;
            brg_batch[k].vvpad.top = 0;
            brg_batch[k].vvpad.bottom = 0;
        }

        const auto brg_ker = brg_kernels_[brg_idx];
        if (do_postops) {
            const brgemm_post_ops_data_t post_ops_data {
                    static_cast<const void *>(bias_w),
                    &oscales[jcp.is_oc_scale * g_oc],
                    post_ops_binary_rhs_arg_vec.data(),
                    static_cast<size_t>(g_oc), 0, dst, 0,
                    static_cast<void *>(src_zp_comp_ptr), nullptr,
                    dst_zero_points, false, src_zp_vals, false, false,
                    dst_scales};

            void *scratch = static_cast<void *>(s8s8_comp_ptr);
            brgemm_kernel_execute_postops(brg_ker, n_ic_blocks, brg_batch,
                    (void *)ptr_C, (void *)ptr_D, post_ops_data, scratch);
        } else {
            void *scratch = static_cast<void *>(s8s8_comp_ptr);
            brgemm_kernel_execute(
                    brg_ker, n_ic_blocks, brg_batch, (void *)ptr_C, scratch);
        }
    };

    const auto do_post_work = (pd()->need_postwork || jcp.use_buffer)
            && icc == pd()->ic_chunks - 1;

    if (nb_ic_b > 0) {
        const auto brg_idx
                = get_brg_idx(kernel_init, is_os_tail, is_oc_tail, false);
        call_brgemm(brg_idx, 0, nb_ic_b, do_post_work && !is_ic_tail);
    }
    if (is_ic_tail) {
        const auto use_init_ker = (kernel_init && nb_ic_b == 0);
        const auto brg_idx
                = get_brg_idx(use_init_ker, is_os_tail, is_oc_tail, true);

        call_brgemm(brg_idx, nb_ic_b, 1, do_post_work);
    }
}

template <cpu_isa_t isa>
status_t brgemm_1x1_convolution_fwd_t<isa>::execute_forward_all(
        const exec_ctx_t &ctx) const {

    brgemm_exec_ctx_t brgemm_ctx(ctx, pd());

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    const auto &jcp = pd()->jcp_;
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const float *oscales = precompute_scales(ctx.get_scratchpad_grantor(),
            src_scales, wei_scales, pd()->OC(), pd()->attr(),
            jcp.scale_adjust_factor);

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);
    const int src_zero_point = src_zero_points ? src_zero_points[0] : 0;

    const auto extra_data_offset
            = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<char *>(brgemm_ctx.weights);
    int32_t *s8s8_compensation = (jcp.s8s8_compensation_required)
            ? reinterpret_cast<int32_t *>(w + extra_data_offset)
            : nullptr;
    int32_t *zp_compensation = (jcp.src_zero_point)
            ? reinterpret_cast<int32_t *>(&w[extra_data_offset])
                    + (jcp.s8s8_compensation_required
                                    ? jcp.s8s8_comp_buffer_size
                                    : 0)
            : nullptr;

    brgemm_batch_element_t *const brg_batch_global
            = (jcp.brg_type != brgemm_strd)
            ? scratchpad.template get<brgemm_batch_element_t>(
                      key_brgemm_primitive_batch)
            : nullptr;
    char *const c_buffer_global = (jcp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;
    char *inp_buffer_base = (jcp.is_rtus)
            ? scratchpad.template get<char>(key_conv_brgemm_inp_buffer)
            : nullptr;
    uint8_t *inp_buffer_mask_base = (jcp.is_rtus)
            ? scratchpad.template get<uint8_t>(key_conv_brgemm_inp_buffer_mask)
            : nullptr;

    if (jcp.is_os_blocking) {
        const int os_chunks = div_up(jcp.nb_os, jcp.nb_os_blocking);
        const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_oc * os_chunks;

#define BRGC_WO(...) \
    parallel(pd()->jcp_.nthr, [&](const int ithr, const int nthr) { \
        if (ithr >= work_amount) return; \
        brgemm_batch_element_t *const brg_batch \
                = brg_batch_global + (size_t)ithr * jcp.adjusted_batch_size; \
        char *const c_buffer = (jcp.use_buffer) \
                ? c_buffer_global + ithr * acc_dsz * jcp.LDC * jcp.M \
                : nullptr; \
        char *inp_buffer = (jcp.is_rtus) \
                ? inp_buffer_base + ithr * src_dsz * jcp.inp_buffer_size \
                : nullptr; \
        uint8_t *__restrict inp_buffer_mask = (jcp.is_rtus) \
                ? inp_buffer_mask_base + ithr * jcp.inp_buffer_mask_size \
                : nullptr; \
        int last_n = -1; \
        int last_g = -1; \
        int last_brg_idx = -1; \
        int start {0}, end {0}; \
        balance211(work_amount, nthr, ithr, start, end); \
        int n {0}, g {0}, ocb {0}, oss {0}; \
        nd_iterator_init(start, __VA_ARGS__); \
        for (auto work = start; work < end; work++) { \
            if (jcp.is_rtus && (last_n != n || last_g != g)) \
                std::memset(inp_buffer_mask, 0, jcp.inp_buffer_mask_size); \
            const auto osb_start = oss * jcp.nb_os_blocking; \
            const auto osb_range \
                    = nstl::min(jcp.nb_os - osb_start, jcp.nb_os_blocking); \
            for (int osb = 0; osb < osb_range; osb++) { \
                const int os = (osb_start + osb) * jcp.os_block; \
                const int od = os / (OH * OW); \
                const int oh = (os % (OH * OW)) / OW; \
                const int ow = os % OW; \
                char *inp_buffer_sp = (jcp.is_rtus) \
                        ? inp_buffer + src_dsz * os * jcp.LDA \
                        : nullptr; \
                for (int icc = 0; icc < pd()->ic_chunks; icc++) { \
                    if (jcp.is_rtus) \
                        maybe_rtus(ithr, brgemm_ctx.src, inp_buffer_sp, \
                                inp_buffer_mask, g, n, icc, od, oh, ow); \
                    exec_ker(brgemm_ctx, ithr, brg_batch, c_buffer, \
                            inp_buffer_sp, g, n, ocb, od, oh, ow, icc, \
                            &last_brg_idx, oscales, src_zero_point, \
                            zp_compensation, dst_zero_points, \
                            s8s8_compensation, dst_scales); \
                } \
            } \
            last_n = n; \
            last_g = g; \
            nd_iterator_step(__VA_ARGS__); \
        } \
    });

        if (jcp.loop_order == loop_ndhwgc)
            BRGC_WO(n, jcp.mb, oss, os_chunks, g, jcp.ngroups, ocb, jcp.nb_oc)
        else if (jcp.loop_order == loop_ngcdhw)
            BRGC_WO(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, oss, os_chunks)
        else
            assert(!"Unknown loop order");

#undef BRGC_WO

    } else {
        const int work_amount
                = jcp.mb * jcp.ngroups * jcp.nb_oc * OD * OH * jcp.nb_ow;

#define BRGC_WO(...) \
    parallel(pd()->jcp_.nthr, [&](const int ithr, const int nthr) { \
        if (ithr >= work_amount) return; \
        brgemm_batch_element_t *const brg_batch \
                = brg_batch_global + (size_t)ithr * jcp.adjusted_batch_size; \
        char *const c_buffer = (jcp.use_buffer) \
                ? c_buffer_global + ithr * acc_dsz * jcp.LDC * jcp.M \
                : nullptr; \
        int last_brg_idx = -1; \
        int start {0}, end {0}; \
        balance211(work_amount, nthr, ithr, start, end); \
        int n {0}, g {0}, ocb {0}, od {0}, oh {0}, owb {0}; \
        nd_iterator_init(start, __VA_ARGS__); \
        for (auto work = start; work < end; work++) { \
            for (int icc = 0; icc < pd()->ic_chunks; icc++) { \
                const int ow = owb * jcp.ow_block; \
                exec_ker(brgemm_ctx, ithr, brg_batch, c_buffer, nullptr, g, n, \
                        ocb, od, oh, ow, icc, &last_brg_idx, oscales, \
                        src_zero_point, zp_compensation, dst_zero_points, \
                        s8s8_compensation, dst_scales); \
            } \
            nd_iterator_step(__VA_ARGS__); \
        } \
    });

        if (jcp.loop_order == loop_ndhwgc)
            BRGC_WO(n, jcp.mb, od, OD, oh, OH, owb, jcp.nb_ow, g, jcp.ngroups,
                    ocb, jcp.nb_oc)
        else if (jcp.loop_order == loop_ngcdhw)
            BRGC_WO(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, od, OD, oh, OH,
                    owb, jcp.nb_ow)
        else
            assert(!"Unknown loop order");

#undef BRGC_WO
    }

    return status::success;
}

template struct brgemm_1x1_convolution_fwd_t<sve_512>;
template struct brgemm_1x1_convolution_fwd_t<sve_256>;
template struct brgemm_1x1_convolution_fwd_t<sve_128>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
