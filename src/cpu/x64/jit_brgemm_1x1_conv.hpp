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

#ifndef CPU_X64_JIT_BRGEMM_1X1_CONV_HPP
#define CPU_X64_JIT_BRGEMM_1X1_CONV_HPP

#include <forward_list>

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"

#include "cpu/x64/amx_tile_configure.hpp"
#include "cpu/x64/brgemm/brgemm.hpp"
#include "cpu/x64/brgemm/brgemm_containers.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/cpu_reducer.hpp"
#include "cpu/x64/jit_brgemm_conv_fp8_cvt_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"
#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct brgemm_1x1_convolution_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brgconv_1x1:", isa, ""),
                brgemm_1x1_convolution_fwd_t);

        status_t init(const engine_t *engine);

        struct brgemm_init_params_t {
            brgemm_init_params_t(int k_accum_idx, dim_t m, dim_t n, dim_t k,
                    dim_t lda, bool wary_tail_read)
                : k_accum_idx_(k_accum_idx)
                , M_(m)
                , N_(n)
                , K_(k)
                , LDA_(lda)
                , wary_tail_read_(wary_tail_read) {}
            const int k_accum_idx_; // controls brgemm:beta param
            const dim_t M_, N_, K_;
            const dim_t LDA_;
            bool wary_tail_read_ {false};
        };

        std::shared_ptr<brgemm_containers::brgemm_desc_container_t> brgs_;
        std::forward_list<brgemm_init_params_t> brgemm_init_params_;

        bool need_postwork_ {};
        dim_t ic_chunks_ {};

        jit_brgemm_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();

    private:
        status_t init_brgemm_desc();
    };

    brgemm_1x1_convolution_fwd_t(const pd_t *apd)
        : primitive_t(apd), bias_d(pd()->weights_md(1)) {}

    ~brgemm_1x1_convolution_fwd_t() override = default;

    status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward_all(ctx);

        if (pd()->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);

        return status::success;
    }

protected:
    status_t init(engine_t *engine) override;

private:
    //  brgemm convolution execution context
    struct brgemm_exec_ctx_t {
        brgemm_exec_ctx_t(const exec_ctx_t &ctx, const pd_t *pd)
            : src(CTX_IN_MEM(const char *, DNNL_ARG_SRC))
            , weights(CTX_IN_MEM(const char *, DNNL_ARG_WEIGHTS))
            , bias(CTX_IN_MEM(const char *, DNNL_ARG_BIAS))
            , dst(CTX_OUT_MEM(char *, DNNL_ARG_DST))
            , post_ops_binary_rhs_arg_vec(binary_injector::prepare_binary_args(
                      pd->attr()->post_ops_, ctx))
            , wsp_tile(ctx.get_scratchpad_grantor().template get<char>(
                      memory_tracking::names::key_conv_amx_tile_buffer)) {}
        const char *__restrict src;
        const char *__restrict weights;
        const char *const __restrict bias;
        char *const __restrict dst;
        const std::vector<const void *> post_ops_binary_rhs_arg_vec;
        char *const wsp_tile;
    };

    void maybe_rtus(int ithr, const char *__restrict src,
            char *__restrict inp_buffer, uint8_t *__restrict inp_buffer_mask,
            int g, int n, dim_t icc, dim_t od, dim_t oh, dim_t ow) const;
    void exec_ker(const brgemm_exec_ctx_t &brgemm_ctx, int ithr,
            brgemm_batch_element_t *const __restrict brg_batch,
            char *const c_buffer, const char *inp_buffer, int g, int n, int ocb,
            dim_t od, dim_t oh, dim_t ow, dim_t icc, int *last_brg_idx,
            const int32_t *src_zero_points, int32_t *src_zp_comp,
            const int32_t *dst_zero_points, int32_t *s8s8_compensation,
            const void *src_scales, const void *wei_scales,
            const void *dst_scales_inv, char *fp8_wsp_base,
            const bool is_last_os = false) const;
    void execute_os_blocking(const brgemm_exec_ctx_t &brgemm_ctx,
            brgemm_batch_element_t *const brg_batch_global,
            const void *src_scales, const void *wei_scales,
            const void *dst_scales, void *dst_scales_inv,
            const int32_t *src_zero_points, int32_t *src_zp_comp,
            const int32_t *dst_zero_points, int32_t *s8s8_compensation,
            char *const c_buffer_global, char *inp_buffer_base,
            uint8_t *inp_buffer_mask_base, char *fp8_wsp_base) const;
    void execute_full_spatial(const brgemm_exec_ctx_t &brgemm_ctx,
            brgemm_batch_element_t *const brg_batch_global,
            const void *src_scales, const void *wei_scales,
            const void *dst_scales, void *dst_scales_inv,
            const int32_t *src_zero_points, int32_t *src_zp_comp,
            const int32_t *dst_zero_points, int32_t *s8s8_compensation,
            char *const c_buffer_global, char *fp8_wsp_base) const;

    status_t execute_forward_all(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    static int get_brg_idx(const jit_brgemm_conv_conf_t &jcp,
            const typename pd_t::brgemm_init_params_t &bparams) {
        const int is_M_tail = bparams.M_ == jcp.M_tail;
        const int is_N_tail = bparams.N_ == jcp.N_tail;
        const int is_K_tail = bparams.K_ == jcp.K_tail;
        return get_brg_idx(bparams.k_accum_idx_, is_M_tail, is_N_tail,
                is_K_tail, bparams.wary_tail_read_);
    }

    static int get_brg_idx(int do_initialization, bool is_M_tail,
            bool is_N_tail, bool is_K_tail, bool wary_tail_read) {
        return ((((int)do_initialization * 2 + (int)is_M_tail) * 2
                        + (int)is_N_tail)
                               * 2
                       + (int)is_K_tail)
                * 2
                + (int)wary_tail_read;
    }

    static int get_ker_po_idx(int is_M_tail, bool is_N_tail) {
        return (int)is_M_tail * 2 + (int)is_N_tail;
    }

    // when there is no M_tail and there are two or more M blocks
    // e.g. 'jcp.nb_os > 1', use a separate BRGeMM and use RTUS input
    // only for the last M block.
    static bool get_extra_m_kernel_req(const jit_brgemm_conv_conf_t &jcp) {
        return jcp.is_reduced_rtus && jcp.M_tail == 0 && jcp.nb_os > 1;
    }

    // Note: when there is only one BRGeMM to compute the M dimension, need to
    // break K accumulation into two separate BRGeMMs, one for IC without RTUS,
    // and another for padded IC through RTUS.
    static bool get_compute_partial_k_in_rtus(
            const jit_brgemm_conv_conf_t &jcp) {
        return jcp.is_reduced_rtus && (!get_extra_m_kernel_req(jcp));
    }

    brgemm_containers::brgemm_kernel_container_t brg_kernels_ {64};
    brgemm_containers::brgemm_palette_container_t brgemm_palettes_ {64};

    std::unique_ptr<jit_brgemm_conv_src_copy_kernel_t> fp8_src_copy_;
    std::unique_ptr<jit_brgemm_conv_fp8_cvt_kernel_t> fp8_weights_cvt_;

    std::unique_ptr<jit_avx512_core_brgemm_conv_trans_kernel::
                    jit_avx512_core_brgemm_conv_rtus_kernel_t>
            rtus_kernel_;

    const memory_desc_wrapper bias_d;

    dim_t ID, IH, IW, OD, OH, OW, SD, SH, SW;
    size_t bia_dsz, acc_dsz, src_dsz, wei_dsz;
    // const variables used for address calculations
    dim_t src_w_sz, src_h_sz, src_d_sz, dst_w_sz, dst_h_sz, dst_d_sz;
    dim_t wei_g_stride, wei_ic_stride, wei_ocb_stride;
    dim_t wei_kw_stride, wei_kh_stride, wei_kd_stride;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
