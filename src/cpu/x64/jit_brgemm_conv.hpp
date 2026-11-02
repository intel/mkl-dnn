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

#ifndef CPU_X64_JIT_BRGEMM_CONV_HPP
#define CPU_X64_JIT_BRGEMM_CONV_HPP

#include <array>
#include <unordered_map>

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
#include "cpu/x64/jit_avx512_core_amx_conv_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_comp_pad_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_fp8_cvt_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_relo_copy_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_trans_kernel.hpp"
#include "cpu/x64/jit_brgemm_conv_utils.hpp"
#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

template <cpu_isa_t isa>
struct brgemm_convolution_fwd_t : public primitive_t {

    struct brgemm_thread_ctx_t;

    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T(JIT_IMPL_NAME_HELPER("brg_conv_fwd:", isa, ""),
                brgemm_convolution_fwd_t);

        status_t init(const engine_t *engine);

        dim_t brgs_sz_ {};
        std::shared_ptr<brgemm_containers::brgemm_desc_container_t>
                brgemm_descriptors_;
        bool with_sum_ = false;
        jit_brgemm_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();

        dim_t ic_chunks {};
        bool need_postwork {};
        dim_t wei_g_stride {}, wei_ic_stride {}, wei_ocb_stride {};
        dim_t wei_kw_stride {}, wei_kh_stride {}, wei_kd_stride {};
        dim_t pbuf_w_sz {}, pbuf_h_sz {}, pbuf_d_sz {};
        int ndims {0};
        dim_t rd {0};

        // batch sizes info for unrolled kernels
        int bs_c {};
        // need custom hasher to use array as key in unordered_map
        template <int asize>
        struct hasher_t {
            size_t operator()(const std::array<int, asize> &a) const {
                size_t seed = 0;
                for (auto e : a)
                    seed = hash_combine(seed, e);
                return seed;
            }
        };
        template <int asize>
        using Arrmap = std::unordered_map<std::array<int, asize>, int,
                hasher_t<asize>>;

        Arrmap<4> batchsizes;
        int brg_indices_c {0};
        // Bounded kernel-variant lookup key/index, not tensor-scale.
        Arrmap<8> brg_indices;

        int get_brg_idx(int m, bool do_initialization, bool is_N_tail,
                bool is_K_tail, int kd_b, int kd_e, int kh_b, int kh_e) const;

        inline int get_bs(
                dim_t kd_b, dim_t kd_e, dim_t kh_b, dim_t kh_e) const {
            const auto kd_l = nstl::min<dim_t>(KD_BLOCK, kd_e - kd_b);
            const auto kh_l = nstl::min<dim_t>(KH_BLOCK, kh_e - kh_b);
            const auto bs = kd_l
                    * (jcp_.is_relo_whi()
                                    ? 1
                                    : (kh_l * (jcp_.is_relo_wi() ? 1 : KW)));
            return static_cast<int>(bs);
        }

        int get_any_brg_idx(bool is_N_tail, bool is_K_tail) const;

        inline dim_t maybe_invert(dim_t k, dim_t K) const {
            return desc()->use_inversion ? K - 1 - k : k;
        }

        // This method calculates the value of k_l
        void init_batch(dim_t icc, const char *src_base, const char *wei_base,
                dim_t n_ic_blocks, dim_t ic_block_s, dim_t iid_b, dim_t iih_b,
                dim_t iiw_b, const dim_t *const __restrict kw_top_vpads,
                const dim_t *const __restrict kw_bottom_vpads, dim_t kd_b,
                dim_t kd_e, dim_t kh_b, dim_t kh_e, dim_t kw_b, dim_t kw_e,
                dim_t &k_l, brgemm_batch_element_t *brg_batch) const;

        void get_A_B(dim_t icc, const char *src_base, const char *wei_base,
                dim_t ic_block_s, dim_t iid_b, dim_t iih_b, dim_t iiw_b,
                dim_t kd_b, dim_t kh_b, const void *&ptrA,
                const void *&ptrB) const;

        status_t add_brg_descriptor(dim_t M, bool is_N_tail, bool is_K_tail,
                bool do_init, int kd_b, int kd_e, int kh_b, int kh_e);

    protected:
        dim_t KD {}, KH {}, KW {}, EXT_KD {}, EXT_KH {}, EXT_KW {}, KS {},
                KD_BLOCK {}, KH_BLOCK {}, KW_BLOCK {}, KD_BLOCK_PAD {},
                KH_BLOCK_PAD {}, ID {}, IH {}, IW {}, IDP {}, IHP {}, IWP {},
                OD {}, OH {}, OW {}, SD {}, SH {}, SW {}, FP {}, TP {}, LP {},
                DD {}, DH {}, DW {};
        size_t acc_dsz {}, bia_dsz {}, src_dsz {}, wei_dsz {}, dst_dsz {};
        dim_t src_w_sz {}, src_h_sz {}, dst_w_sz {}, dst_h_sz {}, wei_ocb_sz {};
        dim_t adj_src_h_sz {}, adj_src_h_offset {}, src_iw_offset {},
                src_d_offset {}, wei_ic_offset {}, wei_kd_offset {},
                wei_kh_offset {}, wei_kw_offset {};
    };

    brgemm_convolution_fwd_t(const pd_t *apd);

    ~brgemm_convolution_fwd_t() override = default;

    status_t execute(const exec_ctx_t &ctx) const override;

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
                      pd->attr()->post_ops_, ctx)) {}
        const char *__restrict src;
        const char *__restrict weights;
        const char *const __restrict bias;
        char *const __restrict dst;
        const std::vector<const void *> post_ops_binary_rhs_arg_vec;
    };

    inline static int get_ker_po_idx(
            dim_t m, bool do_postwork, bool is_N_tail) {
        return static_cast<int>((m * 2 + static_cast<int>(do_postwork)) * 2
                + static_cast<int>(is_N_tail));
    }

    inline static dim_t get_inp_size(dim_t max_src_size, dim_t dst_size,
            dim_t k, dim_t stride, dim_t dilate) {
        const auto res = nstl::min<dim_t>(max_src_size,
                calculate_end_padding(0, dst_size, 0, stride,
                        calculate_extended_filter_size(k, dilate)));
        return res;
    }

    inline dim_t maybe_invert_range(dim_t k, dim_t k_inv, dim_t K) const {
        return pd()->desc()->use_inversion ? K - k_inv : k;
    }

    dim_t get_src_base_offset(
            const brgemm_thread_ctx_t &btc, const dim_t ic) const;

    void ker_base(brgemm_thread_ctx_t &btc) const;
    void ker_trans(brgemm_thread_ctx_t &btc) const;
    void ker_vpad(brgemm_thread_ctx_t &btc) const;

    void perform_outwork(const brgemm_thread_ctx_t &btc, char *dst_base,
            const char *bias_w, dim_t ow, dim_t g_oc, bool is_oc_tail,
            dim_t ker_ow_s, dim_t ker_ow_f, dim_t kd_l, dim_t kh_l,
            bool maybe_do_init, bool do_postwork, size_t comp_ker_offs,
            bool do_post_comp) const;

    void call_brgemm_kernel(const brgemm_thread_ctx_t &btc,
            const brgemm_kernel_t *brg_ker, int batch_size, char *ptr_C,
            char *ptr_D, const char *bias_w, dim_t g_oc, bool do_postops,
            size_t comp_ker_offs, bool do_only_comp) const;

    void maybe_conv_inp(brgemm_thread_ctx_t &btc,
            const brgemm_thread_ctx_t &last_btc,
            const char *__restrict src) const;

    void maybe_conv_weights(const exec_ctx_t &ctx,
            const char *__restrict input_weights,
            const char *__restrict &wei) const;

    status_t add_po_kernel(brgemm_desc_t *bcfg, int ker_idx, bool is_init);
    status_t add_po_kernels(int i_N, dim_t init_bcast_dim, dim_t po_bcast_dim);
    status_t add_brg_kernel(int brg_idx);

    status_t cal_compensation(const char *__restrict weights,
            int32_t *src_zp_buffer, int32_t *s8s8_comp_buffer) const;
    dim_t get_comp_oh(dim_t oh) const;
    dim_t get_comp_ker_idx(dim_t kd_b, dim_t kd_e, dim_t kh_b, dim_t kh_e,
            dim_t kw_b, dim_t kw_e, dim_t oh) const;
    dim_t get_comp_offset(dim_t g, dim_t ocb, dim_t oh, dim_t ow, dim_t kd_b,
            dim_t kd_e, dim_t kh_b, dim_t kh_e, dim_t kw_b, dim_t kw_e) const;
    inline const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    brgemm_containers::brgemm_kernel_container_t brgemm_kernels_;
    brgemm_containers::brgemm_palette_container_t brgemm_palettes_;

    std::vector<std::unique_ptr<jit_brgemm_kernel_post_ops_base_t>> kernels_po_;
    std::unique_ptr<jit_avx512_core_brgemm_conv_trans_kernel::
                    jit_avx512_core_brgemm_conv_trans_kernel_t>
            copy_to_pbuffer_;
    std::unique_ptr<jit_avx512_core_amx_copy_to_pbuffer_t>
            copy_to_relo_pbuffer_;
    std::unique_ptr<jit_brgemm_relo_copy_to_wbuffer_t> copy_to_relo_wbuffer_;
    std::unique_ptr<jit_brgemm_conv_src_copy_kernel_t> fp8_src_copy_;
    std::unique_ptr<jit_brgemm_conv_fp8_cvt_kernel_t> fp8_weights_cvt_;

    std::unique_ptr<jit_generator_t> comp_vpad_pbuffer_;

    size_t acc_dsz, bia_dsz, src_dsz, wei_dsz, dst_dsz;

    const memory_desc_wrapper bias_d;

    // pre - calculated values
    std::vector<dim_t> owb_kw_top_vpads;
    std::vector<dim_t> owb_kw_bottom_vpads;
    std::vector<dim_t> kd_bs, kd_es, kh_bs, kh_es, kw_bs, kw_es, oh_kh_b,
            oh_kh_e, comp_oh, comp_oh_kh_b, comp_oh_kh_e, comp_owb;

    dim_t KD, KH, KW, EXT_KD, EXT_KH, EXT_KW, KS, KD_BLOCK, KH_BLOCK, KW_BLOCK,
            KD_BLOCK_PAD, KH_BLOCK_PAD, ID, IH, IW, IDP, IHP, IWP, OD, OH, OW,
            SD, SH, SW, FP, TP, LP, DD, DH, DW;
    dim_t src_w_sz, src_h_sz, dst_w_sz, dst_h_sz;
    dim_t ker_vpad_sz, comp_ocb_sz, comp_ker_sz, comp_kw_sz, comp_ow_sz;

    bool is_relo_with_relo_weights;
    bool need_compensation;
    bool is_amx;
};

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
