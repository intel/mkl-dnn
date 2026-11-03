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

#ifndef CPU_AARCH64_JIT_BRGEMM_CONV_HPP
#define CPU_AARCH64_JIT_BRGEMM_CONV_HPP

#include <array>
#include <map>
#include <unordered_map>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/brgemm/brgemm_containers.hpp"
#include "cpu/aarch64/jit_brgemm_conv_mmla_pack_kernel.hpp"
#include "cpu/aarch64/jit_brgemm_conv_trans_kernel.hpp"
#include "cpu/aarch64/jit_brgemm_post_ops.hpp"
#include "cpu/cpu_convolution_pd.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

template <cpu_isa_t isa>
struct brgemm_convolution_fwd_t : public primitive_t {

    struct brgemm_thread_ctx_t;

    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        // ------- DECLARE_COMMON_PD_t -----
        DECLARE_COMMON_PD_T(
                (jcp_.use_mmla ? JIT_IMPL_NAME_HELPER(
                                         "brgconv_mmla:", jcp_.isa, "")
                               : JIT_IMPL_NAME_HELPER(
                                         "brgconv:", jcp_.isa, "")),
                brgemm_convolution_fwd_t);

        status_t init(const engine_t *engine);

        int brgs_sz_;
        std::shared_ptr<brgemm_containers::brgemm_desc_container_t>
                brgemm_descriptors_;
        bool with_sum = false;
        jit_brgemm_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();

        int ic_chunks;
        bool need_postwork;
        dim_t wei_g_stride, wei_ic_stride, wei_ocb_stride;
        dim_t wei_kw_stride, wei_kh_stride, wei_kd_stride;
        dim_t pbuf_w_sz, pbuf_h_sz, pbuf_d_sz;

        // batch sizes info for unrolled kernels
        int bs_c;
        // need custom hasher to use array as key in unordered_map
        struct hasher_t {
            size_t operator()(const std::array<int, 4> &a) const {
                size_t seed = 0;
                for (auto e : a)
                    seed = hash_combine(seed, e);
                return seed;
            }
        };
        std::unordered_map<std::array<int, 4>, int, hasher_t> batchsizes;

        inline size_t get_brg_idx(int m, bool do_initialization, bool is_N_tail,
                bool is_K_tail, int kd_b, int kd_e, int kh_b, int kh_e) const {
            int bs_idx = 0;
            if (jcp_.use_uker) {
                const auto bs = batchsizes.find({kd_b, kd_e, kh_b, kh_e});
                if (bs == batchsizes.end()) {
                    assert(!"unregistered batch size");
                    return 0;
                }
                bs_idx = bs->second;
            }
            return (((m * bs_c + bs_idx) * 2
                            + static_cast<int>(do_initialization))
                                   * 2
                           + static_cast<int>(is_N_tail))
                    * 2
                    + static_cast<int>(is_K_tail);
        }

        int get_any_brg_idx(bool is_N_tail, bool is_K_tail) const {
            // return first defined brgemm_descriptor for specified parameters
            const int M_end = nstl::max(jcp_.M, jcp_.M_tail);
            const bool N_begin = (jcp_.N == jcp_.N_tail) ? false : is_N_tail;
            const bool N_end = (jcp_.N == jcp_.N_tail) ? true : is_N_tail;
            const bool K_begin = (jcp_.K == jcp_.K_tail) ? false : is_K_tail;
            const bool K_end = (jcp_.K == jcp_.K_tail) ? true : is_K_tail;
            for_(int m = 0; m < M_end; m++)
            for_(bool i_init : {false, true})
            for_(bool i_N_tail : {N_begin, N_end})
            for_(bool i_K_tail : {K_begin, K_end})
            for (const auto &key_value_pair : batchsizes) {
                const int kd_b = key_value_pair.first[0];
                const int kd_e = key_value_pair.first[1];
                const int kh_b = key_value_pair.first[2];
                const int kh_e = key_value_pair.first[3];
                const auto brg_idx = get_brg_idx(
                        m, i_init, i_N_tail, i_K_tail, kd_b, kd_e, kh_b, kh_e);
                if ((*brgemm_descriptors_)[brg_idx]) return brg_idx;
            }
            return 0;
        }

        inline int maybe_invert(int k, int K) const {
            return desc()->use_inversion ? K - 1 - k : k;
        }

        void init_batch(int icc, const char *src_base, const char *wei_base,
                int n_ic_blocks, int ic_block_s, int iid_b, int iih_b,
                int iiw_b, const dim_t *const __restrict kw_top_vpads,
                const dim_t *const __restrict kw_bottom_vpads, int kd_b,
                int kd_e, int kh_b, int kh_e, int kw_b, int kw_e, int k_l,
                brgemm_batch_element_t *brg_batch) const;

        void get_A_B(int icc, const char *src_base, const char *wei_base,
                int ic_block_s, int iid_b, int iih_b, int iiw_b, int kd_b,
                int kh_b, const void *&ptrA, const void *&ptrB) const;

        status_t add_brg_descriptor(int M, int i_N, int i_K, int i_init,
                int kd_b, int kd_e, int kh_b, int kh_e);

        int ndims = 0;

    protected:
        bool zero_points_ok() const {
            const auto &zp = attr()->zero_points_;

            if (!zp.has_default_values(DNNL_ARG_SRC)) {
                int mask_src = zp.get_mask(DNNL_ARG_SRC);
                const bool ok = mask_src == 0;
                if (!ok) return false;
            }
            if (!zp.has_default_values(DNNL_ARG_DST)) {
                int mask_dst = zp.get_mask(DNNL_ARG_DST);
                const bool ok = mask_dst == 0;
                if (!ok) return false;
            }

            return zp.has_default_values(DNNL_ARG_WEIGHTS);
        }

        int KD, KH, KW, EXT_KD, EXT_KH, EXT_KW, KS, KD_BLOCK, KH_BLOCK,
                KW_BLOCK, KD_BLOCK_PAD, KH_BLOCK_PAD, ID, IH, IW, IDP, IHP, IWP,
                OD, OH, OW, SD, SH, SW, FP, TP, LP, DD, DH, DW;
        size_t acc_dsz, bia_dsz, src_dsz, wei_dsz, dst_dsz;
        dim_t src_w_sz, src_h_sz, src_d_sz, dst_w_sz, dst_h_sz, dst_d_sz,
                wei_ocb_sz;
        dim_t adj_src_h_sz, adj_src_h_offset, src_iw_offset, src_d_offset,
                wei_ic_offset, wei_kd_offset, wei_kh_offset, wei_kw_offset;
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
        const char *const __restrict src;
        const char *const __restrict weights;
        const char *const __restrict bias;
        char *const __restrict dst;
        const std::vector<const void *> post_ops_binary_rhs_arg_vec;
    };

    inline static int get_ker_po_idx(int m, bool do_postwork, bool is_N_tail) {
        return (m * 2 + static_cast<int>(do_postwork)) * 2
                + static_cast<int>(is_N_tail);
    }

    inline static int get_inp_size(
            int max_src_size, int dst_size, int k, int stride, int dilate) {
        const auto res = nstl::min(max_src_size,
                calculate_end_padding(0, dst_size, 0, stride,
                        calculate_extended_filter_size(k, dilate)));
        return res;
    }

    inline int maybe_invert_range(int k, int k_inv, int K) const {
        return pd()->desc()->use_inversion ? K - k_inv : k;
    }

    void get_kw_range(
            int ow, int &kw_s, int &kw_full_s, int &kw_full_e, int &kw_e) const;
    void get_ow_range(int ow, int kw, int &ow_s, int &ow_e) const;

    void ker_base(brgemm_thread_ctx_t &btc) const;
    void ker_trans(brgemm_thread_ctx_t &btc, char *inp_buffer) const;
    void ker_vpad(brgemm_thread_ctx_t &btc) const;

    void perform_outwork(const brgemm_thread_ctx_t &btc, char *dst_base,
            const char *bias_w, int ow, int g_oc, bool is_oc_tail, int ker_ow_s,
            int ker_ow_f, int kd_l, int kh_l, bool maybe_do_init,
            bool do_postwork, bool do_post_comp) const;

    void call_brgemm_kernel(const brgemm_thread_ctx_t &btc,
            const brgemm_kernel_t *brg_ker, int batch_size, char *ptr_C,
            char *ptr_D, const char *bias_w, int g_oc, bool do_postops,
            int comp_ker_offs, bool do_only_comp) const;

    void maybe_conv_inp(int ithr, const char *__restrict src,
            char *__restrict inp_buffer, uint8_t *__restrict inp_buffer_mask,
            int g, int n, int icc, int odb, int ohb, int owb, int last_g,
            int last_n, int last_icc, int last_odb, int last_ohb,
            int last_owb) const;

    status_t add_po_kernel(brgemm_desc_t *bcfg, int ker_idx, bool is_init);
    void add_po_kernels(int i_N, int init_bcast_dim, int po_bcast_dim);
    status_t add_brg_kernel(int M, int i_N, int i_K, int i_init, int kd_b,
            int kd_e, int kh_b, int kh_e);

    status_t cal_compensation(const char *__restrict weights,
            int32_t *src_zp_buffer, int32_t *s8s8_comp_buffer) const;
    int get_comp_ker_idx(const int kd_b, const int kd_e, const int kh_b,
            const int kh_e, const int kw_b, const int kw_e) const;
    int get_comp_offset(const int g, const int ocb, const int ow,
            const int kd_b, const int kd_e, const int kh_b, const int kh_e,
            const int kw_b, const int kw_e) const;
    inline const pd_t *pd() const {
        return static_cast<const pd_t *>(primitive_t::pd().get());
    }

    brgemm_containers::brgemm_kernel_container_t brgemm_kernels_;

    std::map<size_t, std::unique_ptr<jit_brgemm_kernel_post_ops_t<isa>>>
            kernels_po_;
    std::unique_ptr<jit_sve_core_brgemm_conv_trans_kernel::
                    jit_sve_core_brgemm_conv_trans_kernel_t>
            copy_to_pbuffer_;
    std::unique_ptr<jit_generator_t> comp_vpad_pbuffer_;

    // int8 mmla: A packers for the full-K and the K-tail brgemm calls
    std::unique_ptr<jit_brgemm_conv_mmla_pack_kernel::
                    jit_brgemm_conv_mmla_pack_kernel_t>
            mmla_pack_kernels_[2];
    int mmla_pack_K_[2] = {0, 0};

    void mmla_pack_src(brgemm_thread_ctx_t &btc, int bs, int rows,
            bool is_k_tail, int kd_b, int kd_e, int kh_b, int kh_e, int iid,
            int iih, bool mark_dh_padding) const;

    void mmla_compute_wei_sums(const char *weights, int32_t *wsum) const;
    bool mmla_zp_correction(const brgemm_thread_ctx_t &btc, int rows, int N,
            int iid, int iih, const dim_t *kw_top_vpads,
            const dim_t *kw_bottom_vpads, int32_t *corr, dim_t ldc) const;

    size_t acc_dsz, bia_dsz, src_dsz, wei_dsz, dst_dsz;

    const memory_desc_wrapper bias_d;

    // pre - calculated values
    std::vector<dim_t> owb_kw_top_vpads;
    std::vector<dim_t> owb_kw_bottom_vpads;
    std::vector<dim_t> kd_bs, kd_es, kh_bs, kh_es, kw_bs, kw_es;

    int KD, KH, KW, EXT_KD, EXT_KH, EXT_KW, KS, KD_BLOCK, KH_BLOCK, KW_BLOCK,
            KD_BLOCK_PAD, KH_BLOCK_PAD, ID, IH, IW, IDP, IHP, IWP, OD, OH, OW,
            SD, SH, SW, FP, TP, LP, DD, DH, DW;
    dim_t src_w_sz, src_h_sz, src_d_sz, dst_w_sz, dst_h_sz, dst_d_sz;
    dim_t ker_vpad_sz, comp_ocb_sz, comp_ker_sz, comp_kw_sz;

    bool need_compensation;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
