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
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/jit_brgemm_conv.hpp"

#include <vector>

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

using namespace jit_avx512_core_brgemm_conv_trans_kernel;
using namespace jit_uni_brgemm_conv_comp_pad_kernel;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

template <cpu_isa_t isa>
int brgemm_convolution_fwd_t<isa>::pd_t::get_brg_idx(int m,
        bool do_initialization, bool is_N_tail, bool is_K_tail, int kd_b,
        int kd_e, int kh_b, int kh_e) const {
    const auto brg_idx = jcp_.use_uker
            ? brg_indices.find({m, is_N_tail, is_K_tail, do_initialization,
                      kd_b, kd_e, kh_b, kh_e})
            : brg_indices.find({m, is_N_tail, is_K_tail, do_initialization, 0,
                      static_cast<int>(jcp_.kd), 0, static_cast<int>(jcp_.kh)});
    if (brg_idx == brg_indices.end()) return -1;
    return brg_idx->second;
}

template <cpu_isa_t isa>
int brgemm_convolution_fwd_t<isa>::pd_t::get_any_brg_idx(
        bool is_N_tail, bool is_K_tail) const {
    // return first defined brgemm_descriptor for specified parameters
    for (const auto &key_value_pair : brg_indices) {
        const bool i_N = key_value_pair.first[1];
        const bool i_K = key_value_pair.first[2];
        if ((jcp_.N == jcp_.N_tail || is_N_tail == i_N)
                && (jcp_.K == jcp_.K_tail || is_K_tail == i_K))
            return key_value_pair.second;
    }
    return 0;
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::pd_t::init_batch(dim_t icc,
        const char *src_base, const char *wei_base, dim_t n_ic_blocks,
        dim_t ic_block_s, dim_t iid_b, dim_t iih_b, dim_t iiw_b,
        const dim_t *const __restrict kw_top_vpads,
        const dim_t *const __restrict kw_bottom_vpads, dim_t kd_b, dim_t kd_e,
        dim_t kh_b, dim_t kh_e, dim_t kw_b, dim_t kw_e, dim_t &k_l,
        brgemm_batch_element_t *brg_batch) const {
    const char *ptrA {nullptr};
    const char *ptrB {nullptr};
    const auto &jcp = jcp_;

    assert(IMPLICATION(jcp.is_relo(), kw_b == 0));
    assert(IMPLICATION((jcp.is_relo_whi()), kh_b == 0));

    kw_e = jcp.is_relo() ? kw_b + 1 : kw_e;
    kh_e = jcp.is_relo_whi() ? kh_b + 1 : kh_e;

    k_l = (kd_e - kd_b) * (kh_e - kh_b) * (kw_e - kw_b);
    if (k_l == 0) return;

    const dim_t icb = icc * jcp.nb_ic_blocking;
    const dim_t wei_ic_base = icb * jcp.ic_block;

    for (dim_t i_icb = 0; i_icb < n_ic_blocks; i_icb++) {
        const auto ic_off = (ic_block_s + i_icb) * jcp.ic_block;
        const auto wei_ic = wei_ic_base + ic_off;
        const auto n_icb_off = i_icb * k_l;
        const auto src_base_shift = jcp.exec_type == exec_trans
                ? (jcp.copy_block_only ? 0 : (i_icb * pbuf_d_sz))
                : ic_off;
        const auto src_base_ic = src_base + src_base_shift * src_dsz;
        const auto wei_base_ic = wei_base + wei_ic * wei_ic_offset;
        const auto need_A_B = (jcp.use_uker
                && (jcp.brg_type == brgemm_offs
                        || jcp.brg_type == brgemm_static_offs));

        auto k = 0;
        for (dim_t kd = kd_b; kd < kd_e; kd++) {
            const auto id = iid_b + kd * DD;
            const auto src_base_kd = src_base_ic + id * src_d_offset;
            const auto wei_kd = maybe_invert(kd, KD);
            const auto wei_base_kd = wei_base_ic + wei_kd * wei_kd_offset;
            for (dim_t kh = kh_b; kh < kh_e; kh++) {
                const auto ih = iih_b + kh * DH;
                const auto src_base_kh = src_base_kd + ih * adj_src_h_offset;
                const auto wei_kh = maybe_invert(kh, KH);
                const auto wei_base_kh = wei_base_kd + wei_kh * wei_kh_offset;

                for (dim_t kw = kw_b; kw < kw_e; kw++) {
                    const auto iw = iiw_b + kw * DW;
                    const auto b_idx = n_icb_off + k;
                    const auto A_addr = src_base_kh + iw * src_iw_offset;
                    // general wei layout is gOdhwI<block_o><block_i>
                    const auto wei_kw = maybe_invert(kw, KW);
                    const auto B_addr = wei_base_kh + wei_kw * wei_kw_offset;
                    if (b_idx == 0 && need_A_B) {
                        ptrA = A_addr;
                        ptrB = B_addr;
                    }

                    if (jcp.brg_type == brgemm_addr) {
                        brg_batch[b_idx].ptr.A = A_addr;
                        brg_batch[b_idx].ptr.B = B_addr;
                    } else if (jcp.brg_type == brgemm_offs
                            || jcp.brg_type == brgemm_static_offs) {
                        brg_batch[b_idx].offset.A = (dim_t)A_addr - (dim_t)ptrA;
                        brg_batch[b_idx].offset.B = (dim_t)B_addr - (dim_t)ptrB;
                    }
                    if (jcp.max_vpad != 0) {
                        brg_batch[b_idx].vvpad.top = kw_top_vpads[kw];
                        brg_batch[b_idx].vvpad.bottom = kw_bottom_vpads[kw];
                    }

                    k++;
                }
            }
        }
    }
}

template <cpu_isa_t isa>
inline void brgemm_convolution_fwd_t<isa>::pd_t::get_A_B(dim_t icc,
        const char *src_base, const char *wei_base, dim_t ic_block_s,
        dim_t iid_b, dim_t iih_b, dim_t iiw_b, dim_t kd_b, dim_t kh_b,
        const void *&ptrA, const void *&ptrB) const {
    const dim_t icb = icc * jcp_.nb_ic_blocking;
    const dim_t wei_ic_base = icb * jcp_.ic_block;

    // for brgemm_static_offs we need only base A_addr and B_addr
    const auto ic_off = ic_block_s * jcp_.ic_block;
    const auto wei_ic = wei_ic_base + ic_off;
    const auto src_base_shift = jcp_.exec_type == exec_trans ? 0 : ic_off;
    const auto src_base_ic = src_base + src_base_shift * src_dsz;
    const auto wei_base_ic = wei_base + wei_ic * wei_ic_offset;

    const auto id = iid_b + kd_b * DD;
    const auto src_base_kd = src_base_ic + id * src_d_offset;
    const auto wei_kd = maybe_invert(kd_b, KD);
    const auto wei_base_kd = wei_base_ic + wei_kd * wei_kd_offset;
    const auto ih = iih_b + (jcp_.is_relo_whi() ? 0 : kh_b * DH);
    const auto src_base_kh = src_base_kd + ih * adj_src_h_offset;
    const auto wei_kh = maybe_invert(kh_b, KH);
    const auto wei_base_kh = wei_base_kd + wei_kh * wei_kh_offset;

    ptrA = src_base_kh + iiw_b * src_iw_offset;
    const auto wei_kw = maybe_invert(0, KW);
    ptrB = wei_base_kh + wei_kw * wei_kw_offset;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::pd_t::add_brg_descriptor(dim_t vM,
        bool is_N_tail, bool is_K_tail, bool do_init, int kd_b, int kd_e,
        int kh_b, int kh_e) {

    if (do_init && is_K_tail && jcp_.K > 0) return status::success;

    const auto src_type = jcp_.src_dt;
    const auto wei_type = jcp_.wei_dt;
    const auto is_amx = brgemm_convolution_utils::is_amx(isa);

    const float alpha = 1.0;
    const float beta = 1.0;

    auto vbeta = do_init ? 0 : beta;
    auto vN = is_N_tail ? jcp_.N_tail : jcp_.N;
    auto vK = is_K_tail ? jcp_.K_tail : jcp_.K;
    auto vbrgM = jcp_.use_M_mask ? (vM == jcp_.M ? jcp_.brgM : jcp_.brgM_tail)
                                 : vM;
    if (vN == 0 || vK == 0) return status::success;

    int brg_idx = get_brg_idx(static_cast<int>(vM), do_init, is_N_tail,
            is_K_tail, kd_b, kd_e, kh_b, kh_e);
    // if brgemm_desc_t already created then skip this iteration
    if (brg_idx != -1) return status::success;

    brgemm_attr_t brgattr;
    // if need post_ops and there are no intermediate calculations
    // (like ic_chunks > 1 or blocking by kernel) we don't need
    // code without post-ops in brgemm kernel
    if (need_postwork && ic_chunks == 1 && KD_BLOCK == KD && KH_BLOCK == KH
            && KW_BLOCK == KW)
        brgattr.postops_only = true;

    std::vector<char> bd_mask;
    if (jcp_.use_M_mask) {
        auto sm_size = vbrgM;
        bd_mask.resize(sm_size);
        if (jcp_.is_os_blocking) {
            int ibrgM = 0;
            int iM = 0;
            for (int hh = 0; hh < jcp_.oh_block; hh++) {
                auto M_mask = (iM >= vM) ? 0 : 1;
                for (int ww = 0; ww < jcp_.ow_block && ibrgM < sm_size;
                        ww++, ibrgM++, iM += M_mask) {
                    bd_mask[ibrgM] = static_cast<char>(M_mask);
                }
                for (int kk = 0; kk < jcp_.oskip && ibrgM < sm_size;
                        kk++, ibrgM++) {
                    bd_mask[ibrgM] = 0;
                }
            }
            for (; ibrgM < sm_size; ibrgM++) {
                bd_mask[ibrgM] = 0;
            }
        } else {
            for (int ibrgM = 0; ibrgM < sm_size; ibrgM++) {
                bd_mask[ibrgM] = 1;
            }
        }
    }

    std::vector<brgemm_batch_element_t> stoffs;
    if (jcp_.brg_type == brgemm_static_offs) {
        assert(one_of(jcp_.exec_type, exec_trans, exec_base));
        const dim_t kd_f = nstl::min<dim_t>(kd_e, kd_b + KD_BLOCK);
        const dim_t kh_f = nstl::min<dim_t>(kh_e, kh_b + KH_BLOCK);

        assert(jcp_.nb_ic % jcp_.nb_ic_blocking == 0);
        const auto nb_ic_blocks = jcp_.nb_ic_blocking;

        stoffs.resize(jcp_.max_batch + 1);
        dim_t k_l {0};

        init_batch(0, nullptr, nullptr, nb_ic_blocks, 0, 0, 0, 0, nullptr,
                nullptr, kd_b, kd_f, kh_b, kh_f, 0, KW, k_l, stoffs.data());

        // if k_l is 0 then it means the batchsize is 0
        if (k_l == 0) return status::success;
    }

    const auto bs = get_bs(kd_b, kd_e, kh_b, kh_e);

    brgemm_desc_t brg;
    brgattr.bd_mask = bd_mask.data();
    brgattr.static_offsets = stoffs.data();
    brgemm_strides_t brg_strides;
    brg_strides.stride_a = jcp_.brg_stride_a;
    brg_strides.stride_b = jcp_.brg_stride_b;
    brg.req_cal_comp_pads = jcp_.req_brg_comp_pad;
    brg.req_comp_pads_with_bcast
            = jcp_.req_cal_comp_pad && jcp_.exec_type != exec_vpad;
    brg.fp8_with_f16_vnni_block
            = jcp_.is_fp8 && !jcp_.preconvert_fp8 && jcp_.vnni_block == 2;
    const auto strides_ptr
            = (jcp_.brg_type == brgemm_strd) ? &brg_strides : nullptr;
    CHECK(brgemm_desc_init(&brg, isa, jcp_.brg_type, src_type, wei_type, false,
            false, brgemm_row_major, alpha, vbeta, jcp_.LDA, jcp_.LDB, jcp_.LDC,
            vbrgM, vN, vK, strides_ptr));
    brgattr.use_uker = jcp_.use_uker;
    brgattr.use_interleave_stores = jcp_.use_interleave_stores;
    brgattr.hint_prefetching = jcp_.hint_prefetching;
    brgattr.max_bs = bs;
    brgattr.hint_ununroll_bd_loop = jcp_.ununroll_bd_loop;
    brgattr.hint_innermost_loop = jcp_.brgemm_bd_loop_innermost
            ? brgemm_bd_loop_innermost
            : brgemm_innermost_undef;
    brgattr.hint_loop_order = jcp_.brgemm_kernel_loop_order;

    if (jcp_.amx_tile_load_xx) {
        // assuming 2x2 decomposition in amx brgemm kernel
        // and overlap of input by kw
        const auto bd_blocking = 2 * jcp_.amx_h;
        const auto ld_blocking = 2 * 16;
        brgattr.hint_expected_A_size
                = bd_blocking * jcp_.K * jcp_.kd_block * jcp_.kh_block;
        brgattr.hint_expected_B_size = ld_blocking * jcp_.K * jcp_.kd_block
                * jcp_.kh_block * jcp_.kw_block;
        brgattr.hint_expected_C_size = bd_blocking * ld_blocking;
    } else {
        brgattr.hint_expected_A_size = 0;
        brgattr.hint_expected_B_size = 0;
        brgattr.hint_expected_C_size = 0;
    }

    brgattr.wary_A_k_tail_read = false;
    brgattr.bd_mask_level = jcp_.use_M_mask;

    if (is_amx) {
        brgattr.max_top_vpad = 0;
        brgattr.max_bottom_vpad = 0;
    } else {
        brgattr.max_top_vpad = jcp_.max_vpad;
        brgattr.max_bottom_vpad = jcp_.max_vpad;
    }
    brgattr.fpmath_mode = attr()->fpmath_.mode_;
    brgattr.K_koef = (float)bs / static_cast<float>(KW);
    brgattr.use_ace = jcp_.is_ace;

    CHECK(brgemm_desc_set_attr(&brg, brgattr));

    auto LDD = jcp_.oc_without_padding;
    brg.with_sum = with_sum_;
    brg.with_weights_scale_adjust = jcp_.scale_adjust_factor != 1.0f;
    CHECK(brgemm_desc_set_postops(&brg, attr(), &dst_md_, LDD, jcp_.bia_dt));
    CHECK(brgemm_desc_finalize(&brg));

    jcp_.amx_buf_size_per_thread = nstl::max<dim_t>(
            brg.get_wsp_buffer_size(), jcp_.amx_buf_size_per_thread);

    brg_idx = brgemm_descriptors_->insert(brg, bd_mask, stoffs);

    const std::array<int, 8> key = {static_cast<int>(vM), is_N_tail, is_K_tail,
            do_init, kd_b, kd_e, kh_b, kh_e};
    if (brg_indices.find(key) == brg_indices.end()) {
        brg_indices.insert({key, brg_idx});
        brg_indices_c++;
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::pd_t::init(const engine_t *engine) {
    using namespace data_type;
    using namespace utils;
    brgemm_descriptors_
            = std::make_shared<brgemm_containers::brgemm_desc_container_t>();
    ndims = cpu_convolution_fwd_pd_t::ndims();

    const auto src_type = src_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;
    const bool is_int8 = one_of(src_type, u8, s8);
    const bool is_fp8 = one_of(src_type, f8_e4m3, f8_e5m2);

    // The following check will detect if this implementation is being
    // executed through a BWD_D Convolution call and prevent the primitive from
    // executing 'use_inversion == true' as FWD. This can only work if the
    // diff_src_desc and diff_dst_desc are defined in the aforementioned.
    const convolution_desc_t &cd = *desc();
    if (cd.use_inversion
            && one_of(true, types::is_zero_md(&cd.diff_src_desc),
                    types::is_zero_md(&cd.diff_dst_desc)))
        return status::unimplemented;

    using skip_mask_t = primitive_attr_t::skip_mask_t;
    auto skip_mask = skip_mask_t::post_ops | skip_mask_t::sum_dt
            | skip_mask_t::zero_points | skip_mask_t::fpmath_mode;
    if (is_int8 || is_fp8) skip_mask |= skip_mask_t::scales;

    VDISPATCH_CONV(is_fwd(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(IMPLICATION(is_int8,
                           one_of(bias_md_.data_type, data_type::undef, f32,
                                   s32, s8, u8)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(IMPLICATION(!is_int8,
                           one_of(bias_md_.data_type, data_type::undef, f32,
                                   src_type)),
            VERBOSE_UNSUPPORTED_BIAS_CFG);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(attr()->has_default_values(skip_mask, dst_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(attr()->post_ops_.check_sum_consistency(dst_type, is_int8),
            VERBOSE_UNSUPPORTED_POSTOP);
    CHECK(attr_scales_ok());
    CHECK(attr_zero_points_ok());
    VDISPATCH_CONV(
            impl::is_dense_format_kind({src_md(0), weights_md(0), dst_md(0)}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);

    CHECK(brgemm_convolution_utils::init_conf(jcp_, isa, *desc(), src_md_,
            weights_md_, dst_md_, bias_md_, attr_, dnnl_get_max_threads()));

    // 1. The unrolled kernel can be used for exec_trans and exec_base and for
    // amx only. For exec_base it makes sense to use unrolled kernel only if
    // there is no padding by width.
    // 2. For exec_trans block by kw is always KW
    assert(IMPLICATION(jcp_.use_uker,
            brgemm_convolution_utils::is_amx(isa)
                    && one_of(jcp_.exec_type, exec_base, exec_trans)));
    assert(IMPLICATION(jcp_.use_interleave_stores, jcp_.use_uker));

    bs_c = 0;
    brg_indices_c = 0;

    KD = ndims_pick(jcp_.kd, 1, 1);
    KH = ndims_pick(jcp_.kh, jcp_.kh, 1);
    KW = jcp_.kw;

    EXT_KD = ndims_pick(jcp_.ext_kd, 1, 1);
    EXT_KH = ndims_pick(jcp_.ext_kh, jcp_.ext_kh, 1);
    EXT_KW = jcp_.ext_kw;

    IDP = ndims_pick(jcp_.idp, 1, 1);
    IHP = ndims_pick(jcp_.ihp, jcp_.ihp, 1);
    IWP = jcp_.iwp;

    KS = KD * KH * KW;
    KD_BLOCK = ndims_pick(jcp_.kd_block, 1, 1);
    KH_BLOCK = ndims_pick(jcp_.kh_block, jcp_.kh_block, 1);
    KW_BLOCK = jcp_.kw_block;
    KD_BLOCK_PAD = ndims_pick(jcp_.kd_block_pad, 1, 1);
    KH_BLOCK_PAD = ndims_pick(jcp_.kh_block_pad, jcp_.kh_block_pad, 1);
    ID = ndims_pick(jcp_.id, 1, 1);
    IH = ndims_pick(jcp_.ih, jcp_.ih, 1);
    IW = jcp_.iw;
    OD = ndims_pick(jcp_.od, 1, 1);
    OH = ndims_pick(jcp_.oh, jcp_.oh, 1);
    OW = jcp_.ow;
    SD = ndims_pick(jcp_.stride_d, 1, 1);
    SH = ndims_pick(jcp_.stride_h, jcp_.stride_h, 1);
    SW = jcp_.stride_w;
    FP = ndims_pick(jcp_.f_pad, 0, 0);
    TP = ndims_pick(jcp_.t_pad, jcp_.t_pad, 0);
    LP = jcp_.l_pad;
    DD = ndims_pick(jcp_.dilate_d, 0, 0) + 1;
    DH = ndims_pick(jcp_.dilate_h, jcp_.dilate_h, 0) + 1;
    DW = jcp_.dilate_w + 1;

    bia_dsz = jcp_.bia_dsz;
    acc_dsz = jcp_.acc_dsz;
    src_dsz = jcp_.src_dsz;
    wei_dsz = jcp_.wei_dsz;
    dst_dsz = jcp_.dst_dsz;

    // const variables used for address calculations
    src_w_sz = static_cast<dim_t>(IW) * jcp_.ngroups * jcp_.ic_without_padding;
    src_h_sz = IH * src_w_sz;
    dst_w_sz = static_cast<dim_t>(OW) * jcp_.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;
    rd = jcp_.ic;
    if (jcp_.is_relo_wi())
        rd *= jcp_.kw;
    else if (jcp_.is_relo_whi())
        rd *= jcp_.kw * jcp_.kh;

    if (jcp_.is_relo()) {
        auto adj_rd = rnd_up(rd, jcp_.vnni_block);
        if (jcp_.is_rd_padded_to_block)
            adj_rd = rnd_up(adj_rd, 16 * jcp_.vnni_block);
        wei_kw_stride = static_cast<dim_t>(adj_rd)
                * (jcp_.wei_plain ? jcp_.oc_without_padding : jcp_.oc_block);
        wei_kh_stride = wei_kw_stride;
    } else {
        wei_kw_stride = static_cast<dim_t>(jcp_.icp)
                * (jcp_.wei_plain ? jcp_.oc_without_padding : jcp_.oc_block);
        wei_kh_stride = KW * wei_kw_stride;
    }
    wei_kd_stride = (jcp_.is_relo_whi() ? 1 : KH) * wei_kh_stride;
    wei_ocb_stride = jcp_.wei_plain ? jcp_.oc_block : KD * wei_kd_stride;
    wei_g_stride = jcp_.wei_plain ? jcp_.oc : jcp_.nb_oc * wei_ocb_stride;
    wei_ic_stride = jcp_.wei_plain ? jcp_.oc_without_padding : jcp_.oc_block;
    const auto kh_koef = jcp_.is_relo_whi() ? jcp_.kh : 1;

    if (jcp_.copy_block_only) {
        assert(jcp_.exec_type == exec_trans && "Missing copy kernel");
        const auto iw_block = jit_avx512_core_brgemm_conv_trans_kernel_t::dst_w(
                jcp_, static_cast<int>(jcp_.ow_block));
        const auto ih_block = get_inp_size(IHP, jcp_.oh_block, KH, SH, DH - 1);
        const auto id_block = get_inp_size(IDP, jcp_.od_block, KD, SD, DD - 1);

        pbuf_w_sz = (dim_t)jcp_.inp_ic_block * kh_koef * iw_block;
        pbuf_h_sz = pbuf_w_sz * ih_block;
        pbuf_d_sz = pbuf_h_sz * id_block;

    } else {
        pbuf_w_sz = (dim_t)jcp_.inp_ic_block * kh_koef * IWP;
        pbuf_h_sz = pbuf_w_sz * IHP;
        pbuf_d_sz = pbuf_h_sz * IDP;
    }

    adj_src_h_sz = jcp_.exec_type == exec_trans ? pbuf_h_sz : src_h_sz;
    adj_src_h_offset
            = src_dsz * (jcp_.exec_type == exec_trans ? pbuf_w_sz : src_w_sz);

    src_iw_offset = static_cast<dim_t>(src_dsz)
            * (jcp_.exec_type == exec_trans
                            ? jcp_.inp_ic_block * kh_koef
                            : jcp_.ngroups * jcp_.ic_without_padding);
    src_d_offset = static_cast<dim_t>(src_dsz) * adj_src_h_sz;
    wei_ic_offset = static_cast<dim_t>(wei_dsz) * wei_ic_stride;
    wei_kd_offset = static_cast<dim_t>(wei_dsz) * wei_kd_stride;
    wei_kh_offset = static_cast<dim_t>(wei_dsz) * wei_kh_stride
            * ((jcp_.exec_type == exec_trans && jcp_.is_relo_whi()) ? 0 : 1);
    wei_kw_offset = static_cast<dim_t>(wei_dsz) * wei_kw_stride;

    if (jcp_.use_uker) {

        assert(KD % KD_BLOCK == 0);
        assert(KH % KH_BLOCK == 0);

        for (dim_t iod = 0; iod < jcp_.od; iod++) {
            const dim_t iid = iod * SD - FP;
            const dim_t kd_s = div_up(nstl::max<dim_t>(0, -iid), DD);
            const dim_t kd_f = KD
                    - div_up(nstl::max<dim_t>(0, iid - ID + (KD - 1) * DD + 1),
                            DD);
            for (dim_t ioh = 0; ioh < jcp_.oh; ioh++) {

                const auto iih = ioh * SH - TP;
                const auto kh_s = (jcp_.is_os_blocking || jcp_.is_relo_whi())
                        ? 0
                        : div_up(nstl::max<dim_t>(0, -iih), DH);
                const auto kh_f = (jcp_.is_relo_whi()) ? 1
                                                       : KH
                                - div_up(nstl::max<dim_t>(0,
                                                 iih - IH + (KH - 1) * DH + 1),
                                        DH);
                const auto bs = get_bs(kd_s, kd_f, kh_s, kh_f);
                if (bs <= 0) continue;

                const std::array<int, 4> key
                        = {static_cast<int>(kd_s), static_cast<int>(kd_f),
                                static_cast<int>(kh_s), static_cast<int>(kh_f)};
                if (batchsizes.find(key) == batchsizes.end()) {
                    batchsizes.insert({key, bs_c});
                    bs_c++;
                }
            }
        }
    } else {
        batchsizes.insert(
                {{0, static_cast<int>(KD), 0, static_cast<int>(KH)}, bs_c});
        bs_c++;
    }

    brgs_sz_ = 2 * 2 * 2 * 2;
    brgemm_descriptors_->resize(brgs_sz_);

    const auto &p = attr()->post_ops_;
    const int sum_idx = p.find(primitive_kind::sum);
    with_sum_ = (sum_idx != -1);

    // os_blocking is supported for exec_trans only
    assert(IMPLICATION(jcp_.exec_type != exec_trans, !jcp_.is_os_blocking));
    assert(IMPLICATION(jcp_.is_os_blocking,
            jcp_.os_block % jcp_.ow == 0 && jcp_.os_block / jcp_.ow <= jcp_.oh
                    && jcp_.os_block / jcp_.ow == jcp_.oh_block));

    const bool need_compensation
            = (jcp_.src_zero_point || jcp_.s8s8_compensation_required)
            && !jcp_.req_brg_comp_pad;

    ic_chunks = div_up(jcp_.nb_ic, jcp_.nb_ic_blocking);
    need_postwork = jcp_.with_bias || jcp_.with_eltwise || jcp_.with_binary
            || jcp_.with_src_scales || jcp_.with_wei_scales
            || jcp_.with_dst_scales || need_compensation
            || (jcp_.dst_dt != jcp_.acc_dt) || jcp_.with_sum || jcp_.use_M_mask
            || jcp_.src_zero_point || jcp_.dst_zero_point;

    const auto &Mv = (jcp_.M_tail > 0 && jcp_.M_tail != jcp_.M)
            ? std::vector<dim_t> {jcp_.M, jcp_.M_tail}
            : std::vector<dim_t> {jcp_.M};

    std::vector<bool> bv_true {true};
    std::vector<bool> bv_false {false};
    std::vector<bool> bv_both {false, true};

    const auto has_N_tail = jcp_.N_tail > 0 && jcp_.N_tail != jcp_.N;
    std::vector<bool> Nv;
    if (jcp_.N > 0) Nv.push_back(false);
    if (has_N_tail) Nv.push_back(true);

    // exec_trans may still require an is_K_tail kernel even when K_tail == K,
    // because the last-IC-block path is keyed by the K-tail variant.
    const auto has_K_tail = jcp_.K_tail > 0
            && (jcp_.K_tail != jcp_.K
                    || (jcp_.exec_type == exec_trans
                            && jcp_.K_tail == jcp_.ic_block));

    std::vector<bool> Kv;
    if (jcp_.K > 0) Kv.push_back(false);
    if (has_K_tail) Kv.push_back(true);

    const auto first_K_init_only = one_of(jcp_.exec_type, exec_trans, exec_vpad)
            && (jcp_.ic / jcp_.ic_block <= 1)
            && (KD_BLOCK == KD && KH_BLOCK == KH);

    for (const auto &key_value_pair : batchsizes) {
        const int kd_b = key_value_pair.first[0];
        const int kd_e = key_value_pair.first[1];
        const int kh_b = key_value_pair.first[2];
        const int kh_e = key_value_pair.first[3];

        for_(const auto &i_N : Nv)
        for_(const auto &M : Mv)
        for (const auto &i_K : Kv) {
            const std::vector<bool> &init_v = (i_K == Kv.front())
                    ? (first_K_init_only ? bv_true : bv_both)
                    : bv_false;
            for (const auto &i_init : init_v) {
                CHECK(add_brg_descriptor(
                        M, i_N, i_K, i_init, kd_b, kd_e, kh_b, kh_e));
            }
        }
    }

    if (jcp_.exec_type == exec_base) {
        // create brgemm kernels for ow_blocks with padded areas and
        // apply post-ops on final iteration by kw to padded areas in ow_block
        int kw_s {0}, kw_full_s {0}, kw_full_f {0}, kw_f {0}, ow_s {0},
                ow_f {0};
        for (int ow = 0; ow < OW; ow += static_cast<int>(jcp_.ow_block)) {
            brgemm_convolution_utils::get_kw_range(
                    jcp_, ow, kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                brgemm_convolution_utils::get_ow_range(
                        jcp_, ow, kw, ow_s, ow_f);
                const auto M = ow_f - ow_s;
                if (M <= 0) continue;

                for (const auto &key_value_pair : batchsizes) {
                    const int kd_b = key_value_pair.first[0];
                    const int kd_e = key_value_pair.first[1];
                    const int kh_b = key_value_pair.first[2];
                    const int kh_e = key_value_pair.first[3];

                    for_(const auto &i_N : Nv)
                    for (const auto &i_K : Kv) {
                        const std::vector<bool> &init_v = (i_K == Kv.front())
                                ? (first_K_init_only ? bv_true : bv_both)
                                : bv_false;
                        for (const auto &i_init : init_v) {
                            CHECK(add_brg_descriptor(M, i_N, i_K, i_init, kd_b,
                                    kd_e, kh_b, kh_e));
                        }
                    }
                }
            }

            if (kw_f == jcp_.kw && kw_s == 0) break;
        }

        for (dim_t ow = (jcp_.nb_ow - 1) * jcp_.ow_block; ow >= 0;
                ow -= jcp_.ow_block) {
            brgemm_convolution_utils::get_kw_range(jcp_, static_cast<int>(ow),
                    kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                brgemm_convolution_utils::get_ow_range(
                        jcp_, static_cast<int>(ow), kw, ow_s, ow_f);
                if (ow_f - ow_s <= 0) continue;

                auto M = ow_f - ow_s;
                if (M <= 0) continue;
                for (const auto &key_value_pair : batchsizes) {
                    const int kd_b = key_value_pair.first[0];
                    const int kd_e = key_value_pair.first[1];
                    const int kh_b = key_value_pair.first[2];
                    const int kh_e = key_value_pair.first[3];
                    for_(const auto &i_N : Nv)
                    for (const auto &i_K : Kv) {
                        const std::vector<bool> &init_v = (i_K == Kv.front())
                                ? (first_K_init_only ? bv_true : bv_both)
                                : bv_false;
                        for (const auto &i_init : init_v) {
                            CHECK(add_brg_descriptor(M, i_N, i_K, i_init, kd_b,
                                    kd_e, kh_b, kh_e));
                        }
                    }
                }
            }

            if (kw_f == jcp_.kw && kw_s == 0) break;
        }
    }
    brgs_sz_ = brgemm_descriptors_->refs_size();

    brgemm_convolution_utils::set_amx_wsp_per_thread(jcp_);
    auto scratchpad = scratchpad_registry().registrar();
    CHECK(brgemm_convolution_utils::init_scratchpad(
            scratchpad, jcp_, *src_md(), *weights_md(), *dst_md()));

    return status::success;
}

template <cpu_isa_t isa>
dim_t brgemm_convolution_fwd_t<isa>::get_src_base_offset(
        const brgemm_thread_ctx_t &btc, const dim_t ic) const {
    const auto &jcp = pd()->jcp_;
    const memory_desc_wrapper src_d(pd()->src_md());

    // The second arg in template means sub_offset0 = true
    // See `blk_off` method definition.
    const auto batch_offs = btc.n * src_d.blk_off<false, true>(1);
    const auto group_offs
            = btc.g * src_d.blk_off<false, true>(0, 1) * jcp.ic + ic;
    return src_dsz * (src_d.off_l(0) + batch_offs + group_offs);
}

template <cpu_isa_t isa>
brgemm_convolution_fwd_t<isa>::brgemm_convolution_fwd_t(const pd_t *apd)
    : primitive_t(apd), bias_d(pd()->weights_md(1)) {}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::add_brg_kernel(int brg_idx) {
    const auto _pd = pd();
    const auto &brgs = *(_pd->brgemm_descriptors_);

    auto brg = brgs[brg_idx];
    if (!brgemm_kernels_[brg_idx] && brg && brg->bcast_dim > 0
            && brg->load_dim > 0 && brg->reduce_dim > 0) {
        CHECK(brgemm_kernels_.insert(brg_idx, brg));
        if (is_amx && brg->is_tmm) brgemm_palettes_.insert(brg_idx, brg);
    }
    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::add_po_kernel(
        brgemm_desc_t *bcfg, int ker_idx, bool is_init) {
    if (!bcfg) return status::success;
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    bcfg->LDD = (is_init && jcp.use_buffer) ? jcp.LDC : jcp.LDD;
    bcfg->dt_c = (!is_init && jcp.use_buffer) ? jcp.acc_dt : jcp.dst_dt; // inp
    bcfg->dt_d = (is_init && jcp.use_buffer) ? jcp.acc_dt : jcp.dst_dt; // out
    bcfg->typesize_C = static_cast<int>(types::data_type_size(bcfg->dt_c));
    bcfg->typesize_D = static_cast<int>(types::data_type_size(bcfg->dt_d));
    bcfg->alpha = !is_init && IMPLICATION(jcp.with_sum, jcp.use_buffer);
    bcfg->beta = is_init ? 0.f : 1.f;
    // See the comment in `add_po_kernels` why `*_pd->attr()` is needed so far.
    CHECK(safe_ptr_assign(kernels_po_[ker_idx],
            jit_brgemm_kernel_post_ops_base_t::create(
                    isa, *bcfg, *_pd->attr())));
    kernels_po_[ker_idx]->generate_kernel();
    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::add_po_kernels(
        int i_N, dim_t init_bcast_dim, dim_t po_bcast_dim) {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &brgs = *(_pd->brgemm_descriptors_);

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    if (N <= 0) return status::success;
    auto i_K = (jcp.K_tail > 0);

    const int brg_idx = _pd->get_any_brg_idx(i_N, i_K);

    if (init_bcast_dim > 0) {
        if (brgs[brg_idx]) {
            // Note: The particular line below means a copy of brgemm_desc
            // object. The copy here is due to:
            // * PD creation time passed, original objects can't be modified.
            // * PO kernel requires (for some reason) custom values for certain
            //   members in brgemm descriptor.
            // When the copy is performed, it erases underlying memory for
            // attributes and dst_md, which means they can't be used in any
            // further call due to the temporary object on stack (after copy)
            // will be destroyed and the address of, e.g. the address of the sum
            // scale (used in the post-ops kernel), will be invalidated.
            // This copy puts restrictions on what objects can be used in
            // sub-calls and a developer should be careful about that.
            auto init_cfg = *(brgs[brg_idx]);
            auto ker_init_idx = get_ker_po_idx(init_bcast_dim - 1, false, i_N);
            if (init_cfg.load_dim > 0 && kernels_po_[ker_init_idx] == nullptr) {
                init_cfg.bcast_dim = init_bcast_dim;
                CHECK(add_po_kernel(&init_cfg, ker_init_idx, true));
            }
        }
    }

    if ((_pd->need_postwork || jcp.use_buffer) && po_bcast_dim > 0) {
        if (brgs[brg_idx]) {
            auto po_cfg = *(brgs[brg_idx]);
            auto ker_po_idx = get_ker_po_idx(po_bcast_dim - 1, true, i_N);
            if (po_cfg.load_dim > 0 && kernels_po_[ker_po_idx] == nullptr) {
                po_cfg.bcast_dim = po_bcast_dim;
                CHECK(add_po_kernel(&po_cfg, ker_po_idx, false));
            }
        }
    }
    return status::success;
}

template <cpu_isa_t isa>
dim_t brgemm_convolution_fwd_t<isa>::get_comp_oh(dim_t oh) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!(jcp.req_cal_comp_pad && jcp.exec_type == exec_trans)
            || comp_oh_kh_b.empty())
        return 0;

    const dim_t comp_oh_e = comp_oh_kh_b.size();
    const dim_t oh_block = jcp.is_os_blocking
            ? nstl::min<dim_t>(jcp.oh_block, jcp.oh - oh)
            : 1;
    for (dim_t comp_oh_b = 0; comp_oh_b < comp_oh_e; comp_oh_b++) {
        const auto cur_block
                = nstl::min<dim_t>(oh_block, comp_oh_e - comp_oh_b);
        for (dim_t i = 0; i < cur_block; i++) {
            if (oh_kh_b[oh + i] != comp_oh_kh_b[comp_oh_b + i]
                    || oh_kh_e[oh + i] != comp_oh_kh_e[comp_oh_b + i])
                break;
            if (i == cur_block - 1) return comp_oh_b;
        }
    }
    return comp_oh_e;
}

template <cpu_isa_t isa>
dim_t brgemm_convolution_fwd_t<isa>::get_comp_ker_idx(dim_t kd_b, dim_t kd_e,
        dim_t kh_b, dim_t kh_e, dim_t kw_b, dim_t kw_e, dim_t oh) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!jcp.req_cal_comp_pad) return 0;

    assert(kd_e > kd_b && kh_e > kh_b);
    for (dim_t k = 0; k < jcp.ker_ranges_size; k++) {
        if (kd_b == kd_bs[k] && kd_e == kd_es[k] && kh_b == kh_bs[k]
                && kh_e == kh_es[k] && kw_b == kw_bs[k] && kw_e == kw_es[k]
                && oh == comp_oh[k]) {
            return k;
        }
    }

    return -1;
}

template <cpu_isa_t isa>
inline dim_t brgemm_convolution_fwd_t<isa>::get_comp_offset(dim_t g, dim_t ocb,
        dim_t oh, dim_t ow, dim_t kd_b, dim_t kd_e, dim_t kh_b, dim_t kh_e,
        dim_t kw_b, dim_t kw_e) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!jcp.src_zero_point && !jcp.s8s8_compensation_required) return 0;

    const auto cur_comp_oh = get_comp_oh(oh);
    const auto cur_owb = ow / jcp.ow_block;
    const auto cur_comp_ow = jcp.req_cal_comp_pad && jcp.exec_type != exec_vpad
            ? comp_owb[cur_owb] + ow % jcp.ow_block
            : ow;
    const auto comp_idx
            = get_comp_ker_idx(kd_b, kd_e, kh_b, kh_e, kw_b, kw_e, cur_comp_oh);
    assert(IMPLICATION(jcp.req_cal_comp_pad, comp_idx >= 0));
    return jcp.req_cal_comp_pad ? g * comp_ocb_sz + ocb * comp_ker_sz
                    + comp_idx * comp_kw_sz + cur_comp_ow * comp_ow_sz
                                : (g * jcp.nb_oc + ocb) * jcp.oc_block;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::init(engine_t *engine) {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (jcp.preconvert_fp8) {
        jit_brgemm_conv_fp8_cvt_kernel_t::cfg_t cfg;
        cfg.isa = isa;
        cfg.src_dt = _pd->src_md()->data_type;
        cfg.dst_dt = f16;
        CHECK(safe_ptr_assign(
                fp8_src_copy_, new jit_brgemm_conv_src_copy_kernel_t(cfg)));
        CHECK(fp8_src_copy_->create_kernel());
        cfg.src_dt = _pd->weights_md()->data_type;
        CHECK(safe_ptr_assign(
                fp8_weights_cvt_, new jit_brgemm_conv_fp8_cvt_kernel_t(cfg)));
        CHECK(fp8_weights_cvt_->create_kernel());
    }

    bia_dsz = jcp.bia_dsz;
    acc_dsz = jcp.acc_dsz;
    src_dsz = jcp.src_dsz;
    wei_dsz = jcp.wei_dsz;
    dst_dsz = jcp.dst_dsz;

    auto ndims = _pd->ndims;
    if (ndims < 3 || ndims > 5) assert(!"Invalid ndims!");

    KD = ndims_pick(jcp.kd, 1, 1);
    KH = ndims_pick(jcp.kh, jcp.kh, 1);
    KW = jcp.kw;

    EXT_KD = ndims_pick(jcp.ext_kd, 1, 1);
    EXT_KH = ndims_pick(jcp.ext_kh, jcp.ext_kh, 1);
    EXT_KW = jcp.ext_kw;

    IDP = ndims_pick(jcp.idp, 1, 1);
    IHP = ndims_pick(jcp.ihp, jcp.ihp, 1);
    IWP = jcp.iwp;

    KS = KD * KH * KW;
    KD_BLOCK = ndims_pick(jcp.kd_block, 1, 1);
    KH_BLOCK = ndims_pick(jcp.kh_block, jcp.kh_block, 1);
    KW_BLOCK = jcp.kw_block;
    KD_BLOCK_PAD = ndims_pick(jcp.kd_block_pad, 1, 1);
    KH_BLOCK_PAD = ndims_pick(jcp.kh_block_pad, jcp.kh_block_pad, 1);
    ID = ndims_pick(jcp.id, 1, 1);
    IH = ndims_pick(jcp.ih, jcp.ih, 1);
    IW = jcp.iw;
    OD = ndims_pick(jcp.od, 1, 1);
    OH = ndims_pick(jcp.oh, jcp.oh, 1);
    OW = jcp.ow;
    SD = ndims_pick(jcp.stride_d, 1, 1);
    SH = ndims_pick(jcp.stride_h, jcp.stride_h, 1);
    SW = jcp.stride_w;
    FP = ndims_pick(jcp.f_pad, 0, 0);
    TP = ndims_pick(jcp.t_pad, jcp.t_pad, 0);
    LP = jcp.l_pad;
    DD = ndims_pick(jcp.dilate_d, 0, 0) + 1;
    DH = ndims_pick(jcp.dilate_h, jcp.dilate_h, 0) + 1;
    DW = jcp.dilate_w + 1;

    // const variables used for address calculations
    src_w_sz = static_cast<dim_t>(IW) * jcp.ngroups * jcp.ic_without_padding;
    src_h_sz = IH * src_w_sz;
    dst_w_sz = static_cast<dim_t>(OW) * jcp.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;

    comp_ow_sz = static_cast<dim_t>(jcp.oc_block);
    comp_kw_sz = comp_ow_sz * jcp.comp_ow_size;
    comp_ker_sz = jcp.ker_ranges_size * comp_kw_sz;
    comp_ocb_sz = jcp.nb_oc * comp_ker_sz;

    need_compensation = (jcp.src_zero_point || jcp.s8s8_compensation_required)
            && !jcp.req_brg_comp_pad;
    is_relo_with_relo_weights = jcp.is_relo() && jcp.relo_conv_weights;

    // ---- Initialize arrays ---------------------
    brgemm_kernels_.resize(_pd->brgs_sz_);
    brgemm_palettes_.resize(_pd->brgs_sz_);

    // #TODO: this needed only if we have d/h padding more then kd/kh
    int M_begin = 0;
    int M_end = (jcp.M_tail == jcp.M) ? 1 : 2;
    int N_begin = 0;
    int N_end = (jcp.N_tail == jcp.N) ? 1 : 2;

    const dim_t num_po_kernels = nstl::max<dim_t>(jcp.M, jcp.M_tail);
    kernels_po_.resize(num_po_kernels * 2 * 2);

    if (jcp.exec_type == exec_trans) {
        CHECK(safe_ptr_assign(copy_to_pbuffer_,
                new jit_avx512_core_brgemm_conv_trans_kernel_t(jcp)));
        CHECK(copy_to_pbuffer_->create_kernel());
    }

    if (jcp.is_relo_whi()) {
        jit_conv_conf_t ajcp;
        ajcp.is_relo = true;
        ajcp.nb_ic_int = 1;
        ajcp.is_nspc = true;
        ajcp.is_bf32 = jcp.is_bf32;
        ajcp.typesize_in = static_cast<int>(jcp.src_dsz);
        ajcp.ic_block_int = jcp.amx_w;

        ajcp.src_dt = jcp.src_dt;
        ajcp.ngroups = jcp.ngroups;
        ajcp.ic_without_padding = jcp.ic_without_padding;
        ajcp.ic = jcp.ic;
        ajcp.id = jcp.id;
        ajcp.ih = jcp.ih;
        ajcp.iw = jcp.iw;
        ajcp.kd = jcp.kd;
        ajcp.kh = jcp.kh;
        ajcp.kw = jcp.kw;
        ajcp.oc_block = jcp.oc_block;
        ajcp.ic_block = 16;
        ajcp.nb_oc = jcp.nb_oc;
        // let copy routine know that zero padding needs special handling
        // for legacy ISA without native s8s8 instructions
        ajcp.signed_input = jcp.s8s8_compensation_required;

        CHECK(safe_ptr_assign(copy_to_relo_pbuffer_,
                new jit_avx512_core_amx_copy_to_pbuffer_t(ajcp)));
        CHECK(copy_to_relo_pbuffer_->create_kernel());
    }

    if (is_relo_with_relo_weights) {
        jit_brgemm_relo_copy_to_wbuffer_t::cfg_t wjcp;
        wjcp.isa = jcp.isa;
        wjcp.wei_dt = jcp.wei_dt;
        wjcp.out_oc_block = static_cast<int>(jcp.oc_block);
        wjcp.inp_oc_block = 16;
        wjcp.rd = _pd->rd;
        wjcp.is_rd_padded_to_block = jcp.is_rd_padded_to_block;
        wjcp.inp_ocb_offs = KH * KW * jcp.ic * wjcp.inp_oc_block * wei_dsz;

        const auto oc_chunks = jcp.oc_block / wjcp.inp_oc_block;
        const auto inp_nb_oc = div_up(jcp.oc, wjcp.inp_oc_block);
        wjcp.last_occ_to_copy = inp_nb_oc - (jcp.nb_oc - 1) * oc_chunks;

        CHECK(safe_ptr_assign(copy_to_relo_wbuffer_,
                new jit_brgemm_relo_copy_to_wbuffer_t(wjcp)));
        CHECK(copy_to_relo_wbuffer_->create_kernel());
    }

    if (jcp.req_cal_comp_pad) {
        if (is_relo_with_relo_weights) {
            if (is_superset(isa, avx512_core))
                CHECK(safe_ptr_assign(comp_vpad_pbuffer_,
                        new jit_uni_brgemm_conv_relo_comp_pad_kernel_t<
                                Xbyak::Zmm>(jcp)));
            else {
                assert(one_of(isa, avx2_vnni, avx2_vnni_2));
                CHECK(safe_ptr_assign(comp_vpad_pbuffer_,
                        new jit_uni_brgemm_conv_relo_comp_pad_kernel_t<
                                Xbyak::Ymm>(jcp)));
            }
        } else {
            if (is_superset(isa, avx512_core))
                CHECK(safe_ptr_assign(comp_vpad_pbuffer_,
                        new jit_uni_brgemm_conv_comp_pad_kernel_t<Xbyak::Zmm>(
                                jcp)));
            else {
                assert(one_of(isa, avx2_vnni, avx2_vnni_2));
                CHECK(safe_ptr_assign(comp_vpad_pbuffer_,
                        new jit_uni_brgemm_conv_comp_pad_kernel_t<Xbyak::Ymm>(
                                jcp)));
            }
        }
        CHECK(comp_vpad_pbuffer_->create_kernel());
    }

    is_amx = brgemm_convolution_utils::is_amx(isa);

    for (const auto &key_value_pair : _pd->brg_indices) {
        const int brg_idx = key_value_pair.second;
        CHECK(add_brg_kernel(brg_idx));
    }

    for_(int i_N = N_begin; i_N < N_end; i_N++)
    for (int i_M = M_begin; i_M < M_end; i_M++) {
        // init "init" and "po" kernels for cases then we never call brgemm kernels
        // e.g. for d/h padded and dilated filter areas
        const bool filter_in_padding = jcp.f_pad >= EXT_KD
                || jcp.back_pad >= EXT_KD || jcp.t_pad >= EXT_KH
                || jcp.b_pad >= EXT_KH;
        // note: overly simplistic condition. Ideally, the condition would
        // only detect cases where there is strictly no overlap between the
        // input and filter.
        const bool dilate_no_overlap
                = jcp.dilate_d >= jcp.id || jcp.dilate_h >= jcp.ih;
        if (filter_in_padding || dilate_no_overlap) {
            auto M = (i_M) ? jcp.M_tail : jcp.M;
            CHECK(add_po_kernels(i_N, M, M));
        }
    }

    if (jcp.exec_type == exec_base) {
        // create brgemm kernels for ow_blocks with padded areas and
        // apply post-ops on final iteration by kw to padded areas in ow_block
        int kw_s {0}, kw_full_s {0}, kw_full_f {0}, kw_f {0}, ow_s {0},
                ow_f {0};
        for (dim_t ow = 0; ow < OW; ow += jcp.ow_block) {
            brgemm_convolution_utils::get_kw_range(jcp, static_cast<int>(ow),
                    kw_s, kw_full_s, kw_full_f, kw_f);
            bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_side = 0; i_side < 2; i_side++) {
                auto M = is_ow_tail ? jcp.M_tail : jcp.M;
                if (M <= 0) continue;
                brgemm_convolution_utils::get_ow_range(
                        jcp, static_cast<int>(ow), kw_s, ow_s, ow_f);
                const auto init_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                const auto kw_po = kw_f > kw_s ? kw_f - 1 : kw_s;
                brgemm_convolution_utils::get_ow_range(
                        jcp, static_cast<int>(ow), kw_po, ow_s, ow_f);
                const auto po_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                CHECK(add_po_kernels(i_N, init_bcast_dim, po_bcast_dim));
            }

            if (kw_f == jcp.kw && kw_s == 0) break;
        }

        for (dim_t ow = (jcp.nb_ow - 1) * jcp.ow_block; ow >= 0;
                ow -= jcp.ow_block) {
            brgemm_convolution_utils::get_kw_range(jcp, static_cast<int>(ow),
                    kw_s, kw_full_s, kw_full_f, kw_f);

            bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);

            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_side = 0; i_side < 2; i_side++) {
                auto M = is_ow_tail ? jcp.M_tail : jcp.M;
                if (M <= 0) continue;
                brgemm_convolution_utils::get_ow_range(
                        jcp, static_cast<int>(ow), kw_s, ow_s, ow_f);
                const auto init_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                const auto kw_po = kw_f > kw_s ? kw_f - 1 : kw_s;
                brgemm_convolution_utils::get_ow_range(
                        jcp, static_cast<int>(ow), kw_po, ow_s, ow_f);
                const auto po_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                CHECK(add_po_kernels(i_N, init_bcast_dim, po_bcast_dim));
            }

            if (kw_f == jcp.kw && kw_s == 0) break;
        }
    }

    // pre-calculated values
    if (jcp.exec_type == exec_vpad) {
        owb_kw_top_vpads.resize(jcp.nb_ow * jcp.kw);
        owb_kw_bottom_vpads.resize(jcp.nb_ow * jcp.kw);

        for (int owb = 0; owb < jcp.nb_ow; owb++) {
            const dim_t ow = owb * jcp.ow_block;
            const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
            const dim_t ow_b {ow},
                    ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
            const auto ow_l = ow_e - ow_b;
            MAYBE_UNUSED(ow_l);
            assert(0 <= ow_l && ow_l <= jcp.ow_block);
            const auto iiw_b = ow_b * SW - LP;
            const auto iiw_e = (ow_e - 1) * SW - LP + 1;
            const auto iiw_l = iiw_e - iiw_b;
            for (int kw = 0; kw < KW; kw++) {
                const auto iw = iiw_b + kw * DW;
                const auto top_vpad = iw >= 0 ? 0 : div_up(abs(iw), SW);
                const auto bottom_vpad
                        = iw + iiw_l <= IW ? 0 : div_up(iw + iiw_l - IW, SW);
                assert(top_vpad == 0 || bottom_vpad == 0);
                owb_kw_top_vpads[owb * KW + kw] = top_vpad;
                owb_kw_bottom_vpads[owb * KW + kw] = bottom_vpad;
            }
        }
    }

    if (jcp.req_cal_comp_pad && jcp.exec_type == exec_trans) {
        oh_kh_b.resize(jcp.oh);
        oh_kh_e.resize(jcp.oh);

        for (dim_t ohb = 0; ohb < jcp.nb_oh; ohb++) {
            const auto oh_begin = ohb * jcp.oh_block;
            const dim_t oh_end = nstl::min<dim_t>(OH, oh_begin + jcp.oh_block);
            for (dim_t oh = oh_begin; oh < oh_end; oh++) {
                const dim_t iih = ndims_pick(oh * SH - TP, oh * SH - TP, 0);
                const dim_t kh_s_ = div_up(nstl::max<dim_t>(0, -iih), DH);
                const dim_t kh_s = ndims_pick(kh_s_, kh_s_, 0);
                const dim_t kh_f_ = KH
                        - div_up(nstl::max<dim_t>(
                                         0, iih - IH + (KH - 1) * DH + 1),
                                DH);
                const dim_t kh_f = ndims_pick(kh_f_, kh_f_, 1);
                oh_kh_b[oh] = kh_s;
                oh_kh_e[oh] = kh_f;
            }
            for (dim_t oh = oh_begin; oh < oh_end; oh++) {
                const dim_t comp_oh_idx
                        = get_comp_oh(jcp.is_os_blocking ? oh_begin : oh);
                if (comp_oh_kh_b.size() - comp_oh_idx >= 0) {
                    const dim_t comp_oh_begin
                            = oh + comp_oh_kh_b.size() - comp_oh_idx;
                    const dim_t comp_oh_end
                            = jcp.is_os_blocking ? oh_end : oh + 1;
                    for (dim_t comp_oh = comp_oh_begin; comp_oh < comp_oh_end;
                            comp_oh++) {
                        comp_oh_kh_b.push_back(oh_kh_b[comp_oh]);
                        comp_oh_kh_e.push_back(oh_kh_e[comp_oh]);
                    }
                }
            }
        }
    }

    if (jcp.req_cal_comp_pad && jcp.exec_type != exec_vpad) {
        comp_owb.resize(jcp.nb_ow, -1);
        std::vector<dim_t> ow_kw_b(jcp.ow, -1);
        std::vector<dim_t> ow_kw_e(jcp.ow, -1);
        std::vector<dim_t> comp_ow_kw_s(jcp.comp_ow_size, -1);
        std::vector<dim_t> comp_ow_kw_f(jcp.comp_ow_size, -1);
        dim_t comp_ow_l = 0;

        for (dim_t ow = 0; ow < OW;) {
            assert(comp_ow_l <= jcp.comp_ow_size);
            const dim_t iiw = ow * SW - LP;
            const dim_t kw_s = div_up(nstl::max<dim_t>(0, -iiw), DW);
            const dim_t kw_f = KW
                    - div_up(nstl::max<dim_t>(0, iiw - IW + (KW - 1) * DW + 1),
                            DW);

            dim_t ow_e = ow;
            while (ow_e < OW) {
                const dim_t iiw_e = ow_e * SW - LP;
                const dim_t cur_kw_s = div_up(nstl::max<dim_t>(0, -iiw_e), DW);
                const dim_t cur_kw_f = KW
                        - div_up(nstl::max<dim_t>(
                                         0, iiw_e - IW + (KW - 1) * DW + 1),
                                DW);
                ow_kw_b[ow_e] = kw_s;
                ow_kw_e[ow_e] = kw_f;
                if (cur_kw_s != kw_s || cur_kw_f != kw_f) break;
                if (ow_e - ow < jcp.ow_block) {
                    comp_ow_kw_s[comp_ow_l] = kw_s;
                    comp_ow_kw_f[comp_ow_l] = kw_f;
                    comp_ow_l++;
                }
                ow_e++;
            }
            ow = ow_e;
        }

        for (int owb = 0; owb < jcp.nb_ow; owb++) {
            const dim_t ow_b = owb * jcp.ow_block;
            const dim_t ow_e = nstl::min<dim_t>(ow_b + jcp.ow_block, OW);
            const auto comp_ow_f = comp_ow_l - (ow_e - ow_b) + 1;
            MAYBE_UNUSED(comp_ow_f);

            if (owb == 0) {
                comp_owb[owb] = 0;
                continue;
            }

            for_(int i = 0; i < comp_ow_f; i++)
            for (dim_t ow = ow_b; ow <= ow_e; ow++) {
                if (ow == ow_e) {
                    comp_owb[owb] = i;
                    break;
                }
                if (comp_owb[owb] >= 0 || (i + ow - ow_b) >= comp_ow_l
                        || ow_kw_b[ow] != comp_ow_kw_s[i + ow - ow_b]
                        || ow_kw_e[ow] != comp_ow_kw_f[i + ow - ow_b])
                    break;
            }
            assert(comp_owb[owb] >= 0);
        }
    }

    // pre-calculate unique kernel combination
    if (jcp.req_cal_comp_pad) {
        std::set<std::vector<dim_t>> unique_kernels;
        size_t k = 0;
        kd_bs.resize(jcp.ker_ranges_size);
        kd_es.resize(jcp.ker_ranges_size);
        kh_bs.resize(jcp.ker_ranges_size);
        kh_es.resize(jcp.ker_ranges_size);
        kw_bs.resize(jcp.ker_ranges_size);
        kw_es.resize(jcp.ker_ranges_size);
        comp_oh.resize(jcp.ker_ranges_size);

        const auto update_kernels
                = [&](dim_t kd_b, dim_t kd_e, dim_t kh_b, dim_t kh_e,
                          dim_t kw_b, dim_t kw_e, dim_t oh = 0) {
            unique_kernels.insert({kd_b, kd_e, kh_b, kh_e, kw_b, kw_e, oh});
            if (k == unique_kernels.size()) return;
            kd_bs[k] = kd_b;
            kd_es[k] = kd_e;
            kh_bs[k] = kh_b;
            kh_es[k] = kh_e;
            kw_bs[k] = kw_b;
            kw_es[k] = kw_e;
            comp_oh[k] = oh;
            k++;
            assert(k <= static_cast<size_t>(jcp.ker_ranges_size));
        };

        for_(int od = 0; od < jcp.od; od++)
        for_(int ohb = 0; ohb < jcp.nb_oh; ohb++)
        for (int owb = 0; owb < jcp.nb_ow; owb++) {
            auto oh_begin = ohb * jcp.oh_block;
            auto oh_end = nstl::min<dim_t>(OH, oh_begin + jcp.oh_block);
            for (dim_t oh = oh_begin; oh < oh_end; oh++) {
                int kw_s {0}, kw_full_s {0}, kw_f {0}, kw_full_f {0};
                const dim_t ow = owb * jcp.ow_block;
                const dim_t iid = ndims_pick(od * SD - FP, 0, 0);
                const dim_t kd_s = ndims_pick(
                        div_up(nstl::max<dim_t>(0, -iid), DD), 0, 0);
                const dim_t kd_f = ndims_pick(KD
                                - div_up(nstl::max<dim_t>(0,
                                                 iid - ID + (KD - 1) * DD + 1),
                                        DD),
                        1, 1);
                const dim_t iih = ndims_pick(oh * SH - TP, oh * SH - TP, 0);
                const auto kh_s_ = div_up(nstl::max<dim_t>(0, -iih), DH);
                const auto kh_s = ndims_pick(kh_s_, kh_s_, 0);
                const auto kh_f_ = KH
                        - div_up(nstl::max<dim_t>(
                                         0, iih - IH + (KH - 1) * DH + 1),
                                DH);
                const auto kh_f = ndims_pick(kh_f_, kh_f_, 1);
                brgemm_convolution_utils::get_kw_range(jcp,
                        static_cast<int>(ow), kw_s, kw_full_s, kw_full_f, kw_f);
                if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s) {
                    if (jcp.exec_type != exec_trans) {
                        update_kernels(kd_s, kd_f, kh_s, kh_f, 0, KW);
                    } else {
                        const auto comp_oh_idx = jcp.is_os_blocking
                                ? get_comp_oh(oh_begin) + oh - oh_begin
                                : get_comp_oh(oh);
                        update_kernels(
                                kd_s, kd_f, kh_s, kh_f, 0, KW, comp_oh_idx);
                    }
                }
            }
        }
        ker_vpad_sz = k;
    }

    return status::success;
}
template <cpu_isa_t isa>
struct brgemm_convolution_fwd_t<isa>::brgemm_thread_ctx_t {
    brgemm_thread_ctx_t(const brgemm_exec_ctx_t &brgemm_ctx_, int ithr_,
            brgemm_batch_element_t *__restrict brg_batch_, char *c_buffer_,
            char *wsp_tile_, const char *__restrict weights_)
        : brgemm_ctx(brgemm_ctx_)
        , ithr(ithr_)
        , brg_batch(brg_batch_)
        , c_buffer(c_buffer_)
        , wsp_tile(wsp_tile_)
        , weights(weights_) {}

    const brgemm_exec_ctx_t &brgemm_ctx;
    int ithr {0};
    brgemm_batch_element_t *__restrict brg_batch {nullptr};
    char *__restrict c_buffer {nullptr};
    char *__restrict wsp_tile {nullptr};
    int cur_brg_idx {-1};
    dim_t g {-1}, n {-1}, ocb {-1};
    dim_t od {-1}, odb {-1}, oh {-1}, ohb {-1}, owb {-1};
    dim_t icc {-1};
    int32_t src_zp_val {0};
    int32_t *__restrict src_zp_comp_ptr {nullptr};
    const int32_t *__restrict dst_zp_vals {nullptr};
    int32_t *__restrict s8s8_comp_ptr {nullptr};
    const void *__restrict src_scales {nullptr};
    const void *__restrict wei_scales {nullptr};
    const void *__restrict dst_scales {nullptr};
    char *__restrict inp_buffer {nullptr};
    const char *__restrict input {nullptr};
    uint8_t *__restrict inp_buffer_mask {nullptr};
    const char *const __restrict weights {nullptr};
    void *__restrict inp_buffer_zero {nullptr};
    void *__restrict fp8_convert_wsp {nullptr};
};

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const int32_t *src_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC);
    const int32_t *dst_zero_points = CTX_IN_MEM(
            const int32_t *, DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *wei_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const auto &scratchpad = ctx.get_scratchpad_grantor();

    brgemm_exec_ctx_t brgemm_ctx(ctx, _pd);

    if (jcp.preconvert_fp8) {
        auto *src_f16 = scratchpad.template get<char>(key_conv_fp8_src);
        auto *wei_f16 = scratchpad.template get<char>(key_conv_fp8_weights);
        const auto src_nelems = memory_desc_wrapper(_pd->src_md()).nelems(true);
        const auto wei_nelems
                = memory_desc_wrapper(_pd->weights_md()).nelems(true);
        parallel(jcp.nthr, [&](int ithr, int nthr) {
            dim_t start = 0, end = 0;
            balance211(src_nelems, nthr, ithr, start, end);
            jit_brgemm_conv_src_copy_kernel_t::ctx_t p;
            p.src = brgemm_ctx.src + start;
            p.dst = src_f16 + start * jcp.src_dsz;
            p.nelems = end - start;
            (*fp8_src_copy_)(&p);
            balance211(wei_nelems, nthr, ithr, start, end);
            p.src = brgemm_ctx.weights + start;
            p.dst = wei_f16 + start * jcp.wei_dsz;
            p.nelems = end - start;
            (*fp8_weights_cvt_)(&p);
        });
        brgemm_ctx.src = src_f16;
        brgemm_ctx.weights = wei_f16;
    }

    const char *const __restrict src = brgemm_ctx.src;
    const char *__restrict wei = brgemm_ctx.weights;
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto extra_data_offset
            = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<char *>(brgemm_ctx.weights);
    const auto s8s8_comp_offset = jcp.req_cal_comp_pad
            ? jcp.ngroups * jcp.nb_oc * jcp.kd * jcp.kh * jcp.kw * jcp.oc_block
            : jcp.ngroups * jcp.nb_oc * jcp.oc_block;
    int32_t *s8s8_compensation = jcp.s8s8_compensation_required
            ? reinterpret_cast<int32_t *>(w + extra_data_offset)
            : nullptr;
    int32_t *zp_compensation = jcp.src_zero_point
            ? reinterpret_cast<int32_t *>(&w[extra_data_offset])
                    + (jcp.s8s8_compensation_required ? s8s8_comp_offset : 0)
            : nullptr;

    brgemm_batch_element_t *const __restrict brg_batch_global
            = brgemm_convolution_utils::uses_batch_elements(
                      jcp.brg_type, jcp.exec_type)
            ? scratchpad.template get<brgemm_batch_element_t>(
                      key_brgemm_primitive_batch)
            : nullptr;
    char *const __restrict c_buffer_global = (jcp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;

    auto inp_p_buffer = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<char>(key_conv_brgemm_inp_buffer)
            : nullptr;
    auto inp_p_buffer_mask = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<uint8_t>(key_conv_brgemm_inp_buffer_mask)
            : nullptr;
    int32_t *src_zp_comp_base = jcp.src_zero_point
            ? (jcp.req_cal_comp_pad ? scratchpad.template get<int32_t>(
                                              key_brgemm_primitive_zp_comp_a)
                                    : zp_compensation)
            : nullptr;
    int32_t *s8s8_comp_base = jcp.s8s8_compensation_required
            ? (jcp.req_cal_comp_pad ? scratchpad.template get<int32_t>(
                                              key_brgemm_primitive_buffer_comp)
                                    : s8s8_compensation)
            : nullptr;
    auto *fp8_convert_wsp_base = jcp.req_fp8_convert_wsp
            ? scratchpad.template get<char>(
                      key_brgemm_primitive_fp8_convert_wsp)
            : nullptr;

    cal_compensation(wei, src_zp_comp_base, s8s8_comp_base);

    char *const wsp_tile_global = is_amx
            ? scratchpad.template get<char>(key_conv_amx_tile_buffer)
            : nullptr;

    maybe_conv_weights(ctx, wei, wei);

    // --------------- Parallel section ------------------------------
    const dim_t work_amount = static_cast<dim_t>(jcp.mb) * jcp.ngroups
            * jcp.nb_oc * jcp.nb_od * jcp.nb_oh * jcp.nb_ow;
    // TODO: consider loop by icc be innermost because for current
    // implementation if we use buffer then we accumulate in it only on row
    // or made ic_chunks = 1 if use_buffer
    // or (looks more general) increase buffer size to store several rows

    parallel(jcp.nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        brgemm_batch_element_t *const __restrict brg_batch = brg_batch_global
                + static_cast<size_t>(ithr) * jcp.adjusted_batch_size;
        char *const __restrict c_buffer = (jcp.use_buffer)
                ? c_buffer_global + ithr * acc_dsz * jcp.buffer_size
                : nullptr;
        char *const wsp_tile = is_amx
                ? wsp_tile_global + ithr * jcp.amx_buf_size_per_thread
                : nullptr;

        brgemm_thread_ctx_t btc(
                brgemm_ctx, ithr, brg_batch, c_buffer, wsp_tile, wei);
        brgemm_thread_ctx_t last_btc = btc;

        float *dst_scales_inv_ptr = nullptr;
        if (jcp.with_dst_scales) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr
                    = scratchpad.template get<float>(key_conv_dst_scales)
                    + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        assert(IMPLICATION(!jcp.copy_input, !jcp.copy_block_only));
        btc.inp_buffer = (jcp.exec_type == exec_trans && jcp.copy_input)
                ? inp_p_buffer + src_dsz * ithr * jcp.inp_buffer_size
                : nullptr;

        btc.inp_buffer_mask = (jcp.exec_type == exec_trans)
                ? inp_p_buffer_mask + ithr * jcp.inp_buffer_mask_size
                : nullptr;

        btc.fp8_convert_wsp = jcp.req_fp8_convert_wsp
                ? fp8_convert_wsp_base + ithr * jcp.fp8_convert_wsp_size
                : nullptr;

        btc.input = jcp.copy_input ? btc.inp_buffer : src;

        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        dim_t n {0}, g {0}, ocb {0}, odb {0}, ohb {0}, owb {0};
        BRGEMM_CONV_ITERATOR_INIT;
        for (auto work = start; work < end; work++) {
            btc.g = g;
            btc.n = n;
            btc.ocb = ocb;
            btc.odb = odb;
            btc.ohb = ohb;
            btc.owb = owb;
            btc.src_zp_val = src_zero_points ? src_zero_points[0] : 0;
            btc.dst_zp_vals = dst_zero_points;
            btc.src_zp_comp_ptr
                    = jcp.src_zero_point ? src_zp_comp_base : nullptr;
            btc.s8s8_comp_ptr
                    = jcp.s8s8_compensation_required ? s8s8_comp_base : nullptr;
            btc.src_scales = src_scales;
            btc.wei_scales = wei_scales;
            btc.dst_scales = dst_scales_inv_ptr;

            if (jcp.exec_type == exec_trans
                    && (last_btc.n != n || last_btc.g != g)) {
                if (!jcp.copy_block_only)
                    std::memset(btc.inp_buffer_mask, false,
                            jcp.inp_buffer_mask_size);
            }
            auto od_begin = odb * jcp.od_block;
            auto od_end = nstl::min<dim_t>(OD, od_begin + jcp.od_block);
            auto oh_begin = ohb * jcp.oh_block;
            // if is_os_blocking is true then we do only one iteration of loop
            // by oh and process entire oh block in kernel call
            auto oh_end = jcp.is_os_blocking
                    ? oh_begin + 1
                    : nstl::min<dim_t>(OH, oh_begin + jcp.oh_block);
            for_(dim_t od = od_begin; od < od_end; od++)
            for_(dim_t oh = oh_begin; oh < oh_end; oh++)
            for (dim_t icc = 0; icc < _pd->ic_chunks; icc++) {
                btc.od = od;
                btc.oh = oh;
                btc.icc = icc;

                if (jcp.exec_type == exec_base) {
                    ker_base(btc);
                } else if (jcp.exec_type == exec_trans) {
                    maybe_conv_inp(btc, last_btc, src);
                    ker_trans(btc);
                } else if (jcp.exec_type == exec_vpad) {
                    ker_vpad(btc);
                } else
                    assert(!"Unknown exec type");
                last_btc.n = n;
                last_btc.g = g;
                last_btc.icc = icc;
                last_btc.odb = odb;
                last_btc.ohb = ohb;
                last_btc.owb = owb;
            }
            BRGEMM_CONV_ITERATOR_STEP;
        }
        if (is_amx) { amx_tile_release(); }
    });

    if (_pd->wants_zero_pad_dst()) ctx.memory(DNNL_ARG_DST)->zero_pad(ctx);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::cal_compensation(
        const char *__restrict weights, int32_t *src_zp_buffer,
        int32_t *s8s8_comp_buffer) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!jcp.req_cal_comp_pad) return status::success;

    // TODO: taking them by copy might affect the performance.
    std::vector<int> adjusted_k;
    std::vector<int> adjusted_k_l;
    int vpad_k = 0;
    const bool has_relo_large_spatial /* heuristic*/
            = is_relo_with_relo_weights && jcp.oc_block * jcp.ow > 10240;
    while (vpad_k < ker_vpad_sz) {
        int vpad_next_k = vpad_k + 1;
        while (!has_relo_large_spatial && vpad_next_k < ker_vpad_sz) {
            if (kd_bs[vpad_next_k] != kd_bs[vpad_k]
                    || kd_es[vpad_next_k] != kd_es[vpad_k]
                    || kh_bs[vpad_next_k] != kh_bs[vpad_k]
                    || kh_es[vpad_next_k] != kh_es[vpad_k]
                    || kw_bs[vpad_next_k] != kw_bs[vpad_k]
                    || kw_es[vpad_next_k] != kw_es[vpad_k])
                break;
            vpad_next_k++;
        }
        adjusted_k.push_back(vpad_k);
        adjusted_k_l.push_back(vpad_next_k - vpad_k);
        vpad_k = vpad_next_k;
    }

    const int max_ker_sz = static_cast<int>(adjusted_k.size());
    const auto comp_buffer_ow = jcp.exec_type != exec_vpad ? jcp.ow : 1;
    // TODO: revise the thread distribution here because the work_amount may be
    // insufficient
    // TODO: revise comp_vpad_pbuffer_ generator to avoid huge code for cases
    // with big ow
    const auto work_amount
            = static_cast<dim_t>(jcp.ngroups) * jcp.nb_oc * max_ker_sz;
    const auto is_small_shape = work_amount <= jcp.nthr
            && (work_amount * jcp.oc_block * jcp.icp * comp_buffer_ow
                    <= platform::get_per_core_cache_size(1));
    const int nthr = is_small_shape ? 1 : jcp.nthr;

    parallel(nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dim_t g {0}, ocb {0}, adj_k {0};
        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(
                start, g, jcp.ngroups, ocb, jcp.nb_oc, adj_k, max_ker_sz);
        for (auto work = start; work < end; work++) {
            const dim_t k {adjusted_k[adj_k]}, k_l {adjusted_k_l[adj_k]};
            const dim_t kd_bb {kd_bs[k]}, kd_ee {kd_es[k]}, kh_bb {kh_bs[k]},
                    kh_ee {kh_es[k]}, kw_bb {kw_bs[k]}, kw_ee {kw_es[k]};
            assert(kd_ee > kd_bb && kh_ee > kh_bb && kw_ee > kw_bb);

            const dim_t kd_b = maybe_invert_range(kd_bb, kd_ee, KD);
            const dim_t kd_e = maybe_invert_range(kd_ee, kd_bb, KD);
            const dim_t kh_b = maybe_invert_range(kh_bb, kh_ee, KH);
            const dim_t kh_e = maybe_invert_range(kh_ee, kh_bb, KH);
            const dim_t kw_b = maybe_invert_range(kw_bb, kw_ee, KW);
            const dim_t kw_e = maybe_invert_range(kw_ee, kw_bb, KW);

            const auto inp_oc_block
                    = is_relo_with_relo_weights ? 16 : jcp.oc_block;
            const auto wei_ocb = is_relo_with_relo_weights
                    ? ocb * div_up(jcp.oc_block, inp_oc_block)
                    : ocb;
            const auto nb_oc = is_relo_with_relo_weights
                    ? div_up(jcp.oc_block, inp_oc_block)
                    : jcp.nb_oc;
            const auto wei_offs = is_relo_with_relo_weights
                    ? (jcp.is_relo_wi() ? ((((g * nb_oc + wei_ocb) * KD) + kd_b)
                                                          * KH
                                                  + kh_b)
                                              * KW * jcp.ic * inp_oc_block
                                        : (((g * nb_oc + wei_ocb) * KH * KW)
                                                  + kh_b)
                                              * jcp.ic * inp_oc_block)
                    : g * _pd->wei_g_stride + wei_ocb * _pd->wei_ocb_stride
                            + kd_b * _pd->wei_kd_stride
                            + kh_b * _pd->wei_kh_stride
                            + kw_b * _pd->wei_kw_stride;
            const auto buffer_offs
                    = g * comp_ocb_sz + ocb * comp_ker_sz + k * comp_kw_sz;
            if (jcp.src_zero_point && src_zp_buffer)
                std::memset(&src_zp_buffer[buffer_offs], 0,
                        sizeof(int32_t) * comp_kw_sz);
            if (jcp.s8s8_compensation_required && s8s8_comp_buffer)
                std::memset(&s8s8_comp_buffer[buffer_offs], 0,
                        sizeof(int32_t) * comp_kw_sz);

            jit_brgemm_conv_comp_pad_args_t p;

            p.kd_l = kd_e - kd_b;
            p.kh_l = kh_e - kh_b;
            p.kw_l = kw_e - kw_b;
            p.ker_l = k_l;
            p.last_ocb = ocb == jcp.nb_oc - 1;
            p.use_inversion = _pd->desc()->use_inversion;
            p.ptr_in = &weights[wei_offs];
            p.ptr_zp_out = jcp.src_zero_point ? &src_zp_buffer[buffer_offs]
                                              : nullptr;
            p.ptr_cp_out = jcp.s8s8_compensation_required
                    ? &s8s8_comp_buffer[buffer_offs]
                    : nullptr;
            (*comp_vpad_pbuffer_)(&p);

            nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc, adj_k, max_ker_sz);
        }
    });
    return status::success;
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::perform_outwork(
        const brgemm_thread_ctx_t &btc, char *dst_base, const char *bias_w,
        dim_t ow, dim_t g_oc, bool is_oc_tail, dim_t ker_ow_s, dim_t ker_ow_f,
        dim_t kd_l, dim_t kh_l, bool maybe_do_init, bool do_postwork,
        size_t comp_ker_offs, bool do_post_comp) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const auto do_init
            = maybe_do_init && IMPLICATION(jcp.with_sum, jcp.use_buffer);
    if (!do_init && !do_postwork) return;

    const bool is_ow_tail = (OW - ow < jcp.ow_block);

    const auto M = is_ow_tail ? jcp.M_tail : jcp.M;
    const auto valid_kdh_l = kd_l > 0 && kh_l > 0;
    const auto ow_s = valid_kdh_l ? ker_ow_s : ow;
    const auto ow_f = valid_kdh_l ? ker_ow_f : ow;
    assert(ow <= ow_s && ow_s <= ow_f && ow_f <= ow + M);

    brgemm_kernel_post_ops_args_t p;
    if (do_postwork) {
        p.ptr_bias = (void *)(bias_w);
        p.ptr_binary_post_ops_rhs
                = btc.brgemm_ctx.post_ops_binary_rhs_arg_vec.data();
        p.dst_orig = btc.brgemm_ctx.dst;
        p.c_zp_values = btc.dst_zp_vals;
        p.a_comp_val = btc.src_zp_val;
        p.ptr_src_scales = btc.src_scales;
        p.ptr_wei_scales = btc.wei_scales
                ? static_cast<const char *>(btc.wei_scales)
                        + jcp.is_oc_scale * g_oc * sizeof(float)
                : nullptr;
        p.ptr_dst_scales = btc.dst_scales;
    }

    auto call_outwork_ker = [&](bool is_postwork, bool has_postcomp,
                                    dim_t ow_pw_s, dim_t ow_pw_l) {
        auto ker_po_idx = get_ker_po_idx(ow_pw_l - 1, is_postwork, is_oc_tail);
        const auto &outwork_ker = kernels_po_[ker_po_idx].get();
        assert(outwork_ker != nullptr
                && ow_pw_l == outwork_ker->get_bcast_dim());
        if (is_postwork) {
            p.apply_comp = has_postcomp;
            p.a_zp_compensation = has_postcomp && jcp.src_zero_point
                    ? &btc.src_zp_comp_ptr[comp_ker_offs
                              + (ow_pw_s - ow) * comp_ow_sz]
                    : btc.src_zp_comp_ptr;
            p.s8s8_compensation = has_postcomp && jcp.s8s8_compensation_required
                    ? &btc.s8s8_comp_ptr[comp_ker_offs
                              + (ow_pw_s - ow) * comp_ow_sz]
                    : btc.s8s8_comp_ptr;

            p.ptr_out = dst_base
                    + dst_dsz
                            * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                                    + ow_pw_s * jcp.oc_without_padding);
            p.ptr_in = static_cast<void *>(jcp.use_buffer
                            ? (btc.c_buffer
                                      + acc_dsz * (ow_pw_s - ow) * jcp.LDC)
                            : p.ptr_out);
        } else {
            p.apply_comp = has_postcomp;
            char *const ptr_Cz = jcp.use_buffer
                    ? (btc.c_buffer + acc_dsz * (ow_pw_s - ow) * jcp.LDC)
                    : dst_base
                            + dst_dsz
                                    * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                                            + ow_pw_s * jcp.oc_without_padding);
            p.ptr_out = static_cast<void *>(ptr_Cz);
        }
        (*outwork_ker)(&p);
    };

    if (ow < ow_s) {
        // left side
        const auto ow_pw_l = ow_s - ow;
        if (do_init) call_outwork_ker(false, false, ow, ow_pw_l);
        if (do_postwork) call_outwork_ker(true, do_post_comp, ow, ow_pw_l);
    }
    if (ow_f < ow + M) {
        // right side
        const auto ow_pw_l = ow + M - ow_f;
        if (do_init) call_outwork_ker(false, false, ow_f, ow_pw_l);
        if (do_postwork) call_outwork_ker(true, do_post_comp, ow_f, ow_pw_l);
    }
}

template <cpu_isa_t isa>
inline void brgemm_convolution_fwd_t<isa>::call_brgemm_kernel(
        const brgemm_thread_ctx_t &btc, const brgemm_kernel_t *brg_ker,
        int batch_size, char *ptr_C, char *ptr_D, const char *bias_w,
        dim_t g_oc, bool do_postops, size_t comp_ker_offs,
        bool do_only_comp) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    assert(brg_ker != nullptr);

    const auto do_only_pass_comp = !do_postops && jcp.src_zero_point
            && (jcp.req_brg_comp_pad || jcp.max_vpad > 0);
    const auto maybe_do_postops
            = one_of(true, do_postops, do_only_comp, do_only_pass_comp);

    assert(brgemm_convolution_utils::uses_batch_elements(
            jcp.brg_type, jcp.exec_type));
    const auto ptrA = btc.brg_batch[0].ptr.A;
    const auto ptrB = btc.brg_batch[0].ptr.B;

    if (maybe_do_postops) {
        const auto src_zp_ptr = jcp.src_zero_point
                ? &btc.src_zp_comp_ptr[comp_ker_offs]
                : nullptr;
        const auto s8s8_comp = jcp.s8s8_compensation_required
                ? &btc.s8s8_comp_ptr[comp_ker_offs]
                : nullptr;
        const brgemm_post_ops_data_t post_ops_data {
                static_cast<const char *>(bias_w),
                btc.brgemm_ctx.post_ops_binary_rhs_arg_vec.data(),
                static_cast<size_t>(g_oc), 0, btc.brgemm_ctx.dst, 0,
                static_cast<void *>(src_zp_ptr), nullptr, btc.dst_zp_vals,
                false, btc.src_zp_val, do_only_comp, do_only_pass_comp,
                btc.src_scales,
                btc.wei_scales ? static_cast<const char *>(btc.wei_scales)
                                + jcp.is_oc_scale * g_oc * sizeof(float)
                               : nullptr,
                btc.dst_scales};

        void *scratch = is_amx ? static_cast<void *>(btc.wsp_tile)
                : jcp.req_fp8_convert_wsp
                ? static_cast<void *>(btc.fp8_convert_wsp)
                : static_cast<void *>(s8s8_comp);

        if (do_postops)
            brgemm_kernel_execute_postops(brg_ker, batch_size, ptrA, ptrB,
                    btc.brg_batch, ptr_C, ptr_D, post_ops_data, scratch);
        else
            brgemm_kernel_execute_postops(brg_ker, batch_size, ptrA, ptrB,
                    btc.brg_batch, ptr_C, ptr_C, post_ops_data, scratch);
    } else {
        void *scratch = is_amx ? static_cast<void *>(btc.wsp_tile)
                               : static_cast<void *>(btc.fp8_convert_wsp);
        brgemm_kernel_execute(
                brg_ker, batch_size, ptrA, ptrB, btc.brg_batch, ptr_C, scratch);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::maybe_conv_weights(const exec_ctx_t &ctx,
        const char *__restrict input_weights,
        const char *__restrict &wei) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    wei = input_weights;
    if (!jcp.is_relo() || !jcp.relo_conv_weights) return;

    auto wei_buffer = ctx.get_scratchpad_grantor().template get<char>(
            key_conv_amx_wei_buffer);

    auto nb_rd = div_up(_pd->rd, jcp.vnni_block);
    if (jcp.is_rd_padded_to_block) nb_rd = rnd_up(nb_rd, 16);
    const auto inp_oc_block = 16; // this is oc block of inp weights layout

    assert(jcp.oc_block % inp_oc_block == 0);
    const auto oc_chunks = jcp.oc_block / inp_oc_block;
    const auto inp_nb_oc = div_up(jcp.oc, inp_oc_block);

    if (jcp.is_relo_whi()) {
        // reorder weights from (g)Owhi16o to (g)OR16r16o<vnni_granularity>r, where r := whi
        const auto inp_ocb_offs = _pd->rd * inp_oc_block * wei_dsz;
        const auto out_ocb_offs
                = nb_rd * jcp.oc_block * wei_dsz * jcp.vnni_block;

        parallel_nd(jcp.ngroups, jcp.nb_oc,
                [= COMPAT_THIS_CAPTURE](dim_t g, dim_t ocb) {
            auto p = jit_brgemm_relo_copy_to_wbuffer_t::ctx_t();
            const auto inp_ocb = g * inp_nb_oc + ocb * oc_chunks;
            const auto out_ocb = g * jcp.nb_oc + ocb;
            p.src = input_weights + inp_ocb * inp_ocb_offs;
            p.dst = wei_buffer + out_ocb * out_ocb_offs;
            p.last_ocb = (ocb == jcp.nb_oc - 1);
            (*copy_to_relo_wbuffer_)(&p);
        });
    } else if (jcp.is_relo_wi()) {
        // reorder weights from (g)Ohwi16o  to (g)OhR16r<oc_block>o<vnni_granularity>r, where r := wi
        const auto inp_kh_offs = _pd->rd * inp_oc_block * wei_dsz;
        const auto out_kh_offs
                = nb_rd * jcp.oc_block * wei_dsz * jcp.vnni_block;

        parallel_nd(jcp.ngroups, jcp.nb_oc, KH,
                [= COMPAT_THIS_CAPTURE](dim_t g, dim_t ocb, dim_t kh) {
            auto p = jit_brgemm_relo_copy_to_wbuffer_t::ctx_t();
            const auto inp_ocb = g * inp_nb_oc + ocb * oc_chunks;
            const auto out_ocb = g * jcp.nb_oc + ocb;
            p.src = input_weights + (inp_ocb * KH + kh) * inp_kh_offs;
            p.dst = wei_buffer + (out_ocb * KH + kh) * out_kh_offs;
            p.last_ocb = (ocb == jcp.nb_oc - 1);
            (*copy_to_relo_wbuffer_)(&p);
        });
    }
    wei = wei_buffer;
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::maybe_conv_inp(brgemm_thread_ctx_t &btc,
        const brgemm_thread_ctx_t &last_btc, const char *__restrict src) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    assert(IMPLICATION(!jcp.copy_input, !jcp.copy_block_only));
    if (!jcp.copy_input) return;

    const auto icb = btc.icc * jcp.nb_ic_blocking;

#define bmask(icb, odb, ohb, owb) \
    btc.inp_buffer_mask[(((icb) * jcp.nb_od + (odb)) * jcp.nb_oh + (ohb)) \
                    * jcp.nb_ow \
            + (owb)]

    if (jcp.copy_block_only) {
        if (last_btc.g == btc.g && last_btc.n == btc.n
                && last_btc.icc == btc.icc && last_btc.odb == btc.odb
                && last_btc.ohb == btc.ohb && last_btc.owb == btc.owb)
            return;
    } else {
        if (bmask(icb, btc.odb, btc.ohb, btc.owb)) return;
    }

    auto cp = jit_brgemm_conv_trans_kernel_args_t();

    const auto prev_odb
            = (jcp.copy_block_only || btc.odb == 0
                      || bmask(icb, btc.odb - 1, btc.ohb, btc.owb) == 0)
            ? false
            : true;

    const auto prev_ohb
            = (jcp.copy_block_only || btc.ohb == 0
                      || bmask(icb, btc.odb, btc.ohb - 1, btc.owb) == 0)
            ? false
            : true;

    const auto prev_odb_ohb
            = (jcp.copy_block_only
                      || (btc.odb > 0 && btc.ohb > 0
                              && bmask(icb, btc.odb - 1, btc.ohb - 1, btc.owb)
                                      == 0))
            ? false
            : true;

    const auto ic = icb * jcp.inp_ic_block;
    const auto g_ic = btc.g * jcp.ic + ic;
    const auto oh = btc.ohb * jcp.oh_block;
    const auto ow = btc.owb * jcp.ow_block;
    const dim_t iw = nstl::max<dim_t>(0, ow * SW - LP);

    dim_t id_start {0}, id_end {0}, ih_start {0}, ih_end {0};
    dim_t virt_id_start {0}, virt_id_end {0}, virt_ih_start {0},
            virt_ih_end {0};

    auto get_start_end
            = [](dim_t &start, dim_t &end, dim_t &virt_start, dim_t &virt_end,
                      dim_t b, dim_t bs, dim_t i, dim_t o, dim_t s, dim_t p,
                      dim_t k, dim_t d, bool prev) {
        const dim_t o_b = saturate<dim_t>(0, o, b * bs);
        const dim_t prev_o_b = saturate<dim_t>(0, o, (b - 1) * bs);
        const dim_t virt_cur_start = o_b * s - p;
        const dim_t cur_start = saturate<dim_t>(0, i, virt_cur_start);
        const dim_t virt_prev_start = prev_o_b * s - p;
        const dim_t i_bs = get_inp_size(i, bs, k, s, d);
        const dim_t virt_i_bs = calculate_end_padding(
                0, bs, 0, s, calculate_extended_filter_size(k, d));
        const dim_t virt_prev_end = prev ? virt_prev_start + virt_i_bs : -p;
        const dim_t prev_end = prev ? saturate<dim_t>(0, i, virt_prev_end) : 0;
        virt_start = nstl::max(virt_prev_end, virt_cur_start);
        start = nstl::max(prev_end, cur_start);
        virt_end = virt_cur_start + virt_i_bs;
        end = saturate<dim_t>(0, i, cur_start + i_bs);
    };
    get_start_end(id_start, id_end, virt_id_start, virt_id_end, btc.odb,
            jcp.od_block, nstl::min(ID, IDP - FP), OD, SD, FP, KD, DD - 1,
            prev_odb && prev_odb_ohb);
    get_start_end(ih_start, ih_end, virt_ih_start, virt_ih_end, btc.ohb,
            jcp.oh_block, nstl::min(IH, IHP - TP), OH, SH, TP, KH, DH - 1,
            prev_ohb && prev_odb_ohb);

    // how many real data rows to copy (including padding)
    const auto rows_to_copy = ih_end - ih_start;
    cp.owb = btc.owb;
    cp.ic = ic;
    const auto iw_buf = jcp.copy_block_only ? 0 : (ow * SW);
    dim_t inp_offset_start, out_offset_start;

    const auto base_ih_buf = (jcp.copy_block_only ? 0 : ih_start)
            + (jcp.is_relo_whi() ? 0 : TP);

    const memory_desc_wrapper src_d(pd()->src_md());
    const auto base_inp_offset_start = src_d.off_l(0)
            + static_cast<dim_t>(btc.n) * src_d.blk_off<false, true>(1)
            + iw * jcp.ngroups * jcp.ic_without_padding + g_ic;

    if (jcp.is_relo_whi()) {
        const auto base_out_offset_start
                = (jcp.copy_block_only
                                  ? 0
                                  : static_cast<dim_t>(icb) * _pd->pbuf_d_sz)
                + base_ih_buf * _pd->pbuf_w_sz
                + iw_buf * jcp.inp_ic_block * (jcp.is_relo_whi() ? KH : 1);

        // for relo_whi we copy top and bottom padded lines
        auto p = jit_conv_args_t();

        bool has_inp_buffer_overlap = true && last_btc.g == btc.g
                && last_btc.n == btc.n && last_btc.owb == btc.owb;

        for_(dim_t id = id_start; id < id_end; id++)
        for (dim_t doh = 0; doh < jcp.oh_block; doh++) {
            const dim_t ih_overlap = doh == 0
                    ? has_inp_buffer_overlap * nstl::max<dim_t>(0, KH - SH)
                    : 0;
            const dim_t kh_eff = jcp.kh - ih_overlap;

            const auto sdst = base_out_offset_start
                    + btc.ohb
                            * ((jcp.oh_block - 1) * _pd->pbuf_w_sz
                                    + jcp.stride_h * jcp.inp_ic_block)
                    + ih_overlap * jcp.inp_ic_block;

            const dim_t ih_s
                    = (doh + oh) * jcp.stride_h - jcp.t_pad + ih_overlap;
            const dim_t ih_e = ih_s + kh_eff;
            const dim_t ih = nstl::max<dim_t>(0, ih_s);
            p.t_overflow = nstl::max<dim_t>(0, -ih_s);
            p.b_overflow = nstl::min<dim_t>(
                    kh_eff, nstl::max<dim_t>(0, ih_e - jcp.ih));
            p.kh_padding
                    = nstl::max<dim_t>(0, kh_eff - p.t_overflow - p.b_overflow);
            p.kh_offset = kh_eff;

            const dim_t iw_s = ow * jcp.stride_w - jcp.l_pad;
            const dim_t iw_e = iw_s + jcp.iwp;
            p.f_overflow = nstl::max<dim_t>(0, -iw_s);
            p.back_overflow = nstl::max<dim_t>(0, iw_e - jcp.iw);
            p.kw_padding = nstl::max<dim_t>(
                    0, jcp.iwp - p.f_overflow - p.back_overflow);

            const auto first_actual_h = ih;
            inp_offset_start
                    = base_inp_offset_start + first_actual_h * src_w_sz;
            // inp_buffer has physical padding
            out_offset_start = sdst + doh * _pd->pbuf_w_sz;

            const auto inp_offset = inp_offset_start + id * src_h_sz;
            const auto id_buf = id - (jcp.copy_block_only ? id_start : 0) + FP;
            const auto out_offset = out_offset_start + id_buf * _pd->pbuf_h_sz;

            p.src = src + src_dsz * inp_offset;
            p.dst = btc.inp_buffer + src_dsz * out_offset;

            (*copy_to_relo_pbuffer_)(&p);
        }
    } else {
        const auto base_out_offset_start
                = (jcp.copy_block_only
                                  ? 0
                                  : static_cast<dim_t>(icb) * _pd->pbuf_d_sz)
                + base_ih_buf * _pd->pbuf_w_sz + iw_buf * jcp.inp_ic_block;
        // For os_blocking:
        // We have to zero top and bottom padding now
        // taking into account that batch size is always the same (kh_s is 0 for os_blocking)
        // TODO: extend M_mask (may be different for different kh) to avoid copying
        // top/bottom padded rows and avoid extra calculations in kernel
        // also for convolutions with pw == 0 the copy routine maybe not needed
        cp.t_pad = jcp.is_os_blocking ? nstl::max<dim_t>(0, -virt_ih_start) : 0;
        cp.b_pad = jcp.is_os_blocking ? nstl::max<dim_t>(0, virt_ih_end - IH)
                                      : 0;

        cp.h_count = nstl::max<dim_t>(0, rows_to_copy) + cp.t_pad + cp.b_pad;
        inp_offset_start = base_inp_offset_start + ih_start * src_w_sz;
        // inp_buffer has physical padding
        out_offset_start = base_out_offset_start - cp.t_pad * _pd->pbuf_w_sz;

        for (dim_t id = id_start; id < id_end; id++) {
            const auto inp_offset = inp_offset_start + id * src_h_sz;
            const auto id_buf = id - (jcp.copy_block_only ? id_start : 0) + FP;
            const auto out_offset = out_offset_start + id_buf * _pd->pbuf_h_sz;
            cp.src = src + src_dsz * inp_offset;
            cp.dst = btc.inp_buffer + src_dsz * out_offset;
            if (jcp.is_relo()) {
                if (jcp.vnni_block > 1) {
                    int size_to_sero = 0;
                    if (_pd->rd % jcp.vnni_block != 0)
                        size_to_sero = jcp.vnni_block;
                    if (_pd->rd > jcp.simd_w && _pd->rd % jcp.simd_w != 0)
                        size_to_sero = jcp.simd_w;
                    size_to_sero *= static_cast<int>(jcp.src_dsz);
                    void *const __restrict p_zeroing = (char *)cp.dst
                            + src_dsz * cp.h_count * _pd->pbuf_w_sz;
                    if (size_to_sero > 0 && btc.inp_buffer_zero != p_zeroing) {
                        std::memset(p_zeroing, 0, size_to_sero);
                        btc.inp_buffer_zero = p_zeroing;
                    }
                }
                const auto actual_iw_block = nstl::min(jcp.iw_block, IW - iw);
                if (actual_iw_block < jcp.iw_block) {
                    dim_t size_to_sero = 0;
                    const auto zero_elem_size
                            = (dim_t)src_dsz * jcp.inp_ic_block;
                    size_to_sero
                            = zero_elem_size * (jcp.iw_block - actual_iw_block);
                    for (size_t iih = 0; iih < cp.h_count; iih++) {
                        void *const __restrict p_zeroing = (char *)cp.dst
                                + (dim_t)src_dsz * iih * _pd->pbuf_w_sz
                                + zero_elem_size * actual_iw_block;
                        std::memset(p_zeroing, 0, size_to_sero);
                    }
                }
            }
            (*copy_to_pbuffer_)(&cp);
        }
    }

    if (!jcp.copy_block_only) bmask(icb, btc.odb, btc.ohb, btc.owb) = 1;

#undef bmask
}

#define BRGEMM_CONV_KER_HEADER \
    const char *const __restrict src = btc.brgemm_ctx.src; \
    const char *const __restrict weights = btc.weights; \
    const char *const __restrict bias = btc.brgemm_ctx.bias; \
    const dim_t oc = btc.ocb * jcp.oc_block; \
    const dim_t g_oc = btc.g * jcp.oc + oc; \
    const dim_t icb = btc.icc * jcp.nb_ic_blocking; \
    const dim_t ic = icb * jcp.ic_block; \
    const dim_t ow = btc.owb * jcp.ow_block; \
    const dim_t oh = btc.ohb * jcp.oh_block; \
    const dim_t iid = ndims_pick(btc.od * SD - FP, 0, 0); \
    const dim_t kd_s \
            = ndims_pick(div_up(nstl::max<dim_t>(0, -iid), DD), 0, 0); \
    const dim_t kd_f = ndims_pick(KD \
                    - div_up( \
                            nstl::max<dim_t>(0, iid - ID + (KD - 1) * DD + 1), \
                            DD), \
            1, 1); \
    const auto kd_l = kd_f - kd_s; \
    const auto adj_sh = jcp.is_relo_whi() ? 1 : SH; \
    const auto adj_tp = jcp.is_relo_whi() ? 0 : TP; \
    const auto iih = ndims_pick( \
            btc.oh * adj_sh - adj_tp, btc.oh * adj_sh - adj_tp, 0); \
    const auto kh_s_ = div_up(nstl::max<dim_t>(0, -iih), DH); \
    const auto kh_s = (jcp.is_os_blocking || jcp.is_relo_whi()) \
            ? 0 \
            : ndims_pick(kh_s_, kh_s_, 0); \
    const auto kh_f_ = KH \
            - div_up(nstl::max<dim_t>(0, iih - IH + (KH - 1) * DH + 1), DH); \
    const auto kh_f = jcp.is_relo_whi() ? 1 : ndims_pick(kh_f_, kh_f_, 1); \
    const auto kh_l = kh_f - kh_s; \
    const bool is_oc_tail = (jcp.oc - oc < jcp.oc_block); \
    const bool is_ic_tail = (btc.icc == _pd->ic_chunks - 1 \
            && ((jcp.ic - ic) % jcp.ic_block != 0)); \
    const bool is_ow_tail = (OW - ow < jcp.ow_block); \
    const bool is_oh_tail = (OH - oh < jcp.oh_block); \
    const char *const __restrict bias_w \
            = bias ? bias + (bias_d.blk_off(g_oc) * bia_dsz) : nullptr; \
    const auto nb_ic_b = nstl::min<dim_t>(jcp.nb_ic_blocking, jcp.nb_ic - icb) \
            - (is_ic_tail ? 1 : 0); \
    const memory_desc_wrapper dst_d(pd()->dst_md()); \
    char *const __restrict dst_base = btc.brgemm_ctx.dst \
            + dst_dsz \
                    * (dst_d.off_l(0) + btc.n * dst_d.blk_off<false, true>(1) \
                            + btc.g * dst_d.blk_off<false, true>(0, 1) \
                                    * jcp.oc \
                            + oc); \
    char *ptr_C; \
    char *ptr_D; \
    dim_t kd_b(0), kd_e(0), kh_b(0), kh_e(0), k_l(0), iiw_b(0);

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_base(brgemm_thread_ctx_t &btc) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims;

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(is_ow_tail);
    MAYBE_UNUSED(is_oh_tail);

    int kw_s {0}, kw_full_s {0}, kw_f {0}, kw_full_f {0};
    dim_t kw_b {0}, kw_e {0};
    int ow_b {0}, ow_e {0};

    brgemm_convolution_utils::get_kw_range(
            jcp, static_cast<int>(ow), kw_s, kw_full_s, kw_full_f, kw_f);

    const auto src_base = src + get_src_base_offset(btc, ic);
    const auto wei_base = weights
            + wei_dsz
                    * (btc.g * _pd->wei_g_stride
                            + btc.ocb * _pd->wei_ocb_stride);

    const auto call_brgemm = [&](int brg_idx, dim_t ic_block_s,
                                     dim_t n_ic_blocks, size_t comp_ker_offs,
                                     bool do_postops, bool do_only_comp) {
        if (brg_idx == -1) {
            assert(!"Requested brgemm kernel was not created.");
            return;
        }
        const auto brg_ker = brgemm_kernels_[brg_idx];
        brgemm_palettes_.maybe_tile_configure(is_amx, btc.cur_brg_idx, brg_idx);

        if (jcp.brg_type == brgemm_static_offs) {
            const void *ptrA {nullptr}, *ptrB {nullptr};
            _pd->get_A_B(btc.icc, src_base, wei_base, ic_block_s, iid, iih,
                    iiw_b, kd_b, kh_b, ptrA, ptrB);
            btc.brg_batch[0].ptr.A = ptrA;
            btc.brg_batch[0].ptr.B = ptrB;
        } else {
            _pd->init_batch(btc.icc, src_base, wei_base, n_ic_blocks,
                    ic_block_s, iid, iih, iiw_b, nullptr, nullptr, kd_b, kd_e,
                    kh_b, kh_e, kw_b, kw_e, k_l, btc.brg_batch);
            if (k_l <= 0) return;
        }
        call_brgemm_kernel(btc, brg_ker, static_cast<int>(k_l * n_ic_blocks),
                ptr_C, ptr_D, bias_w, g_oc, do_postops, comp_ker_offs,
                do_only_comp);
    };

    const auto kdhw_loop = [&]() {
        if (kw_e - kw_b <= 0) return;
        brgemm_convolution_utils::get_ow_range(
                jcp, static_cast<int>(ow), static_cast<int>(kw_b), ow_b, ow_e);
        const auto do_init
                = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s && kw_b == kw_s;
        const auto do_postwork = _pd->need_postwork
                && btc.icc == (_pd->ic_chunks - 1) && kd_e == kd_f
                && kh_e == kh_f && kw_e == kw_f;
        const auto do_post_comp = do_postwork && need_compensation;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        iiw_b = ow_b * SW - LP;
        ptr_D = dst_base
                + dst_dsz
                        * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                                + ow_b * jcp.oc_without_padding);
        ptr_C = (jcp.use_buffer)
                ? btc.c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                : static_cast<char *>(ptr_D);

        const auto ow_l = ow_e - ow_b;
        assert(0 <= ow_l && ow_l <= jcp.ow_block);

        if (ow_l > 0) {
            const size_t comp_ker_offs = do_postwork
                    ? get_comp_offset(btc.g, btc.ocb, 0, ow_b, kd_s, kd_f, kh_s,
                              kh_f, 0, KW)
                    : 0;

            if (nb_ic_b > 0) {
                const int brg_idx = _pd->get_brg_idx(ow_l, do_init, is_oc_tail,
                        false, static_cast<int>(kd_s), static_cast<int>(kd_f),
                        static_cast<int>(kh_s), static_cast<int>(kh_f));
                call_brgemm(brg_idx, 0, nb_ic_b, comp_ker_offs,
                        do_postwork && !is_ic_tail, false);
            }

            if (is_ic_tail) {
                const auto use_init_ker = (do_init && nb_ic_b == 0);
                const int brg_ic_tail_idx
                        = _pd->get_brg_idx(ow_l, use_init_ker, is_oc_tail, true,
                                static_cast<int>(kd_s), static_cast<int>(kd_f),
                                static_cast<int>(kh_s), static_cast<int>(kh_f));
                call_brgemm(brg_ic_tail_idx, nb_ic_b, 1, comp_ker_offs,
                        do_postwork, false);
            }
        }

        const auto post_comp_ker_offs = get_comp_offset(
                btc.g, btc.ocb, 0, ow, kd_s, kd_f, kh_s, kh_f, 0, KW);
        perform_outwork(btc, dst_base, bias_w, ow, g_oc, is_oc_tail, ow_b, ow_e,
                kd_l, kh_l, do_init, do_postwork, post_comp_ker_offs,
                do_post_comp);
    };

    if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s) {
        // kw values with left padding
        if (kw_s < kw_full_s) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min<dim_t>(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min<dim_t>(kh_f, kh_b + KH_BLOCK_PAD);
                    for (auto kw = kw_s; kw < kw_full_s; kw++) {
                        kw_b = kw;
                        kw_e = kw + 1;
                        kdhw_loop();
                    }
                }
            }
        }

        // kw values covering full ow_block
        if (kw_full_s < kw_full_f) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
                kd_e = nstl::min<dim_t>(kd_f, kd_b + KD_BLOCK);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                    kh_e = nstl::min<dim_t>(kh_f, kh_b + KH_BLOCK);
                    for (kw_b = kw_full_s; kw_b < kw_full_f; kw_b += KW_BLOCK) {
                        kw_e = nstl::min<dim_t>(kw_full_f, kw_b + KW_BLOCK);
                        kdhw_loop();
                    }
                }
            }
        }

        // kw values with right padding
        if (kw_full_f < kw_f) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min<dim_t>(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min<dim_t>(kh_f, kh_b + KH_BLOCK_PAD);
                    for (int kw = kw_full_f; kw < kw_f; kw++) {
                        kw_b = kw;
                        kw_e = kw + 1;
                        kdhw_loop();
                    }
                }
            }
        }
    } else {
        const auto do_init = btc.icc == 0;
        const auto do_postwork
                = _pd->need_postwork && btc.icc == (_pd->ic_chunks - 1);
        brgemm_convolution_utils::get_ow_range(
                jcp, static_cast<int>(ow), static_cast<int>(kw_b), ow_b, ow_e);
        perform_outwork(btc, dst_base, bias_w, ow, g_oc, is_oc_tail, ow_b, ow_e,
                kd_l, kh_l, do_init, do_postwork, 0, false);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_trans(brgemm_thread_ctx_t &btc) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims;

    BRGEMM_CONV_KER_HEADER;

    MAYBE_UNUSED(src);

    const auto wei_base = weights
            + wei_dsz
                    * (btc.g * _pd->wei_g_stride
                            + btc.ocb * _pd->wei_ocb_stride);
    const dim_t ow_b {ow},
            ow_e {ow + (is_ow_tail ? jcp.ow % jcp.ow_block : jcp.ow_block)};
    const dim_t oh_b {oh},
            oh_e {oh + (is_oh_tail ? jcp.oh % jcp.oh_block : jcp.oh_block)};
    const auto iid_shift = jcp.copy_block_only
            ? nstl::max<dim_t>(0, btc.odb * jcp.od_block * SD - FP)
            : 0;
    const auto iih_shift = jcp.copy_block_only
            ? nstl::max<dim_t>(0, btc.ohb * jcp.oh_block * adj_sh - adj_tp)
            : 0;
    const auto iiw_shift
            = jcp.copy_block_only ? (btc.owb * jcp.ow_block * SW) : 0;

    const auto iid_b = iid + FP - iid_shift;
    const auto iih_b = iih + adj_tp - iih_shift;
    iiw_b = ow_b * SW - iiw_shift;
    ptr_D = dst_base
            + dst_dsz
                    * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                            + ow_b * jcp.oc_without_padding);
    ptr_C = (jcp.use_buffer) ? btc.c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                             : static_cast<char *>(ptr_D);

    const auto ow_l = ow_e - ow_b;
    const auto oh_l = oh_e - oh_b;
    assert(0 <= ow_l && ow_l <= jcp.ow_block && 0 <= oh_l
            && oh_l <= jcp.oh_block);

    const auto ker_i = (jcp.is_os_blocking ? oh_l * ow_l : ow_l);
    const auto comp_iih = ndims_pick(btc.oh * SH - TP, btc.oh * SH - TP, 0);
    const auto comp_kh_s_ = div_up(nstl::max<dim_t>(0, -comp_iih), DH);
    const auto comp_kh_f_ = KH
            - div_up(
                    nstl::max<dim_t>(0, comp_iih - IH + (KH - 1) * DH + 1), DH);
    const auto comp_kh_s = ndims_pick(comp_kh_s_, comp_kh_s_, 0);
    const auto comp_kh_f = ndims_pick(comp_kh_f_, comp_kh_f_, 1);

    const auto call_brgemm
            = [&](int brg_idx, dim_t ic_block_s, dim_t n_ic_blocks,
                      size_t comp_ker_offs, bool do_postops) {
        if (brg_idx == -1) {
            assert(!"Requested brgemm kernel was not created.");
            return;
        }
        const auto brg_ker = brgemm_kernels_[brg_idx];
        brgemm_palettes_.maybe_tile_configure(is_amx, btc.cur_brg_idx, brg_idx);

        const auto pbuf_base = btc.input
                + src_dsz
                        * ((jcp.copy_block_only ? 0
                                                : ((icb + ic_block_s)
                                                          * _pd->pbuf_d_sz)))
                + (jcp.is_relo_whi() ? src_dsz * btc.ohb
                                        * ((jcp.oh_block - 1) * _pd->pbuf_w_sz
                                                + jcp.stride_h
                                                        * jcp.inp_ic_block)
                                     : 0);
        const void *ptrA {nullptr}, *ptrB {nullptr};

        if (jcp.brg_type == brgemm_static_offs) {
            _pd->get_A_B(btc.icc, pbuf_base, wei_base, ic_block_s, iid_b, iih_b,
                    iiw_b, kd_b, kh_b, ptrA, ptrB);
            btc.brg_batch[0].ptr.A = ptrA;
            btc.brg_batch[0].ptr.B = ptrB;
        } else {
            _pd->init_batch(btc.icc, pbuf_base, wei_base, n_ic_blocks,
                    ic_block_s, iid_b, iih_b, iiw_b, nullptr, nullptr, kd_b,
                    kd_e, kh_b, kh_e, 0, KW, k_l, btc.brg_batch);
            if (k_l <= 0) return;
        }

        call_brgemm_kernel(btc, brg_ker, static_cast<int>(k_l * n_ic_blocks),
                ptr_C, ptr_D, bias_w, g_oc, do_postops, comp_ker_offs, false);
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = _pd->need_postwork
                && btc.icc == (_pd->ic_chunks - 1) && kd_e == kd_f
                && kh_e == kh_f;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        const auto comp_ker_offs = do_postwork
                ? get_comp_offset(btc.g, btc.ocb, btc.oh, ow_b, kd_s, kd_f,
                          comp_kh_s, comp_kh_f, 0, KW)
                : 0;

        if (nb_ic_b > 0) {
            const int brg_idx = _pd->get_brg_idx(static_cast<int>(ker_i),
                    do_init, is_oc_tail, false, static_cast<int>(kd_s),
                    static_cast<int>(kd_f), static_cast<int>(kh_s),
                    static_cast<int>(kh_f));
            call_brgemm(brg_idx, 0, nb_ic_b, comp_ker_offs,
                    do_postwork && !is_ic_tail);
        }

        if (is_ic_tail) {
            const auto use_init_ker = (do_init && nb_ic_b == 0);
            const int brg_ic_tail_idx = _pd->get_brg_idx(
                    static_cast<int>(ker_i), use_init_ker, is_oc_tail, true,
                    static_cast<int>(kd_s), static_cast<int>(kd_f),
                    static_cast<int>(kh_s), static_cast<int>(kh_f));

            call_brgemm(
                    brg_ic_tail_idx, nb_ic_b, 1, comp_ker_offs, do_postwork);
        }
    };

    if (kd_f > kd_s && kh_f > kh_s) {
        // kw values covering full ow_block
        for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
            kd_e = nstl::min<dim_t>(kd_f, kd_b + KD_BLOCK);
            for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                kh_e = nstl::min<dim_t>(kh_f, kh_b + KH_BLOCK);
                kdhw_loop();
            }
        }
    } else {
        const auto do_init = btc.icc == 0;
        const auto do_postwork
                = _pd->need_postwork && btc.icc == (_pd->ic_chunks - 1);
        perform_outwork(btc, dst_base, bias_w, ow, g_oc, is_oc_tail, ow, ow,
                kd_l, kh_l, do_init, do_postwork, 0, false);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_vpad(brgemm_thread_ctx_t &btc) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims;

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(is_oh_tail);

    const char *const __restrict src_base = src + get_src_base_offset(btc, ic);
    const char *const __restrict wei_base = weights
            + wei_dsz
                    * (btc.g * _pd->wei_g_stride
                            + btc.ocb * _pd->wei_ocb_stride);

    const dim_t ow_b {ow}, ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
    iiw_b = ow_b * SW - LP;
    ptr_D = dst_base
            + dst_dsz
                    * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                            + ow_b * jcp.oc_without_padding);
    ptr_C = (jcp.use_buffer) ? btc.c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                             : static_cast<char *>(ptr_D);

    const auto ow_l = ow_e - ow_b;
    assert(0 <= ow_l && ow_l <= jcp.ow_block);
    const dim_t *const __restrict kw_top_vpads
            = owb_kw_top_vpads.data() + btc.owb * KW;
    const dim_t *const __restrict kw_bottom_vpads
            = owb_kw_bottom_vpads.data() + btc.owb * KW;

    const auto call_brgemm
            = [&](int brg_idx, dim_t ic_block_s, dim_t n_ic_blocks,
                      size_t comp_ker_offs, bool do_postops) {
        if (brg_idx < 0) {
            assert(!"Requested brgemm kernel was not created.");
            return;
        }
        const auto brg_ker = brgemm_kernels_[brg_idx];

        brgemm_palettes_.maybe_tile_configure(is_amx, btc.cur_brg_idx, brg_idx);

        assert(jcp.brg_type != brgemm_static_offs);
        _pd->init_batch(btc.icc, src_base, wei_base, n_ic_blocks, ic_block_s,
                iid, iih, iiw_b, kw_top_vpads, kw_bottom_vpads, kd_b, kd_e,
                kh_b, kh_e, 0, KW, k_l, btc.brg_batch);
        if (k_l <= 0) return;

        call_brgemm_kernel(btc, brg_ker, static_cast<int>(k_l * n_ic_blocks),
                ptr_C, ptr_D, bias_w, g_oc, do_postops, comp_ker_offs, false);
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = _pd->need_postwork
                && btc.icc == (_pd->ic_chunks - 1) && kd_e == kd_f
                && kh_e == kh_f;

        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        const auto comp_offs = get_comp_offset(
                btc.g, btc.ocb, 0, 0, kd_s, kd_f, kh_s, kh_f, 0, KW);

        if (nb_ic_b > 0) {
            const int brg_idx = _pd->get_brg_idx(static_cast<int>(ow_l),
                    do_init, is_oc_tail, false, static_cast<int>(kd_s),
                    static_cast<int>(kd_f), static_cast<int>(kh_s),
                    static_cast<int>(kh_f));
            call_brgemm(
                    brg_idx, 0, nb_ic_b, comp_offs, do_postwork && !is_ic_tail);
        }

        if (is_ic_tail) {
            const auto use_init_ker = (do_init && nb_ic_b == 0);
            const int brg_ic_tail_idx = _pd->get_brg_idx(static_cast<int>(ow_l),
                    use_init_ker, is_oc_tail, true, static_cast<int>(kd_s),
                    static_cast<int>(kd_f), static_cast<int>(kh_s),
                    static_cast<int>(kh_f));

            call_brgemm(brg_ic_tail_idx, nb_ic_b, 1, comp_offs, do_postwork);
        }
    };

    if (kd_f > kd_s && kh_f > kh_s) {
        // kw values covering full ow_block
        for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
            kd_e = nstl::min<dim_t>(kd_f, kd_b + KD_BLOCK);
            for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                kh_e = nstl::min<dim_t>(kh_f, kh_b + KH_BLOCK);
                kdhw_loop();
            }
        }
    } else {
        const auto do_init = btc.icc == 0;
        const auto do_postwork
                = _pd->need_postwork && btc.icc == (_pd->ic_chunks - 1);
        perform_outwork(btc, dst_base, bias_w, ow, g_oc, is_oc_tail, ow, ow,
                kd_l, kh_l, do_init, do_postwork, 0, false);
    }
}

#undef BRGEMM_CONV_KER_HEADER

template struct brgemm_convolution_fwd_t<avx2>;
template struct brgemm_convolution_fwd_t<avx2_vnni>;
template struct brgemm_convolution_fwd_t<avx2_vnni_2>;
template struct brgemm_convolution_fwd_t<avx512_core>;
template struct brgemm_convolution_fwd_t<avx512_core_vnni>;
template struct brgemm_convolution_fwd_t<avx512_core_bf16>;
template struct brgemm_convolution_fwd_t<avx512_core_fp16>;
template struct brgemm_convolution_fwd_t<avx512_core_amx>;
template struct brgemm_convolution_fwd_t<avx512_core_amx_fp16>;
template struct brgemm_convolution_fwd_t<avx10_2>;
template struct brgemm_convolution_fwd_t<avx10_2_amx_2>;
template struct brgemm_convolution_fwd_t<avx10_2_ace>;
} // namespace x64

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
