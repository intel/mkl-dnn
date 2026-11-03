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

#include <cassert>
#include <cstring>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/brgemm/brgemm_utils.hpp"
#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/jit_brgemm_conv.hpp"
#include "cpu/aarch64/jit_brgemm_conv_comp_pad_kernel.hpp"
#include "cpu/aarch64/jit_brgemm_conv_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

using namespace jit_sve_core_brgemm_conv_trans_kernel;
using namespace jit_uni_brgemm_conv_comp_pad_kernel;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::pd_t::init_batch(int icc,
        const char *src_base, const char *wei_base, int n_ic_blocks,
        int ic_block_s, int iid_b, int iih_b, int iiw_b,
        const dim_t *const __restrict kw_top_vpads,
        const dim_t *const __restrict kw_bottom_vpads, int kd_b, int kd_e,
        int kh_b, int kh_e, int kw_b, int kw_e, int k_l,
        brgemm_batch_element_t *brg_batch) const {
    const char *ptrA {nullptr};
    const char *ptrB {nullptr};
    const auto &jcp = jcp_;

    const int icb = icc * jcp.nb_ic_blocking;
    const int ic = icb * jcp.ic_block;

    for (int i_icb = 0; i_icb < n_ic_blocks; i_icb++) {
        const auto ic_off = (ic_block_s + i_icb) * jcp.ic_block;
        const auto wei_ic = ic + ic_off;
        const auto n_icb_off = i_icb * k_l;
        const auto src_base_shift = (jcp.exec_type == exec_trans)
                ? (jcp.copy_block_only ? 0 : (i_icb * pbuf_d_sz))
                : ic_off;
        const auto src_base_ic = src_base + src_base_shift * src_dsz;
        const auto wei_base_ic = wei_base + wei_ic * wei_ic_offset;
        const auto need_A_B = (jcp.use_uker
                && (jcp.brg_type == brgemm_offs
                        || jcp.brg_type == brgemm_static_offs));

        auto k = 0;
        for (int kd = kd_b; kd < kd_e; kd++) {
            const auto id = iid_b + kd * DD;
            const auto src_base_kd = src_base_ic + id * src_d_offset;
            const auto wei_kd = maybe_invert(kd, KD);
            const auto wei_base_kd = wei_base_ic + wei_kd * wei_kd_offset;
            for (int kh = kh_b; kh < kh_e; kh++) {
                const auto ih = (jcp.exec_type == exec_trans && jcp.kh_sets > 1)
                        ? iih_b
                        : (iih_b + kh * DH);
                const auto src_base_kh = src_base_kd + ih * adj_src_h_offset;
                const auto wei_kh = maybe_invert(kh, KH);
                const auto wei_base_kh = wei_base_kd + wei_kh * wei_kh_offset;

                for (int kw = kw_b; kw < kw_e; kw++) {
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
inline void brgemm_convolution_fwd_t<isa>::pd_t::get_A_B(int icc,
        const char *src_base, const char *wei_base, int ic_block_s, int iid_b,
        int iih_b, int iiw_b, int kd_b, int kh_b, const void *&ptrA,
        const void *&ptrB) const {
    const int icb = icc * jcp_.nb_ic_blocking;
    const int ic = icb * jcp_.ic_block;

    // for brgemm_static_offs we need only base A_addr and B_addr
    const auto ic_off = ic_block_s * jcp_.ic_block;
    const auto wei_ic = ic + ic_off;
    const auto src_base_shift = (jcp_.exec_type == exec_trans) ? 0 : ic_off;
    const auto src_base_ic = src_base + src_base_shift * src_dsz;
    const auto wei_base_ic = wei_base + wei_ic * wei_ic_offset;

    const auto id = iid_b + kd_b * DD;
    const auto src_base_kd = src_base_ic + id * src_d_offset;
    const auto wei_kd = maybe_invert(kd_b, KD);
    const auto wei_base_kd = wei_base_ic + wei_kd * wei_kd_offset;
    const auto has_kh_sets = (jcp_.exec_type == exec_trans && jcp_.kh_sets > 1);
    const auto ih = iih_b + (has_kh_sets ? 0 : kh_b * DH);
    const auto src_base_kh = src_base_kd + ih * adj_src_h_offset;
    const auto wei_kh = maybe_invert(kh_b, KH);
    const auto wei_base_kh = wei_base_kd + wei_kh * wei_kh_offset;

    ptrA = src_base_kh + iiw_b * src_iw_offset;
    const auto wei_kw = maybe_invert(0, KW);
    ptrB = wei_base_kh + wei_kw * wei_kw_offset;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::pd_t::add_brg_descriptor(int vM,
        int i_N, int i_K, int i_init, int kd_b, int kd_e, int kh_b, int kh_e) {

    const auto src_type = src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;

    const float alpha = 1.0;
    const float beta = 1.0;

    auto vbeta = (i_init) ? 0 : beta;
    auto vN = (i_N) ? jcp_.N_tail : jcp_.N;
    auto vK = (i_K) ? jcp_.K_tail : jcp_.K;
    auto vbrgM = jcp_.use_M_mask ? (vM == jcp_.M ? jcp_.brgM : jcp_.brgM_tail)
                                 : vM;
    auto brg_idx
            = get_brg_idx(vM - 1, i_init, i_N, i_K, kd_b, kd_e, kh_b, kh_e);
    // if brgemm_desc_t already created then skip this iteration
    if ((*brgemm_descriptors_)[brg_idx] != nullptr) return status::success;
    if (vN == 0 || vK == 0) return status::success;

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
                    bd_mask[ibrgM] = M_mask;
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
        const auto KH_SETS = jcp_.kh_sets;
        const auto KW_SETS = jcp_.kw_sets;

        assert(jcp_.exec_type == exec_trans);
        const auto kd_f = nstl::min(kd_e, kd_b + KD_BLOCK);
        const auto kh_f = nstl::min(kh_e, kh_b + KH_BLOCK);
        const auto k_l = (kd_f - kd_b) * (KH_SETS > 1 ? 1 : (kh_f - kh_b))
                * (KW_SETS > 1 ? 1 : KW);

        assert(jcp_.nb_ic % jcp_.nb_ic_blocking == 0);
        const auto nb_ic_blocks = jcp_.nb_ic_blocking;

        if (k_l > 0) {

            const auto kh_ee = KH_SETS > 1 ? kh_b + 1 : kh_f;
            const auto kw_e = KW_SETS > 1 ? 1 : KW;

            stoffs.resize(jcp_.max_batch + 1);

            init_batch(0, nullptr, nullptr, nb_ic_blocks, 0, 0, 0, 0, nullptr,
                    nullptr, kd_b, kd_f, kh_b, kh_ee, 0, kw_e, k_l,
                    stoffs.data());

        } else {
            // if k_l is 0 then it means the batchsize is 0
            return status::success;
        }
    }

    const auto kd_l = nstl::min(KD_BLOCK, kd_e - kd_b);
    const auto kh_l = nstl::min(KH_BLOCK, kh_e - kh_b);
    const auto bs = kd_l * kh_l * jcp_.kw;

    brgemm_desc_t brg;
    brgattr.bd_mask = bd_mask.data();
    brgattr.static_offsets = stoffs.data();
    brgemm_strides_t brg_strides;
    brg_strides.stride_a = jcp_.brg_stride_a;
    brg_strides.stride_b = jcp_.brg_stride_b;
    brg.req_cal_comp_pads = jcp_.req_brg_comp_pad
            && (jcp_.src_zero_point || jcp_.s8s8_compensation_required);
    const auto strides_ptr
            = (jcp_.brg_type == brgemm_strd) ? &brg_strides : nullptr;
    CHECK(brgemm_desc_init(&brg, isa, jcp_.brg_type, src_type, wei_type, false,
            false, brgemm_row_major, alpha, vbeta, jcp_.LDA, jcp_.LDB, jcp_.LDC,
            vbrgM, vN, vK, strides_ptr));

    // FIXME: use_uker is currently always false. We should add support or
    // remove the dead code.
    brgattr.use_uker = jcp_.use_uker;
    brgattr.use_interleave_stores = jcp_.use_interleave_stores;
    if (jcp_.use_mmla) brgemm_utils::set_mmla_attr(brgattr, vbrgM);
    brgattr.hint_prefetching = jcp_.hint_prefetching;
    brgattr.max_bs = bs;
    brgattr.hint_ununroll_bd_loop = jcp_.ununroll_bd_loop;
    brgattr.hint_innermost_loop = jcp_.brgemm_bd_loop_innermost
            ? brgemm_bd_loop_innermost
            : brgemm_innermost_undef;
    brgattr.hint_expected_A_size = 0;
    brgattr.hint_expected_B_size = 0;
    brgattr.hint_expected_C_size = 0;

    brgattr.wary_A_k_tail_read = false;
    brgattr.bd_mask_level = jcp_.use_M_mask;

    brgattr.max_top_vpad = jcp_.max_vpad;
    brgattr.max_bottom_vpad = jcp_.max_vpad;
    brgattr.fpmath_mode = attr()->fpmath_.mode_;
    brgattr.K_koef = (float)bs / KW;

    CHECK(brgemm_desc_set_attr(&brg, brgattr));

    auto LDD = jcp_.oc_without_padding;
    brg.with_sum = with_sum;
    brg.with_weights_scale_adjust = jcp_.scale_adjust_factor != 1.0f;
    CHECK(brgemm_desc_set_postops(&brg, attr(), &dst_md_, LDD, jcp_.bia_dt));

    brg.zp_a_compensation_per_m = jcp_.use_mmla_vpad_comp;

    CHECK(brgemm_desc_finalize(&brg));

    brgemm_descriptors_->insert(brg_idx, brg, bd_mask, stoffs);

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
    const auto wei_type = weights_md(0)->data_type;
    const auto dst_type = dst_md(0)->data_type;
    const bool is_int8 = one_of(src_type, u8, s8);

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
            | skip_mask_t::zero_points;
    if (is_int8) skip_mask |= skip_mask_t::scales;

    bool ok = is_fwd() && set_default_alg_kind(alg_kind::convolution_direct)
            && IMPLICATION(is_int8,
                    one_of(dst_type, s8, u8, s32, f32, bf16)
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

    CHECK(brgemm_convolution_utils::init_conf(jcp_, isa, *desc(), src_md_,
            weights_md_, dst_md_, bias_md_, attr_, dnnl_get_max_threads()));

    const auto adj_M = nstl::max(jcp_.M, jcp_.M_tail);

    // FIXME: use_uker is currently always false. We should add support or
    // remove the dead code.
    //
    // 1. The unrolled kernel can be used for exec_trans and exec_base.
    // For exec_base it makes sense to use unrolled kernel only if
    // there is no padding by width.
    // 2. For exec_trans block by kw is always KW
    assert(!jcp_.use_uker);
    assert(!jcp_.use_interleave_stores);

    bs_c = 0;

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
    src_d_sz = ID * src_h_sz;
    dst_w_sz = static_cast<dim_t>(OW) * jcp_.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;
    dst_d_sz = OD * dst_h_sz;

    wei_kw_stride = static_cast<dim_t>(jcp_.icp)
            * (jcp_.wei_plain ? jcp_.oc_without_padding : jcp_.oc_block);
    wei_kh_stride = KW * wei_kw_stride;
    wei_kd_stride = KH * wei_kh_stride;
    wei_ocb_stride = jcp_.wei_plain ? jcp_.oc_block : KD * wei_kd_stride;
    wei_g_stride = jcp_.wei_plain ? jcp_.oc : jcp_.nb_oc * wei_ocb_stride;
    wei_ic_stride = jcp_.wei_plain ? jcp_.oc_without_padding : jcp_.oc_block;

    const auto IC_BLOCK = jcp_.ic_block;
    const auto KH_SETS = jcp_.kh_sets;
    const auto KW_SETS = jcp_.kw_sets;

    if (jcp_.copy_block_only) {
        assert(jcp_.exec_type == exec_trans && "Missing copy kernel");
        const auto iw_block = jit_sve_core_brgemm_conv_trans_kernel_t::dst_w(
                jcp_, jcp_.ow_block);
        const auto ih_block = get_inp_size(IHP, jcp_.oh_block, KH, SH, DH - 1);
        const auto id_block = get_inp_size(IDP, jcp_.od_block, KD, SD, DD - 1);

        pbuf_w_sz = (dim_t)IC_BLOCK * KH_SETS * KW_SETS * iw_block;
        pbuf_h_sz = pbuf_w_sz * ih_block;
        pbuf_d_sz = pbuf_h_sz * id_block;

    } else {
        pbuf_w_sz = (dim_t)IC_BLOCK * KH_SETS * KW_SETS * IWP;
        pbuf_h_sz = pbuf_w_sz * IHP;
        pbuf_d_sz = pbuf_h_sz * IDP;
    }

    adj_src_h_sz = (jcp_.exec_type == exec_trans) ? pbuf_h_sz : src_h_sz;
    adj_src_h_offset
            = src_dsz * (jcp_.exec_type == exec_trans ? pbuf_w_sz : src_w_sz);

    src_iw_offset = static_cast<dim_t>(src_dsz)
            * (jcp_.exec_type == exec_trans
                            ? jcp_.ic_block * jcp_.kh_sets * jcp_.kw_sets
                            : jcp_.ngroups * jcp_.ic_without_padding);
    src_d_offset = static_cast<dim_t>(src_dsz) * adj_src_h_sz;
    wei_ic_offset = static_cast<dim_t>(wei_dsz) * wei_ic_stride;
    wei_kd_offset = static_cast<dim_t>(wei_dsz) * wei_kd_stride;
    wei_kh_offset = static_cast<dim_t>(wei_dsz) * wei_kh_stride
            * ((jcp_.exec_type == exec_trans && jcp_.kh_sets > 1) ? 0 : 1);
    wei_kw_offset = static_cast<dim_t>(wei_dsz) * wei_kw_stride;

    // FIXME: use_uker is currently always false. We should add support or
    // remove the dead code.
    if (jcp_.use_uker) {

        assert(KD % KD_BLOCK == 0);
        assert(KH % KH_BLOCK == 0);

        for (int iod = 0; iod < jcp_.od; iod++) {
            const int iid = iod * SD - FP;
            const int kd_s = div_up(max(0, -iid), DD);
            const int kd_f
                    = KD - div_up(max(0, iid - ID + (KD - 1) * DD + 1), DD);
            const auto kd_l = nstl::min(KD_BLOCK, kd_f - kd_s);
            for (int ioh = 0; ioh < jcp_.oh; ioh++) {

                const auto iih = ioh * SH - TP;
                const auto kh_s
                        = jcp_.is_os_blocking ? 0 : div_up(max(0, -iih), DH);
                const auto kh_f
                        = KH - div_up(max(0, iih - IH + (KH - 1) * DH + 1), DH);
                const auto kh_l = nstl::min(KH_BLOCK, kh_f - kh_s);
                const auto bs = kd_l * kh_l * jcp_.kw;
                if (bs <= 0) continue;

                const std::array<int, 4> key = {kd_s, kd_f, kh_s, kh_f};
                if (batchsizes.find(key) == batchsizes.end()) {
                    batchsizes.insert({key, bs_c});
                    bs_c++;
                }
            }
        }
    } else {
        batchsizes.insert({{0, KD, 0, KH}, bs_c});
        bs_c++;
    }

    brgs_sz_ = bs_c * adj_M * 2 * 2 * 2;
    brgemm_descriptors_->resize(brgs_sz_);

    const auto &p = attr()->post_ops_;

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

    // os_blocking is supported for exec_trans only
    assert(IMPLICATION(jcp_.exec_type != exec_trans, !jcp_.is_os_blocking));
    assert(IMPLICATION(jcp_.is_os_blocking,
            jcp_.os_block % jcp_.ow == 0 && jcp_.os_block / jcp_.ow <= jcp_.oh
                    && jcp_.os_block / jcp_.ow == jcp_.oh_block));

    ic_chunks = div_up(jcp_.nb_ic, jcp_.nb_ic_blocking);
    need_postwork = jcp_.with_bias || jcp_.with_eltwise || jcp_.with_binary
            || (one_of(src_type, u8, s8) && wei_type == s8) // oscales needed
            || (jcp_.dst_dt != jcp_.acc_dt) || jcp_.with_sum || jcp_.use_M_mask
            || jcp_.src_zero_point || jcp_.dst_zero_point;

    const int M_begin = 0;
    const int M_end = nstl::max(jcp_.M, jcp_.M_tail);
    const int N_begin = 0;
    const int N_end = (jcp_.N_tail == jcp_.N) ? 1 : 2;
    const int K_begin = 0;
    const int K_end = (jcp_.K_tail == 0) ? 1 : 2;
    const int i_init_begin
            = (IMPLICATION(jcp_.K_tail != 0, jcp_.K_tail == jcp_.K)
                      && jcp_.exec_type == exec_trans
                      && div_up(jcp_.nb_ic, jcp_.nb_ic_blocking) == 1
                      && KD_BLOCK == KD && KH_BLOCK == KH)
            ? 1
            : 0;
    int i_init_end = 2;

    for (int vM = M_end; vM > M_begin; vM--) {
        // init only needed brgemm descriptors
        if ((one_of(jcp_.exec_type, exec_trans, exec_vpad)
                    || (jcp_.exec_type == exec_base && jcp_.l_pad == 0
                            && jcp_.r_pad == 0))
                && vM != jcp_.M && vM != jcp_.M_tail)
            continue;
        for (const auto &key_value_pair : batchsizes) {
            const int kd_b = key_value_pair.first[0];
            const int kd_e = key_value_pair.first[1];
            const int kh_b = key_value_pair.first[2];
            const int kh_e = key_value_pair.first[3];
            for_(int i_init = i_init_begin; i_init < i_init_end; i_init++)
            for_(int i_N = N_begin; i_N < N_end; i_N++)
            for (int i_K = K_begin; i_K < K_end; i_K++) {
                CHECK(add_brg_descriptor(
                        vM, i_N, i_K, i_init, kd_b, kd_e, kh_b, kh_e));
            }
        }
    }

    auto scratchpad = scratchpad_registry().registrar();
    brgemm_convolution_utils::init_scratchpad(scratchpad, jcp_);
    if (need_postwork || jcp_.with_scales) {
        const size_t scales_size = jcp_.is_oc_scale
                ? nstl::max(static_cast<size_t>(OC()),
                          scale_utils::scales_simd_w(isa))
                : scale_utils::scales_simd_w(isa);
        scratchpad.template book<float>(key_precomputed_scales, scales_size,
                brgemm_convolution_utils::P4K);
    }
    if (!attr()->scales_.has_default_values(DNNL_ARG_DST))
        scratchpad.template book<float>(key_conv_dst_scales,
                scale_utils::scales_simd_w(isa), brgemm_convolution_utils::P4K);

    return status::success;
}

template <cpu_isa_t isa>
brgemm_convolution_fwd_t<isa>::brgemm_convolution_fwd_t(const pd_t *apd)
    : primitive_t(apd), bias_d(pd()->weights_md(1)) {}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::get_kw_range(
        int ow, int &kw_s, int &kw_full_s, int &kw_full_f, int &kw_f) const {
    // This function needed for exec_base only
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    // TODO: calculate these values instead direct loop by kw

    const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
    const auto M = is_ow_tail ? jcp.ow_tail : jcp.ow_block;
    kw_s = kw_full_s = kw_full_f = kw_f = -1;
    for (int kw = 0; kw < jcp.kw; kw++) {
        int ow_s {0}, ow_f {0};
        get_ow_range(ow, kw, ow_s, ow_f);
        if (ow_s < ow_f) {
            if (kw_s == -1) kw_s = kw;
            kw_f = kw + 1;
            if (ow_f - ow_s == M) {
                if (kw_full_s == -1) kw_full_s = kw;
                kw_full_f = kw + 1;
            }
        }
    }
    if (kw_f == -1) {
        kw_s = 0;
        kw_f = 0;
    }
    if (kw_full_f == -1) kw_full_s = kw_full_f = kw_f;
}

template <cpu_isa_t isa>
inline void brgemm_convolution_fwd_t<isa>::get_ow_range(
        int ow, int kw, int &ow_s, int &ow_f) const {
    // This function needed for exec_base only
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
    const auto M = is_ow_tail ? jcp.ow_tail : jcp.ow_block;

    const auto IW = jcp.iw;
    const auto SW = jcp.stride_w;
    const auto LP = jcp.l_pad;
    const auto DW = jcp.dilate_w + 1;

    const auto iiw = ow * SW - LP;
    auto iw_lp = iiw + kw * DW;
    const auto iw_rp = iw_lp + (M - 1) * SW - IW + 1;
    ow_s = ow;

    int ker_idx = 0;
    if (iw_lp < 0) {
        iw_lp = nstl::abs(iw_lp);
        ker_idx += div_up(iw_lp, SW);
        ow_s += ker_idx;
    }
    if (iw_rp > 0) ker_idx += div_up(iw_rp, SW);
    ow_f = ow_s + (M - ker_idx);
    ow_s = nstl::min(ow_s, ow + M);
    ow_f = nstl::min(nstl::max(ow_f, ow_s), ow + M);
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::add_brg_kernel(int M, int i_N, int i_K,
        int i_init, int kd_b, int kd_e, int kh_b, int kh_e) {
    if (M <= 0) return status::success;
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &brgs = *(_pd->brgemm_descriptors_);

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    auto K = (i_K) ? jcp.K_tail : jcp.K;
    if (N <= 0 || K <= 0) return status::success;
    auto brg_idx
            = _pd->get_brg_idx(M - 1, i_init, i_N, i_K, kd_b, kd_e, kh_b, kh_e);
    auto brg = brgs[brg_idx];
    if (!brgemm_kernels_[brg_idx] && brg && brg->bcast_dim > 0
            && brg->load_dim > 0 && brg->reduce_dim > 0) {
        CHECK(brgemm_kernels_.insert(brg_idx, brg));
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
    bcfg->alpha = !is_init && IMPLICATION(jcp.with_sum, jcp.use_buffer);
    bcfg->beta = is_init ? 0 : 1;
    CHECK(safe_ptr_assign(kernels_po_[ker_idx],
            new jit_brgemm_kernel_post_ops_t<isa>(jcp, *bcfg, *_pd->attr())));
    kernels_po_.at(ker_idx)->create_kernel();
    return status::success;
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::add_po_kernels(
        int i_N, int init_bcast_dim, int po_bcast_dim) {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &brgs = *(_pd->brgemm_descriptors_);

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    if (N <= 0) return;
    auto i_K = (jcp.K_tail > 0);

    const auto brg_idx = _pd->get_any_brg_idx(i_N, i_K);

    if (init_bcast_dim > 0) {
        if (brgs[brg_idx]) {
            auto init_cfg = *(brgs[brg_idx]);
            auto ker_init_idx = get_ker_po_idx(init_bcast_dim - 1, false, i_N);
            if (init_cfg.load_dim > 0 && kernels_po_.count(ker_init_idx) == 0) {
                init_cfg.bcast_dim = init_bcast_dim;
                add_po_kernel(&init_cfg, ker_init_idx, true);
            }
        }
    }

    if ((_pd->need_postwork || jcp.use_buffer) && po_bcast_dim > 0) {
        if (brgs[brg_idx]) {
            auto po_cfg = *(brgs[brg_idx]);
            auto ker_po_idx = get_ker_po_idx(po_bcast_dim - 1, true, i_N);
            if (po_cfg.load_dim > 0 && kernels_po_.count(ker_po_idx) == 0) {
                po_cfg.bcast_dim = po_bcast_dim;
                add_po_kernel(&po_cfg, ker_po_idx, false);
            }
        }
    }
}
template <cpu_isa_t isa>
int brgemm_convolution_fwd_t<isa>::get_comp_ker_idx(const int kd_b,
        const int kd_e, const int kh_b, const int kh_e, const int kw_b,
        const int kw_e) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!jcp.req_cal_comp_pad) return 0;

    assert(kd_e > kd_b && kh_e > kh_b);
    for (int k = 0; k < jcp.ker_ranges_size; k++) {
        if (kd_b == kd_bs[k] && kd_e == kd_es[k] && kh_b == kh_bs[k]
                && kh_e == kh_es[k] && kw_b == kw_bs[k] && kw_e == kw_es[k]) {
            return k;
        }
    }

    return -1;
}

template <cpu_isa_t isa>
inline dim_t brgemm_convolution_fwd_t<isa>::get_comp_offset(const int g,
        const int ocb, const int ow, const int kd_b, const int kd_e,
        const int kh_b, const int kh_e, const int kw_b, const int kw_e) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!jcp.src_zero_point && !jcp.s8s8_compensation_required) return 0;

    const auto comp_idx = get_comp_ker_idx(kd_b, kd_e, kh_b, kh_e, kw_b, kw_e);
    assert(IMPLICATION(jcp.req_cal_comp_pad, comp_idx >= 0));

    if (!jcp.req_cal_comp_pad) return (g * jcp.nb_oc + ocb) * jcp.oc_block;

    const auto ow_off = jcp.use_mmla_vpad_comp
            ? static_cast<dim_t>(ow) * jcp.oc_block
            : 0;
    return g * comp_group_stride + ocb * comp_ocb_stride
            + comp_idx * comp_range_stride + ow_off;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_fwd_t<isa>::init(engine_t *engine) {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

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
    src_d_sz = ID * src_h_sz;
    dst_w_sz = static_cast<dim_t>(OW) * jcp.oc_without_padding;
    dst_h_sz = OH * dst_w_sz;
    dst_d_sz = OD * dst_h_sz;

    comp_range_stride = static_cast<dim_t>(jcp.oc_block)
            * (jcp.use_mmla_vpad_comp ? OW : 1);
    comp_ocb_stride = jcp.ker_ranges_size * comp_range_stride;
    comp_group_stride = jcp.nb_oc * comp_ocb_stride;

    need_compensation = (jcp.src_zero_point || jcp.s8s8_compensation_required)
            && !jcp.req_brg_comp_pad;

    // ---- Initialize arrays ---------------------
    brgemm_kernels_.resize(_pd->brgs_sz_);

    // #TODO: this needed only if we have d/h padding more then kd/kh
    int M_begin = 0;
    int M_end = (jcp.M_tail == jcp.M) ? 1 : 2;
    int N_begin = 0;
    int N_end = (jcp.N_tail == jcp.N) ? 1 : 2;
    int K_begin = 0;
    int K_end = (jcp.K_tail == 0) ? 1 : 2;
    int i_init_begin = (IMPLICATION(jcp.K_tail != 0, jcp.K_tail == jcp.K)
                               && jcp.exec_type == exec_trans
                               && div_up(jcp.nb_ic, jcp.nb_ic_blocking) == 1
                               && KD_BLOCK == KD && KH_BLOCK == KH)
            ? 1
            : 0;
    int i_init_end = 2;

    if (jcp.exec_type == exec_trans) {
        CHECK(safe_ptr_assign(copy_to_pbuffer_,
                new jit_sve_core_brgemm_conv_trans_kernel_t(jcp)));
        CHECK(copy_to_pbuffer_->create_kernel());
    }

    if (jcp.req_cal_comp_pad) {
        CHECK(safe_ptr_assign(comp_vpad_pbuffer_,
                new jit_uni_brgemm_conv_comp_pad_kernel_t<isa>(jcp)));
        CHECK(comp_vpad_pbuffer_->create_kernel());
    }

    for (const auto &key_value_pair : _pd->batchsizes) {
        const int kd_b = key_value_pair.first[0];
        const int kd_e = key_value_pair.first[1];
        const int kh_b = key_value_pair.first[2];
        const int kh_e = key_value_pair.first[3];

        for_(int i_N = N_begin; i_N < N_end; i_N++)
        for_(int i_M = M_begin; i_M < M_end; i_M++)
        for_(int i_init = i_init_begin; i_init < i_init_end; i_init++)
        for (int i_K = K_begin; i_K < K_end; i_K++) {
            auto M = (i_M) ? jcp.M_tail : jcp.M;
            if (M <= 0) continue;
            add_brg_kernel(M, i_N, i_K, i_init, kd_b, kd_e, kh_b, kh_e);
        }
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
        if (IMPLICATION(jcp.exec_type == exec_trans,
                    filter_in_padding || dilate_no_overlap)) {
            auto M = (i_M) ? jcp.M_tail : jcp.M;
            add_po_kernels(i_N, M, M);
        }
    }

    if (jcp.exec_type == exec_base) {
        // create brgemm kernels for ow_blocks with padded areas and
        // apply post-ops on final iteration by kw to padded areas in ow_block
        int kw_s {0}, kw_full_s {0}, kw_full_f {0}, kw_f {0}, ow_s {0},
                ow_f {0};
        for (int ow = 0; ow < OW; ow += jcp.ow_block) {
            get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                get_ow_range(ow, kw, ow_s, ow_f);
                if (ow_f - ow_s <= 0) continue;

                auto M = ow_f - ow_s;
                if (M <= 0) continue;
                for (const auto &key_value_pair : _pd->batchsizes) {
                    const int kd_b = key_value_pair.first[0];
                    const int kd_e = key_value_pair.first[1];
                    const int kh_b = key_value_pair.first[2];
                    const int kh_e = key_value_pair.first[3];
                    for_(int i_init = 0; i_init < 2; i_init++)
                    for_(int i_N = 0; i_N < 2; i_N++)
                    for (int i_K = 0; i_K < 2; i_K++) {
                        add_brg_kernel(
                                M, i_N, i_K, i_init, kd_b, kd_e, kh_b, kh_e);
                    }
                }
            }

            bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_side = 0; i_side < 2; i_side++) {
                auto M = is_ow_tail ? jcp.M_tail : jcp.M;
                if (M <= 0) continue;
                get_ow_range(ow, kw_s, ow_s, ow_f);
                const auto init_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                get_ow_range(ow, kw_f - 1, ow_s, ow_f);
                const auto po_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                add_po_kernels(i_N, init_bcast_dim, po_bcast_dim);
            }

            if (kw_f == jcp.kw && kw_s == 0) break;
        }

        for (int ow = (jcp.nb_ow - 1) * jcp.ow_block; ow >= 0;
                ow -= jcp.ow_block) {
            get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                get_ow_range(ow, kw, ow_s, ow_f);
                if (ow_f - ow_s <= 0) continue;

                auto M = ow_f - ow_s;
                if (M <= 0) continue;
                for (const auto &key_value_pair : _pd->batchsizes) {
                    const int kd_b = key_value_pair.first[0];
                    const int kd_e = key_value_pair.first[1];
                    const int kh_b = key_value_pair.first[2];
                    const int kh_e = key_value_pair.first[3];
                    for_(int i_init = 0; i_init < 2; i_init++)
                    for_(int i_N = 0; i_N < 2; i_N++)
                    for (int i_K = 0; i_K < 2; i_K++) {
                        add_brg_kernel(
                                M, i_N, i_K, i_init, kd_b, kd_e, kh_b, kh_e);
                    }
                }
            }

            bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);

            for_(int i_N = 0; i_N < 2; i_N++)
            for (int i_side = 0; i_side < 2; i_side++) {
                auto M = is_ow_tail ? jcp.M_tail : jcp.M;
                if (M <= 0) continue;
                get_ow_range(ow, kw_s, ow_s, ow_f);
                const auto init_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                get_ow_range(ow, kw_f - 1, ow_s, ow_f);
                const auto po_bcast_dim
                        = (i_side == 0) ? (ow_s - ow) : (ow + M - ow_f);
                add_po_kernels(i_N, init_bcast_dim, po_bcast_dim);
            }

            if (kw_f == jcp.kw && kw_s == 0) break;
        }
    }

    // pre-calculated values
    if (jcp.exec_type == exec_vpad) {
        owb_kw_top_vpads.resize(jcp.nb_ow * jcp.kw);
        owb_kw_bottom_vpads.resize(jcp.nb_ow * jcp.kw);

        for (int owb = 0; owb < jcp.nb_ow; owb++) {
            const int ow = owb * jcp.ow_block;
            const bool is_ow_tail = (jcp.ow - ow < jcp.ow_block);
            const int ow_b {ow}, ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
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

    // pre-calculate unique kernel combination
    if (jcp.req_cal_comp_pad) {
        std::set<std::vector<int>> unique_kernels;
        size_t k = 0;
        kd_bs.resize(jcp.ker_ranges_size);
        kd_es.resize(jcp.ker_ranges_size);
        kh_bs.resize(jcp.ker_ranges_size);
        kh_es.resize(jcp.ker_ranges_size);
        kw_bs.resize(jcp.ker_ranges_size);
        kw_es.resize(jcp.ker_ranges_size);

        const auto update_kernels = [&](int kd_b, int kd_e, int kh_b, int kh_e,
                                            int kw_b, int kw_e) {
            unique_kernels.insert({kd_b, kd_e, kh_b, kh_e, kw_b, kw_e});
            if (k == unique_kernels.size()) return;
            kd_bs[k] = kd_b;
            kd_es[k] = kd_e;
            kh_bs[k] = kh_b;
            kh_es[k] = kh_e;
            kw_bs[k] = kw_b;
            kw_es[k] = kw_e;
            k++;
            assert(k <= static_cast<size_t>(jcp.ker_ranges_size));
        };

        for_(int odb = 0; odb < jcp.nb_od; odb++)
        for_(int ohb = 0; ohb < jcp.nb_oh; ohb++)
        for (int owb = 0; owb < jcp.nb_ow; owb++) {
            auto od_begin = odb * jcp.od_block;
            auto od_end = nstl::min(OD, od_begin + jcp.od_block);
            auto oh_begin = ohb * jcp.oh_block;
            auto oh_end = jcp.is_os_blocking
                    ? oh_begin + 1
                    : nstl::min(OH, oh_begin + jcp.oh_block);
            for_(int od = od_begin; od < od_end; od++)
            for (int oh = oh_begin; oh < oh_end; oh++) {
                int kw_s {0}, kw_full_s {0}, kw_f {0}, kw_full_f {0};
                const int ow = owb * jcp.ow_block;
                const int iid = ndims_pick(od * SD - FP, 0, 0);
                const int kd_s
                        = ndims_pick(div_up(nstl::max(0, -iid), DD), 0, 0);
                const int kd_f = ndims_pick(
                        KD - div_up(max(0, iid - ID + (KD - 1) * DD + 1), DD),
                        1, 1);
                const int iih = ndims_pick(oh * SH - TP, oh * SH - TP, 0);
                const auto kh_s_ = div_up(max(0, -iih), DH);
                const auto kh_s = ndims_pick(kh_s_, kh_s_, 0);
                const auto kh_f_
                        = KH - div_up(max(0, iih - IH + (KH - 1) * DH + 1), DH);
                const auto kh_f = ndims_pick(kh_f_, kh_f_, 1);
                get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);
                if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s) {
                    if (jcp.exec_type == exec_vpad) {
                        update_kernels(kd_s, kd_f, kh_s, kh_f, 0, KW);
                    } else if (jcp.exec_type == exec_base) {
                        if (kw_s < kw_full_s) {
                            for (auto kw = kw_s; kw < kw_full_s; kw++) {
                                update_kernels(
                                        kd_s, kd_f, kh_s, kh_f, kw, kw + 1);
                            }
                        }
                        if (kw_full_s < kw_full_f) {
                            for (auto kw = kw_full_s; kw < kw_full_f;
                                    kw += KW_BLOCK) {
                                const auto kw_e
                                        = nstl::min(kw_full_f, kw + KW_BLOCK);
                                update_kernels(
                                        kd_s, kd_f, kh_s, kh_f, kw, kw_e);
                            }
                        }
                        if (kw_full_f < kw_f) {
                            for (auto kw = kw_full_f; kw < kw_f; kw++) {
                                update_kernels(
                                        kd_s, kd_f, kh_s, kh_f, kw, kw + 1);
                            }
                        }
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
            char *wsp_tile_)
        : brgemm_ctx(brgemm_ctx_)
        , ithr(ithr_)
        , brg_batch(brg_batch_)
        , c_buffer(c_buffer_)
        , wsp_tile(wsp_tile_) {}

    const brgemm_exec_ctx_t &brgemm_ctx;
    int ithr {0};
    brgemm_batch_element_t *__restrict brg_batch {nullptr};
    char *c_buffer {nullptr};
    char *wsp_tile {nullptr};
    int cur_brg_idx {-1};
    int g {0}, n {0}, ocb {0};
    int od {0}, odb {0}, oh {0}, ohb {0}, owb {0};
    int icc = 0;
    const float *oscales {nullptr};
    int32_t src_zp_val {0};
    int32_t *src_zp_comp_ptr {nullptr};
    const int32_t *dst_zp_vals {nullptr};
    int32_t *s8s8_comp_ptr {nullptr};
    const float *dst_scales {nullptr};
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

    brgemm_exec_ctx_t brgemm_ctx(ctx, _pd);

    const char *const __restrict src = brgemm_ctx.src;
    const char *const __restrict wei = brgemm_ctx.weights;
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

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    const bool needs_oscales = _pd->need_postwork || jcp.with_scales;
    float *const oscales = needs_oscales
            ? scratchpad.template get<float>(key_precomputed_scales)
            : nullptr;
    float *const inv_dst_scales_precomputed
            = !pd()->attr()->scales_.has_default_values(DNNL_ARG_DST)
            ? scratchpad.template get<float>(key_conv_dst_scales)
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

    cal_compensation(wei, src_zp_comp_base, s8s8_comp_base);

    if (needs_oscales || inv_dst_scales_precomputed) {
        parallel(1, [= COMPAT_THIS_CAPTURE](const int, const int) {
            scale_utils::precompute_oscales(isa, oscales, pd()->attr(),
                    src_scales, wei_scales, pd()->OC(),
                    pd()->jcp_.scale_adjust_factor);
            scale_utils::precompute_inv_dst_scales(
                    isa, inv_dst_scales_precomputed, pd()->attr(), dst_scales);
        });
    }

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
        char *inp_buffer = (jcp.exec_type == exec_trans)
                ? inp_p_buffer + src_dsz * ithr * jcp.inp_buffer_size
                : nullptr;
        uint8_t *__restrict inp_buffer_mask = (jcp.exec_type == exec_trans)
                ? inp_p_buffer_mask + ithr * jcp.inp_buffer_mask_size
                : nullptr;
        char *const wsp_tile = nullptr;
        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        int n {0}, g {0}, ocb {0}, odb {0}, ohb {0}, owb {0};
        if (jcp.loop_order == loop_ndhwgc)
            nd_iterator_init(start, n, jcp.mb, odb, jcp.nb_od, ohb, jcp.nb_oh,
                    owb, jcp.nb_ow, g, jcp.ngroups, ocb, jcp.nb_oc);
        else if (jcp.loop_order == loop_ngcdhw)
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc,
                    odb, jcp.nb_od, ohb, jcp.nb_oh, owb, jcp.nb_ow);
        else
            assert(!"Unknown loop order");

        brgemm_thread_ctx_t btc(
                brgemm_ctx, ithr, brg_batch, c_buffer, wsp_tile);

        int last_n = -1;
        int last_g = -1;
        int last_icc = -1;
        int last_odb = -1;
        int last_ohb = -1;
        int last_owb = -1;
        for (auto work = start; work < end; work++) {
            btc.g = g;
            btc.n = n;
            btc.ocb = ocb;
            btc.odb = odb;
            btc.ohb = ohb;
            btc.owb = owb;
            btc.oscales = oscales;
            btc.src_zp_val = src_zero_points ? src_zero_points[0] : 0;
            btc.dst_zp_vals = dst_zero_points;
            btc.src_zp_comp_ptr
                    = jcp.src_zero_point ? src_zp_comp_base : nullptr;
            btc.s8s8_comp_ptr
                    = jcp.s8s8_compensation_required ? s8s8_comp_base : nullptr;
            btc.dst_scales = inv_dst_scales_precomputed;

            if (jcp.exec_type == exec_trans && (last_n != n || last_g != g)) {
                if (!jcp.copy_block_only)
                    std::memset(
                            inp_buffer_mask, false, jcp.inp_buffer_mask_size);
            }
            auto od_begin = odb * jcp.od_block;
            auto od_end = nstl::min(OD, od_begin + jcp.od_block);
            auto oh_begin = ohb * jcp.oh_block;
            // if is_os_blocking is true then we do only one iteration of loop
            // by oh and process entire oh block in kernel call
            auto oh_end = jcp.is_os_blocking
                    ? oh_begin + 1
                    : nstl::min(OH, oh_begin + jcp.oh_block);
            for_(int od = od_begin; od < od_end; od++)
            for (int oh = oh_begin; oh < oh_end; oh++) {
                for (int icc = 0; icc < _pd->ic_chunks; icc++) {
                    btc.od = od;
                    btc.oh = oh;
                    btc.icc = icc;

                    if (jcp.exec_type == exec_base) {
                        ker_base(btc);
                    } else if (jcp.exec_type == exec_trans) {
                        maybe_conv_inp(ithr, src, inp_buffer, inp_buffer_mask,
                                g, n, icc, odb, ohb, owb, last_g, last_n,
                                last_icc, last_odb, last_ohb, last_owb);
                        ker_trans(btc, inp_buffer);
                    } else if (jcp.exec_type == exec_vpad) {
                        ker_vpad(btc);
                    } else
                        assert(!"Unknown exec type");
                    last_n = n;
                    last_g = g;
                    last_icc = icc;
                    last_odb = odb;
                    last_ohb = ohb;
                    last_owb = owb;
                }
            }
            if (jcp.loop_order == loop_ndhwgc)
                nd_iterator_step(n, jcp.mb, odb, jcp.nb_od, ohb, jcp.nb_oh, owb,
                        jcp.nb_ow, g, jcp.ngroups, ocb, jcp.nb_oc);
            else if (jcp.loop_order == loop_ngcdhw)
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocb, jcp.nb_oc, odb,
                        jcp.nb_od, ohb, jcp.nb_oh, owb, jcp.nb_ow);
            else
                assert(!"Unknown loop order");
        }
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
    assert(IMPLICATION(jcp.use_mmla_vpad_comp,
            jcp.oc_block == brgemm_utils::mmla_n_block(jcp.isa)
                    && !jcp.s8s8_compensation_required));

    if (jcp.src_zero_point)
        std::memset(src_zp_buffer, 0, sizeof(int32_t) * jcp.comp_a_buffer_size);
    if (jcp.s8s8_compensation_required)
        std::memset(s8s8_comp_buffer, 0,
                sizeof(int32_t) * jcp.s8s8_comp_buffer_size);

    const auto work_amount
            = static_cast<dim_t>(jcp.ngroups) * jcp.nb_oc * ker_vpad_sz;
    const auto is_small_shape = !jcp.use_mmla_vpad_comp
            && work_amount <= jcp.nthr
            && (work_amount * jcp.oc_block * jcp.icp
                    <= platform::get_per_core_cache_size(1));
    const int nthr = is_small_shape ? 1 : jcp.nthr;

    parallel(nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        dim_t start {0}, end {0};
        int g {0}, ocb {0}, k {0};
        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(start, g, jcp.ngroups, ocb, jcp.nb_oc, k, ker_vpad_sz);
        for (auto work = start; work < end; work++) {
            const dim_t kd_bb {kd_bs[k]}, kd_ee {kd_es[k]}, kh_bb {kh_bs[k]},
                    kh_ee {kh_es[k]};
            assert(kd_ee > kd_bb && kh_ee > kh_bb);

            const auto kd_b = maybe_invert_range(kd_bb, kd_ee, KD);
            const auto kd_e = maybe_invert_range(kd_ee, kd_bb, KD);
            const auto kh_b = maybe_invert_range(kh_bb, kh_ee, KH);
            const auto kh_e = maybe_invert_range(kh_ee, kh_bb, KH);
            const auto base_buffer_offs = g * comp_group_stride
                    + ocb * comp_ocb_stride + k * comp_range_stride;

            const auto run_comp = [&](dim_t kw_bb, dim_t kw_ee, int32_t *zp_out,
                                          int32_t *cp_out) {
                assert(kw_ee > kw_bb);
                const auto kw_b = maybe_invert_range(kw_bb, kw_ee, KW);
                const auto kw_e = maybe_invert_range(kw_ee, kw_bb, KW);
                const auto wei_offs = g * _pd->wei_g_stride
                        + ocb * _pd->wei_ocb_stride + kd_b * _pd->wei_kd_stride
                        + kh_b * _pd->wei_kh_stride + kw_b * _pd->wei_kw_stride;

                jit_brgemm_conv_comp_pad_args_t p {};
                p.kd_l = kd_e - kd_b;
                p.kh_l = kh_e - kh_b;
                p.kw_l = kw_e - kw_b;
                p.ptr_in = &weights[wei_offs];
                p.ptr_zp_out = zp_out;
                p.ptr_cp_out = cp_out;
                (*comp_vpad_pbuffer_)(&p);
            };

            if (jcp.use_mmla_vpad_comp) {
                // Virtual padding changes the valid KW range by output column.
                // Reuse the full range for interior columns and recompute only
                // the boundary columns.
                constexpr int mmla_oc_block
                        = brgemm_utils::mmla_default_ld_block2_plain_a()
                        * cpu_isa_traits<isa>::vlen / sizeof(int32_t);
                alignas(64) int32_t full_kw_comp[mmla_oc_block] = {};
                run_comp(0, KW, full_kw_comp, nullptr);

                for (int ow = 0; ow < OW; ow++) {
                    const int iw = ow * SW - LP;
                    const dim_t kw_bb = div_up(nstl::max(0, -iw), DW);
                    const dim_t kw_ee = KW
                            - div_up(nstl::max(0, iw - IW + (KW - 1) * DW + 1),
                                    DW);
                    auto *const ow_comp = &src_zp_buffer[base_buffer_offs
                            + static_cast<dim_t>(ow) * jcp.oc_block];
                    if (kw_bb == 0 && kw_ee == KW) {
                        std::memcpy(ow_comp, full_kw_comp,
                                sizeof(int32_t) * jcp.oc_block);
                    } else if (kw_ee > kw_bb) {
                        run_comp(kw_bb, kw_ee, ow_comp, nullptr);
                    }
                }
            } else {
                run_comp(kw_bs[k], kw_es[k],
                        jcp.src_zero_point ? &src_zp_buffer[base_buffer_offs]
                                           : nullptr,
                        jcp.s8s8_compensation_required
                                ? &s8s8_comp_buffer[base_buffer_offs]
                                : nullptr);
            }

            nd_iterator_step(g, jcp.ngroups, ocb, jcp.nb_oc, k, ker_vpad_sz);
        }
    });
    return status::success;
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::perform_outwork(
        const brgemm_thread_ctx_t &btc, char *dst_base, const char *bias_w,
        int ow, int g_oc, bool is_oc_tail, int ker_ow_s, int ker_ow_f, int kd_l,
        int kh_l, bool maybe_do_init, bool do_postwork,
        bool do_post_comp) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const auto do_init
            = maybe_do_init && IMPLICATION(jcp.with_sum, jcp.use_buffer);
    if (!do_init && !do_postwork) return;

    assert(!jcp.is_os_blocking);

    const bool is_ow_tail = (OW - ow < jcp.ow_block);

    const auto M = is_ow_tail ? jcp.M_tail : jcp.M;
    const auto kdh_l = kd_l * kh_l;
    const auto ow_s = (kdh_l <= 0) ? ow : ker_ow_s;
    const auto ow_f = (kdh_l <= 0) ? ow : ker_ow_f;
    assert(ow <= ow_s && ow_s <= ow_f && ow_f <= ow + M);

    brgemm_kernel_post_ops_t p;
    if (do_postwork) {
        p.ptr_bias = (void *)(bias_w);
        p.ptr_scales = (void *)(&btc.oscales[jcp.is_oc_scale * g_oc]);
        p.ptr_binary_post_ops_rhs
                = btc.brgemm_ctx.post_ops_binary_rhs_arg_vec.data();
        p.dst_orig = btc.brgemm_ctx.dst;
        p.c_zp_values = btc.dst_zp_vals;
        p.a_comp_val = btc.src_zp_val;
        p.ptr_dst_scales = (void *)btc.dst_scales;
    }

    auto call_outwork_ker = [&](bool is_postwork, bool has_postcomp,
                                    int ow_pw_s, int ow_pw_l) {
        auto ker_po_idx = get_ker_po_idx(ow_pw_l - 1, is_postwork, is_oc_tail);
        const auto outwork_ker = kernels_po_.at(ker_po_idx).get();
        assert(outwork_ker != nullptr && ow_pw_l == outwork_ker->brg.bcast_dim);
        if (is_postwork) {
            p.apply_comp = has_postcomp;
            p.a_zp_compensation = has_postcomp && jcp.src_zero_point
                    ? &btc.src_zp_comp_ptr[ow_pw_s * jcp.LDB]
                    : btc.src_zp_comp_ptr;
            p.s8s8_compensation = has_postcomp && jcp.s8s8_compensation_required
                    ? &btc.s8s8_comp_ptr[ow_pw_s * jcp.LDB]
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
        int batch_size, char *ptr_C, char *ptr_D, const char *bias_w, int g_oc,
        bool do_postops, dim_t comp_ker_offs, bool do_only_comp) const {

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
                &btc.oscales[jcp.is_oc_scale * g_oc],
                btc.brgemm_ctx.post_ops_binary_rhs_arg_vec.data(),
                static_cast<size_t>(g_oc), 0, btc.brgemm_ctx.dst, 0,
                static_cast<void *>(src_zp_ptr), nullptr, btc.dst_zp_vals,
                false, btc.src_zp_val, do_only_comp, do_only_pass_comp,
                btc.dst_scales};

        void *scratch = static_cast<void *>(s8s8_comp);

        if (do_postops)
            brgemm_kernel_execute_postops(brg_ker, batch_size, ptrA, ptrB,
                    btc.brg_batch, ptr_C, ptr_D, post_ops_data, scratch);
        else
            brgemm_kernel_execute_postops(brg_ker, batch_size, ptrA, ptrB,
                    btc.brg_batch, ptr_C, ptr_C, post_ops_data, scratch);
    } else
        brgemm_kernel_execute(brg_ker, batch_size, ptrA, ptrB, btc.brg_batch,
                ptr_C, static_cast<void *>(btc.wsp_tile));
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::maybe_conv_inp(int ithr,
        const char *__restrict src, char *__restrict inp_buffer,
        uint8_t *__restrict inp_buffer_mask, int g, int n, int icc, int odb,
        int ohb, int owb, int last_g, int last_n, int last_icc, int last_odb,
        int last_ohb, int last_owb) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto icb = icc * jcp.nb_ic_blocking;

#define bmask(icb, odb, ohb, owb) \
    inp_buffer_mask[(((icb) * jcp.nb_od + (odb)) * jcp.nb_oh + (ohb)) \
                    * jcp.nb_ow \
            + (owb)]

    if (jcp.copy_block_only) {
        if (last_g == g && last_n == n && last_icc == icc && last_odb == odb
                && last_ohb == ohb && last_owb == owb)
            return;
    } else {
        if (bmask(icb, odb, ohb, owb)) return;
    }

    auto cp = jit_brgemm_conv_trans_kernel_args_t();

    const auto prev_odb = (jcp.copy_block_only || odb == 0
                                  || bmask(icb, odb - 1, ohb, owb) == 0)
            ? false
            : true;

    const auto prev_ohb = (jcp.copy_block_only || ohb == 0
                                  || bmask(icb, odb, ohb - 1, owb) == 0)
            ? false
            : true;

    const auto prev_odb_ohb
            = (jcp.copy_block_only
                      || (odb > 0 && ohb > 0
                              && bmask(icb, odb - 1, ohb - 1, owb) == 0))
            ? false
            : true;

    const auto ic = icb * jcp.ic_block;
    const auto g_ic = g * jcp.ic + ic;
    const auto oh = ohb * jcp.oh_block;
    const auto ow = owb * jcp.ow_block;
    const auto iw = nstl::max(0, ow * SW - LP);

    int id_start {0}, id_end {0}, ih_start {0}, ih_end {0};
    int virt_id_start {0}, virt_id_end {0}, virt_ih_start {0}, virt_ih_end {0};

    auto get_start_end = [](int &start, int &end, int &virt_start,
                                 int &virt_end, int b, int bs, int i, int o,
                                 int s, int p, int k, int d, bool prev) {
        const auto o_b = saturate(0, o, b * bs);
        const auto prev_o_b = saturate(0, o, (b - 1) * bs);
        const auto virt_cur_start = o_b * s - p;
        const auto cur_start = saturate(0, i, virt_cur_start);
        const auto virt_prev_start = prev_o_b * s - p;
        const auto i_bs = get_inp_size(i, bs, k, s, d);
        const auto virt_i_bs = calculate_end_padding(
                0, bs, 0, s, calculate_extended_filter_size(k, d));
        const auto virt_prev_end = prev ? virt_prev_start + virt_i_bs : -p;
        const auto prev_end = prev ? saturate(0, i, virt_prev_end) : 0;
        virt_start = nstl::max(virt_prev_end, virt_cur_start);
        start = nstl::max(prev_end, cur_start);
        virt_end = virt_cur_start + virt_i_bs;
        end = saturate(0, i, cur_start + i_bs);
    };
    get_start_end(id_start, id_end, virt_id_start, virt_id_end, odb,
            jcp.od_block, nstl::min(ID, IDP - FP), OD, SD, FP, KD, DD - 1,
            prev_odb && prev_odb_ohb);
    get_start_end(ih_start, ih_end, virt_ih_start, virt_ih_end, ohb,
            jcp.oh_block, nstl::min(IH, IHP - TP), OH, SH, TP, KH, DH - 1,
            prev_ohb && prev_odb_ohb);

    // how many real data rows to copy (including padding)
    const auto rows_to_copy = ih_end - ih_start;
    cp.owb = owb;
    cp.ic = ic;
    const auto iw_buf = jcp.copy_block_only ? 0 : (ow * SW);
    dim_t inp_offset_start, out_offset_start;

    for (int kh = 0; kh < jcp.kh_sets; kh++) {
        if (jcp.kh_sets > 1) {
            assert(!jcp.is_os_blocking);
            const auto ih_s = oh * SH + kh * DH - TP;
            const auto ih_f = (oh + jcp.oh_block - 1) * SH + kh * DH - TP + 1;

            cp.t_pad = max(0, -ih_s);
            cp.b_pad = max(0, ih_f - jcp.ih);
            cp.h_count = max(0, jcp.oh_block);
            const auto ih_buf = (jcp.copy_block_only ? 0 : ih_start) + TP;

            inp_offset_start = static_cast<dim_t>(n) * src_d_sz
                    + max(ih_s, ih_start) * src_w_sz
                    + iw * jcp.ngroups * jcp.ic_without_padding + g_ic;

            // inp_buffer has physical padding
            out_offset_start = (jcp.copy_block_only ? 0
                                                    : static_cast<dim_t>(icb)
                                                       * _pd->pbuf_d_sz)
                    + ih_buf * _pd->pbuf_w_sz
                    + (iw_buf * jcp.kh_sets + kh) * jcp.kw_sets * jcp.ic_block;
        } else {
            // For os_blocking:
            // We have to zero top and bottom padding now
            // taking into account that batch size is always the same (kh_s is 0 for os_blocking)
            // TODO: extend M_mask (may be different for different kh) to avoid copying
            // top/bottom padded rows and avoid extra calculations in kernel
            // also for convolutions with pw == 0 the copy routine maybe not needed
            cp.t_pad = jcp.is_os_blocking ? max(0, -virt_ih_start) : 0;
            cp.b_pad = jcp.is_os_blocking ? max(0, virt_ih_end - IH) : 0;
            cp.h_count = max(0, rows_to_copy) + cp.t_pad + cp.b_pad;
            const auto ih_buf
                    = (jcp.copy_block_only ? 0 : ih_start) + TP - cp.t_pad;

            inp_offset_start = static_cast<dim_t>(n) * src_d_sz
                    + ih_start * src_w_sz
                    + iw * jcp.ngroups * jcp.ic_without_padding + g_ic;

            // inp_buffer has physical padding
            out_offset_start = (jcp.copy_block_only ? 0
                                                    : static_cast<dim_t>(icb)
                                                       * _pd->pbuf_d_sz)
                    + ih_buf * _pd->pbuf_w_sz
                    + iw_buf * jcp.ic_block * jcp.kh_sets * jcp.kw_sets;
        }

        for (int id = id_start; id < id_end; id++) {
            const auto inp_offset = inp_offset_start + id * src_h_sz;
            const auto id_buf = id - (jcp.copy_block_only ? id_start : 0) + FP;
            const auto out_offset = out_offset_start + id_buf * _pd->pbuf_h_sz;
            cp.src = src + src_dsz * inp_offset;
            cp.dst = inp_buffer + src_dsz * out_offset;
            (*copy_to_pbuffer_)(&cp);
        }
    }
    if (!jcp.copy_block_only) bmask(icb, odb, ohb, owb) = 1;

#undef bmask
}

#define BRGEMM_CONV_KER_HEADER \
    const char *const __restrict src = btc.brgemm_ctx.src; \
    const char *const __restrict weights = btc.brgemm_ctx.weights; \
    const char *const __restrict bias = btc.brgemm_ctx.bias; \
    const int oc = btc.ocb * jcp.oc_block; \
    const int g_oc = btc.g * jcp.oc + oc; \
    const int icb = btc.icc * jcp.nb_ic_blocking; \
    const int ic = icb * jcp.ic_block; \
    const int g_ic = btc.g * jcp.ic + ic; \
    const int ow = btc.owb * jcp.ow_block; \
    const int oh = btc.ohb * jcp.oh_block; \
    const int iid = ndims_pick(btc.od * SD - FP, 0, 0); \
    const int kd_s = ndims_pick(div_up(max(0, -iid), DD), 0, 0); \
    const int kd_f = ndims_pick( \
            KD - div_up(max(0, iid - ID + (KD - 1) * DD + 1), DD), 1, 1); \
    const auto kd_l = kd_f - kd_s; \
    const auto iih = ndims_pick(btc.oh * SH - TP, btc.oh * SH - TP, 0); \
    const auto kh_s_ = div_up(max(0, -iih), DH); \
    const auto kh_s = jcp.is_os_blocking ? 0 : ndims_pick(kh_s_, kh_s_, 0); \
    const auto kh_f_ = KH - div_up(max(0, iih - IH + (KH - 1) * DH + 1), DH); \
    const auto kh_f = ndims_pick(kh_f_, kh_f_, 1); \
    const auto kh_l = kh_f - kh_s; \
    const bool is_oc_tail = (jcp.oc - oc < jcp.oc_block); \
    const bool is_ic_tail = (btc.icc == _pd->ic_chunks - 1 \
            && ((jcp.ic - ic) % jcp.ic_block != 0)); \
    const bool is_ow_tail = (OW - ow < jcp.ow_block); \
    const bool is_oh_tail = (OH - oh < jcp.oh_block); \
    const char *const __restrict bias_w \
            = bias ? bias + (bias_d.blk_off(g_oc) * bia_dsz) : nullptr; \
    const auto nb_ic_b = nstl::min(jcp.nb_ic_blocking, jcp.nb_ic - icb) \
            - (is_ic_tail ? 1 : 0); \
    char *const __restrict dst_base \
            = btc.brgemm_ctx.dst + dst_dsz * (btc.n * dst_d_sz + g_oc); \
    char *ptr_C; \
    char *ptr_D; \
    int kd_b(0), kd_e(0), kh_b(0), kh_e(0), k_l(0), iiw_b(0);

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_base(brgemm_thread_ctx_t &btc) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims;

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(is_ow_tail);
    MAYBE_UNUSED(is_oh_tail);

    int kw_s {0}, kw_full_s {0}, kw_f {0}, kw_full_f {0}, kw_b(0), kw_e(0);

    get_kw_range(ow, kw_s, kw_full_s, kw_full_f, kw_f);

    const auto src_base = src + src_dsz * (btc.n * src_d_sz + g_ic);
    const auto wei_base = weights
            + wei_dsz
                    * (btc.g * _pd->wei_g_stride
                            + btc.ocb * _pd->wei_ocb_stride);

    const auto call_brgemm
            = [&](int brg_idx, int ic_block_s, int n_ic_blocks,
                      dim_t comp_ker_offs, bool do_postops, bool do_only_comp) {
        assert(k_l > 0 && "invalid batch range");
        const auto brg_ker = brgemm_kernels_[brg_idx];

        assert(jcp.brg_type != brgemm_static_offs);
        _pd->init_batch(btc.icc, src_base, wei_base, n_ic_blocks, ic_block_s,
                iid, iih, iiw_b, nullptr, nullptr, kd_b, kd_e, kh_b, kh_e, kw_b,
                kw_e, k_l, btc.brg_batch);

        call_brgemm_kernel(btc, brg_ker, k_l * n_ic_blocks, ptr_C, ptr_D,
                bias_w, g_oc, do_postops, comp_ker_offs, do_only_comp);
    };

    const auto kdhw_loop = [&]() {
        if (kw_e - kw_b <= 0) return;
        int ow_b {0}, ow_e {0};
        get_ow_range(ow, kw_b, ow_b, ow_e);

        const auto do_init
                = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s && kw_b == kw_s;
        const auto do_postwork = _pd->need_postwork
                && btc.icc == (_pd->ic_chunks - 1) && kd_e == kd_f
                && kh_e == kh_f && kw_e == kw_f;
        const auto do_only_comp = need_compensation && kd_e == kd_f
                && kh_e == kh_f && kw_e != kw_f
                && btc.icc == (_pd->ic_chunks - 1);
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (kh_e - kh_b) * (kw_e - kw_b);
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

        if (ow_l > 0 && k_l > 0) {
            const auto comp_ker_offs = get_comp_offset(
                    btc.g, btc.ocb, ow_b, kd_s, kd_f, kh_s, kh_f, kw_b, kw_e);

            const auto ker_i = ow_l - 1;
            int kernel_idx[2][2];
            kernel_idx[false][false] = _pd->get_brg_idx(
                    ker_i, false, is_oc_tail, false, kd_s, kd_f, kh_s, kh_f);
            kernel_idx[true][false] = _pd->get_brg_idx(
                    ker_i, true, is_oc_tail, false, kd_s, kd_f, kh_s, kh_f);
            kernel_idx[false][true] = _pd->get_brg_idx(
                    ker_i, false, is_oc_tail, true, kd_s, kd_f, kh_s, kh_f);
            kernel_idx[true][true] = _pd->get_brg_idx(
                    ker_i, true, is_oc_tail, true, kd_s, kd_f, kh_s, kh_f);

            if (nb_ic_b > 0) {
                const auto brg_idx = kernel_idx[do_init][false];
                call_brgemm(brg_idx, 0, nb_ic_b, comp_ker_offs,
                        do_postwork && !is_ic_tail, do_only_comp);
            }

            if (is_ic_tail) {
                const auto use_init_ker = (do_init && nb_ic_b == 0);
                const auto brg_ic_tail_idx = kernel_idx[use_init_ker][true];
                call_brgemm(brg_ic_tail_idx, nb_ic_b, 1, comp_ker_offs,
                        do_postwork, do_only_comp);
            }
        }

        perform_outwork(btc, dst_base, bias_w, ow, g_oc, is_oc_tail, ow_b, ow_e,
                kd_l, kh_l, do_init, do_postwork, false);
    };

    if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s) {
        // kw values with left padding
        if (kw_s < kw_full_s) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK_PAD);
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
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK);
                    for (kw_b = kw_full_s; kw_b < kw_full_f; kw_b += KW_BLOCK) {
                        kw_e = nstl::min(kw_full_f, kw_b + KW_BLOCK);
                        kdhw_loop();
                    }
                }
            }
        }

        // kw values with right padding
        if (kw_full_f < kw_f) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min(kh_f, kh_b + KH_BLOCK_PAD);
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
        perform_outwork(btc, dst_base, bias_w, ow, g_oc, is_oc_tail, ow, ow,
                kd_l, kh_l, do_init, do_postwork, false);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_trans(
        brgemm_thread_ctx_t &btc, char *inp_buffer) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims;

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(g_ic);
    MAYBE_UNUSED(src);

    const auto wei_base = weights
            + wei_dsz
                    * (btc.g * _pd->wei_g_stride
                            + btc.ocb * _pd->wei_ocb_stride);
    const int ow_b {ow},
            ow_e {ow + (is_ow_tail ? jcp.ow % jcp.ow_block : jcp.ow_block)};
    const int oh_b {oh},
            oh_e {oh + (is_oh_tail ? jcp.oh % jcp.oh_block : jcp.oh_block)};
    const auto iid_shift = jcp.copy_block_only
            ? nstl::max(0, btc.odb * jcp.od_block * SD - FP)
            : 0;
    const auto iih_shift = jcp.copy_block_only
            ? nstl::max(0, btc.ohb * jcp.oh_block * SH - TP)
            : 0;
    const auto iiw_shift
            = jcp.copy_block_only ? (btc.owb * jcp.ow_block * SW) : 0;

    const auto iid_b = iid + FP - iid_shift;
    const auto iih_b = iih + TP - iih_shift;
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

    const auto ker_i = (jcp.is_os_blocking ? oh_l * ow_l : ow_l) - 1;
    const auto kw_e = jcp.kw_sets > 1 ? 1 : KW;

    const auto call_brgemm = [&](int brg_idx, int ic_block_s, int n_ic_blocks,
                                     bool do_postops) {
        assert(k_l > 0 && "invalid batch range");
        const auto brg_ker = brgemm_kernels_[brg_idx];

        const auto kh_ee = jcp.kh_sets > 1 ? kh_b + 1 : kh_e;
        const auto pbuf_base = inp_buffer
                + src_dsz
                        * ((jcp.copy_block_only ? 0
                                                : ((icb + ic_block_s)
                                                          * _pd->pbuf_d_sz)));
        const void *ptrA {nullptr}, *ptrB {nullptr};

        if (jcp.brg_type == brgemm_static_offs) {
            _pd->get_A_B(btc.icc, pbuf_base, wei_base, ic_block_s, iid_b, iih_b,
                    iiw_b, kd_b, kh_b, ptrA, ptrB);
            btc.brg_batch[0].ptr.A = ptrA;
            btc.brg_batch[0].ptr.B = ptrB;
        } else {
            _pd->init_batch(btc.icc, pbuf_base, wei_base, n_ic_blocks,
                    ic_block_s, iid_b, iih_b, iiw_b, nullptr, nullptr, kd_b,
                    kd_e, kh_b, kh_ee, 0, kw_e, k_l, btc.brg_batch);
        }

        call_brgemm_kernel(btc, brg_ker, k_l * n_ic_blocks, ptr_C, ptr_D,
                bias_w, g_oc, do_postops, 0, false);
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = _pd->need_postwork
                && btc.icc == (_pd->ic_chunks - 1) && kd_e == kd_f
                && kh_e == kh_f;
        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (jcp.kh_sets > 1 ? 1 : (kh_e - kh_b))
                * (jcp.kw_sets > 1 ? 1 : KW);

        int kernel_idx[2][2];
        kernel_idx[false][false] = _pd->get_brg_idx(
                ker_i, false, is_oc_tail, false, kd_s, kd_f, kh_s, kh_f);
        kernel_idx[true][false] = _pd->get_brg_idx(
                ker_i, true, is_oc_tail, false, kd_s, kd_f, kh_s, kh_f);
        kernel_idx[false][true] = _pd->get_brg_idx(
                ker_i, false, is_oc_tail, true, kd_s, kd_f, kh_s, kh_f);
        kernel_idx[true][true] = _pd->get_brg_idx(
                ker_i, true, is_oc_tail, true, kd_s, kd_f, kh_s, kh_f);

        if (nb_ic_b > 0) {
            const auto brg_idx = kernel_idx[do_init][false];
            call_brgemm(brg_idx, 0, nb_ic_b, do_postwork && !is_ic_tail);
        }

        if (is_ic_tail) {
            const auto use_init_ker = (do_init && nb_ic_b == 0);
            const auto brg_ic_tail_idx = kernel_idx[use_init_ker][true];
            call_brgemm(brg_ic_tail_idx, nb_ic_b, 1, do_postwork);
        }
    };

    if (kd_f > kd_s && kh_f > kh_s) {
        // kw values covering full ow_block
        for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
            kd_e = nstl::min(kd_f, kd_b + KD_BLOCK);
            for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                kh_e = nstl::min(kh_f, kh_b + KH_BLOCK);
                kdhw_loop();
            }
        }
    } else {
        const auto do_init = btc.icc == 0;
        const auto do_postwork
                = _pd->need_postwork && btc.icc == (_pd->ic_chunks - 1);
        perform_outwork(btc, dst_base, bias_w, ow, g_oc, is_oc_tail, ow, ow,
                kd_l, kh_l, do_init, do_postwork, false);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_fwd_t<isa>::ker_vpad(brgemm_thread_ctx_t &btc) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims;

    BRGEMM_CONV_KER_HEADER;
    MAYBE_UNUSED(is_oh_tail);

    const char *const __restrict src_base
            = src + src_dsz * (btc.n * src_d_sz + g_ic);
    const char *const __restrict wei_base = weights
            + wei_dsz
                    * (btc.g * _pd->wei_g_stride
                            + btc.ocb * _pd->wei_ocb_stride);

    const int ow_b {ow}, ow_e {ow + (is_ow_tail ? jcp.M_tail : jcp.M)};
    iiw_b = ow_b * SW - LP;
    ptr_D = dst_base
            + dst_dsz
                    * (btc.od * dst_h_sz + btc.oh * dst_w_sz
                            + ow_b * jcp.oc_without_padding);
    ptr_C = (jcp.use_buffer) ? btc.c_buffer + acc_dsz * (ow_b - ow) * jcp.LDC
                             : static_cast<char *>(ptr_D);

    const auto ow_l = ow_e - ow_b;
    assert(0 <= ow_l && ow_l <= jcp.ow_block);
    const auto ker_i = ow_l - 1;
    const dim_t *const __restrict kw_top_vpads
            = owb_kw_top_vpads.data() + btc.owb * KW;
    const dim_t *const __restrict kw_bottom_vpads
            = owb_kw_bottom_vpads.data() + btc.owb * KW;

    const auto call_brgemm = [&](int brg_idx, int ic_block_s, int n_ic_blocks,
                                     dim_t comp_ker_offs, bool do_postops) {
        assert(k_l > 0 && "invalid batch range");
        const auto brg_ker = brgemm_kernels_[brg_idx];

        assert(jcp.brg_type != brgemm_static_offs);
        _pd->init_batch(btc.icc, src_base, wei_base, n_ic_blocks, ic_block_s,
                iid, iih, iiw_b, kw_top_vpads, kw_bottom_vpads, kd_b, kd_e,
                kh_b, kh_e, 0, KW, k_l, btc.brg_batch);

        call_brgemm_kernel(btc, brg_ker, k_l * n_ic_blocks, ptr_C, ptr_D,
                bias_w, g_oc, do_postops, comp_ker_offs, false);
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = btc.icc == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = _pd->need_postwork
                && btc.icc == (_pd->ic_chunks - 1) && kd_e == kd_f
                && kh_e == kh_f;

        if (ow_e - ow_b <= 0 && !do_init && !do_postwork) return;

        k_l = (kd_e - kd_b) * (kh_e - kh_b) * KW;
        int kernel_idx[2][2];
        kernel_idx[false][false] = _pd->get_brg_idx(
                ker_i, false, is_oc_tail, false, kd_s, kd_f, kh_s, kh_f);
        kernel_idx[true][false] = _pd->get_brg_idx(
                ker_i, true, is_oc_tail, false, kd_s, kd_f, kh_s, kh_f);
        kernel_idx[false][true] = _pd->get_brg_idx(
                ker_i, false, is_oc_tail, true, kd_s, kd_f, kh_s, kh_f);
        kernel_idx[true][true] = _pd->get_brg_idx(
                ker_i, true, is_oc_tail, true, kd_s, kd_f, kh_s, kh_f);

        const auto comp_offs = get_comp_offset(
                btc.g, btc.ocb, ow, kd_b, kd_e, kh_b, kh_e, 0, KW);

        if (nb_ic_b > 0) {
            const auto brg_idx = kernel_idx[do_init][false];
            call_brgemm(
                    brg_idx, 0, nb_ic_b, comp_offs, do_postwork && !is_ic_tail);
        }

        if (is_ic_tail) {
            const auto use_init_ker = (do_init && nb_ic_b == 0);
            const auto brg_ic_tail_idx = kernel_idx[use_init_ker][true];
            call_brgemm(brg_ic_tail_idx, nb_ic_b, 1, comp_offs, do_postwork);
        }
    };

    if (kd_f > kd_s && kh_f > kh_s) {
        // kw values covering full ow_block
        for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
            kd_e = nstl::min(kd_f, kd_b + KD_BLOCK);
            for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                kh_e = nstl::min(kh_f, kh_b + KH_BLOCK);
                kdhw_loop();
            }
        }
    } else {
        const auto do_init = btc.icc == 0;
        const auto do_postwork
                = _pd->need_postwork && btc.icc == (_pd->ic_chunks - 1);
        perform_outwork(btc, dst_base, bias_w, ow, g_oc, is_oc_tail, ow, ow,
                kd_l, kh_l, do_init, do_postwork, false);
    }
}

#undef BRGEMM_CONV_KER_HEADER
template struct brgemm_convolution_fwd_t<sve_512>;
template struct brgemm_convolution_fwd_t<sve_256>;
template struct brgemm_convolution_fwd_t<sve_128>;

} // namespace aarch64

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
