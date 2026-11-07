/*******************************************************************************
* Copyright 2022 Intel Corporation
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
#include "common/convolution_pd.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"
#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/jit_brgemm_conv_bwd_strided.hpp"
#include "cpu/x64/jit_brgemm_conv_bwd_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::status;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace nstl;
using namespace data_type;

using namespace jit_avx512_core_brgemm_conv_bwd_trans_kernel;
using namespace jit_avx512_core_brgemm_conv_bwd_copy_kernel;
using namespace jit_uni_brgemm_conv_comp_pad_kernel;

#define ndims_pick(v5, v4, v3) \
    ((ndims == 5) ? (v5) : (ndims == 4) ? (v4) : (ndims == 3) ? (v3) : 0)

static bool impl_supports_datatype(data_type_t data_type) {
    switch (data_type) {
        case data_type::bf16:
            return x64::mayiuse(x64::avx512_core)
                    || x64::mayiuse(x64::avx2_vnni_2);
        case data_type::f16:
            return x64::mayiuse(x64::avx512_core_fp16)
                    || x64::mayiuse(x64::avx2_vnni_2);
        case data_type::f8_e5m2:
            return x64::mayiuse(x64::avx10_2)
                    || x64::mayiuse(x64::avx512_core_amx_fp16);
        case data_type::f8_e4m3:
            return x64::mayiuse(x64::avx10_2)
                    || x64::mayiuse(x64::avx512_core_amx_fp16);
        case data_type::f32:
        case data_type::s32:
        case data_type::s8:
        case data_type::u8: return true;
        default: return false;
    }
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_strided_t<isa>::pd_t::init(
        const engine_t *engine) {
    using namespace data_type;

    const auto diff_src_type = diff_src_md(0)->data_type;
    const auto wei_type = weights_md(0)->data_type;
    const auto diff_dst_type = diff_dst_md(0)->data_type;
    const bool is_int8 = one_of(diff_dst_type, u8, s8);

    // The following check will detect if this implementation is being
    // executed through a deconvolution call and prevent the primitive from
    // executing 'use_inversion == true' as BWD_D. This can only work if the
    // src_desc and dst_desc are defined in the aforementioned.
    const convolution_desc_t &cd = *desc();
    if (cd.use_inversion
            && one_of(true, types::is_zero_md(&cd.src_desc),
                    types::is_zero_md(&cd.dst_desc)))
        return status::unimplemented;

    using skip_mask_t = primitive_attr_t::skip_mask_t;
    auto skip_mask = skip_mask_t::fpmath_mode;
    if (cd.use_inversion)
        skip_mask |= skip_mask_t::post_ops | skip_mask_t::sum_dt;
    if (is_int8 && cd.use_inversion)
        skip_mask |= skip_mask_t::scales | skip_mask_t::zero_points;

    const bool is_f32_supported
            = everyone_is(f32, diff_src_type, wei_type, diff_dst_type)
            && IMPLICATION(with_bias(), bias_md_.data_type == f32);

    const bool is_xf16_supported = one_of(wei_type, bf16, f16)
            && wei_type == diff_dst_type && one_of(diff_src_type, wei_type, f32)
            && IMPLICATION(
                    with_bias(), one_of(bias_md_.data_type, f32, wei_type));

    const bool is_int8_supported
            = one_of(diff_src_type, s8, u8, s32, f32, bf16, f16)
            && wei_type == s8 && is_int8
            && IMPLICATION(
                    with_bias(), one_of(bias_md_.data_type, f32, s32, s8, u8))
            && cd.use_inversion /* only deconv uses int8 */;

    const bool is_fp8_supported = one_of(wei_type, f8_e5m2, f8_e4m3)
            && one_of(diff_dst_type, f8_e5m2, f8_e4m3)
            && one_of(diff_src_type, wei_type, f32, f8_e5m2, f8_e4m3)
            && IMPLICATION(
                    with_bias(), one_of(bias_md_.data_type, f32, wei_type));
    const bool is_f32_xf16_supported = one_of(wei_type, bf16, f16)
            && everyone_is(f32, diff_src_type, diff_dst_type)
            && IMPLICATION(with_bias(), bias_md_.data_type == f32);

    VDISPATCH_CONV(
            impl::is_dense_format_kind({src_md(), weights_md(), dst_md()}),
            VERBOSE_UNSUPPORTED_SPARSE_CFG);
    VDISPATCH_CONV(is_bwd_d(), VERBOSE_BAD_PROPKIND);
    VDISPATCH_CONV(
            impl_supports_datatype(diff_src_type), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(impl_supports_datatype(
                           is_f32_xf16_supported ? diff_dst_type : wei_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(
            impl_supports_datatype(diff_dst_type), VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(
            one_of(true, is_f32_supported, is_xf16_supported, is_int8_supported,
                    is_fp8_supported, is_f32_xf16_supported),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(!one_of(f64, diff_src_type, wei_type, diff_dst_type)
                    && IMPLICATION(with_bias(), f64 != bias_md_.data_type),
            VERBOSE_UNSUPPORTED_DT);
    VDISPATCH_CONV(set_default_alg_kind(alg_kind::convolution_direct),
            VERBOSE_BAD_ALGORITHM);
    VDISPATCH_CONV(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");
    VDISPATCH_CONV(attr()->has_default_values(skip_mask, diff_src_type),
            VERBOSE_UNSUPPORTED_ATTR);
    VDISPATCH_CONV(IMPLICATION(cd.use_inversion,
                           attr()->post_ops_.check_sum_consistency(
                                   diff_src_type, is_int8_supported)),
            VERBOSE_UNSUPPORTED_POSTOP);

    CHECK(brgemm_convolution_bwd_utils::init_conf(jcp_, isa, desc_,
            diff_dst_md_, weights_md_, diff_src_md_, bias_md_, attr_,
            dnnl_get_max_threads()));

    brgs_sz_ = 2 * 2 * 2 * 2;
    brgs_ = std::make_shared<brgemm_containers::brgemm_desc_container_t>();
    brgs_->resize(brgs_sz_);

    brg_indices_c = 0;

    auto ndims = jcp_.ndims;

    const auto KD = ndims_pick(jcp_.kd, 1, 1);
    const auto KH = ndims_pick(jcp_.kh, jcp_.kh, 1);
    const auto KW = jcp_.kw;
    const auto KD_BLOCK = ndims_pick(jcp_.kd_block, 1, 1);
    const auto KH_BLOCK = ndims_pick(jcp_.kh_block, jcp_.kh_block, 1);

    const auto SD = ndims_pick(jcp_.stride_d, 1, 1);
    const auto SH = ndims_pick(jcp_.stride_h, jcp_.stride_h, 1);
    const auto SW = jcp_.stride_w;
    const auto IW = jcp_.iw;

    const auto is_amx = brgemm_convolution_bwd_utils::is_amx(isa);

    // Enumerate all possible batch sizes for exec_trans AMX path.
    // k_l = div_up(kd_range, SD) * div_up(kh_range, SH) * div_up(kw_range, SW)
    // kw_s aligns to stride sector (0..SW-1), so kw_range = KW - kw_s.
    std::vector<int> batch_sizes;
    if (jcp_.exec_type == exec_trans && is_amx) {
        std::set<int> bs_set;
        bs_set.insert(0); // for skip_accumulation case
        for (dim_t kd_range = 1; kd_range <= KD_BLOCK; kd_range++) {
            const auto kd_l = div_up(kd_range, SD);
            for (dim_t kh_range = 1; kh_range <= KH_BLOCK; kh_range++) {
                const auto kh_l = div_up(kh_range, SH);
                for (int kw_s_off = 0; kw_s_off < nstl::min<dim_t>(KW, SW);
                        kw_s_off++) {
                    const auto kw_range = KW - kw_s_off;
                    const auto kw_l = div_up(kw_range, SW);
                    bs_set.insert(static_cast<int>(kd_l * kh_l * kw_l));
                }
            }
        }
        batch_sizes.assign(bs_set.begin(), bs_set.end());
    } else {
        batch_sizes.push_back(0); // default: use jcp_.max_batch
    }

    // TODO: this is only needed if we have d/h padding exceeding kd/kh
    int M_begin = 0;
    int M_end = (jcp_.M_tail == jcp_.M) ? 1 : 2;
    int N_begin = 0;
    int N_end = (jcp_.N_tail == jcp_.N) ? 1 : 2;
    int K_begin = 0;
    int K_end = (jcp_.K_tail == jcp_.K) ? 1 : 2;
    int i_init_begin = (div_up(jcp_.nb_oc, jcp_.nb_oc_blocking) == 1
                               && KD_BLOCK == KD && KH_BLOCK == KH)
            ? 1
            : 0;
    int i_init_end = 2;

    for_(int i_N = N_begin; i_N < N_end; i_N++)
    for_(int i_M = M_begin; i_M < M_end; i_M++)
    for_(int i_init = i_init_begin; i_init < i_init_end; i_init++)
    for_(int i_K = K_begin; i_K < K_end; i_K++)
    for (auto bs : batch_sizes) {
        auto M = (i_M) ? jcp_.M_tail : jcp_.M;
        if (M <= 0) continue;
        CHECK(add_brg_descriptor(M, i_N, i_K, i_init, 0, {}, bs));
    }

    // When has_uneven_iw, the tail iw block may have fewer valid spatial
    // points for certain stride sectors.
    if (jcp_.has_uneven_iw && jcp_.exec_type == exec_trans) {
        const auto M_full = jcp_.M > 0 ? jcp_.M : jcp_.M_tail;
        const auto eff_iw = jcp_.iw_tail > 0 ? jcp_.iw_tail : jcp_.iw;
        for (int sw = 0; sw < SW; sw++) {
            if (eff_iw <= sw) continue;
            const dim_t M_sw = div_up(eff_iw - sw, SW);
            if (M_sw > 0 && M_sw < M_full) {
                if (jcp_.use_uker) {
                    // AMX uker: use bd_mask with constant M to avoid tile
                    // reconfiguration. The uker skips masked rows during
                    // postops store.
                    std::vector<char> mask(M_full, 0);
                    for (dim_t i = 0; i < M_sw; i++)
                        mask[i] = 1;
                    for_(int i_N = N_begin; i_N < N_end; i_N++)
                    for_(int i_init = i_init_begin; i_init < i_init_end;
                            i_init++)
                    for_(int i_K = K_begin; i_K < K_end; i_K++)
                    for (auto bs : batch_sizes) {
                        CHECK(add_brg_descriptor(
                                M_full, i_N, i_K, i_init, M_sw, mask, bs));
                    }
                } else {
                    // Without uker: use the actual reduced M directly.
                    for_(int i_N = N_begin; i_N < N_end; i_N++)
                    for_(int i_init = i_init_begin; i_init < i_init_end;
                            i_init++)
                    for (int i_K = K_begin; i_K < K_end; i_K++) {
                        CHECK(add_brg_descriptor(M_sw, i_N, i_K, i_init));
                    }
                }
            }
        }
    }

    if (jcp_.exec_type == exec_base) {
        // create brgemm kernels for iw_blocks with padded areas and
        // apply post-ops on final iteration by kw to padded areas in iw_block
        int kw_s {0}, kw_full_s {0}, kw_full_f {0}, kw_f {0}, iw_s {0},
                M_without_overflow {0};

        auto init_kernels_kw_loop = [&](dim_t sw, dim_t iw) -> status_t {
            const auto iw_str = iw + sw;
            get_kw_range(static_cast<int>(iw_str), static_cast<int>(iw), kw_s,
                    kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                get_iw_range(static_cast<int>(iw_str), static_cast<int>(iw), kw,
                        iw_s, M_without_overflow);
                if (M_without_overflow <= 0) continue;
                for_(int i_init = 0; i_init < 2; i_init++)
                for_(int i_N = 0; i_N < 2; i_N++)
                for (int i_K = 0; i_K < 2; i_K++) {
                    CHECK(add_brg_descriptor(
                            M_without_overflow, i_N, i_K, i_init));
                }
            }
            return status::success;
        };
        for (int sw = 0; sw < SW; sw++) {
            for (dim_t iw = 0; iw < IW; iw += jcp_.iw_block) {
                CHECK(init_kernels_kw_loop(sw, iw));
                if (kw_f == jcp_.kw && kw_s == 0) break;
            }
            for (dim_t iw = (jcp_.nb_iw - 1) * jcp_.iw_block; iw >= 0;
                    iw -= jcp_.iw_block) {
                CHECK(init_kernels_kw_loop(sw, iw));
                if (kw_f == jcp_.kw && kw_s == 0) break;
            }
        }
    }
    brgs_sz_ = static_cast<int>(brgs_->refs_size());

    auto scratchpad = scratchpad_registry().registrar();
    brgemm_convolution_bwd_utils::init_scratchpad(scratchpad, jcp_);

    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_strided_t<isa>::pd_t::add_brg_descriptor(
        dim_t vM, bool is_N_tail, bool is_K_tail, bool do_init,
        dim_t bd_mask_idx, const std::vector<char> &bd_mask, int bs) {

    if (do_init && is_K_tail && jcp_.K > 0) return status::success;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    auto vbeta = do_init ? 0 : beta;
    auto vN = is_N_tail ? jcp_.N_tail : jcp_.N;
    auto vK = is_K_tail ? jcp_.K_tail : jcp_.K;

    const auto is_amx = brgemm_convolution_bwd_utils::is_amx(isa);

    if (vN == 0 || vK == 0) return status::success;

    int brg_idx = get_brg_idx(static_cast<int>(vM), do_init, is_N_tail,
            is_K_tail, static_cast<int>(bd_mask_idx), bs);
    // if brgemm_desc_t already created then skip this iteration
    if (brg_idx != -1) return status::success;

    brgemm_desc_t brg;
    brgemm_strides_t brg_strides;
    brg_strides.stride_a = jcp_.brg_stride_a;
    brg_strides.stride_b = jcp_.brg_stride_b;
    brg.req_cal_comp_pads = jcp_.req_brg_comp_pad;
    brg.req_comp_pads_with_bcast
            = jcp_.req_cal_comp_pad && jcp_.exec_type == exec_trans;
    const auto strides_ptr
            = (jcp_.brg_type == brgemm_strd) ? &brg_strides : nullptr;
    brg.fp8_with_f16_vnni_block = jcp_.is_fp8 && jcp_.vnni_block == 2;
    CHECK(brgemm_desc_init(&brg, isa, jcp_.brg_type, jcp_.src_dt, jcp_.wei_dt,
            false, false, brgemm_row_major, alpha, vbeta, jcp_.LDA, jcp_.LDB,
            jcp_.LDC, vM, vN, vK, strides_ptr));

    brgemm_attr_t brgattr;
    brgattr.use_uker = jcp_.use_uker;
    brgattr.use_interleave_stores = jcp_.use_interleave_stores;
    brgattr.hint_prefetching = jcp_.hint_prefetching;
    brgattr.max_bs = bs > 0 ? bs : jcp_.max_batch;
    brgattr.hint_innermost_loop = jcp_.brgemm_bd_loop_innermost
            ? brgemm_bd_loop_innermost
            : brgemm_innermost_undef;
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
    brgattr.bd_mask = bd_mask.empty() ? nullptr : bd_mask.data();
    brgattr.bd_mask_level = bd_mask.empty() ? 0 : 1;

    if (is_amx) {
        brgattr.max_top_vpad = 0;
        brgattr.max_bottom_vpad = 0;
    } else {
        brgattr.max_top_vpad = jcp_.max_vpad;
        brgattr.max_bottom_vpad = jcp_.max_vpad;
    }
    brgattr.generate_skip_accumulation = true;
    brgattr.use_ace = jcp_.is_ace;
    brgattr.enable_size_driven_roll = true;
    CHECK(brgemm_desc_set_attr(&brg, brgattr));

    auto LDD = jcp_.stride_w * jcp_.ic_without_padding;
    brg.with_sum = jcp_.with_sum;
    brg.with_weights_scale_adjust = jcp_.scale_adjust_factor != 1.0f;
    CHECK(brgemm_desc_set_postops(
            &brg, attr(), &diff_src_md_, LDD, jcp_.bia_dt));
    CHECK(brgemm_desc_finalize(&brg));

    jcp_.amx_buf_size_per_thread = nstl::max<dim_t>(
            brg.get_wsp_buffer_size(), jcp_.amx_buf_size_per_thread);
    brg_idx = brgs_->insert(
            brg, bd_mask, std::vector<brgemm_batch_element_t> {});

    const std::array<int, 6> key = {static_cast<int>(vM), is_N_tail, is_K_tail,
            do_init, static_cast<int>(bd_mask_idx), bs};
    if (brg_indices.find(key) == brg_indices.end()) {
        brg_indices.insert({key, brg_idx});
        brg_indices_c++;
    }

    return status::success;
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::pd_t::get_kw_range(int iw,
        int iw_raw, int &kw_s, int &kw_full_s, int &kw_full_f,
        int &kw_f) const {
    // This function is needed for exec_base only
    brgemm_convolution_bwd_utils::get_kw_range(
            jcp_, iw, iw_raw, kw_s, kw_full_s, kw_full_f, kw_f);
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::pd_t::get_iw_range(
        int iw, int iw_raw, int kw, int &iw_s, int &M_without_overflow) const {
    // This function is needed for exec_base only

    brgemm_convolution_bwd_utils::get_iw_range(
            jcp_, iw, iw_raw, kw, iw_s, M_without_overflow);
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_strided_t<isa>::add_brg_kernel(int brg_idx) {
    const auto _pd = pd();
    const auto &brgs = *(_pd->brgs_);

    auto brg = brgs[brg_idx];
    if (!brg_kernels_[brg_idx] && brg && brg->bcast_dim > 0 && brg->load_dim > 0
            && brg->reduce_dim > 0) {
        CHECK(brg_kernels_.insert(brg_idx, brg));
        if (is_amx && brg->is_tmm) brgemm_palettes_.insert(brg_idx, brg);
    }
    return status::success;
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_strided_t<isa>::add_po_kernel(
        brgemm_desc_t *bcfg, int ker_idx, bool is_init) {
    if (!bcfg) return status::success;
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    bcfg->LDD = (is_init && jcp.use_buffer) ? jcp.LDC : jcp.LDD;
    bcfg->dt_c = (!is_init && jcp.use_buffer) ? jcp.acc_dt : jcp.dst_dt;
    bcfg->dt_d = (is_init && jcp.use_buffer) ? jcp.acc_dt : jcp.dst_dt;
    bcfg->typesize_C = static_cast<int>(types::data_type_size(bcfg->dt_c));
    bcfg->typesize_D = static_cast<int>(types::data_type_size(bcfg->dt_d));
    bcfg->alpha = (!is_init && IMPLICATION(jcp.with_sum, jcp.use_buffer)) ? 1.f
                                                                          : 0.f;
    bcfg->beta = is_init ? 0.f : 1.f;
    CHECK(safe_ptr_assign(kernels_po_[ker_idx],
            jit_brgemm_kernel_post_ops_base_t::create(
                    isa, *bcfg, *_pd->attr())));
    kernels_po_[ker_idx]->generate_kernel();
    return status::success;
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::add_po_kernels(
        int i_N, dim_t init_bcast_dim, dim_t po_bcast_dim) {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto &brgs = *(_pd->brgs_);

    auto N = (i_N) ? jcp.N_tail : jcp.N;
    if (N <= 0) return;
    auto i_K = (jcp.K_tail > 0);

    const int brg_idx = _pd->get_any_brg_idx(i_N, i_K);

    if (init_bcast_dim > 0) {
        if (brgs[brg_idx]) {
            auto init_cfg = *(brgs[brg_idx]);
            auto ker_init_idx = get_ker_po_idx(init_bcast_dim - 1, false, i_N);
            if (init_cfg.load_dim > 0 && kernels_po_[ker_init_idx] == nullptr) {
                init_cfg.bcast_dim = init_bcast_dim;
                add_po_kernel(&init_cfg, ker_init_idx, true);
            }
        }
    }

    if ((need_postwork || jcp.use_buffer) && po_bcast_dim > 0) {
        if (brgs[brg_idx]) {
            auto po_cfg = *(brgs[brg_idx]);
            auto ker_po_idx = get_ker_po_idx(po_bcast_dim - 1, true, i_N);
            if (po_cfg.load_dim > 0 && kernels_po_[ker_po_idx] == nullptr) {
                po_cfg.bcast_dim = po_bcast_dim;
                add_po_kernel(&po_cfg, ker_po_idx, false);
            }
        }
    }
}

template <cpu_isa_t isa>
dim_t brgemm_convolution_bwd_strided_t<isa>::get_comp_ker_idx(dim_t kd_b,
        dim_t kd_e, dim_t kh_b, dim_t kh_e, dim_t kw_b, dim_t kw_e) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!jcp.req_cal_comp_pad) return 0;

    assert(IMPLICATION(kd_e <= kd_b && kh_e <= kh_b,
            everyone_is(0, kd_b, kd_e, kh_b, kh_e, kw_b, kw_e)));
    for (dim_t k = 0; k < jcp.ker_ranges_size; k++) {
        if (kd_b == kd_bs[k] && kd_e == kd_es[k] && kh_b == kh_bs[k]
                && kh_e == kh_es[k] && kw_b == kw_bs[k] && kw_e == kw_es[k]) {
            return k;
        }
    }

    return -1;
}

template <cpu_isa_t isa>
dim_t brgemm_convolution_bwd_strided_t<isa>::get_comp_offset(dim_t g, dim_t icb,
        dim_t iw, dim_t kd_b, dim_t kd_e, dim_t kh_b, dim_t kh_e, dim_t kw_b,
        dim_t kw_e) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!jcp.src_zero_point && !jcp.s8s8_compensation_required) return 0;

    const auto comp_iw = get_comp_iw(iw);
    const auto comp_idx = get_comp_ker_idx(kd_b, kd_e, kh_b, kh_e, kw_b, kw_e);

    assert(IMPLICATION(jcp.req_cal_comp_pad, comp_idx >= 0));

    return jcp.req_cal_comp_pad ? g * comp_icb_sz + icb * comp_ker_sz
                    + comp_idx * comp_kw_sz + comp_iw * comp_iw_sz
                                : (g * jcp.nb_ic + icb) * jcp.ic_block;
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::create_kernels() {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    is_amx = brgemm_convolution_bwd_utils::is_amx(isa);

    for (const auto &key_value_pair : _pd->brg_indices) {
        const int brg_idx = key_value_pair.second;
        add_brg_kernel(brg_idx);
    }

    int M_begin = 0;
    int M_end = (jcp.M_tail == jcp.M) ? 1 : 2;
    int N_begin = 0;
    int N_end = (jcp.N_tail == jcp.N) ? 1 : 2;

    if (jcp.exec_type == exec_base) {
        for_(int i_N = N_begin; i_N < N_end; i_N++)
        for (int i_M = M_begin; i_M < M_end; i_M++) {
            // create post-op kernels for cases when we never call brgemm kernels
            // e.g. for d/h padded areas
            // TODO: do this only if d/h padding > kd/kh
            auto M = (i_M) ? jcp.M_tail : jcp.M;
            add_po_kernels(i_N, M, M);
        }
    } else if (jcp.exec_type == exec_trans && jcp.use_uker) {
        // For exec_trans with AMX uker: create post-op kernels for positions
        // with no valid kd/kh overlap (perform_outwork handles these).
        for_(int i_N = N_begin; i_N < N_end; i_N++)
        for (int i_M = M_begin; i_M < M_end; i_M++) {
            auto M = (i_M) ? jcp.M_tail : jcp.M;
            add_po_kernels(i_N, M, M);
        }
    }

    if (jcp.exec_type == exec_base) {
        // create brgemm kernels for iw_blocks with padded areas and
        // apply post-ops on final iteration by kw to padded areas in iw_block
        int kw_s {0}, kw_full_s {0}, kw_full_f {0}, kw_f {0}, iw_s {0},
                M_without_overflow {0};
        dim_t iw_f {0};

        auto init_kernels_kw_loop = [&](dim_t sw, dim_t iw) {
            const auto iw_str = iw + sw;
            _pd->get_kw_range(static_cast<int>(iw_str), static_cast<int>(iw),
                    kw_s, kw_full_s, kw_full_f, kw_f);
            for (int kw = kw_s; kw < kw_f; kw++) {
                _pd->get_iw_range(static_cast<int>(iw_str),
                        static_cast<int>(iw), kw, iw_s, M_without_overflow);
                if (M_without_overflow <= 0) continue;

                bool is_iw_tail = (jcp.iw - iw < jcp.iw_block);
                for_(int i_N = 0; i_N < 2; i_N++)
                for (int i_side = 0; i_side < 2; i_side++) {
                    const auto M
                            = div_up(is_iw_tail ? jcp.iw_tail : jcp.iw_block,
                                      SW)
                            * SW;
                    if (M <= 0) continue;
                    _pd->get_iw_range(static_cast<int>(iw_str),
                            static_cast<int>(iw), kw, iw_s, M_without_overflow);
                    iw_f = iw_s + (M_without_overflow * SW);
                    const auto init_bcast_dim
                            = ((i_side == 0) ? (iw_s - iw_str)
                                             : (iw_str + M - iw_f))
                            / SW;
                    _pd->get_iw_range(static_cast<int>(iw_str),
                            static_cast<int>(iw), kw_f - kw, iw_s,
                            M_without_overflow);
                    iw_f = iw_s + (M_without_overflow * SW);
                    const auto po_bcast_dim
                            = ((i_side == 0) ? (iw_s - iw_str)
                                             : (iw_str + M - iw_f))
                            / SW;
                    // TODO: this condition is a workaround and should be
                    // integrated with po_bcast_dim calculation or moved to
                    // a separate place where kernels that have
                    // init_bcast_dim == po_bcast_dim are created
                    if (init_bcast_dim > 0 && po_bcast_dim == 0
                            && (need_postwork || jcp.use_buffer))
                        add_po_kernels(i_N, init_bcast_dim, init_bcast_dim);
                    else
                        add_po_kernels(i_N, init_bcast_dim, po_bcast_dim);
                }
            }
        };
        for (int sw = 0; sw < SW; sw++) {
            for (dim_t iw = 0; iw < IW; iw += jcp.iw_block) {
                init_kernels_kw_loop(sw, iw);
                if (kw_f == jcp.kw && kw_s == 0) break;
            }
            for (dim_t iw = (jcp.nb_iw - 1) * jcp.iw_block; iw >= 0;
                    iw -= jcp.iw_block) {
                init_kernels_kw_loop(sw, iw);
                if (kw_f == jcp.kw && kw_s == 0) break;
            }
        }
    }
}

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_strided_t<isa>::init(engine_t *engine) {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    bia_dsz = jcp.bia_dsz;
    acc_dsz = jcp.acc_dsz;
    src_dsz = jcp.src_dsz;
    wei_dsz = jcp.wei_dsz;
    dst_dsz = jcp.dst_dsz;

    auto ndims = _pd->ndims();
    if (ndims < 3 || ndims > 5) assert(!"Invalid ndims!");

    KD = ndims_pick(jcp.kd, 1, 1);
    KH = ndims_pick(jcp.kh, jcp.kh, 1);
    KW = jcp.kw;

    EXT_KD = ndims_pick(jcp.ext_kd, 1, 1);
    EXT_KH = ndims_pick(jcp.ext_kh, jcp.ext_kh, 1);
    EXT_KW = jcp.ext_kw;

    ODP = ndims_pick(jcp.odp, 1, 1);
    OHP = ndims_pick(jcp.ohp, jcp.ohp, 1);
    OWP = jcp.owp;

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

    oc_chunks = div_up(jcp.nb_oc, jcp.nb_oc_blocking);

    // const variables used for address calculations
    src_w_sz = static_cast<dim_t>(OW) * jcp.ngroups * jcp.oc_without_padding;
    src_h_sz = OH * src_w_sz;
    src_d_sz = OD * src_h_sz;
    dst_w_sz = static_cast<dim_t>(IW) * jcp.ic_without_padding;
    dst_h_sz = IH * dst_w_sz;
    dst_d_sz = ID * dst_h_sz;

    wei_oc_sz = static_cast<dim_t>(jcp.ocp) * jcp.ic_block;
    wei_kw_sz = KW * wei_oc_sz;
    wei_kh_sz = KH * wei_kw_sz;
    wei_kd_sz = KD * wei_kh_sz;
    wei_icb_sz = jcp.nb_ic * wei_kd_sz;

    const auto comp_buffer_iw = jcp.exec_type == exec_trans ? jcp.iw : 1;
    comp_iw_sz = static_cast<dim_t>(jcp.ic_block);
    comp_kw_sz = comp_buffer_iw * comp_iw_sz;
    comp_ker_sz = jcp.ker_ranges_size * comp_kw_sz;
    comp_icb_sz = jcp.nb_ic * comp_ker_sz;

    need_compensation = (jcp.src_zero_point || jcp.s8s8_compensation_required)
            && !jcp.req_brg_comp_pad;

    need_postwork = jcp.with_bias || jcp.with_eltwise || jcp.with_binary
            || jcp.with_src_scales || jcp.with_wei_scales || jcp.with_dst_scales
            || need_compensation || (jcp.dst_dt != jcp.acc_dt) || jcp.with_sum
            || jcp.use_M_mask || jcp.src_zero_point || jcp.dst_zero_point;

    // ---- Initialize arrays ---------------------
    brg_kernels_.resize(_pd->brgs_sz_);
    brgemm_palettes_.resize(_pd->brgs_sz_);

    const dim_t num_po_kernels = nstl::max<dim_t>(jcp.M, jcp.M_tail);
    kernels_po_.resize(num_po_kernels * 2 * 2);
    for (int i = 0; i < num_po_kernels; i++) {
        for_(int i_init = 0; i_init < 2; i_init++)
        for (int i_N = 0; i_N < 2; i_N++)
            kernels_po_[get_ker_po_idx(i, i_init, i_N)] = nullptr;
    }

    if (jcp.exec_type == exec_trans) {
        CHECK(safe_ptr_assign(copy_to_pbuffer_,
                new jit_avx512_core_brgemm_conv_bwd_trans_kernel_t<Vmm>(jcp)));
        CHECK(copy_to_pbuffer_->create_kernel());
        if (jcp.has_uneven_iw) {
            CHECK(safe_ptr_assign(copy_to_output_buffer_,
                    new jit_avx512_core_brgemm_conv_bwd_copy_kernel_t<Vmm>(
                            jcp)));
            CHECK(copy_to_output_buffer_->create_kernel());
        }
    }

    if (jcp.req_cal_comp_pad) {
        if (is_superset(isa, avx512_core))
            CHECK(safe_ptr_assign(comp_vpad_pbuffer_,
                    new jit_uni_brgemm_conv_comp_pad_kernel_t<Xbyak::Zmm>(
                            jcp)));
        else if (one_of(isa, avx2_vnni, avx2_vnni_2)) {
            CHECK(safe_ptr_assign(comp_vpad_pbuffer_,
                    new jit_uni_brgemm_conv_comp_pad_kernel_t<Xbyak::Ymm>(
                            jcp)));
        } else
            assert(!"Unsupported ISA for comp pad kernel.");
        CHECK(comp_vpad_pbuffer_->create_kernel());
    }

    const auto ow_block = jcp.owp;
    const auto oh_block = jcp.ohp;
    const auto od_block = jcp.odp;

    pbuf_w_sz = (dim_t)jcp.oc_block * ow_block;
    pbuf_h_sz = pbuf_w_sz * oh_block;
    pbuf_d_sz = pbuf_h_sz * od_block;

    is_amx = brgemm_convolution_bwd_utils::is_amx(isa);

    // create brgemm and post-op kernels
    // post-op kernels are only used with exec_base
    create_kernels();

    // precalculate unique kernel combination
    if (jcp.req_cal_comp_pad)
        brgemm_convolution_bwd_utils::precalculate_comp_pad_kernels(
                jcp, &kd_bs, &kd_es, &kh_bs, &kh_es, &kw_bs, &kw_es);

    return status::success;
}

#define data_blk_off(f, n, c, d, h, w) \
    (((f).ndims() == 3) ? (f).blk_off(n, c, w) \
                        : (((f).ndims() == 4) ? (f).blk_off(n, c, h, w) \
                                              : (f).blk_off(n, c, d, h, w)))

template <cpu_isa_t isa>
status_t brgemm_convolution_bwd_strided_t<isa>::execute(
        const exec_ctx_t &ctx) const {
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

    brgemm_bwd_exec_ctx_t brgemm_ctx(ctx, _pd);

    const memory_desc_wrapper diff_src_d(pd()->diff_src_md(0));

    const char *const __restrict diff_dst = brgemm_ctx.diff_dst;
    const char *const __restrict wei = brgemm_ctx.weights;
    char *const __restrict diff_src = brgemm_ctx.diff_src;
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const auto extra_data_offset
            = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<char *>(brgemm_ctx.weights);
    const auto s8s8_comp_offset = jcp.req_cal_comp_pad
            ? jcp.ngroups * jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw * jcp.ic_block
            : jcp.ngroups * jcp.nb_ic * jcp.ic_block;
    int32_t *s8s8_compensation = jcp.s8s8_compensation_required
            ? reinterpret_cast<int32_t *>(w + extra_data_offset)
            : nullptr;
    int32_t *zp_compensation = jcp.src_zero_point
            ? reinterpret_cast<int32_t *>(&w[extra_data_offset])
                    + (jcp.s8s8_compensation_required ? s8s8_comp_offset : 0)
            : nullptr;

    brgemm_batch_element_t *const __restrict brg_batch_global
            = (jcp.brg_type == brgemm_strd && jcp.exec_type != exec_vpad)
            ? nullptr
            : scratchpad.template get<brgemm_batch_element_t>(
                      key_brgemm_primitive_batch);
    char *const __restrict c_buffer_global = (jcp.use_buffer)
            ? scratchpad.template get<char>(key_brgemm_primitive_buffer)
            : nullptr;

    auto inp_p_buffer = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<char>(key_conv_brgemm_inp_buffer)
            : nullptr;
    auto inp_p_buffer_mask = (jcp.exec_type == exec_trans)
            ? scratchpad.template get<uint8_t>(key_conv_brgemm_inp_buffer_mask)
            : nullptr;

    auto out_p_buffer = (jcp.exec_type == exec_trans && jcp.has_uneven_iw)
            ? scratchpad.template get<char>(key_conv_brgemm_out_buffer)
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

    const dim_t work_amount = static_cast<dim_t>(jcp.mb) * jcp.ngroups
            * jcp.nb_ic * jcp.nb_id * jcp.nb_ih * jcp.nb_iw;

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

        char *const __restrict out_buffer
                = (jcp.exec_type == exec_trans && jcp.has_uneven_iw)
                ? out_p_buffer + dst_dsz * ithr * jcp.out_buffer_size
                : nullptr;

        uint8_t *__restrict inp_buffer_mask = (jcp.exec_type == exec_trans)
                ? inp_p_buffer_mask + ithr * jcp.inp_buffer_mask_size
                : nullptr;

        char *const wsp_tile = is_amx
                ? wsp_tile_global + ithr * 2 * brgemm_convolution_bwd_utils::P4K
                : nullptr;

        char *const fp8_convert_wsp = jcp.req_fp8_convert_wsp
                ? fp8_convert_wsp_base + ithr * jcp.fp8_convert_wsp_size
                : nullptr;

        float *dst_scales_inv_ptr = nullptr;
        if (jcp.with_dst_scales) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr
                    = scratchpad.template get<float>(key_conv_dst_scales)
                    + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        dim_t start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);
        dim_t n {0}, g {0}, icb {0}, idb {0}, ihb {0}, iwb {0};

        nd_iterator_init(start, n, jcp.mb, idb, jcp.nb_id, ihb, jcp.nb_ih, iwb,
                jcp.nb_iw, g, jcp.ngroups, icb, jcp.nb_ic);

        if (jcp.loop_order == loop_ndhwgc)
            nd_iterator_init(start, n, jcp.mb, idb, jcp.nb_id, ihb, jcp.nb_ih,
                    iwb, jcp.nb_iw, g, jcp.ngroups, icb, jcp.nb_ic);
        else if (jcp.loop_order == loop_ngcdhw)
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, icb, jcp.nb_ic,
                    idb, jcp.nb_id, ihb, jcp.nb_ih, iwb, jcp.nb_iw);
        else
            assert(!"Unknown loop order");

        brgemm_bwd_thread_ctx_t btc(brgemm_ctx, ithr, brg_batch, c_buffer,
                out_buffer, wsp_tile, fp8_convert_wsp);

        dim_t last_n = -1;
        dim_t last_g = -1;
        dim_t last_occ = -1;
        dim_t last_idb = -1;
        dim_t last_ihb = -1;
        dim_t last_iwb = -1;
        for (auto work = start; work < end; work++) {
            btc.g = g;
            btc.n = n;
            btc.icb = icb;
            btc.idb = idb;
            btc.ihb = ihb;
            btc.iwb = iwb;
            btc.src_scales = src_scales;
            btc.wei_scales = wei_scales;
            btc.dst_scales = dst_scales_inv_ptr;
            btc.src_zp_val = src_zero_points ? src_zero_points[0] : 0;
            btc.dst_zp_vals = dst_zero_points;
            btc.src_zp_comp_ptr
                    = jcp.src_zero_point ? src_zp_comp_base : nullptr;
            btc.s8s8_comp_ptr
                    = jcp.s8s8_compensation_required ? s8s8_comp_base : nullptr;

            auto id_begin = idb * jcp.id_block;
            auto id_end = nstl::min<dim_t>(ID, id_begin + jcp.id_block);
            auto ih_begin = ihb * jcp.ih_block;
            auto ih_end = nstl::min<dim_t>(IH, ih_begin + jcp.ih_block);
            auto iw_begin = iwb * jcp.iw_block;

            for_(dim_t id = id_begin; id < id_end; id++)
            for (dim_t ih = ih_begin; ih < ih_end; ih++) {
                // Zero out_buffer before processing the tail iw block to
                // prevent stale scratchpad data from leaking into positions
                // that may not be written (e.g., stride sectors with no
                // valid kernel overlap).
                if (jcp.exec_type == exec_trans && jcp.has_uneven_iw
                        && iwb == jcp.nb_iw - 1 && out_buffer) {
                    std::memset(out_buffer, 0, dst_dsz * jcp.out_buffer_size);
                }
                for (int occ = 0; occ < oc_chunks; occ++) {
                    btc.id = id;
                    btc.ih = ih;
                    btc.occ = occ;

                    if (jcp.exec_type == exec_base) {
                        for (int sw = 0; sw < SW; sw++) {
                            btc.sw = sw;
                            ker_base(btc);
                        }
                    } else if (jcp.exec_type == exec_trans) {
                        maybe_trans_inp(ithr, diff_dst, inp_buffer,
                                inp_buffer_mask, g, n, occ, idb, ihb, iwb,
                                last_g, last_n, last_occ, last_idb, last_ihb,
                                last_iwb);
                        for (int sw = 0; sw < SW; sw++) {
                            btc.sw = sw;
                            ker_trans(btc, inp_buffer);
                        }
                    } else
                        assert(!"Unknown exec type");

                    last_n = n;
                    last_g = g;
                    last_occ = occ;
                    last_idb = idb;
                    last_ihb = ihb;
                    last_iwb = iwb;
                }

                if (jcp.exec_type == exec_trans && jcp.has_uneven_iw
                        && iwb == jcp.nb_iw - 1) {
                    const bool is_ic_tail
                            = jcp.ic - (btc.icb * jcp.ic_block) < jcp.ic_block;
                    const dim_t ic_size
                            = is_ic_tail ? jcp.ic % jcp.ic_block : jcp.ic_block;

                    auto cp = jit_brgemm_conv_bwd_copy_kernel_args_t();
                    cp.src = btc.out_buffer;
                    cp.dst = diff_src
                            + data_blk_off(diff_src_d, n,
                                      g * jcp.ic + icb * jcp.ic_block, id, ih,
                                      iw_begin)
                                    * jcp.dst_dsz;
                    cp.num_ic = ic_size;
                    (*copy_to_output_buffer_)(&cp);
                }
            }
            if (jcp.loop_order == loop_ndhwgc)
                nd_iterator_step(n, jcp.mb, idb, jcp.nb_id, ihb, jcp.nb_ih, iwb,
                        jcp.nb_iw, g, jcp.ngroups, icb, jcp.nb_ic);
            else if (jcp.loop_order == loop_ngcdhw)
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, icb, jcp.nb_ic, idb,
                        jcp.nb_id, ihb, jcp.nb_ih, iwb, jcp.nb_iw);
            else
                assert(!"Unknown loop order");
        }
        if (is_amx) { amx_tile_release(); }
    });

    return status::success;
}

#undef data_blk_off

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::cal_compensation(
        const char *__restrict weights, int32_t *src_zp_buffer,
        int32_t *s8s8_comp_buffer) const {
    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (!jcp.req_cal_comp_pad) return;

    const auto comp_buffer_iw = jcp.exec_type == exec_trans ? jcp.iw : 1;
    const auto work_amount
            = static_cast<dim_t>(jcp.ngroups) * jcp.nb_ic * jcp.ker_ranges_size;
    const auto is_small_shape = work_amount <= jcp.nthr
            && (work_amount * jcp.ic_block * jcp.ocp * comp_buffer_iw
                    <= platform::get_per_core_cache_size(1));
    const int nthr = is_small_shape ? 1 : jcp.nthr;

    parallel(nthr, [= COMPAT_THIS_CAPTURE](const int ithr, const int nthr) {
        if (ithr >= work_amount) return;

        dim_t start {0}, end {0};
        dim_t g {0}, icb {0}, k {0};
        balance211(work_amount, nthr, ithr, start, end);
        nd_iterator_init(
                start, g, jcp.ngroups, icb, jcp.nb_ic, k, jcp.ker_ranges_size);
        for (auto work = start; work < end; work++) {
            const dim_t kd_b {kd_bs[k]}, kd_e {kd_es[k]}, kh_b {kh_bs[k]},
                    kh_e {kh_es[k]}, kw_b {kw_bs[k]}, kw_e {kw_es[k]};

            const auto buffer_offs
                    = g * comp_icb_sz + icb * comp_ker_sz + k * comp_kw_sz;
            const auto wei_offs = (g * jcp.nb_ic + icb) * wei_kd_sz
                    + kd_b * wei_kh_sz + (kh_b * wei_kw_sz) + kw_b * wei_oc_sz;
            if (jcp.src_zero_point && src_zp_buffer)
                std::memset(&src_zp_buffer[buffer_offs], 0,
                        sizeof(int32_t) * comp_kw_sz);
            if (jcp.s8s8_compensation_required && s8s8_comp_buffer)
                std::memset(&s8s8_comp_buffer[buffer_offs], 0,
                        sizeof(int32_t) * comp_kw_sz);
            if (!everyone_is(0, kd_e, kd_b, kh_e, kh_b, kw_e, kw_b)) {
                assert(kd_e > kd_b && kh_e > kh_b && kw_e > kw_b);

                jit_brgemm_conv_comp_pad_args_t p;

                p.kd_l = div_up(nstl::max((dim_t)0, kd_e - kd_b), SD);
                p.kh_l = div_up(nstl::max((dim_t)0, kh_e - kh_b), SH);
                p.kw_l = div_up(nstl::max((dim_t)0, kw_e - kw_b), SW);
                p.use_inversion = false;

                p.ptr_in = &weights[wei_offs];
                p.ptr_zp_out = jcp.src_zero_point ? &src_zp_buffer[buffer_offs]
                                                  : nullptr;
                p.ptr_cp_out = jcp.s8s8_compensation_required
                        ? &s8s8_comp_buffer[buffer_offs]
                        : nullptr;

                (*comp_vpad_pbuffer_)(&p);
            }

            // Broadcast center position s8s8 compensation to all positions.
            // For exec_trans + s8s8, the brgemm adds +128 shift to ALL input
            // values including zero-padding. Boundary positions need the same
            // full-range compensation as center. The JIT kernel computes
            // boundary-limited values; broadcast the center to fix them.
            if (jcp.exec_type == exec_trans && jcp.s8s8_compensation_required
                    && s8s8_comp_buffer) {
                const auto SW = jcp.stride_w;
                const auto nb_iw = div_up(jcp.iw, SW);
                const auto ic_block = jcp.ic_block;
                const auto center_iwb = nb_iw / 2;
                for (int sw = 0; sw < SW; sw++) {
                    const auto center_comp_iw = sw * nb_iw + center_iwb;
                    if (center_comp_iw >= jcp.iw) continue;
                    const auto center_offs
                            = buffer_offs + center_comp_iw * ic_block;
                    for (int iwb = 0; iwb < nb_iw; iwb++) {
                        const auto comp_iw = sw * nb_iw + iwb;
                        if (comp_iw >= jcp.iw || iwb == center_iwb) continue;
                        const auto dst_offs = buffer_offs + comp_iw * ic_block;
                        std::memcpy(&s8s8_comp_buffer[dst_offs],
                                &s8s8_comp_buffer[center_offs],
                                ic_block * sizeof(int32_t));
                    }
                }
            }

            nd_iterator_step(
                    g, jcp.ngroups, icb, jcp.nb_ic, k, jcp.ker_ranges_size);
        }
    });
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::perform_outwork(char *dst_base,
        char *dst, char *c_buffer, const char *bias_w, dim_t id, dim_t ih,
        dim_t iw, dim_t iw_raw, dim_t g_ic, bool is_ic_tail, dim_t ker_iw_s,
        dim_t ker_iw_f, dim_t kd_l, dim_t kh_l,
        const void *post_ops_binary_rhs_arg_vec, int32_t src_zp_val,
        int32_t *src_zp_ptr, const int32_t *dst_zp_ptr,
        int32_t *s8s8_compensation, size_t comp_ker_offs, bool maybe_do_init,
        bool do_postwork, bool do_post_comp, const void *src_scales,
        const void *wei_scales, const void *dst_scales) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    const auto do_init
            = maybe_do_init && IMPLICATION(jcp.with_sum, jcp.use_buffer);
    if (!do_init && !do_postwork) return;

    assert(!jcp.is_is_blocking);

    const bool is_iw_tail = (IW - iw_raw < jcp.iw_block);

    const auto M = div_up(is_iw_tail ? jcp.iw_tail : jcp.iw_block, SW) * SW;

    const auto kdh_l = kd_l * kh_l;
    const auto iw_s = (kdh_l <= 0) ? iw : ker_iw_s;
    auto iw_f = (kdh_l <= 0) ? iw : ker_iw_f;
    assert(iw <= iw_s && iw_s <= iw_f && iw_f <= iw + M);

    brgemm_kernel_post_ops_args_t p;
    if (do_postwork) {
        p.ptr_bias = (void *)(bias_w);
        p.ptr_binary_post_ops_rhs = post_ops_binary_rhs_arg_vec;
        p.dst_orig = dst;
        p.c_zp_values = dst_zp_ptr;
        p.a_comp_val = src_zp_val;
        p.ptr_src_scales = src_scales;
        p.ptr_wei_scales = wei_scales ? static_cast<const char *>(wei_scales)
                        + jcp.is_ic_scale * g_ic * sizeof(float)
                                      : nullptr;
        p.ptr_dst_scales = dst_scales;
    }

    auto call_outwork_ker = [&](bool is_postwork, bool has_postcomp,
                                    dim_t iw_pw_s, dim_t iw_pw_l) {
        auto ker_po_idx = get_ker_po_idx(iw_pw_l - 1, is_postwork, is_ic_tail);
        const auto outwork_ker = kernels_po_[ker_po_idx].get();
        const auto comp_iw_s = get_comp_iw(iw_pw_s);
        assert(outwork_ker != nullptr
                && iw_pw_l == outwork_ker->get_bcast_dim());
        if (is_postwork) {
            p.apply_comp = has_postcomp;
            p.a_zp_compensation = has_postcomp && jcp.src_zero_point
                    ? &src_zp_ptr[comp_ker_offs + comp_iw_s * jcp.LDB]
                    : src_zp_ptr;
            p.s8s8_compensation = has_postcomp && jcp.s8s8_compensation_required
                    ? &s8s8_compensation[comp_ker_offs + comp_iw_s * jcp.LDB]
                    : s8s8_compensation;

            p.ptr_out = dst_base
                    + dst_dsz
                            * (id * dst_h_sz + ih * dst_w_sz
                                    + iw_pw_s * jcp.ic_without_padding);
            p.ptr_in = static_cast<void *>(jcp.use_buffer
                            ? (c_buffer
                                      + acc_dsz
                                              * div_up(nstl::max<dim_t>(
                                                               0, iw_pw_s - iw),
                                                      SW)
                                              * jcp.LDC)
                            : p.ptr_out);
        } else {
            p.apply_comp = has_postcomp;
            char *const ptr_Cz = jcp.use_buffer
                    ? (c_buffer
                              + acc_dsz
                                      * div_up(
                                              nstl::max<dim_t>(0, iw_pw_s - iw),
                                              SW)
                                      * jcp.LDC)
                    : dst_base
                            + dst_dsz
                                    * (id * dst_h_sz + ih * dst_w_sz
                                            + iw_pw_s * jcp.ic_without_padding);
            p.ptr_out = static_cast<void *>(ptr_Cz);
        }
        (*outwork_ker)(&p);
    };

    if (iw < iw_s) {
        const auto iw_pw_l = (iw_s - iw) / SW;
        if (do_init) call_outwork_ker(false, false, iw, iw_pw_l);
        if (do_postwork) call_outwork_ker(true, do_post_comp, iw, iw_pw_l);
    }
    if (iw_f < iw + M) {
        const auto iw_pw_l = (iw + M - iw_f) / SW;
        if (do_init) call_outwork_ker(false, false, iw_f, iw_pw_l);
        if (do_postwork) call_outwork_ker(true, do_post_comp, iw_f, iw_pw_l);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::call_brgemm_kernel(
        brgemm_bwd_thread_ctx_t &btc, int brg_idx, int batch_size, char *ptr_C,
        char *ptr_D, const char *bias_w, dim_t g_ic, bool do_postops,
        const void *binary_post_ops_rhs, int32_t src_zp_val,
        int32_t *src_zp_ptr, const int32_t *dst_zp_ptr, int32_t *s8s8_comp,
        bool do_only_comp, bool is_first_call_postops,
        const char *dst_orig_override) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;

    if (brg_idx < 0) return;
    const auto brg_ker = brg_kernels_[brg_idx];
    if (!brg_ker) return;
    if (is_first_call_postops) return;

    const auto do_only_pass_comp = !do_postops && jcp.src_zero_point
            && (jcp.req_brg_comp_pad || jcp.max_vpad > 0);
    const auto do_skip_accm = batch_size == 0;

    brgemm_palettes_.maybe_tile_configure(is_amx, btc.cur_brg_idx, brg_idx);
    const auto maybe_do_postops = one_of(
            true, do_postops, do_only_comp, do_only_pass_comp, do_skip_accm);
    if (maybe_do_postops) {
        const auto dst_orig = dst_orig_override ? dst_orig_override
                                                : btc.brgemm_ctx.diff_src;
        const brgemm_post_ops_data_t post_ops_data {
                static_cast<const char *>(bias_w), binary_post_ops_rhs,
                static_cast<size_t>(g_ic), 0, dst_orig, 0,
                static_cast<void *>(src_zp_ptr), nullptr, dst_zp_ptr,
                do_skip_accm, src_zp_val, do_only_comp, do_only_pass_comp,
                btc.src_scales,
                btc.wei_scales ? static_cast<const char *>(btc.wei_scales)
                                + jcp.is_ic_scale * g_ic * sizeof(float)
                               : nullptr,
                btc.dst_scales};

        void *scratch = is_amx ? static_cast<void *>(btc.wsp_tile)
                : jcp.req_fp8_convert_wsp
                ? static_cast<void *>(btc.fp8_convert_wsp)
                : static_cast<void *>(s8s8_comp);

        if (do_postops || do_skip_accm) {
            brgemm_kernel_execute_postops(brg_ker, batch_size, btc.brg_batch,
                    ptr_C, ptr_D, post_ops_data, scratch);
        } else
            brgemm_kernel_execute_postops(brg_ker, batch_size, btc.brg_batch,
                    ptr_C, ptr_C, post_ops_data, scratch);
    } else {
        void *scratch = is_amx ? static_cast<void *>(btc.wsp_tile)
                               : static_cast<void *>(btc.fp8_convert_wsp);
        brgemm_kernel_execute(
                brg_ker, batch_size, btc.brg_batch, ptr_C, scratch);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::maybe_trans_inp(int ithr,
        const char *__restrict src, char *__restrict inp_buffer,
        uint8_t *__restrict inp_buffer_mask, dim_t g, dim_t n, dim_t occ,
        dim_t idb, dim_t ihb, dim_t iwb, dim_t last_g, dim_t last_n,
        dim_t last_occ, dim_t last_idb, dim_t last_ihb, dim_t last_iwb) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    const auto ocb = occ * jcp.nb_oc_blocking;

    // This function does not work correctly if spatial block is lower than
    // spatial size and not divisible by corresponding stride.
    // TODO: drop this restriction.
    assert(IMPLICATION(jcp.iw_block < jcp.iw, jcp.iw_block % jcp.stride_w == 0)
            && IMPLICATION(
                    jcp.ih_block < jcp.ih, jcp.ih_block % jcp.stride_h == 0)
            && IMPLICATION(
                    jcp.id_block < jcp.id, jcp.id_block % jcp.stride_d == 0));

    if (last_g == g && last_n == n && last_occ == occ && last_idb == idb
            && last_ihb == ihb && last_iwb == iwb)
        return;

    auto cp = jit_brgemm_conv_bwd_trans_kernel_args_t();

    const auto oc = ocb * jcp.oc_block;
    const auto g_oc = g * jcp.oc + oc;

    const auto sw = jcp.l_pad % jcp.stride_w;
    const auto kw = (jcp.kw - 1) % jcp.stride_w;
    const auto kw_x = (jcp.kw - 1) - nstl::modulo(kw - sw, jcp.stride_w);
    const auto ow = (iwb * jcp.iw_block + jcp.l_pad - kw_x * (jcp.dilate_w + 1))
            / jcp.stride_w;

    dim_t od_start {0}, od_end {0}, oh_start {0}, oh_end {0};

    const auto sh = jcp.t_pad % jcp.stride_h;
    const auto kh = (jcp.kh - 1) % jcp.stride_h;
    const auto kh_x = (jcp.kh - 1) - nstl::modulo(kh - sh, jcp.stride_h);
    oh_start = (ihb * jcp.ih_block + jcp.t_pad - kh_x * (jcp.dilate_h + 1))
            / jcp.stride_h;
    oh_end = oh_start + jcp.oh_block;

    const auto sd = jcp.f_pad % jcp.stride_d;
    const auto kd = (jcp.kd - 1) % jcp.stride_d;
    const auto kd_x = (jcp.kd - 1) - nstl::modulo(kd - sd, jcp.stride_d);
    od_start = (idb * jcp.id_block + jcp.f_pad - kd_x * (jcp.dilate_d + 1))
            / jcp.stride_d;
    od_end = od_start + jcp.od_block;

    const dim_t rows_to_copy
            = nstl::min<dim_t>(jcp.oh, oh_end) - nstl::max<dim_t>(0, oh_start);
    cp.iwb = iwb;
    cp.oc = oc;
    const dim_t ow_buf = ow;
    dim_t inp_offset_start, out_offset_start;

    cp.t_pad = 0;
    cp.b_pad = 0;
    cp.h_count = nstl::max<dim_t>(0, rows_to_copy);

    const dim_t oh_buf = nstl::max<dim_t>(0, oh_start);

    inp_offset_start = static_cast<dim_t>(n) * src_d_sz
            + nstl::max<dim_t>(0, oh_start) * src_w_sz
            + nstl::max<dim_t>(0, ow) * jcp.ngroups * jcp.oc_without_padding
            + g_oc;
    out_offset_start = oh_buf * pbuf_w_sz + ow_buf * jcp.oc_block;

    for (dim_t od = nstl::max<dim_t>(0, od_start);
            od < nstl::min<dim_t>(jcp.od, od_end); od++) {
        const dim_t inp_offset = inp_offset_start + od * src_h_sz;
        const dim_t od_buf = od;
        const dim_t out_offset = out_offset_start + od_buf * pbuf_h_sz;
        cp.src = src + src_dsz * inp_offset;
        cp.dst = inp_buffer + src_dsz * out_offset;

        (*copy_to_pbuffer_)(&cp);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::ker_base(
        brgemm_bwd_thread_ctx_t &btc) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims();

    const char *const __restrict weights = btc.brgemm_ctx.weights;
    const char *const __restrict bias = btc.brgemm_ctx.bias;
    char *const __restrict diff_src = btc.brgemm_ctx.diff_src;
    const auto diff_dst = btc.brgemm_ctx.diff_dst;
    const std::vector<const void *> &post_ops_binary_rhs_arg_vec
            = btc.brgemm_ctx.post_ops_binary_rhs_arg_vec;
    const dim_t ic = btc.icb * jcp.ic_block;
    const dim_t g_ic = btc.g * jcp.ic + ic;
    const dim_t ocb = btc.occ * jcp.nb_oc_blocking;
    const dim_t oc = ocb * jcp.oc_block;
    const dim_t g_oc = btc.g * jcp.oc + oc;
    const dim_t iw = static_cast<dim_t>(btc.iwb) * jcp.iw_block + btc.sw;
    const dim_t iw_raw = static_cast<dim_t>(btc.iwb) * jcp.iw_block;
    const dim_t ih = btc.ih;
    const dim_t id = btc.id;
    const bool is_oc_tail
            = (btc.occ == oc_chunks - 1 && ((jcp.oc - oc) % jcp.oc_block != 0));
    const bool is_ic_tail = (jcp.ic - ic < jcp.ic_block);
    const char *const __restrict bias_w
            = bias ? bias + (bias_d.blk_off(g_ic) * bia_dsz) : nullptr;
    int kw_s {0}, kw_full_s {0}, kw_f {0}, kw_full_f {0};
    dim_t kw_b {0}, kw_e {0};
    int kd_s_(0), kh_s_(0), kd_f_(0), kh_f_(0);
    _pd->get_kw_range(static_cast<int>(iw), static_cast<int>(iw_raw), kw_s,
            kw_full_s, kw_full_f, kw_f);

    // od = (id + FP - kd * DD) / SD <-- general relation for all sets of (od, id, kd) that overlap
    // for a given index from diff_src, we need to find the appropriate stride sector
    // omiting kw range, because it was already calculated inside get_kw_range()
    brgemm_convolution_bwd_utils::set_k_range(
            FP, DD, SD, id, OD, KD, kd_s_, kd_f_);
    brgemm_convolution_bwd_utils::set_k_range(
            TP, DH, SH, ih, OH, KH, kh_s_, kh_f_);

    const auto kh_f = ndims_pick(kh_f_, kh_f_, 1);
    const auto kh_s = ndims_pick(kh_s_, kh_s_, 0);

    const auto kd_f = ndims_pick(kd_f_, 1, 1);
    const auto kd_s = ndims_pick(kd_s_, 0, 0);

    const auto diff_dst_base = diff_dst + src_dsz * (btc.n * src_d_sz + g_oc);
    char *const __restrict diff_src_base
            = diff_src + dst_dsz * (btc.n * dst_d_sz + g_ic);
    const auto wei_base
            = weights + wei_dsz * (btc.g * wei_icb_sz + btc.icb * wei_kd_sz);
    const dim_t nb_oc_b = nstl::min<dim_t>(jcp.nb_oc_blocking, jcp.nb_oc - ocb)
            - (is_oc_tail ? 1 : 0);
    char *ptr_C;
    char *ptr_D;
    dim_t kd_b(0), kd_e(0), kh_b(0), kh_e(0), k_l(0);

    const auto kd_l_full = kd_f - kd_s;
    const auto kh_l_full = kh_f - kh_s;

    bool is_first_call_postops = false,
         is_first_call_postops_state_changed = false;
    const auto call_brgemm = [&](dim_t iw, int brg_idx, dim_t oc_block_s,
                                     dim_t n_oc_blocks, size_t comp_ker_offs,
                                     bool do_postops, bool do_only_comp) {
        if (brg_idx < 0) {
            assert(!"Requested brgemm kernel was not created.");
            return;
        }
        const auto kh_ee = kh_e;
        int k_sum = 0;
        int32_t *src_zp = jcp.src_zero_point
                ? &btc.src_zp_comp_ptr[comp_ker_offs]
                : nullptr;
        int32_t *s8s8_comp = jcp.s8s8_compensation_required
                ? &btc.s8s8_comp_ptr[comp_ker_offs]
                : nullptr;
        for (dim_t i_ocb = 0; i_ocb < n_oc_blocks; i_ocb++) {
            const auto oc_off = (oc_block_s + i_ocb) * jcp.oc_block;
            const auto src_oc = oc_off;
            const auto wei_oc = oc + oc_off;
            const auto n_ocb_off = i_ocb * k_l;
            const auto diff_dst_base_oc = diff_dst_base + src_dsz * src_oc;
            const auto wei_base_oc = wei_base + wei_dsz * wei_oc * jcp.ic_block;

            auto k = 0;
            for (dim_t kd = kd_b; kd < kd_e; kd++) {
                auto od = (id - kd * DD + FP);
                if (od % SD != 0) continue;
                od /= SD;
                const auto diff_dst_base_kd
                        = diff_dst_base_oc + src_dsz * od * src_h_sz;
                const auto wei_base_kd = wei_base_oc + wei_dsz * kd * wei_kh_sz;
                for (dim_t kh = kh_b; kh < kh_ee; kh++) {
                    auto oh = (ih - kh * DH + TP);
                    if (oh % SH != 0) continue;
                    oh /= SH;
                    const auto diff_dst_base_kh
                            = diff_dst_base_kd + src_dsz * oh * src_w_sz;
                    const auto wei_base_kh
                            = wei_base_kd + wei_dsz * kh * wei_kw_sz;
                    for (dim_t kw = kw_b; kw < kw_e; kw += SW) {
                        auto ow = (iw - kw * DW + LP) / SW;
                        // inp_buffer layout is Cdhw<oc_block>c
                        btc.brg_batch[n_ocb_off + k].ptr.A = diff_dst_base_kh
                                + src_dsz * (ow /*+ jcp.l_ovf*/) * jcp.ngroups
                                        * jcp.oc_without_padding;
                        btc.brg_batch[n_ocb_off + k].vvpad.top = 0;
                        btc.brg_batch[n_ocb_off + k].vvpad.bottom = 0;
                        // general wei layout is gIdhwO<block_i><block_o>
                        btc.brg_batch[n_ocb_off + k].ptr.B
                                = wei_base_kh + wei_dsz * kw * wei_oc_sz;
                        k++;
                    }
                }
            }
            k_sum += k;
        }
        call_brgemm_kernel(btc, brg_idx, k_sum, ptr_C, ptr_D, bias_w, g_ic,
                do_postops, post_ops_binary_rhs_arg_vec.data(), btc.src_zp_val,
                src_zp, btc.dst_zp_vals, s8s8_comp, do_only_comp,
                is_first_call_postops);
        if (!is_first_call_postops_state_changed) {
            is_first_call_postops = k_sum == 0; // do_skip_accm
            is_first_call_postops_state_changed = true;
        }

        MAYBE_UNUSED(bias_w);
        MAYBE_UNUSED(ptr_C);
        MAYBE_UNUSED(post_ops_binary_rhs_arg_vec);
    };

    const auto kdhw_loop = [&]() {
        if (kw_e - kw_b <= 0 || kw_b >= jcp.kw) return;
        int iw_b {0}, M_without_overflow {0};
        _pd->get_iw_range(static_cast<int>(iw), static_cast<int>(iw_raw),
                static_cast<int>(kw_b), iw_b, M_without_overflow);
        const auto iw_e = iw_b + M_without_overflow;
        const auto do_init
                = btc.occ == 0 && kd_b == kd_s && kh_b == kh_s && kw_b == kw_s;
        const auto do_postwork = need_postwork && btc.occ == (oc_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f
                && (kw_b + SW >= kw_f || kw_e == kw_f);
        const auto do_only_comp = !do_postwork && need_compensation
                && kd_e == kd_f && kh_e == kh_f && kw_e != kw_f
                && btc.occ == (oc_chunks - 1);
        if (iw_e - iw_b <= 0 && !do_init && !do_postwork) return;

        const dim_t kd_l = div_up(nstl::max<dim_t>(0, kd_e - kd_b), SD);
        const dim_t kh_l = div_up(nstl::max<dim_t>(0, kh_e - kh_b), SH);
        const dim_t kw_l = div_up(nstl::max<dim_t>(0, kw_e - kw_b), SW);
        k_l = kd_l * kh_l * kw_l;
        const auto iw_l = iw_e - iw_b;

        ptr_D = diff_src_base
                + dst_dsz
                        * (btc.id * dst_h_sz + btc.ih * dst_w_sz
                                + iw_b * jcp.ic_without_padding);

        ptr_C = (jcp.use_buffer) ? btc.c_buffer
                        + acc_dsz * div_up(nstl::max((dim_t)0, iw_b - iw), SW)
                                * jcp.LDC
                                 : static_cast<char *>(ptr_D);

        assert(0 <= iw_l && iw_l <= jcp.iw_block);

        const auto comp_ker_offs = get_comp_offset(
                btc.g, btc.icb, 0, kd_s, kd_f, kh_s, kh_f, kw_b, kw_e);

        const auto ker_i = iw_l;

        if (iw_l > 0 && k_l > 0) {
            if (nb_oc_b > 0) {
                const int brg_idx
                        = _pd->get_brg_idx(ker_i, do_init, is_ic_tail, false);
                call_brgemm(iw_b, brg_idx, 0, nb_oc_b, comp_ker_offs,
                        do_postwork && !is_oc_tail, do_only_comp);
            }

            if (is_oc_tail) {
                const auto use_init_ker = (do_init && nb_oc_b == 0);
                const int brg_oc_tail_idx = _pd->get_brg_idx(
                        ker_i, use_init_ker, is_ic_tail, true);
                call_brgemm(iw_b, brg_oc_tail_idx, nb_oc_b, 1, comp_ker_offs,
                        do_postwork, do_only_comp);
            }
        }

        const auto iw_ee = iw_b + (M_without_overflow * SW);
        perform_outwork(diff_src_base, diff_src, btc.c_buffer, bias_w, btc.id,
                btc.ih, iw, iw_raw, g_ic, is_ic_tail, iw_b, iw_ee, kd_l, kh_l,
                post_ops_binary_rhs_arg_vec.data(), btc.src_zp_val,
                btc.src_zp_comp_ptr, btc.dst_zp_vals, btc.s8s8_comp_ptr,
                comp_ker_offs, do_init, do_postwork, false, btc.src_scales,
                btc.wei_scales, btc.dst_scales);
    };

    if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s && kw_s < jcp.kw) {
        if (kw_s < kw_full_s) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min<dim_t>(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min<dim_t>(kh_f, kh_b + KH_BLOCK_PAD);
                    for (dim_t kw = kw_s; kw < kw_full_s; kw += SW) {
                        kw_b = kw;
                        kw_e = kw + 1;
                        kdhw_loop();
                    }
                }
            }
        }

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

        if (kw_full_f < kw_f) {
            for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK_PAD) {
                kd_e = nstl::min<dim_t>(kd_f, kd_b + KD_BLOCK_PAD);
                for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK_PAD) {
                    kh_e = nstl::min<dim_t>(kh_f, kh_b + KH_BLOCK_PAD);
                    for (dim_t kw = kw_full_f; kw < kw_f; kw += SW) {
                        kw_b = kw;
                        kw_e = kw + 1;
                        kdhw_loop();
                    }
                }
            }
        }
    } else {
        const auto do_init = btc.occ == 0;
        const auto do_postwork = need_postwork && btc.occ == (oc_chunks - 1);
        perform_outwork(diff_src_base, diff_src, btc.c_buffer, bias_w, btc.id,
                btc.ih, iw, iw_raw, g_ic, is_ic_tail, iw, iw, kd_l_full,
                kh_l_full, post_ops_binary_rhs_arg_vec.data(), btc.src_zp_val,
                btc.src_zp_comp_ptr, btc.dst_zp_vals, btc.s8s8_comp_ptr, 0,
                do_init, do_postwork, false, btc.src_scales, btc.wei_scales,
                btc.dst_scales);
    }
}

template <cpu_isa_t isa>
void brgemm_convolution_bwd_strided_t<isa>::ker_trans(
        brgemm_bwd_thread_ctx_t &btc, char *inp_buffer) const {

    const auto _pd = pd();
    const auto &jcp = _pd->jcp_;
    auto ndims = _pd->ndims();

    const char *const __restrict weights = btc.brgemm_ctx.weights;
    const char *const __restrict bias = btc.brgemm_ctx.bias;
    char *const __restrict diff_src = btc.brgemm_ctx.diff_src;
    const std::vector<const void *> &post_ops_binary_rhs_arg_vec
            = btc.brgemm_ctx.post_ops_binary_rhs_arg_vec;
    const dim_t ic = btc.icb * jcp.ic_block;
    const dim_t g_ic = btc.g * jcp.ic + ic;
    const dim_t ocb = btc.occ * jcp.nb_oc_blocking;
    const dim_t oc = ocb * jcp.oc_block;
    const dim_t iw = static_cast<dim_t>(btc.iwb) * jcp.iw_block + btc.sw;
    const dim_t ih = btc.ih;
    const dim_t id = btc.id;

    // od = (id + FP - kd * DD) / SD <-- general relation for all sets of (od, id, kd) that overlap
    // for a given index from diff_src, we need to find the appropriate stride sector
    int kd_s_(0), kh_s_(0), kw_s(0), kd_f_(0), kh_f_(0), kw_f(0);

    brgemm_convolution_bwd_utils::set_k_range(
            FP, DD, SD, id, OD, KD, kd_s_, kd_f_, false);
    brgemm_convolution_bwd_utils::set_k_range(
            TP, DH, SH, ih, OH, KH, kh_s_, kh_f_, false);
    brgemm_convolution_bwd_utils::set_k_range(
            LP, DW, SW, iw, OW, KW, kw_s, kw_f, true);

    const auto kh_f = ndims_pick(kh_f_, kh_f_, 1);
    const auto kh_s = ndims_pick(kh_s_, kh_s_, 0);

    const auto kd_f = ndims_pick(kd_f_, 1, 1);
    const auto kd_s = ndims_pick(kd_s_, 0, 0);

    const bool is_oc_tail
            = (btc.occ == oc_chunks - 1 && ((jcp.oc - oc) % jcp.oc_block != 0));

    const bool is_ic_tail = (jcp.ic - ic < jcp.ic_block);
    const char *const __restrict bias_w
            = bias ? bias + (bias_d.blk_off(g_ic) * bia_dsz) : nullptr;
    const dim_t nb_oc_b = nstl::min<dim_t>(jcp.nb_oc_blocking, jcp.nb_oc - ocb)
            - (is_oc_tail ? 1 : 0);

    const bool is_tail_iwb = btc.iwb == jcp.nb_iw - 1;
    const bool need_out_buffer = jcp.has_uneven_iw && is_tail_iwb;

    char *const __restrict diff_src_base = (need_out_buffer
                    ? btc.out_buffer
                    : diff_src + dst_dsz * (btc.n * dst_d_sz + g_ic));
    char *ptr_C;
    char *ptr_D;
    dim_t kd_b(0), kd_e(0), kh_b(0), kh_e(0), k_l(0);

    const auto wei_base
            = weights + wei_dsz * (btc.g * wei_icb_sz + btc.icb * wei_kd_sz);
    const dim_t iw_b {iw};

    ptr_D = diff_src_base
            + dst_dsz
                    * (need_out_buffer
                                    ? btc.sw * jcp.ic_without_padding
                                    : (id * dst_h_sz + ih * dst_w_sz
                                              + iw_b * jcp.ic_without_padding));

    ptr_C = (jcp.use_buffer) ? btc.c_buffer : static_cast<char *>(ptr_D);

    const auto full_m = jcp.M > 0 ? jcp.M : jcp.M_tail;
    // When need_out_buffer, the short stripe (sw > 0 with odd IW) has fewer
    // valid spatial points. Use the reduced M only for the postops call to
    // avoid OOB binary operand access. Accumulation is safe with full M
    // (ghost column reads/writes are in-bounds in the padded buffers).
    const auto iw_remaining
            = jcp.iw - static_cast<dim_t>(btc.iwb) * jcp.iw_block - btc.sw;
    const auto ker_i_postops = (need_out_buffer && iw_remaining > 0)
            ? static_cast<int>(div_up(iw_remaining, jcp.stride_w))
            : full_m;

    bool is_first_call_postops = false,
         is_first_call_postops_state_changed = false;
    const auto call_brgemm
            = [&](int brg_idx, dim_t oc_block_s, dim_t n_oc_blocks,
                      size_t comp_ker_offs, bool do_postops) {
        if (brg_idx < 0) {
            assert(!"Requested brgemm kernel was not created.");
            return;
        }
        const auto kh_ee = kh_e;
        const auto kw_e = kw_f;
        const auto pbuf_base = inp_buffer;
        int32_t *src_zp = jcp.src_zero_point
                ? &btc.src_zp_comp_ptr[comp_ker_offs]
                : nullptr;
        int32_t *s8s8_comp = jcp.s8s8_compensation_required
                ? &btc.s8s8_comp_ptr[comp_ker_offs]
                : nullptr;

        int k_sum = 0;
        for (dim_t i_ocb = 0; i_ocb < n_oc_blocks; i_ocb++) {
            const auto oc_off = (oc_block_s + i_ocb) * jcp.oc_block;
            const auto wei_oc = oc + oc_off;
            const auto n_ocb_off = i_ocb * k_l;
            const auto pbuf_base_oc = pbuf_base;
            const auto wei_base_oc = wei_base + wei_dsz * wei_oc * jcp.ic_block;

            auto k = 0;
            for (dim_t kd = kd_b; kd < kd_e; kd++) {
                auto od = (id - kd * DD + FP);
                if (od % SD != 0) continue;
                od /= SD;
                const auto pbuf_base_kd
                        = pbuf_base_oc + src_dsz * od * pbuf_h_sz;
                const auto wei_base_kd = wei_base_oc + wei_dsz * kd * wei_kh_sz;
                for (dim_t kh = kh_b; kh < kh_ee; kh++) {
                    auto oh = (ih - kh * DH + TP);
                    if (oh % SH != 0) continue;
                    oh /= SH;
                    const auto pbuf_base_kh
                            = pbuf_base_kd + src_dsz * oh * pbuf_w_sz;
                    const auto wei_base_kh
                            = wei_base_kd + wei_dsz * kh * wei_kw_sz;
                    for (dim_t kw = kw_s; kw < kw_e; kw += SW) {
                        const auto ow = (iw - kw * DW + LP) / SW;
                        // inp_buffer layout is Cdhw<oc_block>c
                        btc.brg_batch[n_ocb_off + k].ptr.A = pbuf_base_kh
                                + src_dsz * (ow + jcp.l_ovf) * jcp.oc_block;
                        btc.brg_batch[n_ocb_off + k].vvpad.top = 0;
                        btc.brg_batch[n_ocb_off + k].vvpad.bottom = 0;
                        // general wei layout is gIdhwO<block_i><block_o>
                        btc.brg_batch[n_ocb_off + k].ptr.B
                                = wei_base_kh + wei_dsz * kw * wei_oc_sz;
                        k++;
                    }
                }
            }
            k_sum += k;
        }
        // For deconvolution with sum post-op and out_buffer: pre-copy
        // existing output values into out_buffer so sum reads correct data.
        if (do_postops && need_out_buffer && jcp.is_deconv && jcp.with_sum) {
            const auto iw_begin
                    = static_cast<dim_t>(btc.iwb) * jcp.iw_block + btc.sw;
            const auto real_dst = diff_src
                    + dst_dsz
                            * (btc.n * dst_d_sz + g_ic + id * dst_h_sz
                                    + ih * dst_w_sz
                                    + iw_begin * jcp.ic_without_padding);
            const dim_t ic_len = is_ic_tail ? (jcp.ic - ic) : jcp.ic_block;
            const auto ic_bytes = static_cast<size_t>(ic_len) * dst_dsz;
            const auto stride_bytes = static_cast<size_t>(jcp.stride_w)
                    * jcp.ic_without_padding * dst_dsz;
            for (int m = 0; m < ker_i_postops; m++) {
                const auto iw_pos = iw_begin + m * jcp.stride_w;
                if (iw_pos >= jcp.iw) break;
                std::memcpy(ptr_D + m * stride_bytes,
                        real_dst + m * stride_bytes, ic_bytes);
            }
        }
        const char *fake_dst = need_out_buffer && jcp.is_deconv ? btc.out_buffer
                        - dst_dsz
                                * (btc.n * dst_d_sz + g_ic + id * dst_h_sz
                                        + ih * dst_w_sz
                                        + static_cast<dim_t>(btc.iwb)
                                                * jcp.iw_block
                                                * jcp.ic_without_padding)
                                                                : nullptr;
        call_brgemm_kernel(btc, brg_idx, k_sum, ptr_C, ptr_D, bias_w, g_ic,
                do_postops, post_ops_binary_rhs_arg_vec.data(), btc.src_zp_val,
                src_zp, btc.dst_zp_vals, s8s8_comp, false,
                is_first_call_postops, fake_dst);
        if (!is_first_call_postops_state_changed) {
            const auto do_only_pass_comp = !do_postops && jcp.src_zero_point
                    && (jcp.req_brg_comp_pad || jcp.max_vpad > 0);
            const auto do_skip_accm = k_sum == 0;
            is_first_call_postops
                    = one_of(true, do_postops, do_only_pass_comp, do_skip_accm);
            is_first_call_postops_state_changed = true;
        }

        MAYBE_UNUSED(bias_w);
        MAYBE_UNUSED(ptr_C);
        MAYBE_UNUSED(post_ops_binary_rhs_arg_vec);
    };

    const auto kdhw_loop = [&]() {
        const auto do_init = btc.occ == 0 && kd_b == kd_s && kh_b == kh_s;
        const auto do_postwork = need_postwork && btc.occ == (oc_chunks - 1)
                && kd_e == kd_f && kh_e == kh_f;

        const dim_t kd_l = div_up(nstl::max<dim_t>(0, kd_e - kd_b), SD);
        const dim_t kh_l = div_up(nstl::max<dim_t>(0, kh_e - kh_b), SH);
        const dim_t kw_l = div_up(nstl::max<dim_t>(0, kw_f - kw_s), SW);
        k_l = kd_l * kh_l * kw_l;

        const auto comp_ker_offs = kd_l * kh_l > 0
                ? get_comp_offset(
                          btc.g, btc.icb, iw, kd_s, kd_f, kh_s, kh_f, 0, KW)
                : get_comp_offset(btc.g, btc.icb, iw, 0, 0, 0, 0, 0, 0);

        if (nb_oc_b > 0) {
            const auto do_postops_here = do_postwork && !is_oc_tail;
            int brg_idx;
            if (jcp.use_uker) {
                // AMX: always use full_m, masked descriptor for postops
                const auto is_masked
                        = do_postops_here && (ker_i_postops < full_m);
                brg_idx = _pd->get_brg_idx(static_cast<int>(full_m), do_init,
                        is_ic_tail, false,
                        is_masked ? static_cast<int>(ker_i_postops) : 0,
                        static_cast<int>(k_l));
            } else {
                // Non-AMX: use actual reduced M for postops call
                const int m = do_postops_here ? static_cast<int>(ker_i_postops)
                                              : static_cast<int>(full_m);
                brg_idx = _pd->get_brg_idx(m, do_init, is_ic_tail, false);
            }
            call_brgemm(brg_idx, 0, nb_oc_b, comp_ker_offs, do_postops_here);
        }

        if (is_oc_tail) {
            int brg_oc_tail_idx;
            const auto use_init_ker = (do_init && nb_oc_b == 0);
            if (jcp.use_uker) {
                const auto is_masked = do_postwork && (ker_i_postops < full_m);
                brg_oc_tail_idx = _pd->get_brg_idx(static_cast<int>(full_m),
                        use_init_ker, is_ic_tail, true,
                        is_masked ? static_cast<int>(ker_i_postops) : 0,
                        static_cast<int>(k_l));
            } else {
                const int m = do_postwork ? static_cast<int>(ker_i_postops)
                                          : static_cast<int>(full_m);
                brg_oc_tail_idx
                        = _pd->get_brg_idx(m, use_init_ker, is_ic_tail, true);
            }
            call_brgemm(
                    brg_oc_tail_idx, nb_oc_b, 1, comp_ker_offs, do_postwork);
        }
    };

    if (kd_f > kd_s && kh_f > kh_s && kw_f > kw_s) {
        // kw values covering full ow_block
        for (kd_b = kd_s; kd_b < kd_f; kd_b += KD_BLOCK) {
            kd_e = nstl::min<dim_t>(kd_f, kd_b + KD_BLOCK);
            for (kh_b = kh_s; kh_b < kh_f; kh_b += KH_BLOCK) {
                kh_e = nstl::min<dim_t>(kh_f, kh_b + KH_BLOCK);
                kdhw_loop();
            }
        }
    } else {
        // No valid kd/kh positions: k_l will be 0 (skip accumulation).
        if (jcp.use_uker) {
            // AMX uker cannot execute with batch_size=0 at runtime.
            // Use perform_outwork to handle zero-init and post-ops.
            const auto do_init = btc.occ == 0;
            const auto do_postwork
                    = need_postwork && btc.occ == (oc_chunks - 1);
            // For deconv with sum post-op and out_buffer: pre-copy
            // existing output values so sum reads correct data.
            if (do_postwork && need_out_buffer && jcp.is_deconv
                    && jcp.with_sum) {
                const auto iw_begin
                        = static_cast<dim_t>(btc.iwb) * jcp.iw_block + btc.sw;
                const auto real_dst = diff_src
                        + dst_dsz
                                * (btc.n * dst_d_sz + g_ic + id * dst_h_sz
                                        + ih * dst_w_sz
                                        + iw_begin * jcp.ic_without_padding);
                const dim_t ic_len = is_ic_tail ? (jcp.ic - ic) : jcp.ic_block;
                const auto ic_bytes = static_cast<size_t>(ic_len) * dst_dsz;
                const auto stride_bytes = static_cast<size_t>(jcp.stride_w)
                        * jcp.ic_without_padding * dst_dsz;
                for (int m = 0; m < ker_i_postops; m++) {
                    const auto iw_pos = iw_begin + m * jcp.stride_w;
                    if (iw_pos >= jcp.iw) break;
                    std::memcpy(ptr_D + m * stride_bytes,
                            real_dst + m * stride_bytes, ic_bytes);
                }
            }
            if (!need_out_buffer) {
                const dim_t iw_raw = static_cast<dim_t>(btc.iwb) * jcp.iw_block;
                perform_outwork(diff_src_base, diff_src, btc.c_buffer, bias_w,
                        btc.id, btc.ih, iw, iw_raw, g_ic, is_ic_tail, iw, iw, 0,
                        0, post_ops_binary_rhs_arg_vec.data(), btc.src_zp_val,
                        btc.src_zp_comp_ptr, btc.dst_zp_vals, btc.s8s8_comp_ptr,
                        0, do_init, do_postwork, false, btc.src_scales,
                        btc.wei_scales, btc.dst_scales);
            }
        } else {
            // Non-AMX: brgemm handles batch_size=0 (skip_accumulation)
            // correctly at runtime via var_bs.
            // Set kd_e=kd_f and kh_e=kh_f so do_postwork evaluates to true,
            // ensuring post-ops (including sum pre-copy) are applied.
            kd_b = kd_s;
            kd_e = kd_f;
            kh_b = kh_s;
            kh_e = kh_f;
            kdhw_loop();
        }
    }
}

template struct brgemm_convolution_bwd_strided_t<avx2>;
template struct brgemm_convolution_bwd_strided_t<avx2_vnni>;
template struct brgemm_convolution_bwd_strided_t<avx2_vnni_2>;
template struct brgemm_convolution_bwd_strided_t<avx512_core_amx>;
template struct brgemm_convolution_bwd_strided_t<avx512_core_amx_fp16>;
template struct brgemm_convolution_bwd_strided_t<avx512_core>;
template struct brgemm_convolution_bwd_strided_t<avx512_core_vnni>;
template struct brgemm_convolution_bwd_strided_t<avx512_core_bf16>;
template struct brgemm_convolution_bwd_strided_t<avx512_core_fp16>;
template struct brgemm_convolution_bwd_strided_t<avx10_2>;
template struct brgemm_convolution_bwd_strided_t<avx10_2_amx_2>;
template struct brgemm_convolution_bwd_strided_t<avx10_2_ace>;

} // namespace x64

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
