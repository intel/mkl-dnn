/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2022-2025 FUJITSU LIMITED
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
#include "common/dnnl_thread.hpp"
#include "common/utils.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include <type_traits>

#include "jit_uni_deconv_zp_pad_str_kernel.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {
namespace zp {

using namespace Xbyak_aarch64;

jit_uni_deconv_zp_pad_str_kernel_base_t::
        jit_uni_deconv_zp_pad_str_kernel_base_t(const jit_conv_conf_t &jcp)
    : jcp_(jcp)
    , tail_size_(jcp.is_depthwise ? jcp.ngroups % jcp.ch_block
                                  : jcp.oc_without_padding % jcp.oc_block) {}

size_t jit_uni_deconv_zp_pad_str_kernel_base_t::reserve_vmm() {
    return number_reserved_vmms_++;
}

void jit_uni_deconv_zp_pad_str_kernel_base_t::generate() {
    preamble();
    load_addresses();
    init();
    compute();
    apply_zero_point();
    store_result();
    postamble();
}

void jit_uni_deconv_zp_pad_str_kernel_base_t::compute() {

    const dim_t outer_icb_step = jcp_.kd * jcp_.kh * jcp_.kw * jcp_.ic_block
            * jcp_.oc_block * jcp_.ch_block;
    const dim_t inner_icb_step = jcp_.oc_block * jcp_.ch_block * 4;
    const bool ic_tail_exists = jcp_.ic_without_padding % jcp_.ic_block;

    for (dim_t icb = 0; icb < jcp_.nb_ic; ++icb) {
        const bool is_last_icb = icb == jcp_.nb_ic - 1;

        const int n_inner_ic_blk = jcp_.is_depthwise
                ? 1
                : (is_last_icb && ic_tail_exists
                                  ? utils::div_up(jcp_.ic_without_padding
                                                    % jcp_.ic_block,
                                            4)
                                  : (jcp_.ic_block / 4));

        const dim_t outer_wei_offset = icb * outer_icb_step;

        for (int inner_icb = 0; inner_icb < n_inner_ic_blk; inner_icb++) {
            const dim_t inner_wei_offset
                    = outer_wei_offset + inner_icb * inner_icb_step;

            compute_step(inner_wei_offset);
        }
    }
}

template <cpu_isa_t isa>
jit_uni_deconv_zp_pad_str_kernel_t<isa>::jit_uni_deconv_zp_pad_str_kernel_t(
        const jit_conv_conf_t &jcp)
    : jit_uni_deconv_zp_pad_str_kernel_base_t(jcp)
    , result_acc_(reserve_vmm())
    , vmm_tmp_(0)
    , vmm_one_bytes_(jcp.is_depthwise ? 0 : reserve_vmm())
    , vmm_one_words_(0)
    , current_vmm_(number_reserved_vmms_) {}

template <cpu_isa_t isa>
void jit_uni_deconv_zp_pad_str_kernel_t<isa>::init() {
    const ZReg vmm_tmp(0);
    const uint64_t sveLen = get_sve_length();
    const uint64_t isa_sveLen = cpu_isa_traits<isa>::vlen;

    uni_clear(result_acc_);

    assert(sveLen >= cpu_isa_traits<isa>::vlen);
    if (cpu_isa_traits<isa>::vlen == sveLen)
        ptrue(P_ALL_ONE.b);
    else if (isa_sveLen == util::SVE_128)
        ptrue(P_ALL_ONE.b, VL16);
    else if (isa_sveLen == util::SVE_256)
        ptrue(P_ALL_ONE.b, VL32);
    else
        assert(!"unreachable");

    set_preg(ktail_mask_.s, tail_size_, X_TMP_0);

    if (!jcp_.is_depthwise) // fill register byte ones
        dup(vmm_one_bytes_.b, 0x01);
}

template <cpu_isa_t isa>
uint32_t jit_uni_deconv_zp_pad_str_kernel_t<isa>::get_next_vmm_idx() {
    static constexpr int max_v_regs = cpu_isa_traits<isa>::n_vregs;

    const ZReg vmm {static_cast<unsigned int>(current_vmm_++)};

    if (current_vmm_ == max_v_regs) current_vmm_ = number_reserved_vmms_;

    return vmm.getIdx();
}

template <cpu_isa_t isa>
void jit_uni_deconv_zp_pad_str_kernel_t<isa>::compute_step(
        const dim_t icb_offset) {
    const TReg wei_vmm = TReg(get_next_vmm_idx());
    const ZReg wei_zreg = ZReg(wei_vmm.getIdx());
    const VReg wei_vreg = VReg(wei_vmm.getIdx());

    if (jcp_.is_depthwise) {
        if (is_superset(isa, sve_128)) {
            LD_MUL_VL(ld1sb, wei_zreg.s, P_ALL_ONE, reg_wei_, icb_offset, 1);
        } else {
            if (0 <= icb_offset && icb_offset < (1 << 7))
                ldr(SReg(wei_vreg.getIdx()),
                        ptr(reg_wei_, (unsigned)icb_offset));
            else
                ldr(SReg(wei_vreg.getIdx()),
                        ptr(addr_off(reg_wei_, icb_offset, X_DEFAULT_ADDR,
                                X_TMP_0)));

            zip1(wei_vreg.b16, wei_vreg.b16, wei_vreg.b16);
            sxtl(wei_vreg.h8, wei_vreg.b8);
            sxtl(wei_vreg.s4, wei_vreg.h4);
        }
    } else {
        if (is_superset(isa, sve_128)) {
            LD_MUL_VL(ld1w, ZRegS(wei_vmm.getIdx()), P_ALL_ONE, reg_wei_,
                    icb_offset, 4);
        } else {
            if (0 <= icb_offset && icb_offset < (1 << 12))
                ldr(QReg(wei_vmm.getIdx()),
                        ptr(reg_wei_, static_cast<uint32_t>(icb_offset)));
            else
                ldr(QReg(wei_vmm.getIdx()),
                        ptr(addr_off(reg_wei_, icb_offset, X_DEFAULT_ADDR,
                                X_TMP_0)));
        }
    }

    if (jcp_.is_depthwise)
        uni_add(result_acc_.s, result_acc_.s, wei_vmm.s);
    else
        sdot(result_acc_.s, vmm_one_bytes_.b, wei_vmm.b);
}

struct helper_store_t {
    static void store(jit_generator_t *gen, const ZReg &vmm,
            const XReg &reg_dst, const size_t size, const PReg &opmask) {
        gen->st1w(vmm.s, opmask, ptr(reg_dst));
    }
};

template <cpu_isa_t isa>
void jit_uni_deconv_zp_pad_str_kernel_t<isa>::store_result() {

    Label store_no_tail, end;

    if (tail_size_) {
        cmp(reg_last_oc_block_, 0);
        b(EQ, store_no_tail);
        helper_store_t::store(this, result_acc_, reg_dst_,
                tail_size_ * sizeof(int32_t), ktail_mask_);
        b(end);
    }

    L(store_no_tail);
    st1w(ZRegS(result_acc_.getIdx()), P_ALL_ONE / T_z,
            ptr(XReg(reg_dst_.getIdx())));

    L(end);
}

template <cpu_isa_t isa>
void jit_uni_deconv_zp_pad_str_kernel_t<isa>::apply_zero_point() {
    const TReg zp_src_vmm = TReg(get_next_vmm_idx());
    ld1rw(ZRegS(zp_src_vmm.getIdx()), P_ALL_ONE / T_z, ptr(reg_src_zp_));
    mul(ZRegS(result_acc_.getIdx()), P_ALL_ONE / T_m,
            ZRegS(zp_src_vmm.getIdx()));
}

#define PARAM_OFF(x) offsetof(jit_uni_deconv_zp_pad_str_call_params_t, x)

void jit_uni_deconv_zp_pad_str_kernel_base_t::load_addresses() {

    ldr(reg_src_zp_,
            ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(src_zero_point))));
    ldr(reg_wei_, ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(wei))));
    ldr(XReg(reg_dst_.getIdx()),
            ptr(abi_param1, static_cast<uint32_t>(PARAM_OFF(dst_scratchpad))));
    if (tail_size_)
        ldrb(reg_last_oc_block_,
                ptr(abi_param1,
                        static_cast<uint32_t>(PARAM_OFF(last_oc_block))));
}

#undef PARAM_OFF

template <cpu_isa_t isa>
struct helper_create_deconv_ker_t {
    static jit_uni_deconv_zp_pad_str_kernel_base_t *
    create_deconv_zp_pad_str_comp_ker(const jit_conv_conf_t &jcp) {
        using namespace Xbyak_aarch64;

        const int ch_block = jcp.is_depthwise ? jcp.ch_block : jcp.ic_block;
        switch (ch_block) {
            case 16:
                return new jit_uni_deconv_zp_pad_str_kernel_t<sve_512>(jcp);
            case 8: return new jit_uni_deconv_zp_pad_str_kernel_t<sve_256>(jcp);
            case 4: return new jit_uni_deconv_zp_pad_str_kernel_t<sve_128>(jcp);
            default: assert(!"invalid channel blocking");
        }

        return nullptr;
    }
};

template <cpu_isa_t isa>
jit_uni_deconv_zp_pad_str_kernel_base_t *create_deconv_zp_pad_str_comp_ker(
        const jit_conv_conf_t &jcp) {

    return helper_create_deconv_ker_t<isa>::create_deconv_zp_pad_str_comp_ker(
            jcp);
}

#define wht_blk_off(d, g, ...) \
    (with_groups ? (d).blk_off((g), __VA_ARGS__) : (d).blk_off(__VA_ARGS__))

static dim_t wei_off(const memory_desc_wrapper &wei_d, const bool with_groups,
        const dim_t ch_b, const dim_t oc_b, const dim_t d, const dim_t h,
        const dim_t w) {

    const auto ndims = wei_d.ndims() - (with_groups ? 1 : 0);

    switch (ndims) {
        case 5: return wht_blk_off(wei_d, ch_b, oc_b, 0, d, h, w);
        case 4: return wht_blk_off(wei_d, ch_b, oc_b, 0, h, w);
        case 3: return wht_blk_off(wei_d, ch_b, oc_b, 0, w);
        default: assert(!"Unsupported ndims!");
    }

    return 0;
}

static dim_t dst_off(const jit_conv_conf_t &jcp, const dim_t ndims,
        const dim_t g, const dim_t oc, const dim_t d, const dim_t h,
        const dim_t w) {

    const auto &G = jcp.ngroups;
    const auto &OC = jcp.oc_without_padding;
    const auto &OW = jcp.kw;
    const auto &OH = jcp.kh;

    dim_t offset = w;

    if (ndims == 5)
        offset += d * OH * OW + h * OW;
    else if (ndims == 4)
        offset += h * OW;

    if (G == 1) return offset * OC + oc;

    return (offset * OC * G) + g * OC + oc;
}

void compute_deconv_zp_pad_str_comp_ker(const jit_conv_conf_t &jcp,
        const bool with_groups, const memory_desc_wrapper &wei_d,
        const int8_t *wei, const int32_t *src_zp, int32_t *dst,
        jit_uni_deconv_zp_pad_str_kernel_base_t *ker) {

    using namespace dnnl::impl::utils;
    const auto work_amount = jcp.nb_ch * jcp.nb_oc * jcp.kw * jcp.kd * jcp.kh;
    /*
     * Heuristics for parallel computation usage - cost of threads creation
     * may exceed the computation time which leads to performance drop
     */
    static constexpr int parallelization_ratio_thr = 5;
    const int nthrs = (work_amount / jcp.nthr) > parallelization_ratio_thr
            ? jcp.nthr
            : 1;

    parallel(nthrs, [=](const int ithr, const int nthr) {
        int start {0}, end {0};
        balance211(work_amount, nthr, ithr, start, end);

        int ch_b {0}, oc_b {0}, d {0}, h {0}, w {0};
        if (jcp.loop_order == loop_ngc)
            nd_iterator_init(start, ch_b, jcp.nb_ch, oc_b, jcp.nb_oc, d, jcp.kd,
                    h, jcp.kh, w, jcp.kw);
        else if (jcp.loop_order == loop_cgn)
            nd_iterator_init(start, oc_b, jcp.nb_oc, ch_b, jcp.nb_ch, d, jcp.kd,
                    h, jcp.kh, w, jcp.kw);

        for (auto iwork = start; iwork < end; ++iwork) {
            jit_uni_deconv_zp_pad_str_call_params_t params;
            const auto oc = oc_b * jcp.oc_block;
            const auto g = ch_b * jcp.ch_block;
            params.wei = wei + wei_off(wei_d, with_groups, ch_b, oc_b, d, h, w);
            params.src_zero_point = src_zp;
            params.last_oc_block = jcp.is_depthwise ? ch_b == jcp.nb_ch - 1
                                                    : oc_b == jcp.nb_oc - 1;
            params.dst_scratchpad = dst
                    + dst_off(jcp, wei_d.ndims() - (with_groups ? 1 : 0), g, oc,
                            d, h, w);

            (*ker)(&params);

            if (jcp.loop_order == loop_ngc)
                nd_iterator_step(ch_b, jcp.nb_ch, oc_b, jcp.nb_oc, d, jcp.kd, h,
                        jcp.kh, w, jcp.kw);
            else if (jcp.loop_order == loop_cgn)
                nd_iterator_step(oc_b, jcp.nb_oc, ch_b, jcp.nb_ch, d, jcp.kd, h,
                        jcp.kh, w, jcp.kw);
            else
                assert(!"unsupported loop order");
        }
    });
}

static bool stride_exists(const jit_conv_conf_t &jcp) noexcept {
    return jcp.stride_d > 1 || jcp.stride_w > 1 || jcp.stride_h > 1;
}

static bool padding_exists(const jit_conv_conf_t &jcp) noexcept {
    const auto dd = jcp.dilate_d + 1;
    const auto dh = jcp.dilate_h + 1;
    const auto dw = jcp.dilate_w + 1;
    return jcp.kw - jcp.l_pad / dw - 1 || jcp.kw - jcp.r_pad / dw - 1
            || jcp.kh - jcp.t_pad / dh - 1 || jcp.kh - jcp.b_pad / dh - 1
            || jcp.kd - jcp.f_pad / dd - 1 || jcp.kd - jcp.back_pad / dd - 1;
}

bool should_calculate_deconv_zp_src_pad_str_comp(
        const jit_conv_conf_t &jcp) noexcept {
    return jcp.src_zero_point && (stride_exists(jcp) || padding_exists(jcp));
}

template jit_uni_deconv_zp_pad_str_kernel_base_t *
create_deconv_zp_pad_str_comp_ker<sve_512>(const jit_conv_conf_t &jcp);
template jit_uni_deconv_zp_pad_str_kernel_base_t *
create_deconv_zp_pad_str_comp_ker<sve_256>(const jit_conv_conf_t &jcp);
template jit_uni_deconv_zp_pad_str_kernel_base_t *
create_deconv_zp_pad_str_comp_ker<sve_128>(const jit_conv_conf_t &jcp);

} // namespace zp
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
