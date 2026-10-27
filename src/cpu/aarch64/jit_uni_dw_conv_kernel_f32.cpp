/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2021-2024 FUJITSU LIMITED
* Copyright 2024-2026 Arm Ltd. and affiliates
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
#include "common/memory.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_uni_dw_conv_kernel_f32.hpp"

#define GET_OFF(field) static_cast<int32_t>(offsetof(jit_conv_args_t, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;

using namespace Xbyak_aarch64;

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32_t<isa>::load_src(
        int ur_ch_blocks, int ur_w) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        // adjust the src offset so it's always within the range expected by LD_MUL_VL.
        if (this->jcp.with_sum) {
            add_imm(reg_tmp_addr, reg_output,
                    (ch * ocb_stride) * jcp.typesize_out, reg_tmp_imm);
        }
        int b_off = ch * ch_blk;
        if (this->jcp.with_bias) {
            LD_MUL_VL(ld1w, ZRegS(1), P_ALL_ONE, reg_bias,
                    b_off * sizeof(float), sizeof(float));
        }
        for (int ow = 0; ow < ur_w; ow++) {
            ZRegS zregs_acc = ZRegS(fwd_acc_idx(ch, ow, ur_w));

            if (this->jcp.with_bias && zregs_acc.getIdx() != 1) {
                mov(zregs_acc, P_ALL_ONE, ZRegS(1));
            } else
                fmov(zregs_acc); // zero clear

            int o_off = ow * ow_stride;
            if (this->jcp.with_sum) {
                if (jcp.dst_dt == data_type::f32) {
                    LD_MUL_VL(ld1w, ZRegS(0), P_ALL_ONE, reg_tmp_addr,
                            o_off * jcp.typesize_out, jcp.typesize_out);
                    fadd(zregs_acc, zregs_acc, ZRegS(0));
                } else if (jcp.dst_dt == data_type::bf16) {
                    LD_MUL_VL(ld1h, ZRegS(0), P_ALL_ONE, reg_tmp_addr,
                            o_off * jcp.typesize_out, jcp.typesize_out);
                    // Convert BF16 input to FP32
                    lsl(ZRegS(0), ZRegS(0), 16);
                    fadd(zregs_acc, zregs_acc, ZRegS(0));
                } else {
                    assert(!"Unsupported: data type");
                }
            }
        }
    }
}

template <>
void jit_uni_dw_conv_fwd_kernel_f32_t<asimd>::load_src(
        int ur_ch_blocks, int ur_w) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;
    const bool use_upper_acc
            = jcp.src_dt == data_type::f16 && jcp.acc_dt == data_type::f32;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        if (jcp.acc_dt == data_type::f16) {
            // Accumulate products before bias or the previous destination:
            // small f16 products can round away against a large value.
            for (int ow = 0; ow < ur_w; ow++)
                uni_clear(VReg(fwd_acc_idx(ch, ow, ur_w)));
            continue;
        }
        if (this->jcp.with_sum) {
            add_imm(reg_tmp_addr, reg_output,
                    (ch * ocb_stride) * jcp.typesize_out, reg_tmp_imm);
        }
        int b_off = ch * ch_blk;
        if (this->jcp.with_bias) {
            ldr(QReg(1),
                    ptr(addr_off(reg_bias, b_off * sizeof(float),
                            X_DEFAULT_ADDR, reg_tmp_imm)));
            if (jcp.src_dt == data_type::f16)
                ldr(QReg(2),
                        ptr(addr_off(reg_bias, (b_off + 4) * sizeof(float),
                                X_DEFAULT_ADDR, reg_tmp_imm)));
        }
        for (int ow = 0; ow < ur_w; ow++) {
            const int acc_idx = fwd_acc_idx(ch, ow, ur_w);
            const int upper_acc_idx
                    = use_upper_acc ? fwd_acc_idx(ch, ow, ur_w, true) : 0;

            if (this->jcp.with_bias) {
                mov(VReg16B(acc_idx), VReg16B(1));
                if (use_upper_acc) mov(VReg16B(upper_acc_idx), VReg16B(2));
            } else {
                uni_clear(VReg(acc_idx));
                if (use_upper_acc) uni_clear(VReg(upper_acc_idx));
            }

            int o_off = ow * ow_stride;
            if (this->jcp.with_sum) {
                if (jcp.dst_dt == data_type::f32) {
                    ldr(QReg(0),
                            ptr(addr_off(reg_tmp_addr, o_off * jcp.typesize_out,
                                    X_DEFAULT_ADDR, reg_tmp_imm)));
                    fadd(VReg4S(acc_idx), VReg4S(acc_idx), VReg4S(0));
                } else if (jcp.dst_dt == data_type::f16) {
                    ldr(QReg(0),
                            ptr(addr_off(reg_tmp_addr, o_off * jcp.typesize_out,
                                    X_DEFAULT_ADDR, reg_tmp_imm)));
                    fcvtl(VReg4S(3), VReg4H(0));
                    fcvtl2(VReg4S(0), VReg8H(0));
                    fadd(VReg4S(acc_idx), VReg4S(acc_idx), VReg4S(3));
                    fadd(VReg4S(upper_acc_idx), VReg4S(upper_acc_idx),
                            VReg4S(0));
                } else {
                    assert(!"Unsupported: data type");
                }
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32_t<isa>::apply_f16_bias_and_sum(
        int ur_ch_blocks, int ur_w) {
    if (isa != asimd || jcp.acc_dt != data_type::f16
            || (!jcp.with_bias && !jcp.with_sum))
        return;

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        if (jcp.with_bias) {
            const int b_off = ch * ch_blk;
            ldr(QReg(1),
                    ptr(addr_off(reg_bias, b_off * sizeof(float),
                            X_DEFAULT_ADDR, reg_tmp_imm)));
            ldr(QReg(2),
                    ptr(addr_off(reg_bias, (b_off + 4) * sizeof(float),
                            X_DEFAULT_ADDR, reg_tmp_imm)));
            fcvtn(VReg4H(1), VReg4S(1));
            fcvtn2(VReg8H(1), VReg4S(2));
        }
        if (jcp.with_sum)
            add_imm(reg_tmp_addr, reg_output,
                    (ch * ocb_stride) * jcp.typesize_out, reg_tmp_imm);

        for (int ow = 0; ow < ur_w; ow++) {
            const int acc_idx = fwd_acc_idx(ch, ow, ur_w);
            if (jcp.with_bias)
                fadd(VReg8H(acc_idx), VReg8H(acc_idx), VReg8H(1));
            if (jcp.with_sum) {
                ldr(QReg(0),
                        ptr(addr_off(reg_tmp_addr,
                                ow * ow_stride * jcp.typesize_out,
                                X_DEFAULT_ADDR, reg_tmp_imm)));
                fadd(VReg8H(acc_idx), VReg8H(acc_idx), VReg8H(0));
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32_t<isa>::apply_filter_unrolled(
        int ur_ch_blocks, int ur_w, int pad_l, int pad_r) {

    int ch_blk = jcp.ch_block;
    int dilate_h = jcp.dilate_h + 1;
    int dilate_w = jcp.dilate_w + 1;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto iw_stride = src_layout_nxc ? jcp.ngroups : ch_blk;
    const auto ih_stride = jcp.iw * iw_stride;
    const auto icb_stride = src_layout_nxc
            ? ch_blk
            : (jcp.is_fused_conv ? 1 : jcp.ih) * jcp.iw * ch_blk;
    Label iter_exit_label;

    cmp(reg_kh, 0);
    b(EQ, iter_exit_label);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    {
        if (jcp.is_fused_conv) {
            ldr(aux_reg_input, ptr(aux_reg_input_buffer_ptr));
            add(aux_reg_input, aux_reg_input, reg_iw_offset);
        }

        for (int ch = 0; ch < ur_ch_blocks; ch++) {
            // adjust the src offset so it's always within the range expected by LD_MUL_VL.
            add_imm(reg_tmp_addr, aux_reg_input,
                    (ch * icb_stride) * jcp.typesize_in, reg_tmp_imm);
            add_imm(reg_tmp2_addr, aux_reg_kernel,
                    (ch * jcp.kh * jcp.kw * ch_blk) * jcp.typesize_in,
                    reg_tmp_imm);
            for (int kw = 0; kw < jcp.kw; kw++) {
                int ker_off = kw * ch_blk;
                if (isa == asimd) {
                    if (jcp.wei_dt == data_type::f32) {
                        ldr(QReg(0),
                                ptr(addr_off(reg_tmp2_addr,
                                        ker_off * jcp.typesize_in,
                                        X_DEFAULT_ADDR, reg_tmp_imm)));
                    } else if (jcp.wei_dt == data_type::f16) {
                        ldr(QReg(0),
                                ptr(addr_off(reg_tmp2_addr,
                                        ker_off * jcp.typesize_in,
                                        X_DEFAULT_ADDR, reg_tmp_imm)));
                    } else {
                        assert(!"Unsupported: data type");
                    }
                } else {
                    ZReg zregs_ker = get_ker_reg(0);
                    if (jcp.wei_dt == data_type::f32) {
                        LD_MUL_VL(ld1w, zregs_ker.s, P_ALL_ONE, reg_tmp2_addr,
                                ker_off * jcp.typesize_in, jcp.typesize_in);
                    } else if (jcp.wei_dt == data_type::bf16) {
                        LD_MUL_VL(ld1h, zregs_ker.s, P_ALL_ONE, reg_tmp2_addr,
                                ker_off * jcp.typesize_in, jcp.typesize_in);
                    } else {
                        assert(!"Unsupported: data type");
                    }
                }

                int ow_start = get_ow_start(kw, pad_l);
                int ow_end = get_ow_end(ur_w, kw, pad_r);
                for (int ow = ow_start; ow < ow_end; ow++) {
                    int inp_off = (ow * stride_w - pad_l) * iw_stride
                            + kw * dilate_w * iw_stride;
                    if (isa == asimd) {
                        const int acc_idx = fwd_acc_idx(ch, ow, ur_w);
                        if (jcp.src_dt == data_type::f32) {
                            ldr(QReg(1),
                                    ptr(addr_off(reg_tmp_addr,
                                            inp_off * jcp.typesize_in,
                                            X_DEFAULT_ADDR, reg_tmp_imm)));
                            fmla(VReg4S(acc_idx), VReg4S(1), VReg4S(0));
                        } else if (jcp.src_dt == data_type::f16) {
                            ldr(QReg(1),
                                    ptr(addr_off(reg_tmp_addr,
                                            inp_off * jcp.typesize_in,
                                            X_DEFAULT_ADDR, reg_tmp_imm)));
                            if (jcp.acc_dt == data_type::f16) {
                                fmla(VReg8H(acc_idx), VReg8H(1), VReg8H(0));
                            } else {
                                fmlal(VReg4S(acc_idx), VReg4H(1), VReg4H(0));
                                const int upper_acc_idx
                                        = fwd_acc_idx(ch, ow, ur_w, true);
                                fmlal2(VReg4S(upper_acc_idx), VReg4H(1),
                                        VReg4H(0));
                            }
                        } else {
                            assert(!"Unsupported: data type");
                        }
                    } else {
                        ZReg zregs_ker = get_ker_reg(0);
                        ZReg zregs_src = get_src_reg(1);
                        if (jcp.src_dt == data_type::f32) {
                            LD_MUL_VL(ld1w, zregs_src.s, P_ALL_ONE,
                                    reg_tmp_addr, inp_off * jcp.typesize_in,
                                    jcp.typesize_in);
                            ZRegS zregs_acc = ZRegS(fwd_acc_idx(ch, ow, ur_w));
                            fmla(zregs_acc, P_ALL_ONE, zregs_src.s,
                                    zregs_ker.s);
                        } else if (jcp.src_dt == data_type::bf16) {
                            LD_MUL_VL(ld1h, zregs_src.s, P_ALL_ONE,
                                    reg_tmp_addr, inp_off * jcp.typesize_in,
                                    jcp.typesize_in);
                            ZRegS zregs_acc = ZRegS(fwd_acc_idx(ch, ow, ur_w));
                            bfmlalb(zregs_acc, zregs_src.h, zregs_ker.h);
                        } else {
                            assert(!"Unsupported: data type");
                        }
                    }
                }
            }
        }

        add_imm(aux_reg_kernel, aux_reg_kernel,
                jcp.kw * ch_blk * jcp.typesize_in, reg_tmp_imm);
        if (jcp.is_fused_conv) {
            // Move to next row pointer in the buffer
            add_imm(aux_reg_input_buffer_ptr, aux_reg_input_buffer_ptr,
                    sizeof(void *), reg_tmp_imm);
        } else {
            add_imm(aux_reg_input, aux_reg_input,
                    ih_stride * dilate_h * jcp.typesize_in, reg_tmp_imm);
        }

        sub(iter_kh, iter_kh, 1);
        cmp(iter_kh, 0);
        b(GT, kh_label);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32_t<isa>::apply_activation(
        int ur_ch_blocks, int ur_w) {
    if (this->jcp.with_eltwise) {
        eltwise_injector_->load_table_addr();
        if (isa == asimd && jcp.src_dt == data_type::f16) {
            injector_utils::vmm_index_set_t acc_idxs;
            for (int ch = 0; ch < ur_ch_blocks; ch++) {
                for (int ow = 0; ow < ur_w; ow++) {
                    acc_idxs.emplace(fwd_acc_idx(ch, ow, ur_w));
                    acc_idxs.emplace(fwd_acc_idx(ch, ow, ur_w, true));
                }
            }
            eltwise_injector_->compute_vector_range(acc_idxs);
        } else {
            const int first_acc_idx = fwd_acc_idx(0, 0, ur_w);
            const int last_acc_idx
                    = fwd_acc_idx(ur_ch_blocks - 1, ur_w - 1, ur_w);
            eltwise_injector_->compute_vector_range(
                    first_acc_idx, last_acc_idx + 1);
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32_t<isa>::store_dst(
        int ur_ch_blocks, int ur_w) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        // adjust the output offset so it's always within the range expected by ST_MUL_VL.
        add_imm(reg_tmp_addr, reg_output, (ch * ocb_stride) * jcp.typesize_out,
                reg_tmp_imm);
        for (int ow = 0; ow < ur_w; ow++) {
            const int o_off = ow * ow_stride;
            if (jcp.dst_dt == data_type::f32) {
                ZRegS zreg_dst = ZRegS(fwd_acc_idx(ch, ow, ur_w));
                ST_MUL_VL(st1w, zreg_dst, P_ALL_ONE, reg_tmp_addr,
                        o_off * jcp.typesize_out, jcp.typesize_out);
            } else if (jcp.dst_dt == data_type::bf16) {
                ZReg zreg_dst = ZReg(fwd_acc_idx(ch, ow, ur_w));
                // Convert fp32 to bf16
                bfcvt(zreg_dst.h, P_ALL_ONE, zreg_dst.s);
                // Store the bf16 value doing the downcast
                ST_MUL_VL(st1h, zreg_dst.s, P_ALL_ONE, reg_tmp_addr,
                        o_off * jcp.typesize_out, jcp.typesize_out);
            } else {
                assert(!"Unsupported: data type");
            }
        }
    }
}

template <>
void jit_uni_dw_conv_fwd_kernel_f32_t<asimd>::store_dst(
        int ur_ch_blocks, int ur_w) {

    const auto dst_layout_nxc = is_dst_layout_nxc();
    const auto ch_blk = jcp.ch_block;
    const auto ocb_stride = dst_layout_nxc ? ch_blk : jcp.oh * jcp.ow * ch_blk;
    const auto ow_stride = dst_layout_nxc ? jcp.ngroups : ch_blk;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        add_imm(reg_tmp_addr, reg_output, (ch * ocb_stride) * jcp.typesize_out,
                reg_tmp_imm);
        for (int ow = 0; ow < ur_w; ow++) {
            const int o_off = ow * ow_stride;
            const int acc_idx = fwd_acc_idx(ch, ow, ur_w);
            if (jcp.dst_dt == data_type::f32) {
                str(QReg(acc_idx),
                        ptr(addr_off(reg_tmp_addr, o_off * jcp.typesize_out,
                                X_DEFAULT_ADDR, reg_tmp_imm)));
            } else if (jcp.dst_dt == data_type::f16) {
                if (jcp.acc_dt == data_type::f32)
                    fcvtn(VReg4H(acc_idx), VReg4S(acc_idx));
                if (jcp.acc_dt == data_type::f32) {
                    const int upper_acc_idx = fwd_acc_idx(ch, ow, ur_w, true);
                    fcvtn2(VReg8H(acc_idx), VReg4S(upper_acc_idx));
                }
                str(QReg(acc_idx),
                        ptr(addr_off(reg_tmp_addr, o_off * jcp.typesize_out,
                                X_DEFAULT_ADDR, reg_tmp_imm)));
            } else {
                assert(!"Unsupported: data type");
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32_t<isa>::compute_loop(
        int ur_w, int ur_ch_blocks, int pad_l, int pad_r) {

    const bool ch_loop = ur_ch_blocks > jcp.nb_ch_blocking;
    // ch_loop currently happen only when data layout is nxc. The strides are
    // calculated for this layout only.
    const size_t wei_ch_stride = (size_t)jcp.nb_ch_blocking * jcp.kh * jcp.kw
            * jcp.ch_block * jcp.typesize_in;
    const size_t inp_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_in;
    const size_t out_ch_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * jcp.typesize_out;
    const size_t bias_stride
            = (size_t)jcp.nb_ch_blocking * jcp.ch_block * sizeof(float);

    auto compute = [&](int ur_ch_blocks) {
        if (jcp.is_fused_conv) {
            mov(aux_reg_input_buffer_ptr, reg_input_buffer_ptr);
        } else {
            mov(aux_reg_input, reg_input);
        }

        mov(aux_reg_kernel, reg_kernel);
        load_src(ur_ch_blocks, ur_w);
        apply_filter_unrolled(ur_ch_blocks, ur_w, pad_l, pad_r);
        apply_f16_bias_and_sum(ur_ch_blocks, ur_w);
        apply_activation(ur_ch_blocks, ur_w);
        store_dst(ur_ch_blocks, ur_w);
    };

    if (ch_loop) {
        Label ch_loop_label, ch_tail_label, skip_ch_tail_label;
        const int ch_tail = jcp.nb_ch % jcp.nb_ch_blocking;

        mov(aux_reg_ch_blocks, reg_ch_blocks);
        mov(reg_kernel_stack, reg_kernel);
        mov(reg_input_stack, reg_input);
        mov(reg_output_stack, reg_output);
        if (jcp.with_bias) mov(reg_bias_stack, reg_bias);

        if (ch_tail) {
            cmp(aux_reg_ch_blocks, jcp.nb_ch_blocking);
            b(LT, ch_tail_label);
        }

        L(ch_loop_label);
        {
            compute(jcp.nb_ch_blocking);
            add_imm(reg_kernel, reg_kernel, wei_ch_stride, reg_tmp_imm);
            add_imm(reg_input, reg_input, inp_ch_stride, reg_tmp_imm);
            add_imm(reg_output, reg_output, out_ch_stride, reg_tmp_imm);
            if (jcp.with_bias)
                add_imm(reg_bias, reg_bias, bias_stride, reg_tmp_imm);
            sub_imm(aux_reg_ch_blocks, aux_reg_ch_blocks, jcp.nb_ch_blocking,
                    reg_tmp_imm);
            cmp(aux_reg_ch_blocks, jcp.nb_ch_blocking);
            b(GE, ch_loop_label);
        }

        if (ch_tail) {
            L(ch_tail_label);
            cmp(aux_reg_ch_blocks, 0);
            b(LE, skip_ch_tail_label);
            compute(ch_tail);
            L(skip_ch_tail_label);
        }

        if (jcp.with_bias) mov(reg_bias, reg_bias_stack);
        mov(reg_output, reg_output_stack);
        mov(reg_input, reg_input_stack);
        mov(reg_kernel, reg_kernel_stack);

    } else {
        compute(ur_ch_blocks);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32_t<isa>::ow_loop(int ur_ch_blocks) {

    int iw = jcp.iw;
    int ow = jcp.ow;
    int kw = jcp.kw;
    int l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    int stride_w = jcp.stride_w;

    const auto src_layout_nxc = is_src_layout_nxc();
    const auto dat_c_stride = src_layout_nxc ? jcp.ngroups : jcp.ch_block;
    size_t inp_shift = (size_t)jcp.typesize_in * ur_w * stride_w * dat_c_stride;
    size_t out_shift = (size_t)jcp.typesize_out * ur_w * dat_c_stride;

    int inp_shift_pad
            = jcp.typesize_in * (ur_w * stride_w - l_pad) * dat_c_stride;

    int r_pad = nstl::max(0, jcp.r_pad);
    int n_oi = ow / ur_w;
    int r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    assert(jcp.nb_ow <= 1);

    if (r_pad1 > 0) n_oi--;
    mov(reg_oi, 0);
    if (ow == ur_w) {
        compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad);
    } else {
        if (n_oi == 0) {
            compute_loop(ur_w, ur_ch_blocks, l_pad, r_pad1);
            add_imm(reg_input, reg_input, inp_shift_pad, reg_tmp_imm);
            add_imm(reg_output, reg_output, out_shift, reg_tmp_imm);
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        } else {
            if (l_pad > 0) {
                compute_loop(ur_w, ur_ch_blocks, l_pad, 0);
                add_imm(reg_input, reg_input, inp_shift_pad, reg_tmp_imm);
                add_imm(reg_output, reg_output, out_shift, reg_tmp_imm);
                add(reg_oi, reg_oi, 1);
            }
            if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                Label ow_loop_label;
                L(ow_loop_label);
                {
                    compute_loop(ur_w, ur_ch_blocks, 0, 0);
                    add_imm(reg_input, reg_input, inp_shift, reg_tmp_imm);
                    add_imm(reg_output, reg_output, out_shift, reg_tmp_imm);

                    add(reg_oi, reg_oi, 1);
                    cmp(reg_oi, n_oi);
                    b(LT, ow_loop_label);
                }
            }
            if (r_pad1 > 0) {
                compute_loop(ur_w, ur_ch_blocks, 0, r_pad1);
                add_imm(reg_input, reg_input, inp_shift, reg_tmp_imm);
                add_imm(reg_output, reg_output, out_shift, reg_tmp_imm);
            }
            if (ur_w_tail != 0) {
                compute_loop(ur_w_tail, ur_ch_blocks, 0, r_pad);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_fwd_kernel_f32_t<isa>::generate() {
    const int simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);

    // Keep the convolution state that is live across eltwise injection out of
    // the scratch pool. Filter-loop temporaries are dead at that point and are
    // intentionally left available to the injector.
    auto live_gprs = claim_regs<XReg>({reg_input, reg_kernel, reg_ch_blocks,
            reg_output, reg_bias, reg_kh, reg_oi, aux_reg_ch_blocks,
            reg_input_buffer_ptr, reg_bias_stack});

    // The largest accumulator set is used by the full-width, full-channel
    // block. Tail paths use a subset of this range.
    const int max_accumulators = jcp.ur_w * jcp.nb_ch_blocking
            * ((isa == asimd && jcp.src_dt == data_type::f16) ? 2 : 1);
    auto live_accumulators = claim_reg_range<TReg>(4, max_accumulators);

    auto eltwise_table = alloc_reg<XReg>();
    auto eltwise_mask = alloc_reg<PReg>();
    if (jcp.with_eltwise) {
        eltwise_injector_ = utils::make_unique<
                jit_uni_eltwise_injector_t<to_vla_sve(isa)>>(this, reg_pool(),
                jcp.eltwise, eltwise_table.get(), eltwise_mask.get());
    }

    this->preamble();
    //TO DO : renaming predicate register (P_ALL_ONE)
    if (isa != asimd && simd_w_ != cpu_sveLen / sizeof(float))
        set_preg(P_ALL_ONE.s, simd_w_, X_TMP_0);
    if (jcp.is_fused_conv) {
        ldr(reg_input_buffer_ptr, ptr(abi_param1, GET_OFF(src)));
        mov(reg_iw_offset, 0);
    } else {
        ldr(reg_input, ptr(abi_param1, GET_OFF(src)));
    }
    ldr(reg_output, ptr(abi_param1, GET_OFF(dst)));
    ldr(reg_kernel, ptr(abi_param1, GET_OFF(filt)));
    if (jcp.with_bias) { ldr(reg_bias, ptr(abi_param1, GET_OFF(bias))); }
    ldr(reg_kh, ptr(abi_param1, GET_OFF(kh_padding)));
    ldr(reg_ch_blocks, ptr(abi_param1, GET_OFF(ch_blocks)));

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    if (is_src_layout_nxc()) {
        ow_loop(jcp.nb_ch);
    } else {
        cmp(reg_ch_blocks, jcp.nb_ch_blocking);
        b(NE, ch_blocks_tail ? ch_blocks_tail_label : exit_label);

        ow_loop(jcp.nb_ch_blocking); // channel main loop

        if (ch_blocks_tail) {
            L(ch_blocks_tail_label);

            cmp(reg_ch_blocks, ch_blocks_tail);
            b(NE, exit_label);

            ow_loop(ch_blocks_tail); // channel tail loop
        }

        L(exit_label);
    }

    this->postamble();
    if (jcp.with_eltwise) { eltwise_injector_->prepare_table(); }
    eltwise_injector_.reset();
}

template struct jit_uni_dw_conv_fwd_kernel_f32_t<sve_512>;
template struct jit_uni_dw_conv_fwd_kernel_f32_t<sve_256>;
template struct jit_uni_dw_conv_fwd_kernel_f32_t<sve_128>;
template struct jit_uni_dw_conv_fwd_kernel_f32_t<asimd>;

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32_t<isa>::load_ddst(
        int ur_ch_blocks, int ur_str_w) {
    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            ZRegS zregs_acc = get_acc_reg_s(ch * ur_str_w + w);
            fmov(zregs_acc); // zero clear
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32_t<isa>::apply_filter(
        int ur_ch_blocks, int ur_str_w) {
    int kw = jcp.kw;
    int kh = jcp.kh;
    int ow = jcp.ow;
    int oh = jcp.oh;

    int ch_blk = jcp.ch_block;
    int stride_h = jcp.stride_h;
    int stride_w = jcp.stride_w;

    Label iter_exit_label;

    cmp(reg_kh, 0);
    b(EQ, iter_exit_label);

    cmp(reg_kw, 0);
    b(EQ, iter_exit_label);

    mov(iter_kh, reg_kh);
    Label kh_label;
    L(kh_label);
    { // KH loop
        mov(aux1_reg_ddst, aux_reg_ddst);
        mov(aux1_reg_kernel, aux_reg_kernel);

        mov(iter_kw, reg_kw);
        Label kw_label;
        L(kw_label);
        { // KW loop
            for (int ch = 0; ch < ur_ch_blocks;
                    ch++) { // unrolloing channel blocks
                int ker_off = ch * kh * kw * ch_blk;
                ZRegS zregs_ker = get_ker_reg_s(0);

                add_imm(reg_tmp_addr, aux1_reg_kernel, ker_off * sizeof(float),
                        reg_tmp_imm);
                ld1w(zregs_ker, P_ALL_ONE / T_z, ptr(reg_tmp_addr));

                for (int w = 0; w < ur_str_w; w++) {
                    int ddst_off = (ch * oh * ow + w) * ch_blk;

                    ZRegS zregs_src = get_src_reg_s(0);
                    add_imm(reg_tmp_addr, aux1_reg_ddst,
                            ddst_off * sizeof(float), reg_tmp_imm);
                    ld1w(zregs_src, P_ALL_ONE / T_z, ptr(reg_tmp_addr));

                    ZRegS zregs_acc = get_acc_reg_s(ch * ur_str_w + w);
                    fmla(zregs_acc, P_ALL_ONE, zregs_src, zregs_ker);
                }
            }

            add_imm(aux1_reg_kernel, aux1_reg_kernel,
                    ch_blk * stride_w * sizeof(float), reg_tmp_imm);
            sub_imm(aux1_reg_ddst, aux1_reg_ddst,
                    ch_blk * (jcp.dilate_w + 1) * sizeof(float), reg_tmp_imm);
            sub_imm(iter_kw, iter_kw, stride_w, reg_tmp_imm);
            cmp(iter_kw, 0);
            b(GT, kw_label);
        }

        add_imm(aux_reg_kernel, aux_reg_kernel,
                kw * ch_blk * stride_h * sizeof(float), reg_tmp_imm);
        sub_imm(aux_reg_ddst, aux_reg_ddst,
                ow * ch_blk * (jcp.dilate_h + 1) * sizeof(float), reg_tmp_imm);
        sub_imm(iter_kh, iter_kh, stride_h, reg_tmp_imm);
        cmp(iter_kh, 0);
        b(GT, kh_label);
    }

    L(iter_exit_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32_t<isa>::store_dsrc(
        int ur_ch_blocks, int ur_str_w) {
    int ch_blk = jcp.ch_block;
    int iw = jcp.iw;
    int ih = jcp.ih;
    int stride_w = jcp.stride_w;

    for (int ch = 0; ch < ur_ch_blocks; ch++) {
        for (int w = 0; w < ur_str_w; w++) {
            int dsrc_off = (ch * ih * iw + w * stride_w) * ch_blk;
            ZRegS zregs_acc = get_acc_reg_s(ch * ur_str_w + w);

            add_imm(reg_tmp_addr, reg_dsrc, dsrc_off * sizeof(float),
                    reg_tmp_imm);
            st1w(zregs_acc, P_ALL_ONE / T_z, ptr(reg_tmp_addr));
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_data_kernel_f32_t<isa>::loop_body(
        int ur_ch_blocks) {
    Label unrolled_w_label;
    Label tail_w_label;
    Label exit_label;

    L(unrolled_w_label);
    {
        int ur_w = jcp.ur_w;

        cmp(reg_ur_str_w, ur_w);
        b(LT, tail_w_label);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w); // zero clear
        apply_filter(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add_imm(reg_dsrc, reg_dsrc,
                sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w,
                reg_tmp_imm);
        add_imm(reg_ddst, reg_ddst, sizeof(float) * ur_w * jcp.ch_block,
                reg_tmp_imm);

        sub_imm(reg_ur_str_w, reg_ur_str_w, ur_w, reg_tmp_imm);
        b(unrolled_w_label);
    }

    L(tail_w_label);
    {
        int ur_w = 1;

        cmp(reg_ur_str_w, ur_w);
        b(LT, exit_label);

        mov(aux_reg_ddst, reg_ddst);
        mov(aux_reg_kernel, reg_kernel);

        load_ddst(ur_ch_blocks, ur_w);
        apply_filter(ur_ch_blocks, ur_w);
        store_dsrc(ur_ch_blocks, ur_w);

        add_imm(reg_dsrc, reg_dsrc,
                sizeof(float) * ur_w * jcp.ch_block * jcp.stride_w,
                reg_tmp_imm);
        add_imm(reg_ddst, reg_ddst, sizeof(float) * ur_w * jcp.ch_block,
                reg_tmp_imm);

        sub_imm(reg_ur_str_w, reg_ur_str_w, ur_w, reg_tmp_imm);
        b(tail_w_label);
    }

    L(exit_label);
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_data_kernel_f32_t<isa>::generate() {
    preamble();

    ldr(reg_dsrc, ptr(abi_param1, GET_OFF(src)));
    ldr(reg_ddst, ptr(abi_param1, GET_OFF(dst)));
    ldr(reg_kernel, ptr(abi_param1, GET_OFF(filt)));
    ldr(reg_kh, ptr(abi_param1, GET_OFF(kh_padding)));
    ldr(reg_kw, ptr(abi_param1, GET_OFF(kw_padding)));
    ldr(reg_ch_blocks, ptr(abi_param1, GET_OFF(ch_blocks)));
    ldr(reg_ur_str_w, ptr(abi_param1, GET_OFF(ur_str_w)));

    Label ch_blocks_tail_label;
    Label exit_label;

    int ch_blocks_tail = jcp.nb_ch % jcp.nb_ch_blocking;

    cmp(reg_ch_blocks, jcp.nb_ch_blocking);
    b(NE, ch_blocks_tail ? ch_blocks_tail_label : exit_label);

    loop_body(jcp.nb_ch_blocking); // channel main loop

    if (ch_blocks_tail) {
        L(ch_blocks_tail_label);

        cmp(reg_ch_blocks, ch_blocks_tail);
        b(NE, exit_label);

        loop_body(ch_blocks_tail); // channel tail loop
    }

    L(exit_label);

    this->postamble();
}

template struct jit_uni_dw_conv_bwd_data_kernel_f32_t<sve_512>;
template struct jit_uni_dw_conv_bwd_data_kernel_f32_t<sve_256>;

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::zero_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        ZRegS zregs_acc = get_acc_reg_s(i);
        fmov(zregs_acc); // zero clear
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::load_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        int off_filter = i * simd_w;
        add_imm(reg_tmp_addr, reg_tmp_filter, off_filter * sizeof(float),
                reg_tmp_imm);
        if (simd_w == 16) {
            ZReg zreg_acc = get_acc_reg(i);
            ldr(zreg_acc, ptr(reg_tmp_addr));
        } else if (simd_w == 8) {
            ZRegS zregs_acc = get_acc_reg_s(i);
            ld1w(zregs_acc, P_ALL_ONE, ptr(reg_tmp_addr));
        } else {
            assert(!"Unsupport: simd_w != 16, 8");
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::zero_bias() {
    ZRegS zregs_bias = get_bias_reg_s(0);
    fmov(zregs_bias); // zero clear
}
template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::load_bias() {
    if (simd_w == 16) {
        ZReg zreg_bias = get_bias_reg(0);
        ldr(zreg_bias, ptr(reg_bias_baddr));
    } else if (simd_w == 8) {
        ZRegS zregs_bias = get_bias_reg_s(0);
        ld1w(zregs_bias, P_ALL_ONE, ptr(reg_bias_baddr));
    } else {
        assert(!"Unsupport: simd_w != 16, 8");
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::compute_ow_step_unroll(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const int iw_block = ow_block * jcp.stride_w;
    const int right_border = jcp.iw - iw_block;
    const int r_pad = jcp.r_pad;

    const int cascade_input = nstl::min(jcp.stride_w, jcp.kw);

    /* preamble count for number of cascaded LOAD + FMA operation */
    const int input_overlap = nstl::max(jcp.kw - l_pad, 0);
    const bool is_last_block = (unroll_w + ow_block == jcp.ow);

    /* LOAD initial input registers, then cascade LOADs and FMAs*/
    for (int i_ur = 0; i_ur < unroll_w; ++i_ur) {
        int off_output = i_ur * simd_w;
        ZReg zreg_output = get_output_reg(0);
        ZRegS zregs_output = get_output_reg_s(0);

        add_imm(reg_tmp_addr, reg_tmp_output, off_output * sizeof(float),
                reg_tmp_imm);
        if (simd_w == 16) {
            ldr(zreg_output, ptr(reg_tmp_addr));
        } else if (simd_w == 8) {
            ld1w(zregs_output, P_ALL_ONE, ptr(reg_tmp_addr));
        } else {
            assert(!"Unsupport: simd_w != 16, 8");
        }

        if (i_ur == 0) {
            for (int c = 0; c < input_overlap; ++c) {
                int off_input = (c - pad_offset) * simd_w;
                if (off_input < 0 && unroll_w == jcp.ow) continue;

                const bool over_steps_bdry = true && is_last_block
                        && (c - pad_offset + r_pad > right_border);
                if (over_steps_bdry) continue;

                add_imm(reg_tmp_addr, reg_tmp_input, off_input * sizeof(float),
                        reg_tmp_imm);
                if (simd_w == 16) {
                    ZReg zreg_input = get_input_reg(c % jcp.kw);
                    ldr(zreg_input, ptr(reg_tmp_addr));
                } else if (simd_w == 8) {
                    ZRegS zregs_input = get_input_reg_s(c % jcp.kw);
                    ld1w(zregs_input, P_ALL_ONE, ptr(reg_tmp_addr));
                } else {
                    assert(!"Unsupport: simd_w != 16, 8");
                }
            }
        } else {
            for (int c = 0; c < cascade_input; ++c) {
                int overlap = (i_ur - 1) * jcp.stride_w + input_overlap;
                int off_input = (overlap + c - pad_offset) * simd_w;
                if (off_input < 0 || overlap + c + l_pad > right_border)
                    continue;

                const bool over_steps_bdry = true && is_last_block
                        && (overlap + c - pad_offset + r_pad > right_border);
                if (over_steps_bdry) continue;

                add_imm(reg_tmp_addr, reg_tmp_input, off_input * sizeof(float),
                        reg_tmp_imm);
                if (simd_w == 16) {
                    ZReg zreg_input = get_input_reg((overlap + c) % jcp.kw);
                    ldr(zreg_input, ptr(reg_tmp_addr));
                } else if (simd_w == 8) {
                    ZRegS zregs_input = get_input_reg_s((overlap + c) % jcp.kw);
                    ld1w(zregs_input, P_ALL_ONE, ptr(reg_tmp_addr));
                } else {
                    assert(!"Unsupport: simd_w != 16, 8");
                }
            }
        }

        for (int i_kw = 0; i_kw < jcp.kw; ++i_kw) {
            int io_overlap = i_kw + (i_ur * jcp.stride_w);

            /* Don't apply FMAs that fall into the padded region */
            if (io_overlap - l_pad < 0
                    || io_overlap - jcp.l_pad >= right_border)
                continue;

            const bool over_steps_bdry = true && is_last_block
                    && (io_overlap - jcp.l_pad + jcp.r_pad > right_border);
            if (over_steps_bdry) continue;

            ZRegS zregs_input = get_input_reg_s((io_overlap - l_pad) % jcp.kw);
            ZRegS zregs_acc = get_acc_reg_s(i_kw);
            ZRegS zregs_aux = zregs_input;

            fmla(zregs_acc, P_ALL_ONE, zregs_aux, zregs_output);
        }
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::compute_bias_step_unroll(
        const int unroll_w) {
    for (int i = 0; i < unroll_w; ++i) {
        ZRegS zregs_bias = get_bias_reg_s(0);
        int off_output = i * simd_w;
        add_imm(reg_tmp_addr, reg_tmp_output, off_output * sizeof(float),
                reg_tmp_imm);
        if (simd_w == 16) {
            ldr(ZReg(31), ptr(reg_tmp_addr));
        } else if (simd_w == 8) {
            ld1w(ZRegS(31), P_ALL_ONE, ptr(reg_tmp_addr));
        } else {
            assert(!"Unsupport: simd_w != 16, 8");
        }
        fadd(zregs_bias, zregs_bias, ZRegS(31));
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::store_filter() {
    for (int i = 0; i < jcp.kw; ++i) {
        int off_filter = i * simd_w;
        add_imm(reg_tmp_addr, reg_tmp_filter, off_filter * sizeof(float),
                reg_tmp_imm);
        if (simd_w == 16) {
            ZReg zreg_acc = get_acc_reg(i);
            str(zreg_acc, ptr(reg_tmp_addr));
        } else if (simd_w == 8) {
            ZRegS zregs_acc = get_acc_reg_s(i);
            st1w(zregs_acc, P_ALL_ONE, ptr(reg_tmp_addr));
        } else {
            assert(!"Unsupported: simd_w != 16, 8");
        }
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::store_bias() {
    if (simd_w == 16) {
        ZReg zreg_bias = get_bias_reg(0);
        str(zreg_bias, ptr(reg_bias_baddr));
    } else if (simd_w == 8) {
        ZRegS zregs_bias = get_bias_reg_s(0);
        st1w(zregs_bias, P_ALL_ONE, ptr(reg_bias_baddr));
    } else {
        assert(!"Unsupported: simd_w != 16, 8");
    }
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::compute_bias_loop(
        const int block_size) {
    Label oh_label;
    Label ow_blk_label;

    const int unroll_w = nstl::min(block_size, jcp.ow);
    const int unroll_w_trips = jcp.ow / unroll_w;
    const int tail_w = jcp.ow > block_size ? jcp.ow % block_size : 0;

    const int ch_offset = jcp.ch_block;

    ldr(reg_oh,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, oh_index))));
    ldr(reg_oh_worksize,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, oh_count))));

    mov(reg_tmp_output, reg_output_baddr);
    L(oh_label);
    {

        mov_imm(reg_iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
        {

            compute_bias_step_unroll(unroll_w);
            add_imm(reg_tmp_output, reg_tmp_output,
                    unroll_w * ch_offset * sizeof(float), reg_tmp_imm);

            sub(reg_iter_ow_blk, reg_iter_ow_blk, 1);
            cmp(reg_iter_ow_blk, 0);
            b(GT, ow_blk_label);
        }

        if (tail_w > 0) {
            compute_bias_step_unroll(tail_w);
            add_imm(reg_tmp_output, reg_tmp_output,
                    tail_w * ch_offset * sizeof(float), reg_tmp_imm);
        }

        add(reg_oh, reg_oh, 1);
        cmp(reg_oh, reg_oh_worksize);
        b(LT, oh_label);
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::compute_zero_filter() {

    const int ch_offset = jcp.ch_block;

    Label kh_loop_label, skip_zeroing_label;

    ldr(reg_exec_flags,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, exec_flags))));
    and_(reg_exec_flags, reg_exec_flags, FLAG_ZERO_FILTER);
    tst(reg_exec_flags, reg_exec_flags);
    b(EQ, skip_zeroing_label);

    zero_filter();

    mov(reg_tmp_filter, reg_filter_baddr);
    mov_imm(reg_kh, jcp.kh);
    L(kh_loop_label);
    {
        store_filter();

        add_imm(reg_tmp_filter, reg_tmp_filter,
                jcp.kw * ch_offset * sizeof(float), reg_tmp_imm);
        sub(reg_kh, reg_kh, 1);
        cmp(reg_kh, 0);
        b(GT, kh_loop_label);
    }

    /* Comeback pointers */
    sub_imm(reg_tmp_filter, reg_tmp_filter,
            jcp.kh * jcp.kw * ch_offset * sizeof(float), reg_tmp_imm);

    L(skip_zeroing_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::compute_h_step(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    const int ch_offset = jcp.ch_block;

    Label kh_loop_label, skip_loop_label;

    cmp(reg_kh_count, 0);
    b(EQ, skip_loop_label);

    mov(reg_kh, reg_kh_count);
    L(kh_loop_label);
    {
        load_filter();
        compute_ow_step_unroll(unroll_w, l_pad, pad_offset, ow_block);
        store_filter();

        add_imm(reg_tmp_filter, reg_tmp_filter,
                jcp.kw * ch_offset * sizeof(float), reg_tmp_imm);
        add_imm(reg_tmp_input, reg_tmp_input,
                jcp.iw * ch_offset * sizeof(float), reg_tmp_imm);
        sub(reg_kh, reg_kh, 1);
        cmp(reg_kh, 0);
        b(GT, kh_loop_label);
    }

    /* Comeback pointers */
    Label kh_comeback_label;
    mov(reg_kh, reg_kh_count);
    L(kh_comeback_label);
    {
        sub_imm(reg_tmp_input, reg_tmp_input,
                jcp.iw * ch_offset * sizeof(float), reg_tmp_imm);
        sub_imm(reg_tmp_filter, reg_tmp_filter,
                jcp.kw * ch_offset * sizeof(float), reg_tmp_imm);
        sub(reg_kh, reg_kh, 1);
        cmp(reg_kh, 0);
        b(GT, kh_comeback_label);
    }

    L(skip_loop_label);
}

template <cpu_isa_t isa>
inline void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::compute_h_loop(
        int unroll_w, int l_pad, int pad_offset, int ow_block) {

    // last index of output that is not influenced by right padding
    const size_t io_overlap
            = jcp.oh - 1 - utils::div_up(nstl::max(0, jcp.b_pad), jcp.stride_h);

    const int ch_offset = jcp.ch_block;
    const int t_overlap_off = jcp.t_pad % jcp.stride_h == 0 ? jcp.stride_h : 1;
    const int b_overlap_off = jcp.b_pad % jcp.stride_h == 0 ? jcp.stride_h : 1;

    Label tpad_loop_label, h_loop_label, skip_tpad_label, skip_bpad_label;

    ldr(reg_oh,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, oh_index))));
    ldr(reg_oh_worksize,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, oh_count))));
    ldr(reg_kh_count,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, kh_count))));

    mov(reg_tmp_output, reg_output_baddr);
    mov(reg_tmp_input, reg_input_baddr);
    mov(reg_tmp_filter, reg_filter_baddr);

    L(h_loop_label);
    {

        compute_h_step(unroll_w, l_pad, pad_offset, ow_block);

        add_imm(reg_tmp_output, reg_tmp_output,
                jcp.ow * ch_offset * sizeof(float), reg_tmp_imm);

        /* If within the top_pad region */
        if (jcp.t_pad > 0) {
            /* Skip t_pad area if no longer in initial h_block */
            cmp(reg_oh, jcp.t_pad);
            b(GT, skip_tpad_label);

            cmp(reg_kh_count, jcp.kh);
            b(GE, skip_tpad_label);

            add_imm(reg_kh_count, reg_kh_count, t_overlap_off, reg_tmp_imm);
            sub_imm(reg_tmp_filter, reg_tmp_filter,
                    t_overlap_off * jcp.kw * ch_offset * sizeof(float),
                    reg_tmp_imm);

            /* kernel has moved beyond padding (adjust for stride effects) */
            if (jcp.t_pad % jcp.stride_h != 0) {
                int inp_corr = jcp.stride_h - jcp.t_pad % jcp.stride_h;
                add_imm(reg_tmp_input, reg_tmp_input,
                        inp_corr * jcp.iw * ch_offset * sizeof(float),
                        reg_tmp_imm);
            }
            b(tpad_loop_label);
        }

        L(skip_tpad_label);

        cmp(reg_oh, io_overlap);
        b(LT, skip_bpad_label);
        sub_imm(reg_kh_count, reg_kh_count, b_overlap_off, reg_tmp_imm);

        L(skip_bpad_label);
        add_imm(reg_tmp_input, reg_tmp_input,
                jcp.stride_h * jcp.iw * ch_offset * sizeof(float), reg_tmp_imm);

        L(tpad_loop_label);

        add(reg_oh, reg_oh, 1);

        cmp(reg_oh, reg_oh_worksize);
        b(LT, h_loop_label);
    }
}

template <cpu_isa_t isa>
inline void
jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::compute_ow_block_unroll() {

    const int ch_offset = jcp.ch_block;
    int ow = jcp.ow;
    int pad_offset = 0;
    int l_pad = jcp.l_pad;
    int r_pad = jcp.r_pad;

    /* Is this strictly defined by:
     * -code-size (?)
     * -address size (?) */
    const int max_unroll_w = 30;
    const int block_size = 15;

    int unroll_w_tail = 0;
    int unroll_w = 0;
    int unroll_w_trips = 0;
    const bool do_unroll_w = jcp.ow > max_unroll_w;

    if (do_unroll_w) {
        unroll_w = nstl::min(block_size, jcp.ow);
        unroll_w_trips = ow / unroll_w;
        /* calculate tail */
        unroll_w_tail = ow % unroll_w;
        /* Perform some rebalancing if tail too small*/
        if ((unroll_w_tail == 0 && r_pad != 0)
                || (r_pad > 0 && r_pad >= unroll_w_tail)) {
            if (unroll_w_trips > 1) {
                unroll_w_tail += unroll_w;
                unroll_w_trips--;
            } else {
                /* Idealy, this case shouldn't happen */
                unroll_w_tail += (unroll_w - unroll_w / 2);
                unroll_w = unroll_w / 2;
            }
        }
    } else {
        unroll_w_tail = jcp.ow;
    }
    if (jcp.with_bias) {
        Label skip_load_bias;
        ldr(reg_bias_baddr,
                ptr(abi_param1,
                        static_cast<int32_t>(
                                offsetof(jit_dw_conv_args_t, bias))));

        zero_bias();

        ldr(reg_exec_flags,
                ptr(abi_param1,
                        static_cast<int32_t>(
                                offsetof(jit_dw_conv_args_t, exec_flags))));

        and_(reg_exec_flags, reg_exec_flags, FLAG_ZERO_BIAS);
        tst(reg_exec_flags, reg_exec_flags);
        b(NE, skip_load_bias);

        load_bias();

        L(skip_load_bias);
        compute_bias_loop(block_size);

        store_bias();
    }

    /* Pass filter address, then offset for h_padding. */
    compute_zero_filter();
    ldr(reg_kh_offset,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, filter_pad_off))));
    add(reg_filter_baddr, reg_filter_baddr, reg_kh_offset);

    /* compute left padded block */
    if (l_pad && do_unroll_w) {
        compute_h_loop(unroll_w, l_pad, 0, 0);
        add_imm(reg_output_baddr, reg_output_baddr,
                unroll_w * ch_offset * sizeof(float), reg_tmp_imm);
        add_imm(reg_input_baddr, reg_input_baddr,
                unroll_w * jcp.stride_w * ch_offset * sizeof(float),
                reg_tmp_imm);
        unroll_w_trips--;
        pad_offset = l_pad;
        l_pad = 0;
    }

    /* compute middle block */
    Label ow_blk_label;

    /* Insert loop for 'ow' block when middle block needs to execute more
     * than once */
    bool do_ow_blk_loop = unroll_w_trips > 1;
    if (do_ow_blk_loop) {
        mov_imm(reg_iter_ow_blk, unroll_w_trips);
        L(ow_blk_label);
    }
    if (unroll_w_trips > 0) {
        compute_h_loop(unroll_w, l_pad, pad_offset, 0);
        add_imm(reg_output_baddr, reg_output_baddr,
                unroll_w * ch_offset * sizeof(float), reg_tmp_imm);
        add_imm(reg_input_baddr, reg_input_baddr,
                unroll_w * jcp.stride_w * ch_offset * sizeof(float),
                reg_tmp_imm);
    }
    if (do_ow_blk_loop) {
        sub(reg_iter_ow_blk, reg_iter_ow_blk, 1);
        cmp(reg_iter_ow_blk, 0);
        b(GT, ow_blk_label);
    }

    /* compute right padded block */
    if (unroll_w_tail) {
        compute_h_loop(
                unroll_w_tail, l_pad, pad_offset, jcp.ow - unroll_w_tail);
    }
}

template <cpu_isa_t isa>
void jit_uni_dw_conv_bwd_weights_kernel_f32_t<isa>::generate() {
    const int simd_w_ = cpu_isa_traits<isa>::vlen / sizeof(float);
    preamble();
    //TO DO : renaming predicate register (P_ALL_ONE)
    if (simd_w_ != cpu_sveLen / sizeof(float)) {
        set_preg(P_ALL_ONE.s, simd_w_, X_TMP_0);
    }

    if (simd_w_ != 16 && simd_w_ != 8) {
        assert(!"Unsupported: simd_w != 16, 8");
    }

    ldr(reg_input_baddr,
            ptr(abi_param1,
                    static_cast<int32_t>(offsetof(jit_dw_conv_args_t, input))));
    ldr(reg_output_baddr,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, output))));
    ldr(reg_filter_baddr,
            ptr(abi_param1,
                    static_cast<int32_t>(
                            offsetof(jit_dw_conv_args_t, filter))));

    compute_ow_block_unroll();

    this->postamble();
}

template struct jit_uni_dw_conv_bwd_weights_kernel_f32_t<sve_512>;
template struct jit_uni_dw_conv_bwd_weights_kernel_f32_t<sve_256>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
