/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include <assert.h>

#include "common/type_helpers.hpp"

#include "cpu/x64/jit_avx512_core_fp8cvt.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

void fp8_conversion_base_t::bcst_f8_to_f32(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    const Xbyak::Xmm tmp_xmm(xmm_out.getIdx());
    host_->vpbroadcastb(tmp_xmm, op_in);
    vcvt_f8_to_f32(xmm_out, tmp_xmm);
}

void fp8_conversion_e5m2_t::prepare_table() {
    host_->align(64);
    host_->L(label_vnni_permute_index_table_);
    // Data for f16
    for (int i = 0; i < 64; ++i) {
        // 2, 0, 3, 1, ...
        int index = i;
        switch (i % 4) {
            case 0: index = i + 2; break;
            case 1: index = i - 1; break;
            case 2: index = i + 1; break;
            case 3: index = i - 2; break;
        }
        host_->db(index);
    }
    // Data for bf16
    for (int i = 0; i < 32; ++i) {
        int index = 4 * (i / 2) + (i % 2);
        host_->db(index);
    }
    for (int i = 32; i < 64; ++i) {
        int index = 4 * ((i - 32) / 2) + (i % 2) + 2;
        host_->db(index);
    }

    // Other tables are not needed if fp8 is native
    if (is_fp8_native()) return;

    host_->align(64);
    host_->L(label_table_to_f8_);
    // 0: mask for 2nd msb of f16 mantissa, used in rounding
    for (uint8_t u8 = 0; u8 < 32; ++u8) {
        host_->dw(0x0100);
    }
    // 64: mask for 1st msb of f16 mantissa, used to quiet nans
    for (uint8_t u8 = 0; u8 < 32; ++u8) {
        host_->dw(0x0200);
    }
    // 128: rounding bits of f16 mantissa
    for (uint8_t u8 = 0; u8 < 32; ++u8) {
        host_->dw(0x007f);
    }
    // 192: indices to pack high bytes
    for (uint8_t u8 = 0; u8 < 64; ++u8) {
        int idx = u8 * 2 + 1; // 1, 3, 5, ...
        host_->db(idx);
    }
}

void fp8_conversion_e4m3_t::prepare_table() {
    host_->align(64);
    host_->L(label_vnni_permute_index_table_);
    for (int i = 0; i < 32; ++i) {
        int index = 4 * (i / 2) + (i % 2);
        host_->db(index);
    }
    for (int i = 32; i < 64; ++i) {
        int index = 4 * ((i - 32) / 2) + (i % 2) + 2;
        host_->db(index);
    }

    host_->L(label_table_from_f8_);

    // 0: map from f8_e4m3 byte to high byte of bf16 (ignoring sign)
    for (uint8_t u8 = 0; u8 < 128; ++u8) {
        const float8_e4m3_t x8(u8, /* bit_cast = */ true);
        const bfloat16_t x16 = x8;
        const uint16_t u16 = x16.raw_bits_ >> 8;
        host_->db(u16);
    }
    // 128: map from f8_e4m3 byte to low byte of bf16 (ignoring sign)
    for (uint8_t u8 = 0; u8 < 128; ++u8) {
        const float8_e4m3_t x8(u8, /* bit_cast = */ true);
        const bfloat16_t x16 = x8;
        const uint16_t u16 = (x16.raw_bits_ & 0xff);
        host_->db(u16);
    }
    if (!is_fp8_native()) {
        // 256: map from f8_e4m3 byte to high byte of f16 (ignoring sign)
        for (uint8_t u8 = 0; u8 < 128; ++u8) {
            const float8_e4m3_t x8(u8, /* bit_cast = */ true);
            const float16_t x16 = x8;
            const uint16_t u16 = x16.raw >> 8;
            host_->db(u16);
        }
        // 384: map from f8_e4m3 byte to low byte of f16 (ignoring sign)
        for (uint8_t u8 = 0; u8 < 128; ++u8) {
            const float8_e4m3_t x8(u8, /* bit_cast = */ true);
            const float16_t x16 = x8;
            const uint16_t u16 = (x16.raw & 0xff);
            host_->db(u16);
        }
    }
    // 256 or 512: indices to interleave high and low bytes
    for (uint8_t u8 = 0; u8 < 64; ++u8) {
        int idx = (u8 / 2) + 64 * (u8 % 2); // 0, 64, 1, 65, ...
        host_->db(idx);
    }
    // 320 or 576: sign mask for fp8 values
    host_->dq(0x8080808080808080);

    // Other tables are not needed if fp8 is native
    if (is_fp8_native()) return;

    host_->align(64);
    host_->L(label_table_to_f8_);
    // 0: map from f16 sign+exponent word to f8_e4m3 sign+exponent byte
    static const uint16_t sign_exponent_table[64] = {// positive sign
            0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000,
            0x0000, 0x0000, 0x0008, 0x0010, 0x0018, 0x0020, 0x0028, 0x0030,
            0x0038, 0x0040, 0x0048, 0x0050, 0x0058, 0x0060, 0x0068, 0x0070,
            0x0070, 0x0070, 0x0070, 0x0070, 0x0070, 0x0070, 0x0070, 0x0070,
            // negative sign
            0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080, 0x0080,
            0x0080, 0x0080, 0x0088, 0x0090, 0x0098, 0x00a0, 0x00a8, 0x00b0,
            0x00b8, 0x00c0, 0x00c8, 0x00d0, 0x00d8, 0x00e0, 0x00e8, 0x00f0,
            0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0, 0x00f0};
    for (int i = 0; i < 64; i++) {
        host_->dw(sign_exponent_table[i]);
    }
    // 128: map from f16 sign+exponent to f16 rounding shifter
    static const uint16_t rounding_shifter_table[32] = {0x4000, 0x4000, 0x4000,
            0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4000, 0x4400,
            0x4800, 0x4c00, 0x5000, 0x5400, 0x5800, 0x5c00, 0x6000, 0x6400,
            0x6800, 0x6c00, 0x7000, 0x7400, 0x7800, 0x7800, 0x7800, 0x7800,
            0x7800, 0x7800, 0x7800, 0x7800, 0x7800};
    for (int i = 0; i < 32; i++) {
        host_->dw(rounding_shifter_table[i]);
    }
    // 192: max f16 value that needs to be considered ie 480
    //      (anything greater will convert to same qnan in f8_e3m4)
    for (uint8_t u8 = 0; u8 < 16; ++u8) {
        const uint32_t f8_e4m3_max_threshold = 0x5f805f80;
        host_->dd(f8_e4m3_max_threshold);
    }
    // 256: indices to pack low bytes
    for (uint8_t u8 = 0; u8 < 64; ++u8) {
        int idx = u8 * 2; // 0, 2, 4, ...
        host_->db(idx);
    }
    // 320: absolute value mask for f16 values
    host_->dd(0x7fff7fff);
}

void fp8_conversion_e5m2_t::vcvt_f8_to_f16(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    const Xbyak::Ymm ymm_out(xmm_out.getIdx());
    const Xbyak::Zmm zmm_out(xmm_out.getIdx());

    // f16 <- f8_e5m2
    host_->vpmovzxbw(xmm_out, op_in);
    host_->vpsllw(xmm_out, xmm_out, 8);
    // Floating point conversions typically set quiet bit for NaN inputs.
    // Here we add extra conversions to and from f32 to achieve this, but
    // in practice it should be okay to skip this step.
    if (xmm_out.isZMM()) {
        assert(!utils::one_of(xmm_out.getIdx(), xmm_aux1_.getIdx(),
                xmm_aux2_.getIdx(), xmm_aux3_.getIdx()));

        const Xbyak::Ymm ymm_aux1(xmm_aux1_.getIdx());
        const Xbyak::Zmm zmm_aux1(xmm_aux1_.getIdx());
        // process upper bytes
        host_->vextractf64x4(ymm_aux1, zmm_out, 1);
        host_->vcvtph2psx(zmm_aux1, ymm_aux1);
        host_->vcvtps2phx(ymm_aux1, zmm_aux1);
        host_->vinsertf64x4(zmm_out, zmm_out, ymm_aux1, 1);

        // process low bytes
        host_->vextractf64x4(ymm_aux1, zmm_out, 0);
        host_->vcvtph2psx(zmm_aux1, ymm_aux1);
        host_->vcvtps2phx(ymm_aux1, zmm_aux1);
        host_->vinsertf64x4(zmm_out, zmm_out, ymm_aux1, 0);
    } else {
        // f32 <- f16
        host_->vcvtph2psx(zmm_out, ymm_out);
        // f16 <- f32
        host_->vcvtps2phx(ymm_out, zmm_out);
    }
}

void fp8_conversion_e5m2_t::vcvt_f8_to_f16_skip_q_nan(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    // f16 <- f8_e5m2
    host_->vpmovzxbw(xmm_out, op_in);
    host_->vpsllw(xmm_out, xmm_out, 8);
}

void fp8_conversion_e5m2_t::vcvt_f8_to_bf16(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    assert(xmm_out.isZMM());
    assert(xmm_out.getIdx() != xmm_aux3_.getIdx());

    // f16 <- f8_e5m2
    // Floating point conversions typically set quiet bit for NaN inputs.
    // Here we skip this step as it will be handled during f32<-f16 stage.
    host_->vpmovzxbw(xmm_out, op_in);
    host_->vpsllw(xmm_out, xmm_out, 8);

    // copy high bytes to auxiliary register
    const Xbyak::Zmm zmm_out(xmm_out.getIdx());
    const Xbyak::Ymm ymm_aux_out_hi(xmm_aux3_.getIdx());
    host_->vextractf64x4(ymm_aux_out_hi, zmm_out, 1);

    // f32 <- f16
    const Xbyak::Zmm zmm_aux_out_hi(xmm_aux3_.getIdx());
    host_->vcvtph2psx(zmm_aux_out_hi, ymm_aux_out_hi);
    const Xbyak::Ymm ymm_out(xmm_out.getIdx());
    host_->vcvtph2psx(zmm_out, ymm_out);

    // bf16 <- f32
    host_->vcvtneps2bf16(ymm_aux_out_hi, zmm_aux_out_hi);
    host_->vcvtneps2bf16(ymm_out, zmm_out);

    // merge bytes
    host_->vinsertf64x4(zmm_out, zmm_out, ymm_aux_out_hi, 1);
}

void fp8_conversion_e5m2_t::prepare_f8_to_f16_vnni_masks(int zmm_permute_idx) {
    const uint64_t mask_even = 0xAAAAAAAAAAAAAAAA;
    host_->mov(reg64_aux_, mask_even);
    host_->kmovq(kmask_aux_, reg64_aux_);
    const Xbyak::Zmm zmm_permute(zmm_permute_idx);
    host_->vmovups(zmm_permute,
            host_->ptr[host_->rip + label_vnni_permute_index_table_]);
}

void fp8_conversion_e5m2_t::perform_f8_to_f16_vnni_conversion(
        const Xbyak::Zmm &zmm_out1, const Xbyak::Zmm &zmm_out2,
        const Xbyak::Operand &op_in, int zmm_permute_idx) {
    const Xbyak::Zmm zmm_permute(zmm_permute_idx);
    host_->vpermb(zmm_out1, zmm_permute, op_in); // 3210 -> 1302
    host_->vpsllw(zmm_out2, zmm_out1, 8); // 1302 -> 3_2_
    host_->vmovdqu8(
            zmm_out1 | kmask_aux_ | host_->T_z, zmm_out1); // 1302 -> 1_0_
}

void fp8_conversion_e5m2_t::vcvt_f8_to_f16_vnni(const Xbyak::Zmm &zmm_out1,
        const Xbyak::Zmm &zmm_out2, const Xbyak::Operand &op_in) {
    const int zmm_permute_idx = xmm_aux3_.getIdx();
    prepare_f8_to_f16_vnni_masks(zmm_permute_idx);
    perform_f8_to_f16_vnni_conversion(
            zmm_out1, zmm_out2, op_in, zmm_permute_idx);
}

void fp8_conversion_e5m2_t::vcvt_f8_to_bf16_vnni(const Xbyak::Zmm &zmm_out1,
        const Xbyak::Zmm &zmm_out2, const Xbyak::Operand &op_in) {
    const Xbyak::Ymm ymm_out1(zmm_out1.getIdx());
    const Xbyak::Ymm ymm_out2(zmm_out2.getIdx());

    const Xbyak::Zmm zmm_permute(xmm_aux3_.getIdx());

    host_->vmovups(zmm_permute,
            host_->ptr[host_->rip + label_vnni_permute_index_table_ + 64]);
    // first take the two low bytes from each quad and then the two upper bytes
    host_->vpermb(zmm_out1, zmm_permute, op_in);
    // ymm_aux1 contains fp8 values from low bytes
    // move fp8 values from upper bytes to ymm_aux2
    host_->vextractf64x4(ymm_out2, zmm_out1, 1);

    vcvt_f8_to_bf16(zmm_out1, ymm_out1);
    vcvt_f8_to_bf16(zmm_out2, ymm_out2);
}

void fp8_conversion_e5m2_t::vcvt_f8_to_f16_vnni_block(int num_rows,
        const Xbyak::Reg64 &reg_data_in, const Xbyak::Reg64 &reg_stride_in,
        const Xbyak::Reg64 &reg_data_out) {
    const Xbyak::Zmm zmm_aux1(xmm_aux1_.getIdx());
    const Xbyak::Zmm zmm_aux2(xmm_aux2_.getIdx());
    const int zmm_permute_idx = xmm_aux3_.getIdx();

    prepare_f8_to_f16_vnni_masks(zmm_permute_idx);

    const auto zmm_width_in_bytes = cpu_isa_traits_t<avx512_core>::vlen;

    for (int r = 0; r < num_rows; r += 2) {
        perform_f8_to_f16_vnni_conversion(
                zmm_aux1, zmm_aux2, host_->ptr[reg_data_in], zmm_permute_idx);

        host_->vmovups(
                host_->ptr[reg_data_out + r * zmm_width_in_bytes], zmm_aux1);
        host_->vmovups(host_->ptr[reg_data_out + (r + 1) * zmm_width_in_bytes],
                zmm_aux2);
        host_->lea(reg_data_in, host_->ptr[reg_data_in + reg_stride_in]);
    }
}

void fp8_conversion_e5m2_t::vcvt_f8_to_bf16_vnni_block(int num_rows,
        const Xbyak::Reg64 &reg_data_in, const Xbyak::Reg64 &reg_stride_in,
        const Xbyak::Reg64 &reg_data_out) {
    constexpr auto zmm_width_in_bytes = cpu_isa_traits_t<avx512_core>::vlen;
    const Xbyak::Zmm zmm_out1(xmm_aux1_.getIdx());
    const Xbyak::Zmm zmm_out2(xmm_aux2_.getIdx());
    const Xbyak::Ymm ymm_out1(zmm_out1.getIdx());
    const Xbyak::Ymm ymm_out2(zmm_out2.getIdx());

    const Xbyak::Zmm zmm_permute(xmm_aux3_.getIdx());
    for (int r = 0; r < num_rows; r += 2) {
        host_->vmovups(zmm_permute,
                host_->ptr[host_->rip + label_vnni_permute_index_table_ + 64]);

        host_->vpermb(
                zmm_out1, zmm_permute, host_->ptr[reg_data_in]); // 3210 -> 1302
        // ymm_out1 contains fp8 values from low bytes
        // move fp8 values from upper bytes to ymm_out2
        host_->vextractf64x4(ymm_out2, zmm_out1, 1);

        vcvt_f8_to_bf16(zmm_out1, ymm_out1);
        vcvt_f8_to_bf16(zmm_out2, ymm_out2);

        host_->vmovups(
                host_->ptr[reg_data_out + r * zmm_width_in_bytes], zmm_out1);
        host_->vmovups(host_->ptr[reg_data_out + (r + 1) * zmm_width_in_bytes],
                zmm_out2);

        host_->lea(reg_data_in, host_->ptr[reg_data_in + reg_stride_in]);
    }
}

void fp8_conversion_e5m2_t::vcvt_f8_to_f32(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    assert(xmm_out.isZMM());
    const Xbyak::Ymm ymm_out(xmm_out.getIdx());
    const Xbyak::Zmm zmm_out(xmm_out.getIdx());

    // f16 <- f8_e5m2
    // Floating point conversions typically set quiet bit for NaN inputs.
    // Here we skip this step as it will be handled during f32<-f16 stage.
    host_->vpmovzxbw(ymm_mask(xmm_out), op_in);
    host_->vpsllw(ymm_out, ymm_out, 8);
    // f32 <- f16
    host_->vcvtph2psx(zmm_out, ymm_out);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_xf16(const Xbyak::Xmm &xmm_out,
        const Xbyak::Operand &op_in, data_type_t dt) {
    assert(utils::one_of(dt, data_type::bf16, data_type::f16));
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));

    const bool is_f16 = dt == data_type::f16;
    const bool is_native_conv = is_fp8_native();

    if (is_f16 && is_native_conv) {
        if (!op_in.isMEM()) {
            // src register must be at least half the size of dst register (xbyak checkCvt1)
            if (xmm_out.isZMM())
                host_->vcvthf82ph(xmm_out, Xbyak::Ymm(op_in.getIdx()));
            else
                host_->vcvthf82ph(xmm_out, Xbyak::Xmm(op_in.getIdx()));
        } else {
            host_->vcvthf82ph(xmm_out, op_in);
        }
        return;
    }

    const int xf16_hi_bytes_offset = is_f16 ? 256 : 0;
    const int xf16_lo_bytes_offset = is_f16 ? 384 : 128;
    const int interleave_indices_offset = is_native_conv ? 256 : 512;
    const int sign_mask_offset = interleave_indices_offset + 64;

    host_->lea(reg64_aux_,
            host_->ptr[host_->rip + label_table_from_f8_]); // base of table

    // must use full Zmm to properly load all table values
    const Xbyak::Zmm zmm_in(
            op_in.isMEM() ? xmm_aux3_.getIdx() : op_in.getIdx());
    const Xbyak::Zmm zmm_aux1(xmm_aux1_.getIdx());
    const Xbyak::Zmm zmm_aux2(xmm_aux2_.getIdx());
    const Xbyak::Zmm zmm_tmp(xmm_aux3_.getIdx());

    // if output register is ymm then we read xmm only
    const auto vmm_in = xmm_out.isYMM() ? xmm_mask(zmm_in, xmm_out)
                                        : ymm_mask(zmm_in, xmm_out);

    if (op_in.isMEM()) host_->vmovdqu8(vmm_in, op_in);
    // xf16 <- f8_e4m3
    tabulate(data_type::f8_e4m3, zmm_aux1, zmm_in,
            host_->zword[reg64_aux_ + xf16_hi_bytes_offset]); // high byte
    tabulate(data_type::f8_e4m3, zmm_aux2, zmm_in,
            host_->zword[reg64_aux_ + xf16_lo_bytes_offset]); // low byte
    // sign correction
    // 0xf8 means A = A | (B & C)
    host_->vpternlogq(zmm_aux1, zmm_in,
            host_->ptr_b[reg64_aux_ + sign_mask_offset], 0xf8);
    // merge high and low
    host_->vmovdqu64(
            zmm_tmp, host_->zword[reg64_aux_ + interleave_indices_offset]);
    host_->vpermt2b(zmm_aux2, zmm_tmp, zmm_aux1);

    host_->vmovdqu16(xmm_out, zmm_aux2);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_f16(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    vcvt_f8_to_xf16(xmm_out, op_in, data_type::f16);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_bf16(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    vcvt_f8_to_xf16(xmm_out, op_in, data_type::bf16);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_xf16_vnni(const Xbyak::Zmm &zmm_out1,
        const Xbyak::Zmm &zmm_out2, const Xbyak::Operand &op_in,
        data_type_t dt) {
    const Xbyak::Ymm ymm_out1(zmm_out1.getIdx());
    const Xbyak::Ymm ymm_out2(zmm_out2.getIdx());

    const Xbyak::Zmm zmm_permute(xmm_aux3_.getIdx());

    host_->vmovups(zmm_permute,
            host_->ptr[host_->rip + label_vnni_permute_index_table_]);
    // first take the two low bytes from each quad and then the two upper bytes
    host_->vpermb(zmm_out1, zmm_permute, op_in);
    // ymm_aux1 contains fp8 values from low bytes
    // move fp8 values from upper bytes to ymm_aux2
    host_->vextractf64x4(ymm_out2, zmm_out1, 1);

    vcvt_f8_to_xf16(zmm_out1, ymm_out1, dt);
    vcvt_f8_to_xf16(zmm_out2, ymm_out2, dt);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_f16_vnni(const Xbyak::Zmm &zmm_out1,
        const Xbyak::Zmm &zmm_out2, const Xbyak::Operand &op_in) {
    vcvt_f8_to_xf16_vnni(zmm_out1, zmm_out2, op_in, data_type::f16);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_bf16_vnni(const Xbyak::Zmm &zmm_out1,
        const Xbyak::Zmm &zmm_out2, const Xbyak::Operand &op_in) {
    vcvt_f8_to_xf16_vnni(zmm_out1, zmm_out2, op_in, data_type::bf16);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_xf16_vnni_block(int num_rows,
        const Xbyak::Reg64 &reg_data_in, const Xbyak::Reg64 &reg_stride_in,
        const Xbyak::Reg64 &reg_data_out, data_type_t dt) {
    constexpr auto zmm_width_in_bytes = cpu_isa_traits_t<avx512_core>::vlen;
    const Xbyak::Zmm zmm_out1(xmm_aux4_.getIdx());
    const Xbyak::Zmm zmm_out2(xmm_aux5_.getIdx());
    const Xbyak::Ymm ymm_out1(zmm_out1.getIdx());
    const Xbyak::Ymm ymm_out2(zmm_out2.getIdx());

    const Xbyak::Zmm zmm_permute(xmm_aux3_.getIdx());
    for (int r = 0; r < num_rows; r += 2) {
        host_->vmovups(zmm_permute,
                host_->ptr[host_->rip + label_vnni_permute_index_table_]);

        host_->vpermb(
                zmm_out1, zmm_permute, host_->ptr[reg_data_in]); // 3210 -> 1302
        // ymm_out1 contains fp8 values from low bytes
        // move fp8 values from upper bytes to ymm_out2
        host_->vextractf64x4(ymm_out2, zmm_out1, 1);

        vcvt_f8_to_xf16(zmm_out1, ymm_out1, dt);
        vcvt_f8_to_xf16(zmm_out2, ymm_out2, dt);

        host_->vmovups(
                host_->ptr[reg_data_out + r * zmm_width_in_bytes], zmm_out1);
        host_->vmovups(host_->ptr[reg_data_out + (r + 1) * zmm_width_in_bytes],
                zmm_out2);

        host_->lea(reg_data_in, host_->ptr[reg_data_in + reg_stride_in]);
    }
}

void fp8_conversion_e4m3_t::vcvt_f8_to_f16_vnni_block(int num_rows,
        const Xbyak::Reg64 &reg_data_in, const Xbyak::Reg64 &reg_stride_in,
        const Xbyak::Reg64 &reg_data_out) {
    vcvt_f8_to_xf16_vnni_block(
            num_rows, reg_data_in, reg_stride_in, reg_data_out, data_type::f16);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_f32(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    assert(xmm_out.isZMM());
    const Xbyak::Ymm ymm_out(xmm_out.getIdx());
    const Xbyak::Zmm zmm_out(xmm_out.getIdx());

    // f16 <- f8_e4m3
    vcvt_f8_to_f16(ymm_mask(xmm_out), op_in);
    // f32 <- f16
    host_->vcvtph2psx(zmm_out, ymm_out);
}

void fp8_conversion_e4m3_t::bcst_f8_to_f32(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    if (is_fp8_native()) {
        fp8_conversion_base_t::bcst_f8_to_f32(xmm_out, op_in);
        return;
    }
    // The output may alias an aux register, which the emulation overwrites
    // before reading its input. Broadcast into the register the memory
    // operand path loads into instead.
    host_->vpbroadcastb(xmm_aux3_, op_in);
    vcvt_f8_to_f32(xmm_out, xmm_aux3_);
}

void fp8_conversion_e4m3_t::vcvt_f8_to_bf16_vnni_block(int num_rows,
        const Xbyak::Reg64 &reg_data_in, const Xbyak::Reg64 &reg_stride_in,
        const Xbyak::Reg64 &reg_data_out) {
    vcvt_f8_to_xf16_vnni_block(num_rows, reg_data_in, reg_stride_in,
            reg_data_out, data_type::bf16);
}

void fp8_conversion_e5m2_t::vcvt_f32_to_f8(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    assert(xmm_out.isXMM());

    const Xbyak::Ymm ymm_out(xmm_out.getIdx());

    // f16 <- f32
    host_->vcvtps2phx(op_in.isXMM() ? xmm_out : ymm_mask(xmm_out), op_in);
    // f8_e5m2 <- f16 (RNE)
    if (is_fp8_native())
        host_->vcvtph2bf8(xmm_out, ymm_out);
    else
        vcvt_f16_to_f8(xmm_out, ymm_out);
}

void fp8_conversion_e5m2_t::vcvt_f16_to_f8(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    if (is_fp8_native()) {
        host_->vcvtph2bf8(xmm_out, op_in);
        return;
    }

    // load base of table
    host_->lea(reg64_aux_, host_->ptr[host_->rip + label_table_to_f8_]);

    const Xbyak::Ymm ymm_in(
            op_in.isMEM() ? xmm_aux2_.getIdx() : op_in.getIdx());
    const Xbyak::Ymm ymm_aux1(xmm_aux1_.getIdx());
    const Xbyak::Ymm ymm_out(xmm_out.getIdx());

    if (op_in.isMEM()) host_->vmovdqu32(ymm_mask(ymm_in, xmm_out), op_in);

    // create mask from nan values
    host_->vfpclassph(kmask_aux_, ymm_in, 0x81);

    // get 2nd msb of mantissa and move to lsb for RNE tiebreaking
    host_->vpandd(ymm_aux1, ymm_in, host_->ptr_b[reg64_aux_ + 0]);
    host_->vpsrlw(ymm_aux1, ymm_aux1, 8);

    // blend masks for nan handling and rounding
    host_->vmovdqu16(ymm_aux1 | kmask_aux_, host_->ptr[reg64_aux_ + 64]);
    // set 1st msb of mantissa for nans (snan->qnan)
    // set 2nd msb of mantissa for RNE tiebreaking
    host_->vporq(ymm_out, ymm_aux1, ymm_in);

    // perform rounding by adding rounding bits
    host_->vpaddw(ymm_aux1, ymm_out, host_->ptr[reg64_aux_ + 128]);
    // reset nans
    host_->vmovdqu16(ymm_aux1 | kmask_aux_, ymm_out);

    // extract the result (high byte of each word)
    // load odd indices from table
    host_->vmovdqu64(xmm_aux2_, host_->ptr[reg64_aux_ + 192]);
    // pack odd bytes to lower half of register
    // NOTE: there will be garbage in upper half of register
    host_->vpermb(xmm_out, xmm_aux2_, ymm_aux1);
}

void fp8_conversion_e4m3_t::vcvt_f32_to_f8(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    assert(xmm_out.isXMM());

    const Xbyak::Ymm ymm_out(xmm_out.getIdx());

    // f16 <- f32
    host_->vcvtps2phx(ymm_mask(xmm_out), op_in);
    // f8_e4m3 <- f16 (RNE)
    if (is_fp8_native())
        host_->vcvtph2hf8(xmm_out, ymm_out);
    else
        vcvt_f16_to_f8(xmm_out, ymm_out);
}

void fp8_conversion_e4m3_t::vcvt_f16_to_f8(
        const Xbyak::Xmm &xmm_out, const Xbyak::Operand &op_in) {
    assert(utils::one_of(
            true, op_in.isXMM(), op_in.isYMM(), op_in.isZMM(), op_in.isMEM()));
    if (is_fp8_native()) {
        host_->vcvtph2hf8(xmm_out, op_in);
        return;
    }

    host_->lea(reg64_aux_,
            host_->ptr[host_->rip + label_table_to_f8_]); // load base of table

    const Xbyak::Ymm ymm_in(
            op_in.isMEM() ? xmm_aux3_.getIdx() : op_in.getIdx());
    const Xbyak::Ymm ymm_aux1(xmm_aux1_.getIdx());
    const Xbyak::Ymm ymm_aux2(xmm_aux2_.getIdx());
    const Xbyak::Ymm ymm_aux3(xmm_aux3_.getIdx());
    const Xbyak::Ymm ymm_out(xmm_out.getIdx());

    if (op_in.isMEM()) host_->vmovdqu16(ymm_mask(ymm_in, xmm_out), op_in);

    // get absolute value of f16 values
    host_->vpandd(ymm_aux1, ymm_in, host_->ptr_b[reg64_aux_ + 320]);

    // isolate sign + exponent bits
    host_->vpsrlw(ymm_out, ymm_in, 10);

    const Xbyak::Zmm zmm_in(ymm_in.getIdx());
    const Xbyak::Zmm zmm_out(xmm_out.getIdx());
    const Xbyak::Zmm zmm_aux1(xmm_aux1_.getIdx());
    const Xbyak::Zmm zmm_aux2(xmm_aux2_.getIdx());
    // map from f16 sign + exponent to f8_e4m3 sign + exponent?
    tabulate(data_type::f16, zmm_aux2, zmm_out, host_->zword[reg64_aux_]);

    // map from f16 sign + exponent to rounding shifter
    // NOTE: must use full Zmm to access all table values
    host_->vpermw(zmm_out, zmm_out, host_->zword[reg64_aux_ + 128]);

    // all f16 values greater than some threshold will become nan,
    // so we take min w.r.t. this threshold
    host_->vmovdqu64(ymm_aux3, host_->ptr[reg64_aux_ + 192]);
    host_->vpminuw(ymm_aux1, ymm_aux1, ymm_aux3);

    // these additions will not overflow due to taking min above
    // NOTE: must use full Zmm to enable embedded rounding
    host_->vaddph(zmm_out | host_->T_rn_sae, zmm_aux1, zmm_out);
    host_->vpaddw(ymm_out, ymm_aux2, ymm_out);

    // extract the result (low byte of each word)
    // load even indices from table
    host_->vmovdqu64(xmm_aux3_, host_->ptr[reg64_aux_ + 256]);
    // pack even bytes to lower half of register
    // NOTE: there will be garbage in upper half of register
    host_->vpermb(xmm_out, xmm_aux3_, ymm_out);
}

void fp8_conversion_e4m3_t::tabulate(const data_type_t dt,
        const Xbyak::Zmm &zmm_out, const Xbyak::Zmm &zmm_in,
        const Xbyak::Address &addr) {
    host_->vmovdqu64(zmm_out, addr);
    switch (dt) {
        case data_type::f8_e4m3:
            host_->vpermt2b(
                    zmm_out, zmm_in, host_->zword[addr.getRegExp() + 64]);
            break;
        case data_type::f16:
            host_->vpermt2w(
                    zmm_out, zmm_in, host_->zword[addr.getRegExp() + 64]);
            break;
        default: assert(!"Unsupported data type in helper routine");
    }
}

jit_cvt_fp8_t::jit_cvt_fp8_t(f32_convert_mode_t mode)
    : jit_generator_t(jit_name(), avx512_core_fp16), mode_(mode) {
    switch (mode) {
        case f8_e5m2_to_f16:
        case f8_e5m2_to_f32:
        case f16_to_f8_e5m2:
        case f32_to_f8_e5m2:
            safe_ptr_assign(fp8_cvt_,
                    new fp8_conversion_e5m2_t(this, xmm_aux1, xmm_aux2,
                            xmm_aux3, kmask_aux, reg64_aux));
            break;
        case f8_e4m3_to_f16:
        case f8_e4m3_to_f32:
        case f16_to_f8_e4m3:
        case f32_to_f8_e4m3:
            safe_ptr_assign(fp8_cvt_,
                    new fp8_conversion_e4m3_t(this, xmm_aux1, xmm_aux2,
                            xmm_aux3, xmm_aux4, xmm_aux5, reg64_aux));
            break;
        case f16_to_f32:
        case f32_to_f16: break;
        default: assert(!"Invalid mode!");
    }
    create_kernel();
}

void jit_cvt_fp8_t::generate() {
    const Xbyak::Zmm zmm_out(xmm_out.getIdx());
    const Xbyak::Ymm ymm_out(xmm_out.getIdx());
    switch (mode_) {
        case f8_e5m2_to_f32:
        case f8_e4m3_to_f32:
            movzx(reg64_aux, byte[reg64_inp]);
            vmovq(xmm_inp, reg64_aux);
            fp8_cvt_->vcvt_f8_to_f32(zmm_out, xmm_inp);
            vmovss(dword[reg64_out], xmm_out);
            break;
        case f8_e5m2_to_f16:
        case f8_e4m3_to_f16:
            movzx(reg64_aux, byte[reg64_inp]);
            vmovq(xmm_inp, reg64_aux);
            fp8_cvt_->vcvt_f8_to_f16(ymm_out, xmm_inp);
            vmovw(word[reg64_out], xmm_out);
            break;
        case f16_to_f8_e5m2:
        case f16_to_f8_e4m3:
            vmovw(xmm_inp, word[reg64_inp]);
            fp8_cvt_->vcvt_f16_to_f8(xmm_out, xmm_inp);
            vpextrb(byte[reg64_out], xmm_out, 0);
            break;
        case f32_to_f8_e5m2:
        case f32_to_f8_e4m3:
            vmovss(xmm_inp, dword[reg64_inp]);
            fp8_cvt_->vcvt_f32_to_f8(xmm_out, xmm_inp);
            vpextrb(byte[reg64_out], xmm_out, 0);
            break;
        case f16_to_f32:
            vmovw(xmm_inp, word[reg64_inp]);
            vcvtph2psx(xmm_out, xmm_inp);
            vmovss(dword[reg64_out], xmm_out);
            break;
        case f32_to_f16:
            vmovss(xmm_inp, dword[reg64_inp]);
            vcvtps2phx(xmm_out, xmm_inp);
            vmovw(word[reg64_out], xmm_out);
            break;
        default: assert(!"Invalid mode!");
    }
    ret();
    switch (mode_) {
        case f8_e5m2_to_f16:
        case f8_e4m3_to_f16:
        case f8_e5m2_to_f32:
        case f8_e4m3_to_f32:
        case f16_to_f8_e5m2:
        case f16_to_f8_e4m3:
        case f32_to_f8_e5m2:
        case f32_to_f8_e4m3: fp8_cvt_->prepare_table(); break;
        case f16_to_f32:
        case f32_to_f16: break;
        default: assert(!"Invalid mode!");
    }
}

bool try_cvt_f8_e5m2_to_f32(float *out, const float8_e5m2_t *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f8_e5m2_to_f32);
    cvt(out, inp);
    return true;
}

bool try_cvt_f8_e4m3_to_f32(float *out, const float8_e4m3_t *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f8_e4m3_to_f32);
    cvt(out, inp);
    return true;
}

bool try_cvt_f8_e5m2_to_f16(float16_t *out, const float8_e5m2_t *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f8_e5m2_to_f16);
    cvt(out, inp);
    return true;
}

bool try_cvt_f8_e4m3_to_f16(float16_t *out, const float8_e4m3_t *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f8_e4m3_to_f16);
    cvt(out, inp);
    return true;
}

bool try_cvt_f16_to_f8_e5m2(float8_e5m2_t *out, const float16_t *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f16_to_f8_e5m2);
    cvt(out, inp);
    return true;
}

bool try_cvt_f16_to_f8_e4m3(float8_e4m3_t *out, const float16_t *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f16_to_f8_e4m3);
    cvt(out, inp);
    return true;
}

bool try_cvt_f32_to_f8_e5m2(float8_e5m2_t *out, const float *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f32_to_f8_e5m2);
    cvt(out, inp);
    return true;
}

bool try_cvt_f32_to_f8_e4m3(float8_e4m3_t *out, const float *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f32_to_f8_e4m3);
    cvt(out, inp);
    return true;
}

bool try_cvt_f16_to_f32(float *out, const float16_t *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f16_to_f32);
    cvt(out, inp);
    return true;
}

bool try_cvt_f32_to_f16(float16_t *out, const float *inp) {
    if (!mayiuse(cpu_isa_t::avx512_core_fp16)) return false;

    static const jit_cvt_fp8_t cvt(f32_to_f16);
    cvt(out, inp);
    return true;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
