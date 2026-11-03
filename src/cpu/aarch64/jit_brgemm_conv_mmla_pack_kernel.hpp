/*******************************************************************************
* Copyright 2026 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_BRGEMM_CONV_MMLA_PACK_KERNEL_HPP
#define CPU_AARCH64_JIT_BRGEMM_CONV_MMLA_PACK_KERNEL_HPP

#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace jit_brgemm_conv_mmla_pack_kernel {

// Packs source rows into 2x8b interleaved chunks for SVE mmla
struct jit_brgemm_conv_mmla_pack_args_t {
    const brgemm_batch_element_t *batch;
    char *dst;
    const char *pad_row;
    size_t bs;
    size_t rows;
};

struct jit_brgemm_conv_mmla_pack_kernel_t : public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_brgemm_conv_mmla_pack_kernel_t)

    // K: reduction size (bytes) of the brgemm call. lda_bytes: distance
    // between two consecutive rows in the source.
    jit_brgemm_conv_mmla_pack_kernel_t(
            const jit_brgemm_conv_conf_t &ajcp, int K, dim_t lda_bytes);
    ~jit_brgemm_conv_mmla_pack_kernel_t() override = default;

    // Size in bytes of one packed tap for `rows` rows.
    static dim_t tap_size(int rows, int K) {
        return static_cast<dim_t>(utils::div_up(rows, 2)) * utils::div_up(K, 8)
                * 16;
    }

private:
    using XReg = const Xbyak_aarch64::XReg;
    using ZReg = const Xbyak_aarch64::ZReg;
    using PReg = const Xbyak_aarch64::PReg;

    const int K_;
    const dim_t lda_;
    const int vlen_;

    // Registers
    XReg reg_param = abi_param1;
    XReg reg_batch = x1;
    XReg reg_bs = x2;
    XReg reg_dst = x3;
    XReg reg_pad = x4;
    XReg reg_rows = x5;
    XReg reg_row0 = x6; // source address of row 2p of the current tap
    XReg reg_top = x7;
    XReg reg_vend = x8; // rows - vvpad.bottom
    XReg reg_dst_cur = x9;
    XReg reg_r = x10;
    XReg reg_src0 = x11;
    XReg reg_src1 = x12;
    XReg reg_r1 = x13;
    XReg reg_tmp = x14;

    ZReg z_r0 = Xbyak_aarch64::ZReg(0);
    ZReg z_r1 = Xbyak_aarch64::ZReg(1);
    ZReg z_lo = Xbyak_aarch64::ZReg(2);
    ZReg z_hi = Xbyak_aarch64::ZReg(3);

    PReg p_tail_ld = Xbyak_aarch64::PReg(1);
    PReg p_tail_st0 = Xbyak_aarch64::PReg(2);
    PReg p_tail_st1 = Xbyak_aarch64::PReg(3);

    void select_row_src(
            const XReg &dst, const XReg &row_idx, const XReg &row_addr);
    void pack_pair();
    void generate() override;
};

} // namespace jit_brgemm_conv_mmla_pack_kernel

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
