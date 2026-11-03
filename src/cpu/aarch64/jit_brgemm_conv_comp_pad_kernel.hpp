/*******************************************************************************
* Copyright 2022 Intel Corporation
* Copyright 2024 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_BRGEMM_CONV_COMP_PAD_KERNEL_HPP
#define CPU_AARCH64_JIT_BRGEMM_CONV_COMP_PAD_KERNEL_HPP

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

namespace jit_uni_brgemm_conv_comp_pad_kernel {
struct jit_brgemm_conv_comp_pad_args_t {
    const void *ptr_in;
    void *ptr_zp_out;
    void *ptr_cp_out;
    size_t kw_l;
    size_t kh_l;
    size_t kd_l;
};

template <cpu_isa_t isa>
struct jit_uni_brgemm_conv_comp_pad_kernel_t : public jit_generator_t {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_brgemm_conv_comp_pad_kernel_t)

    using XReg = const Xbyak_aarch64::XReg;

    jit_uni_brgemm_conv_comp_pad_kernel_t(const jit_brgemm_conv_conf_t &ajcp);

    ~jit_uni_brgemm_conv_comp_pad_kernel_t() override = default;

protected:
    static constexpr bool is_ymm_ = true;

    jit_brgemm_conv_conf_t jcp_ = utils::zero<decltype(jcp_)>();
    const int inp_dsz_;
    const int out_dsz_;
    const int ic_block_;
    const size_t nb_ic_;
    const size_t inp_ic_sz_;
    const size_t inp_kw_sz_;
    const size_t inp_kh_sz_;
    const size_t inp_kd_sz_;
    const int isa_max_regs;

    // Register decomposition
    const XReg param1 = abi_param1;
    const XReg reg_in = x15;
    const XReg reg_comp_out = x14;
    const XReg reg_zp_comp_out = x13;

    const XReg reg_kd_l = x12;
    const XReg reg_kh_l = x11;
    const XReg reg_kw_l = x10;
    const XReg reg_icb = x9;

    const XReg reg_aux_in = x8;
    const XReg reg_aux_kh_in = x3;
    const XReg reg_aux_kw_in = x6;
    const XReg reg_tmp = x16;

    Xbyak_aarch64::ZReg vmm_tmp = Xbyak_aarch64::ZReg(isa_max_regs - 1);
    Xbyak_aarch64::ZReg vmm_one_bytes = Xbyak_aarch64::ZReg(isa_max_regs - 2);
    Xbyak_aarch64::ZReg vmm_zp_shift = Xbyak_aarch64::ZReg(isa_max_regs - 3);
    Xbyak_aarch64::ZReg vmm_cp_shift = Xbyak_aarch64::ZReg(isa_max_regs - 4);

    Xbyak_aarch64::ZReg zmm_one_words = Xbyak_aarch64::ZReg(27);
    Xbyak_aarch64::ZReg zmm_int8_temp = Xbyak_aarch64::ZReg(26);

    const int n_block2_ = 4;
    const int m_block2_ = cpu_isa_traits<isa>::vlen / sizeof(int32_t);
    const int n_max_regs_ = 4;

    const Xbyak_aarch64::ZReg &vmm_tmp_1() const noexcept { return vmm_tmp; }

    Xbyak_aarch64::ZReg accum(
            const int n_block, const int m, const int n) const;
    size_t out_oc_offset(const int n) const;
    size_t inp_ic_offset(
            const int m_block, const int icb, const int m, const int n) const;
    int compute_ic_step(
            const int m_max_regs, const int m_block, const int n_block) const;

    void store_accumulators(const int m_block, const int n_block);
    void zero_accumulators(const int m_block, const int n_block);
    void compute(const int ic_step, const int m_block, const int n_block,
            const int m_tail, const bool is_mb_tail);
    void icb_loop(const int icb, const int icb_tail, const int ic_step,
            const int m_block, const int mb_tail, const int n_block);
    void khw_loop(const int icb, const int icb_tail, const int ic_step,
            const int m_block, const int mb_tail, const int n_block);
    void load_params();
    void generate() override;
};

} // namespace jit_uni_brgemm_conv_comp_pad_kernel

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
