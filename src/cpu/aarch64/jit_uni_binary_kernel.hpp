/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2022-2023 FUJITSU LIMITED
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

#ifndef CPU_AARCH64_JIT_UNI_BINARY_KERNEL_HPP
#define CPU_AARCH64_JIT_UNI_BINARY_KERNEL_HPP

#include <cassert>

#include "common/binary_pd.hpp"
#include "common/c_types_map.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_primitive_conf.hpp"
#include "cpu/aarch64/utils/jit_io_helper.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

struct binary_kernel_t : public jit_generator_t {
    using op_t = binary_op_t;
    using bcast_t = binary_bcast_t;

    binary_kernel_t(const size_t vlen, const binary_pd_t *pd,
            const jit_binary_conf_t conf, bool tail_kernel = false);
    ~binary_kernel_t() override = default;

    void operator()(jit_uni_binary_args_t *p) {
        jit_generator_t::operator()(p);
    }

    size_t simd_w() const noexcept { return simd_w_; }
    size_t vlen() const noexcept { return vlen_; }

protected:
    size_t get_tail_size() const;

    const size_t vlen_;
    const size_t simd_w_;
    constexpr static int vmm_start_idx_ = 1;
    const binary_pd_t *pd_;
    const jit_binary_conf_t conf_;
    const bool is_tail_kernel_;
    const bool is_src1_outer_dims_tail_;
    const size_t tail_size_;
    const size_t padding_tail_size_;
};

template <cpu_isa_t isa>
struct jit_uni_binary_kernel_t : public binary_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_binary_kernel_t)

    using Vmm = typename cpu_isa_traits<isa>::TReg;

    const XReg reg_param_ = abi_param1;
    const XReg reg_src0_ = x8;
    const XReg reg_src1_ = x9;
    const XReg reg_dst_ = x10;
    const XReg reg_offt_src0_ = x11;
    const XReg reg_outer_dims_range_ = x12;
    const XReg reg_offt_src1_ = x1;
    const XReg reg_src1_stride_range_ = x15;
    const XReg reg_reverse_src1_stride_range_ = x1;
    const XReg reg_reverse_spat_offt_ = x13;
    const XReg reg_tmp_ = x14;
    const XReg reg_tmp1_ = x6;
    const XReg reg_off_rhs_postops_ = x2;
    const XReg reg_scales_src0_ = x3;
    const XReg reg_scales_src1_ = x5;
    const XReg reg_offt_dst_ = x2;
    const PReg tail_opmask_ = p2;
    const PReg cmp_mask = p3;
    const Vmm vmm_tail_vmask_ = Vmm(0);
    const Vmm vreg_sum_scale_ = Vmm(17);
    const VReg xreg_sum_scale_ = VReg(9);
    const Vmm vreg_zero_ = Vmm(18);
    const Vmm vreg_one_ = Vmm(19);
    const Vmm vreg_saturation_lbound_ = Vmm(24);
    const Vmm vreg_saturation_ubound_ = Vmm(20);
    const Vmm vreg_bcast_src1_ = Vmm(21);
    const VReg xreg_bcast_src1_ = VReg(13);
    const Vmm vreg_scales_src0_ = Vmm(22);
    const Vmm vreg_scales_src1_ = Vmm(23);

    const Vmm vmm_indices_ = Vmm(30);
    const Vmm vmm_gathered_src_ = Vmm(31);

    const size_t unroll_regs_ = 8;
    const size_t offt_src0_;
    const size_t offt_src1_;

    static constexpr cpu_isa_t inject_isa = isa;
    std::unique_ptr<io::jit_io_helper_t<to_vla_sve(isa)>> io_;
    std::unique_ptr<
            injector::jit_uni_postops_injector_t<to_vla_sve(inject_isa)>>
            postops_injector_;
    void init_post_ops_injector(
            const XReg &eltwise_table, const PReg &eltwise_mask);
    void apply_postops(int unroll, bool tail);
    void load_kernel_params();
    XReg prepare_xaddr_reg_with_src0_ptr(size_t offt = 0);
    XReg prepare_xaddr_reg_with_src1_ptr(size_t offt = 0);
    XReg prepare_xaddr_reg_with_dst_ptr(size_t offt = 0);
    void perform_op(
            const Vmm &v0, const Vmm &v1, const Vmm &s_src0, const Vmm &s_src1);
    void prepare_isa_kernel();
    void compute_bcast(bool tail);
    void load_src1(const Vmm &vreg_src1, const int offt, bool tail);
    void store(int unroll, bool tail);
    void compute_dst_body(int unroll, bool tail);
    void compute_dst(int unroll, bool tail);
    void forward();
    void forward_over_outer_dims();
    void generate() override;

    template <typename T>
    void compute_cmp_alg(const T &cmp_dst, const Vmm &cmp_src,
            const Vmm &cmp_src2, alg_kind_t alg);

    void push(const XReg &reg);
    void pop(const XReg &reg);

    jit_uni_binary_kernel_t(const binary_pd_t *pd, const jit_binary_conf_t conf,
            bool tail_kernel = false);
    ~jit_uni_binary_kernel_t() override = default;

    const io::saturation_conf_t<Vmm> io_sat_conf_;
    const io::tail_conf_t io_tail_conf_;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
