/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include <functional>

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_gemm_x8s8s32x_conv_zp_src_pad_comp.hpp"
#include "cpu/x64/jit_gemm_x8s8s32x_convolution_utils.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/jit_primitive_conf.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace gemm_x8s8s32x_convolution_utils {
using namespace dnnl::impl::cpu::gemm_x8s8s32x_convolution_utils;

struct jit_pp_ker_t : pp_ker_t, public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            gemm_x8s8s32x_convolution_utils::jit_pp_ker_t);

    jit_pp_ker_t(const convolution_pd_t *pd, const conv_gemm_conf_t &jcp);

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }
    void operator()(void *void_dst, const acc_data_t *acc, const char *bias,
            const float *scales, float dst_scale, float sum_scale,
            float signed_scale, int g, int /* mb */, size_t start, size_t end,
            const zero_point_call_params_t &zp,
            const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
            const exec_ctx_t & /* ctx */, const memory_desc_t & /* dst_md */,
            const single_gemm_conv_chunk_desc_t &) const override;

private:
    void apply_postops(
            const Xbyak::Reg64 &reg_dst, const int idx, const dim_t offset);
    void generate() override;
    void append_zp_src_comp(dim_t offset, int idx, bool apply_mask);
    void load_as_f32(const Xbyak::Zmm &dst, const Xbyak::Opmask &mask,
            const Xbyak::Address &src_addr, const data_type_t &src_dt);

    int vreg_dst_idx(const int idx) const noexcept;
    Xbyak::Zmm get_vreg_dst(int idx) const;
    Xbyak::Zmm get_vreg_bias(int idx) const;
    Xbyak::Zmm get_vreg_prev_dst(int idx) const;
    Xbyak::Zmm get_vreg_zp_comp_src(int idx) const;
    Xbyak::Zmm get_masked_vreg_dst(int idx, bool apply_mask) const;
    Xbyak::Zmm reserve_zmm();

    const Xbyak::Opmask &opmask_binary = k2;

    struct ker_args_t {
        char *dst;
        const acc_data_t *acc;
        const char *bias;
        const float *scales;
        float dst_scale;
        float sum_scale;
        float signed_scale;
        dim_t len;
        dim_t oc_offset;
        const int32_t *zp_src;
        const int32_t *zp_dst;
        const int32_t *zp_src_comp;
        const int32_t *zp_src_pad_comp;
        dim_t g_oc_offset_prologue;
        dim_t g_oc_offset;
        const void *post_ops_binary_rhs_arg_vec;
        const void *dst_orig;
        dim_t h;
        dim_t w;
        dim_t w_size;
        dim_t w_off;
        dim_t zp_src_pad_com_d_offset;
        bool should_apply_zp_src_pad_comp_d;
    };

    std::unique_ptr<injector::jit_uni_postops_injector_t<Xbyak::Zmm>>
            postops_injector_;

    int number_of_reserved_zmm_regs_;
    const dim_t bias_data_type_size_;
    const dim_t dst_data_type_size_;
    const bool saturation_needed_;

    const Xbyak::Reg64 &reg_param_ = rdi;
    const Xbyak::Reg64 &reg_tmp_ = rcx; // intentional for shifting purposes

    const Xbyak::Reg64 &reg_dst_ = rdx;
    const Xbyak::Reg64 &reg_acc_ = rax;
    const Xbyak::Reg64 &reg_bias_ = rbx;
    const Xbyak::Reg64 &reg_scales_ = rsi;
    const Xbyak::Reg64 &reg_len_ = r8;
    const Xbyak::Reg64 &reg_oc_offset_ = r9;
    const Xbyak::Reg64 &reg_rem_mask_short_ = r10;
    const Xbyak::Reg64 &reg_rem_mask_vlen_ = reg_rem_mask_short_;
    const Xbyak::Reg64 &reg_zp_pad_comp_temp_ = r10;
    const Xbyak::Reg64 &reg_zp_pad_comp_ = r11;
    const Xbyak::Reg8 &reg_should_apply_src_pad_comp_ = r13b;

    const Xbyak::Reg64 &reg_tmp_comp_
            = r12; // used to broadcast scalar values to vreg
    const Xbyak::Reg64 &reg_zp_src_comp_ = r14;

    const Xbyak::Zmm vreg_zero_;
    const Xbyak::Zmm vreg_scale_;
    const Xbyak::Zmm vreg_dst_scale_;
    const Xbyak::Zmm vreg_sum_scale_;
    const Xbyak::Zmm vreg_signed_scale_;
    const Xbyak::Zmm vreg_saturation_ubound_;
    const Xbyak::Zmm vreg_zp_dst_common_;

    const Xbyak::Opmask &kreg_rem_mask_short_ = k3;
    const Xbyak::Opmask &kreg_rem_mask_vlen_ = k4;

    static constexpr int def_unroll_ = 4;
    int zmm_step_;
    const int bias_step_factor_;
    const int sum_step_factor_;
    const int max_unroll_;

    std::unique_ptr<jit_gemm_x8s8s32x_zp_pad_comp_helper_t> zp_pad_comp_helper_;
};

jit_pp_ker_t::jit_pp_ker_t(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp)
    : pp_ker_t(pd, jcp)
    , jit_generator_t(jit_name(), avx512_core)
    , number_of_reserved_zmm_regs_(0)
    , bias_data_type_size_(jcp.bias_data_type != data_type::undef
                      ? types::data_type_size(jcp.bias_data_type)
                      : 0u)
    , dst_data_type_size_(types::data_type_size(jcp.dst_data_type))
    , saturation_needed_(utils::one_of(
              jcp_.dst_data_type, data_type::u8, data_type::s8, data_type::s32))
    , vreg_zero_((jcp_.with_eltwise || saturation_needed_) ? reserve_zmm()
                                                           : Xbyak::Zmm(0))
    , vreg_scale_(reserve_zmm())
    , vreg_dst_scale_(reserve_zmm())
    , vreg_sum_scale_(jcp_.with_sum ? reserve_zmm() : Xbyak::Zmm(0))
    , vreg_signed_scale_(jcp_.signed_input ? reserve_zmm() : Xbyak::Zmm(0))
    , vreg_saturation_ubound_(
              saturation_needed_ ? reserve_zmm() : Xbyak::Zmm(0))
    , vreg_zp_dst_common_(jcp_.zp.dst_exists ? reserve_zmm() : Xbyak::Zmm(0))
    , zmm_step_(1u)
    , bias_step_factor_(jcp_.with_bias ? zmm_step_++ : 0u)
    , sum_step_factor_(jcp_.with_sum ? zmm_step_++ : 0)
    , max_unroll_((cpu_isa_traits_t<avx512_core>::n_vregs
                          - number_of_reserved_zmm_regs_)
              / zmm_step_)
    , zp_pad_comp_helper_(jit_gemm_convolution_utils::padding_exists(jcp)
                              && jcp.zp.src_exists
                      ? utils::make_unique<
                                jit_gemm_x8s8s32x_zp_pad_comp_helper_t>(this,
                                jcp_, reg_zp_pad_comp_, reg_zp_pad_comp_temp_,
                                reg_should_apply_src_pad_comp_,
                                pd->src_md()->ndims)
                      : nullptr)

{

    if (jcp.with_eltwise || jcp.with_binary) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        static constexpr size_t helper_vmm_idx = 31;
        // tail_size = 1 just indicates that tailing is to be performed
        // actual tail value is held in opmask passed to injector
        static constexpr size_t tail_size = 1;
        static constexpr bool use_exact_tail_scalar_bcast = false;

#define PARAM_OFF(x) offsetof(ker_args_t, x)
        const rhs_arg_static_params_t rhs_arg_static_params {helper_vmm_idx,
                r13, r14, r15, preserve_gpr, preserve_vmm,
                PARAM_OFF(post_ops_binary_rhs_arg_vec), PARAM_OFF(dst_orig),
                memory_desc_wrapper(pd->dst_md()), tail_size, opmask_binary,
                use_exact_tail_scalar_bcast};
#undef PARAM_OFF

        const static_params_t static_params {reg_param_, rhs_arg_static_params};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<Xbyak::Zmm>>(
                // Sum is applied in the kernel's own store sequence.
                this, jcp_.post_ops, static_params, /* inject_sum = */ false);
    }
}

void jit_pp_ker_t::operator()(void *void_dst, const acc_data_t *acc,
        const char *bias, const float *scales, float dst_scale, float sum_scale,
        float signed_scale, int g, int /* mb */, size_t start, size_t end,
        const zero_point_call_params_t &zp,
        const void *post_ops_binary_rhs_arg_vec, const void *dst_orig,
        const exec_ctx_t & /* ctx */, const memory_desc_t & /* dst_md */,
        const single_gemm_conv_chunk_desc_t &chunk_desc) const {

    if (end <= start) return;

    char *dst = (char *)void_dst;

    const dim_t start_dim = start;
    const dim_t end_dim = end;
    ker_args_t args;
    const dim_t oc_offset = start_dim % jcp_.oc;
    const dim_t os_offset = start_dim / jcp_.oc;
    const dim_t dst_os_stride = jcp_.dst_os_stride;
    const dim_t scale_idx_mult = jcp_.scale_idx_mult;
    args.acc = acc + start_dim;
    args.dst = dst
            + (os_offset * dst_os_stride + oc_offset) * dst_data_type_size_;

    const dim_t g_oc_offset = g * jcp_.oc;
    const dim_t g_oc_offset_prologue = g_oc_offset + oc_offset;
    args.bias = bias + g_oc_offset_prologue * bias_data_type_size_;
    args.zp_src = zp.src
            ? zp.src + (jcp_.zp.src_is_common ? 0 : g_oc_offset_prologue)
            : nullptr;
    args.zp_src_comp
            = zp.src_comp ? zp.src_comp + g_oc_offset_prologue : nullptr;
    args.zp_dst = zp.dst;
    args.scales = scales + scale_idx_mult * g_oc_offset_prologue;
    args.dst_scale = dst_scale;
    args.sum_scale = sum_scale;
    args.signed_scale = signed_scale;
    args.len = end_dim - start_dim;
    args.oc_offset = oc_offset;

    args.g_oc_offset = g_oc_offset;
    args.g_oc_offset_prologue = g_oc_offset_prologue;

    args.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec;
    args.dst_orig = dst_orig;

    if (zp_pad_comp_helper_) {
        args.h = os_offset / chunk_desc.w_size_ + chunk_desc.h_off_;
        args.w = os_offset % chunk_desc.w_size_ + chunk_desc.w_off_;
        args.w_size = chunk_desc.w_size_ + chunk_desc.w_off_;
        args.w_off = chunk_desc.w_off_;
        args.zp_src_pad_comp = zp.src_pad_comp;
        const auto zp_src_pad_comp_d_dim_t
                = zp_pad_comp_helper_->calculate_zp_src_pad_com_d(
                        chunk_desc.d_off_);
        args.zp_src_pad_com_d_offset = zp_src_pad_comp_d_dim_t.offset;
        args.should_apply_zp_src_pad_comp_d
                = zp_src_pad_comp_d_dim_t.should_apply_pad_comp_d;
    }

    jit_generator_t::operator()(&args);
}

Xbyak::Zmm jit_pp_ker_t::reserve_zmm() {
    return Xbyak::Zmm(number_of_reserved_zmm_regs_++);
}

int jit_pp_ker_t::vreg_dst_idx(const int idx) const noexcept {
    return (number_of_reserved_zmm_regs_ + idx * zmm_step_);
}

Xbyak::Zmm jit_pp_ker_t::get_vreg_dst(int idx) const {
    return Xbyak::Zmm(vreg_dst_idx(idx));
}

Xbyak::Zmm jit_pp_ker_t::get_vreg_bias(int idx) const {
    return Xbyak::Zmm(vreg_dst_idx(idx) + bias_step_factor_);
}

Xbyak::Zmm jit_pp_ker_t::get_vreg_prev_dst(int idx) const {
    return Xbyak::Zmm(vreg_dst_idx(idx) + sum_step_factor_);
}

Xbyak::Zmm jit_pp_ker_t::get_masked_vreg_dst(int idx, bool apply_mask) const {
    auto vreg_dst = this->get_vreg_dst(idx);
    if (apply_mask)
        vreg_dst = vreg_dst | kreg_rem_mask_short_;
    else
        vreg_dst = vreg_dst | kreg_rem_mask_vlen_;
    return vreg_dst;
}

void jit_pp_ker_t::append_zp_src_comp(dim_t offset, int idx, bool apply_mask) {
    const auto vreg_dst_masked = get_masked_vreg_dst(idx, apply_mask);
    const auto vreg_dst = get_vreg_dst(idx);
    const dim_t zp_src_comp_offset
            = offset * static_cast<dim_t>(sizeof(int32_t));
    const auto zp_src_comp_addr = ptr[reg_zp_src_comp_ + zp_src_comp_offset];

    vpaddd(vreg_dst_masked, vreg_dst, zp_src_comp_addr);

    if (zp_pad_comp_helper_)
        zp_pad_comp_helper_->zp_src_comp_pad_operation(
                [&](const Xbyak::Reg64 &reg_zp_pad_comp) {
            vpaddd(vreg_dst_masked, vreg_dst,
                    ptr[reg_zp_pad_comp + zp_src_comp_offset]);
        });
}

void jit_pp_ker_t::apply_postops(
        const Xbyak::Reg64 &reg_dst, const int idx, const dim_t offset) {
#define PARAM_OFF(x) offsetof(ker_args_t, x)
    if (jcp_.with_eltwise || jcp_.with_binary) {
        if (jcp_.with_binary) {
            binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
            const auto vmm_idx = vreg_dst_idx(idx);

            rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_dst);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    vmm_idx, offset * dst_data_type_size_);
            rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);

            postops_injector_->compute_vector(
                    vreg_dst_idx(idx), rhs_arg_params);
        } else
            postops_injector_->compute_vector(vreg_dst_idx(idx));
    }
#undef PARAM_OFF
}

void jit_pp_ker_t::load_as_f32(const Xbyak::Zmm &dst,
        const Xbyak::Opmask &mask_reg, const Xbyak::Address &src_addr,
        const data_type_t &src_dt) {

    const auto dst_masked = dst | mask_reg;

    switch (src_dt) {
        case data_type::s8: vpmovsxbd(dst_masked, src_addr); break;
        case data_type::u8: vpmovzxbd(dst_masked, src_addr); break;
        case data_type::s32: vcvtdq2ps(dst_masked, src_addr); break;
        case data_type::f32: vmovups(dst_masked, src_addr); break;
        default: assert(!"unimplemented");
    }

    if (utils::one_of(src_dt, data_type::s8, data_type::u8))
        vcvtdq2ps(dst_masked, dst);
}

void jit_pp_ker_t::generate() {
    using namespace Xbyak;
    using namespace utils;

    int vlen = static_cast<int>(
            cpu_isa_traits_t<avx512_core>::vlen / sizeof(float));
    for (; vlen >= 1 && (jcp_.oc % vlen != 0); --vlen) {}
    const dim_t dst_os_stride = jcp_.dst_os_stride;

    preamble();

#ifdef _WIN32
    mov(reg_param_, rcx);
#endif

#define PARAM_OFF(x) offsetof(ker_args_t, x)
    mov(reg_dst_, ptr[reg_param_ + PARAM_OFF(dst)]);
    mov(reg_acc_, ptr[reg_param_ + PARAM_OFF(acc)]);
    mov(reg_bias_, ptr[reg_param_ + PARAM_OFF(bias)]);
    mov(reg_scales_, ptr[reg_param_ + PARAM_OFF(scales)]);
    mov(reg_len_, ptr[reg_param_ + PARAM_OFF(len)]);
    mov(reg_oc_offset_, ptr[reg_param_ + PARAM_OFF(oc_offset)]);

    if (jcp_.zp.src_exists) {
        mov(reg_zp_src_comp_, ptr[reg_param_ + PARAM_OFF(zp_src_comp)]);
        if (zp_pad_comp_helper_)
            zp_pad_comp_helper_->init(PARAM_OFF(w), PARAM_OFF(h),
                    PARAM_OFF(w_size), PARAM_OFF(w_off),
                    PARAM_OFF(zp_src_pad_comp), PARAM_OFF(g_oc_offset_prologue),
                    PARAM_OFF(g_oc_offset), PARAM_OFF(zp_src_pad_com_d_offset),
                    PARAM_OFF(should_apply_zp_src_pad_comp_d));
    }

    if (jcp_.zp.dst_exists) {
        mov(reg_tmp_, ptr[reg_param_ + PARAM_OFF(zp_dst)]);
        vcvtdq2ps(vreg_zp_dst_common_, ptr_b[reg_tmp_]);
    }

    if (jcp_.with_dst_scale)
        vbroadcastss(vreg_dst_scale_, ptr[reg_param_ + PARAM_OFF(dst_scale)]);
    if (jcp_.with_sum)
        vbroadcastss(vreg_sum_scale_, ptr[reg_param_ + PARAM_OFF(sum_scale)]);
    if (jcp_.signed_input)
        vbroadcastss(
                vreg_signed_scale_, ptr[reg_param_ + PARAM_OFF(signed_scale)]);
    if (jcp_.scale_idx_mult == 0) vbroadcastss(vreg_scale_, dword[reg_scales_]);
#undef PARAM_OFF

    mov(reg_rem_mask_vlen_, 1);
    shl(reg_rem_mask_vlen_, vlen);
    sub(reg_rem_mask_vlen_, 1);
    kmovq(kreg_rem_mask_vlen_, reg_rem_mask_vlen_);

    if (jcp_.with_eltwise) vxorps(vreg_zero_, vreg_zero_, vreg_zero_);
    if (saturation_needed_)
        init_saturate_f32(vreg_zero_, vreg_saturation_ubound_, reg_tmp_comp_,
                data_type::f32, jcp_.dst_data_type);

    // Load accumulated value, convert to float, apply sum (if any),
    // bias (if any), scaling, and relu (if any);
    // then convert to destination type and store
    const auto compute = [&](dim_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc_
                + offset * static_cast<dim_t>(sizeof(acc_data_t))];

        const auto &mask_reg
                = apply_mask ? kreg_rem_mask_short_ : kreg_rem_mask_vlen_;

        if (jcp_.scale_idx_mult > 0) {
            assert(jcp_.scale_idx_mult == 1);
            const auto scale_addr = ptr[reg_scales_
                    + offset * static_cast<dim_t>(sizeof(float))];
            auto vreg_scale = vreg_scale_;
            vreg_scale = vreg_scale | mask_reg;
            vmovups(vreg_scale, scale_addr);
        }

        if (jcp_.with_binary) kmovq(opmask_binary, mask_reg);

        const auto vreg_dst_masked = get_masked_vreg_dst(idx, apply_mask);
        const auto vreg_dst = get_vreg_dst(idx);
        if (jcp_.zp.src_exists) {
            vmovups(vreg_dst_masked, acc_addr);
            append_zp_src_comp(offset, idx, apply_mask);
            vcvtdq2ps(vreg_dst_masked, vreg_dst);
        } else {
            vcvtdq2ps(vreg_dst_masked, acc_addr);
        }

        if (jcp_.signed_input)
            vmulps(vreg_dst_masked, vreg_dst, vreg_signed_scale_);

        vmulps(vreg_dst_masked, vreg_dst, vreg_scale_);

        if (jcp_.with_bias) {
            const auto bias_addr
                    = ptr[reg_bias_ + offset * bias_data_type_size_];
            const auto vreg_bias = get_vreg_bias(idx);
            load_as_f32(vreg_bias, mask_reg, bias_addr, jcp_.bias_data_type);
            vaddps(vreg_dst_masked, vreg_dst, vreg_bias);
        }

        const auto dst_addr = ptr[reg_dst_ + offset * dst_data_type_size_];

        if (jcp_.with_sum) {
            const auto vreg_prev_dst = get_vreg_prev_dst(idx);
            load_as_f32(vreg_prev_dst, mask_reg, dst_addr, jcp_.sum_data_type);
            vfmadd231ps(vreg_dst_masked, vreg_prev_dst, vreg_sum_scale_);
        }

        apply_postops(reg_dst_, idx, offset);

        if (jcp_.with_dst_scale) {
            vmulps(vreg_dst_masked, vreg_dst, vreg_dst_scale_);
        }

        if (jcp_.zp.dst_exists) {
            vaddps(vreg_dst_masked, vreg_dst, vreg_zp_dst_common_);
        }

        if (saturation_needed_) {
            saturate_cvt_f32(get_vreg_dst(idx), vreg_zero_,
                    vreg_saturation_ubound_, jcp_.dst_data_type);
        }

        switch (jcp_.dst_data_type) {
            case data_type::s8: vpmovsdb(dst_addr, vreg_dst_masked); break;
            case data_type::u8: vpmovusdb(dst_addr, vreg_dst_masked); break;
            case data_type::f32:
            case data_type::s32: vmovups(dst_addr, vreg_dst_masked); break;
            default: assert(!"unimplemented");
        }
    };

    // Advance all pointers by an immediate
    const auto advance_ptrs_imm
            = [&](const dim_t offset, const dim_t binary_offset) {
        add(reg_dst_, offset * dst_data_type_size_);
        add(reg_acc_, offset * static_cast<dim_t>(sizeof(acc_data_t)));
        if (jcp_.scale_idx_mult) {
            assert(jcp_.scale_idx_mult == 1);
            add(reg_scales_, offset * static_cast<dim_t>(sizeof(float)));
        }
        if (jcp_.with_bias) add(reg_bias_, offset * bias_data_type_size_);
        if (jcp_.zp.src_exists) {
            add(reg_zp_src_comp_, offset * static_cast<dim_t>(sizeof(int32_t)));

            if (zp_pad_comp_helper_) {
                zp_pad_comp_helper_->zp_src_comp_pad_operation(
                        [&](const Xbyak::Reg64 &reg_zp_pad_comp) {
                    add(reg_zp_pad_comp,
                            offset * static_cast<dim_t>(sizeof(int32_t)));
                });
            }
        }
    };

    // Advance all pointers by a value stored in a register
    const auto advance_ptrs_reg
            = [&](const Reg64 offset, const Reg64 binary_offset) {
        lea(reg_dst_,
                ptr[reg_dst_ + offset * static_cast<int>(dst_data_type_size_)]);
        lea(reg_acc_, ptr[reg_acc_ + offset * sizeof(acc_data_t)]);
        if (jcp_.scale_idx_mult) {
            assert(jcp_.scale_idx_mult == 1);
            lea(reg_scales_, ptr[reg_scales_ + offset * sizeof(float)]);
        }
        if (jcp_.with_bias)
            lea(reg_bias_,
                    ptr[reg_bias_
                            + offset * static_cast<int>(bias_data_type_size_)]);

        if (jcp_.zp.src_exists) {
            lea(reg_zp_src_comp_,
                    ptr[reg_zp_src_comp_ + offset * sizeof(int32_t)]);

            if (zp_pad_comp_helper_)
                zp_pad_comp_helper_->zp_src_comp_pad_operation(
                        [&](const Xbyak::Reg64 &reg_zp_pad_comp) {
                    lea(reg_zp_pad_comp,
                            ptr[reg_zp_pad_comp + offset * sizeof(int32_t)]);
                });
        }
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    const auto rewind_ptrs = [&]() {
        if (jcp_.with_bias) sub(reg_bias_, jcp_.oc * bias_data_type_size_);
        if (jcp_.zp.src_exists) {
            const dim_t offset = jcp_.oc * static_cast<dim_t>(sizeof(int32_t));
            sub(reg_zp_src_comp_, offset);
            if (zp_pad_comp_helper_)
                zp_pad_comp_helper_->load_next_point_zp_src_comp_pad_addr();
        }
        if (jcp_.scale_idx_mult) {
            assert(jcp_.scale_idx_mult == 1);
            sub(reg_scales_, jcp_.oc * static_cast<dim_t>(sizeof(float)));
        }
        add(reg_dst_, (dst_os_stride - jcp_.oc) * dst_data_type_size_);
    };

    //                    <--------- OC --------------->
    //
    // ^  ................+..............+-------------+.......................
    // |  .               : not accessed |Prologue loop|                      .
    // |  .               +--------------+-------------+                      .
    //    .               |                            |                      .
    // O  .               |  Main loop (unrolled)      |                      .
    // S  .               |                            |                      .
    //    .               +--------------+-------------+                      .
    // |  .               | Epilogue loop|not accessed :                      .
    // v  ................+--------------+.............+.......................

    Label prologue_end;
    cmp(reg_oc_offset_, 0);
    je(prologue_end, T_NEAR);

    // Prologue loop
    {
        mov(reg_tmp_, jcp_.oc);
        sub(reg_tmp_, reg_oc_offset_);
        cmp(reg_tmp_, reg_len_);
        cmovg(reg_tmp_, reg_len_);
        sub(reg_len_, reg_tmp_);

        Label prologue_loop, prologue_loop_tail, prologue_loop_end;
        cmp(reg_tmp_, vlen);
        jle(prologue_loop_tail, T_NEAR);
        L(prologue_loop);
        {
            compute(0, max_unroll_ - 1, false);
            advance_ptrs_imm(vlen, vlen);
            sub(reg_tmp_, vlen);
            cmp(reg_tmp_, vlen);
            jge(prologue_loop, T_NEAR);
        }

        L(prologue_loop_tail);
        mov(reg_rem_mask_short_, 1);
        // cl == reg_tmp_ because reg_tmp_ <= vlen here
        shl(reg_rem_mask_short_, cl);
        sub(reg_rem_mask_short_, 1);
        jz(prologue_loop_end, T_NEAR);

        kmovq(kreg_rem_mask_short_, reg_rem_mask_short_);
        compute(0, max_unroll_ - 1, true);
        advance_ptrs_reg(reg_tmp_, reg_tmp_);

        L(prologue_loop_end);
        rewind_ptrs();
    }
    L(prologue_end);

    // Main loop
    Label main_loop_end;
    {
        cmp(reg_len_, jcp_.oc);
        jle(main_loop_end, T_NEAR);

        Label main_loop;
        L(main_loop);
        {
            dim_t OC_loop, OC_tail;
            if (jcp_.oc < max_unroll_ * vlen) {
                // Fully unroll small loops
                OC_loop = 0;
                OC_tail = jcp_.oc;
            } else {
                OC_loop = vlen * def_unroll_;
                OC_tail = jcp_.oc % OC_loop;
            }

            assert(!!OC_loop || !!OC_tail);

            const dim_t vlen_tail = OC_tail % vlen;
            if (vlen_tail) {
                const uint32_t tail_mask = (uint32_t(1) << vlen_tail) - 1;
                mov(reg_tmp_, tail_mask);
                kmovq(kreg_rem_mask_short_, reg_tmp_);
            }

            if (OC_loop) {
                mov(reg_tmp_, rnd_dn(jcp_.oc, OC_loop));
                Label oc_loop;
                L(oc_loop);
                {
                    for (dim_t offset = 0; offset < OC_loop; offset += vlen)
                        compute(offset, static_cast<int>(offset / vlen), false);
                    advance_ptrs_imm(OC_loop, vlen);
                    sub(reg_tmp_, OC_loop);
                    jnz(oc_loop);
                }
            }

            if (OC_tail) {
                for (dim_t offset = 0; offset < OC_tail; offset += vlen) {
                    bool use_mask = (offset + vlen) > OC_tail;
                    compute(offset, static_cast<int>(offset / vlen), use_mask);
                }
                const dim_t oc_tail_rem = OC_tail % vlen;
                const dim_t binary_offset = oc_tail_rem ? oc_tail_rem : vlen;
                advance_ptrs_imm(OC_tail, binary_offset);
            }

            rewind_ptrs();
            sub(reg_len_, jcp_.oc);
            cmp(reg_len_, jcp_.oc);
            jge(main_loop, T_NEAR);
        }
    }
    L(main_loop_end);

    // Epilogue loop
    Label epilogue_end;
    {
        cmp(reg_len_, 0);
        je(epilogue_end, T_NEAR);

        Label epilogue_loop, epilogue_loop_tail;
        cmp(reg_len_, vlen);
        jle(epilogue_loop_tail, T_NEAR);
        L(epilogue_loop);
        {
            compute(0, 0, false);
            sub(reg_len_, vlen);
            advance_ptrs_imm(vlen, vlen);
            cmp(reg_len_, vlen);
            jge(epilogue_loop, T_NEAR);
        }

        L(epilogue_loop_tail);
        mov(reg_tmp_,
                reg_len_); // reg_tmp_ is rcx, and we need cl for the shift
        mov(reg_rem_mask_short_, 1);
        shl(reg_rem_mask_short_, cl); // reg_tmp_ == rcx and reg_tail < vlen
        sub(reg_rem_mask_short_, 1);
        jz(epilogue_end, T_NEAR);
        kmovq(kreg_rem_mask_short_, reg_rem_mask_short_);
        compute(0, 0, true);
    }

    L(epilogue_end);

    if (zp_pad_comp_helper_) zp_pad_comp_helper_->fin();

    postamble();

    if (jcp_.with_eltwise)
        postops_injector_->prepare_table(/* generate = */ true);
}

bool mayiuse_jit_pp_kernel(data_type_t dst_dt) noexcept {
    const auto is_bf16_dst_dt = dst_dt == data_type::bf16;
    return mayiuse(avx512_core) && !is_bf16_dst_dt;
}

pp_ker_t *jit_pp_ker_create(
        const convolution_pd_t *pd, const conv_gemm_conf_t &jcp) {
    return mayiuse_jit_pp_kernel(pd->dst_md()->data_type)
            ? new jit_pp_ker_t(pd, jcp)
            : nullptr;
}

bool post_ops_ok(const post_ops_t &post_ops, const memory_desc_wrapper *dst_d) {
    using namespace x64::injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = false;
    return mayiuse_jit_pp_kernel(dst_d->data_type())
            && dnnl::impl::cpu::x64::injector::post_ops_ok(
                    {avx512_core, {binary, eltwise, sum}, post_ops, dst_d,
                            sum_at_pos_0_only, sum_requires_scale_one});
}

} // namespace gemm_x8s8s32x_convolution_utils
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
