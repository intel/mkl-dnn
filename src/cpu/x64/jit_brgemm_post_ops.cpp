/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "cpu/x64/jit_brgemm_post_ops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

jit_brgemm_kernel_post_ops_base_t *jit_brgemm_kernel_post_ops_base_t::create(
        cpu_isa_t isa, const brgemm_desc_t &abrg,
        const primitive_attr_t &aattr) {
    if (utils::one_of(isa, avx2, avx2_vnni, avx2_vnni_2)) {
        return new jit_brgemm_kernel_post_ops_t<Xbyak::Ymm>(abrg, aattr);
    } else {
        return new jit_brgemm_kernel_post_ops_t<Xbyak::Zmm>(abrg, aattr);
    }
}

#define GET_OFF(field) offsetof(brgemm_kernel_post_ops_args_t, field)

template <typename Vmm>
dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::jit_brgemm_kernel_post_ops_t(const brgemm_desc_t &abrg,
        const primitive_attr_t &aattr)
    : jit_generator_t(jit_name(), abrg.isa_impl)
    , brg_(abrg)
    , attr_(aattr)
    , max_vregs_(isa_num_vregs(brg_.isa_impl))
    , with_binary_non_scalar_bcast_(brg_.with_binary
              && binary_injector::any_binary_postop_rhs_non_scalar_broadcast(
                      attr_.post_ops_, memory_desc_wrapper(brg_.dst_md()))) {
    bool has_f8_e5m2_binary_postops = false;
    bool has_f8_e4m3_binary_postops = false;
    if (brg_.with_binary) {
        const auto &post_ops = attr_.post_ops_;
        for (int i = 0; i < post_ops.len(); i++) {
            const auto &entry = post_ops.entry_[i];
            if (!entry.is_binary()) continue;
            has_f8_e5m2_binary_postops
                    = entry.binary.src1_desc.data_type == data_type::f8_e5m2
                    || has_f8_e5m2_binary_postops;
            has_f8_e4m3_binary_postops
                    = entry.binary.src1_desc.data_type == data_type::f8_e4m3
                    || has_f8_e4m3_binary_postops;
        }
    }

    if (brg_.is_bf16_emu)
        bf16_emu_ = utils::make_unique<bf16_emulation_t>(this, emu_reserv_1,
                emu_reserv_2, emu_reserv_3, emu_scratch, emu_reserv_4,
                emu_reserv_4);
    if (brg_.is_fp8 || has_f8_e5m2_binary_postops
            || has_f8_e4m3_binary_postops) {
        if (utils::one_of(data_type::f8_e5m2, brg_.dt_a, brg_.dt_b, brg_.dt_d)
                || has_f8_e5m2_binary_postops)
            f8_e5m2_cvt_ = utils::make_unique<fp8_conversion_e5m2_t>(this,
                    emu_reserv_1, emu_reserv_2, emu_reserv_3, emu_mask,
                    emu_scratch);
        if (utils::one_of(data_type::f8_e4m3, brg_.dt_a, brg_.dt_b, brg_.dt_d)
                || has_f8_e4m3_binary_postops)
            f8_e4m3_cvt_ = utils::make_unique<fp8_conversion_e4m3_t>(this,
                    emu_reserv_1, emu_reserv_2, emu_reserv_3, emu_reserv_4,
                    emu_reserv_5, emu_scratch);
    }

    if (brg_.beta != 0) {
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = true;
        static constexpr bool use_exact_tail_scalar_bcast = false;

        const binary_injector::rhs_arg_static_params_t rhs_sp {
                vmm_tmp(4).getIdx(), this->r14, this->r15, this->r13,
                preserve_gpr, preserve_vmm, GET_OFF(ptr_binary_post_ops_rhs),
                GET_OFF(dst_orig), memory_desc_wrapper(brg_.dst_md()),
                static_cast<int>(brg_.load_dim % brg_.ld_block), k_tail_mask,
                use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t bsp(this->param1,
                binary_injector::get_all_strategies_supported_by_injector(),
                rhs_sp, f8_e5m2_cvt_.get(), f8_e4m3_cvt_.get());

        const bool save_state = brg_.with_eltwise;
        const auto &reserved_eltwise_gpr = reg_reserved_eltwise;
        const auto reserved_eltwise_maskr = Xbyak::Opmask(1);

        const eltwise_injector::static_params_t esp {
                save_state, reserved_eltwise_gpr, reserved_eltwise_maskr};

        postops_injector_ = utils::make_unique<po_injector_t>(this,
                attr_.post_ops_, bsp, esp,
                /* inject_sum = */ brg_.with_sum);
    }

    inp_dt_ = brg_.dt_c;
    out_dt_ = brg_.dt_d;
    bia_dt_ = brg_.dt_bias;

    inp_typesize_ = brg_.typesize_C;
    out_typesize_ = brg_.typesize_D;
    bia_typesize_ = brg_.typesize_bias;
}

template <typename Vmm>
dim_t dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::zp_c_values_offset(int n,
        bool is_tail /*= false*/) const noexcept {
    if (brg_.zp_type_c == brgemm_broadcast_t::per_n) {
        return (is_tail) ? sizeof(int32_t) * brg_.ldb_tail
                         : sizeof(int32_t) * n * brg_.ld_block;
    }

    return 0;
}

template <typename Vmm>
dim_t dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::zp_comp_a_vpad_offset(int n, int m,
        bool is_tail /*= false*/) const noexcept {
    return (is_tail) ? sizeof(int32_t) * (brg_.ldb_tail + m * brg_.LDB)
                     : sizeof(int32_t) * (n * brg_.ld_block + m * brg_.LDB);
}

template <typename Vmm>
dim_t dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::mb_zp_comp_a_offset(int m_block) const noexcept {
    return sizeof(int32_t) * m_block * brg_.LDB;
}

template <typename Vmm>
dim_t dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::compensation_vpad_offset(int n, int m,
        bool is_tail /*= false*/) const noexcept {
    return (is_tail) ? sizeof(int32_t) * (brg_.ldb_tail + m * brg_.LDB)
                     : sizeof(int32_t) * (n * brg_.ld_block + m * brg_.LDB);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::cvt2ps(
        data_type_t type_in, const Vmm vmm_in, const Xbyak::Operand &op,
        int tail_size, bool store, Xbyak::Opmask ktail_mask,
        bool skip_cvt2ps /*= false*/) {
    const bool is_tail = op.isMEM()
            && tail_size != vreg_traits_t<Vmm>::vlen / sizeof(float)
            // The current kernel is written such that tail_size = 0 implies
            // no tail and full vmm must be processed.
            && tail_size > 0;

    if (IMPLICATION(is_tail, isa_has_masks(brg_.isa_impl))) {
        const Vmm vmm = maybe_mask(vmm_in, is_tail, store, ktail_mask);
        switch (type_in) {
            case data_type::f32:
            case data_type::s32: vmovups(vmm, op); break;
            case data_type::s8: vpmovsxbd(vmm, op); break;
            case data_type::u8: vpmovzxbd(vmm, op); break;
            case data_type::bf16:
                vpmovzxwd(vmm, op);
                vpslld(vmm, vmm, 16);
                break;
            case data_type::f16: vcvtph2ps(vmm, op); break;
            case data_type::f8_e5m2:
                f8_e5m2_cvt_->vcvt_f8_to_f32(vmm, op);
                break;
            case data_type::f8_e4m3:
                f8_e4m3_cvt_->vcvt_f8_to_f32(vmm, op);
                break;
            default: assert(!"unsupported data type");
        }
    } else {
        load_data(type_in, vmm_in, op.getAddress(), tail_size);
    }
    if (!skip_cvt2ps && types::is_integral_dt(type_in))
        uni_vcvtdq2ps(vmm_in, vmm_in);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<
        Vmm>::inject_attr_postops(int m_block, int n_block, int tail /*= 0*/) {
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;

    // Sum post-op requires binary parameters to be set.
    if (with_binary_non_scalar_bcast_ || brg_.with_sum) {
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const auto vmm_idx = vector(m, n, n_block).getIdx();
            const size_t aux_output_offset
                    = out_typesize_ * (m * brg_.LDD + n * brg_.ld_block);

            rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, aux_reg_out);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    vmm_idx, aux_output_offset);
            if (tail) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
        }
    }

    postops_injector_->compute_vector_range(
            0, m_block * n_block, rhs_arg_params);
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::apply_comp(
        int m_block, int n_block, int tail /*= 0*/) {
    auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
    const bool has_tail = tail > 0;
    if (brg_.zp_type_a != brgemm_broadcast_t::none) {
        auto vmm_zp_a_val = vmm_tmp(1);
        mov(reg_zp_a_val, ptr[rsp + reg_zp_a_val_offs_]);
        uni_vpbroadcastd(vmm_zp_a_val, reg_zp_a_val.cvt32());

        mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
        const auto vmm_zp_comp_a = vmm_tmp(0);
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const size_t zp_comp_offset
                    = sizeof(int32_t) * (n * brg_.ld_block + m * brg_.LDB);
            auto zp_comp_a_addr = is_superset(brg_.isa_impl, avx512_core)
                    ? EVEX_compress_addr(aux_reg_zp_a_comp, zp_comp_offset)
                    : ptr[aux_reg_zp_a_comp + zp_comp_offset];
            if (IMPLICATION(has_tail, isa_has_masks(brg_.isa_impl))) {
                auto vmm_zp_comp_a_masked
                        = maybe_mask(vmm_zp_comp_a, has_tail, false, k_mask);
                vmovups(vmm_zp_comp_a_masked, zp_comp_a_addr);
            } else {
                load_data(data_type::s32, vmm_zp_comp_a, zp_comp_a_addr, tail);
            }
            uni_vpmulld(vmm_zp_comp_a, vmm_zp_a_val, zp_comp_a_addr);

            auto vmm = vector(m, n, n_block);
            uni_vpaddd(vmm, vmm, vmm_zp_comp_a);
        }
    }

    if (brg_.req_s8s8_compensation) {
        mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
        const auto vmm_comp = vmm_tmp(0);
        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const size_t s8s8_comp_offset
                    = sizeof(int32_t) * (n * brg_.ld_block + m * brg_.LDB);

            auto comp_addr = is_superset(brg_.isa_impl, avx512_core)
                    ? EVEX_compress_addr(aux_reg_s8s8_comp, s8s8_comp_offset)
                    : ptr[aux_reg_s8s8_comp + s8s8_comp_offset];
            if (IMPLICATION(tail > 0, isa_has_masks(brg_.isa_impl))) {
                auto vmm_comp_masked
                        = maybe_mask(vmm_comp, tail > 0, false, k_mask);
                vmovups(vmm_comp_masked, comp_addr);
            } else
                load_data(data_type::s32, vmm_comp, comp_addr, tail);

            auto vmm = vector(m, n, n_block);
            uni_vpaddd(vmm, vmm, vmm_comp);
        }
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::maybe_apply_comp(
        int m_block, int n_block, int tail /*= 0*/) {
    Xbyak::Label label_apply_without_comp;
    mov(reg_apply_comp, ptr[rsp + reg_apply_comp_offs_]);
    cmp(reg_apply_comp, 0);
    je(label_apply_without_comp, T_NEAR);
    apply_comp(m_block, n_block, tail);
    L_aligned(label_apply_without_comp);

    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        uni_vcvtdq2ps(vector(m, n, n_block), vector(m, n, n_block));
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::apply_post_ops(
        int m_block, int n_block, int tail /*= 0*/) {
    const auto vector = [=](int m, int n) { return Vmm(m * n_block + n); };
    auto k_mask = (tail == 0) ? k_full_mask : k_tail_mask;
    const auto req_comp = brg_.is_int8 && brg_.beta != 0
            && (brg_.req_s8s8_compensation
                    || brg_.zp_type_a != brgemm_broadcast_t::none);

    // brg_.alpha == 0 means initialize registers, 1 means read from input
    // brg_.beta == 0 means skip postwork, 1 means do postwork
    // req_comp == true -> convert accumulated values to f32 after applying
    // compensation to avoid the loss of accuracy when converting s32 to f32
    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        if (brg_.alpha == 0) {
            // have to init vmm each time because vectors may have been
            // changed in the previous iterations
            uni_vpxor(vector(m, n), vector(m, n), vector(m, n));
        } else {
            auto inp_addr = ptr[aux_reg_in
                    + inp_typesize_ * (m * brg_.LDC + n * brg_.ld_block)];
            cvt2ps(inp_dt_, vector(m, n), inp_addr, tail, false, k_mask,
                    req_comp);
        }
    }

    if (req_comp) maybe_apply_comp(m_block, n_block, tail);

    const bool has_ptr_b_support = is_superset(brg_.isa_impl, avx512_core);
    if (brg_.beta != 0 && brg_.with_src_scales) {
        mov(aux_reg_src_scales, ptr[rsp + reg_src_scales_offs_]);
        auto vmm_src_scales = vmm_tmp(0);
        if (!has_ptr_b_support)
            vbroadcastss(vmm_src_scales, ptr[aux_reg_src_scales]);

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto vmm = vector(m, n);
            if (has_ptr_b_support) {
                vmulps(vmm, vmm, ptr_b[aux_reg_src_scales]);
            } else {
                vmulps(vmm, vmm, vmm_src_scales);
            }
        }
    }
    if (brg_.beta != 0 && brg_.with_wei_scales) {
        assert(brg_.dt_wei_scales == data_type::f32);

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            const auto addr = ptr[aux_reg_wei_scales
                    + brg_.is_per_n_wei_scales * sizeof(float)
                            * (n * brg_.ld_block)];
            const bool is_tail = tail > 0;
            const bool is_single_scale = brg_.is_single_wei_scale;

            const auto vmm = vector(m, n);
            const auto vmm_wei_scales = vmm_tmp(0);
            if (is_single_scale) {
                if (has_ptr_b_support) {
                    // Same isa has masks support.
                    const auto vmm_m = maybe_mask(vmm, is_tail, false, k_mask);
                    vmulps(vmm_m, vmm, ptr_b[aux_reg_wei_scales]);
                } else {
                    vbroadcastss(vmm_wei_scales, addr);
                    vmulps(vmm, vmm, vmm_wei_scales);
                }
            } else {
                if (IMPLICATION(is_tail, isa_has_masks(brg_.isa_impl))) {
                    const auto vmm_m = maybe_mask(vmm, is_tail, false, k_mask);
                    const auto vmm_wei_scales_masked = maybe_mask(
                            vmm_wei_scales, is_tail, false, k_mask);
                    vmovups(vmm_wei_scales_masked, addr);
                    vmulps(vmm_m, vmm, vmm_wei_scales);
                } else {
                    load_data(data_type::f32, vmm_wei_scales, addr, tail);
                    vmulps(vmm, vmm, vmm_wei_scales);
                }
            }
        }
    }

    if (brg_.beta != 0 && brg_.with_weights_scale_adjust) {
        // It's the only value that can be used for scale adjust so far.
        mov(aux_reg_scale_adjust, float2int(1.f / 0.5f));
        auto vmm_scale_adjust = vmm_tmp(0);
        auto xmm_scale_adjust = Xbyak::Xmm(vmm_scale_adjust.getIdx());
        uni_vmovq(xmm_scale_adjust, aux_reg_scale_adjust);
        uni_vbroadcastss(vmm_scale_adjust, xmm_scale_adjust);

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto vmm = vector(m, n);
            vmulps(vmm, vmm, vmm_scale_adjust);
        }
    }

    if (brg_.beta != 0 && brg_.with_bias) {
        for (int n = 0; n < n_block; n++) {
            auto vmm_bias = vmm_tmp(0);
            auto bias_addr
                    = ptr[aux_reg_bias + bia_typesize_ * (n * brg_.ld_block)];
            cvt2ps(bia_dt_, vmm_bias, bias_addr, tail, false, k_mask);
            for (int m = 0; m < m_block; m++) {
                vaddps(vector(m, n), vmm_bias);
            }
        }
    }

    if (postops_injector_) inject_attr_postops(m_block, n_block, tail);

    if (brg_.beta != 0 && brg_.with_dst_scales) {
        mov(aux_reg_dst_scales, ptr[rsp + reg_dst_scales_offs_]);
        auto vmm_dst_scales = vmm_tmp(0);
        if (!has_ptr_b_support)
            vbroadcastss(vmm_dst_scales, ptr[aux_reg_dst_scales]);

        for_(int m = 0; m < m_block; m++)
        for (int n = 0; n < n_block; n++) {
            auto vmm = vector(m, n);
            if (has_ptr_b_support) {
                vmulps(vmm, vmm, ptr_b[aux_reg_dst_scales]);
            } else {
                vmulps(vmm, vmm, vmm_dst_scales);
            }
        }
    }

    if (brg_.beta != 0 && brg_.zp_type_c != brgemm_broadcast_t::none) {
        mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
        auto vmm_zp_c = vmm_tmp(0);
        if (brg_.zp_type_c == brgemm_broadcast_t::per_tensor) {
            if (is_superset(brg_.isa_impl, avx512_core))
                vcvtdq2ps(vmm_zp_c,
                        EVEX_compress_addr(aux_reg_zp_c_values, 0, true));
            else {
                uni_vbroadcastss(vmm_zp_c, ptr[aux_reg_zp_c_values]);
                uni_vcvtdq2ps(vmm_zp_c, vmm_zp_c);
            }
        }
        for (int n = 0; n < n_block; n++) {
            if (brg_.zp_type_c == brgemm_broadcast_t::per_n) {
                const dim_t zp_c_off = zp_c_values_offset(n);
                auto zp_c_addr = is_superset(brg_.isa_impl, avx512_core)
                        ? EVEX_compress_addr(aux_reg_zp_c_values, zp_c_off)
                        : ptr[aux_reg_zp_c_values + zp_c_off];
                cvt2ps(data_type::s32, vmm_zp_c, zp_c_addr, tail, false,
                        k_mask);
            }
            for (int m = 0; m < m_block; m++) {
                const auto vmm = vector(m, n);
                uni_vaddps(vmm, vmm, vmm_zp_c);
            }
        }
    }

    const bool dt_requires_saturation = types::is_integral_dt(out_dt_);

    const reg64_t reg_tmp_gpr = reg_tmp;
    auto vmm_lbound = vmm_tmp(0);
    auto vmm_ubound = vmm_tmp(1);
    const bool use_sat_cvt
            = dt_requires_saturation && isa_has_sat_cvt(brg_.isa_impl, out_dt_);
    if (dt_requires_saturation) {
        init_saturate_f32(vmm_lbound, vmm_ubound, reg_tmp_gpr, data_type::f32,
                out_dt_, false, use_sat_cvt);
    }

    if (brg_.is_bf16_emu) bf16_emu_->init_vcvtneps2bf16();

    for_(int m = 0; m < m_block; m++)
    for (int n = 0; n < n_block; n++) {
        // incase of tail, stores are unconditionally masked, regardless
        // of `n`, implying n_block must be equal to `1`.
        assert(IMPLICATION(tail > 0, n_block == 1));
        auto vmm = vector(m, n);
        const size_t offset
                = out_typesize_ * (m * brg_.LDD + n * brg_.ld_block);
        const auto addr = ptr[aux_reg_out + offset];

        if (dt_requires_saturation) {
            saturate_cvt_f32(
                    vmm, vmm_lbound, vmm_ubound, out_dt_, false, use_sat_cvt);
        }
        if (use_sat_cvt) {
            auto xmm = Xbyak::Xmm(vmm.getIdx());
            auto r_xmm = maybe_mask(xmm, tail > 0, true, k_mask);
            assert(utils::one_of(out_dt_, data_type::s8, data_type::u8));
            auto vmm_perm = vmm_ubound;
            vpermb(vmm, vmm_perm, vmm);
            vmovdqu8(addr, r_xmm);
            continue;
        }

        if (is_superset(brg_.isa_impl, avx512_core)) {
            auto vmm_masked = maybe_mask(vmm, tail > 0, true, k_mask);
            Vmm_lower_t vmm_low = Vmm_lower_t(vmm.getIdx());
            Vmm_lower2_t vmm_low2 = Vmm_lower2_t(vmm_low.getIdx());
            auto vmm_low_masked = maybe_mask(vmm_low, tail > 0, true, k_mask);
            auto vmm_low2_masked = maybe_mask(vmm_low2, tail > 0, true, k_mask);
            switch (out_dt_) {
                case data_type::f32:
                case data_type::s32: uni_vmovups(addr, vmm_masked); break;
                case data_type::bf16:
                    if (brg_.is_bf16_emu) {
                        bf16_emu_->vcvtneps2bf16(vmm_low, vmm);
                        vmovdqu16(addr, vmm_low_masked);
                    } else {
                        vcvtneps2bf16(vmm_low, vmm);
                        vmovdqu16(addr, vmm_low_masked);
                    }
                    break;
                case data_type::f16:
                    vcvtps2ph(vmm_low, vmm, _op_mxcsr);
                    vmovdqu16(addr, vmm_low_masked);
                    break;
                case data_type::f8_e5m2:
                    f8_e5m2_cvt_->vcvt_f32_to_f8(vmm_low2, vmm);
                    vmovdqu8(addr, vmm_low2_masked);
                    break;
                case data_type::f8_e4m3:
                    f8_e4m3_cvt_->vcvt_f32_to_f8(vmm_low2, vmm);
                    vmovdqu8(addr, vmm_low2_masked);
                    break;
                case data_type::s8:
                    if (dt_requires_saturation
                            && isa_has_sat_cvt(brg_.isa_impl, out_dt_))
                        vpmovusdb(addr, vmm_masked);
                    else
                        vpmovsdb(addr, vmm_masked);
                    break;
                case data_type::u8: vpmovusdb(addr, vmm_masked); break;
                default: assert(!"unknown dst_dt");
            }
        } else {
            const int simd_w = vreg_traits_t<Vmm>::vlen / sizeof(float);
            const int nelems = tail > 0 ? tail : simd_w;
            store_data(out_dt_, vmm, aux_reg_out, offset, nelems);
        }
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::loop_by_N(
        int m_block, int nb2, int nb2_tail, int nb_tail) {
    if (brg_.alpha != 0) { mov(aux_reg_in, reg_in); }
    if (brg_.beta != 0) {
        if (brg_.with_bias) mov(aux_reg_bias, reg_bias);
        if (brg_.zp_type_c != brgemm_broadcast_t::none) {
            mov(aux_reg_zp_c_values, ptr[rsp + reg_zp_c_values_offs_]);
            mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
        }
        if (brg_.zp_type_a != brgemm_broadcast_t::none) {
            mov(aux_reg_zp_a_comp, ptr[rsp + reg_zp_a_comp_offs_]);
            mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
        }
        if (brg_.req_s8s8_compensation) {
            mov(aux_reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
            mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
        }
        if (brg_.with_wei_scales) mov(aux_reg_wei_scales, reg_wei_scales);
    }
    mov(aux_reg_out, reg_out);

    for (int n_loop_ = 0; n_loop_ < nb2; n_loop_++) {
        apply_post_ops(m_block, n_block2_);

        const auto oc_l_offset = n_block2_ * brg_.ld_block;

        add(aux_reg_out, out_typesize_ * oc_l_offset);
        if (brg_.alpha != 0) { add(aux_reg_in, inp_typesize_ * oc_l_offset); }
        if (brg_.beta != 0) {
            if (brg_.with_bias) add(aux_reg_bias, bia_typesize_ * oc_l_offset);
            if (brg_.zp_type_c != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
                add(aux_reg_zp_c_values, zp_c_values_offset(n_block2_));
                mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
            }
            if (brg_.zp_type_a != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                add(aux_reg_zp_a_comp,
                        static_cast<dim_t>(sizeof(int32_t)) * oc_l_offset);
                mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
            }
            if (brg_.req_s8s8_compensation) {
                mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                add(aux_reg_s8s8_comp,
                        static_cast<dim_t>(sizeof(int32_t)) * oc_l_offset);
                mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
            }
            if (brg_.with_wei_scales)
                add(aux_reg_wei_scales,
                        static_cast<dim_t>(
                                brg_.is_per_n_wei_scales * sizeof(float))
                                * oc_l_offset);
        }
    }
    if (nb2_tail > 0) {
        apply_post_ops(m_block, nb2_tail);
        const auto oc_l_offset = nb2_tail * brg_.ld_block;

        add(aux_reg_out, out_typesize_ * oc_l_offset);
        if (brg_.alpha != 0) { add(aux_reg_in, inp_typesize_ * oc_l_offset); }
        if (brg_.beta != 0) {
            if (brg_.with_bias) add(aux_reg_bias, bia_typesize_ * oc_l_offset);
            if (brg_.zp_type_c != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
                add(aux_reg_zp_c_values, zp_c_values_offset(nb2_tail));
                mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
            }
            if (brg_.zp_type_a != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                add(aux_reg_zp_a_comp,
                        static_cast<dim_t>(sizeof(int32_t)) * oc_l_offset);
                mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
            }
            if (brg_.req_s8s8_compensation) {
                mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                add(aux_reg_s8s8_comp,
                        static_cast<dim_t>(sizeof(int32_t)) * oc_l_offset);
                mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
            }
            if (brg_.with_wei_scales)
                add(aux_reg_wei_scales,
                        static_cast<dim_t>(
                                brg_.is_per_n_wei_scales * sizeof(float))
                                * oc_l_offset);
        }
    }
    if (nb_tail > 0) {
        apply_post_ops(m_block, 1, nb_tail);

        if (brg_.alpha != 0) { add(aux_reg_in, inp_typesize_ * (nb_tail)); }
        if (brg_.beta != 0) {
            if (brg_.with_bias) add(aux_reg_bias, bia_typesize_ * (nb_tail));
            if (brg_.zp_type_c != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_c_values, ptr[rsp + aux_reg_zp_c_values_offs_]);
                add(aux_reg_zp_c_values, zp_c_values_offset(1, nb_tail));
                mov(ptr[rsp + aux_reg_zp_c_values_offs_], aux_reg_zp_c_values);
            }
            if (brg_.zp_type_a != brgemm_broadcast_t::none) {
                mov(aux_reg_zp_a_comp, ptr[rsp + aux_reg_zp_a_comp_offs_]);
                add(aux_reg_zp_a_comp,
                        static_cast<dim_t>(sizeof(int32_t)) * nb_tail);
                mov(ptr[rsp + aux_reg_zp_a_comp_offs_], aux_reg_zp_a_comp);
            }
            if (brg_.req_s8s8_compensation) {
                mov(aux_reg_s8s8_comp, ptr[rsp + aux_reg_s8s8_comp_offs_]);
                add(aux_reg_s8s8_comp,
                        static_cast<dim_t>(sizeof(int32_t)) * nb_tail);
                mov(ptr[rsp + aux_reg_s8s8_comp_offs_], aux_reg_s8s8_comp);
            }
            if (brg_.with_wei_scales)
                add(aux_reg_wei_scales,
                        brg_.is_per_n_wei_scales * bia_typesize_ * (nb_tail));
        }
        add(aux_reg_out, out_typesize_ * (nb_tail));
    }
}

template <typename Vmm>
void dnnl::impl::cpu::x64::jit_brgemm_kernel_post_ops_t<Vmm>::generate() {
    preamble();

    sub(rsp, stack_space_needed_);

    const int nb = static_cast<int>(brg_.load_dim / brg_.ld_block);
    const int nb_tail = static_cast<int>(brg_.load_dim % brg_.ld_block);

    const int nb2 = nb / n_block2_;
    const int nb2_tail = nb % n_block2_;
    int n_block = (nb2 == 0) ? nstl::max(1, nb2_tail) : n_block2_;

    int m_max_regs = (brg_.is_bf16_emu ? 24
                    : (brg_.is_fp8_via_convert() || brg_.is_fp8_post_ops_emu())
                    ? 23
                    : max_vregs_ - 4);
    m_max_regs /= n_block;
    int m_block
            = static_cast<int>(nstl::min<dim_t>(brg_.bcast_dim, m_max_regs));

    const dim_t mb = brg_.bcast_dim / m_block;
    const int mb_tail = static_cast<int>(brg_.bcast_dim % m_block);

    if (isa_has_masks(brg_.isa_impl)) {
        const auto full_mask = size_t {0xffffffffffffffff};
        const auto tail_mask = size_t((1 << nb_tail) - 1);

        reg64_t reg_mask = reg_tmp;
        mov(reg_mask, full_mask);
        kmovq(k_full_mask, reg_mask);
        mov(reg_mask, tail_mask);
        kmovq(k_tail_mask, reg_mask);
    }

    if (brg_.alpha != 0) { mov(reg_in, ptr[param1 + GET_OFF(ptr_in)]); }
    if (brg_.beta != 0) {
        if (brg_.with_src_scales) {
            mov(reg_src_scales, ptr[param1 + GET_OFF(ptr_src_scales)]);
            mov(ptr[rsp + reg_src_scales_offs_], reg_src_scales);
        }
        if (brg_.with_wei_scales) {
            mov(reg_wei_scales, ptr[param1 + GET_OFF(ptr_wei_scales)]);
        }
        mov(reg_apply_comp, ptr[param1 + GET_OFF(apply_comp)]);
        mov(ptr[rsp + reg_apply_comp_offs_], reg_apply_comp);

        if (brg_.with_bias) mov(reg_bias, ptr[param1 + GET_OFF(ptr_bias)]);
        if (brg_.zp_type_c != brgemm_broadcast_t::none) {
            mov(reg_zp_c_values, ptr[param1 + GET_OFF(c_zp_values)]);
            mov(ptr[rsp + reg_zp_c_values_offs_], reg_zp_c_values);
        }
        if (brg_.zp_type_a != brgemm_broadcast_t::none) {
            mov(reg_zp_a_comp, ptr[param1 + GET_OFF(a_zp_compensation)]);
            mov(ptr[rsp + reg_zp_a_comp_offs_], reg_zp_a_comp);

            mov(reg_zp_a_val, ptr[param1 + GET_OFF(a_comp_val)]);
            mov(ptr[rsp + reg_zp_a_val_offs_], reg_zp_a_val);
        }
        if (brg_.req_s8s8_compensation) {
            mov(reg_s8s8_comp, ptr[param1 + GET_OFF(s8s8_compensation)]);
            mov(ptr[rsp + reg_s8s8_comp_offs_], reg_s8s8_comp);
        }
        if (brg_.with_dst_scales) {
            mov(reg_dst_scales, ptr[param1 + GET_OFF(ptr_dst_scales)]);
            mov(ptr[rsp + reg_dst_scales_offs_], reg_dst_scales);
        }
    }
    mov(reg_out, ptr[param1 + GET_OFF(ptr_out)]);

    for (dim_t mb_ = 0; mb_ < mb; mb_++) {
        loop_by_N(m_block, nb2, nb2_tail, nb_tail);

        if (brg_.alpha != 0) add(reg_in, inp_typesize_ * (m_block * brg_.LDC));
        if (brg_.beta != 0) {
            if (brg_.zp_type_a != brgemm_broadcast_t::none) {
                mov(reg_zp_a_comp, ptr[rsp + reg_zp_a_comp_offs_]);
                add(reg_zp_a_comp, mb_zp_comp_a_offset(m_block));
                mov(ptr[rsp + reg_zp_a_comp_offs_], reg_zp_a_comp);
            }
            if (brg_.req_s8s8_compensation) {
                mov(reg_s8s8_comp, ptr[rsp + reg_s8s8_comp_offs_]);
                add(reg_s8s8_comp, mb_compensation_offset(m_block));
                mov(ptr[rsp + reg_s8s8_comp_offs_], reg_s8s8_comp);
            }
        }
        add(reg_out, out_typesize_ * (m_block * brg_.LDD));
    }
    if (mb_tail > 0) loop_by_N(mb_tail, nb2, nb2_tail, nb_tail);

    add(rsp, stack_space_needed_);

    postamble();

    if (postops_injector_)
        postops_injector_->prepare_table(/* generate = */ true);
    if (brg_.is_fp8_via_convert()) {
        if (f8_e5m2_cvt_) f8_e5m2_cvt_->prepare_table();
        if (f8_e4m3_cvt_) f8_e4m3_cvt_->prepare_table();
    }
}

#undef GET_OFF

template struct jit_brgemm_kernel_post_ops_t<Xbyak::Ymm>;
template struct jit_brgemm_kernel_post_ops_t<Xbyak::Zmm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
