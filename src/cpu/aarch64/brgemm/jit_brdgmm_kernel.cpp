/*******************************************************************************
* Copyright 2021 Intel Corporation
* Copyright 2024-2026 FUJITSU LIMITED
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

#include "common/c_types_map.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/brgemm/brgemm_types.hpp"
#include "cpu/aarch64/brgemm/jit_brdgmm_kernel.hpp"
#include "cpu/aarch64/cpu_barrier.hpp"
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"

#define GET_OFF(field) offsetof(brgemm_kernel_params_t, field)
#define GET_OFF_BATCH_ELEMENT(field) offsetof(brgemm_batch_element_t, field)

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::utils;

jit_brdgmm_kernel_base_t::jit_brdgmm_kernel_base_t(const brgemm_desc_t &abrd)
    : jit_generator_t(nullptr, MAX_CODE_SIZE, true, sve_512)
    , brg(abrd)
    , simd_w_(cpu_isa_traits<sve_512>::vlen / brg.typesize_C) {

    if (brg.with_eltwise || brg.with_binary || brg.with_sum) {
        static constexpr bool use_exact_tail_scalar_bcast = false;
        const auto dst_md_wrapper = memory_desc_wrapper(brg.dst_md);
        const size_t tail = tail_length();

        static const bcast_set_t enabled_bcast_strategy
                = {broadcasting_strategy_t::scalar,
                        broadcasting_strategy_t::per_oc,
                        broadcasting_strategy_t::no_broadcast};
        const binary_injector::rhs_arg_static_params_t rhs_sp {
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(data_C_ptr_),
                dst_md_wrapper, tail, k_mask, use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t bsp {
                this->param1, enabled_bcast_strategy, rhs_sp};

        postops_injector_ = utils::make_unique<po_injector_t>(
                this, reg_pool(), brg.attr->post_ops_, bsp, X_TMP_0, PReg(1));

        with_binary_non_scalar_bcast_
                = binary_injector::any_binary_postop_rhs_non_scalar_broadcast(
                        brg.attr->post_ops_, dst_md_wrapper);
    }
}

void jit_brdgmm_kernel_base_t::read_params() {
    Label label_done;

    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(BS), X_TMP_0);
    ldr(reg_BS, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_C), X_TMP_0);
    ldr(reg_aux_C, ptr(X_DEFAULT_ADDR));
    add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_D), X_TMP_0);
    ldr(reg_aux_D, ptr(X_DEFAULT_ADDR));

    if (brg.type == brgemm_offs) {
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_A), X_TMP_0);
        ldr(reg_A, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_B), X_TMP_0);
        ldr(reg_B, ptr(X_DEFAULT_ADDR));
    } else if (brg.type == brgemm_strd) {
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_A), X_TMP_0);
        ldr(reg_aux1_A, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_B), X_TMP_0);
        ldr(reg_aux1_B, ptr(X_DEFAULT_ADDR));
        if (brg.brgattr.max_bs > 1) {
            add_imm(X_DEFAULT_ADDR, sp, reg_A_offs_, X_TMP_0);
            str(reg_aux1_A, ptr(X_DEFAULT_ADDR));
            add_imm(X_DEFAULT_ADDR, sp, reg_B_offs_, X_TMP_0);
            str(reg_aux1_B, ptr(X_DEFAULT_ADDR));
        }
    }

    if (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad()) {
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(batch), X_TMP_0);
        ldr(reg_aux_batch_addr, ptr(X_DEFAULT_ADDR));
        if (brg.brgattr.max_bs > 1) {
            add_imm(X_DEFAULT_ADDR, sp, reg_batch0_addr_offs_, X_TMP_0);
            str(reg_aux_batch_addr, ptr(X_DEFAULT_ADDR));
        }
    }

    if (brg.with_bias) {
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_bias), X_TMP_0);
        ldr(reg_tmp, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, sp, reg_bias_offs_, X_TMP_0);
        str(reg_tmp, ptr(X_DEFAULT_ADDR));
    }

    if (brg.with_scales) {
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_scales), X_TMP_0);
        ldr(reg_tmp, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, sp, reg_scales_offs_, X_TMP_0);
        str(reg_tmp, ptr(X_DEFAULT_ADDR));
    }

    if (brg.with_dst_scales) {
        add_imm(X_DEFAULT_ADDR, param1, GET_OFF(ptr_dst_scales), X_TMP_0);
        ldr(reg_tmp, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, sp, reg_dst_scales_offs_, X_TMP_0);
        str(reg_tmp, ptr(X_DEFAULT_ADDR));
    }

    if (brg.with_binary) {
        add_imm(X_DEFAULT_ADDR, sp, abi_param1_offs_, X_TMP_0);
        str(param1, ptr(X_DEFAULT_ADDR));
    }
}

void jit_brdgmm_kernel_base_t::load_accumulators(int m_blocks, int n_blocks) {
    const int v_substep = vnni_substep();
    for_(int v = 0; v < v_substep; ++v)
    for_(int m = 0; m < m_blocks; ++m)
    for (int n = 0; n < n_blocks; ++n) {
        auto vmm = accm(m_blocks, n_blocks, m, n, v);
        eor(vmm.d, vmm.d, vmm.d);
    }
}

void jit_brdgmm_kernel_base_t::restore_A_B_matrices() {
    if (brg.brgattr.max_bs > 1
            && (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad())) {
        add_imm(X_DEFAULT_ADDR, sp, reg_batch0_addr_offs_, X_TMP_0);
        ldr(reg_aux_batch_addr, ptr(X_DEFAULT_ADDR));
    }

    if (brg.type == brgemm_strd && brg.brgattr.max_bs > 1) {
        add_imm(X_DEFAULT_ADDR, sp, reg_A_offs_, X_TMP_0);
        ldr(reg_aux1_A, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, sp, reg_B_offs_, X_TMP_0);
        ldr(reg_aux1_B, ptr(X_DEFAULT_ADDR));
    }
}

void jit_brdgmm_kernel_base_t::set_A_B_matrices() {

    if (brg.type == brgemm_addr) {
        add_imm(X_DEFAULT_ADDR, reg_aux_batch_addr,
                GET_OFF_BATCH_ELEMENT(ptr.A), X_TMP_0);
        ldr(reg_aux_A, ptr(X_DEFAULT_ADDR));
        add_imm(X_DEFAULT_ADDR, reg_aux_batch_addr,
                GET_OFF_BATCH_ELEMENT(ptr.B), X_TMP_0);
        ldr(reg_aux_B, ptr(X_DEFAULT_ADDR));
    } else if (brg.type == brgemm_offs) {
        mov(reg_aux_A, reg_A);
        mov(reg_aux_B, reg_B);
        add_imm(X_DEFAULT_ADDR, reg_aux_batch_addr,
                GET_OFF_BATCH_ELEMENT(offset.A), X_TMP_0);
        ldr(X_TMP_1, ptr(X_DEFAULT_ADDR));
        add(reg_aux_A, reg_aux_A, X_TMP_1);
        add_imm(X_DEFAULT_ADDR, reg_aux_batch_addr,
                GET_OFF_BATCH_ELEMENT(offset.B), X_TMP_0);
        ldr(X_TMP_1, ptr(X_DEFAULT_ADDR));
        add(reg_aux_B, reg_aux_B, X_TMP_1);
    } else if (brg.type == brgemm_strd) {
        mov(reg_aux_A, reg_aux1_A);
        mov(reg_aux_B, reg_aux1_B);
        if (brg.brgattr.max_bs > 1) {
            add_imm(reg_aux1_A, reg_aux1_A, brg.stride_a, X_TMP_0);
            add_imm(reg_aux1_B, reg_aux1_B, brg.stride_b, X_TMP_0);
        }
    }

    add(reg_aux_A, reg_aux_A, reg_a_offset);
    mov_imm(X_TMP_1, brg.typesize_B);
    mul(X_TMP_1, X_TMP_1, reg_aux_N);
    add(reg_aux_B, reg_aux_B, X_TMP_1);
}

void jit_brdgmm_kernel_base_t::advance_A_B_matrices() {
    if (brg.brgattr.max_bs > 1
            && (one_of(brg.type, brgemm_addr, brgemm_offs) || has_vpad()))
        add_imm(reg_aux_batch_addr, reg_aux_batch_addr,
                sizeof(brgemm_batch_element_t), X_TMP_0);
}

void jit_brdgmm_kernel_base_t::cvt2ps(data_type_t type_in, const ZReg vmm_in,
        const AdrNoOfs &op, bool mask_flag, bool store) {
    auto mask = mask_flag ? k_mask : P_ALL_ONE;
    switch (type_in) {
        case data_type::f32:
        case data_type::s32:
            if (mask_flag && store) {
                ld1w(vmm_tmp(0).s, mask / T_z, op);
                mov(vmm_in.s, mask / T_m, vmm_tmp(0).s);
            } else
                ld1w(vmm_in.s, mask / T_z, op);
            break;
        case data_type::bf16:
            if (mask_flag && store) {
                ld1h(vmm_tmp(0).s, mask / T_z, op);
                lsl(vmm_tmp(0).s, vmm_tmp(0).s, 16);
                mov(vmm_in.s, mask / T_m, vmm_tmp(0).s);
            } else {
                ld1h(vmm_in.s, mask / T_z, op);
                lsl(vmm_in.s, vmm_in.s, 16);
            }
            break;
        case data_type::s8:
            if (mask_flag && store) {
                ld1sb(vmm_tmp(0).s, mask / T_z, op);
                mov(vmm_in.s, mask / T_m, vmm_tmp(0).s);
            } else {
                ld1sb(vmm_in.s, mask / T_z, op);
            }
            break;
        case data_type::u8:
            if (mask_flag && store) {
                ld1b(vmm_tmp(0).s, mask / T_z, op);
                mov(vmm_in.s, mask / T_m, vmm_tmp(0).s);
            } else {
                ld1b(vmm_in.s, mask / T_z, op);
            }
            break;

        default: assert(!"unsupported data type");
    }
    if (types::is_integral_dt(type_in)) {
        scvtf(vmm_in.s, P_ALL_ONE / T_z, vmm_in.s);
    }
}

void jit_brdgmm_kernel_base_t::apply_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    injector_utils::vmm_index_set_t vmm_idxs_param;
    const int v_substep = vnni_substep();

    // collect vmm_idx's to apply post ops.
    for_(int v_i = 0; v_i < v_substep; ++v_i)
    for_(int m_i = 0; m_i < m_blocks; ++m_i)
    for (int n_i = 0; n_i < n_blocks; ++n_i) {
        if (get_substep_simd(n_i, v_i, has_n_tail) <= 0) continue;
        const auto vmm_idx = accm(m_blocks, n_blocks, m_i, n_i, v_i).getIdx();
        vmm_idxs_param.insert(vmm_idx);
    }

    if (brg.with_binary) {
        add_imm(X_DEFAULT_ADDR, sp, abi_param1_offs_, X_TMP_0);
        ldr(reg_binary_params, ptr(X_DEFAULT_ADDR));

        if (with_binary_non_scalar_bcast_) {

            for_(int v_i = 0; v_i < v_substep; ++v_i)
            for_(int m_i = 0; m_i < m_blocks; m_i++)
            for (int n_i = 0; n_i < n_blocks; n_i++) {
                const int substep_simd = get_substep_simd(n_i, v_i, has_n_tail);
                if (substep_simd <= 0) continue;
                const auto vmm_idx
                        = accm(m_blocks, n_blocks, m_i, n_i, v_i).getIdx();
                rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_aux_D);
                rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                        vmm_idx, D_offset(m_i, n_i, v_i));

                if (n_i + 1 == n_blocks && has_n_tail && substep_simd < simd_w_)
                    rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
            }
        }
    }

    const auto sum_injector
            = [&](const injector_utils::vmm_index_set_t &vmm_idxs) {
        const float *p_sum_scale = &brg.sum_scale;
        const int32_t *p_sum_zp = &brg.sum_zp;
        const bool p_sum_scale_reg_set = *p_sum_scale != 1.f;
        const bool p_sum_zp_reg_set = *p_sum_zp != 0;

        const injector_utils::conditional_register_preserve_guard_t<sve>
                register_guard_sum_scale(
                        (with_binary_non_scalar_bcast_) && p_sum_scale_reg_set,
                        this, {reg_ptr_sum_scale});
        const injector_utils::conditional_register_preserve_guard_t<sve>
                register_guard_sum_zp(p_sum_zp_reg_set, this, {reg_ptr_sum_zp});

        auto vmm_sum_zp = vmm_tmp(0);
        auto vmm_sum_scale = vmm_tmp(1);

        if (p_sum_scale_reg_set) {
            mov_imm(reg_ptr_sum_scale, reinterpret_cast<size_t>(p_sum_scale));
            ld1rw(vmm_sum_scale.s, P_ALL_ONE / T_z, ptr(reg_ptr_sum_scale));
        }

        if (p_sum_zp_reg_set) {
            mov_imm(reg_ptr_sum_zp, reinterpret_cast<size_t>(p_sum_zp));
            ld1rw(vmm_sum_zp.s, P_ALL_ONE / T_z, ptr(reg_ptr_sum_zp));
            scvtf(vmm_sum_zp.s, P_ALL_ONE / T_z, vmm_sum_zp.s);
        }

        for_(int m_i = 0; m_i < m_blocks; m_i++)
        for_(int n_i = 0; n_i < n_blocks; n_i++)
        for (int v_i = 0; v_i < v_substep; v_i++) {
            const int substep_simd = get_substep_simd(n_i, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const auto vmm = accm(m_blocks, n_blocks, m_i, n_i, v_i);
            if (vmm_idxs.count(vmm.getIdx()) == 0) continue;
            add_imm(X_DEFAULT_ADDR, reg_aux_D, D_offset(m_i, n_i, v_i),
                    X_TMP_0);
            const auto addr = ptr(X_DEFAULT_ADDR);
            const auto vmm_prev_dst = vmm_tmp(2);
            cvt2ps(brg.sum_dt, vmm_prev_dst, addr, substep_simd != simd_w_,
                    false);
            if (p_sum_zp_reg_set)
                fsub(vmm_prev_dst.s, vmm_prev_dst.s, vmm_sum_zp.s);
            if (!p_sum_scale_reg_set)
                fadd(vmm.s, vmm.s, vmm_prev_dst.s);
            else {
                fmla(vmm.s, P_ALL_ONE / T_z, vmm_prev_dst.s, vmm_sum_scale.s);
            }
        }
    };

    if (brg.with_sum) {
        postops_injector_->set_lambda_injector(
                primitive_kind::sum, sum_injector);
    }

    auto live_accumulators = claim_reg_idxs<ZReg>(vmm_idxs_param);
    auto live_gprs = claim_regs<XReg>({param1, reg_A, reg_B, reg_aux_batch_addr,
            reg_BS, reg_BS_loop, reg_aux_M, reg_aux_D, reg_aux_C, reg_aux_A,
            reg_aux_B, reg_a_offset, reg_aux_N, reg_aux_A_vpad_top,
            reg_aux_A_vpad_bottom, reg_table_base, X_TMP_0});
    auto live_preds
            = claim_regs<PReg>({PReg(1), k_mask, k_tail_mask, kblend_mask});
    postops_injector_->compute_vector_range(vmm_idxs_param, rhs_arg_params);
}

void jit_brdgmm_kernel_base_t::store_accumulators_apply_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    const bool dq2ps_required = brg.is_int8;
    const int v_substep = vnni_substep();
    if (brg.with_scales) {
        add_imm(X_DEFAULT_ADDR, sp, reg_scales_offs_, X_TMP_0);
        ldr(reg_aux_scales, ptr(X_DEFAULT_ADDR));
        if (brg.is_oc_scale) {
            mov_imm(X_TMP_1, sizeof(float));
            mul(X_TMP_1, reg_aux_N, X_TMP_1);
            add(reg_aux_scales, X_TMP_1, reg_aux_scales);
        }
        for_(int m = 0; m < m_blocks; m++)
        for_(int n = 0; n < n_blocks; n++)
        for (int v_i = 0; v_i < v_substep; ++v_i) {
            const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const bool mask_flag = substep_simd < simd_w_;
            const ZReg vmm = accm(m_blocks, n_blocks, m, n, v_i);
            if (dq2ps_required) { scvtf(vmm.s, P_ALL_ONE / T_z, vmm.s); }
            const ZReg z_tmp = push_z_tmp(vmm, vmm);
            if (brg.is_oc_scale) {
                add_imm(X_DEFAULT_ADDR, reg_aux_scales, scales_offset(n, v_i),
                        X_TMP_0);
                if (mask_flag) { //zeroing
                    ld1rw(z_tmp.s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
                    fmul(vmm.s, vmm.s, z_tmp.s);
                    not_(P_TMP_0.b, P_ALL_ONE.b, k_mask.b);
                    mov(vmm.s, P_TMP_0 / T_m, 0);
                } else { //no mask
                    ld1w(z_tmp.s, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
                    fmul(vmm.s, vmm.s, z_tmp.s);
                }
            } else {
                if (mask_flag) { //zeroing
                    ld1rw(z_tmp.s, P_ALL_ONE / T_z, ptr(reg_aux_scales));
                    fmul(vmm.s, vmm.s, z_tmp.s);
                    not_(P_TMP_0.b, P_ALL_ONE.b, k_mask.b);
                    mov(vmm.s, P_TMP_0 / T_m, 0);
                } else { //no mask
                    ld1w(z_tmp.s, P_ALL_ONE / T_z, ptr(reg_aux_scales));
                    fmul(vmm.s, vmm.s, z_tmp.s);
                }
            }
            pop_z_tmp(z_tmp);
        }
    }

    if (brg.with_bias) {
        add_imm(X_DEFAULT_ADDR, sp, reg_bias_offs_, X_TMP_0);
        ldr(reg_aux_bias, ptr(X_DEFAULT_ADDR));
        mov_imm(X_TMP_1, brg.typesize_bias);
        mul(X_TMP_1, reg_aux_N, X_TMP_1);
        add(reg_aux_bias, X_TMP_1, reg_aux_bias);
    }

    for_(int v_i = 0; v_i < v_substep; ++v_i)
    for (int n = 0; n < n_blocks; n++) {
        auto vmm_bias = vmm_tmp(0);
        const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
        if (substep_simd <= 0) continue;
        if (brg.with_bias) {
            const bool mask_flag = has_n_tail && n + 1 == n_blocks;
            add_imm(X_DEFAULT_ADDR, reg_aux_bias, bias_offset(n, v_i), X_TMP_0);
            auto ptr_bias = ptr(X_DEFAULT_ADDR);
            cvt2ps(brg.dt_bias, vmm_bias, ptr_bias, mask_flag, false);
        }
        for (int m = 0; m < m_blocks; m++) {
            auto vmm = accm(m_blocks, n_blocks, m, n, v_i);
            if (dq2ps_required && !brg.with_scales)
                scvtf(vmm.s, P_ALL_ONE / T_z, vmm.s);
            if (brg.with_bias) { fadd(vmm.s, vmm.s, vmm_bias.s); }
        }
    }

    if (postops_injector_) apply_post_ops(m_blocks, n_blocks, has_n_tail);

    if (brg.with_dst_scales) {
        add_imm(X_DEFAULT_ADDR, sp, reg_dst_scales_offs_, X_TMP_0);
        ldr(reg_aux_dst_scales, ptr(X_DEFAULT_ADDR));
        auto vmm_dst_scales = vmm_tmp(0);
        ld1rw(vmm_dst_scales.s, P_ALL_ONE / T_z, ptr(reg_aux_dst_scales));

        for_(int m = 0; m < m_blocks; m++)
        for_(int n = 0; n < n_blocks; n++)
        for (int v_i = 0; v_i < v_substep; ++v_i) {
            const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const bool mask_flag = substep_simd < simd_w_;
            const ZReg vmm = accm(m_blocks, n_blocks, m, n, v_i);
            const ZReg z_tmp = push_z_tmp(vmm, vmm);
            if (mask_flag) { //zeroing
                ld1rw(z_tmp.s, P_ALL_ONE / T_z, ptr(reg_aux_dst_scales));
                fmul(vmm.s, vmm.s, z_tmp.s);
                not_(P_TMP_0.b, P_ALL_ONE.b, k_mask.b);
                mov(vmm.s, P_TMP_0 / T_m, 0);
            } else { //no mask
                ld1w(z_tmp.s, P_ALL_ONE / T_z, ptr(reg_aux_dst_scales));
                fmul(vmm.s, vmm.s, z_tmp.s);
            }
            pop_z_tmp(z_tmp);
        }
    }

    const bool dt_requires_saturation
            = one_of(brg.dt_d, data_type::u8, data_type::s8, data_type::s32);
    auto vmm_lbound = vmm_tmp(0);
    auto vmm_ubound = vmm_tmp(1);
    if (dt_requires_saturation) {
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp, data_type::f32, brg.dt_d);
    }

    for (int m = 0; m < m_blocks; m++) {
        for_(int n = 0; n < n_blocks; n++)
        for (int v_i = 0; v_i < v_substep; ++v_i) {
            const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
            if (substep_simd <= 0) continue;
            const auto offset = D_offset(m, n, v_i);
            add_imm(X_DEFAULT_ADDR, reg_aux_D, offset, X_TMP_0);
            auto addr = ptr(X_DEFAULT_ADDR);
            auto vmm = accm(m_blocks, n_blocks, m, n, v_i);
            if (dt_requires_saturation) {
                saturate_f32(vmm, vmm_lbound, vmm_ubound, brg.dt_d, P_ALL_ONE);
                frinti(vmm.s, P_ALL_ONE, vmm.s);
                fcvtzs(vmm.s, P_ALL_ONE, vmm.s);
            }
            const bool mask_flag = n + 1 == n_blocks && has_n_tail;
            auto mask = mask_flag ? k_mask / T_m : P_ALL_ONE / T_m;
            switch (brg.dt_d) {
                case data_type::f32:
                case data_type::s32: st1w(vmm.s, mask, addr); break;
                case data_type::bf16:
                    bfcvt(vmm.h, mask, vmm.s);
                    st1h(vmm.s, mask, addr);
                    break;
                case data_type::s8:
                    smin(vmm.s, std::numeric_limits<int8_t>::max());
                    smax(vmm.s, std::numeric_limits<int8_t>::min());
                    st1b(vmm.s, mask, addr);
                    break;
                case data_type::u8:
                    umin(vmm.s, std::numeric_limits<uint8_t>::max());
                    st1b(vmm.s, mask, addr);
                    break;
                default: assert(!"unknown dst_dt");
            }
        }
    }
}

void jit_brdgmm_kernel_base_t::store_accumulators_without_post_ops(
        int m_blocks, int n_blocks, bool has_n_tail) {

    const bool dt_requires_saturation
            = brg.is_int8 && brg.dt_c != data_type::s32;
    auto vmm_lbound = vmm_tmp(0);
    auto vmm_ubound = vmm_tmp(1);
    if (dt_requires_saturation) {
        init_saturate_f32(
                vmm_lbound, vmm_ubound, reg_tmp, data_type::f32, brg.dt_d);
    }

    for_(int m = 0; m < m_blocks; m++)
    for_(int n = 0; n < n_blocks; n++)
    for (int v_i = 0; v_i < vnni_substep(); ++v_i) {
        const int substep_simd = get_substep_simd(n, v_i, has_n_tail);
        if (substep_simd <= 0) continue;
        const bool mask_flag = substep_simd < simd_w_;
        auto vmm_acc = accm(m_blocks, n_blocks, m, n, v_i);
        if (dt_requires_saturation) {
            saturate_f32(vmm_acc, vmm_lbound, vmm_ubound, brg.dt_d, P_ALL_ONE);
            frinti(vmm_acc.s, P_ALL_ONE, vmm_acc.s);
            fcvtzs(vmm_acc.s, P_ALL_ONE, vmm_acc.s);
        }
        const auto offset = C_offset(m, n, v_i);
        add_imm(X_DEFAULT_ADDR, reg_aux_C, offset, X_TMP_0);
        st1w(vmm_acc.s, (mask_flag ? k_mask : P_ALL_ONE) / T_m,
                ptr(X_DEFAULT_ADDR));
    }
}

void jit_brdgmm_kernel_base_t::maybe_transpose_interleaved_vnni_to_plain(
        int m_blocks, int n_blocks, bool has_n_tail) {

    if (vnni_substep() == 1) return;
    assert(!"unsupported\n");
}

void jit_brdgmm_kernel_base_t::store_accumulators(
        int m_blocks, int n_blocks, bool has_n_tail) {

    maybe_transpose_interleaved_vnni_to_plain(m_blocks, n_blocks, has_n_tail);

    const bool are_post_ops_applicable = one_of(true, brg.with_eltwise,
            brg.with_binary, brg.with_scales, brg.with_bias, brg.with_sum,
            brg.dt_d != brg.dt_c, brg.with_dst_scales);

    Label label_done;
    if (are_post_ops_applicable) {
        store_accumulators_apply_post_ops(m_blocks, n_blocks, has_n_tail);
    } else {
        store_accumulators_without_post_ops(m_blocks, n_blocks, has_n_tail);
    }
}

void jit_brdgmm_kernel_base_t::load_a(
        ZReg vmma, int m_i, int n_i, int v_i, bool has_n_tail) {
    const int n_blocks
            = has_n_tail && n_block2_tail() > 0 ? n_block2_tail() : n_block2();
    const int substep_simd = get_substep_simd(n_i, v_i, has_n_tail);
    const bool is_tail_block = has_n_tail && n_i + 1 == n_blocks;
    const bool mask_flag = substep_simd < simd_w_;
    add_imm(X_DEFAULT_ADDR, reg_aux_A,
            A_offset(m_i, n_i) + is_tail_block * v_i * simd_w_ * brg.typesize_A,
            X_TMP_0);
    const auto addr = ptr(X_DEFAULT_ADDR);
    auto mask = mask_flag ? k_mask / T_z : P_ALL_ONE / T_z;
    if (brg.dt_a == data_type::f32) {
        ld1w(vmma.s, mask, addr);
    } else if (brg.dt_a == data_type::bf16) {
        ld1h(vmma.s, mask, addr);
        lsl(vmma.s, vmma.s, 16);
    } else if (brg.is_int8) {
        ld1b(vmma.s, mask, addr);
    } else {
        assert(!"unsupported data type\n");
    }
}

void jit_brdgmm_kernel_base_t::load_b(
        ZReg vmmb, int n_i, int v_i, bool has_n_tail) {
    // for B matrix we assume memory is padded and it is safe to load simd
    // elements.
    const int n_blocks
            = has_n_tail && n_block2_tail() > 0 ? n_block2_tail() : n_block2();
    const bool is_tail_block = has_n_tail && (n_i + 1 == n_blocks);
    add_imm(X_DEFAULT_ADDR, reg_aux_B,
            B_offset(n_i) + is_tail_block * v_i * simd_w_ * brg.typesize_B,
            X_TMP_0);
    const auto addr = ptr(X_DEFAULT_ADDR);
    if (brg.dt_b == data_type::f32) {
        ld1w(vmmb.s, P_ALL_ONE / T_z, addr);
    } else if (brg.dt_b == data_type::bf16) {
        ld1h(vmmb.s, P_ALL_ONE / T_z, addr);
        lsl(vmmb.s, vmmb.s, 16);
    } else if (brg.dt_b == data_type::s8) {
        ld1sb(vmmb.s, P_ALL_ONE / T_z, addr);
    } else {
        assert(!"unsupported data type\n");
    }
}

void jit_brdgmm_kernel_base_t::brdgmm_microkernel(int m_blocks, int n_blocks,
        bool has_top_padding, bool has_bottom_padding, bool has_tail) {

    const bool has_padding = has_top_padding || has_bottom_padding;
    const int max_bvmms
            = accm(m_blocks, n_blocks, 0, 0, 0).getIdx() - vmm_b(0).getIdx();
    const int v_substep = vnni_substep();

    auto dot_product = [&](ZReg vmma, ZReg vmmb, int m_i, int n_i, int v_i) {
        auto vmm_acc = accm(m_blocks, n_blocks, m_i, n_i, v_i);
        if (brg.is_f32) {
            const ZReg z_tmp = push_z_tmp(vmm_acc, vmmb);
            if (is_fma_embd()) {
                const bool mask_flag = has_tail && (n_i + 1 == n_blocks);
                add_imm(X_DEFAULT_ADDR, reg_aux_A, A_offset(m_i, n_i), X_TMP_0);
                const auto addr = ptr(X_DEFAULT_ADDR);
                auto mask = mask_flag ? k_mask : P_ALL_ONE;
                ld1w(z_tmp.s, mask / T_z, addr);
                fmla(vmm_acc.s, mask / T_m, vmmb.s, z_tmp.s);
            } else {
                fmla(vmm_acc.s, P_ALL_ONE / T_m, vmma.s, vmmb.s);
            }
            pop_z_tmp(z_tmp);
        } else if (brg.is_bf16) {
            bfdot(vmm_acc.s, vmmb.h, vmma.h);
        } else if (brg.is_int8 && isa_has_s8s8(brg.isa_impl)) {
            if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::u8)
                udot(vmm_acc.s, vmmb.b, vmma.b);
            else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::s8)
                sdot(vmm_acc.s, vmma.b, vmmb.b);
            else if (brg.dt_a == data_type::u8 && brg.dt_b == data_type::s8)
                usdot(vmm_acc.s, vmma.b, vmmb.b);
            else if (brg.dt_a == data_type::s8 && brg.dt_b == data_type::u8)
                assert(!"unsupported\n");
        }
    };

    if (!has_padding) {
        // preload vmm_b if possible.
        for_(int v_i = 0; v_i < v_substep; ++v_i)
        for (int nb_i = 0; nb_i < n_blocks; nb_i += max_bvmms) {
            const int n_e = nstl::min(nb_i + max_bvmms, n_blocks) - nb_i;
            for (int i = 0; i < n_e; ++i) {
                const int n_i = nb_i + i;
                if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                load_b(vmm_b(i), n_i, v_i, has_tail);
                // ld_idx : zreg number to load data from memory for use by dot_product()
                ld_idx = i + 1;
            }
            for_(int m_i = 0; m_i < m_blocks; ++m_i)
            for (int i = 0; i < n_e; ++i) {
                const int n_i = nb_i + i;
                if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                if (!is_fma_embd()) load_a(vmm_a(), m_i, n_i, v_i, has_tail);
                dot_product(vmm_a(), vmm_b(i), m_i, n_i, v_i);
            }
        }
    } else {
        const int max_req_preload_vmms = n_blocks * vnni_substep();
        const int n_preload_b_vmms = max_bvmms >= max_req_preload_vmms
                ? max_req_preload_vmms
                : max_bvmms - 1 /*for ad-hoc load*/;
        for (int i = 0; i < n_preload_b_vmms; ++i) {
            const int n_i = i % n_blocks;
            const int v_i = i / n_blocks;
            if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
            load_b(vmm_b(i), n_i, v_i, has_tail);
        }

        Label done;
        Label jmp_table_base;
        std::vector<Label> jmp_table_labels(m_blocks);
        if (has_top_padding) {
#define BR_INSTRUCTION_SIZE 4
            // jmp table
            adr(reg_table_base, jmp_table_base);
            mov_imm(X_TMP_1, sizeof(BR_INSTRUCTION_SIZE));
            mul(X_TMP_1, reg_aux_A_vpad_top, X_TMP_1);
            add(reg_table_base, X_TMP_1, reg_table_base);
            br(reg_table_base);
            align(8);
            L(jmp_table_base);
            for (int m_i = 0; m_i < m_blocks; ++m_i) {
                putL(jmp_table_labels[m_i]); // b <label>
            }
        }

        for (int m_i = 0; m_i < m_blocks; ++m_i) {
            L(jmp_table_labels[m_i]);
            if (has_bottom_padding) {
                cmp_imm(reg_aux_A_vpad_bottom, m_blocks - m_i, X_TMP_0);
                b(GE, done);
            }

            for_(int v_i = 0, p_b_i = 0; v_i < v_substep; ++v_i)
            for (int n_i = 0; n_i < n_blocks; ++n_i, ++p_b_i) {
                if (get_substep_simd(n_i, v_i, has_tail) <= 0) continue;
                if (!is_fma_embd()) load_a(vmm_a(), m_i, n_i, v_i, has_tail);
                if (p_b_i < n_preload_b_vmms) {
                    dot_product(vmm_a(), vmm_b(p_b_i), m_i, n_i, v_i);
                } else {
                    // preloaded vmm_b not available
                    const int b_idx = max_bvmms - 1;
                    load_b(vmm_b(b_idx), n_i, v_i, has_tail);
                    dot_product(vmm_a(), vmm_b(b_idx), m_i, n_i, v_i);
                }
            }
        }
        L(done);
    }
}

void jit_brdgmm_kernel_base_t::batch_loop(
        const int m_blocks, const int n_blocks, bool has_n_tail) {

    auto get_padding_info = [&]() {
        const bool do_check_effective_padding = check_effective_padding();
        if (has_vpad()) {
            Label no_top_padding;

            if (brg.brgattr.max_bottom_vpad > 0) {
                if (do_check_effective_padding) {
                    Label done_adjust_bottom_padding;
                    mov(reg_aux_A_vpad_bottom, reg_aux_M);
                    add_imm(reg_aux_A_vpad_bottom, reg_aux_A_vpad_bottom,
                            m_blocks - M(), X_TMP_0);
                    add_imm(X_DEFAULT_ADDR, reg_aux_batch_addr,
                            GET_OFF_BATCH_ELEMENT(vvpad.bottom), X_TMP_0);
                    ldr(X_TMP_1, ptr(X_DEFAULT_ADDR));
                    adds(reg_aux_A_vpad_bottom, reg_aux_A_vpad_bottom, X_TMP_1);
                    b(GE, done_adjust_bottom_padding);
                    eor(reg_aux_A_vpad_bottom, reg_aux_A_vpad_bottom,
                            reg_aux_A_vpad_bottom);
                    L(done_adjust_bottom_padding);
                } else {
                    add_imm(X_DEFAULT_ADDR, reg_aux_batch_addr,
                            GET_OFF_BATCH_ELEMENT(vvpad.bottom), X_TMP_0);
                    ldr(reg_aux_A_vpad_bottom, ptr(X_DEFAULT_ADDR));
                }
                mov(reg_total_padding, reg_aux_A_vpad_bottom);
            }
            if (brg.brgattr.max_top_vpad > 0) {
                add_imm(X_DEFAULT_ADDR, reg_aux_batch_addr,
                        GET_OFF_BATCH_ELEMENT(vvpad.top), X_TMP_0);
                ldr(reg_aux_A_vpad_top, ptr(X_DEFAULT_ADDR));
                if (do_check_effective_padding) {
                    Label done_adjust_top_padding;
                    subs(reg_aux_A_vpad_top, reg_aux_A_vpad_top, reg_aux_M);
                    b(GE, done_adjust_top_padding);
                    eor(reg_aux_A_vpad_top, reg_aux_A_vpad_top,
                            reg_aux_A_vpad_top);
                    L(done_adjust_top_padding);
                }
                if (brg.brgattr.max_bottom_vpad > 0) {
                    add(reg_total_padding, reg_total_padding,
                            reg_aux_A_vpad_top);
                } else {
                    mov(reg_total_padding, reg_aux_A_vpad_top);
                }
            }
        }
    };

    auto call_brdgmm_microkernel = [&]() {
        const int tpad = brg.brgattr.max_top_vpad;
        const int bpad = brg.brgattr.max_bottom_vpad;
        const bool vpad_exists = has_vpad();
        Label microkernel_with_padding, done_microkernel;

        if (vpad_exists) {
            cmp(reg_total_padding, 0);
            b(GT, microkernel_with_padding);
        }
        brdgmm_microkernel(m_blocks, n_blocks, false, false, has_n_tail);
        if (vpad_exists) {
            b(done_microkernel);
            L(microkernel_with_padding);
            if ((tpad + bpad) >= m_blocks) {
                cmp_imm(reg_total_padding, m_blocks, X_TMP_0);
                b(GE, done_microkernel);
            }
            brdgmm_microkernel(m_blocks, n_blocks, tpad, bpad, has_n_tail);
        }
        L(done_microkernel);
    };

    Label bs_loop_label, done_bs_loop;
    load_accumulators(m_blocks, n_blocks);
    cmp(reg_BS, 0);
    b(LE, done_bs_loop);
    mov(reg_BS_loop, reg_BS);
    restore_A_B_matrices();

    L(bs_loop_label);
    {
        set_A_B_matrices();
        get_padding_info();
        advance_A_B_matrices();
        call_brdgmm_microkernel();
        subs(reg_BS_loop, reg_BS_loop, 1);
        b(GT, bs_loop_label);
    }

    L(done_bs_loop);

    store_accumulators(m_blocks, n_blocks, has_n_tail);
}

void jit_brdgmm_kernel_base_t::compute_loop() {

    const bool has_m_block2_tail = m_block2_tail() > 0;
    const int loop_m = (nb_m_block2() - has_m_block2_tail);
    const bool do_loop_m = loop_m > 1;

    const bool has_n_block2_tail = n_block2_tail() > 0;
    const bool need_separate_n_block1_tail_block = n_block1_tail() != 0
            && !has_n_block2_tail && nb_n_block2() > 1 && false;
    const int loop_n = nb_n_block2() - has_n_block2_tail
            - need_separate_n_block1_tail_block;
    const bool do_loop_n = loop_n > 1;
    const bool loop_n_update_aux_ptrs = do_loop_n || (loop_n < nb_n_block2());

    auto n_loop = [&](int m_blocks) {
        Label n_loop_label;
        const int n_blocks = n_block2();
        const int n_loop_step = oc_logical_offset(n_blocks);
        const int n_loop_work = loop_n * n_blocks * n_block1();
        const bool vlen_tail_in_loop = n_block1_tail() != 0
                && !need_separate_n_block1_tail_block && !has_n_block2_tail;
        eor(reg_aux_N, reg_aux_N, reg_aux_N);

        L(n_loop_label);
        {
            if (do_loop_n) {
                if (vlen_tail_in_loop) {
                    Label done_k_mask;
                    cmp(reg_aux_N, n_loop_work - n_loop_step);
                    b(LT, done_k_mask);
                    mov(k_mask.b, P_ALL_ONE / T_z, k_tail_mask.b);
                    L(done_k_mask);
                }
            }

            batch_loop(m_blocks, n_blocks, vlen_tail_in_loop);

            if (loop_n_update_aux_ptrs) {
                add_imm(reg_aux_N, reg_aux_N, n_loop_step, X_TMP_0);
                add_imm(reg_a_offset, reg_a_offset,
                        n_loop_step * brg.typesize_A, X_TMP_0);
                add_imm(reg_aux_C, reg_aux_C, n_loop_step * brg.typesize_C,
                        X_TMP_0);
                add_imm(reg_aux_D, reg_aux_D, n_loop_step * brg.typesize_D,
                        X_TMP_0);
            }

            if (do_loop_n) {
                cmp_imm(reg_aux_N, n_loop_work, X_TMP_0);
                b(LT, n_loop_label);
            }
        }

        if (need_separate_n_block1_tail_block)
            batch_loop(m_blocks, n_blocks, true);

        if (has_n_block2_tail) {
            batch_loop(m_blocks, n_block2_tail(), n_block1_tail() != 0);
        }
    };

    auto m_loop = [&]() {
        Label m_loop_label;
        const int m_blocks = m_block2();
        const bool reset_mask
                = n_block1_tail() != 0 && do_loop_n && !has_n_block2_tail;

        eor(reg_aux_M, reg_aux_M, reg_aux_M);
        eor(reg_a_offset, reg_a_offset, reg_a_offset);

        L(m_loop_label);
        {
            if (reset_mask) { ptrue(k_mask.b); }
            n_loop(m_blocks);

            if (do_loop_m || has_m_block2_tail) {
                add_imm(reg_aux_M, reg_aux_M, m_blocks, X_TMP_0);
                const int n_loop_offset
                        = loop_n_update_aux_ptrs * loop_n * n_block2();
                add_imm(reg_a_offset, reg_a_offset,
                        A_offset(m_blocks, -n_loop_offset), X_TMP_0);
                add_imm(reg_aux_C, reg_aux_C,
                        C_offset(m_blocks, -n_loop_offset, 0), X_TMP_0);
                add_imm(reg_aux_D, reg_aux_D,
                        D_offset(m_blocks, -n_loop_offset, 0), X_TMP_0);
            }

            if (do_loop_m) {
                cmp_imm(reg_aux_M, loop_m * m_block2(), X_TMP_0);
                b(LT, m_loop_label);
            }
        }

        if (m_block2_tail() > 0) {
            if (reset_mask) { ptrue(k_mask.b); }
            n_loop(m_block2_tail());
        }
    };

    assert(m_block1_tail() == 0);
    m_loop();
}

void jit_brdgmm_kernel_base_t::init_masks() {

    if (n_block1_tail() != 0) {
        const bool has_n_block2_tail = n_block2_tail() > 0;
        if (has_n_block2_tail || nb_n_block2() <= 1) {
            // The mask can be set only once.
            set_preg(k_mask.s, n_block1_tail(), X_TMP_0);
        } else {
            // Need to adjust mask, and set only when needed.
            // So store it temporarily in k_tail_mask.
            set_preg(k_tail_mask.s, n_block1_tail(), X_TMP_0);
        }
    } else if (brg.with_binary) {
        // the post-ops injector seems to use mask unconditionally
        // set a default mask.
        ptrue(k_mask.b);
    }
}

void jit_brdgmm_kernel_base_t::generate() {

    preamble();
    sub_imm(sp, sp, utils::rnd_up(stack_space_needed_, 16), X_TMP_0);

    init_masks();

    read_params();
    compute_loop();

    add_imm(sp, sp, utils::rnd_up(stack_space_needed_, 16), X_TMP_0);
    postamble();

    if (brg.with_eltwise) postops_injector_->prepare_table();
}

brdgmm_kernel_t::brdgmm_kernel_t(const brgemm_desc_t abrd)
    : brgemm_kernel_(new jit_brdgmm_kernel_base_t(abrd)) {}

status_t brdgmm_kernel_t::create_kernel() {
    return brgemm_kernel_->create_kernel();
}

void brdgmm_kernel_t::operator()(brgemm_kernel_params_t *params) const {
    (*brgemm_kernel_)(params);
}

const jit_generator_t *brdgmm_kernel_t::get_jit_generator() const {
    return brgemm_kernel_;
}

brdgmm_kernel_t::~brdgmm_kernel_t() {
    delete brgemm_kernel_;
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
