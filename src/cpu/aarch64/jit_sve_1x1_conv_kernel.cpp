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
#include <assert.h>
#include <float.h>

#include "common/c_types_map.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/platform.hpp"

#include "cpu/aarch64/injectors/injector_utils.hpp"
#include "cpu/aarch64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/aarch64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/aarch64/jit_sve_1x1_conv_kernel.hpp"
#include "cpu/aarch64/jit_uni_1x1_conv_utils.hpp"

#define GET_OFF(field) \
    static_cast<int32_t>(offsetof(jit_1x1_conv_args_t, field))

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::prop_kind;
using namespace dnnl::impl::utils;

template <cpu_isa_t isa>
jit_sve_1x1_conv_kernel_t<isa>::jit_sve_1x1_conv_kernel_t(
        const jit_1x1_conv_conf_t &ajcp, const primitive_attr_t &attr,
        const memory_desc_t &dst_md)
    : jcp(ajcp), attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary) {
        using namespace binary_injector;
        const size_t tail_size = jcp.oc_without_padding % isa_simd_width_;
        static constexpr bool use_exact_tail_scalar_bcast = true;

        const rhs_arg_static_params_t rhs_arg_static_params {
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(dst_orig),
                memory_desc_wrapper(dst_md), tail_size, k_load_dim_mask,
                use_exact_tail_scalar_bcast};
        const static_params_t static_params {
                this->param1, rhs_arg_static_params};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<to_vla_sve(isa)>>(this,
                reg_pool(), jcp.post_ops, static_params, X_TMP_0, PReg(1));
    }
}

template <cpu_isa_t isa>
void jit_sve_1x1_conv_kernel_t<isa>::bcast_loop(int load_loop_blk) {

    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_output_data, reg_output_data);
    ldr(reg_bcast_loop_iter, ptr(sp, reg_bcast_loop_work_offt));

    Label bcast_loop;
    Label bcast_loop_tail;
    Label large_tail;

    cmp_imm(reg_bcast_loop_iter, jcp.bcast_block, reg_tmp_imm);
    b(LT, bcast_loop_tail);

    L(bcast_loop);
    {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            if (i + 1 == num_substeps) L(large_tail);
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add_vl_or_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_substep, reg_tmp_imm);
                add_vl_or_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_substep, reg_tmp_imm);
            } else {
                add_vl_or_imm(aux1_reg_bcast_data, aux1_reg_bcast_data,
                        jcp.bcast_loop_bcast_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_bcast_substep,
                        reg_tmp_imm);
                add_vl_or_imm(aux_reg_output_data, aux_reg_output_data,
                        jcp.bcast_loop_output_step
                                - (num_substeps - 1)
                                        * jcp.bcast_loop_output_substep,
                        reg_tmp_imm);
            }
            subs_imm(reg_bcast_loop_iter, reg_bcast_loop_iter, jcp.ur,
                    reg_tmp_imm);
        }
        cmp_imm(reg_bcast_loop_iter, jcp.bcast_block, reg_tmp_imm);
        b(GE, bcast_loop);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        if (jcp.ur_tail >= jcp.ur) {
            cmp_imm(reg_bcast_loop_iter, jcp.ur, reg_tmp_imm);
            b(GE, large_tail);
        }
        if (jcp.ur_tail % jcp.ur) {
            cmp(reg_bcast_loop_iter, 0);
            b(LE, bcast_loop_tail_out);
            reduce_loop(load_loop_blk, jcp.ur_tail % jcp.ur, 0, true);
            L(bcast_loop_tail_out);
        }
    }
}

static int vreg_accum_idx(
        const int load_loop_blk, const int i_load, const int i_ur) {
    return (i_ur * load_loop_blk + i_load);
}

template <typename F>
static void iterate(const int load_loop_blk, const int ur, const bool mask_tail,
        const F &fun) {
    for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
        const bool mask_flag = mask_tail && i_load + 1 == load_loop_blk;
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            fun(mask_flag, i_load, i_ur);
    }
}
template <typename F>
static void iterate(const int load_loop_blk, const int ur, const F &fun) {
    iterate(load_loop_blk, ur, false, fun);
}

template <cpu_isa_t isa>
void jit_sve_1x1_conv_kernel_t<isa>::apply_postops(
        const bool is_out_layout_nxc, const int load_loop_blk, const int ur) {
    injector_utils::vmm_index_set_t vmm_idxs;
    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
    if (jcp.with_binary) {
        const auto mask_tail = jcp.oc_without_padding % jcp.load_block;
        iterate(load_loop_blk, ur, mask_tail,
                [&](const bool mask_flag, const int i_load, const int i_ur) {
            const auto vmm_idx = vreg_accum_idx(load_loop_blk, i_load, i_ur);
            vmm_idxs.emplace(vmm_idx);

            rhs_arg_params.vmm_idx_to_out_reg.emplace(
                    vmm_idx, aux_reg_output_data);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(vmm_idx,
                    get_output_offset(is_out_layout_nxc, i_load, i_ur));
            if (mask_flag) rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx);
        });

        ldr(abi_param1, ptr(sp, reg_abi_param1_backup));

    } else {
        iterate(load_loop_blk, ur,
                [&](const bool, const int i_load, const int i_ur) {
            vmm_idxs.emplace(vreg_accum_idx(load_loop_blk, i_load, i_ur));
        });
    }

    auto live_accumulators = claim_reg_idxs<ZReg>(vmm_idxs);
    auto live_gprs = claim_regs<XReg>({reg_reduce_pos_flag, reduce_loop_iter,
            reg_reduce_loop_work, reg_bcast_loop_iter, reg_bcast_data,
            reg_load_data, reg_output_data, reg_bias_data, aux1_reg_bcast_data,
            aux_reg_output_data, aux_reg_bcast_data, aux_reg_load_data,
            reg_load_loop_work, reg_relu_ns, X_TMP_0});
    auto live_preds = claim_regs<PReg>(
            {PReg(1), k_load_dim_mask, k_load_dim_tail_mask});
    if (jcp.with_binary)
        postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);
    else
        postops_injector_->compute_vector_range(vmm_idxs);
}

template <cpu_isa_t isa>
void jit_sve_1x1_conv_kernel_t<isa>::reduce_loop(
        int load_loop_blk, int ur, int substep, bool wraparound) {

    const bool out_layout_nxc = is_out_layout_nxc(jcp);
    const bool load_layout_nxc = is_load_layout_nxc(jcp);
    const bool bcast_layout_nxc = is_bcast_layout_nxc(jcp);
    const int reduce_dim_tail = jcp.reduce_dim % jcp.reduce_block;
    const int load_dim_tail = jcp.load_dim % jcp.load_block;

    auto vreg_load
            = [=](int i_load) { return ZReg(ur * load_loop_blk + i_load); };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return ZReg(vreg_accum_idx(load_loop_blk, i_load, i_ur));
    };

    auto bias_offset = [=](int i_load) {
        return jcp.typesize_out * jcp.oc_block * i_load;
    };

    auto bcast_offset = [=](int i_reduce, int i_ur) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        int offt;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            assert(jcp.reduce_loop_unroll == jcp.reduce_block);
            const int reduce_mul = bcast_layout_nxc ? jcp.reduce_dim
                                                    : jcp.reduce_loop_unroll;
            offt = (i_reduce == jcp.reduce_loop_unroll)
                    ? (jcp.bcast_dim + i_ur) * reduce_mul
                    : i_ur * reduce_mul + i_reduce;
        } else {
            int rmul = bcast_layout_nxc ? jcp.ic : jcp.ic_block;
            offt = i_reduce * rmul + i_ur;
        }
        return jcp.typesize_in * offt;
    };

    auto load_offset = [=](int i_reduce, int i_load) {
        int offt;
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;
        int lmul = jcp.load_block
                * (load_layout_nxc ? 1
                                   : utils::rnd_up(
                                             jcp.reduce_dim, jcp.reduce_block));
        int rmul = load_layout_nxc ? jcp.load_dim : jcp.load_block;
        offt = i_load * lmul + u0 * rmul;
        return u1 * jcp.reduce_loop_load_step + jcp.typesize_in * offt;
    };

    auto output_offset = [=](const bool is_out_layout_nxc, const int i_load,
                                 const int i_ur) {
        int32_t offset;
        if (one_of(jcp.prop_kind, forward_training, forward_inference,
                    backward_data)) {
            int i_load_shift = is_out_layout_nxc
                    ? jcp.load_block
                    : (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim)
                            * jcp.load_block;
            int i_ur_shift = is_out_layout_nxc ? jcp.load_dim : jcp.load_block;
            offset = (i_load * i_load_shift + i_ur * i_ur_shift)
                    * jcp.typesize_out;
        } else {
            offset = jcp.typesize_out * jcp.load_block * i_ur;
        }
        return offset;
    };

    auto init = [=]() {
        Label init_done;
        Label init_zero;

        if (jcp.with_bias
                && one_of(jcp.prop_kind, forward_training, forward_inference)) {
            tst(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            b(EQ, init_zero);

            for (int i_load = 0; i_load < load_loop_blk; i_load++) {
                auto vreg_acc = vreg_accum(i_load, 0);
                if (i_load + 1 == load_loop_blk && load_dim_tail)
                    LD_MUL_VL(ld1w, vreg_acc.s, k_load_dim_mask, reg_bias_data,
                            bias_offset(i_load), 4);
                else
                    LD_MUL_VL(ld1w, vreg_acc.s, P_ALL_ONE, reg_bias_data,
                            bias_offset(i_load), 4);
                for (int i_ur = 1; i_ur < ur; ++i_ur) {
                    auto vreg_acc = vreg_accum(i_load, i_ur);
                    mov(vreg_acc.d, vreg_accum(i_load, 0).d);
                }
            }
            b(init_done);
        }

        L(init_zero);

        /* Zero clear */
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                eor(r.d, r.d, r.d);
            }
        L(init_done);
    };

    auto store = [=]() {
        Label store_noadd;
        if (!jcp.with_sum) {
            tst(reg_reduce_pos_flag, FLAG_REDUCE_FIRST);
            b(NE, store_noadd);
        }

        auto output_addr = reg_tmp_addr;
        mov(output_addr, aux_reg_output_data);
        for (int i_ur = 0; i_ur < ur; ++i_ur)
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                if (!one_of(jcp.prop_kind, forward_training, forward_inference,
                            backward_data)) {
                    mov(X_TMP_0, i_load);
                    madd(output_addr, X_TMP_0, reg_output_stride,
                            aux_reg_output_data);
                }

                auto offset = output_offset(out_layout_nxc, i_load, i_ur);
                auto r = vreg_accum(i_load, i_ur).s;
                if (i_load + 1 == load_loop_blk && load_dim_tail)
                    LD_MUL_VL(ld1w, zreg_tmp.s, k_load_dim_mask, output_addr,
                            offset, 4);
                else
                    LD_MUL_VL(ld1w, zreg_tmp.s, P_ALL_ONE, output_addr, offset,
                            4);
                fadd(r, r, zreg_tmp.s);
            }

        L(store_noadd);
        if (jcp.with_eltwise || jcp.with_binary) {
            Label store_nopostops;
            tst(reg_reduce_pos_flag, FLAG_REDUCE_LAST);
            b(EQ, store_nopostops);

            apply_postops(out_layout_nxc, load_loop_blk, ur);

            L(store_nopostops);
        }

        auto store_output = [=](bool output_is_aligned) {
            const auto mask_flag = load_dim_tail;
            auto output_addr = reg_tmp_addr;
            mov(output_addr, aux_reg_output_data);
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    if (!one_of(jcp.prop_kind, forward_training,
                                forward_inference, backward_data)) {
                        mov(X_TMP_0, i_load);
                        madd(output_addr, X_TMP_0, reg_output_stride,
                                aux_reg_output_data);
                    }
                    auto vreg_acc = vreg_accum(i_load, i_ur);
                    // for nxc_layout-bwd_w, weights are still padded and the
                    // output_ptr here can be uninitialized scratchpad.
                    // To ensure final output (after reduction) is zero-padded,
                    // here we zero-pad output by omitting the mask.
                    if (jcp.prop_kind != backward_weights
                            && (i_load + 1 == load_loop_blk && mask_flag)) {
                        ST_MUL_VL(st1w, vreg_acc.s, k_load_dim_mask,
                                output_addr,
                                output_offset(out_layout_nxc, i_load, i_ur), 4);
                    } else {
                        ST_MUL_VL(st1w, vreg_acc.s, P_ALL_ONE, output_addr,
                                output_offset(out_layout_nxc, i_load, i_ur), 4);
                    }
                }
            }
        };

        Label unaligned_store, end_store;
        tst(aux_reg_output_data, cpu_isa_traits<isa>::vlen - 1);
        b(NE, unaligned_store);
        store_output(true);
        b(end_store);
        L(unaligned_store);
        { store_output(false); }
        L(end_store);
    };

    auto fma_block = [=](bool last_block) {
        const int i_reduce_end = reduce_dim_tail && last_block
                ? reduce_dim_tail
                : jcp.reduce_loop_unroll;

        auto bcast_addr = reg_tmp_addr;
        mov(bcast_addr, aux_reg_bcast_data);
        int base_offset = 0;
        for (int i_reduce = 0; i_reduce < i_reduce_end; i_reduce++) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                auto vreg = vreg_load(i_load);
                if (i_load + 1 == load_loop_blk && load_dim_tail)
                    LD_MUL_VL(ld1w, vreg.s, k_load_dim_mask, aux_reg_load_data,
                            load_offset(i_reduce, i_load), 4);
                else
                    LD_MUL_VL(ld1w, vreg.s, P_ALL_ONE, aux_reg_load_data,
                            load_offset(i_reduce, i_load), 4);
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                int offset = bcast_offset(i_reduce, i_ur);
                int offset_ = offset - base_offset;

                if (jcp.expl_bcast && load_loop_blk > 1) {
                    // ldr immediate has to be >=-256 and < 255.
                    // Refer to https://developer.arm.com/documentation/ddi0487/latest.
                    if (offset_ < -256 || offset_ > 255) {
                        add_vl_or_imm(bcast_addr, bcast_addr, offset_, X_TMP_0);
                        base_offset = offset;
                        offset_ = 0;
                    }
                    ldr(W_TMP_0, ptr(bcast_addr, offset_));
                    dup(vreg_bcast.s, W_TMP_0);
                } else {
                    // ld1rw has to be a multiple of 4 in the range [0, 252].
                    // Refer to https://developer.arm.com/documentation/ddi0596/2020-12/SVE-Instructions/LD1RW--Load-and-broadcast-unsigned-word-to-vector-.
                    if (offset_ < 0 || offset_ > 252 || offset_ % 4 != 0) {
                        add_vl_or_imm(bcast_addr, bcast_addr, offset_, X_TMP_0);
                        base_offset = offset;
                        offset_ = 0;
                    }
                    ld1rw(zreg_tmp.s, P_ALL_ONE, ptr(bcast_addr, offset_));
                }

                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    auto vreg_acc = vreg_accum(i_load, i_ur);
                    if (i_load + 1 == load_loop_blk && load_dim_tail) {
                        fmla(vreg_acc.s, k_load_dim_mask / T_m,
                                vreg_load(i_load).s, zreg_tmp.s);
                    } else {
                        fmla(vreg_acc.s, P_ALL_ONE / T_m, vreg_load(i_load).s,
                                zreg_tmp.s);
                    }
                }
            }
        }
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    subs_imm(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll,
            reg_tmp_imm);
    b(LE, reduce_loop_tail);

    L(reduce_loop);
    {
        fma_block(false);
        add_vl_or_imm(aux_reg_bcast_data, aux_reg_bcast_data,
                jcp.reduce_loop_bcast_step, reg_tmp_imm);
        add_vl_or_imm(aux_reg_load_data, aux_reg_load_data,
                jcp.reduce_loop_load_step, reg_tmp_imm);
        subs_imm(reduce_loop_iter, reduce_loop_iter, jcp.reduce_loop_unroll,
                reg_tmp_imm);
        b(GT, reduce_loop);
    }

    L(reduce_loop_tail);
    fma_block(true);

    store();
}

template <cpu_isa_t isa>
void jit_sve_1x1_conv_kernel_t<isa>::generate() {
    preamble();
    sub_imm(sp, sp, utils::rnd_up(stack_space_needed, 16), X_TMP_0);

    if (jcp.with_binary) {
        const auto zeroed_reg = x15;
        eor(zeroed_reg, zeroed_reg, zeroed_reg);
        str(zeroed_reg, ptr(sp, reg_binary_post_op_acc_off));
        str(param1, ptr(sp, reg_abi_param1_backup));
    }

    /* Pointers indicate weight, input, and output data */
    ldr(reg_bcast_data, ptr(abi_param1, GET_OFF(bcast_data))); // Input
    ldr(reg_load_data, ptr(abi_param1, GET_OFF(load_data))); // Weight
    ldr(reg_output_data, ptr(abi_param1, GET_OFF(output_data))); // Output

    /* Pointer indicates bias data if the layer has bias option */
    if (jcp.with_bias) ldr(reg_bias_data, ptr(abi_param1, GET_OFF(bias_data)));

    /* Get workloads of each loop */
    ldr(reg_load_loop_work, ptr(abi_param1, GET_OFF(load_dim)));
    ldr(reg_bcast_loop_work, ptr(abi_param1, GET_OFF(bcast_dim)));
    str(reg_bcast_loop_work, ptr(sp, reg_bcast_loop_work_offt));
    ldr(reg_reduce_loop_work, ptr(abi_param1, GET_OFF(reduce_dim)));

    /* A flag for controlling reduce loop */
    ldr(reg_reduce_pos_flag, ptr(abi_param1, GET_OFF(first_last_flag)));
    if (jcp.prop_kind == backward_weights)
        ldr(reg_output_stride, ptr(param1, GET_OFF(output_stride)));

    const int load_dim_tail
            = (one_of(jcp.prop_kind, forward_training, forward_inference)
                              ? jcp.oc_without_padding
                              : jcp.load_dim)
            % jcp.load_block;
    if (load_dim_tail) {
        const WReg w_tmp(reg_load_dim_tail_mask.getIdx());
        mov_imm(w_tmp, (1 << load_dim_tail) - 1);
        st1w(zreg_tmp1.s, P_ALL_ONE / T_z, ptr(sp, -1, MUL_VL));
        index(zreg_tmp.s, 0, 1);
        mov(zreg_tmp1.s, 1);
        lsl(zreg_tmp1.s, P_ALL_ONE / T_m, zreg_tmp.s);
        dup(zreg_tmp.s, w_tmp);
        and_(zreg_tmp.d, zreg_tmp.d, zreg_tmp1.d);
        cmpne(k_load_dim_tail_mask.s, P_ALL_ONE, zreg_tmp.s, 0);
        ldr(zreg_tmp1, ptr(sp, -1, MUL_VL));
    }

    auto load_loop_body = [=](int load_loop_blk) {
        if (load_dim_tail) {
            eor(k_load_dim_mask.b, P_ALL_ONE / T_z, k_load_dim_mask.b,
                    k_load_dim_mask.b);
            not_(k_load_dim_mask.b, P_ALL_ONE / T_z, k_load_dim_mask.b);
        }
        subs_imm(reg_load_loop_work, reg_load_loop_work,
                load_loop_blk * jcp.load_loop_iter_step, reg_tmp_imm);
        if (load_dim_tail) {
            Label no_update_mask;
            b(GE, no_update_mask);
            mov(k_load_dim_mask.b, k_load_dim_tail_mask.b);
            L(no_update_mask);
        }
        bcast_loop(load_loop_blk);
        add_vl_or_imm(reg_load_data, reg_load_data,
                load_loop_blk * jcp.load_loop_load_step, reg_tmp_imm);
        switch (jcp.prop_kind) {
            case forward_training:
            case forward_inference:
                add_vl_or_imm(reg_bias_data, reg_bias_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out,
                        reg_tmp_imm);
                add_vl_or_imm(reg_output_data, reg_output_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out
                                * (is_out_layout_nxc(jcp)
                                                ? 1
                                                : (jcp.with_dw_conv
                                                                  ? jcp.ow
                                                                  : jcp.bcast_dim)),
                        reg_tmp_imm);
                if (jcp.with_binary) {
                    const auto oc_off_oprnd = aux_reg_load_data;
                    ldr(oc_off_oprnd, ptr(sp, reg_binary_post_op_acc_off));
                    add_vl_or_imm(oc_off_oprnd, oc_off_oprnd,
                            jcp.load_block * load_loop_blk, X_TMP_0);
                    str(oc_off_oprnd, ptr(sp, reg_binary_post_op_acc_off));
                }
                break;
            case backward_data:
                add_vl_or_imm(reg_output_data, reg_output_data,
                        load_loop_blk * jcp.load_block * jcp.typesize_out
                                * (is_out_layout_nxc(jcp) ? 1 : jcp.bcast_dim),
                        reg_tmp_imm);
                break;
            case backward_weights:
                for (int i_load = 0; i_load < load_loop_blk; i_load++)
                    add(reg_output_data, reg_output_data, reg_output_stride);
                break;
            default: assert(!"invalid prop_kind");
        }
    };

    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);

    Label load_loop_blk[7];

    // with an implicit load_loop_block          {6, 5, 4, 3, 2,  1}
    static const int ur_cases_fma_embd_bcast[] = {2, 4, 5, 8, 14, 32};
    static const int ur_cases_fma_expl_bcast[] = {2, 5, 6, 9, 14, 32};

    const int size_ur_cases_fma = jcp.expl_bcast
            ? sizeof(ur_cases_fma_expl_bcast)
            : sizeof(ur_cases_fma_embd_bcast);

    const int *ur_cases_fma = jcp.expl_bcast ? ur_cases_fma_expl_bcast
                                             : ur_cases_fma_embd_bcast;
    const int *ur_cases = ur_cases_fma;
    const int num_ur_cases = size_ur_cases_fma / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.nb_load > label_idx && jcp.ur <= ur_cases[ur_idx]) {
            cmp_imm(reg_load_loop_work, simd_w * (label_idx + 1), reg_tmp_imm);
            b(LE, load_loop_blk[label_idx]);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.nb_load > label_idx && jcp.ur <= ur_cases[ur_idx]) {
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    b(LE, load_loop_blk[num_ur_cases]);
                }
                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    cmp_imm(reg_load_loop_work, 2 * label_idx * simd_w,
                            reg_tmp_imm);
                    b(EQ, load_loop_blk[label_idx - 1]);
                }
                cmp_imm(reg_load_loop_work, label_idx * simd_w, reg_tmp_imm);
                b(GT, load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx >= 0; --idx) {
                cmp_imm(reg_load_loop_work, simd_w * (idx + 1), reg_tmp_imm);
                b(GE, load_loop_blk[idx]);
            }
            if (ur_idx < num_ur_cases - 2) {
                cmp_imm(reg_load_loop_work, simd_w, reg_tmp_imm);
                b(LE, load_loop_blk[0]);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    add_imm(sp, sp, utils::rnd_up(stack_space_needed, 16), X_TMP_0);
    postamble();

    if (jcp.with_eltwise) postops_injector_->prepare_table();
}

template <cpu_isa_t isa>
status_t jit_sve_1x1_conv_kernel_t<isa>::init_conf(jit_1x1_conv_conf_t &jcp,
        const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, const memory_desc_wrapper &dst_d,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {

    /* arch check */
    if (!mayiuse(isa)) { return status::unimplemented; }
    jcp.isa = isa;

    if (!everyone_is(data_type::f32, src_d.data_type(), weights_d.data_type(),
                dst_d.data_type())) {
        return status::unimplemented;
    }

    jcp.nthr = nthreads;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int ndims = src_d.ndims();
    /* Forward_[training, inference], backward_[data, weight] */
    jcp.prop_kind = cd.prop_kind;

    /* Check group option */
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    /* Batchsize */
    jcp.mb = src_d.dims()[0];
    /* Channel */
    jcp.oc_without_padding = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc = jcp.oc_without_padding;
    jcp.ic_without_padding = src_d.dims()[1] / jcp.ngroups;
    jcp.ic = jcp.ic_without_padding;
    /* D, H, W */
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    /* Kernel size */
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    /* padding params */
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    /* stride params */
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];
    /* bias info */
    jcp.with_bias = pick_by_prop_kind(jcp.prop_kind, cd.bias_desc.format_kind,
                            format_kind::undef, cd.diff_bias_desc.format_kind)
            != format_kind::undef;

    /* Spatials */
    jcp.os = jcp.od * jcp.oh * jcp.ow;
    jcp.is = jcp.id * jcp.ih * jcp.iw;

    /* Depthwise conv check */
    const auto &post_ops = attr.post_ops_;
    const int dw_conv_ind = post_ops.find(primitive_kind::convolution);
    jcp.with_dw_conv = dw_conv_ind != -1;
    if (jcp.with_dw_conv) { return status::unimplemented; }

    /* Post operation check */
    // Using dw_conv_ind as upper-bound below, as post-ops after it will be
    // handled in depthwise convolution.
    const int eltwise_ind
            = post_ops.find(primitive_kind::eltwise, 0, dw_conv_ind);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        if (dst_d.data_type() == data_type::s32) {
            return status::unimplemented;
        }
    }

    const int sum_ind = post_ops.find(primitive_kind::sum, 0, dw_conv_ind);
    jcp.with_sum = sum_ind != -1;

    const int binary_ind
            = post_ops.find(primitive_kind::binary, 0, dw_conv_ind);
    jcp.with_binary = binary_ind != -1;

    if (dw_conv_ind >= 0) {
        // dw_conv and post_ops after it are handled externally, so skip them
        jcp.post_ops.entry_.assign(post_ops.entry_.cbegin(),
                post_ops.entry_.cbegin() + dw_conv_ind);
    } else {
        jcp.post_ops = post_ops;
    }

    /* Data format check */
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    bool is_data_layout_nxc;
    format_tag_t required_dat_tag;

    switch (isa) {
        case sve_512: {
            const auto dat_tag_nCx16c
                    = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
            jcp.src_tag = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
            jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
            is_data_layout_nxc
                    = utils::everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
            required_dat_tag
                    = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
            break;
        }
        case sve_256: {
            const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
            jcp.src_tag = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
            jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx8c);
            is_data_layout_nxc
                    = utils::everyone_is(dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
            required_dat_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
            break;
        }
        case sve_128: {
            if (jcp.prop_kind == prop_kind::backward_data) {
                // SVE128 BWD_D uses plain activations only.
                jcp.src_tag = src_d.matches_one_of_tag(dat_tag_nxc);
                jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag_nxc);
                is_data_layout_nxc = utils::everyone_is(
                        dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
                required_dat_tag = dat_tag_nxc;
            } else {
                const auto dat_tag_nCx4c
                        = pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
                jcp.src_tag
                        = src_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx4c);
                jcp.dst_tag
                        = dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx4c);
                is_data_layout_nxc = utils::everyone_is(
                        dat_tag_nxc, jcp.src_tag, jcp.dst_tag);
                required_dat_tag
                        = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx4c;
            }
            break;
        }
        default: return status::unimplemented;
    }
    /* Channel padding check */
    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1
            && src_d.data_type() == data_type::f32;

    /* Input and output must be multiple of simd_w */
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simd_w);
        jcp.ic = rnd_up(jcp.ic, simd_w);
    }

    using namespace injector;

    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;
    static constexpr bool sum_requires_zp_zero = true;
    const bool post_ops_ok_ = post_ops_ok(post_ops_ok_args_t(jcp.isa,
            {eltwise, binary, sum}, jcp.post_ops, &dst_d, sum_at_pos_0_only,
            sum_requires_scale_one, sum_requires_zp_zero));
    if (!post_ops_ok_) { return status::unimplemented; }

    bool args_ok = true && jcp.ngroups == 1 && jcp.src_tag == required_dat_tag
            && jcp.dst_tag == required_dat_tag
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0)
            && jcp.f_pad == 0 && jcp.t_pad == 0 && jcp.l_pad == 0
            && jcp.stride_w == 1 && jcp.stride_h == 1 && jcp.stride_d == 1
            && jcp.kd == 1 && jcp.kh == 1 && jcp.kw == 1 && jcp.ow == jcp.iw
            && jcp.oh == jcp.ih && jcp.od == jcp.id; // enforce rpad=0
    if (!args_ok) { return status::unimplemented; }

    /* Channel blocking size is simd_w */
    jcp.ic_block = jcp.oc_block = simd_w;

    switch (isa) {
        case sve_512: {
            jcp.ver = ver_sve_512;
            break;
        }
        case sve_256: {
            jcp.ver = ver_sve_256;
            break;
        }
        case sve_128: {
            jcp.ver = ver_sve_128;
            break;
        }
        default: return status::unimplemented;
    }

    if (everyone_is(data_type::f32, src_d.data_type(), weights_d.data_type(),
                dst_d.data_type())) {
        const int is_bwd_d = jcp.prop_kind == backward_data;

        /* Set weight data layout tag */
        format_tag_t wei_tag;
        switch (isa) {
            case sve_512: {
                wei_tag = with_groups
                        ? pick(2 * ndims - 6 + is_bwd_d, gOIw16i16o, gIOw16o16i,
                                  gOIhw16i16o, gIOhw16o16i, gOIdhw16i16o,
                                  gIOdhw16o16i)
                        : pick(2 * ndims - 6 + is_bwd_d, OIw16i16o, IOw16o16i,
                                  OIhw16i16o, IOhw16o16i, OIdhw16i16o,
                                  IOdhw16o16i);
                break;
            }
            case sve_256: {
                wei_tag = with_groups
                        ? pick(2 * ndims - 6 + is_bwd_d, gOIw8i8o, gIOw8o8i,
                                  gOIhw8i8o, gIOhw8o8i, gOIdhw8i8o, gIOdhw8o8i)
                        : pick(2 * ndims - 6 + is_bwd_d, OIw8i8o, IOw8o8i,
                                  OIhw8i8o, IOhw8o8i, OIdhw8i8o, IOdhw8o8i);
                break;
            }
            case sve_128: {
                wei_tag = with_groups
                        ? pick(2 * ndims - 6 + is_bwd_d, gOIw4i4o, gIOw4o4i,
                                  gOIhw4i4o, gIOhw4o4i, gOIdhw4i4o, gIOdhw4o4i)
                        : pick(2 * ndims - 6 + is_bwd_d, OIw4i4o, IOw4o4i,
                                  OIhw4i4o, IOhw4o4i, OIdhw4i4o, IOdhw4o4i);
                break;
            }
            default: return status::unimplemented;
        }

        jcp.wei_tag = weights_d.matches_one_of_tag(wei_tag);

        if (jcp.wei_tag != wei_tag) return status::unimplemented;

        jcp.typesize_in = sizeof(prec_traits_t<data_type::f32>::type);
        jcp.typesize_out = sizeof(prec_traits_t<data_type::f32>::type);
    } else {
        // TODO: currently, only support fp32;
        return status::unimplemented;
    }

    /* once all the formats are set, check the padding consistency */

    if (!is_data_layout_nxc) {
        args_ok = true && jcp.ic <= src_d.padded_dims()[1]
                && jcp.oc <= dst_d.padded_dims()[1]
                && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
                && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
        if (!args_ok) { return status::unimplemented; }
    }

    // TODO: Optimize bellow params
    const int SMALL_SPATIAL = 10;
    const int BIG_SPATIAL = 65;
    const int BIG_REDUCE_DIM = 1024;
    const int BIG_LOAD_DIM = (jcp.reduce_dim >= 512) ? 256 : 512;

    int load_blocking {0};
    int load_blocking_max {0};
    int bcast_blocking {0};
    int bcast_blocking_max {0};
    int reduce_blocking {0};
    int reduce_blocking_max {0};

    jcp.load_grp_count = 1;

    // TODO: mov check funcs into platform files
    const int L1_capacity
            = platform::get_per_core_cache_size(1) / sizeof(float);
    const int L2_size = platform::get_per_core_cache_size(2) / sizeof(float);
    const int L2_capacity = (L2_size * 3) / 4;

    /* FWD, BWD data */

    if (one_of(jcp.prop_kind, forward_training, forward_inference,
                backward_data)) {
        if (one_of(jcp.prop_kind, forward_training, forward_inference)) {
            /* Forward */
            if (jcp.with_dw_conv) jcp.ur = nstl::min(jcp.ow, jcp.ur);
            jcp.reduce_dim = jcp.ic; // src channel
            jcp.reduce_block = jcp.ic_block; // src simd_w

            jcp.load_dim = jcp.oc; // dst channel
            jcp.load_block = jcp.oc_block; // dst simd_W

            jcp.bcast_dim = jcp.is; // src H*W
        } else {
            /* Backward data */
            jcp.reduce_dim = jcp.oc; // src channel
            jcp.reduce_block = jcp.oc_block; // src simd_w

            jcp.load_dim = jcp.ic; // dst channel
            jcp.load_block = jcp.ic_block; // dst simd_w

            jcp.bcast_dim = jcp.os; // src H*W
        }
        /* # of consecutive channel elements  */
        jcp.reduce_loop_unroll = jcp.reduce_block;

        /* Offset to move to the next 16 input channel elements with the same H*W position */
        jcp.reduce_loop_bcast_step = jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? 1 : jcp.bcast_dim) * jcp.typesize_in;

        /* Offset: 16o*16i (filter) */
        jcp.reduce_loop_load_step
                = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;

        /* Offset: I/16 * 16o */
        jcp.load_loop_load_step
                = (utils::rnd_up(jcp.reduce_dim, jcp.reduce_block))
                * jcp.load_block * jcp.typesize_in;

        /* adjusting registry blocking */
        int max_regs, min_regs, size_threshold;

        /* spatial : H*D of dst */
        const int spatial
                = (one_of(jcp.prop_kind, forward_training, forward_inference))
                ? jcp.od * jcp.oh // forward
                : jcp.id * jcp.ih; // backward

        if ((8 * jcp.mb) / jcp.nthr >= 1
                // NHWC perf: RN50 mb=1
                || (is_data_layout_nxc && jcp.mb == 1)) {
            max_regs = 9; // max # of ur_w
            min_regs = 6; // min # of ur_w
            size_threshold = 14;
            jcp.expl_bcast = false; // ld1rw is faster for most cases

            /*
            *  H*D of dst  > SMALL_SPATIAL
            */
            if (jcp.load_dim > 128 && jcp.load_dim < BIG_LOAD_DIM
                    && spatial > SMALL_SPATIAL && spatial < BIG_SPATIAL
                    && jcp.reduce_dim < 256) {
                max_regs = 6;
                min_regs = 5;
            }
        } else {
            max_regs = 30;
            min_regs = 9;
            size_threshold = 14;
            jcp.expl_bcast = false;
            jcp.use_vmovntps = true;
        }
        jcp.ur = 1;

        for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
            /*
             *  H*D of dst >= size_threshold, (H*D of dst) % ur_w == 0
             *  or
             *  H*D of dst < size_threshold, (H*W of dst) % ur_w == 0
             */
            if ((spatial >= size_threshold && spatial % ur_w == 0)
                    || (spatial < size_threshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }

        if (jcp.ur == 1) {
            // If ur = 1, then min(max_regs, H*W of dst)
            jcp.ur = nstl::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i--) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0) break;
                }
            }
        }
        jcp.bcast_block = jcp.ur; // block size of bcast (input data)
        /* Number of steps for the dst address to output, used in bcast_loop() */
        jcp.bcast_loop_output_step = jcp.ur * jcp.typesize_out
                * (is_data_layout_nxc ? jcp.load_dim : jcp.load_block);
        jcp.bcast_loop_output_substep = -1; // unused

        /* Number of steps for the src address to be broadcasted in bcast_loop() */
        jcp.bcast_loop_bcast_step = jcp.ur * jcp.typesize_in
                * (is_data_layout_nxc ? jcp.reduce_dim : jcp.reduce_block);
        jcp.bcast_loop_bcast_substep = -1; // unused

        jcp.load_loop_iter_step = jcp.load_block;

        if (jcp.prop_kind == backward_data)
            jcp.loop_order = loop_lbr;
        else
            jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

        int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
        int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);
        int nb_load = div_up(jcp.load_dim, jcp.load_block);
        if (is_data_layout_nxc) {
            reduce_blocking = jcp.reduce_dim;
        } else if (jcp.expl_bcast) {
            if (jcp.load_dim <= BIG_LOAD_DIM && spatial > SMALL_SPATIAL
                    && spatial < BIG_SPATIAL) {
                reduce_blocking = nstl::min(jcp.reduce_dim, 80);
            } else if (spatial > SMALL_SPATIAL)
                reduce_blocking = nstl::min(jcp.reduce_dim, 512);
            else
                reduce_blocking = nstl::min(jcp.reduce_dim, 256);
        } else {
            reduce_blocking = nb_reduce;
            if (spatial <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 16;
            else if (spatial > SMALL_SPATIAL
                    && jcp.reduce_dim >= BIG_REDUCE_DIM)
                reduce_blocking = 8;
            reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
            reduce_blocking *= jcp.reduce_block;
        }

        // Check input data cache aliasing.
        // For other ISA constants may be updated.
        // 64 * 1024 is chosen due to 1MB L2 16-way cache.
        // 7 is empirical value. It is about half of 16.
        // So we leave about half of the set for other data - weights, dst
        int way_size = (16 * 1024) / jcp.typesize_in;
        int max_hits = 7;
        if (!is_data_layout_nxc
                && jcp.bcast_dim * reduce_blocking > way_size * max_hits) {
            int nrb = reduce_blocking / simd_w;
            int sp = jcp.bcast_dim;
            int wl = way_size / simd_w;
            for (int start_off = 0; start_off < jcp.ur; start_off++) {
                for (int off = start_off, hits = 0; off < sp * nrb; off += wl) {
                    if (off % sp >= jcp.ur || ++hits < max_hits) continue;
                    int max_r_blocking = simd_w * nstl::max(1, (off + wl) / sp);
                    reduce_blocking
                            = nstl::min(reduce_blocking, max_r_blocking);
                    break;
                }
            }
        }

        if (reduce_blocking < jcp.reduce_dim) {
            if (jcp.prop_kind == backward_data)
                jcp.loop_order = reduce_src ? loop_lbr : loop_rlb;
            else
                jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
        }
        load_blocking = jcp.load_dim;

        /* Number of weight elements to be loaded for dest */
        int load_size = jcp.load_dim * jcp.reduce_dim;
        /* Number of elements to be broadcasted from src */
        auto bcast_size
                = (dim_t)jcp.mb * jcp.ngroups * jcp.bcast_dim * jcp.reduce_dim;

        /* 12 cores per CMG */
        if (jcp.nthr <= 12 && jcp.mb < jcp.nthr
                && nb_load * nb_bcast > jcp.nthr) {
            // Some heuristic here
            float calc_koef = 0.01, best_cost = FLT_MAX;
            int n_lgc = jcp.nthr;
            float ratio = (float)load_size / (float)bcast_size;
            int best_lgc = ratio > 1 ? n_lgc : 1;
            auto calc_job_cost = [&](int lb, int tg, float mem_k) {
                int bb_size = jcp.mb * div_up(nb_bcast, tg);
                float calc_size = (float)(bb_size * jcp.ur)
                        * (lb * jcp.load_block) * jcp.reduce_dim;
                float mem_size = (float)(bb_size * jcp.ur + lb * jcp.load_block)
                        * jcp.reduce_dim;
                return calc_koef * calc_size + mem_k * mem_size;
            };
            for (int lgc, ilgc = 0; ilgc < n_lgc; ilgc++) {
                lgc = ratio > 1 ? n_lgc - ilgc : ilgc + 1;
                int min_lb = nb_load / lgc;
                int max_lb = div_up(nb_load, lgc);
                int min_tg = jcp.nthr / lgc;
                int max_tg = div_up(jcp.nthr, lgc);
                // Some heuristic here
                float mem_koef = (max_tg == 1) ? 1.f : 1.3f;
                float job_cost = 0.;
                if (jcp.nthr % lgc < nb_load % lgc) {
                    job_cost = calc_job_cost(max_lb, min_tg, mem_koef);
                } else {
                    auto job_cost1 = calc_job_cost(max_lb, max_tg, mem_koef);
                    auto job_cost2 = calc_job_cost(min_lb, min_tg, mem_koef);
                    job_cost = nstl::max(job_cost1, job_cost2);
                }

                if (job_cost < best_cost) {
                    best_lgc = lgc;
                    best_cost = job_cost;
                }
            }
            jcp.load_grp_count = best_lgc;
            load_blocking
                    = div_up(nb_load, jcp.load_grp_count) * jcp.load_block;
        } else {
            jcp.load_grp_count
                    = div_up(jcp.nthr, jcp.mb * jcp.ngroups * nb_bcast);
            jcp.load_grp_count = best_divider(jcp.nthr, jcp.load_grp_count,
                    2 * jcp.load_grp_count, false);
        }
        if (jcp.expl_bcast && jcp.bcast_dim <= 64 && load_size >= L2_size) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
        } else if (jcp.bcast_dim <= 49 && jcp.mb <= jcp.nthr
                && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
            jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2);
            load_blocking = jcp.load_block;
        }

        auto get_thr_eff = [=](int load_chunk, int nthr) {
            int lgc = div_up(nb_load, load_chunk);
            int thr_per_grp = div_up(nthr, lgc);
            int bcast_per_thr
                    = div_up(jcp.mb * nb_bcast, thr_per_grp) * jcp.bcast_block;
            int load_per_thr = load_chunk * simd_w;
            float data_norm = (bcast_per_thr + load_per_thr) / 2.f;
            float data_eff
                    = (bcast_per_thr * load_per_thr) / (data_norm * data_norm);
            float thr_eff_over_grp
                    = (float)nstl::max(1, nthr / lgc) / div_up(nthr, lgc);
            float thr_eff_in_grp = ((float)jcp.mb * nb_bcast)
                    / rnd_up(jcp.mb * nb_bcast, thr_per_grp);
            float thr_eff = thr_eff_over_grp * thr_eff_in_grp;
            float load_eff = (float)nb_load / rnd_up(nb_load, lgc);
            float overall_eff = data_eff + thr_eff + load_eff;
            return overall_eff;
        };

        auto get_load_chunk = [=](int nthr) {
            float best_eff = -1.0f;
            int best_lgc = 1;
            float eff;

            for (int load_chunk = 1; load_chunk <= nb_load; load_chunk++) {
                int lgc = div_up(nb_load, load_chunk);
                if (lgc > nthr) continue;
                eff = get_thr_eff(load_chunk, nthr);
                if (eff > best_eff) {
                    best_eff = eff;
                    best_lgc = lgc;
                }
            }
            return best_lgc;
        };

        /* adjust the thread decomposition
         * to improve the thr_eff for small problem size
         * the threshold 8192 is empirical 
         * TODO: Threshold can be increase for init stride > 1*/
        if (sizeof(float) * bcast_size < 8192 && jcp.mb < jcp.nthr
                && nb_load * nb_bcast < jcp.nthr) {
            float best_thr_eff = -1.0f;
            float thr_eff = -1.0f;
            int overall_lgc = jcp.load_grp_count;
            int lgc = 1;
            int best_nthr = jcp.nthr;
            int end_nthr = with_groups ? jcp.ngroups : 1;
            for (int nthr = jcp.nthr / 2; nthr >= end_nthr; nthr--) {
                lgc = get_load_chunk(nthr);
                thr_eff = get_thr_eff(lgc, nthr);
                if (best_thr_eff < thr_eff) {
                    best_thr_eff = thr_eff;
                    overall_lgc = lgc;
                    best_nthr = nthr;
                }
            }
            jcp.nthr = best_nthr;
            jcp.load_grp_count = overall_lgc;
            load_blocking
                    = div_up(nb_load, jcp.load_grp_count) * jcp.load_block;
        }

        bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                                 div_up(jcp.nthr, jcp.load_grp_count))
                * jcp.bcast_block;
        bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
        bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

        int space_for_bcast = (L2_capacity - /* kernel_size - */
                2 * jcp.load_block * reduce_blocking - jcp.ur * reduce_blocking
                - 3 * 1024);
        if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity) space_for_bcast /= 2;

        int bcast_in_cache
                = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
        bcast_blocking = nstl::min(
                bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));
        // NHWC perf
        if (is_data_layout_nxc) bcast_blocking = jcp.bcast_block;

        load_blocking_max = load_blocking;
        bcast_blocking_max = bcast_blocking * 3 / 2;
        reduce_blocking_max = reduce_blocking;

        jcp.ur_tail = (jcp.with_dw_conv ? jcp.ow : jcp.bcast_dim) % jcp.ur;

    } else if (jcp.prop_kind == backward_weights) { /* BWD weight */

        jcp.reduce_dim = jcp.is;

        jcp.reduce_block = best_divider(jcp.reduce_dim, 7, 16, true);
        if (jcp.reduce_dim % jcp.reduce_block != 0)
            jcp.reduce_block = best_divider(jcp.iw, 4, jcp.iw, false);
        if (jcp.reduce_block > 256) { jcp.reduce_block = 1; }

        jcp.load_dim = jcp.oc;
        jcp.load_block = jcp.oc_block;

        jcp.bcast_dim = jcp.ic;
        jcp.bcast_block = jcp.ic_block;

        if (jcp.reduce_block <= 19 &&
                // maskrcnn optimization for nxc; don't reduce ur when ocb<=1
                !(is_data_layout_nxc && jcp.load_dim <= jcp.load_block)) {
            // if reduce_block is big then generated JIT code may be big
            // for small values of ur because reduce_loop_unroll = reduce_block
            jcp.ur = jcp.bcast_block / 2;
            jcp.expl_bcast = true;
        } else {
            jcp.ur = jcp.bcast_block;
            jcp.expl_bcast = false;
        }

        jcp.ur_tail = jcp.bcast_dim % jcp.bcast_block;
        jcp.reduce_loop_unroll = jcp.reduce_block;
        jcp.reduce_loop_bcast_step = jcp.typesize_in * jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? jcp.ic : jcp.ic_block);
        jcp.reduce_loop_load_step = jcp.typesize_in * jcp.reduce_loop_unroll
                * (is_data_layout_nxc ? jcp.oc : jcp.oc_block);

        jcp.bcast_loop_output_step
                = jcp.oc_block * jcp.ic_block * jcp.typesize_out;
        jcp.bcast_loop_output_substep
                = jcp.oc_block * jcp.ur * jcp.typesize_out;
        jcp.bcast_loop_bcast_step = jcp.ic_block
                * (is_data_layout_nxc ? 1
                                      : utils::rnd_up(jcp.reduce_dim,
                                                jcp.reduce_block))
                * jcp.typesize_in;
        jcp.bcast_loop_bcast_substep = jcp.ur * jcp.typesize_in;

        jcp.load_loop_load_step = jcp.typesize_in * jcp.oc_block
                * (is_data_layout_nxc ? 1 : jcp.os);
        jcp.load_loop_iter_step = jcp.oc_block;

        /* --- */
        balance(jcp);

        load_blocking = div_up(jcp.load_dim, jcp.load_block);
        load_blocking = best_divider(load_blocking, 16, load_blocking, false);
        load_blocking *= jcp.load_block;

        load_blocking_max = load_blocking;
        assert(IMPLICATION(
                !is_data_layout_nxc, jcp.load_dim % load_blocking == 0));

        int max_bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        int min_bcast_blocking = 5;

        bcast_blocking = div_up(jcp.bcast_dim, jcp.bcast_block);
        bcast_blocking = best_divider(
                bcast_blocking, min_bcast_blocking, max_bcast_blocking, false);
        bcast_blocking *= jcp.bcast_block;
        bcast_blocking_max = bcast_blocking;
        assert(IMPLICATION(
                !is_data_layout_nxc, jcp.bcast_dim % bcast_blocking == 0));

        // for reduction balance
        if (is_data_layout_nxc && jcp.reduce_dim >= BIG_SPATIAL * BIG_SPATIAL
                && jcp.load_dim >= BIG_LOAD_DIM / 2) {
            reduce_blocking = rnd_up(nstl::min(jcp.ow, 256), jcp.reduce_block);
        } else {
            int max_reduce_blocking
                    = nstl::min(L1_capacity / jcp.ur, jcp.reduce_dim);
            int min_reduce_blocking = nstl::min(
                    L1_capacity / jcp.ur, nstl::max(jcp.iw, jcp.ih));
            reduce_blocking = best_divider(jcp.reduce_dim, min_reduce_blocking,
                    max_reduce_blocking, true);
            reduce_blocking
                    = nstl::max(rnd_dn(reduce_blocking, jcp.reduce_block),
                            jcp.reduce_block);
        }

        reduce_blocking_max = rnd_dn(reduce_blocking * 3 / 2, jcp.reduce_block);
    } else {
        return status::unimplemented;
    }

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);

    if (!is_data_layout_nxc) {
        assert(load_blocking % jcp.load_block == 0);
        assert(reduce_blocking % jcp.reduce_block == 0);
        assert(load_blocking_max % jcp.load_block == 0);
        assert(reduce_blocking_max % jcp.reduce_block == 0);
        assert(jcp.reduce_dim % jcp.reduce_block == 0);
    }

    assert(jcp.bcast_block % jcp.ur == 0);

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = utils::div_up(load_blocking, jcp.load_block);
    jcp.nb_load_blocking_max = utils::div_up(load_blocking_max, jcp.load_block);
    jcp.nb_reduce_blocking = utils::div_up(reduce_blocking, jcp.reduce_block);
    jcp.nb_reduce_blocking_max
            = utils::div_up(reduce_blocking_max, jcp.reduce_block);

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);
    return status::success;
}
template <cpu_isa_t isa>
void jit_sve_1x1_conv_kernel_t<isa>::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp) {

    using namespace dnnl::impl::memory_tracking::names;

    // Fox nxc layout bias is padded only for bwd_wb direction, as  bias
    // reduction kernels can't handle tails yet.
    if (jcp.with_bias && jcp.prop_kind != backward_data
            && (jcp.oc != jcp.oc_without_padding // blocked layout
                    || (jcp.prop_kind == backward_weights // nxc layout
                            && jcp.oc % jcp.oc_block != 0))) {

        const size_t nelems_padded_bias
                = jcp.ngroups * utils::rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book(
                key_conv_padded_bias, nelems_padded_bias, jcp.typesize_out);
    }

    if (jcp.prop_kind == backward_weights) {
        const size_t wei_size = (size_t)jcp.ngroups
                * rnd_up(jcp.oc, jcp.oc_block) * rnd_up(jcp.ic, jcp.ic_block);
        scratchpad.book(key_conv_wei_reduction, wei_size * (jcp.nthr_mb - 1),
                jcp.typesize_out);
    }
}

/* BWD W*/
template <cpu_isa_t isa>
void jit_sve_1x1_conv_kernel_t<isa>::balance(jit_1x1_conv_conf_t &jcp) {
    int nthreads = jcp.nthr;
    // initialize jcp reduction threading properties
    jcp.nthr = jcp.nthr_mb = jcp.nthr_g = jcp.nthr_oc_b = jcp.nthr_ic_b = 1;
    if (nthreads < jcp.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        return;
    }
    const int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    const int nb_load = div_up(jcp.load_dim, jcp.load_block);
    const int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    jcp.nthr_g = jcp.ngroups;
    const int nthr = nthreads / jcp.nthr_g;

    auto calc_mem_cost = [=](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level
        * optimizer tries to minimize memory consumption. few notes: (n1)
        * unclear why, but that essentially helps first convolution...
        *  (n2) assuming the reduction over minibatch is always there:
        *    - instead of 8 it should be 5 here (write ~= 2 read):
        *      kernel: temporal workspace 1 write
        *      reduction: 1 read from workspace and 1 write to the diff_wei
        *    - but experiments showed 8 works better than 5 or 6... */
        int bcast_koeff = 1;
        int load_koeff = 1;
        int output_koeff = 12;
        return 0
                + (size_t)bcast_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_bcast, nthr_ic_b)
                * jcp.ic_block * jcp.reduce_block / jcp.stride_h
                / jcp.stride_w /* (n1) */
                + (size_t)load_koeff * div_up(jcp.mb * nb_reduce, nthr_mb)
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
                * jcp.oc_block * jcp.reduce_block
                + (size_t)output_koeff /* (n2) */
                * div_up(jcp.ngroups, jcp.nthr_g) * div_up(nb_load, nthr_oc_b)
                * div_up(nb_bcast, nthr_ic_b) * jcp.ic_block * jcp.oc_block;
    };

    int nthr_mb = 1, nthr_oc_b = 1, nthr_ic_b = 1;
    auto best_mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = nstl::min(nthr, jcp.mb * nb_reduce);
    for (nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max = nstl::min(nthr_par, nb_load);
        for (nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            nthr_ic_b = nstl::min(nthr_par / nthr_oc_b, nb_bcast);
            auto mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                jcp.nthr_mb = nthr_mb;
                jcp.nthr_oc_b = nthr_oc_b;
                jcp.nthr_ic_b = nthr_ic_b;
            }
        }
    }
    if (jcp.nthr_mb > nthreads / 2 && jcp.nthr_mb < nthreads)
        jcp.nthr_mb = nstl::min(jcp.mb, nthreads);

    jcp.nthr = jcp.nthr_mb * jcp.nthr_g * jcp.nthr_oc_b * jcp.nthr_ic_b;
    assert(jcp.nthr <= nthreads);
}

template struct jit_sve_1x1_conv_kernel_t<sve_512>;
template struct jit_sve_1x1_conv_kernel_t<sve_256>;
template struct jit_sve_1x1_conv_kernel_t<sve_128>;
} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
