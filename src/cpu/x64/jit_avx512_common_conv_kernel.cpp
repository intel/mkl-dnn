/*******************************************************************************
* Copyright 2016 Intel Corporation
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
#include "common/math_utils.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/platform.hpp"
#include "cpu/x64/cpu_barrier.hpp"
#include "cpu/x64/injectors/injector_utils.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/x64/jit_avx512_common_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_conv_args_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

using namespace dnnl::impl::format_tag;
using namespace dnnl::impl::memory_tracking::names;
using namespace dnnl::impl::utils;
using namespace Xbyak;

namespace {

constexpr auto small_spatial = 14;

inline void pick_loop_order(jit_conv_conf_t &jcp) {
    using namespace prop_kind;
    assert(one_of(
            jcp.prop_kind, forward_training, forward_inference, backward_data));
    auto w = (jcp.prop_kind == backward_data) ? jcp.iw : jcp.ow;
    auto h = (jcp.prop_kind == backward_data) ? jcp.ih : jcp.oh;

    // The w in the loop order is currently ignored by 3D BWD_D
    jcp.loop_order = (w <= small_spatial && h <= small_spatial) ? loop_cwgn
                                                                : loop_gncw;
    if (utils::one_of(jcp.src_tag, format_tag::ndhwc, format_tag::nhwc,
                format_tag::nwc)
            && jcp.ngroups > 1 && jcp.oc < 16)
        jcp.loop_order = loop_nhwcg;
}

inline status_t init_tag(format_tag_t &tag, memory_desc_t &md,
        const memory_desc_wrapper &mdw, const format_tag_t tag_value) {
    if (mdw.format_kind() == format_kind::any) {
        CHECK(memory_desc_init_by_tag(md, tag_value));
        tag = tag_value;
    } else {
        tag = mdw.matches_one_of_tag(tag_value);
    }

    if (tag != tag_value) return status::unimplemented;

    return status::success;
}

inline bool is_1stconv(const jit_conv_conf_t &jcp) {
    if (mayiuse(avx512_core))
        return (jcp.ic < 16 && jcp.ngroups == 1);
    else
        return one_of(jcp.ic, 1, 3);
}

inline bool is_ow_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_ow > 1);
}

inline bool is_iw_threading_on(const jit_conv_conf_t &jcp) {
    return (jcp.nb_iw > 1);
}

} // namespace

template <typename Vmm>
jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::
        jit_avx512_common_conv_fwd_kernel_vmm_t(const jit_conv_conf_t &ajcp,
                const primitive_attr_t &attr, const memory_desc_t &dst_md)
    : jit_generator_t(jit_name(), avx512_core), jcp(ajcp), attr_(attr) {
    if (jcp.with_eltwise || jcp.with_binary) {
        using namespace binary_injector;
        static constexpr bool preserve_gpr = true;
        static constexpr bool preserve_vmm = false;
        static constexpr size_t helper_vmm_idx = 31;
        const int tail_size = jcp.oc_without_padding % isa_simd_width_;
        static constexpr bool use_exact_tail_scalar_bcast = false;

        const binary_injector::rhs_arg_static_params_t rhs_args_static_params {
                helper_vmm_idx, reg_tmp, r15, r14, preserve_gpr, preserve_vmm,
                GET_OFF(post_ops_binary_rhs_arg_vec), GET_OFF(dst_orig),
                memory_desc_wrapper(dst_md), tail_size, postops_mask,
                use_exact_tail_scalar_bcast};
        const binary_injector::static_params_t static_params {
                this->param1, rhs_args_static_params};

        postops_injector_ = utils::make_unique<
                injector::jit_uni_postops_injector_t<Xbyak::Zmm>>(
                // Sum is folded into accumulator initialization.
                this, jcp.post_ops, static_params, /* inject_sum = */ false);
    }
}

template <typename Vmm>
void jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::prepare_output(int ur_w) {
    if (jcp.is_g_packed) {
        for (int j = 0; j < ur_w; j++)
            for (int k = 0; k < g_pack_oc; k++) {
                Vmm vmm = vmm_g_pack_acc(j, k);
                vpxord(vmm, vmm, vmm);
            }
        return;
    }
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            vpxord(vmm, vmm, vmm);
        }
}

// Sums the accumulators of each output point over ic. Within each 128-bit lane
// it transposes 4 accumulators (one per oc, lanes hold ic) and adds the rows,
// which gives lanes (group, oc) in vmm_out(i_ur, 0).
template <typename Vmm>
void jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::reduce_g_packed_output(
        int ur_w) {
    const Vmm t0(28), t1(29), t2(30), t3(31);
    // vmm_out(j, 0) overwrites accumulators of points <= j only, which are
    // already reduced or are read before being written.
    for (int j = 0; j < ur_w; j++) {
        vunpcklps(t0, vmm_g_pack_acc(j, 0), vmm_g_pack_acc(j, 1));
        vunpckhps(t1, vmm_g_pack_acc(j, 0), vmm_g_pack_acc(j, 1));
        vaddps(t0, t0, t1);
        vunpcklps(t2, vmm_g_pack_acc(j, 2), vmm_g_pack_acc(j, 3));
        vunpckhps(t3, vmm_g_pack_acc(j, 2), vmm_g_pack_acc(j, 3));
        vaddps(t2, t2, t3);
        vshufps(t1, t0, t2, 0x44);
        vshufps(t3, t0, t2, 0xee);
        vaddps(vmm_out(j, 0), t1, t3);
    }
}

template <typename F>
static void iterate(const int nb_oc_blocking, const int ur_w,
        const bool oc_tail, const bool force_masking, const F &fun) {
    for (int i_load = 0; i_load < nb_oc_blocking; i_load++) {
        const auto mask_flag
                = force_masking || (oc_tail && i_load + 1 == nb_oc_blocking);
        for (int i_ur = 0; i_ur < ur_w; i_ur++)
            fun(mask_flag, i_load, i_ur);
    }
}
template <typename F>
static void iterate(const int nb_oc_blocking, const int ur_w, const F &fun) {
    iterate(nb_oc_blocking, ur_w, false, false, fun);
}

template <typename Vmm>
void jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::apply_postops(int ur_w) {
    injector_utils::vmm_index_set_t vmm_idxs;
    if (jcp.with_binary) {
        binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
        const bool mask_tail = jcp.oc_without_padding % jcp.simd_w;
        const bool oc_blk_is_smaller_than_vmm = jcp.oc_block < isa_simd_width_;
        iterate(jcp.nb_oc_blocking, ur_w, mask_tail, oc_blk_is_smaller_than_vmm,
                [&](const bool mask_flag, const int i_load, const int i_ur) {
            const size_t aux_output_l_off = get_output_offset(i_ur, i_load);
            const auto vmm_idx = vmm_out_idx(i_ur, i_load);
            vmm_idxs.emplace(vmm_idx);

            rhs_arg_params.vmm_idx_to_out_reg.emplace(vmm_idx, reg_out);
            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                    vmm_idx, aux_output_l_off);
            if (mask_flag) { rhs_arg_params.vmm_tail_idx_.emplace(vmm_idx); }
        });

        postops_injector_->compute_vector_range(vmm_idxs, rhs_arg_params);
    } else {
        iterate(jcp.nb_oc_blocking, ur_w,
                [&](const bool, const int i_load, const int i_ur) {
            vmm_idxs.emplace(vmm_out_idx(i_ur, i_load));
        });
        postops_injector_->compute_vector_range(vmm_idxs);
    }
}

template <typename Vmm>
void jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::store_output(int ur_w) {
    Label no_update_label, store_label, post_ops_label;

    mov(reg_channel, ptr[param1 + GET_OFF(flags)]);

    if (jcp.with_bias) { mov(reg_bias, ptr[param1 + GET_OFF(bias)]); }
    const int oc_tail = jcp.oc_tail;

    if (!jcp.with_sum) {
        test(reg_channel, FLAG_IC_FIRST);
        jnz(no_update_label, T_NEAR);
    }

    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            // mask only needed for last oc_block
            if (oc_tail && k + 1 == jcp.nb_oc_blocking)
                vmm = vmm | k_oc_tail_mask | T_z;
            size_t aux_output_offset = get_output_offset(j, k);
            vaddps(vmm,
                    make_safe_addr(
                            reg_out, aux_output_offset, reg_out_long_offt));
        }

    if (!jcp.with_sum) {
        jmp(post_ops_label, T_NEAR);
    } else {
        test(reg_channel, FLAG_IC_FIRST);
        jz(post_ops_label, T_NEAR);
    }

    L(no_update_label);
    if (jcp.with_bias) {
        for (int k = 0; k < jcp.nb_oc_blocking; k++) {
            const dim_t bias_offset = jcp.typesize_out * k * jcp.oc_block;
            for (int j = 0; j < ur_w; j++) {
                Vmm vmm = vmm_out(j, k);
                // mask only needed for last oc_block
                if (oc_tail && k + 1 == jcp.nb_oc_blocking)
                    vmm = vmm | k_oc_tail_mask | T_z;
                vaddps(vmm, EVEX_compress_addr(reg_bias, bias_offset));
            }
        }
    }

    L(post_ops_label);

    if (jcp.with_eltwise || jcp.with_binary) {
        test(reg_channel, FLAG_IC_LAST);
        jz(store_label, T_NEAR);

        apply_postops(ur_w);
    }

    L(store_label);

    const auto is_padding = jcp.oc_without_padding != jcp.oc;
    for (int k = 0; k < jcp.nb_oc_blocking; k++)
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            // mask only needed for last oc_block
            if (oc_tail && k + 1 == jcp.nb_oc_blocking) {
                if (is_padding)
                    vmovups(vmm | k_oc_tail_mask | T_z, vmm);
                else
                    vmm = vmm | k_oc_tail_mask;
            }
            size_t aux_output_offset = get_output_offset(j, k);

            vmovups(EVEX_compress_addr_safe(
                            reg_out, aux_output_offset, reg_out_long_offt),
                    vmm);
        }
}

template <typename Vmm>
void jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::compute_loop_fma(
        int ur_w, dim_t pad_l, dim_t pad_r) {
    const bool is_source_layout_nxc = is_src_layout_nxc();
    const bool icb_loop_in_compute_function = is_source_layout_nxc;
    const int ic_tail = jcp.ic_tail;
    const int oc_tail = jcp.oc == jcp.oc_without_padding ? jcp.oc_tail : 0;
    dim_t iw = jcp.iw;
    const dim_t kw = jcp.kw;
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const int nb_oc_block = jcp.nb_oc_blocking;
    Label kh_label, kd_label;
    std::vector<Label> ic_tail_jmp(kw);

    // It seems that this compute_loop currently only handles one block of oc.
    // assert if it is extended in future to catch unpadded_oc_tail.
    assert(IMPLICATION(oc_tail, nb_oc_block == 1));

    const dim_t num_ker_loads = ic_block * nb_oc_block * kw;
    int ker_pipeline_depth = oc_tail || ic_tail
            ? 1
            : static_cast<int>(nstl::min<dim_t>(4, num_ker_loads));
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    dim_t inp_mul = is_source_layout_nxc ? jcp.ngroups * jcp.ic
                                         : (!jcp.is_1stconv ? ic_block : 1);

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        if (icb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            assert(aux_reg_ker_d == reg_ker);
            push(aux_reg_ker_d);
        } else
            mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);
        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    align(16);
    L(kh_label);
    {
        int step = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int ic = 0; ic < ic_block; ic++) {
                if (ic_tail && ic >= ic_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.ic == ic_tail)
                        break;
                    else if (ic == ic_tail) {
                        cmp(reg_channel, ic_tail);
                        je(ic_tail_jmp[ki], T_NEAR);
                    }
                }
                dim_t aux_kernel_offset = 0;
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        aux_kernel_offset = get_kernel_offset(ki, ic, 0, i);
                        vmovups(vmm_ker(i),
                                EVEX_compress_addr(
                                        aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    const dim_t load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                            = (step + load_offset) % ker_pipeline_depth;
                    aux_kernel_offset
                            = get_kernel_offset(ki, ic, 0, load_offset);
                    vmovups(vmm_ker(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }

                Vmm vmm_kernel = vmm_ker(step % ker_pipeline_depth);
                const int j_start = get_ow_start(ki, pad_l);
                const int j_end = get_ow_end(ur_w, ki, pad_r);
                for (int j = j_start; j < j_end; j++) {
                    const dim_t aux_input_offset
                            = get_input_offset(ki, ic, j, pad_l);
                    auto addr = EVEX_compress_addr_safe(
                            aux_reg_inp, aux_input_offset, reg_long_offt, true);
                    vfmadd231ps(vmm_out(j, 0), vmm_kernel, addr);
                }
                step++;
            }
            L(ic_tail_jmp[ki]);
        }
        const dim_t ker_shift = jcp.typesize_in * kw * oc_block * ic_block;
        add(aux_reg_ker, ker_shift);
        const dim_t inp_shift
                = jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul;
        add(aux_reg_inp, inp_shift);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        const dim_t inp_shift
                = typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul;
        add(aux_reg_inp_d, inp_shift);
        const dim_t ker_shift
                = typesize * jcp.kw * jcp.kh * jcp.oc_block * jcp.ic_block;
        add(aux_reg_ker_d, ker_shift);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        if (icb_loop_in_compute_function) pop(aux_reg_ker_d);
        pop(reg_out);
    }
}

template <typename Vmm>
void jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::compute_loop_fma_core(
        int ur_w, dim_t pad_l, dim_t pad_r) {
    const dim_t kw = jcp.kw;
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const int nb_oc_block = jcp.nb_oc_blocking;
    const bool is_source_layout_nxc = is_src_layout_nxc();
    const bool icb_loop_in_compute_function = is_source_layout_nxc;
    const int ic_tail = jcp.ic_tail;

    Label kh_label, kd_label;
    std::vector<Label> ic_tail_jmp(kw);
    dim_t shift_kernel_ptr = jcp.typesize_in * jcp.kw * ker_point_size();
    dim_t inp_mul = is_source_layout_nxc ? jcp.ngroups * jcp.ic
                                         : (!jcp.is_1stconv ? ic_block : 1);

    dim_t shift_input_ptr
            = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * inp_mul;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_inp, reg_inp);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_out);

        mov(reg_ki, ptr[param1 + GET_OFF(kd_padding)]);
        if (icb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            assert(aux_reg_ker_d == reg_ker);
            push(aux_reg_ker_d);
        } else
            mov(aux_reg_ker_d, ptr[param1 + GET_OFF(filt)]);

        mov(aux_reg_inp_d, reg_inp);

        L(kd_label);
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_inp, aux_reg_inp_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            const int jj_start = get_ow_start(ki, pad_l);
            const int jj_end = get_ow_end(ur_w, ki, pad_r);
            if (jcp.is_g_packed) {
                assert(ic_tail == 0 && nb_oc_block == 1);
                if (jj_end - jj_start <= 0) continue;
                for (int k = 0; k < g_pack_oc; k++)
                    vmovups(vmm_g_pack_wei(k),
                            EVEX_compress_addr(aux_reg_ker,
                                    jcp.typesize_in
                                            * (ki * ker_point_size()
                                                    + k * oc_block)));
                for (int jj = jj_start; jj < jj_end; jj++) {
                    vmovups(vmm_g_pack_inp,
                            EVEX_compress_addr_safe(aux_reg_inp,
                                    get_input_offset(ki, 0, jj, pad_l),
                                    reg_long_offt));
                    for (int k = 0; k < g_pack_oc; k++)
                        vfmadd231ps(vmm_g_pack_acc(jj, k), vmm_g_pack_wei(k),
                                vmm_g_pack_inp);
                }
                continue;
            }
            for (int ic = 0; ic < ic_block; ic++) {
                if (ic_tail && ic >= ic_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.ic == ic_tail)
                        break;
                    else if (ic == ic_tail) {
                        cmp(reg_channel, ic_tail);
                        je(ic_tail_jmp[ki], T_NEAR);
                    }
                }
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        dim_t aux_input_offset
                                = get_input_offset(ki, ic, jj, pad_l);
                        vbroadcastss(vmm_inp(jj, nb_oc_block),
                                EVEX_compress_addr_safe(aux_reg_inp,
                                        aux_input_offset, reg_long_offt));
                    }
                }
                for (int ii = 0; ii < nb_oc_block; ii++) {
                    dim_t aux_kernel_offset = jcp.typesize_in
                            * (ii * jcp.nb_ic * jcp.kh * jcp.kw * jcp.kd
                                            * ic_block * oc_block
                                    + ki * ic_block * oc_block + ic * oc_block);
                    if (jj_end - jj_start > 0)
                        vmovups(vmm_wei,
                                EVEX_compress_addr(
                                        aux_reg_ker, aux_kernel_offset));
                    for (int jj = jj_start; jj < jj_end; jj++)
                        if (jcp.kernel_kind == expl_bcast)
                            vfmadd231ps(vmm_out(jj, ii),
                                    vmm_inp(jj, nb_oc_block), vmm_wei);
                        else {
                            dim_t aux_input_offset
                                    = get_input_offset(ki, ic, jj, pad_l);
                            vfmadd231ps(vmm_out(jj, ii), vmm_wei,
                                    EVEX_compress_addr_safe(aux_reg_inp,
                                            aux_input_offset, reg_long_offt,
                                            true));
                        }
                }
            }
            L(ic_tail_jmp[ki]);
        }
        add(aux_reg_ker, shift_kernel_ptr);
        add(aux_reg_inp, shift_input_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_inp_d,
                typesize * (jcp.dilate_d + 1) * jcp.ih * jcp.iw * inp_mul);
        const dim_t ker_shift = typesize * jcp.kw * jcp.kh * ker_point_size();
        add(aux_reg_ker_d, ker_shift);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        if (icb_loop_in_compute_function) pop(aux_reg_ker_d);
        pop(reg_out);
    }
}

template <typename Vmm>
void jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::compute_loop(
        int ur_w, dim_t pad_l, dim_t pad_r) {
    if (jcp.ndims == 5) push(reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        if ((jcp.dilate_d >= jcp.id)
                || (jcp.kd - 1) * (jcp.dilate_d + 1)
                        < nstl::max(jcp.f_pad, jcp.back_pad)) {
            mov(reg_kj, ptr[param1 + GET_OFF(kd_padding)]);
            cmp(reg_kj, 0);
            jle(skip_compute_loop, T_NEAR);
        }
    }
    if ((jcp.dilate_h >= jcp.ih)
            || (jcp.kh - 1) * (jcp.dilate_h + 1)
                    < nstl::max(jcp.t_pad, jcp.b_pad)) {
        mov(reg_kj, ptr[param1 + GET_OFF(kh_padding)]);
        cmp(reg_kj, 0);
        jle(skip_compute_loop, T_NEAR);
    }

    Label ic_loop;
    const bool generate_icb_loop = jcp.nb_ic > 1 && is_src_layout_nxc();
    if (generate_icb_loop) {
        push(reg_inp);
        push(reg_ker);

        mov(reg_channel, ptr[param1 + GET_OFF(reduce_work)]);
        L(ic_loop);
    }

    if (jcp.is_1stconv && jcp.kernel_kind != expl_bcast)
        compute_loop_fma(ur_w, pad_l, pad_r);
    else if (jcp.kernel_kind == embd_bcast && jcp.nb_oc_blocking == 1)
        compute_loop_fma(ur_w, pad_l, pad_r);
    else
        compute_loop_fma_core(ur_w, pad_l, pad_r);

    if (generate_icb_loop) {
        assert(is_src_layout_nxc());
        const dim_t inp_shift = jcp.ic_block * jcp.typesize_in;
        add(reg_inp, inp_shift);
        const size_t ker_shift = (size_t)jcp.kd * jcp.kh * jcp.kw * jcp.ic_block
                * jcp.oc_block * jcp.typesize_in;
        safe_add(reg_ker, ker_shift, reg_ker_long_offt);
        sub(reg_channel, jcp.ic_block);
        jg(ic_loop, T_NEAR);

        pop(reg_ker);
        pop(reg_inp);
    }

    if (jcp.is_g_packed) reduce_g_packed_output(ur_w);

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) pop(reg_oi);
}

template <typename Vmm>
void jit_avx512_common_conv_fwd_kernel_vmm_t<Vmm>::generate() {
    const dim_t iw = jcp.iw;
    const dim_t ow = jcp.ow;
    dim_t ow_block = jcp.ow_block;
    dim_t nb_ow = jcp.nb_ow;
    const dim_t kw = jcp.kw;
    const dim_t l_pad = jcp.l_pad;
    int ur_w = jcp.ur_w;
    int ur_w_tail = jcp.ur_w_tail;
    const dim_t stride_w = jcp.stride_w;

    dim_t inp_mult = is_src_layout_nxc() ? jcp.ngroups * jcp.ic
                                         : (jcp.is_1stconv ? 1 : jcp.ic_block);
    dim_t inp_shift_pad
            = jcp.typesize_in * (ur_w * stride_w - l_pad) * inp_mult;
    dim_t inp_shift = jcp.typesize_in * ur_w * stride_w * inp_mult;
    dim_t inp_shift_pad_second_block = -1 * jcp.typesize_in * l_pad * inp_mult;
    dim_t out_shift = jcp.typesize_out * ur_w
            * (is_dst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block);

    preamble();
    mov(reg_inp, ptr[param1 + GET_OFF(src)]);
    mov(reg_out, ptr[param1 + GET_OFF(dst)]);
    mov(reg_ker, ptr[param1 + GET_OFF(filt)]);
    mov(reg_kh, ptr[param1 + GET_OFF(kh_padding)]);

    const int oc_tail = jcp.oc_tail;
    if (oc_tail) {
        Label done;
        // dummy mask all 1's
        kxnorw(k_oc_tail_mask, k_oc_tail_mask, k_oc_tail_mask);
        mov(reg_load_work, ptr[param1 + GET_OFF(load_work)]);
        cmp(reg_load_work, jcp.nb_oc_blocking * jcp.oc_block);
        je(done, T_NEAR);
        Reg32 reg_tail_32 = reg_tail.cvt32();
        mov(reg_tail_32, (1 << oc_tail) - 1);
        kmovw(k_oc_tail_mask, reg_tail_32);
        L(done);
        kmovw(postops_mask, k_oc_tail_mask);
    } else if (jcp.with_binary)
        if (jcp.oc_block != isa_simd_width_) {
            const int mask = (1 << jcp.oc_block) - 1;
            const Reg32 reg_tail_32 = reg_tail.cvt32();
            mov(reg_tail_32, mask);
            kmovw(postops_mask, reg_tail_32);
        }

    const dim_t r_pad = nstl::max<dim_t>(0, jcp.r_pad);
    dim_t n_oi = ow / ur_w;
    const dim_t r_pad1 = calculate_end_padding(l_pad, ur_w * n_oi, iw, stride_w,
            calculate_extended_filter_size(kw, jcp.dilate_w));

    if (!is_ow_threading_on(jcp)) {
        // ow is being processed as a whole - with left and right paddings
        if (r_pad1 > 0) n_oi--;

        if (ow == ur_w) {
            compute_loop(ur_w, l_pad, r_pad);
        } else {
            if (n_oi == 0) {
                compute_loop(ur_w, l_pad, r_pad1);
                add(reg_inp, inp_shift_pad);
                add(reg_out, out_shift);
                if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
            } else {
                xor_(reg_oi, reg_oi);
                if (l_pad > 0) {
                    compute_loop(ur_w, l_pad, 0);
                    add(reg_inp, inp_shift_pad);
                    add(reg_out, out_shift);
                    inc(reg_oi);
                }
                if ((l_pad <= 0 && n_oi > 0) || (l_pad > 0 && n_oi > 1)) {
                    Label ow_loop_label;
                    L(ow_loop_label);
                    {
                        compute_loop(ur_w, 0, 0);
                        add(reg_inp, inp_shift);
                        add(reg_out, out_shift);
                        inc(reg_oi);
                        cmp(reg_oi, n_oi);
                        jl(ow_loop_label, T_NEAR);
                    }
                }
                if (r_pad1 > 0) {
                    compute_loop(ur_w, 0, r_pad1);
                    add(reg_inp, inp_shift);
                    add(reg_out, out_shift);
                }
                if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
            }
        }
    } else {
        // ow block is only processed.
        // Number of block is passed as parameter owb,
        // and padding processing depends on this number.

        Label end_label, last_oi_label, middle_ow_blocks_label, tail_label;
        Label oi_loop_label, oi_loop_start_label, oi_loop_end_label;

        assert(ow_block % ur_w == 0);
        int n_oi_not_last_ow_block = static_cast<int>(ow_block / ur_w);
        // to simplify code (and general regs usage),
        // size of ow block must be >= 2 * ur_w
        assert(n_oi_not_last_ow_block > 1);
        int n_oi_next_last_ow_block = n_oi_not_last_ow_block;
        int n_oi_first_ow_block = n_oi_not_last_ow_block;

        int n_oi_last_ow_block
                = static_cast<int>((ow - ow_block * (nb_ow - 1)) / ur_w);

        // prepare right padding
        bool next_last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block == 0;
        bool first_ow_block_padded
                = next_last_ow_block_padded && jcp.nb_ow == 2;
        bool last_ow_block_padded = r_pad1 > 0 && n_oi_last_ow_block > 0;

        if (last_ow_block_padded)
            n_oi_last_ow_block--;
        else if (first_ow_block_padded)
            n_oi_first_ow_block--;
        else if (next_last_ow_block_padded)
            n_oi_next_last_ow_block--;

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, 0); // is that the first ow-block ?
        jg(middle_ow_blocks_label, T_NEAR);

        // the first ow block, compute left padding

        mov(reg_oi, n_oi_first_ow_block);

        if (l_pad > 0) {
            compute_loop(ur_w, l_pad, 0);
            add(reg_inp, inp_shift_pad);
            add(reg_out, out_shift);
            dec(reg_oi);
        }
        jmp(oi_loop_label, T_NEAR);

        // middle or last ow block entry

        L(middle_ow_blocks_label);

        if (l_pad > 0) {
            // just to consider left padding, not compute
            add(reg_inp, inp_shift_pad_second_block);
        }

        // set number of iteration for oi-loop
        cmp(reg_owb, jcp.nb_ow - 1); // last ow-block ?
        mov(reg_oi, n_oi_last_ow_block);
        je(oi_loop_label, T_NEAR);
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        mov(reg_oi, n_oi_next_last_ow_block);
        je(oi_loop_label, T_NEAR);
        mov(reg_oi, n_oi_not_last_ow_block); // other middle ow-blocks

        // oi loop w/o padding
        L(oi_loop_label);
        L(oi_loop_start_label);
        cmp(reg_oi, 0);
        jle(oi_loop_end_label, T_NEAR);

        compute_loop(ur_w, 0, 0);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);
        dec(reg_oi);
        jmp(oi_loop_start_label, T_NEAR);
        L(oi_loop_end_label);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);

        cmp(reg_owb, 0); // first ow-block ?
        if (first_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        cmp(reg_owb, jcp.nb_ow - 2); // next to last ow-block ?
        jl(end_label, T_NEAR);
        if (next_last_ow_block_padded) {
            je(last_oi_label, T_NEAR);
        } else {
            je(end_label, T_NEAR);
        }
        // that is last block
        if (!last_ow_block_padded) { jmp(tail_label, T_NEAR); }

        // last oi block with right padding
        L(last_oi_label);
        compute_loop(ur_w, 0, r_pad1);
        add(reg_inp, inp_shift);
        add(reg_out, out_shift);

        mov(reg_owb, ptr[param1 + GET_OFF(owb)]);
        cmp(reg_owb, jcp.nb_ow - 1); // last ow_block?
        jl(end_label, T_NEAR);

        L(tail_label);
        if (ur_w_tail != 0) { compute_loop(ur_w_tail, 0, r_pad); }
        L(end_label);
    }
    postamble();

    if (jcp.with_eltwise)
        postops_injector_->prepare_table(/* generate = */ true);
}

status_t jit_avx512_common_conv_fwd_kernel_t::init_conf(jit_conv_conf_t &jcp,
        const convolution_desc_t &cd, memory_desc_t &src_md,
        memory_desc_t &weights_md, memory_desc_t &dst_md,
        memory_desc_t &bias_md, primitive_attr_t &attr, int nthreads) {
    using namespace prop_kind;

    if (!mayiuse(avx512_core)) return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    VDISPATCH_CONV_IC(everyone_is(data_type::f32, src_d.data_type(),
                              weights_d.data_type(), dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT_CFG);
    // Big int (> INT_MAX) values are unsupported and jcp fields may overflow
    // TODO: change data type of jcp fields to size_t
    VDISPATCH_CONV_IC(!has_large_size(cd, src_d, weights_d, dst_d),
            VERBOSE_BAD_PARAM, "large size is not supported");

    const int regs = 28;
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();
    jcp.nthr = jcp.aligned_threads = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;
    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];
    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];
    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];
    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    const dim_t ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const dim_t ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    const dim_t ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    VDISPATCH_CONV_IC(!kernel_outside_src, VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "weights and src size mismatch");

    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx4c = pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = src_d.mb_stride_relaxed_match(dat_tag_nxc,
            dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c, dat_tag_ncx);
    auto curr_dst_tag = dst_d.mb_stride_relaxed_match(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    bool is_data_layout_nxc = IMPLICATION(curr_src_tag != dat_tag_nxc,
                                      src_d.format_kind() == format_kind::any)
            && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                    dst_d.format_kind() == format_kind::any)
            && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);

    jcp.is_1stconv = is_1stconv(jcp);

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1
            && src_d.data_type() == data_type::f32;

    const int full_simd_w = cpu_isa_traits_t<avx512_core>::vlen / typesize;
    jcp.simd_w = full_simd_w;
    bool ok_to_try_lower_zmm = true
            && IMPLICATION(is_data_layout_nxc,
                    jcp.oc < full_simd_w && jcp.ic < full_simd_w
                            && jcp.ngroups > 1)
            && mayiuse(avx512_core) && src_d.data_type() == data_type::f32
            && !jcp.is_1stconv && !ok_to_pad_channels
            && (jcp.ic % jcp.simd_w != 0 || jcp.oc % jcp.simd_w != 0);

    if (ok_to_try_lower_zmm) {
        for (auto simd : {8, 4}) {
            if (jcp.ic % simd == 0 && jcp.oc % simd == 0) {
                jcp.simd_w = simd;
                break;
            }
        }
    }
    // There is no block-4 kernel. With nxc, use a wider block and handle the
    // channels as its tail. Size it by oc, which is kept in a single block.
    if (is_data_layout_nxc && jcp.simd_w == 4)
        jcp.simd_w = jcp.oc <= 8 ? 8 : 16;

    // Groups with 4 channels: compute pairs of groups as one group with
    // 8 channels in a Ymm. Lanes hold (group, ic), each output point
    // accumulates 4 vectors (one per oc) that are reduced over ic at the end.
    // It needs weights blocked as [g/2][kd][kh][kw][4o][2g][4i].
    memory_desc_t g_packed_wei_md = weights_md;
    {
        const int wei_ndims = weights_md.ndims;
        blocking_desc_t blk = {};
        blk.inner_nblks = 3;
        blk.inner_blks[0] = 4;
        blk.inner_idxs[0] = 1;
        blk.inner_blks[1] = 2;
        blk.inner_idxs[1] = 0;
        blk.inner_blks[2] = 4;
        blk.inner_idxs[2] = 2;
        for (int d = 0; d < wei_ndims; d++)
            blk.strides[d] = wei_ndims - d;
        if (with_groups)
            CHECK(memory_desc_init_by_blocking_desc(g_packed_wei_md, blk));
    }
    const dim_t g_packed_ur_w = nstl::min<dim_t>(jcp.ow, 6);
    jcp.is_g_packed = is_data_layout_nxc && with_groups && jcp.ngroups % 2 == 0
            && jcp.ic == 4 && jcp.oc == 4
            && IMPLICATION(weights_d.format_kind() != format_kind::any,
                    weights_md == g_packed_wei_md)
            && jcp.l_pad <= g_packed_ur_w
            && calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ow % g_packed_ur_w,
                       jcp.iw, jcp.stride_w, ext_kw)
                    <= g_packed_ur_w;
    if (jcp.is_g_packed) {
        jcp.ngroups /= 2;
        jcp.ic = jcp.ic_without_padding = 8;
        jcp.oc = jcp.oc_without_padding = 8;
        jcp.simd_w = 8;
    }

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }
    VDISPATCH_CONV_IC(
            IMPLICATION(!is_data_layout_nxc,
                    jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0),
            VERBOSE_BLOCKING_FAIL, "bad blocking dimensions");

    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.simd_w : 0;
    jcp.oc_tail = jcp.oc_without_padding % jcp.simd_w;

    format_tag_t src_tag, dst_tag, wei_tag;

    if (jcp.simd_w == 8) {
        assert(with_groups);
        src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx8c;
        dst_tag = src_tag;
        wei_tag = pick(ndims - 3, gOIw8i8o, gOIhw8i8o, gOIdhw8i8o);
    } else if (jcp.simd_w == 4) {
        assert(with_groups);
        src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx4c;
        dst_tag = src_tag;
        wei_tag = pick(ndims - 3, gOIw4i4o, gOIhw4i4o, gOIdhw4i4o);
    } else {
        dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
        src_tag = is_data_layout_nxc
                ? dat_tag_nxc
                : (jcp.is_1stconv ? dat_tag_ncx : dat_tag_nCx16c);
        wei_tag = pick(2 * ndims - 6 + with_groups, OIw16i16o, gOIw16i16o,
                OIhw16i16o, gOIhw16i16o, OIdhw16i16o, gOIdhw16i16o);
    }

    if (jcp.is_1stconv) {
        wei_tag = with_groups
                ? ((jcp.simd_w == 4)
                                  ? pick(ndims - 3, gOwi4o, gOhwi4o, gOdhwi4o)
                                  : pick(ndims - 3, gOwi16o, gOhwi16o,
                                            gOdhwi16o))
                : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);
    }

    if (src_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(src_md, src_tag));
    else
        VDISPATCH_CONV_IC(
                curr_src_tag == src_tag, VERBOSE_UNSUPPORTED_TAG_S, "src");
    jcp.src_tag = src_tag;

    if (dst_md.format_kind == format_kind::any)
        CHECK(memory_desc_init_by_tag(dst_md, dst_tag));
    else
        VDISPATCH_CONV_IC(
                curr_dst_tag == dst_tag, VERBOSE_UNSUPPORTED_TAG_S, "dst");
    jcp.dst_tag = dst_tag;

    if (jcp.is_g_packed) {
        if (weights_d.format_kind() == format_kind::any)
            weights_md = g_packed_wei_md;
        jcp.wei_tag = format_tag::undef;
    } else {
        VDISPATCH_CONV_IC(init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag)
                        == status::success,
                VERBOSE_UNSUPPORTED_TAG);
    }

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(bias_md, x));
    }

    CHECK(attr.set_default_formats(&dst_md));

    const auto &post_ops = attr.post_ops_;
    jcp.with_sum = post_ops.find(primitive_kind::sum) != -1;
    const int eltwise_ind = post_ops.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise) {
        VDISPATCH_CONV_IC(
                dst_d.data_type() != data_type::s32, VERBOSE_UNSUPPORTED_DT);
    }
    const int binary_ind = post_ops.find(primitive_kind::binary);
    const int prelu_ind = post_ops.find(primitive_kind::prelu);
    jcp.with_binary = !everyone_is(-1, binary_ind, prelu_ind);

    jcp.post_ops = post_ops;

    using namespace injector;
    static constexpr bool sum_at_pos_0_only = true;
    static constexpr bool sum_requires_scale_one = true;
    static constexpr bool sum_requires_zp_zero = true;
    bool post_ops_ok_ = post_ops_ok(post_ops_ok_args_t(avx512_core,
            {eltwise, binary, sum}, jcp.post_ops, &dst_d, sum_at_pos_0_only,
            sum_requires_scale_one, sum_requires_zp_zero));
    // temporary workaround that skips avx2 implementation for ternary
    // post-ops with scalar broadcasting to avoid register collisions.
    post_ops_ok_ = post_ops_ok_
            && IMPLICATION(jcp.with_binary,
                    !binary_injector::
                            any_binary_postop_rhs_with_ternary_scalar_bcast(
                                    post_ops, dst_d));
    VDISPATCH_CONV_IC(post_ops_ok_, VERBOSE_UNSUPPORTED_POSTOP);

    jcp.typesize_in = typesize;
    jcp.typesize_out = typesize;

    if (jcp.is_1stconv) {
        jcp.ur_w = static_cast<int>(nstl::min<dim_t>(jcp.ow, regs));
    } else {
        // avx512_core guard - just to avoid possible regression for other archs
        if (mayiuse(avx512_core)) {
            jcp.ur_w = static_cast<int>(nstl::min<dim_t>(jcp.ow, regs));
        } else {
            for (int ur_w = regs; ur_w > 0; --ur_w) {
                if (jcp.ow % ur_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
        if ((ndims == 5 && jcp.ur_w <= 8) || (jcp.ur_w <= 1)) {
            jcp.ur_w = static_cast<int>(nstl::min<dim_t>(jcp.ow, regs));
        }
    }
    // TODO (Tanya): currently applied to Segnet convolutions only.
    // Need to try for other topologies
    if (jcp.ow > 150 && jcp.ur_w < regs / 2) jcp.ur_w = regs;

    dim_t n_oi = jcp.ow / jcp.ur_w;
    dim_t r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ur_w * n_oi, jcp.iw, jcp.stride_w, ext_kw);
    if (jcp.l_pad > 0 && r_pad > 0) n_oi--;

    // Heuristic to optimize code size on KNX
    bool large_code_size = jcp.ur_w != jcp.ow && jcp.l_pad > 0 && r_pad > 0
            && ((jcp.l_pad <= 0 && n_oi > 0) || (jcp.l_pad > 0 && n_oi > 1));
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const dim_t num_ops_per_reg = 6 + jcp.ic_block * jcp.kw;
        int mult = 1;
        if (jcp.l_pad > 0) mult += 1;
        if (r_pad > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs / 2; --ur_w) {
            if ((double)(ur_w * mult * num_ops_per_reg) * 9.0 < max_code_size) {
                jcp.ur_w = ur_w;
                break;
            }
        }
    }

    /* Grouped channel offset to support 'non-blocked data' format for
     * convolution sizes with '(input_channel / ngroups) < simd' */
    jcp.nonblk_group_off = static_cast<int>(
            (jcp.ngroups > 1 && one_of(jcp.src_tag, ncw, nchw, ncdhw)) ? jcp.ic
                                                                       : 1);

    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);
    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    auto is_ow_threading_applicable
            = [&]() { return (!jcp.is_1stconv && one_of(jcp.ndims, 3, 4)); };

    jcp.ow_block = jcp.ow;

    auto get_thr_eff = [&](int nb_oc_blocking, dim_t ow_block, int nthr) {
        const dim_t nb_ow = div_up(jcp.ow, ow_block);
        const dim_t nb_oc_chunks = div_up(jcp.nb_oc, nb_oc_blocking);
        const dim_t work_amount = jcp.mb * jcp.oh * nb_oc_chunks * nb_ow;
        float disbalance = (float)jcp.ow / (float)rnd_up(jcp.ow, ow_block);
        float thr_eff = disbalance * (float)work_amount
                / (float)rnd_up(work_amount, nthr);
        return thr_eff;
    };

    auto get_ow_block = [&](int nb_oc_blocking, int ur_w, int nthr) {
        dim_t res_ow_block = jcp.ow;
        float eff = get_thr_eff(nb_oc_blocking, res_ow_block, nthr);
        if (!is_ow_threading_applicable()) return res_ow_block;

        int L2_part = (platform::get_per_core_cache_size(2) * 7 / 8) / typesize;
        const dim_t size_src_chunk = jcp.ic_block * ur_w * jcp.kh;
        const dim_t size_dst_chunk = jcp.oc_block * nb_oc_blocking * ur_w;
        const dim_t size_wei_chunk = jcp.oc_block * nb_oc_blocking
                * jcp.ic_block * jcp.kw * jcp.kh;
        const dim_t nurw_cache = (L2_part - 2 * size_wei_chunk)
                / (2 * size_dst_chunk + 2 * size_src_chunk);
        // current design of generate() requires ow_block >= 2 * ur_w
        const dim_t ow_block_cache = ur_w * nstl::max<dim_t>(2, nurw_cache);

        dim_t ow_block_thr = ow_block_cache;
        eff = get_thr_eff(nb_oc_blocking, ow_block_thr, nthr);

        const dim_t max_nb_ow = div_up(jcp.ow, 2 * ur_w);
        const dim_t start_nb_ow = div_up(jcp.ow, ow_block_thr);
        for (dim_t nb_ow = start_nb_ow; nb_ow <= max_nb_ow; nb_ow++) {
            const dim_t ow_block = nstl::min<dim_t>(
                    rnd_up(div_up(jcp.ow, nb_ow), ur_w), jcp.ow);
            float eff_threshold = 0.9f;
            if (ow_block < nb_oc_blocking * jcp.oc_block && eff > eff_threshold)
                break;
            if (div_up(jcp.ow, ow_block) != nb_ow) continue;
            float thr_eff = get_thr_eff(nb_oc_blocking, ow_block, nthr);
            float eff_step = 1.f;
            if (ow_block >= 2 * ur_w && thr_eff > eff_step * eff) {
                ow_block_thr = ow_block;
                eff = thr_eff;
            }
            eff_threshold = 0.98f;
            if (eff > eff_threshold) break;
        }
        res_ow_block = nstl::min<dim_t>(
                jcp.ow, nstl::max<dim_t>(2 * ur_w, ow_block_thr));
        eff = get_thr_eff(nb_oc_blocking, res_ow_block, nthr);
        return res_ow_block;
    };

    const size_t L1_cache_size = platform::get_per_core_cache_size(1);
    if (mayiuse(avx512_core)) {
        int try_nb_oc_blocking = 2;
        const size_t ker_inp_size = typesize * div_up(jcp.iw, jcp.stride_w)
                * jcp.ic_block * jcp.kh * jcp.kd;
        const size_t ker_out_size
                = typesize * jcp.ow * jcp.oc_block * try_nb_oc_blocking;
        size_t ker_wei_size = static_cast<size_t>(typesize) * jcp.kh * jcp.kw
                * jcp.ic_block * jcp.oc_block * try_nb_oc_blocking * jcp.kd;
        size_t ker_total_size = ker_inp_size + ker_out_size + ker_wei_size;

        bool embd_bcast_condition_base = true
                && (jcp.kw == 3 && jcp.ow <= 28
                        && ker_total_size < L1_cache_size)
                && !(jcp.kw == 3 && jcp.ow == 13 && jcp.ic >= 192)
                && !(jcp.kw == 3 && jcp.ow == 28 && jcp.ic >= 512);
        // These conditions define a set of shapes with 'ow = 1' which
        // have a very limited optimization space for performance. Try
        // to optimize by using a larger 'nb_oc_blocking' size.
        bool expl_bcast_condition
                = everyone_is(1, jcp.ngroups, jcp.mb, jcp.stride_h, jcp.ow,
                          jcp.stride_w, jcp.id, jcp.od, jcp.kd, jcp.stride_d)
                && jcp.iw == jcp.kw && jcp.nb_oc > 1
                && everyone_is(0, jcp.l_pad, jcp.r_pad, jcp.dilate_w, jcp.f_pad,
                        jcp.back_pad, jcp.dilate_d)
                && jcp.oh >= 60 && jcp.kh >= 3;

        bool embd_bcast_condition = !expl_bcast_condition
                && (jcp.kw > 3
                        || (jcp.stride_w == 1 && jcp.stride_h == 1
                                && embd_bcast_condition_base)
                        || ((jcp.stride_w != 1 || jcp.stride_h != 1)
                                && ((jcp.mb <= 16
                                        && (jcp.oc <= 192 || jcp.oh <= 10)
                                        && embd_bcast_condition_base)))
                        || (jcp.mb == 1
                                && (jcp.ur_w >= jcp.ow || jcp.is_1stconv
                                        || (jcp.ow <= 147 && jcp.oc <= 96))));

        if (jcp.mb == 1) {
            const size_t inp_size = jcp.mb * div_up(jcp.ih, jcp.stride_h)
                    * div_up(jcp.iw, jcp.stride_w) * jcp.ic;
            const size_t wei_size = jcp.ic * jcp.oc * jcp.kh * jcp.kw;

            // Estimate whether we need to limit the number of threads
            // and calculate this number. Includes some heuristic.
            const dim_t oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
            const dim_t work_amount = jcp.mb * jcp.ngroups * oc_chunks * jcp.oh;
            const int job_size_min = static_cast<int>(work_amount / nthreads);
            const int job_size_max
                    = static_cast<int>(div_up(work_amount, nthreads));
            const dim_t ch_max = rnd_up(jcp.oh, job_size_max);
            const dim_t ch_min = (job_size_min == 0)
                    ? jcp.oh
                    : rnd_up(jcp.oh, job_size_min);
            bool not_aligned_max = ch_max % jcp.oh != 0 && ch_max / jcp.oh < 2
                    && (jcp.oh != 8 || ch_max / jcp.oh > 1);
            bool not_aligned_min = ch_min % jcp.oh != 0 && ch_min / jcp.oh < 2
                    && (jcp.oh != 8 || ch_min / jcp.oh > 1);
            bool eligible_case = (jcp.stride_h == 1 && jcp.stride_w == 1)
                    || nthreads > oc_chunks;
            if (jcp.loop_order == loop_cgn && oc_chunks > 1 && nthreads > 1
                    && wei_size / inp_size > 24
                    && (not_aligned_max || not_aligned_min) && eligible_case) {
                // Try to find number of threads > nthreads / 2 such that
                // oc_chunks is a multiple of nthreads, or nthreads is a
                // multiple of oc_chunks. Otherwise, keep default value.
                // TODO: implement a task-based alternative without throttling.
                jcp.aligned_threads = jcp.nthr;
                for (int i = jcp.nthr; i > jcp.nthr / 2; i--) {
                    if (oc_chunks % i == 0 || i % oc_chunks == 0) {
                        jcp.aligned_threads = i;
                        break;
                    }
                }
            }
        }

        const int max_nb_oc = 5;
        if (embd_bcast_condition) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = static_cast<int>(nstl::min<dim_t>(jcp.ow, regs));
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            const unsigned int L1_cache_size
                    = platform::get_per_core_cache_size(1);
            if (ker_total_size < L1_cache_size && jcp.ow <= 8 && jcp.kh <= 3
                    && jcp.kw <= 3 && jcp.nb_oc % try_nb_oc_blocking == 0
                    && IMPLICATION(jcp.is_1stconv, jcp.mb == 1)
                    && IMPLICATION(jcp.mb == 1, jcp.ur_w < jcp.ow)) {
                jcp.nb_oc_blocking = try_nb_oc_blocking;
                jcp.ur_w = static_cast<int>(nstl::min<dim_t>(
                        jcp.ow, 31 / (jcp.nb_oc_blocking + 1)));
            }
        } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_ic_blocking = 1;
            if (IMPLICATION(jcp.is_1stconv, jcp.mb > 1)
                    || expl_bcast_condition) {
                float best_thr_eff = 0.f;
                int best_nb_oc_blocking = 1;
                for (int i = static_cast<int>(
                             nstl::min<dim_t>(jcp.nb_oc, max_nb_oc));
                        i > 0; i--) {
                    if (jcp.nb_oc % i == 0) {
                        if (expl_bcast_condition) {
                            best_nb_oc_blocking = i;
                            break;
                        } else {
                            int ur_w = static_cast<int>(
                                    nstl::min<dim_t>(jcp.ow, 31 / (i + 1)));
                            dim_t ow_block = get_ow_block(i, ur_w, jcp.nthr);
                            float thr_eff = get_thr_eff(i, ow_block, jcp.nthr);
                            if (thr_eff > 1.05f * best_thr_eff) {
                                best_nb_oc_blocking = i;
                                best_thr_eff = thr_eff;
                            }
                        }
                    }
                }
                jcp.nb_oc_blocking = best_nb_oc_blocking;
                jcp.ur_w = static_cast<int>(nstl::min<dim_t>(
                        jcp.ow, 31 / (jcp.nb_oc_blocking + 1)));
            }
        }
    }

    if (jcp.is_g_packed) {
        // 4 accumulators per output point, 4 weights and 4 temporary
        // registers.
        jcp.kernel_kind = expl_bcast;
        jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
        jcp.ur_w = static_cast<int>(g_packed_ur_w);
    }

    jcp.ur_w_tail = jcp.ow % jcp.ur_w;

    bool args_ok = true && jcp.l_pad <= jcp.ur_w
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= dst_d.padded_dims()[1]
            && IMPLICATION(!jcp.is_g_packed,
                    jcp.ic <= weights_d.padded_dims()[with_groups + 1]
                            && jcp.oc <= weights_d.padded_dims()[with_groups
                                       + 0]);
    VDISPATCH_CONV_IC(args_ok, VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "weight and src size mismatch");

    int r_pad_no_tail = static_cast<int>(nstl::max<dim_t>(0,
            calculate_end_padding(jcp.l_pad, jcp.ow - jcp.ur_w_tail, jcp.iw,
                    jcp.stride_w, ext_kw)));
    VDISPATCH_CONV_IC(r_pad_no_tail <= jcp.ur_w,
            VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "width unroll exceeds padding size");

    pick_loop_order(jcp);

    jcp.nb_ic_L2 = jcp.nb_ic;

    jcp.ow_block = get_ow_block(jcp.nb_oc_blocking, jcp.ur_w, jcp.nthr);
    jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);
    float thr_eff = get_thr_eff(jcp.nb_oc_blocking, jcp.ow_block, jcp.nthr);

    /* adjust the thread decomposition
     * to improve the thr_eff for small size problem
     * the threshold L1_cache_size is empirical */
    size_t wei_size
            = (size_t)typesize * jcp.ic * jcp.oc * jcp.kh * jcp.kw * jcp.kd;
    size_t out_size = (size_t)jcp.mb * jcp.typesize_out * jcp.oc * jcp.oh
            * jcp.ow * jcp.od;
    size_t inp_size = (size_t)jcp.mb * jcp.typesize_in * jcp.ic * jcp.ih
            * jcp.iw * jcp.id;
    size_t total_size = jcp.ngroups * (wei_size + out_size + inp_size);
    float eff_threshold = 0.98f;

    if (thr_eff < eff_threshold && jcp.ngroups < jcp.nthr
            && (total_size < L1_cache_size)) {
        dim_t ow_block = jcp.ow_block;
        float best_thr_eff = -1.0f;
        float eff = -1.0f;
        const int end_nthr = with_groups ? static_cast<int>(jcp.ngroups) : 1;
        for (int nthr = jcp.nthr / 2; nthr >= end_nthr; nthr--) {
            ow_block = get_ow_block(jcp.nb_oc_blocking, jcp.ur_w, nthr);
            eff = get_thr_eff(jcp.nb_oc_blocking, ow_block, nthr);
            if (eff > 1.1f * best_thr_eff) {
                best_thr_eff = eff;
                jcp.ow_block = ow_block;
                jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);
                jcp.nthr = jcp.aligned_threads = nthr;
                if (best_thr_eff > eff_threshold) break;
            }
        }
    }

    const int L2_size = platform::get_per_core_cache_size(2) / typesize;
    // Source and output data needs to fit in L2,
    // leaving some space for weights and prefetching.
    int h_L2 = int(((0.6f * (float)L2_size) / (float)jcp.simd_w
                           - (float)(nstl::min<dim_t>(0, jcp.kh - jcp.stride_h)
                                   * jcp.iw))
            / (float)(jcp.stride_h * jcp.iw + jcp.ow));
    jcp.h_blocking = nstl::max<dim_t>(1, nstl::min<dim_t>(jcp.oh, h_L2));

    if (is_data_layout_nxc) {
        // TODO: improve L2 blocking for large IC
        const int nb_ic_theshold_L2 = 32;
        if (jcp.nb_ic > nb_ic_theshold_L2 && jcp.nb_ic < 2 * nb_ic_theshold_L2)
            jcp.nb_ic_L2 = div_up(jcp.nb_ic, 2);
        else
            jcp.nb_ic_L2 = nstl::min<dim_t>(nb_ic_theshold_L2, jcp.nb_ic);
    }

    // A rough check on code size
    // TODO: come up with a tighter bound
    {
        const int max_code_size = 256 * 1024; // default size of jit generator
        int mult = 1 + (jcp.l_pad > 0) + (r_pad > 0);
        const float max_instruction_size = 15;
        float ur_fac = (float)jcp.kw * (float)jcp.ic_block
                * (float)jcp.nb_oc_blocking * (float)jcp.ur_w;
        float code_size = (float)mult * ur_fac * max_instruction_size;
        VDISPATCH_CONV_IC(
                code_size <= max_code_size, "code size limit exceeded");
    }

    return status::success;
}

void jit_avx512_common_conv_fwd_kernel_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.with_bias && jcp.oc != jcp.oc_without_padding)
        scratchpad.book(key_conv_padded_bias, jcp.oc, jcp.typesize_out);
}

template <typename Vmm>
void jit_avx512_common_conv_bwd_data_kernel_f32_vmm_t<Vmm>::prepare_output(
        int ur_w) {
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            vpxord(vmm, vmm, vmm);
        }
    }
}

template <typename Vmm>
void jit_avx512_common_conv_bwd_data_kernel_f32_vmm_t<Vmm>::store_output(
        int ur_w) {
    Label no_update_label;
    const int ic_tail = jcp.ic_without_padding % jcp.simd_w;
    const bool dsrc_layout_nxc = is_dsrc_layout_nxc();
    mov(reg_channel, ptr[param + GET_OFF(channel)]);
    cmp(reg_channel, 0);
    je(no_update_label, T_NEAR);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            size_t aux_src_offset = get_diff_src_offset(j, k);
            vaddps(vmm,
                    EVEX_compress_addr_safe(
                            reg_src, aux_src_offset, reg_long_offt));
        }
    }

    L(no_update_label);
    for (int k = 0; k < jcp.nb_ic_blocking; k++) {
        for (int j = 0; j < ur_w; j++) {
            Vmm vmm = vmm_out(j, k);
            // mask only needed for last oc_block
            if (ic_tail && k + 1 == jcp.nb_ic_blocking && dsrc_layout_nxc)
                vmm = vmm | k_ic_tail_mask;
            size_t aux_src_offset = get_diff_src_offset(j, k);
            vmovups(EVEX_compress_addr_safe(
                            reg_src, aux_src_offset, reg_long_offt),
                    vmm);
        }
    }
}

template <typename Vmm>
void jit_avx512_common_conv_bwd_data_kernel_f32_vmm_t<Vmm>::compute_loop_fma(
        int ur_w, dim_t l_overflow, dim_t r_overflow) {
    Label kh_label, kd_label;
    dim_t kw = jcp.kw;
    dim_t ow = jcp.ow;

    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const int stride_w = static_cast<int>(jcp.stride_w);
    dim_t stride_h = jcp.stride_h;

    int ker_pipeline_depth = 4;
    assert(ker_reg_base_idx + ker_pipeline_depth <= 32);
    assert(oc_block >= ker_pipeline_depth);

    const int num_ker_loads = static_cast<int>(oc_block * kw);
    const bool ddst_layout_nxc = is_ddst_layout_nxc();
    dim_t oc_mult = ddst_layout_nxc ? jcp.ngroups * jcp.oc : oc_block;
    const bool ocb_loop_in_compute_function = ddst_layout_nxc;

    const int ic_tail = jcp.ic_tail;
    const int oc_tail = jcp.oc_tail;
    std::vector<Label> oc_tail_jmp(kw);
    if (ic_tail || oc_tail) ker_pipeline_depth = 1;

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }

    if (jcp.ndims == 5) {
        push(reg_src);

        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        if (ocb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            assert(aux_reg_ker_d == reg_ker);
            push(aux_reg_ker_d);
        } else
            mov(aux_reg_ker_d, ptr[param + GET_OFF(filt)]);

        L(kd_label);
        mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        int step = 0;
        for (int ki = 0; ki < kw; ki++) {
            for (int oc = 0; oc < oc_block; oc++) {
                if (oc_tail && oc >= oc_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.oc == oc_tail)
                        break;
                    else if (oc == oc_tail) {
                        cmp(reg_channel, oc_tail);
                        je(oc_tail_jmp[ki], T_NEAR);
                    }
                }
                if (step == 0) {
                    for (int i = 0; i < ker_pipeline_depth; i++) {
                        dim_t aux_kernel_offset = typesize
                                * ((oc + i) * oc_block
                                        + ki * ic_block * oc_block);
                        vmovups(vmm_ker(i),
                                EVEX_compress_addr(
                                        aux_reg_ker, aux_kernel_offset));
                    }
                } else if (step < num_ker_loads - ker_pipeline_depth + 1) {
                    int load_offset = ker_pipeline_depth - 1;
                    int ker_load_reg_idx
                            = (step + load_offset) % ker_pipeline_depth;
                    dim_t aux_kernel_offset = typesize
                            * ((oc + load_offset) * oc_block
                                    + ki * ic_block * oc_block);
                    vmovups(vmm_ker(ker_load_reg_idx),
                            EVEX_compress_addr(aux_reg_ker, aux_kernel_offset));
                }

                auto vmm_kernel = vmm_ker(step % ker_pipeline_depth);

                const int jj_start = get_iw_start(ki, l_overflow);
                const int jj_end = get_iw_end(ur_w, ki, r_overflow);
                const dim_t dil_w = jcp.dilate_w + 1;
                const dim_t ref_jj_start = nstl::max<dim_t>(
                        0, l_overflow - (kw - 1 - ki) * dil_w);
                const dim_t ref_jj_end
                        = ur_w - nstl::max<dim_t>(0, r_overflow - ki * dil_w);
                assert(IMPLICATION(stride_w == 1,
                        jj_start == ref_jj_start && jj_end == ref_jj_end));
                UNUSED(dil_w);
                UNUSED(ref_jj_start);
                UNUSED(ref_jj_end);

                for (int jj = jj_start; jj < jj_end; jj += stride_w) {
                    assert((jj + jcp.l_pad - ki * (jcp.dilate_w + 1)) % stride_w
                            == 0);
                    dim_t aux_dst_offset = get_dst_offset(jj, oc, ki);
                    vfmadd231ps(vmm_out(jj, 0), vmm_kernel,
                            EVEX_compress_addr(
                                    aux_reg_dst, aux_dst_offset, true));
                }
                step++;
            }
            L(oc_tail_jmp[ki]);
        }

        const dim_t ker_shift = typesize * stride_h * kw * oc_block * ic_block;
        add(aux_reg_ker, ker_shift);
        const dim_t ddst_shift = typesize * (jcp.dilate_h + 1) * ow * oc_mult;
        sub(aux_reg_dst, ddst_shift);

        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        const dim_t depth_ddst_shift
                = typesize * (jcp.dilate_d + 1) * jcp.oh * ow * oc_mult;
        sub(aux_reg_dst_d, depth_ddst_shift);
        const dim_t depth_ker_shift = typesize * jcp.stride_d * jcp.kw * jcp.kh
                * oc_block * ic_block;
        add(aux_reg_ker_d, depth_ker_shift);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);
        if (ocb_loop_in_compute_function) pop(aux_reg_ker_d);
    }

    if (jcp.ndims == 5) { pop(reg_src); }
}

template <typename Vmm>
void jit_avx512_common_conv_bwd_data_kernel_f32_vmm_t<
        Vmm>::compute_loop_fma_core(int ur_w, dim_t l_overflow,
        dim_t r_overflow, int k_offset) {
    dim_t kw = jcp.kw;
    dim_t ow = jcp.ow;
    const int stride_w = static_cast<int>(jcp.stride_w);
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const int nb_ic_block = jcp.nb_ic_blocking;
    Label kh_label, kd_label;

    const bool ddst_layout_nxc = is_ddst_layout_nxc();
    dim_t shift_ker_ptr = typesize * kw * oc_block * ic_block;
    dim_t oc_mult = ddst_layout_nxc ? jcp.ngroups * jcp.oc : oc_block;
    dim_t shift_dst_ptr = typesize * (jcp.dilate_h + 1) * ow * oc_mult;

    const int oc_tail = jcp.oc_tail;
    const int max_filter_size = 20;
    Label oc_tail_jmp[max_filter_size];

    auto kernel_offset = [this](dim_t icb, dim_t oc, dim_t ki) -> dim_t {
        const dim_t blk_idx = icb * jcp.kh * jcp.kw * jcp.kd + ki;
        const dim_t blk_offset = blk_idx * jcp.oc_block * jcp.ic_block;
        const dim_t oc_offset = oc * jcp.oc_block;
        return typesize * (blk_offset + oc_offset);
    };

    if (one_of(jcp.ndims, 3, 4)) {
        mov(aux_reg_dst, reg_dst);
        mov(aux_reg_ker, reg_ker);
    }

    const bool ocb_loop_in_compute_function = ddst_layout_nxc;
    if (jcp.ndims == 5) {
        push(reg_src);

        mov(reg_ki, ptr[param + GET_OFF(kd_padding)]);
        mov(aux_reg_dst_d, reg_dst);
        if (ocb_loop_in_compute_function) {
            // need to continue with the same kernel pointer, but as
            // aux_reg_ker_d == reg_ker we need to save its value and restore
            // it after kd loop
            assert(aux_reg_ker_d == reg_ker);
            push(aux_reg_ker_d);
        } else
            mov(aux_reg_ker_d, ptr[param + GET_OFF(filt)]);

        L(kd_label);
        mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    } else {
        mov(reg_kj, reg_kh);
    }

    if (jcp.ndims == 5) {
        mov(aux_reg_dst, aux_reg_dst_d);
        mov(aux_reg_ker, aux_reg_ker_d);
    }

    L(kh_label);
    {
        for (int ki = 0; ki < kw; ki++) {
            const int jj_start = get_iw_start(ki, l_overflow);
            const int jj_end = get_iw_end(ur_w, ki, r_overflow);
            for (int oc = 0; oc < oc_block; oc++) {
                if (oc_tail && oc >= oc_tail) {
                    // if src has only tails to compute, skip early
                    if (jcp.oc == oc_tail)
                        break;
                    else if (oc == oc_tail) {
                        cmp(reg_channel, oc_tail);
                        je(oc_tail_jmp[ki], T_NEAR);
                    }
                }
                if (jcp.kernel_kind == expl_bcast) {
                    for (int jj = jj_start; jj < jj_end; jj++) {
                        dim_t aux_output_offset = get_dst_offset(jj, oc, ki);
                        vbroadcastss(vmm_inp(jj, nb_ic_block),
                                ptr[aux_reg_dst + aux_output_offset]);
                    }
                }
                for (int ii = 0; ii < nb_ic_block; ii++) {
                    dim_t aux_kernel_offset
                            = kernel_offset(ii, oc, ki + k_offset);
                    if (jj_end - jj_start > 0)
                        vmovups(vmm_wei,
                                EVEX_compress_addr(
                                        aux_reg_ker, aux_kernel_offset));
                    for (int jj = jj_start; jj < jj_end; jj += stride_w)
                        if (jcp.kernel_kind == expl_bcast)
                            vfmadd231ps(vmm_out(jj, ii),
                                    vmm_inp(jj, nb_ic_block), vmm_wei);
                        else
                            vfmadd231ps(vmm_out(jj, ii), vmm_wei,
                                    EVEX_compress_addr(aux_reg_dst,
                                            get_dst_offset(jj, oc, ki), true));
                }
            }
            L(oc_tail_jmp[ki]);
        }
        add(aux_reg_ker, shift_ker_ptr);
        sub(aux_reg_dst, shift_dst_ptr);
        dec(reg_kj);
        cmp(reg_kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        sub(aux_reg_dst_d,
                typesize * (jcp.dilate_d + 1) * jcp.oh * ow * oc_mult);
        add(aux_reg_ker_d, typesize * jcp.kw * jcp.kh * oc_block * ic_block);

        dec(reg_ki);
        cmp(reg_ki, 0);
        jg(kd_label, T_NEAR);

        if (ocb_loop_in_compute_function) pop(aux_reg_ker_d);
        pop(reg_src);
    }
}

template <typename Vmm>
inline void jit_avx512_common_conv_bwd_data_kernel_f32_vmm_t<Vmm>::compute_loop(
        int ur_w, dim_t l_overflow, dim_t r_overflow, int k_offset) {
    if (jcp.ndims == 5) push(reg_oi);

    prepare_output(ur_w);

    Label skip_compute_loop;
    if (jcp.ndims == 5) {
        mov(reg_kj, ptr[param + GET_OFF(kd_padding)]);
        cmp(reg_kj, 0);
        jle(skip_compute_loop, T_NEAR);
    }
    mov(reg_kj, ptr[param + GET_OFF(kh_padding)]);
    cmp(reg_kj, 0);
    jle(skip_compute_loop, T_NEAR);

    const bool generate_ocb_loop = jcp.nb_oc > 1 && is_ddst_layout_nxc();
    Label oc_loop;
    if (generate_ocb_loop) {
        push(reg_dst);
        push(reg_ker);

        mov(reg_channel, ptr[param1 + GET_OFF(reduce_work)]);
        L(oc_loop);
    }

    if (jcp.kernel_kind == embd_bcast && jcp.nb_ic_blocking == 1)
        compute_loop_fma(ur_w, l_overflow, r_overflow);
    else
        compute_loop_fma_core(ur_w, l_overflow, r_overflow, k_offset);

    if (generate_ocb_loop) {
        add(reg_dst, jcp.oc_block * typesize);
        const dim_t ker_shift = jcp.nb_ic * jcp.kd * jcp.kh * jcp.kw
                * jcp.ic_block * jcp.oc_block * typesize;
        add(reg_ker, ker_shift);
        sub(reg_channel, jcp.oc_block);
        jg(oc_loop, T_NEAR);

        pop(reg_ker);
        pop(reg_dst);
    }

    L(skip_compute_loop);
    store_output(ur_w);
    if (jcp.ndims == 5) pop(reg_oi);
}

template <typename Vmm>
void jit_avx512_common_conv_bwd_data_kernel_f32_vmm_t<Vmm>::generate() {
    const dim_t iw = jcp.iw;
    const dim_t kw = jcp.kw;
    int ur_w = jcp.ur_w;
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const dim_t nb_iw = jcp.nb_iw;
    const dim_t iw_block = jcp.iw_block;
    int ur_w_tail = jcp.ur_w_tail;
    int dilate_w = static_cast<int>(jcp.dilate_w) + 1;
    int stride_w = static_cast<int>(jcp.stride_w);

    dim_t dst_shift = jcp.typesize_in * (ur_w / stride_w)
            * (is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : oc_block);
    dim_t src_shift = jcp.typesize_out * ur_w
            * (is_dsrc_layout_nxc() ? jcp.ngroups * jcp.ic : ic_block);

    preamble();

    mov(reg_src, ptr[param + GET_OFF(src)]);
    mov(reg_dst, ptr[param + GET_OFF(dst)]);
    mov(reg_ker, ptr[param + GET_OFF(filt)]);

    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);

    const int ic_tail = jcp.ic_tail;
    if (ic_tail) {
        Label masking_done;
        // dummy mask all 1's
        kxnorw(k_ic_tail_mask, k_ic_tail_mask, k_ic_tail_mask);
        mov(reg_load_work, ptr[param1 + GET_OFF(load_work)]);
        cmp(reg_load_work, jcp.nb_ic_blocking * jcp.ic_block);
        je(masking_done, T_NEAR);
        Reg32 reg_tail_32 = reg_tail.cvt32();
        mov(reg_tail_32, (1 << ic_tail) - 1);
        kmovw(k_ic_tail_mask, reg_tail_32);
        L(masking_done);
    }

    int l_overflow = static_cast<int>(
            nstl::max<dim_t>(0, ((kw - 1) * dilate_w - jcp.l_pad) / stride_w));
    int r_overflow = static_cast<int>(nstl::max<dim_t>(0,
            ((kw - 1) * dilate_w - nstl::max<dim_t>(0, jcp.r_pad)) / stride_w));
    int r_overflow_no_tail = static_cast<int>(nstl::max<dim_t>(0,
            ((kw - 1) * dilate_w - nstl::max<dim_t>(0, jcp.r_pad + ur_w_tail))
                    / stride_w));

    int body_l_overflow = 0, body_r_overflow = 0;
    int n_oi = static_cast<int>(iw / ur_w);
    int head_n_oi = 0, body_n_oi = 0, pretail_n_oi = 0, tail_n_oi = 0;
    int head_thread = 0, pretail_thread = 0, tail_thread = 0;
    bool threaded = is_iw_threading_on(jcp);
    Label head_label, body_label, pretail_label, tail_label, end_label;
    assert(n_oi > 0);
    if (r_overflow_no_tail > 0) n_oi--;
    if (l_overflow > 0) n_oi--;
    if (n_oi < 0) {
        // l_overflow and r_overflow_no_tail are handled in the same compute_loop.
        // Perform one iteration of body handling l_overflow and r_overflow_no_tail.
        // TODO: Align other convolution kernels with this kernel. This version
        // now uses r_overflow_no_tail instead of r_overflow in compute loop, this was
        // done since when iw == ur_w, ur_w_tail == 0 and thus
        // r_overflow_no_tail seems more appropriate
        body_l_overflow = l_overflow;
        body_r_overflow = r_overflow_no_tail;
        n_oi = 1;
        l_overflow = 0;
        r_overflow_no_tail = 0;
    }

    if (!threaded) {
        if (n_oi > 1) { mov(reg_oi, n_oi); }
    } else {
        // Setup for threaded code generation, and jump into the correct
        // portion of code for execution.
        head_thread = 0;
        tail_thread = static_cast<int>(nb_iw - 1);
        pretail_thread = tail_thread;

        int base_n_oi = static_cast<int>(iw_block / ur_w);
        head_n_oi = l_overflow > 0 ? base_n_oi - 1 : base_n_oi;
        tail_n_oi = static_cast<int>((iw - iw_block * (nb_iw - 1)) / ur_w);
        pretail_n_oi = tail_n_oi;
        if (r_overflow_no_tail > 0) {
            if (tail_n_oi > 0) {
                pretail_n_oi--;
                tail_n_oi = pretail_n_oi;
            } else {
                // pretail_thread and tail_thread are different
                pretail_n_oi = base_n_oi - 1;
                pretail_thread = tail_thread - 1;
            }
            if (head_thread == pretail_thread) {
                head_n_oi--;
                pretail_n_oi = 0;
                tail_n_oi = 0;
            }
        }
        body_n_oi = (head_thread < pretail_thread - 1) ? base_n_oi : 0;

        // n_oi is used to determine how much control flow in the body portion
        // of the code needs generated. As such, n_oi needs to be set to the
        // maximum number of iterations it will be used the body code section.
        n_oi = nstl::max(body_n_oi, head_n_oi);
        n_oi = nstl::max(n_oi, pretail_n_oi);

        assert(iw_block % ur_w == 0);
        mov(reg_iwb, ptr[param1 + GET_OFF(iwb)]);

        if (head_n_oi != 0) mov(reg_oi, head_n_oi);
        cmp(reg_iwb, head_thread);
        je(head_label, T_NEAR);

        cmp(reg_iwb, pretail_thread);
        if (pretail_n_oi == 0) {
            je(pretail_label, T_NEAR);
        } else {
            mov(reg_oi, pretail_n_oi);
            je(body_label, T_NEAR);
        }
        if (pretail_thread != tail_thread) {
            cmp(reg_iwb, tail_thread);
            je(tail_label, T_NEAR);
        }
        if (body_n_oi != 0) {
            mov(reg_oi, body_n_oi);
            jmp(body_label, T_NEAR);
        } else {
            jmp(end_label, T_NEAR);
        }
    }
    L(head_label);
    if (l_overflow > 0) {
        compute_loop(ur_w, l_overflow, 0);
        if (threaded && head_n_oi == 0 && head_thread != pretail_thread)
            jmp(end_label, T_NEAR);
        else {
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
        }
    }
    L(body_label);
    if (n_oi > 0) {
        Label ow_loop_label;
        L(ow_loop_label);
        {
            compute_loop(ur_w, body_l_overflow, body_r_overflow);
            if (n_oi > 1 || r_overflow_no_tail > 0 || ur_w_tail != 0) {
                add(reg_src, src_shift);
                if (!jcp.large_w_filter) { add(reg_dst, dst_shift); }
            }
            if (n_oi > 1) {
                sub(reg_oi, 1);
                jg(ow_loop_label, T_NEAR);
            }
        }
    }
    if (threaded) {
        mov(reg_iwb, ptr[param1 + GET_OFF(iwb)]);
        cmp(reg_iwb, pretail_thread);
        jne(end_label, T_NEAR);
    }
    L(pretail_label);
    if (r_overflow_no_tail > 0) {
        compute_loop(ur_w, 0, r_overflow_no_tail);
        if (ur_w_tail != 0) {
            if (threaded && tail_thread != pretail_thread)
                jmp(end_label, T_NEAR);
            add(reg_src, src_shift);
            add(reg_dst, dst_shift);
        }
    }
    L(tail_label);
    if (ur_w_tail != 0) {
        /* if 'filter-width > ur_w' then the main loop only partially computes
         * width, ur_w_tail needs to offset the initial ur_w from the filter
         * address. */
        if (jcp.large_w_filter)
            compute_loop(ur_w_tail, body_l_overflow, r_overflow - ur_w, ur_w);
        else
            compute_loop(ur_w_tail, 0, r_overflow);
    }
    L(end_label);

    postamble();
}

status_t jit_avx512_common_conv_bwd_data_kernel_f32_t::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &diff_src_md, memory_desc_t &weights_md,
        memory_desc_t &diff_dst_md, int nthreads) {
    if (!mayiuse(avx512_core)) return status::unimplemented;

    const memory_desc_wrapper diff_src_d(&diff_src_md);
    const memory_desc_wrapper weights_d(&weights_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);
    jcp = zero<decltype(jcp)>();

    VDISPATCH_CONV_IC(everyone_is(data_type::f32, diff_dst_d.data_type(),
                              weights_d.data_type(), diff_src_d.data_type()),
            VERBOSE_UNSUPPORTED_DT_CFG);

    // Big int (> INT_MAX) values are unsupported and jcp fields may overflow
    // TODO: change data type of jcp fields to size_t
    VDISPATCH_CONV_IC(!has_large_size(cd, diff_src_d, weights_d, diff_dst_d),
            VERBOSE_BAD_PARAM, "large size is not supported");

    const bool with_groups = weights_d.ndims() == diff_src_d.ndims() + 1;
    int ndims = diff_src_d.ndims();

    jcp.nthr = jcp.aligned_threads = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = diff_src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = diff_src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    jcp.id = (ndims == 5) ? diff_src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : diff_src_d.dims()[ndims - 2];
    jcp.iw = diff_src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];
    VDISPATCH_CONV_IC(!((jcp.dilate_w != 0 && jcp.stride_w != 1)
                              || (jcp.dilate_d != 0 && jcp.stride_d != 1)
                              || (jcp.dilate_h != 0 && jcp.stride_h != 1)),
            VERBOSE_UNSUPPORTED_FEATURE,
            "unsupported shape with 'stride > 1' when 'dilate > 0'");

    const dim_t ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const dim_t ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    const dim_t ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);
    jcp.r_pad = calculate_end_padding(
            jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw);
    jcp.b_pad = calculate_end_padding(
            jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh);
    jcp.back_pad = calculate_end_padding(
            jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd);
    bool kernel_outside_src = false || ext_kw <= jcp.l_pad
            || ext_kw <= jcp.r_pad || ext_kh <= jcp.t_pad || ext_kh <= jcp.b_pad
            || ext_kd <= jcp.f_pad || ext_kd <= jcp.back_pad;
    VDISPATCH_CONV_IC(!kernel_outside_src, VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "weights and src size mismatch");

    jcp.aligned_threads = 0;
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_nCx4c = pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
    const auto dat_tag_nCx8c = pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = diff_src_d.mb_stride_relaxed_match(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    auto curr_dst_tag = diff_dst_d.mb_stride_relaxed_match(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_nCx8c, dat_tag_nCx4c);
    bool is_data_layout_nxc
            = IMPLICATION(curr_src_tag != dat_tag_nxc,
                      diff_src_d.format_kind() == format_kind::any)
            && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                    diff_dst_d.format_kind() == format_kind::any)
            && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);

    jcp.is_1stconv = false;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1
            && diff_src_d.data_type() == data_type::f32;

    const int full_simd_w = cpu_isa_traits_t<avx512_core>::vlen / typesize;
    jcp.simd_w = full_simd_w;
    bool ok_to_try_lower_zmm = true
            && IMPLICATION(is_data_layout_nxc,
                    jcp.ic < full_simd_w && jcp.oc < full_simd_w
                            && jcp.ngroups > 1)
            && mayiuse(avx512_core) && diff_src_d.data_type() == data_type::f32
            && !jcp.is_1stconv
            && (jcp.oc % jcp.simd_w != 0 || jcp.ic % jcp.simd_w != 0)
            && !ok_to_pad_channels;

    if (ok_to_try_lower_zmm) {
        for (auto simd : {8, 4}) {
            if (jcp.ic % simd == 0 && jcp.oc % simd == 0) {
                jcp.simd_w = simd;
                break;
            }
        }
    }
    // There is no block-4 kernel. With nxc, use a wider block and handle the
    // channels as its tail. Size it by ic, which is kept in a single block.
    if (is_data_layout_nxc && jcp.simd_w == 4)
        jcp.simd_w = jcp.ic <= 8 ? 8 : 16;

    jcp.oc_block = jcp.simd_w;
    jcp.ic_block = jcp.is_1stconv ? jcp.ic : jcp.simd_w;

    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, jcp.oc_block);
        jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
    }

    VDISPATCH_CONV_IC(
            IMPLICATION(!is_data_layout_nxc,
                    jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0),
            VERBOSE_BLOCKING_FAIL, "bad blocking dimensions");
    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % jcp.simd_w : 0;

    format_tag_t dat_tag, wei_tag;
    const auto nxc_tag = pick(ndims - 3, nwc, nhwc, ndhwc);

    if (jcp.simd_w == 8) {
        assert(with_groups);
        dat_tag = is_data_layout_nxc ? nxc_tag
                                     : pick(ndims - 3, nCw8c, nChw8c, nCdhw8c);
        wei_tag = pick(ndims - 3, gOIw8o8i, gOIhw8o8i, gOIdhw8o8i);
    } else if (jcp.simd_w == 4) {
        assert(with_groups);
        dat_tag = is_data_layout_nxc ? nxc_tag
                                     : pick(ndims - 3, nCw4c, nChw4c, nCdhw4c);
        wei_tag = pick(ndims - 3, gOIw4o4i, gOIhw4o4i, gOIdhw4o4i);
    } else {
        dat_tag = is_data_layout_nxc
                ? pick(ndims - 3, nwc, nhwc, ndhwc)
                : pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
        wei_tag = pick(2 * ndims - 6 + with_groups, OIw16o16i, gOIw16o16i,
                OIhw16o16i, gOIhw16o16i, OIdhw16o16i, gOIdhw16o16i);
    }

    if (diff_src_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_src_md, dat_tag));
    } else
        VDISPATCH_CONV_IC(
                curr_src_tag == dat_tag, VERBOSE_UNSUPPORTED_TAG_S, "src");
    jcp.src_tag = dat_tag;

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dat_tag));
    } else
        VDISPATCH_CONV_IC(
                curr_dst_tag == dat_tag, VERBOSE_UNSUPPORTED_TAG_S, "dst");
    jcp.dst_tag = dat_tag;

    VDISPATCH_CONV_IC(init_tag(jcp.wei_tag, weights_md, weights_d, wei_tag)
                    == status::success,
            VERBOSE_UNSUPPORTED_TAG);

    jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    jcp.ur_w = static_cast<int>(jcp.stride_w);

    int regs = 28;
    if (jcp.iw <= regs)
        jcp.ur_w = static_cast<int>(jcp.iw);
    else {
        for (int ur_w = regs; ur_w > 0; --ur_w)
            if (ur_w % jcp.stride_w == 0) {
                jcp.ur_w = ur_w;
                break;
            }
    }
    int l_overflow = static_cast<int>(nstl::max<dim_t>(
            0, ((jcp.kw - 1) * (jcp.dilate_w + 1) - jcp.l_pad) / jcp.stride_w));
    int r_overflow_no_tail = static_cast<int>(nstl::max<dim_t>(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max<dim_t>(0, jcp.r_pad + jcp.iw % jcp.ur_w))
                    / jcp.stride_w));
    dim_t n_oi = jcp.iw / jcp.ur_w;
    if (r_overflow_no_tail > 0) n_oi--;

    jcp.typesize_in = typesize;
    jcp.typesize_out = typesize;

    jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;

    // Heuristic to optimize code size on KNX
    bool large_code_size = (jcp.ur_w != jcp.ow)
            && ((l_overflow <= 0 && n_oi > 0) || (l_overflow > 0 && n_oi > 1))
            && (r_overflow_no_tail > 0) && (l_overflow > 0);
    if (large_code_size) {
        const int max_code_size = 24 * 1024;
        const dim_t num_ops_per_reg = 6 + jcp.oc_block * jcp.kw;
        int mult = 1;
        if (l_overflow > 0) mult += 1;
        if (r_overflow_no_tail > 0) mult += 1;
        for (int ur_w = jcp.ur_w; ur_w > regs / 2; --ur_w) {
            if ((double)((ur_w / jcp.stride_w) * mult * num_ops_per_reg) * 9.2
                    < max_code_size) {
                if (ur_w % jcp.stride_w == 0) {
                    jcp.ur_w = ur_w;
                    break;
                }
            }
        }
    }

    /* Support for large filter 'kw > 14' is only possible when ur_w is small
     * (e.g ur_w = 1) because of register allocation (max_reg = 31) */
    const int min_filter_size = 14;
    /* Don't let JIT generate too big of a code which might result in an
     * out-of-memory crash. */
    const int max_filter_size = 20;

    /* These conditions define a set of shapes with 'ow = 1' which
     * have a very limited optimization space for performance.
     * Optimize by using a targeted 'jcp.nb_ic_blocking' value. */
    jcp.large_w_filter = jcp.kw >= min_filter_size && jcp.kw < max_filter_size
            && jcp.ow == 1 && jcp.nb_ic > 1 && jcp.kw == jcp.iw
            && jcp.stride_w == 1
            && utils::everyone_is(0, jcp.dilate_d, jcp.dilate_h, jcp.dilate_w);

    const unsigned int L1_cache_size = platform::get_per_core_cache_size(1);
    if (mayiuse(avx512_core)) {
        int try_nb_ic_blocking = 2;
        const size_t ker_inp_size = typesize * jcp.iw * jcp.ic_block
                * try_nb_ic_blocking * jcp.kh;
        const size_t ker_out_size = typesize * jcp.ow * jcp.oc_block;
        const size_t ker_wei_size = typesize * jcp.kh * jcp.kw * jcp.ic_block
                * jcp.oc_block * try_nb_ic_blocking;
        size_t ker_total_size = ker_inp_size + ker_out_size + ker_wei_size;
        bool use_expl_bcast
                = !(jcp.kw == 1 || (jcp.kw == 5 && jcp.iw < 8)
                          || (jcp.kw < 5
                                  && ((jcp.iw <= 5
                                              || (jcp.iw > 8 && jcp.iw <= 13))
                                          || ker_total_size > L1_cache_size)))
                || jcp.stride_h > 1 || jcp.stride_d > 1;
        if (use_expl_bcast && !jcp.large_w_filter) {
            jcp.kernel_kind = embd_bcast;
            jcp.ur_w = static_cast<int>(nstl::min<dim_t>(jcp.iw, regs));
            jcp.nb_ic_blocking = jcp.nb_oc_blocking = 1;
            if (!(jcp.kw > 3
                        || (jcp.kw == 3 && ker_total_size < L1_cache_size
                                && jcp.ow > 8))
                    && jcp.stride_h == 1 && jcp.stride_d == 1)
                if (jcp.nb_ic % try_nb_ic_blocking == 0) {
                    jcp.nb_ic_blocking = try_nb_ic_blocking;
                    jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
                    if (jcp.iw < jcp.ur_w) jcp.ur_w = static_cast<int>(jcp.iw);
                }
        } else {
            jcp.kernel_kind = expl_bcast;
            jcp.nb_oc_blocking = 1;
            jcp.nb_ic_blocking = jcp.large_w_filter ? 2 : 4;
            if (jcp.nb_ic < jcp.nb_ic_blocking)
                jcp.nb_ic_blocking = static_cast<int>(jcp.nb_ic);
            if (jcp.nb_ic % jcp.nb_ic_blocking != 0)
                for (int i = jcp.nb_ic_blocking; i > 0; i--)
                    if (jcp.nb_ic % i == 0) {
                        jcp.nb_ic_blocking = i;
                        break;
                    }
            jcp.ur_w = 31 / (jcp.nb_ic_blocking + 1);
            if (jcp.iw < jcp.ur_w) jcp.ur_w = static_cast<int>(jcp.iw);
        }
    }
    jcp.ur_w_tail = jcp.iw % jcp.ur_w;

    auto is_iw_threading_applicable = [&]() { return one_of(jcp.ndims, 3, 4); };

    auto get_thr_eff = [&](int nb_ic_blocking, dim_t iw_block, int nthr) {
        // Cost heuristic for threading overhead. Determined using OMP.
        const float iw_block_cost = 32.0f;

        const dim_t nb_iw = div_up(jcp.iw, iw_block);
        const dim_t nb_ic_chunks = div_up(jcp.nb_ic, nb_ic_blocking);
        const dim_t work_amount = jcp.mb * jcp.ih * nb_ic_chunks * nb_iw;
        float disbalance = (float)jcp.iw / (float)rnd_up(jcp.iw, iw_block);
        float block_overhead
                = nstl::max(0.0f, 1.0f - iw_block_cost / (float)iw_block);
        float thr_eff = block_overhead * disbalance
                * ((float)work_amount / (float)rnd_up(work_amount, nthr));
        return thr_eff;
    };

    auto get_iw_block
            = [&](int nb_ic_blocking, int ur_w, float &eff, int nthr) {
        int res_iw_block = static_cast<int>(jcp.iw);
        if (!is_iw_threading_applicable()) return res_iw_block;

        const dim_t max_nb_iw = div_up(jcp.iw, 2 * ur_w);
        dim_t iw_block_thr;

        if (jcp.ndims == 3) {
            // Blocking optimization to prevent data from leaving cache This
            // blocking optimization does not handle height blocking, so it does
            // not apply to higher dimensions.
            // TODO: Implement a more general optimization taking into account
            // the height dimension.
            int L2_part
                    = (platform::get_per_core_cache_size(2) * 7 / 8) / typesize;
            const dim_t size_diff_src_chunk
                    = jcp.ic_block * nb_ic_blocking * ur_w;
            const dim_t size_diff_dst_chunk = jcp.oc_block * ur_w;
            const dim_t size_wei_chunk
                    = jcp.ic_block * nb_ic_blocking * jcp.oc_block * jcp.kw;
            const dim_t nurw_cache = (L2_part - 2 * size_wei_chunk)
                    / (2 * size_diff_dst_chunk + 2 * size_diff_src_chunk);
            // current design of generate() requires iw_block >= 2 * ur_w
            const dim_t iw_block_cache = ur_w * nstl::max<dim_t>(2, nurw_cache);

            iw_block_thr = iw_block_cache;
        } else
            iw_block_thr = jcp.iw;
        eff = get_thr_eff(nb_ic_blocking, iw_block_thr, nthr);

        // Search for most efficient threading over iw_blocks.
        const dim_t start_nb_iw = div_up(jcp.iw, iw_block_thr);
        for (dim_t nb_iw = start_nb_iw; nb_iw <= max_nb_iw; nb_iw++) {
            float eff_threshold = 0.98f;
            if (eff > eff_threshold) break;
            const dim_t iw_block = nstl::min<dim_t>(
                    rnd_up(div_up(jcp.iw, nb_iw), ur_w), jcp.iw);
            if (div_up(jcp.iw, iw_block) != nb_iw) continue;
            float thr_eff = get_thr_eff(nb_ic_blocking, iw_block, nthr);
            if (iw_block >= 2 * ur_w && thr_eff > eff) {
                iw_block_thr = iw_block;
                eff = thr_eff;
            }
        }
        res_iw_block = static_cast<int>(nstl::min<dim_t>(
                jcp.iw, nstl::max<dim_t>(2 * ur_w, iw_block_thr)));
        return res_iw_block;
    };

    float thr_eff = -1.0f;
    jcp.iw_block
            = get_iw_block(jcp.nb_ic_blocking, jcp.ur_w, thr_eff, jcp.nthr);
    jcp.nb_iw = div_up(jcp.iw, jcp.iw_block);

    /* adjust the thread decomposition 
     * to improve the thr_eff for small size problem
     * the threshold L1_cache_size is empirical */
    size_t wei_size
            = (size_t)typesize * jcp.ic * jcp.oc * jcp.kh * jcp.kw * jcp.kd;
    size_t out_size = (size_t)jcp.mb * jcp.typesize_out * jcp.oc * jcp.oh
            * jcp.ow * jcp.od;
    size_t inp_size = (size_t)jcp.mb * jcp.typesize_in * jcp.ic * jcp.ih
            * jcp.iw * jcp.id;
    size_t total_size = jcp.ngroups * (wei_size + out_size + inp_size);

    if (jcp.ngroups < jcp.nthr && (total_size < L1_cache_size)) {
        int iw_block = static_cast<int>(jcp.iw_block);
        const int end_nthr
                = with_groups ? static_cast<int>(jcp.ngroups) : ndims - 2;
        float eff = -1.0f;
        float best_thr_eff = -1.0f;
        // When thr_eff equals zero (cannot get the proper effciency)
        // simply set the thread as 4 now
        // And update the eff inside get_iw_block to avoid redundant
        // computation when thr_eff != 0 but current eff == 0
        if (thr_eff == 0.f) {
            jcp.nthr = nstl::min(jcp.nthr, 4);
        } else {
            for (int nthr = jcp.nthr / 2; nthr >= end_nthr; nthr--) {
                iw_block
                        = get_iw_block(jcp.nb_ic_blocking, jcp.ur_w, eff, nthr);
                if (eff > 1.1f * best_thr_eff) {
                    best_thr_eff = eff;
                    jcp.iw_block = iw_block;
                    jcp.nb_iw = div_up(jcp.iw, jcp.iw_block);
                    jcp.nthr = jcp.aligned_threads = nthr;
                    if (best_thr_eff > 0.98f) break;
                }
            }
        }
    }

    VDISPATCH_CONV_IC(
            !(l_overflow * jcp.stride_w > jcp.ur_w && !jcp.large_w_filter),
            VERBOSE_BAD_PARAM, "stride, unroll width");

    r_overflow_no_tail = static_cast<int>(nstl::max<dim_t>(0,
            ((jcp.kw - 1) * (jcp.dilate_w + 1)
                    - nstl::max<dim_t>(0, jcp.r_pad + jcp.ur_w_tail))
                    / jcp.stride_w));
    bool tails_not_ok = false
            /* maximum 1 ur_w block with r_overflow so far */
            || r_overflow_no_tail * jcp.stride_w > jcp.ur_w
            /* ur_w must be a multiple of stride */
            || ((jcp.iw > jcp.ur_w) && (jcp.ur_w % jcp.stride_w != 0))
            /* r_pad must not extend beyond ur_w_tail */
            || ((jcp.iw > jcp.ur_w) && (jcp.r_pad + jcp.ur_w_tail < 0));
    VDISPATCH_CONV_IC(!tails_not_ok, VERBOSE_UNSUPPORTED_FEATURE,
            "tail size unsupported");

    pick_loop_order(jcp);

    jcp.nb_oc_L2 = jcp.nb_oc;
    if (is_data_layout_nxc) {
        // TODO: improve L2 blocking for large OC
        const int nb_oc_theshold_L2 = 32;
        if (jcp.nb_oc > nb_oc_theshold_L2 && jcp.nb_oc < 2 * nb_oc_theshold_L2)
            jcp.nb_oc_L2 = div_up(jcp.nb_oc, 2);
        else
            jcp.nb_oc_L2 = nstl::min<dim_t>(nb_oc_theshold_L2, jcp.nb_oc);
    }

    bool args_ok = true && jcp.ic <= diff_src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= weights_d.padded_dims()[with_groups + 0];
    VDISPATCH_CONV_IC(args_ok, VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "weight and src size mismatch");

    // A rough check on code size
    // TODO: come up with a tighter bound
    {
        const int max_code_size = 256 * 1024; // default size of jit generator
        int mult = 1 + (l_overflow > 0) + (r_overflow_no_tail > 0);
        const float max_instruction_size = 15;
        float ur_fac = (float)jcp.kw * (float)jcp.oc_block
                * (float)jcp.nb_ic_blocking * (float)jcp.ur_w;
        float code_size = (float)mult * ur_fac * max_instruction_size;
        VDISPATCH_CONV_IC(!(code_size > max_code_size && !jcp.large_w_filter),
                "code size limit exceeded");
    }

    return status::success;
}

void jit_avx512_common_conv_bwd_data_kernel_f32_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    UNUSED(scratchpad);
    UNUSED(jcp);
}

// Initialize static data members
const int jit_avx512_common_conv_bwd_weights_kernel_f32_t::max_ur_w = 28;
const int jit_avx512_common_conv_bwd_weights_kernel_f32_t::min_oh_reduce = 9;

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::
        od_step_comeback_pointers() {
    Label kd_comeback_label;

    /* 'depth' loop count bound by 'kd_work_size' */
    mov(kj, reg_kd_count);
    L(kd_comeback_label);
    {
        const dim_t inp_mult = is_src_layout_nxc()
                ? jcp.ngroups * jcp.ic
                : (jcp.is_1stconv ? 1 : jcp.ic_block);
        const dim_t iw = jcp.iw;
        sub(reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mult);
        sub(reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * jcp.ic_block
                        * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kd_comeback_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::
        oh_step_comeback_pointers() {
    Label kh_comeback_label, kd_comeback_label;
    mov(kj, reg_kh);
    L(kh_comeback_label);
    {
        const dim_t kw = jcp.is_hw_transp ? 1 : jcp.kw;
        const dim_t inp_mult = is_src_layout_nxc()
                ? jcp.ngroups * jcp.ic
                : (jcp.is_1stconv ? 1 : jcp.ic_block);
        const dim_t iw = jcp.is_hw_transp ? 1 : jcp.iw;
        sub(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mult);
        sub(reg_kernel, jcp.typesize_out * kw * jcp.ic_block * jcp.oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_comeback_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::compute_ic_block_step_fma(
        int ur_w, dim_t pad_l, dim_t pad_r, int ic_block_step,
        dim_t input_offset, dim_t kernel_offset, dim_t output_offset,
        bool input_wraparound) {

    const dim_t kw = jcp.is_hw_transp ? jcp.tr_kw : jcp.kw;
    const dim_t iw = jcp.is_hw_transp ? jcp.tr_iw : jcp.iw;
    const dim_t kw_tr_mult = jcp.is_hw_transp ? jcp.kw : 1;
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    auto get_ker_offt = [&](dim_t i_kw, dim_t i_ic) {
        return typesize * (i_kw * kw_tr_mult * ic_block + i_ic) * jcp.oc_block
                + kernel_offset;
    };
    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(Zmm(i_kw * ic_block_step + i_ic),
                    EVEX_compress_addr(reg_kernel, get_ker_offt(i_kw, i_ic)));
    const dim_t out_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : oc_block;
    const int oc_tail = jcp.oc_tail;

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        const int ddst_pipeline_start_idx
                = static_cast<int>(ic_block_step * kw);
        static constexpr int ddst_pipeline_len = 4;
        auto get_ddst_reg_idx = [ddst_pipeline_start_idx](int ur_idx) {
            return ddst_pipeline_start_idx + (ur_idx) % ddst_pipeline_len;
        };
        auto get_ddst_offt = [out_mult, output_offset](int ur_idx) {
            return typesize * ur_idx * out_mult + output_offset;
        };

        if (i_ur == 0) {
            for (int i = 0; i < nstl::min(ddst_pipeline_len, ur_w); i++) {
                int ur_idx = i_ur + i;
                auto zmm_ddst = Zmm(get_ddst_reg_idx(ur_idx));
                if (oc_tail) zmm_ddst = zmm_ddst | k_oc_mask | T_z;
                vmovups(zmm_ddst,
                        EVEX_compress_addr(reg_output, get_ddst_offt(ur_idx)));
            }
        } else if (i_ur + ddst_pipeline_len - 1 < ur_w) {

            int ur_idx = i_ur + ddst_pipeline_len - 1;

            auto zmm_ddst = Zmm(get_ddst_reg_idx(ur_idx));
            if (oc_tail) zmm_ddst = zmm_ddst | k_oc_mask | T_z;
            vmovups(zmm_ddst,
                    EVEX_compress_addr(reg_output, get_ddst_offt(ur_idx)));
        }

        for (int i_kw = 0; i_kw < kw; i_kw++) {
            dim_t i_iw = get_iw_idx(i_ur, i_kw, pad_l);
            if (i_iw < 0 || i_iw > get_iw_idx(ur_w - 1, kw - 1, pad_l) - pad_r
                    || get_iw_idx(i_ur, i_kw, jcp.l_pad) >= iw)
                continue;
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                vfmadd231ps(Zmm(i_kw * ic_block_step + i_ic),
                        Zmm(get_ddst_reg_idx(i_ur)),
                        EVEX_compress_addr_safe(reg_input,
                                get_full_src_offset(i_iw, i_ic, input_offset),
                                reg_long_offt, true));
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++)
            vmovups(EVEX_compress_addr(reg_kernel, get_ker_offt(i_kw, i_ic)),
                    Zmm(i_kw * ic_block_step + i_ic));
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::
        compute_ic_block_step_fma_expl(int ur_w, dim_t pad_l, dim_t pad_r,
                int ic_block_step, dim_t input_offset, dim_t kernel_offset,
                dim_t output_offset, bool input_wraparound) {
    dim_t kw = jcp.kw;
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const int oc_tail = jcp.oc_tail;
    const bool ddst_layout_nxc = is_ddst_layout_nxc();
    const int max_regs = 32;
    const int ddst_pipeline_start_idx
            = static_cast<int>(2 * ic_block_step * kw);
    const int ddst_pipeline_len
            = ddst_layout_nxc ? 1 : max_regs - ddst_pipeline_start_idx;
    const dim_t iw_last_value = get_iw_idx(ur_w - 1, kw - 1, pad_l) - pad_r;
    assert(jcp.stride_w == 1 && jcp.dilate_w == 0 && ddst_pipeline_len > 0
            && jcp.kernel_kind == expl_bcast);

    const dim_t out_mult = ddst_layout_nxc ? jcp.ngroups * jcp.oc : oc_block;
    auto get_diff_wei_reg_idx = [ic_block_step](int i_kw, int i_ic) {
        return i_kw * ic_block_step + i_ic;
    };
    auto get_src_reg_idx = [&](dim_t i_iw, int i_ic) {
        return static_cast<int>(kw * ic_block_step
                + ((i_iw + pad_l) % kw) * ic_block_step + i_ic);
    };
    auto get_diff_dst_reg_idx
            = [ddst_pipeline_start_idx, ddst_pipeline_len](int i_ur) {
        return ddst_pipeline_start_idx + i_ur % ddst_pipeline_len;
    };

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto zmm_ker = Zmm(get_diff_wei_reg_idx(i_kw, i_ic));
            vpxord(zmm_ker, zmm_ker, zmm_ker);
        }

    for (int i_ur = 0; i_ur < ur_w; i_ur++) {
        if (i_ur == 0) {
            for (int i = 0; i < nstl::min(ddst_pipeline_len, ur_w); i++) {
                auto addr_out = EVEX_compress_addr(
                        reg_output, typesize * i * out_mult + output_offset);
                auto zmm_ddst = Zmm(get_diff_dst_reg_idx(i));
                if (oc_tail) zmm_ddst = zmm_ddst | k_oc_mask | T_z;
                vmovups(zmm_ddst, addr_out);
            }

            for (int i_kw = 0; i_kw < kw; i_kw++) {
                dim_t i_iw = get_iw_idx(0, i_kw, pad_l);
                if (i_iw < 0 || i_iw > iw_last_value) continue;

                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    auto addr_inp = EVEX_compress_addr_safe(reg_input,
                            get_full_src_offset(i_iw, i_ic, input_offset),
                            reg_long_offt);
                    vbroadcastss(Zmm(get_src_reg_idx(i_iw, i_ic)), addr_inp);
                }
            }
        } else {
            int diff_dst_load_idx = i_ur + ddst_pipeline_len - 1;
            if (diff_dst_load_idx < ur_w) {
                auto addr_out = EVEX_compress_addr(reg_output,
                        typesize * diff_dst_load_idx * out_mult
                                + output_offset);
                auto zmm_ddst = Zmm(get_diff_dst_reg_idx(diff_dst_load_idx));
                if (oc_tail) zmm_ddst = zmm_ddst | k_oc_mask | T_z;
                vmovups(zmm_ddst, addr_out);
            }

            dim_t i_iw = get_iw_idx(i_ur, kw - 1, pad_l);
            if (i_iw >= 0 && i_iw <= iw_last_value) {
                for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                    auto addr_inp = EVEX_compress_addr_safe(reg_input,
                            get_full_src_offset(i_iw, i_ic, input_offset),
                            reg_long_offt);
                    vbroadcastss(Zmm(get_src_reg_idx(i_iw, i_ic)), addr_inp);
                }
            }
        }
        for (int i_kw = 0; i_kw < kw; i_kw++) {
            dim_t i_iw = get_iw_idx(i_ur, i_kw, pad_l);
            if (i_iw < 0 || i_iw > iw_last_value) continue;
            for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
                vfmadd231ps(Zmm(get_diff_wei_reg_idx(i_kw, i_ic)),
                        Zmm(get_diff_dst_reg_idx(i_ur)),
                        Zmm(get_src_reg_idx(i_iw, i_ic)));
            }
        }
    }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto addr_ker = EVEX_compress_addr(reg_kernel,
                    typesize * (i_kw * ic_block + i_ic) * jcp.oc_block
                            + kernel_offset);
            vaddps(Zmm(get_diff_wei_reg_idx(i_kw, i_ic)), addr_ker);
        }

    for (int i_kw = 0; i_kw < kw; i_kw++)
        for (int i_ic = 0; i_ic < ic_block_step; i_ic++) {
            auto addr_ker = EVEX_compress_addr(reg_kernel,
                    typesize * (i_kw * ic_block + i_ic) * jcp.oc_block
                            + kernel_offset);
            vmovups(addr_ker, Zmm(get_diff_wei_reg_idx(i_kw, i_ic)));
        }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::compute_ic_block_step(
        int ur_w, dim_t pad_l, dim_t pad_r, int ic_block_step,
        dim_t input_offset, dim_t kernel_offset, dim_t output_offset,
        bool input_wraparound) {
    if (jcp.kernel_kind == expl_bcast)
        compute_ic_block_step_fma_expl(ur_w, pad_l, pad_r, ic_block_step,
                input_offset, kernel_offset, output_offset, input_wraparound);
    else
        compute_ic_block_step_fma(ur_w, pad_l, pad_r, ic_block_step,
                input_offset, kernel_offset, output_offset, input_wraparound);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t ::
        compute_oh_step_unroll_ow_icblock(int ic_block_step, int max_ur_w) {
    UNUSED(max_ur_w);

    Label kh_label, kd_label;

    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const bool src_layout_nxc = is_src_layout_nxc();
    dim_t inp_mul = src_layout_nxc ? jcp.ngroups * jcp.ic
                                   : (!jcp.is_1stconv ? ic_block : 1);
    dim_t iw = jcp.iw;

    int r_pad = static_cast<int>(nstl::max<dim_t>(0, jcp.r_pad));
    dim_t l_pad = jcp.l_pad;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    const int ic_tail = jcp.ic_tail;
    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
    mov(kj, reg_kh);
    L(kh_label);
    {
        Label icb_block_label, icb_block_label_cb, ic_tail_loop, ic_tail_label;
        if (generate_icb_loop || ic_tail) {
            push(reg_input);
            push(reg_kernel);
            mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
        }

        if (ic_tail) {
            cmp(reg_icb, ic_block);
            jl(ic_tail_loop, T_NEAR);
        }

        const int ic_tail_loop_work = rnd_dn(ic_tail, ic_block_step);
        Label icb_block_label_end;
        L(icb_block_label);
        for (int i_b_ic = 0; i_b_ic < jcp.ic_block; i_b_ic += ic_block_step) {
            const int input_offset = jcp.typesize_in * i_b_ic;
            compute_ic_block_step(jcp.ur_w, l_pad, r_pad, ic_block_step,
                    input_offset, jcp.typesize_out * i_b_ic * jcp.oc_block, 0,
                    i_b_ic + ic_block_step >= jcp.ic_block);
            if (generate_icb_loop || ic_tail) sub(reg_icb, ic_block_step);
            if (ic_tail && i_b_ic + ic_block_step == ic_tail_loop_work) {
                cmp(reg_icb, ic_block_step);
                jl(icb_block_label_end, T_NEAR);
            }
        }
        L(icb_block_label_end);

        const dim_t input_icb_shift = jcp.typesize_in * ic_block;
        const size_t kernel_icb_shift = (size_t)jcp.typesize_out * jcp.kd
                * jcp.kh * jcp.kw * ic_block * oc_block;

        if (generate_icb_loop) {
            // icb loop supported for src in nxc layout only
            assert(src_layout_nxc);
            add(reg_input, input_icb_shift);
            safe_add(reg_kernel, kernel_icb_shift, reg_long_offt);
            cmp(reg_icb, ic_block);
            jge(icb_block_label, T_NEAR);
        }

        if (ic_tail) {
            L(ic_tail_loop);
            Label skip_ic_tail;
            cmp(reg_icb, 0);
            jle(skip_ic_tail, T_NEAR);
            if (ic_tail_loop_work) {
                cmp(reg_icb, ic_tail_loop_work);
                jge(icb_block_label, T_NEAR);
                if (generate_icb_loop) {
                    // compensate offset added in generate_icb_loop
                    sub(reg_input, input_icb_shift);
                    safe_sub(reg_kernel, kernel_icb_shift, reg_long_offt);
                }
            }

            L(ic_tail_label);
            if (ic_tail % ic_block_step) {
                cmp(reg_icb, 0);
                jle(skip_ic_tail, T_NEAR);
                const int i_b_ic = ic_tail_loop_work;
                const int input_offset = jcp.typesize_in * i_b_ic;
                compute_ic_block_step(jcp.ur_w, l_pad, r_pad,
                        ic_tail % ic_block_step, input_offset,
                        jcp.typesize_out * i_b_ic * jcp.oc_block, 0);
            }
            L(skip_ic_tail);
        }

        if (generate_icb_loop || ic_tail) {
            pop(reg_kernel);
            pop(reg_input);
        }

        add(reg_input, jcp.typesize_in * (jcp.dilate_h + 1) * iw * inp_mul);
        add(reg_kernel, jcp.typesize_out * jcp.kw * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }

    if (jcp.ndims == 5) {
        add(aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * iw * inp_mul);
        add(aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t ::
        compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w) {
    Label kh_label, ic_block_label, ic_tail_loop_label, ic_tail_label, kd_label;
    const bool src_layout_nxc = is_src_layout_nxc();
    dim_t inp_mul = src_layout_nxc ? jcp.ngroups * jcp.ic
                                   : (!jcp.is_1stconv ? jcp.ic_block : 1);
    const int ic_tail = jcp.ic_tail;
    UNUSED(max_ur_w);

    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;

    dim_t inp_icb_sp_stride = jcp.is_hw_transp ? 1 : jcp.iw;
    dim_t ow = jcp.is_hw_transp ? jcp.oh : jcp.ow;

    int r_pad = static_cast<int>(nstl::max<dim_t>(0, jcp.r_pad));
    dim_t l_pad = jcp.l_pad;

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
    mov(kj, reg_kh);
    L(kh_label);
    {
        Label icb_block_label;
        if (generate_icb_loop || ic_tail) {
            push(reg_input);
            push(reg_kernel);
            mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
        }

        if (ic_tail) {
            cmp(reg_icb, ic_block);
            jl(ic_tail_loop_label, T_NEAR);
        }

        L(icb_block_label);
        Label icb_block_label_end;
        mov(b_ic, ic_block);
        L(ic_block_label);
        {
            compute_ic_block_step(
                    static_cast<int>(ow), l_pad, r_pad, ic_block_step, 0, 0, 0);
            size_t inp_icblk_stride = jcp.is_1stconv && !src_layout_nxc
                    ? (size_t)jcp.ih * jcp.iw * jcp.id
                    : 1;
            size_t input_offset
                    = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            safe_add(reg_input, input_offset, reg_long_offt);
            add(reg_kernel, jcp.typesize_out * ic_block_step * oc_block);
            sub(b_ic, ic_block_step);
            if (generate_icb_loop || ic_tail) sub(reg_icb, ic_block_step);
            cmp(b_ic, ic_block_step);
            jge(ic_block_label, T_NEAR);
        }
        L(icb_block_label_end);

        const dim_t input_shift = jcp.typesize_in * (jcp.dilate_h + 1)
                * inp_icb_sp_stride * inp_mul;

        if (generate_icb_loop || ic_tail) {
            const size_t kernel_icb_shift = (size_t)jcp.typesize_out * jcp.kd
                    * jcp.kh * jcp.kw * ic_block * oc_block;
            if (generate_icb_loop) {
                // icb loop supported for src in nxc layout only
                assert(src_layout_nxc);
                Label icb_loop_done;
                safe_add(reg_kernel,
                        kernel_icb_shift
                                - jcp.typesize_out * ic_block * oc_block,
                        reg_long_offt);
                cmp(reg_icb, ic_block);
                jge(icb_block_label, T_NEAR);
                L(icb_loop_done);
            }

            L(ic_tail_loop_label);
            if (ic_tail) {
                Label skip_ic_tail;
                const int ic_tail_loop_work = rnd_dn(ic_tail, ic_block_step);
                cmp(reg_icb, 0);
                jle(skip_ic_tail, T_NEAR);
                mov(b_ic, reg_icb);
                if (ic_tail_loop_work) {
                    cmp(reg_icb, ic_block_step);
                    jge(ic_block_label, T_NEAR);
                    if (generate_icb_loop) {
                        // compensate offset added in generate_icb_loop
                        safe_sub(reg_kernel,
                                kernel_icb_shift
                                        - jcp.typesize_out * ic_block
                                                * oc_block,
                                reg_long_offt);
                    }
                }

                L(ic_tail_label);
                if (ic_tail % ic_block_step) {
                    cmp(reg_icb, 0);
                    jle(skip_ic_tail, T_NEAR);
                    compute_ic_block_step(static_cast<int>(ow), l_pad, r_pad,
                            ic_tail % ic_block_step, 0, 0, 0);
                }
                L(skip_ic_tail);
            }

            pop(reg_kernel);
            pop(reg_input);

            add(reg_input, input_shift);
            add(reg_kernel, jcp.typesize_out * jcp.kw * ic_block * oc_block);

        } else if (jcp.is_1stconv && !src_layout_nxc) {
            size_t input_offset = (size_t)jcp.typesize_in * jcp.id * jcp.ih
                    * jcp.iw * ic_block;
            safe_sub(reg_input, input_offset, reg_long_offt);
            add(reg_input, input_shift);
        } else {
            add(reg_input, input_shift - jcp.typesize_in * jcp.ic_block);
        }

        if (!jcp.is_hw_transp && !(generate_icb_loop || ic_tail))
            add(reg_kernel,
                    jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.iw
                        * inp_mul);
        add(aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t ::compute_oh_step_common(
        int ic_block_step, int max_ur_w) {
    using namespace nstl;
    Label kh_label, ic_block_label, ic_tail_loop_label, ic_tail_label, kd_label;

    const bool src_layout_nxc = is_src_layout_nxc();
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;

    int ow = static_cast<int>(jcp.is_hw_transp ? jcp.oh : jcp.ow);
    int r_pad = static_cast<int>(max<dim_t>(0, jcp.r_pad));
    int l_pad = static_cast<int>(jcp.l_pad);

    int ur_w = static_cast<int>(min<dim_t>(ow, max_ur_w));
    dim_t ur_w_trips = ow / ur_w;
    dim_t ur_w_tail = ow % ur_w;
    if ((ur_w_tail == 0 && r_pad != 0) || (r_pad > 0 && r_pad >= ur_w_tail)) {
        if (ur_w_trips > 1) {
            ur_w_tail += ur_w;
            ur_w_trips--;
        } else {
            ur_w_tail += (ur_w - ur_w / 2);
            ur_w = ur_w / 2;
        }
    }

    assert(l_pad <= max_ur_w);
    dim_t inp_mult = src_layout_nxc
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : ic_block * (jcp.is_hw_transp ? jcp.iw : 1));
    dim_t out_mult = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : oc_block;
    dim_t input_comeback
            = max<dim_t>(ur_w_trips * ur_w * jcp.stride_w - l_pad, 0)
            * inp_mult;
    dim_t output_comeback = ur_w_trips * ur_w * out_mult;
    const int ic_tail = jcp.ic_tail;
    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;

    auto ic_loop = [&](int ic_block_step) {
        Label ow_block_label, ic_block_inner_label;
        int ur_w_blocks = static_cast<int>(ur_w_trips);

        dim_t l_pad_tail = max<dim_t>(l_pad - ur_w, 0);
        L(ic_block_inner_label);
        if (l_pad != 0) {
            ur_w_blocks--;
            compute_ic_block_step(ur_w, l_pad, 0, ic_block_step, 0, 0, 0);
            dim_t iw_offset = ur_w * jcp.stride_w - l_pad;
            if (iw_offset > 0)
                add(reg_input, jcp.typesize_in * iw_offset * inp_mult);
            add(reg_output, jcp.typesize_in * ur_w * out_mult);
        }

        assert(IMPLICATION(l_pad_tail > 0, ur_w_blocks <= 1));
        if (ur_w_blocks > 0) {
            xor_(reg_ur_w_trips, reg_ur_w_trips);
            L(ow_block_label);
            {
                compute_ic_block_step(
                        ur_w, l_pad_tail, 0, ic_block_step, 0, 0, 0);
                add(reg_input,
                        jcp.typesize_in * (ur_w * jcp.stride_w - l_pad_tail)
                                * inp_mult);
                add(reg_output, jcp.typesize_in * ur_w * out_mult);

                inc(reg_ur_w_trips);
                cmp(reg_ur_w_trips, ur_w_blocks);
                jl(ow_block_label, T_NEAR);
                l_pad_tail = max<dim_t>(l_pad_tail - ur_w, 0);
            }
        }

        if (ur_w_tail > 0)
            compute_ic_block_step(static_cast<int>(ur_w_tail), l_pad_tail,
                    r_pad, ic_block_step, 0, 0, 0);

        sub(reg_output, jcp.typesize_in * output_comeback);
    };

    if (jcp.ndims == 5) {
        L(kd_label);
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
    }

    mov(kj, reg_kh);
    L(kh_label);
    {
        Label icb_block_label, icb_block_label_cb;
        if (generate_icb_loop || ic_tail) {
            // TODO: May be broadcast work?
            push(reg_input);
            push(reg_kernel);
            mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
        }

        if (ic_tail) {
            cmp(reg_icb, ic_block);
            jl(ic_tail_loop_label, T_NEAR);
        }

        L(icb_block_label);
        mov(b_ic, ic_block);
        L(ic_block_label);
        Label ic_block_label_end;
        {
            ic_loop(ic_block_step);
            sub(reg_input, jcp.typesize_in * input_comeback);
            dim_t inp_icblk_stride = jcp.is_1stconv && !src_layout_nxc
                    ? jcp.ih * jcp.iw * jcp.id
                    : 1;
            size_t input_offset
                    = inp_icblk_stride * jcp.typesize_in * ic_block_step;
            safe_add(reg_input, input_offset, reg_long_offt);
            add(reg_kernel, jcp.typesize_out * ic_block_step * oc_block);
            sub(b_ic, ic_block_step);
            if (generate_icb_loop || ic_tail) sub(reg_icb, ic_block_step);
            cmp(b_ic, ic_block_step);
            jge(ic_block_label, T_NEAR);
        }
        L(ic_block_label_end);

        const dim_t input_shift
                = jcp.typesize_in * (jcp.dilate_h + 1) * jcp.iw * inp_mult;

        if (generate_icb_loop || ic_tail) {
            const size_t kernel_icb_loop_shift_bytes = (size_t)jcp.typesize_out
                    * jcp.kd * jcp.kh * jcp.kw * ic_block * oc_block;

            if (generate_icb_loop) {
                // icb loop supported for src in nxc layout only
                assert(src_layout_nxc);
                safe_add(reg_kernel,
                        kernel_icb_loop_shift_bytes
                                - jcp.typesize_out * ic_block * oc_block,
                        reg_long_offt);

                cmp(reg_icb, ic_block);
                jge(icb_block_label, T_NEAR);
            }

            L(ic_tail_loop_label);
            if (ic_tail) {
                Label skip_ic_tail;
                const int ic_tail_loop_work = rnd_dn(ic_tail, ic_block_step);
                cmp(reg_icb, 0);
                jle(skip_ic_tail, T_NEAR);
                mov(b_ic, reg_icb);
                if (ic_tail_loop_work) {
                    cmp(reg_icb, ic_block_step);
                    jge(ic_block_label, T_NEAR);
                    if (generate_icb_loop) {
                        // compensate offset added in generate_icb_loop
                        safe_sub(reg_kernel,
                                kernel_icb_loop_shift_bytes
                                        - jcp.typesize_out * ic_block
                                                * oc_block,
                                reg_long_offt);
                    }
                }

                L(ic_tail_label);
                if (ic_tail % ic_block_step) {
                    cmp(reg_icb, 0);
                    jle(skip_ic_tail, T_NEAR);
                    ic_loop(ic_tail % ic_block_step);
                }
                L(skip_ic_tail);
            }

            pop(reg_kernel);
            pop(reg_input);

            add(reg_input, input_shift);
            add(reg_kernel, jcp.typesize_out * jcp.kw * ic_block * oc_block);
        } else if (jcp.is_1stconv && !src_layout_nxc) {
            size_t input_offset = (size_t)jcp.typesize_in * jcp.id * jcp.ih
                    * jcp.iw * ic_block;
            safe_sub(reg_input, input_offset, reg_long_offt);
            add(reg_input, input_shift);
        } else if (!jcp.is_hw_transp) {
            add(reg_input, input_shift - jcp.typesize_in * ic_block);
        }
        if (!jcp.is_hw_transp && !(generate_icb_loop || ic_tail))
            add(reg_kernel,
                    jcp.typesize_out * (jcp.kw - 1) * ic_block * oc_block);
        dec(kj);
        cmp(kj, 0);
        jg(kh_label, T_NEAR);
    }
    if (jcp.ndims == 5) {
        add(aux_reg_input,
                jcp.typesize_in * (jcp.dilate_d + 1) * jcp.ih * jcp.iw
                        * inp_mult);
        add(aux_reg_kernel,
                jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block);
        dec(ki);
        cmp(ki, 0);
        jg(kd_label, T_NEAR);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t ::compute_oh_step_disp() {
    int ic_block_step;
    if (jcp.kernel_kind == expl_bcast)
        ic_block_step = jcp.kw <= 3 ? 4 : (jcp.kw <= 7 ? 2 : 1);
    else
        ic_block_step = jcp.kw <= 3 ? 8 : (jcp.kw <= 7 ? 4 : 2);

    if (jcp.is_1stconv) {
        bool large_code = jcp.kw >= 7 && (jcp.l_pad > 0 || jcp.t_pad > 0);
        ic_block_step = (jcp.kw * jcp.ic_block <= 28 && !large_code)
                ? static_cast<int>(jcp.ic_block)
                : 1;
    }

    bool too_large_to_unroll = (jcp.kw > 1 || jcp.kh > 1 || jcp.kd > 1)
            && (jcp.stride_w > 1 || jcp.stride_h > 1 || jcp.stride_d > 1);

    dim_t ow = jcp.is_hw_transp ? jcp.oh : jcp.ow;
    if (jcp.ndims == 5) {
        /* NOTE: reg_kd_count = aux_reg_input = r12. The following order of
         * 'movs' must be guaranteed. */
        mov(ki, reg_kd_count);
        push(reg_kd_count);
        mov(aux_reg_input, reg_input);
        mov(aux_reg_kernel, reg_kernel);
    }

    if (jcp.kw <= 3 && ow <= 16 && !too_large_to_unroll)
        compute_oh_step_unroll_ow_icblock(ic_block_step, max_ur_w);
    else if (ow <= max_ur_w)
        compute_oh_step_unroll_ow(ic_block_step, max_ur_w);
    else
        compute_oh_step_common(ic_block_step, max_ur_w);

    if (jcp.ndims == 5) {
        mov(reg_input, aux_reg_input);
        mov(reg_kernel, aux_reg_kernel);
        pop(reg_kd_count);
        od_step_comeback_pointers();
    } else {
        oh_step_comeback_pointers();
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::maybe_zero_kernel() {
    Label skip_zeroing, zeroing_loop;

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jz(skip_zeroing, T_NEAR);

    Zmm zero = Zmm(0);
    vpxord(zero, zero, zero);
    const bool generate_icb_loop = jcp.nb_ic_blocking_max > 1;
    const dim_t kernel_block_bytes = static_cast<dim_t>(jcp.ic_block)
            * jcp.oc_block * jcp.kw * jcp.kh * jcp.kd * jcp.typesize_out;
    Label icb_block_label, icb_block_label_cb;
    if (generate_icb_loop) {
        push(reg_kernel);

        mov(reg_icb, ptr[param + GET_OFF(reduce_work)]);
        L(icb_block_label);
    }

    xor_(reg_tmp, reg_tmp);
    L(zeroing_loop);
    {
        assert(jcp.oc_block * jcp.typesize_out
                == cpu_isa_traits_t<avx512_core>::vlen);
        for (int ic1 = 0; ic1 < jcp.ic_block; ic1++)
            vmovups(ptr[reg_kernel + reg_tmp
                            + ic1 * jcp.oc_block * jcp.typesize_out],
                    zero);
        add(reg_tmp, jcp.ic_block * jcp.oc_block * jcp.typesize_out);
        cmp(reg_tmp, kernel_block_bytes);
        jnz(zeroing_loop);
    }
    if (generate_icb_loop) {
        add(reg_kernel, kernel_block_bytes);
        sub(reg_icb, jcp.ic_block);
        cmp(reg_icb, 0);
        jg(icb_block_label, T_NEAR);

        pop(reg_kernel);
    }

    L(skip_zeroing);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::bias_kernel_2d() {
    assert(jcp.ndims == 4); // only supports 2d
    Label skip_bias, bias_loop;
    const int oc_tail = jcp.oc_tail;

    mov(reg_tmp, ptr[param1 + GET_OFF(flags)]);
    mov(reg_bias, ptr[param + GET_OFF(bias)]);
    test(reg_tmp, reg_tmp);
    jnz(skip_bias, T_NEAR);

    vmovups(Zmm(0), ptr[reg_bias]);

    mov(reg_oi, jcp.ow);
    xor_(reg_tmp, reg_tmp);
    L(bias_loop);
    {
        auto zmm_out = Zmm(1);
        if (oc_tail) zmm_out = zmm_out | k_oc_mask | T_z;
        vmovups(zmm_out, ptr[reg_output + reg_tmp]);
        vaddps(Zmm(0), Zmm(0), Zmm(1));
        const dim_t oc_stride
                = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
        add(reg_tmp, jcp.typesize_out * oc_stride);
        dec(reg_oi);
        jg(bias_loop);
    }
    vmovups(ptr[reg_bias], Zmm(0));

    L(skip_bias);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::bias_kernel_3d() {
    assert(jcp.ndims == 5); // only supports 3d
    Label skip_bias, bias_loop, skip_load_bias;
    const bool oc_tail = jcp.oc_tail;

    mov(reg_tmp, ptr[param + GET_OFF(flags)]);
    test(reg_tmp, reg_tmp);
    jne(skip_bias, T_NEAR);

    mov(reg_bias, ptr[param + GET_OFF(bias)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    vpxord(Zmm(1), Zmm(1), Zmm(1));

    mov(reg_tmp, ptr[param + GET_OFF(channel)]);
    cmp(reg_tmp, 0);
    jne(skip_load_bias, T_NEAR);
    vmovups(Zmm(1), ptr[reg_bias]);

    L(skip_load_bias);

    mov(reg_oi, ptr[param + GET_OFF(os_index_end)]);
    sub(reg_oi, ptr[param + GET_OFF(os_index_begin)]);
    cmp(reg_oi, 0);
    jle(skip_bias, T_NEAR); // no iterations along depth dimension

    const dim_t oc_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
    mov(reg_tmp, oc_mult * jcp.ow * jcp.oh * jcp.typesize_out);
    imul(reg_oi, reg_tmp);

    xor_(reg_tmp, reg_tmp);
    L(bias_loop);
    {
        auto zmm_out = Zmm(0);
        if (oc_tail) zmm_out = zmm_out | k_oc_mask | T_z;
        vmovups(zmm_out, ptr[reg_output + reg_tmp]);
        vaddps(Zmm(1), Zmm(1), Zmm(0));
        add(reg_tmp, oc_mult * jcp.typesize_out);
        cmp(reg_tmp, reg_oi);
        jl(bias_loop);
    }
    vmovups(ptr[reg_bias], Zmm(1));

    L(skip_bias);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t ::
        compute_oh_loop_common() {
    assert(one_of(jcp.harness, harness_mb_reduction, harness_3d_reduction));
    dim_t b_pad = jcp.b_pad;
    dim_t t_pad = jcp.t_pad;
    bool is_dilated = jcp.dilate_h != 0;
    dim_t dilate_h = jcp.dilate_h + 1;
    dim_t stride_h = jcp.stride_h;
    const dim_t inp_mult = is_src_layout_nxc()
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : jcp.ic_block);
    const dim_t out_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
    dim_t iw = jcp.is_hw_transp ? 1 : jcp.iw;
    Label oh_label, oh_label_end, oh_tpad_label, oh_tpad_tail_label,
            oh_bpad_label, oh_bpad_label_end, oh_dilate_label_shift,
            oh_dilate_label_noshift, oh_dilate_label_end;

    dim_t ow = jcp.is_hw_transp ? jcp.oh : jcp.ow;
    dim_t oh = jcp.is_hw_transp ? jcp.ow : jcp.oh;
    dim_t kw = jcp.is_hw_transp ? jcp.tr_kw : jcp.kw;
    dim_t kh = jcp.is_hw_transp ? jcp.tr_kh : jcp.kh;
    dim_t ih = jcp.is_hw_transp ? jcp.tr_ih : jcp.ih;
    dim_t ihp = jcp.is_hw_transp ? jcp.tr_ih : jcp.ihp;

    assert(IMPLICATION(jcp.is_hw_transp,
            everyone_is(1, oh, stride_h, dilate_h)
                    && everyone_is(0, b_pad, t_pad)));

    mov(reg_kh, kh);
    xor_(reg_oj, reg_oj);
    /* Compute 'top' edge */
    if (t_pad > 0) {
        const dim_t kh_range = 1 + (kh - 1) * dilate_h;
        const dim_t overflow
                = nstl::max<dim_t>(0, kh - div_up(t_pad + ih, dilate_h));
        const dim_t underflow = div_up(t_pad, dilate_h);
        const dim_t initial_inp_ker_overlap = kh - overflow - underflow;
        mov(reg_kh, initial_inp_ker_overlap);
        add(reg_kernel,
                jcp.typesize_out * underflow * kw * jcp.ic_block
                        * jcp.oc_block);
        // generate loop to process kernel while it remains within t_pad + ih
        if (kh_range < t_pad + ih) {
            if (is_dilated) {
                const int tail = static_cast<int>(t_pad % dilate_h);
                const int shift
                        = tail == 0 ? 0 : static_cast<int>(dilate_h - tail);
                mov(reg_tmp, shift);
                if (tail != 0)
                    add(reg_input, jcp.typesize_in * shift * iw * inp_mult);
            }
            L(oh_tpad_label);
            {
                cmp(reg_oj, oh);
                jge(oh_label_end, T_NEAR);

                compute_oh_step_disp();
                add(reg_output, jcp.typesize_in * ow * out_mult);
                if (is_dilated) {
                    inc(reg_tmp);
                    cmp(reg_tmp, dilate_h);
                    jl(oh_dilate_label_shift, T_NEAR);
                    // unshift input as new kernel element enters
                    sub(reg_input,
                            jcp.typesize_in * (dilate_h - 1) * iw * inp_mult);
                    xor_(reg_tmp, reg_tmp);
                }
                // kernel overlap only changes when (t_pad + oj) % dilate_h == 0
                sub(reg_kernel,
                        jcp.typesize_out * stride_h * kw * jcp.ic_block
                                * jcp.oc_block);
                add(reg_kh, stride_h);
                if (is_dilated) {
                    jmp(oh_dilate_label_noshift, T_NEAR);
                    L(oh_dilate_label_shift);
                    // shift input as old kernel element progresses
                    add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
                    L(oh_dilate_label_noshift);
                }
                inc(reg_oj);

                // final number of kernel elements that overlap with input
                const int final_inp_ker_overlap = static_cast<int>(
                        nstl::min<dim_t>(kh, div_up(ih, dilate_h)));
                cmp(reg_kh, final_inp_ker_overlap);
                jl(oh_tpad_label, T_NEAR);
            }
        }
        // need second loop to process kernel if it is larger than the input
        // (does not apply to dilations as they must have unit stride)
        if (kh_range
                >= ih + (t_pad % stride_h == 0 ? stride_h : t_pad % stride_h)) {
            assert(!is_dilated);
            mov(reg_kh, ih);
            L(oh_tpad_tail_label);
            {
                cmp(reg_oj, oh);
                jge(oh_label_end, T_NEAR);

                compute_oh_step_disp();
                add(reg_output, jcp.typesize_in * ow * out_mult);
                sub(reg_kernel,
                        jcp.typesize_out * stride_h * kw * jcp.ic_block
                                * jcp.oc_block);

                inc(reg_oj);
                cmp(reg_oj, nstl::min(utils::div_up(t_pad, stride_h), oh));
                jl(oh_tpad_tail_label, T_NEAR);
            }
        }
        // correct any excess shifts to kernel and input
        // (does not apply to dilations as they must have unit stride,
        //  kernel must fit inside input, and padding is smaller than input)
        if (t_pad <= oh * stride_h) {
            // kernel has moved beyond padding (adjust for stride effects)
            if (t_pad % stride_h != 0) {
                assert(!is_dilated);
                const dim_t inp_corr = stride_h - t_pad % stride_h;
                add(reg_kernel,
                        jcp.typesize_out * inp_corr * kw * jcp.ic_block
                                * jcp.oc_block);
                add(reg_input, jcp.typesize_in * inp_corr * iw * inp_mult);
            }
        } else {
            // kernel still overlaps padding (complete reset)
            assert(!is_dilated);
            sub(reg_kernel,
                    jcp.typesize_out * (t_pad - oh * stride_h) * kw
                            * jcp.ic_block * jcp.oc_block);
        }
    }

    const int oj_end_value = static_cast<int>(nstl::min<dim_t>(oh,
            utils::div_up(
                    nstl::max<dim_t>(0, ihp - b_pad - (kh - 1) * dilate_h),
                    stride_h)));
    cmp(reg_oj, oj_end_value);
    jge(oh_label_end, T_NEAR);

    /* Compute middle block(s) */
    mov(reg_kh, kh);
    L(oh_label);
    {
        compute_oh_step_disp();
        add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
        add(reg_output, jcp.typesize_in * ow * out_mult);

        inc(reg_oj);
        cmp(reg_oj, oj_end_value);
        jl(oh_label, T_NEAR);
    }
    L(oh_label_end);

    /* Compute bottom edge */
    if (b_pad > 0) {
        cmp(reg_oj, oh);
        jge(oh_bpad_label_end, T_NEAR);

        if (is_dilated) {
            mov(reg_kh, kh - 1); // assumes unit stride for dilations
            mov(reg_tmp, 0);
        } else {
            mov(reg_kh, ihp - b_pad);
            imul(reg_tmp, reg_oj, stride_h);
            sub(reg_kh, reg_tmp);
        }
        L(oh_bpad_label);
        {
            compute_oh_step_disp();
            add(reg_input, jcp.typesize_in * stride_h * iw * inp_mult);
            add(reg_output, jcp.typesize_in * ow * out_mult);
            if (is_dilated) {
                inc(reg_tmp);
                cmp(reg_tmp, dilate_h);
                jl(oh_dilate_label_end, T_NEAR);
                xor_(reg_tmp, reg_tmp);
            }
            sub(reg_kh, stride_h);
            cmp(reg_kh, 0);
            jle(oh_bpad_label_end, T_NEAR);
            if (is_dilated) L(oh_dilate_label_end);

            inc(reg_oj);
            cmp(reg_oj, oh);
            jl(oh_bpad_label, T_NEAR);
        }
        L(oh_bpad_label_end);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::
        compute_oh_loop_partial() {
    assert(jcp.harness == harness_2d_reduction);
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const dim_t inp_mult = is_src_layout_nxc()
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : jcp.ic_block);
    const dim_t out_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;
    const dim_t input_bottom_padding_overlap
            = div_up(nstl::max<dim_t>(0, jcp.ih + jcp.t_pad - (jcp.kh - 1)),
                    jcp.stride_h);
    const dim_t bottom_pad_input_correction
            = jcp.ih + jcp.t_pad - input_bottom_padding_overlap * jcp.stride_h;

    const dim_t filter_shift = jcp.typesize_out * jcp.kw * ic_block * oc_block;
    const dim_t input_shift = jcp.typesize_in * jcp.iw * inp_mult;
    const dim_t output_shift = jcp.typesize_out * jcp.ow * out_mult;

    Label loop_begin_label, loop_end_label, common_block_label,
            top_padding_end_label, bottom_padding_end_label,
            bottom_padding_label;

    if (jcp.with_bias) {
        Label skip_zero_bias;
        mov(reg_bias, ptr[param1 + GET_OFF(bias)]);
        mov(reg_tmp, ptr[param1 + GET_OFF(channel)]);
        test(reg_tmp, reg_tmp);
        jz(skip_zero_bias, T_NEAR);
        mov(reg_tmp, ptr[param1 + GET_OFF(flags)]);
        test(reg_tmp, reg_tmp);
        jnz(skip_zero_bias, T_NEAR);
        vpxord(Zmm(1), Zmm(1), Zmm(1));
        vmovups(ptr[reg_bias], Zmm(1));
        L(skip_zero_bias);
    }

    /* Offset filter position to adjust for top padding */
    add(reg_kernel, ptr[param + GET_OFF(kh_offset)]);

    mov(reg_oj, ptr[param + GET_OFF(os_index_begin)]);
    mov(reg_kh, ptr[param + GET_OFF(kh_padding)]);

    cmp(reg_kh, 0);
    jle(loop_end_label, T_NEAR); // no iterations along kh
    cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
    jge(loop_end_label, T_NEAR); // no iterations along height dimension

    L(loop_begin_label);

    if (jcp.with_bias) bias_kernel_2d();
    compute_oh_step_disp();

    /* Compute 'top' edge */
    if (jcp.t_pad > 0) {

        /* Check if within top padding region */
        cmp(reg_oj, div_up(jcp.t_pad, jcp.stride_h));
        jge(top_padding_end_label, T_NEAR);

        /* Increment step counter and adjust filter position */
        sub(reg_kernel, filter_shift * jcp.stride_h);
        add(reg_kh, jcp.stride_h);

        /* Final number of kernel elements that overlap with input */
        const dim_t inp_ker_overlap = nstl::min<dim_t>(jcp.kh, jcp.ih);
        cmp(reg_kh, inp_ker_overlap);
        jle(common_block_label, T_NEAR);

        /* Correct any excess shifts to kernel and input */
        if (jcp.t_pad <= jcp.oh * jcp.stride_h) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.t_pad % jcp.stride_h != 0) {
                dim_t inp_corr = jcp.stride_h - jcp.t_pad % jcp.stride_h;
                add(reg_kernel, filter_shift * inp_corr);
                add(reg_input, input_shift * inp_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub(reg_kernel, (jcp.t_pad - jcp.oh * jcp.stride_h) * filter_shift);
        }

        /* Set filter element count for outside the t_pad region */
        if (jcp.t_pad + jcp.ih < jcp.kh + jcp.stride_h) {
            // filter now overlaps with b_pad
            mov(reg_kh, bottom_pad_input_correction);
        } else {
            mov(reg_kh, inp_ker_overlap);
        }

        jmp(common_block_label);

        L(top_padding_end_label);
    }

    /* Compute 'bottom' edge */
    if (jcp.b_pad > 0) {

        /* Check if within bottom padding region */
        cmp(reg_oj, input_bottom_padding_overlap - 1);
        jl(bottom_padding_end_label, T_NEAR);
        jg(bottom_padding_label, T_NEAR);

        /* Execute overlap correction between the filter and the initial
         * bottom padding region. */
        mov(reg_kh, bottom_pad_input_correction);
        jmp(bottom_padding_end_label, T_NEAR);

        L(bottom_padding_label);
        sub(reg_kh, jcp.stride_h);
        cmp(reg_kh, 0);
        jle(loop_end_label, T_NEAR);

        L(bottom_padding_end_label);
    }

    /* Compute middle block */
    add(reg_input, input_shift * jcp.stride_h);

    /* Execute common block and loop */
    L(common_block_label);
    add(reg_output, output_shift);
    inc(reg_oj);
    cmp(reg_oj, ptr[param + GET_OFF(os_index_end)]);
    jl(loop_begin_label, T_NEAR);

    L(loop_end_label);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::
        compute_od_loop_partial() {
    assert(jcp.harness == harness_3d_reduction);
    const dim_t ic_block = jcp.ic_block;
    const dim_t oc_block = jcp.oc_block;
    const dim_t inp_mult = is_src_layout_nxc()
            ? jcp.ngroups * jcp.ic
            : (jcp.is_1stconv ? 1 : jcp.ic_block);
    const dim_t out_mult
            = is_ddst_layout_nxc() ? jcp.ngroups * jcp.oc : jcp.oc_block;

    dim_t iw = jcp.iw;
    dim_t ow = jcp.ow;
    const dim_t input_backpad_overlap
            = div_up(nstl::max<dim_t>(0, jcp.id + jcp.f_pad - (jcp.kd - 1)),
                    jcp.stride_d);
    const dim_t back_pad_input_correction
            = jcp.id + jcp.f_pad - input_backpad_overlap * jcp.stride_d;

    const dim_t filter_shift
            = jcp.typesize_out * jcp.kh * jcp.kw * ic_block * oc_block;
    const dim_t input_shift = jcp.typesize_in * jcp.ih * iw * inp_mult;
    const dim_t output_shift = jcp.typesize_in * jcp.oh * ow * out_mult;

    Label d_loop_label, loop_end_label, common_block_label, fpad_end_label,
            backpad_end_label, backpad_label;

    if (jcp.with_bias) bias_kernel_3d();

    /* initially offset 'kd' by f_pad */
    add(reg_kernel, ptr[param + GET_OFF(kd_offset)]);

    mov(reg_input_d, ptr[param + GET_OFF(src)]);
    mov(reg_output_d, ptr[param + GET_OFF(dst)]);
    mov(reg_d_index, ptr[param + GET_OFF(os_index_begin)]);
    mov(reg_kd_count, ptr[param + GET_OFF(kd_padding)]);

    cmp(reg_kd_count, 0);
    jle(loop_end_label, T_NEAR); // no iterations along kd
    cmp(reg_d_index, ptr[param + GET_OFF(os_index_end)]);
    jge(loop_end_label, T_NEAR); // no iterations along depth dimension

    L(d_loop_label);

    mov(reg_input, reg_input_d);
    mov(reg_output, reg_output_d);

    push(reg_input_d);
    push(reg_output_d);
    push(reg_d_index);

    compute_oh_loop_common();

    pop(reg_d_index);
    pop(reg_output_d);
    pop(reg_input_d);

    /* Compute 'front' edge */
    if (jcp.f_pad > 0) {

        /* Check if within fpad region */
        cmp(reg_d_index, div_up(jcp.f_pad, jcp.stride_d));
        jge(fpad_end_label, T_NEAR);

        /* Fpad steps */
        sub(reg_kernel, filter_shift * jcp.stride_d);
        add(reg_kd_count, jcp.stride_d);

        /* Final number of kernel elements that overlap with input */
        const int inp_ker_overlap
                = static_cast<int>(nstl::min<dim_t>(jcp.kd, jcp.id));
        cmp(reg_kd_count, inp_ker_overlap);
        jle(common_block_label, T_NEAR);

        /* Correct any excess shifts to kernel and input */
        if (jcp.f_pad <= jcp.od * jcp.stride_d) {
            /* Filter has moved beyond padding (adjust for stride effects) */
            if (jcp.f_pad % jcp.stride_d != 0) {
                const dim_t inp_corr = jcp.stride_d - jcp.f_pad % jcp.stride_d;
                add(reg_kernel, filter_shift * inp_corr);
                add(reg_input_d, input_shift * inp_corr);
            }
        } else {
            /* Filter still overlaps padding (complete reset) */
            sub(reg_kernel, (jcp.f_pad - jcp.od * jcp.stride_d) * filter_shift);
        }

        /* Set filter element count for outside the f_pad region */
        if (jcp.f_pad + jcp.id < jcp.kd + jcp.stride_d) {
            // filter now overlaps with back_pad
            mov(reg_kd_count, back_pad_input_correction);
        } else {
            mov(reg_kd_count, inp_ker_overlap);
        }

        jmp(common_block_label);

        L(fpad_end_label);
    }

    /* Compute bottom edge */
    if (jcp.back_pad > 0) {

        /* Check if within back_pad region */
        cmp(reg_d_index, input_backpad_overlap - 1);
        jl(backpad_end_label, T_NEAR);
        jg(backpad_label, T_NEAR);

        /* Execute overlap correction between the filter and the initial
         * back_pad region. */
        mov(reg_kd_count, back_pad_input_correction);
        jmp(backpad_end_label, T_NEAR);

        L(backpad_label);
        sub(reg_kd_count, jcp.stride_d);
        cmp(reg_kd_count, 0);
        jle(loop_end_label, T_NEAR);

        L(backpad_end_label);
    }

    /* Compute middle block */
    add(reg_input_d, input_shift * jcp.stride_d);

    /* Execute common block and loop */
    L(common_block_label);
    add(reg_output_d, output_shift);
    inc(reg_d_index);
    cmp(reg_d_index, ptr[param + GET_OFF(os_index_end)]);
    jl(d_loop_label, T_NEAR);

    L(loop_end_label);
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::compute_loop() {

    maybe_zero_kernel();

    switch (jcp.harness) {
        case harness_2d_reduction: compute_oh_loop_partial(); break;
        case harness_3d_reduction: compute_od_loop_partial(); break;
        case harness_mb_reduction: compute_oh_loop_common(); break;
        case harness_nxc: break;
        default: assert(!"Invalid harness type");
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::generate_microkernel() {

    reg64_t reg_dwei = abi_param1;
    reg64_t reg_src = abi_param2;
    reg64_t reg_ddst = abi_param3;
    reg64_t reg_iw_base = abi_param4;
    reg64_t aux_reg_icb = r10;
    reg64_t aux_reg_kwb = r11;
    reg64_t reg_src_save = r12;
    reg64_t reg_dwei_save = r13;
    reg64_t reg_iw_base_save = r14;
    reg64_t reg_tmp = r15;

    //Currently kernel is small so passing parameters via registers is preferred
    //whenever possible
#ifdef _WIN32
    // Must be a scratch register since load is before preamble
    reg64_t reg_owb = rax;
    mov(reg_owb, ptr[get_stack_params_address(false)]);
#else
    reg64_t reg_owb = abi_param5;
#endif

    preamble();

    const int kw_unroll = jcp.ur_kw;
    const int ow_unroll = jcp.ur_ow;
    const int iw_unroll = ow_unroll + kw_unroll - 1;
    const int ic_unroll = jcp.ur_ic;

    const int ker_reg_count = ic_unroll;
    const int src_reg_count = iw_unroll * ic_unroll;
    const int ddst_reg_count = ow_unroll;

    MAYBE_UNUSED(ddst_reg_count);
    assert(ker_reg_count + src_reg_count + ddst_reg_count <= 32);

    auto dwei_offset = [&](dim_t i_kw, dim_t i_ic) {
        const dim_t oc_block_size = sizeof(float);
        const dim_t ic_block_size = jcp.oc_block * oc_block_size;
        const dim_t kw_block_size = jcp.ic_block * ic_block_size;
        const dim_t kh_block_size = jcp.kw * kw_block_size;
        const dim_t kd_block_size = jcp.kh * kh_block_size;
        const dim_t icb_block_size = jcp.kd * kd_block_size;

        const dim_t icb = i_ic / jcp.ic_block;
        i_ic = i_ic % jcp.ic_block;

        return icb * icb_block_size + i_kw * kw_block_size
                + i_ic * ic_block_size;
    };

    auto src_offset = [&](int i_ic, dim_t i_iw) -> dim_t {
        const int ic_block_size = sizeof(float);
        const dim_t g_block_size = jcp.ic * ic_block_size;
        const dim_t iw_block_size = jcp.ngroups * g_block_size;

        return i_iw * iw_block_size + i_ic * ic_block_size;
    };

    auto ddst_offset = [&](int i_ow) {
        const int oc_block_size = sizeof(float);
        const dim_t g_block_size = jcp.oc * oc_block_size;
        const dim_t ow_block_size = jcp.ngroups * g_block_size;

        return i_ow * ow_block_size;
    };

    auto get_src_zmm = [ic_unroll, ker_reg_count](int iw_index, int i_ic) {
        int zmm_index = iw_index * ic_unroll + i_ic + ker_reg_count;
        return Zmm(zmm_index);
    };

    auto get_ddst_zmm = [src_reg_count, ker_reg_count](int i_ow) {
        int zmm_index = i_ow + src_reg_count + ker_reg_count;
        return Zmm(zmm_index);
    };

    auto get_ker_zmm = [](int i_ic) { return Zmm(i_ic); };

    auto load_ddsts = [&](int ur_ow) {
        for (int i_ow = 0; i_ow < ur_ow; i_ow++) {
            vmovups(get_ddst_zmm(i_ow), zword[reg_ddst + ddst_offset(i_ow)]);
        }
    };

    auto load_srcs = [&](int ur_iw, int ur_ic, bool is_iw_edge) {
        Label iw_load_end;
        if (is_iw_edge) {
            for_(int i_iw_index = 0; i_iw_index < ur_iw; i_iw_index++)
            for (int i_ic = 0; i_ic < ur_ic; i_ic++) {
                vpxord(get_src_zmm(i_iw_index, i_ic),
                        get_src_zmm(i_iw_index, i_ic),
                        get_src_zmm(i_iw_index, i_ic));
            }
        }

        for (int i_iw_index = 0; i_iw_index < ur_iw; i_iw_index++) {
            Label ic_load_end;
            if (is_iw_edge) {
                cmp(reg_iw_base, jcp.iw - i_iw_index * jcp.stride_w);
                jge(iw_load_end, T_NEAR);
                if (jcp.l_pad > 0) {
                    cmp(reg_iw_base, -i_iw_index * jcp.stride_w);
                    jl(ic_load_end, T_NEAR);
                }
            }
            for (int i_ic = 0; i_ic < ur_ic; i_ic++) {
                vbroadcastss(get_src_zmm(i_iw_index, i_ic),
                        zword[reg_src
                                + src_offset(i_ic, jcp.stride_w * i_iw_index)]);
            }
            L(ic_load_end);
        }
        L(iw_load_end);
    };

    auto compute_kernel = [&](int ur_ow, int ur_ic, int ur_kw, int is_iw_edge) {
        Label kw_loop_end;
        load_srcs(ur_ow + ur_kw - 1, ur_ic, is_iw_edge);

        for (int i_kw = 0; i_kw < ur_kw; i_kw++) {
            for (int i_ic = 0; i_ic < ur_ic; i_ic++) {
                vpxord(get_ker_zmm(i_ic), get_ker_zmm(i_ic), get_ker_zmm(i_ic));
            }
            for (int i_ow = 0; i_ow < ur_ow; i_ow++) {
                for (int i_ic = 0; i_ic < ur_ic; i_ic++) {
                    vfmadd231ps(get_ker_zmm(i_ic),
                            get_src_zmm(i_ow + i_kw, i_ic), get_ddst_zmm(i_ow));
                }
            }
            for (int i_ic = 0; i_ic < ur_ic; i_ic++) {
                const dim_t ker_offset = dwei_offset(i_kw, i_ic);
                vaddps(get_ker_zmm(i_ic), zword[reg_dwei + ker_offset]);
                vmovups(zword[reg_dwei + ker_offset], get_ker_zmm(i_ic));
            }
        }

        L(kw_loop_end);
    };

    auto kw_loop = [&](int ur_ow, int ur_ic, int is_iw_edge) {
        Label kwb_loop_begin, kwb_loop_end;
        const int kw_tail = jcp.kw % kw_unroll;
        const int kw_iter = static_cast<int>(jcp.kw / kw_unroll);

        if (kw_iter > 0) {
            if (kw_iter > 1) {
                mov(aux_reg_kwb, jcp.kw - kw_tail);
                L(kwb_loop_begin);
            }
            compute_kernel(ur_ow, ur_ic, kw_unroll, is_iw_edge);

            if (kw_iter > 1 || kw_tail) {
                add(reg_iw_base, (jcp.dilate_w + 1) * kw_unroll);
                add(reg_src, src_offset(0, (jcp.dilate_w + 1) * kw_unroll));
                add(reg_dwei, dwei_offset(kw_unroll, 0));
            }

            if (kw_iter > 1) {
                sub(aux_reg_kwb, kw_unroll);
                jg(kwb_loop_begin, T_NEAR);
            }
        }

        if (kw_tail) compute_kernel(ur_ow, ur_ic, kw_tail, is_iw_edge);

        L(kwb_loop_end);
    };

    auto ic_loop = [&](int ur_ow, int is_iw_edge) {
        Label icb_loop_begin, icb_loop_end;
        const int ic_tail = jcp.ic % ic_unroll;
        const int ic_iter = static_cast<int>(jcp.ic / ic_unroll);

        if (ic_iter > 0) {
            if (ic_iter > 1 || ic_tail) {
                mov(aux_reg_icb, jcp.ic - ic_tail);
                L(icb_loop_begin);
                // Saving onto the stack here appears to significantly slow down
                // code execution. If this kernel runs out of registers, getting
                // rid of the *_save registers should be possible by using
                // subtracts to restore the value and maintain performance.
                mov(reg_src_save, reg_src);
                mov(reg_dwei_save, reg_dwei);
                mov(reg_iw_base_save, reg_iw_base);
            }

            kw_loop(ur_ow, ic_unroll, is_iw_edge);

            if (ic_iter > 1 || ic_tail) {
                mov(reg_iw_base, reg_iw_base_save);
                mov(reg_dwei, reg_dwei_save);
                mov(reg_src, reg_src_save);

                Label inter_block_increment, increment_finish;
                sub(aux_reg_icb, ic_unroll);
                if (jcp.ic > jcp.ic_block) {
                    const int log2_ic_block = 4;
                    lea(reg_tmp, ptr[aux_reg_icb - jcp.ic - ic_tail]);
                    test(reg_tmp, (1 << log2_ic_block) - 1);
                    jnz(inter_block_increment, T_NEAR);

                    add(reg_dwei,
                            dwei_offset(0, jcp.ic_block)
                                    - dwei_offset(0, jcp.ic_block - ic_unroll));
                    jmp(increment_finish);
                    L(inter_block_increment);
                }
                add(reg_dwei, dwei_offset(0, ic_unroll));
                L(increment_finish);

                add(reg_src, src_offset(ic_unroll, 0));
            }
            if (ic_iter > 1) {
                cmp(aux_reg_icb, 0);
                jg(icb_loop_begin, T_NEAR);
            }
        }

        if (ic_tail) kw_loop(ur_ow, ic_tail, is_iw_edge);

        L(icb_loop_end);
    };

    auto ic_loop_dispatch = [&](int ur_ow) {
        Label iw_edge_case, ic_end;

        const dim_t iw_overflow_bound = jcp.iw - (ur_ow - 1) * jcp.stride_w
                - (jcp.kw - 1) * (jcp.dilate_w + 1);
        cmp(reg_iw_base, iw_overflow_bound);
        jge(iw_edge_case, T_NEAR);
        if (jcp.l_pad > 0) {
            cmp(reg_iw_base, 0);
            jl(iw_edge_case, T_NEAR);
        }

        ic_loop(ur_ow, false);
        jmp(ic_end, T_NEAR);

        L(iw_edge_case);
        ic_loop(ur_ow, true);

        L(ic_end);
    };

    Label ow_end, ow_tail;
    int ow_tail_size = jcp.ow % ow_unroll;
    cmp(reg_owb, jcp.ow - ow_tail_size);
    jge(ow_tail, T_NEAR);

    load_ddsts(ow_unroll);
    ic_loop_dispatch(ow_unroll);
    jmp(ow_end, T_NEAR);

    L(ow_tail);
    load_ddsts(ow_tail_size);
    ic_loop_dispatch(ow_tail_size);

    L(ow_end);

    postamble();
    ret();
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::generate_kernel() {
    preamble();

    mov(reg_input, ptr[param + GET_OFF(src)]);
    mov(reg_output, ptr[param + GET_OFF(dst)]);
    mov(reg_kernel, ptr[param + GET_OFF(filt)]);

    const int oc_tail = jcp.oc_tail;
    if (oc_tail) {
        Label skip;
        Reg32 reg_tail_32 = reg_oc_tail.cvt32();
        if (jcp.nb_oc > 1) {
            kxnorw(k_oc_mask, k_oc_mask, k_oc_mask);
            mov(reg_oc_tail, ptr[param + GET_OFF(load_work)]);
            cmp(reg_oc_tail, 16);
            je(skip, T_NEAR);
        }
        mov(reg_tail_32, (1 << oc_tail) - 1);
        kmovw(k_oc_mask, reg_tail_32);
        L(skip);
    }
    compute_loop();

    postamble();
}

status_t jit_avx512_common_conv_bwd_weights_kernel_f32_t::init_conf(
        jit_conv_conf_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &diff_weights_md,
        memory_desc_t &diff_bias_md, memory_desc_t &diff_dst_md, int nthreads) {
    if (!mayiuse(avx512_core)) return status::unimplemented;

    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper diff_weights_d(&diff_weights_md);
    const memory_desc_wrapper diff_bias_d(&diff_bias_md);
    const memory_desc_wrapper diff_dst_d(&diff_dst_md);

    VDISPATCH_CONV_IC(
            utils::everyone_is(data_type::f32, src_d.data_type(),
                    diff_weights_d.data_type(), diff_dst_d.data_type()),
            VERBOSE_UNSUPPORTED_DT_CFG);

    // Big int (> INT_MAX) values are unsupported and jcp fields may overflow
    // TODO: change data type of jcp fields to size_t
    VDISPATCH_CONV_IC(!has_large_size(cd, src_d, diff_weights_d, diff_dst_d),
            VERBOSE_BAD_PARAM, "large size is not supported");

    const bool with_groups = diff_weights_d.ndims() == src_d.ndims() + 1;
    int ndims = src_d.ndims();

    jcp = zero<decltype(jcp)>();

    jcp.simd_w = cpu_isa_traits_t<avx512_core>::vlen / typesize;
    jcp.nthr = jcp.aligned_threads = nthreads;
    jcp.ndims = ndims;
    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? diff_weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = diff_dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;

    jcp.id = (ndims == 5) ? src_d.dims()[2] : 1;
    jcp.ih = (ndims == 3) ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = (ndims == 5) ? diff_dst_d.dims()[2] : 1;
    jcp.oh = (ndims == 3) ? 1 : diff_dst_d.dims()[ndims - 2];
    jcp.ow = diff_dst_d.dims()[ndims - 1];

    jcp.kd = (ndims == 5) ? diff_weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = (ndims == 3) ? 1 : diff_weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = diff_weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = (ndims == 5) ? cd.padding[0][0] : 0;
    jcp.t_pad = (ndims == 3) ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = (ndims == 5) ? cd.strides[0] : 1;
    jcp.stride_h = (ndims == 3) ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = (ndims == 5) ? cd.dilates[0] : 0;
    jcp.dilate_h = (ndims == 3) ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    const dim_t ext_kw = calculate_extended_filter_size(jcp.kw, jcp.dilate_w);
    const dim_t ext_kh = calculate_extended_filter_size(jcp.kh, jcp.dilate_h);
    const dim_t ext_kd = calculate_extended_filter_size(jcp.kd, jcp.dilate_d);

    bool ok = true
            // general condition to simplify dilations
            && IMPLICATION(jcp.dilate_d != 0, jcp.stride_d == 1)
            && IMPLICATION(jcp.dilate_h != 0, jcp.stride_h == 1)
            // special condition to simplify dilations in compute_oh_loop_common
            && IMPLICATION(jcp.dilate_h != 0, ext_kh <= jcp.ih);

    VDISPATCH_CONV_IC(ok, VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "unsupported shape with 'stride > 1' when 'dilate > 0'");

    jcp.r_pad = nstl::max<dim_t>(0,
            calculate_end_padding(
                    jcp.l_pad, jcp.ow, jcp.iw, jcp.stride_w, ext_kw));
    jcp.b_pad = nstl::max<dim_t>(0,
            calculate_end_padding(
                    jcp.t_pad, jcp.oh, jcp.ih, jcp.stride_h, ext_kh));
    jcp.back_pad = nstl::max<dim_t>(0,
            calculate_end_padding(
                    jcp.f_pad, jcp.od, jcp.id, jcp.stride_d, ext_kd));

    /* XXX: currently, does not support dilation_d > 0 */
    VDISPATCH_CONV_IC(!(ndims == 5 && jcp.dilate_d > 0),
            VERBOSE_UNSUPPORTED_FEATURE, "does not support dilation > 0");

    /* Set bounds for large filter 'kw > 14' support and optimized JIT
     * implementation for small output-width 'ow = 1' */
    const int min_filter_size = 14;
    const int max_filter_size = 20;
    const auto dat_tag_nxc = pick(ndims - 3, nwc, nhwc, ndhwc);
    const auto dat_tag_ncx = pick(ndims - 3, ncw, nchw, ncdhw);
    const auto dat_tag_nCx16c = pick(ndims - 3, nCw16c, nChw16c, nCdhw16c);
    auto curr_src_tag = src_d.matches_one_of_tag(
            dat_tag_nxc, dat_tag_nCx16c, dat_tag_ncx);
    auto curr_dst_tag
            = diff_dst_d.matches_one_of_tag(dat_tag_nxc, dat_tag_nCx16c);
    bool is_data_layout_nxc = IMPLICATION(curr_src_tag != dat_tag_nxc,
                                      src_d.format_kind() == format_kind::any)
            && IMPLICATION(curr_dst_tag != dat_tag_nxc,
                    diff_dst_d.format_kind() == format_kind::any)
            && utils::one_of(dat_tag_nxc, curr_src_tag, curr_dst_tag);

    /* Optimization: when `output-width == 1' deploy a special case of the
     * JIT-Kernel by unrolling with regards to height instead of width for
     * the source and filter tensors. The JIT-Kernel also transposes the
     * strides for the input and filter memory access. */
    jcp.is_hw_transp = !is_data_layout_nxc && ndims == 4
            && jcp.kw >= min_filter_size && jcp.kw < max_filter_size
            && jcp.kh <= min_filter_size && jcp.ow == 1 && jcp.kw == jcp.iw
            && everyone_is(1, jcp.stride_w, jcp.stride_h)
            && everyone_is(0, jcp.dilate_h, jcp.dilate_w)
            && everyone_is(0, jcp.l_pad, jcp.t_pad, jcp.r_pad, jcp.b_pad);
    if (jcp.is_hw_transp) {
        jcp.tr_kw = jcp.kh;
        jcp.tr_kh = jcp.kw;
        jcp.tr_iw = jcp.ih;
        jcp.tr_ih = jcp.iw;
    }

    jcp.ihp = jcp.ih + jcp.t_pad + jcp.b_pad;
    jcp.iwp = jcp.iw + jcp.l_pad + jcp.r_pad;
    jcp.ohp = jcp.oh;
    jcp.owp = jcp.ow;
    jcp.aligned_threads = 0;

    /* check for the 1st convolution */
    jcp.is_1stconv = is_1stconv(jcp);

    jcp.oc_block = jcp.simd_w;

    bool ok_to_pad_channels = true && !is_data_layout_nxc && jcp.ngroups == 1
            && src_d.data_type() == data_type::f32;

    if (ok_to_pad_channels) jcp.oc = rnd_up(jcp.oc, jcp.simd_w);

    VDISPATCH_CONV_IC(
            IMPLICATION(!is_data_layout_nxc, jcp.oc % jcp.oc_block == 0),
            VERBOSE_BLOCKING_FAIL, "bad params oc, oc_block");
    jcp.ic_tail = is_data_layout_nxc ? jcp.ic % jcp.simd_w : 0;
    jcp.oc_tail = is_data_layout_nxc ? jcp.oc % jcp.simd_w : 0;

    auto dst_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
    auto wei_tag = with_groups
            ? pick(ndims - 3, gOIw16i16o, gOIhw16i16o, gOIdhw16i16o)
            : pick(ndims - 3, OIw16i16o, OIhw16i16o, OIdhw16i16o);

    if (diff_dst_md.format_kind == format_kind::any) {
        CHECK(memory_desc_init_by_tag(diff_dst_md, dst_tag));
    } else
        VDISPATCH_CONV_IC(
                curr_dst_tag == dst_tag, VERBOSE_UNSUPPORTED_TAG_S, "dst");
    jcp.dst_tag = dst_tag;

    /* conditions on bias memory */
    jcp.with_bias = cd.diff_bias_desc.format_kind != format_kind::undef;
    if (jcp.with_bias) {
        if (diff_bias_d.format_kind() == format_kind::any)
            CHECK(memory_desc_init_by_tag(diff_bias_md, x));
    }

    jcp.nb_oc = div_up(jcp.oc, jcp.oc_block);

    /* kernel applicability check wrt boundaries
     * the conditions are quite general across the kernels we have,
     * but ideally the check should belong to a specific kernel... */
    const dim_t max_pad_h = ext_kh / 2;
    const bool boundaries_ok = true && jcp.l_pad < ext_kw && jcp.r_pad < ext_kw
            && jcp.t_pad <= max_pad_h && jcp.b_pad <= max_pad_h
            && jcp.f_pad < ext_kd && jcp.back_pad < ext_kd
            && IMPLICATION(jcp.f_pad > 0, jcp.kd < jcp.id + jcp.f_pad)
            && jcp.l_pad <= max_ur_w && jcp.r_pad <= max_ur_w;
    VDISPATCH_CONV_IC(boundaries_ok, VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "padding size unsupported (overflow)");

    /* yet another common check */
    if (!jcp.is_hw_transp && jcp.kw > 14) return status::unimplemented;

    /* setting register strategy */
    const dim_t unroll_dim = jcp.is_hw_transp ? jcp.oh : jcp.ow;
    for (int ur_w = static_cast<int>(nstl::min<dim_t>(max_ur_w, unroll_dim));
            ur_w > 0; --ur_w) {
        if (unroll_dim % ur_w == 0) {
            jcp.ur_w = ur_w;
            break;
        }
    }

    if (jcp.is_1stconv) {
        auto src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_ncx;
        if (src_d.format_kind() == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, src_tag));
        } else {
            // if `ic == 1`, then `nxc` and `ncx` are effectively equivalent
            if (jcp.ic == 1 && one_of(curr_src_tag, dat_tag_nxc, dat_tag_ncx))
                src_tag = curr_src_tag;
            VDISPATCH_CONV_IC(
                    curr_src_tag == src_tag, VERBOSE_UNSUPPORTED_TAG_S, "src");
        }
        jcp.src_tag = src_tag;

        const bool src_ok = IMPLICATION(!is_data_layout_nxc,
                (one_of(jcp.ic, 1, 2, 3) && jcp.ngroups == 1));
        VDISPATCH_CONV_IC(
                src_ok, VERBOSE_BAD_PARAM, "ic, number of groups for src");

        jcp.ic_block = jcp.ic;

        wei_tag = with_groups ? pick(ndims - 3, gOwi16o, gOhwi16o, gOdhwi16o)
                              : pick(ndims - 3, Owi16o, Ohwi16o, Odhwi16o);

        VDISPATCH_CONV_IC(
                init_tag(jcp.wei_tag, diff_weights_md, diff_weights_d, wei_tag)
                        == status::success,
                VERBOSE_UNSUPPORTED_TAG);

        jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);

    } else {
        auto src_tag = is_data_layout_nxc ? dat_tag_nxc : dat_tag_nCx16c;
        if (src_md.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md, src_tag));
        } else
            VDISPATCH_CONV_IC(
                    curr_src_tag == src_tag, VERBOSE_UNSUPPORTED_TAG_S, "src");
        jcp.src_tag = src_tag;

        VDISPATCH_CONV_IC(
                init_tag(jcp.wei_tag, diff_weights_md, diff_weights_d, wei_tag)
                        == status::success,
                VERBOSE_UNSUPPORTED_TAG);

        jcp.ic_block = jcp.simd_w;
        if (ok_to_pad_channels) jcp.ic = rnd_up(jcp.ic, jcp.ic_block);
        jcp.nb_ic = div_up(jcp.ic, jcp.ic_block);
    }

    jcp.typesize_in = typesize;
    jcp.typesize_out = typesize;

    dim_t src_size = static_cast<dim_t>(jcp.mb)
            * (is_data_layout_nxc ? jcp.ic : rnd_up(jcp.ic, jcp.ic_block))
            * jcp.id * jcp.ih * jcp.iw * jcp.typesize_in;

    dim_t diff_dst_size = static_cast<dim_t>(jcp.mb)
            * (is_data_layout_nxc ? jcp.oc : rnd_up(jcp.oc, jcp.oc_block))
            * jcp.id * jcp.ih * jcp.iw * jcp.typesize_in;

    VDISPATCH_CONV_IC(src_size <= INT_MAX, VERBOSE_UNSUPPORTED_FEATURE,
            "src size > INT_MAX is not supported");

    VDISPATCH_CONV_IC(diff_dst_size <= INT_MAX, VERBOSE_UNSUPPORTED_FEATURE,
            "diff_dst size > INT_MAX is not supported");

    bool use_nxc_harness = false;
    if (is_data_layout_nxc) {
        dim_t kernel_size = static_cast<dim_t>(jcp.ic) * jcp.oc * jcp.kd
                * jcp.kh * jcp.kw * jcp.typesize_out;
        dim_t data_size = src_size + diff_dst_size;

        // The advantage of the nxc kernel is cache traversal, this comes at a
        // cost of extra work updating the weights buffers more often. As such,
        // if everything fits in cache, this kernel is at a disadvantage to the
        // inner loop over ow. More optimizing/balancing is required to
        // determine when this is needed for multidimensional kernels because
        // the data reuses within the kernel height/depth dimension make the
        // computation more computationally bound and cache traversal advantage
        // less important. Due to the current blocked weights format, the
        // weights and the data buffers cannot both be traversed optimally, so
        // for performance, the weights must fit in cache.
        const unsigned int L2_cache_size = platform::get_per_core_cache_size(2);
        use_nxc_harness
                = (data_size / nthreads + kernel_size > L2_cache_size / 3)
                && (jcp.oc % jcp.simd_w == 0) && (jcp.ic % jcp.simd_w == 0)
                && jcp.kw > 1 && ndims == 3
                && (kernel_size < L2_cache_size / 2);
    }

    jcp.harness = use_nxc_harness ? harness_nxc
            : ndims == 5          ? harness_3d_reduction
                                  : harness_mb_reduction;
    if (jcp.dilate_h == 0 && jcp.ndims == 4 && jcp.oh > min_oh_reduce
            && !jcp.is_hw_transp && !is_data_layout_nxc)
        jcp.harness = harness_2d_reduction; // 2d harness with oh reduction
    bool args_ok = true
            && IMPLICATION(!is_data_layout_nxc,
                    jcp.ic % jcp.ic_block == 0 && jcp.oc % jcp.oc_block == 0)
            && jcp.ic <= src_d.padded_dims()[1]
            && jcp.oc <= diff_dst_d.padded_dims()[1]
            && jcp.ic <= diff_weights_d.padded_dims()[with_groups + 1]
            && jcp.oc <= diff_weights_d.padded_dims()[with_groups + 0];
    VDISPATCH_CONV_IC(args_ok, VERBOSE_UNSUPPORTED_PAD_FEATURE,
            "weight and src size mismatch");

    int nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b;
    if (jcp.harness == harness_nxc) {
        // The harness_nxc is quite different from the other kernels. The
        // init_conf function should probably be refactored so that it calls
        // functions along the line of tune_nxc, tune_fma which
        // independently tune the kernels for each implementation with tuning
        // common to multiple implementations performed by helper functions.
        // This will help maintainability and help prevent the different
        // implementations from stepping on each other.
        int zmm_regs = 32;

        // Block by ic and kw in the compute kernel to decrease loads from the
        // src buffer
        jcp.ur_ic = 2 - jcp.ic % 2;
        jcp.ur_kw = 1;
        if (jcp.stride_w == jcp.dilate_w + 1) {
            jcp.ur_kw = static_cast<int>(jcp.kw);
            if (jcp.kw > 7) {
                // Blocking by kw is more effective than by ic in the compute
                // kernel since neighbor kw operations share src data
                jcp.ur_ic = 1;
                if (jcp.kw > zmm_regs / (jcp.ur_ic + 1))
                    jcp.ur_kw = jcp.kw % (zmm_regs / (jcp.ur_ic + 1));
            }
        }

        // Unroll by ow to decrease updates to diff_weights. In practice, this
        // should be approximately 1/4 - 1/2 of the zmm registers
        jcp.ur_ow = static_cast<int>(nstl::min<dim_t>(
                (zmm_regs - jcp.ur_kw * jcp.ur_ic) / (jcp.ur_ic + 1), jcp.ow));

        int work_amount_base = static_cast<int>(jcp.mb * jcp.od * jcp.oh);
        int ow_iter = static_cast<int>(div_up(jcp.ow, jcp.ur_ow));
        int nthr_ow = nstl::min(
                jcp.nthr / math::gcd(work_amount_base, jcp.nthr), ow_iter);
        int ow_block = div_up(ow_iter, nthr_ow) * jcp.ur_ow;

        jcp.ow_block = ow_block;
        jcp.nb_ow = div_up(jcp.ow, jcp.ow_block);

        // Choose a simple parallelization method. A more advance may need made
        // later
        const dim_t work_amount = jcp.mb * jcp.od * jcp.oh * jcp.nb_ow;
        nthr_mb = nstl::min(jcp.nthr, static_cast<int>(work_amount));
        nthr_g = 1;
        nthr_oc_b = 1;
        nthr_ic_b = 1;
        nthr = nthr_mb * nthr_g * nthr_oc_b * nthr_ic_b;
    } else { // balancing
        balance(jcp, nthr, nthr_mb, nthr_g, nthr_oc_b, nthr_ic_b, jcp.nthr);
    }

    jcp.nthr = nthr;
    jcp.nthr_mb = nthr_mb;
    jcp.nthr_g = nthr_g;
    jcp.nthr_oc_b = nthr_oc_b;
    jcp.nthr_ic_b = nthr_ic_b;

    jcp.kernel_kind = embd_bcast;
    if (is_data_layout_nxc && jcp.stride_w == 1 && jcp.dilate_w == 0
            && !jcp.is_1stconv) {
        jcp.kernel_kind = expl_bcast;
    }

    jcp.nb_ic_blocking_max = 1;
    if (is_data_layout_nxc && (jcp.ow > max_ur_w || jcp.ndims == 5)) {
        assert(!jcp.is_hw_transp);
        jcp.nb_ic_blocking_max = static_cast<int>(
                nstl::min<dim_t>(8, div_up(jcp.nb_ic, jcp.nthr_ic_b)));
    }

    return status::success;
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::init_scratchpad(
        memory_tracking::registrar_t &scratchpad, const jit_conv_conf_t &jcp) {
    if (jcp.nthr_mb > 1) {
        const auto wei_size = static_cast<size_t>(jcp.ngroups)
                * rnd_up(jcp.oc, jcp.oc_block) * rnd_up(jcp.ic, jcp.ic_block)
                * jcp.kh * jcp.kw * jcp.kd;
        const auto bia_size = jcp.ngroups * rnd_up(jcp.oc, jcp.oc_block);
        const size_t wei_bia_reduction_size = wei_size + bia_size;

        scratchpad.book(key_conv_wei_bia_reduction,
                wei_bia_reduction_size * (jcp.nthr_mb - 1), jcp.typesize_out);
        scratchpad.book<simple_barrier::ctx_t>(
                key_conv_wei_bia_reduction_bctx, 1);
    }

    if (jcp.with_bias && jcp.oc_without_padding % jcp.oc_block != 0) {
        const size_t nelems_padded_bias
                = jcp.ngroups * utils::rnd_up(jcp.oc, jcp.oc_block);
        scratchpad.book(
                key_conv_padded_bias, nelems_padded_bias, jcp.typesize_out);
    }
}

void jit_avx512_common_conv_bwd_weights_kernel_f32_t::balance(
        const jit_conv_conf_t &j, int &nthr_, int &nthr_mb_, int &nthr_g_,
        int &nthr_oc_b_, int &nthr_ic_b_, int nthreads) {
    nthr_ = nthr_mb_ = nthr_g_ = nthr_oc_b_ = nthr_ic_b_ = 1;

    if (nthreads < j.ngroups) {
        /* simplification... fortunately it doesn't hurt much */
        nthr_ = nthr_g_ = nthreads;
        return;
    }

    nthr_g_ = static_cast<int>(j.ngroups);
    const int nthr = nthreads / nthr_g_;

    const dim_t ih = j.is_hw_transp ? j.tr_ih : j.ih;
    const dim_t oh = j.is_hw_transp ? j.ow : j.oh;

    const dim_t ih_reduce = j.harness == harness_2d_reduction ? ih : 1;
    const dim_t oh_reduce = j.harness == harness_2d_reduction ? oh : 1;
    const dim_t ih_no_reduce = j.harness == harness_2d_reduction ? 1 : ih;
    const dim_t oh_no_reduce = j.harness == harness_2d_reduction ? 1 : oh;
    const int nthr_oh_reduce
            = static_cast<int>(nstl::max<dim_t>(1, oh_reduce / min_oh_reduce));

    auto calc_mem_cost = [&](int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        /* calculate per thread memory cost (read/write). high level optimizer
         * tries to minimize memory consumption. few notes:
         *  (n1) unclear why, but that essentially helps first convolution...
         *  (n2) assuming the reduction over minibatch is always there:
         *    - instead of 8 it should be 5 here (write ~= 2 read):
         *      kernel: temporal workspace 1 write
         *      reduction: 1 read from workspace and 1 write to the diff_wei
         *    - but experiments showed 8 works better than 5 or 6... */

        const dim_t src_coef = 1;
        const dim_t dst_coef = 1;
        const dim_t wei_coef = 8;
        const dim_t iw = j.is_hw_transp ? j.tr_iw : j.iw;
        const dim_t ow = j.is_hw_transp ? j.oh : j.ow;

        return 0
                + src_coef * div_up(j.mb * ih_reduce, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_ic, nthr_ic_b)
                * j.ic_block * ih_no_reduce * iw * j.id / j.stride_d
                / j.stride_h / j.stride_w /* (n1) */
                + dst_coef * div_up(j.mb * oh_reduce, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_oc, nthr_oc_b)
                * j.oc_block * oh_no_reduce * ow * j.od
                + wei_coef /* (n2) */
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_oc, nthr_oc_b)
                * div_up(j.nb_ic, nthr_ic_b) * j.kh * j.kw * j.kd * j.ic_block
                * j.oc_block;
    };

    dim_t best_mem_cost = calc_mem_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);

    /* step 1: find the best thread distribution with lowest memory cost */
    const int nthr_mb_max = static_cast<int>(
            nstl::min<dim_t>(nthr, j.mb * j.od * nthr_oh_reduce));
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max
                = static_cast<int>(nstl::min<dim_t>(nthr_par, j.nb_oc));
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = static_cast<int>(
                    nstl::min<dim_t>(nthr_par / nthr_oc_b, j.nb_ic));

            dim_t mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            if (mem_cost <= best_mem_cost) {
                best_mem_cost = mem_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    auto calc_comp_cost = [&j, oh_reduce, nthr_g_](
                                  int nthr_mb, int nthr_oc_b, int nthr_ic_b) {
        return (dim_t)div_up(j.mb * oh_reduce, nthr_mb)
                * div_up(j.ngroups, nthr_g_) * div_up(j.nb_oc, nthr_oc_b)
                * div_up(j.nb_ic, nthr_ic_b);
    };

    /* step 2: search for a thread distribution with lower compute cost.
    * the constrains:
    *  - memory cost cannot exceed 110% of the best found in the step 1
    *  - unless compute cost is 133% lower than the current best case
    * note: both constants were found empirically */
    dim_t best_comp_cost = calc_comp_cost(nthr_mb_, nthr_oc_b_, nthr_ic_b_);
    for (int nthr_mb = 1; nthr_mb <= nthr_mb_max; ++nthr_mb) {
        const int nthr_par = nthr / nthr_mb;
        const int nthr_oc_b_max
                = static_cast<int>(nstl::min<dim_t>(nthr_par, j.nb_oc));
        for (int nthr_oc_b = 1; nthr_oc_b <= nthr_oc_b_max; ++nthr_oc_b) {
            int nthr_ic_b = static_cast<int>(
                    nstl::min<dim_t>(nthr_par / nthr_oc_b, j.nb_ic));
            dim_t mem_cost = calc_mem_cost(nthr_mb, nthr_oc_b, nthr_ic_b);
            dim_t comp_cost = calc_comp_cost(nthr_mb, nthr_oc_b, nthr_ic_b);

            const bool opt1 = comp_cost <= best_comp_cost
                    && IMPLICATION(!j.is_hw_transp,
                            (double)mem_cost < 1.1 * (double)best_mem_cost);
            const bool opt2 = 4 * comp_cost <= 3 * best_comp_cost;

            if (opt1 || opt2) {
                best_comp_cost = comp_cost;
                nthr_mb_ = nthr_mb;
                nthr_oc_b_ = nthr_oc_b;
                nthr_ic_b_ = nthr_ic_b;
            }
        }
    }

    if (nthr_mb_ > nthreads / 2 && nthr_mb_ < nthreads)
        nthr_mb_ = static_cast<int>(
                nstl::min<dim_t>(j.mb * j.od * nthr_oh_reduce, nthreads));
    nthr_ = nthr_mb_ * nthr_g_ * nthr_oc_b_ * nthr_ic_b_;

    assert(nthr_ <= nthreads);
}

template struct jit_avx512_common_conv_fwd_kernel_vmm_t<Zmm>;
template struct jit_avx512_common_conv_fwd_kernel_vmm_t<Ymm>;
template struct jit_avx512_common_conv_bwd_data_kernel_f32_vmm_t<Zmm>;
template struct jit_avx512_common_conv_bwd_data_kernel_f32_vmm_t<Ymm>;

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
