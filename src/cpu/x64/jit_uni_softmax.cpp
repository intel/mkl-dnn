/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_primitive.hpp"

#include "cpu/x64/jit_generator.hpp"

#include "cpu/x64/injectors/jit_uni_postops_injector.hpp"
#include "cpu/x64/jit_uni_softmax.hpp"
#include "cpu/x64/utils/jit_io_helper.hpp"

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER < 1900)
// Intel Compilers 17.x and 18.x do not like that diff_src_ptr() is only used
// in a single descendant class and marks it as unused. This breaks builds
// with DNNL_WERROR=on. Disabling the warning for this file seems to be less
// ugly than all the fixes that I came up with.
#pragma warning disable : 177
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {

namespace softmax_impl {
using namespace Xbyak;
using namespace data_type;

template <cpu_isa_t isa>
struct jit_softmax_dense_kernel_t : jit_softmax_kernel_base_t,
                                    public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_dense_kernel_t)

    using Vmm = typename cpu_isa_traits_t<isa>::Vmm;
    const AddressFrame &vmmword = is_superset(isa, avx512_core) ? zword : yword;
    static constexpr auto vlen = cpu_isa_traits_t<isa>::vlen;
    static constexpr auto n_vregs = cpu_isa_traits_t<isa>::n_vregs;
    static constexpr auto simd_w_ = vlen / sizeof(float); // bf16 works on ymms

    const memory_desc_wrapper src_d_, dst_d_, diff_dst_d_;
    io::jit_io_multi_dt_helper_t<Vmm> io_;

    std::unique_ptr<jit_uni_eltwise_injector_t<Vmm>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_t<Vmm>> log_injector_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<Vmm>>
            postops_injector_;

    Reg64 reg_param = abi_param1;

    Reg64 reg_exp_injector_table = rax;
    Reg64 reg_log_injector_table = rbx;
    Reg64 reg_src = r8;
    Reg64 reg_diff_src = reg_src;
    Reg64 reg_dst = r9;
    Reg64 reg_diff_dst = r14;
    Reg64 reg_src_spat_offt = r10;
    Reg64 reg_process_n_elems = r11;
    Reg64 reg_reverse_n_elems = r12;
    Reg64 reg_tmp = r13;
    Reg64 reg_dst_spat_offt = r15;
    Reg64 reg_diff_dst_spat_offt = reg_log_injector_table;
    Reg64 reg_tmp2 = reg_log_injector_table;
    Reg64 reg_interim = reg_diff_dst;
    Reg64 reg_interim_spat_offt = abi_not_param1;
    Reg64 reg_src_scales = rsi;
    Reg64 reg_dst_scales = rdx;

    Opmask injector_mask = Opmask(1);

    Vmm vtmp; // assigned at placed where used
    Vmm tail_vmask = Vmm(0);
    Xmm xneg_flt_max = Xmm(12);
    Vmm vneg_flt_max = Vmm(is_superset(isa, avx512_core) ? 28 : 12);
    Xmm xone = Xmm(13);
    Vmm vone = Vmm(is_superset(isa, avx512_core) ? 29 : 13);
    Vmm vsum = Vmm(is_superset(isa, avx512_core) ? 30 : 14);
    Vmm vmax = Vmm(is_superset(isa, avx512_core) ? 31 : 15);
    Vmm vsbr = vsum; // must be not equal to vmax
    Vmm vzero = Vmm(is_superset(isa, avx512_core) ? 21 : 11);
    Vmm vcvt_vmm = Vmm(is_superset(isa, avx512_core) ? 22 : 10);
    Vmm vsaturation_ubound = vneg_flt_max;

    bool is_bf16_ = false;
    bool is_f16_ = false;
    bool is_avx2_ne_xf16_ = false;
    bool is_softmax_ = pd_->is_softmax();
    bool is_logsoftmax_ = pd_->is_logsoftmax();
    bool axis_has_padding_;
    bool need_scratchpad_;
    bool with_postops_ = false;
    bool with_binary_ = false;
    bool with_eltwise_ = false;
    bool with_src_scales_ = false;
    bool with_dst_scales_ = false;
    bool use_ext_aux_vmms_ = false;

    int unroll_regs_ = 4;

    dim_t axis_simd_full_;
    int axis_simd_tail_;
    dim_t n_loops_;
    int loop_tail_;
    dim_t process_n_elems_;
    dim_t src_next_vreg_stride_;
    dim_t interim_next_vreg_stride_;
    dim_t dst_next_vreg_stride_;
    dim_t diff_dst_next_vreg_stride_;

    const int bf16_emu_zmm_1_idx_ = 23;
    const int bf16_emu_zmm_2_idx_ = 24;
    const int bf16_emu_zmm_3_idx_ = 25;
    const int bf16_emu_zmm_4_idx_ = 26;
    const int tail_opmask_idx_ = 2;

    const int fp8_emu_zmm_1_idx_ = 23;
    const int fp8_emu_zmm_2_idx_ = 24;
    const int fp8_emu_zmm_3_idx_ = 25;
    const int fp8_emu_zmm_4_idx_ = 26;
    const int fp8_emu_zmm_5_idx_ = 27;
    const int fp8_emu_kmask_idx_ = 3;

    Opmask tail_opmask = Opmask(tail_opmask_idx_);

    void operator()(const call_params_t *p) const override {
        return jit_generator_t::operator()(p);
    }

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

    bool is_data_type_xf16(data_type_t dt) {
        return utils::one_of(dt, bf16, f16);
    }

    void compute_predefined_variables() {
        n_loops_ = axis_simd_full_ / unroll_regs_;
        loop_tail_
                = static_cast<int>(axis_simd_full_ - n_loops_ * unroll_regs_);
        process_n_elems_ = compute_process_n_elems(dst_d_);
        src_next_vreg_stride_ = compute_next_vreg_stride(src_d_);
        interim_next_vreg_stride_ = simd_w_ * sizeof(float);
        dst_next_vreg_stride_ = compute_next_vreg_stride(dst_d_);
        if (!pd_->is_fwd())
            diff_dst_next_vreg_stride_ = compute_next_vreg_stride(diff_dst_d_);
        axis_has_padding_ = pd_->axis_size(true) != pd_->axis_size();
    }

    size_t compute_process_n_elems(const memory_desc_wrapper &mdw) {
        const auto &bd = mdw.blocking_desc();
        if (bd.inner_nblks) return bd.strides[pd_->axis()];
        return simd_w_;
    }

    size_t compute_next_vreg_stride(const memory_desc_wrapper &mdw) {
        const auto &bd = mdw.blocking_desc();
        size_t axis_next_elem_stride = simd_w_;
        if (bd.inner_nblks)
            axis_next_elem_stride
                    = static_cast<size_t>(bd.strides[pd_->axis()]);
        return axis_next_elem_stride * mdw.data_type_size();
    }

    void load_common_params() {
        mov(reg_tmp, float2int(1.0f));
        uni_vmovq(xone, reg_tmp);
        uni_vbroadcastss(vone, xone);
        mov(reg_tmp, float2int(-FLT_MAX));
        uni_vmovq(xneg_flt_max, reg_tmp);
        uni_vbroadcastss(vneg_flt_max, xneg_flt_max);

#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_process_n_elems, ptr[reg_param + PARAM_OFF(process_n_elems)]);
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        if (pd_->is_fwd())
            mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        else {
            mov(reg_diff_src, ptr[reg_param + PARAM_OFF(src)]); // src is reused
            mov(reg_diff_dst, ptr[reg_param + PARAM_OFF(diff_dst)]);
        }
        if (need_scratchpad_) {
            mov(reg_interim, ptr[reg_param + PARAM_OFF(interim)]);
        }
        mov(reg_src_scales, ptr[reg_param + PARAM_OFF(src_scales)]);
        mov(reg_dst_scales, ptr[reg_param + PARAM_OFF(dst_scales)]);
    }

    Address diff_src_ptr(size_t offt = 0) {
        return vmmword[reg_diff_src + reg_src_spat_offt + offt];
    }

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + reg_src_spat_offt + offt];
    }

    Address interim_ptr(size_t offt = 0) {
        return vmmword[reg_interim + reg_interim_spat_offt + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_dst_spat_offt + offt];
    }

    Address diff_dst_ptr(size_t offt = 0) {
        return vmmword[reg_diff_dst + reg_diff_dst_spat_offt + offt];
    }

    enum class op_t : unsigned { max, sum };

    void perform_op(
            const Vmm &vmm_dst, const Vmm &vmm1, const Vmm &vmm2, op_t op) {
        if (op == op_t::max)
            uni_vmaxps(vmm_dst, vmm1, vmm2);
        else if (op == op_t::sum)
            uni_vaddps(vmm_dst, vmm1, vmm2);
    }

    void get_horizontal_op(const Vmm &vsrc, const Vmm &vtmp, op_t op) {
        const Zmm &zsrc = Zmm(vsrc.getIdx());
        const Zmm &ztmp = Zmm(vtmp.getIdx());
        const Ymm &ysrc = Ymm(vsrc.getIdx());
        const Ymm &ytmp = Ymm(vtmp.getIdx());

        if (is_superset(isa, avx512_core)) {
            vshuff32x4(ztmp, zsrc, zsrc, 0x4E); // 256-bit shuffle
            perform_op(vsrc, vsrc, vtmp, op);
            vshuff32x4(ztmp, zsrc, zsrc, 0xB1); // 128/256-bit shuffle
            perform_op(vsrc, vsrc, vtmp, op);
        } else if (is_superset(isa, avx2)) {
            vperm2f128(ytmp, ysrc, ysrc, 0x1); // 128/256-bit shuffle
            perform_op(vsrc, vsrc, vtmp, op);
        }
        uni_vshufps(vtmp, vsrc, vsrc, 0x4E); // 64/128-bit shuffle
        perform_op(vsrc, vsrc, vtmp, op);
        uni_vshufps(vtmp, vsrc, vsrc, 0xB1); // 32/64-bit shuffle
        perform_op(vsrc, vsrc, vtmp, op);
    }

    // This function is responsible for setting the unrolling level which
    // affects vmm indices. Unrolling also affects pre-body and post-body calls
    // that should be done just once but should use a proper unroll value.
    // To avoid leaking of unrolling logic outside from this function, it takes
    // functions to execute pre-body and post-body, too.
    template <typename pre_body_t, typename body_t, typename post_body_t>
    void axis_loop(pre_body_t pre_body, body_t body, post_body_t post_body) {
        Label body_unroll_loop, body_unroll_tail_loop, tail_axis, loop_end;

        // reverse_spat_offt to dispatch between labels
        mov(reg_reverse_n_elems, reg_process_n_elems);
        xor_(reg_src_spat_offt, reg_src_spat_offt); // src/diff_src addr
        xor_(reg_dst_spat_offt, reg_dst_spat_offt); // dst addr
        if (need_scratchpad_)
            xor_(reg_interim_spat_offt, reg_interim_spat_offt); // scratch addr
        if (!pd_->is_fwd())
            xor_(reg_diff_dst_spat_offt, reg_diff_dst_spat_offt); // d_dst addr

        // `pre_body` and `post_body` functions are called with the maximum
        // unroll value of a `body` function as they operate over vmms,
        // which numeration depends on that value.
        // `loop_tail_` is bounded by `unroll_regs_`.
        const int max_body_unroll = n_loops_ ? unroll_regs_
                : loop_tail_                 ? loop_tail_
                                             : 1;
        pre_body(max_body_unroll);

        L(body_unroll_loop);
        {
            if (n_loops_) {
                cmp(reg_reverse_n_elems, unroll_regs_ * process_n_elems_);
                jl(body_unroll_tail_loop, T_NEAR);

                body(unroll_regs_, max_body_unroll, false);
                sub(reg_reverse_n_elems, unroll_regs_ * process_n_elems_);
                add(reg_src_spat_offt, unroll_regs_ * src_next_vreg_stride_);
                add(reg_dst_spat_offt, unroll_regs_ * dst_next_vreg_stride_);
                if (need_scratchpad_)
                    add(reg_interim_spat_offt,
                            unroll_regs_ * interim_next_vreg_stride_);
                if (!pd_->is_fwd())
                    add(reg_diff_dst_spat_offt,
                            unroll_regs_ * diff_dst_next_vreg_stride_);
                jmp(body_unroll_loop);
            }
        }

        L(body_unroll_tail_loop);
        {
            if (loop_tail_) {
                cmp(reg_reverse_n_elems, loop_tail_ * process_n_elems_);
                jl(tail_axis, T_NEAR);

                body(loop_tail_, max_body_unroll, false);
                sub(reg_reverse_n_elems, loop_tail_ * process_n_elems_);
                add(reg_src_spat_offt, loop_tail_ * src_next_vreg_stride_);
                add(reg_dst_spat_offt, loop_tail_ * dst_next_vreg_stride_);
                if (need_scratchpad_)
                    add(reg_interim_spat_offt,
                            loop_tail_ * interim_next_vreg_stride_);
                if (!pd_->is_fwd())
                    add(reg_diff_dst_spat_offt,
                            loop_tail_ * diff_dst_next_vreg_stride_);
            }
        }

        L(tail_axis);
        {
            if (axis_simd_tail_) {
                cmp(reg_reverse_n_elems, 1);
                jl(loop_end, T_NEAR);

                body(1, max_body_unroll, true);
            }
        }

        L(loop_end);

        post_body(max_body_unroll);
    }

    void uni_vaddps_maybe_tail(
            const Vmm &v1, const Vmm &v2, const Vmm &vtmp, const bool tail) {
        if (tail) {
            if (is_superset(isa, avx512_core)) {
                uni_vaddps(v1 | tail_opmask, v1, v2);
            } else {
                uni_vpxor(vtmp, vtmp, vtmp);
                uni_vblendvps(vtmp, vtmp, v2, tail_vmask);
                uni_vaddps(v1, v1, vtmp);
            }
        } else
            uni_vaddps(v1, v1, v2);
    }

    void uni_vmaxps_maybe_tail(
            const Vmm &v1, const Vmm &v2, const Vmm &vtmp, const bool tail) {
        if (tail) {
            if (is_superset(isa, avx512_core)) {
                uni_vmaxps(v1 | tail_opmask, v1, v2);
            } else {
                uni_vblendvps(v2, vneg_flt_max, v2, tail_vmask);
                uni_vmaxps(v1, v1, v2);
            }
        } else
            uni_vmaxps(v1, v1, v2);
    }

    void store(const Address &addr, const Vmm &vmm, data_type_t dt,
            bool tail = false) {
        // Use temporary register in storing when convertion is needed
        // Or we need to restore data back to fp32 since we apply exp after
        // storing and data should be fp32
        const bool need_restore = is_logsoftmax_ && dt != f32;
        Vmm src_vmm = vmm;

        if (tail && axis_has_padding_) {
            if (is_superset(isa, avx512_core)) {
                src_vmm = vcvt_vmm | tail_opmask;
                uni_vxorps(vcvt_vmm, vcvt_vmm, vcvt_vmm);
                uni_vmovups(src_vmm, vmm);
                src_vmm = vcvt_vmm;
            } else {
                uni_vpxor(vcvt_vmm, vcvt_vmm, vcvt_vmm);
                uni_vblendvps(vcvt_vmm, vcvt_vmm, src_vmm, tail_vmask);
                src_vmm = vcvt_vmm;
            }
        } else if (need_restore) {
            uni_vmovups(vcvt_vmm, vmm);
            src_vmm = vcvt_vmm;
        }

        io_[dt]->store(src_vmm, addr, tail && !axis_has_padding_);
    }

    Vmm get_aux_vmm(const Vmm &vmm, int unroll) {
        return Vmm(vmm.getIdx() + unroll);
    }

    // TODO: introduce independent vmax split code for SRF.
    // Use ne_convert instruction to load xf16 even/odd elements from memory
    void accumulate_avx2_ne_xf16_vmax() {
        // flush to -FLT_MAX before accumulation
        uni_vmovups(vmax, vneg_flt_max);

        const auto pre_body = [](int max_unroll) {};

        const auto body = [&](int unroll, int max_unroll, bool tail = false) {
            for (int i = 0; i < unroll; i += 2) {
                const bool can_load_two_simdw = unroll - i >= 2;
                Vmm vreg_tmp_src_even = Vmm(i + 1);
                Vmm vreg_tmp_src_odd = Vmm(i + 2);
                vtmp = Vmm(i + 3);
                if (can_load_two_simdw) {
                    io_[src_d_.data_type()]->load_two_simdw_xf16(
                            src_ptr(src_next_vreg_stride_ * i),
                            vreg_tmp_src_even, vreg_tmp_src_odd);
                } else
                    io_[src_d_.data_type()]->load(
                            src_ptr(src_next_vreg_stride_ * i),
                            vreg_tmp_src_even, tail);
                uni_vmaxps_maybe_tail(vmax, vreg_tmp_src_even, vtmp, tail);
                if (can_load_two_simdw)
                    uni_vmaxps_maybe_tail(vmax, vreg_tmp_src_odd, vtmp, tail);
            }
        };

        const auto post_body = [](int max_unroll) {};

        axis_loop(pre_body, body, post_body);

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    void accumulate_vmax() {
        if (is_avx2_ne_xf16_ && is_data_type_xf16(src_d_.data_type())) {
            accumulate_avx2_ne_xf16_vmax();
            return;
        }

        const auto pre_body = [&](int max_unroll) {
            // flush to -FLT_MAX before accumulation
            for (int i = 0; i < max_unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                Vmm vreg_tmp_max = get_aux_vmm(vreg_tmp_src, max_unroll);
                uni_vmovups(vreg_tmp_max, vneg_flt_max);
            }
        };

        // Each unroll encounter accumulates maximum values into its own vmm.
        // It removes dependency on a single vmm when reading data, but
        // introduces a synchronization between them in a post-body call.
        const auto body = [&](int unroll, int max_unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                Vmm vreg_tmp_max = get_aux_vmm(vreg_tmp_src, max_unroll);
                // do maxps directly from memory on f32 avx2 for performance
                if (!tail && is_superset(isa, avx2)
                        && !is_superset(isa, avx512_core)
                        && src_d_.data_type() == f32) {
                    uni_vmaxps(vreg_tmp_max, vreg_tmp_max,
                            src_ptr(src_next_vreg_stride_ * i));
                } else {
                    io_[src_d_.data_type()]->load(
                            src_ptr(src_next_vreg_stride_ * i), vreg_tmp_src,
                            tail);
                    uni_vmaxps_maybe_tail(
                            vreg_tmp_max, vreg_tmp_src, vtmp = vsum, tail);
                }
            }
        };

        const auto post_body = [&](int max_unroll) {
            assert(utils::one_of(max_unroll, 4, 3, 2, 1));

            Vmm vreg_tmp_max0 = Vmm(0 + max_unroll + 1);
            Vmm vreg_tmp_max1 = Vmm(1 + max_unroll + 1);
            Vmm vreg_tmp_max2 = Vmm(2 + max_unroll + 1);
            Vmm vreg_tmp_max3 = Vmm(3 + max_unroll + 1);

            switch (max_unroll) {
                case 4: {
                    perform_op(vreg_tmp_max0, vreg_tmp_max0, vreg_tmp_max1,
                            op_t::max);
                    perform_op(vreg_tmp_max2, vreg_tmp_max2, vreg_tmp_max3,
                            op_t::max);
                    perform_op(vmax, vreg_tmp_max0, vreg_tmp_max2, op_t::max);
                } break;
                case 3: {
                    perform_op(vreg_tmp_max0, vreg_tmp_max0, vreg_tmp_max1,
                            op_t::max);
                    perform_op(vmax, vreg_tmp_max0, vreg_tmp_max2, op_t::max);
                } break;
                case 2: {
                    perform_op(vmax, vreg_tmp_max0, vreg_tmp_max1, op_t::max);
                } break;
                case 1: {
                    uni_vmovups(vmax, vreg_tmp_max0);
                } break;
                default: break;
            }
        };

        axis_loop(pre_body, body, post_body);

        get_horizontal_op(vmax, vtmp = vsum, op_t::max);
    }

    // TODO: introduce independent vmax split code for SRF.
    // Use ne_convert instruction to load xf16 even/odd elements from memory
    void accumulate_avx2_ne_xf16_vsum() {
        // Initialize saturation vector register
        io_.init_saturate_f32({dst_d_.data_type()});

        uni_vpxor(vsum, vsum, vsum); // flush to zero before accumulation

        const auto pre_body = [](int max_unroll) {};

        const auto body = [&](int unroll, int max_unroll, bool tail = false) {
            for (int i = 0; i < unroll; i += 2) {
                const bool can_load_two_simdw = unroll - i >= 2;
                Vmm vreg_tmp_src_even = Vmm(i + 1);
                Vmm vreg_tmp_src_odd = Vmm(i + 2);
                vtmp = Vmm(i + 3);
                if (can_load_two_simdw) {
                    io_[src_d_.data_type()]->load_two_simdw_xf16(
                            src_ptr(src_next_vreg_stride_ * i),
                            vreg_tmp_src_even, vreg_tmp_src_odd);
                    io_[src_d_.data_type()]->merge_interleaved_to_plain(
                            vreg_tmp_src_even, vreg_tmp_src_odd, vtmp);
                } else
                    io_[src_d_.data_type()]->load(
                            src_ptr(src_next_vreg_stride_ * i),
                            vreg_tmp_src_even, tail);
                for (int i_odd = 0; i_odd < 2 && i_odd + i < unroll; i_odd++) {
                    const auto vreg_tmp_src
                            = i_odd ? vreg_tmp_src_odd : vreg_tmp_src_even;
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                    if (is_logsoftmax_) { // store before applying exp
                        if (need_scratchpad_)
                            store(interim_ptr(interim_next_vreg_stride_
                                          * (i + i_odd)),
                                    vreg_tmp_src, f32, tail);
                        else
                            store(dst_ptr(dst_next_vreg_stride_ * (i + i_odd)),
                                    vreg_tmp_src, dst_d_.data_type(), tail);
                    }
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                    uni_vaddps_maybe_tail(vsum, vreg_tmp_src, vtmp, tail);
                    if (is_softmax_) { // store after applying exp
                        if (need_scratchpad_)
                            store(interim_ptr(interim_next_vreg_stride_
                                          * (i + i_odd)),
                                    vreg_tmp_src, f32, tail);
                        else
                            store(dst_ptr(dst_next_vreg_stride_ * (i + i_odd)),
                                    vreg_tmp_src, dst_d_.data_type(), tail);
                    }
                }
            }
        };

        const auto post_body = [](int max_unroll) {};

        axis_loop(pre_body, body, post_body);

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);

        if (pd_->alg_kind() == alg_kind::softmax_accurate_inf_as_zero) {
            Xbyak::Label skip_div;
            // `vptest` sets the `ZF` flag if all bits in the result are 0 of
            // the bitwise AND of source operands.
            // Note: using Vmm(1) is an ugly workaround EVEX versus VEX encoding
            // as `vsum` uses index `30` on avx512_core. `Vmm(1)` is a tmp vreg
            // used to read data just above. Should be safe.
            uni_vmovups(Vmm(1), vsum);
            uni_vptest(Xmm(1), Xmm(1));
            jz(skip_div, T_NEAR); // Check if ZF is set.
            uni_vdivps(vsum, vone, vsum, vtmp = vmax);
            L(skip_div);
        } else if (is_softmax_) {
            uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        } else if (is_logsoftmax_) {
            log_injector_->compute_vector(vsum.getIdx());
        }
    }

    void accumulate_vsum() {
        if (is_avx2_ne_xf16_ && is_data_type_xf16(src_d_.data_type())) {
            accumulate_avx2_ne_xf16_vsum();
            return;
        }

        // Initialize saturation vector register
        io_.init_saturate_f32({dst_d_.data_type()});

        const auto pre_body = [&](int max_unroll) {
            // flush to zero before accumulation
            for (int i = 0; i < max_unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                Vmm vreg_tmp_max = get_aux_vmm(vreg_tmp_src, max_unroll);
                uni_vpxor(vreg_tmp_max, vreg_tmp_max, vreg_tmp_max);
            }
        };

        // Each unroll encounter accumulates maximum values into its own vmm.
        // It removes dependency on a single vmm when reading data, but
        // introduces a synchronization between them in a post-body call.
        const auto body = [&](int unroll, int max_unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                io_[src_d_.data_type()]->load(
                        src_ptr(src_next_vreg_stride_ * i), vreg_tmp_src, tail);
                uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                if (is_logsoftmax_) { // store before applying exp
                    if (need_scratchpad_)
                        store(interim_ptr(interim_next_vreg_stride_ * i),
                                vreg_tmp_src, f32, tail);
                    else
                        store(dst_ptr(dst_next_vreg_stride_ * i), vreg_tmp_src,
                                dst_d_.data_type(), tail);
                }
            }
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                Vmm vreg_tmp_sum = get_aux_vmm(vreg_tmp_src, 1 * max_unroll);
                if (use_ext_aux_vmms_) {
                    // Prepare indices for exp aux vmms.
                    injector_utils::vmm_index_set_t exp_aux_indices;
                    const auto exp_vmm_aux_count
                            = jit_uni_eltwise_injector_t<Vmm>::aux_vecs_count(
                                    isa, alg_kind::eltwise_exp, pd_->is_fwd(),
                                    0.f);
                    for (int j = 0; j < exp_vmm_aux_count; j++) {
                        // Insert the next idx starting after `vreg_tmp_sum`.
                        exp_aux_indices.insert(
                                get_aux_vmm(vreg_tmp_sum, (j + 1) * max_unroll)
                                        .getIdx());
                    }
                    exp_injector_->compute_vector(
                            vreg_tmp_src.getIdx(), exp_aux_indices);
                } else {
                    exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                }
                uni_vaddps_maybe_tail(
                        vreg_tmp_sum, vreg_tmp_src, vtmp = vmax, tail);
            }
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (is_softmax_) { // store after applying exp
                    if (need_scratchpad_)
                        store(interim_ptr(interim_next_vreg_stride_ * i),
                                vreg_tmp_src, f32, tail);
                    else
                        store(dst_ptr(dst_next_vreg_stride_ * i), vreg_tmp_src,
                                dst_d_.data_type(), tail);
                }
            }
        };

        const auto post_body = [&](int max_unroll) {
            assert(utils::one_of(max_unroll, 4, 3, 2, 1));

            Vmm vreg_tmp_sum0 = Vmm(0 + max_unroll + 1);
            Vmm vreg_tmp_sum1 = Vmm(1 + max_unroll + 1);
            Vmm vreg_tmp_sum2 = Vmm(2 + max_unroll + 1);
            Vmm vreg_tmp_sum3 = Vmm(3 + max_unroll + 1);

            switch (max_unroll) {
                case 4: {
                    perform_op(vreg_tmp_sum0, vreg_tmp_sum0, vreg_tmp_sum1,
                            op_t::sum);
                    perform_op(vreg_tmp_sum2, vreg_tmp_sum2, vreg_tmp_sum3,
                            op_t::sum);
                    perform_op(vsum, vreg_tmp_sum0, vreg_tmp_sum2, op_t::sum);
                } break;
                case 3: {
                    perform_op(vreg_tmp_sum0, vreg_tmp_sum0, vreg_tmp_sum1,
                            op_t::sum);
                    perform_op(vsum, vreg_tmp_sum0, vreg_tmp_sum2, op_t::sum);
                } break;
                case 2: {
                    perform_op(vsum, vreg_tmp_sum0, vreg_tmp_sum1, op_t::sum);
                } break;
                case 1: {
                    uni_vmovups(vsum, vreg_tmp_sum0);
                } break;
                default: break;
            }
        };

        axis_loop(pre_body, body, post_body);

        get_horizontal_op(vsum, vtmp = vmax, op_t::sum);

        if (pd_->alg_kind() == alg_kind::softmax_accurate_inf_as_zero) {
            Xbyak::Label skip_div;
            // `vptest` sets the `ZF` flag if all bits in the result are 0 of
            // the bitwise AND of source operands.
            // Note: using Vmm(1) is an ugly workaround EVEX versus VEX encoding
            // as `vsum` uses index `30` on avx512_core. `Vmm(1)` is a tmp vreg
            // used to read data just above. Should be safe.
            uni_vmovups(Vmm(1), vsum);
            uni_vptest(Xmm(1), Xmm(1));
            jz(skip_div, T_NEAR); // Check if ZF is set.
            uni_vdivps(vsum, vone, vsum, vtmp = vmax);
            L(skip_div);
        } else if (is_softmax_) {
            uni_vdivps(vsum, vone, vsum, vtmp = vmax);
        } else if (is_logsoftmax_) {
            log_injector_->compute_vector(vsum.getIdx());
        }
    }

    // Use ne_convert instruction to load xf16 even/odd elements from memory
    void compute_avx2_ne_xf16_dst() {
        const auto pre_body = [](int max_unroll) {};

        const auto body = [&](int unroll, int max_unroll, bool tail = false) {
            for (int i = 0; i < unroll; i += 2) {
                const bool can_load_two_simdw = unroll - i >= 2;
                Vmm vreg_tmp_src_even = Vmm(i + 1);
                Vmm vreg_tmp_src_odd = Vmm(i + 2);
                vtmp = Vmm(i + 3);
                if (can_load_two_simdw) {
                    if (!need_scratchpad_) {
                        io_[dst_d_.data_type()]->load_two_simdw_xf16(
                                dst_ptr(dst_next_vreg_stride_ * i),
                                vreg_tmp_src_even, vreg_tmp_src_odd);
                        io_[dst_d_.data_type()]->merge_interleaved_to_plain(
                                vreg_tmp_src_even, vreg_tmp_src_odd, vtmp);
                    } else {
                        io_[f32]->load(
                                interim_ptr(interim_next_vreg_stride_ * i),
                                vreg_tmp_src_even, tail);
                        io_[f32]->load(interim_ptr(interim_next_vreg_stride_
                                               * (i + 1)),
                                vreg_tmp_src_odd, tail);
                    }
                } else {
                    if (!need_scratchpad_) {
                        io_[dst_d_.data_type()]->load(
                                dst_ptr(dst_next_vreg_stride_ * i),
                                vreg_tmp_src_even, tail);
                    } else {
                        io_[f32]->load(
                                interim_ptr(interim_next_vreg_stride_ * i),
                                vreg_tmp_src_even, tail);
                    }
                }
                for (int i_odd = 0; i_odd < 2 && i_odd + i < unroll; i_odd++) {
                    const auto vreg_tmp_src
                            = i_odd ? vreg_tmp_src_odd : vreg_tmp_src_even;
                    if (is_softmax_)
                        uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                    if (is_logsoftmax_)
                        uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);

                    if (with_src_scales_) {
                        Vmm vscale = vmax;
                        uni_vbroadcastss(vscale, ptr[reg_src_scales]);
                        uni_vmulps(vreg_tmp_src, vreg_tmp_src, vscale);
                    }
                    if (with_postops_) {
                        binary_injector::rhs_arg_dynamic_params_t
                                rhs_arg_params;
                        if (with_binary_) {
                            rhs_arg_params.vmm_idx_to_out_addr.emplace(
                                    vreg_tmp_src.getIdx(), dst_ptr());
                            rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                                    vreg_tmp_src.getIdx(),
                                    dst_next_vreg_stride_ * (i + i_odd));
                            if (tail)
                                rhs_arg_params.vmm_tail_idx_.emplace(
                                        vreg_tmp_src.getIdx());
                        }
                        postops_injector_->compute_vector(
                                vreg_tmp_src.getIdx(), rhs_arg_params);
                    }
                    if (with_dst_scales_) {
                        Vmm vscale = vmax;
                        uni_vbroadcastss(vscale, ptr[reg_dst_scales]);
                        uni_vmulps(vreg_tmp_src, vreg_tmp_src, vscale);
                    }

                    store(dst_ptr(dst_next_vreg_stride_ * (i + i_odd)),
                            vreg_tmp_src, dst_d_.data_type(), tail);
                }
            }
        };

        const auto post_body = [](int max_unroll) {};

        axis_loop(pre_body, body, post_body);
    }

    void compute_dst() {
        if (is_avx2_ne_xf16_ && is_data_type_xf16(dst_d_.data_type())) {
            compute_avx2_ne_xf16_dst();
            return;
        }

        const auto pre_body = [](int max_unroll) {};

        const auto body = [&](int unroll, int max_unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                if (need_scratchpad_)
                    io_[f32]->load(interim_ptr(interim_next_vreg_stride_ * i),
                            vreg_tmp_src, tail);
                else
                    io_[dst_d_.data_type()]->load(
                            dst_ptr(dst_next_vreg_stride_ * i), vreg_tmp_src,
                            tail);
            }
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                Vmm vreg_tmp_scale = get_aux_vmm(vreg_tmp_src, 1 * max_unroll);

                if (is_softmax_) uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                if (is_logsoftmax_)
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);

                if (with_src_scales_) {
                    uni_vbroadcastss(vreg_tmp_scale, ptr[reg_src_scales]);
                    uni_vmulps(vreg_tmp_src, vreg_tmp_src, vreg_tmp_scale);
                }
                if (with_postops_) {
                    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
                    if (with_binary_) {
                        rhs_arg_params.vmm_idx_to_out_addr.emplace(
                                vreg_tmp_src.getIdx(), dst_ptr());
                        rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                                vreg_tmp_src.getIdx(),
                                dst_next_vreg_stride_ * i);
                        if (tail)
                            rhs_arg_params.vmm_tail_idx_.emplace(
                                    vreg_tmp_src.getIdx());
                    }
                    postops_injector_->compute_vector(
                            vreg_tmp_src.getIdx(), rhs_arg_params);
                }
                if (with_dst_scales_) {
                    uni_vbroadcastss(vreg_tmp_scale, ptr[reg_dst_scales]);
                    uni_vmulps(vreg_tmp_src, vreg_tmp_src, vreg_tmp_scale);
                }
            }
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                store(dst_ptr(dst_next_vreg_stride_ * i), vreg_tmp_src,
                        dst_d_.data_type(), tail);
            }
        };

        const auto post_body = [](int max_unroll) {};

        axis_loop(pre_body, body, post_body);
    }

    void accumulate_vsbr() {
        uni_vpxor(vsbr, vsbr, vsbr); // flush to zero before accumulation

        const auto pre_body = [](int max_unroll) {};

        const auto body = [&](int unroll, int max_unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_dst = Vmm(i * 2 + 1);
                Vmm vreg_tmp_diff_dst = Vmm(i * 2 + 2);
                io_[diff_dst_d_.data_type()]->load(
                        diff_dst_ptr(diff_dst_next_vreg_stride_ * i),
                        vreg_tmp_diff_dst, tail);
                if (is_softmax_) {
                    io_[dst_d_.data_type()]->load(
                            dst_ptr(dst_next_vreg_stride_ * i), vreg_tmp_dst,
                            tail);
                    uni_vmulps(
                            vreg_tmp_diff_dst, vreg_tmp_diff_dst, vreg_tmp_dst);
                }
                uni_vaddps(vsbr, vsbr, vreg_tmp_diff_dst);
            }
        };

        const auto post_body = [](int max_unroll) {};

        axis_loop(pre_body, body, post_body);

        get_horizontal_op(vsbr, vtmp = vmax, op_t::sum);
    }

    void compute_diff_src() {
        const auto pre_body = [](int max_unroll) {};

        const auto body = [&](int unroll, int max_unroll, bool tail = false) {
            for (int i = 0; i < unroll; i++) {
                Vmm vreg_tmp_dst = Vmm(i * 2 + 1);
                Vmm vreg_tmp_diff_dst = Vmm(i * 2 + 2);
                io_[dst_d_.data_type()]->load(
                        dst_ptr(dst_next_vreg_stride_ * i), vreg_tmp_dst, tail);
                io_[diff_dst_d_.data_type()]->load(
                        diff_dst_ptr(diff_dst_next_vreg_stride_ * i),
                        vreg_tmp_diff_dst, tail);
                if (is_softmax_) {
                    uni_vsubps(vreg_tmp_diff_dst, vreg_tmp_diff_dst, vsbr);
                    uni_vmulps(
                            vreg_tmp_diff_dst, vreg_tmp_dst, vreg_tmp_diff_dst);
                }
                if (is_logsoftmax_) {
                    exp_injector_->compute_vector(vreg_tmp_dst.getIdx());
                    uni_vfnmadd231ps(vreg_tmp_diff_dst, vreg_tmp_dst, vsbr);
                }
                store(diff_src_ptr(src_next_vreg_stride_ * i),
                        vreg_tmp_diff_dst, src_d_.data_type(), tail);
            }
        };

        const auto post_body = [](int max_unroll) {};

        axis_loop(pre_body, body, post_body);
    }

    void forward() {
        accumulate_vmax();
        accumulate_vsum();
        compute_dst();
    }

    void backward() {
        accumulate_vsbr();
        compute_diff_src();
    }

    // either this stub or duplication at each jit_binary_t ctor due to methods
    // that are participated are not defined at the moment of base ctor
    // initialization.
    void generate() override {
        if (pd_->is_fwd() || is_logsoftmax_)
            exp_injector_.reset(new jit_uni_eltwise_injector_t<Vmm>(this,
                    alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, data_type::f32,
                    !use_ext_aux_vmms_, reg_exp_injector_table, injector_mask));
        if (pd_->is_fwd() && is_logsoftmax_) {
            log_injector_.reset(new jit_uni_eltwise_injector_t<Vmm>(this,
                    alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, data_type::f32,
                    true, reg_log_injector_table, injector_mask));
        }
        if (with_postops_) {
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            static constexpr std::size_t tmp_vmm_injector = 0u;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    tmp_vmm_injector, this->r14, this->r15, this->r13,
                    preserve_gpr, preserve_vmm,
                    PARAM_OFF(post_ops_binary_rhs_arg_vec), PARAM_OFF(dst_orig),
                    dst_d_, axis_simd_tail_, tail_opmask,
                    use_exact_tail_scalar_bcast};

            const binary_injector::static_params_t bsp {
                    reg_param, get_supported_bcast_strategies(), rhs_sp};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<Vmm>>(this,
                    pd_->attr()->post_ops_, bsp, /* inject_sum = */ false);
        }
#undef PARAM_OFF

        compute_predefined_variables();
        preamble();
        io_.init_bf16();
        if (exp_injector_) exp_injector_->load_table_addr();
        if (log_injector_) log_injector_->load_table_addr();
        if (axis_simd_tail_) io_.prepare_tail_mask();
        load_common_params();
        if (pd_->is_fwd())
            forward();
        else
            backward();
        postamble();
        if (exp_injector_) exp_injector_->prepare_table();
        if (log_injector_) log_injector_->prepare_table();
        if (with_eltwise_ && postops_injector_)
            postops_injector_->prepare_table(/* generate = */ true);
        io_.prepare_table_fp8();
    }

    jit_softmax_dense_kernel_t(const softmax_pd_t *pd)
        : jit_softmax_kernel_base_t(pd)
        , jit_generator_t(jit_name(), isa)
        , src_d_(pd_->invariant_src_md())
        , dst_d_(pd_->dst_md())
        , diff_dst_d_(pd_->diff_dst_md())
        , is_bf16_(utils::one_of(bf16, src_d_.data_type(), dst_d_.data_type()))
        , is_f16_(utils::one_of(f16, src_d_.data_type(), dst_d_.data_type()))
        , is_avx2_ne_xf16_(mayiuse(avx2_vnni_2) && !mayiuse(avx512_core)
                  && (is_bf16_ || is_f16_))
        // Note: must be aligned with pd_t::init()->init_scratchpad();
        , need_scratchpad_(pd_->is_fwd() && dst_d_.data_type() != f32
                  && /* !relaxed_acc */ !(
                          src_d_.data_type() == dst_d_.data_type()
                          && !types::is_integral_dt(dst_d_.data_type())
                          && utils::one_of(pd_->attr()->acc_mode_,
                                  accumulation_mode::relaxed,
                                  accumulation_mode::any)))
        , use_ext_aux_vmms_(!is_logsoftmax_ && n_vregs > 16)
        , axis_simd_full_(pd_->axis_size() / simd_w_)
        , axis_simd_tail_(static_cast<int>(pd_->axis_size() % simd_w_)) {

        const auto &post_ops = pd_->attr()->post_ops_;
        with_postops_ = post_ops.len() != 0;
        with_binary_ = post_ops.find(primitive_kind::binary) != -1;
        with_eltwise_ = post_ops.find(primitive_kind::eltwise) != -1;

        const auto &attr_scales = pd_->attr()->scales_;
        with_src_scales_ = is_superset(isa, avx2)
                && !attr_scales.has_default_values(DNNL_ARG_SRC);
        with_dst_scales_ = is_superset(isa, avx2)
                && !attr_scales.has_default_values(DNNL_ARG_DST);

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx_, tail_vmask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx_,
                bf16_emu_zmm_2_idx_, bf16_emu_zmm_3_idx_, reg_tmp,
                bf16_emu_zmm_4_idx_);
        io::io_saturation_conf_t io_saturation_conf(
                vzero.getIdx(), vsaturation_ubound.getIdx(), reg_tmp);
        io::io_emu_fp8_conf_t io_fp8_conf(fp8_emu_zmm_1_idx_,
                fp8_emu_zmm_2_idx_, fp8_emu_zmm_3_idx_, fp8_emu_zmm_4_idx_,
                fp8_emu_zmm_5_idx_, fp8_emu_kmask_idx_, reg_tmp);
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, isa,
                {src_d_.data_type(), dst_d_.data_type(), f32 /* stats */},
                io_conf, io_tail_conf, io_bf16_conf,
                {{dst_d_.data_type(), io_saturation_conf}}, utils::nullopt,
                io_fp8_conf);
    }
};

template <cpu_isa_t isa>
struct jit_softmax_strided_kernel_t : jit_softmax_kernel_base_t,
                                      public jit_generator_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_strided_kernel_t)

    using Vmm = typename cpu_isa_traits_t<isa>::Vmm;
    const AddressFrame &vmmword = is_superset(isa, avx512_core) ? zword : yword;
    static constexpr auto vlen = cpu_isa_traits_t<isa>::vlen;
    static constexpr int simd_w_ = vlen / sizeof(float); // bf16 works on ymms

    const memory_desc_wrapper src_d_, dst_d_;
    io::jit_io_multi_dt_helper_t<Vmm> io_;

    std::unique_ptr<jit_uni_eltwise_injector_t<Vmm>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_t<Vmm>> log_injector_;
    std::unique_ptr<injector::jit_uni_postops_injector_t<Vmm>>
            postops_injector_;

    Reg64 reg_param = abi_param1;

    // Not used GPRs: abi_not_param, rdx
    Reg64 reg_exp_injector_table = rax;
    Reg64 reg_log_injector_table = rbx;
    Reg64 reg_src = r8;
    Reg64 reg_dst = r9;
    Reg64 reg_src_spat_offt = r10;
    Reg64 reg_dst_spat_offt = r15;
    Reg64 reg_interim_spat_offt = rsi;
    Reg64 reg_reverse_n_elems = r12;
    Reg64 reg_tmp = r13;
    Reg64 reg_tmp2 = reg_log_injector_table;
    Reg64 reg_interim = r14;
    Reg64 reg_reverse_axis_elems = r11;

    Opmask injector_mask = Opmask(1);

    Vmm tail_vmask = Vmm(0);
    Vmm vsrc_scale = Vmm(is_superset(isa, avx512_core) ? 21 : 9);
    Vmm vdst_scale = Vmm(is_superset(isa, avx512_core) ? 22 : 10);
    // 23-26 are reserved for bf16 emulation.
    Vmm vzero = Vmm(is_superset(isa, avx512_core) ? 27 : 11);
    Vmm vsaturation_ubound = Vmm(is_superset(isa, avx512_core) ? 28 : 12);
    Vmm vcvt = Vmm(is_superset(isa, avx512_core) ? 29 : 13);
    Xmm xone = Xmm(14);
    Vmm vone = Vmm(is_superset(isa, avx512_core) ? 30 : 14);
    Xmm xneg_flt_max = Xmm(15);
    Vmm vneg_flt_max = Vmm(is_superset(isa, avx512_core) ? 31 : 15);

    bool is_softmax_ = pd_->is_softmax();
    bool is_logsoftmax_ = pd_->is_logsoftmax();
    bool need_scratchpad_;
    bool with_postops_ = false;
    bool with_binary_ = false;
    bool with_eltwise_ = false;
    bool with_src_scales_ = false;
    bool with_dst_scales_ = false;

    int unroll_inner_size_ = 4;
    int unroll_axis_size_ = 8;

    dim_t axis_size_;
    int axis_size_unroll_tail_;
    dim_t axis_stride_;
    dim_t axis_simd_full_;
    int axis_simd_tail_;
    dim_t n_loops_;
    int loop_tail_;
    dim_t src_next_vreg_stride_;
    dim_t interim_next_vreg_stride_;
    dim_t dst_next_vreg_stride_;

    const int bf16_emu_zmm_1_idx_ = 23;
    const int bf16_emu_zmm_2_idx_ = 24;
    const int bf16_emu_zmm_3_idx_ = 25;
    const int bf16_emu_zmm_4_idx_ = 26;
    const int tail_opmask_idx_ = 2;

    const int fp8_emu_zmm_1_idx_ = 23;
    const int fp8_emu_zmm_2_idx_ = 24;
    const int fp8_emu_zmm_3_idx_ = 25;
    const int fp8_emu_zmm_4_idx_ = 26;
    const int fp8_emu_zmm_5_idx_ = 27;
    const int fp8_emu_kmask_idx_ = 3;

    Opmask tail_opmask = Opmask(tail_opmask_idx_);

    void operator()(const call_params_t *p) const override {
        return jit_generator_t::operator()(p);
    }

    status_t create_kernel() override {
        return jit_generator_t::create_kernel();
    }

    void compute_predefined_variables() {
        // `axis_simd_full_` is actually `inner_simd_full_`.
        n_loops_ = axis_simd_full_ / unroll_inner_size_;
        loop_tail_ = static_cast<int>(
                axis_simd_full_ - n_loops_ * unroll_inner_size_);

        axis_size_unroll_tail_
                = static_cast<int>(axis_size_ % unroll_axis_size_);

        src_next_vreg_stride_ = compute_next_vreg_stride(src_d_);
        interim_next_vreg_stride_ = vlen;
        dst_next_vreg_stride_ = compute_next_vreg_stride(dst_d_);
    }

    size_t compute_next_vreg_stride(const memory_desc_wrapper &mdw) {
        return axis_stride_ * mdw.data_type_size();
    }

    void load_common_params() {
        mov(reg_tmp, float2int(1.0f));
        uni_vmovq(xone, reg_tmp);
        uni_vbroadcastss(vone, xone);
        mov(reg_tmp, float2int(-FLT_MAX));
        uni_vmovq(xneg_flt_max, reg_tmp);
        uni_vbroadcastss(vneg_flt_max, xneg_flt_max);

#define PARAM_OFF(x) offsetof(call_params_t, x)
        mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
        mov(reg_src, ptr[reg_param + PARAM_OFF(src)]);
        if (need_scratchpad_) {
            mov(reg_interim, ptr[reg_param + PARAM_OFF(interim)]);
        }
        if (with_src_scales_) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(src_scales)]);
            uni_vbroadcastss(vsrc_scale, ptr[reg_tmp]);
        }
        if (with_dst_scales_) {
            mov(reg_tmp, ptr[reg_param + PARAM_OFF(dst_scales)]);
            uni_vbroadcastss(vdst_scale, ptr[reg_tmp]);
        }
    }

    Address src_ptr(size_t offt = 0) {
        return vmmword[reg_src + reg_src_spat_offt + offt];
    }

    Address interim_ptr(size_t offt = 0) {
        return vmmword[reg_interim + reg_interim_spat_offt + offt];
    }

    Address dst_ptr(size_t offt = 0) {
        return vmmword[reg_dst + reg_dst_spat_offt + offt];
    }

    void uni_vaddps_maybe_tail(
            const Vmm &v1, const Vmm &v2, const Vmm &vtmp, const bool tail) {
        if (tail) {
            if (is_superset(isa, avx512_core)) {
                uni_vaddps(v1 | tail_opmask, v1, v2);
            } else {
                uni_vpxor(vtmp, vtmp, vtmp);
                uni_vblendvps(vtmp, vtmp, v2, tail_vmask);
                uni_vaddps(v1, v1, vtmp);
            }
        } else
            uni_vaddps(v1, v1, v2);
    }

    void uni_vmaxps_maybe_tail(
            const Vmm &v1, const Vmm &v2, const Vmm &vtmp, const bool tail) {
        if (tail) {
            if (is_superset(isa, avx512_core)) {
                uni_vmaxps(v1 | tail_opmask, v1, v2);
            } else {
                uni_vblendvps(v2, vneg_flt_max, v2, tail_vmask);
                uni_vmaxps(v1, v1, v2);
            }
        } else
            uni_vmaxps(v1, v1, v2);
    }

    void store(const Address &addr, const Vmm &vmm, data_type_t dt,
            bool tail = false) {
        // Use temporary register in storing when conversion is needed
        // Or we need to restore data back to fp32 since we apply exp after
        // storing and data should be fp32
        const bool need_restore = is_logsoftmax_ && dt != f32;
        Vmm src_vmm = vmm;

        if (need_restore) {
            uni_vmovups(vcvt, vmm);
            src_vmm = vcvt;
        }

        io_[dt]->store(src_vmm, addr, tail);
    }

    // The function provides unrolling over axis size. A single compute block
    // is simd_w by N axis points, where axis is strided in memory. The last
    // piece would be `axis_size_ % N`. The purpose of the axis size unrolling
    // is to save on the kernel size, improve the instruction cache rate, and
    // prevent huge stride accesses.
    template <typename body_t>
    void axis_size_loop_unroll(body_t body, int inner_unroll, bool tail) {
        Label axis_size_body, axis_size_tail;

        mov(reg_reverse_axis_elems, axis_size_);

        L(axis_size_body);
        {
            if (axis_size_ >= unroll_axis_size_) {
                cmp(reg_reverse_axis_elems, unroll_axis_size_);
                jl(axis_size_tail, T_NEAR);

                body(unroll_axis_size_, inner_unroll, tail);

                add(reg_src_spat_offt,
                        unroll_axis_size_ * src_next_vreg_stride_);
                add(reg_interim_spat_offt,
                        unroll_axis_size_ * interim_next_vreg_stride_);
                add(reg_dst_spat_offt,
                        unroll_axis_size_ * dst_next_vreg_stride_);

                sub(reg_reverse_axis_elems, unroll_axis_size_);
                jmp(axis_size_body);
            }
        }

        L(axis_size_tail);
        {
            if (axis_size_unroll_tail_) {
                body(axis_size_unroll_tail_, inner_unroll, tail);

                add(reg_src_spat_offt,
                        axis_size_unroll_tail_ * src_next_vreg_stride_);
                add(reg_interim_spat_offt,
                        axis_size_unroll_tail_ * interim_next_vreg_stride_);
                add(reg_dst_spat_offt,
                        axis_size_unroll_tail_ * dst_next_vreg_stride_);
            }
        }
        // Restore initial offsets for the next round.
        sub(reg_src_spat_offt, axis_size_ * src_next_vreg_stride_);
        sub(reg_interim_spat_offt, axis_size_ * interim_next_vreg_stride_);
        sub(reg_dst_spat_offt, axis_size_ * dst_next_vreg_stride_);
    }

    // The function provides complete softmax algorithm over axis_size_. Stages:
    // * Fill vmax registers with -FLT_MAX, once before the loop.
    // * A loop to collect vmax values.
    // * Reset vsum registers with zeros, once before the loop.
    // * A loop to collect sum and apply exponent.
    // * Apply division to use multiplication when storing results.
    // * A loop to store output after division.
    // Why loops are needed, see `axis_size_loop_unroll` description.
    void axis_full_cycle(int unroll_inner, bool tail) {
        const auto vmax_body
                = [&](int unroll_axis, int unroll_inner, bool tail) {
            for_(dim_t a = 0; a < unroll_axis; a++)
            for (int i = 0; i < unroll_inner; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                Vmm vmax = get_vmax(vreg_tmp_src, unroll_inner);
                Vmm vtmp = get_vsum(vreg_tmp_src, unroll_inner);
                // do maxps directly from memory on f32 avx2 for performance
                // purpose.
                if (!tail && is_superset(isa, avx2)
                        && !is_superset(isa, avx512_core)
                        && src_d_.data_type() == f32) {
                    uni_vmaxps(vmax, vmax, src_ptr(get_src_stride(a, i)));
                } else {
                    io_[src_d_.data_type()]->load(
                            src_ptr(get_src_stride(a, i)), vreg_tmp_src, tail);
                    uni_vmaxps_maybe_tail(vmax, vreg_tmp_src, vtmp, tail);
                }
            }
        };

        const auto vsum_body
                = [&](int unroll_axis, int unroll_inner, bool tail) {
            for_(dim_t a = 0; a < unroll_axis; a++)
            for (int i = 0; i < unroll_inner; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                Vmm vmax = get_vmax(vreg_tmp_src, unroll_inner);
                Vmm vsum = get_vsum(vreg_tmp_src, unroll_inner);
                // AVX2 and below would scratch a register, thus, vmax register
                // can't be used there. Luckily, tail case only has a single
                // unroll value, thus, can just use the next reg after `vsum`.
                Vmm vtmp = tail && !is_superset(isa, avx512_core)
                        ? Vmm(vsum.getIdx() + 1)
                        : Vmm();

                io_[src_d_.data_type()]->load(
                        src_ptr(get_src_stride(a, i)), vreg_tmp_src, tail);
                uni_vsubps(vreg_tmp_src, vreg_tmp_src, vmax);
                if (is_logsoftmax_) { // store before applying exp
                    if (need_scratchpad_)
                        store(interim_ptr(get_interim_stride(a)), vreg_tmp_src,
                                f32, tail);
                    else
                        store(dst_ptr(get_dst_stride(a, i)), vreg_tmp_src,
                                dst_d_.data_type(), tail);
                }
                exp_injector_->compute_vector(vreg_tmp_src.getIdx());
                uni_vaddps_maybe_tail(vsum, vreg_tmp_src, vtmp, tail);
                if (is_softmax_) { // store after applying exp
                    if (need_scratchpad_)
                        store(interim_ptr(get_interim_stride(a)), vreg_tmp_src,
                                f32, tail);
                    else
                        store(dst_ptr(get_dst_stride(a, i)), vreg_tmp_src,
                                dst_d_.data_type(), tail);
                }
            }
        };

        const auto store_body
                = [&](int unroll_axis, int unroll_inner, bool tail) {
            for_(dim_t a = 0; a < unroll_axis; a++)
            for (int i = 0; i < unroll_inner; i++) {
                Vmm vreg_tmp_src = Vmm(i + 1);
                Vmm vsum = get_vsum(vreg_tmp_src, unroll_inner);
                if (need_scratchpad_)
                    io_[f32]->load(interim_ptr(get_interim_stride(a)),
                            vreg_tmp_src, tail);
                else
                    io_[dst_d_.data_type()]->load(
                            dst_ptr(get_dst_stride(a, i)), vreg_tmp_src, tail);

                if (is_softmax_) uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsum);
                if (is_logsoftmax_)
                    uni_vsubps(vreg_tmp_src, vreg_tmp_src, vsum);

                if (with_src_scales_) {
                    uni_vmulps(vreg_tmp_src, vreg_tmp_src, vsrc_scale);
                }
                if (with_postops_) {
                    binary_injector::rhs_arg_dynamic_params_t rhs_arg_params;
                    if (with_binary_) {
                        rhs_arg_params.vmm_idx_to_out_addr.emplace(
                                vreg_tmp_src.getIdx(), dst_ptr());
                        rhs_arg_params.vmm_idx_to_out_elem_off_val.emplace(
                                vreg_tmp_src.getIdx(), get_dst_stride(a, i));
                        if (tail)
                            rhs_arg_params.vmm_tail_idx_.emplace(
                                    vreg_tmp_src.getIdx());
                    }
                    postops_injector_->compute_vector(
                            vreg_tmp_src.getIdx(), rhs_arg_params);
                }
                if (with_dst_scales_) {
                    uni_vmulps(vreg_tmp_src, vreg_tmp_src, vdst_scale);
                }
                store(dst_ptr(get_dst_stride(a, i)), vreg_tmp_src,
                        dst_d_.data_type(), tail);
            }
        };

        // flush to -FLT_MAX before accumulation
        for (int i = 0; i < unroll_inner; i++) {
            Vmm vreg_tmp_src = Vmm(i + 1);
            Vmm vmax = get_vmax(vreg_tmp_src, unroll_inner);
            uni_vmovups(vmax, vneg_flt_max);
        }

        axis_size_loop_unroll(vmax_body, unroll_inner, tail);

        // flush to zero before accumulation
        for (int i = 0; i < unroll_inner; i++) {
            Vmm vreg_tmp_src = Vmm(i + 1);
            Vmm vsum = get_vsum(vreg_tmp_src, unroll_inner);
            uni_vpxor(vsum, vsum, vsum);
        }

        axis_size_loop_unroll(vsum_body, unroll_inner, tail);

        for (int i = 0; i < unroll_inner; i++) {
            Vmm vreg_tmp_src = Vmm(i + 1);
            Vmm vtmp = get_vmax(vreg_tmp_src, unroll_inner);
            Vmm vsum = get_vsum(vreg_tmp_src, unroll_inner);

            if (pd_->alg_kind() == alg_kind::softmax_accurate_inf_as_zero) {
                Xbyak::Label skip_div;
                // `vptest` sets the `ZF` flag if all bits in the result are 0
                // of the bitwise AND of source operands.
                // Note: using Vmm(1) is an ugly workaround EVEX versus VEX
                // encoding as `vsum` uses index `30` on avx512_core. `Vmm(1)`
                // is a tmp vreg used to read data just above. Should be safe.
                uni_vmovups(Vmm(1), vsum);
                uni_vptest(Xmm(1), Xmm(1));
                jz(skip_div, T_NEAR); // Check if ZF is set.
                uni_vdivps(vsum, vone, vsum, vtmp);
                L(skip_div);
            } else if (is_softmax_) {
                uni_vdivps(vsum, vone, vsum, vtmp);
            } else if (is_logsoftmax_) {
                log_injector_->compute_vector(vsum.getIdx());
            }
        }

        axis_size_loop_unroll(store_body, unroll_inner, tail);

        const dim_t src_dt_size = src_d_.data_type_size();
        const dim_t dst_dt_size = dst_d_.data_type_size();
        add(reg_src_spat_offt, unroll_inner * simd_w_ * src_dt_size);
        add(reg_dst_spat_offt, unroll_inner * simd_w_ * dst_dt_size);
    }

    // The function provides unrolling over inner size. A single compute block
    // is simd_w by axis_size_, where axis is strided in memory. The purpose of
    // inner size unrolling is to improve reading/writing pattern and increase
    // memory bandwidth.
    void inner_size_loop_unroll() {
        Label unroll_loop, unroll_tail_loop, tail_loop, loop_end;

        // reverse_spat_offt to dispatch between labels
        mov(reg_reverse_n_elems, ptr[reg_param + PARAM_OFF(process_n_elems)]);
        xor_(reg_src_spat_offt, reg_src_spat_offt);
        xor_(reg_interim_spat_offt, reg_interim_spat_offt);
        xor_(reg_dst_spat_offt, reg_dst_spat_offt);
        L(unroll_loop);
        {
            if (n_loops_) {
                cmp(reg_reverse_n_elems, unroll_inner_size_ * simd_w_);
                jl(unroll_tail_loop, T_NEAR);

                axis_full_cycle(unroll_inner_size_, false);

                sub(reg_reverse_n_elems, unroll_inner_size_ * simd_w_);
                jmp(unroll_loop);
            }
        }

        L(unroll_tail_loop);
        {
            if (loop_tail_) {
                cmp(reg_reverse_n_elems, loop_tail_ * simd_w_);
                jl(tail_loop, T_NEAR);

                axis_full_cycle(loop_tail_, false);

                sub(reg_reverse_n_elems, loop_tail_ * simd_w_);
                // No jump, unroll_tail run once.
            }
        }

        L(tail_loop);
        {
            if (axis_simd_tail_) {
                cmp(reg_reverse_n_elems, 1);
                jl(loop_end, T_NEAR);

                axis_full_cycle(1, true);
            }
        }

        L(loop_end);
    }

    Vmm get_vmax(const Vmm &vmm, int unroll_inner) {
        return Vmm(vmm.getIdx() + unroll_inner);
    }

    Vmm get_vsum(const Vmm &vmm, int unroll_inner) {
        return Vmm(vmm.getIdx() + 2 * unroll_inner);
    }

    size_t get_src_stride(dim_t axis_idx, int unroll_inner_i) {
        return src_next_vreg_stride_ * axis_idx
                + unroll_inner_i * simd_w_ * src_d_.data_type_size();
    }

    size_t get_dst_stride(dim_t axis_idx, int unroll_inner_i) {
        return dst_next_vreg_stride_ * axis_idx
                + unroll_inner_i * simd_w_ * dst_d_.data_type_size();
    }

    size_t get_interim_stride(dim_t axis_idx) {
        return interim_next_vreg_stride_ * axis_idx;
    }

    // TODO: add interleaved support for xf16 on avx2_vnni_2.
    void forward() { inner_size_loop_unroll(); }

    void generate() override {
        if (pd_->is_fwd() || is_logsoftmax_)
            exp_injector_.reset(new jit_uni_eltwise_injector_t<Vmm>(this,
                    alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, data_type::f32,
                    true, reg_exp_injector_table, injector_mask));
        if (pd_->is_fwd() && is_logsoftmax_) {
            log_injector_.reset(new jit_uni_eltwise_injector_t<Vmm>(this,
                    alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, data_type::f32,
                    true, reg_log_injector_table, injector_mask));
        }
        if (with_postops_) {
            static constexpr bool preserve_gpr = true;
            static constexpr bool preserve_vmm = true;
            static constexpr bool use_exact_tail_scalar_bcast = false;
            static constexpr std::size_t tmp_vmm_injector = 0u;

            const binary_injector::rhs_arg_static_params_t rhs_sp {
                    tmp_vmm_injector, this->r14, this->r15, this->r13,
                    preserve_gpr, preserve_vmm,
                    PARAM_OFF(post_ops_binary_rhs_arg_vec), PARAM_OFF(dst_orig),
                    dst_d_, axis_simd_tail_, tail_opmask,
                    use_exact_tail_scalar_bcast};

            const binary_injector::static_params_t bsp {
                    reg_param, get_supported_bcast_strategies(), rhs_sp};

            postops_injector_ = utils::make_unique<
                    injector::jit_uni_postops_injector_t<Vmm>>(this,
                    pd_->attr()->post_ops_, bsp, /* inject_sum = */ false);
        }
#undef PARAM_OFF

        compute_predefined_variables();
        preamble();
        io_.init_bf16();
        // Initialize saturation vector register
        if (is_superset(isa, avx2)) {
            io_.init_saturate_f32({dst_d_.data_type()});
        }
        if (exp_injector_) exp_injector_->load_table_addr();
        if (log_injector_) log_injector_->load_table_addr();
        if (axis_simd_tail_) io_.prepare_tail_mask();
        load_common_params();
        if (pd_->is_fwd()) forward();
        postamble();
        if (exp_injector_) exp_injector_->prepare_table();
        if (log_injector_) log_injector_->prepare_table();
        if (with_eltwise_ && postops_injector_)
            postops_injector_->prepare_table(/* generate = */ true);
        io_.prepare_table_fp8();
    }

    jit_softmax_strided_kernel_t(const softmax_pd_t *pd)
        : jit_softmax_kernel_base_t(pd)
        , jit_generator_t(jit_name(), isa)
        , src_d_(pd_->invariant_src_md())
        , dst_d_(pd_->dst_md())
        // Note: must be aligned with pd_t::init()->init_scratchpad();
        , need_scratchpad_(pd_->is_fwd() && dst_d_.data_type() != f32
                  && /* !relaxed_acc */ !(
                          src_d_.data_type() == dst_d_.data_type()
                          && !types::is_integral_dt(dst_d_.data_type())
                          && utils::one_of(pd_->attr()->acc_mode_,
                                  accumulation_mode::relaxed,
                                  accumulation_mode::any)))
        , axis_size_(pd_->axis_size())
        // `axis_stride_`, `axis_simd_full_` and `axis_simd_tail_` are only
        // different pieces from the dense version.
        , axis_stride_(pd_->axis_stride())
        , axis_simd_full_(axis_stride_ / simd_w_)
        , axis_simd_tail_(static_cast<int>(axis_stride_ % simd_w_)) {

        // Scratchpad size is limited to a single simd_w, thus, no unrolling
        // for such cases.
        if (need_scratchpad_)
            unroll_inner_size_ = 1;
        else if (mayiuse(avx2) && !mayiuse(avx512_core))
            unroll_inner_size_ = 2;

        const auto &post_ops = pd_->attr()->post_ops_;
        with_postops_ = post_ops.len() != 0;
        with_binary_ = post_ops.find(primitive_kind::binary) != -1;
        with_eltwise_ = post_ops.find(primitive_kind::eltwise) != -1;

        const auto &attr_scales = pd_->attr()->scales_;
        with_src_scales_ = is_superset(isa, avx2)
                && !attr_scales.has_default_values(DNNL_ARG_SRC);
        with_dst_scales_ = is_superset(isa, avx2)
                && !attr_scales.has_default_values(DNNL_ARG_DST);

        io::io_conf_t io_conf;
        io::io_tail_conf_t io_tail_conf(simd_w_, axis_simd_tail_,
                tail_opmask_idx_, tail_vmask.getIdx(), reg_tmp);
        io::io_emu_bf16_conf_t io_bf16_conf(bf16_emu_zmm_1_idx_,
                bf16_emu_zmm_2_idx_, bf16_emu_zmm_3_idx_, reg_tmp,
                bf16_emu_zmm_4_idx_);
        io::io_saturation_conf_t io_saturation_conf(
                vzero.getIdx(), vsaturation_ubound.getIdx(), reg_tmp);
        io::io_emu_fp8_conf_t io_fp8_conf(fp8_emu_zmm_1_idx_,
                fp8_emu_zmm_2_idx_, fp8_emu_zmm_3_idx_, fp8_emu_zmm_4_idx_,
                fp8_emu_zmm_5_idx_, fp8_emu_kmask_idx_, reg_tmp);
        io_ = io::jit_io_multi_dt_helper_t<Vmm>(this, isa,
                {src_d_.data_type(), dst_d_.data_type(), f32 /* stats */},
                io_conf, io_tail_conf, io_bf16_conf,
                {{dst_d_.data_type(), io_saturation_conf}}, utils::nullopt,
                io_fp8_conf);
    }
};

// Kernels are split in two implementations since strided version operates over
// inner dimension processing `simd_w` axes at a time. This implies different
// registers usage. To avoid collision and simplify the support, having a second
// class is easier though certain pieces are same.
jit_softmax_kernel_base_t *jit_softmax_kernel_base_t::create(
        const softmax_pd_t *pd, const cpu_isa_t isa,
        bool axis_is_plain_and_strided) {

#define HANDLE_ISA(isa_) \
    if ((isa_) == isa) { \
        if (axis_is_plain_and_strided) \
            return new jit_softmax_strided_kernel_t<isa_>(pd); \
        else \
            return new jit_softmax_dense_kernel_t<isa_>(pd); \
    }
    REG_AVX512_ISA(HANDLE_ISA(avx512_core_fp16));
    REG_AVX512_ISA(HANDLE_ISA(avx512_core_bf16));
    REG_AVX512_ISA(HANDLE_ISA(avx512_core));
    REG_AVX2_ISA(HANDLE_ISA(avx2_vnni_2));
    REG_AVX2_ISA(HANDLE_ISA(avx2));
#undef HANDLE_ISA
    assert(!"kernel is empty.");
    return nullptr;
}

std::vector<cpu_isa_t> get_supported_isa(bool is_fwd) {
    if (is_fwd)
        return {avx512_core_fp16, avx512_core_bf16, avx512_core, avx2_vnni_2,
                avx2};
    else
        return {avx512_core_fp16, avx512_core_bf16, avx512_core};
}

// `per_oc_spatial` cover strided axis case since algorithm requires a single
// element being broadcasted.
bcast_set_t get_supported_bcast_strategies() {
    return {broadcasting_strategy_t::scalar, broadcasting_strategy_t::per_oc,
            broadcasting_strategy_t::per_oc_spatial,
            broadcasting_strategy_t::no_broadcast};
}
} // namespace softmax_impl

jit_uni_softmax_fwd_t::jit_uni_softmax_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

status_t jit_uni_softmax_fwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(ker_,
            softmax_impl::jit_softmax_kernel_base_t::create(
                    pd(), pd()->isa_, pd()->axis_is_plain_and_strided_)));
    if (ker_) CHECK(ker_->create_kernel());
    return status::success;
}

status_t jit_uni_softmax_fwd_t::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);

    const void *src_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC);
    const void *dst_scales
            = CTX_IN_MEM(const void *, DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST);

    const auto &scratchpad = ctx.get_scratchpad_grantor();
    auto scratchpad_ptr = scratchpad.template get<char>(
            memory_tracking::names::key_softmax_interim_store);

    const auto post_ops_binary_rhs_arg_vec
            = binary_injector::prepare_binary_args(
                    pd()->attr()->post_ops_, ctx);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto src_data_type_size = src_d.data_type_size();
    const auto dst_data_type_size = dst_d.data_type_size();
    const auto &bd = src_d.blocking_desc();

    const auto axis_stride = pd()->axis_stride();
    const auto axis_is_blocked = axis_stride != 1 && bd.inner_nblks;

    dim_t inner_stride = 1;
    dim_t inner_size = 1;
    if (axis_is_blocked) {
        // `inner_size` includes spatial for aBxNb blocked formats.
        // Impl parallelize over `inner_size` for blocked layout.
        inner_stride = bd.inner_blks[bd.inner_nblks - 1];
        inner_size = axis_stride / inner_stride;
    }

    dim_t outer_stride = pd()->axis_size(true) * inner_size;
    dim_t outer_size = src_d.nelems(true) / outer_stride;
    // `process_n_elems` covers:
    // * Plain dense axis with `axis_size` as a basic block.
    // * Blocked axis with `axis_size` times `inner_size` to provide proper next
    //   reg stride. Inner size will be parallelized. Non-padded axis size is
    //   needed to properly handle zero padding.
    dim_t process_n_elems = pd()->axis_size() * inner_size;
    static constexpr int unroll_block_size = 64; // 4 unroll x 16 simd_w
    dim_t n_unrolled_blocks = 0;
    dim_t unroll_block_size_tail = axis_stride % unroll_block_size;
    if (pd()->axis_is_plain_and_strided_) {
        outer_stride = pd()->axis_size(true) * axis_stride;
        outer_size = src_d.nelems(true) / outer_stride;
        if (outer_size == 1) {
            // Divide inner size by blocks and share a piece between threads.
            n_unrolled_blocks = utils::div_up(axis_stride, unroll_block_size);
            inner_size = n_unrolled_blocks;
            inner_stride = unroll_block_size;
        } else {
            // Just give a full inner size, suboptimal but simple. The major
            // challenge is tailed cases when it involves complex computations
            // of src/dst offsets and proper number of elements to process.
            process_n_elems = axis_stride;
        }
    }

    const int nthr = pd()->nthr_;
    const char *dst_orig_ptr = dst;

    VDEBUGINFO(1, primitive, softmax,
            "%s,src=%p dst=%p outer_size=%" PRId64 " outer_stride=%" PRId64
            " inner_size=%" PRId64 " inner_stride=%" PRId64
            " axis_stride=%" PRId64,
            pd()->impl_name(), src, dst, outer_size, outer_stride, inner_size,
            inner_stride, axis_stride);

    parallel_nd_ext(nthr, outer_size, inner_size,
            [= COMPAT_THIS_CAPTURE](int ithr, int, dim_t ou, dim_t in) {
        dim_t offset = (ou * outer_stride + in * inner_stride);
        const char *src_ptr = src + offset * src_data_type_size;
        char *dst_ptr = dst + offset * dst_data_type_size;
        char *interim_ptr = scratchpad_ptr
                ? scratchpad_ptr + ithr * pd()->scratch_size_per_thr_
                : nullptr;
        float *dst_scales_inv_ptr = nullptr;
        if (!pd()->attr()->scales_.has_default_values(DNNL_ARG_DST)) {
            const float *dst_scales_ptr
                    = static_cast<const float *>(dst_scales);
            dst_scales_inv_ptr
                    = scratchpad.template get<float>(
                              memory_tracking::names::key_softmax_dst_scales)
                    + ithr;
            dst_scales_inv_ptr[0] = 1.f / dst_scales_ptr[0];
        }

        softmax_impl::jit_softmax_kernel_base_t::call_params_t p;
        if (pd()->axis_is_plain_and_strided_ && outer_size == 1) {
            // Special case when inner size is split between threads.
            assert(n_unrolled_blocks > 0);
            p.process_n_elems = in == inner_size - 1 && unroll_block_size_tail
                    ? unroll_block_size_tail
                    : unroll_block_size;
        } else {
            p.process_n_elems = process_n_elems;
        }
        p.src = src_ptr;
        p.dst = dst_ptr;
        p.interim = interim_ptr;
        p.src_scales = src_scales;
        p.dst_scales = dst_scales_inv_ptr;
        // post-ops
        p.dst_orig = dst_orig_ptr;
        p.post_ops_binary_rhs_arg_vec = post_ops_binary_rhs_arg_vec.data();
        (*ker_)(&p);
    });

    return status::success;
}

jit_uni_softmax_bwd_t::jit_uni_softmax_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

status_t jit_uni_softmax_bwd_t::init(engine_t *engine) {
    CHECK(safe_ptr_assign(ker_,
            softmax_impl::jit_softmax_kernel_base_t::create(
                    pd(), pd()->isa_, false)));
    if (ker_) CHECK(ker_->create_kernel());
    return status::success;
}

status_t jit_uni_softmax_bwd_t::execute(const exec_ctx_t &ctx) const {
    auto dst = CTX_IN_MEM(const char *, DNNL_ARG_DST);
    auto diff_dst = CTX_IN_MEM(const char *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(char *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const auto dst_data_type_size = dst_d.data_type_size();
    const auto diff_dst_data_type_size = diff_dst_d.data_type_size();
    const auto diff_src_data_type_size = diff_src_d.data_type_size();
    const auto &bd = dst_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto process_n_elems = pd()->axis_size() * inner_size;
    const auto outer_stride = pd()->axis_size(true) * inner_size;
    const auto outer_size = dst_d.nelems(true) / outer_stride;

    parallel_nd(outer_size, inner_size,
            [= COMPAT_THIS_CAPTURE](dim_t ou, dim_t in) {
        dim_t offset = (ou * outer_stride + in * inner_stride);
        char *diff_src_ptr = diff_src + offset * diff_src_data_type_size;
        const char *dst_ptr = dst + offset * dst_data_type_size;
        const char *diff_dst_ptr = diff_dst + offset * diff_dst_data_type_size;
        softmax_impl::jit_softmax_kernel_base_t::call_params_t p;
        p.process_n_elems = process_n_elems;
        p.src = diff_src_ptr;
        p.dst = dst_ptr;
        p.diff_dst = diff_dst_ptr;
        (*ker_)(&p);
    });

    return status::success;
}

} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
