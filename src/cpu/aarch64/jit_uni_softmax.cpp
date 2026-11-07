/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2020-2024 FUJITSU LIMITED
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

#include <cassert>
#include <memory>

#include "common/c_types_map.hpp"
#include "common/compiler_workarounds.hpp"
#include "common/dnnl_thread.hpp"
#include "common/memory_tracking.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_uni_softmax.hpp"
#include "cpu/cpu_primitive.hpp"

#include "xbyak_aarch64/xbyak_aarch64_util.h"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

struct jit_softmax_base_t : public jit_generator_t {
    struct call_params_t {
        // keep all sizes at 8 bytes -- jit code expects this
        const void *src, *dst, *diff_dst; // src dubs as diff_src
        const void *interim; // scratch memory for intermediate storage
        const void *src_scales; // src_scales defined for all data type cases
        const void *dst_scales; // dst_scales defined for all data type cases
        size_t process_n_elems;
    };
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_softmax_t)

    const softmax_pd_t *pd_;
    const memory_desc_wrapper src_d_, dst_d_, diff_dst_d_;

    void operator()(const call_params_t *p) { jit_generator_t::operator()(p); }

    XReg reg_param = abi_param1;

    XReg reg_exp_injector_table = x1;
    XReg reg_log_injector_table = x3;
    XReg reg_src = x8;
    XReg reg_diff_src = reg_src;
    XReg reg_dst = x9;
    XReg reg_diff_dst = x14;
    XReg reg_src_spat_offt = x10;
    XReg reg_process_n_elems = x11;
    XReg reg_reverse_n_elems = x12;
    XReg reg_tmp = x13;
    XReg reg_dst_spat_offt = x16;
    XReg reg_diff_dst_spat_offt = reg_log_injector_table;
    XReg reg_interim = reg_diff_dst;
    XReg reg_interim_spat_offt = abi_not_param1;
    XReg reg_src_scales = x6;
    XReg reg_dst_scales = x7;

    const PReg p_shuff0 = p11;
    const PReg p_shuff1 = p5;
    const PReg injector_mask = p1;
    const PReg injector_tmp = p6;
    const PReg tail_opmask = p2;

    static constexpr int data_vreg_start_idx = 8;
    static const int vtmp_idx = 25;
    static const int vneg_flt_max_idx = 26;
    static const int vone_idx = 27;
    static const int vsum_idx = 28;
    static const int vmax_idx = 29;
    static const int vsbr_idx = vsum_idx; // must be not equal to vmax_idx
    static const int vzero_idx = 30;
    static const int vsaturation_ubound_idx = vneg_flt_max_idx;
    static const int v_tmp0_idx = 31;

    bool is_softmax_ = pd_->is_softmax();
    bool is_logsoftmax_ = pd_->is_logsoftmax();
    bool is_inf_as_zero_
            = pd_->alg_kind() == alg_kind::softmax_accurate_inf_as_zero;
    bool need_src_scale_
            = !pd_->attr()->scales_.has_default_values(DNNL_ARG_SRC);
    bool need_dst_scale_
            = !pd_->attr()->scales_.has_default_values(DNNL_ARG_DST);
    bool axis_is_blocked_;
    bool need_scratchpad_;

    size_t vlen_ = 0;
    size_t simd_w_ = 0;
    size_t unroll_regs_ = pd_->is_fwd() ? 16 : 4;

    size_t axis_simd_full_;
    size_t axis_simd_tail_;
    size_t n_loops_;
    size_t loop_tail_;
    size_t process_n_elems_;
    size_t src_axis_stride_;
    size_t interim_axis_stride_;
    size_t dst_axis_stride_;
    size_t diff_dst_axis_stride_;

    enum class op_t : unsigned { max, sum };

    void load_common_params();
    void compute_predefined_variables();
    size_t compute_process_n_elems(const memory_desc_wrapper &mdw) const;
    size_t compute_axis_stride(const memory_desc_wrapper &mdw) const;
    XReg xreg_addr(const XReg &base, const XReg &off = XReg(DUMMY_IDX),
            const int disp = 0);
    XReg src_ptr(size_t offt = 0);
    XReg dst_ptr(size_t offt = 0);
    XReg interim_ptr(size_t offt = 0);

    template <typename body_t>
    void axis_loop(body_t body);
    void forward();

    virtual void accumulate_vmax() = 0;
    virtual void accumulate_vsum() = 0;
    virtual void compute_dst() = 0;

    jit_softmax_base_t(const softmax_pd_t *pd, int vlen);
};

void jit_softmax_base_t::compute_predefined_variables() {
    axis_simd_full_ = pd_->axis_size() / simd_w_;
    axis_simd_tail_ = pd_->axis_size() % simd_w_;
    n_loops_ = axis_simd_full_ / unroll_regs_;
    loop_tail_ = axis_simd_full_ - n_loops_ * unroll_regs_;
    process_n_elems_ = compute_process_n_elems(dst_d_);
    src_axis_stride_ = compute_axis_stride(src_d_);
    interim_axis_stride_ = simd_w_ * sizeof(float);
    dst_axis_stride_ = compute_axis_stride(dst_d_);
    if (!pd_->is_fwd())
        diff_dst_axis_stride_ = compute_axis_stride(diff_dst_d_);
    axis_is_blocked_ = pd_->axis_size(true) != pd_->axis_size();
}

size_t jit_softmax_base_t::compute_process_n_elems(
        const memory_desc_wrapper &mdw) const {
    const auto &bd = mdw.blocking_desc();
    if (bd.inner_nblks) return bd.strides[pd_->axis()];
    return simd_w_;
}

size_t jit_softmax_base_t::compute_axis_stride(
        const memory_desc_wrapper &mdw) const {
    return compute_process_n_elems(mdw) * mdw.data_type_size();
}

void jit_softmax_base_t::load_common_params() {
    mov(W_TMP_0, float2int(-FLT_MAX));
    if (vlen_ > util::SVE_128) {
        fmov(ZReg(vone_idx).s, 1.0);
        dup(ZReg(vneg_flt_max_idx).s, W_TMP_0);
    } else {
        fmov(VReg(vone_idx).s, 1.0);
        dup(VReg(vneg_flt_max_idx).s, W_TMP_0);
    }

#define PARAM_OFF(x) offsetof(call_params_t, x)
#define PARAM_LOAD(reg, var) \
    add_imm(X_DEFAULT_ADDR, reg_param, PARAM_OFF(var), X_TMP_0); \
    ldr(reg, ptr(X_DEFAULT_ADDR));

    /* Address offset must be less than 256. */
    PARAM_LOAD(reg_process_n_elems, process_n_elems);
    PARAM_LOAD(reg_dst, dst);
    if (pd_->is_fwd()) {
        PARAM_LOAD(reg_src, src);
    } else {
        PARAM_LOAD(reg_diff_src, src);
        PARAM_LOAD(reg_diff_dst, diff_dst);
    }
    if (need_scratchpad_) { PARAM_LOAD(reg_interim, interim); }
    if (need_src_scale_) { PARAM_LOAD(reg_src_scales, src_scales); }
    if (need_dst_scale_) { PARAM_LOAD(reg_dst_scales, dst_scales); }
#undef PARAM_OFF
#undef PARAM_LOAD
}

XReg jit_softmax_base_t::xreg_addr(
        const XReg &base, const XReg &off, const int disp) {
    XReg x_addr = base;
    uint32_t offIdx = off.getIdx();

    if (offIdx <= SP_IDX) {
        add(X_DEFAULT_ADDR, base, off);
        x_addr = X_DEFAULT_ADDR;
    }
    if (disp) {
        add_imm(X_DEFAULT_ADDR, x_addr, disp, X_TMP_0);
        x_addr = X_DEFAULT_ADDR;
    }

    return x_addr;
}

XReg jit_softmax_base_t::src_ptr(size_t offt) {
    return xreg_addr(reg_src, reg_src_spat_offt, offt);
}

XReg jit_softmax_base_t::dst_ptr(size_t offt) {
    return xreg_addr(reg_dst, reg_dst_spat_offt, offt);
}

XReg jit_softmax_base_t::interim_ptr(size_t offt) {
    return xreg_addr(reg_interim, reg_interim_spat_offt, offt);
}

template <typename body_t>
void jit_softmax_base_t::axis_loop(body_t body) {
    Label main_loop, tail_loop, tail_axis;

    // reverse_spat_offt to dispatch between labels
    mov(reg_reverse_n_elems, reg_process_n_elems);
    mov_imm(reg_src_spat_offt, 0); // src/diff_src addr
    mov_imm(reg_dst_spat_offt, 0); // dst addr
    if (need_scratchpad_) mov_imm(reg_interim_spat_offt, 0); // scratch addr
    if (!pd_->is_fwd()) mov_imm(reg_diff_dst_spat_offt, 0); // d_dst addr
    L(main_loop);
    {
        if (n_loops_) {
            cmp_imm(reg_reverse_n_elems, unroll_regs_ * process_n_elems_,
                    X_TMP_0);
            b(LT, tail_loop);

            body(unroll_regs_, false);
            sub_imm(reg_reverse_n_elems, reg_reverse_n_elems,
                    unroll_regs_ * process_n_elems_, X_TMP_0);
            add_imm(reg_src_spat_offt, reg_src_spat_offt,
                    unroll_regs_ * src_axis_stride_, X_TMP_0);
            add_imm(reg_dst_spat_offt, reg_dst_spat_offt,
                    unroll_regs_ * dst_axis_stride_, X_TMP_0);
            if (need_scratchpad_)
                add_imm(reg_interim_spat_offt, reg_interim_spat_offt,
                        unroll_regs_ * interim_axis_stride_, X_TMP_0);
            if (!pd_->is_fwd())
                add_imm(reg_diff_dst_spat_offt, reg_diff_dst_spat_offt,
                        unroll_regs_ * diff_dst_axis_stride_, X_TMP_0);
            b(main_loop);
        }
    }

    L(tail_loop);
    {
        if (loop_tail_) {
            body(loop_tail_, false);
            add_imm(reg_src_spat_offt, reg_src_spat_offt,
                    loop_tail_ * src_axis_stride_, X_TMP_0);
            add_imm(reg_dst_spat_offt, reg_dst_spat_offt,
                    loop_tail_ * dst_axis_stride_, X_TMP_0);
            if (need_scratchpad_)
                add_imm(reg_interim_spat_offt, reg_interim_spat_offt,
                        loop_tail_ * interim_axis_stride_, X_TMP_0);
            if (!pd_->is_fwd())
                add_imm(reg_diff_dst_spat_offt, reg_diff_dst_spat_offt,
                        loop_tail_ * diff_dst_axis_stride_, X_TMP_0);
        }
    }

    L(tail_axis);
    {
        if (axis_simd_tail_) { body(1, true); }
    }
}

void jit_softmax_base_t::forward() {
    accumulate_vmax();
    accumulate_vsum();
    compute_dst();
}

jit_softmax_base_t::jit_softmax_base_t(const softmax_pd_t *pd, int vlen)
    : jit_generator_t(nullptr, MAX_CODE_SIZE, true)
    , pd_(pd)
    , src_d_(pd_->is_fwd() ? pd_->src_md() : pd_->diff_src_md())
    , dst_d_(pd_->dst_md())
    , diff_dst_d_(pd_->diff_dst_md())
    , need_scratchpad_(pd_->is_fwd()
              && utils::one_of(dst_d_.data_type(), data_type::u8, data_type::s8,
                      data_type::bf16, data_type::f16))
    , vlen_(vlen)
    , simd_w_(vlen / sizeof(float)) {}

// SVE JIT softmax generator
struct jit_softmax_sve_t : public jit_softmax_base_t {
    std::unique_ptr<jit_uni_eltwise_injector_t<sve>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_t<sve>> log_injector_;

    const ZReg vtmp = ZReg(vtmp_idx);
    const ZReg vneg_flt_max = ZReg(vneg_flt_max_idx);
    const ZReg vone = ZReg(vone_idx);
    const ZReg vsum = ZReg(vsum_idx);
    const ZReg vmax = ZReg(vmax_idx);
    const ZReg vsbr = ZReg(vsbr_idx);
    const ZReg vzero = ZReg(vzero_idx);
    const ZReg vsaturation_ubound = ZReg(vsaturation_ubound_idx);
    const ZReg v_tmp0 = ZReg(v_tmp0_idx);

    void store(const XReg &addr, const ZReg &vmm, data_type_t dt, bool tail);
    void load(const ZReg &vmm, const XReg &addr, data_type_t dt, bool tail);
    void get_horizontal_op(const ZReg &v, op_t op);
    void accumulate_vmax() override;
    void accumulate_vsum() override;
    void compute_dst() override;
    void generate() override;

    // SVE-specific functions
    void uni_fmax(const ZReg &dst, const ZReg &src, const ZReg &src2,
            const PReg &mask = PReg(DUMMY_IDX));
    XReg diff_src_ptr(size_t offt = 0);
    XReg diff_dst_ptr(size_t offt = 0);
    void prepare_tail_mask();
    void accumulate_vsbr();
    void compute_diff_src();
    void backward();
    void prepare_mask();
    void restore_mask();

    jit_softmax_sve_t(const softmax_pd_t *pd, int vlen);
};

void jit_softmax_sve_t::uni_fmax(
        const ZReg &dst, const ZReg &src, const ZReg &src2, const PReg &mask) {
    const uint32_t idxDst = dst.getIdx();
    const uint32_t idxSrc = src.getIdx();
    const uint32_t idxSrc2 = src2.getIdx();
    uint32_t pattern = 0;
    PReg mask_reg(DUMMY_IDX);

    pattern += (idxDst == idxSrc) ? (1 << 2) : 0;
    pattern += (idxDst == idxSrc2) ? (1 << 1) : 0;
    pattern += (idxSrc == idxSrc2) ? 1 : 0;

    if (mask.getIdx() == DUMMY_IDX)
        mask_reg = P_ALL_ONE;
    else
        mask_reg = mask;

    switch (pattern) {
        case 0x4: /* dst = src && dst != src2 && src != src2
                    This is the most popular case. */
            fmax(dst.s, mask_reg / T_m, src2.s);
            break;
        default: assert(!"Unreachable!"); break;
    }
}

XReg jit_softmax_sve_t::diff_src_ptr(size_t offt) {
    return xreg_addr(reg_diff_src, reg_src_spat_offt, offt);
}

XReg jit_softmax_sve_t::diff_dst_ptr(size_t offt) {
    return xreg_addr(reg_diff_dst, reg_diff_dst_spat_offt, offt);
}

void jit_softmax_sve_t::store(
        const XReg &addr, const ZReg &vmm, data_type_t dt, bool tail) {
    PReg opmask = P_ALL_ONE;
    bool tail_mask_valid = false;
    auto effective_addr = addr;
    ZReg src_vmm = vmm;

    if (tail) {
        if (utils::one_of(
                    dt, data_type::f32, data_type::bf16, data_type::f16)) {
            if (axis_is_blocked_) {
                src_vmm = vzero;
                eor(vzero.d, vzero.d, vzero.d);
                mov(src_vmm.s, tail_opmask / T_m, vmm.s);
                effective_addr = addr;
            } else {
                effective_addr = addr;
                tail_mask_valid = true;
            }
        } else { // int8 store instructions assume mask on register
            tail_mask_valid = true;
        }
    }

    if (tail_mask_valid) opmask = tail_opmask;

    switch (dt) {
        case data_type::f32:
            st1w(src_vmm.s, opmask, ptr(effective_addr));
            break;
        case data_type::bf16:
            bfcvt(src_vmm.h, P_ALL_ONE / T_m, src_vmm.s);
            st1h(src_vmm.s, opmask, ptr(effective_addr));
            break;
        case data_type::f16:
            fcvt(src_vmm.h, P_ALL_ONE / T_m, src_vmm.s);
            st1h(src_vmm.s, opmask, ptr(effective_addr));
            break;
        case data_type::u8:
            eor(vzero.d, vzero.d, vzero.d); // since vzero might be spoiled
            saturate_f32(
                    vmm, vzero, vsaturation_ubound, data_type::u8, P_ALL_ONE);
            frinti(vmm.s, P_ALL_ONE / T_m, vmm.s);
            fcvtzu(vmm.s, P_ALL_ONE / T_m, vmm.s);
            smin(vmm.s, 127);
            st1b(vmm.s, opmask, ptr(effective_addr));
            // Need to restore data back to fp32 since we apply exp after
            // storing and data should be fp32.
            if (is_logsoftmax_) scvtf(vmm.s, opmask / T_m, vmm.s);
            break;
        case data_type::s8:
            saturate_f32(
                    vmm, vzero, vsaturation_ubound, data_type::s8, P_ALL_ONE);
            frinti(vmm.s, opmask / T_m, vmm.s);
            fcvtzs(vmm.s, opmask / T_m, vmm.s);
            smin(vmm.s, 127);
            smax(vmm.s, -128);
            st1b(vmm.s, opmask, ptr(effective_addr));
            // Need to restore data back to fp32 since we apply exp after
            // storing and data should be fp32.
            if (is_logsoftmax_) scvtf(vmm.s, opmask / T_m, vmm.s);
            break;
        default: assert(!"unsupported"); break;
    }
}

void jit_softmax_sve_t::load(
        const ZReg &vmm, const XReg &addr, const data_type_t dt, bool tail) {
    PReg tmp_mask = P_ALL_ONE;
    ZRegS effective_vmm = vmm.s;

    if (tail) tmp_mask = tail_opmask;

    switch (dt) {
        case data_type::f32: ld1w(effective_vmm, tmp_mask, ptr(addr)); break;
        case data_type::bf16:
            ld1h(effective_vmm, tmp_mask / T_z, ptr(addr));
            lsl(effective_vmm, effective_vmm,
                    0x10); // Shift left by 16 bits
            break;
        case data_type::f16:
            ld1h(effective_vmm, tmp_mask / T_z, ptr(addr));
            fcvt(effective_vmm, P_ALL_ONE / T_m, ZRegH(effective_vmm.getIdx()));
            break;
        case data_type::u8:
            ld1b(effective_vmm, tmp_mask / T_z, ptr(addr));
            scvtf(effective_vmm, P_ALL_ONE / T_m, effective_vmm);
            break;
        case data_type::s8:
            ld1sb(effective_vmm, tmp_mask / T_z, ptr(addr));
            scvtf(effective_vmm, P_ALL_ONE / T_m, effective_vmm);
            break;
        default: assert(!"unsupported"); break;
    }
}

void jit_softmax_sve_t::prepare_tail_mask() {
    set_preg(tail_opmask.s, axis_simd_tail_, X_TMP_0);
}

void jit_softmax_sve_t::get_horizontal_op(const ZReg &v, op_t op) {
    if (op == op_t::max)
        fmaxv(SReg(v.getIdx()), P_ALL_ONE, v.s);
    else
        faddv(SReg(v.getIdx()), P_ALL_ONE, v.s);

    dup(v.s, v.s[0]);
}

void jit_softmax_sve_t::accumulate_vmax() {
    // flush to -FLT_MAX before accumulation
    mov(vmax.d, vneg_flt_max.d);

    axis_loop([&](int unroll, bool tail = false) {
        for (int i = 0; i < unroll; i++) {
            ZReg vreg_tmp_src = ZReg(data_vreg_start_idx + i);
            load(vreg_tmp_src, src_ptr(src_axis_stride_ * i),
                    src_d_.data_type(), tail);
            if (tail) {
                uni_fmax(vmax, vmax, vreg_tmp_src, tail_opmask);
            } else {
                uni_fmax(vmax, vmax, vreg_tmp_src);
            }
        }
    });

    get_horizontal_op(vmax, op_t::max);
}

void jit_softmax_sve_t::accumulate_vsum() {
    // Initialize saturation vector register
    if (utils::one_of(dst_d_.data_type(), data_type::u8, data_type::s8)) {
        init_saturate_f32(vzero, vsaturation_ubound, reg_tmp, data_type::f32,
                dst_d_.data_type());
    }

    eor(vsum.d, vsum.d, vsum.d); // flush to zero before accumulation

    axis_loop([&](int unroll, bool tail = false) {
        const int vmm_start = data_vreg_start_idx;
        const int vmm_end = vmm_start + unroll;
        for (int i = 0; i < unroll; i++) {
            ZReg vreg_tmp_src = ZReg(data_vreg_start_idx + i);
            load(vreg_tmp_src, src_ptr(src_axis_stride_ * i),
                    src_d_.data_type(), tail);
            fsub(vreg_tmp_src.s, vreg_tmp_src.s, vmax.s);
            if (is_logsoftmax_) { // store before applying exp
                if (need_scratchpad_) {
                    store(interim_ptr(interim_axis_stride_ * i), vreg_tmp_src,
                            data_type::f32, tail);
                } else {
                    store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                            dst_d_.data_type(), tail);
                }
            }
        }

        exp_injector_->compute_vector_range(vmm_start, vmm_end);

        for (int i = 0; i < unroll; i++) {
            ZReg vreg_tmp_src = ZReg(data_vreg_start_idx + i);
            if (tail)
                fadd(vsum.s, tail_opmask / T_m, vreg_tmp_src.s);
            else
                fadd(vsum.s, vsum.s, vreg_tmp_src.s);
            if (is_softmax_) { // store after applying exp
                if (need_scratchpad_) {
                    store(interim_ptr(interim_axis_stride_ * i), vreg_tmp_src,
                            data_type::f32, tail);
                } else {
                    store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                            dst_d_.data_type(), tail);
                }
            }
        }
    });

    get_horizontal_op(vsum, op_t::sum);
    if (is_softmax_) {
        mov(v_tmp0.d, vsum.d);
        mov(vsum.d, P_ALL_ONE, vone.d);
        fdiv(vsum.s, P_ALL_ONE / T_m, v_tmp0.s);
        if (is_inf_as_zero_) {
            fdup(vtmp.s, 0.5);
            fcmlt(P_TMP.s, P_ALL_ONE / T_z, v_tmp0.s, vtmp.s);
            mov(vsum.s, P_TMP / T_m, 0);
        }
    }
    if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
}

void jit_softmax_sve_t::compute_dst() {
    axis_loop([&](int unroll, bool tail) {
        for (int i = 0; i < unroll; i++) {
            ZReg vreg_tmp_src = ZReg(data_vreg_start_idx + i);
            if (need_scratchpad_) {
                load(vreg_tmp_src, interim_ptr(interim_axis_stride_ * i),
                        data_type::f32, tail);
            } else {
                load(vreg_tmp_src, dst_ptr(dst_axis_stride_ * i),
                        dst_d_.data_type(), tail);
            }

            if (is_softmax_) { fmul(vreg_tmp_src.s, vreg_tmp_src.s, vsum.s); }
            if (is_logsoftmax_) {
                fsub(vreg_tmp_src.s, vreg_tmp_src.s, vsum.s);
            }

            ZReg vscale = vmax;
            if (need_src_scale_) {
                ldr(vscale, ptr(reg_src_scales));
                fmul(vreg_tmp_src.s, vreg_tmp_src.s, vscale.s);
            }

            if (need_dst_scale_) {
                ldr(vscale, ptr(reg_dst_scales));
                fmul(vreg_tmp_src.s, vreg_tmp_src.s, vscale.s);
            }

            store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                    dst_d_.data_type(), tail);
        }
    });
}

void jit_softmax_sve_t::accumulate_vsbr() {
    eor(vsbr.d, vsbr.d, vsbr.d); // flush to zero before accumulation

    axis_loop([&](int unroll, bool tail = false) {
        for (int i = 0; i < unroll; i++) {
            ZReg vreg_tmp_dst = ZReg(i * 2 + 1);
            ZReg vreg_tmp_diff_dst = ZReg(i * 2 + 2);
            load(vreg_tmp_diff_dst, diff_dst_ptr(diff_dst_axis_stride_ * i),
                    diff_dst_d_.data_type(), tail);
            if (is_softmax_) {
                load(vreg_tmp_dst, dst_ptr(dst_axis_stride_ * i),
                        dst_d_.data_type(), tail);
                fmul(vreg_tmp_diff_dst.s, vreg_tmp_diff_dst.s, vreg_tmp_dst.s);
            }
            fadd(vsbr.s, vsbr.s, vreg_tmp_diff_dst.s);
        }
    });

    get_horizontal_op(vsbr, op_t::sum);
}

void jit_softmax_sve_t::compute_diff_src() {
    axis_loop([&](int unroll, bool tail) {
        for (int i = 0; i < unroll; i++) {
            ZReg vreg_tmp_dst = ZReg(i * 2 + data_vreg_start_idx);
            ZReg vreg_tmp_diff_dst = ZReg(i * 2 + data_vreg_start_idx + 1);
            load(vreg_tmp_dst, dst_ptr(dst_axis_stride_ * i),
                    dst_d_.data_type(), tail);
            load(vreg_tmp_diff_dst, diff_dst_ptr(diff_dst_axis_stride_ * i),
                    diff_dst_d_.data_type(), tail);
            if (is_softmax_) {
                fsub(vreg_tmp_diff_dst.s, vreg_tmp_diff_dst.s, vsbr.s);
                fmul(vreg_tmp_diff_dst.s, vreg_tmp_dst.s, vreg_tmp_diff_dst.s);
            }
            if (is_logsoftmax_) {
                exp_injector_->compute_vector(vreg_tmp_dst.getIdx());
                fmls(vreg_tmp_diff_dst.s, P_ALL_ONE / T_m, vreg_tmp_dst.s,
                        vsbr.s);
            }
            store(diff_src_ptr(src_axis_stride_ * i), vreg_tmp_diff_dst,
                    src_d_.data_type(), tail);
        }
    });
}

void jit_softmax_sve_t::backward() {
    accumulate_vsbr();
    compute_diff_src();
}

void jit_softmax_sve_t::prepare_mask() {
    if (simd_w_ > cpu_isa_traits<sve_128>::vlen / sizeof(float)) {
        sub_imm(sp, sp, 64 * 2, X_TMP_0);
        str(p_shuff0, ptr(sp, 0, MUL_VL));
        str(p_shuff1, ptr(sp, 1, MUL_VL));
        not_(P_TMP_1.b, P_ALL_ONE, P_ALL_ONE.b);
        trn1(p_shuff0.d, P_ALL_ONE.d, P_TMP_1.d);
        trn1(p_shuff0.d, p_shuff0.d, p_shuff0.d);
        trn1(p_shuff1.s, P_ALL_ONE.s, P_TMP_1.s);
    }

    if (simd_w_ != cpu_sveLen / sizeof(float))
        set_preg(P_ALL_ONE.s, simd_w_, X_TMP_0);
}

void jit_softmax_sve_t::restore_mask() {
    if (simd_w_ > cpu_isa_traits<sve_128>::vlen / sizeof(float)) {
        ldr(p_shuff0, ptr(sp, 0, MUL_VL));
        ldr(p_shuff1, ptr(sp, 1, MUL_VL));
        add_imm(sp, sp, 64 * 2, X_TMP_0);
    }
}

void jit_softmax_sve_t::generate() {
    if (pd_->is_fwd() || is_logsoftmax_) {
        exp_injector_.reset(new jit_uni_eltwise_injector_t<sve>(this,
                alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, false,
                reg_exp_injector_table, injector_mask, injector_tmp));
        exp_injector_->set_input_range(-INFINITY, 0.f);
    }
    if (pd_->is_fwd() && is_logsoftmax_) {
        log_injector_.reset(new jit_uni_eltwise_injector_t<sve>(this,
                alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, true,
                reg_log_injector_table, injector_mask, injector_tmp));
    }

    compute_predefined_variables();
    preamble();

    prepare_mask();

    if (exp_injector_) exp_injector_->load_table_addr();
    if (log_injector_) log_injector_->load_table_addr();
    if (axis_simd_tail_) prepare_tail_mask();
    load_common_params();
    if (pd_->is_fwd())
        forward();
    else
        backward();

    restore_mask();
    postamble();
    if (exp_injector_) exp_injector_->prepare_table();
    if (log_injector_) log_injector_->prepare_table();
}

jit_softmax_sve_t::jit_softmax_sve_t(const softmax_pd_t *pd, int vlen)
    : jit_softmax_base_t(pd, vlen) {}

// ASIMD JIT softmax generator
struct jit_softmax_asimd_t : public jit_softmax_base_t {
    std::unique_ptr<jit_uni_eltwise_injector_t<asimd>> exp_injector_;
    std::unique_ptr<jit_uni_eltwise_injector_t<asimd>> log_injector_;

    const VReg vtmp = VReg(vtmp_idx);
    const VReg vneg_flt_max = VReg(vneg_flt_max_idx);
    const VReg vone = VReg(vone_idx);
    const VReg vsum = VReg(vsum_idx);
    const VReg vmax = VReg(vmax_idx);
    const VReg vzero = VReg(vzero_idx);

    void store(
            const XReg &addr, const VReg &vmm, const data_type_t dt, bool tail);
    void load(const VReg &vmm, const XReg &addr, const data_type_t dt,
            bool tail, const VReg &fill);
    void get_horizontal_op(const VReg &v, op_t op);
    void accumulate_vmax() override;
    void accumulate_vsum() override;
    void compute_dst() override;
    void generate() override;

    // ASIMD-specific functions
    void clear_unused_tail_lanes(const VReg &vmm);

    jit_softmax_asimd_t(const softmax_pd_t *pd);
};

void jit_softmax_asimd_t::load(const VReg &vmm, const XReg &addr,
        const data_type_t dt, bool tail, const VReg &fill) {
    if (!tail) {
        switch (dt) {
            case data_type::f32: ldr(QReg(vmm.getIdx()), ptr(addr)); break;
            case data_type::f16:
                ld1(VReg4H(vmm.getIdx()), ptr(addr));
                fcvtl(VReg4S(vmm.getIdx()), VReg4H(vmm.getIdx()));
                break;
            default: assert(!"unsupported"); break;
        }
        return;
    }

    mov(VReg16B(vmm.getIdx()), VReg16B(fill.getIdx()));
    mov(X_TMP_0, addr);

    if (dt == data_type::f32) {
        // Handle tail lane 0 separately.
        ld1(VReg4S(vmm.getIdx())[0], ptr(X_TMP_0));
        // Fill lanes [1, axis_simd_tail_) with per-lane loads.
        for (size_t i = 1; i < axis_simd_tail_; i++) {
            add_imm(X_TMP_1, X_TMP_0, i * sizeof(float), X_TMP_2);
            ld1(VReg4S(vmm.getIdx())[i], ptr(X_TMP_1));
        }
    } else if (dt == data_type::f16) {
        HReg cvt_vmm = HReg(24);
        // Handle tail lane 0 separately.
        ldr(cvt_vmm, ptr(X_TMP_0));
        fcvt(SReg(cvt_vmm.getIdx()), cvt_vmm);
        mov(VReg4S(vmm.getIdx())[0], VReg4S(cvt_vmm.getIdx())[0]);
        // Fill lanes [1, axis_simd_tail_) with per-lane loads.
        for (size_t i = 1; i < axis_simd_tail_; i++) {
            add_imm(X_TMP_1, X_TMP_0, i * sizeof(float16_t), X_TMP_2);
            ldr(cvt_vmm, ptr(X_TMP_1));
            fcvt(SReg(cvt_vmm.getIdx()), cvt_vmm);
            mov(VReg4S(vmm.getIdx())[i], VReg4S(cvt_vmm.getIdx())[0]);
        }
    } else {
        assert(!"unsupported");
    }
}

void jit_softmax_asimd_t::clear_unused_tail_lanes(const VReg &vmm) {
    if (!axis_simd_tail_) return;
    for (size_t lane = axis_simd_tail_; lane < simd_w_; lane++)
        mov(vmm.s[lane], vzero.s[0]);
}

void jit_softmax_asimd_t::store(
        const XReg &addr, const VReg &vmm, const data_type_t dt, bool tail) {
    if (!tail || axis_is_blocked_) {
        switch (dt) {
            case data_type::f32: str(QReg(vmm.getIdx()), ptr(addr)); break;
            case data_type::f16:
                fcvtn(VReg4H(vmm.getIdx()), vmm.s);
                st1(VReg4H(vmm.getIdx()), ptr(addr));
                break;
            default: assert(!"unsupported"); break;
        }
        return;
    }

    mov(X_TMP_0, addr);
    if (dt == data_type::f32) {
        // Handle tail lane 0 separately.
        st1(VReg4S(vmm.getIdx())[0], ptr(X_TMP_0));
        // Store lanes [1, axis_simd_tail_) with per-lane stores.
        for (size_t i = 1; i < axis_simd_tail_; i++) {
            add_imm(X_TMP_1, X_TMP_0, i * sizeof(float), X_TMP_2);
            st1(VReg4S(vmm.getIdx())[i], ptr(X_TMP_1));
        }
    } else if (dt == data_type::f16) {
        fcvtn(VReg4H(vmm.getIdx()), vmm.s);
        // Handle tail lane 0 separately.
        st1(VReg4H(vmm.getIdx())[0], ptr(X_TMP_0));
        // Store lanes [1, axis_simd_tail_) with per-lane stores.
        for (size_t i = 1; i < axis_simd_tail_; i++) {
            add_imm(X_TMP_1, X_TMP_0, i * sizeof(float16_t), X_TMP_2);
            st1(VReg4H(vmm.getIdx())[i], ptr(X_TMP_1));
        }
    } else {
        assert(!"unsupported");
    }
}

void jit_softmax_asimd_t::get_horizontal_op(const VReg &v, op_t op) {
    if (op == op_t::max) {
        fmaxv(SReg(v.getIdx()), v.s);
    } else {
        faddp(v.s, v.s, v.s);
        faddp(v.s, v.s, v.s);
    }

    dup(v.s, v.s[0]);
}

void jit_softmax_asimd_t::accumulate_vmax() {
    mov(VReg16B(vmax.getIdx()), VReg16B(vneg_flt_max.getIdx()));

    axis_loop([&](int unroll, bool tail) {
        for (int i = 0; i < unroll; i++) {
            VReg vreg_tmp_src = VReg(data_vreg_start_idx + i);
            load(vreg_tmp_src, src_ptr(src_axis_stride_ * i),
                    src_d_.data_type(), tail, vneg_flt_max);
            fmax(vmax.s, vmax.s, vreg_tmp_src.s);
        }
    });

    get_horizontal_op(vmax, op_t::max);
}

void jit_softmax_asimd_t::accumulate_vsum() {
    eor(vsum.b, vsum.b, vsum.b); // flush to zero before accumulation
    eor(vzero.b, vzero.b, vzero.b);

    axis_loop([&](int unroll, bool tail) {
        const int vmm_start = data_vreg_start_idx;
        const int vmm_end = vmm_start + unroll;

        for (int i = 0; i < unroll; i++) {
            VReg vreg_tmp_src = VReg(data_vreg_start_idx + i);
            load(vreg_tmp_src, src_ptr(src_axis_stride_ * i),
                    src_d_.data_type(), tail, vzero);
            fsub(vreg_tmp_src.s, vreg_tmp_src.s, vmax.s);
            if (is_logsoftmax_) { // store before applying exp
                if (need_scratchpad_) {
                    store(interim_ptr(interim_axis_stride_ * i), vreg_tmp_src,
                            data_type::f32, tail);
                } else {
                    store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                            dst_d_.data_type(), tail);
                }
            }
        }

        exp_injector_->compute_vector_range(vmm_start, vmm_end);

        for (int i = 0; i < unroll; i++) {
            VReg vreg_tmp_src = VReg(data_vreg_start_idx + i);
            if (tail) clear_unused_tail_lanes(vreg_tmp_src);
            fadd(vsum.s, vsum.s, vreg_tmp_src.s);
            if (is_softmax_) { // store after applying exp
                if (need_scratchpad_) {
                    store(interim_ptr(interim_axis_stride_ * i), vreg_tmp_src,
                            data_type::f32, tail);
                } else {
                    store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                            dst_d_.data_type(), tail);
                }
            }
        }
    });

    get_horizontal_op(vsum, op_t::sum);
    if (is_softmax_) {
        if (is_inf_as_zero_) {
            fmov(vtmp.s, 0.5);
            fcmgt(vtmp.s, vtmp.s, vsum.s);
        }
        fdiv(vsum.s, vone.s, vsum.s);
        if (is_inf_as_zero_) bic(vsum.b, vsum.b, vtmp.b);
    }
    if (is_logsoftmax_) log_injector_->compute_vector(vsum.getIdx());
}

void jit_softmax_asimd_t::compute_dst() {
    axis_loop([&](int unroll, bool tail) {
        for (int i = 0; i < unroll; i++) {
            VReg vreg_tmp_src = VReg(data_vreg_start_idx + i);
            if (need_scratchpad_) {
                load(vreg_tmp_src, interim_ptr(interim_axis_stride_ * i),
                        data_type::f32, tail, vzero);
            } else {
                load(vreg_tmp_src, dst_ptr(dst_axis_stride_ * i),
                        dst_d_.data_type(), tail, vzero);
            }
            if (is_softmax_) { fmul(vreg_tmp_src.s, vreg_tmp_src.s, vsum.s); }
            if (is_logsoftmax_) {
                fsub(vreg_tmp_src.s, vreg_tmp_src.s, vsum.s);
                if (tail && axis_is_blocked_) {
                    clear_unused_tail_lanes(vreg_tmp_src);
                }
            }

            if (need_src_scale_) {
                const auto &v_src_scale = vmax;
                ld1r(v_src_scale.s, ptr(reg_src_scales));
                fmul(vreg_tmp_src.s, vreg_tmp_src.s, v_src_scale.s);
            }

            if (need_dst_scale_) {
                const auto &v_dst_scale = vmax;
                ld1r(v_dst_scale.s, ptr(reg_dst_scales));
                fmul(vreg_tmp_src.s, vreg_tmp_src.s, v_dst_scale.s);
            }

            store(dst_ptr(dst_axis_stride_ * i), vreg_tmp_src,
                    dst_d_.data_type(), tail);
        }
    });
}

void jit_softmax_asimd_t::generate() {
    assert(pd_->is_fwd());

    exp_injector_.reset(new jit_uni_eltwise_injector_t<asimd>(this,
            alg_kind::eltwise_exp, 0.0f, 0.0f, 1.0f, false,
            reg_exp_injector_table, injector_mask, injector_tmp));
    exp_injector_->set_input_range(-INFINITY, 0.f);

    if (is_logsoftmax_) {
        log_injector_.reset(new jit_uni_eltwise_injector_t<asimd>(this,
                alg_kind::eltwise_log, 0.0f, 0.0f, 1.0f, true,
                reg_log_injector_table, injector_mask, injector_tmp));
    }

    compute_predefined_variables();
    preamble();

    exp_injector_->load_table_addr();
    if (log_injector_) log_injector_->load_table_addr();
    load_common_params();
    forward();

    postamble();
    exp_injector_->prepare_table();
    if (log_injector_) log_injector_->prepare_table();
}

jit_softmax_asimd_t::jit_softmax_asimd_t(const softmax_pd_t *pd)
    : jit_softmax_base_t(pd, cpu_isa_traits<asimd>::vlen) {}

// JIT softmax specializations
template <cpu_isa_t isa>
struct jit_softmax_t;

template <>
struct jit_softmax_t<sve> : public jit_softmax_sve_t {
    jit_softmax_t(const softmax_pd_t *pd)
        : jit_softmax_sve_t(pd, simd_bytes(sve)) {}
};

template <>
struct jit_softmax_t<asimd> : public jit_softmax_asimd_t {
    jit_softmax_t(const softmax_pd_t *pd) : jit_softmax_asimd_t(pd) {}
};

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::jit_uni_softmax_fwd_t(const pd_t *apd)
    : primitive_t(apd)
    , softmax_driver_(utils::make_unique<softmax_impl::driver_t<isa>>(pd())) {}

template <cpu_isa_t isa>
jit_uni_softmax_fwd_t<isa>::~jit_uni_softmax_fwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::init(engine_t *engine) {
    return softmax_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_fwd_t<isa>::execute(const exec_ctx_t &ctx) const {
    const auto src = CTX_IN_MEM(const char *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(char *, DNNL_ARG_DST);
    auto scratchpad_ptr = ctx.get_scratchpad_grantor().template get<char>(
            memory_tracking::names::key_softmax_interim_store);

    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const auto src_data_type_size = src_d.data_type_size();
    const auto dst_data_type_size = dst_d.data_type_size();
    const auto &bd = src_d.blocking_desc();
    const auto axis = pd()->axis();

    const auto axis_size_padded = pd()->axis_size(true);
    const auto inner_stride
            = bd.inner_nblks ? bd.inner_blks[bd.inner_nblks - 1] : (dim_t)1;
    const auto inner_size = bd.strides[axis] / inner_stride;
    const auto process_n_elems = pd()->axis_size() * inner_size;
    const auto outer_stride = axis_size_padded * inner_size;
    const auto outer_size = src_d.nelems(true) / outer_stride;

    const int nthr = pd()->nthr_;

    parallel_nd_ext(nthr, outer_size, inner_size,
            [= COMPAT_THIS_CAPTURE](int ithr, int, dim_t ou, dim_t in) {
        dim_t offset = (ou * outer_stride + in * inner_stride);
        const char *src_ptr = src + offset * src_data_type_size;
        char *dst_ptr = dst + offset * dst_data_type_size;
        char *interim_ptr = scratchpad_ptr
                ? scratchpad_ptr + ithr * axis_size_padded * sizeof(float)
                : nullptr;
        softmax_driver_->exec(src_ptr, dst_ptr, interim_ptr, src_scales,
                dst_scales, process_n_elems);
    });

    return status::success;
}

template <cpu_isa_t isa>
jit_uni_softmax_bwd_t<isa>::jit_uni_softmax_bwd_t(const pd_t *apd)
    : primitive_t(apd)
    , softmax_driver_(utils::make_unique<softmax_impl::driver_t<isa>>(pd())) {}

template <cpu_isa_t isa>
jit_uni_softmax_bwd_t<isa>::~jit_uni_softmax_bwd_t() = default;

template <cpu_isa_t isa>
status_t jit_uni_softmax_bwd_t<isa>::init(engine_t *engine) {
    return softmax_driver_->create_kernel();
}

template <cpu_isa_t isa>
status_t jit_uni_softmax_bwd_t<isa>::execute(const exec_ctx_t &ctx) const {
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
        softmax_driver_->exec(
                diff_src_ptr, dst_ptr, diff_dst_ptr, process_n_elems);
    });

    return status::success;
}

namespace softmax_impl {

template <cpu_isa_t isa>
struct driver_t {

    driver_t(const softmax_pd_t *pd) : pd_(pd), ker_(pd_) {}

    void exec(const void *src, void *dst, void *interim, const void *src_scales,
            const void *dst_scales, const dim_t process_n_elems) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.process_n_elems = process_n_elems;
        p.src = src;
        p.dst = dst;
        p.interim = interim;
        p.src_scales = src_scales;
        p.dst_scales = dst_scales;
        ker_(&p);
    }

    void exec(void *diff_src, const void *dst, const void *diff_dst,
            const dim_t process_n_elems) {
        typename jit_softmax_t<isa>::call_params_t p;
        p.process_n_elems = process_n_elems;
        p.src = diff_src;
        p.dst = dst;
        p.diff_dst = diff_dst;
        ker_(&p);
    }

    status_t create_kernel() { return ker_.create_kernel(); }

private:
    const softmax_pd_t *pd_;
    jit_softmax_t<isa> ker_;
};

} // namespace softmax_impl

/* struct instantiation */
template struct jit_uni_softmax_fwd_t<sve>;
template struct jit_uni_softmax_fwd_t<asimd>;

template struct jit_uni_softmax_bwd_t<sve>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
