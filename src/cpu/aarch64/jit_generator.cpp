/*******************************************************************************
* Copyright 2016 Intel Corporation
* Copyright 2020-2025 FUJITSU LIMITED
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
#include <cstddef>

#include "cpu/aarch64/jit_generator.hpp"
#include "cpu/aarch64/jit_op_imm_check.hpp"

using namespace Xbyak_aarch64;

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

void jit_generator_t::preamble() {
    uint64_t sveLen = get_sve_length();

    stp(x29, x30, pre_ptr(sp, -16));
    /* x29 is a frame pointer. */
    mov(x29, sp);
    sub(sp, sp, static_cast<int64_t>(preserved_stack_bytes) - 16);

    /* x9 can be used as a temporal register. */
    mov(x9, sp);

    if (vreg_to_preserve) {
        st4((v8.d - v11.d)[0], post_ptr(x9, vreg_bytes_preserve * 4));
        st4((v12.d - v15.d)[0], post_ptr(x9, vreg_bytes_preserve * 4));
    }
    for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
        stp(abi_save_gpr_regs[i], abi_save_gpr_regs[i + 1],
                post_ptr(x9, xreg_size_in_bytes * 2));
    }

    if (sveLen) { /* SVE is available. */
        ptrue(P_ALL_ONE.b);
        pfalse(P_ALL_ZERO.b);
    }
    if (sveLen >= Xbyak_aarch64::util::SVE_256) {
        ptrue(P_NOT_128.b, VL16);
        not_(P_NOT_128.b, P_ALL_ONE / T_z, P_NOT_128.b);
    }
    if (sveLen >= Xbyak_aarch64::util::SVE_512) {
        ptrue(P_NOT_256.b, VL32);
        not_(P_NOT_256.b, P_ALL_ONE / T_z, P_NOT_256.b);
    }
}

void jit_generator_t::postamble() {
    mov(x9, sp);

    if (vreg_to_preserve) {
        ld4((v8.d - v11.d)[0], post_ptr(x9, vreg_bytes_preserve * 4));
        ld4((v12.d - v15.d)[0], post_ptr(x9, vreg_bytes_preserve * 4));
    }

    for (size_t i = 0; i < num_abi_save_gpr_regs; i += 2) {
        ldp(XReg(abi_save_gpr_regs[i]), XReg(abi_save_gpr_regs[i + 1]),
                post_ptr(x9, xreg_size_in_bytes * 2));
    }

    add(sp, sp, static_cast<int64_t>(preserved_stack_bytes) - 16);
    ldp(x29, x30, post_ptr(sp, 16));
    ret();
}

void jit_generator_t::L(Label &label) {
    CodeGenerator::L(label);
}

void jit_generator_t::L_aligned(Label &label, int alignment) {
    align(alignment);
    L(label);
}

// Return XReg which points to base + stride_bytes * n in as few instructions as possible
// We return a XReg so that we can use the dst or base reg
XReg jit_generator_t::strided_addr(const XReg &dst, const XReg &base,
        const XReg &reg_stride_bytes, const int32_t stride_bytes,
        const int32_t n, const XReg &tmp) {
    int32_t offset_bytes = stride_bytes * n;
    if (n == 0) {
        // (0 instructions) Use the base
        return base;
    } else if (is_imm12(offset_bytes)) {
        // (1 instruction) The byte offset can be expressed as an int12 immediate in the add instruction
        add(dst, base, static_cast<uint32_t>(offset_bytes & 0xfff));
        return dst;
    } else {
        const auto LSL = ShMod::LSL;
        switch (n) {
            case 1:
                // 1 instruction
                add(dst, base, reg_stride_bytes);
                return dst;
            case 2: add(dst, base, reg_stride_bytes, LSL, 1); return dst;
            case 4: add(dst, base, reg_stride_bytes, LSL, 2); return dst;
            case 8: add(dst, base, reg_stride_bytes, LSL, 3); return dst;
            case 16: add(dst, base, reg_stride_bytes, LSL, 4); return dst;
            default: add_imm(dst, base, offset_bytes, tmp); return dst;
        }
    }
}

void jit_generator_t::uni_add(
        const VReg4S &x1, const VReg4S &x2, const VReg4S &op) {
    add(x1, x2, op);
}

void jit_generator_t::uni_add(const ZReg &x1, const ZReg &x2, const ZReg &op) {
    add(ZRegS(x1.getIdx()), ZRegS(x2.getIdx()), ZRegS(op.getIdx()));
}

void jit_generator_t::uni_clear(const VReg &dst) {
    // movi 0 on any part of the SIMD&FP reg will also zero out the rest of the
    // SIMD vector (see the Arm ARM C7.1.1).
    // We use this because movi d is zero latency on Neoverse V2
    movi(DReg(dst.getIdx()), 0);
}
void jit_generator_t::uni_clear(const ZReg &dst) {
    // movi 0 on any part of the SIMD&FP reg will also zero out the rest of the SVE vector
    // (see the Arm ARM C7.1.1 and B1.4.3 R_WKYLB)
    // We use this because movi d is zero latency on Neoverse V2
    movi(DReg(dst.getIdx()), 0);
}

// Float addition across all vector elements
void jit_generator_t::uni_fadd_reduce(const SReg &dst, const ZRegS &src) {
    faddv(dst, P_ALL_ONE / T_z, src);
}

void jit_generator_t::uni_fadd_reduce(const SReg &dst, const VReg4S &src) {
    const auto &vreg_dst = VReg(dst.getIdx());

    faddp(vreg_dst.s, src, src);
    faddp(vreg_dst.s2, vreg_dst.s2, vreg_dst.s2);
}

void jit_generator_t::uni_fcvtzs(const VReg4S &d, const VReg4S &s) {
    fcvtzs(d, s);
}

void jit_generator_t::uni_fcvtzs(const ZRegS &d, const ZRegS &s) {
    fcvtzs(d, P_ALL_ONE / T_z, s);
}

void jit_generator_t::uni_fdiv(const VReg4S &dst, const VReg4S &src,
        const VReg4S &src2, const VReg4S &tmp, const PReg &pred) {
    UNUSED(tmp);
    UNUSED(pred);
    fdiv(dst, src, src2);
}

void jit_generator_t::uni_fmad(
        const VReg4S &dst, const VReg4S &src, const VReg4S &src2) {
    fmul(dst, dst, src);
    fadd(dst, dst, src2);
}

void jit_generator_t::float_point_fused_multiply_add(
        const ZReg &dst_vmm, const ZReg &src1_vmm, const ZReg &src2_vmm) {
    fmla(dst_vmm.s, P_ALL_ONE / T_m, src1_vmm.s, src2_vmm.s);
}

void jit_generator_t::float_point_fused_multiply_add(
        const VReg &dst_vmm, const VReg &src1_vmm, const VReg &src2_vmm) {
    fmla(dst_vmm.s, src1_vmm.s, src2_vmm.s);
}

void jit_generator_t::uni_fmax(
        const VReg4S &dst, const VReg4S &src, const VReg4S &src2) {
    uint32_t dstIdx = dst.getIdx();
    uint32_t srcIdx = src.getIdx();
    if (dstIdx != srcIdx) mov(VReg16B(dstIdx), VReg16B(srcIdx));
    fmax(dst, src, src2);
}

void jit_generator_t::uni_fmaxnm(
        const VReg4S &dst, const VReg4S &src, const VReg4S &src2) {
    fmaxnm(dst, src, src2);
}

void jit_generator_t::uni_fmin(
        const VReg4S &dst, const VReg4S &src, const VReg4S &src2) {
    uint32_t dstIdx = dst.getIdx();
    uint32_t srcIdx = src.getIdx();
    if (dstIdx != srcIdx) mov(VReg16B(dstIdx), VReg16B(srcIdx));
    fmin(dst, src, src2);
}

void jit_generator_t::contiguous_load_unsigned_words(
        const ZReg &src_vmm, const XReg &addr) {
    ld1w(src_vmm.s, P_ALL_ONE / T_z, ptr(addr));
}

void jit_generator_t::contiguous_load_unsigned_words(
        const VReg &src_vmm, const XReg &addr) {
    ld1(src_vmm.s, ptr(addr));
}

void jit_generator_t::uni_frinti(const VReg4S &d, const VReg4S &s) {
    frinti(d, s);
}

void jit_generator_t::uni_frinti(const ZRegS &d, const ZRegS &s) {
    frinti(d, P_ALL_ONE / T_m, s);
}

void jit_generator_t::uni_fsqrt(const VReg4S &dst, const VReg4S &src) {
    fsqrt(dst, src);
}

void jit_generator_t::uni_fsub(
        const VReg4S &v1, const VReg4S &v2, const VReg4S &v3) {
    fsub(v1, v2, v3);
}

void jit_generator_t::uni_fsub(
        const ZRegS &z1, const ZRegS &z2, const ZRegS &z3) {
    fsub(z1, z2, z3);
}

void jit_generator_t::uni_eor(const VReg &v1, const VReg &v2, const VReg &v3) {
    eor(VReg16B(v1.getIdx()), VReg16B(v2.getIdx()), VReg16B(v3.getIdx()));
}

void jit_generator_t::uni_eor(const ZReg &z1, const ZReg &z2, const ZReg &z3) {
    eor(ZRegD(z1.getIdx()), ZRegD(z2.getIdx()), ZRegD(z3.getIdx()));
}

void jit_generator_t::uni_ld1rw(
        const VReg4S &dst, const XReg &base, const int64_t off) {
    if (off == 0) {
        ld1r(dst, ptr(base));
    } else {
        add_imm(X_DEFAULT_ADDR, base, off, X_TMP_0);
        ld1r(dst, ptr(X_DEFAULT_ADDR));
    }
}

void jit_generator_t::uni_ld1rw(
        const ZRegS &dst, const XReg &base, const int64_t off) {
    if (-32 <= off && off < 32) {
        ld1rw(dst, P_ALL_ONE / T_z, ptr(base, (int)off));
    } else {
        add_imm(X_DEFAULT_ADDR, base, off, X_TMP_0);
        ld1rw(dst, P_ALL_ONE / T_z, ptr(X_DEFAULT_ADDR));
    }
}

void jit_generator_t::uni_ldr(const VReg &dst, const XReg &addr) {
    ldr(QReg(dst.getIdx()), ptr(addr));
}

void jit_generator_t::uni_ldr(const ZReg &dst, const XReg &addr) {
    ldr(dst, ptr(addr));
}

void jit_generator_t::uni_load_vmm_stack(
        const VReg &dst, const XReg &base, size_t vmm_slot) {
    ldr(QReg(dst.getIdx()), ptr(base, static_cast<uint32_t>(vmm_slot * 16)));
}

void jit_generator_t::uni_load_vmm_stack(
        const ZReg &dst, const XReg &base, size_t vmm_slot) {
    ldr(dst, ptr(base, vmm_slot, MUL_VL));
}

void jit_generator_t::uni_mov(const VReg &d, const VReg &s) {
    if (d.getIdx() != s.getIdx()) { mov(d.b16, s.b16); }
}

void jit_generator_t::uni_mov(const ZReg &d, const ZReg &s) {
    if (d.getIdx() != s.getIdx()) { mov(d.d, s.d); }
}

void jit_generator_t::uni_dup(const VReg4S &dst, int32_t value) {
    assert(value >= -128 && value <= 127);

    if (value >= 0)
        movi(dst, static_cast<uint32_t>(value));
    else
        mvni(dst, static_cast<uint32_t>(~value));
}

void jit_generator_t::uni_dup(const ZRegS &dst, int32_t value) {
    assert(value >= -128 && value <= 127);
    dup(dst, value);
}

void jit_generator_t::uni_orr(const VReg &d, const VReg &s0, const VReg &s1) {
    orr(d.b16, s0.b16, s1.b16);
}

void jit_generator_t::uni_orr(const ZReg &d, const ZReg &s0, const ZReg &s1) {
    orr(d.d, s0.d, s1.d);
}

void jit_generator_t::uni_scvtf(const VReg4S &t0, const VReg4S &t1) {
    scvtf(t0, t1);
}

void jit_generator_t::uni_scvtf(const ZRegS &t0, const ZRegS &t1) {
    scvtf(t0, P_ALL_ONE / T_m, t1);
}

void jit_generator_t::uni_str(const VReg &src, const XReg &addr) {
    str(QReg(src.getIdx()), ptr(addr));
}

void jit_generator_t::uni_str(const ZReg &src, const XReg &addr) {
    str(src, ptr(addr));
}

void jit_generator_t::uni_store_vmm_stack(
        const VReg &src, const XReg &base, size_t vmm_slot) {
    str(QReg(src.getIdx()), ptr(base, static_cast<uint32_t>(vmm_slot * 16)));
}

void jit_generator_t::uni_store_vmm_stack(
        const ZReg &src, const XReg &base, size_t vmm_slot) {
    str(src, ptr(base, vmm_slot, MUL_VL));
}

void jit_generator_t::uni_reserve_vmm_stack(size_t count, const VReg &) {
    sub(sp, sp, count * 16);
}

void jit_generator_t::uni_reserve_vmm_stack(size_t count, const ZReg &) {
    addvl(sp, sp, -static_cast<int>(count));
}

void jit_generator_t::uni_release_vmm_stack(size_t count, const VReg &) {
    add(sp, sp, count * 16);
}

void jit_generator_t::uni_release_vmm_stack(size_t count, const ZReg &) {
    addvl(sp, sp, static_cast<int>(count));
}

void jit_generator_t::jit_breakpoint() {
#if defined(DNNL_DEV_MODE)
    // Meaningless magic number. Sort of looks like "xbyak" if you close your
    // eyes completely.
    brk(0xb7ac);

    // The nop is just useful to sidestep a quirk of gdb. To skip over the
    // breakpoint, the user has to manually increment the program counter, but
    // the gdb UI does not update until the next step-insturction command. This
    // nop gives us a single instruction margin of error to stop at the real
    // point of interest.
    nop();
#endif
}

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
