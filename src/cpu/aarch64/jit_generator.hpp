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

#ifndef CPU_AARCH64_JIT_GENERATOR_HPP
#define CPU_AARCH64_JIT_GENERATOR_HPP

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <memory>
#include <set>
#include <vector>
#include <initializer_list>

#include "common/nstl.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/cpu_isa_traits.hpp"
#include "cpu/aarch64/jit_op_imm_check.hpp"
#include "cpu/jit_utils/jit_utils.hpp"

#if defined(_WIN32) && !defined(__GNUC__)
#define STRUCT_ALIGN(al, ...) __declspec(align(al)) __VA_ARGS__
#else
#define STRUCT_ALIGN(al, ...) __VA_ARGS__ __attribute__((__aligned__(al)))
#endif

#define DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_name) \
    const char *name() const override { \
        return STRINGIFY(jit_name); \
    } \
    const char *source_file() const override { \
        return __FILE__; \
    }

#define LD_MUL_VL(mn, op, mask, addr, off, size) \
    do { \
        if (use_mul_vl(off, size, cpu_sveLen)) \
            mn(op, (mask) / T_z, \
                    ptr(addr, compute_off_mul_vl(off, size, cpu_sveLen), \
                            MUL_VL)); \
        else \
            mn(op, (mask) / T_z, \
                    ptr(addr_off(addr, off, X_DEFAULT_ADDR, X_TMP_0))); \
    } while (0)

#define ST_MUL_VL(mn, op, mask, addr, off, size) \
    do { \
        if (use_mul_vl(off, size, cpu_sveLen)) \
            mn(op, (mask), \
                    ptr(addr, compute_off_mul_vl(off, size, cpu_sveLen), \
                            MUL_VL)); \
        else \
            mn(op, (mask), ptr(addr_off(addr, off, X_DEFAULT_ADDR, X_TMP_0))); \
    } while (0)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

class jit_generator_t : public Xbyak_aarch64::CodeGenerator {
private:
    static constexpr size_t xreg_size_in_bytes = 8;
    // Only bottom 8byte must be preserved.
    static constexpr size_t vreg_bytes_preserve = 8;
    static constexpr size_t vreg_to_preserve = 8; // VREG8 - VREG15

    static constexpr size_t num_abi_save_gpr_regs = 10;

    static constexpr size_t preserved_stack_bytes = xreg_size_in_bytes
                    * (num_abi_save_gpr_regs + /*frame pointer*/ 1
                            + /*link register*/ 1)
            + vreg_bytes_preserve * vreg_to_preserve;

    static constexpr size_t size_of_abi_save_regs
            = num_abi_save_gpr_regs * xreg_size_in_bytes
            + vreg_to_preserve * vreg_bytes_preserve;

public:
    // See "Procedure Call Standsard for the ARM 64-bit Architecture (AArch64)"
    const Xbyak_aarch64::XReg abi_param1 = x0;
    const Xbyak_aarch64::XReg &abi_param2 = x1;
    const Xbyak_aarch64::XReg &abi_param3 = x2;
    const Xbyak_aarch64::XReg &abi_param4 = x3;
    const Xbyak_aarch64::XReg &abi_param5 = x4;
    const Xbyak_aarch64::XReg &abi_param6 = x5;
    const Xbyak_aarch64::XReg &abi_param7 = x6;
    const Xbyak_aarch64::XReg &abi_param8 = x7;
    const Xbyak_aarch64::XReg &abi_not_param1 = x15;

    // Callee-saved registers
    const std::array<Xbyak_aarch64::XReg, num_abi_save_gpr_regs>
            abi_save_gpr_regs
            = {x19, x20, x21, x22, x23, x24, x25, x26, x27, x28};

    static constexpr int MAX_CODE_SIZE = 256 * 1024;

    const uint64_t cpu_sveLen = get_sve_length();

    const Xbyak_aarch64::WReg W_TMP_0 = w23;
    const Xbyak_aarch64::WReg W_TMP_1 = w24;
    const Xbyak_aarch64::WReg W_TMP_2 = w25;
    const Xbyak_aarch64::WReg W_TMP_3 = w26;
    const Xbyak_aarch64::WReg W_TMP_4 = w27;
    const Xbyak_aarch64::XReg X_TMP_0 = x23;
    const Xbyak_aarch64::XReg X_TMP_1 = x24;
    const Xbyak_aarch64::XReg X_TMP_2 = x25;
    const Xbyak_aarch64::XReg X_TMP_3 = x26;
    const Xbyak_aarch64::XReg X_TMP_4 = x27;
    const Xbyak_aarch64::XReg X_DEFAULT_ADDR = x28;
    const Xbyak_aarch64::PReg P_TMP = p7;
    const Xbyak_aarch64::PReg P_TMP_0 = p11;
    const Xbyak_aarch64::PReg P_TMP_1 = p12;
    const Xbyak_aarch64::PReg P_ALL_ZERO = p10;
    const Xbyak_aarch64::PReg P_NOT_256 = p13;
    const Xbyak_aarch64::PReg P_NOT_128 = p14;
    const Xbyak_aarch64::PReg P_ALL_ONE = p0;

    const std::array<Xbyak_aarch64::XReg, 5> temp_xregs
            = {X_TMP_0, X_TMP_1, X_TMP_2, X_TMP_3, X_TMP_4};

    const Xbyak_aarch64::XReg param1 = abi_param1;
    constexpr static uint32_t DUMMY_IDX = 99;

    inline size_t get_size_of_abi_save_regs() const {
        return size_of_abi_save_regs;
    }

    void preamble();

    void postamble();

    // Disallow char-based labels completely
    void L(const char *label) = delete;
    void L(Xbyak_aarch64::Label &label);

    void L_aligned(Xbyak_aarch64::Label &label, int alignment = 16);

    template <typename T>
    Xbyak_aarch64::XReg addr_off(const Xbyak_aarch64::XReg &base, const T off,
            const Xbyak_aarch64::XReg &addr, const Xbyak_aarch64::XReg &x_tmp) {
        if (off == 0) return base;

        add_imm(addr, base, off, x_tmp);
        return addr;
    }

    template <typename TReg>
    void ldr_imm(
            const TReg &reg, const Xbyak_aarch64::XReg &addr, uint32_t offset) {
        if (is_imm12(offset)) {
            ldr(reg, ptr(addr, offset));
        } else {
            add_imm(X_DEFAULT_ADDR, addr, offset, X_DEFAULT_ADDR);
            ldr(reg, ptr(X_DEFAULT_ADDR));
        }
    }

    template <typename TReg>
    void str_imm(
            const TReg &reg, const Xbyak_aarch64::XReg &addr, uint32_t offset) {
        if (is_imm12(offset)) {
            str(reg, ptr(addr, offset));
        } else {
            add_imm(X_DEFAULT_ADDR, addr, offset, X_DEFAULT_ADDR);
            str(reg, ptr(X_DEFAULT_ADDR));
        }
    }

    int compute_off_mul_vl(int off, int size, int cpu_sveLen) {
        const int mul_vl_len = (cpu_sveLen / 4) * size;
        return off / mul_vl_len;
    }

    bool use_mul_vl(int off, int size, int cpu_sveLen) {
        const int mul_vl_len = (cpu_sveLen / 4) * size;
        int off_mod = off % mul_vl_len;
        int off_mul_vl = off / mul_vl_len;
        return (off_mod == 0 && -8 <= off_mul_vl && off_mul_vl <= 7);
    }

    void add_vl_or_imm(Xbyak_aarch64::XReg dst, Xbyak_aarch64::XReg src,
            int offset, Xbyak_aarch64::XReg tmp) {
        // If offset is a multiple of the vector length and
        // offset / vector_length is compatible with addvl
        // use the addvl instruction. Refer to https://developer.arm.com/documentation/ddi0596/2021-03/SVE-Instructions/ADDVL--Add-multiple-of-vector-register-size-to-scalar-register-
        // for details.
        if ((offset % cpu_sveLen == 0)
                && (offset / static_cast<int>(cpu_sveLen) >= -32)
                && (offset / cpu_sveLen <= 31))
            addvl(dst, src, offset / cpu_sveLen);
        else
            add_imm(dst, src, offset, tmp);
    }

    // Return XReg which points to base + stride_bytes * n in as few instructions as possible
    // We return a XReg so that we can use the dst or base reg
    Xbyak_aarch64::XReg strided_addr(const Xbyak_aarch64::XReg &dst,
            const Xbyak_aarch64::XReg &base,
            const Xbyak_aarch64::XReg &reg_stride_bytes,
            const int32_t stride_bytes, const int32_t n,
            const Xbyak_aarch64::XReg &tmp);

    template <typename PRegBHSD>
    void set_preg(const PRegBHSD &p, size_t tail_size,
            const Xbyak_aarch64::XReg x_tmp) {
        using namespace Xbyak_aarch64;

        switch (tail_size) {
            case 0: pfalse(PRegB(p.getIdx())); return;
            case 1: ptrue(p, VL1); return;
            case 2: ptrue(p, VL2); return;
            case 3: ptrue(p, VL3); return;
            case 4: ptrue(p, VL4); return;
            case 5: ptrue(p, VL5); return;
            case 6: ptrue(p, VL6); return;
            case 7: ptrue(p, VL7); return;
            case 8: ptrue(p, VL8); return;
            case 16: ptrue(p, VL16); return;
            case 32: ptrue(p, VL32); return;
            case 64: ptrue(p, VL64); return;
        }

        mov_imm(x_tmp, tail_size);
        whilelo(p, xzr, x_tmp);
    }

    template <typename T>
    void uni_add(const T &x1, const T &x2, const T &op) {
        add(x1, x2, op);
    }

    void uni_add(const Xbyak_aarch64::VReg4S &x1,
            const Xbyak_aarch64::VReg4S &x2, const Xbyak_aarch64::VReg4S &op);

    void uni_add(const Xbyak_aarch64::ZReg &x1, const Xbyak_aarch64::ZReg &x2,
            const Xbyak_aarch64::ZReg &op);

    template <typename T>
    void udiv_mod(const T &q, const T &r, const T &divend, const T &divisor) {
        assert(q.getIdx() != divisor.getIdx());
        assert(q.getIdx() != divend.getIdx());
        assert(r.getIdx() != divend.getIdx());

        udiv(q, divend, divisor);
        mul(r, q, divisor);
        sub(r, divend, r);
    }

    template <typename T>
    void umod(const T &r, const T &divend, const T &divisor) {
        assert(r.getIdx() != divend.getIdx());
        assert(r.getIdx() != divisor.getIdx());

        udiv(r, divend, divisor);
        mul(r, r, divisor);
        sub(r, divend, r);
    }

    void uni_clear(const Xbyak_aarch64::VReg &dst);
    void uni_clear(const Xbyak_aarch64::ZReg &dst);

    template <typename T>
    void uni_fadd(const T &dst, const T &src, const T &src2) {
        fadd(dst, src, src2);
    }

    // Float addition across all vector elements
    void uni_fadd_reduce(
            const Xbyak_aarch64::SReg &dst, const Xbyak_aarch64::ZRegS &src);

    void uni_fadd_reduce(
            const Xbyak_aarch64::SReg &dst, const Xbyak_aarch64::VReg4S &src);

    void uni_fcvtzs(
            const Xbyak_aarch64::VReg4S &d, const Xbyak_aarch64::VReg4S &s);

    void uni_fcvtzs(
            const Xbyak_aarch64::ZRegS &d, const Xbyak_aarch64::ZRegS &s);

    template <typename TReg>
    void uni_fdiv(const TReg &dst, const TReg &src, const TReg &src2,
            const TReg &tmp, const Xbyak_aarch64::PReg &pred) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        uint32_t src2Idx = src2.getIdx();
        uint32_t tmpIdx = tmp.getIdx();

        if (dstIdx == src2Idx) {
            assert(tmpIdx != srcIdx && tmpIdx != src2Idx);

            mov(Xbyak_aarch64::ZRegD(tmpIdx), Xbyak_aarch64::ZRegD(src2Idx));
            mov(dst, pred / Xbyak_aarch64::T_m, src);
            fdiv(dst, pred / Xbyak_aarch64::T_m, tmp);
        } else if (dstIdx == srcIdx) {
            fdiv(dst, pred / Xbyak_aarch64::T_m, src2);
        } else {
            mov(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src);
            fdiv(dst, pred / Xbyak_aarch64::T_m, src2);
        }
    }

    template <typename TReg>
    void uni_fdiv(const TReg &dst, const TReg &src, const TReg &src2) {
        fdiv(dst, src, src2);
    }

    void uni_fdiv(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src, const Xbyak_aarch64::VReg4S &src2,
            const Xbyak_aarch64::VReg4S &tmp, const Xbyak_aarch64::PReg &pred);

    template <typename T>
    void uni_fmad(const T &dst, const T &src, const T &src2) {
        fmad(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src, src2);
    }

    void uni_fmad(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src,
            const Xbyak_aarch64::VReg4S &src2);

    void float_point_fused_multiply_add(const Xbyak_aarch64::ZReg &dst_vmm,
            const Xbyak_aarch64::ZReg &src1_vmm,
            const Xbyak_aarch64::ZReg &src2_vmm);

    void float_point_fused_multiply_add(const Xbyak_aarch64::VReg &dst_vmm,
            const Xbyak_aarch64::VReg &src1_vmm,
            const Xbyak_aarch64::VReg &src2_vmm);

    template <typename T>
    void uni_fmax(const T &dst, const T &src, const T &src2) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        if (dstIdx != srcIdx)
            mov(Xbyak_aarch64::ZRegD(dstIdx), Xbyak_aarch64::ZRegD(srcIdx));
        fmax(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src2);
    }

    void uni_fmax(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src,
            const Xbyak_aarch64::VReg4S &src2);

    template <typename T>
    void uni_fmaxnm(const T &dst, const T &src, const T &src2) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        if (dstIdx != srcIdx)
            mov(Xbyak_aarch64::ZRegD(dstIdx), Xbyak_aarch64::ZRegD(srcIdx));
        fmaxnm(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src2);
    }

    void uni_fmaxnm(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src,
            const Xbyak_aarch64::VReg4S &src2);

    template <typename T>
    void uni_fmin(const T &dst, const T &src, const T &src2) {
        uint32_t dstIdx = dst.getIdx();
        uint32_t srcIdx = src.getIdx();
        if (dstIdx != srcIdx)
            mov(Xbyak_aarch64::ZRegD(dstIdx), Xbyak_aarch64::ZRegD(srcIdx));
        fmin(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src2);
    }

    void uni_fmin(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::VReg4S &src,
            const Xbyak_aarch64::VReg4S &src2);

    template <typename T>
    void uni_fmul(const T &dst, const T &src, const T &src2) {
        fmul(dst, src, src2);
    }

    void contiguous_load_unsigned_words(const Xbyak_aarch64::ZReg &src_vmm,
            const Xbyak_aarch64::XReg &addr);

    void contiguous_load_unsigned_words(const Xbyak_aarch64::VReg &src_vmm,
            const Xbyak_aarch64::XReg &addr);

    void uni_frinti(
            const Xbyak_aarch64::VReg4S &d, const Xbyak_aarch64::VReg4S &s);

    void uni_frinti(
            const Xbyak_aarch64::ZRegS &d, const Xbyak_aarch64::ZRegS &s);

    template <typename T>
    void uni_fsqrt(const T &dst, const T &src) {
        fsqrt(dst, P_ALL_ONE / Xbyak_aarch64::T_m, src);
    }

    void uni_fsqrt(
            const Xbyak_aarch64::VReg4S &dst, const Xbyak_aarch64::VReg4S &src);

    void uni_fsub(const Xbyak_aarch64::VReg4S &v1,
            const Xbyak_aarch64::VReg4S &v2, const Xbyak_aarch64::VReg4S &v3);

    template <typename T>
    void uni_fsub(const T &dst, const T &src, const T &src2) {
        fsub(dst, src, src2);
    }

    void uni_fsub(const Xbyak_aarch64::ZRegS &z1,
            const Xbyak_aarch64::ZRegS &z2, const Xbyak_aarch64::ZRegS &z3);

    void uni_eor(const Xbyak_aarch64::VReg &v1, const Xbyak_aarch64::VReg &v2,
            const Xbyak_aarch64::VReg &v3);

    void uni_eor(const Xbyak_aarch64::ZReg &z1, const Xbyak_aarch64::ZReg &z2,
            const Xbyak_aarch64::ZReg &z3);

    void uni_ld1rw(const Xbyak_aarch64::VReg4S &dst,
            const Xbyak_aarch64::XReg &base, const int64_t off);

    void uni_ld1rw(const Xbyak_aarch64::ZRegS &dst,
            const Xbyak_aarch64::XReg &base, const int64_t off);

    void uni_ldr(
            const Xbyak_aarch64::VReg &dst, const Xbyak_aarch64::XReg &addr);

    void uni_ldr(
            const Xbyak_aarch64::ZReg &dst, const Xbyak_aarch64::XReg &addr);

    void uni_load_vmm_stack(const Xbyak_aarch64::VReg &dst,
            const Xbyak_aarch64::XReg &base, size_t vmm_slot);

    void uni_load_vmm_stack(const Xbyak_aarch64::ZReg &dst,
            const Xbyak_aarch64::XReg &base, size_t vmm_slot);

    template <typename T>
    void uni_ldr(const Xbyak_aarch64::ZReg &r, const Xbyak_aarch64::XReg &base,
            const T off) {
        const int off_mod = off % cpu_sveLen;
        const int off_mul_vl = off / cpu_sveLen;

        if (off_mod == 0 && -256 <= off_mul_vl && off_mul_vl <= 255) {
            ldr(r, Xbyak_aarch64::ptr(base, off_mul_vl, Xbyak_aarch64::MUL_VL));
        } else {
            const int offset = off_mod * 0x10 * (cpu_sveLen / 16);
            add_imm(X_DEFAULT_ADDR, base, offset, X_TMP_0);
            ldr(r, Xbyak_aarch64::ptr(X_DEFAULT_ADDR));
        }
    }

    template <typename T>
    void uni_ldr(const Xbyak_aarch64::VReg &r, const Xbyak_aarch64::XReg &base,
            const T off) {
        const int off_mod = off % 16;
        const int off_mul_vl = off / 16;

        if (off_mod == 0 && 0 <= off_mul_vl && off_mul_vl <= 65520) {
            ldr(Xbyak_aarch64::QReg(r.getIdx()), Xbyak_aarch64::ptr(base, off));
        } else {
            add_imm(X_DEFAULT_ADDR, base, off_mod * 4, X_TMP_0);
            ldr(Xbyak_aarch64::QReg(r.getIdx()),
                    Xbyak_aarch64::ptr(X_DEFAULT_ADDR));
        }
    }

    void uni_mov(const Xbyak_aarch64::VReg &d, const Xbyak_aarch64::VReg &s);

    void uni_mov(const Xbyak_aarch64::ZReg &d, const Xbyak_aarch64::ZReg &s);

    void uni_dup(const Xbyak_aarch64::VReg4S &dst, int32_t value);

    void uni_dup(const Xbyak_aarch64::ZRegS &dst, int32_t value);

    void uni_orr(const Xbyak_aarch64::VReg &d, const Xbyak_aarch64::VReg &s0,
            const Xbyak_aarch64::VReg &s1);

    void uni_orr(const Xbyak_aarch64::ZReg &d, const Xbyak_aarch64::ZReg &s0,
            const Xbyak_aarch64::ZReg &s1);

    void uni_scvtf(
            const Xbyak_aarch64::VReg4S &t0, const Xbyak_aarch64::VReg4S &t1);

    void uni_scvtf(
            const Xbyak_aarch64::ZRegS &t0, const Xbyak_aarch64::ZRegS &t1);

    void uni_str(
            const Xbyak_aarch64::VReg &src, const Xbyak_aarch64::XReg &addr);

    void uni_str(
            const Xbyak_aarch64::ZReg &src, const Xbyak_aarch64::XReg &addr);

    void uni_store_vmm_stack(const Xbyak_aarch64::VReg &src,
            const Xbyak_aarch64::XReg &base, size_t vmm_slot);

    void uni_store_vmm_stack(const Xbyak_aarch64::ZReg &src,
            const Xbyak_aarch64::XReg &base, size_t vmm_slot);

    template <typename Vmm>
    void uni_reserve_vmm_stack(size_t count) {
        uni_reserve_vmm_stack(count, Vmm(0));
    }

    template <typename Vmm>
    void uni_release_vmm_stack(size_t count) {
        uni_release_vmm_stack(count, Vmm(0));
    }

    template <typename T>
    void uni_sub(const T &x1, const T &x2, const T &op) {
        sub(x1, x2, op);
    }

    /*
      Saturation facility functions. enable to prepare the register
      holding the saturation upperbound and apply the saturation on
      the floating point register
     */
    template <typename Vmm>
    void init_vmm(Vmm vmm, Xbyak_aarch64::XReg reg_tmp, float value) {
        using namespace data_type;
        bool isSVE = get_sve_length() ? true : false;
        Xbyak_aarch64::ZRegS z_tmp(vmm.getIdx());
        Xbyak_aarch64::VReg4S v_tmp(vmm.getIdx());
        Xbyak_aarch64::WReg w_tmp(reg_tmp.getIdx());
        mov_imm(w_tmp, float2int(value));
        if (isSVE) /* SVE is available. */
            dup(z_tmp, w_tmp);
        else
            dup(v_tmp, w_tmp);
    }

    // Roughly equivalent to:
    // for(index = max - 1; index >= 0; --index) {
    //   body();
    // }
    template <typename TReg, typename Func>
    void asm_for(const TReg &index_reg, const TReg &max_reg, const Func &body) {
        Xbyak_aarch64::Label loop_begin, loop_end;

        // check for early exit
        cmp(max_reg, 0);
        ble(loop_end);

        sub(index_reg, max_reg, 1);
        L(loop_begin);

        body();

        subs(index_reg, index_reg, 1);
        bge(loop_begin);

        L(loop_end);
    }

    // Roughly equivalent to:
    // for(index = max - step; index >= 0; index -= step) {
    //   body();
    // }
    template <typename TReg, typename Func>
    void asm_for_step(const TReg &index_reg, const TReg &max_reg, int64_t step,
            const Func &body) {
        Xbyak_aarch64::Label loop_begin, loop_end;

        cmp_imm(max_reg, step, X_TMP_0);
        blt(loop_end);

        sub_imm(index_reg, max_reg, step, X_TMP_0);

        L(loop_begin);

        body();

        subs(index_reg, index_reg, step);
        bge(loop_begin);

        L(loop_end);
    }

    // Only providing a 64-bit version so we don't have to worry about bad
    // immediate ranges, and signed vs unsigned concerns
    // (e.g, Using a WReg but providing max_imm > INT32_MAX).
    template <typename Func>
    void asm_for(const Xbyak_aarch64::XReg &index_reg, int64_t max_imm,
            const Func &body) {
        Xbyak_aarch64::Label loop_begin;

        // check for early exit
        if (max_imm <= 0) { return; }

        mov_imm(index_reg, max_imm - 1);
        L(loop_begin);

        body();

        subs(index_reg, index_reg, 1);
        bge(loop_begin);
    }

    // Roughly equivalent to:
    // do {
    //   body();
    // } while (--n > 0)
    template <typename TReg, typename Func>
    void asm_do_while(const TReg &index_reg, const Func &body) {
        Xbyak_aarch64::Label loop_begin;

        L(loop_begin);

        body();

        subs(index_reg, index_reg, 1);
        bgt(loop_begin);
    }

    template <typename Vmm>
    void init_saturate_f32(Vmm vmm_lbound, Vmm vmm_ubound,
            Xbyak_aarch64::XReg reg_tmp, data_type_t idt, data_type_t odt,
            bool force_lbound = false) {
        using namespace data_type;
        bool isSVE = get_sve_length() ? true : false;

        if (!((idt == f32) && utils::one_of(odt, u8, data_type::s8, s32)))
            return;

        assert(IMPLICATION(
                idt == u8, vmm_lbound.getIdx() != vmm_ubound.getIdx()));
        // No need to saturate on lower bound for signed integer types, as
        // the conversion to int would return INT_MIN, and then proper
        // saturation will happen in store_data. The param force_lbound, will
        // force saturate values unconditionally to lbound.
        if (odt == u8) {
            if (isSVE) /* SVE is available. */
                dup(Xbyak_aarch64::ZRegS(vmm_lbound.getIdx()), 0);
            else if (mayiuse(asimd))
                movi(Xbyak_aarch64::VReg4S(vmm_lbound.getIdx()), 0);
            else
                assert(!"unreachable");
        } else if (force_lbound) {
            const float saturation_lbound
                    = odt == data_type::s8 ? INT8_MIN : INT32_MIN;
            init_vmm(vmm_lbound, reg_tmp, saturation_lbound);
        }

        float saturation_ubound = types::max_value<float>(odt);
        init_vmm(vmm_ubound, reg_tmp, saturation_ubound);
    }

    template <typename Vmm>
    void saturate_f32(const Vmm &vmm, const Vmm &vmm_lbound,
            const Vmm &vmm_ubound, data_type_t odt,
            const Xbyak_aarch64::PReg &p_true, bool force_lbound = false) {
        // This function is used to saturate to odt in f32 before converting
        // to s32 in order to avoid bad saturation due to cvtps2dq
        // behavior (it returns INT_MIN if the f32 is out of the
        // s32 range)
        using namespace data_type;
        bool isSVE = get_sve_length() ? true : false;

        if (!utils::one_of(odt, u8, data_type::s8, s32)) return;

        Xbyak_aarch64::VReg4S v_tmp(vmm.getIdx());
        Xbyak_aarch64::VReg4S v_lbound(vmm_lbound.getIdx());
        Xbyak_aarch64::VReg4S v_ubound(vmm_ubound.getIdx());
        Xbyak_aarch64::ZRegS z_tmp(vmm.getIdx());
        Xbyak_aarch64::ZRegS z_lbound(vmm_lbound.getIdx());
        Xbyak_aarch64::ZRegS z_ubound(vmm_ubound.getIdx());

        // no need to apply lower saturation bound when odt is
        // signed, as cvtps2dq will return MIN_INT if the value
        // does not fit. The param force_lbound, will force saturate values
        // unconditionally to lbound.
        if (odt == u8 || force_lbound) {
            if (isSVE) /* SVE is available. */
                fmax(z_tmp, p_true / Xbyak_aarch64::T_m, z_lbound);
            else if (mayiuse(asimd))
                fmax(v_tmp, v_tmp, v_lbound);
            else
                assert(!"unreachable");
        }
        if (isSVE) /* SVE is available. */
            fmin(z_tmp, p_true / Xbyak_aarch64::T_m, z_ubound);
        else if (mayiuse(asimd))
            fmin(v_tmp, v_tmp, v_ubound);
        else
            assert(!"unreachable");
    }

    /* A utility function to process f32 tail (load, store or other) depending
     * on tail size, stored in Reg64. Tail size must be value from 0 to 3/7
     * (Xmm/Ymm). Tail process functions require integer as argument to specify
     * behavior for each tail size.
     *
     * Only supported for Xmm and Ymm.
     */
    template <cpu_isa_t isa>
    void runtime_tail_process(const Xbyak_aarch64::XReg &reg_tail,
            const Xbyak_aarch64::XReg &reg_tmp,
            const std::function<void(int)> &tail_process) {
        const size_t simd_w = simd_elems(data_type_t::dnnl_f32, isa);

        Xbyak_aarch64::Label label_tbl, label_tbl_end;
        std::vector<Xbyak_aarch64::Label> l_case(simd_w);

        adr(reg_tmp, label_tbl);
        mov_imm(X_TMP_0, sizeof(void *));
        madd(X_DEFAULT_ADDR, reg_tail, X_TMP_0, reg_tmp);
        br(X_DEFAULT_ADDR);

        // create jump table
        L(label_tbl);
        for (size_t i = 0; i < simd_w; i++)
            putL(l_case[i]);

        // cases for each tail size - from 0 to 3/7
        L(l_case[0]);
        b(label_tbl_end);
        for (size_t i = 1; i < simd_w; i++) {
            L(l_case[i]);
            tail_process(i);
            b(label_tbl_end);
        }
        L(label_tbl_end);
    }

    void jit_breakpoint();

    DNNL_DISALLOW_COPY_AND_ASSIGN(jit_generator_t);

    jit_generator_t(void *code_ptr = nullptr, size_t code_size = MAX_CODE_SIZE,
            bool use_autogrow = true, cpu_isa_t max_cpu_isa = isa_all)
        : Xbyak_aarch64::CodeGenerator(code_size,
                  (code_ptr == nullptr && use_autogrow)
                          ? Xbyak_aarch64::AutoGrow
                          : code_ptr)
        , max_cpu_isa_(max_cpu_isa) {}
    ~jit_generator_t() override = default;

    virtual const char *name() const = 0;
    virtual const char *source_file() const = 0;

    void register_jit_code(const uint8_t *code, size_t code_size) const {
        jit_utils::register_jit_code(code, code_size, name(), source_file());
    }

    const uint8_t *jit_ker() const { return jit_ker_; }

    template <typename... kernel_args_t>
    void operator()(kernel_args_t... args) const {
        using jit_kernel_func_t = void (*)(const kernel_args_t... args);
        auto *fptr = (jit_kernel_func_t)jit_ker_;
        (*fptr)(std::forward<kernel_args_t>(args)...);
    }

    virtual status_t create_kernel() {
        generate();
        jit_ker_ = getCode();
        return (jit_ker_) ? status::success : status::runtime_error;
    }

private:
    void uni_reserve_vmm_stack(size_t count, const Xbyak_aarch64::VReg &);
    void uni_reserve_vmm_stack(size_t count, const Xbyak_aarch64::ZReg &);
    void uni_release_vmm_stack(size_t count, const Xbyak_aarch64::VReg &);
    void uni_release_vmm_stack(size_t count, const Xbyak_aarch64::ZReg &);

    template <typename RegT>
    using scoped_reg_t = Xbyak_aarch64::RegPoolManager::Scoped<RegT>;

    template <typename RegT, size_t... Is>
    std::array<scoped_reg_t<RegT>, sizeof...(Is)> alloc_regs_impl(
            nstl::index_sequence<Is...>) {
        return {{(static_cast<void>(Is), alloc_reg<RegT>())...}};
    }

    void init_reg_pool() {
        assert(!reg_pool_);
        reg_pool_.reset(new Xbyak_aarch64::RegPoolManager());

        // Reserve the ABI parameter and precomputed predicate constants. Note
        // that we do not reserve any of the other jit_generator temporaries, if
        // you are actively using the reg pool, you must use it for all
        // temporaries.
        reg_pool_->reserve_gp(abi_param1.getIdx());
        for (const auto &reg : {P_ALL_ONE, P_ALL_ZERO, P_NOT_256, P_NOT_128})
            reg_pool_->reserve_pred(reg.getIdx());
    }

    template <typename RegT>
    bool reg_is_trackable(const RegT &reg) {
        const auto contains_idx = [&](const std::vector<int> &idxs) {
            return std::find(idxs.begin(), idxs.end(), reg.getIdx())
                    != idxs.end();
        };
        switch (Xbyak_aarch64::reg_family<RegT>::value) {
            case Xbyak_aarch64::RegFamily::gp:
                return contains_idx(reg_pool().get_all_free_gp())
                        || contains_idx(reg_pool().get_all_preserved_gp());
            case Xbyak_aarch64::RegFamily::vec:
                return contains_idx(reg_pool().get_all_free_vec())
                        || contains_idx(reg_pool().get_all_preserved_vec());
            case Xbyak_aarch64::RegFamily::pred:
                return contains_idx(reg_pool().get_all_free_pred())
                        || contains_idx(reg_pool().get_all_preserved_pred());
            default: return false;
        }
    }

    const cpu_isa_t max_cpu_isa_;
    std::unique_ptr<Xbyak_aarch64::RegPoolManager> reg_pool_;
    const uint8_t *getCode() {
        ready();
        if (!is_initialized()) return nullptr;

        const uint8_t *code = CodeGenerator::getCode();
        const size_t code_size_in_bytes = getSize();

        register_jit_code(code, code_size_in_bytes);
        return code;
    }

    inline bool is_valid_isa(cpu_isa_t isa) {
        return is_subset(isa, max_cpu_isa_) && mayiuse(isa);
    }

    static inline bool is_initialized() {
        /* At the moment, Xbyak_aarch64 does not have GetError()\
         so that return dummy result. */
        return true;
    }

protected:
    Xbyak_aarch64::RegPoolManager &reg_pool() {
        if (!reg_pool_) init_reg_pool();
        return *reg_pool_;
    }

    // Allocate pool-selected scratch for this lexical scope.
    template <typename RegT>
    scoped_reg_t<RegT> alloc_reg() {
        return reg_pool().allocScoped<RegT>();
    }

    template <typename RegT, size_t N>
    std::array<scoped_reg_t<RegT>, N> alloc_regs() {
        return alloc_regs_impl<RegT>(nstl::make_index_sequence<N> {});
    }

    // Claim caller-selected live state so it cannot be chosen as scratch.
    template <typename RegT>
    scoped_reg_t<RegT> claim_reg(const RegT &reg) {
        return reg_pool().allocScoped(reg);
    }

    template <typename RegT>
    std::vector<scoped_reg_t<RegT>> claim_regs(
            std::initializer_list<RegT> regs) {
        std::vector<scoped_reg_t<RegT>> claimed_regs;
        std::set<int> claimed_idxs;
        claimed_regs.reserve(regs.size());
        for (const auto &reg : regs) {
            if (!claimed_idxs.insert(reg.getIdx()).second) continue;
            if (reg_pool().reg_in_use(reg)) continue;
            if (reg_is_trackable(reg))
                claimed_regs.emplace_back(reg_pool().allocScoped(reg));
        }
        return claimed_regs;
    }

    template <typename RegT>
    std::vector<scoped_reg_t<RegT>> claim_reg_range(
            int first_idx, size_t count) {
        std::vector<scoped_reg_t<RegT>> claimed_regs;
        claimed_regs.reserve(count);
        for (size_t i = 0; i < count; ++i)
            claimed_regs.emplace_back(
                    reg_pool().allocScoped<RegT>(first_idx + i));
        return claimed_regs;
    }

    template <typename RegT, typename IndexContainer>
    std::vector<scoped_reg_t<RegT>> claim_reg_idxs(const IndexContainer &idxs) {
        std::vector<scoped_reg_t<RegT>> claimed_regs;
        claimed_regs.reserve(idxs.size());
        for (const auto idx : idxs)
            claimed_regs.emplace_back(
                    reg_pool().allocScoped<RegT>(static_cast<int>(idx)));
        return claimed_regs;
    }

    virtual void generate() = 0;
    const uint8_t *jit_ker_ = nullptr;
};

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
