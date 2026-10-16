/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#ifndef CPU_X64_IR_EMITTER_BACKEND_AVX512_HPP
#define CPU_X64_IR_EMITTER_BACKEND_AVX512_HPP

// AVX-512 backend for the emitter (see `emit()` in `emitter.cpp`).
//
// This backend contains every vector and mask instruction for one ISA family
// and covers all AVX-512 extensions (avx512_core, avx512_core_bf16,
// avx512_core_fp16, and so on). The generic emitter iterates over the IR and
// resolves each virtual register to a physical register index. The backend is
// the only code on this path that constructs Xbyak `Zmm` and `Opmask`
// registers, so each operation builds its own registers from the indices it is
// given.
//
// A mask is a k-register, and a masked access is an EVEX write mask on the
// access instruction itself rather than a separate instruction. A mask
// therefore allocates from a register file of its own (see
// `make_reg_config()`).

#include <cassert>

#include "common/c_types_map.hpp"
#include "cpu/x64/ir/emitter/emitter.hpp"
#include "cpu/x64/jit_generator.hpp"
#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

struct avx512_backend_t {
    avx512_backend_t(jit_generator_t &gen, cpu_isa_t isa)
        : gen_(gen), isa(isa) {}

    jit_generator_t &gen() { return gen_; }

    // Plain vector ops.
    void vzero(int d) { // dst = 0
        gen().vpxord(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(d));
    }

    // Move a whole register to or from memory as it is. The emitter uses this
    // for spill slots, where the bytes come back exactly as they went out, so
    // there is no data type and never a conversion.
    void vload_raw(int d, int base, dim_t disp) { // dst = [base + disp]
        gen().vmovups(Xbyak::Zmm(d), gen().ptr[Xbyak::Reg64(base) + (int)disp]);
    }

    void vstore_raw(int base, dim_t disp, int s) { // [base + disp] = src
        gen().vmovups(gen().ptr[Xbyak::Reg64(base) + (int)disp], Xbyak::Zmm(s));
    }

    // Load a full vector. The pair of data types selects the form:
    //   f32  -> f32   plain move
    //   s32  -> s32   plain move
    //   bf16 -> bf16  plain move
    //   bf16 -> f32   half a register read and widened
    //   f16  -> f32   half a register read and widened
    void vload(int d, int base, dim_t disp, data_type_t mem_dt,
            data_type_t reg_dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];

        if (mem_dt == reg_dt
                && utils::one_of(reg_dt, data_type::f32, data_type::s32,
                        data_type::bf16)) {
            gen().vmovups(Xbyak::Zmm(d), addr);
        } else if (mem_dt == data_type::bf16 && reg_dt == data_type::f32) {
            gen().vpmovzxwd(Xbyak::Zmm(d), addr);
            gen().vpslld(Xbyak::Zmm(d), Xbyak::Zmm(d), 16);
        } else if (mem_dt == data_type::f16 && reg_dt == data_type::f32) {
            gen().vcvtph2ps(Xbyak::Zmm(d), addr);
        } else {
            JIT_ASSERT(!"vload: dtype not implemented");
        }
    }

    // Store a full vector.
    void vstore(int base, dim_t disp, int s, data_type_t mem_dt,
            data_type_t reg_dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];
        if (mem_dt == data_type::f32 && reg_dt == data_type::f32)
            gen().vmovups(addr, Xbyak::Zmm(s));
        else { JIT_ASSERT(!"vstore: dtype not implemented"); }
    }

    // Load one element and zero the rest of `dst`.
    void vload_scalar(int d, int base, dim_t disp, data_type_t mem_dt,
            data_type_t reg_dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];
        if (mem_dt == data_type::f32 && reg_dt == data_type::f32)
            gen().vmovss(Xbyak::Xmm(d), addr);
        else { JIT_ASSERT(!"vload_scalar: dtype not implemented"); }
    }

    // Store element 0 of `src`.
    void vstore_scalar(int base, dim_t disp, int s, data_type_t mem_dt,
            data_type_t reg_dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];
        if (mem_dt == data_type::f32 && reg_dt == data_type::f32)
            gen().vmovss(addr, Xbyak::Xmm(s));
        else { JIT_ASSERT(!"vstore_scalar: dtype not implemented"); }
    }

    // Load one element and replicate it across `dst`. A bf16 or f16 element is
    // widened to f32.
    void vload_bcast(int d, int base, dim_t disp, data_type_t mem_dt,
            data_type_t reg_dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];
        if (mem_dt == data_type::f32 && reg_dt == data_type::f32) {
            gen().vbroadcastss(Xbyak::Zmm(d), addr);
        } else if (mem_dt == data_type::bf16 && reg_dt == data_type::f32) {
            gen().vpbroadcastw(Xbyak::Zmm(d), addr);
            gen().vpslld(Xbyak::Zmm(d), Xbyak::Zmm(d), 16);
        } else if (mem_dt == data_type::f16 && reg_dt == data_type::f32) {
            gen().vpbroadcastw(Xbyak::Ymm(d), addr);
            gen().vcvtph2ps(Xbyak::Zmm(d), Xbyak::Ymm(d));
        } else {
            JIT_ASSERT(!"vload_bcast: dtype not implemented");
        }
    }

    // Load `n_elems` elements of `mem_dt` at [base + disp] into the low lanes
    // of `d`, each widened (zero-extended) to `d`'s wider element type `reg_dt`.
    void vload_widen(int d, int base, dim_t disp, int n_elems,
            data_type_t mem_dt, data_type_t reg_dt) {
        if (mem_dt != data_type::u8 || reg_dt != data_type::s32) {
            JIT_ASSERT(!"vload_widen: dtype not implemented");
            return;
        }
        const int simd_w = vlen / (int)types::data_type_size(reg_dt);
        assert(n_elems > 0 && n_elems <= simd_w);
        const Xbyak::Reg64 base_reg(base);
        if (n_elems == simd_w) {
            gen().vpmovzxbd(Xbyak::Zmm(d), gen().ptr[base_reg + (int)disp]);
        } else {
            // Tail: gather `n_elems` bytes into the low xmm, zeroing the rest,
            // so the read never runs past the tail, then widen to dwords.
            gen().load_bytes(Xbyak::Xmm(d), base_reg, (int)disp, n_elems);
            gen().vpmovzxbd(Xbyak::Zmm(d), Xbyak::Xmm(d));
        }
    }

    void vadd(int d, int s, data_type_t dt) { // dst += s0
        if (dt == data_type::f32)
            gen().vaddps(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(s));
        else { JIT_ASSERT(!"vadd: dtype not implemented"); }
    }

    void vsub(int d, int s, data_type_t dt) { // dst -= s0
        if (dt == data_type::f32)
            gen().vsubps(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(s));
        else { JIT_ASSERT(!"vsub: dtype not implemented"); }
    }

    void vmul(int d, int s, data_type_t dt) { // dst *= s0
        if (dt == data_type::f32)
            gen().vmulps(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(s));
        else { JIT_ASSERT(!"vmul: dtype not implemented"); }
    }

    void vdiv(int d, int s, data_type_t dt) { // dst /= s0
        if (dt == data_type::f32)
            gen().vdivps(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(s));
        else { JIT_ASSERT(!"vdiv: dtype not implemented"); }
    }

    void vmax(int d, int s, data_type_t dt) { // dst = max(dst, s0)
        if (dt == data_type::f32)
            gen().vmaxps(Xbyak::Zmm(d), Xbyak::Zmm(d), Xbyak::Zmm(s));
        else { JIT_ASSERT(!"vmax: dtype not implemented"); }
    }

    // Build the predicate mask for `s != 0`. On AVX-512 a mask is an opmask,
    // one bit per lane, set where the lane is nonzero.
    void vcmp_ne_zero(int d, int s, data_type_t dt) {
        if (dt == data_type::s32) {
            gen().vptestmd(Xbyak::Opmask(d), Xbyak::Zmm(s), Xbyak::Zmm(s));
        } else {
            JIT_ASSERT(!"vcmp_ne_zero: dtype not implemented");
        }
    }

    // dst = mask ? s0 : dst. `mask` is a per-lane predicate opmask.
    void vblend(int d, int s, int mask, data_type_t dt) {
        if (dt == data_type::f32) {
            JIT_ASSERT(mask >= 1 && mask <= 7);
            gen().vblendmps(Xbyak::Zmm(d) | Xbyak::Opmask(mask), Xbyak::Zmm(d),
                    Xbyak::Zmm(s));
        } else {
            JIT_ASSERT(!"vblend: dtype not implemented");
        }
    }

    // dst = broadcast of s0's element 0 across all lanes.
    void vbcast(int d, int s, data_type_t dt) {
        if (dt == data_type::f32)
            gen().vbroadcastss(Xbyak::Zmm(d), Xbyak::Xmm(s));
        else { JIT_ASSERT(!"vbcast: dtype not implemented"); }
    }

    // Broadcast a constant written into the data section.
    void vbcast_imm(int d, uint32_t bits, data_section_t &data) {
        std::vector<unsigned char> bytes(sizeof(bits));
        std::memcpy(bytes.data(), &bits, sizeof(bits));
        data.constants.emplace_back(std::move(bytes), Xbyak::Label());
        Xbyak::Label &lbl = data.constants.back().second;
        gen().vbroadcastss(Xbyak::Zmm(d), gen().ptr[gen().rip + lbl]);
    }

    // dst += a * b. The multiplicand dtype `src_dt` selects the instruction.
    // The accumulator (dst) is always f32. An f16 multiplicand, and a bf16 one
    // that is not consumed by `vdpbf16ps`, arrives here as f32, widened by the
    // load (see `vload`).
    void vdot(int d, int a, int b, data_type_t src_dt) {
        if (src_dt == data_type::f32) {
            gen().vfmadd231ps(Xbyak::Zmm(d), Xbyak::Zmm(a), Xbyak::Zmm(b));
        } else if (src_dt == data_type::bf16) {
            JIT_ASSERT(is_superset(isa, avx512_core_bf16)
                    && "vdot: bf16 needs avx512_core_bf16");
            gen().vdpbf16ps(Xbyak::Zmm(d), Xbyak::Zmm(a), Xbyak::Zmm(b));
        } else {
            JIT_ASSERT(!"vdot: dtype not implemented");
        }
    }

    void vhreduce(int d, int ws, data_type_t dt) {
        if (dt == data_type::f32)
            regops::horizontal_add_ps(&gen(), Xbyak::Zmm(d), Xbyak::Zmm(ws));
        else { JIT_ASSERT(!"vhreduce: dtype not implemented"); }
    }

    void vhreduce_max(int d, int ws, data_type_t dt) {
        if (dt == data_type::f32)
            regops::horizontal_max_ps(&gen(), Xbyak::Zmm(d), Xbyak::Zmm(ws));
        else { JIT_ASSERT(!"vhreduce_max: dtype not implemented"); }
    }

    // Masked vector ops.
    //
    // The allocator never spills a k-register (see `make_reg_config()`), so a
    // spilled mask means the kernel needs more masks than the k-register file
    // has. This could be enabled, but none of the existing kernels runs out of
    // k-registers, so it remains disabled as a guardrail during the migration.
    void load_mask_raw(int /*d*/, int /*base*/, dim_t /*disp*/) {
        JIT_ASSERT(!"load_mask_raw: k-registers are not spillable");
    }

    void store_mask_raw(int /*base*/, dim_t /*disp*/, int /*s*/) {
        JIT_ASSERT(!"store_mask_raw: k-registers are not spillable");
    }

    // Create a mask with `n_elems` active elements. The mask is built inside
    // the k-register file: `kxnorq` sets all 64 bits and `kshiftrq` keeps the
    // low `n_elems` of them. The alternative, `kmov` from an immediate in a
    // gpr, would need a gpr the operation does not have. `data` is unused,
    // since nothing is loaded from memory.
    //
    // The number of active bits is `n_elems` for every element size. A widening
    // load preserves the element count, so a narrower element type changes what
    // one bit masks, not how many bits are set (see `vload_masked`).
    void set_mask_imm(int d, int n_elems, data_section_t & /*data*/) {
        assert(n_elems > 0 && n_elems < 64);
        const Xbyak::Opmask k(d);
        gen().kxnorq(k, k, k);
        gen().kshiftrq(k, k, (uint8_t)(64 - n_elems));
    }

    // Load the elements selected by `mask` and zero the rest of `dst` (`T_z`).
    // An inactive element has to read as zero, or a tail iteration would
    // accumulate whatever the register happened to hold into the dot product.
    // The mask counts elements of `mem_dt`, so a converting form masks the
    // narrow read and widens what is left.
    void vload_masked(int d, int base, dim_t disp, int mask, data_type_t mem_dt,
            data_type_t reg_dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];
        const Xbyak::Opmask k(mask);

        if (mem_dt == data_type::f32 && reg_dt == data_type::f32) {
            gen().vmovups(Xbyak::Zmm(d) | k | gen().T_z, addr);
        } else if (mem_dt == data_type::bf16 && reg_dt == data_type::bf16) {
            gen().vmovdqu16(Xbyak::Zmm(d) | k | gen().T_z, addr);
        } else if (mem_dt == data_type::bf16 && reg_dt == data_type::f32) {
            gen().vpmovzxwd(Xbyak::Zmm(d) | k | gen().T_z, addr);
            gen().vpslld(Xbyak::Zmm(d), Xbyak::Zmm(d), 16);
        } else if (mem_dt == data_type::f16 && reg_dt == data_type::f32) {
            gen().vmovdqu16(Xbyak::Ymm(d) | k | gen().T_z, addr);
            gen().vcvtph2ps(Xbyak::Zmm(d), Xbyak::Ymm(d));
        } else {
            JIT_ASSERT(!"vload_masked: dtype not implemented");
        }
    }

    // Store the elements of `src` selected by `mask`. There is no `T_z` here,
    // since masking a store is already a merge into memory.
    void vstore_masked(int base, dim_t disp, int s, int mask,
            data_type_t mem_dt, data_type_t reg_dt) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + (int)disp];

        if (mem_dt == data_type::f32 && reg_dt == data_type::f32) {
            gen().vmovups(addr | Xbyak::Opmask(mask), Xbyak::Zmm(s));
        } else {
            JIT_ASSERT(!"vstore_masked: dtype not implemented");
        }
    }

private:
    // The backend is used during the emitter step, which is called from an
    // IR-based kernel during kernel generation. The kernel owns `gen_`.
    jit_generator_t &gen_;

    // ISA is used to dispatch ISA-specific instructions (e.g. on
    // avx512_core_bf16).
    cpu_isa_t isa;

    // Vector register width in bytes.
    const int vlen = cpu_isa_traits_t<avx512_core>::vlen;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
