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

#ifndef CPU_X64_MATMUL_JIT_INT4_UTILS_HPP
#define CPU_X64_MATMUL_JIT_INT4_UTILS_HPP

#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace matmul {
namespace int4_utils {

// Byte constants used by `expand_to_int8()`: 0x0F in every byte of `vmm_0f`
// and, for s4, 0x08 in every byte of `vmm_8`. `reg_tmp` is clobbered.
template <typename Vmm>
void init_expand_to_int8_consts(jit_generator_t *code, const Vmm &vmm_0f,
        const Vmm &vmm_8, bool is_signed, const Xbyak::Reg64 &reg_tmp) {
    code->mov(reg_tmp.cvt32(), 0x0F);
    code->vpbroadcastb(vmm_0f, reg_tmp.cvt8());
    if (is_signed) {
        code->mov(reg_tmp.cvt32(), 0x08);
        code->vpbroadcastb(vmm_8, reg_tmp.cvt8());
    }
}

// Expands the int4 values at `src` (half a vector register of bytes) into
// int8 values in `vmm_dst`, keeping their order: byte 2j gets the low nibble
// of source byte j and byte 2j + 1 its high nibble. s4 values are sign
// extended. `vmm_tmp` is clobbered. Requires AVX512BW (and AVX512VL for Ymm).
template <typename Vmm>
void expand_to_int8(jit_generator_t *code, const Vmm &vmm_dst,
        const Vmm &vmm_tmp, const Xbyak::Address &src, const Vmm &vmm_0f,
        const Vmm &vmm_8, bool is_signed) {
    code->vpmovzxbw(vmm_dst, src);
    code->vpsllw(vmm_tmp, vmm_dst, 4);
    // (word | word << 4) & 0x0F0F
    code->vpternlogq(vmm_dst, vmm_tmp, vmm_0f, 0xA8);
    if (is_signed) {
        // Sign extension of a 4-bit value: (x ^ 8) - 8.
        code->vpxord(vmm_dst, vmm_dst, vmm_8);
        code->vpsubb(vmm_dst, vmm_dst, vmm_8);
    }
}

} // namespace int4_utils
} // namespace matmul
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
