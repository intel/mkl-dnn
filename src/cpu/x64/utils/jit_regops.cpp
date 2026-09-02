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

#include "common/z_magic.hpp"

#include "cpu/x64/utils/jit_regops.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace regops {

void horizontal_add_ps(
        jit_generator_t *code, Xbyak::Ymm src, Xbyak::Ymm workspace) {
    const Xbyak::Xmm xmm_ws {workspace.getIdx()};
    const Xbyak::Xmm xmm_src {src.getIdx()};

    // vextractf128 is VEX-only and cannot encode ymm16-31. Use the EVEX
    // vextractf32x4 for high registers.
    const bool requires_evex = src.getIdx() >= 16 || workspace.getIdx() >= 16;
    if (requires_evex)
        code->vextractf32x4(xmm_ws, src, 1);
    else
        code->vextractf128(xmm_ws, src, 1);

    code->vaddps(xmm_src, xmm_src, xmm_ws);

    // XMM indices 16-31 can only be encoded with EVEX. vhaddps/haddps have no
    // EVEX form so for high registers we have to use an EVEX-encodable
    // vshufps + vaddps reduction.
    if (requires_evex) {
        assert(code->is_valid_isa(avx512_core));
        code->vshufps(xmm_ws, xmm_src, xmm_src, 0xB1); // [1,0,3,2]
        code->vaddps(xmm_src, xmm_src, xmm_ws);
        code->vshufps(xmm_ws, xmm_src, xmm_src, 0x4E); // [2,3,0,1]
        code->vaddps(xmm_src, xmm_src, xmm_ws);
    } else {
        assert(code->is_valid_isa(avx2));
        code->vhaddps(xmm_src, xmm_src, xmm_src);
        code->vhaddps(xmm_src, xmm_src, xmm_src);
    }
}

void horizontal_add_ps(
        jit_generator_t *code, Xbyak::Zmm src, Xbyak::Zmm workspace) {
    const Xbyak::Ymm ymm_ws {workspace.getIdx()};
    const Xbyak::Ymm ymm_src {src.getIdx()};

    code->vextractf64x4(ymm_ws, src, 1);
    code->vaddps(ymm_src, ymm_src, ymm_ws);
    horizontal_add_ps(code, ymm_src, ymm_ws);
}

void horizontal_max_ps(
        jit_generator_t *code, Xbyak::Xmm src, Xbyak::Xmm workspace) {
    // There is no horizontal-max instruction, so reduce with shuffle + vmaxps.
    // vshufps has both VEX and EVEX encodings, so one path covers all registers,
    // but ymm16-31 still need AVX-512.
    const bool requires_evex = src.getIdx() >= 16 || workspace.getIdx() >= 16;
    if (requires_evex) assert(code->is_valid_isa(avx512_core));

    code->vshufps(workspace, src, src, 0xB1); // [1,0,3,2]
    code->vmaxps(src, src, workspace);
    code->vshufps(workspace, src, src, 0x4E); // [2,3,0,1]
    code->vmaxps(src, src, workspace);
}

void horizontal_max_ps(
        jit_generator_t *code, Xbyak::Ymm src, Xbyak::Ymm workspace) {
    const Xbyak::Xmm xmm_ws {workspace.getIdx()};
    const Xbyak::Xmm xmm_src {src.getIdx()};

    // vextractf128 is VEX-only and cannot encode ymm16-31. Use the EVEX
    // vextractf32x4 for high registers.
    const bool requires_evex = src.getIdx() >= 16 || workspace.getIdx() >= 16;
    if (requires_evex)
        code->vextractf32x4(xmm_ws, src, 1);
    else
        code->vextractf128(xmm_ws, src, 1);

    code->vmaxps(xmm_src, xmm_src, xmm_ws);
    horizontal_max_ps(code, xmm_src, xmm_ws);
}

void horizontal_max_ps(
        jit_generator_t *code, Xbyak::Zmm src, Xbyak::Zmm workspace) {
    const Xbyak::Ymm ymm_ws {workspace.getIdx()};
    const Xbyak::Ymm ymm_src {src.getIdx()};

    code->vextractf64x4(ymm_ws, src, 1);
    code->vmaxps(ymm_src, ymm_src, ymm_ws);
    horizontal_max_ps(code, ymm_src, ymm_ws);
}

} // namespace regops
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
