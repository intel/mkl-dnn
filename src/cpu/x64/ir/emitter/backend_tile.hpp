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

#ifndef CPU_X64_IR_EMITTER_BACKEND_TILE_HPP
#define CPU_X64_IR_EMITTER_BACKEND_TILE_HPP

// Tile backend for the emitter (see `emit()` in `emitter.cpp`).
//
// This backend contains every tile instruction. Tile instructions do not
// depend on the vector ISA family, so the emitter uses this backend next to
// either vector backend. The emitter passes the tile number of each tile vreg,
// and each operation constructs its Xbyak `Tmm` registers from it.

#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

struct tile_backend_t {
    explicit tile_backend_t(jit_generator_t &gen) : gen_(gen) {}

    jit_generator_t &gen() { return gen_; }

    void tzero(int d) { // dst = 0
        gen().tilezero(Xbyak::Tmm(d));
    }

    // Load and store a tile, one row every `stride` bytes. The stride is the
    // index register of the address.
    void tload(int d, int base, dim_t disp, int stride, mem_hint_t hint) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + Xbyak::Reg64(stride)
                + (int)disp];
        if ((hint & mem_hint_t::non_temporal) != mem_hint_t::none)
            gen().tileloaddt1(Xbyak::Tmm(d), addr);
        else
            gen().tileloadd(Xbyak::Tmm(d), addr);
    }

    void tstore(int base, dim_t disp, int s, int stride) {
        const auto addr = gen().ptr[Xbyak::Reg64(base) + Xbyak::Reg64(stride)
                + (int)disp];
        gen().tilestored(addr, Xbyak::Tmm(s));
    }

private:
    // The kernel owns `gen_`.
    jit_generator_t &gen_;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
