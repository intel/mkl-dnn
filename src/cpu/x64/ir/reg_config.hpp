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

#ifndef CPU_X64_IR_REG_CONFIG_HPP
#define CPU_X64_IR_REG_CONFIG_HPP

#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// One physical register is selected before the emitter runs and passed in.
// The register allocator avoids using it, so it is always available:
//  - `param_reg`: stores a pointer to the kernel's input arguments.
//
// `pools` contains the registers that are available for allocation for each
// register kind, along with the stack space needed when a value for that
// kind is spilled.
struct reg_config_t {
    reg_pools_t pools;
    int param_reg = 0;
};

// The tile configuration the kernel runs with. The IR does not configure the
// tiles. The kernel or its caller does.
//
//   shapes - shape of each tile register, indexed by its number. A tile that
//            the configuration leaves unused has the empty shape. A kernel
//            without tiles has no shapes.
struct tile_config_t {
    std::vector<tile_shape_t> shapes;
};

// Creates the register config for the given ISA.
//
// The number of vector registers and their spill-slot size are inferred
// directly from the ISA. For example, AVX2 uses 16 vector registers with
// 32-byte spill slots, while AVX-512 uses 32 vector registers with 64-byte
// spill slots.
//
// There are 16 gprs, plus `r16` to `r31` where the machine has Intel APX
// enabled, except on AVX2*.
//
// A mask allocates from the vector file on AVX2*, where a mask is a vector
// register and spills like one, and from a dedicated k-register file on
// AVX-512, which is not spillable.
//
// A tile allocates from the file of the tile registers that `tiles` configures
// with its shape. There is one such file per distinct shape, and it is not
// spillable.
//
// Some registers are reserved and are not included in the allocatable pools:
// - `rsp_reg` (stack pointer)
// - `param_reg` (parameter pointer)
// - `rbp` (frame pointer) when the build option `DNNL_SAFE_RBP` was specified.
// - `reserved_masks` opmasks, on AVX-512 only
//
// Spilled values need no reserved register. The allocator gives each operation
// a temp register for every spilled operand (see `temp_reg_t`).
//
// `reserved_masks` names the opmasks a kernel hands to code outside the IR that
// writes them without restoring them. The JIT post-ops injector is the one such
// consumer today (see `postops_injector_t`). On AVX2* a mask is a vector
// register, so `reserved_masks` is ignored there.
//
// Export for testing.
reg_config_t DNNL_API make_reg_config(cpu_isa_t isa, int param_reg, int rsp_reg,
        const std::vector<int> &reserved_masks,
        const tile_config_t &tiles = {});

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
