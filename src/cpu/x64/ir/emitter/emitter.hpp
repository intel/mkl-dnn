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

#ifndef CPU_X64_IR_EMITTER_EMITTER_HPP
#define CPU_X64_IR_EMITTER_EMITTER_HPP

// Lowers an IR along with its register allocation into target machine code,
// using a `jit_generator`.

#include <deque>
#include <utility>
#include <vector>

#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/ir/reg_alloc.hpp"
#include "cpu/x64/ir/reg_config.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Static data the emitter returns to be used after the kernel code
// (after postamble). For example: AVX2 mask tables.
//
// A `std::deque` keeps element addresses stable. A label is referenced while
// the emitter goes through the IR and is only bound later by
// `emit_data_section`, so the addresses must stay valid as more constants are
// appended.
//
// `begin_offset` is the code size right after the postamble, where the static
// data starts. `emit_data_section()` sets it. The debug output counts
// everything the kernel writes from that point on as static data.
// It is the one field that exists only for the debug output (see Design
// Principles in `ir/README.md`).
struct data_section_t {
    std::deque<std::pair<std::vector<unsigned char>, Xbyak::Label>> constants;
    static constexpr int alignment = 32;
    size_t begin_offset = 0;
};

// Lowering/code emission pass. Walks the IR once and turns each abstract
// operation into target-specific instructions, using the physical registers
// chosen by the allocator.
//
// Values that cannot stay in a register are spilled on the stack. An operation
// that reads or writes one gets a temp register for it from the allocator (see
// `temp_reg_t`). Spilled inputs are loaded into their temps before use. Results
// are stored back on the stack after use.
//
// Supporting another target ISA mainly requires providing a similar backend
// (see `emitter/backend_avx2.hpp`) or extending an existing one, then adding a
// dispatch branch in `emit()`.
//
// `data` collects static constants the lowering needs (e.g. mask tables). The
// emitter appends to it and references each entry rip-relative. The caller
// then emits the bytes with `emit_data_section()` after the `postamble()`.
//
// Lowers the `inject_postops` operation onto the JIT post-ops injector, which
// is not IR-based (see `postops_injector.hpp`). Only `emit()` needs it, so a
// forward declaration keeps this header free of post-ops specifics.
struct postops_injector_t;

// `eltwise_fn` lowers the `veltwise` operation to the JIT eltwise injector.
// Like `inject`, it is the interoperability layer between the IR and the non-IR
// injector: given the eltwise algorithm and the physical vec register index
// holding the value, it applies the algorithm in place. The injector preserves
// the registers it borrows, so it stays outside the IR allocator. A single
// callback serves every eltwise algorithm, so supporting a new one (tanh,
// gelu, ...) needs no change here. When the IR has no `veltwise` op it is
// unused and may be left empty.
using eltwise_fn_t = std::function<void(alg_kind_t alg, int vec_phys)>;

// This is the main entry point for the emitter. It dispatches by ISA family to
// the matching backend.
//
// `postops` is the interoperability layer between the IR and the non-IR
// injector, so everything specific to hooking Xbyak code up to the IR lives
// there, including the fixed registers the injector reads. Those are
// ISA-specific (an AVX-512 tail opmask has no AVX2 counterpart), which is why
// neither the IR builder nor the emitter names them. It is null when the kernel
// has no post-ops, which is also when the IR has no `inject_postops` operation.
//
// Export for testing.
void DNNL_API emit(jit_generator_t &gen, const ir_t &ir,
        const reg_alloc_result_t &alloc, const reg_config_t &reg_cfg,
        data_section_t &data, postops_injector_t *postops = nullptr,
        const eltwise_fn_t &eltwise_fn = {});

// Emit the accumulated static data after the kernel's postamble. It aligns,
// binds each label and then write the data bytes with `db`.
// Must be called once, after the emitter and the postamble.
void DNNL_API emit_data_section(jit_generator_t &gen, data_section_t &data);

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
