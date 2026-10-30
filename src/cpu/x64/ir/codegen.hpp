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

#ifndef CPU_X64_IR_CODEGEN_HPP
#define CPU_X64_IR_CODEGEN_HPP

#include "common/broadcast_strategy.hpp"
#include "common/c_types_map.hpp"
#include "cpu/x64/injectors/jit_uni_binary_injector.hpp"
#include "cpu/x64/ir/ir.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// The kernel-specific inputs of the post-ops injector (see
// `postops_injector_t`). `generate_kernel()` supplies the rest.
//
//   post_ops       - attribute post-ops chain to apply, or null when the IR has
//                    no `inject_postops` operation
//   dst_md         - destination memory descriptor (used by binary post-ops)
//   bcast_set      - broadcasting strategies the binary injector may pick for
//                    a right-hand-side argument. Defaults to every strategy
//                    the injector supports
//   rhs_arg_offset - byte offset in the kernel argument struct of the binary
//                    right-hand-side argument pointer array
//   dst_orig_off   - byte offset in the kernel argument struct of the
//                    destination origin pointer
struct postops_config_t {
    const post_ops_t *post_ops = nullptr;
    const memory_desc_t *dst_md = nullptr;
    bcast_set_t bcast_set
            = binary_injector::get_all_strategies_supported_by_injector();
    int rhs_arg_offset = 0;
    dim_t dst_orig_off = 0;
};

// Generates the complete kernel for `ir` into `gen`.
//
// The stages run in this order:
//   1. register configuration
//   2. register allocation
//   3. ABI preamble and spill-frame reservation
//   4. lowering (see `emit()`)
//   5. spill-frame teardown and ABI postamble
//   6. static data, then the post-ops constant table
//
// The register configuration, emitter, and post-ops injector all take the ISA
// from `gen.max_cpu_isa()`. The kernel argument pointer is `abi_param1`.
//
// The post-ops injector is created only when `postops.post_ops` is set.
void generate_kernel(jit_generator_t &gen, const ir_t &ir,
        const postops_config_t &postops = {});

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
