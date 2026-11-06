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

#ifndef CPU_X64_IR_ELTWISE_INJECTOR_HPP
#define CPU_X64_IR_ELTWISE_INJECTOR_HPP

// Driver for the JIT-based eltwise injector. It lowers a unary elementwise IR
// operation applied in place to a vec register, e.g. `vexp` (see `emitter.hpp`
// and `exp_fn_t`).
//
// The IR operation and the emitter callback that drives this
// `eltwise_injector_t` are builder-independent. An IR-based kernel sets it up
// in `generate()`:
//   1. Create this object from the elementwise algorithm.
//   2. Wrap it in a callback passed to the emitter that calls `apply()`.
//   3. Call `prepare_table()` once after the postamble.
//
// It parallels `postops_injector_t`; the difference is that this one applies a
// single unary algorithm in place to one register mid-stream, rather than a
// post-ops chain to a set of accumulators at store time.
//
// ISA handling:
// - The only ISA-specific detail is the vector width (Ymm or Zmm), chosen from
//   the `isa`. The width-specific injector is stored type-erased, so IR-based
//   kernels do not need to be templated.
//
// Register handling:
// - The injector saves and restores every register it uses (aux vector and gpr
//   registers and its table pointer), so its registers need not be reserved in
//   the IR allocator. The save/restore cost is paid once per call.

#include <memory>

#include "common/c_types_map.hpp"
#include "cpu/x64/cpu_isa_traits.hpp"
#include "cpu/x64/jit_generator.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

// Interfaces marked with DNNL_API are exported for testing purposes only.
struct eltwise_injector_t {
    // gen   - generator the injected code is emitted into
    // isa   - kernel ISA (selects the vector width)
    // alg   - elementwise algorithm to apply (e.g. eltwise_exp)
    // alpha, beta, scale - algorithm parameters (unused by parameterless algs)
    DNNL_API eltwise_injector_t(jit_generator_t &gen, cpu_isa_t isa,
            alg_kind_t alg, float alpha, float beta, float scale);

    eltwise_injector_t(const eltwise_injector_t &) = delete;
    eltwise_injector_t &operator=(const eltwise_injector_t &) = delete;

    // Apply the algorithm in place to the vec register at `vec_phys` (a
    // physical vec register index).
    void DNNL_API apply(int vec_phys);

    // Emit the constant table. Call once, after the postamble.
    void DNNL_API prepare_table();

private:
    // Type-erased `jit_uni_eltwise_injector_t<Vmm>`. Cast to the target type
    // when used, based on `is_zmm_`.
    std::shared_ptr<void> injector_;
    bool is_zmm_ = false;
};

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
