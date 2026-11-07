/*******************************************************************************
* Copyright 2026 Barcelona Supercomputing Center
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

#ifndef RVJIT_IMPLEMENTATION_RVV_BASE_PROGRAMMING_MODEL_HPP
#define RVJIT_IMPLEMENTATION_RVV_BASE_PROGRAMMING_MODEL_HPP

#include "rvjit/implementation/rv64/isa/buffer/stack.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/const_arithmetic.hpp"
#include "rvjit/implementation/rv64/isa/operation/control_flow/control_flow.hpp"
#include "rvjit/implementation/rv64/isa/operation/transfer/transfer.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rv64/loop/loopgen.hpp"
#include "rvjit/implementation/rv64/stream/iterator/iterator.hpp"
#include "rvjit/implementation/rvv/isa/buffer/architecture.hpp"
#include "rvjit/implementation/rvv/isa/buffer/buffer_pool.hpp"
#include "rvjit/implementation/rvv/isa/operation/memory/memory.hpp"
#include "rvjit/implementation/rvv/programming_model.hpp"
#include "rvjit/implementation/rvv/rvv.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"

namespace rvjit {
namespace rvv {

/// The concrete components a `base_programming_model_t` is built on
struct base_components_t {
    rv64::emitter_t e;
    rv64::const_arithmetic::const_arithmetic_t arith;
    buffer::architecture_t arch;
    buffer::buffer_pool_t pool;
    rv64::stack_t stack;
    rv64::transfer::transfer_t xfer;
    rv64::control_flow::control_flow_t cf;
    rv64::loop::loopgen_t loops;
    rv64::stream::iterator::iterator_t iterators;
    memory::memory_t mem;
    stream::memory_operand::memory_operand_t memops;

    /// Creates the components over `cg` in dependency order
    ///
    /// @param cg The code generator instructions are emitted through
    /// @param regs The registers the pool draws from
    base_components_t(codegen_t &cg, const buffer::architecture_t &regs)
        : e(cg)
        , arith(e)
        , arch(regs)
        , pool(arch)
        , stack(e, pool)
        , xfer(e, arith)
        , cf(e, arith, xfer)
        , loops(e, arith, cf, pool, stack, xfer)
        , iterators(e, arith, xfer, pool)
        , mem(e)
        , memops(e, iterators, mem, pool) {}

    base_components_t(const base_components_t &) = delete;
    base_components_t &operator=(const base_components_t &) = delete;
};

/// An RVV programming model over concrete component instances
///
/// @note The components, built before the model, stay reachable by their
///     usual names, e.g. `stack` to preserve the callee-saved registers
struct base_programming_model_t : public base_components_t,
                                  public programming_model_t {
    /// Creates the components and the model over `cg`
    ///
    /// @param cg The code generator instructions are emitted through
    /// @param model The hardware model
    /// @param regs The registers the pool draws from
    ///
    /// @pre `model` must outlive the programming model
    base_programming_model_t(codegen_t &cg, const vpu_t &model,
            const buffer::architecture_t &regs
            = buffer::architecture_t::default_list())
        : base_components_t(cg, regs)
        , programming_model_t(
                  e, pool, loops, stack, xfer, arith, cf, memops, model) {}
};

} // namespace rvv
} // namespace rvjit

#endif
