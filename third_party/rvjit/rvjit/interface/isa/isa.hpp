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

#ifndef RVJIT_INTERFACE_ISA_ISA_HPP
#define RVJIT_INTERFACE_ISA_ISA_HPP

#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/buffer/stack.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"
#include "rvjit/interface/isa/operation/memory.hpp"
#include "rvjit/interface/isa/operation/transfer.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {

/// Composite interface of the operation and buffer components of an ISA
///
/// @details Implementations aggregate the components behind each accessor
struct isa_t {
    static constexpr const char *NAME = "rvjit::isa";

    virtual ~isa_t() = default;

    /// Arithmetic operation expressions
    virtual operation::arithmetic_t &arithmetic() = 0;

    /// Memory operation expressions
    virtual operation::memory_t &memory() = 0;

    /// Value materialization, persistence, and update expressions
    virtual const operation::transfer_t &transfer() const = 0;

    /// Constant-folded integer arithmetic expressions
    virtual const operation::const_arithmetic_t &const_arithmetic() const = 0;

    /// Branching and looping expressions
    virtual const operation::control_flow_t &control_flow() const = 0;

    /// Buffer allocation expressions
    virtual buffer::buffer_pool_t &buffers() = 0;

    /// Callee-saved preservation and stack slot allocation
    virtual stack_t &stack() = 0;
};

/// Assembles an `isa_t` from arithmetic/memory implementations
///
/// @details `ArithOp`/`MemOp` must be `emitter_t`-constructible
///     `operation::arithmetic_t`/`operation::memory_t` implementers
///
/// @note The pool, stack, transfer, const-arithmetic and control-flow
///     components are caller-owned references
template <typename ArithOp, typename MemOp>
struct isa_compose_t : public isa_t {
    /// Assembles the operations around caller-owned components
    ///
    /// @param e The emitter the arithmetic and memory operations emit through
    /// @param pool The register allocation pool
    /// @param stack The stack allocator
    /// @param xfer The value transfer operations
    /// @param arith The constant arithmetic operations
    /// @param cf The control flow operations
    explicit isa_compose_t(emitter_t e, buffer::buffer_pool_t &pool,
            stack_t &stack, const operation::transfer_t &xfer,
            const operation::const_arithmetic_t &arith,
            const operation::control_flow_t &cf)
        : arithmetic_(e)
        , memory_(e)
        , pool_(pool)
        , stack_(stack)
        , transfer_(xfer)
        , const_arithmetic_(arith)
        , control_flow_(cf) {}

    operation::arithmetic_t &arithmetic() override { return arithmetic_; }
    operation::memory_t &memory() override { return memory_; }
    const operation::transfer_t &transfer() const override { return transfer_; }
    const operation::const_arithmetic_t &const_arithmetic() const override {
        return const_arithmetic_;
    }
    const operation::control_flow_t &control_flow() const override {
        return control_flow_;
    }
    buffer::buffer_pool_t &buffers() override { return pool_; }
    stack_t &stack() override { return stack_; }

protected:
    ArithOp arithmetic_;
    MemOp memory_;
    buffer::buffer_pool_t &pool_;
    stack_t &stack_;
    const operation::transfer_t &transfer_;
    const operation::const_arithmetic_t &const_arithmetic_;
    const operation::control_flow_t &control_flow_;
};

} // namespace rvjit

#endif
