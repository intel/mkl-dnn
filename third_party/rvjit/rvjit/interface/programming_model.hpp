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

#ifndef RVJIT_INTERFACE_PROGRAMMING_MODEL_HPP
#define RVJIT_INTERFACE_PROGRAMMING_MODEL_HPP

#include <array>

#include "rvjit/interface/isa/isa.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/loop/loopgen.hpp"
#include "rvjit/interface/stream/stream.hpp"

namespace rvjit {

using scheduler_t = stream::scheduler_t;

/// Composite interface for the isa, loop and stream components
///
/// @details Implementations aggregate the components behind each accessor
struct programming_model_t {
    static constexpr const char *NAME = "rvjit::programming_model";

    virtual ~programming_model_t() = default;

    /// The operation and buffer components
    virtual isa_t &isa() = 0;

    /// Loop dispatch expressions
    virtual loop::loopgen_t &loop() = 0;

    /// The stream components
    virtual stream::stream_gen_t &streams() = 0;

    /// The components whose allocations a transaction must roll back together
    std::array<transactional_t *, 5> guards() {
        return {&isa().buffers(), &isa().stack(), &loop(),
                &streams().memory_operand(), &streams().operand()};
    }

    /// Runs `f` in a scope whose registers and stack slots end with it
    ///
    /// @param given The integer registers drawn first inside the scope
    /// @param f A callable returning `shortfall_t`
    ///
    /// @pre The scope matches the lifetime of its values in the emitted code
    template <typename F>
    shortfall_t scope(const buffer::buffer_t &given, F &&f) {
        return isa().stack().scope(
                [&] { return isa().buffers().scope(given, f); });
    }

    /// Runs `f` in a scope that is given no registers
    ///
    /// @param f A callable returning `shortfall_t`
    template <typename F>
    shortfall_t scope(F &&f) {
        return scope(buffer::buffer_t {}, f);
    }
};

/// Assembles a `programming_model_t` from arithmetic/memory implementations
///
/// @details `ArithOp`/`MemOp` must be `emitter_t`-constructible
///     `operation::arithmetic_t`/`operation::memory_t` implementers
///
/// @note Pool, loop, stack, transfer, const-arithmetic and control-flow
///     components are caller-owned references
template <typename ArithOp, typename MemOp>
struct programming_model_compose_t : public programming_model_t {
    /// Assembles the model around caller-owned components
    ///
    /// @param e The emitter the arithmetic and memory operations emit through
    /// @param pool The register allocation pool
    /// @param loop The loop factory
    /// @param stack The stack allocator
    /// @param xfer The value transfer operations
    /// @param arith The constant arithmetic operations
    /// @param cf The control flow operations
    /// @param memops The memory operand factory
    /// @param sched The scheduler for composing instruction streams
    explicit programming_model_compose_t(emitter_t e,
            buffer::buffer_pool_t &pool, loop::loopgen_t &loop, stack_t &stack,
            const operation::transfer_t &xfer,
            const operation::const_arithmetic_t &arith,
            const operation::control_flow_t &cf,
            stream::memory_operand::memory_operand_t &memops,
            const scheduler_t &sched)
        : isa_(e, pool, stack, xfer, arith, cf)
        , loop_(loop)
        , streams_(isa_.arithmetic(), memops, sched) {}

    isa_t &isa() override { return isa_; }
    loop::loopgen_t &loop() override { return loop_; }
    stream::stream_gen_t &streams() override { return streams_; }

protected:
    isa_compose_t<ArithOp, MemOp> isa_;
    loop::loopgen_t &loop_;
    stream::stream_gen_compose_t streams_;
};

} // namespace rvjit

#endif
