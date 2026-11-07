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

#ifndef RVJIT_INTERFACE_STREAM_STREAM_HPP
#define RVJIT_INTERFACE_STREAM_STREAM_HPP

#include "rvjit/implementation/common/loop/reduce.hpp"
#include "rvjit/implementation/common/stream/combinator.hpp"
#include "rvjit/implementation/common/stream/operand/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/loop/reduce_stream.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"
#include "rvjit/interface/stream/scheduler.hpp"

namespace rvjit {
namespace stream {

/// Composite interface of the stream components
///
/// @details Implementations aggregate the components behind each accessor
struct stream_gen_t {
    static constexpr const char *NAME = "rvjit::stream";

    virtual ~stream_gen_t() = default;

    /// A scheduler for composing instruction streams
    virtual const scheduler_t &scheduler() const = 0;

    /// The register-only operand construction component
    virtual common::stream::operand::operand_t &operand() = 0;

    /// The memory-operand construction component
    virtual memory_operand::memory_operand_t &memory_operand() = 0;

    /// The reduction dispatch over operand streams
    virtual loop::reduce_stream_t &reduce() = 0;
};

/// Assembles a `stream_gen_t` around caller-owned components
///
/// @note The operand component zeroes accumulators through the arithmetic
///     operation it is given, and the memory operand factory is a reference
struct stream_gen_compose_t : public stream_gen_t {
    /// Assembles the stream components
    ///
    /// @param arith The arithmetic operation operand streams zero through
    /// @param memops The memory operand factory
    /// @param sched The scheduler for composing instruction streams
    explicit stream_gen_compose_t(operation::arithmetic_t &arith,
            memory_operand::memory_operand_t &memops, const scheduler_t &sched)
        : memory_operand_(memops)
        , scheduler_(sched)
        , operand_(arith)
        , combinator_(sched)
        , reduce_(combinator_) {}

    const scheduler_t &scheduler() const override { return scheduler_; }
    common::stream::operand::operand_t &operand() override { return operand_; }
    memory_operand::memory_operand_t &memory_operand() override {
        return memory_operand_;
    }
    loop::reduce_stream_t &reduce() override { return reduce_; }

protected:
    memory_operand::memory_operand_t &memory_operand_;
    scheduler_t scheduler_;
    common::stream::operand::operand_t operand_;
    common::stream::combinator_t combinator_;
    loop::reduce_t reduce_;
};

} // namespace stream
} // namespace rvjit

#endif
