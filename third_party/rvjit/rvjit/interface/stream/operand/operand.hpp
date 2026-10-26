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

#ifndef RVJIT_INTERFACE_STREAM_OPERAND_OPERAND_HPP
#define RVJIT_INTERFACE_STREAM_OPERAND_OPERAND_HPP

#include "rvjit/interface/isa/operand.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/stream/buffer_map.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"
#include "rvjit/interface/stream/stream_table.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace operand {

/// Stream of registers related to operands on an instruction stream
struct operand_impl_t : public stream_impl_t {
    static constexpr const char *NAME = "rvjit::stream::operand";

    /// Creates an unconfigured operand
    ///
    /// @param arith The arithmetic operations that zero an accumulator
    explicit operand_impl_t(operation::arithmetic_t *arith = nullptr)
        : arith_(arith) {}

    /// The stream handed out in place of one that failed to be created
    static operand_impl_t *invalid() {
        static operand_impl_t none;
        return &none;
    }

    /// Configures the stream as a register view, optionally zeroed by `init`
    ///
    /// @param block The registers and their layout
    /// @param traits The type of every register
    /// @param zero Whether the init phase zeroes every register
    bool init(const buffer_map_t &block, const rvjit::operand_t &traits,
            bool zero) {
        block_ = block;
        traits_ = traits;
        zero_ = zero;
        return true;
    }

    int ndims() const override { return NDIMS; }

    int size(int d) const override {
        CHECK_OR_RET(is_valid(), 0, "invalid stream");
        return block_.shape[d];
    }

    bool is_valid() const override { return block_.is_valid(); }

    bool resize(const index_t &shape) override {
        CHECK(is_valid(), "invalid stream");
        for (int d = 0; d < NDIMS; ++d)
            block_.shape[d] = shape[d];
        return true;
    }

    name_t get(const index_t &idx) const override {
        CHECK_OR_RET(is_valid(), name_t(), "invalid stream");
        return block_({idx[0], idx[1]});
    }

    const buffer::buffer_t &buffer() const override {
        static const buffer::buffer_t none;
        CHECK_OR_RET(is_valid(), none, "invalid stream");
        return block_.storage;
    }

protected:
    void produce_(const index_t &) override {}

    void init_(const index_t &idx) override {
        if (!zero_ || !arith_) return;
        arithmetic::op_t op(traits_);
        op.rd.name = block_({idx[0], idx[1]});
        arith_->zero(op);
    }

private:
    operation::arithmetic_t *arith_;
    buffer_map_t block_;
    rvjit::operand_t traits_;
    bool zero_ = false;
};

/// Creates register-only operand streams
///
/// @note Transactions reclaim the streams created since `begin()`
struct operand_t : public stream_table_t<operand_impl_t, 8> {
    static constexpr const char *NAME = "rvjit::stream::operand";

    /// Creates a factory whose accumulators cannot zero themselves
    operand_t()
        : stream_table_t<operand_impl_t, 8>(
                  static_cast<operation::arithmetic_t *>(nullptr)) {}

    /// Creates a factory
    ///
    /// @param arith The arithmetic operations that zero an accumulator
    explicit operand_t(operation::arithmetic_t &arith)
        : stream_table_t<operand_impl_t, 8>(&arith) {}

    /// Creates a plain register view without an associated code stream
    ///
    /// @param block The registers and their layout
    /// @param traits The type of every register
    ///
    /// @note Returns `invalid()` if the table is full
    operand_impl_t *buffered(
            const buffer_map_t &block, rvjit::operand_t traits = {}) {
        return add_(block, traits, false);
    }

    /// Creates a register view whose init stream zeroes every register
    ///
    /// @param block The registers and their layout
    /// @param traits The type of every register
    ///
    /// @note Returns `invalid()` if the table is full
    operand_impl_t *accumulator(
            const buffer_map_t &block, rvjit::operand_t traits) {
        return add_(block, traits, true);
    }

private:
    operand_impl_t *add_(const buffer_map_t &block,
            const rvjit::operand_t &traits, bool zero) {
        operand_impl_t *o = next_();
        CHECK_OR_RET(o, invalid(), "operand table full");
        CHECK_OR_RET(o->init(block, traits, zero), invalid(),
                "unable to create the operand");
        take_();
        return o;
    }
};

} // namespace operand
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
