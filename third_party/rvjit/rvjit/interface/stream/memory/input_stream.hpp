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

#ifndef RVJIT_INTERFACE_STREAM_MEMORY_INPUT_STREAM_HPP
#define RVJIT_INTERFACE_STREAM_MEMORY_INPUT_STREAM_HPP

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/conversion.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/stream/buffer_map.hpp"
#include "rvjit/interface/stream/memory/conversion_stream.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"
#include "rvjit/interface/stream/scheduler.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace memory_operand {

/// A stream loading values and converting them to a target in-register type
///
/// @note `fetch` and `convert` are separate so that a consumer can issue the
///     loads of several positions ahead of their conversions
///
/// @pre Without a `configure` switch, the code runs in a configuration where
///     the single step of the conversion is valid (e.g. the vtype of the
///     narrow type), and the caller owns the load stream
struct input_stream_t : public stream_impl_t {
    static constexpr const char *NAME = "rvjit::stream::memory_operand::input";

    input_stream_t() = default;

    /// A stream that is never valid
    static input_stream_t *invalid() {
        static input_stream_t none;
        return &none;
    }

    /// Configures the stream over a load operand
    ///
    /// @param arith The arithmetic emitting the conversion
    /// @param load The load operand, which the stream drives
    /// @param t The loaded type (the operand's) and the type the registers
    ///     compute in
    /// @param offset The registers between a wide register and its upper
    ///     part, the load operand's variant `offset`
    /// @param configure The configuration switch, which the stream restores to
    ///     the computing type's width after converting
    ///
    /// @note Returns false for a type pair no chain of steps of `arith`
    ///     converts
    bool init(const operation::arithmetic_t &arith, memory_operand_impl_t *load,
            const type_parameter_t &t, int offset,
            const conversion::configure_t &configure
            = conversion::configure_t()) {
        CHECK(load && load->is_valid(), "invalid load operand");
        CHECK(load->op().dir == memory::dir_t::load, "not a load operand");
        const memory::op_t &op = load->op();
        CHECK(op.data.type == t.mem, "a load of another type");
        CHECK(!t.narrows(), "a loaded type wider than the computing type");
        conversion_t conv;
        CHECK(conv.plan_load(t, op.data.form, arith),
                "unsupported conversion of the loaded type");
        CHECK(conv_.init(arith, conv, load, t.reg, offset, configure),
                "invalid conversion");
        load_ = load;
        return true;
    }

    /// Whether the loaded type differs from the computing type
    bool converts() const { return conv_.converts(); }

    int ndims() const override { return load_->ndims(); }
    int size(int d) const override { return load_->size(d); }
    bool is_valid() const override { return load_ && load_->is_valid(); }
    bool resize(const index_t &shape) override {
        CHECK(is_valid(), "invalid stream");
        return load_->resize(shape);
    }

    /// The register at `idx`, holding the computing type once converted
    name_t get(const index_t &idx) const override { return conv_.get(idx); }

    /// The ring of the load operand
    ///
    /// @note Registers of the wide groups are offset by `offset` registers
    const buffer::buffer_t &buffer() const override { return load_->buffer(); }

    /// The ring of the load operand, for a caller reshaping its layout
    buffer_map_t &regs() { return load_->regs(); }

    /// Emits the load of position `idx`
    void fetch(const index_t &idx) { load_->iteration(idx); }

    /// Emits the conversion of position `idx`, nothing when the types match
    void convert(const index_t &idx) { conv_.iteration(idx); }

    /// Emits the loads of `n` positions, then their conversions step by step,
    /// each step in its own configuration
    ///
    /// @param at The callable giving the `j`-th position, `index_t(int)`
    /// @param n The number of positions
    template <typename At>
    void fetch_convert(const At &at, int n) {
        if (conv_.converts()) conv_.configure(0);
        for (int j = 0; j < n; ++j)
            fetch(at(j));
        conv_.emit_steps(0, conv_.conversion().size(), at, n);
        conv_.restore();
    }

protected:
    /// Emits the load of position `idx`, then its conversion step by step,
    /// each step in its own configuration
    void produce_(const index_t &idx) override {
        fetch_convert([&idx](int) { return idx; }, 1);
    }

    /// Builds whatever pivot chains the load's addressing needs
    void init_(const index_t &idx) override { load_->init_iteration(idx); }

private:
    conversion_stream_t conv_;
    memory_operand_impl_t *load_ = nullptr;
};

} // namespace memory_operand
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
