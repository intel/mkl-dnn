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

#ifndef RVJIT_INTERFACE_STREAM_MEMORY_CONVERSION_STREAM_HPP
#define RVJIT_INTERFACE_STREAM_MEMORY_CONVERSION_STREAM_HPP

#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/isa/conversion.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {
namespace memory_operand {

/// A stream converting the values another stream's registers hold
///
/// @details Each position converts the register of the same position of the
///     `regs` stream. A narrowing (or same-width) conversion runs in place;
///     a widening one reads the narrow value `offset` registers above the
///     wide register it fills, the highest part of its group, and continues
///     in place. A position emits the selected step, or every step
///
/// @note The configuration is switched through `configure` only when a step
///     runs at another width than the current one, starting from the width
///     the stream is created in, which `restore()` returns to
///
/// @pre Without `configure`, the caller runs every step in a valid
///     configuration
struct conversion_stream_t : public stream_impl_t {
    static constexpr const char *NAME
            = "rvjit::stream::memory_operand::conversion";

    conversion_stream_t() = default;

    /// Configures the stream over the registers of `regs`
    ///
    /// @param arith The arithmetic emitting the steps
    /// @param conv The conversion
    /// @param regs The stream whose registers hold the values
    /// @param home The type whose width is in effect around the stream
    /// @param offset The registers between a wide register and the narrow
    ///     part a widening reads, 0 unless the conversion widens
    /// @param configure The configuration switch, empty when the caller keeps
    ///     a valid configuration
    bool init(const operation::arithmetic_t &arith, const conversion_t &conv,
            stream_impl_t *regs, type_t home, int offset = 0,
            const conversion::configure_t &configure
            = conversion::configure_t()) {
        CHECK(regs, "no register stream");
        const bool widens
                = !conv.empty() && sewb(conv.to()) > sewb(conv.from());
        CHECK(!widens || offset > 0,
                "a widening needs the upper part of a group");
        arith_ = &arith;
        conv_ = conv;
        regs_ = regs;
        offset_ = widens ? offset : 0;
        home_ = cur_ = home;
        configure_ = configure;
        step_ = -1;
        return true;
    }

    /// The conversion
    const conversion_t &conversion() const { return conv_; }

    /// Whether there is anything to convert
    bool converts() const { return !conv_.empty(); }

    /// Makes `iteration()` emit step `s` only, or every step for -1
    ///
    /// @param s The step
    void select(int s) { step_ = s; }

    /// Switches to the configuration of step `s`, if it differs
    ///
    /// @param s The step
    bool configure(int s) { return reconfigure_(conv_.step(s).config()); }

    /// Switches back to the width the stream was created in, if it differs
    bool restore() { return reconfigure_(home_); }

    /// Emits steps `first` up to `last` (excluded) of every position, step by
    /// step, each in its own configuration
    ///
    /// @param first The first step
    /// @param last The step after the last one
    void emit_steps(int first, int last) {
        for (int s = first; s < last; ++s) {
            configure(s);
            select(s);
            stream_impl_t::emit();
        }
        select(-1);
    }

    /// Emits steps `first` up to `last` (excluded) of `n` positions, step by
    /// step, each in its own configuration
    ///
    /// @param first The first step
    /// @param last The step after the last one
    /// @param at The callable giving the `j`-th position, `index_t(int)`
    /// @param n The number of positions
    template <typename At>
    void emit_steps(int first, int last, const At &at, int n) {
        for (int s = first; s < last; ++s) {
            configure(s);
            select(s);
            for (int j = 0; j < n; ++j)
                iteration(at(j));
        }
        select(-1);
    }

    int ndims() const override { return regs_->ndims(); }
    int size(int d) const override { return regs_->size(d); }
    bool is_valid() const override { return regs_ && regs_->is_valid(); }
    bool resize(const index_t &shape) override { return regs_->resize(shape); }

    /// The register holding the converted value at `idx`
    name_t get(const index_t &idx) const override {
        return static_cast<name_t>(regs_->get(idx) - offset_);
    }

    const buffer::buffer_t &buffer() const override { return regs_->buffer(); }

protected:
    void produce_(const index_t &idx) override {
        const int first = step_ < 0 ? 0 : step_;
        const int last = step_ < 0 ? conv_.size() : step_ + 1;
        const name_t wide = get(idx);
        for (int s = first; s < last; ++s) {
            const name_t src = s == 0 ? regs_->get(idx) : wide;
            emit_step_(s, wide, src);
        }
    }

private:
    const operation::arithmetic_t *arith_ = nullptr;
    conversion_t conv_;
    stream_impl_t *regs_ = nullptr;
    int offset_ = 0;
    type_t home_ = type_t::none;
    type_t cur_ = type_t::none;
    conversion::configure_t configure_;
    int step_ = -1;

    // Emits step `s` from `src` into `dst`, in its configuration
    bool emit_step_(int s, name_t dst, name_t src) const {
        const arithmetic::op_t o = conv_.op(s, dst, src);
        switch (conv_.step(s).kind) {
            case conversion::kind_t::extend: return arith_->extend(o);
            case conversion::kind_t::widen: return arith_->widen(o);
            case conversion::kind_t::convert: return arith_->convert(o);
            default: return arith_->narrow(o);
        }
    }

    bool reconfigure_(type_t t) {
        if (!configure_ || sewb(t) == sewb(cur_)) return true;
        CHECK(configure_(t), "no configuration of the step's width");
        cur_ = t;
        return true;
    }
};

} // namespace memory_operand
} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
