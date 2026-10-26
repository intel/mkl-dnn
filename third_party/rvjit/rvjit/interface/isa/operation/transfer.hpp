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

#ifndef RVJIT_INTERFACE_ISA_OPERATION_TRANSFER_HPP
#define RVJIT_INTERFACE_ISA_OPERATION_TRANSFER_HPP

#include <initializer_list>

#include "rvjit/interface/induction/bind.hpp"
#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/induction/step.hpp"
#include "rvjit/interface/isa/buffered.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace operation {

/// General value movement between registers, sources, slots, and steps
struct transfer_t {
    static constexpr const char *NAME = "rvjit::operation::transfer";

    virtual ~transfer_t() = default;

    /// General value materialization from a source
    ///
    /// @param dst The register receiving the value
    /// @param src The origin of the value
    virtual void get(const reg_t &dst, const source_t &src) const {
        (void)dst;
        (void)src;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// General value persistence into a slot
    ///
    /// @param slot The location receiving the value
    /// @param value The register holding the value
    virtual void put(const slot_t &slot, const reg_t &value) const {
        (void)slot;
        (void)value;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// General update of a value by a compile-time known step
    ///
    /// @param rd The register receiving the updated value
    /// @param rs The register holding the value
    /// @param step The update
    ///
    /// @note `rd == rs` updates in place
    virtual void advance(
            const reg_t &rd, const reg_t &rs, const step_t &step) const {
        (void)rd;
        (void)rs;
        (void)step;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// Initializes the `m` buffer from its source if `needs_to_materialize()`
    ///
    /// @param m The object initialized
    void init(const buffered_t &m) const {
        if (!m.needs_to_materialize()) return;
        if (!m.is_valid()) {
            LOG_FAIL(__func__, "a materializable object lacks its resources");
            return;
        }
        get(reg_t(m.supplied().id(0)), m.source());
    }

    /// Initializes every element of `list`, in order
    ///
    /// @param list The objects initialized
    template <typename T>
    void init(std::initializer_list<T> list) const {
        for (const T &item : list)
            init(item);
    }

    /// Initializes every element in `[first, last)`, in order
    ///
    /// @param first The first object initialized
    /// @param last One past the last object initialized
    template <typename It>
    void init(It first, It last) const {
        for (; first != last; ++first)
            init(*first);
    }

    /// Saves the value of `b` into its slot, if it has one
    ///
    /// @param b The binding saved
    void preserve(const bind_t &b) const {
        if (b.slot.is_none()) return;
        put(b.slot, b.reg);
    }

    /// Reloads the value of `b` from its slot or origin, if it has one
    ///
    /// @param b The binding reloaded
    void restore(const bind_t &b) const {
        const source_t src = b.reload();
        if (src.is_none() || !b.is_valid()) return;
        get(b.reg, src);
    }

    /// Updates the value of `b` using its step, if it has one
    ///
    /// @param b The binding updated
    void advance(const bind_t &b) const {
        if (b.step.is_none()) return;
        advance(b.reg, b.reg, b.step);
    }

    /// Reloads the value of `b` from a register and updates it in one go
    ///
    /// @param b The binding reloaded and updated
    ///
    /// @note Equals `restore(b)` then `advance(b)` with one less move
    ///
    /// @pre `b.reload()` is a register source and `b.step` is not `none`
    void reload_advance(const bind_t &b) const {
        advance(b.reg, b.reload().buffer(), b.step);
    }
};

/// Assembles a `transfer_t` from individual operation implementations
///
/// @details `GetOp` and `PutOp` must be constructible from `emitter_t`,
///     `AdvanceOp` from `emitter_t` and a `const_arithmetic_t`
template <typename GetOp, typename PutOp, typename AdvanceOp>
struct transfer_compose_t : public transfer_t {
    using transfer_t::advance;

    /// Constructs every operation implementation
    ///
    /// @param e The emitter the operations emit through
    /// @param arith The constant arithmetic `advance` uses
    transfer_compose_t(emitter_t e, const const_arithmetic_t &arith)
        : get_(e), put_(e), advance_(e, arith) {}

    void get(const reg_t &dst, const source_t &src) const override {
        get_(dst, src);
    }

    void put(const slot_t &slot, const reg_t &value) const override {
        put_(slot, value);
    }

    void advance(const reg_t &rd, const reg_t &rs,
            const step_t &step) const override {
        advance_(rd, rs, step);
    }

protected:
    GetOp get_;
    PutOp put_;
    AdvanceOp advance_;
};

} // namespace operation
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
