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

#ifndef RVJIT_INTERFACE_ISA_BUFFER_STACK_HPP
#define RVJIT_INTERFACE_ISA_BUFFER_STACK_HPP

#include <algorithm>
#include <initializer_list>

#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/buffer/uses_transactions.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {

/// Stack allocation state rolled back by a transaction
struct stack_state_t {
    int offset = 0;
    int peak = 0;

    stack_state_t() = default;

    /// Creates a stack state
    ///
    /// @param offset The bump cursor
    /// @param peak The deepest offset reached
    stack_state_t(int offset, int peak) : offset(offset), peak(peak) {}
};

/// Manages the architecture stack allocation and related routines
struct stack_t : public uses_transactions_t<stack_state_t, 8> {
    static constexpr const char *NAME = "rvjit::stack";
    static constexpr int N_TRANSACTIONS = 8;

    /// Creates a stack allocator
    ///
    /// @param pool The pool whose drawn callee-saved registers `preserve()`
    ///     saves
    explicit stack_t(buffer::buffer_pool_t &pool) : pool_(pool) {}
    virtual ~stack_t() = default;

    /// Preserves every callee-saved gpr the buffer pool has drawn
    ///
    /// @pre Caller must not further allocate registers after calling preserve
    virtual void preserve() { LOG_FAIL(__func__, "unimplemented"); }

    /// Restores all callee-saved gpr allocated using the buffer pool
    ///
    /// @note Restores exactly the registers the last preserve() call saved
    virtual void restore() { LOG_FAIL(__func__, "unimplemented"); }

    /// Preserves a list of gpr in the same order as they are given
    ///
    /// @param regs The registers saved
    ///
    /// @pre Caller should emit preserve/restore call pairs with the same list
    virtual void preserve(std::initializer_list<reg_t> regs) {
        (void)regs;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// Restores a list of gpr in the same order as they are given
    ///
    /// @param regs The registers restored
    ///
    /// @pre Caller should emit preserve/restore call pairs with the same list
    virtual void restore(std::initializer_list<reg_t> regs) {
        (void)regs;
        LOG_FAIL(__func__, "unimplemented");
    }

    /// General stack reservation
    ///
    /// @pre Caller must not further allocate stack slots after calling this
    virtual void reserve_stack() { LOG_FAIL(__func__, "unimplemented"); }

    /// General stack release
    ///
    /// @pre A call to this method must happen after `reserve_stack()`
    virtual bool release_stack() {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    /// General stack-slot bump allocation
    ///
    /// @details Advances the stack by 8 bytes and create a pointer to the slot
    ///
    /// @note Pure bookkeeping, emits nothing
    pointer_t new_stack_slot() {
        const int offset = stack_offset_ - reserved_top_;
        stack_offset_ += 8;
        peak_ = std::max(peak_, stack_offset_);
        return pointer_t::stack_at(offset);
    }

    /// General stack-slot bump allocation as a save location
    ///
    /// @note Pure bookkeeping, emits nothing
    slot_t new_slot() { return slot_t::memory(new_stack_slot()); }

    /// Runs `f` in a scope whose slots end with it
    ///
    /// @param f A callable returning `shortfall_t`
    ///
    /// @note The slots `f` allocates are free again for any later slot
    template <typename F>
    shortfall_t scope(F &&f) {
        const stack_state_t entry = state_();
        const shortfall_t s = f();
        stack_offset_ = entry.offset;
        return s;
    }

protected:
    /// @pre No `reserve_stack()`/`release_stack()` call while one is open
    bool on_begin_(stack_state_t &st) override {
        st = state_();
        return true;
    }

    /// Reclaims every slot allocated since the innermost `begin()`
    bool on_drop_(const stack_state_t &st) override {
        stack_offset_ = st.offset;
        peak_ = st.peak;
        return true;
    }

    stack_state_t state_() const { return {stack_offset_, peak_}; }

    buffer::buffer_pool_t &pool_;
    int stack_offset_ = 0;
    int peak_ = 0; // top of every slot allocated so far
    int reserved_top_ = 0; // cumulative bytes on every `reserve_stack()` call
    array_t<int, N_TRANSACTIONS> deltas_; // Individual reservation deltas
    buffer::buffer_t preserved_; // registers the last preserve() saved
    bool has_preserved_ = false;
};

} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
