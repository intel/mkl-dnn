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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_BUFFER_STACK_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_BUFFER_STACK_HPP

#include <initializer_list>

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/buffer/buffer_pool.hpp"
#include "rvjit/interface/isa/buffer/stack.hpp"
#include "rvjit/interface/isa/types.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rv64 {

/// Scalar RV64 backend for `rvjit::stack_t`
struct stack_t : public rvjit::stack_t {
    static constexpr const char *NAME = "rvjit::rv64::stack";

    /// Creates a stack allocator emitting through `e`
    ///
    /// @param e The emitter instructions are emitted through
    /// @param pool The pool whose drawn callee-saved registers `preserve()`
    ///     saves
    stack_t(emitter_t e, rvjit::buffer::buffer_pool_t &pool)
        : rvjit::stack_t(pool), e_(e) {}

    void preserve() override {
        preserved_ = callee_saved_(pool_.integers().used());
        has_preserved_ = true;
        preserve_regs_(preserved_.size(),
                [&](int i) { return Reg(preserved_.id(i)); });
    }

    /// @note Restores exactly the registers the last `preserve()` saved
    void restore() override {
        const auto used = callee_saved_(pool_.integers().used());
        if (!has_preserved_) {
            LOG_FAIL_IF(!used.is_empty(), "undetected matching preserve call");
            return;
        }
        LOG_FAIL_IF(used != preserved_,
                "invalid allocation validation since preserve");
        restore_regs_(preserved_.size(),
                [&](int i) { return Reg(preserved_.id(i)); });
        preserved_ = {};
        has_preserved_ = false;
    }

    void preserve(std::initializer_list<reg_t> list) override {
        preserve_regs_(list.size(), [&](int i) { return *(list.begin() + i); });
    }

    void restore(std::initializer_list<reg_t> list) override {
        restore_regs_(list.size(), [&](int i) { return *(list.begin() + i); });
    }

    /// @note Reserves the bytes allocated since the last open
    ///     `reserve_stack()`, so nested pairs only reserve their own growth
    void reserve_stack() override {
        const int delta = peak_ - reserved_top_;
        if (delta) e_->addi(sp, sp, -delta);
        if (!deltas_.push_back(delta))
            LOG_FAIL(__func__, "nested stack reservations overflow");
        reserved_top_ += delta;
        stack_offset_ = peak_;
    }

    /// @note Releases the most recent open `reserve_stack()`, in LIFO order
    bool release_stack() override {
        CHECK(deltas_.size() != 0, "corrupted stack release");
        const int delta = deltas_.pop_back();
        if (delta) e_->addi(sp, sp, delta);
        stack_offset_ -= delta;
        reserved_top_ -= delta;
        peak_ = stack_offset_;
        return true;
    }

private:
    /// The callee-saved registers of `regs`
    static rvjit::buffer::buffer_t callee_saved_(
            const rvjit::buffer::buffer_t &regs) {
        rvjit::buffer::buffer_t b;
        for (int i = 0; i < regs.size(); ++i)
            if (is_callee_saved_gpr(regs.id(i))) b.set(regs.id(i));
        return b;
    }

    /// Shared sp-relative save emission behind preserve()/preserve(list)
    template <typename At>
    void preserve_regs_(int n, At &&at) {
        if (n <= 0) return;
        e_->addi(sp, sp, -(n * 8));
        for (int i = 0; i < n; ++i)
            e_->sd(at(i), sp, i * 8);
    }

    /// Shared sp-relative reload emission behind restore()/restore(list)
    template <typename At>
    void restore_regs_(int n, At &&at) {
        if (n <= 0) return;
        for (int i = 0; i < n; ++i)
            e_->ld(at(i), sp, i * 8);
        e_->addi(sp, sp, n * 8);
    }

    emitter_t e_;
};

} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
