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

#ifndef RVJIT_INTERFACE_LOOP_LOOPGEN_HPP
#define RVJIT_INTERFACE_LOOP_LOOPGEN_HPP

#include <array>
#include <utility>
#include <type_traits>

#include "rvjit/interface/isa/buffer/uses_transactions.hpp"
#include "rvjit/interface/loop/loop.hpp"
#include "rvjit/interface/loop/plan.hpp"

#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace loop {

/// Per-loop states to rewind to on `drop()`, recorded by `begin()`
struct frames_mark_t {
    static constexpr int CAPACITY = 32;
    state_t states[CAPACITY];
    int loops = 0;
};

/// Builds loops, referred to by `loop_t` pointers
///
/// @note Loops are resolved here and allocated by `loop_t::allocate()`
/// @note Transactions cover the variable bindings and the lending levels
struct loopgen_t : public uses_transactions_t<frames_mark_t, 8> {
    static constexpr const char *NAME = "rvjit::loopgen";

    virtual ~loopgen_t() = default;

    /// Resolves a `plan_t` into a concrete loop
    ///
    /// @param p The plan
    ///
    /// @note Returns `loop_t::invalid()` on failure
    /// @note The returned pointer is valid during this object's lifetime
    virtual loop_t *new_loop(const plan_t &p) {
        (void)p;
        LOG_FAIL(__func__, "unimplemented");
        return loop_t::invalid();
    }

    /// A loop over independent iterations
    ///
    /// @param s The trip count
    /// @param u The largest unroll factor
    /// @param g The work units one iteration covers
    /// @param d The reach of the branches around the loop body
    ///
    /// @note Returns `loop_t::invalid()` on failure
    loop_t *data_parallel(const size_parameter_t &s, int u, int g = 1,
            control_flow::distance_t d = control_flow::distance_t::short_) {
        return new_loop(plan_t::data_parallel(s, u, g).with_distance(d));
    }

    /// A loop whose stages accumulate into shared state
    ///
    /// @param s The trip count
    /// @param u The largest unroll factor
    /// @param g The work units one iteration covers
    /// @param d The reach of the branches around the loop body
    ///
    /// @note Returns `loop_t::invalid()` on failure
    loop_t *reduce(const size_parameter_t &s, int u, int g = 1,
            control_flow::distance_t d = control_flow::distance_t::short_) {
        return new_loop(plan_t::reduce(s, u, g).with_distance(d));
    }

    /// Resolves a `plan_t` into a concrete loop sharing `src`'s resources
    ///
    /// @param p The plan
    /// @param src The loop whose resources the new loop adopts
    ///
    /// @pre `src` was returned by this object
    /// @pre `src` is allocated before the new loop
    virtual loop_t *alias(const plan_t &p, loop_t &src) {
        (void)p;
        (void)src;
        LOG_FAIL(__func__, "unimplemented");
        return loop_t::invalid();
    }

    /// A loop of the kind of `src` adopting its resources
    ///
    /// @param src The loop whose resources the new loop adopts
    /// @param s The trip count
    /// @param u The largest unroll factor
    /// @param g The work units one iteration covers
    /// @param d The reach of the branches around the loop body
    ///
    /// @pre `src` was returned by this object
    loop_t *alias(loop_t &src, const size_parameter_t &s, int u, int g = 1,
            control_flow::distance_t d = control_flow::distance_t::short_) {
        return alias(plan_t(src.kind(), s, u, g).with_distance(d), src);
    }

    /// The distance the branches around a loop body of `instrs` instructions
    /// must reach
    ///
    /// @param instrs The instructions in the loop body
    virtual control_flow::distance_t reach(int instrs) const {
        (void)instrs;
        LOG_FAIL(__func__, "unimplemented");
        return control_flow::distance_t::short_;
    }

protected:
    bool on_begin_(frames_mark_t &) override {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }

    bool on_drop_(const frames_mark_t &) override {
        LOG_FAIL(__func__, "unimplemented");
        return false;
    }
};

/// Assembles a concrete `loopgen_t` holding up to `N` loops of type `Impl`
///
/// @details `Impl` is constructed in place from the constructor arguments
///     and provides `init(plan)` and `init_alias(plan, src)`
template <typename Impl, int N>
struct loopgen_compose_t : public loopgen_t {
    static_assert(
            std::is_base_of<loop_t, Impl>::value, "Impl must be a loop_t");
    static_assert(N <= frames_mark_t::CAPACITY, "too many loops");

    static constexpr const char *NAME = "rvjit::loopgen_compose";
    static constexpr int CAPACITY = N;

    /// Constructs every loop in place
    ///
    /// @param args The arguments forwarded to the constructor of each `Impl`
    template <typename... Args>
    explicit loopgen_compose_t(Args &&...args)
        : storage_(make_(typename make_index_seq_t<N>::type(), args...)) {}

    loopgen_compose_t(const loopgen_compose_t &) = delete;
    loopgen_compose_t &operator=(const loopgen_compose_t &) = delete;

    loop_t *new_loop(const plan_t &p) override {
        CHECK_OR_RET(cursor_ < N, loop_t::invalid(), "loop storage exhausted");
        Impl &slot = storage_[cursor_];
        CHECK_OR_RET(slot.init(p), loop_t::invalid(),
                "unable to initialize the loop");
        ++cursor_;
        return &slot;
    }

    using loopgen_t::alias;

    control_flow::distance_t reach(int instrs) const override {
        return Impl::reach(instrs);
    }

    loop_t *alias(const plan_t &p, loop_t &src) override {
        Impl *from = owned_(src);
        CHECK_OR_RET(from, loop_t::invalid(),
                "a source loop not owned by this object");
        CHECK_OR_RET(cursor_ < N, loop_t::invalid(), "loop storage exhausted");
        Impl &slot = storage_[cursor_];
        CHECK_OR_RET(slot.init_alias(p, *from), loop_t::invalid(),
                "unable to initialize the loop");
        ++cursor_;
        return &slot;
    }

protected:
    bool on_begin_(frames_mark_t &mark) override {
        mark.loops = cursor_;
        for (int i = 0; i < cursor_; ++i)
            mark.states[i] = storage_[i].checkpoint();
        return true;
    }

    /// @note Rewinds every loop to its state at `begin()`
    bool on_drop_(const frames_mark_t &mark) override {
        for (int i = 0; i < mark.loops; ++i)
            storage_[i].rewind(mark.states[i]);
        return true;
    }

private:
    template <std::size_t... Is, typename... Args>
    static std::array<Impl, N> make_(index_seq_t<Is...>, Args &...args) {
        return std::array<Impl, N> {{((void)Is, Impl(args...))...}};
    }

    Impl *owned_(const loop_t &src) {
        for (int i = 0; i < cursor_; ++i)
            if (static_cast<const loop_t *>(&storage_[i]) == &src)
                return &storage_[i];
        return nullptr;
    }

    std::array<Impl, N> storage_;
    int cursor_ = 0;
};

} // namespace loop
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
