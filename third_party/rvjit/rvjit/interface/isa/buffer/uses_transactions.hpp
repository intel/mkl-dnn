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

#ifndef RVJIT_INTERFACE_ISA_BUFFER_USES_TRANSACTIONS_HPP
#define RVJIT_INTERFACE_ISA_BUFFER_USES_TRANSACTIONS_HPP

#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {

/// General nestable transaction interface
///
/// @note Bookkeeping only, implementations must emit no code
struct transactional_t {
    /// Opens a transaction
    ///
    /// @note Transactions nest; `commit()`/`drop()` close the innermost one
    virtual bool begin() = 0;

    /// Keeps every change made since the innermost `begin()`
    virtual bool commit() = 0;

    /// Reverts every change made since the innermost `begin()`
    virtual bool drop() = 0;

    /// Number of currently open transactions
    virtual int open_transactions() const = 0;

protected:
    ~transactional_t() = default;
};

/// Scope guard over one transaction on each of up to `CAPACITY` components
///
/// @details Opens transactions in argument order, closes them in reverse
struct transaction_t {
    static constexpr const char *NAME = "rvjit::transaction";
    static constexpr int CAPACITY = 8;

    /// Opens a transaction on every component, or on none if any fails
    ///
    /// @param ts The components, each a `transactional_t`
    template <typename... Ts>
    explicit transaction_t(Ts &...ts) {
        static_assert(sizeof...(Ts) <= CAPACITY, "too many components");
        transactional_t *list[] = {&ts...};
        for (transactional_t *t : list) {
            if (!begin_(t)) return;
        }
        open_ = true;
    }

    /// Tag selecting the runtime-sized list constructor
    struct list_t {};

    /// Opens a transaction on every component of a runtime-sized list
    ///
    /// @param tag Selects this constructor
    /// @param list The components
    /// @param n The number of components in `list`
    transaction_t(list_t tag, transactional_t *const *list, int n) {
        (void)tag;
        if (n > CAPACITY) return;
        for (int i = 0; i < n; ++i) {
            if (!begin_(list[i])) return;
        }
        open_ = true;
    }
    ~transaction_t() { drop(); }

    transaction_t(const transaction_t &) = delete;
    transaction_t &operator=(const transaction_t &) = delete;

    /// Whether every transaction opened successfully and is still pending
    explicit operator bool() const { return open_; }

    /// Keeps every change made since the transactions opened
    bool commit() {
        if (!open_) return false;
        return close_(true);
    }

    /// Reverts every change made since the transactions opened
    bool drop() {
        if (!open_ && n_ == 0) return false;
        return close_(false);
    }

private:
    /// Begins `t`, dropping everything opened so far if it fails
    bool begin_(transactional_t *t) {
        if (t->begin()) {
            parts_[n_++] = t;
            return true;
        }
        drop();
        return false;
    }

    bool close_(bool keep) {
        bool ok = true;
        while (n_ > 0) {
            transactional_t *t = parts_[--n_];
            ok = (keep ? t->commit() : t->drop()) && ok;
        }
        open_ = false;
        return ok;
    }

    transactional_t *parts_[CAPACITY] = {};
    int n_ = 0;
    bool open_ = false;
};

/// General nestable transaction bookkeeping
///
/// @details Keeps a LIFO of up to `N` `State` values, one per open transaction
///
/// @note Bookkeeping only, hooks must emit no code
template <typename State, int N = 8>
struct uses_transactions_t : public transactional_t {
    static constexpr const char *NAME = "rvjit::uses_transactions";

    bool begin() override {
        CHECK(!open_.full(), "nested transactions overflow");
        State state {};
        if (!on_begin_(state)) return false;
        open_.push_back(state);
        return true;
    }

    bool commit() override {
        CHECK(open_.size() > 0, "unbalanced commit");
        return on_commit_(open_.pop_back());
    }

    bool drop() override {
        CHECK(open_.size() > 0, "unbalanced drop");
        return on_drop_(open_.pop_back());
    }

    int open_transactions() const override { return open_.size(); }

protected:
    ~uses_transactions_t() = default;

    /// Saves the current state into `state`
    virtual bool on_begin_(State &state) = 0;

    /// Called after the innermost transaction is closed by `commit()`
    virtual bool on_commit_(const State &state) {
        (void)state;
        return true;
    }

    /// Called after the innermost transaction is closed by `drop()`
    virtual bool on_drop_(const State &state) = 0;

private:
    array_t<State, N> open_;
};

} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
