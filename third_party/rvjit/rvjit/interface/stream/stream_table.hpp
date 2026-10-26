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

#ifndef RVJIT_INTERFACE_STREAM_STREAM_TABLE_HPP
#define RVJIT_INTERFACE_STREAM_STREAM_TABLE_HPP

#include <array>
#include <utility>
#include <type_traits>

#include "rvjit/interface/isa/buffer/uses_transactions.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"

#include "rvjit/utils/utils.hpp"

#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace stream {

/// A factory's storage of up to `N` streams of type `Impl`
///
/// @details Each `Impl` is constructed in place from the constructor
///     arguments; `Impl` provides its own `init(...)` to be configured
///
/// @note A transaction covers the slots taken since `begin()`, so `drop()`
///     reclaims them; pointers to dropped streams must not be used afterwards
template <typename Impl, int N>
struct stream_table_t : public uses_transactions_t<int, 8> {
    static_assert(std::is_base_of<stream_impl_t, Impl>::value,
            "Impl must be a stream_impl_t");

    static constexpr const char *NAME = "rvjit::stream::table";
    static constexpr int CAPACITY = N;

    /// Constructs every stream in place
    ///
    /// @param args The arguments forwarded to the constructor of each `Impl`
    template <typename... Args>
    explicit stream_table_t(Args &&...args)
        : storage_(make_(typename make_index_seq_t<N>::type(), args...)) {}

    stream_table_t(const stream_table_t &) = delete;
    stream_table_t &operator=(const stream_table_t &) = delete;

    /// A stream that is never valid, handed out in place of one that failed
    ///
    /// @note It is no slot, so transactions neither take nor reclaim it
    static Impl *invalid() { return Impl::invalid(); }

protected:
    /// The next free slot, or nullptr if the table is full
    ///
    /// @note The slot is only taken once `take_()` is called
    Impl *next_() {
        CHECK_OR_RET(cursor_ < N, nullptr, "stream table full");
        return &storage_[cursor_];
    }

    /// Takes the slot `next_()` returned
    void take_() { ++cursor_; }

    bool on_begin_(int &mark) override {
        mark = cursor_;
        return true;
    }

    bool on_drop_(const int &mark) override {
        cursor_ = mark;
        return true;
    }

private:
    template <std::size_t... Is, typename... Args>
    static std::array<Impl, N> make_(index_seq_t<Is...>, Args &...args) {
        return std::array<Impl, N> {{((void)Is, Impl(args...))...}};
    }

    std::array<Impl, N> storage_;
    int cursor_ = 0;
};

} // namespace stream
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
