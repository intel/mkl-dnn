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

#ifndef RVJIT_INTERFACE_STREAM_ITERATOR_TYPES_HPP
#define RVJIT_INTERFACE_STREAM_ITERATOR_TYPES_HPP

namespace rvjit {
namespace stream {
namespace iterator {

/// Identifies an iterator strategy by its position in a strategy set
///
/// @note `any` is a request-only value that postpones the choice to the
///     resolver, which tries the set's strategies in order
class strategy_id_t {
public:
    constexpr strategy_id_t() = default;

    /// Creates the id of the strategy at position `id` of a strategy set
    ///
    /// @param id The position
    constexpr explicit strategy_id_t(int id) : id_(id) {}

    /// The id postponing the choice to the resolver
    static constexpr strategy_id_t any() { return strategy_id_t(); }

    /// Whether the choice is postponed to the resolver
    constexpr bool is_any() const { return id_ < 0; }

    /// The position in the strategy set
    constexpr int id() const { return id_; }

    /// Whether both ids name the same strategy
    ///
    /// @param o The id compared against
    constexpr bool operator==(const strategy_id_t &o) const {
        return id_ == o.id_;
    }

    /// Whether the ids name different strategies
    ///
    /// @param o The id compared against
    constexpr bool operator!=(const strategy_id_t &o) const {
        return id_ != o.id_;
    }

private:
    int id_ = -1;
};

/// How an iterator addresses elements to service a consumer memory operation
///
enum class addressing_t {
    base_only, // Serves operations without an offset field
    base_simm, // Optimizes pointer arithmetics using the offset field
};

/// Where the iterator base pointer lands after the last iteration
///
/// @note Lets the pointer state chain into another iterator
class update_t {
public:
    enum class kind_t {
        preserve, // Leaves the base pointer unchanged
        advance, // Moves the pointer past `axis`'s whole declared span
    };

    update_t() : update_t(advance()) {}

    /// Create an instance that preserves the pointer value
    static update_t preserve() { return update_t(kind_t::preserve, 0); }

    /// Create an instance that moves the pointer past `axis`'s whole span
    ///
    /// @param axis The axis whose span the pointer moves past
    ///
    /// @note The iterator owns the base pointer's register and may address
    ///     through it; after each pass it holds its start, plus the span, plus
    ///     the plan's `origin_bias()`
    ///
    /// @pre Nothing else reads the base pointer's register while the stream
    ///     runs
    static update_t advance(int axis = 0) {
        return update_t(kind_t::advance, axis);
    }

    /// What happens to the pointer
    kind_t kind() const { return kind_; }

    /// The axis whose span the pointer moves past
    int axis() const { return axis_; }

private:
    update_t(kind_t kind, int axis) : kind_(kind), axis_(axis) {}

    kind_t kind_;
    int axis_;
};

} // namespace iterator
} // namespace stream
} // namespace rvjit

#endif
