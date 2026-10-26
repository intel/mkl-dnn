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

#ifndef RVJIT_IMPLEMENTATION_RV64_STREAM_ITERATOR_ITERATOR_HPP
#define RVJIT_IMPLEMENTATION_RV64_STREAM_ITERATOR_ITERATOR_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rv64/stream/iterator/strategies/guided.hpp"
#include "rvjit/implementation/rv64/stream/iterator/strategies/pivot.hpp"
#include "rvjit/interface/stream/iterator/iterator.hpp"

namespace rvjit {
namespace rv64 {
namespace stream {
namespace iterator {

/// Ids of the strategies of `strategies()`
static constexpr rvjit::stream::iterator::strategy_id_t pivot {0};
static constexpr rvjit::stream::iterator::strategy_id_t guided {1};

/// The RV64 strategies as a cost ladder: `pivot` yields the best streams and
/// may cost the most registers, `guided` is the cost floor
inline const rvjit::stream::iterator::strategy_set_t &strategies() {
    static const pivot_t pivot_strategy;
    static const guided_t guided_strategy;
    static const rvjit::stream::iterator::strategy_set_t set = [] {
        rvjit::stream::iterator::strategy_set_t s;
        s.add(pivot_strategy);
        s.add(guided_strategy);
        return s;
    }();
    return set;
}

/// Pointer iterators over the RV64 strategies
struct iterator_t : rvjit::stream::iterator::iterator_t {
    /// Creates an iterator factory over the RV64 strategies
    ///
    /// @param e The emitter instructions are emitted through
    /// @param arith The constant arithmetic component
    /// @param xfer The value transfer component
    /// @param pool The pool the iterator registers are drawn from
    iterator_t(rvjit::emitter_t e, const operation::const_arithmetic_t &arith,
            const operation::transfer_t &xfer,
            rvjit::buffer::buffer_pool_t &pool)
        : rvjit::stream::iterator::iterator_t(e, arith, xfer, pool,
                  rvjit::rv64::stream::iterator::strategies()) {}

    /// Resolves a plan over the RV64 strategies
    ///
    /// @param request What the caller asks of the iterator
    /// @param ntmp The integer registers available
    ///
    /// @note Static: needs no emitter or pool state
    static rvjit::stream::iterator::plan_t plan(
            const rvjit::stream::iterator::request_t &request, int ntmp) {
        return resolve(
                rvjit::rv64::stream::iterator::strategies(), request, ntmp);
    }
};

} // namespace iterator
} // namespace stream
} // namespace rv64
} // namespace rvjit

#endif
