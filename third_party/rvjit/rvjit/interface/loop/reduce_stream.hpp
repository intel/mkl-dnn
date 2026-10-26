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

#ifndef RVJIT_INTERFACE_LOOP_REDUCE_STREAM_HPP
#define RVJIT_INTERFACE_LOOP_REDUCE_STREAM_HPP

#include "rvjit/interface/core/callback.hpp"
#include "rvjit/interface/core/index.hpp"
#include "rvjit/interface/loop/loop.hpp"
#include "rvjit/interface/stream/scheduler.hpp"
#include "rvjit/interface/stream/stream_impl.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace loop {

using policy_t = stream::policy_t;
using stream::ref_t;
using on_nest_t = stream::on_nest_t;
using on_init_t = callback_t<>;

/// Abstract K-reduction dispatch interface
struct reduce_stream_t {
    static constexpr const char *NAME = "rvjit::loop";

    virtual ~reduce_stream_t() = default;

    /// Emits the reduction
    ///
    /// @details `rd` accumulates, `rs1`/`rs2` are macc's `src1`/`src2`, `loop`
    ///     is the reduction loop (e.g. K), `delay` schedules the dense pass
    ///
    /// @pre `rs1`/`rs2` axis-0 size is at least `loop`'s widest stage width
    ///
    /// @note `rs1`/`rs2` are narrowed in place per stage and restored to
    ///     their entry shape on return
    /// @note `after_init`, if set, runs once between the init streams and the
    ///     loop
    virtual void operator()(ref_t rd, ref_t rs1, ref_t rs2, loop_t &loop,
            const policy_t &delay, const on_nest_t &cb,
            const on_init_t &after_init = {}) const
            = 0;

    /// Largest `delay` `operator()` can take without a deferred `cb` reading
    /// a register already rewritten
    ///
    /// @param rs1 The stream whose own axes vary fastest
    /// @param rs2 The stream whose own axes are reused per shared position
    ///
    /// @pre `get(idx)` of `rs1`/`rs2` names the register holding position
    ///     `idx` at their current shapes, and their productions are the only
    ///     writers of the registers `cb` reads
    /// @pre The shapes are those `operator()` runs with; recompute after any
    ///     `resize()`. Narrower stages of `loop` only remove overwrites
    virtual int max_delay(ref_t rs1, ref_t rs2) const {
        (void)rs1;
        (void)rs2;
        LOG_FAIL(__func__, "unimplemented");
        return 0;
    }
};

} // namespace loop
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
