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

#ifndef RVJIT_IMPLEMENTATION_COMMON_LOOP_REDUCE_HPP
#define RVJIT_IMPLEMENTATION_COMMON_LOOP_REDUCE_HPP

#include "rvjit/implementation/common/stream/combinator.hpp"
#include "rvjit/interface/loop/reduce_stream.hpp"

namespace rvjit {
namespace loop {

/// ISA-agnostic K-reduction implementation
///
/// @details Interleaves the init streams once, then dispatches `loop`'s
///     stages, narrowing `rs1`/`rs2` in place and nesting them on axis 0
struct reduce_t : public reduce_stream_t {
    static constexpr const char *NAME = "rvjit::loop";

    /// Creates the reduction over a stream combinator
    ///
    /// @param stream The combinator interleaving and nesting the streams
    explicit reduce_t(common::stream::combinator_t &stream) : stream_(stream) {}

    void operator()(ref_t rd, ref_t rs1, ref_t rs2, loop_t &loop,
            const policy_t &delay, const on_nest_t &cb,
            const on_init_t &after_init = {}) const override {
        stream_.interleaved({rd.init(), rs1.init(), rs2.init()});
        if (after_init) after_init();

        const int k1 = rs1.size(0), n1 = rs1.size(1);
        const int k2 = rs2.size(0), n2 = rs2.size(1);
        loop.generate([&](int u) {
            rs1.resize({u, n1});
            rs2.resize({u, n2});
            stream_.nested_1d(delay, rs2, rs1, cb);
        });

        rs1.resize({k1, n1});
        rs2.resize({k2, n2});
    }

    int max_delay(ref_t rs1, ref_t rs2) const override {
        return stream_.max_nested_delay(1, rs2, rs1);
    }

private:
    common::stream::combinator_t &stream_;
};

} // namespace loop
} // namespace rvjit

#endif
