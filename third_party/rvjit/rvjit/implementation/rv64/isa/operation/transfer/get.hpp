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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_TRANSFER_GET_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_TRANSFER_GET_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/source.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rv64 {
namespace transfer {

struct get_t {
    static constexpr const char *NAME = "rvjit::rv64::transfer";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit get_t(rvjit::emitter_t e) : e_(e) {}

    void operator()(const Reg &dst, const source_t &src) const {
        switch (src.kind()) {
            case source_t::kind_t::immediate: {
                const auto const_src = const_t(src.immediate());
                if (const_src.has_imm())
                    e_->li(dst, const_src.imm());
                else
                    e_->mv(dst, const_src.buffer());
                break;
            }
            case source_t::kind_t::adopt:
            case source_t::kind_t::copy: e_->mv(dst, src.buffer()); break;
            case source_t::kind_t::load: {
                const pointer_t p = src.memory();
                e_->ld(dst, p.base, p.offset);
                break;
            }
            default: LOG_FAIL(__func__, "an unsupported source kind"); break;
        }
    }

private:
    emitter_t e_;
};

} // namespace transfer
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
