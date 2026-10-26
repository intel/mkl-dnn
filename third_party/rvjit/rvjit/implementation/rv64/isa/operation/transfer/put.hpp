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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_TRANSFER_PUT_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_TRANSFER_PUT_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/induction/slot.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rv64 {
namespace transfer {

struct put_t {
    static constexpr const char *NAME = "rvjit::rv64::transfer";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit put_t(rvjit::emitter_t e) : e_(e) {}

    void operator()(const slot_t &slot, const Reg &value) const {
        switch (slot.kind()) {
            case slot_t::kind_t::buffer: e_->mv(slot.buffer(), value); break;
            case slot_t::kind_t::memory: {
                const pointer_t p = slot.memory();
                e_->sd(value, p.base, p.offset);
                break;
            }
            default: LOG_FAIL(__func__, "an unsupported slot kind"); break;
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
