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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONST_ARITHMETIC_MUL_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONST_ARITHMETIC_MUL_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace const_arithmetic {

struct mul_t {
    static constexpr const char *NAME = "rvjit::rv64::const_arithmetic";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit mul_t(rvjit::emitter_t e) : e_(e) {}

    bool operator()(
            const Reg &rd, const Reg &rs1, const rvjit::const_t &c) const {
        if (c.has_imm()) {
            const auto imm = c.imm();
            const auto abs_imm = imm < 0 ? -imm : imm;

            if (is_pow2(abs_imm)) {
                // Fold a power-of-two constant into a shift
                const int shammt = ilog2q(abs_imm);
                if (shammt > 0 || rd != rs1) e_->slli(rd, rs1, shammt);
                if (imm < 0) e_->sub(rd, zero, rd);
                return true;
            }
        }

        // Constant from register
        if (c.has_reg()) {
            e_->mul(rd, rs1, c.buffer());
            return true;
        }

        // Constant from immediate, not a power of two
        if (c.has_imm() && rd != rs1) {
            // Fallback to mul if rd and rs1 are not aliased
            e_->li(rd, c.imm());
            e_->mul(rd, rs1, rd);
            return true;
        }

        // Illegal alias of rd and rs1
        LOG_FAIL("mul", "illegal rd/rs1 alias");
        return false;
    }

private:
    emitter_t e_;
};

} // namespace const_arithmetic
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
