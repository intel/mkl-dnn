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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_TRANSFER_ADVANCE_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_TRANSFER_ADVANCE_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/induction/step.hpp"
#include "rvjit/interface/isa/const.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rv64 {
namespace transfer {

struct advance_t {
    static constexpr const char *NAME = "rvjit::rv64::transfer";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    /// @param arith The constant arithmetic component
    advance_t(rvjit::emitter_t e, const operation::const_arithmetic_t &arith)
        : e_(e), arith_(arith) {}

    void operator()(const Reg &rd, const Reg &rs, const step_t &step) const {
        // An aliased zero step has nothing to emit, a moving one needs a mv
        const bool noop_zero = rd == rs;
        switch (step.kind) {
            case step_t::kind_t::add:
                if (!step.src1.is_valid()) {
                    LOG_FAIL(__func__, "an invalid step constant");
                    return;
                }
                arith_.add(rd, rs, step.src1, noop_zero);
                break;
            case step_t::kind_t::sub:
                if (!step.src1.is_valid()) {
                    LOG_FAIL(__func__, "an invalid step constant");
                    return;
                }
                if (step.src1.has_reg())
                    e_->sub(rd, rs, step.src1.buffer());
                else
                    arith_.add(rd, rs, const_t(-step.src1.imm()), noop_zero);
                break;
            case step_t::kind_t::macc: macc_(rd, rs, step); break;
            default: LOG_FAIL(__func__, "an unsupported step kind"); break;
        }
    }

private:
    /// Whether `rd` can hold the product while `rs` is still to be added
    ///
    /// @note True when `rd` neither aliases `rs` nor a step operand register
    static bool is_scratch_(const Reg &rd, const Reg &rs, const step_t &step) {
        if (rd == rs) return false;
        if (step.src1.has_reg() && step.src1.buffer() == rd) return false;
        if (step.src2.has_reg() && step.src2.buffer() == rd) return false;
        return true;
    }

    /// Updates `rd = rs + step.src1 * step.src2`
    ///
    /// @note Folds into a single `addi` when the product is a simm12
    /// @note Uses `rd` as the product scratch when it does not alias `rs`
    ///     or an operand, the step's own buffer otherwise
    void macc_(const Reg &rd, const Reg &rs, const step_t &step) const {
        if (!step.macc_needs_multiply()) {
            arith_.add(rd, rs, const_t(step.src1.imm() * step.src2.imm()),
                    rd == rs);
            return;
        }

        const bool use_rd = is_scratch_(rd, rs, step);
        if (!use_rd && !step.has_buffer()) {
            LOG_FAIL(__func__, "a missing macc temporary buffer");
            return;
        }
        const Reg tmp = use_rd ? rd : Reg(step.buffer());

        if (step.src1.has_imm() && step.src2.has_imm()) {
            const int64_t res = static_cast<int64_t>(step.src1.imm())
                    * static_cast<int64_t>(step.src2.imm());
            e_->li(tmp, static_cast<int>(res));
        } else if (step.src1.has_reg()) {
            arith_.mul(tmp, step.src1.buffer(), step.src2);
        } else if (step.src2.has_reg()) {
            arith_.mul(tmp, step.src2.buffer(), step.src1);
        } else {
            LOG_FAIL(__func__, "a macc operand without immediates or buffer");
            return;
        }
        e_->add(rd, rs, tmp);
    }

    emitter_t e_;
    const operation::const_arithmetic_t &arith_;
};

} // namespace transfer
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
