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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_ARITHMETIC_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/bitcast.hpp"
#include "rvjit/implementation/rv64/isa/operation/arithmetic/zero.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {
namespace rv64 {
namespace arithmetic {

struct arithmetic_t : public operation::arithmetic_t {
    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit arithmetic_t(rvjit::emitter_t e) : zero_(e), bitcast_(e) {}

    bool supports_zero(const rvjit::arithmetic::op_t &op) const override {
        return zero_.supports(op);
    }
    bool zero(const rvjit::arithmetic::op_t &op) const override {
        return zero_(op);
    }
    bool supports_bitcast(const rvjit::arithmetic::op_t &op) const override {
        return bitcast_.supports(op);
    }
    bool bitcast(const rvjit::arithmetic::op_t &op) const override {
        return bitcast_(op);
    }

protected:
    zero_t zero_;
    bitcast_t bitcast_;
};

} // namespace arithmetic
} // namespace rv64
} // namespace rvjit

#endif
