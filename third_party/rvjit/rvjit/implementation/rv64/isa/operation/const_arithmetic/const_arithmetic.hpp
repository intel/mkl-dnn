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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONST_ARITHMETIC_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONST_ARITHMETIC_HPP

#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/add.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/align_down.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/and.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/div.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/mul.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/const_arithmetic.hpp"

namespace rvjit {
namespace rv64 {
namespace const_arithmetic {

using const_arithmetic_t = operation::const_arithmetic_compose_t<add_t, mul_t,
        div_t, align_down_t, and_t>;

} // namespace const_arithmetic
} // namespace rv64
} // namespace rvjit

#endif
