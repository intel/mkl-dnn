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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_ARITHMETIC_HPP

#include "rvjit/implementation/rv64/isa/operation/arithmetic/bitcast.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/add.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/convert.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/extend.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/macc.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/mul.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/narrow.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/widen.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/zero.hpp"
#include "rvjit/interface/isa/operation/arithmetic.hpp"

namespace rvjit {
namespace rvv {
namespace arithmetic {

using arithmetic_t = operation::arithmetic_compose_t<macc_t, zero_t, mul_t,
        add_t, rv64::arithmetic::bitcast_t, narrow_t, widen_t, extend_t,
        convert_t>;

} // namespace arithmetic
} // namespace rvv
} // namespace rvjit

#endif
