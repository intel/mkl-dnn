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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONTROL_FLOW_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_CONTROL_FLOW_HPP

#include "rvjit/implementation/rv64/isa/operation/control_flow/if.hpp"
#include "rvjit/implementation/rv64/isa/operation/control_flow/if_else.hpp"
#include "rvjit/implementation/rv64/isa/operation/control_flow/switch_case.hpp"
#include "rvjit/implementation/rv64/isa/operation/control_flow/while.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/control_flow.hpp"

namespace rvjit {
namespace rv64 {
namespace control_flow {

using control_flow_t = operation::control_flow_compose_t<if_t, if_else_t,
        while_t, switch_case_t>;

} // namespace control_flow
} // namespace rv64
} // namespace rvjit

#endif
