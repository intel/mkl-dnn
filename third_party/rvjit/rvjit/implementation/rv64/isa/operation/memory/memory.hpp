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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_MEMORY_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_MEMORY_HPP

#include "rvjit/implementation/rv64/isa/operation/memory/load.hpp"
#include "rvjit/implementation/rv64/isa/operation/memory/store.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/memory.hpp"

namespace rvjit {
namespace rv64 {
namespace memory {

using memory_t = operation::memory_compose_t<load_t, store_t>;

} // namespace memory
} // namespace rv64
} // namespace rvjit

#endif
