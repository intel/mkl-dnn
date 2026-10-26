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

#ifndef RVJIT_RVJIT_HPP
#define RVJIT_RVJIT_HPP

/// Umbrella header pulling in all components
///
/// @note Callers compose whichever components they need directly

#include "rvjit/implementation/common/loop/reduce.hpp"
#include "rvjit/implementation/common/stream/combinator.hpp"
#include "rvjit/implementation/rv64/isa/buffer/stack.hpp"
#include "rvjit/implementation/rv64/isa/operation/const_arithmetic/const_arithmetic.hpp"
#include "rvjit/implementation/rv64/isa/operation/control_flow/control_flow.hpp"
#include "rvjit/implementation/rv64/isa/operation/transfer/transfer.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/implementation/rv64/loop/loopgen.hpp"
#include "rvjit/implementation/rv64/stream/iterator/iterator.hpp"
#include "rvjit/implementation/rvv/base_programming_model.hpp"
#include "rvjit/implementation/rvv/isa/buffer/architecture.hpp"
#include "rvjit/implementation/rvv/isa/buffer/buffer_pool.hpp"
#include "rvjit/implementation/rvv/isa/operation/arithmetic/arithmetic.hpp"
#include "rvjit/implementation/rvv/isa/operation/memory/memory.hpp"
#include "rvjit/implementation/rvv/linalg/linalg.hpp"
#include "rvjit/implementation/rvv/programming_model.hpp"
#include "rvjit/implementation/rvv/rvv.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/interface/linalg/linalg.hpp"
#include "rvjit/interface/stream/memory/memory_operand.hpp"
#include "rvjit/utils/utils.hpp"

#endif
