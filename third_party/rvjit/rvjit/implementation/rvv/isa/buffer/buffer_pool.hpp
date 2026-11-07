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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_BUFFER_BUFFER_POOL_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_BUFFER_BUFFER_POOL_HPP

#include "rvjit/implementation/common/isa/buffer/allocator/group.hpp"
#include "rvjit/implementation/common/isa/buffer/allocator/named.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/buffer/buffer_pool.hpp"

namespace rvjit {
namespace rvv {
namespace buffer {

using buffer_pool_t
        = rvjit::buffer::buffer_compose_t<rvjit::buffer::allocator::named_t,
                rvjit::buffer::allocator::named_t,
                rvjit::buffer::allocator::group_t>;

} // namespace buffer
} // namespace rvv
} // namespace rvjit

#endif
