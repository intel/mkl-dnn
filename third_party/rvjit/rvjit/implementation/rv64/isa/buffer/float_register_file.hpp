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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_BUFFER_FLOAT_REGISTER_FILE_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_BUFFER_FLOAT_REGISTER_FILE_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/buffer/architecture.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {
namespace rv64 {
namespace buffer {

/// The default float register list (temporaries and arguments)
struct float_register_file_t {
    /// The default list, in temporaries then arguments order
    static rvjit::buffer::buffer_file_t standard() {
        using namespace Xbyak_riscv;
        static const FReg floats[] = {ft0, ft1, ft2, ft3, ft4, ft5, ft6, ft7,
                ft8, ft9, ft10, ft11, fa0, fa1, fa2, fa3, fa4, fa5, fa6, fa7};
        const rvjit::buffer::buffer_t regs(
                floats, floats + sizeof(floats) / sizeof(floats[0]));
        rvjit::buffer::order_t order;
        for (const FReg &f : floats)
            order.push_back(reg_name(f));
        return rvjit::buffer::buffer_file_t(regs, regs.ordered(order));
    }
};

} // namespace buffer
} // namespace rv64
} // namespace rvjit

#endif
