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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_BUFFER_VECTOR_REGISTER_FILE_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_BUFFER_VECTOR_REGISTER_FILE_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/buffer/architecture.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {
namespace rvv {
namespace buffer {

/// The default vector register list (all registers)
///
/// @note No vector register is callee-saved on RV64's calling convention,
///     so the full v0..v31 range is always exposed
struct vector_register_file_t {
    /// The default list, in ascending order
    static rvjit::buffer::buffer_file_t standard() {
        using namespace Xbyak_riscv;
        static const VReg vecs[32] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22,
                v23, v24, v25, v26, v27, v28, v29, v30, v31};
        return rvjit::buffer::buffer_file_t(
                rvjit::buffer::buffer_t(vecs, vecs + 32));
    }
};

} // namespace buffer
} // namespace rvv
} // namespace rvjit

#endif
