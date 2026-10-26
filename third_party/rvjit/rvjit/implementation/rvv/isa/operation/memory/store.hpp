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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_MEMORY_STORE_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_MEMORY_STORE_HPP

#include "rvjit/implementation/rv64/isa/operation/memory/store.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rvv {
namespace memory {

/// Extension of `rv64::memory::store_t` with rvv store expressions
///
/// @details Vector forms emit a unit-stride, strided or indexed instruction
///     per `op.vector_mode`, and `addr.offset` is ignored by all of them
///
/// @pre Vector forms need a valid SEW/LMUL combination set via `vsetvli` and
///     a `vs` name compatible with it
struct store_t : public rv64::memory::store_t {
    static constexpr const char *NAME = "rvjit::rvv::memory";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit store_t(rvjit::emitter_t e) : rv64::memory::store_t(e) {}

    /// Whether `op` is a supported scalar or vector store
    ///
    /// @param op The operation checked
    bool supports(const rvjit::memory::op_t &op) const {
        if (!op.data.is_defined()) return false;
        if (op.data.is_scalar()) return rv64::memory::store_t::supports(op);
        const int w = op.data.width;
        return w == 1 || w == 2 || w == 4 || w == 8;
    }

    /// General vector and scalar memory store
    ///
    /// @details Emission follows `op.data` and `op.vector_mode`
    ///
    /// @param op The operation, in the scalar or vector form
    ///
    /// @pre Scalar `op.data` follows `rv64::memory::store_t::operator()`
    bool operator()(const rvjit::memory::op_t &op) const {
        if (op.data.is_scalar()) return rv64::memory::store_t::operator()(op);

        CHECK(op.data.is_valid(), "invalid operand descriptor");
        CHECK(supports(op), "unsupported datatype width");
        const VReg vs(op.data.name);
        switch (op.vector_mode) {
            case rvjit::memory::vec_addr_t::unit:
                vse(vs, op.addr, op.data.width,
                        rv64::isa_traits_base_t::vm(op.predicate));
                return true;
            case rvjit::memory::vec_addr_t::stride:
                vsse(vs, op.addr, reg_t(op.args), op.data.width,
                        rv64::isa_traits_base_t::vm(op.predicate));
                return true;
            case rvjit::memory::vec_addr_t::index:
                vsuxei(vs, op.addr, vreg_t(op.args), op.data.width,
                        rv64::isa_traits_base_t::vm(op.predicate));
                return true;
        }
        LOG_FAIL("store", "invalid descriptor");
        return false;
    }

    /// General unit-stride vector store
    ///
    /// @param vs The source register
    /// @param addr The memory address stored to
    /// @param width The element size in bytes
    /// @param vm The vector mask policy
    ///
    /// @details Emits vse{8,16,32,64}_v depending on `width`
    ///
    void vse(const VReg &vs, const rvjit::pointer_t &addr, int width,
            VM vm = VM::unmasked) const {
        switch (width) {
            case 1: e_->vse8_v(vs, addr.base, vm); break;
            case 2: e_->vse16_v(vs, addr.base, vm); break;
            case 4: e_->vse32_v(vs, addr.base, vm); break;
            case 8: e_->vse64_v(vs, addr.base, vm); break;
        }
    }

    /// General strided vector store
    ///
    /// @param vs The source register
    /// @param addr The memory address stored to
    /// @param stride The register holding the byte distance between elements
    /// @param width The element size in bytes
    /// @param vm The vector mask policy
    ///
    /// @details Emits vsse{8,16,32,64}_v depending on `width`
    ///
    void vsse(const VReg &vs, const rvjit::pointer_t &addr, const Reg &stride,
            int width, VM vm = VM::unmasked) const {
        switch (width) {
            case 1: e_->vsse8_v(vs, addr.base, stride, vm); break;
            case 2: e_->vsse16_v(vs, addr.base, stride, vm); break;
            case 4: e_->vsse32_v(vs, addr.base, stride, vm); break;
            case 8: e_->vsse64_v(vs, addr.base, stride, vm); break;
        }
    }

    /// General indexed vector store
    ///
    /// @param vs The source register
    /// @param addr The memory address stored to
    /// @param index The vector register holding the byte offsets
    /// @param iwidth The size in bytes of each element of `index`
    /// @param vm The vector mask policy
    ///
    /// @details Emits vsuxei{8,16,32,64}_v depending on `iwidth`
    void vsuxei(const VReg &vs, const rvjit::pointer_t &addr, const VReg &index,
            int iwidth, VM vm = VM::unmasked) const {
        switch (iwidth) {
            case 1: e_->vsuxei8_v(vs, addr.base, index, vm); break;
            case 2: e_->vsuxei16_v(vs, addr.base, index, vm); break;
            case 4: e_->vsuxei32_v(vs, addr.base, index, vm); break;
            case 8: e_->vsuxei64_v(vs, addr.base, index, vm); break;
        }
    }
};

} // namespace memory
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
