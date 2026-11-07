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

#ifndef RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_MEMORY_LOAD_HPP
#define RVJIT_IMPLEMENTATION_RVV_ISA_OPERATION_MEMORY_LOAD_HPP

#include "rvjit/implementation/rv64/isa/operation/memory/load.hpp"
#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"

namespace rvjit {
namespace rvv {
namespace memory {

/// Extension of `rv64::memory::load_t` with rvv load expressions
///
/// @details Vector forms emit a unit-stride, strided or indexed instruction
///     per `op.vector_mode`, and `addr.offset` is ignored by all of them
///
/// @pre Vector forms need a valid SEW/LMUL combination set via `vsetvli` and
///     a `vd` name compatible with it
struct load_t : public rv64::memory::load_t {
    static constexpr const char *NAME = "rvjit::rvv::memory";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit load_t(rvjit::emitter_t e) : rv64::memory::load_t(e) {}

    /// Whether `op` is a supported scalar or vector load
    ///
    /// @param op The operation checked
    bool supports(const rvjit::memory::op_t &op) const {
        if (!op.data.is_defined()) return false;
        if (op.data.is_scalar()) return rv64::memory::load_t::supports(op);
        const int w = op.data.width;
        return w == 1 || w == 2 || w == 4 || w == 8;
    }

    /// General vector and scalar memory load
    ///
    /// @details Emission follows `op.data` and `op.vector_mode`
    ///
    /// @param op The operation, in the scalar or vector form
    ///
    /// @pre Scalar `op.data` follows `rv64::memory::load_t::operator()`
    bool operator()(const rvjit::memory::op_t &op) const {
        if (op.data.is_scalar()) return rv64::memory::load_t::operator()(op);

        CHECK(op.data.is_valid(), "invalid operand descriptor");
        CHECK(supports(op), "unsupported datatype width");
        const VReg vd(op.data.name);
        switch (op.vector_mode) {
            case rvjit::memory::vec_addr_t::unit:
                vle(vd, op.addr, op.data.width,
                        rv64::isa_traits_base_t::vm(op.predicate));
                return true;
            case rvjit::memory::vec_addr_t::stride:
                vlse(vd, op.addr, reg_t(op.args), op.data.width,
                        rv64::isa_traits_base_t::vm(op.predicate));
                return true;
            case rvjit::memory::vec_addr_t::index:
                vluxei(vd, op.addr, vreg_t(op.args), op.data.width,
                        rv64::isa_traits_base_t::vm(op.predicate));
                return true;
        }
        LOG_FAIL("operator()", "invalid descriptor");
        return false;
    }

    /// General unit-stride vector load
    ///
    /// @param vd The destination register
    /// @param addr The memory address loaded from
    /// @param width The element size in bytes
    /// @param vm The vector mask policy
    ///
    /// @details Emits vle{8,16,32,64}_v depending on `width`
    ///
    void vle(const VReg &vd, const rvjit::pointer_t &addr, int width,
            VM vm = VM::unmasked) const {
        switch (width) {
            case 1: e_->vle8_v(vd, addr.base, vm); break;
            case 2: e_->vle16_v(vd, addr.base, vm); break;
            case 4: e_->vle32_v(vd, addr.base, vm); break;
            case 8: e_->vle64_v(vd, addr.base, vm); break;
        }
    }

    /// General strided vector load
    ///
    /// @param vd The destination register
    /// @param addr The memory address loaded from
    /// @param stride The register holding the byte distance between elements
    /// @param width The element size in bytes
    /// @param vm The vector mask policy
    ///
    /// @details Emits vlse{8,16,32,64}_v depending on `width`
    ///
    void vlse(const VReg &vd, const rvjit::pointer_t &addr, const Reg &stride,
            int width, VM vm = VM::unmasked) const {
        switch (width) {
            case 1: e_->vlse8_v(vd, addr.base, stride, vm); break;
            case 2: e_->vlse16_v(vd, addr.base, stride, vm); break;
            case 4: e_->vlse32_v(vd, addr.base, stride, vm); break;
            case 8: e_->vlse64_v(vd, addr.base, stride, vm); break;
        }
    }

    /// General indexed vector load
    ///
    /// @param vd The destination register
    /// @param addr The memory address loaded from
    /// @param index The vector register holding the byte offsets
    /// @param iwidth The size in bytes of each element of `index`
    /// @param vm The vector mask policy
    ///
    /// @details Emits vluxei{8,16,32,64}_v depending on `iwidth`
    void vluxei(const VReg &vd, const rvjit::pointer_t &addr, const VReg &index,
            int iwidth, VM vm = VM::unmasked) const {
        switch (iwidth) {
            case 1: e_->vluxei8_v(vd, addr.base, index, vm); break;
            case 2: e_->vluxei16_v(vd, addr.base, index, vm); break;
            case 4: e_->vluxei32_v(vd, addr.base, index, vm); break;
            case 8: e_->vluxei64_v(vd, addr.base, index, vm); break;
        }
    }
};

} // namespace memory
} // namespace rvv
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
