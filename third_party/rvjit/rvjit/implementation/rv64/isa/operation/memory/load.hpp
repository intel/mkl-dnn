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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_MEMORY_LOAD_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_OPERATION_MEMORY_LOAD_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/operation/memory.hpp"
#include "rvjit/interface/isa/pointer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/debug_def.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace memory {

/// Scalar load expression
struct load_t {
    static constexpr const char *NAME = "rvjit::rv64::memory";

    /// Creates the component
    ///
    /// @param e The emitter instructions are emitted through
    explicit load_t(rvjit::emitter_t e) : e_(e) {}

    /// Whether `op.data` has a scalar load width
    ///
    /// @param op The operation checked
    bool supports(const rvjit::memory::op_t &op) const {
        if (!op.data.is_defined()) return false;
        const int w = op.data.width;
        return op.data.is_int() ? (w == 1 || w == 2 || w == 4 || w == 8)
                                : (w == 2 || w == 4 || w == 8);
    }

    /// General scalar memory load
    ///
    /// @param op The operation, in the scalar form
    ///
    /// @details Emits an integer or float load based on `op.data`
    bool operator()(const rvjit::memory::op_t &op) const {
        CHECK(op.data.is_valid(), "invalid operand descriptor");
        CHECK(supports(op), "unsupported datatype width");
        if (op.data.is_int())
            xload(reg_t(op.data.name), op.addr, op.data.width,
                    op.data.is_signed());
        else
            fload(freg_t(op.data.name), op.addr, op.data.width);
        return true;
    }

    /// General scalar float load
    ///
    /// @param fd The destination register
    /// @param addr The memory address loaded from
    /// @param width The access size in bytes
    ///
    /// @details Emits flh, flw, or fld based on `width`
    ///
    /// @note Emits nothing if `width` has no supported float load
    ///
    /// @pre `width` must be 2, 4, or 8 (bytes)
    void fload(const FReg &fd, const rvjit::pointer_t &addr, int width) const {
        switch (width) {
            case 2: e_->flh(fd, addr.base, addr.offset); break;
            case 4: e_->flw(fd, addr.base, addr.offset); break;
            case 8: e_->fld(fd, addr.base, addr.offset); break;
            default: LOG_FAIL("load", "unsupported datatype width"); break;
        }
    }

    /// General scalar integer load
    ///
    /// @param rd The destination register
    /// @param addr The memory address loaded from
    /// @param width The access size in bytes
    /// @param is_signed Whether narrow loads sign-extend, else zero-extend
    ///
    /// @details Emits lb/lbu, lh/lhu, lw/lwu, or ld based on `width`/`is_signed`
    ///
    /// @note Emits nothing if `width` is unsupported
    ///
    /// @pre `width` must be 1, 2, 4, or 8 (bytes)
    void xload(const Reg &rd, const rvjit::pointer_t &addr, int width,
            bool is_signed = true) const {
        switch (width) {
            case 1:
                is_signed ? e_->lb(rd, addr.base, addr.offset)
                          : e_->lbu(rd, addr.base, addr.offset);
                break;
            case 2:
                is_signed ? e_->lh(rd, addr.base, addr.offset)
                          : e_->lhu(rd, addr.base, addr.offset);
                break;
            case 4:
                is_signed ? e_->lw(rd, addr.base, addr.offset)
                          : e_->lwu(rd, addr.base, addr.offset);
                break;
            case 8: e_->ld(rd, addr.base, addr.offset); break;
            default: LOG_FAIL("load", "unsupported datatype width"); break;
        }
    }

protected:
    emitter_t e_;
};

} // namespace memory
} // namespace rv64
} // namespace rvjit

#include "rvjit/utils/debug_undef.hpp"

#endif
