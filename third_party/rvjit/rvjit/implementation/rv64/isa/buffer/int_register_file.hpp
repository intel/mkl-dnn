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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_BUFFER_INT_REGISTER_FILE_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_BUFFER_INT_REGISTER_FILE_HPP

#include "rvjit/implementation/rv64/isa_traits.hpp"
#include "rvjit/interface/isa/buffer/architecture.hpp"
#include "rvjit/interface/isa/buffer/buffer.hpp"
#include "rvjit/interface/isa/types.hpp"
#include "rvjit/utils/utils.hpp"

namespace rvjit {
namespace rv64 {
namespace buffer {

/// The default general purpose register list (temp, arguments, and saved)
///
/// @note The int list includes all general purpose registers
/// @note The `a0` register often carries the JIT kernel arguments, and so
///     it is removed from the pool unless requested with `include_a0`
///
/// @details The default list features 27 registers, reordered so every
///     caller-saved register precedes every callee-saved one
struct int_register_file_t {
    /// The default list, retaining `a0` only if `include_a0`
    ///
    /// @param include_a0 Whether `a0` is part of the list
    static rvjit::buffer::buffer_file_t standard(bool include_a0 = false) {
        int n = 0;
        const Reg *ints = default_list_(n);
        rvjit::buffer::buffer_t regs(ints, ints + n);
        if (!include_a0) regs.clear(reg_name(Xbyak_riscv::a0));
        return rvjit::buffer::buffer_file_t(regs, order());
    }

    /// The allocation order over every general purpose register
    ///
    /// @details Lists temporaries, arguments, then saved registers, followed
    ///     by the remaining registers; every caller-saved id precedes every
    ///     callee-saved one
    static const rvjit::buffer::order_t &order() {
        static const rvjit::buffer::order_t o = [] {
            // The default list, then every remaining name in ascending order
            int n = 0;
            const Reg *ints = default_list_(n);
            rvjit::buffer::order_t head;
            for (int i = 0; i < n; ++i)
                head.push_back(reg_name(ints[i]));
            rvjit::buffer::buffer_t everything;
            for (int i = 0; i < rvjit::buffer::buffer_t::MAXN; ++i)
                everything.set(i);
            return reorder_caller_saved_first(everything.ordered(head));
        }();
        return o;
    }

    /// Reorders `regs` so every caller-saved id precedes every callee-saved
    /// id
    ///
    /// @param regs The registers reordered
    ///
    /// @details Each group keeps its own relative order from `regs`
    static rvjit::buffer::order_t reorder_caller_saved_first(
            const rvjit::buffer::order_t &regs) {
        rvjit::buffer::order_t r;
        for (int i = 0; i < regs.size(); ++i)
            if (!is_callee_saved_gpr(regs[i])) r.push_back(regs[i]);
        for (int i = 0; i < regs.size(); ++i)
            if (is_callee_saved_gpr(regs[i])) r.push_back(regs[i]);
        return r;
    }

private:
    static const Reg *default_list_(int &n) {
        using namespace Xbyak_riscv;
        static const Reg ints[]
                = {t0, t1, t2, t3, t4, t5, t6, a0, a1, a2, a3, a4, a5, a6, a7,
                        s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11};
        n = sizeof(ints) / sizeof(ints[0]);
        return ints;
    }
};

} // namespace buffer
} // namespace rv64
} // namespace rvjit

#endif
