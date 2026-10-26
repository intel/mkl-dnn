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

#ifndef RVJIT_IMPLEMENTATION_RV64_ISA_TRAITS_HPP
#define RVJIT_IMPLEMENTATION_RV64_ISA_TRAITS_HPP

#include <cstdint>
#include <type_traits>

#ifndef XBYAK_RISCV_V
#define XBYAK_RISCV_V 1
#endif

#include "xbyak_riscv/xbyak_riscv.hpp"

#include "rvjit/interface/isa/isa_traits.hpp"
#include "rvjit/interface/isa/register.hpp"
#include "rvjit/interface/isa/types.hpp"

namespace rvjit {

using codegen_t = Xbyak_riscv::CodeGenerator;
using Reg = Xbyak_riscv::Reg;
using FReg = Xbyak_riscv::FReg;
using VReg = Xbyak_riscv::VReg;
using Label = Xbyak_riscv::Label;
using LMUL = Xbyak_riscv::LMUL;
using SEW = Xbyak_riscv::SEW;
using VM = Xbyak_riscv::VM;
using VTA = Xbyak_riscv::VTA;
using VMA = Xbyak_riscv::VMA;

using Xbyak_riscv::sp; // The ISA stack pointer register
using Xbyak_riscv::zero; // The ISA zero register

template <typename CG>
struct isa_traits<CG,
        typename std::enable_if<std::is_base_of<codegen_t, CG>::value>::type> {
    using codegen_t = rvjit::codegen_t;
};

namespace rv64 {

/// The RV64 definitions shared by every native type it describes
struct isa_traits_base_t {
    /// The interface name of the general-purpose register `idx`
    ///
    /// @param idx The native register index
    static name_t int_name(int idx) {
        if (idx == 0) return zero_name;
        if (idx == 2) return stack_name;
        return static_cast<name_t>(idx);
    }

    /// The native index of the general-purpose register named `n`
    ///
    /// @param n The interface name
    ///
    /// @note An absent register is the zero register
    static int int_index(name_t n) {
        if (n == no_name || n == zero_name) return 0;
        if (n == stack_name) return 2;
        return n;
    }

    /// The native index of the floating-point or vector register named `n`
    ///
    /// @param n The interface name
    ///
    /// @note An absent register is register 0
    static int index(name_t n) { return n == no_name ? 0 : n; }

    /// The vector mask operand of a predicate
    ///
    /// @param p The predicate
    static VM vm(predicate_t p) {
        return p == predicate_t::masked ? VM::masked : VM::unmasked;
    }

    /// Whether `id` names a callee-saved (`s`-register) integer register
    ///
    /// @param id The register name
    static bool is_callee_saved(name_t id) {
        // s0=x8, s1=x9, s2..s11=x18..x27
        static constexpr uint32_t s_mask = (1u << 8) | (1u << 9) | (1u << 18)
                | (1u << 19) | (1u << 20) | (1u << 21) | (1u << 22) | (1u << 23)
                | (1u << 24) | (1u << 25) | (1u << 26) | (1u << 27);
        return id < 32 && (s_mask & (1u << id)) != 0;
    }
};

/// Typed handle to the RV64 code generator
///
/// @note Converts to and from the interface's opaque `rvjit::emitter_t`
class emitter_t {
public:
    /// A detached handle, only fit for objects that never emit
    emitter_t() = default;

    /// Creates a handle to `cg`, which must outlive it
    ///
    /// @param cg The code generator instructions are emitted through
    explicit emitter_t(codegen_t &cg) : cg_(&cg) {}

    /// Recovers the code generator an interface handle refers to
    ///
    /// @param e The interface handle
    ///
    /// @pre `e` is detached or was created from a `codegen_t`
    emitter_t(const rvjit::emitter_t &e) : cg_(e.as<codegen_t>()) {}

    /// The interface handle to the same code generator
    operator rvjit::emitter_t() const {
        return cg_ ? rvjit::emitter_t(*cg_) : rvjit::emitter_t();
    }

    /// The code generator, for emitting instructions
    codegen_t *operator->() const { return cg_; }

    /// The code generator
    codegen_t &cg() const { return *cg_; }

private:
    codegen_t *cg_ = nullptr;
};

} // namespace rv64

template <>
struct isa_traits<Reg> : rv64::isa_traits_base_t {
    static constexpr resource_t resource = resource_t::integers;
    static name_t name(const Reg &r) { return int_name(r.getIdx()); }
    static Reg make(name_t n) { return Reg(int_index(n)); }
};

template <>
struct isa_traits<FReg> : rv64::isa_traits_base_t {
    static constexpr resource_t resource = resource_t::floats;
    static name_t name(const FReg &r) { return name_t(r.getIdx()); }
    static FReg make(name_t n) { return FReg(index(n)); }
};

template <>
struct isa_traits<VReg> : rv64::isa_traits_base_t {
    static constexpr resource_t resource = resource_t::vectors;
    static name_t name(const VReg &r) { return name_t(r.getIdx()); }
    static VReg make(name_t n) { return VReg(index(n)); }
};

/// Whether `id` names an RV64 callee-saved (`s`-register) integer register
///
/// @param id The register name
inline bool is_callee_saved_gpr(name_t id) {
    return rv64::isa_traits_base_t::is_callee_saved(id);
}

/// The number of vector registers in a group for a given LMUL setting
///
/// @param m The LMUL setting
inline int vgroup_size(const LMUL &m) {
    switch (m) {
        case LMUL::m1: return 1;
        case LMUL::m2: return 2;
        case LMUL::m4: return 4;
        case LMUL::m8: return 8;
        default: return 1;
    }
}

} // namespace rvjit

#endif
