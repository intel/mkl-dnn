/*******************************************************************************
* Copyright 2026 Intel Corporation
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
#ifndef CPU_X64_XBYAK_REG_MANAGER_HPP
#define CPU_X64_XBYAK_REG_MANAGER_HPP

#include <cstdint>
#include <set>
#include <stdexcept>
#include <vector>
#include <type_traits>

#define XBYAK64
#define XBYAK_NO_OP_NAMES
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace Xbyak {
// Static definitions for different types of registers in relation to which family they belong to.
enum class RegFamily { GP, Vec, Opmask };

template <class RegT>
struct reg_family;

// General Purpose register families (8-bit, 16-bit, 32-bit, 64-bit)
template <>
struct reg_family<Reg8> {
    static constexpr RegFamily value = RegFamily::GP;
};
template <>
struct reg_family<Reg16> {
    static constexpr RegFamily value = RegFamily::GP;
};
template <>
struct reg_family<Reg32> {
    static constexpr RegFamily value = RegFamily::GP;
};
template <>
struct reg_family<Reg64> {
    static constexpr RegFamily value = RegFamily::GP;
};

// Vector register families (XMM, YMM, ZMM)
template <>
struct reg_family<Xmm> {
    static constexpr RegFamily value = RegFamily::Vec;
};
template <>
struct reg_family<Ymm> {
    static constexpr RegFamily value = RegFamily::Vec;
};
template <>
struct reg_family<Zmm> {
    static constexpr RegFamily value = RegFamily::Vec;
};

// Opmask registers (k0-k7)
template <>
struct reg_family<Opmask> {
    static constexpr RegFamily value = RegFamily::Opmask;
};

class RegPoolManager {
public:
    // Constructor - detects APX and AVX-512 support and sets up register pools accordingly
    RegPoolManager() {
        Xbyak::util::Cpu cpu;
        uint64_t xcr0 = 0;
        if (cpu.has(Xbyak::Util::Cpu::tOSXSAVE)) {
            xcr0 = cpu.getXfeature();
        }

        // Detect APX for extended GP registers (r16-r31)
        if (cpu.has(Xbyak::Util::Cpu::tAPX_F)) {
            has_apx_ = ((xcr0 >> 19) & 1) == 1; // Check if XCR0[19] is set for APX support
        }
        max_gp_reg_idx_ = has_apx_ ? 31 : 15;

        // Detect AVX-512 for extended vector registers (zmm16-zmm31)
        if (cpu.has(Xbyak::Util::Cpu::tAVX512F)) {
            has_avx512_ = ((xcr0 >> 7) & 1) == 1; // Check if XCR0[7] is set for AVX-512 support
        }
        max_vec_reg_idx_ = has_avx512_ ? 31 : 15;

        // If APX is available, add r16-r31 to the free pool
        // APX registers r16-r31 are caller-saved (call-clobbered)
        if (has_apx_) {
            for (int i = 16; i <= 31; ++i) {
                free_gp_regs.insert(i);
            }
        }

        // If AVX-512 is available, add zmm16-zmm31 to the free pool
        // Extended vector registers are caller-saved (call-clobbered)
        if (has_avx512_) {
            for (int i = 16; i <= 31; ++i) {
                free_vec_regs.insert(i);
            }
        }
    }

    // Usage:
    // Reg64 rax = rm.alloc<Reg64>(); // Allocate next available 64-bit register (freed by user)
    // rm.free(rax);                  // Free rax

    // register allocation method - accepts int to specify an unused reg, or no arg to get next free reg
    template <class RegT>
    RegT alloc() {
        switch (reg_family<RegT>::value) {
            case RegFamily::GP: {
                const int idx = next_gp_idx();
                gp_reg(idx);
                return RegT(idx);
            }
            case RegFamily::Vec: {
                const int idx = next_vec_idx();
                vec_reg(idx);
                return RegT(idx);
            }
            case RegFamily::Opmask: {
                const int idx = next_opmask_idx();
                opmask_reg(idx);
                return RegT(idx);
            }
            default: throw std::runtime_error("Unknown register family");
        }
    }

    template <class RegT>
    RegT alloc(int idx) {
        switch (reg_family<RegT>::value) {
            case RegFamily::GP: gp_reg(idx); return RegT(idx);
            case RegFamily::Vec: vec_reg(idx); return RegT(idx);
            case RegFamily::Opmask: opmask_reg(idx); return RegT(idx);
            default: throw std::runtime_error("Unknown register family");
        }
    }

    // takes register object and moves it from in use to free set
    template <class RegT>
    void free(RegT reg) {
        const int idx = reg.getIdx();
        switch (reg_family<RegT>::value) {
            case RegFamily::GP: release_gp(idx); break;
            case RegFamily::Vec: release_vec(idx); break;
            case RegFamily::Opmask: release_opmask(idx); break;
            default: throw std::runtime_error("Unknown register family");
        }
    }

    // getter methods - return vectors of register indices representing a set
    std::vector<int> get_free_gps() const {
        return make_index_vector(free_gp_regs);
    }
    std::vector<int> get_in_use_gps() const {
        return make_index_vector(in_use_gp);
    }
    std::vector<int> get_preserved_gps() const {
        return make_index_vector(preserved_gp);
    }
    std::vector<int> get_used_gps() const { return make_index_vector(used_gp); }

    std::vector<int> get_free_vecs() const {
        return make_index_vector(free_vec_regs);
    }
    std::vector<int> get_in_use_vecs() const {
        return make_index_vector(in_use_vec);
    }
    std::vector<int> get_preserved_vecs() const {
        return make_index_vector(preserved_vec);
    }
    std::vector<int> get_used_vecs() const {
        return make_index_vector(used_vec);
    }

    std::vector<int> get_free_opmasks() const {
        return make_index_vector(free_opmask_regs);
    }
    std::vector<int> get_in_use_opmasks() const {
        return make_index_vector(in_use_opmask);
    }
    std::vector<int> get_preserved_opmasks() const {
        return make_index_vector(preserved_opmask);
    }
    std::vector<int> get_used_opmasks() const {
        return make_index_vector(used_opmask);
    }

    // member function - add a register to the free pool of general registers
    void add_to_gp_pool(const Reg64 &reg) { add_to_gp_pool(reg.getIdx()); }
    void add_to_gp_pool(int idx) {
        if ((idx < 0 || idx > max_gp_reg_idx_))
            throw std::runtime_error("Register index out of range");
        const bool in_free = free_gp_regs.count(idx) != 0;
        const bool in_preserved = preserved_gp.count(idx) != 0;
        const bool in_use = in_use_gp.count(idx) != 0;
        if (in_free || in_preserved || in_use)
            throw std::runtime_error("Register already tracked");
        free_gp_regs.insert(idx);
    }

    // helper function - returns true if a register object is currently in the used set of registers
    template <class RegT>
    bool reg_in_use(const RegT &reg) const {
        return reg_in_use_idx(reg.getIdx(), reg_family<RegT>::value);
    }

    // helper functions - returns true if an index in a given family is in use
    bool gp_idx_in_use(int reg_idx) const {
        return reg_in_use_idx(reg_idx, RegFamily::GP);
    }
    bool vec_idx_in_use(int reg_idx) const {
        return reg_in_use_idx(reg_idx, RegFamily::Vec);
    }
    bool opmask_idx_in_use(int reg_idx) const {
        return reg_in_use_idx(reg_idx, RegFamily::Opmask);
    }

    // scoped register handling with RAII
    // usage: Reg64 r10 = rm.alloc<Reg64>(10);
    //        auto scoped = rm.makeScoped(r10);
    // or
    //        auto scoped_reg = rm.makeScoped(rm.alloc<Reg64>());
    // r10 & scoped_reg will free at end of scope when guards' dtors called.
    template <class Reg>
    class Scoped {
    public:
        explicit Scoped(RegPoolManager &rm, Reg r)
            // pointer to allocator, allocate scoped reg at construction & track, unowned = nullptr
            : rm_(&rm), reg_(r) {
            validate_scoped_reg(rm_, reg_);
        }

        // if object is owner of scoped reg and goes out of scope, deallocate
        ~Scoped() {
            if (!rm_) return;
            rm_->free(reg_);
        }

        // disable copy - scoped regs are move only to avoid ownership/double free issues as per RAII
        Scoped(const Scoped &) = delete;
        Scoped &operator=(const Scoped &) = delete;

        // move constructor - used when scoped regs initialised from rvalue (incl. std::move)
        Scoped(Scoped &&other) noexcept : rm_(other.rm_), reg_(other.reg_) {
            other.rm_ = nullptr; // set previous owner to no longer own
        }

        // expose underlying register for implicit use in JIT helpers
        operator const Reg &() const noexcept { return reg_; }
        const Reg &get() const noexcept { return reg_; }

    private:
        RegPoolManager *rm_
                = nullptr; // pointer to allocator, initialised as nullptr
        Reg reg_ {};
    };

    // helper factory - calls Scoped ctor
    template <class Reg>
    inline Scoped<Reg> makeScoped(Reg r) & {
        return Scoped<Reg>(*this, r);
    }

    // helper methods to query APX support
    bool has_apx() const { return has_apx_; }
    int max_gp_registers() const { return max_gp_reg_idx_ + 1; }

    // helper methods to query AVX-512 support
    bool has_avx512() const { return has_avx512_; }
    int max_vec_registers() const { return max_vec_reg_idx_ + 1; }

    // helper methods to return special registers as per x86-64 calling convention (System V AMD64 ABI)
    // Stack pointer: rsp
    inline Reg64 _stack_pointer() {
        used_gp.insert(4); // rsp is index 4
        return Reg64(4);
    }
    // Base pointer: rbp
    inline Reg64 _base_pointer() {
        used_gp.insert(5); // rbp is index 5
        return Reg64(5);
    }
    // Opmask k0: special mask register that means "unmasked" (no masking)
    // When k0 is used as a write mask, all elements are written (effectively no masking)
    inline Opmask _opmask_k0() {
        used_opmask.insert(0); // k0 is index 0
        return Opmask(0);
    }

private:
    // helper method - converts members of set to vector
    static inline std::vector<int> make_index_vector(const std::set<int> &set) {
        std::vector<int> indices;
        indices.reserve(set.size());
        for (int idx : set)
            indices.emplace_back(idx);
        return indices;
    }

    // helper method - checks reg in use before scoping
    template <class RegT>
    static void validate_scoped_reg(RegPoolManager *rm, RegT reg) {
        const RegFamily family = reg_family<RegT>::value;
        if (!rm->reg_in_use(reg))
            throw std::runtime_error(scoped_reg_error(family));
    }

    // helper switch case for error messages
    static const char *scoped_reg_error(RegFamily family) noexcept {
        switch (family) {
            case RegFamily::GP:
                return "Cannot create GP scoped reg for a register that is not "
                       "in use";
            case RegFamily::Vec:
                return "Cannot create Vec scoped reg for a register that is "
                       "not in use";
            case RegFamily::Opmask:
                return "Cannot create Opmask scoped reg for a register that is "
                       "not in use";
            default: return "Cannot create scoped reg for unknown family";
        }
    }

    // helper method - checks if a register index for a given family is currently in use
    bool reg_in_use_idx(int idx, RegFamily family) const {
        switch (family) {
            case RegFamily::GP:
                if (idx < 0 || idx > max_gp_reg_idx_) {
                    throw std::runtime_error("GP register index out of range");
                }
                return in_use_gp.find(idx) != in_use_gp.end();
            case RegFamily::Vec:
                if (idx < 0 || idx > max_vec_reg_idx_) {
                    throw std::runtime_error("Vec register index out of range");
                }
                return in_use_vec.find(idx) != in_use_vec.end();
            case RegFamily::Opmask:
                if (idx < 0 || idx > 7) {
                    throw std::runtime_error(
                            "Opmask register index out of range");
                }
                return in_use_opmask.find(idx) != in_use_opmask.end();
            default: throw std::runtime_error("Unknown register family");
        }
    }

    // helper method - finds next free register for a given family
    int next_gp_idx() const {
        if (!free_gp_regs.empty()) return *free_gp_regs.begin();
        if (!preserved_gp.empty()) return *preserved_gp.begin();
        throw std::runtime_error("No free GP registers available");
    }
    int next_vec_idx() const {
        if (!free_vec_regs.empty()) return *free_vec_regs.begin();
        if (!preserved_vec.empty()) return *preserved_vec.begin();
        throw std::runtime_error("No free Vec registers available");
    }
    int next_opmask_idx() const {
        if (!free_opmask_regs.empty()) return *free_opmask_regs.begin();
        if (!preserved_opmask.empty()) return *preserved_opmask.begin();
        throw std::runtime_error("No free Opmask registers available");
    }

    // tracking for in-use indices for a given register family
    void gp_reg(int idx) {
        if (reg_in_use_idx(idx, RegFamily::GP))
            throw std::runtime_error("Specified GP register currently in use");
        auto it = free_gp_regs.find(idx);
        auto pres_it = preserved_gp.find(idx);
        if (it != free_gp_regs.end()) {
            in_use_gp.insert(idx);
            free_gp_regs.erase(it);
            used_gp.insert(idx);
        } else if (pres_it != preserved_gp.end()) {
            in_use_gp.insert(idx);
            preserved_gp.erase(pres_it);
            used_gp.insert(idx);
        } else {
            throw std::runtime_error(
                    "Requested register not in free/preserved pools.");
        }
    }
    void vec_reg(int idx) {
        if (reg_in_use_idx(idx, RegFamily::Vec))
            throw std::runtime_error("Specified Vec register currently in use");
        auto it = free_vec_regs.find(idx);
        auto pres_it = preserved_vec.find(idx);
        if (it != free_vec_regs.end()) {
            in_use_vec.insert(idx);
            free_vec_regs.erase(it);
            used_vec.insert(idx);
        } else if (pres_it != preserved_vec.end()) {
            in_use_vec.insert(idx);
            preserved_vec.erase(pres_it);
            used_vec.insert(idx);
        } else {
            throw std::runtime_error(
                    "Requested register not in free/preserved/in-use sets.");
        }
    }
    void opmask_reg(int idx) {
        if (reg_in_use_idx(idx, RegFamily::Opmask))
            throw std::runtime_error(
                    "Specified Opmask register currently in use");
        auto it = free_opmask_regs.find(idx);
        auto pres_it = preserved_opmask.find(idx);
        if (it != free_opmask_regs.end()) {
            in_use_opmask.insert(idx);
            free_opmask_regs.erase(it);
            used_opmask.insert(idx);
        } else if (pres_it != preserved_opmask.end()) {
            in_use_opmask.insert(idx);
            preserved_opmask.erase(pres_it);
            used_opmask.insert(idx);
        } else {
            throw std::runtime_error(
                    "Requested opmask register not in free/preserved pools.");
        }
    }

    // member function - moves given index from in-use set to free set for given family
    void release_gp(int idx) {
        if ((idx < 0 || idx > max_gp_reg_idx_))
            throw std::runtime_error("Register index out of range");
        auto it = in_use_gp.find(idx);
        if (it == in_use_gp.end())
            throw std::runtime_error("GP register not in use");
        in_use_gp.erase(it);
        free_gp_regs.insert(idx);
    }
    void release_vec(int idx) {
        if ((idx < 0 || idx > max_vec_reg_idx_))
            throw std::runtime_error("Register index out of range");
        auto it = in_use_vec.find(idx);
        if (it == in_use_vec.end())
            throw std::runtime_error("Vec register not in use");
        in_use_vec.erase(it);
        free_vec_regs.insert(idx);
    }
    void release_opmask(int idx) {
        if ((idx < 0 || idx > 7))
            throw std::runtime_error("Opmask register index out of range");
        auto it = in_use_opmask.find(idx);
        if (it == in_use_opmask.end())
            throw std::runtime_error("Opmask register not in use");
        in_use_opmask.erase(it);
        free_opmask_regs.insert(idx);
    }

    // General Purpose registers (GP):
    // Indices: rax=0, rcx=1, rdx=2, rbx=3, rsp=4, rbp=5, rsi=6, rdi=7, r8-r15=8-15
    // Note: rsp (4) and rbp (5) are special and not typically allocated
#ifdef _WIN32
    // Windows x64 calling convention:
    // Volatile (caller-saved): rax, rcx, rdx, r8-r11 (7 registers)
    // Non-volatile (callee-saved): rbx, rbp, rdi, rsi, r12-r15 (8 registers)
    static const std::set<int> &base_free_gp() {
        static const std::set<int> s {0, 1, 2, 8, 9, 10, 11};
        return s;
    }
    static const std::set<int> &base_preserved_gp() {
        static const std::set<int> s {3, 6, 7, 12, 13, 14, 15};
        return s;
    }
#else
    // System V AMD64 ABI calling convention for Linux/macOS:
    // Call-clobbered (caller-saved): rax, rcx, rdx, rsi, rdi, r8-r11 (9 registers)
    // Call-preserved (callee-saved): rbx, r12-r15 (5 registers)
    static const std::set<int> &base_free_gp() {
        static const std::set<int> s {0, 1, 2, 6, 7, 8, 9, 10, 11};
        return s;
    }
    static const std::set<int> &base_preserved_gp() {
        static const std::set<int> s {3, 12, 13, 14, 15};
        return s;
    }
#endif

    std::set<int> used_gp;
    std::set<int> in_use_gp;
    std::set<int> free_gp_regs = base_free_gp();
    std::set<int> preserved_gp = base_preserved_gp();

    // Vector registers (XMM/YMM/ZMM):
    // - SSE/AVX/AVX2: 0-15 (xmm0-xmm15, ymm0-ymm15)
    // - AVX-512: 0-31 (xmm0-xmm31, ymm0-ymm31, zmm0-zmm31)
    // Note: Only include 0-15 by default; extended registers (16-31) are added in constructor if AVX-512 is detected
#ifdef _WIN32
    // Windows x64 calling convention:
    // Volatile (caller-saved): xmm0-xmm5 (6 registers)
    // Non-volatile (callee-saved): xmm6-xmm15 (10 registers)
    static const std::set<int> &base_free_vec() {
        static const std::set<int> s {0, 1, 2, 3, 4, 5};
        return s;
    }
    static const std::set<int> &base_preserved_vec() {
        static const std::set<int> s {6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return s;
    }
#else
    // System V AMD64 ABI (Linux/macOS):
    // Call-clobbered: all vector registers are caller-saved
    static const std::set<int> &base_free_vec() {
        static const std::set<int> s {
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        return s;
    }
    static const std::set<int> &base_preserved_vec() {
        static const std::set<int> s {};
        return s;
    }
#endif

    std::set<int> used_vec;
    std::set<int> in_use_vec;
    std::set<int> free_vec_regs = base_free_vec();
    std::set<int> preserved_vec = base_preserved_vec();

    // Opmask registers (k0-k7) for AVX-512
    // k0 has special meaning (unmasked), typically k1-k7 are used
    // All opmask registers are call-clobbered
    static const std::set<int> &base_free_opmask() {
        static const std::set<int> s {1, 2, 3, 4, 5, 6, 7};
        return s;
    }
    // No preserved opmask registers
    static const std::set<int> &base_preserved_opmask() {
        static const std::set<int> s {};
        return s;
    }

    std::set<int> used_opmask;
    std::set<int> in_use_opmask;
    std::set<int> free_opmask_regs = base_free_opmask();
    std::set<int> preserved_opmask = base_preserved_opmask();

    // APX feature support
    bool has_apx_ = false;
    int max_gp_reg_idx_ = 15; // 15 without APX, 31 with APX

    // AVX-512 feature support
    bool has_avx512_ = false;
    // 15 without (SSE/AVX/AVX2), 31 with AVX-512
    int max_vec_reg_idx_ = 15;
};

} // namespace Xbyak

#endif // CPU_X64_XBYAK_REG_MANAGER_HPP
