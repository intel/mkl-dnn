/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef GEMMSTONE_GENERATOR_PIECES_ALLOCATORS_HPP
#define GEMMSTONE_GENERATOR_PIECES_ALLOCATORS_HPP

#include <cstdint>

#include "internal/ngen_includes.hpp"

GEMMSTONE_NAMESPACE_START

// Allocator for SWSB tokens.
class TokenAllocator {
public:
    TokenAllocator(ngen::HW hw, int grfCount = 128);

    int8_t tryAlloc();
    void release(int8_t token)                      { free |= (1u << token); }
    void safeRelease(int8_t &token)                 { if (token >= 0) release(token); token = -1; }

protected:
    uint32_t free;
};

// Virtualized flag register.
struct VirtualFlag {
    uint8_t idx : 6;
    uint8_t n : 2;

    constexpr VirtualFlag() : idx(0), n(0) {}
    VirtualFlag(const ngen::FlagRegister &flag, ngen::HW hw) {
        idx = flag.index(hw);
        n = flag.getBytes() >> 1;
    }
    explicit constexpr VirtualFlag(int idx_, int n_ = 1) : idx(idx_), n(n_) {}

    ngen::FlagRegister toPhysical(ngen::HW hw = ngen::HW::Unknown) const;

    friend inline bool operator==(VirtualFlag vf1, VirtualFlag vf2) { return vf1.idx == vf2.idx && vf1.n == vf2.n; }
    friend inline bool operator!=(VirtualFlag vf1, VirtualFlag vf2) { return !(vf1 == vf2); }

    bool operator!() const          { return (idx == 0) && (n == 0); }
    explicit operator bool() const  { return !!*this; }

    void clear()                    { *this = VirtualFlag(); }

    int getBytes() const            { return n << 1; }
};

// Allocator for virtual flag registers.
class VirtualFlagAllocator {
public:
    VirtualFlagAllocator(ngen::HW hw_) : hw(hw_), free(~uint64_t(0)) {
        nflag = ngen::FlagRegister::subcount(hw_);
    }

    VirtualFlag allocVirtual(int n = 1);
    ngen::FlagRegister alloc(int n = 1);
    ngen::FlagRegister allocSubreg0();
    ngen::FlagRegister tryAlloc(int n = 1);

    void claim(VirtualFlag vflag)                   { free &= ~mask(vflag); }
    void claim(const ngen::FlagRegister &reg)       { claim(VirtualFlag(reg, hw)); }
    void release(VirtualFlag vflag)                 { free |= mask(vflag); }
    void release(const ngen::FlagRegister &reg)     { release(VirtualFlag(reg, hw)); unlock(reg); }
    void safeRelease(VirtualFlag &vflag)            { if (vflag) release(vflag); vflag.clear(); }
    void safeRelease(ngen::FlagRegister &reg)       { if (reg.isValid()) release(reg); reg.invalidate(); }
    bool isFree(VirtualFlag vflag)            const { return !(~free & mask(vflag)); }
    bool isFree(const ngen::FlagRegister &reg) const { return isFree(VirtualFlag(reg, hw)); }

    bool isVirtual(VirtualFlag vflag)               { return (vflag.idx >= nflag); }
    bool isVirtual(const ngen::FlagRegister &reg)   { return isVirtual(VirtualFlag(reg, hw)); }

    bool lock(VirtualFlag vflag, bool allowAlreadyLocked = false);
    bool lock(const ngen::FlagRegister &reg, bool allowAlreadyLocked = false) { return lock(VirtualFlag(reg, hw), allowAlreadyLocked); }
    void unlock(VirtualFlag vflag)                  { locked &= ~mask(vflag); }
    void unlock(const ngen::FlagRegister &reg)      { unlock(VirtualFlag(reg, hw)); }
    bool isLocked(VirtualFlag vflag)          const { return !(~locked & mask(vflag)); }
    bool isLocked(const ngen::FlagRegister &reg) const { return isLocked(VirtualFlag(reg, hw)); }
    bool canLock(int n = 1) const;
    void freeUnlocked();
    void freeVFlagTempAllocs()                      { free |= vtemps; vtemps = 0; }

    ngen::FlagRegister assignPhysical(VirtualFlag vflag);

protected:
    ngen::HW hw;                    // GPU architecture
    uint64_t free;                  // Bitmask: free virtual flags
    uint8_t locked = 0;             // Bitmask: locked physical flags (= unavailable for vflag usage)
    uint8_t vtemps = 0;             // Bitmask: temporary allocations for physical assignments of virtual flags */
    uint8_t nextPhys = 0;           // Next physical flag to try for physical flag assignment.
    uint8_t nflag;                  // # of physical flags in HW.

    static uint64_t mask(VirtualFlag vflag)         { return mask(vflag.idx, vflag.n); }
    static uint64_t mask(int idx, int n)            { return (uint64_t(1) << (idx + n)) - (uint64_t(1) << idx); }
};

GEMMSTONE_NAMESPACE_END

#endif
