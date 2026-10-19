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


#include "allocators.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;

FlagRegister VirtualFlag::toPhysical(ngen::HW hw) const
{
    if (n == 2)
        return FlagRegister(idx >> 1);
    else
        return FlagRegister::createFromIndex(idx);
}

VirtualFlag VirtualFlagAllocator::allocVirtual(int n)
{
    if (!free)
        throw out_of_registers_exception();
    if (n > 2)
        stub();

    uint64_t bmask = free;
    if (n == 2)
        bmask = (bmask & (bmask >> 1)) & 0x5555555555555555;
    int base = ngen::utils::bsf(bmask);

    VirtualFlag vflag{base, n};
    claim(vflag);

    return vflag;
}

FlagRegister VirtualFlagAllocator::tryAlloc(int n)
{
    auto vflag = allocVirtual(n);
    if (isVirtual(vflag)) {
        release(vflag);
        return FlagRegister{};
    }

    lock(vflag);

    return vflag.toPhysical(hw);
}

FlagRegister VirtualFlagAllocator::alloc(int n)
{
    auto flag = tryAlloc(n);
    if (flag.isInvalid())
        throw out_of_registers_exception();

    return flag;
}

FlagRegister VirtualFlagAllocator::allocSubreg0()
{
    auto flag = alloc(2);
    release(FlagRegister{flag.getARFBase(), 1});
    return FlagRegister{flag.getARFBase(), 0};
}

FlagRegister VirtualFlagAllocator::assignPhysical(VirtualFlag vflag)
{
    VirtualFlag pflag;

    // Starting at nextPhys, find an unlocked flag register.
    // Allocate it temporarily if it's not allocated.
    for (int i = nextPhys; i < nextPhys + nflag; i++) {
        if (i & (vflag.n - 1)) continue;
        int idx = i % nflag;
        auto msk = mask(idx, vflag.n);
        if ((locked & msk) == 0) {
            vtemps |= (free & msk);
            free &= ~vtemps;
            nextPhys = (idx + vflag.n) % nflag;
            pflag = VirtualFlag{idx, vflag.n};
            break;
        }
    }

    if (!pflag)
        throw out_of_registers_exception();

    return pflag.toPhysical(hw);
}

bool VirtualFlagAllocator::lock(VirtualFlag vflag, bool allowAlreadyLocked)
{
    bool wasLocked = isLocked(vflag);
    if (wasLocked && !allowAlreadyLocked)
        stub("Illegally locking an already-locked flag register");
    locked |= mask(vflag);
    return wasLocked;
}

bool VirtualFlagAllocator::canLock(int n) const
{
    uint32_t unlocked = ~locked & ((1 << nflag) - 1);
    if (n == 2)
        unlocked = (unlocked & (unlocked >> 1)) & 0x55;
    return (unlocked != 0);
}

void VirtualFlagAllocator::freeUnlocked()
{
    uint32_t unlocked = ~locked & ((1 << nflag) - 1);
    free |= unlocked;
}

TokenAllocator::TokenAllocator(HW hw, int grfCount)
{
    free = (1ull << tokenCount(hw, grfCount)) - 1;
}

int8_t TokenAllocator::tryAlloc()
{
    if (free) {
        int8_t token = ngen::utils::bsf(free);
        free &= ~(1 << token);
        return token;
    } else
        return -1;
}

GEMMSTONE_NAMESPACE_END
