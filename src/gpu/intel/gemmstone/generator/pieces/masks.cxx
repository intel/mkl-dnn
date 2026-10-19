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


#include "alloc_utils.hpp"
#include "gemmstone/generator.hpp"
#include "state_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using namespace ngen::utils;
using std::vector;


// Assign mask registers to a register layout.
// The assignments parameter is both input and output:
//     existing assignments will be reused if compatible, and new assignments
//     created as necessary.
template <HW hw>
bool Generator<hw>::assignMasks(RegisterLayout &layout, LoopType rloop, LoopType cloop, vector<MaskAssignment> &assignments,
                                const CommonStrategy &strategy, CommonState &state, bool retryVirtual,
                                const vector<MaskAssignment> *existing)
{
    std::vector<VirtualFlag*> updated;

    // Loop through layout, collecting masks.
    //  - For each unique mask+loop+offset, allocate an index (flag reg)
    //  - Store new assignment if unique and update flag reg in layout.
    bool success = true, retry = false;
    do {
        auto nassignOriginal = int(assignments.size());
        bool outOfRegs = retry = false;

        for (RegisterBlock &block: layout) {
            for (bool row: {false, true}) {
                MaskAssignment thisAssignment;

                const auto &mask = row ? block.rowMask : block.colMask;
                auto loop        = row ? rloop         : cloop;
                auto &flag = block.flag[int(!row)];

                if (flag || (loop == LoopNone))
                    continue;
                else if (mask) {
                    thisAssignment.mask = mask;
                    thisAssignment.offset = row ? block.offsetR : block.offsetC;
                    thisAssignment.var = loop;
                } else {
                    flag.clear();
                    continue;
                }

                // Look for compatible mask.
                bool gotMask = false;
                auto checkCompatible = [&](const MaskAssignment &a) {
                    if (!gotMask && a.compatible(thisAssignment)) {
                        flag = a.flag;
                        updated.push_back(&flag);
                        gotMask = true;
                    }
                };

                for (auto &a: assignments)
                    checkCompatible(a);
                if (existing) for (auto &a: *existing)
                    checkCompatible(a);

                if (!gotMask) {
                    // No compatible mask, so make a new assignment.
                    thisAssignment.flag = state.allocVFlag(hw, (block.simdSize + 0xF) >> 4);
                    assignments.push_back(thisAssignment);
                    if (state.raVFlag.isVirtual(thisAssignment.flag) && !state.vflagsEnabled()) {
                        outOfRegs = true;
                        break;
                    }
                    flag = thisAssignment.flag;
                    updated.push_back(&flag);
                }
            }
        }

        if (outOfRegs) {
            // Not enough (virtual) flag registers! Free any masks we added to the list.
            safeReleaseMaskAssignments(assignments, state, nassignOriginal);
            if (retryVirtual && !state.vflagsEnabled()) {
                status << "Not enough flag registers available. Retrying with virtual flags." << status_stream::endl;
                allocVFlagStorage(strategy, state);
                retry = true;
            } else {
                status << "Not enough flag registers available." << status_stream::endl;
                success = false;
            }

            for (auto fptr: updated) fptr->clear();
            updated.clear();
        }
    } while (retry);

    return success;
}

// Output code for loading a mask into a flag register.
template <HW hw>
void Generator<hw>::loadMask(MaskAssignment assignment, Subregister index, const CommonStrategy &strategy, CommonState &state, int offset)
{
    auto flagIdx = assignment.flag;
    RegData flag = getMaskFlag(hw, flagIdx, state);

    if (assignment.mask.fixed.isFixed) {
        // Load fixed mask. Easy.
        mov(1, flag, uint16_t(assignment.mask.fixed.value));
    } else {
        // Load a variable mask, which requires some minor bit-twiddling.
        auto &vmask = assignment.mask.variable;

        uint32_t rsizeScaled = std::max<uint32_t>(vmask.rsize >> vmask.rshift, 1);
        uint32_t maskLen = vmask.bitRep * vmask.maskRep * rsizeScaled;
        uint32_t fullMask = (uint64_t(1) << maskLen) - 1;
        uint32_t rep1Mask = (uint64_t(1) << (vmask.bitRep * rsizeScaled)) - 1;
        uint32_t repMultiplier = fullMask / rep1Mask;

        auto flagType = flag.getType();
        auto mask0Type = getBytes(flagType) >= 4 ? DataType::uq : flagType;

        if (vmask.rsize == 1) {
            // Simple threshold comparison.
            offset += assignment.offset;
            offset <<= vmask.rshift;
            if (flag.isARF())
                cmp(int(maskLen) | gt | static_cast<FlagRegister &>(flag), index, offset);
            else {
                auto sflag = flag;
                sflag.setType(flagType == DataType::ud ? DataType::d : DataType::w);
                add(1 | sat, sflag, -index, offset);
                asr(1, sflag, sflag, getBytes(flagType) * 8 - 1);
            }
        } else {
            auto temp = state.ra.alloc_sub(flagType, getHint(HintType::Bank0));
            auto mask0 = state.ra.alloc_sub(mask0Type, getHint(HintType::Bank1));
            auto mask = mask0.reinterpret(0, flagType);
            auto mindex = index;
            auto rdivide = 1 << vmask.rshift;

            if (vmask.rshift) {
                add(1 | sat, temp, mindex, -offset + rdivide - 1);
                shr(1, temp, temp, uint16_t(vmask.rshift));
                mindex = temp;
                offset = 0;
            }
            if (vmask.bitRep > 1) {
                if (offset > 0) {
                    add(1 | sat, temp, mindex, -offset);
                    mindex = temp;
                    offset = 0;
                }
                mulConstant(1, temp, mindex, vmask.bitRep);
                mindex = temp;
            }
            uint16_t tshift = vmask.bitRep * (rsizeScaled + div_up(assignment.offset + offset, rdivide));
            add(1 | sat, temp, -mindex, tshift);
            if (tshift >= 32)
                min_(1, temp, temp, vmask.bitRep * rsizeScaled);            // Ensure shift count doesn't overflow.
            emov(1, mask0, rep1Mask, strategy, state);
            if (vmask.maskRep == 1) {
                bool twoStage = (!flag.isARF() && getBytes(mask0Type) > 4);
                auto flag1 = twoStage ? mask0 : flag;
                vmask.reverse ? shl(1, flag1, mask0, temp)
                              : shr(1, flag1, mask0, temp);
                if (twoStage) mov(1, flag, mask);
            } else {
                vmask.reverse ? stub() // need shl + and
                              : shr(1, mask0, mask0, temp);
                if (repMultiplier & 0x10000)
                    mov(1, mask.uw(1), mask.uw(0));
                mul(1, flag, mask, uint16_t(repMultiplier));
            }

            state.ra.safeRelease(temp);
            state.ra.safeRelease(mask0);
        }
    }
}

// Output code for loading all masks in a mask assignment list to flag registers.
template <HW hw>
void Generator<hw>::loadMasks(const vector<MaskAssignment> &assignments, Subregister (&indices)[3],
                              const CommonStrategy &strategy, CommonState &state, int start)
{
    for (size_t an = start; an < assignments.size(); an++) {
        auto &a = assignments[an];
        auto av = static_cast<int>(a.var);
        loadMask(a, indices[av], strategy, state);
    }
}

// Variant that allows additional constant offsets to be specified.
template <HW hw>
void Generator<hw>::loadMasks(const vector<MaskAssignment> &assignments, Subregister (&indices)[3], int (&offsets)[3],
                              const CommonStrategy &strategy, CommonState &state, int start)
{
    for (size_t an = start; an < assignments.size(); an++) {
        auto &a = assignments[an];
        auto av = static_cast<int>(a.var);
        loadMask(a, indices[av], strategy, state, offsets[av]);
    }
}

GEMMSTONE_NAMESPACE_END
