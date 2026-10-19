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

#include "state_utils.hpp"
#include "hw_utils.hpp"

using namespace ngen;
using std::vector;

GEMMSTONE_NAMESPACE_START

// Release fused remainder-related state variables.
void releaseFusedRemainders(GEMMState &state)
{
    state.ra.safeRelease(state.remFusedStorage);
    state.remaindersFused[LoopM] = Subregister{};
    state.remaindersFused[LoopN] = Subregister{};
}

void releaseCoopRemainders(GEMMState &state)
{
    for (LoopType loop: {LoopM, LoopN, LoopK})
        if (state.remaindersCoop[loop] != state.remainders[loop])
            state.ra.safeRelease(state.remaindersCoop[loop]);
}

// Allocate temporary registers for emulating atomic addition.
void allocEAtomicAddRegs(HW hw, Type T, const RegisterLayout &layout,
                         const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, CommonState &state,
                         const FlagRegister &flag)
{
    if (hasNativeAtomicAdd(hw, T.real(), atype, astrategy)) return;

    int maxNReg = 0;
    for (const auto &block : layout)
        maxNReg = std::max(maxNReg, block.nregs());

    if (maxNReg == 0) return;

    state.eatomicAddRegs[0] = state.ra.alloc_range(maxNReg * 2);
    state.eatomicAddRegs[1] = state.ra.alloc_range(maxNReg);
    state.vflagEAtomicAdd = flag.isValid() ? VirtualFlag(flag, hw)
                                           : state.allocVFlag(hw);
}

void freeEAtomicAddRegs(CommonState &state, const FlagRegister &flag)
{
    state.ra.safeRelease(state.eatomicAddRegs[0]);
    state.ra.safeRelease(state.eatomicAddRegs[1]);
    if (flag.isInvalid())
        state.raVFlag.release(state.vflagEAtomicAdd);
}

void releaseMaskAssignments(vector<MaskAssignment> &assignments, CommonState &state, int start)
{
    for (size_t an = start; an < assignments.size(); an++)
        state.raVFlag.release(assignments[an].flag);

    state.wipeActiveVFlags();
}

void reclaimMaskAssignments(vector<MaskAssignment> &assignments, CommonState &state, int start)
{
    for (size_t an = start; an < assignments.size(); an++)
        state.raVFlag.claim(assignments[an].flag);
}

void safeReleaseMaskAssignments(vector<MaskAssignment> &assignments, CommonState &state, int start)
{
    releaseMaskAssignments(assignments, state, start);
    assignments.resize(start);
}

RegData getMaskFlag(HW hw, VirtualFlag vflag, CommonState &state)
{
    if (state.vflagsEnabled()) {
        return state.vflagStorage.sub(hw, vflag.idx, DataType::uw)
                                 .reinterpret(0, vflag.n == 2 ? DataType::ud : DataType::uw);
    } else if (!state.raVFlag.isVirtual(vflag)) {
        auto pflag = vflag.toPhysical(hw);
        state.usePhysicalFlag(pflag, hw);
        return pflag;
    } else
        stub("Need virtual flag registers");
}

GEMMSTONE_NAMESPACE_END
