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


#ifndef GEMMSTONE_GENERATOR_PIECES_STATE_UTILS_HPP
#define GEMMSTONE_GENERATOR_PIECES_STATE_UTILS_HPP

#include "gemmstone/type.hpp"
#include "state.hpp"
#include "alloc_utils.hpp"

GEMMSTONE_NAMESPACE_START

// Release various m/n loop remainder variables.
void releaseFusedRemainders(GEMMState &state);
void releaseCoopRemainders(GEMMState &state);

// Allocate temporaries for emulated atomic add.
void allocEAtomicAddRegs(ngen::HW hw, Type T, const RegisterLayout &layout,
                         const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy, CommonState &state,
                         const ngen::FlagRegister &flag = ngen::FlagRegister());

// Free temporaries for emulated atomic add.
void freeEAtomicAddRegs(CommonState &state, const ngen::FlagRegister &flag = ngen::FlagRegister());


// Release all masks in a mask assignment. If 'start' is specified, only the masks
//  at index 'start' and above will be released.
void releaseMaskAssignments(std::vector<MaskAssignment> &assignments, CommonState &state, int start = 0);

// Reclaim mask assignments after previous release.
void reclaimMaskAssignments(std::vector<MaskAssignment> &assignments, CommonState &state, int start = 0);

// Release all masks in a mask assignment and clear assignments.
void safeReleaseMaskAssignments(std::vector<MaskAssignment> &assignments, CommonState &state, int start = 0);

// Release the index vector.
static inline void releaseIndexVec(CommonState &state) {
    safeReleaseRanges(state.indexVec, state);
    state.ivEntries = 0;
}

// Release an LDMultiples object.
static inline void releaseLDMultiples(LDMultiples &multiples, CommonState &state) {
    state.ra.release(multiples.range);
    multiples = LDMultiples{};
}

// Get a virtual flag register, either as a flag register or in GRF.
ngen::RegData getMaskFlag(ngen::HW hw, VirtualFlag vflag, CommonState &state);

// Notify k loop that the all-purpose flag (flagAP) has been overwritten.
static inline void kLoopModifiedFlagAP(GEMMState &state) {
    state.lastThresh = 0;
}

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
