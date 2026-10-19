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


#ifndef GEMMSTONE_GENERATOR_PIECES_LAYOUT_UTILS_HPP
#define GEMMSTONE_GENERATOR_PIECES_LAYOUT_UTILS_HPP

#include "internal/ngen_includes.hpp"
#include "gemmstone/type.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "state.hpp"

GEMMSTONE_NAMESPACE_START


// Get an element's linear offset in a tiled layout (in registers or in memory).
int untile(Type T, const MatrixAddressing &atype, int component, int i, int j, int r, int c, int tileR, int tileC, bool reverse = false);

static inline int untile(Type T, const MatrixAddressing &atype, const RegisterBlock &block, int r, int c, int tileR, int tileC, bool reverse = false) {
    return untile(T, atype, block.component, block.offsetR, block.offsetC, r, c, tileR, tileC, reverse);
}

static inline int untile(Type T, const MatrixAddressing &atype, const RegisterBlock &block, int r = 0, int c = 0, bool reverse = false) {
    return untile(T, atype, block, r, c, atype.tileR, atype.tileC, reverse);
}

static inline int untile(Type T, const MatrixAddressing &atype, int component, int i, int j, int r = 0, int c = 0, bool reverse = false) {
    return untile(T, atype, component, i, j, r, c, atype.tileR, atype.tileC, reverse);
}

// Return the number of matrix elements in a tile of size (r,c) that are
//   guaranteed to be consecutive in memory.
int consecutiveElements(int r, int c, const MatrixAddressing &atype);

// Get minimum row/column granularity for a matrix in memory.
//   (i.e. the number of rows/columns must be a multiple of rgran/cgran, respectively.)
void getGranularities(const MatrixAddressing &atype, int &rgran, int &cgran);

// Detect crosspacked cases that should be converted to equivalent transposed layouts.
static inline bool isLargeCrosspack(Type T, int crosspack) {
    return (crosspack * T > 4) && (crosspack > 1);
}

// Check if a matrix will arrive column-major in registers, without creating a RegisterBlock.
static inline bool isRegisterColMajor(Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) {
    return isColMajor(atype.layout) ^ isTransposing(astrategy.accessType) ^ isLargeCrosspack(T, atype.crosspack);
}

// Check if pseudo-block (rather than true block) access is required for this block.
bool needsPseudoblock(ngen::HW hw, Type T, int r, int c,
                      const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                      bool writable, bool masked);

// Allocate address registers for a layout.
void allocAddrRegs(ngen::HW hw, std::vector<ngen::RegisterRange> &addrRegs, const RegisterLayout &layout,
                   CommonState &state, ngen::Bundle hint = ngen::Bundle());

// Attempt to allocate address registers for a layout. Returns true if successful.
bool tryAllocAddrRegs(ngen::HW hw, std::vector<ngen::RegisterRange> &addrRegs, const RegisterLayout &layout,
                      CommonState &state, ngen::Bundle hint = ngen::Bundle());

// Find the subregister offset containing the first address of a header.
int getAddr0Offset(const RegisterBlock &block, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy);

// Get a subregister containing the (shifted) address of the (0,0) entry of a layout.
ngen::Subregister getOriginAddr(const RegisterLayout &layout, const std::vector<ngen::RegisterRange> &addrRegs,
                                int *shiftOut = nullptr);

// Check if a block occupies a contiguous portion of registers in the given GRFMultirange.
// If so, return index of the block's first register in the range.
int contiguityCheck(ngen::HW hw, const RegisterBlock &block, const GRFMultirange &range);

// Retrieve the subrange of a given GRFMultirange holding the matrix data from a given block.
GRFMultirange subrange(GRFMultirange r, ngen::HW hw, Type T, const RegisterBlock &block);

// Check if descriptor-based remainders are available for the given set of parameters.
//   Returns zero if not, otherwise the maximum fragment size.
int checkDescriptorRemainder(ngen::HW hw, Type T, int r, int c, bool column, bool writable,
                             const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
