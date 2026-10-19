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

#ifndef GEMMSTONE_GENERATOR_PIECES_REGISTER_LAYOUT_HPP
#define GEMMSTONE_GENERATOR_PIECES_REGISTER_LAYOUT_HPP

#include <array>
#include <cstdint>

#include "internal/ngen_includes.hpp"

#include "gemmstone/problem.hpp"
#include "gemmstone/strategy.hpp"
#include "gemmstone/type.hpp"

#include "allocators.hpp"
#include "grf_multirange.hpp"


GEMMSTONE_NAMESPACE_START

struct MatrixAddressingStrategy;

// MaskInfo: logical description of a message mask, used to ensure in-bounds matrix accesses.
struct MaskInfo {
    union {
        struct {
            uint8_t isFixed : 1;  // = false (variable mask)
            uint8_t reverse : 1;  // True to reverse mask.
            uint8_t rshift : 6;   // Power of 2 by which to divide index before forming mask. Fractions are rounded up.
                                  // Note maskRep * bitRep * (rsize >> rshift) = # mask bits.
            uint8_t rsize;        // Maximum remainder value. (e.g. 16 if we need the last 4 bits of the index).
            uint8_t maskRep;      // # of repetitions of mask pattern.
            uint8_t bitRep;       // # of times each mask bit is repeated.
        } variable;
        struct {
            uint8_t isFixed : 1;  // = true (fixed mask)
            uint8_t _ : 7;
            uint8_t rsize;        // Maximum remainder value.
            uint16_t value;       // Mask value.
        } fixed;
        uint32_t raw;
    };

    MaskInfo() : fixed{true,0,0,0xFFFF} {}

    bool operator!()         const { return fixed.isFixed && fixed.value == 0xFFFF; }
    explicit operator bool() const { return !!*this; }

    static MaskInfo None() { return MaskInfo(); }

    friend bool operator==(const MaskInfo &i1, const MaskInfo &i2) {
        return i1.raw == i2.raw;
    }
    friend bool operator!=(const MaskInfo &i1, const MaskInfo &i2) {
        return !(i1 == i2);
    }
};

// Preferences for remainder handling.
enum RemainderOptions : uint8_t {
    AvoidFragment = 0,      // Avoid/allow making blocks that will need to be broken up
    AllowFragment = 1,      //  ("fragmented") during remainder handling.
    AllowDescriptors = 2,   // Allow indirect send descriptor-based remainder handling.
    AllowFragDesc = 3,      // Allow fragmentation and descriptors.
    NoFixedMasks = 4,       // Do not allow fixed masks.
    AllowFragDescNFM = 7,   // Allow fragmentation and descriptors, but no fixed masks
};

// RegisterBlock encapsulates a single matrix tile resident in registers,
//   along with information needed to move it to/from memory.
// Generally speaking RegisterBlocks map 1-1 to individual load/store/atomic instructions,
//   but some instructions may have multiple RegisterBlocks, e.g. 2D block arrays.
// It is also possible for RegisterBlocks not to be backed by memory, indicated by
//   a zero simdSize field.
struct RegisterBlock {
    /* Register layout information. */
    uint16_t nr = 0, nc = 0;    // Size of this block.
    uint16_t ld;                // Leading dimension, in elements.
    uint16_t offsetR, offsetC;  // Row and column offset within matrix block.
    uint8_t colMajor : 1;       // Is this block column-major? (columns stored consecutively inside each register)
    uint8_t splitComplex : 1;   // True if complex data split into successive real and imaginary parts.
    uint8_t byteGlue : 1;       // True if strided sub-byte data is unit stride within each byte.
    uint8_t : 5;
    uint8_t crosspack;          // Crosspack for this block (1 if none).
    uint8_t component;          // Component # for this block.
    int8_t cxComponent;         // Complex component # for this block (-1 if not complex or interleaved).
    uint16_t bytes;             // # of bytes in this block.
    uint16_t offsetBytes;       // Byte offset within register block.
    ngen::HW hw;                // GPU architecture.

    /* Load/store information. */
    uint8_t remainderR : 1;     // Row remaindering enabled?
    uint8_t remainderC : 1;     // Column remaindering enabled?
    uint8_t noRowsOK : 1;       // Can handle no rows (in mask/descriptor)?
    uint8_t noColsOK : 1;       // Can handle no columns (in mask/descriptor)?
    uint8_t descRemR : 1;       // Row remainders can be handled by changing the descriptor?
    uint8_t descRemC : 1;       // Column remainders can be handled by changing the descriptor?
    uint8_t descAssigned : 1;   // True if address registers have been assigned for this block's descriptors.
    uint8_t writable : 1;       // True if block is set up for writing.

    uint8_t ebytes;             // Size of element in bytes, e.g. 4 for scattered_dword, 16 for block_hword
    uint8_t count;              // Element count.
    uint8_t extra;              // Extra info. For block accesses, 1 means aligned OWord, 0 unaligned. For scattered accesses, # of consecutive elements.
    uint8_t simdSize;           // SIMD size for load/stores (0 indicating no associated load/store.)
    uint8_t msgRegs;            // Underlying register count for load/store operation (may be different from nregs()).
    std::array<VirtualFlag, 2> flag;
                                // Assigned flag register indices ([0] -> row, [1] -> column)
    uint8_t flagAny : 1;        // Use .anyh?
    uint8_t flagAll : 1;        // Use .allh?
    uint8_t flagInvert : 1;     // Invert flag?
    uint8_t hasNoLoad : 1;      // Does this load/store cover additional (no-load) RegisterBlocks? (packed layouts)
    uint8_t : 4;
    uint8_t sfid;               // SFID for this block.
    uint8_t rowFragment;        // If this block needs fragmenting to support row/column remainders, the maximum block size (power of 2) to fragment down to.
    uint8_t colFragment;        //     Zero if no fragmenting needed.
    uint8_t addrShift;          // log2(address units). e.g. 0 if byte addresses should be used, 4 if oword addresses should be used.

    MaskInfo rowMask;           // Row mask for this block.
    MaskInfo colMask;           // Column mask for this block.

    int32_t offsetAddr;         // Address offset, for sharing address registers. For 2D addressing, contains x/y offsets in low/high words.

    static constexpr int8_t Interleaved = -1;     // Value for cxComponent indicating interleaved real/imaginary data.

    // Constructors.
    RegisterBlock() {}
    RegisterBlock(ngen::HW hw_, Type T, int r, int c, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                  bool remainderR = false, bool remainderC = false, bool writable = false, RemainderOptions remOpts = AvoidFragment,
                  int maxRBlock = 0, int maxCBlock = 0);

    /* Similar to the regular constructor, but in case of errors returns an invalid RegisterBlock,
       instead of throwing an exception */
    static RegisterBlock tryCreate(ngen::HW hw_, Type T, int r, int c, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                                   bool remainderR = false, bool remainderC = false, bool writable = false, RemainderOptions remOpts = AvoidFragment,
                                   int maxRBlock = 0, int maxCBlock = 0);

    /* Re-create the RegisterBlock, preserving its location within the surrounding RegisterLayout. */
    void recreate(ngen::HW hw_, Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                  RemainderOptions remOpts = AvoidFragment);
    bool tryRecreate(ngen::HW hw_, Type T, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                     RemainderOptions remOpts = AvoidFragment);

    // Slice this register block, selecting a subset [x1,x2) of its rows (column = false) or columns (column = true).
    RegisterBlock slice(Type T, bool column, int x1, int x2, int x1Unclamped, int x2Unclamped, bool overrunOK,
                        const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const;

    // Same as `slice`, but returns an invalid RegisterBlock in case of errors, instead of throwing an exception.
    RegisterBlock trySlice(Type T, bool column, int x1, int x2, int x1Unclamped, int x2Unclamped, bool overrunOK,
                           const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const;

    // Find the subregister in a RegisterBlock corresponding to the element at offset (ii, jj).
    // Optionally returns the number of consecutive elements beginning with this one in *outNElems.
    ngen::Subregister find(Type T, int ii, int jj, const GRFMultirange &regs, int *outNElems = nullptr,
                           int cxComponent = -1, int component = 0) const;

    // Similar to find, but returns the region associated with succeeding elements in the block.
    // If allow2D is true, the return value is allowed to be a true 2D region.
    //   Otherwise, the return region will always be a constant stride (1D) region.
    ngen::RegisterRegion findRegion(Type T, int ii, int jj, const GRFMultirange &regs, int *outNElems = nullptr,
                                    int cxComponent = -1, int component = 0, bool allow2D = false) const;


    /* Helpers used in creating layouts. */
    void simplify(Type T);      // If block is completely crosspacked, convert to equivalent layout without crosspack.
    void compact(Type T);       // Make a RegisterBlock smaller by contracting the leading dimension, if possible.

    void calcBytes(Type T);
    void calcBytes(Type T, const MatrixAddressingStrategy &astrategy);

    /* Queries */
    bool valid() const       { return nr && nc; }
    bool operator!() const   { return !valid(); }
    operator bool()  const   { return valid();  }

    bool isLoadBlock() const { return simdSize > 0; }
    void unlinkFromMemory()  { simdSize = 0; }

    bool grfAligned() const  { return (offsetBytes & (ngen::GRF::bytes(hw) - 1)) == 0; }
    int nregs() const;
    int offsetReg() const;

    bool isSplitComplex() const { return (splitComplex || cxComponent != Interleaved); }

    /* Mask handling */
    bool hasFlag() const     { return flag[0] || flag[1]; }
    void clearFlag()         { flag[0].clear(); flag[1].clear(); flagAll = flagAny = flagInvert = false; }
    void eraseMask()         { clearFlag(); rowMask = MaskInfo(); colMask = MaskInfo(); }

    /* Address offset handling */
    ngen::Offset2D offset2D() const { return ngen::Offset2D(int16_t(offsetAddr & 0xFFFF), int16_t(offsetAddr >> 16)); }
    void set2DOffset(int16_t x, int16_t y) { offsetAddr = uint16_t(x) | (uint32_t(uint16_t(y)) << 16); }
    void subAddrOffset(int32_t aoff, bool is2D) {
        if (is2D)
            set2DOffset((offsetAddr - aoff) & 0xFFFF, (offsetAddr >> 16) - (aoff >> 16));
        else
            offsetAddr -= aoff;
    }

    // Get effective access type to use when setting up addresses.
    AccessType effectiveAccessType(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const;

    // Get effective access type to use when performing loads/stores.
    AccessType implAccessType(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const;

    // Check if pseudo-block access should use channel scattered access internally.
    bool pseudoblockUseChannelScattered(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const;

    // Get width/height/array size parameters for underlying 2D block load message.
    void getBlock2DWH(int &w, int &h, int &count, const MatrixAddressing &atype, bool prefetch, int *outMultiX = nullptr) const;

    // Count the number of address/header GRFs required by a RegisterBlock.
    int addrGRFs(const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy) const;

private:
    RegisterBlock(ngen::HW hw_, Type T, int r, int c, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                  bool remainderR, bool remainderC, bool writable, RemainderOptions remOpts,
                  int maxRBlock, int maxCBlock, bool _allowFailure);
};

// A RegisterLayout represents a logical matrix tile, internally divided into one or
//   more RegisterBlock subtiles.
class RegisterLayout {
    using V = std::vector<RegisterBlock>;

    ngen::HW hw;
    Type T;
    int nr, nc;
    MatrixAddressing atype;
    MatrixAddressingStrategy astrategy;
    bool remainderR = false, remainderC = false;
    bool writable = true;
    bool initialized = false;

    V list;

public:
    /* Constructors */

    // (1) Create an uninitialized (invalid) layout.
    RegisterLayout() {}

    // (2) Create a RegisterLayout for reading from (and optionally writing to) memory.
    RegisterLayout(ngen::HW hw_, Type T_, int r, int c, const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_,
                   bool remainderR_ = false, bool remainderC_ = false, bool writable_ = false, RemainderOptions remOpts = AvoidFragment,
                   int maxRBlock = 0, int maxCBlock = 0, bool reverseOrder = false);

    // (3) Create a RegisterLayout unbacked by memory.
    RegisterLayout(ngen::HW hw_, Type T_, int r, int c, bool colMajor,
                   int crosspack = 1, int tileR = 0, int tileC = 0,
                   bool allowPartialRegs = true, bool fullySplitCx = false);

    // (4), (5) Create a RegisterLayout from a manually-compiled list of RegisterBlocks.
    RegisterLayout(Type T_, const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_,
                   std::vector<RegisterBlock> &&list_);

    RegisterLayout(Type T_, std::vector<RegisterBlock> &&list_)
            : RegisterLayout(T_, MatrixAddressing(), MatrixAddressingStrategy(), std::move(list_)) {}

    // Similar to constructor (2), but in case of errors returns an invalid RegisterBlock,
    //    instead of throwing an exception.
    static RegisterLayout tryCreate(ngen::HW hw_, Type T_, int r, int c, const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_,
                                    bool remainderR_ = false, bool remainderC_ = false, bool writable_ = false, RemainderOptions remOpts = AvoidFragment,
                                    int maxRBlock = 0, int maxCBlock = 0, bool reverseOrder = false);

    /* Basic information */
    int rows()  const { return nr; }
    int cols()  const { return nc; }
    Type type() const { return T; }

    bool colMajor() const;
    int crosspack() const;

    bool empty() const { return list.empty(); }
    bool valid() const { return initialized; }

    bool operator!() const { return !valid(); }
    operator bool()  const { return valid();  }

    MatrixAddressing &addressing()                             { return atype; }
    const MatrixAddressing &addressing()                 const { return atype; }

    MatrixAddressingStrategy &addressingStrategy()             { return astrategy; }
    const MatrixAddressingStrategy &addressingStrategy() const { return astrategy; }

    /* Iterating over blocks */
    int blocks() const { return int(list.size()); }

    V::iterator       begin()        { return list.begin(); }
    V::const_iterator begin()  const { return list.cbegin(); }
    V::const_iterator cbegin() const { return list.cbegin(); }

    V::iterator       end()          { return list.end(); }
    V::const_iterator end()    const { return list.cend(); }
    V::const_iterator cend()   const { return list.cend(); }

    RegisterBlock &operator[](int b)             { return list[b]; }
    const RegisterBlock &operator[](int b) const { return list[b]; }

    RegisterBlock &back()            { return list.back(); }

    const V &allBlocks()             { return list; }

    RegisterLayout int3UnpackLayout() const;

    /* Element lookups */

    // Find the subregister in a layout corresponding to element (i, j).
    // Optional outputs:
    //    outBlock: pointer to the containing RegisterBlock
    //   outNElems: the number of contiguous elements available in the layout, starting from this one.
    ngen::Subregister find(int i, int j, const GRFMultirange &regs,
                           int *outNElems = nullptr, const RegisterBlock **outBlock = nullptr,
                           int cxComponent = -1, int component = 0) const;

    // Similar to find, but returns the region associated with succeeding elements in the block.
    // If allow2D is true, the return value is allowed to be a true 2D region.
    //   Otherwise, the return region will always be a constant stride (1D) region.
    ngen::RegisterRegion findRegion(int i, int j, const GRFMultirange &regs,
                                    int *outNElems = nullptr, const RegisterBlock **outBlock = nullptr,
                                    int cxComponent = -1, int component = 0, bool allow2D = false) const;

    /* Layout queries */

    // Check if every block in a layout has the given crosspack, with no padding.
    bool hasFullCrosspack(int crosspack) const;

    // Check if the layout is tiled with the given tiling.
    bool hasTiling(int tileR, int tileC) const;

    // Check if a layout has remainders enabled.
    bool hasRemainders(bool remainderR = true, bool remainderC = true) const;

    // Check if a layout has any kind of fragmenting.
    bool hasFragmenting(bool ignoreWholeFragR = false, bool ignoreWholeFragC = false) const;

    // Check if a layout has any masking.
    bool hasMasking() const;

    // Check if a layout has any flag registers assigned.
    bool hasFlags() const;

    // Find the maximum block size in a layout, in registers.
    int maxLoadBlock() const;

    // Count the number of registers needed by a register layout.
    int regs() const;

    /* Layout mutations */

    // Reset to an empty layout.
    void clear() { *this = RegisterLayout(); }

    // Unlink a layout from its in-memory representation.
    void unlinkFromMemory();

    // Get a clone of this layout, unlinked from its in-memory representation.
    RegisterLayout unlinkedClone() const {
        auto clone = *this;
        clone.unlinkFromMemory();
        return clone;
    }

    // Change data types.
    void cast(Type Tnew) { T = Tnew; }

    // Repoint this layout at a different in-memory representation.
    void retarget(const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_);

    // Re-order a layout so that registers appear in appropriate order (row or column major).
    void sort(bool reverse = false);

    // Assign a single mask to all blocks in a layout.
    void assignUniformMask(ngen::FlagRegister flag, ngen::HW hw, int idx = 0);

    // Assign runtime-computed descriptor information to all blocks in this layout.
    // Returns true if successful; false if not all blocks in layout are compatible.
    bool assignAllDescs();

    // Match the register offsets in this register layout to another, reference layout.
    // Returns true if successful. If not successful, the layout is unchanged.
    bool match(const RegisterLayout &ref);

    // Similar to match but allows either layout to change to match the other.
    friend inline bool matchBidirectional(RegisterLayout &layout1, RegisterLayout &layout2) {
        return layout1.match(layout2) || layout2.match(layout1);
    }

    // Update layout dimensions if the RegisterBlock list has been manually modified.
    void updateDims();

    /* Slicing/update routines. Routines beginning with `try` return invalid layouts in case of errors,
       instead of throwing an exception. */

    // Copy this layout, splitting blocks as needed so that every block is contained in some
    //   block of a reference layout.
    RegisterLayout reblock(std::vector<int32_t> &blockMap, const RegisterLayout &ref) const;

    // Retrieve a slice from a layout (a subset of its rows or columns).
    RegisterLayout slice(bool column, int x1, int x2, bool overrunOK, bool decoalesce = false) const;

    // Slice a layout and accompanying address registers.
    RegisterLayout slice(std::vector<ngen::RegisterRange> &subaddrs, const std::vector<ngen::RegisterRange> &addrs,
                         bool column, int x1, int x2, bool overrunOK) const;

    // Slice a layout, also returning indices of associated address registers.
    RegisterLayout slice(std::vector<int> &indices, bool column, int x1, int x2, bool overrunOK) const;

    RegisterLayout trySlice(bool column, int x1, int x2, bool overrunOK, bool decoalesce = false) const;
    RegisterLayout trySlice(std::vector<ngen::RegisterRange> &subaddrs, const std::vector<ngen::RegisterRange> &addrs,
                            bool column, int x1, int x2, bool overrunOK) const;
    RegisterLayout trySlice(std::vector<int> &indices, bool column, int x1, int x2, bool overrunOK) const;

    // Attempt to create a 2D block layout that matches an existing layout.
    // Currently only generates regular/transpose 2D block (no VNNI support).
    RegisterLayout tryUpgradeToBlock2D(const MatrixAddressing &atype2D, const MatrixAddressingStrategy &astrategy2D) const;

protected:
    RegisterLayout(ngen::HW hw_, Type T_, int r, int c, const MatrixAddressing &atype_, const MatrixAddressingStrategy &astrategy_,
                   bool remainderR_, bool remainderC_, bool writable_, RemainderOptions remOpts,
                   int maxRBlock, int maxCBlock, bool reverseOrder, bool);

    bool appendBlocks(int r, int c, int roff, int coff, RemainderOptions remOpts, int maxRBlock = 0, int maxCBlock = 0);
    bool append1DBlocks(int r, int c);

    void coalesceAddrs();

    void postprocess();
    void postprocess2D();
    void postprocessMultitile();
    void postprocessLargeCP();
    void postprocessDPASW();

    void finalize();

    RegisterLayout trySlice(std::vector<ngen::RegisterRange> *subaddrs, std::vector<int> *indices, const std::vector<ngen::RegisterRange> *addrs,
                            bool column, int x1, int x2, bool overrunOK, bool decoalesce = false) const;
};


GEMMSTONE_NAMESPACE_END

#endif /* header guard */
