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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_DRIVER_INFO_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_DRIVER_INFO_HPP

#include <cstdint>
#ifdef STANDALONE
#include <iostream>
#endif
#include <string>

#include "gemmstone/config.hpp"
#include "gemmstone/type.hpp"

GEMMSTONE_NAMESPACE_START

// Loop identifiers.
enum LoopType : uint8_t {
    LoopM = 0,
    LoopN = 1,
    LoopK = 2,
    LoopPersistent = 0x40,          // Flag OR'ed with other loop types, indicating persistent threads.
    LoopMNBoustrophedonMNK = 0x80,  // Fused m/n indices (boustrophedon ordering), with MNK nested inside
    LoopMNBoustrophedonNMK = 0x81,  // Fused n/m indices (boustrophedon ordering), with NMK nested inside
    LoopMNHilbertMNK = 0x90,        // Fused m/n indices (Hilbert ordering), with MNK nested inside
    LoopMNHilbertNMK = 0x91,        // Fused n/m indices (Hilbert ordering), with NMK nested inside
    LoopMNLinearMNK = 0xA0,         // Fused m/n indices (simple linear ordering), with MNK nested inside
    LoopMNLinearNMK = 0xA1,         // Fused n/m indices (simple linear ordering), with NMK nested inside
    LoopMNNestedLinearMNK = 0xB0,   // Fused m/n indices (nested linear ordering), with MNK nested inside
    LoopMNNestedLinearNMK = 0xB1,   // Fused n/m indices (nested linear ordering), with NMK nested inside
    LoopAny = 0xFF,
    LoopNone = 0xFF
};

// Workgroup sizing options.
enum WGType : uint8_t {
    WGDynamic = 0,                  // Dynamic m/n work group size (can shrink or expand)
    WGFixed = 1,                    // Fixed m/n work group size
    WGShrinkable = 2,               // m/n work group size can shrink but not expand
};

// Flags.
enum DriverInfoFlags : uint64_t {
    FlagKRemainderHandling = 1,        // GEMM kernel performs k remainder handling
    FlagKParallel = 2,                 // GEMM kernel is parallelized in the k dimension.
    FlagZParallel = 2,                 // Copy kernel is parallelized in the z dimension.
    FlagKParallelLocal = 4,            // GEMM kernel is parallelized in the k dimension inside a workgroup.
    FlagKParallelVariable = 8,         // GEMM kernel uses variable k-parallelization (see GEMMStrategy::kParallelVariable).
    FlagFusedBeta = 0x10,              // GEMM kernel does fused beta scaling + atomics.
    FlagFusedPostOps = 0x20,           // GEMM kernel does fused atomics + post-ops.
    FlagTempC = 0x40,                  // GEMM kernel needs temporary C buffer.
    FlagAltFusedBeta = 0x80,           // GEMM kernel uses alternate fused beta scaling + atomics logic.
    FlagAutoAtomic = 0x100,            // GEMM kernel may use atomic C accesses automatically for beta = 1.
    FlagShrinkWGK = 0x200,             // With local k-parallelization, automatically shrink wgK to fit dispatch to GPU.
    FlagAlphaPtr = 0x400,              // Pass alpha by pointer.
    FlagBetaPtr = 0x800,               // Pass beta by pointer.
    FlagFixedWGK = 0x1000,             // With local k-parallelization, wgK is fixed
    FlagZeroTempC = 0x2000,            // GEMM kernel expects temporary C buffer to be pre-zeroed and will leave zeros there.
    FlagNondeterministic = 0x4000,     // Kernel produces nondeterministic results.
    FlagMaskFillGoal = 0xF0000,        // Fraction of available thread slots to fill, in sixteenths
    FlagShiftFillGoal = 16,            //   (starting bit)
    FlagScrambleM = 0x100000,          // Scramble WGs in m dimension.
    FlagScrambleN = 0x200000,          // Scramble WGs in n dimension.
    FlagExtraWG = 0x400000,            // Add an additional workgroup.
    FlagSeparateASums = 0x1000000,     // Kernel needs a buffer with A row/group sums.
    FlagSeparateBSums = 0x2000000,     // Kernel needs a buffer with B column/group sums.
    FlagCInterleave = 0x8000000,       // C interleave: interleave columns of C among threads (chunk = 64/Tc_ext).
};

// Driver information, shared by all kernel types.
struct CommonDriverInfo {
    int subgroupSize;               // Declared subgroup size (unrelated to actual SIMD lengths in kernel)
    LoopType fusedLoop;             // Loop dimension in which EUs are fused (if any).
    int grfCount;                   // # of GRFs used by kernel.
    LoopType loopOrder[3];          // Loops corresponding to x/y/z dimensions of kernel dispatch.
    int blocking[3];                // Standard blocking sizes in m/n/k dimensions.
    int blockingAlt[3];             // Alternative blocking sizes in m/n/k dimensions.
    int unroll[3];                  // m/n/k unrolls.
    int wg[3];                      // HW threads per workgroup in m/n/k dimensions.
    int wgExpand;                   // If > 1, workgroup size needs to be scaled by this factor.
    WGType wgUpdate;                // Work group type showing how/if work group sizes can be updated.
    uint64_t flags;                 // Bitfield with additional boolean kernel attributes (see DriverInfoFlags enum).
    int slm;                        // Minimum SLM allocation.
    int perKSLM;                    // If > 0, dynamically allocate at least perKSLM * wg[LoopK] bytes of SLM.
    int alignment[3];               // Address alignment requirements for A,B,C (gemm) or S,D (copy).
    bool support4GB[3];             // True if >4GB buffers allowed for A,B,C (gemm) or S,D (copy).
#ifdef STANDALONE
    std::string problem;            // String representation of problem.
    std::string extra;              // String representation of any Extra information.
#endif

    bool fusedEUs()           const { return (fusedLoop != LoopNone); }
    bool isMNK()              const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopM || l == LoopMNHilbertMNK || l == LoopMNBoustrophedonMNK || l == LoopMNLinearMNK || l == LoopMNNestedLinearMNK; }
    bool isNMK()              const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopN || l == LoopMNHilbertNMK || l == LoopMNBoustrophedonNMK || l == LoopMNLinearNMK || l == LoopMNNestedLinearNMK; }
    bool isHilbert()          const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopMNHilbertMNK || l == LoopMNHilbertNMK; }
    bool isBoustrophedon()    const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopMNBoustrophedonMNK || l == LoopMNBoustrophedonNMK; }
    bool isSimpleLinear()     const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopMNLinearMNK || l == LoopMNLinearNMK; }
    bool isNestedLinear()     const { auto l = loopOrder[0] & ~LoopPersistent; return l == LoopMNNestedLinearMNK || l == LoopMNNestedLinearNMK; }
    bool isLinearOrder()      const { return (loopOrder[0] != LoopNone) && (loopOrder[0] & 0x80); }
    bool isPersistent()       const { return (loopOrder[0] != LoopNone) && (loopOrder[0] & LoopPersistent); }
    bool fixedWG()            const { return wgUpdate == WGFixed; }
    int  threadsPerWG()       const { return wg[LoopM] * wg[LoopN] * wg[LoopK] * wgExpand; }
    bool kRemainderHandling() const { return flags & FlagKRemainderHandling; }
    bool kParallel()          const { return flags & FlagKParallel; }
    bool zParallel()          const { return flags & FlagZParallel; }
    bool kParallelLocal()     const { return flags & FlagKParallelLocal; }
    bool kParallelVariable()  const { return flags & FlagKParallelVariable; }
    bool fusedBeta()          const { return flags & FlagFusedBeta; }
    bool fusedPostOps()       const { return flags & FlagFusedPostOps; }
    bool needsTempC()         const { return flags & FlagTempC; }
    bool zeroTempC()          const { return flags & FlagZeroTempC; }
    bool altFusedBeta()       const { return flags & FlagAltFusedBeta; }
    bool mayUseAutoAtomic()   const { return flags & FlagAutoAtomic; }
    bool shrinkWGK()          const { return flags & FlagShrinkWGK; }
    bool alphaPtr()           const { return flags & FlagAlphaPtr; }
    bool betaPtr()            const { return flags & FlagBetaPtr; }
    bool fixedWGK()           const { return flags & FlagFixedWGK; }
    bool nondeterministic()   const { return flags & FlagNondeterministic; }
    bool scrambleM()          const { return flags & FlagScrambleM; }
    bool scrambleN()          const { return flags & FlagScrambleN; }
    int extraWGs()            const { return (flags & FlagExtraWG) ? 1 : 0; }
    bool needsSeparateASums() const { return flags & FlagSeparateASums; }
    bool needsSeparateBSums() const { return flags & FlagSeparateBSums; }

    int wgTile(LoopType l)    const { return unroll[l] * wg[l]; }
    int kPadding()            const { return (kParallel() || kParallelVariable()) ? blockingAlt[LoopK] : 0; }

    float fillGoal() const {
        auto sixteenths = (flags & FlagMaskFillGoal) >> FlagShiftFillGoal;
        return (sixteenths > 0) ? (sixteenths / 16.0f) : 1.0f;
    }

    bool cInterleaveEnabled() const {
        return flags & FlagCInterleave;
    }

    // Returns the C interleave chunk size (columns of C per thread).
    int cInterleaveChunk(Type Tc_ext) const {
        return cInterleaveEnabled() ? 64 / Tc_ext : 1;
    }

#ifdef STANDALONE
    inline void dump(std::ostream &s, bool withExtras = true) const;
    static inline CommonDriverInfo undump(std::istream &s, bool withExtras = true);
#endif
};

// Definitions for flag arguments to kernels.
enum {
    FlagCOColumn = 0x4,
    FlagCORow = 0x8,
    FlagNonfinalKBlock = 0x10,
    FlagNoninitialKBlock = 0x80,
    FlagLateFusedGEMMDone = 0x100,
    FlagEarlyFusedGEMMDone = 0x200,
    FlagStoreSums = 0x400,
    FlagKSlicing = 0x1000,
    FlagLeader = 0x2000,
    FlagKPartitioned = 0x4000,
    FlagDidBeta = 0x100,
    FlagSkipBetaCheck = 0x200,
    FlagKSlice2 = 0x10000,
};

#ifdef STANDALONE
void CommonDriverInfo::dump(std::ostream &s, bool withExtras) const
{
    s << std::boolalpha << '{';
    s << subgroupSize << ", ";
    s << "(LoopType) " << int(fusedLoop) << ", ";
    s << grfCount << ", {";
    s << "(LoopType) " << int(loopOrder[0]) << ", ";
    s << "(LoopType) " << int(loopOrder[1]) << ", ";
    s << "(LoopType) " << int(loopOrder[2]) << "}, {";
    s << blocking[0] << ", ";
    s << blocking[1] << ", ";
    s << blocking[2] << "}, {";
    s << blockingAlt[0] << ", ";
    s << blockingAlt[1] << ", ";
    s << blockingAlt[2] << "}, {";
    s << unroll[0] << ", ";
    s << unroll[1] << ", ";
    s << unroll[2] << "}, {";
    s << wg[0] << ", ";
    s << wg[1] << ", ";
    s << wg[2] << "}, ";
    s << wgExpand << ", ";
    s << "(WGType) " << int(wgUpdate) << ", ";
    s << flags << ", ";
    s << slm << ", ";
    s << perKSLM << ", {";
    s << alignment[0] << ", ";
    s << alignment[1] << ", ";
    s << alignment[2] << "}, {";
    s << support4GB[0] << ", ";
    s << support4GB[1] << ", ";
    s << support4GB[2] << '}';
    if (withExtras) {
        s << ", \"";
        s << problem << "\", \"";
        s << extra << "\"";
    }
    s << '}';
    s << std::noboolalpha;
}

CommonDriverInfo CommonDriverInfo::undump(std::istream &s, bool withExtras)
{
    auto expect = [&](char c) {
        char in;
        s >> std::ws;

        if (s.eof())
            throw std::runtime_error("Fatal error parsing CommonDriverInfo");

        s >> in;

        if (in != c)
            throw std::runtime_error("Fatal error parsing CommonDriverInfo");
    };

    auto readLoop = [&]() {
        int intInput;
        s >> std::ws;
        if (s.peek() == '(') {
            std::string cast = "(LoopType)";
            for (auto &c: cast)
                expect(c);
        }
        s >> intInput;
        return static_cast<LoopType>(intInput);
    };

    auto readWG = [&]() {
        s >> std::ws;
        if (s.peek() == '(') {
            int intInput;
            std::string cast = "(WGType)";
            for (auto &c: cast)
                expect(c);
            s >> intInput;
            return static_cast<WGType>(intInput);
        } else {
            bool fixedWG;
            s >> fixedWG;
            return fixedWG ? WGFixed : WGDynamic;
        }
    };

    auto readFlags = [&]() {
        uint64_t flags = 0;
        s >> std::ws;
        if (!std::isalpha(s.peek()))
            s >> flags;
        else {
            bool kRemainderHandling, kParallel, kParallelLocal;

            s >> kRemainderHandling;
            expect(',');
            s >> kParallel;
            expect(',');
            s >> kParallelLocal;

            if (kRemainderHandling) flags |= FlagKRemainderHandling;
            if (kParallel)          flags |= FlagKParallel;
            if (kParallelLocal)     flags |= FlagKParallelLocal;
        }
        return flags;
    };

    auto read2 = [&](int (&array)[2]) {
        expect('{');
        for (int i = 0; i < 2; i++) {
            if (i > 0) expect(',');
            s >> array[i];
        }
        expect('}');
    };

    auto read3 = [&](int (&array)[3]) {
        expect('{');
        for (int i = 0; i < 3; i++) {
            if (i > 0) expect(',');
            s >> array[i];
        }
        expect('}');
    };

    CommonDriverInfo result;

    // Read lines until we get to one starting with '{'
    s >> std::ws;
    while (s.peek() != '{') {
        std::string line;
        std::getline(s, line);
        s >> std::ws;
        if (s.eof())
            throw std::runtime_error("Fatal error parsing CommonDriverInfo");
    }

    s >> std::boolalpha;
    expect('{');
    s >> result.subgroupSize;
    expect(',');
    result.fusedLoop = readLoop();
    expect(',');
    s >> result.grfCount;
    expect(',');
    expect('{');
    for (int i = 0; i < 3; i++) {
        if (i > 0) expect(',');
        result.loopOrder[i] = readLoop();
    }
    expect('}');
    expect(',');
    read3(result.blocking);
    expect(',');
    read3(result.blockingAlt);
    expect(',');
    read3(result.unroll);
    expect(',');
    read3(result.wg);
    expect(',');
    s >> result.wgExpand;
    expect(',');
    result.wgUpdate = readWG();
    expect(',');
    result.flags = readFlags();
    expect(',');
    s >> result.slm;
    expect(',');
    s >> result.perKSLM;
    expect(',');
    read3(result.alignment);
    expect(',');
    expect('{');
    for (int i = 0; i < 3; i++) {
        if (i > 0) expect(',');
        s >> result.support4GB[i];
    }
    expect('}');
    if (withExtras) {
        expect(',');
        expect('"');
        std::getline(s, result.problem, '"');
        expect(',');
        expect('"');
        std::getline(s, result.extra, '"');
    }
    expect('}');
    s >> std::noboolalpha;

    return result;
}

enum TimepointType : uint32_t {
    TimepointInvalid        = 0,            // Not a timepoint -- marks end of timepoint array
    TimepointTmEvent        = 1,            // Timer interrupted, value suspect (ORs with other values below)
    TimepointThreadStart    = 0x10000,      // Thread start
    TimepointThreadEnd      = 0x20000,      // Thread end
    TimepointSubthreadStart = 0x30000,      // Sub-thread start (persistent threads)
    TimepointSubthreadEnd   = 0x40000,      // Sub-thread end (persistent threads)
    TimepointAccumDone      = 0x50000,      // C accumulation complete
};

struct Timepoint {
    uint32_t type;                          // See TimepointType enum
    uint32_t time;
};

#endif /* STANDALONE */

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
