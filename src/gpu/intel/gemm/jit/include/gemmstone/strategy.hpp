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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_STRATEGY_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_STRATEGY_HPP

#include "gemmstone/config.hpp"

#include "internal/ngen_includes.hpp"
#include "internal/utils.hpp"

#include "gemmstone/driver_info.hpp"
#include "ngen_emulation.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/type.hpp"

GEMMSTONE_NAMESPACE_START

/* Zero-padding to allow various types to be bytewise compared. */
#define ZPAD(x, bytes) uint8_t pad##x[bytes] = {};

// Selects the load/store/atomic messages used to access a matrix.
enum class AccessType : uint8_t {
    Scattered,          // Use scattered accesses (transposes matrices)
    ChannelScattered,   // Use untyped surface reads (transposes matrices)
    Block,              // Use block messages
    PseudoBlock,        // Use scattered accesses to emulate block accesses
    Block2D,            // Use 2D block messages
    Block2DTranspose,   // Use 2D block messages with transposition
    Block2DVNNI,        // Use 2D block messages with VNNI transform
    CacheLine,          // Use scattered accesses with one element per cache line
};

static inline bool isBlocklike(AccessType t)
{
    return one_of(t, {AccessType::Block, AccessType::PseudoBlock});
}

static inline bool isBlock2D(AccessType t)
{
    return one_of(t, {AccessType::Block2D, AccessType::Block2DTranspose, AccessType::Block2DVNNI});
}

static inline bool isTransposing(AccessType t)
{
    return one_of(t, {AccessType::Scattered, AccessType::ChannelScattered, AccessType::Block2DTranspose});
}

// Strategies for choosing scattered access SIMD width.
enum class ScatterSIMD : uint8_t {
    Default,
    Wide,                   // Prefer wider SIMD (more scattered lanes)
    Narrow                  // Prefer narrower SIMD (more consecutive access)
};

// Strategies for accessing a matrix from memory.
struct MatrixAddressingStrategy {
    ngen::AddressBase base;                     // Base for addressing (A64/BTS/...)
    AccessType accessType
        = AccessType::Block;                    // Block/scattered/etc. access
    uint8_t tileR = 0, tileC = 0;               // Desired tiling (0 if none) in registers.
    ScatterSIMD smode = ScatterSIMD::Default;   // SIMD selection for scattered accesses.
    uint8_t padded : 1;                         // Allow read/write overruns?
    uint8_t atomic : 1;                         // Atomic access? (only relevant for C)
    uint8_t address2D : 1;                      // Use 2D addressing? (media block-style loads)
    uint8_t prefetch : 1;                       // Prefetch only?
    uint8_t pfLoad : 1;                         // Arrange blocks as if prefetching, but actually load data.
    uint8_t newDP : 1;                          // Use new dataport messages? (XeHPG+)
    uint8_t dpasw : 1;                          // DPASW half layout?
    uint8_t noExtraPad : 1;                     // Avoid extra padding?
    uint8_t noCoalesce : 1;                     // Disable address coalescing?
    uint8_t pad0 : 7;
    ngen::CacheSettingsLSC cachingR             // Cache policies for LSC reads.
        = ngen::CacheSettingsLSC::Default;
    ngen::CacheSettingsLSC cachingW             // Cache policies for LSC writes.
        = ngen::CacheSettingsLSC::Default;

    MatrixAddressingStrategy() : padded(false)
                               , atomic(false)
                               , address2D(false)
                               , prefetch(false)
                               , pfLoad(false)
                               , newDP(false)
                               , dpasw(false)
                               , noExtraPad(false)
                               , noCoalesce(false)
                               , pad0(0) {}

    void preflight(ngen::HW hw);
    void forceA64();
    void assignSurface(uint8_t index) { if (!base.isStateless()) base.setIndex(index); }

    ngen::GlobalAccessType getGlobalAccessType() const {
        return base.isStateless() ? ngen::GlobalAccessType::Stateless : ngen::GlobalAccessType::Surface;
    }
};

inline void transposeAccessType(MatrixAddressingStrategy &astrategy)
{
    astrategy.accessType = isTransposing(astrategy.accessType) ? AccessType::Block :
                                  astrategy.base.isStateless() ? AccessType::Scattered
                                                               : AccessType::ChannelScattered;
}

// Remainder handling strategies.
enum class RemainderHandling : uint8_t {
    Ignore,         // Assume no remainder, or handled by hardware bounds checking.
    General,        // Handle all remainder cases.
    Split,          // Generate copies of the kernel with and without remainder handling.
    KnownRemainder, // Assume remainder case; don't create special code for non-remainder case.
};

// How to split A/B amongst threads in a workgroup.
enum class CoopSplit {
    K,          // Split in k dimension (within each thread's m/n range)
    MN,         // Split in m/n dimensions
    Linear,     // Split in linear index order
    FullK,      // Split in k dimension (within the entire workgroup's m/n range)
};

// Methods for traversing a matrix.
enum class WalkOrder : uint8_t {
    HW2D,           // Rely on HW thread dispatch for ordering
    SimpleLinear,   // Simple 1D->2D mapping in column-major/row-major order
    NestedLinear,   // Fixed-size blocks of WGs traversed in column/row-major order
    Hilbertlike,    // Cache-oblivious Hilbert curve-based order
    Boustrophedon,  // Cache-aware panel boustrophedon walk order
};

// Places to store r0 header.
enum class MoveR0 {
    None, Acc, Addr, GRF
};

// Strategy parameters shared between different kernel types.
struct CommonStrategy {
    int subgroupSize = 8;                       // Subgroup size provided to OpenCL runtime.
    int GRFs = 128;                             // # of GRFs to use.
    bool fused = false;                         // Fused EU handling enabled?
    bool dualGRF = true;                        // Enable two-GRF instructions.
    bool ieeeDenormals = true;                  // Enable IEEE-compliant denormals.
    bool spf = true;                            // Enable Single Program Flow (SPF) mode in EUs.
    MoveR0 moveR0 = MoveR0::Acc;                // Where to store r0 information.
    bool readSuppressionWA = true;              // Workaround for HW issue with read suppression after fused sends.
    bool multitile = true;                      // Enable multitile (implicit scaling) support?
    bool wgInSS = false;                        // Pretend to use barriers so that each WG belongs to 1 SS/DSS.
    bool finalFence = false;                    // Issue global memory fence before EOT.
    bool chainPrefetches = false;               // Chain prefetches in {Atomic} sequences to reuse tokens?
                                    ZPAD(A, 3)
    int pauseCycles = 0x0100;                   // Number of cycles to pause when waiting in a spin-loop.
    bool simulation = false;                    // For use in simulator?
    bool systolicAvailable = false;             // True if systolic array present.
    bool avoidIncConflicts = true;              // If true, duplicate address increments across banks to avoid bundle conflicts.
    bool isDSLGenerator = false;
    ngen::HW raHW = ngen::HW::Unknown;          // Pretend to be a different GPU for register allocation purposes.
    ngen::ThreadArbitrationMode arbitrationMode
        = ngen::ThreadArbitrationMode::Default; // Thread arbitration policy to use.
    int activeThreads = 0;                      // # of active threads (0 = dynamic).

    ngen::EmulationStrategy emulate;
                                    ZPAD(C, 2)

    CommonStrategy() = default;
    CommonStrategy(ngen::HW hw, int stepping = 0);
    void preflight(ngen::HW hw, const CommonProblem &problem);
};

// Strategy parameters for GEMM kernels.
struct GEMMStrategyPOD : public CommonStrategy {
    int blocking[3] = {0};                       // Recommended block size in each dimension (m/n/k) -- for driver.
    int blockingAlt[3] = {0};                    // Alternate block size in each dimension (m/n/k) -- for driver.
                                                 //     m/n alternates are for Hilbert-ordered kernels when Hilbert ordering disabled.
                                                 //     k alternate is for multi-tile execution with implicit scaling.
    int unroll[3] = {0};                         // Unrolls in each dimension (m/n/k), indexed by LoopType.
    int unrollK_masked = 0;                      // k unroll to use when masking.
    int extraKAlign = 1;                         // Additional k alignment when blocking.
    LoopType loopOrder[3]
        = {LoopM, LoopN, LoopK};                 // Expected order of loops in driver code (in order from innermost to outermost).
    LoopType fusedLoop = LoopM;                  // Direction of fusing if threads fused.
    WalkOrder cWalkOrder = WalkOrder::HW2D;      // Order for traversing tiles of C
    bool persistent = false;                     // Use persistent thread model?
    bool reverse[2] = {false, false};            // Reverse m/n walk order?
    bool scramble[2] = {false, false};           // Scramble m/n walk order?
    bool fmaBoustrophedon = false;               // Use boustrophedon ordering inside FMA/DPAS blocks?
                                    ZPAD(A, 1)
    int fmaSIMD = 0;                             // Vector length for FMA (0 = default = 2 GRFs).
    int kChain = 0;                              // # of FMAs to chain in k dimension.
    int dotVL = 0;                               // If > 0, use dot products of the given length, instead of outer products.
    int wg[3] = {0,0,0};                         // m/n/k workgroup sizes, 0 if unconstrained. Indexed by LoopType.
    WGType forceWGUpdate = WGDynamic;            // Force work group update type.
    bool forceFixedWGK = false;                  // Force fixed k workgroup size.
                                    ZPAD(B, 2)
    int wgPadFactor = 1;                         // If > 1, pad workgroup with empty threads.
    MatrixAddressingStrategy A, B, C;            // Strategies for accessing A/B/C.
    MatrixAddressingStrategy AO, BO, CO;         // Strategies for accessing A/B/C offsets.
    MatrixAddressingStrategy A_scale, B_scale;   // Strategies for accessing A/B scales.
    MatrixAddressingStrategy Ag, Bg;             // Strategies for accessing A/B groupwise reductions.
    int ka_load = 0, kb_load = 0;                // How much of A/B is loaded at once, in k dimension
    int ka_load_masked = 0, kb_load_masked = 0;  // Same as above, when masking m/n (0 = default = same as ka/kb_load)
    int ka_repack = 0, kb_repack = 0;            // k chunk size for A/B repacking (0 = default)
    bool loadBFirst = false;                     // If true, load B before A (default A then B).
    bool doubleMasking = false;                  // Allow A/B to be masked in both dimensions.
    bool kDescRem = false;                       // Allow descriptor-based k remainder handling for A/B.
    bool slmA = false, slmB = false;             // Whether to copy A/B to SLM.
    bool splitCopy = false;                      // Separate SLM copy and compute threads?
    bool tlbWarmup = false;                      // Enable TLB warmup?
    bool tokenAlloc = false;                     // Enable dedicated SWSB token allocation for k loop loads/stores.
    int slmBuffersA = 0;                          // # of A SLM buffers, 0 for none.
    int slmBuffersB = 0;                          // # of B SLM buffers, 0 for none.
    int unrollKSLM = 0;                          // k unroll for SLM copies (0 = auto = unroll[LoopK]/slmCopies)
    int unrollKSLMMasked = 0;                    //   Alternate value to use with masking (0 = same as unrollKSLM)
    bool slmATrans = false, slmBTrans = false;   // Whether A/B SLM data should be completely crosspacked (transposed).
                                    ZPAD(D, 2)
    int A_copies = 1, B_copies = 1;              // # of copies of A/B matrices, for latency absorption
    int slmCopies = 1;                           // # of copies of loaded A/B matrices for SLM copies.
    bool slmRepackAhead = false;                 // Repack SLM data ahead of stores?
                                    ZPAD(E, 3)
    int optAlignAB = 0;                          // Optional alignment for A/B. If > 0, create two versions of k loop, one for A/B aligned to this value, one not.
    bool optAlignAB2D = false;                   //   If true, create two version of k loop, one for A/B aligned to block 2D requirements, one not.
                                    ZPAD(F, 3)
    AccessType unalignedAccA, unalignedAccB;     // Access types to use for A/B on unaligned path.
                                    ZPAD(G, 2)
    int ka_prefetch = 0, kb_prefetch = 0;        // Chunk size for prefetching A/B.
    int ka_pfStride = 0, kb_pfStride = 0;        // k stride between A/B prefetches.
    bool cooperativePF = true;                   // Enable WG-cooperative A/B prefetches.
    bool pfaux = false;                          // Auxiliary (quant scale/offset/group-sum) prefetches alongside A/B; off by default (register pressure).
                                    ZPAD(H, 2)
    int prefetchA = 0, prefetchB = 0, prefetchC = 0;                // Prefetch distances, in units of unrollK.
    int prefetchAMasked = 0, prefetchBMasked = 0;                   // Same as above, when masking m/n.
    MatrixAddressingStrategy A_prefetch, B_prefetch, C_prefetch;    // Strategies for prefetching A/B/C.
    bool l3PrefetchA = false;                    // Enable L3 prefetch for A?
    bool l3PrefetchB = false;                    // Enable L3 prefetch for B?
                                    ZPAD(HH, 2)
    int prefetchABL3 = 0;                        // L3 prefetch distance for A/B.
    int ka_prefetchL3 = 0, kb_prefetchL3 = 0;    // Chunk size for L3 prefetch of A/B.
    MatrixAddressingStrategy AB_prefetchL3;      // Strategy for L3 prefetch of A/B.
    enum {
        CSeparate,                                   // C stored in its own bundle, A/B in the other bundle.
        ACB,                                         // A, then C, then B
        BCA,                                         // B, then C, then A
        VNC,                                         // A/B (broadcast matrix second), then C
        ABInterleave,                                // A/B interleaved, then C
        NSeparate,                                   // Broadcast input stored in its own bundle(s)
        VAvoid,                                      // C registers allocated to avoid non-broadcast inputs
    } registerScheme = CSeparate;                // Register layout scheme.
    int nSeparateChunk = 0;                      // If > 0, chunk size for NSeparate, to facilitate switching layouts.
    bool kParallel = false;                      // If true, generate k-parallelized kernel using global memory reduction.
    bool kParallelLocal = false;                 // If true, generate k-parallelized kernel using local memory reduction.
    bool kInterleave = false;                    //   Interleave threads in k dimension?
                                    ZPAD(I, 1)
    int kInterleaveChunk = 0;                    //     Minimum chunk size for interleaving (0 for automatic).
    bool shrinkWGK = false;                      //   Shrink wgK automatically to try to fit dispatch in 1 wave (or smaller)?
                                    ZPAD(J, 3)
    int fillGoal = 0;                            //     With shrinkWGK, try to fill this fraction of available thread slots, measured in sixteenths (0 = default).
    bool cInterleave = false;                    // Enable C-interleaved tile: interleave columns of C among threads (chunk = 64/Tc_ext).
    bool kParallelVariable = false;              // If true, generate kernel that uses variable k-parallelization for load balancing.
    bool fuseBeta = false;                       //   Fuse beta scaling into kernel? (kParallel/kParallelVariable, requires linear ordering)
    bool fusePostOps = false;                    //   Fuse post-operations into kernel? (kParallel/kParallelVariable, requires linear ordering)
    bool altFusedBeta = false;                   //   Enable alternate beta fusion implementation? (requires sequential dispatch)
    bool zeroTempC = false;                      //   Use pre-zeroed temporary C memory.
    bool relaxedAccumulation = false;            //   Allow downconversion of partial contributions to Tc_ext.
                                                 //     If false (default), only downconvert C at the end of the calculation.
                                    ZPAD(K, 1)
    int kPadding = 32;                           //   Pad k dimension when load balancing (kParallel/kParallelVariable)
    int barrierFreq = 0;                         // If > 0, set a periodic barrier every barrierFreq k loops to keep threads together.
    bool splitBarrier = false;                   //   Use split barriers for these periodic barriers?
    bool altCRemainder = false;                  // Use alternative double-loop C remainder code?
    bool block2DCRemainder = false;              // Generate block 2D C remainder path?
    bool block2DCFull = false;                   //   Use block 2D C remainder path even for full tiles?
    int cRepackPanel = 0;                        // Size of panels for repacking C (0 = automatic)
    int repackC = 0;                             // Repack C every repackC k loops.
    bool cAccumulators = false;                  // Use accumulator registers for part of C (to save a few registers)?
    bool cLoadAhead = false;                     // Load C before doing FMAs?
    bool autoatomic = true;                      // Automatically use C atomics for beta = 1 kernels?
    bool forceCopyC = false;                     // Force C to be copied before the update step?
    bool noJumpTables = false;                   // Disallow jump tables?
    RemainderHandling remHandling[3] = {         // m, n, k remainder handling.
        RemainderHandling::Split,
        RemainderHandling::Split,
        RemainderHandling::General,
    };
    bool jointSplit = true;                      // Use remainder kernel for both m and n dimensions if both are split.
                                    ZPAD(L, 3)
    int mSplitThresh = 0, nSplitThresh = 0;      // m/n minimum thresholds for using split remainder handling. 0 means always use split.
    bool atomicFMA = false;                      // Use {Atomic} FMA chains.
    bool extendedAtomicFMA = false;              // Use longer {Atomic} FMA chains.
    bool stallAfterLoad = false;                 // Insert stalls after load operations.
    bool checkAdd32 = false;                     // Check inside kernel if inner loop additions can be done in 32-bit.
    bool delayABInc = true;                      // Delay A/B increment a few outer products in the k loop.
                                    ZPAD(M, 3)
    CoopSplit coopA = CoopSplit::K;              // How to split SLM copies, cooperative prefetches amongst threads in a workgroup
    CoopSplit coopB = CoopSplit::K;
    bool slmEarlyKMask = false;                  // Prepare A/B reads to use k-masking (when applicable) in main loop, instead of waiting for remainder.
    bool slmUseIncrCopy = true;                  // Use incremental SLM copies if needed.
    bool slmAltBarriers = false;                 // Alternate fenceless SLM buffering algorithm.
    bool strictFence = false;                    // Add extra SLM fences that are not usually required on HW.
    bool skipFence = false;                      // Skip SLM fences that theoretically should be required but HW doesn't need.
    bool slmFenceWARWA = false;                  // Work around buggy SLM fence that doesn't protect against WAR hazards.
    bool systolic = false;                       // Use systolic array if applicable.
    bool dpasw = false;                          // Use DPASW for fused EU architectures.
    bool fixedSystolic = false;                  // Use hardcoded systolic inner loop for 32x32 or 32x48 unrolls.
                                    ZPAD(N, 3)
    int namedBarriers[2] = {0, 0};               // # of named barriers in m, n dimensions (0 to use regular barriers).
    bool skewLocalIDs = false;                   // Remap local IDs for large workgroups so that threads on the same EU don't depend on the same data.
    bool xParallel = false;                      // TRSM: parallelize in x dimension.
    bool checkBeta1 = false;                     // If true, check for beta = 1 and handle specially.
    bool panelCheck = false;                     // If true, check for out-of-bounds panel reads.
    bool insideSK = false;                       // Inside a superkernel?
                                    ZPAD(O, 3)

    GEMMStrategyPOD() = default;
    GEMMStrategyPOD(ngen::HW hw, int stepping = 0) : CommonStrategy(hw, stepping) {}
};

#undef ZPAD

struct GEMMStrategy : public GEMMStrategyPOD
{
    std::vector<MatrixAddressingStrategy> binary; // Strategies for accessing binary postop data.

    GEMMStrategy() = default;
    GEMMStrategy(ngen::HW hw, int stepping = 0) : GEMMStrategyPOD(hw, stepping) {}

    void preflight(ngen::HW hw, const GEMMProblem &problem);
    int defaultBlockK(ngen::HW hw, const GEMMProblem &problem) const;
    bool minimize(ngen::HW hw, const GEMMProblem &problem);

    void trimKChain(ngen::HW hw, int k, const GEMMProblem &problem);

    void disableAtomics()                                   { C.atomic = CO.atomic = autoatomic = false; }

    int wgTile(LoopType l)                            const { return unroll[l] * wg[l]; }
    int cInterleaveChunk(Type Tc_ext)                 const { return cInterleave ? 64 / Tc_ext : 1; }

    bool lateExit()                                   const { return (std::max(slmBuffersA, slmBuffersB) > 0) || barrierFreq || kParallelLocal || fuseBeta || fusePostOps || cooperativePF; }

    int slmABufBlockSize(const GEMMProblem &problem)  const { return fixedSystolic ? 1152 : int(slmA) * unroll[LoopM] * unrollKSLM * problem.Ta * problem.Ta.components(); }
    int slmBBufBlockSize(const GEMMProblem &problem)  const { return fixedSystolic ? 1536 : int(slmB) * unroll[LoopN] * unrollKSLM * problem.Tb * problem.Tb.components(); }

    int slmGEMMABufSize(const GEMMProblem &problem)   const { return slmABufBlockSize(problem) * wg[LoopM] * wg[LoopK] * slmBuffersA; }
    int slmGEMMBBufSize(const GEMMProblem &problem)   const { return slmBBufBlockSize(problem) * wg[LoopN] * wg[LoopK] * slmBuffersB; }
    int slmABufSize(const GEMMProblem &problem)       const { return slmGEMMABufSize(problem); }
    int slmBBufSize(const GEMMProblem &problem)       const { return slmGEMMBBufSize(problem); }
    int slmSysgemmBlockSize()                         const { return 1152 * wg[LoopM] + 1536 * wg[LoopN]; }
    bool variableSLM()                                const { return kParallelLocal; }
    int slmBarriersPerUnroll()                        const { return (std::max(slmBuffersA, slmBuffersB) == 0) ? 0 :
                                                                     (std::max(slmBuffersA, slmBuffersB) == 1) ? 2 : 1; }

    int ka_inc() const { return slmA ? unrollKSLM : ka_load; }
    int kb_inc() const { return slmB ? unrollKSLM : kb_load; }

    bool persistentLoop()     const { return persistent || kParallelVariable; }

    bool needsMNLocalIDs()    const { return xParallel || (std::max(slmBuffersA, slmBuffersB) > 0) || cooperativePF || kParallelLocal || persistentLoop()
                                                       || namedBarriers[LoopM] || namedBarriers[LoopN] || (dpasw && !fixedSystolic) || (l3PrefetchA || l3PrefetchB); }
    bool needsKLocalIDs()     const { return kParallelLocal || persistentLoop(); }
    bool needsKLoopBarrier()  const { return (barrierFreq > 0) || (std::max(slmBuffersA, slmBuffersB) > 0); }
    bool needsBarrier()       const { return needsKLoopBarrier() || xParallel || kParallelLocal || fuseBeta || fusePostOps; }

    bool needsUnnamedBarrier(const GEMMProblem &problem) const;
    bool needsNamedBarriersM(const GEMMProblem &problem) const;
    bool needsNamedBarriersN(const GEMMProblem &problem) const;

    bool fusedM() const  { return fused && (fusedLoop == LoopM); }
    bool fusedN() const  { return fused && (fusedLoop == LoopN); }

    WGType getWGType(const GEMMProblem &problem) const {
        if (forceWGUpdate == WGFixed)
            return WGFixed;
        if ((std::max(slmBuffersA, slmBuffersB) > 0) || (forceWGUpdate == WGFixed) || namedBarriers[LoopM] || namedBarriers[LoopN])
            return WGFixed;
        if (cooperativePF)
            return WGFixed;     /* until flexible cooperative PF enabled */
        if (cWalkOrder == WalkOrder::NestedLinear)
            return WGFixed;
        if (forceWGUpdate == WGShrinkable)
            return WGShrinkable;
        return WGDynamic;
    }

    bool fixedWG(const GEMMProblem &problem) const { return (getWGType(problem) == WGFixed); }
    bool fixedWGK() const { return forceFixedWGK || kInterleave || (kParallelLocal && wgPadFactor > 1); }

    bool linearOrder() const { return cWalkOrder != WalkOrder::HW2D; }

    bool legalAAlignment(const GEMMProblem &problem, int align) { return problem.isLegalAAlignment(align, unroll[LoopM]); }
    bool legalBAlignment(const GEMMProblem &problem, int align) { return problem.isLegalBAlignment(align, unroll[LoopN]); }

    int repackPeriod(bool isA, const GEMMProblem &problem) const;
    int kAlign(const GEMMProblem &problem) const;

    int statusFlagStride() const { return 64 * (int(fuseBeta) + int(fusePostOps)); }
    bool needsTempC(const GEMMProblem &problem) const;
    bool nondeterministic(const GEMMProblem &problem) const;

    bool checkAdd32Rem() const { return checkAdd32 && emulate.emulate64; }

    bool allowDoubleMasking(LoopType loop) const { return doubleMasking || unroll[loop] == 1; }

    bool registerOutput() const { return C.base.getModel() == ngen::ModelInvalid; }

    bool aqGroupKCompatible(int groupK) const { return groupKCompatible(groupK, slmA ? unrollKSLM : ka_load); }
    bool bqGroupKCompatible(int groupK) const { return groupKCompatible(groupK, slmB ? unrollKSLM : kb_load); }
    static bool groupKCompatible(int groupK, int kStep) { return gcd(groupK, kStep) >= std::min(kStep, 16); }

    void serialize(SerializationStream &s) const
    {
        const GEMMStrategyPOD &pod = *this;
        s.append(pod);
        for (const auto &astrategy: binary)
            s.append(astrategy);
    }
};

bool useAutoAtomic(ngen::HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy, bool ignoreBeta = false);
void downgradeAPFAccess(const GEMMProblem &problem, GEMMStrategy &strategy);
void downgradeBPFAccess(const GEMMProblem &problem, GEMMStrategy &strategy);

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
