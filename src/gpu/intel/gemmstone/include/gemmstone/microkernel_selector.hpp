/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_SELECTOR_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_MICROKERNEL_SELECTOR_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/kernel_evaluator.hpp"
#include "gemmstone/microkernel/package.hpp"
#include "gemmstone/microkernel/protocol.hpp"

GEMMSTONE_NAMESPACE_START
namespace microkernel {

/* Hardware information for microkernel provider */
struct HWInformation {
    uint32_t gmdid;
    int euCount;
    bool systolicAvailable;
    bool isEfficient64Bit;
};

/* Host kernel facts constraining microkernel register allocation. */
struct HostPayload {
    HostPayload(int simd, int argumentBytes, int liveBytes = 0)
        : simd(simd), argumentBytes(argumentBytes), liveBytes(liveBytes) {}

    int simd;             /* host kernel SIMD width */
    int argumentBytes;    /* payload bytes; microkernel registers are placed above it */
    int liveBytes;        /* host bytes live across the call, competing for the rest */
};

struct GEMMOptions {
    bool localA = false;
    bool localB = false;
    bool addToC = false;
    bool slmPtr = false;
    bool offsetA = false;
    bool offsetB = false;
    bool offsetAHostScalar = false;
    bool offsetBHostScalar = false;
    bool scaleA = false;
    bool scaleB = false;
    bool kParallelLocal = false;

    GEMMOptions() = default;
    ngen::InterfaceHandler generateInterface(ngen::HW hw, HostPayload host) const;
    GEMMOptions transpose() const;
};

/* Main entrypoint for microkernel auto-selection */
using StrategyAdjuster = std::function<void(GEMMStrategy&)>;

/* A microkernel configuration ready for code generation: a catalog or heuristic
   strategy, adjusted, sized for a GRF mode, and preflighted. Each candidate owns
   its problem, as building a candidate updates it. */
struct GEMMCandidate {
    GEMMProblem problem;
    GEMMStrategy strategy;

#ifndef STANDALONE
    /* Serialization for kernel cache. */
    void serialize(SerializationStream &s) const
    {
        s.append(problem, strategy);
    }
#endif
};

/* Ranked microkernel candidates for one request, built without generating code.
   The problem's C must be column-major. */
class GEMMSelector {
public:
    GEMMSelector(const GEMMOptions &options, HostPayload host, HWInformation hwInfo, SizeParams sizes,
                 const GEMMProblem &problem,
                 const std::vector<StrategyRequirement> &reqs = std::vector<StrategyRequirement>(),
                 StrategyAdjuster strategyAdjuster = {});

    /* Builds the next candidate that passes preflight, in order of preference.
       Returns false once the candidates are exhausted. */
    bool next(GEMMCandidate &candidate);

private:
    bool prepare(const kcatalog::Entry *entry, GEMMCandidate &candidate) const;
    bool finish(const kcatalog::Entry *entry, GEMMCandidate &candidate) const;

    GEMMOptions options;
    HostPayload host;
    HWInformation hwInfo;
    SizeParams sizes;
    GEMMProblem baseProblem;
    std::vector<StrategyRequirement> reqs;
    StrategyAdjuster strategyAdjuster;
    ngen::HW hw = ngen::HW::Unknown;
    int stepping = 0;
    EvaluateAuxOutput auxParams;
    std::vector<const kcatalog::Entry *> entries;
    size_t nextEntry = 0;
    bool pendingLargeGRF = false;
    const kcatalog::Entry *largeGRFEntry = nullptr;
    GEMMCandidate largeGRFCandidate;
};

/* Generates the microkernel for a candidate. Throws on failure. */
Package generateGEMM(const GEMMOptions &options, HostPayload host, HWInformation hwInfo,
                     const GEMMCandidate &candidate);

Package selectGEMM(const GEMMOptions &options, HostPayload host, HWInformation hwInfo, SizeParams sizes,
                                     const GEMMProblem &problem,
                                     const std::vector<StrategyRequirement> &reqs = std::vector<StrategyRequirement>(),
                                     StrategyAdjuster strategyAdjuster = {}, SelectionObserver *observer = nullptr);

/* Transitional overload for callers that do not yet pass a HostPayload.
   Defaults the host payload base to r8, which may mismatch the host kernel and
   clobber the OpenCL kernel arguments. Use selectGEMM with HostPayload.
*/
Package selectGEMM(const GEMMOptions &options, HWInformation hwInfo, SizeParams sizes,
                                     const GEMMProblem &problem,
                                     const std::vector<StrategyRequirement> &reqs = std::vector<StrategyRequirement>(),
                                     StrategyAdjuster strategyAdjuster = {}, SelectionObserver *observer = nullptr);

/* Helpers */
static inline int alignmentForLD(int ld)
{
    for (int x = 1; x <= 64; x <<= 1)
        if (ld & x) return x;
    return 128;
};

}
GEMMSTONE_NAMESPACE_END

#endif /* header guard */
