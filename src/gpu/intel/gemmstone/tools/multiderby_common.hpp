/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_MULTIDERBY_COMMON_HPP
#define GEMMSTONE_TOOLS_MULTIDERBY_COMMON_HPP

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <optional>
#include <vector>

/* Fixed-size arrays with named entries, using CRTP + empty base class optimization. */
template <typename T, typename V>
struct VBase {
    static constexpr size_t size()    { return sizeof(V) / sizeof(T); }
    T  operator[](ptrdiff_t i)  const { return reinterpret_cast<const T*>(this)[i]; }
    T &operator[](ptrdiff_t i)        { return reinterpret_cast<T*>(this)[i]; }

    static size_t index(T V::*field) {
        static const V v{};
        return &(v.*field) - reinterpret_cast<const T*>(&v);
    }

    friend bool operator==(const VBase &v1, const VBase &v2) {
        return !std::memcmp(&v1, &v2, sizeof(V));
    }
    friend bool operator!=(const VBase &v1, const VBase &v2) { return !(v1 == v2); }

    size_t hash() const {
        size_t h = 0;
        for (size_t i = 0; i < size(); i++) {
            h = (h << 27) | (h >> (8 * sizeof(h) - 27));
            h ^= (*this)[i];
        }
        return h;
    }
};

/* Problem vectors (matrix sizes) */
struct ProblemVector : public VBase<uint32_t, ProblemVector>
{
    uint32_t m, n, k;
};

/* Strategy vectors (kernels) */
struct StrategyVector : public VBase<int16_t, StrategyVector>
{
    int16_t wgTileM;       // wgTileM / m vector length
    int16_t wgTileN;
    int16_t lgWGMNK;       // log2(wgM * wgN * wgK)
    int16_t lgWGM;         // log2(wgM)
    int16_t lgWGK;         // log2(wgK)
    int16_t lgWGExpand;    // log2(wgExpand)
    int16_t ikr;
    int16_t computeType;   // ComputeType enum
    int16_t macroType;     // MacroType enum
    int16_t kaLoad;        // ka_load * baseKLoad, special interpretation for <= 0 values
    int16_t kbLoad;        // kb_load * baseKLoad
    int16_t A_copies;
    int16_t B_copies;
    int16_t kaPF;          // ka_prefetch * baseKLoad
    int16_t kbPF;          // kb_prefetch * baseKLoad
    int16_t pfADistScaled; // prefetchA * aBytes; 0 -> no A prefetch
    int16_t pfBDistScaled; // prefetchB * bBytes; 0 -> no B prefetch
    int16_t pfMessage;     // PFMessage enum
    int16_t copyA;
    int16_t copyB;
    int16_t unrollKSLM;    // unrollKSLM * baseKLoad
    int16_t slmCopies;
    int16_t slm3x;
    int16_t splitM;
    int16_t splitN;
    int16_t fullySplitK;
    int16_t grfMode;
    int16_t ql;
    int16_t nb;
    int16_t pfaux;
    int16_t barrierMode;   // BarrierMode enum
    int16_t crMode;
    int16_t splitCxC;
    int16_t persistent;

    enum ComputeType {
        DPAS = 0,
        Axpy = 1,   /* also includes dp4a */
        Dot = 2
    };

    enum MacroType {
        AF = 0,
        XAF = 1,
        XAFStall = 2,
        XAFRR = 3
    };

    enum PFMessage {
        BlockPF = 0,
        DefaultPF = 1,
        CompressedCLPF = 2
    };

    enum BarrierMode {
        SBNone = 0,
        SBLoose = 1,
        SBTight = 2
    };

    enum CRMode {
        CRDefault = 0,
        CRTight = 1,
        CRSeparate = 2
    };

    int lgWGN()         const { return lgWGMNK - lgWGM - lgWGK; }
};

struct MultiderbySettings {
    static constexpr float minLatencyBoundRuntimeMS = 0.005f;
    static constexpr double minRefreshInterval = 0.2;
    static constexpr double maxRefreshInterval = 2.0;
    static constexpr double targetRenderOverhead = 0.02;  // Limit render time to a fraction of total wall time
};

struct MultiderbyOptions {
    bool verboseComm = false;
    bool verboseBenchmark = false;
    bool showSensitivities = false;
    bool showTiming = false;
    bool doMemBound = false;
    bool doComputeBound = false;
    bool includeCBandwidth = false;
    std::optional<bool> nonPowerOf2Tiles;
    std::optional<bool> coldCache;
    int agentCount = 32;
    std::filesystem::path checkpointIn;
    std::filesystem::path checkpointOut;
    std::filesystem::path inputData;
    std::filesystem::path outputData;
    std::filesystem::path outputKernelList;
    std::filesystem::path outputCrashedKernelList;
    int checkpointPeriod = 300;
    int timeLimit = 0;
    std::vector<ProblemVector> problems;
    std::vector<ProblemVector> rooflineProblems;
    std::vector<std::string> strategiesIn;
    bool addRandomProblems = false;
    bool runOptimizer = true;
    bool dumpData = false;
    std::optional<bool> slmA;   // Allow SLM-buffered copies (ca/cab) for A when set; default follows layout.
    std::optional<bool> slmB;   // Allow SLM-buffered copies (cb/cab) for B when set; default follows layout.
};

#endif
