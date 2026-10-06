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

#ifndef GEMMSTONE_TOOLS_MULTIDERBY_CLIENT_HPP
#define GEMMSTONE_TOOLS_MULTIDERBY_CLIENT_HPP

#include "multiderby_common.hpp"
#include "multiopt/client.hpp"
#include "multiopt/data.hpp"
#include "gemmstone/problem.hpp"
#include "gemmstone/type.hpp"
#include "tools/benchstone/benchstone.hpp"
#include "tools/benchstone/kernel_info.hpp"

#include <cstdio>
#include <deque>
#include <memory>
#include <optional>
#include <string>


class MultiderbyClient final : public multiopt::Client<ProblemVector, StrategyVector>
{
    using PV = ProblemVector;
    using SV = StrategyVector;
    friend class MultiderbyOptimizer;

public:
    MultiderbyClient(const std::string &stem, const MultiderbyOptions &options_);

    /* PV operations */
    float absRoofline(const PV &pv);
    float distance(const PV &pv1, const PV &pv2);
    PV randomPV();
    bool legalPV(const PV &pv);

    static std::string pvToString(const PV &pv);
    static PV stringToPV(const std::string &problem);

    std::vector<PV> defaultPVGrid();

    float memRoofline(const PV &pv) const;
    float computeRoofline(const PV &pv, bool fmaOnly = false) const;
    bool isMemBound(const PV &pv, bool fmaOnly = false) const;
    bool isLatencyBound(const PV &pv);

    /* SV operations */
    bool legalize(SV &sv, const PV &pv);

    const SV &minLegalSV()  { return minLegal; }
    const SV &maxLegalSV()  { return maxLegal; }

    const SV &minRandomSV() { return minRandom; }
    const SV &maxRandomSV() { return maxRandom; }

    const LegalSVEntries &legalSVEntries() { return legalEntries; }

    SV randomSV(const PV &pv);

    std::string svToString(const SV &sv, bool final = false);
    SV stringToSV(const std::string &strategy);

    /* Benchmarking */
    void preload(const multiopt::AgentRequest<PV, SV> &req);
    void benchmark(multiopt::AgentRequest<PV, SV> &req);
    bool ready(const multiopt::AgentRequest<PV, SV> &req, double maxWaitSeconds = 0.0);

    const std::vector<std::string> &crashedStrategyList() const { return crashedStrategies; }

    // Cumulative time blocked on the frontend; read-and-reset periodically (e.g. -vt).
    double frontendWaitTime() { auto t = frontendWait; frontendWait = 0.0; return t; }

protected:
    std::unique_ptr<benchstone::Driver> bench;
    const MultiderbyOptions &options;

    // Careful (confirmatory) vs. exploratory (fast, noisier) precision tiers.
    int carefulWarmupReps = 2, carefulTimeMinReps = 10, carefulTimeMaxReps = 50;
    double carefulStdDevGoal = 0.005;
    int exploratoryWarmupReps = 1, exploratoryTimeMinReps = 2, exploratoryTimeMaxReps = 6;
    double exploratoryStdDevGoal = 0.03;

    int kernelCacheSize = 8;
    std::deque<std::pair<multiopt::Labeled<SV>, std::string>> preloaded;
    std::vector<std::string> crashedStrategies;

    bool verboseBenchmark = false;
    bool startSeries = false;

    double frontendWait = 0.0;   // Accumulated time blocked on frontend kernel generation.

    ngen::Product product;
    ngen::HW hw;
    std::vector<int> grfModes;
    bool fusedEUs;

    bool doMemBound, doComputeBound, coldCache;
    bool nonPowerOf2Tiles;
    bool checkRarer = false;                    // Expand strategy search space?

    float bps;
    float flops, flopsFMA;

    std::string kernelType, precisions, layouts;
    std::string strategyExtras;

    gemmstone::Type Ta, Tb, Tc, Ta_ext, Tb_ext, Tc_ext;
    gemmstone::MatrixAddressing A, B, C;

    bool dp4aOK, dpasOK;
    bool cRepack;

    int cGRFFill;                               // # C elements that fit in one thread's GRF
    int ssCGRFFill;                             // # C elements that fit in one subslice's GRF

    uint32_t maxM = 16384, maxN = 16384, maxK = 8192;
    uint32_t maxMN = 4096 * 4096;
    uint32_t maxWGTile = 1024;
    uint32_t minMNK = 32768;

    SV minLegal, maxLegal, minRandom, maxRandom;
    LegalSVEntries legalEntries;

    void benchstoneInit();
    std::optional<KernelInfo> benchstonePrepareKernel(const std::string &kernel);
    TestResult benchstoneTest(KernelInfo &kinfo, TestMode mode,
                              uint32_t b, int32_t m, int32_t n, int32_t k,
                              int32_t lda, int32_t ldb, int32_t ldc);

    std::vector<int16_t> &legalEntriesFor(int16_t SV::*field) { return legalEntries[SV::index(field)]; }

    int minOPCount(const SV &sv) const;
    int baseKLoad(const SV &sv) const;
    int baseKLoadScale(const SV &sv, int multiplier) const;
    int baseKLoadUnscale(const SV &sv, int val) const;
    int mVectorLength(const SV &sv) const;

    bool benchSeriesActive = false;
};

#endif
