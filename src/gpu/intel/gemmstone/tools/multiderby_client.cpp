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

#include "multiderby_client.hpp"

#include "multiopt/agent.hpp"

#include "gemmstone/strategy_parser.hpp"

#include "tools/benchstone/benchstone.hpp"
#include "tools/benchstone/kernel_info.hpp"
#include "tools/frontend.hpp"

#include "derby_utils.hpp"
#include "hw_queries.hpp"
#include "vt100_codes.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <optional>

using PV = ProblemVector;
using SV = StrategyVector;

using namespace multiopt;
using namespace ngen::utils;
using ngen::HW;
using namespace gemmstone;

namespace {

std::vector<std::unique_ptr<frontend::async::Job>> multiFrontendJobs;

void startFrontendJob(const std::string &kernel)
{
    auto it = std::find_if(multiFrontendJobs.begin(), multiFrontendJobs.end(),
                           [&](const auto &elem) { return elem->kernel.empty(); });
    if (it != multiFrontendJobs.end()) {
        (*it)->kernel = kernel;
        (*it)->launch();
        return;
    }
    multiFrontendJobs.emplace_back(std::make_unique<frontend::async::Job>(kernel));
}

std::optional<frontend::async::Job *> getFrontendJob(const std::string &kernel)
{
    auto it = std::find_if(multiFrontendJobs.begin(), multiFrontendJobs.end(),
                           [&](const auto &elem) { return elem->kernel == kernel; });
    if (it == multiFrontendJobs.end()) return std::nullopt;
    return it->get();
}

} // namespace

MultiderbyClient::MultiderbyClient(const std::string &stem, const MultiderbyOptions &options_)
        : doMemBound(options.doMemBound), doComputeBound(options.doComputeBound), verboseBenchmark(options.verboseBenchmark),
          // Compile-pipeline depth scales with agentCount, clamped to [4, 128].
          kernelCacheSize(std::min(std::max(options_.agentCount * 2, 4), 128)),
          options(options_)
{
    // Stem parsing.
    std::stringstream ss(stem);
    std::string s;

    kernelType.clear();
    ss >> s >> std::ws;
    kernelType = s;
    while (utils::kernelModifier(s) != utils::KernelModifier::None) {
        if (!(ss >> s)) fail("Missing kernel type");
        ss >> std::ws;
        kernelType += ' ' + s;
    }

    ss >> precisions;
    auto cp = precisions.c_str();
    cp = parsePrecisions(cp, Ta_ext, Ta);
    cp = parsePrecisions(cp, Tb_ext, Tb);
    cp = parsePrecisions(cp, Tc, Tc_ext);

    ss >> layouts;
    cp = layouts.c_str();
    cp = parseLayout(cp, A);
    cp = parseLayout(cp, B);
    cp = parseLayout(cp, C);

    if (A.alignment < Ta_ext.paddedSize()) A.setAlignment(Ta_ext.paddedSize());
    if (B.alignment < Tb_ext.paddedSize()) B.setAlignment(Tb_ext.paddedSize());
    if (C.alignment < Tc_ext.paddedSize()) C.setAlignment(Tc_ext.paddedSize());

    if (!ss) fail("Could not parse input");

    while (!ss.eof()) {
        ss >> s;
        strategyExtras += ' ';
        strategyExtras += s;
    }

    cRepack = Ta.isInteger() && Tb.isInteger() && Tc.isFP();

    // HW-specific initialization.
    product = detectDevice(false);
    hw = ngen::getCore(product.family);
    grfModes.clear();
    if (hw < HW::XeHP) {
        grfModes = {128};
    } else if (hw < HW::Xe3p) {
        grfModes = {128, 256};
    } else if (product.family == ngen::ProductFamily::CRI) {
        grfModes = {192, 512};
    } else {
        grfModes = {128, 192, 256, 512};
    }

    fusedEUs = (hw >= HW::Gen12LP && hw <= HW::XeHPG);

    dp4aOK = Ta.isInt8() && Tb.isInt8() && (hw >= HW::Gen12LP);

    dpasOK = false;
    if (hw >= HW::XeHP) {
        bool cOK = one_of(Tc.real(), {Type::s32, Type::u32, Type::f32});
        if (hw >= HW::XeHPC)
            cOK |= one_of(Tc.real(), {Type::f16, Type::bf16});
        bool aOK = (Ta.real().paddedSize() < 4);
        bool bOK = (Tb.real().paddedSize() < 4);
        if (hw >= HW::XeHPC) {
            aOK |= Ta.real() == Type::tf32;
            bOK |= Tb.real() == Type::tf32;
        }
        dpasOK = aOK && bOK && cOK;
    }

    bool dotOK = (A.layout == MatrixLayout::T) && (B.layout == MatrixLayout::N);
    bool pfOK = (hw >= HW::XeHPC);
    bool nbOK = (hw >= HW::XeHPC);
    bool pfauxOK = contains(kernelType, "scale") || contains(kernelType, "offset") || contains(strategyExtras, "pfaux");

    // Temporarily disable to avoid a hang
    nbOK &= hw < HW::Xe3p;

    // Gather memory + compute peaks.
    if (auto e = getenv("MEMORY_BW"))
        bps = std::atof(e) * 1e9f;
    else
        fail("Missing memory BW; please specify via MEMORY_BW=<GB/s> environment variable");

    auto getFlops = [&](bool sys) {
        return peakFlops(product, euCount(), getDeviceMHz() * 1e6, Ta, Tb, Tc, sys);
    };

    flops = getFlops(dpasOK);
    flopsFMA = getFlops(false);

    // Check whether we are doing compute-bound and/or memory-bound problems.
    if (!doComputeBound && !doMemBound)
        for (const auto &pv: options.problems)
            (isMemBound(pv) ? doMemBound : doComputeBound) = true;

    if (!doComputeBound && !doMemBound)
        doComputeBound = doMemBound = true;

    // Bounds on problem space.
    if (!options.problems.empty()) {
        maxM = maxN = maxK = 1;
        for (const auto &pv: options.problems) {
            maxM = std::max(maxM, pv.m);
            maxN = std::max(maxN, pv.n);
            maxK = std::max(maxK, pv.k);
        }
    } else {
        if (Tc.paddedSize() >= 8) maxMN /= (Tc.size() / 4);
        if (Tc.isComplex()) maxK >>= 1;
    }

    if (dpasOK)
        minMNK <<= 6;

    // Decide on cold cache usage.
    coldCache = options.coldCache ? *options.coldCache : doMemBound;

    // Decide on power-of-2 tile usage.
    nonPowerOf2Tiles = options.nonPowerOf2Tiles ? *options.nonPowerOf2Tiles : !options.problems.empty();

    // Choose admissible WG tile sizes.
    auto &wgTileMs = legalEntriesFor(&SV::wgTileM);
    if (options.nonPowerOf2Tiles)
        wgTileMs = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64};
    else
        wgTileMs = {1, 2, 4, 8, 16, 32, 64};

    auto &wgTileNs = legalEntriesFor(&SV::wgTileN);
    wgTileNs = {1, 2, 4, 8};
    if (options.nonPowerOf2Tiles)
        wgTileNs.push_back(12);
    for (int dyad = 4; dyad <= (maxWGTile >> 3); dyad <<= 1) {
        wgTileNs.push_back(dyad * 4);
        if (options.nonPowerOf2Tiles) {
            wgTileNs.push_back(dyad * 5);
            wgTileNs.push_back(dyad * 6);
            wgTileNs.push_back(dyad * 7);
        }
    }
    wgTileNs.push_back(maxWGTile);

    // Set up boundaries on SV space.
    auto rbounds = [&](int16_t SV::*field, int16_t loR, int16_t hiR, int16_t lo, int16_t hi) {
        minLegal.*field = lo;
        maxLegal.*field = hi;
        minRandom.*field = loR;
        maxRandom.*field = hiR;
    };

    auto cbounds = [&](int16_t SV::*field, int16_t lo, int16_t hi) {
        rbounds(field, lo, hi, lo, hi);
    };

    auto lbounds = [&](int16_t SV::*field) {
        const auto &l = legalEntriesFor(field);
        cbounds(field, l.front(), l.back());
    };

    auto binary   = [&](int16_t SV::*field)            { cbounds(field, 0, 1); };
    auto binaryIf = [&](int16_t SV::*field, bool cond) { cbounds(field, 0, cond ? 1 : 0); };

    bool checkRare = false;

    int maxMVL = grfBytes() / Tc;
    cGRFFill = grfModes.back() * maxMVL;
    ssCGRFFill = cGRFFill * threadsPerSS(grfModes.back());
    int lgSSThreads = ilog2(threadsPerSS(grfModes.front()));

    int minMult = 0;
    SV testSV;
    for (auto ct: {SV::DPAS, SV::Dot, SV::Axpy}) {
        testSV.computeType = ct;
        minMult = std::min(minMult, ilog2(minOPCount(testSV)) - ilog2(baseKLoad(testSV)));
    }
    minMult++;

    auto maxComputeType = (dpasOK ? SV::DPAS : SV::Axpy);

    lbounds(&SV::wgTileM);
    lbounds(&SV::wgTileN);
    rbounds(&SV::lgWGMNK, lgSSThreads - 2, lgSSThreads, 0, lgSSThreads);
    cbounds(&SV::lgWGM, 0, lgSSThreads);
    cbounds(&SV::lgWGK, 0, lgSSThreads);
    if (doMemBound) {
        rbounds(&SV::lgWGExpand, 0, 0, 0, 2);
        binary(&SV::ikr);
        cbounds(&SV::computeType, maxComputeType, dotOK ? SV::Dot : SV::Axpy);
    } else {
        cbounds(&SV::lgWGExpand, 0, 0);
        cbounds(&SV::ikr, 0, 0);
        cbounds(&SV::computeType, maxComputeType, maxComputeType);
    }
    cbounds(&SV::macroType, SV::AF, dpasOK ? SV::XAFRR : SV::AF);
    rbounds(&SV::kaLoad, 1, 4, minMult, 16);
    rbounds(&SV::kbLoad, 1, 4, minMult, 16);
    rbounds(&SV::A_copies, 1, 2, 1, checkRare ? 4 : 2);
    rbounds(&SV::B_copies, 1, 2, 1, checkRare ? 4 : 2);
    if (pfOK) {
        rbounds(&SV::kaPF, 1, 4, minMult, 4);
        rbounds(&SV::kbPF, 1, 4, minMult, 4);
        rbounds(&SV::pfADistScaled, 0, 256, 0, 1024);
        rbounds(&SV::pfBDistScaled, 0, 256, 0, 1024);
    } else {
        cbounds(&SV::kaPF, 1, 1);
        cbounds(&SV::kbPF, 1, 1);
        cbounds(&SV::pfADistScaled, 0, 0);
        cbounds(&SV::pfBDistScaled, 0, 0);
    }
    cbounds(&SV::pfMessage, SV::BlockPF, SV::DefaultPF /* SV::CompressedCLPF */);
    // Constrain copyA/copyB based on CLI --slm options and layout defaults.
    // Default: disallow SLM when operand is K-major (A if T, B if N).
    // - A/B access strategies exist that format data to be directly compatible with fma/systolic instructions.
    //   Using SLM for data manipulation is not required, and can explode generation times, especially for small data types.
    int allowSLMA = options.slmA.value_or(A.layout != MatrixLayout::T) ? 1 : 0;
    int allowSLMB = options.slmB.value_or(B.layout != MatrixLayout::N) ? 1 : 0;
    cbounds(&SV::copyA, 0, allowSLMA);
    cbounds(&SV::copyB, 0, allowSLMB);
    rbounds(&SV::unrollKSLM, 1, 4, minMult, 16);
    rbounds(&SV::slmCopies, 1, 2, 1, checkRare ? 4 : 2);
    binary(&SV::slm3x);
    binary(&SV::splitM);
    binary(&SV::splitN);
    binaryIf(&SV::fullySplitK, A.layout == MatrixLayout::N || B.layout == MatrixLayout::T);
    cbounds(&SV::grfMode, 0, grfModes.size() - 1);
    binary(&SV::ql);
    binaryIf(&SV::nb, nbOK);
    cbounds(&SV::pfaux, pfauxOK, pfauxOK);
    cbounds(&SV::barrierMode, SV::SBNone, SV::SBTight);
    cbounds(&SV::crMode, SV::CRDefault, cRepack ? SV::CRSeparate : SV::CRDefault);
    binaryIf(&SV::splitCxC, Tc.isComplex());
    binaryIf(&SV::persistent, hw >= HW::XeHPC);

    // Benchstone launch and setup.
    benchstoneInit();
}

void MultiderbyClient::benchstoneInit()
{
    if (!options.runOptimizer) return;

    if (bench) {
        if (bench->isKernelLoaded())
            bench->finishKernel();
        bench->finish();
    }

    bench = benchstone::createDriver(benchstone::RuntimeBackend::OpenCL);
    bench->Quiet = true;
    bench->WarmupReps = carefulWarmupReps;
    bench->TimeMinReps = carefulTimeMinReps;
    bench->TimeMaxReps = carefulTimeMaxReps;
    bench->RelativeMeanStdDevGoal = carefulStdDevGoal;
    if (coldCache) {
        bench->ColdA = true;
        bench->ColdB = true;
    }
    bench->init();
    benchSeriesActive = false;
    startSeries = true;
    if (bench->isSimulation()) {
        bench->WarmupReps=0;
        bench->TimeMinReps = 1;
        bench->TimeMaxReps = 1;
    }
}

std::optional<KernelInfo> MultiderbyClient::benchstonePrepareKernel(const std::string &kernel)
{
    if (bench->isKernelLoaded())
        bench->finishKernel();

    frontend::Result res;
    auto genStart = std::chrono::steady_clock::now();
    if (auto job = getFrontendJob(kernel); job.has_value()) {
        res = job.value()->finish();
    } else {
        auto generator = frontend::Generator::Create();
        res = generator->generate(kernel);
    }
    frontendWait += (std::chrono::steady_clock::now() - genStart) / std::chrono::duration<double>(1);

    if (!res.success) {
        if (options.verboseComm)
            std::cerr << "Failed to generate kernel for strategy " << kernel << ": " << res.info << std::endl;
        return std::nullopt;
    }

    if (options.verboseComm)
        std::cerr << "Kernel generation success for strategy " << kernel << std::endl;

    KernelInfo info(res.info);
    if (!bench->initKernel(info, res.binary)) {
        if (options.verboseBenchmark)
            std::cerr << "Failed to initialize kernel for strategy " << kernel << ": " << res.info << std::endl;
        return std::nullopt;
    }

    if (options.verboseBenchmark)
        std::cerr << "Kernel initialization success for strategy " << kernel << std::endl;

    return info;
}

TestResult MultiderbyClient::benchstoneTest(KernelInfo &kinfo, TestMode mode,
                                            uint32_t b, int32_t m, int32_t n,
                                            int32_t k, int32_t lda, int32_t ldb, int32_t ldc)
{
    TestResult result = bench->test(kinfo, mode, b, m, n, k, lda, ldb, ldc,
                                    false, false, false, 'F', false, false);
    if (result.accuracy == TestAccuracy::RuntimeError && options.verboseBenchmark)
        std::cerr << "Runtime error while benchmarking: " << result.failingCase << std::endl;
    return result;
}

float MultiderbyClient::memRoofline(const PV &pv) const
{
    float memBytes = float(pv.m) * (pv.k * Ta_ext)
                   + float(pv.n) * (pv.k * Tb_ext);
    if (options.includeCBandwidth)
        memBytes += float(pv.m) * (pv.n * Tc_ext);

    return bps / memBytes;
}

float MultiderbyClient::computeRoofline(const PV &pv, bool fmaOnly) const
{
    float compute = 2.0f * pv.m * pv.n * pv.k;
    return (fmaOnly ? flopsFMA : flops) / compute;
}

float MultiderbyClient::absRoofline(const PV &pv)
{
    return std::min(memRoofline(pv), computeRoofline(pv));
}

bool MultiderbyClient::isMemBound(const PV &pv, bool fmaOnly) const
{
    return (memRoofline(pv) < computeRoofline(pv, fmaOnly));
}

bool MultiderbyClient::isLatencyBound(const PV &pv)
{
    // Problems faster than MultiderbySettings::minLatencyBoundRuntimeMS at peak performance
    // are excluded from auto-generated problem sets (-pc/-pm/-pr).
    float peakTimeMS = 1000.0f / absRoofline(pv);

    return (peakTimeMS < MultiderbySettings::minLatencyBoundRuntimeMS);
}

float MultiderbyClient::distance(const PV &pv1, const PV &pv2)
{
    float dm = std::log2(float(pv1.m) / float(pv2.m));
    float dn = std::log2(float(pv1.n) / float(pv2.n));
    float dk = std::log2(float(std::max(pv1.k, 1u)) / float(std::max(pv2.k, 1u)));
    return std::sqrt(dm*dm + dn*dn + dk*dk);
}

PV MultiderbyClient::randomPV()
{
    const float logMaxM = std::log2(float(maxM));
    const float logMaxN = std::log2(float(maxN));

    PV pv;
    do {
        pv.m = std::floor(std::exp2(randFloat(1, logMaxM)));
        pv.n = std::floor(std::exp2(randFloat(1, logMaxN)));

        auto sk = randFloat(1, std::sqrt(float(maxK)));
        pv.k = std::floor(sk * sk);
    } while (!legalPV(pv));
    return pv;
}

bool MultiderbyClient::legalPV(const PV &pv)
{
    return (float(pv.m) * pv.n <= maxMN)
        && (float(pv.m) * pv.n * pv.k >= minMNK)
        && (isMemBound(pv) ? doMemBound : doComputeBound)
        && !isLatencyBound(pv);
}

std::string MultiderbyClient::pvToString(const PV &pv)
{
    return std::to_string(pv.m) + 'x' + std::to_string(pv.n) + 'x' + std::to_string(pv.k);
}

PV MultiderbyClient::stringToPV(const std::string &problem)
{
    PV pv;
    char separator;
    std::stringstream s(problem);
    s >> pv.m >> std::ws >> separator >> pv.n >> std::ws >> separator >> pv.k;
    if (!s) fail("Failed to parse problem size. Expected format: MxNxK, got " + problem);
    return pv;
}

std::vector<PV> MultiderbyClient::defaultPVGrid()
{
    std::vector<PV> grid;

    /* Isolate non-power-of-2 factors of EU count, e.g. 224 = 7 * 2^5 -> 7 */
    auto np2 = ssCount();
    while ((np2 & 1) == 0)
        np2 >>= 1;

    /* Choose k values */
    std::vector<int> allK;
    for (int k = maxK; k >= 64; k >>= 1)
        allK.push_back(k);

    auto add = [&](PV pv) {
        for (auto &k: allK) {
            pv.k = k;
            if (legalPV(pv))
                grid.push_back(pv);
        }
    };

    grid.reserve(ilog2(maxM) * ilog2(maxN) * allK.size());

    /* Power-of-2 m/n grid for memory-bound sizes */
    for (uint32_t m = 1; m <= maxM; m *= 2) {
        for (uint32_t n = 1; n <= maxN; n *= 2) {
            PV pv{{}, m, n, maxK};
            if (np2 == 1 || isMemBound(pv))
                add(pv);
        }
    }

    /* Power-of-2 SS tile size grid for compute-bound sizes */
    if (np2 > 1) for (uint32_t m = 1; m <= maxM; m *= 2) {
        for (uint32_t n = np2; n <= maxN; n *= 2) {
            PV pv{{}, m, n, maxK};
            if (!isMemBound(pv))
                add(pv);
        }
    }

    return grid;
}

bool MultiderbyClient::legalize(SV &sv, const PV &pv)
{
    bool memboundFMA = isMemBound(pv, true);

    // Only allow dot for GEMVs.
    if (sv.computeType == SV::Dot)
        if (pv.m > 1 && pv.n > 1)
            sv.computeType = SV::Axpy;

    // Some strategies should only be searched for memory bound cases, or only compute bound cases.
    if (memboundFMA)
        sv.macroType = SV::AF;
    else {
        sv.lgWGExpand = 0;
        if (dpasOK)
            sv.computeType = SV::DPAS;
    }

    int mVL = mVectorLength(sv);

    // Some strategy knobs are irrelevant/not helpful without other knobs.
    if (sv.computeType != SV::DPAS)
        sv.macroType = SV::AF;
    if (sv.computeType == SV::Dot)
        sv.copyA = sv.copyB = 0;

    if (sv.copyA || sv.copyB) {
        sv.fullySplitK = 0;         /* not yet supported */
        sv.barrierMode = SV::SBNone;
    } else {
        sv.unrollKSLM = 1;
        sv.slmCopies = 1;
        sv.slm3x = 0;
        sv.ql = 0;
        sv.nb = 0;
    }
    if (sv.pfADistScaled == 0) sv.kaPF = 1;
    if (sv.pfBDistScaled == 0) sv.kbPF = 1;
    if (sv.lgWGK == 0) sv.ikr = 0;

    // Tile legalization.
    int16_t maxLgWG = ilog2(threadsPerSS(grfModes[sv.grfMode]));
    for (auto *lw: {&sv.lgWGM, &sv.lgWGK, &sv.lgWGMNK})
        *lw = std::min(*lw, maxLgWG);
    sv.lgWGMNK = std::max<int16_t>(sv.lgWGMNK, sv.lgWGM + sv.lgWGK);
    while (true) {
        if (sv.lgWGMNK <= maxLgWG) break;
        if (sv.lgWGK > 0) sv.lgWGMNK--, sv.lgWGK--;
        if (sv.lgWGMNK <= maxLgWG) break;
        if (sv.lgWGM > 0) sv.lgWGMNK--, sv.lgWGM--;
    }
    sv.lgWGExpand = std::min<int16_t>(sv.lgWGExpand, maxLgWG - sv.lgWGMNK);

    auto lgWGN = sv.lgWGMNK - sv.lgWGM - sv.lgWGK;

    sv.wgTileM = align_up(sv.wgTileM, 1 << sv.lgWGM);

    sv.wgTileN = align_up(sv.wgTileN, 1 << lgWGN);
    if (!checkRarer) {
        int align = 4 << lgWGN;
        if (pv.n > align)
            sv.wgTileN = align_up(sv.wgTileN, align);
        else
            sv.wgTileN = roundup_pow2(sv.wgTileN);
    }

    if (fusedEUs && sv.lgWGM == 0 && lgWGN == 0)
        return false;

    auto wgM = 1 << sv.lgWGM, wgN = 1 << lgWGN;

    auto wgTileM = sv.wgTileM * mVL;
    int unrollM = wgTileM >> sv.lgWGM;
    int unrollN = sv.wgTileN >> lgWGN;

    // m/n cooperation legalization.
    if (unrollM % wgN) sv.splitM = 0;
    if (unrollN % wgM) sv.splitN = 0;

    bool ska = sv.fullySplitK && !sv.splitM && A.layout == MatrixLayout::N;
    bool skb = sv.fullySplitK && !sv.splitN && B.layout == MatrixLayout::T;
    sv.fullySplitK = ska || skb;

    int kaAlignCoop = ska ? wgM * wgN :
               !sv.splitM ? wgN :
                            1;
    int kbAlignCoop = skb ? wgM * wgN :
               !sv.splitN ? wgM :
                            1;

    // k chunk legalization.
    int16_t minMult = ilog2(minOPCount(sv)) - ilog2(baseKLoad(sv)) + 1;
    sv.kaLoad     = std::max(sv.kaLoad, minMult);
    sv.kbLoad     = std::max(sv.kbLoad, minMult);
    sv.kaPF       = std::max(sv.kaPF, minMult);
    sv.kbPF       = std::max(sv.kbPF, minMult);
    sv.unrollKSLM = std::max(sv.unrollKSLM, minMult);

    int kaLoad = baseKLoadScale(sv, sv.kaLoad);
    int kbLoad = baseKLoadScale(sv, sv.kbLoad);
    int unrollKSLM = baseKLoadScale(sv, sv.unrollKSLM);

        /* Only allow power-of-2 k chunk sizes for SLM copies */
    bool pow2ALoad = sv.copyA || (Ta.perByte() > 1 && A.layout == MatrixLayout::T);
    bool pow2BLoad = sv.copyB || (Tb.perByte() > 1 && B.layout == MatrixLayout::N);

    if (pow2ALoad) {
        kaLoad = roundup_pow2(kaLoad);
        sv.kaLoad = baseKLoadUnscale(sv, kaLoad);
    }
    if (pow2BLoad) {
        kbLoad = roundup_pow2(kbLoad);
        sv.kbLoad = baseKLoadUnscale(sv, kbLoad);
    }

    // SLM legalization.
    if (sv.copyA && sv.copyB) {
        if (sv.computeType != SV::DPAS) {
            kbLoad = kaLoad;
            sv.kbLoad = sv.kaLoad;
        }
        if (kaLoad * (sv.A_copies - 1) != kbLoad * (sv.B_copies - 1)) {
            kbLoad = kaLoad;
            sv.kbLoad = sv.kaLoad;
            sv.B_copies = sv.A_copies;
        }
    }

    if (sv.copyA || sv.copyB) {
        unrollKSLM = roundup_pow2(unrollKSLM);
        if (sv.copyA) unrollKSLM = std::max({unrollKSLM, kaLoad, kaAlignCoop});
        if (sv.copyB) unrollKSLM = std::max({unrollKSLM, kbLoad, kbAlignCoop});
        sv.unrollKSLM = baseKLoadUnscale(sv, unrollKSLM);
    }

    // Prefetch tidying and legalization.
    int pfAlign = baseKLoad(sv) >> 1;
    sv.pfADistScaled = align_up(sv.pfADistScaled, pfAlign * Ta);
    sv.pfBDistScaled = align_up(sv.pfBDistScaled, pfAlign * Tb);

    sv.kaPF = baseKLoadUnscale(sv, align_up(baseKLoadScale(sv, sv.kaPF), kaAlignCoop));
    sv.kbPF = baseKLoadUnscale(sv, align_up(baseKLoadScale(sv, sv.kbPF), kbAlignCoop));

    // Named barrier legalization.
    if (sv.nb) {
        int nb = 1;
        if (sv.copyA) nb += wgM << sv.lgWGK;
        if (sv.copyB) nb += wgN << sv.lgWGK;
        if (nb >= 32) sv.nb = 0;
    }

    // Roughly lower bound register usage, and adjust GRF mode to accomodate if possible.
    // Otherwise just fail this SV.
    int cExpand = (sv.computeType == SV::Dot) ? (grfBytes() / Tc) : 1;
    if (sv.crMode == SV::CRSeparate) cExpand *= 2;
    if (Tc.isComplex() && !sv.splitCxC) cExpand *= 2;

    int minBytes = unrollM * kaLoad * sv.A_copies * Ta * Ta.components()
                 + unrollN * kbLoad * sv.B_copies * Tb * Tb.components()
                 + unrollM * unrollN              * Tc * Tc.components() * cExpand;
    if (sv.copyA) minBytes += (unrollM * unrollKSLM * Ta) >> lgWGN;
    if (sv.copyB) minBytes += (unrollN * unrollKSLM * Tb) >> sv.lgWGM;

    while (minBytes > grfModes[sv.grfMode] * grfBytes())
        if (++sv.grfMode >= grfModes.size())
            return false;

    // After configuring GRF mode, check for legal workgroup size.
    if ((threadsPerSS(grfModes[sv.grfMode]) >> sv.lgWGMNK >> sv.lgWGExpand) == 0)
        return false;

    return true;
}

SV MultiderbyClient::randomSV(const PV &pv)
{
    bool membound    = isMemBound(pv);
    bool memboundFMA = isMemBound(pv, true);

    while (true) {
        // Mostly rely on standard random SV generation.
        SV sv;
        for (int i = 0; i < SV::size(); i++) {
            auto &l = legalEntries[i];
            if (l.empty())
                sv[i] = randInt(minRandom[i], maxRandom[i]);
            else
                sv[i] = l[randInt(l.size())];
        }

        // Only allow dot for GEMVs.
        if (sv.computeType == SV::Dot)
            if (pv.m > 1 && pv.n > 1)
                sv.computeType = SV::Axpy;
        if (sv.computeType == SV::Axpy && dpasOK && !memboundFMA)
            sv.computeType = SV::DPAS;

        // Choose reasonable WG tile size given PV:
        //   a) WG tile should not be much larger than PV
        //   b) If compute bound, fewer than 4 full waves, unless tile size >= 0.25*GRF
        //   c) Aspect ratio should be no more than 2x minimum possible.
        int mVL = mVectorLength(sv);
        int wgPerWave = ssCount() * threadsPerSS(grfModes[sv.grfMode])
                            >> sv.lgWGMNK >> sv.lgWGExpand;

        auto maxWGTileM = std::max(8u, roundup_pow2(pv.m));
        auto maxWGTileN = std::max(8u, roundup_pow2(pv.n));

        auto &wgTileMs = legalEntriesFor(&SV::wgTileM);
        auto &wgTileNs = legalEntriesFor(&SV::wgTileN);

        uint32_t maxWGTileMIdx = wgTileMs.size() - 1;
        while (wgTileMs[maxWGTileMIdx] * mVL > maxWGTileM && maxWGTileMIdx > 0)
            maxWGTileMIdx--;
        maxWGTileM = wgTileMs[maxWGTileMIdx];

        uint32_t maxWGTileNIdx = wgTileNs.size() - 1;
        while (wgTileNs[maxWGTileNIdx] > maxWGTileN && maxWGTileNIdx > 0)
            maxWGTileNIdx--;
        maxWGTileN = wgTileNs[maxWGTileNIdx];

        int maxTilesPerGRF = cRepack ? 8 : 4;

        while (true) {
            uint32_t wgTileMIdx = randInt<uint32_t>(maxWGTileMIdx + 1);     // (a)
            uint32_t wgTileNIdx = randInt<uint32_t>(maxWGTileNIdx + 1);     // (a)
            auto wgTileM = wgTileMs[wgTileMIdx] * mVL;
            auto wgTileN = wgTileNs[wgTileNIdx];
            if (!membound && (wgTileM * wgTileN < maxWGTileM * maxWGTileN))
                if (wgTileM * wgTileN * maxTilesPerGRF < ssCGRFFill)
                    if (wgTileM * wgTileN * 4 * wgPerWave <= pv.m * pv.n)
                        continue;                                           // (b)
            if (wgTileM < pv.m && wgTileN < pv.n)
                if (wgTileM > 2 * wgTileN || wgTileN > 2 * wgTileM)
                    continue;                                               // (c)
            sv.wgTileM = wgTileMs[wgTileMIdx];
            sv.wgTileN = wgTileN;
            break;
        }

        // Better sampling of wgM/N/K. lgWGMNK has been chosen already.
        do {
            sv.lgWGM = randInt<int16_t>(0, sv.lgWGMNK);
            sv.lgWGK = randInt<int16_t>(0, sv.lgWGMNK);
        } while (sv.lgWGM + sv.lgWGK > sv.lgWGMNK);

        // Improve odds of testing no-prefetch cases:
        sv.pfADistScaled = std::max(0, randInt(-32, int(maxRandom.pfADistScaled)));
        sv.pfBDistScaled = std::max(0, randInt(-32, int(maxRandom.pfBDistScaled)));

        // ... and legalize. If at first you don't succeed: try, try again.
        if (!legalize(sv, pv))
            continue;

        if ((sv.wgTileM >> sv.lgWGM) > 8)
            continue;

        return sv;
    }
}

std::string MultiderbyClient::svToString(const SV &sv, bool final)
{
    char abModel = 'a', cModel = 'a';

    // Interpret SV entries.
    int wgTileM = sv.wgTileM * mVectorLength(sv);
    int wgTileN = sv.wgTileN;
    int unrollM = wgTileM >> sv.lgWGM;
    int unrollN = wgTileN >> sv.lgWGN();
    int wgM = 1 << sv.lgWGM;
    int wgN = 1 << sv.lgWGN();
    int wgK = 1 << sv.lgWGK;
    int wgExpand = 1 << sv.lgWGExpand;
    int grfs = grfModes[sv.grfMode];

    int kaLoad = baseKLoadScale(sv, sv.kaLoad);
    int kbLoad = baseKLoadScale(sv, sv.kbLoad);
    int kaPrefetch = baseKLoadScale(sv, sv.kaPF);
    int kbPrefetch = baseKLoadScale(sv, sv.kbPF);
    int unrollKSLM = baseKLoadScale(sv, sv.unrollKSLM);
    int prefetchA = std::max<int>(0, sv.pfADistScaled) / Ta;
    int prefetchB = std::max<int>(0, sv.pfBDistScaled) / Tb;

    if (hw < HW::XeHPC && (unrollM * unrollN * 4 >= cGRFFill))
        abModel = 's';

    bool sys = (sv.computeType == SV::DPAS);
    bool dot = (sv.computeType == SV::Dot);
    bool dp4a = !sys && !dot && dp4aOK;

    int align2D = 65536, align2DT = 65536, align2DPF = 65536;
    if (hw >= HW::Xe3p)
        align2D = align2DT = align2DPF = 4;
    else if (hw >= HW::Xe2)
        align2D = align2DT = align2DPF = 16;
    else if (hw >= HW::XeHPC) {
        align2D = 8;
        align2DT = align2DPF = 4;
    }

    bool canBlock2DA   = (A.alignment >= align2D)  && !dot;
    bool canBlock2DB   = (B.alignment >= align2D)  && !dot;
    bool canBlock2DTB  = (B.alignment >= align2DT) && !dot;
    bool canBlock2DPFA = (A.alignment >= align2DPF);
    bool canBlock2DPFB = (B.alignment >= align2DPF);

    bool doBlock2DA  = canBlock2DA  && (sys || (A.layout == MatrixLayout::T && Ta_ext.real().paddedSize() <= 4) || unrollM * Ta_ext <= 64);
    bool doBlock2DB  = canBlock2DB  && (sys || (B.layout == MatrixLayout::N && Tb_ext.real().paddedSize() <= 4) || unrollN * Tb_ext <= 64);
    bool doBlock2DTB = canBlock2DTB && (sys || (B.layout == MatrixLayout::N && Tb_ext.real().paddedSize() <= 4) || unrollN * Tb_ext <= 64);

    if (A.layout == MatrixLayout::T && Ta_ext.real().paddedSize() == 8)
        doBlock2DA  &= (unrollM * Ta_ext.real() % 64 == 0);  /* only height = 8 for d64 transpose 2D block */
    if (B.layout == MatrixLayout::N && Tb_ext.real().paddedSize() == 8) {
        doBlock2DB  &= (unrollN * Tb_ext.real() % 64 == 0);
        doBlock2DTB &= (unrollN * Tb_ext.real() % 64 == 0);
    }

    char aAccess, bAccess, cAccess = 'B';
    bool forceFN = (wgM == 1);
    bool forceSN = false;

    auto useVNNI = [](Type T) { return one_of(T.bits(), {8, 16}); };

    switch (A.layout) {
        case MatrixLayout::N:
            aAccess = 'B';
            if ((sys || dp4a) && doBlock2DA)
                aAccess = useVNNI(Ta_ext) ? 'v' : 'm';
            if (dot)
                aAccess = 'S';
            break;
        case MatrixLayout::T:
            aAccess = doBlock2DA ? 't' : 'S';
            if (dot)
                aAccess = 'B';
            break;
        default:  aAccess = 'B'; break;
    }

    switch (B.layout) {
        case MatrixLayout::N:
            bAccess = 'S';
            if (sys || dp4a)
                bAccess = doBlock2DB ? 'm' : 'B';
            if (dot)
                bAccess = 'B';
            break;
        case MatrixLayout::T:
            bAccess = 'B';
            if (sys || dp4a) {
                if (doBlock2DB && sys)
                    bAccess = !sv.copyB ? 't' : useVNNI(Tb_ext) ? 'v' : 'm';
                else if (doBlock2DTB && sys) {
                    bAccess = 't';
                    forceSN = true;
                } else
                    bAccess = 'S';
            }
            if (dot)
                bAccess = 'B';
            break;
        default:  bAccess = 'B'; break;
    }

    if (hw < HW::XeHPC) {
        if (aAccess == 'S' && sv.copyA) aAccess = 'B';      // TODO: make configurable
        if (bAccess == 'S' && sv.copyB) bAccess = 'B';
        if (cAccess == 'B' && C.layout == MatrixLayout::T) forceFN = true;
        if (cAccess != 'B' && C.layout == MatrixLayout::N) forceFN = true;
    }

    if (unrollN == 1 && !doBlock2DB && !dot)
        bAccess = (B.layout == MatrixLayout::T) ? 'S' : 'B';

    if (hw < HW::XeHPG) {
        aAccess = std::tolower(aAccess);                    // Use HDC messages if LSC not available
        bAccess = std::tolower(bAccess);
        cAccess = std::tolower(cAccess);
    }

    bool block2DA = isBlock2D(aAccess);
    bool block2DB = isBlock2D(bAccess);

    // SLM usage check.
    int slm = 0;
    if (sv.copyA) slm += wgTileM * unrollKSLM * Ta * Ta.components();
    if (sv.copyB) slm += wgTileN * unrollKSLM * Tb * Tb.components();
    slm *= wgK;

    int wgPerSS = (threadsPerSS(grfs) >> sv.lgWGMNK) >> sv.lgWGExpand;
    int slmBuffers = std::min(4, maxSLM() / (std::max(wgPerSS, 1) * std::max(slm, 1)));
    if (slmBuffers <= 0) return "";

    if (sv.slm3x && slmBuffers == 4)
        slmBuffers = 3;

    // Prefetch accesses.
    std::string aPFAccess = (A.layout == MatrixLayout::T ? "S1," : "B");
    std::string bPFAccess = (B.layout == MatrixLayout::N ? "S1," : "B");

    if (unrollM * Ta_ext <= 64) aPFAccess = "S";
    if (unrollN * Tb_ext <= 64) bPFAccess = "S";

    bool pf1D = (sv.pfMessage == SV::BlockPF);

    bool noPF2DA = pf1D && canBlock2DPFA && aPFAccess[0] == 'B';
    bool noPF2DB = pf1D && canBlock2DPFB && bPFAccess[0] == 'B';
    if (canBlock2DPFA && !noPF2DA) aPFAccess = "m";
    if (canBlock2DPFB && !noPF2DB) bPFAccess = "m";
    if (sv.pfMessage == SV::CompressedCLPF) aPFAccess = bPFAccess = "C";

    if (hw < HW::XeHPG) {
        aPFAccess = lowercase(aPFAccess);
        bPFAccess = lowercase(bPFAccess);
    }

    // Create strategy string.
    std::string spec;
    spec.reserve(128);
    spec = kernelType;
    spec += ' ';
    spec += precisions;
    spec += ' ';
    spec += layouts;
    spec += ' ';
    spec += std::to_string(unrollM);
    spec += ' ';
    spec += std::to_string(unrollN);
    spec += ' ';
    spec += abModel;
    spec += aAccess;
    spec += std::to_string(kaLoad);
    if (sv.A_copies > 1) {
        spec += 'x';
        spec += std::to_string(sv.A_copies);
    }
    if (prefetchA) {
        spec += '+';
        spec += aPFAccess;
        spec += std::to_string(kaPrefetch);
        spec += '@';
        spec += std::to_string(prefetchA);
    }
    spec += ' ';

    spec += abModel;
    spec += bAccess;
    spec += std::to_string(kbLoad);
    if (sv.B_copies > 1) {
        spec += 'x';
        spec += std::to_string(sv.B_copies);
    }
    if (prefetchB) {
        spec += '+';
        spec += bPFAccess;
        spec += std::to_string(kbPrefetch);
        spec += '@';
        spec += std::to_string(prefetchB);
    }
    spec += ' ';

    spec += cModel;
    spec += cAccess;

    spec += " wg ";
    spec += std::to_string(wgM);
    spec += 'x';
    spec += std::to_string(wgN);

    if (wgK > 1) {
        spec += 'x';
        spec += std::to_string(wgK);
        spec += sv.ikr ? " ikr" : " kr";
    }

    if (wgExpand > 1) spec += " wx" + std::to_string(wgExpand);

    if (!sys && !dot && !dp4a && kaLoad > 1 && kbLoad > 1 && !Tc.isInteger()) {
        spec += " kc";
        spec += std::to_string(std::min(kaLoad, kbLoad));
    }

    if (dot) spec += " dot";

    if (sv.copyA || sv.copyB) {
        spec += " c";
        if (sv.copyA) spec += 'a';
        if (sv.copyB) spec += 'b';
        spec += std::to_string(slmBuffers);
        if (sv.slmCopies > 1) {
            spec += 'x';
            spec += std::to_string(sv.slmCopies);
        }
        spec += " ks";
        spec += std::to_string(unrollKSLM);
    }

    if (sv.ql)
        spec += " ql";

    if (sv.pfaux)
        spec += " pfaux";

    if (sv.nb) {
        int nbM = sv.copyA ? wgM : 1;
        int nbN = sv.copyB ? wgN : 1;
        if (nbM == 1 && nbN == 1)
            nbM = wgM, nbN = wgN;
        if (nbM == 1) nbM = 0;
        if (nbN == 1) nbN = 0;
        if (nbM || nbN) {
            spec += " nb ";
            spec += std::to_string(nbM);
            spec += 'x';
            spec += std::to_string(nbN);
        }
    }

    if (sys) {
        spec += " sys";
        if (hw < HW::XeHPC)
            if ((unrollN % 2 == 0) && (unrollN < 8 || (unrollN % 8 == 0)))
                spec += " dw";

        const char *macroStrings[4] = {" af", " xaf", " xaf st", " xaf rr"};
        spec += macroStrings[sv.macroType % 4];
        if (sv.macroType > 0 && unrollN * Tc.complexComponents() > 32)
            spec += " fx";
    }

    if (forceFN)
        spec += " fn nmk";

    if (sys && ((Ta.components() == 1 && Tb.components() == 1) || Tc.isComplex()))
        spec += " vav";
    else if (hw >= HW::XeHP)
        spec += " nse";
    else
        spec += " acb";

    bool useHilbert = (hw == HW::XeHPC) || !final;  // Use Hilbert rather than boustrophedon to reduce the chance that
                                                    //  related WGs land on the same SS causing anomalously high performance
                                                    //  that only applies to the special test sizes.

    if (sv.persistent)
        spec += " li pt";
    else
        spec += useHilbert ? " hi" : " bo";

    if (final)
        spec += " sr";
    else if ((sv.splitCxC || (Tc.paddedSize() < 4)) && hw >= HW::XeHPC)
        spec += " sr";
    else
        spec += " ar";                              // For speed in kernel generation
    if (cAccess == 'B' && hw >= HW::XeHPC)
        spec += " br";

    if (sv.barrierMode != SV::SBNone) {
        int sb = (sv.barrierMode == SV::SBTight) ? 1 : std::min(256 / Ta, 256 / Tb);
        spec += " sb";
        spec += std::to_string(sb);
    }

    if (hw < HW::XeHPG)
        spec += " bk0";

    if (sv.splitM)
        if (sv.copyA || prefetchA)
            spec += " sm";
    if (sv.splitN || forceSN)
        if (sv.copyB || prefetchB)
            spec += " sn";
    if (sv.fullySplitK) {
        if (A.layout == MatrixLayout::N)
            spec += " ska";
        if (B.layout == MatrixLayout::T)
            spec += " skb";
    }

    if (sv.splitCxC) {
        spec += " xc";
        if (sys)
            spec += " cc";
    }

    spec += " dm";

    if (sv.crMode == SV::CRSeparate)
        spec += " cr0";
    else if (sv.crMode == SV::CRTight)
        spec += " cr" + std::to_string(mVectorLength(sv));

    if (grfs != 128) {
        spec += " grf";
        spec += std::to_string(grfs);
    }

    if ((Ta.components() > 1 || Tb.components() > 1) && Ta.isFP() && Tb.isFP())
        if (!contains(strategyExtras, "cl"))
            spec += " cl1";

    if (abModel == 's' && !contains(strategyExtras, "pab") && !contains(strategyExtras, "np"))
        spec += " pab";

    if (!strategyExtras.empty()) {
        spec += ' ';
        spec += strategyExtras;
    }

    return spec;
}

SV MultiderbyClient::stringToSV(const std::string &strategy)
{
    std::stringstream s(strategy);
    SV sv;

    for (size_t i = 0; i < sv.size(); i++)
        sv[i] = 0;

    auto has = [&](const char *word) { return contains(strategy, word); };
    auto hasArg = [&](const std::string_view &word, bool attached) -> std::string {
        auto pos = strategy.find(word);
        if (pos == strategy.npos) return "";
        pos += word.size();
        if (!attached) while (pos < strategy.size() && strategy[pos] == ' ') pos++;
        std::string arg;
        for (; pos < strategy.size() && strategy[pos] != ' '; pos++)
            arg += strategy[pos];
        return arg;
    };

    if (has("dot"))
        sv.computeType = SV::Dot;
    else if (has("sys"))
        sv.computeType = SV::DPAS;
    else
        sv.computeType = SV::Axpy;

    int mVL = mVectorLength(sv);

    int unrollM = 0, unrollN = 0;
    int wgM = 1, wgN = 1, wgK = 1;
    std::string aLoad, bLoad, cLoad;

    s >> unrollM >> unrollN >> aLoad >> bLoad >> cLoad;
    if (!s || unrollM == 0 || unrollN == 0)
        fail("Could not parse input strategy: " + strategy);

    auto wgArg = hasArg("wg", false);
    if (wgArg.empty()) wgArg = "2x8";

    std::stringstream wgs(wgArg);
    char eat;
    wgs >> wgM >> std::ws >> eat >> wgN;
    if (has("kr"))
        wgs >> std::ws >> eat >> wgK;

    sv.wgTileM = unrollM * wgM / mVL;
    sv.wgTileN = unrollN * wgN;

    sv.lgWGMNK = ilog2(wgM * wgN * wgK);
    sv.lgWGM = ilog2(wgM);
    sv.lgWGK = ilog2(wgK);

    if (auto arg = hasArg("wx", true); !arg.empty())
        sv.lgWGExpand = ilog2(std::stoi(arg));

    sv.ikr = has("ikr");

    sv.macroType = SV::AF;
    if (has("xaf")) {
        if (has("rr"))      sv.macroType = SV::XAFRR;
        else if (has("st")) sv.macroType = SV::XAFStall;
        else                sv.macroType = SV::XAF;
    }

    auto ignoreSlash = [](std::stringstream &s, auto &val) {
        char eat;
        if (s.peek() == '/') s >> eat >> val;
    };

    auto parseAB = [&](const std::string &xLoad, int16_t &kLoadScaled, int16_t &copies, int16_t &kPFScaled, int16_t &pfDist, int16_t &pfMessage) {
        copies = 1;
        kPFScaled = 0;
        pfDist = 0;
        std::stringstream s(xLoad);
        char eat, pfType;
        int ieat, kLoad, kPF;

        s >> eat >> eat;
        ignoreSlash(s, eat);
        s >> kLoad;
        kLoadScaled = baseKLoadUnscale(sv, kLoad);
        ignoreSlash(s, ieat);
        if (s.peek() == '#') s >> eat >> ieat >> eat >> ieat;
        if (s.peek() == 'x') s >> eat >> copies;
        if (s.peek() == '+') {
            // As in parseStrategy: +<access><k_prefetch>[,<k_pfStride>][@<distance>]. The SV holds the stride,
            //   which defaults to k_prefetch (svToString writes scattered prefetches as S1,<stride>).
            s >> eat >> pfType >> kPF;
            if (s.peek() == ',') s >> eat >> kPF;
            if (s.peek() == '@') s >> eat >> pfDist;
            pfType = std::tolower(pfType);
            if (pfType == 'c') pfMessage = SV::CompressedCLPF;  // TODO: BlockPF
            kPFScaled = baseKLoadUnscale(sv, kPF);
        }
    };

    sv.pfMessage = SV::DefaultPF;
    parseAB(aLoad, sv.kaLoad, sv.A_copies, sv.kaPF, sv.pfADistScaled, sv.pfMessage);
    parseAB(bLoad, sv.kbLoad, sv.B_copies, sv.kbPF, sv.pfBDistScaled, sv.pfMessage);
    sv.pfADistScaled = int16_t(sv.pfADistScaled * Ta);
    sv.pfBDistScaled = int16_t(sv.pfBDistScaled * Tb);

    std::string slmArg = "";
    if (auto arg = hasArg("ca", true); !arg.empty()) {
        slmArg = arg;
        sv.copyA = true;
    } else if (auto arg = hasArg("cb", true); !arg.empty()) {
        slmArg = arg;
        sv.copyB = true;
    } else if (auto arg = hasArg("cab", true); !arg.empty()) {
        slmArg = arg;
        sv.copyA = sv.copyB = true;
    }

    if (!slmArg.empty()) {
        sv.slm3x = (slmArg[0] == '3');
        if (slmArg.size() > 3 && slmArg[1] == 'x')
            sv.slmCopies = slmArg[2] - '0';
        sv.unrollKSLM = 1;
        if (auto arg = hasArg("ks", true); !arg.empty())
            sv.unrollKSLM = baseKLoadUnscale(sv, std::stoi(arg));
    }

    sv.splitM = has("sm");
    sv.splitN = has("sn");
    sv.fullySplitK = has("ska") || has("skb");

    int grf = 128;
    if (auto arg = hasArg("grf", true); !arg.empty())
        grf = std::stoi(arg);
    sv.grfMode = int16_t(grfModes.size() - 1);
    for (size_t i = 0; i + 1 < grfModes.size(); i++)
        if (grf <= grfModes[i])
            sv.grfMode = i;

    sv.ql = has("ql");
    sv.nb = has("nb");
    sv.pfaux = has("pfaux");

    sv.barrierMode = SV::SBNone;
    if (auto arg = hasArg("sb", true); !arg.empty())
        sv.barrierMode = (std::stoi(arg) <= 32) ? SV::SBTight : SV::SBLoose;

    if (has("cr0"))
        sv.crMode = SV::CRSeparate;
    else if (auto arg = hasArg("cr", true); !arg.empty())
        if (std::stoi(arg) <= mVL)
            sv.crMode = SV::CRTight;

    sv.splitCxC = has("xc");
    sv.persistent = has("pt");

    return sv;
}

void MultiderbyClient::preload(const AgentRequest<PV, SV> &req)
{
    if (!options.runOptimizer) return;

    // Avoid duplicate preloads of the same SV already queued.
    for (auto &p: preloaded)
        if (p.first == req.sv)
            return;

    auto sstr = svToString(req.sv);
    if (sstr.empty())
        return;

    preloaded.push_back({req.sv, sstr});

    if (int(preloaded.size()) <= kernelCacheSize)
        startFrontendJob(sstr);
}

bool MultiderbyClient::ready(const AgentRequest<PV, SV> &req, double maxWaitSeconds)
{
    auto sstr = svToString(req.sv);
    if (sstr.empty()) return true;               // no kernel needed; won't block.

    auto job = getFrontendJob(sstr);
    if (!job.has_value()) return true;           // not tracked as an async compile.
    return (*job)->ready(std::chrono::duration<double>(maxWaitSeconds));
}

void MultiderbyClient::benchmark(AgentRequest<PV, SV> &req)
{
    // Search the whole preloaded queue; this SV may not be the oldest entry.
    auto it = std::find_if(preloaded.begin(), preloaded.end(),
                           [&](auto &p) { return p.first == req.sv; });
    bool cached = it != preloaded.end();

    // Decode SV to strategy string.
    std::string sstr;
    if (cached) {
        sstr = std::move(it->second);
        preloaded.erase(it);
    } else
        sstr = svToString(req.sv);

    if (sstr.empty())
        return;

    // Prepare kernel.
    auto kernelInfo = benchstonePrepareKernel(sstr);

    // Collect benchmark data if kernel generated successfully.
    if (kernelInfo.has_value()) {

        if (verboseBenchmark)
            std::cerr << "Testing " << vt(VT_CYAN) << sstr << vt(VT_RESET) << " on " << req.pvs.size() << " problems\n";

        if (startSeries) {
            startSeries = false;
            (void) benchstoneTest(*kernelInfo, TestMode::BeginSeries, 1, maxM, maxN, maxK, 0, 0, 0);
            benchSeriesActive = true;
        }

        req.perf.resize(req.pvs.size());
        for (auto &p: req.perf)
            p.value = p.variance = std::numeric_limits<float>::quiet_NaN();

        bool crashed = false;

        auto collect = [&](size_t start, size_t end) {
            if (crashed) return;
            for (size_t i = start; i < end; i++) {
                auto &pv = req.pvs[i].vec;
                auto mode = benchSeriesActive ? TestMode::TimeOneOfSeries : TestMode::Time;
                auto result = benchstoneTest(*kernelInfo, mode, 1, pv.m, pv.n, pv.k, 0, 0, 0);
                if (result.accuracy == TestAccuracy::RuntimeError) {
                    crashed = true;
                    return;
                }

                // Transform (time, stddev of time) -> (1/time, variance of 1/time).
                double time = result.time;
                double stddev = result.meanStddevBestAvailable();
                double istddev = stddev / (time * time);
                req.perf[i].value = 1. / time;
                req.perf[i].variance = istddev * istddev;
                if (!std::isfinite(req.perf[i].variance))
                    req.perf[i].variance = 0.f;
            }
        };

        auto bail = (req.minPerfCareful > 0.f) ? (1. / req.minPerfCareful) : 0.;
        bench->PerfBailThreshold = bail;

        // Fast/loose timing for exploratory probes; careful timing once a candidate
        // has cleared the coarse minPerf0 bar.
        auto setPrecision = [&](bool careful) {
            if (bench->isSimulation()) return;  // leave minimal-rep simulation settings alone
            bench->WarmupReps            = careful ? carefulWarmupReps    : exploratoryWarmupReps;
            bench->TimeMinReps           = careful ? carefulTimeMinReps   : exploratoryTimeMinReps;
            bench->TimeMaxReps           = careful ? carefulTimeMaxReps   : exploratoryTimeMaxReps;
            bench->RelativeMeanStdDevGoal = careful ? carefulStdDevGoal   : exploratoryStdDevGoal;
        };

        if (req.minPerf0 <= 0.0f) {
            setPrecision(false);
            collect(0, req.pvs.size());
        } else {
            setPrecision(false);
            collect(0, 1);
            if (req.perf[0].value >= req.minPerf0) {
                setPrecision(true);
                collect(1, req.pvs.size());
            }
        }

        if (crashed) {
            // Reset and remember this kernel if runtime failed.
            for (auto &p: req.perf)
                p.value = p.variance = std::numeric_limits<float>::quiet_NaN();
            crashedStrategies.push_back(sstr);
        }
    }

    // Keep the pipeline topped up.
    if (int(preloaded.size()) >= kernelCacheSize)
        startFrontendJob(preloaded[kernelCacheSize - 1].second);
}

int MultiderbyClient::mVectorLength(const SV &sv) const
{
    switch (sv.computeType) {
        case SV::DPAS: return grfBytes() >> 2;
        case SV::Dot:  return 1;
        default:       return grfBytes() / Tc;
    }
}

int MultiderbyClient::minOPCount(const SV &sv) const
{
    switch (sv.computeType) {
        case SV::DPAS: {
            int count = std::min(32 / Ta.real(), 32 / Tb.real());
            return (product.family == ngen::ProductFamily::CRI) ? count * 2 : count;
        }
        case SV::Dot:  return grfBytes() / Tc;
        default:       return dp4aOK ? 4 : 1;
    }
}
int MultiderbyClient::baseKLoad(const SV &sv) const
{
    switch (sv.computeType) {
        case SV::DPAS: return minOPCount(sv);
        case SV::Dot:  return minOPCount(sv) * 2;
        default:       return std::min({16, 32 / Ta, 32 / Tb});
    }
}

int MultiderbyClient::baseKLoadScale(const SV &sv, int multiplier) const
{
    int base = baseKLoad(sv);
    int min = minOPCount(sv);

    if (multiplier > 0)
        return base * multiplier;
    else
        return std::max(min, base >> (1 - multiplier));
}

int MultiderbyClient::baseKLoadUnscale(const SV &sv, int val) const
{
    int base = baseKLoad(sv);
    if (val >= base)
        return div_up(val, base);
    else
        return 1 - ilog2(base / val);
}
