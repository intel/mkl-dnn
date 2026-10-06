/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "benchstone.hpp"
#include "benchstone_common.hpp"
#include "kernel_info.hpp"
#include "tools/exit_codes.hpp"
#include "tools/frontend.hpp"
#include "tools/utils.hpp"
#include "gemmstone/type.hpp"
#include "gemmstone/driver_info.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif

bool DoCrossingCheck = false;

std::vector<std::unique_ptr<frontend::async::Job>> FrontendJobs;

void startJob(const std::string &kernel) {
    // Find an existing finished job to re-launch
    auto it = std::find_if(FrontendJobs.begin(), FrontendJobs.end(),
        [&](const auto &elem) { return elem->kernel.empty(); });
    
    if (it != FrontendJobs.end()) {
        (*it)->kernel = kernel;
        (*it)->launch();
        return;
    }

    // Add a new job
    FrontendJobs.emplace_back(std::make_unique<frontend::async::Job>(kernel));
}

std::optional<frontend::async::Job*> getJob(const std::string &kernel) {
    auto it = std::find_if(FrontendJobs.begin(), FrontendJobs.end(),
        [&](const auto &elem) { return elem->kernel == kernel; });

    if (it == FrontendJobs.end()) return std::nullopt;
    return (*it).get();
}

const char *units(bool inSeconds, bool memBound) { 
    return inSeconds ? "s" :
            memBound ? " GB/s"
                     : "%";
}

const char *csvunits(bool inSeconds, bool memBound) {
    return inSeconds ? "" :
            memBound ? ""
                     : "%";
}

const char *csvheader(bool inSeconds, bool memBound) {
    return inSeconds ? "Time (s)" :
            memBound ? "GB/s"
                     : "Efficiency (%)";
}

TestResult test(benchstone::Driver &driver, bool serverMode, KernelInfo &info, TestMode mode,
                uint32_t batchSize, int32_t m, int32_t n, int32_t k,
                int32_t lda, int32_t ldb, int32_t ldc, bool conja, bool conjb,
                bool unit, char cOffType, bool aOffset, bool bOffset) {
    try {
        auto result = driver.test(info, mode, batchSize, m, n, k, lda, ldb, ldc, conja, conjb, unit,
                                  cOffType, aOffset, bOffset);
        if (!serverMode && result.accuracy == TestAccuracy::RuntimeError) {
            std::cerr << result.failingCase << std::endl;
#ifndef _WIN32
            __asm("int3");
#endif
            exit(EXIT_FAILURE);
        }
        return result;
    } catch (std::exception &e) {
        std::cerr << e.what() << std::endl;
#ifndef _WIN32
        __asm("int3");
#endif
        exit(EXIT_FAILURE);
    }
}

void inputSize(const char *arg, std::initializer_list<int32_t*> out)
{
    auto fail = [=]{ throw std::runtime_error(std::string("Error parsing size argument ") + arg); };

    std::stringstream ss(arg);
    bool first = true;

    for (auto optr: out) {
        if (!first) {
            char x;
            ss >> std::ws >> x;
            if (x != 'x') fail();
        }
        ss >> *optr;
        first = false;
    }
    if (!ss) fail();
    ss >> std::ws;
    if (!ss.eof()) fail();
}

std::vector<BlockingInfo> inputSKParallel(benchstone::Driver &bench, std::istream &s)
{
    std::vector<BlockingInfo> blocking;

    while (!s.eof()) {
        BlockingInfo entry;
        char c;

        s >> std::ws >> c;
        if (s.eof()) break;
        switch (c) {
            case 'm': entry.var = LoopM; break;
            case 'n': entry.var = LoopN; break;
            case 'k': entry.var = LoopK; break;
            case 'f': bench.FixedBlocking = true; continue;
            case 'b': s >> bench.BiasOverride; continue;
            case 'h': s >> bench.HilbertBailOverride; continue;
            case 's': s >> bench.BoustrophedonSliceOverride; continue;
            default: goto done;
        }
        s >> entry.val;

        blocking.push_back(entry);
    }

done:
    return blocking;
}

FusedDriverParams inputFusedParams(std::istream &s)
{
    FusedDriverParams params;

    while (!s.eof()) {
        char c;
        int val;

        s >> std::ws >> c;
        if (s.eof()) break;
        s >> val;

        switch (c) {
            case 'w': params.majorWalkDim = val; break;
            case 'b': params.blockMN      = val; break;
            case 'k': params.blockK       = val; break;
            case 'z': params.initialZ0    = val; break;
            case 'Z': params.stdZ0        = val; break;
            case 'm': params.footprint[0] = val; break;
            case 'n': params.footprint[1] = val; break;
            case 't': params.tidThresh    = val; break;
            case 'r': params.reuse        = val; break;
            default: goto done;
        }
    }

done:
    return params;
}

void printKernelInfo(const KernelInfo &info)
{
    if (info.isCopy())
        std::cout << info.copy[0].problem;
    else {
        auto &problem = info.l3[0].problem;
        int sp = 0;
        size_t trim;
        for (trim = problem.size() - 1; trim >= 0; trim--) {
            if (problem[trim] == ' ') sp++;
            if (sp == 2) break;
        }
        trim++;

        std::cout << problem.substr(0, trim);
        std::cout << info.l3[0].unroll[LoopM] << ' ';
        std::cout << info.l3[0].unroll[LoopN] << ' ';
        std::cout << problem.substr(trim, std::string::npos);
    }
}

TestResult accuracyTests(benchstone::Driver &bench, KernelInfo &info, bool serverMode)
{
    (void)serverMode;
    bool oldQuiet = bench.Quiet;
    auto result = bench.runAccuracyTests(info);
    bench.Quiet = oldQuiet;
    return result;
}

template <typename T>
struct Sequence {
    T start = 0, end = 0, stride = 0;

    Sequence() {}
    Sequence(T val) : start(val), end(val), stride(0) {}

    T count()           const { return (stride == 0) ? 1 : (1 + std::max<T>((end - start) / stride, 0)); }
    T operator[](int i) const { return start + i * stride; }
    T max()             const { return std::max(start, end); }
};

template <typename T>
std::istream &operator>>(std::istream &s, Sequence<T> &seq)
{
    char eat;
    s >> seq.start;
    if (s.peek() == '-') {
        s >> eat >> seq.end;
        if (s.peek() == ':')
            s >> eat >> seq.stride;
        else
            seq.stride = T(1);
    } else {
        seq.end = seq.start;
        seq.stride = T(0);
    }
    return s;
}

struct BatchCase {
    Sequence<int32_t> m, n, k, b = 1;
    Sequence<int32_t> lda = -1, ldb = -1, ldc = -1;
};

BatchCase parseBatchCase(std::istream &s, const KernelInfo &info, bool tightLD = false)
{
    BatchCase bcase;

    if (info.isBatched())
        s >> bcase.b;
    s >> bcase.m >> bcase.n >> bcase.k;

    if (!s.eof())
        s >> bcase.lda >> bcase.ldb >> bcase.ldc;
    else if (tightLD)
        bcase.lda = bcase.ldb = bcase.ldc = 0;

    return bcase;
}

void runBatch(benchstone::Driver &bench, const char *batchFile, KernelInfo &info, bool derbyOutput, bool secondsOutput) {
    std::ifstream batchLines(batchFile);
    if (!batchLines) {
        fprintf(stderr, "Could not open batch file %s\n", batchFile);
        exit(1);
    }

    batchLines >> std::skipws;

    std::vector<BatchCase> input;
    int32_t maxM = 1, maxN = 1, maxK = 1, maxBatch = 1;
    int64_t single_elems = 0;
    int icheck = 0;
    double minCheckPerf = 0., checkPerf = 0.;
    bool perfCheckFailed = false;
    bool tightLD = false;
    bool memoryBound = false;

    batchLines >> std::ws;

    while (!batchLines.eof()) {
        auto char0 = batchLines.peek();
        if (char0 == '#') {
            std::string comment;
            std::getline(batchLines, comment);
        } else if (char0 == '!') {
            std::string directive;

            batchLines >> directive;
            if (directive == "!membound")
                memoryBound = true;
            else if (directive == "!seconds")
                secondsOutput = true;
            else if (directive == "!tight_ld")
                tightLD = true;
            else if (directive == "!warmup_reps")
                batchLines >> bench.WarmupReps;
            else if (directive == "!time_reps") {
                batchLines >> bench.TimeMinReps;
                batchLines >> std::ws;
                if (batchLines.peek() == '-') {
                    char eat;
                    batchLines >> eat >> bench.TimeMaxReps;
                } else
                    bench.TimeMaxReps = bench.TimeMinReps;
            } else if (directive == "!min_perf") {
                if (icheck != 0)
                    throw std::runtime_error("Multiple !min_perf directives found");
                icheck = int(input.size());
                batchLines >> minCheckPerf;
            } else if (directive == "!stddev_goal") {
                batchLines >> bench.RelativeMeanStdDevGoal >> std::ws;
            } else
                throw std::runtime_error("Unknown directive in batch file");
        } else {
            std::string lineStr;
            std::getline(batchLines, lineStr);
            std::stringstream line(lineStr);

            BatchCase bcase;

            if (info.isBatched())
                line >> bcase.b;
            line >> bcase.m >> bcase.n >> bcase.k;

            if (!line.eof())
                line >> bcase.lda >> bcase.ldb >> bcase.ldc;
            else if (tightLD)
                bcase.lda = bcase.ldb = bcase.ldc = 0;

            maxM = std::max(maxM, bcase.m.max());
            maxN = std::max(maxN, bcase.n.max());
            maxK = std::max(maxK, bcase.k.max());
            maxBatch = std::max(maxBatch, bcase.b.max());
            auto sumSequence = [](const auto &seq) -> int64_t {
                int64_t sum = 0;
                for (int i = 0; i < seq.count(); i++)
                    sum += seq[i];
                return sum;
            };
            int64_t sumB = sumSequence(bcase.b);
            int64_t sumM = sumSequence(bcase.m);
            int64_t sumN = sumSequence(bcase.n);
            int64_t sumK = sumSequence(bcase.k);
            int64_t countM = bcase.m.count();
            int64_t countN = bcase.n.count();
            int64_t countK = bcase.k.count();
            // Equivalent to sum(B*(MN+MK+NK)) over all cases
            single_elems += sumB * (countK * sumM * sumN
                                  + countN * sumM * sumK
                                  + countM * sumN * sumK);
            input.push_back(bcase);
        }

        if (batchLines.bad()) {
            fprintf(stderr, "Error reading batch file.\n");
            exit(1);
        }

        batchLines >> std::ws;
    }

    // Use series mode if it decreases the total amount of elements required to initialize
    int64_t series_elems = maxBatch * (maxM * maxN + maxM * maxK + maxN * maxK);
    bool series = series_elems < single_elems;

    if (series)
        test(bench, false, info, TestMode::BeginSeries, maxBatch, maxM, maxN, maxK, -1, -1, -1, false, false, false, 'F', false, false);

    bool OldQuiet = bench.Quiet;
    bench.Quiet = true;
    std::vector<double> outputs;

    bool showEff = bench.getEnableMetrics() && info.isL3();

    if (OldQuiet && !derbyOutput) {
        if (info.isBatched())
            printf("Batch size, ");
        printf("m, n, k, %s", csvheader(secondsOutput, memoryBound));
        if (showEff)
            printf(", Effective %s", csvheader(secondsOutput, memoryBound));
        printf("\n");
    }
    for (size_t i = 0; i < input.size(); i++) {
        const auto &bseq = input[i].b, &mseq = input[i].m, &nseq = input[i].n, &kseq = input[i].k;
        const auto &ldaseq = input[i].lda, &ldbseq = input[i].ldb, &ldcseq = input[i].ldc;

        if (icheck > 0 && i == icheck) {
            // Check for minimum performance and bail if not met.
            if (secondsOutput) {
                checkPerf = *std::min_element(outputs.begin(), outputs.end());
                perfCheckFailed = (checkPerf > minCheckPerf);
            } else {
                checkPerf = *std::max_element(outputs.begin(), outputs.end());
                perfCheckFailed = (checkPerf < minCheckPerf);
            }
            if (perfCheckFailed) {
                if (!OldQuiet)
                    yprintf("Minimum performance not met; terminating batch operation.\n");
                break;
            }
        }

        int count = std::max({bseq.count(), mseq.count(), nseq.count(), kseq.count(), ldaseq.count(), ldbseq.count(), ldcseq.count()});
        for (int c = 0; c < count; c++) {
            auto b = bseq[c], m = mseq[c], n = nseq[c], k = kseq[c];
            auto lda = ldaseq[c], ldb = ldbseq[c], ldc = ldcseq[c];
            auto mode = series ? TestMode::TimeOneOfSeries : TestMode::Time;
            auto result = test(bench, false, info, mode, b, m, n, k, lda, ldb, ldc, false, false, false, 'F', false, false);
            double output = secondsOutput ? result.time
                            : memoryBound ? result.bw * 1e-9
                                          : result.efficiency;
            double effOutput = secondsOutput ? result.effTime
                               : memoryBound ? result.bw * 1e-9
                                             : result.effEfficiency;
            if (!derbyOutput) {
                if (OldQuiet) {
                    if (info.isBatched())
                        printf("%d, ", b);
                    printf("%d, %d, %d, %.4g%s", m, n, k, output, csvunits(secondsOutput, memoryBound));
                    if (showEff)
                        printf(", %.4g%s", effOutput, csvunits(secondsOutput, memoryBound));
                    printf("\n");
                } else {
                    if (info.isBatched())
                        printf("%d x ", b);
                    printf("%d x %d x %d ", m, n, k);
                    if (lda >= 0 || ldb >= 0 || ldc >= 0)
                        printf("[%d, %d, %d] ", lda, ldb, ldc);
                    int precision = secondsOutput ? 6 : 1;
                    printf("%.*f%s", precision, output, units(secondsOutput, memoryBound));
                    if (showEff)
                        printf(" (effective: %.*f%s)", precision, effOutput, units(secondsOutput, memoryBound));
                    printf("\n");
                }
            }
            outputs.push_back(output);
        }
    }

    if (series)
        test(bench, false, info, TestMode::EndSeries, maxBatch, maxM, maxN, maxK, -1, -1, -1, false, false, false, 'F', false, false);

    if (!outputs.empty()) {
        std::sort(outputs.begin(), outputs.end());
        auto noutputs = outputs.size();
        auto maxOutput = outputs.back();
        auto medOutput = (noutputs & 1) ? outputs[noutputs >> 1]
                                        : 0.5 * (outputs[noutputs >> 1] + outputs[(noutputs >> 1) - 1]);
        if (perfCheckFailed)
            medOutput = maxOutput = 0.;
        if (derbyOutput)
            printf("%.3f %.3f %.3f\n", medOutput, maxOutput, checkPerf);
        else if (!OldQuiet)
            printf("%smedian%s %.2f%s %smax%s %.2f%s\n", vt(VT_RED), vt(VT_RESET), medOutput, units(secondsOutput, memoryBound), vt(VT_BLUE), vt(VT_RESET), maxOutput, units(secondsOutput, memoryBound));
    }
}

void getPerformanceLargeSize(const KernelInfo &info, int &m0, int &n0, int &k0, int &b0, int shift, bool smallprb)
{
    if (smallprb) {
        m0 = 127; n0 = 99; k0 = 135; b0 = 3;
    } else if (info.isBatched()) {
        m0 = n0 = k0 = 128 >> shift;
        b0 = 1024;
    } else {
        char precision = info.isCopy() ? info.copy[0].precisionD : info.l3[0].precisionC;
        switch (precision) {
            case 'F':
            case 'O': m0 = n0 = k0 = 12288; break;
            case 'B':
            case 'H': m0 = n0 = k0 = 8192; break;
            case 'T':
            case 'J':
            case 'K':
            case 'S': m0 = n0 = k0 = 8192; break;
            case 'I': m0 = n0 = k0 = 8192; break;
            case 'D': m0 = n0 = k0 = 6144; break;
            case 'U':
            case 'C': m0 = n0 = k0 = 6144; break;
            case 'Z': m0 = n0 = k0 = 4608; break;
            default: throw std::runtime_error("Unsupported precision");
        }
        m0 >>= shift;
        n0 >>= shift;
        k0 >>= shift;
        if (info.isL3()) {
            m0 = align_up(m0, info.mbAlign());
            n0 = align_up(n0, info.nbAlign());
            k0 = align_up(k0, info.kbAlign());
        } else {
            m0 = align_up(m0, info.copy[0].unrollX());
            k0 = align_up(k0, info.copy[0].unrollY());
        }
    }
}

int findCardNumber(int maxCards = 8)
{
    uint32_t cmask = 0;
    for (int i = 0; i < maxCards; i++) {
        std::string rp0 = "/sys/class/drm/card";
        rp0 += std::to_string(i);
        rp0 += "/gt_RP0_freq_mhz";
        std::ifstream f(rp0);
        if (f.is_open())
            cmask |= (1u << i);
    }
    if (!cmask)
        return -1;
    else if (cmask & (cmask - 1))
        return -1;
    else
        return int(floor(log2(double(cmask))));
}

void runFreqSweep(benchstone::Driver &bench, KernelInfo &info, int batchSize, int m, int n, int k, int lda, int ldb, int ldc, bool secondsOutput, bool memoryBound) {
#ifdef _WIN32
    bool root = false;
#else
    bool root = (geteuid() == 0);
#endif
    if (!bench.getEnableMetrics())
        throw std::runtime_error("Metrics information is not available");
    auto card = getenv("CARD");
    int cardNumber = card ? atoi(card) : findCardNumber();

    if (cardNumber < 0)
        throw std::runtime_error("Either none or multiple Intel GPUs found. Please set the CARD environment variable to the correct card # as listed in /sys/class/drm/...");

    std::string base = "/sys/class/drm/card" + std::to_string(cardNumber) + '/';
    std::string sudo = (root ? "" : "sudo ");
    std::string boostMHzFile = base + "gt_boost_freq_mhz";
    std::string maxMHzFile = base + "gt_max_freq_mhz";
    std::string minMHzFile = base + "gt_min_freq_mhz";

    int minMHz = utils::readValueFromFile<int>(base + "gt_RPn_freq_mhz", -1);
    int maxMHz = utils::readValueFromFile<int>(base + "gt_RP0_freq_mhz", -1);

    if (minMHz <= 0 || maxMHz <= 0)
        throw std::runtime_error("Could not read GPU min/max frequencies");

    if (!root && system("sudo echo -n"))
        throw std::runtime_error("Root access is needed to adjust frequencies");

    int oldMin   = utils::readValueFromFile<int>(minMHzFile, -1);
    int oldMax   = utils::readValueFromFile<int>(maxMHzFile, -1);
    int oldBoost = utils::readValueFromFile<int>(boostMHzFile, -1);

    if (oldMin < 0 || oldMax < 0 || oldBoost < 0)
        throw std::runtime_error("Could not read current GPU frequency settings");

    auto setMHz   = [&](int mhz, const std::string &file) {
        std::string cmd = sudo + "sh -c \"echo " + std::to_string(mhz) + " > " + file + '\"';
        int status = system(cmd.c_str());
        if(status) throw std::runtime_error("Failed to set GPU frequency");
    };
    auto setMin   = [&](int mhz) { setMHz(mhz, minMHzFile); };
    auto setMax   = [&](int mhz) { setMHz(mhz, maxMHzFile); };
    auto setBoost = [&](int mhz) { setMHz(mhz, boostMHzFile); };

    bool OldQuiet = bench.Quiet;
    bench.Quiet = true;

    if (OldQuiet) printf("Fixed MHz, Actual MHz, Effective efficiency (%%)\n");

    test(bench, false, info, TestMode::BeginSeries, batchSize, m, n, k, lda, ldb, ldc, false, false, false, 'F', false, false);

    for (int mhz = minMHz; mhz <= maxMHz; mhz += 50) {
        if (mhz == minMHz) {
            setMin(mhz);
            setMax(mhz);
            setBoost(mhz);
        } else {
            setBoost(mhz);
            setMax(mhz);
        }
        auto timeResult = test(bench, false, info, TestMode::TimeOneOfSeries, batchSize, m, n, k, lda, ldb, ldc, false, false, false, 'F', false, false);
        auto actualMHz = timeResult.avgFreq * 1e-6;
        if (OldQuiet)
            printf("%d, %.2f, %.4g%s\n", mhz, actualMHz, timeResult.effEfficiency, csvunits(secondsOutput, memoryBound));
        else
            printf("%4d MHz fixed -> %4d MHz actual  (%.1f%s)\n", mhz, int(std::round(actualMHz)), timeResult.effEfficiency, units(secondsOutput, memoryBound));
    }

    test(bench, false, info, TestMode::EndSeries, batchSize, m, n, k, lda, ldb, ldc, false, false, false, 'F', false, false);

    bench.Quiet = OldQuiet;
    setMin(oldMin);
    setMax(oldMax);
    setBoost(oldBoost);
}

template <typename F>
void errorCase(const benchstone::Driver &bench, F messageFunc)
{
    if (bench.Quiet)
        std::cerr << ' ';
    else
        std::cerr << vt(VT_BOLD VT_RED) << "Error testing case" << vt(VT_RESET) << ' ';
    std::cerr << vt(VT_BOLD VT_RED);
    messageFunc();
    std::cerr << vt(VT_RESET);
    if (!bench.Quiet)
        std::cerr << std::endl;
}

void printOutcome(const benchstone::Driver &bench, const TestResult &result, const KernelInfo &info) {
    switch (result.accuracy) {
        case TestAccuracy::Unchecked:    yprintf(bench.Quiet ? " Unchecked " : bench.Loud ? "Unchecked\n" : " Unchecked\n");             break;
        case TestAccuracy::OK:           gprintf(bench.Quiet ? " OK " : bench.Loud ? "OK\n" : " OK\n");                                  break;
        case TestAccuracy::Inaccurate:   if (bench.Quiet) rprintf(info.readCheck() ? " Read overrun " : " Inaccurate ");                 break;
        case TestAccuracy::Overrun:      if (bench.Quiet) rprintf(" Overrun ");                                                          break;
        case TestAccuracy::RuntimeError: rprintf(bench.Quiet ? " Runtime error " : bench.Loud ? "Runtime error\n" : " Runtime error\n"); break;
    }
};

void runFullAccuracy(benchstone::Driver &bench, KernelInfo &info, int batchSize, int m, int n, int k, int lda, int ldb, int ldc) {
    int kb = info.kbAlign();
    bool kRemHandling = info.kRemHandling();

    auto addrOffsets = info.getAddrOffsets();
    if (info.isCopy())
        k = info.copy[0].inner() ? n : m;
    if (!kRemHandling && (k % kb))
        yprintf(bench.Quiet ? " Misaligned k" : "Warning: k is misaligned and k remainder handling is disabled.\n");
    else {
        TestResult result;
        for (bool inplace : {false, true}) {
        for (bool addrOffset : {false, true}) {
        for (bool aOffset : {false, true}) {
        for (bool bOffset : {false, true}) {
        for (char cOffType : {'F', 'C', 'R', 'M'}) {
        for (bool unit : {false, true}) {
        for (bool oneFill : {true, false}) {
        for (bool conja : {false, true}) {
        for (bool conjb : {false, true}) {
        for (bool beta0 : {false, true}) {
        for (bool crosscheck : {false, true}) {
            bool allowCrossCheck = (bench.runtime() == benchstone::RuntimeBackend::LevelZero) && DoCrossingCheck;
            if (crosscheck && !allowCrossCheck) continue;
            if (beta0 && (info.isCopy() || info.l3[0].fixedBeta()))
                continue;
            if (!info.canConjAB(conja, conjb))
                continue;
            if ((cOffType != 'F') && !info.canCOffset())
                continue;
            if ((cOffType == 'M') && !info.canCMatrixOffset())
                continue;
            if (aOffset && (info.isCopy() || !info.l3[0].aOffset))
                continue;
            if (bOffset && (info.isCopy() || !info.l3[0].bOffset))
                continue;
            if ((inplace || unit) && !(info.isL3() && info.l3[0].trsm3()))
                continue;
            if (inplace && info.strongReadCheck())
                continue;
            if (addrOffset && (info.isCopy() || (info.l3[0].copyA() && info.l3[0].copyB())))
                continue;
            bench.AddrOffsetOverride[0] = int(addrOffset) * addrOffsets[0];
            bench.AddrOffsetOverride[1] = int(addrOffset) * addrOffsets[1];
            bench.AddrOffsetOverride[2] = int(addrOffset) * addrOffsets[2];
            if (crosscheck)
                bench.ExtraFlagsA = bench.ExtraFlagsB = bench.ExtraFlagsC = bench.ExtraFlagsS = bench.ExtraFlagsD = BUF_CROSS_4GB;
            bench.OneFill = oneFill;
            bench.TestBeta0 = beta0;
            bench.TestInPlace = inplace;
            bench.TestNonKParallelGlobal = bench.TestBeta0 && info.l3[0].kParallel() && !info.l3[0].fusedBeta();
            result = test(bench, false, info, TestMode::Accuracy, batchSize, m, n, k, lda, ldb, ldc, conja, conjb, unit, cOffType, aOffset, bOffset);
            if (crosscheck)
                bench.ExtraFlagsA = bench.ExtraFlagsB = bench.ExtraFlagsC = bench.ExtraFlagsS = bench.ExtraFlagsD = 0;
            if (result.failed()) {
                errorCase(bench, [&](){
                    fprintf(stderr, "[%s", oneFill ? "all 1s" : "random data");
                    if (addrOffset) fprintf(stderr, ", odd offsets");
                    if (conja) fprintf(stderr, ", conja");
                    if (conjb) fprintf(stderr, ", conjb");
                    if (unit) fprintf(stderr, ", unit");
                    if (aOffset) fprintf(stderr, ", A offset");
                    if (bOffset) fprintf(stderr, ", B offset");
                    if (info.canCOffset()) {
                        if (cOffType == 'F') fprintf(stderr, ", C fixed offset");
                        if (cOffType == 'C') fprintf(stderr, ", C column offsets");
                        if (cOffType == 'R') fprintf(stderr, ", C row offsets");
                        if (cOffType == 'M') fprintf(stderr, ", C matrix offset");
                    }
                    if (inplace) fprintf(stderr, ", in place");
                    if (crosscheck) fprintf(stderr, ", crossing 4GB");
                    fprintf(stderr, "]");
                });
                goto test_failure_size;
            }
        }
        }
        }
        }
        }
        }
        }
        }
        }
        }
        }

test_failure_size:
        printOutcome(bench, result, info);
    }
}

TestResult runPerformance(benchstone::Driver &bench, KernelInfo &info, int batchSize, int m, int n, int k, int lda, int ldb, int ldc) {
    return test(bench, false, info, TestMode::Time, batchSize, m, n, k, lda, ldb, ldc, false, false, false, 'F', false, false);
}

void chromeTraceTrackingData(const benchstone::Driver &bench)
{
    using TT = gemmstone::TimepointType;
    std::vector<gemmstone::Timepoint> trackingData;
    int stride, trackingWGSize;
    std::tie(trackingData, stride, trackingWGSize) = bench.getChromeTrackingData();

    int nstreams = int(trackingData.size() / stride);

    const auto *data = trackingData.data();

    std::ofstream s("gemm_trace.json");

    s << "[\n";

    bool isFirst = true;
    for (int istream = 0; istream < nstreams; istream++) {
        auto *sdata0 = data + (istream * stride);
        auto *edata = data + ((istream + 1) * stride);

        /* Check for TmEvent and discard this stream if it occurred */
        bool tmEvent = false;
        for (auto sdata = sdata0; sdata < edata && sdata->type != TT::TimepointInvalid; sdata++)
            tmEvent |= ((sdata->type & TT::TimepointTmEvent) != 0);

        if (tmEvent) continue;

        /* Otherwise output the trace */
        for (auto sdata = sdata0; sdata < edata && sdata->type !=TT:: TimepointInvalid; sdata++) {
            auto baseType = sdata->type & 0xFFFF0000;
            char ph = 'i';
            const char *name = "unknown";
            int pid = istream / trackingWGSize;
            int tid = istream % trackingWGSize;
            switch (baseType) {
                case TT::TimepointThreadStart:    ph = 'B'; name = "thread"; break;
                case TT::TimepointThreadEnd:      ph = 'E'; name = "thread"; break;
                case TT::TimepointSubthreadStart: ph = 'B'; name = "subthread"; break;
                case TT::TimepointSubthreadEnd:   ph = 'E'; name = "subthread"; break;
                case TT::TimepointAccumDone:      name = "accumulated"; break;
                default:                      break;
            }

            if (!isFirst) s << ",\n";
            s << "{\"ph\":\"" << ph << "\", \"name\":\"" << name << "\", \"pid\":\"" << pid << "\", \"tid\":\"" << tid << "\", \"ts\":" << sdata->time << "}";
            isFirst = false;
        }
    }
    s << "\n]\n";
}

void printTrackingData(const benchstone::Driver &bench)
{
    using TT = gemmstone::TimepointType;
    std::vector<gemmstone::Timepoint> trackingData;
    int stride, trackingWGSize;
    std::tie(trackingData, stride, trackingWGSize) = bench.getChromeTrackingData();
    int nstreams = int(trackingData.size() / stride);

    const auto *data = trackingData.data();

    printf("\nLast tracking data:\n");

    for (int istream = 0; istream < nstreams; istream++) {
        auto *sdata = data + (istream * stride);
        auto *edata = data + ((istream + 1) * stride);

        if (sdata[0].type == TT::TimepointInvalid) continue;    /* no data here */

        printf("Stream %d: ", istream);
        bool haveBase = false;
        uint32_t base = 0;
        for (; sdata < edata && sdata->type != TT::TimepointInvalid; sdata++) {
            auto baseType = sdata->type & 0xFFFF0000;
            switch (baseType) {
                case TT::TimepointThreadStart:    printf(" start"); break;
                case TT::TimepointThreadEnd:      printf(" end"); break;
                case TT::TimepointSubthreadStart: printf(" sstart"); break;
                case TT::TimepointSubthreadEnd:   printf(" send"); break;
                case TT::TimepointAccumDone:      printf(" accum"); break;
                default:                      printf(" ??"); break;
            }

            haveBase ? printf("[+%u]", uint32_t(sdata->time - base))
                     : printf("[0x%X]", sdata->time);

            if (sdata->type & TT::TimepointTmEvent)
                putchar('*');

            if (!haveBase && (baseType == TT::TimepointThreadStart)) {
                base = sdata->time;
                haveBase = true;
            }
        }
        putchar('\n');
    }
}

int main(int argc, char **argv) {
    const char *batchFile = nullptr;
    int32_t m = 128, n = 128, k = 128, batchSize = 1;
    int32_t lda = -1, ldb = -1, ldc = -1;
    bool specificSize = false;
    bool tileSweep = false, freqSweep = false;
    bool derbyOutput = false;
    int ntiles = 1;

    bool secondsOutput = false;
    bool memoryBound = false;

    auto runtimeBackend = benchstone::RuntimeBackend::OpenCL;
    int argw = 1;
    for (int argr = 1; argr < argc; argr++) {
        if (!strcmp(argv[argr], "--level-zero")) {
            runtimeBackend = benchstone::RuntimeBackend::LevelZero;
            continue;
        }
        argv[argw++] = argv[argr];
    }
    argc = argw;
    argv[argc] = nullptr;

    auto benchPtr = benchstone::createDriver(runtimeBackend);
    if (!benchPtr) {
        if (runtimeBackend == benchstone::RuntimeBackend::LevelZero)
            std::cerr << "Level Zero runtime was requested but is not available in this build" << std::endl;
        else
            std::cerr << "OpenCL runtime is not available in this build" << std::endl;
        return EXIT_CODE_MISC_ERROR;
    }
    auto &bench = *benchPtr;

    // Get parameters from executable file name.
    bench.ZeroCopy     = strstr(*argv, "zc");
    bench.ZeroFill     = strstr(*argv, "zero");
    bench.ColdA        = strstr(*argv, "cold"); // all on or all off
    bench.ColdB        = strstr(*argv, "cold"); // all on or all off
    bench.ColdC        = strstr(*argv, "cold"); // all on or all off
    bench.Quiet        = strstr(*argv, "quiet");
    bench.Loud         = strstr(*argv, "loud");
    bench.SmallTest    = strstr(*argv, "small");
    bench.NoBlocking   = strstr(*argv, "noblock");
    bench.GenerateAUB  = strstr(*argv, "aub");
    bench.NoBlocking  |= bench.GenerateAUB;
    bench.TimingOnly   = strstr(*argv, "nocheck");
    bench.DeviceTime   = !strstr(*argv, "wall_time");

    bool AccuracyOnly  = strstr(*argv, "accuracy");
    memoryBound  = strstr(*argv, "mbound");

    bool showKernel = true;
    auto hideKernel = getenv("HIDE_KERNEL");
    if (hideKernel && atoi(hideKernel))
        showKernel = false;

    auto progress = getenv("PROGRESS");
    if (progress) bench.Progress = atoi(progress);

    auto cold = getenv("COLD");
    if (cold) {
        bench.ColdA = bench.ColdA || strchr(cold, 'A');
        bench.ColdB = bench.ColdB || strchr(cold, 'B');
        bench.ColdC = bench.ColdC || strchr(cold, 'C');
        bench.ColdICache = bench.ColdICache || strchr(cold, 'I');
        bench.ColdTLB = bench.ColdTLB || strchr(cold, 'T');
        if (strchr(cold, '1')) bench.ColdA = bench.ColdB = bench.ColdC = true;
    }

    if ((bench.runtime() == benchstone::RuntimeBackend::OpenCL) && bench.ColdTLB)
        throw std::runtime_error("Cold TLB testing not supported with OpenCL runtime");

    auto reps = getenv("WARMUP_REPS");
    if (reps) bench.CopyWarmupReps = bench.WarmupReps = atoi(reps);
    reps = getenv("TIME_REPS");
    if (reps) bench.TimeMinReps = bench.TimeMaxReps = atoi(reps);
    reps = getenv("TIME_MIN_REPS");
    if (reps) bench.TimeMinReps = atoi(reps);
    reps = getenv("TIME_MAX_REPS");
    if (reps) bench.TimeMaxReps = atoi(reps);

    auto sdGoal = getenv("STDDEV_GOAL");
    if (sdGoal) bench.RelativeMeanStdDevGoal = atof(sdGoal);

    auto perfBail = getenv("PERF_BAIL");
    if (perfBail) bench.PerfBailThreshold = atof(perfBail);

    auto forceNBlockM = getenv("M_BLOCKS");
    if (forceNBlockM) bench.NBlockOverrides[0] = atoi(forceNBlockM);

    auto forceNBlockN = getenv("N_BLOCKS");
    if (forceNBlockN) bench.NBlockOverrides[1] = atoi(forceNBlockN);

    auto forceNBlockK = getenv("K_BLOCKS");
    if (forceNBlockK) bench.NBlockOverrides[2] = atoi(forceNBlockK);

    auto addrOffsetA = getenv("A_OFFSET_ELEMS");
    if (addrOffsetA) bench.AddrOffsetOverride[0] = atoi(addrOffsetA);

    auto addrOffsetB = getenv("B_OFFSET_ELEMS");
    if (addrOffsetB) bench.AddrOffsetOverride[1] = atoi(addrOffsetB);

    auto addrOffsetC = getenv("C_OFFSET_ELEMS");
    if (addrOffsetC) bench.AddrOffsetOverride[2] = atoi(addrOffsetC);

    auto oddLDs = getenv("ODD_LDAB");
    if (oddLDs && atoi(oddLDs)) bench.OddLD[0] = bench.OddLD[1] = true;

    bool ChromeTraceTrackingData = true;
    auto printTracking = getenv("PRINT_TRACKING");
    if (printTracking) ChromeTraceTrackingData = false;

    auto fixWGK = getenv("FIX_WG_K");
    if (fixWGK) bench.FixWGK = atoi(fixWGK);

    auto coType = getenv("CO_TYPE");
    if (coType) bench.TestCOffsetType = coType[0];

    auto skipInitEnv = getenv("SKIP_INIT");
    bool skipInit = skipInitEnv && atoi(skipInitEnv);

    // CLI-only environment variables
    auto crossing = getenv("CHECK_CROSS_4GB");
    DoCrossingCheck = crossing && atoi(crossing);

    // Check for server mode.
    argv++; argc--;

    bool ServerMode = false;
    bool oldColdA = bench.ColdA;
    if (argc && !strcmp(*argv, "server")) {
        ServerMode = true;
        bench.Quiet = true;
        bench.ColdA = true;        /* temporarily set cold cache mode to initialize cache-flush kernels in rtInit */
#ifndef _WIN32
        setenv("BENCHMARK_ID", std::to_string(getpid()).c_str(), false);
#endif
    }

    // Check for flags.
    while (argc && argv[0][0] == '-') {
        if (!strcmp(argv[0] + 1, "q")) bench.Quiet = true;
        else if (!strcmp(argv[0] + 1, "v")) bench.Loud = true;
        else if (!strcmp(argv[0] + 1, "a")) AccuracyOnly = true;
        else if (!strcmp(argv[0] + 1, "p")) bench.TimingOnly = true;
        argv++; argc--;
    }

    if (skipInit)
      for (auto *flag :
           {&bench.ExtraFlagsA, &bench.ExtraFlagsB, &bench.ExtraFlagsC,
            &bench.ExtraFlagsCO, &bench.ExtraFlagsD, &bench.ExtraFlagsP,
            &bench.ExtraFlagsS, &bench.ExtraFlagsX})
        *flag |= BUF_UNINIT;

    // Initialize runtime.
    bench.init();

    // Initialize metrics if needed
    if (!AccuracyOnly) {
        bool collectMetrics = !bench.isSimulation();
        auto enableMetrics = getenv("ENABLE_METRICS");
        if (enableMetrics) collectMetrics = atoi(enableMetrics);

#ifdef _WIN32
        bool root = false;
#else
        // Check if we have access to device metrics.
        bool root = (geteuid() == 0);
        if (collectMetrics && !root) {
            const char *filename = "/proc/sys/dev/i915/perf_stream_paranoid";
            if (bench.Hardware >= ngen::HW::Xe2)
                filename = "/proc/sys/dev/xe/observation_paranoid";
            collectMetrics = !utils::readValueFromFile<bool>(filename, true);
            if (!collectMetrics && (!bench.Quiet || ServerMode)) {
                yprintf("GPU performance counters cannot be accessed. Either run as root or execute\n"
                        "  sudo sh -c 'echo 0 > %s'\n", filename);
            }
        }
#endif
        bench.enableMetrics(collectMetrics);
    }

    if (bench.isSimulation()) {
        bench.WarmupReps = bench.TimeMinReps = bench.TimeMaxReps = 1;
        if (!bench.Loud && !ServerMode) bench.Progress = true;
        if (bench.TimingOnly && bench.Hardware == ngen::HW::Xe3p) skipInit = true;  /* work around slow H2D memcpy */
    }

    KernelInfo info;

    // Read arguments.
    if (!ServerMode) {
        auto res = frontend::Result::fromTmpFile();
        bool ok = !res.binary.empty();
        if (ok) {
            info = KernelInfo(res.info);
            ok = bench.initKernel(info, res.binary);
        }
        if (!ok)
            throw std::runtime_error("Could not load kernel");
        memoryBound |= info.isCopy();
        if (info.isCopy()) {
            if (argc && !strcmp(*argv, "block")) {
                argv++; argc--;
                bench.NBlockOverrides[2] = atoi(*argv++); argc--;
            } else if (argc && !strcmp(*argv, "noblock")) {
                argv++; argc--;
                bench.NoBlocking = true;
            }
            if (argc && !strcmp(*argv, "wg")) {
                argv++; argc--;
                bench.WGOverride[0] = atoi(*argv++); argc--;
            }
            if (argc && !strcmp(*argv, "derby")) {
                argc--; argv++; derbyOutput = true;
            }
            if ((argc == 1) && std::isalpha(**argv)) {
                argc--; batchFile = *argv++;
            } else {
                if (argc && strchr(*argv, 'x')) {
                    inputSize(*argv++, {&m, &n}); argc--; specificSize = true;
                } else {
                    if (argc) { m = atoi(*argv++); argc--; specificSize = true; }
                    if (argc) { n = atoi(*argv++); argc--; }
                }
                if (argc) { lda = atoi(*argv++); argc--; bench.SpecifiedLDs = true; }
                if (argc) { ldb = atoi(*argv++); argc--; }
            }
        } else {
            if (argc && !strcmp(*argv, "block")) {
                argv++; argc--;
                std::stringstream s;
                for (; argc && strcmp(*argv, "--"); argc--, argv++)
                    s << *argv << ' ';
                bench.SKParallelOverride = inputSKParallel(bench, s);
                bench.DoSKParallelOverride = true;
                if (argc) {
                    argv++; argc--;
                }
            } else if (argc && !strcmp(*argv, "noblock")) {
                argv++; argc--;
                bench.NoBlocking = true;
            }
            if (argc && !strcmp(*argv, "fparams")) {
                argv++; argc--;
                std::stringstream s;
                for (; argc && strcmp(*argv, "--"); argc--, argv++)
                    s << *argv << ' ';
                bench.FusedParamsOverride = inputFusedParams(s);
                bench.DoFusedParamsOverride = true;
                if (argc) {
                    argv++; argc--;
                }
            }
            if (argc && !strcmp(*argv, "wg")) {
                argv++; argc--;
                if (argc) {
                    bench.WGOverride[0] = atoi(*argv);
                    auto second = strchr(*argv, 'x');
                    if (second) {
                        bench.WGOverride[1] = atoi(second + 1);
                        auto third = strchr(second + 1, 'x');
                        if (third)
                            bench.WGOverride[2] = atoi(third + 1);
                    }
                    argv++; argc--;
                }
            }
            if (argc && !strcmp(*argv, "transpose")) {
                argv++; argc--;
                if (argc) {
                    bench.TransposeOverride[0] = (toupper((*argv)[0]) == 'T');
                    bench.TransposeOverride[1] = **argv && (toupper((*argv)[1]) == 'T');
                    bench.DoTransposeOverride = true;
                    argv++; argc--;
                }
            }
            if (argc && !strcmp(*argv, "derby")) {
                argc--; argv++; derbyOutput = true;
            }
            if (argc && !strcmp(*argv, "freqsweep")) {
                argc--; argv++; freqSweep = true;
            }
            if (argc && !strcmp(*argv, "tilesweep")) {
                tileSweep = true;
                argc--; argv++;
                if (argc) { ntiles = atoi(*argv++); argc--; }
            } else if ((argc == 1) && !std::isdigit(**argv)) {
                argc--; batchFile = *argv++;
            } else {
                if (argc && strchr(*argv, 'x')) {
                    argc--; specificSize = true;
                    if (info.isBatched())
                        inputSize(*argv++, {&batchSize, &m, &n, &k});
                    else
                        inputSize(*argv++, {&m, &n, &k});
                } else {
                    if (info.isBatched() && argc) { batchSize = atoi(*argv++); argc--; specificSize = true; }
                    if (argc) { m = atoi(*argv++); argc--; specificSize = true; }
                    if (argc) { n = atoi(*argv++); argc--; }
                    if (argc) { k = atoi(*argv++); argc--; }
                }
                if (argc) { lda = atoi(*argv++); argc--; bench.SpecifiedLDs = true; }
                if (argc) { ldb = atoi(*argv++); argc--; }
                if (argc) { ldc = atoi(*argv++); argc--; }
                for (int i = 0; i < 3; i++)
                    if (argc) { bench.NBlockOverrides[i] = atoi(*argv++); argc--; }
            }
        }

        if (specificSize && (m <= 0 || n <= 0 || k < 0))
            throw std::runtime_error("Invalid parameters.");
    }

    // Testing.
    bool batchType = (batchFile || tileSweep || freqSweep || ServerMode);
    if (!bench.Quiet)
        std::cout << "Testing kernel ";
    if (showKernel && (!batchType || !bench.Quiet))
        printKernelInfo(info);
    if (bench.Loud || ((bench.TimingOnly || batchType) && !bench.Quiet))
        std::cout << std::endl;
    else {
        if (!bench.Quiet) std::cout << "...";
        fflush(stdout);
    }

    TestResult result;
    if (ServerMode) {
        bench.ColdA = oldColdA;
        bool inBatch = false;
        bool showStddev = false;
        std::cin >> std::skipws >> std::ws;

        for (; !std::cin.eof(); std::cin >> std::ws) {
            std::string lineStr;
            std::getline(std::cin, lineStr);
            char eat;

            if (lineStr.size() == 0) continue;

            std::stringstream line(lineStr);

            if (lineStr[0] == '!') {
                std::string directive;
                line >> directive;

                if (directive == "!begin_series") {
                    if (inBatch) {
                        std::cout << "?error already in batch mode\n";
                        continue;
                    }
                    inBatch = true;
                    int b, m, n, k, lda, ldb, ldc;
                    line >> b >> m >> n >> k >> lda >> ldb >> ldc;
                    test(bench, ServerMode, info, TestMode::BeginSeries, b, m, n, k, lda, ldb, ldc, false, false, false, 'F', false, false);
                    std::cout << "?ok\n";
                } else if (directive == "!end_series") {
                    if (inBatch) {
                        inBatch = false;
                        test(bench, ServerMode, info, TestMode::EndSeries, 1, 1, 1, 1, 1, 1, 1, false, false, false, 'F', false, false);
                    }
                    std::cout << "?ok\n";
                } else if (directive == "!warmup_reps") {
                    line >> bench.WarmupReps;
                    std::cout << "?ok\n";
                } else if (directive == "!time_reps") {
                    line >> bench.TimeMinReps >> std::ws;
                    if (!line.eof() && line.peek() == '-')
                        line >> eat >> bench.TimeMaxReps;
                    else
                        bench.TimeMaxReps = bench.TimeMinReps;
                    std::cout << "?ok\n";
                } else if (directive == "!stddev_goal") {
                    line >> bench.RelativeMeanStdDevGoal >> std::ws;
                    std::cout << "?ok\n";
                } else if (directive == "!perf_bail") {
                    line >> bench.PerfBailThreshold >> std::ws;
                    std::cout << "?ok\n";
                } else if (directive == "!membound") {
                    memoryBound = true;
                    line >> std::ws;
                    if (!line.eof())
                        line >> memoryBound;
                    std::cout << "?ok\n";
                } else if (directive == "!seconds") {
                    secondsOutput = true;
                    std::cout << "?ok\n";
                } else if (directive == "!cold") {
                    std::string cold_options;
                    line >> cold_options;
                    if (cold_options.find("A") != std::string::npos) bench.ColdA = true;
                    if (cold_options.find("B") != std::string::npos) bench.ColdB = true;
                    if (cold_options.find("C") != std::string::npos) bench.ColdC = true;
                    std::cout << "?ok\n";
                } else if (directive == "!echo") {
                    std::string word;
                    line >> word;
                    std::cout << word << std::endl;
                } else if (directive == "!bye")
                    break;
                else if (directive == "!kernel") {
                    bench.finishKernel();
                    std::string kernel = lineStr.substr(8);
                    auto job = getJob(kernel);
                    frontend::Result res;
                    if (job.has_value()) {
                        res = job.value()->finish();
                        job.value()->kernel.clear(); // Mark for reuse
                    } else {
                        std::unique_ptr<frontend::Generator> generator = frontend::Generator::Create();
                        res = generator->generate(kernel);
                    }
                    if (res.success) {
                        info = KernelInfo(res.info);
                        bool ok = bench.initKernel(info, res.binary);
                        if (ok) {
                            std::cout << "?ok\n";
                            memoryBound |= info.isCopy();
                        } else {
                            std::cout << "?error kernel could not be generated (error during benchstone init)\n";
                        }
                    } else {
                        std::cout << "?error kernel could not be generated (" + res.info + ")\n";
                    }
                } else if (directive == "!kernel_preload") {
                    std::string kernel = lineStr.substr(16);
                    startJob(kernel);
                    std::cout << "?ok\n";
                } else if (directive == "!accuracy") {
                    if (inBatch) {
                        inBatch = false;
                        test(bench, ServerMode, info, TestMode::EndSeries, 1, 1, 1, 1, 1, 1, 1, false, false, false, 'F', false, false);
                    }
                    auto result = accuracyTests(bench, info, ServerMode);
                    switch (result.accuracy) {
                        case TestAccuracy::Unchecked:
                        case TestAccuracy::OK:           std::cout << "?ok\n";            break;
                        case TestAccuracy::Inaccurate:   std::cout << "?inaccurate\n";    break;
                        case TestAccuracy::Overrun:      std::cout << "?overrun\n";       break;
                        case TestAccuracy::RuntimeError: std::cout << "?runtime error\n"; break;
                    }
                } else if (directive == "!query_eu_count")
                    std::cout << "?result " << bench.info.EUs << std::endl;
                else if (directive == "!query_hw_threads") {
                    if (!bench.isKernelLoaded())
                        std::cout << "?error no kernel is loaded\n";
                    int threads = info.isCopy() ? bench.hwThreads(info.copy[0]) : bench.hwThreads(info.l3[0]);
                    std::cout << "?result " << threads << std::endl;
                } else if (directive == "!query_wg") {
                    if (!bench.isKernelLoaded())
                        std::cout << "?error no kernel is loaded\n";
                    auto &kinfo = info.isCopy() ? static_cast<gemmstone::CommonDriverInfo&>(info.copy[0])
                                                : static_cast<gemmstone::CommonDriverInfo&>(info.l3[0]);
                    std::cout << "?result " << kinfo.wg[0] << ',' << kinfo.wg[1] << ',' << kinfo.wg[2] << std::endl;
                } else if (directive == "!show_stddev") {
                    showStddev = true;
                    std::cout << "?ok\n";
                } else
                    std::cout << "?error unknown directive received\n";
            } else if (!bench.isKernelLoaded())
                std::cout << "?error no kernel is loaded\n";
            else {
                int b = 0, m = 1, n = 1, k = 1, lda = -1, ldb = -1, ldc = -1;
                line >> b >> m >> n >> k >> lda >> ldb >> ldc;
                auto mode = inBatch ? TestMode::TimeOneOfSeries : TestMode::Time;
                result = test(bench, ServerMode, info, mode, b, m, n, k, lda, ldb, ldc, false, false, false, 'F', false, false);
                if (result.accuracy == TestAccuracy::RuntimeError)
                    std::cout << "?error runtime error\n";
                else {
                    double output = secondsOutput ? result.time
                                    : memoryBound ? result.bw * 1e-9
                                                  : result.efficiency;
                    double effOutput = secondsOutput ? result.effTime
                                       : memoryBound ? result.bw * 1e-9
                                                     : result.effEfficiency;
                    printf("%d %d %d %d %d %d %d %.6g %.6g ", b, m, n, k, lda, ldb, ldc, output, effOutput);
                    if (showStddev)
                        printf("%.6g ", result.meanStddevBestAvailable());
                    printf("%s\n", csvunits(secondsOutput, memoryBound));
                }
            }
            fflush(stdout);
        }
    } else if (batchFile) {
        runBatch(bench, batchFile, info, derbyOutput, secondsOutput);
    } else if (freqSweep) {
        if (!specificSize) getPerformanceLargeSize(info, m, n, k, batchSize, 0, bench.SmallTest);
        runFreqSweep(bench, info, batchSize, m, n, k, lda, ldb, ldc, secondsOutput, memoryBound);
    } else if (specificSize) {
        if (!bench.GenerateAUB && !bench.TimingOnly) {
            runFullAccuracy(bench, info, batchSize, m, n, k, lda, ldb, ldc);
        }

        if (!AccuracyOnly) {
            bench.OddLD[0] = bench.OddLD[1] = bench.OddLD[2] = false;
            bench.AddrOffsetOverride[0] = bench.AddrOffsetOverride[1] = bench.AddrOffsetOverride[2] = 0;
            bench.OneFill = bench.TestBeta0 = bench.TestInPlace = bench.TestNonKParallelGlobal = false;

            auto timeResult = runPerformance(bench, info, batchSize, m, n, k, lda, ldb, ldc);

            double output = secondsOutput ? timeResult.time
                            : memoryBound ? timeResult.bw
                                          : timeResult.efficiency;
            int precision = secondsOutput ? 6 : 1;
            if (bench.Quiet && !bench.GenerateAUB) printf(" %.*f%s", precision, output, units(secondsOutput, memoryBound));
        }
        if (bench.Quiet) printf("\n");
    } else if (tileSweep) {
        if (!info.isL3()) gemmstone::stub();
        auto &l3info = info.l3[0];
        int wgCount = bench.hwThreads(l3info) / (l3info.wg[LoopM] * l3info.wg[LoopN] * l3info.wg[LoopK] * l3info.wgExpand);
        wgCount *= ntiles;
        bench.WGLimit = wgCount;
        bench.NoBlocking = true;
        bool OldQuiet = bench.Quiet;
        bench.Quiet = true;
        bool showEff = !secondsOutput && !memoryBound && bench.getEnableMetrics();
        integer blockK;
        std::tie(std::ignore, std::ignore, blockK) = bench.getBlocking(1024, 1024, 2048, l3info);
        k = int32_t(blockK);
        for (int wgM = wgCount; wgM >= 1; wgM--) {
            int wgN = div_up(wgCount, wgM);
            wgM = div_up(wgCount, wgN);
            if ((wgM * wgN != wgCount) && !l3info.isLinearOrder())
                continue;
            m = wgM * l3info.wgTile(LoopM);
            n = wgN * l3info.wgTile(LoopN);
            auto result = runPerformance(bench, info, batchSize, m, n, k, lda, ldb, ldc);
            double output = secondsOutput ? result.time
                            : memoryBound ? result.bw * 1e-9
                                          : result.efficiency;
            double effOutput = secondsOutput ? result.effTime
                               : memoryBound ? result.bw * 1e-9
                                             : result.effEfficiency;
            int precision = secondsOutput ? 6 : 1;
            if (OldQuiet) {
                printf("%d, %d, %d, %d, %d, %.*f%s", wgM, wgN, m, n, k, precision, output, csvunits(secondsOutput, memoryBound));
                if (showEff)
                    printf(", %.*f%s", precision, effOutput, csvunits(secondsOutput, memoryBound));
            } else {
                printf("(%d x %d WGs) \t%d x %d x %d \t%.*f%s", wgM, wgN, m, n, k, precision, output, units(secondsOutput, memoryBound));
                if (showEff)
                    printf(" (effective: %.*f%s)", precision, effOutput, csvunits(secondsOutput, memoryBound));
            }
            printf("\n");
        }
    } else {
        if (!bench.TimingOnly) {
            auto result = accuracyTests(bench, info, ServerMode);
            if (result.failed())
                errorCase(bench, [&]{ fprintf(stderr, "[%s]", result.failingCase.c_str()); });
            printOutcome(bench, result, info);
        }

        if (!AccuracyOnly) {
            int m0, n0, k0, b0;
            getPerformanceLargeSize(info, m0, n0, k0, b0, 0, bench.SmallTest);

            if (bench.Quiet) {
                constexpr int steps = 8;

                if (bench.TimingOnly) printf(" ");

                test(bench, ServerMode, info, TestMode::BeginSeries, b0, m0, n0, k0, -1, -1, -1, false, false, false, 'F', false, false);
                for (int i = 1; i <= steps; i++) {
                    int bi = b0;
                    int mi = (i * m0) / steps;
                    int ki = (i * n0) / steps;
                    int ni = (i * k0) / steps;
                    auto timeResult = runPerformance(bench, info, bi, mi, ni, ki, -1, -1, -1);
                    double output = secondsOutput ? timeResult.time
                                    : memoryBound ? timeResult.bw
                                                  : timeResult.efficiency;
                    int precision = secondsOutput ? 6 : 1;
                    printf("%.*f%s ", precision, output, units(secondsOutput, memoryBound));
                    fflush(stdout);
                }
                test(bench, ServerMode, info, TestMode::EndSeries, b0, m0, n0, k0, -1, -1, -1, false, false, false, 'F', false, false);
            } else {
                int m1, n1, k1, b1;
                getPerformanceLargeSize(info, m1, n1, k1, b1, 2, bench.SmallTest);
                runPerformance(bench, info, b1, m1, n1, k1, -1, -1, -1);
                runPerformance(bench, info, b0, m0, n0, k0, -1, -1, -1);
            }
        }

        if (bench.Quiet) printf("\n");
    }

    bench.finishKernel();

    if (ChromeTraceTrackingData)
        chromeTraceTrackingData(bench);
    else
        printTrackingData(bench);

    bench.finish();

    switch (result.accuracy) {
        case TestAccuracy::Unchecked:
        case TestAccuracy::OK:           return EXIT_SUCCESS;
        case TestAccuracy::Inaccurate:   return EXIT_CODE_INACCURATE;
        case TestAccuracy::Overrun:      return EXIT_CODE_OVERRUN;
        case TestAccuracy::RuntimeError: return EXIT_CODE_MISC_ERROR;
    }
    return EXIT_SUCCESS;
}
