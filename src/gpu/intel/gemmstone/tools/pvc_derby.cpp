/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <algorithm>
#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "derby_utils.hpp"
#include "utils.hpp"

#include "unistd.h"
#include "sys/wait.h"
#include <fcntl.h>

#include <random>

enum class TestStatus {
    OK,
    Stub,
    HWUnsupported,
    OutOfRegisters,
    GenerationError,
    Overrun,
    Inaccurate,
    SystemError,
    BenchmarkError,
};

using HW = ngen::HW;

struct Group {
    uint16_t um, un;
    uint8_t wgM, wgN, wgK, wx;
    uint8_t ikr : 1;
    uint8_t copyA : 1;
    uint8_t copyB : 1;
    uint8_t kb : 1;
    uint8_t forceSN : 1;
    uint8_t forceFN : 1;
    uint8_t flagged : 1;
    uint8_t baseKLoad;
    char aAccess, bAccess, cAccess;
    uint16_t grfs;
    uint16_t id;
    uint32_t featureMask;
    float dscale = 0.0f;
};

struct Features {
    uint32_t kaLoadMult : 2;
    uint32_t kbLoadMult : 2;
    uint32_t aCopies : 1;
    uint32_t bCopies : 1;
    uint32_t aPFMult : 2;
    uint32_t bPFMult : 2;
    uint32_t pfDist : 3;
    uint32_t ukSLMMult : 2;
    uint32_t slmCopies : 1;
    uint32_t splitC : 1;
    uint32_t slm3x : 1;
    uint32_t macroType : 2;
    uint32_t rr : 1;
    uint32_t sm : 1;
    uint32_t sn : 1;
    uint32_t nb : 1;
    uint32_t pf1D : 1;
    uint32_t fn : 1;
    uint32_t : 6;
};

static_assert(sizeof(Features) == sizeof(uint32_t), "Features bitfield does not have expected size.");

enum : uint32_t {
    FeatureKALoadMult   =  0b11u << 0,
    FeatureKBLoadMult   =  0b11u << 2,
    FeatureACopies      =     1u << 4,
    FeatureBCopies      =     1u << 5,
    FeatureAPFMult      =  0b11u << 6,
    FeatureBPFMult      =  0b11u << 8,
    FeaturePFDist       = 0b111u << 10,
    FeatureUKSLMMult    =  0b11u << 13,
    FeatureSLMCopies    =     1u << 15,
    FeatureSplitC       =     1u << 16,
    FeatureSLM3x        =     1u << 17,
    FeatureMacroType    =  0b11u << 18,
    FeatureRR           =     1u << 20,
    FeatureSM           =     1u << 21,
    FeatureSN           =     1u << 22,
    FeatureNB           =     1u << 23,
    Feature1DPF         =     1u << 24,
    FeatureFN           =     1u << 25,
};

struct StrategyID {
    uint32_t groupID, fvector;
};

struct GroupList {
    std::vector<Group> groups;
    bool sparse = false;

    int randomGroupID() const;
    Group &lookup(int groupID);
    const Group &lookup(int groupID) const;
    void filter();
};

struct Stem {
    std::string all;
    bool copy;
    std::string type;
    std::string precisions;
    std::string layouts;
    std::string alpha = "1", beta = "0";
    std::string extras;

    // Derived information:
    int ssUnrollM, ssUnrollN;
    std::string basicLayouts;
    std::string internalPrecisions;
    std::string externalPrecisions;
    int alignments[3];
    int components[3] = {1, 1, 1};
    int internalBytes[3], externalBytes[3];
    bool sys, dp4a, dot;
};

struct StrategySet {
    std::string spec;
    int unrollM, unrollN;
    StrategyID id;
    int streamID;
};

class Stream {
public:
    StrategyID baseSID;
    float baseGoodness = -1.0f;
    StrategySet candidate;
    int id;

    const Stem &stem;
    const GroupList &list;

    Stream(int id_, const Stem &stem_, const GroupList &list_);

    void reset(const StrategyID &sid, float goodness = -1.0f);

    void candidateSucceeded(float candidateGoodness);
    void candidateFailed();

    StrategySet getCandidate()   const { return candidate; }

protected:
    void mutate();
};

struct TestResults {
    TestStatus status;
    double effMedian = 0., effMax = 0., effMean = 0.;
    double checkPerf = 0.;
    int count = 1;

    double score() const;
    void merge(const TestResults &other);
};

struct BenchmarkData {
    StrategySet sset;
    TestResults results;
};

char ModelChar = 'a';
char CAccess = 'B';
int MaxWGK = 256;
int MaxWGPerSS = 4;
bool NotThin = false;
bool MemoryBound = false;
bool ColdA = false;
bool ColdB = false;
bool ColdC = false;
std::vector<std::string> InputCases;
bool ShowInputs = false;
int MaxB = 1, MaxM = 0, MaxN = 0, MaxK = 0;
int KernelCacheSize = 8;
int TopPicks = 20;
int TopPicksPerGroup = 10;
bool StartNewSeries = false;
bool MeanScore = false;
bool BenchmarkVerbose = false;
bool AlignedSizes = true;

volatile bool gotSIGUSR1 = false;
void sigusr1(int) {
    gotSIGUSR1 = true;
}

volatile bool gotSIGPIPE = false;
void sigpipe(int) {
    gotSIGPIPE = true;
}

template <typename T>
T pop(std::deque<T> &d)
{
    auto result = d.front();
    d.pop_front();
    return result;
}

[[noreturn]] void usage(const char *pname)
{
    printf("Usage: %s <type> <precisions> <layouts> <SS m unroll> <SS n unroll> [<alpha> <beta>]\n", pname);
    printf("   e.g. gemm SSS ABN 64 64 1 1\n");
    exit(1);
}

int typeBytes(char c)
{
    switch (std::toupper(c)) {
        case 'F':                   /* for now */
        case 'O':
        case 'Q': return 1;
        case 'H':
        case 'B':
        case 'W': return 2;
        case 'I':
        case 'J':
        case 'K':
        case 'T':
        case 'S': return 4;
        case 'C':
        case 'U':
        case 'D': return 8;
        case 'Z': return 16;
        default: fail("Unknown data type specified.");
    }
}

std::string extractPrecisions(const std::string &in, bool external, int *components = nullptr)
{
    std::string out;

    for (auto c = in.begin(); c < in.end(); c++) {
        if (*c == '[') {
            c++;
            out += ((out.length() >= 2) ^ external) ? *c++ : *++c;
            c++;
            if (*c == 'x') {
                c++;
                if (components)
                    components[out.length() - 1] = (*c - '0');
                c++;
            }
        } else
            out += *c;
    }

    return out;
}

std::string extractBasicLayouts(const std::string &in)
{
    std::string out;

    for (auto c: in) {
        if (std::isalpha(c))
            out += c;
    }

    return out;
}

void extractAlignments(const std::string &in, int (&alignments)[3])
{
    std::stringstream ss(in);
    int i = 0;

    while (!ss.eof() && i < 3) {
        char c;
        int v;
        ss >> c;
        if (ss.peek() == '#')
            ss >> c >> v;
        if (ss.peek() == ',')
            ss >> c >> v;
        if (ss.peek() == '@')
            ss >> c >> alignments[i];
        else
            alignments[i] = 0;
        i++;
    }
}

pid_t BenchmarkPID = 0;
FILE *BenchmarkIn = nullptr, *BenchmarkOut = nullptr;

template <typename ... Args>
void sendToBenchmark(const std::string &fmt, Args ... args) {
    if (BenchmarkIn == nullptr) {
        fail("Writing to nullptr benchstone input");
    }

    // format the string
    int size_i = std::snprintf( nullptr, 0, fmt.c_str(), args ... ) + 1;
    if( size_i <= 0 ) fail("Error during formatting benchstone input");
    auto size = static_cast<size_t>( size_i );
    std::string msg(size, '\0');
    std::snprintf( msg.data(), size, fmt.c_str(), args ... );

    // Print
    if (BenchmarkVerbose) {
        printf("%s", msg.c_str());
    }
    fprintf(BenchmarkIn, "%s", msg.c_str());
    fflush(BenchmarkIn);
}

void checkBenchmark()
{
    int wstatus = 0;
    if (!gotSIGPIPE) {
        if (waitpid(BenchmarkPID, &wstatus, WNOHANG) <= 0)
            return;
        if (WIFSIGNALED(wstatus) && WTERMSIG(wstatus) == SIGINT)
            raise(SIGINT);
    }
    fail("Benchmark unexpectedly quit");
}

const char *readBenchmarkLine()
{
    static size_t bytes = 0;
    static char *line = nullptr;
    auto read = getline(&line, &bytes, BenchmarkOut);

    if(read <= 0) return "";
    line[read - 1] = '\0';
    if (BenchmarkVerbose) {
        printf(">> %s\n", line);
    }
    return line;
}

bool isBenchmarkOK()
{
    checkBenchmark();
    std::string result = readBenchmarkLine();
    return result == "?ok";
}

void expectBenchmarkOK()
{
    if (!isBenchmarkOK())
        fail("Benchmark directive failed");
}

void startBenchmark()
{
    std::array<int, 2> pipeIn, pipeOut;
    const int PipeRead = 0, PipeWrite = 1;

    if (pipe(&pipeIn[0]) < 0 || pipe(&pipeOut[0]) < 0)
        fail("Could not create pipe");

    BenchmarkPID = fork();
    if (BenchmarkPID == 0) {
        if (dup2(pipeIn[PipeRead], STDIN_FILENO) == -1)
            fail("Could not duplicate fd");
        if (dup2(pipeOut[PipeWrite], STDOUT_FILENO) == -1)
            fail("Could not duplicate fd");

        for (auto &fd: pipeIn)
            close(fd);
        for (auto &fd: pipeOut)
            close(fd);

        (void) execl("./benchstone", "benchstone", "server", nullptr);
        fail("Could not launch child process");
    } else if (BenchmarkPID < 0)
        fail("Could not fork");

    BenchmarkIn  = fdopen(pipeIn[PipeWrite], "w");
    BenchmarkOut = fdopen(pipeOut[PipeRead], "r");
    close(pipeIn[PipeRead]);
    close(pipeOut[PipeWrite]);

    sendToBenchmark("!echo ?ok\n");
    if (!isBenchmarkOK())
        fail("Could not start benchstone");

    sendToBenchmark("!membound %d\n", int(MemoryBound));
    std::string cold_str;
    if (ColdA) cold_str.append("A");
    if (ColdB) cold_str.append("B");
    if (ColdC) cold_str.append("C");
    sendToBenchmark("!cold %s\n", cold_str.c_str());
    expectBenchmarkOK();
    expectBenchmarkOK();
}

void preloadKernel(const std::string &desc)
{
    sendToBenchmark("!kernel_preload %s\n", desc.c_str());
    expectBenchmarkOK();
}

TestStatus prepKernel(const std::string &desc)
{
    sendToBenchmark("!kernel %s\n", desc.c_str());
    bool ok = isBenchmarkOK();

    if (ok && StartNewSeries && MaxB > 0 && MaxM > 0 && MaxN > 0 && MaxK > 0) {
        StartNewSeries = false;
        sendToBenchmark("!end_series\n");
        sendToBenchmark("!begin_series %d %d %d %d -2 -2 -2\n", MaxB, MaxM, MaxN, MaxK);
        expectBenchmarkOK();
        expectBenchmarkOK();
    }

    return ok ? TestStatus::OK : TestStatus::GenerationError;    // TODO improve
}

BenchmarkData runBenchmark(const StrategySet &sset, bool verify = true)
{
    BenchmarkData bdata;
    bdata.sset = sset;
    bdata.results.status = prepKernel(sset.spec);
    if (bdata.results.status != TestStatus::OK) return bdata;

    if (verify) {
        sendToBenchmark("!end_series\n!accuracy\n");
        expectBenchmarkOK();
        if (!isBenchmarkOK())
            bdata.results.status = TestStatus::Inaccurate;
    }

    std::stringstream ss;
    for (auto &input: InputCases) {
        ss << input << std::endl;
    }
    std::string benchin = ss.str();
    sendToBenchmark("%s", benchin.c_str());

    std::vector<double> results;
    for ([[maybe_unused]] auto &input: InputCases) {
        double nominal = 0., effective = 0.;

        auto line = readBenchmarkLine();
        if (line[0] == '?')
            bdata.results.status = TestStatus::BenchmarkError;
        else if (sscanf(line, "%*d %*d %*d %*d %*d %*d %*d %lf %lf %*s\n", &nominal, &effective) != 2)
            fail("Unexpected output from benchstone");

        results.push_back(nominal);
    }

    bdata.results.effMedian = bdata.results.effMax = bdata.results.effMean = 0.;

    std::sort(results.begin(), results.end());
    int n = results.size();
    if (n > 0) {
        bdata.results.effMedian = (n & 1) ? results[n>>1] : 0.5*(results[n>>1] + results[(n>>1) - 1]);
        bdata.results.effMax = results.back();
        for (auto &r: results)
            bdata.results.effMean += r;
        bdata.results.effMean /= n;
    }

    return bdata;
}

void prepareInput(const Stem &stem, int stage)
{
    MaxB = 1; MaxM = 1; MaxN = 1; MaxK = 1;
    StartNewSeries = true;
    InputCases.clear();

    auto inputFileOverride = getenv("INPUT_FILE");
    if (inputFileOverride) {
        std::string line;
        std::ifstream inputFile(inputFileOverride);
        while (getline(inputFile, line)) {
            int b = 0, m = 0, n = 0, k = 0;
            std::stringstream ss(line);
            ss >> b >> m >> n >> k;
            if (b <= 0 || m <= 0 || n <= 0 || k < 0 || !ss)
                fail("Invalid syntax in input file");
            MaxB = std::max(MaxB, b);
            MaxM = std::max(MaxM, m);
            MaxN = std::max(MaxN, n);
            MaxK = std::max(MaxK, k);
            InputCases.push_back(std::move(line));
        }
        if (InputCases.empty())
            fail("Empty input file");
        MeanScore = true;
        AlignedSizes = false;

        if (ShowInputs) for (auto &input: InputCases)
            std::cerr << vt(VT_CYAN) << input << vt(VT_RESET) << "\n";
        return;
    }

    int ssUnrollM = stem.ssUnrollM;
    int ssUnrollN = stem.ssUnrollN;

    if (ssUnrollM == 0 && ssUnrollN == 0)
        fail("An input file must be provided if no subslice tile specified.");

    int sss = ssCount();
    std::vector<std::tuple<int, int, int>> expands;

    // Generate some full-tile tile sizes. If SS tile is thin, assume that we want
    //  the full-tile size to be thin as well, unless specified.
    int ratio = MemoryBound ? 1 :
                    NotThin ? 4096 : 4;
    if (ssUnrollM == 0) {
        expands.push_back({0, 1, 1});
        expands.push_back({0, 2, 1});
        expands.push_back({0, 3, 1});
    } else if (ssUnrollN == 0) {
        expands.push_back({0, 1, 1});
        expands.push_back({0, 1, 2});
        expands.push_back({0, 1, 3});
    } else if (ssUnrollM > ratio * ssUnrollN) {
        expands.push_back({0, sss, 1});
        if (sss % 2 == 0)
            expands.push_back({0, sss / 2, 2});
    } else if (ssUnrollN > ratio * ssUnrollM) {
        expands.push_back({0, 1, sss});
        if (sss % 2 == 0)
            expands.push_back({0, 2, sss / 2});
    } else {
        for (int nm = 1; nm <= sss; nm++) {
            if (sss % nm) continue;
            int nn = sss / nm;
            expands.push_back({std::abs(nm * ssUnrollM - nn * ssUnrollN), nm, nn});
        }

        std::sort(expands.begin(), expands.end());
    }

    int nexpand = (stage > 1) ? 1 : std::min<int>(expands.size(), 3);
    int mbase = (ssUnrollM == 0) ? 4096 : ssUnrollM;
    int nbase = (ssUnrollN == 0) ? 4096 : ssUnrollN;
    int kbase = std::min(1024, 4096 / typeBytes(stem.internalPrecisions[2]));
    std::vector<int> ks;

    int minSize = 0;
    if (getenv("MIN_SIZE")) minSize = atoi(getenv("MIN_SIZE"));

    ks.push_back(kbase);
    if (stage > 1)
        ks.push_back(kbase * 2);
    ks.push_back(kbase * 4);

    for (int iexpand = 0; iexpand < nexpand; iexpand++) {
        for (auto k: ks) {
            int m = mbase * std::get<1>(expands[iexpand]);
            int n = nbase * std::get<2>(expands[iexpand]);
            if (std::max(m, n) * k < minSize) {
                k = minSize / std::max(m, n);
                k = ((k + 31) / 32) * 32;
            }
            std::string input = "1 ";
            input += std::to_string(m);
            input += ' ';
            input += std::to_string(n);
            input += ' ';
            input += std::to_string(k);
            MaxM = std::max(MaxM, m);
            MaxN = std::max(MaxN, n);
            MaxK = std::max(MaxK, k);
            InputCases.push_back(input);

            if (stage > 2) {
                // Add bad LD cases.
                input += " -2 -2 -2";
                InputCases.push_back(input);
            }
        }
    }

    const char *reps = nullptr;
    const char *stddev = nullptr;

    switch (stage) {
        case 1: reps = "20-50";  stddev = "0.01";  break;
        case 2: reps = "50-100"; stddev = "0.005"; break;
        case 3: reps = "50-100"; stddev = "0.002"; break;
        case 4: reps = "50-100"; stddev = "0.001"; break;
    }

    if (reps) {
        sendToBenchmark("!time_reps %s\n!stddev_goal %s\n", reps, stddev);
        expectBenchmarkOK();
        expectBenchmarkOK();
    }

    if (ShowInputs) for (auto &input: InputCases)
        std::cerr << vt(VT_CYAN) << input << vt(VT_RESET) << "\n";
}

double TestResults::score() const
{
    return MeanScore ? effMean
                     : (effMedian + effMax);
}

void TestResults::merge(const TestResults &other)
{
    int ncount = count + other.count;

#define WEIGHTED_AVERAGE(x) x = (x * count + other.x * other.count) / ncount;

    WEIGHTED_AVERAGE(effMedian)
    WEIGHTED_AVERAGE(effMean)
    WEIGHTED_AVERAGE(effMax)

    status = static_cast<TestStatus>(std::max(static_cast<int>(status),
                                              static_cast<int>(other.status)));

#undef WEIGHTED_AVERAGE

    count = ncount;
}

float distance(const Group &group, uint32_t fvector1, uint32_t fvector2)
{
    Features f1, f2;
    std::memcpy(&f1, &fvector1, sizeof(Features));
    std::memcpy(&f2, &fvector2, sizeof(Features));

    float d = 0;

#define WEIGHT(f, w) d += w * std::abs(int(f1.f) - int(f2.f))

    WEIGHT(kaLoadMult, 1);
    WEIGHT(kbLoadMult, 1);
    WEIGHT(aCopies, 1);
    WEIGHT(bCopies, 1);
    WEIGHT(aPFMult, 0.25f);
    WEIGHT(bPFMult, 0.25f);
    WEIGHT(pfDist, 0.125f);
    WEIGHT(ukSLMMult, 1);
    WEIGHT(slmCopies, 0.25);
    WEIGHT(splitC, 2);
    WEIGHT(slm3x, 0.25);
    WEIGHT(macroType, 0.125);
    WEIGHT(rr, 0.25);
    WEIGHT(sm, 2);
    WEIGHT(sn, 2);
    WEIGHT(nb, 0.125);
    WEIGHT(pf1D, 0.5);
    WEIGHT(fn, 0.25);

#undef WEIGHT

    if (group.dscale > 0)
        return std::clamp(d * group.dscale, 0.0f, 1.0f);
    else
        return d;
}

GroupList generateGroups(const Stem &stem)
{
    GroupList list;

    if (stem.copy)
        fail("Copy auto-optimization not implemented yet.");

    int ssUM = stem.ssUnrollM;
    int ssUN = stem.ssUnrollN;
    auto hw = currentHW();
    bool fusedEUs = (hw >= HW::Gen12LP && hw <= HW::XeHPG);

    char aLayout = stem.basicLayouts[0];
    char bLayout = stem.basicLayouts[1];
    char cLayout = stem.basicLayouts[2];
    int aBytes = stem.internalBytes[0];
    int bBytes = stem.internalBytes[1];
    int cBytes = stem.internalBytes[2];
    int aBytesExt = stem.externalBytes[0];
    int bBytesExt = stem.externalBytes[1];
    int aComponents = stem.components[0];
    int bComponents = stem.components[1];
    int cComponents = stem.components[2];
    bool aComplex = isComplex(stem.internalPrecisions[0]);
    bool bComplex = isComplex(stem.internalPrecisions[1]);
    bool cComplex = isComplex(stem.internalPrecisions[2]);
    bool aSubByte = isSubByte(stem.externalPrecisions[0]);

    int aBytesReal = aBytes, aBytesExtReal = aBytesExt;
    int bBytesReal = bBytes, bBytesExtReal = bBytesExt;
    int cBytesReal = cBytes;

    if (aComplex) aBytesReal >>= 1, aBytesExtReal >>= 1;
    if (bComplex) bBytesReal >>= 1, bBytesExtReal >>= 1;
    if (cComplex) cBytesReal >>= 1;

    bool sys = stem.sys;
    bool dot = stem.dot;
    bool dp4a = stem.dp4a;

    bool forceCA = getenv("FORCE_COPY_A");
    bool forceCB = getenv("FORCE_COPY_B");
    bool allowCA = getenv("TRY_SLM");
    bool allowCB = allowCA;
    bool canBlock2DA   = (hw >= HW::XeHPC) && (stem.alignments[0] >= 8) && !dot;
    bool canBlock2DB   = (hw >= HW::XeHPC) && (stem.alignments[1] >= 8) && !dot;
    bool canBlock2DTB  = (hw >= HW::XeHPC) && (stem.alignments[1] >= 4) && !dot;
    bool canBlock2DPFA = (hw >= HW::XeHPC) && (stem.alignments[0] >= 4);
    bool canBlock2DPFB = (hw >= HW::XeHPC) && (stem.alignments[1] >= 4);
    bool allowSmallUM = dot || getenv("ALLOW_SMALL_UM");
    bool allowKB = (std::stof(stem.beta) == 1.);

    forceCB |= (sys && bLayout == 'T' && !canBlock2DTB);
    allowCA |= forceCA;
    allowCB |= forceCB;

    int mCP = 1;
    if (sys || dp4a) mCP = std::max(mCP, 4 / aBytes);

    int maxUM = ssUM ? ssUM : 128;
    int maxUN = ssUN ? ssUN : 128;

    std::vector<int> ums, uns;
    for (int um = maxUM; um > 0; um >>= 1) {
        if ((um * aBytes * mCP) % grfBytes() && um < maxUM && !allowSmallUM)
            break;
        ums.push_back(um);
        if (um & 1)
            break;
    }

    for (int un = maxUN; un >= maxUN / 16; un >>= 1) {
        uns.push_back(un);
        if (un & 1)
            break;
    }

    const struct {
        const std::vector<int>& operator()() const {
          if (MemoryBound) return membound;
          return other;
        };

        const std::vector<int> membound = {1, 2, 4};
        const std::vector<int> other = {1};
    } wx_range;

    for (bool largeGRF : {false, true}) {
    for (int um : ums) {
    for (int un : uns) {
    for (int wgM : {1, 2, 4, 8, 16}) {
    for (int wgN : {1, 2, 4, 8, 16}) {
    for (int wgK : {1, 2, 4, 8, 16, 32}) {
    for (int wx: wx_range()) {
    for (bool ikr : {false, true}) {
    for (bool kb : {false, true}) {
    for (bool copyA : {false, true}) {
    for (bool copyB : {false, true}) {
        int grfs = largeGRF ? 256 : 128;
        if (hw >= HW::Xe3p)
            grfs = largeGRF ? 512 : 192;
        if (grfs > 128 && hw < HW::XeHP) continue;
        if (kb && !allowKB) continue;
        if (copyA && !allowCA) continue;
        if (copyB && !allowCB) continue;
        if (!copyA && forceCA && um > 1) continue;
        if (!copyB && forceCB && un > 1) continue;
        if (copyA && !forceCA && wgN == 1) continue;
        if (copyB && !forceCB && wgM == 1) continue;

        if (wgK <= 0 || wgK > MaxWGK) continue;
        if (wgM * wgN * wgK * wx > maxWG()) continue;
        if (wgK == 1 && ikr) continue;
        if (((copyA && aLayout != 'T') || (copyB && bLayout != 'N')) && ikr) continue;

        int wgUM = wgM * um;
        int wgUN = wgN * un;

        if (ssUM % wgUM || ssUN % wgUN) continue;

        int wgPerSSM = ssUM / wgUM;
        int wgPerSSN = ssUN / wgUN;
        int wgPerSS = wgPerSSM * wgPerSSN;

        if (wgPerSS > MaxWGPerSS * wx) continue;

        int ssThreads = wgPerSS * wgM * wgN * wgK * wx;
        if (ssThreads > threadsPerSS(grfs)) continue;
        if (ssThreads >= threadsPerSS(grfs) && kb) continue;

        int baseKLoad = 2;
        if (hw >= HW::XeHPC) {
            baseKLoad = 8;
            if (cBytes * um * un * cComponents * 2 > grfs * grfBytes())
                baseKLoad = 4;
        }

        if (sys)
            baseKLoad = 32 / std::max(aBytesReal, bBytesReal);
        else if (dot)
            baseKLoad = 2 * grfBytes() / cBytesReal;
        else if (cComplex)
            baseKLoad >>= 1;

        baseKLoad = std::max(baseKLoad, mCP);

        int minBytes = aBytes * um * baseKLoad * aComponents
                     + bBytes * un * baseKLoad * bComponents
                     + cBytes * um * un * cComponents;
        if (minBytes > grfs * grfBytes()) continue;

        // Record group.
        Group group;
        group.um = um;
        group.un = un;
        group.wgM = wgM;
        group.wgN = wgN;
        group.wgK = wgK;
        group.wx = wx;
        group.ikr = ikr;
        group.copyA = copyA;
        group.copyB = copyB;
        group.kb = kb;
        group.baseKLoad = baseKLoad;
        group.grfs = grfs;
        group.id = list.groups.size();

        // A/B/C access.
        bool forceFN = (group.wgM & 1) != 0;
        bool forceSN = false;

        auto &aAccess = group.aAccess, &bAccess = group.bAccess, &cAccess = group.cAccess;
        cAccess = CAccess;

        bool doBlock2DA  = canBlock2DA  && (sys || (aLayout == 'T' && aBytesExtReal <= 4) || um * aBytesExt <= 64);
        bool doBlock2DB  = canBlock2DB  && (sys || (bLayout == 'N' && bBytesExtReal <= 4) || un * bBytesExt <= 64);
        bool doBlock2DTB = canBlock2DTB && (sys || (bLayout == 'N' && bBytesExtReal <= 4) || un * bBytesExt <= 64);

        if (aLayout == 'T' && aBytesExtReal == 8)
            doBlock2DA  &= (um * aBytesExtReal % 64 == 0);  /* only height = 8 for d64 transpose 2D block */
        if (bLayout == 'N' && bBytesExtReal == 8) {
            doBlock2DB  &= (un * bBytesExtReal % 64 == 0);
            doBlock2DTB &= (un * bBytesExtReal % 64 == 0);
        }

        switch (aLayout) {
            case 'N':
                aAccess = 'B';
                if ((sys || dp4a) && doBlock2DA)
                    aAccess = (aBytesExt >= 4 || aSubByte) ? 'm' : 'v';
                if (dot)
                    aAccess = 'S';
                break;
            case 'T':
                aAccess = doBlock2DA ? 't' : 'S';
                if (dot)
                    aAccess = 'B';
                break;
            default:  aAccess = 'B'; break;
        }

        switch (bLayout) {
            case 'N':
                bAccess = 'S';
                if (sys || dp4a)
                    bAccess = doBlock2DB ? 'm' : 'B';
                if (dot)
                    bAccess = 'B';
                break;
            case 'T':
                bAccess = 'B';
                if (sys || dp4a) {
                    if (doBlock2DB && sys)
                        bAccess = !copyB ? 't' : (bBytesExt >= 4 ? 'm' : 'v');
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
            if (aAccess == 'S' && copyA) aAccess = 'B';
            if (bAccess == 'S' && copyB) bAccess = 'B';
            if (cAccess == 'B' && cLayout == 'T') forceFN = true;
            if (cAccess != 'B' && cLayout == 'N') forceFN = true;
        }

        if (un == 1 && !doBlock2DB && !dot) bAccess = (bLayout == 'T') ? 'S' : 'B';

        if (hw < HW::XeHPG) {
            aAccess = std::tolower(aAccess);
            bAccess = std::tolower(bAccess);
            cAccess = std::tolower(cAccess);
        }

        // Decide on features.
        auto &features = group.featureMask;

        features = FeatureKALoadMult | FeatureKBLoadMult | FeatureACopies | FeatureBCopies;

        if (cComplex && !(sys && un > 1))           features |= FeatureSplitC;
        if (sys)                                    features |= FeatureMacroType;
        if (hw >= HW::XeHPC && sys)                 features |= FeatureRR;
        if (hw >= HW::XeHPC)                        features |= FeatureAPFMult | FeatureBPFMult | FeaturePFDist;
        if (copyA || copyB) {                       features |= FeatureUKSLMMult | FeatureSLMCopies | FeatureSLM3x;
            if (hw >= HW::XeHPC)                    features |= FeatureNB;
        }
        if (hw >= HW::XeHPC || copyA || copyB) {    features |= FeatureSM;
            if (!forceSN)                           features |= FeatureSN;
        }
        if (fusedEUs && !forceFN && !sys)           features |= FeatureFN;

        if (canBlock2DPFA && aLayout == 'N' && um * aBytesExt > 64)  features |= Feature1DPF;
        if (canBlock2DPFB && bLayout == 'T' && un * bBytesExt > 64)  features |= Feature1DPF;

        group.forceSN = forceSN;
        group.forceFN = forceFN && fusedEUs;

        group.dscale = 1.0f / distance(group, 0, ~0);   /* Calculate normalization factor (1 / max distance). */
        list.groups.push_back(group);
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

    return list;
}

void generateStrategyString(StrategySet &sset, const Stem &stem, const Group &group, bool final = false)
{
    Features features;
    std::memcpy(&features, &sset.id.fvector, sizeof(Features));

    // Map feature indices to logical values.
    int kaLoadMult  = 1 + features.kaLoadMult;  // {1, 2, 4}
    int kbLoadMult  = 1 + features.kbLoadMult;  // {1, 2, 4}
    int aCopies     = 1 + features.aCopies;     // {1, 2}
    int bCopies     = 1 + features.bCopies;     // {1, 2}
    int aPFMult     = features.aPFMult;         // {0, 1, 2, 4}
    int bPFMult     = features.bPFMult;         // {0, 1, 2, 4}
    int pfDistHalf  = 1 + features.pfDist;      // {1, 1.5, 2, ..., 4, 4.5}
    int ukSLMMult   = 1 + features.ukSLMMult;   // {1, 2, 4}
    int slmCopies   = 1 + features.slmCopies;   // {1, 2}

    if (stem.dot) {
        if (kaLoadMult == 3) kaLoadMult = 8;
        if (kbLoadMult == 3) kbLoadMult = 8;
    }

    if (stem.dp4a)
        if (ukSLMMult == 3) ukSLMMult = 8;

    if (kaLoadMult == 3 || kbLoadMult == 3 || ukSLMMult == 3) return;

    if (aPFMult == 3) aPFMult++;
    if (bPFMult == 3) bPFMult++;

    auto hw = currentHW();
    int um = group.um, un = group.un;
    int wgUM = um * group.wgM, wgUN = un * group.wgN;

    bool copyA = group.copyA, copyB = group.copyB;

    char aModel = ModelChar, bModel = ModelChar, cModel = ModelChar;
    char aLayout = stem.basicLayouts[0];
    char bLayout = stem.basicLayouts[1];
    int aBytes = stem.internalBytes[0];
    int bBytes = stem.internalBytes[1];
    int cBytes = stem.internalBytes[2];
    int aBytesExt = stem.externalBytes[0];
    int bBytesExt = stem.externalBytes[1];
    int aComponents = stem.components[0];
    int bComponents = stem.components[1];
    int cComponents = stem.components[2];
    bool cInteger = isInteger(stem.internalPrecisions[2]);
    bool aComplex = isComplex(stem.internalPrecisions[0]);
    bool bComplex = isComplex(stem.internalPrecisions[1]);
    bool cComplex = isComplex(stem.internalPrecisions[2]);
    int baseKLoad = group.baseKLoad;

    bool sys = stem.sys, dot = stem.dot;
    bool block2DA = isBlock2D(group.aAccess);
    bool block2DB = isBlock2D(group.bAccess);
    bool canBlock2DPFA = (hw >= HW::XeHPC) && (stem.alignments[0] >= 4);
    bool canBlock2DPFB = (hw >= HW::XeHPC) && (stem.alignments[1] >= 4);

    int kaLoad     = baseKLoad * kaLoadMult;
    int kbLoad     = baseKLoad * kbLoadMult;
    int aPF        = baseKLoad * aPFMult;
    int bPF        = baseKLoad * bPFMult;
    int aPFDist    = (baseKLoad * pfDistHalf) >> 1;
    int bPFDist    = (baseKLoad * pfDistHalf) >> 1;
    int unrollKSLM = baseKLoad * ukSLMMult;

    // Sanity check register usage.
    int minBytes = aBytes * um * kaLoad * aCopies * aComponents
                 + bBytes * un * kbLoad * bCopies * bComponents
                 + cBytes * um * un * cComponents;
    if (minBytes > group.grfs * grfBytes()) return;

    // SLM usage check.
    int slm = 0;
    if (copyA) slm += wgUM * unrollKSLM * aBytes * aComponents;
    if (copyB) slm += wgUN * unrollKSLM * bBytes * bComponents;
    slm *= group.wgK;

    int wgPerSS = (stem.ssUnrollM * stem.ssUnrollN) / (wgUM * wgUN);
    if (wgPerSS == 0)
        wgPerSS = threadsPerSS(group.grfs) / (group.wgM * group.wgN * group.wgK * group.wx);
    int slmBuffers = std::min(4, maxSLM() / (std::max(wgPerSS, 1) * std::max(slm, 1)));
    if (slmBuffers <= 0) return;

    if (features.slm3x) {
        if (slmBuffers != 4) return;
        slmBuffers = 3;
    }

    if (copyA && unrollKSLM < kaLoad) return;
    if (copyB && unrollKSLM < kbLoad) return;

    // Prefetch accesses.
    std::string aPFAccess = (aLayout == 'T' ? "S1," : "B");
    std::string bPFAccess = (bLayout == 'N' ? "S1," : "B");

    if (um * aBytesExt <= 64) aPFAccess = "S";
    if (un * bBytesExt <= 64) bPFAccess = "S";

    bool noPF2DA = features.pf1D && canBlock2DPFA && aPFAccess[0] == 'B';
    bool noPF2DB = features.pf1D && canBlock2DPFB && bPFAccess[0] == 'B';
    if (features.pf1D && !noPF2DA && !noPF2DB) return;
    if (canBlock2DPFA && !noPF2DA) aPFAccess = "m";
    if (canBlock2DPFB && !noPF2DB) bPFAccess = "m";

    if (hw < HW::XeHPG) {
        aPFAccess = lowercase(aPFAccess);
        bPFAccess = lowercase(bPFAccess);
    }

    // Create strategy string.
    std::string spec;
    spec.reserve(128);
    spec = stem.type;
    spec += ' ';
    spec += stem.precisions;
    spec += ' ';
    spec += stem.layouts;
    spec += ' ';
    spec += std::to_string(um);
    spec += ' ';
    spec += std::to_string(un);
    spec += ' ';
    spec += stem.alpha;
    spec += ' ';
    spec += stem.beta;
    spec += ' ';
    spec += aModel;
    spec += group.aAccess;
    spec += std::to_string(kaLoad);
    if (aCopies > 1) {
        spec += 'x';
        spec += std::to_string(aCopies);
    }
    if (aPF > 0) {
        spec += '+';
        spec += aPFAccess;
        spec += std::to_string(aPF);
        spec += '@';
        spec += std::to_string(aPFDist);
    }
    spec += ' ';

    spec += bModel;
    spec += group.bAccess;
    spec += std::to_string(kbLoad);
    if (bCopies > 1) {
        spec += 'x';
        spec += std::to_string(bCopies);
    }
    if (bPF > 0) {
        spec += '+';
        spec += bPFAccess;
        spec += std::to_string(bPF);
        spec += '@';
        spec += std::to_string(bPFDist);
    }
    spec += ' ';

    spec += cModel;
    spec += group.cAccess;

    spec += " wg ";
    spec += std::to_string(group.wgM);
    spec += 'x';
    spec += std::to_string(group.wgN);

    if (group.wgK > 1) {
        spec += 'x';
        spec += std::to_string(group.wgK);
        spec += group.ikr ? " ikr" : " kr";
    }

    if (group.wx > 1) spec += " wx" + std::to_string(group.wx);

    if (!sys && !dot && kaLoad > 1 && kbLoad > 1 && !cInteger) {
        spec += " kc";
        spec += std::to_string(std::min(kaLoad, kbLoad));
    }

    if (copyA || copyB) {
        spec += " c";
        if (copyA) spec += 'a';
        if (copyB) spec += 'b';
        spec += std::to_string(slmBuffers);
        if (slmCopies > 1) {
            spec += 'x';
            spec += std::to_string(slmCopies);
        }
        spec += " ks";
        spec += std::to_string(unrollKSLM);
        if (slmBuffers <= 2 && hw >= HW::XeHPC)
            spec += " ql";  /* try not to lock up entire EU with hard barrier */
    }

    if (sys) {
        if (features.macroType == 3 && un * (cComplex ? 2 : 1) <= 32)    /* only use fx if >4 src2s involved */
            return;
        const char *macroStrings[4] = {" af", " xaf", " xaf st", " xaf fx"};
        spec += macroStrings[features.macroType];
    }

    if (features.rr)
        spec += " rr";

    if (sys && hw < HW::XeHPC && (un % 2 == 0) && (un < 8 || (un % 8 == 0)))
        spec += " dw";

    if (features.fn || group.forceFN)
        spec += " fn nmk";

    if (sys && ((aComponents == 1 && bComponents == 1) || cComplex))
        spec += " vav";
    else
        spec += (hw >= HW::XeHP ? " nse" : " acb");

    if (hw >= HW::XeHPC)
        spec += " hi pt";   // Using Hilbert rather than boustrophedon to reduce the chance that
                            //  related WGs land on the same SS causing anomalously high performance
                            //  that only applies to the special test sizes.

    if (hw == HW::XeHPG)
        spec += final ? " bo" : " hi";  // For final kernels, use bo (faster), but use hi during
                                        //  autotuning for the reason given above.

    if (final) {
        spec += " sr";
        if (group.cAccess == 'B' && hw >= HW::XeHPC)
            spec += " br";
    } else if ((features.splitC || (cBytes < 4)) && hw >= HW::XeHPC)
        spec += " sr";
    else
        spec += " ar";      // For speed in kernel generation

    if (!copyA && !copyB) {
        int sb = std::max({32, baseKLoad * 4, kaLoad, kbLoad, aPF, bPF});
        spec += " sb";
        spec += std::to_string(sb);
    }

    if (hw < HW::XeHPG)
        spec += " bk0";

    if (features.sm) {
        if (copyA || aPF)
            spec += " sm";
        else
            return;
    }
    if (features.sn || group.forceSN) {
        if (copyB || bPF)
            spec += " sn";
        else
            return;
    }

    if (features.nb) {
        int nbM = (copyA && group.wgM > 1) ? group.wgM : 0;
        int nbN = (copyB && group.wgN > 1) ? group.wgN : 0;
        if (nbM == 0 && nbN == 0) return;
        spec += " nb ";
        spec += std::to_string(nbM);
        spec += 'x';
        spec += std::to_string(nbN);
    }

    bool doXA = false, doXB = false;

    if (aComplex) {
        bool isZ = stem.internalPrecisions[0] == 'Z';
        doXA |= features.splitC && !block2DA && (aLayout == 'N' || !copyA);
        doXA |= (group.aAccess == 't') && (isZ || features.splitC);
        if (doXA) spec += " xa";
    }

    if (bComplex) {
        bool isZ = stem.internalPrecisions[1] == 'Z';
        doXB |= !block2DB && ((isZ && bLayout == 'N') || sys);
        doXB |= isZ && (group.bAccess == 't');
        if (doXB) spec += " xb";
    }

    if (features.splitC) {
        spec += " xc";
        if (sys)
            spec += " cc";
    }

    spec += " dm";

    if (group.grfs != 128) {
        spec += " grf";
        spec += std::to_string(group.grfs);
    }

    if (group.kb) spec += " kb";

    if (!stem.extras.empty()) {
        spec += ' ';
        spec += stem.extras;
    }

    if (ModelChar == 's' && !contains(stem.extras, "pab") && !contains(stem.extras, "np"))
        spec += " pab";

    sset.spec = std::move(spec);
}

StrategySet generateStrategy(const Stem &stem, const Group &group, uint32_t fvector)
{
    StrategySet sset;
    sset.id.groupID = group.id;
    sset.id.fvector = fvector;
    sset.unrollM = group.um;
    sset.unrollN = group.un;
    generateStrategyString(sset, stem, group);
    return sset;
}

int insertSorted(std::vector<BenchmarkData> &sortedBData, BenchmarkData bdata)
{
    // Check for an existing entry with this strategy ID.
    // Remove any existing entry and merge with the current one.
    // Also limit the number of strategies per group.
    int inGroup = 0;
    for (auto iter = sortedBData.begin(); iter != sortedBData.end(); ) {
        if (iter->sset.id.groupID == bdata.sset.id.groupID) {
            if (iter->sset.id.fvector == bdata.sset.id.fvector) {
                bdata.results.merge(iter->results);
                sortedBData.erase(iter);
                break;
            } else if (++inGroup >= TopPicksPerGroup) {
                iter = sortedBData.erase(iter);
                continue;
            }
        }
        iter++;
    }

    // Insert into sorted array and return new index.
    const auto &before = std::upper_bound(sortedBData.begin(), sortedBData.end(), bdata,
                                          [](const BenchmarkData &bdata1, const BenchmarkData &bdata2) {
        return (bdata1.results.score() > bdata2.results.score());
    });

    int newIndex = std::distance(sortedBData.begin(), before);

    sortedBData.insert(before, bdata);

    return newIndex;
}

Stem inputStem(int argc, char **argv)
{
    Stem stem;
    int phase = 0;
    constexpr auto minPhases = 5, maxPhases = 8;
    std::string ssUnrollMStr, ssUnrollNStr;

    std::string *entries[maxPhases] = {
        &stem.type, &stem.precisions, &stem.layouts,
        &ssUnrollMStr, &ssUnrollNStr,
        &stem.alpha, &stem.beta, &stem.extras};

    auto append = [](std::string &s, auto t) {
        if (!s.empty()) s += ' ';
        s += t;
    };

    bool newPhase = true;
    for (int i = 1; i < argc; i++) {
        std::string term = argv[i];
        if (newPhase)
            entries[phase]->clear();
        append(*entries[phase], term);

        newPhase = true;
        if (phase == 0 && utils::kernelModifier(term) != utils::KernelModifier::None)
            newPhase = false;

        newPhase &= (phase < (maxPhases - 1));
        if (newPhase)
            phase++;
    }

    if (phase < minPhases)
        fail("Not enough arguments.");

    for (phase = 0; phase < maxPhases - 1; phase++)
        append(stem.all, *entries[phase]);

    stem.copy = false;
    stem.internalPrecisions = extractPrecisions(stem.precisions, false, stem.components);
    stem.externalPrecisions = extractPrecisions(stem.precisions, true);
    stem.basicLayouts = extractBasicLayouts(stem.layouts);
    extractAlignments(stem.layouts, stem.alignments);
    for (int i = 0; i < 3; i++) {
        stem.internalBytes[i] = typeBytes(stem.internalPrecisions[i]);
        stem.externalBytes[i] = typeBytes(stem.externalPrecisions[i]);
        if (stem.alignments[i] == 0)
            stem.alignments[i] = stem.externalBytes[i];
    }
    stem.sys = contains(stem.extras, "sys");
    stem.dot = contains(stem.extras, "dot");
    stem.dp4a = !stem.sys && std::toupper(stem.internalPrecisions[0]) == 'O' && std::toupper(stem.internalPrecisions[1]) == 'O';

    auto getUnroll = [](const std::string &s) {
        if (s == "-")
            return 0;
        else
            return std::stoi(s);
    };

    stem.ssUnrollM = getUnroll(ssUnrollMStr);
    stem.ssUnrollN = getUnroll(ssUnrollNStr);

    if (stem.ssUnrollM < 0 || stem.ssUnrollN < 0)
        fail("Invalid subslice tile size specified.");

    // Calculate C component count.
    auto &cComponents = stem.components[2];
    cComponents = std::max({stem.components[0], stem.components[1], stem.components[2]});
    auto cLimitPos = stem.extras.find("cl");
    if (cLimitPos != std::string::npos)
        cComponents = std::min(cComponents, std::stoi(stem.extras.substr(cLimitPos + 2)));

    return stem;
}

uint32_t rand32()
{
    static std::independent_bits_engine<std::mt19937, 32, uint32_t> generator(1);
    return generator();
}

bool randBool(float prob)
{
    return (rand32() < std::ldexp(prob, 32));
}

int GroupList::randomGroupID() const
{
    auto idx = rand32() % groups.size();
    return groups[idx].id;
}

const Group &GroupList::lookup(int groupID) const
{
    if (sparse) {
        for (auto &group: groups)
            if (group.id == groupID)
                return group;
        fail("Internal error: invalid group ID");
    } else
        return groups[groupID];
}

Group &GroupList::lookup(int groupID)
{
    return const_cast<Group &>(const_cast<const GroupList *>(this)->lookup(groupID));    /* ugh, C++ */
}

void GroupList::filter()
{
    groups.erase(std::remove_if(groups.begin(), groups.end(), [](const Group &group) {
                     return !group.flagged;
                 }),
                 groups.end());
    sparse = true;
}

Stream::Stream(int id_, const Stem &stem_, const GroupList &list_) : id(id_), stem(stem_), list(list_)
{
    mutate();
    baseSID = candidate.id;
}

void Stream::reset(const StrategyID &sid, float goodness)
{
    baseSID = sid;
    baseGoodness = goodness;
    mutate();
}

void Stream::candidateSucceeded(float goodness)
{
    baseSID = candidate.id;
    baseGoodness = goodness;
    mutate();
}

void Stream::candidateFailed()
{
    mutate();
}

void Stream::mutate()
{
    bool newGroup = true;

    if (baseGoodness > 0) {
        const float maxKeepProb = 0.95;         /* maximum probability we stay with this group */
        float keepGroupProb = std::min(baseGoodness, maxKeepProb);
        newGroup = !randBool(keepGroupProb);
    }

    StrategyID candidateSID;
    candidateSID.groupID = newGroup ? list.randomGroupID()
                                    : baseSID.groupID;

    auto *group = &list.lookup(candidateSID.groupID);

    candidate.spec.clear();
    int tries = 0;
    const int maxTriesPerGroup = 50;
    do {
        if (++tries >= maxTriesPerGroup) {
            // Couldn't find a valid strategy in this group. Try another group.
            tries = 0;
            newGroup = true;
            candidateSID.groupID = list.randomGroupID();
            group = &list.lookup(candidateSID.groupID);
        }

        // Choose a feature vector in the selected group.
        bool accept = false;
        do {
            candidateSID.fvector = rand32() & group->featureMask;

            if (newGroup)
                accept = true;
            else {
                float rgoodness = std::clamp((baseGoodness - 0.5f) * 2, 0.0f, 1.0f);
                float dist = distance(*group, baseSID.fvector, candidateSID.fvector);
                float acceptProb = (1 - rgoodness) * dist + rgoodness * (1 - dist);
                accept = randBool(acceptProb);
            }
        } while (!accept);

        candidate = generateStrategy(stem, *group, candidateSID.fvector);
    } while (candidate.spec.empty());

    candidate.streamID = id;
}

void printStatus(FILE *where, TestStatus status)
{
    const char *color = VT_RED, *name = "";
    switch (status) {
        case TestStatus::OK:              color = VT_GREEN;  name = "OK    "; break;
        case TestStatus::Stub:            color = VT_YELLOW; name = "Unimpl"; break;
        case TestStatus::HWUnsupported:   color = VT_YELLOW; name = "Unsupp"; break;
        case TestStatus::OutOfRegisters:  color = VT_YELLOW; name = "Spill "; break;
        case TestStatus::GenerationError: color = VT_RED;    name = "GenErr"; break;
        case TestStatus::SystemError:     color = VT_RED;    name = "SysErr"; break;
        case TestStatus::BenchmarkError:  color = VT_RED;    name = "BMErr "; break;
        case TestStatus::Inaccurate:      color = VT_RED;    name = "Inacc "; break;
        case TestStatus::Overrun:         color = VT_RED;    name = "WrOOB "; break;
    }
    utils::cprintf(where, color, "%s", name);
}

void printResults(FILE *where, const TestResults &results, bool highlight = false)
{
    const char *hbegin = highlight ? vt(VT_BOLD)  : "";
    const char *hend   = highlight ? vt(VT_RESET) : "";
    const char *blue   = vt(VT_BLUE), *reset = vt(VT_RESET);

    printStatus(where, results.status);
    if (MeanScore)
        fprintf(where, " %smean %s%s%.2f%s",   blue, reset, hbegin, results.effMean,   hend);
    else {
        fprintf(where, " %smedian %s%s%.2f%s", blue, reset, hbegin, results.effMedian, hend);
        fprintf(where, " %smax %s%s%.2f%s",    blue, reset, hbegin, results.effMax,    hend);
    }
}

void printStrategySet(FILE *where, const StrategySet &sset, bool highlight = false)
{
    if (highlight)
        utils::cprintf(where, VT_BOLD, " %s", sset.spec.c_str());
    else
        fprintf(where, " %s", sset.spec.c_str());
}

void printBenchmarkData(FILE *where, const BenchmarkData &bdata, bool highlight = false)
{
    printResults(where, bdata.results, highlight);
    printStrategySet(where, bdata.sset, highlight);
    fprintf(where, "\n");
}

int main(int argc, char **argv)
{
    int screenLimitSeconds = 300;
    bool showEachKernel = false;

    if (argc <= 1)
        usage(argv[0]);

    setenv("FORCE_ZEBIN", "1", false);  /* use zebin by default for faster kernel generation */

    auto lenv = getenv("SCREEN_TIME");
    if (lenv)
        screenLimitSeconds = atoi(lenv);

    auto krenv = getenv("MAX_WG_K");
    if (krenv)
        MaxWGK = atoi(krenv);

    auto wsenv = getenv("MAX_WG_PER_SS");
    if (wsenv)
        MaxWGPerSS = atoi(wsenv);

    auto geenv = getenv("SHOW_GENERR");
    bool showGenErr = geenv && atoi(geenv);

    auto tcenv = getenv("C_ACCESS");
    if (tcenv && *tcenv) CAccess = *tcenv;

    auto mbenv = getenv("MEMORY_BOUND");
    if (mbenv) MemoryBound = atoi(mbenv);

    auto thenv = getenv("THIN");
    if (thenv) NotThin = !atoi(thenv);

    auto kcenv = getenv("KERNEL_CACHE_SIZE");
    if (kcenv) KernelCacheSize = atoi(kcenv);

    auto cenv = getenv("COLD");
    if (cenv) {
        ColdA = ColdA || strstr(cenv, "A");
        ColdB = ColdB || strstr(cenv, "B");
        ColdC = ColdC || strstr(cenv, "C");
        if (strstr(cenv, "1")) ColdA = ColdB = ColdC = true;
    }

    auto saenv = getenv("SHOW_ALL_KERNELS");
    if (saenv) showEachKernel = atoi(saenv);

    auto sienv = getenv("SHOW_INPUTS");
    if (sienv) ShowInputs = atoi(sienv);

    auto menv = getenv("MODEL");
    if (menv && menv[0]) ModelChar = menv[0];

    auto benchenv = getenv("BENCHMARK_VERBOSE");
    if (benchenv) BenchmarkVerbose = atoi(benchenv);

    auto stem = inputStem(argc, argv);

    MemoryBound |= ((stem.ssUnrollM == 0) != (stem.ssUnrollN == 0));

    testPipes();
    startBenchmark();

    signal(SIGUSR1, sigusr1);
    signal(SIGPIPE, sigpipe);

    // Sorted benchmark results.
    std::vector<BenchmarkData> sortedBData;

    // Prepare kernel groups and streams.
    auto groups = generateGroups(stem);
    if (groups.groups.empty())
        fail("Could not generate any kernels. Tile size is probably too large.");

    const int nstreams = KernelCacheSize;
    std::vector<Stream> streams;
    for (int i = 0; i < nstreams; i++)
        streams.emplace_back(i, stem, groups);

    // Initialize strategy queue.
    std::deque<StrategySet> queued, preloaded;
    for (auto &s: streams)
        queued.push_back(s.getCandidate());

    // Moving kernels from queued -> preloaded.
    auto preload = [&] {
        while (preloaded.size() < KernelCacheSize && !queued.empty()) {
            auto strategy = pop(queued);
            preloadKernel(strategy.spec);
            preloaded.push_back(std::move(strategy));
        }
    };

    // Progress reports.
    int lastInsert = -1;

    auto top = [&]() {
        int bad = 0;
        for (int i = 0; i < sortedBData.size(); i++) {
            bool last = (i == lastInsert);
            bool print = (i < 30 || last);
            if (sortedBData[i].results.status != TestStatus::OK)
                print |= (bad++ < 10);

            if (print)
                printBenchmarkData(stderr, sortedBData[i], last);
        }
        fflush(stdout);
    };

    // Screening stages.
    int kept = 0, removed = 0;

    for (int stage = 1; stage <= 3; stage++) {
        lastInsert = -1;
        auto start = std::chrono::steady_clock::now();
        auto lastTop = std::chrono::steady_clock::now();

        prepareInput(stem, stage);

        while (true) {
            preload();
            if (preloaded.empty()) break;

            auto strategy = pop(preloaded);

            auto now = std::chrono::steady_clock::now();
            auto elapsed    = (now -   start) / std::chrono::seconds(1);
            auto elapsedTop = (now - lastTop) / std::chrono::seconds(1);

            if (elapsedTop >= 1) {
                std::cerr << vt(VT_CLEAR_SCREEN);
                std::cerr << "Screening stage " << stage << ": " << kept << " kept, " << removed << " removed... ";
                printStrategySet(stderr, strategy);
                std::cerr << "\n\n";
                float avgGoodness = 0.0f;
                for (auto &s: streams) avgGoodness += s.baseGoodness;
                avgGoodness /= streams.size();
                std::cerr << "  average goodness: " << int(avgGoodness * 100) << "%\t\t"
                          << "  kernels per second: " << (kept + removed) / elapsed << "\n\n";
                top();
                lastTop = now;
            }

            if (showEachKernel) {
                std::cerr << vt(VT_MAGENTA);
                printStrategySet(stderr, strategy);
                std::cerr << vt(VT_RESET);
                std::cerr << " ... ";
            }

            auto bdata = runBenchmark(strategy, false);

            if (showEachKernel) {
                std::cerr << "benchstone complete\n";
            }

            auto streamID = bdata.sset.streamID;
            Stream *stream = nullptr;

            if (streamID >= 0)
                stream = &streams[streamID];

            if ((bdata.results.status == TestStatus::OK && bdata.results.effMax > 0.)
                    || (bdata.results.status == TestStatus::GenerationError && showGenErr)) {
                lastInsert = insertSorted(sortedBData, bdata);

                // Calculate goodness: ratio of this score to top score.
                // Limit goodness if there aren't enough strategies yet (less than keepThresh).
                const int keepThresh = 100;
                float goodness = bdata.results.score() / sortedBData[0].results.score();
                goodness = std::min(goodness, kept / float(keepThresh));

                kept++;
                if (stream) stream->candidateSucceeded(goodness);
            } else {
                removed++;
                if (stream) stream->candidateFailed();
            }

            if (stream)
                queued.push_back(stream->getCandidate());

            if (screenLimitSeconds > 0) {
                if (elapsed >= screenLimitSeconds) {
                    std::cerr << "Reached time limit of " << screenLimitSeconds << "s for screening stage.\n";
                    break;
                }
            }

            if (gotSIGUSR1) {
                gotSIGUSR1 = false;
                std::cerr << "Caught SIGUSR1, concluding screening stage\n";
                break;
            }
        }

        // Prepare for next stage:
        //   - select top strategies and queue them for next stage
        //   - reinitialize streams from top strategies
        queued.clear();
        preloaded.clear();

        const float closeEnough = 0.03; /* keep if score within this fraction of top score */
        const int minKeep = 50;         /* but at least this many strategies for next stage */
        const int maxKeep = 200;        /* and no more than this many */

        for (auto &group: groups.groups)    /* unflag all groups */
            group.flagged = false;

        int scount = sortedBData.size();
        if (scount > 0) {
            float topScore = sortedBData[0].results.score();
            for (int i = 0; i < scount && i < maxKeep; i++) {
                auto &bdata = sortedBData[i];

                float goodness = bdata.results.score() / topScore;

                bdata.sset.streamID = -1;   /* dissociate from old stream */
                queued.push_back(bdata.sset);
                groups.lookup(bdata.sset.id.groupID).flagged = true;

                if (i >= minKeep && goodness < (1.0f - closeEnough))
                    break;
            }

            groups.filter();
            TopPicksPerGroup = std::max<int>(TopPicksPerGroup, TopPicks / groups.groups.size());

            /* Wait until after filtering groups to reset streams */
            for (int i = 0; i < scount && i < nstreams; i++) {
                auto &bdata = sortedBData[i];
                float goodness = bdata.results.score() / topScore;
                streams[i].reset(bdata.sset.id, goodness);
            }

            lastInsert = -1;
            sortedBData.clear();
        }

        kept = removed = 0;

        /* Append new candidates */
        for (auto &s: streams)
            queued.push_back(s.getCandidate());
    }

    // Final stage.
    int npick = std::min<int>(queued.size(), TopPicks);

    prepareInput(stem, 4);

    for (int i = 0; i < npick; i++) {
        auto strategy = pop(queued);
        generateStrategyString(strategy, stem, groups.lookup(strategy.id.groupID), true);
        std::cerr << vt(VT_CLEAR_SCREEN);
        std::cerr << "Testing " << i + 1 << '/' << npick << "... ";
        printStrategySet(stderr, strategy);
        std::cerr << "\n\n";
        top();
        lastInsert = insertSorted(sortedBData, runBenchmark(strategy));
    }

    std::cerr << vt(VT_CLEAR_SCREEN);
    for (int i = 0; i < sortedBData.size(); i++)
        printBenchmarkData(stdout, sortedBData[i]);
}
