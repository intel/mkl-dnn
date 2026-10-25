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

#include "gemmstone/strategy_parser.hpp"
#include "gemmstone/kernel_catalog.hpp"
#include "internal/utils.hpp"
#include "pieces/layout_utils.hpp"

#include <cctype>
#include <sstream>
#include <string>
#include <unordered_map>

GEMMSTONE_NAMESPACE_START

using namespace ngen;


static void unparseAccessType(std::ostream &s, const MatrixAddressingStrategy &astrategy);
static void unparseAddressBase(std::ostream &s, ngen::AddressBase base);
static void unparseCaching(HW hw, std::ostream &s, const MatrixAddressingStrategy &astrategy);
static void unparseTiling(std::ostream &s, const MatrixAddressingStrategy &astrategy);

static HW parseRAHW(const std::string &name) {
    if (name == "xehpc")
        return HW::XeHPC;

    stub("Only raxehpc is supported for register allocation override.");
    return HW::Unknown;
}

static const char *unparseRAHW(HW hw) {
    if (hw == HW::XeHPC)
        return "xehpc";
    return nullptr;
}

bool native64Bit(HW hw)
{
    EmulationStrategy emulate(hw);
    return !emulate.emulate64;
}

AccessType getAccessType(char c)
{
    switch (std::tolower(c)) {
        case 'b': return AccessType::Block;
        case 'p': return AccessType::PseudoBlock;
        case 's': return AccessType::Scattered;
        case 'u': return AccessType::ChannelScattered;
        case 'm': return AccessType::Block2D;
        case 't': return AccessType::Block2DTranspose;
        case 'v': return AccessType::Block2DVNNI;
        case 'c': return AccessType::CacheLine;
        default: stub("Unknown access type.");
    }
}

char downgradeBlock2D(char c)
{
    switch (std::tolower(c)) {
        case 'm':
        case 'v': return 'b';
        case 't': return 's';
        default:  return c;
    }
}

AddressBase getAddressBase(char c)
{
    switch (c) {
        case 'a': return AddressBase::createA64(true);
        case 'c': return AddressBase::createCC(0);
        case 'l': return AddressBase::createSLM();
        case 'm': return AddressBase::createSC(0);
        case 'r': return AddressBase{};
        case 's': return AddressBase::createBTS(0);
        default: stub("Unknown address space.");
    }
}

CacheSettingsLSC getCaching(char l1, char l3)
{
    if (l1 == 'd' && l3 == 'd')
        return CacheSettingsLSC::Default;

    bool l3cached = (l3 == 'c') || (l3 == 'b');
    switch (l1) {
        case 'u': return l3cached ? CacheSettingsLSC::L1UC_L3C : CacheSettingsLSC::L1UC_L3UC;
        case 't':
        case 'c': return l3cached ? CacheSettingsLSC::L1C_L3C  : CacheSettingsLSC::L1C_L3UC;
        case 's': return l3cached ? CacheSettingsLSC::L1S_L3C  : CacheSettingsLSC::L1S_L3UC;
        case 'b':
        case 'i': return CacheSettingsLSC::L1IAR_L3C; break;
        default: stub("Unknown cache setting");
    }
}

CacheSettingsLSC getCaching(char l1, char l2, char l3)
{
    if (l3 == 'u' || l3 == 'i') return getCaching(l1, l2);

    if (l3 == 'c' || l3 == 'b') {
        bool l2cached = (l2 == 'c') || (l2 == 'b');
        switch (l1) {
            case 'u':
                return l2cached ? CacheSettingsLSC::L1UC_L2C_L3C
                                : CacheSettingsLSC::L1UC_L2UC_L3C;
            case 't':
            case 'c':
                return l2cached ? CacheSettingsLSC::L1C_L2C_L3C
                                : CacheSettingsLSC::L1C_L2UC_L3C;
            case 's':
                return l2cached ? CacheSettingsLSC::L1S_L2C_L3C
                                : CacheSettingsLSC::L1S_L2UC_L3C;
            case 'b':
                if (!l2cached) return CacheSettingsLSC::L1WB_L2UC_L3WB;
            default: break;
        }
    }

    stub("Unknown cache setting");
}

CacheSettingsLSC getCachingEntry(std::stringstream &s, HW hw)
{
    if (hw >= HW::Xe3p) {
        char l1, l2, l3;
        s >> l1 >> l2 >> l3;
        return getCaching(l1, l2, l3);
    } else {
        char l1, l3;
        s >> l1 >> l3;
        return getCaching(l1, l3);
    }
}

void getCaching(std::stringstream &s, ProductFamily family, MatrixAddressingStrategy &astrategy, bool leaveDefault = false)
{
    auto hw = getCore(family);
    auto &cachingR = astrategy.cachingR;
    auto &cachingW = astrategy.cachingW;

    if (!leaveDefault) {
        cachingR = CacheSettingsLSC::L1C_L3C;
        // L1 uncached writes perform better on PVC.
        cachingW = (hw == HW::XeHPC) ? CacheSettingsLSC::L1UC_L3WB
                                     : CacheSettingsLSC::L1WB_L3WB;
    }

    if (s.peek() == '{') {
        char eat;
        s >> eat;
        cachingR = getCachingEntry(s, hw);
        s >> eat;
        if (eat == '/') {
            cachingW = getCachingEntry(s, hw);
            s >> eat;
        }
        if (eat != '}')
            stub("Invalid caching syntax");
    }
}

static void getTiling(std::stringstream &s, MatrixAddressingStrategy &astrategy)
{
    if (s.peek() == '#') {
        char eat = 0;
        int in = 0;
        s >> eat >> in, astrategy.tileR = in;
        s >> eat >> in, astrategy.tileC = in;
    }
}

struct ParserContext {
    ParserContext(GEMMStrategy &strategy, const GEMMProblem &problem, HW hw, std::stringstream &s)
        : strategy(strategy), problem(problem), hw(hw), s(s) {}
    GEMMStrategy &strategy;
    const GEMMProblem &problem;
    HW hw;
    std::stringstream &s;
    bool overrideFusedLoop = false;
    bool gotSR = false;
};

void parseStrategy(const std::string &str, const GEMMProblem &problem, GEMMStrategy &strategy)
{
    auto hw = getCore(problem.product.family);
    std::stringstream s(str);
    s.imbue(std::locale::classic());

    char eat, asA, asB, asC, accessA, accessB, accessC;
    char accessAUnaligned = '\0', accessBUnaligned = '\0';
    char accessAPrefetch = 's', accessBPrefetch = 's', accessCPrefetch = 's';
    char accessABPrefetchL3 = 'b';

    s >> std::ws >> asA >> accessA;
        if (s.peek() == '/') s >> eat >> accessAUnaligned;
        s >> strategy.ka_load;
        if (s.peek() == '/') s >> eat >> strategy.ka_load_masked;
        getTiling(s, strategy.A);
        if (s.peek() == 'x') s >> eat >> strategy.A_copies;
        getCaching(s, problem.product.family, strategy.A);
    if (s.peek() == '+') {
        s >> eat;
        if (s.peek() != '+') {
            strategy.prefetchA = 1;
            s >> accessAPrefetch >> strategy.ka_prefetch;
            if (s.peek() == ',') s >> eat >> strategy.ka_pfStride;
            if (s.peek() == '@') s >> eat >> strategy.prefetchA;
            if (s.peek() == '/') s >> eat >> strategy.prefetchAMasked;
            else strategy.prefetchAMasked = strategy.prefetchA;
            getCaching(s, problem.product.family, strategy.A_prefetch);
        }
    }
    if (s.peek() == '+') {
        strategy.l3PrefetchA = true;
        s >> eat >> accessABPrefetchL3 >> strategy.ka_prefetchL3;
        if (s.peek() == '@') s >> eat >> strategy.prefetchABL3;
        getCaching(s, problem.product.family, strategy.AB_prefetchL3, true);
    }
    s >> std::ws >> asB >> accessB;
        if (s.peek() == '/') s >> eat >> accessBUnaligned;
        s >> strategy.kb_load;
        if (s.peek() == '/') s >> eat >> strategy.kb_load_masked;
        getTiling(s, strategy.B);
        if (s.peek() == 'x') s >> eat >> strategy.B_copies;
        getCaching(s, problem.product.family, strategy.B);
    if (s.peek() == '+') {
        s >> eat;
        if (s.peek() != '+') {
            strategy.prefetchB = 1;
            s >> accessBPrefetch >> strategy.kb_prefetch;
            if (s.peek() == ',') s >> eat >> strategy.kb_pfStride;
            if (s.peek() == '@') s >> eat >> strategy.prefetchB;
            if (s.peek() == '/') s >> eat >> strategy.prefetchBMasked;
            else strategy.prefetchBMasked = strategy.prefetchB;
            getCaching(s, problem.product.family, strategy.B_prefetch);
        }
    }
    if (s.peek() == '+') {
        strategy.l3PrefetchB = true;
        s >> eat >> accessABPrefetchL3 >> strategy.kb_prefetchL3;
        if (s.peek() == '@') s >> eat >> strategy.prefetchABL3;
        getCaching(s, problem.product.family, strategy.AB_prefetchL3, true);
    }
    s >> std::ws >> asC >> accessC;
        getTiling(s, strategy.C);
        getCaching(s, problem.product.family, strategy.C);
    if (s.peek() == '+') {
        strategy.prefetchC = 1;
        s >> eat >> accessCPrefetch;
        if (s.peek() == '@') s >> eat >> strategy.prefetchC;
        getCaching(s, problem.product.family, strategy.C_prefetch);
    }

    auto A64 = AddressBase::createA64(true);
    auto BTS = AddressBase::createBTS(0);

    if (!accessAUnaligned) accessAUnaligned = downgradeBlock2D(accessA);
    if (!accessBUnaligned) accessBUnaligned = downgradeBlock2D(accessB);

    strategy.A.base = strategy.A_prefetch.base = getAddressBase(asA);
    strategy.B.base = strategy.B_prefetch.base = getAddressBase(asB);
    strategy.C.base = strategy.C_prefetch.base = getAddressBase(asC);
    strategy.CO.base = (hw >= HW::XeHPG) ? A64 : BTS;
    strategy.A.newDP = bool(std::isupper(accessA));
    strategy.B.newDP = bool(std::isupper(accessB));
    strategy.C.newDP = bool(std::isupper(accessC));
    strategy.CO.newDP = strategy.C.newDP;
    strategy.A.accessType = getAccessType(accessA);
    strategy.B.accessType = getAccessType(accessB);
    strategy.C.accessType = getAccessType(accessC);
    strategy.unalignedAccA = getAccessType(accessAUnaligned);
    strategy.unalignedAccB = getAccessType(accessBUnaligned);
    strategy.A.cachingW = CacheSettingsLSC::Default;
    strategy.B.cachingW = CacheSettingsLSC::Default;
    strategy.CO.cachingR = CacheSettingsLSC::L1C_L3C;
    if (hw >= HW::Xe3p) strategy.CO.cachingR = CacheSettingsLSC::L1C_L2C_L3C;
    strategy.A_prefetch.prefetch = true;
    strategy.B_prefetch.prefetch = true;
    strategy.C_prefetch.prefetch = true;
    strategy.A_prefetch.newDP = bool(std::isupper(accessAPrefetch));
    strategy.B_prefetch.newDP = bool(std::isupper(accessBPrefetch));
    strategy.C_prefetch.newDP = bool(std::isupper(accessCPrefetch));
    strategy.A_prefetch.accessType = getAccessType(accessAPrefetch);
    strategy.B_prefetch.accessType = getAccessType(accessBPrefetch);
    strategy.C_prefetch.accessType = getAccessType(accessCPrefetch);
    strategy.A_prefetch.cachingW = CacheSettingsLSC::Default;
    strategy.B_prefetch.cachingW = CacheSettingsLSC::Default;
    strategy.C_prefetch.cachingW = CacheSettingsLSC::Default;

    strategy.AB_prefetchL3.prefetch = true;
    strategy.AB_prefetchL3.newDP = true;
    strategy.AB_prefetchL3.padded = true;
    strategy.AB_prefetchL3.accessType = getAccessType(accessABPrefetchL3);
    strategy.AB_prefetchL3.base = getAddressBase(strategy.l3PrefetchA ? asA : asB);
    if (strategy.AB_prefetchL3.cachingR == CacheSettingsLSC::Default) {
        strategy.AB_prefetchL3.cachingR = CacheSettingsLSC::L1UC_L3C;
        if (hw >= HW::Xe3p)
            strategy.AB_prefetchL3.cachingR = CacheSettingsLSC::L1UC_L2C_L3C;
    }

    strategy.A.padded |= isPacked(problem.A.layout);
    strategy.B.padded |= isPacked(problem.B.layout);
    strategy.A_prefetch.padded |= isPacked(problem.A.layout);
    strategy.B_prefetch.padded |= isPacked(problem.B.layout);

    strategy.unroll[LoopK] = 1;
    strategy.checkAdd32 = !native64Bit(hw) || (hw == HW::XeHPC);
    strategy.altCRemainder |= (strategy.C.accessType == AccessType::Block) || strategy.kParallel;

    // These parsers are for basic flags, with no attached parametrization. For parameters with
    // associated values (e.g. grf256), use a basic if/else if chain afterward
    using BasicParameterParser = void(*)(ParserContext&);
    using BasicParser = std::unordered_map<std::string, BasicParameterParser>;
    static const BasicParser parser {
        /* Allocation Settings */
        {"cs", [](ParserContext& ctx) { ctx.strategy.registerScheme = GEMMStrategy::CSeparate; }},
        {"acb", [](ParserContext& ctx) { ctx.strategy.registerScheme = GEMMStrategy::ACB; }},
        {"bca", [](ParserContext& ctx) { ctx.strategy.registerScheme = GEMMStrategy::BCA; }},
        {"vnc", [](ParserContext& ctx) { ctx.strategy.registerScheme = GEMMStrategy::VNC; }},
        {"int", [](ParserContext& ctx) { ctx.strategy.registerScheme = GEMMStrategy::ABInterleave; }},
        {"nse", [](ParserContext& ctx) { ctx.strategy.registerScheme = GEMMStrategy::NSeparate; }},
        {"vav", [](ParserContext& ctx) { ctx.strategy.registerScheme = GEMMStrategy::VAvoid; }},
        {"tk", [](ParserContext& ctx) { ctx.strategy.tokenAlloc = true; }},

        /* Dispatch Settings */
        {"mnk", [](ParserContext& ctx) {
            ctx.strategy.loopOrder[0] = LoopM;
            ctx.strategy.loopOrder[1] = LoopN;
            ctx.strategy.loopOrder[2] = LoopK;
        }},
        {"nmk", [](ParserContext& ctx) {
            ctx.strategy.loopOrder[0] = LoopN;
            ctx.strategy.loopOrder[1] = LoopM;
            ctx.strategy.loopOrder[2] = LoopK;
        }},
        {"bo", [](ParserContext& ctx) { ctx.strategy.cWalkOrder = WalkOrder::Boustrophedon; }},
        {"hi", [](ParserContext& ctx) { ctx.strategy.cWalkOrder = WalkOrder::Hilbertlike; }},
        {"nl", [](ParserContext& ctx) { ctx.strategy.cWalkOrder = WalkOrder::NestedLinear; }},
        {"li", [](ParserContext& ctx) { ctx.strategy.cWalkOrder = WalkOrder::SimpleLinear; }},
        {"pt", [](ParserContext& ctx) { ctx.strategy.persistent = true; }},
        {"fm", [](ParserContext& ctx) {
            ctx.strategy.fusedLoop = LoopM;
            ctx.overrideFusedLoop = true;
        }},
        {"fn", [](ParserContext& ctx) {
            ctx.strategy.fusedLoop = LoopN;
            ctx.overrideFusedLoop = true;
        }},
        {"rm", [](ParserContext& ctx) { ctx.strategy.reverse[LoopM] = true; }},
        {"rn", [](ParserContext& ctx) { ctx.strategy.reverse[LoopN] = true; }},
        {"ym", [](ParserContext& ctx) { ctx.strategy.scramble[LoopM] = true; }},
        {"yn", [](ParserContext& ctx) { ctx.strategy.scramble[LoopN] = true; }},
        {"cti", [](ParserContext& ctx) { ctx.strategy.cInterleave = true; }},
        
        /* K-Parallelism Settings */
        {"kr", [](ParserContext& ctx) { ctx.strategy.kParallelLocal = true; }},
        {"akr", [](ParserContext& ctx) { ctx.strategy.kParallelLocal = ctx.strategy.shrinkWGK = true; }},
        {"ikr", [](ParserContext& ctx) { ctx.strategy.kParallelLocal = ctx.strategy.kInterleave = true; }},
        {"fb", [](ParserContext& ctx) { ctx.strategy.fuseBeta = true; }},
        {"afb", [](ParserContext& ctx) { ctx.strategy.fuseBeta = ctx.strategy.altFusedBeta = true; }},
        {"fp", [](ParserContext& ctx) { ctx.strategy.fusePostOps = true; }},
        {"zt", [](ParserContext& ctx) { ctx.strategy.zeroTempC = true; }},
        {"rx", [](ParserContext& ctx) { ctx.strategy.relaxedAccumulation = true; }},
        {"kb", [](ParserContext& ctx) { ctx.strategy.kParallel = true; }},
        {"kv", [](ParserContext& ctx) { ctx.strategy.kParallelVariable = true; }},

        /* Systolic Settings */
        {"sys", [](ParserContext& ctx) { ctx.strategy.systolic = true; }},
        {"fs", [](ParserContext& ctx) {
            ctx.strategy.fixedSystolic = ctx.strategy.systolic = true;
            ctx.strategy.CO.base = AddressBase::createBTS(0);
        }},
        {"sc", [](ParserContext& ctx) { ctx.strategy.splitCopy = true; }},
        {"dw", [](ParserContext& ctx) { ctx.strategy.dpasw = true; }},
        {"af", [](ParserContext& ctx) { ctx.strategy.atomicFMA = true; }},
        {"xaf", [](ParserContext& ctx) { ctx.strategy.atomicFMA = ctx.strategy.extendedAtomicFMA = true; }},
        {"fx", [](ParserContext& ctx) { ctx.strategy.fmaBoustrophedon = true; }},

        // Thread Cooperation Settings
        {"sm", [](ParserContext& ctx) { ctx.strategy.coopA = CoopSplit::MN; }},
        {"sn", [](ParserContext& ctx) { ctx.strategy.coopB = CoopSplit::MN; }},
        {"ska", [](ParserContext& ctx) { ctx.strategy.coopA = CoopSplit::FullK; }},
        {"skb", [](ParserContext& ctx) { ctx.strategy.coopB = CoopSplit::FullK; }},

        /* Miscellaneous Settings */
        {"ch", [](ParserContext& ctx) { ctx.strategy.checkAdd32 = true; }},
        {"nch", [](ParserContext& ctx) { ctx.strategy.checkAdd32 = false; }},
        {"cc", [](ParserContext& ctx) { ctx.strategy.forceCopyC = true; }},
        {"sf", [](ParserContext& ctx) { ctx.strategy.strictFence = true; }},
        {"ws", [](ParserContext& ctx) { ctx.strategy.wgInSS = true; }},
        {"wc", [](ParserContext& ctx) { ctx.strategy.C.smode = ScatterSIMD::Wide; }},
        {"njs", [](ParserContext& ctx) { ctx.strategy.jointSplit = false; }},
        {"up", [](ParserContext& ctx) { ctx.strategy.panelCheck = true; }},
        {"ek", [](ParserContext& ctx) { ctx.strategy.slmEarlyKMask = true; }},
        {"di", [](ParserContext& ctx) { ctx.strategy.delayABInc = true; }},
        {"pfaux", [](ParserContext& ctx) { ctx.strategy.pfaux = true; }},
        {"nq", [](ParserContext& ctx) {
            auto &strategy = ctx.strategy;
            strategy.A.noExtraPad = strategy.A_prefetch.noExtraPad = true;
            strategy.B.noExtraPad = strategy.B_prefetch.noExtraPad = true;
            strategy.C.noExtraPad = strategy.C_prefetch.noExtraPad = true;
            strategy.CO.noExtraPad = true;
        }},

        /* Remainder Handling */
        {"ar", [](ParserContext& ctx) { ctx.strategy.altCRemainder = true; }},
        {"sr", [](ParserContext& ctx) {
            ctx.strategy.altCRemainder = false;
            ctx.gotSR = true;
        }},
        {"br", [](ParserContext& ctx) { ctx.strategy.block2DCRemainder = true; }},
        {"bf", [](ParserContext& ctx) { ctx.strategy.block2DCFull = ctx.strategy.block2DCRemainder = true; }},

        /* Load and Compute */
        {"ac", [](ParserContext& ctx) { ctx.strategy.cAccumulators = true; }},
        {"el", [](ParserContext& ctx) { ctx.strategy.cLoadAhead = true; }},
        {"ba", [](ParserContext& ctx) { ctx.strategy.loadBFirst = true; }},
        {"st", [](ParserContext& ctx) { ctx.strategy.stallAfterLoad = true; }},
        {"dm", [](ParserContext& ctx) { ctx.strategy.doubleMasking = true; }},
        {"kd", [](ParserContext& ctx) { ctx.strategy.kDescRem = true; }},
        {"cp", [](ParserContext& ctx) { ctx.strategy.chainPrefetches = true; }},
        {"dsl", [](ParserContext& ctx) { ctx.strategy.isDSLGenerator = true; }},

        /* SLM Settings */
        {"ni", [](ParserContext& ctx) { ctx.strategy.slmUseIncrCopy = false; }},
        {"ta", [](ParserContext& ctx) { ctx.strategy.slmATrans = true; }},
        {"tb", [](ParserContext& ctx) { ctx.strategy.slmBTrans = true; }},

        /* Padding Settings */
        {"np", [](ParserContext& ctx) {
            ctx.strategy.A.padded = ctx.strategy.A_prefetch.padded = false;
            ctx.strategy.B.padded = ctx.strategy.B_prefetch.padded = false;
        }},
        {"pab", [](ParserContext& ctx) {
            ctx.strategy.A.padded = ctx.strategy.A_prefetch.padded = true;
            ctx.strategy.B.padded = ctx.strategy.B_prefetch.padded = true;
        }},
        {"pc", [](ParserContext& ctx) { ctx.strategy.C.padded = ctx.strategy.C_prefetch.padded = true; }},

        /* Atomic Settings */
        {"au", [](ParserContext& ctx) { ctx.strategy.C.atomic = ctx.strategy.CO.atomic = true; }},
        {"nau", [](ParserContext& ctx) { ctx.strategy.C.atomic = ctx.strategy.CO.atomic = ctx.strategy.autoatomic = false; }},

        /* Workgroup Settings */
        {"ff", [](ParserContext& ctx) { ctx.strategy.forceWGUpdate = WGFixed; }},
        {"ffk", [](ParserContext& ctx) { ctx.strategy.forceFixedWGK = true; }},
        {"wg", [](ParserContext& ctx) {
            char x;
            ctx.s >> ctx.strategy.wg[LoopM];
            ctx.s >> x;
            ctx.s >> ctx.strategy.wg[LoopN];
            ctx.strategy.wg[LoopK] = 0;
            if (ctx.s.peek() == 'x')
                ctx.s >> x >> ctx.strategy.wg[LoopK];
        }},
        {"nb", [](ParserContext& ctx) {
            char x;
            ctx.s >> ctx.strategy.namedBarriers[LoopM];
            ctx.s >> std::ws >> x;
            ctx.s >> ctx.strategy.namedBarriers[LoopN];
        }},
        {"fg", [](ParserContext& ctx) {
            float fillGoal;
            ctx.s >> fillGoal;
            ctx.strategy.fillGoal = int(fillGoal * 16);
        }},

        /* Thread Arbitration */
        {"of", [](ParserContext& ctx) { ctx.strategy.arbitrationMode = ThreadArbitrationMode::OldestFirst; }},
        {"rr", [](ParserContext& ctx) { ctx.strategy.arbitrationMode = ThreadArbitrationMode::RoundRobin; }},
        {"rrs", [](ParserContext& ctx) { ctx.strategy.arbitrationMode = ThreadArbitrationMode::RoundRobinOnStall; }},

        /* Alignment and Warmup */
        {"l2d", [](ParserContext& ctx) { ctx.strategy.optAlignAB2D = true; }},
        {"wt", [](ParserContext& ctx) { ctx.strategy.tlbWarmup = true; }},
    };

    ParserContext ctx{strategy, problem, hw, s};

    while (!s.eof()) {
        std::string mod;
        s >> mod;

        // Check the dispatch table first
        auto it = parser.find(mod);
        if (it != parser.end()) {
            it->second(ctx);
            continue;
        }

        if (mod.substr(0, 3) == "grf") {
            mod.erase(0,3);
            strategy.GRFs = std::stoi(mod);
        } else if (mod.substr(0, 3) == "dot") {
            mod.erase(0,3);
            strategy.dotVL = mod.empty() ? 1 : std::stoi(mod);
        } else if (mod.length() >= 2) {
            if (mod.substr(0, 2) == "ms")
                strategy.mSplitThresh = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ns")
                strategy.nSplitThresh = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ra" && mod.length() > 2 && std::isdigit(mod[2]))
                strategy.ka_repack = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ra")
                strategy.raHW = parseRAHW(mod.substr(2));
            else if (mod.substr(0, 2) == "rb")
                strategy.kb_repack = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ki")
                strategy.kInterleaveChunk = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "kc")
                strategy.kChain = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ka")
                strategy.extraKAlign = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "ks") {
                char eat;
                std::stringstream ms(mod);
                ms.imbue(std::locale::classic());
                ms >> eat >> eat >> strategy.unrollKSLM;
                if (!ms.eof() && (ms.peek() == '/'))
                    ms >> eat >> strategy.unrollKSLMMasked;
            } else if (mod.substr(0, 2) == "sb") {
                strategy.barrierFreq = stoi(mod.substr(2));
                strategy.splitBarrier = true;
            } else if (mod.substr(0, 2) == "pk")
                strategy.kPadding = stoi(mod.substr(2));
            else if (mod.substr(0, 2) == "cr") {
                strategy.cRepackPanel = stoi(mod.substr(2));
                if (strategy.cRepackPanel == 0)
                    strategy.cRepackPanel = 1024;   /* arbitrary large value */
            } else if (mod.substr(0, 2) == "rc") {
                strategy.repackC = stoi(mod.substr(2));
                if (strategy.cRepackPanel == 0)
                    strategy.cRepackPanel = 1024;   /* arbitrary large value */
            } else if (mod.substr(0, 2) == "wx") {
                strategy.wgPadFactor = stoi(mod.substr(2));
                strategy.forceWGUpdate = WGFixed;
            } else if (mod.substr(0, 2) == "ql") {
                strategy.skewLocalIDs = true;
            } else switch (mod[0]) {
                case 'b':
                    if (isdigit(mod[1]))
                        strategy.barrierFreq = stoi(mod.substr(1));
                    else {
                        LoopType loop;
                        switch (mod[1]) {
                            case 'm': loop = LoopM; break;
                            case 'n': loop = LoopN; break;
                            case 'k': loop = LoopK; break;
                            default: stub("Unknown strategy modifier.");
                        }
                        size_t alt;
                        strategy.blocking[loop] = stoi(mod.substr(2), &alt);
                        if (strategy.blocking[loop] == 0)
                            strategy.blocking[loop] = 16777216;
                        alt += 3;
                        if (mod.length() > alt)
                            strategy.blockingAlt[loop] = stoi(mod.substr(alt));
                    }
                    break;
                case 'c': {
                    mod.erase(0, 1);
                    if (mod[0] == 'a') {
                        bool cab_case = false;
                        mod.erase(0, 1);
                        if (mod[0] == 'b') {
                            mod.erase(0, 1);
                            cab_case = true;
                        }
                        std::stringstream ms(mod);
                        strategy.slmA = true;
                        ms.imbue(std::locale::classic());
                        ms >> strategy.slmBuffersA;
                        if (cab_case) {
                            strategy.slmBuffersB = strategy.slmBuffersA;
                            strategy.slmB = true;
                        }
                        ms >> eat;
                        if (!ms.eof())
                            ms >> strategy.slmCopies; // TODO: split to A/B copies
                    }
                    if (mod[0] == 'b') {
                        bool cba_case = false;
                        mod.erase(0, 1);
                        if (mod[0] == 'a') {
                            mod.erase(0, 1);
                            cba_case = true;
                        }
                        std::stringstream ms(mod);
                        strategy.slmB = true;
                        ms.imbue(std::locale::classic());
                        ms >> strategy.slmBuffersB;
                        if (cba_case) {
                            strategy.slmBuffersA = strategy.slmBuffersB;
                            strategy.slmA = true;
                        }
                        ms >> eat;
                        if (!ms.eof())
                            ms >> strategy.slmCopies; // TODO: split to A/B copies
                    }
                    if (strategy.slmA && strategy.slmB && strategy.slmBuffersB != strategy.slmBuffersA)
                        stub("SLM buffer counts for A and B must match.");

                    break;
                }
                case 'k': {
                    char eat;
                    std::stringstream ms(mod);
                    ms.imbue(std::locale::classic());
                    ms >> eat >> strategy.unroll[LoopK];
                    if (!ms.eof() && (ms.peek() == '/'))
                        ms >> eat >> strategy.unrollK_masked;
                    break;
                } case 'l':
                    strategy.optAlignAB = stoi(mod.substr(1));
                    break;
                default:
                    stub("Unknown strategy modifier.");
            }
        } else if (!mod.empty())
            stub("Unknown strategy modifier.");
    }

    if (!ctx.overrideFusedLoop) {
        if (strategy.fused) {
            if (strategy.wg[LoopM] == 1)
                strategy.fusedLoop = LoopN;
            else if (strategy.wg[LoopN] == 1)
                strategy.fusedLoop = LoopM;
            else
                strategy.fusedLoop = strategy.loopOrder[0];
        } else
            strategy.fusedLoop = strategy.loopOrder[0];
    }

    if (strategy.ka_pfStride == 0) strategy.ka_pfStride = strategy.ka_prefetch;
    if (strategy.kb_pfStride == 0) strategy.kb_pfStride = strategy.kb_prefetch;

    if (strategy.block2DCRemainder && !ctx.gotSR) strategy.altCRemainder = true;

    if (strategy.persistentLoop() && !strategy.linearOrder()) strategy.cWalkOrder = WalkOrder::SimpleLinear;

    size_t poCount = problem.postOps.len();
    strategy.binary.resize(poCount);
    for (auto &astrategy: strategy.binary) {
        astrategy.base = (hw >= HW::XeHPC) ? A64 : BTS;
        astrategy.newDP = strategy.C.newDP;
    }

    bool surfaceAq = ((problem.quantized2DA() || problem.needsSeparateASums()) && !strategy.A.base.isStateless());
    bool surfaceBq = ((problem.quantized2DB() || problem.needsSeparateBSums()) && !strategy.B.base.isStateless());

    bool surfaceAO = surfaceAq && problem.aOffset2D();
    bool surfaceBO = surfaceBq && problem.bOffset2D();

    strategy.AO.base = (surfaceAO ? BTS : A64);
    strategy.BO.base = (surfaceBO ? BTS : A64);
    strategy.Ag.base = strategy.A_scale.base = (surfaceAq ? BTS : A64);
    strategy.Bg.base = strategy.B_scale.base = (surfaceBq ? BTS : A64);

    strategy.AO.newDP = strategy.A_scale.newDP = strategy.Ag.newDP = (hw >= HW::XeHPG);
    strategy.BO.newDP = strategy.B_scale.newDP = strategy.Bg.newDP = (hw >= HW::XeHPG);
}

// Deprecated, kept for backward compatibility.
void parseStrategy(const std::string &str, HW hw, const GEMMProblem &problem, GEMMStrategy &strategy)
{
    if (getCore(problem.product.family) == hw)
        return parseStrategy(str, problem, strategy);
    auto problemHW = problem;
    problemHW.product = Product(genericProductFamily(hw), 0, PlatformType::Unknown);
    parseStrategy(str, problemHW, strategy);
}

void adjustStrategy(HW hw, const GEMMProblem &problem, GEMMStrategy &strategy, const char *tags)
{
    auto *gemmAStrategy = &strategy.A, *gemmBStrategy = &strategy.B;
    if (problem.A.needA64) strategy.A.forceA64();
    if (problem.B.needA64) strategy.B.forceA64();
    if (problem.C.needA64) strategy.C.forceA64();

    // 2D block accesses use 2D addressing where supported.
    auto use2DAddressing = [](MatrixAddressingStrategy &astrategy) {
        astrategy.address2D |= isBlock2D(astrategy.accessType);
    };

    if (!isPacked(problem.A.layout)) {
        use2DAddressing(strategy.A);
        use2DAddressing(strategy.A_prefetch);
    }
    if (!isPacked(problem.B.layout)) {
        use2DAddressing(strategy.B);
        use2DAddressing(strategy.B_prefetch);
    }
    if (!isPacked(problem.C.layout)) {
        use2DAddressing(strategy.C);
        use2DAddressing(strategy.C_prefetch);
    }
    if (!(strategy.l3PrefetchA && isPacked(problem.A.layout)) && !(strategy.l3PrefetchB && isPacked(problem.B.layout)))
        use2DAddressing(strategy.AB_prefetchL3);

    // Notify kernel generator to downgrade block 2D prefetches if block 2D cannot be used.
    if (tags && !strategy.optAlignAB2D) {
        bool block2DA = false, block2DB = false;
        while (*tags) {
            block2DA |= (*tags == kcatalog::ReqBlock2DA);
            block2DB |= (*tags == kcatalog::ReqBlock2DB);
            tags++;
        }

        strategy.A_prefetch.preflight(hw);
        strategy.B_prefetch.preflight(hw);
        if (!block2DA && strategy.A_prefetch.address2D)
            strategy.A_prefetch.address2D = false;
        if (!block2DB && strategy.B_prefetch.address2D)
            strategy.B_prefetch.address2D = false;
    }

    // No need to use split remainder handling for 2D block accesses as there's no penalty for masking.
    if (isBlock2D(strategy.A.accessType) && (!strategy.prefetchA || isBlock2D(strategy.A_prefetch.accessType)))
        if (!problem.quantized2DA() && !problem.needsAGroupSums())
            strategy.remHandling[LoopM] = RemainderHandling::General;
    if (isBlock2D(strategy.B.accessType) && (!strategy.prefetchB || isBlock2D(strategy.B_prefetch.accessType)))
        if (!problem.quantized2DB() && !problem.needsBGroupSums())
            strategy.remHandling[LoopN] = RemainderHandling::General;

    // Also don't split remainder handling if padded.
    if (gemmAStrategy->padded) strategy.remHandling[LoopM] = RemainderHandling::General;
    if (gemmBStrategy->padded) strategy.remHandling[LoopN] = RemainderHandling::General;

    // But always use split remainder handling when prefetching C if it _isn't_ block 2D
    //  ... in that case there are no C prefetches on the remainder path.
    if (strategy.prefetchC && !isBlock2D(strategy.C_prefetch.accessType))
        strategy.remHandling[LoopM] = strategy.remHandling[LoopN] = RemainderHandling::Split;

    // ... and always use split remainder handling on later GPUs when panel checks are active.
    // kv late-exits through the fused beta/post-ops preflight derives.
    bool lateExit = strategy.lateExit() || strategy.kParallelVariable;
    if (strategy.panelCheck && lateExit && hw >= HW::XeHP && !strategy.fixedSystolic) {
        if (isPacked(problem.A.layout)) strategy.remHandling[LoopM] = RemainderHandling::Split;
        if (isPacked(problem.B.layout)) strategy.remHandling[LoopN] = RemainderHandling::Split;
    }

    // Finally, no remainder handling needed for GEMV-type kernels.
    if (strategy.unroll[LoopM] * strategy.wg[LoopM] == 1) strategy.remHandling[LoopM] = RemainderHandling::Ignore;
    if (strategy.unroll[LoopN] * strategy.wg[LoopN] == 1) strategy.remHandling[LoopN] = RemainderHandling::Ignore;
}

const char *parseLayout(const char *s, MatrixAddressing &atype)
{
    atype.crosspack = 1;
    atype.alignment = 0;

    switch (*s++) {
        case 'A': atype.layout = MatrixLayout::Pc; break;
        case 'B': atype.layout = MatrixLayout::Pr; break;
        case 'N': atype.layout = MatrixLayout::N;  break;
        case 'T': atype.layout = MatrixLayout::T;  break;
        case '?': atype.layout = MatrixLayout::N;  break; // Either N/T; used for fused GEMM.
        default:
            throw std::runtime_error("Unknown matrix layout requested.");
    }

    if (isdigit(*s))
        atype.crosspack = (uint8_t)strtol(s, (char **) &s, 10);
    if (*s == '#') {
        s++;
        atype.tileR = (uint16_t)strtol(s, (char **) &s, 10);
        if (*s != ',') throw std::runtime_error("Bad tiling syntax; expected #<rows>,<columns>.");
        s++;
        atype.tileC = (uint16_t)strtol(s, (char **) &s, 10);
    }
    if (*s == '%') {
        s++;
        atype.packSize = (uint32_t)strtol(s, (char **) &s, 10);
    }
    if (*s == '@') {
        s++;
        atype.alignment = (uint8_t)strtol(s, (char **) &s, 10);
    }

    return s;
}

const char *parsePrecision(const char *s, Type &precision)
{
    if (*s) {
        precision = charPrecision(*s++);
    }
    return s;
}

const char *parsePrecisions(const char *s, Type &precision1, Type &precision2)
{
    if (*s == '[') {
        s++;
        s = parsePrecision(s, precision1);
        s = parsePrecision(s, precision2);
        if (*s++ != ']')
            throw std::runtime_error("Syntax error in precisions; expected ]");
    } else {
        s = parsePrecision(s, precision1);
        precision2 = precision1;
    }

    return s;
}

std::string unparseStrategy(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy)
{
    std::stringstream s;
    s.imbue(std::locale::classic());

    bool anyOptAlignAB = strategy.optAlignAB || strategy.optAlignAB2D;

    unparseAddressBase(s, strategy.A.base);
    unparseAccessType(s, strategy.A);
    if (anyOptAlignAB && strategy.unalignedAccA != strategy.A.accessType) {
        auto unalignedA = strategy.A;
        unalignedA.accessType = strategy.unalignedAccA;
        s << '/';
        unparseAccessType(s, unalignedA);
    }
    s << strategy.ka_load;
    if (strategy.ka_load_masked != strategy.ka_load)
        s << '/' << strategy.ka_load_masked;
    unparseTiling(s, strategy.A);
    if (strategy.A_copies > 1)
        s << 'x' << strategy.A_copies;
    unparseCaching(hw, s, strategy.A);
    if (strategy.prefetchA) {
        s << '+';
        unparseAccessType(s, strategy.A_prefetch);
        s << strategy.ka_prefetch << ',' << strategy.ka_pfStride << '@' << strategy.prefetchA;
        if (strategy.prefetchAMasked != strategy.prefetchA)
            s << '/' << strategy.prefetchAMasked;
        unparseCaching(hw, s, strategy.A_prefetch);
    }
    if (strategy.l3PrefetchA) {
        if (!strategy.prefetchA) s << '+';
        s << '+';
        unparseAccessType(s, strategy.AB_prefetchL3);
        s << strategy.ka_prefetchL3 << '@' << strategy.prefetchABL3;
        unparseCaching(hw, s, strategy.AB_prefetchL3);
    }
    s << ' ';
    unparseAddressBase(s, strategy.B.base);
    unparseAccessType(s, strategy.B);
    if (anyOptAlignAB && strategy.unalignedAccB != strategy.B.accessType) {
        auto unalignedB = strategy.B;
        unalignedB.accessType = strategy.unalignedAccB;
        s << '/';
        unparseAccessType(s, unalignedB);
    }
    s << strategy.kb_load;
    if (strategy.kb_load_masked != strategy.kb_load)
        s << '/' << strategy.kb_load_masked;
    unparseTiling(s, strategy.B);
    if (strategy.B_copies > 1)
        s << 'x' << strategy.B_copies;
    unparseCaching(hw, s, strategy.B);
    if (strategy.prefetchB) {
        s << '+';
        unparseAccessType(s, strategy.B_prefetch);
        s << strategy.kb_prefetch << ',' << strategy.kb_pfStride << '@' << strategy.prefetchB;
        if (strategy.prefetchBMasked != strategy.prefetchB)
            s << '/' << strategy.prefetchBMasked;
        unparseCaching(hw, s, strategy.B_prefetch);
    }
    if (strategy.l3PrefetchB) {
        if (!strategy.prefetchB) s << '+';
        s << '+';
        unparseAccessType(s, strategy.AB_prefetchL3);
        s << strategy.kb_prefetchL3 << '@' << strategy.prefetchABL3;
        unparseCaching(hw, s, strategy.AB_prefetchL3);
    }
    s << ' ';
    unparseAddressBase(s, strategy.C.base);
    unparseAccessType(s, strategy.C);
    unparseTiling(s, strategy.C);
    unparseCaching(hw, s, strategy.C);
    if (strategy.prefetchC) {
        s << '+';
        unparseAccessType(s, strategy.C_prefetch);
        s << '@' << strategy.prefetchC;
        unparseCaching(hw, s, strategy.C_prefetch);
    }

    if (strategy.slmA || strategy.slmB) {
        s << " c";
        if (strategy.slmA) s << 'a';
        if (strategy.slmB) s << 'b';
        s << std::max(strategy.slmBuffersA, strategy.slmBuffersB);
        if (strategy.slmCopies > 1)
            s << 'x' << strategy.slmCopies;

        if (strategy.unrollKSLM != 0) {
            s << " ks" << strategy.unrollKSLM;
            if (strategy.unrollKSLMMasked > 0 && strategy.unrollKSLMMasked != strategy.unrollKSLM)
                s << '/' << strategy.unrollKSLMMasked;
        }
    }

    if (strategy.wg[LoopM] != 0) {
        s << " wg " << strategy.wg[LoopM] << 'x' << strategy.wg[LoopN];
        if (strategy.wg[LoopK] > 1)
            s << 'x' << strategy.wg[LoopK];
    }

    if (strategy.wgPadFactor > 1)           s << " wx" << strategy.wgPadFactor;
    if (strategy.forceWGUpdate == WGFixed)  s << " ff";
    if (strategy.forceFixedWGK)             s << " ffk";


    if (strategy.kParallelLocal)    s << (strategy.shrinkWGK   ? " akr" :
                                          strategy.kInterleave ? " ikr" :
                                                                 " kr");
    if (strategy.fillGoal)          s << " fg " << strategy.fillGoal * (1. / 16);
    if (strategy.kInterleave)       s << " ki" << strategy.kInterleaveChunk;

    if (strategy.fixedSystolic)
        s << " fs";
    else if (strategy.systolic) {
        s << " sys";
        if (strategy.dpasw)
            s << " dw";
    }

    if (strategy.isDSLGenerator) s << " dsl";

    if (strategy.dotVL) {
        s << " dot";
        if (strategy.dotVL > 1) s << strategy.dotVL;
    }

    if (strategy.kChain > 0)                s << " kc" << strategy.kChain;

    if (strategy.atomicFMA)                 s << (strategy.extendedAtomicFMA ? " xaf" : " af");
    if (strategy.stallAfterLoad)            s << " st";
    if (strategy.fmaBoustrophedon)          s << " fx";

    s << " k" << strategy.unroll[LoopK];
    if (strategy.unrollK_masked > 0) s << '/' << strategy.unrollK_masked;
    if (strategy.extraKAlign > 1) s << " ka" << strategy.extraKAlign;

    if (strategy.GRFs != 128)
        s << " grf" << strategy.GRFs;
    if (strategy.raHW != hw)
        if (auto ra = unparseRAHW(strategy.raHW))
            s << " ra" << ra;
    if (strategy.ka_repack) s << " ra" << strategy.ka_repack;
    if (strategy.kb_repack) s << " rb" << strategy.kb_repack;

    if (strategy.coopA == CoopSplit::MN)    s << " sm";
    if (strategy.coopA == CoopSplit::FullK) s << " ska";
    if (strategy.coopB == CoopSplit::MN)    s << " sn";
    if (strategy.coopB == CoopSplit::FullK) s << " skb";

    switch (strategy.registerScheme) {
        case GEMMStrategy::CSeparate:       s << " cs";  break;
        case GEMMStrategy::ACB:             s << " acb"; break;
        case GEMMStrategy::BCA:             s << " bca"; break;
        case GEMMStrategy::VNC:             s << " vnc"; break;
        case GEMMStrategy::ABInterleave:    s << " int"; break;
        case GEMMStrategy::NSeparate:       s << " nse"; break;
        case GEMMStrategy::VAvoid:          s << " vav"; break;
    }

    if (strategy.cAccumulators)             s << " ac";
    if (strategy.cLoadAhead)                s << " el";
    if (strategy.delayABInc)                s << " di";
    if (strategy.pfaux)                     s << " pfaux";
    if (strategy.loadBFirst)                s << " ba";
    if (strategy.doubleMasking)             s << " dm";
    if (strategy.kDescRem)                  s << " kd";
    if (strategy.splitCopy)                 s << " sc";
    if (!strategy.slmUseIncrCopy)           s << " ni";
    if (strategy.slmEarlyKMask)             s << " ek";
    if (strategy.strictFence)               s << " sf";
    if (strategy.slmATrans)                 s << " ta";
    if (strategy.slmBTrans)                 s << " tb";
    if (strategy.chainPrefetches)           s << " cp";
    if (!strategy.altCRemainder)            s << " sr";
    else if (!strategy.block2DCRemainder)   s << " ar";
    if (strategy.block2DCFull)              s << " bf";
    else if (strategy.block2DCRemainder)    s << " br";
    if (strategy.panelCheck)                s << " up";
    if (strategy.reverse[LoopM])            s << " rm";
    if (strategy.reverse[LoopN])            s << " rn";
    if (strategy.scramble[LoopM])           s << " ym";
    if (strategy.scramble[LoopN])           s << " yn";
    if (strategy.tlbWarmup)                 s << " wt";
    if (strategy.tokenAlloc)                s << " tk";

    if (strategy.checkAdd32 && !strategy.emulate.emulate64) s << " ch";
    if (!strategy.checkAdd32 && strategy.emulate.emulate64) s << " nch";
    if (strategy.wgInSS)                                    s << " ws";
    if (strategy.C.smode == ScatterSIMD::Wide)              s << " wc";
    if (strategy.forceCopyC)                                s << " cc";

    if (!strategy.jointSplit)           s << " njs";
    if (strategy.mSplitThresh)          s << " ms" << strategy.mSplitThresh;
    if (strategy.nSplitThresh)          s << " ns" << strategy.nSplitThresh;
    if (strategy.cInterleave)           s << " cti";
    if (strategy.kParallel)             s << " kb";
    if (strategy.kParallelVariable)     s << " kv";
    if (strategy.fuseBeta)              s << (strategy.altFusedBeta ? " afb" : " fb");
    if (strategy.fusePostOps)           s << " fp";
    if (strategy.zeroTempC)             s << " zt";
    if (strategy.relaxedAccumulation)   s << " rx";
    if (strategy.kPadding)              s << " pk" << strategy.kPadding;

    if (strategy.C.atomic && !strategy.kParallel && !strategy.kParallelVariable)
        s << " au";
    else if (!strategy.autoatomic)
        s << " nau";

    if (strategy.namedBarriers[LoopM] || strategy.namedBarriers[LoopN])
        s << " nb " << strategy.namedBarriers[LoopM] << 'x' << strategy.namedBarriers[LoopN];

    switch (strategy.cWalkOrder) {
        case WalkOrder::Hilbertlike:      s << " hi"; break;
        case WalkOrder::Boustrophedon:    s << " bo"; break;
        case WalkOrder::SimpleLinear:     s << " li"; break;
        case WalkOrder::NestedLinear:     s << " nl"; break;
        default: break;
    }

    if (strategy.persistent)
        s << " pt";

    if (strategy.loopOrder[0] != LoopM || strategy.loopOrder[1] != LoopN || strategy.loopOrder[2] != LoopK) {
        s << ' ';
        for (int l = 0; l < 3; l++)
            s << "mnk"[strategy.loopOrder[l]];
    }

    if (strategy.fused) s << " f" << "mnk"[strategy.fusedLoop];

    switch (strategy.arbitrationMode) {
        case ThreadArbitrationMode::OldestFirst:            s << " of"; break;
        case ThreadArbitrationMode::RoundRobin:             s << " rr"; break;
        case ThreadArbitrationMode::RoundRobinOnStall:      s << " rrs"; break;
        default: break;
    }

    if (strategy.optAlignAB > 0)    s << " l" << strategy.optAlignAB;
    if (strategy.optAlignAB2D)      s << " l2d";

    bool nq = false;
    for (auto &astrategy: {strategy.A, strategy.B, strategy.C, strategy.CO,
                           strategy.A_prefetch, strategy.B_prefetch, strategy.C_prefetch}) {
        nq |= astrategy.noExtraPad;
    }
    if (nq) s << " nq";

    if (strategy.barrierFreq > 0) {
        s << (strategy.splitBarrier ? " sb" : " b")
          << strategy.barrierFreq;
    }

    if (strategy.cRepackPanel)      s << " cr" << strategy.cRepackPanel;

    if (strategy.skewLocalIDs) {
        s << " ql";
    }

    bool npA = !isPacked(problem.A.layout) && !strategy.A.padded;
    bool npB = !isPacked(problem.B.layout) && !strategy.B.padded;
    bool pabA = !isPacked(problem.A.layout) && strategy.A.padded;
    bool pabB = !isPacked(problem.B.layout) && strategy.B.padded;
    bool pc = !isPacked(problem.C.layout) && strategy.C.padded;

    if (pabA || pabB)   s << " pab";
    if (npA || npB)     s << " np";
    if (pc)             s << " pc";

    for (auto loop: {LoopM, LoopN, LoopK}) if (strategy.blocking[loop] != 0) {
        s << " b" << "mnk"[loop] << strategy.blocking[loop];
        bool checkAlt = (loop == LoopK) ? (hw == HW::XeHP)
                                        : (strategy.cWalkOrder == WalkOrder::Hilbertlike);
        if (checkAlt && (strategy.blockingAlt[loop] != 0))
            s << '/' << strategy.blockingAlt[loop];
    }

    return s.str();
}


void unparseAccessType(std::ostream &s, const MatrixAddressingStrategy &astrategy)
{
    char c = ' ';
    switch (astrategy.accessType) {
        case AccessType::Block:             c = 'b'; break;
        case AccessType::PseudoBlock:       c = 'p'; break;
        case AccessType::Scattered:         c = 's'; break;
        case AccessType::ChannelScattered:  c = 'u'; break;
        case AccessType::Block2D:           c = 'm'; break;
        case AccessType::Block2DTranspose:  c = 't'; break;
        case AccessType::Block2DVNNI:       c = 'v'; break;
        case AccessType::CacheLine:         c = 'c'; break;
    }
    if (astrategy.newDP)
        c = std::toupper(c);
    s << c;
}

void unparseAddressBase(std::ostream &s, ngen::AddressBase base)
{
    switch (base.getModel()) {
        case ngen::AddressModel::ModelA64:     s << 'a'; break;
        case ngen::AddressModel::ModelCC:      s << 'c'; break;
        case ngen::AddressModel::ModelSLM:     s << 'l'; break;
        case ngen::AddressModel::ModelSC:      s << 'm'; break;
        case ngen::AddressModel::ModelBTS:     s << 's'; break;
        case ngen::AddressModel::ModelInvalid: s << 'r'; break;
        default: s << 'u'; break;
    }
}

void unparseCaching(HW hw, std::ostream &s, const MatrixAddressingStrategy &astrategy)
{
    auto cachingR = astrategy.cachingR;
    auto cachingW = astrategy.cachingW;

    if (!astrategy.newDP)
        return;
    if (cachingR == CacheSettingsLSC::Default && cachingW == CacheSettingsLSC::Default)
        return;

    s << '{';

    if (hw >= HW::Xe3p) {
        switch (cachingR) {
            case CacheSettingsLSC::Default:             s << "ddd"; break;
            case CacheSettingsLSC::L1UC_L2UC_L3UC:      s << "uuu"; break;
            case CacheSettingsLSC::L1UC_L2UC_L3C:       s << "uuc"; break;
            case CacheSettingsLSC::L1UC_L2C_L3UC:       s << "ucu"; break;
            case CacheSettingsLSC::L1UC_L2C_L3C:        s << "ucc"; break;
            case CacheSettingsLSC::L1C_L2UC_L3UC:       s << "cuu"; break;
            case CacheSettingsLSC::L1C_L2UC_L3C:        s << "cuc"; break;
            case CacheSettingsLSC::L1C_L2C_L3UC:        s << "ccu"; break;
            case CacheSettingsLSC::L1C_L2C_L3C:         s << "ccc"; break;
            case CacheSettingsLSC::L1S_L2UC_L3UC:       s << "suu"; break;
            case CacheSettingsLSC::L1S_L2UC_L3C:        s << "suc"; break;
            case CacheSettingsLSC::L1S_L2C_L3UC:        s << "scu"; break;
            case CacheSettingsLSC::L1S_L2C_L3C:         s << "scc"; break;
            case CacheSettingsLSC::L1IAR_L2IAR_L3IAR:   s << "iii"; break;
            default:                                    s << "???"; break;
        }
        if (cachingW != CacheSettingsLSC::Default) {
            s << '/';
            switch (cachingW) {
                case CacheSettingsLSC::L1UC_L2UC_L3UC:  s << "uuu"; break;
                case CacheSettingsLSC::L1UC_L2UC_L3WB:  s << "uub"; break;
                case CacheSettingsLSC::L1UC_L2WB_L3UC:  s << "ubu"; break;
                case CacheSettingsLSC::L1WT_L2UC_L3UC:  s << "tuu"; break;
                case CacheSettingsLSC::L1WT_L2UC_L3WB:  s << "tub"; break;
                case CacheSettingsLSC::L1WT_L2WB_L3UC:  s << "tbu"; break;
                case CacheSettingsLSC::L1S_L2UC_L3UC:   s << "suu"; break;
                case CacheSettingsLSC::L1S_L2UC_L3WB:   s << "sub"; break;
                case CacheSettingsLSC::L1S_L2WB_L3UC:   s << "sbu"; break;
                case CacheSettingsLSC::L1WB_L2WB_L3UC:  s << "bbu"; break;
                case CacheSettingsLSC::L1WB_L2UC_L3WB:  s << "bub"; break;
                default:                                s << "???"; break;
            }
        }
    } else {
        switch (cachingR) {
            case CacheSettingsLSC::Default:   s << "dd"; break;
            case CacheSettingsLSC::L1UC_L3UC: s << "uu"; break;
            case CacheSettingsLSC::L1UC_L3C:  s << "uc"; break;
            case CacheSettingsLSC::L1C_L3UC:  s << "cu"; break;
            case CacheSettingsLSC::L1C_L3C:   s << "cc"; break;
            case CacheSettingsLSC::L1S_L3UC:  s << "su"; break;
            case CacheSettingsLSC::L1S_L3C:   s << "sc"; break;
            case CacheSettingsLSC::L1IAR_L3C: s << "ic"; break;
            case CacheSettingsLSC::L1C_L3CC:  s << "cn"; break;
            case CacheSettingsLSC::L1UC_L3CC: s << "un"; break;
            default:                          s << "??"; break;
        }
        if (cachingW != CacheSettingsLSC::Default) {
            s << '/';
            switch (cachingW) {
                case CacheSettingsLSC::L1UC_L3UC: s << "uu"; break;
                case CacheSettingsLSC::L1UC_L3C:  s << "ub"; break;
                case CacheSettingsLSC::L1C_L3UC:  s << "tu"; break;
                case CacheSettingsLSC::L1C_L3C:   s << "tb"; break;
                case CacheSettingsLSC::L1S_L3UC:  s << "su"; break;
                case CacheSettingsLSC::L1S_L3C:   s << "sb"; break;
                case CacheSettingsLSC::L1IAR_L3C: s << "bb"; break;
                default:                          s << "??"; break;
            }
        }
    }
    s << '}';
}

void unparseTiling(std::ostream &s, const MatrixAddressingStrategy &astrategy)
{
    if (astrategy.tileR || astrategy.tileC)
        s << '#' << int(astrategy.tileR) << ',' << int(astrategy.tileC);
}


GEMMSTONE_NAMESPACE_END
