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

#include "gemmstone/config.hpp"

#define BINARY_OUTPUT

#include "gemmstone/microkernel_selector.hpp"
#include "gemmstone/generator.hpp"
#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/strategy_parser.hpp"
#include "ngen_decoder.hpp"
#include "npack/neo_packager.hpp"
#include "pieces/compute_utils.hpp"
#include "pieces/hw_utils.hpp"

#ifdef STANDALONE
// oneDNN ISA registration macros, enabling all ISAs.
#define REG_XEHPG_ISA(...) __VA_ARGS__
#define REG_XEHPC_ISA(...) __VA_ARGS__
#define REG_XE2_ISA(...) __VA_ARGS__
#define REG_XE3_ISA(...) __VA_ARGS__
#define REG_XE3P_ISA(...) __VA_ARGS__
#endif

GEMMSTONE_NAMESPACE_START
namespace microkernel {

#define _CATALOG_ CatalogMMR
#include "selector/db/ukernel_mmr.db"
;
#undef _CATALOG_

#define _CATALOG_ CatalogLMR
#include "selector/db/ukernel_lmr.db"
;
#undef _CATALOG_

#define _CATALOG_ CatalogMLR
#include "selector/db/ukernel_mlr.db"
;
#undef _CATALOG_

using namespace ngen;

static inline bool getStrategyByHeuristics(HW hw, GEMMStrategy &strategy, bool localA, bool localB,
                                           GEMMProblem &problem, HWInformation hwInfo, SizeParams sizes,
                                           const std::vector<StrategyRequirement> &reqs);

static constexpr int smallGRF = 128, largeGRF = 256;

// Choose a microkernel's GRF mode from an estimate of its register footprint.
// Returns true if small GRF was selected, i.e. a large-GRF retry would differ.
static inline bool chooseMicrokernelGRFMode(HW hw, const GEMMProblem &problem, GEMMStrategy &s,
                                            int hostLiveBytes)
{
    /* Mode governs the host kernel too; reserve measured only on 32-byte GRFs. */
    int hostReserve = (hw < HW::XeHPC)
            ? GRF::bytesToGRFs(hw, std::max(hostLiveBytes, 0))
            : 0;

    int aElems = s.unroll[LoopM] * s.ka_load;
    int bElems = s.unroll[LoopN] * s.kb_load;
    int aRegs = GRF::bytesToGRFs(hw, aElems * std::max(s.A_copies, 1) * problem.Ta_ext);
    int bRegs = GRF::bytesToGRFs(hw, bElems * std::max(s.B_copies, 1) * problem.Tb_ext);

    /* Upconverted A/B keep a second, wider buffer alongside the loaded one. */
    if (problem.Ta_ext != problem.Ta) aRegs += GRF::bytesToGRFs(hw, aElems * problem.Ta);
    if (problem.Tb_ext != problem.Tb) bRegs += GRF::bytesToGRFs(hw, bElems * problem.Tb);

    int cRegs = GRF::bytesToGRFs(hw, s.unroll[LoopM] * s.unroll[LoopN] * problem.Tc_compute());

    bool useSmallGRF = (aRegs + bRegs + cRegs) <= (smallGRF - hostReserve);
    s.GRFs = useSmallGRF ? smallGRF : largeGRF;
    return useSmallGRF;
}

std::vector<Protocol::Argument> arguments(const GEMMOptions &o) {
    auto In = Protocol::Argument::In;
    auto Out = Protocol::Argument::Out;

    auto LocalPointer = StructuredType::LocalPointer;
    auto GlobalPointer = StructuredType::GlobalPointer;
    auto s32 = StructuredType::s32;

    static Protocol::Argument args[] = {
            {"a", In, GlobalPointer},
            {"lda", In, s32},
            {"b", In, GlobalPointer},
            {"ldb", In, s32},
            {"c", Out, 2},
            {"m", In, s32},
            {"n", In, s32},
            {"k", In, s32},
            {"i0", In, s32},
            {"j0", In, s32},
            {"h0", In, s32},
            {"local_id_m", In, s32},
            {"local_id_n", In, s32},
    };
    std::vector<Protocol::Argument> argsV
            = {args, args + sizeof(args) / sizeof(args[0])};

    if (o.localA) argsV[0].stype.format = LocalPointer;
    if (o.localB) argsV[2].stype.format = LocalPointer;
    if (o.addToC) argsV[4].direction = Protocol::Argument::InOut;
    if (o.kParallelLocal) argsV.push_back({"local_id_k", In, s32});
    if (o.slmPtr) argsV.push_back({"slm", In, LocalPointer});
    if (o.scaleA) argsV.push_back({"a_scale", In, GlobalPointer});
    if (o.offsetA) argsV.push_back({"a_offset", In, o.offsetAHostScalar ? StructuredType(StructuredType::s16) : StructuredType(GlobalPointer)});
    if (o.offsetA || o.scaleA) argsV.push_back({"ldaq", In, s32});
    if (o.scaleB) argsV.push_back({"b_scale", In, GlobalPointer});
    if (o.offsetB) argsV.push_back({"b_offset", In, o.offsetBHostScalar ? StructuredType(StructuredType::s16) : StructuredType(GlobalPointer)});
    if (o.offsetB || o.scaleB) { argsV.push_back({"ldbq", In, s32}); }

    return argsV;
}

std::vector<Protocol::Setting> settings() {
    static Protocol::Setting settings[] = {
            {"sg_tile_m"},
            {"sg_tile_n"},
            {"wg_tile_m"},
            {"wg_tile_n"},
            {"sg_per_wg_m"},
            {"sg_per_wg_n"},
            {"sg_per_wg_k"},
            {"slm_size"},
    };
    static std::vector<Protocol::Setting> settingsV
            = {settings, settings + sizeof(settings) / sizeof(settings[0])};
    return settingsV;
}

Protocol makeProtocol(const GEMMOptions &o) {
    return {"ugemm", arguments(o), settings()};
}

InterfaceHandler GEMMOptions::generateInterface(HW hw, HostPayload host) const {
    /* Set up arguments for microkernel */
    InterfaceHandler interface(hw);

    if (host.simd <= 0 || host.argumentBytes <= 0) {
        interface.setArgumentBase(ngen::GRF(8));
    } else {
        /* Place microkernel arguments above the host kernel's thread payload:
           r0, the local IDs, then the cross-thread arguments. */
        interface.requireLocalID(3);
        /* SIMD fixes the local-ID register stride, hence the cross-thread base. */
        interface.requireSIMD(host.simd);
        interface.setArgumentBase(GRF(interface.getCrossthreadBase().getBase()
                                      + GRF::bytesToGRFs(hw, host.argumentBytes)));
    }
    interface.newArgument("A", localA ? ExternalArgumentType::LocalPtr : ExternalArgumentType::GlobalPtr);
    interface.newArgument("lda", DataType::d);
    interface.newArgument("B", localB ? ExternalArgumentType::LocalPtr : ExternalArgumentType::GlobalPtr);
    interface.newArgument("ldb", DataType::d);
    interface.newArgument("m", DataType::d);
    interface.newArgument("n", DataType::d);
    interface.newArgument("k", DataType::d);
    interface.newArgument("i0", DataType::d);
    interface.newArgument("j0", DataType::d);
    interface.newArgument("h0", DataType::d);
    interface.newArgument("local_id_m", DataType::d);
    interface.newArgument("local_id_n", DataType::d);
    if (kParallelLocal)    interface.newArgument("local_id_k", DataType::d);
    if (slmPtr)            interface.newArgument("slm_base", ExternalArgumentType::LocalPtr);
    if (scaleA)            interface.newArgument("a_scale_ptr", ExternalArgumentType::GlobalPtr);
    if (offsetA) {
        if (offsetAHostScalar) interface.newArgument("ao", DataType::w);
        else                  interface.newArgument("ao_ptr", ExternalArgumentType::GlobalPtr);
    }
    if (scaleA || offsetA) interface.newArgument("ldaq", DataType::d);
    if (scaleB)            interface.newArgument("b_scale_ptr", ExternalArgumentType::GlobalPtr);
    if (offsetB) {
        if (offsetBHostScalar) interface.newArgument("bo", DataType::w);
        else                  interface.newArgument("bo_ptr", ExternalArgumentType::GlobalPtr);
    }
    if (scaleB || offsetB) interface.newArgument("ldbq", DataType::d);
    return interface;
}

GEMMOptions GEMMOptions::transpose() const {
    GEMMOptions ret = *this;
    std::swap(ret.localA, ret.localB);
    std::swap(ret.scaleA, ret.scaleB);
    std::swap(ret.offsetA, ret.offsetB);
    std::swap(ret.offsetAHostScalar, ret.offsetBHostScalar);
    return ret;
}

std::string strategyToString(HW hw, const GEMMProblem &problem, const GEMMStrategy &strategy) {
    std::stringstream ss;
    ss << problem.toString() << " "
       << std::to_string(strategy.unroll[LoopM])
       << " "
       << std::to_string(strategy.unroll[LoopN])
       << " "
       << problem.scalarsToString()
       << " "
       << unparseStrategy(hw, problem, strategy);
    return ss.str();
}

GEMMSelector::GEMMSelector(const GEMMOptions &options_, HostPayload host_, HWInformation hwInfo_,
                           SizeParams sizes_, const GEMMProblem &problem_,
                           const std::vector<StrategyRequirement> &reqs_,
                           StrategyAdjuster strategyAdjuster_)
    : options(options_), host(host_), hwInfo(hwInfo_), sizes(sizes_), baseProblem(problem_),
      reqs(reqs_), strategyAdjuster(std::move(strategyAdjuster_))
{
    auto &problem = baseProblem;

    if (!isColMajor(problem.C.layout))
        stub("Microkernel selection requires column-major C");

    problem.alpha = 1;
    problem.beta = options.addToC ? 1 : 0;

    problem.C.setAlignment(4);

    if (options.scaleA != problem.aScale2D() || options.scaleB != problem.bScale2D())
        stub("Protocol scales do not match problem description");
    if (options.offsetA != (problem.hasAOffsetPtr() || problem.aOffsetHostScalar())
            || options.offsetB != (problem.hasBOffsetPtr() || problem.bOffsetHostScalar())
            || options.offsetAHostScalar != problem.aOffsetHostScalar()
            || options.offsetBHostScalar != problem.bOffsetHostScalar())
        stub("Protocol offsets do not match problem description");

    /* Get hardware information */
    auto product = npack::decodeHWIPVersion(hwInfo.gmdid);
    hw = getCore(product.family);
    stepping = hwInfo.gmdid & 0xFF;

    problem.product = product;
    problem.autoTypeConversions(hwInfo.systolicAvailable);
    /* Strip internal upconversions */
    auto problemMatch = problem;
    if (problemMatch.Ta_ext.bits() < problemMatch.Ta.bits()) problemMatch.Ta = problemMatch.Ta_ext;
    if (problemMatch.Tb_ext.bits() < problemMatch.Tb.bits()) problemMatch.Tb = problemMatch.Tb_ext;

    /* Create catalog matcher */
    MatchParams matchParams(hw, hwInfo.systolicAvailable, product, problemMatch);

    matchParams.sizes = sizes;
    matchParams.stepping = stepping;
    matchParams.nExtraReqs = int(reqs.size());
    matchParams.extraReqs = reqs.data();

    if (hw == ngen::HW::Xe3p && !hwInfo.isEfficient64Bit)
        matchParams.selector.hw = kcatalog::HWTagXeHPC;

    auto tags = const_cast<char *>(matchParams.tags);
    while (*tags)
        tags++;

    /* Xe2 requires stronger alignment for block 2D. */
    bool can2DA = true, can2DB = true;
    if (hw == HW::Xe2 || hw == HW::Xe3) {
        can2DA &= (problem.A.alignment % 16 == 0);
        can2DB &= (problem.B.alignment % 16 == 0);
    }

    if (can2DA) *tags++ = kcatalog::ReqBlock2DA;
    if (can2DB) *tags++ = kcatalog::ReqBlock2DB;

    /* Provide information for kernel selection */
    EvaluateParams evalParams;
    evalParams.sizes = matchParams.sizes;
    evalParams.alpha = 1;
    evalParams.beta = 0;
    evalParams.product = product;
    evalParams.euCount = hwInfo.euCount;

    kcatalog::Catalog catalog = [&]() {
        if (options.localA)
            return kcatalog::Catalog(CatalogLMR);
        else if (options.localB)
            return kcatalog::Catalog(CatalogMLR);
        else
            return kcatalog::Catalog(CatalogMMR);
    }();

    /* Call kernel selector */
    entries = select(catalog, 1, &matchParams, evalParams, auxParams);
    auto last_entry = std::remove_if(begin(entries), end(entries), [&](const kcatalog::Entry* e) {
        GEMMStrategy strategy(hw, stepping);
        strategy.unroll[LoopM] = e->driverInfo.unroll[LoopM];
        strategy.unroll[LoopN] = e->driverInfo.unroll[LoopN];
        parseStrategy(e->strategy, problem, strategy);
        return (!options.kParallelLocal && strategy.kParallelLocal) ||
            // named barriers are not supported by generateShim
            (strategy.namedBarriers[LoopM] > 0 ||
            strategy.namedBarriers[LoopN] > 0);
    });
    entries.erase(last_entry, end(entries));
    if(!reqs.empty())
        entries.push_back(nullptr); // Try heuristics if no kernel found
    if (getVerbose(gemmstone::GEMMVerbose::DebugInfo) >= 4) {
        auto problemVerbose = problemMatch;
        for(const kcatalog::Entry *e : entries) {
            if(e) {
                GEMMStrategy strategy(hw, stepping);
                strategy.unroll[LoopM] = e->driverInfo.unroll[LoopM];
                strategy.unroll[LoopN] = e->driverInfo.unroll[LoopN];
                problemVerbose.A.setAlignment(e->driverInfo.alignment[0]);
                problemVerbose.B.setAlignment(e->driverInfo.alignment[1]);
                parseStrategy(e->strategy, problemVerbose, strategy);
                std::cout << "entry candidate: "
                          << gemmstone::evaluate(*e, evalParams, auxParams) << " "
                          << e->selector.hw << " "
                          << strategyToString(hw, problemVerbose, strategy) << std::endl;
            } else {
                if(!reqs.empty())
                    std::cout << "entry candidate: heuristics\n";
            }
        }
    }
}

bool GEMMSelector::next(GEMMCandidate &candidate)
{
    /* Large GRF variant of the last entry, whose small GRF variant was returned before. */
    if (pendingLargeGRF) {
        pendingLargeGRF = false;
        candidate = std::move(largeGRFCandidate);
        if (finish(largeGRFEntry, candidate))
            return true;
    }

    while (nextEntry < entries.size()) {
        const kcatalog::Entry *entry = entries[nextEntry++];
        GEMMCandidate base;
        if (!prepare(entry, base))
            continue;

        /* Size the GRF mode for inlining into the host kernel, for both catalog and heuristic
           strategies. If the estimate selected the small mode, the large mode follows as the
           next candidate in case it was low and generation runs out of registers. */
        GEMMCandidate small = base;
        if (chooseMicrokernelGRFMode(hw, small.problem, small.strategy, host.liveBytes)
                && finish(entry, small)) {
            largeGRFCandidate = std::move(base);
            largeGRFCandidate.strategy.GRFs = largeGRF;
            largeGRFEntry = entry;
            pendingLargeGRF = true;
            candidate = std::move(small);
            return true;
        }

        candidate = std::move(base);
        candidate.strategy.GRFs = largeGRF;
        if (finish(entry, candidate))
            return true;
    }

    return false;
}

/* Build a candidate's strategy from a catalog entry, or from heuristics if there is no entry. */
bool GEMMSelector::prepare(const kcatalog::Entry *entry, GEMMCandidate &candidate) const
{
    auto &problem = candidate.problem;
    auto &strategy = candidate.strategy;

    problem = baseProblem;
    strategy = GEMMStrategy(hw, stepping);

    if (entry) {
        problem.A.setAlignment(std::max(problem.Ta.paddedSize(), entry->driverInfo.alignment[0]));
        problem.B.setAlignment(std::max(problem.Tb.paddedSize(), entry->driverInfo.alignment[1]));

        /* Prepare strategy parameters */
        strategy.unroll[LoopM] = entry->driverInfo.unroll[LoopM];
        strategy.unroll[LoopN] = entry->driverInfo.unroll[LoopN];
        parseStrategy(entry->strategy, problem, strategy);
        adjustStrategy(hw, problem, strategy);
        modifyStrategy(strategy, auxParams);

        /* Xe2-XeHPC compatibility logic */
        if (hw == ngen::HW::Xe2 || hw == ngen::HW::Xe3) {
            // Use XeHPC register banking on Xe2/Xe3, in order
            //   to successfully reuse XeHPC strategies.
            strategy.raHW = ngen::HW::XeHPC;

            // Bump up alignments to 16 bytes for block 2D if available.
            bool block2DA = false, block2DB = false;
            for (auto c = entry->restrictions.tags; *c; c++) {
                block2DA |= (*c == kcatalog::ReqBlock2DA);
                block2DB |= (*c == kcatalog::ReqBlock2DB);
            }
            if (block2DA && strategy.legalAAlignment(problem, 16))
                problem.A.setAlignment(std::max<int>(problem.A.alignment, 16));
            if (block2DB && strategy.legalBAlignment(problem, 16))
                problem.B.setAlignment(std::max<int>(problem.B.alignment, 16));
        }
        if (hw == ngen::HW::Xe3p && !hwInfo.isEfficient64Bit) {
            // Use XeHPC banking if reusing XeHPC strategies (legacy mode)
            strategy.raHW = ngen::HW::XeHPC;
        }
    } else if (!reqs.empty() &&
               !getStrategyByHeuristics(hw, strategy, options.localA, options.localB, problem, hwInfo,
                                        sizes, reqs))
        return false; /* No heuristic strategy found */

    strategy.systolicAvailable &= hwInfo.systolicAvailable;

    /* Disable strategies not related to microkernels */
    strategy.kParallel = strategy.kParallelVariable = strategy.persistent = false;
    strategy.cWalkOrder = WalkOrder::HW2D;

    /* Disable k-parallelization if the protocol does not allow it */
    if (!options.kParallelLocal) {
        strategy.kParallelLocal = strategy.kInterleave = false;
        strategy.wg[LoopK] = 1;
    }

    /* Adjust strategy for performance */
    if (strategy.barrierFreq > 0 && sizes.k < 4 * strategy.barrierFreq)
        strategy.barrierFreq = 0;

    /* Keep size down by only using checkAdd32 when really needed */
    strategy.checkAdd32 &= (hw != HW::XeHPC);

    /* C output in registers */
    strategy.C.base = AddressBase{};

    return true;
}

/* Run the caller's adjuster and preflight on a candidate sized for its GRF mode. */
bool GEMMSelector::finish(const kcatalog::Entry *entry, GEMMCandidate &candidate) const
{
    auto &problem = candidate.problem;
    auto &strategy = candidate.strategy;

    /* Allow caller to adjust strategy further */
    if (strategyAdjuster) strategyAdjuster(strategy);

    try {
        strategy.preflight(hw, problem);
    } catch (const std::runtime_error &ex) {
        if (getVerbose(gemmstone::GEMMVerbose::DebugInfo) >= 2) {
            std::cout << "preflight failed(" << ex.what() << "):"
                      << strategyToString(hw, problem, strategy) << std::endl;
        }
        return false;
    }

    /* Update problem from strategy */
    if (isPacked(problem.A.layout))
        problem.A.packSize = strategy.unroll[LoopM];
    if (isPacked(problem.B.layout))
        problem.B.packSize = strategy.unroll[LoopN];

    if (getVerbose(gemmstone::GEMMVerbose::DebugInfo) >= 2) {
        std::cout << "attempting " << (entry ? "db " : "heuristic ")
                  << "strategy: " << strategyToString(hw, problem, strategy) << std::endl;
    }

    return true;
}

/* Generate a candidate, mapping it back to the caller's operands when C was transposed for selection. */
static Package generatePackage(const GEMMOptions &options, HostPayload host, HWInformation hwInfo,
                               const GEMMCandidate &candidate, bool transC)
{
    auto product = npack::decodeHWIPVersion(hwInfo.gmdid);
    auto hw = getCore(product.family);
    auto stepping = hwInfo.gmdid & 0xFF;

    /* Generate interface */
    GEMMOptions effOptions = transC ? options.transpose() : options;
    InterfaceHandler interface = effOptions.generateInterface(hw, host);

    /* Generate microkernel */
    #define ARCH_DISPATCH(arch)                                                                         \
        case HW::arch: {                                                                                \
            Generator<HW::arch> generator(product);                                                     \
            generator.setStepping(stepping);                                                            \
            return generator.gemmMicrokernelPackage(candidate.problem, candidate.strategy, interface,   \
                                                    makeProtocol(options), hwInfo.gmdid, transC);       \
        }
    switch (hw) {
        REG_XEHPG_ISA(ARCH_DISPATCH(XeHPG))
        REG_XEHPC_ISA(ARCH_DISPATCH(XeHPC))
        REG_XE2_ISA(ARCH_DISPATCH(Xe2))
        REG_XE3_ISA(ARCH_DISPATCH(Xe3))
        REG_XE3P_ISA(ARCH_DISPATCH(Xe3p))
        default: throw std::runtime_error("Unsupported architecture");
    }
    #undef ARCH_DISPATCH
}

Package generateGEMM(const GEMMOptions &options, HostPayload host, HWInformation hwInfo,
                     const GEMMCandidate &candidate)
{
    return generatePackage(options, host, hwInfo, candidate, false);
}

Package selectGEMM(const GEMMOptions &options, HostPayload host, HWInformation hwInfo, SizeParams sizes,
                   const GEMMProblem &problem_, const std::vector<StrategyRequirement> &reqs_,
                   StrategyAdjuster strategyAdjuster, SelectionObserver *observer)
{
    /* Selection works on column-major C, so transpose the request otherwise. */
    bool transC = !isColMajor(problem_.C.layout);

    auto problem = problem_;
    auto reqs = reqs_;

    if (transC) {
        problem.transpose();
        std::swap(sizes.m, sizes.n);
        for (auto &req: reqs)
            req.transpose();
    }

    GEMMOptions effOptions = transC ? options.transpose() : options;
    GEMMSelector selector(effOptions, host, hwInfo, sizes, problem, reqs, strategyAdjuster);

    GEMMCandidate candidate;
    while (selector.next(candidate)) {
        try {
            return generatePackage(options, host, hwInfo, candidate, transC);
        } catch (const std::runtime_error &ex) {
            /* Try next strategy */
            if (getVerbose(gemmstone::GEMMVerbose::DebugInfo) >= 2) {
                auto hw = getCore(npack::decodeHWIPVersion(hwInfo.gmdid).family);
                std::cout << "strategy failed(" << ex.what() << "):"
                          << strategyToString(hw, candidate.problem, candidate.strategy) << std::endl;
            }
        }
    }
    throw std::runtime_error("No matching kernel");
}

Package selectGEMM(const GEMMOptions &options, HWInformation hwInfo, SizeParams sizes,
                   const GEMMProblem &problem, const std::vector<StrategyRequirement> &reqs,
                   StrategyAdjuster strategyAdjuster, SelectionObserver *observer)
{
    return selectGEMM(options, HostPayload(0, 0), hwInfo, sizes, problem, reqs,
                      strategyAdjuster, observer);
}

static inline bool getStrategyByHeuristics(HW hw, GEMMStrategy &strategy, bool localA, bool localB,
                                           GEMMProblem &problem, HWInformation hwInfo, SizeParams sizes,
                                           const std::vector<StrategyRequirement> &reqs)
{
    if (problem.C.layout == MatrixLayout::T) return false;

    int min2DAlignmentA = block2DMinAlignment(hw, problem.A, strategy.A, /* asIfBlock2D */ true);
    int min2DAlignmentB = block2DMinAlignment(hw, problem.B, strategy.B, /* asIfBlock2D */ true);

    auto &s = strategy;
    s.unroll[LoopK] = 1;
    s.wg[LoopK] = 1;
    s.unroll[LoopM] = s.unroll[LoopN] = 0;
    s.wg[LoopM] = s.wg[LoopN] = 0;

    for (auto &req: reqs) switch (req.param) {
        case StrategyRequirement::UnrollM: s.unroll[LoopM] = req.value; break;
        case StrategyRequirement::UnrollN: s.unroll[LoopN] = req.value; break;
        case StrategyRequirement::WGM:         s.wg[LoopM] = req.value; break;
        case StrategyRequirement::WGN:         s.wg[LoopN] = req.value; break;
        case StrategyRequirement::WGK:         s.wg[LoopK] = req.value; break;
        default: break;
    }

    if (s.wgTile(LoopM) * s.wgTile(LoopN) == 0)
        return false;

    bool systolic = hwInfo.systolicAvailable &&
                   (problem.Ta.paddedSize() <= 2 || problem.Ta == Type::tf32) &&
                   (problem.Tb.paddedSize() <= 2 || problem.Tb == Type::tf32);
    // Non-systolic integer dot products require byte operands. Keep the
    // external int4/2 format and let the generator unpack it for computation.
    if (!systolic) {
        if (problem.Ta.isSubByteInt())
            problem.Ta = problem.Ta.isSigned() ? Type::s8 : Type::u8;
        if (problem.Tb.isSubByteInt())
            problem.Tb = problem.Tb.isSigned() ? Type::s8 : Type::u8;
    }
    bool block2DA = (hw >= HW::XeHPC) && systolic && (problem.A.alignment % min2DAlignmentA) == 0;
    bool block2DB = (hw >= HW::XeHPC) && systolic && (problem.B.alignment % min2DAlignmentB) == 0;
    bool useNewDP = (hw >= HW::XeHP);

    // u3 is computed as u4, so size its loads as u4
    Type Ta_load = problem.Ta_ext.is3() ? Type(Type::u4) : problem.Ta_ext;

    s.ka_load = s.kb_load = 16;
    if (!systolic) {
        // Keep eight bytes per integer operand without enlarging the other
        // operand's load for mixed-precision inputs. Preserve floating-point defaults.
        s.ka_load = Ta_load.isInteger() ? 8 / Ta_load : 4;
        s.kb_load = problem.Tb_ext.isInteger() ? 8 / problem.Tb_ext : 4;
    }

    auto m_iter = std::max(1, s.unroll[LoopM] / elementsPerGRF<uint32_t>(hw));
    if (problem.A.layout == MatrixLayout::Pc) {
        s.A.accessType = AccessType::Block;
        s.A_copies = 2;
        s.A.padded = true;
    } else if (!block2DA) {
        s.A.accessType = AccessType::Block;
        if (systolic)
            s.ka_load = (problem.A.layout == MatrixLayout::T) ? 64 / Ta_load / m_iter : 16;
        s.slmA = (hw >= HW::XeHP);
    } else if (problem.A.layout == MatrixLayout::T) {
        s.A.accessType = AccessType::Block2DTranspose;
        s.ka_load = 64 / Ta_load / m_iter;
    } else if (problem.A.layout == MatrixLayout::N) {
        s.A.accessType = AccessType::Block2DVNNI;
        s.ka_load =  s.unroll[LoopM] / m_iter;
        s.A_copies = 2;
    }

    if (problem.B.layout == MatrixLayout::Pr) {
        s.B.accessType = AccessType::Block;
        s.B.padded = true;
        s.B_copies = 2;
    } else if (!block2DB) {
        s.B.accessType = AccessType::Block;
        if (systolic) {
            s.doubleMasking = true;
            s.kb_load = (problem.B.layout == MatrixLayout::N) ? 32 : 16;
        }
        s.slmB = (hw >= HW::XeHP);
    } else if (problem.B.layout == MatrixLayout::T)
        s.B.accessType = AccessType::Block2DTranspose;
    else if (problem.B.layout == MatrixLayout::N) {
        s.B.accessType = AccessType::Block2D;
        s.kb_load = 32;
    }

    s.C.accessType = AccessType::Block;

    bool slmDequantize2DA = (problem.aOffset2D() || problem.aScale2D()) && s.slmA;
    bool slmDequantize2DB = (problem.bOffset2D() || problem.bScale2D()) && s.slmB;
    if (slmDequantize2DA && slmDequantize2DB) {
        // A and B quantization parameters share one SLM load schedule, so each
        // thread's SLM copy must cover the same k range for both. A k split
        // divides unrollKSLM by the workgroup size, so split both in m/n.
        if (s.unroll[LoopM] % s.wg[LoopN] || s.unroll[LoopN] % s.wg[LoopM])
            return false;
        s.ka_load = s.kb_load = std::min(64, std::max(s.ka_load, s.kb_load));
        s.coopA = s.coopB = CoopSplit::MN;
    }

    s.A.base = localA ? AddressBase::createSLM() : AddressBase::createA64(true);
    s.B.base = localB ? AddressBase::createSLM() : AddressBase::createA64(true);
    s.A.newDP = s.B.newDP = useNewDP;
    s.A.cachingR = s.B.cachingR = CacheSettingsLSC::L1C_L3C;

    s.A_prefetch = s.A;
    s.B_prefetch = s.B;
    s.A_prefetch.prefetch = s.B_prefetch.prefetch = true;

    s.AO.newDP = s.A_scale.newDP = useNewDP;
    s.BO.newDP = s.B_scale.newDP = useNewDP;

    if (!localA && block2DA) {
        if (!isPacked(problem.A.layout))
            s.A_prefetch.accessType = AccessType::Block2D;
        s.prefetchA = s.prefetchAMasked = 2 * s.ka_load;
        s.ka_pfStride = s.ka_prefetch = s.ka_load;
    }

    if (!localB && block2DB) {
        if (!isPacked(problem.B.layout))
            s.B_prefetch.accessType = AccessType::Block2D;
        s.prefetchB = s.prefetchBMasked = 2 * s.kb_load;
        s.kb_pfStride = s.kb_prefetch = s.kb_load;
    }

    if(block2DA && !localA) {
        problem.A.alignment = std::min(problem.A.alignment,
                                        static_cast<uint8_t>(block2DMinAlignment(hw, problem.A, strategy.A)));
    } else {
        problem.A.setAlignment(std::min<uint8_t>(problem.A.alignment, s.unroll[LoopM] * problem.Ta));
    }
    if(block2DB && !localB) {
        problem.B.alignment = std::min(problem.B.alignment,
                                        static_cast<uint8_t>(block2DMinAlignment(hw, problem.B, strategy.B)));
    } else {
        problem.B.alignment = std::min<uint8_t>(16, problem.B.alignment);
    }

    s.systolic = systolic;
    if (systolic && hw >= HW::XeHPC) {
        s.extendedAtomicFMA = s.atomicFMA = true;
        if (hw >= HW::Xe3p && problem.product.family >= ngen::ProductFamily::CRI) {
            s.kChain = 2;
        }
    }
    s.registerScheme = GEMMStrategy::VAvoid;

    /* GRF mode is chosen by the caller via chooseMicrokernelGRFMode(). */
    if (localA && !localB)
        s.loadBFirst = true;

    if (s.slmA || s.slmB) {
        s.slmBuffersA = s.slmBuffersB = 1;
        s.unrollKSLM = std::max(int(s.slmA) * s.ka_load, int(s.slmB) * s.kb_load);

        // Remainder handling can change an MN copy split to a K split. Keep
        // complete integer crosspacks in each thread's SLM copy in either case.
        // Use compute types: packed int4 inputs may have been upconverted.
        auto slmCP = targetSLMCrosspack(problem, s);
        if (s.slmA && problem.Ta.isInteger())
            s.unrollKSLM = lcm(s.unrollKSLM,
                    std::get<0>(slmCP) * s.wg[LoopN]);
        if (s.slmB && problem.Tb.isInteger())
            s.unrollKSLM = lcm(s.unrollKSLM,
                    std::get<1>(slmCP) * s.wg[LoopM]);

        // Quantization parameters are loaded per unrollKSLM, or per group if
        // unrollKSLM is not a multiple of the group size.
        if (slmDequantize2DA && slmDequantize2DB) {
            auto kqLoad = [&](int groupK) {
                return (s.unrollKSLM % groupK == 0) ? s.unrollKSLM : groupK;
            };
            if (kqLoad(problem.aqGroupK) != kqLoad(problem.bqGroupK))
                return false;
        }
    }

    if (hw == HW::Xe2 || hw == HW::Xe3)
        s.raHW = HW::XeHPC;
    // Use XeHPC banking if reusing XeHPC strategies (legacy mode)
    if (hw == ngen::HW::Xe3p && !hwInfo.isEfficient64Bit) {
        s.raHW = ngen::HW::XeHPC;
    }

    adjustStrategy(hw, problem, strategy);

    return true;
}

}
GEMMSTONE_NAMESPACE_END
