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

#define ENABLE_MICROKERNELS

#include "device_utils.hpp"
#include "frontend.hpp"
#include "kernel_modifiers.hpp"
#include "recipes.hpp"
#include "gemmstone/generator.hpp"
#include "gemmstone/runtime.hpp"
#include "gemmstone/dsl/kernel.hpp"
#include "gemmstone/strategy_parser.hpp"
#include "generator/pieces/layout_utils.hpp"
#include "utils.hpp"

#include <cctype>
#ifdef BINARY_OUTPUT
#include <chrono>
#endif
#include <cstdlib>
#include <cstring>
#include <memory>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#else
#include <pwd.h>
#ifdef OPENCL_OUTPUT
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif
#endif

using namespace ngen;
using namespace gemmstone;

namespace frontend {
struct BLASInterface : public Generator {
    BLASInterface(ngen::Product product_) : product(product_) {}
    virtual ~BLASInterface() = default;
    virtual Result generate(const std::string &kernel, bool infoOnly = false) override;
    ngen::Product getProduct() const { return product; }

    virtual std::string getNotes() = 0;
    virtual void setStepping(int stepping_) = 0;
    virtual void gemmMicrokernel(GEMMProblem problem, GEMMStrategy strategy, const InterfaceHandler &interface_) = 0;
    virtual void gemm(GEMMProblem problem, GEMMStrategy strategy, const InterfaceHandler &interface_) = 0;
#ifdef ASM_OUTPUT
    virtual void enableLineNumbers(bool enable = true) = 0;
    virtual void disableAutoSWSB() = 0;
    virtual void getCode(std::ostream &out) = 0;
    virtual void getPartialCode(std::ostream &out) = 0;
#elif defined(BINARY_OUTPUT)
    virtual std::vector<uint8_t> getCode() = 0;
#elif defined(OPENCL_OUTPUT)
    virtual std::vector<uint8_t> getBinary(cl_context context, cl_device_id device) = 0;
    // ELFCodeGenerator interface
    virtual std::vector<uint8_t> getBinary() = 0;
#elif defined(ZEBIN_OUTPUT)
    virtual std::vector<uint8_t> getBinary() = 0;
#endif
protected:
    ngen::Product product;
};

template <ngen::HW hw>
class BLASImpl : public BLASInterface {
public:
    BLASImpl(ngen::Product product) : BLASInterface(product), impl(product) {
        impl.enableCancelAutoSWSB();
    };
    ~BLASImpl() override = default;
    void cancelAutoSWSB() override {
        return impl.cancelAutoSWSB();
    }
    std::string getNotes() override {
        if (gemm_desc) {
            std::ostringstream oss;
#ifdef ASM_OUTPUT
            oss << "// ";
#endif
            gemmstone::Generator<hw>::driverInfo(gemm_desc->problem, gemm_desc->strategy).dump(oss);
            oss << "\n";
            return oss.str();
        }
        return impl.getNotes();
    }
    void setStepping(int stepping) override {
        impl.setStepping(stepping);
    }
    void gemmMicrokernel(GEMMProblem problem, GEMMStrategy strategy, const InterfaceHandler &interface_) override {
        return impl.gemmMicrokernel(problem, strategy, interface_);
    }
    void gemm(GEMMProblem problem, GEMMStrategy strategy, const InterfaceHandler &interface_) override {
        if (strategy.isDSLGenerator) {
            auto ir_hw = dsl::hw_t(product.family);
            gemm_desc.reset(new GEMMKernelDesc(problem, strategy, interface_, ir_hw));
            return;
        };
        return impl.gemm(problem, strategy, interface_);
    }
#ifdef ASM_OUTPUT
    void enableLineNumbers(bool enable = true) override {
        impl.enableLineNumbers(enable);
    }
    void disableAutoSWSB() override {
#ifdef NGEN_ASM_HAS_DISABLE_AUTOSWSB
        impl.disableAutoSWSB();
#endif
    }
    void getCode(std::ostream &out) override {
        if(gemm_desc) { out << make_asm(*gemm_desc); return; }
        impl.getCode(out);
    }
    void getPartialCode(std::ostream &out) override {
        if(gemm_desc) stub();
        impl.getPartialCode(out);
    }
#elif defined(BINARY_OUTPUT)
    std::vector<uint8_t> getCode() override {
        if (gemm_desc) return make_binary(*gemm_desc);
        return impl.getCode();
    }
#elif defined(OPENCL_OUTPUT)
    std::vector<uint8_t> getBinary(cl_context context, cl_device_id device) override {
        if (gemm_desc) return make_binary(*gemm_desc, device, context);
        return impl.getBinary(context, device);
    }
    std::vector<uint8_t> getBinary() override {
        if (gemm_desc) return make_binary(*gemm_desc);
        return impl.ELFCodeGenerator<hw>::getBinary();
    }
#elif defined(ZEBIN_OUTPUT)
    std::vector<uint8_t> getBinary() override {
        if (gemm_desc) stub();
        return impl.getBinary();
    }
#endif
private:
    gemmstone::Generator<hw> impl;
    std::unique_ptr<GEMMKernelDesc> gemm_desc = nullptr;
};

bool IsHWSupported(ngen::HW hw)
{
    switch (hw) {
#ifdef FRONTEND_ENABLE_12LP
        case ngen::HW::Gen12LP: return true;
#endif
#ifdef FRONTEND_ENABLE_12HP
        case ngen::HW::XeHP: return true;
#endif
#ifdef FRONTEND_ENABLE_12p7
        case ngen::HW::XeHPG: return true;
#endif
#ifdef FRONTEND_ENABLE_12p8
        case ngen::HW::XeHPC: return true;
#endif
#ifdef FRONTEND_ENABLE_Xe2
        case ngen::HW::Xe2: return true;
#endif
#ifdef FRONTEND_ENABLE_Xe3
        case ngen::HW::Xe3: return true;
#endif
#ifdef FRONTEND_ENABLE_Xe3p
        case ngen::HW::Xe3p: return true;
#endif
        default: return false;
    }
}

std::unique_ptr<BLASInterface> GetGenerator(ngen::Product product)
{
#define CASE(arch) \
    case ngen::HW::arch: \
        return std::make_unique<BLASImpl<ngen::HW::arch>>(product);

    ngen::HW hw = getCore(product.family);
    switch (hw) {
#ifdef FRONTEND_ENABLE_12LP
        CASE(Gen12LP)
#endif
#ifdef FRONTEND_ENABLE_12HP
        CASE(XeHP)
#endif
#ifdef FRONTEND_ENABLE_12p7
        CASE(XeHPG)
#endif
#ifdef FRONTEND_ENABLE_12p8
        CASE(XeHPC)
#endif
#ifdef FRONTEND_ENABLE_Xe2
        CASE(Xe2)
#endif
#ifdef FRONTEND_ENABLE_Xe3
        CASE(Xe3)
#endif
#ifdef FRONTEND_ENABLE_Xe3p
        CASE(Xe3p)
#endif
        default: break;
    }
    throw std::runtime_error("Frontend not compiled for chosen hardware (check value of GPUS in CMake settings)");
#undef CASE
}

bool strchrs(const char *haystack, const char *needle)
{
    for (; *needle; needle++)
        if (!strchr(haystack, *needle))
            return false;
    return true;
}

static inline char layout_char(MatrixLayout layout) { return std::tolower(layoutChar(layout)); }
static inline char precision_char(Type T) { return std::tolower(precisionChar(T)); }

static inline const char *precision_str(Type T) {
    switch (T) {
        case Type::f16:  return "f16";
        case Type::f32:  return "f32";
        case Type::f64:  return "f64";
        case Type::u8:   return "u8";
        case Type::s8:   return "s8";
        case Type::s16:  return "s16";
        case Type::u32:  return "u32";
        case Type::s32:  return "s32";
        case Type::bf16: return "bf16";
        case Type::tf32: return "tf32";
        default:         return "?";
    }
}

struct Level3KernelArgs {
    Type Ta, Tb, Tc, Ts;
    Type Ta_internal, Tb_internal, Tc_internal;
    MatrixAddressing A, B, C, A_ext, B_ext;
    int unrollM[2], unrollN[2];
    Scalar alpha, beta;
    UpdateType updateType;
    ::utils::KernelModifiers modifiers;
    bool trsm;
    bool left;
    bool lower;
    bool unit, checkUnit;
    bool inplace;
    bool kParallel, kParallelLocal;
    char strategyTag = '\0';
    std::string strategy;
    std::string copyStrategies[2][2];
    GEMMProblem::MockSettings mock;
    GEMMStrategy::TrackingSettings tracking;
    bool micro;
    LoopType fusedLoop;

    bool multiM() const { return unrollM[1] != 0; }
    bool multiN() const { return unrollN[1] != 0; }
};

struct EnvConfig {
    EnvConfig() {
        auto mklInterface = getenv("MKL_INTERFACE");
        if (mklInterface) MKLInterface = atoi(mklInterface);

        auto hplInterface = getenv("HPL_INTERFACE");
        if (hplInterface) {
            HPLInterface = atoi(hplInterface);
            MKLInterface |= HPLInterface;
        }

        if (getenv("NONUNIFORM_WG"))
            UniformWG = false;
    }
    bool MKLInterface = false;
    bool HPLInterface = false;
    bool UniformWG = true;
};

bool fusedEUs(ngen::HW hw)
{
    return (hw == HW::Gen12LP) || (hw == HW::XeHP) || (hw == HW::XeHPG);
}

InterfaceHandler createInterface(ngen::HW hw)
{
    InterfaceHandler interface{hw};
#ifdef ZEBIN_OUTPUT
    interface.setInlineGRFCount(0);
#endif

    return interface;
}

bool recipeHWMatch(ngen::HW hwIn, ngen::HW hwRef)
{
    auto bucket = [](HW hw) -> int { return(hw >= HW::Gen12LP) + (hw >= HW::XeHP) + (hw >= HW::XeHPG) + (hw >= HW::XeHPC); };
    return (bucket(hwIn) == bucket(hwRef)) || (hwIn == HW::XeHPG && hwRef == HW::XeHP);
}

void strategize(ngen::Product product, const frontend::EnvConfig &fargs, CommonStrategy &strategy)
{
    ngen::HW hw = getCore(product.family);
    strategy.emulate = EmulationStrategy(hw, product.stepping);
    strategy.simulation = simulation();
    if (product.family == ngen::ProductFamily::MTL)
        strategy.systolicAvailable = false;
    if (getenv("SYSTOLIC"))
        strategy.systolicAvailable = atoi(getenv("SYSTOLIC"));
}

void strategize(ngen::Product product, Level3KernelArgs args, const frontend::EnvConfig &fargs, int kid0, int kid1, GEMMProblem &problem, GEMMStrategy &strategy, bool forceKRemHandling)
{
    ngen::HW hw = getCore(product.family);
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;

    auto ALayout = problem.A.layout;
    auto BLayout = problem.B.layout;
    auto CLayout = problem.C.layout;
    auto unrollM = args.unrollM[kid0];
    auto unrollN = args.unrollN[kid1];
    bool gotStrategy = false;
    bool trsm = false;

    auto *gemmA = &problem.A, *gemmB = &problem.B;
    auto *gemmAStrategy = &strategy.A, *gemmBStrategy = &strategy.B;

    // Base strategy parameters.
    strategize(product, fargs, strategy);
    strategy.unroll[LoopM] = unrollM;
    strategy.unroll[LoopN] = unrollN;
    if (!fargs.MKLInterface && !forceKRemHandling && isPacked(ALayout) && isPacked(BLayout))
        strategy.remHandling[LoopK] = RemainderHandling::Ignore;

    // User-specified strategy.
    if (!gotStrategy && !args.strategy.empty()) {
        parseStrategy(args.strategy, hw, problem, strategy);
        gotStrategy = true;
    }

    // Table-based strategies.
    if (!gotStrategy) {
        auto effALayout = ALayout, effBLayout = BLayout;

        if (trsm && !args.inplace)
            (args.left ? effBLayout : effALayout) = CLayout;

        char altLayoutC = (isColMajor(CLayout) ^ isLargeCrosspack(Tc, problem.C.crosspack)) ? 'N' : 'T';

        auto originalProblem = problem;
        auto originalStrategy = strategy;

        for (int r = 0; r < gemmRecipeCount; r++) {
            auto &recipe = gemmRecipes[r];
            if (recipeHWMatch(hw, recipe.hw)
                    && recipe.precisions[0] == std::toupper(precisionChar(Ta))
                    && recipe.precisions[1] == std::toupper(precisionChar(Tb))
                    && recipe.precisions[2] == std::toupper(precisionChar(Tc))
                    && recipe.layouts[0] == layoutChar(effALayout)
                    && recipe.layouts[1] == layoutChar(effBLayout)
                    && (recipe.layouts[2] == layoutChar(CLayout)
                    ||  recipe.layouts[2] == altLayoutC)
                    && recipe.extras.aCP == gemmA->crosspack
                    && recipe.extras.bCP == gemmB->crosspack
                    && (gemmA->alignment % recipe.extras.aAlign) == 0
                    && (gemmB->alignment % recipe.extras.bAlign) == 0
                    && recipe.unrollM == unrollM
                    && recipe.unrollN == unrollN
                    && recipe.tag == args.strategyTag) {

                bool bestFit = (recipe.hw == hw) && (recipe.layouts[2] == layoutChar(CLayout));

                auto thisProblem = originalProblem;
                auto thisStrategy = originalStrategy;
                parseStrategy(recipe.strategyString, hw, thisProblem, thisStrategy);

                // Accept this as current best option.
                problem = thisProblem;
                strategy = thisStrategy;
                gotStrategy = true;

                if (problem.batch == BatchMode::Nonstrided || problem.batch == BatchMode::Variable)
                    bestFit &= strategy.A.base.isStateless() && strategy.B.base.isStateless() && strategy.C.base.isStateless();

                // Look for a better option if this wasn't perfect.
                if (bestFit) break;
            }
        }

        if (!gotStrategy && args.strategyTag)
            throw std::runtime_error("Unknown strategy tag.");
    }

    // Default strategy.
    if (!gotStrategy) {
        strategy.unroll[LoopK] = 1;
        strategy.A_copies = 1;
        strategy.B_copies = 1;
        strategy.ka_load = 1;
        strategy.kb_load = 1;

        strategy.A.base = AddressBase::createA64(true);
        strategy.B.base = AddressBase::createA64(true);
        strategy.C.base = AddressBase::createA64(true);

        strategy.A_scale.base = AddressBase::createA64(true);
        strategy.B_scale.base = AddressBase::createA64(true);

        strategy.CO.base = (hw >= HW::XeHPC || args.modifiers.sumA || args.modifiers.sumB) ? AddressBase::createA64(true) : AddressBase::createBTS(0);
        strategy.A.accessType = (gemmA->layout == MatrixLayout::T) ? AccessType::Scattered : AccessType::Block;
        strategy.B.accessType = (gemmB->layout == MatrixLayout::N) ? AccessType::Scattered : AccessType::Block;
        strategy.C.accessType = (Tc.size() < 4 || hw >= HW::XeHP || unrollM >= unrollN) ? AccessType::Block : AccessType::Scattered;


        strategy.A_scale.accessType  = (gemmB->layout == MatrixLayout::N) ? AccessType::Scattered : AccessType::Block;
        strategy.B_scale.accessType  = (gemmB->layout == MatrixLayout::N) ? AccessType::Scattered : AccessType::Block;

        strategy.A.padded |= isPacked(problem.A.layout);
        strategy.B.padded |= isPacked(problem.B.layout);

        strategy.B_scale.padded |= isPacked(problem.B.layout);

        strategy.A_prefetch.padded |= isPacked(problem.A.layout);
        strategy.B_prefetch.padded |= isPacked(problem.B.layout);

        strategy.altCRemainder |= (strategy.C.accessType == AccessType::Block) || strategy.kParallel;

        gotStrategy = true;
    }

    // Overlay parameters.
    strategy.tracking = args.tracking;

    if (args.kParallel) {
        strategy.C.atomic = strategy.kParallel = true;
        strategy.CO.atomic = args.modifiers.sumA || args.modifiers.sumB;
    }

    if (args.kParallelLocal)
        strategy.kParallelLocal = true;

    // Override usage of new dataport messages.
    bool defaultNewDP = (hw >= HW::XeHPC);
    bool forceNewDP = (hw >= HW::XeHPC);
    if (getenv("FORCE_NEW_DP")) {
        defaultNewDP = atoi(getenv("FORCE_NEW_DP"));
        forceNewDP = true;
    }
    if (hw >= HW::XeHPG && forceNewDP) {
        strategy.A.newDP = strategy.A_prefetch.newDP = defaultNewDP;
        strategy.B.newDP = strategy.B_prefetch.newDP = defaultNewDP;
        strategy.C.newDP = strategy.C_prefetch.newDP = defaultNewDP;
        strategy.CO.newDP = defaultNewDP;
    }

    // Temporarily use XeHPC register banking for Xe2/Xe3.
    if ((hw == HW::Xe2 || hw == HW::Xe3) && !getenv("NATIVE_XE2_RA"))
        strategy.raHW = HW::XeHPC;

    // Or force XeHPC register banking if requested.
    if (getenv("FORCE_XEHPC_RA"))
        strategy.raHW = HW::XeHPC;

    // Apply recommended strategy adjustments.
    adjustStrategy(hw, problem, strategy);

    // Preflight on the resulting strategy.
    strategy.preflight(hw, problem);
}

void handleDefaultAlignment(MatrixAddressing &atype, Type T)
{
    if (atype.alignment == 0) atype.setAlignment(atype.defaultAlignment(T));
}

void createLevel3Problem(GEMMProblem &problem, const Level3KernelArgs &args, const frontend::EnvConfig &fargs, ngen::HW hw)
{
    auto Ta = args.Ta_internal, Tb = args.Tb_internal, Tc = args.Tc_internal, Ts = args.Ts;
    auto Ta_ext = args.Ta, Tb_ext = args.Tb, Tc_ext = args.Tc;

    // Set up problem structure.
    auto ALayout = args.A.layout;
    auto BLayout = args.B.layout;
    auto CLayout = args.C.layout;
    auto updateType = args.updateType;

    bool aPacked = isPacked(ALayout);
    bool bPacked = isPacked(BLayout);

    bool aoPrecalc = args.modifiers.aOffset.precalc || bPacked;
    bool boPrecalc = args.modifiers.bOffset.precalc || aPacked;

    problem.Ta = Ta;
    problem.Tb = Tb;
    problem.Tc = Tc;
    problem.Ts = Ts;
    problem.Ta_ext = Ta_ext;
    problem.Tb_ext = Tb_ext;
    problem.Tc_ext = Tc_ext;
    problem.A = args.A;
    if (problem.A.packSize == 0)  problem.A.packSize = args.unrollM[0];
    handleDefaultAlignment(problem.A, problem.Ta_ext);
    problem.B = args.B;
    if (problem.B.packSize == 0)  problem.B.packSize = args.unrollN[0];
    handleDefaultAlignment(problem.B, problem.Tb_ext);
    problem.C = args.C;
    problem.C.packSize = (CLayout == MatrixLayout::Pc) ? problem.A.packSize :
                        (CLayout == MatrixLayout::Pr) ? problem.B.packSize : 0;
    handleDefaultAlignment(problem.C, problem.Tc_ext);
    problem.aOffset = !args.modifiers.aOffset.enabled ? ABOffset::None :
                                  aoPrecalc ? ABOffset::Load :
                                              ABOffset::Calc;
    problem.bOffset = !args.modifiers.bOffset.enabled ? ABOffset::None :
                                  boPrecalc ? ABOffset::Load :
                                              ABOffset::Calc;
    problem.cOffset = args.modifiers.bias ? COffset::Pre :
            args.modifiers.cOffset.enabled ? COffset::Post :
                                   COffset::None;
    problem.aoPtrDims = args.modifiers.aOffset.ptrDims;
    problem.boPtrDims = args.modifiers.bOffset.ptrDims;
    problem.coPtrDims = args.modifiers.cOffset.ptrDims;
    problem.Tco = args.modifiers.cOffset.T;
    if (problem.Tco == Type::invalid) problem.Tco = Tc_ext;
    problem.asPtrDims = args.modifiers.aScale.ptrDims;
    problem.bsPtrDims = args.modifiers.bScale.ptrDims;
    if (problem.aoPtrDims >= 0) {
        problem.Tao = args.modifiers.aOffset.T;
        if (problem.Tao == Type::invalid)
            problem.Tao = (problem.Ta.paddedSize() == 1) ? Type(Type::u8) : problem.Tc;
    } else if (args.modifiers.aOffset.enabled)
        problem.Tao = Type::s16;
    if (problem.aOffset2D()) {
        problem.AO.layout = MatrixLayout::N;
        problem.AO.alignment = problem.Tao.paddedSize();
    }
    if (problem.boPtrDims >= 0) {
        problem.Tbo = args.modifiers.bOffset.T;
        if (problem.Tbo == Type::invalid)
            problem.Tbo = (problem.Tb.paddedSize() == 1) ? Type(Type::u8) : problem.Tc;
    } else if (args.modifiers.bOffset.enabled)
        problem.Tbo = Type::s16;
    if (problem.bOffset2D()) {
        problem.BO.layout = MatrixLayout::N;
        problem.BO.alignment = problem.Tbo.paddedSize();
    }
    auto defaultScaleType = [](Type Tx_ext, Type Tother) -> Type {
        if (Tx_ext == Type::nf4) return Type::f16;
        if (Tx_ext.isFP() && Tx_ext.bits() <= 8) return Type::f8_e8m0;
        if (Tother.isFP()) return Tother;
        return Type::f16;
    };
    if (problem.aScale2D()) {
        problem.Ta_scale = args.modifiers.aScale.T;
        if (problem.Ta_scale == Type::invalid)
            problem.Ta_scale = defaultScaleType(problem.Ta_ext, problem.Tb);
        problem.A_scale.layout = MatrixLayout::N;
        problem.A_scale.alignment = problem.Ta_scale.paddedSize();
    }
    if (problem.bScale2D()) {
        problem.Tb_scale = args.modifiers.bScale.T;
        if (problem.Tb_scale == Type::invalid)
            problem.Tb_scale = defaultScaleType(problem.Tb_ext, problem.Ta);
        problem.B_scale.layout = MatrixLayout::N;
        problem.B_scale.alignment = problem.Tb_scale.paddedSize();
    }
    if (problem.needsAGroupSums()) {
        problem.Tag = Ta.isInteger() ? Type::s32 : Type::f32;
        problem.Ag.layout = MatrixLayout::N;
        problem.Ag.alignment = problem.Tag.paddedSize();
    }
    if (problem.needsBGroupSums()) {
        problem.Tbg = Tb.isInteger() ? Type::s32 : Type::f32;
        problem.Bg.layout = MatrixLayout::N;
        problem.Bg.alignment = problem.Tbg.paddedSize();
    }
    if (args.modifiers.aOffset.ptrDims == 2 && args.modifiers.aScale.ptrDims == 2 && args.modifiers.aOffset.groupSize != args.modifiers.aScale.groupSize)
        throw std::runtime_error("A offset and scale must have the same group size");
    if (args.modifiers.bOffset.ptrDims == 2 && args.modifiers.bScale.ptrDims == 2 && args.modifiers.bOffset.groupSize != args.modifiers.bScale.groupSize)
        throw std::runtime_error("B offset and scale must have the same group size");
    for (auto &aq: {args.modifiers.aOffset, args.modifiers.aScale}) if (aq.ptrDims == 2) {
        problem.aqGroupM = aq.groupSize[0];
        problem.aqGroupK = aq.groupSize[1];
    }
    for (auto &bq: {args.modifiers.bOffset, args.modifiers.bScale}) if (bq.ptrDims == 2) {
        problem.bqGroupK = bq.groupSize[0];
        problem.bqGroupN = bq.groupSize[1];
    }
    if (problem.needsAGroupSums() && args.modifiers.aOffset.ptrDims < 2 && args.modifiers.aScale.ptrDims < 2)
        problem.aqGroupK = problem.bqGroupK;
    else if (problem.needsBGroupSums() && args.modifiers.bOffset.ptrDims < 2 && args.modifiers.bScale.ptrDims < 2)
        problem.bqGroupK = problem.aqGroupK;
    problem.sumA = args.modifiers.sumA;
    problem.sumB = args.modifiers.sumB;
    if (problem.usesCOPtr()) {
        problem.CO.crosspack = 1;
        problem.CO.setAlignment(Tc.size());
        problem.CO.layout = MatrixLayout::N;
    }
    problem.alpha = args.alpha;
    problem.beta = args.beta;
    problem.readCheck = args.modifiers.readCheck;
    problem.readCheckA = args.modifiers.readCheckA;
    problem.readCheckB = args.modifiers.readCheckB;
    problem.readCheckC = args.modifiers.readCheckC;
    problem.mock = args.mock;

    problem.batch = args.modifiers.batch;
    problem.batchDims = (args.modifiers.batch != BatchMode::None) ? 1 : 0;

    problem.nonuniformWGs = !fargs.UniformWG;
}

void addPrecisionToKernelName(std::string &kernel_name, Type Ta, Type Tb, Type Tc)
{
    if ((Ta == Tb) && (Tb == Tc))
        kernel_name[0] = precision_char(Tc);
    else {
        std::string suffix = "_";
        suffix += precision_str(Ta);
        suffix += precision_str(Tb);
        suffix += precision_str(Tc);
        kernel_name.erase(0, 1);
        kernel_name.insert(4, suffix);
    }
}

void generateLevel3Kernel(BLASInterface *generator, Level3KernelArgs &args, const frontend::EnvConfig &fargs, std::string &info)
{
    ngen::Product product = generator->getProduct();
    ngen::HW hw = getCore(product.family);
    GEMMProblem problem;
    struct {
        std::vector<GEMMStrategy> substrategies;
    } strategy;

    generator->setStepping(product.stepping);

    // Set up problem structure.
    problem.product = product;
    createLevel3Problem(problem, args, fargs, hw);

    // Nothing else enables bdpas here, so turn it on whenever the hardware
    // supports it natively.
    if (problem.nativeBDPAS())
        problem.bdpasEnabled = true;

    auto Tc = problem.Tc, Ts = problem.Ts;
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext, Tc_ext = problem.Tc_ext;
    auto ALayout = args.A.layout;
    auto BLayout = args.B.layout;
    auto updateType = args.updateType;
    bool trsm = false;

    bool MKLCopyBased = fargs.MKLInterface && isPacked(ALayout) && isPacked(BLayout);
    bool MKLNoCopy = fargs.MKLInterface && !isPacked(ALayout) && !isPacked(BLayout);

    auto nstrategies = (args.multiM() ? 2 : 1) * (args.multiN() ? 2 : 1);
    strategy.substrategies.resize(nstrategies, GEMMStrategy(hw, product.stepping));

    int kidx = 0;
    for (int kid0 = 0; kid0 <= int(args.multiM()); kid0++)
        for (int kid1 = 0; kid1 <= int(args.multiN()); kid1++)
            strategize(product, args, fargs, kid0, kid1, problem, strategy.substrategies[kidx++], args.modifiers.superkernel);

    if (args.modifiers.superkernel) {
        stub("Superkernels are not supported.");
    }

#ifdef ASM_OUTPUT
    for (const auto &ss : strategy.substrategies)
        info += "//  " + unparseStrategy(hw, problem, ss) + "\n";
#endif

    auto &strategy0 = strategy.substrategies[0];

    problem.autoTypeConversions(strategy0.systolicAvailable);

    // Set up arguments.
    auto interface = createInterface(hw);

    auto aAccess = strategy0.A.getGlobalAccessType();
    auto bAccess = strategy0.B.getGlobalAccessType();
    auto cAccess = strategy0.C.getGlobalAccessType();
    auto aoAccess = strategy0.AO.getGlobalAccessType();
    auto boAccess = strategy0.BO.getGlobalAccessType();
    auto coAccess = strategy0.CO.getGlobalAccessType();
    auto asAccess = strategy0.A_scale.getGlobalAccessType();
    auto bsAccess = strategy0.B_scale.getGlobalAccessType();
    auto agAccess = strategy0.Ag.getGlobalAccessType();
    auto bgAccess = strategy0.Bg.getGlobalAccessType();

    auto argPtrType = [](const MatrixAddressingStrategy &astrategy) {
        return (astrategy.base.getModel() == ModelSLM) ? ExternalArgumentType::LocalPtr
                                                    : ExternalArgumentType::GlobalPtr;
    };

    bool needsFlags = false;

    if (args.micro) {
        interface.setArgumentBase(GRF(8));
        interface.newArgument("A", argPtrType(strategy0.A), aAccess);
        interface.newArgument("lda", DataType::ud);
        interface.newArgument("B", argPtrType(strategy0.B), bAccess);
        interface.newArgument("ldb", DataType::ud);
        interface.newArgument("m", DataType::d);
        interface.newArgument("n", DataType::d);
        interface.newArgument("k", DataType::d);
        interface.newArgument("i0", DataType::ud);
        interface.newArgument("j0", DataType::ud);
        interface.newArgument("h0", DataType::ud);
        interface.newArgument("local_id_m", DataType::ud);
        interface.newArgument("local_id_n", DataType::ud);
    } else if (!MKLCopyBased) {
        interface.newArgument("A", argPtrType(strategy0.A), aAccess);
        interface.newArgument("B", argPtrType(strategy0.B), bAccess);
        interface.newArgument("C", argPtrType(strategy0.C), cAccess);
        interface.newArgument("offset_A", DataType::q);
        interface.newArgument("offset_B", DataType::q);
        interface.newArgument("offset_C", DataType::q);
        if (args.modifiers.batch == BatchMode::Strided) {
            interface.newArgument("stride_A0", DataType::q);
            interface.newArgument("stride_B0", DataType::q);
            interface.newArgument("stride_C0", DataType::q);
        } else if (args.modifiers.batch == BatchMode::Nonstrided)
            interface.newArgument("offset_batch", DataType::ud);
        else if (args.modifiers.batch == BatchMode::Variable) {
            interface.newArgument("incr_a_array", DataType::ud);
            interface.newArgument("incr_b_array", DataType::ud);
        }
        interface.newArgument("lda", DataType::d);
        interface.newArgument("ldb", DataType::d);
        interface.newArgument("ldc", DataType::d);
        interface.newArgument("m", DataType::d);
        interface.newArgument("n", DataType::d);
        interface.newArgument("k", DataType::d);
        if (args.modifiers.hpl) {
            interface.newArgument("P", ExternalArgumentType::GlobalPtr, cAccess);
            interface.newArgument("offset_P", DataType::q);
            interface.newArgument("ldp", DataType::ud);
        }
        if (args.alpha.pointer())
            interface.newArgument("alpha_ptr", ExternalArgumentType::GlobalPtr);
        else
            interface.newArgument("alpha_real", Ts.real().ngen());
        if (args.beta.pointer()) {
            interface.newArgument("beta_ptr", ExternalArgumentType::GlobalPtr);
            needsFlags = true;
        } else
            interface.newArgument("beta_real", Ts.real().ngen());
        if (Tc.isComplex()) {
            if (!args.alpha.pointer()) interface.newArgument("alpha_imag", Ts.real().ngen());
            if (!args.beta.pointer())  interface.newArgument("beta_imag",  Ts.real().ngen());
            needsFlags |= !isPacked(ALayout) || !isPacked(BLayout);
        }
        if (args.modifiers.batch == BatchMode::Variable) {
            interface.newArgument("alpha_array", ExternalArgumentType::GlobalPtr);
            interface.newArgument("beta_array", ExternalArgumentType::GlobalPtr);
            interface.newArgument("incr_alpha", DataType::ud);
            interface.newArgument("incr_beta", DataType::ud);
        }
    }

    if ((args.modifiers.aOffset.enabled && args.modifiers.aOffset.ptrDims == -1)
            || (args.modifiers.bOffset.enabled && args.modifiers.bOffset.ptrDims == -1))
        interface.newArgument("abo", DataType::ud);
    if (problem.aoPtrDims >= 0)
        interface.newArgument("ao_ptr", ExternalArgumentType::GlobalPtr, aoAccess);
    if (problem.boPtrDims >= 0)
        interface.newArgument("bo_ptr", ExternalArgumentType::GlobalPtr, boAccess);
    if (problem.aScale2D())
        interface.newArgument("a_scale_ptr", ExternalArgumentType::GlobalPtr, asAccess);
    if (problem.bScale2D())
        interface.newArgument("b_scale_ptr", ExternalArgumentType::GlobalPtr, bsAccess);
    if (problem.needsSeparateASums())
        interface.newArgument("ag_ptr", ExternalArgumentType::GlobalPtr, agAccess);
    if (problem.needsSeparateBSums())
        interface.newArgument("bg_ptr", ExternalArgumentType::GlobalPtr, bgAccess);
    if (problem.aOffset2D() || problem.aScale2D() || problem.needsAGroupSums())
        interface.newArgument("ldaq", DataType::d);
    if (problem.bOffset2D() || problem.bScale2D() || problem.needsBGroupSums())
        interface.newArgument("ldbq", DataType::d);
    if (problem.usesCOPtr()) {
        interface.newArgument("co_ptr", ExternalArgumentType::GlobalPtr, coAccess);
        interface.newArgument("offset_CO", DataType::q);
        if (problem.cOffset != COffset::None)
            needsFlags = true;
        if (problem.allowMatrixOffset())
            interface.newArgument("ldco", DataType::d);
    } else if (problem.cOffsetHostScalar()) {
        interface.newArgument("co", DataType::w);
    }
    if (strategy0.needsTempC(problem))
        interface.newArgument("temp_C", ExternalArgumentType::GlobalPtr, cAccess);
    (void) needsFlags;
    interface.newArgument("flags", DataType::ud);
    if ((strategy0.kParallel || strategy0.kParallelLocal) && !strategy0.kParallelVariable)
        interface.newArgument("k0", DataType::d);
    if (strategy0.fuseBeta || strategy0.fusePostOps)
        interface.newArgument("status", ExternalArgumentType::GlobalPtr, GlobalAccessType::Stateless);
    if (strategy0.linearOrder()) {
        interface.newArgument("group_count_m", DataType::ud);
        interface.newArgument("group_count_n", DataType::ud);
    }
    if (strategy0.kParallel && strategy0.fuseBeta)
        interface.newArgument("group_count_k", DataType::ud);
    if (strategy0.scramble[LoopM])
        interface.newArgument("group_count_m_recip", DataType::ud);
    if (strategy0.scramble[LoopN])
        interface.newArgument("group_count_n_recip", DataType::ud);
    if (strategy0.cWalkOrder == WalkOrder::SimpleLinear || strategy0.cWalkOrder == WalkOrder::NestedLinear)
        interface.newArgument("group_count_recip", DataType::ud);
    else if (strategy0.cWalkOrder == WalkOrder::Hilbertlike) {
        interface.newArgument("hilbert_vd", DataType::ud);
        interface.newArgument("hilbert_uvd_recip", DataType::ud);
        interface.newArgument("hilbert_bail", DataType::ud);
    } else if (strategy0.cWalkOrder == WalkOrder::Boustrophedon) {
        interface.newArgument("bslice", DataType::d);
        interface.newArgument("bthresh", DataType::d);
    }
    if (strategy0.scramble[LoopM])
        interface.newArgument("wg_stride_m", DataType::ud);
    if (strategy0.scramble[LoopN])
        interface.newArgument("wg_stride_n", DataType::ud);
    if (strategy0.kParallelVariable) {
        interface.newArgument("k0", DataType::ud);
        interface.newArgument("kv_config", DataType::ud);
        interface.newArgument("k_recip", DataType::ud);
    }
    if (strategy0.persistent)
        interface.newArgument("group_stride", DataType::ud);
    if (strategy0.persistent && strategy0.cWalkOrder == WalkOrder::Hilbertlike && (args.updateType != UpdateType::Full))
        interface.newArgument("group_count", DataType::ud);
    if (strategy0.variableSLM())
        interface.newArgument("local_mem", ExternalArgumentType::LocalPtr);
    if (problem.aoPtrDims >= 1 || problem.aScale2D())
        interface.newArgument("offset_Aq", DataType::q);
    if (problem.boPtrDims >= 1 || problem.bScale2D())
        interface.newArgument("offset_Bq", DataType::q);
    if (strategy0.tracking.enabled) {
        interface.newArgument("tracking", ExternalArgumentType::GlobalPtr, GlobalAccessType::Surface);
        interface.newArgument("tracking_stride", DataType::uw);
    }

    // Keep the declared argument layout; ngen would otherwise repack it.
    if (hw >= HW::XeHPG) interface.allowArgumentRearrangement(false);

    interface.externalName("gemm_kernel");

#if defined(ASM_OUTPUT) && defined(NGEN_ASM_HAS_GET_PARTIAL_CODE)
    try {
#endif

    if (args.micro)
        generator->gemmMicrokernel(problem, strategy.substrategies[0], interface);
    else
        generator->gemm(problem, strategy.substrategies[0], interface);

#if defined(ASM_OUTPUT) && defined(NGEN_ASM_HAS_GET_PARTIAL_CODE)
    } catch (...) {
        generator->getPartialCode(std::cerr);
        throw;
    }
#endif
}

const char *getLayout(const char *s, MatrixAddressing &external, MatrixAddressing &internal)
{
    if (*s == '[') {
        s = parseLayout(s + 1, external);
        s = parseLayout(s, internal);
        if (*s++ != ']')
            throw std::runtime_error("Expected ].");
    } else {
        s = parseLayout(s, external);
        internal = external;
    }

    return s;
}

void getLayouts(const char *s, MatrixAddressing &A, MatrixAddressing &B)
{
    s = parseLayout(s, A);
    s = parseLayout(s, B);
}

void getLayouts(const char *s, MatrixAddressing &A, MatrixAddressing &B, MatrixAddressing &C, MatrixAddressing &A_int, MatrixAddressing &B_int)
{
    MatrixAddressing C_int;

    s = getLayout(s, A, A_int);
    s = getLayout(s, B, B_int);
    (void) getLayout(s, C, C_int);
}

void getPrecisions(const char *s, Type &precisionA, Type &precisionB, Type &precisionSum)
{
    s = parsePrecision(s, precisionA);
    s = parsePrecision(s, precisionB);
    if (*s == '+') {
        s++;
        s = parsePrecision(s, precisionSum);
    } else
        precisionSum = Type::invalid;
}

void getPrecisions(const char *s, Type &precisionA, Type &precisionB, Type &precisionC, Type &precisionS, Type &precisionAInt, Type &precisionBInt, Type &precisionCInt)
{
    s = parsePrecisions(s, precisionA, precisionAInt);
    s = parsePrecisions(s, precisionB, precisionBInt);
    s = parsePrecisions(s, precisionCInt, precisionC);
    precisionS = precisionCInt;
    s = parsePrecision(s, precisionS);
}

void getUnrolls(const char *s, int (&unroll)[2])
{
    unroll[0] = std::atoi(s);
    auto second = std::strchr(s, '/');
    unroll[1] = second ? std::atoi(second + 1) : 0;

    if (unroll[0] <= 0 || unroll[1] < 0)
        throw std::runtime_error("Invalid unroll specified.");
}

Scalar getScalar(const char *s)
{
    Scalar result;
    if (s[0] == '-' && (std::isspace(s[1]) || s[1] == '\0'))
        result = Scalar::Variable;
    else if (s[0] == '@')
        result = Scalar::Pointer;
    else if (s[0] == 'R')
        result = Scalar::RealPointer;
    else
        result = std::atoi(s);
    return result;
}

void parseMock(int &argc, char **&argv, Level3KernelArgs &args)
{
    if (argc > 0) if (!std::strcmp(*argv, "mock")) {
        argv++; argc--;
        for (; argc > 0; argc--, argv++) {
            if (!std::strcmp(*argv, "c"))
                args.mock.cUpdate = true;
            else if (!std::strcmp(*argv, "cr"))
                args.mock.cLoad = true;
            else if (!std::strcmp(*argv, "cw"))
                args.mock.cStore = true;
            else if (!std::strcmp(*argv, "cn"))
                args.mock.cInit = true;
            else if (!std::strcmp(*argv, "kkr"))
                args.mock.keepKReduce = true;
            else if (!std::strcmp(*argv, "ab0"))
                args.mock.abZeroInc = true;
            else if (!std::strcmp(*argv, "abi"))
                args.mock.abInc = true;
            else if (!std::strcmp(*argv, "ab"))
                args.mock.abLoad = true;
            else if (!std::strcmp(*argv, "abr"))
                args.mock.abLoad = args.mock.abRandom = true;
            else if (!std::strcmp(*argv, "abn"))
                args.mock.abLoad = args.mock.abInit = true;
            else if (!std::strcmp(*argv, "sw"))
                args.mock.slmW = true;
            else if (!std::strcmp(*argv, "m"))
                args.mock.multiply = true;
            else if (!std::strcmp(*argv, "sn"))
                args.mock.slmInit = true;
            else if (!std::strcmp(*argv, "b"))
                args.mock.barriers = true;
            else if (!std::strcmp(*argv, "--")) {
                argv++; argc--;
                break;
            } else
                throw std::runtime_error("Unknown mock parameter.");
        }
    }
}

void parseTracking(int &argc, char **&argv, Level3KernelArgs &args)
{
    if (argc > 0) if (!std::strcmp(*argv, "track")) {
        argv++; argc--;
        args.tracking.enabled = true;
        for (; argc > 0; argc--, argv++) {
            if (!std::strcmp(*argv, "all")) {
                args.tracking.startEnd = true;
                args.tracking.accumDone = true;
            } else if (!std::strcmp(*argv, "se"))
                args.tracking.startEnd = true;
            else if (!std::strcmp(*argv, "ad"))
                args.tracking.accumDone = true;
            else if (!std::strcmp(*argv, "--")) {
                argv++; argc--;
                break;
            } else
                throw std::runtime_error("Unknown tracking type requested.");
        }
    }
}

#ifdef OPENCL_OUTPUT
void handleCL(cl_int status)
{
    if (status != CL_SUCCESS)
        throw std::runtime_error("OpenCL error occurred.\n");
}

cl_device_id findGPU()
{
    static constexpr auto MaxPlatforms = 16;
    static constexpr auto MaxDevices = 16;
    cl_platform_id plat_ids[MaxPlatforms];
    cl_device_id dev_ids[MaxDevices];
    cl_uint nplat, ndev;

    handleCL(clGetPlatformIDs(MaxPlatforms, plat_ids, &nplat));
    if (nplat == 0) throw std::runtime_error("No OpenCL platforms are available.");

    for (cl_uint plat_idx = 0; plat_idx < nplat; plat_idx++) {
        auto plat = plat_ids[plat_idx];

        cl_int status = clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, MaxDevices, dev_ids, &ndev);

        if (status == CL_DEVICE_NOT_FOUND)
            continue;
        handleCL(status);

        if (ndev <= 0)
            continue;

        return dev_ids[0];
    }

    throw std::runtime_error("No GPUs found.");
}
#endif

std::unique_ptr<Generator> Generator::Create() {
    Product product = detectDevice(true);
    return GetGenerator(product);
}

std::unique_ptr<Generator> Generator::Create(Product product) {
    return GetGenerator(product);
}

bool Generator::IsSupported(ngen::HW hw) {
    return IsHWSupported(hw);
}

Result generate_impl(BLASInterface *generator, int argc, char **argv, bool infoOnly) {
    Product product = generator->getProduct();

    Result res;
    EnvConfig fargs;

    ::utils::KernelModifiers modifiers;
    bool micro = false;

    bool disableAutoSWSB = false;

    auto daEnv = getenv("FORCE_AUTODEPS");
    if (daEnv && std::string(daEnv) != "0")
        disableAutoSWSB = true;

    while (argc > 0 && modifiers.parse(*argv)) {
        argv++; argc--;
    }

    if (argc == 0)
        throw std::runtime_error("Missing kernel type");
    auto type = *argv;

    if (type[0] == 'u')
        micro = true, type++;

    bool gemm = !std::strncmp(type, "gemm", 4);
    bool symm = !std::strncmp(type, "symm", 4);
    bool hemm = !std::strncmp(type, "hemm", 4);
    bool trmm = !std::strncmp(type, "trmm", 4);
    bool trsm = !std::strncmp(type, "trsm", 4);

    if (gemm || symm || hemm || trmm || trsm) {
        Level3KernelArgs args;

        if ((modifiers.aScale.enabled && !modifiers.aScale.grouped())
                || (modifiers.bScale.enabled && !modifiers.bScale.grouped()))
            throw std::runtime_error("Only grouped scale supported");
        args.modifiers = modifiers;
        args.alpha = 1;
        args.beta = 0;
        args.kParallel = false;
        args.kParallelLocal = false;
        args.micro = micro;

        args.Ta = args.Tb = args.Tc = args.Ts = Type::f32;
        args.Ta_internal = args.Tb_internal = args.Tc_internal = Type::f32;

        UpdateType updateType = UpdateType::Full;

        if (gemm) {
            switch (type[4]) {
                case '\0':
                case ' ': updateType = UpdateType::Full;          break;
                default: throw std::runtime_error("Unknown gemm kernel type requested.");
            }

        } else if (trmm || trsm) {
            char *cp = type + 4;
            args.inplace = args.checkUnit = true;
            args.unit = false;
            if (*cp == '3') {
                args.inplace = false;
                cp++;
            }
            args.left  = (*cp == 'l'); if (*cp) cp++;
            args.lower = (*cp == 'l'); if (*cp) cp++;
            if (*cp) {
                args.unit = (*cp == 'u');
                args.checkUnit = false;
                cp++;
            }
        } else if (symm || hemm) {
            char *cp = type + 4;
            args.left  = (*cp == 'l'); if (*cp) cp++;
            args.lower = (*cp == 'l'); if (*cp) cp++;
        }

        args.updateType = updateType;

        argv++; argc--;
        if (argc) { getPrecisions(*argv++, args.Ta, args.Tb, args.Tc, args.Ts, args.Ta_internal, args.Tb_internal, args.Tc_internal); argc--; } else return Result("failed to parse data types");
        if (argc) { getLayouts(*argv++, args.A_ext, args.B_ext, args.C, args.A, args.B); argc--; } else return Result("failed to parse layout");
        if (argc) { getUnrolls(*argv++, args.unrollM); argc--; } else return Result("failed to parse unrolls");
        if (argc) { getUnrolls(*argv++, args.unrollN); argc--; } else return Result("failed to parse unrolls");
        if (argc && (!std::isalpha(**argv) || **argv == 'R')) {
            if (argc) { args.alpha = getScalar(*argv++); argc--; }
            if (argc) { args.beta = getScalar(*argv++); argc--; }
        } else for (int i = 0; i < argc; i++)
            if (!std::strcmp(argv[i], "kb"))
                args.beta = 1;

        if (argc && isupper(**argv)) {
            args.strategyTag = **argv;
            argv++; argc--;
        }

        if (argc && !std::strcmp(*argv, "kb")) {
            argv++; argc--;
            args.kParallel = true;
        }

        if (argc && !std::strcmp(*argv, "kr")) {
            argv++; argc--;
            args.kParallelLocal = true;
        }

        if (argc) parseMock(argc, argv, args);

        if (argc) parseTracking(argc, argv, args);

        if (argc) {
            // Parse remaining args as strategy parameters.
            std::stringstream ss;
            for (; argc > 0; argc--, argv++) {
                if (!std::strcmp(*argv, "--")) {
                    argc--; argv++; break;
                }
                ss << *argv << ' ';
            }
            args.strategy = ss.str();

            if (modifiers.fused) {
                for (bool copyB : {false, true}) {
                    for (bool trans : {false, true}) {
                        if (argc == 0) break;

                        std::stringstream css;
                        for (; argc > 0; argc--, argv++) {
                            if (!std::strcmp(*argv, "--")) {
                                argc--; argv++; break;
                            }
                            css << *argv << ' ';
                        }
                        args.copyStrategies[copyB][trans] = css.str();
                    }
                }
            }

            parseMock(argc, argv, args);
            parseTracking(argc, argv, args);
        }

            generateLevel3Kernel(generator, args, fargs, res.asmStr);
    } else {
        return Result("failed to parse kernel type: " + std::string(type));
    }
    res.info = generator->getNotes();

    if (!infoOnly) {
#if defined(_WIN32) && !defined(ASM_OUTPUT)
        _setmode(_fileno(stdout), _O_BINARY);
#endif

#ifdef ASM_OUTPUT
        if (disableAutoSWSB)
            generator->disableAutoSWSB();
        if (simulation())
            generator->enableLineNumbers();
        std::ostringstream asmStream;
        generator->getCode(asmStream);
        res.asmStr += asmStream.str();
#elif defined(BINARY_OUTPUT)
        auto start = std::chrono::steady_clock::now();
        res.binary = generator->getCode();
        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<double> diff = end - start;
        if (getenv("TIME_AUTOSWSB")) {
            res.info = std::to_string(diff.count() * 1e3) + "\n";
        }
#elif defined(OPENCL_OUTPUT)
        if (getenv("FORCE_ZEBIN"))
            res.binary = generator->getBinary();
        else {
            auto device = findGPU();
            auto context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
            if (!context) throw std::runtime_error("clCreateContext failed.");
            res.binary = generator->getBinary(context, device);
            clReleaseContext(context);
        }
#elif defined(ZEBIN_OUTPUT)
        res.binary = generator->getBinary();
#endif
    }
    res.success = true;
    return res;
}

Result Result::fromTmpFile()
{
    Result data;
#ifdef _WIN32
    const char *temp = "C:\\Temp\\";
    const char *user = getenv("USERNAME");
#else
    const char *temp = "/tmp/";
    const char *user = getpwuid(geteuid())->pw_name;
#endif
    // User can set this environment variable to enable loading from a different tmp file.
    // When set to the user's PID, lets us use frontend from multiple processes simultaneously
    const char *benchmarkID = getenv("BENCHMARK_ID");
    if (!benchmarkID) benchmarkID = "";
    std::string tmp_stem = std::string(temp) + "gemm." + user + benchmarkID;
    std::string binary_fname = tmp_stem + ".elf";
    std::string note_fname = tmp_stem + ".note";

    auto ifile = ::utils::openFile(note_fname.c_str(), "r");
    ::utils::readFile(data.info, ifile);
    fclose(ifile);

    auto bfile = ::utils::openFile(binary_fname.c_str(), "rb");
    ::utils::readFile(data.binary, bfile);
    fclose(bfile);

    data.success = !data.binary.empty();
    return data;
}

//
Result BLASInterface::generate(const std::string &kernelArgs, bool infoOnly) {
    if (kernelArgs.empty())
        return Result::fromTmpFile();

    auto words = ::utils::extractWords(kernelArgs);
    std::vector<char *> argv;
    for (auto &word : words)
        argv.push_back(word.data());

    try {
        return generate_impl(this, argv.size(), argv.data(), infoOnly);
    } catch (std::exception &e) {
        return Result(e.what());
    }
}

namespace async {

Job::Job() {
    product = detectDevice(true);
    generator = GetGenerator(product);
}

void Job::launch() {
    // Reset the generator
    generator = GetGenerator(product);
    launchTime = std::chrono::steady_clock::now();
    future = std::async(std::launch::async, [this]() {
        return generator->generate(kernel);
      });
}

Result Job::finish() {
    // Job is already complete
    if (future.valid()) {
        auto timeout = std::chrono::milliseconds(5000);
        if (auto e = ::getenv("FRONTEND_TIMEOUT_MS")) {
            timeout = std::chrono::milliseconds(atoi(e));
        }
        // Budget covers time since launch, not just time spent waiting here.
        auto elapsedSinceLaunch = std::chrono::steady_clock::now() - launchTime;
        auto remaining = timeout - std::chrono::duration_cast<std::chrono::milliseconds>(elapsedSinceLaunch);
        if (remaining < std::chrono::milliseconds(0))
            remaining = std::chrono::milliseconds(0);
        auto status = future.wait_for(remaining);
        if (status == std::future_status::ready) {
            try {
                Result res = future.get();
                clear();
                return res;
            } catch (std::exception &e) {
                std::cout << "Error during future collection: " << e.what() << std::endl;
                throw;
            }
        } else {
            generator->cancelAutoSWSB();
            future.wait();
            std::string tstatus = status == std::future_status::ready ? "ready"
                                : status == std::future_status::deferred ? "deferred"
                                : status == std::future_status::timeout ? "timeout"
                                : "other";
            std::string formatted_ms = std::to_string(timeout.count()) + "ms";
            clear();
            return Result("Frontend timeout (" + tstatus + "): " + formatted_ms);
        }
    }
    return Result("Frontend Job never started");
}

Job::~Job() {
    try {
        finish();
    } catch (std::exception &e) {
        // Swallow exceptions
    }
}

} // namespace async
} // namespace frontend
