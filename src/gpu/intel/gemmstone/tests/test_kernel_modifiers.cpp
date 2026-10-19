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

#include "tools/utils.hpp"
#include "tools/frontend.hpp"
#include "tools/kernel_catalog.hpp"
#include "tools/kernel_modifiers.hpp"
#include "tools/benchstone/kernel_info.hpp"
#include "tests/test_common.hpp"

#include <algorithm>
#include <sstream>

namespace {

KernelInfo problemInfo(const std::string &problem)
{
    gemmstone::CommonDriverInfo driver = {};
    driver.problem = problem;
    driver.unroll[gemmstone::LoopM] = driver.unroll[gemmstone::LoopN] = 8;
    std::ostringstream stream;
    driver.dump(stream);
    return KernelInfo(stream.str());
}

utils::KernelType kernelType(const std::string &description)
{
    std::istringstream stream(description);
    return utils::readKernelType([&]() {
        std::string word;
        if (!(stream >> word)) throw std::runtime_error("Missing kernel type");
        return word;
    });
}

} // namespace

int main()
{
    bool tests_pass = true;
    using utils::KernelModifier;

    for (const auto *word : {"rc", "rcab", "super", "superll",
                            "fused", "batchs", "batchn", "batchnv", "offsetab",
                            "scalea[g32]", "suma", "sumb", "bias", "hpl"})
        EXPECT(utils::kernelModifier(word) != KernelModifier::None);
    for (const auto *word : {"gemm", "ugemm", "gemmlh", "trsm3llu", "symmll", "copy",
                            "as1", "unknown", "batch", "not_offset", "summary", "test", "persistent"})
        EXPECT(utils::kernelModifier(word) == KernelModifier::None);

    utils::KernelModifiers parsed;
    EXPECT(!parsed.readCheck && !parsed.superkernel && !parsed.fused);
    EXPECT(parsed.batch == gemmstone::BatchMode::None);
    EXPECT(!parsed.aOffset.enabled && !parsed.bOffset.enabled && !parsed.cOffset.enabled);
    EXPECT(!parsed.aScale.enabled && !parsed.bScale.enabled);
    EXPECT(!parsed.sumA && !parsed.sumB && !parsed.bias && !parsed.hpl);
    utils::KernelModifiers empty("");
    EXPECT(!empty.aScale.enabled && !empty.aOffset.enabled && !empty.sumA);
    utils::KernelModifiers constructed("batchs rcab suma scalea[g32,H]");
    EXPECT(constructed.batch == gemmstone::BatchMode::Strided && constructed.sumA);
    EXPECT(constructed.readCheckA && constructed.readCheckB && !constructed.readCheckC);
    EXPECT(constructed.aScale.grouped() && constructed.aScale.T == gemmstone::Type::f16);
    EXPECT((constructed.aScale.groupSize == std::array<int, 2>{1, 32}));
    EXPECT(!parsed.parse("gemm"));
    EXPECT(parsed.parse("scalea[g32,H]"));
    EXPECT(parsed.aScale.grouped() && !parsed.bScale.enabled);
    EXPECT((parsed.aScale.groupSize == std::array<int, 2>{1, 32}));
    EXPECT(parsed.aScale.T == gemmstone::Type::f16);
    EXPECT(parsed.parse("scaleb[G64x2,S]"));
    EXPECT(parsed.bScale.grouped() && parsed.bScale.precalc);
    EXPECT((parsed.bScale.groupSize == std::array<int, 2>{64, 2}));
    EXPECT(parsed.bScale.T == gemmstone::Type::f32);
    EXPECT(parsed.parse("offsetb[g64,O]a[v,H]c[p,S]"));
    EXPECT(parsed.aOffset.enabled && parsed.aOffset.ptrDims == 1);
    EXPECT(parsed.aOffset.T == gemmstone::Type::f16);
    EXPECT((parsed.bOffset.groupSize == std::array<int, 2>{64, 1}));
    EXPECT(parsed.cOffset.enabled && parsed.cOffset.ptrDims == 0);
    for (const auto *word : {"batchs", "rcab", "superll", "fused", "suma", "sumb", "bias", "hpl"})
        EXPECT(parsed.parse(word));
    EXPECT(parsed.batch == gemmstone::BatchMode::Strided);
    EXPECT(parsed.readCheck && parsed.readCheckA && parsed.readCheckB && !parsed.readCheckC);
    EXPECT(parsed.superkernel && parsed.fused && parsed.sumA && parsed.sumB && parsed.bias && parsed.hpl);
    EXPECT(parsed.parse("rc"));
    EXPECT(parsed.readCheckA && parsed.readCheckB && parsed.readCheckC);
    EXPECT(parsed.parse("offseta"));
    EXPECT(parsed.aOffset.enabled && parsed.aOffset.ptrDims == -1);
    EXPECT(parsed.aOffset.T == gemmstone::Type::invalid && !parsed.aOffset.precalc);
    EXPECT(parsed.parse("offsetb[V,O]"));
    EXPECT(parsed.bOffset.precalc && parsed.bOffset.ptrDims == 1);
    EXPECT(parsed.bOffset.T == gemmstone::Type::s8);
    EXPECT(parsed.parse("scalea[p]"));
    EXPECT(parsed.aScale.enabled && !parsed.aScale.grouped());
    EXPECT(parsed.bScale.grouped());

    for (const auto *word : {"scale", "offset", "scalez[g32]", "scalec[g32]", "offsetd",
                            "scalea[", "scalea[]", "offseta[g]", "scalea[g0]", "scalea[g-1]",
                            "scalea[g99999999999999999999]", "scalea[g32x]", "scalea[g32,S",
                            "scalea[g32,]", "scalea[g32,?]", "scalea[g32,Sx]", "scalea[g32]b[bad]",
                            "offseta[g0]", "offseta[p,?]", "offseta[g32]b[bad]", "rcz"}) {
        bool rejected = false;
        try { parsed.parse(word); }
        catch (const std::runtime_error &) { rejected = true; }
        EXPECT(rejected);
        EXPECT(parsed.aScale.ptrDims == 0 && parsed.aScale.T == gemmstone::Type::invalid);
        EXPECT((parsed.bScale.groupSize == std::array<int, 2>{64, 2}));
        EXPECT(parsed.aOffset.ptrDims == -1 && parsed.aOffset.T == gemmstone::Type::invalid);
        EXPECT(parsed.bOffset.precalc && parsed.bOffset.T == gemmstone::Type::s8);
    }

    EXPECT(kernelType("gemm").base == "gemm");
    EXPECT(kernelType("gemm").modifiers.empty());
    EXPECT(kernelType("scaleb[g32] suma offseta batchs scalea[g32] gemm").base == "gemm");
    EXPECT(kernelType("scaleb[g32] suma offseta batchs scalea[g32] gemm").modifiers
            == "scaleb[g32] suma offseta batchs scalea[g32]");
    EXPECT(kernelType("hpl bias sumb suma rc superll ugemm").base == "ugemm");
    EXPECT(kernelType("hpl bias sumb suma rc superll ugemm").modifiers
            == "hpl bias sumb suma rc superll");
    for (const auto *base : {"gemm", "symmll", "trsm3llu", "ugemm"}) {
        auto type = kernelType(std::string("offseta suma scalea[g32] ") + base + " SSS NNN");
        EXPECT(type.base == base);
        EXPECT(type.modifiers == "offseta suma scalea[g32]");
    }
    bool rejected = false;
    try { kernelType("offseta scalea[g32]"); }
    catch (const std::runtime_error &) { rejected = true; }
    EXPECT(rejected);
    rejected = false;
    try { utils::KernelModifiers invalid("not-a-modifier"); }
    catch (const std::runtime_error &) { rejected = true; }
    EXPECT(rejected);

    // Exercise every ordering, including separate offset/scale tokens and both sums.
    std::vector<std::string> modifiers = {
        "offseta[g1x32,S]", "offsetb[g32x1,S]",
        "scalea[g1x32,S]", "scaleb[g32x1,S]", "suma", "sumb"
    };
    std::sort(modifiers.begin(), modifiers.end());
    do {
        std::string prefix;
        for (const auto &modifier : modifiers) prefix += modifier + ' ';
        auto type = kernelType(prefix + "gemm");
        EXPECT(type.base == "gemm");
        EXPECT(type.modifiers + ' ' == prefix);
        utils::KernelModifiers parameters(type.modifiers);
        EXPECT(parameters.aOffset.grouped() && parameters.bOffset.grouped());
        EXPECT(parameters.aScale.grouped() && parameters.bScale.grouped());
        EXPECT((parameters.aScale.groupSize == std::array<int, 2>{1, 32}));
        EXPECT((parameters.bScale.groupSize == std::array<int, 2>{32, 1}));
        auto tags = catalog::modifierTags(type.modifiers);
        for (char tag : {gemmstone::kcatalog::ReqABOffset, gemmstone::kcatalog::ReqOffsetMultiDim,
                         gemmstone::kcatalog::ReqScale2D, gemmstone::kcatalog::ReqSumA,
                         gemmstone::kcatalog::ReqSumB})
            EXPECT(utils::contains(tags, tag));
        auto info = problemInfo(prefix + "gemm SSS NNN 1 0").l3.at(0);
        EXPECT(info.aOffset == 'g' && info.bOffset == 'g');
        EXPECT(info.scaleA && info.scaleB && info.sumA && info.sumB);
        EXPECT((info.aqGroup == std::array<int, 2>{1, 32}));
        EXPECT((info.bqGroup == std::array<int, 2>{32, 1}));
        EXPECT(info.precisionAScale == 'S' && info.precisionBScale == 'S');
        EXPECT(info.precisionA == 'S' && info.layoutA == 'N');
        EXPECT(info.alpha == 1 && info.beta == 0);
    } while (std::next_permutation(modifiers.begin(), modifiers.end()));

    auto batchTags = catalog::modifierTags("batchn batchs batchnv");
    EXPECT(utils::contains(batchTags, gemmstone::kcatalog::ReqBatch));
    EXPECT(utils::contains(batchTags, gemmstone::kcatalog::ReqBatchN));
    EXPECT(!utils::contains(catalog::modifierTags("batchnv batchs"), gemmstone::kcatalog::ReqBatchN));
    EXPECT(catalog::modifierTags("offsetc bias rc super").empty());
    EXPECT(!utils::contains(catalog::modifierTags("offseta[g32] offseta"),
            gemmstone::kcatalog::ReqOffsetMultiDim));

    for (const auto *scales : {"scalea[g32]", "scaleb[g32x1,S]", "scalea[g32]b",
                              "scaleab[g32]", "scalea[s]b[g32]", "scalea[g32]b[s]",
                              "scalea[G32]", "scaleb[g32]a[s]", "scalea[s] scaleb[g32]",
                              "scalea[g32] scaleb[s]", "scalea[g32]b[g32] scalea[s]"})
        EXPECT(utils::contains(catalog::modifierTags(scales), gemmstone::kcatalog::ReqScale2D));
    for (const auto *scales : {"scalea", "scaleb", "scaleab", "scalea[s]", "scaleb[p,S]",
                              "scalea[v]b[s]", "scalea[s] scaleb[v]",
                              "scalea[g32] scalea[s]", "scalea[g32]b[g32] scaleab"})
        EXPECT(!utils::contains(catalog::modifierTags(scales), gemmstone::kcatalog::ReqScale2D));
    for (const auto *offsets : {"offseta[v]", "offsetb[G32x1,S]", "offsetb[v]a[s]",
                               "offseta[g32]c", "offseta[g32]b[v] offseta[s]"}) {
        auto tags = catalog::modifierTags(offsets);
        EXPECT(utils::contains(tags, gemmstone::kcatalog::ReqABOffset));
        EXPECT(utils::contains(tags, gemmstone::kcatalog::ReqOffsetMultiDim));
    }
    for (const auto *offsets : {"offseta", "offsetab[p,S]", "offsetb[s]a[p]",
                               "offseta[g32]b[v] offsetab", "offsetc[g32]"}) {
        auto tags = catalog::modifierTags(offsets);
        EXPECT(!utils::contains(tags, gemmstone::kcatalog::ReqOffsetMultiDim));
    }
    EXPECT(catalog::modifierTags("offsetc[g32]").empty());
    for (const auto *word : {"scalea[g32", "scalez[g32]", "scalec[g32]",
                             "offseta[g32", "offsetz[v]"}) {
        rejected = false;
        try { catalog::modifierTags(word); }
        catch (const std::runtime_error &) { rejected = true; }
        EXPECT(rejected);
    }

    auto shorthand = problemInfo("scaleb[g32,H] offseta[g16,S] scalea[g16,S] gemm SSS NNN 1 0").l3.at(0);
    EXPECT((shorthand.aqGroup == std::array<int, 2>{1, 16}));
    EXPECT((shorthand.bqGroup == std::array<int, 2>{32, 1}));
    EXPECT(shorthand.precisionAScale == 'S' && shorthand.precisionBScale == 'H');
    auto overwritten = problemInfo("scalea[g32,S] scalea[g64,H] gemm SSS NNN 1 0").l3.at(0);
    EXPECT((overwritten.aqGroup == std::array<int, 2>{1, 64}));
    EXPECT(overwritten.precisionAScale == 'H');

    for (const auto *type : {"gemm", "gemml", "gemmuh", "ugemm", "trsm3llu", "symmll"}) {
        auto info = problemInfo(std::string("hpl bias sumb suma batchnv superll rcab ") + type
                + " SSS NNN 1 0").l3.at(0);
        EXPECT(info.hpl && info.cBias && info.sumA && info.sumB);
        EXPECT(info.batch == 'v' && info.superkernel && info.readCheck);
        EXPECT(info.precisionA == 'S');
        if (std::string(type) == "ugemm") EXPECT(!info.triangular);
        if (std::string(type) == "gemmuh") EXPECT(info.triangular && info.hermitian && !info.lower);
        if (std::string(type) == "trsm3llu") EXPECT(info.trsm && !info.inplace && info.left && info.lower && info.unit);
        if (std::string(type) == "symmll") EXPECT(info.symm && info.left && info.lower);
    }

    ngen::Product product;
    for (auto family : {ngen::ProductFamily::GenericXeHPC, ngen::ProductFamily::GenericGen12LP,
                        ngen::ProductFamily::GenericXeHPG, ngen::ProductFamily::GenericXeHP,
                        ngen::ProductFamily::GenericXe2, ngen::ProductFamily::GenericXe3,
                        ngen::ProductFamily::GenericXe3p,
    }) {
        if (frontend::Generator::IsSupported(ngen::getCore(family))) {
            product.family = family;
            break;
        }
    }
    EXPECT(product.family != ngen::ProductFamily::Unknown);
    if (product.family == ngen::ProductFamily::Unknown) return 1;
    auto generate = [=](const std::string &description) {
        auto generator = frontend::Generator::Create(product);
        return generator->generate(description, true);
    };
    const std::string suffix = "gemm SSS NNN 8 8 1 0 as1 as1 as";
    auto baseline = generate(suffix);
    EXPECT(baseline.success);
    auto whitespace = generate("\tgemm\nSSS\tNNN 8 8 1 0 as1 as1 as\n");
    EXPECT(whitespace.success && whitespace.info == baseline.info);

    for (const auto &prefix : {
            std::string("batchs offseta offsetb "),
            std::string("offsetb batchs offseta ")}) {
        auto result = generate(prefix + suffix);
        if (!result.success) std::fprintf(stderr, "%s\n", result.info.c_str());
        EXPECT(result.success);
        if (result.success) {
            auto info = KernelInfo(result.info).l3.at(0);
            EXPECT(info.batch == 's' && info.aOffset && info.bOffset);
        }
    }
    auto canonical = generate("batchs offsetab " + suffix);
    auto reordered = generate("offsetb batchs offseta " + suffix);
    EXPECT(canonical.success && reordered.success && canonical.info == reordered.info);
    auto repeated = generate("offseta[p,S] batchn offseta batchs offsetb " + suffix);
    EXPECT(repeated.success && repeated.info == canonical.info);

    modifiers = {"offseta", "scalea[g1x32,S]", "scaleb[g32x1,S]", "suma"};
    std::sort(modifiers.begin(), modifiers.end());
    std::string expected;
    do {
        std::string prefix;
        for (const auto &modifier : modifiers) prefix += modifier + ' ';
        auto result = generate(prefix + suffix);
        if (!result.success) std::fprintf(stderr, "%s\n", result.info.c_str());
        EXPECT(result.success);
        if (result.success) {
            if (expected.empty()) expected = result.info;
            EXPECT(result.info == expected);
            auto info = KernelInfo(result.info).l3.at(0);
            EXPECT(info.aOffset && info.scaleA && info.scaleB && info.sumA);
        }
    } while (std::next_permutation(modifiers.begin(), modifiers.end()));

    EXPECT(!generate("offseta scalea[g32]").success);
    EXPECT(!generate(" \t\n").success);
    EXPECT(!generate("scalez[g32] " + suffix).success);
    EXPECT(!generate("offsetz " + suffix).success);
    EXPECT(!generate("test " + suffix).success);
    EXPECT(!generate("persistent " + suffix).success);
    auto persistent = generate(suffix + " pt");
    EXPECT(persistent.success);
    if (persistent.success) {
        auto info = KernelInfo(persistent.info).l3.at(0);
        EXPECT((info.loopOrder[0] & gemmstone::LoopPersistent) != 0);
    }

    return tests_pass ? 0 : 1;
}
