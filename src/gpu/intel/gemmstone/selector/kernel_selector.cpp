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

#include "gemmstone/kernel_selector.hpp"
#include "gemmstone/kernel_evaluator.hpp"

#include <cassert>
#include <cctype>
#include <cstring>
#include <vector>

GEMMSTONE_NAMESPACE_START

inline bool layoutMatch(const char *lref, const char *lpattern)
{
    return (lref[0] == lpattern[0]);        // This is a sufficient check for now.
}

inline bool precisionMatch(const char *pref, const char *ppattern, bool ignoreCase)
{
    char mask = ignoreCase ? ~0x20 : ~0;
    auto match = [mask](char c1, char c2) { return (c1 & mask) == (c2 & mask); };

    bool ok = false;
    ok = ok || (ppattern[0] == '?');
    ok = ok || match(pref[0], ppattern[0]);
    ok = ok || (ppattern[0] == '[' && match(pref[0], ppattern[1]));
    if (ok && pref[0] == '[') {
        ok = ok && match(pref[1], ppattern[1]) && match(pref[2], ppattern[2]);
        for (int i = 3; pref[i] != '\0'; i++) {
            if (pref[i] != ppattern[i]) { ok = false; break; }
        }
    }
    return ok;
}

inline bool precisionMinimumMatch(const char *pref, char pmin)
{
    uint8_t sizeTable[0x20] = {
//         A  B  C  D  E  F  G  H  I  J  K  L  M  N  O
        0, 0, 2, 8, 8, 0, 0, 0, 2, 4, 4, 4, 0, 0, 0, 1,
//      P  Q  R  S  T  U  V  W  X  Y  Z
        0, 1, 0, 4, 4, 8, 0, 2, 0, 0, 16, 0, 0, 0, 0, 0
    };

    return (sizeTable[pref[0] & 0x1F] >= sizeTable[pmin & 0x1F]);
}

inline bool alignmentMatch(int aref, int apattern)
{
    if (aref == 0) aref = 1;
    return (apattern % aref == 0);
}

inline bool tagMatch(const char *tref, const char *tpattern)
{
    for (auto c = tref; *c; c++) {
        // Lowercase tags -> must not match pattern
        // Uppercase tags -> must match pattern
        int cu = *c & ~0x20;     // toupper(c)
        bool match = (std::strchr(tpattern, cu) != nullptr);
        bool wantMatch = (*c & 0x20) == 0;
        if (match != wantMatch)
            return false;
    }
    return true;
}

inline bool strategyMatch(const CommonDriverInfo &info, const StrategyRequirement &req)
{
    int actual = 0;
    switch (req.param) {
        case StrategyRequirement::UnrollM:  actual = info.unroll[LoopM]; break;
        case StrategyRequirement::UnrollN:  actual = info.unroll[LoopN]; break;
        case StrategyRequirement::WGTileM:  actual = info.wgTile(LoopM); break;
        case StrategyRequirement::WGTileN:  actual = info.wgTile(LoopN); break;
        case StrategyRequirement::WGTileMN: actual = info.wgTile(LoopM) * info.wgTile(LoopN); break;
        case StrategyRequirement::WGM:      actual = info.wg[LoopM]; break;
        case StrategyRequirement::WGN:      actual = info.wg[LoopN]; break;
        case StrategyRequirement::WGK:      actual = info.wg[LoopK]; break;
        case StrategyRequirement::WG:       actual = info.wg[LoopM] * info.wg[LoopN] * info.wg[LoopK]; break;
        default: return false;
    }

    switch (req.relation) {
        case StrategyRequirement::Equals:  return (actual == req.value);
        case StrategyRequirement::AtLeast: return (actual >= req.value);
        case StrategyRequirement::AtMost:  return (actual <= req.value);
        default: return false;
    }
}

bool matches(const kcatalog::Entry &e, const MatchParams &p)
{
    bool ok = true;

    if (e.restrictions.steppingMin >= 0)
        ok = ok && (p.stepping >= e.restrictions.steppingMin);
    if (e.restrictions.steppingMax >= 0)
        ok = ok && (p.stepping < e.restrictions.steppingMax);
    ok = ok && layoutMatch(e.selector.layouts[0], p.selector.layouts[0]);
    ok = ok && layoutMatch(e.selector.layouts[1], p.selector.layouts[1]);
    ok = ok && layoutMatch(e.selector.layouts[2], p.selector.layouts[2]);
    ok = ok && precisionMatch(e.selector.precisions[2], p.selector.precisions[2], p.ignoreCase);
    if (p.precisionCExt)
        ok = ok && precisionMinimumMatch(e.selector.precisions[2], p.precisionCExt);
    for (int i = 0; i < 3; i++)
        ok = ok && alignmentMatch(e.restrictions.alignment[i], p.alignment[i]);
    ok = ok && tagMatch(e.restrictions.tags, p.tags);

    if (!p.ignoreSizes) {
        int64_t mnk[3] = {p.sizes.m, p.sizes.n, p.sizes.k};
        for (int i = 0; i < 3; i++) {
            if (e.restrictions.allowedSizesMin[i] >= 0)
                ok = ok && (mnk[i] >= e.restrictions.allowedSizesMin[i]);
            if (e.restrictions.allowedSizesMax[i] >= 0)
                ok = ok && (mnk[i] <= e.restrictions.allowedSizesMax[i]);
        }
    }

    ok = ok && (e.driverInfo.unroll[LoopM] % p.unrollReq[LoopM] == 0);
    ok = ok && (e.driverInfo.unroll[LoopN] % p.unrollReq[LoopN] == 0);
    ok = ok && (e.driverInfo.unroll[LoopK] % p.unrollReq[LoopK] == 0);

    for (int i = 0; i < p.nExtraReqs; i++)
        ok = ok && strategyMatch(e.driverInfo, p.extraReqs[i]);

    // Should already be matched.
    ok = ok && (e.selector.hw == p.selector.hw);
    ok = ok && precisionMatch(e.selector.precisions[0], p.selector.precisions[0], p.ignoreCase);
    ok = ok && precisionMatch(e.selector.precisions[1], p.selector.precisions[1], p.ignoreCase);

    return ok;
}

// Check whether one set of A/B alignments (alignA1, alignB1) is strictly less aligned than
//  another (alignA2, alignB2).
bool lessAligned(int alignA1, int alignB1, int alignA2, int alignB2)
{
    alignA1 = std::max(alignA1, 4);
    alignA2 = std::max(alignA2, 4);
    alignB1 = std::max(alignB1, 4);
    alignB2 = std::max(alignB2, 4);
    return (alignA1 <= alignA2) && (alignB1 <= alignB2) && (alignA1 + alignB1 < alignB1 + alignB2);
}

struct EntryData {
    EntryData(const kcatalog::Entry *entry_, double score_) : entry(entry_), score(score_) {}
    const kcatalog::Entry *entry;
    double score;
};

// Inner kernel selection logic.
// Choose the best entry, if any, matching one of the given patterns.
const std::vector<const kcatalog::Entry *> getEntries(const kcatalog::Catalog &catalog, int npatterns, const MatchParams *patterns, const EvaluateParams &eparams, EvaluateAuxOutput &aux,  SelectionObserver * observer)
{
    // TODO: omit evaluation if only one match, if aux output not needed.
    std::vector<EntryData> keys;
    for (int ipattern = 0; ipattern < npatterns; ipattern++) {
        for (auto it = match(catalog, patterns[ipattern]); it; it++) {
            // Late tag checking. If late tags do not match, we skip entry.
            if (tagMatch(it->restrictions.tags, patterns[ipattern].lateTags)) {
                auto score = evaluate(*it, eparams, aux);
                keys.emplace_back(&*it, score);
                if (observer) {
                    (*observer)(&*it, score, aux);
                }
            }
        }
    }

    // u3 problems also match int4 ("[FO]") patterns; prefer dedicated u3
    // entries over those, since int4 performance models underestimate u3 cost.
    auto isU3Entry = [](const kcatalog::Entry *e) {
        for (int i = 0; i < 2; i++) {
            const char *p = e->selector.precisions[i];
            char c = (p[0] == '[') ? p[1] : p[0];
            if ((c & ~0x20) == 'K') return true;
        }
        return false;
    };

    auto less = [&](const EntryData &lhs, const EntryData &rhs){
                      bool lhsFallback = (lhs.entry->restrictions.tags[0] == kcatalog::ReqAlignFallback);
                      int  lhsAlignA = std::max(lhs.entry->restrictions.alignment[0], 4);
                      int  lhsAlignB = std::max(lhs.entry->restrictions.alignment[1], 4);
                      bool rhsFallback = (rhs.entry->restrictions.tags[0] == kcatalog::ReqAlignFallback);
                      int  rhsAlignA = std::max(rhs.entry->restrictions.alignment[0], 4);
                      int  rhsAlignB = std::max(rhs.entry->restrictions.alignment[1], 4);
                      if (rhsFallback && lessAligned(rhsAlignA, rhsAlignB, lhsAlignA, lhsAlignB)) return true;
                      if (lhsFallback && lessAligned(lhsAlignA, lhsAlignB, rhsAlignA, rhsAlignB)) return false;
                      bool lhsU3 = isU3Entry(lhs.entry), rhsU3 = isU3Entry(rhs.entry);
                      if (lhsU3 != rhsU3) return lhsU3;
                      if (lhs.score < rhs.score) return true;
                      if (lhs.score > rhs.score) return false;
                      return (lhs.entry < rhs.entry);
    };
    std::sort(keys.begin(), keys.end(), less);

    // Unpack into vector of entries (dropping score)
    std::vector<const kcatalog::Entry *> entries;
    entries.reserve(keys.size());
    for (auto &key : keys)
        entries.push_back(key.entry);

    return entries;
}

// User-facing kernel selection logic.
// Includes architecture and data type fallbacks.
const std::vector<const kcatalog::Entry *> select(const kcatalog::Catalog &catalog, const MatchParams &pattern, const EvaluateParams &eparams, EvaluateAuxOutput &aux, SelectionObserver *observer)
{
    return select(catalog, 1, &pattern, eparams, aux, observer);
}

const std::vector<const kcatalog::Entry *> select(const kcatalog::Catalog &catalog, int npatterns, const MatchParams *patterns, const EvaluateParams &eparams, EvaluateAuxOutput &aux, SelectionObserver *observer)
{
    using namespace kcatalog;

    std::vector<const kcatalog::Entry *> entries;
    if (npatterns == 0 || !patterns)
        return entries;

    auto result = getEntries(catalog, npatterns, patterns, eparams, aux, observer);
    entries.insert(entries.end(), result.begin(), result.end());

    // Architecture fallback loop.
    bool first = true;
    auto hw = patterns[0].selector.hw;
    do {
        std::vector<MatchParams> modPatterns;
        modPatterns.reserve(npatterns);
        for (int i = 0; i < npatterns; i++) {
            modPatterns.emplace_back(patterns[i]);
            modPatterns.back().selector.hw = hw;
        }

        // Type fallback loop.
        while (true) {
            if (!first) {
                auto res =  getEntries(catalog, npatterns, modPatterns.data(), eparams, aux, observer);
                entries.insert(entries.end(), res.begin(), res.end());
            }
            first = false;

            /* Try capital types */
            auto hasLowercase = [](kcatalog::string &str) {
                bool lc = false;
                for (auto c = str; *c; c++)
                    lc |= (*c >= 'a' && *c <= 'z');
                return lc;
            };

            bool changed = false;
            for (auto &p: modPatterns) {
                if (p.ignoreCase) continue;
                p.ignoreCase = true;
                for (int d = 0; d < 3; d++)
                    changed |= hasLowercase(p.selector.precisions[d]);
            }
            if (changed) continue;

            /* Specific pattern modifications */
            auto iequal = [](kcatalog::string &str, char c) {
                return str[0] && ((str[0] & ~0x20) == (c & ~0x20)) && (str[1] == '\0');
            };

            for (auto &p: modPatterns) {
                auto match = [&](const char *precisions) {
                    for (int i = 0; i < 3; i++)
                        if (precisions[i] && !iequal(p.selector.precisions[i], precisions[i]))
                            return false;
                    return true;
                };

                // u3 normally unpacks via planInt3Upconvert, falling back
                // to u4 ('f') strategies. If jit/pd.cpp instead promotes
                // u3 to int8 ("[kO]"), match "[FO]"-style int8 strategies.
                auto isU3 = [](kcatalog::string str) {
                    char c = (str[0] == '[') ? str[1] : str[0];
                    return c && ((c & ~0x20) == 'K');
                };
                auto fallbackU3 = [&](kcatalog::string &str) {
                    if (!isU3(str)) return false;
                    if (str[0] == '[') {
                        switch (str[2]) {
                            // Prefer the dedicated int4-as-int8 bracketed
                            // strategies (e.g. "[FO]") over plain "O"/"o":
                            // u3 unpacks to int8 the same way int4 does.
                            case 'O': str = "[FO]"; return true;
                            case 'o': str = "[Fo]"; return true;
                            default: break;
                        }
                    }
                    str = "f";
                    return true;
                };
                bool u3Fallback = false;
                u3Fallback |= fallbackU3(p.selector.precisions[0]);
                u3Fallback |= fallbackU3(p.selector.precisions[1]);
                if (u3Fallback) {
                    changed = true;
                    continue;
                }

                if (match("FO"))
                    p.selector.precisions[0] = "[FO]";
                else if (match("BB"))
                    p.selector.precisions[0] = p.selector.precisions[1] = "H";
                else continue;

                changed = true;
            }
            if (changed) continue;

            break;
        }

        // Architecture fallbacks, e.g. Xe2 inherits XeHPC strategies.
        switch (hw) {
            case HWTagXe2:  hw = HWTagXeHPC; break;
            case HWTagXe3:  hw = HWTagXe2; break;
            case HWTagXe3p: hw = HWTagXe3; break;
            default:        hw = 0; break;
        }
    } while (hw);

    return entries;
}

template <bool upper>
const kcatalog::Entry *upper_lower_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector)
{
    int n = catalog.entryCount;
    const kcatalog::Entry *cur = catalog.entries;

    while (n > 0) {
        auto half = n >> 1;
        auto mid = cur + half;
        if (upper ? (*mid <= selector) : (*mid < selector)) {
            cur = mid + 1;
            n = n - half - 1;
        } else
            n = half;
    }

    return cur;
}

const kcatalog::Entry *lower_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector) {
    return upper_lower_bound<false>(catalog, selector);
}

const kcatalog::Entry *upper_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector) {
    return upper_lower_bound<true>(catalog, selector);
}


MatchParamsBase::MatchParamsBase(ngen::HW hw, bool systolicAvailable, const ngen::Product &product, const GEMMProblem &problem_)
{
    using namespace kcatalog;

    auto problem = problem_;

    if(problem.Tao.isSubByte() || problem.Ta_scale.isSubByte()){
        unrollReq[LoopM] = std::max(problem.Tao.perByte(), problem.Ta_scale.perByte());
    }
    if(problem.Tbo.isSubByte() || problem.Tb_scale.isSubByte()){
        unrollReq[LoopN] = std::max(problem.Tbo.perByte(), problem.Tb_scale.perByte());
    }
    if(problem.Ta_scale.is3()){
        unrollReq[LoopM] = 8;
    }
    if(problem.Tb_scale.is3()){
        unrollReq[LoopN] = 8;
    }

    ReqBDPASDims = problem.preferBDPAS();

    if (ReqBDPASDims) {
        unrollReq[LoopM] = 8;
        unrollReq[LoopN] = 8;
    }

    if(problem.hasCMXScale()){
        unrollReq[LoopM] = 32;
    }

    switch (hw) {
        default: assert(!"Unknown architecture");
        case ngen::HW::Gen12LP: selector.hw = kcatalog::HWTagGen12LP; break;
        case ngen::HW::XeHPG:   selector.hw = kcatalog::HWTagXeHPG;   break;
        case ngen::HW::XeHPC:   selector.hw = kcatalog::HWTagXeHPC;   break;
        case ngen::HW::Xe2:     selector.hw = kcatalog::HWTagXe2;     break;
        case ngen::HW::Xe3:     selector.hw = kcatalog::HWTagXe3;     break;
        case ngen::HW::Xe3p:    selector.hw = kcatalog::HWTagXe3p;    break;
    }

    auto &C = problem.C;
    auto equivCLayout = C.layout;
    if (isPacked(equivCLayout)) {
        bool colMajor = (C.layout == MatrixLayout::Pc) ^ (C.crosspack * problem.Tc > 4);
        equivCLayout = (colMajor ? MatrixLayout::N : MatrixLayout::T);
    }

    auto makeABConvert = [](Type T, Type T_ext, bool mixed_fp, char *out) {
        if ((mixed_fp && T_ext.isSubsetOf(T)) || (T == T_ext))
            out[0] = precisionChar(T_ext);
        else {
            out[0] = '[';
            out[1] = precisionChar(T_ext);
            out[2] = precisionChar(T);
            out[3] = ']';
        }
    };

    selector.kernelType = "gemm";

    std::fill(temp.begin(), temp.end(), '\0');

    const bool mixed_fp = problem.Ta_ext.isFP() != problem.Tb_ext.isFP();
    makeABConvert(problem.Ta, problem.Ta_ext, mixed_fp, &temp[0]);
    makeABConvert(problem.Tb, problem.Tb_ext, mixed_fp, &temp[5]);
    temp[10] = precisionChar(problem.Tc);
    temp[12] = layoutChar(problem.A.layout);
    temp[14] = layoutChar(problem.B.layout);
    temp[16] = layoutChar(equivCLayout);

    selector.precisions[0] = &temp[0];
    selector.precisions[1] = &temp[5];
    selector.precisions[2] = &temp[10];
    selector.layouts[0] = &temp[12];
    selector.layouts[1] = &temp[14];
    selector.layouts[2] = &temp[16];

    precisionCExt = precisionChar(problem.Tc_ext);

    alignment[0] = problem.A.alignment;
    alignment[1] = problem.B.alignment;
    alignment[2] = problem.C.alignment;

    char *tagPtr = &temp[18];
    lateTags = tagPtr;

    // Late-only tags. Don't choose lower-performing kernels
    //  just to fuse reductions. Instead do reductions in a separate kernel.
    if (problem.sumA)
        *tagPtr++ = ReqSumA;
    if (problem.sumB)
        *tagPtr++ = ReqSumB;

    tags = tagPtr;

    if (systolicAvailable)
        *tagPtr++ = ReqSystolic;

    if (product.type == ngen::PlatformType::Integrated)
        *tagPtr++ = ReqIntegrated;
    if (product.type == ngen::PlatformType::Discrete)
        *tagPtr++ = ReqDiscrete;

    if (problem.batch != BatchMode::None) {
        *tagPtr++ = ReqBatch;
        if (problem.batchDims > 1)
            *tagPtr++ = ReqBatchMultiDim;
    }

    if (problem.aOffset != ABOffset::None || problem.bOffset != ABOffset::None)
        *tagPtr++ = ReqABOffset;
    if (problem.aoPtrDims > 0 || problem.boPtrDims > 0)
        *tagPtr++ = ReqOffsetMultiDim;

    problem.autoTypeConversions(systolicAvailable);
    if (problem.needsASums() && !problem.sumA) *tagPtr++ = ReqSumA;
    if (problem.needsBSums() && !problem.sumB) *tagPtr++ = ReqSumB;

    if (one_of(hw, {ngen::HW::Xe2, ngen::HW::Xe3, ngen::HW::Xe3p})) *tagPtr++ = ReqXe2Block2D;

    if (problem.aScale2D() || problem.bScale2D()) *tagPtr++ = ReqScale2D;

    assert(tagPtr < temp.data() + temp.size());  // leave room for the string terminator

    sizes.batch = sizes.m = sizes.n = sizes.k = 0;
}

void StrategyRequirement::transpose()
{
    switch (param) {
        case UnrollM: param = UnrollN; break;
        case UnrollN: param = UnrollM; break;
        case WGTileM: param = WGTileN; break;
        case WGTileN: param = WGTileM; break;
        case WGM:     param = WGN;     break;
        case WGN:     param = WGM;     break;
        default:                       break;
    }
}

GEMMSTONE_NAMESPACE_END
