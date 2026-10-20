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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_KERNEL_CATALOG_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_KERNEL_CATALOG_HPP

#include <string>
#include <cstring>
#include <tuple>
#include <array>
#include <utility>

#include "gemmstone/config.hpp"
#include "gemmstone/driver_info.hpp"

GEMMSTONE_NAMESPACE_START

namespace kcatalog {

using string = const char *;

struct Restrictions {
    int steppingMin;                               // If >= 0, minimum supported stepping (inclusive)
    int steppingMax;                               // If >= 0, maximum supported stepping (exclusive)
    int acceptSizesMin[3];                         // m/n/k ranges where kernel always accepted.
    int acceptSizesMax[3];                         //   (see kernel_evaluator.cpp for more details)
    int allowedSizesMin[3];                        // m/n/k ranges outside of which kernel always rejected.
    int allowedSizesMax[3];
    int alignment[3];                              // A/B/C alignment requirements.
    string tags;                                    // see RestrictionTags for entries.

    friend bool operator==(const Restrictions &r1, const Restrictions &r2)
    {
        bool equal = (r1.steppingMin == r2.steppingMin)
                  && (r1.steppingMax == r2.steppingMax)
                  && (std::strcmp(r1.tags, r2.tags) == 0);

        for (int i = 0; i < 3; i++) {
            equal &= (r1.acceptSizesMin[i]  == r2.acceptSizesMin[i])
                  && (r1.acceptSizesMax[i]  == r2.acceptSizesMax[i])
                  && (r1.allowedSizesMin[i] == r2.allowedSizesMin[i])
                  && (r1.allowedSizesMax[i] == r2.allowedSizesMax[i])
                  && (r1.alignment[i]       == r2.alignment[i]);
        }

        return equal;
    }
};

// Member initializers would prevent C++11 aggregate initialization of static catalogs.
constexpr Restrictions defaultRestrictions()
{
    return {-1, -1, {-1, -1, -1}, {-1, -1, -1},
            {-1, -1, -1}, {-1, -1, -1}, {1, 1, 1}, ""};
}

enum RestrictionTags : char {
    ReqDisabled = '@',
    ReqAlignFallback = '#',
    ReqBlock2DA = 'A',       ReqNoBlock2DA = 'a',
    ReqBlock2DB = 'B',       ReqNoBlock2DB = 'b',
    ReqBlock2DC = 'C',       ReqNoBlock2DC = 'c',
    Req64BitA = 'X',         ReqNo64BitA = 'x',
    Req64BitB = 'Y',         ReqNo64BitB = 'y',
    Req64BitC = 'Z',         ReqNo64BitC = 'z',
    ReqBatch = 'V',          ReqNoBatch = 'v',
    ReqBatchMultiDim = 'W',  ReqNoBatchMultiDim = 'w',
    ReqSumA = 'Q',           ReqNoSumA = 'q',
    ReqSumB = 'P',           ReqNoSumB = 'p',
    ReqABOffset = 'R',       ReqNoABOffset = 'r',
    ReqOffsetMultiDim = 'O', ReqNoOffsetMultiDim = 'o',
    ReqSystolic = 'I',       ReqNoSystolic = 'i',
    ReqIntegrated = 'H',     ReqNoIntegrated = 'h',
    ReqDiscrete = 'J',       ReqNoDiscrete = 'j',
    ReqBatchN = 'N',         ReqNoBatchN = 'n',
    ReqCustom1 = 'D',        ReqNoCustom1 = 'd',
    ReqXe2Block2D = 'G',     ReqNoXe2Block2D = 'g',
    ReqScale2D = 'K',        ReqNoScale2D = 'k',
};

enum HWTags : char {
    HWTagGen12LP = 'C',
    HWTagXeHPG = 'E',
    HWTagXeHPC = 'F',
    HWTagXe2 = 'G',
    HWTagXe3 = 'H',
    HWTagXe3p = 'I',
};

struct Selector {
    char hw;    // see HWTags for entries
    string kernelType;
    string precisions[3];
    string layouts[3];

    friend bool operator<(const Selector &sel1, const Selector &sel2) {
        auto tupleize = [](const Selector &sel) {
            bool compoundA = sel.precisions[0][0] == '[';
            bool compoundB = sel.precisions[1][0] == '[';
            char A_ext = sel.precisions[0][compoundA ? 1 : 0];
            char A     = sel.precisions[0][compoundA ? 2 : 0];
            char B_ext = sel.precisions[1][compoundB ? 1 : 0];
            char B     = sel.precisions[1][compoundB ? 2 : 0];
            return std::make_tuple(sel.hw,
                                   A_ext & 0x1F, A & 0x1F,
                                   B_ext & 0x1F, B & 0x1F,
                                   sel.layouts[0][0], sel.layouts[1][0]);
        };
        return tupleize(sel1) < tupleize(sel2);
    };
    friend bool operator>(const Selector &sel1, const Selector &sel2) { return sel2 < sel1; }
    friend bool operator<=(const Selector &sel1, const Selector &sel2) { return !(sel2 < sel1); }
    friend bool operator>=(const Selector &sel1, const Selector &sel2) { return !(sel1 < sel2); }

    std::string str(const int (&alignments)[3]) const {
        std::string result{hw};
        result.append(" ");
        result.append(kernelType);
        result.append(" ");
        for (auto &p : precisions) {
            result.append(p);
        }
        result.append(" ");
        for (int i = 0; i < 3; i++) {
            result.append(layouts[i]);
            if (alignments[i] != 1) {
                result.append("@");
                result.append(std::to_string(alignments[i]));
            }
        }
        return result;
    }
};

enum : int {
    // Model 'W' parameters
    ParamWPriority = 0,
    ParamWCount,

    // Model 'S' parameters
    ParamS_Cm0 = 0,     // Minimum constant overhead, beta = 0
    ParamS_Cm1,         // Minimum constant overhead, beta = 1
    ParamS_C00,         // Overhead per partial wave, constant coefficient, beta = 0
    ParamS_C01,         // Overhead per partial wave, constant coefficient, beta = 1
    ParamS_C10,         // Overhead per partial wave, linear coefficient, beta = 0
    ParamS_C11,         // Overhead per partial wave, linear coefficient, beta = 1
    ParamS_Ma,          // A per-element load cost
    ParamS_Mb,          // B per-element load cost
    ParamS_Ef,          // Peak efficiency, full waves
    ParamS_Ep0,         // Peak efficiency, partial wave, constant coefficient
    ParamS_Ep1,         // Peak efficiency, partial wave, linear coefficient
    ParamS_Em,          // Load balancing weight factor (0 = ignore load balancing, 1 = full weight for load balancing term)
    ParamS_Fp,          // Max sustained frequency ratio nominal freq/actual freq.
    ParamS_Fr0,         // FMA count at which frequency starts dropping.
    ParamS_Fr1,         // FMA count at which frequency stops dropping.
    ParamSCount,

    // Model 'E' parameters
    ParamE_C0 = 0,      // Constant overhead per partial wave, constant coefficient
    ParamE_C1,          // Constant overhead per partial wave, linear coefficient
    ParamE_Ck0,         // Per-wgK overhead per partial wave, constant coefficient
    ParamE_Ck1,         // Per-wgK overhead per partial wave, linear coefficient
    ParamE_Cb0,         // Fused beta overhead per partial wave, constant coefficient
    ParamE_Cb1,         // Fused beta overhead per partial wave, linear coefficient
    ParamE_Ma,          // A per-element load cost
    ParamE_Mb,          // B per-element load cost
    ParamE_Mc,          // C per-element update cost (store/atomic add only)
    ParamE_Mcu,         // C per-element update cost (load + store)
    ParamE_Ef,          // Peak efficiency, full waves
    ParamE_Ep0,         // Peak efficiency, partial wave, constant coefficient
    ParamE_Ep1,         // Peak efficiency, partial wave, linear coefficient
    ParamE_Em,          // Load balancing weight factor (0 = ignore load balancing, 1 = full weight for load balancing term)
    ParamE_Fp,          // Max sustained frequency ratio nominal freq/actual freq.
    ParamE_Fr0,         // FMA count at which frequency starts dropping.
    ParamE_Fr1,         // FMA count at which frequency stops dropping.
    ParamECount,

    ParamE_Cp0 = ParamE_Cb1,    // Fused post-op overhead per partial wave, constant coefficient

    // Maximum possible parameter count
    MaxParamCount = ParamECount,
};

struct Model {
    char id;
    int paramCount;
    double params[MaxParamCount];
};

struct Entry {
    Selector selector;
    Restrictions restrictions;
    string strategy;
    CommonDriverInfo driverInfo;
    Model model;

    friend bool operator<(const Entry &e1, const Entry &e2)   { return e1.selector <  e2.selector; }
    friend bool operator>(const Entry &e1, const Entry &e2)   { return e1.selector >  e2.selector; }
    friend bool operator<=(const Entry &e1, const Entry &e2)  { return e1.selector <= e2.selector; }
    friend bool operator>=(const Entry &e1, const Entry &e2)  { return e1.selector >= e2.selector; }
    friend bool operator<(const Entry &e, const Selector &s)  { return e.selector <  s; }
    friend bool operator>(const Entry &e, const Selector &s)  { return e.selector >  s; }
    friend bool operator<=(const Entry &e, const Selector &s) { return e.selector <= s; }
    friend bool operator>=(const Entry &e, const Selector &s) { return e.selector >= s; }
    friend bool operator<(const Selector &s, const Entry &e)  { return s <  e.selector; }
    friend bool operator>(const Selector &s, const Entry &e)  { return s >  e.selector; }
    friend bool operator<=(const Selector &s, const Entry &e) { return s <= e.selector; }
    friend bool operator>=(const Selector &s, const Entry &e) { return s >= e.selector; }

    std::string str() const {
        std::string result = selector.str(restrictions.alignment);
        result.append(" ");
        result.append(std::to_string(driverInfo.unroll[LoopM]));
        result.append(" ");
        result.append(std::to_string(driverInfo.unroll[LoopN]));
        result.append(" ");
        result.append(strategy);
        return result;
    }
};

struct Catalog {
    template <std::size_t N>
    Catalog(const std::array<Entry, N> &a): entryCount(N), entries(a.data()) {}
    int entryCount;
    Catalog(int count, const Entry *data): entryCount(count), entries(data) {}
    const Entry *entries;
};

#if __cplusplus >= 202002L && __cpp_lib_to_array >= 201907L
template <std::size_t N>
constexpr std::array<Entry, N> toArray(Entry (&&a)[N]) {
    return std::to_array(std::move(a));
}
#else
#if __cplusplus >= 201402L && __cpp_lib_integer_sequence >= 201304L
template <std::size_t... I> using IndexSequence = std::index_sequence<I...>;
template <std::size_t N> using MakeIndexSequence = std::make_index_sequence<N>;
#else
// C++11 fallback for std::make_index_sequence
template <std::size_t... I> struct IndexSequence {};
template <typename S1, typename S2> struct ConcatSequence;
template <std::size_t... I1, std::size_t... I2>
struct ConcatSequence<IndexSequence<I1...>, IndexSequence<I2...>> {
    using type = IndexSequence<I1..., (sizeof...(I1) + I2)...>;
};
template <std::size_t N> struct MakeIndexSequenceImpl {
    using type = typename ConcatSequence<
            typename MakeIndexSequenceImpl<N / 2>::type,
            typename MakeIndexSequenceImpl<N - N / 2>::type>::type;
};
template <> struct MakeIndexSequenceImpl<0> { using type = IndexSequence<>; };
template <> struct MakeIndexSequenceImpl<1> { using type = IndexSequence<0>; };
template <std::size_t N>
using MakeIndexSequence = typename MakeIndexSequenceImpl<N>::type;
#endif

template <std::size_t N, std::size_t... I>
constexpr std::array<Entry, N>
toArrayImpl(const Entry (&a)[N], IndexSequence<I...>) {
    return {{a[I]...}};
}
template <std::size_t N>
constexpr std::array<Entry, N> toArray(Entry (&&a)[N]) {
    return toArrayImpl(a, MakeIndexSequence<N>{});
}
#endif

} /* namespace kcatalog */

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
