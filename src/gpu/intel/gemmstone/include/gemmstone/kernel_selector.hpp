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

#ifndef GEMMSTONE_INCLUDE_GEMMSTONE_KERNEL_SELECTOR_HPP
#define GEMMSTONE_INCLUDE_GEMMSTONE_KERNEL_SELECTOR_HPP

#include "gemmstone/config.hpp"
#include "gemmstone/kernel_catalog.hpp"
#include "gemmstone/kernel_evaluator.hpp"

#include <algorithm>
#include <functional>

GEMMSTONE_NAMESPACE_START

// Basic kernel selection API.
struct StrategyRequirement {
    enum Parameter { UnrollM, UnrollN, WGTileM, WGTileN, WGTileMN, WGM, WGN, WGK, WG } param;
    enum Relation { Equals, AtLeast, AtMost } relation;
    int value;

    StrategyRequirement(Parameter param_, Relation relation_, int value_) : param(param_), relation(relation_), value(value_) {}

    template <typename T> friend StrategyRequirement operator==(Parameter param_, T value_) { return StrategyRequirement(param_, Equals,  int(value_)); }
    template <typename T> friend StrategyRequirement operator<=(Parameter param_, T value_) { return StrategyRequirement(param_, AtMost,  int(value_)); }
    template <typename T> friend StrategyRequirement operator>=(Parameter param_, T value_) { return StrategyRequirement(param_, AtLeast, int(value_)); }

    void transpose();
};

struct MatchParamsBase
{
    kcatalog::Selector selector;
    SizeParams sizes;
    char precisionCExt = 0;
    bool ignoreSizes = false;
    bool ignoreCase = false;
    bool ReqBDPASDims = false;
    int stepping = 0;
    int alignment[3] = {0, 0, 0};
    int unrollReq[3] = {1, 1, 1};
    kcatalog::string tags, lateTags;
    int nExtraReqs = 0;
    const StrategyRequirement *extraReqs = nullptr;

    MatchParamsBase() = default;
    MatchParamsBase(ngen::HW hw, bool systolicAvailable, const ngen::Product &product, const GEMMProblem &problem);

protected:
    std::array<char, 40> temp;
};

struct MatchParams : public MatchParamsBase
{
    MatchParams() = default;
    MatchParams(ngen::HW hw, bool systolicAvailable, const ngen::Product &product, const GEMMProblem &problem)
            : MatchParamsBase(hw, systolicAvailable, product, problem) {}

    // NOLINTNEXTLINE(bugprone-copy-constructor-init)
    MatchParams(const MatchParams &other) { *this = other; }
    MatchParams &operator=(const MatchParams &other) {
        static_cast<MatchParamsBase &>(*this) = other;

        auto transfer = [&](const char *&value) {
            for (size_t offset = 0; offset < temp.size(); offset++) {
                if (value == other.temp.data() + offset) {
                    value = temp.data() + offset;
                    break;
                }
            }
        };

        transfer(selector.precisions[0]);
        transfer(selector.precisions[1]);
        transfer(selector.precisions[2]);
        transfer(selector.layouts[0]);
        transfer(selector.layouts[1]);
        transfer(selector.layouts[2]);
        transfer(tags);
        transfer(lateTags);

        return *this;
    }
};

using SelectionObserver = std::function<void (const kcatalog::Entry *entry, double score, EvaluateAuxOutput aux)>;

const std::vector<const kcatalog::Entry *> select(const kcatalog::Catalog &catalog, const MatchParams &pattern, const EvaluateParams &eparams, EvaluateAuxOutput &aux, SelectionObserver *observer = nullptr);
const std::vector<const kcatalog::Entry *> select(const kcatalog::Catalog &catalog, int npatterns, const MatchParams *patterns, const EvaluateParams &eparams, EvaluateAuxOutput &aux, SelectionObserver *observer = nullptr);

// Extended API for iterating over all matching kernels.
bool matches(const kcatalog::Entry &e, const MatchParams &pattern);
bool lessAligned(int alignA1, int alignB1, int alignA2, int alignB2);

const kcatalog::Entry *lower_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector);
const kcatalog::Entry *upper_bound(const kcatalog::Catalog &catalog, const kcatalog::Selector &selector);

class EntryIterator {
public:
    EntryIterator(const kcatalog::Catalog &catalog_, const MatchParams &pattern_)
        : catalog(catalog_), pattern(pattern_), begin(lower_bound(catalog_, pattern_.selector)), end(upper_bound(catalog_, pattern_.selector)), current(begin) {
        findNextMatch();
    }

    operator bool() const { return current < end; }

    EntryIterator &operator++() {
        ++current;
        findNextMatch();
        return *this;
    }

    EntryIterator operator++(int) {
        auto old = *this;
        operator++();
        return old;
    }

    const kcatalog::Entry &operator*()  const { return  *current; }
    const kcatalog::Entry *operator->() const { return &*current; }

    friend bool operator==(const EntryIterator &i1, const EntryIterator &i2) {
        return (i1.current == i2.current);
    }
    friend bool operator!=(const EntryIterator &i1, const EntryIterator &i2) {
        return !(i1 == i2);
    }

protected:
    const kcatalog::Catalog &catalog;
    MatchParams pattern;
    const kcatalog::Entry *begin, *end, *current;

    void findNextMatch() {
        for (; current < end; current++) {
            if (matches(*current, pattern))
                break;
        }
    }
};

inline EntryIterator match(const kcatalog::Catalog &catalog, const MatchParams &pattern)
{
    return EntryIterator(catalog, pattern);
}

GEMMSTONE_NAMESPACE_END

#endif /* header guard */
