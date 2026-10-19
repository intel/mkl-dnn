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

#include "tools/kernel_catalog.hpp"
#include "gemmstone/driver_info.hpp"
#include "tools/utils.hpp"
#include "tools/kernel_modifiers.hpp"
#include "internal/utils.hpp"

#include <algorithm>
#include <istream>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <unordered_set>

using namespace gemmstone;

namespace {

using utils::expect;

struct InternPool {
    std::mutex mutex;
    std::unordered_set<std::string> strings;
};

InternPool &internPool()
{
    static InternPool *const pool = new InternPool;
    return *pool;
}

bool isInterned(const char *value)
{
    if (!value) return false;
    auto &pool = internPool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    auto it = pool.strings.find(value);
    return it != pool.strings.end() && it->c_str() == value;
}

void deserializeField(std::istream &s, int &i)
{
    if (!(s >> i))
        throw std::runtime_error("could not parse integer in catalog");
}

void deserializeField(std::istream &s, char &c)
{
    expect(s, '\'');
    if (!(s >> c))
        throw std::runtime_error("could not parse character in catalog");
    expect(s, '\'');
}

void deserializeField(std::istream &s, const char *&str)
{
    expect(s, '"');
    std::string value;
    if (!std::getline(s, value, '"'))
        throw std::runtime_error("could not parse string in catalog");
    str = catalog::intern(value);
}

template <typename T, int n>
void deserializeField(std::istream &s, T (&array)[n])
{
    expect(s, '{');
    deserializeField(s, array[0]);
    for (int i = 1; i < n; i++) {
        expect(s, ',');
        deserializeField(s, array[i]);
    }
    expect(s, '}');
}

void deserializeField(std::istream &s, kcatalog::Selector &sel)
{
    expect(s, '{');
    deserializeField(s, sel.hw);
    expect(s, ',');
    deserializeField(s, sel.kernelType);
    expect(s, ',');
    deserializeField(s, sel.precisions);
    expect(s, ',');
    deserializeField(s, sel.layouts);
    expect(s, '}');
}

void deserializeField(std::istream &s, kcatalog::Restrictions &r)
{
    expect(s, '{');
    deserializeField(s, r.steppingMin);
    expect(s, ',');
    deserializeField(s, r.steppingMax);
    expect(s, ',');
    deserializeField(s, r.acceptSizesMin);
    expect(s, ',');
    deserializeField(s, r.acceptSizesMax);
    expect(s, ',');
    deserializeField(s, r.allowedSizesMin);
    expect(s, ',');
    deserializeField(s, r.allowedSizesMax);
    expect(s, ',');
    deserializeField(s, r.alignment);
    expect(s, ',');
    deserializeField(s, r.tags);
    expect(s, '}');
}

void deserializeField(std::istream &s, kcatalog::Model &m)
{
    expect(s, '{');
    expect(s, '\'');
    if (!(s >> m.id))
        throw std::runtime_error("could not parse model ID in catalog");
    expect(s, '\'');
    expect(s, ',');
    deserializeField(s, m.paramCount);
    if (m.paramCount < 0 || m.paramCount > kcatalog::MaxParamCount)
        throw std::runtime_error("invalid parameter count in catalog model");
    expect(s, ',');
    expect(s, '{');
    for (int i = 0; i < m.paramCount; i++) {
        if (i > 0) expect(s, ',');
        if (!(s >> m.params[i]))
            throw std::runtime_error("could not parse model parameter in catalog");
    }
    expect(s, '}');
    expect(s, '}');
}

void deserializeField(std::istream &s, kcatalog::Entry &e)
{
    expect(s, '{');
    deserializeField(s, e.selector);
    expect(s, ',');
    deserializeField(s, e.restrictions);
    expect(s, ',');
    deserializeField(s, e.strategy);
    expect(s, ',');
    e.driverInfo = gemmstone::CommonDriverInfo::undump(s, false);
    expect(s, ',');
    deserializeField(s, e.model);
    expect(s, '}');
}

void serializeField(std::ostream &s, int i)
{
    s << i;
}

void serializeField(std::ostream &s, char c)
{
    s << '\'' << c << '\'';
}

void serializeField(std::ostream &s, const char *str)
{
    s << '"' << str << '"';
}

template <typename T, int n>
void serializeField(std::ostream &s, const T (&array)[n])
{
    s << '{';
    serializeField(s, array[0]);
    for (int i = 1; i < n; i++) {
        s << ", ";
        serializeField(s, array[i]);
    }
    s << '}';
}

void serializeField(std::ostream &s, const kcatalog::Selector &sel)
{
    s << '{';
    serializeField(s, sel.hw);
    s << ", ";
    serializeField(s, sel.kernelType);
    s << ", ";
    serializeField(s, sel.precisions);
    s << ", ";
    serializeField(s, sel.layouts);
    s << "}";
}

void serializeField(std::ostream &s, const kcatalog::Restrictions &r)
{
    s << "{";
    s << r.steppingMin << ", " << r.steppingMax << ", ";
    serializeField(s, r.acceptSizesMin);
    s << ", ";
    serializeField(s, r.acceptSizesMax);
    s << ", ";
    serializeField(s, r.allowedSizesMin);
    s << ", ";
    serializeField(s, r.allowedSizesMax);
    s << ", ";
    serializeField(s, r.alignment);
    s << ", ";
    serializeField(s, r.tags);
    s << '}';
}

void serializeField(std::ostream &s, const kcatalog::Model &m)
{
    s << '{';
    serializeField(s, m.id);
    s << ", " << m.paramCount << ", {";
    if (m.paramCount > 0) s << m.params[0];
    for (int i = 1; i < m.paramCount; i++)
        s << ", " << m.params[i];
    s << "}}";
}

void serializeField(std::ostream &s, const kcatalog::Entry &e)
{
    s << '{';
    serializeField(s, e.selector);
    s << ", ";
    serializeField(s, e.restrictions);
    s << ", ";
    serializeField(s, e.strategy);
    s << ", ";
    e.driverInfo.dump(s, false);
    s << ", ";
    serializeField(s, e.model);
    s << "}";
}

} // namespace

namespace catalog {

std::string modifierTags(const std::string &modifiers)
{
    utils::KernelModifiers parsed(modifiers);
    std::string tags;
    if (parsed.batch != BatchMode::None) {
        tags += kcatalog::ReqBatch;
        if (parsed.batch != BatchMode::Strided) tags += kcatalog::ReqBatchN;
    }
    if (parsed.sumA) tags += kcatalog::ReqSumA;
    if (parsed.sumB) tags += kcatalog::ReqSumB;
    if (parsed.aOffset.enabled || parsed.bOffset.enabled) tags += kcatalog::ReqABOffset;
    if ((parsed.aOffset.enabled && parsed.aOffset.ptrDims > 0)
            || (parsed.bOffset.enabled && parsed.bOffset.ptrDims > 0))
        tags += kcatalog::ReqOffsetMultiDim;
    if (parsed.aScale.grouped() || parsed.bScale.grouped()) tags += kcatalog::ReqScale2D;
    return tags;
}

const char *intern(const std::string &value)
{
    auto &pool = internPool();
    std::lock_guard<std::mutex> lock(pool.mutex);
    return pool.strings.insert(value).first->c_str();
}

void validate(const kcatalog::Entry &entry)
{
    gemm_assert(isInterned(entry.selector.kernelType), "kernel type is not interned");
    gemm_assert(isInterned(entry.restrictions.tags), "restriction tags are not interned");
    gemm_assert(isInterned(entry.strategy), "strategy is not interned");
    for (int i = 0; i < 3; i++) {
        gemm_assert(isInterned(entry.selector.precisions[i]), "precision is not interned");
        gemm_assert(isInterned(entry.selector.layouts[i]), "layout is not interned");
    }
}

char tag(ngen::HW hw)
{
    using namespace kcatalog;
    switch (hw) {
        case ngen::HW::Gen12LP: return HWTagGen12LP;
        case ngen::HW::XeHPG:  return HWTagXeHPG;
        case ngen::HW::XeHPC:  return HWTagXeHPC;
        case ngen::HW::Xe2:    return HWTagXe2;
        case ngen::HW::Xe3:    return HWTagXe3;
        case ngen::HW::Xe3p:   return HWTagXe3p;
        default:               return '?';
    }
}

void addDefaultRestrictions(kcatalog::Entry &entry)
{
    using utils::contains;

    std::string tags = entry.restrictions.tags;

    if (!entry.driverInfo.support4GB[0]) tags += kcatalog::ReqNo64BitA;
    if (!entry.driverInfo.support4GB[1]) tags += kcatalog::ReqNo64BitB;
    if (!entry.driverInfo.support4GB[2]) tags += kcatalog::ReqNo64BitC;

    if (entry.driverInfo.kParallel())
        tags += kcatalog::ReqNoBatch;

    if (contains(entry.strategy, "sys"))
        tags += kcatalog::ReqSystolic;

    // Note we only get 'extra' on a freshly added entry, with information from frontend.
    // The catalog does not store 'extra'.
    if (contains(entry.driverInfo.extra, "a2d"))
        tags += kcatalog::ReqBlock2DA;
    if (contains(entry.driverInfo.extra, "b2d"))
        tags += kcatalog::ReqBlock2DB;
    if (contains(entry.driverInfo.extra, "c2d"))
        tags += kcatalog::ReqBlock2DC;

    if (entry.selector.hw == kcatalog::HWTagXeHPC) {
        if (contains(entry.selector.precisions[0], 'T') || contains(entry.selector.precisions[1], 'T')) {
            entry.restrictions.steppingMin = 5; /* PVC-XT B0 */
        }
    }

    auto extPrecisionBytes = [](const kcatalog::string &precision) {
        std::string_view view(precision);
        char ext = view.at(0);
        if (ext == '[')
            ext = view.at(1);
        auto type = charPrecision(ext);
        return double(type.paddedSize()) / double(type.perByte());
    };

    if (entry.selector.hw == kcatalog::HWTagXeHPC) {
        bool noXe2 = false;
        if (contains(tags, kcatalog::ReqBlock2DA) && entry.selector.layouts[0][0] == 'N')
            noXe2 |= (int(entry.driverInfo.unroll[LoopM] * extPrecisionBytes(entry.selector.precisions[0])) % 16) != 0;
        if (contains(tags, kcatalog::ReqBlock2DB) && entry.selector.layouts[1][0] == 'T')
            noXe2 |= (int(entry.driverInfo.unroll[LoopN] * extPrecisionBytes(entry.selector.precisions[1])) % 16) != 0;
        if (noXe2)
            tags += kcatalog::ReqNoXe2Block2D;
    }

    auto wgTileM = entry.driverInfo.wgTile(LoopM);
    auto wgTileN = entry.driverInfo.wgTile(LoopN);

    std::sort(tags.begin(), tags.end());
    entry.restrictions.tags = intern(tags);
}

void serialize(std::ostream &s, const std::vector<kcatalog::Entry> &entries, const std::string &header)
{
    s << header;
    s << Sentinel << "\nauto _CATALOG_ = kcatalog::toArray({\n";

    bool first = true;
    for (const auto &e: entries) {
        if (!first)
            s << ",\n";
        first = false;
        serializeField(s, e);
    }

    s << "\n});\n";
}

std::vector<kcatalog::Entry> deserialize(std::istream &s, std::string &header)
{
    header.clear();

    std::string line;
    while (true) {
        if (!s)
            throw std::runtime_error("could not find sentinel at beginning of catalog");
        std::getline(s, line);
        if (line == Sentinel)
            break;
        header += line;
        header += '\n';
    }

    std::getline(s, line); // consume "auto _CATALOG_ = kcatalog::toArray({" line

    std::vector<kcatalog::Entry> entries;
    while (true) {
        s >> std::ws;
        if (!s || s.eof())
            throw std::runtime_error("could not parse catalog");

        if (s.peek() == '}') break;

        if (!entries.empty()) expect(s, ',');

        kcatalog::Entry e = {};
        deserializeField(s, e);
        validate(e);
        entries.push_back(std::move(e));
    }

    expect(s, '}');
    expect(s, ')');
    expect(s, ';');

    return entries;
}

} // namespace catalog
