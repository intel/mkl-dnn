/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#ifndef GEMMSTONE_TOOLS_MULTIOPT_DATA_HPP
#define GEMMSTONE_TOOLS_MULTIOPT_DATA_HPP

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "client.hpp"
#include "serialization.hpp"

namespace multiopt {

/* Pre-indexed and hashed problem/strategy vectors */
template <typename V>
struct Labeled {
    uint32_t index, hash;
    V vec;

    friend bool operator==(Labeled<V> &l1, Labeled<V> &l2) {
        return (l1.hash == l2.hash) && (l1.vec == l2.vec);
    }

    operator V&()                       { return vec; }
    operator const V&()           const { return vec; }
    auto &operator[](ptrdiff_t i)       { return vec[i]; }
    auto  operator[](ptrdiff_t i) const { return vec[i]; }

    friend Serialized &operator<<(Serialized &s, const Labeled<V> &l) {
        return s << l.index << l.hash << l.vec;
    }
    friend Serialized &operator>>(Serialized &s, Labeled<V> &l) {
        return s >> l.index >> l.hash >> l.vec;
    }
};

} /* namespace multiopt */

namespace std {
template <typename V>
struct hash<multiopt::Labeled<V>> {
    size_t operator()(const multiopt::Labeled<V> &lv) const { return lv.hash; }
};
} /* namespace std */

namespace multiopt {

/* Vector-like structure with O(1) membership query */
template <typename T>
class mvector {
    std::unordered_set<T> all;
    std::vector<const T*> v;

public:
    size_t size()                 const { return v.size(); }
    bool empty()                  const { return size() == 0; }

    auto begin()                        { return all.begin(); }
    auto begin()                  const { return all.cbegin(); }

    auto end()                          { return all.end(); }
    auto end()                    const { return all.cend(); }

    void clear()                        { v.clear(); all.clear(); }
    void reserve(size_t count)          { v.reserve(count); all.reserve(count); }

    const T &operator[](size_t i) const { return *v[i]; }

    bool contains(const T &t)     const { return all.count(t) > 0; }

    const T &insert(const T &t) {
        auto result = all.insert(t);
        if (result.second)
            v.push_back(&*result.first);
        return *result.first;
    }

    friend Serialized &operator<<(Serialized &s, const mvector<T> &v) {
        s << v.size();
        for (size_t i = 0; i < v.size(); i++)
            s << v[i];
        return s;
    }
    friend Serialized &operator>>(Serialized &s, mvector<T> &v) {
        v.clear();
        size_t size;
        s >> size;
        for (; size > 0; size--) {
            T t;
            s >> t;
            v.insert(t);
        }
        return s;
    }
};

struct Varianced {
    float value;
    float variance;

    operator float() const { return value; }
};

/* Per-PV data storage container */
struct PVData {
    struct Entry {
        uint32_t sv;
        Varianced perf;
    };

    float roofline = 0.0f;
    float absRoofline = 0.0f;
    int topIndex = -1;

    std::vector<uint32_t> neighbors;    /* neighboring PVs -- NB: neighbor relation is transitive */
    std::vector<Entry> data;

    inline const Entry &topData() const;

    friend Serialized &operator<<(Serialized &s, const PVData &d) {
        return s << d.roofline << d.absRoofline << d.topIndex << d.neighbors << d.data;
    }
    friend Serialized &operator>>(Serialized &s, PVData &d) {
        return s >> d.roofline >> d.absRoofline >> d.topIndex >> d.neighbors >> d.data;
    }
};

/* Main data storage for multiopt */
template <typename PV, typename SV>
class DataStore {
    template <typename V>
    struct LabelReferencer {
        uint32_t index;
        void *list;
    };

    Client<PV, SV> &client;

    mvector<Labeled<PV>> pvs;
    mvector<Labeled<SV>> svs;
    std::vector<PVData> pvData;

    std::vector<bool> svsWithData;
    std::unordered_map<uint32_t, uint32_t> spreadSVList;    // (sv# -> max pv#)

    int emptyPVs = 0;
    int nonEmptySVs = 0;

    /* Sensitivity: variances of abs(log2(perf ratio)) arising from
                    changing SV by 1 in each dimension */
    std::array<float, SV::size()> sensitivities;

    /* Data for computing sensitivities */
    std::vector<double> sdiffs, sobs2;
    int nextSensitivityCalc;
    int lastSensitivityRank = 0;

    /* Derived rank-revealing basis for sampled strategy deltas. */
    mutable std::array<std::array<double, SV::size()>, SV::size()> sensitivityBasis {};
    mutable size_t sensitivityBasisSamples = 0;
    mutable int sensitivityBasisRank = 0;

    /* Neighborhood sizes */
    float neighborRadius = 0.f;
    float neighborUmbra = 0.f;

    /* Other settings */
    float atticSpace = 1000.0f;

    /* Non-persistent settings (not serialized) */
    int frozenPVs = 0;

    void updateSensitivityBasis(const double *diff) const;
    void updateSensitivityBasis() const;

public:
    friend Serialized &operator<<(Serialized &s, const DataStore<PV, SV> &d) {
        return s << d.pvs << d.svs << d.pvData << d.svsWithData << d.spreadSVList << d.emptyPVs << d.nonEmptySVs
                 << d.sensitivities << d.sdiffs << d.sobs2 << d.nextSensitivityCalc << d.lastSensitivityRank
                 << d.neighborRadius << d.neighborUmbra << d.atticSpace;
    }

    friend Serialized &operator>>(Serialized &s, DataStore<PV, SV> &d) {
        return s >> d.pvs >> d.svs >> d.pvData >> d.svsWithData >> d.spreadSVList >> d.emptyPVs >> d.nonEmptySVs
                 >> d.sensitivities >> d.sdiffs >> d.sobs2 >> d.nextSensitivityCalc >> d.lastSensitivityRank
                 >> d.neighborRadius >> d.neighborUmbra >> d.atticSpace;
    }

    explicit DataStore(Client<PV, SV> &client_);

    Client<PV, SV> &getClient() { return client; }

    const Labeled<PV> &registerPV(const PV &pv);
    const Labeled<SV> &registerSV(const SV &sv);

    const mvector<Labeled<PV>> &allPVs() const { return pvs; }
    const mvector<Labeled<SV>> &allSVs() const { return svs; }

    std::unordered_map<uint32_t, uint32_t> &spreadSVs() { return spreadSVList; }

    PVData &find(const Labeled<PV> &pv);
    const PVData &find(const Labeled<PV> &pv) const;
    const PVData &find(int index) const;

    void add(Varianced perf, const Labeled<PV> &pv, const Labeled<SV> &sv);
    void add(Varianced perf, Varianced prevPerf, const Labeled<PV> &pv, const Labeled<SV> &sv, const Labeled<SV> &prevSV);

    float sensitivity(int i) const { return sensitivities[i]; }
    void setInitialSensitivity(int i, float s);
    bool sensitivityProbeIncreasesRank(const SV &sv, const SV &prevSV) const;

    float perfVariance(const SV &sv1, const SV &sv2) const;
    Varianced estimateSuboptimality(const Labeled<PV> &pv, const SV &sv, bool absolute = false) const;

    float suboptimality(const Labeled<PV> &lpv, bool absolute = false) const;
    float suboptimality(const Labeled<PV> &lpv, float perf, bool absolute = false) const;

    bool anyAbsolutelySuboptimal() const;

    float roofline(const PV &pv);
    void reroof(const Labeled<PV> &lpv);

    float getAtticSpace() const     { return atticSpace; }
    void setAtticSpace(float space) { atticSpace = space; }

    void initializeNeighbors();

    // Neighbors of `pv` within radius = umbraFactor * neighborUmbra.
    std::vector<uint32_t> wideNeighbors(const Labeled<PV> &pv, float umbraFactor) const;

    int unpopulatedPVs() const { return emptyPVs; }
    int populatedSVs() const { return nonEmptySVs; }

    void freezePVs(int npv) { frozenPVs = std::min(npv, int(pvs.size())); }
    int getFrozenPVs() const { return frozenPVs; }

    bool sensitivitiesAvailable() const;
    float sensitivitySamplingProgress() const;

protected:
    void findNeighbors(const Labeled<PV> &pv);
    void updateRoofline(float &relPerf, const PV &src, const Labeled<PV> &ref,
                        float neighborRadius, float neighborUmbra);
};

} /* namespace multiopt */

#endif
