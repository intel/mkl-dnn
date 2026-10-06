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

#include "data.hpp"
#include "settings.hpp"

#ifndef GEMMSTONE_WITH_MKL
#include "mini_mkl.hpp"
#elif __has_include("mkl/mkl.h")
#include "mkl/mkl.h"
#else
#include "mkl.h"
#endif

#include <algorithm>
#include <fstream>
#include <iostream>

namespace multiopt {

template <typename PV, typename SV>
DataStore<PV, SV>::DataStore(Client<PV, SV> &client_) : client(client_)
{
    auto &minLegal = client.minLegalSV();
    auto &maxLegal = client.maxLegalSV();

    nextSensitivityCalc = Settings::minSensitivitySamples * SV::size();
    sdiffs.reserve(nextSensitivityCalc * SV::size());
    sobs2.reserve(nextSensitivityCalc);

    std::fill(sensitivities.begin(), sensitivities.end(), std::numeric_limits<float>::quiet_NaN());

    /* Special handling for SV entries that cannot be altered.
       Set sensitivities for such entries to zero, and add pseudo-observations to ensure sdiffs has full rank. */
    for (size_t i = 0; i < SV::size(); i++) if (minLegal[i] == maxLegal[i]) {
        sensitivities[i] = 0.f;
        sobs2.push_back(0.);

        auto off = sdiffs.size();
        sdiffs.resize(sdiffs.size() + SV::size());
        for (size_t j = 0; j < SV::size(); j++)
            sdiffs[off + j] = (i == j);
    }
    updateSensitivityBasis();
    lastSensitivityRank = sensitivityBasisRank;
}

static inline uint32_t reduceHash(size_t h)
{
    return (h & 0xFFFFFFFF) ^ (h >> 32);
}

template <typename PV, typename SV>
void DataStore<PV, SV>::updateSensitivityBasis(const double *diff) const
{
    if (sensitivityBasisRank == SV::size()) return;
    std::array<double, SV::size()> residual;
    double norm0 = 0.;
    for (int i = 0; i < SV::size(); i++) {
        residual[i] = diff[i];
        norm0 += residual[i] * residual[i];
    }

    for (int i = 0; i < sensitivityBasisRank; i++) {
        double projection = 0.;
        for (int j = 0; j < SV::size(); j++)
            projection += residual[j] * sensitivityBasis[i][j];
        for (int j = 0; j < SV::size(); j++)
            residual[j] -= projection * sensitivityBasis[i][j];
    }

    double norm = 0.;
    for (auto value: residual)
        norm += value * value;
    if (norm <= norm0 * 1e-12)
        return;

    norm = std::sqrt(norm);
    for (int i = 0; i < SV::size(); i++)
        sensitivityBasis[sensitivityBasisRank][i] = residual[i] / norm;
    sensitivityBasisRank++;
}

template <typename PV, typename SV>
void DataStore<PV, SV>::updateSensitivityBasis() const
{
    if (sensitivityBasisSamples > sobs2.size()) {
        sensitivityBasisRank = 0;
        sensitivityBasisSamples = 0;
    }
    for (; sensitivityBasisSamples < sobs2.size(); sensitivityBasisSamples++)
        updateSensitivityBasis(&sdiffs[sensitivityBasisSamples * SV::size()]);
}

template <typename PV, typename SV>
const Labeled<PV> &DataStore<PV, SV>::registerPV(const PV &pv)
{
    Labeled<PV> lpv{uint32_t(pvs.size()), reduceHash(pv.hash()), pv};

    const auto &registered = pvs.insert(lpv);
    if (registered.index >= pvData.size()) {
        /* New PV -- initialize its PVData */
        PVData d;
        d.absRoofline = client.absRoofline(pv);
        pvData.push_back(std::move(d));
        findNeighbors(lpv);
        reroof(lpv);
        emptyPVs++;
    }

    assert(pvs.size() == pvData.size());

    return registered;
}

template <typename PV, typename SV>
const Labeled<SV> &DataStore<PV, SV>::registerSV(const SV &sv)
{
    Labeled<SV> lsv{uint32_t(svs.size()), reduceHash(sv.hash()), sv};
    auto &registered = svs.insert(lsv);
    svsWithData.resize(svs.size());
    return registered;
}

template <typename PV, typename SV>
const PVData &DataStore<PV, SV>::find(int index) const
{
    if (index > pvData.size())
        throw std::invalid_argument("pv");
    return pvData[index];
}

template <typename PV, typename SV>
const PVData &DataStore<PV, SV>::find(const Labeled<PV> &pv) const
{
    return find(pv.index);
}

template <typename PV, typename SV>
PVData &DataStore<PV, SV>::find(const Labeled<PV> &pv)
{
    return const_cast<PVData &>((const_cast<const DataStore<PV, SV> *>(this))->find(pv));
}

template <typename PV, typename SV>
void DataStore<PV, SV>::add(Varianced perf, const Labeled<PV> &pv, const Labeled<SV> &sv)
{
    if (!std::isfinite(perf) || pv.index < frozenPVs) return;

    auto &pvd = find(pv);
    auto nentries = uint32_t(pvd.data.size());

    /* Check if we already have data on this SV */
    bool existing = false;
    uint32_t dindex = 0;
    for (dindex = 0; dindex < nentries; dindex++) {
        auto &e = pvd.data[dindex];
        if (e.sv == sv.index) {
            /* Keep the "better" data, where better means higher (mean - stddev) */
            if (e.perf.value - std::sqrt(e.perf.variance) <= perf.value - std::sqrt(perf.variance))
                e.perf = perf;
            existing = true;
            break;
        }
    }

    if (!existing) {
        dindex = nentries;
        pvd.data.push_back(PVData::Entry{sv.index, perf});
        if (!svsWithData[sv.index]) {
            nonEmptySVs++;
            svsWithData[sv.index] = true;
        }
    }

    /* If new top performer, update rooflines for our neighborhood */
    if (nentries == 0 || pvd.data[pvd.topIndex].perf < perf) {
        pvd.topIndex = dindex;
        pvd.roofline = std::max(pvd.roofline, float(perf));
        for (auto &n: pvd.neighbors)
            reroof(pvs[n]);
    }

    if (nentries == 0)
        emptyPVs--;
}

template <typename PV, typename SV>
void DataStore<PV, SV>::add(Varianced perf, Varianced prevPerf,
                            const Labeled<PV> &pv, const Labeled<SV> &sv, const Labeled<SV> &prevSV)
{
    if (!std::isfinite(perf) || pv.index < frozenPVs) return;
    add(perf, pv, sv);

    /* Log data */
    auto obs = std::abs(std::log2(double(perf) / prevPerf));
    sobs2.push_back(obs * obs);

    auto nsamples = int(sobs2.size());

    sdiffs.resize(sdiffs.size() + SV::size());
    auto *diff = &sdiffs[(nsamples - 1) * SV::size()];
    for (int i = 0; i < SV::size(); i++)
        diff[i] = std::abs(sv[i] - prevSV[i]);
    updateSensitivityBasis();
    lastSensitivityRank = sensitivityBasisRank;

    /* Periodically update sensitivities, using maximum likelihood estimation.
       For our model, MLE reduces to a least-squares fit of diff-weighted (sdiff) sensitivities
       to squared log-perf ratios (sobs2). */

    if (nsamples >= nextSensitivityCalc) {
        nextSensitivityCalc *= Settings::sensitivityCalcMultiplier;

        auto A = sdiffs, b = sobs2;
        bool regularize = lastSensitivityRank < SV::size();
        auto solveSamples = nsamples;
        if (regularize) {
            auto ridge = std::sqrt(Settings::sensitivityRegularization);
            A.resize((nsamples + SV::size()) * SV::size(), 0.);
            b.resize(nsamples + SV::size(), 0.);
            for (int i = 0; i < SV::size(); i++)
                A[(nsamples + i) * SV::size() + i] = ridge;
            solveSamples += SV::size();

            if (getenv("SENSITIVITY_DUMP"))
                std::cerr << "Sensitivity fit remains rank-deficient after " << nsamples
                          << " samples (rank " << lastSensitivityRank << "/" << SV::size()
                          << "); using ridge regularization.\n";
            if (getenv("SENSITIVITY_DUMP")) for (int j = 0; j < SV::size(); j++) {
                int changed = 0;
                for (int i = 0; i < nsamples; i++)
                    changed += (sdiffs[i * SV::size() + j] != 0.);
                std::cerr << "  SV dimension " << j << ": changed in " << changed << "/"
                          << nsamples << " samples.\n";
            }
        }
        auto info = LAPACKE_dgels(LAPACK_COL_MAJOR, 'T', /*m*/ SV::size(), /*n*/ solveSamples, /*rhs*/ 1,
                                  A.data(), /*lda*/ SV::size(), b.data(), /*ldb*/ solveSamples);

        if (getenv("SENSITIVITY_DUMP")) {
            std::ofstream As("A.txt");
            for (int i = 0; i < nsamples; i++) {
                for (int j = 0; j < SV::size(); j++)
                    As << sdiffs[i * SV::size() + j] << ' ';
                As << std::endl;
            }
            std::ofstream Bs("b.txt");
            for (auto &v: sobs2)
                Bs << v << std::endl;
        }

        if (info == 0) {
            for (int i = 0; i < SV::size(); i++)
                sensitivities[i] = b[i];
        }
    }
}

template <typename PV, typename SV>
Varianced DataStore<PV, SV>::estimateSuboptimality(const Labeled<PV> &pv, const SV &sv, bool absolute) const
{
    const auto &pvd = find(pv);

    /* Obtain mean from weighted sum of normal PDFs from each existing data point,
       choosing weights to minimize variance:
            \sum_i w_i N(mu_i, s_i^2)
                ->  w_i = s_i^{-2} / (\sum_j s_j^{-2})
     */
    double weightedSubopt = 0.;
    double totalIVariance = 0.;

    if (pvd.data.empty())
        return Varianced{0.0f, 0.0f};

    for (auto &e: pvd.data) {
        auto subopt = suboptimality(pv, e.perf, absolute);
        auto variance = perfVariance(sv, svs[e.sv]);
        if (variance <= 1e-9)
            return Varianced{subopt, 0.0f};
        double ivariance = 1. / variance;
        totalIVariance += ivariance;
        weightedSubopt += ivariance * subopt;
    }

    return Varianced{float(weightedSubopt / totalIVariance), float(1. / totalIVariance)};
}

template <typename PV, typename SV>
float DataStore<PV, SV>::perfVariance(const SV &sv1, const SV &sv2) const
{
    float variance = 0.0f;
    for (int i = 0; i < SV::size(); i++)
        variance += std::abs(sv1[i] - sv2[i]) * sensitivities[i];
    return variance;
}

template <typename PV, typename SV>
void DataStore<PV, SV>::updateRoofline(float &relPerf, const PV &src, const Labeled<PV> &ref,
                                       float neighborRadius, float neighborUmbra)
{
    // Use the neighbor's measured efficiency, not its roofline: propagating rooflines would chain
    //   the maximum transitively (weight 1 within the umbra) across the whole neighbor graph.
    auto &pvd = find(ref);
    if (pvd.topIndex < 0) return;
    float relPerfN = pvd.topData().perf.value / pvd.absRoofline;
    float dist = client.distance(src, ref);
    float weight;
    if (neighborRadius <= neighborUmbra)
        weight = (dist == 0.f) ? 1.f : 0.f;
    else {
        weight = (neighborRadius - dist) / (neighborRadius - neighborUmbra);
        weight = std::clamp(weight, 0.f, 1.f);
    }
    relPerf = std::max(relPerf, relPerfN * weight);
}

template <typename PV, typename SV>
void DataStore<PV, SV>::reroof(const Labeled<PV> &lpv)
{
    auto &pvd = find(lpv);

    float relPerf = pvd.roofline / pvd.absRoofline;
    for (auto &n: pvd.neighbors)
        updateRoofline(relPerf, lpv, pvs[n], neighborRadius, neighborUmbra);

    pvd.roofline = relPerf * pvd.absRoofline;
}

template <typename PV, typename SV>
float DataStore<PV, SV>::roofline(const PV &pv)
{
    float relPerf = 0.;
    for (size_t i = 0; i < pvs.size(); i++)
        updateRoofline(relPerf, pv, pvs[i], neighborRadius, neighborUmbra);

    return relPerf * client.absRoofline(pv);
}

template <typename PV, typename SV>
void DataStore<PV, SV>::findNeighbors(const Labeled<PV> &pv)
{
    auto &pvd = find(pv);

    // Plain linear time algorithm, improve if needed.
    for (size_t i = 0; i < pvs.size(); i++) {
        if (i == pv.index) continue;
        if (client.distance(pv, pvs[i]) <= neighborRadius) {
            pvd.neighbors.push_back(i);
            pvData[i].neighbors.push_back(pv.index);
        }
    }
}

template <typename PV, typename SV>
std::vector<uint32_t> DataStore<PV, SV>::wideNeighbors(const Labeled<PV> &pv, float umbraFactor) const
{
    std::vector<uint32_t> result;
    auto radius = umbraFactor * neighborUmbra;

    for (size_t i = 0; i < pvs.size(); i++) {
        if (i == pv.index) continue;
        if (client.distance(pv, pvs[i]) <= radius)
            result.push_back(uint32_t(i));
    }

    return result;
}

template <typename PV, typename SV>
void DataStore<PV, SV>::initializeNeighbors()
{
    // Plain O(n^2) algorithm, improve if needed.
    auto npv = pvs.size();
    assert(npv > 0);

    // First pass: identify neighborhood size.
    //   Aim to have approximately dim(PV) neighbors in the core of the neighborhood.
    std::vector<float> distances;
    float minNeighborhood = 0.;
    float avgNeighborhood = 0.;
    distances.reserve(npv);
    for (size_t i = 0; i < npv; i++) {
        distances.clear();
        for (size_t j = 0; j < npv; j++)
            distances.push_back(client.distance(pvs[i], pvs[j]));
        std::sort(distances.begin(), distances.end());
        float voteAvg = distances[std::min(npv, PV::size() + 1) - 1];
        float voteMin = distances[std::min<size_t>(npv - 1, 1)];
        avgNeighborhood += voteAvg;
        minNeighborhood = std::max(minNeighborhood, voteMin);
    }

    avgNeighborhood /= npv;

    neighborUmbra = std::max(avgNeighborhood, minNeighborhood);
    neighborRadius = neighborUmbra * 2;

    // Second pass: generate neighborhoods.
    for (size_t i = 0; i < npv; i++) {
        for (size_t j = i + 1; j < npv; j++) {
            auto &pv1 = pvs[i], &pv2 = pvs[j];
            auto &pvd1 = pvData[i], &pvd2 = pvData[j];
            if (client.distance(pv1, pv2) <= neighborRadius) {
                pvd1.neighbors.push_back(j);
                pvd2.neighbors.push_back(i);
            }
        }
    }
}

template <typename PV, typename SV>
float DataStore<PV, SV>::suboptimality(const Labeled<PV> &lpv, bool absolute) const
{
    auto &pvd = find(lpv);

    if (pvd.topIndex < 0)
        return 1.f;
    else
        return suboptimality(lpv, pvd.data[pvd.topIndex].perf, absolute);
}

template <typename PV, typename SV>
float DataStore<PV, SV>::suboptimality(const Labeled<PV> &lpv, float perf, bool absolute) const
{
    auto &pvd = find(lpv);

    auto roof = pvd.absRoofline;
    if (!absolute && pvd.roofline > 0.0f)
        roof = std::min(roof, pvd.roofline * (1.0f + atticSpace));
    return std::log2(std::max(1.0f, roof / perf));
}

template <typename PV, typename SV>
bool DataStore<PV, SV>::anyAbsolutelySuboptimal() const
{
    for (auto &pvd: pvData) {
        if (pvd.topIndex < 0)
            return true;
        if (pvd.data[pvd.topIndex].perf < pvd.absRoofline)
            return true;
    }
    return false;
}

template <typename PV, typename SV>
void DataStore<PV, SV>::setInitialSensitivity(int i, float s)
{
    if (sensitivities[i] != 0.f)
        sensitivities[i] = s;
}

template <typename PV, typename SV>
bool DataStore<PV, SV>::sensitivityProbeIncreasesRank(const SV &sv, const SV &prevSV) const
{
    updateSensitivityBasis();
    if (sensitivityBasisRank == SV::size())
        return false;

    std::array<double, SV::size()> diff;
    for (int i = 0; i < SV::size(); i++)
        diff[i] = std::abs(sv[i] - prevSV[i]);

    std::array<double, SV::size()> residual = diff;
    double norm0 = 0.;
    for (auto value: residual)
        norm0 += value * value;
    for (int i = 0; i < sensitivityBasisRank; i++) {
        double projection = 0.;
        for (int j = 0; j < SV::size(); j++)
            projection += residual[j] * sensitivityBasis[i][j];
        for (int j = 0; j < SV::size(); j++)
            residual[j] -= projection * sensitivityBasis[i][j];
    }

    double norm = 0.;
    for (auto value: residual)
        norm += value * value;
    return norm > norm0 * 1e-12;
}

template <typename PV, typename SV>
bool DataStore<PV, SV>::sensitivitiesAvailable() const
{
    for (auto s: sensitivities)
        if (!std::isfinite(s))
            return false;
    return true;
}

template <typename PV, typename SV>
float DataStore<PV, SV>::sensitivitySamplingProgress() const
{
    if (sensitivitiesAvailable())
        return 1.0f;

    auto progressSamples = std::min(float(sobs2.size()) / (SV::size() * Settings::minSensitivitySamples), 1.0f);
    auto progressRank = float(lastSensitivityRank) / SV::size();
    return 0.5f * (progressSamples + progressRank);
}

const PVData::Entry &PVData::topData() const
{
    if (topIndex < 0)
        throw std::runtime_error("No data");
    return data[topIndex];
}


#if defined(MULTIOPT_PV) && defined(MULTIOPT_SV)
template class DataStore<MULTIOPT_PV, MULTIOPT_SV>;
#endif

} /* namespace multiopt */
