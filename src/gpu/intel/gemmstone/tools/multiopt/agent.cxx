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

#include "agent.hpp"
#include "settings.hpp"

#include <algorithm>
#include <cassert>

namespace multiopt {

template <typename PV, typename SV>
Agent<PV, SV>::Agent(DataStore<PV, SV> &data_) : data(data_)
{
    reset();
}

template <typename PV, typename SV>
Agent<PV, SV>::Agent(DataStore<PV, SV> &data_, const Labeled<PV> &pv_, const Labeled<SV> &sv_) : data(data_)
{
    reset(pv_, sv_);
}

template <typename PV, typename SV>
void Agent<PV, SV>::reset()
{
    perf.value = std::numeric_limits<float>::quiet_NaN();
    prevPerf = perf;
    pendingReq = false;
    newTarget();
}

template <typename PV, typename SV>
void Agent<PV, SV>::reset(const Labeled<PV> &pv_, const Labeled<SV> &sv_)
{
    pv = pv_;
    sv = prevSV = sv_;
    perf.value = std::numeric_limits<float>::quiet_NaN();
    prevPerf = perf;
    pendingReq = false;
    consecutiveFreshFailures = 0;
}

template <typename PV, typename SV>
AgentRequest<PV, SV> Agent<PV, SV>::next(Stage stage)
{
    if (stage == Stage::Seed)
        return nextTarget();

    auto &minLegal = client().minLegalSV();
    auto &maxLegal = client().maxLegalSV();
    auto &minRandom = client().minRandomSV();
    auto &maxRandom = client().maxRandomSV();
    auto &legal = client().legalSVEntries();
    using SVT = ElementType<SV>;

    prevSV = sv;
    prevPerf = perf;

    bool unknownSens = !data.sensitivitiesAvailable();

    // Choose a random neighboring SV with probability depending
    //   on our current suboptimality and the sensitivity metric.
    //
    // If we are near the roofline, prefer moving to nearby SVs (close w.r.t. sensitivity metric).
    // If we are far below the roofline, prefer moving to faraway SVs.
    int tries = 0;
    bool mustAccept = false;
    int sensitivityProbeAttempts = 0;
    while (true) {
        // If some sensitivities are unknown, decide if we should preferentially tackle them.
        bool probeSens = unknownSens && sensitivityProbeAttempts < 2 * SV::size()
                      && (stage == Stage::Sensitivity || randBool(Settings::agentProbTestUnknownSensitivity));

        // Randomly update a random dimension of the strategy.
        int i;
        if (probeSens) {
            std::vector<int> unknownDimensions;
            for (int j = 0; j < SV::size(); j++)
                if (!std::isfinite(data.sensitivity(j)))
                    unknownDimensions.push_back(j);
            i = unknownDimensions[randInt(unknownDimensions.size())];
        } else
            i = randInt(SV::size());

        auto candSV = prevSV.vec;
        if (legal[i].empty()) {
            auto smin = std::max<int>(minLegal[i], std::min<int>(minRandom[i], candSV[i] - 1));
            auto smax = std::min<int>(maxLegal[i], std::max<int>(maxRandom[i], candSV[i] + 1));
            candSV[i] = randInt(smin, smax);
        } else {
            auto &l = legal[i];
            int imin, imax;
            for (imin = 0; imin + 1 < l.size(); imin++)
                if (l[imin] >= minRandom[i] || l[imin + 1] >= candSV[i])
                    break;
            for (imax = l.size() - 1; imax > 0; imax--)
                if (l[imax] <= maxRandom[i] || l[imax - 1] >= candSV[i])
                    break;
            candSV[i] = l[randInt(imin, imax)];
        }

        // Legalize it.
        if (!client().legalize(candSV, pv) || candSV == prevSV)
            continue;   /* try again */

        if (probeSens && !data.sensitivityProbeIncreasesRank(candSV, prevSV)) {
            sensitivityProbeAttempts++;
            continue;
        }

        // Decide whether to accept it.
        if (mustAccept || acceptStep(pv, candSV)) {
            sv = data.registerSV(candSV);
            break;
        }

        // If we fail enough steps, re-target.
        if (++tries >= Settings::agentMaxFailedAccepts) {
            newTarget();
            mustAccept = true;
        }
    }

    auto &pvd = data.find(pv);

    // Collect a list of PVs (problem sizes) to benchmark.
    AgentRequest<PV, SV> req;
    req.sv = sv;
    req.pvs.reserve(pvd.neighbors.size());
    req.pvs.push_back(pv);
    if (pvd.topIndex >= 0) {
        req.minPerf0       = pvd.data[pvd.topIndex].perf * Settings::agentNeighborCheckThreshold;
        req.minPerfCareful = pvd.data[pvd.topIndex].perf * Settings::agentCarefulCheckThreshold;
    }

    // Also check neighboring PVs (will be skipped if performance is too low on this PV)
    for (auto &n: pvd.neighbors) {
        auto &neighbor = data.allPVs()[n];
        req.pvs.push_back(neighbor);
    }

    pendingReq = true;
    return req;
}

template <typename PV, typename SV>
void Agent<PV, SV>::assignTarget(const Labeled<PV> &pv_, int successGoal, int attemptLimit)
{
    pv = pv_;
    targetSuccesses = 0;
    targetAttempts = 0;
    targetSuccessGoal = successGoal;
    targetAttemptLimit = attemptLimit;
    targetActive = true;
    pendingReq = false;
}

// Draw a random legal SV for the assigned target PV, plus bonus neighbor batching.
template <typename PV, typename SV>
AgentRequest<PV, SV> Agent<PV, SV>::nextTarget()
{
    sv = data.registerSV(client().randomSV(pv));

    AgentRequest<PV, SV> req;
    req.sv = sv;
    req.pvs.push_back(pv);

    for (auto &n: data.wideNeighbors(pv, 2.0f)) {
        auto &neighbor = data.allPVs()[n];
        SV candidate = sv;
        if (client().legalize(candidate, neighbor) && candidate == sv)
            req.pvs.push_back(neighbor);
    }

    pendingReq = true;
    return req;
}

template <typename PV, typename SV>
void Agent<PV, SV>::notifyTarget(AgentRequest<PV, SV> &result)
{
    pendingReq = false;

    bool success = result.valid() && std::isfinite(result.perf[0]);
    if (success) {
        data.add(result.perf[0], result.pvs[0], sv);
        for (size_t i = 1; i < result.pvs.size(); i++)
            if (std::isfinite(result.perf[i]))
                data.add(result.perf[i], result.pvs[i], sv);   // Bonus neighbor data.
        targetSuccesses++;
    }
    targetAttempts++;

    if (targetSuccesses >= targetSuccessGoal || targetAttempts >= targetAttemptLimit)
        targetActive = false;
}

template <typename PV, typename SV>
bool Agent<PV, SV>::acceptStep(const Labeled<PV> &pv, const SV &sv)
{
    // If we don't have any sensitivity information yet, accept anything.
    if (!data.sensitivitiesAvailable())
        return true;

    // If our current best perf on this PV is too low, accept anything.
    auto curBest = data.suboptimality(pv);
    if (curBest >= Settings::agentRandomStepThresh)
        return true;

    // Estimate the probability that this SV will be sufficiently faster than the current SV,
    //   or better than the current best SV.
    auto est = data.estimateSuboptimality(pv, sv);
    auto rcdf = [](Varianced n, float x) {
        return 0.5f * std::erfc((x - n.value) / std::sqrt(2 * n.variance));
    };
    auto thresh = curBest;
    if (std::isfinite(prevPerf.value)) {
        auto prevSubopt = data.suboptimality(pv, prevPerf);
        thresh += (1.0f - Settings::agentSuboptimalityStepImprovement) * (prevSubopt - curBest);
    }
    if (est.variance <= 0.0f)
        return (est < thresh);

    auto iprob = rcdf(est, thresh) / rcdf(est, 0.0f);

    // Randomly accept SV based on this probability.
    return randBool(iprob);
}

template <typename PV, typename SV>
void Agent<PV, SV>::notify(AgentRequest<PV, SV> &result, Stage stage)
{
    if (stage == Stage::Seed) {
        notifyTarget(result);
        return;
    }

    float probNewPV = 0.0f;

    pendingReq = false;
    steps++;

    if (result.valid() && std::isfinite(result.perf[0])) {
        // Successful benchmarking. Record benchmarking results.
        successfulSteps++;
        consecutiveFreshFailures = 0;

        auto oldBest = data.suboptimality(pv);

        /* Performance at target PVs */
        perf = result.perf[0];

        if (std::isfinite(prevPerf))
            data.add(perf, prevPerf, pv, sv, prevSV);
        else
            data.add(perf, pv, sv);

        /* Performance at neighboring PVs */
        assert(result.pvs.size() == result.perf.size());
        for (size_t i = 1; i < result.pvs.size(); i++)
            if (std::isfinite(result.perf[i]))
                data.add(result.perf[i], result.pvs[i], sv);

        if (std::isfinite(prevPerf)) {
            auto subopt = data.suboptimality(pv, perf);
            auto prevSubopt = data.suboptimality(pv, prevPerf);

            auto probAccept = std::exp2(-std::max(0.f, subopt - prevSubopt) * Settings::agentAcceptSVSuboptDecay);
            if (!randBool(probAccept)) {
                sv = prevSV;
                perf = prevPerf;
                probNewPV = 1.0f - (1.0f - Settings::agentSwitchPVMinProbability) * std::exp2(std::max(0.f, oldBest - subopt) * Settings::agentSwitchPVSuboptDecay);
            }
        }
    } else {
        // Failed benchmarking.
        perf.value = std::numeric_limits<float>::quiet_NaN();
        if (std::isfinite(prevPerf)) {
            /* Retreat to known good location, but consider retarget */
            perf = prevPerf;
            sv = prevSV;
            probNewPV = Settings::agentProbabilitySwitchPVOnFailure;
        } else {
            /* Retry a few times at this PV before retargeting. */
            if (++consecutiveFreshFailures >= Settings::agentMaxFailedAccepts)
                probNewPV = 1.0f;
            else
                probNewPV = 0.0f;
        }
    }

    // Probabilistically jump to a new PV.
    if (randBool(probNewPV))
        newTarget();
}

template <typename PV, typename SV>
void Agent<PV, SV>::newTarget()
{
    // Randomly choose PV, weighted by suboptimality.
    auto &allPVs = data.allPVs();
    int pv0 = data.getFrozenPVs();
    assert(pv0 < allPVs.size());
    bool checkedAnySubopt = false;
    bool anySubopt = false;
    while (true) {
        auto &candPV = allPVs[pv0 + randInt(allPVs.size() - pv0)];
        auto subopt = data.suboptimality(candPV);
        auto prob = Settings::agentNewPVMinProbability
                    + (1 - Settings::agentNewPVMinProbability) * Settings::agentNewPVWeight * subopt;
        bool ok = (subopt > 0 && randBool(prob));

        if (!checkedAnySubopt) {        /* Avoid infinite loops if all PVs at roofline */
            checkedAnySubopt = true;
            anySubopt = (subopt > 0) || data.anyAbsolutelySuboptimal();
        }

        if (ok || !anySubopt) {
            pv = candPV;
            break;
        }
    }

    // Then choose an SV.
    // First try current best SV, then generate random SVs.
    bool done = false;
    auto &pvd = data.find(pv);
    if (pvd.topIndex >= 0) {
        /* Choose current best with probability depending on its suboptimality. */
        auto &candSV = data.allSVs()[pvd.data[pvd.topIndex].sv];
        auto prob = 1.0f;
        if (data.sensitivitiesAvailable()) {
            auto subopt = data.estimateSuboptimality(pv, candSV);
            prob = std::exp2(-subopt * Settings::agentChooseTopSVSuboptDecay);
        }
        if (randBool(prob)) {
            sv = candSV;
            done = true;
        }
    }

    int tries = 0;
    while (!done) {
        auto candSV = client().randomSV(pv);
        if (acceptStep(pv, candSV)) {
            sv = data.registerSV(candSV);
            break;
        }
        if (++tries >= Settings::agentMaxFailedAccepts) {
            if (pvd.topIndex >= 0)
                sv = data.allSVs()[pvd.data[pvd.topIndex].sv];
            else
                sv = data.registerSV(candSV);
            break;
        }
    }

    reset(pv, sv);
}

template <typename PV, typename SV>
void SpreadingAgent<PV, SV>::reset(std::vector<uint32_t> &&isvs_)
{
    isvs = isvs_;
    curISV = -1;
    activePVs.clear();
    pendingBatch.clear();
    lastNonadjacent = false;
}

template <typename PV, typename SV>
bool SpreadingAgent<PV, SV>::topCandidate(Varianced perf, const PVData &pvd)
{
    return (perf.value + Settings::nstddevTopSV * std::sqrt(perf.variance) >= pvd.topData().perf);
}

template <typename PV, typename SV>
AgentRequest<PV, SV> SpreadingAgent<PV, SV>::next()
{
    if (outstandingReq)
        return AgentRequest<PV, SV>();

    // Refill pendingBatch from activePVs, or advance to a new SV.
    while (pendingBatch.empty()) {
        bool newSV = activePVs.empty();
        if (newSV)
            curISV++;

        if (curISV >= isvs.size())
            return AgentRequest<PV, SV>();

        auto &sv = data.allSVs()[isvs[curISV]];

        // New SV to spread. Look through PVs, and:
        //  - Accumulate unordered_set of PVs for which SV already computed
        //  - Make list of (PV, est subopt) pairs, sorted by subopt
        //  - Initialize BFS list with neighborhoods of all PVs where this SV is a top contender
        auto npv = data.allPVs().size();
        if (newSV) {
            completedPVs.clear();
            seedPVs.clear();
            nextSeed = 0;
            unsuccessfulNonadjacent = 0;

            /* If this SV has already been spread, only check new PVs. */
            auto &pv0 = data.spreadSVs()[sv.index];
            minPV = pv0;
            pv0 = npv;

            std::vector<float> subopts;
            seedPVs.reserve(npv);
            subopts.reserve(npv);

            for (size_t i = 0; i < npv; i++) {
                auto &pvd = data.find(i);
                const PVData::Entry *entry = nullptr;
                for (auto &e: pvd.data) {
                    if (e.sv == sv.index) {
                        entry = &e;
                        break;
                    }
                }

                float subopt;
                if (entry) {
                    if (topCandidate(entry->perf, pvd))
                        for (auto n: pvd.neighbors)
                            activePVs.insert(n);
                    completedPVs.insert(i);
                    subopt = std::numeric_limits<float>::max();
                } else
                    subopt = data.estimateSuboptimality(data.allPVs()[i], sv);
                seedPVs.push_back(i);
                subopts.push_back(subopt);
            }

            auto *soPtr = subopts.data();
            std::sort(seedPVs.begin(), seedPVs.end(), [soPtr](uint32_t pv1, uint32_t pv2) {
                return soPtr[pv1] < soPtr[pv2];
            });

            seedPVs.resize(npv - completedPVs.size());
        }

        // Move activePVs into pendingBatch, sent in spreadBatchSize chunks.
        pendingBatch.reserve(pendingBatch.size() + activePVs.size());
        for (auto &ipv: activePVs)
            if (ipv >= minPV && !completedPVs.count(ipv))
                pendingBatch.push_back(ipv);
        activePVs.clear();
    }

    // Send out the next chunk of the current wave.
    auto &sv = data.allSVs()[isvs[curISV]];
    AgentRequest<PV, SV> req;
    req.sv = sv;
    size_t n = std::min<size_t>(Settings::spreadBatchSize, pendingBatch.size());
    req.pvs.reserve(n);
    for (size_t i = 0; i < n; i++)
        req.pvs.push_back(data.allPVs()[pendingBatch[i]]);
    pendingBatch.erase(pendingBatch.begin(), pendingBatch.begin() + n);

    outstandingReq = true;
    return req;
}

template <typename PV, typename SV>
void SpreadingAgent<PV, SV>::notify(AgentRequest<PV, SV> &request)
{
    outstandingReq = false;

    for (size_t i = 0; i < request.perf.size(); i++) {
        // Record data.
        auto &perf = request.perf[i];
        completedPVs.insert(request.pvs[i].index);
        data.add(perf, request.pvs[i], request.sv);

        // If this SV is a candidate for the optimum SV on this PV, probe its neighbors.
        auto &pvd = data.find(request.pvs[i]);
        if (pvd.data.empty() || topCandidate(perf, pvd))
            for (auto &n: pvd.neighbors)
                activePVs.insert(n);
        else if (lastNonadjacent) {
            // Stop spreading this SV after we have failed too many times to produce a top candidate on nonadjacent PVs.
            if (++unsuccessfulNonadjacent >= Settings::maxUnsuccessfulNonadjacent)
                return;
        }
    }

    // Wait until the wave is fully drained before deciding how it continues.
    if (!pendingBatch.empty())
        return;

    if (activePVs.empty()) {
        // No more neighbors. Restart search at nonadjacent PVs where this SV is expected to do well.
        int added = 0;
        for (; nextSeed < seedPVs.size() && added < Settings::spreadNonadjacentBatch; nextSeed++) {
            auto ipv = seedPVs[nextSeed];
            if (!completedPVs.count(ipv)) {
                activePVs.insert(ipv);
                added++;
            }
        }
        lastNonadjacent = true;
    } else
        lastNonadjacent = false;
}

#if defined(MULTIOPT_PV) && defined(MULTIOPT_SV)
template class Agent<MULTIOPT_PV, MULTIOPT_SV>;
template class SpreadingAgent<MULTIOPT_PV, MULTIOPT_SV>;
#endif

} /* namespace multiopt */
