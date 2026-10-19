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

#include "optimizer.hpp"
#include "settings.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace multiopt {

template <typename PV, typename SV>
Optimizer<PV, SV>::Optimizer(Client<PV, SV> &client_, const std::vector<PV> &initialPVs,
                             const std::vector<SV> &initialSVs, int nagents_)
        : client(client_), data(client_), spreadingAgent(data), nagents(nagents_), extractor(data)
{
    /* Initialize data storage with given PVs/SVs */
    for (auto &sv: initialSVs)
        auto lsv = data.registerSV(sv);
    for (auto &pv: initialPVs)
        auto lpv = data.registerPV(pv);
    data.initializeNeighbors();

    for (size_t i = 0; i < data.allPVs().size(); i++)
        seedQueue.push_back(uint32_t(i));

    agents.reserve(nagents);
}

// Main optimizer loop.
template <typename PV, typename SV>
void Optimizer<PV, SV>::run(double timeout)
{
    interrupted = false;
    auto start = std::chrono::steady_clock::now();

    // Repair agents stuck with no queued request (e.g. after checkpoint restore).
    if (!agents.empty()) {
        bool anyAgentQueued = false;
        for (auto &r: requestQueue)
            if (r.agentID >= 0) { anyAgentQueued = true; break; }
        if (!anyAgentQueued) {
            bool anyReady = false;
            for (auto &a: agents) anyReady |= a.ready();
            if (!anyReady) {
                for (auto &a: agents) a.reset();
                for (size_t i = 0; i < agents.size(); i++) nextAgentRequest(i);
            }
        }
    }

    while (!interrupted) {
        // Keep draining spreading-agent work even once frozen.
        if (spreadingAgent.hasWork()) {
            request(spreadingAgent.next(), spreadingAgentID);

            // Extract as soon as the agent has drained its whole batch of SVs.
            if (!spreadingAgent.hasWork())
                topKernels = extractor.extract();
        }

        if (!frozen && agents.empty()) {
            for (int i = 0; i < nagents; i++)
                agents.emplace_back(data);
            seedAgentDone.assign(agents.size(), false);

            auto rstart = std::chrono::steady_clock::now();
            for (size_t i = 0; i < agents.size(); i++)
                nextAgentRequest(i);
            times.requests += (std::chrono::steady_clock::now() - rstart) / std::chrono::duration<float>(1);
        }

        // Stage transitions, checked every iteration.
        bool transitioned = false;
        if (stage == Stage::Seed) {
            bool allDone = seedQueue.empty();
            for (bool d: seedAgentDone)
                allDone &= d;
            if (allDone) {
                stage = Stage::Sensitivity;
                transitioned = true;
            }
        } else if (stage == Stage::Sensitivity && data.sensitivitiesAvailable()) {
            stage = Stage::Search;
            transitioned = true;
        }
        if (transitioned) {
            auto rstart = std::chrono::steady_clock::now();
            for (size_t i = 0; i < agents.size(); i++)
                if (agents[i].ready()) nextAgentRequest(i);
            times.requests += (std::chrono::steady_clock::now() - rstart) / std::chrono::duration<float>(1);
        }

        // Tally stalled agents and find the oldest ready request.
        int agentEntries = 0, agentReady = 0;
        auto firstReady = requestQueue.end();
        for (auto it = requestQueue.begin(); it != requestQueue.end(); ++it) {
            bool ready = (timeout <= 0 || client.ready(*it));
            if (it->agentID >= 0) {
                agentEntries++;
                if (ready) agentReady++;
            }
            if (ready && firstReady == requestQueue.end())
                firstReady = it;
        }
        agentsStalled = agentEntries - agentReady;

        // If nothing is ready, sleep briefly and retry.
        if (firstReady == requestQueue.end()) {
            if (timeout <= 0)
                break;
            auto elapsed = std::chrono::steady_clock::now() - start;
            if (elapsed / std::chrono::duration<double>(1) >= timeout)
                break;
            std::this_thread::sleep_for(std::chrono::duration<double>(Settings::pollInterval));
            continue;
        }
        auto req = *firstReady;
        requestQueue.erase(firstReady);

        // When frozen (finishing up), skip init-phase requests
        if (frozen && req.agentID < 0 && req.agentID != spreadingAgentID)
            continue;

        auto bstart = std::chrono::steady_clock::now();
        if (!req.empty())
            client.benchmark(req);
        times.benchmark += (std::chrono::steady_clock::now() - bstart) / std::chrono::duration<float>(1);

        // If request has an agent ID, sent to that client.
        // Otherwise send directly to the data store.
        auto nstart = std::chrono::steady_clock::now();
        if (req.agentID >= 0)
            agents[req.agentID].notify(req, stage);
        else if (req.agentID == spreadingAgentID)
            spreadingAgent.notify(req);
        else if (req.valid())
            for (size_t i = 0; i < req.perf.size(); i++)
                data.add(req.perf[i], req.pvs[i], req.sv);
        times.notifies += (std::chrono::steady_clock::now() - nstart) / std::chrono::duration<float>(1);

        // Requeue this agent's next request.
        if (req.agentID >= 0 && !frozen) {
            if (stage != Stage::Seed)
                data.setAtticSpace(Settings::atticMin + (Settings::atticStart - Settings::atticMin) * (Settings::atticDecay / (iterations + Settings::atticDecay)));
            auto rstart = std::chrono::steady_clock::now();
            nextAgentRequest(size_t(req.agentID));
            times.requests += (std::chrono::steady_clock::now() - rstart) / std::chrono::duration<float>(1);
        }

        iterations++;

        if (stage == Stage::Search) {
            // Periodically insert a random new PV.
            if (addNewPVs && (iterations % Settings::newPVEvery == 0))
                requestInitialPVData(data.registerPV(client.randomPV()));

            // Periodically spread top-ranked SVs to other PVs.
            if (iterations % Settings::spreadEvery == 0)
                spreadTopSVs();
        }

        // Check for timeout.
        if (timeout > 0) {
            auto now = std::chrono::steady_clock::now();
            if ((now - start) / std::chrono::duration<double>(1) >= timeout)
                break;
        }
    }

    times.total += (std::chrono::steady_clock::now() - start) / std::chrono::duration<float>(1);
}

template <typename PV, typename SV>
void Optimizer<PV, SV>::interrupt()
{
    interrupted = true;
}

template <typename PV, typename SV>
void Optimizer<PV, SV>::freeze()
{
    frozen = true;
}

template <typename PV, typename SV>
void Optimizer<PV, SV>::unfreeze()
{
    frozen = false;
}

template <typename PV, typename SV>
void Optimizer<PV, SV>::request(const AgentRequest<PV, SV> &req)
{
    if (req.empty()) return;   // Nothing to submit.
    requestQueue.push_back(req);
    client.preload(req);
}

template <typename PV, typename SV>
void Optimizer<PV, SV>::request(AgentRequest<PV, SV> &&req, int agentID)
{
    if (req.empty()) return;   // Nothing to submit.
    req.agentID = agentID;
    requestQueue.push_back(req);

    auto &nreq = requestQueue.back();
    client.preload(nreq);
}

// Generate and enqueue agent i's next request, per the current stage.
template <typename PV, typename SV>
void Optimizer<PV, SV>::nextAgentRequest(size_t i)
{
    if (stage == Stage::Seed) {
        // Each agent seeds one dedicated PV at a time until seedQueue is empty.
        if (seedAgentDone[i]) return;
        if (!agents[i].hasTarget()) {
            if (seedQueue.empty()) { seedAgentDone[i] = true; return; }
            agents[i].assignTarget(data.allPVs()[seedQueue.front()], Settings::initialSVsPerPV,
                                    Settings::initialSVsPerPV * Settings::seedGiveUpMultiplier);
            seedQueue.pop_front();
        }
    }
    request(agents[i].next(stage), int(i));
}

template <typename PV, typename SV>
void Optimizer<PV, SV>::requestInitialPVData(const Labeled<PV> &pv)
{
    auto pvd = data.find(pv);

    if (!pvd.data.empty()) return;

    AgentRequest<PV, SV> req;
    req.pvs.push_back(pv);

    std::unordered_set<uint32_t> isvs;

    for (auto &n: pvd.neighbors) {
        auto &pvd = data.find(n);
        if (pvd.topIndex >= 0)
            isvs.insert(pvd.data[pvd.topIndex].sv);
    }

    for (auto &isv: isvs) {
        req.sv = data.allSVs()[isv];
        request(req);
    }
}

// Locate top SVs and identify other PVs where they may perform well.
template <typename PV, typename SV>
void Optimizer<PV, SV>::spreadTopSVs()
{
    if (spreadingAgent.hasWork()) return;

    // Collect unspread top-ranked SVs for each PV and sort by minimum suboptimality.
    std::unordered_map<uint32_t, float> topSVs;
    auto npv = data.allPVs().size();

    for (auto &pv: data.allPVs()) {
        auto pvd = data.find(pv);
        if (pvd.topIndex < 0) continue;
        auto topSV = pvd.topData().sv;
        auto subopt = data.suboptimality(pv);

        if (auto is = data.spreadSVs().find(topSV); is == data.spreadSVs().end() || is->second < npv) {
            auto it = topSVs.find(topSV);
            if (it == topSVs.end())
                topSVs[topSV] = subopt;
            else
                it->second = std::min(it->second, subopt);
        }
    }

    std::vector<std::pair<float, uint32_t>> sortedTopSVs;
    sortedTopSVs.reserve(topSVs.size());
    for (auto &it: topSVs)
        sortedTopSVs.push_back({it.second, it.first});

    std::sort(sortedTopSVs.begin(), sortedTopSVs.end());

    // Limit list to some preset max length and initialize agent.
    std::vector<uint32_t> agentSVs;
    agentSVs.reserve(Settings::maxSpreadSVs);
    for (auto &top: sortedTopSVs) {
        agentSVs.push_back(top.second);
        if (agentSVs.size() >= Settings::maxSpreadSVs) break;
    }
    spreadingAgent.reset(std::move(agentSVs));
}

#if defined(MULTIOPT_PV) && defined(MULTIOPT_SV)
template class Optimizer<MULTIOPT_PV, MULTIOPT_SV>;
#endif

} /* namespace multiopt */
