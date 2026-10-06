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

#ifndef GEMMSTONE_TOOLS_MULTIOPT_AGENT_HPP
#define GEMMSTONE_TOOLS_MULTIOPT_AGENT_HPP

#include <unordered_set>
#include <vector>

#include "data.hpp"
#include "serialization.hpp"

namespace multiopt {

// Stage shared by Agent and Optimizer: Seed, Sensitivity, or Search.
enum class Stage { Seed, Sensitivity, Search };

template <typename PV, typename SV>
struct AgentRequest
{
    int agentID = -1;               //     unique identifier
    Labeled<SV> sv;                 // --> strategy
    std::vector<Labeled<PV>> pvs;   // --> problem sizes
    std::vector<Varianced> perf;    // <-- measured performance
    float minPerf0 = 0.0f;          // --> minimum performance on first PV
    float minPerfCareful = 0.0f;    // --> minimum performance hint for benchmarking.
                                    //       Exact performance below this threshold is unimportant.

    bool valid() const { return !perf.empty(); }
    bool empty() const { return pvs.empty();   }

    friend Serialized &operator<<(Serialized &s, const AgentRequest &r) {
        return s << r.agentID << r.sv << r.pvs << r.perf << r.minPerf0 << r.minPerfCareful;
    }
    friend Serialized &operator>>(Serialized &s, AgentRequest &r) {
        return s >> r.agentID >> r.sv >> r.pvs >> r.perf >> r.minPerf0 >> r.minPerfCareful;
    }
};

template <typename PV, typename SV>
class Agent
{
    DataStore<PV, SV> &data;
    Labeled<PV> pv;
    Labeled<SV> sv, prevSV;
    Varianced perf, prevPerf;
    bool pendingReq = false;

    int steps = 0;
    int successfulSteps = 0;
    int consecutiveFreshFailures = 0;   // consecutive failures at a never-measured PV/SV target.

    // Bookkeeping for a pinned target PV (see assignTarget()).
    bool targetActive = false;
    int targetSuccesses = 0;
    int targetAttempts = 0;
    int targetSuccessGoal = 0;
    int targetAttemptLimit = 0;

public:
    friend Serialized &operator<<(Serialized &s, const Agent &a) {
        return s << a.pv << a.sv << a.prevSV << a.perf << a.prevPerf << a.pendingReq << a.steps << a.successfulSteps
                 << a.consecutiveFreshFailures << a.targetActive << a.targetSuccesses << a.targetAttempts
                 << a.targetSuccessGoal << a.targetAttemptLimit;
    }
    friend Serialized &operator>>(Serialized &s, Agent &a) {
        return s >> a.pv >> a.sv >> a.prevSV >> a.perf >> a.prevPerf >> a.pendingReq >> a.steps >> a.successfulSteps
                 >> a.consecutiveFreshFailures >> a.targetActive >> a.targetSuccesses >> a.targetAttempts
                 >> a.targetSuccessGoal >> a.targetAttemptLimit;
    }

    Agent(DataStore<PV, SV> &data_);
    Agent(DataStore<PV, SV> &data_, const Labeled<PV> &pv_, const Labeled<SV> &sv_);

    void reset();
    void reset(const Labeled<PV> &pv_, const Labeled<SV> &sv_);

    AgentRequest<PV, SV> next(Stage stage);
    void notify(AgentRequest<PV, SV> &result, Stage stage);

    void newTarget();

    bool ready() const { return !pendingReq; }

    Client<PV, SV> &client() { return data.getClient(); }

    float successRate() const { return float(successfulSteps) / steps; }
    void resetSuccessRate()   { steps = successfulSteps = 0; }

    // Pinned-target interface, e.g. for the Seed stage.
    bool hasTarget() const { return targetActive; }
    void assignTarget(const Labeled<PV> &pv_, int successGoal, int attemptLimit);

protected:
    bool acceptStep(const Labeled<PV> &pv, const SV &sv);

    AgentRequest<PV, SV> nextTarget();
    void notifyTarget(AgentRequest<PV, SV> &result);
};

template <typename PV, typename SV>
class SpreadingAgent
{
    DataStore<PV, SV> &data;

    std::vector<uint32_t> isvs;

    std::unordered_set<uint32_t> completedPVs;
    std::unordered_set<uint32_t> activePVs;      // Next-wave frontier, accumulated via notify().
    std::vector<uint32_t> pendingBatch;          // Current wave's unsent PVs, drained in
                                                  // Settings::spreadBatchSize chunks.
    std::vector<uint32_t> seedPVs;

    bool lastNonadjacent = false;
    bool outstandingReq = false;
    int unsuccessfulNonadjacent = 0;
    int nextSeed = 0;
    int curISV = 0;
    int minPV = 0;

public:
    friend Serialized &operator<<(Serialized &s, const SpreadingAgent &a) {
        return s << a.isvs << a.completedPVs << a.activePVs << a.pendingBatch << a.seedPVs << a.lastNonadjacent
                 << a.outstandingReq << a.unsuccessfulNonadjacent << a.nextSeed << a.curISV << a.minPV;
    }
    friend Serialized &operator>>(Serialized &s, SpreadingAgent &a) {
        return s >> a.isvs >> a.completedPVs >> a.activePVs >> a.pendingBatch >> a.seedPVs >> a.lastNonadjacent
                 >> a.outstandingReq >> a.unsuccessfulNonadjacent >> a.nextSeed >> a.curISV >> a.minPV;
    }

    SpreadingAgent(DataStore<PV, SV> &data_) : data(data_) {}

    void reset(std::vector<uint32_t> &&isvs_);

    AgentRequest<PV, SV> next();
    void notify(AgentRequest<PV, SV> &result);

    bool hasWork() const { return curISV < int(isvs.size()); }

    // Progress of the current spreading pass, valid in any Stage.
    struct Progress {
        size_t svsDone = 0, svsTotal = 0;    // strategies fully spread / total queued to spread
        size_t pvsDone = 0, pvsTotal = 0;    // problems checked / total problems, for the strategy in progress
    };

    Progress progress() const {
        Progress p;
        p.svsTotal = isvs.size();
        p.svsDone = std::min<size_t>(size_t(std::max(curISV, 0)), isvs.size());
        p.pvsTotal = data.allPVs().size();
        p.pvsDone = completedPVs.size();
        return p;
    }

protected:
    bool topCandidate(Varianced perf, const PVData &pvd);
};

} /* namespace multiopt */

#endif
