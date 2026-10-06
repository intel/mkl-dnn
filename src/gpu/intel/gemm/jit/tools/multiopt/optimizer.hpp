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

#ifndef GEMMSTONE_TOOLS_MULTIOPT_OPTIMIZER_HPP
#define GEMMSTONE_TOOLS_MULTIOPT_OPTIMIZER_HPP

#include <deque>
#include <vector>

#include "agent.hpp"
#include "data.hpp"
#include "extractor.hpp"
#include "serialization.hpp"
#include "settings.hpp"

namespace multiopt {

template <typename PV, typename SV>
class Optimizer
{
protected:
    Client<PV, SV> &client;
    DataStore<PV, SV> data;
    std::vector<Agent<PV, SV>> agents;
    SpreadingAgent<PV, SV> spreadingAgent;

    // FIFO queue, at most one outstanding request per agent.
    std::deque<AgentRequest<PV, SV>> requestQueue;
    Extractor<PV, SV> extractor;

    std::vector<std::pair<int, int>> topKernels;

    volatile bool interrupted = false;
    volatile bool frozen = false;

    int iterations = 0;
    int initialSVsTested = 0;
    int nagents;
    bool addNewPVs = false;

    struct {
        float benchmark = 0.0f, requests = 0.0f, notifies = 0.0f, total = 0.0f;
    } times;

    int agentsStalled = 0;   // Agents blocked on kernel generation as of the last run() call.

    Stage stage = Stage::Seed;
    std::deque<uint32_t> seedQueue;      // PVs awaiting their Seed-stage pass.
    std::vector<bool> seedAgentDone;     // Per-agent: no more seed work remaining.

public:
    static constexpr int spreadingAgentID = -2;

    friend Serialized &operator<<(Serialized &s, const Optimizer &o) {
        s << o.data << o.agents.size();
        for (auto &agent: o.agents)
            s << agent;
        return s << o.spreadingAgent << o.requestQueue << o.extractor << o.topKernels
                 << o.iterations << o.initialSVsTested << o.nagents << o.addNewPVs << o.times
                 << o.stage << o.seedQueue << o.seedAgentDone;
    }
    friend Serialized &operator>>(Serialized &s, Optimizer &o) {
        o.interrupted = false;
        o.frozen = false;
        s >> o.data;
        size_t agentCount;
        s >> agentCount;
        o.agents.clear();
        for (size_t i = 0; i < agentCount; i++) {
            o.agents.emplace_back(o.data);
            s >> o.agents.back();
        }
        return s >> o.spreadingAgent >> o.requestQueue >> o.extractor >> o.topKernels
                 >> o.iterations >> o.initialSVsTested >> o.nagents >> o.addNewPVs >> o.times
                 >> o.stage >> o.seedQueue >> o.seedAgentDone;
    }

    Optimizer(Client<PV, SV> &client_, const std::vector<PV> &initialPVs, const std::vector<SV> &initialSVs = std::vector<SV>{}, int nagents_ = Settings::defaultAgents);

    void run(double timeout = 0.);  /* Start or restart optimization. Optional timeout in seconds. */
    void interrupt();               /* Interrupt optimization -- may be called from a signal handler or another thread */
    void freeze();                  /* Complete existing operations */
    void unfreeze();

    void request(const AgentRequest<PV, SV> &request);
    void request(AgentRequest<PV, SV> &&request, int agentID = -1);

    void spreadTopSVs();
    void enableNewPVs(bool enable = true) { addNewPVs = enable; }
    void freezePVs(int npv)               { data.freezePVs(npv); }

    // Lets a caller drain run() in bounded chunks and redraw progress meanwhile.
    bool hasPendingWork() const { return !requestQueue.empty() || spreadingAgent.hasWork(); }
    bool spreading() const     { return spreadingAgent.hasWork(); }
    typename SpreadingAgent<PV, SV>::Progress spreadProgress() const { return spreadingAgent.progress(); }

protected:
    void requestInitialPVData(const Labeled<PV> &pv);
    void nextAgentRequest(size_t i);
};

} /* namespace multiopt */

#endif
