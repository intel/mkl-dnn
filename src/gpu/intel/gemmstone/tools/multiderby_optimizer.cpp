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

#include "multiderby_optimizer.hpp"
#include "vt100_codes.hpp"

#include <iomanip>
#include <numeric>

#define MULTIOPT_PV ProblemVector
#define MULTIOPT_SV StrategyVector
#include "multiopt/agent.cxx"
#include "multiopt/data.cxx"
#include "multiopt/extractor.cxx"
#include "multiopt/optimizer.cxx"

using namespace multiopt;

MultiderbyOptimizer::MultiderbyOptimizer(Client<PV, SV> &client_, const MultiderbyOptions &options, const std::vector<PV> &initialPVs, const std::vector<SV> &initialSVs)
    : Optimizer(client_, initialPVs, initialSVs, options.agentCount)
{
    addNewPVs = options.addRandomProblems;
}

std::string MultiderbyOptimizer::status()
{
    auto npv = data.allPVs().size();
    auto nsv = data.allSVs().size();
    auto psv = data.populatedSVs();
    auto ppv = npv - data.unpopulatedPVs();
    auto sprogress = data.sensitivitySamplingProgress();
    auto attic = data.getAtticSpace();

    double avgSuboptRoof = 0.;
    double avgSuboptDrop = 0.;
    size_t dataPoints = 0;
    for (size_t i = 0; i < npv; i++) {
        auto &pv = data.allPVs()[i];
        auto &pvd = data.find(pv);
        if (!pvd.data.empty()) {
            avgSuboptRoof += data.suboptimality(pv, true);
            avgSuboptDrop += data.suboptimality(pv, false);
        }
        dataPoints += pvd.data.size();
    }
    avgSuboptRoof /= double(ppv);
    avgSuboptDrop /= double(ppv);
    float avgEffRoof = std::exp2(-avgSuboptRoof);
    float avgEffDrop = std::exp2(-avgSuboptDrop);

    float stepSR = 0.f;
    for (auto &agent: agents) {
        stepSR += agent.successRate();
        agent.resetSuccessRate();
    }
    stepSR /= nagents;

    auto now = std::chrono::steady_clock::now();

    double dpPerSecond = 0.;
    if (lastDPCount > 0)
        dpPerSecond = (dataPoints - lastDPCount) / ((now - lastUpdate) / std::chrono::duration<double>(1));
    lastUpdate = now;
    lastDPCount = dataPoints;

    std::stringstream s;

    s << npv << " problems, "
      << psv << " strategies (tried " << nsv << "), "
      << dataPoints << " data points";
    if (dpPerSecond > 0.)
        s << " (" << dpPerSecond << " points/s)";
    if (std::isfinite(stepSR))
        s << ", " << int(stepSR * 100) << "% step success";

    s << "\nStage: ";
    switch (stage) {
        case Stage::Seed: {
            int seedProgress = npv ? int(((npv - seedQueue.size()) * 100) / npv) : 100;
            s << "seed (" << seedProgress << "% complete)";
            break;
        }
        case Stage::Sensitivity:
            s << "sensitivity (" << int(sprogress * 100) << "% complete)";
            break;
        case Stage::Search:
            s << "search";
            if (!spreadingAgent.hasWork()) {
                s << ", average efficiency: " << int(avgEffRoof * 100) << "% abs, " << int(avgEffDrop * 100) << "% eff";
                if (attic < 1000.0f)
                    s << " (attic: " << int(attic * 100) << "%)";
            }
            break;
    }

    // Shown regardless of stage: 's' can trigger spreading before Search.
    if (spreadingAgent.hasWork()) {
        auto p = spreadingAgent.progress();
        s << "\nSpreading top kernels: " << p.svsDone << "/" << p.svsTotal << " strategies";
        if (p.pvsTotal > 0)
            s << " (" << int(100.0 * p.pvsDone / p.pvsTotal) << "% of problems checked for current strategy)";
    }

    return s.str();
}

std::string MultiderbyOptimizer::kernelList() const
{
    std::stringstream s;

    topKernelStatus(s);
    svStatus(s, topKernels.empty() ? 15 : 10);

    return s.str();
}

std::string MultiderbyOptimizer::finalKernelList() const
{
    std::string s;
    auto &mdClient = dynamic_cast<MultiderbyClient&>(client);

    for (auto &it: topKernels) {
        auto isv = it.first;
        auto &sv = data.allSVs()[isv];
        s += mdClient.svToString(sv, true);
        s +='\n';
    }
    return s;
}

std::vector<multiopt::Labeled<StrategyVector>> MultiderbyOptimizer::getTopKernels() const {
    std::vector<Labeled<StrategyVector>> res;
    for (auto &it: topKernels) {
        auto &sv = data.allSVs()[it.first];
        res.push_back(sv);
    }
    return res;
}

void MultiderbyOptimizer::svStatus(std::stringstream &s, int maxDisplayed) const
{
    /* Collect top SV for each PV, aggregating identical SVs. */
    std::unordered_map<uint32_t, std::pair<float, uint32_t>> topSVs0, topSVs1;    // sv# -> (subopt, pv#)

    for (auto &pv: data.allPVs()) {
        auto pvd = data.find(pv);
        if (pvd.topIndex < 0) continue;
        auto topSV = pvd.topData().sv;
        auto subopt = data.suboptimality(pv);
        auto value = std::make_pair(subopt, pv.index);

        for (bool best: {false, true}) {
            auto &topSVs = best ? topSVs1 : topSVs0;
            auto it = topSVs.find(topSV);
            if (it == topSVs.end())
                topSVs[topSV] = value;
            else if ((it->second.first > subopt) == best)
                it->second = value;
        };
    }

    /* Sort by minimum/maximum suboptimality */
    std::vector<std::pair<float, uint32_t>> sorted0, sorted1;       // (subopt, pv#)
    sorted0.reserve(topSVs0.size());
    sorted1.reserve(topSVs1.size());
    for (auto &it: topSVs0) sorted0.push_back(it.second);
    for (auto &it: topSVs1) sorted1.push_back(it.second);

    std::sort(sorted0.rbegin(), sorted0.rend());
    std::sort(sorted1.begin(), sorted1.end());

    /* Display them */
    auto &mdClient = dynamic_cast<MultiderbyClient&>(client);

    auto addTop = [&](auto &list) {
        int count = 0;
        for (auto &it: list) {
            if (count++ >= maxDisplayed)
                break;
            auto subopt = it.first;
            auto ipv = it.second;
            auto &pv = data.allPVs()[ipv];
            auto &pvd = data.find(ipv);
            auto isv = pvd.topData().sv;
            auto &sv = data.allSVs()[isv];

            auto abs = std::exp2(-data.suboptimality(pv, true));
            auto eff = std::exp2(-subopt);

            float perf = pvd.topData().perf;
            bool memBound = mdClient.isMemBound(pv);
            float perfUnits = memBound ? (perf / mdClient.memRoofline(pv) * mdClient.bps * 1e-9f)
                                       : (perf / mdClient.computeRoofline(pv) * mdClient.flops * 1e-9f);
            const char *unitStr = memBound ? " GB/s" : " GF";

            s << VT_MAGENTA << int(100 * abs) << "%\t" VT_RESET;
            s << VT_BLUE << int(100 * eff) << "%\t" VT_RESET;
            s << std::fixed << std::setprecision(0) << perfUnits << unitStr << "\t";
            s << std::defaultfloat;
            s << VT_BOLD "[" << mdClient.pvToString(pv) << "]\t" VT_RESET;
            s << mdClient.svToString(sv);
            s << '\n';
        }
    };

    s << "     " VT_SET_TAB "     " VT_SET_TAB "           " VT_SET_TAB "                   " VT_SET_TAB "\n";
    s << "Best problem sizes:   (" VT_MAGENTA "\u25A0" VT_RESET " = absolute efficiency, " VT_BLUE "\u25A0" VT_RESET " = relative to current roofline)\n";
    addTop(sorted1);
    s << "\nWorst problem sizes:\n";
    addTop(sorted0);
    s << VT_CLEAR_TABS;
}

void MultiderbyOptimizer::topKernelStatus(std::stringstream &s, int maxDisplayed) const
{
    int totalPVs = 0;
    int npv = data.allPVs().size();
    int shown = 0;

    auto &mdClient = dynamic_cast<MultiderbyClient&>(client);

    if (topKernels.empty()) {
        s << "No strategies selected yet. Gathering data...\n";
        return;
    }

    s << "Selected strategies:   (" VT_ORANGE "\u25A0" VT_RESET " = cumulative percentage of problem sizes covered)\n";
    s << "       " VT_SET_TAB "\r";
    for (auto &it: topKernels) {
        if (shown >= maxDisplayed) break;
        totalPVs += it.second;
        auto isv = it.first;
        auto &sv = data.allSVs()[isv];

        s << VT_ORANGE << int(100. * totalPVs / npv) << "%\t" VT_RESET
          << mdClient.svToString(sv, true) << '\n';
        shown++;
    }

    if (maxDisplayed < topKernels.size())
        s << "... " << (topKernels.size() - maxDisplayed) << " more strategies\n";
    s << VT_CLEAR_TABS;
}

std::string MultiderbyOptimizer::sensitivities() const
{
    std::stringstream s;

    if (!data.sensitivitiesAvailable()) return "";

    s << "Sensitivities: ";
    for (size_t i = 0; i < SV::size(); i++) {
        if (i > 0) s << ", ";
        auto sens = data.sensitivity(i);
        if (std::isfinite(sens))
            s << sens;
        else
            s << "nan";
    }
    return s.str();
}

std::string MultiderbyOptimizer::timings()
{
    std::stringstream s;
    auto &mdClient = dynamic_cast<MultiderbyClient&>(client);
    auto genTime = mdClient.frontendWaitTime();

    s << "Time: " << times.total << "s ("
                  << (times.benchmark - genTime) << "s benchmark + "
                  << genTime             << "s generation + "
                  << times.requests      << "s requests + "
                  << times.notifies      << "s notifies)";

    // Agents whose queued request is still waiting on kernel generation.
    s << "\nAgents stalled on generation: " << agentsStalled << " / " << agents.size();

    times.total = times.benchmark = times.requests = times.notifies = 0.0f;

    return s.str();
}

std::string MultiderbyOptimizer::roofline(const PV &pv)
{
    std::stringstream s;
    auto &mdClient = dynamic_cast<MultiderbyClient&>(client);

    auto saveAttic = data.getAtticSpace();
    data.setAtticSpace(0);

    auto perf = data.roofline(pv);
    auto eff = perf / client.absRoofline(pv);
    auto flops = perf / mdClient.computeRoofline(pv) * mdClient.flops;
    auto bps = perf / mdClient.memRoofline(pv) * mdClient.bps;

    data.setAtticSpace(saveAttic);

    s << mdClient.pvToString(pv) << ": ";

    if (perf > 0) {
        s << flops * 1.0e-9 << " GF, ";
        s << bps * 1.0e-9 << " GB/s, ";
        s << eff * 100 << "% efficiency";
    } else
        s << "no data available";

    return s.str();
}

std::string MultiderbyOptimizer::dump() const
{
    std::stringstream s;
    auto &mdClient = dynamic_cast<MultiderbyClient&>(client);

    auto npv = data.allPVs().size();
    auto nsv = data.allSVs().size();

    s << npv << " problem vectors, " << nsv << " strategy vectors\n\n";

    for (size_t i = 0; i < npv; i++) {
        auto &pv = data.allPVs()[i];
        auto &pvd = data.find(pv);
        bool memBound = mdClient.isMemBound(pv);
        const char *unitStr = memBound ? " GB/s" : " GF";

        auto toUnits = [&](float perf) {
            return memBound ? (perf / mdClient.memRoofline(pv) * mdClient.bps * 1e-9f)
                            : (perf / mdClient.computeRoofline(pv) * mdClient.flops * 1e-9f);
        };

        s << "=== [" << mdClient.pvToString(pv) << "]";
        s << (memBound ? " (memory-bound)" : " (compute-bound)");
        s << "  absRoofline: " << std::fixed << std::setprecision(1) << toUnits(pvd.absRoofline) << unitStr;
        if (pvd.roofline > 0)
            s << "  effRoofline: " << toUnits(pvd.roofline) << unitStr;
        s << std::defaultfloat << "\n";

        if (pvd.data.empty()) {
            s << "  (no data)\n\n";
            continue;
        }

        std::vector<size_t> order(pvd.data.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return pvd.data[a].perf.value > pvd.data[b].perf.value;
        });

        for (auto idx : order) {
            auto &entry = pvd.data[idx];
            auto &sv = data.allSVs()[entry.sv];
            float perf = entry.perf.value;
            float eff = perf / pvd.absRoofline;
            bool isTop = (int(idx) == pvd.topIndex);

            s << (isTop ? "  * " : "    ");
            s << std::fixed << std::setprecision(1) << (eff * 100) << "%  "
              << std::setprecision(0) << toUnits(perf) << unitStr << "  "
              << std::defaultfloat;
            s << mdClient.svToString(sv) << "\n";
        }
        s << "\n";
    }

    return s.str();
}
