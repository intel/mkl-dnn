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

#ifndef GEMMSTONE_TOOLS_MULTIDERBY_OPTIMIZER_HPP
#define GEMMSTONE_TOOLS_MULTIDERBY_OPTIMIZER_HPP

#include "multiopt/data.hpp"
#include "multiopt/optimizer.hpp"
#include "multiopt/extractor.hpp"
#include "multiderby_client.hpp"
#include "multiderby_common.hpp"

#include <chrono>

class MultiderbyOptimizer : public multiopt::Optimizer<ProblemVector, StrategyVector>
{
    using PV = ProblemVector;
    using SV = StrategyVector;
    using super = multiopt::Optimizer<PV, SV>;

    std::chrono::steady_clock::time_point lastUpdate;
    int lastDPCount = 0;

public:
    MultiderbyOptimizer(multiopt::Client<PV, SV> &client_, const MultiderbyOptions &options,
                        const std::vector<PV> &initialPVs, const std::vector<SV> &initialSVs = std::vector<SV>{});

    friend multiopt::Serialized &operator<<(multiopt::Serialized &s, const MultiderbyOptimizer &o) {
        return s << (const super&) o;
    }
    friend multiopt::Serialized &operator>>(multiopt::Serialized &s, MultiderbyOptimizer &o) {
        return s >> (super&) o;
    }

    std::string status();
    std::string kernelList() const;
    std::string finalKernelList() const;
    std::string sensitivities() const;
    std::string timings();

    void updateTopKernels() { spreadTopSVs(); }
    std::vector<multiopt::Labeled<SV>> getTopKernels() const;
    bool haveKernelList()   { return !topKernels.empty(); }

    std::string roofline(const PV &pv);
    std::string dump() const;

protected:
    void svStatus(std::stringstream &s, int maxDisplayed = 15) const;
    void topKernelStatus(std::stringstream &s, int maxDisplayed = 15) const;
};

#endif
