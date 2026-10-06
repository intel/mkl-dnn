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

#include "extractor.hpp"

#include <unordered_map>

namespace multiopt {

template <typename PV, typename SV>
std::vector<std::pair<int, int>> Extractor<PV, SV>::extract(float minRelativeEfficiency, int svLimit) const
{
    std::unordered_set<int> ipvs;
    std::unordered_multimap<int, int> topSVs;       // (sv#, pv#)

    const int npv = int(data.allPVs().size());
    const int nsv = int(data.allSVs().size());

    // Greedy algorithm: repeatedly pick SV that is "good enough" (within maxSubopt of optimal)
    //                   for the largest number of PVs.

    for (int i = 0; i < npv; i++)
        ipvs.insert(int(i));

    std::vector<std::pair<int, int>> result;

    while (!ipvs.empty()) {
        topSVs.clear();
        for (auto ipv: ipvs) {
            auto &pvd = data.find(ipv);
            if (pvd.data.empty()) continue;
            /* TODO: sort pvd data once */
            auto thresh = pvd.topData().perf * minRelativeEfficiency;
            for (auto &e: pvd.data)
                if (e.perf >= thresh)
                    topSVs.insert({e.sv, ipv});
        }

        // No PV has usable data yet; nothing left to assign.
        if (topSVs.empty())
            break;

        int bestPVCount = 0;
        int bestSV = 0;
        for (int isv = 0; isv < nsv; isv++) {
            int pvCount = topSVs.count(isv);
            if (pvCount > bestPVCount) {
                bestPVCount = pvCount;
                bestSV = isv;
            }
        }

        auto takenPVs = topSVs.equal_range(bestSV);
        for (auto it = takenPVs.first; it != takenPVs.second; it++)
            ipvs.erase(it->second);

        result.push_back({bestSV, bestPVCount});

        if (svLimit > 0 && result.size() > svLimit)
            break;
    }

    return result;
}



#if defined(MULTIOPT_PV) && defined(MULTIOPT_SV)
template class Extractor<MULTIOPT_PV, MULTIOPT_SV>;
#endif

} /* namespace multiopt */
