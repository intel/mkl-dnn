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

#ifndef GEMMSTONE_TOOLS_MULTIOPT_CLIENT_HPP
#define GEMMSTONE_TOOLS_MULTIOPT_CLIENT_HPP

#include <vector>

#include "utils.hpp"

namespace multiopt {

template <typename PV, typename SV> struct AgentRequest;

/* Virtual base class for the optimization client */
template <typename PV, typename SV>
class Client {
protected:
    using LegalSVEntries = std::array<std::vector<ElementType<SV>>, SV::size()>;

public:
    /* PV operations */
    virtual float absRoofline(const PV &pv) = 0;                    // Compute absolute performance roofline for this PV.
    virtual float distance(const PV &pv1, const PV &pv2) = 0;       // Measure distance between two PVs, in some user-defined metric.
    virtual PV randomPV() = 0;                                      // Randomly generate a new PV.

    /* SV operations */
    virtual bool legalize(SV &sv, const PV &pv) = 0;                // Modify given SV to be legal for given PV if possible. Returns true upon success.

    virtual const SV &minLegalSV() = 0;                             // Bounds on legal SVs.
    virtual const SV &maxLegalSV() = 0;

    virtual const SV &minRandomSV() = 0;                            // Bounds on randomly-generated SVs.
    virtual const SV &maxRandomSV() = 0;                            // These may be tighter than the legal bounds above.

    virtual const LegalSVEntries &legalSVEntries();                 // Sorted list of legal values for each SV entry. If the list for an index is empty,
                                                                    //   legal values are given by minLegalSV/maxLegalSV.

    inline virtual SV randomSV(const PV &pv);                       // Randomly generate a new SV.

    /* Benchmarking */
    virtual void preload(const AgentRequest<PV, SV> &req) = 0;      // Do any pre-work needed (JIT, precompute) for a benchmarking request.
    virtual void benchmark(AgentRequest<PV, SV> &req) = 0;          // Collect performance data.

    // True if req's kernel is ready to benchmark, waiting up to maxWaitSeconds.
    virtual bool ready(const AgentRequest<PV, SV> &req, double maxWaitSeconds = 0.0) { return true; }
};


/* Default (empty) legal SV entry lists */
template <typename PV, typename SV>
const Client<PV, SV>::LegalSVEntries &Client<PV, SV>::legalSVEntries()
{
    static LegalSVEntries empty{};
    return empty;
}

/* Default random SV generation */
template <typename PV, typename SV>
SV Client<PV, SV>::randomSV(const PV &pv)
{
    SV sv;
    auto &minSV = minRandomSV();
    auto &maxSV = maxRandomSV();
    auto &legal = legalSVEntries();

    while (true) {
        for (int i = 0; i < SV::size(); i++) {
            if (legal[i].empty())
                sv[i] = randInt(minSV[i], maxSV[i]);
            else
                sv[i] = legal[i][randInt(legal[i].size())];
        }
        if (legalize(sv, pv))
            return sv;
    }
}

} /* namespace multiopt */

#endif
