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

#ifndef GEMMSTONE_TOOLS_MULTIOPT_EXTRACTOR_HPP
#define GEMMSTONE_TOOLS_MULTIOPT_EXTRACTOR_HPP

#include "data.hpp"
#include "serialization.hpp"
#include "settings.hpp"

namespace multiopt {

template <typename PV, typename SV>
class Extractor {
public:
    Extractor(const DataStore<PV, SV> &data_) : data(data_) {}

    //             (strategy#, #PVs covered)
    std::vector<std::pair<int, int>> extract(float minRelativeEfficiency = Settings::extractorDefaultMinRelEff, int svLimit = 0) const;

    friend Serialized &operator<<(Serialized &s, const Extractor &e) { return s; }
    friend Serialized &operator>>(Serialized &s, Extractor &e)       { return s; }

protected:
    const DataStore<PV, SV> &data;
};

} /* namespace multiopt */

#endif
