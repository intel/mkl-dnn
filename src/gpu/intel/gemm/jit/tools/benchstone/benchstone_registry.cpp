/*******************************************************************************
* Copyright 2026 Intel Corporation
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

#include "benchstone.hpp"

#include <unordered_map>

namespace {
std::unordered_map<benchstone::RuntimeBackend, benchstone::DriverFactory> &registry()
{
    static std::unordered_map<benchstone::RuntimeBackend, benchstone::DriverFactory> r;
    return r;
}
} // namespace

namespace benchstone {

void registerDriverFactory(RuntimeBackend runtime, DriverFactory factory)
{
    registry()[runtime] = std::move(factory);
}

std::unique_ptr<Driver> createDriver(RuntimeBackend runtime)
{
    auto &r = registry();
    auto it = r.find(runtime);
    if (it == r.end() || !it->second) return nullptr;
    return it->second();
}

} // namespace benchstone
