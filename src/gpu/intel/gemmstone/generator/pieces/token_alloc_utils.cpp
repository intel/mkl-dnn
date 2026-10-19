/*******************************************************************************
* Copyright 2019 Intel Corporation
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


#include "token_alloc_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


bool allocateTokens(const RegisterLayout &layout, const GRFMultirange &regs, CommonState &state, const vector<RegisterRange> &addrs)
{
    bool success = true;
    size_t origSize = state.tokenMap.size();
    auto saveTA = state.tokenAllocator;

    for (int l = 0; l < layout.blocks(); l++) {
        if (!layout[l].isLoadBlock()) continue;

        auto token = state.tokenAllocator.tryAlloc();
        if (token < 0)
            success = false;
        else {
            auto regKey = !regs.empty() ? regs[layout[l].offsetReg()]
                                        : addrs[l];
            if (regKey.isInvalid()) continue;
            state.tokenMap.push_back(std::make_pair(regKey.getBase(), token));
        }
    }

    if (!success) {
        state.tokenAllocator = saveTA;
        state.tokenMap.resize(origSize);
    }

    return success;
}

void clearMappedTokenAllocations(HW hw, CommonState &state)
{
    for (auto &entry: state.tokenMap)
        state.tokenAllocator.release(entry.second);
    state.tokenMap.clear();
}

void clearTokenAllocations(HW hw, CommonState &state)
{
    state.tokenMap.clear();
    state.tokenAllocator = TokenAllocator(hw);
}

GEMMSTONE_NAMESPACE_END
