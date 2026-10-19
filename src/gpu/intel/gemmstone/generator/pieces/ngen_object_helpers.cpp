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


#include "ngen_object_helpers.hpp"
#include "ngen_emulation.hpp"
#include "hw_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;


void movePipes(Subregister &s, bool sizeCanChange)
{
    DataType type = s.getType();

    switch (type) {
        case DataType::bf8:
        case DataType::hf8: type = DataType::ub; break;
        case DataType::bf:
        case DataType::hf: type = DataType::uw; break;
        case DataType::tf32:
        case DataType::f:  type = DataType::ud; break;
        case DataType::df: if (sizeCanChange) type = DataType::ud; break;
        case DataType::w:
        case DataType::uw: type = DataType::hf; break;
        case DataType::d:
        case DataType::ud: type = DataType::f; break;
        case DataType::q:
        case DataType::uq: if (sizeCanChange) type = DataType::f; break;
        default: break;
    }

    s = s.reinterpret(0, type);
}

void moveToIntPipe(Subregister &s)
{
    DataType type = s.getType();

    switch (type) {
        case DataType::bf8:
        case DataType::hf8: type = DataType::ub; break;
        case DataType::bf:
        case DataType::hf: type = DataType::uw; break;
        case DataType::q:
        case DataType::uq:
        case DataType::f:
        case DataType::tf32:
        case DataType::df: type = DataType::ud; break;
        default: break;
    }

    s = s.reinterpret(0, type);
}

void moveToIntPipe(int esize, RegData &s)
{
    switch (s.getType()) {
        case DataType::bf8:
        case DataType::hf8: s.setType(DataType::ub); break;
        case DataType::bf:
        case DataType::hf: s.setType(DataType::uw); break;
        case DataType::q:
        case DataType::uq:
        case DataType::tf32:
        case DataType::f:  s.setType(DataType::ud); break;
        case DataType::df:
            s.setType(DataType::uq);
            EmulationImplementation::makeDWPair(s, esize);
            break;
        default: break;
    }
}

int elementDiff(HW hw, const RegData &r1, const RegData &r2)
{
    return elementsPerGRF(hw, r1.getType()) * (r1.getBase()          - r2.getBase())
                                            + (r1.getLogicalOffset() - r2.getLogicalOffset());
}

CacheSettingsLSC makeL1Uncacheable(CacheSettingsLSC c)
{
    switch (c) {
        case CacheSettingsLSC::L1UC_L3UC:
        case CacheSettingsLSC::L1C_L3UC:
        case CacheSettingsLSC::L1S_L3UC:
            return CacheSettingsLSC::L1UC_L3UC;
        default:
            return CacheSettingsLSC::L1UC_L3C;
    }
}

GEMMSTONE_NAMESPACE_END
