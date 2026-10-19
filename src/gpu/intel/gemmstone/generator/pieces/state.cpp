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


#include "state.hpp"
#include "hw_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;

Subregister SubregisterPair::getReg(int idx) const
{
    auto r = regs[idx & 1];
    if (negative)
        r = -r;
    return r;
}

Subregister SubregisterPair::getRegAvoiding(HW hw, const RegData &rd) const
{
    if (Bundle::same_bank(hw, rd, regs[0]))
        return getReg(1);
    else
        return getReg(0);
}

VirtualFlag CommonState::allocVFlag(ngen::HW hw, int n)
{
    auto flag = raVFlag.allocVirtual(n);

    if (vflagsEnabled()) {
        int ne = elementsPerGRF<uint16_t>(hw);
        int nvflag = vflagStorage.getLen() * ne;

        for (int v0 = nvflag; v0 <= flag.idx; v0 += ne)
            vflagStorage.append(ra.alloc());
    }

    return flag;
}

void CommonState::wipeActiveVFlags()
{
    for (int i = 0; i < int(activeVFlags.size()); i++)
        if (!raVFlag.isLocked(VirtualFlag(i)))
            activeVFlags[i].clear();
    raVFlag.freeVFlagTempAllocs();
}

void CommonState::allocEmulate64Temp(const EmulationStrategy &estrategy)
{
    int ntemp = 0;
    if (estrategy.emulateDWxDW)  ntemp = 1;
    if (estrategy.emulate64)     ntemp = 2;
    if (estrategy.emulate64_mul) ntemp = 2;

    for (int q = 0; q < ntemp; q++)
        emulate.temp[q] = ra.alloc();
}

GRF GEMMState::r0InfoGRF() const
{
    if (r0_info.isValid()) {
        if (r0_info.isARF()) stub();
        return GRF{r0_info.getBase()};
    }
    return GRF{};
}

void GEMMState::setTacc(Type T)
{
    Tacc = T;
    C_layout.cast(T);
}

GEMMSTONE_NAMESPACE_END
