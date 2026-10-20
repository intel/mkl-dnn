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

GEMMSTONE_NAMESPACE_START

using namespace ngen;


#ifdef STANDALONE
// Debug output of nGEN objects.
std::ostream &operator<<(std::ostream &str, const Register &reg)
{
    LabelManager lm;
    if (reg.isInvalid())
        str << "(none)";
    else
        reg.outputText(str, PrintDetail::base, lm);
    return str;
}

std::ostream &operator<<(std::ostream &str, const Subregister &subreg)
{
    LabelManager lm;
    if (subreg.isInvalid())
        str << "(none)";
    else
        subreg.outputText(str, PrintDetail::sub, lm);
    return str;
}

std::ostream &operator<<(std::ostream &str, const GRFRange &range)
{
    if (range.isInvalid())
        str << "(none)";
    else {
        str << 'r' << range.getBase();
        if (range.getLen() > 1)
            str << "-r" << (range.getBase() + range.getLen() - 1);
    }
    return str;
}

std::ostream &operator<<(std::ostream &str, const GRFMultirange &ranges)
{
    bool first = true;
    for (auto &r : ranges.ranges) {
        if (!first) str << ' ';
        str << r;
        first = false;
    }
    return str;
}

template <typename T>
std::ostream &operator<<(std::ostream &str, const std::vector<T> &vec)
{
    bool comma = false;
    for (auto e : vec) {
        if (comma) str << ", ";
        str << e;
        comma = true;
    }
    return str;
}

// Dump a state structure.
void GEMMState::dump(std::ostream &s)
{
    s << "----------------------------------\n";
    s << " A addrs       " << A_addrs << '\n';
    s << "     [slm]     " << Ai_addrs << " -> " << Ao_addrs << '\n';
    s << " B addrs       " << B_addrs << '\n';
    s << "     [slm]     " << Bi_addrs << " -> " << Bo_addrs << '\n';
    s << " C addrs       " << C_addrs[0] << ' ' << C_addrs[1] << '\n';
    s << "  A regs       " << A_regs << '\n';
    s << "     [repack]  " << Ar_regs << '\n';
    s << "     [slm]     " << Ai_regs << " -> " << Ao_regs << '\n';
    s << "  B regs       " << B_regs << '\n';
    s << "     [repack]  " << Br_regs << '\n';
    s << "     [slm]     " << Bi_regs << " -> " << Bo_regs << '\n';
    s << "  C regs       " << C_regs << '\n';
    s << "----------------------------------\n";
}
#endif /* STANDALONE */

GEMMSTONE_NAMESPACE_END
