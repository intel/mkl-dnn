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

#ifndef GEMMSTONE_TOOLS_HW_QUERIES_HPP
#define GEMMSTONE_TOOLS_HW_QUERIES_HPP

#include "gemmstone/type.hpp"

#include <cstdlib>


inline int systolicDepth(const ngen::Product &product)
{
    using namespace ngen;
    switch (getCore(product.family)) {
        case HW::XeHPG:
        case HW::Xe2:
        case HW::Xe3:   return 4;
        case HW::XeHP:
        case HW::XeHPC: return 8;
        case HW::Xe3p:  return (product.family == ProductFamily::CRI) ? 16 : 4;
        default:        return 0;
    }
}

inline double throughput(const ngen::Product &product, gemmstone::Type Ta, gemmstone::Type Tb, gemmstone::Type Tc, bool systolic)
{
    using namespace ngen;
    using namespace gemmstone;

    auto hw = getCore(product.family);
    auto Ta_real = Ta.real(), Tb_real = Tb.real(), Tc_real = Tc.real();

    // Basic throughput: 1 GRF register's worth of FMAs per cycle.
    int tputBytes = (hw >= HW::XeHPC) ? 64 : 32;
    double tput = 2 * tputBytes * (8. / std::max({Ta_real.bits(), Tb_real.bits(), Tc_real.bits()}));

    // Double precision adjustments.
    if (Tc_real == Type::f64) switch (hw) {
        case HW::XeHPG: tput /= 16; break;
        case HW::XeHPC:
        case HW::Xe3p:  tput *= 2; break;
        case HW::Xe2:   tput /= 8; break;
        default: break;
    }

    if (systolic) {
        // Systolic adjustments.
        if (!(Ta_real == Type::f32 && Tb_real == Type::f32)) {
            tput = 2 * tputBytes * (8. / std::max(Ta_real.bits(), Tb_real.bits()));
            tput *= systolicDepth(product);
        }
    } else if (Ta_real.isInt8() && Tb_real.isInt8() && Tc_real.asSigned() == Type::s32) {
        // int8 adjustments.
        if (hw >= HW::Gen12LP)
            tput *= 4;
        else
            tput /= 2;
    }

    // Multicomponent GEMMs.
    int A_components = Ta.components(), B_components = Tb.components();
    int cmul = A_components * B_components;
    if (Ta.isFP() && Tb.isFP()) {
	int mc = std::min(A_components, B_components);
	cmul -= (mc * (mc - 1)) >> 1;
    }

    tput /= cmul;

    return tput;
}

inline double peakFlops(const ngen::Product &product, double euCount, double clockHz,
                                gemmstone::Type Ta, gemmstone::Type Tb, gemmstone::Type Tc, bool systolic)
{
    double flops = euCount * clockHz * throughput(product, Ta, Tb, Tc, systolic);
    flops /= (Ta.complexComponents() * Tb.complexComponents());

    if (systolic)
        if (auto e = getenv("SYS_RATIO"))   // scale systolic peak (e.g. rate-limited HW)
            flops *= std::atof(e);

    return flops;
}

#endif
