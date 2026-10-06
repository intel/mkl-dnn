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


#include "gemmstone/generator.hpp"
#include "hw_utils.hpp"
#include "kernel_queries.hpp"
#include "layout_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;

template <HW hw>
std::string Generator<hw>::getNotes()
{
    return noteStream.str();
}

template <HW hw>
void Generator<hw>::gemmKernelNotes(GEMMProblem &problem, GEMMStrategy &strategy)
{
#ifdef ASM_OUTPUT
    noteStream << "// ";
#endif
    driverInfoNotes(driverInfo(problem, strategy));
    noteStream << std::endl;
}

template <HW hw>
void Generator<hw>::driverInfoNotes(const CommonDriverInfo &info)
{
    info.dump(noteStream);
}

// Static performance analysis of a GEMM kernel.
template <HW hw>
void Generator<hw>::gemmKernelAnalysis(GEMMProblem &problem, GEMMStrategy &strategy, GEMMState &state)
{
    auto Ta = problem.Ta, Tb = problem.Tb, Tc = problem.Tc;
    auto Ta_ext = problem.Ta_ext, Tb_ext = problem.Tb_ext;

    bool cColMajor = isRegisterColMajor(Tc, problem.C, strategy.C);
    double fmaPerCycle = elementsPerGRF(hw, Tc.real());
    if (strategy.systolic) {
        fmaPerCycle = elementsPerGRF(hw, Ta.real());
        if (hw == HW::XeHP)
            fmaPerCycle *= 8;
        else if (hw == HW::XeHPG)
            fmaPerCycle *= 4;
        else if (hw == HW::XeHPC)
            fmaPerCycle *= 8;
    } else if (hw >= HW::Gen12LP && problem.isIGEMM())
        fmaPerCycle *= Tc.real().size();

    if (Tc.isComplex()) fmaPerCycle *= 0.25;

    int l1RBW, l3RBW, slmRBW, slmWBW;
    switch (hw) {
        case HW::Gen12LP:
            l1RBW = l3RBW = 64;
            slmRBW = slmWBW = 128;
            break;
        case HW::XeHP:
        case HW::XeHPG:
            l1RBW = l3RBW = 64;
            slmRBW = 256;
            slmWBW = 128;
            break;
        case HW::XeHPC:
        case HW::Xe2:       /* TODO check */
        case HW::Xe3:       /* TODO check */
        case HW::Xe3p:       /* TODO check */
            l3RBW = 64;
            l1RBW = slmRBW = 512;
            slmWBW = 256;
            break;
        default: return;
    }

    int eusPerSS = eusPerSubslice(hw);
    int slm = int(slmCapacity(problem.product));
    int threads = threadsPerEU(problem.product, strategy);

    int eusPerWG = strategy.wg[LoopM] * strategy.wg[LoopN] * strategy.wg[LoopK] * strategy.wgPadFactor;
    if (strategy.splitCopy)
        eusPerWG *= 2;
    int threadsPerWG = div_up(eusPerWG, eusPerSS);

    double compute = strategy.unroll[LoopM] * strategy.unroll[LoopN] / fmaPerCycle;

    double l3TrafficA = strategy.unroll[LoopM] * Ta_ext;
    double l3TrafficB = strategy.unroll[LoopN] * Tb_ext;
    double l3UniqueTrafficA = l3TrafficA / strategy.wg[LoopN];
    double l3UniqueTrafficB = l3TrafficB / strategy.wg[LoopM];

    if (strategy.dpasw)
        (cColMajor ? l3TrafficB : l3TrafficA) /= 2;

    if (strategy.slmA) l3TrafficA = l3UniqueTrafficA;
    if (strategy.slmB) l3TrafficB = l3UniqueTrafficB;

    double l1Cycles = 0, l3Cycles = 0;

    if (l1RBW > l3RBW) {
        l1Cycles = (l3TrafficA + l3TrafficB) * eusPerSS / l1RBW;
        if (!strategy.slmA) l3TrafficA = l3UniqueTrafficA;
        if (!strategy.slmB) l3TrafficB = l3UniqueTrafficB;
    }

    l3Cycles = (l3TrafficA + l3TrafficB) * eusPerSS / l3RBW;

    double slmWTrafficA = strategy.slmA ? ((l3TrafficA * Ta.bits()) / Ta_ext.bits()) : 0;
    double slmWTrafficB = strategy.slmB ? ((l3TrafficB * Tb.bits()) / Tb_ext.bits()) : 0;

    double slmWCycles = (slmWTrafficA + slmWTrafficB) * eusPerSS / slmWBW;

    double slmRTrafficA = strategy.slmA ? strategy.unroll[LoopM] * Ta : 0;
    double slmRTrafficB = strategy.slmB ? strategy.unroll[LoopN] * Tb : 0;

    if (strategy.dpasw)
        (cColMajor ? slmRTrafficB : slmRTrafficA) /= 2;

    double slmRCycles = (slmRTrafficA + slmRTrafficB) * eusPerSS / slmRBW;
    double slmCycles = slmRCycles + slmWCycles;

    auto slmTotal = double(gemmSLMSize(hw, problem, strategy));
    int slmThreads = std::min(threads, int(slm / std::max(1., slmTotal)) * threadsPerWG);

    double l3Bottleneck = compute / l3Cycles;
    double slmBottleneck = (slmCycles > 0) ? compute / slmCycles : 1.0;

    std::stringstream s;

    if (l1Cycles > 0) {
        double l1Bottleneck = compute / l1Cycles;
        s << "L1 bottleneck:    ";
        if (l1Bottleneck < 1.)
            s << "\x1B[31;1m" << l1Bottleneck * 100 << "%\x1B[0m";
        else
            s << "none";
        s << " (" << l1Cycles << " / " << compute << " cycles)\n";
    }

    s << "L3 bottleneck:    ";
    if (l3Bottleneck < 1.)
        s << "\x1B[31;1m" << l3Bottleneck * 100 << "%\x1B[0m";
    else
        s << "none";
    s << " (" << l3Cycles << " / " << compute << " cycles)\n";

    s << "SLM bottleneck:   ";
    if (slmBottleneck < 1.)
        s << "\x1B[31;1m" << slmBottleneck * 100 << "%\x1B[0m";
    else
        s << "none";
    s << " (" << slmRCycles << '+' << slmWCycles << " / " << compute << " cycles)\n";

    s << "SLM thread limit: ";
    if (slmThreads < threads) s << "\x1B[31;1m";
    s << slmThreads  << " / " << threads;
    if (slmThreads < threads) s << "\x1B[0m";
    s << " threads (" << slmTotal*(threads/threadsPerWG)/1024. << " / " << slm/1024. << " KB)\n";

    comment(s.str());
}

GEMMSTONE_NAMESPACE_END
