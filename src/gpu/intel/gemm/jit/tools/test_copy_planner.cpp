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

#include "generator/pieces/copy_plan.hpp"
#include "generator/pieces/ngen_object_helpers.hpp"
#include "gemmstone/type.hpp"

#include <functional>

using namespace ngen;
using namespace gemmstone;

DataType toDT(std::string name)
{
#define CASE(x) if (name == #x) return DataType::x;
    CASE(u2)
    CASE(s2)
    CASE(u4)
    CASE(s4)
    CASE(ub)
    CASE(b)
    CASE(uw)
    CASE(w)
    CASE(ud)
    CASE(d)
    CASE(uq)
    CASE(q)
    CASE(e2m1)
    CASE(hf8)
    CASE(bf8)
    CASE(hf)
    CASE(bf)
    CASE(f)
    CASE(df)
    CASE(tf32)
    if (name == "e8m0") return Type::ngen_e8m0();
    if (name == "nf4")  return Type::ngen_nf4();
    throw std::runtime_error("Unknown data type");
#undef CASE
}

HW toHW(std::string name)
{
#define CASE(x) if (name == #x) return HW::x;
    CASE(Gen12LP)
    CASE(XeHPG)
    CASE(XeHPC)
    CASE(Xe2)
    CASE(Xe3)
    CASE(Xe3p)
    throw std::runtime_error("Unknown HW");
#undef CASE
}

class Testbed : public AsmCodeGenerator
{
    friend class gemmstone::CopyInstruction;
    friend class gemmstone::CopyResource;
public:
    Testbed(HW hw) : AsmCodeGenerator(hw, std::cout) {}
};

int main(int argc, char **argv)
{
    int simd = 64;
    int istride = 1, ostride = 1;
    auto idt = DataType::ud, odt = DataType::ud;

    if (argc <= 5) {
        std::cerr << "Usage: " << argv[0] << " simd_size dst_type dst_stride src_type src_stride [#interleaves]\n"
                     "  e.g. " << argv[0] << " 16 hf 1 s4 1\n"
                     "\n"
                     "Types are specified with assembly syntax:\n"
                     "    df, f, hf, bf, hf8, bf8, e2m1, uq, q, ud, d, uw, w, ub, b, u4, s4, u2, s2\n"
                     "\n"
                     "Use the HW environment variable to set GPU type (XeHPG, XeHPC, etc.)\n";
        exit(1);
    }

    if (argc > 1) simd = atoi(argv[1]);
    if (argc > 2) odt = toDT(argv[2]);
    if (argc > 3) ostride = atoi(argv[3]);
    if (argc > 4) idt = toDT(argv[4]);
    if (argc > 5) istride = atoi(argv[5]);

    int interleaves = ostride;
    if (argc > 6) interleaves = atoi(argv[6]);

    HW hw = HW::XeHPC;
    if (getenv("HW")) hw = toHW(getenv("HW"));

    bool sys = (hw >= HW::XeHP);
    if (getenv("SYS")) sys = atoi(getenv("SYS"));

    InstructionModifier mod;
    if (getenv("SAT") && atoi(getenv("SAT")))
        mod |= InstructionModifier::createSaturate();

    CopyPlan plan{genericProductFamily(hw), sys};

    for (int o = 0; o < interleaves; o++) {
        if (istride == 0)
            plan.appendDestructiveMov(simd, mod, GRF(0).sub(o, odt)(ostride), GRF(10).sub(o, idt)(ostride * std::max(1, getBits(odt) / getBits(idt))));
        else
            plan.appendDestructiveMov(simd, mod, GRF(0).sub(o, odt)(ostride), GRF(10 * (o + 1)).sub(0, idt)(istride), true);
    }

    std::cout << "Before:\n";
    plan.dump();
    plan.transform();

    if (getenv("MATERIALIZE")) {
        RegisterAllocator ra(hw);
        ra.setRegisterCount(128);
        ra.claim(GRFRange(0, 128));

        int gtemp = 20;
        if (getenv("GTEMP")) gtemp = atoi(getenv("GTEMP"));
        ra.release(GRFRange(100, gtemp));

        CopyPlan::GRFAllocator grfAllocator = [&](int count, GRFRange &range) {
            if (count > 0)
                range = ra.tryAllocRange(count);
            else
                ra.release(range);
        };

        CopyPlan::FlagAllocator flagAllocator = [&](int bytes, FlagRegister &flag) {
            if (bytes > 0)
                flag = ra.tryAllocFlag(bytes == 2);
            else
                ra.release(flag);
        };

        plan.materializeTemps(grfAllocator, flagAllocator);
    }

    std::cout << "\nAfter:\n";

    if (getenv("GENERATE")) {
        Testbed g{hw};
        plan.execute(g);
    } else
        plan.dump();

    int outRegs = div_up(simd * ostride, bytesToElements(GRF::bytes(hw), odt));
    std::cout << "\nCycles/output register: " << double(plan.cycleCount()) / outRegs << std::endl;
}
