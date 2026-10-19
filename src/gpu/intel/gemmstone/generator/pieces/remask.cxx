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
#include "map.hpp"
#include "layout_utils.hpp"
#include "remask.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


template <HW hw>
void Generator<hw>::setupTeardownRemask(Type T, int index, bool setup, int nq, Subregister remQ,
                                        const CommonStrategy &strategy, CommonState &state,
                                        int fixedOffQ, const Subregister &variableOffQ)
{
    if (T.paddedSize() > 4) T = Type::u32;

    if (setup) {
        bool subByte = T.isSubByte();
        bool thirdByte = T.is3();
        bool packedSubByte = subByte && !thirdByte;
        Type Tsrc = T;
        if (packedSubByte) {
            nq = div_up(nq, Tsrc.perByte());
            T = Type::u8;
        } else if (thirdByte) {
            // Mask u3 data at whole 8-element group granularity.
            T = Type::u8;
        }

        auto maskAlloc = state.ra.alloc_range(div_up(T.paddedSize(), 2) * div_up(nq * 2, GRF::bytes(hw)));
        auto masks = state.remaskRegs[index] = maskAlloc;
        int ne16 = elementsPerGRF(hw, Type::u16);
        int n16 = std::min(nq, ne16);
        int ne = elementsPerGRF(hw, T);
        auto flag = state.raVFlag.tryAlloc((n16 > 16) ? 2 : 1);
        bool useCMP = flag.isValid() && (T.paddedSize() < 4);     // apparent issues with 4b sequence

        bool freeRemQ = false;
        bool haveVariableOff = variableOffQ.isValid();
        bool haveFixedOff = (fixedOffQ != 0);

        if (haveVariableOff || haveFixedOff || packedSubByte) {
            auto nremQ = state.ra.alloc_sub<uint32_t>();
            freeRemQ = true;

            if (haveVariableOff && haveFixedOff)
                eadd3(1, nremQ, remQ, -variableOffQ, -fixedOffQ);
            else if (haveVariableOff)
                add(1, nremQ, remQ, -variableOffQ);
            else if (haveFixedOff)
                add(1, nremQ, remQ, -fixedOffQ);
            if (packedSubByte)
                divUp(nremQ, (haveVariableOff || haveFixedOff) ? nremQ : remQ, Tsrc.perByte(), strategy, state, true);
            remQ = nremQ;
        }
        mov<uint16_t>(8, masks[0][0](1), Immediate::uv(0,1,2,3,4,5,6,7));
        if (nq > 8)
            mov<uint16_t>(8, masks[0][8](1), Immediate::uv(8,9,10,11,12,13,14,15));
        if (GRF::bytes(hw) > 32 && nq > 16)
            add<uint16_t>(16, masks[0][16](1), masks[0][0](1), 16);
        add<uint16_t>(n16, masks[0], masks[0], -remQ.w());
        if (!useCMP) for (int q0 = n16; q0 < nq; q0 += n16)
            add<uint16_t>(n16, masks[q0 / n16], masks[0], q0);
        switch (T.paddedSize()) {
            case 1:
            case 2:
                if (useCMP) {
                    for (int q0 = n16; q0 < nq; q0 += n16)
                        cmp<int16_t>(n16 | lt | flag, masks[q0 / n16], masks[0], -q0);
                    asr<int16_t>(n16, masks[0], masks[0], 15);
                } else {
                    map(hw, Type::s16, masks, masks, strategy, [this](int simd, const RegData &r1, const RegData &) {
                        asr(simd, r1, r1, 15);
                    });
                }
                if (T.paddedSize() == 1)
                    for (int q0 = 0; q0 < nq; q0 += n16)
                        mov(n16, masks[q0 / ne].ub(q0 % ne)(1), masks[q0 / n16].ub(1)(2));
                break;
            case 4:
                for (int qq0 = div_up(nq, ne16) - 1; qq0 >= 1; qq0--) {
                    useCMP ? cmp(ne16 | lt | flag, masks[qq0 * 2].d(), masks[qq0].w(), -qq0 * ne16)
                           : asr(ne16, masks[qq0 * 2].d(), masks[qq0].w(), 15);
                }
                if (nq > (ne16 / 2))
                    asr(ne16 / 2, masks[1].d(), masks[0].w(ne16 / 2)(1), 15);
                asr(ne16 / 2, masks[0].d(), masks[0].w(), 15);
                break;
            default: stub();
        }

        if (freeRemQ) state.ra.safeRelease(remQ);
        state.raVFlag.safeRelease(flag);
    } else
        state.ra.safeRelease(state.remaskRegs[index]);
}

template <HW hw>
void Generator<hw>::remaskLayout(int index, bool column,
                                 const RegisterLayout &layout, const GRFMultirange &regs,
                                 const CommonStrategy &strategy, CommonState &state, int offset)
{
    auto T = layout.type();
    for (auto &block: layout) {
        auto crosspack = block.crosspack;
        bool colMajor = block.colMajor;
        int component = block.component;
        int nx = colMajor ? block.nr : block.nc;
        int ny = colMajor ? block.nc : block.nr;
        auto Tr = T;

        const int qCX = -1;

        for (int y0 = 0; y0 < ny; y0 += crosspack) {
            for (int x0 = 0; x0 < nx; ) {
                auto ii0 = colMajor ? x0 : y0;
                auto jj0 = colMajor ? y0 : x0;
                auto i0 = ii0 + block.offsetR;
                auto j0 = jj0 + block.offsetC;

                int ne;
                auto sub = block.find(T, ii0, jj0, regs, &ne, qCX, component);

                auto necp = ne * crosspack;
                necp = std::min(necp, 2 * elementsPerGRF(hw, Tr));
                if (!T.is3() && ((necp * Tr) & 3)) stub();

                int mstride;
                Type mtype = Type::u32;
                int dwCrosspack = T.is3() ? crosspack : std::max(1, 4 / Tr);

                if (colMajor != column && crosspack == 1)
                    mstride = 1;
                else if (colMajor != column && crosspack == dwCrosspack)
                    mstride = 1, mtype = Tr.asSignedInt();
                else if (colMajor == column && crosspack == dwCrosspack)
                    mstride = 0;
                else
                    stub();

                int moff = (offset + (column ? j0 : i0)) * Tr / mtype;
                int mreg = moff / elementsPerGRF(hw, mtype);
                int msub = moff % elementsPerGRF(hw, mtype);
                auto mask = state.remaskRegs[index][mreg].sub(msub, mtype.ngen());
                auto mregion = mask(mstride);
                if (Tr.paddedSize() > 4 && mstride == 1)
                    mregion = mask(1, Tr.size() / 4, 0);
                and_<uint32_t>((necp * Tr) / 4, sub.ud()(1), mregion, sub.ud()(1));
                x0 += necp / crosspack;
            }
        }
    }
}

template <HW hw>
void Generator<hw>::remaskLayoutSingle(int index, bool column, int nq, Subregister remQ,
                                       const RegisterLayout &layout, const GRFMultirange &regs,
                                       const CommonStrategy &strategy, CommonState &state,
                                       int fixedOffQ, const Subregister &variableOffQ, int maskOff)
{
    setupTeardownRemask(layout.type(), index, true, nq, remQ, strategy, state, fixedOffQ, variableOffQ);
    remaskLayout(index, column, layout, regs, strategy, state, maskOff);
    setupTeardownRemask(layout.type(), index, false, nq, remQ, strategy, state);
}

GEMMSTONE_NAMESPACE_END
