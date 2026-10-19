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


#include "layout_utils.hpp"
#include "hw_utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace ngen;
using std::vector;


int untile(Type T, const MatrixAddressing &atype, int component, int i, int j, int r, int c, int tileR, int tileC, bool reverse)
{
    bool cm = isColMajor(atype.layout) ^ reverse;

    if (isPacked(atype.layout)) {
        (cm ? r : c) = atype.packSize;

        auto &pl = (cm ? c : r);
        if (atype.panelLength)
            pl = atype.panelLength;
    }

    int cpR = cm ? 1 : atype.crosspack;
    int cpC = cm ? atype.crosspack : 1;

    if (tileR == 0) tileR = r;
    if (tileC == 0) tileC = c;

    int rstride  = cm ? tileC : c;
    int cstride  = cm ? r : tileR;
    int rtstride = cm ? cpC : tileC;
    int ctstride = cm ? tileR : cpR;

    rstride *= T.components();
    cstride *= T.components();

    if (tileR == 0) tileR = 1;    /* arbitrary value */
    if (tileC == 0) tileC = 1;

    int iTile = i % tileR;
    int jTile = j % tileC;
    i -= iTile; j -= jTile;
    int iCP = iTile % cpR;
    int jCP = jTile % cpC;
    iTile -= iCP; jTile -= jCP;
    int idx = i * rstride + j * cstride + tileR * tileC * component + iTile * rtstride + jTile * ctstride + iCP + jCP;
    return idx;
}

int consecutiveElements(int r, int c, const MatrixAddressing &atype)
{
    int x = isColMajor(atype.layout) ? r : c;
    int y = isColMajor(atype.layout) ? c : r;

    if (isPacked(atype.layout)) {
        int effTileX = (atype.layout == MatrixLayout::Pc) ? atype.tileR : atype.tileC;
        int effTileY = (atype.layout == MatrixLayout::Pc) ? atype.tileC : atype.tileR;
        if (!effTileX) effTileX = atype.packSize;
        if (!effTileY) effTileY = atype.crosspack;

        if (y % effTileY == 0) {
            if (static_cast<uint32_t>(x) == atype.packSize)
                return x * y;
            else if (x % effTileX == 0)
                return x * effTileY;
        }
        if (y % atype.crosspack == 0)
            return std::min(x, effTileX) * atype.crosspack;
    }

    return x;
}

void getGranularities(const MatrixAddressing &atype, int &rgran, int &cgran)
{
    auto &xgran = isColMajor(atype.layout) ? cgran : rgran;
    auto &ygran = isColMajor(atype.layout) ? rgran : cgran;
    rgran = std::max<int>(atype.tileR, 1);
    cgran = std::max<int>(atype.tileC, 1);
    xgran = std::max<int>(xgran, atype.crosspack);
    if (isPacked(atype.layout))
        ygran = std::max<int>(ygran, atype.packSize);
}

bool needsPseudoblock(HW hw, Type T, int r, int c,
                      const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy,
                      bool writable, bool masked)
{
    if (astrategy.accessType == AccessType::PseudoBlock) return true;
    if (astrategy.accessType != AccessType::Block) return false;

    auto consecutive = consecutiveElements(r, c, atype);
    bool dwAligned = (atype.alignment & 0x3) == 0;
    bool owAligned = (atype.alignment & 0xF) == 0;
    bool pseudo = !dwAligned
               || ((consecutive * T) & 0x3)
               || (writable && ((consecutive * T) & 0xF) && !astrategy.newDP)
               || (writable && !owAligned && !astrategy.newDP)
               || (writable && masked && (T.paddedSize() & 3))
               || (masked && !owAligned && (hw >= HW::XeHP || astrategy.base.getModel() != ModelA64))
               || (astrategy.newDP && masked)
               || (hw >= HW::XeHPC && masked)
               || (hw >= HW::XeHPC && !astrategy.padded && !astrategy.newDP && ((r * c * T) & 0xF))
               || astrategy.atomic
               || (isColMajor(atype.layout) ? c : r) % atype.crosspack
               || ((astrategy.base.getModel() == ModelSLM) && !(owAligned || astrategy.newDP));

    return pseudo;
}

bool tryAllocAddrRegs(HW hw, vector<RegisterRange> &addrRegs, const RegisterLayout &layout,
                      CommonState &state, Bundle hint)
{
    auto nblocks = layout.blocks();
    bool ok = true;

    addrRegs.resize(nblocks);

    RegisterRange last;
    for (int l = 0; l < nblocks && ok; l++) {
        if (layout[l].offsetAddr == 0) {
            auto count = layout[l].addrGRFs(layout.addressing(), layout.addressingStrategy());
            if (count < 1) continue;
            last = state.ra.tryAllocRange(count, hint);
            ok &= last.isValid();
        }
        addrRegs[l] = last;
    }

    if (!ok) {
        for (auto &regs: addrRegs) state.ra.safeRelease(regs);
        addrRegs.clear();
    }

    return ok;
}

void allocAddrRegs(HW hw, vector<RegisterRange> &addrRegs, const RegisterLayout &layout,
                   CommonState &state, Bundle hint)
{
    if (!tryAllocAddrRegs(hw, addrRegs, layout, state, hint))
        throw out_of_registers_exception();
}

int getAddr0Offset(const RegisterBlock &block, const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    if (astrategy.newDP) return 0;
    if (astrategy.base.getModel() == ModelA64) return 0;
    if (block.effectiveAccessType(atype, astrategy) == AccessType::Block) return 2;
    return 0;
}

Subregister getOriginAddr(const RegisterLayout &layout, const vector<RegisterRange> &addrRegs, int *shiftOut)
{
    auto &atype = layout.addressing();
    auto &astrategy = layout.addressingStrategy();

    bool a64 = (astrategy.base.getModel() == ModelA64);

    for (int b = 0; b < layout.blocks(); b++) {
        const auto &block = layout[b];
        if ((block.offsetR != 0) || (block.offsetC != 0))
            continue;

        int off = getAddr0Offset(block, atype, astrategy);

        if (shiftOut) *shiftOut = block.addrShift;
        return addrRegs[b][0].sub(off, a64 ? DataType::uq : DataType::ud);
    }

    if (shiftOut) *shiftOut = 0;
    return Subregister();
}

int contiguityCheck(HW hw, const RegisterBlock &block, const GRFMultirange &range)
{
    auto offsetBytes = block.offsetBytes;
    if (offsetBytes & (GRF::bytes(hw) - 1))
        if (block.isLoadBlock())
            stub();
    auto offsetReg = offsetBytes >> GRF::log2Bytes(hw);
    auto lastReg = GRF::bytesToGRFs(hw, offsetBytes + block.bytes);
    if (!range.contiguous(offsetReg, lastReg - offsetReg)) stub();

    return offsetReg;
}

GRFMultirange subrange(GRFMultirange r, HW hw, Type T, const RegisterBlock &block)
{
    int ne = elementsPerGRF(hw, T);
    int ldGRFs = div_up(block.ld, ne);
    int ldUsedGRFs = div_up(block.colMajor ? block.nr : block.nc, ne);
    int td = block.colMajor ? block.nc : block.nr;

    if (ldUsedGRFs >= ldGRFs)
        return r.subrange(block.offsetReg(), block.nregs());
    else {
        int offReg = block.offsetReg();
        GRFMultirange result = r.subrange(offReg, ldUsedGRFs);
        for (int y = 1; y < td; y++) {
            offReg += ldGRFs;
            result.append(r.subrange(offReg, ldUsedGRFs));
        }
        return result;
    }
}

int checkDescriptorRemainder(HW hw, Type T, int r, int c, bool column, bool writable,
                             const MatrixAddressing &atype, const MatrixAddressingStrategy &astrategy)
{
    RegisterBlock block(hw, T, r, c, atype, astrategy, !column, column, writable, AllowFragDesc);

    if (!block.valid())                              return 0;
    if (r % block.nr || c % block.nc)                return 0;
    if (!(column ? block.descRemC : block.descRemR)) return 0;

    return column ? block.colFragment : block.rowFragment;
}


GEMMSTONE_NAMESPACE_END
