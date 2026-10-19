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


#include "cooperative_split.hpp"
#include "internal/utils.hpp"

GEMMSTONE_NAMESPACE_START


// Split A/B matrix between threads.
bool tryCoopSplit(bool isA, int &splitR, int &splitC, int r, int c, int mnFull, CoopSplit stype, const MatrixAddressing &atype, const GEMMStrategy &strategy, AccessType pfAccess, Type T)
{
    auto &mn      = isA ? r : c;
    auto &k       = isA ? c : r;
    auto &splitMN = isA ? splitR : splitC;
    auto &splitK  = isA ? splitC : splitR;
    auto tileMN   = isA ? atype.tileR : atype.tileC;
    auto tileK    = isA ? atype.tileC : atype.tileR;
    int threads   = strategy.wg[isA ? LoopN : LoopM];

    bool ok = false;

    // HW clamps block 2D accesses at surface edges.
    bool allowUnevenSplit = isBlock2D(pfAccess);
    int alignX = 1, alignY = 1;
    if (allowUnevenSplit) {
        alignX = 32 / gcd(32, T.bits());
        if (pfAccess == AccessType::Block2DVNNI) alignY = alignX;
        if (pfAccess == AccessType::Block2DTranspose && T.paddedSize() > 4) alignY = 8;
    }

    auto split = [&](int n, bool rows) {
        ok = (n % threads == 0);
        if (ok || !allowUnevenSplit) return n / threads;
        int s = align_up(div_up(n, threads), (rows == isColMajor(atype.layout)) ? alignX : alignY);
        ok = (div_up(n, s) == threads);     // no idle threads
        return s;
    };

    switch (stype) {
        case CoopSplit::FullK:
            threads = strategy.wg[LoopM] * strategy.wg[LoopN];
            /* fall through */
        case CoopSplit::K:
            splitMN = (stype == CoopSplit::K) ? mn : mnFull;
            splitK = split(k, !isA);
            break;
        case CoopSplit::MN:
            splitMN = split(mn, isA);
            splitK = k;
            break;
        case CoopSplit::Linear: {
            int elems = r * c;
            ok = (elems % threads == 0);
            int selems = elems / threads;
            int cp = atype.crosspack;

            if (!tileK) tileK = k;
            if (!tileMN) tileMN = mn;

            // First try splitting into tiles in k dimension.
            if (selems >= (tileK * mn)) {
                ok &= (selems % (tileK * mn) == 0);
                splitMN = mn;
                splitK = k / threads;
                break;
            }

            ok &= (threads % (k / tileK) == 0);
            if (!ok) break;
            threads /= (k / tileK);

            // Then try splitting into tiles in m/n dimensions as well.
            if (selems >= (tileK * tileMN)) {
                ok &= (selems % (tileK * tileMN) == 0);
                splitMN = mn / threads;
                splitK = tileK;
                break;
            }

            ok &= (threads % (mn / tileMN) == 0);
            if (!ok) break;
            threads /= (mn / tileMN);

            // Then try splitting each tile in the k dimension.
            if (selems >= (cp * tileMN)) {
                ok &= (selems % (cp * tileMN) == 0);
                splitMN = tileMN;
                splitK = tileK / threads;
                break;
            }

            ok &= (threads % (tileK / cp) == 0);
            if (!ok) break;
            threads /= (tileK / cp);

            // Finally try splitting in the m/n dimensions.
            ok &= (selems % cp == 0);
            splitMN = tileMN / threads;
            splitK = cp;
            break;
        }
    }

    return ok;
}

void coopSplit(bool isA, int &splitR, int &splitC, int r, int c, int mnFull, CoopSplit stype, const MatrixAddressing &atype, const GEMMStrategy &strategy, AccessType pfAccess, Type T)
{
    if (!tryCoopSplit(isA, splitR, splitC, r, c, mnFull, stype, atype, strategy, pfAccess, T))
        stub("Cooperative operation cannot be split evenly between threads.");
}

// Return the natural splitting (maximizing contiguous memory accesses) for matrix A.
CoopSplit naturalSplitA(MatrixLayout layout)
{
    switch (layout) {
        case MatrixLayout::Pr:
        case MatrixLayout::Pc: return CoopSplit::Linear;
        case MatrixLayout::N:  return CoopSplit::FullK;
        case MatrixLayout::T:  return CoopSplit::MN;
        default: stub();
    }
}

CoopSplit naturalSplitB(MatrixLayout layout)
{
    return naturalSplitA(transposeLayout(layout));
}

GEMMSTONE_NAMESPACE_END
