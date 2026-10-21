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

#include "gemmstone/problem.hpp"

#include <sstream>

GEMMSTONE_NAMESPACE_START

using namespace ngen;


// Transpose a GEMM problem.
void GEMMProblem::transpose()
{
    std::swap(A, B);
    std::swap(AO, BO);
    std::swap(A_scale, B_scale);
    std::swap(Ag, Bg);
    std::swap(Ta, Tb);
    std::swap(Ta_ext, Tb_ext);
    std::swap(Tao, Tbo);
    std::swap(Ta_scale, Tb_scale);
    std::swap(aOffset, bOffset);
    std::swap(aOffsetAfterScale, bOffsetAfterScale);
    std::swap(aoPtrDims, boPtrDims);
    std::swap(asPtrDims, bsPtrDims);
    std::swap(aqGroupM, bqGroupN);
    std::swap(aqGroupK, bqGroupK);
    std::swap(sumA, sumB);
    postOps.transpose();
    for (auto &bsrc: binary)
        bsrc.transpose();
    A.transpose();
    B.transpose();
    C.transpose();
    AO.transpose();
    BO.transpose();
    CO.transpose();
    A_scale.transpose();
    B_scale.transpose();
    Ag.transpose();
    Bg.transpose();
}

static inline void append(std::ostringstream &ss, const Scalar &x);
static inline void append(std::ostringstream &ss, Type T, const MatrixAddressing &atype);
static inline void append(std::ostringstream &s, Type T1);
static inline void append(std::ostringstream &s, Type T1, Type T2);

std::string GEMMProblem::toString() const
{
    std::ostringstream ss;
    ss.imbue(std::locale::classic());

    switch (batch) {
        default:                                      break;
        case BatchMode::Strided:    ss << "batchs ";  break;
        case BatchMode::Nonstrided: ss << "batchn ";  break;
        case BatchMode::Variable:   ss << "batchnv "; break;
    }

    auto appendQString = [&](char matrix, Type T, int ptrDims, int xqGroupR, int xqGroupC, bool precalc = false) {
        ss << matrix;
        if (ptrDims < -1 || ptrDims > 2) return;
        ss << "[";
        char tchar = "spvg"[ptrDims + 1];
        if (precalc) tchar = std::toupper(tchar);
        ss << tchar;
        if (ptrDims == 2)
            ss << std::max(xqGroupR, 1) << 'x' << std::max(xqGroupC, 1);
        ss << ',';
        append(ss, T);
        ss << ']';
    };

    bool offseta = (aOffset != ABOffset::None);
    bool offsetb = (bOffset != ABOffset::None);
    bool offsetc = (cOffset == COffset::Post);
    if (offseta || offsetb || offsetc) {
        ss << "offset";
        if (offseta) appendQString('a', Tao, aoPtrDims, aqGroupM, aqGroupK, (aOffset == ABOffset::Load));
        if (offsetb) appendQString('b', Tbo, boPtrDims, bqGroupK, bqGroupN, (bOffset == ABOffset::Load));
        if (offsetc) appendQString('c', Tco, coPtrDims, 0, 0);
        ss << ' ';
    }

    if (aScale2D() || bScale2D()) {
        ss << "scale";
        if (aScale2D()) appendQString('a', Ta_scale, 2, aqGroupM, aqGroupK);
        if (bScale2D()) appendQString('b', Tb_scale, 2, bqGroupK, bqGroupN);
        ss << ' ';
    }

    if (sumA) ss << "suma ";
    if (sumB) ss << "sumb ";

    if (cOffset == COffset::Pre) ss << "bias ";

    ss << "gemm";

    ss << ' ';
    append(ss, Ta_ext, Ta);
    append(ss, Tb_ext, Tb);
    append(ss, Tc, Tc_ext);
    if (Ts != Tc)
        append(ss, Ts);
    ss << ' ';
    append(ss, Ta_ext, A);
    append(ss, Tb_ext, B);
    append(ss, Tc_ext, C);

    return ss.str();
}

std::string GEMMProblem::scalarsToString() const
{
    std::ostringstream ss;
    ss.imbue(std::locale::classic());
    append(ss, alpha);
    ss << ' ';
    append(ss, beta);
    return ss.str();
}


static inline void append(std::ostringstream &ss, const Scalar &x)
{
    switch (x.getType()) {
        case Scalar::Variable:    ss << '-'; break;
        case Scalar::Pointer:     ss << '@'; break;
        case Scalar::RealPointer: ss << 'R'; break;
        case Scalar::Fixed:       ss << int(x); break;
    }
}

static inline void append(std::ostringstream &ss, Type T, const MatrixAddressing &atype)
{
    ss << "NTAB"[static_cast<int>(atype.layout)];
    if (atype.crosspack > 1)
        ss << int(atype.crosspack);
    if (atype.tileR || atype.tileC) {
        ss << '#' << int(atype.tileR);
        ss << ',' << int(atype.tileC);
    }
    if (isPacked(atype.layout))
        ss << '%' << atype.packSize;
    if (atype.alignment != atype.defaultAlignment(T))
        ss << '@' << int(atype.alignment);
}

static inline void append(std::ostringstream &ss, Type T1)
{
    ss << precisionChar(T1);
}

static inline void append(std::ostringstream &ss, Type T1, Type T2)
{
    if (T1 == T2)
        append(ss, T1);
    else {
        ss << '[';
        append(ss, T1);
        append(ss, T2);
        ss << ']';
    }
}

GEMMSTONE_NAMESPACE_END
