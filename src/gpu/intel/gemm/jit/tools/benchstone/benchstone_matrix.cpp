/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#include "benchstone_matrix.hpp"

#include <fstream>


using namespace gemmstone;

MatrixName transpose(MatrixName name)
{
    switch (name) {
        case MatrixName::COPYL_S:      return MatrixName::COPYU_S;
        case MatrixName::COPYU_S:      return MatrixName::COPYL_S;
        case MatrixName::TRSMLL_U:     return MatrixName::TRSMRU_U;
        case MatrixName::TRSMLU_U:     return MatrixName::TRSMRL_U;
        case MatrixName::TRSMRL_U:     return MatrixName::TRSMLU_U;
        case MatrixName::TRSMRU_U:     return MatrixName::TRSMLL_U;
        case MatrixName::TRSMLLU_U:    return MatrixName::TRSMRUU_U;
        case MatrixName::TRSMLUU_U:    return MatrixName::TRSMRLU_U;
        case MatrixName::TRSMRLU_U:    return MatrixName::TRSMLUU_U;
        case MatrixName::TRSMRUU_U:    return MatrixName::TRSMLLU_U;
        case MatrixName::TRSMLL_V:     return MatrixName::TRSMRU_V;
        case MatrixName::TRSMLU_V:     return MatrixName::TRSMRL_V;
        case MatrixName::TRSMRL_V:     return MatrixName::TRSMLU_V;
        case MatrixName::TRSMRU_V:     return MatrixName::TRSMLL_V;
        case MatrixName::ROW_RAMP:     return MatrixName::COL_RAMP;
        case MatrixName::COL_RAMP:     return MatrixName::ROW_RAMP;
        case MatrixName::ROW_STRIPE:   return MatrixName::COL_STRIPE;
        case MatrixName::COL_STRIPE:   return MatrixName::ROW_STRIPE;
        default:                       return name;
    }
}


template <typename fp>
bool _matrixRead(Matrix &dst, MatrixName name)
{
    const char *prefix;

    prefix = getenv("DATA_PATH");
    if (!prefix) return false;

    std::string fname = prefix;
    fname += '/';
    switch (name) {
        case MatrixName::GEMM_A: fname += 'A'; break;
        case MatrixName::GEMM_B: fname += 'B'; break;
        case MatrixName::GEMM_C: fname += 'C'; break;
        default: return false;
    }
    fname += ".txt";

    std::ifstream fs(fname);
    if (!fs)
        return false;

    for (integer j = 0; j < dst.n; j++) {
        for (integer i = 0; i < dst.m; i++) {
            double v;
            char eat;

            if (fs.eof()) {
                fprintf(stderr, "Input file %s ended unexpectedly at (%d,%d)\n", fname.c_str(), int(i), int(j));
                return false;
            }
            fs >> v >> std::ws;
            if (fs.peek() == ',') fs >> eat;

            dst.getUnchecked<fp>(i, j) = fp(v);
        }
    }

    return true;
}

template <typename fp>
fp randUniform(fp center, real_t<fp> radius)
{
    using fp_real = real_t<fp>;

    fp in_unit_disc;
    if constexpr (isComplex<fp>()) {
        do {
            in_unit_disc = complex<fp_real>(2 * fp_real(rand()) / fp_real(RAND_MAX) - 1,
                                            2 * fp_real(rand()) / fp_real(RAND_MAX) - 1);
        } while (std::abs(in_unit_disc) > 1);
    } else
        in_unit_disc = 2 * fp(rand()) / fp(RAND_MAX) - 1;

    return center + in_unit_disc * radius;
}

template <>
half randUniform(half center, half radius)
{
    return randUniform<float>(center, radius);
}

template <>
complex<half> randUniform(complex<half> center, half radius)
{
    return randUniform<complex<float>>(complex<float>(center.real(), center.imag()), radius);
}

template <>
bfloat8 randUniform(bfloat8 center, bfloat8 radius)
{
    return randUniform<float>(center, radius);
}

template <>
hfloat8 randUniform(hfloat8 center, hfloat8 radius)
{
    return randUniform<float>(center, radius);
}

template <typename fp>
fp randImaginary(real_t<fp> center, real_t<fp> radius)
{
    if constexpr (isComplex<fp>())
        return fp(0, randUniform<real_t<fp>>(center, radius));
    else
        return fp(0);
}

template <typename fp>
inline typename std::enable_if<!isIntegral<fp>(), fp>::type
fillVal(integer i, integer j, integer m, integer n, MatrixName name, const FillModifiers &mods = FillModifiers{})
{
    switch (name) {
        case MatrixName::ZERO:         return 0;
        case MatrixName::ONE_HALF:     return fp(1) / 2;
        case MatrixName::ONE:          return 1;
        case MatrixName::EYE:          return fp(i == j ? 1 : 0);
        case MatrixName::NANS:         return qnan<fp>();
        case MatrixName::ROW_RAMP:     return fp(1 + i);
        case MatrixName::COL_RAMP:     return fp(1 + j);
        case MatrixName::ROW_STRIPE:   return fp(0 + (i == mods.stripeIndex));
        case MatrixName::COL_STRIPE:   return fp(0 + (j == mods.stripeIndex));
        case MatrixName::DIAG_RAMP:    return fp(1 + i + j);
        case MatrixName::RANDOM:       return randUniform<fp>(0, 1);
        default:                       break;
    }

    if (mods.ones) switch (name) {
        case MatrixName::COPYL_S:      return fp(i >= j);
        case MatrixName::COPYU_S:      return fp(j >= i);
        case MatrixName::COPY_D:       return 0;
        case MatrixName::TRSMLL_V:     return fp(1 + i);
        case MatrixName::TRSMLU_V:     return fp(m - i);
        case MatrixName::TRSMRU_V:     return fp(1 + j);
        case MatrixName::TRSMRL_V:     return fp(n - j);
        default:                       return 1;
    }

    if (mods.zeros) switch (name) {
        case MatrixName::COPYTRSM_S:   return 1;
        default:                       return 0;
    }

    switch (name) {
        case MatrixName::COPYTRSM_S:   return (i == j) ? randUniform<fp>(1, 0.5) : randUniform<fp>(0, 1);
        case MatrixName::TRSMLL_U:     return (i == j) ? fp(1 + i + randImaginary<fp>(0, 0.5)) : randUniform<fp>(-0.5, 0.5);
        case MatrixName::TRSMRU_U:     return (i == j) ? fp(1 + j + randImaginary<fp>(0, 0.5)) : randUniform<fp>(-0.5, 0.5);
        case MatrixName::TRSMLU_U:     return (i == j) ? fp(m - i + randImaginary<fp>(0, 0.5)) : randUniform<fp>(-0.5, 0.5);
        case MatrixName::TRSMRL_U:     return (i == j) ? fp(n - j + randImaginary<fp>(0, 0.5)) : randUniform<fp>(-0.5, 0.5);
        case MatrixName::TRSMLLU_U:    return (i == j) ? fp(1) : fp(randUniform<fp>(-0.5, 0.5) / (1 + i));
        case MatrixName::TRSMRUU_U:    return (i == j) ? fp(1) : fp(randUniform<fp>(-0.5, 0.5) / (1 + j));
        case MatrixName::TRSMLUU_U:    return (i == j) ? fp(1) : fp(randUniform<fp>(-0.5, 0.5) / (m - i));
        case MatrixName::TRSMRLU_U:    return (i == j) ? fp(1) : fp(randUniform<fp>(-0.5, 0.5) / (n - j));
        case MatrixName::TRSMLL_V:
        case MatrixName::TRSMLU_V:
        case MatrixName::TRSMRL_V:
        case MatrixName::TRSMRU_V:     return randUniform<fp>(0.5, 0.5);
        case MatrixName::GEMM_C:       if (mods.beta0) return qnan<fp>(); // fall through otherwise
        default:                       return randUniform<fp>(0, 1);
    }
}

template <typename fp>
inline typename std::enable_if<isIntegral<fp>(), fp>::type
fillVal(integer i, integer j, integer m, integer n, MatrixName name, const FillModifiers &mods = FillModifiers{})
{
    switch (name) {
        case MatrixName::ZERO:         return 0;
        case MatrixName::ONE:          return 1;
        default:                       break;
    }

    auto val = fillVal<float>(i, j, m, n, name, mods);
    if (name != MatrixName::RANDOM) {
        return fp(val);
    }
    constexpr auto rbits = (sizeof(fp) == 1) ? 4 : (6 * sizeof(fp));
    auto range = mods.ones ? 1 : (1 << rbits);

    if (name == MatrixName::AO || name == MatrixName::BO)
        range = std::min(range, 256);

    if (!std::is_signed<fp>::value) {
        val = std::abs(val);
        if (name != MatrixName::AO_SMALL && name != MatrixName::BO_SMALL)
            if (val < 0.05) return ~fp(0);
    }
    return fp(val * range);
}

template <typename fp>
fp fillVal(MatrixName name, const FillModifiers &mods = FillModifiers{})
{
    return fillVal<fp>(0, 0, 1, 1, name, mods);
}


template <typename fp>
std::vector<fp> randomSource(integer nrand = 1407)
{
    std::vector<fp> p(nrand);
    if constexpr (isIntegral<fp>()) {
        auto range = 1 << (4 * sizeof(fp));
        for (integer ij = 0; ij < nrand; ij++)
            p[ij] = fp(randUniform<float>(0, float(range)));
    } else {
        for (integer ij = 0; ij < nrand; ij++)
            p[ij] = randUniform<fp>(0, 1);
    }
    return p;
}

/* Initialize matrix data */
template <typename fp>
void _matrixFill(Matrix &M, MatrixName name, const FillModifiers &mods)
{
    auto v = M.view();
    auto p = M.data.as<fp>();

    if (name == MatrixName::UNINIT)
        return;

    if (v.trans) {
        v.transpose();
        name = transpose(name);
    }

    auto m = v.m, n = v.n, mpad = v.mpad, npad = v.npad, ld = v.ld;

    /* Read data from file if requested */
    if (_matrixRead<fp>(M, name))
        return;

    /* Special fast initialization sequences */
    switch (name) {
        case MatrixName::SERIES_ZERO:
            memset(p, 0, M.data.bytes);
            return;
        case MatrixName::SERIES_ONE:
            for (integer ij = 0; ij < M.elements(); ij++)
                p[ij] = fp(1);
            return;
        case MatrixName::SERIES_RANDOM: {
            // Generate a small number of random values and repeat them.
            static auto rsrc = randomSource<fp>();
            integer nrand = integer(rsrc.size());
            for (integer ij0 = 0; ij0 < M.elements(); ij0 += nrand) {
                auto chunk = std::min(nrand, M.elements() - ij0);
                memmove(p + ij0, rsrc.data(), chunk * sizeof(fp));
            }
            return;
        }
        case MatrixName::ZERO_PAD: {
            for (integer j = 0; j < n; j++)
                for (integer i = m; i < mpad; i++)
                    v.nget<fp>(i, j) = 0;
            for (integer j = n; j < npad; j++)
                for (integer i = 0; i < mpad; i++)
                    v.nget<fp>(i, j) = 0;
            return;
        }
        default: break;
    }

    /* Default path: element-by-element initialization */
    for (integer j = 0; j < n; j++) {
        for (integer i = 0; i < m; i++)
            v.nget<fp>(i, j) = fillVal<fp>(i, j, m, n, name, mods);
        for (integer i = m; i < ld; i++)
            v.nget<fp>(i, j) = qnan<fp>();
    }
    for (integer j = n; j < npad; j++)
        for (integer i = 0; i < ld; i++)
            v.nget<fp>(i, j) = qnan<fp>();
}

// Avoid false positive read checks when read masking has granularity given by `align`,
//  by extending the matrix of 0's with extra 0's on the borders.
template <typename fp>
void _matrixAlignForReadCheck(Matrix &M)
{
    constexpr int align = 4 * typeEnum<fp>.perByte();
    constexpr integer nfp = align / sizeof(fp);

    auto v = M.nontransposedView();

    if (nfp <= 1 || v.ld % nfp || v.stride % nfp)
        return;

    for (integer j = 0; j < v.npad; j++)
        for (integer i = 0; i < v.mpad; i += nfp)
            if (v.nget<fp>(i, j) == 0)
                for (integer ii = 1; (ii < nfp) && (i + ii < v.mpad); ii++)
                    v.nget<fp>(i + ii, j) = 0;
}

template <typename fp>
void _matrixReverseReadCheck(Matrix &M)
{
    if constexpr (!isIntegral<fp>()) {
        if (sizeof(fp) > 8) stub();
        _matrixReverseReadCheck<storage_t<fp>>(M);
    } else {
        auto v = M.nontransposedView();

        for (integer j = 0; j < v.npad; j++)
            for (integer i = 0; i < v.mpad; i++)
                v.nget<fp>(i, j) = (v.nget<fp>(i, j) == 0) ? fp(~fp(0)) : fp(0);
    }
}

template <typename fp>
void _matrixCopyTriangle(Matrix &dst, const Matrix &src, bool lower)
{
    if (src.m != dst.m || src.n != dst.n || src.trans != dst.trans)
        throw std::runtime_error("Inconsistent dimensions");

    auto vs = src.nontransposedView();
    auto vd = dst.nontransposedView();
    lower ^= dst.trans;

    auto m = vd.m, n = vd.n;

    if (lower) {
        for (integer j = 0; j < n; j++)
            for (integer i = j + 1; i < m; i++)
                vd.nget<fp>(i, j) = vs.nget<fp>(i, j);
    } else {
        for (integer j = 0; j < n; j++)
            for (integer i = 0; i < j && i < m; i++)
                vd.nget<fp>(i, j) = vs.nget<fp>(i, j);
    }
}

template <typename fp>
void _matrixMakeTriangular(Matrix &M, bool lower, bool fillNaN)
{
    auto v = M.nontransposedView();
    lower ^= M.trans;

    auto fill = fillNaN ? qnan<fp>() : fp(0);
    auto m = v.m, n = v.n;

    if (lower) {
        for (integer j = 0; j < n; j++)
            for (integer i = 0; i < j && i < m; i++)
                v.nget<fp>(i, j) = fill;
    } else {
        for (integer j = 0; j < n; j++)
            for (integer i = j + 1; i < m; i++)
                v.nget<fp>(i, j) = fill;
    }
}

template <typename fp>
void _matrixMakeSymmetric(Matrix &M, bool lower)
{
    if (!M.square()) throw std::runtime_error("Matrix is not square");

    auto v = M.nontransposedView();
    auto n = v.n;

    lower ^= M.trans;

    if (lower) {
        for (integer j = 0; j < n; j++)
            for (integer i = 0; i < j; i++)
                v.nget<fp>(i, j) = v.nget<fp>(j, i);
    } else {
        for (integer j = 0; j < n; j++)
            for (integer i = j + 1; i < n; i++)
                v.nget<fp>(i, j) = v.nget<fp>(j, i);
    }
}

template <typename fp>
void _matrixMakeHermitian(Matrix &M, bool lower)
{
    if (!M.square()) throw std::runtime_error("Matrix is not square");

    if constexpr (!isComplex<fp>())
        _matrixMakeSymmetric<fp>(M, lower);
    else {
        auto v = M.nontransposedView();
        auto n = v.n;

        lower ^= M.trans;

        for (integer j = 0; j < n; j++)
            v.nget<fp>(j, j).imag(0);

        if (lower) {
            for (integer j = 0; j < n; j++)
                for (integer i = 0; i < j; i++)
                    v.nget<fp>(i, j) = conj(v.nget<fp>(j, i));
        } else {
            for (integer j = 0; j < n; j++)
                for (integer i = j + 1; i < n; i++)
                    v.nget<fp>(i, j) = conj(v.nget<fp>(j, i));
        }
    }
}

template <typename fp>
void _matrixMakeUnitDiagonal(Matrix &M)
{
    auto mn = std::min(M.m, M.n);

    for (integer d = 0; d < mn; d++)
        M.nget<fp>(d, d) = 1;
}

template <typename fp>
void _matrixInvertDiagonal(Matrix &M)
{
    auto mn = std::min(M.m, M.n);

    for (integer d = 0; d < mn; d++)
        M.nget<fp>(d, d) = 1 / M.nget<fp>(d, d);
}

template <typename fp>
void _matrixNegateDiagonal(Matrix &M)
{
    auto mn = std::min(M.m, M.n);

    for (integer d = 0; d < mn; d++)
        M.nget<fp>(d, d) = -M.nget<fp>(d, d);
}

template <typename fp>
void _matrixFillImagDiagonal(Matrix &M, MatrixName fill)
{
    if constexpr (isComplex<fp>()) {
        auto fillIm = fillVal<real_t<fp>>(fill);
        auto mn = std::min(M.m, M.n);

        for (integer d = 0; d < mn; d++)
            M.nget<fp>(d, d).imag(fillIm);
    }
}

template <typename fp>
void _matrixConjugate(Matrix &M)
{
    if constexpr (isComplex<fp>()) {
        auto v = M.nontransposedView();

        for (integer j = 0; j < v.n; j++)
            for (integer i = 0; i < v.m; i++)
                v.nget<fp>(i, j) = conj(v.nget<fp>(i, j));
    }
}

template <typename fpd, typename fps>
void _matrixCopyI(Matrix &dst, const Matrix &src)
{
    if (dst.trans) {
        auto dstt = dst.tview();
        return _matrixCopyI<fpd, fps>(dstt, src.tview());
    }

    auto ps = src.data.asUnchecked<fps>();
    auto pd = dst.data.asUnchecked<fpd>();
    auto m = dst.m;
    auto n = dst.n;
    auto ldd = dst.ld, lds = src.ld;

    if (m > src.m || n > src.n)
        throw std::runtime_error("Inconsistent matrix sizes");

    if (src.trans) {
        for (integer j = 0; j < n; j++)
            for (integer i = 0; i < m; i++)
                pd[i + j * ldd] = convert<fps,fpd>(ps[j + i * lds]);
    } else {
        for (integer j = 0; j < n; j++)
            for (integer i = 0; i < m; i++)
                pd[i + j * ldd] = convert<fps,fpd>(ps[i + j * lds]);
    }
}

template <typename fp>
void _matrixCopy(Matrix &dst, const Matrix &src)
{
    // One of dst/src types is guaranteed to match fp.
    // Any type not matching fp must be a supertype.
    if (dst.T == src.T)
        return _matrixCopyI<storage_t<fp>, storage_t<fp>>(dst, src);
    else if (src.T == typeEnum<fp>) {
        if (dst.T == Type::f32)
            return _matrixCopyI<float, fp>(dst, src);
        else if (dst.T == Type::f64)
            return _matrixCopyI<double, fp>(dst, src);
    } else if (dst.T == typeEnum<fp>) {
        if (src.T == Type::f32)
            return _matrixCopyI<fp, float>(dst, src);
        else if (src.T == Type::f64)
            return _matrixCopyI<fp, double>(dst, src);
    }

    throw std::runtime_error("Unexpected type combination");
}

template <typename fp>
void _matrixOffset(Matrix &dst, const Matrix &off, char offType)
{
    integer rstride = 0, cstride = 0;

    switch (offType) {
        case 'F': break;
        case 'R': rstride = 1; break;
        case 'C': cstride = 1; break;
        case 'M': rstride = 1; cstride = off.ld; break;
        default: stub();
    }

    auto m = dst.m, n = dst.n, ld = dst.ld;
    auto p = dst.data.as<fp>();
    auto po = off.data.as<fp>();

    if (dst.trans) {
        std::swap(m, n);
        std::swap(rstride, cstride);
    }

    for (integer j = 0; j < n; j++)
        for (integer i = 0; i < m; i++)
            p[i + j*ld] += po[i*rstride + j*cstride];
}

template <typename fp>
void _matrixOffsetScalar(Matrix &dst, superscalar off)
{
    auto v = dst.nontransposedView();
    auto m = v.m, n = v.n, ld = v.ld;

    fp offFP = convert<superscalar, fp>(off);

    for (integer j = 0; j < n; j++)
        for (integer i = 0; i < m; i++)
            v.nget<fp>(i, j) += offFP;
}

template <typename fp>
void _matrixLinearCombination(Matrix &M1, const Matrix &M2, superscalar alpha_, superscalar beta_)
{
    auto m = M1.m, n = M1.n, ld1 = M1.ld, ld2 = M2.ld;
    if (m != M2.m || n != M2.n || M1.trans != M2.trans)
        throw std::runtime_error("Inconsistent sizes");

    if (M1.trans)
        std::swap(m, n);

    auto p1 = M1.data.as<fp>();
    auto p2 = M2.data.as<fp>();

    using Ts = supertype_t<fp>;
    Ts alpha = alpha_, beta = beta_;

    if (beta != 0) {
        for (integer j = 0; j < n; j++)
            for (integer i = 0; i < m; i++)
                p1[i + j * ld1] = convert<Ts,fp>(alpha * p2[i + j * ld2] + beta * p1[i + j * ld1]);
    } else {
        for (integer j = 0; j < n; j++)
            for (integer i = 0; i < m; i++)
                p1[i + j * ld1] = convert<Ts,fp>(alpha * p2[i + j * ld2]);
    }
}

template <typename fp>
void _matrixDequantize(Matrix &dst, std::array<int, 2> group, const Matrix &offset, const Matrix &scale)
{
    if (dst.trans) {
        auto dstt = dst.tview();
        std::swap(group[0], group[1]);
        return _matrixDequantize<fp>(dstt, group, offset.tview(), scale.tview());
    }

    auto m = dst.m, n = dst.n;

#define APPLY(OP) do { \
                      for (integer j = 0; j < n; j++) for (integer i = 0; i < m; i++) dst.nget<fp>(i, j) OP; \
                  } while(false)

    if (offset.valid()) {
        if (offset.m == 1 & offset.n == 1)
            APPLY(-= offset.getUnchecked<fp>(0, 0));
        else if (offset.m == m && offset.n == 1)
            APPLY(-= offset.getUnchecked<fp>(i, 0));
        else if (offset.m == 1 && offset.n == n)
            APPLY(-= offset.getUnchecked<fp>(0, j));
        else
            APPLY(-= offset.getUnchecked<fp>(i / group[0], j / group[1]));
    }

    if (scale.valid())
        APPLY(*= scale.getUnchecked<fp>(i / group[0], j / group[1]));

#undef APPLY
}

template <typename fp> void _matrixRowSum(Matrix &dst, const Matrix &src);
template <typename fp> void _matrixColumnSum(Matrix &dst, const Matrix &src);

template <typename fp>
void _matrixRowSum(Matrix &dst, const Matrix &src)
{
    using fp_sum = supertype_t<fp>;
    using fp_acc = decltype(fp_sum(0) + 0);

    if (src.trans) {
        auto dstt = dst.tview();
        return _matrixColumnSum<fp>(dstt, src.tview());
    }

    auto v = src.nontransposedView();
    auto m = v.m, n = v.n, ld = v.ld;

    auto pd = dst.data.as<fp_sum>();

    for (integer i = 0; i < m; i++) {
        fp_acc isum = 0;
        for (integer j = 0; j < n; j++)
            isum = fp_acc(isum + v.nget<fp>(i, j));
        pd[i] = convert<fp_acc, fp_sum>(isum);
    }
}

template <typename fp>
void _matrixColumnSum(Matrix &dst, const Matrix &src)
{
    using fp_sum = supertype_t<fp>;
    using fp_acc = decltype(fp_sum(0) + 0);

    if (src.trans) {
        auto dstt = dst.tview();
        return _matrixRowSum<fp>(dstt, src.tview());
    }

    auto v = src.nontransposedView();
    auto m = v.m, n = v.n, ld = v.ld;

    auto pd = dst.data.as<fp_sum>();

    for (integer j = 0; j < n; j++) {
        fp_acc jsum = 0;
        for (integer i = 0; i < m; i++)
            jsum = fp_acc(jsum + v.nget<fp>(i, j));
        pd[j] = convert<fp_acc, fp_sum>(jsum);
    }
}

template <typename fp> void _matrixRowGroupSum(Matrix &dst, const Matrix &src, int group);
template <typename fp> void _matrixColumnGroupSum(Matrix &dst, const Matrix &src, int group);

template <typename fp>
void _matrixRowGroupSum(Matrix &dst, const Matrix &src, int group)
{
    using ST = supertype_t<fp>;

    if (src.trans) {
        auto dstt = dst.tview();
        return _matrixColumnGroupSum<fp>(dstt, src.tview(), group);
    }

    auto v = src.nontransposedView();
    auto m = v.m, n = v.n, ld = v.ld;

    for (integer i = 0; i < m; i++) {
        for (integer j = 0, jg = 0; j < n; jg++) {
            ST isum = 0;
            for (integer jj = 0; jj < group && j < n; j++, jj++)
                isum += v.nget<ST>(i, j);
            dst.getUnchecked<fp>(i, jg) = convert<ST, fp>(isum);
        }
    }
}

template <typename fp>
void _matrixColumnGroupSum(Matrix &dst, const Matrix &src, int group)
{
    using ST = supertype_t<fp>;

    if (src.trans) {
        auto dstt = dst.tview();
        return _matrixRowGroupSum<fp>(dstt, src.tview(), group);
    }

    auto v = src.nontransposedView();
    auto m = v.m, n = v.n, ld = v.ld;

    for (integer j = 0; j < n; j++) {
        for (integer i = 0, ig = 0; i < m; ig++) {
            ST jsum = 0;
            for (integer ii = 0; ii < group && i < m; i++, ii++)
                jsum += v.nget<ST>(i, j);
            dst.getUnchecked<fp>(ig, j) = convert<ST, fp>(jsum);
        }
    }
}

template <typename fp, integer crosspack, integer tileR, integer tileC>
void _matrixPackC(TypedData &dst, integer ldp, const Matrix &src, integer unroll, bool sum, MatrixName padFill)
{
    using fp_sum = sum_t<fp>;

    auto effTileR = tileR ? tileR : unroll;
    auto effTileC = tileC ? tileC : crosspack;

    bool trans = src.trans;
    auto m = src.m, n = src.n;
    auto lds = src.ld;
    auto ps = src.data.as<fp>();
    auto pd = dst.as<fp>();

    bool preservePadding = (padFill == MatrixName::UNINIT);
    fp padVal = fillVal<fp>(padFill);

    auto ss = sum ? std::max<integer>(effTileC, sizeof(fp_sum) / sizeof(fp)) : 0;
    auto jlim = ldp - ss;

    for (integer i0 = 0; i0 < m; i0 += unroll) {
        // Sums are stored at the end of each panel. ldp must leave space for them.
        fp_sum *psum = (fp_sum *) &pd[(i0 + unroll) * ldp];
        psum -= unroll;

        if (sum) for (integer ii = 0; ii < unroll; ii++)
            psum[ii] = 0;

        for (integer j0 = 0; j0 < jlim; j0 += effTileC)
        for (integer ii0 = 0; ii0 < unroll; ii0 += effTileR)
        for (integer j1 = 0; j1 < effTileC && (j0+j1) < jlim; j1 += crosspack)
        for (integer ii1 = 0; ii1 < effTileR; ii1++)
        for (integer jj = 0; jj < crosspack; jj++) {
            auto i = i0 + ii0 + ii1;
            auto j = j0 + j1 + jj;
            fp val;
            if (i >= m || j0 >= n) {
                if (preservePadding) break;
                val = padVal;
            } else if (j >= n)
                val = 0;                // Zero-pad remainder in case of crosspack/tiling
            else if (!trans)
                val = ps[i + lds*j];
            else
                val = ps[lds*i + j];
            pd[i0 * ldp + j0 * unroll + ii0 * effTileC + j1 * effTileR + ii1 * crosspack + jj] = val;
            if (sum && j < n) psum[ii0 + ii1] += fp_sum(val);
        }
    }
}

template <typename fp, integer targetCP, integer tileR, integer tileC>
void _matrixPackC(TypedData &dst, integer ldp, const Matrix &src, integer unroll, integer crosspackD, bool sum, MatrixName padFill)
{
    if (targetCP == crosspackD)
        _matrixPackC<fp, targetCP, tileR, tileC>(dst, ldp, src, unroll, sum, padFill);
    else if (targetCP > 1)
        _matrixPackC<fp, (targetCP >> 1), tileR, tileC>(dst, ldp, src, unroll, crosspackD, sum, padFill);
    else
        throw std::runtime_error("Unsupported crosspack.");
}

template <typename fp>
void _matrixPackC(TypedData &dst, integer ldp, const Matrix &src, integer unroll, integer crosspackD,
                  int tileRD, int tileCD, bool sum, MatrixName padFill)
{
#define DISPATCH(TR, TC, TCP)                                                           \
    if (tileRD == TR && tileCD == TC)                                                   \
        _matrixPackC<fp, TCP, TR, TC>(dst, ldp, src, unroll, crosspackD, sum, padFill); \
    else

    DISPATCH(0, 0, 64)
    DISPATCH(8, 8, 4)
    DISPATCH(8, 16, 4)
    DISPATCH(16, 8, 4)
    DISPATCH(8, 32, 4)
    DISPATCH(32, 8, 4)
    DISPATCH(16, 16, 4)
    DISPATCH(16, 32, 4)
    DISPATCH(8, 64, 8)
    DISPATCH(16, 64, 8)
    throw std::runtime_error("Unsupported tiling.");
#undef DISPATCH
}

template <typename fp>
void _matrixPackCTriangular(TypedData &dst, integer ldp, const Matrix &src,
                            integer unroll, bool lower, bool unit, MatrixName padFill)
{
    auto m = src.m, n = src.n, lds = src.ld;
    auto psrc = src.data.as<fp>();
    auto pdst = dst.as<fp>();

    fp padVal = fillVal<fp>(padFill);

    bool backward = !lower;
    integer n0pad = align_up(m, unroll);
    for (integer i = 0; i < m; i += unroll) {
        for (integer j = 0; j < ldp; j++) {
            for (integer ii = 0; ii < unroll; ii++) {
                auto i0 = i + ii, j0 = j;

                fp val;
                if ((i0 == j0) && unit)
                    val = 1;
                else if (i0 >= m || j0 >= n || ((i0 < j0) && lower) || ((i0 > j0) && !lower))
                    val = padVal;
                else
                    val = src.getUnchecked<fp>(i0, j0);

                if (i0 == j0)
                    val = fp(1) / val;
                else
                    val = -val;

                if (backward && (j0 >= n) && (j0 < n0pad))
                    val = 0;

                pdst[i * ldp + j * unroll + ii] = val;
            }
        }
    }
}

template <typename fp>
void _matrixUnpackC(Matrix &dst, const TypedData &src, integer ldp, integer unroll, integer crosspackD, int tileRD, int tileCD)
{
    bool trans = dst.trans;
    auto m = dst.m, n = dst.n;
    auto ldd = dst.ld;

    auto ps = src.as<fp>();
    auto pd = dst.data.as<fp>();

    if (crosspackD == 1 && tileRD == 0 && tileCD == 0) {
        // Simple path: no crosspack, tiling
        for (integer i = 0; i < m; i += unroll)
        for (integer j = 0; j < n; j++)
        for (integer ii = 0; ii < unroll && (i + ii < m); ii++) {
            fp val = ps[i * ldp + j * unroll + ii];
            if (!trans)
                pd[i + ii + ldd * j] = val;
            else
                pd[ldd * (i + ii) + j] = val;
        }
    } else {
        // Full path
        integer effTileR = tileRD ? tileRD : unroll;
        integer effTileC = tileCD ? tileCD : crosspackD;
        for (integer i0 = 0; i0 < m; i0 += unroll)
        for (integer j0 = 0; j0 < ldp; j0 += effTileC)
        for (integer ii0 = 0; ii0 < unroll; ii0 += effTileR)
        for (integer j1 = 0; j1 < effTileC && (j0+j1) < ldp; j1 += crosspackD)
        for (integer ii1 = 0; ii1 < effTileR; ii1++)
        for (integer jj = 0; jj < crosspackD; jj++) {
            auto i = i0 + ii0 + ii1;
            auto j = j0 + j1 + jj;
            if (i < m && j < n) {
                fp val = ps[i0 * ldp + j0 * unroll + ii0 * effTileC + j1 * effTileR + ii1 * crosspackD + jj];
                if (!trans)
                    pd[i + ldd * j] = val;
                else
                    pd[i * ldd + j] = val;
            }
        }
    }
}

template <typename fp>
void _scalarDowncast(superscalar &s)
{
    s = convert<fp, superscalar>(convert<superscalar, fp>(s));
}


/* Dispatch layer, converting runtime types to template types */
#define BTYPE_TEMPLATIFY(f) TYPE_TEMPLATIFY_FILTERED(f, batchify)
BTYPE_TEMPLATIFY(matrixCopy)
BTYPE_TEMPLATIFY(matrixFill)
BTYPE_TEMPLATIFY(matrixOffset)
BTYPE_TEMPLATIFY(matrixOffsetScalar)
BTYPE_TEMPLATIFY(matrixLinearCombination)
BTYPE_TEMPLATIFY(matrixDequantize)
BTYPE_TEMPLATIFY(matrixRowSum)
BTYPE_TEMPLATIFY(matrixRowGroupSum)
BTYPE_TEMPLATIFY(matrixAlignForReadCheck)
BTYPE_TEMPLATIFY(matrixReverseReadCheck)
BTYPE_TEMPLATIFY(matrixCopyTriangle)
BTYPE_TEMPLATIFY(matrixMakeTriangular)
BTYPE_TEMPLATIFY(matrixMakeSymmetric)
BTYPE_TEMPLATIFY(matrixMakeHermitian)
BTYPE_TEMPLATIFY(matrixMakeUnitDiagonal)
BTYPE_TEMPLATIFY(matrixInvertDiagonal)
BTYPE_TEMPLATIFY(matrixNegateDiagonal)
BTYPE_TEMPLATIFY(matrixFillImagDiagonal)
BTYPE_TEMPLATIFY(matrixConjugate)
TYPE_TEMPLATIFY(matrixPackC)
TYPE_TEMPLATIFY(matrixPackCTriangular)
TYPE_TEMPLATIFY(matrixUnpackC)
TYPE_TEMPLATIFY(scalarDowncast)


Matrix Matrix::view()
{
    return Matrix(*this, data.view());
}

Matrix Matrix::nontransposedView()
{
    auto result = view();
    if (result.trans) result.transpose();
    return result;
}

Matrix Matrix::paddedView()
{
    auto result = view();
    result.m = result.mpad;
    result.n = result.npad;
    return result;
}

Matrix Matrix::linearPaddedView()
{
    auto result = view();
    result.trans = false;
    result.m = result.mpad = result.ld = result.paddedElements();
    result.n = result.npad = 1;
    result.batch = 1;
    result.stride = 0;
    return result;
}

Matrix &Matrix::transpose()
{
    trans = !trans;
    std::swap(m, n);
    std::swap(mpad, npad);
    return *this;
}

Matrix Matrix::subview(integer i0, integer j0, integer b0,
                       integer nm, integer nn, integer nb) const
{
    auto v = view();
    v.data = v.data + offset(i0, j0, b0);
    v.m = v.mpad = nm;
    v.n = v.npad = nn;
    v.batch = nb;
    return v;
}

Matrix Matrix::selectBatch(integer b) const
{
    auto v = view();
    v.data = v.data + offset(0, 0, b);
    v.batch = 1;
    return v;
}

Matrix Matrix::copyAs(Type newType) const
{
    Matrix clone{newType, *this};
    copy(clone, *this);
    return clone;
}

void Matrix::fill(MatrixName name, const FillModifiers &mods)
{
    matrixFill(T, *this, name, mods);
}

void Matrix::zero()
{
    fill(MatrixName::ZERO);
}

void Matrix::zeroPaddedArea()
{
    fill(MatrixName::ZERO_PAD);
}

void Matrix::alignForReadCheck()                      { matrixAlignForReadCheck(T, *this);   }
void Matrix::reverseReadCheck()                       { matrixReverseReadCheck(T, *this);    }
void Matrix::conjugate()                              { matrixConjugate(T, *this);           }
void Matrix::makeUnitDiagonal()                       { matrixMakeUnitDiagonal(T, *this);    }
void Matrix::invertDiagonal()                         { matrixInvertDiagonal(T, *this);      }
void Matrix::negateDiagonal()                         { matrixNegateDiagonal(T, *this);      }
void Matrix::fillImagDiagonal(MatrixName fill)        { matrixFillImagDiagonal(T, *this, fill); }
void Matrix::makeTriangular(bool lower, bool fillNaN) { matrixMakeTriangular(T, *this, lower, fillNaN); }
void Matrix::makeSymmetric(bool lower)                { matrixMakeSymmetric(T, *this, lower);  }
void Matrix::makeHermitian(bool lower)                { matrixMakeHermitian(T, *this, lower);  }

void copy(Matrix &dst, const Matrix &src)
{
    if (dst.T == src.T || isSupertype(src.T))
        matrixCopy(dst.T, dst, src);
    else if (isSupertype(dst.T))
        matrixCopy(src.T, dst, src);
    else {
        auto ST = supertype(dst.T, src.T);
        Matrix tmp(ST, src);
        matrixCopy(src.T, tmp, src);
        matrixCopy(dst.T, dst, tmp);
    }
}

void copyAll(Matrix &dst, const Matrix &src)
{
    auto dstp = dst.paddedView();
    copy(dstp, src.paddedView());
}

void copyTriangle(Matrix &dst, const Matrix &src, bool lower)
{
    if (dst.T != src.T)
        throw std::runtime_error("Incompatible types");
    matrixCopyTriangle(dst.T, dst, src, lower);
}

void transpose(Matrix &dst, const Matrix &src)
{
    copy(dst, src.tview());
}

void transposeAll(Matrix &dst, const Matrix &src)
{
    copyAll(dst, src.tview());
}

void offset(Matrix &dst, superscalar off)
{
    matrixOffsetScalar(dst.T, dst, off);
}

void offset(Matrix &dst, const Matrix &off, char offType)
{
    if (dst.T != off.T) throw std::runtime_error("Incompatible types");
    matrixOffset(dst.T, dst, off, offType);
}

void scale(Matrix &dst, superscalar alpha)
{
    linearCombination(dst, dst, alpha, 0);
}

void linearCombination(Matrix &M1, const Matrix &M2, superscalar alpha, superscalar beta)
{
    if (M1.T != M2.T) throw std::runtime_error("Incompatible types");
    matrixLinearCombination(M1.T, M1, M2, alpha, beta);
}

void dequantize(Matrix &dst, std::array<int, 2> group, const Matrix &offset, const Matrix &scale)
{
    auto vo = offset.view(), vs = scale.view();

    if (offset.valid() && offset.T != dst.T)
        vo = offset.copyAs(dst.T);
    if (scale.valid() && scale.T != dst.T)
        vs = scale.copyAs(dst.T);
    matrixDequantize(dst.T, dst, group, vo, vs);
}

void rowSum(Matrix &dst, const Matrix &src)
{
    auto ST = supertype(src.T);
    if (ST == dst.T)
        matrixRowSum(src.T, dst, src);
    else {
        Matrix tmp(ST, dst);
        matrixRowSum(src.T, tmp, src);
        matrixCopy(dst.T, dst, tmp);
    }
}

void columnSum(Matrix &dst, const Matrix &src)
{
    auto dstt = dst.tview();
    rowSum(dstt, src.tview());
}

void rowGroupSum(Matrix &dst, const Matrix &src, const Matrix &offset, int group)
{
    auto ST = supertype(dst.T);
    Matrix tmp(ST, src);

    copy(tmp, src);
    if (offset.valid())
        dequantize(tmp, {1, group}, offset, Matrix{});
    matrixRowGroupSum(dst.T, dst, tmp, group);
}

void columnGroupSum(Matrix &dst, const Matrix &src, const Matrix &offset, int group)
{
    auto dstt = dst.tview();
    rowGroupSum(dstt, src.tview(), offset.tview(), group);
}

void packC(TypedData &dst, integer ldp, const Matrix &src, integer unroll, integer crosspackD,
           int tileRD, int tileCD, bool sum, MatrixName padFill)
{
    if (dst.T != src.T) throw std::runtime_error("Incompatible types");
    if (src.batch > 1)  throw std::runtime_error("Batching not supported");
    matrixPackC(dst.T, dst, ldp, src, unroll, crosspackD, tileRD, tileCD, sum, padFill);
}

void packCTriangular(TypedData &dst, integer ldp, const Matrix &src, integer unroll,
                     bool lower, bool unit, MatrixName padFill)
{
    if (dst.T != src.T) throw std::runtime_error("Incompatible types");
    if (src.batch > 1)  throw std::runtime_error("Batching not supported");
    matrixPackCTriangular(dst.T, dst, ldp, src, unroll, lower, unit, padFill);
}

void unpackC(Matrix &dst, const TypedData &src, integer ldp, integer unroll, integer crosspackD, int tileRD, int tileCD)
{
    if (dst.T != src.T) throw std::runtime_error("Incompatible types");
    if (dst.batch > 1)  throw std::runtime_error("Batching not supported");
    matrixUnpackC(dst.T, dst, src, ldp, unroll, crosspackD, tileRD, tileCD);
}

void packR(TypedData &dst, integer ldp, const Matrix &src, integer unroll, integer crosspackD,
           int tileRD, int tileCD, bool sum, MatrixName padFill)
{
    packC(dst, ldp, src.tview(), unroll, crosspackD, tileCD, tileRD, sum, transpose(padFill));
}

void packRTriangular(TypedData &dst, integer ldp, const Matrix &src, integer unroll,
                     bool lower, bool unit, MatrixName padFill)
{
    packCTriangular(dst, ldp, src.tview(), unroll, !lower, unit, transpose(padFill));
}

void unpackR(Matrix &dst, const TypedData &src, integer ldp, integer unroll, integer crosspackD, int tileRD, int tileCD)
{
    auto dstt = dst.tview();
    unpackC(dstt, src, ldp, unroll, crosspackD, tileCD, tileRD);
}

void downcast(superscalar &s, Type T)
{
    scalarDowncast(T, s);
}
