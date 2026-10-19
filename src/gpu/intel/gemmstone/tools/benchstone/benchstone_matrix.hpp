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

#ifndef GEMMSTONE_TOOLS_BENCHSTONE_BENCHSTONE_MATRIX_HPP
#define GEMMSTONE_TOOLS_BENCHSTONE_BENCHSTONE_MATRIX_HPP

#include "benchstone_common.hpp"

#include <cstdlib>
#include <cstring>

enum class MatrixName {
    GEMM_A, GEMM_B, GEMM_C, GEMM_CO,
    AO, BO, AO_SMALL, BO_SMALL, A_SCALE, B_SCALE,
    TRSMLL_U, TRSMLU_U, TRSMRL_U, TRSMRU_U,
    TRSMLL_V, TRSMLU_V, TRSMRL_V, TRSMRU_V,
    TRSMLLU_U, TRSMLUU_U, TRSMRLU_U, TRSMRUU_U,
    COPY_S, COPYU_S, COPYL_S, COPYTRSM_S, COPY_D,
    ZERO, ONE, EYE, ONE_HALF, NANS,
    ROW_RAMP, ROW_STRIPE, COL_RAMP, COL_STRIPE, DIAG_RAMP, RANDOM,
    SERIES_ZERO, SERIES_ONE, SERIES_RANDOM,
    UNINIT, ZERO_PAD  /* for padding only */
};

MatrixName transpose(MatrixName name);

struct FillModifiers {
    bool zeros = false, ones = false;
    bool beta0 = false;
    int stripeIndex = 0;
};


class Matrix
{
public:
    using Type = gemmstone::Type;

    Matrix() : T(Type::invalid), trans(false), m(0), n(0), ld(0), mpad(0), npad(0), batch(1), stride(0) {}
    Matrix(Type T_, bool trans_, integer m_, integer n_, integer ld_, bool alloc = true)
            : T(T_), trans(trans_), m(m_), n(n_), ld(ld_), mpad(m_), npad(n_), batch(1), stride(0) { if (alloc) allocate(); }
    Matrix(Type T_, bool trans_, integer m_, integer n_, integer ld_, integer mpad_, integer npad_, bool alloc = true)
            : T(T_), trans(trans_), m(m_), n(n_), ld(ld_), mpad(mpad_), npad(npad_), batch(1), stride(0) { padCheck(); if (alloc) allocate(); }
    Matrix(Type T_, bool trans_, integer m_, integer n_, integer ld_, integer mpad_, integer npad_, integer batch_, integer stride_, bool alloc = true)
            : T(T_), trans(trans_), m(m_), n(n_), ld(ld_), mpad(mpad_), npad(npad_), batch(batch_), stride(stride_) { padCheck(); if (alloc) allocate(); }
    Matrix(Type T_, const Matrix &config_, bool alloc = true)
            : Matrix(T_, config_.trans, config_.m, config_.n, config_.ld, config_.mpad, config_.npad, config_.batch, config_.stride, alloc) {}

    Matrix(const Matrix &other, TypedData newData)
            : T(other.T), trans(other.trans), m(other.m), n(other.n), ld(other.ld),
              mpad(other.mpad), npad(other.npad), batch(other.batch), stride(other.stride),
              data(std::move(newData)) {}

    /* Factory method for constructing single-element matrices */
    template <typename U>
    static inline Matrix scalar(Type T_, U value);

    bool valid() const             { return T != Type::invalid; }
    bool square() const            { return m == n; }

    static integer elements(bool trans, integer ld, integer m, integer n) {
        return ld * (trans ? m : n);
    }

    integer elements() const       { return batch * elements(trans, ld, m, n); }
    integer paddedElements() const { return std::max(elements(trans, ld, mpad, npad), batch * stride); }

    integer offset(integer i, integer j, integer b = 0) const {
        return (trans ? (j + i*ld) : (i + j*ld)) + b*stride;
    }
    template <typename U>
    U &get(integer i, integer j, integer b = 0) {
        return data.as<U>()[offset(i, j, b)];
    }

    template <typename U>
    U get(integer i, integer j, integer b = 0) const {
        return data.as<U>()[offset(i, j, b)];
    }

    template <typename U>
    U &getUnchecked(integer i, integer j, integer b = 0) {
        return data.asUnchecked<U>()[offset(i, j, b)];
    }

    template <typename U>
    U getUnchecked(integer i, integer j, integer b = 0) const {
        return data.asUnchecked<U>()[offset(i, j, b)];
    }

    template <typename U>
    U &nget(integer i, integer j, integer b = 0) {
        return data.asUnchecked<U>()[i + j*ld + b*stride];
    }

    Matrix view();
    Matrix tview()                  { return view().transpose(); }
    Matrix nontransposedView();
    Matrix paddedView();
    Matrix linearPaddedView();

    const Matrix view() const              { return const_cast<Matrix*>(this)->view(); }
    const Matrix tview() const             { return const_cast<Matrix*>(this)->tview(); }
    const Matrix paddedView() const        { return const_cast<Matrix*>(this)->paddedView(); }
    const Matrix nontransposedView() const { return const_cast<Matrix*>(this)->nontransposedView(); }

    Matrix subview(integer i0, integer j0, integer b0, integer nm, integer nn, integer nb) const;
    Matrix subview(integer i0, integer j0, integer nm, integer nn) const {
        return subview(i0, j0, 0, nm, nn, batch);
    }
    Matrix selectBatch(integer b = 0) const;

    Matrix &transpose();

    Matrix copyAs(Type newType) const;

    void fill(MatrixName name, const FillModifiers &mods = FillModifiers{});
    void zero();
    void zeroPaddedArea();

    void alignForReadCheck();
    void reverseReadCheck();

    void conjugate();

    void makeTriangular(bool lower, bool fillNaN = false);
    void makeSymmetric(bool lower);
    void makeHermitian(bool lower);

    void makeUnitDiagonal();
    void invertDiagonal();
    void negateDiagonal();
    void fillImagDiagonal(MatrixName fill);

    Type T;
    bool trans;
    integer m, n, ld;
    integer mpad, npad;
    integer batch, stride;

    TypedData data;

private:
    void allocate() {
        data = TypedData(T, paddedElements());
    }

    void padCheck() {
        if (m > mpad || n > npad)
            throw std::runtime_error("Invalid padding dimensions");
    }
};

void copy(Matrix &dst, const Matrix &src);
void copyAll(Matrix &dst, const Matrix &src);
void copyTriangle(Matrix &dst, const Matrix &src, bool lower);
void transpose(Matrix &dst, const Matrix &src);
void transposeAll(Matrix &dst, const Matrix &src);
void offset(Matrix &dst, superscalar off);
void offset(Matrix &dst, const Matrix &off, char offType);
void scale(Matrix &dst, superscalar alpha);
void linearCombination(Matrix &M1, const Matrix &M2, superscalar alpha, superscalar beta);
void dequantize(Matrix &dst, std::array<int, 2> group, const Matrix &offset, const Matrix &scale);

void rowSum(Matrix &dst, const Matrix &src);
void columnSum(Matrix &dst, const Matrix &src);
void rowGroupSum(Matrix &dst, const Matrix &src, const Matrix &offset, int group);
void columnGroupSum(Matrix &dst, const Matrix &src, const Matrix &offset, int group);

void packC(TypedData &dst, integer ldp, const Matrix &src, integer unroll, integer crosspackD = 1,
           int tileRD = 0, int tileCD = 0, bool sum = false, MatrixName padFill = MatrixName::ZERO);
void packCTriangular(TypedData &dst, integer ldp, const Matrix &src, integer unroll,
                     bool lower, bool unit, MatrixName padFill = MatrixName::ZERO);
void unpackC(Matrix &dst, const TypedData &src, integer ldp, integer unroll, integer crosspackD = 1,
             int tileRD = 0, int tileCD = 0);

void packR(TypedData &dst, integer ldp, const Matrix &src, integer unroll, integer crosspackD = 1,
           int tileRD = 0, int tileCD = 0, bool sum = false, MatrixName padFill = MatrixName::ZERO);
void packRTriangular(TypedData &dst, integer ldp, const Matrix &src, integer unroll,
                     bool lower, bool unit, MatrixName padFill = MatrixName::ZERO);
void unpackR(Matrix &dst, const TypedData &src, integer ldp, integer unroll, integer crosspackD = 1,
             int tileRD = 0, int tileCD = 0);

void downcast(superscalar &s, gemmstone::Type T);


template <typename U>
Matrix Matrix::scalar(gemmstone::Type T_, U value)
{
    Matrix src{typeEnum<U>, false, 1, 1, 1};
    src.nget<U>(0, 0) = value;
    return src.copyAs(T_);
}

inline void parseNameOverride(MatrixName &name, const char *env)
{
    auto override = getenv(env);
    if (!override)
        return;

    if (!strcmp(override, "ZERO")) name = MatrixName::ZERO;
    else if (!strcmp(override, "ONE")) name = MatrixName::ONE;
    else if (!strcmp(override, "EYE")) name = MatrixName::EYE;
    else if (!strcmp(override, "ONE_HALF")) name = MatrixName::ONE_HALF;
    else if (!strcmp(override, "NAN")) name = MatrixName::NANS;
    else if (!strcmp(override, "ROW_RAMP")) name = MatrixName::ROW_RAMP;
    else if (!strcmp(override, "ROW_STRIPE")) name = MatrixName::ROW_STRIPE;
    else if (!strcmp(override, "COL_RAMP")) name = MatrixName::COL_RAMP;
    else if (!strcmp(override, "COL_STRIPE")) name = MatrixName::COL_STRIPE;
    else if (!strcmp(override, "DIAG_RAMP")) name = MatrixName::DIAG_RAMP;
    else if (!strcmp(override, "RANDOM")) name = MatrixName::RANDOM;
    else
        yprintf("Unknown matrix pattern specified in environment variable %s\n", env);

}



/* Batching automatization */
template <typename T> static inline auto bselect(const T &x, integer) { return x; }
template <> auto bselect(const Matrix &x, integer i) { return x.selectBatch(i % x.batch); }

template <typename T> static inline size_t bsize(const T &x)      { return 1; }
template <> size_t bsize(const Matrix &x)           { return x.batch; }

template <typename F, typename... Args>
void batchify(F f, Args&&... args)
{
    auto batch = std::max({bsize(args)...});
    for (integer b = 0; b < batch; b++) {
        auto margs = std::make_tuple(bselect(args, b)...);
        std::apply(f, margs);
    }
}

#endif
