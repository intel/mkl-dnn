/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#include "mini_mkl.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <chrono>
#include <vector>

template <bool transa, bool transb, typename Ta, typename Tb, typename Tc, typename Ts>
void gemm(int m, int n, int k, Ts alpha, const Ta *A, int lda, const Tb *B, int ldb, Ts beta, Tc *C, int ldc)
{
    if (beta == Ts(0)) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                Tc acc = 0;
                for (int h = 0; h < k; h++)
                    acc += A[!transa ? (i + h * lda) : (i * lda + h)] * B[!transb ? (h + j * ldb) : (h * ldb + j)];
                C[i + j * ldc] = Tc(alpha * acc);
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < m; i++) {
                Tc acc = 0;
                for (int h = 0; h < k; h++)
                    acc += A[!transa ? (i + h * lda) : (i * lda + h)] * B[!transb ? (h + j * ldb) : (h * ldb + j)];
                C[i + j * ldc] = Tc(alpha * acc + beta * C[i + j * ldc]);
            }
        }
    }
}

template <typename Ta, typename Tb, typename Tc, typename Ts>
void gemm(bool transa, bool transb, int m, int n, int k, Ts alpha, const Ta *A, int lda, const Tb *B, int ldb, Ts beta, Tc *C, int ldc)
{
    if (!transa)
        if (!transb)
            gemm<false, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            gemm<false, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    else
        if (!transb)
            gemm<true, false>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        else
            gemm<true, true>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cblas_gemm_s8u8s32_inner(bool rowMajor, bool transa, bool transb, char offset, MKL_INT m, MKL_INT n, MKL_INT k, float alpha, int8_t *A, MKL_INT lda, int8_t oa, uint8_t *B, MKL_INT ldb, uint8_t ob, float beta, int32_t *C, MKL_INT ldc, int32_t *oc)
{
    if (rowMajor || offset != 'F' || oa || ob || oc[0])
        throw std::runtime_error("Unsupported case.");
    gemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

void gemm_s8u8s32_inner(char *transa, char *transb, char *offset, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, const MKL_INT8 *A, MKL_INT *lda, MKL_INT8 *oa, const MKL_UINT8 *B, MKL_INT *ldb, MKL_INT8 *ob, float *beta, MKL_INT32 *C, MKL_INT *ldc, MKL_INT32 *oc)
{
    if (*oa || *ob || *offset != 'F' || oc[0])
        throw std::runtime_error("Unsupported case.");
    gemm((*transa == 'T'), (*transb == 'T'), *m, *n, *k, *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

extern "C" {

void sgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *A, MKL_INT *lda, float *B, MKL_INT *ldb, float *beta, float *C, MKL_INT *ldc)
{
    gemm((*transa == 'T'), (*transb == 'T'), *m, *n, *k, *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

void dgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *A, MKL_INT *lda, double *B, MKL_INT *ldb, double *beta, double *C, MKL_INT *ldc)
{
    gemm((*transa == 'T'), (*transb == 'T'), *m, *n, *k, *alpha, A, *lda, B, *ldb, *beta, C, *ldc);
}

void cgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, MKL_Complex8 *alpha, MKL_Complex8 *A, MKL_INT *lda, MKL_Complex8 *B, MKL_INT *ldb, MKL_Complex8 *beta, MKL_Complex8 *C, MKL_INT *ldc)
{
    using CT = std::complex<float>;
    gemm((*transa == 'T'), (*transb == 'T'), *m, *n, *k, *((CT *) alpha), (CT *) A, *lda, (CT *) B, *ldb, *((CT *) beta), (CT *) C, *ldc);
}

void zgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, MKL_Complex16 *alpha, MKL_Complex16 *A, MKL_INT *lda, MKL_Complex16 *B, MKL_INT *ldb, MKL_Complex16 *beta, MKL_Complex16 *C, MKL_INT *ldc)
{
    using CT = std::complex<double>;
    gemm((*transa == 'T'), (*transb == 'T'), *m, *n, *k, *((CT *) alpha), (CT *) A, *lda, (CT *) B, *ldb, *((CT *) beta), (CT *) C, *ldc);
}

void scgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, MKL_Complex8 *alpha, float *A, MKL_INT *lda, MKL_Complex8 *B, MKL_INT *ldb, MKL_Complex8 *beta, MKL_Complex8 *C, MKL_INT *ldc)
{
    using CT = std::complex<float>;
    gemm((*transa == 'T'), (*transb == 'T'), *m, *n, *k, *((CT *) alpha), A, *lda, (CT *) B, *ldb, *((CT *) beta), (CT *) C, *ldc);
}

void dzgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, MKL_Complex16 *alpha, double *A, MKL_INT *lda, MKL_Complex16 *B, MKL_INT *ldb, MKL_Complex16 *beta, MKL_Complex16 *C, MKL_INT *ldc)
{
    using CT = std::complex<double>;
    gemm((*transa == 'T'), (*transb == 'T'), *m, *n, *k, *((CT *) alpha), A, *lda, (CT *) B, *ldb, *((CT *) beta), (CT *) C, *ldc);
}

void cblas_gemm_s8u8s32(bool rowMajor, bool transa, bool transb, char offset, MKL_INT m, MKL_INT n, MKL_INT k, float alpha, int8_t *A, MKL_INT lda, int8_t oa, uint8_t *B, MKL_INT ldb, uint8_t ob, float beta, int32_t *C, MKL_INT ldc, int32_t *oc)
{
    cblas_gemm_s8u8s32_inner(rowMajor, transa, transb, offset, m, n, k, alpha, A, lda, oa, B, ldb, ob, beta, C, ldc, oc);
}

void gemm_s8u8s32(char *transa, char *transb, char *offset, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, const MKL_INT8 *A, MKL_INT *lda, MKL_INT8 *oa, const MKL_UINT8 *B, MKL_INT *ldb, MKL_INT8 *ob, float *beta, MKL_INT32 *C, MKL_INT *ldc, MKL_INT32 *oc)
{
    gemm_s8u8s32_inner(transa, transb, offset, m, n, k, alpha, A, lda, oa, B, ldb, ob, beta, C, ldc, oc);
}

void strsm(char *side, char *uplo, char *trans, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *A, MKL_INT *lda, float *B, MKL_INT *ldb) {}
void dtrsm(char *side, char *uplo, char *trans, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *A, MKL_INT *lda, double *B, MKL_INT *ldb) {}
void ctrsm(char *side, char *uplo, char *trans, char *diag, MKL_INT *m, MKL_INT *n, MKL_Complex8 *alpha, MKL_Complex8 *A, MKL_INT *lda, MKL_Complex8 *B, MKL_INT *ldb) {}
void ztrsm(char *side, char *uplo, char *trans, char *diag, MKL_INT *m, MKL_INT *n, MKL_Complex16 *alpha, MKL_Complex16 *A, MKL_INT *lda, MKL_Complex16 *B, MKL_INT *ldb) {}


double dsecnd()
{
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int LAPACKE_dgels(int layout, char trans, int m, int n, int nrhs, double *A, int lda, double *B, int ldb)
{
    if (layout != LAPACK_COL_MAJOR) return -1; // Unsupported.

    // Handle transpose: form A^T explicitly, then solve as 'N'.
    std::vector<double> AT;
    if (trans == 'T' || trans == 't') {
        AT.resize(size_t(m) * n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                AT[j + size_t(i) * n] = A[i + size_t(j) * lda];
        A = AT.data();
        std::swap(m, n);
        lda = m;
    }

    if (m < n) return -1; // Unsupported.

    // Householder QR: overwrite A with R (upper tri) + reflectors (below).
    for (int j = 0; j < n; j++) {
        double norm2 = 0;
        for (int i = j; i < m; i++)
            norm2 += A[i + size_t(j) * lda] * A[i + size_t(j) * lda];
        double norm = std::sqrt(norm2);
        if (norm < 1e-300) continue;

        double sign = (A[j + size_t(j) * lda] >= 0) ? 1.0 : -1.0;
        double u1   = A[j + size_t(j) * lda] + sign * norm;

        for (int i = j + 1; i < m; i++)
            A[i + size_t(j) * lda] /= u1;

        double tau = sign * u1 / norm;

        A[j + size_t(j) * lda] = -sign * norm;

        // Apply reflector to remaining columns of A.
        for (int k = j + 1; k < n; k++) {
            double dot = A[j + size_t(k) * lda];
            for (int i = j + 1; i < m; i++)
                dot += A[i + size_t(j) * lda] * A[i + size_t(k) * lda];
            dot *= tau;
            A[j + size_t(k) * lda] -= dot;
            for (int i = j + 1; i < m; i++)
                A[i + size_t(k) * lda] -= A[i + size_t(j) * lda] * dot;
        }

        // Apply reflector to B.
        for (int k = 0; k < nrhs; k++) {
            double dot = B[j + size_t(k) * ldb];
            for (int i = j + 1; i < m; i++)
                dot += A[i + size_t(j) * lda] * B[i + size_t(k) * ldb];
            dot *= tau;
            B[j + size_t(k) * ldb] -= dot;
            for (int i = j + 1; i < m; i++)
                B[i + size_t(k) * ldb] -= A[i + size_t(j) * lda] * dot;
        }
    }

    // Back-substitution: R * X = Q^T * B.
    for (int k = 0; k < nrhs; k++) {
        for (int j = n - 1; j >= 0; j--) {
            double diag = A[j + size_t(j) * lda];
            if (diag == 0) return j + 1;
            B[j + size_t(k) * ldb] /= diag;
            for (int i = 0; i < j; i++)
                B[i + size_t(k) * ldb] -= A[i + size_t(j) * lda] * B[j + size_t(k) * ldb];
        }
    }

    return 0;
}

}
