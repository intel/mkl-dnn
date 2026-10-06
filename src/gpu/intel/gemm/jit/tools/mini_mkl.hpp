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

#ifndef GEMMSTONE_TOOLS_MINI_MKL_HPP
#define GEMMSTONE_TOOLS_MINI_MKL_HPP

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CblasColMajor 0

typedef bool CBLAS_TRANSPOSE;
#define CblasNoTrans false
#define CblasTrans true
#define CblasFixOffset 'F'

typedef int     MKL_INT;
typedef int32_t MKL_INT32;
typedef int8_t  MKL_INT8;
typedef uint8_t MKL_UINT8;

#ifndef MKL_Complex8
typedef struct {
    float re, im;
} MKL_Complex8;
#endif

#ifndef MKL_Complex16
typedef struct {
    double re, im;
} MKL_Complex16;
#endif

extern void sgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, float *A, MKL_INT *lda, float *B, MKL_INT *ldb, float *beta, float *C, MKL_INT *ldc);
extern void dgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, double *alpha, double *A, MKL_INT *lda, double *B, MKL_INT *ldb, double *beta, double *C, MKL_INT *ldc);
extern void cgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, MKL_Complex8 *alpha, MKL_Complex8 *A, MKL_INT *lda, MKL_Complex8 *B, MKL_INT *ldb, MKL_Complex8 *beta, MKL_Complex8 *C, MKL_INT *ldc);
extern void zgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, MKL_Complex16 *alpha, MKL_Complex16 *A, MKL_INT *lda, MKL_Complex16 *B, MKL_INT *ldb, MKL_Complex16 *beta, MKL_Complex16 *C, MKL_INT *ldc);
extern void scgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, MKL_Complex8 *alpha, float *A, MKL_INT *lda, MKL_Complex8 *B, MKL_INT *ldb, MKL_Complex8 *beta, MKL_Complex8 *C, MKL_INT *ldc);
extern void dzgemm(char *transa, char *transb, MKL_INT *m, MKL_INT *n, MKL_INT *k, MKL_Complex16 *alpha, double *A, MKL_INT *lda, MKL_Complex16 *B, MKL_INT *ldb, MKL_Complex16 *beta, MKL_Complex16 *C, MKL_INT *ldc);

extern void strsm(char *side, char *uplo, char *trans, char *diag, MKL_INT *m, MKL_INT *n, float *alpha, float *A, MKL_INT *lda, float *B, MKL_INT *ldb);
extern void dtrsm(char *side, char *uplo, char *trans, char *diag, MKL_INT *m, MKL_INT *n, double *alpha, double *A, MKL_INT *lda, double *B, MKL_INT *ldb);
extern void ctrsm(char *side, char *uplo, char *trans, char *diag, MKL_INT *m, MKL_INT *n, MKL_Complex8 *alpha, MKL_Complex8 *A, MKL_INT *lda, MKL_Complex8 *B, MKL_INT *ldb);
extern void ztrsm(char *side, char *uplo, char *trans, char *diag, MKL_INT *m, MKL_INT *n, MKL_Complex16 *alpha, MKL_Complex16 *A, MKL_INT *lda, MKL_Complex16 *B, MKL_INT *ldb);

extern void cblas_gemm_s8u8s32(bool rowMajor, bool transa, bool transb, char offset, MKL_INT m, MKL_INT n, MKL_INT k, float alpha, int8_t *A, MKL_INT lda, int8_t oa, uint8_t *B, MKL_INT ldb, uint8_t ob, float beta, int32_t *C, MKL_INT ldc, int32_t *oc);
extern void gemm_s8u8s32(char *transa, char *transb, char *offset, MKL_INT *m, MKL_INT *n, MKL_INT *k, float *alpha, const MKL_INT8 *A, MKL_INT *lda, MKL_INT8 *oa, const MKL_UINT8 *B, MKL_INT *ldb, MKL_INT8 *ob, float *beta, MKL_INT32 *C, MKL_INT *ldc, MKL_INT32 *oc);

extern double dsecnd();

#ifndef LAPACK_COL_MAJOR
#define LAPACK_COL_MAJOR 102
#endif

extern int LAPACKE_dgels(int matrix_layout, char trans, int m, int n, int nrhs,
                         double *a, int lda, double *b, int ldb);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif
