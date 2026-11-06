#ifndef CPU_AARCH64_GEMM_BF16_NEON_GEMM_BF16BF16F32_HPP
#define CPU_AARCH64_GEMM_BF16_NEON_GEMM_BF16BF16F32_HPP

#include "oneapi/dnnl/dnnl_types.h"

#include "common/bfloat16.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

dnnl_status_t neon_gemm_bf16bf16f32(const char *transa, const char *transb,
        const dim_t *M, const dim_t *N, const dim_t *K, const float *alpha,
        const bfloat16_t *A, const dim_t *lda, const bfloat16_t *B,
        const dim_t *ldb, const float *beta, float *C, const dim_t *ldc);

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
