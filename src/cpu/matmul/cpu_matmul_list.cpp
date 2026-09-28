/*******************************************************************************
* Copyright 2019 Intel Corporation
* Copyright 2024-2026 FUJITSU LIMITED
* Copyright 2021-2026 Arm Ltd. and affiliates
* Copyright 2026 Advanced Micro Devices, Inc.
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

#include "cpu/cpu_engine.hpp"

#include "cpu/matmul/gemm_bf16_matmul.hpp"
#include "cpu/matmul/gemm_f32_matmul.hpp"
#include "cpu/matmul/gemm_x8s8s32x_matmul.hpp"
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
#include "cpu/matmul/ref_grouped_gemm.hpp"
#endif
#include "cpu/matmul/ref_matmul.hpp"
#include "cpu/matmul/ref_matmul_int8.hpp"
#include "cpu/matmul/ref_sparse_matmul.hpp"

#if DNNL_X64
#include "cpu/x64/matmul/brgemm_matmul.hpp"
#include "cpu/x64/matmul/jit_uni_sparse_matmul.hpp"
#if DNNL_X64_USE_ZEN
#include "cpu/x64/zen64/matmul/zen_lowp_matmul.hpp"
#include "cpu/x64/zen64/matmul/zen_matmul.hpp"
#if DNNL_EXPERIMENTAL_GROUPED_MEMORY
#include "cpu/x64/zen64/matmul/zen_grouped_matmul.hpp"
#endif
#endif
using namespace dnnl::impl::cpu::x64::matmul;
using namespace dnnl::impl::cpu::x64;
#if DNNL_X64_USE_ZEN
using namespace dnnl::impl::cpu::x64::zen::matmul;
#endif
#elif DNNL_AARCH64
#include "cpu/aarch64/matmul/brgemm_matmul.hpp"
#include "cpu/aarch64/matmul/jit_bf16_matmul.hpp"
#include "cpu/aarch64/matmul/jit_int8_matmul.hpp"
#if DNNL_AARCH64_USE_KAI
#include "cpu/aarch64/matmul/kai_matmul.hpp"
#endif
#ifdef DNNL_AARCH64_USE_ACL
#include "cpu/aarch64/matmul/acl_lowp_matmul.hpp"
#include "cpu/aarch64/matmul/acl_lowp_matmul_sq.hpp"
#include "cpu/aarch64/matmul/acl_matmul.hpp"
#endif
using namespace dnnl::impl::cpu::aarch64::matmul;
using namespace dnnl::impl::cpu::aarch64;
#elif DNNL_RV64
#include "cpu/rv64/rvv_brgemm_matmul.hpp"
#include "cpu/rv64/rvv_matmul.hpp"
using namespace dnnl::impl::cpu::rv64::matmul;
using namespace dnnl::impl::cpu::rv64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;
using namespace dnnl::impl::cpu::matmul;

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_MATMUL_P({
        CPU_INSTANCE_AARCH64(brgemm_matmul_t<sme>)
        CPU_INSTANCE_AARCH64(brgemm_matmul_t<sve_512>)
        CPU_INSTANCE_AARCH64(jit_int8_matmul_t<sve>)
        CPU_INSTANCE_AARCH64_KAI(kai_matmul_t)
        CPU_INSTANCE_AARCH64_ACL(acl_lowp_matmul_sq_t)
        CPU_INSTANCE_AARCH64_ACL(acl_lowp_matmul_t)
        CPU_INSTANCE_AARCH64_ACL(acl_matmul_t)
        CPU_INSTANCE_AARCH64(jit_bf16_matmul_t)
        CPU_INSTANCE_AARCH64(brgemm_matmul_t<sve_256>)
        CPU_INSTANCE_AARCH64(brgemm_matmul_t<sve_128>)
        CPU_INSTANCE_X64_ZEN(zen_lowp_matmul_t)
        CPU_INSTANCE_X64_ZEN(zen_matmul_t)
        CPU_INSTANCE_AMX(brgemm_matmul_t<avx10_2_ace>)
        CPU_INSTANCE_AMX(brgemm_matmul_t<avx10_2_amx_2>)
        CPU_INSTANCE_AMX(brgemm_matmul_t<avx512_core_amx_fp16>)
        CPU_INSTANCE_AMX(brgemm_matmul_t<avx512_core_amx>)
        CPU_INSTANCE_AVX512(brgemm_matmul_t<avx10_2>)
        CPU_INSTANCE_AVX512(brgemm_matmul_t<avx512_core_fp16>)
        CPU_INSTANCE_AVX512(brgemm_matmul_t<avx512_core_bf16>)
        CPU_INSTANCE_AVX512(brgemm_matmul_t<avx512_core_vnni>)
        CPU_INSTANCE_AVX512(brgemm_matmul_t<avx512_core>)
        CPU_INSTANCE_AVX2(brgemm_matmul_t<avx2_vnni_2>)
        CPU_INSTANCE_AVX2(brgemm_matmul_t<avx2_vnni>)
        CPU_INSTANCE_AVX2(brgemm_matmul_t<avx2>)
        CPU_INSTANCE_RV64(rvv_brgemm_matmul_t)
        CPU_INSTANCE_RV64(rvv_matmul_t)
        CPU_INSTANCE(gemm_f32_matmul_t)
        CPU_INSTANCE(gemm_bf16_matmul_t<f32>)
        CPU_INSTANCE(gemm_bf16_matmul_t<bf16>)
        CPU_INSTANCE(gemm_x8s8s32x_matmul_t)
        CPU_INSTANCE(ref_matmul_t)
        CPU_INSTANCE(ref_matmul_int8_t)
        CPU_INSTANCE_X64(jit_uni_sparse_matmul_t)
        CPU_INSTANCE(ref_sparse_matmul_t)
        CPU_INSTANCE_X64_ZEN_GROUPED(zen_grouped_matmul_t)
        CPU_INSTANCE_GROUPED(ref_grouped_t)
        /* eol */
        nullptr,
});
// clang-format on
} // namespace

const impl_list_item_t *get_matmul_impl_list(const matmul_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
