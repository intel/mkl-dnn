/*******************************************************************************
* Copyright 2020 Intel Corporation
* Copyright 2020, 2026 Arm Ltd. and affiliates
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

#ifndef CPU_PLATFORM_HPP
#define CPU_PLATFORM_HPP

#include "oneapi/dnnl/dnnl_config.h"

#include "common/c_types_map.hpp"
#include "common/impl_registration.hpp"
#include "common/z_magic.hpp"

// Possible architectures:
// - DNNL_X64
// - DNNL_AARCH64
// - DNNL_PPC64
// - DNNL_S390X
// - DNNL_RV64
// - DNNL_ARCH_GENERIC
// Target architecture macro is set to 1, others to 0. All macros are defined.

#if defined(DNNL_X64) + defined(DNNL_AARCH64) + defined(DNNL_PPC64) \
                + defined(DNNL_S390X) + defined(DNNL_RV64) \
                + defined(DNNL_ARCH_GENERIC) \
        == 0
#if defined(__x86_64__) || defined(_M_X64)
#define DNNL_X64 1
#elif defined(__aarch64__)
#define DNNL_AARCH64 1
#elif defined(__powerpc64__) || defined(__PPC64__) || defined(_ARCH_PPC64)
#define DNNL_PPC64 1
#elif defined(__s390x__)
#define DNNL_S390X 1
#elif defined(__riscv)
#define DNNL_RV64 1
#else
#define DNNL_ARCH_GENERIC 1
#endif
#endif // defined(DNNL_X64) + ... == 0

#if defined(DNNL_X64) + defined(DNNL_AARCH64) + defined(DNNL_PPC64) \
                + defined(DNNL_S390X) + defined(DNNL_RV64) \
                + defined(DNNL_ARCH_GENERIC) \
        != 1
#error One and only one architecture should be defined at a time
#endif

#if !defined(DNNL_X64)
#define DNNL_X64 0
#endif
#if !defined(DNNL_AARCH64)
#define DNNL_AARCH64 0
#endif
#if !defined(DNNL_PPC64)
#define DNNL_PPC64 0
#endif
#if !defined(DNNL_S390X)
#define DNNL_S390X 0
#endif
#if !defined(DNNL_RV64)
#define DNNL_RV64 0
#endif
#if !defined(DNNL_ARCH_GENERIC)
#define DNNL_ARCH_GENERIC 0
#endif

// Helper macros: expand the parameters only on the corresponding architecture.
// Equivalent to: #if DNNL_$ARCH ... #endif
#define DNNL_X64_ONLY(...) Z_CONDITIONAL_DO(DNNL_X64, __VA_ARGS__)
#define DNNL_PPC64_ONLY(...) Z_CONDITIONAL_DO(DNNL_PPC64, __VA_ARGS__)
#define DNNL_S390X_ONLY(...) Z_CONDITIONAL_DO(DNNL_S390X_ONLY, __VA_ARGS__)
#define DNNL_AARCH64_ONLY(...) Z_CONDITIONAL_DO(DNNL_AARCH64, __VA_ARGS__)
#define DNNL_RV64_ONLY(...) Z_CONDITIONAL_DO(DNNL_RV64, __VA_ARGS__)

// Negation of the helper macros above
#define DNNL_NON_X64_ONLY(...) Z_CONDITIONAL_DO(Z_NOT(DNNL_X64), __VA_ARGS__)

// Using Arm Compute Library kernels is optional for AArch64 builds
// and can be enabled with the DNNL_AARCH64_USE_ACL CMake option
#if defined(DNNL_AARCH64) && defined(DNNL_AARCH64_USE_ACL)
#define DNNL_AARCH64_ACL_ONLY(...) __VA_ARGS__
#else
#define DNNL_AARCH64_ACL_ONLY(...)
#endif

// Building KleidiAI-based implementations can be enabled with
// the DNNL_AARCH64_USE_KAI CMake option.
#ifndef DNNL_AARCH64_USE_KAI
#define DNNL_AARCH64_USE_KAI 0
#endif
#if DNNL_AARCH64 && DNNL_AARCH64_USE_KAI
#define DNNL_AARCH64_KAI_ONLY(...) __VA_ARGS__
#else
#define DNNL_AARCH64_KAI_ONLY(...)
#endif

// Using Zen kernels is optional for x64 builds
// and can be enabled with the DNNL_X64_USE_ZEN CMake option
#if DNNL_X64 && DNNL_X64_USE_ZEN
#define DNNL_X64_ZEN(...) __VA_ARGS__
#else
#define DNNL_X64_ZEN(...)
#endif

// Primitive ISA section for configuring knobs.
// Note: MSVC preprocessor by some reason "eats" symbols it's not supposed to
// if __VA_ARGS__ is passed as empty. Then things happen like this for non-x64:
// impl0, AMX(X64_impl1), impl2, ... -> impl0   impl2, ...
// resulting in compilation error. Such problem happens for lists interleaving
// X64 impls and non-X64 for non-X64 build.
#if DNNL_X64
// Note: unlike workload or primitive set, these macros will work with impl
// items directly, thus, just make an item disappear, no empty lists.
#define __BUILD_AMX BUILD_PRIMITIVE_CPU_ISA_ALL || BUILD_AMX
#define __BUILD_AVX512 __BUILD_AMX || BUILD_AVX512
#define __BUILD_AVX2 __BUILD_AVX512 || BUILD_AVX2
#else
#define __BUILD_AMX 0
#define __BUILD_AVX512 0
#define __BUILD_AVX2 0
#endif

#if __BUILD_AMX
#define REG_AMX_ISA(...) __VA_ARGS__
#else
#define REG_AMX_ISA(...)
#endif

#if __BUILD_AVX512
#define REG_AVX512_ISA(...) __VA_ARGS__
#else
#define REG_AVX512_ISA(...)
#endif

#if __BUILD_AVX2
#define REG_AVX2_ISA(...) __VA_ARGS__
#else
#define REG_AVX2_ISA(...)
#endif

namespace dnnl {
namespace impl {
namespace cpu {
namespace platform {

const char *get_isa_info();
dnnl_cpu_isa_t get_effective_cpu_isa();
status_t set_max_cpu_isa(dnnl_cpu_isa_t isa);
status_t set_cpu_isa_hints(dnnl_cpu_isa_hints_t isa_hints);
dnnl_cpu_isa_hints_t get_cpu_isa_hints();

bool DNNL_API prefer_ymm_requested();
// This call is limited to performing checks on plain C-code implementations
// (e.g. 'ref' and 'simple_primitive') and should avoid any x64 JIT
// implementations since these require specific code-path updates.
bool DNNL_API has_data_type_support(data_type_t data_type);
bool DNNL_API has_training_support(data_type_t data_type);
// Status-returning counterparts of the checks above for implementation
// dispatching. Return `unsupported_data_type` when the data type requires an
// ISA the current CPU lacks, and `unimplemented` when no CPU of this
// architecture supports it.
status_t DNNL_API data_type_support_status(data_type_t data_type);
status_t DNNL_API training_support_status(data_type_t data_type);
bool DNNL_API has_optimized_gemm();
float DNNL_API s8s8_weights_scale_factor();

unsigned DNNL_API get_per_core_cache_size(int level);
uint32_t get_num_ways_in_cache(int level);
uint32_t get_num_sets_in_cache(int level);
unsigned DNNL_API get_num_cores();
#if DNNL_CPU_THREADING_RUNTIME == DNNL_RUNTIME_THREADPOOL
unsigned DNNL_API get_max_threads_to_use();
#endif

constexpr int get_cache_line_size() {
    return 64;
}

int get_vector_register_size();

// Helper to avoid #ifdefs for DNNL_PPC64
static constexpr bool is_ppc64() {
#if DNNL_PPC64
    return true;
#else
    return false;
#endif
}

size_t get_timestamp();

} // namespace platform

// XXX: find a better place for these values?
enum {
    PAGE_4K = 4096,
    PAGE_2M = 2097152,
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif
