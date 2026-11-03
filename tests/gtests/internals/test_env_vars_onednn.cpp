/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifdef _WIN32
#include <windows.h>
#endif

#include "stdlib.h"

#include "dnnl_test_common.hpp"
#include "gtest/gtest.h"

#include "oneapi/dnnl/dnnl.hpp"

#include "tests/test_isa_common.hpp"

// Note: use one non-default value to validate functionality.

namespace {

void custom_setenv(const char *name, const char *value, int overwrite) {
#ifdef _WIN32
    auto status = SetEnvironmentVariable(name, value);
    EXPECT_NE(status, 0);
#else
    auto status = ::setenv(name, value, overwrite);
    EXPECT_EQ(status, 0);
#endif
}

} // namespace

namespace dnnl {

#if DNNL_X64
TEST(onednn_max_cpu_isa_env_var_test, TestEnvVars) {
    const bool has_cpu = DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE;

    custom_setenv("ONEDNN_MAX_CPU_ISA", "AVX2", 1);
    auto got = get_effective_cpu_isa();
    (void)got;

#if defined(DNNL_ENABLE_MAX_CPU_ISA)
#if DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    const auto expected = mayiuse(impl::cpu::x64::avx2) ? cpu_isa::avx2
                                                        : cpu_isa::isa_default;
#else
    const auto expected = cpu_isa::isa_default;
#endif
    EXPECT_EQ(got, expected);
#elif DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (mayiuse(impl::cpu::x64::avx512_core)) { EXPECT_NE(got, cpu_isa::avx2); }
#endif

    if (has_cpu) {
        auto st = set_max_cpu_isa(cpu_isa::avx2);
        EXPECT_EQ(st, status::invalid_arguments);
    }

    custom_setenv("ONEDNN_MAX_CPU_ISA", "AVX512_CORE", 1);
    got = get_effective_cpu_isa();
#if defined(DNNL_ENABLE_MAX_CPU_ISA)
    EXPECT_EQ(got, expected);
#elif DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE
    if (mayiuse(impl::cpu::x64::avx512_core_vnni)) {
        EXPECT_NE(got, cpu_isa::avx512_core);
    }
#endif
}
#endif // DNNL_X64

#if DNNL_X64
TEST(onednn_cpu_isa_hints_var_test, TestEnvVars) {
    const bool has_cpu = DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE;
    (void)has_cpu;

    custom_setenv("ONEDNN_CPU_ISA_HINTS", "PREFER_YMM", 1);
    auto got = get_cpu_isa_hints();

#if defined(DNNL_ENABLE_CPU_ISA_HINTS)
    // Expect env var value to be set when env variable feature is enabled.
    EXPECT_EQ(
            got, has_cpu ? cpu_isa_hints::prefer_ymm : cpu_isa_hints::no_hints);
#else
    // Otherwise, don't expect it to be set.
    EXPECT_NE(got, cpu_isa_hints::prefer_ymm);
#endif

#if (DNNL_CPU_RUNTIME != DNNL_RUNTIME_NONE)
    // `get_cpu_isa_hints` freezes the hints value, any call to set it
    // again results in runtime_error.
    auto st = set_cpu_isa_hints(cpu_isa_hints::no_hints);
    EXPECT_EQ(st, status::runtime_error);
#endif
}
#endif // DNNL_X64

TEST(onednn_primitive_cache_capacity_env_var_test, TestEnvVars) {
    custom_setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "11", 1);
    auto got = get_primitive_cache_capacity();
#ifndef DNNL_DISABLE_PRIMITIVE_CACHE
    EXPECT_EQ(got, 11);

    set_primitive_cache_capacity(8);
    auto func_got = get_primitive_cache_capacity();
    EXPECT_EQ(func_got, 8);
#else
    EXPECT_EQ(got, 0);
#endif
}

TEST(onednn_default_fpmath_mode_env_var_test, TestEnvVars) {
    custom_setenv("ONEDNN_DEFAULT_FPMATH_MODE", "BF16", 1);
    EXPECT_EQ(get_default_fpmath_mode(), fpmath_mode::bf16);

    EXPECT_EQ(set_default_fpmath_mode(fpmath_mode::strict), status::success);
    EXPECT_EQ(get_default_fpmath_mode(), fpmath_mode::strict);
}

TEST(onednn_verbose_env_var_test, TestEnvVars) {
    custom_setenv("ONEDNN_VERBOSE", "profile", 1);
#if !defined(DISABLE_VERBOSE)
    EXPECT_TRUE(verbose_profiling_enabled());
#endif

    // The setting function takes precedence over the environment variable.
    EXPECT_EQ(set_verbose(0), status::success);
    EXPECT_FALSE(verbose_profiling_enabled());

    // With asynchronous verbose profiling, in-order GPU streams use
    // device-measured timings only if exec profiling was on at stream
    // creation (enabling it later via set_verbose() falls back to
    // host-measured timings).
}

// The rest of the variables have no programmable public API to identify if
// they were set through env var or not, so they are not tested here.

} // namespace dnnl
