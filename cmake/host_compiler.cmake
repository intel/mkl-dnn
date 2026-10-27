#===============================================================================
# Copyright 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

if(host_compiler_cmake_included)
    return()
endif()
set(host_compiler_cmake_included true)

# There is nothing to do for the default host compiler.
if(DPCPP_HOST_COMPILER_KIND STREQUAL "DEFAULT")
    return()
endif()

if(NOT DPCPP_HOST_COMPILER_KIND MATCHES "^(GNU|CLANG|MSVC)$")
    message(FATAL_ERROR "The DNNL_DPCPP_HOST_COMPILER value ${DNNL_DPCPP_HOST_COMPILER} is not supported")
endif()

if(NOT DNNL_TARGET_ARCH STREQUAL "X64")
    message(FATAL_ERROR "The DNNL_DPCPP_HOST_COMPILER option is only supported for DNNL_TARGET_ARCH=X64")
endif()

if(DPCPP_HOST_COMPILER_KIND MATCHES "^(GNU|CLANG)$")
    # Common flags for GNU and CLANG IDs.
    platform_unix_and_mingw_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    platform_unix_and_mingw_common_cxx_flags(DPCPP_HOST_COMPILER_OPTS)
    # Stripping Wundef before passing it into a host compiler.
    string(REGEX REPLACE "-Wundef" "" DPCPP_HOST_COMPILER_OPTS "${DPCPP_HOST_COMPILER_OPTS}")

    sdl_unix_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    sdl_unix_src_ccxx_flags(DPCPP_SRC_COMPILER_OPTS)
    sdl_unix_example_ccxx_flags(DPCPP_EXAMPLE_COMPILER_OPTS)

    # SYCL uses C++17 features in headers hence C++17 support should be enabled
    # for host compiler.
    # The main compiler driver doesn't automatically specify C++ standard for
    # custom host compilers.
    append(DPCPP_HOST_COMPILER_OPTS "-std=c++17")

    # Unconditionally enable OpenMP during compilation to use `#pragma omp simd`
    append(DPCPP_HOST_COMPILER_OPTS "-fopenmp")

    if(UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
        append(DPCPP_HOST_COMPILER_OPTS "${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        append(DPCPP_HOST_COMPILER_OPTS "${CMAKE_CXX_FLAGS_DEBUG}")
    endif()

    # Intel oneAPI DPC++/C++ Compiler version 2025.2 and earlier emits spurious
    # deprecation warnings from system headers when used
    # with '-fsycl-host-compiler' option.
    # Open source version identifies as Clang and does not offer a reliable way
    # to detect version. Warning is suppressed unconditionally in this case.
    if((CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM"
        AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "2025.3")
        OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        append(DPCPP_HOST_COMPILER_OPTS "-Wno-deprecated-declarations")
    endif()

    # Using single_task, cgh.copy, cgh.fill may cause the following warning:
    # "warning: `clang::sycl_kernel` scoped attribute directive ignored [-Wattributes]"
    # We don't have control over it so just suppress it for the time being.
    append(DPCPP_HOST_COMPILER_OPTS "-Wno-attributes")

    if(DPCPP_HOST_COMPILER_KIND STREQUAL "GNU")
        platform_gnu_nowarn_ccxx_flags(DPCPP_CXX_NOWARN_FLAGS ${DPCPP_HOST_COMPILER_MAJOR_VER}.${DPCPP_HOST_COMPILER_MINOR_VER})
        sdl_gnu_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS DPCPP_HOST_COMPILER_VER)

        # SYCL headers contain some comments that trigger warning with GNU compiler
        append(DPCPP_HOST_COMPILER_OPTS "-Wno-comment")

        # Host compiler operates on preprocessed files and headers, and it
        # mistakenly assumes that anonymous namespace types are used from a header
        # which is not always the case.
        append(DPCPP_HOST_COMPILER_OPTS "-Wno-subobject-linkage")
    elseif(DPCPP_HOST_COMPILER_KIND STREQUAL "CLANG")
        platform_clang_nowarn_ccxx_flags(DPCPP_CXX_NOWARN_FLAGS)
    endif()
elseif(DPCPP_HOST_COMPILER_KIND STREQUAL "MSVC")
    append(DPCPP_HOST_COMPILER_OPTS "/nologo")
    append_if(DNNL_WERROR DPCPP_HOST_COMPILER_OPTS "/WX")
    platform_msvc_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    platform_msvc_nowarn_ccxx_flags(DPCPP_CXX_NOWARN_FLAGS)
    sdl_msvc_common_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)

    # SYCL uses C++17 features in headers hence C++17 support should be enabled
    # for host compiler.
    append(DPCPP_HOST_COMPILER_OPTS "/std:c++17")

    # Unconditionally enable OpenMP during compilation to use `#pragma omp simd`.
    # Objects compiled with `/openmp` link against the MSVC OpenMP runtime
    # (vcomp).
    append(DPCPP_HOST_COMPILER_OPTS "/openmp")

    # Build type flags of the main compiler cannot be reused as they are not
    # MSVC-compatible for compilers with GNU-like command line (clang++). The
    # C runtime library must match the one used by the main compiler. Debug
    # information is embedded into objects (/Z7) as concurrent host compiler
    # instances cannot share the same program database file.
    set(_host_flags_DEBUG "/MDd /Z7 /Ob0 /Od /RTC1")
    set(_host_flags_RELEASE "/MD /O2 /Ob2 /DNDEBUG")
    set(_host_flags_RELWITHDEBINFO "/MD /Z7 /O2 /Ob1 /DNDEBUG")
    set(_host_flags_MINSIZEREL "/MD /O1 /Ob1 /DNDEBUG")
    set(_host_flags_RELWITHASSERT "/MD /O2 /Ob2")
    set(_host_flags_RELWITHMDD "/MDd /O2 /Ob2")
    if(DEFINED _host_flags_${UPPERCASE_CMAKE_BUILD_TYPE})
        append(DPCPP_HOST_COMPILER_OPTS "${_host_flags_${UPPERCASE_CMAKE_BUILD_TYPE}}")
    else()
        append(DPCPP_HOST_COMPILER_OPTS "/MD")
    endif()

    # The compiler driver doesn't define SYCL_LANGUAGE_VERSION for the host
    # compiler, so sycl.hpp emits a warning on every include. GNU and Clang
    # host compilers suppress it as the SYCL headers are system headers, but
    # MSVC prints it via `#pragma message` regardless.
    append(DPCPP_HOST_COMPILER_OPTS "/DSYCL_DISABLE_FSYCL_SYCLHPP_WARNING")

    # Older compiler drivers pass SYCL include directories to the host
    # compiler with `-I` instead of `/external:I`, so warnings from SYCL
    # headers are reported and the ones enabled by `/sdl` fail the build.
    append(DPCPP_HOST_COMPILER_OPTS "/external:anglebrackets /external:W0")

    # Host objects produced by MSVC lack the SYCL runtime `/DEFAULTLIB`
    # directives that the compiler embeds. The driver adds the SYCL runtime
    # libraries to the link line only if `-fsycl-host-compiler` is passed at
    # link time as well. The SYCL library matching the C runtime library is
    # linked explicitly (see SYCL.cmake).
    string(APPEND CMAKE_CXX_LINK_FLAGS
        " -fsycl-host-compiler=\"${DPCPP_HOST_COMPILER}\" ")
endif()

# DPCPP_HOST_COMPILER_KIND specific flags.
if(DNNL_ARCH_OPT_FLAGS STREQUAL "HostOpts")
    if(DPCPP_HOST_COMPILER_KIND STREQUAL "GNU")
        platform_gnu_x64_arch_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    elseif(DPCPP_HOST_COMPILER_KIND STREQUAL "CLANG")
        platform_clang_x64_arch_ccxx_flags(DPCPP_HOST_COMPILER_OPTS)
    endif()
    # MSVC defaults already target the x64 baseline.
else()
    # Assumption is that the passed flags are compatible with the host compiler
    append(DPCPP_HOST_COMPILER_OPTS "${DNNL_ARCH_OPT_FLAGS}")
endif()

# When using a non-default host compiler the main compiler doesn't
# handle some arguments properly and issues the warning.
# Suppress the warning until the bug is fixed.
append(CMAKE_CXX_FLAGS "-Wno-unused-command-line-argument")

# Option `-fsycl-unnamed-lambda` is enabled by default, but not compatible
# with `-fsycl-host-compiler`. While icpx driver adds
# `-fno-sycl-unnamed-lambda` to avoid build issues clang++ does not do that.
append(CMAKE_CXX_FLAGS "-fno-sycl-unnamed-lambda")

# Quote the path as it may contain spaces on Windows.
append(CMAKE_CXX_FLAGS "-fsycl-host-compiler=\"${DPCPP_HOST_COMPILER}\"")
append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_HOST_COMPILER_OPTS}")
