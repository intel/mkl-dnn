#===============================================================================
# Copyright 2016-2024 Intel Corporation
# Copyright 2020, 2026 Arm Ltd. and affiliates
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

cmake_minimum_required(VERSION 3.13)

# CMake minimum required version enables all policies introduced in minimum
# version and earlier versions. Policies introduced in future versions
# are handled individually in the section below.

# CMake 3.14: Install rules from add_subdirectory() calls are interleaved
# with those in caller.
if(POLICY CMP0082)
    cmake_policy(SET CMP0082 NEW)
endif()

# CMake 3.27: The FindPythonInterp and FindPythonLibs modules are removed.
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 NEW)
endif()

# CMake 3.27: The FindCUDA module is removed.
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

# CMake 4.3: CMAKE_<LANG>_LINK_FLAGS adds link flags to all target types
if(POLICY CMP0210)
    cmake_policy(SET CMP0210 NEW)
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    message(STATUS "CMAKE_BUILD_TYPE is unset, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel RelWithAssert RelWithMDd...")
endif()
if (CMAKE_GENERATOR MATCHES "^Visual Studio")
    message(STATUS
"oneDNN build configuration is based on the CMAKE_BUILD_TYPE value, but
 the CMake generator '${CMAKE_GENERATOR}' does not respect it and requires
 using the --config option to choose the build type. Changing the build type
 using the --config option requires rerunning CMake from scratch with a
 matching CMAKE_BUILD_TYPE value.")
endif()

set(PROJECT_NAME "oneDNN")
set(PROJECT_FULL_NAME "oneAPI Deep Neural Network Library (oneDNN)")
set(PROJECT_VERSION "3.14.2")

project(${PROJECT_NAME} VERSION "${PROJECT_VERSION}" LANGUAGES C CXX)

if (NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "oneDNN supports 64 bit platforms only")
endif()

# Set the target architecture.
if(NOT DNNL_TARGET_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64.*|AARCH64.*|arm64.*|ARM64.*)")
        set(DNNL_TARGET_ARCH "AARCH64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ppc64.*|PPC64.*|powerpc64.*)")
        set(DNNL_TARGET_ARCH "PPC64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(s390x.*|S390X.*)")
        set(DNNL_TARGET_ARCH "S390X")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(rv.*|RV.*|riscv.*|RISCV.*)")
        set(DNNL_TARGET_ARCH "RV64")
    else()
        set(DNNL_TARGET_ARCH "X64")
    endif()
endif()
message(STATUS "DNNL_TARGET_ARCH: ${DNNL_TARGET_ARCH}")

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_SRC_CCXX_FLAGS)       # SRC specifics
set(CMAKE_EXAMPLE_CCXX_FLAGS)   # EXAMPLE specifics
set(CMAKE_TEST_CCXX_FLAGS)      # TESTS specifics

string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPERCASE_CMAKE_BUILD_TYPE)

set(LIB_PACKAGE_NAME "dnnl")

include("cmake/utils.cmake")
include("cmake/options.cmake")
include("cmake/SYCL.cmake")
include("cmake/OpenMP.cmake")
include("cmake/TBB.cmake")
include("cmake/Threadpool.cmake")
include("cmake/ZenDNN.cmake")
include("cmake/OpenCL.cmake")
include("cmake/LevelZero.cmake")
include("cmake/platform.cmake")
include("cmake/SDL.cmake")
include("cmake/ACL.cmake")
include("cmake/blas.cmake")
include("cmake/doc.cmake")
include("cmake/version.cmake")
include("cmake/coverage.cmake")
include("cmake/build_types.cmake")
include("cmake/testing.cmake")
include("cmake/host_compiler.cmake")
include("cmake/configuring_primitive_list.cmake")

if(DNNL_AARCH64_USE_KAI)
    add_definitions_with_host_compiler(-DDNNL_AARCH64_USE_KAI=1)
endif()

if(UNIX OR MINGW)
    # CMAKE_<lang>_STANDARD support, so set it to our defaults, unless
    # overridden by the user
    if(NOT DEFINED CMAKE_C_STANDARD)
        set(CMAKE_C_STANDARD 99)
    endif()
    if(NOT DEFINED CMAKE_CXX_STANDARD AND NOT DNNL_WITH_SYCL)
        set(CMAKE_CXX_STANDARD 11)
    endif()

    # Disable -std=gnuXX and -std=gnu++XX
    set(CMAKE_C_EXTENSIONS OFF)
    set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# Handle cases when OpenMP runtime is requested but not found: override CPU
# runtime to be sequential
if(DNNL_CPU_RUNTIME STREQUAL "OMP" AND
   DNNL_CPU_THREADING_RUNTIME STREQUAL "SEQ")
    set(DNNL_CPU_RUNTIME "SEQ" CACHE STRING "" FORCE)
endif()

enable_testing()

include_directories_with_host_compiler_before(${PROJECT_SOURCE_DIR}/include)

configure_file(
    "${PROJECT_SOURCE_DIR}/include/oneapi/dnnl/dnnl_config.h.in"
    "${PROJECT_BINARY_DIR}/include/oneapi/dnnl/dnnl_config.h"
)
include_directories_with_host_compiler_before(${PROJECT_BINARY_DIR}/include)

configure_file(
    "${PROJECT_SOURCE_DIR}/README.binary.in"
    "${PROJECT_BINARY_DIR}/README"
)

if(DNNL_INSTALL_MODE STREQUAL "BUNDLE" AND NOT DEFINED CMAKE_INSTALL_LIBDIR)
    # define CMAKE_INSTALL_LIBDIR as "lib" in the case of bundle
    set(CMAKE_INSTALL_LIBDIR "lib")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(tests)

if(DNNL_INSTALL_MODE STREQUAL "BUNDLE")
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
    install(FILES THIRD-PARTY-PROGRAMS DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
    install(FILES ${PROJECT_BINARY_DIR}/README DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
else()
    # Cannot use CMAKE_INSTALL_DOCDIR since it uses PROJECT_NAME and not DNNL_LIBRARY_NAME
    install(FILES LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
    install(FILES THIRD-PARTY-PROGRAMS DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
    install(FILES ${PROJECT_BINARY_DIR}/README DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME})
endif()
