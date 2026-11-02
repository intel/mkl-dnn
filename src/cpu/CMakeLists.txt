#===============================================================================
# Copyright 2019 Intel Corporation
# Copyright 2020 Arm Ltd. and affiliates
# Copyright 2021 FUJITSU LIMITED
# Copyright 2026 Advanced Micro Devices, Inc.
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

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp)
file(GLOB_RECURSE SOURCES_EXTRA
    ${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/matmul/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/reorder/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/rnn/*.[ch]pp
    ${CMAKE_CURRENT_SOURCE_DIR}/ukernel/*.[ch]pp
    )

foreach(SOURCE_FILE ${SOURCES_EXTRA})
    list(APPEND SOURCES "${SOURCE_FILE}")
endforeach()

if((DNNL_TARGET_ARCH STREQUAL "X64") OR (DNNL_TARGET_ARCH STREQUAL "AARCH64") OR (DNNL_TARGET_ARCH STREQUAL "RV64"))
    file(GLOB_RECURSE SOURCES_JIT_UTILS
        ${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/*.[ch]
        ${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/*.[ch]pp
    )
    foreach(SOURCE_FILE ${SOURCES_JIT_UTILS})
        list(APPEND SOURCES "${SOURCE_FILE}")
    endforeach()
endif()

if(DNNL_TARGET_ARCH STREQUAL "PPC64")
    file(GLOB FILES_REQUIRED_OPT
        ${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.[ch]pp
    )
    if(NOT UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        set_source_files_properties(${FILES_REQUIRED_OPT}
            PROPERTIES COMPILE_FLAGS "-O3 -funroll-loops")
    endif()
endif()

if(NOT DNNL_ENABLE_JIT_PROFILING)
    # XXX: the profiling interface will still be built and present
    add_definitions_with_host_compiler(-DDNNL_ENABLE_JIT_PROFILING=0)
    # Don't enable support for linux_perf and VTune Profiler
    if((DNNL_TARGET_ARCH STREQUAL "X64") OR (DNNL_TARGET_ARCH STREQUAL "AARCH64"))
        list(REMOVE_ITEM SOURCES_EXTRA
            "${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/jitprofiling/jitprofiling.c"
            "${CMAKE_CURRENT_SOURCE_DIR}/jit_utils/linux_perf/linux_perf.cpp"
            )
    endif()
endif()

if(DNNL_WERROR)
    # check if warn_unused_result can be used in an alias
    # only try with DNNL_WERROR enabled to avoid spurious warnings
    set(COMPILER_ALLOWS_ALIAS_ATTRIBUTES_SOURCE
    "
        typedef enum { dnnl_status_success = 0, } dnnl_status_t;

        using status_t __attribute__((warn_unused_result)) = dnnl_status_t;

        int main() { return 0; }
    ")
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("${COMPILER_ALLOWS_ALIAS_ATTRIBUTES_SOURCE}" COMPILER_ALLOWS_ALIAS_ATTRIBUTES)

    if(${COMPILER_ALLOWS_ALIAS_ATTRIBUTES})
        file(GLOB FILES_WITH_STATUS_NODISCARD
            ${CMAKE_CURRENT_SOURCE_DIR}/rnn/*.cpp
            )
        set_source_files_properties(${FILES_WITH_STATUS_NODISCARD}
            PROPERTIES COMPILE_FLAGS "-DDNNL_STATUS_NODISCARD")
    endif()
endif()

# C4244 is only fixed for x64 without ZenDNN, keep it suppressed elsewhere.
if (NOT DNNL_TARGET_ARCH STREQUAL "X64" OR DNNL_X64_USE_ZEN)
    if (MSVC)
        if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
            # int64_t -> int (tent)
            append(CMAKE_C_FLAGS "/wd4244")
            append(CMAKE_CXX_FLAGS "/wd4244")
        endif()
    endif()
endif()

# Clang/IntelLLVM counterpart of MSVC C4244/C4267, kept enabled above for x64.
# int-to-float is excluded: MSVC reports it at /W4 only, so it is out of parity.
# GCC is skipped: it has no equivalent sub-flag and -Wconversion is far broader.
if (DNNL_TARGET_ARCH STREQUAL "X64" AND NOT DNNL_X64_USE_ZEN AND NOT MSVC)
    if(CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang"
            OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
        append(CMAKE_CXX_FLAGS "-Wimplicit-int-conversion -Wshorten-64-to-32")
        append(CMAKE_CXX_FLAGS
            "-Wimplicit-float-conversion -Wno-implicit-int-float-conversion")
        # Keep the diagnostics scoped to oneDNN sources.
        include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/third_party/xbyak)
    endif()
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # to make computations more stable and to align the jitted code
    # with the reference one use precise division and square root
    # by default
    file(GLOB FILES_REQUIRED_PREC_SQRT
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp
        )
    file(GLOB FILES_REQUIRED_PREC_DIV
        ${CMAKE_CURRENT_SOURCE_DIR}/*resampling*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/*normalization*.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/ref_eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/nhwc_pooling.cpp
        )
    if(WIN32)
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "/Qprec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "/Qprec-div")
    else()
        set_source_files_properties(${FILES_REQUIRED_PREC_SQRT}
            PROPERTIES COMPILE_FLAGS "-prec-sqrt")
        set_source_files_properties(${FILES_REQUIRED_PREC_DIV}
            PROPERTIES COMPILE_FLAGS "-prec-div")
    endif()
endif()

if(DNNL_WITH_SYCL)
    if(DNNL_CPU_RUNTIME MATCHES "^(DPCPP|SYCL)$")
        add_subdirectory(sycl)
    endif()

    set(FILES_WITHNO_OPT)
    foreach(src in ${SOURCES})
        string(REGEX MATCH ".*jit.*kern.cpp" match ${src})
        if(match)
            list(APPEND FILES_WITHNO_OPT ${src})
        endif()
    endforeach()

    set_source_files_properties(${FILES_WITHNO_OPT}
        PROPERTIES COMPILE_FLAGS "-O0")
endif()

if(NOT DNNL_CPU_RUNTIME STREQUAL "THREADPOOL")
    # avoid building and linking empty objects
    list(REMOVE_ITEM SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/cpu_stream_threadpool.cpp")
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_cpu)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
if (DNNL_X64_USE_ZEN)
    # Link ZenDNN to dnnl_cpu (propagates includes and dependencies automatically)
    target_link_zendnnl(${OBJ_LIB} PRIVATE)
    # ZenDNN public headers require C++17. Scoped to this target only.
    target_compile_features(${OBJ_LIB} PRIVATE cxx_std_17)
endif()
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

if (DNNL_TARGET_ARCH STREQUAL "X64")
    add_subdirectory(x64)
endif()
if (DNNL_TARGET_ARCH STREQUAL "AARCH64")
    add_subdirectory(aarch64)
endif()
if (DNNL_TARGET_ARCH STREQUAL "PPC64")
    add_subdirectory(ppc64)
endif()
if (DNNL_TARGET_ARCH STREQUAL "S390X")
    add_subdirectory(s390x)
endif()
if (DNNL_TARGET_ARCH STREQUAL "RV64")
    add_subdirectory(rv64)
endif()
