#===============================================================================
# Copyright 2019 Intel Corporation
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

if(SYCL_cmake_included)
    return()
endif()
set(SYCL_cmake_included true)

if("${DNNL_CPU_RUNTIME}" MATCHES "^(DPCPP|SYCL)$" AND NOT DNNL_GPU_RUNTIME STREQUAL DNNL_CPU_RUNTIME)
    message(FATAL_ERROR "CPU runtime ${DNNL_CPU_RUNTIME} requires GPU runtime ${DNNL_CPU_RUNTIME}")
endif()

if(DNNL_CPU_RUNTIME STREQUAL "DPCPP" OR DNNL_CPU_RUNTIME STREQUAL "SYCL")
    set(DNNL_CPU_SYCL true)
else()
    set(DNNL_CPU_SYCL false)
endif()

if(DNNL_GPU_RUNTIME STREQUAL "DPCPP" OR DNNL_GPU_RUNTIME STREQUAL "SYCL")
    set(DNNL_GPU_SYCL true)
    set(DNNL_SYCL_CUDA OFF)
    set(DNNL_SYCL_HIP OFF)
    set(DNNL_SYCL_GENERIC OFF)
    if(DNNL_GPU_VENDOR STREQUAL "NVIDIA")
        set(DNNL_SYCL_CUDA ON)
    endif()
    if(DNNL_GPU_VENDOR STREQUAL "AMD")
        set(DNNL_SYCL_HIP ON)
    endif()
    if(DNNL_GPU_VENDOR STREQUAL "GENERIC")
        set(DNNL_SYCL_GENERIC ON)
    endif()
else()
    set(DNNL_GPU_SYCL false)
endif()

if(DNNL_CPU_SYCL OR DNNL_GPU_SYCL)
    set(DNNL_WITH_SYCL true)
else()
    set(DNNL_WITH_SYCL false)
endif()

include("cmake/host_compiler_id.cmake")

if(NOT DNNL_WITH_SYCL)
    return()
endif()

include(FindPackageHandleStandardArgs)
include("cmake/dpcpp_driver_check.cmake")

# Link SYCL library explicitly for open-source compiler on Windows and for
# MSVC host compiler, as host objects produced by MSVC lack the `/DEFAULTLIB`
# directive for the SYCL library, and at link time the compiler driver selects
# the release version of the library regardless of the C runtime library used.
# In other cases, the compiler is able to automatically link it.
if(WIN32 AND (CMAKE_BASE_NAME STREQUAL "clang++"
        OR DPCPP_HOST_COMPILER_KIND STREQUAL "MSVC"))
    # TODO: we can drop this workaround once an open-source release
    # for Windows has a fix for the issue.
    get_filename_component(_sycl_lib_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    foreach(sycl_lib_version 9 8 7 6 "")
        if(UPPERCASE_CMAKE_BUILD_TYPE MATCHES "^(DEBUG|RELWITHMDD)$")
            set(SYCL_LIBRARY_NAME "sycl${sycl_lib_version}d")
        else()
            set(SYCL_LIBRARY_NAME "sycl${sycl_lib_version}")
        endif()

        find_library(SYCL_LIBRARY ${SYCL_LIBRARY_NAME}
            HINTS "${_sycl_lib_dir}/../lib")

        if(EXISTS "${SYCL_LIBRARY}")
            list(APPEND EXTRA_SHARED_LIBS ${SYCL_LIBRARY})
            set(SYCL_LIBRARY_FOUND TRUE)
            break()
        endif()
    endforeach()
    if(NOT SYCL_LIBRARY_FOUND)
        message(FATAL_ERROR "Cannot find a SYCL library")
    endif()
endif()

# CUDA and ROCm contain OpenCL headers that conflict with the OpenCL
# headers located in the compiler's directory.
# The workaround is to get interface include directories from all CUDA/ROCm
# import targets and lower their priority via `-idirafter` so that the
# compiler picks up the proper OpenCL headers.
macro(adjust_headers_priority targets)
    if(NOT WIN32)
        set(include_dirs)
        foreach(import_target ${targets})
            get_target_property(import_target_include_dirs ${import_target} INTERFACE_INCLUDE_DIRECTORIES)
            set_target_properties(${import_target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "")
            list(APPEND include_dirs ${import_target_include_dirs})
        endforeach()

        list(REMOVE_DUPLICATES include_dirs)
        foreach(include_dir ${include_dirs})
            append(CMAKE_CXX_FLAGS "-idirafter${include_dir}")
        endforeach()
    endif()
endmacro()

macro(suppress_warnings_for_nvidia_target)
    # XXX: Suppress warning coming from SYCL headers:
    #   error: use of function template name with no prior declaration in
    #   function call with eplicit template arguments is a C++20 extension
    append(CMAKE_CXX_FLAGS "-Wno-c++20-extensions")

    # Suppress LLVM warning about not supporting latest cuda. It's safe enough
    # as long as no new cuda features are used in SYCL kernels.
    append(CMAKE_CXX_FLAGS "-Wno-unknown-cuda-version")
endmacro()

if(DNNL_SYCL_CUDA)
    suppress_warnings_for_nvidia_target()
    find_package(cuBLAS REQUIRED)
    find_package(cublasLt REQUIRED)
    find_package(cuDNN REQUIRED)

    adjust_headers_priority("cuBLAS::cuBLAS;cuDNN::cuDNN;cublasLt::cublasLt")
    add_definitions_with_host_compiler("-DCUDA_NO_HALF")

    list(APPEND EXTRA_SHARED_LIBS cuBLAS::cuBLAS cuDNN::cuDNN cublasLt::cublasLt)
    message(STATUS "DPC++ support is enabled (CUDA)")
elseif(DNNL_SYCL_HIP)
    find_package(HIP REQUIRED)
    find_package(rocBLAS REQUIRED)
    find_package(MIOpen REQUIRED)

    adjust_headers_priority("HIP::HIP;rocBLAS::rocBLAS;MIOpen::MIOpen")
    add_definitions_with_host_compiler("-D__HIP_PLATFORM_AMD__=1")

    if(NOT "${DNNL_AMD_SYCL_KERNELS_TARGET_ARCH}" STREQUAL "")
        add_definitions(-DDNNL_AMD_ENABLE_SYCL_KERNELS)
        set(DNNL_AMD_ENABLE_SYCL_KERNELS TRUE)
    endif()

    list(APPEND EXTRA_SHARED_LIBS HIP::HIP rocBLAS::rocBLAS MIOpen::MIOpen)
    message(STATUS "DPC++ support is enabled (HIP)")
elseif(DNNL_SYCL_GENERIC)
    CHECK_CXX_COMPILER_FLAG("-fsycl -fsycl-targets=nvptx64-nvidia-cuda" NVIDIA_TARGET_SUPPORTED)

    if(NVIDIA_TARGET_SUPPORTED)
        suppress_warnings_for_nvidia_target()
    endif()
else()
    # In order to support large shapes.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sycl-id-queries-fit-in-int")
    message(STATUS "DPC++ support is enabled (OpenCL and Level Zero)")
endif()

if(DNNL_INTERNAL_ENABLE_GENERIC_SYCL_KERNELS)
    add_definitions(-DDNNL_ENABLE_SYCL_KERNELS)
endif()

# XXX: Suppress warning coming from SYCL headers:
#   #pragma message("The Intel extensions have been moved into cl_ext.h.
#   Please include cl_ext.h directly.")
if(NOT WIN32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-#pragma-messages")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl")

if(DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER)
    include(CheckCXXSourceCompiles)
    set(CHECK_SYCL_KERNEL_COMPILER_SOURCE
    "
        #include <sycl/sycl.hpp>
        namespace syclex = sycl::ext::oneapi::experimental;
        int main() {
            sycl::device d;
            (void)d.ext_oneapi_can_build(syclex::source_language::opencl);
            return 0;
        }
    ")
    CHECK_CXX_SOURCE_COMPILES(
        "${CHECK_SYCL_KERNEL_COMPILER_SOURCE}"
        SYCL_KERNEL_COMPILER_DETECTED)
    if(NOT SYCL_KERNEL_COMPILER_DETECTED)
        message(FATAL_ERROR
"SYCL implementation does not support OpenCL kernel compiler extension. Make sure that SYCL and OCLOC are correctly installed.")
    endif()
endif()

if(DNNL_EXPERIMENTAL_ENABLE_SYCL_PROFILING_TAG AND DNNL_VERBOSE AND DNNL_GPU_SYCL)
    include(CheckCXXSourceCompiles)
    set(CHECK_DNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG_SOURCE
    "
        #include <sycl/sycl.hpp>
        #ifndef DNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG
        #error sycl_ext_oneapi_profiling_tag not supported
        #endif
        int main() {
            sycl::queue q;
            sycl::ext::oneapi::experimental::submit_profiling_tag(q);
            return 0;
        }
    ")
    CHECK_CXX_SOURCE_COMPILES(
        "${CHECK_DNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG_SOURCE}"
        DNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG)
    if(DNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG)
        add_definitions(-DDNNL_USE_SYCL_EXT_ONEAPI_PROFILING_TAG)
    else()
        message(STATUS "sycl_ext_oneapi_profiling_tag extension is not available 
        - verbose profiler will be paused without a profiling-enabled SYCL queue")
    endif()
endif()
