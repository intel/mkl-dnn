#===============================================================================
# Copyright 2020 Intel Corporation
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

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

add_subdirectory(generic)

if(DNNL_GPU_VENDOR STREQUAL "INTEL")
    include_directories_with_host_compiler(${CMAKE_CURRENT_SOURCE_DIR}/intel/jit/config/)

    set(ONEDNN_NGEN_DIR "${PROJECT_SOURCE_DIR}/third_party/ngen" CACHE PATH "Path to nGEN source code")
    add_definitions_with_host_compiler(-DNGEN_CONFIG)
    include_directories_with_host_compiler(${ONEDNN_NGEN_DIR})

    set(ONEDNN_GEMMSTONE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/intel/gemmstone" CACHE PATH "Path to gemmstone source code")
    add_definitions_with_host_compiler(-DGEMMSTONE_CONFIG)
    include_directories_with_host_compiler(${ONEDNN_GEMMSTONE_DIR}/include)

    add_subdirectory(intel)
endif()

if(DNNL_GPU_VENDOR STREQUAL "NVIDIA")
    add_subdirectory(nvidia)
endif()

if(DNNL_GPU_VENDOR STREQUAL "AMD")
    add_subdirectory(amd)
endif()

if (NOT DNNL_GPU_VENDOR STREQUAL "INTEL")
    if (MSVC)
        if(${CMAKE_CXX_COMPILER_ID} STREQUAL MSVC)
            # int64_t -> int (tent)
            append(CMAKE_CCXX_NOWARN_FLAGS "/wd4244")
        endif()
    endif()
endif()

# check if warn_unused_result can be used in an alias
set(COMPILER_ALLOWS_ALIAS_ATTRIBUTES_SOURCE
"
    typedef enum { dnnl_status_success = 0, } dnnl_status_t;
    using status_t __attribute__((warn_unused_result)) = dnnl_status_t;
    int main() { return 0; }
")
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-Werror")
check_cxx_source_compiles("${COMPILER_ALLOWS_ALIAS_ATTRIBUTES_SOURCE}" COMPILER_ALLOWS_ALIAS_ATTRIBUTES)
unset(CMAKE_REQUIRED_FLAGS)

if(${COMPILER_ALLOWS_ALIAS_ATTRIBUTES})
    if(DNNL_GPU_VENDOR STREQUAL "INTEL")
        add_definitions_with_host_compiler(-DDNNL_STATUS_NODISCARD)
    endif()
endif()

if(DNNL_DISABLE_GPU_REF_KERNELS)
    message(STATUS "GPU Compute reference kernels are disabled")
else()
    message(STATUS "GPU Compute reference kernels are enabled")
endif()
