#===============================================================================
# Copyright 2024 Intel Corporation
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

# TODO: Extend to all -Wconversion warnings
if(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" OR CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    append(CMAKE_CXX_FLAGS "-Wbitfield-enum-conversion -Wbool-conversion -Wconstant-conversion -Wenum-conversion -Wimplicit-int-conversion -Wliteral-conversion -Wnon-literal-null-conversion -Wnull-conversion")
    add_definitions("-DENABLE_LLVM_WCONVERSION")
endif()

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp)
file(GLOB SOURCES_GEMM ${CMAKE_CURRENT_SOURCE_DIR}/gemm/*.[ch]pp)
list(APPEND SOURCES "${SOURCES_GEMM}")
set(DIRS "binary;bnorm;conv;concat;eltwise;gnorm;ip;lnorm;lrn;matmul;pool;prelu;reduction;reorder;resampling;rnn;sdpa;shuffle;softmax;sum;zeropad")
foreach(dir ${DIRS})
    file(GLOB_RECURSE SOURCES_EXTRA
        ${CMAKE_CURRENT_SOURCE_DIR}/${dir}/*.[ch]pp
        )
    list(APPEND SOURCES "${SOURCES_EXTRA}")
endforeach()
add_subdirectory(gemm/jit)

include("${PROJECT_SOURCE_DIR}/cmake/gen_gpu_kernel_list.cmake")

file(GLOB_RECURSE CL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cl)
# gemm/jit OpenCL kernels are benchstone-only
list(FILTER CL_SOURCES EXCLUDE REGEX "/gemm/jit/")
file(GLOB_RECURSE CL_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hxx
    )
# gemm/jit headers are C++, not OpenCL
list(FILTER CL_HEADERS EXCLUDE REGEX "/gemm/jit/")

set(kernel_list_templ "${PROJECT_SOURCE_DIR}/src/gpu/intel/ocl_kernel_list.cpp.in")
set(kernel_list_src "${PROJECT_BINARY_DIR}/src/gpu/intel/ocl_kernel_list.cpp")

gen_gpu_kernel_list(
    "${kernel_list_templ}"
    "${kernel_list_src}"
    "${CL_SOURCES}"
    "${CL_HEADERS}")

list(APPEND SOURCES ${kernel_list_src})
list(FILTER SOURCES EXCLUDE REGEX "conv/jit/v2/planner")
if (DNNL_EXPERIMENTAL AND DNNL_DEV_MODE AND DNNL_GPU_RUNTIME STREQUAL "OCL")
add_subdirectory("conv/jit/v2/planner")
endif()

add_subdirectory(compute)
add_subdirectory(jit)

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    add_subdirectory(ocl)
endif()
if(DNNL_GPU_RUNTIME STREQUAL "ZE")
    add_subdirectory(ze)
endif()
if(DNNL_WITH_SYCL)
    add_subdirectory(sycl)
# INCLUDE_EXTRA_DIRS_FOR_SYCL:
# For the code sharing purpose, SYCL backend includes headers from OpenCL and
# Level Zero backends but this puts certain restrictions on what can compile in
# the latter files. To avoid extra manipulations with files, some functions are
# guarded with evident runtime selection but this guarding is done solely for
# the purpose of compilation under SYCL backend.
    add_subdirectory(ocl)
    add_subdirectory(ze)
endif()

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_intel)
add_library(${OBJ_LIB} OBJECT ${SOURCES})
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
