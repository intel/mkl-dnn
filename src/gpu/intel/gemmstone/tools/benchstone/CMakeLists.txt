#===============================================================================
# Copyright 2026 Intel Corporation
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

set(BENCHSTONE_SOURCES
    benchstone_accuracy.cpp
    benchstone_registry.cpp
    benchstone_ocl.cpp
    kernel_info.cpp
    benchstone_matrix.cpp
    benchstone_ref.cpp
    ../perf_lock.cpp)

if(LevelZero_FOUND)
    list(APPEND BENCHSTONE_SOURCES benchstone_ze.cpp)
else()
    message(WARNING "Level Zero was not found; benchstone will not be able to use the Level Zero runtime.")
endif()

# Embed cache kernel OpenCL sources.
set(CACHE_KERNELS_CL "")
foreach(KERNEL flush lsc_flush flush_icache cache_fill)
    set(KERNEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/../ocl_kernels/${KERNEL}.cl)
    file(READ ${KERNEL_FILE} KERNEL_SOURCE)
    string(APPEND CACHE_KERNELS_CL "const char *const ${KERNEL}_cl = R\"CL(${KERNEL_SOURCE})CL\";\n")
    string(APPEND CACHE_KERNELS_CL "const char *const ${KERNEL}_name = \"${KERNEL}_kernel\";\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${KERNEL_FILE})
endforeach()
file(CONFIGURE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/flush_kernel_cl.h CONTENT "@CACHE_KERNELS_CL@" @ONLY)

# oneDNN's MDAPI header, for builds from src/gpu/intel/gemmstone
get_filename_component(dnnl_root ${PROJECT_SOURCE_DIR}/../../../.. ABSOLUTE)
if(EXISTS ${dnnl_root}/third_party/mdapi/mdapi/metrics_discovery_api.h)
    set(MDAPI_INCLUDE_DIR ${dnnl_root}/third_party/mdapi)
else()
    set(MDAPI_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party)
endif()

add_library(gemmstone_benchstone STATIC ${BENCHSTONE_SOURCES})
target_include_directories(gemmstone_benchstone PRIVATE ${CMAKE_CURRENT_BINARY_DIR} ${MDAPI_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/third_party)
target_link_libraries(gemmstone_benchstone PUBLIC OpenCL::OpenCL)
if(GEMMSTONE_WITH_MKL)
    target_link_libraries(gemmstone_benchstone PUBLIC BLAS::BLAS)
    target_compile_definitions(gemmstone_benchstone PRIVATE GEMMSTONE_WITH_MKL)
else()
    target_link_libraries(gemmstone_benchstone PRIVATE mini_mkl)
endif()
if(LevelZero_FOUND)
    target_link_libraries(gemmstone_benchstone PUBLIC LevelZero::LevelZero)
endif()
if(NOT WIN32)
    target_link_libraries(gemmstone_benchstone PUBLIC dl)
endif()

function(link_benchstone_lib target)
    target_link_libraries(${target} PRIVATE gemmstone_benchstone)

    # Backend factories are registered via static initializers; force-include
    # backend objects so registration always runs.
    if(MSVC)
        target_link_options(${target} PRIVATE "/WHOLEARCHIVE:$<TARGET_FILE:gemmstone_benchstone>")
    else()
        target_link_options(${target} PRIVATE
            "-Wl,--whole-archive,$<TARGET_FILE:gemmstone_benchstone>,--no-whole-archive")
    endif()
endfunction()

add_executable(benchstone benchstone_cli.cpp)
target_link_libraries(benchstone PRIVATE gemmstone_ofrontend)
link_benchstone_lib(benchstone)

if(NOT WIN32)
    #set_source_files_properties(benchstone_matrix.cpp PROPERTIES COMPILE_FLAGS "-g0 -O3")
endif()
