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

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS)
find_package(LAPACK)
find_package(OpenCL REQUIRED)

if(BLAS_FOUND AND LAPACK_FOUND)
    set(GEMMSTONE_WITH_MKL true)
else()
    message(WARNING "MKL not found; using mini_mkl package (slow) for CPU reference results.")
    add_library(mini_mkl STATIC mini_mkl.cpp mini_mkl.hpp)
    target_include_directories(mini_mkl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    if(WIN32)
        target_compile_options(mini_mkl PRIVATE /bigobj)
    else()
        target_compile_options(mini_mkl PRIVATE -g0 -O3)
    endif()
endif()

if(WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

# Misc utilities

add_executable(whichgen_ngen whichgen_ngen.cpp)
target_link_libraries(whichgen_ngen PRIVATE OpenCL::OpenCL)

add_executable(minicloc minicloc.c)
target_link_libraries(minicloc PRIVATE OpenCL::OpenCL)

if(LevelZero_FOUND)
  include_directories(${PROJECT_SOURCE_DIR}/third_party)
endif()

# Frontend targets

function(make_frontend_target format name)
    set(FRONTEND_EXECUTABLE ${format}frontend)
    set(FRONTEND_LIB gemmstone_${FRONTEND_EXECUTABLE})

    add_library(${FRONTEND_LIB} frontend.cpp)
    target_compile_definitions(${FRONTEND_LIB} PRIVATE FORMAT_ABBREV_${format})
    target_link_libraries(${FRONTEND_LIB} PUBLIC gemmstone_generator_${name})
    target_link_libraries(${FRONTEND_LIB} PRIVATE OpenCL::OpenCL)

    foreach(GPU ${GPUS})
        target_compile_definitions(${FRONTEND_LIB} PRIVATE FRONTEND_ENABLE_${GPU})
    endforeach()

    add_executable(${FRONTEND_EXECUTABLE} frontend_cli.cpp)
    target_link_libraries(${FRONTEND_EXECUTABLE} PRIVATE ${FRONTEND_LIB})
    if(NOT WIN32)
        target_link_libraries(${FRONTEND_EXECUTABLE} PRIVATE rt)
    endif()
endfunction()

make_frontend_target("" asm)
make_frontend_target("b" bin)
if(LevelZero_FOUND)
    make_frontend_target("e" ze)
endif()
if(OpenCL_FOUND)
    make_frontend_target("o" ocl)
endif()

# Copy planner test target
add_executable(test_copy_planner test_copy_planner.cpp)
target_link_libraries(test_copy_planner generator)

add_subdirectory(benchstone)

add_executable(test_kernel_modifiers ../tests/test_kernel_modifiers.cpp benchstone/kernel_info.cpp kernel_catalog.cpp)
target_link_libraries(test_kernel_modifiers PRIVATE gemmstone_ofrontend catalog_tools)
add_test(NAME kernel_modifiers COMMAND test_kernel_modifiers)

if(OpenCL_FOUND)
    add_executable(ktool ktool.cpp kernel_fit.cpp kernel_catalog.cpp)
    target_link_libraries(ktool PRIVATE catalog_tools)
    target_link_libraries(ktool PRIVATE OpenCL::OpenCL gemmstone_ofrontend)
    if(LevelZero_FOUND)
        target_link_libraries(ktool PRIVATE LevelZero::LevelZero)
    endif()
    link_benchstone_lib(ktool)
    if(GEMMSTONE_WITH_MKL)
        target_link_libraries(ktool PRIVATE LAPACK::LAPACK)
        target_compile_definitions(ktool PRIVATE GEMMSTONE_WITH_MKL)
    else()
        target_link_libraries(ktool PRIVATE mini_mkl)
    endif()
    add_test(NAME "ktool_gemm_generation" COMMAND ktool "-g" WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")
else()
    message(WARNING "OpenCL was not found; not building ktool.")
endif()

# Derby targets (Linux only)

if(NOT WIN32)
    add_executable(pvc_derby pvc_derby.cpp derby_utils.cpp)
    add_dependencies(pvc_derby benchstone)
    target_compile_options(pvc_derby PRIVATE -Wno-format-security)
    target_link_libraries(pvc_derby OpenCL::OpenCL)
endif()

add_executable(multiderby multiderby.cpp multiderby_client.cpp multiderby_optimizer.cpp derby_utils.cpp)
target_link_libraries(multiderby PRIVATE gemmstone_ofrontend)
target_link_libraries(multiderby PRIVATE generator)
link_benchstone_lib(multiderby)
if(GEMMSTONE_WITH_MKL)
    target_link_libraries(multiderby PRIVATE BLAS::BLAS LAPACK::LAPACK)
    target_compile_definitions(multiderby PRIVATE GEMMSTONE_WITH_MKL)
else()
    target_link_libraries(multiderby PRIVATE mini_mkl)
endif()

# Place all binaries directly in the build folder for easy use

get_property(TOOL_TARGETS DIRECTORY . PROPERTY BUILDSYSTEM_TARGETS)
list(APPEND TOOL_TARGETS benchstone)
set_target_properties(${TOOL_TARGETS} PROPERTIES RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_BINARY_DIR}>)
