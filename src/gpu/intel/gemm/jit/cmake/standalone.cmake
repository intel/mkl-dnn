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

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# Check and enable various tools to speed up build
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "Found ccache: ${CCACHE_PROGRAM}")
endif()

find_program(LLD_PROGRAM lld)
if(LLD_PROGRAM)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        # --allow-shlib-undefined matches ld.bfd's default. This to avoid failures
        # when a dependency lib (e.g. libze_loader.so) was compiled against a newer glibc version.
        # Symbols resolve at runtime.
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=lld -Wl,--allow-shlib-undefined")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fuse-ld=lld -Wl,--allow-shlib-undefined")
        message(STATUS "Linking with LLD")
    endif()
endif()

if(WIN32)
    add_definitions(-DNOMINMAX)
    add_definitions(-DWIN32_LEAN_AND_MEAN)
endif()

# Check GPU architectures to build by default. Targets will be made for all
#   architectures but will not be built by default.

set(ALL_GPUS "12LP;12HP;12p7;12p8;Xe2;Xe3;Xe3p")
set(GPUS "12p7;12p8;Xe2;Xe3;Xe3p" CACHE STRING "Semicolon-separated list of GPU architectures to build by default")
foreach(GPU ${GPUS})
    if(NOT ${GPU} IN_LIST ALL_GPUS)
        message(FATAL_ERROR "Unknown GPU architecture: ${GPU}")
    endif()
    string(TOUPPER "${GPU}" GPU)
    add_compile_definitions("GEMMSTONE_BUILD_${GPU}")
endforeach()
add_compile_definitions(PRERELEASE_HW=0)

message(STATUS "GPU architectures built by default (set with -DGPUS=<...>): ${GPUS}")

# Find dependent libraries: nGEN, Level Zero, and OpenCL

find_package(NGen REQUIRED)

# Enable supported runtime
add_compile_definitions(GEMMSTONE_WITH_ASM_RUNTIME=1)
add_compile_definitions(GEMMSTONE_WITH_BINARY_RUNTIME=1)
find_package(OpenCL)
if(OpenCL_FOUND)
  add_compile_definitions(GEMMSTONE_WITH_OPENCL_RUNTIME=1)
  include_directories(${OpenCL_INCLUDE_DIRS})
else()
  message(WARNING "OpenCL library was not found, disabling support.")
endif()

find_package(LevelZero)
if(LevelZero_FOUND)
  add_compile_definitions(GEMMSTONE_WITH_L0_RUNTIME=1)
  include_directories(${LevelZero_INCLUDE_DIRS})
else()
  message(WARNING "Level Zero library was not found, disabling support.")
endif()

if(MSVC)
    string(APPEND CMAKE_CXX_FLAGS " /Zc:__cplusplus")
endif()

# Set global compiler options.

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED True)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

link_libraries(NGen::NGen)

# pre-glibc-2.34 toolchains need explicit linking with libdl and libpthread.
link_libraries(${CMAKE_DL_LIBS})

find_package(Threads REQUIRED)
link_libraries(Threads::Threads)

add_compile_options(-DNGEN_NO_OP_NAMES=1)
add_compile_definitions(NGEN_SAFE=1)
add_compile_definitions(STANDALONE=1)

include_directories(. include $ENV{INCLUDE})

enable_testing()

# Recurse.
add_subdirectory(dsl)
add_subdirectory(examples)
add_subdirectory(generator)
add_subdirectory(generator_dsl)
add_subdirectory(selector)
add_subdirectory(tests)
add_subdirectory(tools)

# Copy scripts and suites into build directory.

file(COPY tools/matlab DESTINATION .)
file(COPY selector/db DESTINATION selector)

function(copy_scripts_by_type EXT)
    file(GLOB SCRIPT_LIST scripts/*.${EXT})
    foreach(SCRIPT ${SCRIPT_LIST})
        get_filename_component(SCRIPT_BASE ${SCRIPT} NAME)
        configure_file(${SCRIPT} ${SCRIPT_BASE} COPYONLY)
    endforeach()
endfunction()

if(WIN32)
    copy_scripts_by_type(bat)
else()
    copy_scripts_by_type(sh)
    copy_scripts_by_type(pl)
endif()
