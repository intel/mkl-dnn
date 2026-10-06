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

# generator.cpp instantiates Generator<HW> and must be compiled once per
# (GPU architecture, output format) pair; everything else here is common.
file(GLOB MICROKERNEL_SOURCES microkernel/*.cpp)
add_library(microkernel OBJECT ${MICROKERNEL_SOURCES})

file(GLOB GENERATOR_SOURCES *.cpp pieces/*.cpp)
list(FILTER GENERATOR_SOURCES EXCLUDE REGEX ".*(generator|ngen_elf|microkernel_selector)\.cpp")

add_library(generator STATIC ${GENERATOR_SOURCES})
target_link_libraries(generator PRIVATE microkernel)

# Binary output only: linked into the bin generator.
add_library(microkernel_selector OBJECT microkernel_selector.cpp)
target_compile_definitions(microkernel_selector PRIVATE FORMAT_ABBREV_b)

foreach(GPU ${GPUS})
    set(NGEN_ELF_LIB ngen_elf_${GPU})
    add_library(${NGEN_ELF_LIB} OBJECT ngen_elf.cpp)
    target_compile_definitions(${NGEN_ELF_LIB} PRIVATE INSTANTIATE_HW_${GPU})
    set(NGEN_ELF_LIB_${GPU} ${NGEN_ELF_LIB})
endforeach()

function(gemmstone_add_generator_library format name)
    set(GENERATOR_LIB gemmstone_generator_${name})

    set(GPU_OBJECTS "")
    foreach(GPU ${GPUS})
        set(GPU_LIB ${GENERATOR_LIB}_${GPU})
        add_library(${GPU_LIB} OBJECT generator.cpp)
        target_compile_definitions(${GPU_LIB} PRIVATE INSTANTIATE_HW_${GPU} FORMAT_ABBREV_${format})
        list(APPEND GPU_OBJECTS $<TARGET_OBJECTS:${GPU_LIB}>)

        if(NOT format STREQUAL "")
            list(APPEND GPU_OBJECTS $<TARGET_OBJECTS:${NGEN_ELF_LIB_${GPU}}>)
        endif()
    endforeach()

    add_library(${GENERATOR_LIB} STATIC $<TARGET_OBJECTS:generator> ${GPU_OBJECTS})
    target_link_libraries(${GENERATOR_LIB} PUBLIC gemmstone_dsl generator_dsl)
    if(${name} STREQUAL "bin")
      target_sources(${GENERATOR_LIB} PRIVATE $<TARGET_OBJECTS:microkernel_selector>)
      # The binary microkernel selector uses the catalog selection implementation.
      target_link_libraries(${GENERATOR_LIB} PRIVATE catalog_tools)
      target_link_libraries(${GENERATOR_LIB} PRIVATE microkernel)
    endif()
endfunction()

gemmstone_add_generator_library(""  asm)   # ASM output    -> gemmstone_generator_asm
gemmstone_add_generator_library("b" bin)   # binary output -> gemmstone_generator_bin
if(LevelZero_FOUND)
    gemmstone_add_generator_library("e" ze)    # ZEBIN output  -> gemmstone_generator_ze
endif()
if(OpenCL_FOUND)
    gemmstone_add_generator_library("o" ocl)   # OpenCL output -> gemmstone_generator_ocl
endif()
