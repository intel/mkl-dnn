#===============================================================================
# Copyright 2023 KNS Group LLC (YADRO)
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

file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]
    ${CMAKE_CURRENT_SOURCE_DIR}/*.[ch]pp
)

# Build sources into an object library
set(OBJ_LIB ${DNNL_LIBRARY_NAME}_cpu_riscv)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

# Add compiled object files to oneDNN dependencies
set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
             $<TARGET_OBJECTS:${OBJ_LIB}>)

# Add xbyak_riscv include path for JIT code generation
target_include_directories(${OBJ_LIB} PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party
)

# rvjit is a header-only library vendored in third_party/rvjit. It emits
# vector code through xbyak_riscv, and reports failures through oneDNN's
# debuginfo log in developer mode.
target_include_directories(${OBJ_LIB} PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/rvjit
)
target_compile_definitions(${OBJ_LIB} PRIVATE
    XBYAK_RISCV_V=1
    RVJIT_VERBOSE_HEADER="cpu/rv64/rvjit/rvjit_verbose.hpp"
    $<$<BOOL:${DNNL_DEV_MODE}>:RVJIT_DEBUG>
)
