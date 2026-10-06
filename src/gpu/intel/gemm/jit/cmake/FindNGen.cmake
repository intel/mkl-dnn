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

#.rst:
# FindNGen
# ----------
#
# Finds the nGEN library
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``NGen::NGen``, if
# nGEN has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables::
#
#   NGEN_FOUND          - True if nGEN was found
#   NGEN_INCLUDE_DIRS   - include directories for nGEN
#
# The module will also define the following cache variables::
#
#   NGEN_INCLUDE_DIR    - the nGEN include directory
#

# Prioritize NGEN_ROOT
list(APPEND ngen_root_hints
            ${NGEN_ROOT}
            $ENV{NGEN_ROOT}
            ${CMAKE_SOURCE_DIR}/..)

# oneDNN's bundled nGEN, for builds from src/gpu/intel/gemm/jit
get_filename_component(dnnl_root ${CMAKE_SOURCE_DIR}/../../../../.. ABSOLUTE)
if(EXISTS ${dnnl_root}/third_party/ngen/ngen.hpp)
    list(APPEND ngen_root_hints ${dnnl_root}/third_party)
endif()

set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(ngen_root_hints)
    list(INSERT CMAKE_PREFIX_PATH 0 ${ngen_root_hints})
endif()

find_path(NGEN_INCLUDE_DIR
  NAMES
    ngen.hpp
  HINTS
    ${ngen_root_hints}
  PATH_SUFFIXES
    ngen)

message(STATUS "Found nGEN headers: ${NGEN_INCLUDE_DIR}")

set(NGEN_INCLUDE_DIRS ${NGEN_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NGen
  FOUND_VAR NGEN_FOUND
  REQUIRED_VARS NGEN_INCLUDE_DIR)

mark_as_advanced(
  NGEN_INCLUDE_DIR)

if(NGEN_FOUND AND NOT TARGET NGen::NGen)
  add_library(NGen::NGen INTERFACE IMPORTED)
  set_target_properties(NGen::NGen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NGEN_INCLUDE_DIRS}")
endif()

# Reverting the CMAKE_PREFIX_PATH to its original state
set(CMAKE_PREFIX_PATH ${original_cmake_prefix_path})
