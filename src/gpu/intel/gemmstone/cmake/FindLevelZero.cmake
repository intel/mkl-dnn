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
# FindLevelZero
# ----------
#
# Finds Level Zero Loader
#
# IMPORTED Targets
# ^^^^^^^^^^^^^^^^
#
# This module defines :prop_tgt:`IMPORTED` target ``LevelZero::LevelZero``, if
# Level Zero has been found.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables::
#
#   LevelZero_FOUND          - True if Level Zero was found
#   LevelZero_INCLUDE_DIRS   - include directories for Level Zero
#   LevelZero_LIBRARIES      - link against this library to use Level Zero
#
# The module will also define two cache variables::
#
#   LevelZero_INCLUDE_DIR    - the Level Zero include directory
#   LevelZero_LIBRARY        - the path to the Level Zero library
#

# Prioritize L0ROOT
list(APPEND l0_root_hints
            ${L0ROOT}
            $ENV{L0ROOT})

set(original_cmake_prefix_path ${CMAKE_PREFIX_PATH})
if(l0_root_hints)
    list(INSERT CMAKE_PREFIX_PATH 0 ${l0_root_hints})
endif()

find_path(LevelZero_INCLUDE_DIR
  NAMES
    level_zero/ze_api.h
  HINTS
    ${l0_root_hints}
  PATH_SUFFIXES
    include/)

message(STATUS "Found Level Zero headers: ${LevelZero_INCLUDE_DIR}")

if(WIN32)
  find_library(LevelZero_LIBRARY
    NAMES ze_loader
    PATH_SUFFIXES
      lib
      lib/x86_64-linux-gnu)
else()
  find_library(LevelZero_LIBRARY
    NAMES ze_loader
    PATHS
    PATH_SUFFIXES
      lib)
endif()

set(LevelZero_LIBRARIES ${LevelZero_LIBRARY})
set(LevelZero_INCLUDE_DIRS ${LevelZero_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LevelZero
  FOUND_VAR LevelZero_FOUND
  REQUIRED_VARS LevelZero_LIBRARY LevelZero_INCLUDE_DIR)

mark_as_advanced(
  LevelZero_INCLUDE_DIR
  LevelZero_LIBRARY)

if(LevelZero_FOUND AND NOT TARGET LevelZero::LevelZero)
  add_library(LevelZero::LevelZero UNKNOWN IMPORTED)
  set_target_properties(LevelZero::LevelZero PROPERTIES
    IMPORTED_LOCATION "${LevelZero_LIBRARY}")
  set_target_properties(LevelZero::LevelZero PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LevelZero_INCLUDE_DIRS}")
endif()

# Reverting the CMAKE_PREFIX_PATH to its original state
set(CMAKE_PREFIX_PATH ${original_cmake_prefix_path})
