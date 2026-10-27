#===============================================================================
# Copyright 2017 Intel Corporation
# Copyright 2021 FUJITSU LIMITED
# Copyright 2026 Arm Ltd. and affiliates
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

# Manage secure Development Lifecycle-related compiler flags
#===============================================================================

if(SDL_cmake_included)
    return()
endif()
set(SDL_cmake_included true)
include("cmake/utils.cmake")

# The flags that can be used for the main and host compilers should be moved to
# the macros to avoid code duplication and ensure consistency.
macro(sdl_unix_common_ccxx_flags var)
    append(${var} "-fPIC -Wformat -Wformat-security")
endmacro()

macro(sdl_gnu_common_ccxx_flags var gnu_version)
    append(${var} "-fstack-protector-strong")
    if(NOT (${gnu_version} VERSION_LESS 8.0) 
       AND (DNNL_TARGET_ARCH STREQUAL "X64"))
        append(${var} "-fcf-protection=full")
    endif()
endmacro()

# GCC might be very paranoid for partial structure initialization, e.g.
#   struct { int a, b; } s = { 0, };
# However the behavior is triggered by `Wmissing-field-initializers`
# only. To prevent warnings on users' side who use the library and turn
# this warning on, let's use it too. Applicable for the library sources
# and interfaces only (tests currently rely on that fact heavily)
macro(sdl_unix_src_ccxx_flags var)
    append(${var} "-Wmissing-field-initializers")
endmacro()

macro(sdl_unix_example_ccxx_flags var)
    # At this point the flags for src and examples are the same
    sdl_unix_src_ccxx_flags(${var})
endmacro()

macro(sdl_msvc_common_ccxx_flags var)
    append(${var} "/EHsr /GS /Gy /guard:cf /DYNAMICBASE /sdl")
endmacro()

set(ONEDNN_SDL_COMPILER_FLAGS)
set(ONEDNN_SDL_LINKER_FLAGS)

if(UNIX)
    sdl_unix_common_ccxx_flags(ONEDNN_SDL_COMPILER_FLAGS)
    sdl_unix_src_ccxx_flags(CMAKE_SRC_CCXX_FLAGS)
    sdl_unix_example_ccxx_flags(CMAKE_EXAMPLE_CCXX_FLAGS)
    if(UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
        append(ONEDNN_SDL_COMPILER_FLAGS "-D_FORTIFY_SOURCE=2")
    endif()
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        sdl_gnu_common_ccxx_flags(ONEDNN_SDL_COMPILER_FLAGS
                                  CMAKE_CXX_COMPILER_VERSION)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
        get_filename_component(CXX_CMD_NAME ${CMAKE_CXX_COMPILER} NAME)
        # Fujitsu CXX compiler does not document/guarantee operation of stack protector flags
        if(NOT CXX_CMD_NAME STREQUAL "FCC")
            append(ONEDNN_SDL_COMPILER_FLAGS "-fstack-protector-strong")
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        append(ONEDNN_SDL_COMPILER_FLAGS "-fstack-protector")
    endif()
    if(APPLE)
        append(ONEDNN_SDL_LINKER_FLAGS "-Wl,-bind_at_load")
    else()
        # Only applies to executables.
        append(CMAKE_EXE_LINKER_FLAGS "-pie")
        append(ONEDNN_SDL_LINKER_FLAGS "-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    endif()
elseif(WIN32)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        sdl_msvc_common_ccxx_flags(ONEDNN_SDL_COMPILER_FLAGS)
        append(ONEDNN_SDL_LINKER_FLAGS "/NXCOMPAT")
        if(NOT UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
            append(ONEDNN_SDL_LINKER_FLAGS "/LTCG /INCREMENTAL:NO")
        endif()
    elseif(CMAKE_BASE_NAME STREQUAL "icx")
        append(ONEDNN_SDL_COMPILER_FLAGS "/EHs /GS /Gy /guard:cf /Wformat /Wformat-security")
        append(ONEDNN_SDL_LINKER_FLAGS "/Qoption,link,/NXCOMPAT")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        append(ONEDNN_SDL_COMPILER_FLAGS "-Wformat -Wformat-security")
        if(UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
            append(ONEDNN_SDL_COMPILER_FLAGS "-D_FORTIFY_SOURCE=2")
        endif()
        get_filename_component(CXX_CMD_NAME ${CMAKE_CXX_COMPILER} NAME)
        # Fujitsu CXX compiler does not document/guarantee operation of stack protector flags
        if(NOT CXX_CMD_NAME STREQUAL "FCC")
            append(ONEDNN_SDL_COMPILER_FLAGS "-fstack-protector-strong")
        endif()
        append(ONEDNN_SDL_LINKER_FLAGS "-Xlinker /NXCOMPAT")
        if(NOT UPPERCASE_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
            append(ONEDNN_SDL_LINKER_FLAGS "-Xlinker /LTCG -Xlinker /INCREMENTAL:NO")
        endif()
    endif()

    if(NOT MINGW)
        # For a Windows build, a malicious DLL can be injected because of the
        # uncontrolled search order for load-time linked libraries defined for a
        # Windows setting. The following cmake flags change the search order so that
        # DLLs are loaded from the current working directory only if it is under a path
        # in the Safe Load List.
        if(CMAKE_BASE_NAME STREQUAL "icx")
            # add ICX-style linker flags
            append(ONEDNN_SDL_LINKER_FLAGS "/Qoption,link,/DEPENDENTLOADFLAG:0x2000")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            # add Clang-style linker flags
            append(ONEDNN_SDL_LINKER_FLAGS "-Xlinker /DEPENDENTLOADFLAG:0x2000")
        else()
            # Default to MSVC-style definition
            append(ONEDNN_SDL_LINKER_FLAGS "/DEPENDENTLOADFLAG:0x2000")
        endif()
    endif()
endif()

append(CMAKE_C_FLAGS "${ONEDNN_SDL_COMPILER_FLAGS}")
append(CMAKE_CXX_FLAGS "${ONEDNN_SDL_COMPILER_FLAGS}")
append(CMAKE_SHARED_LINKER_FLAGS "${ONEDNN_SDL_LINKER_FLAGS}")
append(CMAKE_EXE_LINKER_FLAGS "${ONEDNN_SDL_LINKER_FLAGS}")
