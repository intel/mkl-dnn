#===============================================================================
# Copyright 2018 Intel Corporation
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

# Auxiliary build functions
#===============================================================================

if(utils_cmake_included)
    return()
endif()
set(utils_cmake_included true)
include("cmake/options.cmake")

# Common configuration for tests / test cases on Windows
function(maybe_configure_windows_test name kind)
    if(WIN32 AND (NOT DNNL_BUILD_FOR_CI))
        string(REPLACE  ";" "\;" PATH "${CTESTCONFIG_PATH};$ENV{PATH}")
        set_property(${kind} ${name} PROPERTY ENVIRONMENT "PATH=${PATH}")
        if(TARGET ${name} AND CMAKE_GENERATOR MATCHES "Visual Studio")
            configure_file(${PROJECT_SOURCE_DIR}/cmake/template.vcxproj.user
                ${name}.vcxproj.user @ONLY)
        endif()
    endif()
endfunction()

# Add test (aka add_test), but possibly appends an emulator
function(add_dnnl_test name command)
    add_test(${name} ${DNNL_TARGET_EMULATOR} ${command} ${ARGN})
endfunction()

# Register new executable/test
#   name -- name of the executable
#   srcs -- list of source, if many must be enclosed with ""
#   test -- "test" to mark executable as a test, "" otherwise
#   arg4 -- (optional) list of extra library dependencies
function(register_exe name srcs test)
    add_executable(${name} ${srcs})
    target_link_libraries(${name} ${LIB_PACKAGE_NAME} ${EXTRA_SHARED_LIBS} ${EXTRA_STATIC_LIBS} ${ARGV3})
    if("x${test}" STREQUAL "xtest")
        add_dnnl_test(${name} ${name})
        maybe_configure_windows_test(${name} TEST)
    else()
        maybe_configure_windows_test(${name} TARGET)
    endif()
endfunction()

# Append to a variable
#   var = var + value
macro(append var value)
    set(${var} "${${var}} ${value}")
endmacro()

# Set variable depending on condition:
#   var = cond ? val_if_true : val_if_false
macro(set_ternary var condition val_if_true val_if_false)
    if (${condition})
        set(${var} "${val_if_true}")
    else()
        set(${var} "${val_if_false}")
    endif()
endmacro()

# Conditionally set a variable
#   if (cond) var = value
macro(set_if condition var value)
    if (${condition})
        set(${var} "${value}")
    endif()
endmacro()

# Conditionally append
#   if (cond) var = var + value
macro(append_if condition var value)
    if (${condition})
        append(${var} "${value}")
    endif()
endmacro()

# Append options for sycl host compiler
macro(append_host_compiler_options var opts)
    if(NOT DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT")
        if(${var} MATCHES "-fsycl-host-compiler-options")
            if("${ARGV2}" STREQUAL "BEFORE") # prepend
                string(REGEX REPLACE
                    "(.*)(-fsycl-host-compiler-options=)\"([^\"]*)\"(.*)"
                    "\\1\\2\"${opts} \\3\"\\4" ${var} ${${var}})
            elseif("${ARGV2}" STREQUAL "") # append
                string(REGEX REPLACE
                    "(.*)(-fsycl-host-compiler-options=)\"([^\"]*)\"(.*)"
                    "\\1\\2\"\\3 ${opts}\"\\4" ${var} ${${var}})
            else()
                message(FATAL_ERROR "Unknown argument: ${ARGV2}")
            endif()
        else()
            append(${var} "-fsycl-host-compiler-options=\"${opts}\"")
        endif()
    endif()
endmacro()

macro(include_directories_with_host_compiler)
    foreach(inc_dir ${ARGV})
        include_directories(${inc_dir})
        append_host_compiler_options(CMAKE_CXX_FLAGS "-I${inc_dir}")
    endforeach()
endmacro()

macro(include_directories_with_host_compiler_before)
    foreach(inc_dir ${ARGV})
        include_directories(BEFORE ${inc_dir})
        append_host_compiler_options(CMAKE_CXX_FLAGS "-I${inc_dir}" BEFORE)
    endforeach()
endmacro()

macro(add_definitions_with_host_compiler)
    foreach(def ${ARGV})
        add_definitions(${def})
        append_host_compiler_options(CMAKE_CXX_FLAGS "${def}")
    endforeach()
endmacro()

macro(remove_definitions_with_host_compiler)
    foreach(def ${ARGV})
        remove_definitions(${def})
        if(NOT DNNL_DPCPP_HOST_COMPILER STREQUAL "DEFAULT"
            AND CMAKE_CXX_FLAGS MATCHES "-fsycl-host-compiler-options=\"([^\"]*)\"")
            set(_host_opts " ${CMAKE_MATCH_1} ")
            string(REPLACE " ${def} " " " _host_opts "${_host_opts}")
            string(STRIP "${_host_opts}" _host_opts)
            string(REGEX REPLACE
                "(-fsycl-host-compiler-options=)\"[^\"]*\""
                "\\1\"${_host_opts}\"" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        endif()
    endforeach()
endmacro()

# Append a path to path_list variable (Windows-only version)
macro(append_to_windows_path_list path_list path)
    file(TO_NATIVE_PATH "${path}" append_to_windows_path_list_tmp__)
    if(${path_list})
        set(${path_list}
            "${${path_list}};${append_to_windows_path_list_tmp__}")
    else()
        set(${path_list}
            "${append_to_windows_path_list_tmp__}")
    endif()
endmacro()

# Strip paths from libraries before populating INSTALL_INTERFACE
function(target_link_libraries_install target list)
    foreach(lib ${list})
        get_filename_component(base "${lib}" NAME)
        target_link_libraries(${target} PUBLIC "$<INSTALL_INTERFACE:${base}>")
    endforeach(lib)
endfunction()

function(find_libm var)
    if(UNIX)
        find_library(${var} m REQUIRED)
    endif()
endfunction()
