#===============================================================================
# Copyright 2017 Intel Corporation
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

file(GLOB_RECURSE HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/*.hpp
    )
file(GLOB_RECURSE SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )
list(APPEND SOURCES ${TEST_THREAD})
if(ONEDNN_BUILD_GRAPH)
    add_definitions_with_host_compiler(-DBUILD_GRAPH)
else()
    file(GLOB_RECURSE GRAPH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/graph/*.cpp)
    list(REMOVE_ITEM SOURCES ${GRAPH_SRC})
endif()
include_directories_with_host_compiler(
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

if(BENCHDNN_USE_RDPMC)
    add_definitions_with_host_compiler(-DBENCHDNN_USE_RDPMC)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append_if(WIN32 CMAKE_CXX_FLAGS "-Qprec-div -Qprec-sqrt")
    append_if(UNIX  CMAKE_CXX_FLAGS "-prec-div -prec-sqrt -fp-model precise")
endif()

if(UNIX AND NOT APPLE AND NOT QNXNTO)
    find_library(LIBRT rt)
elseif(QNXNTO)
    find_library(LIBREGEX regex)
    find_library(LIBSOCKET socket)
endif()
register_exe(benchdnn "${SOURCES}" "" "${LIBRT};${LIBREGEX};${LIBSOCKET}")

file(COPY inputs DESTINATION .)

if(WIN32 AND (NOT DNNL_BUILD_FOR_CI))
    string(REPLACE  ";" "\;" PATH "${CTESTCONFIG_PATH};$ENV{PATH}")
    configure_file(
        "${PROJECT_SOURCE_DIR}/cmake/run_with_env.bat.in"
        "${PROJECT_BINARY_DIR}/run_with_env.bat"
    )
endif()

function(register_benchdnn_test engine driver test_file)
    if (ARGC GREATER 3)
        message(ERROR "Incorrect use of function")
    endif()

    set(test_mode "C")
    if(DNNL_TEST_SET_HAS_NO_CORR EQUAL 1)
        set(test_mode "R")
    elseif(DNNL_TEST_SET_HAS_ADD_BITWISE EQUAL 1)
        # Form a list to iterate over when registring tests
        set(test_mode "${test_mode};B")
    endif()

    # Parallel creation introduces race conditions inside native libraries.
    # Additionally, only needed for speeding up JIT compilation of OCL kernels.
    set(mode_modifier "")
    if(${engine} MATCHES "gpu"
       AND NOT ${DNNL_GPU_VENDOR} MATCHES "NVIDIA"
       AND NOT ${DNNL_GPU_VENDOR} MATCHES "AMD")
        set(mode_modifier "--mode-modifier=P")
    endif()

    foreach(tm ${test_mode})
        string(REPLACE "test_" "test_benchdnn_mode${tm}_" target_name ${test_file})
        set(cmd "--mode=${tm} ${mode_modifier} -v1 --engine=${engine} --${driver} --batch=${test_file}")
        set(benchdnn_target ${target_name}_${engine})

        if(DNNL_TEST_SET_HAS_GRAPH_EXE EQUAL 1)
            string(PREPEND cmd "--execution-mode=native_graph ")
        endif()

        if(NOT WIN32 OR DNNL_BUILD_FOR_CI)
            string(REPLACE " " ";" cmd "benchdnn ${cmd}")
            add_dnnl_test(${benchdnn_target} ${cmd})
        else()
            string(REPLACE " " ";" cmd "$<TARGET_FILE:benchdnn> ${cmd}")

            if(WIN32)
                set(cmd "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${cmd}")
                set(ARGV2 "cmd;/c;${PROJECT_BINARY_DIR}/run_with_env.bat;${ARGV2}")
            endif()

            add_custom_target(${benchdnn_target}
                COMMAND ${cmd}
                DEPENDS benchdnn
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )

            set_target_properties(${benchdnn_target} PROPERTIES
                EXCLUDE_FROM_DEFAULT_BUILD TRUE)
            maybe_configure_windows_test(${benchdnn_target} TARGET)

            # Create non-suffixed target for compatibility
            if (engine STREQUAL "cpu")
                add_custom_target(${target_name} DEPENDS ${benchdnn_target})
                maybe_configure_windows_test(${target_name} TARGET)
            endif()
        endif()
    endforeach()
endfunction()


set(stack_checker_pattern "^(bnorm|concat|conv|eltwise|ip|lrn|matmul|pool|reorder|softmax|sum)$")

function(register_all_tests engine driver test_files)
    if(DNNL_ENABLE_STACK_CHECKER AND NOT ${driver} MATCHES ${stack_checker_pattern})
        return()
    endif()

    foreach(test_file ${test_files})
        register_benchdnn_test(${engine} ${driver} ${test_file})
    endforeach()
endfunction()

# The following section is responsible to register test_benchdnn_ targets
# depending on DNNL_TEST_SET value.
set(has_gpu false)
if(NOT DNNL_GPU_RUNTIME STREQUAL "NONE")
    set(has_gpu true)
endif()
set(has_cpu false)
if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
    set(has_cpu true)
endif()

# Very sad CMake older then 3.2 does not support continue() command.
file(GLOB all_drivers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inputs inputs/*)

if(NOT ONEDNN_BUILD_GRAPH)
    list(REMOVE_ITEM all_drivers "graph")
endif()

foreach(driver ${all_drivers})
    set(driver_dir ${CMAKE_CURRENT_SOURCE_DIR}/inputs/${driver})
    # Collect input files in groups
    file(GLOB test_files_large_cpu RELATIVE ${driver_dir}
            inputs/${driver}/test_*_large$)
    file(GLOB test_files_gpu_ci RELATIVE ${driver_dir}
            inputs/${driver}/test_*_gpu_ci)
    file(GLOB test_files_gpu RELATIVE ${driver_dir} inputs/${driver}/test_*_gpu)
    file(GLOB test_files_ci RELATIVE ${driver_dir} inputs/${driver}/test_*_ci)
    file(GLOB test_files_smoke RELATIVE ${driver_dir} inputs/${driver}/test_*_smoke)
    file(GLOB test_files_weekly RELATIVE ${driver_dir} inputs/${driver}/test_*_weekly)
    file(GLOB test_files_cpu RELATIVE ${driver_dir} inputs/${driver}/test_*)
    set(test_files_cpu_ci "")
    if(driver STREQUAL "sdpa")
        list(APPEND test_files_cpu_ci "test_sdpa_cpu_ci")
        list(REMOVE_ITEM test_files_ci ${test_files_cpu_ci})
    endif()
    file(GLOB test_files_with_all RELATIVE ${driver_dir}
        inputs/brgemm/test_brgemm_all inputs/conv/test_conv_all
        inputs/graph/test_graph_all inputs/rnn/test_lstm_all)

    if(DNNL_TEST_SET_COVERAGE EQUAL DNNL_TEST_SET_SMOKE)
        if(has_gpu)
            register_all_tests(gpu "${driver}" "${test_files_smoke}")
        endif()
        if(has_cpu)
            register_all_tests(cpu "${driver}" "${test_files_smoke}")
        endif()
    elseif(DNNL_TEST_SET_COVERAGE EQUAL DNNL_TEST_SET_CI)
        if(NOT DNNL_EXPERIMENTAL_GROUPED_MEMORY)
            list(REMOVE_ITEM test_files_ci "test_matmul_grouped_ci")
        endif()

        # gpu_ci files may happen if standard coverage is not applicable on gpu.
        # Filter out gpu_ci inputs from ci
        foreach(test_file ${test_files_gpu_ci})
            string(REPLACE "${test_file}" "" test_files_ci "${test_files_ci}")
        endforeach()

        # use gpu_ci if not empty
        if(has_gpu)
            if(test_files_gpu_ci)
                register_all_tests(gpu "${driver}" "${test_files_gpu_ci}")
            else()
                register_all_tests(gpu "${driver}" "${test_files_ci}")
            endif()
        endif()
        if(has_cpu)
            register_all_tests(cpu "${driver}" "${test_files_ci}")
            register_all_tests(cpu "${driver}" "${test_files_cpu_ci}")
        endif()
    elseif(DNNL_TEST_SET_COVERAGE EQUAL DNNL_TEST_SET_NIGHTLY)
        if(NOT DNNL_EXPERIMENTAL_GROUPED_MEMORY)
            list(REMOVE_ITEM test_files_gpu "test_matmul_grouped_gpu")
            list(REMOVE_ITEM test_files_cpu "test_matmul_grouped_gpu")
            list(REMOVE_ITEM test_files_cpu "test_matmul_grouped")
        endif()

        ## Filter out gpu, large cpu, weekly and invalid inputs from cpu
        ##
        ## Note: put longer/specialized names first to keep filtering working.
        foreach(test_file ${test_files_large_cpu} ${test_files_gpu_ci}
            ${test_files_gpu} ${test_files_ci} ${test_files_cpu_ci}
            ${test_files_smoke}
            ${test_files_weekly})
            string(REPLACE "${test_file}" "" test_files_cpu "${test_files_cpu}")
        endforeach()
        ## Then filter files with "all" suffix by exact name since there may
        ## exist files like test_driver_all_something.
        foreach(test_file ${test_files_with_all})
            string(REPLACE "${test_file};" "" test_files_cpu "${test_files_cpu}")
        endforeach()

        if(has_cpu AND NOT DNNL_USE_CLANG_SANITIZER)
            register_all_tests(cpu "${driver}" "${test_files_large_cpu}")
        endif()
        if(has_gpu)
            register_all_tests(gpu "${driver}" "${test_files_gpu}")
        endif()
        if(has_cpu)
            register_all_tests(cpu "${driver}" "${test_files_cpu}")
        endif()
    elseif(DNNL_TEST_SET_COVERAGE EQUAL DNNL_TEST_SET_WEEKLY)
        # Weekly coverage registers only the dedicated `_weekly` inputs.
        if(has_gpu)
            register_all_tests(gpu "${driver}" "${test_files_weekly}")
        endif()
        if(has_cpu)
            register_all_tests(cpu "${driver}" "${test_files_weekly}")
        endif()
    endif()
endforeach()
