#===============================================================================
# Copyright 2021 Intel Corporation
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
set(gpu_rt_pattern "(SYCL|DPCPP|OCL)")
set(sycl_rt_pattern "(SYCL|DPCPP)")

function(register_graph_api_test_suite test_suite_name exe_name)
    if(${test_suite_name} MATCHES "_cpu")
        if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
            add_test(${test_suite_name} ${exe_name} --engine=cpu)
            maybe_configure_windows_test(${test_suite_name} TEST)
        endif()
    elseif(${test_suite_name} MATCHES "_gpu")
        if(DNNL_GPU_RUNTIME MATCHES ${gpu_rt_pattern})
            add_test(${test_suite_name} ${exe_name} --engine=gpu)
            maybe_configure_windows_test(${test_suite_name} TEST)
        endif()
    else()
        if(DNNL_GPU_RUNTIME MATCHES ${gpu_rt_pattern})
            add_test(${test_suite_name}_gpu ${exe_name} --engine=gpu)
            maybe_configure_windows_test(${test_suite_name}_gpu TEST)
        endif()

        if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE")
            add_test(${test_suite_name}_cpu ${exe_name} --engine=cpu)
            maybe_configure_windows_test(${test_suite_name}_cpu TEST)
        endif()
    endif()
endfunction()

FILE(GLOB COMMON_API_TEST_DEPS
    ${CMAKE_CURRENT_SOURCE_DIR}/api_test_main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_api_common.cpp
    ${TEST_ALLOCATOR}
    ${TEST_THREAD}
)

# engine independent test
FILE(GLOB API_TEST_ENGINE_INDEPENDENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_add_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_constant_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_filter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_graph_dump.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_constant_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_engine.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_graph_dump.cpp
)

FILE(GLOB API_TEST_ENGINE_DEPENDENT_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_compile_parametrized.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_c_api_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_cpp_api_concurrent_execution.cpp
)

foreach(TEST_FILE ${API_TEST_ENGINE_INDEPENDENT_SOURCES})
    get_filename_component(exe_name ${TEST_FILE} NAME_WE)
    string(REPLACE "test_" "test_graph_" exe_name ${exe_name})
    add_executable(${exe_name} ${TEST_FILE} ${COMMON_API_TEST_DEPS})
    target_link_libraries(${exe_name}
        dnnl_gtest
        ${DNNL_LIBRARY_NAME}
        ${EXTRA_SHARED_LIBS}
    )
    register_graph_api_test_suite(${exe_name} ${exe_name})
endforeach()

foreach(TEST_FILE ${API_TEST_ENGINE_DEPENDENT_SOURCES})
    get_filename_component(exe_name ${TEST_FILE} NAME_WE)
    string(REPLACE "test_" "test_graph_" exe_name ${exe_name})
    add_executable(${exe_name} ${TEST_FILE} ${COMMON_API_TEST_DEPS})
    target_link_libraries(${exe_name}
        dnnl_gtest
        ${DNNL_LIBRARY_NAME}
        ${EXTRA_SHARED_LIBS}
    )
    register_graph_api_test_suite(${exe_name} ${exe_name})
endforeach()

if(DNNL_CPU_RUNTIME MATCHES ${sycl_rt_pattern} OR DNNL_GPU_RUNTIME MATCHES ${sycl_rt_pattern})
    add_subdirectory(sycl)
endif()

if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    add_subdirectory(ocl)
endif()
