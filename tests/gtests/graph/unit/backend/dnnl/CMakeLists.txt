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

set(TEST_EXE test_graph_unit_dnnl)

# make gtest and library same definition for subgraph_visualizer_t
if(NOT ONEDNN_ENABLE_GRAPH_DUMP)
    add_definitions_with_host_compiler(-DDNNL_DISABLE_GRAPH_DUMP)
endif()

FILE(GLOB DNNL_COMMON_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dnnl_infer_shape_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fusion_info_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_insert_ops_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_internal_attrs_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_id_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_propagator_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logical_tensor_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_memory_planning_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_schema_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_partition_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_thread_local_cache_cpu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dnnl_utils_cpu.cpp
)

# op compile/execution need test on both CPU and GPU
FILE(GLOB DNNL_OP_EXECUTION_TEST_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/test_common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_constant_cache.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scratchpad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_subgraph_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_executable.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_batch_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_binary_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_concat.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_convtranspose.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_dequantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_eltwise.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_group_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_interpolate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_large_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layer_norm.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_matmul.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mqa_decomp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pool.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_prelu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_quantize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reduce.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_reorder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_sdp_decomp.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_softmax.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_typecast.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pass.cpp
)

# SDPA/MQA decompose kernel only support OMP and THREADPOOL runtime.
if(NOT (DNNL_CPU_RUNTIME STREQUAL "OMP" OR DNNL_CPU_RUNTIME STREQUAL "THREADPOOL"))
    list(REMOVE_ITEM DNNL_OP_EXECUTION_TEST_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/test_sdp_decomp.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/test_mqa_decomp.cpp"
    )
endif()

if(_ONEDNN_GRAPH_LAYOUT_DEBUG)
    file(GLOB FILES_REQUIRED_LAYOUT_DEBUG
        ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_id.cpp
    )
    set_source_files_properties(${FILES_REQUIRED_LAYOUT_DEBUG}
        PROPERTIES COMPILE_FLAGS "-DDNNL_GRAPH_LAYOUT_DEBUG")
endif()

set(OBJ_LIB graph_unit_test_dnnl_backend)
add_library(${OBJ_LIB} OBJECT ${DNNL_OP_EXECUTION_TEST_SOURCES} ${DNNL_COMMON_TEST_SOURCES})
set_property(GLOBAL APPEND PROPERTY GRAPH_UNIT_TEST_DEPS $<TARGET_OBJECTS:${OBJ_LIB}>)

foreach(TEST_FILE ${DNNL_COMMON_TEST_SOURCES})
    get_filename_component(file_name ${TEST_FILE} NAME_WE)
    string(REPLACE "_cpu" "" file_name ${file_name})
    string(REPLACE "test_" "test_graph_unit_dnnl_" test_suite_name ${file_name})
    register_graph_test_suite(${test_suite_name}_cpu "${file_name}*")
endforeach()

foreach(TEST_FILE ${DNNL_OP_EXECUTION_TEST_SOURCES})
    get_filename_component(file_name ${TEST_FILE} NAME_WE)
    string(REPLACE "test_" "test_graph_unit_dnnl_" test_suite_name ${file_name})
    register_graph_test_suite(${test_suite_name} "${file_name}*")
endforeach()
