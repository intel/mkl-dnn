#===============================================================================
# Copyright 2016 Intel Corporation
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

if (NOT DNNL_BUILD_EXAMPLES OR DNNL_ENABLE_STACK_CHECKER)
    return()
endif()

# propagate EXAMPLE specific flags
append(CMAKE_C_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_EXAMPLE_CCXX_FLAGS}")

# Link the examples to the OpenCL runtime
if(DNNL_GPU_RUNTIME STREQUAL "OCL")
    list(APPEND EXTRA_SHARED_LIBS OpenCL::OpenCL)
endif()

if(DNNL_WITH_SYCL)
    if(DNNL_SYCL_GENERIC)
        CHECK_CXX_COMPILER_FLAG("-fsycl -fsycl-targets=nvptx64-nvidia-cuda" NVIDIA_TARGET_SUPPORTED)
    endif()

    # Enable linking SYCL kernels.
    if(DNNL_SYCL_CUDA OR (DNNL_SYCL_GENERIC AND NVIDIA_TARGET_SUPPORTED))
        append(CMAKE_CXX_FLAGS "-fsycl-targets=nvptx64-nvidia-cuda")
        append(CMAKE_CXX_FLAGS "-Wno-linker-warnings")
    endif()

    if(DNNL_AMD_ENABLE_SYCL_KERNELS)
        append(CMAKE_CXX_FLAGS "-fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=${DNNL_AMD_SYCL_KERNELS_TARGET_ARCH}")
    endif()
endif()

# propagate sanitizer flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_SANITIZER_FLAGS}")

# propagate nowarn flags
append(CMAKE_C_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")
append(CMAKE_CXX_FLAGS "${CMAKE_CCXX_NOWARN_FLAGS}")

include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/include)
include_directories_with_host_compiler(${PROJECT_SOURCE_DIR}/examples)

append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_EXAMPLE_CXX_FLAGS}")
append_host_compiler_options(CMAKE_CXX_FLAGS "${DPCPP_CXX_NOWARN_FLAGS}")

file(GLOB_RECURSE sources *.cpp *.c)
file(GLOB_RECURSE headers *.hpp *.h)

if(NOT DNNL_EXPERIMENTAL_UKERNEL)
    list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/ukernels/cpu_brgemm.cpp)
endif()

if(NOT DNNL_EXPERIMENTAL_GROUPED_MEMORY)
    list(REMOVE_ITEM sources
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_grouped.cpp)
endif()

# Remove tests for CUDA which use unimplemented primitives
if(DNNL_SYCL_CUDA)
    list(REMOVE_ITEM sources
        ${CMAKE_CURRENT_SOURCE_DIR}/bnorm_u8_via_binary_postops.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rnn_training_f32.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tutorials/matmul/inference_int8_matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/binary.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/lstm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/layer_normalization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/reorder.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/shuffle.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/group_normalization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/vanilla_rnn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/lbr_gru.cpp)
endif()

# Remove examples for Graph API if graph component is not enabled
if(NOT ONEDNN_BUILD_GRAPH)
    list(REMOVE_ITEM sources
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/cpu_getting_started.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/sycl_getting_started.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/cpu_inference_int8.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/cpu_single_op_partition.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/sycl_single_op_partition.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/gpu_opencl_getting_started.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/sdpa.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/mqa.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/sdpa_stacked_qkv.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/gqa.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/gated_mlp.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/gated_mlp_wei_combined.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/gated_mlp_int4.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/sdpa_bottom_right_causal_mask.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/gqa_training.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/sdpa_quantized.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/sdpa_dynamic_quantized.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/graph/gqa_dynamic_quantized.cpp
        )
endif()

if(DNNL_SYCL_GENERIC)
    list(REMOVE_ITEM sources
        # XXX: Enable when InnerProduct is implemented
        ${CMAKE_CURRENT_SOURCE_DIR}/cnn_inference_f32.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/inner_product.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/rnn_training_f32.cpp
        # XXX: Enable when Reduction is implemented
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/reduction.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/group_normalization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/lbr_gru.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/lstm.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/vanilla_rnn.cpp)
endif()

if(DNNL_SYCL_HIP)
    # Build examples for supported primitives that support required features.
    set(sources)
    list(APPEND sources
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/softmax.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/lrn.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/eltwise.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/reduction.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/matmul.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/matmul_perf.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/primitives/inner_product.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cnn_inference_f32.c
        ${CMAKE_CURRENT_SOURCE_DIR}/cnn_inference_int8.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/cnn_training_bf16.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/getting_started.cpp)
endif()

# Skip SYCL, GPU and cross-engine examples
foreach(f ${sources})
    get_filename_component(f_name ${f} NAME)
    if(DNNL_GPU_RUNTIME STREQUAL "NONE" AND ${f_name} MATCHES "^(cross_engine|gpu)")
        list(REMOVE_ITEM sources "${f}")
    endif()
    if(NOT DNNL_WITH_SYCL AND ${f_name} MATCHES "^sycl")
        list(REMOVE_ITEM sources "${f}")
    endif()
endforeach()

# In case of SYCL, skip CPU examples that directly work with raw pointers
if(DNNL_CPU_SYCL)
    foreach(f ${sources})
        get_filename_component(fname ${f} NAME)
        if(${fname} MATCHES "cpu_")
            list(REMOVE_ITEM sources "${f}")
        endif()
    endforeach()
endif()

# Do not build C examples for TBB threading runtime because
# TBB doesn't provide C API to do explicit finalization.
if (DNNL_CPU_RUNTIME STREQUAL "TBB" OR DNNL_CPU_SYCL)
    foreach(f ${sources})
        get_filename_component(fname ${f} NAME)
        if(${fname} MATCHES ".*\\.c$")
            list(REMOVE_ITEM sources "${f}")
        endif()
    endforeach()
endif()

# Do not build RNN examples for threadpool as it doesn't support asynchronous
# version of it which can't be distinguished.
if (DNNL_CPU_RUNTIME STREQUAL "THREADPOOL")
    foreach(f ${sources})
        get_filename_component(fname ${f} NAME)
        if(${fname} MATCHES "rnn_")
            list(REMOVE_ITEM sources "${f}")
        endif()
    endforeach()
endif()

foreach(src ${sources})
    file(RELATIVE_PATH src_rel_path ${CMAKE_CURRENT_SOURCE_DIR} ${src})
    string(REGEX REPLACE "[/_\\.]" "-" example_name ${src_rel_path})

    # Put hw-specific part of the name in front.
    # It is important for examples in subdirectories.
    foreach(pat "cpu-" "gpu-" "cross-engine-")
        string(REGEX REPLACE "^(.*)${pat}" "${pat}\\1"
            example_name ${example_name})
    endforeach()

    if(${example_name} MATCHES "(cross-engine|cpu|gpu)-")
        if(NOT DNNL_CPU_RUNTIME STREQUAL "NONE" OR ${example_name} MATCHES "gpu-")
            # Example name contains cross-engine, cpu or gpu
            find_libm(LIBM)
            if(NOT ${example_name} MATCHES ".*opencl" OR DNNL_GPU_RUNTIME STREQUAL "OCL")
                register_exe(${example_name} ${src} "test" ${LIBM})
            endif()
        endif()
    else()
        set(cpu_rt_pattern "(SEQ|OMP|TBB|SYCL|DPCPP)")
        set(gpu_rt_pattern "(OCL|ZE|SYCL|DPCPP)")
        if(${example_name} MATCHES "sycl.*")
            set(cpu_rt_pattern "(SYCL|DPCPP)")
            set(gpu_rt_pattern "(SYCL|DPCPP)")
        endif()
        if(DNNL_CPU_RUNTIME MATCHES ${cpu_rt_pattern})
            # Adding test for CPU
            add_dnnl_test("cpu-${example_name}" "${example_name}" cpu)
            maybe_configure_windows_test("cpu-${example_name}" TEST)
        endif()
        if(DNNL_GPU_RUNTIME MATCHES ${gpu_rt_pattern})
            # Adding test for GPU
            add_dnnl_test("gpu-${example_name}" "${example_name}" gpu)
            maybe_configure_windows_test("gpu-${example_name}" TEST)
        endif()
        find_libm(LIBM)
        register_exe(${example_name} ${src} "" ${LIBM})
    endif()
endforeach()

if (DNNL_INSTALL_MODE STREQUAL "BUNDLE")
    set(BUNDLE_EXAMPLES_DIR "${CMAKE_INSTALL_DATAROOTDIR}/doc/${LIB_PACKAGE_NAME}/examples")

    configure_file(CMakeLists.txt.in CMakeLists.txt @ONLY)
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt
        ${sources} ${headers}
        DESTINATION ${BUNDLE_EXAMPLES_DIR})

    if(DNNL_WITH_SYCL)
        install(FILES
            ${PROJECT_SOURCE_DIR}/cmake/dpcpp_driver_check.cmake
            DESTINATION ${BUNDLE_EXAMPLES_DIR})
    else()
    # Skip matmul examples with SYCL
        install(DIRECTORY
            tutorials
            DESTINATION ${BUNDLE_EXAMPLES_DIR})
    endif()

    if(WIN32)
        install(FILES
            ${PROJECT_SOURCE_DIR}/cmake/template.vcxproj.user
            DESTINATION ${BUNDLE_EXAMPLES_DIR})
    endif()
endif()
