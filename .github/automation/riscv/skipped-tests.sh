#!/usr/bin/env bash

# *******************************************************************************
# Copyright 2025 Arm Limited and affiliates.
# Copyright 2025 Intel Corporation
# Copyright 2026 Institute of Software, Chinese Academy of Sciences
# SPDX-License-Identifier: Apache-2.0
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
# *******************************************************************************

# Tests to skip for RISC-V architecture.

set -eo pipefail

# Skip the SDPA example that exceeds the CI time budget under QEMU.
SKIPPED_TESTS="cpu-graph-sdpa-cpp"

if [[ "$ONEDNN_TEST_SET" == "SMOKE" ]]; then
    # Skip time-consuming tests (QEMU is slow)
    SKIPPED_TESTS+="|cpu-cnn-training-f32-cpp"
    SKIPPED_TESTS+="|cpu-cnn-inference-f32-cpp"
    SKIPPED_TESTS+="|cpu-cnn-training-f32-c"
    SKIPPED_TESTS+="|cpu-graph-gated-mlp-int4-cpp"
    SKIPPED_TESTS+="|cpu-performance-profiling-cpp"
    SKIPPED_TESTS+="|cpu-rnn-training-f32-cpp"
    SKIPPED_TESTS+="|test_convolution_backward_data_f32"
    SKIPPED_TESTS+="|test_convolution_backward_weights_f32"
    SKIPPED_TESTS+="|test_convolution_eltwise_forward_f32"
    SKIPPED_TESTS+="|test_convolution_eltwise_forward_x8s8f32s32"
    SKIPPED_TESTS+="|test_convolution_forward_f32"
    SKIPPED_TESTS+="|test_pooling_backward"
    SKIPPED_TESTS+="|test_pooling_forward"
    SKIPPED_TESTS+="|test_gemm_bf16bf16f32"
    SKIPPED_TESTS+="|test_gemm_f32"
    SKIPPED_TESTS+="|test_gemm_s8s8s32"
    SKIPPED_TESTS+="|test_gemm_u8s8s32"
    SKIPPED_TESTS+="|test_graph_unit_dnnl_mqa_decomp_cpu"
    SKIPPED_TESTS+="|test_graph_unit_dnnl_sdp_decomp_cpu"
elif [[ "$ONEDNN_TEST_SET" == "CI" ]]; then
    SKIPPED_TESTS+="|test_benchdnn_modeC_graph_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_matmul_ci_cpu"
    SKIPPED_TESTS+="|test_benchdnn_modeC_ip_ci_cpu"
fi

echo "$SKIPPED_TESTS"
