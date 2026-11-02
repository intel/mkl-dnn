/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

/// @example matmul_grouped.cpp
/// > Annotated version: @ref matmul_grouped_cpp

/// @page matmul_grouped_cpp_brief
/// @brief This C++ API example demonstrates how to execute grouped matrix-matrix
/// multiplication, using grouped encoding and [MatMul](@ref dev_guide_matmul) primitive,
/// that is a commonly used functionality in Mixture-of-Experts (MoE).

/// @page matmul_grouped_cpp MatMul with Grouped Encoding
/// \copybrief matmul_grouped_cpp_brief
///
/// Steps in this examples cover:
/// - How to create memory descriptors with grouped encoding
/// - Specifying variable-dimension groups with offsets array
/// - Using per-token (row-wise) src and per-expert-column (column-wise)
///   wei scales with f8_e4m3 data
/// - Executing matmul primitive
///
/// @include matmul_grouped.cpp

void grouped_matmul_example(engine::kind engine_kind) {
    // Create execution engine and stream for computation
    engine eng(engine_kind, 0);
    stream engine_stream(eng);

    // Sample token distribution across experts
    const memory::dim num_experts = 4; // Number of experts in the MoE model
    std::vector<int32_t> tokens_per_expert = {12, 8, 0, 10};

    // Build cumulative offsets (exclusive-end boundaries)
    // offsets[i] = total tokens up to and including expert i
    std::vector<int32_t> offsets(num_experts);
    offsets[0] = tokens_per_expert[0];
    for (memory::dim i = 1; i < num_experts; ++i) {
        offsets[i] = offsets[i - 1] + tokens_per_expert[i];
    }

    // Total tokens number across all experts
    memory::dim total_tokens = std::accumulate(
            tokens_per_expert.begin(), tokens_per_expert.end(), memory::dim(0));

    std::cout << "Number of experts: " << num_experts << std::endl;

    std::cout << "Token distribution: " << total_tokens << " total tokens";
    std::cout << " routed to " << num_experts << " experts";
    std::cout << " (";
    for (memory::dim i = 0; i < num_experts; ++i) {
        std::cout << tokens_per_expert[i];
        if (i < num_experts - 1) std::cout << ", ";
    }
    std::cout << " tokens per expert)" << std::endl;

    // src is [total_tokens, K] with grouped encoding
    // wei is [num_experts, K, N] with standard 3D format
    // dst is [total_tokens, N] with grouped encoding
    const memory::dim K = 64; // Input feature dimension
    const memory::dim N = 128; // Output feature dimension

    std::cout << "Input dimensions: K=" << K << " (features), N=" << N
              << " (outputs)" << std::endl;
    std::cout << "Weights: [" << num_experts << ", " << K << ", " << N
              << "] tensor in acb format (experts × output_dim × input_dim)"
              << std::endl;
    std::cout << std::endl;

    // FP8 row-wise recipe: f8_e4m3 src/wei, bf16 dst, f32 scales
    // src and wei are filled with raw bytes for simplicity
    std::vector<uint8_t> src_data(total_tokens * K);
    for (int i = 0; i < total_tokens * K; ++i)
        src_data[i] = static_cast<uint8_t>(i % 128);

    std::vector<uint8_t> weights_data(num_experts * N * K);
    for (int i = 0; i < num_experts * N * K; ++i)
        weights_data[i] = static_cast<uint8_t>(i % 128);

    std::vector<uint16_t> dst_data(total_tokens * N, 0);

    // Create memory descriptors with grouped encoding
    // variable_dim_idx=0 indicates M dimension varies per group
    memory::dims src_dims = {total_tokens, K};
    memory::dims weights_dims = {num_experts, K, N};
    memory::dims dst_dims = {total_tokens, N};

    auto src_md = memory::desc::grouped(
            src_dims, memory::data_type::f8_e4m3, 0, num_experts);
    auto dst_md = memory::desc::grouped(
            dst_dims, memory::data_type::bf16, 0, num_experts);
    auto weights_md = memory::desc(
            weights_dims, memory::data_type::f8_e4m3, memory::format_tag::acb);

    // Create memory objects
    // Grouped memory has 2 buffers:
    //     - buffer 0: concatenated data values
    //     - buffer 1: cumulative offsets array
    auto src_mem = memory(src_md, eng);
    auto dst_mem = memory(dst_md, eng);
    auto weights_mem = memory(weights_md, eng);

    // Write data to buffer 0 (data values)
    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), weights_mem);

    // Write offsets to buffer 1 (offsets buffer)
    // Both src and dst must use identical offsets since token distribution
    // is the same for input and output (each expert processes the same tokens)
    write_to_dnnl_memory(offsets.data(), src_mem, 1);
    write_to_dnnl_memory(offsets.data(), dst_mem, 1);

    // Create primitive attributes with scales
    primitive_attr matmul_attr;

    // Row-wise (per-token) src scales: one f32 scale per token.
    // The buffer is concatenated across experts, mirroring how src tokens
    // are concatenated in the grouped src memory.
    std::vector<float> src_scales(total_tokens);
    for (int32_t i = 0; i < total_tokens; ++i)
        src_scales[i] = 1.0f + (i % 100) / 500.0f;

    memory::desc src_scales_md(
            {total_tokens}, memory::data_type::f32, memory::format_tag::a);
    auto src_scales_mem = memory(src_scales_md, eng);
    write_to_dnnl_memory(src_scales.data(), src_scales_mem);
    matmul_attr.set_scales_mask(DNNL_ARG_SRC, (1 << 0));

    // Column-wise wei scales: per-expert and per-column
    std::vector<float> wei_scales(num_experts * N);
    for (int32_t i = 0; i < num_experts * N; ++i)
        wei_scales[i] = 0.9f + (i % 200) / 1000.0f;

    memory::desc wei_scales_md(
            {num_experts, N}, memory::data_type::f32, memory::format_tag::ab);
    auto wei_scales_mem = memory(wei_scales_md, eng);
    write_to_dnnl_memory(wei_scales.data(), wei_scales_mem);
    matmul_attr.set_scales_mask(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 2));

    matmul matmul_prim;
    // Create matmul primitive descriptor and the primitive if it's supported
    // on this platform.
    try {
        auto matmul_pd = matmul::primitive_desc(
                eng, src_md, weights_md, dst_md, matmul_attr);
        matmul_prim = matmul(matmul_pd);
    } catch (error &e) {
        if (e.status == dnnl_unimplemented
                || e.status == dnnl_unsupported_data_type)
            throw example_allows_unimplemented {
                    "No f8 implementation is available for this platform.\n"
                    "Please refer to the developer guide for details."};

        // on any other error just re-throw
        throw;
    }

    // Execute the primitive
    matmul_prim.execute(engine_stream,
            {{DNNL_ARG_SRC, src_mem}, {DNNL_ARG_WEIGHTS, weights_mem},
                    {DNNL_ARG_DST, dst_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, src_scales_mem},
                    {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wei_scales_mem}});

    // Wait for completion
    engine_stream.wait();
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    return handle_example_errors(grouped_matmul_example, engine_kind);
}
