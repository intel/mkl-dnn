/*******************************************************************************
* Copyright 2023 Intel Corporation
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

/// @example cpu_matmul_weights_compression.cpp
/// > Annotated version: @ref cpu_matmul_weights_compression_cpp

/// @page cpu_matmul_weights_compression_cpp_brief
/// @brief This C++ API example demonstrates how to create and execute a
/// [MatMul](@ref dev_guide_matmul) primitive that uses a weights tensor
/// encoded with the packed sparse encoding.

/// @page cpu_matmul_weights_compression_cpp MatMul with Packed Sparse Weights
/// \copybrief cpu_matmul_weights_compression_cpp_brief
///
/// @include cpu_matmul_weights_compression.cpp

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

using namespace dnnl;

void matmul_example(dnnl::engine::kind engine_kind) {
    // Create execution dnnl::engine.
    dnnl::engine engine(engine_kind, 0);

    // Create dnnl::stream.
    dnnl::stream engine_stream(engine);

    // Tensor dimensions.
    const memory::dim M = 512, K = 512, N = 512;

    // Source (src), weights, and destination (dst) tensors dimensions.
    memory::dims src_dims = {M, K};
    memory::dims weights_dims = {K, N};
    memory::dims dst_dims = {M, N};

    // Allocate buffers.
    std::vector<float> src_data(product(src_dims));
    std::vector<float> weights_data(product(weights_dims));
    std::vector<float> dst_data(product(dst_dims));

    // Initialize src, weights.
    std::generate(src_data.begin(), src_data.end(), []() {
        static int i = 0;
        return std::cos(i++ / 10.f);
    });

    std::generate(weights_data.begin(), weights_data.end(), [&]() {
        static const float density = 0.1f;
        static std::default_random_engine def_gen;
        static std::bernoulli_distribution b_dist(density);
        const auto is_one = b_dist(def_gen);

        static int i = 1;
        return std::sin(i++ * 2.f) * is_one;
    });

    const memory::dim nnz = std::count_if(weights_data.begin(),
            weights_data.end(), [](float v) { return v != 0.0f; });

    auto src_md = memory::desc(
            src_dims, memory::data_type::f32, memory::format_tag::ab);
    auto dst_md = memory::desc(
            dst_dims, memory::data_type::f32, memory::format_tag::ab);

    auto src_mem = memory(src_md, engine);
    auto dst_mem = memory(dst_md, engine);

    auto user_src_mem = memory(
            {src_dims, memory::data_type::f32, memory::format_tag::ab}, engine);
    auto user_weights_mem = memory(
            {weights_dims, memory::data_type::f32, memory::format_tag::ab},
            engine);
    auto user_dst_mem = memory(
            {dst_dims, memory::data_type::f32, memory::format_tag::ab}, engine);

    write_to_dnnl_memory(src_data.data(), src_mem);
    write_to_dnnl_memory(weights_data.data(), user_weights_mem);

    auto matmul_src_md = memory::desc(
            src_dims, memory::data_type::u8, memory::format_tag::any);
    auto matmul_weights_md
            = memory::desc::packed(weights_dims, memory::data_type::s8, nnz);
    auto matmul_dst_md = memory::desc(
            dst_dims, memory::data_type::u8, memory::format_tag::any);

    matmul::primitive_desc matmul_pd;
    try {
        matmul_pd = matmul::primitive_desc(
                engine, matmul_src_md, matmul_weights_md, matmul_dst_md);
    } catch (error &e) {
        if (e.status == dnnl_unimplemented
                || e.status == dnnl_unsupported_data_type)
            throw example_allows_unimplemented {
                    "No matmul implementation with packed encoding support is "
                    "available for this platform.\nPlease refer to the "
                    "developer guide for details."};

        // on any other error just re-throw
        throw;
    }

    auto matmul_src_mem = user_src_mem;
    auto matmul_weights_mem = user_weights_mem;
    auto matmul_dst_mem = user_dst_mem;

    auto matmul_prim = matmul(matmul_pd);

    if (matmul_pd.src_desc() != user_src_mem.get_desc()) {
        matmul_src_mem = memory(matmul_pd.src_desc(), engine);
        reorder(user_src_mem, matmul_src_mem)
                .execute(engine_stream, user_src_mem, matmul_src_mem);
    }

    // Use reorder to pack the weights.
    auto wei_packed_md = matmul_pd.weights_desc();
    const int nhandles = wei_packed_md.get_num_handles();
    std::vector<void *> wei_handles(nhandles);
    std::vector<std::vector<char>> wei_buffers(nhandles);
    for (int h = 0; h < nhandles; h++) {
        const size_t buf_sz = wei_packed_md.get_size(h);
        wei_buffers[h].resize(buf_sz);
        wei_handles[h] = wei_buffers[h].data();
    }

    if (wei_packed_md != user_weights_mem.get_desc()) {
        matmul_weights_mem
                = memory(wei_packed_md, engine, std::move(wei_handles));
        reorder(user_weights_mem, matmul_weights_mem)
                .execute(engine_stream, user_weights_mem, matmul_weights_mem);
    }

    if (matmul_pd.dst_desc() != user_dst_mem.get_desc()) {
        matmul_dst_mem = memory(matmul_pd.dst_desc(), engine);
        reorder(user_dst_mem, matmul_dst_mem)
                .execute(engine_stream, user_dst_mem, matmul_dst_mem);
    }

    // Primitive arguments.
    std::unordered_map<int, memory> matmul_args;
    matmul_args.insert({DNNL_ARG_SRC, matmul_src_mem});
    matmul_args.insert({DNNL_ARG_WEIGHTS, matmul_weights_mem});
    matmul_args.insert({DNNL_ARG_DST, matmul_dst_mem});

    // Primitive execution: matrix multiplication with ReLU.
    matmul_prim.execute(engine_stream, matmul_args);

    // Wait for the computation to finalize.
    engine_stream.wait();

    // Read data from memory object's handle.
    read_from_dnnl_memory(dst_data.data(), dst_mem);
}

int main(int argc, char **argv) {
    return handle_example_errors(matmul_example, parse_engine_kind(argc, argv));
}
