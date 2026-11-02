/*******************************************************************************
* Copyright 2025 Intel Corporation
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

/// @example mxfp_matmul.cpp
/// > Annotated version: @ref mxfp_matmul_cpp
///
/// @page mxfp_matmul_cpp_brief
/// @brief C++ API example demonstrating how one can use
/// [MatMul](@ref dev_guide_matmul) with MXFP8 datatype in inference.
///
/// @page mxfp_matmul_cpp MatMul Tutorial: MXFP8 Inference
/// \copybrief mxfp_matmul_cpp_brief
///
/// Concepts:
/// - Dynamic quantization compliant with MX specification
///   - Scales: dnnl::primitive_attr::set_scales()
/// - Create primitive once, use multiple times
///
/// @include mxfp_matmul.cpp

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "oneapi/dnnl/dnnl.hpp"

#include "example_utils.hpp"

using namespace dnnl;

namespace {

void init_vector(std::vector<float> &v) {
    std::mt19937 gen;
    std::uniform_real_distribution<float> u(0, 1);
    for (auto &e : v)
        e = u(gen);
}

uint8_t f32_to_e8m0(const float &a) {
    // Note: memcpy can be replaced with bit_cast in C++20
    uint32_t a_s32;
    std::memcpy(&a_s32, &a, sizeof(float));
    uint8_t a_e8m0 = (a_s32 >> 23) & 0xff;
    return a_e8m0;
}

float e8m0_to_f32(const uint8_t a) {
    float r_f32;
    uint32_t r_s32;

    if (a == 0xff) return std::numeric_limits<float>::quiet_NaN();

    // Note: memcpy can be replaced with bit_cast in C++20
    if (a == 0x00)
        r_s32 = uint32_t(0x00400000); // 2^-127 encoding in float
    else
        r_s32 = uint32_t(a) << 23;
    std::memcpy(&r_f32, &r_s32, sizeof(float));
    return r_f32;
}
} // namespace

const memory::dim mx_block_size = 32;
int number_of_runs = 1;

// Create a MatMul primitive descriptor with:
//
// - Matrices A and C are non-transposed, B is transposed
// - All matrices uses MXFP8 format with e4m3 elements,
//   e8m0 scales, and blocks of size 32.
// - The scales values are precomputed for A and B as they are already
//   quantized
// - The scales values for C will be computed according to MX spec by
//   the MatMul primitive and written to DNNL_ARG_ATTR_SCALES |
//   DNNL_ARG_DST memory argument during execution
matmul::primitive_desc matmul_pd_create(
        int64_t M, int64_t N, int64_t K, const engine &eng) {
    memory::desc a_md(
            {M, K}, memory::data_type::f8_e4m3, {K, 1}); // M x K layout
    memory::desc b_md({K, N}, memory::data_type::f8_e4m3, {1, K});
    memory::desc c_md(
            {M, N}, memory::data_type::f8_e4m3, {N, 1}); // M x N layout

    // Create scales attributes to indicate scales datatype, group
    // shapes, and how they are computed:
    // - user-provided for DNNL_ARG_SRC and DNNL_ARG_WEIGHTS memory arguments
    // - library-computed according to MX spec for DNNL_ARG_DST memory argument
    int mask = (1 << 1) | (1 << 0);
    primitive_attr attr;
    attr.set_scales(DNNL_ARG_SRC, mask, {1, 32}, memory::data_type::e8m0);
    attr.set_scales(DNNL_ARG_WEIGHTS, mask, {32, 1}, memory::data_type::e8m0);
    // Specifying the dynamic_mx quantization mode signals the compute
    // primitive to effectively compute MX compliant scales, and write
    // them to DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST buffer.
    attr.set_scales(DNNL_ARG_DST, mask, {1, 32}, memory::data_type::e8m0, false,
            quantization_mode::dynamic_mx);

    // Create a MatMul primitive descriptor
    try {
        return matmul::primitive_desc(eng, a_md, b_md, c_md, attr);
    } catch (error &e) {
        if (e.status == dnnl_unimplemented
                || e.status == dnnl_unsupported_data_type)
            throw example_allows_unimplemented {
                    "No mxfp8 matmul implementation is available for this "
                    "platform.\n"
                    "Please refer to the developer guide for details."};

        // on any other error just re-throw
        throw;
    }
}

// Takes A_mem and B_mem as inputs, and returns quantized version and scales
// Matrix is assumed row major
void quantize_input(memory &in_e4m3, memory &in_scales_e8m0) {
    // This is the conversion routine defined by OCP MX spec v1
    const auto dims = in_e4m3.get_desc().get_dims();
    const auto nelems = product(dims);

    // Initialize f32 random values
    std::vector<float> in_buff(nelems);
    init_vector(in_buff);

    std::vector<uint8_t> scales_buff(nelems);

    assert((dims[dims.size() - 1] % mx_block_size) == 0);

    // We compute the e8m0 scaling factors of each mx block in the following loop
    memory::dim nblocks = nelems / mx_block_size;
    for (memory::dim i = 0; i < nblocks; ++i) {
        // We first compute the scale value for the block
        float block_amax = 0.0f;
        for (memory::dim j = 0; j < mx_block_size; ++j)
            block_amax = std::max(
                    block_amax, std::abs(in_buff[i * mx_block_size + j]));
        const float max_e4m3 = 448.f;
        uint8_t e8m0_scale = f32_to_e8m0(block_amax) - f32_to_e8m0(max_e4m3);
        scales_buff[i] = e8m0_scale;

        // We then apply that scale inside the block. We do that
        // inplace as the f32 buffer is not reused.
        float f32_scale = e8m0_to_f32(e8m0_scale);
        for (memory::dim j = 0; j < mx_block_size; ++j)
            in_buff[i * mx_block_size + j] *= f32_scale;
    }

    // we now downconvert to e4m3 through reorder
    memory in_f32({in_e4m3.get_desc().get_dims(), memory::data_type::f32,
                          in_e4m3.get_desc().get_strides()},
            in_e4m3.get_engine());
    write_to_dnnl_memory(in_buff.data(), in_f32);
    reorder(in_f32, in_e4m3);

    write_to_dnnl_memory(scales_buff.data(), in_scales_e8m0);
}

void mxfp_matmul(engine::kind engine_kind) {
    engine eng(engine_kind, 0);

    const int64_t K = 128;
    const int64_t N = 64;
    const int64_t M = 96;

    auto matmul_pd = matmul_pd_create(M, N, K, eng);
    matmul matmul_p(matmul_pd);

    // The following code initializes the inputs that are typically
    // provided:
    // - activations are quantized by previous layer
    // - weights can be quantized offline ahead of time.
    auto a_desc = matmul_pd.src_desc();
    memory A_e4m3_elems_mem(a_desc, eng);
    memory A_e8m0_scales_mem(
            {{M * (K / mx_block_size)}, memory::data_type::e8m0, {1}}, eng);
    quantize_input(A_e4m3_elems_mem, A_e8m0_scales_mem);

    auto b_desc = matmul_pd.weights_desc();
    memory B_e4m3_elems_mem(b_desc, eng);
    memory B_e8m0_scales_mem(
            {{(K / mx_block_size) * N}, memory::data_type::e8m0, {1}}, eng);
    quantize_input(B_e4m3_elems_mem, B_e8m0_scales_mem);

    // For C, we only allocate as those will be populated by the
    // matmul execute call.
    memory C_e4m3_elems_mem(matmul_pd.dst_desc(), eng);
    memory C_e8m0_scales_mem(
            {{M * (N / mx_block_size)}, memory::data_type::e8m0, {1}}, eng);

    // Now MatMul primitive is run on a stream.  For SRC, WEIGHTS and
    // DST, we provide both elements and associated scales as separate
    // buffers.
    stream s(eng);
    for (int run = 0; run < number_of_runs; ++run)
        matmul_p.execute(s,
                {{DNNL_ARG_SRC, A_e4m3_elems_mem},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,
                                A_e8m0_scales_mem},
                        {DNNL_ARG_WEIGHTS, B_e4m3_elems_mem},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS,
                                B_e8m0_scales_mem},
                        {DNNL_ARG_DST, C_e4m3_elems_mem},
                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                                C_e8m0_scales_mem}});
    s.wait();
}

int main(int argc, char **argv) {
    engine::kind engine_kind = parse_engine_kind(argc, argv);
    return handle_example_errors(mxfp_matmul, engine_kind);
}
