.. index:: pair: page; MatMul Tutorial: MXFP8 Inference
.. _doxid-mxfp_matmul_cpp:

MatMul Tutorial: MXFP8 Inference
================================

C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` with MXFP8 datatype in inference.

Concepts:

* Dynamic quantization compliant with MX specification
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales() <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`

* Create primitive once, use multiple times

.. ref-code-block:: cpp

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
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
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
	
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` mx_block_size = 32;
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
	:ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>` matmul_pd_create(
	        int64_t M, int64_t N, int64_t K, const :ref:`engine <doxid-structdnnl_1_1engine>` &eng) {
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` a_md(
	            {M, K}, :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`, {K, 1}); // M x K layout
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` b_md({K, N}, :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`, {1, K});
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` c_md(
	            {M, N}, :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`, {N, 1}); // M x N layout
	
	    // Create scales attributes to indicate scales datatype, group
	    // shapes, and how they are computed:
	    // - user-provided for DNNL_ARG_SRC and DNNL_ARG_WEIGHTS memory arguments
	    // - library-computed according to MX spec for DNNL_ARG_DST memory argument
	    int mask = (1 << 1) | (1 << 0);
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	    attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, mask, {1, 32}, :ref:`memory::data_type::e8m0 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea8af1e244959fd40c752655b5d39980eb>`);
	    attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, mask, {32, 1}, :ref:`memory::data_type::e8m0 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea8af1e244959fd40c752655b5d39980eb>`);
	    // Specifying the dynamic_mx quantization mode signals the compute
	    // primitive to effectively compute MX compliant scales, and write
	    // them to DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST buffer.
	    attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, mask, {1, 32}, :ref:`memory::data_type::e8m0 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea8af1e244959fd40c752655b5d39980eb>`, false,
	            :ref:`quantization_mode::dynamic_mx <doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2ca10dabb84b08ade6e41ee83eba1e96f9d>`);
	
	    // Create a MatMul primitive descriptor
	    try {
	        return :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(eng, a_md, b_md, c_md, attr);
	    } catch (:ref:`error <doxid-structdnnl_1_1error>` &e) {
	        if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`)
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
	void quantize_input(:ref:`memory <doxid-structdnnl_1_1memory>` &in_e4m3, :ref:`memory <doxid-structdnnl_1_1memory>` &in_scales_e8m0) {
	    // This is the conversion routine defined by OCP MX spec v1
	    const auto dims = in_e4m3.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`();
	    const auto nelems = product(dims);
	
	    // Initialize f32 random values
	    std::vector<float> in_buff(nelems);
	    init_vector(in_buff);
	
	    std::vector<uint8_t> scales_buff(nelems);
	
	    assert((dims[dims.size() - 1] % mx_block_size) == 0);
	
	    // We compute the e8m0 scaling factors of each mx block in the following loop
	    :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` nblocks = nelems / mx_block_size;
	    for (:ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` i = 0; i < nblocks; ++i) {
	        // We first compute the scale value for the block
	        float block_amax = 0.0f;
	        for (:ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` j = 0; j < mx_block_size; ++j)
	            block_amax = std::max(
	                    block_amax, std::abs(in_buff[i * mx_block_size + j]));
	        const float max_e4m3 = 448.f;
	        uint8_t e8m0_scale = f32_to_e8m0(block_amax) - f32_to_e8m0(max_e4m3);
	        scales_buff[i] = e8m0_scale;
	
	        // We then apply that scale inside the block. We do that
	        // inplace as the f32 buffer is not reused.
	        float f32_scale = e8m0_to_f32(e8m0_scale);
	        for (:ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` j = 0; j < mx_block_size; ++j)
	            in_buff[i * mx_block_size + j] *= f32_scale;
	    }
	
	    // we now downconvert to e4m3 through reorder
	    :ref:`memory <doxid-structdnnl_1_1memory>` in_f32({in_e4m3.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`(), :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	                          in_e4m3.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_strides <doxid-structdnnl_1_1memory_1_1desc_1aa4b72acda1a8c929cdc6829e715930f4>`()},
	            in_e4m3.:ref:`get_engine <doxid-structdnnl_1_1memory_1a9074709c5af8dc9d25dd9a98c4d1dbd3>`());
	    write_to_dnnl_memory(in_buff.data(), in_f32);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(in_f32, in_e4m3);
	
	    write_to_dnnl_memory(scales_buff.data(), in_scales_e8m0);
	}
	
	void mxfp_matmul(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	
	    const int64_t K = 128;
	    const int64_t N = 64;
	    const int64_t M = 96;
	
	    auto matmul_pd = matmul_pd_create(M, N, K, eng);
	    :ref:`matmul <doxid-structdnnl_1_1matmul>` matmul_p(matmul_pd);
	
	    // The following code initializes the inputs that are typically
	    // provided:
	    // - activations are quantized by previous layer
	    // - weights can be quantized offline ahead of time.
	    auto a_desc = matmul_pd.src_desc();
	    :ref:`memory <doxid-structdnnl_1_1memory>` A_e4m3_elems_mem(a_desc, eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` A_e8m0_scales_mem(
	            {{M * (K / mx_block_size)}, :ref:`memory::data_type::e8m0 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea8af1e244959fd40c752655b5d39980eb>`, {1}}, eng);
	    quantize_input(A_e4m3_elems_mem, A_e8m0_scales_mem);
	
	    auto b_desc = matmul_pd.weights_desc();
	    :ref:`memory <doxid-structdnnl_1_1memory>` B_e4m3_elems_mem(b_desc, eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` B_e8m0_scales_mem(
	            {{(K / mx_block_size) * N}, :ref:`memory::data_type::e8m0 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea8af1e244959fd40c752655b5d39980eb>`, {1}}, eng);
	    quantize_input(B_e4m3_elems_mem, B_e8m0_scales_mem);
	
	    // For C, we only allocate as those will be populated by the
	    // matmul execute call.
	    :ref:`memory <doxid-structdnnl_1_1memory>` C_e4m3_elems_mem(matmul_pd.dst_desc(), eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` C_e8m0_scales_mem(
	            {{M * (N / mx_block_size)}, :ref:`memory::data_type::e8m0 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea8af1e244959fd40c752655b5d39980eb>`, {1}}, eng);
	
	    // Now MatMul primitive is run on a stream.  For SRC, WEIGHTS and
	    // DST, we provide both elements and associated scales as separate
	    // buffers.
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	    for (int run = 0; run < number_of_runs; ++run)
	        matmul_p.:ref:`execute <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>`(s,
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
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv);
	    return handle_example_errors(mxfp_matmul, engine_kind);
	}

