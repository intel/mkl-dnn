.. index:: pair: example; weights_decompression_matmul.cpp
.. _doxid-weights_decompression_matmul_8cpp-example:

weights_decompression_matmul.cpp
================================

Annotated version: :ref:`MatMul Tutorial: weights decompression <doxid-weights_decompression_matmul_cpp>`

Annotated version: :ref:`MatMul Tutorial: weights decompression <doxid-weights_decompression_matmul_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2023-2024 Intel Corporation
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
	
	} // namespace
	
	int number_of_runs = 1;
	
	// Create a MatMul primitive descriptor for the following op:
	// C_f32 = A_f32 * (B_s8 - zp_B) * sc_B[:]
	//
	// Here:
	// - Matrices A and C are of f32 data type.
	// - The B matrix is stored as int8_t, its zero point is zp_B, and all its
	//   dimensions are known. This matrix can be a matrix of compressed weights
	//   in an MLP topology.
	// - The weights scaling values are not known at the primitive creation time.
	:ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>` matmul_pd_create(
	        int64_t M, int64_t N, int64_t K, int64_t G, const :ref:`engine <doxid-structdnnl_1_1engine>` &eng) {
	
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` a_md({M, K}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {K, 1}); // M x K layout
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` b_md({K, N}, :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` c_md({M, N}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {N, 1}); // M x N layout
	
	    // Create attributes and indicate that the alpha and zero points are
	    // runtime parameters
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	    // Set scales with multiple scales along K and N dimensions and with groups along K.
	    attr.set_scales(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`,
	            /* mask */ (1 << 0) + (1 << 1), {G, 1}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`);
	    // Set a single zero point with s8 data type.
	    attr.set_zero_points(
	            :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, /* mask */ 0, {}, :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`);
	    // Set fpmath mode with `apply_to_int=true` to apply fpmath mode behavior to
	    // integral primitives (in this example, matmul).
	    attr.set_fpmath_mode(:ref:`fpmath_mode::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, true);
	
	    // Create a MatMul primitive descriptor
	    return :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(eng, a_md, b_md, c_md, attr);
	}
	
	void prepare_input(:ref:`memory <doxid-structdnnl_1_1memory>` &A_f32_mem, :ref:`memory <doxid-structdnnl_1_1memory>` &sc_B_mem, :ref:`memory <doxid-structdnnl_1_1memory>` &zp_B_mem) {
	    int64_t M = A_f32_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[0];
	    int64_t N = sc_B_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[0];
	    int64_t K = A_f32_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[1];
	    int64_t NUM_G = sc_B_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[1];
	
	    std::vector<float> A_f32(M * K);
	    init_vector(A_f32);
	
	    std::vector<float> sc_B(NUM_G * N);
	    init_vector(sc_B);
	
	    int8_t zp_B = 2;
	
	    write_to_dnnl_memory(A_f32.data(), A_f32_mem);
	    write_to_dnnl_memory(&zp_B, zp_B_mem);
	    write_to_dnnl_memory(sc_B.data(), sc_B_mem);
	}
	
	void infer(const :ref:`matmul <doxid-structdnnl_1_1matmul>` &matmul_p, int64_t M, int64_t N, int64_t K, int64_t G,
	        const :ref:`memory <doxid-structdnnl_1_1memory>` &B_s8_mem, const :ref:`engine <doxid-structdnnl_1_1engine>` &eng) {
	    // input of the current layer / operation
	    :ref:`memory <doxid-structdnnl_1_1memory>` A_f32_mem({{M, K}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {K, 1}}, eng);
	    // De-quantization parameters (eg. Scale and Shift)
	    const int64_t n_groups = K / G;
	    :ref:`memory <doxid-structdnnl_1_1memory>` sc_B_mem({{N, n_groups}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {1, N}}, eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` zp_B_mem({{1}, :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, {1}}, eng);
	
	    // the function below fills dnnl::memory with some values
	    // these memories, typically, come from the previous layers / operations
	    // with meaningful data inside
	    prepare_input(A_f32_mem, sc_B_mem, zp_B_mem);
	
	    // output - no initialization required
	    :ref:`memory <doxid-structdnnl_1_1memory>` C_f32_mem({{M, N}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {N, 1}}, eng);
	
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	    for (int run = 0; run < number_of_runs; ++run)
	        matmul_p.:ref:`execute <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>`(s,
	                {{DNNL_ARG_SRC, A_f32_mem}, {DNNL_ARG_WEIGHTS, B_s8_mem},
	                        {DNNL_ARG_DST, C_f32_mem},
	                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_B_mem},
	                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS,
	                                zp_B_mem}});
	    s.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	}
	
	void weights_decompression_matmul(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	
	    const int64_t K = 96;
	    const int64_t N = 1000;
	    const int64_t M = 100;
	    // Quantization Group size for scales
	    const int64_t G = K / 2;
	
	    auto matmul_pd = matmul_pd_create(M, N, K, G, eng);
	
	    // Original weights stored as float in a known format
	    std::vector<float> B_f32(K * N);
	    init_vector(B_f32);
	
	    // Pre-packed weights stored as int8_t
	    :ref:`memory <doxid-structdnnl_1_1memory>` B_s8_mem(matmul_pd.:ref:`weights_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc_1a0be2d3c1fd1674bd6808c0e82c035c2f>`(), eng);
	    {
	        :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	        :ref:`memory <doxid-structdnnl_1_1memory>` B_f32_mem(
	                {{K, N}, memory::data_type::f32, memory::format_tag::ab}, eng);
	        write_to_dnnl_memory(B_f32.data(), B_f32_mem);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(B_f32_mem, B_s8_mem).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, B_f32_mem, B_s8_mem);
	        s.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	    }
	
	    :ref:`matmul <doxid-structdnnl_1_1matmul>` matmul_p(matmul_pd);
	
	    infer(matmul_p, M, N, K, G, B_s8_mem, eng);
	}
	
	int main(int argc, char **argv) {
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv);
	    // GPU is not supported
	    if (engine_kind != engine::kind::cpu) return 0;
	    return handle_example_errors(weights_decompression_matmul, engine_kind);
	}
