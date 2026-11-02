.. index:: pair: page; MatMul Tutorial: INT8 Inference
.. _doxid-inference_int8_matmul_cpp:

MatMul Tutorial: INT8 Inference
===============================

C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` fused with ReLU in INT8 inference.

Concepts:

* Asymmetric quantization
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales_mask() <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`
  
  * Zero points: :ref:`dnnl::primitive_attr::set_zero_points_mask() <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`

* :ref:`Operation fusion <doxid-dev_guide_attributes_post_ops>`

* Create primitive once, use multiple times
  
  * Run-time tensor shapes: :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`

* Weights pre-packing: use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`

Assumptions:

#. The shape of the weights (matrix :math:`B(K, N)`) is known in advance, the data type is ``int8_t`` and centered around 0 (i.e. the zero point is 0).

#. The shapes of the source matrix :math:`A` and destination matrix :math:`C` are partially unknown. Both matrices use ``uint8_t`` data type and might have arbitrary zero points (specified at execution time only).

#. Scaling (re-quantization) factor specified at run-time only.

Since the shape of weights is known in advance, the MatMul weights can be created with format tag :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` to enable the library to choose the most appropriate layout for best performance.

.. warning:: 

   The format tag :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` doesn't work for memory descriptors that have one or more unknown dimensions and/or strides.
   
   


.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2019-2022 Intel Corporation
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
	
	void init_vector(std::vector<uint8_t> &v) {
	    std::mt19937 gen;
	    std::uniform_int_distribution<unsigned int> u(0, 255);
	    for (auto &e : v)
	        e = static_cast<uint8_t>(u(gen));
	}
	
	} // namespace
	
	int number_of_runs = 1;
	
	// Create a MatMul primitive descriptor for the following op:
	// C_u8 = ReLU(sc_A * sc_B[:] * (A_u8 - zp_A) * B_s8) / sc_C + zp_C
	//
	// Here:
	// - Matrices A and C are known to be non-transposed but their M dimension is
	//   not known. They can be activation matrices in an MLP topology and the M
	//   dimension can be the mini-batch dimension.
	// - zp_A and zp_C are zero points for matrices A and C which are stored as
	//   uint8_t. These are run-time parameters that are not known at the primitive
	//   creation time.
	// - The B matrix is stored as int8_t, its zero point is 0, and all its
	//   dimensions are known. This matrix can be a matrix of weights in an MLP
	//   topology.
	// - The scaling values are not known at the primitive creation time.
	:ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>` matmul_pd_create(
	        int64_t K, int64_t N, const :ref:`engine <doxid-structdnnl_1_1engine>` &eng) {
	    const int64_t M = :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`;
	
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` a_md({M, K}, :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, {K, 1}); // M x K layout
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` b_md({K, N}, :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` c_md({M, N}, :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, {N, 1}); // M x N layout
	
	    // Create attributes and indicate that the alpha and zero points are
	    // runtime parameters
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	    attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, /* mask */ 0);
	    attr.set_scales_mask(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, /* mask */ 1 << 1);
	    attr.set_scales_mask(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, /* mask */ 0);
	    attr.set_zero_points_mask(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, /* mask */ 0);
	    attr.set_zero_points_mask(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, /* mask */ 0);
	    :ref:`post_ops <doxid-structdnnl_1_1post__ops>` po;
	    po.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(:ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, 0.f, 0.f);
	    attr.set_post_ops(po);
	
	    // Create a MatMul primitive descriptor
	    return :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(eng, a_md, b_md, c_md, attr);
	}
	
	void prepare_input(:ref:`memory <doxid-structdnnl_1_1memory>` &A_u8_mem, :ref:`memory <doxid-structdnnl_1_1memory>` &sc_A_mem, :ref:`memory <doxid-structdnnl_1_1memory>` &sc_B_mem,
	        :ref:`memory <doxid-structdnnl_1_1memory>` &sc_C_mem, :ref:`memory <doxid-structdnnl_1_1memory>` &zp_A_mem, :ref:`memory <doxid-structdnnl_1_1memory>` &zp_C_mem) {
	    int64_t M = A_u8_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[0];
	    int64_t N = sc_B_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[0];
	    int64_t K = A_u8_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[1];
	
	    std::vector<uint8_t> A_u8(M * K);
	    init_vector(A_u8);
	
	    std::vector<float> sc_B(N);
	    init_vector(sc_B);
	
	    float sc_A = 0.5f;
	    float sc_C = 0.25f;
	    int32_t zp_A = 128, zp_C = 40;
	
	    write_to_dnnl_memory(A_u8.data(), A_u8_mem);
	    write_to_dnnl_memory(&zp_A, zp_A_mem);
	    write_to_dnnl_memory(&zp_C, zp_C_mem);
	    write_to_dnnl_memory(&sc_A, sc_A_mem);
	    write_to_dnnl_memory(sc_B.data(), sc_B_mem);
	    write_to_dnnl_memory(&sc_C, sc_C_mem);
	}
	
	void sanity_check(:ref:`memory <doxid-structdnnl_1_1memory>` &C_u8_mem, :ref:`memory <doxid-structdnnl_1_1memory>` &zp_C_mem) {
	    int64_t M = C_u8_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[0];
	    int64_t N = C_u8_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`()[1];
	    int32_t zp_C = 0;
	    std::vector<uint8_t> C_u8(M * N);
	
	    read_from_dnnl_memory(C_u8.data(), C_u8_mem);
	    read_from_dnnl_memory(&zp_C, zp_C_mem);
	
	    // simple check: C_u8 >= zp_C
	    for (int64_t i = 0; i < M * N; ++i)
	        if (C_u8[i] < zp_C)
	            throw std::logic_error(
	                    "Smoke check failed."
	                    "\n\tQuantized value is smaller than the zero point,"
	                    "\n\twhich should not happen since ReLU was applied.");
	}
	
	void infer(const :ref:`matmul <doxid-structdnnl_1_1matmul>` &matmul_p, int64_t M, int64_t N, int64_t K,
	        const :ref:`memory <doxid-structdnnl_1_1memory>` &B_s8_mem, const :ref:`engine <doxid-structdnnl_1_1engine>` &eng) {
	    // inputs of the current layer / operation
	    :ref:`memory <doxid-structdnnl_1_1memory>` A_u8_mem({{M, K}, :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, {K, 1}}, eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` zp_A_mem({{1}, :ref:`memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`, {1}}, eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` zp_C_mem({{1}, :ref:`memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`, {1}}, eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` sc_A_mem({{1}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {1}}, eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` sc_B_mem({{N}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {1}}, eng);
	    :ref:`memory <doxid-structdnnl_1_1memory>` sc_C_mem({{1}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, {1}}, eng);
	
	    // the function below fills dnnl::memory with some values
	    // these memories, typically, come from the previous layers / operations
	    // with meaningful data inside
	    prepare_input(A_u8_mem, sc_A_mem, sc_B_mem, sc_C_mem, zp_A_mem, zp_C_mem);
	
	    // output - no initialization required
	    :ref:`memory <doxid-structdnnl_1_1memory>` C_u8_mem({{M, N}, :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, {N, 1}}, eng);
	
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	    for (int run = 0; run < number_of_runs; ++run)
	        matmul_p.:ref:`execute <doxid-structdnnl_1_1primitive_1a2c112f2449a18a87310dee2ecd8c64eb>`(s,
	                {{DNNL_ARG_SRC, A_u8_mem}, {DNNL_ARG_WEIGHTS, B_s8_mem},
	                        {DNNL_ARG_DST, C_u8_mem},
	                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, sc_A_mem},
	                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, sc_B_mem},
	                        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, sc_C_mem},
	                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC, zp_A_mem},
	                        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_DST, zp_C_mem}});
	    s.wait();
	
	    // a sanity check for the correctness of the output
	    sanity_check(C_u8_mem, zp_C_mem);
	}
	
	void inference_int8_matmul(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	
	    const int64_t K = 96;
	    const int64_t N = 1000;
	    auto matmul_pd = matmul_pd_create(K, N, eng);
	
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
	        s.wait();
	    }
	
	    :ref:`matmul <doxid-structdnnl_1_1matmul>` matmul_p(matmul_pd);
	
	    for (int64_t M : {1, 100})
	        infer(matmul_p, M, N, K, B_s8_mem, eng);
	}
	
	int main(int argc, char **argv) {
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv);
	    return handle_example_errors(inference_int8_matmul, engine_kind);
	}

