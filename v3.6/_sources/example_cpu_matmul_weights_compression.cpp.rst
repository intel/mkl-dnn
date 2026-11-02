.. index:: pair: example; cpu_matmul_weights_compression.cpp
.. _doxid-cpu_matmul_weights_compression_8cpp-example:

cpu_matmul_weights_compression.cpp
==================================

Annotated version: :ref:`MatMul Primitive Example <doxid-cpu_matmul_weights_compression_cpp>`

Annotated version: :ref:`MatMul Primitive Example <doxid-cpu_matmul_weights_compression_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`MatMul <doxid-dev_guide_matmul>` primitive that uses a weights tensor encoded with the packed sparse encoding.

.. ref-code-block:: cpp

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
	
	
	#include <algorithm>
	#include <cmath>
	#include <iostream>
	#include <random>
	#include <string>
	#include <vector>
	
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	void matmul_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` M = 512, K = 512, N = 512;
	
	    // Source (src), weights, and destination (dst) tensors dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {M, K};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` weights_dims = {K, N};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_dims = {M, N};
	
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
	
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` nnz = std::count_if(weights_data.begin(),
	            weights_data.end(), [](float v) { return v != 0.0f; });
	
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::ab);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(dst_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::ab);
	
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    auto user_src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::ab}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto user_weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({weights_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::ab}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto user_dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({dst_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::ab}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    write_to_dnnl_memory(src_data.data(), src_mem);
	    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
	
	    auto matmul_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, dt::u8, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto matmul_weights_md = :ref:`memory::desc::packed <doxid-structdnnl_1_1memory_1_1desc_1a4fd3a581a042d66f0d6243665321621a>`(weights_dims, dt::s8, nnz);
	    auto matmul_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(dst_dims, dt::u8, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>` matmul_pd;
	    try {
	        matmul_pd = :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(
	                :ref:`engine <doxid-structdnnl_1_1engine>`, matmul_src_md, matmul_weights_md, matmul_dst_md);
	    } catch (:ref:`error <doxid-structdnnl_1_1error>` &e) {
	        if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`)
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
	
	    auto matmul_prim = :ref:`matmul <doxid-structdnnl_1_1matmul>`(matmul_pd);
	
	    if (matmul_pd.:ref:`src_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc_1a9b9fc61ab0fe6354dd96757ede7b92dc>`() != user_src_mem.get_desc()) {
	        matmul_src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(matmul_pd.:ref:`src_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc_1a9b9fc61ab0fe6354dd96757ede7b92dc>`(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_src_mem, matmul_src_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_src_mem, matmul_src_mem);
	    }
	
	    // Use reorder to pack the weights.
	    auto wei_packed_md = matmul_pd.:ref:`weights_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc_1a0be2d3c1fd1674bd6808c0e82c035c2f>`();
	    const int nhandles = wei_packed_md.:ref:`get_num_handles <doxid-structdnnl_1_1memory_1_1desc_1ad1f0ad6584fa547dba0dd72d54b9162b>`();
	    std::vector<void *> wei_handles(nhandles);
	    std::vector<std::vector<char>> wei_buffers(nhandles);
	    for (int h = 0; h < nhandles; h++) {
	        const size_t buf_sz = wei_packed_md.get_size(h);
	        wei_buffers[h].resize(buf_sz);
	        wei_handles[h] = wei_buffers[h].data();
	    }
	
	    if (wei_packed_md != user_weights_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        matmul_weights_mem
	                = :ref:`memory <doxid-structdnnl_1_1memory>`(wei_packed_md, :ref:`engine <doxid-structdnnl_1_1engine>`, std::move(wei_handles));
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_mem, matmul_weights_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_weights_mem, matmul_weights_mem);
	    }
	
	    if (matmul_pd.:ref:`dst_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc_1ad35cf09a2aaf3cd7db751b6c01d44f80>`() != user_dst_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        matmul_dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(matmul_pd.:ref:`dst_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc_1ad35cf09a2aaf3cd7db751b6c01d44f80>`(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_dst_mem, matmul_dst_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_dst_mem, matmul_dst_mem);
	    }
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> matmul_args;
	    matmul_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, matmul_src_mem});
	    matmul_args.insert({:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, matmul_weights_mem});
	    matmul_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, matmul_dst_mem});
	
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
