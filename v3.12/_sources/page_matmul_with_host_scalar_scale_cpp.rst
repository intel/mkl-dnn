.. index:: pair: page; MatMul with Host Scalar Scale example
.. _doxid-matmul_with_host_scalar_scale_cpp:

MatMul with Host Scalar Scale example
=====================================

This C++ API example demonstrates matrix multiplication (C = alpha \* A \* B) with a scalar scaling factor residing on the host.

The workflow includes following steps:

* Initialize a oneDNN engine and stream for computation.

* Allocate and initialize matrices A and B.

* Create oneDNN memory objects for matrices A, B, and C.

* Prepare a scalar (alpha) as a host-side float value and wrap it in a oneDNN memory object.

* Create a matmul primitive descriptor with the scalar scale attribute.

* Create a matmul primitive.

* Execute the matmul primitive.

* Validate the result.

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
	
	#include <algorithm>
	#include <cmath>
	#include <iostream>
	#include <string>
	#include <vector>
	
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	
	
	
	// Compare straightforward matrix multiplication (C = alpha * A * B)
	// with the result from oneDNN memory.
	bool check_result(const std::vector<float> &a_data,
	        const std::vector<float> &b_data, int M, int N, int K, float alpha,
	        :ref:`dnnl::memory <doxid-structdnnl_1_1memory>` &c_mem) {
	    std::vector<float> c_ref(M * N, 0.0f);
	    // a: M x K, w: K x N, c: M x N
	    for (int i = 0; i < M; ++i) {
	        for (int j = 0; j < N; ++j) {
	            c_ref[i * N + j] = 0.0f;
	            for (int k = 0; k < K; ++k) {
	                c_ref[i * N + j] += a_data[i * K + k] * b_data[k * N + j];
	            }
	            c_ref[i * N + j] *= alpha;
	        }
	    }
	
	    std::vector<float> c_result(M * N, 0.0f);
	    read_from_dnnl_memory(c_result.data(), c_mem);
	
	    for (int i = 0; i < M; ++i) {
	        for (int j = 0; j < N; ++j) {
	            if (std::abs(c_result[i * N + j] - c_ref[i * N + j]) > 1e-5) {
	                return false;
	            }
	        }
	    }
	    return true;
	}
	
	// Simple matrix multiplication with alpha as scalar memory
	void simple_matmul_with_host_scalar(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    // Initialize a oneDNN engine and stream for computation
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	
	    // Define the dimensions for matrices A (MxK), B (KxN), and C (MxN)
	    const int M = 3, N = 3, K = 3;
	
	    // Allocate and initialize matrix A with float values
	    // and create a oneDNN memory object for it
	    std::vector<float> a_data(M * K, 0.0f);
	    for (int i = 0; i < M * K; ++i) {
	        a_data[i] = static_cast<float>(i + 1);
	    }
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` a_dims = {M, K};
	    :ref:`memory <doxid-structdnnl_1_1memory>` a_mem({a_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`}, eng);
	    write_to_dnnl_memory(a_data.data(), a_mem);
	
	    // Allocate and initialize matrix B with values based on the sum of their indices
	    // and create a oneDNN memory object for it
	    std::vector<float> b_data(K * N, 0.0f);
	    for (int i = 0; i < K; ++i) {
	        for (int j = 0; j < N; ++j) {
	            b_data[i * N + j] = static_cast<float>(i + j);
	        }
	    }
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` b_dims = {K, N};
	    :ref:`memory <doxid-structdnnl_1_1memory>` b_mem({b_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`}, eng);
	    write_to_dnnl_memory(b_data.data(), b_mem);
	
	    // Create oneDNN memory object for the output matrix C
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` c_dims = {M, N};
	    :ref:`memory <doxid-structdnnl_1_1memory>` c_mem({c_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`}, eng);
	
	    // Prepare a scalar (alpha) as a host-side float value and wrap it in a oneDNN memory object
	    float alpha = 2.0f;
	    :ref:`memory <doxid-structdnnl_1_1memory>` alpha_m(:ref:`memory::desc::host_scalar <doxid-structdnnl_1_1memory_1_1desc_1a27db39fcff710e27f134e107a1ec8857>`(:ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`), alpha);
	
	    // Create a matmul primitive descriptor with scaling for source memory (A)
	    // Set scaling mask to 0 and use host scalar for alpha
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	    attr.:ref:`set_host_scale <doxid-structdnnl_1_1primitive__attr_1a7b035390cde177453afae9c5b5a7c29e>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`);
	    :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>` matmul_pd(
	            eng, a_mem.get_desc(), b_mem.get_desc(), c_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`(), attr);
	
	    // Create a matmul primitive
	    :ref:`matmul <doxid-structdnnl_1_1matmul>` matmul_prim(matmul_pd);
	
	    // Prepare the arguments map for the matmul execution
	    std::unordered_map<int, memory> args = {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, a_mem},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, b_mem}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, c_mem},
	            {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, alpha_m}};
	
	    // Execute matmul
	    matmul_prim.execute(s, args);
	    s.wait();
	
	    // Verify results
	    if (!check_result(a_data, b_data, M, N, N, alpha, c_mem)) {
	        throw :ref:`std::runtime_error <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808>`("Result verification failed!");
	    }
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            simple_matmul_with_host_scalar, parse_engine_kind(argc, argv));
	}

