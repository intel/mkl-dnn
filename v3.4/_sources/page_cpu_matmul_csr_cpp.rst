.. index:: pair: page; MatMul Primitive Example
.. _doxid-cpu_matmul_csr_cpp:

MatMul Primitive Example
========================

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
	#include <string>
	#include <vector>
	
	#include "dnnl.hpp"
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	bool check_result(:ref:`dnnl::memory <doxid-structdnnl_1_1memory>` dst_mem) {
	    // clang-format off
	    const std::vector<float> expected_result = {8.750000, 11.250000, 2.500000,
	                                                6.000000,  2.250000, 3.750000,
	                                               19.000000, 15.500000, 5.250000,
	                                                4.000000,  7.000000, 3.000000};
	    // clang-format on
	
	    std::vector<float> dst_data(expected_result.size());
	    read_from_dnnl_memory(dst_data.data(), dst_mem);
	    return expected_result == dst_data;
	}
	
	void sparse_matmul() {
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`, 0);
	
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` M = 4;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` K = 6;
	
	    // A sparse matrix represented in the CSR format.
	    std::vector<float> src_csr_values = {2.5f, 1.5f, 1.5f, 2.5f, 2.0f};
	    std::vector<int32_t> src_csr_indices = {0, 2, 0, 5, 1};
	    std::vector<int32_t> src_csr_pointers = {0, 1, 2, 4, 5, 5};
	
	    // clang-format off
	    std::vector<float> weights_data = {3.5f, 4.5f, 1.0f,
	                                       2.0f, 3.5f, 1.5f,
	                                       4.0f, 1.5f, 2.5f,
	                                       3.5f, 5.5f, 4.5f,
	                                       1.5f, 2.5f, 5.5f,
	                                       5.5f, 3.5f, 1.5f};
	    // clang-format on
	
	    const int nnz = static_cast<int>(src_csr_values.size());
	
	    // Create a memory descriptor for CSR format by providing information
	    // about number of non-zero entries and data types of metadata.
	    const auto src_csr_md
	            = :ref:`memory::desc::csr <doxid-structdnnl_1_1memory_1_1desc_1a7fe93a14828506260740fb439eaf6ed4>`({M, K}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, nnz, :ref:`dt::s32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215aa860868d23f3a68323a2e3f6563d7f31>`, :ref:`dt::s32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215aa860868d23f3a68323a2e3f6563d7f31>`);
	    const auto wei_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({K, N}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::oi);
	    const auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({M, N}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nc);
	
	    // This memory is created for the given values and metadata of CSR format.
	    :ref:`memory <doxid-structdnnl_1_1memory>` src_csr_mem(src_csr_md, :ref:`engine <doxid-structdnnl_1_1engine>`,
	            {src_csr_values.data(), src_csr_indices.data(),
	                    src_csr_pointers.data()});
	    :ref:`memory <doxid-structdnnl_1_1memory>` wei_mem(wei_md, :ref:`engine <doxid-structdnnl_1_1engine>`, weights_data.data());
	    :ref:`memory <doxid-structdnnl_1_1memory>` dst_mem(dst_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` :ref:`stream <doxid-structdnnl_1_1stream>`(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    auto sparse_matmul_pd
	            = :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`, src_csr_md, wei_md, dst_md);
	    auto sparse_matmul_prim = :ref:`matmul <doxid-structdnnl_1_1matmul>`(sparse_matmul_pd);
	
	    std::unordered_map<int, memory> sparse_matmul_args;
	    sparse_matmul_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_csr_mem});
	    sparse_matmul_args.insert({:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_mem});
	    sparse_matmul_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem});
	
	    sparse_matmul_prim.execute(:ref:`stream <doxid-structdnnl_1_1stream>`, sparse_matmul_args);
	    :ref:`stream <doxid-structdnnl_1_1stream>`.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	    if (!check_result(dst_mem)) throw :ref:`std::runtime_error <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808>`("Unexpected output.");
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors({:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`}, sparse_matmul);
	}

