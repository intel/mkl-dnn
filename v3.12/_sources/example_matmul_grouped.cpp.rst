.. index:: pair: example; matmul_grouped.cpp
.. _doxid-matmul_grouped_8cpp-example:

matmul_grouped.cpp
==================

Annotated version: :ref:`MatMul with Grouped Encoding <doxid-matmul_grouped_cpp>`



.. ref-code-block:: cpp

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
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	
	
	
	void grouped_matmul_example(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    // Create execution engine and stream for computation
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	    :ref:`stream <doxid-structdnnl_1_1stream>` engine_stream(eng);
	
	    // Sample token distribution across experts
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` num_experts = 4; // Number of experts in the MoE model
	    std::vector<int32_t> tokens_per_expert = {12, 8, 0, 10};
	
	    // Build cumulative offsets (exclusive-end boundaries)
	    // offsets[i] = total tokens up to and including expert i
	    std::vector<int32_t> offsets(num_experts);
	    offsets[0] = tokens_per_expert[0];
	    for (:ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` i = 1; i < num_experts; ++i) {
	        offsets[i] = offsets[i - 1] + tokens_per_expert[i];
	    }
	
	    // Total tokens number across all experts
	    :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` total_tokens = std::accumulate(
	            tokens_per_expert.begin(), tokens_per_expert.end(), :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>`(0));
	
	    std::cout << "Number of experts: " << num_experts << std::endl;
	
	    std::cout << "Token distribution: " << total_tokens << " total tokens";
	    std::cout << " routed to " << num_experts << " experts";
	    std::cout << " (";
	    for (:ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` i = 0; i < num_experts; ++i) {
	        std::cout << tokens_per_expert[i];
	        if (i < num_experts - 1) std::cout << ", ";
	    }
	    std::cout << " tokens per expert)" << std::endl;
	
	    // src is [total_tokens, K] with grouped encoding
	    // wei is [num_experts, K, N] with standard 3D format
	    // dst is [total_tokens, N] with grouped encoding
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` K = 64; // Input feature dimension
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` N = 128; // Output feature dimension
	
	    std::cout << "Input dimensions: K=" << K << " (features), N=" << N
	              << " (outputs)" << std::endl;
	    std::cout << "Weights: [" << num_experts << ", " << K << ", " << N
	              << "] tensor in acb format (experts × output_dim × input_dim)"
	              << std::endl;
	    std::cout << std::endl;
	
	    std::vector<float> src_data(total_tokens * K);
	    for (int i = 0; i < total_tokens * K; ++i) {
	        src_data[i] = i / 10.f;
	    }
	
	    std::vector<float> weights_data(num_experts * N * K);
	    for (int e = 0; e < num_experts; ++e) {
	        for (int n = 0; n < N; ++n) {
	            for (int k = 0; k < K; ++k) {
	                weights_data[e * N * K + n * K + k]
	                        = (e * K * N + k * N + n) / 20.f;
	            }
	        }
	    }
	
	    std::vector<float> dst_data(total_tokens * N, 0.0f);
	
	    // Create memory descriptors with grouped encoding
	    // variable_dim_idx=0 indicates M dimension varies per group
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` src_dims = {total_tokens, K};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` weights_dims = {num_experts, K, N};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` dst_dims = {total_tokens, N};
	
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc::grouped <doxid-structdnnl_1_1memory_1_1desc_1a56f86b15ab0ad07356207766c72661d0>`(
	            src_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, 0, num_experts);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc::grouped <doxid-structdnnl_1_1memory_1_1desc_1a56f86b15ab0ad07356207766c72661d0>`(
	            dst_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, 0, num_experts);
	    auto :ref:`weights_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a06ba7b00a8c95dcf3a90e16d00eeb0e9>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            weights_dims, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::acb <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa5ff832d9bca8241d653279756f3ccd11>`);
	
	    // Create memory objects
	    // Grouped memory has 2 buffers:
	    //     - buffer 0: concatenated data values
	    //     - buffer 1: cumulative offsets array
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, eng);
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_md, eng);
	    auto weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(weights_md, eng);
	
	    // Write data to buffer 0 (data values)
	    write_to_dnnl_memory(src_data.data(), src_mem);
	    write_to_dnnl_memory(weights_data.data(), weights_mem);
	
	    // Write offsets to buffer 1 (offsets buffer)
	    // Both src and dst must use identical offsets since token distribution
	    // is the same for input and output (each expert processes the same tokens)
	    write_to_dnnl_memory(offsets.data(), src_mem, 1);
	    write_to_dnnl_memory(offsets.data(), dst_mem, 1);
	
	    // Create primitive attributes with scales
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` matmul_attr;
	
	    // Row-wise (per-token) src scales: one scale per token
	    std::vector<float> src_scales(total_tokens);
	    for (int32_t i = 0; i < total_tokens; ++i)
	        src_scales[i] = 1.0f + (i % 100) / 500.0f;
	
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` src_scales_md(
	            {total_tokens}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::a <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa0cc175b9c0f1b6a831c399e269772661>`);
	    auto src_scales_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_scales_md, eng);
	    write_to_dnnl_memory(src_scales.data(), src_scales_mem);
	    matmul_attr.set_scales_mask(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, (1 << 0));
	
	    // Column-wise wei scales: per-expert and per-column
	    std::vector<float> wei_scales(num_experts * N);
	    for (int32_t i = 0; i < num_experts * N; ++i)
	        wei_scales[i] = 0.9f + (i % 200) / 1000.0f;
	
	    :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>` wei_scales_md(
	            {num_experts, N}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto wei_scales_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(wei_scales_md, eng);
	    write_to_dnnl_memory(wei_scales.data(), wei_scales_mem);
	    matmul_attr.set_scales_mask(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, (1 << 0) | (1 << 2));
	
	    // Create matmul primitive descriptor and the primitive
	    auto matmul_pd = :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(
	            eng, src_md, weights_md, dst_md, matmul_attr);
	    auto matmul_prim = :ref:`matmul <doxid-structdnnl_1_1matmul>`(matmul_pd);
	
	    // Execute the primitive
	    matmul_prim.execute(engine_stream,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, weights_mem},
	                    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_scales_mem},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_scales_mem}});
	
	    // Wait for completion
	    engine_stream.wait();
	}
	
	int main(int argc, char **argv) {
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv);
	    return handle_example_errors(grouped_matmul_example, engine_kind);
	}
