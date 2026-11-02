.. index:: pair: page; Matrix Multiplication with f8 Quantization
.. _doxid-matmul_f8_quantization_cpp:

Matrix Multiplication with f8 Quantization
==========================================

C++ API example demonstrating how to use f8_e5m2 and f8_e4m3 data types for :ref:`MatMul <doxid-dev_guide_matmul>` with scaling for quantization.

Specification of f8 Formats:

* f8_e5m2 : 1 sign + 5 exponent + 2 mantissa bits, max value is 57,344.

* f8_e4m3 : 1 sign + 4 exponent + 3 mantissa bits, max value is 448.

Concepts:

* f8 quantization.
  
  * f8_e5m2 and f8_e4m3 data type conversion from f32 is done using :ref:`Reorder primitive <doxid-dev_guide_reorder>` with simple scaling factors.

* Matrix multiplication with f8 inputs and f32 output.
  
  * Scaling is done using :ref:`dnnl::primitive_attr::set_scales_mask() <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`.

.. warning:: 

   This example uses a naive quantization approach that computes a single scaling factor per matrix based on maximum absolute values in the data. Real-world workloads require more complex quantization schemes for optimal results.
   
   


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
	#include <limits>
	#include <numeric>
	#include <stdexcept>
	#include <string>
	#include <vector>
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	float decode_f8_e4m3(uint8_t f8_val) {
	    if (f8_val == 0) return 0.0f;
	
	    // Extract bit components: f8_e4m3 format is S EEEE MMM (bit 7 to 0)
	    const uint8_t sign = (f8_val >> 7) & 0x1; // Bit 7: sign
	    const uint8_t exp = (f8_val >> 3) & 0xF; // Bits 6-3: 4-bit exponent
	    const uint8_t mant = f8_val & 0x7; // Bits 2-0: 3-bit mantissa
	
	    // Only exp=15, mant=7 is NaN (no infinity)
	    if (exp == 15 && mant == 7) {
	        return std::numeric_limits<float>::quiet_NaN();
	    }
	
	    float result;
	    if (exp == 0) {
	        // Denormal: 0.mant * 2^(-6)
	        result = (float)mant / 8.0f * powf(2.0f, -6);
	    } else {
	        // Normal: (1 + mant/2^(3)) * 2^(exp-7)
	        result = (1.0f + (float)mant / 8.0f) * powf(2.0f, (int)exp - 7);
	    }
	
	    return sign ? -result : result;
	}
	
	float decode_f8_e5m2(uint8_t f8_val) {
	    if (f8_val == 0) return 0.0f;
	
	    // Extract bit components: f8_e5m2 format is S EEEEE MM (bit 7 to 0)
	    const uint8_t sign = (f8_val >> 7) & 0x1; // Bit 7: sign
	    const uint8_t exp = (f8_val >> 2) & 0x1F; // Bits 6-2: 5-bit exponent
	    const uint8_t mant = f8_val & 0x3; // Bits 1-0: 2-bit mantissa
	
	    // Handle special cases (infinity and NaN)
	    if (exp == 31) {
	        if (mant == 0) {
	            return (sign ? -1.0f : 1.0f) * INFINITY; // Infinity
	        } else {
	            return std::numeric_limits<float>::quiet_NaN(); // NaN
	        }
	    }
	
	    float result;
	    if (exp == 0) {
	        // Denormal: 0.mant * 2^(-14)
	        result = (float)mant / 4.0f * powf(2.0f, -14);
	    } else {
	        // Normal: (1 + mant/2^(2)) * 2^(exp-15)
	        result = (1.0f + (float)mant / 4.0f) * powf(2.0f, (int)exp - 15);
	    }
	
	    return sign ? -result : result;
	}
	
	std::string get_f8_type_name(:ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` dt) {
	    switch (dt) {
	        case :ref:`memory::data_type::f8_e5m2 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea12ad5ee1ad075296bc5566a2d366678c>`: return "f8_e5m2";
	        case :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`: return "f8_e4m3";
	        default: return "Unsupported data type";
	    }
	}
	
	float return_max_value(:ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` dt) {
	    switch (dt) {
	        case :ref:`memory::data_type::f8_e5m2 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea12ad5ee1ad075296bc5566a2d366678c>`:
	            // f8_e5m2: 1 sign bit + 5 bit exponent (bias=15) + 2 bit mantissa
	            // Per OCP f8 spec: infinity = 11111.00, NaN = 11111.{01, 10, 11}
	            // Max: exponent=30, mantissa=11 (in binary) -> 1.75 × 2^(30-15) = 57344
	            return 57344.0f;
	        case :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`:
	            // f8_e4m3: 1 sign bit + 4 bit exponent (bias=7) + 3 bit mantissa
	            // Per OCP f8 spec: no infinity, NaN = 1111.111
	            // Max: exponent=15, mantissa=110 (in binary) -> 1.75 × 2^(15-7) = 448
	            return 448.0f;
	        default: throw std::invalid_argument("Unsupported data type");
	    }
	}
	
	float compute_naive_quantization(const float *data, size_t size,
	        :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` dst_type, const std::string &label) {
	    if (dst_type != :ref:`memory::data_type::f8_e5m2 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea12ad5ee1ad075296bc5566a2d366678c>`
	            && dst_type != :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`) {
	        throw std::invalid_argument("Unsupported data type");
	    }
	
	    // Find the maximum absolute value in the data
	    float max_abs = 0.0f;
	    for (size_t i = 0; i < size; ++i) {
	        max_abs = std::max(max_abs, std::abs(data[i]));
	    }
	
	    // Get theoretical maximum value for the target f8 format
	    float f8_max = return_max_value(dst_type);
	
	    // Only apply scaling if values exceed the f8 range
	    float scale;
	    if (max_abs <= f8_max) {
	        scale = 1.0f;
	        std::cout << "  " << label << " fits in " << get_f8_type_name(dst_type)
	                  << " (max=" << max_abs << ", f8_max=" << f8_max << ")"
	                  << std::endl;
	    } else {
	        scale = max_abs / f8_max;
	        std::cout << "  " << label << " max (" << max_abs << ") > "
	                  << get_f8_type_name(dst_type) << " max (" << f8_max
	                  << "), scaling: " << scale << std::endl;
	    }
	
	    return scale;
	}
	
	void perform_matmul_with_f8_quantization(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind,
	        :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` f8_type = :ref:`memory::data_type::f8_e5m2 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea12ad5ee1ad075296bc5566a2d366678c>`) {
	    if (f8_type != :ref:`memory::data_type::f8_e5m2 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea12ad5ee1ad075296bc5566a2d366678c>`
	            && f8_type != :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`) {
	        throw std::invalid_argument("Unsupported data type");
	    }
	
	    // Create execution dnnl::engine
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	
	    // Create dnnl::stream
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	
	    // Matrix dimensions for A * B = C
	    const int M = 4, K = 8, N = 4;
	
	    std::cout << get_f8_type_name(f8_type)
	              << " Quantization Example:" << std::endl;
	    std::cout << "  Matrix dimensions: A(" << M << "x" << K << ") * B(" << K
	              << "x" << N << ") = C(" << M << "x" << N << ")" << std::endl;
	
	    // Initialize input data with float values, and fill matrices with
	    // sample data to demonstrate scaling behavior.
	    // Source: values within f8_e4m3 range (< 448) - should not need scaling for E4M3.
	    // Weights: values exceeding f8_e4m3 range (> 448) - will need scaling for E4M3.
	    std::vector<float> src_f32(M * K);
	    std::vector<float> weights_f32(K * N);
	    std::iota(src_f32.begin(), src_f32.end(),
	            100.0f); // Each value is 100+ (fits in both formats)
	    std::iota(weights_f32.begin(), weights_f32.end(),
	            450.0f); // Each value is 450+ (exceeds f8_e4m3 max of 448)
	
	    // Create memory for inputs and outputs in f32 format
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            {M, K}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto :ref:`weights_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a06ba7b00a8c95dcf3a90e16d00eeb0e9>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            {K, N}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            {M, N}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, eng);
	    write_to_dnnl_memory(src_f32.data(), src_mem);
	    auto weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(weights_md, eng);
	    write_to_dnnl_memory(weights_f32.data(), weights_mem);
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_md, eng);
	
	    // Create f8 memory descriptors for quantized data
	    auto src_f8_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({M, K}, f8_type, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	    auto weights_f8_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({K, N}, f8_type, :ref:`memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>`);
	
	    auto src_f8_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_f8_md, eng);
	    auto weights_f8_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(weights_f8_md, eng);
	
	    // Step 1: Compute scaling factors for quantization
	    std::cout << "\nStep 1: Computing scaling factors for f32 to "
	              << get_f8_type_name(f8_type) << " quantization" << std::endl;
	
	    float src_scale = compute_naive_quantization(
	            src_f32.data(), src_f32.size(), f8_type, "Source");
	    float weights_scale = compute_naive_quantization(
	            weights_f32.data(), weights_f32.size(), f8_type, "Weights");
	
	    // Step 2: Quantize f32 to f8 format with scaling
	    std::cout << "\nStep 2: Quantizing f32 data to "
	              << get_f8_type_name(f8_type) << " format with scaling"
	              << std::endl;
	
	    // Create memory for scales
	    auto src_scale_mem
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{1}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::x <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9dd4e461268c8034f5c8564e155c67a6>`}, eng);
	    write_to_dnnl_memory(&src_scale, src_scale_mem);
	
	    auto weights_scale_mem
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{1}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::x <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa9dd4e461268c8034f5c8564e155c67a6>`}, eng);
	    write_to_dnnl_memory(&weights_scale, weights_scale_mem);
	
	    // Create reorder primitives with scaling attributes
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` src_attr, weights_attr;
	    src_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, 0);
	    weights_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, 0);
	
	    // Check if f8 reorders are supported on this platform
	    try {
	        :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(eng, src_md, eng, src_f8_md, src_attr);
	        :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	                eng, weights_md, eng, weights_f8_md, weights_attr);
	    } catch (:ref:`error <doxid-structdnnl_1_1error>` &e) {
	        if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`)
	            throw example_allows_unimplemented {
	                    "No f8 reorder implementation is available for this "
	                    "platform.\n"
	                    "Please refer to the developer guide for details."};
	
	        // on any other error just re-throw
	        throw;
	    }
	
	    auto reorder_src_pd
	            = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(eng, src_md, eng, src_f8_md, src_attr);
	    auto reorder_weights_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            eng, weights_md, eng, weights_f8_md, weights_attr);
	
	    auto reorder_src = :ref:`reorder <doxid-structdnnl_1_1reorder>`(reorder_src_pd);
	    auto reorder_weights = :ref:`reorder <doxid-structdnnl_1_1reorder>`(reorder_weights_pd);
	
	    // Execute reorders with scaling
	    reorder_src.execute(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_f8_mem},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_scale_mem}});
	    reorder_weights.execute(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, weights_mem}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, weights_f8_mem},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, weights_scale_mem}});
	    s.wait();
	
	    // Show key quantization results
	    std::cout << "  Quantization summary:" << std::endl;
	    std::cout << "    Scaling factors: src=" << src_scale
	              << ", weights=" << weights_scale << std::endl;
	
	    // Read a few f8 values to demonstrate quantization
	    std::vector<uint8_t> weights_f8_data(K * N);
	    read_from_dnnl_memory(weights_f8_data.data(), weights_f8_mem);
	
	    auto decode_f8 = (f8_type == :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`) ? decode_f8_e4m3
	                                                             : decode_f8_e5m2;
	    std::cout << "    Sample: f32=" << weights_f32[0]
	              << " -> f8=" << (int)weights_f8_data[0]
	              << " -> decoded=" << decode_f8(weights_f8_data[0])
	              << " (f8 as float)"
	              << " -> final=" << decode_f8(weights_f8_data[0]) * weights_scale
	              << " (dequantized)" << std::endl;
	
	    std::cout << "  Successfully quantized inputs to "
	              << get_f8_type_name(f8_type) << " format with scaling"
	              << std::endl;
	
	    // Step 3: Matrix multiplication with f8
	    std::cout << "\nStep 3: Performing matrix multiplication with "
	              << get_f8_type_name(f8_type) << " inputs" << std::endl;
	
	    // Create matmul with dequantization attributes
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` matmul_attr;
	    matmul_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, 0);
	    matmul_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, 0);
	
	    // Check if f8 matmul is supported on this platform
	    try {
	        :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(
	                eng, src_f8_md, weights_f8_md, dst_md, matmul_attr);
	    } catch (:ref:`error <doxid-structdnnl_1_1error>` &e) {
	        if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`)
	            throw example_allows_unimplemented {
	                    "No f8 matmul implementation is available for this "
	                    "platform.\n"
	                    "Please refer to the developer guide for details."};
	
	        // on any other error just re-throw
	        throw;
	    }
	
	    auto matmul_pd = :ref:`matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(
	            eng, src_f8_md, weights_f8_md, dst_md, matmul_attr);
	    auto matmul_prim = :ref:`matmul <doxid-structdnnl_1_1matmul>`(matmul_pd);
	
	    // Execute matmul with dequantization
	    matmul_prim.execute(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_f8_mem}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, weights_f8_mem},
	                    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_scale_mem},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`,
	                            weights_scale_mem}});
	    s.wait();
	
	    std::cout << "  Matrix multiplication completed successfully" << std::endl;
	
	    // Read result for validation
	    std::vector<float> dst_result(M * N);
	    read_from_dnnl_memory(dst_result.data(), dst_mem);
	
	    // Step 4: Validate results
	    std::cout << "\nStep 4: Validating results against f32 reference"
	              << std::endl;
	
	    // Compute reference result with f32 precision
	    std::vector<float> ref_result(M * N, 0.0f);
	    for (int m = 0; m < M; ++m) {
	        for (int n = 0; n < N; ++n) {
	            for (int k = 0; k < K; ++k) {
	                ref_result[m * N + n]
	                        += src_f32[m * K + k] * weights_f32[k * N + n];
	            }
	        }
	    }
	
	    // Calculate relative error between f8 and f32 results
	    float max_rel_error = 0.0f;
	
	    // Use the dst_result vector that we already read instead of direct memory access
	    // This ensures compatibility with GPU where get_data_handle() may not work
	    for (int i = 0; i < M * N; ++i) {
	        if (std::abs(ref_result[i]) > 1e-6f) {
	            float rel_error = std::abs(dst_result[i] - ref_result[i])
	                    / std::abs(ref_result[i]);
	            max_rel_error = std::max(max_rel_error, rel_error);
	        }
	    }
	
	    // For example purposes set tolerance to 15%
	    const float tolerance = 0.15f;
	    bool validation_passed = max_rel_error < tolerance;
	
	    std::cout << "  Validation " << (validation_passed ? "PASSED" : "FAILED")
	              << " (max relative error: " << max_rel_error * 100.0f
	              << "%, tolerance: " << tolerance * 100.0f << "%)" << std::endl;
	
	    if (!validation_passed) {
	        throw :ref:`std::runtime_error <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808>`(
	                "  Validation failed: results exceed expected tolerance");
	    }
	}
	
	void run_f8_tutorials(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    // Sample 1: f8_e5m2
	    std::cout << "Sample 1: f8_e5m2 Format" << std::endl;
	    std::cout << "==========================" << std::endl;
	    perform_matmul_with_f8_quantization(
	            engine_kind, :ref:`memory::data_type::f8_e5m2 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea12ad5ee1ad075296bc5566a2d366678c>`);
	    std::cout << "f8_e5m2 tutorial completed successfully" << std::endl
	              << std::endl;
	
	    // Sample 2: f8_e4m3
	    std::cout << "Sample 2: f8_e4m3 Format" << std::endl;
	    std::cout << "==========================" << std::endl;
	    perform_matmul_with_f8_quantization(
	            engine_kind, :ref:`memory::data_type::f8_e4m3 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf5ede3d43b879551314bbb05684fa758>`);
	    std::cout << "f8_e4m3 tutorial completed successfully" << std::endl
	              << std::endl;
	}
	
	int main(int argc, char **argv) {
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv);
	    return handle_example_errors(run_f8_tutorials, engine_kind);
	}

