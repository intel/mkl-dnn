.. index:: pair: example; bnorm_u8_via_binary_postops.cpp
.. _doxid-bnorm_u8_via_binary_postops_8cpp-example:

bnorm_u8_via_binary_postops.cpp
===============================

The example implements the Batch normalization u8 via the following operations: binary_sub(src, mean), binary_div(tmp_dst, variance), binary_mul(tmp_dst, scale), binary_add(tmp_dst, shift). Annotated version: :ref:`Bnorm u8 by binary post-ops example <doxid-bnorm_u8_via_binary_postops_cpp>`

The example implements the Batch normalization u8 via the following operations: binary_sub(src, mean), binary_div(tmp_dst, variance), binary_mul(tmp_dst, scale), binary_add(tmp_dst, shift). Annotated version: :ref:`Bnorm u8 by binary post-ops example <doxid-bnorm_u8_via_binary_postops_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2020-2022 Intel Corporation
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
	
	void bnorm_u8_via_binary_postops(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3, // batch size
	            IC = 3, // channels
	            IH = 150, // tensor height
	            IW = 150; // tensor width
	
	    // Tensors dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {N, IC, IH, IW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` params_dims = {1, IC, 1, 1};
	
	    // Allocate buffers.
	    std::vector<float> src_data(product(src_dims));
	    std::vector<float> mean_data(product(params_dims));
	    std::vector<float> variance_data(product(params_dims));
	    std::vector<float> scale_data(product(params_dims));
	    std::vector<float> shift_data(product(params_dims));
	    std::vector<float> oscale_data(product(params_dims));
	
	    // Initialize
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	    std::generate(mean_data.begin(), mean_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 2.f);
	    });
	    std::generate(variance_data.begin(), variance_data.end(), []() {
	        static int i = 0;
	        float value = std::abs(std::sin(i++ * 4.f));
	        // Avoid division by zero. Variance should be positive.
	        return value == 0.f ? 1.f : value;
	    });
	    std::generate(scale_data.begin(), scale_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 6.f);
	    });
	    std::generate(shift_data.begin(), shift_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 8.f);
	    });
	    std::generate(
	            oscale_data.begin(), oscale_data.end(), []() { return 0.5f; });
	
	    // Create descriptors.
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, dt::u8, tag::nhwc);
	    auto mean_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(params_dims, dt::f32, tag::nhwc);
	    auto variance_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(params_dims, dt::f32, tag::nhwc);
	    auto scale_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(params_dims, dt::f32, tag::nhwc);
	    auto shift_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(params_dims, dt::f32, tag::nhwc);
	    auto oscale_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(params_dims, dt::f32, tag::nhwc);
	
	    // Create src memory objects.
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto mean_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(mean_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto variance_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(variance_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto scale_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(scale_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto shift_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(shift_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto oscale_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(oscale_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_data.data(), src_mem);
	    write_to_dnnl_memory(mean_data.data(), mean_mem);
	    write_to_dnnl_memory(variance_data.data(), variance_mem);
	    write_to_dnnl_memory(scale_data.data(), scale_mem);
	    write_to_dnnl_memory(shift_data.data(), shift_mem);
	    write_to_dnnl_memory(oscale_data.data(), oscale_mem);
	
	    // Bnorm operation with scale and shift
	    :ref:`post_ops <doxid-structdnnl_1_1post__ops>` binary_ops;
	    // dst_tmp = dst_tmp / variance
	    binary_ops.:ref:`append_binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`(:ref:`algorithm::binary_div <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a2835085341c109a886106f1b671aff71>`, variance_md);
	    // dst_tmp = dst_tmp * scale
	    binary_ops.append_binary(:ref:`algorithm::binary_mul <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0905fc5c22e79a8eed0988681eb6a0ae>`, scale_md);
	    // dst_tmp = dst_tmp + shift
	    binary_ops.append_binary(:ref:`algorithm::binary_add <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab2c3faf084cf82b5603946995f637b35>`, shift_md);
	    // dst = dst_tmp * output_scale (only for re-quantization)
	    binary_ops.append_binary(:ref:`algorithm::binary_mul <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0905fc5c22e79a8eed0988681eb6a0ae>`, oscale_md);
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` binary_attr;
	    binary_attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(binary_ops);
	
	    // Create primitive descriptor.
	    // dst_tmp = src - mean
	    auto binary_pd = :ref:`binary::primitive_desc <doxid-structdnnl_1_1binary_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`, :ref:`algorithm::binary_sub <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a979309f9436f7ebfa278b0ce682dd706>`,
	            src_md, mean_md, src_md, binary_attr);
	
	    // Create the primitive.
	    auto binary_prim = :ref:`binary <doxid-structdnnl_1_1binary>`(binary_pd);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> binary_args;
	    binary_args.insert({:ref:`DNNL_ARG_SRC_0 <doxid-group__dnnl__api__primitives__common_1ga53dc83e64489cd69bd82c1c2025eb5bd>`, src_mem});
	    binary_args.insert({:ref:`DNNL_ARG_SRC_1 <doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`, mean_mem});
	    // In-place mode (dst is src)
	    binary_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_mem});
	    binary_args.insert(
	            {:ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>`(0) | :ref:`DNNL_ARG_SRC_1 <doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`, variance_mem});
	    binary_args.insert(
	            {:ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>`(1) | :ref:`DNNL_ARG_SRC_1 <doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`, scale_mem});
	    binary_args.insert(
	            {:ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>`(2) | :ref:`DNNL_ARG_SRC_1 <doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`, shift_mem});
	    binary_args.insert(
	            {:ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>`(3) | :ref:`DNNL_ARG_SRC_1 <doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`, oscale_mem});
	
	    // Primitive execution
	    binary_prim.execute(engine_stream, binary_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(src_data.data(), src_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            bnorm_u8_via_binary_postops, parse_engine_kind(argc, argv));
	}
