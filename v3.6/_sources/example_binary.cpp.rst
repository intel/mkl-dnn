.. index:: pair: example; binary.cpp
.. _doxid-binary_8cpp-example:

binary.cpp
==========

Annotated version: :ref:`Binary Primitive Example <doxid-binary_example_cpp>`

Annotated version: :ref:`Binary Primitive Example <doxid-binary_example_cpp>`



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
	
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	void binary_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3, // batch size
	            IC = 3, // channels
	            IH = 150, // tensor height
	            IW = 150; // tensor width
	
	    // Source (src_0 and src_1) and destination (dst) tensors dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_0_dims = {N, IC, IH, IW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_1_dims = {N, IC, IH, 1};
	
	    // Allocate buffers.
	    std::vector<float> src_0_data(product(src_0_dims));
	    std::vector<float> src_1_data(product(src_1_dims));
	
	    // Initialize src_0 and src_1 (src).
	    std::generate(src_0_data.begin(), src_0_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	    std::generate(src_1_data.begin(), src_1_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 2.f);
	    });
	
	    // Create src and dst memory descriptors.
	    auto src_0_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_0_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	    auto src_1_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_1_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_0_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	
	    // Create src memory objects.
	    auto src_0_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_0_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto src_1_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_1_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_0_data.data(), src_0_mem);
	    write_to_dnnl_memory(src_1_data.data(), src_1_mem);
	
	    // Create primitive post-ops (ReLU).
	    const float alpha = 0.f;
	    const float beta = 0.f;
	    :ref:`post_ops <doxid-structdnnl_1_1post__ops>` binary_ops;
	    binary_ops.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(:ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, alpha, beta);
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` binary_attr;
	    binary_attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(binary_ops);
	
	    // Create primitive descriptor.
	    auto binary_pd = :ref:`binary::primitive_desc <doxid-structdnnl_1_1binary_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`, :ref:`algorithm::binary_mul <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a0905fc5c22e79a8eed0988681eb6a0ae>`,
	            src_0_md, src_1_md, dst_md, binary_attr);
	
	    // Create the primitive.
	    auto binary_prim = :ref:`binary <doxid-structdnnl_1_1binary>`(binary_pd);
	
	    // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
	    std::unordered_map<int, memory> binary_args;
	    binary_args.insert({:ref:`DNNL_ARG_SRC_0 <doxid-group__dnnl__api__primitives__common_1ga53dc83e64489cd69bd82c1c2025eb5bd>`, src_0_mem});
	    binary_args.insert({:ref:`DNNL_ARG_SRC_1 <doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`, src_1_mem});
	    binary_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_0_mem});
	
	    // Primitive execution: binary with ReLU.
	    binary_prim.execute(engine_stream, binary_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(src_0_data.data(), src_0_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(binary_example, parse_engine_kind(argc, argv));
	}
