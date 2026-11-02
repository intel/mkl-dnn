.. index:: pair: example; reorder.cpp
.. _doxid-reorder_8cpp-example:

reorder.cpp
===========

Annotated version: :ref:`Reorder Primitive Example <doxid-reorder_example_cpp>`

Annotated version: :ref:`Reorder Primitive Example <doxid-reorder_example_cpp>`



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
	
	void reorder_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3, // batch size
	            IC = 3, // channels
	            IH = 227, // tensor height
	            IW = 227; // tensor width
	
	    // Source (src) and destination (dst) tensors dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {N, IC, IH, IW};
	
	    // Allocate buffers.
	    std::vector<float> src_data(product(src_dims));
	    std::vector<int8_t> dst_data(product(src_dims));
	
	    // Initialize src tensor.
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	
	    // Create memory descriptors and memory objects for src and dst.
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, dt::s8, tag::nhwc);
	
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_data.data(), src_mem);
	
	    // Per-channel scales.
	    std::vector<float> scales(IC);
	    std::generate(scales.begin(), scales.end(), []() {
	        static int i = 0;
	        return 64.f + 5.f * i++;
	    });
	
	    // Dimension of the dst tensor where the output scales will be applied
	    const int ic_dim = 1;
	
	    // Create primitive post-ops (per-channel output scales)
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` reorder_attr;
	    reorder_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, 1 << ic_dim);
	    auto dst_scales_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({{IC}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    write_to_dnnl_memory(scales.data(), dst_scales_mem);
	
	    // Create primitive descriptor.
	    auto reorder_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            :ref:`engine <doxid-structdnnl_1_1engine>`, src_md, :ref:`engine <doxid-structdnnl_1_1engine>`, dst_md, reorder_attr);
	
	    // Create the primitive.
	    auto reorder_prim = :ref:`reorder <doxid-structdnnl_1_1reorder>`(reorder_pd);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> reorder_args;
	    reorder_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem});
	    reorder_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem});
	    reorder_args.insert({:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_scales_mem});
	
	    // Primitive execution: reorder with scaled sum.
	    reorder_prim.execute(engine_stream, reorder_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_data.data(), dst_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            reorder_example, parse_engine_kind(argc, argv));
	}
