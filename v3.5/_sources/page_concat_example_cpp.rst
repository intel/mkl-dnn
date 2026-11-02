.. index:: pair: page; Concat Primitive Example
.. _doxid-concat_example_cpp:

Concat Primitive Example
========================

This C++ API example demonstrates how to create and execute a :ref:`Concat <doxid-dev_guide_concat>` primitive.

Key optimizations included in this example:

* Identical source (src) memory formats.

* Creation of optimized memory format for destination (dst) from the primitive descriptor

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
	
	void concat_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3, // batch size
	            IC = 3, // channels
	            IH = 120, // tensor height
	            IW = 120; // tensor width
	
	    // Number of source (src) tensors.
	    const int num_src = 10;
	
	    // Concatenation axis.
	    const int axis = 1;
	
	    // src tensors dimensions
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {N, IC, IH, IW};
	
	    // Allocate buffers.
	    std::vector<float> src_data(product(src_dims));
	
	    // Initialize src.
	    // NOTE: In this example, the same src memory buffer is used to demonstrate
	    // concatenation for simplicity
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	
	    // Create a memory descriptor and memory object for each src tensor.
	    std::vector<memory::desc> src_mds;
	    std::vector<memory> src_mems;
	
	    for (int n = 0; n < num_src; ++n) {
	        auto md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	        auto mem = :ref:`memory <doxid-structdnnl_1_1memory>`(md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	        // Write data to memory object's handle.
	        write_to_dnnl_memory(src_data.data(), mem);
	
	        src_mds.push_back(md);
	        src_mems.push_back(mem);
	    }
	
	    // Create primitive descriptor.
	    auto concat_pd = :ref:`concat::primitive_desc <doxid-structdnnl_1_1concat_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`, axis, src_mds);
	
	    // Create destination (dst) memory object using the memory descriptor
	    // created by the primitive.
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(concat_pd.dst_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create the primitive.
	    auto concat_prim = :ref:`concat <doxid-structdnnl_1_1concat>`(concat_pd);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> concat_args;
	    for (int n = 0; n < num_src; ++n)
	        concat_args.insert({DNNL_ARG_MULTIPLE_SRC + n, src_mems[n]});
	    concat_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem});
	
	    // Primitive execution: concatenation.
	    concat_prim.execute(engine_stream, concat_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(concat_example, parse_engine_kind(argc, argv));
	}

