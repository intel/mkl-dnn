.. index:: pair: page; Sum Primitive Example
.. _doxid-sum_example_cpp:

Sum Primitive Example
=====================

This C++ API example demonstrates how to create and execute a :ref:`Sum <doxid-dev_guide_sum>` primitive.

Key optimizations included in this example:

* Identical memory formats for source (src) and destination (dst) tensors.

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
	
	void sum_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
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
	    std::vector<float> dst_data(product(src_dims));
	
	    // Initialize src.
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	
	    // Number of src tensors.
	    const int num_src = 10;
	
	    // Scaling factors.
	    std::vector<float> scales(num_src);
	    std::generate(scales.begin(), scales.end(),
	            [](int n = 0) { return sin(float(n)); });
	
	    // Create an array of memory descriptors and memory objects for src tensors.
	    std::vector<memory::desc> :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>`;
	    std::vector<memory> src_mem;
	
	    for (int n = 0; n < num_src; ++n) {
	        auto md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, dt::f32, tag::nchw);
	        auto mem = :ref:`memory <doxid-structdnnl_1_1memory>`(md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	        // Write data to memory object's handle.
	        write_to_dnnl_memory(src_data.data(), mem);
	
	        :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>`.push_back(md);
	        src_mem.push_back(mem);
	    }
	
	    // Create primitive descriptor.
	    auto sum_pd = :ref:`sum::primitive_desc <doxid-structdnnl_1_1sum_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`, scales, src_md);
	
	    // Create the primitive.
	    auto sum_prim = :ref:`sum <doxid-structdnnl_1_1sum>`(sum_pd);
	
	    // Create memory object for dst.
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(sum_pd.dst_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> sum_args;
	    sum_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem});
	    for (int n = 0; n < num_src; ++n) {
	        sum_args.insert({:ref:`DNNL_ARG_MULTIPLE_SRC <doxid-group__dnnl__api__primitives__common_1ga1f0da423df3fb6853ddcbe6ffe964267>` + n, src_mem[n]});
	    }
	
	    // Primitive execution: sum.
	    sum_prim.execute(engine_stream, sum_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_data.data(), dst_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(sum_example, parse_engine_kind(argc, argv));
	}

