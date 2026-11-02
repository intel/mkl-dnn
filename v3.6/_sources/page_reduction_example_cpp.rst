.. index:: pair: page; Reduction Primitive Example
.. _doxid-reduction_example_cpp:

Reduction Primitive Example
===========================

This C++ API example demonstrates how to create and execute a :ref:`Reduction <doxid-dev_guide_reduction>` primitive.

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
	
	
	#include <cmath>
	
	#include "example_utils.hpp"
	#include "oneapi/dnnl/dnnl.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	void reduction_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
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
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_dims = {1, IC, 1, 1};
	
	    // Allocate buffers.
	    std::vector<float> src_data(product(src_dims));
	    std::vector<float> dst_data(product(dst_dims));
	
	    // Initialize src tensor.
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	
	    // Create src and dst memory descriptors and memory objects.
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(dst_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_data.data(), src_mem);
	
	    // Create primitive descriptor.
	    auto reduction_pd = :ref:`reduction::primitive_desc <doxid-structdnnl_1_1reduction_1_1primitive__desc>`(
	            :ref:`engine <doxid-structdnnl_1_1engine>`, :ref:`algorithm::reduction_sum <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640af710a5a162f3ac1c005876b5c96a6087>`, src_md, dst_md, 0.f, 0.f);
	
	    // Create the primitive.
	    auto reduction_prim = :ref:`reduction <doxid-structdnnl_1_1reduction>`(reduction_pd);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> reduction_args;
	    reduction_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem});
	    reduction_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem});
	
	    // Primitive execution: Reduction (Sum).
	    reduction_prim.execute(engine_stream, reduction_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_data.data(), dst_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            reduction_example, parse_engine_kind(argc, argv));
	}

