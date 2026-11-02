.. index:: pair: example; memory_format_propagation.cpp
.. _doxid-memory_format_propagation_8cpp-example:

memory_format_propagation.cpp
=============================

This example demonstrates memory format propagation, which is critical for deep learning applications performance. Annotated version: :ref:`Memory Format Propagation <doxid-memory_format_propagation_cpp>`

This example demonstrates memory format propagation, which is critical for deep learning applications performance. Annotated version: :ref:`Memory Format Propagation <doxid-memory_format_propagation_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2019-2022 Intel Corporation
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
	
	
	#include <iostream>
	#include <sstream>
	#include <string>
	
	
	#include "oneapi/dnnl/dnnl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	void memory_format_propagation_tutorial(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    // [Initialize engine and stream]
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	    // [Initialize engine and stream]
	
	    // [Create placeholder memory descriptors]
	    // Tensor and kernel dimensions. We use the same 3x3 kernel with padding=1
	    // for both convolution and pooling primitives, which means that the
	    // activation tensor shapes do not change.
	    const int N = 1, H = 14, W = 14, IC = 128, OC = 256, KH = 3, KW = 3;
	    auto conv_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({N, IC, H, W}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	            :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` // let convolution choose memory format
	    );
	    auto conv_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	            {OC, IC, KH, KW}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	            :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` // let convolution choose memory format
	    );
	    auto conv_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({N, OC, H, W}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	            :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` // let convolution choose memory format
	    );
	    const auto &pool_dst_md = conv_dst_md; // shape does not change
	    // [Create placeholder memory descriptors]
	
	    // [Create convolution and pooling primitive descriptors]
	    auto conv_pd = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(
	            eng, :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_auto <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640acfdececd63a8bc0cfe1021ad614e2ded>`,
	            conv_src_md, conv_weights_md,
	            conv_dst_md, // shape information
	            {1, 1}, // strides
	            {1, 1}, {1, 1} // left and right padding
	    );
	
	    auto pool_pd
	            = :ref:`pooling_forward::primitive_desc <doxid-structdnnl_1_1pooling__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	                    :ref:`algorithm::pooling_max <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`, conv_pd.dst_desc(),
	                    pool_dst_md, // shape information
	                    {1, 1}, {KH, KW}, // strides and kernel
	                    {0, 0}, // dilation
	                    {1, 1}, {1, 1} // left and right padding
	            );
	    // [Create convolution and pooling primitive descriptors]
	
	    // [Create source and destination memory objects]
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(
	            {{N, IC, H, W}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`},
	            eng);
	    auto weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({{OC, IC, KH, KW}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	                                      :ref:`memory::format_tag::oihw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa14b72a467aeefa06a5cb802ec4a7743c>`},
	            eng);
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(
	            {{N, OC, H, W}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`},
	            eng);
	    // [Create source and destination memory objects]
	
	    // [Determine if source needs to be reordered]
	    bool need_reorder_src = conv_pd.src_desc() != src_mem.get_desc();
	    // [Determine if source needs to be reordered]
	
	    // [Determine if weights and destination need to be reordered]
	    bool need_reorder_weights
	            = conv_pd.weights_desc() != weights_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`();
	    bool need_reorder_dst = conv_pd.dst_desc() != dst_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`();
	    // [Determine if weights and destination need to be reordered]
	
	    // [Allocate intermediate buffers if necessary]
	    auto conv_src_mem
	            = need_reorder_src ? :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.src_desc(), eng) : src_mem;
	    auto conv_weights_mem = need_reorder_weights
	            ? :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.weights_desc(), eng)
	            : weights_mem;
	    auto conv_dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.dst_desc(), eng);
	    auto pool_dst_mem
	            = need_reorder_dst ? :ref:`memory <doxid-structdnnl_1_1memory>`(pool_pd.:ref:`dst_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc_1af0496636db4f6284e9225baa45b85cfa>`(), eng) : dst_mem;
	    // [Allocate intermediate buffers if necessary]
	
	    // [Perform reorders for source data if necessary]
	    if (need_reorder_src) {
	        auto reorder_src = :ref:`reorder <doxid-structdnnl_1_1reorder>`(src_mem, conv_src_mem);
	        reorder_src.execute(
	                s, {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, src_mem}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_src_mem}});
	        s.wait(); // wait for the reorder to complete
	    }
	
	    if (need_reorder_weights) {
	        auto reorder_weights = :ref:`reorder <doxid-structdnnl_1_1reorder>`(weights_mem, conv_weights_mem);
	        reorder_weights.execute(s,
	                {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, weights_mem},
	                        {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_weights_mem}});
	        s.wait(); // wait for the reorder to complete
	    }
	    // [Perform reorders for source data if necessary]
	
	    // [Create and execute convolution and pooling primitives]
	    auto conv_scratchpad_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.scratchpad_desc(), eng);
	    auto conv = :ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv_pd);
	    conv.execute(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src_mem}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_weights_mem},
	                    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst_mem}});
	    auto pool_scratchpad_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(pool_pd.:ref:`scratchpad_desc <doxid-structdnnl_1_1primitive__desc__base_1a6238358ec03afd57fb20dffa65b48d2f>`(), eng);
	    auto pool = :ref:`pooling_forward <doxid-structdnnl_1_1pooling__forward>`(pool_pd);
	    pool.execute(
	            s, {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_dst_mem}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool_dst_mem}});
	    s.wait();
	    // [Create and execute convolution and pooling primitives]
	
	    // [Reorder destination data if necessary]
	    if (need_reorder_dst) {
	        auto reorder_dst = :ref:`reorder <doxid-structdnnl_1_1reorder>`(pool_dst_mem, dst_mem);
	        reorder_dst.execute(
	                s, {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, pool_dst_mem}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, dst_mem}});
	        s.wait();
	    }
	    // [Reorder destination data if necessary]
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            memory_format_propagation_tutorial, parse_engine_kind(argc, argv));
	}
