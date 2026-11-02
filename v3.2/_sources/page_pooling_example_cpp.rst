.. index:: pair: page; Pooling Primitive Example
.. _doxid-pooling_example_cpp:

Pooling Primitive Example
=========================

This C++ API example demonstrates how to create and execute a :ref:`Pooling <doxid-dev_guide_pooling>` primitive in forward training propagation mode.

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
	
	void pooling_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3, // batch size
	            IC = 3, // input channels
	            IH = 27, // input tensor height
	            IW = 27, // input tensor width
	            KH = 11, // kernel height
	            KW = 11, // kernel width
	            PH_L = 0, // height padding: left
	            PH_R = 0, // height padding: right
	            PW_L = 0, // width padding: left
	            PW_R = 0, // width padding: right
	            SH = 4, // height-wise stride
	            SW = 4, // width-wise stride
	            DH = 1, // height-wise dilation
	            DW = 1; // width-wise dilation
	
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` OH = (IH - ((KH - 1) * DH + KH) + PH_L + PH_R) / SH + 1;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` OW = (IW - ((KW - 1) * DW + KW) + PW_L + PW_R) / SW + 1;
	
	    // Source (src) and destination (dst) tensors dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {N, IC, IH, IW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_dims = {N, IC, OH, OW};
	
	    // Kernel dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` kernel_dims = {KH, KW};
	
	    // Strides, padding dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` strides_dims = {SH, SW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` padding_dims_l = {PH_L, PW_L};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` padding_dims_r = {PH_R, PW_R};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dilation = {DH, DW};
	
	    // Allocate buffers.
	    std::vector<float> src_data(product(src_dims));
	    std::vector<float> dst_data(product(dst_dims));
	
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	
	    // Create memory descriptors and memory objects for src and dst.
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, dt::f32, tag::nchw);
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(dst_dims, dt::f32, tag::nchw);
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_data.data(), src_mem);
	
	    // Create primitive descriptor.
	    auto pooling_pd = :ref:`pooling_forward::primitive_desc <doxid-structdnnl_1_1pooling__forward_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`,
	            :ref:`prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, :ref:`algorithm::pooling_max <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`, src_md, dst_md,
	            strides_dims, kernel_dims, dilation, padding_dims_l,
	            padding_dims_r);
	
	    // Create workspace memory objects using memory descriptor created by the
	    // primitive descriptor.
	    // NOTE: Here, the workspace is required to save the indices where maximum
	    // was found, and is used in backward pooling to perform upsampling.
	    auto workspace_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(pooling_pd.workspace_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create the primitive.
	    auto pooling_prim = :ref:`pooling_forward <doxid-structdnnl_1_1pooling__forward>`(pooling_pd);
	
	    // Primitive arguments. Set up in-place execution by assigning src as DST.
	    std::unordered_map<int, memory> pooling_args;
	    pooling_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem});
	    pooling_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem});
	    pooling_args.insert({:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, workspace_mem});
	
	    // Primitive execution: pooling.
	    pooling_prim.execute(engine_stream, pooling_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_data.data(), dst_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            pooling_example, parse_engine_kind(argc, argv));
	}

