.. index:: pair: page; Primitive Example
.. _doxid-prelu_example_cpp:

Primitive Example
=================

This C++ API example demonstrates how to create and execute an :ref:`PReLU <doxid-dev_guide_prelu>` primitive in forward training propagation mode.

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
	#include <string>
	#include <vector>
	
	#include "dnnl.hpp"
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	void prelu_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3, // batch size
	            IC = 3, // channels
	            IH = 227, // tensor height
	            IW = 227; // tensor width
	
	    // Source (src), weights and destination (dst) tensors dimensions.
	    const :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {N, IC, IH, IW};
	    const :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` weights_dims = {N, IC, IH, IW};
	    const :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_dims = {N, IC, IH, IW};
	
	    // Allocate buffers. In this example, out-of-place primitive execution is
	    // demonstrated since both src and dst are required for later backward
	    // propagation.
	    std::vector<float> src_data(product(src_dims));
	    std::vector<float> weights_data(product(weights_dims));
	    std::vector<float> dst_data(product(dst_dims));
	
	    // Initialize src tensor.
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	
	    // Initialize weights tensor.
	    std::fill(weights_data.begin(), weights_data.end(), 0.3f);
	
	    // Create memory objects for tensor data (src, weights, dst). In this
	    // example, NCHW layout is assumed for src, weights and dst.
	    auto user_src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto user_weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({weights_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto user_dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({dst_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create memory descriptors for the primitive. Src tag is set
	    // to match src memory object. Setting weights tag to format_tag::any
	    // enables the PReLU primitive to choose memory layout for an optimized
	    // primitive implementation, and that layout may differ from the one
	    // provided by the user.
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	    auto :ref:`weights_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a06ba7b00a8c95dcf3a90e16d00eeb0e9>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(weights_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_data.data(), user_src_mem);
	    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
	
	    // Create primitive descriptor.
	    auto prelu_pd = :ref:`prelu_forward::primitive_desc <doxid-structdnnl_1_1prelu__forward_1_1primitive__desc>`(
	            :ref:`engine <doxid-structdnnl_1_1engine>`, :ref:`prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, src_md, weights_md, dst_md);
	
	    // For now, assume that the weights memory layout generated
	    // by the primitive and the one provided by the user are identical.
	    auto prelu_weights_mem = user_weights_mem;
	
	    // Reorder the data in case the weights memory layout generated by
	    // the primitive and the one provided by the user are different. In this
	    // case, we create additional memory object with internal buffers that will
	    // contain the reordered data.
	    if (prelu_pd.weights_desc() != user_weights_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        prelu_weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(prelu_pd.weights_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_mem, prelu_weights_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_weights_mem, prelu_weights_mem);
	    }
	
	    // Create the primitive.
	    auto prelu_prim = :ref:`prelu_forward <doxid-structdnnl_1_1prelu__forward>`(prelu_pd);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> prelu_args;
	    prelu_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, user_src_mem});
	    prelu_args.insert({:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, prelu_weights_mem});
	    prelu_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, user_dst_mem});
	
	    // Primitive execution: PReLU.
	    prelu_prim.execute(engine_stream, prelu_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(prelu_example, parse_engine_kind(argc, argv));
	}

