.. index:: pair: example; inner_product.cpp
.. _doxid-inner_product_8cpp-example:

inner_product.cpp
=================

Annotated version: :ref:`Inner Product Primitive Example <doxid-inner_product_example_cpp>`

Annotated version: :ref:`Inner Product Primitive Example <doxid-inner_product_example_cpp>`



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
	
	void inner_product_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3, // batch size
	            IC = 3, // input channels
	            IH = 227, // tensor height
	            IW = 227, // tensor width
	            OC = 96; // output channels
	
	    // Source (src), weights, bias, and destination (dst) tensors
	    // dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {N, IC, IH, IW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` weights_dims = {OC, IC, IH, IW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` bias_dims = {OC};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_dims = {N, OC};
	
	    // Allocate buffers.
	    std::vector<float> src_data(product(src_dims));
	    std::vector<float> weights_data(product(weights_dims));
	    std::vector<float> bias_data(OC);
	    std::vector<float> dst_data(product(dst_dims));
	
	    // Initialize src, weights, and bias tensors.
	    std::generate(src_data.begin(), src_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	    std::generate(weights_data.begin(), weights_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 2.f);
	    });
	    std::generate(bias_data.begin(), bias_data.end(), []() {
	        static int i = 0;
	        return std::tanh(float(i++));
	    });
	
	    // Create memory descriptors and memory objects for src and dst. In this
	    // example, NCHW layout is assumed.
	    auto :ref:`src_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a90a729e395453e1d9411ad416c796819>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw);
	    auto bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(bias_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::a);
	    auto :ref:`dst_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a701158248eed4e5fc84610f2f6026493>` = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(dst_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nc);
	
	    auto src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto bias_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(bias_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create memory object for user's layout for weights. In this example, OIHW
	    // is assumed.
	    auto user_weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({weights_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::oihw}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handles.
	    write_to_dnnl_memory(src_data.data(), src_mem);
	    write_to_dnnl_memory(bias_data.data(), bias_mem);
	    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
	
	    // Create memory descriptor for weights with format_tag::any. This enables
	    // the inner product primitive to choose the memory layout for an optimized
	    // primitive implementation, and this format may differ from the one
	    // provided by the user.
	    auto inner_product_weights_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(weights_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // Create primitive post-ops (ReLU).
	    const float alpha = 0.f;
	    const float beta = 0.f;
	    :ref:`post_ops <doxid-structdnnl_1_1post__ops>` inner_product_ops;
	    inner_product_ops.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(:ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, alpha, beta);
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` inner_product_attr;
	    inner_product_attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(inner_product_ops);
	
	    // Create inner product primitive descriptor.
	    auto inner_product_pd = :ref:`inner_product_forward::primitive_desc <doxid-structdnnl_1_1inner__product__forward_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`,
	            :ref:`prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, src_md, inner_product_weights_md,
	            bias_md, dst_md, inner_product_attr);
	
	    // For now, assume that the weights memory layout generated by the primitive
	    // and the one provided by the user are identical.
	    auto inner_product_weights_mem = user_weights_mem;
	
	    // Reorder the data in case the weights memory layout generated by the
	    // primitive and the one provided by the user are different. In this case,
	    // we create additional memory objects with internal buffers that will
	    // contain the reordered data.
	    if (inner_product_pd.weights_desc() != user_weights_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        inner_product_weights_mem
	                = :ref:`memory <doxid-structdnnl_1_1memory>`(inner_product_pd.weights_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_mem, inner_product_weights_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_weights_mem,
	                        inner_product_weights_mem);
	    }
	
	    // Create the primitive.
	    auto inner_product_prim = :ref:`inner_product_forward <doxid-structdnnl_1_1inner__product__forward>`(inner_product_pd);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> inner_product_args;
	    inner_product_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mem});
	    inner_product_args.insert({:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, inner_product_weights_mem});
	    inner_product_args.insert({:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, bias_mem});
	    inner_product_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mem});
	
	    // Primitive execution: inner-product with ReLU.
	    inner_product_prim.execute(engine_stream, inner_product_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_data.data(), dst_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            inner_product_example, parse_engine_kind(argc, argv));
	}
