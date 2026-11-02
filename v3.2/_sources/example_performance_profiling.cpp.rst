.. index:: pair: example; performance_profiling.cpp
.. _doxid-performance_profiling_8cpp-example:

performance_profiling.cpp
=========================

This example demonstrates the best practices for application performance optimizations with oneDNN. Annotated version: :ref:`Performance Profiling Example <doxid-performance_profiling_cpp>`

This example demonstrates the best practices for application performance optimizations with oneDNN. Annotated version: :ref:`Performance Profiling Example <doxid-performance_profiling_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2019-2023 Intel Corporation
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
	#include <stdexcept>
	#include <vector>
	
	#include "oneapi/dnnl/dnnl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	// [Prologue]
	
	// Set Strides and Padding
	const :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` strides = {4, 4};
	const :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` padding = {0, 0};
	
	// [Prologue]
	//
	// function to init data
	void init_data(:ref:`memory <doxid-structdnnl_1_1memory>` &m, float v) {
	    size_t size = m.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_size <doxid-structdnnl_1_1memory_1_1desc_1abfa095ac138d4d2ef8efd3739e343f08>`() / sizeof(float);
	    std::vector<float> data(size, v);
	    write_to_dnnl_memory(data.data(), m);
	}
	
	// function to execute non-fused relu
	void create_and_execute_relu(:ref:`memory <doxid-structdnnl_1_1memory>` &data, :ref:`engine <doxid-structdnnl_1_1engine>` &eng, :ref:`stream <doxid-structdnnl_1_1stream>` &s) {
	    // relu operates on whatever data format is given to it
	
	    // create a primitive
	    auto relu_pd = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`,
	            data.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`(), data.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`(), 0.f, 0.f);
	    auto relu = :ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu_pd);
	
	    // execute it (in-place)
	    relu.execute(s, {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, data}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, data}});
	}
	
	// [Create post_op attr with relu]
	// function to create post-op attribute for fused relu
	:ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` create_attr_with_relu_post_op() {
	    // create a post-op with relu
	    :ref:`post_ops <doxid-structdnnl_1_1post__ops>` ops;
	    ops.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(:ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, 0.f, 0.f);
	
	    // create an attribute and set the corresponding post op
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	    attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(ops);
	
	    return attr;
	}
	// [Create post_op attr with relu]
	
	// Implementation for naive convolution on nchw (data) and oihw (weights),
	// followed by execution of non-fused relu
	void conv_relu_naive(const :ref:`memory <doxid-structdnnl_1_1memory>` &user_src, const :ref:`memory <doxid-structdnnl_1_1memory>` &user_wei,
	        :ref:`memory <doxid-structdnnl_1_1memory>` user_dst, :ref:`engine <doxid-structdnnl_1_1engine>` &eng, :ref:`stream <doxid-structdnnl_1_1stream>` &s) {
	    // [Create mem_desc]
	    // copy the dimensions and format from user's memory
	    auto conv_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_src.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`());
	    auto conv_wei_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_wei.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`());
	    auto conv_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`());
	    // [Create mem_desc]
	    // [Create conv_prim_desc]
	    // create a convolution primitive descriptor
	    auto conv_pd = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	            conv_src_md, conv_wei_md, conv_dst_md, strides, padding, padding);
	    // [Create conv_prim_desc]
	    // [Create conv_primitive]
	    // create convolution primitive
	    auto conv = :ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv_pd);
	    // [Create conv_primitive]
	    // [Add to stream]
	    // execute convolution by adding it to the stream s
	    conv.execute(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, user_src}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, user_wei},
	                    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, user_dst}});
	    // [Add to stream]
	    // [Create and execute relu]
	    // execute relu (on convolution's destination format, whatever it is)
	    create_and_execute_relu(user_dst, eng, s);
	    s.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	    // [Create and execute relu]
	}
	
	// Implementation for convolution on blocked format for data and
	// weights, followed by execution of non-fused relu
	void conv_relu_blocked(:ref:`memory <doxid-structdnnl_1_1memory>` user_src, :ref:`memory <doxid-structdnnl_1_1memory>` user_wei, :ref:`memory <doxid-structdnnl_1_1memory>` user_dst,
	        :ref:`engine <doxid-structdnnl_1_1engine>` &eng, :ref:`stream <doxid-structdnnl_1_1stream>` &s) {
	    // [Create mem_desc with tag=any]
	    // copy the dimensions and data type from user's memory and set format tag
	    // to "any" to allow convolution to pick the best implementation
	    auto conv_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_src.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`(),
	            user_src.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_data_type <doxid-structdnnl_1_1memory_1_1desc_1aada0dc594d12f25331d4d7cf84c08e75>`(), :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_wei_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_wei.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`(),
	            user_wei.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_data_type <doxid-structdnnl_1_1memory_1_1desc_1aada0dc594d12f25331d4d7cf84c08e75>`(), :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`(),
	            user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_data_type <doxid-structdnnl_1_1memory_1_1desc_1aada0dc594d12f25331d4d7cf84c08e75>`(), :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    // [Create mem_desc with tag=any]
	
	    // [Create conv_prim_desc implementation2]
	    // create a convolution primitive descriptor and primitive
	    auto conv_pd = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	            conv_src_md, conv_wei_md, conv_dst_md, strides, padding, padding);
	    // [Create conv_prim_desc implementation2]
	    // [Conditionally create and execute reorder prims]
	    // prepare convolution source
	    :ref:`memory <doxid-structdnnl_1_1memory>` conv_src = user_src;
	    if (conv_pd.src_desc() != user_src.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        conv_src = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.src_desc(), eng);
	        auto r_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(user_src, conv_src);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(r_pd).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_src, conv_src);
	    }
	
	    // prepare convolution weights
	    :ref:`memory <doxid-structdnnl_1_1memory>` conv_wei = user_wei;
	    if (conv_pd.weights_desc() != user_wei.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        conv_wei = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.weights_desc(), eng);
	        auto r_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(user_wei, conv_wei);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(r_pd).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_wei, conv_wei);
	    }
	
	    // prepare convolution destination
	    :ref:`memory <doxid-structdnnl_1_1memory>` conv_dst = user_dst;
	    if (conv_pd.dst_desc() != user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`())
	        conv_dst = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.dst_desc(), eng);
	    // [Conditionally create and execute reorder prims]
	    // [Create conv_primitive implementation2]
	    // create convolution primitive
	    auto conv = :ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv_pd);
	    // [Create conv_primitive implementation2]
	    // [Add to stream implementation2]
	    // execute convolution by adding it to the stream s
	    conv.execute(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_wei},
	                    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst}});
	    // [Add to stream implementation2]
	    // [Create and execute relu implementation2]
	    // execute relu (on convolution's destination format, whatever it is)
	    create_and_execute_relu(conv_dst, eng, s);
	    // [Create and execute relu implementation2]
	    if (conv_pd.dst_desc() != user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        auto r_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(conv_dst, user_dst);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(r_pd).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, conv_dst, user_dst);
	    }
	    s.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	    // reorder data to the user's format if needed.
	}
	
	// Implementation for convolution on blocked format for data and
	// weights and the relu operation fused via a post-op attribute added to the
	// convolution prim_descriptor
	void conv_relu_fused(:ref:`memory <doxid-structdnnl_1_1memory>` user_src, :ref:`memory <doxid-structdnnl_1_1memory>` user_wei, :ref:`memory <doxid-structdnnl_1_1memory>` user_dst,
	        const :ref:`engine <doxid-structdnnl_1_1engine>` &eng, :ref:`stream <doxid-structdnnl_1_1stream>` &s) {
	    // copy the dimensions data type from user's memory and set format tag
	    // to any to allow convolution to pick the best implementation
	    auto conv_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_src.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`(),
	            user_src.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_data_type <doxid-structdnnl_1_1memory_1_1desc_1aada0dc594d12f25331d4d7cf84c08e75>`(), :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_wei_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_wei.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`(),
	            user_wei.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_data_type <doxid-structdnnl_1_1memory_1_1desc_1aada0dc594d12f25331d4d7cf84c08e75>`(), :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_dims <doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`(),
	            user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_data_type <doxid-structdnnl_1_1memory_1_1desc_1aada0dc594d12f25331d4d7cf84c08e75>`(), :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	
	    // Next the convolution prim descriptor is created, which inherits the ReLU
	    // [Create prim_desc with attr]
	    // create an attribute for fused relu
	    auto attr = create_attr_with_relu_post_op();
	
	    // create a convolution primitive descriptor
	    auto conv_pd = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	            conv_src_md, conv_wei_md, conv_dst_md, strides, padding, padding,
	            attr);
	    // [Create prim_desc with attr]
	    // prepare convolution source
	    :ref:`memory <doxid-structdnnl_1_1memory>` conv_src = user_src;
	    if (conv_pd.src_desc() != user_src.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        conv_src = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.src_desc(), eng);
	        auto r_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(user_src, conv_src);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(r_pd).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_src, conv_src);
	    }
	
	    // prepare convolution weights
	    :ref:`memory <doxid-structdnnl_1_1memory>` conv_wei = user_wei;
	    if (conv_pd.weights_desc() != user_wei.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        conv_wei = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.weights_desc(), eng);
	        auto r_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(user_wei, conv_wei);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(r_pd).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_wei, conv_wei);
	    }
	
	    // prepare convolution destination
	    :ref:`memory <doxid-structdnnl_1_1memory>` conv_dst = user_dst;
	    if (conv_pd.dst_desc() != user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`())
	        conv_dst = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.dst_desc(), eng);
	    // [Create conv_primitive implementation3]
	    // create convolution primitive
	    auto conv = :ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv_pd);
	    // [Create conv_primitive implementation3]
	    // [Add to stream implementation3]
	    // execute convolution by adding it to the stream s
	    conv.execute(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src}, {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_wei},
	                    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst}});
	    // [Add to stream implementation3]
	    // reorder data to user's format if needed
	    if (conv_pd.dst_desc() != user_dst.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        auto r_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(conv_dst, user_dst);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(r_pd).:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, conv_dst, user_dst);
	    }
	    s.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	}
	
	
	void performance_profiling(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, int argc, char **argv) {
	    // Initialize engine
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	
	    // Initialize stream
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	    // [Set dimensions]
	    // set dimensions for synthetic data and weights
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` BATCH = 128;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` IC = 3, OC = 96;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` IH = 227, KH = 11, OH = 55;
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` IW = 227, KW = 11, OW = 55;
	    // [Set dimensions]
	
	    // [Create memory objects]
	    // create oneDNN memory objects for user's tensors (in nchw and oihw formats)
	    auto user_src = :ref:`memory <doxid-structdnnl_1_1memory>`({{BATCH, IC, IH, IW}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	                                   :ref:`memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`},
	            eng);
	    auto user_wei = :ref:`memory <doxid-structdnnl_1_1memory>`({{OC, IC, KH, KW}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	                                   :ref:`memory::format_tag::oihw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa14b72a467aeefa06a5cb802ec4a7743c>`},
	            eng);
	    auto user_dst = :ref:`memory <doxid-structdnnl_1_1memory>`({{BATCH, OC, OH, OW}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	                                   :ref:`memory::format_tag::nchw <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3faded7ac40158367123c5467281d44cbeb>`},
	            eng);
	    // [Create memory objects]
	
	    // fill source, destination, and weights with synthetic data
	    init_data(user_src, 1);
	    init_data(user_dst, -1);
	    init_data(user_wei, .5);
	
	    // set implementation ("naive"||"blocked"||"fused") setting implementation
	    // to "validation" will run all implementations
	    std::string implementation;
	    if (argc <= 2)
	        implementation = "validation";
	    else if (argc == 3)
	        implementation = argv[2];
	
	    if (!(implementation == "validation" || implementation == "naive"
	                || implementation == "blocked" || implementation == "fused")) {
	        std::cout << "The implementation can be one of:\n";
	        std::cout << " - naive: NCHW format without fusion\n";
	        std::cout << " - blocked: format propagation without fusion\n";
	        std::cout << " - fused: format propagation with fusion\n";
	        std::cout << " - validation: runs all implementations\n\n";
	        std::cout << "Validation will run if no parameters are specified.\n\n";
	
	        throw std::invalid_argument("Incorrect input arguments.");
	    }
	
	    if (implementation == "naive" || implementation == "validation") {
	        std::cout << "Implementation: naive.\n";
	        // run conv + relu w/o fusing
	        conv_relu_naive(user_src, user_wei, user_dst, eng, s);
	        std::cout << "Conv + ReLU w/ nchw format completed.\n";
	    }
	
	    if (implementation == "blocked" || implementation == "validation") {
	        std::cout << "Implementation: blocked.\n";
	        // run conv + relu w/o fusing
	        conv_relu_blocked(user_src, user_wei, user_dst, eng, s);
	        std::cout << "Conv + ReLU w/ blocked format completed.\n";
	    }
	
	    if (implementation == "fused" || implementation == "validation") {
	        std::cout << "Implementation: fused.\n";
	        // run conv + relu w/ fusing
	        conv_relu_fused(user_src, user_wei, user_dst, eng, s);
	        std::cout << "Conv + ReLU w/ fusing completed.\n";
	    }
	}
	
	int main(int argc, char **argv) {
	    :ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind = parse_engine_kind(argc, argv, 1);
	    return handle_example_errors(
	            performance_profiling, engine_kind, argc, argv);
	}
