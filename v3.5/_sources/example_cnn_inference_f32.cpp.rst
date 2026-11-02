.. index:: pair: example; cnn_inference_f32.cpp
.. _doxid-cnn_inference_f32_8cpp-example:

cnn_inference_f32.cpp
=====================

This C++ API example demonstrates how to build an AlexNet neural network topology for forward-pass inference. Annotated version: :ref:`CNN f32 inference example <doxid-cnn_inference_f32_cpp>`

This C++ API example demonstrates how to build an AlexNet neural network topology for forward-pass inference. Annotated version: :ref:`CNN f32 inference example <doxid-cnn_inference_f32_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2016-2022 Intel Corporation
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
	
	
	
	#include <assert.h>
	
	#include <chrono>
	#include <vector>
	#include <unordered_map>
	
	#include "oneapi/dnnl/dnnl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	void simple_net(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, int times = 100) {
	    using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	    using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	    //[Initialize engine and stream]
	    :ref:`engine <doxid-structdnnl_1_1engine>` eng(engine_kind, 0);
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	    //[Initialize engine and stream]
	
	    //[Create network]
	    std::vector<primitive> net;
	    std::vector<std::unordered_map<int, memory>> net_args;
	    //[Create network]
	
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` batch = 1;
	
	    // AlexNet: conv1
	    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
	    // strides: {4, 4}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv1_src_tz = {batch, 3, 227, 227};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv1_weights_tz = {96, 3, 11, 11};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv1_bias_tz = {96};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv1_dst_tz = {batch, 96, 55, 55};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv1_strides = {4, 4};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv1_padding = {0, 0};
	
	    //[Allocate buffers]
	    std::vector<float> user_src(batch * 3 * 227 * 227);
	    std::vector<float> user_dst(batch * 1000);
	    std::vector<float> conv1_weights(product(conv1_weights_tz));
	    std::vector<float> conv1_bias(product(conv1_bias_tz));
	    //[Allocate buffers]
	
	    //[Create user memory]
	    auto user_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv1_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, eng);
	    write_to_dnnl_memory(user_src.data(), user_src_memory);
	    auto user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv1_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::oihw}, eng);
	    write_to_dnnl_memory(conv1_weights.data(), user_weights_memory);
	    auto conv1_user_bias_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv1_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);
	    //[Create user memory]
	
	    //[Create convolution memory descriptors]
	    auto conv1_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv1_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv1_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv1_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv1_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv1_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv1_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv1_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    //[Create convolution memory descriptors]
	
	    //[Create convolution primitive descriptor]
	    auto conv1_prim_desc = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	            conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
	            conv1_strides, conv1_padding, conv1_padding);
	    //[Create convolution primitive descriptor]
	
	    //[Reorder data and weights]
	    auto conv1_src_memory = user_src_memory;
	    if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
	        conv1_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv1_prim_desc.src_desc(), eng);
	        net.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(user_src_memory, conv1_src_memory));
	        net_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, user_src_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv1_src_memory}});
	    }
	
	    auto conv1_weights_memory = user_weights_memory;
	    if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
	        conv1_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv1_prim_desc.weights_desc(), eng);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_memory, conv1_weights_memory)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_weights_memory, conv1_weights_memory);
	    }
	    //[Reorder data and weights]
	
	    //[Create memory for output]
	    auto conv1_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv1_prim_desc.dst_desc(), eng);
	    //[Create memory for output]
	
	    //[Create convolution primitive]
	    net.push_back(:ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv1_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv1_src_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv1_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv1_user_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv1_dst_memory}});
	    //[Create convolution primitive]
	
	    // AlexNet: relu1
	    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
	    const float negative1_slope = 0.0f;
	
	    //[Create relu primitive]
	    auto relu1_prim_desc
	            = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	                    :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, conv1_dst_memory.get_desc(),
	                    conv1_dst_memory.get_desc(), negative1_slope);
	
	    net.push_back(:ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu1_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv1_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv1_dst_memory}});
	    //[Create relu primitive]
	
	    // AlexNet: lrn1
	    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
	    // local size: 5
	    // alpha1: 0.0001
	    // beta1: 0.75
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` local1_size = 5;
	    const float alpha1 = 0.0001f;
	    const float beta1 = 0.75f;
	    const float k1 = 1.0f;
	
	    // create lrn primitive and add it to net
	    auto lrn1_prim_desc = :ref:`lrn_forward::primitive_desc <doxid-structdnnl_1_1lrn__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::lrn_across_channels <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab9e2d858b551792385a4b5b86672b24b>`,
	            conv1_dst_memory.get_desc(), conv1_dst_memory.get_desc(),
	            local1_size, alpha1, beta1, k1);
	    auto lrn1_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(lrn1_prim_desc.dst_desc(), eng);
	
	    net.push_back(:ref:`lrn_forward <doxid-structdnnl_1_1lrn__forward>`(lrn1_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv1_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, lrn1_dst_memory}});
	
	    // AlexNet: pool1
	    // {batch, 96, 55, 55} -> {batch, 96, 27, 27}
	    // kernel: {3, 3}
	    // strides: {2, 2}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool1_dst_tz = {batch, 96, 27, 27};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool1_kernel = {3, 3};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool1_strides = {2, 2};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool_dilation = {0, 0};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool_padding = {0, 0};
	
	    auto pool1_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({pool1_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    //[Create pooling primitive]
	    auto pool1_pd = :ref:`pooling_forward::primitive_desc <doxid-structdnnl_1_1pooling__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::pooling_max <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`,
	            lrn1_dst_memory.get_desc(), pool1_dst_md, pool1_strides,
	            pool1_kernel, pool_dilation, pool_padding, pool_padding);
	    auto pool1_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(pool1_pd.dst_desc(), eng);
	
	    net.push_back(:ref:`pooling_forward <doxid-structdnnl_1_1pooling__forward>`(pool1_pd));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, lrn1_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool1_dst_memory}});
	    //[Create pooling primitive]
	
	    // AlexNet: conv2
	    // {batch, 96, 27, 27} (x) {2, 128, 48, 5, 5} -> {batch, 256, 27, 27}
	    // strides: {1, 1}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv2_src_tz = {batch, 96, 27, 27};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv2_weights_tz = {2, 128, 48, 5, 5};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv2_bias_tz = {256};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv2_dst_tz = {batch, 256, 27, 27};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv2_strides = {1, 1};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv2_padding = {2, 2};
	
	    std::vector<float> conv2_weights(product(conv2_weights_tz));
	    std::vector<float> conv2_bias(product(conv2_bias_tz));
	
	    // create memory for user data
	    auto conv2_user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv2_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::goihw}, eng);
	    write_to_dnnl_memory(conv2_weights.data(), conv2_user_weights_memory);
	    auto conv2_user_bias_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv2_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(conv2_bias.data(), conv2_user_bias_memory);
	
	    // create memory descriptors for convolution data w/ no specified format
	    auto conv2_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv2_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv2_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv2_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv2_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv2_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv2_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv2_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a convolution
	    auto conv2_prim_desc = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	            conv2_src_md, conv2_weights_md, conv2_bias_md, conv2_dst_md,
	            conv2_strides, conv2_padding, conv2_padding);
	
	    auto conv2_src_memory = pool1_dst_memory;
	    if (conv2_prim_desc.src_desc() != conv2_src_memory.get_desc()) {
	        conv2_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv2_prim_desc.src_desc(), eng);
	        net.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(pool1_dst_memory, conv2_src_memory));
	        net_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, pool1_dst_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv2_src_memory}});
	    }
	
	    auto conv2_weights_memory = conv2_user_weights_memory;
	    if (conv2_prim_desc.weights_desc()
	            != conv2_user_weights_memory.get_desc()) {
	        conv2_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv2_prim_desc.weights_desc(), eng);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(conv2_user_weights_memory, conv2_weights_memory)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, conv2_user_weights_memory, conv2_weights_memory);
	    }
	
	    auto conv2_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv2_prim_desc.dst_desc(), eng);
	
	    // create convolution primitive and add it to net
	    net.push_back(:ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv2_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv2_src_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv2_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv2_user_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv2_dst_memory}});
	
	    // AlexNet: relu2
	    // {batch, 256, 27, 27} -> {batch, 256, 27, 27}
	    const float negative2_slope = 0.0f;
	
	    // create relu primitive and add it to net
	    auto relu2_prim_desc
	            = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	                    :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, conv2_dst_memory.get_desc(),
	                    conv2_dst_memory.get_desc(), negative2_slope);
	
	    net.push_back(:ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu2_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv2_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv2_dst_memory}});
	
	    // AlexNet: lrn2
	    // {batch, 256, 27, 27} -> {batch, 256, 27, 27}
	    // local size: 5
	    // alpha2: 0.0001
	    // beta2: 0.75
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` local2_size = 5;
	    const float alpha2 = 0.0001f;
	    const float beta2 = 0.75f;
	    const float k2 = 1.0f;
	
	    // create lrn primitive and add it to net
	    auto lrn2_prim_desc
	            = :ref:`lrn_forward::primitive_desc <doxid-structdnnl_1_1lrn__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	                    :ref:`algorithm::lrn_across_channels <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab9e2d858b551792385a4b5b86672b24b>`, conv2_prim_desc.dst_desc(),
	                    conv2_prim_desc.dst_desc(), local2_size, alpha2, beta2, k2);
	    auto lrn2_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(lrn2_prim_desc.dst_desc(), eng);
	
	    net.push_back(:ref:`lrn_forward <doxid-structdnnl_1_1lrn__forward>`(lrn2_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv2_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, lrn2_dst_memory}});
	
	    // AlexNet: pool2
	    // {batch, 256, 27, 27} -> {batch, 256, 13, 13}
	    // kernel: {3, 3}
	    // strides: {2, 2}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool2_dst_tz = {batch, 256, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool2_kernel = {3, 3};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool2_strides = {2, 2};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool2_dilation = {0, 0};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool2_padding = {0, 0};
	
	    auto pool2_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({pool2_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a pooling
	    auto pool2_pd = :ref:`pooling_forward::primitive_desc <doxid-structdnnl_1_1pooling__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::pooling_max <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`,
	            lrn2_dst_memory.get_desc(), pool2_dst_md, pool2_strides,
	            pool2_kernel, pool2_dilation, pool2_padding, pool2_padding);
	    auto pool2_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(pool2_pd.dst_desc(), eng);
	
	    // create pooling primitive an add it to net
	    net.push_back(:ref:`pooling_forward <doxid-structdnnl_1_1pooling__forward>`(pool2_pd));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, lrn2_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool2_dst_memory}});
	
	    // AlexNet: conv3
	    // {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13};
	    // strides: {1, 1}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv3_src_tz = {batch, 256, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv3_weights_tz = {384, 256, 3, 3};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv3_bias_tz = {384};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv3_dst_tz = {batch, 384, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv3_strides = {1, 1};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv3_padding = {1, 1};
	
	    std::vector<float> conv3_weights(product(conv3_weights_tz));
	    std::vector<float> conv3_bias(product(conv3_bias_tz));
	
	    // create memory for user data
	    auto conv3_user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv3_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::oihw}, eng);
	    write_to_dnnl_memory(conv3_weights.data(), conv3_user_weights_memory);
	    auto conv3_user_bias_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv3_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(conv3_bias.data(), conv3_user_bias_memory);
	
	    // create memory descriptors for convolution data w/ no specified format
	    auto conv3_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv3_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv3_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv3_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv3_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv3_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv3_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv3_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a convolution
	    auto conv3_prim_desc = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	            conv3_src_md, conv3_weights_md, conv3_bias_md, conv3_dst_md,
	            conv3_strides, conv3_padding, conv3_padding);
	
	    auto conv3_src_memory = pool2_dst_memory;
	    if (conv3_prim_desc.src_desc() != conv3_src_memory.get_desc()) {
	        conv3_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv3_prim_desc.src_desc(), eng);
	        net.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(pool2_dst_memory, conv3_src_memory));
	        net_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, pool2_dst_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv3_src_memory}});
	    }
	
	    auto conv3_weights_memory = conv3_user_weights_memory;
	    if (conv3_prim_desc.weights_desc()
	            != conv3_user_weights_memory.get_desc()) {
	        conv3_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv3_prim_desc.weights_desc(), eng);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(conv3_user_weights_memory, conv3_weights_memory)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, conv3_user_weights_memory, conv3_weights_memory);
	    }
	
	    auto conv3_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv3_prim_desc.dst_desc(), eng);
	
	    // create convolution primitive and add it to net
	    net.push_back(:ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv3_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv3_src_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv3_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv3_user_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv3_dst_memory}});
	
	    // AlexNet: relu3
	    // {batch, 384, 13, 13} -> {batch, 384, 13, 13}
	    const float negative3_slope = 0.0f;
	
	    // create relu primitive and add it to net
	    auto relu3_prim_desc
	            = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	                    :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, conv3_dst_memory.get_desc(),
	                    conv3_dst_memory.get_desc(), negative3_slope);
	
	    net.push_back(:ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu3_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv3_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv3_dst_memory}});
	
	    // AlexNet: conv4
	    // {batch, 384, 13, 13} (x)  {2, 192, 192, 3, 3}; ->
	    // {batch, 384, 13, 13};
	    // strides: {1, 1}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv4_src_tz = {batch, 384, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv4_weights_tz = {2, 192, 192, 3, 3};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv4_bias_tz = {384};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv4_dst_tz = {batch, 384, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv4_strides = {1, 1};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv4_padding = {1, 1};
	
	    std::vector<float> conv4_weights(product(conv4_weights_tz));
	    std::vector<float> conv4_bias(product(conv4_bias_tz));
	
	    // create memory for user data
	    auto conv4_user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv4_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::goihw}, eng);
	    write_to_dnnl_memory(conv4_weights.data(), conv4_user_weights_memory);
	    auto conv4_user_bias_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv4_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(conv4_bias.data(), conv4_user_bias_memory);
	
	    // create memory descriptors for convolution data w/ no specified format
	    auto conv4_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv4_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv4_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv4_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv4_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv4_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv4_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv4_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a convolution
	    auto conv4_prim_desc = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	            conv4_src_md, conv4_weights_md, conv4_bias_md, conv4_dst_md,
	            conv4_strides, conv4_padding, conv4_padding);
	
	    auto conv4_src_memory = conv3_dst_memory;
	    if (conv4_prim_desc.src_desc() != conv4_src_memory.get_desc()) {
	        conv4_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv4_prim_desc.src_desc(), eng);
	        net.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(conv3_dst_memory, conv4_src_memory));
	        net_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv3_dst_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv4_src_memory}});
	    }
	
	    auto conv4_weights_memory = conv4_user_weights_memory;
	    if (conv4_prim_desc.weights_desc()
	            != conv4_user_weights_memory.get_desc()) {
	        conv4_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv4_prim_desc.weights_desc(), eng);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(conv4_user_weights_memory, conv4_weights_memory)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, conv4_user_weights_memory, conv4_weights_memory);
	    }
	
	    auto conv4_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv4_prim_desc.dst_desc(), eng);
	
	    // create convolution primitive and add it to net
	    net.push_back(:ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv4_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv4_src_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv4_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv4_user_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv4_dst_memory}});
	
	    // AlexNet: relu4
	    // {batch, 384, 13, 13} -> {batch, 384, 13, 13}
	    const float negative4_slope = 0.0f;
	
	    // create relu primitive and add it to net
	    auto relu4_prim_desc
	            = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	                    :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, conv4_dst_memory.get_desc(),
	                    conv4_dst_memory.get_desc(), negative4_slope);
	
	    net.push_back(:ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu4_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv4_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv4_dst_memory}});
	
	    // AlexNet: conv5
	    // {batch, 384, 13, 13} (x)  {2, 128, 192, 3, 3}; -> {batch, 256, 13, 13};
	    // strides: {1, 1}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv5_src_tz = {batch, 384, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv5_weights_tz = {2, 128, 192, 3, 3};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv5_bias_tz = {256};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv5_dst_tz = {batch, 256, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv5_strides = {1, 1};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv5_padding = {1, 1};
	
	    std::vector<float> conv5_weights(product(conv5_weights_tz));
	    std::vector<float> conv5_bias(product(conv5_bias_tz));
	
	    // create memory for user data
	    auto conv5_user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv5_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::goihw}, eng);
	    write_to_dnnl_memory(conv5_weights.data(), conv5_user_weights_memory);
	    auto conv5_user_bias_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv5_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(conv5_bias.data(), conv5_user_bias_memory);
	
	    // create memory descriptors for convolution data w/ no specified format
	    auto conv5_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv5_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv5_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv5_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv5_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv5_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv5_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv5_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a convolution
	    auto conv5_prim_desc = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	            conv5_src_md, conv5_weights_md, conv5_bias_md, conv5_dst_md,
	            conv5_strides, conv5_padding, conv5_padding);
	
	    auto conv5_src_memory = conv4_dst_memory;
	    if (conv5_prim_desc.src_desc() != conv5_src_memory.get_desc()) {
	        conv5_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv5_prim_desc.src_desc(), eng);
	        net.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(conv4_dst_memory, conv5_src_memory));
	        net_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv4_dst_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv5_src_memory}});
	    }
	
	    auto conv5_weights_memory = conv5_user_weights_memory;
	    if (conv5_prim_desc.weights_desc()
	            != conv5_user_weights_memory.get_desc()) {
	        conv5_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv5_prim_desc.weights_desc(), eng);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(conv5_user_weights_memory, conv5_weights_memory)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, conv5_user_weights_memory, conv5_weights_memory);
	    }
	
	    auto conv5_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv5_prim_desc.dst_desc(), eng);
	
	    // create convolution primitive and add it to net
	    net.push_back(:ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv5_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv5_src_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv5_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv5_user_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv5_dst_memory}});
	
	    // AlexNet: relu5
	    // {batch, 256, 13, 13} -> {batch, 256, 13, 13}
	    const float negative5_slope = 0.0f;
	
	    // create relu primitive and add it to net
	    auto relu5_prim_desc
	            = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	                    :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, conv5_dst_memory.get_desc(),
	                    conv5_dst_memory.get_desc(), negative5_slope);
	
	    net.push_back(:ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu5_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv5_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv5_dst_memory}});
	
	    // AlexNet: pool5
	    // {batch, 256, 13, 13} -> {batch, 256, 6, 6}
	    // kernel: {3, 3}
	    // strides: {2, 2}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool5_dst_tz = {batch, 256, 6, 6};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool5_kernel = {3, 3};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool5_strides = {2, 2};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool5_dilation = {0, 0};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool5_padding = {0, 0};
	
	    std::vector<float> pool5_dst(product(pool5_dst_tz));
	
	    auto pool5_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({pool5_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a pooling
	    auto pool5_pd = :ref:`pooling_forward::primitive_desc <doxid-structdnnl_1_1pooling__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, :ref:`algorithm::pooling_max <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`,
	            conv5_dst_memory.get_desc(), pool5_dst_md, pool5_strides,
	            pool5_kernel, pool5_dilation, pool5_padding, pool5_padding);
	
	    auto pool5_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(pool5_pd.dst_desc(), eng);
	
	    // create pooling primitive an add it to net
	    net.push_back(:ref:`pooling_forward <doxid-structdnnl_1_1pooling__forward>`(pool5_pd));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv5_dst_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool5_dst_memory}});
	
	    // fc6 inner product {batch, 256, 6, 6} (x) {4096, 256, 6, 6}-> {batch,
	    // 4096}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc6_src_tz = {batch, 256, 6, 6};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc6_weights_tz = {4096, 256, 6, 6};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc6_bias_tz = {4096};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc6_dst_tz = {batch, 4096};
	
	    std::vector<float> fc6_weights(product(fc6_weights_tz));
	    std::vector<float> fc6_bias(product(fc6_bias_tz));
	
	    // create memory for user data
	    auto fc6_user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{fc6_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::oihw}, eng);
	    write_to_dnnl_memory(fc6_weights.data(), fc6_user_weights_memory);
	    auto fc6_user_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{fc6_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(fc6_bias.data(), fc6_user_bias_memory);
	
	    // create memory descriptors for convolution data w/ no specified format
	    auto fc6_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc6_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto fc6_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc6_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto fc6_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc6_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto fc6_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc6_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a inner_product
	    auto fc6_prim_desc = :ref:`inner_product_forward::primitive_desc <doxid-structdnnl_1_1inner__product__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, fc6_src_md, fc6_weights_md,
	            fc6_bias_md, fc6_dst_md);
	
	    auto fc6_src_memory = pool5_dst_memory;
	    if (fc6_prim_desc.src_desc() != fc6_src_memory.get_desc()) {
	        fc6_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(fc6_prim_desc.src_desc(), eng);
	        net.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(pool5_dst_memory, fc6_src_memory));
	        net_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, pool5_dst_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, fc6_src_memory}});
	    }
	
	    auto fc6_weights_memory = fc6_user_weights_memory;
	    if (fc6_prim_desc.weights_desc() != fc6_user_weights_memory.get_desc()) {
	        fc6_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(fc6_prim_desc.weights_desc(), eng);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(fc6_user_weights_memory, fc6_weights_memory)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, fc6_user_weights_memory, fc6_weights_memory);
	    }
	
	    auto fc6_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(fc6_prim_desc.dst_desc(), eng);
	
	    // create convolution primitive and add it to net
	    net.push_back(:ref:`inner_product_forward <doxid-structdnnl_1_1inner__product__forward>`(fc6_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, fc6_src_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, fc6_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, fc6_user_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, fc6_dst_memory}});
	
	    // fc7 inner product {batch, 4096} (x) {4096, 4096}-> {batch, 4096}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc7_weights_tz = {4096, 4096};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc7_bias_tz = {4096};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc7_dst_tz = {batch, 4096};
	
	    std::vector<float> fc7_weights(product(fc7_weights_tz));
	    std::vector<float> fc7_bias(product(fc7_bias_tz));
	
	    // create memory for user data
	    auto fc7_user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{fc7_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nc}, eng);
	    write_to_dnnl_memory(fc7_weights.data(), fc7_user_weights_memory);
	
	    auto fc7_user_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{fc7_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(fc7_bias.data(), fc7_user_bias_memory);
	
	    // create memory descriptors for convolution data w/ no specified format
	    auto fc7_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc7_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto fc7_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc7_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto fc7_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc7_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a inner_product
	    auto fc7_prim_desc = :ref:`inner_product_forward::primitive_desc <doxid-structdnnl_1_1inner__product__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, fc6_dst_memory.get_desc(),
	            fc7_weights_md, fc7_bias_md, fc7_dst_md);
	
	    auto fc7_weights_memory = fc7_user_weights_memory;
	    if (fc7_prim_desc.weights_desc() != fc7_user_weights_memory.get_desc()) {
	        fc7_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(fc7_prim_desc.weights_desc(), eng);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(fc7_user_weights_memory, fc7_weights_memory)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, fc7_user_weights_memory, fc7_weights_memory);
	    }
	
	    auto fc7_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(fc7_prim_desc.dst_desc(), eng);
	
	    // create convolution primitive and add it to net
	    net.push_back(:ref:`inner_product_forward <doxid-structdnnl_1_1inner__product__forward>`(fc7_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, fc6_dst_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, fc7_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, fc7_user_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, fc7_dst_memory}});
	
	    // fc8 inner product {batch, 4096} (x) {1000, 4096}-> {batch, 1000}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc8_weights_tz = {1000, 4096};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc8_bias_tz = {1000};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` fc8_dst_tz = {batch, 1000};
	
	    std::vector<float> fc8_weights(product(fc8_weights_tz));
	    std::vector<float> fc8_bias(product(fc8_bias_tz));
	
	    // create memory for user data
	    auto fc8_user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{fc8_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nc}, eng);
	    write_to_dnnl_memory(fc8_weights.data(), fc8_user_weights_memory);
	    auto fc8_user_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{fc8_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(fc8_bias.data(), fc8_user_bias_memory);
	    auto user_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{fc8_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nc}, eng);
	    write_to_dnnl_memory(user_dst.data(), user_dst_memory);
	
	    // create memory descriptors for convolution data w/ no specified format
	    auto fc8_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc8_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto fc8_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc8_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto fc8_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({fc8_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a inner_product
	    auto fc8_prim_desc = :ref:`inner_product_forward::primitive_desc <doxid-structdnnl_1_1inner__product__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`, fc7_dst_memory.get_desc(),
	            fc8_weights_md, fc8_bias_md, fc8_dst_md);
	
	    auto fc8_weights_memory = fc8_user_weights_memory;
	    if (fc8_prim_desc.weights_desc() != fc8_user_weights_memory.get_desc()) {
	        fc8_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(fc8_prim_desc.weights_desc(), eng);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(fc8_user_weights_memory, fc8_weights_memory)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, fc8_user_weights_memory, fc8_weights_memory);
	    }
	
	    auto fc8_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(fc8_prim_desc.dst_desc(), eng);
	
	    // create convolution primitive and add it to net
	    net.push_back(:ref:`inner_product_forward <doxid-structdnnl_1_1inner__product__forward>`(fc8_prim_desc));
	    net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, fc7_dst_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, fc8_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, fc8_user_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, fc8_dst_memory}});
	
	    // create reorder between internal and user data if it is needed and
	    // add it to net after pooling
	    if (fc8_dst_memory != user_dst_memory) {
	        net.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(fc8_dst_memory, user_dst_memory));
	        net_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, fc8_dst_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, user_dst_memory}});
	    }
	
	    //[Execute model]
	    for (int j = 0; j < times; ++j) {
	        assert(net.size() == net_args.size() && "something is missing");
	        for (size_t i = 0; i < net.size(); ++i)
	            net.at(i).execute(s, net_args.at(i));
	    }
	    //[Execute model]
	
	    s.wait();
	}
	
	void cnn_inference_f32(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    auto begin = std::chrono::duration_cast<std::chrono::milliseconds>(
	            std::chrono::steady_clock::now().time_since_epoch())
	                         .count();
	    int times = 100;
	    simple_net(engine_kind, times);
	    auto end = std::chrono::duration_cast<std::chrono::milliseconds>(
	            std::chrono::steady_clock::now().time_since_epoch())
	                       .count();
	    std::cout << "Use time: " << (end - begin) / (times + 0.0)
	              << " ms per iteration." << std::endl;
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            cnn_inference_f32, parse_engine_kind(argc, argv));
	}
