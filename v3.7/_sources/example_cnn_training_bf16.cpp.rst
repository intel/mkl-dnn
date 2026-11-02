.. index:: pair: example; cnn_training_bf16.cpp
.. _doxid-cnn_training_bf16_8cpp-example:

cnn_training_bf16.cpp
=====================

This C++ API example demonstrates how to build an AlexNet model training using the bfloat16 data type. Annotated version: :ref:`CNN bf16 training example <doxid-cnn_training_bf16_cpp>`

This C++ API example demonstrates how to build an AlexNet model training using the bfloat16 data type. Annotated version: :ref:`CNN bf16 training example <doxid-cnn_training_bf16_cpp>`



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
	
	
	#include <cassert>
	#include <cmath>
	#include <iostream>
	#include <stdexcept>
	
	#include "oneapi/dnnl/dnnl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	void simple_net(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	    using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	    auto eng = :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	
	    // Vector of primitives and their execute arguments
	    std::vector<primitive> net_fwd, net_bwd;
	    std::vector<std::unordered_map<int, memory>> net_fwd_args, net_bwd_args;
	
	    const int batch = 32;
	
	    // float data type is used for user data
	    std::vector<float> net_src(batch * 3 * 227 * 227);
	
	    // initializing non-zero values for src
	    for (size_t i = 0; i < net_src.size(); ++i)
	        net_src[i] = sinf((float)i);
	
	    // AlexNet: conv
	    // {batch, 3, 227, 227} (x) {96, 3, 11, 11} -> {batch, 96, 55, 55}
	    // strides: {4, 4}
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_src_tz = {batch, 3, 227, 227};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_weights_tz = {96, 3, 11, 11};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_bias_tz = {96};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_dst_tz = {batch, 96, 55, 55};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_strides = {4, 4};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_padding = {0, 0};
	
	    // float data type is used for user data
	    std::vector<float> conv_weights(product(conv_weights_tz));
	    std::vector<float> conv_bias(product(conv_bias_tz));
	
	    // initializing non-zero values for weights and bias
	    for (size_t i = 0; i < conv_weights.size(); ++i)
	        conv_weights[i] = sinf((float)i);
	    for (size_t i = 0; i < conv_bias.size(); ++i)
	        conv_bias[i] = sinf((float)i);
	
	    // create memory for user data
	    auto conv_user_src_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, eng);
	    write_to_dnnl_memory(net_src.data(), conv_user_src_memory);
	
	    auto conv_user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::oihw}, eng);
	    write_to_dnnl_memory(conv_weights.data(), conv_user_weights_memory);
	
	    auto conv_user_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(conv_bias.data(), conv_user_bias_memory);
	
	    // create memory descriptors for bfloat16 convolution data w/ no specified
	    // format tag(`any`)
	    // tag `any` lets a primitive(convolution in this case)
	    // chose the memory format preferred for best performance.
	    auto conv_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_src_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_weights_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_dst_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    // here bias data type is set to bf16.
	    // additionally, f32 data type is supported for bf16 convolution.
	    auto conv_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_bias_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a convolution primitive descriptor
	
	    // check if bf16 convolution is supported
	    try {
	        :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`,
	                :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`, conv_src_md, conv_weights_md,
	                conv_bias_md, conv_dst_md, conv_strides, conv_padding,
	                conv_padding);
	    } catch (:ref:`error <doxid-structdnnl_1_1error>` &e) {
	        if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`)
	            throw example_allows_unimplemented {
	                    "No bf16 convolution implementation is available for this "
	                    "platform.\n"
	                    "Please refer to the developer guide for details."};
	
	        // on any other error just re-throw
	        throw;
	    }
	
	    auto conv_pd = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`,
	            :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`, conv_src_md, conv_weights_md,
	            conv_bias_md, conv_dst_md, conv_strides, conv_padding,
	            conv_padding);
	
	    // create reorder primitives between user input and conv src if needed
	    auto conv_src_memory = conv_user_src_memory;
	    if (conv_pd.src_desc() != conv_user_src_memory.get_desc()) {
	        conv_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.src_desc(), eng);
	        net_fwd.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(conv_user_src_memory, conv_src_memory));
	        net_fwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv_user_src_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_src_memory}});
	    }
	
	    auto conv_weights_memory = conv_user_weights_memory;
	    if (conv_pd.weights_desc() != conv_user_weights_memory.get_desc()) {
	        conv_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.weights_desc(), eng);
	        net_fwd.push_back(
	                :ref:`reorder <doxid-structdnnl_1_1reorder>`(conv_user_weights_memory, conv_weights_memory));
	        net_fwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv_user_weights_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_weights_memory}});
	    }
	
	    // convert bias from f32 to bf16 as convolution descriptor is created with
	    // bias data type as bf16.
	    auto conv_bias_memory = conv_user_bias_memory;
	    if (conv_pd.bias_desc() != conv_user_bias_memory.get_desc()) {
	        conv_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.bias_desc(), eng);
	        net_fwd.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(conv_user_bias_memory, conv_bias_memory));
	        net_fwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv_user_bias_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_bias_memory}});
	    }
	
	    // create memory for conv dst
	    auto conv_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_pd.dst_desc(), eng);
	
	    // finally create a convolution primitive
	    net_fwd.push_back(:ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv_pd));
	    net_fwd_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src_memory},
	            {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_weights_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv_bias_memory},
	            {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst_memory}});
	
	    // AlexNet: relu
	    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` relu_data_tz = {batch, 96, 55, 55};
	    const float negative_slope = 0.0f;
	
	    // create relu primitive desc
	    // keep memory format tag of source same as the format tag of convolution
	    // output in order to avoid reorder
	    auto relu_pd = :ref:`eltwise_forward::primitive_desc <doxid-structdnnl_1_1eltwise__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`,
	            :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, conv_pd.dst_desc(), conv_pd.dst_desc(),
	            negative_slope);
	
	    // create relu dst memory
	    auto relu_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(relu_pd.dst_desc(), eng);
	
	    // finally create a relu primitive
	    net_fwd.push_back(:ref:`eltwise_forward <doxid-structdnnl_1_1eltwise__forward>`(relu_pd));
	    net_fwd_args.push_back(
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_dst_memory}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, relu_dst_memory}});
	
	    // AlexNet: lrn
	    // {batch, 96, 55, 55} -> {batch, 96, 55, 55}
	    // local size: 5
	    // alpha: 0.0001
	    // beta: 0.75
	    // k: 1.0
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` lrn_data_tz = {batch, 96, 55, 55};
	    const uint32_t local_size = 5;
	    const float alpha = 0.0001f;
	    const float beta = 0.75f;
	    const float k = 1.0f;
	
	    // create a lrn primitive descriptor
	    auto lrn_pd = :ref:`lrn_forward::primitive_desc <doxid-structdnnl_1_1lrn__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`,
	            :ref:`algorithm::lrn_across_channels <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab9e2d858b551792385a4b5b86672b24b>`, relu_pd.dst_desc(),
	            relu_pd.dst_desc(), local_size, alpha, beta, k);
	
	    // create lrn dst memory
	    auto lrn_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(lrn_pd.dst_desc(), eng);
	
	    // create workspace only in training and only for forward primitive
	    // query lrn_pd for workspace, this memory will be shared with forward lrn
	    auto lrn_workspace_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(lrn_pd.workspace_desc(), eng);
	
	    // finally create a lrn primitive
	    net_fwd.push_back(:ref:`lrn_forward <doxid-structdnnl_1_1lrn__forward>`(lrn_pd));
	    net_fwd_args.push_back(
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, relu_dst_memory}, {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, lrn_dst_memory},
	                    {:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, lrn_workspace_memory}});
	
	    // AlexNet: pool
	    // {batch, 96, 55, 55} -> {batch, 96, 27, 27}
	    // kernel: {3, 3}
	    // strides: {2, 2}
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool_dst_tz = {batch, 96, 27, 27};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool_kernel = {3, 3};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool_strides = {2, 2};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool_dilation = {0, 0};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` pool_padding = {0, 0};
	
	    // create memory for pool dst data in user format
	    auto pool_user_dst_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{pool_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, eng);
	
	    // create pool dst memory descriptor in format any for bfloat16 data type
	    auto pool_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({pool_dst_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // create a pooling primitive descriptor
	    auto pool_pd = :ref:`pooling_forward::primitive_desc <doxid-structdnnl_1_1pooling__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`,
	            :ref:`algorithm::pooling_max <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`, lrn_dst_memory.get_desc(), pool_dst_md,
	            pool_strides, pool_kernel, pool_dilation, pool_padding,
	            pool_padding);
	
	    // create pooling workspace memory if training
	    auto pool_workspace_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(pool_pd.workspace_desc(), eng);
	
	    // create a pooling primitive
	    net_fwd.push_back(:ref:`pooling_forward <doxid-structdnnl_1_1pooling__forward>`(pool_pd));
	    // leave DST unknown for now (see the next reorder)
	    net_fwd_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, lrn_dst_memory},
	            // delay putting DST until reorder (if needed)
	            {:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, pool_workspace_memory}});
	
	    // create reorder primitive between pool dst and user dst format
	    // if needed
	    auto pool_dst_memory = pool_user_dst_memory;
	    if (pool_pd.dst_desc() != pool_user_dst_memory.get_desc()) {
	        pool_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(pool_pd.dst_desc(), eng);
	        net_fwd_args.back().insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool_dst_memory});
	
	        net_fwd.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(pool_dst_memory, pool_user_dst_memory));
	        net_fwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, pool_dst_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, pool_user_dst_memory}});
	    } else {
	        net_fwd_args.back().insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool_dst_memory});
	    }
	
	    //-----------------------------------------------------------------------
	    //----------------- Backward Stream -------------------------------------
	    // ... user diff_data in float data type ...
	    std::vector<float> net_diff_dst(batch * 96 * 27 * 27);
	    for (size_t i = 0; i < net_diff_dst.size(); ++i)
	        net_diff_dst[i] = sinf((float)i);
	
	    // create memory for user diff dst data stored in float data type
	    auto pool_user_diff_dst_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{pool_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, eng);
	    write_to_dnnl_memory(net_diff_dst.data(), pool_user_diff_dst_memory);
	
	    // Backward pooling
	    // create memory descriptors for pooling
	    auto pool_diff_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({lrn_data_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto pool_diff_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({pool_dst_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // backward primitive descriptor needs to hint forward descriptor
	    auto pool_bwd_pd = :ref:`pooling_backward::primitive_desc <doxid-structdnnl_1_1pooling__backward_1_1primitive__desc>`(eng,
	            :ref:`algorithm::pooling_max <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a8c73d4bb88a0497586a74256bb338e88>`, pool_diff_src_md, pool_diff_dst_md,
	            pool_strides, pool_kernel, pool_dilation, pool_padding,
	            pool_padding, pool_pd);
	
	    // create reorder primitive between user diff dst and pool diff dst
	    // if required
	    auto pool_diff_dst_memory = pool_user_diff_dst_memory;
	    if (pool_dst_memory.get_desc() != pool_user_diff_dst_memory.get_desc()) {
	        pool_diff_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(pool_dst_memory.get_desc(), eng);
	        net_bwd.push_back(
	                :ref:`reorder <doxid-structdnnl_1_1reorder>`(pool_user_diff_dst_memory, pool_diff_dst_memory));
	        net_bwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, pool_user_diff_dst_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, pool_diff_dst_memory}});
	    }
	
	    // create memory for pool diff src
	    auto pool_diff_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(pool_bwd_pd.diff_src_desc(), eng);
	
	    // finally create backward pooling primitive
	    net_bwd.push_back(:ref:`pooling_backward <doxid-structdnnl_1_1pooling__backward>`(pool_bwd_pd));
	    net_bwd_args.push_back({{:ref:`DNNL_ARG_DIFF_DST <doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`, pool_diff_dst_memory},
	            {:ref:`DNNL_ARG_DIFF_SRC <doxid-group__dnnl__api__primitives__common_1ga18ee0e360399cfe9d3b58a13dfcb9333>`, pool_diff_src_memory},
	            {:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, pool_workspace_memory}});
	
	    // Backward lrn
	    auto lrn_diff_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({lrn_data_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    const auto &lrn_diff_src_md = lrn_diff_dst_md;
	
	    // create backward lrn primitive descriptor
	    auto lrn_bwd_pd = :ref:`lrn_backward::primitive_desc <doxid-structdnnl_1_1lrn__backward_1_1primitive__desc>`(eng,
	            :ref:`algorithm::lrn_across_channels <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab9e2d858b551792385a4b5b86672b24b>`, lrn_diff_src_md, lrn_diff_dst_md,
	            lrn_pd.src_desc(), local_size, alpha, beta, k, lrn_pd);
	
	    // create reorder primitive between pool diff src and lrn diff dst
	    // if required
	    auto lrn_diff_dst_memory = pool_diff_src_memory;
	    if (lrn_diff_dst_memory.get_desc() != lrn_bwd_pd.diff_dst_desc()) {
	        lrn_diff_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(lrn_bwd_pd.diff_dst_desc(), eng);
	        net_bwd.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(pool_diff_src_memory, lrn_diff_dst_memory));
	        net_bwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, pool_diff_src_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, lrn_diff_dst_memory}});
	    }
	
	    // create memory for lrn diff src
	    auto lrn_diff_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(lrn_bwd_pd.diff_src_desc(), eng);
	
	    // finally create a lrn backward primitive
	    // backward lrn needs src: relu dst in this topology
	    net_bwd.push_back(:ref:`lrn_backward <doxid-structdnnl_1_1lrn__backward>`(lrn_bwd_pd));
	    net_bwd_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, relu_dst_memory},
	            {:ref:`DNNL_ARG_DIFF_DST <doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`, lrn_diff_dst_memory},
	            {:ref:`DNNL_ARG_DIFF_SRC <doxid-group__dnnl__api__primitives__common_1ga18ee0e360399cfe9d3b58a13dfcb9333>`, lrn_diff_src_memory},
	            {:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, lrn_workspace_memory}});
	
	    // Backward relu
	    auto relu_diff_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({relu_data_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto relu_diff_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({relu_data_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto relu_src_md = conv_pd.dst_desc();
	
	    // create backward relu primitive_descriptor
	    auto relu_bwd_pd = :ref:`eltwise_backward::primitive_desc <doxid-structdnnl_1_1eltwise__backward_1_1primitive__desc>`(eng,
	            :ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, relu_diff_src_md, relu_diff_dst_md,
	            relu_src_md, negative_slope, relu_pd);
	
	    // create reorder primitive between lrn diff src and relu diff dst
	    // if required
	    auto relu_diff_dst_memory = lrn_diff_src_memory;
	    if (relu_diff_dst_memory.get_desc() != relu_bwd_pd.diff_dst_desc()) {
	        relu_diff_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(relu_bwd_pd.diff_dst_desc(), eng);
	        net_bwd.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(lrn_diff_src_memory, relu_diff_dst_memory));
	        net_bwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, lrn_diff_src_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, relu_diff_dst_memory}});
	    }
	
	    // create memory for relu diff src
	    auto relu_diff_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(relu_bwd_pd.diff_src_desc(), eng);
	
	    // finally create a backward relu primitive
	    net_bwd.push_back(:ref:`eltwise_backward <doxid-structdnnl_1_1eltwise__backward>`(relu_bwd_pd));
	    net_bwd_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_dst_memory},
	            {:ref:`DNNL_ARG_DIFF_DST <doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`, relu_diff_dst_memory},
	            {:ref:`DNNL_ARG_DIFF_SRC <doxid-group__dnnl__api__primitives__common_1ga18ee0e360399cfe9d3b58a13dfcb9333>`, relu_diff_src_memory}});
	
	    // Backward convolution with respect to weights
	    // create user format diff weights and diff bias memory for float data type
	
	    auto conv_user_diff_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, eng);
	    auto conv_diff_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	
	    // create memory descriptors for bfloat16 convolution data
	    auto conv_bwd_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_src_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_diff_weights_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_weights_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_diff_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_dst_tz}, :ref:`dt::bf16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725afe2904d9fb3b0f4a81c92b03dec11424>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // use diff bias provided by the user
	    auto conv_diff_bias_md = conv_diff_bias_memory.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`();
	
	    // create backward convolution primitive descriptor
	    auto conv_bwd_weights_pd = :ref:`convolution_backward_weights::primitive_desc <doxid-structdnnl_1_1convolution__backward__weights_1_1primitive__desc>`(eng,
	            :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`, conv_bwd_src_md,
	            conv_diff_weights_md, conv_diff_bias_md, conv_diff_dst_md,
	            conv_strides, conv_padding, conv_padding, conv_pd);
	
	    // for best performance convolution backward might chose
	    // different memory format for src and diff_dst
	    // than the memory formats preferred by forward convolution
	    // for src and dst respectively
	    // create reorder primitives for src from forward convolution to the
	    // format chosen by backward convolution
	    auto conv_bwd_src_memory = conv_src_memory;
	    if (conv_bwd_weights_pd.src_desc() != conv_src_memory.get_desc()) {
	        conv_bwd_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_bwd_weights_pd.src_desc(), eng);
	        net_bwd.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(conv_src_memory, conv_bwd_src_memory));
	        net_bwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv_src_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_bwd_src_memory}});
	    }
	
	    // create reorder primitives for diff_dst between diff_src from relu_bwd
	    // and format preferred by conv_diff_weights
	    auto conv_diff_dst_memory = relu_diff_src_memory;
	    if (conv_bwd_weights_pd.diff_dst_desc()
	            != relu_diff_src_memory.get_desc()) {
	        conv_diff_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_bwd_weights_pd.diff_dst_desc(), eng);
	        net_bwd.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(relu_diff_src_memory, conv_diff_dst_memory));
	        net_bwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, relu_diff_src_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_diff_dst_memory}});
	    }
	
	    // create backward convolution primitive
	    net_bwd.push_back(:ref:`convolution_backward_weights <doxid-structdnnl_1_1convolution__backward__weights>`(conv_bwd_weights_pd));
	    net_bwd_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_bwd_src_memory},
	            {:ref:`DNNL_ARG_DIFF_DST <doxid-group__dnnl__api__primitives__common_1gac9302f4cbd2668bf9a98ba99d752b971>`, conv_diff_dst_memory},
	            // delay putting DIFF_WEIGHTS until reorder (if needed)
	            {:ref:`DNNL_ARG_DIFF_BIAS <doxid-group__dnnl__api__primitives__common_1ga1cd79979dda6df65ec45eef32a839901>`, conv_diff_bias_memory}});
	
	    // create reorder primitives between conv diff weights and user diff weights
	    // if needed
	    auto conv_diff_weights_memory = conv_user_diff_weights_memory;
	    if (conv_bwd_weights_pd.diff_weights_desc()
	            != conv_user_diff_weights_memory.get_desc()) {
	        conv_diff_weights_memory
	                = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_bwd_weights_pd.diff_weights_desc(), eng);
	        net_bwd_args.back().insert(
	                {:ref:`DNNL_ARG_DIFF_WEIGHTS <doxid-group__dnnl__api__primitives__common_1ga3324092ef421f77aebee83b0117cac60>`, conv_diff_weights_memory});
	
	        net_bwd.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(
	                conv_diff_weights_memory, conv_user_diff_weights_memory));
	        net_bwd_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv_diff_weights_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_user_diff_weights_memory}});
	    } else {
	        net_bwd_args.back().insert(
	                {:ref:`DNNL_ARG_DIFF_WEIGHTS <doxid-group__dnnl__api__primitives__common_1ga3324092ef421f77aebee83b0117cac60>`, conv_diff_weights_memory});
	    }
	
	    // didn't we forget anything?
	    assert(net_fwd.size() == net_fwd_args.size() && "something is missing");
	    assert(net_bwd.size() == net_bwd_args.size() && "something is missing");
	
	    int n_iter = 1; // number of iterations for training
	    // execute
	    while (n_iter) {
	        // forward
	        for (size_t i = 0; i < net_fwd.size(); ++i)
	            net_fwd.at(i).execute(s, net_fwd_args.at(i));
	
	        // update net_diff_dst
	        // auto net_output = pool_user_dst_memory.get_data_handle();
	        // ..user updates net_diff_dst using net_output...
	        // some user defined func update_diff_dst(net_diff_dst.data(),
	        // net_output)
	
	        for (size_t i = 0; i < net_bwd.size(); ++i)
	            net_bwd.at(i).execute(s, net_bwd_args.at(i));
	        // update weights and bias using diff weights and bias
	        //
	        // auto net_diff_weights
	        //     = conv_user_diff_weights_memory.get_data_handle();
	        // auto net_diff_bias = conv_diff_bias_memory.get_data_handle();
	        //
	        // ...user updates weights and bias using diff weights and bias...
	        //
	        // some user defined func update_weights(conv_weights.data(),
	        // conv_bias.data(), net_diff_weights, net_diff_bias);
	
	        --n_iter;
	    }
	
	    s.wait();
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(simple_net, parse_engine_kind(argc, argv));
	}
