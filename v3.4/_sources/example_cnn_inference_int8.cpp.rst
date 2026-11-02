.. index:: pair: example; cnn_inference_int8.cpp
.. _doxid-cnn_inference_int8_8cpp-example:

cnn_inference_int8.cpp
======================

This C++ API example demonstrates how to run AlexNet's conv3 and relu3 with int8 data type. Annotated version: :ref:`CNN int8 inference example <doxid-cnn_inference_int8_cpp>`

This C++ API example demonstrates how to run AlexNet's conv3 and relu3 with int8 data type. Annotated version: :ref:`CNN int8 inference example <doxid-cnn_inference_int8_cpp>`



.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2018-2022 Intel Corporation
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
	
	
	
	#include <stdexcept>
	
	#include "oneapi/dnnl/dnnl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	void simple_net_int8(:ref:`engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	    using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	    auto eng = :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(eng);
	
	    const int batch = 8;
	
	    //[Configure tensor shapes]
	    // AlexNet: conv3
	    // {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13}
	    // strides: {1, 1}
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_src_tz = {batch, 256, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_weights_tz = {384, 256, 3, 3};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_bias_tz = {384};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_dst_tz = {batch, 384, 13, 13};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_strides = {1, 1};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` conv_padding = {1, 1};
	    //[Configure tensor shapes]
	
	    //[Choose scaling factors]
	    // Choose scaling factors for input, weight and output
	    std::vector<float> src_scales = {1.8f};
	    std::vector<float> weight_scales = {2.0f};
	    std::vector<float> dst_scales = {0.55f};
	
	    //[Choose scaling factors]
	
	    //[Set scaling mask]
	    const int src_mask = 0;
	    const int weight_mask = 0;
	    const int dst_mask = 0;
	    //[Set scaling mask]
	
	    // Allocate input and output buffers for user data
	    std::vector<float> user_src(batch * 256 * 13 * 13);
	    std::vector<float> user_dst(batch * 384 * 13 * 13);
	
	    // Allocate and fill buffers for weights and bias
	    std::vector<float> conv_weights(product(conv_weights_tz));
	    std::vector<float> conv_bias(product(conv_bias_tz));
	
	    //[Allocate buffers]
	    auto user_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_src_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, eng);
	    write_to_dnnl_memory(user_src.data(), user_src_memory);
	    auto user_weights_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_weights_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::oihw}, eng);
	    write_to_dnnl_memory(conv_weights.data(), user_weights_memory);
	    auto user_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_bias_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x}, eng);
	    write_to_dnnl_memory(conv_bias.data(), user_bias_memory);
	    //[Allocate buffers]
	
	    //[Create convolution memory descriptors]
	    auto conv_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_src_tz}, dt::u8, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_bias_tz}, dt::s8, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_weights_tz}, dt::s8, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto conv_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({conv_dst_tz}, dt::u8, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    //[Create convolution memory descriptors]
	
	    //[Configure scaling]
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` conv_attr;
	    conv_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mask);
	    conv_attr.set_scales_mask(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, weight_mask);
	    conv_attr.set_scales_mask(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mask);
	
	    // Prepare dst scales
	    auto dst_scale_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({1}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x);
	    auto dst_scale_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_scale_md, eng);
	    write_to_dnnl_memory(dst_scales.data(), dst_scale_memory);
	    //[Configure scaling]
	
	    //[Configure post-ops]
	    const float ops_alpha = 0.f; // relu negative slope
	    const float ops_beta = 0.f;
	    :ref:`post_ops <doxid-structdnnl_1_1post__ops>` ops;
	    ops.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(:ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, ops_alpha, ops_beta);
	    conv_attr.set_post_ops(ops);
	    //[Configure post-ops]
	
	    // check if int8 convolution is supported
	    try {
	        :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng, :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`,
	                :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`, conv_src_md, conv_weights_md,
	                conv_bias_md, conv_dst_md, conv_strides, conv_padding,
	                conv_padding, conv_attr);
	    } catch (:ref:`error <doxid-structdnnl_1_1error>` &e) {
	        if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`)
	            throw example_allows_unimplemented {
	                    "No int8 convolution implementation is available for this "
	                    "platform.\n"
	                    "Please refer to the developer guide for details."};
	
	        // on any other error just re-throw
	        throw;
	    }
	
	    //[Create convolution primitive descriptor]
	    auto conv_prim_desc = :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(eng,
	            :ref:`prop_kind::forward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`, :ref:`algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`, conv_src_md,
	            conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
	            conv_padding, conv_padding, conv_attr);
	    //[Create convolution primitive descriptor]
	
	    //[Quantize data and weights]
	    auto conv_src_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_prim_desc.src_desc(), eng);
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` src_attr;
	    src_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_mask);
	    auto src_scale_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({1}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x);
	    auto src_scale_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(src_scale_md, eng);
	    write_to_dnnl_memory(src_scales.data(), src_scale_memory);
	    auto src_reorder_pd
	            = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(eng, user_src_memory.get_desc(), eng,
	                    conv_src_memory.get_desc(), src_attr);
	    auto src_reorder = :ref:`reorder <doxid-structdnnl_1_1reorder>`(src_reorder_pd);
	    src_reorder.execute(s,
	            {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, user_src_memory}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_src_memory},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_scale_memory}});
	
	    auto conv_weights_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_prim_desc.weights_desc(), eng);
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` weight_attr;
	    weight_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, weight_mask);
	    auto wei_scale_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({1}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::x);
	    auto wei_scale_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(wei_scale_md, eng);
	    write_to_dnnl_memory(weight_scales.data(), wei_scale_memory);
	    auto weight_reorder_pd
	            = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(eng, user_weights_memory.get_desc(), eng,
	                    conv_weights_memory.get_desc(), weight_attr);
	    auto weight_reorder = :ref:`reorder <doxid-structdnnl_1_1reorder>`(weight_reorder_pd);
	    weight_reorder.execute(s,
	            {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, user_weights_memory},
	                    {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_weights_memory},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, wei_scale_memory}});
	
	    auto conv_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_prim_desc.bias_desc(), eng);
	    write_to_dnnl_memory(conv_bias.data(), conv_bias_memory);
	    //[Quantize data and weights]
	
	    auto conv_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(conv_prim_desc.dst_desc(), eng);
	
	    //[Create convolution primitive]
	    auto conv = :ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>`(conv_prim_desc);
	    conv.execute(s,
	            {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src_memory},
	                    {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_weights_memory},
	                    {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv_bias_memory},
	                    {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst_memory},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_scale_memory},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_scale_memory},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_scale_memory}});
	    //[Create convolution primitive]
	
	    auto user_dst_memory = :ref:`memory <doxid-structdnnl_1_1memory>`({{conv_dst_tz}, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, eng);
	    write_to_dnnl_memory(user_dst.data(), user_dst_memory);
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` dst_attr;
	    dst_attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, dst_mask);
	    auto dst_reorder_pd
	            = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(eng, conv_dst_memory.get_desc(), eng,
	                    user_dst_memory.get_desc(), dst_attr);
	    auto dst_reorder = :ref:`reorder <doxid-structdnnl_1_1reorder>`(dst_reorder_pd);
	    dst_reorder.execute(s,
	            {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv_dst_memory}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, user_dst_memory},
	                    {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, dst_scale_memory}});
	    //[Dequantize the result]
	
	    s.wait();
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            simple_net_int8, parse_engine_kind(argc, argv));
	}
