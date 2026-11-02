.. index:: pair: page; Deconvolution Primitive Example
.. _doxid-deconvolution_example_cpp:

Deconvolution Primitive Example
===============================

This C++ API example demonstrates how to create and execute a :ref:`Deconvolution <doxid-dev_guide_convolution>` primitive in forward propagation mode.

Key optimizations included in this example:

* Creation of optimized memory format from the primitive descriptor;

* Primitive attributes with fused post-ops.

.. ref-code-block:: cpp

	/*******************************************************************************
	* Copyright 2024 Intel Corporation
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
	
	void deconvolution_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 3, // batch size
	            IC = 32, // input channels
	            IH = 13, // input height
	            IW = 13, // input width
	            OC = 64, // output channels
	            KH = 3, // weights height
	            KW = 3, // weights width
	            PH_L = 1, // height padding: left
	            PH_R = 1, // height padding: right
	            PW_L = 1, // width padding: left
	            PW_R = 1, // width padding: right
	            SH = 4, // height-wise stride
	            SW = 4, // width-wise stride
	            // In a convolution operation, the output height and
	            // width are computed as:
	            // OH = (IH - KH + PH_L + PH_R) / SH + 1
	            // OW = (IW - KW + PW_L + PW_R) / SW + 1
	            // However, in a deconvolution operation, the computation
	            // is reversed:
	            OH = (IH - 1) * SH - PH_L - PH_R + KH, // output height
	            OW = (IW - 1) * SW - PW_L - PW_R + KW; // output width
	
	    // Source (src), weights, bias, and destination (dst) tensors
	    // dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {N, IC, IH, IW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` weights_dims = {OC, IC, KH, KW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` bias_dims = {OC};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_dims = {N, OC, OH, OW};
	
	    // Strides, padding dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` strides_dims = {SH, SW};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` padding_dims_l = {PH_L, PW_L};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` padding_dims_r = {PH_R, PW_R};
	
	    // Allocate buffers.
	    std::vector<float> src_data(product(src_dims));
	    std::vector<float> weights_data(product(weights_dims));
	    std::vector<float> bias_data(OC);
	    std::vector<float> dst_data(product(dst_dims));
	
	    // Initialize src, weights, and dst tensors.
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
	
	    // Create memory objects for tensor data (src, weights, dst). In this
	    // example, NCHW layout is assumed for src and dst, and OIHW for weights.
	    auto user_src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto user_weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({weights_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::oihw}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto user_dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`({dst_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::nchw}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create memory descriptors with format_tag::any for the primitive. This
	    // enables the deconvolution primitive to choose memory layouts for an
	    // optimized primitive implementation, and these layouts may differ from the
	    // ones provided by the user.
	    auto deconv_src_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto deconv_weights_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(weights_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto deconv_dst_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(dst_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // Create memory descriptor and memory object for input bias.
	    auto user_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(bias_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::a);
	    auto user_bias_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(user_bias_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_data.data(), user_src_mem);
	    write_to_dnnl_memory(weights_data.data(), user_weights_mem);
	    write_to_dnnl_memory(bias_data.data(), user_bias_mem);
	
	    // Create primitive post-ops (ReLU).
	    const float alpha = 0.f;
	    const float beta = 0.f;
	    :ref:`post_ops <doxid-structdnnl_1_1post__ops>` deconv_ops;
	    deconv_ops.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(:ref:`algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`, alpha, beta);
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` deconv_attr;
	    deconv_attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(deconv_ops);
	
	    // Create primitive descriptor.
	    // Here we use deconvolution which is a transposed convolution.
	    // The way the weights are applied is the key difference between convolution
	    // and deconvolution. In a convolution, the weights are used to reduce
	    // the input data, while in a deconvolution, they are used to expand
	    // the input data.
	    auto deconv_pd = :ref:`deconvolution_forward::primitive_desc <doxid-structdnnl_1_1deconvolution__forward_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`,
	            :ref:`prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`, :ref:`algorithm::deconvolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a73f81608d2b7315f04c438fb8be5f99c>`,
	            deconv_src_md, deconv_weights_md, user_bias_md, deconv_dst_md,
	            strides_dims, padding_dims_l, padding_dims_r, deconv_attr);
	
	    // For now, assume that the src, weights, and dst memory layouts generated
	    // by the primitive and the ones provided by the user are identical.
	    auto deconv_src_mem = user_src_mem;
	    auto deconv_weights_mem = user_weights_mem;
	    auto deconv_dst_mem = user_dst_mem;
	
	    // Reorder the data in case the src and weights memory layouts generated by
	    // the primitive and the ones provided by the user are different. In this
	    // case, we create additional memory objects with internal buffers that will
	    // contain the reordered data. The data in dst will be reordered after the
	    // deconvolution computation has finalized.
	    if (deconv_pd.src_desc() != user_src_mem.get_desc()) {
	        deconv_src_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(deconv_pd.src_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_src_mem, deconv_src_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_src_mem, deconv_src_mem);
	    }
	
	    if (deconv_pd.weights_desc() != user_weights_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        deconv_weights_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(deconv_pd.weights_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_mem, deconv_weights_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_weights_mem, deconv_weights_mem);
	    }
	
	    if (deconv_pd.dst_desc() != user_dst_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        deconv_dst_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(deconv_pd.dst_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    }
	
	    // Create the primitive.
	    auto deconv_prim = :ref:`deconvolution_forward <doxid-structdnnl_1_1deconvolution__forward>`(deconv_pd);
	
	    // Primitive arguments.
	    std::unordered_map<int, memory> deconv_args;
	    deconv_args.insert({:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, deconv_src_mem});
	    deconv_args.insert({:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, deconv_weights_mem});
	    deconv_args.insert({:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_bias_mem});
	    deconv_args.insert({:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, deconv_dst_mem});
	
	    // Primitive execution: deconvolution with ReLU.
	    deconv_prim.execute(engine_stream, deconv_args);
	
	    // Reorder the data in case the dst memory descriptor generated by the
	    // primitive and the one provided by the user are different.
	    if (deconv_pd.dst_desc() != user_dst_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(deconv_dst_mem, user_dst_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, deconv_dst_mem, user_dst_mem);
	    } else
	        user_dst_mem = deconv_dst_mem;
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_data.data(), user_dst_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            deconvolution_example, parse_engine_kind(argc, argv));
	}

