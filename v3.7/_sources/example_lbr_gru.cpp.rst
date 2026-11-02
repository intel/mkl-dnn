.. index:: pair: example; lbr_gru.cpp
.. _doxid-lbr_gru_8cpp-example:

lbr_gru.cpp
===========

Annotated version: :ref:`Linear-Before-Reset GRU RNN Primitive Example <doxid-lbr_gru_example_cpp>`

Annotated version: :ref:`Linear-Before-Reset GRU RNN Primitive Example <doxid-lbr_gru_example_cpp>`



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
	
	#include "dnnl.hpp"
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using :ref:`tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` = :ref:`memory::format_tag <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
	using :ref:`dt <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` = :ref:`memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
	
	void lbr_gru_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 2, // batch size
	            T = 3, // time steps
	            IC = 2, // src channels
	            OC = 3, // dst channels
	            G = 3, // gates
	            L = 1, // layers
	            D = 1, // directions
	            E = 1; // extra Bias number. Extra Bias for u' gate
	
	    // Source (src), weights, bias, attention, and destination (dst) tensors
	    // dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {T, N, IC};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` weights_layer_dims = {L, D, IC, G, OC};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` weights_iter_dims = {L, D, OC, G, OC};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` bias_dims = {L, D, G + E, OC};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_layer_dims = {T, N, OC};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_iter_dims = {L, D, N, OC};
	
	    // Allocate buffers.
	    std::vector<float> src_layer_data(product(src_dims));
	    std::vector<float> weights_layer_data(product(weights_layer_dims));
	    std::vector<float> weights_iter_data(product(weights_iter_dims));
	    std::vector<float> bias_data(product(bias_dims));
	    std::vector<float> dst_layer_data(product(dst_layer_dims));
	    std::vector<float> dst_iter_data(product(dst_iter_dims));
	
	    // Initialize src, weights, and bias tensors.
	    std::generate(src_layer_data.begin(), src_layer_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	    std::generate(weights_layer_data.begin(), weights_layer_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 2.f);
	    });
	    std::generate(weights_iter_data.begin(), weights_iter_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 2.f);
	    });
	    std::generate(bias_data.begin(), bias_data.end(), []() {
	        static int i = 0;
	        return std::tanh(float(i++));
	    });
	
	    // Create memory descriptors and memory objects for src, bias, and dst.
	    auto src_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::tnc);
	    auto bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(bias_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::ldgo);
	    auto dst_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(dst_layer_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::tnc);
	
	    auto src_layer_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_layer_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto bias_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(bias_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_layer_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_layer_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create memory objects for weights using user's memory layout. In this
	    // example, LDIGO (num_layers, num_directions, input_channels, num_gates,
	    // output_channels) is assumed.
	    auto user_weights_layer_mem
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({weights_layer_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::ldigo}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto user_weights_iter_mem
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({weights_iter_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, tag::ldigo}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    // For GRU cells, the gates order is update, reset and output
	    // gate except the bias. For the bias tensor, the gates order is
	    // u, r, o and u' gate.
	    write_to_dnnl_memory(src_layer_data.data(), src_layer_mem);
	    write_to_dnnl_memory(bias_data.data(), bias_mem);
	    write_to_dnnl_memory(weights_layer_data.data(), user_weights_layer_mem);
	    write_to_dnnl_memory(weights_iter_data.data(), user_weights_iter_mem);
	
	    // Create memory descriptors for weights with format_tag::any. This enables
	    // the lbr_gru primitive to choose the optimized memory layout.
	    auto weights_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(weights_layer_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto weights_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(weights_iter_dims, :ref:`dt::f32 <doxid-group__dnnl__api__accumulation__mode_1ggad6b8b3ca2e61b8a9703227f4d58ac215a512dc597be7ae761876315165dc8bd2e>`, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // Optional memory descriptors for recurrent data.
	    // Default memory descriptor for initial hidden states of the GRU cells
	    auto src_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`();
	    auto dst_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`();
	
	    // Create primitive descriptor.
	    auto lbr_gru_pd = :ref:`lbr_gru_forward::primitive_desc <doxid-structdnnl_1_1lbr__gru__forward_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`,
	            :ref:`prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`,
	            :ref:`rnn_direction::unidirectional_left2right <doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a04f4bf4bc6a47e30f0353597e244c44a>`, src_layer_md, src_iter_md,
	            weights_layer_md, weights_iter_md, bias_md, dst_layer_md,
	            dst_iter_md);
	
	    // For now, assume that the weights memory layout generated by the primitive
	    // and the ones provided by the user are identical.
	    auto weights_layer_mem = user_weights_layer_mem;
	    auto weights_iter_mem = user_weights_iter_mem;
	
	    // Reorder the data in case the weights memory layout generated by the
	    // primitive and the one provided by the user are different. In this case,
	    // we create additional memory objects with internal buffers that will
	    // contain the reordered data.
	    if (lbr_gru_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
	        weights_layer_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lbr_gru_pd.weights_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_layer_mem, weights_layer_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_weights_layer_mem,
	                        weights_layer_mem);
	    }
	
	    if (lbr_gru_pd.weights_iter_desc() != user_weights_iter_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        weights_iter_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lbr_gru_pd.weights_iter_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_iter_mem, weights_iter_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(
	                        engine_stream, user_weights_iter_mem, weights_iter_mem);
	    }
	
	    // Create the memory objects from the primitive descriptor. A workspace is
	    // also required for Linear-Before-Reset GRU RNN.
	    // NOTE: Here, the workspace is required for later usage in backward
	    // propagation mode.
	    auto src_iter_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lbr_gru_pd.src_iter_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_iter_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lbr_gru_pd.dst_iter_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto workspace_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lbr_gru_pd.workspace_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create the primitive.
	    auto lbr_gru_prim = :ref:`lbr_gru_forward <doxid-structdnnl_1_1lbr__gru__forward>`(lbr_gru_pd);
	
	    // Primitive arguments
	    std::unordered_map<int, memory> lbr_gru_args;
	    lbr_gru_args.insert({:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, src_layer_mem});
	    lbr_gru_args.insert({:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, weights_layer_mem});
	    lbr_gru_args.insert({:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, weights_iter_mem});
	    lbr_gru_args.insert({:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, bias_mem});
	    lbr_gru_args.insert({:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, dst_layer_mem});
	    lbr_gru_args.insert({:ref:`DNNL_ARG_SRC_ITER <doxid-group__dnnl__api__primitives__common_1gaf35f4f604284f1b00bb35bffd0f7a143>`, src_iter_mem});
	    lbr_gru_args.insert({:ref:`DNNL_ARG_DST_ITER <doxid-group__dnnl__api__primitives__common_1ga13b91cbd3f531d9c90227895a275d5a6>`, dst_iter_mem});
	    lbr_gru_args.insert({:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, workspace_mem});
	
	    // Primitive execution: lbr_gru.
	    lbr_gru_prim.execute(engine_stream, lbr_gru_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_layer_data.data(), dst_layer_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(
	            lbr_gru_example, parse_engine_kind(argc, argv));
	}
