.. index:: pair: example; lstm.cpp
.. _doxid-lstm_8cpp-example:

lstm.cpp
========

Annotated version: :ref:`LSTM RNN Primitive Example <doxid-lstm_example_cpp>`

Annotated version: :ref:`LSTM RNN Primitive Example <doxid-lstm_example_cpp>`



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
	
	void lstm_example(:ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind) {
	
	    // Create execution dnnl::engine.
	    :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-structdnnl_1_1engine>`(engine_kind, 0);
	
	    // Create dnnl::stream.
	    :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` engine_stream(:ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Tensor dimensions.
	    const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` N = 26, // batch size
	            T = 6, // time steps
	            C = 12, // channels
	            G = 4, // gates
	            L = 4, // layers
	            D = 1; // directions
	
	    // Source (src), weights, bias, and destination (dst) tensors
	    // dimensions.
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` src_dims = {T, N, C};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` weights_dims = {L, D, C, G, C};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` bias_dims = {L, D, G, C};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dst_dims = {T, N, C};
	
	    // Allocate buffers.
	    std::vector<float> src_layer_data(product(src_dims));
	    std::vector<float> weights_layer_data(product(weights_dims));
	    std::vector<float> weights_iter_data(product(weights_dims));
	    std::vector<float> dst_layer_data(product(dst_dims));
	    std::vector<float> bias_data(product(bias_dims));
	
	    // Initialize src, weights, and bias tensors.
	    std::generate(src_layer_data.begin(), src_layer_data.end(), []() {
	        static int i = 0;
	        return std::cos(i++ / 10.f);
	    });
	    std::generate(weights_layer_data.begin(), weights_layer_data.end(), []() {
	        static int i = 0;
	        return std::sin(i++ * 2.f);
	    });
	    std::generate(bias_data.begin(), bias_data.end(), []() {
	        static int i = 0;
	        return std::tanh(float(i++));
	    });
	
	    // Create memory descriptors and memory objects for src, bias, and dst.
	    auto src_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(src_dims, dt::f32, tag::tnc);
	    auto bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(bias_dims, dt::f32, tag::ldgo);
	    auto dst_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(dst_dims, dt::f32, tag::tnc);
	
	    auto src_layer_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(src_layer_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto bias_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(bias_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_layer_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(dst_layer_md, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create memory objects for weights using user's memory layout. In this
	    // example, LDIGO is assumed.
	    auto user_weights_layer_mem
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({weights_dims, dt::f32, tag::ldigo}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto user_weights_iter_mem
	            = :ref:`memory <doxid-structdnnl_1_1memory>`({weights_dims, dt::f32, tag::ldigo}, :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Write data to memory object's handle.
	    write_to_dnnl_memory(src_layer_data.data(), src_layer_mem);
	    write_to_dnnl_memory(bias_data.data(), bias_mem);
	    write_to_dnnl_memory(weights_layer_data.data(), user_weights_layer_mem);
	    write_to_dnnl_memory(weights_iter_data.data(), user_weights_iter_mem);
	
	    // Create memory descriptors for weights with format_tag::any. This enables
	    // the LSTM primitive to choose the optimized memory layout.
	    auto lstm_weights_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(weights_dims, dt::f32, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto lstm_weights_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(weights_dims, dt::f32, :ref:`tag::any <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725a100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    // Optional memory descriptors for recurrent data.
	    auto src_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`();
	    auto src_iter_c_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`();
	    auto dst_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`();
	    auto dst_iter_c_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`();
	
	    // Create primitive descriptor.
	    auto lstm_pd = :ref:`lstm_forward::primitive_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc>`(:ref:`engine <doxid-structdnnl_1_1engine>`,
	            :ref:`prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`,
	            :ref:`rnn_direction::unidirectional_left2right <doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a04f4bf4bc6a47e30f0353597e244c44a>`, src_layer_md, src_iter_md,
	            src_iter_c_md, lstm_weights_layer_md, lstm_weights_iter_md, bias_md,
	            dst_layer_md, dst_iter_md, dst_iter_c_md);
	
	    // For now, assume that the weights memory layout generated by the primitive
	    // and the ones provided by the user are identical.
	    auto lstm_weights_layer_mem = user_weights_layer_mem;
	    auto lstm_weights_iter_mem = user_weights_iter_mem;
	
	    // Reorder the data in case the weights memory layout generated by the
	    // primitive and the one provided by the user are different. In this case,
	    // we create additional memory objects with internal buffers that will
	    // contain the reordered data.
	    if (lstm_pd.weights_desc() != user_weights_layer_mem.get_desc()) {
	        lstm_weights_layer_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lstm_pd.weights_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_layer_mem, lstm_weights_layer_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_weights_layer_mem,
	                        lstm_weights_layer_mem);
	    }
	
	    if (lstm_pd.weights_iter_desc() != user_weights_iter_mem.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`()) {
	        lstm_weights_iter_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lstm_pd.weights_iter_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	        :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_weights_iter_mem, lstm_weights_iter_mem)
	                .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(engine_stream, user_weights_iter_mem,
	                        lstm_weights_iter_mem);
	    }
	
	    // Create the memory objects from the primitive descriptor. A workspace is
	    // also required for LSTM.
	    // NOTE: Here, the workspace is required for later usage in backward
	    // propagation mode.
	    auto src_iter_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lstm_pd.src_iter_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto src_iter_c_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lstm_pd.src_iter_c_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto weights_iter_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lstm_pd.weights_iter_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_iter_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lstm_pd.dst_iter_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto dst_iter_c_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lstm_pd.dst_iter_c_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	    auto workspace_mem = :ref:`memory <doxid-structdnnl_1_1memory>`(lstm_pd.workspace_desc(), :ref:`engine <doxid-structdnnl_1_1engine>`);
	
	    // Create the primitive.
	    auto lstm_prim = :ref:`lstm_forward <doxid-structdnnl_1_1lstm__forward>`(lstm_pd);
	
	    // Primitive arguments
	    std::unordered_map<int, memory> lstm_args;
	    lstm_args.insert({:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, src_layer_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, lstm_weights_layer_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, lstm_weights_iter_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, bias_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, dst_layer_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_SRC_ITER <doxid-group__dnnl__api__primitives__common_1gaf35f4f604284f1b00bb35bffd0f7a143>`, src_iter_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_SRC_ITER_C <doxid-group__dnnl__api__primitives__common_1ga8ef6969516e717208a33766542410410>`, src_iter_c_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_DST_ITER <doxid-group__dnnl__api__primitives__common_1ga13b91cbd3f531d9c90227895a275d5a6>`, dst_iter_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_DST_ITER_C <doxid-group__dnnl__api__primitives__common_1ga8b77d8716fc0ab9923d6cb409dbdf900>`, dst_iter_c_mem});
	    lstm_args.insert({:ref:`DNNL_ARG_WORKSPACE <doxid-group__dnnl__api__primitives__common_1ga550c80e1b9ba4f541202a7ac98be117f>`, workspace_mem});
	
	    // Primitive execution: LSTM.
	    lstm_prim.execute(engine_stream, lstm_args);
	
	    // Wait for the computation to finalize.
	    engine_stream.wait();
	
	    // Read data from memory object's handle.
	    read_from_dnnl_memory(dst_layer_data.data(), dst_layer_mem);
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors(lstm_example, parse_engine_kind(argc, argv));
	}
