.. index:: pair: example; cpu_rnn_inference_int8.cpp
.. _doxid-cpu_rnn_inference_int8_8cpp-example:

cpu_rnn_inference_int8.cpp
==========================

This C++ API example demonstrates how to build GNMT model inference. Annotated version: :ref:`RNN int8 inference example <doxid-cpu_rnn_inference_int8_cpp>`

This C++ API example demonstrates how to build GNMT model inference. Annotated version: :ref:`RNN int8 inference example <doxid-cpu_rnn_inference_int8_cpp>`



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
	
	
	
	#include <assert.h>
	
	#include <cstring>
	#include <iostream>
	#include <math.h>
	#include <numeric>
	#include <string>
	
	#include "oneapi/dnnl/dnnl.hpp"
	
	#include "example_utils.hpp"
	
	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	using dim_t = :ref:`dnnl::memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`;
	
	const dim_t batch = 32;
	const dim_t src_seq_length_max = 10;
	const dim_t tgt_seq_length_max = 10;
	
	const dim_t feature_size = 256;
	
	const dim_t enc_bidir_n_layers = 1;
	const dim_t enc_unidir_n_layers = 3;
	const dim_t dec_n_layers = 4;
	
	const int lstm_n_gates = 4;
	
	std::vector<int32_t> weighted_src_layer(batch *feature_size, 1);
	std::vector<float> alignment_model(
	        src_seq_length_max *batch *feature_size, 1.0f);
	std::vector<float> alignments(src_seq_length_max *batch, 1.0f);
	std::vector<float> exp_sums(batch, 1.0f);
	
	void compute_weighted_annotations(float *weighted_annotations,
	        dim_t src_seq_length_max, dim_t batch, dim_t feature_size,
	        float *weights_annot, float *annotations) {
	    // annotations(aka enc_dst_layer) is (t, n, 2c)
	    // weights_annot is (2c, c)
	
	    dim_t num_weighted_annotations = src_seq_length_max * batch;
	    // annotation[i] = GEMM(weights_annot, enc_dst_layer[i]);
	    :ref:`dnnl_sgemm <doxid-group__dnnl__api__blas_1ga75ee119765bdac249200fda42c0617f8>`('N', 'N', num_weighted_annotations, feature_size, feature_size,
	            1.f, annotations, feature_size, weights_annot, feature_size, 0.f,
	            weighted_annotations, feature_size);
	}
	
	void compute_sum_of_rows(
	        int8_t *a, dim_t rows, dim_t cols, int32_t *a_reduced) {
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(1)
	    for (dim_t i = 0; i < cols; i++) {
	        a_reduced[i] = 0;
	        for (dim_t j = 0; j < rows; j++) {
	            a_reduced[i] += (int32_t)a[i * rows + j];
	        }
	    }
	}
	
	void compute_attention(float *context_vectors, dim_t src_seq_length_max,
	        dim_t batch, dim_t feature_size, int8_t *weights_src_layer,
	        float weights_src_layer_scale, int32_t *compensation,
	        uint8_t *dec_src_layer, float dec_src_layer_scale,
	        float dec_src_layer_shift, uint8_t *annotations,
	        float *weighted_annotations, float *weights_alignments) {
	    // dst_iter : (n, c) matrix
	    // src_layer: (n, c) matrix
	    // weighted_annotations (t, n, c)
	
	    // weights_yi is (c, c)
	    // weights_ai is (c, 1)
	    // tmp[i] is (n, c)
	    // a[i] is (n, 1)
	    // p is (n, 1)
	
	    // first we precompute the weighted_dec_src_layer
	    int32_t co = 0;
	    :ref:`dnnl_gemm_u8s8s32 <doxid-group__dnnl__api__blas_1gaef24848fd198d8a178d3ad95a78c1767>`('N', 'N', 'F', batch, feature_size, feature_size, 1.f,
	            dec_src_layer, feature_size, 0, weights_src_layer, feature_size, 0,
	            0.f, weighted_src_layer.data(), feature_size, &co);
	
	    // then we compute the alignment model
	    float *alignment_model_ptr = alignment_model.data();
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
	    for (dim_t i = 0; i < src_seq_length_max; i++) {
	        for (dim_t j = 0; j < batch; j++) {
	            for (dim_t k = 0; k < feature_size; k++) {
	                size_t tnc_offset
	                        = i * batch * feature_size + j * feature_size + k;
	                alignment_model_ptr[tnc_offset]
	                        = tanhf((float)(weighted_src_layer[j * feature_size + k]
	                                        - dec_src_layer_shift * compensation[k])
	                                        / (dec_src_layer_scale
	                                                * weights_src_layer_scale)
	                                + weighted_annotations[tnc_offset]);
	            }
	        }
	    }
	
	    // gemv with alignments weights. the resulting alignments are in alignments
	    dim_t num_weighted_annotations = src_seq_length_max * batch;
	    :ref:`dnnl_sgemm <doxid-group__dnnl__api__blas_1ga75ee119765bdac249200fda42c0617f8>`('N', 'N', num_weighted_annotations, 1, feature_size, 1.f,
	            alignment_model_ptr, feature_size, weights_alignments, 1, 0.f,
	            alignments.data(), 1);
	
	    // softmax on alignments. the resulting context weights are in alignments
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(1)
	    for (dim_t i = 0; i < batch; i++)
	        exp_sums[i] = 0.0f;
	
	    // For each batch j, in the expression: exp(A_i) / \sum_i exp(A_i)
	    // we calculate max_idx t so that A_i <= A_t and calculate the expression as
	    //         exp(A_i - A_t) / \sum_i exp(A_i - A_t)
	    // which mitigates the overflow errors
	    :ref:`std <doxid-namespacestd>`::vector<dim_t> max_idx(batch, 0);
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(1)
	    for (dim_t j = 0; j < batch; j++) {
	        for (dim_t i = 1; i < src_seq_length_max; i++) {
	            if (alignments[i * batch + j] > alignments[(i - 1) * batch + j])
	                max_idx[j] = i;
	        }
	    }
	
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(1)
	    for (dim_t j = 0; j < batch; j++) {
	        auto max_idx_val = alignments[max_idx[j] * batch + j];
	        for (dim_t i = 0; i < src_seq_length_max; i++) {
	            alignments[i * batch + j] -= max_idx_val;
	            alignments[i * batch + j] = expf(alignments[i * batch + j]);
	            exp_sums[j] += alignments[i * batch + j];
	        }
	    }
	
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
	    for (dim_t i = 0; i < src_seq_length_max; i++)
	        for (dim_t j = 0; j < batch; j++)
	            alignments[i * batch + j] /= exp_sums[j];
	
	    // then we compute the context vectors
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
	    for (dim_t i = 0; i < batch; i++)
	        for (dim_t j = 0; j < feature_size; j++)
	            context_vectors[i * (feature_size + feature_size) + feature_size
	                    + j]
	                    = 0.0f;
	
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
	    for (dim_t i = 0; i < batch; i++)
	        for (dim_t j = 0; j < feature_size; j++)
	            for (dim_t k = 0; k < src_seq_length_max; k++)
	                context_vectors[i * (feature_size + feature_size) + feature_size
	                        + j]
	                        += alignments[k * batch + i]
	                        * (((float)annotations[j
	                                    + feature_size * (i + batch * k)]
	                                   - dec_src_layer_shift)
	                                / dec_src_layer_scale);
	}
	
	void copy_context(
	        float *src_iter, dim_t n_layers, dim_t batch, dim_t feature_size) {
	    // we copy the context from the first layer to all other layers
	    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(3)
	    for (dim_t k = 1; k < n_layers; k++)
	        for (dim_t j = 0; j < batch; j++)
	            for (dim_t i = 0; i < feature_size; i++)
	                src_iter[(k * batch + j) * (feature_size + feature_size)
	                        + feature_size + i]
	                        = src_iter[j * (feature_size + feature_size)
	                                + feature_size + i];
	}
	
	void simple_net() {
	    //[Initialize engine and stream]
	    auto cpu_engine = :ref:`engine <doxid-structdnnl_1_1engine>`(:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`, 0);
	    :ref:`stream <doxid-structdnnl_1_1stream>` s(cpu_engine);
	    //[Initialize engine and stream]
	
	    //[declare net]
	    std::vector<primitive> encoder_net, decoder_net;
	    std::vector<std::unordered_map<int, memory>> encoder_net_args,
	            decoder_net_args;
	
	    std::vector<float> net_src(batch * src_seq_length_max * feature_size, 0.1f);
	    std::vector<float> net_dst(batch * tgt_seq_length_max * feature_size, 0.1f);
	    //[declare net]
	
	    // Quantization factors for f32 data
	
	    const float data_shift = 64.;
	    const float data_scale = 63.;
	    const int weights_scale_mask = 0
	            + (1 << 3) // bit, indicating the unique scales for `g` dim in `ldigo`
	            + (1 << 4); // bit, indicating the unique scales for `o` dim in `ldigo`
	    //[quantize]
	    std::vector<float> weights_scales(lstm_n_gates * feature_size);
	    // assign halves of vector with arbitrary values
	    const dim_t scales_half = lstm_n_gates * feature_size / 2;
	    std::fill(
	            weights_scales.begin(), weights_scales.begin() + scales_half, 30.f);
	    std::fill(
	            weights_scales.begin() + scales_half, weights_scales.end(), 65.5f);
	    //[quantize]
	
	    //[Initialize encoder memory]
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` enc_bidir_src_layer_tz
	            = {src_seq_length_max, batch, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` enc_bidir_weights_layer_tz
	            = {enc_bidir_n_layers, 2, feature_size, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` enc_bidir_weights_iter_tz
	            = {enc_bidir_n_layers, 2, feature_size, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` enc_bidir_bias_tz
	            = {enc_bidir_n_layers, 2, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` enc_bidir_dst_layer_tz
	            = {src_seq_length_max, batch, 2 * feature_size};
	
	    //[Initialize encoder memory]
	
	
	    std::vector<float> user_enc_bidir_wei_layer(
	            enc_bidir_n_layers * 2 * feature_size * lstm_n_gates * feature_size,
	            0.3f);
	    std::vector<float> user_enc_bidir_wei_iter(
	            enc_bidir_n_layers * 2 * feature_size * lstm_n_gates * feature_size,
	            0.2f);
	    std::vector<float> user_enc_bidir_bias(
	            enc_bidir_n_layers * 2 * lstm_n_gates * feature_size, 1.0f);
	
	    //[data memory creation]
	    auto user_enc_bidir_src_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_src_layer_tz},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::tnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac775cf954921a129a65eb929476de911>`);
	
	    auto user_enc_bidir_wei_layer_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_weights_layer_tz}, :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	                    :ref:`memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	
	    auto user_enc_bidir_wei_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_weights_iter_tz},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	
	    auto user_enc_bidir_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_bias_tz},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldgo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8690cd92ccee6a0ad55faccc0346aab>`);
	
	    auto user_enc_bidir_src_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(user_enc_bidir_src_layer_md, cpu_engine, net_src.data());
	    auto user_enc_bidir_wei_layer_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(user_enc_bidir_wei_layer_md,
	            cpu_engine, user_enc_bidir_wei_layer.data());
	    auto user_enc_bidir_wei_iter_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(user_enc_bidir_wei_iter_md,
	            cpu_engine, user_enc_bidir_wei_iter.data());
	    auto user_enc_bidir_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(
	            user_enc_bidir_bias_md, cpu_engine, user_enc_bidir_bias.data());
	    //[data memory creation]
	
	    //[memory desc for RNN data]
	    auto enc_bidir_src_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_src_layer_tz},
	            :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    auto enc_bidir_wei_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_weights_layer_tz},
	            :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    auto enc_bidir_wei_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_weights_iter_tz},
	            :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    auto enc_bidir_dst_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_dst_layer_tz},
	            :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    //[memory desc for RNN data]
	
	
	    //[RNN attri]
	    :ref:`primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	    attr.:ref:`set_rnn_data_qparams <doxid-structdnnl_1_1primitive__attr_1a39ce5aa8b06ed331d8e2158108cc8324>`(data_scale, data_shift);
	    attr.set_rnn_weights_qparams(weights_scale_mask, weights_scales);
	
	    // check if int8 LSTM is supported
	    :ref:`lstm_forward::primitive_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc>` enc_bidir_prim_desc;
	    try {
	        enc_bidir_prim_desc = :ref:`lstm_forward::primitive_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc>`(cpu_engine,
	                :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	                :ref:`rnn_direction::bidirectional_concat <doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a7a1bb9f8699e8c03cbe4bd681fb50830>`, enc_bidir_src_layer_md,
	                :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), enc_bidir_wei_layer_md,
	                enc_bidir_wei_iter_md, user_enc_bidir_bias_md,
	                enc_bidir_dst_layer_md, :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), attr);
	    } catch (:ref:`error <doxid-structdnnl_1_1error>` &e) {
	        if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`)
	            throw example_allows_unimplemented {
	                    "No int8 LSTM implementation is available for this "
	                    "platform.\n"
	                    "Please refer to the developer guide for details."};
	
	        // on any other error just re-throw
	        throw;
	    }
	
	    //[RNN attri]
	
	    //[reorder input data]
	    auto enc_bidir_src_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_bidir_prim_desc.:ref:`src_layer_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc_1afd262a03436e463c97bb5dbe4b54a89d>`(), cpu_engine);
	    auto enc_bidir_src_layer_reorder_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            user_enc_bidir_src_layer_memory, enc_bidir_src_layer_memory, attr);
	    encoder_net.push_back(:ref:`reorder <doxid-structdnnl_1_1reorder>`(enc_bidir_src_layer_reorder_pd));
	    encoder_net_args.push_back(
	            {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, user_enc_bidir_src_layer_memory},
	                    {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, enc_bidir_src_layer_memory}});
	    //[reorder input data]
	
	    auto enc_bidir_wei_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_bidir_prim_desc.:ref:`weights_layer_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc_1a832e7468c8062760a262a82fdf7b8976>`(), cpu_engine);
	    auto enc_bidir_wei_layer_reorder_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            user_enc_bidir_wei_layer_memory, enc_bidir_wei_layer_memory, attr);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(enc_bidir_wei_layer_reorder_pd)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_enc_bidir_wei_layer_memory,
	                    enc_bidir_wei_layer_memory);
	
	    auto enc_bidir_wei_iter_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_bidir_prim_desc.:ref:`weights_iter_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc_1a3b3aa227de71f38560588b535b19cee7>`(), cpu_engine);
	    auto enc_bidir_wei_iter_reorder_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            user_enc_bidir_wei_iter_memory, enc_bidir_wei_iter_memory, attr);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(enc_bidir_wei_iter_reorder_pd)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_enc_bidir_wei_iter_memory,
	                    enc_bidir_wei_iter_memory);
	
	    auto enc_bidir_dst_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_bidir_prim_desc.:ref:`dst_layer_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc_1a47da32a15db013f1d5859a55522aa2f5>`(), cpu_engine);
	
	    //[push bi rnn to encoder net]
	    encoder_net.push_back(:ref:`lstm_forward <doxid-structdnnl_1_1lstm__forward>`(enc_bidir_prim_desc));
	    encoder_net_args.push_back(
	            {{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, enc_bidir_src_layer_memory},
	                    {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, enc_bidir_wei_layer_memory},
	                    {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, enc_bidir_wei_iter_memory},
	                    {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_enc_bidir_bias_memory},
	                    {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, enc_bidir_dst_layer_memory}});
	    //[push bi rnn to encoder net]
	
	    //[first uni layer]
	    std::vector<float> user_enc_uni_first_wei_layer(
	            1 * 1 * 2 * feature_size * lstm_n_gates * feature_size, 0.3f);
	    std::vector<float> user_enc_uni_first_wei_iter(
	            1 * 1 * feature_size * lstm_n_gates * feature_size, 0.2f);
	    std::vector<float> user_enc_uni_first_bias(
	            1 * 1 * lstm_n_gates * feature_size, 1.0f);
	    //[first uni layer]
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_enc_uni_first_wei_layer_dims
	            = {1, 1, 2 * feature_size, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_enc_uni_first_wei_iter_dims
	            = {1, 1, feature_size, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_enc_uni_first_bias_dims
	            = {1, 1, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` enc_uni_first_dst_layer_dims
	            = {src_seq_length_max, batch, feature_size};
	
	    auto user_enc_uni_first_wei_layer_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_first_wei_layer_dims},
	                    :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	    auto user_enc_uni_first_wei_iter_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_first_wei_iter_dims},
	                    :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	    auto user_enc_uni_first_bias_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_first_bias_dims},
	                    :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldgo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8690cd92ccee6a0ad55faccc0346aab>`);
	    auto user_enc_uni_first_wei_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(user_enc_uni_first_wei_layer_md, cpu_engine,
	                    user_enc_uni_first_wei_layer.data());
	    auto user_enc_uni_first_wei_iter_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(user_enc_uni_first_wei_iter_md, cpu_engine,
	                    user_enc_uni_first_wei_iter.data());
	    auto user_enc_uni_first_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(user_enc_uni_first_bias_md,
	            cpu_engine, user_enc_uni_first_bias.data());
	
	    auto enc_uni_first_wei_layer_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_first_wei_layer_dims},
	                    :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto enc_uni_first_wei_iter_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_first_wei_iter_dims},
	                    :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto enc_uni_first_dst_layer_md
	            = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_uni_first_dst_layer_dims},
	                    :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    //[create uni first]
	
	    auto enc_uni_first_prim_desc = :ref:`lstm_forward::primitive_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc>`(cpu_engine,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	            :ref:`rnn_direction::unidirectional_left2right <doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a04f4bf4bc6a47e30f0353597e244c44a>`, enc_bidir_dst_layer_md,
	            :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), enc_uni_first_wei_layer_md,
	            enc_uni_first_wei_iter_md, user_enc_uni_first_bias_md,
	            enc_uni_first_dst_layer_md, :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), attr);
	
	    //[create uni first]
	
	    auto enc_uni_first_wei_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_uni_first_prim_desc.weights_layer_desc(), cpu_engine);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_enc_uni_first_wei_layer_memory, enc_uni_first_wei_layer_memory)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_enc_uni_first_wei_layer_memory,
	                    enc_uni_first_wei_layer_memory);
	
	    auto enc_uni_first_wei_iter_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_uni_first_prim_desc.weights_iter_desc(), cpu_engine);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(user_enc_uni_first_wei_iter_memory, enc_uni_first_wei_iter_memory)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_enc_uni_first_wei_iter_memory,
	                    enc_uni_first_wei_iter_memory);
	
	    auto enc_uni_first_dst_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_uni_first_prim_desc.dst_layer_desc(), cpu_engine);
	
	    //[push first uni rnn to encoder net]
	    encoder_net.push_back(:ref:`lstm_forward <doxid-structdnnl_1_1lstm__forward>`(enc_uni_first_prim_desc));
	    encoder_net_args.push_back(
	            {{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, enc_bidir_dst_layer_memory},
	                    {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, enc_uni_first_wei_layer_memory},
	                    {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, enc_uni_first_wei_iter_memory},
	                    {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_enc_uni_first_bias_memory},
	                    {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, enc_uni_first_dst_layer_memory}});
	    //[push first uni rnn to encoder net]
	
	    //[remaining uni layers]
	    std::vector<float> user_enc_uni_wei_layer((enc_unidir_n_layers - 1) * 1
	                    * feature_size * lstm_n_gates * feature_size,
	            0.3f);
	    std::vector<float> user_enc_uni_wei_iter((enc_unidir_n_layers - 1) * 1
	                    * feature_size * lstm_n_gates * feature_size,
	            0.2f);
	    std::vector<float> user_enc_uni_bias(
	            (enc_unidir_n_layers - 1) * 1 * lstm_n_gates * feature_size, 1.0f);
	    //[remaining uni layers]
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_enc_uni_wei_layer_dims = {(enc_unidir_n_layers - 1), 1,
	            feature_size, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_enc_uni_wei_iter_dims = {(enc_unidir_n_layers - 1), 1,
	            feature_size, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_enc_uni_bias_dims
	            = {(enc_unidir_n_layers - 1), 1, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` enc_dst_layer_dims = {src_seq_length_max, batch, feature_size};
	
	    auto user_enc_uni_wei_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_wei_layer_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	    auto user_enc_uni_wei_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_wei_iter_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	    auto user_enc_uni_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_bias_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldgo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8690cd92ccee6a0ad55faccc0346aab>`);
	
	    auto user_enc_uni_wei_layer_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(user_enc_uni_wei_layer_md,
	            cpu_engine, user_enc_uni_wei_layer.data());
	    auto user_enc_uni_wei_iter_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(
	            user_enc_uni_wei_iter_md, cpu_engine, user_enc_uni_wei_iter.data());
	    auto user_enc_uni_bias_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(
	            user_enc_uni_bias_md, cpu_engine, user_enc_uni_bias.data());
	
	    auto enc_uni_wei_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_wei_layer_dims},
	            :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto enc_uni_wei_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_enc_uni_wei_iter_dims},
	            :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto enc_dst_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_dst_layer_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    //[create uni rnn]
	
	    auto enc_uni_prim_desc = :ref:`lstm_forward::primitive_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc>`(cpu_engine,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	            :ref:`rnn_direction::unidirectional_left2right <doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a04f4bf4bc6a47e30f0353597e244c44a>`,
	            enc_uni_first_dst_layer_md, :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(),
	            enc_uni_wei_layer_md, enc_uni_wei_iter_md, user_enc_uni_bias_md,
	            enc_dst_layer_md, :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(), attr);
	    //[create uni rnn]
	
	    auto enc_uni_wei_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_uni_prim_desc.weights_layer_desc(), cpu_engine);
	    auto enc_uni_wei_layer_reorder_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            user_enc_uni_wei_layer_memory, enc_uni_wei_layer_memory, attr);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(enc_uni_wei_layer_reorder_pd)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(
	                    s, user_enc_uni_wei_layer_memory, enc_uni_wei_layer_memory);
	
	    auto enc_uni_wei_iter_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_uni_prim_desc.weights_iter_desc(), cpu_engine);
	    auto enc_uni_wei_iter_reorder_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            user_enc_uni_wei_iter_memory, enc_uni_wei_iter_memory, attr);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(enc_uni_wei_iter_reorder_pd)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_enc_uni_wei_iter_memory, enc_uni_wei_iter_memory);
	
	    auto enc_dst_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(enc_uni_prim_desc.dst_layer_desc(), cpu_engine);
	
	    //[push uni rnn to encoder net]
	    encoder_net.push_back(:ref:`lstm_forward <doxid-structdnnl_1_1lstm__forward>`(enc_uni_prim_desc));
	    encoder_net_args.push_back(
	            {{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, enc_uni_first_dst_layer_memory},
	                    {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, enc_uni_wei_layer_memory},
	                    {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, enc_uni_wei_iter_memory},
	                    {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_enc_uni_bias_memory},
	                    {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, enc_dst_layer_memory}});
	    //[push uni rnn to encoder net]
	
	    //[dec mem dim]
	    std::vector<float> user_dec_wei_layer(
	            dec_n_layers * 1 * feature_size * lstm_n_gates * feature_size,
	            0.2f);
	    std::vector<float> user_dec_wei_iter(dec_n_layers * 1
	                    * (feature_size + feature_size) * lstm_n_gates
	                    * feature_size,
	            0.3f);
	    std::vector<float> user_dec_bias(
	            dec_n_layers * 1 * lstm_n_gates * feature_size, 1.0f);
	    std::vector<int8_t> user_weights_attention_src_layer(
	            feature_size * feature_size, 1);
	    float weights_attention_scale = 127.;
	    std::vector<float> user_weights_annotation(
	            feature_size * feature_size, 1.0f);
	    std::vector<float> user_weights_alignments(feature_size, 1.0f);
	    // Buffer to store decoder output for all iterations
	    std::vector<uint8_t> dec_dst(tgt_seq_length_max * batch * feature_size, 0);
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_dec_wei_layer_dims
	            = {dec_n_layers, 1, feature_size, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_dec_wei_iter_dims = {dec_n_layers, 1,
	            feature_size + feature_size, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` user_dec_bias_dims
	            = {dec_n_layers, 1, lstm_n_gates, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dec_src_layer_dims = {1, batch, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dec_dst_layer_dims = {1, batch, feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dec_dst_iter_c_dims = {dec_n_layers, 1, batch, feature_size};
	    //[dec mem dim]
	
	    // We will use the same memory for dec_src_iter and dec_dst_iter
	    // However, dec_src_iter has a context vector but not
	    // dec_dst_iter.
	    // To resolve this we will create one memory that holds the
	    // context vector as well as the both the hidden and cell states.
	    // For the dst_iter, we will use a view on this memory.
	    // Note that the cell state will be padded by
	    // feature_size values. However, we do not compute or
	    // access those.
	    //[noctx mem dim]
	    std::vector<float> dec_dst_iter(
	            dec_n_layers * batch * 2 * feature_size, 1.0f);
	
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dec_dst_iter_dims
	            = {dec_n_layers, 1, batch, feature_size + feature_size};
	    :ref:`memory::dims <doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` dec_dst_iter_noctx_dims
	            = {dec_n_layers, 1, batch, feature_size};
	    //[noctx mem dim]
	
	    //[dec mem desc]
	    auto user_dec_wei_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_dec_wei_layer_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	    auto user_dec_wei_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_dec_wei_iter_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	    auto user_dec_bias_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_dec_bias_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldgo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8690cd92ccee6a0ad55faccc0346aab>`);
	    auto dec_src_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({dec_src_layer_dims},
	            :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, :ref:`memory::format_tag::tnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac775cf954921a129a65eb929476de911>`);
	    auto dec_dst_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({dec_dst_layer_dims},
	            :ref:`memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`, :ref:`memory::format_tag::tnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac775cf954921a129a65eb929476de911>`);
	    auto dec_dst_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({dec_dst_iter_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab49be97ff353a86d84d06d98f846b61d>`);
	    auto dec_dst_iter_c_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({dec_dst_iter_c_dims},
	            :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`memory::format_tag::ldnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab49be97ff353a86d84d06d98f846b61d>`);
	    //[dec mem desc]
	
	    //[create dec memory]
	    auto user_dec_wei_layer_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(
	            user_dec_wei_layer_md, cpu_engine, user_dec_wei_layer.data());
	    auto user_dec_wei_iter_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(
	            user_dec_wei_iter_md, cpu_engine, user_dec_wei_iter.data());
	    auto user_dec_bias_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(user_dec_bias_md, cpu_engine, user_dec_bias.data());
	    auto dec_src_layer_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(dec_src_layer_md, cpu_engine);
	    auto dec_dst_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(dec_dst_layer_md, cpu_engine, dec_dst.data());
	    auto dec_dst_iter_c_memory = :ref:`memory <doxid-structdnnl_1_1memory>`(dec_dst_iter_c_md, cpu_engine);
	    //[create dec memory]
	
	    // Create memory descriptors for RNN data w/o specified layout
	    auto dec_wei_layer_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_dec_wei_layer_dims},
	            :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	    auto dec_wei_iter_md = :ref:`memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_dec_wei_iter_dims},
	            :ref:`memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`, :ref:`memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`);
	
	    //[create noctx mem]
	    auto dec_dst_iter_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(dec_dst_iter_md, cpu_engine, dec_dst_iter.data());
	    auto dec_dst_iter_noctx_md = dec_dst_iter_md.:ref:`submemory_desc <doxid-structdnnl_1_1memory_1_1desc_1a7de2abef3b34e94c5dfa16e1fc3f3aab>`(
	            dec_dst_iter_noctx_dims, {0, 0, 0, 0, 0});
	    //[create noctx mem]
	
	    auto dec_ctx_prim_desc = :ref:`lstm_forward::primitive_desc <doxid-structdnnl_1_1lstm__forward_1_1primitive__desc>`(cpu_engine,
	            :ref:`prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	            :ref:`rnn_direction::unidirectional_left2right <doxid-group__dnnl__api__rnn_1gga33315cf335d1cbe26fd6b70d956e23d5a04f4bf4bc6a47e30f0353597e244c44a>`, dec_src_layer_md,
	            dec_dst_iter_md, dec_dst_iter_c_md, dec_wei_layer_md,
	            dec_wei_iter_md, user_dec_bias_md, dec_dst_layer_md,
	            dec_dst_iter_noctx_md, dec_dst_iter_c_md, attr);
	
	    //[dec reorder]
	    auto dec_wei_layer_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(dec_ctx_prim_desc.weights_layer_desc(), cpu_engine);
	    auto dec_wei_layer_reorder_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            user_dec_wei_layer_memory, dec_wei_layer_memory, attr);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(dec_wei_layer_reorder_pd)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_dec_wei_layer_memory, dec_wei_layer_memory);
	    //[dec reorder]
	
	    auto dec_wei_iter_memory
	            = :ref:`memory <doxid-structdnnl_1_1memory>`(dec_ctx_prim_desc.weights_iter_desc(), cpu_engine);
	    auto dec_wei_iter_reorder_pd = :ref:`reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	            user_dec_wei_iter_memory, dec_wei_iter_memory, attr);
	    :ref:`reorder <doxid-structdnnl_1_1reorder>`(dec_wei_iter_reorder_pd)
	            .:ref:`execute <doxid-structdnnl_1_1reorder_1ab9d5265274a13d4afa1fe33d784a1027>`(s, user_dec_wei_iter_memory, dec_wei_iter_memory);
	
	    decoder_net.push_back(:ref:`lstm_forward <doxid-structdnnl_1_1lstm__forward>`(dec_ctx_prim_desc));
	    decoder_net_args.push_back({{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, dec_src_layer_memory},
	            {:ref:`DNNL_ARG_SRC_ITER <doxid-group__dnnl__api__primitives__common_1gaf35f4f604284f1b00bb35bffd0f7a143>`, dec_dst_iter_memory},
	            {:ref:`DNNL_ARG_SRC_ITER_C <doxid-group__dnnl__api__primitives__common_1ga8ef6969516e717208a33766542410410>`, dec_dst_iter_c_memory},
	            {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, dec_wei_layer_memory},
	            {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, dec_wei_iter_memory},
	            {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_dec_bias_memory},
	            {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, dec_dst_layer_memory},
	            {:ref:`DNNL_ARG_DST_ITER <doxid-group__dnnl__api__primitives__common_1ga13b91cbd3f531d9c90227895a275d5a6>`, dec_dst_iter_memory},
	            {:ref:`DNNL_ARG_DST_ITER_C <doxid-group__dnnl__api__primitives__common_1ga8b77d8716fc0ab9923d6cb409dbdf900>`, dec_dst_iter_c_memory}});
	
	    // Allocating temporary buffers for attention mechanism
	    std::vector<float> weighted_annotations(
	            src_seq_length_max * batch * feature_size, 1.0f);
	    std::vector<int32_t> weights_attention_sum_rows(feature_size, 1);
	
	
	    auto :ref:`execute <doxid-namespacednnl_1_1graph_1_1sycl__interop_1acc5ff56ff0f276367b047c3c73093a67>` = [&]() {
	        assert(encoder_net.size() == encoder_net_args.size()
	                && "something is missing");
	        //[run enc]
	        for (size_t p = 0; p < encoder_net.size(); ++p)
	            encoder_net.at(p).execute(s, encoder_net_args.at(p));
	        //[run enc]
	
	        // compute the weighted annotations once before the decoder
	        //[weight ano]
	        compute_weighted_annotations(weighted_annotations.data(),
	                src_seq_length_max, batch, feature_size,
	                user_weights_annotation.data(),
	                (float *)enc_dst_layer_memory.get_data_handle());
	        //[weight ano]
	        //[s8u8s32]
	        compute_sum_of_rows(user_weights_attention_src_layer.data(),
	                feature_size, feature_size, weights_attention_sum_rows.data());
	        //[s8u8s32]
	
	        //[init src_layer]
	        memset(dec_src_layer_memory.:ref:`get_data_handle <doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>`(), 0,
	                dec_src_layer_memory.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_size <doxid-structdnnl_1_1memory_1_1desc_1abfa095ac138d4d2ef8efd3739e343f08>`());
	        //[init src_layer]
	
	        for (dim_t i = 0; i < tgt_seq_length_max; i++) {
	            uint8_t *src_att_layer_handle
	                    = (uint8_t *)dec_src_layer_memory.:ref:`get_data_handle <doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>`();
	            float *src_att_iter_handle
	                    = (float *)dec_dst_iter_memory.get_data_handle();
	
	            //[att ctx]
	            compute_attention(src_att_iter_handle, src_seq_length_max, batch,
	                    feature_size, user_weights_attention_src_layer.data(),
	                    weights_attention_scale, weights_attention_sum_rows.data(),
	                    src_att_layer_handle, data_scale, data_shift,
	                    (uint8_t *)enc_bidir_dst_layer_memory.get_data_handle(),
	                    weighted_annotations.data(),
	                    user_weights_alignments.data());
	            //[att ctx]
	
	            //[cp ctx]
	            copy_context(
	                    src_att_iter_handle, dec_n_layers, batch, feature_size);
	            //[cp ctx]
	
	            assert(decoder_net.size() == decoder_net_args.size()
	                    && "something is missing");
	            //[run dec iter]
	            for (size_t p = 0; p < decoder_net.size(); ++p)
	                decoder_net.at(p).execute(s, decoder_net_args.at(p));
	            //[run dec iter]
	
	            //[set handle]
	            auto dst_layer_handle
	                    = (uint8_t *)dec_dst_layer_memory.get_data_handle();
	            dec_src_layer_memory.:ref:`set_data_handle <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>`(dst_layer_handle);
	            dec_dst_layer_memory.set_data_handle(
	                    dst_layer_handle + batch * feature_size);
	            //[set handle]
	        }
	    };
	
	    std::cout << "Parameters:" << std::endl
	              << " batch = " << batch << std::endl
	              << " feature size = " << feature_size << std::endl
	              << " maximum source sequence length = " << src_seq_length_max
	              << std::endl
	              << " maximum target sequence length = " << tgt_seq_length_max
	              << std::endl
	              << " number of layers of the bidirectional encoder = "
	              << enc_bidir_n_layers << std::endl
	              << " number of layers of the unidirectional encoder = "
	              << enc_unidir_n_layers << std::endl
	              << " number of layers of the decoder = " << dec_n_layers
	              << std::endl;
	
	    :ref:`execute <doxid-namespacednnl_1_1graph_1_1sycl__interop_1acc5ff56ff0f276367b047c3c73093a67>`();
	    s.wait();
	}
	
	int main(int argc, char **argv) {
	    return handle_example_errors({:ref:`engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`}, simple_net);
	}
