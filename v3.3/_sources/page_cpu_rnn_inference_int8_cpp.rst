.. index:: pair: page; RNN int8 inference example
.. _doxid-cpu_rnn_inference_int8_cpp:

RNN int8 inference example
==========================

This C++ API example demonstrates how to build GNMT model inference.

This C++ API example demonstrates how to build GNMT model inference.

Example code: :ref:`cpu_rnn_inference_int8.cpp <doxid-cpu_rnn_inference_int8_8cpp-example>`

For the encoder we use:

* one primitive for the bidirectional layer of the encoder

* one primitive for all remaining unidirectional layers in the encoder For the decoder we use:

* one primitive for the first iteration

* one primitive for all subsequent iterations in the decoder. Note that in this example, this primitive computes the states in place.

* the attention mechanism is implemented separately as there is no support for the context vectors in oneDNN yet

Initialize a CPU engine and stream. The last parameter in the call represents the index of the engine.

.. ref-code-block:: cpp

	auto cpu_engine = :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`(engine::kind::cpu, 0);
	stream s(cpu_engine);

Declare encoder net and decoder net

.. ref-code-block:: cpp

	std::vector<primitive> encoder_net, decoder_net;
	std::vector<std::unordered_map<int, memory>> encoder_net_args,
	        decoder_net_args;

	std::vector<float> net_src(batch * src_seq_length_max * feature_size, 0.1f);
	std::vector<float> net_dst(batch * tgt_seq_length_max * feature_size, 0.1f);

Quantization factors for f32 data

.. ref-code-block:: cpp

	std::vector<float> weights_scales(lstm_n_gates * feature_size);
	// assign halves of vector with arbitrary values
	const dim_t scales_half = lstm_n_gates * feature_size / 2;
	std::fill(
	        weights_scales.begin(), weights_scales.begin() + scales_half, 30.f);
	std::fill(
	        weights_scales.begin() + scales_half, weights_scales.end(), 65.5f);

Encoder

Initialize Encoder Memory

.. ref-code-block:: cpp

	memory::dims enc_bidir_src_layer_tz
	        = {src_seq_length_max, batch, feature_size};
	memory::dims enc_bidir_weights_layer_tz
	        = {enc_bidir_n_layers, 2, feature_size, lstm_n_gates, feature_size};
	memory::dims enc_bidir_weights_iter_tz
	        = {enc_bidir_n_layers, 2, feature_size, lstm_n_gates, feature_size};
	memory::dims enc_bidir_bias_tz
	        = {enc_bidir_n_layers, 2, lstm_n_gates, feature_size};
	memory::dims enc_bidir_dst_layer_tz
	        = {src_seq_length_max, batch, 2 * feature_size};

Encoder: 1 bidirectional layer and 7 unidirectional layers

Create the memory for user data

.. ref-code-block:: cpp

	auto user_enc_bidir_src_layer_md = memory::desc({enc_bidir_src_layer_tz},
	        memory::data_type::f32, memory::format_tag::tnc);

	auto user_enc_bidir_wei_layer_md
	        = memory::desc({enc_bidir_weights_layer_tz}, memory::data_type::f32,
	                memory::format_tag::ldigo);

	auto user_enc_bidir_wei_iter_md = memory::desc({enc_bidir_weights_iter_tz},
	        memory::data_type::f32, memory::format_tag::ldigo);

	auto user_enc_bidir_bias_md = memory::desc({enc_bidir_bias_tz},
	        memory::data_type::f32, memory::format_tag::ldgo);

	auto user_enc_bidir_src_layer_memory
	        = memory(user_enc_bidir_src_layer_md, cpu_engine, net_src.data());
	auto user_enc_bidir_wei_layer_memory = memory(user_enc_bidir_wei_layer_md,
	        cpu_engine, user_enc_bidir_wei_layer.data());
	auto user_enc_bidir_wei_iter_memory = memory(user_enc_bidir_wei_iter_md,
	        cpu_engine, user_enc_bidir_wei_iter.data());
	auto user_enc_bidir_bias_memory = memory(
	        user_enc_bidir_bias_md, cpu_engine, user_enc_bidir_bias.data());

Create memory descriptors for RNN data w/o specified layout

.. ref-code-block:: cpp

	auto enc_bidir_src_layer_md = memory::desc({enc_bidir_src_layer_tz},
	        memory::data_type::u8, memory::format_tag::any);

	auto enc_bidir_wei_layer_md = memory::desc({enc_bidir_weights_layer_tz},
	        memory::data_type::s8, memory::format_tag::any);

	auto enc_bidir_wei_iter_md = memory::desc({enc_bidir_weights_iter_tz},
	        memory::data_type::s8, memory::format_tag::any);

	auto enc_bidir_dst_layer_md = memory::desc({enc_bidir_dst_layer_tz},
	        memory::data_type::u8, memory::format_tag::any);

Create bidirectional RNN

Define RNN attributes that store quantization parameters

.. ref-code-block:: cpp

	primitive_attr attr;
	attr.set_rnn_data_qparams(data_scale, data_shift);
	attr.set_rnn_weights_qparams(weights_scale_mask, weights_scales);

	// check if int8 LSTM is supported
	lstm_forward::primitive_desc enc_bidir_prim_desc;
	try {
	    enc_bidir_prim_desc = lstm_forward::primitive_desc(cpu_engine,
	            prop_kind::forward_inference,
	            rnn_direction::bidirectional_concat, enc_bidir_src_layer_md,
	            memory::desc(), memory::desc(), enc_bidir_wei_layer_md,
	            enc_bidir_wei_iter_md, user_enc_bidir_bias_md,
	            enc_bidir_dst_layer_md, memory::desc(), memory::desc(), attr);
	} catch (error &e) {
	    if (e.status == :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`)
	        throw example_allows_unimplemented {
	                "No int8 LSTM implementation is available for this "
	                "platform.\n"
	                "Please refer to the developer guide for details."};

	    // on any other error just re-throw
	    throw;
	}

Create memory for input data and use reorders to quantize values to int8 NOTE: same attributes are used when creating RNN primitive and reorders

.. ref-code-block:: cpp

	auto enc_bidir_src_layer_memory
	        = memory(enc_bidir_prim_desc.src_layer_desc(), cpu_engine);
	auto enc_bidir_src_layer_reorder_pd = reorder::primitive_desc(
	        user_enc_bidir_src_layer_memory, enc_bidir_src_layer_memory, attr);
	encoder_net.push_back(reorder(enc_bidir_src_layer_reorder_pd));
	encoder_net_args.push_back(
	        {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, user_enc_bidir_src_layer_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, enc_bidir_src_layer_memory}});

Encoder : add the bidirectional rnn primitive with related arguments into encoder_net

.. ref-code-block:: cpp

	encoder_net.push_back(lstm_forward(enc_bidir_prim_desc));
	encoder_net_args.push_back(
	        {{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, enc_bidir_src_layer_memory},
	                {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, enc_bidir_wei_layer_memory},
	                {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, enc_bidir_wei_iter_memory},
	                {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_enc_bidir_bias_memory},
	                {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, enc_bidir_dst_layer_memory}});

Encoder: unidirectional layers

First unidirectinal layer scales 2 \* feature_size output of bidirectional layer to feature_size output

.. ref-code-block:: cpp

	std::vector<float> user_enc_uni_first_wei_layer(
	        1 * 1 * 2 * feature_size * lstm_n_gates * feature_size, 0.3f);
	std::vector<float> user_enc_uni_first_wei_iter(
	        1 * 1 * feature_size * lstm_n_gates * feature_size, 0.2f);
	std::vector<float> user_enc_uni_first_bias(
	        1 * 1 * lstm_n_gates * feature_size, 1.0f);

Encoder : Create unidirection RNN for first cell

.. ref-code-block:: cpp


	auto enc_uni_first_prim_desc = lstm_forward::primitive_desc(cpu_engine,
	        prop_kind::forward_inference,
	        rnn_direction::unidirectional_left2right, enc_bidir_dst_layer_md,
	        memory::desc(), memory::desc(), enc_uni_first_wei_layer_md,
	        enc_uni_first_wei_iter_md, user_enc_uni_first_bias_md,
	        enc_uni_first_dst_layer_md, memory::desc(), memory::desc(), attr);

Encoder : add the first unidirectional rnn primitive with related arguments into encoder_net

.. ref-code-block:: cpp

	encoder_net.push_back(lstm_forward(enc_uni_first_prim_desc));
	encoder_net_args.push_back(
	        {{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, enc_bidir_dst_layer_memory},
	                {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, enc_uni_first_wei_layer_memory},
	                {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, enc_uni_first_wei_iter_memory},
	                {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_enc_uni_first_bias_memory},
	                {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, enc_uni_first_dst_layer_memory}});

Encoder : Remaining unidirectional layers

.. ref-code-block:: cpp

	std::vector<float> user_enc_uni_wei_layer((enc_unidir_n_layers - 1) * 1
	                * feature_size * lstm_n_gates * feature_size,
	        0.3f);
	std::vector<float> user_enc_uni_wei_iter((enc_unidir_n_layers - 1) * 1
	                * feature_size * lstm_n_gates * feature_size,
	        0.2f);
	std::vector<float> user_enc_uni_bias(
	        (enc_unidir_n_layers - 1) * 1 * lstm_n_gates * feature_size, 1.0f);

Encoder : Create unidirection RNN cell

.. ref-code-block:: cpp


	auto enc_uni_prim_desc = lstm_forward::primitive_desc(cpu_engine,
	        prop_kind::forward_inference,
	        rnn_direction::unidirectional_left2right,
	        enc_uni_first_dst_layer_md, memory::desc(), memory::desc(),
	        enc_uni_wei_layer_md, enc_uni_wei_iter_md, user_enc_uni_bias_md,
	        enc_dst_layer_md, memory::desc(), memory::desc(), attr);

Encoder : add the unidirectional rnn primitive with related arguments into encoder_net

.. ref-code-block:: cpp

	encoder_net.push_back(lstm_forward(enc_uni_prim_desc));
	encoder_net_args.push_back(
	        {{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, enc_uni_first_dst_layer_memory},
	                {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, enc_uni_wei_layer_memory},
	                {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, enc_uni_wei_iter_memory},
	                {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_enc_uni_bias_memory},
	                {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, enc_dst_layer_memory}});

Decoder with attention mechanism

Decoder : declare memory dimensions

.. ref-code-block:: cpp

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

	memory::dims user_dec_wei_layer_dims
	        = {dec_n_layers, 1, feature_size, lstm_n_gates, feature_size};
	memory::dims user_dec_wei_iter_dims = {dec_n_layers, 1,
	        feature_size + feature_size, lstm_n_gates, feature_size};
	memory::dims user_dec_bias_dims
	        = {dec_n_layers, 1, lstm_n_gates, feature_size};
	memory::dims dec_src_layer_dims = {1, batch, feature_size};
	memory::dims dec_dst_layer_dims = {1, batch, feature_size};
	memory::dims dec_dst_iter_c_dims = {dec_n_layers, 1, batch, feature_size};

.. ref-code-block:: cpp

	std::vector<float> dec_dst_iter(
	        dec_n_layers * batch * 2 * feature_size, 1.0f);

	memory::dims dec_dst_iter_dims
	        = {dec_n_layers, 1, batch, feature_size + feature_size};
	memory::dims dec_dst_iter_noctx_dims
	        = {dec_n_layers, 1, batch, feature_size};

Decoder : create memory description Create memory descriptors for RNN data w/o specified layout

.. ref-code-block:: cpp

	auto user_dec_wei_layer_md = memory::desc({user_dec_wei_layer_dims},
	        memory::data_type::f32, memory::format_tag::ldigo);
	auto user_dec_wei_iter_md = memory::desc({user_dec_wei_iter_dims},
	        memory::data_type::f32, memory::format_tag::ldigo);
	auto user_dec_bias_md = memory::desc({user_dec_bias_dims},
	        memory::data_type::f32, memory::format_tag::ldgo);
	auto dec_src_layer_md = memory::desc({dec_src_layer_dims},
	        memory::data_type::u8, memory::format_tag::tnc);
	auto dec_dst_layer_md = memory::desc({dec_dst_layer_dims},
	        memory::data_type::u8, memory::format_tag::tnc);
	auto dec_dst_iter_md = memory::desc({dec_dst_iter_dims},
	        memory::data_type::f32, memory::format_tag::ldnc);
	auto dec_dst_iter_c_md = memory::desc({dec_dst_iter_c_dims},
	        memory::data_type::f32, memory::format_tag::ldnc);

Decoder : Create memory

.. ref-code-block:: cpp

	auto user_dec_wei_layer_memory = memory(
	        user_dec_wei_layer_md, cpu_engine, user_dec_wei_layer.data());
	auto user_dec_wei_iter_memory = memory(
	        user_dec_wei_iter_md, cpu_engine, user_dec_wei_iter.data());
	auto user_dec_bias_memory
	        = memory(user_dec_bias_md, cpu_engine, user_dec_bias.data());
	auto dec_src_layer_memory = memory(dec_src_layer_md, cpu_engine);
	auto dec_dst_layer_memory
	        = memory(dec_dst_layer_md, cpu_engine, dec_dst.data());
	auto dec_dst_iter_c_memory = memory(dec_dst_iter_c_md, cpu_engine);

Decoder : As mentioned above, we create a view without context out of the memory with context.

.. ref-code-block:: cpp

	auto dec_dst_iter_memory
	        = memory(dec_dst_iter_md, cpu_engine, dec_dst_iter.data());
	auto dec_dst_iter_noctx_md = dec_dst_iter_md.submemory_desc(
	        dec_dst_iter_noctx_dims, {0, 0, 0, 0, 0});

Decoder : Create memory for input data and use reorders to quantize values to int8

.. ref-code-block:: cpp

	auto dec_wei_layer_memory
	        = memory(dec_ctx_prim_desc.weights_layer_desc(), cpu_engine);
	auto dec_wei_layer_reorder_pd = reorder::primitive_desc(
	        user_dec_wei_layer_memory, dec_wei_layer_memory, attr);
	reorder(dec_wei_layer_reorder_pd)
	        .execute(s, user_dec_wei_layer_memory, dec_wei_layer_memory);

Execution

run encoder (1 stream)

.. ref-code-block:: cpp

	for (size_t p = 0; p < encoder_net.size(); ++p)
	    encoder_net.at(p).execute(s, encoder_net_args.at(p));

we compute the weighted annotations once before the decoder

.. ref-code-block:: cpp

	compute_weighted_annotations(weighted_annotations.data(),
	        src_seq_length_max, batch, feature_size,
	        user_weights_annotation.data(),
	        (float *)enc_dst_layer_memory.get_data_handle());

precompute compensation for s8u8s32 gemm in compute attention

.. ref-code-block:: cpp

	compute_sum_of_rows(user_weights_attention_src_layer.data(),
	        feature_size, feature_size, weights_attention_sum_rows.data());

We initialize src_layer to the embedding of the end of sequence character, which are assumed to be 0 here

.. ref-code-block:: cpp

	memset(dec_src_layer_memory.get_data_handle(), 0,
	        dec_src_layer_memory.get_desc().get_size());

From now on, src points to the output of the last iteration

Compute attention context vector into the first layer src_iter

.. ref-code-block:: cpp

	compute_attention(src_att_iter_handle, src_seq_length_max, batch,
	        feature_size, user_weights_attention_src_layer.data(),
	        weights_attention_scale, weights_attention_sum_rows.data(),
	        src_att_layer_handle, data_scale, data_shift,
	        (uint8_t *)enc_bidir_dst_layer_memory.get_data_handle(),
	        weighted_annotations.data(),
	        user_weights_alignments.data());

copy the context vectors to all layers of src_iter

.. ref-code-block:: cpp

	copy_context(
	        src_att_iter_handle, dec_n_layers, batch, feature_size);

run the decoder iteration

.. ref-code-block:: cpp

	for (size_t p = 0; p < decoder_net.size(); ++p)
	    decoder_net.at(p).execute(s, decoder_net_args.at(p));

Move the handle on the src/dst layer to the next iteration

.. ref-code-block:: cpp

	auto dst_layer_handle
	        = (uint8_t *)dec_dst_layer_memory.get_data_handle();
	dec_src_layer_memory.set_data_handle(dst_layer_handle);
	dec_dst_layer_memory.set_data_handle(
	        dst_layer_handle + batch * feature_size);

