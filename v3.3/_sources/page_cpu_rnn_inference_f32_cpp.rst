.. index:: pair: page; RNN f32 inference example
.. _doxid-cpu_rnn_inference_f32_cpp:

RNN f32 inference example
=========================

This C++ API example demonstrates how to build GNMT model inference.

This C++ API example demonstrates how to build GNMT model inference.

Example code: :ref:`cpu_rnn_inference_f32.cpp <doxid-cpu_rnn_inference_f32_8cpp-example>`

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

	std::vector<float> net_src(batch * src_seq_length_max * feature_size, 1.0f);
	std::vector<float> net_dst(batch * tgt_seq_length_max * feature_size, 1.0f);

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

	auto user_enc_bidir_src_layer_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	        {enc_bidir_src_layer_tz}, :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	        :ref:`dnnl::memory::format_tag::tnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac775cf954921a129a65eb929476de911>`);

	auto user_enc_bidir_wei_layer_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	        {enc_bidir_weights_layer_tz}, :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	        :ref:`dnnl::memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);

	auto user_enc_bidir_wei_iter_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	        {enc_bidir_weights_iter_tz}, :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	        :ref:`dnnl::memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);

	auto user_enc_bidir_bias_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({enc_bidir_bias_tz},
	        :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`dnnl::memory::format_tag::ldgo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8690cd92ccee6a0ad55faccc0346aab>`);

	auto user_enc_bidir_src_layer_memory = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(
	        user_enc_bidir_src_layer_md, cpu_engine, net_src.data());
	auto user_enc_bidir_wei_layer_memory
	        = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(user_enc_bidir_wei_layer_md, cpu_engine,
	                user_enc_bidir_wei_layer.data());
	auto user_enc_bidir_wei_iter_memory
	        = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(user_enc_bidir_wei_iter_md, cpu_engine,
	                user_enc_bidir_wei_iter.data());
	auto user_enc_bidir_bias_memory = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(
	        user_enc_bidir_bias_md, cpu_engine, user_enc_bidir_bias.data());

Create memory descriptors for RNN data w/o specified layout

.. ref-code-block:: cpp

	auto enc_bidir_wei_layer_md = memory::desc({enc_bidir_weights_layer_tz},
	        memory::data_type::f32, memory::format_tag::any);

	auto enc_bidir_wei_iter_md = memory::desc({enc_bidir_weights_iter_tz},
	        memory::data_type::f32, memory::format_tag::any);

	auto enc_bidir_dst_layer_md = memory::desc({enc_bidir_dst_layer_tz},
	        memory::data_type::f32, memory::format_tag::any);

Create bidirectional RNN

.. ref-code-block:: cpp


	auto enc_bidir_prim_desc = lstm_forward::primitive_desc(cpu_engine,
	        prop_kind::forward_inference, rnn_direction::bidirectional_concat,
	        user_enc_bidir_src_layer_md, memory::desc(), memory::desc(),
	        enc_bidir_wei_layer_md, enc_bidir_wei_iter_md,
	        user_enc_bidir_bias_md, enc_bidir_dst_layer_md, memory::desc(),
	        memory::desc());

Create memory for input data and use reorders to reorder user data to internal representation

.. ref-code-block:: cpp

	auto enc_bidir_wei_layer_memory
	        = memory(enc_bidir_prim_desc.weights_layer_desc(), cpu_engine);
	auto enc_bidir_wei_layer_reorder_pd = reorder::primitive_desc(
	        user_enc_bidir_wei_layer_memory, enc_bidir_wei_layer_memory);
	reorder(enc_bidir_wei_layer_reorder_pd)
	        .execute(s, user_enc_bidir_wei_layer_memory,
	                enc_bidir_wei_layer_memory);

Encoder : add the bidirectional rnn primitive with related arguments into encoder_net

.. ref-code-block:: cpp

	encoder_net.push_back(lstm_forward(enc_bidir_prim_desc));
	encoder_net_args.push_back(
	        {{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, user_enc_bidir_src_layer_memory},
	                {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, enc_bidir_wei_layer_memory},
	                {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, enc_bidir_wei_iter_memory},
	                {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_enc_bidir_bias_memory},
	                {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, enc_bidir_dst_layer_memory}});

Encoder: unidirectional layers

First unidirectinal layer scales 2 \* feature_size output of bidirectional layer to feature_size output

.. ref-code-block:: cpp

	std::vector<float> user_enc_uni_first_wei_layer(
	        1 * 1 * 2 * feature_size * lstm_n_gates * feature_size, 1.0f);
	std::vector<float> user_enc_uni_first_wei_iter(
	        1 * 1 * feature_size * lstm_n_gates * feature_size, 1.0f);
	std::vector<float> user_enc_uni_first_bias(
	        1 * 1 * lstm_n_gates * feature_size, 1.0f);

Encoder : Create unidirection RNN for first cell

.. ref-code-block:: cpp

	auto enc_uni_first_prim_desc = lstm_forward::primitive_desc(cpu_engine,
	        prop_kind::forward_inference,
	        rnn_direction::unidirectional_left2right, enc_bidir_dst_layer_md,
	        memory::desc(), memory::desc(), enc_uni_first_wei_layer_md,
	        enc_uni_first_wei_iter_md, user_enc_uni_first_bias_md,
	        enc_uni_first_dst_layer_md, memory::desc(), memory::desc());

Encoder : add the first unidirectional rnn primitive with related arguments into encoder_net

.. ref-code-block:: cpp

	// TODO: add a reorder when they will be available
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
	        1.0f);
	std::vector<float> user_enc_uni_wei_iter((enc_unidir_n_layers - 1) * 1
	                * feature_size * lstm_n_gates * feature_size,
	        1.0f);
	std::vector<float> user_enc_uni_bias(
	        (enc_unidir_n_layers - 1) * 1 * lstm_n_gates * feature_size, 1.0f);

Encoder : Create unidirection RNN cell

.. ref-code-block:: cpp

	auto enc_uni_prim_desc = lstm_forward::primitive_desc(cpu_engine,
	        prop_kind::forward_inference,
	        rnn_direction::unidirectional_left2right,
	        enc_uni_first_dst_layer_md, memory::desc(), memory::desc(),
	        enc_uni_wei_layer_md, enc_uni_wei_iter_md, user_enc_uni_bias_md,
	        enc_dst_layer_md, memory::desc(), memory::desc());

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
	        1.0f);
	std::vector<float> user_dec_wei_iter(dec_n_layers * 1
	                * (feature_size + feature_size) * lstm_n_gates
	                * feature_size,
	        1.0f);
	std::vector<float> user_dec_bias(
	        dec_n_layers * 1 * lstm_n_gates * feature_size, 1.0f);
	std::vector<float> user_dec_dst(
	        tgt_seq_length_max * batch * feature_size, 1.0f);
	std::vector<float> user_weights_attention_src_layer(
	        feature_size * feature_size, 1.0f);
	std::vector<float> user_weights_annotation(
	        feature_size * feature_size, 1.0f);
	std::vector<float> user_weights_alignments(feature_size, 1.0f);

	memory::dims user_dec_wei_layer_dims
	        = {dec_n_layers, 1, feature_size, lstm_n_gates, feature_size};
	memory::dims user_dec_wei_iter_dims = {dec_n_layers, 1,
	        feature_size + feature_size, lstm_n_gates, feature_size};
	memory::dims user_dec_bias_dims
	        = {dec_n_layers, 1, lstm_n_gates, feature_size};

	memory::dims dec_src_layer_dims = {1, batch, feature_size};
	memory::dims dec_dst_layer_dims = {1, batch, feature_size};
	memory::dims dec_dst_iter_c_dims = {dec_n_layers, 1, batch, feature_size};

We will use the same memory for dec_src_iter and dec_dst_iter However, dec_src_iter has a context vector but not dec_dst_iter. To resolve this we will create one memory that holds the context vector as well as the both the hidden and cell states. The dst_iter will be a sub-memory of this memory. Note that the cell state will be padded by feature_size values. However, we do not compute or access those.

.. ref-code-block:: cpp

	memory::dims dec_dst_iter_dims
	        = {dec_n_layers, 1, batch, feature_size + feature_size};
	memory::dims dec_dst_iter_noctx_dims
	        = {dec_n_layers, 1, batch, feature_size};

Decoder : create memory description

.. ref-code-block:: cpp

	auto user_dec_wei_layer_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_dec_wei_layer_dims},
	        :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`dnnl::memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	auto user_dec_wei_iter_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_dec_wei_iter_dims},
	        :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`dnnl::memory::format_tag::ldigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa4e62e330c56963f9ead98490cd57ef7b>`);
	auto user_dec_bias_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({user_dec_bias_dims},
	        :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`dnnl::memory::format_tag::ldgo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab8690cd92ccee6a0ad55faccc0346aab>`);
	auto dec_dst_layer_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({dec_dst_layer_dims},
	        :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`dnnl::memory::format_tag::tnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac775cf954921a129a65eb929476de911>`);
	auto dec_src_layer_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({dec_src_layer_dims},
	        :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`dnnl::memory::format_tag::tnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fac775cf954921a129a65eb929476de911>`);
	auto dec_dst_iter_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({dec_dst_iter_dims},
	        :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`dnnl::memory::format_tag::ldnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab49be97ff353a86d84d06d98f846b61d>`);
	auto dec_dst_iter_c_md = :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`({dec_dst_iter_c_dims},
	        :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, :ref:`dnnl::memory::format_tag::ldnc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fab49be97ff353a86d84d06d98f846b61d>`);

Decoder : Create memory

.. ref-code-block:: cpp

	auto user_dec_wei_layer_memory = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(
	        user_dec_wei_layer_md, cpu_engine, user_dec_wei_layer.data());
	auto user_dec_wei_iter_memory = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(
	        user_dec_wei_iter_md, cpu_engine, user_dec_wei_iter.data());
	auto user_dec_bias_memory
	        = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(user_dec_bias_md, cpu_engine, user_dec_bias.data());
	auto user_dec_dst_layer_memory
	        = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(dec_dst_layer_md, cpu_engine, user_dec_dst.data());
	auto dec_src_layer_memory = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(dec_src_layer_md, cpu_engine);
	auto dec_dst_iter_c_memory = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(dec_dst_iter_c_md, cpu_engine);

Decoder : As mentioned above, we create a view without context out of the memory with context.

.. ref-code-block:: cpp

	auto dec_dst_iter_memory = :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`(dec_dst_iter_md, cpu_engine);
	auto dec_dst_iter_noctx_md = dec_dst_iter_md.:ref:`submemory_desc <doxid-structdnnl_1_1memory_1_1desc_1a7de2abef3b34e94c5dfa16e1fc3f3aab>`(
	        dec_dst_iter_noctx_dims, {0, 0, 0, 0, 0});

Decoder : Create RNN decoder cell

.. ref-code-block:: cpp

	auto dec_ctx_prim_desc = lstm_forward::primitive_desc(cpu_engine,
	        prop_kind::forward_inference,
	        rnn_direction::unidirectional_left2right, dec_src_layer_md,
	        dec_dst_iter_md, dec_dst_iter_c_md, dec_wei_layer_md,
	        dec_wei_iter_md, user_dec_bias_md, dec_dst_layer_md,
	        dec_dst_iter_noctx_md, dec_dst_iter_c_md);

Decoder : reorder weight memory

.. ref-code-block:: cpp

	auto dec_wei_layer_memory
	        = memory(dec_ctx_prim_desc.weights_layer_desc(), cpu_engine);
	auto dec_wei_layer_reorder_pd = reorder::primitive_desc(
	        user_dec_wei_layer_memory, dec_wei_layer_memory);
	reorder(dec_wei_layer_reorder_pd)
	        .execute(s, user_dec_wei_layer_memory, dec_wei_layer_memory);

	auto dec_wei_iter_memory
	        = memory(dec_ctx_prim_desc.weights_iter_desc(), cpu_engine);
	auto dec_wei_iter_reorder_pd = reorder::primitive_desc(
	        user_dec_wei_iter_memory, dec_wei_iter_memory);
	reorder(dec_wei_iter_reorder_pd)
	        .execute(s, user_dec_wei_iter_memory, dec_wei_iter_memory);

Decoder : add the rnn primitive with related arguments into decoder_net

.. ref-code-block:: cpp

	// TODO: add a reorder when they will be available
	decoder_net.push_back(lstm_forward(dec_ctx_prim_desc));
	decoder_net_args.push_back({{:ref:`DNNL_ARG_SRC_LAYER <doxid-group__dnnl__api__primitives__common_1gab91ce4d04cf4e98e3a407daa0676764f>`, dec_src_layer_memory},
	        {:ref:`DNNL_ARG_SRC_ITER <doxid-group__dnnl__api__primitives__common_1gaf35f4f604284f1b00bb35bffd0f7a143>`, dec_dst_iter_memory},
	        {:ref:`DNNL_ARG_SRC_ITER_C <doxid-group__dnnl__api__primitives__common_1ga8ef6969516e717208a33766542410410>`, dec_dst_iter_c_memory},
	        {:ref:`DNNL_ARG_WEIGHTS_LAYER <doxid-group__dnnl__api__primitives__common_1ga1ac9e1f1327be3902b488b64bae1b4c5>`, dec_wei_layer_memory},
	        {:ref:`DNNL_ARG_WEIGHTS_ITER <doxid-group__dnnl__api__primitives__common_1ga5a9c39486c01ad263e29677a32735af8>`, dec_wei_iter_memory},
	        {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, user_dec_bias_memory},
	        {:ref:`DNNL_ARG_DST_LAYER <doxid-group__dnnl__api__primitives__common_1gacfc123a6a4ff3b4af4cd27ed66fb8528>`, user_dec_dst_layer_memory},
	        {:ref:`DNNL_ARG_DST_ITER <doxid-group__dnnl__api__primitives__common_1ga13b91cbd3f531d9c90227895a275d5a6>`, dec_dst_iter_memory},
	        {:ref:`DNNL_ARG_DST_ITER_C <doxid-group__dnnl__api__primitives__common_1ga8b77d8716fc0ab9923d6cb409dbdf900>`, dec_dst_iter_c_memory}});

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

We initialize src_layer to the embedding of the end of sequence character, which are assumed to be 0 here

.. ref-code-block:: cpp

	memset(dec_src_layer_memory.:ref:`get_data_handle <doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>`(), 0,
	        dec_src_layer_memory.:ref:`get_desc <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`().:ref:`get_size <doxid-structdnnl_1_1memory_1_1desc_1abfa095ac138d4d2ef8efd3739e343f08>`());

From now on, src points to the output of the last iteration

Compute attention context vector into the first layer src_iter

.. ref-code-block:: cpp

	compute_attention(src_att_iter_handle, src_seq_length_max, batch,
	        feature_size, user_weights_attention_src_layer.data(),
	        src_att_layer_handle,
	        (float *)enc_bidir_dst_layer_memory.get_data_handle(),
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
	        = (float *)user_dec_dst_layer_memory.:ref:`get_data_handle <doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>`();
	dec_src_layer_memory.:ref:`set_data_handle <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>`(dst_layer_handle);
	user_dec_dst_layer_memory.:ref:`set_data_handle <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>`(
	        dst_layer_handle + batch * feature_size);

