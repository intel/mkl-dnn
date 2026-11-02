.. index:: pair: page; CNN f32 inference example
.. _doxid-cnn_inference_f32_cpp:

CNN f32 inference example
=========================

This C++ API example demonstrates how to build an AlexNet neural network topology for forward-pass inference.

This C++ API example demonstrates how to build an AlexNet neural network topology for forward-pass inference.

Example code: :ref:`cnn_inference_f32.cpp <doxid-cnn_inference_f32_8cpp-example>`

Some key take-aways include:

* How tensors are implemented and submitted to primitives.

* How primitives are created.

* How primitives are sequentially submitted to the network, where the output from primitives is passed as input to the next primitive. The latter specifies a dependency between the primitive input and output data.

* Specific 'inference-only' configurations.

* Limiting the number of reorders performed that are detrimental to performance.

The example implements the AlexNet layers as numbered primitives (for example, conv1, pool1, conv2).

Initialize an engine and stream. The last parameter in the call represents the index of the engine.

.. ref-code-block:: cpp

	:ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>` eng(engine_kind, 0);
	stream s(eng);























Create a vector for the primitives and a vector to hold memory that will be used as arguments.

.. ref-code-block:: cpp

	std::vector<primitive> net;
	std::vector<std::unordered_map<int, memory>> net_args;





















Allocate buffers for input and output data, weights, and bias.

.. ref-code-block:: cpp

	std::vector<float> user_src(batch * 3 * 227 * 227);
	std::vector<float> user_dst(batch * 1000);
	std::vector<float> conv1_weights(product(conv1_weights_tz));
	std::vector<float> conv1_bias(product(conv1_bias_tz));



















Create memory that describes data layout in the buffers. This example uses tag::nchw (batch-channels-height-width) for input data and tag::oihw for weights.

.. ref-code-block:: cpp

	auto user_src_memory = memory({{conv1_src_tz}, dt::f32, tag::nchw}, eng);
	write_to_dnnl_memory(user_src.data(), user_src_memory);
	auto user_weights_memory
	        = memory({{conv1_weights_tz}, dt::f32, tag::oihw}, eng);
	write_to_dnnl_memory(conv1_weights.data(), user_weights_memory);
	auto conv1_user_bias_memory
	        = memory({{conv1_bias_tz}, dt::f32, tag::x}, eng);
	write_to_dnnl_memory(conv1_bias.data(), conv1_user_bias_memory);

















Create memory descriptors with layout tag::any. The ``any`` format enables the convolution primitive to choose the data format that will result in best performance based on its input parameters (convolution kernel sizes, strides, padding, and so on). If the resulting format is different from ``nchw``, the user data must be transformed to the format required for the convolution (as explained below).

.. ref-code-block:: cpp

	auto conv1_src_md = memory::desc({conv1_src_tz}, dt::f32, tag::any);
	auto conv1_bias_md = memory::desc({conv1_bias_tz}, dt::f32, tag::any);
	auto conv1_weights_md = memory::desc({conv1_weights_tz}, dt::f32, tag::any);
	auto conv1_dst_md = memory::desc({conv1_dst_tz}, dt::f32, tag::any);















Create a convolution primitive descriptor by specifying engine, propagation kind, :ref:`convolution algorithm <doxid-dev_guide_convolution>`, shapes of input, weights, bias, output, convolution strides, padding, and kind of padding. Propagation kind is set to prop_kind::forward_inference to optimize for inference execution and omit computations that are necessary only for backward propagation. Once created, it has specific formats instead of the ``any`` format.

.. ref-code-block:: cpp

	auto conv1_prim_desc = convolution_forward::primitive_desc(eng,
	        prop_kind::forward_inference, algorithm::convolution_direct,
	        conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
	        conv1_strides, conv1_padding, conv1_padding);













Check whether data and weights formats required by convolution is different from the user format. In case it is different change the layout using reorder primitive.

.. ref-code-block:: cpp

	auto conv1_src_memory = user_src_memory;
	if (conv1_prim_desc.src_desc() != user_src_memory.get_desc()) {
	    conv1_src_memory = memory(conv1_prim_desc.src_desc(), eng);
	    net.push_back(reorder(user_src_memory, conv1_src_memory));
	    net_args.push_back({{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, user_src_memory},
	            {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv1_src_memory}});
	}

	auto conv1_weights_memory = user_weights_memory;
	if (conv1_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
	    conv1_weights_memory = memory(conv1_prim_desc.weights_desc(), eng);
	    reorder(user_weights_memory, conv1_weights_memory)
	            .execute(s, user_weights_memory, conv1_weights_memory);
	}











Create a memory primitive for output.

.. ref-code-block:: cpp

	auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);









Create a convolution primitive and add it to the net.

.. ref-code-block:: cpp

	auto conv1_dst_memory = memory(conv1_prim_desc.dst_desc(), eng);







Create the relu primitive. For better performance, keep the input data format for ReLU (as well as for other operation primitives until another convolution or inner product is encountered) the same as the one chosen for convolution. Also note that ReLU is done in-place by using conv1 memory.

.. ref-code-block:: cpp

	auto relu1_prim_desc
	        = eltwise_forward::primitive_desc(eng, prop_kind::forward_inference,
	                algorithm::eltwise_relu, conv1_dst_memory.get_desc(),
	                conv1_dst_memory.get_desc(), negative1_slope);

	net.push_back(eltwise_forward(relu1_prim_desc));
	net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv1_dst_memory},
	        {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv1_dst_memory}});





For training execution, pooling requires a private workspace memory to perform the backward pass. However, pooling should not use 'workspace' for inference, because this is detrimental to performance.

.. ref-code-block:: cpp

	auto pool1_pd = pooling_forward::primitive_desc(eng,
	        prop_kind::forward_inference, algorithm::pooling_max,
	        lrn1_dst_memory.get_desc(), pool1_dst_md, pool1_strides,
	        pool1_kernel, pool_dilation, pool_padding, pool_padding);
	auto pool1_dst_memory = memory(pool1_pd.dst_desc(), eng);

	net.push_back(pooling_forward(pool1_pd));
	net_args.push_back({{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, lrn1_dst_memory},
	        {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, pool1_dst_memory}});

The example continues to create more layers according to the AlexNet topology.

Finally, execute the primitives. For this example, the net is executed multiple times and each execution is timed individually.

.. ref-code-block:: cpp

	for (int j = 0; j < times; ++j) {
	    assert(net.size() == net_args.size() && "something is missing");
	    for (size_t i = 0; i < net.size(); ++i)
	        net.at(i).execute(s, net_args.at(i));
	}

