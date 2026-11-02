.. index:: pair: page; CNN int8 inference example
.. _doxid-cnn_inference_int8_cpp:

CNN int8 inference example
==========================

This C++ API example demonstrates how to run AlexNet's conv3 and relu3 with int8 data type.

This C++ API example demonstrates how to run AlexNet's conv3 and relu3 with int8 data type.

Example code: :ref:`cnn_inference_int8.cpp <doxid-cnn_inference_int8_8cpp-example>`

Configure tensor shapes

.. ref-code-block:: cpp

	// AlexNet: conv3
	// {batch, 256, 13, 13} (x)  {384, 256, 3, 3}; -> {batch, 384, 13, 13}
	// strides: {1, 1}
	memory::dims conv_src_tz = {batch, 256, 13, 13};
	memory::dims conv_weights_tz = {384, 256, 3, 3};
	memory::dims conv_bias_tz = {384};
	memory::dims conv_dst_tz = {batch, 384, 13, 13};
	memory::dims conv_strides = {1, 1};
	memory::dims conv_padding = {1, 1};





















Next, the example configures the scales used to quantize f32 data into int8. For this example, the scaling value is chosen as an arbitrary number, although in a realistic scenario, it should be calculated from a set of precomputed values as previously mentioned.

.. ref-code-block:: cpp

	// Choose scaling factors for input, weight and output
	std::vector<float> src_scales = {1.8f};
	std::vector<float> weight_scales = {2.0f};
	std::vector<float> dst_scales = {0.55f};



















The source, weights, bias and destination datasets use the single-scale format with mask set to '0'.

.. ref-code-block:: cpp

	const int src_mask = 0;
	const int weight_mask = 0;
	const int dst_mask = 0;

















Create the memory primitives for user data (source, weights, and bias). The user data will be in its original 32-bit floating point format.

.. ref-code-block:: cpp

	auto user_src_memory = memory({{conv_src_tz}, dt::f32, tag::nchw}, eng);
	write_to_dnnl_memory(user_src.data(), user_src_memory);
	auto user_weights_memory
	        = memory({{conv_weights_tz}, dt::f32, tag::oihw}, eng);
	write_to_dnnl_memory(conv_weights.data(), user_weights_memory);
	auto user_bias_memory = memory({{conv_bias_tz}, dt::f32, tag::x}, eng);
	write_to_dnnl_memory(conv_bias.data(), user_bias_memory);















Create a memory descriptor for each convolution parameter. The convolution data uses 8-bit integer values, so the memory descriptors are configured as:

* 8-bit unsigned (u8) for source and destination.

* 8-bit signed (s8) for bias and weights.

Note The destination type is chosen as unsigned because the convolution applies a ReLU operation where data results :math:`\geq 0`.



.. ref-code-block:: cpp

	auto conv_src_md = memory::desc({conv_src_tz}, dt::u8, tag::any);
	auto conv_bias_md = memory::desc({conv_bias_tz}, dt::s8, tag::any);
	auto conv_weights_md = memory::desc({conv_weights_tz}, dt::s8, tag::any);
	auto conv_dst_md = memory::desc({conv_dst_tz}, dt::u8, tag::any);













Configuring int8-specific parameters in an int8 primitive is done via the Attributes Primitive. Create an attributes object for the convolution and configure it accordingly.

.. ref-code-block:: cpp

	primitive_attr conv_attr;
	conv_attr.set_scales_mask(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_mask);
	conv_attr.set_scales_mask(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, weight_mask);
	conv_attr.set_scales_mask(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_mask);

	// Prepare dst scales
	auto dst_scale_md = memory::desc({1}, dt::f32, tag::x);
	auto dst_scale_memory = memory(dst_scale_md, eng);
	write_to_dnnl_memory(dst_scales.data(), dst_scale_memory);











The ReLU layer from Alexnet is executed through the PostOps feature. Create a PostOps object and configure it to execute an eltwise relu operation.

.. ref-code-block:: cpp

	const float ops_alpha = 0.f; // relu negative slope
	const float ops_beta = 0.f;
	post_ops ops;
	ops.append_eltwise(algorithm::eltwise_relu, ops_alpha, ops_beta);
	conv_attr.set_post_ops(ops);









Create a primitive descriptor passing the int8 memory descriptors and int8 attributes to the constructor. The primitive descriptor for the convolution will contain the specific memory formats for the computation.

.. ref-code-block:: cpp

	auto conv_prim_desc = convolution_forward::primitive_desc(eng,
	        prop_kind::forward, algorithm::convolution_direct, conv_src_md,
	        conv_weights_md, conv_bias_md, conv_dst_md, conv_strides,
	        conv_padding, conv_padding, conv_attr);







Create a memory for each of the convolution's data input parameters (source, bias, weights, and destination). Using the convolution primitive descriptor as the creation parameter enables oneDNN to configure the memory formats for the convolution.

Scaling parameters are passed to the reorder primitive via the attributes primitive.

User memory must be transformed into convolution-friendly memory (for int8 and memory format). A reorder layer performs the data transformation from f32 (the original user data) into int8 format (the data used for the convolution). In addition, the reorder transforms the user data into the required memory format (as explained in the simple_net example).

.. ref-code-block:: cpp

	auto conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
	primitive_attr src_attr;
	src_attr.set_scales_mask(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_mask);
	auto src_scale_md = memory::desc({1}, dt::f32, tag::x);
	auto src_scale_memory = memory(src_scale_md, eng);
	write_to_dnnl_memory(src_scales.data(), src_scale_memory);
	auto src_reorder_pd
	        = reorder::primitive_desc(eng, user_src_memory.get_desc(), eng,
	                conv_src_memory.get_desc(), src_attr);
	auto src_reorder = reorder(src_reorder_pd);
	src_reorder.execute(s,
	        {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, user_src_memory}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_src_memory},
	                {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, src_scale_memory}});

	auto conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
	primitive_attr weight_attr;
	weight_attr.set_scales_mask(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, weight_mask);
	auto wei_scale_md = memory::desc({1}, dt::f32, tag::x);
	auto wei_scale_memory = memory(wei_scale_md, eng);
	write_to_dnnl_memory(weight_scales.data(), wei_scale_memory);
	auto weight_reorder_pd
	        = reorder::primitive_desc(eng, user_weights_memory.get_desc(), eng,
	                conv_weights_memory.get_desc(), weight_attr);
	auto weight_reorder = reorder(weight_reorder_pd);
	weight_reorder.execute(s,
	        {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, user_weights_memory},
	                {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, conv_weights_memory},
	                {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, wei_scale_memory}});

	auto conv_bias_memory = memory(conv_prim_desc.bias_desc(), eng);
	write_to_dnnl_memory(conv_bias.data(), conv_bias_memory);





Create the convolution primitive and add it to the net. The int8 example computes the same Convolution +ReLU layers from AlexNet simple-net.cpp using the int8 and PostOps approach. Although performance is not measured here, in practice it would require less computation time to achieve similar results.

.. ref-code-block:: cpp

	auto conv = convolution_forward(conv_prim_desc);
	conv.execute(s,
	        {{:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src_memory},
	                {:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, conv_weights_memory},
	                {:ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`, conv_bias_memory},
	                {:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, conv_dst_memory},
	                {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_scale_memory},
	                {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_scale_memory},
	                {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, dst_scale_memory}});

Finally, dst memory may be dequantized from int8 into the original f32 format. Create a memory primitive for the user data in the original 32-bit floating point format and then apply a reorder to transform the computation output data.

.. ref-code-block:: cpp

	auto user_dst_memory = memory({{conv_dst_tz}, dt::f32, tag::nchw}, eng);
	write_to_dnnl_memory(user_dst.data(), user_dst_memory);
	primitive_attr dst_attr;
	dst_attr.set_scales_mask(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, dst_mask);
	auto dst_reorder_pd
	        = reorder::primitive_desc(eng, conv_dst_memory.get_desc(), eng,
	                user_dst_memory.get_desc(), dst_attr);
	auto dst_reorder = reorder(dst_reorder_pd);
	dst_reorder.execute(s,
	        {{:ref:`DNNL_ARG_FROM <doxid-group__dnnl__api__primitives__common_1ga953b34f004a8222b04e21851487c611a>`, conv_dst_memory}, {:ref:`DNNL_ARG_TO <doxid-group__dnnl__api__primitives__common_1gaf700c3396987b450413c8df5d78bafd9>`, user_dst_memory},
	                {:ref:`DNNL_ARG_ATTR_SCALES <doxid-group__dnnl__api__primitives__common_1ga7f52f0ef5ceb99e163f3ba7f83c18aed>` | :ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, dst_scale_memory}});

[Dequantize the result]

