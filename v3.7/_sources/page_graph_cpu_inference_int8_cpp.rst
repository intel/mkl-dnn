.. index:: pair: page; Convolution int8 inference example with Graph API
.. _doxid-graph_cpu_inference_int8_cpp:

Convolution int8 inference example with Graph API
=================================================

This is an example to demonstrate how to build an int8 graph with Graph API and run it on CPU.

This is an example to demonstrate how to build an int8 graph with Graph API and run it on CPU.

Example code: :ref:`cpu_inference_int8.cpp <doxid-cpu_inference_int8_8cpp-example>`

Some assumptions in this example:

* Only workflow is demonstrated without checking correctness

* Unsupported partitions should be handled by users themselves



.. _doxid-graph_cpu_inference_int8_cpp_1graph_cpu_inference_int8_cpp_headers:

Public headers
~~~~~~~~~~~~~~

To start using oneDNN Graph, we must include the ``dnnl_graph.hpp`` header file in the application. All the C++ APIs reside in namespace ``:ref:`dnnl::graph <doxid-namespacednnl_1_1graph>```.

.. ref-code-block:: cpp

	#include <iostream>
	#include <memory>
	#include <vector>
	#include <unordered_map>
	#include <unordered_set>
	
	#include <assert.h>
	
	#include "oneapi/dnnl/dnnl_graph.hpp"
	
	#include "example_utils.hpp"
	#include "graph_example_utils.hpp"
	
	using namespace :ref:`dnnl::graph <doxid-namespacednnl_1_1graph>`;
	using :ref:`data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` = :ref:`logical_tensor::data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>`;
	using :ref:`layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` = :ref:`logical_tensor::layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>`;
	using :ref:`property_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` = :ref:`logical_tensor::property_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>`;
	using dim = :ref:`logical_tensor::dim <doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>`;
	using dims = :ref:`logical_tensor::dims <doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`;





.. _doxid-graph_cpu_inference_int8_cpp_1graph_cpu_inference_int8_cpp_tutorial:

simple_pattern_int8() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-graph_cpu_inference_int8_cpp_1graph_cpu_inference_int8_cpp_get_partition:

Build Graph and Get Partitions
------------------------------

In this section, we are trying to build a graph indicating an int8 convolution with relu post-op. After that, we can get all of partitions which are determined by backend.

Create input/output :ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` and op for the first ``Dequantize``.

.. ref-code-block:: cpp

	logical_tensor dequant0_src_desc {0, data_type::u8};
	logical_tensor conv_src_desc {1, data_type::f32};
	op dequant0(2, op::kind::Dequantize, {dequant0_src_desc}, {conv_src_desc},
	        "dequant0");
	dequant0.set_attr<std::string>(op::attr::qtype, "per_tensor");
	dequant0.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
	dequant0.set_attr<std::vector<int64_t>>(op::attr::zps, {10});

Create input/output :ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` and op for the second ``Dequantize``.

.. note:: 

   It's necessary to provide scale and weight information on the ``Dequantize`` on weight.
   
   

.. note:: 

   Users can set weight property type to ``constant`` to enable dnnl weight cache for better performance
   
   


.. ref-code-block:: cpp

	logical_tensor dequant1_src_desc {3, data_type::s8};
	logical_tensor conv_weight_desc {
	        4, data_type::f32, 4, layout_type::undef, property_type::constant};
	op dequant1(5, op::kind::Dequantize, {dequant1_src_desc},
	        {conv_weight_desc}, "dequant1");
	dequant1.set_attr<std::string>(op::attr::qtype, "per_channel");
	// the memory format of weight is XIO, which indicates channel equals
	// to 64 for the convolution.
	std::vector<float> wei_scales(64, 0.1f);
	dims wei_zps(64, 0);
	dequant1.set_attr<std::vector<float>>(op::attr::scales, wei_scales);
	dequant1.set_attr<std::vector<int64_t>>(op::attr::zps, wei_zps);
	dequant1.set_attr<int64_t>(op::attr::axis, 1);













Create input/output :ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` the op for ``Convolution``.

.. ref-code-block:: cpp

	logical_tensor conv_bias_desc {
	        6, data_type::f32, 1, layout_type::undef, property_type::constant};
	logical_tensor conv_dst_desc {7, data_type::f32, layout_type::undef};

	// create the convolution op
	op conv(8, op::kind::Convolution,
	        {conv_src_desc, conv_weight_desc, conv_bias_desc}, {conv_dst_desc},
	        "conv");
	conv.set_attr<dims>(op::attr::strides, {1, 1});
	conv.set_attr<dims>(op::attr::pads_begin, {0, 0});
	conv.set_attr<dims>(op::attr::pads_end, {0, 0});
	conv.set_attr<dims>(op::attr::dilations, {1, 1});
	conv.set_attr<std::string>(op::attr::data_format, "NXC");
	conv.set_attr<std::string>(op::attr::weights_format, "XIO");
	conv.set_attr<int64_t>(op::attr::groups, 1);











Create input/output :ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` the op for ``ReLu``.

.. ref-code-block:: cpp

	logical_tensor relu_dst_desc {9, data_type::f32, layout_type::undef};
	op relu(10, op::kind::ReLU, {conv_dst_desc}, {relu_dst_desc}, "relu");









Create input/output :ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` the op for ``Quantize``.

.. ref-code-block:: cpp

	logical_tensor quant_dst_desc {11, data_type::u8, layout_type::undef};
	op quant(
	        12, op::kind::Quantize, {relu_dst_desc}, {quant_dst_desc}, "quant");
	quant.set_attr<std::string>(op::attr::qtype, "per_tensor");
	quant.set_attr<std::vector<float>>(op::attr::scales, {0.1f});
	quant.set_attr<std::vector<int64_t>>(op::attr::zps, {10});







Finally, those created ops will be added into the graph. The graph inside will maintain a list to store all these ops. To create a graph, :ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` is needed because the returned partitions maybe vary on different devices. For this example, we use CPU engine.

.. note:: 

   The order of adding op doesn't matter. The connection will be obtained through logical tensors.
   
   
Create graph and add ops to the graph

.. ref-code-block:: cpp

	graph g(:ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`);

	g.add_op(dequant0);
	g.add_op(dequant1);
	g.add_op(conv);
	g.add_op(relu);
	g.add_op(quant);





After finished above operations, we can get partitions by calling :ref:`dnnl::graph::graph::get_partitions() <doxid-classdnnl_1_1graph_1_1graph_1a116d3552e3b0e6c739a1564329bde014>`.

In this example, the graph will be partitioned into one partition.

.. ref-code-block:: cpp

	auto partitions = g.get_partitions();





.. _doxid-graph_cpu_inference_int8_cpp_1graph_cpu_inference_int8_cpp_compile:

Compile and Execute Partition
-----------------------------

In the real case, users like framework should provide device information at this stage. But in this example, we just use a self-defined device to simulate the real behavior.

Create a :ref:`dnnl::engine <doxid-structdnnl_1_1engine>`. Also, set a user-defined :ref:`dnnl::graph::allocator <doxid-classdnnl_1_1graph_1_1allocator>` to this engine.

.. ref-code-block:: cpp

	allocator alloc {};
	:ref:`dnnl::engine <doxid-structdnnl_1_1engine>` eng
	        = :ref:`make_engine_with_allocator <doxid-group__dnnl__graph__api__engine_1ga42ac93753b2a12d14b29704fe3b0b2fa>`(:ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`, 0, alloc);
	:ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm {eng};

Compile the partition to generate compiled partition with the input and output logical tensors.

.. ref-code-block:: cpp

	compiled_partition cp = partition.compile(inputs, outputs, eng);





Execute the compiled partition on the specified stream.

.. ref-code-block:: cpp

	cp.execute(strm, inputs_ts, outputs_ts);

