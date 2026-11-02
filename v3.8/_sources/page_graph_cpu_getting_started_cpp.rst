.. index:: pair: page; Getting started on CPU with Graph API
.. _doxid-graph_cpu_getting_started_cpp:

Getting started on CPU with Graph API
=====================================

This is an example to demonstrate how to build a simple graph and run it on CPU.

This is an example to demonstrate how to build a simple graph and run it on CPU.

Example code: :ref:`cpu_getting_started.cpp <doxid-cpu_getting_started_8cpp-example>`

Some key take-aways included in this example:

* how to build a graph and get partitions from it

* how to create an engine, allocator and stream

* how to compile a partition

* how to execute a compiled partition

Some assumptions in this example:

* Only workflow is demonstrated without checking correctness

* Unsupported partitions should be handled by users themselves



.. _doxid-graph_cpu_getting_started_cpp_1graph_cpu_getting_started_cpp_headers:

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
	using dim = :ref:`logical_tensor::dim <doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>`;
	using dims = :ref:`logical_tensor::dims <doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`;





.. _doxid-graph_cpu_getting_started_cpp_1graph_cpu_getting_started_cpp_tutorial:

cpu_getting_started_tutorial() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-graph_cpu_getting_started_cpp_1graph_cpu_getting_started_cpp_get_partition:

Build Graph and Get Partitions
------------------------------

In this section, we are trying to build a graph containing the pattern ``conv0->relu0->conv1->relu1``. After that, we can get all of partitions which are determined by backend.

To build a graph, the connection relationship of different ops must be known. In oneDNN Graph, :ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` is used to express such relationship. So, next step is to create logical tensors for these ops including inputs and outputs.

.. note:: 

   It's not necessary to provide concrete shape/layout information at graph partitioning stage. Users can provide these information till compilation stage.
   
   
Create input/output :ref:`dnnl::graph::logical_tensor <doxid-classdnnl_1_1graph_1_1logical__tensor>` for first ``Convolution`` op.

.. ref-code-block:: cpp

	logical_tensor conv0_src_desc {0, data_type::f32};
	logical_tensor conv0_weight_desc {1, data_type::f32};
	logical_tensor conv0_dst_desc {2, data_type::f32};





















Create first ``Convolution`` op (:ref:`dnnl::graph::op <doxid-classdnnl_1_1graph_1_1op>`) and attaches attributes to it, such as ``strides``, ``pads_begin``, ``pads_end``, ``data_format``, etc.

.. ref-code-block:: cpp

	op conv0(0, op::kind::Convolution, {conv0_src_desc, conv0_weight_desc},
	        {conv0_dst_desc}, "conv0");
	conv0.set_attr<dims>(op::attr::strides, {4, 4});
	conv0.set_attr<dims>(op::attr::pads_begin, {0, 0});
	conv0.set_attr<dims>(op::attr::pads_end, {0, 0});
	conv0.set_attr<dims>(op::attr::dilations, {1, 1});
	conv0.set_attr<int64_t>(op::attr::groups, 1);
	conv0.set_attr<std::string>(op::attr::data_format, "NCX");
	conv0.set_attr<std::string>(op::attr::weights_format, "OIX");



















Create input/output logical tensors for first ``BiasAdd`` op and create the first ``BiasAdd`` op

.. ref-code-block:: cpp

	logical_tensor conv0_bias_desc {3, data_type::f32};
	logical_tensor conv0_bias_add_dst_desc {
	        4, data_type::f32, layout_type::undef};
	op conv0_bias_add(1, op::kind::BiasAdd, {conv0_dst_desc, conv0_bias_desc},
	        {conv0_bias_add_dst_desc}, "conv0_bias_add");
	conv0_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");

















Create output logical tensors for first ``Relu`` op and create the op.

.. ref-code-block:: cpp

	logical_tensor relu0_dst_desc {5, data_type::f32};
	op relu0(2, op::kind::ReLU, {conv0_bias_add_dst_desc}, {relu0_dst_desc},
	        "relu0");















Create input/output logical tensors for second ``Convolution`` op and create the second ``Convolution`` op.

.. ref-code-block:: cpp

	logical_tensor conv1_weight_desc {6, data_type::f32};
	logical_tensor conv1_dst_desc {7, data_type::f32};
	op conv1(3, op::kind::Convolution, {relu0_dst_desc, conv1_weight_desc},
	        {conv1_dst_desc}, "conv1");
	conv1.set_attr<dims>(op::attr::strides, {1, 1});
	conv1.set_attr<dims>(op::attr::pads_begin, {0, 0});
	conv1.set_attr<dims>(op::attr::pads_end, {0, 0});
	conv1.set_attr<dims>(op::attr::dilations, {1, 1});
	conv1.set_attr<int64_t>(op::attr::groups, 1);
	conv1.set_attr<std::string>(op::attr::data_format, "NCX");
	conv1.set_attr<std::string>(op::attr::weights_format, "OIX");













Create input/output logical tensors for second ``BiasAdd`` op and create the op.

.. ref-code-block:: cpp

	logical_tensor conv1_bias_desc {8, data_type::f32};
	logical_tensor conv1_bias_add_dst_desc {9, data_type::f32};
	op conv1_bias_add(4, op::kind::BiasAdd, {conv1_dst_desc, conv1_bias_desc},
	        {conv1_bias_add_dst_desc}, "conv1_bias_add");
	conv1_bias_add.set_attr<std::string>(op::attr::data_format, "NCX");











Create output logical tensors for second ``Relu`` op and create the op

.. ref-code-block:: cpp

	logical_tensor relu1_dst_desc {10, data_type::f32};
	op relu1(5, op::kind::ReLU, {conv1_bias_add_dst_desc}, {relu1_dst_desc},
	        "relu1");









Finally, those created ops will be added into the graph. The graph inside will maintain a list to store all these ops. To create a graph, :ref:`dnnl::engine::kind <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` is needed because the returned partitions maybe vary on different devices. For this example, we use CPU engine.

.. note:: 

   The order of adding op doesn't matter. The connection will be obtained through logical tensors.
   
   
Create graph and add ops to the graph

.. ref-code-block:: cpp

	graph g(:ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`);

	g.add_op(conv0);
	g.add_op(conv0_bias_add);
	g.add_op(relu0);

	g.add_op(conv1);
	g.add_op(conv1_bias_add);
	g.add_op(relu1);







After adding all ops into the graph, call :ref:`dnnl::graph::graph::get_partitions() <doxid-classdnnl_1_1graph_1_1graph_1a116d3552e3b0e6c739a1564329bde014>` to indicate that the graph building is over and is ready for partitioning. Adding new ops into a finalized graph or partitioning a unfinalized graph will both lead to a failure.

.. ref-code-block:: cpp

	g.finalize();





After finished above operations, we can get partitions by calling :ref:`dnnl::graph::graph::get_partitions() <doxid-classdnnl_1_1graph_1_1graph_1a116d3552e3b0e6c739a1564329bde014>`.

In this example, the graph will be partitioned into two partitions:

#. conv0 + conv0_bias_add + relu0

#. conv1 + conv1_bias_add + relu1

.. ref-code-block:: cpp

	auto partitions = g.get_partitions();





.. _doxid-graph_cpu_getting_started_cpp_1graph_cpu_getting_started_cpp_compile:

Compile and Execute Partition
-----------------------------

In the real case, users like framework should provide device information at this stage. But in this example, we just use a self-defined device to simulate the real behavior.

Create a :ref:`dnnl::engine <doxid-structdnnl_1_1engine>`. Also, set a user-defined :ref:`dnnl::graph::allocator <doxid-classdnnl_1_1graph_1_1allocator>` to this engine.

.. ref-code-block:: cpp

	allocator alloc {};
	:ref:`dnnl::engine <doxid-structdnnl_1_1engine>` eng
	        = :ref:`make_engine_with_allocator <doxid-group__dnnl__graph__api__engine_1ga42ac93753b2a12d14b29704fe3b0b2fa>`(:ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`, 0, alloc);

Create a :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` on a given engine

.. ref-code-block:: cpp

	:ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm {eng};







Compile the partition to generate compiled partition with the input and output logical tensors.

.. ref-code-block:: cpp

	compiled_partition cp = partition.compile(inputs, outputs, eng);





Execute the compiled partition on the specified stream.

.. ref-code-block:: cpp

	cp.execute(strm, inputs_ts, outputs_ts);

