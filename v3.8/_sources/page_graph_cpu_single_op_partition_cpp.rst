.. index:: pair: page; Single op partition on CPU
.. _doxid-graph_cpu_single_op_partition_cpp:

Single op partition on CPU
==========================

This is an example to demonstrate how to build a simple op graph and run it on CPU.

This is an example to demonstrate how to build a simple op graph and run it on CPU.

Example code: :ref:`cpu_single_op_partition.cpp <doxid-cpu_single_op_partition_8cpp-example>`

Some key take-aways included in this example:

* how to build a single-op partition quickly

* how to create an engine, allocator and stream

* how to compile a partition

* how to execute a compiled partition

Some assumptions in this example:

* Only workflow is demonstrated without checking correctness

* Unsupported partitions should be handled by users themselves



.. _doxid-graph_cpu_single_op_partition_cpp_1graph_cpu_single_op_partition_cpp_headers:

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





.. _doxid-graph_cpu_single_op_partition_cpp_1graph_cpu_single_op_partition_cpp_tutorial:

cpu_single_op_partition_tutorial() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-graph_cpu_single_op_partition_cpp_1graph_cpu_single_op_partition_cpp_get_partition:

Build Graph and Get Partitions
------------------------------

In this section, we are trying to create a partition containing the single op ``matmul`` without building a graph and getting partition.

Create the ``Matmul`` op (:ref:`dnnl::graph::op <doxid-classdnnl_1_1graph_1_1op>`) and attaches attributes to it, including ``transpose_a`` and ``transpose_b``.

.. ref-code-block:: cpp

	logical_tensor matmul_src0_desc {0, data_type::f32};
	logical_tensor matmul_src1_desc {1, data_type::f32};
	logical_tensor matmul_dst_desc {2, data_type::f32};
	op matmul(0, op::kind::MatMul, {matmul_src0_desc, matmul_src1_desc},
	        {matmul_dst_desc}, "matmul");
	matmul.set_attr<bool>(op::attr::transpose_a, false);
	matmul.set_attr<bool>(op::attr::transpose_b, false);





.. _doxid-graph_cpu_single_op_partition_cpp_1graph_cpu_single_op_partition_cpp_compile:

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









Skip building graph and getting partition, and directly create the single-op partition

.. ref-code-block:: cpp

	partition part(matmul, :ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`);







Compile the partition to generate compiled partition with the input and output logical tensors.

.. ref-code-block:: cpp

	compiled_partition cp = part.compile(inputs, outputs, eng);





Execute the compiled partition on the specified stream.

.. ref-code-block:: cpp

	cp.execute(strm, inputs_ts, outputs_ts);

