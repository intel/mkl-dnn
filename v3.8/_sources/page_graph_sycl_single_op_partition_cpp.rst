.. index:: pair: page; Single op partition on GPU
.. _doxid-graph_sycl_single_op_partition_cpp:

Single op partition on GPU
==========================

This is an example to demonstrate how to build a simple op graph and run it on gpu.

This is an example to demonstrate how to build a simple op graph and run it on gpu.

Example code: :ref:`sycl_single_op_partition.cpp <doxid-sycl_single_op_partition_8cpp-example>`

Some key take-aways included in this example:

* how to build a single-op partition quickly

* how to create an engine, allocator and stream

* how to compile a partition

* how to execute a compiled partition

Some assumptions in this example:

* Only workflow is demonstrated without checking correctness

* Unsupported partitions should be handled by users themselves



.. _doxid-graph_sycl_single_op_partition_cpp_1graph_sycl_single_op_partition_cpp_headers:

Public headers
~~~~~~~~~~~~~~

To start using oneDNN Graph, we must include the ``dnnl_graph.hpp`` header file in the application. All the C++ APIs reside in namespace ``:ref:`dnnl::graph <doxid-namespacednnl_1_1graph>```.

.. ref-code-block:: cpp

	#include "oneapi/dnnl/dnnl_graph.hpp"
	#include "oneapi/dnnl/dnnl_graph_sycl.hpp"
	#include "oneapi/dnnl/dnnl_sycl.hpp"
	using namespace :ref:`dnnl::graph <doxid-namespacednnl_1_1graph>`;
	using namespace :ref:`sycl <doxid-namespacesycl>`;
	
	#include <assert.h>
	#include <iostream>
	#include <memory>
	#include <vector>
	#include <unordered_map>
	#include <unordered_set>
	
	#include "example_utils.hpp"
	#include "graph_example_utils.hpp"
	
	using namespace :ref:`dnnl::graph <doxid-namespacednnl_1_1graph>`;
	using :ref:`data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` = :ref:`logical_tensor::data_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>`;
	using :ref:`layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` = :ref:`logical_tensor::layout_type <doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>`;
	using dim = :ref:`logical_tensor::dim <doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>`;
	using dims = :ref:`logical_tensor::dims <doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`;





.. _doxid-graph_sycl_single_op_partition_cpp_1graph_sycl_single_op_partition_cpp_tutorial:

sycl_single_op_partition_tutorial() function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. _doxid-graph_sycl_single_op_partition_cpp_1graph_sycl_single_op_partition_cpp_get_partition:

Build Graph and Get Partitions
------------------------------

In this section, we are trying to create a partition containing the single op ``matmul`` without building a graph and getting partition.

Create first ``Matmul`` op (:ref:`dnnl::graph::op <doxid-classdnnl_1_1graph_1_1op>`) and attaches attributes to it, including ``transpose_a`` and ``transpose_b``.

.. ref-code-block:: cpp

	logical_tensor matmul_src0_desc {0, data_type::f32};
	logical_tensor matmul_src1_desc {1, data_type::f32};
	logical_tensor matmul_dst_desc {2, data_type::f32};
	op matmul(0, op::kind::MatMul, {matmul_src0_desc, matmul_src1_desc},
	        {matmul_dst_desc}, "matmul");
	matmul.set_attr<bool>(op::attr::transpose_a, false);
	matmul.set_attr<bool>(op::attr::transpose_b, false);





.. _doxid-graph_sycl_single_op_partition_cpp_1graph_sycl_single_op_partition_cpp_compile:

Compile and Execute Partition
-----------------------------

In the real case, users like framework should provide device information at this stage. But in this example, we just use a self-defined device to simulate the real behavior.

Create a :ref:`dnnl::graph::allocator <doxid-classdnnl_1_1graph_1_1allocator>` with two user-defined :ref:`dnnl_graph_sycl_allocate_f <doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>` and :ref:`dnnl_graph_sycl_deallocate_f <doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>` call-back functions.

.. ref-code-block:: cpp

	allocator alloc = :ref:`sycl_interop::make_allocator <doxid-namespacednnl_1_1graph_1_1ocl__interop_1a74e6e92b50043bf02f8b936a481af85a>`(
	        sycl_malloc_wrapper, sycl_free_wrapper);

Define SYCL queue (code outside of oneDNN graph)

.. ref-code-block:: cpp

	sycl::queue q = (ekind == engine::kind::gpu)
	        ? sycl::queue(
	                sycl::gpu_selector_v, sycl::property::queue::in_order {})
	        : sycl::queue(
	                sycl::cpu_selector_v, sycl::property::queue::in_order {});













Create a :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` based on SYCL device and context. Also, set a user-defined :ref:`dnnl::graph::allocator <doxid-classdnnl_1_1graph_1_1allocator>` to this engine.

.. ref-code-block:: cpp

	:ref:`dnnl::engine <doxid-structdnnl_1_1engine>` eng = :ref:`sycl_interop::make_engine_with_allocator <doxid-group__dnnl__graph__api__engine_1ga42ac93753b2a12d14b29704fe3b0b2fa>`(
	        q.get_device(), q.get_context(), alloc);











Create a :ref:`dnnl::stream <doxid-structdnnl_1_1stream>` on a given engine

.. ref-code-block:: cpp

	:ref:`dnnl::stream <doxid-structdnnl_1_1stream>` strm = :ref:`dnnl::sycl_interop::make_stream <doxid-namespacednnl_1_1sycl__interop_1a170bddd16d53869fc18412894400ccab>`(eng, q);









Skip building graph and getting partition, and directly create the single-op partition

.. ref-code-block:: cpp

	partition part(matmul, :ref:`dnnl::engine::kind::cpu <doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1aad9747e2da342bdb995f6389533ad1a3d>`);







Compile the partition to generate compiled partition with the input and output logical tensors.

.. ref-code-block:: cpp

	compiled_partition cp = part.compile(inputs, outputs, eng);





Execute the compiled partition on the specified stream.

.. ref-code-block:: cpp

	cp.execute(strm, inputs_ts, outputs_ts);

