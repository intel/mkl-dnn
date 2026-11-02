.. index:: pair: namespace; dnnl::graph::ocl_interop
.. _doxid-namespacednnl_1_1graph_1_1ocl__interop:

namespace dnnl::graph::ocl_interop
==================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

OpenCL interoperability namespace. :ref:`More...<details-namespacednnl_1_1graph_1_1ocl__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace ocl_interop {

	// global functions

	:ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>` :ref:`make_allocator<doxid-namespacednnl_1_1graph_1_1ocl__interop_1a74e6e92b50043bf02f8b936a481af85a>`(
		:ref:`dnnl_graph_ocl_allocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga2900b26adec541b7577667ad3b55fa4d>` ocl_malloc,
		:ref:`dnnl_graph_ocl_deallocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga93912e04c48608be40c1c656cc721ac9>` ocl_free
		);

	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`make_engine_with_allocator<doxid-namespacednnl_1_1graph_1_1ocl__interop_1a930e4e54c56a12b7a0ff0f133541da3e>`(
		cl_device_id device,
		cl_context context,
		const :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`& alloc
		);

	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`make_engine_with_allocator<doxid-namespacednnl_1_1graph_1_1ocl__interop_1ae068e56aeb8ba7d88f3b64c5c478098b>`(
		cl_device_id device,
		cl_context context,
		const :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`& alloc,
		const std::vector<uint8_t>& cache_blob
		);

	cl_event :ref:`execute<doxid-namespacednnl_1_1graph_1_1ocl__interop_1a8b1d57febf09dc0621d7aa2a8dc13035>`(
		:ref:`compiled_partition<doxid-classdnnl_1_1graph_1_1compiled__partition>`& c_partition,
		:ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& inputs,
		std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& outputs,
		const std::vector<cl_event>& deps = {}
		);

	} // namespace ocl_interop
.. _details-namespacednnl_1_1graph_1_1ocl__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

OpenCL interoperability namespace.

Global Functions
----------------

.. index:: pair: function; make_allocator
.. _doxid-namespacednnl_1_1graph_1_1ocl__interop_1a74e6e92b50043bf02f8b936a481af85a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>` make_allocator(
		:ref:`dnnl_graph_ocl_allocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga2900b26adec541b7577667ad3b55fa4d>` ocl_malloc,
		:ref:`dnnl_graph_ocl_deallocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga93912e04c48608be40c1c656cc721ac9>` ocl_free
		)

Constructs an allocator from OpenCL malloc and free function pointer.

OpenCL allocator should be used for OpenCL GPU runtime. Currently, only device USM allocator is supported.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- ocl_malloc

		- The pointer to OpenCL malloc function

	*
		- ocl_free

		- The pointer to OpenCL free function



.. rubric:: Returns:

Created allocator

.. index:: pair: function; make_engine_with_allocator
.. _doxid-namespacednnl_1_1graph_1_1ocl__interop_1a930e4e54c56a12b7a0ff0f133541da3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` make_engine_with_allocator(
		cl_device_id device,
		cl_context context,
		const :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`& alloc
		)

Constructs an engine from an OpenCL device, an OpenCL context, and an allocator.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- device

		- A valid OpenCL device to construct the engine

	*
		- context

		- A valid OpenCL context to construct the engine

	*
		- alloc

		- An allocator to associate with the engine



.. rubric:: Returns:

Created engine

.. index:: pair: function; make_engine_with_allocator
.. _doxid-namespacednnl_1_1graph_1_1ocl__interop_1ae068e56aeb8ba7d88f3b64c5c478098b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` make_engine_with_allocator(
		cl_device_id device,
		cl_context context,
		const :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`& alloc,
		const std::vector<uint8_t>& cache_blob
		)

Constructs an engine from an OpenCL device, an OpenCL context, an allocator, and a serialized engine cache blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- device

		- A valid OpenCL device to construct the engine

	*
		- context

		- A valid OpenCL context to construct the engine

	*
		- alloc

		- An allocator to associate with the engine

	*
		- cache_blob

		- Cache blob serialized beforehand



.. rubric:: Returns:

Created engine

.. index:: pair: function; execute
.. _doxid-namespacednnl_1_1graph_1_1ocl__interop_1a8b1d57febf09dc0621d7aa2a8dc13035:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cl_event execute(
		:ref:`compiled_partition<doxid-classdnnl_1_1graph_1_1compiled__partition>`& c_partition,
		:ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& inputs,
		std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& outputs,
		const std::vector<cl_event>& deps = {}
		)

Executes a compiled partition in a specified stream and returns a OpenCL event.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- c_partition

		- Compiled partition to execute.

	*
		- astream

		- Stream object to run over

	*
		- inputs

		- Arguments map.

	*
		- outputs

		- Arguments map.

	*
		- deps

		- Optional vector with ``cl_event`` dependencies.



.. rubric:: Returns:

Output event.

