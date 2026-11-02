.. index:: pair: group; OpenCL interoperability API API
.. _doxid-group__dnnl__graph__api__ocl__interop:

OpenCL interoperability API API
===============================

.. toctree::
	:hidden:

	namespace_dnnl_graph_ocl_interop.rst

Overview
~~~~~~~~

extensions to interact with the underlying OpenCL run-time. :ref:`More...<details-group__dnnl__graph__api__ocl__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`dnnl::graph::ocl_interop<doxid-namespacednnl_1_1graph_1_1ocl__interop>`;

	// typedefs

	typedef void* (*:ref:`dnnl_graph_ocl_allocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga2900b26adec541b7577667ad3b55fa4d>`)(
		size_t size,
		size_t alignment,
		cl_device_id device,
		cl_context context
		);

	typedef void (*:ref:`dnnl_graph_ocl_deallocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga93912e04c48608be40c1c656cc721ac9>`)(
		void *buf,
		cl_device_id device,
		cl_context context,
		cl_event event
		);

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_ocl_interop_allocator_create<doxid-group__dnnl__graph__api__ocl__interop_1ga23e311433c1e0b5bf4b63d84bad3d4d3>`(
		:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`* allocator,
		:ref:`dnnl_graph_ocl_allocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga2900b26adec541b7577667ad3b55fa4d>` ocl_malloc,
		:ref:`dnnl_graph_ocl_deallocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga93912e04c48608be40c1c656cc721ac9>` ocl_free
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_ocl_interop_make_engine_with_allocator<doxid-group__dnnl__graph__api__ocl__interop_1ga1286b7d76d81ded6ac900bbb853b44f7>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_ocl_interop_make_engine_from_cache_blob_with_allocator<doxid-group__dnnl__graph__api__ocl__interop_1ga5fc1536f94c4b6544d8d69687d483431>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc,
		size_t size,
		const uint8_t* cache_blob
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_ocl_interop_compiled_partition_execute<doxid-group__dnnl__graph__api__ocl__interop_1ga3b551e3717b977fe1874a096cbb0bd20>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		size_t num_inputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* inputs,
		size_t num_outputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* outputs,
		const cl_event* deps,
		int ndeps,
		cl_event* return_event
		);

.. _details-group__dnnl__graph__api__ocl__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

extensions to interact with the underlying OpenCL run-time.

Typedefs
--------

.. index:: pair: typedef; dnnl_graph_ocl_allocate_f
.. _doxid-group__dnnl__graph__api__ocl__interop_1ga2900b26adec541b7577667ad3b55fa4d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef void* (*dnnl_graph_ocl_allocate_f)(
		size_t size,
		size_t alignment,
		cl_device_id device,
		cl_context context
		)

Allocation call-back function interface for OpenCL.

OpenCL allocator should be used for OpenCL GPU runtime. The call-back should return a USM device memory pointer.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- size

		- Memory size in bytes for requested allocation

	*
		- alignment

		- The minimum alignment in bytes for the requested allocation

	*
		- device

		- A valid OpenCL device used to allocate

	*
		- context

		- A valid OpenCL context used to allocate



.. rubric:: Returns:

The memory address of the requested USM allocation.

.. index:: pair: typedef; dnnl_graph_ocl_deallocate_f
.. _doxid-group__dnnl__graph__api__ocl__interop_1ga93912e04c48608be40c1c656cc721ac9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef void (*dnnl_graph_ocl_deallocate_f)(
		void *buf,
		cl_device_id device,
		cl_context context,
		cl_event event
		)

Deallocation call-back function interface for OpenCL.

OpenCL allocator should be used for OpenCL runtime. The call-back should deallocate a USM device memory returned by :ref:`dnnl_graph_ocl_allocate_f <doxid-group__dnnl__graph__api__ocl__interop_1ga2900b26adec541b7577667ad3b55fa4d>`. The event should be completed before deallocate the USM.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- buf

		- The USM allocation to be released

	*
		- device

		- A valid OpenCL device the USM associated with

	*
		- context

		- A valid OpenCL context used to free the USM allocation

	*
		- event

		- A event which the USM deallocation depends on

Global Functions
----------------

.. index:: pair: function; dnnl_graph_ocl_interop_allocator_create
.. _doxid-group__dnnl__graph__api__ocl__interop_1ga23e311433c1e0b5bf4b63d84bad3d4d3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_ocl_interop_allocator_create(
		:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`* allocator,
		:ref:`dnnl_graph_ocl_allocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga2900b26adec541b7577667ad3b55fa4d>` ocl_malloc,
		:ref:`dnnl_graph_ocl_deallocate_f<doxid-group__dnnl__graph__api__ocl__interop_1ga93912e04c48608be40c1c656cc721ac9>` ocl_free
		)

Creates an allocator with the given allocation and deallocation call-back function pointers.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- allocator

		- Output allocator

	*
		- ocl_malloc

		- A pointer to OpenCL malloc function

	*
		- ocl_free

		- A pointer to OpenCL free function



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_ocl_interop_make_engine_with_allocator
.. _doxid-group__dnnl__graph__api__ocl__interop_1ga1286b7d76d81ded6ac900bbb853b44f7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_ocl_interop_make_engine_with_allocator(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc
		)

This API is a supplement for existing oneDNN engine API: dnnl_status_t DNNL_API dnnl_ocl_interop_engine_create( dnnl_engine_t \*engine, cl_device_id device, cl_context context);.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Output engine.

	*
		- device

		- Underlying OpenCL device to use for the engine.

	*
		- context

		- Underlying OpenCL context to use for the engine.

	*
		- alloc

		- Underlying allocator to use for the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_ocl_interop_make_engine_from_cache_blob_with_allocator
.. _doxid-group__dnnl__graph__api__ocl__interop_1ga5fc1536f94c4b6544d8d69687d483431:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_ocl_interop_make_engine_from_cache_blob_with_allocator(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc,
		size_t size,
		const uint8_t* cache_blob
		)

This API is a supplement for existing oneDNN engine API: dnnl_status_t DNNL_API dnnl_ocl_interop_engine_create_from_cache_blob( dnnl_engine_t \*engine, cl_device_id device, cl_context context, size_t size, const uint8_t \*cache_blob);.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Output engine.

	*
		- device

		- The OpenCL device that this engine will encapsulate.

	*
		- context

		- The OpenCL context (containing the device) that this engine will use for all operations.

	*
		- alloc

		- Underlying allocator to use for the engine.

	*
		- size

		- Size of the cache blob in bytes.

	*
		- cache_blob

		- Cache blob of size ``size``.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_ocl_interop_compiled_partition_execute
.. _doxid-group__dnnl__graph__api__ocl__interop_1ga3b551e3717b977fe1874a096cbb0bd20:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_ocl_interop_compiled_partition_execute(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		size_t num_inputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* inputs,
		size_t num_outputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* outputs,
		const cl_event* deps,
		int ndeps,
		cl_event* return_event
		)

Execute a compiled partition with OpenCL runtime.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- compiled_partition

		- The handle of target compiled_partition.

	*
		- stream

		- The stream used for execution

	*
		- num_inputs

		- The number of input tensors

	*
		- inputs

		- A list of input tensors

	*
		- num_outputs

		- The number of output tensors

	*
		- outputs

		- A non-empty list of output tensors

	*
		- deps

		- Optional handle of list with ``cl_event`` dependencies.

	*
		- ndeps

		- Number of dependencies.

	*
		- return_event

		- The handle of cl_event.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

