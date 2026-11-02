.. index:: pair: group; SYCL interoperability API
.. _doxid-group__dnnl__graph__api__sycl__interop:

SYCL interoperability API
=========================

.. toctree::
	:hidden:

	namespace_dnnl_graph_sycl_interop.rst

Overview
~~~~~~~~

API extensions to interact with the underlying SYCL run-time. :ref:`More...<details-group__dnnl__graph__api__sycl__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`dnnl::graph::sycl_interop<doxid-namespacednnl_1_1graph_1_1sycl__interop>`;

	// typedefs

	typedef void* (*:ref:`dnnl_graph_sycl_allocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>`)(
		size_t size,
		size_t alignment,
		const void *dev,
		const void *context
		);

	typedef void (*:ref:`dnnl_graph_sycl_deallocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>`)(
		void *buf,
		const void *dev,
		const void *context,
		void *event
		);

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_sycl_interop_allocator_create<doxid-group__dnnl__graph__api__sycl__interop_1ga06e949434a4fc257e1c89185e97593dc>`(
		:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`* allocator,
		:ref:`dnnl_graph_sycl_allocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>` sycl_malloc,
		:ref:`dnnl_graph_sycl_deallocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>` sycl_free
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_sycl_interop_make_engine_with_allocator<doxid-group__dnnl__graph__api__sycl__interop_1ga84bf2a778aeb99c8134c541ee2b603bd>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		const void* device,
		const void* context,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_sycl_interop_compiled_partition_execute<doxid-group__dnnl__graph__api__sycl__interop_1ga7e51f65c06cd550a282db11ee86b8e47>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		size_t num_inputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* inputs,
		size_t num_outputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* outputs,
		const void* deps,
		void* sycl_event
		);

.. _details-group__dnnl__graph__api__sycl__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

API extensions to interact with the underlying SYCL run-time.

Typedefs
--------

.. index:: pair: typedef; dnnl_graph_sycl_allocate_f
.. _doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef void* (*dnnl_graph_sycl_allocate_f)(
		size_t size,
		size_t alignment,
		const void *dev,
		const void *context
		)

Allocation call-back function interface for SYCL.

SYCL allocator should be used for SYCL runtime and host allocator should be used for non-SYCL. The call-back should return a USM device memory pointer.

.. index:: pair: typedef; dnnl_graph_sycl_deallocate_f
.. _doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef void (*dnnl_graph_sycl_deallocate_f)(
		void *buf,
		const void *dev,
		const void *context,
		void *event
		)

Deallocation call-back function interface for SYCL.

SYCL allocator should be used for SYCL runtime and host allocator should be used for non-SYCL. The call-back should deallocate a USM device memory returned by :ref:`dnnl_graph_sycl_allocate_f <doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>`.

Global Functions
----------------

.. index:: pair: function; dnnl_graph_sycl_interop_allocator_create
.. _doxid-group__dnnl__graph__api__sycl__interop_1ga06e949434a4fc257e1c89185e97593dc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_sycl_interop_allocator_create(
		:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`* allocator,
		:ref:`dnnl_graph_sycl_allocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>` sycl_malloc,
		:ref:`dnnl_graph_sycl_deallocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>` sycl_free
		)

Creates an allocator with the given allocation and deallocation call-back function pointers.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- allocator

		- Output allocator

	*
		- sycl_malloc

		- A pointer to SYCL malloc function

	*
		- sycl_free

		- A pointer to SYCL free function



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_sycl_interop_make_engine_with_allocator
.. _doxid-group__dnnl__graph__api__sycl__interop_1ga84bf2a778aeb99c8134c541ee2b603bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_sycl_interop_make_engine_with_allocator(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		const void* device,
		const void* context,
		:ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>` alloc
		)

This API is a supplement for existing onednn engine API.

.. index:: pair: function; dnnl_graph_sycl_interop_compiled_partition_execute
.. _doxid-group__dnnl__graph__api__sycl__interop_1ga7e51f65c06cd550a282db11ee86b8e47:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_sycl_interop_compiled_partition_execute(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		size_t num_inputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* inputs,
		size_t num_outputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* outputs,
		const void* deps,
		void* sycl_event
		)

Execute a compiled partition with sycl runtime.



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

		- Optional handle of list with ``sycl::event`` dependencies.

	*
		- sycl_event

		- The handle of sycl event.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

