.. index:: pair: group; SYCL interoperability API
.. _doxid-group__dnnl__api__sycl__interop:

SYCL interoperability API
=========================

.. toctree::
	:hidden:

	namespace_dnnl_sycl_interop.rst
	enum_dnnl_sycl_interop_memory_kind_t.rst

Overview
~~~~~~~~

API extensions to interact with the underlying SYCL run-time. :ref:`More...<details-group__dnnl__api__sycl__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`dnnl::sycl_interop<doxid-namespacednnl_1_1sycl__interop>`;

	// enums

	enum :ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_engine_create<doxid-group__dnnl__api__sycl__interop_1ga7d768ee527493380e13fdf2983b32b70>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		const void* device,
		const void* context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_engine_get_context<doxid-group__dnnl__api__sycl__interop_1ga23b777c4d60c3dd9b542126973cb69a5>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void** context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_engine_get_device<doxid-group__dnnl__api__sycl__interop_1gaf6a88b22743cc5ca54ad46ee88fbb71e>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void** device
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_memory_create<doxid-group__dnnl__api__sycl__interop_1gafe0b9a934268c1954b87475d376d600e>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>` memory_kind,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_memory_get_memory_kind<doxid-group__dnnl__api__sycl__interop_1ga3c24bba041823efb72fd4ce003a4436c>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>`* memory_kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_memory_set_buffer<doxid-group__dnnl__api__sycl__interop_1ga62c1cb33d766f2035d83b7010db7adf9>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* buffer
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_stream_create<doxid-group__dnnl__api__sycl__interop_1ga64dc62b1586d688afcd110840e570cd5>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* queue
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_stream_get_queue<doxid-group__dnnl__api__sycl__interop_1gab279fc3922a8b4ab59d4b328f1610172>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		void** queue
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sycl_interop_primitive_execute<doxid-group__dnnl__api__sycl__interop_1ga49aea2229b2e3afcd66e31ef76fcbe64>`(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args,
		const void* deps,
		void* return_event
		);

.. _details-group__dnnl__api__sycl__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

API extensions to interact with the underlying SYCL run-time.



.. rubric:: See also:

:ref:`DPC++ Interoperability <doxid-dev_guide_dpcpp_interoperability>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_sycl_interop_engine_create
.. _doxid-group__dnnl__api__sycl__interop_1ga7d768ee527493380e13fdf2983b32b70:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_engine_create(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		const void* device,
		const void* context
		)

Creates an engine associated with a SYCL device and a SYCL context.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Output engine.

	*
		- device

		- Pointer to the SYCL device to use for the engine.

	*
		- context

		- Pointer to the SYCL context to use for the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sycl_interop_engine_get_context
.. _doxid-group__dnnl__api__sycl__interop_1ga23b777c4d60c3dd9b542126973cb69a5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_engine_get_context(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void** context
		)

Returns the SYCL context associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query.

	*
		- context

		- Pointer to the underlying SYCL context of the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sycl_interop_engine_get_device
.. _doxid-group__dnnl__api__sycl__interop_1gaf6a88b22743cc5ca54ad46ee88fbb71e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_engine_get_device(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void** device
		)

Returns the SYCL device associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query.

	*
		- device

		- Pointer to the underlying SYCL device of the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sycl_interop_memory_create
.. _doxid-group__dnnl__api__sycl__interop_1gafe0b9a934268c1954b87475d376d600e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_memory_create(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>` memory_kind,
		void* handle
		)

Creates a memory object.

Unless ``handle`` is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if:

* :ref:`dnnl_memory_set_data_handle() <doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>` had been called, if ``memory_kind`` is equal to dnnl_sycl_interop_usm, or

* :ref:`dnnl_sycl_interop_memory_set_buffer() <doxid-group__dnnl__api__sycl__interop_1ga62c1cb33d766f2035d83b7010db7adf9>` has been called, if ``memory_kind`` is equal to dnnl_sycl_interop_buffer.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Output memory object.

	*
		- memory_desc

		- Memory descriptor.

	*
		- engine

		- Engine to use.

	*
		- memory_kind

		- Memory allocation kind to specify the type of handle.

	*
		- handle

		- 
		  Handle of the memory buffer to use as an underlying storage.
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` to be equal to dnnl_sycl_interop_usm.
		  
		  * A pointer to SYCL buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` be equal to be equal to dnnl_sycl_interop_buffer.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer that corresponds to the memory allocation kind ``memory_kind`` for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sycl_interop_memory_get_memory_kind
.. _doxid-group__dnnl__api__sycl__interop_1ga3c24bba041823efb72fd4ce003a4436c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_memory_get_memory_kind(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>`* memory_kind
		)

Returns the memory allocation kind associated with a memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory to query.

	*
		- memory_kind

		- Output underlying memory allocation kind of the memory object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sycl_interop_memory_set_buffer
.. _doxid-group__dnnl__api__sycl__interop_1ga62c1cb33d766f2035d83b7010db7adf9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_memory_set_buffer(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* buffer
		)

Sets a SYCL buffer for a memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- buffer

		- SYCL buffer to be set in the memory object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sycl_interop_stream_create
.. _doxid-group__dnnl__api__sycl__interop_1ga64dc62b1586d688afcd110840e570cd5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_stream_create(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* queue
		)

Creates an execution stream for a given engine associated with a SYCL queue.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Output execution stream.

	*
		- engine

		- Engine to create the execution stream on.

	*
		- queue

		- SYCL queue to use.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sycl_interop_stream_get_queue
.. _doxid-group__dnnl__api__sycl__interop_1gab279fc3922a8b4ab59d4b328f1610172:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_stream_get_queue(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		void** queue
		)

Returns the SYCL queue associated with an execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Execution stream to query.

	*
		- queue

		- Output SYCL command queue.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_sycl_interop_primitive_execute
.. _doxid-group__dnnl__api__sycl__interop_1ga49aea2229b2e3afcd66e31ef76fcbe64:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sycl_interop_primitive_execute(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args,
		const void* deps,
		void* return_event
		)

Executes computations specified by the primitive in a specified stream and returns a SYCL event.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive

		- Primitive to execute.

	*
		- stream

		- Stream to use.

	*
		- nargs

		- Number of arguments.

	*
		- args

		- Array of arguments. Each argument is an <index, :ref:`dnnl_memory_t <doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`> pair. The index is one of the ``DNNL_ARG_*`` values such as ``DNNL_ARG_SRC``. Unless runtime shapes are used (see :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`), the memory object must have the same memory descriptor as that returned by :ref:`dnnl_primitive_desc_query_md <doxid-group__dnnl__api__primitives__common_1ga22d7722f49cf30215fa4354429106873>` (:ref:`dnnl_query_exec_arg_md <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ac7ecf09260d89d54ddd7f35c51a244da>`, index).

	*
		- deps

		- A pointer to std::vector<sycl::event> that contains dependencies.

	*
		- return_event

		- Output event.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

