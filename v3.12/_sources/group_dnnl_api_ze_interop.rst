.. index:: pair: group; Level Zero interoperability API
.. _doxid-group__dnnl__api__ze__interop:

Level Zero interoperability API
===============================

.. toctree::
	:hidden:

	namespace_dnnl_ze_interop.rst

Overview
~~~~~~~~

API extensions to interact with the underlying Level Zero run-time. :ref:`More...<details-group__dnnl__api__ze__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`dnnl::ze_interop<doxid-namespacednnl_1_1ze__interop>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ze_interop_engine_create<doxid-group__dnnl__api__ze__interop_1gacd1128872929a2949781027c9a53bb9d>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		ze_driver_handle_t driver,
		ze_device_handle_t device,
		ze_context_handle_t context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ze_interop_engine_get_context<doxid-group__dnnl__api__ze__interop_1gaac241e8e0fd2821810c37b1654f4b860>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		ze_context_handle_t* context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ze_interop_engine_get_device<doxid-group__dnnl__api__ze__interop_1ga75268a10b903481ec94b4eb16a2e5843>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		ze_device_handle_t* device
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ze_interop_engine_get_driver<doxid-group__dnnl__api__ze__interop_1gae276cf5c283bfb4faa7e186e76fa466a>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		ze_driver_handle_t* driver
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ze_interop_stream_create<doxid-group__dnnl__api__ze__interop_1ga34230de0437ced14603f85f7901fba93>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		ze_command_list_handle_t list,
		int profiling
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ze_interop_stream_get_list<doxid-group__dnnl__api__ze__interop_1ga410b2e08f2bd94287bcbe08c3778918a>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		ze_command_list_handle_t* list
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ze_interop_memory_create<doxid-group__dnnl__api__ze__interop_1gad39c5ee134fbeadb3f2ea58418106e7e>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		int nhandles,
		void** handles
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ze_interop_primitive_execute<doxid-group__dnnl__api__ze__interop_1ga9d01549ef3799f20eed6109a2792c7e0>`(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args,
		int ndeps,
		const ze_event_handle_t* deps,
		ze_event_handle_t* return_event
		);

.. _details-group__dnnl__api__ze__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

API extensions to interact with the underlying Level Zero run-time.



.. rubric:: See also:

dev_guide_level_zero_interoperability in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_ze_interop_engine_create
.. _doxid-group__dnnl__api__ze__interop_1gacd1128872929a2949781027c9a53bb9d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ze_interop_engine_create(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		ze_driver_handle_t driver,
		ze_device_handle_t device,
		ze_context_handle_t context
		)

Creates an engine associated with a Level Zero device and a Level Zero context.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Output engine.

	*
		- driver

		- Pointer to the Level Zero driver to use for the engine.

	*
		- device

		- Pointer to the Level Zero device to use for the engine.

	*
		- context

		- Pointer to the Level Zero context to use for the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ze_interop_engine_get_context
.. _doxid-group__dnnl__api__ze__interop_1gaac241e8e0fd2821810c37b1654f4b860:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ze_interop_engine_get_context(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		ze_context_handle_t* context
		)

Returns the Level Zero context associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query.

	*
		- context

		- Pointer to the underlying Level Zero context of the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ze_interop_engine_get_device
.. _doxid-group__dnnl__api__ze__interop_1ga75268a10b903481ec94b4eb16a2e5843:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ze_interop_engine_get_device(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		ze_device_handle_t* device
		)

Returns the Level Zero device associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query.

	*
		- device

		- Pointer to the underlying Level Zero device of the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ze_interop_engine_get_driver
.. _doxid-group__dnnl__api__ze__interop_1gae276cf5c283bfb4faa7e186e76fa466a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ze_interop_engine_get_driver(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		ze_driver_handle_t* driver
		)

Returns the Level Zero driver associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query.

	*
		- driver

		- Pointer to the underlying Level Zero driver of the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ze_interop_stream_create
.. _doxid-group__dnnl__api__ze__interop_1ga34230de0437ced14603f85f7901fba93:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ze_interop_stream_create(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		ze_command_list_handle_t list,
		int profiling
		)

Creates an execution stream for a given engine associated with a Level Zero command list.



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
		- list

		- Level Zero command list to use.

	*
		- profiling

		- Flag enabling GPU kernels profiling.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ze_interop_stream_get_list
.. _doxid-group__dnnl__api__ze__interop_1ga410b2e08f2bd94287bcbe08c3778918a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ze_interop_stream_get_list(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		ze_command_list_handle_t* list
		)

Returns the Level Zero command list associated with an execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Execution stream to query.

	*
		- list

		- Output Level Zero command list.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ze_interop_memory_create
.. _doxid-group__dnnl__api__ze__interop_1gad39c5ee134fbeadb3f2ea58418106e7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ze_interop_memory_create(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		int nhandles,
		void** handles
		)

Creates a memory object.

Unless ``handle`` is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if :ref:`dnnl_memory_set_data_handle() <doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>` had been called.



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
		- nhandles

		- Number of handles.

	*
		- handles

		- 
		  Handles of the memory buffers to use as underlying storages.
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ze_interop_primitive_execute
.. _doxid-group__dnnl__api__ze__interop_1ga9d01549ef3799f20eed6109a2792c7e0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ze_interop_primitive_execute(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args,
		int ndeps,
		const ze_event_handle_t* deps,
		ze_event_handle_t* return_event
		)

Executes computations specified by the primitive in a specified stream and returns a Level Zero event.



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
		- ndeps

		- Number of dependencies.

	*
		- deps

		- A pointer to a vector of size ``ndeps`` that contains dependencies.

	*
		- return_event

		- Output event.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

