.. index:: pair: group; OpenCL interoperability API
.. _doxid-group__dnnl__api__ocl__interop:

OpenCL interoperability API
===========================

.. toctree::
	:hidden:

	namespace_dnnl_ocl_interop.rst
	enum_dnnl_ocl_interop_memory_kind_t.rst

Overview
~~~~~~~~

API extensions to interact with the underlying OpenCL run-time. :ref:`More...<details-group__dnnl__api__ocl__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`dnnl::ocl_interop<doxid-namespacednnl_1_1ocl__interop>`;

	// enums

	enum :ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_create<doxid-group__dnnl__api__ocl__interop_1gad5b8aba7d6108ba727505d0db5062342>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>` memory_kind,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_create_v2<doxid-group__dnnl__api__ocl__interop_1ga99ab37a67dc9b4f12cb952a95dd63784>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>` memory_kind,
		int nhandles,
		void** handles
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_get_memory_kind<doxid-group__dnnl__api__ocl__interop_1gaa6c00a54ba3ca30d00d9e1c43e9b4bc2>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>`* memory_kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_get_mem_object<doxid-group__dnnl__api__ocl__interop_1ga383b09734d764bb45872b2c65f7dad70>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		cl_mem* mem_object
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_memory_set_mem_object<doxid-group__dnnl__api__ocl__interop_1ga493946609c6bae83329241c4950edbce>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		cl_mem mem_object
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_get_cache_blob_id<doxid-group__dnnl__api__ocl__interop_1gad1e18db981b46c04640dde395f75845c>`(
		cl_device_id device,
		size_t* size,
		uint8_t* cache_blob_id
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_get_cache_blob<doxid-group__dnnl__api__ocl__interop_1gae29834208ef008eb43ab8f82985999f5>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		size_t* size,
		uint8_t* cache_blob
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_create_from_cache_blob<doxid-group__dnnl__api__ocl__interop_1gaf4d8ed8673cf2d90a326cb0e66a41ccd>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context,
		size_t size,
		const uint8_t* cache_blob
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_create<doxid-group__dnnl__api__ocl__interop_1ga52edd1810d72a2a08a881b122c7ada70>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_engine_get_context<doxid-group__dnnl__api__ocl__interop_1ga6be452e1d11ad63d1f20a072258547c8>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_context* context
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_get_device<doxid-group__dnnl__api__ocl__interop_1gafd0a653afb5a16d4d1fc71cd0615e44a>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_device_id* device
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_stream_create<doxid-group__dnnl__api__ocl__interop_1ga9a9007c6661472d701b2bbfb43ddf07c>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_command_queue queue
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_stream_get_command_queue<doxid-group__dnnl__api__ocl__interop_1ga0290e83f9217e83eba910454348c0819>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		cl_command_queue* queue
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ocl_interop_primitive_execute<doxid-group__dnnl__api__ocl__interop_1ga95b6ba55db1e53a8b530ef38b09d1953>`(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args,
		const cl_event* deps,
		int ndeps,
		cl_event* return_event
		);

.. _details-group__dnnl__api__ocl__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

API extensions to interact with the underlying OpenCL run-time.



.. rubric:: See also:

:ref:`OpenCL Interoperability <doxid-dev_guide_opencl_interoperability>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_ocl_interop_memory_create
.. _doxid-group__dnnl__api__ocl__interop_1gad5b8aba7d6108ba727505d0db5062342:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_memory_create(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>` memory_kind,
		void* handle
		)

Creates a memory object.

Unless ``handle`` is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if:

* :ref:`dnnl_memory_set_data_handle() <doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>` has been called, if ``memory_kind`` is equal to dnnl_ocl_interop_usm, or

* :ref:`dnnl_ocl_interop_memory_set_mem_object() <doxid-group__dnnl__api__ocl__interop_1ga493946609c6bae83329241c4950edbce>` has been called, if ``memory_kind`` is equal to dnnl_ocl_interop_buffer.



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
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` to be equal to dnnl_ocl_interop_usm.
		  
		  * An OpenCL buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` be equal to be equal to dnnl_ocl_interop_buffer.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer that corresponds to the memory allocation kind ``memory_kind`` for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_memory_create_v2
.. _doxid-group__dnnl__api__ocl__interop_1ga99ab37a67dc9b4f12cb952a95dd63784:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_memory_create_v2(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>` memory_kind,
		int nhandles,
		void** handles
		)

Creates a memory object with multiple handles.



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

		- Memory allocation kind to specify the type of handles.

	*
		- nhandles

		- Number of handles.

	*
		- handles

		- 
		  Handles of the memory buffers to use as underlying storages. For each element of the ``handles`` array the following applies:
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` to be equal to dnnl_ocl_interop_usm.
		  
		  * An OpenCL buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` be equal to be equal to dnnl_ocl_interop_buffer.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer that corresponds to the memory allocation kind ``memory_kind`` for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_memory_get_memory_kind
.. _doxid-group__dnnl__api__ocl__interop_1gaa6c00a54ba3ca30d00d9e1c43e9b4bc2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_memory_get_memory_kind(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>`* memory_kind
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

.. index:: pair: function; dnnl_ocl_interop_memory_get_mem_object
.. _doxid-group__dnnl__api__ocl__interop_1ga383b09734d764bb45872b2c65f7dad70:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_memory_get_mem_object(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		cl_mem* mem_object
		)

Returns an OpenCL memory object associated with a memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- mem_object

		- Output OpenCL memory object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_memory_set_mem_object
.. _doxid-group__dnnl__api__ocl__interop_1ga493946609c6bae83329241c4950edbce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_memory_set_mem_object(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		cl_mem mem_object
		)

Sets OpenCL memory object associated with a memory object.

For behavioral details, see :ref:`dnnl_memory_set_data_handle() <doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- mem_object

		- OpenCL memory object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_engine_get_cache_blob_id
.. _doxid-group__dnnl__api__ocl__interop_1gad1e18db981b46c04640dde395f75845c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_engine_get_cache_blob_id(
		cl_device_id device,
		size_t* size,
		uint8_t* cache_blob_id
		)

Retrieves a cache blob ID for the OpenCL device.

.. warning:: 

   This API is intended to be used with :ref:`dnnl_ocl_interop_engine_get_cache_blob() <doxid-group__dnnl__api__ocl__interop_1gae29834208ef008eb43ab8f82985999f5>` and :ref:`dnnl_ocl_interop_engine_create_from_cache_blob() <doxid-group__dnnl__api__ocl__interop_1gaf4d8ed8673cf2d90a326cb0e66a41ccd>`. The returned cache blob ID can only be used as an ID of the cache blob returned by :ref:`dnnl_ocl_interop_engine_get_cache_blob() <doxid-group__dnnl__api__ocl__interop_1gae29834208ef008eb43ab8f82985999f5>`.
   
   

.. note:: 

   The cache blob ID can be empty (``size`` will be 0 and ``cache_blob_id`` will be nullptr) if oneDNN doesn't have anything to put in the cache blob. (:ref:`dnnl_ocl_interop_engine_get_cache_blob <doxid-group__dnnl__api__ocl__interop_1gae29834208ef008eb43ab8f82985999f5>` will return an empty cache blob).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- device

		- An OpenCL device.

	*
		- size

		- Size of the cache blob ID in bytes.

	*
		- cache_blob_id

		- Cache blob id of size ``size``. If the ``cache_blob_id`` is nullptr then the size of the cache blob ID is returned in ``size``.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_engine_get_cache_blob
.. _doxid-group__dnnl__api__ocl__interop_1gae29834208ef008eb43ab8f82985999f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_engine_get_cache_blob(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		size_t* size,
		uint8_t* cache_blob
		)

Retrieves a cache blob associated with the given engine.

.. note:: 

   The cache blob can be empty (``size`` will be 0 and ``cache_blob`` will be nullptr) if oneDNN doesn't have anything to put in the cache blob. It's the user's responsibility to check whether it's empty prior to passing it to :ref:`dnnl_ocl_interop_engine_create_from_cache_blob() <doxid-group__dnnl__api__ocl__interop_1gaf4d8ed8673cf2d90a326cb0e66a41ccd>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query for the cache blob.

	*
		- size

		- Size of the cache blob in bytes.

	*
		- cache_blob

		- Cache blob of size ``size``. If the ``cache_blob`` is nullptr then the size of the cache blob is returned in ``size``.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_engine_create_from_cache_blob
.. _doxid-group__dnnl__api__ocl__interop_1gaf4d8ed8673cf2d90a326cb0e66a41ccd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_engine_create_from_cache_blob(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context,
		size_t size,
		const uint8_t* cache_blob
		)

Creates an engine from the given cache blob.



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
		- size

		- Size of the cache blob in bytes.

	*
		- cache_blob

		- Cache blob of size ``size``.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_engine_create
.. _doxid-group__dnnl__api__ocl__interop_1ga52edd1810d72a2a08a881b122c7ada70:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_engine_create(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		cl_device_id device,
		cl_context context
		)

Creates an engine associated with an OpenCL device and an OpenCL context.



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



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_engine_get_context
.. _doxid-group__dnnl__api__ocl__interop_1ga6be452e1d11ad63d1f20a072258547c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_engine_get_context(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_context* context
		)

Returns the OpenCL context associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query.

	*
		- context

		- Output underlying OpenCL context of the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_get_device
.. _doxid-group__dnnl__api__ocl__interop_1gafd0a653afb5a16d4d1fc71cd0615e44a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_get_device(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_device_id* device
		)

Returns the OpenCL device associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query.

	*
		- device

		- Output underlying OpenCL device of the engine.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_stream_create
.. _doxid-group__dnnl__api__ocl__interop_1ga9a9007c6661472d701b2bbfb43ddf07c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_stream_create(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		cl_command_queue queue
		)

Creates an execution stream for a given engine associated with an OpenCL command queue.



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

		- OpenCL command queue to use.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_stream_get_command_queue
.. _doxid-group__dnnl__api__ocl__interop_1ga0290e83f9217e83eba910454348c0819:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_stream_get_command_queue(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		cl_command_queue* queue
		)

Returns the OpenCL command queue associated with an execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Execution stream to query.

	*
		- queue

		- Output OpenCL command queue.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ocl_interop_primitive_execute
.. _doxid-group__dnnl__api__ocl__interop_1ga95b6ba55db1e53a8b530ef38b09d1953:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ocl_interop_primitive_execute(
		:ref:`const_dnnl_primitive_t<doxid-group__dnnl__api__primitives__common_1ga3a24919ac3820e4a196bd4e50a0972c5>` primitive,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		int nargs,
		const :ref:`dnnl_exec_arg_t<doxid-structdnnl__exec__arg__t>`* args,
		const cl_event* deps,
		int ndeps,
		cl_event* return_event
		)

Executes computations specified by the primitive in a specified stream and returns an OpenCL event.



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

		- A pointer to a vector of size ``ndeps`` that contains dependencies.

	*
		- ndeps

		- Number of dependencies.

	*
		- return_event

		- Output event. It's the user's responsibility to manage lifetime of the event. Can be NULL. When ``stream`` is in-order NULL will be returned.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

