.. index:: pair: namespace; dnnl::ze_interop
.. _doxid-namespacednnl_1_1ze__interop:

namespace dnnl::ze_interop
==========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Level Zero interoperability namespace. :ref:`More...<details-namespacednnl_1_1ze__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace ze_interop {

	// global functions

	std::vector<uint8_t> :ref:`get_engine_cache_blob_id<doxid-namespacednnl_1_1ze__interop_1a6ca2d4fd3930a446df7d6d845aca5b9f>`(
		ze_driver_handle_t driver,
		ze_device_handle_t device
		);

	std::vector<uint8_t> :ref:`get_engine_cache_blob<doxid-namespacednnl_1_1ze__interop_1ad3ed7a97985fdfa5dc6ce6aeecb7f9af>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);

	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`make_engine<doxid-namespacednnl_1_1ze__interop_1a30b006025a67f34ae4b183882bd26094>`(
		ze_driver_handle_t driver,
		ze_device_handle_t device,
		ze_context_handle_t context,
		const std::vector<uint8_t>& cache_blob
		);

	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`make_engine<doxid-namespacednnl_1_1ze__interop_1a536a52755503c79316d4bebb018b6394>`(
		ze_driver_handle_t adriver,
		ze_device_handle_t adevice,
		ze_context_handle_t acontext
		);

	ze_context_handle_t :ref:`get_context<doxid-namespacednnl_1_1ze__interop_1af1b31375baf405c74d4d4e223ed828af>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);
	ze_device_handle_t :ref:`get_device<doxid-namespacednnl_1_1ze__interop_1a797d4b8181489c3da6781d2a7c90e347>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);
	ze_driver_handle_t :ref:`get_driver<doxid-namespacednnl_1_1ze__interop_1aa47b43958425ae2a696bc999f524fce9>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);

	:ref:`stream<doxid-structdnnl_1_1stream>` :ref:`make_stream<doxid-namespacednnl_1_1ze__interop_1ab290ce4074af3dcee977536d9424f25f>`(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		ze_command_list_handle_t alist,
		bool aprofiling = false
		);

	ze_command_list_handle_t :ref:`get_list<doxid-namespacednnl_1_1ze__interop_1a02cb113e0f7223ce995b6b8c3391c34e>`(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream);

	:ref:`memory<doxid-structdnnl_1_1memory>` :ref:`make_memory<doxid-namespacednnl_1_1ze__interop_1a972e58fed45d861a1412c4256561b17e>`(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		std::vector<void*> handles = {}
		);

	:ref:`memory<doxid-structdnnl_1_1memory>` :ref:`make_memory<doxid-namespacednnl_1_1ze__interop_1a43fd01076ba78315051dde2370f7feff>`(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		void* handle
		);

	ze_event_handle_t :ref:`execute<doxid-namespacednnl_1_1ze__interop_1a99e5491f5b45c2c44a1fab29d641400a>`(
		const :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`& aprimitive,
		const :ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args,
		const std::vector<ze_event_handle_t>& deps = {}
		);

	} // namespace ze_interop
.. _details-namespacednnl_1_1ze__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Level Zero interoperability namespace.

Global Functions
----------------

.. index:: pair: function; get_engine_cache_blob_id
.. _doxid-namespacednnl_1_1ze__interop_1a6ca2d4fd3930a446df7d6d845aca5b9f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<uint8_t> get_engine_cache_blob_id(
		ze_driver_handle_t driver,
		ze_device_handle_t device
		)

Returns the cache blob ID of the Level Zero device.

.. warning:: 

   This API is intended to be used with :ref:`dnnl::ze_interop::get_engine_cache_blob(const engine &) <doxid-namespacednnl_1_1ze__interop_1ad3ed7a97985fdfa5dc6ce6aeecb7f9af>` and :ref:`dnnl::ze_interop::make_engine(ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t, const std::vector\<uint8_t> &) <doxid-namespacednnl_1_1ze__interop_1a30b006025a67f34ae4b183882bd26094>`. The returned cache blob ID can only be used as an ID of the cache blob returned by :ref:`dnnl::ze_interop::get_engine_cache_blob(const engine &) <doxid-namespacednnl_1_1ze__interop_1ad3ed7a97985fdfa5dc6ce6aeecb7f9af>`.
   
   

.. note:: 

   The cache blob ID can be empty (``size`` will be 0 and ``cache_blob_id`` will be nullptr) if oneDNN doesn't have anything to put in the cache blob. (:ref:`dnnl_ze_interop_engine_get_cache_blob <doxid-group__dnnl__api__ze__interop_1gae48eaec22ed32acb6ff38e0bc0d7b8e5>` will return an empty cache blob).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- driver

		- A Level Zero driver.

	*
		- device

		- A Level Zero device.



.. rubric:: Returns:

A vector containing the cache blob ID.

.. index:: pair: function; get_engine_cache_blob
.. _doxid-namespacednnl_1_1ze__interop_1ad3ed7a97985fdfa5dc6ce6aeecb7f9af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<uint8_t> get_engine_cache_blob(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns a cache blob for the engine.

.. note:: 

   The cache blob vector can be empty if oneDNN doesn't have anything to put in the cache blob. It's the user's responsibility to check whether it's empty prior to passing it to :ref:`dnnl::ze_interop::make_engine(ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t, const std::vector\<uint8_t> &) <doxid-namespacednnl_1_1ze__interop_1a30b006025a67f34ae4b183882bd26094>`



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to query for the cache blob.



.. rubric:: Returns:

Vector containing the cache blob.

.. index:: pair: function; make_engine
.. _doxid-namespacednnl_1_1ze__interop_1a30b006025a67f34ae4b183882bd26094:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` make_engine(
		ze_driver_handle_t driver,
		ze_device_handle_t device,
		ze_context_handle_t context,
		const std::vector<uint8_t>& cache_blob
		)

Constructs an engine from the given cache blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- driver

		- The Level Zero driver that this engine will encapsulate.

	*
		- device

		- The Level Zero device that this engine will encapsulate.

	*
		- context

		- The Level Zero context (containing the device) that this engine will use for all operations.

	*
		- cache_blob

		- Cache blob.



.. rubric:: Returns:

An engine.

.. index:: pair: function; make_engine
.. _doxid-namespacednnl_1_1ze__interop_1a536a52755503c79316d4bebb018b6394:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` make_engine(
		ze_driver_handle_t adriver,
		ze_device_handle_t adevice,
		ze_context_handle_t acontext
		)

Constructs an engine from Level Zero device and context objects.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- adriver

		- Level Zero driver.

	*
		- adevice

		- Level Zero device.

	*
		- acontext

		- Level Zero context.



.. rubric:: Returns:

Created engine.

.. index:: pair: function; get_context
.. _doxid-namespacednnl_1_1ze__interop_1af1b31375baf405c74d4d4e223ed828af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ze_context_handle_t get_context(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns the Level Zero context associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to query.



.. rubric:: Returns:

The underlying Level Zero context of the engine.

.. index:: pair: function; get_device
.. _doxid-namespacednnl_1_1ze__interop_1a797d4b8181489c3da6781d2a7c90e347:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ze_device_handle_t get_device(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns the Level Zero device associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to query.



.. rubric:: Returns:

The underlying Level Zero device of the engine.

.. index:: pair: function; get_driver
.. _doxid-namespacednnl_1_1ze__interop_1aa47b43958425ae2a696bc999f524fce9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ze_driver_handle_t get_driver(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns the Level Zero driver associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to query.



.. rubric:: Returns:

The underlying Level Zero driver of the engine.

.. index:: pair: function; make_stream
.. _doxid-namespacednnl_1_1ze__interop_1ab290ce4074af3dcee977536d9424f25f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`stream<doxid-structdnnl_1_1stream>` make_stream(
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		ze_command_list_handle_t alist,
		bool aprofiling = false
		)

Creates an execution stream for a given engine associated with a Level Zero command list.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine object to use for the stream.

	*
		- alist

		- Level Zero immediate command list to use for the stream.

	*
		- aprofiling

		- Flag enabling GPU kernel profiling.



.. rubric:: Returns:

An execution stream.

.. index:: pair: function; get_list
.. _doxid-namespacednnl_1_1ze__interop_1a02cb113e0f7223ce995b6b8c3391c34e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ze_command_list_handle_t get_list(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream)

Returns the Level Zero immediate command list associated with an execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- Execution stream to query.



.. rubric:: Returns:

Level Zero immediate command list object.

.. index:: pair: function; make_memory
.. _doxid-namespacednnl_1_1ze__interop_1a972e58fed45d861a1412c4256561b17e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory<doxid-structdnnl_1_1memory>` make_memory(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		std::vector<void*> handles = {}
		)

Creates a memory object with multiple handles.

If the ``handles`` vector is not provided the library will allocate all buffers as if all handles have the special value DNNL_MEMORY_ALLOCATE.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Memory descriptor.

	*
		- aengine

		- Engine to use.

	*
		- handles

		- 
		  Handles of the memory buffers to use as underlying storages. For each element of the ``handles`` array the following applies:
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

Created memory object.

.. index:: pair: function; make_memory
.. _doxid-namespacednnl_1_1ze__interop_1a43fd01076ba78315051dde2370f7feff:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory<doxid-structdnnl_1_1memory>` make_memory(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		void* handle
		)

Creates a memory object.

Unless ``handle`` is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if:

* dnnl::ze_interop::set_mem_object() has been called.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Memory descriptor.

	*
		- aengine

		- Engine to use.

	*
		- handle

		- 
		  Handle of the memory buffer to use as an underlying storage.
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

Created memory object.

.. index:: pair: function; execute
.. _doxid-namespacednnl_1_1ze__interop_1a99e5491f5b45c2c44a1fab29d641400a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	ze_event_handle_t execute(
		const :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`& aprimitive,
		const :ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args,
		const std::vector<ze_event_handle_t>& deps = {}
		)

Executes computations specified by the primitive in a specified stream and returns a Level Zero event.

Arguments are passed via an arguments map containing <index, memory object> pairs. The index must be one of the ``DNNL_ARG_*`` values such as ``DNNL_ARG_SRC``, and the memory must have a memory descriptor matching the one returned by :ref:`dnnl::primitive_desc::query_md <doxid-structdnnl_1_1primitive__desc__base_1a35d24b553ba6aa807516e9470fdd7d16>` (:ref:`query::exec_arg_md <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1ad531896cf1d66c4832790f428623f164>`, index) unless using dynamic shapes (see :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aprimitive

		- Primitive to execute.

	*
		- astream

		- Stream object. The stream must belong to the same engine as the primitive.

	*
		- args

		- Arguments map.

	*
		- deps

		- Optional vector with ``ze_event_handle_t`` dependencies.



.. rubric:: Returns:

Output event.

