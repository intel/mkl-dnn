.. index:: pair: namespace; dnnl::ocl_interop
.. _doxid-namespacednnl_1_1ocl__interop:

namespace dnnl::ocl_interop
===========================

.. toctree::
	:hidden:

	enum_dnnl_ocl_interop_memory_kind.rst

Overview
~~~~~~~~

OpenCL interoperability namespace. :ref:`More...<details-namespacednnl_1_1ocl__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace ocl_interop {

	// enums

	enum :ref:`memory_kind<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481>`;

	// global functions

	:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>` :ref:`convert_to_c<doxid-namespacednnl_1_1ocl__interop_1a5dc60a792c457e048fab0b88e69c384f>`(:ref:`memory_kind<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481>` akind);
	std::vector<uint8_t> :ref:`get_engine_cache_blob_id<doxid-namespacednnl_1_1ocl__interop_1a969a55a4a7d84dee549ebd2c3bcc2518>`(cl_device_id device);
	std::vector<uint8_t> :ref:`get_engine_cache_blob<doxid-namespacednnl_1_1ocl__interop_1a55f93340fcf71df592d7a2c903513823>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);

	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`make_engine<doxid-namespacednnl_1_1ocl__interop_1a4de69784af3621cf7b83041a2a9de071>`(
		cl_device_id device,
		cl_context context,
		const std::vector<uint8_t>& cache_blob
		);

	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`make_engine<doxid-namespacednnl_1_1ocl__interop_1aaa1b1a194ca813f3db12effd29a359d7>`(cl_device_id device, cl_context context);
	cl_context :ref:`get_context<doxid-namespacednnl_1_1ocl__interop_1a248df8106d035e5a7e1ac5fd196c93c3>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);
	cl_device_id :ref:`get_device<doxid-namespacednnl_1_1ocl__interop_1a37ef1ccb75d09063ed049076fb23b927>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);
	:ref:`stream<doxid-structdnnl_1_1stream>` :ref:`make_stream<doxid-namespacednnl_1_1ocl__interop_1ad29aa52fd99fb371018ae6761b0bc8fa>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, cl_command_queue queue);
	cl_command_queue :ref:`get_command_queue<doxid-namespacednnl_1_1ocl__interop_1a14281f69db5178363ff0c971510d0452>`(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream);
	cl_mem :ref:`get_mem_object<doxid-namespacednnl_1_1ocl__interop_1ac117d62fba9de220fe53b0eedb9671f9>`(const :ref:`memory<doxid-structdnnl_1_1memory>`& amemory);
	void :ref:`set_mem_object<doxid-namespacednnl_1_1ocl__interop_1abe99da7a9ae3286ba6a950921a07eaf0>`(:ref:`memory<doxid-structdnnl_1_1memory>`& amemory, cl_mem mem_object);
	:ref:`memory_kind<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481>` :ref:`get_memory_kind<doxid-namespacednnl_1_1ocl__interop_1aa94bfc5feb0de9752012d60f4de1ad2f>`(const :ref:`memory<doxid-structdnnl_1_1memory>`& amemory);

	:ref:`memory<doxid-structdnnl_1_1memory>` :ref:`make_memory<doxid-namespacednnl_1_1ocl__interop_1acfb8e3d4cdcff9244e9b530b3f4c4a9d>`(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`memory_kind<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481>` kind,
		void* handle = :ref:`DNNL_MEMORY_ALLOCATE<doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`
		);

	:ref:`memory<doxid-structdnnl_1_1memory>` :ref:`make_memory<doxid-namespacednnl_1_1ocl__interop_1a8be4eaee886f3d99f154cfe5d2544994>`(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		cl_mem mem_object
		);

	cl_event :ref:`execute<doxid-namespacednnl_1_1ocl__interop_1af585a09a66d7bde78dc7b33557768501>`(
		const :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`& aprimitive,
		const :ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args,
		const std::vector<cl_event>& deps = {}
		);

	} // namespace ocl_interop
.. _details-namespacednnl_1_1ocl__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

OpenCL interoperability namespace.

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-namespacednnl_1_1ocl__interop_1a5dc60a792c457e048fab0b88e69c384f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_ocl_interop_memory_kind_t<doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>` convert_to_c(:ref:`memory_kind<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481>` akind)

Converts a memory allocation kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- C++ API memory allocation kind enum value.



.. rubric:: Returns:

Corresponding C API memory allocation kind enum value.

.. index:: pair: function; get_engine_cache_blob_id
.. _doxid-namespacednnl_1_1ocl__interop_1a969a55a4a7d84dee549ebd2c3bcc2518:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<uint8_t> get_engine_cache_blob_id(cl_device_id device)

Returns the cache blob ID of the OpenCL device.

.. warning:: 

   This API is intended to be used with :ref:`dnnl::ocl_interop::get_engine_cache_blob() <doxid-namespacednnl_1_1ocl__interop_1a55f93340fcf71df592d7a2c903513823>` and :ref:`dnnl::ocl_interop::make_engine(cl_device_id, cl_context, const std::vector\<uint8_t> &) <doxid-namespacednnl_1_1ocl__interop_1a4de69784af3621cf7b83041a2a9de071>`. The returned cache blob ID can only be used as an ID of the cache blob returned by :ref:`dnnl::ocl_interop::get_engine_cache_blob() <doxid-namespacednnl_1_1ocl__interop_1a55f93340fcf71df592d7a2c903513823>`.
   
   

.. note:: 

   The cache blob ID can be empty (``size`` will be 0 and ``cache_blob_id`` will be nullptr) if oneDNN doesn't have anything to put in the cache blob. (:ref:`dnnl_ocl_interop_engine_get_cache_blob <doxid-group__dnnl__api__ocl__interop_1gae29834208ef008eb43ab8f82985999f5>` will return an empty cache blob).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- device

		- An OpenCL device.



.. rubric:: Returns:

A vector containing the cache blob ID.

.. index:: pair: function; get_engine_cache_blob
.. _doxid-namespacednnl_1_1ocl__interop_1a55f93340fcf71df592d7a2c903513823:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<uint8_t> get_engine_cache_blob(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns a cache blob for the engine.

.. note:: 

   The cache blob vector can be empty if oneDNN doesn't have anything to put in the cache blob. It's the user's responsibility to check whether it's empty prior to passing it to :ref:`dnnl::ocl_interop::make_engine(cl_device_id, cl_context, const std::vector\<uint8_t> &) <doxid-namespacednnl_1_1ocl__interop_1a4de69784af3621cf7b83041a2a9de071>`



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to query for the cache blob.



.. rubric:: Returns:

Vector containing the cache blob.

.. index:: pair: function; make_engine
.. _doxid-namespacednnl_1_1ocl__interop_1a4de69784af3621cf7b83041a2a9de071:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` make_engine(
		cl_device_id device,
		cl_context context,
		const std::vector<uint8_t>& cache_blob
		)

Constructs an engine from the given cache blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- device

		- The OpenCL device that this engine will encapsulate.

	*
		- context

		- The OpenCL context (containing the device) that this engine will use for all operations.

	*
		- cache_blob

		- Cache blob.



.. rubric:: Returns:

An engine.

.. index:: pair: function; make_engine
.. _doxid-namespacednnl_1_1ocl__interop_1aaa1b1a194ca813f3db12effd29a359d7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` make_engine(cl_device_id device, cl_context context)

Constructs an engine from OpenCL device and context objects.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- device

		- The OpenCL device that this engine will encapsulate.

	*
		- context

		- The OpenCL context (containing the device) that this engine will use for all operations.



.. rubric:: Returns:

An engine.

.. index:: pair: function; get_context
.. _doxid-namespacednnl_1_1ocl__interop_1a248df8106d035e5a7e1ac5fd196c93c3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cl_context get_context(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns OpenCL context associated with the engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- An engine.



.. rubric:: Returns:

Underlying OpenCL context.

.. index:: pair: function; get_device
.. _doxid-namespacednnl_1_1ocl__interop_1a37ef1ccb75d09063ed049076fb23b927:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cl_device_id get_device(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns OpenCL device associated with the engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- An engine.



.. rubric:: Returns:

Underlying OpenCL device.

.. index:: pair: function; make_stream
.. _doxid-namespacednnl_1_1ocl__interop_1ad29aa52fd99fb371018ae6761b0bc8fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`stream<doxid-structdnnl_1_1stream>` make_stream(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, cl_command_queue queue)

Constructs an execution stream for the specified engine and OpenCL queue.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to create the stream on.

	*
		- queue

		- OpenCL queue to use for the stream.



.. rubric:: Returns:

An execution stream.

.. index:: pair: function; get_command_queue
.. _doxid-namespacednnl_1_1ocl__interop_1a14281f69db5178363ff0c971510d0452:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cl_command_queue get_command_queue(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream)

Returns OpenCL queue object associated with the execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- An execution stream.



.. rubric:: Returns:

Underlying OpenCL queue.

.. index:: pair: function; get_mem_object
.. _doxid-namespacednnl_1_1ocl__interop_1ac117d62fba9de220fe53b0eedb9671f9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cl_mem get_mem_object(const :ref:`memory<doxid-structdnnl_1_1memory>`& amemory)

Returns the OpenCL memory object associated with the memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- amemory

		- A memory object.



.. rubric:: Returns:

Underlying OpenCL memory object.

.. index:: pair: function; set_mem_object
.. _doxid-namespacednnl_1_1ocl__interop_1abe99da7a9ae3286ba6a950921a07eaf0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_mem_object(:ref:`memory<doxid-structdnnl_1_1memory>`& amemory, cl_mem mem_object)

Sets the OpenCL memory object associated with the memory object.

For behavioral details see :ref:`memory::set_data_handle() <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- amemory

		- A memory object.

	*
		- mem_object

		- OpenCL cl_mem object to use as the underlying storage. It must have at least get_desc().get_size() bytes allocated.

.. index:: pair: function; get_memory_kind
.. _doxid-namespacednnl_1_1ocl__interop_1aa94bfc5feb0de9752012d60f4de1ad2f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory_kind<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481>` get_memory_kind(const :ref:`memory<doxid-structdnnl_1_1memory>`& amemory)

Returns the memory allocation kind associated with a memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- amemory

		- A memory object.



.. rubric:: Returns:

The underlying memory allocation kind of the memory object.

.. index:: pair: function; make_memory
.. _doxid-namespacednnl_1_1ocl__interop_1acfb8e3d4cdcff9244e9b530b3f4c4a9d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory<doxid-structdnnl_1_1memory>` make_memory(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`memory_kind<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481>` kind,
		void* handle = :ref:`DNNL_MEMORY_ALLOCATE<doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`
		)

Creates a memory object.

Unless ``handle`` is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if:

* :ref:`dnnl::memory::set_data_handle() <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>` had been called, if ``memory_kind`` is equal to :ref:`dnnl::ocl_interop::memory_kind::usm <doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a81e61a0cab904f0e620dd3226f7f6582>`, or

* :ref:`dnnl::ocl_interop::set_mem_object() <doxid-namespacednnl_1_1ocl__interop_1abe99da7a9ae3286ba6a950921a07eaf0>` has been called, if ``memory_kind`` is equal to :ref:`dnnl::ocl_interop::memory_kind::buffer <doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a7f2db423a49b305459147332fb01cf87>`.



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
		- kind

		- Memory allocation kind to specify the type of handle.

	*
		- handle

		- 
		  Handle of the memory buffer to use as an underlying storage.
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` to be equal to :ref:`dnnl::ocl_interop::memory_kind::usm <doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a81e61a0cab904f0e620dd3226f7f6582>`.
		  
		  * An OpenCL buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` be equal to be equal to :ref:`dnnl::ocl_interop::memory_kind::buffer <doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a7f2db423a49b305459147332fb01cf87>`.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer that corresponds to the memory allocation kind ``memory_kind`` for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

Created memory object.

.. index:: pair: function; make_memory
.. _doxid-namespacednnl_1_1ocl__interop_1a8be4eaee886f3d99f154cfe5d2544994:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory<doxid-structdnnl_1_1memory>` make_memory(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		cl_mem mem_object
		)

Constructs a memory object from an OpenCL buffer.



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
		- mem_object

		- An OpenCL buffer to use.



.. rubric:: Returns:

Created memory object.

.. index:: pair: function; execute
.. _doxid-namespacednnl_1_1ocl__interop_1af585a09a66d7bde78dc7b33557768501:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	cl_event execute(
		const :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`& aprimitive,
		const :ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args,
		const std::vector<cl_event>& deps = {}
		)

Executes computations specified by the primitive in a specified stream and returns a SYCL event.

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

		- Optional vector with ``cl_event`` dependencies.



.. rubric:: Returns:

Output event. It's the user's responsibility to manage lifetime of the event.

