.. index:: pair: namespace; dnnl::sycl_interop
.. _doxid-namespacednnl_1_1sycl__interop:

namespace dnnl::sycl_interop
============================

.. toctree::
	:hidden:

	enum_dnnl_sycl_interop_memory_kind.rst

Overview
~~~~~~~~

SYCL interoperability namespace. :ref:`More...<details-namespacednnl_1_1sycl__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace sycl_interop {

	// enums

	enum :ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>`;

	// global functions

	:ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>` :ref:`convert_to_c<doxid-namespacednnl_1_1sycl__interop_1a61c88b2b3dd997b5e99232c218a8ac8f>`(:ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>` akind);
	:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`make_engine<doxid-namespacednnl_1_1sycl__interop_1a683783e1493808bd6ac2204d5efa63a8>`(const sycl::device& adevice, const sycl::context& acontext);
	sycl::context :ref:`get_context<doxid-namespacednnl_1_1sycl__interop_1a5227caa35295b41dcdd57f8abaa7551b>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);
	sycl::device :ref:`get_device<doxid-namespacednnl_1_1sycl__interop_1adddf805d923929f373fb6233f1fd4a27>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);
	:ref:`stream<doxid-structdnnl_1_1stream>` :ref:`make_stream<doxid-namespacednnl_1_1sycl__interop_1a170bddd16d53869fc18412894400ccab>`(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, sycl::queue& aqueue);
	sycl::queue :ref:`get_queue<doxid-namespacednnl_1_1sycl__interop_1a59a9e92e8ff59c1282270fc6edad4274>`(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream);

	template <typename T, int ndims = 1>
	:ref:`sycl::buffer<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`<T, ndims> :ref:`get_buffer<doxid-namespacednnl_1_1sycl__interop_1a3a982d9d12f29f0856cba970b470d4d0>`(const :ref:`memory<doxid-structdnnl_1_1memory>`& amemory);

	template <typename T, int ndims>
	void :ref:`set_buffer<doxid-namespacednnl_1_1sycl__interop_1abc037ad6dca6da72275911e1d4a21473>`(
		:ref:`memory<doxid-structdnnl_1_1memory>`& amemory,
		:ref:`sycl::buffer<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`<T, ndims>& abuffer
		);

	:ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>` :ref:`get_memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9859fd3ed9a833cc88cb02882051cffb>`(const :ref:`memory<doxid-structdnnl_1_1memory>`& amemory);

	:ref:`memory<doxid-structdnnl_1_1memory>` :ref:`make_memory<doxid-namespacednnl_1_1sycl__interop_1a5f3bf8334f86018201e14fec6a666be4>`(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>` kind,
		std::vector<void*> handles = {}
		);

	:ref:`memory<doxid-structdnnl_1_1memory>` :ref:`make_memory<doxid-namespacednnl_1_1sycl__interop_1a30115d67d52bebe64fa39d6003f753e6>`(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>` kind,
		void* handle
		);

	template <typename T, int ndims = 1>
	:ref:`memory<doxid-structdnnl_1_1memory>` :ref:`make_memory<doxid-namespacednnl_1_1sycl__interop_1a58de74caadf2b2bc3d22cb557682ef47>`(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`sycl::buffer<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`<T, ndims>& abuffer
		);

	sycl::event :ref:`execute<doxid-namespacednnl_1_1sycl__interop_1a30c5c906dfba71774528710613165c14>`(
		const :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`& aprimitive,
		const :ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args,
		const std::vector<sycl::event>& deps = {}
		);

	} // namespace sycl_interop
.. _details-namespacednnl_1_1sycl__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

SYCL interoperability namespace.

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-namespacednnl_1_1sycl__interop_1a61c88b2b3dd997b5e99232c218a8ac8f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_sycl_interop_memory_kind_t<doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>` convert_to_c(:ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>` akind)

Converts a memory allocation kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- C++ API memory allocation kind enum value.



.. rubric:: Returns:

Corresponding C API memory allocation kind enum value.

.. index:: pair: function; make_engine
.. _doxid-namespacednnl_1_1sycl__interop_1a683783e1493808bd6ac2204d5efa63a8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` make_engine(const sycl::device& adevice, const sycl::context& acontext)

Constructs an engine from SYCL device and context objects.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- adevice

		- SYCL device.

	*
		- acontext

		- SYCL context.



.. rubric:: Returns:

Created engine.

.. index:: pair: function; get_context
.. _doxid-namespacednnl_1_1sycl__interop_1a5227caa35295b41dcdd57f8abaa7551b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sycl::context get_context(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns the SYCL context associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to query.



.. rubric:: Returns:

The underlying SYCL device of the engine.

.. index:: pair: function; get_device
.. _doxid-namespacednnl_1_1sycl__interop_1adddf805d923929f373fb6233f1fd4a27:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sycl::device get_device(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Returns the SYCL device associated with an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine to query.



.. rubric:: Returns:

The underlying SYCL context of the engine.

.. index:: pair: function; make_stream
.. _doxid-namespacednnl_1_1sycl__interop_1a170bddd16d53869fc18412894400ccab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`stream<doxid-structdnnl_1_1stream>` make_stream(const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, sycl::queue& aqueue)

Creates an execution stream for a given engine associated with a SYCL queue.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- aengine

		- Engine object to use for the stream.

	*
		- aqueue

		- SYCL queue to use for the stream.



.. rubric:: Returns:

An execution stream.

.. index:: pair: function; get_queue
.. _doxid-namespacednnl_1_1sycl__interop_1a59a9e92e8ff59c1282270fc6edad4274:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sycl::queue get_queue(const :ref:`stream<doxid-structdnnl_1_1stream>`& astream)

Returns the SYCL queue associated with an execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- Execution stream to query.



.. rubric:: Returns:

SYCL queue object.

.. index:: pair: function; get_buffer
.. _doxid-namespacednnl_1_1sycl__interop_1a3a982d9d12f29f0856cba970b470d4d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename T, int ndims = 1>
	:ref:`sycl::buffer<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`<T, ndims> get_buffer(const :ref:`memory<doxid-structdnnl_1_1memory>`& amemory)

Returns the SYCL buffer associated with a memory object.

Throws an exception if the memory allocation kind associated with the memory object is not equal to :ref:`dnnl::sycl_interop::memory_kind::buffer <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- T

		- Type of the requested buffer.

	*
		- ndims

		- Number of dimensions of the requested buffer.

	*
		- amemory

		- Memory object.



.. rubric:: Returns:

SYCL buffer associated with the memory object.

.. index:: pair: function; set_buffer
.. _doxid-namespacednnl_1_1sycl__interop_1abc037ad6dca6da72275911e1d4a21473:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename T, int ndims>
	void set_buffer(
		:ref:`memory<doxid-structdnnl_1_1memory>`& amemory,
		:ref:`sycl::buffer<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`<T, ndims>& abuffer
		)

Sets SYCL buffer associated with a memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- T

		- Type of the buffer.

	*
		- ndims

		- Number of dimensions of the buffer.

	*
		- amemory

		- Memory object to change.

	*
		- abuffer

		- SYCL buffer.

.. index:: pair: function; get_memory_kind
.. _doxid-namespacednnl_1_1sycl__interop_1a9859fd3ed9a833cc88cb02882051cffb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>` get_memory_kind(const :ref:`memory<doxid-structdnnl_1_1memory>`& amemory)

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
.. _doxid-namespacednnl_1_1sycl__interop_1a5f3bf8334f86018201e14fec6a666be4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory<doxid-structdnnl_1_1memory>` make_memory(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>` kind,
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
		- kind

		- Memory allocation kind to specify the type of handles.

	*
		- handles

		- 
		  Handles of the memory buffers to use as underlying storages. For each element of the ``handles`` array the following applies:
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` to be equal to :ref:`dnnl::sycl_interop::memory_kind::usm <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba81e61a0cab904f0e620dd3226f7f6582>`.
		  
		  * A pointer to SYCL buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` be equal to be equal to :ref:`dnnl::sycl_interop::memory_kind::buffer <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer that corresponds to the memory allocation kind ``memory_kind`` for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

Created memory object.

.. index:: pair: function; make_memory
.. _doxid-namespacednnl_1_1sycl__interop_1a30115d67d52bebe64fa39d6003f753e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory<doxid-structdnnl_1_1memory>` make_memory(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`memory_kind<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>` kind,
		void* handle
		)

Creates a memory object.

Unless ``handle`` is equal to DNNL_MEMORY_NONE or DNNL_MEMORY_ALLOCATE, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if:

* :ref:`dnnl::memory::set_data_handle() <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>` had been called, if ``memory_kind`` is equal to :ref:`dnnl::sycl_interop::memory_kind::usm <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba81e61a0cab904f0e620dd3226f7f6582>`, or

* :ref:`dnnl::sycl_interop::set_buffer() <doxid-namespacednnl_1_1sycl__interop_1abc037ad6dca6da72275911e1d4a21473>` has been called, if ``memory_kind`` is equal to :ref:`dnnl::sycl_interop::memory_kind::buffer <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`.



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
		  
		  * A USM pointer to the user-allocated buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` to be equal to :ref:`dnnl::sycl_interop::memory_kind::usm <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba81e61a0cab904f0e620dd3226f7f6582>`.
		  
		  * A pointer to SYCL buffer. In this case the library doesn't own the buffer. Requires ``memory_kind`` be equal to be equal to :ref:`dnnl::sycl_interop::memory_kind::buffer <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer that corresponds to the memory allocation kind ``memory_kind`` for the memory object. In this case the library owns the buffer.
		  
		  * The DNNL_MEMORY_NONE specific value. Instructs the library to create memory object without an underlying buffer.



.. rubric:: Returns:

Created memory object.

.. index:: pair: function; make_memory
.. _doxid-namespacednnl_1_1sycl__interop_1a58de74caadf2b2bc3d22cb557682ef47:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename T, int ndims = 1>
	:ref:`memory<doxid-structdnnl_1_1memory>` make_memory(
		const :ref:`memory::desc<doxid-structdnnl_1_1memory_1_1desc>`& memory_desc,
		const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine,
		:ref:`sycl::buffer<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>`<T, ndims>& abuffer
		)

Constructs a memory object from a SYCL buffer.



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
		- abuffer

		- A SYCL buffer to use.



.. rubric:: Returns:

Created memory object.

.. index:: pair: function; execute
.. _doxid-namespacednnl_1_1sycl__interop_1a30c5c906dfba71774528710613165c14:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sycl::event execute(
		const :ref:`dnnl::primitive<doxid-structdnnl_1_1primitive>`& aprimitive,
		const :ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::unordered_map<int, :ref:`memory<doxid-structdnnl_1_1memory>`>& args,
		const std::vector<sycl::event>& deps = {}
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

		- Optional vector with ``sycl::event`` dependencies.



.. rubric:: Returns:

Output event.

