.. index:: pair: struct; dnnl::memory
.. _doxid-structdnnl_1_1memory:

struct dnnl::memory
===================

.. toctree::
	:hidden:

	enum_dnnl_memory_data_type.rst
	enum_dnnl_memory_format_kind.rst
	enum_dnnl_memory_format_tag.rst
	enum_dnnl_memory_sparse_encoding.rst
	struct_dnnl_memory_desc-2.rst

Overview
~~~~~~~~

Memory object. :ref:`More...<details-structdnnl_1_1memory>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct memory: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// typedefs
	
		typedef :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` :ref:`dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`;
		typedef std::vector<:ref:`dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`> :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`;

		// enums
	
		enum :ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>`;
		enum :ref:`format_kind<doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105f>`;
		enum :ref:`format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>`;
		enum :ref:`sparse_encoding<doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966>`;

		// structs
	
		struct :ref:`desc<doxid-structdnnl_1_1memory_1_1desc>`;

		// construction
	
		:ref:`memory<doxid-structdnnl_1_1memory_1a5f509ba0d38054c147f1dceef1a42d44>`();
		:ref:`memory<doxid-structdnnl_1_1memory_1a7463ff54b529ec2b5392230861212a09>`(const :ref:`desc<doxid-structdnnl_1_1memory_1_1desc>`& md, const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, void* handle);
		:ref:`memory<doxid-structdnnl_1_1memory_1adfb0936410dd301a28fd2883ae97f01f>`(const :ref:`desc<doxid-structdnnl_1_1memory_1_1desc>`& md, const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, std::vector<void*> handles);
		:ref:`memory<doxid-structdnnl_1_1memory_1ab0892880e22c2cced48c44f405ced029>`(const :ref:`desc<doxid-structdnnl_1_1memory_1_1desc>`& md, const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine);

		// methods
	
		template <typename T>
		static void :ref:`validate_dims<doxid-structdnnl_1_1memory_1a5a02c4d5aa4a07650977ba57ed65bd9a>`(
			const std::vector<T>& v,
			int min_size = 0
			);
	
		static size_t :ref:`data_type_size<doxid-structdnnl_1_1memory_1ac4064e92cc225fbb6a0431b90004511c>`(:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type);
		static :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` :target:`convert_to_c<doxid-structdnnl_1_1memory_1a582cbe883d93acb3882a627fe64d9858>`(:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type);
		static :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` :target:`convert_to_c<doxid-structdnnl_1_1memory_1ac0c82ffa0ea01aca3884b8440b18002c>`(:ref:`format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` format);
		:ref:`desc<doxid-structdnnl_1_1memory_1_1desc>` :ref:`get_desc<doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>`() const;
		:ref:`engine<doxid-structdnnl_1_1engine>` :ref:`get_engine<doxid-structdnnl_1_1memory_1a9074709c5af8dc9d25dd9a98c4d1dbd3>`() const;
		void* :ref:`get_data_handle<doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>`(int index = 0) const;
		void :ref:`set_data_handle<doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>`(void* handle, int index = 0) const;
	
		template <typename T = void>
		T* :ref:`map_data<doxid-structdnnl_1_1memory_1a65ef31e805fe967e6924868d2b4a210a>`(int index = 0) const;
	
		void :ref:`unmap_data<doxid-structdnnl_1_1memory_1a2b5abd181b3cb0394492458ac26de677>`(void* mapped_ptr, int index = 0) const;
		:ref:`handle<doxid-structdnnl_1_1memory_1a5c631f7e5e4c92a13edb8e3422d3a973>`();
		:ref:`handle<doxid-structdnnl_1_1memory_1a022001b5b9c8940a1326a02b61fc4860>`();
		:ref:`handle<doxid-structdnnl_1_1memory_1aa13f3ecf4db240717074814412c7e70c>`();
		:ref:`handle<doxid-structdnnl_1_1memory_1a9c408c09fce1278f5cb0d1fa9818fc86>`();
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// methods
	
		:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1a4ad1ff54e4aafeb560a869c49aa20b52>` (const :ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>&);
		:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& :ref:`operator =<doxid-structdnnl_1_1handle_1af3f85524f3d83abdd4917b46ce23e727>` (:ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>&&);
		void :ref:`reset<doxid-structdnnl_1_1handle_1a8862ef3d31c3b19bd88395e0b1373909>`(T t, bool weak = false);
		T :ref:`get<doxid-structdnnl_1_1handle_1a2208243e1d147a0be9da87fff46ced7e>`(bool allow_empty = false) const;
		:ref:`operator T<doxid-structdnnl_1_1handle_1a498e45a0937a32367b400b09dbc3dac3>` () const;
		:ref:`operator bool<doxid-structdnnl_1_1handle_1ad14e2635ad97d873f0114ed77c1f55d5>` () const;
		bool :ref:`operator ==<doxid-structdnnl_1_1handle_1a069b5ea2a2c13fc4ebefd4fb51d0899e>` (const :ref:`handle<doxid-structdnnl_1_1handle>`<T, traits>& other) const;
		bool :ref:`operator !=<doxid-structdnnl_1_1handle_1a1895f4cd3fc3eca7560756c0c508e5ab>` (const :ref:`handle<doxid-structdnnl_1_1handle>`& other) const;

.. _details-structdnnl_1_1memory:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory object.

A memory object encapsulates a handle to a memory buffer allocated on a specific engine, tensor dimensions, data type, and memory format, which is the way tensor indices map to offsets in linear memory space. Memory objects are passed to primitives during execution.

Typedefs
--------

.. index:: pair: typedef; dim
.. _doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dim

Integer type for representing dimension sizes and indices.

.. index:: pair: typedef; dims
.. _doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef std::vector<:ref:`dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>`> dims

Vector of dimensions.

Implementations are free to force a limit on the vector's length.

Construction
------------

.. index:: pair: function; memory
.. _doxid-structdnnl_1_1memory_1a5f509ba0d38054c147f1dceef1a42d44:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	memory()

Default constructor.

Constructs an empty memory object, which can be used to indicate absence of a parameter.

.. index:: pair: function; memory
.. _doxid-structdnnl_1_1memory_1a7463ff54b529ec2b5392230861212a09:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	memory(const :ref:`desc<doxid-structdnnl_1_1memory_1_1desc>`& md, const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, void* handle)

Constructs a memory object.

Unless ``handle`` is equal to :ref:`DNNL_MEMORY_NONE <doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>`, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if :ref:`dnnl::memory::set_data_handle() <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>` had been called.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- md

		- Memory descriptor.

	*
		- aengine

		- Engine to store the data on.

	*
		- handle

		- 
		  Handle of the memory buffer to use.
		  
		  * A pointer to the user-allocated buffer. In this case the library doesn't own the buffer.
		  
		  * The :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>` special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
		  
		  * :ref:`DNNL_MEMORY_NONE <doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>` to create :ref:`dnnl::memory <doxid-structdnnl_1_1memory>` without an underlying buffer.



.. rubric:: See also:

:ref:`memory::set_data_handle() <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>`

.. index:: pair: function; memory
.. _doxid-structdnnl_1_1memory_1adfb0936410dd301a28fd2883ae97f01f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	memory(const :ref:`desc<doxid-structdnnl_1_1memory_1_1desc>`& md, const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine, std::vector<void*> handles)

Constructs a memory object with multiple handles.

Unless ``handle`` is equal to :ref:`DNNL_MEMORY_NONE <doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>`, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if :ref:`dnnl::memory::set_data_handle() <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>` had been called.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- md

		- Memory descriptor.

	*
		- aengine

		- Engine to store the data on.

	*
		- handles

		- 
		  Handles of the memory buffers to use. For each element of the ``handles`` vector the following applies:
		  
		  * A pointer to the user-allocated buffer. In this case the library doesn't own the buffer.
		  
		  * The :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>` special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
		  
		  * :ref:`DNNL_MEMORY_NONE <doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>` Instructs the library to skip allocation of the memory buffer.



.. rubric:: See also:

:ref:`memory::set_data_handle() <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>`

.. index:: pair: function; memory
.. _doxid-structdnnl_1_1memory_1ab0892880e22c2cced48c44f405ced029:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	memory(const :ref:`desc<doxid-structdnnl_1_1memory_1_1desc>`& md, const :ref:`engine<doxid-structdnnl_1_1engine>`& aengine)

Constructs a memory object.

The underlying buffer(s) for the memory will be allocated by the library.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- md

		- Memory descriptor.

	*
		- aengine

		- Engine to store the data on.

Methods
-------

.. index:: pair: function; validate_dims
.. _doxid-structdnnl_1_1memory_1a5a02c4d5aa4a07650977ba57ed65bd9a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename T>
	static void validate_dims(
		const std::vector<T>& v,
		int min_size = 0
		)

Helper function that validates that an ``std::vector`` of dimensions can be safely converted to the C API array :ref:`dnnl_dims_t <doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>`.

Throws if validation fails.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- v

		- Vector of dimensions.

	*
		- min_size

		- Minimum expected size of the vector.

.. index:: pair: function; data_type_size
.. _doxid-structdnnl_1_1memory_1ac4064e92cc225fbb6a0431b90004511c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static size_t data_type_size(:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type)

Returns size of data type in bytes.



.. rubric:: Returns:

The number of bytes occupied by data type.

.. index:: pair: function; get_desc
.. _doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`desc<doxid-structdnnl_1_1memory_1_1desc>` get_desc() const

Returns the associated memory descriptor.

.. index:: pair: function; get_engine
.. _doxid-structdnnl_1_1memory_1a9074709c5af8dc9d25dd9a98c4d1dbd3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`engine<doxid-structdnnl_1_1engine>` get_engine() const

Returns the associated engine.

.. index:: pair: function; get_data_handle
.. _doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void* get_data_handle(int index = 0) const

Returns an underlying memory buffer that corresponds to the given index.

On the CPU engine, or when using USM, this is a pointer to the allocated memory.

.. index:: pair: function; set_data_handle
.. _doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void set_data_handle(void* handle, int index = 0) const

Sets an underlying memory buffer that corresponds to the given index.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- handle

		- Memory buffer to use. On the CPU engine or when USM is used, the memory buffer is a pointer to the actual data. For OpenCL it is a cl_mem. It must have at least :ref:`dnnl::memory::desc::get_size() <doxid-structdnnl_1_1memory_1_1desc_1abfa095ac138d4d2ef8efd3739e343f08>` bytes allocated.

	*
		- index

		- Memory index to attach the buffer. Defaults to 0.

.. index:: pair: function; map_data
.. _doxid-structdnnl_1_1memory_1a65ef31e805fe967e6924868d2b4a210a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename T = void>
	T* map_data(int index = 0) const

Maps a memory object and returns a host-side pointer to a memory buffer with a copy of its contents.

The memory buffer corresponds to the given index.

Mapping enables read/write directly from/to the memory contents for engines that do not support direct memory access.

Mapping is an exclusive operation - a memory object cannot be used in other operations until it is unmapped via :ref:`dnnl::memory::unmap_data() <doxid-structdnnl_1_1memory_1a2b5abd181b3cb0394492458ac26de677>` call.

.. note:: 

   Any primitives working with the memory should be completed before the memory is mapped. Use :ref:`dnnl::stream::wait() <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>` to synchronize the corresponding execution stream.
   
   

.. note:: 

   The map_data and unmap_data functions are provided mainly for debug and testing purposes and their performance may be suboptimal.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- T

		- Data type to return a pointer to.

	*
		- index

		- Index of the buffer. Defaults to 0.



.. rubric:: Returns:

Pointer to the mapped memory.

.. index:: pair: function; unmap_data
.. _doxid-structdnnl_1_1memory_1a2b5abd181b3cb0394492458ac26de677:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void unmap_data(void* mapped_ptr, int index = 0) const

Unmaps a memory object and writes back any changes made to the previously mapped memory buffer.

The memory buffer corresponds to the given index.

.. note:: 

   The map_data and unmap_data functions are provided mainly for debug and testing purposes and their performance may be suboptimal.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- mapped_ptr

		- A pointer previously returned by :ref:`dnnl::memory::map_data() <doxid-structdnnl_1_1memory_1a65ef31e805fe967e6924868d2b4a210a>`.

	*
		- index

		- Index of the buffer. Defaults to 0.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1memory_1a5c631f7e5e4c92a13edb8e3422d3a973:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Constructs an empty handle object.

.. warning:: 

   Uninitialized object cannot be used in most library calls and is equivalent to a null pointer. Any attempt to use its methods, or passing it to the other library function, will cause an exception to be thrown.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1memory_1a022001b5b9c8940a1326a02b61fc4860:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Copy constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1memory_1aa13f3ecf4db240717074814412c7e70c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Move constructor.

.. index:: pair: function; handle
.. _doxid-structdnnl_1_1memory_1a9c408c09fce1278f5cb0d1fa9818fc86:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	handle()

Constructs a handle wrapper object from a C API handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- t

		- The C API handle to wrap.

	*
		- weak

		- A flag specifying whether to construct a weak wrapper; defaults to ``false``.

