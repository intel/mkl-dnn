.. index:: pair: group; Memory
.. _doxid-group__dnnl__api__cpp__memory:

Memory
======

.. toctree::
	:hidden:

	struct_dnnl_memory-2.rst

A container that describes and stores data. Memory objects can contain data of various types and formats. There are two levels of abstraction:

#. Memory descriptor engine-agnostic logical description of data (number of dimensions, dimension sizes, and data type), and, optionally, the information about the physical format of data in memory. If this information is not known yet, a memory descriptor can be created with :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`. This allows compute-intensive primitives to choose the best format for computation. The user is responsible for reordering the data into the chosen format when formats do not match.
   
   A memory descriptor can be initialized either by specifying dimensions and a memory format tag or strides for each of them, or by manipulating the dnnl_memory_desc_t structure directly.
   
   .. warning:: 
   
      The latter approach requires understanding how the physical data representation is mapped to the structure and is discouraged. This topic is discussed in :ref:`Understanding Memory Formats <doxid-dev_guide_understanding_memory_formats>`.
      
      
   The user can query the amount of memory required by a memory descriptor using the :ref:`dnnl::memory::desc::get_size() <doxid-structdnnl_1_1memory_1_1desc_1abfa095ac138d4d2ef8efd3739e343f08>` function. The size of data in general cannot be computed as the product of dimensions multiplied by the size of the data type. So users are required to use this function for better code portability.
   
   Two memory descriptors can be compared using the equality and inequality operators. The comparison is especially useful when checking whether it is necessary to reorder data from the user's data format to a primitive's format.

#. Memory object an engine-specific object that handles the memory buffer and its description (a memory descriptor). For the CPU engine or with USM, the memory buffer handle is simply a pointer to ``void``. The memory buffer can be queried using :ref:`dnnl::memory::get_data_handle() <doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>` and set using :ref:`dnnl::memory::set_data_handle() <doxid-structdnnl_1_1memory_1a34d1c7dbe9c6302b197f22c300e67aed>`. The underlying SYCL buffer, when used, can be queried using :ref:`dnnl::sycl_interop::get_buffer <doxid-namespacednnl_1_1sycl__interop_1a3a982d9d12f29f0856cba970b470d4d0>` and set using :ref:`dnnl::sycl_interop::set_buffer <doxid-namespacednnl_1_1sycl__interop_1abc037ad6dca6da72275911e1d4a21473>`. A memory object can also be queried for the underlying memory descriptor and for its engine using :ref:`dnnl::memory::get_desc() <doxid-structdnnl_1_1memory_1ad8a1ad28ed7acf9c34c69e4b882c6e92>` and :ref:`dnnl::memory::get_engine() <doxid-structdnnl_1_1memory_1a9074709c5af8dc9d25dd9a98c4d1dbd3>`.

Along with ordinary memory descriptors with all dimensions being positive, the library supports zero-volume memory descriptors with one or more dimensions set to zero. This is used to support the NumPy\* convention. If a zero-volume memory is passed to a primitive, the primitive typically does not perform any computations with this memory. For example:

* A concatenation primitive would ignore all memory object with zeroes in the concat dimension / axis.

* A forward convolution with a source memory object with zero in the minibatch dimension would always produce a destination memory object with a zero in the minibatch dimension and perform no computations.

* However, a forward convolution with a zero in one of the weights dimensions is ill-defined and is considered to be an error by the library because there is no clear definition of what the output values should be.

Memory buffer of a zero-volume memory is never accessed.


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::memory<doxid-structdnnl_1_1memory>`;

	// global functions

	bool :target:`dnnl::operator ==<doxid-group__dnnl__api__cpp__memory_1gaf97bbd7e992c0e211da42bc6eaf12758>` (:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a, :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b);
	bool :target:`dnnl::operator !=<doxid-group__dnnl__api__cpp__memory_1ga03fa7afa494ab8dc8484f83a34ce20a6>` (:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a, :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b);
	bool :target:`dnnl::operator ==<doxid-group__dnnl__api__cpp__memory_1gafa9de7b46bedc943161863d3eaa84100>` (:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a, :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b);
	bool :target:`dnnl::operator !=<doxid-group__dnnl__api__cpp__memory_1ga7d9b4a4b2297d66c9495d3a1f2769167>` (:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a, :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b);
	bool :target:`dnnl::operator ==<doxid-group__dnnl__api__cpp__memory_1ga659960c63f701a0608368e89c5c4ab04>` (:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` a, :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` b);
	bool :target:`dnnl::operator !=<doxid-group__dnnl__api__cpp__memory_1gaca9a006590333e3764895a66f0e1a3f2>` (:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` a, :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` b);
	bool :target:`dnnl::operator ==<doxid-group__dnnl__api__cpp__memory_1ga3ae4b6e7ef0bf507b64d875a7c24ae7e>` (:ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` a, :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` b);
	bool :target:`dnnl::operator !=<doxid-group__dnnl__api__cpp__memory_1ga6806a5794c45a09b5a9948a5628ffc34>` (:ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` a, :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` b);

