.. index:: pair: page; DPC++ Interoperability
.. _doxid-dev_guide_dpcpp_interoperability:

DPC++ Interoperability
======================

:ref:`API Reference <doxid-group__dnnl__api__sycl__interop>`

Overview
~~~~~~~~

oneDNN may use the DPC++ runtime for CPU and GPU engines to interact with the hardware. Users may need to use oneDNN with other code that uses DPC++. For that purpose, the library provides API extensions to interoperate with underlying SYCL objects. This interoperability API is defined in the ``dnnl_sycl.hpp`` header.

One of the possible scenarios is executing a SYCL kernel for a custom operation not provided by oneDNN. In this case, the library provides all the necessary API to "seamlessly" submit a kernel, sharing the execution context with oneDNN: using the same device and queue.

The interoperability API is provided for two scenarios:

* Construction of oneDNN objects based on existing SYCL objects

* Accessing SYCL objects for existing oneDNN objects

The mapping between oneDNN and SYCL objects is provided in the following table:

======================  =======================================  
oneDNN object           SYCL object(s)                           
======================  =======================================  
Engine                  ``sycl::device`` and ``sycl::context``   
Stream                  ``sycl::queue``                          
Memory (Buffer-based)   ``sycl::buffer<uint8_t, 1>``             
Memory (USM-based)      Unified Shared Memory (USM) pointer      
======================  =======================================

The table below summarizes how to construct oneDNN objects based on SYCL objects and how to query underlying SYCL objects for existing oneDNN objects.

======================  ==================================================================================================================================================================================  =================================================================================================================================================================================================================================================================  
oneDNN object           API to construct oneDNN object                                                                                                                                                      API to access SYCL object(s)                                                                                                                                                                                                                                       
======================  ==================================================================================================================================================================================  =================================================================================================================================================================================================================================================================  
Engine                  :ref:`dnnl::sycl_interop::make_engine(const sycl::device &, const sycl::context &) <doxid-namespacednnl_1_1sycl__interop_1a683783e1493808bd6ac2204d5efa63a8>`                       :ref:`dnnl::sycl_interop::get_device(const engine &) <doxid-namespacednnl_1_1sycl__interop_1adddf805d923929f373fb6233f1fd4a27>` :ref:`dnnl::sycl_interop::get_context(const engine &) <doxid-namespacednnl_1_1sycl__interop_1a5227caa35295b41dcdd57f8abaa7551b>`   
Stream                  :ref:`dnnl::sycl_interop::make_stream(const engine &, sycl::queue &) <doxid-namespacednnl_1_1sycl__interop_1a170bddd16d53869fc18412894400ccab>`                                     :ref:`dnnl::sycl_interop::get_queue(const stream &) <doxid-namespacednnl_1_1sycl__interop_1a59a9e92e8ff59c1282270fc6edad4274>`                                                                                                                                     
Memory (Buffer-based)   :ref:`dnnl::sycl_interop::make_memory(const memory::desc &, const engine &, sycl::buffer\<T, ndims> &) <doxid-namespacednnl_1_1sycl__interop_1a58de74caadf2b2bc3d22cb557682ef47>`   :ref:`dnnl::sycl_interop::get_buffer\<T, ndims>(const memory &) <doxid-namespacednnl_1_1sycl__interop_1a3a982d9d12f29f0856cba970b470d4d0>`                                                                                                                         
Memory (USM-based)      :ref:`dnnl::memory(const memory::desc &, const engine &, void \*) <doxid-structdnnl_1_1memory>`                                                                                     :ref:`dnnl::memory::get_data_handle() <doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>`                                                                                                                                                             
======================  ==================================================================================================================================================================================  =================================================================================================================================================================================================================================================================

.. note:: 

   Internally, library buffer-based memory objects use 1D ``uint8_t`` SYCL buffers; however, the user may initialize and access memory using SYCL buffers. of a different type. In this case, buffers will be reinterpreted to the underlying type ``sycl::buffer<uint8_t, 1>``.
   
   


SYCL Buffers and DPC++ USM Interfaces for Memory Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The memory model in SYCL 1.2.1 is based on SYCL buffers. DPC++ further extends the programming model with a Unified Shared Memory (USM) alternative, which provides the ability to allocate and use memory in a uniform way on host and DPC++ devices.

oneDNN supports both programming models. USM is the default and can be used with the usual oneDNN :ref:`memory API <doxid-group__dnnl__api__memory>`. The buffer-based programming model requires using the interoperability API.

To construct a oneDNN memory object, use one of the following interfaces:

* :ref:`dnnl::sycl_interop::make_memory(const memory::desc &, const engine &, sycl_interop::memory_kind kind, void \*handle) <doxid-namespacednnl_1_1sycl__interop_1ac4f19c86efba789310d287ad1edfb657>`
  
  Constructs a USM-based or buffer-based memory object depending on memory allocation kind ``kind``. The ``handle`` could be one of special values :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>` or :ref:`DNNL_MEMORY_NONE <doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>`, or it could be a user-provided USM pointer. The latter works only when ``kind`` is :ref:`dnnl::sycl_interop::memory_kind::usm <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba81e61a0cab904f0e620dd3226f7f6582>` and the ``handle`` is a USM-allocated pointer.

* :ref:`dnnl::memory(const memory::desc &, const engine &, void \*) <doxid-structdnnl_1_1memory>`
  
  Constructs a USM-based memory object. The call is equivalent to calling the function above with with ``kind`` equal to :ref:`dnnl::sycl_interop::memory_kind::usm <doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba81e61a0cab904f0e620dd3226f7f6582>`.

* :ref:`dnnl::sycl_interop::make_memory(const memory::desc &, const engine &, sycl::buffer\<T, ndims> &) <doxid-namespacednnl_1_1sycl__interop_1a58de74caadf2b2bc3d22cb557682ef47>`
  
  Constructs a buffer-based memory object based on a user-provided SYCL buffer.

To identify whether a memory object is USM-based or buffer-based, :ref:`dnnl::sycl_interop::get_memory_kind() <doxid-namespacednnl_1_1sycl__interop_1a9859fd3ed9a833cc88cb02882051cffb>` query can be used.

Handling Dependencies with USM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SYCL queues could be in-order or out-of-order. For out-of-order queues, the order of execution is defined by the dependencies between SYCL tasks. The runtime tracks dependencies based on accessors created for SYCL buffers. USM pointers cannot be used to create accessors and users must handle dependencies on their own using SYCL events.

oneDNN provides two mechanisms to handle dependencies when USM memory is used:

#. :ref:`dnnl::sycl_interop::execute() <doxid-namespacednnl_1_1sycl__interop_1a30c5c906dfba71774528710613165c14>` interface
   
   This interface enables you to pass dependencies between primitives using SYCL events. In this case, the user is responsible for passing proper dependencies for every primitive execution.

#. In-order oneDNN stream
   
   oneDNN enables you to create in-order streams when submitted primitives are executed in the order they were submitted. Using in-order streams prevents possible read-before-write or concurrent read/write issues.

