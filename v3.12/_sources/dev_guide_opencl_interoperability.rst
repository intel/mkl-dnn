.. index:: pair: page; OpenCL Interoperability
.. _doxid-dev_guide_opencl_interoperability:

OpenCL Interoperability
=======================

:ref:`API Reference <doxid-group__dnnl__api__ocl__interop>`

Overview
~~~~~~~~

oneDNN uses the OpenCL runtime for GPU engines to interact with the GPU. Users may need to use oneDNN with other code that uses OpenCL. For that purpose, the library provides API extensions to interoperate with underlying OpenCL objects. This interoperability API is defined in the ``dnnl_ocl.hpp`` header.

The interoperability API is provided for two scenarios:

* Construction of oneDNN objects based on existing OpenCL objects

* Accessing OpenCL objects for existing oneDNN objects

The mapping between oneDNN and OpenCL objects is provided in the following table:

======================  ====================================  
oneDNN object           OpenCL object(s)                      
======================  ====================================  
Engine                  ``cl_device_id`` and ``cl_context``   
Stream                  ``cl_command_queue``                  
Memory (Buffer-based)   ``cl_mem``                            
Memory (USM-based)      Unified Shared Memory (USM) pointer   
======================  ====================================

The table below summarizes how to construct oneDNN objects based on OpenCL objects and how to query underlying OpenCL objects for existing oneDNN objects.

======================  ========================================================================================================================================================================================  =============================================================================================================================================================================================================================================================  
oneDNN object           API to construct oneDNN object                                                                                                                                                            API to access OpenCL object(s)                                                                                                                                                                                                                                 
======================  ========================================================================================================================================================================================  =============================================================================================================================================================================================================================================================  
Engine                  :ref:`dnnl::ocl_interop::make_engine(cl_device_id, cl_context) <doxid-namespacednnl_1_1ocl__interop_1aaa1b1a194ca813f3db12effd29a359d7>`                                                  :ref:`dnnl::ocl_interop::get_device(const engine &) <doxid-namespacednnl_1_1ocl__interop_1a37ef1ccb75d09063ed049076fb23b927>` :ref:`dnnl::ocl_interop::get_context(const engine &) <doxid-namespacednnl_1_1ocl__interop_1a248df8106d035e5a7e1ac5fd196c93c3>`   
Stream                  :ref:`dnnl::ocl_interop::make_stream(const engine &, cl_command_queue) <doxid-namespacednnl_1_1ocl__interop_1ad29aa52fd99fb371018ae6761b0bc8fa>`                                          :ref:`dnnl::ocl_interop::get_command_queue(const stream &) <doxid-namespacednnl_1_1ocl__interop_1a14281f69db5178363ff0c971510d0452>`                                                                                                                           
Memory (Buffer-based)   :ref:`dnnl::memory(const memory::desc &, const engine &, cl_mem) <doxid-structdnnl_1_1memory>`                                                                                            :ref:`dnnl::ocl_interop::get_mem_object(const memory &) <doxid-namespacednnl_1_1ocl__interop_1ac117d62fba9de220fe53b0eedb9671f9>`                                                                                                                              
Memory (USM-based)      :ref:`dnnl::ocl_interop::make_memory(const memory::desc &, const engine &, ocl_interop::memory_kind, void \*) <doxid-namespacednnl_1_1ocl__interop_1a085c04e3979cd20f35aa4e887e364c16>`   :ref:`dnnl::memory::get_data_handle() <doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>`                                                                                                                                                         
======================  ========================================================================================================================================================================================  =============================================================================================================================================================================================================================================================

OpenCL Buffers and USM Interfaces for Memory Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The memory model in OpenCL is based on OpenCL buffers. `Intel extension <https://registry.khronos.org/OpenCL/extensions/intel/cl_intel_unified_shared_memory.html>`__ further extends the programming model with a Unified Shared Memory (USM) alternative, which provides the ability to allocate and use memory in a uniform way on host and OpenCL devices.

oneDNN supports both buffer and USM memory models. The buffer model is the default. The USM model requires using the interoperability API.

To construct a oneDNN memory object, use one of the following interfaces:

* :ref:`dnnl::ocl_interop::make_memory(const memory::desc &, const engine &, ocl_interop::memory_kind kind, void \*handle) <doxid-namespacednnl_1_1ocl__interop_1a085c04e3979cd20f35aa4e887e364c16>`
  
  Constructs a USM-based or buffer-based memory object depending on memory allocation kind ``kind``. The ``handle`` could be one of special values :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>` or :ref:`DNNL_MEMORY_NONE <doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>`, or it could be a user-provided USM pointer. The latter works only when ``kind`` is :ref:`dnnl::ocl_interop::memory_kind::usm <doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a81e61a0cab904f0e620dd3226f7f6582>`.

* :ref:`dnnl::memory(const memory::desc &, const engine &, void \*) <doxid-structdnnl_1_1memory>`
  
  Constructs a buffer-based memory object. The call is equivalent to calling the function above with with ``kind`` equal to :ref:`dnnl::ocl_interop::memory_kind::buffer <doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a7f2db423a49b305459147332fb01cf87>`.

* :ref:`dnnl::ocl_interop::make_memory(const memory::desc &, const engine &, cl_mem) <doxid-namespacednnl_1_1ocl__interop_1a8be4eaee886f3d99f154cfe5d2544994>`
  
  Constructs a buffer-based memory object based on a user-provided OpenCL buffer.

To identify whether a memory object is USM-based or buffer-based, :ref:`dnnl::ocl_interop::get_memory_kind() <doxid-namespacednnl_1_1ocl__interop_1aa94bfc5feb0de9752012d60f4de1ad2f>` query can be used.

Handling Dependencies
~~~~~~~~~~~~~~~~~~~~~

OpenCL queues could be in-order or out-of-order. For out-of-order queues, the order of execution is defined by the dependencies between OpenCL tasks therefore users must handle the dependencies using OpenCL events.

oneDNN provides two mechanisms to handle dependencies:

#. :ref:`dnnl::ocl_interop::execute() <doxid-namespacednnl_1_1ocl__interop_1af585a09a66d7bde78dc7b33557768501>` interface
   
   This interface enables the user to pass dependencies between primitives using OpenCL events. In this case, the user is responsible for passing proper dependencies for every primitive execution.

#. In-order oneDNN stream
   
   oneDNN enables the user to create in-order streams when submitted primitives are executed in the order they were submitted. Using in-order streams prevents possible read-before-write or concurrent read/write issues.

.. note:: 

   oneDNN follows retain/release OpenCL semantics when using OpenCL objects during construction. An OpenCL object is retained on construction and released on destruction. This ensures that the OpenCL object will not be destroyed while the oneDNN object stores a reference to it.
   
   

.. note:: 

   The access interfaces do not retain the OpenCL object. It is the user's responsibility to retain the returned OpenCL object if necessary.
   
   

.. note:: 

   It's the user's responsibility to manage lifetime of the OpenCL event returned by :ref:`dnnl::ocl_interop::execute() <doxid-namespacednnl_1_1ocl__interop_1af585a09a66d7bde78dc7b33557768501>`.
   
   

.. note:: 

   USM memory doesn't support retain/release OpenCL semantics. When constructing a oneDNN memory object using a user-provided USM pointer oneDNN doesn't own the provided memory. It's user's responsibility to manage its lifetime.

