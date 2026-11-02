.. index:: pair: page; Level Zero Interoperability
.. _doxid-dev_guide_level_zero_interoperability:

Level Zero Interoperability
===========================

:ref:`API Reference <doxid-group__dnnl__api__ze__interop>`

Overview
~~~~~~~~

oneDNN uses the Level Zero runtime for GPU engines to interact with the GPU. Users may need to use oneDNN with other code that uses Level Zero. For that purpose, the library provides API extensions to interoperate with underlying Level Zero objects. This interoperability API is defined in the ``dnnl_ze.hpp`` header.

The interoperability API is provided for two scenarios:

* Construction of oneDNN objects based on existing Level Zero objects.

* Accessing Level Zero objects for existing oneDNN objects.

The mapping between oneDNN and Level Zero objects is provided in the following table:

===================  ==============================================================================  
oneDNN object        Level Zero object(s)                                                            
===================  ==============================================================================  
Engine               ``ze_driver_handle_t`` , ``ze_device_handle_t`` , and ``ze_context_handle_t``   
Stream               ``ze_command_list_handle_t``                                                    
Memory (USM-based)   Unified Shared Memory (USM) pointer                                             
===================  ==============================================================================

The table below summarizes how to construct oneDNN objects based on Level Zero objects and how to query underlying Level Zero objects for existing oneDNN objects.

===================  ==========================================================================================================================================================================  =============================================================================================================================  
oneDNN object        API to construct oneDNN object                                                                                                                                              API to access Level Zero object(s)                                                                                             
===================  ==========================================================================================================================================================================  =============================================================================================================================  
Engine               :ref:`dnnl::ze_interop::make_engine(ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t) <doxid-namespacednnl_1_1ze__interop_1a536a52755503c79316d4bebb018b6394>`   :ref:`dnnl::ze_interop::get_driver(const engine &) <doxid-namespacednnl_1_1ze__interop_1aa47b43958425ae2a696bc999f524fce9>`    
\                    \                                                                                                                                                                           :ref:`dnnl::ze_interop::get_device(const engine &) <doxid-namespacednnl_1_1ze__interop_1a797d4b8181489c3da6781d2a7c90e347>`    
\                    \                                                                                                                                                                           :ref:`dnnl::ze_interop::get_context(const engine &) <doxid-namespacednnl_1_1ze__interop_1af1b31375baf405c74d4d4e223ed828af>`   
Stream               :ref:`dnnl::ze_interop::make_stream(const engine &, ze_command_list_handle_t, bool) <doxid-namespacednnl_1_1ze__interop_1ab290ce4074af3dcee977536d9424f25f>`                :ref:`dnnl::ze_interop::get_list(const stream &) <doxid-namespacednnl_1_1ze__interop_1a02cb113e0f7223ce995b6b8c3391c34e>`      
Memory (USM-based)   :ref:`dnnl::ze_interop::make_memory(const memory::desc &, const engine &, void \*) <doxid-namespacednnl_1_1ze__interop_1a43fd01076ba78315051dde2370f7feff>`                 :ref:`dnnl::memory::get_data_handle() <doxid-structdnnl_1_1memory_1a24aaca8359e9de0f517c7d3c699a2209>`                         
===================  ==========================================================================================================================================================================  =============================================================================================================================

Level Zero USM Interfaces for Memory Objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The memory model in Level Zero is based on Level Zero USM, which provides the ability to allocate and use memory in a uniform way on host and Level Zero devices.

To construct a oneDNN memory object, use the following interface:

* :ref:`dnnl::ze_interop::make_memory(const memory::desc &, const engine &, void \*) <doxid-namespacednnl_1_1ze__interop_1a43fd01076ba78315051dde2370f7feff>`
  
  Constructs a USM-based memory object. The ``handle`` could be one of special values :ref:`DNNL_MEMORY_ALLOCATE <doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>` or :ref:`DNNL_MEMORY_NONE <doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>`, or it could be a user-provided USM pointer.

Handling Dependencies
~~~~~~~~~~~~~~~~~~~~~

.. note:: 

   Only Level Zero in-order immediate command lists are supported.
   
   
Unlike the OpenCL API, the Level Zero API doesn't have a notion of retain/release mechanics when it comes to managing its objects. The object, once created, must be managed by some party. Thus, objects created by oneDNN are managed by oneDNN, and objects created by the user and passed to oneDNN are not managed by oneDNN, the library only stores references to objects. It means it's the user responsibility to manage the lifetime of objects created on their side while the library operates with them.

oneDNN provides two mechanisms to handle dependencies:

#. :ref:`dnnl::ze_interop::execute() <doxid-namespacednnl_1_1ze__interop_1a99e5491f5b45c2c44a1fab29d641400a>` interface
   
   This interface enables the user to pass dependencies between primitives using Level Zero events. In this case, the user is responsible for passing proper dependencies for every primitive execution.

#. In-order oneDNN stream
   
   oneDNN enables the user to create in-order streams in which submitted primitives are executed in the order they were submitted. Using in-order streams prevents possible read-before-write or concurrent read/write issues.

.. note:: 

   The access interfaces do not retain the Level Zero object. It is the user's responsibility to retain the returned Level Zero object if necessary.
   
   

.. note:: 

   Current version of API manages Level Zero events lifetime, and it's attached to the stream lifetime. All returned events after stream destruction become invalidated.
   
   

.. note:: 

   USM memory doesn't support retain/release Level Zero semantics. When constructing a oneDNN memory object using a user-provided USM pointer oneDNN doesn't own the provided memory. It's the user's responsibility to manage its lifetime.

