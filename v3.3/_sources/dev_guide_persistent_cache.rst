.. index:: pair: page; Persistent Cache
.. _doxid-dev_guide_persistent_cache:

Persistent Cache
================

Creating some oneDNN abstractions can be costly for various reasons. Usually, oneDNN mitigates that overhead by caching such objects but the cache has no effect when the objects are created for the first time. For some applications it can be critical to reduce that overhead.

oneDNN provides an API that can be used to create a persistent cache for such oneDNN abstractions. User can use that API to obtain a cache blob ID and a cache blob to use them as a key and value respectively.

.. note:: 

   Content and size of the cache blob ID and cache blob objects are not specified.
   
   

.. note:: 

   oneDNN version and git commit hash (:ref:`dnnl_version_t::hash <doxid-structdnnl__version__t_1a1c2a7e8b3f26ed376b537dd511feccad>`) affect equality of the cache blob IDs. That is, the queried cache blob ID will be different for different oneDNN versions and git commit hashes.
   
   

.. warning:: 

   The git commit hash may not be available if the git package was not found during a CMake call. In this case, the cache blob ID will be the same for different hashes. This may result in fetching a wrong cache blob from persistent cache.
   
   


Primitive
~~~~~~~~~

* The cache blob ID can be obtained via :ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`dnnl::primitive_desc_base::get_cache_blob_id <doxid-structdnnl_1_1primitive__desc__base_1a435862df4d543eb8296424880212b22d>`

* The cache blob can be obtained via :ref:`dnnl::primitive::get_cache_blob <doxid-group__dnnl__api__primitives__common_1ga8bf59b36c745ee8eaec9d0dd22e266e9>`

* Each primitive class provides a constructor that takes the cache blob along with the primitive descriptor.

Relation to Primitive Cache
---------------------------

In the case when a primitive is created from a cache blob and the identical primitive is present in the primitive cache the one from primitive cache will be returned to the user, and the given cache blob will not be used. Otherwise, the cache blob will be used to speed up the primitive creation. The information about how the primitive was created (``cache_miss``, ``cache_hit`` or ``from_cache_blob``) is part of the verbose output for verbose level 2 (:ref:`Verbose Mode <doxid-dev_guide_verbose>`).

API Usage Example
-----------------

The following pseudo-code demonstrates a simple example of persistent cache implementation for primitives using the oneDNN API:

.. ref-code-block:: cpp

	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	{
	    :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>` conv_pd(desc, attr, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    :ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>` conv(conv_pd);
	
	    std::vector<uint8_t> key = conv_pd.get_cache_blob_id();
	    std::vector<uint8_t> value = conv.get_cache_blob();
	    store_cache_blob_on_disk(key, value);
	}
	
	{
	    :ref:`convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>` conv_pd(desc, attr, :ref:`engine <doxid-structdnnl_1_1engine>`);
	    std::vector<uint8_t> key = conv_pd.get_cache_blob_id();
	    std::vector<uint8_t> value = load_cache_blob_from_disk(key);
	    :ref:`convolution_forward <doxid-structdnnl_1_1convolution__forward>` conv_from_cache_blob(conv_pd, value);
	}

Engine
~~~~~~

* The cache blob ID can be obtained via :ref:`dnnl::ocl_interop::get_engine_cache_blob_id <doxid-namespacednnl_1_1ocl__interop_1a969a55a4a7d84dee549ebd2c3bcc2518>`

* The cache blob can obtained via :ref:`dnnl::ocl_interop::get_engine_cache_blob <doxid-namespacednnl_1_1ocl__interop_1a55f93340fcf71df592d7a2c903513823>`

* Engine can be created with the cache blob via :ref:`dnnl::ocl_interop::make_engine(cl_device_id, cl_context, const std::vector\<uint8_t> &) <doxid-namespacednnl_1_1ocl__interop_1a4de69784af3621cf7b83041a2a9de071>`

API Usage Example
-----------------

The following pseudo-code demonstrates a simple example of persistent cache implementation for OpenCL engines using the oneDNN API:

.. ref-code-block:: cpp

	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	
	{
	    cl_device_id device = ...;
	    cl_context context = ...;
	
	    :ref:`engine <doxid-structdnnl_1_1engine>` ocl_engine = :ref:`ocl_interop::make_engine <doxid-namespacednnl_1_1ocl__interop_1a4de69784af3621cf7b83041a2a9de071>`(device, context);
	    std::vector<uint8_t> key = :ref:`get_engine_cache_blob_id <doxid-namespacednnl_1_1ocl__interop_1a969a55a4a7d84dee549ebd2c3bcc2518>`(:ref:`ocl_interop::get_device <doxid-namespacednnl_1_1ocl__interop_1a37ef1ccb75d09063ed049076fb23b927>`(ocl_engine));
	    std::vector<uint8_t> value = :ref:`get_engine_cache_blob <doxid-namespacednnl_1_1ocl__interop_1a55f93340fcf71df592d7a2c903513823>`(ocl_engine);
	
	    store_cache_blob_on_disk(key, value);
	}
	
	{
	    cl_device_id device = ...;
	    cl_context context = ...;
	
	    std::vector<uint8_t> key = :ref:`get_engine_cache_blob_id <doxid-namespacednnl_1_1ocl__interop_1a969a55a4a7d84dee549ebd2c3bcc2518>`(device);
	    std::vector<uint8_t> value = load_cache_blob_from_disk(key);
	    :ref:`engine <doxid-structdnnl_1_1engine>` ocl_engine = :ref:`ocl_interop::make_engine <doxid-namespacednnl_1_1ocl__interop_1a4de69784af3621cf7b83041a2a9de071>`(device, context, value);
	}

Limitations
~~~~~~~~~~~

* The API is implemented for the OpenCL runtime only. For CPU engine kind and other runtimes the library will return :ref:`dnnl_unimplemented <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>` in the case of the C API or throw a corresponding :ref:`dnnl::error <doxid-structdnnl_1_1error>` exception in the case of the C++ API.

* Currently, the library cannot differentiate cache blob created for devices that have different stepping therefore the cache blob can be safely used only on the system where it was created.

