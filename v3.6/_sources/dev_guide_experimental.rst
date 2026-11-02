.. index:: pair: page; Experimental features
.. _doxid-dev_guide_experimental:

Experimental features
=====================

To test aggressive performance optimizations that might affect accuracy or new API and functionality without an impact to regular users, oneDNN provides experimental features.

Build-time Controls
~~~~~~~~~~~~~~~~~~~

There are two kinds of experimental features:

#. Features that can be enabled at runtime with an environment variable. To enable such experimental features, the library should be built with a CMake option ``ONEDNN_EXPERIMENTAL=ON``. Each experimental feature has to be individually selected using environment variables.

#. Features that can be enabled only with a build time option. To enable such experimental features, the library should be built with a CMake option that corresponds to a particular feature.

Both kinds of experimental features can be enabled simultaneously.

Experimental features
~~~~~~~~~~~~~~~~~~~~~

=========================================  ====================================================================================================================================================================  
Environment variable                       Description                                                                                                                                                           
=========================================  ====================================================================================================================================================================  
ONEDNN_EXPERIMENTAL_BNORM_STATS_ONE_PASS   Calculate mean and variance in batch normalization(BN) in single pass ( `RFC <https://github.com/oneapi-src/oneDNN/tree/rfcs/rfcs/20210519-single-pass-bnorm>`__ ).   
=========================================  ====================================================================================================================================================================

===========================================  ===================================================================  
Build time option                            Description                                                          
===========================================  ===================================================================  
ONEDNN_EXPERIMENTAL_SPARSE                   Enable experimental API and functionality for sparse domain.         
ONEDNN_EXPERIMENTAL_UKERNEL                  Enable experimental microkernel APIs and functionalities.            
ONEDNN_EXPERIMENTAL_PROFILING                Enable experimental profiling API.                                   
ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND   Enable experimental graph compiler backend of the graph component.   
ONEDNN_EXPERIMENTAL_LOGGING                  Enable experimental logging support for oneDNN verbose mode.         
===========================================  ===================================================================

Features details
~~~~~~~~~~~~~~~~

ONEDNN_EXPERIMENTAL_SPARSE
--------------------------

This option extends the existing API and adds a new one to support sparse functionality in oneDNN.

API
+++

The main change is in oneDNN memory object semantics. Now, the memory object can have multiple underlying buffers. In the case of regular dense computations, the memory object always contains a single buffer. But in the case of sparse computations, the memory object always contains one buffer for values and an arbitrary number of additional buffers for metadata.

The underlying buffers are enumerated starting with 0, meaning that each buffer has its own number. The buffer with values always has index 0.

In most cases, the API that works with underlying buffers takes a buffer index. The exception is the API for creating a memory object. In that case, the API takes a vector of buffers. The order of the buffers in the vector matters and should correspond to the buffers' indices.

oneDNN also introduces a new format kind :ref:`dnnl::memory::format_kind::sparse <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa5dabba66ddc7b1e6f193ff73d3c55e94>`. Sparse encoding (a.k.a. sparse format) is an enumeration type that specifies how data is encoded. Currently, oneDNN supports Compressed Sparse Row (CSR), Sorted Co-ordinate (COO) Sparse Format, and PACKED sparse encodings (:ref:`dnnl::memory::sparse_encoding::csr <doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a1f8c50db95e9ead5645e32f8df5baa7b>`, :ref:`dnnl::memory::sparse_encoding::coo <doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a03a6ff0db560bbdbcd4c86cd94b35971>`, :ref:`dnnl::memory::sparse_encoding::packed <doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966af59dcd306ec32930f1e78a1d82280b48>`) for CPU engine, and, only sorted COO (Co-ordinate Sparse Format) for GPU engine.

The memory descriptor has dedicated static member functions for creating memory descriptors for different sparse encodings.

Each encoding defines the number and meaning of the buffers.

================  ============================================================================  
Sparse encoding   Buffers                                                                       
================  ============================================================================  
CSR               0 - values, 1 - indices, 2 - pointers                                         
Sorted COO        0 - values, 1 to *ndims* - indices ( *ndims* - number of tensor dimensions)   
PACKED            The meaning and content are unspecified                                       
================  ============================================================================

The pseudocode below demonstrates how to create a memory object for the CSR and COO sparse encodings and use the new API to work with the underlying handles.

###### CSR Encoding:

.. ref-code-block:: cpp

	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` M = 4, N = 6;
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` nnz = 5;
	const auto values_dt = :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`;
	const auto indices_dt = :ref:`memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`;
	const auto pointers_dt = :ref:`memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`;
	
	// Create a memory descriptor for CSR sparse encoding.
	const auto csr_md = :ref:`memory::desc::csr <doxid-structdnnl_1_1memory_1_1desc_1a7fe93a14828506260740fb439eaf6ed4>`(
	        {M, N}, // Dimensions
	        values_dt, // Data type of values
	        nnz, // Number of non-zero entries
	        indices_dt, // Data type of indices (metadata)
	        pointers_dt); // Data type of pointers (metadata)
	
	// A sparse matrix represented in the CSR format.
	std::vector<float> csr_values = {2.5f, 1.5f, 1.5f, 2.5f, 2.0f};
	std::vector<int32_t> csr_indices = {0, 2, 0, 5, 1};
	std::vector<int32_t> csr_pointers = {0, 1, 2, 4, 5, 5};
	
	// Create a memory object for the given buffers with values and metadata.
	:ref:`memory <doxid-structdnnl_1_1memory>` csr_mem(csr_md, :ref:`engine <doxid-structdnnl_1_1engine>`, {
	    csr_values.data(), // Buffer with values
	    csr_indices.data(), // Buffer with indices (metadata)
	    csr_pointers.data() // Buffer with pointers (metadata)
	    });
	
	const auto values_sz = csr_mem.get_size(0);
	const auto indices_sz = csr_mem.get_size(1);
	const auto pointers_sz = csr_mem.get_size(2);
	
	assert(values_sz == csr_values.size() * sizeof(float));
	assert(indices_sz == csr_indices.size() * sizeof(int32_t));
	assert(pointers_sz == csr_pointers.size() * sizeof(int32_t));
	
	void *values_handle = csr_mem.get_data_handle(0);
	void *indices_handle = csr_mem.get_data_handle(1);
	void *pointers_handle = csr_mem.get_data_handle(2);
	
	assert(values_handle == (void *)csr_values.data());
	assert(indices_handle == (void *)csr_indices.data());
	assert(pointers_handle == (void *)csr_pointers.data());

###### Sorted COO Encoding:

.. ref-code-block:: cpp

	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` M = 4, N = 6;
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` nnz = 5;
	const auto values_dt = :ref:`memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`;
	const auto indices_dt = :ref:`memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`;
	
	// Create a memory descriptor for COO sparse encoding.
	const auto coo_md = :ref:`memory::desc::coo <doxid-structdnnl_1_1memory_1_1desc_1a231f8a88d9f90f50ea2ae86c00182128>`(
	        {M, N}, // Dimensions
	        values_dt, // Data type of values
	        nnz, // Number of non-zero entries
	        indices_dt); // Data type of indices (metadata)
	
	// A sparse matrix represented in the COO format.
	std::vector<float> coo_values = {2.5f, 1.5f, 1.5f, 2.5f, 2.0f};
	std::vector<int32_t> coo_row_indices = {0, 1, 2, 2, 3};
	std::vector<int32_t> coo_col_indices = {0, 2, 0, 5, 1};
	
	// Create a memory object for the given buffers with values and metadata.
	:ref:`memory <doxid-structdnnl_1_1memory>` coo_mem(coo_md, :ref:`engine <doxid-structdnnl_1_1engine>`, {
	    coo_values.data(), // Buffer with values
	    coo_row_indices.data(), // Buffer with row indices (metadata)
	    coo_col_indices.data() // Buffer with column indices (metadata)
	    });
	
	const auto values_sz = coo_mem.get_size(0);
	const auto indices_sz = coo_mem.get_size(1);
	
	assert(values_sz == coo_values.size() * sizeof(float));
	assert(indices_sz == coo_row_indices.size() * sizeof(int32_t));
	assert(indices_sz == coo_col_indices.size() * sizeof(int32_t));
	
	void *values_handle = coo_mem.get_data_handle(0);
	void *row_indices_handle = coo_mem.get_data_handle(1);
	void *col_indices_handle = coo_mem.get_data_handle(2);
	
	assert(values_handle == (void *)coo_values.data());
	assert(row_indices_handle == (void *)coo_row_indices.data());
	assert(col_indices_handle == (void *)coo_col_indices.data());

A memory descriptor created for the sparse encoding PACKED cannot be used to create a memory object. It can only be used to create a primitive descriptor to query the actual memory descriptor (similar to the format tag ``any``).

Primitives
++++++++++

Matrix Multiplication
*********************

This option enables the matmul primitive that can work with sparse input tensors.

CSR encoding
^^^^^^^^^^^^

Supported only for the CPU engine. Only one of the input tensors can be sparse. The output tensor is always dense.

The following data type combinations are supported:

==========================  ========  
Values (src, weight, dst)   Indices   
==========================  ========  
f16, f16, f16               s32       
f32, f32, f32               s32       
==========================  ========

The following format tags are supported for dense input/output tensors:

* ab

See the example :ref:`here <doxid-cpu_matmul_csr_cpp>`.

Benchdnn can be used to test matmul with a CSR input tensor as follows: ``./benchdnn --matmul --encoding=csr+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128``

For the case above, the number of non-zero elements for the source tensor is calculated as max(4 \* 1000000 \* (1 - 0.99), 1).

COO encoding
^^^^^^^^^^^^

Supported only for the CPU and GPU engines. Only one of the input tensors can be sparse. The output tensor is always dense.

The following data type combinations are supported:

==========================  ========  
Values (src, weight, dst)   Indices   
==========================  ========  
f16, f16, f16               s32       
f32, f32, f32               s32       
==========================  ========

The following format tags are supported for dense weights tensor:

* ab

* ba

The following format tags are supported for dense destination tensor:

* ab

See the example :ref:`here <doxid-cpu_matmul_coo_cpp>`.

Benchdnn can be used to test matmul with a COO input tensor as follows: ``./benchdnn --matmul --encoding=coo+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128``

For the case above, the number of non-zero elements for the source tensor is calculated as max(4 \* 1000000 \* (1 - 0.99), 1).

PACKED encoding
^^^^^^^^^^^^^^^

Only the weights tensor is allowed to be sparse. The other tensors are always dense.

In general, it is expected that all matmul related functionality (e.g. post-ops, scales, zero-points, etc) that is supported for the dense weights should also work for the sparse weights.

Currently, matmul has the following limitations for the PACKED encoding:

* Supported only for the CPU engine

* Only Intel Advanced Matrix Extensions (Intel AMX) instruction set architecture (ISA) is supported

* Only ``s8`` data type for the weights is supported

* Only 1 batch dimension is supported

See the example :ref:`here <doxid-cpu_matmul_weights_compression_cpp>`.

Benchdnn can be used to test matmul with the PACKED weights tensor as follows: ``./benchdnn --matmul --dt=s8:s8:s32 --encoding=:packed+0.99: 3x512x1024:1x1024x512``

For the case above, the number of non-zero elements for the weights tensor is calculated as max(1024 \* 512 \* (1 - 0.99), 1).

Reorder
*******

Currently, there is only one reorder for packing a dense tensor, i.e. converting a dense tensor that is in ``ab`` format to a sparse tensor that is encoded with the ``PACKED`` encoding.

In general, it is expected that all reorder-related functionality (e.g. scales, zero-points, etc) that is supported for the dense destination tensor should also work for the sparse one.

Common Limitations
++++++++++++++++++

* The interoperability API to get/set data handles is not supported. Use the runtime agnostic API to do that.

* Sparse memory and memory descriptor can only be used with the Matrix Multiplication and Reorder primitives.

ONEDNN_EXPERIMENTAL_UKERNEL
---------------------------

This option enables a new set of CPU-only APIs to support block-level functionalities. By composing these low-level, sequential operations, users can implement their own custom operations/fusions, and tailor blocking/threading logic to their applications.

More details on this API are available in the :ref:`Microkernel APIs <doxid-dev_guide_ukernel_basic_concepts>` section".

ONEDNN_EXPERIMENTAL_PROFILING
-----------------------------

This option enables profiling API that can be used to query different profiling data.

There are two ways to use the profiling capabilities:

* Create a queue with enabled profiling capabilities and use the interoperability API to create a oneDNN stream with the queue. The library will identify that the queue supports profiling and will collect profiling data

* Create a oneDNN stream using runtime agnostic API and enable profiling capabilities using the stream flag ``stream::flags::profiling``

Below is a pseudo-code that demonstrates the profiling API usage with a user-provided queue.

.. ref-code-block:: cpp

	:ref:`dnnl::engine <doxid-structdnnl_1_1engine>` :ref:`engine <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1aad1943a9fd6d3d7ee1e6af41a5b0d3e7>`(engine::kind::gpu, 0);
	// Create a queue with enabled profiling mode.
	cl_command_queue ocl_queue {};
	cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
	ocl_queue = clCreateCommandQueueWithProperties(ocl_interop::get_context(engine),
	    ocl_interop::get_device(engine), props, ...);
	// Create dnnl::stream with the queue.
	:ref:`dnnl::stream <doxid-structdnnl_1_1stream>` stream = ocl_interop::make_stream(engine, ocl_queue);
	// Create a convolution primitive ... //
	// Reset profiler's state.
	:ref:`dnnl::reset_profiling <doxid-group__dnnl__api__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e>`(stream);
	// Enqueue same primitive twice and wait for both executions to complete.
	conv_prim.execute(stream, ...)
	conv_prim.execute(stream, ...)
	stream.:ref:`wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>`();
	// Query profiling data. The vector size will be equal to the number of
	// executions happened on the stream since the last `dnnl::reset_profiling`
	// call.
	std::vector<uint64_t> nsecs = :ref:`dnnl::get_profiling_data <doxid-group__dnnl__api__profiling_1ga0dc451b94cbeacb7a5e0c73c3071ee4e>`(stream, profiling_data_kind::time);
	assert(nsecs.size() == 2);
	// Reset profiler's state.
	:ref:`dnnl::reset_profiling <doxid-group__dnnl__api__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e>`(stream);

.. warning:: 

   * When the stream is created with enabled profiling capabilities it will collect profiling data for each primitive execution. It is the user's responsibility to reset the profiler's state to avoid consuming all memory resources in the system.
   
   


Limitations
+++++++++++

* Only GPU engines with OpenCL and SYCL runtimes are supported

* Only Intel vendor is supported for SYCL runtime

* Out-of-order queue is not supported

ONEDNN_EXPERIMENTAL_GRAPH_COMPILER_BACKEND
------------------------------------------

This option extends the coverage scope of the graph API to cover larger fusion patterns apart from primitive patterns. Refer to :ref:`Graph Compiler <doxid-dev_guide_graph_compiler>` for more details.

.. warning:: 

   * Enabling some experimental features does not guarantee that the library will utilize them
   
   * Enabling some experimental features might change the accuracy of oneDNN primitives
   
   


ONEDNN_EXPERIMENTAL_LOGGING
---------------------------

This option introduces logging support in oneDNN which allows one to save the verbose outputs generated by oneDNN applications to user-specified logfiles. By setting ``ONEDNN_EXPERIMENTAL_LOGGING=ON``, a logging mechanism is built into oneDNN using the third-party `spdlog <https://github.com/gabime/spdlog>`__ library. Logging can then be enabled while running different applications by specifying the logfile path using ``ONEDNN_VERBOSE_LOGFILE`` :

.. ref-code-block:: cpp

	$ ONEDNN_VERBOSE=all ONEDNN_VERBOSE_LOGFILE=./logs/cnn_test_logger.log ./examples/cnn-inference-f32-cpp

When logging is enabled while running an application, it also requires that the verbose mode be enabled for the run using ``ONEDNN_VERBOSE``. When no logfile is specified, logging is automatically disabled and the verbose output is printed only to the console. For the specified logfile path, the logger creates the base directory and the logfile if they do not already exist. When the specified logfile already exists, the output is appended to the existing file until it reaches the maximum file size. Note: Multiple instances using the same filepath for ``DNNL_VERBOSE_LOGFILE`` will write to the same file during the API run. The spdlog mechanism supports handling multiple instances concurrently if they write to the same logfile but the expectation is to specify different logfiles for different instances via the runtime variables.

By default, logging is disabled in oneDNN and any verbose output generated by oneDNN is printed only to ``stdout``. The API is executed as a rotating lazy logger with a file size specified by ``ONEDNN_VERBOSE_LOGFILE_SIZE(=1024*1024*50)``. When logging is enabled, the user has the option to print verbose output to both ``stdout`` and the logfile by setting ``ONEDNN_VERBOSE_LOG_WITH_CONSOLE=1``. The runtime controls for oneDNN logging are listed as follows:

================================  ====================================================  
Runtime variable                  Description                                           
================================  ====================================================  
ONEDNN_VERBOSE_LOGFILE            Enables verbose logging and specifies logfile path.   
ONEDNN_VERBOSE_LOGFILE_SIZE       Specifies maximum size for the logfile.               
ONEDNN_VERBOSE_NUM_LOGFILES       Number of rotating logfiles for the logger.           
ONEDNN_VERBOSE_LOG_WITH_CONSOLE   Enables printing to both stdout and the logfile.      
================================  ====================================================

