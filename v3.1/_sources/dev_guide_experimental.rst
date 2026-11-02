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

===========================  =============================================================  
Build time option            Description                                                    
===========================  =============================================================  
ONEDNN_EXPERIMENTAL_SPARSE   Enable experimental API and functionality for sparse domain.   
===========================  =============================================================

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

oneDNN also introduces a new format kind :ref:`dnnl::memory::format_kind::sparse <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa5dabba66ddc7b1e6f193ff73d3c55e94>`, sparse encoding. Sparse encoding (a.k.a. sparse format) is an enumeration type that specifies how data is encoded. Currently, oneDNN only supports CSR (Compressed sparse row) sparse encoding (:ref:`dnnl::memory::sparse_encoding::csr <doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a1f8c50db95e9ead5645e32f8df5baa7b>`).

The memory descriptor has dedicated static member functions for creating memory descriptors for different sparse encodings.

Each encoding defines the number and meaning of the buffers.

================  ======================================  
Sparse encoding   Buffers                                 
================  ======================================  
CSR               0 - values, 1 - indices, 2 - pointers   
================  ======================================

Pseudo-code with creating a memory object for CSR sparse encoding.

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

Primitives
++++++++++

The option enables a matmul primitive that can work with sparse input tensors. Only one of the input tensors is allowed to be sparse. The output tensor is always dense.

The following data types combinations are supported:

=======  ========  =========  
Values   Indices   Pointers   
=======  ========  =========  
f32      s32       s32        
=======  ========  =========

The following sparse encodings are supported:

* CSR

The following format tags are supported for dense input/output tensors:

* ab

Benchdnn can be used to test the sparse matmul as follows: ``./benchdnn --matmul --encoding=csr+0.99:: --wtag=ab --dtag=ab 4x1000000:1000000x128``

For the case above, the number of non-zero elements for the source tensor is calculated as max(4 \* 1000000 \* (1 - 0.99)), 1).

Limitations
+++++++++++

* This functionality is not supported for SYCL and OpenCL runtimes

* The interoperability API for sparse memory is not provided

* Sparse memory and memory descriptor can only be used with the Matrix Multiplication primitive

* Sparse memory can be created only for a CPU engine

.. warning:: 

   * Enabling experimental features does not guarantee that the library will utilize them
   
   * Enabling experimental features might change the accuracy of oneDNN primitives

