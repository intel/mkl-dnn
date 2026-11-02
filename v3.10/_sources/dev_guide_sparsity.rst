.. index:: pair: page; Sparse memory formats
.. _doxid-dev_guide_sparsity:

Sparse memory formats
=====================

API
~~~

oneDNN support format kind :ref:`dnnl::memory::format_kind::sparse <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa5dabba66ddc7b1e6f193ff73d3c55e94>` to describe sparse tensors. Sparse encoding (a.k.a. sparse format) is an enumeration type that specifies how data is encoded. Currently, oneDNN supports Compressed Sparse Row (CSR), Sorted Co-ordinate (COO) Sparse Format, and PACKED sparse encodings (:ref:`dnnl::memory::sparse_encoding::csr <doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a1f8c50db95e9ead5645e32f8df5baa7b>`, :ref:`dnnl::memory::sparse_encoding::coo <doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a03a6ff0db560bbdbcd4c86cd94b35971>`, :ref:`dnnl::memory::sparse_encoding::packed <doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966af59dcd306ec32930f1e78a1d82280b48>`) for CPU engine, and, only sorted COO (Co-ordinate Sparse Format) for GPU engine.

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

CSR Encoding
~~~~~~~~~~~~

.. ref-code-block:: cpp

	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` M = 4, N = 6;
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` nnz = 5;
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

Sorted COO Encoding
~~~~~~~~~~~~~~~~~~~

.. ref-code-block:: cpp

	using namespace :ref:`dnnl <doxid-namespacednnl>`;
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` M = 4, N = 6;
	const :ref:`memory::dim <doxid-structdnnl_1_1memory_1a281426f169daa042dcf5379c8fce21a9>` nnz = 5;
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

