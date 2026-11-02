.. index:: pair: page; Grouped Memory Format for Variable-Size Batching
.. _doxid-dev_guide_grouped_mem:

Grouped Memory Format for Variable-Size Batching
================================================

.. note:: 

   This is an :ref:`experimental feature <doxid-dev_guide_experimental>`. Build oneDNN with ``ONEDNN_EXPERIMENTAL_GROUPED_MEMORY=ON`` to enable grouped memory support.
   
   
Grouped memory format handles data with variable-sized groups, commonly used in Mixture-of-Experts (MoE). Unlike regular batching where all tensors in the batch share the same size, grouped format allows one dimension to vary per group while keeping other dimensions uniform.

Memory Layout
~~~~~~~~~~~~~

Grouped format is described by two buffers and an indication of which dimension varies across groups (``variable_dim_idx``):

=======  =============================================  ===============  
Buffer   Description                                    Data Type        
=======  =============================================  ===============  
0        Concatenated data from all the groups          User-specified   
1        Cumulative offsets defining group boundaries   s32              
=======  =============================================  ===============

Variable Dimension
------------------

The ``variable_dim_idx`` parameter specifies which dimension varies in size across groups. For example, in a 2D tensor ``[M, K]`` :

* If ``variable_dim_idx=0``, dimension M varies per group while K stays constant.

* If ``variable_dim_idx=1``, dimension K varies per group while M stays constant.

The value specified for the variable dimension in the memory descriptor represents the total size summed across all groups. Individual group sizes are determined by the offsets buffer at execution time.

Offsets Buffer
--------------

The offsets buffer contains cumulative indices that define group boundaries. Each offset marks the end position of a group in the concatenated data buffer.

For groups with sizes ``[M_0, M_1, M_2, ..., M_{num_groups-1}]``, the offsets array is:

.. ref-code-block:: cpp

	[M_0, M_0+M_1, M_0+M_1+M_2, ..., sum(M)]

with length equal to ``num_groups``.

Note, that empty groups (size = 0) are valid and common in MoE when no tokens route to an expert. Consecutive offsets will be equal: ``offsets[g-1] == offsets[g]``.

Grouped Memory Descriptor API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create a grouped memory descriptor, use the following C++ API:

.. ref-code-block:: cpp

	static memory::desc memory::desc::grouped(
	    const dims &dims,              // Tensor dimensions (for variable dim use total size)
	    data_type dtype,               // Data type (e.g., f32, s8)
	    int variable_dim_idx,          // Index of dimension that varies per group
	    int num_groups);               // Number of groups

Creating and Using Grouped Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For instance, 2D grouped tensor ``[total_M, K]`` with ``variable_dim_idx = 0`` :

* Dimension ``0`` (``M``) is variable per group with ``total_M`` being the sum of all group sizes, number of groups is ``N``

* Dimension ``1`` (``K``) is constant across all groups

* Buffer 0 is concatenated data values ``[group0 | group1 | ... | groupN-1]``

* Buffer 1 is offsets array marking group boundaries

.. ref-code-block:: cpp

	// Example: MoE layer with 8 experts processing variable token counts
	// Routing result: tokens_per_expert = {800, 600, 700, 500, 650, 450, 550, 750}
	// Total sequence length: 5000 tokens
	// Offsets: {800, 1400, 2100, 2600, 3250, 3700, 4250, 5000}
	
	const int num_groups = 8;       // Number of expert networks
	const int total_tokens = 5000;  // Total tokens across all experts
	const int K = 512;              // Feature dimension (hidden size)
	
	// Create grouped memory descriptor
	auto grouped_md = memory::desc::grouped(
	    {total_M, K},              // dims: [total_M, K]
	    memory::data_type::f32,    // data type
	    0,                         // variable_dim_idx: dimension 0 varies
	    num_groups);               // number of groups
	
	// Prepare data buffers
	std::vector<float> values(total_M * K);
	std::vector<int32_t> offsets(num_groups);
	
	// Create memory object with both buffers
	memory grouped_mem(grouped_md, engine, {values.data(), offsets.data()});
	
	// Access individual buffers
	void* values_handle = grouped_mem.get_data_handle(0);
	void* offsets_handle = grouped_mem.get_data_handle(1);
	size_t values_size = grouped_mem.get_size(0);  // total_M * K * sizeof(float)
	size_t offsets_size = grouped_mem.get_size(1); // num_groups * sizeof(int32_t)

Examples
~~~~~~~~

See :ref:`MatMul with Grouped Encoding <doxid-matmul_grouped_cpp>` for an example of using grouped memory format with MatMul primitive.

