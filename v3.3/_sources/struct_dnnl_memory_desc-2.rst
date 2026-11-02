.. index:: pair: struct; dnnl::memory::desc
.. _doxid-structdnnl_1_1memory_1_1desc:

struct dnnl::memory::desc
=========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

A memory descriptor. :ref:`More...<details-structdnnl_1_1memory_1_1desc>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>
	
	struct desc: public :ref:`dnnl::handle<doxid-structdnnl_1_1handle>`
	{
		// construction
	
		:ref:`desc<doxid-structdnnl_1_1memory_1_1desc_1a2a12f9b43aae8c214d695b321b543b5c>`();
	
		:ref:`desc<doxid-structdnnl_1_1memory_1_1desc_1a03f068d5a2e5b2d043d2f203717a0ceb>`(
			const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims,
			:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type,
			:ref:`format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` aformat_tag,
			bool allow_empty = false
			);
	
		:ref:`desc<doxid-structdnnl_1_1memory_1_1desc_1ac0471538db2d230492a07356929c859c>`(
			const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims,
			:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type,
			const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& strides,
			bool allow_empty = false
			);
	
		:ref:`desc<doxid-structdnnl_1_1memory_1_1desc_1a1aa2c1e9b4933160d1f282cd13844a36>`(:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>` md);

		// methods
	
		desc :ref:`submemory_desc<doxid-structdnnl_1_1memory_1_1desc_1a7de2abef3b34e94c5dfa16e1fc3f3aab>`(
			const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims,
			const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& offsets,
			bool allow_empty = false
			) const;
	
		desc :ref:`reshape<doxid-structdnnl_1_1memory_1_1desc_1ab95a6fbd16dd8b7c421611d39d49fe3f>`(const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims, bool allow_empty = false) const;
		desc :ref:`permute_axes<doxid-structdnnl_1_1memory_1_1desc_1a70e831b990a91d61bc44e57f6e895a83>`(const std::vector<int>& permutation, bool allow_empty = false) const;
		int :ref:`get_ndims<doxid-structdnnl_1_1memory_1_1desc_1a2d41c66b694a41a9d124905c140888cf>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_padded_dims<doxid-structdnnl_1_1memory_1_1desc_1acd2d72575ac8ff954b241c569fba1a5b>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_padded_offsets<doxid-structdnnl_1_1memory_1_1desc_1a51b3c0ec2c9a9095b0522140e5ad01fb>`() const;
		:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :ref:`get_submemory_offset<doxid-structdnnl_1_1memory_1_1desc_1a1ea0b7976edfe0833e6434799970e702>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_strides<doxid-structdnnl_1_1memory_1_1desc_1aa4b72acda1a8c929cdc6829e715930f4>`() const;
		int :ref:`get_inner_nblks<doxid-structdnnl_1_1memory_1_1desc_1a984dc7374db3f61618f268fb439c7d48>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_inner_blks<doxid-structdnnl_1_1memory_1_1desc_1a9fe65ce60f80aba31f9325a1ebc91165>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_inner_idxs<doxid-structdnnl_1_1memory_1_1desc_1ad8972312d2884584a676e74a2dcdd1e8>`() const;
		int :ref:`get_num_handles<doxid-structdnnl_1_1memory_1_1desc_1ad1f0ad6584fa547dba0dd72d54b9162b>`() const;
		:ref:`dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` :ref:`get_nnz<doxid-structdnnl_1_1memory_1_1desc_1af15b390ff5f200b75c8bd606c6d10794>`() const;
		:ref:`memory::sparse_encoding<doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966>` :ref:`get_sparse_encoding<doxid-structdnnl_1_1memory_1_1desc_1a056c9425560d09e71b154adc611f5e4b>`() const;
		:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` :ref:`get_data_type<doxid-structdnnl_1_1memory_1_1desc_1aada0dc594d12f25331d4d7cf84c08e75>`(int index = 0) const;
		:ref:`memory::format_kind<doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105f>` :ref:`get_format_kind<doxid-structdnnl_1_1memory_1_1desc_1a3e107c30cab4b49c1ea991d09681dfcd>`() const;
		:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` :ref:`get_dims<doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830>`() const;
		size_t :ref:`get_size<doxid-structdnnl_1_1memory_1_1desc_1abfa095ac138d4d2ef8efd3739e343f08>`(int index = 0) const;
		bool :ref:`is_zero<doxid-structdnnl_1_1memory_1_1desc_1aa162a1ba5621a799c8a909c726f021a2>`() const;
		bool :ref:`operator ==<doxid-structdnnl_1_1memory_1_1desc_1a9d623dab6f8a8ebc34b0da95814e3728>` (const desc& other) const;
		bool :ref:`operator !=<doxid-structdnnl_1_1memory_1_1desc_1a5a1bc8d3b88a2c304fd9535ea45fbb72>` (const desc& other) const;
	
		static desc :ref:`csr<doxid-structdnnl_1_1memory_1_1desc_1a7fe93a14828506260740fb439eaf6ed4>`(
			const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims,
			:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type,
			:ref:`dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` nnz,
			:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` index_dt,
			:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` pointer_dt,
			bool allow_empty = false
			);
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

.. _details-structdnnl_1_1memory_1_1desc:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A memory descriptor.

Construction
------------

.. index:: pair: function; desc
.. _doxid-structdnnl_1_1memory_1_1desc_1a2a12f9b43aae8c214d695b321b543b5c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	desc()

Constructs a zero (empty) memory descriptor.

Such a memory descriptor can be used to indicate absence of an argument.

.. index:: pair: function; desc
.. _doxid-structdnnl_1_1memory_1_1desc_1a03f068d5a2e5b2d043d2f203717a0ceb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	desc(
		const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims,
		:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type,
		:ref:`format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` aformat_tag,
		bool allow_empty = false
		)

Constructs a memory descriptor.

.. note:: 

   The logical order of dimensions corresponds to the ``abc...`` format tag, and the physical meaning of the dimensions depends both on the primitive that would operate on this memory and the operation context.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- adims

		- Tensor dimensions.

	*
		- adata_type

		- Data precision/type.

	*
		- aformat_tag

		- Memory format tag.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case a zero memory descriptor will be constructed. This flag is optional and defaults to false.

.. index:: pair: function; desc
.. _doxid-structdnnl_1_1memory_1_1desc_1ac0471538db2d230492a07356929c859c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	desc(
		const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims,
		:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type,
		const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& strides,
		bool allow_empty = false
		)

Constructs a memory descriptor by strides.

.. note:: 

   The logical order of dimensions corresponds to the ``abc...`` format tag, and the physical meaning of the dimensions depends both on the primitive that would operate on this memory and the operation context.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- adims

		- Tensor dimensions.

	*
		- adata_type

		- Data precision/type.

	*
		- strides

		- Strides for each dimension.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case a zero memory descriptor will be constructed. This flag is optional and defaults to false.

.. index:: pair: function; desc
.. _doxid-structdnnl_1_1memory_1_1desc_1a1aa2c1e9b4933160d1f282cd13844a36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	desc(:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>` md)

Construct a memory descriptor from a C API :ref:`dnnl_memory_desc_t <doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>` handle.

The resulting handle is not weak and the C handle will be destroyed during the destruction of the C++ object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- md

		- The C API memory descriptor.

Methods
-------

.. index:: pair: function; submemory_desc
.. _doxid-structdnnl_1_1memory_1_1desc_1a7de2abef3b34e94c5dfa16e1fc3f3aab:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	desc submemory_desc(
		const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims,
		const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& offsets,
		bool allow_empty = false
		) const

Constructs a memory descriptor for a region inside an area described by this memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- adims

		- Sizes of the region.

	*
		- offsets

		- Offsets to the region from the encompassing memory object in each dimension.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case a zero memory descriptor will be returned. This flag is optional and defaults to false.



.. rubric:: Returns:

A memory descriptor for the region.

.. index:: pair: function; reshape
.. _doxid-structdnnl_1_1memory_1_1desc_1ab95a6fbd16dd8b7c421611d39d49fe3f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	desc reshape(const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims, bool allow_empty = false) const

Constructs a memory descriptor by reshaping an existing one.

The new memory descriptor inherits the data type. This operation is valid only for memory descriptors that have format_kind set to :ref:`dnnl::memory::format_kind::blocked <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa61326117ed4a9ddf3f754e71e119e5b3>` or :ref:`dnnl::memory::format_kind::any <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa100b8cad7cf2a56f6df78f171f97a1ec>`.

The operation ensures that the transformation of the physical memory format corresponds to the transformation of the logical dimensions. If such transformation is impossible, the function either throws an exception (default) or returns a zero memory descriptor depending on the ``allow_empty`` flag.

The reshape operation can be described as a combination of the following basic operations:

#. Add a dimension of size ``1``. This is always possible.

#. Remove a dimension of size ``1``. This is possible only if the dimension has no padding (i.e. ``padded_dims[dim] == dims[dim] && dims[dim] == 1``).

#. Split a dimension into multiple ones. This is possible only if the product of all tensor dimensions stays constant and the dimension being split does not have padding (i.e. ``padded_dims[dim] = dims[dim]``).

#. Join multiple consecutive dimensions into a single one. As in the cases above, this requires that the dimensions do not have padding and that the memory format is such that in physical memory these dimensions are dense and have the same order as their logical counterparts. This also assumes that these dimensions are not blocked.
   
   * Here, 'dense' means: ``stride for dim[i] == (stride for dim[i + 1]) * dim[i + 1]``;
   
   * And 'same order' means: ``i < j`` if and only if ``stride for dim[j] <= stride for dim[i]``.

.. warning:: 

   Some combinations of physical memory layout and/or offsets or dimensions may result in a failure to make a reshape.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- adims

		- New dimensions. The product of dimensions must remain constant.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case a zero memory descriptor will be returned. This flag is optional and defaults to false.



.. rubric:: Returns:

A new memory descriptor with new dimensions.

.. index:: pair: function; permute_axes
.. _doxid-structdnnl_1_1memory_1_1desc_1a70e831b990a91d61bc44e57f6e895a83:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	desc permute_axes(const std::vector<int>& permutation, bool allow_empty = false) const

Constructs a memory descriptor by permuting axes in an existing one.

The physical memory layout representation is adjusted accordingly to maintain the consistency between the logical and physical parts of the memory descriptor. The new memory descriptor inherits the data type.

The new memory descriptor inherits the data type. This operation is valid only for memory descriptors that have format_kind set to :ref:`dnnl::memory::format_kind::blocked <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa61326117ed4a9ddf3f754e71e119e5b3>` or :ref:`dnnl::memory::format_kind::any <doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa100b8cad7cf2a56f6df78f171f97a1ec>`.

The logical axes will be permuted in the following manner:

.. ref-code-block:: cpp

	for (i = 0; i < get_ndims(); i++)
	    new_desc.dims()[permutation[i]] = dims()[i];

Example:

.. ref-code-block:: cpp

	std::vector<int> permutation = {1, 0}; // swap the first and
	                                       // the second axes
	dnnl::memory::desc in_md(
	        {2, 3}, data_type, memory::format_tag::ab);
	dnnl::memory::desc expect_out_md(
	        {3, 2}, data_type, memory::format_tag::ba);
	
	assert(in_md.permute_axes(permutation) == expect_out_md);



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- permutation

		- Axes permutation.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case a zero memory descriptor will be returned. This flag is optional and defaults to false.



.. rubric:: Returns:

A new memory descriptor with new dimensions.

.. index:: pair: function; get_ndims
.. _doxid-structdnnl_1_1memory_1_1desc_1a2d41c66b694a41a9d124905c140888cf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int get_ndims() const

Returns a number of dimensions of the memory descriptor.



.. rubric:: Returns:

A number of dimensions.

.. index:: pair: function; get_padded_dims
.. _doxid-structdnnl_1_1memory_1_1desc_1acd2d72575ac8ff954b241c569fba1a5b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_padded_dims() const

Returns padded dimensions of the memory descriptor.



.. rubric:: Returns:

A copy of the padded dimensions vector.

.. index:: pair: function; get_padded_offsets
.. _doxid-structdnnl_1_1memory_1_1desc_1a51b3c0ec2c9a9095b0522140e5ad01fb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_padded_offsets() const

Returns padded offsets of the memory descriptor.



.. rubric:: Returns:

A copy of the padded offsets vector.

.. index:: pair: function; get_submemory_offset
.. _doxid-structdnnl_1_1memory_1_1desc_1a1ea0b7976edfe0833e6434799970e702:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` get_submemory_offset() const

Returns a submemory offset of the memory descriptor.



.. rubric:: Returns:

A submemory offset.

.. index:: pair: function; get_strides
.. _doxid-structdnnl_1_1memory_1_1desc_1aa4b72acda1a8c929cdc6829e715930f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_strides() const

Returns strides of the memory descriptor.

.. note:: 

   This API is only applicable to memory descriptors with format kind :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>`.



.. rubric:: Returns:

A copy of the strides vector.

An empty :ref:`dnnl::memory::dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` if the memory descriptor does not have strides.

.. index:: pair: function; get_inner_nblks
.. _doxid-structdnnl_1_1memory_1_1desc_1a984dc7374db3f61618f268fb439c7d48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int get_inner_nblks() const

Returns a number of inner blocks of the memory descriptor.

.. note:: 

   This API is only applicable to memory descriptors with format kind :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>`.



.. rubric:: Returns:

A number of inner blocks.

.. index:: pair: function; get_inner_blks
.. _doxid-structdnnl_1_1memory_1_1desc_1a9fe65ce60f80aba31f9325a1ebc91165:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_inner_blks() const

Returns inner blocks of the memory descriptor.

.. note:: 

   This API is only applicable to memory descriptors with format kind :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>`.



.. rubric:: Returns:

A copy of the inner blocks vector.

An empty :ref:`dnnl::memory::dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` if the memory descriptor does not have inner blocks.

.. index:: pair: function; get_inner_idxs
.. _doxid-structdnnl_1_1memory_1_1desc_1ad8972312d2884584a676e74a2dcdd1e8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_inner_idxs() const

Returns inner indices of the memory descriptor.

.. note:: 

   This API is only applicable to memory descriptors with format kind :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>`.



.. rubric:: Returns:

A copy of the inner indices vector.

An empty :ref:`dnnl::memory::dims <doxid-group__dnnl__api__primitives__common_1gga94efdd650364f4d9776cfb9b711cbdc1a5927205243f12cdc70612cba6dc874fa>` if the memory descriptor does not have inner indices.

.. index:: pair: function; get_num_handles
.. _doxid-structdnnl_1_1memory_1_1desc_1ad1f0ad6584fa547dba0dd72d54b9162b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int get_num_handles() const

Returns number of handles.



.. rubric:: Returns:

A number of handles.

.. index:: pair: function; get_nnz
.. _doxid-structdnnl_1_1memory_1_1desc_1af15b390ff5f200b75c8bd606c6d10794:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` get_nnz() const

Returns a number of non-zero entries of the memory descriptor.



.. rubric:: Returns:

A number non-zero entries.

.. index:: pair: function; get_sparse_encoding
.. _doxid-structdnnl_1_1memory_1_1desc_1a056c9425560d09e71b154adc611f5e4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::sparse_encoding<doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966>` get_sparse_encoding() const

Returns the sparse encoding of the memory descriptor.



.. rubric:: Returns:

the sparse encoding kind.

.. index:: pair: function; get_data_type
.. _doxid-structdnnl_1_1memory_1_1desc_1aada0dc594d12f25331d4d7cf84c08e75:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` get_data_type(int index = 0) const

Returns the data type of the memory descriptor.



.. rubric:: Returns:

The data type.

.. index:: pair: function; get_format_kind
.. _doxid-structdnnl_1_1memory_1_1desc_1a3e107c30cab4b49c1ea991d09681dfcd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::format_kind<doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105f>` get_format_kind() const

Returns the format kind of the memory descriptor.



.. rubric:: Returns:

the format kind.

.. index:: pair: function; get_dims
.. _doxid-structdnnl_1_1memory_1_1desc_1a525c3c9e3946275b3f386c2f79e8b830:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`memory::dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>` get_dims() const

Returns dimensions of the memory descriptor.

Potentially expensive due to the data copy involved.



.. rubric:: Returns:

A copy of the dimensions vector.

.. index:: pair: function; get_size
.. _doxid-structdnnl_1_1memory_1_1desc_1abfa095ac138d4d2ef8efd3739e343f08:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t get_size(int index = 0) const

Returns size of the memory descriptor in bytes.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- index

		- Data index. Defaults to 0.



.. rubric:: Returns:

The number of bytes required to allocate a memory buffer for data with a particular ``index`` described by this memory descriptor including the padding area.

.. index:: pair: function; is_zero
.. _doxid-structdnnl_1_1memory_1_1desc_1aa162a1ba5621a799c8a909c726f021a2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool is_zero() const

Checks whether the memory descriptor is zero (empty).



.. rubric:: Returns:

``true`` if the memory descriptor describes an empty memory and ``false`` otherwise.

.. index:: pair: function; operator==
.. _doxid-structdnnl_1_1memory_1_1desc_1a9d623dab6f8a8ebc34b0da95814e3728:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool operator == (const desc& other) const

An equality operator.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- other

		- Another memory descriptor.



.. rubric:: Returns:

Whether this and the other memory descriptors have the same format tag, dimensions, strides, blocking, etc.

.. index:: pair: function; operator!=
.. _doxid-structdnnl_1_1memory_1_1desc_1a5a1bc8d3b88a2c304fd9535ea45fbb72:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool operator != (const desc& other) const

An inequality operator.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- other

		- Another memory descriptor.



.. rubric:: Returns:

Whether this and the other memory descriptors describe different memory.

.. index:: pair: function; csr
.. _doxid-structdnnl_1_1memory_1_1desc_1a7fe93a14828506260740fb439eaf6ed4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static desc csr(
		const :ref:`dims<doxid-structdnnl_1_1memory_1afdd20764d58c0b517d5a31276672aeb8>`& adims,
		:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` adata_type,
		:ref:`dim<doxid-structdnnl_1_1memory_1a6ad818e4699872cc913474fa5f122cd5>` nnz,
		:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` index_dt,
		:ref:`data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` pointer_dt,
		bool allow_empty = false
		)

Function for creating a memory descriptor for CSR sparse encoding.

The created memory descriptor will describe a memory object that contains 3 buffers. The buffers have the following meaning and assigned numbers (index):

* 0: values

* 1: indices

* 2: pointers



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- adims

		- Tensor dimensions.

	*
		- adata_type

		- Data precision/type.

	*
		- nnz

		- Number of non-zero entries.

	*
		- index_dt

		- Data type of indices.

	*
		- pointer_dt

		- Data type of pointers.

	*
		- allow_empty

		- A flag signifying whether construction is allowed to fail without throwing an exception. In this case a zero memory descriptor will be constructed. This flag is optional and defaults to false.

