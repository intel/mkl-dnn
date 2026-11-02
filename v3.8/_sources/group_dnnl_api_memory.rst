.. index:: pair: group; Memory
.. _doxid-group__dnnl__api__memory:

Memory
======

.. toctree::
	:hidden:

	enum_dnnl_format_kind_t.rst
	enum_dnnl_format_tag_t.rst
	enum_dnnl_profiling_data_kind_t.rst
	enum_dnnl_sparse_encoding_t.rst
	struct_dnnl_memory.rst
	struct_dnnl_memory_desc.rst
	struct_dnnl_memory-2.rst

Overview
~~~~~~~~

A container that describes and stores data. :ref:`More...<details-group__dnnl__api__memory>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct :ref:`dnnl_memory_desc<doxid-structdnnl__memory__desc>`* :ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`;
	typedef const struct :ref:`dnnl_memory_desc<doxid-structdnnl__memory__desc>`* :ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>`;
	typedef struct :ref:`dnnl_memory<doxid-structdnnl__memory>`* :ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`;
	typedef const struct :ref:`dnnl_memory<doxid-structdnnl__memory>`* :ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>`;

	// enums

	enum :ref:`dnnl_format_kind_t<doxid-group__dnnl__api__memory_1gaa75cad747fa467d9dc527d943ba3367d>`;
	enum :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>`;
	enum :ref:`dnnl_profiling_data_kind_t<doxid-group__dnnl__api__memory_1ga7ac0b200fe8227f70d08832ffc9c51f4>`;
	enum :ref:`dnnl_sparse_encoding_t<doxid-group__dnnl__api__memory_1gad5c084dc8593f175172318438996b552>`;

	// structs

	struct :ref:`dnnl_memory<doxid-structdnnl__memory>`;
	struct :ref:`dnnl_memory_desc<doxid-structdnnl__memory__desc>`;
	struct :ref:`dnnl::memory<doxid-structdnnl_1_1memory>`;

	// global functions

	bool :target:`dnnl::operator ==<doxid-group__dnnl__api__memory_1gaf97bbd7e992c0e211da42bc6eaf12758>` (:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a, :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b);
	bool :target:`dnnl::operator !=<doxid-group__dnnl__api__memory_1ga03fa7afa494ab8dc8484f83a34ce20a6>` (:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a, :ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` b);
	bool :target:`dnnl::operator ==<doxid-group__dnnl__api__memory_1gafa9de7b46bedc943161863d3eaa84100>` (:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a, :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b);
	bool :target:`dnnl::operator !=<doxid-group__dnnl__api__memory_1ga7d9b4a4b2297d66c9495d3a1f2769167>` (:ref:`memory::data_type<doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` a, :ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b);
	bool :target:`dnnl::operator ==<doxid-group__dnnl__api__memory_1ga659960c63f701a0608368e89c5c4ab04>` (:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` a, :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` b);
	bool :target:`dnnl::operator !=<doxid-group__dnnl__api__memory_1gaca9a006590333e3764895a66f0e1a3f2>` (:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` a, :ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` b);
	bool :target:`dnnl::operator ==<doxid-group__dnnl__api__memory_1ga3ae4b6e7ef0bf507b64d875a7c24ae7e>` (:ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` a, :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` b);
	bool :target:`dnnl::operator !=<doxid-group__dnnl__api__memory_1ga6806a5794c45a09b5a9948a5628ffc34>` (:ref:`memory::format_tag<doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3f>` a, :ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` b);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_destroy<doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad>`(:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>` memory_desc);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_clone<doxid-group__dnnl__api__memory_1ga46bc058f1cabc17a49bedfd2633151f7>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` existing_memory_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_get_blob<doxid-group__dnnl__api__memory_1ga467a3501016b5563c8db6f4088201c15>`(
		uint8_t* blob,
		size_t* size,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_blob<doxid-group__dnnl__api__memory_1gaba5f63c523d773634f55393124937a39>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		const uint8_t* blob
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_strides<doxid-group__dnnl__api__memory_1ga97217bb7179b751aa52bc867ac0092fd>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_tag<doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` tag
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_csr_encoding<doxid-group__dnnl__api__memory_1gad072492c74c31bbc576b96bea15cb09c>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` nnz,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` indices_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` pointers_dt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_coo_encoding<doxid-group__dnnl__api__memory_1gae270c1cabaf2529f49ebd3d0b686789e>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` nnz,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` indices_dt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_with_packed_encoding<doxid-group__dnnl__api__memory_1ga63cf90976688f5fc411ad6d7967aa33d>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` nnz
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_create_submemory<doxid-group__dnnl__api__memory_1ga44a99d9ec1dfb2bd80a59e656aebd3b4>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` parent_memory_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` offsets
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_reshape<doxid-group__dnnl__api__memory_1gac6985dc70a545b3aa8415d97b990167b>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* out_memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` in_memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_permute_axes<doxid-group__dnnl__api__memory_1ga2d1ffe0e07d0be1ab066ac912edbbade>`(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* out_memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` in_memory_desc,
		const int* permutation
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_query<doxid-group__dnnl__api__memory_1gacc0b7e295e3e970ba738ad5515d8f837>`(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		void* result
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_desc_query_v2<doxid-group__dnnl__api__memory_1gad083b8bb9d7bbae44e7e33adbd8234d1>`(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index,
		void* result
		);

	int DNNL_API :ref:`dnnl_memory_desc_equal<doxid-group__dnnl__api__memory_1gad722c21c9af227ac7dc25c3ab649aae5>`(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` lhs,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` rhs
		);

	size_t DNNL_API :ref:`dnnl_memory_desc_get_size<doxid-group__dnnl__api__memory_1gae7569a047fdd954866df70f01b63e647>`(:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc);

	size_t DNNL_API :ref:`dnnl_memory_desc_get_size_v2<doxid-group__dnnl__api__memory_1gad8ada49d1107442436109ec1de73f370>`(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		int index
		);

	size_t DNNL_API :ref:`dnnl_data_type_size<doxid-group__dnnl__api__memory_1ga2016d117865455e5d117173dae1b52cb>`(:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_create<doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_create_v2<doxid-group__dnnl__api__memory_1ga90300ec211ec108950e2c4916d56a78a>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		int nhandles,
		void** handles
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_get_memory_desc<doxid-group__dnnl__api__memory_1ga82045853279cc76f52672b8172afdaee>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>`* memory_desc
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_get_engine<doxid-group__dnnl__api__memory_1ga583a4a06428de7d6c4251313e57ad814>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_map_data<doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** mapped_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_map_data_v2<doxid-group__dnnl__api__memory_1ga6946e1edc50752e4ff1ae1b67bace1d9>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** mapped_ptr,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_unmap_data<doxid-group__dnnl__api__memory_1ga46dd4eb02eade91cadd0b9f85b4eccd4>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void* mapped_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_unmap_data_v2<doxid-group__dnnl__api__memory_1gadc4c9d7f47a209373bb5db0186136318>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void* mapped_ptr,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_get_data_handle<doxid-group__dnnl__api__memory_1ga71efa7bd0ac194fdec98fb908b8ba9c5>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_set_data_handle<doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_get_data_handle_v2<doxid-group__dnnl__api__memory_1ga662d48fd7e6f5c4df0e543eea11d94a7>`(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** handle,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_set_data_handle_v2<doxid-group__dnnl__api__memory_1ga5b815baf872121e73204d844b5a0e9fa>`(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* handle,
		int index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_memory_destroy<doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52>`(:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory);

	// macros

	#define :ref:`DNNL_MEMORY_ALLOCATE<doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4>`
	#define :ref:`DNNL_MEMORY_NONE<doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343>`
	#define :ref:`DNNL_RUNTIME_DIM_VAL<doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`
	#define :ref:`DNNL_RUNTIME_F32_VAL<doxid-group__dnnl__api__memory_1gab16365c11b4dc88fbb453edb51f1979f>`
	#define :ref:`DNNL_RUNTIME_S32_VAL<doxid-group__dnnl__api__memory_1ga30139d5110e9e895ccd93fe503ca4c35>`
	#define :ref:`DNNL_RUNTIME_SIZE_VAL<doxid-group__dnnl__api__memory_1ga61466fbd352b6c94b6541977fbe199b8>`

.. _details-group__dnnl__api__memory:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A container that describes and stores data.

Memory objects can contain data of various types and formats. There are two levels of abstraction:

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

Typedefs
--------

.. index:: pair: typedef; dnnl_memory_desc_t
.. _doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_memory_desc<doxid-structdnnl__memory__desc>`* dnnl_memory_desc_t

A memory descriptor handle.

.. index:: pair: typedef; const_dnnl_memory_desc_t
.. _doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_memory_desc<doxid-structdnnl__memory__desc>`* const_dnnl_memory_desc_t

A memory descriptor handle.

.. index:: pair: typedef; dnnl_memory_t
.. _doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_memory<doxid-structdnnl__memory>`* dnnl_memory_t

A memory handle.

.. index:: pair: typedef; const_dnnl_memory_t
.. _doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_memory<doxid-structdnnl__memory>`* const_dnnl_memory_t

A constant memory handle.

Global Functions
----------------

.. index:: pair: function; dnnl_memory_desc_destroy
.. _doxid-group__dnnl__api__memory_1ga836fbf5e9a20cd10b452d2928f82b4ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_destroy(:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>` memory_desc)

Destroys a memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Memory descriptor to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_clone
.. _doxid-group__dnnl__api__memory_1ga46bc058f1cabc17a49bedfd2633151f7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_clone(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` existing_memory_desc
		)

Clones a memory descriptor.

The resulting memory descriptor must be destroyed separately.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Output memory descriptor.

	*
		- existing_memory_desc

		- Memory descriptor to clone.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_get_blob
.. _doxid-group__dnnl__api__memory_1ga467a3501016b5563c8db6f4088201c15:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_get_blob(
		uint8_t* blob,
		size_t* size,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc
		)

Retrieves a binary blob associated with the given memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- blob

		- Output pointer to binary blob. If not nullptr, size bytes of the memory descriptor blob are written.

	*
		- size

		- Output pointer to the size of the binary blob in bytes. Size is written if blob is nullptr.

	*
		- memory_desc

		- input memory descriptor to serialize



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_create_with_blob
.. _doxid-group__dnnl__api__memory_1gaba5f63c523d773634f55393124937a39:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_create_with_blob(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		const uint8_t* blob
		)

Creates a memory descriptor from a memory descriptor binary blob.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Output pointer to a newly allocated memory descriptor.

	*
		- blob

		- Pointer to a memory descriptor binary blob.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_create_with_strides
.. _doxid-group__dnnl__api__memory_1ga97217bb7179b751aa52bc867ac0092fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_create_with_strides(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides
		)

Creates a memory descriptor using dimensions and strides.

.. note:: 

   As always, the logical order of dimensions corresponds to the ``abc...`` format tag, and the physical meaning of the dimensions depends on both the primitive that consumes the memory and the context of that consumption.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Output memory descriptor.

	*
		- ndims

		- Number of dimensions

	*
		- dims

		- Array of dimensions.

	*
		- data_type

		- Elements data type.

	*
		- strides

		- Strides in each dimension.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_create_with_tag
.. _doxid-group__dnnl__api__memory_1gaa326fcf2176d2f9e28f513259f4f8326:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_create_with_tag(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_format_tag_t<doxid-group__dnnl__api__memory_1ga395e42b594683adb25ed2d842bb3091d>` tag
		)

Creates a memory descriptor using dimensions and memory format tag.

.. note:: 

   As always, the logical order of dimensions corresponds to the ``abc...`` format tag, and the physical meaning of the dimensions depends on both the primitive that consumes the memory and the context of that consumption.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Output memory descriptor.

	*
		- ndims

		- Number of dimensions

	*
		- dims

		- Array of dimensions.

	*
		- data_type

		- Elements data type.

	*
		- tag

		- Memory format tag. Can be :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` which would allow a primitive to chose the final memory format. In this case the format_kind field of the memory descriptor would be set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_create_with_csr_encoding
.. _doxid-group__dnnl__api__memory_1gad072492c74c31bbc576b96bea15cb09c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_create_with_csr_encoding(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` nnz,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` indices_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` pointers_dt
		)

Creates a memory descriptor for CSR encoding.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Output memory descriptor.

	*
		- ndims

		- Number of dimensions

	*
		- dims

		- Array of dimensions.

	*
		- data_type

		- Elements data type.

	*
		- nnz

		- Number of non-zero entries.

	*
		- indices_dt

		- Data type of indices.

	*
		- pointers_dt

		- Data type of pointers.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_create_with_coo_encoding
.. _doxid-group__dnnl__api__memory_1gae270c1cabaf2529f49ebd3d0b686789e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_create_with_coo_encoding(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` nnz,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` indices_dt
		)

Creates a memory descriptor for COO encoding.

The created memory descriptor will describe a memory object that contains n+1 buffers for an n-dimensional tensor. The buffers have the following meaning and assigned numbers (index):

* 0: values

* 1: indices for dimension 0

* 2: indices for dimension 1 ...

* n: indices for dimension n-1



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Output memory descriptor.

	*
		- ndims

		- Number of dimensions.

	*
		- dims

		- Array of dimensions.

	*
		- data_type

		- Elements data type.

	*
		- nnz

		- Number of non-zero entries.

	*
		- indices_dt

		- Data type of indices.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_create_with_packed_encoding
.. _doxid-group__dnnl__api__memory_1ga63cf90976688f5fc411ad6d7967aa33d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_create_with_packed_encoding(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` nnz
		)

Creates a memory descriptor for packed sparse encoding.

The created memory descriptor cannot be used to create a memory object. It can only be used to create a primitive descriptor to query the actual memory descriptor (similar to the format tag ``any``).

.. warning:: 

   The meaning and content of the handles of the memory object that is created using the queried memory descriptor are unspecified therefore using the content is an undefined behavior.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Output memory descriptor.

	*
		- ndims

		- Number of dimensions

	*
		- dims

		- Array of dimensions.

	*
		- data_type

		- Elements data type.

	*
		- nnz

		- Number of non-zero entries.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_create_submemory
.. _doxid-group__dnnl__api__memory_1ga44a99d9ec1dfb2bd80a59e656aebd3b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_create_submemory(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` parent_memory_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` offsets
		)

Creates a memory descriptor for a region inside an area described by an existing memory descriptor.

.. warning:: 

   Some combinations of physical memory layout and/or offsets or dims may result in a failure to create a submemory.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Output memory descriptor.

	*
		- parent_memory_desc

		- An existing memory descriptor.

	*
		- dims

		- Sizes of the region.

	*
		- offsets

		- Offsets to the region from the encompassing memory object in each dimension



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_reshape
.. _doxid-group__dnnl__api__memory_1gac6985dc70a545b3aa8415d97b990167b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_reshape(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* out_memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` in_memory_desc,
		int ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims
		)

Creates a memory descriptor by reshaping an existing one.

The new memory descriptor inherits the data type. This operation is valid only for memory descriptors that have format_kind :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>` or :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.

The resulting memory descriptor must be destroyed separately.

The operation ensures the transformation of the physical memory format corresponds to the transformation of the logical dimensions. If such transformation is impossible, the function returns :ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>`.

The reshape operation can be described as a combination of the following basic operations:

#. Add a dimension of size ``1``. This is always possible.

#. Remove a dimension of size ``1``. This is possible only if the dimension has no padding (i.e. ``padded_dims[dim] == dims[dim] && dims[dim] == 1``).

#. Split a dimension into multiple ones. This is possible only if the size of the dimension is exactly equal to the product of the split ones and the dimension does not have padding (i.e. ``padded_dims[dim] = dims[dim]``).

#. Joining multiple consecutive dimensions into a single one. As in the cases above, this requires that the dimensions do not have padding and that the memory format is such that in physical memory these dimensions are dense and have the same order as their logical counterparts. This also assumes that these dimensions are not blocked.
   
   * Here, dense means: ``stride for dim[i] == (stride for dim[i + 1]) * dim[i + 1]``;
   
   * And same order means: ``i < j`` if and only if ``stride for dim[j] <= stride for dim[i]``.

.. warning:: 

   Some combinations of physical memory layout and/or offsets or dimensions may result in a failure to make a reshape.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- out_memory_desc

		- Output memory descriptor.

	*
		- in_memory_desc

		- An existing memory descriptor. Must have format_kind set to :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>` or :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.

	*
		- ndims

		- Number of dimensions for the output memory descriptor.

	*
		- dims

		- Dimensions for the output memory descriptor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_permute_axes
.. _doxid-group__dnnl__api__memory_1ga2d1ffe0e07d0be1ab066ac912edbbade:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_permute_axes(
		:ref:`dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1gad281fd59c474d46a60f9b3a165e9374f>`* out_memory_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` in_memory_desc,
		const int* permutation
		)

Creates a memory descriptor by permuting axes in an existing one.

The physical memory layout representation is adjusted accordingly to maintain the consistency between the logical and physical parts of the memory descriptor.

The resulting memory descriptor must be destroyed separately.

The new memory descriptor inherits the data type. This operation is valid only for memory descriptors that have format_kind set to :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>` or :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.

The logical axes will be permuted in the following manner:

.. ref-code-block:: cpp

	for (i: 0 .. in_memory_desc->ndims)
	    out_memory_desc->dims[permutation[i]] = in_memory_desc->dims[i];

Example:

.. ref-code-block:: cpp

	dnnl_memory_desc_t in_md, out_md, expect_out_md;
	
	const int permutation[] = {1, 0}; // swap the first and the second axes
	
	dnnl_dims_t in_dims = {2, 3}, out_dims = {3, 2};
	dnnl_format_tag_t in_tag = dnnl_ab, out_tag = dnnl_ba;
	
	dnnl_memory_desc_create_with_tag(
	        &in_md, 2, in_dims, data_type, in_tag);
	dnnl_memory_desc_create_with_tag(
	        &expect_out_md, 2, out_dims, data_type, out_tag);
	
	dnnl_memory_desc_permute_axes(&out_md, in_md, permutation);
	assert(dnnl_memory_desc_equal(out_md, expect_out_md));
	
	dnnl_memory_desc_destroy(in_md);
	dnnl_memory_desc_destroy(out_md);
	dnnl_memory_desc_destroy(expect_out_md);



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- out_memory_desc

		- Output memory descriptor.

	*
		- in_memory_desc

		- An existing memory descriptor. Must have format_kind set to :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>` or :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.

	*
		- permutation

		- Axes permutation (of size ``in_memory_desc->ndims``).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_query
.. _doxid-group__dnnl__api__memory_1gacc0b7e295e3e970ba738ad5515d8f837:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_query(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		void* result
		)

Queries a memory descriptor for various pieces of information.

The following information can be queried:

* Number of dimensions (:ref:`dnnl_query_ndims_s32 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059afe40d0bef09ca1d2567c46eb413e8580>`)

* Dimensions (:ref:`dnnl_query_dims <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059abe3af06a74e32063626361f1902aaa87>`) in the following order:
  
  * CNN data tensors: mini-batch, channel, spatial (``{N, C, [[D,] H,] W}``)
  
  * CNN weight tensors: group (optional), output channel, input channel, spatial (``{[G,] O, I, [[D,] H,] W}``)
  
  * RNN data tensors: time, mini-batch, channels (``{T, N, C}``) or layers, directions, states, mini-batch, channels (``{L, D, S, N, C}``)
  
  * RNN weight tensor: layers, directions, input channel, gates, output channels (``{L, D, I, G, O}``)

* Data type of the tensor elements (:ref:`dnnl_query_data_type <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aab9ebb3344a6e3b283801c8266b56530>`)

* Padded dimensions (:ref:`dnnl_query_padded_dims <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2bc0848a5ee584227253aa71773db112>`) - size of the data including padding in each dimension

* Padded offsets (:ref:`dnnl_query_padded_offsets <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8f91293e9b3007cc89ce919852139a36>`) - per-dimension offset from the padding to actual data, the top-level tensor with offsets applied must lie within the padding area.

* Submemory offset (:ref:`dnnl_query_submemory_offset_s64 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a58f5f05e331cf0974fbccad0e2429e67>`) - offset from memory origin to the current block, non-zero only in a description of a memory sub-block.

* Format kind (:ref:`dnnl_query_format_kind <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ad534a84e6f4709a8f597bf8558730c3e>`) - memory format kind

.. note:: 

   The order of dimensions does not depend on the memory format, so whether the data is laid out in :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` or :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` the dims for 4D CN data tensor would be ``{N, C, H, W}``.
   
   
The following queries are applicable only to format kind :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>`.

* Strides (:ref:`dnnl_query_strides <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ab5f542868da5bc8c3b9d3a80b6e46d25>`) between the outermost blocks or in case of plain (non-blocked) formats the strides between dimensions

* Number of innermost blocks (:ref:`dnnl_query_inner_nblks_s32 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a942da7995fe07b02ba1d48be13c6d951>`), e.g. ``{4, 16, 4}`` in case of ``OIhw_4i16o4i``

* Size of the innermost blocks (:ref:`dnnl_query_inner_blks <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a6c18535baa6bdb2a264c4e62e5f66b73>`), e.g. 3 in case of ``OIhw_4i16o4i_``

* Logical indices of the blocks (:ref:`dnnl_query_inner_idxs <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ae65233dcfb5128c05ed7c97319c00a35>`), e.g. ``{1, 0, 1}`` in case of ``4i16o4i``, because ``i`` is the 1st dim and ``o`` is the 0st dim



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Memory descriptor.

	*
		- what

		- Parameter to query.

	*
		- result

		- Output result. The type depends on the query. For example, it must be a ``dnnl_dims_t**`` if querying for a strides.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_query_v2
.. _doxid-group__dnnl__api__memory_1gad083b8bb9d7bbae44e7e33adbd8234d1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_desc_query_v2(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_query_t<doxid-group__dnnl__api__primitives__common_1ga9e5235563cf7cfc10fa89f415de98059>` what,
		int index,
		void* result
		)

Queries a memory descriptor for various pieces of information.

This version support additional queries :ref:`dnnl_query_sparse_encoding <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a21815bb69d71340b0556f123ba6fdd69>`, :ref:`dnnl_query_nnz_s64 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a5ca45f20f5864e069149106f21f5ff92>` :ref:`dnnl_query_num_handles_s32 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a7d92c3824fd1811f6bc641e2fdfbc2bb>` and :ref:`dnnl_query_data_type <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aab9ebb3344a6e3b283801c8266b56530>` for a particular buffer.

The following information can be queried:

* Number of dimensions (:ref:`dnnl_query_ndims_s32 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059afe40d0bef09ca1d2567c46eb413e8580>`)

* Dimensions (:ref:`dnnl_query_dims <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059abe3af06a74e32063626361f1902aaa87>`) in the following order:
  
  * CNN data tensors: mini-batch, channel, spatial (``{N, C, [[D,] H,] W}``)
  
  * CNN weight tensors: group (optional), output channel, input channel, spatial (``{[G,] O, I, [[D,] H,] W}``)
  
  * RNN data tensors: time, mini-batch, channels (``{T, N, C}``) or layers, directions, states, mini-batch, channels (``{L, D, S, N, C}``)
  
  * RNN weight tensor: layers, directions, input channel, gates, output channels (``{L, D, I, G, O}``)

* Data type of the tensor elements (:ref:`dnnl_query_data_type <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aab9ebb3344a6e3b283801c8266b56530>`)

* Padded dimensions (:ref:`dnnl_query_padded_dims <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a2bc0848a5ee584227253aa71773db112>`) - size of the data including padding in each dimension

* Padded offsets (:ref:`dnnl_query_padded_offsets <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a8f91293e9b3007cc89ce919852139a36>`) - per-dimension offset from the padding to actual data, the top-level tensor with offsets applied must lie within the padding area.

* Submemory offset (:ref:`dnnl_query_submemory_offset_s64 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a58f5f05e331cf0974fbccad0e2429e67>`) - offset from memory origin to the current block, non-zero only in a description of a memory sub-block.

* Format kind (:ref:`dnnl_query_format_kind <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ad534a84e6f4709a8f597bf8558730c3e>`) - memory format kind

.. note:: 

   The order of dimensions does not depend on the memory format, so whether the data is laid out in :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` or :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` the dims for 4D CN data tensor would be ``{N, C, H, W}``.
   
   
The following queries are applicable only to format kind :ref:`dnnl_blocked <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>`.

* Strides (:ref:`dnnl_query_strides <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ab5f542868da5bc8c3b9d3a80b6e46d25>`) between the outermost blocks or in case of plain (non-blocked) formats the strides between dimensions

* Number of innermost blocks (:ref:`dnnl_query_inner_nblks_s32 <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a942da7995fe07b02ba1d48be13c6d951>`), e.g. ``{4, 16, 4}`` in case of ``OIhw_4i16o4i``

* Size of the innermost blocks (:ref:`dnnl_query_inner_blks <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059a6c18535baa6bdb2a264c4e62e5f66b73>`), e.g. 3 in case of ``OIhw_4i16o4i_``

* Logical indices of the blocks (:ref:`dnnl_query_inner_idxs <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059ae65233dcfb5128c05ed7c97319c00a35>`), e.g. ``{1, 0, 1}`` in case of ``4i16o4i``, because ``i`` is the 1st dim and ``o`` is the 0st dim



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Memory descriptor.

	*
		- what

		- Parameter to query.

	*
		- index

		- Index of the parameter to query for. It is mostly used with :ref:`dnnl_query_data_type <doxid-group__dnnl__api__primitives__common_1gga9e5235563cf7cfc10fa89f415de98059aab9ebb3344a6e3b283801c8266b56530>` to specify which data type is being queried. The main data type (data type of values) has always index 0. For other indices please refer to the API for creating a memory descriptor for sparse encoding.

	*
		- result

		- Output result. The type depends on the query. For example, it must be a ``dnnl_dims_t**`` if querying for a strides.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_desc_equal
.. _doxid-group__dnnl__api__memory_1gad722c21c9af227ac7dc25c3ab649aae5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int DNNL_API dnnl_memory_desc_equal(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` lhs,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` rhs
		)

Compares two memory descriptors.

Use this function to identify whether a reorder is required between the two memories



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- lhs

		- Left-hand side of the comparison.

	*
		- rhs

		- Right-hand side of the comparison.



.. rubric:: Returns:

1 if the descriptors are the same.

0 if the descriptors are different.

.. index:: pair: function; dnnl_memory_desc_get_size
.. _doxid-group__dnnl__api__memory_1gae7569a047fdd954866df70f01b63e647:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t DNNL_API dnnl_memory_desc_get_size(:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc)

Returns the size of a memory descriptor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Memory descriptor.



.. rubric:: Returns:

The number of bytes required for memory described by a memory descriptor.

.. index:: pair: function; dnnl_memory_desc_get_size_v2
.. _doxid-group__dnnl__api__memory_1gad8ada49d1107442436109ec1de73f370:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t DNNL_API dnnl_memory_desc_get_size_v2(
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		int index
		)

Returns the size of the data that corresponds to the given index.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory_desc

		- Memory descriptor.

	*
		- index

		- Index of the buffer.



.. rubric:: Returns:

The number of bytes required for the requested data.

.. index:: pair: function; dnnl_data_type_size
.. _doxid-group__dnnl__api__memory_1ga2016d117865455e5d117173dae1b52cb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t DNNL_API dnnl_data_type_size(:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type)

Returns the size of data type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- data_type

		- Data type.



.. rubric:: Returns:

The number of bytes occupied by data type.

.. index:: pair: function; dnnl_memory_create
.. _doxid-group__dnnl__api__memory_1ga24c17a1c03c05be8907114f9b46f0761:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_create(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* handle
		)

Creates a memory object.

Unless ``handle`` is equal to DNNL_MEMORY_NONE, the constructed memory object will have the underlying buffer set. In this case, the buffer will be initialized as if :ref:`dnnl_memory_set_data_handle() <doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>` had been called.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Output memory object.

	*
		- memory_desc

		- Memory descriptor.

	*
		- engine

		- Engine to use.

	*
		- handle

		- 
		  Handle of the memory buffer to use as an underlying storage.
		  
		  * A pointer to the user-allocated buffer. In this case the library doesn't own the buffer.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
		  
		  * DNNL_MEMORY_NONE to create :ref:`dnnl_memory <doxid-structdnnl__memory>` without an underlying buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`dnnl_memory_set_data_handle() <doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf>`

.. index:: pair: function; dnnl_memory_create_v2
.. _doxid-group__dnnl__api__memory_1ga90300ec211ec108950e2c4916d56a78a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_create_v2(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>`* memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` memory_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		int nhandles,
		void** handles
		)

Creates a memory object with multiple handles.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Output memory object.

	*
		- memory_desc

		- Memory descriptor.

	*
		- engine

		- Engine to use.

	*
		- nhandles

		- Number of handles.

	*
		- handles

		- 
		  Handles of the memory buffers to use as underlying storages. For each element of the ``handles`` array the following applies:
		  
		  * A pointer to the user-allocated buffer. In this case the library doesn't own the buffer.
		  
		  * The DNNL_MEMORY_ALLOCATE special value. Instructs the library to allocate the buffer for the memory object. In this case the library owns the buffer.
		  
		  * DNNL_MEMORY_NONE Instructs the library to skip allocation of the memory buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_get_memory_desc
.. _doxid-group__dnnl__api__memory_1ga82045853279cc76f52672b8172afdaee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_get_memory_desc(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>`* memory_desc
		)

Returns the memory descriptor for a memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- memory_desc

		- Output memory descriptor (a copy).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_get_engine
.. _doxid-group__dnnl__api__memory_1ga583a4a06428de7d6c4251313e57ad814:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_get_engine(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		)

Returns the engine of a memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- engine

		- Output engine on which the memory is located.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_map_data
.. _doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_map_data(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** mapped_ptr
		)

Maps a memory object and returns a host-side pointer to a memory buffer with a copy of its contents.

Mapping enables explicit direct access to memory contents for the engines that do not support it implicitly.

Mapping is an exclusive operation - a memory object cannot be used in other operations until this memory object is unmapped.

.. note:: 

   Any primitives working with ``memory`` should be completed before the memory is mapped. Use dnnl_stream_wait to synchronize the corresponding execution stream.
   
   

.. note:: 

   The :ref:`dnnl_memory_map_data() <doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>` and :ref:`dnnl_memory_unmap_data() <doxid-group__dnnl__api__memory_1ga46dd4eb02eade91cadd0b9f85b4eccd4>` functions are mainly provided for debug and testing purposes, and their performance may be suboptimal.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- mapped_ptr

		- Output pointer to the mapped buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_map_data_v2
.. _doxid-group__dnnl__api__memory_1ga6946e1edc50752e4ff1ae1b67bace1d9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_map_data_v2(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** mapped_ptr,
		int index
		)

Maps a memory object and returns a host-side pointer to a memory buffer with a copy of its contents.

The memory buffer corresponds to the given index.

Mapping enables explicit direct access to memory contents for the engines that do not support it implicitly.

Mapping is an exclusive operation - a memory object cannot be used in other operations until this memory object is unmapped.

.. note:: 

   Any primitives working with ``memory`` should be completed before the memory is mapped. Use dnnl_stream_wait to synchronize the corresponding execution stream.
   
   

.. note:: 

   The :ref:`dnnl_memory_map_data() <doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>` and :ref:`dnnl_memory_unmap_data() <doxid-group__dnnl__api__memory_1ga46dd4eb02eade91cadd0b9f85b4eccd4>` functions are mainly provided for debug and testing purposes, and their performance may be suboptimal.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- mapped_ptr

		- Output pointer to the mapped buffer.

	*
		- index

		- Index of the buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_unmap_data
.. _doxid-group__dnnl__api__memory_1ga46dd4eb02eade91cadd0b9f85b4eccd4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_unmap_data(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void* mapped_ptr
		)

Unmaps a memory object and writes back any changes made to the previously mapped memory buffer.

The pointer to the mapped buffer must be obtained via the :ref:`dnnl_memory_map_data() <doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>` call.

.. note:: 

   The :ref:`dnnl_memory_map_data() <doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>` and :ref:`dnnl_memory_unmap_data() <doxid-group__dnnl__api__memory_1ga46dd4eb02eade91cadd0b9f85b4eccd4>` functions are mainly provided for debug and testing purposes, and their performance may be suboptimal.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- mapped_ptr

		- Pointer to the mapped buffer that must have been obtained using the :ref:`dnnl_memory_map_data() <doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>` function.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_unmap_data_v2
.. _doxid-group__dnnl__api__memory_1gadc4c9d7f47a209373bb5db0186136318:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_unmap_data_v2(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void* mapped_ptr,
		int index
		)

Unmaps a memory object and writes back any changes made to the previously mapped memory buffer.

The pointer to the mapped buffer must be obtained via the :ref:`dnnl_memory_map_data() <doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>` call. The buffer corresponds to the given index.

.. note:: 

   The :ref:`dnnl_memory_map_data() <doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>` and :ref:`dnnl_memory_unmap_data() <doxid-group__dnnl__api__memory_1ga46dd4eb02eade91cadd0b9f85b4eccd4>` functions are mainly provided for debug and testing purposes, and their performance may be suboptimal.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- mapped_ptr

		- Pointer to the mapped buffer that must have been obtained using the :ref:`dnnl_memory_map_data() <doxid-group__dnnl__api__memory_1gac9006cdf6816b8bef7be3455ab0ceb49>` function.

	*
		- index

		- Index of the buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_get_data_handle
.. _doxid-group__dnnl__api__memory_1ga71efa7bd0ac194fdec98fb908b8ba9c5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_get_data_handle(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** handle
		)

Returns memory object's data handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- handle

		- Output data handle. For the CPU engine, the data handle is a pointer to the actual data. For OpenCL it is a cl_mem.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_set_data_handle
.. _doxid-group__dnnl__api__memory_1ga6888f8c17f272d6729c9bc258ed41fcf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_set_data_handle(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* handle
		)

Sets the underlying memory buffer.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- handle

		- Data handle. For the CPU engine or when USM is used, the memory buffer is a pointer to the actual data. For OpenCL it is a ``cl_mem``.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_get_data_handle_v2
.. _doxid-group__dnnl__api__memory_1ga662d48fd7e6f5c4df0e543eea11d94a7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_get_data_handle_v2(
		:ref:`const_dnnl_memory_t<doxid-group__dnnl__api__memory_1ga0f89ee8e9b55b302b3f5277d11302f7e>` memory,
		void** handle,
		int index
		)

Returns an underlying memory buffer that corresponds to the given index.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- handle

		- Data handle. For the CPU engine or when USM is used, the memory buffer is a pointer to the actual data. For OpenCL it is a ``cl_mem``.

	*
		- index

		- Index of the buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_set_data_handle_v2
.. _doxid-group__dnnl__api__memory_1ga5b815baf872121e73204d844b5a0e9fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_set_data_handle_v2(
		:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory,
		void* handle,
		int index
		)

Sets an underlying memory buffer that corresponds to the given index.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object.

	*
		- handle

		- Data handle. For the CPU engine or when USM is used, the memory buffer is a pointer to the actual data. For OpenCL it is a ``cl_mem``.

	*
		- index

		- Index of the buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_memory_destroy
.. _doxid-group__dnnl__api__memory_1gaa219225aae8e53489caab3fe1bc80a52:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_memory_destroy(:ref:`dnnl_memory_t<doxid-group__dnnl__api__memory_1ga2b79954bd7bb293e766a89189e8440fd>` memory)

Destroys a memory object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- memory

		- Memory object to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

Macros
------

.. index:: pair: define; DNNL_MEMORY_ALLOCATE
.. _doxid-group__dnnl__api__memory_1gaf19cbfbf1f0a9508283f2a25561ae0e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_MEMORY_ALLOCATE

Special pointer value that indicates that the library needs to allocate an underlying buffer for a memory object.

.. index:: pair: define; DNNL_MEMORY_NONE
.. _doxid-group__dnnl__api__memory_1ga96c8752fb3cb4f01cf64bf56190b1343:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_MEMORY_NONE

Special pointer value that indicates that a memory object should not have an underlying buffer.

.. index:: pair: define; DNNL_RUNTIME_DIM_VAL
.. _doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_RUNTIME_DIM_VAL

A wildcard value for dimensions that are unknown at a primitive creation time.

.. index:: pair: define; DNNL_RUNTIME_F32_VAL
.. _doxid-group__dnnl__api__memory_1gab16365c11b4dc88fbb453edb51f1979f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_RUNTIME_F32_VAL

A wildcard value for floating point values that are unknown at a primitive creation time.

.. index:: pair: define; DNNL_RUNTIME_S32_VAL
.. _doxid-group__dnnl__api__memory_1ga30139d5110e9e895ccd93fe503ca4c35:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_RUNTIME_S32_VAL

A wildcard value for int32_t values that are unknown at a primitive creation time.

.. index:: pair: define; DNNL_RUNTIME_SIZE_VAL
.. _doxid-group__dnnl__api__memory_1ga61466fbd352b6c94b6541977fbe199b8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_RUNTIME_SIZE_VAL

A ``size_t`` counterpart of the DNNL_RUNTIME_DIM_VAL.

For instance, this value is returned by :ref:`dnnl_memory_desc_get_size() <doxid-group__dnnl__api__memory_1gae7569a047fdd954866df70f01b63e647>` if either of the dimensions or strides equal to :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`.

