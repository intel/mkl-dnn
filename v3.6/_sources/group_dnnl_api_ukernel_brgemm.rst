.. index:: pair: group; BRGeMM ukernel
.. _doxid-group__dnnl__api__ukernel__brgemm:

BRGeMM ukernel
==============

.. toctree::
	:hidden:

	struct_dnnl_ukernel_brgemm.rst
	struct_dnnl_brgemm.rst
	struct_dnnl_transform.rst
	struct_dnnl_ukernel_transform.rst

Overview
~~~~~~~~

BRGeMM ukernel routines. :ref:`More...<details-group__dnnl__api__ukernel__brgemm>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct :ref:`dnnl_brgemm<doxid-structdnnl__brgemm>`* :ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>`;
	typedef const struct :ref:`dnnl_brgemm<doxid-structdnnl__brgemm>`* :ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>`;
	typedef struct :ref:`dnnl_transform<doxid-structdnnl__transform>`* :ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga0a05ad64b8e8617112a045d12876b6e1>`;
	typedef const struct :ref:`dnnl_transform<doxid-structdnnl__transform>`* :ref:`const_dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga1ef3c8a87d676f5b644c2677062ba485>`;

	// structs

	struct :ref:`dnnl::ukernel::brgemm<doxid-structdnnl_1_1ukernel_1_1brgemm>`;
	struct :ref:`dnnl_brgemm<doxid-structdnnl__brgemm>`;
	struct :ref:`dnnl_transform<doxid-structdnnl__transform>`;
	struct :ref:`dnnl::ukernel::transform<doxid-structdnnl_1_1ukernel_1_1transform>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_create<doxid-group__dnnl__api__ukernel__brgemm_1ga346f0fae2ec1d20e2919f8728e6f891b>`(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>`* brgemm,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` batch_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` c_dt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_set_add_C<doxid-group__dnnl__api__ukernel__brgemm_1ga7ba78ec52cd53efb6fb169bd807d35d4>`(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm, int add_C);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_set_post_ops<doxid-group__dnnl__api__ukernel__brgemm_1ga8ea564233baca3938615799e238f1e0a>`(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldd,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` d_dt,
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_set_A_scales<doxid-group__dnnl__api__ukernel__brgemm_1gab65ecdb9b86fbddd96f4b031d2e7c720>`(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm,
		int a_scale_mask
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_set_B_scales<doxid-group__dnnl__api__ukernel__brgemm_1gac649e96fa065f13879e3299a9c0d8782>`(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm,
		int b_scale_mask
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_set_D_scales<doxid-group__dnnl__api__ukernel__brgemm_1gafda5c7a7124c9235cc930e3893e27c6b>`(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm,
		int d_scale_mask
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_finalize<doxid-group__dnnl__api__ukernel__brgemm_1gaa25413b0931da573eda22fb2004d5eeb>`(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_get_B_pack_type<doxid-group__dnnl__api__ukernel__brgemm_1gae918ea867b1a33522329695ad467f2ba>`(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		:ref:`dnnl_pack_type_t<doxid-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97>`* pack_type
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_get_scratchpad_size<doxid-group__dnnl__api__ukernel__brgemm_1ga7e9145a254a5807361252222f46c7236>`(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		size_t* size
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_set_hw_context<doxid-group__dnnl__api__ukernel__brgemm_1gadce89da8fca565cbc41d0ed7bcbc2a62>`(:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_release_hw_context<doxid-group__dnnl__api__ukernel__brgemm_1gaa804b65b7182dbc365ca465eb0651859>`();
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_generate<doxid-group__dnnl__api__ukernel__brgemm_1gaae5ee87a3fb22fce849a5c0de71d4cb6>`(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_execute<doxid-group__dnnl__api__ukernel__brgemm_1ga73ad5c1d29039310540bfb243b4ce17d>`(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		const void* A_ptr,
		const void* B_ptr,
		const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* A_B_offsets,
		void* C_ptr,
		void* scratchpad_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_execute_postops<doxid-group__dnnl__api__ukernel__brgemm_1ga3e92e657b97d1d5545a32ee9d9f57872>`(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		const void* A,
		const void* B,
		const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* A_B_offsets,
		const void* C_ptr,
		void* D_ptr,
		void* scratchpad_ptr,
		:ref:`const_dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1ga19b82b33015ab0abfe2630236d3da7fc>` attr_params
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_destroy<doxid-group__dnnl__api__ukernel__brgemm_1gaa16a3c2c3657d5fe968fc58d651041fa>`(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_transform_create<doxid-group__dnnl__api__ukernel__brgemm_1ga8b8ace47537f66365a9794c9f589d89d>`(
		:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga0a05ad64b8e8617112a045d12876b6e1>`* transform,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_pack_type_t<doxid-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97>` in_pack_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` in_ld,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` out_ld,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` in_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` out_dt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_transform_generate<doxid-group__dnnl__api__ukernel__brgemm_1ga75b9793b4f57eee2f4858c373e4cc49a>`(:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga0a05ad64b8e8617112a045d12876b6e1>` transform);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_transform_execute<doxid-group__dnnl__api__ukernel__brgemm_1ga3467d4f77ce81fb64065ca0fecb19226>`(
		:ref:`const_dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga1ef3c8a87d676f5b644c2677062ba485>` transform,
		const void* in_ptr,
		void* out_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_transform_destroy<doxid-group__dnnl__api__ukernel__brgemm_1gacea3f51d81b00d087fbd82259acaee4b>`(:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga0a05ad64b8e8617112a045d12876b6e1>` transform);

.. _details-group__dnnl__api__ukernel__brgemm:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

BRGeMM ukernel routines.

Typedefs
--------

.. index:: pair: typedef; dnnl_brgemm_t
.. _doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_brgemm<doxid-structdnnl__brgemm>`* dnnl_brgemm_t

A brgemm ukernel handle.

.. index:: pair: typedef; const_dnnl_brgemm_t
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_brgemm<doxid-structdnnl__brgemm>`* const_dnnl_brgemm_t

A constant brgemm ukernel handle.

.. index:: pair: typedef; dnnl_transform_t
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga0a05ad64b8e8617112a045d12876b6e1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_transform<doxid-structdnnl__transform>`* dnnl_transform_t

A transform routine handle.

.. index:: pair: typedef; const_dnnl_transform_t
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga1ef3c8a87d676f5b644c2677062ba485:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_transform<doxid-structdnnl__transform>`* const_dnnl_transform_t

A constant transform routine handle.

Global Functions
----------------

.. index:: pair: function; dnnl_brgemm_create
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga346f0fae2ec1d20e2919f8728e6f891b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_create(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>`* brgemm,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` batch_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` c_dt
		)

Creates a BRGeMM ukernel object.

Operates by the following formula: ``C = [A x B]``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- Output BRGeMM ukernel object.

	*
		- M

		- Dimension M of tensor A.

	*
		- N

		- Dimension N of tensor B.

	*
		- K

		- Dimension K of tensors A and B.

	*
		- batch_size

		- Number of batches to process.

	*
		- lda

		- Leading dimension of tensor A.

	*
		- ldb

		- Leading dimension of tensor B.

	*
		- ldc

		- Leading dimension of tensor C.

	*
		- a_dt

		- Data type of tensor A.

	*
		- b_dt

		- Data type of tensor B.

	*
		- c_dt

		- Data type of tensor C. Must be dnnl_f32.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_set_add_C
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga7ba78ec52cd53efb6fb169bd807d35d4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_set_add_C(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm, int add_C)

Sets adding an intermediate result to the output tensor C instead of writing: ``C += [A x B]``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- add_C

		- Value to indicate addition. Can be ``0`` to skip addition, and ``1`` to apply addition.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_set_post_ops
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga8ea564233baca3938615799e238f1e0a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_set_post_ops(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldd,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` d_dt,
		:ref:`const_dnnl_post_ops_t<doxid-group__dnnl__api__attributes_1ga997bc4a3d9d2ce50238b1c035963fc39>` post_ops
		)

Sets post-operations to a BRGeMM ukernel object: ``D = post-operations(C)``.

Post-operations applies if one of the following holds:

* Non-empty attributes are specified.

* Output data type ``d_dt`` is different from accumulation data type ``c_dt``.

If any of conditions happens, the final call of the accumulation chain must be ``dnnl_brgemm_execute_postops``, and ``dnnl_brgemm_execute``, otherwise.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- ldd

		- Leading dimension of tensor D.

	*
		- d_dt

		- Data type of tensor D.

	*
		- post_ops

		- Primitive post operations attribute to extend the kernel operations.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_set_A_scales
.. _doxid-group__dnnl__api__ukernel__brgemm_1gab65ecdb9b86fbddd96f4b031d2e7c720:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_set_A_scales(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm,
		int a_scale_mask
		)

Sets tensor A scales mask to a BRGeMM ukernel object.

For quantization flavor tensor A scales apply to accumulation buffer once C is ready.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- a_scale_mask

		- Tensor A scale mask. Can be ``0`` only.

.. index:: pair: function; dnnl_brgemm_set_B_scales
.. _doxid-group__dnnl__api__ukernel__brgemm_1gac649e96fa065f13879e3299a9c0d8782:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_set_B_scales(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm,
		int b_scale_mask
		)

Sets tensor B scales mask to a BRGeMM ukernel object.

For quantization flavor tensor B scales apply to accumulation buffer once C is ready.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- b_scale_mask

		- Tensor B scale mask. Can be ``0`` and ``2`` only.

.. index:: pair: function; dnnl_brgemm_set_D_scales
.. _doxid-group__dnnl__api__ukernel__brgemm_1gafda5c7a7124c9235cc930e3893e27c6b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_set_D_scales(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm,
		int d_scale_mask
		)

Sets tensor D scales mask to a BRGeMM ukernel object.

For quantization flavor tensor D scales apply after all post-ops are applied.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- d_scale_mask

		- Tensor D scale mask. Can be ``0`` only.

.. index:: pair: function; dnnl_brgemm_finalize
.. _doxid-group__dnnl__api__ukernel__brgemm_1gaa25413b0931da573eda22fb2004d5eeb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_finalize(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm)

Finalizes initialization of a BRGeMM ukernel object.

This step is mandatory to query information from the object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- Output BRGeMM ukernel object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_get_B_pack_type
.. _doxid-group__dnnl__api__ukernel__brgemm_1gae918ea867b1a33522329695ad467f2ba:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_get_B_pack_type(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		:ref:`dnnl_pack_type_t<doxid-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97>`* pack_type
		)

Returns the packing type expected by a tensor B of a BRGeMM ukernel object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- pack_type

		- Output packing type. Can be ``dnnl_brgemm_no_pack`` if packing is not expected, and ``dnnl_brgemm_pack_32``, otherwise.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_get_scratchpad_size
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga7e9145a254a5807361252222f46c7236:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_get_scratchpad_size(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		size_t* size
		)

Returns the size of a scratchpad memory needed for the BRGeMM ukernel object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- size

		- Output size of a buffer required for the BRGeMM ukernel object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_set_hw_context
.. _doxid-group__dnnl__api__ukernel__brgemm_1gadce89da8fca565cbc41d0ed7bcbc2a62:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_set_hw_context(:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm)

Initializes the hardware-specific context.

If no initialization required, returns the success status.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_release_hw_context
.. _doxid-group__dnnl__api__ukernel__brgemm_1gaa804b65b7182dbc365ca465eb0651859:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_release_hw_context()

Releases the hardware-specific context.

Must be used after all the execution calls to BRGeMM ukernel objects.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_generate
.. _doxid-group__dnnl__api__ukernel__brgemm_1gaae5ee87a3fb22fce849a5c0de71d4cb6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_generate(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm)

Generates an executable part of BRGeMM ukernel object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_execute
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga73ad5c1d29039310540bfb243b4ce17d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_execute(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		const void* A_ptr,
		const void* B_ptr,
		const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* A_B_offsets,
		void* C_ptr,
		void* scratchpad_ptr
		)

Executes a BRGeMM ukernel object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- A_ptr

		- Base pointer to a tensor A.

	*
		- B_ptr

		- Base pointer to a tensor B.

	*
		- A_B_offsets

		- Pointer to the set of tensor A and tensor B offsets for each batch; the set must be contiguous in memory. Single batch should supply offsets for both tensors A and B simultaneously. The number of batches must coincide with the ``batch_size`` value passed at the creation stage.

	*
		- C_ptr

		- Pointer to a tensor C (accumulation buffer).

	*
		- scratchpad_ptr

		- Pointer to a scratchpad buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_execute_postops
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga3e92e657b97d1d5545a32ee9d9f57872:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_execute_postops(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		const void* A,
		const void* B,
		const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* A_B_offsets,
		const void* C_ptr,
		void* D_ptr,
		void* scratchpad_ptr,
		:ref:`const_dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1ga19b82b33015ab0abfe2630236d3da7fc>` attr_params
		)

Executes a BRGeMM ukernel object with post operations.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object.

	*
		- A

		- Base pointer to a tensor A.

	*
		- B

		- Base pointer to a tensor B.

	*
		- A_B_offsets

		- Pointer to a set of tensor A and tensor B offsets for each batch. A set must be contiguous in memory. A single batch should supply offsets for both tensors A and B simultaneously. The number of batches must coincide with the ``batch_size`` value passed at the creation stage.

	*
		- C_ptr

		- Pointer to a tensor C (accumulation buffer).

	*
		- D_ptr

		- Pointer to a tensor D (output buffer).

	*
		- scratchpad_ptr

		- Pointer to a scratchpad buffer.

	*
		- attr_params

		- Ukernel attributes memory storage.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_destroy
.. _doxid-group__dnnl__api__ukernel__brgemm_1gaa16a3c2c3657d5fe968fc58d651041fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_destroy(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm)

Destroys a BRGeMM ukernel object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm

		- BRGeMM ukernel object to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_transform_create
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga8b8ace47537f66365a9794c9f589d89d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_transform_create(
		:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga0a05ad64b8e8617112a045d12876b6e1>`* transform,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_pack_type_t<doxid-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97>` in_pack_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` in_ld,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` out_ld,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` in_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` out_dt
		)

Creates a transform object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- transform

		- Output transform object.

	*
		- K

		- Dimension K.

	*
		- N

		- Dimension N.

	*
		- in_pack_type

		- Input packing type. Must be one of ``dnnl_pack_type_no_trans``, or ``dnnl_pack_type_trans``.

	*
		- in_ld

		- Input leading dimension.

	*
		- out_ld

		- Output leading dimension. When packing data, it specifies a block by N dimension.

	*
		- in_dt

		- Input data type.

	*
		- out_dt

		- Output data type.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_transform_generate
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga75b9793b4f57eee2f4858c373e4cc49a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_transform_generate(:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga0a05ad64b8e8617112a045d12876b6e1>` transform)

Generates an executable part of transform object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- transform

		- Transform object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_transform_execute
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga3467d4f77ce81fb64065ca0fecb19226:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_transform_execute(
		:ref:`const_dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga1ef3c8a87d676f5b644c2677062ba485>` transform,
		const void* in_ptr,
		void* out_ptr
		)

Executes a transform object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- transform

		- Transform object.

	*
		- in_ptr

		- Pointer to an input buffer.

	*
		- out_ptr

		- Pointer to an output buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_transform_destroy
.. _doxid-group__dnnl__api__ukernel__brgemm_1gacea3f51d81b00d087fbd82259acaee4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_transform_destroy(:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__brgemm_1ga0a05ad64b8e8617112a045d12876b6e1>` transform)

Destroys a transform object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- transform

		- Transform object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

