.. index:: pair: group; BRGeMM ukernel
.. _doxid-group__dnnl__api__ukernel__brgemm:

BRGeMM ukernel
==============

.. toctree::
	:hidden:

	struct_dnnl_ukernel_brgemm.rst
	struct_dnnl_ukernel_brgemm_pack_b.rst
	struct_dnnl_brgemm.rst
	struct_dnnl_brgemm_pack_b.rst

Overview
~~~~~~~~

BRGeMM ukernel routines. :ref:`More...<details-group__dnnl__api__ukernel__brgemm>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct :ref:`dnnl_brgemm<doxid-structdnnl__brgemm>`* :ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>`;
	typedef const struct :ref:`dnnl_brgemm<doxid-structdnnl__brgemm>`* :ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>`;
	typedef struct :ref:`dnnl_brgemm_pack_b<doxid-structdnnl__brgemm__pack___b>`* :ref:`dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1ga9a2e5ffe68e34041379663ebba70551c>`;
	typedef const struct :ref:`dnnl_brgemm_pack_b<doxid-structdnnl__brgemm__pack___b>`* :ref:`const_dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1gad8575d541cb84b7c2c986153bcb05922>`;

	// structs

	struct :ref:`dnnl::ukernel::brgemm<doxid-structdnnl_1_1ukernel_1_1brgemm>`;
	struct :ref:`dnnl::ukernel::brgemm_pack_b<doxid-structdnnl_1_1ukernel_1_1brgemm__pack___b>`;
	struct :ref:`dnnl_brgemm<doxid-structdnnl__brgemm>`;
	struct :ref:`dnnl_brgemm_pack_b<doxid-structdnnl__brgemm__pack___b>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_create<doxid-group__dnnl__api__ukernel__brgemm_1ga40b46767adf2abf826bebc91d77127b5>`(
		:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>`* brgemm,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` M,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` batch_size,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` lda,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldb,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldd,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` c_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` d_dt,
		float alpha,
		float beta,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
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

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_execute_postops<doxid-group__dnnl__api__ukernel__brgemm_1gaacd9727cf708b7f38c7dee269458de83>`(
		:ref:`const_dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1ga947c9620af194ed1a775145e6f76f467>` brgemm,
		const void* A,
		const void* B,
		const :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>`* A_B_offsets,
		const void* C_ptr,
		void* D_ptr,
		void* scratchpad_ptr,
		const void* binary_po_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_destroy<doxid-group__dnnl__api__ukernel__brgemm_1gaa16a3c2c3657d5fe968fc58d651041fa>`(:ref:`dnnl_brgemm_t<doxid-group__dnnl__api__ukernel__brgemm_1gadd52a9a48f0d80d17777d9a2e6484dea>` brgemm);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_pack_b_create<doxid-group__dnnl__api__ukernel__brgemm_1gaee69a9efed11a9f466ff390a9b530abc>`(
		:ref:`dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1ga9a2e5ffe68e34041379663ebba70551c>`* brgemm_pack_b,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` in_ld,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` out_ld,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` in_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` out_dt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_pack_b_need_pack<doxid-group__dnnl__api__ukernel__brgemm_1ga42adc7638688ad9c03e851b554129227>`(
		:ref:`const_dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1gad8575d541cb84b7c2c986153bcb05922>` brgemm_pack_b,
		int* need_pack
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_pack_b_generate<doxid-group__dnnl__api__ukernel__brgemm_1ga4898736a5b9aab2233c0b4ae2ab4e00d>`(:ref:`dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1ga9a2e5ffe68e34041379663ebba70551c>` brgemm_pack_b);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_pack_b_execute<doxid-group__dnnl__api__ukernel__brgemm_1ga02eeaff0d947c494384f4f0668d93f4c>`(
		:ref:`const_dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1gad8575d541cb84b7c2c986153bcb05922>` brgemm_pack_b,
		const void* in_ptr,
		void* out_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_brgemm_pack_b_destroy<doxid-group__dnnl__api__ukernel__brgemm_1gaeeff5faf1520fc07822b29cce773f855>`(:ref:`dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1ga9a2e5ffe68e34041379663ebba70551c>` brgemm_pack_b);

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

.. index:: pair: typedef; dnnl_brgemm_pack_b_t
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga9a2e5ffe68e34041379663ebba70551c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_brgemm_pack_b<doxid-structdnnl__brgemm__pack___b>`* dnnl_brgemm_pack_b_t

A brgemm ukernel packing B routine handle.

.. index:: pair: typedef; const_dnnl_brgemm_pack_b_t
.. _doxid-group__dnnl__api__ukernel__brgemm_1gad8575d541cb84b7c2c986153bcb05922:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_brgemm_pack_b<doxid-structdnnl__brgemm__pack___b>`* const_dnnl_brgemm_pack_b_t

A constant brgemm ukernel packing B routine handle.

Global Functions
----------------

.. index:: pair: function; dnnl_brgemm_create
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga40b46767adf2abf826bebc91d77127b5:

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
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` ldd,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` a_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` b_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` c_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` d_dt,
		float alpha,
		float beta,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a BRGeMM ukernel object.

Operates by the following formula: ``C = alpha * [A x B] + beta * C``. ``D = post-operations(C)``.

Post-operations applies if one of the following holds:

* Non-empty attributes are specified.

* Output data type ``d_dt`` is different from accumulation data type ``c_dt``.

If any of conditions happens, the final call of the accumulation chain must be ``dnnl_brgemm_execute_postops``, and ``dnnl_brgemm_execute``, otherwise.



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
		- ldd

		- Leading dimension of tensor D.

	*
		- a_dt

		- Data type of tensor A.

	*
		- b_dt

		- Data type of tensor B.

	*
		- c_dt

		- Data type of tensor C. Must be dnnl_f32.

	*
		- d_dt

		- Data type of tensor D.

	*
		- alpha

		- Scale for an accumulation output.

	*
		- beta

		- Scale for a tensor C to append on an accumulation output.

	*
		- attr

		- Primitive attributes to extend the kernel operations.



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
.. _doxid-group__dnnl__api__ukernel__brgemm_1gaacd9727cf708b7f38c7dee269458de83:

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
		const void* binary_po_ptr
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
		- binary_po_ptr

		- Pointer to binary post-op data.



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

.. index:: pair: function; dnnl_brgemm_pack_b_create
.. _doxid-group__dnnl__api__ukernel__brgemm_1gaee69a9efed11a9f466ff390a9b530abc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_pack_b_create(
		:ref:`dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1ga9a2e5ffe68e34041379663ebba70551c>`* brgemm_pack_b,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` in_ld,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` out_ld,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` in_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` out_dt
		)

Creates a BRGeMM ukernel packing tensor B object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm_pack_b

		- Output BRGeMM ukernel packing B object.

	*
		- K

		- Dimension K.

	*
		- N

		- Dimension N.

	*
		- in_ld

		- Input leading dimension.

	*
		- out_ld

		- Output leading dimension. Specifies a block by N dimension during data packing.

	*
		- in_dt

		- Input data type.

	*
		- out_dt

		- Output data type.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_pack_b_need_pack
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga42adc7638688ad9c03e851b554129227:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_pack_b_need_pack(
		:ref:`const_dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1gad8575d541cb84b7c2c986153bcb05922>` brgemm_pack_b,
		int* need_pack
		)

Returns the flag if packing is expected by BRGeMM ukernel kernel.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm_pack_b

		- BRGeMM ukernel packing B object.

	*
		- need_pack

		- Output flag specifying if packing is needed. Possible values are 0 (not needed) and 1 (needed).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_pack_b_generate
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga4898736a5b9aab2233c0b4ae2ab4e00d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_pack_b_generate(:ref:`dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1ga9a2e5ffe68e34041379663ebba70551c>` brgemm_pack_b)

Generates an executable part of BRGeMM ukernel packing B object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm_pack_b

		- BRGeMM ukernel packing B object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_pack_b_execute
.. _doxid-group__dnnl__api__ukernel__brgemm_1ga02eeaff0d947c494384f4f0668d93f4c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_pack_b_execute(
		:ref:`const_dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1gad8575d541cb84b7c2c986153bcb05922>` brgemm_pack_b,
		const void* in_ptr,
		void* out_ptr
		)

Executes a BRGeMM ukernel packing tensor B object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm_pack_b

		- BRGeMM ukernel packing B object.

	*
		- in_ptr

		- Pointer to an input buffer.

	*
		- out_ptr

		- Pointer to an output buffer.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_brgemm_pack_b_destroy
.. _doxid-group__dnnl__api__ukernel__brgemm_1gaeeff5faf1520fc07822b29cce773f855:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_brgemm_pack_b_destroy(:ref:`dnnl_brgemm_pack_b_t<doxid-group__dnnl__api__ukernel__brgemm_1ga9a2e5ffe68e34041379663ebba70551c>` brgemm_pack_b)

Destroys a BRGeMM ukernel packing tensor B object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- brgemm_pack_b

		- BRGeMM ukernel packing B object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

