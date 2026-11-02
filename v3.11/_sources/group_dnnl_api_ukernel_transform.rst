.. index:: pair: group; Transform ukernel
.. _doxid-group__dnnl__api__ukernel__transform:

Transform ukernel
=================

.. toctree::
	:hidden:

	struct_dnnl_transform.rst
	struct_dnnl_ukernel_transform.rst

Overview
~~~~~~~~

Transform routines :ref:`More...<details-group__dnnl__api__ukernel__transform>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct :ref:`dnnl_transform<doxid-structdnnl__transform>`* :ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga0a05ad64b8e8617112a045d12876b6e1>`;
	typedef const struct :ref:`dnnl_transform<doxid-structdnnl__transform>`* :ref:`const_dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga1ef3c8a87d676f5b644c2677062ba485>`;

	// structs

	struct :ref:`dnnl_transform<doxid-structdnnl__transform>`;
	struct :ref:`dnnl::ukernel::transform<doxid-structdnnl_1_1ukernel_1_1transform>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_transform_create<doxid-group__dnnl__api__ukernel__transform_1ga8b8ace47537f66365a9794c9f589d89d>`(
		:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga0a05ad64b8e8617112a045d12876b6e1>`* transform,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` K,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` N,
		:ref:`dnnl_pack_type_t<doxid-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97>` in_pack_type,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` in_ld,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` out_ld,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` in_dt,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` out_dt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_transform_generate<doxid-group__dnnl__api__ukernel__transform_1ga75b9793b4f57eee2f4858c373e4cc49a>`(:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga0a05ad64b8e8617112a045d12876b6e1>` transform);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_transform_execute<doxid-group__dnnl__api__ukernel__transform_1ga3467d4f77ce81fb64065ca0fecb19226>`(
		:ref:`const_dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga1ef3c8a87d676f5b644c2677062ba485>` transform,
		const void* in_ptr,
		void* out_ptr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_transform_destroy<doxid-group__dnnl__api__ukernel__transform_1gacea3f51d81b00d087fbd82259acaee4b>`(:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga0a05ad64b8e8617112a045d12876b6e1>` transform);

.. _details-group__dnnl__api__ukernel__transform:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Transform routines

Typedefs
--------

.. index:: pair: typedef; dnnl_transform_t
.. _doxid-group__dnnl__api__ukernel__transform_1ga0a05ad64b8e8617112a045d12876b6e1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_transform<doxid-structdnnl__transform>`* dnnl_transform_t

A transform routine handle.

.. index:: pair: typedef; const_dnnl_transform_t
.. _doxid-group__dnnl__api__ukernel__transform_1ga1ef3c8a87d676f5b644c2677062ba485:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_transform<doxid-structdnnl__transform>`* const_dnnl_transform_t

A constant transform routine handle.

Global Functions
----------------

.. index:: pair: function; dnnl_transform_create
.. _doxid-group__dnnl__api__ukernel__transform_1ga8b8ace47537f66365a9794c9f589d89d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_transform_create(
		:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga0a05ad64b8e8617112a045d12876b6e1>`* transform,
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
.. _doxid-group__dnnl__api__ukernel__transform_1ga75b9793b4f57eee2f4858c373e4cc49a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_transform_generate(:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga0a05ad64b8e8617112a045d12876b6e1>` transform)

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
.. _doxid-group__dnnl__api__ukernel__transform_1ga3467d4f77ce81fb64065ca0fecb19226:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_transform_execute(
		:ref:`const_dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga1ef3c8a87d676f5b644c2677062ba485>` transform,
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
.. _doxid-group__dnnl__api__ukernel__transform_1gacea3f51d81b00d087fbd82259acaee4b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_transform_destroy(:ref:`dnnl_transform_t<doxid-group__dnnl__api__ukernel__transform_1ga0a05ad64b8e8617112a045d12876b6e1>` transform)

Destroys a transform object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- transform

		- Transform object.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

