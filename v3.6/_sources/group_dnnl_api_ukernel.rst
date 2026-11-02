.. index:: pair: group; Ukernels
.. _doxid-group__dnnl__api__ukernel:

Ukernels
========

.. toctree::
	:hidden:

	group_dnnl_api_ukernel_brgemm.rst
	namespace_dnnl_ukernel.rst
	enum_dnnl_pack_type_t.rst
	struct_dnnl_ukernel_attr_params.rst

Overview
~~~~~~~~

Collection of ukernels. :ref:`More...<details-group__dnnl__api__ukernel>`

|	:ref:`BRGeMM ukernel<doxid-group__dnnl__api__ukernel__brgemm>`



.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// namespaces

	namespace :ref:`dnnl::ukernel<doxid-namespacednnl_1_1ukernel>`;

	// typedefs

	typedef struct :ref:`dnnl_ukernel_attr_params<doxid-structdnnl__ukernel__attr__params>`* :ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>`;
	typedef const struct :ref:`dnnl_ukernel_attr_params<doxid-structdnnl__ukernel__attr__params>`* :ref:`const_dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1ga19b82b33015ab0abfe2630236d3da7fc>`;

	// enums

	enum :ref:`dnnl_pack_type_t<doxid-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97>`;

	// structs

	struct :ref:`dnnl_ukernel_attr_params<doxid-structdnnl__ukernel__attr__params>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ukernel_attr_params_create<doxid-group__dnnl__api__ukernel_1gab2f224520506861ae9b8c146a4442135>`(:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>`* attr_params);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ukernel_attr_params_set_post_ops_args<doxid-group__dnnl__api__ukernel_1ga3bb0b636e6d8c0f3d02881a400ed9699>`(
		:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params,
		const void** post_ops_args
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ukernel_attr_params_set_A_scales<doxid-group__dnnl__api__ukernel_1gacfdd737a8cc310a757296b85b40ae968>`(
		:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params,
		const void* a_scales
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ukernel_attr_params_set_B_scales<doxid-group__dnnl__api__ukernel_1ga494c64f1a6d08fac2d6b73405c6cda03>`(
		:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params,
		const void* b_scales
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ukernel_attr_params_set_D_scales<doxid-group__dnnl__api__ukernel_1ga6bf4e2f2a72c9096d2e5d3c82d1e3d3a>`(
		:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params,
		const void* d_scales
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_ukernel_attr_params_destroy<doxid-group__dnnl__api__ukernel_1ga27ac291dec0aa15302586f1324662b8c>`(:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params);

.. _details-group__dnnl__api__ukernel:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Collection of ukernels.

Typedefs
--------

.. index:: pair: typedef; dnnl_ukernel_attr_params_t
.. _doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_ukernel_attr_params<doxid-structdnnl__ukernel__attr__params>`* dnnl_ukernel_attr_params_t

A ukernel attributes memory storage handle.

.. index:: pair: typedef; const_dnnl_ukernel_attr_params_t
.. _doxid-group__dnnl__api__ukernel_1ga19b82b33015ab0abfe2630236d3da7fc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_ukernel_attr_params<doxid-structdnnl__ukernel__attr__params>`* const_dnnl_ukernel_attr_params_t

A constant ukernel attributes memory storage handle.

Global Functions
----------------

.. index:: pair: function; dnnl_ukernel_attr_params_create
.. _doxid-group__dnnl__api__ukernel_1gab2f224520506861ae9b8c146a4442135:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ukernel_attr_params_create(:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>`* attr_params)

Creates a ukernel attributes memory storage.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr_params

		- Output ukernel attributes memory storage.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ukernel_attr_params_set_post_ops_args
.. _doxid-group__dnnl__api__ukernel_1ga3bb0b636e6d8c0f3d02881a400ed9699:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ukernel_attr_params_set_post_ops_args(
		:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params,
		const void** post_ops_args
		)

Sets post-operations arguments to a storage.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr_params

		- Memory pointers storage object.

	*
		- post_ops_args

		- A pointer to pointers of post_ops storages. Expected to be packed together.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ukernel_attr_params_set_A_scales
.. _doxid-group__dnnl__api__ukernel_1gacfdd737a8cc310a757296b85b40ae968:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ukernel_attr_params_set_A_scales(
		:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params,
		const void* a_scales
		)

Sets tensor A scales argument to a storage.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr_params

		- Memory pointers storage object.

	*
		- a_scales

		- Pointer to the scales storage.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ukernel_attr_params_set_B_scales
.. _doxid-group__dnnl__api__ukernel_1ga494c64f1a6d08fac2d6b73405c6cda03:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ukernel_attr_params_set_B_scales(
		:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params,
		const void* b_scales
		)

Sets tensor B scales argument to a storage.

If ``dnnl_brgemm_set_B_scales`` used mask of 2, then at least N values of selected data type are expected.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr_params

		- Memory pointers storage object.

	*
		- b_scales

		- Pointer to the scales storage.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ukernel_attr_params_set_D_scales
.. _doxid-group__dnnl__api__ukernel_1ga6bf4e2f2a72c9096d2e5d3c82d1e3d3a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ukernel_attr_params_set_D_scales(
		:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params,
		const void* d_scales
		)

Sets tensor D scales argument to a storage.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr_params

		- Memory pointers storage object.

	*
		- d_scales

		- Pointer to the scales storage.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_ukernel_attr_params_destroy
.. _doxid-group__dnnl__api__ukernel_1ga27ac291dec0aa15302586f1324662b8c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_ukernel_attr_params_destroy(:ref:`dnnl_ukernel_attr_params_t<doxid-group__dnnl__api__ukernel_1gaf7f7dfb78cca7754753dd845d362f515>` attr_params)

Destroys a ukernel attributes memory storage.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- attr_params

		- Memory pointers storage object to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

