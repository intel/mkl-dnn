.. index:: pair: group; Reduction
.. _doxid-group__dnnl__api__reduction:

Reduction
=========

.. toctree::
	:hidden:

	struct_dnnl_reduction.rst

Overview
~~~~~~~~

A primitive to compute reduction operation on data tensor using min, max, mul, sum, mean and norm_lp operations. :ref:`More...<details-group__dnnl__api__reduction>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::reduction<doxid-structdnnl_1_1reduction>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_reduction_primitive_desc_create<doxid-group__dnnl__api__reduction_1gab26d42a8553d69b5fc3fbdb3b44d3f98>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		float p,
		float eps,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__reduction:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to compute reduction operation on data tensor using min, max, mul, sum, mean and norm_lp operations.



.. rubric:: See also:

:ref:`Reduction <doxid-dev_guide_reduction>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_reduction_primitive_desc_create
.. _doxid-group__dnnl__api__reduction_1gab26d42a8553d69b5fc3fbdb3b44d3f98:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_reduction_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		float p,
		float eps,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a reduction primitive.

.. note:: 

   Destination memory descriptor is allowed to be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive descriptor.

	*
		- engine

		- Engine to use.

	*
		- alg_kind

		- reduction algorithm kind. Possible values: :ref:`dnnl_reduction_max <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aae4722e394206cf9774ae45db959854e>`, :ref:`dnnl_reduction_min <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3edeac87290d164cfd3e79adcb6ed91a>`, :ref:`dnnl_reduction_sum <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ae74491a0b7bfe0720be69e3732894818>`, :ref:`dnnl_reduction_mul <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9ff432e67749e211f5f0f64d5f707359>`, :ref:`dnnl_reduction_mean <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ac88d2b9bc130483c177868888c705694>`, :ref:`dnnl_reduction_norm_lp_max <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad6459b4162ab312f59fa48bf9dcf35c3>`, :ref:`dnnl_reduction_norm_lp_sum <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a21c93597a1be438219bbbd832830f096>`, :ref:`dnnl_reduction_norm_lp_power_p_max <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3838df4d5d37de3237359043ccebfba1>`, :ref:`dnnl_reduction_norm_lp_power_p_sum <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23adcb83e9f76b3beaeb831a59cd257d7dd>`.

	*
		- p

		- Algorithm specific parameter.

	*
		- eps

		- Algorithm specific parameter.

	*
		- src_desc

		- Source memory descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

