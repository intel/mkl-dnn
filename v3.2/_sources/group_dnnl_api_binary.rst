.. index:: pair: group; Binary
.. _doxid-group__dnnl__api__binary:

Binary
======

.. toctree::
	:hidden:

	struct_dnnl_binary.rst

Overview
~~~~~~~~

A primitive to perform tensor operations over two tensors. :ref:`More...<details-group__dnnl__api__binary>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::binary<doxid-structdnnl_1_1binary>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_binary_primitive_desc_create<doxid-group__dnnl__api__binary_1ga50078dffd48c6ebd6f6671b7656f5cdb>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src0_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src1_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__binary:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to perform tensor operations over two tensors.



.. rubric:: See also:

:ref:`Binary <doxid-dev_guide_binary>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_binary_primitive_desc_create
.. _doxid-group__dnnl__api__binary_1ga50078dffd48c6ebd6f6671b7656f5cdb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_binary_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src0_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src1_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a binary primitive.

.. note:: 

   Memory descriptors ``src1_desc`` and ``dst_desc`` are alloweded to be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.
   
   

.. note:: 

   Both memory descriptors must have the same number of dimensions. Element broadcasting is supported for memory descriptor ``src1_desc`` and are applied to ``src1_desc`` dimensions that have size equal to 1.



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

		- Algorithm kind. Valid values are :ref:`dnnl_binary_add <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad4c6d69ac6f6b443449923d51325886d>`, :ref:`dnnl_binary_mul <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ade272a5bcb8af2b2cb0bc691c78b4e36>`, :ref:`dnnl_binary_max <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23af93b25a1cd108fbecfdbee9f1cfcdd88>`, :ref:`dnnl_binary_min <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a21a9b503c9d06cea5f231fd170e623cc>`, :ref:`dnnl_binary_div <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23ad63a6855c4f438cabd245b0bbff61cf4>`, :ref:`dnnl_binary_sub <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a551dc23f954000fe81a97c9bd8ca4899>`, :ref:`dnnl_binary_ge <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a8303a5bb9566ad2cd1323653a81dc494>`, :ref:`dnnl_binary_gt <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23aae40b748b416aa218f420be2f6afbce4>`, :ref:`dnnl_binary_le <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23acd36606bc4250410a573a15b2a984457>`, :ref:`dnnl_binary_lt <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23abd093dc24480cf7a3e7a11c4d77dcafe>`, :ref:`dnnl_binary_eq <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a5b81e36f1c758682df8070d344d6f9b8>` and :ref:`dnnl_binary_ne <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a3f48bade6a3e91fc7880fe823bd4d263>`.

	*
		- src0_desc

		- Source 0 memory descriptor.

	*
		- src1_desc

		- Source 1 memory descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

