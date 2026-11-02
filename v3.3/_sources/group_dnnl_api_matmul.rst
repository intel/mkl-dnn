.. index:: pair: group; Matrix Multiplication
.. _doxid-group__dnnl__api__matmul:

Matrix Multiplication
=====================

.. toctree::
	:hidden:

	struct_dnnl_matmul.rst

Overview
~~~~~~~~

A primitive to perform matrix-matrix multiplication. :ref:`More...<details-group__dnnl__api__matmul>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::matmul<doxid-structdnnl_1_1matmul>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_matmul_primitive_desc_create<doxid-group__dnnl__api__matmul_1gaac0ca3eed6070331c7d4020028b00fe6>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__matmul:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to perform matrix-matrix multiplication.

The batched mode is supported with 3D tensors.



.. rubric:: See also:

:ref:`Matrix Multiplication <doxid-dev_guide_matmul>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_matmul_primitive_desc_create
.. _doxid-group__dnnl__api__matmul_1gaac0ca3eed6070331c7d4020028b00fe6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_matmul_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a matrix multiplication primitive.



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
		- src_desc

		- Source memory descriptor (matrix A)

	*
		- weights_desc

		- Weights memory descriptor (matrix B)

	*
		- bias_desc

		- Bias memory descriptor. Passing NULL, a zero memory descriptor, or a memory descriptor with format_kind set to :ref:`dnnl_format_kind_undef <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367dac86d377bba856ea7aa9679ecf65c8364>` disables the bias term.

	*
		- dst_desc

		- Destination memory descriptor (matrix C).

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

