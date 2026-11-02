.. index:: pair: group; Sum
.. _doxid-group__dnnl__api__sum:

Sum
===

.. toctree::
	:hidden:

	struct_dnnl_sum.rst

Overview
~~~~~~~~

A primitive to sum multiple tensors. :ref:`More...<details-group__dnnl__api__sum>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::sum<doxid-structdnnl_1_1sum>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_sum_primitive_desc_create<doxid-group__dnnl__api__sum_1ga10b304125badf7e33eea8ddead1f2e3e>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* sum_primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int n,
		const float* scales,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` const* src_descs,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__sum:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to sum multiple tensors.



.. rubric:: See also:

:ref:`Sum <doxid-dev_guide_sum>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_sum_primitive_desc_create
.. _doxid-group__dnnl__api__sum_1ga10b304125badf7e33eea8ddead1f2e3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_sum_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* sum_primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int n,
		const float* scales,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` const* src_descs,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for an (out-of-place) sum primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- sum_primitive_desc

		- Output primitive descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- n

		- Number of source parameters.

	*
		- scales

		- Vector of scales to multiply data in each source memory by.

	*
		- src_descs

		- Array of source memory descriptors having ``n`` elements.

	*
		- attr

		- Primitive attributes to use (can be NULL).

	*
		- engine

		- Engine to use.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

