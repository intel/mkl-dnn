.. index:: pair: group; Concat
.. _doxid-group__dnnl__api__concat:

Concat
======

.. toctree::
	:hidden:

	struct_dnnl_concat.rst

Overview
~~~~~~~~

A primitive to concatenate data by arbitrary dimension. :ref:`More...<details-group__dnnl__api__concat>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::concat<doxid-structdnnl_1_1concat>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_concat_primitive_desc_create<doxid-group__dnnl__api__concat_1ga1bf9669d55a86d8ac8ff10d3e28f52b8>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* concat_primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int n,
		int concat_dimension,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` const* src_descs,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__concat:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to concatenate data by arbitrary dimension.



.. rubric:: See also:

:ref:`Concat <doxid-dev_guide_concat>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_concat_primitive_desc_create
.. _doxid-group__dnnl__api__concat_1ga1bf9669d55a86d8ac8ff10d3e28f52b8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_concat_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* concat_primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		int n,
		int concat_dimension,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` const* src_descs,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for an out-of-place concatenation primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- concat_primitive_desc

		- Output primitive descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- n

		- Number of source parameters.

	*
		- concat_dimension

		- Source tensors will be concatenated over dimension with this index. Note that order of dimensions does not depend on memory format.

	*
		- src_descs

		- Array of source memory descriptors with ``n`` elements.

	*
		- attr

		- Primitive attributes to use (can be NULL).

	*
		- engine

		- Engine to use.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

