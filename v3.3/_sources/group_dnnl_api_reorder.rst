.. index:: pair: group; Reorder
.. _doxid-group__dnnl__api__reorder:

Reorder
=======

.. toctree::
	:hidden:

	struct_dnnl_reorder.rst

Overview
~~~~~~~~

A primitive to copy data between two memory objects. :ref:`More...<details-group__dnnl__api__reorder>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::reorder<doxid-structdnnl_1_1reorder>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_reorder_primitive_desc_create<doxid-group__dnnl__api__reorder_1ga20e455d1b6b20fb8a2a9210def44263b>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* reorder_primitive_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` src_engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` dst_engine,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__reorder:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to copy data between two memory objects.

This primitive is typically used to change the way the data is laid out in memory.



.. rubric:: See also:

:ref:`Reorder <doxid-dev_guide_reorder>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_reorder_primitive_desc_create
.. _doxid-group__dnnl__api__reorder_1ga20e455d1b6b20fb8a2a9210def44263b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_reorder_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* reorder_primitive_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` src_engine,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` dst_engine,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a reorder primitive.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- reorder_primitive_desc

		- Output primitive descriptor.

	*
		- src_desc

		- Source memory descriptor.

	*
		- src_engine

		- Engine on which the source memory object will be located.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- dst_engine

		- Engine on which the destination memory object will be located.

	*
		- attr

		- Primitive attributes to use (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

