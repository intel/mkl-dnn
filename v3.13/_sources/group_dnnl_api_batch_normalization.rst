.. index:: pair: group; Batch Normalization
.. _doxid-group__dnnl__api__batch__normalization:

Batch Normalization
===================

.. toctree::
	:hidden:

	struct_dnnl_batch_normalization_backward.rst
	struct_dnnl_batch_normalization_forward.rst

Overview
~~~~~~~~

A primitive to perform batch normalization. :ref:`More...<details-group__dnnl__api__batch__normalization>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::batch_normalization_backward<doxid-structdnnl_1_1batch__normalization__backward>`;
	struct :ref:`dnnl::batch_normalization_forward<doxid-structdnnl_1_1batch__normalization__forward>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_batch_normalization_forward_primitive_desc_create<doxid-group__dnnl__api__batch__normalization_1ga65dc3c0a16325b360a7e6fb676900b2b>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_batch_normalization_backward_primitive_desc_create<doxid-group__dnnl__api__batch__normalization_1ga851cc56a63560c3de6217d6d812d169f>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__batch__normalization:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to perform batch normalization.

Both forward and backward propagation primitives support in-place operation; that is, src and dst can refer to the same memory for forward propagation, and diff_dst and diff_src can refer to the same memory for backward propagation.

The batch normalization primitives computations can be controlled by specifying different :ref:`dnnl::normalization_flags <doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>` values. For example, batch normalization forward propagation can be configured to either compute the mean and variance or take them as arguments. It can either perform scaling and shifting using gamma and beta parameters or not. Optionally, it can also perform a fused ReLU, which in case of training would also require a workspace.



.. rubric:: See also:

:ref:`Batch Normalization <doxid-dev_guide_batch_normalization>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_batch_normalization_forward_primitive_desc_create
.. _doxid-group__dnnl__api__batch__normalization_1ga65dc3c0a16325b360a7e6fb676900b2b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_batch_normalization_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a batch normalization forward propagation primitive.

.. note:: 

   In-place operation is supported: the dst can refer to the same memory as the src.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive_descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

	*
		- src_desc

		- Source memory descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- epsilon

		- Batch normalization epsilon parameter.

	*
		- flags

		- Batch normalization flags (:ref:`dnnl_normalization_flags_t <doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>`).

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_batch_normalization_backward_primitive_desc_create
.. _doxid-group__dnnl__api__batch__normalization_1ga851cc56a63560c3de6217d6d812d169f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_batch_normalization_backward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		float epsilon,
		unsigned flags,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a batch normalization backward propagation primitive.

.. note:: 

   In-place operation is supported: the diff_dst can refer to the same memory as the diff_src.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- primitive_desc

		- Output primitive_descriptor.

	*
		- engine

		- Engine to use.

	*
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_backward_data <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a524dd6cb2ed9680bbd170ba15261d218>` and :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>` (diffs for all parameters are computed in this case).

	*
		- diff_src_desc

		- Diff source memory descriptor.

	*
		- diff_dst_desc

		- Diff destination memory descriptor.

	*
		- src_desc

		- Source memory descriptor.

	*
		- epsilon

		- Batch normalization epsilon parameter.

	*
		- flags

		- Batch normalization flags (:ref:`dnnl_normalization_flags_t <doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>`).

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

