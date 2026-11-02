.. index:: pair: group; LRN
.. _doxid-group__dnnl__api__lrn:

LRN
===

.. toctree::
	:hidden:

	struct_dnnl_lrn_backward.rst
	struct_dnnl_lrn_forward.rst

Overview
~~~~~~~~

A primitive to perform local response normalization (LRN) across or within channels. :ref:`More...<details-group__dnnl__api__lrn>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::lrn_backward<doxid-structdnnl_1_1lrn__backward>`;
	struct :ref:`dnnl::lrn_forward<doxid-structdnnl_1_1lrn__forward>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lrn_forward_primitive_desc_create<doxid-group__dnnl__api__lrn_1ga7d2550452cd5858747686b338cfde252>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` local_size,
		float alpha,
		float beta,
		float k,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_lrn_backward_primitive_desc_create<doxid-group__dnnl__api__lrn_1gafc38999581f962346f08517ef3383617>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` local_size,
		float alpha,
		float beta,
		float k,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__lrn:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to perform local response normalization (LRN) across or within channels.



.. rubric:: See also:

:ref:`Local Response Normalization (LRN) <doxid-dev_guide_lrn>` in developer guide

Global Functions
----------------

.. index:: pair: function; dnnl_lrn_forward_primitive_desc_create
.. _doxid-group__dnnl__api__lrn_1ga7d2550452cd5858747686b338cfde252:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_lrn_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` local_size,
		float alpha,
		float beta,
		float k,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for an LRN forward propagation primitive.



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
		- alg_kind

		- LRN algorithm kind: either :ref:`dnnl_lrn_across_channels <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a540b116253bf1290b9536929198d18fd>` or :ref:`dnnl_lrn_within_channel <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a922fdd348b6a3e6bbe589025691d7171>`.

	*
		- src_desc

		- Source memory descriptor.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- local_size

		- Regularization local size.

	*
		- alpha

		- The alpha regularization parameter.

	*
		- beta

		- The beta regularization parameter.

	*
		- k

		- The k regularization parameter.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_lrn_backward_primitive_desc_create
.. _doxid-group__dnnl__api__lrn_1gafc38999581f962346f08517ef3383617:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_lrn_backward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` local_size,
		float alpha,
		float beta,
		float k,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for an LRN backward propagation primitive.



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
		- alg_kind

		- LRN algorithm kind: either :ref:`dnnl_lrn_across_channels <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a540b116253bf1290b9536929198d18fd>` or :ref:`dnnl_lrn_within_channel <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a922fdd348b6a3e6bbe589025691d7171>`.

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
		- local_size

		- Regularization local size.

	*
		- alpha

		- The alpha regularization parameter.

	*
		- beta

		- The beta regularization parameter.

	*
		- k

		- The k regularization parameter.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

