.. index:: pair: group; Deconvolution
.. _doxid-group__dnnl__api__deconvolution:

Deconvolution
=============

.. toctree::
	:hidden:

	struct_dnnl_deconvolution_backward_data.rst
	struct_dnnl_deconvolution_backward_weights.rst
	struct_dnnl_deconvolution_forward.rst

Overview
~~~~~~~~

A primitive to perform 1D, 2D or 3D deconvolution. :ref:`More...<details-group__dnnl__api__deconvolution>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// structs

	struct :ref:`dnnl::deconvolution_backward_data<doxid-structdnnl_1_1deconvolution__backward__data>`;
	struct :ref:`dnnl::deconvolution_backward_weights<doxid-structdnnl_1_1deconvolution__backward__weights>`;
	struct :ref:`dnnl::deconvolution_forward<doxid-structdnnl_1_1deconvolution__forward>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_deconvolution_forward_primitive_desc_create<doxid-group__dnnl__api__deconvolution_1gaf0d6b55570014911d30a867e3de12258>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_deconvolution_backward_data_primitive_desc_create<doxid-group__dnnl__api__deconvolution_1ga531dbfb58d4fe4526c96c982dd13780c>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_deconvolution_backward_weights_primitive_desc_create<doxid-group__dnnl__api__deconvolution_1ga45ad0e8c95597f9dc8fc36676cf38f55>`(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		);

.. _details-group__dnnl__api__deconvolution:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A primitive to perform 1D, 2D or 3D deconvolution.

Supported variants are forward propagation, backward propagation, and weights gradient with or without bias.

Global Functions
----------------

.. index:: pair: function; dnnl_deconvolution_forward_primitive_desc_create
.. _doxid-group__dnnl__api__deconvolution_1gaf0d6b55570014911d30a867e3de12258:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_deconvolution_forward_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_prop_kind_t<doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>` prop_kind,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a deconvolution forward propagation primitive.

.. note:: 

   Memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.
   
   
Arrays ``strides``, ``dilates``, ``padding_l``, and ``padding_r`` contain values for spatial dimensions only and hence must have the same number of elements as there are spatial dimensions. The order of values is the same as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.



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
		- prop_kind

		- Propagation kind. Possible values are :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>`.

	*
		- alg_kind

		- Deconvolution algorithm. Possible values are :ref:`dnnl_deconvolution_direct <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a575e3d69d108a8a1e62af755dda0ef5f>`, :ref:`dnnl_deconvolution_winograd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9b11a60748225144fdb960988e9b0cb9>`.

	*
		- src_desc

		- Source memory descriptor.

	*
		- weights_desc

		- Weights memory descriptor.

	*
		- bias_desc

		- Bias memory descriptor. Passing NULL, a zero memory descriptor, or a memory descriptor with format_kind set to :ref:`dnnl_format_kind_undef <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367dac86d377bba856ea7aa9679ecf65c8364>` disables the bias term.

	*
		- dst_desc

		- Destination memory descriptor.

	*
		- strides

		- Array of strides for spatial dimension.

	*
		- dilates

		- Array of dilations for spatial dimension. A zero value means no dilation in the corresponding dimension.

	*
		- padding_l

		- Array of padding values for low indices for each spatial dimension ``([[front,] top,] left)``.

	*
		- padding_r

		- Array of padding values for high indices for each spatial dimension ``([[back,] bottom,] right)``. Can be NULL in which case padding is considered to be symmetrical.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_deconvolution_backward_data_primitive_desc_create
.. _doxid-group__dnnl__api__deconvolution_1ga531dbfb58d4fe4526c96c982dd13780c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_deconvolution_backward_data_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a deconvolution backward propagation primitive.

.. note:: 

   Memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.
   
   
Arrays ``strides``, ``dilates``, ``padding_l``, and ``padding_r`` contain values for spatial dimensions only and hence must have the same number of elements as there are spatial dimensions. The order of values is the same as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.



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

		- Deconvolution algorithm. Possible values are :ref:`dnnl_deconvolution_direct <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a575e3d69d108a8a1e62af755dda0ef5f>`, :ref:`dnnl_deconvolution_winograd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9b11a60748225144fdb960988e9b0cb9>`.

	*
		- diff_src_desc

		- Diff source memory descriptor.

	*
		- weights_desc

		- Weights memory descriptor.

	*
		- diff_dst_desc

		- Diff destination memory descriptor.

	*
		- strides

		- Array of strides for spatial dimension.

	*
		- dilates

		- Array of dilations for spatial dimension. A zero value means no dilation in the corresponding dimension.

	*
		- padding_l

		- Array of padding values for low indices for each spatial dimension ``([[front,] top,] left)``.

	*
		- padding_r

		- Array of padding values for high indices for each spatial dimension ``([[back,] bottom,] right)``. Can be NULL in which case padding is considered to be symmetrical.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_deconvolution_backward_weights_primitive_desc_create
.. _doxid-group__dnnl__api__deconvolution_1ga45ad0e8c95597f9dc8fc36676cf38f55:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_deconvolution_backward_weights_primitive_desc_create(
		:ref:`dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gaabde3e27edf071b62b39f47bace7efd6>`* primitive_desc,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_alg_kind_t<doxid-group__dnnl__api__primitives__common_1ga96946c805f6c4922c38c37049ab95d23>` alg_kind,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` src_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_weights_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_bias_desc,
		:ref:`const_dnnl_memory_desc_t<doxid-group__dnnl__api__memory_1ga402f0cb4399cd56445803cfa433aac6d>` diff_dst_desc,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dilates,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_l,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` padding_r,
		:ref:`const_dnnl_primitive_desc_t<doxid-group__dnnl__api__primitives__common_1gab604e76e39a791e855bc6bb4ee13c63f>` hint_fwd_pd,
		:ref:`const_dnnl_primitive_attr_t<doxid-group__dnnl__api__attributes_1ga871d7ee732a90fec43f1c878581bb59a>` attr
		)

Creates a primitive descriptor for a deconvolution weights gradient primitive.

.. note:: 

   Memory descriptors can be initialized with :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>` or with format_kind set to :ref:`dnnl_format_kind_any <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`.
   
   
Arrays ``strides``, ``dilates``, ``padding_l``, and ``padding_r`` contain values for spatial dimensions only and hence must have the same number of elements as there are spatial dimensions. The order of values is the same as in the tensor: depth (for 3D tensors), height (for 3D and 2D tensors), and width.



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

		- Deconvolution algorithm. Possible values are :ref:`dnnl_deconvolution_direct <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a575e3d69d108a8a1e62af755dda0ef5f>`, :ref:`dnnl_deconvolution_winograd <doxid-group__dnnl__api__primitives__common_1gga96946c805f6c4922c38c37049ab95d23a9b11a60748225144fdb960988e9b0cb9>`.

	*
		- src_desc

		- Source memory descriptor.

	*
		- diff_weights_desc

		- Diff weights memory descriptor.

	*
		- diff_bias_desc

		- Diff bias memory descriptor. Passing NULL, a zero memory descriptor, or a memory descriptor with format_kind set to :ref:`dnnl_format_kind_undef <doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367dac86d377bba856ea7aa9679ecf65c8364>` disables the bias term.

	*
		- diff_dst_desc

		- Diff destination memory descriptor.

	*
		- strides

		- Array of strides for spatial dimension.

	*
		- dilates

		- Array of dilations for spatial dimension. A zero value means no dilation in the corresponding dimension.

	*
		- padding_l

		- Array of padding values for low indices for each spatial dimension ``([[front,] top,] left)``.

	*
		- padding_r

		- Array of padding values for high indices for each spatial dimension ``([[back,] bottom,] right)``. Can be NULL in which case padding is considered to be symmetrical.

	*
		- hint_fwd_pd

		- Primitive descriptor for a respective forward propagation primitive.

	*
		- attr

		- Primitive attributes (can be NULL).



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

