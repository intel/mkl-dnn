.. index:: pair: group; Logical Tensor
.. _doxid-group__dnnl__graph__api__logical__tensor:

Logical Tensor
==============

.. toctree::
	:hidden:

	enum_dnnl_graph_layout_type_t.rst
	enum_dnnl_graph_tensor_property_t.rst
	struct_dnnl_graph_logical_tensor_t.rst
	class_dnnl_graph_logical_tensor.rst

Overview
~~~~~~~~

Logical tensor describes the meta-data of the input or output tensor, like elements data type, number of dimensions, size for each dimension (shape), layout, and the property of the tensor. :ref:`More...<details-group__dnnl__graph__api__logical__tensor>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>`;
	enum :ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>`;

	// structs

	struct :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`;

	// classes

	class :ref:`dnnl::graph::logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_init<doxid-group__dnnl__graph__api__logical__tensor_1gab18ae5c4f5bfe5bd966305d9c2690a7e>`(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		:ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>` ltype,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_init_with_dims<doxid-group__dnnl__graph__api__logical__tensor_1ga13f140ecc327c9d8acb5a5832b2d0710>`(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>` ltype,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_init_with_strides<doxid-group__dnnl__graph__api__logical__tensor_1ga719f24a5aec5fc929a3ab620d6d5dc97>`(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_get_mem_size<doxid-group__dnnl__graph__api__logical__tensor_1ga56f57a976b591e6d428daea2f115207c>`(
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t* size
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_logical_tensor_is_equal<doxid-group__dnnl__graph__api__logical__tensor_1gacc21c4aa2240c9a56616259e7ed71df0>`(
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt1,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt2,
		uint8_t* is_equal
		);

	// macros

	#define :ref:`DNNL_GRAPH_UNKNOWN_DIM<doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84>`
	#define :ref:`DNNL_GRAPH_UNKNOWN_NDIMS<doxid-group__dnnl__graph__api__logical__tensor_1ga49497533d28f67dc4cce08fe210bf4bf>`

.. _details-group__dnnl__graph__api__logical__tensor:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Logical tensor describes the meta-data of the input or output tensor, like elements data type, number of dimensions, size for each dimension (shape), layout, and the property of the tensor.

Each logical tensor has an unique ID. The library uses logical tensor IDs to build up the connections between operations if the output of one operation has the same ID as the input of another operation. The meta-data in a logical tensor may be enriched in the framework graph as it progresses toward final execution. For example, the library doesn't require detailed shape information at the operation and graph creation stage. But shape information of input logical tensor will be required at partition compilation stage. Logical tensor is not mutable. Users must create a new logical tensor with the same ID to pass any new additional information to oneDNN Graph API. Please note that the library also has unique IDs for operations. The ID should be unique among different logical tensors, but it can have the same value between a logical tensor and an operation.

Global Functions
----------------

.. index:: pair: function; dnnl_graph_logical_tensor_init
.. _doxid-group__dnnl__graph__api__logical__tensor_1gab18ae5c4f5bfe5bd966305d9c2690a7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_logical_tensor_init(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		:ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>` ltype,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		)

Initializes a logical tensor with id, data type, number of dimensions, layout type, and property.

The logical tensor's dims are unknown with this interface.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- logical_tensor

		- Output logical tensor.

	*
		- tid

		- The unique id of the output logical tensor.

	*
		- dtype

		- Elements data type.

	*
		- ndims

		- Number of dimensions.

	*
		- ltype

		- Layout type of the underlying tensor buffer.

	*
		- ptype

		- Tensor property type.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_logical_tensor_init_with_dims
.. _doxid-group__dnnl__graph__api__logical__tensor_1ga13f140ecc327c9d8acb5a5832b2d0710:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_logical_tensor_init_with_dims(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		:ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>` ltype,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		)

Initializes a logical tensor with basic information and dims.

The logical tensor's dimensions and layout will be initialized according to the input arguments.

.. note:: 

   If dims contains all valid values and layout type is :ref:`dnnl_graph_layout_type_strided <doxid-group__dnnl__graph__api__logical__tensor_1gga5b552d8a81835eb955253410bf012694aa9ea14026cc47aafffdcb92c00a1b1ea>`. The strides field in :ref:`dnnl_graph_logical_tensor_t <doxid-structdnnl__graph__logical__tensor__t>` will be calculated in a row major and contiguous way. Otherwise, Accessing the strides field is an undefined behavior.
   
   
Eg. dims (2, 3, 4, 5) will get strides (60, 20, 5, 1)



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- logical_tensor

		- Output logical tensor.

	*
		- tid

		- The unique id of output logical tensor.

	*
		- dtype

		- Elements data type.

	*
		- ndims

		- Number of dimensions.

	*
		- dims

		- Array of dimensions.

	*
		- ltype

		- Layout type of the underlying tensor memory.

	*
		- ptype

		- Tensor property type.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_logical_tensor_init_with_strides
.. _doxid-group__dnnl__graph__api__logical__tensor_1ga719f24a5aec5fc929a3ab620d6d5dc97:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_logical_tensor_init_with_strides(
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t tid,
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` dtype,
		int32_t ndims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims,
		const :ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides,
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` ptype
		)

Initializes a logical tensor with dimensions and strides provided by user.

.. note:: 

   Once strides are explicitly provided through the API, the ``layout_type`` in :ref:`dnnl_graph_logical_tensor_t <doxid-structdnnl__graph__logical__tensor__t>` can only be :ref:`dnnl_graph_layout_type_strided <doxid-group__dnnl__graph__api__logical__tensor_1gga5b552d8a81835eb955253410bf012694aa9ea14026cc47aafffdcb92c00a1b1ea>` or :ref:`dnnl_graph_layout_type_any <doxid-group__dnnl__graph__api__logical__tensor_1gga5b552d8a81835eb955253410bf012694afc5178ef75924c4f130c70cf7b223203>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- logical_tensor

		- Output logical tensor.

	*
		- tid

		- The unique id of output logical tensor.

	*
		- dtype

		- Elements data type.

	*
		- ndims

		- Number of dimensions.

	*
		- dims

		- Array of dimensions.

	*
		- strides

		- Array of strides.

	*
		- ptype

		- Tensor property type.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_logical_tensor_get_mem_size
.. _doxid-group__dnnl__graph__api__logical__tensor_1ga56f57a976b591e6d428daea2f115207c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_logical_tensor_get_mem_size(
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		size_t* size
		)

Returns the memory size described by the logical tensor.

If it's a strided layout, the size will be calculated by ``dims`` and ``strides``. If it's an opaque layout, the size will be decided by ``layout_id``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- logical_tensor

		- Logical tensor.

	*
		- size

		- Output memory size in bytes.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_logical_tensor_is_equal
.. _doxid-group__dnnl__graph__api__logical__tensor_1gacc21c4aa2240c9a56616259e7ed71df0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_logical_tensor_is_equal(
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt1,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt2,
		uint8_t* is_equal
		)

Compares if two logical tenors are equal.

Users can decide accordingly if layout reordering is needed for two logical tensors. The method will return true for below two circumstances:

#. the two logical tensors are equal regarding each field in the struct, eg. id, ndims, dims, layout type, property, etc.

#. If all other fields are equal but the layout types in two logical tensors are different, the method will return true when the underlying memory layout is the same. For example, one logical tensor has strided layout type while the other one has opaque layout type, but underneath, both layouts are NHWC, the method will still return true for this case.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- lt1

		- The handle of first logical tensor.

	*
		- lt2

		- The handle of second logical tensor.

	*
		- is_equal

		- 1 if these two logical tensors are equal, 0 otherwise.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

Macros
------

.. index:: pair: define; DNNL_GRAPH_UNKNOWN_DIM
.. _doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_GRAPH_UNKNOWN_DIM

A wildcard value for dimensions that are unknown at a tensor or operation creation time.

.. index:: pair: define; DNNL_GRAPH_UNKNOWN_NDIMS
.. _doxid-group__dnnl__graph__api__logical__tensor_1ga49497533d28f67dc4cce08fe210bf4bf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	#define DNNL_GRAPH_UNKNOWN_NDIMS

A wildcard value for number of dimensions which is unknown at a tensor or operation creation time.

