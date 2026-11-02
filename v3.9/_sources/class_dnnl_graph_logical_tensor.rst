.. index:: pair: class; dnnl::graph::logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor:

class dnnl::graph::logical_tensor
=================================

.. toctree::
	:hidden:

	enum_dnnl_graph_logical_tensor_data_type.rst
	enum_dnnl_graph_logical_tensor_layout_type.rst
	enum_dnnl_graph_logical_tensor_property_type.rst

Overview
~~~~~~~~

Logical tensor object. :ref:`More...<details-classdnnl_1_1graph_1_1logical__tensor>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>
	
	class logical_tensor
	{
	public:
		// typedefs
	
		typedef :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` :ref:`dim<doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>`;
		typedef std::vector<:ref:`dim<doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>`> :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`;

		// enums
	
		enum :ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>`;
		enum :ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>`;
		enum :ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>`;

		// construction
	
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor_1a01cec4c5633637fb344d03e5547f9945>`();
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor_1a8557c46e004ed95249688ccc1e29fdc6>`(const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`& c_data);
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad9e3190a8cbabfcdbfae558284711208>`(const logical_tensor& other);
	
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor_1a710c862218dd3ed13b5ac254f670d9f6>`(
			size_t tid,
			:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
			int32_t ndims,
			:ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` ltype,
			:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` ptype = :ref:`property_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`
			);
	
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor_1a65a85083e3ef174e1353ad7a9b4ecc65>`(
			size_t tid,
			:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
			:ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` ltype = :ref:`layout_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867af31ee5e3824f1f5e5d206bdf3029f22b>`
			);
	
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor_1acf9c5886af3b07d8206a5da32de7437a>`(
			size_t tid,
			:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
			const :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`& adims,
			:ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` ltype,
			:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` ptype = :ref:`property_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`
			);
	
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor_1a2786bd0be4f2f0526aa62d11b6698d36>`(
			size_t tid,
			:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
			const :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`& adims,
			const :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`& strides,
			:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` ptype = :ref:`property_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`
			);
	
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor_1a03c23795e6a8bccb0a2cd1ad86e746d9>`(
			size_t tid,
			:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
			const :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`& adims,
			size_t lid,
			:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` ptype = :ref:`property_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`
			);

		// methods
	
		logical_tensor& :ref:`operator =<doxid-classdnnl_1_1graph_1_1logical__tensor_1afbb1d508c701fd9d09409e0dae4e67e2>` (const logical_tensor& other);
		:ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>` :ref:`get_dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a721a183a109d7e73fd13a91d453907f1>`() const;
		size_t :ref:`get_id<doxid-classdnnl_1_1graph_1_1logical__tensor_1a29439e81366ba8db75180dc8c975d1ca>`() const;
		:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` :ref:`get_data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1aaea19b3ce4512e5f2e1d0c68d9f0677f>`() const;
		:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` :ref:`get_property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a4171d99da0693368639e35d0745f3001>`() const;
		:ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` :ref:`get_layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1adf186c42eb7e4718f4985845a1101ac5>`() const;
		size_t :ref:`get_layout_id<doxid-classdnnl_1_1graph_1_1logical__tensor_1ab89484366b6e253cc0326e54622b5dfa>`() const;
		:ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>` :ref:`get_strides<doxid-classdnnl_1_1graph_1_1logical__tensor_1aaf316ade59379a433455b9ba4738397a>`() const;
		size_t :ref:`get_mem_size<doxid-classdnnl_1_1graph_1_1logical__tensor_1a12b73d1201259d4260de5603f62c7f15>`() const;
		bool :ref:`is_equal<doxid-classdnnl_1_1graph_1_1logical__tensor_1ae6dbebb8a2e295c97eb02de86d655406>`(const logical_tensor& lt) const;
	};
.. _details-classdnnl_1_1graph_1_1logical__tensor:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Logical tensor object.

Typedefs
--------

.. index:: pair: typedef; dim
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef :ref:`dnnl_dim_t<doxid-group__dnnl__api__data__types_1ga872631b12a112bf43fba985cba24dd20>` dim

Integer type for representing dimension sizes and indices.

.. index:: pair: typedef; dims
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef std::vector<:ref:`dim<doxid-classdnnl_1_1graph_1_1logical__tensor_1a759c7b96472681049e17716334a2b334>`> dims

Vector of dimensions.

Implementations are free to force a limit on the vector's length.

Construction
------------

.. index:: pair: function; logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a01cec4c5633637fb344d03e5547f9945:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor()

default constructor construct an empty object

.. index:: pair: function; logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a8557c46e004ed95249688ccc1e29fdc6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor(const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`& c_data)

Constructs a logical tensor object.

.. index:: pair: function; logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1ad9e3190a8cbabfcdbfae558284711208:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor(const logical_tensor& other)

Copy.

.. index:: pair: function; logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a710c862218dd3ed13b5ac254f670d9f6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor(
		size_t tid,
		:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
		int32_t ndims,
		:ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` ltype,
		:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` ptype = :ref:`property_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`
		)

Constructs a logical tensor object with ID, data type, ndims, layout type, and property type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tid

		- Logical tensor ID.

	*
		- dtype

		- Elements data type.

	*
		- ndims

		- Number of dimensions. -1 means unknown (see :ref:`DNNL_GRAPH_UNKNOWN_NDIMS <doxid-group__dnnl__graph__api__logical__tensor_1ga49497533d28f67dc4cce08fe210bf4bf>`) and 0 means a scalar tensor.

	*
		- ltype

		- Layout type.

	*
		- ptype

		- Property type.

.. index:: pair: function; logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a65a85083e3ef174e1353ad7a9b4ecc65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor(
		size_t tid,
		:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
		:ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` ltype = :ref:`layout_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867af31ee5e3824f1f5e5d206bdf3029f22b>`
		)

Delegated constructor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tid

		- Logical tensor ID.

	*
		- dtype

		- Elements data type.

	*
		- ltype

		- Layout type.

.. index:: pair: function; logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1acf9c5886af3b07d8206a5da32de7437a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor(
		size_t tid,
		:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
		const :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`& adims,
		:ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` ltype,
		:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` ptype = :ref:`property_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`
		)

Constructs a logical tensor object with basic information and detailed dims.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tid

		- Logical tensor ID.

	*
		- dtype

		- Elements data type.

	*
		- adims

		- Logical tensor dimensions. :ref:`DNNL_GRAPH_UNKNOWN_DIM <doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84>` means the size of that dimension is unknown. 0 is used to define zero-dimension tensor.

	*
		- ltype

		- Layout type. If it's strided, the strides field in the output logical tensor will be deduced accordingly.

	*
		- ptype

		- Property type.

.. index:: pair: function; logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a2786bd0be4f2f0526aa62d11b6698d36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor(
		size_t tid,
		:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
		const :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`& adims,
		const :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`& strides,
		:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` ptype = :ref:`property_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`
		)

Constructs a logical tensor object with detailed dims and strides.

The layout_type of the output logical tensor object will always be strided.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tid

		- Logical tensor ID.

	*
		- dtype

		- Elements data type.

	*
		- adims

		- Logical tensor dimensions. :ref:`DNNL_GRAPH_UNKNOWN_DIM <doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84>` means the size of that dimension is unknown. 0 is used to define zero-dimension tensor.

	*
		- strides

		- Logical tensor strides. :ref:`DNNL_GRAPH_UNKNOWN_DIM <doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84>` means the stride of the dimension is unknown. The library currently doesn't support other negative stride values.

	*
		- ptype

		- Property type.

.. index:: pair: function; logical_tensor
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a03c23795e6a8bccb0a2cd1ad86e746d9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor(
		size_t tid,
		:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` dtype,
		const :ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>`& adims,
		size_t lid,
		:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` ptype = :ref:`property_type::undef<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82af31ee5e3824f1f5e5d206bdf3029f22b>`
		)

Constructs a logical tensor object with detailed dims and an opaque layout ID.

layout_type of the output logical tensor object will always be opaque.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tid

		- Logical tensor ID.

	*
		- dtype

		- Elements data type.

	*
		- adims

		- Logical tensor dimensions. :ref:`DNNL_GRAPH_UNKNOWN_DIM <doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84>` means the size of that dimension is unknown. 0 is used to define zero-dimension tensor.

	*
		- lid

		- Opaque layout id.

	*
		- ptype

		- Property type

Methods
-------

.. index:: pair: function; operator=
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1afbb1d508c701fd9d09409e0dae4e67e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	logical_tensor& operator = (const logical_tensor& other)

Assign.

.. index:: pair: function; get_dims
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a721a183a109d7e73fd13a91d453907f1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>` get_dims() const

Returns dimensions of a logical tensor.



.. rubric:: Returns:

A vector describing the size of each dimension.

.. index:: pair: function; get_id
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a29439e81366ba8db75180dc8c975d1ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t get_id() const

Returns the unique id of a logical tensor.



.. rubric:: Returns:

An integer value describing the ID.

.. index:: pair: function; get_data_type
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1aaea19b3ce4512e5f2e1d0c68d9f0677f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`data_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1acddb1dc65b7b4feede7710a719f32227>` get_data_type() const

Returns the data type of a logical tensor.



.. rubric:: Returns:

The data type.

.. index:: pair: function; get_property_type
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a4171d99da0693368639e35d0745f3001:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`property_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1a037ba7c242d8127d2f42c0c2aef29d82>` get_property_type() const

Returns the property type of a logical tensor.



.. rubric:: Returns:

The property type.

.. index:: pair: function; get_layout_type
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1adf186c42eb7e4718f4985845a1101ac5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`layout_type<doxid-classdnnl_1_1graph_1_1logical__tensor_1ad3fcaff44671577e56adb03b770f4867>` get_layout_type() const

Returns the layout type of a logical tensor.



.. rubric:: Returns:

The layout type.

.. index:: pair: function; get_layout_id
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1ab89484366b6e253cc0326e54622b5dfa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t get_layout_id() const

Returns the layout ID of a logical tensor.

The API should be called on a logical tensor with opaque layout type. Otherwise, an exception will be raised.



.. rubric:: Returns:

Layout ID.

.. index:: pair: function; get_strides
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1aaf316ade59379a433455b9ba4738397a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dims<doxid-classdnnl_1_1graph_1_1logical__tensor_1a31af724d1ea783a09b6900d69b43ddc7>` get_strides() const

Returns the strides of a logical tensor.

The API should be called on a logical tensor with strided layout type. Otherwise, an exception will be raised.



.. rubric:: Returns:

A vector describing the stride size of each dimension.

.. index:: pair: function; get_mem_size
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1a12b73d1201259d4260de5603f62c7f15:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t get_mem_size() const

Returns memory size in bytes required by this logical tensor.



.. rubric:: Returns:

The memory size in bytes.

.. index:: pair: function; is_equal
.. _doxid-classdnnl_1_1graph_1_1logical__tensor_1ae6dbebb8a2e295c97eb02de86d655406:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool is_equal(const logical_tensor& lt) const

Compares if two logical tenors are equal.

Users can decide accordingly if layout reordering is needed for two logical tensors. The method will return true for below two circumstances:

#. the two logical tensors are equal regarding each field in the struct, eg. id, ndims, dims, layout type, property, etc.

#. If all other fields are equal but the layout types in two logical tensors are different, the method will return true when the underlying memory layout is the same. For example, one logical tensor has strided layout type while the other one has opaque layout type, but underneath, both layouts are NHWC, the method will still return true for this case.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- lt

		- The input logical tensor to be compared.



.. rubric:: Returns:

``true`` if the two logical tensors are equal. ``false`` otherwise

