.. index:: pair: struct; dnnl_graph_logical_tensor_t
.. _doxid-structdnnl__graph__logical__tensor__t:

struct dnnl_graph_logical_tensor_t
==================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Logical tensor. :ref:`More...<details-structdnnl__graph__logical__tensor__t>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph_types.h>
	
	struct dnnl_graph_logical_tensor_t
	{
		// fields
	
		size_t :ref:`id<doxid-structdnnl__graph__logical__tensor__t_1a35ea59fc96c5ac3bdab66f9cc8a43b14>`;
		int :ref:`ndims<doxid-structdnnl__graph__logical__tensor__t_1a4afb34ddde9afec29db7e23835ecc7fd>`;
		:ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` :ref:`dims<doxid-structdnnl__graph__logical__tensor__t_1a9d92b96f039e80c2f72e1368613045fe>`;
		:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` :ref:`data_type<doxid-structdnnl__graph__logical__tensor__t_1a5828df15766e990f199060a2b28ca312>`;
		:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` :ref:`property<doxid-structdnnl__graph__logical__tensor__t_1acc7672209797a339ccc09b8d6d10f6e6>`;
		:ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>` :ref:`layout_type<doxid-structdnnl__graph__logical__tensor__t_1a5e8dd6031f49c19cc2920e26c08f3be9>`;
		:ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` :ref:`strides<doxid-structdnnl__graph__logical__tensor__t_1a59d798e69e02e2f0012a75f6b6905471>`;
		size_t :ref:`layout_id<doxid-structdnnl__graph__logical__tensor__t_1af0fc868250dbefcd97d8e356c278bcdd>`;
		union dnnl_graph_logical_tensor_t::@3 :target:`layout<doxid-structdnnl__graph__logical__tensor__t_1a7d71056e832ea771b80e270e5e5323a4>`;
	};
.. _details-structdnnl__graph__logical__tensor__t:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Logical tensor.

It is based on an ID, a number of dimensions, dimensions themselves, element data type, tensor property and tensor memory layout.

Fields
------

.. index:: pair: variable; id
.. _doxid-structdnnl__graph__logical__tensor__t_1a35ea59fc96c5ac3bdab66f9cc8a43b14:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t id

Unique id of each logical tensor.

The library uses logical tensor IDs to build up the connections between operations if the output of one operation has the same ID as the input of another operation.

.. index:: pair: variable; ndims
.. _doxid-structdnnl__graph__logical__tensor__t_1a4afb34ddde9afec29db7e23835ecc7fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int ndims

Number of dimensions.

-1 means unknown (DNNL_GRAPH_UNKNOWN_NDIMS). 0 is used to define scalar tensor.

.. index:: pair: variable; dims
.. _doxid-structdnnl__graph__logical__tensor__t_1a9d92b96f039e80c2f72e1368613045fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` dims

Size of each dimension.

:ref:`DNNL_GRAPH_UNKNOWN_DIM <doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84>` means the size of that dimension is unknown. 0 is used to define zero-dimension tensor. The library supports to deduce output shapes according to input shapes during compilation. Unlike memory descriptor in oneDNN primitive API, the order of dimensions is not defined in logical tensor. It is defined by the operations which respect the order through the attributes :ref:`dnnl_graph_op_attr_data_format <doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a3f5e3951f43b1bb7d58545a8b707b5e2>` or :ref:`dnnl_graph_op_attr_weights_format <doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a63800d1e05815c9b724dcc603883f9d9>`. For example, for a Convolution with ``data_format=NXC``, it means the first element of dims of activation tensor is mini-batch size, the last effective element of dims is channel size, and other elements between them are spatial dimensions.

.. index:: pair: variable; data_type
.. _doxid-structdnnl__graph__logical__tensor__t_1a5828df15766e990f199060a2b28ca312:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_data_type_t<doxid-group__dnnl__api__data__types_1ga012ba1c84ff24bdd068f9d2f9b26a130>` data_type

Data type of the tensor elements.

.. index:: pair: variable; property
.. _doxid-structdnnl__graph__logical__tensor__t_1acc7672209797a339ccc09b8d6d10f6e6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_graph_tensor_property_t<doxid-group__dnnl__graph__api__logical__tensor_1gadf98ec2238dd9001c6fe7870ebf1b19f>` property

Property type of the tensor.

.. index:: pair: variable; layout_type
.. _doxid-structdnnl__graph__logical__tensor__t_1a5e8dd6031f49c19cc2920e26c08f3be9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_graph_layout_type_t<doxid-group__dnnl__graph__api__logical__tensor_1ga5b552d8a81835eb955253410bf012694>` layout_type

Layout type of the tensor.

.. index:: pair: variable; strides
.. _doxid-structdnnl__graph__logical__tensor__t_1a59d798e69e02e2f0012a75f6b6905471:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_dims_t<doxid-group__dnnl__api__data__types_1ga8331e1160e52a5d4babe96736464095a>` strides

The field is valid when ``layout_type`` is :ref:`dnnl_graph_layout_type_strided <doxid-group__dnnl__graph__api__logical__tensor_1gga5b552d8a81835eb955253410bf012694aa9ea14026cc47aafffdcb92c00a1b1ea>`.

:ref:`DNNL_GRAPH_UNKNOWN_DIM <doxid-group__dnnl__graph__api__logical__tensor_1ga45a2f66e2234c3ff0c5d4a06582cca84>` means the stride of the dimension is unknown. The library currently doesn't support other negative stride values.

.. index:: pair: variable; layout_id
.. _doxid-structdnnl__graph__logical__tensor__t_1af0fc868250dbefcd97d8e356c278bcdd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t layout_id

The field is valid when ``layout_type`` is :ref:`dnnl_graph_layout_type_opaque <doxid-group__dnnl__graph__api__logical__tensor_1gga5b552d8a81835eb955253410bf012694a214016e723853d6d9d753871cd5f25b7>`.

An opaque layout ID is usually generated by a partition which is compiled with layout type any.

