.. index:: pair: group; Tensor
.. _doxid-group__dnnl__graph__api__tensor:

Tensor
======

.. toctree::
	:hidden:

	class_dnnl_graph_tensor.rst

Overview
~~~~~~~~

Tensor is an abstraction for multi-dimensional input and output data needed in the execution of a compiled partition. :ref:`More...<details-group__dnnl__graph__api__tensor>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct dnnl_graph_tensor* :ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>`;
	typedef const struct dnnl_graph_tensor* :ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`;

	// classes

	class :ref:`dnnl::graph::tensor<doxid-classdnnl_1_1graph_1_1tensor>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_create<doxid-group__dnnl__graph__api__tensor_1gaf1ce44e7c73d38f7dfd2e4f374d341e5>`(
		:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>`* tensor,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_destroy<doxid-group__dnnl__graph__api__tensor_1ga42e42c1059fbb4f86919754d31c5888d>`(:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>` tensor);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_get_data_handle<doxid-group__dnnl__graph__api__tensor_1ga39f0b7ce6ba2067dc0a166075abebb16>`(
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>` tensor,
		void** handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_set_data_handle<doxid-group__dnnl__graph__api__tensor_1ga2b81562df6173e0f2ff1b4360c4cf3ec>`(
		:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>` tensor,
		void* handle
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_tensor_get_engine<doxid-group__dnnl__graph__api__tensor_1ga09d0a460550d1b399a0614c20663f73b>`(
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>` tensor,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		);

.. _details-group__dnnl__graph__api__tensor:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Tensor is an abstraction for multi-dimensional input and output data needed in the execution of a compiled partition.

A tensor object encapsulates a handle to a memory buffer allocated on a specific engine and a logical tensor which describes the dimensions, elements data type, and memory layout.

Typedefs
--------

.. index:: pair: typedef; dnnl_graph_tensor_t
.. _doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct dnnl_graph_tensor* dnnl_graph_tensor_t

A tensor handle.

.. index:: pair: typedef; const_dnnl_graph_tensor_t
.. _doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct dnnl_graph_tensor* const_dnnl_graph_tensor_t

A constant tensor handle.

Global Functions
----------------

.. index:: pair: function; dnnl_graph_tensor_create
.. _doxid-group__dnnl__graph__api__tensor_1gaf1ce44e7c73d38f7dfd2e4f374d341e5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_tensor_create(
		:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>`* tensor,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* logical_tensor,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		void* handle
		)

Creates a tensor with logical tensor, engine, and data handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tensor

		- Output tensor.

	*
		- logical_tensor

		- Description for this tensor.

	*
		- engine

		- Engine to use.

	*
		- handle

		- Handle of the memory buffer to use as an underlying storage.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_tensor_destroy
.. _doxid-group__dnnl__graph__api__tensor_1ga42e42c1059fbb4f86919754d31c5888d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_tensor_destroy(:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>` tensor)

Destroys a tensor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tensor

		- The tensor to be destroyed.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_tensor_get_data_handle
.. _doxid-group__dnnl__graph__api__tensor_1ga39f0b7ce6ba2067dc0a166075abebb16:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_tensor_get_data_handle(
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>` tensor,
		void** handle
		)

Gets the data handle of a tensor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tensor

		- The input tensor.

	*
		- handle

		- Pointer to the data of input tensor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_tensor_set_data_handle
.. _doxid-group__dnnl__graph__api__tensor_1ga2b81562df6173e0f2ff1b4360c4cf3ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_tensor_set_data_handle(
		:ref:`dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga77c7c6168286b2a791ecea37336d25d4>` tensor,
		void* handle
		)

Set data handle for a tensor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tensor

		- The input tensor.

	*
		- handle

		- New data handle for tensor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_tensor_get_engine
.. _doxid-group__dnnl__graph__api__tensor_1ga09d0a460550d1b399a0614c20663f73b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_tensor_get_engine(
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>` tensor,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		)

Returns the engine of a tensor object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tensor

		- The input tensor.

	*
		- engine

		- Output engine on which the tensor is located.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

