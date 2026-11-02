.. index:: pair: group; Op
.. _doxid-group__dnnl__graph__api__op:

Op
==

.. toctree::
	:hidden:

	enum_dnnl_graph_op_attr_t.rst
	enum_dnnl_graph_op_kind_t.rst
	class_dnnl_graph_op.rst

Overview
~~~~~~~~

OP is an abstraction of computation logic for deep neural network operations. :ref:`More...<details-group__dnnl__graph__api__op>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct dnnl_graph_op* :ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>`;
	typedef const struct dnnl_graph_op* :ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>`;

	// enums

	enum :ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>`;
	enum :ref:`dnnl_graph_op_kind_t<doxid-group__dnnl__graph__api__op_1gad3d8d1611b566cade947d9d30225d5b2>`;

	// classes

	class :ref:`dnnl::graph::op<doxid-classdnnl_1_1graph_1_1op>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_create<doxid-group__dnnl__graph__api__op_1ga89f9449ddd533e166e3deaf253520ba1>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>`* op,
		size_t id,
		:ref:`dnnl_graph_op_kind_t<doxid-group__dnnl__graph__api__op_1gad3d8d1611b566cade947d9d30225d5b2>` kind,
		const char* verbose_name
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_destroy<doxid-group__dnnl__graph__api__op_1ga9078b97ce5f2e44cb318d08ff96fe391>`(:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_add_input<doxid-group__dnnl__graph__api__op_1gac1cc01522c2328069e8bd045f563554f>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* input
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_add_output<doxid-group__dnnl__graph__api__op_1gad2ada5d285eb5cc8aa38785585525b3d>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* output
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_set_attr_f32<doxid-group__dnnl__graph__api__op_1gaa4605432c3cd40570607a40a1448e777>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const float* value,
		size_t value_len
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_set_attr_bool<doxid-group__dnnl__graph__api__op_1ga122b16165d16f9e1b36fa04c4df783de>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const uint8_t* value,
		size_t value_len
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_set_attr_s64<doxid-group__dnnl__graph__api__op_1gaca7be5242f3fd61421bcc49365129965>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const int64_t* value,
		size_t value_len
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_set_attr_str<doxid-group__dnnl__graph__api__op_1gae832731052f5072256527a73326a7d43>`(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const char* value,
		size_t value_len
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_get_id<doxid-group__dnnl__graph__api__op_1ga9258f54424d3e9f3e88356982864d1e0>`(
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		size_t* id
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_op_get_kind<doxid-group__dnnl__graph__api__op_1ga11559f93efe532d71c0c6284896d8444>`(
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		:ref:`dnnl_graph_op_kind_t<doxid-group__dnnl__graph__api__op_1gad3d8d1611b566cade947d9d30225d5b2>`* kind
		);

.. _details-group__dnnl__graph__api__op:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

OP is an abstraction of computation logic for deep neural network operations.

An op object encapsulates an operation kind which describes the computation logic, an unique ID which differentiates operations with the same kind, and logical tensors which describes the input and output of the operation and its connections to other operations in the graph.

Typedefs
--------

.. index:: pair: typedef; dnnl_graph_op_t
.. _doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct dnnl_graph_op* dnnl_graph_op_t

An operation handle.

.. index:: pair: typedef; const_dnnl_graph_op_t
.. _doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct dnnl_graph_op* const_dnnl_graph_op_t

A constant operation handle.

Global Functions
----------------

.. index:: pair: function; dnnl_graph_op_create
.. _doxid-group__dnnl__graph__api__op_1ga89f9449ddd533e166e3deaf253520ba1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_create(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>`* op,
		size_t id,
		:ref:`dnnl_graph_op_kind_t<doxid-group__dnnl__graph__api__op_1gad3d8d1611b566cade947d9d30225d5b2>` kind,
		const char* verbose_name
		)

Initializes an op with unique id, kind, and name.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Output op

	*
		- id

		- The unique id of the output op.

	*
		- kind

		- The op kind.

	*
		- verbose_name

		- The string added as the op name.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_destroy
.. _doxid-group__dnnl__graph__api__op_1ga9078b97ce5f2e44cb318d08ff96fe391:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_destroy(:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op)

Destroys an op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- The op to be destroyed.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_add_input
.. _doxid-group__dnnl__graph__api__op_1gac1cc01522c2328069e8bd045f563554f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_add_input(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* input
		)

Adds input logical tensor to the op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Input op.

	*
		- input

		- The input logical tensor to be added.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_add_output
.. _doxid-group__dnnl__graph__api__op_1gad2ada5d285eb5cc8aa38785585525b3d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_add_output(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* output
		)

Adds output logical tensor to the op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Input op.

	*
		- output

		- The output logical tensor to be added.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_set_attr_f32
.. _doxid-group__dnnl__graph__api__op_1gaa4605432c3cd40570607a40a1448e777:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_set_attr_f32(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const float* value,
		size_t value_len
		)

Sets floating point attribute to an op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Input op.

	*
		- name

		- The attribute's name.

	*
		- value

		- The attribute's value.

	*
		- value_len

		- The number of value element.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_set_attr_bool
.. _doxid-group__dnnl__graph__api__op_1ga122b16165d16f9e1b36fa04c4df783de:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_set_attr_bool(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const uint8_t* value,
		size_t value_len
		)

Sets boolean attribute to an op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Input op.

	*
		- name

		- The attribute's name.

	*
		- value

		- The attribute's value.

	*
		- value_len

		- The number of value element.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_set_attr_s64
.. _doxid-group__dnnl__graph__api__op_1gaca7be5242f3fd61421bcc49365129965:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_set_attr_s64(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const int64_t* value,
		size_t value_len
		)

Sets integer attribute to an op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Input op.

	*
		- name

		- The attribute's name.

	*
		- value

		- The attribute's value.

	*
		- value_len

		- The number of value element.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_set_attr_str
.. _doxid-group__dnnl__graph__api__op_1gae832731052f5072256527a73326a7d43:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_set_attr_str(
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op,
		:ref:`dnnl_graph_op_attr_t<doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>` name,
		const char* value,
		size_t value_len
		)

Sets string attribute to an op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Input op.

	*
		- name

		- The attribute's name.

	*
		- value

		- The attribute's value.

	*
		- value_len

		- The length of the string value.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_get_id
.. _doxid-group__dnnl__graph__api__op_1ga9258f54424d3e9f3e88356982864d1e0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_get_id(
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		size_t* id
		)

Returns the unique id of an op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Input op.

	*
		- id

		- Output the unique id.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_op_get_kind
.. _doxid-group__dnnl__graph__api__op_1ga11559f93efe532d71c0c6284896d8444:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_op_get_kind(
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		:ref:`dnnl_graph_op_kind_t<doxid-group__dnnl__graph__api__op_1gad3d8d1611b566cade947d9d30225d5b2>`* kind
		)

Returns the kind of an op.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- Input op.

	*
		- kind

		- Output op kind.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

