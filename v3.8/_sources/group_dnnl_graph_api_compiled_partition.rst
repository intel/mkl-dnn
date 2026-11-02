.. index:: pair: group; Compiled Partition
.. _doxid-group__dnnl__graph__api__compiled__partition:

Compiled Partition
==================

.. toctree::
	:hidden:

	struct_dnnl_graph_inplace_pair_t.rst
	class_dnnl_graph_compiled_partition.rst

Overview
~~~~~~~~

A compiled partition represents the generated kernels specialized for a partition on a target hardware (engine) with input and output information specified by the logical tensors. :ref:`More...<details-group__dnnl__graph__api__compiled__partition>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct dnnl_graph_compiled_partition* :ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>`;
	typedef const struct dnnl_graph_compiled_partition* :ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>`;

	// structs

	struct :ref:`dnnl_graph_inplace_pair_t<doxid-structdnnl__graph__inplace__pair__t>`;

	// classes

	class :ref:`dnnl::graph::compiled_partition<doxid-classdnnl_1_1graph_1_1compiled__partition>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_create<doxid-group__dnnl__graph__api__compiled__partition_1ga86b0ee196722ef06f4525416a7a41e92>`(
		:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>`* compiled_partition,
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_execute<doxid-group__dnnl__graph__api__compiled__partition_1ga34203136d999a80d09dcc24d0a3d2268>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		size_t num_inputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* inputs,
		size_t num_outputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* outputs
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_destroy<doxid-group__dnnl__graph__api__compiled__partition_1gae89b0fccf8e91d7796f304a9f14b8dec>`(:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>` compiled_partition);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_query_logical_tensor<doxid-group__dnnl__graph__api__compiled__partition_1ga03a2fbb5505cc60962e05f1cd0e60f6a>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		size_t tid,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_compiled_partition_get_inplace_ports<doxid-group__dnnl__graph__api__compiled__partition_1ga5fc8b08404ce7e7063eac908e59c0158>`(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		size_t* num_inplace_pairs,
		const :ref:`dnnl_graph_inplace_pair_t<doxid-structdnnl__graph__inplace__pair__t>`** inplace_pairs
		);

.. _details-group__dnnl__graph__api__compiled__partition:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A compiled partition represents the generated kernels specialized for a partition on a target hardware (engine) with input and output information specified by the logical tensors.

Typedefs
--------

.. index:: pair: typedef; dnnl_graph_compiled_partition_t
.. _doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct dnnl_graph_compiled_partition* dnnl_graph_compiled_partition_t

A compiled partition handle.

.. index:: pair: typedef; const_dnnl_graph_compiled_partition_t
.. _doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct dnnl_graph_compiled_partition* const_dnnl_graph_compiled_partition_t

A constant compiled partition handle.

Global Functions
----------------

.. index:: pair: function; dnnl_graph_compiled_partition_create
.. _doxid-group__dnnl__graph__api__compiled__partition_1ga86b0ee196722ef06f4525416a7a41e92:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_compiled_partition_create(
		:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>`* compiled_partition,
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition
		)

Creates a new compiled partition handle.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- compiled_partition

		- The handle of output compiled partition.

	*
		- partition

		- The handle of input partition.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_compiled_partition_execute
.. _doxid-group__dnnl__graph__api__compiled__partition_1ga34203136d999a80d09dcc24d0a3d2268:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_compiled_partition_execute(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		size_t num_inputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* inputs,
		size_t num_outputs,
		:ref:`const_dnnl_graph_tensor_t<doxid-group__dnnl__graph__api__tensor_1ga501fef96950f38448cb326c776e8d068>`* outputs
		)

Executes a compiled partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- compiled_partition

		- The handle of target compiled partition.

	*
		- stream

		- The stream used for execution.

	*
		- num_inputs

		- The number of input tensors.

	*
		- inputs

		- A list of input tensors.

	*
		- num_outputs

		- The number of output tensors.

	*
		- outputs

		- A non-empty list of output tensors.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_compiled_partition_destroy
.. _doxid-group__dnnl__graph__api__compiled__partition_1gae89b0fccf8e91d7796f304a9f14b8dec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_compiled_partition_destroy(:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>` compiled_partition)

Destroys a compiled partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- compiled_partition

		- The compiled partition to be destroyed.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_compiled_partition_query_logical_tensor
.. _doxid-group__dnnl__graph__api__compiled__partition_1ga03a2fbb5505cc60962e05f1cd0e60f6a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_compiled_partition_query_logical_tensor(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		size_t tid,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* lt
		)

Queries an input or output logical tensor according to tensor ID.

If the tensor ID doesn't belong to any input or output of the compiled partition, an error status :ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` will be returned by the API.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- compiled_partition

		- The handle of target compiled_partition.

	*
		- tid

		- The unique id of required tensor.

	*
		- lt

		- The output logical tensor.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_compiled_partition_get_inplace_ports
.. _doxid-group__dnnl__graph__api__compiled__partition_1ga5fc8b08404ce7e7063eac908e59c0158:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_compiled_partition_get_inplace_ports(
		:ref:`const_dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1gac1af164b5c86e9a3ff3c13583da98f06>` compiled_partition,
		size_t* num_inplace_pairs,
		const :ref:`dnnl_graph_inplace_pair_t<doxid-structdnnl__graph__inplace__pair__t>`** inplace_pairs
		)

Returns the hint of in-place pairs from a compiled partition.

It indicates that an input and an output of the partition can share the same memory buffer for computation. In-place computation helps to reduce the memory footprint and improves cache locality. But since the library may not have a global view of user's application, it's possible that the tensor with ``input_id`` is used at other places in user's computation graph. In this case, the user should take the in-place pair as a hint and pass a different memory buffer for output tensor to avoid overwriting the input memory buffer which will probably cause unexpected incorrect results.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- compiled_partition

		- The handle of target compiled_partition.

	*
		- num_inplace_pairs

		- The number of in-place pairs.

	*
		- inplace_pairs

		- The handle of in-place pairs.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

