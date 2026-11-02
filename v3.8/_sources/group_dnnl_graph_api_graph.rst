.. index:: pair: group; Graph
.. _doxid-group__dnnl__graph__api__graph:

Graph
=====

.. toctree::
	:hidden:

	class_dnnl_graph_graph.rst

Overview
~~~~~~~~

Graph represents a computational DAG with a set of operations. :ref:`More...<details-group__dnnl__graph__api__graph>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct dnnl_graph_graph* :ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>`;
	typedef const struct dnnl_graph_graph* :ref:`const_dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaac5dc221891a9aa79eb148cce05544f5>`;

	// classes

	class :ref:`dnnl::graph::graph<doxid-classdnnl_1_1graph_1_1graph>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_create<doxid-group__dnnl__graph__api__graph_1gae0ebc9c6eada0fe52f136b63850e1b4c>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>`* graph,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_create_with_fpmath_mode<doxid-group__dnnl__graph__api__graph_1gae13c8c4edc6cd30c96d81329e3973c83>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>`* graph,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_destroy<doxid-group__dnnl__graph__api__graph_1gaac9d64ff0a5a010ff8800f70f472e207>`(:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_set_fpmath_mode<doxid-group__dnnl__graph__api__graph_1ga540bb5df44c5234060efd06bb60ebc7f>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode,
		int apply_to_int
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_get_fpmath_mode<doxid-group__dnnl__graph__api__graph_1gaae1d0c2dc53bb625bb8c8fcb4d30cf09>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode,
		int* apply_to_int
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_add_op<doxid-group__dnnl__graph__api__graph_1ga69eff2efb6ccf06f27827278a974d5be>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_finalize<doxid-group__dnnl__graph__api__graph_1ga3ef94c50f6451091d549dd7f6e085ff6>`(:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_is_finalized<doxid-group__dnnl__graph__api__graph_1gaf850e0710060ffebe1418c3addf1955e>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		uint8_t* finalized
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_filter<doxid-group__dnnl__graph__api__graph_1ga28f03fcd7dcfaac5eb2a3ba08fba3ff0>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_graph_partition_policy_t<doxid-group__dnnl__graph__api__partition_1ga7e24b277b64600ef3a83dac2e8dfa83b>` policy
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_get_partition_num<doxid-group__dnnl__graph__api__graph_1ga603d39f0b799244de8b157c2967646d1>`(
		:ref:`const_dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaac5dc221891a9aa79eb148cce05544f5>` graph,
		size_t* num
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_graph_get_partitions<doxid-group__dnnl__graph__api__graph_1ga68fdf628f24078ccdd89755cdd090881>`(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		size_t num,
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>`* partitions
		);

.. _details-group__dnnl__graph__api__graph:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Graph represents a computational DAG with a set of operations.

:ref:`dnnl::graph::graph::add_op() <doxid-classdnnl_1_1graph_1_1graph_1a1cdf41276f953ecc482df858408c9ff0>` adds an operation and its input and output logical tensors into a graph. The library accumulates the operations and logical tensors and constructs and validates the graph as an internal state. A graph object is associated to a specific engine kind. The partitions returned from the graph will inherit the engine kind of the graph.

Typedefs
--------

.. index:: pair: typedef; dnnl_graph_graph_t
.. _doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct dnnl_graph_graph* dnnl_graph_graph_t

A graph handle.

.. index:: pair: typedef; const_dnnl_graph_graph_t
.. _doxid-group__dnnl__graph__api__graph_1gaac5dc221891a9aa79eb148cce05544f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct dnnl_graph_graph* const_dnnl_graph_graph_t

A constant graph handle.

Global Functions
----------------

.. index:: pair: function; dnnl_graph_graph_create
.. _doxid-group__dnnl__graph__api__graph_1gae0ebc9c6eada0fe52f136b63850e1b4c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_create(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>`* graph,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind
		)

Creates a new empty graph.

A graph is associated to a specific engine kind. The partitions returned from the graph will inherit the engine kind of the graph.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The handle of output graph.

	*
		- engine_kind

		- The target engine kind.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_create_with_fpmath_mode
.. _doxid-group__dnnl__graph__api__graph_1gae13c8c4edc6cd30c96d81329e3973c83:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_create_with_fpmath_mode(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>`* graph,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` engine_kind,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode
		)

Creates a new empty graph with an engine kind and a floating-point math mode.

All partitions returned from the graph will inherit the engine kind and floating-point math mode.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The handle of output graph.

	*
		- engine_kind

		- The kind for engine.

	*
		- mode

		- The floating-point math mode.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_destroy
.. _doxid-group__dnnl__graph__api__graph_1gaac9d64ff0a5a010ff8800f70f472e207:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_destroy(:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph)

Destroys a graph.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The graph to be destroyed.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_set_fpmath_mode
.. _doxid-group__dnnl__graph__api__graph_1ga540bb5df44c5234060efd06bb60ebc7f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_set_fpmath_mode(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>` mode,
		int apply_to_int
		)

Set the floating point math mode for a graph.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The target graph.

	*
		- mode

		- The floating-point math mode.

	*
		- apply_to_int

		- The flag that controls whether to use floating-point arithmetic for integral operations.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_get_fpmath_mode
.. _doxid-group__dnnl__graph__api__graph_1gaae1d0c2dc53bb625bb8c8fcb4d30cf09:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_get_fpmath_mode(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_fpmath_mode_t<doxid-group__dnnl__api__fpmath__mode_1ga62f956692c5a70353f164e09ff524912>`* mode,
		int* apply_to_int
		)

Get the floating point math mode for a graph.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The target graph.

	*
		- mode

		- The floating-point math mode.

	*
		- apply_to_int

		- The flag that controls whether to use floating-point arithmetic for integral operations.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_add_op
.. _doxid-group__dnnl__graph__api__graph_1ga69eff2efb6ccf06f27827278a974d5be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_add_op(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1ga7a26d33507389facd89c77a7bd042834>` op
		)

Adds an operation into a graph.

The API will return failure if the operator has already been added to the graph or the operation cannot pass the schema check in the library (eg. input and output numbers and data types, the attributes of the operation, etc.).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The target graph.

	*
		- op

		- The operation to be added.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_finalize
.. _doxid-group__dnnl__graph__api__graph_1ga3ef94c50f6451091d549dd7f6e085ff6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_finalize(:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph)

Finalizes a graph.

It means users have finished adding operations into the graph and the graph is ready for partitioning. Adding a new operation into a finalized graph will return failures. Similarly, partitioning on a un-finalized graph will also return failures.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The target graph to be finalized.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_is_finalized
.. _doxid-group__dnnl__graph__api__graph_1gaf850e0710060ffebe1418c3addf1955e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_is_finalized(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		uint8_t* finalized
		)

Checks if a graph is finalized.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The target graph to be finalized.

	*
		- finalized

		- Output the finalization status. 0 means then graph is not finalized. Other values means the graph is finalized.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_filter
.. _doxid-group__dnnl__graph__api__graph_1ga28f03fcd7dcfaac5eb2a3ba08fba3ff0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_filter(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		:ref:`dnnl_graph_partition_policy_t<doxid-group__dnnl__graph__api__partition_1ga7e24b277b64600ef3a83dac2e8dfa83b>` policy
		)

Filters a graph.

Partitions will be claimed internally according to the capability of the library, the engine kind, and the policy.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The target graph.

	*
		- policy

		- The partition policy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_get_partition_num
.. _doxid-group__dnnl__graph__api__graph_1ga603d39f0b799244de8b157c2967646d1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_get_partition_num(
		:ref:`const_dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaac5dc221891a9aa79eb148cce05544f5>` graph,
		size_t* num
		)

Returns the number of partitions of a graph.

The API should be called after a partition is already filtered. Otherwise, the output number is zero.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The graph.

	*
		- num

		- Output the number of partitions.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_graph_get_partitions
.. _doxid-group__dnnl__graph__api__graph_1ga68fdf628f24078ccdd89755cdd090881:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_graph_get_partitions(
		:ref:`dnnl_graph_graph_t<doxid-group__dnnl__graph__api__graph_1gaf5f09913d5fb57129a38a8bb779e1e71>` graph,
		size_t num,
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>`* partitions
		)

Returns the partitions from a filtered graph.

Output partition instances will be written into the parameter ``partitions``. Users need to make sure ``partitions`` is valid and has enough space to accept the partition instances. Each output partition instance should be destroyed via :ref:`dnnl_graph_partition_destroy <doxid-group__dnnl__graph__api__partition_1ga44f173aef7d5c593d305d6abd0927507>` explicitly after use.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- graph

		- The target graph.

	*
		- num

		- The number of partitions.

	*
		- partitions

		- Output the partitions.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

