.. index:: pair: group; Partition
.. _doxid-group__dnnl__graph__api__partition:

Partition
=========

.. toctree::
	:hidden:

	enum_dnnl_graph_partition_policy_t.rst
	class_dnnl_graph_partition.rst

Overview
~~~~~~~~

Partition represents a collection of operations and their input and output logical tensors identified by library as the basic unit for compilation and execution. :ref:`More...<details-group__dnnl__graph__api__partition>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct dnnl_graph_partition* :ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>`;
	typedef const struct dnnl_graph_partition* :ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>`;

	// enums

	enum :ref:`dnnl_graph_partition_policy_t<doxid-group__dnnl__graph__api__partition_1ga7e24b277b64600ef3a83dac2e8dfa83b>`;

	// classes

	class :ref:`dnnl::graph::partition<doxid-classdnnl_1_1graph_1_1partition>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_create_with_op<doxid-group__dnnl__graph__api__partition_1gade597975f67997d0242315b847e288aa>`(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>`* partition,
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` ekind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_destroy<doxid-group__dnnl__graph__api__partition_1ga44f173aef7d5c593d305d6abd0927507>`(:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_op_num<doxid-group__dnnl__graph__api__partition_1gaf54ad50ee43f413a0e9bcd2ed3866d30>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_ops<doxid-group__dnnl__graph__api__partition_1ga194ebb49cbf9bcb26f6bd94c202fd76c>`(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition,
		size_t num,
		size_t* ids
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_id<doxid-group__dnnl__graph__api__partition_1ga4f193dd55464dfb5d74a44e0d06ecba3>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* id
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_compile<doxid-group__dnnl__graph__api__partition_1ga0de016808a18bea4d23694dacd438035>`(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition,
		:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>` compiled_partition,
		size_t in_num,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`** inputs,
		size_t out_num,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`** outputs,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_input_ports_num<doxid-group__dnnl__graph__api__partition_1gae68562298df62a9d3fc12042ac2b9ab2>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_input_ports<doxid-group__dnnl__graph__api__partition_1ga32868fe8a784b661c0d2865fa530bbe5>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t num,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* inputs
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_output_ports_num<doxid-group__dnnl__graph__api__partition_1ga574edb86ed4abb2fc129cba4bb66e7c9>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_output_ports<doxid-group__dnnl__graph__api__partition_1gaea3a1581038bc059c14bef77f5034ebd>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t num,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* outputs
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_is_supported<doxid-group__dnnl__graph__api__partition_1ga9899bfedf8d4e3530f76824c318cb0d5>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		uint8_t* is_supported
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_partition_get_engine_kind<doxid-group__dnnl__graph__api__partition_1ga54a4685d7b3deee2728ea4a6268bc822>`(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`* kind
		);

.. _details-group__dnnl__graph__api__partition:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Partition represents a collection of operations and their input and output logical tensors identified by library as the basic unit for compilation and execution.

Typedefs
--------

.. index:: pair: typedef; dnnl_graph_partition_t
.. _doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct dnnl_graph_partition* dnnl_graph_partition_t

A partition handle.

.. index:: pair: typedef; const_dnnl_graph_partition_t
.. _doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct dnnl_graph_partition* const_dnnl_graph_partition_t

A constant partition handle.

Global Functions
----------------

.. index:: pair: function; dnnl_graph_partition_create_with_op
.. _doxid-group__dnnl__graph__api__partition_1gade597975f67997d0242315b847e288aa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_create_with_op(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>`* partition,
		:ref:`const_dnnl_graph_op_t<doxid-group__dnnl__graph__api__op_1gad7b0799ea1aec4c3544f0a155f8d192b>` op,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` ekind
		)

Creates a new partition with a given operator and engine kind.

The API is used to create a partition from an operation directly without creating the graph and calling ``get_partitions()``. The output partition contains only one operation specified by the parameter. The output partition instance should be destroyed via :ref:`dnnl_graph_partition_destroy <doxid-group__dnnl__graph__api__partition_1ga44f173aef7d5c593d305d6abd0927507>` after use.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The handle of output partition.

	*
		- op

		- The operation used to create partition.

	*
		- ekind

		- The engine kind used to create partition.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_destroy
.. _doxid-group__dnnl__graph__api__partition_1ga44f173aef7d5c593d305d6abd0927507:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_destroy(:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition)

Destroys a partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The partition to be destroyed.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_get_op_num
.. _doxid-group__dnnl__graph__api__partition_1gaf54ad50ee43f413a0e9bcd2ed3866d30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_get_op_num(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		)

Returns the number of operations in a partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- num

		- Output the number of operations.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_get_ops
.. _doxid-group__dnnl__graph__api__partition_1ga194ebb49cbf9bcb26f6bd94c202fd76c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_get_ops(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition,
		size_t num,
		size_t* ids
		)

Returns the list of op IDs of the partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- num

		- The number of ops.

	*
		- ids

		- Output the op IDs.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_get_id
.. _doxid-group__dnnl__graph__api__partition_1ga4f193dd55464dfb5d74a44e0d06ecba3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_get_id(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* id
		)

Returns the ID of a partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- id

		- Output the ID of the partition.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_compile
.. _doxid-group__dnnl__graph__api__partition_1ga0de016808a18bea4d23694dacd438035:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_compile(
		:ref:`dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga2fdc3001503c7b586d5fc16637872ce4>` partition,
		:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>` compiled_partition,
		size_t in_num,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`** inputs,
		size_t out_num,
		const :ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`** outputs,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine
		)

Compiles a partition with given input and output logical tensors.

The output logical tensors can contain unknown dimensions. For this case, the compilation will deduce the output shapes according to input shapes. The output logical tensors can also have layout type ``any``. The compilation will choose the optimal layout for output tensors. The optimal layout will be represented as an opaque layout ID saved in the output logical tensor.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- compiled_partition

		- Output compiled partition.

	*
		- in_num

		- The number of input logical tensors.

	*
		- inputs

		- A list of input logical tensors.

	*
		- out_num

		- The number of output logical tensors.

	*
		- outputs

		- A list of output logical tensors.

	*
		- engine

		- The target engine of the compilation.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_get_input_ports_num
.. _doxid-group__dnnl__graph__api__partition_1gae68562298df62a9d3fc12042ac2b9ab2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_get_input_ports_num(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		)

Returns the number of input logical tensors of a partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- num

		- Output the number of input logical tensors.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_get_input_ports
.. _doxid-group__dnnl__graph__api__partition_1ga32868fe8a784b661c0d2865fa530bbe5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_get_input_ports(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t num,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* inputs
		)

Returns a list of input logical tensors from a partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- num

		- The number of input logical tensors.

	*
		- inputs

		- The list of input logical tensors.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_get_output_ports_num
.. _doxid-group__dnnl__graph__api__partition_1ga574edb86ed4abb2fc129cba4bb66e7c9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_get_output_ports_num(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t* num
		)

Returns the number of output logical tensors of a partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- num

		- Output the number of output logical tensors.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_get_output_ports
.. _doxid-group__dnnl__graph__api__partition_1gaea3a1581038bc059c14bef77f5034ebd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_get_output_ports(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		size_t num,
		:ref:`dnnl_graph_logical_tensor_t<doxid-structdnnl__graph__logical__tensor__t>`* outputs
		)

Returns a list of output logical tensors from a partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- num

		- The number of output logical tensors.

	*
		- outputs

		- The list of output logical tensors.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_is_supported
.. _doxid-group__dnnl__graph__api__partition_1ga9899bfedf8d4e3530f76824c318cb0d5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_is_supported(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		uint8_t* is_supported
		)

Returns the supporting status of a partition.

Some operations may not be supported by the library under certain circumstances. During partitioning stage, unsupported partitions will be returned to users with each containing an unsupported operation. Users should check the supporting status of a partition before transforming the computation graph or compiling the partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- is_supported

		- Output flag to indicate the supporting status. 0 means unsupported while 1 means supported.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_partition_get_engine_kind
.. _doxid-group__dnnl__graph__api__partition_1ga54a4685d7b3deee2728ea4a6268bc822:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_partition_get_engine_kind(
		:ref:`const_dnnl_graph_partition_t<doxid-group__dnnl__graph__api__partition_1ga9c9c5e4412a1c29f3fbf28f1567bd825>` partition,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`* kind
		)

Returns the engine kind of a partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- partition

		- The target partition.

	*
		- kind

		- The output engine kind.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

