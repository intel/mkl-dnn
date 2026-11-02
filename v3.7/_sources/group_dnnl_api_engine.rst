.. index:: pair: group; Engine
.. _doxid-group__dnnl__api__engine:

Engine
======

.. toctree::
	:hidden:

	enum_dnnl_engine_kind_t.rst
	struct_dnnl_engine.rst
	struct_dnnl_engine-2.rst

Overview
~~~~~~~~

An abstraction of a computational device: a CPU, a specific GPU card in the system, etc. :ref:`More...<details-group__dnnl__api__engine>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct :ref:`dnnl_engine<doxid-structdnnl__engine>`* :ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`;

	// enums

	enum :ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`;

	// structs

	struct :ref:`dnnl_engine<doxid-structdnnl__engine>`;
	struct :ref:`dnnl::engine<doxid-structdnnl_1_1engine>`;

	// global functions

	:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` :ref:`dnnl::convert_to_c<doxid-group__dnnl__api__engine_1gae472e59f404ba6527988b046ef24c743>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind);
	size_t DNNL_API :ref:`dnnl_engine_get_count<doxid-group__dnnl__api__engine_1gadff5935622df99a2f89acb5cbea09ab5>`(:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_engine_create<doxid-group__dnnl__api__engine_1gab84f82f3011349cbfe368b61882834fd>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind,
		size_t index
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_engine_get_kind<doxid-group__dnnl__api__engine_1ga8a38bdce17f51616d03310a8e8764c8c>`(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`* kind
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_engine_destroy<doxid-group__dnnl__api__engine_1ga8d6976b3792cf1ef64d01545929b4d8f>`(:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine);

.. _details-group__dnnl__api__engine:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

An abstraction of a computational device: a CPU, a specific GPU card in the system, etc.

Most primitives are created to execute computations on one specific engine. The only exceptions are reorder primitives that transfer data between two different engines.



.. rubric:: See also:

:ref:`Basic Concepts <doxid-dev_guide_basic_concepts>`

Typedefs
--------

.. index:: pair: typedef; dnnl_engine_t
.. _doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_engine<doxid-structdnnl__engine>`* dnnl_engine_t

An engine handle.

Global Functions
----------------

.. index:: pair: function; convert_to_c
.. _doxid-group__dnnl__api__engine_1gae472e59f404ba6527988b046ef24c743:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` dnnl::convert_to_c(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` akind)

Converts engine kind enum value from C++ API to C API type.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- akind

		- C++ API engine kind enum value.



.. rubric:: Returns:

Corresponding C API engine kind enum value.

.. index:: pair: function; dnnl_engine_get_count
.. _doxid-group__dnnl__api__engine_1gadff5935622df99a2f89acb5cbea09ab5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t DNNL_API dnnl_engine_get_count(:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind)

Returns the number of engines of a particular kind.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- kind

		- Kind of engines to count.



.. rubric:: Returns:

Count of the engines.

.. index:: pair: function; dnnl_engine_create
.. _doxid-group__dnnl__api__engine_1gab84f82f3011349cbfe368b61882834fd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_engine_create(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` kind,
		size_t index
		)

Creates an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Output engine.

	*
		- kind

		- Engine kind.

	*
		- index

		- Engine index that should be between 0 and the count of engines of the requested kind.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_engine_get_kind
.. _doxid-group__dnnl__api__engine_1ga8a38bdce17f51616d03310a8e8764c8c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_engine_get_kind(
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>`* kind
		)

Returns the kind of an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to query.

	*
		- kind

		- Output engine kind.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_engine_destroy
.. _doxid-group__dnnl__api__engine_1ga8d6976b3792cf1ef64d01545929b4d8f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_engine_destroy(:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine)

Destroys an engine.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine

		- Engine to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

