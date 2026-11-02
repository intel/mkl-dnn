.. index:: pair: group; Stream
.. _doxid-group__dnnl__api__stream:

Stream
======

.. toctree::
	:hidden:

	enum_dnnl_stream_flags_t.rst
	struct_dnnl_stream.rst
	struct_dnnl_stream-2.rst

Overview
~~~~~~~~

An encapsulation of execution context tied to a particular engine. :ref:`More...<details-group__dnnl__api__stream>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef struct :ref:`dnnl_stream<doxid-structdnnl__stream>`* :ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`;
	typedef const struct :ref:`dnnl_stream<doxid-structdnnl__stream>`* :ref:`const_dnnl_stream_t<doxid-group__dnnl__api__stream_1gaeac91f003af4e2138c84082acc126c36>`;

	// enums

	enum :ref:`dnnl_stream_flags_t<doxid-group__dnnl__api__stream_1ga3d74cfed8fe92b0e4498a1f2bdab5547>`;

	// structs

	struct :ref:`dnnl_stream<doxid-structdnnl__stream>`;
	struct :ref:`dnnl::stream<doxid-structdnnl_1_1stream>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_stream_create<doxid-group__dnnl__api__stream_1gaefca700bdec59b22c05f248df5bb3354>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		unsigned flags
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_stream_get_engine<doxid-group__dnnl__api__stream_1ga817016eb87a4d87a889f32b52b71a93b>`(
		:ref:`const_dnnl_stream_t<doxid-group__dnnl__api__stream_1gaeac91f003af4e2138c84082acc126c36>` stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_stream_wait<doxid-group__dnnl__api__stream_1ga6a8175b9384349b1ee73a78a24b5883f>`(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_stream_destroy<doxid-group__dnnl__api__stream_1gae7fe8b23136cafa62a39301799cd6e44>`(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream);

.. _details-group__dnnl__api__stream:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

An encapsulation of execution context tied to a particular engine.



.. rubric:: See also:

:ref:`Basic Concepts <doxid-dev_guide_basic_concepts>`

Typedefs
--------

.. index:: pair: typedef; dnnl_stream_t
.. _doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct :ref:`dnnl_stream<doxid-structdnnl__stream>`* dnnl_stream_t

An execution stream handle.

.. index:: pair: typedef; const_dnnl_stream_t
.. _doxid-group__dnnl__api__stream_1gaeac91f003af4e2138c84082acc126c36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct :ref:`dnnl_stream<doxid-structdnnl__stream>`* const_dnnl_stream_t

A constant execution stream handle.

Global Functions
----------------

.. index:: pair: function; dnnl_stream_create
.. _doxid-group__dnnl__api__stream_1gaefca700bdec59b22c05f248df5bb3354:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_stream_create(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>`* stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>` engine,
		unsigned flags
		)

Creates an execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Output execution stream.

	*
		- engine

		- Engine to create the execution stream on.

	*
		- flags

		- Stream behavior flags (



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.



.. rubric:: See also:

:ref:`dnnl_stream_flags_t <doxid-group__dnnl__api__stream_1ga3d74cfed8fe92b0e4498a1f2bdab5547>`).

.. index:: pair: function; dnnl_stream_get_engine
.. _doxid-group__dnnl__api__stream_1ga817016eb87a4d87a889f32b52b71a93b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_stream_get_engine(
		:ref:`const_dnnl_stream_t<doxid-group__dnnl__api__stream_1gaeac91f003af4e2138c84082acc126c36>` stream,
		:ref:`dnnl_engine_t<doxid-group__dnnl__api__engine_1ga1ce7843660e8203ed6e1af9bfd23e14f>`* engine
		)

Returns the engine of a stream object.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Stream object.

	*
		- engine

		- Output engine on which the stream is created.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_stream_wait
.. _doxid-group__dnnl__api__stream_1ga6a8175b9384349b1ee73a78a24b5883f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_stream_wait(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream)

Waits for all primitives in the execution stream to finish computations.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Execution stream.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_stream_destroy
.. _doxid-group__dnnl__api__stream_1gae7fe8b23136cafa62a39301799cd6e44:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_stream_destroy(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream)

Destroys an execution stream.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Execution stream to destroy.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

