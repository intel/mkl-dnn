.. index:: pair: group; Profiling
.. _doxid-group__dnnl__api__profiling:

Profiling
=========

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_reset_profiling<doxid-group__dnnl__api__profiling_1gaaf7e8e00d675e7362ccf75b30a9c47bd>`(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_query_profiling_data<doxid-group__dnnl__api__profiling_1gae92506d856399892636be1c86a3a94a7>`(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		:ref:`dnnl_profiling_data_kind_t<doxid-group__dnnl__api__memory_1ga7ac0b200fe8227f70d08832ffc9c51f4>` data_kind,
		int* num_entries,
		uint64_t* data
		);

.. _details-group__dnnl__api__profiling:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; dnnl_reset_profiling
.. _doxid-group__dnnl__api__profiling_1gaaf7e8e00d675e7362ccf75b30a9c47bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_reset_profiling(:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream)

Resets a profiler's state.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Stream associated with the profiler.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

.. index:: pair: function; dnnl_query_profiling_data
.. _doxid-group__dnnl__api__profiling_1gae92506d856399892636be1c86a3a94a7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_query_profiling_data(
		:ref:`dnnl_stream_t<doxid-group__dnnl__api__stream_1ga735eb19cfd205c108c468b5657de4eca>` stream,
		:ref:`dnnl_profiling_data_kind_t<doxid-group__dnnl__api__memory_1ga7ac0b200fe8227f70d08832ffc9c51f4>` data_kind,
		int* num_entries,
		uint64_t* data
		)

Queries profiling data. The profiling data accumulates for each primitive execution. The ``num_entries`` will be equal to the number of executions since the last ``dnnl_reset_profiling`` call. In order to query the ``num_entries`` the ``data`` parameter should be NULL. When ``data`` is NULL then the ``data_kind`` parameter is ignored.

The profiling data can be reset by calling :ref:`dnnl_reset_profiling <doxid-group__dnnl__api__profiling_1gaaf7e8e00d675e7362ccf75b30a9c47bd>`.

.. note:: 

   It is required to wait for all submitted primitives to complete using :ref:`dnnl_stream_wait <doxid-group__dnnl__api__stream_1ga6a8175b9384349b1ee73a78a24b5883f>` prior to querying profiling data.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Stream that was used for executing a primitive that is being profiled.

	*
		- data_kind

		- Profiling data kind to query.

	*
		- num_entries

		- Number of profiling data entries.

	*
		- data

		- Profiling data.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success and a status describing the error otherwise.

