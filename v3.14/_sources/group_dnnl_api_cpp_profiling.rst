.. index:: pair: group; Profiling
.. _doxid-group__dnnl__api__cpp__profiling:

Profiling
=========

.. toctree::
	:hidden:

	enum_dnnl_profiling_data_kind.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl::profiling_data_kind<doxid-group__dnnl__api__cpp__profiling_1gab19f8c7379c446429c9a4b043d64b4aa>`;

	// global functions

	void :ref:`dnnl::reset_profiling<doxid-group__dnnl__api__cpp__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e>`(:ref:`stream<doxid-structdnnl_1_1stream>`& stream);

	std::vector<uint64_t> :ref:`dnnl::get_profiling_data<doxid-group__dnnl__api__cpp__profiling_1ga0dc451b94cbeacb7a5e0c73c3071ee4e>`(
		:ref:`stream<doxid-structdnnl_1_1stream>`& stream,
		:ref:`profiling_data_kind<doxid-group__dnnl__api__cpp__profiling_1gab19f8c7379c446429c9a4b043d64b4aa>` data_kind
		);

.. _details-group__dnnl__api__cpp__profiling:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Global Functions
----------------

.. index:: pair: function; reset_profiling
.. _doxid-group__dnnl__api__cpp__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dnnl::reset_profiling(:ref:`stream<doxid-structdnnl_1_1stream>`& stream)

Resets a profiler's state.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Stream associated with the profiler.

.. index:: pair: function; get_profiling_data
.. _doxid-group__dnnl__api__cpp__profiling_1ga0dc451b94cbeacb7a5e0c73c3071ee4e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<uint64_t> dnnl::get_profiling_data(
		:ref:`stream<doxid-structdnnl_1_1stream>`& stream,
		:ref:`profiling_data_kind<doxid-group__dnnl__api__cpp__profiling_1gab19f8c7379c446429c9a4b043d64b4aa>` data_kind
		)

Returns requested profiling data. The profiling data accumulates for each primitive execution. The size of the vector will be equal to the number of executions since the last ``:ref:`dnnl::reset_profiling <doxid-group__dnnl__api__cpp__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e>``` call.

The profiling data can be reset by calling :ref:`dnnl::reset_profiling <doxid-group__dnnl__api__cpp__profiling_1ga1d9547121faf3f10c23989c3ef05bc1e>`.

.. note:: 

   It is required to wait for all submitted primitives to complete using :ref:`dnnl::stream::wait <doxid-structdnnl_1_1stream_1a59985fa8746436057cf51a820ef8929c>` prior to querying profiling data.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- stream

		- Stream that was used for executing a primitive that is being profiled.

	*
		- data_kind

		- Profiling data kind to query.



.. rubric:: Returns:

A vector with the requested profiling data.

