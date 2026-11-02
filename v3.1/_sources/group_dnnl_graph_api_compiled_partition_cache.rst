.. index:: pair: group; Compiled Partition Cache
.. _doxid-group__dnnl__graph__api__compiled__partition__cache:

Compiled Partition Cache
========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

A set of functions that provide compiled partition cache control. :ref:`More...<details-group__dnnl__graph__api__compiled__partition__cache>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// global functions

	int :ref:`dnnl::graph::get_compiled_partition_cache_capacity<doxid-group__dnnl__graph__api__compiled__partition__cache_1ga6a1bba962b6499e55ba1fb81692d04b7>`();
	void :ref:`dnnl::graph::set_compiled_partition_cache_capacity<doxid-group__dnnl__graph__api__compiled__partition__cache_1ga2260760847caf233f319967e73811319>`(int capacity);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_get_compiled_partition_cache_capacity<doxid-group__dnnl__graph__api__compiled__partition__cache_1ga341079185a3e263dc490a8d24d0fdc94>`(int* capacity);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_set_compiled_partition_cache_capacity<doxid-group__dnnl__graph__api__compiled__partition__cache_1gabed28f32f3f39e2b4053c5b53620a292>`(int capacity);

.. _details-group__dnnl__graph__api__compiled__partition__cache:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A set of functions that provide compiled partition cache control.

Global Functions
----------------

.. index:: pair: function; get_compiled_partition_cache_capacity
.. _doxid-group__dnnl__graph__api__compiled__partition__cache_1ga6a1bba962b6499e55ba1fb81692d04b7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int dnnl::graph::get_compiled_partition_cache_capacity()

Returns the number of compiled partition that can be held in the compiled partition cache at the same time.

.. index:: pair: function; set_compiled_partition_cache_capacity
.. _doxid-group__dnnl__graph__api__compiled__partition__cache_1ga2260760847caf233f319967e73811319:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dnnl::graph::set_compiled_partition_cache_capacity(int capacity)

Sets a number of compiled partitions that can be held in the compiled partition cache at the same time.

The default capacity of compiled partition cache is 1024.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- capacity

		- Compiled partition cache capacity to set. The default cache capacity is 1024. If a new ``capacity`` is less than a number of compiled partition that the compiled partition cache already has, then the excess entries will be evicted. Setting the ``capacity`` to 0 clears the compiled partition cache and disables it. Concurrently modifying ``capacity`` is safe.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``capacity`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

.. index:: pair: function; dnnl_graph_get_compiled_partition_cache_capacity
.. _doxid-group__dnnl__graph__api__compiled__partition__cache_1ga341079185a3e263dc490a8d24d0fdc94:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_get_compiled_partition_cache_capacity(int* capacity)

Returns the number of compiled partitions that can be held in the compiled partition cache at the same time.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- capacity

		- Compiled partition cache capacity to query. Concurrently accessing ``capacity`` is safe.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``capacity`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

.. index:: pair: function; dnnl_graph_set_compiled_partition_cache_capacity
.. _doxid-group__dnnl__graph__api__compiled__partition__cache_1gabed28f32f3f39e2b4053c5b53620a292:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_set_compiled_partition_cache_capacity(int capacity)

Sets a number of compiled partitions that can be held in the compiled partition cache at the same time.

The default capacity of compiled partition cache is 1024.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- capacity

		- Compiled partition cache capacity to set. The default cache capacity is 1024. If a new ``capacity`` is less than a number of compiled partition that the compiled partition cache already has, then the excess entries will be evicted. Setting the ``capacity`` to 0 clears the compiled partition cache and disables it. Concurrently modifying ``capacity`` is safe.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``capacity`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

