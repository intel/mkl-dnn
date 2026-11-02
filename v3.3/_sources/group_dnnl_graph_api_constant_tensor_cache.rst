.. index:: pair: group; Constant Tensor Cache
.. _doxid-group__dnnl__graph__api__constant__tensor__cache:

Constant Tensor Cache
=====================

.. toctree::
	:hidden:

Overview
~~~~~~~~

A set of functions that provide constant tensor cache control. :ref:`More...<details-group__dnnl__graph__api__constant__tensor__cache>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// global functions

	void :ref:`dnnl::graph::set_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1gad0a7165aed94d587fb232b0e7e4783ce>`(int flag);
	int :ref:`dnnl::graph::get_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1gaad7e9879171c5825df486370f73b8cf2>`();
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_set_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1ga9e37974d35ff5aafe1cbae2f69a2ab00>`(int flag);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_get_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1ga79be61eb82b59a52145bb730197283c1>`(int* flag);

.. _details-group__dnnl__graph__api__constant__tensor__cache:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A set of functions that provide constant tensor cache control.

Global Functions
----------------

.. index:: pair: function; set_constant_tensor_cache
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1gad0a7165aed94d587fb232b0e7e4783ce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dnnl::graph::set_constant_tensor_cache(int flag)

Control the enabling or disabling of constant tensor cache.

This API must be called once before compilation stage. By default, constant tensor cache is disabled in the library.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- flag

		- Set to positive value to enable the cache and set to 0 to disable the cache. Negative values are invalid.

.. index:: pair: function; get_constant_tensor_cache
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1gaad7e9879171c5825df486370f73b8cf2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int dnnl::graph::get_constant_tensor_cache()

Return the enabling status of constant tensor cache.

.. index:: pair: function; dnnl_graph_set_constant_tensor_cache
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1ga9e37974d35ff5aafe1cbae2f69a2ab00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_set_constant_tensor_cache(int flag)

Control the enabling or disabling of constant tensor cache.

This API must be called once before compilation stage. By default, constant tensor cache is disabled in the library.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- flag

		- Set to positive value to enable the cache and set to 0 to disable the cache. Negative values are invalid.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``flag`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

.. index:: pair: function; dnnl_graph_get_constant_tensor_cache
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1ga79be61eb82b59a52145bb730197283c1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_get_constant_tensor_cache(int* flag)

Return the enabling or disabling status of constant tensor cache.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- flag

		- The constant tensor cache enabling status to query.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``flag`` value is nullptr, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

