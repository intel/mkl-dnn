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

	void :ref:`dnnl::graph::set_constant_tensor_cache_capacity<doxid-group__dnnl__graph__api__constant__tensor__cache_1gafb2292cee6b5833286a3c19bad4ee93e>`(
		:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind,
		size_t size
		);

	size_t :ref:`dnnl::graph::get_constant_tensor_cache_capacity<doxid-group__dnnl__graph__api__constant__tensor__cache_1ga737fd9a92da083bc778f43752f8b3ffb>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_set_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1ga9e37974d35ff5aafe1cbae2f69a2ab00>`(int flag);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_get_constant_tensor_cache<doxid-group__dnnl__graph__api__constant__tensor__cache_1ga79be61eb82b59a52145bb730197283c1>`(int* flag);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_set_constant_tensor_cache_capacity<doxid-group__dnnl__graph__api__constant__tensor__cache_1gac9088e72c59a66d02c68e7200ae59b2d>`(
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` eng_kind,
		size_t size
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_get_constant_tensor_cache_capacity<doxid-group__dnnl__graph__api__constant__tensor__cache_1gad6822ef150b8de02dec49ffd49a83f87>`(
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` eng_kind,
		size_t* size
		);

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
.. note:: 

   This API is deprecated and will be removed in future release, please use the set_constant_tensor_cache_capacity API to disable constant tensor cache by setting it's capacity to zero.



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

.. note:: 

   This API is deprecated and will be removed in future release, please use the get_constant_tensor_cache_capacity API to check the enabling status by checking it's capacity.

.. index:: pair: function; set_constant_tensor_cache_capacity
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1gafb2292cee6b5833286a3c19bad4ee93e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dnnl::graph::set_constant_tensor_cache_capacity(
		:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind,
		size_t size
		)

Control the capacity for the constant tensor cache that used for specific engine kind.

This API is thread safe and can be called multiple times at runtime. The capacity is set to zero by default which means the cache is disabled. When calling this API, the corresponding cache will be flushed. Setting capacity to 0 means to clear all cached tensors and disable cache. Once the capacity limit is reached, no new tensors will be cached. If there are multiple devices for an engine kind, the capacity set here is for each device.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- kind

		- The engine kind that the constant tensor cache used for.

	*
		- size

		- The constant tensor cache capacity size to set.

.. index:: pair: function; get_constant_tensor_cache_capacity
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1ga737fd9a92da083bc778f43752f8b3ffb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t dnnl::graph::get_constant_tensor_cache_capacity(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` kind)

Return the current capacity of constant tensor cache.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- kind

		- The engine kind that the constant tensor cache used for.

.. index:: pair: function; dnnl_graph_set_constant_tensor_cache
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1ga9e37974d35ff5aafe1cbae2f69a2ab00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_set_constant_tensor_cache(int flag)

Control the enabling or disabling of constant tensor cache.

This API must be called once before compilation stage. By default, constant tensor cache is disabled in the library.

.. note:: 

   This API is deprecated and will be removed in future release, please use the dnnl_graph_set_constant_tensor_cache_capacity API to disable constant tensor cache by setting it's capacity to zero.



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

.. note:: 

   This API is deprecated and will be removed in future release, please use the dnnl_graph_get_constant_tensor_cache_capacity API to check the enabling status by checking it's capacity.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- flag

		- The constant tensor cache enabling status to query.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``flag`` value is nullptr, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

.. index:: pair: function; dnnl_graph_set_constant_tensor_cache_capacity
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1gac9088e72c59a66d02c68e7200ae59b2d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_set_constant_tensor_cache_capacity(
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` eng_kind,
		size_t size
		)

Control the capacity for the constant tensor cache that used for specific engine kind.

This API is thread safe and can be called multiple times at runtime. The capacity is set to zero by default which means the cache is disabled. When calling this API, the corresponding cache will be flushed. Setting capacity to 0 means to clear all cached tensors and disable cache. Once the capacity limit is reached, no new tensors will be cached. If there are multiple devices for an engine kind, the capacity set here is for each device.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- eng_kind

		- The engine kind that the constant tensor cache used for.

	*
		- size

		- The constant tensor cache capacity size to set.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``eng_kind`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

.. index:: pair: function; dnnl_graph_get_constant_tensor_cache_capacity
.. _doxid-group__dnnl__graph__api__constant__tensor__cache_1gad6822ef150b8de02dec49ffd49a83f87:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_get_constant_tensor_cache_capacity(
		:ref:`dnnl_engine_kind_t<doxid-group__dnnl__api__engine_1ga04b3dd9eba628ea02218a52c4c4363a2>` eng_kind,
		size_t* size
		)

Return the current capacity of constant tensor cache.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- eng_kind

		- The engine kind that the constant tensor cache used for.

	*
		- size

		- The constant tensor cache capacity size to query.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``eng_kind`` value is nullptr or the ``size`` is nullptr, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

