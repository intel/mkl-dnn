.. index:: pair: group; Primitive Cache
.. _doxid-group__dnnl__api__primitive__cache:

Primitive Cache
===============

.. toctree::
	:hidden:

Overview
~~~~~~~~

A set of functions that provide primitive cache control. :ref:`More...<details-group__dnnl__api__primitive__cache>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// global functions

	int :ref:`dnnl::get_primitive_cache_capacity<doxid-group__dnnl__api__primitive__cache_1gacc0f23351595504f3e2c2b6fcf603770>`();
	void :ref:`dnnl::set_primitive_cache_capacity<doxid-group__dnnl__api__primitive__cache_1ga12eefad64ac6917a161994c005abe69c>`(int capacity);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_get_primitive_cache_capacity<doxid-group__dnnl__api__primitive__cache_1gaaffb070446181187b04ee1a321cc24f0>`(int* capacity);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_set_primitive_cache_capacity<doxid-group__dnnl__api__primitive__cache_1ga53456304297195ae9f053cc60ffe70a2>`(int capacity);

.. _details-group__dnnl__api__primitive__cache:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A set of functions that provide primitive cache control.

Global Functions
----------------

.. index:: pair: function; get_primitive_cache_capacity
.. _doxid-group__dnnl__api__primitive__cache_1gacc0f23351595504f3e2c2b6fcf603770:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int dnnl::get_primitive_cache_capacity()

Returns the number of primitives that can be held in the primitive cache at the same time.

.. index:: pair: function; set_primitive_cache_capacity
.. _doxid-group__dnnl__api__primitive__cache_1ga12eefad64ac6917a161994c005abe69c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void dnnl::set_primitive_cache_capacity(int capacity)

Sets a number of primitives that can be held in the primitive cache at a time.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- capacity

		- Primitive cache capacity to set. If a new ``capacity`` is less than a number of primitives that the primitive cache already has then the excess entries will be evicted. Setting the ``capacity`` to 0 clears the primitive cache and disables it. Concurrently modifying ``capacity`` is safe.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``capacity`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; dnnl_get_primitive_cache_capacity
.. _doxid-group__dnnl__api__primitive__cache_1gaaffb070446181187b04ee1a321cc24f0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_get_primitive_cache_capacity(int* capacity)

Returns the number of primitives that can be held in the primitive cache at the same time.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- capacity

		- Primitive cache capacity to query. Concurrently accessing ``capacity`` is safe.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``capacity`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; dnnl_set_primitive_cache_capacity
.. _doxid-group__dnnl__api__primitive__cache_1ga53456304297195ae9f053cc60ffe70a2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_set_primitive_cache_capacity(int capacity)

Sets a number of primitives that can be held in the primitive cache at a time.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- capacity

		- Primitive cache capacity to set. If a new ``capacity`` is less than a number of primitives that the primitive cache already has then the excess entries will be evicted. Setting the ``capacity`` to 0 clears the primitive cache and disables it. Concurrently modifying ``capacity`` is safe.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` / :ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``capacity`` value is invalid, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` / :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

