.. index:: pair: group; Allocator
.. _doxid-group__dnnl__graph__api__allocator:

Allocator
=========

.. toctree::
	:hidden:

	class_dnnl_graph_allocator.rst

Overview
~~~~~~~~

Definitions of allocator which is used to acquire memory resources in partition compilation and execution. :ref:`More...<details-group__dnnl__graph__api__allocator>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// typedefs

	typedef void* (*:ref:`dnnl_graph_host_allocate_f<doxid-group__dnnl__graph__api__allocator_1gae34382069edddd880a407f22c5dfd8e1>`)(
		size_t size,
		size_t alignment
		);

	typedef void (*:ref:`dnnl_graph_host_deallocate_f<doxid-group__dnnl__graph__api__allocator_1gaaa02889e076ef93c15da152bba7d29b0>`)(void *);
	typedef struct dnnl_graph_allocator* :ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`;
	typedef const struct dnnl_graph_allocator* :ref:`const_dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499>`;

	// classes

	class :ref:`dnnl::graph::allocator<doxid-classdnnl_1_1graph_1_1allocator>`;

	// global functions

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_allocator_create<doxid-group__dnnl__graph__api__allocator_1gaac19f3f00e51bdd323be1a9073282fcd>`(
		:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`* allocator,
		:ref:`dnnl_graph_host_allocate_f<doxid-group__dnnl__graph__api__allocator_1gae34382069edddd880a407f22c5dfd8e1>` host_malloc,
		:ref:`dnnl_graph_host_deallocate_f<doxid-group__dnnl__graph__api__allocator_1gaaa02889e076ef93c15da152bba7d29b0>` host_free
		);

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_allocator_destroy<doxid-group__dnnl__graph__api__allocator_1gad2c3000cd39878198f6e461a30dd42c8>`(:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>` allocator);

.. _details-group__dnnl__graph__api__allocator:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Definitions of allocator which is used to acquire memory resources in partition compilation and execution.

SYCL allocator (:ref:`dnnl::graph::sycl_interop::make_allocator <doxid-namespacednnl_1_1graph_1_1sycl__interop_1afbfd5202a21eebb29d010f14bcbbbb13>`) should be used for SYCL runtime and host allocator should be used for non-SYCL.

Typedefs
--------

.. index:: pair: typedef; dnnl_graph_host_allocate_f
.. _doxid-group__dnnl__graph__api__allocator_1gae34382069edddd880a407f22c5dfd8e1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef void* (*dnnl_graph_host_allocate_f)(
		size_t size,
		size_t alignment
		)

Allocation call-back function interface for host.

For SYCL allocator, see :ref:`dnnl_graph_sycl_allocate_f <doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>`.

.. index:: pair: typedef; dnnl_graph_host_deallocate_f
.. _doxid-group__dnnl__graph__api__allocator_1gaaa02889e076ef93c15da152bba7d29b0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef void (*dnnl_graph_host_deallocate_f)(void *)

Deallocation call-back function interface for host.

For SYCL allocator, see :ref:`dnnl_graph_sycl_deallocate_f <doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>`.

.. index:: pair: typedef; dnnl_graph_allocator_t
.. _doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef struct dnnl_graph_allocator* dnnl_graph_allocator_t

An allocator handle.

.. index:: pair: typedef; const_dnnl_graph_allocator_t
.. _doxid-group__dnnl__graph__api__allocator_1ga82fcfed1f65be71d0d1c5cf865f8f499:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef const struct dnnl_graph_allocator* const_dnnl_graph_allocator_t

A constant allocator handle.

Global Functions
----------------

.. index:: pair: function; dnnl_graph_allocator_create
.. _doxid-group__dnnl__graph__api__allocator_1gaac19f3f00e51bdd323be1a9073282fcd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_allocator_create(
		:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>`* allocator,
		:ref:`dnnl_graph_host_allocate_f<doxid-group__dnnl__graph__api__allocator_1gae34382069edddd880a407f22c5dfd8e1>` host_malloc,
		:ref:`dnnl_graph_host_deallocate_f<doxid-group__dnnl__graph__api__allocator_1gaaa02889e076ef93c15da152bba7d29b0>` host_free
		)

Creates a host allocator with the given allocation and deallocation call-back function pointers.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- allocator

		- Output allocator.

	*
		- host_malloc

		- A pointer to malloc function for host.

	*
		- host_free

		- A pointer to free function for host.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

.. index:: pair: function; dnnl_graph_allocator_destroy
.. _doxid-group__dnnl__graph__api__allocator_1gad2c3000cd39878198f6e461a30dd42c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_allocator_destroy(:ref:`dnnl_graph_allocator_t<doxid-group__dnnl__graph__api__allocator_1ga7e5ba6788922a000348e762ac8c88cc6>` allocator)

Destroys an allocator.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- allocator

		- The allocator to be destroyed.



.. rubric:: Returns:

:ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success or a status describing the error otherwise.

