.. index:: pair: class; dnnl::graph::allocator
.. _doxid-classdnnl_1_1graph_1_1allocator:

class dnnl::graph::allocator
============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Allocator. :ref:`More...<details-classdnnl_1_1graph_1_1allocator>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>
	
	class allocator: public allocator_handle
	{
	public:
		// construction
	
		:ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator_1a3eee4ea26be21d46e52dfa8c52764c74>`(
			:ref:`dnnl_graph_host_allocate_f<doxid-group__dnnl__graph__api__allocator_1gae34382069edddd880a407f22c5dfd8e1>` host_malloc,
			:ref:`dnnl_graph_host_deallocate_f<doxid-group__dnnl__graph__api__allocator_1gaaa02889e076ef93c15da152bba7d29b0>` host_free
			);
	
		:ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator_1a0df3aa313d3d7699ab8a3ee687800dd8>`();
	};
.. _details-classdnnl_1_1graph_1_1allocator:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Allocator.

Construction
------------

.. index:: pair: function; allocator
.. _doxid-classdnnl_1_1graph_1_1allocator_1a3eee4ea26be21d46e52dfa8c52764c74:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	allocator(
		:ref:`dnnl_graph_host_allocate_f<doxid-group__dnnl__graph__api__allocator_1gae34382069edddd880a407f22c5dfd8e1>` host_malloc,
		:ref:`dnnl_graph_host_deallocate_f<doxid-group__dnnl__graph__api__allocator_1gaaa02889e076ef93c15da152bba7d29b0>` host_free
		)

Constructs an allocator according to given function pointers.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- host_malloc

		- A pointer to malloc function for CPU

	*
		- host_free

		- A pointer to free function for CPU

.. index:: pair: function; allocator
.. _doxid-classdnnl_1_1graph_1_1allocator_1a0df3aa313d3d7699ab8a3ee687800dd8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	allocator()

Default constructor.

