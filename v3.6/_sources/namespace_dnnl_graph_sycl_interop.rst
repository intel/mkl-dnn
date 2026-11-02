.. index:: pair: namespace; dnnl::graph::sycl_interop
.. _doxid-namespacednnl_1_1graph_1_1sycl__interop:

namespace dnnl::graph::sycl_interop
===================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

SYCL interoperability namespace. :ref:`More...<details-namespacednnl_1_1graph_1_1sycl__interop>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace sycl_interop {

	// global functions

	:ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>` :ref:`make_allocator<doxid-namespacednnl_1_1graph_1_1sycl__interop_1afbfd5202a21eebb29d010f14bcbbbb13>`(
		:ref:`dnnl_graph_sycl_allocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>` sycl_malloc,
		:ref:`dnnl_graph_sycl_deallocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>` sycl_free
		);

	:ref:`engine<doxid-structdnnl_1_1engine>` :target:`make_engine_with_allocator<doxid-namespacednnl_1_1graph_1_1sycl__interop_1ae334ece83846829ecb935ed731d10d3d>`(
		const sycl::device& adevice,
		const sycl::context& acontext,
		const :ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>`& alloc
		);

	sycl::event :ref:`execute<doxid-namespacednnl_1_1graph_1_1sycl__interop_1acc5ff56ff0f276367b047c3c73093a67>`(
		:ref:`compiled_partition<doxid-classdnnl_1_1graph_1_1compiled__partition>`& c_partition,
		:ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& inputs,
		std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& outputs,
		const std::vector<sycl::event>& deps = {}
		);

	} // namespace sycl_interop
.. _details-namespacednnl_1_1graph_1_1sycl__interop:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

SYCL interoperability namespace.

Global Functions
----------------

.. index:: pair: function; make_allocator
.. _doxid-namespacednnl_1_1graph_1_1sycl__interop_1afbfd5202a21eebb29d010f14bcbbbb13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`allocator<doxid-classdnnl_1_1graph_1_1allocator>` make_allocator(
		:ref:`dnnl_graph_sycl_allocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga74d9aec0f8f9c3a9da2cbf2df5cc1e8c>` sycl_malloc,
		:ref:`dnnl_graph_sycl_deallocate_f<doxid-group__dnnl__graph__api__sycl__interop_1ga77936c59bb8456176973fa03f990298f>` sycl_free
		)

Constructs an allocator from SYCL malloc and free function pointer.

SYCL allocator should be used for SYCL runtime and host allocator should be used for non-SYCL. Currently, only device USM allocator is supported.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- sycl_malloc

		- The pointer to SYCL malloc function

	*
		- sycl_free

		- The pointer to SYCL free function



.. rubric:: Returns:

Created allocator

.. index:: pair: function; execute
.. _doxid-namespacednnl_1_1graph_1_1sycl__interop_1acc5ff56ff0f276367b047c3c73093a67:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sycl::event execute(
		:ref:`compiled_partition<doxid-classdnnl_1_1graph_1_1compiled__partition>`& c_partition,
		:ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& inputs,
		std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& outputs,
		const std::vector<sycl::event>& deps = {}
		)

Executes a compiled partition in a specified stream and returns a SYCL event.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- c_partition

		- Compiled partition to execute.

	*
		- astream

		- Stream object to run over

	*
		- inputs

		- Arguments map.

	*
		- outputs

		- Arguments map.

	*
		- deps

		- Optional vector with ``sycl::event`` dependencies.



.. rubric:: Returns:

Output event.

