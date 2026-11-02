.. index:: pair: class; dnnl::graph::compiled_partition
.. _doxid-classdnnl_1_1graph_1_1compiled__partition:

class dnnl::graph::compiled_partition
=====================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

A compiled partition object. :ref:`More...<details-classdnnl_1_1graph_1_1compiled__partition>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>
	
	class compiled_partition: public compiled_partition_handle
	{
	public:
		// construction
	
		:ref:`compiled_partition<doxid-classdnnl_1_1graph_1_1compiled__partition_1a3cf34b8ed11d40de19bac51d2d94ae30>`();
		:ref:`compiled_partition<doxid-classdnnl_1_1graph_1_1compiled__partition_1a398d2eb4d4d6253a1413af74c215b671>`(:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>` compiled_partition);

		// methods
	
		:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>` :ref:`query_logical_tensor<doxid-classdnnl_1_1graph_1_1compiled__partition_1a85962826e94cc3cefb3c19c0fadc4e09>`(size_t tid) const;
		std::vector<std::pair<size_t, size_t>> :ref:`get_inplace_ports<doxid-classdnnl_1_1graph_1_1compiled__partition_1ab80ec9f6a37ddbf7b2636fb35acf74f2>`() const;
	
		void :ref:`execute<doxid-classdnnl_1_1graph_1_1compiled__partition_1a558ed47b3cbc5cc2167001da3faa0339>`(
			:ref:`stream<doxid-structdnnl_1_1stream>`& astream,
			const std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& inputs,
			const std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& outputs
			) const;
	};
.. _details-classdnnl_1_1graph_1_1compiled__partition:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A compiled partition object.

Construction
------------

.. index:: pair: function; compiled_partition
.. _doxid-classdnnl_1_1graph_1_1compiled__partition_1a3cf34b8ed11d40de19bac51d2d94ae30:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	compiled_partition()

Default constructor. Constructs an empty object.

.. index:: pair: function; compiled_partition
.. _doxid-classdnnl_1_1graph_1_1compiled__partition_1a398d2eb4d4d6253a1413af74c215b671:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	compiled_partition(:ref:`dnnl_graph_compiled_partition_t<doxid-group__dnnl__graph__api__compiled__partition_1ga7578c6d5c3efdbaddd7b8e19429f546a>` compiled_partition)

Constructs a compiled partition object.

Methods
-------

.. index:: pair: function; query_logical_tensor
.. _doxid-classdnnl_1_1graph_1_1compiled__partition_1a85962826e94cc3cefb3c19c0fadc4e09:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`logical_tensor<doxid-classdnnl_1_1graph_1_1logical__tensor>` query_logical_tensor(size_t tid) const

Queries an input or output logical tensor according to tensor ID.

If the tensor ID doesn't belong to any input or output of the compiled partition, an exception will be raised by the API.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- tid

		- The unique id of required tensor.



.. rubric:: Returns:

The logical tensor.

.. index:: pair: function; get_inplace_ports
.. _doxid-classdnnl_1_1graph_1_1compiled__partition_1ab80ec9f6a37ddbf7b2636fb35acf74f2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<std::pair<size_t, size_t>> get_inplace_ports() const

Returns the hint of in-place pairs from a compiled partition.

It indicates that an input and an output of the partition can share the same memory buffer for computation. In-place computation helps to reduce the memory footprint and improves cache locality. But since the library may not have a global view of user's application, it's possible that the input tensor is used at other places in user's computation graph. In this case, the user should take the in-place pair as a hint and pass a different memory buffer for output tensor to avoid overwriting the input memory buffer which will probably cause unexpected incorrect results.



.. rubric:: Returns:

A list of pairs of input and output IDs.

.. index:: pair: function; execute
.. _doxid-classdnnl_1_1graph_1_1compiled__partition_1a558ed47b3cbc5cc2167001da3faa0339:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void execute(
		:ref:`stream<doxid-structdnnl_1_1stream>`& astream,
		const std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& inputs,
		const std::vector<:ref:`tensor<doxid-classdnnl_1_1graph_1_1tensor>`>& outputs
		) const

Execute a compiled partition.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- astream

		- Stream object to run over.

	*
		- inputs

		- A list of input tensors.

	*
		- outputs

		- A list of output tensors.

