.. index:: pair: class; dnnl::graph::graph
.. _doxid-classdnnl_1_1graph_1_1graph:

class dnnl::graph::graph
========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

A graph object. :ref:`More...<details-classdnnl_1_1graph_1_1graph>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>
	
	class graph: public graph_handle
	{
	public:
		// construction
	
		:ref:`graph<doxid-classdnnl_1_1graph_1_1graph_1ab9989b31612c971d723b21bce61f3812>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind);
		:ref:`graph<doxid-classdnnl_1_1graph_1_1graph_1af26fca906c40d577ae129c635bc08039>`(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, :ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode);

		// methods
	
		:ref:`status<doxid-group__dnnl__graph__api__status_1gadf0aa6dbce815494117ab7f44ffb598c>` :ref:`add_op<doxid-classdnnl_1_1graph_1_1graph_1a1cdf41276f953ecc482df858408c9ff0>`(const :ref:`op<doxid-classdnnl_1_1graph_1_1op>`& op, bool allow_exception = true);
		void :ref:`finalize<doxid-classdnnl_1_1graph_1_1graph_1a4454e3093a112022cd607c4c3cc66ee2>`();
		bool :ref:`is_finalized<doxid-classdnnl_1_1graph_1_1graph_1ac92aaf7475714f78ed18327a9f9210bf>`() const;
		std::vector<:ref:`partition<doxid-classdnnl_1_1graph_1_1partition>`> :ref:`get_partitions<doxid-classdnnl_1_1graph_1_1graph_1a116d3552e3b0e6c739a1564329bde014>`(:ref:`partition::policy<doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3c>` policy = :ref:`partition::policy::fusion<doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3ca051de32597041e41f73b97d61c67a13b>`);
	};
.. _details-classdnnl_1_1graph_1_1graph:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A graph object.

Construction
------------

.. index:: pair: function; graph
.. _doxid-classdnnl_1_1graph_1_1graph_1ab9989b31612c971d723b21bce61f3812:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	graph(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind)

Constructs a graph with an engine kind.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine_kind

		- Engine kind.

.. index:: pair: function; graph
.. _doxid-classdnnl_1_1graph_1_1graph_1af26fca906c40d577ae129c635bc08039:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	graph(:ref:`engine::kind<doxid-structdnnl_1_1engine_1a2635da16314dcbdb9bd9ea431316bb1a>` engine_kind, :ref:`fpmath_mode<doxid-group__dnnl__api__fpmath__mode_1ga0ad94cbef13dce222933422bfdcfa725>` mode)

Creates a new empty graph with an engine kind and a floating-point math mode.

All partitions returned from the graph will inherit the engine kind and floating-point math mode.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- engine_kind

		- Engine kind.

	*
		- mode

		- Floating-point math mode.

Methods
-------

.. index:: pair: function; add_op
.. _doxid-classdnnl_1_1graph_1_1graph_1a1cdf41276f953ecc482df858408c9ff0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__graph__api__status_1gadf0aa6dbce815494117ab7f44ffb598c>` add_op(const :ref:`op<doxid-classdnnl_1_1graph_1_1op>`& op, bool allow_exception = true)

Adds an op into the graph to construct a computational DAG.

The API will return failure if the operator has already been added to the graph or the operation cannot pass the schema check in the library (eg. input and output numbers and data types, the attributes of the operation, etc.).



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- op

		- An operation to be added.

	*
		- allow_exception

		- A flag indicating whether the method is allowed to throw an exception if it fails to add the op to the graph.



.. rubric:: Returns:

:ref:`status::success <doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca260ca9dd8a4577fc00b7bd5810298076>` or a status describing the error otherwise.

.. index:: pair: function; finalize
.. _doxid-classdnnl_1_1graph_1_1graph_1a4454e3093a112022cd607c4c3cc66ee2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void finalize()

Finalizes a graph.

It means users have finished adding operations into the graph and the graph is ready for partitioning. Adding a new operation into a finalized graph will return failures. Similarly, partitioning on a un-finalized graph will also return failures.

.. index:: pair: function; is_finalized
.. _doxid-classdnnl_1_1graph_1_1graph_1ac92aaf7475714f78ed18327a9f9210bf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool is_finalized() const

Checks if a graph is finalized.



.. rubric:: Returns:

True if the graph is finalized or false if the graph is not finalized.

.. index:: pair: function; get_partitions
.. _doxid-classdnnl_1_1graph_1_1graph_1a116d3552e3b0e6c739a1564329bde014:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<:ref:`partition<doxid-classdnnl_1_1graph_1_1partition>`> get_partitions(:ref:`partition::policy<doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3c>` policy = :ref:`partition::policy::fusion<doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3ca051de32597041e41f73b97d61c67a13b>`)

Gets filtered partitions from a graph.

Partitions will be claimed internally according to the capability of the library, the engine kind of the graph, and the policy.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- policy

		- Partition policy, defaults to policy :ref:`dnnl::graph::partition::policy::fusion <doxid-classdnnl_1_1graph_1_1partition_1a439c0490ea8ea85f2a12ec7b320a9a3ca051de32597041e41f73b97d61c67a13b>`.



.. rubric:: Returns:

A vector storing the partitions.

