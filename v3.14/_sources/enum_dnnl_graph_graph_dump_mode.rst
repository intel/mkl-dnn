.. index:: pair: enum; graph_dump_mode
.. _doxid-group__dnnl__graph__api__dump__mode_1ga48d03b4285480cd0df9d587ddeec293d:

enum dnnl::graph::graph_dump_mode
=================================

Overview
~~~~~~~~

Dump mode bitmask for graph debugging utilities. :ref:`More...<details-group__dnnl__graph__api__dump__mode_1ga48d03b4285480cd0df9d587ddeec293d>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>

	enum graph_dump_mode
	{
	    :ref:`none<doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293da334c4a4c42fdb79d7ebc3e73b517e6f8>`     = dnnl_graph_dump_mode_none,
	    :ref:`subgraph<doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293da24dba1f0943308cd60c77ff0a1662a57>` = dnnl_graph_dump_mode_subgraph,
	    :ref:`graph<doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293daf8b0b924ebd7046dbfa85a856e4682c8>`    = dnnl_graph_dump_mode_graph,
	};

.. _details-group__dnnl__graph__api__dump__mode_1ga48d03b4285480cd0df9d587ddeec293d:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Dump mode bitmask for graph debugging utilities.

Enum Values
-----------

.. index:: pair: enumvalue; none
.. _doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293da334c4a4c42fdb79d7ebc3e73b517e6f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	none

Disable all graph dumps.

.. index:: pair: enumvalue; subgraph
.. _doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293da24dba1f0943308cd60c77ff0a1662a57:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	subgraph

Dump subgraphs extracted during partitioning.

.. index:: pair: enumvalue; graph
.. _doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293daf8b0b924ebd7046dbfa85a856e4682c8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	graph

Dump the full graph prior to partitioning.

