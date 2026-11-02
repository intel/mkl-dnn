.. index:: pair: enum; dnnl_graph_dump_mode_t
.. _doxid-group__dnnl__graph__api__dump__mode_1gad9501cc148ae98ba5477d524b5149994:

enum dnnl_graph_dump_mode_t
===========================

Overview
~~~~~~~~

Dump mode bitmask for graph debugging utilities. :ref:`More...<details-group__dnnl__graph__api__dump__mode_1gad9501cc148ae98ba5477d524b5149994>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph_types.h>

	enum dnnl_graph_dump_mode_t
	{
	    :ref:`dnnl_graph_dump_mode_none<doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994aae63eb4551a7ce6fbb924dfe5e2222cb>`     = 0x0U,
	    :ref:`dnnl_graph_dump_mode_subgraph<doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994af9b1327c82bceda2e70fe33503608c73>` = 0x1U,
	    :ref:`dnnl_graph_dump_mode_graph<doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994aa30033e6d99aa1088634f476eab6235c>`    = 0x2U,
	};

.. _details-group__dnnl__graph__api__dump__mode_1gad9501cc148ae98ba5477d524b5149994:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Dump mode bitmask for graph debugging utilities.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_graph_dump_mode_none
.. _doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994aae63eb4551a7ce6fbb924dfe5e2222cb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_dump_mode_none

Disable all graph dumps.

.. index:: pair: enumvalue; dnnl_graph_dump_mode_subgraph
.. _doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994af9b1327c82bceda2e70fe33503608c73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_dump_mode_subgraph

Dump subgraphs extracted during partitioning.

.. index:: pair: enumvalue; dnnl_graph_dump_mode_graph
.. _doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994aa30033e6d99aa1088634f476eab6235c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_dump_mode_graph

Dump the full graph prior to partitioning.

