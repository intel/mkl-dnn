.. index:: pair: group; Dump Mode
.. _doxid-group__dnnl__graph__api__dump__mode:

Dump Mode
=========

.. toctree::
	:hidden:

	enum_dnnl_graph_dump_mode_t.rst
	enum_dnnl_graph_graph_dump_mode.rst

Overview
~~~~~~~~

Control graph dumping behavior :ref:`More...<details-group__dnnl__graph__api__dump__mode>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	// enums

	enum :ref:`dnnl_graph_dump_mode_t<doxid-group__dnnl__graph__api__dump__mode_1gad9501cc148ae98ba5477d524b5149994>`;
	enum :ref:`dnnl::graph::graph_dump_mode<doxid-group__dnnl__graph__api__dump__mode_1ga48d03b4285480cd0df9d587ddeec293d>`;

	// global functions

	:ref:`status<doxid-group__dnnl__graph__api__status_1gadf0aa6dbce815494117ab7f44ffb598c>` :ref:`dnnl::graph::set_dump_mode<doxid-group__dnnl__graph__api__dump__mode_1ga9f15a813c28dd3839d741484b15fdbea>`(:ref:`graph_dump_mode<doxid-group__dnnl__graph__api__dump__mode_1ga48d03b4285480cd0df9d587ddeec293d>` modes);
	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API :ref:`dnnl_graph_set_dump_mode<doxid-group__dnnl__graph__api__dump__mode_1gafee362a7c0b089f1aa319b6496640702>`(:ref:`dnnl_graph_dump_mode_t<doxid-group__dnnl__graph__api__dump__mode_1gad9501cc148ae98ba5477d524b5149994>` modes);

.. _details-group__dnnl__graph__api__dump__mode:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Control graph dumping behavior

Global Functions
----------------

.. index:: pair: function; set_dump_mode
.. _doxid-group__dnnl__graph__api__dump__mode_1ga9f15a813c28dd3839d741484b15fdbea:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`status<doxid-group__dnnl__graph__api__status_1gadf0aa6dbce815494117ab7f44ffb598c>` dnnl::graph::set_dump_mode(:ref:`graph_dump_mode<doxid-group__dnnl__graph__api__dump__mode_1ga48d03b4285480cd0df9d587ddeec293d>` modes)

Configures graph dump modes at runtime.

.. note:: 

   Enabling graph dump affects performance. This setting overrides the ONEDNN_GRAPH_DUMP environment variable.
   
   
Bitmask combinations using bitwise operators are supported. For instance, ``graph | subgraph`` enables both modes, ``none | graph`` behaves like ``graph``, and ``none & graph`` behaves like ``none``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- modes

		- 
		  Bitmask composed of values from :ref:`dnnl::graph::graph_dump_mode <doxid-group__dnnl__graph__api__dump__mode_1ga48d03b4285480cd0df9d587ddeec293d>`. Accepted values:
		  
		  * :ref:`dnnl::graph::graph_dump_mode::graph <doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293daf8b0b924ebd7046dbfa85a856e4682c8>` : dump the full graph prior to partitioning.
		  
		  * :ref:`dnnl::graph::graph_dump_mode::subgraph <doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293da24dba1f0943308cd60c77ff0a1662a57>` : dump each partitioned subgraph.
		  
		  * :ref:`dnnl::graph::graph_dump_mode::none <doxid-group__dnnl__graph__api__dump__mode_1gga48d03b4285480cd0df9d587ddeec293da334c4a4c42fdb79d7ebc3e73b517e6f8>` : disable all graph dumping.



.. rubric:: Returns:

:ref:`dnnl::status::invalid_arguments <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` if the ``modes`` value contains unsupported bits or graph dump is disabled, and :ref:`dnnl::status::success <doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>` on success.

.. index:: pair: function; dnnl_graph_set_dump_mode
.. _doxid-group__dnnl__graph__api__dump__mode_1gafee362a7c0b089f1aa319b6496640702:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`dnnl_status_t<doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>` DNNL_API dnnl_graph_set_dump_mode(:ref:`dnnl_graph_dump_mode_t<doxid-group__dnnl__graph__api__dump__mode_1gad9501cc148ae98ba5477d524b5149994>` modes)

Configures graph dump modes at runtime.

.. note:: 

   Enabling graph dump affects performance. This setting overrides the ONEDNN_GRAPH_DUMP environment variable.
   
   
Bitmask combinations using bitwise operators are supported. For instance, ``graph | subgraph`` enables both modes, ``none | graph`` behaves like ``graph``, and ``none & graph`` behaves like ``none``.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- modes

		- 
		  Bitmask composed of values from :ref:`dnnl_graph_dump_mode_t <doxid-group__dnnl__graph__api__dump__mode_1gad9501cc148ae98ba5477d524b5149994>`. Accepted values:
		  
		  * :ref:`dnnl_graph_dump_mode_graph <doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994aa30033e6d99aa1088634f476eab6235c>` : dump the full graph prior to partitioning.
		  
		  * :ref:`dnnl_graph_dump_mode_subgraph <doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994af9b1327c82bceda2e70fe33503608c73>` : dump each partitioned subgraph.
		  
		  * :ref:`dnnl_graph_dump_mode_none <doxid-group__dnnl__graph__api__dump__mode_1ggad9501cc148ae98ba5477d524b5149994aae63eb4551a7ce6fbb924dfe5e2222cb>` : disable all graph dumping.



.. rubric:: Returns:

:ref:`dnnl_invalid_arguments <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` if the ``modes`` value contains unsupported bits or graph dump is disabled, and :ref:`dnnl_success <doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>` on success.

