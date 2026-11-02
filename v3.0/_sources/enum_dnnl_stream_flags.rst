.. index:: pair: enum; flags
.. _doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59:

enum dnnl::stream::flags
========================

Overview
~~~~~~~~

Stream flags. Can be combined using the bitwise OR operator. :ref:`More...<details-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common.hpp>

	enum flags
	{
	    :ref:`in_order<doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59af51b25ca6f591d130cd0b575bf7821b3>`      = dnnl_stream_in_order,
	    :ref:`out_of_order<doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59a6f68b64d0cb895344cb033c850457f0b>`  = dnnl_stream_out_of_order,
	    :ref:`default_flags<doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59aeffb2d149f637ca450767d77cd927108>` = dnnl_stream_default_flags,
	};

.. _details-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Stream flags. Can be combined using the bitwise OR operator.

Enum Values
-----------

.. index:: pair: enumvalue; in_order
.. _doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59af51b25ca6f591d130cd0b575bf7821b3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	in_order

In-order execution.

.. index:: pair: enumvalue; out_of_order
.. _doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59a6f68b64d0cb895344cb033c850457f0b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	out_of_order

Out-of-order execution.

.. index:: pair: enumvalue; default_flags
.. _doxid-structdnnl_1_1stream_1abc7ec7dfa1718f366abd8f495164de59aeffb2d149f637ca450767d77cd927108:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	default_flags

Default stream configuration.

