.. index:: pair: enum; dnnl_stream_flags_t
.. _doxid-group__dnnl__api__stream_1ga3d74cfed8fe92b0e4498a1f2bdab5547:

enum dnnl_stream_flags_t
========================

Overview
~~~~~~~~

Stream flags. :ref:`More...<details-group__dnnl__api__stream_1ga3d74cfed8fe92b0e4498a1f2bdab5547>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common_types.h>

	enum dnnl_stream_flags_t
	{
	    :target:`dnnl_stream_in_order<doxid-group__dnnl__api__stream_1gga3d74cfed8fe92b0e4498a1f2bdab5547a4cb57d18d4769548e671aaff38a80478>`      = 0x1U,
	    :ref:`dnnl_stream_out_of_order<doxid-group__dnnl__api__stream_1gga3d74cfed8fe92b0e4498a1f2bdab5547a8e9e057b28a7898a9b52120b8c7bd356>`  = 0x2U,
	    :ref:`dnnl_stream_default_flags<doxid-group__dnnl__api__stream_1gga3d74cfed8fe92b0e4498a1f2bdab5547acf05c543bccebd58e6d4e0db7137fb92>` = dnnl_stream_in_order,
	};

.. _details-group__dnnl__api__stream_1ga3d74cfed8fe92b0e4498a1f2bdab5547:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Stream flags.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_stream_out_of_order
.. _doxid-group__dnnl__api__stream_1gga3d74cfed8fe92b0e4498a1f2bdab5547a8e9e057b28a7898a9b52120b8c7bd356:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_stream_out_of_order

Out-of-order execution.

.. index:: pair: enumvalue; dnnl_stream_default_flags
.. _doxid-group__dnnl__api__stream_1gga3d74cfed8fe92b0e4498a1f2bdab5547acf05c543bccebd58e6d4e0db7137fb92:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_stream_default_flags

Default stream configuration.

