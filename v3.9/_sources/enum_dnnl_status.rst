.. index:: pair: enum; status
.. _doxid-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd:

enum dnnl::status
=================

Overview
~~~~~~~~

Status values returned by the library functions. :ref:`More...<details-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum status
	{
	    :ref:`success<doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076>`           = dnnl_success,
	    :ref:`out_of_memory<doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5ea42cc8dbd653d9afd96c1d2fd147e2>`     = dnnl_out_of_memory,
	    :ref:`invalid_arguments<doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0>` = dnnl_invalid_arguments,
	    :ref:`unimplemented<doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817>`     = dnnl_unimplemented,
	    :ref:`last_impl_reached<doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda00a7539e4d3cddaf78ae0cc2892cbb9b>` = dnnl_last_impl_reached,
	    :ref:`runtime_error<doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808>`     = dnnl_runtime_error,
	    :ref:`not_required<doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda20ab085a3c46af7db87fc1806865b329>`      = dnnl_not_required,
	};

.. _details-group__dnnl__api__service_1ga7acc4d3516304ae68a1289551d8f2cdd:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Status values returned by the library functions.

Enum Values
-----------

.. index:: pair: enumvalue; success
.. _doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda260ca9dd8a4577fc00b7bd5810298076:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	success

The operation was successful.

.. index:: pair: enumvalue; out_of_memory
.. _doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5ea42cc8dbd653d9afd96c1d2fd147e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	out_of_memory

The operation failed due to an out-of-memory condition.

.. index:: pair: enumvalue; invalid_arguments
.. _doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda242ac674d98ee2191f0bbf6de851d2d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	invalid_arguments

The operation failed because of incorrect function arguments.

.. index:: pair: enumvalue; unimplemented
.. _doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda4316423dfe3ade85c292aa38185f9817:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unimplemented

The operation failed because requested functionality is not implemented.

.. index:: pair: enumvalue; last_impl_reached
.. _doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda00a7539e4d3cddaf78ae0cc2892cbb9b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	last_impl_reached

The last available implementation is reached.

.. index:: pair: enumvalue; runtime_error
.. _doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda5b32065884bcc1f2ed126c47e6410808:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	runtime_error

Primitive or engine failed on execution.

.. index:: pair: enumvalue; not_required
.. _doxid-group__dnnl__api__service_1gga7acc4d3516304ae68a1289551d8f2cdda20ab085a3c46af7db87fc1806865b329:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	not_required

Queried element is not required for given primitive.

