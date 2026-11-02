.. index:: pair: enum; status
.. _doxid-group__dnnl__graph__api__status_1gadf0aa6dbce815494117ab7f44ffb598c:

enum dnnl::graph::status
========================

Overview
~~~~~~~~

Status values returned by the library functions. :ref:`More...<details-group__dnnl__graph__api__status_1gadf0aa6dbce815494117ab7f44ffb598c>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>

	enum status
	{
	    :ref:`success<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca260ca9dd8a4577fc00b7bd5810298076>`           = dnnl_success,
	    :ref:`out_of_memory<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca5ea42cc8dbd653d9afd96c1d2fd147e2>`     = dnnl_out_of_memory,
	    :ref:`invalid_arguments<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca242ac674d98ee2191f0bbf6de851d2d0>` = dnnl_invalid_arguments,
	    :ref:`unimplemented<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca4316423dfe3ade85c292aa38185f9817>`     = dnnl_unimplemented,
	    :ref:`last_impl_reached<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca00a7539e4d3cddaf78ae0cc2892cbb9b>` = dnnl_last_impl_reached,
	    :ref:`runtime_error<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca5b32065884bcc1f2ed126c47e6410808>`     = dnnl_runtime_error,
	    :ref:`not_required<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca20ab085a3c46af7db87fc1806865b329>`      = dnnl_not_required,
	    :ref:`invalid_graph<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca8b8d49961b84b28360a61571cb55cbfb>`     = dnnl_invalid_graph,
	    :ref:`invalid_graph_op<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598cab89b9125be83213081d93290688304b2>`  = dnnl_invalid_graph_op,
	    :ref:`invalid_shape<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca2dd90f82a71235e156a227f9349c1246>`     = dnnl_invalid_shape,
	    :ref:`invalid_data_type<doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598cae8c616c26029d0c1cce5d1be03f8095b>` = dnnl_invalid_data_type,
	};

.. _details-group__dnnl__graph__api__status_1gadf0aa6dbce815494117ab7f44ffb598c:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Status values returned by the library functions.

Enum Values
-----------

.. index:: pair: enumvalue; success
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca260ca9dd8a4577fc00b7bd5810298076:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	success

The operation was successful.

.. index:: pair: enumvalue; out_of_memory
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca5ea42cc8dbd653d9afd96c1d2fd147e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	out_of_memory

The operation failed due to an out-of-memory condition.

.. index:: pair: enumvalue; invalid_arguments
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca242ac674d98ee2191f0bbf6de851d2d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	invalid_arguments

The operation failed because of incorrect function arguments.

.. index:: pair: enumvalue; unimplemented
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca4316423dfe3ade85c292aa38185f9817:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unimplemented

The operation failed because requested functionality is not implemented.

.. index:: pair: enumvalue; last_impl_reached
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca00a7539e4d3cddaf78ae0cc2892cbb9b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	last_impl_reached

The last available implementation is reached.

.. index:: pair: enumvalue; runtime_error
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca5b32065884bcc1f2ed126c47e6410808:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	runtime_error

Primitive or engine failed on execution.

.. index:: pair: enumvalue; not_required
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca20ab085a3c46af7db87fc1806865b329:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	not_required

Queried element is not required for given primitive.

.. index:: pair: enumvalue; invalid_graph
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca8b8d49961b84b28360a61571cb55cbfb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	invalid_graph

The graph is not legitimate.

.. index:: pair: enumvalue; invalid_graph_op
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598cab89b9125be83213081d93290688304b2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	invalid_graph_op

The operation is not legitimate according to op schema.

.. index:: pair: enumvalue; invalid_shape
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598ca2dd90f82a71235e156a227f9349c1246:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	invalid_shape

The shape cannot be inferred or compiled.

.. index:: pair: enumvalue; invalid_data_type
.. _doxid-group__dnnl__graph__api__status_1ggadf0aa6dbce815494117ab7f44ffb598cae8c616c26029d0c1cce5d1be03f8095b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	invalid_data_type

The data type cannot be inferred or compiled.

