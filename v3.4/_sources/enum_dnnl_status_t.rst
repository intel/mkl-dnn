.. index:: pair: enum; dnnl_status_t
.. _doxid-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a:

enum dnnl_status_t
==================

Overview
~~~~~~~~

Status values returned by the library functions. :ref:`More...<details-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_common_types.h>

	enum dnnl_status_t
	{
	    :ref:`dnnl_success<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c>`           = 0,
	    :ref:`dnnl_out_of_memory<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa45f782c74a21417c4266939a79b404e0>`     = 1,
	    :ref:`dnnl_invalid_arguments<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c>` = 2,
	    :ref:`dnnl_unimplemented<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1>`     = 3,
	    :ref:`dnnl_last_impl_reached<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa2c6653ea2885f9dbafdc0bf2ee8693f8>` = 4,
	    :ref:`dnnl_runtime_error<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa38efb4adabcae7c9e6479e8ee1242b9b>`     = 5,
	    :ref:`dnnl_not_required<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaff3988320148106126bce50dd76d6a97>`      = 6,
	    :ref:`dnnl_invalid_graph<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa422aad40b808cb81e38d3e7baa6e78b>`     = 7,
	    :ref:`dnnl_invalid_graph_op<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaae30b860726877d791242920642b05d0>`  = 8,
	    :ref:`dnnl_invalid_shape<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa7e82bd83ca24e2eedce6ef4d1e6db0ae>`     = 9,
	    :ref:`dnnl_invalid_data_type<doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa9ef42fb91374d80fcbc9d823fd7771ac>` = 10,
	};

.. _details-group__dnnl__api__utils_1gad24f9ded06e34d3ee71e7fc4b408d57a:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Status values returned by the library functions.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_success
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa31395e9dccc103cf166cf7b38fc5b9c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_success

The operation was successful.

.. index:: pair: enumvalue; dnnl_out_of_memory
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa45f782c74a21417c4266939a79b404e0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_out_of_memory

The operation failed due to an out-of-memory condition.

.. index:: pair: enumvalue; dnnl_invalid_arguments
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaecec97c787d74a33924abcf16ae4f51c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_invalid_arguments

The operation failed because of incorrect function arguments.

.. index:: pair: enumvalue; dnnl_unimplemented
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa3a8579e8afc4e23344cd3115b0e81de1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_unimplemented

The operation failed because requested functionality is not implemented.

.. index:: pair: enumvalue; dnnl_last_impl_reached
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa2c6653ea2885f9dbafdc0bf2ee8693f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_last_impl_reached

The last available implementation is reached.

.. index:: pair: enumvalue; dnnl_runtime_error
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa38efb4adabcae7c9e6479e8ee1242b9b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_runtime_error

Primitive or engine failed on execution.

.. index:: pair: enumvalue; dnnl_not_required
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaff3988320148106126bce50dd76d6a97:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_not_required

Queried element is not required for given primitive.

.. index:: pair: enumvalue; dnnl_invalid_graph
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaa422aad40b808cb81e38d3e7baa6e78b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_invalid_graph

The graph is not legitimate.

.. index:: pair: enumvalue; dnnl_invalid_graph_op
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aaae30b860726877d791242920642b05d0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_invalid_graph_op

The operation is not legitimate according to op schema.

.. index:: pair: enumvalue; dnnl_invalid_shape
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa7e82bd83ca24e2eedce6ef4d1e6db0ae:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_invalid_shape

The shape cannot be inferred or compiled.

.. index:: pair: enumvalue; dnnl_invalid_data_type
.. _doxid-group__dnnl__api__utils_1ggad24f9ded06e34d3ee71e7fc4b408d57aa9ef42fb91374d80fcbc9d823fd7771ac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_invalid_data_type

The data type cannot be inferred or compiled.

