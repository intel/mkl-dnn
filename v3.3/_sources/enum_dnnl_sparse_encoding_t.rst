.. index:: pair: enum; dnnl_sparse_encoding_t
.. _doxid-group__dnnl__api__memory_1gad5c084dc8593f175172318438996b552:

enum dnnl_sparse_encoding_t
===========================

Overview
~~~~~~~~

Sparse encodings. :ref:`More...<details-group__dnnl__api__memory_1gad5c084dc8593f175172318438996b552>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_sparse_encoding_t
	{
	    :ref:`dnnl_sparse_encoding_undef<doxid-group__dnnl__api__memory_1ggad5c084dc8593f175172318438996b552a7064b7ea7ccf34a1e93e9c95e7e5d883>` = 0,
	    :ref:`dnnl_csr<doxid-group__dnnl__api__memory_1ggad5c084dc8593f175172318438996b552af28be8ee8fe86aec95798fe1b6106aac>`,
	};

.. _details-group__dnnl__api__memory_1gad5c084dc8593f175172318438996b552:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Sparse encodings.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_sparse_encoding_undef
.. _doxid-group__dnnl__api__memory_1ggad5c084dc8593f175172318438996b552a7064b7ea7ccf34a1e93e9c95e7e5d883:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_sparse_encoding_undef

Undefined sparse encoding kind, used for empty memory descriptors.

.. index:: pair: enumvalue; dnnl_csr
.. _doxid-group__dnnl__api__memory_1ggad5c084dc8593f175172318438996b552af28be8ee8fe86aec95798fe1b6106aac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_csr

Compressed Sparse Row (CSR) encoding.

