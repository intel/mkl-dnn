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
	    :ref:`dnnl_packed<doxid-group__dnnl__api__memory_1ggad5c084dc8593f175172318438996b552ae50fa2fb6b590dc9031aed7f7e59e7f3>`,
	    :ref:`dnnl_coo<doxid-group__dnnl__api__memory_1ggad5c084dc8593f175172318438996b552a6ed78ca42610783e4aa3c16b6427f9d1>`,
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

.. index:: pair: enumvalue; dnnl_packed
.. _doxid-group__dnnl__api__memory_1ggad5c084dc8593f175172318438996b552ae50fa2fb6b590dc9031aed7f7e59e7f3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_packed

An encoding that is used for an opaque storage schema for tensors with unstructured sparsity.

A memory descriptor with the packed encoding cannot be used to create a memory object. It can only be used to create a primitive descriptor to query the actual memory descriptor (similar to the format tag ``any``).

.. index:: pair: enumvalue; dnnl_coo
.. _doxid-group__dnnl__api__memory_1ggad5c084dc8593f175172318438996b552a6ed78ca42610783e4aa3c16b6427f9d1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_coo

Coordinate Sparse Encoding (COO).

