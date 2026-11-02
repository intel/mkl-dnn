.. index:: pair: enum; sparse_encoding
.. _doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966:

enum dnnl::memory::sparse_encoding
==================================

Overview
~~~~~~~~

Sparse encodings. :ref:`More...<details-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum sparse_encoding
	{
	    :ref:`undef<doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966af31ee5e3824f1f5e5d206bdf3029f22b>`  = dnnl_sparse_encoding_undef,
	    :ref:`csr<doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a1f8c50db95e9ead5645e32f8df5baa7b>`    = dnnl_csr,
	    :ref:`packed<doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966af59dcd306ec32930f1e78a1d82280b48>` = dnnl_packed,
	    :ref:`coo<doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a03a6ff0db560bbdbcd4c86cd94b35971>`    = dnnl_coo,
	};

.. _details-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Sparse encodings.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966af31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined sparse encoding kind, used for empty memory descriptors.

.. index:: pair: enumvalue; csr
.. _doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a1f8c50db95e9ead5645e32f8df5baa7b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	csr

Compressed Sparse Row (CSR) encoding.

.. index:: pair: enumvalue; packed
.. _doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966af59dcd306ec32930f1e78a1d82280b48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	packed

An encoding that is used for an opaque storage schema for tensors with unstructured sparsity.

A memory descriptor with the packed encoding cannot be used to create a memory object. It can only be used to create a primitive descriptor to query the actual memory descriptor (similar to the format tag ``any``).

.. index:: pair: enumvalue; coo
.. _doxid-structdnnl_1_1memory_1ab465a354090df7cc6d27cec0e037b966a03a6ff0db560bbdbcd4c86cd94b35971:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	coo

Coordinate Sparse (COO) encoding.

