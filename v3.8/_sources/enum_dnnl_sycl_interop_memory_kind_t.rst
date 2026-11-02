.. index:: pair: enum; dnnl_sycl_interop_memory_kind_t
.. _doxid-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c:

enum dnnl_sycl_interop_memory_kind_t
====================================

Overview
~~~~~~~~

Memory allocation kind. :ref:`More...<details-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_sycl_types.h>

	enum dnnl_sycl_interop_memory_kind_t
	{
	    :ref:`dnnl_sycl_interop_usm<doxid-group__dnnl__api__sycl__interop_1gga8315f93ce0f395f59420094f3456b96caaabe6c0b7c6796f5fe800d455d65e05f>`,
	    :ref:`dnnl_sycl_interop_buffer<doxid-group__dnnl__api__sycl__interop_1gga8315f93ce0f395f59420094f3456b96ca08ce10b9e333a0c8f2f9463769c868ed>`,
	};

.. _details-group__dnnl__api__sycl__interop_1ga8315f93ce0f395f59420094f3456b96c:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory allocation kind.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_sycl_interop_usm
.. _doxid-group__dnnl__api__sycl__interop_1gga8315f93ce0f395f59420094f3456b96caaabe6c0b7c6796f5fe800d455d65e05f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_sycl_interop_usm

USM (device, shared, host, or unknown) memory allocation kind - default.

.. index:: pair: enumvalue; dnnl_sycl_interop_buffer
.. _doxid-group__dnnl__api__sycl__interop_1gga8315f93ce0f395f59420094f3456b96ca08ce10b9e333a0c8f2f9463769c868ed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_sycl_interop_buffer

Buffer memory allocation kind.

