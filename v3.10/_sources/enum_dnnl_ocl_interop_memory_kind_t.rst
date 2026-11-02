.. index:: pair: enum; dnnl_ocl_interop_memory_kind_t
.. _doxid-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16:

enum dnnl_ocl_interop_memory_kind_t
===================================

Overview
~~~~~~~~

Memory allocation kind. :ref:`More...<details-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_ocl_types.h>

	enum dnnl_ocl_interop_memory_kind_t
	{
	    :ref:`dnnl_ocl_interop_usm<doxid-group__dnnl__api__ocl__interop_1gga410bffb44ad08e8d2628711e5ea16d16a8a7f817075cadd9d5d5053302026aeac>`,
	    :ref:`dnnl_ocl_interop_buffer<doxid-group__dnnl__api__ocl__interop_1gga410bffb44ad08e8d2628711e5ea16d16aca21719d5bb5f5e989c89bd0f8e450c6>`,
	};

.. _details-group__dnnl__api__ocl__interop_1ga410bffb44ad08e8d2628711e5ea16d16:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory allocation kind.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_ocl_interop_usm
.. _doxid-group__dnnl__api__ocl__interop_1gga410bffb44ad08e8d2628711e5ea16d16a8a7f817075cadd9d5d5053302026aeac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ocl_interop_usm

USM (device, shared, host, or unknown) memory allocation kind.

.. index:: pair: enumvalue; dnnl_ocl_interop_buffer
.. _doxid-group__dnnl__api__ocl__interop_1gga410bffb44ad08e8d2628711e5ea16d16aca21719d5bb5f5e989c89bd0f8e450c6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_ocl_interop_buffer

Buffer memory allocation kind - default.

