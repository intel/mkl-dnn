.. index:: pair: enum; memory_kind
.. _doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb:

enum dnnl::sycl_interop::memory_kind
====================================

Overview
~~~~~~~~

Memory allocation kind. :ref:`More...<details-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_sycl.hpp>

	enum memory_kind
	{
	    :ref:`usm<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba81e61a0cab904f0e620dd3226f7f6582>`    = dnnl_sycl_interop_usm,
	    :ref:`buffer<doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87>` = dnnl_sycl_interop_buffer,
	};

.. _details-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeb:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory allocation kind.

Enum Values
-----------

.. index:: pair: enumvalue; usm
.. _doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba81e61a0cab904f0e620dd3226f7f6582:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	usm

USM (device, shared, host, or unknown) memory allocation kind - default.

.. index:: pair: enumvalue; buffer
.. _doxid-namespacednnl_1_1sycl__interop_1a9c7def46b2c0556f56e2f0aab5fbffeba7f2db423a49b305459147332fb01cf87:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	buffer

Buffer memory allocation kind.

