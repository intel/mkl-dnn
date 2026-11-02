.. index:: pair: enum; memory_kind
.. _doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481:

enum dnnl::ocl_interop::memory_kind
===================================

Overview
~~~~~~~~

Memory allocation kind. :ref:`More...<details-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_ocl.hpp>

	enum memory_kind
	{
	    :ref:`usm<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a81e61a0cab904f0e620dd3226f7f6582>`    = dnnl_ocl_interop_usm,
	    :ref:`buffer<doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a7f2db423a49b305459147332fb01cf87>` = dnnl_ocl_interop_buffer,
	};

.. _details-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory allocation kind.

Enum Values
-----------

.. index:: pair: enumvalue; usm
.. _doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a81e61a0cab904f0e620dd3226f7f6582:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	usm

USM (device, shared, host, or unknown) memory allocation kind.

.. index:: pair: enumvalue; buffer
.. _doxid-namespacednnl_1_1ocl__interop_1a8a53a7aed8cf616ebdf09e2bd7912481a7f2db423a49b305459147332fb01cf87:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	buffer

Buffer memory allocation kind - default.

