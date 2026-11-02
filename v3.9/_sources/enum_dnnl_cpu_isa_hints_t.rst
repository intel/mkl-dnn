.. index:: pair: enum; dnnl_cpu_isa_hints_t
.. _doxid-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d:

enum dnnl_cpu_isa_hints_t
=========================

Overview
~~~~~~~~

CPU ISA hints flags. :ref:`More...<details-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_cpu_isa_hints_t
	{
	    :ref:`dnnl_cpu_isa_no_hints<doxid-group__dnnl__api__service_1ggaf356412d94e35579bd509ed1fa174f5da9e598ac27ce94827b20cab264d623da4>`   = 0x0,
	    :ref:`dnnl_cpu_isa_prefer_ymm<doxid-group__dnnl__api__service_1ggaf356412d94e35579bd509ed1fa174f5daf9dd6f8367a4de1e12aa617307edbe41>` = 0x1,
	};

.. _details-group__dnnl__api__service_1gaf356412d94e35579bd509ed1fa174f5d:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

CPU ISA hints flags.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_cpu_isa_no_hints
.. _doxid-group__dnnl__api__service_1ggaf356412d94e35579bd509ed1fa174f5da9e598ac27ce94827b20cab264d623da4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_no_hints

No hints (use default features)

.. index:: pair: enumvalue; dnnl_cpu_isa_prefer_ymm
.. _doxid-group__dnnl__api__service_1ggaf356412d94e35579bd509ed1fa174f5daf9dd6f8367a4de1e12aa617307edbe41:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_cpu_isa_prefer_ymm

Prefer to exclusively use Ymm registers for computations.

