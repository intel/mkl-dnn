.. index:: pair: enum; cpu_isa_hints
.. _doxid-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4:

enum dnnl::cpu_isa_hints
========================

Overview
~~~~~~~~

CPU ISA hints flags. :ref:`More...<details-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum cpu_isa_hints
	{
	    :ref:`no_hints<doxid-group__dnnl__api__service_1ggaf574021058ebc6965da14fc4387dd0c4a5c2d3f6f845dca6d90d7a1c445644c99>`   = dnnl_cpu_isa_no_hints,
	    :ref:`prefer_ymm<doxid-group__dnnl__api__service_1ggaf574021058ebc6965da14fc4387dd0c4ad5d95963017a7ba00e7ddc69b67cacb6>` = dnnl_cpu_isa_prefer_ymm,
	};

.. _details-group__dnnl__api__service_1gaf574021058ebc6965da14fc4387dd0c4:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

CPU ISA hints flags.

Enum Values
-----------

.. index:: pair: enumvalue; no_hints
.. _doxid-group__dnnl__api__service_1ggaf574021058ebc6965da14fc4387dd0c4a5c2d3f6f845dca6d90d7a1c445644c99:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	no_hints

No hints (use default features)

.. index:: pair: enumvalue; prefer_ymm
.. _doxid-group__dnnl__api__service_1ggaf574021058ebc6965da14fc4387dd0c4ad5d95963017a7ba00e7ddc69b67cacb6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	prefer_ymm

Prefer to exclusively use Ymm registers for computations.

