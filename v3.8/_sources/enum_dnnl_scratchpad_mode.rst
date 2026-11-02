.. index:: pair: enum; scratchpad_mode
.. _doxid-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f:

enum dnnl::scratchpad_mode
==========================

Overview
~~~~~~~~

Scratchpad mode. :ref:`More...<details-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum scratchpad_mode
	{
	    :ref:`library<doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07fad521f765a49c72507257a2620612ee96>` = dnnl_scratchpad_mode_library,
	    :ref:`user<doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07faee11cbb19052e40b07aac0ca060c23ee>`    = dnnl_scratchpad_mode_user,
	};

.. _details-group__dnnl__api__attributes_1gac24d40ceea0256c7d6cc3a383a0fa07f:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Scratchpad mode.

Enum Values
-----------

.. index:: pair: enumvalue; library
.. _doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07fad521f765a49c72507257a2620612ee96:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	library

The library manages the scratchpad allocation according to the policy specified by the ``DNNL_ENABLE_CONCURRENT_EXEC`` :ref:`build option <doxid-dev_guide_build_options>` (default).

When ``DNNL_ENABLE_CONCURRENT_EXEC=OFF`` (default), the library scratchpad is common to all primitives to reduce the memory footprint. This configuration comes with limited thread-safety properties, namely primitives can be created and executed in parallel but cannot migrate between threads (in other words, each primitive should be executed in the same thread it was created in).

When ``DNNL_ENABLE_CONCURRENT_EXEC=ON``, the library scratchpad is private to each primitive. The memory footprint is larger than when using ``DNNL_ENABLE_CONCURRENT_EXEC=OFF`` but different primitives can be created and run concurrently (the same primitive cannot be run concurrently from two different threads though).

.. index:: pair: enumvalue; user
.. _doxid-group__dnnl__api__attributes_1ggac24d40ceea0256c7d6cc3a383a0fa07faee11cbb19052e40b07aac0ca060c23ee:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	user

The user manages the scratchpad allocation by querying and providing the scratchpad memory to primitives.

This mode is thread-safe as long as the scratchpad buffers are not used concurrently by two primitive executions.

