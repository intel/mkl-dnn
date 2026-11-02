.. index:: pair: enum; dnnl_scratchpad_mode_t
.. _doxid-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4:

enum dnnl_scratchpad_mode_t
===========================

Overview
~~~~~~~~

Scratchpad mode. :ref:`More...<details-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_scratchpad_mode_t
	{
	    :ref:`dnnl_scratchpad_mode_library<doxid-group__dnnl__api__attributes_1ggacda323181ab267e571c31435b0817de4ac6aab09a2f8ef442a6a59800549b0487>`,
	    :ref:`dnnl_scratchpad_mode_user<doxid-group__dnnl__api__attributes_1ggacda323181ab267e571c31435b0817de4a7e9d97b9ceefc5e47512d83c097d6927>`,
	};

.. _details-group__dnnl__api__attributes_1gacda323181ab267e571c31435b0817de4:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Scratchpad mode.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_scratchpad_mode_library
.. _doxid-group__dnnl__api__attributes_1ggacda323181ab267e571c31435b0817de4ac6aab09a2f8ef442a6a59800549b0487:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_scratchpad_mode_library

The library manages the scratchpad allocation according to the policy specified by the ``DNNL_ENABLE_CONCURRENT_EXEC`` :ref:`build option <doxid-dev_guide_build_options>` (default).

When ``DNNL_ENABLE_CONCURRENT_EXEC=OFF`` (default), the library scratchpad is common to all primitives to reduce the memory footprint. This configuration comes with limited thread-safety properties, namely primitives can be created and executed in parallel but cannot migrate between threads (in other words, each primitive should be executed in the same thread it was created in).

When ``DNNL_ENABLE_CONCURRENT_EXEC=ON``, the library scratchpad is private to each primitive. The memory footprint is larger than when using ``DNNL_ENABLE_CONCURRENT_EXEC=OFF`` but different primitives can be created and run concurrently (the same primitive cannot be run concurrently from two different threads though).

.. index:: pair: enumvalue; dnnl_scratchpad_mode_user
.. _doxid-group__dnnl__api__attributes_1ggacda323181ab267e571c31435b0817de4a7e9d97b9ceefc5e47512d83c097d6927:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_scratchpad_mode_user

The user manages the scratchpad allocation by querying and providing the scratchpad memory to primitives.

This mode is thread-safe as long as the scratchpad buffers are not used concurrently by two primitive executions.

