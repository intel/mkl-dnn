.. index:: pair: enum; profiling_data_kind
.. _doxid-group__dnnl__api__profiling_1gab19f8c7379c446429c9a4b043d64b4aa:

enum dnnl::profiling_data_kind
==============================

Overview
~~~~~~~~

Profiling data kind. :ref:`More...<details-group__dnnl__api__profiling_1gab19f8c7379c446429c9a4b043d64b4aa>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum profiling_data_kind
	{
	    :ref:`undef<doxid-group__dnnl__api__profiling_1ggab19f8c7379c446429c9a4b043d64b4aaaf31ee5e3824f1f5e5d206bdf3029f22b>` = dnnl_profiling_data_kind_undef,
	    :ref:`time<doxid-group__dnnl__api__profiling_1ggab19f8c7379c446429c9a4b043d64b4aaa07cc694b9b3fc636710fa08b6922c42b>`  = dnnl_profiling_data_kind_time,
	};

.. _details-group__dnnl__api__profiling_1gab19f8c7379c446429c9a4b043d64b4aa:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Profiling data kind.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-group__dnnl__api__profiling_1ggab19f8c7379c446429c9a4b043d64b4aaaf31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined profiling data kind.

.. index:: pair: enumvalue; time
.. _doxid-group__dnnl__api__profiling_1ggab19f8c7379c446429c9a4b043d64b4aaa07cc694b9b3fc636710fa08b6922c42b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	time

Data kind to query an execution time in nanoseconds.

