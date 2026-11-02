.. index:: pair: enum; dnnl_profiling_data_kind_t
.. _doxid-group__dnnl__api__memory_1ga7ac0b200fe8227f70d08832ffc9c51f4:

enum dnnl_profiling_data_kind_t
===============================

Overview
~~~~~~~~

Profiling data kind. :ref:`More...<details-group__dnnl__api__memory_1ga7ac0b200fe8227f70d08832ffc9c51f4>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_profiling_data_kind_t
	{
	    :ref:`dnnl_profiling_data_kind_undef<doxid-group__dnnl__api__memory_1gga7ac0b200fe8227f70d08832ffc9c51f4a811bd14118ff8c4ee3d52671e8fbea89>` = 0,
	    :ref:`dnnl_profiling_data_kind_time<doxid-group__dnnl__api__memory_1gga7ac0b200fe8227f70d08832ffc9c51f4a3251f520d667be9a179ecf857e8b3a3b>`,
	};

.. _details-group__dnnl__api__memory_1ga7ac0b200fe8227f70d08832ffc9c51f4:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Profiling data kind.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_profiling_data_kind_undef
.. _doxid-group__dnnl__api__memory_1gga7ac0b200fe8227f70d08832ffc9c51f4a811bd14118ff8c4ee3d52671e8fbea89:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_profiling_data_kind_undef

Undefined profiling data kind.

.. index:: pair: enumvalue; dnnl_profiling_data_kind_time
.. _doxid-group__dnnl__api__memory_1gga7ac0b200fe8227f70d08832ffc9c51f4a3251f520d667be9a179ecf857e8b3a3b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_profiling_data_kind_time

Data kind to query an execution time in nanoseconds.

