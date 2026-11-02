.. index:: pair: enum; dnnl_format_kind_t
.. _doxid-group__dnnl__api__memory_1gaa75cad747fa467d9dc527d943ba3367d:

enum dnnl_format_kind_t
=======================

Overview
~~~~~~~~

Memory format kind. :ref:`More...<details-group__dnnl__api__memory_1gaa75cad747fa467d9dc527d943ba3367d>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_format_kind_t
	{
	    :ref:`dnnl_format_kind_undef<doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367dac86d377bba856ea7aa9679ecf65c8364>`  = 0,
	    :ref:`dnnl_format_kind_any<doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83>`,
	    :ref:`dnnl_blocked<doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16>`,
	    :ref:`dnnl_format_kind_opaque<doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da44f131bbbd690fd1f4f94b47279657fe>`,
	    :ref:`dnnl_format_kind_sparse<doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da7da1e739fcafae789d5a031c653de219>`,
	    :ref:`dnnl_format_kind_max<doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da4fd2fa6aae763e75f1d128a04c5bdafb>`    = 0x7fff,
	};

.. _details-group__dnnl__api__memory_1gaa75cad747fa467d9dc527d943ba3367d:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory format kind.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_format_kind_undef
.. _doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367dac86d377bba856ea7aa9679ecf65c8364:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_format_kind_undef

Undefined memory format kind, used for empty memory descriptors.

.. index:: pair: enumvalue; dnnl_format_kind_any
.. _doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da77ae35388e04dc3e98d90675a7110c83:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_format_kind_any

A special format kind that indicates that the actual format will be selected by a primitive automatically.

.. index:: pair: enumvalue; dnnl_blocked
.. _doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da30498f5adbc7d8017979a2201725ff16:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_blocked

A tensor in a generic format described by the stride and blocking values in each dimension.

.. index:: pair: enumvalue; dnnl_format_kind_opaque
.. _doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da44f131bbbd690fd1f4f94b47279657fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_format_kind_opaque

A special format kind that indicates that tensor format is opaque.

.. index:: pair: enumvalue; dnnl_format_kind_sparse
.. _doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da7da1e739fcafae789d5a031c653de219:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_format_kind_sparse

Format kind for sparse tensors.

.. index:: pair: enumvalue; dnnl_format_kind_max
.. _doxid-group__dnnl__api__memory_1ggaa75cad747fa467d9dc527d943ba3367da4fd2fa6aae763e75f1d128a04c5bdafb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_format_kind_max

Parameter to allow internal only format kinds without undefined behavior.

This parameter is chosen to be valid for so long as sizeof(int) >= 2.

