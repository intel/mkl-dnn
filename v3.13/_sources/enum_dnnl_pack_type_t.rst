.. index:: pair: enum; dnnl_pack_type_t
.. _doxid-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97:

enum dnnl_pack_type_t
=====================

Overview
~~~~~~~~

Packing specification. :ref:`More...<details-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_ukernel_types.h>

	enum dnnl_pack_type_t
	{
	    :ref:`dnnl_pack_type_undef<doxid-group__dnnl__api__ukernel_1ggae3d5cfb974745e876830f87c3315ec97a755e1c0735579a218ff60fb0b383544f>`    = 0,
	    :ref:`dnnl_pack_type_no_trans<doxid-group__dnnl__api__ukernel_1ggae3d5cfb974745e876830f87c3315ec97ab40a1ac713940990711e6080082efcc4>`,
	    :ref:`dnnl_pack_type_trans<doxid-group__dnnl__api__ukernel_1ggae3d5cfb974745e876830f87c3315ec97a17ca6ba9a049e1b6b5a6091d3dcffba7>`,
	    :ref:`dnnl_pack_type_pack32<doxid-group__dnnl__api__ukernel_1ggae3d5cfb974745e876830f87c3315ec97ae8d9c265f7df64579b17b78fe156d195>`,
	};

.. _details-group__dnnl__api__ukernel_1gae3d5cfb974745e876830f87c3315ec97:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Packing specification.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_pack_type_undef
.. _doxid-group__dnnl__api__ukernel_1ggae3d5cfb974745e876830f87c3315ec97a755e1c0735579a218ff60fb0b383544f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_pack_type_undef

Undefined pack type. A guard value.

.. index:: pair: enumvalue; dnnl_pack_type_no_trans
.. _doxid-group__dnnl__api__ukernel_1ggae3d5cfb974745e876830f87c3315ec97ab40a1ac713940990711e6080082efcc4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_pack_type_no_trans

Plain, not transposed layout. Similar to format_tag::ab.

.. index:: pair: enumvalue; dnnl_pack_type_trans
.. _doxid-group__dnnl__api__ukernel_1ggae3d5cfb974745e876830f87c3315ec97a17ca6ba9a049e1b6b5a6091d3dcffba7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_pack_type_trans

Plain, transposed layout. Similar to format_tag::ba.

.. index:: pair: enumvalue; dnnl_pack_type_pack32
.. _doxid-group__dnnl__api__ukernel_1ggae3d5cfb974745e876830f87c3315ec97ae8d9c265f7df64579b17b78fe156d195:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_pack_type_pack32

Packed by 32 bits along K dimension layout.

