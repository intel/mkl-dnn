.. index:: pair: enum; pack_type
.. _doxid-group__dnnl__api__ukernel__utils_1ga241c23d0afdf43a79d51ef701a9f7c54:

enum dnnl::ukernel::pack_type
=============================

Overview
~~~~~~~~

Packing specification. :ref:`More...<details-group__dnnl__api__ukernel__utils_1ga241c23d0afdf43a79d51ef701a9f7c54>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_ukernel.hpp>

	enum pack_type
	{
	    :ref:`undef<doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54af31ee5e3824f1f5e5d206bdf3029f22b>`    = dnnl_pack_type_undef,
	    :ref:`no_trans<doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54a76659c0424cb9f2555bc14e7d947db13>` = dnnl_pack_type_no_trans,
	    :ref:`trans<doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54a4738019ef434f24099319565cd5185e5>`    = dnnl_pack_type_trans,
	    :ref:`pack32<doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54a120dce00fbef2144bdd023da3aecaa6b>`   = dnnl_pack_type_pack32,
	};

.. _details-group__dnnl__api__ukernel__utils_1ga241c23d0afdf43a79d51ef701a9f7c54:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Packing specification.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54af31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined pack type. A guard value.

.. index:: pair: enumvalue; no_trans
.. _doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54a76659c0424cb9f2555bc14e7d947db13:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	no_trans

Plain, not transposed layout. Similar to format_tag::ab.

.. index:: pair: enumvalue; trans
.. _doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54a4738019ef434f24099319565cd5185e5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	trans

Plain, transposed layout. Similar to format_tag::ba.

.. index:: pair: enumvalue; pack32
.. _doxid-group__dnnl__api__ukernel__utils_1gga241c23d0afdf43a79d51ef701a9f7c54a120dce00fbef2144bdd023da3aecaa6b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	pack32

Packed by 32 bits along K dimension layout.

