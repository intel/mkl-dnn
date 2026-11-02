.. index:: pair: enum; format_kind
.. _doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105f:

enum dnnl::memory::format_kind
==============================

Overview
~~~~~~~~

Memory format kind. :ref:`More...<details-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105f>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum format_kind
	{
	    :ref:`undef<doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105faf31ee5e3824f1f5e5d206bdf3029f22b>`   = dnnl_format_kind_undef,
	    :ref:`any<doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa100b8cad7cf2a56f6df78f171f97a1ec>`     = dnnl_format_kind_any,
	    :ref:`blocked<doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa61326117ed4a9ddf3f754e71e119e5b3>` = dnnl_blocked,
	    :ref:`opaque<doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa94619f8a70068b2591c2eed622525b0e>`  = dnnl_format_kind_opaque,
	};

.. _details-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105f:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Memory format kind.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105faf31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined memory format kind, used for empty memory descriptors.

.. index:: pair: enumvalue; any
.. _doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa100b8cad7cf2a56f6df78f171f97a1ec:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	any

A special format kind that indicates that the actual format will be selected by a primitive automatically.

.. index:: pair: enumvalue; blocked
.. _doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa61326117ed4a9ddf3f754e71e119e5b3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	blocked

A tensor in a generic format described by the stride and blocking values in each dimension.

.. index:: pair: enumvalue; opaque
.. _doxid-structdnnl_1_1memory_1aabcadfb0e23a36a91272fc571cff105fa94619f8a70068b2591c2eed622525b0e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	opaque

A special format kind that indicates that tensor format is opaque.

