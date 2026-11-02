.. index:: pair: enum; rnn_flags
.. _doxid-group__dnnl__api__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e:

enum dnnl::rnn_flags
====================

Overview
~~~~~~~~

RNN cell flags. :ref:`More...<details-group__dnnl__api__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum rnn_flags
	{
	    :ref:`undef<doxid-group__dnnl__api__rnn_1ggad27d0db2a86ae3072207769f5c2ddd1eaf31ee5e3824f1f5e5d206bdf3029f22b>`                  = dnnl_rnn_flags_undef,
	    :ref:`diff_weights_overwrite<doxid-group__dnnl__api__rnn_1ggad27d0db2a86ae3072207769f5c2ddd1ea45d36496fd68402e5800e09197fd04a6>` = dnnl_rnn_flags_diff_weights_overwrite,
	};

.. _details-group__dnnl__api__rnn_1gad27d0db2a86ae3072207769f5c2ddd1e:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

RNN cell flags.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-group__dnnl__api__rnn_1ggad27d0db2a86ae3072207769f5c2ddd1eaf31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined RNN flags.

.. index:: pair: enumvalue; diff_weights_overwrite
.. _doxid-group__dnnl__api__rnn_1ggad27d0db2a86ae3072207769f5c2ddd1ea45d36496fd68402e5800e09197fd04a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	diff_weights_overwrite

Do not add weights gradient to existing diff_weights memory.

