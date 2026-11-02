.. index:: pair: enum; prop_kind
.. _doxid-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f:

enum dnnl::prop_kind
====================

Overview
~~~~~~~~

Propagation kind. :ref:`More...<details-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum prop_kind
	{
	    :ref:`undef<doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b>`             = dnnl_prop_kind_undef,
	    :ref:`forward_training<doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`  = dnnl_forward_training,
	    :ref:`forward_inference<doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>` = dnnl_forward_inference,
	    :ref:`forward<doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8>`           = dnnl_forward,
	    :ref:`backward<doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa195fe59b6f103787a914aead0f3db502>`          = dnnl_backward,
	    :ref:`backward_data<doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faa12627cacf73ecb7ef088beedd650e96>`     = dnnl_backward_data,
	    :ref:`backward_weights<doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa1a002980f340e61153a9f7de4f794cf6>`  = dnnl_backward_weights,
	    :ref:`backward_bias<doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fab6b02795407dc7897e390e48f1d0ea02>`     = dnnl_backward_bias,
	};

.. _details-group__dnnl__api__attributes_1gac7db48f6583aa9903e54c2a39d65438f:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Propagation kind.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faf31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined propagation kind.

.. index:: pair: enumvalue; forward_training
.. _doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	forward_training

Forward data propagation (training mode).

In this mode, primitives perform computations necessary for subsequent backward propagation.

.. index:: pair: enumvalue; forward_inference
.. _doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	forward_inference

Forward data propagation (inference mode).

In this mode, primitives perform only computations that are necessary for inference and omit computations that are necessary only for backward propagation.

.. index:: pair: enumvalue; forward
.. _doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa965dbaac085fc891bfbbd4f9d145bbc8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	forward

Forward data propagation, alias for :ref:`dnnl::prop_kind::forward_training <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa24775787fab8f13aa4809e1ce8f82aeb>`.

.. index:: pair: enumvalue; backward
.. _doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa195fe59b6f103787a914aead0f3db502:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	backward

Backward propagation (with respect to all parameters).

.. index:: pair: enumvalue; backward_data
.. _doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438faa12627cacf73ecb7ef088beedd650e96:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	backward_data

Backward data propagation.

.. index:: pair: enumvalue; backward_weights
.. _doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa1a002980f340e61153a9f7de4f794cf6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	backward_weights

Backward weights propagation.

.. index:: pair: enumvalue; backward_bias
.. _doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fab6b02795407dc7897e390e48f1d0ea02:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	backward_bias

Backward bias propagation.

