.. index:: pair: enum; dnnl_prop_kind_t
.. _doxid-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4:

enum dnnl_prop_kind_t
=====================

Overview
~~~~~~~~

Kinds of propagation. :ref:`More...<details-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_prop_kind_t
	{
	    :ref:`dnnl_prop_kind_undef<doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a68c8ec51f34458261cac7d12eee01372>`   = 0,
	    :ref:`dnnl_forward_training<doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>`  = 64,
	    :ref:`dnnl_forward_inference<doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` = 96,
	    :ref:`dnnl_forward<doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb>`           = dnnl_forward_training,
	    :ref:`dnnl_backward<doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`          = 128,
	    :ref:`dnnl_backward_data<doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a524dd6cb2ed9680bbd170ba15261d218>`     = 160,
	    :ref:`dnnl_backward_weights<doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a28ab91d299557b64b2f844548b6504bd>`  = 192,
	    :ref:`dnnl_backward_bias<doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4aa15193604f51a9b42ecff7469bf51cc2>`     = 193,
	};

.. _details-group__dnnl__api__primitives__common_1gae3c1f22ae55645782923fbfd8b07d0c4:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Kinds of propagation.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_prop_kind_undef
.. _doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a68c8ec51f34458261cac7d12eee01372:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_prop_kind_undef

Undefined propagation type.

.. index:: pair: enumvalue; dnnl_forward_training
.. _doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_forward_training

Forward data propagation (training mode).

In this mode primitives perform computations necessary for subsequent backward propagation.

.. index:: pair: enumvalue; dnnl_forward_inference
.. _doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_forward_inference

Forward data propagation (inference mode).

In this mode primitives perform only computations that are necessary for inference and omit computations that are necessary only for backward propagation.

.. index:: pair: enumvalue; dnnl_forward
.. _doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a6a59d07a8414bb69b3cb9904ed302adb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_forward

Forward data propagation (alias for ``dnnl_forward_training``).

.. index:: pair: enumvalue; dnnl_backward
.. _doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_backward

Backward propagation (with respect to all parameters).

.. index:: pair: enumvalue; dnnl_backward_data
.. _doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a524dd6cb2ed9680bbd170ba15261d218:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_backward_data

Backward data propagation.

.. index:: pair: enumvalue; dnnl_backward_weights
.. _doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a28ab91d299557b64b2f844548b6504bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_backward_weights

Backward weights propagation.

.. index:: pair: enumvalue; dnnl_backward_bias
.. _doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4aa15193604f51a9b42ecff7469bf51cc2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_backward_bias

Backward bias propagation.

