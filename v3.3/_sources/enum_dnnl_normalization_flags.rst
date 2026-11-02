.. index:: pair: enum; normalization_flags
.. _doxid-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be:

enum dnnl::normalization_flags
==============================

Overview
~~~~~~~~

Flags for normalization primitives. :ref:`More...<details-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl.hpp>

	enum normalization_flags
	{
	    :ref:`none<doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002bea334c4a4c42fdb79d7ebc3e73b517e6f8>`               = dnnl_normalization_flags_none,
	    :ref:`use_global_stats<doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002bea95768ff8afb8ee75dc24be0d307627f8>`   = dnnl_use_global_stats,
	    :ref:`use_scale<doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002beab989b02160429ba2696a658ec7a0f8e1>`          = dnnl_use_scale,
	    :ref:`use_shift<doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002beac5d8386f67a826c8ea1c1ae59a39586f>`          = dnnl_use_shift,
	    :ref:`fuse_norm_relu<doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002bea898ce555425ee54271096bc9c8e0400c>`     = dnnl_fuse_norm_relu,
	    :ref:`fuse_norm_add_relu<doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002bea6983328cc15d696e9f2756c8e8940370>` = dnnl_fuse_norm_add_relu,
	};

.. _details-group__dnnl__api__primitives__common_1gad8ef0fcbb7b10cae3d67dd46892002be:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Flags for normalization primitives.

Enum Values
-----------

.. index:: pair: enumvalue; none
.. _doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002bea334c4a4c42fdb79d7ebc3e73b517e6f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	none

Use no normalization flags.

If specified, the library computes mean and variance on forward propagation for training and inference, outputs them on forward propagation for training, and computes the respective derivatives on backward propagation.

.. index:: pair: enumvalue; use_global_stats
.. _doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002bea95768ff8afb8ee75dc24be0d307627f8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	use_global_stats

Use global statistics.

If specified, the library uses mean and variance provided by the user as an input on forward propagation and does not compute their derivatives on backward propagation. Otherwise, the library computes mean and variance on forward propagation for training and inference, outputs them on forward propagation for training, and computes the respective derivatives on backward propagation.

.. index:: pair: enumvalue; use_scale
.. _doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002beab989b02160429ba2696a658ec7a0f8e1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	use_scale

Use scale parameter.

If specified, the user is expected to pass scale as input on forward propagation. On backward propagation of type :ref:`dnnl::prop_kind::backward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa195fe59b6f103787a914aead0f3db502>`, the library computes its derivative.

.. index:: pair: enumvalue; use_shift
.. _doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002beac5d8386f67a826c8ea1c1ae59a39586f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	use_shift

Use shift parameter.

If specified, the user is expected to pass shift as input on forward propagation. On backward propagation of type :ref:`dnnl::prop_kind::backward <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa195fe59b6f103787a914aead0f3db502>`, the library computes its derivative.

.. index:: pair: enumvalue; fuse_norm_relu
.. _doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002bea898ce555425ee54271096bc9c8e0400c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	fuse_norm_relu

Fuse normalization with ReLU.

On training, normalization will require the workspace to implement backward propagation. On inference, the workspace is not required and behavior is the same as when normalization is fused with ReLU using the post-ops API.

.. index:: pair: enumvalue; fuse_norm_add_relu
.. _doxid-group__dnnl__api__primitives__common_1ggad8ef0fcbb7b10cae3d67dd46892002bea6983328cc15d696e9f2756c8e8940370:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	fuse_norm_add_relu

Fuse normalization with elementwise binary Add and then fuse with ReLU.

On training, normalization will require the workspace to implement backward propagation. On inference, the workspace is not required.

