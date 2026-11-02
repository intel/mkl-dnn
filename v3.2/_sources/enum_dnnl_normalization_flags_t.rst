.. index:: pair: enum; dnnl_normalization_flags_t
.. _doxid-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3:

enum dnnl_normalization_flags_t
===============================

Overview
~~~~~~~~

Flags for normalization primitives. :ref:`More...<details-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_types.h>

	enum dnnl_normalization_flags_t
	{
	    :ref:`dnnl_normalization_flags_none<doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3ab71f2077a94fd4bbc107a09b115a24a4>` = 0x0U,
	    :ref:`dnnl_use_global_stats<doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>`         = 0x1U,
	    :ref:`dnnl_use_scale<doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65>`                = 0x2U,
	    :ref:`dnnl_use_shift<doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0>`                = 0x4U,
	    :ref:`dnnl_fuse_norm_relu<doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a7150bdb66ef194e6ee11fbaa85a34ada>`           = 0x8U,
	    :ref:`dnnl_fuse_norm_add_relu<doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4>`       = 0x10U,
	};

.. _details-group__dnnl__api__primitives__common_1ga301f673522a400c7c1e75f518431c9a3:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Flags for normalization primitives.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_normalization_flags_none
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3ab71f2077a94fd4bbc107a09b115a24a4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_normalization_flags_none

Use no normalization flags.

If specified

* on forward training propagation mean and variance are computed and stored as output

* on backward propagation compute full derivative wrt data

* on backward propagation prop_kind == :ref:`dnnl_backward_data <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a524dd6cb2ed9680bbd170ba15261d218>` has the same behavior as prop_kind == :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`

.. index:: pair: enumvalue; dnnl_use_global_stats
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_use_global_stats

Use global statistics.

If specified

* on forward propagation use mean and variance provided by user (input)

* on backward propagation reduces the amount of computations, since mean and variance are considered as constants

If not specified:

* on forward propagation mean and variance are computed and stored as output

* on backward propagation compute full derivative wrt data

.. index:: pair: enumvalue; dnnl_use_scale
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_use_scale

Use scale parameter.

If specified:

* on forward propagation use scale for the normalization results

* on backward propagation (for prop_kind == :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`) compute diff wrt scale (hence one extra output used)

.. index:: pair: enumvalue; dnnl_use_shift
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_use_shift

Use shift parameter.

If specified:

* on forward propagation use shift (aka bias) for the normalization results

* on backward propagation (for prop_kind == :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`) compute diff wrt shift (hence one extra output used)

.. index:: pair: enumvalue; dnnl_fuse_norm_relu
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a7150bdb66ef194e6ee11fbaa85a34ada:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fuse_norm_relu

Fuse with ReLU.

The flag implies negative slope being 0. On training this is the only configuration supported. For inference, to use non-zero negative slope consider using :ref:`Primitive Attributes: Post-ops <doxid-dev_guide_attributes_post_ops>`.

If specified:

* on inference this option behaves the same as if the primitive were fused with ReLU using post ops API with zero negative slope.

* on training primitive requires workspace (required to be able to perform backward pass)

.. index:: pair: enumvalue; dnnl_fuse_norm_add_relu
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fuse_norm_add_relu

Fuse with Add and then fuse with ReLU.

If specified:

* on forward propagation apply element-wise binary Add operation to to the normalization results with an additional input tensor and then apply ReLU with negative slope being 0.

* on training primitive requires workspace (required to be able to perform backward pass).

* on backward propagation save the result of backward ReLU operation with input tensor and workspace from forward pass to extra output tensor and then perform backward normalization.

