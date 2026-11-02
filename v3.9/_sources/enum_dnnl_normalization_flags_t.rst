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
	    :ref:`dnnl_rms_norm<doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a8c7e8830c1320a1db61e7634c29a9a60>`                 = 0x20U,
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

If specified, the library computes mean and variance on forward propagation for training and inference, outputs them on forward propagation for training, and computes the respective derivatives on backward propagation.

.. note:: 

   Backward propagation of type prop_kind == :ref:`dnnl_backward_data <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a524dd6cb2ed9680bbd170ba15261d218>` has the same behavior as prop_kind == :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`.

.. index:: pair: enumvalue; dnnl_use_global_stats
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_use_global_stats

Use global statistics.

If specified, the library uses mean and variance provided by the user as an input on forward propagation and does not compute their derivatives on backward propagation. Otherwise, the library computes mean and variance on forward propagation for training and inference, outputs them on forward propagation for training, and computes the respective derivatives on backward propagation.

.. index:: pair: enumvalue; dnnl_use_scale
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a01bf8edab9d40fd6a1f8827ee485dc65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_use_scale

Use scale parameter.

If specified, the user is expected to pass scale as input on forward propagation. On backward propagation of type :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`, the library computes its derivative.

.. index:: pair: enumvalue; dnnl_use_shift
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3afeb8455811d27d7835503a3740679df0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_use_shift

Use shift parameter.

If specified, the user is expected to pass shift as input on forward propagation. On backward propagation of type :ref:`dnnl_backward <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a326a5e31769302972e7bded555e1cc10>`, the library computes its derivative.

.. index:: pair: enumvalue; dnnl_fuse_norm_relu
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a7150bdb66ef194e6ee11fbaa85a34ada:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fuse_norm_relu

Fuse normalization with ReLU.

On training, normalization will require the workspace to implement backward propagation. On inference, the workspace is not required and behavior is the same as when normalization is fused with ReLU using the post-ops API.

.. note:: 

   The flag implies negative slope being 0. On training this is the only configuration supported. For inference, to use non-zero negative slope consider using :ref:`Primitive Attributes: Post-ops <doxid-dev_guide_attributes_post_ops>`.

.. index:: pair: enumvalue; dnnl_fuse_norm_add_relu
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3af324d9603806aae4ca3044e1e25534b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_fuse_norm_add_relu

Fuse normalization with an elementwise binary Add operation followed by ReLU.

During training, normalization will require a workspace to implement backward propagation. For inference, the workspace is not needed. On forward propagation, an elementwise binary Add operation is applied to the normalization results with an additional input tensor, followed by ReLU with a negative slope of 0. On backward propagation, the result of the backward ReLU operation with the input tensor and workspace from the forward pass is saved to an extra output tensor, and backward normalization is performed.

.. index:: pair: enumvalue; dnnl_rms_norm
.. _doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3a8c7e8830c1320a1db61e7634c29a9a60:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_rms_norm

Use Root Mean Square (RMS) Normalization.

In forward propagation, the mean is considered zero, and RMS norm is used instead of variance for scaling. Only the RMS norm is output during forward propagation for training. In backward propagation, the library calculates the derivative with respect to the RMS norm only, assuming the mean is zero.

.. note:: 

   When used with :ref:`dnnl_use_global_stats <doxid-group__dnnl__api__primitives__common_1gga301f673522a400c7c1e75f518431c9a3aec04425c28af752c0f8b4dc5ae11fb19>`, only RMS norm is required to be provided as input.

