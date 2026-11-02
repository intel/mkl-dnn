.. index:: pair: enum; dnnl_graph_op_attr_t
.. _doxid-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832:

enum dnnl_graph_op_attr_t
=========================

Overview
~~~~~~~~

Attributes of operations. :ref:`More...<details-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph_types.h>

	enum dnnl_graph_op_attr_t
	{
	    :ref:`dnnl_graph_op_attr_undef<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832af29d647b9ab14f52143eab725f598881>`                          = 0,
	    :ref:`dnnl_graph_op_attr_alpha<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a23f876973125c172c40e166503fcd380>`                          = 0x1,
	    :ref:`dnnl_graph_op_attr_beta<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a90dd6c02f9974b4ab68dabc17f6be576>`,
	    :ref:`dnnl_graph_op_attr_epsilon<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ae648365c7c28fadc330a852be9e66dca>`,
	    :ref:`dnnl_graph_op_attr_max<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a3031c0dcbbb8fdef7f2e6a8b6013fbe3>`,
	    :ref:`dnnl_graph_op_attr_min<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a0e5c8cf627d7cf34929ee3333db69180>`,
	    :ref:`dnnl_graph_op_attr_momentum<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a7680bb9aeb9227c2182af4a9d6f39923>`,
	    :ref:`dnnl_graph_op_attr_scales<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ac419fdbe6a4dd75f260e15e099ee69ad>`                         = 0x20,
	    :ref:`dnnl_graph_op_attr_axis<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a3423bc18cf4a529d02fb510594e8fa11>`                           = 0x30,
	    :ref:`dnnl_graph_op_attr_begin_norm_axis<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a227f82acd3e1dda739500ec52f216adb>`,
	    :ref:`dnnl_graph_op_attr_groups<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a5212f4a352a5d8197c3314b8232a12e2>`,
	    :ref:`dnnl_graph_op_attr_axes<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832adfc409d19c0db55446bea791f2d6d002>`                           = 0x40,
	    :ref:`dnnl_graph_op_attr_dilations<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ae4a124e36295a9d25e39840b4de42136>`,
	    :ref:`dnnl_graph_op_attr_dst_shape<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ae2a4b1da4350a191e00251f3ff06e8a2>`,
	    :ref:`dnnl_graph_op_attr_kernel<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832af4873033b38648631ae260b6c71dd42a>`,
	    :ref:`dnnl_graph_op_attr_order<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aaf37c7da0d9ebaa04f2dde457adf3783>`,
	    :ref:`dnnl_graph_op_attr_output_padding<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6802b98659c502370cc2b71becc90995>`,
	    :ref:`dnnl_graph_op_attr_pads_begin<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832abbb2697080b0208aeae508831718db10>`,
	    :ref:`dnnl_graph_op_attr_pads_end<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832af3d7a3b02e2abe925d6560f95086828d>`,
	    :ref:`dnnl_graph_op_attr_shape<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aa83e649f70a180bcb658295154d115c7>`,
	    :ref:`dnnl_graph_op_attr_sizes<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a4274f5013a56357c46a55ca02979e548>`,
	    :ref:`dnnl_graph_op_attr_src_shape<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6e97fe3a88af029eed5962df01305a0c>`,
	    :ref:`dnnl_graph_op_attr_strides<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aa736b04ce456907b8f02dd795a6902dd>`,
	    :ref:`dnnl_graph_op_attr_weights_shape<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6d05ad611dcbf729ce9198252c7a23b8>`,
	    :ref:`dnnl_graph_op_attr_zps<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a335b71c4cdde6e56606c7868a6569f3e>`,
	    :ref:`dnnl_graph_op_attr_group_shape<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6d25efb98da07928f14ac50f2e2507af>`,
	    :ref:`dnnl_graph_op_attr_exclude_pad<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aeb5b29d6c34ea6dc6baf325ab1c72b27>`                    = 0x60,
	    :ref:`dnnl_graph_op_attr_keep_dims<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832adf5464c345a2fa0c6cc9a5b436e85da9>`,
	    :ref:`dnnl_graph_op_attr_keep_stats<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a8b528737a39915ca4f5281e5b165e0bd>`,
	    :ref:`dnnl_graph_op_attr_per_channel_broadcast<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a77e3fc0248022e7c6560b34d5d3938ca>`,
	    :ref:`dnnl_graph_op_attr_special_zero<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ae780406804a8365209e2da03c4e6e703>`,
	    :ref:`dnnl_graph_op_attr_transpose_a<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a507d7ebf21a45e5392887b0e5c0dfeb6>`,
	    :ref:`dnnl_graph_op_attr_transpose_b<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a013a30a1417936029ec6ae9b5f69737d>`,
	    :ref:`dnnl_graph_op_attr_use_affine<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ac0f3edb6cd7a360029e6e507071492f4>`,
	    :ref:`dnnl_graph_op_attr_use_dst<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a9eaee6fa44e761a2c715b18624000e7c>`,
	    :ref:`dnnl_graph_op_attr_auto_broadcast<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a90f5cc388132fcc03854ac9fff882ace>`                 = 0x80,
	    :ref:`dnnl_graph_op_attr_auto_pad<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a0c1bdcdf4859a0752ae1ca289407c90e>`,
	    :ref:`dnnl_graph_op_attr_coordinate_transformation_mode<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832afcfd0a9d1097d68a8049c902d9806f00>`,
	    :ref:`dnnl_graph_op_attr_data_format<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a3f5e3951f43b1bb7d58545a8b707b5e2>`,
	    :ref:`dnnl_graph_op_attr_mode<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6e7d717c647469cd5dfb7caf2902b381>`,
	    :ref:`dnnl_graph_op_attr_qtype<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a897245601f49dd68fdcb3674ffe024a4>`,
	    :ref:`dnnl_graph_op_attr_rounding_type<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aca13756072eac044d5d7867a1b9fd06c>`,
	    :ref:`dnnl_graph_op_attr_weights_format<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a63800d1e05815c9b724dcc603883f9d9>`,
	    :ref:`dnnl_graph_op_attr_end<doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832adcd42cc792ee52d5e8b5351d4d06a2d7>`                            = 0xFF,
	};

.. _details-group__dnnl__graph__api__op_1ga106f069a858125ba0dd4d585b8f4e832:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Attributes of operations.

Enum Values
-----------

.. index:: pair: enumvalue; dnnl_graph_op_attr_undef
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832af29d647b9ab14f52143eab725f598881:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_undef

Undefined op attribute.

.. index:: pair: enumvalue; dnnl_graph_op_attr_alpha
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a23f876973125c172c40e166503fcd380:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_alpha

Specifies an alpha attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_beta
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a90dd6c02f9974b4ab68dabc17f6be576:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_beta

Specifies an beta attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_epsilon
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ae648365c7c28fadc330a852be9e66dca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_epsilon

Specifies an epsilon attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_max
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a3031c0dcbbb8fdef7f2e6a8b6013fbe3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_max

Specifies a max attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_min
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a0e5c8cf627d7cf34929ee3333db69180:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_min

Specifies a min attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_momentum
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a7680bb9aeb9227c2182af4a9d6f39923:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_momentum

Specifies a momentum attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_scales
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ac419fdbe6a4dd75f260e15e099ee69ad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_scales

Specifies a scales attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_axis
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a3423bc18cf4a529d02fb510594e8fa11:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_axis

Specifies an axis attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_begin_norm_axis
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a227f82acd3e1dda739500ec52f216adb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_begin_norm_axis

Specifies a begin_norm_axis attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_groups
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a5212f4a352a5d8197c3314b8232a12e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_groups

Specifies a groups attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_axes
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832adfc409d19c0db55446bea791f2d6d002:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_axes

Specifies an axes attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_dilations
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ae4a124e36295a9d25e39840b4de42136:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_dilations

Specifies a dilations attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_dst_shape
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ae2a4b1da4350a191e00251f3ff06e8a2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_dst_shape

Specifies an dst_shape attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_kernel
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832af4873033b38648631ae260b6c71dd42a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_kernel

Specifies a kernel attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_order
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aaf37c7da0d9ebaa04f2dde457adf3783:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_order

Specifies an order attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_output_padding
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6802b98659c502370cc2b71becc90995:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_output_padding

Specifies an output_padding attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_pads_begin
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832abbb2697080b0208aeae508831718db10:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_pads_begin

Specifies a pads_begin attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_pads_end
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832af3d7a3b02e2abe925d6560f95086828d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_pads_end

Specifies a pads_end attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_shape
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aa83e649f70a180bcb658295154d115c7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_shape

Specifies a shape attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_sizes
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a4274f5013a56357c46a55ca02979e548:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_sizes

Specifies a sizes attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_src_shape
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6e97fe3a88af029eed5962df01305a0c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_src_shape

Specifies a input_shape attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_strides
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aa736b04ce456907b8f02dd795a6902dd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_strides

Specifies a strides attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_weights_shape
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6d05ad611dcbf729ce9198252c7a23b8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_weights_shape

Specifies a weight_shape attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_zps
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a335b71c4cdde6e56606c7868a6569f3e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_zps

Specifies a zps attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_group_shape
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6d25efb98da07928f14ac50f2e2507af:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_group_shape

Specifies a group shape attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_exclude_pad
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aeb5b29d6c34ea6dc6baf325ab1c72b27:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_exclude_pad

Specifies an exclude_pad attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_keep_dims
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832adf5464c345a2fa0c6cc9a5b436e85da9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_keep_dims

Specifies a keep_dims attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_keep_stats
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a8b528737a39915ca4f5281e5b165e0bd:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_keep_stats

Specifies a keep_stats attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_per_channel_broadcast
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a77e3fc0248022e7c6560b34d5d3938ca:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_per_channel_broadcast

Specifies a per_channel_broadcast attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_special_zero
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ae780406804a8365209e2da03c4e6e703:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_special_zero

Specifies a special_zero attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_transpose_a
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a507d7ebf21a45e5392887b0e5c0dfeb6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_transpose_a

Specifies a transpose_a attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_transpose_b
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a013a30a1417936029ec6ae9b5f69737d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_transpose_b

Specifies a transpose_b attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_use_affine
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832ac0f3edb6cd7a360029e6e507071492f4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_use_affine

Specifies an use_affine attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_use_dst
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a9eaee6fa44e761a2c715b18624000e7c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_use_dst

Specifies an use_dst attribute to an op.

.. index:: pair: enumvalue; dnnl_graph_op_attr_auto_broadcast
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a90f5cc388132fcc03854ac9fff882ace:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_auto_broadcast

Specifies an auto_broadcast attribute to an op.

The value can be "none" or "numpy".

.. index:: pair: enumvalue; dnnl_graph_op_attr_auto_pad
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a0c1bdcdf4859a0752ae1ca289407c90e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_auto_pad

Specifies an auto_pad attribute to an op.

The value can be "none", "same_upper", "same_lower", or "valid".

.. index:: pair: enumvalue; dnnl_graph_op_attr_coordinate_transformation_mode
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832afcfd0a9d1097d68a8049c902d9806f00:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_coordinate_transformation_mode

Specifies an coordinate_transformation_mode attribute to an op.

The value can be "half_pixel" or "align_corners". The attribute is defined for Interpolate operations.

.. index:: pair: enumvalue; dnnl_graph_op_attr_data_format
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a3f5e3951f43b1bb7d58545a8b707b5e2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_data_format

Specifies a data_format of an op. The value can be "NCX" or "NXC".

.. index:: pair: enumvalue; dnnl_graph_op_attr_mode
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a6e7d717c647469cd5dfb7caf2902b381:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_mode

Specifies a mode attribute of an op.

The value can be "nearest", "linear", "bilinear", or "trilinear". The attribute is defined for Interpolate operations.

.. index:: pair: enumvalue; dnnl_graph_op_attr_qtype
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a897245601f49dd68fdcb3674ffe024a4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_qtype

Specifies a qtype attribute to an op.

The value can be "per_channel" or "per_tensor". The attribute is defined for quantization operations.

.. index:: pair: enumvalue; dnnl_graph_op_attr_rounding_type
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832aca13756072eac044d5d7867a1b9fd06c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_rounding_type

Specifies a rounding_type attribute to an op.

The value can be "ceil" or "floor".

.. index:: pair: enumvalue; dnnl_graph_op_attr_weights_format
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832a63800d1e05815c9b724dcc603883f9d9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_weights_format

Specifies a weights_format of an op.

The value can be "OIX", "XIO", "IOX", or "XOI". Different operations may support different values.

.. index:: pair: enumvalue; dnnl_graph_op_attr_end
.. _doxid-group__dnnl__graph__api__op_1gga106f069a858125ba0dd4d585b8f4e832adcd42cc792ee52d5e8b5351d4d06a2d7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dnnl_graph_op_attr_end

Specifies the end of all above exteral attributes for check.

