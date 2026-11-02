.. index:: pair: enum; attr
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684:

enum dnnl::graph::op::attr
==========================

Overview
~~~~~~~~

Attributes of operations. :ref:`More...<details-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <dnnl_graph.hpp>

	enum attr
	{
	    :ref:`undef<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684af31ee5e3824f1f5e5d206bdf3029f22b>`                          = dnnl_graph_op_attr_undef,
	    :ref:`alpha<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2c1743a391305fbf367df8e4f069f9f9>`                          = dnnl_graph_op_attr_alpha,
	    :ref:`beta<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a987bcab01b929eb2c07877b224215c92>`                           = dnnl_graph_op_attr_beta,
	    :ref:`epsilon<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3cd38ab30e1e7002d239dd1a75a6dfa8>`                        = dnnl_graph_op_attr_epsilon,
	    :ref:`max<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2ffe4e77325d9a7152f7086ea7aa5114>`                            = dnnl_graph_op_attr_max,
	    :ref:`min<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad8bd79cc131920d5de426f914d17405a>`                            = dnnl_graph_op_attr_min,
	    :ref:`momentum<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3a749f8e94241d303c81e056e18621d4>`                       = dnnl_graph_op_attr_momentum,
	    :ref:`scales<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8e7bb02b763a2e07d30b4ab24beb7fa1>`                         = dnnl_graph_op_attr_scales,
	    :ref:`axis<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad>`                           = dnnl_graph_op_attr_axis,
	    :ref:`begin_norm_axis<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ac4fe88742dd733999b9a5e4db0322415>`                = dnnl_graph_op_attr_begin_norm_axis,
	    :ref:`groups<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1471e4e05a4db95d353cc867fe317314>`                         = dnnl_graph_op_attr_groups,
	    :ref:`axes<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a42b47eed53099988e3cb7be539eb92e0>`                           = dnnl_graph_op_attr_axes,
	    :ref:`dilations<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684acbcf9c952f6e423b94fe04593665b49e>`                      = dnnl_graph_op_attr_dilations,
	    :ref:`dst_shape<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8ab1066346d3720658f87bb7686f7a22>`                      = dnnl_graph_op_attr_dst_shape,
	    :ref:`kernel<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a50484c19f1afdaf3841a0d821ed393d2>`                         = dnnl_graph_op_attr_kernel,
	    :ref:`order<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a70a17ffa722a3985b86d30b034ad06d7>`                          = dnnl_graph_op_attr_order,
	    :ref:`output_padding<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a16e84dbe0f1d0f82b74ebd187a0fe466>`                 = dnnl_graph_op_attr_output_padding,
	    :ref:`pads_begin<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310>`                     = dnnl_graph_op_attr_pads_begin,
	    :ref:`pads_end<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6>`                       = dnnl_graph_op_attr_pads_end,
	    :ref:`shape<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8c73a98a300905900337f535531dfca6>`                          = dnnl_graph_op_attr_shape,
	    :ref:`sizes<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ab027168eed2f9d69319d4819454b8ab4>`                          = dnnl_graph_op_attr_sizes,
	    :ref:`src_shape<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a27bbe1bc8190497bf47ed8bbab478a8b>`                      = dnnl_graph_op_attr_src_shape,
	    :ref:`strides<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a>`                        = dnnl_graph_op_attr_strides,
	    :ref:`weights_shape<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a62793d74da7cb2cac94dc9e5d7516151>`                  = dnnl_graph_op_attr_weights_shape,
	    :ref:`zps<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a5c284a074767998e9708c3656d41a91c>`                            = dnnl_graph_op_attr_zps,
	    :ref:`exclude_pad<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9e17a7762faf53a18315187610b2351c>`                    = dnnl_graph_op_attr_exclude_pad,
	    :ref:`keep_dims<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4ff344d49c4967e273f5e2a7b6f866b9>`                      = dnnl_graph_op_attr_keep_dims,
	    :ref:`keep_stats<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ac83b685e59ae9a2f78e9996886186e99>`                     = dnnl_graph_op_attr_keep_stats,
	    :ref:`per_channel_broadcast<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a652a82e843431baeacb5dfdedfd49d12>`          = dnnl_graph_op_attr_per_channel_broadcast,
	    :ref:`special_zero<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1ae9768d4bee269575f7464724cd97fa>`                   = dnnl_graph_op_attr_special_zero,
	    :ref:`transpose_a<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8739d82596ce4e8592bde9475504c430>`                    = dnnl_graph_op_attr_transpose_a,
	    :ref:`transpose_b<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684aa842de682cfdaec3291bbdffa551f4d7>`                    = dnnl_graph_op_attr_transpose_b,
	    :ref:`use_affine<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a014a6940b2c348a18720fcc350cb8e16>`                     = dnnl_graph_op_attr_use_affine,
	    :ref:`use_dst<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a36cda38ebb5a6a6b42b9789b20bd818c>`                        = dnnl_graph_op_attr_use_dst,
	    :ref:`auto_broadcast<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a0624e198ec0ae510048b88ff934822cc>`                 = dnnl_graph_op_attr_auto_broadcast,
	    :ref:`auto_pad<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9a6ac749896e044fe3122bd98e44ac9b>`                       = dnnl_graph_op_attr_auto_pad,
	    :ref:`coordinate_transformation_mode<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a171f02207298aa1f95eacc0907efe069>` = dnnl_graph_op_attr_coordinate_transformation_mode,
	    :ref:`data_format<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5>`                    = dnnl_graph_op_attr_data_format,
	    :ref:`mode<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a15d61712450a686a7f365adf4fef581f>`                           = dnnl_graph_op_attr_mode,
	    :ref:`qtype<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a63da59315662c87a47b7a1a4847e675e>`                          = dnnl_graph_op_attr_qtype,
	    :ref:`rounding_type<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae09cfc230f470609746f3021591072e3>`                  = dnnl_graph_op_attr_rounding_type,
	    :ref:`weights_format<doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a51c305464b90b1e5e4092ccfb5e904a7>`                 = dnnl_graph_op_attr_weights_format,
	};

.. _details-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Attributes of operations.

Different operations support different attributes. Check the document of each operation for what attributes are supported and what are the potential values for them. Missing required attribute or illegal attribute value may lead to failure when adding the operation to a graph.

Enum Values
-----------

.. index:: pair: enumvalue; undef
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684af31ee5e3824f1f5e5d206bdf3029f22b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	undef

Undefined op attribute.

.. index:: pair: enumvalue; alpha
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2c1743a391305fbf367df8e4f069f9f9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	alpha

Specifies an alpha attribute to an op.

.. index:: pair: enumvalue; beta
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a987bcab01b929eb2c07877b224215c92:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	beta

Specifies an beta attribute to an op.

.. index:: pair: enumvalue; epsilon
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3cd38ab30e1e7002d239dd1a75a6dfa8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	epsilon

Specifies an epsilon attribute to an op.

.. index:: pair: enumvalue; max
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a2ffe4e77325d9a7152f7086ea7aa5114:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	max

Specifies a max attribute to an op.

.. index:: pair: enumvalue; min
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad8bd79cc131920d5de426f914d17405a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	min

Specifies a min attribute to an op.

.. index:: pair: enumvalue; momentum
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3a749f8e94241d303c81e056e18621d4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	momentum

Specifies a momentum attribute to an op.

.. index:: pair: enumvalue; scales
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8e7bb02b763a2e07d30b4ab24beb7fa1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	scales

Specifies a scales attribute to an op.

.. index:: pair: enumvalue; axis
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a433169d5d9bcbb6d43f0d288e68f0cad:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	axis

Specifies an axis attribute to an op.

.. index:: pair: enumvalue; begin_norm_axis
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ac4fe88742dd733999b9a5e4db0322415:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	begin_norm_axis

Specifies a begin_norm_axis attribute to an op.

.. index:: pair: enumvalue; groups
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1471e4e05a4db95d353cc867fe317314:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	groups

Specifies a groups attribute to an op.

.. index:: pair: enumvalue; axes
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a42b47eed53099988e3cb7be539eb92e0:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	axes

Specifies an axes attribute to an op.

.. index:: pair: enumvalue; dilations
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684acbcf9c952f6e423b94fe04593665b49e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dilations

Specifies a dilations attribute to an op.

.. index:: pair: enumvalue; dst_shape
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8ab1066346d3720658f87bb7686f7a22:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	dst_shape

Specifies an dst_shape attribute to an op.

.. index:: pair: enumvalue; kernel
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a50484c19f1afdaf3841a0d821ed393d2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	kernel

Specifies a kernel attribute to an op.

.. index:: pair: enumvalue; order
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a70a17ffa722a3985b86d30b034ad06d7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	order

Specifies an order attribute to an op.

.. index:: pair: enumvalue; output_padding
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a16e84dbe0f1d0f82b74ebd187a0fe466:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	output_padding

Specifies an output_padding attribute to an op.

.. index:: pair: enumvalue; pads_begin
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ad9563b69290681059378cb6b98127310:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	pads_begin

Specifies a pads_begin attribute to an op.

.. index:: pair: enumvalue; pads_end
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae9dcd3256fd8b6e2b6385091cffe2cd6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	pads_end

Specifies a pads_end attribute to an op.

.. index:: pair: enumvalue; shape
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8c73a98a300905900337f535531dfca6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	shape

Specifies a shape attribute to an op.

.. index:: pair: enumvalue; sizes
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ab027168eed2f9d69319d4819454b8ab4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	sizes

Specifies a sizes attribute to an op.

.. index:: pair: enumvalue; src_shape
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a27bbe1bc8190497bf47ed8bbab478a8b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	src_shape

Specifies an src_shape attribute to an op.

.. index:: pair: enumvalue; strides
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a3372f3d8ac7d6db0997a8fe6b38d549a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	strides

Specifies a strides attribute to an op.

.. index:: pair: enumvalue; weights_shape
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a62793d74da7cb2cac94dc9e5d7516151:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	weights_shape

Specifies a weight_shape attribute to an op.

.. index:: pair: enumvalue; zps
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a5c284a074767998e9708c3656d41a91c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	zps

Specifies a zps attribute to an op.

.. index:: pair: enumvalue; exclude_pad
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9e17a7762faf53a18315187610b2351c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	exclude_pad

Specifies an exclude_pad attribute to an op.

.. index:: pair: enumvalue; keep_dims
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4ff344d49c4967e273f5e2a7b6f866b9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	keep_dims

Specifies a keep_dims attribute to an op.

.. index:: pair: enumvalue; keep_stats
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ac83b685e59ae9a2f78e9996886186e99:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	keep_stats

Specifies a keep_stats attribute to an op.

.. index:: pair: enumvalue; per_channel_broadcast
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a652a82e843431baeacb5dfdedfd49d12:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	per_channel_broadcast

Specifies a per_channel_broadcast attribute to an op.

.. index:: pair: enumvalue; special_zero
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a1ae9768d4bee269575f7464724cd97fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	special_zero

Specifies a special_zero attribute to an op.

.. index:: pair: enumvalue; transpose_a
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a8739d82596ce4e8592bde9475504c430:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	transpose_a

Specifies a transpose_a attribute to an op.

.. index:: pair: enumvalue; transpose_b
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684aa842de682cfdaec3291bbdffa551f4d7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	transpose_b

Specifies a transpose_b attribute to an op.

.. index:: pair: enumvalue; use_affine
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a014a6940b2c348a18720fcc350cb8e16:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	use_affine

Specifies an use_affine attribute to an op.

.. index:: pair: enumvalue; use_dst
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a36cda38ebb5a6a6b42b9789b20bd818c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	use_dst

Specifies an use_dst attribute to an op.

.. index:: pair: enumvalue; auto_broadcast
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a0624e198ec0ae510048b88ff934822cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	auto_broadcast

Specifies an auto_broadcast attribute to an op.

The value can be "none" or "numpy".

.. index:: pair: enumvalue; auto_pad
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a9a6ac749896e044fe3122bd98e44ac9b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	auto_pad

Specifies an auto_pad attribute to an op.

The value can be "none", "same_upper", "same_lower", or "valid".

.. index:: pair: enumvalue; coordinate_transformation_mode
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a171f02207298aa1f95eacc0907efe069:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	coordinate_transformation_mode

Specifies an coordinate_transformation_mode attribute to an op.

The value can be "half_pixel" or "align_corners". The attribute is defined for Interpolate operations.

.. index:: pair: enumvalue; data_format
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a4abbd547d2eb3887fd8613bb8be33cc5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	data_format

Specifies a data_format of an op. The value can be "NCX" or "NXC".

.. index:: pair: enumvalue; mode
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a15d61712450a686a7f365adf4fef581f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	mode

Specifies a mode attribute of an op.

The value can be "nearest", "linear", "bilinear", or "trilinear". The attribute is defined for Interpolate operations.

.. index:: pair: enumvalue; qtype
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a63da59315662c87a47b7a1a4847e675e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	qtype

Specifies a qtype attribute to an op.

The value can be "per_channel" or "per_tensor". The attribute is defined for quantization operations.

.. index:: pair: enumvalue; rounding_type
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684ae09cfc230f470609746f3021591072e3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	rounding_type

Specifies a rounding_type attribute to an op.

The value can be "ceil" or "floor".

.. index:: pair: enumvalue; weights_format
.. _doxid-classdnnl_1_1graph_1_1op_1ac7650c0c15849338f9c558f53ce82684a51c305464b90b1e5e4092ccfb5e904a7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	weights_format

Specifies a weights_format of an op.

The value can be "OIX", "XIO", "IOX", or "XOI". Different operations may support different values.

