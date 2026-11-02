.. index:: pair: page; Primitive Attributes: Post-ops
.. _doxid-dev_guide_attributes_post_ops:

Primitive Attributes: Post-ops
==============================

oneDNN implements some basic capabilities of operation fusion using the post-ops attributes API. The operation fusion typically reduces the memory bandwidth pressure hence leading to higher performance.

Post-ops are operations that are appended after a primitive. They are implemented using the :ref:`Primitive Attributes <doxid-dev_guide_attributes>` mechanism. If there are multiple post-ops, they are executed in the order they have been appended.

Currently the following post-ops are supported by the library:

* :ref:`Eltwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_eltwise>`

* :ref:`Sum <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_sum>`

* :ref:`Depthwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise>`

* :ref:`Binary <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_binary>`

* :ref:`PReLu <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_prelu>`

Just like :ref:`Primitive Attributes <doxid-dev_guide_attributes>`, the post-ops are represented by an opaque structure (:ref:`dnnl_post_ops_t <doxid-group__dnnl__api__attributes_1ga7d715ce1a81606584df9dcf045976401>` in C API and :ref:`dnnl::post_ops <doxid-structdnnl_1_1post__ops>` in C++ API) which is copied once it is attached to the attributes using the C++ :ref:`dnnl::primitive_attr::set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>` or C :ref:`dnnl_primitive_attr_set_post_ops <doxid-group__dnnl__api__attributes_1ga7045d42606599f156bfca69820c21ea2>` functions. The attributes then must be passed to a primitive descriptor creation function to take effect. Below is a simple skeleton for the C++ API:

.. ref-code-block:: cpp

	:ref:`dnnl::post_ops <doxid-structdnnl_1_1post__ops>` po; // default empty post-ops
	assert(po.:ref:`len <doxid-structdnnl_1_1post__ops_1a84653b68d83c2d84d3ac432a8dc1f5fd>`() == 0); // no post-ops attached
	
	po.append_SOMETHING(params); // append some particular post-op
	po.append_SOMETHING_ELSE(other_params); // append one more post-op
	
	// (!) Note that the order in which post-ops are appended matters!
	assert(po.:ref:`len <doxid-structdnnl_1_1post__ops_1a84653b68d83c2d84d3ac432a8dc1f5fd>`() == 2);
	
	:ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr; // default attributes
	attr.set_post_ops(po); // attach the post-ops to the attr
	
	// further po changes would not affect attr
	
	primitive::primitive_desc op_pd(engine, params, attr); // create a pd with the attr

.. note:: 

   Different post-ops can be chained together by appending one after another. Note that the appending order matters: the sequence of the post operations is executed in the order of appearance. The maximum number of post operations supported in the library is 32.
   
   

.. warning:: 

   Different primitives may have different post-ops support. Each primitive documentation page contains information about what kind of post operations it supports. Moreover, the support might also depend on the actual implementation of a primitive. For instance, the library may not support post-ops for primitive reference implementations (which are typically very slow, so there is no point in doing the actual fusion). Robust code should handle errors accordingly. See the :ref:`section on attributes error handling <doxid-dev_guide_attributes_1dev_guide_attributes_error_handling>`.
   
   

.. note:: 

   Post-ops do not change the memory format of the operation destination memory object.
   
   
The post-op object can be inspected using the :ref:`dnnl::post_ops::kind() <doxid-structdnnl_1_1post__ops_1a454acad1a18f2763f07b42912778c0f8>` function that takes an index of the post-op (which must be less than the value returned by :ref:`dnnl::post_ops::len() <doxid-structdnnl_1_1post__ops_1a84653b68d83c2d84d3ac432a8dc1f5fd>`), and returns its kind.

Supported Post-ops
~~~~~~~~~~~~~~~~~~

:target:`doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_eltwise`

Eltwise Post-op
---------------

The eltwise post-op enables fusing a primitive with an :ref:`Eltwise <doxid-dev_guide_eltwise>` primitive. This is probably one of the most popular kinds of fusion: an eltwise (typically an activation function) with preceding convolution or inner product.

The :ref:`dnnl::primitive::kind <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` of this post-op is :ref:`dnnl::primitive::kind::eltwise <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a98b908c7d0339bb6a4832db44fc2c8da>`.

API:

* C: :ref:`dnnl_post_ops_append_eltwise <doxid-group__dnnl__api__attributes_1gaf5927e8931bf113abb94837541cec662>`

* C++: :ref:`dnnl::post_ops::append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`

The parameters (C++ API for simplicity):

.. ref-code-block:: cpp

	void :ref:`dnnl::post_ops::append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(
	        :ref:`algorithm <doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` alg, float alpha, float beta // same as in eltwise primitive
	        );

The ``alg``, ``alpha``, and ``beta`` parameters are the same as in :ref:`Eltwise <doxid-dev_guide_eltwise>`.

The eltwise post-op replaces:

.. math::

	\dst[:] = \operatorname{Op}(...)

with

.. math::

	\dst[:] = \operatorname{eltwise}( \operatorname{Op}(...) )

The intermediate result of :math:`\operatorname{Op}(...)` is not preserved. Hence, in most cases this kind of fusion cannot be used during training.

:target:`doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_sum`

Sum Post-op
-----------

The sum post-op accumulates the result of a primitive with the existing data. Prior to accumulating the result, the existing value would be shifted by the zero point and multiplied by scale.

The kind of this post-op is :ref:`dnnl::primitive::kind::sum <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a1d623b89683f9ce4e074de1676d12416>`.

This feature might improve performance for cases like residual learning blocks, where the result of a convolution is accumulated to the previously computed activations. The scale parameter can be used in :ref:`INT8 <doxid-dev_guide_attributes_quantization>` inference only when the result and previous activations have different magnitudes. For all other cases the scale must be ``1.0``.

The sum post-op replaces

.. math::

	\dst[:] = \operatorname{Op}(...)

with

.. math::

	\dst[:] = scale \cdot (\dst[:] - zero\_point) + \operatorname{Op}(...)

If the data type parameter is specified, the original destination tensor will be reinterpreted as a tensor with the provided data type. Because it is a reinterpretation, data_type and the destination data type must have the same size. As a result, the computation will be:

.. math::

	\dst(:) = scale \cdot (\operatorname{as\_data\_type}(\dst[:]) - zero\_point) + \operatorname{Op}(...)

.. note:: 

   * GPU
     
     * Currently only a u8/s8 data type parameter is supported.
     
     * Zero point is not supported.
   
   
:target:`doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise`

Depthwise Post-op
-----------------

Appends a Depthwise convolution as a post-op. This post-op can only be fused with 1x1 convolution as generally seen in models (like MobileNet_v1) that use a stack of Separable convolutions: Depthwise convolution followed by 1x1 convolution. The stack of these Separable convolutions (like in MobileNet_v1) provide an opportunity to fuse 1x1-Convolution with bandwidth-limited Depthwise convolution.

The :ref:`dnnl::primitive::kind <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` of this post-op is :ref:`dnnl::primitive::kind::convolution <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169aa9595c1c24c33b16056d2ad07e71682d>`.

API:

* C: :ref:`dnnl_post_ops_append_dw <doxid-group__dnnl__api__attributes_1ga38509493009271e2b8c6d8fadb1fcac1>`

* C++: :ref:`dnnl::post_ops::append_dw <doxid-structdnnl_1_1post__ops_1a55aad3b45a25087e0045a005384bde3a>`

For better readability, below we assume a 2D convolution and use the following notations:

* ``conv_1x1`` Convolution with weights spatial=1 i.e., ``kh`` = ``kw`` = 1.

* ``conv_dw`` Depthwise convolution with weights spatial=3 i.e., ``kh`` = ``kw`` = 3, ``g`` = ``oc`` = ``ic`` and ``pad_l`` = ``pad_r`` = {1, 1}.

The Depthwise post-op replaces

.. math::

	dst[:] = Conv_{1x1}(...)

with

.. math::

	dst[:] = Conv_{dw}(Conv_{1x1}(...))

The final output dimensions of the after post-op is defined as

.. math::

	dst_{conv_dw} = \{ n, oc_{1x1}, \operatorname{ceil}(oh_{conv_{1x1}}/stride), \operatorname{ceil}(ow_{conv_{1x1}}/stride) \}

where ``oh_conv_1x1``, ``ow_conv_1x1`` are height and width of conv_1x1 destination.

.. image:: img_depthwise_fusion.jpg
	:alt: Fusion

Supported data types

==========================  ===================================  ====================================  ==========  
conv 1x1 output data type   depthwise post-op output data type   depthwise post-op weights data type   dep         
==========================  ===================================  ====================================  ==========  
u8, s8                      u8, s8, s32, f32                     s8                                    f32, s32    
f32                         f32                                  f32                                   f32         
bf16                        bf16, f32                            bf16                                  f32, bf16   
f16                         f16, f32                             f16                                   f32, f16    
==========================  ===================================  ====================================  ==========

.. note:: 

   * Though it is called a post-operation type, it does not follow the post-operation convention which implies an application of operation in f32 data type.
   
   * Currently only supported for 2D 1x1 convolution.
   
   * Sum or another depthwise post-ops cannot be a part of post-op chain.
   
   * The ``dst_1x1``, ``wei_dw`` and ``dst_dw`` are assumed to be :ref:`dnnl_format_tag_any <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dafee39ac6fff0325cae43cd66495c18ac>`.
   
   * Operation descriptor for base 1x1 convolution requires spatial dimensions of destination memory descriptor to coincide with source spatial dimensions. It is important for cases when depthwise post-op stride is not equal to ``1``. In this case, the queried destination descriptor after fusion will not coincide with the one passed to base convolution. It means that if intermediate object is utilized in other places in user application, its lifetime has to be handled by user separately since the library does not provide a mechanism to query an intermediate output of base convolution.
   
   * Currently, f16 support for depthwise fusion is only through reference fusion implementation. Thus, performance gain is not expected for this data type.
   
   
:target:`doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_binary`

Binary Post-op
--------------

The binary post-op enables fusing a primitive with a :ref:`Binary <doxid-dev_guide_binary>` primitive.

The :ref:`dnnl::primitive::kind <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` of this post-op is :ref:`dnnl::primitive::kind::binary <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a9d7183f16acce70658f686ae7f1a4d20>`.

API:

* C: :ref:`dnnl_post_ops_append_binary <doxid-group__dnnl__api__attributes_1gabc40e53d80f6f1d61cc5b17807d2446c>`

* C++: :ref:`dnnl::post_ops::append_binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`

The parameters (C++ API for simplicity):

.. ref-code-block:: cpp

	void :ref:`dnnl::post_ops::append_binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`(
	        :ref:`algorithm <doxid-group__dnnl__api__attributes_1ga00377dd4982333e42e8ae1d09a309640>` alg, // binary algorithm to apply
	        const memory::desc &src1 // memory descriptor for a second memory operand
	        );

The ``alg`` and ``src1`` parameters are the same as in :ref:`Binary <doxid-dev_guide_binary>`.

The binary post-op replaces:

.. math::

	\dst[:] = \operatorname{Op}(...)

with

.. math::

	\dst[:] = \operatorname{binary}(\operatorname{Op}(...), Source\_1[:])

The intermediate result of :math:`\operatorname{Op}(...)` is not preserved. Hence, in most cases this kind of fusion cannot be used during training.

Currently the following scenarios are optimized:

* Per tensor broadcast, when :math:`Source\_1` is represented as a one-element tensor, i.e. {1, 1, 1, 1} for 2D spatial :math:`\operatorname{Op}(...)`.

* Per channels (i.e. dimension 1) broadcast, when a ``dim[1]`` value of :math:`Source\_1` coincides with a ``dim[1]`` value of :math:`\operatorname{Op}(...)`, i.e. {1, C, 1, 1} for 2D spatial :math:`\operatorname{Op}(...)`.

* Per element broadcast, when :math:`Source\_1` coincides with :math:`\operatorname{Op}(...)`. In this case user may create ``src1`` memory descriptor with ``format_tag::any`` or set a specific tag. However, in later case if tags mismatch with :math:`\operatorname{Op}(...)`, it would result in suboptimal performance. In case of using ``format_tag::any``, a primitive descriptor of the operation will initialize a memory descriptor for binary post-operation which format may be queried from attributes using ``:ref:`dnnl::post_ops::get_params_binary <doxid-structdnnl_1_1post__ops_1a0a367859cac33d597743de491d26dcb9>`(...)`` function call.

:target:`doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_prelu`

Prelu Post-op
-------------

The prelu post-op enables fusing a primitive with a :ref:`PReLU <doxid-dev_guide_prelu>` primitive.

The :ref:`dnnl::primitive::kind <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169>` of this post-op is :ref:`dnnl::primitive::kind::prelu <doxid-structdnnl_1_1primitive_1ad1ec93215a0cf3aa0a32bae0c2cd9169a837c39f77d473b24eb27c0758d5c7c1b>`.

API:

* C: :ref:`dnnl_post_ops_append_prelu <doxid-group__dnnl__api__attributes_1ga833465b0aac349988b29245e1112656f>`

* C++: :ref:`dnnl::post_ops::append_prelu <doxid-structdnnl_1_1post__ops_1a1e538118474ac643c6da726a8a658b70>`

The parameters (C++ API for simplicity):

.. ref-code-block:: cpp

	void :ref:`dnnl::post_ops::append_prelu <doxid-structdnnl_1_1post__ops_1a1e538118474ac643c6da726a8a658b70>`(
	    int mask /*mask describing prelu weights broadcast.*/);

The prelu post-op replaces:

.. math::

	\dst[:] = \operatorname{Op}(...)

with

.. math::

	\dst[:] = \operatorname{prelu}(\operatorname{Op}(...), weights[:])

Assumptions:

* the weights tensor is passed in runtime using :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(index) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` \| DNNL_ARG_WEIGHTS mechanism, where index is the sequence number of the prelu in post-operations chain;

* only fp32 weights tensor data type is supported;

* only plain layout (a, ab, acb, acdb, acdeb) is supported for weights tensor;

* mask defines the correspondence between the output tensor dimensions and the prelu weights tensor. The set i-th bit indicates that a dedicated weights value is used for each index along that dimension. Mask 0 value means common (scalar) weights value for the whole output tensor.

* the order of dimensions does not depend on how elements are laid out in memory. For example:
  
  * for a 2D CNN activations tensor the order is always (n, c)
  
  * for a 4D CNN activations tensor the order is always (n, c, h, w)

Examples of Chained Post-ops
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different post-ops can be chained together by appending one after another. Note that the order matters: the post-ops are executed in the order they have been appended.

Let's consider some examples.

Sum -> ReLU
-----------

This pattern is pretty common for the CNN topologies of the ResNet family.

.. ref-code-block:: cpp

	:ref:`dnnl::post_ops <doxid-structdnnl_1_1post__ops>` po;
	po.:ref:`append_sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`();
	po.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(
	        /* alg kind  = */ :ref:`dnnl::algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`,
	        /* neg slope = */ 0.f,
	        /* unused for relu */ 0.f);
	
	:ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(po);
	
	convolution_forward::primitive_desc(conv_d, attr, engine);

This will lead to the following primitive behavior:

.. math::

	\dst[:] = \operatorname{ReLU}(\dst[:] + \operatorname{conv}(\src[:], \weights[:])

:target:`doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_with_scales`

Tanh -> Sum -> ScaleShift
-------------------------

This is a hypothetical example that illustrates the sequence of operations applied. We also set all the scales to values other than 1.0 and use :ref:`dnnl::primitive_attr::set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>` which will be covered in :ref:`Primitive Attributes: Quantization <doxid-dev_guide_attributes_quantization>`.

.. ref-code-block:: cpp

	:ref:`dnnl::post_ops <doxid-structdnnl_1_1post__ops>` po;
	po.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(
	        /* alg kind  = */ :ref:`dnnl::algorithm::eltwise_tanh <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a38dd7159307eab45742c78e72f06abb0>`,
	        /* unused for tanh */ 0.f,
	        /* unused for tanh */ 0.f);
	po.:ref:`append_sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`();
	po.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(
	        /* alg kind     = */ :ref:`dnnl::algorithm::eltwise_linear <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a21aba6844d2de47b92ab1d110f561945>`,
	        /* linear scale = */ alpha,
	        /* linear shift = */ beta);
	
	:ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, 0);
	attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, 0);
	attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, 0);
	attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(po);
	
	convolution_forward::primitive_desc(conv_d, attr, engine);

This will lead to the following primitive behavior (for better readability the tensors are designated by their names only; i.e., ``[:]`` is omitted):

.. math::

	\dst = s_{linear} \cdot ( \alpha \cdot ( s_{sum} \cdot \dst + s_{tanh} \cdot \tanh ( s_{conv} \cdot \operatorname{conv}(\src, \weights) ) ) + \beta )

:target:`doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise_fusion`

Relu -> Depthwise -> Relu
-------------------------

An example of fusing depthwise convolution with 1x1 convolution in MobileNet.

.. ref-code-block:: cpp

	:ref:`dnnl::post_ops <doxid-structdnnl_1_1post__ops>` po;
	
	po.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(
	        /* alg kind  = */ :ref:`dnnl::algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`,
	        /* neg slope = */ 0.f,
	        /* unused for relu */ 0.f);
	
	po.:ref:`append_dw <doxid-structdnnl_1_1post__ops_1a55aad3b45a25087e0045a005384bde3a>`(
	        /* depthwise weights data type = */ :ref:`dnnl::memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`,
	        /* depthwise bias data type (undef implies no bias) = */ :ref:`dnnl::memory::data_type::undef <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaf31ee5e3824f1f5e5d206bdf3029f22b>`,
	        /* depthwise destination data type = */ :ref:`dnnl::memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`,
	        /* kernel size of fused depthwise convolution = */ kernel,
	        /* stride size of fused depthwise convolution = */ stride,
	        /* padding size of fused depthwise convolution = */ padding)
	
	po.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(
	        /* alg kind  = */ :ref:`dnnl::algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`,
	        /* neg slope = */ 0.f,
	        /* unused for relu */ 0.f);
	
	:ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, 0);
	attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_ATTR_POST_OP_DW <doxid-group__dnnl__api__primitives__common_1ga47534804c9b2f9ede6b875f6cb08cc35>` | :ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, 0);
	attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(po);
	
	auto cpd = convolution_forward::primitive_desc(conv_1x1, attr, engine);
	auto dw_weight_md = cpd.query(query::exec_arg_md,
	                :ref:`DNNL_ARG_ATTR_POST_OP_DW <doxid-group__dnnl__api__primitives__common_1ga47534804c9b2f9ede6b875f6cb08cc35>` | :ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`);
	auto dw_bias_md = cpd.query(query::exec_arg_md,
	                :ref:`DNNL_ARG_ATTR_POST_OP_DW <doxid-group__dnnl__api__primitives__common_1ga47534804c9b2f9ede6b875f6cb08cc35>` | :ref:`DNNL_ARG_BIAS <doxid-group__dnnl__api__primitives__common_1gad0cbc09942aba93fbe3c0c2e09166f0d>`);

This will lead to the following primitive behaviour:

.. math::

	dst = ReLU_{depthwise} ( scales_{depthwise} \cdot ( conv_{depthwise} ( ReLU_{1x1} ( scales_{conv_{1x1}} \cdot ( conv_{1x1}() ) ) ) ) )

:target:`doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_binary_fusion`

Binary
------

An example of fusing convolution with binary post-op with per channel addition.

.. ref-code-block:: cpp

	:ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` conv_dst_md {MB, C, H, W}; /* 2D conv destination memory desc */
	
	:ref:`dnnl::post_ops <doxid-structdnnl_1_1post__ops>` po;
	
	/* Append eltwise post-op prior the binary post-op */
	po.:ref:`append_eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`(
	        /* alg kind  = */ :ref:`dnnl::algorithm::eltwise_relu <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640aba09bebb742494255b90b43871c01c69>`,
	        /* neg slope = */ 0.f,
	        /* unused for relu */ 0.f);
	
	/* Note that `C` coincides with the one from `conv_dst_md`. Also note that only
	 * supported memory format for src1 memory is `nchw` (or `abcd`) format. */
	po.:ref:`append_binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`(
	        /* alg kind = */ :ref:`dnnl::algorithm::binary_add <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640ab2c3faf084cf82b5603946995f637b35>`,
	        /* src1_md = */ :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>`(
	                {1, C, 1, 1},
	                :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	                :ref:`dnnl::memory::format_tag::abcd <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fae2fc714c4727ee9395f324cd2e7f331f>`));
	
	:ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	attr.:ref:`set_post_ops <doxid-structdnnl_1_1primitive__attr_1ac830fa9f4fcf480b494d73153ad579bf>`(po);
	
	auto cpd = convolution_forward::primitive_desc(conv, attr, engine);
	
	/* To set memory argument for binary post-op, the following should take place: */
	std::unordered_map<int, memory> args;
	
	args.insert(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, conv_src_memory);
	...
	int binary_post_op_position = 1; /* hard coded here, but may be queried */
	args.insert(
	        :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>`(binary_post_op_position) | :ref:`DNNL_ARG_SRC_1 <doxid-group__dnnl__api__primitives__common_1gadc5a5761633c05f4378780d23b7c9692>`, /* note parentheses around index */
	        binary_post_op_src1_memory);

