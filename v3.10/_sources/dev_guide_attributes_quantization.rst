.. index:: pair: page; Primitive Attributes: Quantization
.. _doxid-dev_guide_attributes_quantization:

Primitive Attributes: Quantization
==================================

:target:`doxid-dev_guide_attributes_quantization_1dgaq_intro`

Introduction
~~~~~~~~~~~~

Some primitives in the library support input/output tensors with the INT8 (either signed or unsigned) data type. The primary goal is to support reduced precision inference on the compatible hardware.

Related materials:

* `Lower Numerical Precision Deep Learning Inference and Training <https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf>`__

* An example with annotations: :ref:`Int8 Inference <doxid-dev_guide_inference_int8>`

Quantization Model
~~~~~~~~~~~~~~~~~~

The primary quantization model that the library assumes is the following:

.. math::

	x_{f32}[:] = scale_{x} \cdot (x_{int8}[:] - zp_{x})

where :math:`scale_{x}` is a scaling factor in float format, :math:`zp_{x}` is the zero point in int32 format, and :math:`[:]` is used to denote elementwise application of the formula to the arrays. In order to provide best performance, oneDNN does not compute those scaling factors and zero-points as part of primitive computation. Those should be computed and provided by the user.

These quantization parameters can either be computed ahead of time using calibration tools (static quantization) or at runtime based on the actual minimum and maximum values of a tensor (dynamic quantization). Either method can be used in conjunction with oneDNN, as the quantization parameters are passed to the oneDNN primitives at execution time.

To support int8 quantization, primitives should be created and executed as follow:

* during primitive descriptor creation, if one or multiple inputs are int8 (signed or not), then the primitive will behave as a quantized integer operation.

* still during primitive descriptor creation, the dimensionality of the scaling factors and zero-point should be provided using masks (e.g. one scale per tensor, one scale per channel, ...).

* finally, during primitive execution, the user must provide the actual quantization parameters as arguments to the execute function. Scales are ``f32`` values, and zero-points are ``s32`` values.

For performance reasons, each primitive implementation typically supports only a subset of quantization parameter masks. For example, convolution typically supports per-tensor or per-channel scales (no zero-point) for weights, and per-tensor scaling factor and zero-points for activation.

This guide does not cover how the appropriate scaling factor can be found. Refer to the materials in the :ref:`Introduction <doxid-dev_guide_attributes_quantization_1dgaq_intro>`.

Numerical behavior
------------------

Primitive implementations are allowed to convert int8 inputs to wider datatypes (e.g. int16 or int32), as those conversions do not impact accuracy.

During execution, primitives implementations avoid integer overflows and maintain integer accuracy by using wider datatypes (e.g. int32) for intermediate values and accumulators. Those are then converted as necessary before the result is written to the output memory objects.

When converting to integral datatypes, implementations typically saturate, whereas for floating-point datatypes, underflow/overflow can occur. To force saturation in floating-point datatypes use :ref:`dev_guide_attributes_post_ops_eltwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_eltwise>` with clip algorithm.

.. warning:: 

   Depending on the architecture, the behavior of int8 computations might slightly vary. For more details, refer to :ref:`Nuances of int8 Computations <doxid-dev_guide_int8_computations>`.
   
   
When multiple operations are fused in a single primitive using the :ref:`post ops attribute <doxid-dev_guide_attributes_post_ops>`, those are assumed to be computed in f32 precision. As a result the destination quantization parameters are applied after the post-ops as follow:

.. math::

	\dst[:] = post\_ops(OP(src[:], weights[:], ...)) / scale_{\dst} + zp_{\dst}

Quantizing/dequantizing values between post-operations can still be achieved using one of :ref:`eltwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_eltwise>`, :ref:`binary <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_binary>`, or the scale parameter of the appropriate post-operation.

Example: Convolution Quantization Workflow
------------------------------------------

Consider a convolution with bias. The tensors are represented as:

* :math:`\src_{f32}[:] = scale_{\src} \cdot (\src_{int8}[:] - zp_{\src})`

* :math:`\weights_{f32}[:] = scale_{\weights} \cdot \weights_{int8}[:]`

* :math:`\dst_{f32}[:] = scale_{\dst} \cdot (\dst_{int8}[:] - zp_{\dst})`

Here the :math:`\src_{f32}, \weights_{f32}, \dst_{f32}` are not computed at all, the whole work happens with int8 tensors.So the task is to compute the :math:`\dst_{int8}` tensor, using the :math:`\src_{int8}`, :math:`\weights_{int8}` tensors passed at execution time, as well as the corresponding quantization parameters :math:`scale_{\src}`, :math:`scale_{\weights}`, :math:`scale_{\dst}`, and :math:`zp_{\src}`, :math:`zp_{\dst}`. Mathematically, the computations are:

.. math::

	\dst_{int8}[:] = \operatorname{f32\_to\_int8}( (scale_{\src} \cdot scale_{\weights} \cdot \operatorname{s32\_to\_f32}(conv_{s32}(\src_{int8}, \weights_{int8}) - zp_{\src} \cdot comp_{s32}) + bias_{f32}) / scale_{\dst} + zp_{\dst} )

where

* :math:`\operatorname{conv}_{s32}` is just a regular convolution which takes source and weights with int8 data type and compute the result in int32 data type (int32 is chosen to avoid overflows during the computations);

* :math:`comp_{s32}` is a compensation term to account for :math:`\src` non-zero zero-point. This term is computed by the oneDNN library and can typically be pre-computed ahead of time, for example during weights reorder.

* :math:`\operatorname{f32\_to\_s8}()` converts an ``f32`` value to ``s8`` with potential saturation if the values are out of the range of the int8 data type.

* :math:`\operatorname{s32\_to\_f32}()` converts an ``int8`` value to ``f32`` with potential rounding. This conversion is typically necessary to apply ``f32`` scaling factors.

Per-Channel Scaling
-------------------

Some of the primitives have limited support of multiple scales for a quantized tensor. The most popular use case is the :ref:`Convolution <doxid-dev_guide_convolution>` primitive that supports per-output-channel scaling factors for the weights, meaning that the actual convolution computations would need to scale different output channels differently. This is possible without significant performance loss because the per-output-channel re-quantization is only required at the very end of the computations. It seems impossible to implement the same trick for the input channels, since that would require re-quantization for every input data point.

* :math:`\src_{f32}(n, ic, ih, iw) = scale_{\src} \cdot \src_{int8}(n, ic, ih, iw)`

* :math:`\weights_{f32}(oc, ic, kh, kw) = scale_{\weights}(oc) \cdot \weights_{int8}(oc, ic, kh, kw)`

* :math:`\dst_{f32}(n, oc, oh, ow) = scale_{\dst} \cdot \dst_{int8}(n, oc, oh, ow)`

Note that now the weights' scaling factor depends on :math:`oc`.

To compute the :math:`\dst_{int8}` we need to perform the following:

.. math::

	\dst_{int8}(n, oc, oh, ow) = \operatorname{f32\_to\_int8}( \frac{scale_{\src} \cdot scale_{\weights}(oc) \cdot conv_{s32}(\src_{int8}, \weights_{int8})|_{(n, oc, oh, ow)} + \bias_{f32}}{scale_{\dst}} ).

The user is responsible for preparing quantized weights accordingly. To do that, oneDNN provides reorders that can perform per-channel scaling:

.. math::

	\weights_{int8}(oc, ic, kh, kw) = \operatorname{f32\_to\_int8}( \weights_{f32}(oc, ic, kh, kw) / scale_{weights}(oc) ).

API
~~~

The library API to support for INT8 was designed for the model described above. However, it does not require users to follow exactly this model. As long as users can fit their model into the given functionality everything should work fine. Having this in mind we tried to design a minimal and simple yet powerful enough quantization API.

The most common data type for data tensors during INT8 inference is :ref:`dnnl::memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>` and :ref:`dnnl::memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`. All the scaling factors related to tensors are not attached in any way to the oneDNN memory objects and should be maintained by users.

The library essentially extends the ability of the primitives to scale the output before storing the result to the memory with the destination data type. That's exactly the minimum that we need to support INT8 inference (check the equations above only :math:`output\_scale` is non-standard).

The scaling happens in the single precision floating point data type (:ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`). Before storing, the result is downconverted to the destination data type with saturation if required. The rounding happens according to the current HW setting (for instance, on CPU according to the MXCSR register).

:target:`doxid-dev_guide_attributes_quantization_1dev_guide_attributes_quantization_scales`

Argument Scaling
----------------

The library uses :ref:`Primitive Attributes <doxid-dev_guide_attributes>` API for setting the scaling factors for most of the primitives. The supporting attributes can be found in the documentation for each primitive. The unsupported cases are handled according to the :ref:`attributes error handling section <doxid-dev_guide_attributes_1dev_guide_attributes_error_handling>`.

API:

* C: :ref:`dnnl_primitive_attr_set_scales_mask <doxid-group__dnnl__api__attributes_1gad7eac877f75cfa282be094b1e48cb71d>`

* C++: :ref:`dnnl::primitive_attr::set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`

Primitives support scales only when the data type of computation is an integer.

The parameters (C++ API for simplicity):

.. ref-code-block:: cpp

	void :ref:`dnnl::primitive_attr::set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(int arg, int mask);

In the simplest case, when there is only one common scale the attribute changes the op behavior from

.. math::

	\dst[:] = Op(...)

to

.. math::

	\dst[:] = scale \cdot Op(...).

To support scales per one or several dimensions, users must set the appropriate mask.

Say the destination is a :math:`D_0 \times ... \times D_{n-1}` tensor and we want to have output scales per :math:`d_i` dimension (where :math:`0 \le d_i < n`).

Then the mask should be set to:

* :math:`mask = \sum \limits_{d_i} 2^{d_i}`,

and the number of scales should be:

* ``scales.size()`` = :math:`\prod\limits_{d_i}D_{d_i}`.

Example 1: weights quantization with per-output-channel scaling
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. ref-code-block:: cpp

	   // weights dimensions
	   const int OC, IC, KH, KW;
	
	   // original f32 weights in plain format
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` wei_plain_f32_md(
	           {OC, IC, KH, KW},                 // dims
	           :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,     // the data originally in f32
	           :ref:`dnnl::memory::format_tag::hwigo <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fafd710c828421b3c91725b0e5aa53ecc6>`   // the plain memory format
	           );
	
	   // the scaling factors for quantized weights
	   // An unique scale for each output-channel.
	   std::vector<float> wei_scales(OC) = { /* values */ };
	   :ref:`dnnl::memory <doxid-structdnnl_1_1memory>`();
	
	   // int8 convolution primitive descriptor
	   :ref:`dnnl::convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>` conv_pd(/* see the next example */);
	
	   // query the convolution weights memory descriptor
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` wei_conv_s8_md = conv_pd.weights_desc();
	
	   // prepare the attributes for the reorder
	   :ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	   const int quantization_mask = 0
	       | (1 << 0);  // scale per  OC dimension, which is the dim #0
	   attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, quantization_mask);
	
	   // create reorder that would perform:
	   //   wei_s8(oc, ic, kh, kw) <- wei_f32(oc, ic, kh, kw) / scale(oc)
	   // including the data format conversion.
	   auto wei_reorder_pd = :ref:`dnnl::reorder::primitive_desc <doxid-structdnnl_1_1reorder_1_1primitive__desc>`(
	           wei_plain_f32_md, engine, // source
	           wei_conv_s8_md, engine, // destination,
	           attr);
	   auto wei_reorder = :ref:`dnnl::reorder <doxid-structdnnl_1_1reorder>`(wei_reorder_pd);
	
	// ...

Example 2: convolution with per-output-channel quantization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example is complementary to the previous example (which should ideally be the first one). Let's say we want to create an int8 convolution with per-output channel scaling.

.. ref-code-block:: cpp

	   const float src_scale; // src_f32[:] = src_scale * src_s8[:]
	   const float dst_scale; // dst_f32[:] = dst_scale * dst_s8[:]
	
	   // the scaling factors for quantized weights (as declared above)
	   // An unique scale for each output-channel.
	   std::vector<float> wei_scales(OC) = {...};
	
	
	   // Src, weights, and dst memory descriptors for convolution,
	   // with memory format tag == any to allow a convolution implementation
	   // to chose the appropriate memory format
	
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` src_conv_s8_any_md(
	           {BATCH, IC, IH, IW},          // dims
	           :ref:`dnnl::memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`,  // the data originally in s8
	           :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` // let convolution to choose
	           );
	
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` wei_conv_s8_any_md(
	           {OC, IC, KH, KW},             // dims
	           :ref:`dnnl::memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`,  // the data originally in s8
	           :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` // let convolution to choose
	           );
	
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` dst_conv_s8_any_md(...);  // ditto
	
	   // prepare the attributes for the convolution
	   :ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	   const int data_mask = 0; // scale and zero-point per tensor for source and destination
	   const int wei_mask = 0
	       | (1 << 0); // scale per OC dimension, which is the dim #0 on weights tensor:
	                   // (   OC, IC, KH, KW)
	                   //      0   1   2   3
	
	   attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, data_mask);
	   attr.:ref:`set_zero_points_mask <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, data_mask);
	
	   attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_mask);
	
	   attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, data_mask);
	   attr.:ref:`set_zero_points_mask <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`, data_mask);
	
	   // create a convolution primitive descriptor
	   auto conv_pd = :ref:`dnnl::convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>`(
	           :ref:`dnnl::prop_kind::forward_inference <doxid-group__dnnl__api__attributes_1ggac7db48f6583aa9903e54c2a39d65438fa3b9fad4f80d45368f856b5403198ac4c>`,
	           :ref:`dnnl::algorithm::convolution_direct <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640a5028ad8f818a45333a8a0eefad35c5c0>`,
	           src_conv_s8_any_md,                     // what's important is that
	           wei_conv_s8_any_md,                     // we specified that we want
	           dst_conv_s8_any_md,                     // computations in s8
	           strides, padding_l, padding_r,
	           dnnl::padding_kind::zero
	           attr);   // the attributes describe the quantization flow
	// ...

Example 3: matmul with advanced quantization
++++++++++++++++++++++++++++++++++++++++++++

This example describes a process of weights decompression, or weights-only-quantization (WoQ), in matmul primitive which may be found when running Large Language Models (LLM). The advanced quantization here refers to additional grouping introduced over reduction dimension besides traditional per-N quantization.

.. ref-code-block:: cpp

	   // Src, weights, and dst memory descriptors for matmul.
	   // Consider simple 2D matmul case.
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` src_f16_any_md(...);
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` wei_s8_any_md(
	           {K (256), N (512)},           // dims
	           :ref:`dnnl::memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`,  // the data originally in s8
	           :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` // let matmul to choose
	           );
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` dst_f16_any_md(...);
	
	   // prepare the attributes
	   :ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	   // scale per K and N dimensions:
	   const int wei_mask = (1 << 0) | (1 << 1);
	   // K dimension specifies the group size of `128`. It means that each 128
	   // elements over K dimension will share a single value. For a given example,
	   // there will be two groups, thus, two values referring to a single N value.
	   std::vector<dim_t> wei_groups = {128, 1}
	
	   // the scaling factors for quantized weights (as declared above)
	   // A unique scale for each gK (256 / 128 = 2) times N, total 1024 elements.
	   std::vector<half> wei_scales(gK, N) = {...};
	
	   attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a7b5f3e38305645eb52f05c4fe0401e34>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_mask, wei_groups, data_type::f16);
	
	   // Additionally, to instruct the library to perform weights decompression,
	   // fpmath mode must be set with a flag set to `true`:
	   attr.:ref:`set_fpmath_mode <doxid-structdnnl_1_1primitive__attr_1ab00639157a283596834ee5b0e8478a2d>`(fpmath_mode::f16, /* apply_to_int = */ true);
	
	   // create a matmul primitive descriptor
	   auto matmul_pd = :ref:`dnnl::matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(
	           engine,
	           src_f16_any_md,
	           wei_s8_any_md,
	           dst_f16_any_md,
	           attr);   // the attributes describe the quantization flow
	// ...

Example 4: matmul with precomputed reductions and advanced quantization
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This example is a complementary addition to the one above. It describes a process of dynamic quantization with weights's tensor asymmetric quantization and external precomputed reductions of the source tensor.

The case arises from the technique of quantizing source tensor on-the-fly (on the application side) and passing both quantized source and weights tensors to the library.

It's important that precomputed reductions appear from weights zero-points to provide accurate result when zero-points datatype is s8, in which case it's impossible to apply them on-the-fly without potential accuracy loss.

.. ref-code-block:: cpp

	   // Src, weights, and dst memory descriptors for matmul.
	   // Consider simple 2D matmul case.
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` src_u8_any_md(
	           {M (64), K (256)},            // dims
	           :ref:`dnnl::memory::data_type::u8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea077393852be20e37026d6281827662f2>`,  // the data originally in u8
	           :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` // let matmul to choose
	           );
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` wei_s8_any_md(
	           {K (256), N (512)},           // dims
	           :ref:`dnnl::memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`,  // the data originally in s8
	           :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` // let matmul to choose
	           );
	   :ref:`dnnl::memory::desc <doxid-structdnnl_1_1memory_1_1desc>` dst_f16_any_md(...);
	
	   // prepare the attributes
	   :ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	   // scale per K and N dimensions:
	   const int wei_mask = (1 << 0) | (1 << 1);
	   // K dimension specifies the group size of `128`. It means that each 128
	   // elements over K dimension will share a single value. For a given example,
	   // there will be two groups, thus, two values referring to a single N value.
	   std::vector<dim_t> wei_scales_groups = {128, 1}
	
	   // The scaling factors for quantized weights (as declared above)
	   // A unique scale for each scale_gK (256 / 128 = 2) times N, total 1024
	   // elements.
	   std::vector<half> wei_scales(scale_gK, N) = {...};
	
	   attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a7b5f3e38305645eb52f05c4fe0401e34>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_mask, wei_scales_groups,
	           data_type::f16);
	
	   // Zero-points would have the same mask as grouping applies for them as well.
	   // For example, let it use the different size of the group.
	   std::vector<dim_t> wei_zp_groups = {64, 1};
	
	   // The zero-point factors for quantized weights (as declared above)
	   // A unique zero-point for each zp_gK (256 / 64 = 4) times N, total 2048
	   // elements.
	   std::vector<half> wei_zps(zp_gK, N) = {...};
	
	   attr.:ref:`set_zero_points <doxid-structdnnl_1_1primitive__attr_1a2a8693f2aba0541ccd59470b41321175>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_mask, wei_zp_groups,
	           data_type::s8);
	
	   // Now, specify the precomputed reductions.
	   // Note that it's specified for source tensor.
	   // It means it should have full-size source tensor mask (which in this
	   // example coincides with `wei_mask`), and groups would be over another
	   // dimension, same as zero-points group size.
	   std::vector<dim_t> src_pr_groups = {1, 64};
	
	   // The precomputed reduction factors for quantized sources.
	   // A unique reduction for each M times pr_gK (256 / 64 = 4), total 256
	   // elements.
	   std::vector<half> src_prs(M, pr_gK) = {...};
	
	   attr.:ref:`set_precomputed_reductions <doxid-structdnnl_1_1primitive__attr_1a24a349d345ac97756a54b01b634b1b3c>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, src_tensor_mask,
	           src_pr_groups);
	
	   // fpmath mode is not required in case of dynamic quantization as it's
	   // treated as classical quantization case.
	
	   // create a matmul primitive descriptor
	   auto matmul_pd = :ref:`dnnl::matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(
	           engine,
	           src_s8_any_md,
	           wei_s8_any_md,
	           dst_f16_any_md,
	           attr);   // the attributes describe the quantization flow
	// ...

Special Case: Host-side Scalar Scale and Zero-point
---------------------------------------------------

When using the GPU engine, host-side scalar scales and zero-points are supported to reduce copying of data from host to device. A memory object for scale or zero-point host value should be created as a host-side scalar (see :ref:`Host-Side Scalars Support <doxid-dev_guide_host_side_scalars>` for details) and passed to the primitive execution function. The host scales or zero-points attributes should also be set using the following API:

.. ref-code-block:: cpp

	:ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	attr.:ref:`set_host_scale <doxid-structdnnl_1_1primitive__attr_1a7b035390cde177453afae9c5b5a7c29e>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`,
	           memory::data_type::f32);
	
	attr.:ref:`set_host_zero_point <doxid-structdnnl_1_1primitive__attr_1ac6aac2aa4418da036964baa3a35ed879>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`,
	           memory::data_type::s32);

