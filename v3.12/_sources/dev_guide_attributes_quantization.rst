.. index:: pair: page; Quantization
.. _doxid-dev_guide_attributes_quantization:

Quantization
============

:target:`doxid-dev_guide_attributes_quantization_1dgaq_intro`

Introduction
~~~~~~~~~~~~

Some primitives support input and output tensors with ``int8`` data types, both signed and unsigned, enabling reduced-precision inference on supported hardware.

Similarly, some primitives support `Open Compute Project (OCP) 8-bit Floating Point (f8) data types <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ designed to accelerate AI workloads, including training and inference of large neural networks. Lowering precision to 8 bits with ``f8`` enables faster computation and reduced memory usage.

See also:

* `Lower Numerical Precision Deep Learning Inference and Training <https://www.intel.com/content/dam/develop/external/us/en/documents/lower-numerical-precision-deep-learning-jan2018-754765.pdf>`__

Quantization Model
~~~~~~~~~~~~~~~~~~

oneDNN supports two main categories of quantization:

* Static Quantization (see :ref:`quantization_mode::dnnl_quantization_mode_static_sazp <doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a69f0c0079f39bec2332481404e199315>`) with scales only (symmetric) or scales and zero-points (asymmetric), where scales are applied after zero-point.

* Dynamic Quantization (see :ref:`quantization_mode::dnnl_quantization_mode_dynamic_mx <doxid-group__dnnl__api__attributes_1gga5342e1d6b2a09ea01660b3a3c2400826a9d29b3c3bf3c43cab388533e093cd8a6>`) compliant with the `OCP Microscaling (MX) Formats Specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__.

To support quantization, primitives should be created and executed as follows:

* During primitive descriptor creation source, weights or destination memory descriptors use low precision datatype (e.g., ``s8`` or ``f8_e4m3``).

* During primitive descriptor creation group size, data types, and broadcasting masks of the scaling factors and zero-point are provided using primitive attributes.

* During primitive execution the actual quantization parameters are provided as arguments to the execute function.

For performance reasons, each primitive implementation typically supports only a subset of quantization parameter masks, group sizes and data type combinations. Which combination is supported and optimized is listed in each primitive documentation page.

This guide does not cover how the appropriate scaling factor can be found. Refer to the materials in the :ref:`Introduction <doxid-dev_guide_attributes_quantization_1dgaq_intro>`.

Static Quantization
-------------------

The only formula for static quantization currently supported by oneDNN is with scales applied after zero-point as follows:

.. math::

	x_{f32}[:] = scale_{x} \cdot (x_{quant}[:] - zp_{x})

where :math:`x_{f32}` and :math:`x_{quant}` are the non-quantized and quantized representation of :math:`x` respectively, :math:`scale_{x}` is a scaling factor in a floating-point format, :math:`zp_{x}` is a zero point (typically in integral format), and :math:`[:]` is used to denote element-wise application of the formula to the arrays.

In this model, oneDNN assumes that quantization parameters are inputs provided by the user and the library does not compute those scaling factors and zero-points as part of primitive computation.

These quantization parameters can either be computed ahead of time using calibration tools or at runtime based on the actual minimum and maximum values of a tensor. Either method can be used in conjunction with oneDNN static quantization, as long as the quantization parameters are passed as input to the oneDNN primitives at execution time.

Dynamic Quantization
--------------------

oneDNN supports two dynamic quantization modes, for scales only (no zero-point) following the formula:

.. math::

	x_{f32}[:] = scale_{x} \cdot x_{quant}[:]

where :math:`x_{f32}` and :math:`x_{quant}` are the non-quantized and quantized representation of :math:`x` respectively, and :math:`scale_{x}` is a scaling factor.

When using ``quantization_mode::dynamic_mx``, :math:`scale_{x}` is computed following the `OCP MX Formats Specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__, namely :math:`scale_{x}` :

* has ``e8m0`` datatype,

* is computed for each group of size ``32``,

* is computed as the largest power-of-two less than or equal to the maximum absolute value of the group divided by the largest power-of-two representable in the :math:`x_{quant}` datatype, e.g., :math:`E8M0(amax(x_{quant}[:])) / E8M0(MAX\_QUANT\_DT)`.

When using ``quantization_mode::dynamic_fp``, :math:`scale_{x}` is computed in ``f32`` first and then converted to a scale datatype, namely :math:`scale_{x}` :

* has ``f8_e4m3`` datatype,

* is computed for each group of size ``16``,

* is computed as :math:`SCALE\_DT(amax(x_{quant}[:]) / MAX\_QUANT\_DT)`.

General Numerical Behavior Notes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Primitive implementations are allowed to convert inputs to wider data types (e.g., ``int8`` to ``int16`` or ``int32``), when those conversions do not impact accuracy.

During execution, primitives implementations avoid integer overflows and maintain integer accuracy by using wider data types (e.g., ``int32``) for intermediate values and accumulators.

Results are then converted as necessary before the result is written to the output memory objects.

The scales are applied in single precision floating point data type (:ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`) before downconversion to the destination data type. When converting to integral data types, implementations typically saturate, whereas for floating-point data types, underflow/overflow can occur. To force saturation in floating-point data types use :ref:`dev_guide_attributes_post_ops_eltwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_eltwise>` with clip algorithm. Rounding happens according to :ref:`rounding mode attribute <doxid-dev_guide_attributes_rounding_mode>`.

.. warning:: 

   Depending on the architecture, the behavior of ``int8`` computations might slightly vary. For more details, refer to :ref:`Nuances of int8 Computations <doxid-dev_guide_int8_computations>`.
   
   
When multiple operations are fused in a single primitive using the :ref:`post ops attribute <doxid-dev_guide_attributes_post_ops>`, those are assumed to be computed in ``f32`` precision. As a result the destination quantization parameters are applied after the post-ops as follows:

.. math::

	\dst[:] = post\_ops(OP(src[:], weights[:], ...)) / scale_{\dst} + zp_{\dst}

Quantizing and dequantizing values between post-operations can be achieved using one of :ref:`eltwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_eltwise>`, :ref:`binary <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_binary>`, or the scale parameter of the appropriate post-operation.

Relevant APIs and Supported Granularity Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

oneDNN provides APIs to set scales, zero-points, and precomputed reductions for different quantization levels from global (per-tensor) to fine-grained block-wise.

:target:`doxid-dev_guide_attributes_quantization_1dgaq_scaling`

Argument Scaling
----------------

The library uses :ref:`Primitive Attributes <doxid-dev_guide_attributes>` API for setting the scaling factors for most of the primitives. The supporting attributes can be found in the documentation for each primitive. The unsupported cases are handled according to the :ref:`attributes error handling section <doxid-dev_guide_attributes_1dev_guide_attributes_error_handling>`.

Available Scaling API Methods
+++++++++++++++++++++++++++++

oneDNN provides the following methods for setting scaling factors:

.. ref-code-block:: cpp

	// Legacy method with simple mask-based scaling
	void :ref:`dnnl::primitive_attr::set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(int arg, int mask);
	
	// Generic method with groups support
	void :ref:`dnnl::primitive_attr::set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(int arg, int mask,
	                                      const :ref:`dnnl::memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` &groups,
	                                      :ref:`dnnl::memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`,
	                                      bool is_on_host = false,
	                                      :ref:`quantization_mode <doxid-group__dnnl__api__attributes_1ga43df4b809a4544d34bbc106d3e409b2c>` qmode = quantization_mode::static_sazp);
	
	// Convenience method for single host-side scalar
	void :ref:`dnnl::primitive_attr::set_host_scale <doxid-structdnnl_1_1primitive__attr_1a7b035390cde177453afae9c5b5a7c29e>`(int arg,
	                                          :ref:`dnnl::memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`);

Key parameters of the scaling API methods are summarized below:

===============  ===============================================================================  ==============================================================================================================  
Parameter        Options*                                                                         Description                                                                                                     
===============  ===============================================================================  ==============================================================================================================  
``arg``          ``DNNL_ARG_SRC`` , ``DNNL_ARG_WEIGHTS`` , ``DNNL_ARG_DST`` , ``DNNL_ARG_BIAS``   Tensor to scale                                                                                                 
``mask``         ``0`` , ``1<<dim`` , ``(1<<d1)+(1<<d2)``                                         Scaling granularity: global, per-dimension, multi-dimensional                                                   
``groups``       ``{}`` , ``{G}`` , ``{G1,G2,...}``                                               Block quantization: none, single-size, multi-dimensional blocks                                                 
``data_type``    ``f32`` , ``bf16`` , ``f16`` , ``f8_e5m2`` , ``f8_e4m3`` , ``e8m0``              Scaling factor data type                                                                                        
``is_on_host``   ``true`` / ``false``                                                             Host vs device memory location of scaling factor                                                                
``qmode``        ``static_sazp`` , ``dynamic_mx`` , ``dynamic_fp``                                Quantization mode: static with scales and zero-points, dynamic (MXFP8 compatible), dynamic (NVFP4 compatible)   
===============  ===============================================================================  ==============================================================================================================

(\*) Support for quantization options varies based on individual primitive and target hardware. Refer to primitives documentation for the details.

Supported Scaling Granularity Levels
++++++++++++++++++++++++++++++++++++

oneDNN supports the following scaling granularity levels to support different quantization schemes:

* `Per-tensor scaling <#per-tensor-scaling>`__ (``mask=0``) uses a single scaling factor for the entire tensor, making it the simplest approach.

* `Per-channel scaling <#per-channel-scaling>`__ (``mask=1<<dim``) applies different scaling factors along a specific dimension, for instance commonly used for CNN weights.

* `Block scaling <#block-scaling>`__ subdivides tensor dimensions into smaller blocks with individual scaling factors, important for large transformer models and advanced quantization techniques.

* `Multi-dimensional scaling <#multi-dimensional-scaling>`__ (``mask=(1<<dim1)+(1<<dim2)``) provides independent scaling factors along multiple tensor dimensions, useful for complex activations where both batch and channel dimensions need separate scaling.

Per-tensor Scaling
******************

In the simplest case, when there is only one common scaling factor the attribute changes the op behavior from

.. math::

	\dst[:] = Op(...)

to

.. math::

	\dst[:] = scale \cdot Op(...).

.. ref-code-block:: cpp

	// Using full set_scales API (recommended)
	attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, 0, {}, :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`);
	
	// Using convenience set_host_scale API for host-side scaling factor
	attr.:ref:`set_host_scale <doxid-structdnnl_1_1primitive__attr_1a7b035390cde177453afae9c5b5a7c29e>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`);
	
	// Using legacy set_scales_mask API
	attr.:ref:`set_scales_mask <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, 0);
	
	// Scaling factors: 1 value
	// Usage: All elements use same scaling factor

.. note:: 

   For more details on global scaling with a single scaling factor residing on host, use :ref:`host-side scalar scaling <doxid-dev_guide_attributes_quantization_1host-side-scalars-and-zero-points>` (``set_host_scale``) to avoid device memory transfer overhead.
   
   
See examples:

* `Convolution with Per-output-channel Quantization <#convolution-with-per-output-channel-quantization>`__

Per-Channel Scaling
*******************

Per-channel scaling applies different scaling factors along specific tensor dimensions. For instance, it is commonly used for CNN weights where each output channel has its own scaling factor.

.. ref-code-block:: cpp

	// Scaling factor per output channel (dimension 0 of weights)
	attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, 1 << 0, {}, :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`);
	
	// Tensor: [OC, IC, H, W] = [64, 128, 3, 3]
	// Scaling factors: 64 values (one per output channel)
	// Usage: Each output channel gets its own scaling factor

See examples:

* `Weights Preparation with Per-output-channel Scaling <#weights-preparation-with-per-output-channel-scaling>`__

* `Convolution with Per-output-channel Quantization <#convolution-with-per-output-channel-quantization>`__

* :ref:`MatMul Tutorial: INT8 Inference <doxid-inference_int8_matmul_cpp>`

Block Scaling
*************

Groups enable block-wise quantization by subdividing tensor dimensions into smaller blocks, each with its own scaling factor. This might help balance accuracy and efficiency by providing more granular quantization than per-tensor scaling.

.. ref-code-block:: cpp

	// Weight shape: [K, N] = [1024, 512] with groups [32, 1]
	// Creates 32 groups along K dimension, each with its own scaling factor per N value
	std::vector<dnnl::memory::dim_t> groups = {32, 1};
	attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, (1 << 0) + (1 << 1), groups,
	                :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`);
	
	// Tensor: [K, N] = [1024, 512]
	// Scaling factors: 32 × 512 = 16,384 values (one per group)
	// Usage: Each (group_k, n) combination gets its own scaling factor

See examples:

* `Matmul with Advanced Quantization <#matmul-with-advanced-quantization>`__

* `Matmul with Precomputed Reductions and Advanced Quantization <#matmul-with-precomputed-reductions-and-advanced-quantization>`__

* :ref:`MatMul Tutorial: Weight-only Quantization <doxid-matmul_with_weight_only_quantization_cpp>`

Special Case: MX-compatible Block Scaling (or Dynamic Quantization)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MX-compatible block scaling uses ``e8m0`` data type for scaling factors and ``dynamic_mx`` quantization mode to align with the `OCP MX Formats Specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__.

.. ref-code-block:: cpp

	// Set MX-compatible block scaling for weights
	attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, 1 << 0, {32}, :ref:`dnnl::memory::data_type::e8m0 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea8af1e244959fd40c752655b5d39980eb>`,
	                false /*on device*/, :ref:`dnnl::quantization_mode::dynamic_mx <doxid-group__dnnl__api__attributes_1gga43df4b809a4544d34bbc106d3e409b2ca10dabb84b08ade6e41ee83eba1e96f9d>`);
	
	// Tensor: [K, N] = [1024, 512]
	// Scaling factors: 32 values (one per group of 32 in K dimension)
	// Usage: Each group of 32 in K dimension gets its own scaling factor

See example :ref:`MatMul Tutorial: MXFP8 Inference <doxid-mxfp_matmul_cpp>`.

Multi-Dimensional Scaling
*************************

Multi-dimensional scaling applies scaling factors across multiple tensor dimensions simultaneously.

For scaling factors per dimensions :math:`d_i`, set ``mask =`` :math:`\sum_{d_i} 2^{d_i}`.

Resulting scaling factor count without groups: :math:`\prod_{d_i} D_{d_i}`, with groups: :math:`\prod_{d_i} G_{d_i}`.

.. ref-code-block:: cpp

	// Scaling factors vary along batch and channel dimensions
	attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, (1 << 0) + (1 << 1), {},
	                :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`, false);
	
	// Tensor: [N, C, H, W] = [8, 64, 32, 32]
	// Scaling factors needed: 8 * 64 = 512 values
	// Usage: Each (batch, channel) combination gets its own scaling factor

See examples:

* `Matmul with Advanced Quantization <#matmul-with-advanced-quantization>`__

* `Matmul with Precomputed Reductions and Advanced Quantization <#matmul-with-precomputed-reductions-and-advanced-quantization>`__

* :ref:`MatMul Tutorial: Weight-only Quantization <doxid-matmul_with_weight_only_quantization_cpp>`

:target:`doxid-dev_guide_attributes_quantization_1dgaq_zps`

Argument Zero-Points
--------------------

Zero-points handle the quantization case where the quantized integer range does not center around zero.

The library uses :ref:`Primitive Attributes <doxid-dev_guide_attributes>` API for setting zero-points for most primitives. The supporting attributes can be found in the documentation for each primitive. The unsupported cases are handled according to the :ref:`attributes error handling section <doxid-dev_guide_attributes_1dev_guide_attributes_error_handling>`.

Available Zero-Point API Methods
++++++++++++++++++++++++++++++++

oneDNN provides the following methods for setting zero-points:

.. ref-code-block:: cpp

	// Legacy method with simple mask-based zero-points
	void :ref:`dnnl::primitive_attr::set_zero_points_mask <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`(int arg, int mask);
	
	// Generic method with groups support
	void :ref:`dnnl::primitive_attr::set_zero_points <doxid-structdnnl_1_1primitive__attr_1a2a8693f2aba0541ccd59470b41321175>`(int arg, int mask,
	                                          const :ref:`dnnl::memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` &groups,
	                                          :ref:`dnnl::memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`dnnl::memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`,
	                                          bool is_on_host = false);
	
	// Convenience method for single host-side scalar
	void :ref:`dnnl::primitive_attr::set_host_zero_point <doxid-structdnnl_1_1primitive__attr_1ac6aac2aa4418da036964baa3a35ed879>`(int arg,
	                                              :ref:`dnnl::memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`dnnl::memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`);

Key parameters of the zero-point API methods are summarized below:

===============  ===========================================================  =================================================================  
Parameter        Options*                                                     Description                                                        
===============  ===========================================================  =================================================================  
``arg``          ``DNNL_ARG_SRC`` , ``DNNL_ARG_WEIGHTS`` , ``DNNL_ARG_DST``   Tensor to apply zero-point                                         
``mask``         ``0`` , ``1<<dim`` , ``(1<<d1)+(1<<d2)``                     Zero-point granularity: global, per-dimension, multi-dimensional   
``groups``       ``{}`` , ``{G}`` , ``{G1,G2,...}``                           Block quantization: none, single-size, multi-dimensional blocks    
``data_type``    ``s32`` , ``s8`` , ``u8`` , ``s4`` , ``u4``                  Zero-point data type                                               
``is_on_host``   ``true`` / ``false``                                         Host vs device memory location of zero-point                       
===============  ===========================================================  =================================================================

(\*) Support for quantization options varies based on individual primitive and target hardware. Refer to primitives documentation for the details.

Supported Zero-Point Granularity Levels
+++++++++++++++++++++++++++++++++++++++

Zero-point granularity mirrors the scaling factor granularity described above. The same mask and groups concepts apply:

* Per-tensor zero-point (``mask=0``): Single zero-point for entire tensor

* Per-channel zero-points (``mask=1<<dim``): Different zero-points per channel

* Block zero-points (``mask`` with ``groups``): Block-wise zero-points

* Multi-dimensional zero-points (``mask=(1<<dim1)+(1<<dim2)``): Independent zero-points across multiple dimensions

.. ref-code-block:: cpp

	// Per-tensor zero-point
	attr.:ref:`set_zero_points <doxid-structdnnl_1_1primitive__attr_1a2a8693f2aba0541ccd59470b41321175>`(:ref:`DNNL_ARG_SRC <doxid-group__dnnl__api__primitives__common_1gac37ad67b48edeb9e742af0e50b70fe09>`, 0, {}, :ref:`dnnl::memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`);
	
	// Per-channel zero-points
	attr.:ref:`set_zero_points <doxid-structdnnl_1_1primitive__attr_1a2a8693f2aba0541ccd59470b41321175>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, 1 << 0, {}, :ref:`dnnl::memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`);
	
	// Block zero-points
	std::vector<dnnl::memory::dim_t> groups = {64, 1};
	attr.:ref:`set_zero_points <doxid-structdnnl_1_1primitive__attr_1a2a8693f2aba0541ccd59470b41321175>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, (1 << 0) + (1 << 1), groups,
	                     :ref:`dnnl::memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`);

See examples:

* `Convolution with Per-output-channel Quantization <#convolution-with-per-output-channel-quantization>`__

* `Matmul with Precomputed Reductions and Advanced Quantization <#matmul-with-precomputed-reductions-and-advanced-quantization>`__

* :ref:`MatMul Tutorial: INT8 Inference <doxid-inference_int8_matmul_cpp>`

* :ref:`MatMul Tutorial: Weight-only Quantization <doxid-matmul_with_weight_only_quantization_cpp>`

:target:`doxid-dev_guide_attributes_quantization_1host-side-scalars-and-zero-points`

Special Case: Host-side Scalar Scaling Factor and Zero-point
------------------------------------------------------------

When using the GPU engine and per-tensor quantization, host-side scaling factor and zero-point are supported to reduce copying of data from host to device. A memory object for scaling factor or zero-point value should be created as a host-side scalar (see :ref:`Host-Side Scalars Support <doxid-dev_guide_host_side_scalars>` for details) and passed to the primitive execution function.

The host scaling factor or zero-point attributes could also be set using the following convenience API:

.. ref-code-block:: cpp

	:ref:`dnnl::primitive_attr <doxid-structdnnl_1_1primitive__attr>` attr;
	attr.:ref:`set_host_scale <doxid-structdnnl_1_1primitive__attr_1a7b035390cde177453afae9c5b5a7c29e>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`,
	           :ref:`dnnl::memory::data_type::f32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea512dc597be7ae761876315165dc8bd2e>`);
	
	attr.:ref:`set_host_zero_point <doxid-structdnnl_1_1primitive__attr_1ac6aac2aa4418da036964baa3a35ed879>`(:ref:`DNNL_ARG_DST <doxid-group__dnnl__api__primitives__common_1ga3ca217e4a06d42a0ede3c018383c388f>`,
	           :ref:`dnnl::memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`);

See examples:

* :ref:`MatMul with Host Scalar Scale example <doxid-matmul_with_host_scalar_scale_cpp>`

:target:`doxid-dev_guide_attributes_quantization_1dgaq_precomputed_reductions`

Precomputed Reductions
----------------------

Precomputed reductions could help optimize performance for Large Language Models (LLM).

When using block-wise zero-points for quantized weights, the library must compute reductions over the source tensor during matrix multiplication. This involves summing source tensor values across groups along the reduction dimension:

.. math::

	\dst_{m,n}=\sum_{g=0}^{G-1}\sum_{k={K\over{G}}g}^{{K\over{G}}(g+1)-1}{\src_{m,k}(\weights_{k,n}-zp_{\weights}(g,n))}=\sum_{k=0}^{K-1}{\src_{m,k}\weights_{k,n}}-\sum_{g=0}^{G-1}zp_{\weights}(g,n)\underbrace{\sum_{k={K\over{G}}g}^{{K\over{G}}(g+1)-1}\src_{m,k}}_{R_{m,g}}

where ``R`` represents the precomputed reductions that can be calculated externally when quantizing the source tensor, therefore removing the need for the library to compute them at runtime.

The library uses :ref:`Primitive Attributes <doxid-dev_guide_attributes>` API for setting precomputed reductions. The supporting attributes can be found in the documentation for each primitive. The unsupported cases are handled according to the :ref:`attributes error handling section <doxid-dev_guide_attributes_1dev_guide_attributes_error_handling>`.

Available Precomputed Reductions API Method
+++++++++++++++++++++++++++++++++++++++++++

oneDNN provides the following method for setting precomputed reductions:

.. ref-code-block:: cpp

	void :ref:`dnnl::primitive_attr::set_precomputed_reductions <doxid-structdnnl_1_1primitive__attr_1a24a349d345ac97756a54b01b634b1b3c>`(int arg, int mask,
	        const :ref:`dnnl::memory::dims <doxid-structdnnl_1_1memory_1a7d9f4b6ad8caf3969f436cd9ff27e9bb>` &groups,
	        :ref:`dnnl::memory::data_type <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dce>` data_type = :ref:`dnnl::memory::data_type::s32 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa860868d23f3a68323a2e3f6563d7f31>`);

Key parameters of the precomputed reductions API method are summarized below:

==============  =========================================  ================================================================  
Parameter       Options*                                   Description                                                       
==============  =========================================  ================================================================  
``arg``         ``DNNL_ARG_SRC``                           Tensor to apply precomputed reductions                            
``mask``        ``0`` , ``1<<dim`` , ``(1<<d1)+(1<<d2)``   Reduction granularity: global, per-dimension, multi-dimensional   
``groups``      ``{}`` , ``{G}`` , ``{G1,G2,...}``         Block quantization: none, single-size, multi-dimensional blocks   
``data_type``   ``s32``                                    Reduction data type                                               
==============  =========================================  ================================================================

.. note:: 

   The following limitations apply when using precomputed reductions:
   
   * Requires weight zero-points: Cannot be used without weights zero-points specified.
   
   * Full matrix mask required: Must have full A matrix mask, meaning broadcast is not supported.
   
   
(\*) Support for quantization options varies based on individual primitive and target hardware. Refer to primitives documentation for the details.

See examples:

* `Matmul with Precomputed Reductions and Advanced Quantization <#matrix-multiplication-with-precomputed-reductions-and-advanced-quantization>`__

Quantization Workflows Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Breakdown of Convolution with INT8 Quantization
-----------------------------------------------

Consider a convolution with bias. The tensors are represented as:

* :math:`\src_{f32}[:] = scale_{\src} \cdot (\src_{int8}[:] - zp_{\src})`

* :math:`\weights_{f32}[:] = scale_{\weights} \cdot \weights_{int8}[:]`

* :math:`\dst_{f32}[:] = scale_{\dst} \cdot (\dst_{int8}[:] - zp_{\dst})`

Here the :math:`\src_{f32}, \weights_{f32}, \dst_{f32}` are not computed at all, the whole work happens with int8 tensors. So the task is to compute the :math:`\dst_{int8}` tensor, using the :math:`\src_{int8}`, :math:`\weights_{int8}` tensors passed at execution time, as well as the corresponding quantization parameters :math:`scale_{\src}`, :math:`scale_{\weights}`, :math:`scale_{\dst}`, and :math:`zp_{\src}`, :math:`zp_{\dst}`. Mathematically, the computations are:

.. math::

	\dst_{int8}[:] = \operatorname{f32\_to\_int8}( (scale_{\src} \cdot scale_{\weights} \cdot \operatorname{s32\_to\_f32}(conv_{s32}(\src_{int8}, \weights_{int8}) - zp_{\src} \cdot comp_{s32}) + bias_{f32}) / scale_{\dst} + zp_{\dst} )

where

* :math:`\operatorname{conv}_{s32}` is just a regular convolution which takes source and weights with int8 data type and compute the result in int32 data type (int32 is chosen to avoid overflows during the computations);

* :math:`comp_{s32}` is a compensation term to account for :math:`\src` non-zero zero-point. This term is computed by the oneDNN library and can typically be pre-computed ahead of time, for example during weights reorder.

* :math:`\operatorname{f32\_to\_s8}()` converts an ``f32`` value to ``s8`` with potential saturation if the values are out of the range of the int8 data type.

* :math:`\operatorname{s32\_to\_f32}()` converts an ``int8`` value to ``f32`` with potential rounding. This conversion is typically necessary to apply ``f32`` scaling factors.

Per-Channel Scaling Specifics
+++++++++++++++++++++++++++++

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

Weights Preparation with Per-output-channel Scaling
+++++++++++++++++++++++++++++++++++++++++++++++++++

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
	   :ref:`dnnl::convolution_forward::primitive_desc <doxid-structdnnl_1_1convolution__forward_1_1primitive__desc>` conv_pd(/* see the convolution workflow section */);
	
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

Convolution with Per-output-channel Quantization
++++++++++++++++++++++++++++++++++++++++++++++++

Building upon the weights preparation shown above, this section shows the complete workflow for an int8 convolution that combines per-output-channel weight scaling with global source and destination scaling.

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

Matrix Multiplication with Weight-only Quantization (WoQ)
---------------------------------------------------------

This example describes a process of weight-only quantization (WoQ) in matmul primitive which may be found when running Large Language Models (LLM). The advanced quantization here implies additional grouping introduced over reduction dimension besides traditional per-N quantization.

Weight-only quantization (WoQ) is the runtime process of converting integer weights back to floating-point format during computations. The primitive dequantizes weights using provided scales and zero-points, and converts them to the computation precision specified by :ref:`dnnl::primitive_attr::set_fpmath_mode <doxid-structdnnl_1_1primitive__attr_1ab00639157a283596834ee5b0e8478a2d>`. See :ref:`Floating-point Math Mode <doxid-dev_guide_attributes_fpmath_mode>` for details, and the code snippet below for an example of setting fpmath mode. For a full tutorial, refer to :ref:`MatMul Tutorial: Weight-only Quantization <doxid-matmul_with_weight_only_quantization_cpp>`.

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
	
	   attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_mask, wei_groups, :ref:`dnnl::memory::data_type::f16 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa2449b6477c1fef79be4202906486876>`);
	
	   // Additionally, to instruct the library to perform weights dequantization,
	   // fpmath mode must be set with a flag set to `true`:
	   attr.:ref:`set_fpmath_mode <doxid-structdnnl_1_1primitive__attr_1ab00639157a283596834ee5b0e8478a2d>`(:ref:`dnnl::fpmath_mode::f16 <doxid-group__dnnl__api__fpmath__mode_1gga0ad94cbef13dce222933422bfdcfa725aa2449b6477c1fef79be4202906486876>`, /* apply_to_int = */ true);
	
	   // create a matmul primitive descriptor
	   auto matmul_pd = :ref:`dnnl::matmul::primitive_desc <doxid-structdnnl_1_1matmul_1_1primitive__desc>`(
	           engine,
	           src_f16_any_md,
	           wei_s8_any_md,
	           dst_f16_any_md,
	           attr);   // the attributes describe the quantization flow
	// ...

Matrix Multiplication with Precomputed Reductions and Advanced Quantization
---------------------------------------------------------------------------

This example extends the `Weight-only Quantization <#matrix-multiplication-with-weight-only-quantization-woq>`__ workflow by adding asymmetric weight quantization and external precomputed reductions.

This scenario occurs when quantizing the source tensor at runtime on the application-side, while passing both quantized source and weights to the library.

Precomputed reductions are important when using ``s8`` zero-points for weights, as applying them during computations would cause accuracy loss.

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
	
	   attr.:ref:`set_scales <doxid-structdnnl_1_1primitive__attr_1a35152c75fb7d3e44aa7e51e9ac25c3b0>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_mask, wei_scales_groups,
	           :ref:`dnnl::memory::data_type::f16 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dceaa2449b6477c1fef79be4202906486876>`);
	
	   // Zero-points would have the same mask as grouping applies for them as well.
	   // For example, let it use the different size of the group.
	   std::vector<dim_t> wei_zp_groups = {64, 1};
	
	   // The zero-point factors for quantized weights (as declared above)
	   // A unique zero-point for each zp_gK (256 / 64 = 4) times N, total 2048
	   // elements.
	   std::vector<half> wei_zps(zp_gK, N) = {...};
	
	   attr.:ref:`set_zero_points <doxid-structdnnl_1_1primitive__attr_1a2a8693f2aba0541ccd59470b41321175>`(:ref:`DNNL_ARG_WEIGHTS <doxid-group__dnnl__api__primitives__common_1gaf279f28c59a807e71a70c719db56c5b3>`, wei_mask, wei_zp_groups,
	           :ref:`dnnl::memory::data_type::s8 <doxid-structdnnl_1_1memory_1a8e83474ec3a50e08e37af76c8c075dcea3e8d88fdd85d7153525e0647cdd97686>`);
	
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

