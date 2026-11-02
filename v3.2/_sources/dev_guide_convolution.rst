.. index:: pair: page; Convolution
.. _doxid-dev_guide_convolution:

Convolution
===========

:ref:`API Reference <doxid-group__dnnl__api__convolution>`

General
~~~~~~~

The convolution primitive computes forward, backward, or weight update for a batched convolution operation on 1D, 2D, or 3D spatial data with bias.

The convolution operation is defined by the following formulas. We show formulas only for 2D spatial data which are straightforward to generalize to cases of higher and lower dimensions. Variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`.

.. note:: 

   Mathematical operation commonly called "convolution" in the context of deep learning workloads is actually cross-correlation.
   
   
Let :math:`\src`, :math:`\weights` and :math:`\dst` be :math:`N \times IC \times IH \times IW`, :math:`OC \times IC \times KH \times KW`, and :math:`N \times OC \times OH \times OW` tensors respectively. Let :math:`\bias` be a 1D tensor with :math:`OC` elements.

Furthermore, let the remaining convolution parameters be:

=================================  =============  =============  =============  =============================================================================================================================  
Parameter                          Depth          Height         Width          Comment                                                                                                                        
=================================  =============  =============  =============  =============================================================================================================================  
Padding: Front, top, and left      :math:`PD_L`   :math:`PH_L`   :math:`PW_L`   In the API we use ``padding_l`` to indicate the corresponding vector of paddings ( ``_l`` in the name stands for **left** )    
Padding: Back, bottom, and right   :math:`PD_R`   :math:`PH_R`   :math:`PW_R`   In the API we use ``padding_r`` to indicate the corresponding vector of paddings ( ``_r`` in the name stands for **right** )   
Stride                             :math:`SD`     :math:`SH`     :math:`SW`     Convolution without strides is defined by setting the stride parameters to 1                                                   
Dilation                           :math:`DD`     :math:`DH`     :math:`DW`     Non-dilated convolution is defined by setting the dilation parameters to 0                                                     
=================================  =============  =============  =============  =============================================================================================================================

The following formulas show how oneDNN computes convolutions. They are broken down into several types to simplify the exposition, but in reality the convolution types can be combined.

To further simplify the formulas, we assume that :math:`\src(n, ic, ih, iw) = 0` if :math:`ih < 0`, or :math:`ih \geq IH`, or :math:`iw < 0`, or :math:`iw \geq IW`.

Forward
-------

Regular Convolution
+++++++++++++++++++

.. math::

	\dst(n, oc, oh, ow) = \bias(oc) \\ + \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1} \src(n, ic, oh \cdot SH + kh - PH_L, ow \cdot SW + kw - PW_L) \cdot \weights(oc, ic, kh, kw).

Here:

* :math:`OH = \left\lfloor{\frac{IH - KH + PH_L + PH_R}{SH}} \right\rfloor + 1,`

* :math:`OW = \left\lfloor{\frac{IW - KW + PW_L + PW_R}{SW}} \right\rfloor + 1.`

Convolution with Groups
+++++++++++++++++++++++

In the API, oneDNN adds a separate groups dimension to memory objects representing :math:`\weights` tensors and represents weights as :math:`G \times OC_G \times IC_G \times KH \times KW` 5D tensors for 2D convolutions with groups.

.. math::

	\dst(n, g \cdot OC_G + oc_g, oh, ow) = \bias(g \cdot OC_G + oc_g) \\ + \sum_{ic_g=0}^{IC_G-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1} \src(n, g \cdot IC_G + ic_g, oh \cdot SH + kh - PH_L, ow \cdot SW + kw - PW_L) \cdot \weights(g, oc_g, ic_g, kh, kw),

where

* :math:`IC_G = \frac{IC}{G}`,

* :math:`OC_G = \frac{OC}{G}`, and

* :math:`oc_g \in [0, OC_G).`

The case when :math:`OC_G = IC_G = 1` is also known as a depthwise convolution.

Convolution with Dilation
+++++++++++++++++++++++++

.. math::

	\dst(n, oc, oh, ow) = \bias(oc) \\ + \sum_{ic=0}^{IC-1}\sum_{kh=0}^{KH-1}\sum_{kw=0}^{KW-1} \src(n, ic, oh \cdot SH + kh \cdot (DH + 1) - PH_L, ow \cdot SW + kw \cdot (DW + 1) - PW_L) \cdot \weights(oc, ic, kh, kw).

Here:

* :math:`OH = \left\lfloor{\frac{IH - DKH + PH_L + PH_R}{SH}} \right\rfloor + 1,` where :math:`DKH = 1 + (KH - 1) \cdot (DH + 1)`, and

* :math:`OW = \left\lfloor{\frac{IW - DKW + PW_L + PW_R}{SW}} \right\rfloor + 1,` where :math:`DKW = 1 + (KW - 1) \cdot (DW + 1)`.

Deconvolution (Transposed Convolution)
++++++++++++++++++++++++++++++++++++++

Deconvolutions (also called fractionally strided convolutions or transposed convolutions) work by swapping the forward and backward passes of a convolution. One way to put it is to note that the weights define a convolution, but whether it is a direct convolution or a transposed convolution is determined by how the forward and backward passes are computed.

Difference Between Forward Training and Forward Inference
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There is no difference between the :ref:`dnnl_forward_training <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a992e03bebfe623ac876b3636333bbce0>` and :ref:`dnnl_forward_inference <doxid-group__dnnl__api__primitives__common_1ggae3c1f22ae55645782923fbfd8b07d0c4a2f77a568a675dec649eb0450c997856d>` propagation kinds.

Backward
--------

The backward propagation computes :math:`\diffsrc` based on :math:`\diffdst` and :math:`\weights`.

The weights update computes :math:`\diffweights` and :math:`\diffbias` based on :math:`\diffdst` and :math:`\src`.

.. note:: 

   The optimized memory formats :math:`\src` and :math:`\weights` might be different on forward propagation, backward propagation, and weights update.
   
   


Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  =================================================================================================================================================================  
Primitive input/output          Execution argument index                                                                                                                                           
==============================  =================================================================================================================================================================  
:math:`\src`                    DNNL_ARG_SRC                                                                                                                                                       
:math:`\weights`                DNNL_ARG_WEIGHTS                                                                                                                                                   
:math:`\bias`                   DNNL_ARG_BIAS                                                                                                                                                      
:math:`\dst`                    DNNL_ARG_DST                                                                                                                                                       
:math:`\diffsrc`                DNNL_ARG_DIFF_SRC                                                                                                                                                  
:math:`\diffweights`            DNNL_ARG_DIFF_WEIGHTS                                                                                                                                              
:math:`\diffbias`               DNNL_ARG_DIFF_BIAS                                                                                                                                                 
:math:`\diffdst`                DNNL_ARG_DIFF_DST                                                                                                                                                  
:math:`depthwise`               DNNL_ARG_ATTR_POST_OP_DW                                                                                                                                           
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1   
==============================  =================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

N/A.

Data Types
----------

Convolution primitive supports the following combination of data types for source, destination, and weights memory objects:

===============  ==========  ==========  ============================  ============================  
Propagation      Source      Weights     Destination                   Bias                          
===============  ==========  ==========  ============================  ============================  
forward          f32         f32         f32, u8, s8                   f32                           
forward          f16         f16         f16, f32, u8, s8              f16, f32                      
forward          u8, s8      s8          u8, s8, s32, f32, f16, bf16   u8, s8, s32, f32, f16, bf16   
forward          bf16        bf16        f32, bf16                     f32, bf16                     
forward          f64         f64         f64                           f64                           
backward         f32, bf16   bf16        bf16                                                        
backward         f32, f16    f16         f16                                                         
backward         f32         f32         f32                           f32                           
backward         f64         f64         f64                           f64                           
weights update   bf16        f32, bf16   bf16, s8, u8                  f32, bf16                     
weights update   f16         f32, f16    f16                           f32, f16                      
===============  ==========  ==========  ============================  ============================

.. warning:: 

   There might be hardware and/or implementation specific restrictions. Check :ref:`Implementation Limitations <doxid-dev_guide_convolution_1dg_conv_impl_limits>` section below.
   
   


Data Representation
-------------------

Like other CNN primitives, the convolution primitive expects the following tensors:

========  ==============================================  ===============================================================  
Spatial   Source / Destination                            Weights                                                          
========  ==============================================  ===============================================================  
1D        :math:`N \times C \times W`                     :math:`[G \times ] OC \times IC \times KW`                       
2D        :math:`N \times C \times H \times W`            :math:`[G \times ] OC \times IC \times KH \times KW`             
3D        :math:`N \times C \times D \times H \times W`   :math:`[G \times ] OC \times IC \times KD \times KH \times KW`   
========  ==============================================  ===============================================================

Physical format of data and weights memory objects is critical for convolution primitive performance. In the oneDNN programming model, convolution is one of the few primitives that support the placeholder memory format tag :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` (shortened to ``any`` from now on) and can define data and weight memory objects format based on the primitive parameters. When using ``any`` it is necessary to first create a convolution primitive descriptor and then query it for the actual data and weight memory objects formats.

While convolution primitives can be created with memory formats specified explicitly, the performance may be suboptimal. The table below shows the combinations of memory formats the convolution primitive is optimized for.

============================================================================================================================================================================================================================================================================================================================================================================  ============================================================================================================================================================================================================================================================================================================================================================================  ===========================================  
Source / Destination                                                                                                                                                                                                                                                                                                                                                          Weights                                                                                                                                                                                                                                                                                                                                                                       Limitations                                  
============================================================================================================================================================================================================================================================================================================================================================================  ============================================================================================================================================================================================================================================================================================================================================================================  ===========================================  
``any``                                                                                                                                                                                                                                                                                                                                                                       ``any``                                                                                                                                                                                                                                                                                                                                                                       N/A                                          
:ref:`dnnl_nwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>` , :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` , :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>`   ``any``                                                                                                                                                                                                                                                                                                                                                                       N/A                                          
:ref:`dnnl_nwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da9f756dbdc1e949646c95f83e0f51bc43>` , :ref:`dnnl_nhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae50c534446b3c18cc018b3946b3cebd7>` , :ref:`dnnl_ndhwc <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091daa0d8b24eefd029e214080d3787114fc2>`   :ref:`dnnl_wio <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da93eecc25f8ab1b07604b632401aa28e5>` , :ref:`dnnl_hwio <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da4f4c7bd98c6d53fb3b69e1c8df0a80f6>` , :ref:`dnnl_dhwio <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae4885779f955beeddc25443a3f8c2a63>`   Only on GPUs with Xe-HPC architecture only   
:ref:`dnnl_ncw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dab55cb1d54480dd7f796bf66eea3ad32f>` , :ref:`dnnl_nchw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091da83a751aedeb59613312339d0f8b90f54>` , :ref:`dnnl_ncdhw <doxid-group__dnnl__api__memory_1gga395e42b594683adb25ed2d842bb3091dae33b8c6790e5d37324f18a019658d464>`   ``any``                                                                                                                                                                                                                                                                                                                                                                       Only on CPU                                  
============================================================================================================================================================================================================================================================================================================================================================================  ============================================================================================================================================================================================================================================================================================================================================================================  ===========================================

Post-ops and Attributes
-----------------------

Post-ops and attributes enable you to modify the behavior of the convolution primitive by applying the output scale to the result of the primitive and by chaining certain operations after the primitive. The following attributes and post-ops are supported:

============  ==========  ============================================================================================  ===========================================================================================  =============================================================================================================  
Propagation   Type        Operation                                                                                     Description                                                                                  Restrictions                                                                                                   
============  ==========  ============================================================================================  ===========================================================================================  =============================================================================================================  
forward       attribute   :ref:`Scale <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`         Scales the result of convolution by given scale factor(s)                                    int8 convolutions only                                                                                         
forward       attribute   :ref:`Zero points <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`   Sets zero point(s) for the corresponding tensors                                             int8 convolutions only                                                                                         
forward       post-op     :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`             Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result                                                                                                                         
forward       post-op     :ref:`Sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`                 Adds the operation result to the destination tensor instead of overwriting it                                                                                                                               
forward       post-op     :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`              Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result             General binary post-op restrictions                                                                            
forward       post-op     :ref:`Depthwise <doxid-structdnnl_1_1post__ops_1a55aad3b45a25087e0045a005384bde3a>`           Applies a :ref:`Convolution <doxid-group__dnnl__api__convolution>` operation to the result   See :ref:`a separate section <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_depthwise>`   
============  ==========  ============================================================================================  ===========================================================================================  =============================================================================================================

The following masks are supported by the primitive:

* 0, which applies one zero point value to an entire tensor, and

* 2, which applies a zero point value per each element in a ``IC`` or ``OC`` dimension for ``DNNL_ARG_SRC`` or ``DNNL_ARG_DST`` arguments respectively.

When scales and/or zero-points masks are specified, the user must provide the corresponding scales and/or zero-points as additional input memory objects with argument ``DNNL_ARG_ATTR_SCALES | DNNL_ARG_${MEMORY_INDEX}`` or ``DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_${MEMORY_INDEX}`` during the execution stage. For instance, a source tensor zero points memory argument would be passed with index (``DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC``).

.. note:: 

   The library does not prevent using post-ops in training, but note that not all post-ops are feasible for training usage. For instance, using ReLU with non-zero negative slope parameter as a post-op would not produce an additional output ``workspace`` that is required to compute backward propagation correctly. Hence, in this particular case one should use separate convolution and eltwise primitives for training.
   
   
The library supports any number and order of post operations, but only the following sequences deploy optimized code:

==============================  =============================================  
Type of convolutions            Post-ops sequence supported                    
==============================  =============================================  
f32, bf16 and f16 convolution   eltwise, sum, sum -> eltwise                   
int8 convolution                eltwise, sum, sum -> eltwise, eltwise -> sum   
==============================  =============================================

The operations during attributes and post-ops applying are done in single precision floating point data type. The conversion to the actual destination data type happens just before the actual storing.

Example 1
+++++++++

Consider the following pseudo-code:

.. ref-code-block:: cpp

	primitive_attr attr;
	attr.set_output_scale(mask=0, alpha);
	attr.set_post_ops({
	        { sum={scale=beta} },
	        { eltwise={scale=gamma, type=tanh, alpha=ignore, beta=ignored } }
	    });
	
	convolution_forward(src, weights, dst, attr);

The would lead to the following:

.. math::

	\dst(\overline{x}) = \gamma \cdot \tanh \left( \alpha \cdot conv(\src, \weights) + \beta \cdot \dst(\overline{x}) \right)

Example 2
+++++++++

The following pseudo-code:

.. ref-code-block:: cpp

	primitive_attr attr;
	attr.set_output_scale(mask=0, alpha);
	attr.set_post_ops({
	        { eltwise={scale=gamma, type=relu, alpha=eta, beta=ignored } },
	        { sum={scale=beta} }
	    });
	
	convolution_forward(src, weights, dst, attr);

That would lead to the following:

.. math::

	\dst(\overline{x}) = \beta \cdot \dst(\overline{x}) + \gamma \cdot ReLU \left( \alpha \cdot conv(\src, \weights), \eta \right)

Example 3
+++++++++

The following pseudo-code:

.. ref-code-block:: cpp

	primitive_attr attr;
	attr.set_output_scale(mask=0, alpha);
	attr.set_zero_point(src, mask=0, shift_src);
	attr.set_zero_point(dst, mask=0, shift_dst);
	attr.set_post_ops({
	        { eltwise={scale=gamma, type=relu, alpha=eta, beta=ignored } }
	    });
	
	convolution_forward(src, weights, dst, attr);

That would lead to the following:

.. math::

	\dst(\overline{x}) = \gamma \cdot ReLU \left( \alpha \cdot conv(\src - shift_{src}, \weights), \eta \right) + shift_{dst}

Algorithms
~~~~~~~~~~

oneDNN implements convolution primitives using several different algorithms:

* Direct. The convolution operation is computed directly using SIMD instructions. This is the algorithm used for the most shapes and supports int8, f32, bf16, f16 and f64 (only on GPU engine) data types.

* Winograd. This algorithm reduces computational complexity of convolution at the expense of accuracy loss and additional memory operations. The implementation is based on the `Fast Algorithms for Convolutional Neural Networks by A. Lavin and S. Gray <https://arxiv.org/abs/1509.09308>`__. The Winograd algorithm often results in the best performance, but it is applicable only to particular shapes. Winograd supports GPU (f16 and f32).

* Implicit GEMM. The convolution operation is reinterpreted in terms of matrix-matrix multiplication by rearranging the source data into a :ref:`scratchpad memory <doxid-dev_guide_attributes_scratchpad>`. This is a fallback algorithm that is dispatched automatically when other implementations are not available. GEMM convolution supports the int8, f32, and bf16 data types.

Direct Algorithm
----------------

oneDNN supports the direct convolution algorithm on all supported platforms for the following conditions:

* Data and weights memory formats are defined by the convolution primitive (user passes ``any``).

* The number of channels per group is a multiple of SIMD width for grouped convolutions.

* For each spatial direction padding does not exceed one half of the corresponding dimension of the weights tensor.

* Weights tensor width does not exceed 14.

In case any of these constraints are not met, the implementation will silently fall back to an explicit GEMM algorithm.

:target:`doxid-dev_guide_convolution_1dg_winograd_conv`

Winograd Convolution
--------------------

oneDNN supports the Winograd convolution algorithm only on GPU engine.

The following side effects should be weighed against the (potential) performance boost achieved from using the Winograd algorithm:

* Memory consumption. Winograd implementation in oneDNN requires additional scratchpad memory to store intermediate results. As more convolutions using Winograd are added to the topology, the amount of memory required can grow significantly. This growth can be controlled if the scratchpad memory can be reused across multiple primitives. See :ref:`Primitive Attributes: Scratchpad <doxid-dev_guide_attributes_scratchpad>` for more details.

* Accuracy. In some cases Winograd convolution produce results that are significantly less accurate than results from the direct convolution.

Create a Winograd convolution by simply creating a convolution primitive descriptor (step 6 in :ref:`simple network example <doxid-cnn_inference_f32_cpp>` specifying the Winograd algorithm. The rest of the steps are exactly the same.

.. ref-code-block:: cpp

	auto conv1_pd = convolution_forward::primitive_desc(engine,
	    prop_kind::forward_inference, algorithm::convolution_winograd,
	    conv1_src_md, conv1_weights_md, conv1_bias_md, conv1_dst_md,
	    conv1_strides, conv1_padding_l, conv1_padding_r);

Automatic Algorithm Selection
-----------------------------

oneDNN supports ``:ref:`dnnl::algorithm::convolution_auto <doxid-group__dnnl__api__attributes_1gga00377dd4982333e42e8ae1d09a309640acfdececd63a8bc0cfe1021ad614e2ded>``` algorithm that instructs the library to automatically select the best algorithm based on the heuristics that take into account tensor shapes and the number of logical processors available. (For automatic selection to work as intended, use the same thread affinity settings when creating the convolution as when executing the convolution.)

:target:`doxid-dev_guide_convolution_1dg_conv_impl_limits`

Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Refer to :ref:`Data Types <doxid-dev_guide_data_types>` for limitations related to data types support.

#. See :ref:`Winograd Convolution <doxid-dev_guide_convolution_1dg_winograd_conv>` section for limitations of Winograd algorithm implementations.

#. GPU
   
   * Depthwise post-op is not supported

Performance Tips
~~~~~~~~~~~~~~~~

* Use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for source, weights, and destinations memory format tags when create a convolution primitive to allow the library to choose the most appropriate memory format.

Example
~~~~~~~

:ref:`Convolution Primitive Example <doxid-convolution_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`Convolution <doxid-dev_guide_convolution>` primitive in forward propagation mode in two configurations - with and without groups.

Key optimizations included in this example:

* Creation of optimized memory format from the primitive descriptor;

* Primitive attributes with fused post-ops.

