.. index:: pair: page; Matrix Multiplication
.. _doxid-dev_guide_matmul:

Matrix Multiplication
=====================

:ref:`API Reference <doxid-group__dnnl__api__matmul>`

General
~~~~~~~

The matrix multiplication (MatMul) primitive computes the product of two 2D tensors with optional bias addition (the variable names follow the standard :ref:`Naming Conventions <doxid-dev_guide_conventions>`):

.. math::

	\dst(m, n) = \sum_{k=0}^{K - 1} \left( \src(m, k) \cdot \weights(k, n) \right) + \bias(m, n)

The MatMul primitive also supports batching multiple independent matrix multiplication operations, in which case the tensors can be up to 12D:

.. math::

	\dst(bs_0, bs_1, \ldots, m, n) = \sum_{k=0}^{K - 1} \left( \src(bs_0, bs_1, \ldots, m, k) \cdot \weights(bs_0, bs_1, \ldots, k, n) \right) + \bias(bs_0, bs_1, \ldots, m, n)

MatMul also supports implicit broadcast semantics i.e., :math:`\src` can be broadcasted into :math:`\weights` if the corresponding dimension in :math:`\src` is 1 (and vice versa). However, all tensors (including :math:`\bias`, if it exists) must have the same number of dimensions.

The shape of :math:`\dst` only depends on :math:`\src` and :math:`\weights` tensors. The :math:`\bias` cannot change the dimensions of :math:`\dst` by broadcasting. In other words, for every dimension, the following constraint must hold true: ``dimension(bias) == dimension(dst) || dimension(bias) == 1``.

Execution Arguments
~~~~~~~~~~~~~~~~~~~

When executed, the inputs and outputs should be mapped to an execution argument index as specified by the following table.

==============================  ==================================================================================================================================================================  
Primitive input/output          Execution argument index                                                                                                                                            
==============================  ==================================================================================================================================================================  
:math:`\src`                    DNNL_ARG_SRC                                                                                                                                                        
:math:`\weights`                DNNL_ARG_WEIGHTS                                                                                                                                                    
:math:`\bias`                   DNNL_ARG_BIAS                                                                                                                                                       
:math:`\dst`                    DNNL_ARG_DST                                                                                                                                                        
:math:`\text{binary post-op}`   :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(binary_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_SRC_1    
:math:`\text{prelu post-op}`    :ref:`DNNL_ARG_ATTR_MULTIPLE_POST_OP(prelu_post_op_position) <doxid-group__dnnl__api__primitives__common_1ga30839136bbf81b03a173e0842ae015e1>` | DNNL_ARG_WEIGHTS   
==============================  ==================================================================================================================================================================

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

General Notes
-------------

#. The MatMul primitive supports input and output tensors with run-time specified shapes and memory formats. The run-time specified dimensions or strides are specified using the :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>` wildcard value during the primitive initialization and creation stage. At the execution stage, the user must pass fully specified memory objects so that the primitive is able to perform the computations. Note that the less information about shapes or format is available at the creation stage, the less performant execution will be. In particular, if the shape is not known at creation stage, one cannot use the special format tag :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` to enable an implementation to choose the most appropriate memory format for the corresponding input or output shapes. On the other hand, run-time specified shapes enable users to create a primitive once and use it in different situations.

#. Inconsistency with dimensions being "primitive-creation-time-defined" vs "runtime-defined" is invalid. For example, :math:`\src` and :math:`\weights` with dimensions set to ``{3, 4, 4}`` and ``{DNNL_RUNTIME_DIM_VAL, 4, 4}`` respectively is invalid.

#. The broadcasting shape consistency check is not done for the dimensions with :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`. It is user responsibility to make sure the dimensions for the tensors are valid.

#. Multiple batch dimensions and broadcasting of batch dimensions of ``src`` and ``weights`` are supported for both CPU and GPU engines.
   
   Please check tutorials below to see :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>` support in use.

Data Types
----------

The MatMul primitive supports the following combinations of data types for source, destination, weights, and bias tensors:

===============  ========  ============================  ============================  
Source           Weights   Destination                   Bias                          
===============  ========  ============================  ============================  
f32              f32       f32                           f32                           
f16              f16       f16, u8, s8                   f16, f32                      
bf16             bf16      f32, bf16                     bf16, f32                     
f32, bf16, f16   u8, s8    f32, bf16, f16                f32, bf16, f16                
u8, s8           s8        u8, s8, s32, f32, f16, bf16   u8, s8, s32, f32, f16, bf16   
f8_e5m2          f8_e5m2   f32, f16, bf16, f8_e5m2       f32, bf16, f16                
===============  ========  ============================  ============================

Data Representation
-------------------

The MatMul primitive expects the following tensors:

=====  ====================================  ====================================  ====================================  ===========================================================  
Dims   Source                                Weights                               Destination                           Bias                                                         
=====  ====================================  ====================================  ====================================  ===========================================================  
2D     M :math:`\times` K                    K :math:`\times` N                    M :math:`\times` N                    None or :math:`(M \text{ or } 1) \times (N \text{ or } 1)`   
ND     S :math:`\times` M :math:`\times` K   W :math:`\times` K :math:`\times` N   D :math:`\times` M :math:`\times` N   None or B                                                    
=====  ====================================  ====================================  ====================================  ===========================================================

where for the sake of notational convenience, we have

.. math::

	S = \prod_{i = 0}^{ND - 3} \mathrm{src\_dims}[i], \; W = \prod_{i = 0}^{ND - 3} \mathrm{weights\_dims}[i] \\ D = \prod_{i = 0}^{ND - 3} \mathrm{\dst\_dims}[i], \; B = \prod_{i = 0}^{ND - 1} \left( \mathrm{\dst\_dims}[i] \mbox{ or } 1 \right)

The MatMul primitive is generally optimized for the case in which memory objects use plain memory formats. Additionally, the :math:`\src` and :math:`\weights` must have at least one of the axes ``m`` or ``k`` and ``n`` or ``k`` contiguous (i.e., stride=1) respectively. However, it is recommended to use the placeholder memory format :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` if an input tensor is reused across multiple executions. In this case, the primitive will set the most appropriate memory format for the corresponding input tensor.

The memory format of the destination tensor should always be plain with ``n`` axis contiguous. For example, :ref:`dnnl::memory::format_tag::ab <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa187ef4436122d1cc2f40dc2b92f0eba0>` for the 2D case and :ref:`dnnl::memory::format_tag::abc <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa900150983cd24fb0d6963f7d28e17f72>` or :ref:`dnnl::memory::format_tag::bac <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa79ec16df80b57696a03bb364410061f3>` for the 3D one.

Attributes and Post-ops
-----------------------

Attributes and post-ops enable modifying the behavior of the MatMul primitive. The following attributes and post-ops are supported:

==========  ============================================================================================  ====================================================================================  ====================================  
Type        Operation                                                                                     Description                                                                           Restrictions                          
==========  ============================================================================================  ====================================================================================  ====================================  
Attribute   :ref:`Scales <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`        Scales the result by given scale factor(s)                                                                                  
Attribute   :ref:`Zero-points <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`   Sets zero point(s) for the corresponding tensors                                      Int8 computations only                
Post-op     :ref:`Eltwise <doxid-structdnnl_1_1post__ops_1a60ce0e18ec1ef06006e7d72e7aa865be>`             Applies an :ref:`Eltwise <doxid-group__dnnl__api__eltwise>` operation to the result                                         
Post-op     :ref:`Sum <doxid-structdnnl_1_1post__ops_1a74d080df8502bdeb8895a0443433af8c>`                 Adds the operation result to the destination tensor instead of overwriting it                                               
Post-op     :ref:`Binary <doxid-structdnnl_1_1post__ops_1a40bb2b39a685726ac54873b203be41b5>`              Applies a :ref:`Binary <doxid-group__dnnl__api__binary>` operation to the result      General binary post-op restrictions   
Post-op     :ref:`Prelu <doxid-structdnnl_1_1post__ops_1a1e538118474ac643c6da726a8a658b70>`               Applies an :ref:`PReLU <doxid-group__dnnl__api__prelu>` operation to the result                                             
==========  ============================================================================================  ====================================================================================  ====================================

The following masks are supported by the primitive:

* 0, which applies one scale / zero point value to an entire tensor, and

* 2, which applies a scale value per column along the ``n`` dimension for ``DNNL_ARG_WEIGHTS``.

When scales and/or zero-points masks are specified, the user must provide the corresponding scales and/or zero-points as additional input memory objects with argument ``DNNL_ARG_ATTR_SCALES | DNNL_ARG_${MEMORY_INDEX}`` or ``DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_${MEMORY_INDEX}`` during the execution stage. For instance, a source tensor zero points memory argument would be passed with index (``DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_SRC``).

.. note:: 

   Please check tutorials below to see run-time attributes in use.
   
   


Implementation Limitations
~~~~~~~~~~~~~~~~~~~~~~~~~~

#. Check :ref:`Data Types <doxid-dev_guide_data_types>`.

#. GPU
   
   * Supports up to 6 dimensions.
   
   * Source zero point mask of ``0`` is only supported.
   
   * Sum post-op doesn't support data type other than destination data type.
   
   * Bias of bf16 data type is supported for configuration with bf16 source data type and weights bf16 data type, and up to three dimensional matrices.
   
   * Only reference support is available for f8_e4m3. Optimized implementation for f8_e5m2 is available only on Intel(R) Data Center GPU Max Series.
   
   * Configuration with int8 source data type, s8 weight data type and bf16 destination data type don't support:
     
     * Destination zero point.
     
     * Runtime dimensions.
     
     * Three and higher dimensional matrices.

#. CPU
   
   * Configuration with int8 source data type, s8 weight data type and f16 destination data type isn't supported.
   
   * Configuration with floating point source data type, integer weights data type and floating point destination data type is not optimized.
   
   * Only reference support for fp8 data types (f8_e5m2, f8_e4m3) is is available on CPU.

Performance Tips
~~~~~~~~~~~~~~~~

* Use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>` for either of the input tensors if and only if the shape of the corresponding tensor is fully known at creation time and it is possible to cache reordered tensors across multiple primitive executions. For instance, a good candidate for reuse are the weights tensors during inference: their shapes and data types are known in advance; thus they can be reordered during the first inference pass and can be reused during the subsequent passes. However, if any of the input tensors cannot be reused, it is best to force the primitive to use the same format as that used by the tensors.

Examples
~~~~~~~~

The following examples are available:

Matrix Multiplication Primitive Examples
----------------------------------------

:ref:`MatMul Primitive Example <doxid-matmul_example_cpp>`

This C++ API example demonstrates how to create and execute a :ref:`MatMul <doxid-dev_guide_matmul>` primitive.

Key optimizations included in this example:

* Primitive attributes with fused post-ops.

:ref:`MatMul Tutorial: Comparison with SGEMM <doxid-cpu_sgemm_and_matmul_cpp>` (CPU only)

C++ API example demonstrating :ref:`MatMul <doxid-dev_guide_matmul>` as a replacement for SGEMM functions.

Concepts:

* Create primitive once, use multiple times
  
  * Run-time tensor shapes: :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales_mask() <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`

:ref:`MatMul Tutorial: INT8 Inference <doxid-inference_int8_matmul_cpp>`

C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` fused with ReLU in INT8 inference.

Concepts:

* Asymmetric quantization
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales_mask() <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`
  
  * Zero points: :ref:`dnnl::primitive_attr::set_zero_points_mask() <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`

* :ref:`Operation fusion <doxid-dev_guide_attributes_post_ops>`

* Create primitive once, use multiple times
  
  * Run-time tensor shapes: :ref:`DNNL_RUNTIME_DIM_VAL <doxid-group__dnnl__api__memory_1gaa596c5a6102df77a550bad98f0d5cc12>`

* Weights pre-packing: use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`

:ref:`MatMul Tutorial: Quantization <doxid-cpu_matmul_quantization_cpp>` (CPU only)

C++ API example demonstrating how one can perform reduced precision matrix-matrix multiplication using :ref:`MatMul <doxid-dev_guide_matmul>` and the accuracy of the result compared to the floating point computations.

Concepts:

* Static and dynamic quantization

* Asymmetric quantization
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales_mask() <doxid-structdnnl_1_1primitive__attr_1ac3dc9efa6702a5eba6f289f1b3907590>`
  
  * Zero points: :ref:`dnnl::primitive_attr::set_zero_points_mask() <doxid-structdnnl_1_1primitive__attr_1a8935d36d48fe5db9476b30b02791d822>`

:ref:`MatMul Tutorial: Weights decompression <doxid-weights_decompression_matmul_cpp>` (CPU only)

C++ API example demonstrating how one can use :ref:`MatMul <doxid-dev_guide_matmul>` with compressed weights.

Concepts:

* Asymmetric quantization
  
  * Scales: :ref:`dnnl::primitive_attr::set_scales() <doxid-structdnnl_1_1primitive__attr_1a29e8f33119d42bf7d259eafc6e6548d6>`
  
  * Zero points: :ref:`dnnl::primitive_attr::set_zero_points() <doxid-structdnnl_1_1primitive__attr_1aa7a57b0ba198c418981d41c5289fed8e>`

* :ref:`Operation fusion <doxid-dev_guide_attributes_post_ops>`

* Create primitive once, use multiple times

* Weights pre-packing: use :ref:`dnnl::memory::format_tag::any <doxid-structdnnl_1_1memory_1a8e71077ed6a5f7fb7b3e6e1a5a2ecf3fa100b8cad7cf2a56f6df78f171f97a1ec>`

