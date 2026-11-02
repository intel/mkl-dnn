.. index:: pair: page; Data Types
.. _doxid-dev_guide_data_types:

Data Types
==========

oneDNN functionality supports a number of numerical data types. IEEE single precision floating-point (fp32) is considered to be the golden standard in deep learning applications and is supported in all the library functions. The purpose of low precision data types support is to improve performance of compute intensive operations, such as convolutions, inner product, and recurrent neural network cells in comparison to fp32.

==========  ===========================================================================================================================================================================================  
Data type   Description                                                                                                                                                                                  
==========  ===========================================================================================================================================================================================  
f32         `IEEE single precision floating-point <https://en.wikipedia.org/wiki/Single-precision_floating-point_format#IEEE_754_single-precision_binary_floating-point_format:_binary32>`__             
bf16        `non-IEEE 16-bit floating-point <https://www.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf>`__                                    
f16         `IEEE half precision floating-point <https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16>`__                   
s8/u8       signed/unsigned 8-bit integer                                                                                                                                                                
s4/u4       signed/unsigned 4-bit integer                                                                                                                                                                
s32         signed/unsigned 32-bit integer                                                                                                                                                               
f64         `IEEE double precision floating-point <https://en.wikipedia.org/wiki/Double-precision_floating-point_format#IEEE_754_double-precision_binary_floating-point_format:_binary64>`__             
f8_e5m2     `OFP8 standard 8-bit floating-point <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ with 5 exponent and 2 mantissa bits   
f8_e4m3     `OFP8 standard 8-bit floating-point <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ with 4 exponent and 3 mantissa bits   
e8m0        `MX standard 8-bit scaling type <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__                                                                   
f4_e2m1     `MX standard 4-bit floating-point <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__ with 2 exponent and 1 mantissa bits                             
f4_e3m0     4-bit floating-point with 3 exponent bits and no mantissa bit                                                                                                                                
==========  ===========================================================================================================================================================================================

Inference and Training
~~~~~~~~~~~~~~~~~~~~~~

oneDNN supports training and inference with the following data types:

==========  ==========  ==========  
Data type   Inference   Training    
==========  ==========  ==========  
f64         ``+`` (1)   ``+`` (1)   
f32         ``+``       ``+``       
bf16        ``+``       ``+``       
f16         ``+``       ``+``       
f8_e5m2     ``+``       ``+``       
f8_e4m3     ``+``       ``+``       
s8          ``+``                   
u8          ``+``                   
f4_e2m1     ``+``                   
f4_e3m0                             
s4          ``+`` (2)               
u4          ``+`` (2)               
==========  ==========  ==========

Footnotes:

#. f64 support is limited to matmul, convolution, reorder, layer normalization, and pooling primitives on Intel GPUs.

#. s4/u4 data types are only supported as a storage data type for weights argument in case of weights decompression. For more details, refer to :ref:`Matmul Tutorial: weights decompression <doxid-weights_decompression_matmul_cpp>`.

.. note:: 

   Data type support may also be limited by hardware capabilities. Refer to :ref:`Hardware Limitations <doxid-dev_guide_data_types_1data_types_hardware_limitations>` section below for details.
   
   

.. note:: 

   Using lower precision arithmetic may require changes in the deep learning model implementation.
   
   
See topics for the corresponding data types details:

* :ref:`Int8 Inference <doxid-dev_guide_inference_int8>`

* :ref:`Primitive Attributes: Quantization <doxid-dev_guide_attributes_quantization>`

* :ref:`Bfloat16 Training <doxid-dev_guide_training_bf16>`

* :ref:`Primitive Attributes: floating-point math mode <doxid-dev_guide_attributes_fpmath_mode>`

* :ref:`MatMul Tutorial: weights decompression <doxid-weights_decompression_matmul_cpp>`

Individual primitives may have additional limitations with respect to data type by each primitive is included in the corresponding sections of the developer guide.

General numerical behavior of the oneDNN library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During a primitive computation, oneDNN can use different datatypes than those of the inputs/outputs. In particular, oneDNN uses wider accumulator datatypes (s32 for integral computations, and f32/f64 for floating-point computations), and converts intermediate results to f32 before applying post-ops (f64 configuration does not support post-ops). The following formula governs the datatypes dynamic during a primitive computation:

.. math::

	\operatorname{convert_{dst\_dt}} ( \operatorname{zp_{dst}} + 1/\operatorname{scale_{dst}} * \operatorname{postops_{f32}} (\operatorname{convert_{f32}} (\operatorname{Op}(\operatorname{src_{src\_dt}}, \operatorname{weights_{wei\_dt}}, ...))))

The ``Op`` output datatype depends on the datatype of its inputs:

* if ``src``, ``weights``, ... are floating-point datatype (f32, f16, bf16, f8_e5m2, f8_e4m3, f4_e2m1, f4_e3m0), then the ``Op`` outputs f32 elements.

* if ``src``, ``weights``, ... are integral datatypes (s8, u8, s32), then the ``Op`` outputs s32 elements.

* if the primitive allows to mix input datatypes, the ``Op`` outputs datatype will be s32 if its weights are an integral datatype, or f32 otherwise.

The accumulation datatype used during ``Op`` computation is governed by the ``accumulation_mode`` attribute of the primitive. By default, f32 is used for floating-point primitives (or f64 for f64 primitives) and s32 is used for integral primitives.

No downconversions are allowed by default, but can be enabled using the floating-point math controls described in :ref:`Primitive Attributes: floating-point math mode <doxid-dev_guide_attributes_fpmath_mode>`.

The :math:`convert_{dst\_dt}` conversion is guaranteed to be faithfully rounded but not guaranteed to be correctly rounded (the returned value is not always the closest one but one of the two closest representable value). In particular, some hardware platforms have no direct conversion instructions from f32 data type to low-precision data types such as fp8 or fp4, and will perform conversion through an intermediate data type (for example f16 or bf16), which may result in `double rounding <https://en.wikipedia.org/wiki/Rounding#Double_rounding>`__.

Conversions to integral datatypes saturate upon overflow, whereas conversions to floating-point datatypes don't. To force saturation behavior for floating-point datatypes use :ref:`dev_guide_attributes_post_ops_eltwise <doxid-dev_guide_attributes_post_ops_1dev_guide_attributes_post_ops_eltwise>` with clip algorithm.

Rounding mode and denormal handling
-----------------------------------

oneDNN floating-point computation behavior follows the floating-point environment for the given device runtime by default. In particular, the floating-point environment can control:

* the rounding mode. It is set to round-to-nearest tie-even by default on x64 systems as well as devices running on SYCL and openCL runtime.

* the handling of denormal values. Computation on denormals are not flushed to zero by default. Note denormal handling can negatively impact performance on x64 systems.

.. note:: 

   For CPU devices, the default floating-point environment is defined by the C and C++ standards in the following header:
   
   .. ref-code-block:: cpp
   
   	#include <fenv.h>
   
   Rounding mode can be changed globally using the ``fesetround()`` C function.
   
   

.. note:: 

   Most DNN applications do not require precise computations with denormal numbers and flushing these denormals to zero can improve performance. On x64 systems, the floating-point environment can be updated to allow flushing denormals to zero as follow:
   
   .. ref-code-block:: cpp
   
   	#include <xmmintrin.h>
   	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
   
   

.. note:: 

   On some hardware architectures, low-precision datatype acceleration ignores floating-point environment and will flush denormal outputs to zero (FTZ). In particular this is the case for Intel AMX instruction set.
   
   
oneDNN also exposes non-standard stochastic rounding through the ``rounding_mode`` primitive attribute. More details on this attribute can be found in :ref:`Primitive Attributes: rounding mode <doxid-dev_guide_attributes_rounding_mode>`.

:target:`doxid-dev_guide_data_types_1data_types_hardware_limitations`

Hardware Limitations
~~~~~~~~~~~~~~~~~~~~

While all the platforms oneDNN supports have hardware acceleration for fp32 arithmetics, that is not the case for other data types. Support for low precision data types may not be available for older platforms. The next sections explain limitations that exist for low precision data types for Intel(R) Architecture processors, Intel Processor Graphics and Xe Architecture graphics.

Intel(R) Architecture Processors
--------------------------------

oneDNN performance optimizations for Intel Architecture Processors are specialized based on Instruction Set Architecture (ISA). The following table indicates data types support for every supported ISA:

=====================================================  ====  ======  ==========  ==========  ==========  ======  ========  ======  
ISA                                                    f64   f32     bf16        f16         s8/u8       f8      f4_e2m1   s4/u4   
=====================================================  ====  ======  ==========  ==========  ==========  ======  ========  ======  
Intel SSE4.1                                                 ``+``                                                                 
Intel AVX                                                    ``+``                                                                 
Intel AVX2                                                   ``+``                           ``+`` (1)                             
Intel AVX2 with Intel DL Boost (int8)                        ``+``                           ``+``                                 
Intel AVX-512                                                ``+``   ``.`` (2)               ``+`` (1)                             
Intel AVX-512 with Intel DL Boost (int8)                     ``+``   ``.`` (2)               ``+``                                 
Intel AVX-512 with Intel DL Boost (int8, bf16)               ``+``   ``+``                   ``+``                                 
Intel AVX2 with Intel DL Boost (int8) and NE_CONVERT         ``+``   ``.``       ``.``       ``+``                                 
Intel AVX10.1/512 with Intel AMX (int8, bf16)                ``+``   ``+``       ``.`` (3)   ``+``                         ``.``   
Intel AVX10.1/512 with Intel AMX (int8, bf16, f16)           ``+``   ``+``       ``+``       ``+``       ``.``             ``.``   
Intel AVX10.2                                                ``+``   ``+``       ``+``       ``+``       ``.``             ``.``   
Intel AVX10.2 with Intel AMX (int8, bf16, fp16, fp8)         ``+``   ``+``       ``+``       ``+``       ``+``             ``.``   
=====================================================  ====  ======  ==========  ==========  ==========  ======  ========  ======

Legend:

* ``+`` indicates oneDNN uses hardware-native compute support for this data type.

* ``.`` indicates oneDNN supports this data type via conversion to a higher precision data type.

Footnotes:

#. See :ref:`Nuances of int8 Computations <doxid-dev_guide_int8_computations>` in the Developer Guide for additional limitations related to int8 arithmetic.

#. The library has functional bfloat16 support on processors with Intel AVX-512 Byte and Word Instructions (AVX512BW) support for validation purposes. The performance of bfloat16 primitives on platforms without hardware acceleration for bfloat16 is 3-4x lower in comparison to the same operations on the fp32 data type.

#. Intel AVX-512 f16 instructions accumulate to f16. To avoid overflow, the f16 primitives might up-convert the data to f32 before performing math operations. This can lead to scenarios where a f16 primitive may perform slower than similar f32 primitive.

Intel(R) Processor Graphics and Xe Architecture graphics
--------------------------------------------------------

oneDNN performance optimizations for Intel Processor graphics and Xe Architecture graphics are specialized based on device microarchitecture (uArch). The following uArchs and associated devices have specialized optimizations in the library:

* Xe-LP
  
  * Intel UHD Graphics for 11th-14th Gen Intel(R) Processors
  
  * Intel Iris Xe Graphics
  
  * Intel Iris Xe MAX Graphics (formerly DG1)

* Xe-LPG
  
  * Intel Graphics for Intel Core Ultra processors (formerly Meteor Lake)

* Xe-HPG
  
  * Intel Arc A-Series Graphics (formerly Achemist)
  
  * Intel Data Center GPU Flex Series (formerly Arctic Sound)

* Xe-HPC
  
  * Intel Data Center GPU Max Series (formerly Ponte Vecchio)

* Xe2-LPG
  
  * Intel Graphics for Intel Core Ultra processors (Series 2) (formerly Lunar Lake)

* Xe2-HPG
  
  * Intel Arc B-Series Graphics (formerly Battlemage)

* Xe3-LPG
  
  * Intel Arc Graphics for future Intel Core Ultra processors (code name Panther Lake)

The following table indicates the data types support for each uArch supported by oneDNN.

========  ======  ======  ======  ==========  ======  ======  ========  ======  
ISA       f64     f32     bf16    f16         s8/u8   f8      f4_e2m1   s4/u4   
========  ======  ======  ======  ==========  ======  ======  ========  ======  
Xe-LP             ``+``   ``.``   ``+`` (1)   ``+``                             
Xe-LPG            ``+``   ``.``   ``+`` (1)   ``+``                             
Xe-HPG            ``+``   ``+``   ``+``       ``+``   ``.``             ``.``   
Xe-HPC    ``+``   ``+``   ``+``   ``+``       ``+``   ``.``   ``.``     ``.``   
Xe2-LPG   ``+``   ``+``   ``+``   ``+``       ``+``   ``.``   ``.``     ``.``   
Xe2-HPG   ``+``   ``+``   ``+``   ``+``       ``+``   ``.``   ``.``     ``.``   
Xe3-LPG   ``+``   ``+``   ``+``   ``+``       ``+``   ``.``   ``.``     ``.``   
========  ======  ======  ======  ==========  ======  ======  ========  ======

Legend:

* ``+`` indicates oneDNN uses hardware-native compute support for this data type.

* ``.`` indicates oneDNN supports this data type via conversion to a higher precision data type.

Footnotes:

#. Xe-LP architecture does not natively support f16 operations with f32 accumulation. Consider using :ref:`relaxed accumulation mode <doxid-dev_guide_attributes_accumulation_mode>` for the best performance results.

