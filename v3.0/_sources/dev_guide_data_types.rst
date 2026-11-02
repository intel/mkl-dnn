.. index:: pair: page; Data Types
.. _doxid-dev_guide_data_types:

Data Types
==========

oneDNN functionality supports a number of numerical data types. IEEE single precision floating-point (fp32) is considered to be the golden standard in deep learning applications and is supported in all the library functions. The purpose of low precision data types support is to improve performance of compute intensive operations, such as convolutions, inner product, and recurrent neural network cells in comparison to fp32.

==========  =================================================================================================================================================================================  
Data type   Desc                                                                                                                                                                               
==========  =================================================================================================================================================================================  
f32         `IEEE single precision floating-point <https://en.wikipedia.org/wiki/Single-precision_floating-point_format#IEEE_754_single-precision_binary_floating-point_format:_binary32>`__   
bf16        `non-IEEE 16-bit floating-point <https://software.intel.com/content/www/us/en/develop/download/bfloat16-hardware-numerics-definition.html>`__                                      
f16         `IEEE half precision floating-point <https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16>`__         
s8/u8       signed/unsigned 8-bit integer                                                                                                                                                      
f64         `IEEE double precision floating-point <https://en.wikipedia.org/wiki/Double-precision_floating-point_format#IEEE_754_double-precision_binary_floating-point_format:_binary64>`__   
==========  =================================================================================================================================================================================

Inference and Training
~~~~~~~~~~~~~~~~~~~~~~

oneDNN supports training and inference with the following data types:

===========  ======================  ===========================  
Usage mode   CPU                     GPU                          
===========  ======================  ===========================  
Inference    f32, bf16, f16, s8/u8   f32, bf16, f16, s8/u8, f64   
Training     f32, bf16, f16          f32, bf16, f64               
===========  ======================  ===========================

.. note:: 

   Using lower precision arithmetic may require changes in the deep learning model implementation.
   
   

.. note:: 

   f64 is only supported for convolution primitive, on the GPU engine.
   
   
See topics for the corresponding data types details:

* :ref:`Int8 Inference <doxid-dev_guide_inference_int8>`
  
  * :ref:`Primitive Attributes: Quantization <doxid-dev_guide_attributes_quantization>`

* :ref:`Bfloat16 Training <doxid-dev_guide_training_bf16>`

Individual primitives may have additional limitations with respect to data type by each primitive is included in the corresponding sections of the developer guide.

General numerical behavior of the oneDNN library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During a primitive computation, oneDNN can use different datatypes than those of the inputs/outputs. In particular, oneDNN uses wider accumulator datatypes (s32 for integral computations, and f32 for floating-point computations), and converts intermediate results to f32 before applying post-ops (f64 configuration does not support post-ops). The following formula governs the datatypes dynamic during a primitive computation:

.. math::

	\operatorname{convert_{dst\_dt}} ( \operatorname{dst\_zero\_point_{f32}} + \operatorname{postops_{f32}} (\operatorname{oscale_{f32}} * \operatorname{convert_{f32}} (\operatorname{Op}(\operatorname{src_{src\_dt}}, \operatorname{weights_{wei\_dt}}, ...))))

The ``Op`` output datatype depends on the datatype of its inputs:

* if ``src``, ``weights``, ... are floating-point datatype (f32, f16, bf16), then the ``Op`` outputs f32 elements.

* if ``src``, ``weights``, ... are integral datatypes (s8, u8, s32), then the ``Op`` outputs s32 elements.

* if the primitive allows to mix input datatypes, the ``Op`` outputs datatype will be s32 if its weights are an integral datatype, or f32 otherwise.

No downconversions are allowed by default, but can be enabled using the floating-point math controls described in :ref:`Primitive Attributes: floating-point math mode <doxid-dev_guide_attributes_fpmath_mode>`.

Floating-point environment
--------------------------

oneDNN floating-point computation behavior is controlled by the floating-point environment as defined by the C and C++ standards, in the fenv.h header. In particular, the floating-point environment can control:

* the rounding mode. It is set to round-to-nearest tie-even by default on x64 systems and can be changed using the fesetround() C function.

* the handling of denormal values. Computation on denormals can negatively impact performance on x64 systems and are not flushed to zero by default.

.. note:: 

   Most DNN applications do not require precise computations with denormal numbers and flushing these denormals to zero can improve performance. On x64 systems, the floating-point environment can be updated to allow flushing denormals to zero as follow:
   
   .. ref-code-block:: cpp
   
   	#include <xmmintrin.h>
   	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
   
   


Hardware Limitations
~~~~~~~~~~~~~~~~~~~~

While all the platforms oneDNN supports have hardware acceleration for fp32 arithmetics, that is not the case for other data types. Support for low precision data types may not be available for older platforms. The next sections explain limitations that exist for low precision data types for Intel(R) Architecture processors, Intel Processor Graphics and Xe Architecture graphics.

Intel(R) Architecture Processors
--------------------------------

oneDNN performance optimizations for Intel Architecture Processors are specialized based on Instruction Set Architecture (ISA). The following ISA have specialized optimizations in the library:

* Intel Streaming SIMD Extensions 4.1 (Intel SSE4.1)

* Intel Advanced Vector Extensions (Intel AVX)

* Intel Advanced Vector Extensions 2 (Intel AVX2)

* Intel Advanced Vector Extensions 512 (Intel AVX-512)

* Intel Deep Learning Boost (Intel DL Boost)

* Intel Advanced Matrix Extensions (Intel AMX)

The following table indicates the minimal supported ISA for each of the data types that oneDNN recognizes.

==========  =====================================  
Data type   Mini                                   
==========  =====================================  
f32         Intel SSE4.1                           
s8, u8      Intel AVX2                             
bf16        Intel DL Boost with bfloat16 support   
f16         Intel AVX512-FP16                      
==========  =====================================

.. note:: 

   See :ref:`Nuances of int8 Computations <doxid-dev_guide_int8_computations>` in the Developer Guide for additional limitations related to int8 arithmetic.
   
   

.. note:: 

   The library has functional bfloat16 support on processors with Intel AVX-512 Byte and Word Instructions (AVX512BW) support for validation purposes. The performance of bfloat16 primitives on platforms without hardware acceleration for bfloat16 is 3-4x lower in comparison to the same operations on the fp32 data type.
   
   

.. note:: 

   The Intel AMX instructions ignore the floating-point environment flag and always round to nearest tie-even and flush denormals to zero.
   
   

.. note:: 

   f64 configuration is not available for the CPU engine.
   
   

.. note:: 

   The current f16 CPU instructions accumulate to f16. To avoid overflow, the f16 primitives might up-convert the data to f32 before performing math operations. This can lead to scenarios where a f16 primitive may perform slower than similar f32 primitive.
   
   


Intel(R) Processor Graphics and Xe Architecture graphics
--------------------------------------------------------

oneDNN performance optimizations for Intel Processor graphics and Xe Architecture graphics are specialized based on device microarchitecture (uArch). The following uArchs have specialized optimizations in the library:

* GEN9 (also covers GEN11)

* Xe-LP (previoulsy known as GEN12LP)

* Xe-HP

The following table indicates the minimal supported uArch for each of the data types that oneDNN recognizes.

==========  ======  
Data type   Mini    
==========  ======  
f32         GEN9    
s8, u8      Xe-LP   
bf16        Xe-HP   
f16         GEN9    
==========  ======

.. note:: 

   * f64 configurations are only supported on the GPU engines with HW capability for double-precision floating-point.
   
   * f16 operations may accumulate to f16 on the GPU architectures older than Xe-HPC. Newer architectures accumulate to f32.

