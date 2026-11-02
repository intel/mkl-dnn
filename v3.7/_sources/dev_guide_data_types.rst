.. index:: pair: page; Data Types
.. _doxid-dev_guide_data_types:

Data Types
==========

oneDNN functionality supports a number of numerical data types. IEEE single precision floating-point (fp32) is considered to be the golden standard in deep learning applications and is supported in all the library functions. The purpose of low precision data types support is to improve performance of compute intensive operations, such as convolutions, inner product, and recurrent neural network cells in comparison to fp32. Boolean data type is used for Graph Compiler to optimize operations which take bool as inputs and/or outputs data type.

==========  ===========================================================================================================================================================================================  
Data type   Description                                                                                                                                                                                  
==========  ===========================================================================================================================================================================================  
f32         `IEEE single precision floating-point <https://en.wikipedia.org/wiki/Single-precision_floating-point_format#IEEE_754_single-precision_binary_floating-point_format:_binary32>`__             
bf16        `non-IEEE 16-bit floating-point <https://www.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf>`__                                    
f16         `IEEE half precision floating-point <https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16>`__                   
s8/u8       signed/unsigned 8-bit integer                                                                                                                                                                
s4/u4       signed/unsigned 4-bit integer                                                                                                                                                                
f64         `IEEE double precision floating-point <https://en.wikipedia.org/wiki/Double-precision_floating-point_format#IEEE_754_double-precision_binary_floating-point_format:_binary64>`__             
boolean     bool (size is C++ implementation defined)                                                                                                                                                    
f8_e5m2     `OFP8 standard 8-bit floating-point <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ with 5 exponent and 2 mantissa bits   
f8_e4m3     `OFP8 standard 8-bit floating-point <https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf>`__ with 4 exponent and 3 mantissa bits   
e8m0        `MX standard 8-bit scaling type <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__                                                                   
f4_e2m1     `MX standard 4-bit floating-point <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__ with 2 exponent and 1 mantissa bits                             
f4_e3m0     4-bit floating-point with 3 exponent bits and no mantissa bit                                                                                                                                
==========  ===========================================================================================================================================================================================

.. note:: 

   Boolean is only supported in the Graph Compiler in CPU engines. No primitives support boolean during primitive computation.
   
   


Inference and Training
~~~~~~~~~~~~~~~~~~~~~~

oneDNN supports training and inference with the following data types:

===========  =========================================================================  ============================================  
Usage mode   CPU                                                                        GPU                                           
===========  =========================================================================  ============================================  
Inference    f32, bf16, f16, f8_e5m2/f8_e4m3, f4_e2m1, f4_e3m0, s8/u8, s4/u4, boolean   f32, bf16, f16, f8_e5m2/f8_e4m3, s8/u8, f64   
Training     f32, bf16, f16                                                             f32, bf16, f16, f64                           
===========  =========================================================================  ============================================

.. note:: 

   Using lower precision arithmetic may require changes in the deep learning model implementation.
   
   

.. note:: 

   f64 is supported only for matmul, convolution, reorder, layer normalization, and pooling primitives on the GPU engine.
   
   

.. note:: 

   Boolean is only supported by the oneDNN graph API when the graph compiler backend is enabled.
   
   

.. note:: 

   s4/u4 data types are only supported as a storage data type for weights argument in case of weights decompression. For more details, refer to :ref:`Matmul Tutorial: weights decompression <doxid-weights_decompression_matmul_cpp>`.
   
   
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

	\operatorname{convert_{dst\_dt}} ( \operatorname{dst\_zero\_point_{f32}} + \operatorname{postops_{f32}} (\operatorname{oscale_{f32}} * \operatorname{convert_{f32}} (\operatorname{Op}(\operatorname{src_{src\_dt}}, \operatorname{weights_{wei\_dt}}, ...))))

The ``Op`` output datatype depends on the datatype of its inputs:

* if ``src``, ``weights``, ... are floating-point datatype (f32, f16, bf16, f8_e5m2, f8_e4m3, f4_e2m1, f4_e3m0), then the ``Op`` outputs f32 elements.

* if ``src``, ``weights``, ... are integral datatypes (s8, u8, s32), then the ``Op`` outputs s32 elements.

* if the primitive allows to mix input datatypes, the ``Op`` outputs datatype will be s32 if its weights are an integral datatype, or f32 otherwise.

The accumulation datatype used during ``Op`` computation is governed by the ``accumulation_mode`` attribute of the primitive. By default, f32 is used for floating-point primitives (or f64 for f64 primitives) and s32 is used for integral primitives.

No downconversions are allowed by default, but can be enabled using the floating-point math controls described in :ref:`Primitive Attributes: floating-point math mode <doxid-dev_guide_attributes_fpmath_mode>`.

Rounding mode and denormal handling
-----------------------------------

oneDNN floating-point computation behavior follows the floating-point environment for the given device runtime by default. In particular, the floating-point environment can control:

* the rounding mode. It is set to round-to-nearest tie-even by default on x64 systems as well as devices running on SYCL and openCL runtime.

* the handling of denormal values. Computation on denormals are not flushed to zero by default. Note denormal handling can negatively impact performance on x64 systems.

.. note:: 

   For CPU devices, the default floating-point environment is defined by the C and C++ standards in the fenv.h header. Rounding mode can be changed globally using the fesetround() C function.
   
   

.. note:: 

   Most DNN applications do not require precise computations with denormal numbers and flushing these denormals to zero can improve performance. On x64 systems, the floating-point environment can be updated to allow flushing denormals to zero as follow:
   
   .. ref-code-block:: cpp
   
   	#include <xmmintrin.h>
   	_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
   
   

.. note:: 

   On some hardware architectures, low-precision datatype acceleration ignores floating-point environment and will flush denormal outputs to zero (FTZ). In particular this is the case for Intel AMX instruction set.
   
   
oneDNN also exposes non-standard stochastic rounding through the ``rounding_mode`` primitive attribute. More details on this attribute can be found in :ref:`Primitive Attributes: rounding mode <doxid-dev_guide_attributes_rounding_mode>`.

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

=================  =====================================  
Data type          Minimal supported ISA                  
=================  =====================================  
f32                Intel SSE4.1                           
s8, u8             Intel AVX2                             
bf16               Intel DL Boost with bfloat16 support   
f16                Intel AVX512-FP16                      
boolean            Intel AVX2                             
f8_e5m2, f8_e4m3   Intel AVX512-FP16                      
f4_e2m1, f4_e3m0   TBA                                    
=================  =====================================

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

oneDNN performance optimizations for Intel Processor graphics and Xe Architecture graphics are specialized based on device microarchitecture (uArch). The following uArchs and associated devices have specialized optimizations in the library:

* Xe-LP (accelerated u8, s8 support via DP4A)
  
  * Intel(R) UHD Graphics for 11th-14th Gen Intel(R) Processors
  
  * Intel(R) Iris(R) Xe Graphics
  
  * Intel(R) Iris(R) Xe MAX Graphics (formerly DG1)

* Xe-HPG (accelerated f16, bf16, u8, and s8 support via Intel(R) Xe Matrix Extensions (Intel(R) XMX), aka DPAS)
  
  * Intel(R) Arc(TM) Graphics (formerly Achemist)
  
  * Intel(R) Data Center GPU Flex Series (formerly Arctic Sound)

* Xe-HPC (accelerated f16, bf16, u8, and s8 support via DPAS and f64 support via MAD)
  
  * Intel(R) Data Center GPU Max Series (formerly Ponte Vecchio)

The following table indicates the data types with performant compute primitives for each uArch supported by oneDNN. Unless otherwise noted, all data types have reference support on all architectures.

=======  ================================================================  
uArch    Supported Data types                                              
=======  ================================================================  
Xe-LP    f32, f16, s8, u8                                                  
Xe-HPG   f32, f16, bf16, s8, u8                                            
Xe-HPC   f64, f32, bf16, f16, s8, u8                                       
TBA      f64, f32, bf16, f16, s8, u8, f8_e5m2, f8_e4m3, f4_e2m1, f4_e3m0   
=======  ================================================================

.. note:: 

   f64 configurations are only supported on GPU engines with HW capability for double-precision floating-point.
   
   

.. note:: 

   f8_e5m2 compute operations have limited performance through upconversion on Xe-HPC.
   
   

.. note:: 

   f16 operations may be faster with f16 accumulation on GPU architectures older than Xe-HPC. Newer architectures accumulate to f32.
   
   

.. note:: 

   Boolean is only supported by the oneDNN graph API when the graph compiler backend is enabled. The graph compiler backend only supports the CPU engine.

