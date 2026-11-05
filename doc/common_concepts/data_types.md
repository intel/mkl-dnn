Data Types {#dev_guide_data_types}
==================================

oneDNN functionality supports a number of numerical
data types. IEEE single precision floating-point (`f32`) is considered
to be the golden standard in deep learning applications and is supported
in all the library functions. The purpose of low precision data types
support is to improve performance of compute intensive operations, such as
convolutions, inner product, and recurrent neural network cells
in comparison to `f32`.

| Data type | Description                                                                                                                                                                             |
|:----------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| f32       | [IEEE single precision floating-point](https://en.wikipedia.org/wiki/Single-precision_floating-point_format#IEEE_754_single-precision_binary_floating-point_format:_binary32)           |
| bf16      | [non-IEEE 16-bit floating-point](https://www.intel.com/content/dam/develop/external/us/en/documents/bf16-hardware-numerics-definition-white-paper.pdf)                                  |
| f16       | [IEEE half precision floating-point](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16)                 |
| s8/u8     | signed/unsigned 8-bit integer                                                                                                                                                           |
| s4/u4     | signed/unsigned 4-bit integer                                                                                                                                                           |
| s32       | signed 32-bit integer                                                                                                                                                          |
| f64       | [IEEE double precision floating-point](https://en.wikipedia.org/wiki/Double-precision_floating-point_format#IEEE_754_double-precision_binary_floating-point_format:_binary64)           |
| f8\_e5m2  | [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) with 5 exponent and 2 mantissa bits |
| f8\_e4m3  | [OFP8 standard 8-bit floating-point](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf) with 4 exponent and 3 mantissa bits |
| e8m0      | [MX standard 8-bit scaling type](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)                                                                 |
| f4\_e2m1  | [MX standard 4-bit floating-point](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) with 2 exponent and 1 mantissa bits                           |
| u2        | unsigned 2-bit integer                                                                                                                                                           |
| u3        | unsigned 3-bit integer. Flat LSB-first bit-packing is used with value i at bits [3i:3i+2], 8 values per 3 bytes                                     |

## Inference and Training

oneDNN supports training and inference with the following data types:

| Data type | Inference | Training |
| :-------- | :-------: | :------: |
| f64       | `+`(1)    | `+`(1)   |
| f32       | `+`       | `+`      |
| bf16      | `+`       | `+`      |
| f16       | `+`       | `+`      |
| fp8(3)    | `+`       | `+`      |
| s8        | `+`       |          |
| u8        | `+`       |          |
| f4\_e2m1  | `+`       |          |
| s4        | `+`(2)    |          |
| u4        | `+`(2)    |          |
| u3        | `+`(2)    |          |
| u2        | `+`(2)    |          |

Footnotes:
1. `f64` support is limited to matmul, convolution, reorder, layer normalization, and
   pooling primitives on Intel GPUs.
2. `s4`/`u4`/`u2`/`u3` data types are only supported as a storage data type for weights argument
   in case of weight-only quantization. For more details, refer to
   [Matmul Tutorial: weight-only quantization](@ref matmul_with_weight_only_quantization_cpp).
3. `fp8` data type includes `f8_e5m2` and `f8_e4m3`.

@note
    Data type support may also be limited by hardware capabilities. Refer to
    [Hardware Limitations](@ref data_types_hardware_limitations) section below
    for details.

@note
    Using lower precision arithmetic may require changes in the deep learning
    model implementation.

See topics for the corresponding data types details:
 * @ref dev_guide_inference_int8
 * @ref dev_guide_attributes_quantization
 * @ref dev_guide_training_bf16
 * @ref dev_guide_attributes_fpmath_mode
 * @ref matmul_with_weight_only_quantization_cpp
 * @ref matmul_f8_quantization_cpp

Individual primitives may have additional limitations with respect to data type
by each primitive is included in the corresponding sections of the developer
guide.

## General numerical behavior of the oneDNN library

During a primitive computation, oneDNN can use different datatypes
than those of the inputs/outputs. In particular, oneDNN uses wider
accumulator datatypes (`s32` for integral computations, and `f32`/`f64` for
floating-point computations), and converts intermediate results to `f32`
before applying post-ops (`f64` configuration does not support
post-ops).  The following formula governs the datatypes dynamic during
a primitive computation:

\f[
\operatorname{convert_{dst\_dt}} ( \operatorname{zp_{dst}} + 1/\operatorname{scale_{dst}} * \operatorname{postops_{f32}} (\operatorname{convert_{f32}} (\operatorname{Op}(\operatorname{src_{src\_dt}}, \operatorname{weights_{wei\_dt}}, ...))))
\f]

The `Op` output datatype depends on the datatype of its inputs:
- if `src`, `weights`, ... are floating-point datatype (`f32`, `f16`,
  `bf16`, `f8_e5m2`, `f8_e4m3`, `f4_e2m1`), then the `Op` outputs `f32` elements.
- if `src`, `weights`, ... are integral datatypes (`s8`, `u8`, `s32`), then
  the `Op` outputs `s32` elements.
- if the primitive allows to mix input datatypes, the `Op` outputs
  datatype will be `s32` if its weights are an integral datatype, or `f32`
  otherwise.

The accumulation datatype used during `Op` computation is governed by
the `accumulation_mode` attribute of the primitive. By default, `f32` is
used for floating-point primitives (or `f64` for `f64` primitives) and `s32`
is used for integral primitives.

No downconversions are allowed by default, but can be enabled using
the floating-point math controls described in @ref
dev_guide_attributes_fpmath_mode.

The \f$convert_{dst\_dt}\f$ conversion is guaranteed to be faithfully
rounded but not guaranteed to be correctly rounded (the returned value
is not always the closest one but one of the two closest representable
value). In particular, some hardware platforms have no direct
conversion instructions from `f32` data type to low-precision data types
such as `f8` or `f4`, and will perform conversion through an
intermediate data type (for example `f16` or `bf16`), which may result in
[double rounding](https://en.wikipedia.org/wiki/Rounding#Double_rounding).

Conversions to integral datatypes saturate upon overflow. Conversions of
floating-point datatypes compliant to MXFP schemes saturate upon overflow,
whereas conversions to any other floating-point datatypes don't.


### Rounding mode and denormal handling

oneDNN floating-point computation behavior follows the floating-point
environment for the given device runtime by default. In particular,
the floating-point environment can control:
- the rounding mode. It is set to round-to-nearest tie-even by default
  on x64 systems as well as devices running on SYCL and openCL runtime.
- the handling of denormal values. Computation on denormals are not
  flushed to zero by default. Note denormal handling can negatively
  impact performance on x64 systems.

@note
  For CPU devices, the default floating-point environment is defined by
  the C and C++ standards in the following header:
~~~cpp
#include <fenv.h>
~~~
  Rounding mode can be changed globally using the `fesetround()` C function.

@note
  Most DNN applications do not require precise computations with denormal
  numbers and flushing these denormals to zero can improve performance.
  On x64 systems, the floating-point environment can be updated to allow
  flushing denormals to zero as follow:
~~~cpp
#include <xmmintrin.h>
_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
~~~

@note
  On some hardware architectures, low-precision datatype acceleration
  ignores floating-point environment and will flush denormal outputs
  to zero (FTZ). In particular this is the case for Intel AMX
  instruction set.

oneDNN also exposes non-standard stochastic rounding through the
`rounding_mode` primitive attribute. More details on this attribute
can be found in @ref dev_guide_attributes_rounding_mode.

@anchor data_types_hardware_limitations
## Hardware Limitations

While all the platforms oneDNN supports have hardware acceleration for
`f32` arithmetic, that is not the case for other data types. Support
for low precision data types may not be available for older
platforms. The next sections explain limitations that exist for low
precision data types for Intel 64/AMD64 based processors and Intel Graphpics.

### Intel 64/AMD64 Processors

oneDNN performance optimizations for Intel Architecture Processors are
specialized based on Instruction Set Architecture (ISA). The following
table indicates data types support for every supported ISA:

| ISA                                                  | f64     | f32     | bf16    | f16     | s8/u8   | fp8     | f4_e2m1 | s4/u4    |
| ---------------------------------------------------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- |
| Intel AVX2                                           |         | `+`     |         |         | `+`(1)  |         |         |          |
| Intel AVX2 with Intel DL Boost (int8)                |         | `+`     |         |         | `+`     |         |         |          |
| Intel AVX-512                                        |         | `+`     | `.`(2)  |         | `+`(1)  |         | `.`(4)  |          |
| Intel AVX-512 with Intel DL Boost (int8)             |         | `+`     | `.`(2)  |         | `+`     |         | `.`(4)  |          |
| Intel AVX-512 with Intel DL Boost (int8, bf16)       |         | `+`     | `+`     |         | `+`     |         | `.`(4)  |          |
| Intel AVX2 with Intel DL Boost (int8) and NE_CONVERT |         | `+`     | `.`     | `.`     | `+`     |         |         |          |
| Intel AVX10.1 with Intel AMX (int8, bf16)            |         | `+`     | `+`     | `.`(3)  | `+`     |         | `.`(4)  | `.`      |
| Intel AVX10.1 with Intel AMX (int8, bf16, f16)       |         | `+`     | `+`     | `+`     | `+`     | `.`     | `.`(4)  | `.`      |
| Intel AVX10.2                                        |         | `+`     | `+`     | `+`     | `+`     | `.`     | `.`(4)  | `.`      |
| Intel AVX10.2 with Intel AMX (int8, bf16, fp16, fp8) |         | `+`     | `+`     | `+`     | `+`     | `+`     | `.`(4)  | `.`      |

Legend:
* `+` indicates oneDNN uses hardware-native compute support for this data type.
* `.` indicates oneDNN supports this data type via conversion to a higher precision data type.

Footnotes:
1. See @ref dev_guide_int8_computations in the Developer Guide for additional
   limitations related to `u8`/`s8` arithmetic.
2. The library has functional `bf16` support on processors with
   Intel AVX-512 Byte and Word Instructions (AVX512BW) support for validation
   purposes. The performance of `bf16` primitives on platforms without
   hardware acceleration for `bf16` is 3-4x lower in comparison to
   the same operations on the `f32` data type.
3. Intel AVX-512 `f16` instructions accumulate to `f16`. To avoid overflow, the `f16`
   primitives might up-convert the data to `f32` before performing math operations.
   This can lead to scenarios where an `f16` primitive may perform slower than
   similar `f32` primitive.
4. `f4_e2m1` weights are supported by MatMul with `f32` source and destination
   via decompression to `f32` using Intel AVX-512. This includes MXFP4 weights
   with `e8m0` scales for groups of 32 elements along the reduction dimension.

### Intel Graphics

oneDNN performance optimizations for Intel Graphics are specialized based on
device microarchitecture (uArch). The following uArchs and associated devices
have specialized optimizations in the library:
 * Xe-LPG
   * Intel Graphics for Intel Core Ultra Series 1 processors (formerly Meteor Lake)
 * Xe-LPG+
   * Intel Graphics for Intel Core Ultra Series 2 processors (formerly Arrow Lake)
 * Xe-HPG
   * Intel Arc A-Series Graphics (formerly Achemist)
   * Intel Data Center GPU Flex Series (formerly Arctic Sound)
 * Xe-HPC
   * Intel Data Center GPU Max Series (formerly Ponte Vecchio)
 * Xe2-LPG
   * Intel Graphics for Intel Core Ultra Series 2 processors (formerly Lunar Lake)
 * Xe2-HPG
   * Intel Arc B-Series Graphics (formerly Battlemage)
 * Xe3-LPG
   * Intel Graphics for Intel Core Ultra Series 3 processors (formerly Panther Lake)
   * Intel Graphics for Intel Core Series 3 processors (code name Wildcat Lake)
   * Intel Graphics for future Intel Core Ultra processors (code name Nova Lake S)
 * Xe3p-LPG
   * Intel Graphics for future Intel Core Ultra processors (code name Nova Lake P)
 * Xe3p-XPC
   * Intel GPU D-Series (formerly Crescent Island)

The following table indicates the data types support for each uArch supported by oneDNN.

| uArch    | f64     | f32     | bf16    | f16     | s8/u8   | f8      | f4_e2m1 | s4/u4/u2 |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- | -------- |
| Xe-LPG   |         | `+`     | `.`     | `+`(1)  | `+`     |         |         |          |
| Xe-LPG+  |         | `+`     | `+`     | `+`     | `+`     |         |         |          |
| Xe-HPG   |         | `+`     | `+`     | `+`     | `+`     | `.`     |         | `.`      |
| Xe-HPC   | `+`     | `+`     | `+`     | `+`     | `+`     | `.`     | `.`     | `.`      |
| Xe2-LPG  | `+`     | `+`     | `+`     | `+`     | `+`     | `.`     | `.`     | `.`      |
| Xe2-HPG  | `+`     | `+`     | `+`     | `+`     | `+`     | `.`     | `.`     | `.`      |
| Xe3-LPG  | `+`     | `+`     | `+`     | `+`     | `+`     | `.`     | `.`     | `.`      |
| Xe3p-LPG | `+`     | `+`     | `+`     | `+`     | `+`     | `+`     | `.`     | `.`      |
| Xe3p-XPC | `+`     | `+`     | `+`     | `+`     | `+`     | `+`     | `+`     | `.`      |

Legend:
* `+` indicates oneDNN uses hardware-native compute support for this data type.
* `.` indicates oneDNN supports this data type via conversion to a higher precision data type.

Footnotes:
1. Xe-LPG architecture does not natively support `f16` operations with `f32`
   accumulation. Consider using
   [relaxed accumulation mode](@ref dev_guide_attributes_accumulation_mode)
   for the best performance results.
