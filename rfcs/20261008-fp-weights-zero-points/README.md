# RFC: Floating-Point Zero Points for Weight Decompression in Matmul

## Introduction

oneDNN matmul supports decompression of integer weights using per-group scales
and zero points, with the following semantics:

$$
W_f = (W_{\mathrm{int}} - zp) \times scale
$$

Integer weight types such as `s8`, `u8`, `s4`, and `u4` can be decompressed
this way. Currently, `dnnl_primitive_attr_set_zero_points_v2()` restricts
zero-point data types to integers (`s32`, `s8`, `u8`, `s4`, `u4`, and `u2`).

However, several widely used asymmetric quantization formats represent their
zero points as non-integer values. GGUF K-quants, used by llama.cpp, are a
prominent example. Q4_K stores each 256-weight block using an f16 super-scale
and super-min, together with 6-bit sub-scales and sub-mins for groups of 32
weights. Q5_K uses a similar representation, while Q4_1 and Q5_1 also encode
an additive minimum.

For these formats, the decompression expression can be written as:

$$
w = d \times q - m
$$

Mapping this representation to oneDNN's decompression semantics gives:

$$
scale = d,\qquad zp = \frac{m}{d}
$$

Because $m/d$ is not necessarily an integer, converting it to an integer zero
point introduces quantization error.

### Problem: Accuracy loss from zero-point rounding

Frameworks integrating with oneDNN must currently round $m/d$ to an integer.
This introduces an error of up to half a quantization step, shifting every
weight in a group in the same direction. Unlike random quantization error, this
systematic bias does not average out.

On Intel GPUs, this rounding is a major source of the accuracy gap between
llama.cpp's OpenVINO backend, which uses oneDNN, and its CPU backend.

The following table compares the KL divergence of the next-token distribution
against the llama.cpp CPU backend at a context length of 1024 tokens, using
one-token decode steps on an Intel Arc B390. Lower values indicate closer
agreement with the CPU reference.

| Model (GGUF Q4_K_M) | Rounded zero point | f16 zero point | Vulkan backend |
| ------------------- | -----------------: | -------------: | -------------: |
| Gemma-4 E2B         |              0.363 |         0.0196 |         0.0187 |
| Llama-3.2 1B        |              0.041 |         0.0019 |         0.0014 |
| SmolLM2 1.7B        |              0.073 |         0.0103 |         0.0092 |
| Phi-3.5 mini        |              0.031 |         0.0048 |         0.0040 |
| Phi-4 mini          |              0.036 |         0.0108 |         0.0030 |
| Qwen3.5 4B          |              0.025 |         0.0032 |         0.0039 |

The Vulkan backend reads Q4_K natively and is included as a reference.
Supporting f16 zero points brings the OpenVINO backend substantially closer to
the CPU reference across all six models.

OpenVINO's GGUF frontend encounters the same limitation: its default Q4_K
mapping uses u4 weights with integer zero points. Its opt-in path for exact
zero points is documented as able to prevent compressed fully connected
fusion or slow down matmul. Native floating-point zero-point support in oneDNN
would enable exact decompression without requiring this trade-off.

### Cost of existing workarounds

Without native support, frameworks must either accept the accuracy loss or use
an alternative execution path.

In OpenVINO, using f16 zero points currently causes many fully connected layers
to fall back from oneDNN to an OpenCL kernel. For a Gemma-4 E2B decode step,
130 of 261 fully connected executions are affected.

The measured impact is substantial:

| Model        | Prefill throughput (512 tokens) | Decode throughput (1K context) | Additional GPU memory |
| ------------ | ------------------------------: | -----------------------------: | --------------------: |
| Gemma-4 E2B  |                            -67% |                          -6.6% |                2.1 GB |
| Llama-3.2 1B |                            -61% |                          -8.5% |                1.2 GB |

These results motivate extending oneDNN's existing weight-decompression
mechanism rather than requiring frameworks to implement correction logic or
bypass oneDNN.

## Proposal

Extend the supported zero-point data types for weight decompression to include
`f16`, `bf16`, and `f32`.

For example:

```c
dnnl_primitive_attr_set_zero_points(
        attr, DNNL_ARG_WEIGHTS, mask, 2, groups, dnnl_f16);
```

The decompression semantics remain unchanged:

$$
W_f = (W_{\mathrm{int}} - zp) \times scale
$$

The zero point is read using its declared data type instead of being
restricted to an integer representation. This allows frameworks to preserve
the original floating-point zero point without rounding.

### Scope and restrictions

* **Supported arguments:** `DNNL_ARG_WEIGHTS`, `DNNL_ARG_WEIGHTS_1`, and
  `DNNL_ARG_WEIGHTS_2`.
* **Supported data types:** `f16`, `bf16`, and `f32` for weight zero points.
  Integer zero points remain supported.
* **Supported operation:** Weight decompression for integer weights (`s4`,
  `u4`, `s8`, and `u8`) with floating-point activations. Integer computation
  (integer activations) keeps integer weight zero points.
* **Unchanged behavior:** Source and destination zero points remain restricted
  to integer types. Host-side scalar zero points are outside the scope of this
  proposal.
* **Implementation support:** Implementations that do not support
  floating-point weight zero points must return `unimplemented`.

This is an opt-in capability: accepting a data type at the API level does not
imply that every implementation must support it.

### API and ABI compatibility

No new API entry point is required. The existing
`dnnl_primitive_attr_set_zero_points_v2()` API (and
`dnnl_primitive_attr_set_zero_points()`, which calls it) is extended to accept
additional data types for weight arguments.

Existing configurations and integer-zero-point use cases remain unchanged. The
proposal does not require an ABI change.

### Prototype implementation

The prototype modifies 12 files, with approximately 130 lines changed,
including tests.

#### API validation and capability checks

* Extend `dnnl_primitive_attr_set_zero_points_v2()` validation to accept
  `f16`, `bf16`, and `f32` for weight arguments only.
* Add `fp_weights_zero_points_ok()` to `primitive_desc_t`, returning `false`
  by default.
* In `primitive_desc_t::create()`, after `init()`, reject (`unimplemented`)
  any implementation that receives a floating-point zero point for
  `DNNL_ARG_WEIGHTS`, `DNNL_ARG_WEIGHTS_1`, or `DNNL_ARG_WEIGHTS_2` and has not
  opted in. The check covers every primitive kind. The SDPA primitive, which
  takes its weight zero points through separate attributes, checks them when
  its descriptor is created.

This approach preserves existing behavior by default and allows individual
implementations to enable support without requiring changes to every
implementation. A central check matters here: some implementations accept
non-default zero-point data types without checking them (for example, the
reference GPU convolution), and would otherwise read an f16 zero point as an
integer.

#### Intel GPU support

The Intel GPU GEMM implementation (`jit:gemm`) opts in when it decompresses
the weights, that is, with floating-point activations. With integer
activations it computes in integers and does not opt in.

The GEMM-with-post-ops wrapper and the GPU matmul implementation that
delegates to GEMM forward the capability check to the GEMM implementation they
wrap.

In our reading of the kernel, the zero point is already converted to f16
before it is subtracted, so the prototype requires no kernel changes. The new
tests confirm this for `u4`, `s4`, and `u8` weights.

In the prototype, `jit:gemm` computes `f16` zero points with `f16` and `f32`
activations. `bf16` and `f32` zero points, and `bf16` activations, currently
return `unimplemented`.

#### Other implementations

CPU implementations and the reference GPU matmul implementation currently
interpret weight zero points as integers. They therefore continue to reject
floating-point zero points until explicit support is implemented.

Supporting floating-point zero points in the reference implementation should
be straightforward: load the zero point using its declared data type and
perform the subtraction in floating point.

### Performance

Existing configurations incur no additional runtime cost because the new
capability is disabled by default and the existing integer-zero-point
execution paths remain unchanged.

For the Intel GPU GEMM implementation, f16 zero points use the same
implementation (`jit:gemm`) as integer zero points. The primary additional
cost is the larger zero-point tensor.

The zero-point tensor grows from 4 to 16 bits per group of 32 weights, that
is, from 4.625 to 5.0 bits per weight for u4 weights with f16 scales.

End to end, with the prototype and the corresponding OpenVINO GPU plugin
change, exact f16 zero points compared with integer zero points on the same
build (llama.cpp's OpenVINO backend, Intel Arc B390, two rounds):

| Model        | Decode throughput (1K context) | Additional peak GPU memory (4K context) |
| ------------ | -----------------------------: | --------------------------------------: |
| Gemma-4 E2B  |                          -1.9% |                                 0.13 GB |
| Phi-4 mini   |                          -1.0% |                                 0.10 GB |
| Qwen3.5 4B   |                          -1.4% |                                 0.23 GB |
| SmolLM2 1.7B |                          -0.8% |                                 0.04 GB |

Prefill throughput (512 tokens) stays within the measurement noise of this
machine; for Gemma-4 E2B, the model with the most stable prefill
measurements, it changes by -1.4%. All fully connected layers stay on
`jit:gemm`. Two other models, Llama-3.2 1B and Phi-3.5 mini, varied by up to
10% between rounds and are not listed.

This overhead is the cost of retaining the more accurate zero-point
representation; it avoids the substantially larger penalties associated with
alternative execution paths.

### Build and dependencies

No new build dependencies or external libraries are required.

### Testing and validation

#### Test coverage

* **`test_iface_attr`:** Updated zero-point data-type validation tests, and a
  new test that no implementation accepts floating-point weight zero points
  for integer matmul or for convolution.
* **`harness_matmul_decompression_fp_zp`:** New benchdnn input covering f16
  zero points with `u4`, `s4`, and `u8` weights; grouped (`32x1`) and
  per-channel scales; shapes ranging from 1 to 512 rows; and three weight
  layouts.
* **benchdnn reference implementation:** Extended to support floating-point
  zero points and non-integer test values.

#### Prototype results

The prototype is based on oneDNN main at commit `847a8616df` and was tested on
an Intel Arc B390.

| Test                                   | Result                                                     |
| -------------------------------------- | ---------------------------------------------------------- |
| `test_iface_attr`                      | All passed on CPU (28) and GPU (26; CPU-only tests skip)   |
| New benchdnn input on GPU (`jit:gemm`) | 54/54 passed                                               |
| New benchdnn input on CPU              | All cases skipped, as expected                             |
| Unsupported cases on GPU and CPU       | `unimplemented`, as expected (see below)                   |
| `test_matmul_ci` on GPU                | Same result as unmodified main on all 5,838 cases that ran |

The unsupported cases checked with benchdnn are integer activations (`s8` and
`u8`, with `u8`, `s8`, and `u4` weights), `bf16` activations, `bf16` and `f32`
zero points, CPU matmul, and convolution with an `f16` weights zero point.
In an earlier version of the prototype, two of them (integer matmul with a
per-channel `f16` zero point, and the reference GPU convolution) produced
wrong results instead of `unimplemented`, which is why `jit:gemm` opts in only
for weights decompression and the check is central.

In `test_matmul_ci`, both builds give 5,569 passed, 264 mistrusted, and 5
failed cases. One case stops the run with `CL_OUT_OF_RESOURCES` on this
machine, with and without the change, so the suite was run in two parts
around it; 144 cases are skipped by benchdnn in both builds.

With the corresponding OpenVINO GPU plugin change
([openvinotoolkit/openvino#38696](https://github.com/openvinotoolkit/openvino/pull/38696)),
all 261 fully connected executions in a Gemma-4 E2B decode step use
`jit:gemm`. The resulting accuracy measurements are shown in the Introduction.

### Alternatives considered

#### Continue rounding zero points (status quo)

Round $m/d$ to an integer and retain the existing implementation.

**Trade-off:** No implementation changes, but systematic weight-decompression
error causes the accuracy regressions shown in the Introduction.

#### Apply a correction term outside oneDNN

Retain an integer zero point and compensate for the residual using the
per-group sum of activations.

**Trade-off:** This can recover the exact result, but requires additional
computation for each layer. An earlier experiment measured a 5.2% decode
performance regression.

#### Dequantize weights to f16 in the framework

Decompress weights before passing them to oneDNN.

**Trade-off:** Avoids zero-point rounding but increases weight storage and
bandwidth requirements by about 3.5x relative to Q4_K (16 instead of 4.5 bits
per weight).

#### Introduce a separate offset attribute

Represent decompression as:

$$
W_f = W \times scale - offset
$$

**Trade-off:** This more directly matches the GGUF representation and avoids
computing $m/d$. However, it introduces a new attribute instead of extending
the supported data types of an existing one. For the kernels in question, the
resulting computation is equivalent.

#### Execute affected layers outside oneDNN

Use an alternative kernel when floating-point zero points are required.

**Trade-off:** This is the current OpenVINO workaround, with the prefill,
decode, and GPU memory penalties described in the Introduction.

### Execution plan

1. **Agree on the supported data types:** Determine whether the initial scope
   should include only `f16` or also `bf16` and `f32`.
2. **Land the initial implementation:** Extend API validation, add the
   capability mechanism, enable Intel GPU GEMM, and add tests.
3. **Expand implementation coverage:** Add CPU and reference implementation
   support in a subsequent change, if required.

## Open Questions

1. **Data-type scope:** Should the API extension support only `f16`, which
   addresses the immediate GGUF use case, or include `bf16` and `f32` for
   completeness and reference testing?
2. **CPU support:** Should CPU implementations support floating-point zero
   points in the initial change, or should unsupported implementations return
   `unimplemented` until follow-up work?
3. **API design:** Is extending the existing zero-point data-type support
   preferable to introducing a separate offset attribute that directly
   represents the GGUF decompression formula?
