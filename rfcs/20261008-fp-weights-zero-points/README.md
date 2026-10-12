# RFC: Scales-Before-Zero-Points Quantization Mode for Matmul Weights

## Introduction

oneDNN matmul supports decompression of integer weights with per-group scales
and zero points. The decompressed value is a scaled difference with an integer
zero point (the `static_sazp` quantization mode):

$$
W_f = scale \times (W_{\mathrm{int}} - zp)
$$

Many widely used asymmetric weight quantization formats use a different affine
form instead: the scaled integer value plus an offset,

$$
W_f = scale \times W_{\mathrm{int}} + b
$$

Two prominent users:

* **GGUF K-quants (llama.cpp, OpenVINO's GGUF frontend).** Q4_K stores each
  256-weight block with an f16 super-scale and super-min, and 6-bit sub-scales
  and sub-mins for groups of 32 weights. A group dequantizes as
  $w = d \times q - m$, where $d$ and $m$ are the products of the super- and
  sub-values. Q5_K uses the same form. Q4_1 and Q5_1 store the offset with the
  opposite sign and add it ($w = d \times q + m$).
* **PyTorch int4 weight-only quantization (gpt-fast, torchao, torchchat).**
  Weights are `uint4` values with an implicit offset of 8, dequantized as
  $w = (q - 8) \times scale + zero$ with `bf16` scales and zeros, in groups of
  32 to 256 weights along K (128 by default).

To use `static_sazp`, a framework has to express the offset as an integer zero
point: $zp = -b / scale$ ($m/d$ for the K-quants, $-zero/scale$ for PyTorch).
That ratio is generally not an integer.

### Problem: accuracy loss from zero-point rounding

Frameworks integrating with oneDNN must currently round $-b / scale$ to an
integer. The error is up to half a quantization step, and it shifts every
weight of a group in the same direction. Unlike random quantization error, this
systematic bias does not average out.

On Intel GPUs, this rounding is a major source of the accuracy gap between
llama.cpp's OpenVINO backend, which uses oneDNN, and its CPU backend. The table
compares the KL divergence of the next-token distribution against the llama.cpp
CPU backend at a context length of 1024 tokens, using one-token decode steps on
an Intel Arc B390 (lower is closer to the CPU reference). The Vulkan backend,
which reads Q4_K natively, is included as a reference; the last column is the
part of the gap between the rounded zero point and Vulkan that the exact zero
point closes.

| Model (GGUF Q4_K_M) | Rounded zero point | Exact zero point | Vulkan backend | Gap closed |
| ------------------- | -----------------: | ---------------: | -------------: | ---------: |
| Gemma-4 E2B         |              0.363 |           0.0196 |         0.0187 |      99.7% |
| Llama-3.2 1B        |              0.041 |           0.0019 |         0.0014 |      98.7% |
| SmolLM2 1.7B        |              0.073 |           0.0103 |         0.0092 |      98.3% |
| Phi-3.5 mini        |              0.031 |           0.0048 |         0.0040 |      97.0% |
| Phi-4 mini          |              0.036 |           0.0108 |         0.0030 |      76.4% |
| Qwen3.5 4B          |              0.025 |           0.0032 |         0.0039 |       103% |

The cause is the same for every model; the models differ in how sensitive they
are to the shift and in how many of their tensors use Q4_K. Phi-4 mini's
remaining gap is not from zero points: llama.cpp's OpenVINO backend requantizes
its Q6_K token embedding, which is also its output layer, to 8 bits. Keeping
that tensor in Q6_K brings Phi-4 mini to 0.0031 (Vulkan: 0.0030).

The "exact zero point" column was measured with the first version of this
proposal, which stored $m/d$ as an `f16` zero point under `static_sazp`. The
mode proposed here represents the same weights with the offset itself as the
zero point, which avoids the division.

OpenVINO's GGUF frontend has the same limitation: its default Q4_K mapping uses
u4 weights with integer zero points, and its opt-in path for exact zero points
is documented as able to prevent compressed fully connected fusion or slow down
matmul.

### Cost of existing workarounds

Without native support, frameworks either accept the accuracy loss or use
another execution path. In OpenVINO, floating-point zero points currently make
many fully connected layers fall back from oneDNN to an OpenCL kernel (130 of
261 fully connected executions in a Gemma-4 E2B decode step):

| Model        | Prefill throughput (512 tokens) | Decode throughput (1K context) | Additional GPU memory |
| ------------ | ------------------------------: | -----------------------------: | --------------------: |
| Gemma-4 E2B  |                            -67% |                          -6.6% |                2.1 GB |
| Llama-3.2 1B |                            -61% |                          -8.5% |                1.2 GB |

## Proposal

Add a quantization mode, `static_sbzp` (scales before zero points), in which
the zero point is subtracted from the already scaled value:

$$
W_f = scale \times W_{\mathrm{int}} - zp
$$

Both modes are defined in the dequantization direction, from the quantized
value to the floating-point one:

| Mode          | Dequantization                          | Zero-point data types               |
| ------------- | --------------------------------------- | ----------------------------------- |
| `static_sazp` | $scale \times (W_{\mathrm{int}} - zp)$  | `s32`, `s8`, `u8`, `s4`, `u4`, `u2` |
| `static_sbzp` | $scale \times W_{\mathrm{int}} - zp$    | `f16`, `bf16`                       |

Because the zero point applies to a value that is already scaled, it is a
floating-point value, and floating-point zero points are allowed only in this
mode. Under `static_sazp`, zero points stay integer-only, as today.

Destination quantization, $dst = x / scale_{dst} + zp_{dst}$, applies the zero
point after the scale, but it goes in the other direction, from the
floating-point value to the quantized one, and it is the inverse of
`static_sazp`: from $x = scale \times (q - zp)$ follows $q = x / scale + zp$.
The inverse of `static_sbzp` would be $q = (x + zp) / scale$. So the existing
destination quantization is `static_sazp`, and this proposal doesn't change it;
`static_sbzp` is not accepted for the destination (or the source).

### API

The mode is set on both the scales and the zero points of an argument, and must
be the same for both.

* Scales: the existing `dnnl_primitive_attr_set_scales_v3()`, which already
  takes a `dnnl_quantization_mode_t`, accepts
  `dnnl_quantization_mode_static_sbzp`.
* Zero points: a new `dnnl_primitive_attr_set_zero_points_v3()`, mirroring
  `set_scales_v3()` (`set_zero_points_v2()` already exists and adds the data
  type and groups, so the version with a mode is v3):

```c
dnnl_status_t dnnl_primitive_attr_set_zero_points_v3(
        dnnl_primitive_attr_t attr, int arg, int mask, int ndims,
        const dnnl_dims_t group_dims, dnnl_data_type_t data_type,
        int is_on_host, dnnl_quantization_mode_t qmode);
```

* C++: `quantization_mode::static_sbzp`, and an optional `qmode` argument
  (default `static_sazp`) on `primitive_attr::set_zero_points()`, as
  `set_scales()` has.

Example (Q4_K weights, groups of 32 along K):

```cpp
attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {32, 1},
        memory::data_type::f16, false, quantization_mode::static_sbzp);
attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {32, 1},
        memory::data_type::f16, false, quantization_mode::static_sbzp);
```

Existing calls (`set_scales`, `set_scales_v2`, `set_zero_points`,
`set_zero_points_v2`) keep `static_sazp`, so existing behavior doesn't change.
The mode is part of the attribute, so it is part of the primitive cache key and
is printed in verbose output (`attr-scales:wei:3:f16:32x1:static_sbzp`).

### Scope

The initial scope is the scenario both users need, weights decompression in
matmul. Everything outside it is rejected, either when the attribute is set or
when the primitive is created, and can be added later without changing the API.

| Item                         | Supported                                                                                         | Rejected                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Primitive                    | matmul                                                                                            | all others (convolution, inner product, SDPA, gated MLP, ...)                                    |
| Argument                     | `DNNL_ARG_WEIGHTS`                                                                                | source, destination, other weights arguments                                                    |
| Weights data type            | `u4`, `s4`, `u8`                                                                                  | others                                                                                           |
| Activations                  | `f16`, `bf16`                                                                                     | `f32`; integer activations (integer computation)                                                 |
| Destination                  | `f16`, `bf16`, `f32`                                                                              | integer destinations                                                                             |
| Scales and zero points       | the activations' type (`f16` or `bf16`), the same for both                                        | `f32`; integer zero points; a type different from the activations or from each other            |
| Granularity                  | both grouped along K with the same groups ($G > 1$, $K$ divisible by $G$, at least two groups) | common, per-channel, a single group, different groups for scales and zero points, host scalars |
| Presence                     | scales and zero points both set, both in `static_sbzp`                                            | only one of them set, or modes that differ                                                      |
| Other attributes             | post-ops; destination scales; `fpmath` mode with `apply_to_int`, as weights decompression already requires | source scales or zero points, precomputed reductions                              |
| Implementations              | Intel GPU `jit:gemm` (optimized); CPU and GPU reference matmul (functional, same scope)           | all others return `unimplemented`                                                               |
| Graph API                    | unchanged                                                                                         |                                                                                                  |

Per-channel scales are left out because `jit:gemm` applies them to the result
after the reduction, which works only when the zero point comes first. They can
be added later (as a post-op on the result, or a single fused multiply-add).

### Mapping examples

* **GGUF K-quants (Q4_K, Q5_K):** `u4` weights for Q4_K and `u8` weights for
  Q5_K's 5-bit values; scale $= d$ and zero point $= m$, the stored minimum,
  both per group of 32. No division is needed. **Q4_1, Q5_1:** zero point
  $= -m$.
* **PyTorch int4 weight-only:** flipping the top bit of each `uint4` value
  gives the `s4` value $q - 8$; scale $= scale$, zero point $= -zero$, both
  `bf16` with PyTorch's group size.

## Implementation

The prototype is a single commit on oneDNN main (`2dd09b52c`, October 12,
2026) and changes 36 files (686 lines added, 75 removed), including tests and
documentation.

### Attributes and checks

* `dnnl_quantization_mode_static_sbzp` (C) and `quantization_mode::static_sbzp`
  (C++); `set_zero_points_v3()`.
* When the attribute is set: `static_sbzp` is accepted only for
  `DNNL_ARG_WEIGHTS`, with `f16` or `bf16` data types and not on the host;
  floating-point zero points only with `static_sbzp`.
* Default deny: implementations list the attributes they support in
  `primitive_attr_t::has_default_values()`, and a new skip-mask bit,
  `static_sbzp`, is checked there. Unless an implementation passes this bit,
  any attribute that uses the mode is rejected, so implementations that don't
  check the quantization mode today cannot misread a floating-point zero
  point. Matmul implementations that use `attr_scales_ok()` and
  `attr_zero_points_ok()` also reject it, because those accept only
  `static_sazp` by default.
* Matmul (common checks): the scope rules in the table above, which need both
  the scales and the zero points to be final.
* SDPA and gated MLP pass weight quantization attributes to the GEMM
  microkernels separately and reject `static_sbzp` explicitly.

### Intel GPU `jit:gemm`

`jit:gemm` already applies the zero point and the grouped scale to the weights
in registers before the dot product (subtract, then multiply). For
`static_sbzp`, a new problem flag reverses the order, and the scale and the
zero point are applied together in one fused multiply-add
($q \times scale + (-zp)$):

* `u4`/`s4` weights: the int4 conversion runs without the zero point, which
  `static_sazp` folds into it; the multiply-add follows.
* `u8` weights (general path): the multiply-add replaces the subtract and the
  multiply.
* `bf16` activations: the zero point is subtracted after the scale, in a
  separate operation, because the scale and the zero point have different
  internal types there.

The flag is part of the kernel key. The GEMM microkernels (SDPA, gated MLP)
don't support it.

### Reference implementations

* CPU reference matmul: $w = q \times scale - zp$ for `static_sbzp`.
* GPU reference matmul: per-element floating-point dequantization, with the
  zero point read using its data type.

### Tests and documentation

* benchdnn: `:static_sbzp` (or `:sbzp`) suffix on `--attr-scales` and
  `--attr-zero-points`; reference support; floating-point zero-point test
  values; a new input, `harness_matmul_decompression_sbzp` (`u4`, `s4`, `u8`
  weights; `f16` and `bf16`; groups 32 to 256; `ab`, `ba` and `any` weights
  layouts; post-ops), added to `test_matmul_gpu` and `test_matmul_all`.
* gtests (`test_iface_attr`): API validation and matmul creation rules.
* Documentation: the quantization attributes guide (both modes, the
  destination note, the scope, and the mapping examples), the matmul page, and
  the API reference.

## Validation

Tested with the prototype as described (scale and zero point in one fused
multiply-add) on two GPU generations: an Intel Arc B390 (Xe3, integrated) and
an Intel Arc B580 (Xe2, discrete). The B580 results are for the final commit.
The B390 results are for the same change on an older main, before two checks
were added that reject configurations only the reference implementations ran
(scales and zero points of a type different from the activations, and integer
destinations); every configuration tested there is still accepted.

| Test                                        | Arc B390                                       | Arc B580                       |
| ------------------------------------------- | ---------------------------------------------- | ------------------------------ |
| `test_iface_attr`                           | All passed on CPU (29) and GPU (27)            | All passed on CPU (29) and GPU (27) |
| New benchdnn input on GPU                   | 159/159 passed, all executed by `jit:gemm`     | 159/159 passed, all executed by `jit:gemm` |
| New benchdnn input on CPU (reference)       | 159/159 passed                                 | not run (host CPU without `f16`/`bf16` support; skipped by benchdnn) |
| LLM-sized shapes on GPU (1)                 | 20/20 passed                                   | 20/20 passed                   |
| `test_matmul_ci` on GPU                     | Same status as main for every case that ran with both builds (5,983 cases) | 5,569 passed, 264 mistrusted, 5 failed, 145 skipped (5,983 cases); the 5 failed cases (none uses `static_sbzp`) fail the same way on main |

(1) `f16` `u4` groups 32 and 128, `f16` `s4` group 64, `f16` `u8` group 128,
`bf16` `s4` group 128; M = 1, 32, 512; K and N up to 8192.

An earlier version of the prototype, which subtracted the zero point in a
separate operation, also passed 32/32 LLM-sized cases (including `bf16` `u4`),
and a negative control: with `jit:gemm` forced back to the `static_sazp` order,
all 159 cases of the new input failed, so the tests tell the two orders apart.

In `test_matmul_ci` on the B390, both builds give 5,569 passed, 264
mistrusted, 5 failed and 145 skipped cases. On that machine some cases stop the
run with `CL_OUT_OF_RESOURCES` with and without the change (for example an
`f8_e5m2` case with `e8m0` scales and no zero points, which fails the same way
on main when run alone), and which case stops a run varies, so the suite was
run in segments around them.

## Performance

Existing configurations are unaffected: the new mode is opt-in, and the
`static_sazp` code paths don't change (`test_matmul_ci` gives the same results
as main).

In `jit:gemm`, `static_sbzp` applies the scale and the zero point to the
weights in one fused multiply-add ($q \times scale - zp$), so it needs the same
number of operations per weight as `static_sazp`, which fuses the subtraction
into the int4 conversion.

Kernel time with cold caches (benchdnn `--mode=P --cold-cache=all`, so the
weights, scales and zero points are read from memory as in an LLM decode step;
Intel Arc B390, `f16` activations and scales, `u4` weights, minimum over three
rounds, ms):

| Groups | Shape (M x K x N)  | `static_sbzp`, `f16` zp | First RFC version, `f16` zp | `static_sazp`, `u4` zp |
| -----: | ------------------ | ----------------------: | --------------------------: | ---------------------: |
|    128 | 1 x 4096 x 4096    |                  0.0833 |                      0.0839 |                 0.0820 |
|        | 1 x 3072 x 8192    |                  0.1284 |                      0.1272 |                 0.1230 |
|        | 1 x 8192 x 3072    |                  0.1252 |                      0.1254 |                 0.1224 |
|        | 32 x 4096 x 4096   |                  0.1796 |                      0.1821 |                 0.1827 |
|        | 512 x 4096 x 4096  |                  0.3827 |                      0.3681 |                 0.3823 |
|     32 | 1 x 4096 x 4096    |                  0.0954 |                      0.0957 |                 0.0888 |
|        | 1 x 3072 x 8192    |                  0.1423 |                      0.1422 |                 0.1328 |
|        | 1 x 8192 x 3072    |                  0.1435 |                      0.1435 |                 0.1333 |
|        | 32 x 4096 x 4096   |                  0.2064 |                      0.2229 |                 0.2164 |
|        | 512 x 4096 x 4096  |                  0.3869 |                      0.3697 |                 0.3804 |

The "first RFC version" column is the earlier prototype of this proposal
(`f16` zero points under `static_sazp`, based on an older main). For single-row
products the kernels are limited by memory bandwidth (about 107 GB/s for
1 x 4096 x 4096 with groups of 128), and `static_sbzp` takes the same time as
that prototype. Compared with integer zero points it takes 2% to 4% longer
with groups of 128 and 7% to 8% longer with groups of 32, which is the extra
data of `f16` zero points (16 instead of 4 bits per group: 2.3% and 8.1% more
bytes per weight). For 32 and 512 rows, the differences are within the
run-to-run variation.

With warm caches (the same benchdnn runs without `--cold-cache`), where the
dequantization arithmetic instead of the memory traffic decides, `static_sbzp`
also takes the same time as the first version: 0.0242 ms against 0.0263 ms for
1 x 4096 x 4096 and 0.0433 ms against 0.0439 ms for 1 x 3072 x 8192 (groups of
128).

Other weight types, groups of 128, cold caches:

* `u8` weights (general dequantization path): `static_sbzp` and `u8` zero
  points take the same time (1 x 4096 x 4096: 0.1544 and 0.1542 ms;
  1 x 3072 x 8192: 0.2318 and 0.2319 ms).
* `bf16` activations with `s4` weights (the PyTorch case): `static_sbzp` with
  `bf16` zero points is as fast as or faster than `static_sazp` with `s8` zero
  points (0.0855 against 0.0861 ms for 1 x 4096 x 4096, 0.1361 against
  0.1453 ms for 1 x 3072 x 8192, 0.1279 against 0.1395 ms for 1 x 8192 x 3072),
  and 5% to 8% slower than without zero points.

On the Arc B580 (cold caches, minimum over three rounds, ms), the comparison
with integer zero points is the same:

| Groups | Shape (M x K x N)  | `static_sbzp`, `f16` zp | `static_sazp`, `u4` zp |
| -----: | ------------------ | ----------------------: | ---------------------: |
|    128 | 1 x 4096 x 4096    |                  0.0510 |                 0.0505 |
|        | 1 x 3072 x 8192    |                  0.0342 |                 0.0335 |
|        | 1 x 8192 x 3072    |                  0.0533 |                 0.0519 |
|        | 32 x 4096 x 4096   |                  0.0948 |                 0.0950 |
|        | 512 x 4096 x 4096  |                  0.1909 |                 0.1952 |
|     32 | 1 x 4096 x 4096    |                  0.0469 |                 0.0455 |
|        | 1 x 3072 x 8192    |                  0.0380 |                 0.0356 |
|        | 1 x 8192 x 3072    |                  0.0541 |                 0.0530 |
|        | 32 x 4096 x 4096   |                  0.0950 |                 0.0971 |
|        | 512 x 4096 x 4096  |                  0.2124 |                 0.2126 |

Single-row products take 1% to 3% longer with groups of 128 and 2% to 7%
longer with groups of 32; for 32 and 512 rows the times are equal or slightly
lower. With `bf16` activations and `s4` weights (groups of 128),
`static_sbzp` with `bf16` zero points is again faster than `static_sazp` with
`s8` zero points for single rows (0.0525 against 0.0558 ms, 0.0372 against
0.0429 ms and 0.0790 against 0.0991 ms for the three shapes) and equal for 32
and 512 rows.

End to end, the first version of this proposal cost 0.8% to 1.9% of decode
throughput at a 1K context in llama.cpp's OpenVINO backend, with prefill
within the run-to-run variation, compared with rounded integer zero points.
`static_sbzp` has the same kernel time, so we expect the same.

## Accuracy

The weights `static_sbzp` reconstructs from GGUF Q4_K data are as close to the
exact values as with the first RFC version, and much closer than with rounded
integer zero points. Dequantization error over all 96 Q4_K tensors of
Llama-3.2 1B (830 million weights), against exact dequantization in double
precision, with the arithmetic done in `f16` as in the kernel:

| Zero point                                          | RMS error (% of RMS weight) | Max. abs. error |
| --------------------------------------------------- | --------------------------: | --------------: |
| Rounded integer, $scale \times (q - round(m/d))$    |                       8.23% |         5.7e-02 |
| `f16` $m/d$ under `static_sazp` (first RFC version) |                      0.071% |         9.5e-04 |
| `f16` $m$ under `static_sbzp` (one fused operation) |                      0.069% |         7.3e-04 |

This is a numerical simulation of the dequantization (Python, `f16` rounding
after each operation of the kernel), not a measurement on the GPU. End-to-end
results with OpenVINO and llama.cpp need the OpenVINO GPU plugin to be adapted
to the new mode, and are not measured yet.

## Alternatives considered

### Continue rounding zero points (status quo)

No implementation changes, but the systematic decompression error causes the
accuracy loss shown in the Introduction, and PyTorch's form can't be expressed
exactly.

### Floating-point zero points under the existing mode

The first version of this proposal: allow `f16`, `bf16` and `f32` zero points
with $W_f = scale \times (W_{\mathrm{int}} - zp)$, opted in per implementation.
It needs no new API and gives the same accuracy, but the framework still
computes $-b / scale$, the zero point loses its integer meaning without the
semantics saying so, and implementations that ignore the zero-point data type
would misread it. A mode makes the semantics explicit and lets every
implementation reject it by default.

### The mode on the scales only

Set `static_sbzp` through `set_scales_v3()` and keep the zero-point API
unchanged. Fewer API changes, but the zero point's meaning would depend on
another attribute. Setting the mode on both, and requiring them to match, keeps
each attribute self-describing.

### A separate offset attribute

Add a new attribute for $b$ in $W_f = scale \times W_{\mathrm{int}} + b$. It
matches the formats directly, but it duplicates the zero-point machinery
(memory arguments, masks, groups, data types) for what is a zero point with a
different order of operations.

### Apply a correction term outside oneDNN

Keep an integer zero point and compensate for the residual with the per-group
sum of activations. Exact, but it adds work to every layer; an earlier
experiment measured a 5.2% decode slowdown.

### Dequantize the weights in the framework

Avoids the zero point entirely but stores and reads weights as 16-bit values,
about 3.5 times the size of Q4_K.

### Execute affected layers outside oneDNN

OpenVINO's current workaround, with the prefill, decode and memory costs shown
above.

### Native GGUF block formats

Out of scope. The framework already converts GGUF blocks into standard oneDNN
inputs (u4 weights with a scale and a zero point per group of 32); only the
order of operations is missing.

## Execution plan

1. Agree on the mode, the API and the scope in this RFC.
2. Land the change: API and checks, reference implementations, Intel GPU
   `jit:gemm`, benchdnn and gtests, documentation.
3. Later, on request: per-channel scales, integer activations (see the open
   questions), other weights arguments (gated MLP), optimized CPU
   implementations, more primitives.

## Open questions

1. **Scales and zero-point data types:** the scope requires the scales and
   zero points to have the activations' data type. Is there a need for other
   combinations (for example, `bf16` activations with `f16` scales and zero
   points)?
2. **Validation on other GPUs:** the prototype is tested on Intel Arc B390
   (Xe3) and B580 (Xe2) GPUs. We would like the oneDNN team's help to cover
   other generations (for example Xe-LPG, Xe-HPG and Xe-HPC).
3. **Integer activations:** frameworks can quantize activations to `s8` at run
   time (OpenVINO's dynamic quantization for compressed weights). With
   `static_sbzp` weights, the zero point term of each group becomes
   $zp \times \sum x$ without the weights scale, so it needs per-group sums of
   the activations, as precomputed reductions provide for `static_sazp`. This
   proposal rejects integer activations with `static_sbzp`; should it be
   planned as the first extension?
